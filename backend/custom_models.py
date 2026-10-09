"""
Custom Model Manager — handles user-imported GGUF and Safetensors models.

Allows users to drop arbitrary .gguf / .safetensors files into models/custom/
or specify absolute paths on their system, creating managed model profiles
compatible with SD.CPP and QuantFunc runtimes.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from .settings import get_settings
from .registry.loader import get_registry, reload_registry, ModelSpec


def get_custom_models_dir() -> Path:
    """Return path to models/custom/ directory, creating it if necessary."""
    settings = get_settings()
    custom_dir = settings.models_dir.parent / "custom"
    custom_dir.mkdir(parents=True, exist_ok=True)
    return custom_dir


def list_custom_files() -> list[dict]:
    """Scan models/custom/ for compatible weights files."""
    custom_dir = get_custom_models_dir()
    supported_exts = {".gguf", ".safetensors", ".ckpt", ".pt", ".bin"}
    files = []

    for path in sorted(custom_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in supported_exts:
            try:
                st = path.stat()
                size_bytes = st.st_size
                size_gb = round(size_bytes / (1024 ** 3), 2)
                rel_path = str(path.relative_to(custom_dir)).replace("\\", "/")
                ext = path.suffix.lower().lstrip(".")
                
                # Guess file role
                name_lower = path.name.lower()
                role = "unknown"
                if any(k in name_lower for k in ("vae", "ae.")):
                    role = "vae"
                elif any(k in name_lower for k in ("t5", "clip", "text_encoder", "llm", "qwen3vl")):
                    role = "text_encoder"
                elif any(k in name_lower for k in ("dit", "transformer", "diffusion", "schnell", "dev", "qwen-image")):
                    role = "diffusion_model"
                elif any(k in name_lower for k in ("turbo", "checkpoint", "sdxl", "sd15", "v1-5")):
                    role = "checkpoint"

                files.append({
                    "name": path.name,
                    "rel_path": rel_path,
                    "abs_path": str(path.resolve()),
                    "size_bytes": size_bytes,
                    "size_gb": size_gb,
                    "ext": ext,
                    "role": role,
                })
            except Exception:
                continue

    return files


def slugify(text: str) -> str:
    """Generate a clean slug for model ID."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    return text.strip("_")


def register_custom_model(data: dict) -> dict:
    """Register a custom model configuration as a ModelSpec JSON manifest."""
    name = data.get("name", "").strip()
    if not name:
        raise ValueError("Model name is required")

    model_file = data.get("model_file", "").strip()
    if not model_file:
        raise ValueError("Model file path is required")

    custom_dir = get_custom_models_dir()
    file_path = Path(model_file)
    if not file_path.is_absolute():
        resolved_file = custom_dir / model_file
        if not resolved_file.exists():
            raise ValueError(f"File not found in models/custom/: {model_file}")
        abs_file = resolved_file
    else:
        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")
        abs_file = file_path

    st = abs_file.stat()
    size_gb = round(st.st_size / (1024 ** 3), 2)

    # Slug ID
    raw_slug = slugify(name)
    model_id = f"custom_{raw_slug}"

    engine = data.get("engine", "sd_cpp")
    family = data.get("family", "flux").lower()
    pipeline = data.get("pipeline", "text_to_image")
    steps = int(data.get("steps", 20 if family != "sdxl" else 1))
    cfg = float(data.get("cfg", 3.5 if family != "sdxl" else 1.0))
    sampler = data.get("sampler", "euler")
    checkpoint_type = data.get("checkpoint_type", "diffusion_model")  # "checkpoint" or "diffusion_model"

    engine_args: dict = {}
    if engine == "sd_cpp":
        if checkpoint_type == "checkpoint":
            engine_args["checkpoint"] = str(abs_file)
        else:
            engine_args["diffusion_model"] = str(abs_file)

        # Optional companion files
        llm_path = data.get("llm_file")
        if llm_path:
            p = Path(llm_path)
            engine_args["llm"] = str(p if p.is_absolute() else (custom_dir / llm_path).resolve())

        t5_path = data.get("t5_file")
        if t5_path:
            p = Path(t5_path)
            engine_args["t5xxl"] = str(p if p.is_absolute() else (custom_dir / t5_path).resolve())

        clip_path = data.get("clip_file")
        if clip_path:
            p = Path(clip_path)
            engine_args["clip_l"] = str(p if p.is_absolute() else (custom_dir / clip_path).resolve())

        vae_path = data.get("vae_file")
        if vae_path:
            p = Path(vae_path)
            engine_args["vae"] = str(p if p.is_absolute() else (custom_dir / vae_path).resolve())

        if data.get("vae_format"):
            engine_args["vae_format"] = data["vae_format"]

        if data.get("flow_shift"):
            engine_args["flow_shift"] = float(data["flow_shift"])

    manifest = {
        "id": model_id,
        "name": f"{name} (Custom)",
        "family": family,
        "pipeline": pipeline,
        "engine": engine,
        "engine_version": "sd_cpp" if engine == "sd_cpp" else "0.0.17",
        "is_custom": True,
        "description": data.get("description", f"Custom {family.upper()} model imported by user ({abs_file.name})."),
        "engine_args": engine_args,
        "transformers": [
            {
                "id": "custom",
                "path": str(abs_file),
                "tier": "universal",
                "backend": engine,
                "steps": steps,
                "size_gb": size_gb,
                "recommended": True,
                "note": f"Custom imported file: {abs_file.name}",
            }
        ],
        "requirements": {
            "vram_min_gb": int(data.get("vram_min_gb", 8)),
            "vram_recommended_gb": 12,
            "gpu_min_sm": 75,
            "default_vram_mode": "auto",
        },
        "scheduler_presets": {
            "default": {
                "sampler": sampler,
                "steps": steps,
                "cfg": cfg,
            }
        },
    }

    # If text encoder specified, populate text_encoders list
    if engine_args.get("llm"):
        manifest["text_encoders"] = [
            {
                "id": "custom_llm",
                "name": Path(engine_args["llm"]).name,
                "file": engine_args["llm"],
                "precision": "custom",
                "recommended": True,
            }
        ]

    registry_dir = Path(__file__).parent / "registry" / "models"
    out_json = registry_dir / f"{model_id}.json"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Reload registry so it is immediately accessible
    reload_registry()

    return manifest


def delete_custom_model(model_id: str) -> bool:
    """Delete a custom model manifest."""
    if not model_id.startswith("custom_"):
        raise ValueError("Cannot delete non-custom system models")

    registry_dir = Path(__file__).parent / "registry" / "models"
    target = registry_dir / f"{model_id}.json"
    if target.exists():
        target.unlink()
        reload_registry()
        return True
    return False
