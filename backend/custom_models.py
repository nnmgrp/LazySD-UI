"""
Custom Model Manager — handles user-imported GGUF/Safetensors models and Hugging Face Hub discovery.

Allows users to:
1. Drop arbitrary .gguf / .safetensors files into models/custom/ or specify absolute paths on PC.
2. Auto-detect model family, quantization, role, steps, and CFG.
3. Search Hugging Face Hub and inspect any repository's GGUF/Safetensors file tree to register
   downloadable model profiles in one click.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

import requests

from .settings import get_settings
from .registry.loader import get_registry, reload_registry


def get_custom_models_dir() -> Path:
    """Return path to models/custom/ directory, creating it if necessary."""
    settings = get_settings()
    custom_dir = settings.models_dir.parent / "custom"
    custom_dir.mkdir(parents=True, exist_ok=True)
    return custom_dir


def detect_file_metadata(filename: str) -> dict[str, Any]:
    """Guess family, role, quantization, steps, and CFG from a weight filename."""
    name_lower = filename.lower()

    # Role detection
    role = "diffusion_model"
    if any(k in name_lower for k in ("vae", "ae.safetensors", "ae.sft", "pig")):
        role = "vae"
    elif any(k in name_lower for k in ("t5", "clip", "text_encoder", "llm", "qwen3", "qwen2.5", "mmproj", "umt5")):
        role = "text_encoder"
    elif any(k in name_lower for k in ("sdxl", "sd_xl", "v1-5", "sd15", "juggernaut", "pony", "illustrious")):
        role = "checkpoint"

    # Family detection
    family = "flux"
    if "qwen" in name_lower:
        family = "qwen"
    elif any(k in name_lower for k in ("sdxl", "sd_xl", "pony", "illustrious")):
        family = "sdxl"
    elif any(k in name_lower for k in ("z-image", "z_image", "zimage")):
        family = "z-image"
    elif any(k in name_lower for k in ("klein", "flux")):
        family = "flux"

    # Quantization detection
    quant_match = re.search(
        r"(q8_0|q6_k|q5_k_m|q5_k_s|q5_0|q4_k_m|q4_k_s|q4_0|q3_k_m|q3_k_s|q2_k|iq4_nl|iq4_xs|bf16|fp16|f16|fp8|int8|int4)",
        name_lower,
    )
    quant = quant_match.group(1).upper() if quant_match else "CUSTOM"

    # Default steps & CFG
    is_distilled_fast = any(k in name_lower for k in ("schnell", "turbo", "lightning", "hyper")) and "base" not in name_lower
    if family == "sdxl":
        steps = 1 if is_distilled_fast else 20
        cfg = 1.0 if is_distilled_fast else 7.0
    elif family == "qwen":
        steps = 8 if is_distilled_fast else 20
        cfg = 1.0 if is_distilled_fast else 3.0
    elif family == "flux":
        if "klein" in name_lower and "base" not in name_lower:
            steps = 4
            cfg = 1.0
        elif is_distilled_fast:
            steps = 4
            cfg = 1.0
        else:
            steps = 20
            cfg = 3.5
    else:
        steps = 20
        cfg = 3.5

    return {
        "role": role,
        "family": family,
        "quant": quant,
        "steps": steps,
        "cfg": cfg,
    }


def list_custom_files() -> list[dict]:
    """Scan models/custom/ for compatible weights files with auto-detected metadata."""
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
                meta = detect_file_metadata(path.name)

                files.append({
                    "name": path.name,
                    "rel_path": rel_path,
                    "abs_path": str(path.resolve()),
                    "size_bytes": size_bytes,
                    "size_gb": size_gb,
                    "ext": ext,
                    "role": meta["role"],
                    "family": meta["family"],
                    "quant": meta["quant"],
                    "steps": meta["steps"],
                    "cfg": meta["cfg"],
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
    meta = detect_file_metadata(abs_file.name)

    raw_slug = slugify(name) or "model"
    model_id = f"custom_{raw_slug}"

    engine = data.get("engine", "sd_cpp")
    family = (data.get("family") or meta["family"] or "flux").lower()
    pipeline = data.get("pipeline", "text_to_image")
    steps = int(data.get("steps") or meta["steps"])
    cfg = float(data.get("cfg") if data.get("cfg") is not None else meta["cfg"])
    sampler = data.get("sampler", "euler")
    checkpoint_type = data.get("checkpoint_type") or ("checkpoint" if meta["role"] == "checkpoint" else "diffusion_model")

    engine_args: dict[str, Any] = {}
    if engine == "sd_cpp":
        if checkpoint_type == "checkpoint":
            engine_args["checkpoint"] = str(abs_file)
        else:
            engine_args["diffusion_model"] = str(abs_file)

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
        "name": name if "(Custom)" in name else f"{name} (Custom)",
        "family": family,
        "pipeline": pipeline,
        "engine": engine,
        "engine_version": "sd_cpp" if engine == "sd_cpp" else "0.0.17",
        "is_custom": True,
        "default_cfg": cfg,
        "description": data.get("description") or f"Custom {family.upper()} model ({abs_file.name}, {size_gb} GB).",
        "engine_args": engine_args,
        "transformers": [
            {
                "id": meta["quant"].lower() if meta["quant"] != "CUSTOM" else "custom",
                "path": str(abs_file),
                "tier": "universal",
                "backend": engine,
                "steps": steps,
                "size_gb": size_gb,
                "recommended": True,
                "note": f"Local file: {abs_file.name}",
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

    reload_registry()
    return manifest


def search_hf_models(query: str, limit: int = 12) -> list[dict[str, Any]]:
    """Search Hugging Face Hub for GGUF / image generation models."""
    q = (query or "").strip()
    if not q:
        q = "flux gguf"
    settings = get_settings()
    headers = {}
    if settings.hf_token:
        headers["Authorization"] = f"Bearer {settings.hf_token}"

    params = {
        "search": q,
        "limit": max(1, min(limit, 25)),
        "sort": "downloads",
        "direction": "-1",
    }
    resp = requests.get("https://huggingface.co/api/models", params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    items = resp.json()
    results = []
    for item in items:
        repo_id = item.get("id") or item.get("modelId")
        if not repo_id:
            continue
        tags = item.get("tags") or []
        results.append({
            "repo_id": repo_id,
            "downloads": item.get("downloads", 0),
            "likes": item.get("likes", 0),
            "pipeline_tag": item.get("pipeline_tag") or "",
            "tags": [t for t in tags if t in ("gguf", "diffusers", "flux", "sdxl", "text-to-image", "image-to-image")][:6],
            "last_modified": item.get("lastModified"),
        })
    return results


def inspect_hf_repo(repo_id: str) -> dict[str, Any]:
    """Inspect a Hugging Face repository and classify all weight files (.gguf / .safetensors)."""
    clean_repo = (repo_id or "").strip().replace("https://huggingface.co/", "").strip("/")
    if "/tree/" in clean_repo:
        clean_repo = clean_repo.split("/tree/")[0]
    if "/blob/" in clean_repo:
        clean_repo = clean_repo.split("/blob/")[0]
    if not clean_repo or "/" not in clean_repo:
        raise ValueError("Укажите корректный Hugging Face репозиторий в формате author/repo-name")

    settings = get_settings()
    headers = {}
    if settings.hf_token:
        headers["Authorization"] = f"Bearer {settings.hf_token}"

    url = f"https://huggingface.co/api/models/{clean_repo}?blobs=true"
    resp = requests.get(url, headers=headers, timeout=15)
    if resp.status_code == 404:
        raise ValueError(f"Репозиторий не найден на Hugging Face: {clean_repo}")
    resp.raise_for_status()
    info = resp.json()

    siblings = info.get("siblings") or []
    diffusion_files = []
    text_encoder_files = []
    vae_files = []

    repo_meta = detect_file_metadata(clean_repo)

    for sib in siblings:
        rfilename = sib.get("rfilename") or ""
        lower = rfilename.lower()
        if not lower.endswith((".gguf", ".safetensors")):
            continue
        if "mmproj" in lower:
            continue
        size_bytes = sib.get("size") or 0
        size_gb = round(size_bytes / (1024 ** 3), 2) if size_bytes else 0.0
        meta = detect_file_metadata(Path(rfilename).name)
        entry = {
            "path": rfilename,
            "name": Path(rfilename).name,
            "size_gb": size_gb,
            "quant": meta["quant"],
            "role": meta["role"],
            "steps": meta["steps"],
            "cfg": meta["cfg"],
        }
        if meta["role"] == "vae":
            vae_files.append(entry)
        elif meta["role"] == "text_encoder":
            text_encoder_files.append(entry)
        else:
            diffusion_files.append(entry)

    return {
        "repo_id": clean_repo,
        "suggested_name": clean_repo.split("/")[-1].replace("-", " "),
        "suggested_family": repo_meta["family"],
        "suggested_steps": repo_meta["steps"],
        "suggested_cfg": repo_meta["cfg"],
        "diffusion_files": diffusion_files,
        "text_encoder_files": text_encoder_files,
        "vae_files": vae_files,
    }


def register_hf_custom_model(data: dict[str, Any]) -> dict[str, Any]:
    """Register a downloadable model profile directly from a Hugging Face repository."""
    repo_id = (data.get("repo_id") or "").strip().replace("https://huggingface.co/", "").strip("/")
    if not repo_id or "/" not in repo_id:
        raise ValueError("Некорректный repo_id Hugging Face")

    name = (data.get("name") or repo_id.split("/")[-1]).strip()
    family = (data.get("family") or detect_file_metadata(repo_id)["family"] or "flux").lower()
    steps = int(data.get("steps") or 20)
    cfg = float(data.get("cfg") if data.get("cfg") is not None else 3.5)
    selected_files: list[dict[str, Any]] = data.get("variants") or []
    if not selected_files:
        raise ValueError("Выберите хотя бы один файл модели (.gguf / .safetensors) из репозитория")

    raw_slug = slugify(name) or slugify(repo_id.split("/")[-1])
    model_id = f"custom_hf_{raw_slug}"

    transformers = []
    for idx, vf in enumerate(selected_files):
        fpath = vf.get("path") or vf.get("name")
        if not fpath:
            continue
        meta = detect_file_metadata(fpath)
        vid = slugify(Path(fpath).stem) or f"var_{idx}"
        transformers.append({
            "id": vid,
            "path": fpath,
            "tier": "universal",
            "backend": meta["quant"].lower() if meta["quant"] != "CUSTOM" else "gguf",
            "steps": steps,
            "size_gb": float(vf.get("size_gb") or 4.0),
            "recommended": idx == 0,
            "repo": repo_id,
            "note": f"{Path(fpath).name} ({meta['quant']})",
        })

    # Default text encoders & VAE based on family if not overridden
    text_encoders = data.get("text_encoders") or []
    engine_args: dict[str, Any] = {
        "diffusion_model": transformers[0]["path"],
    }

    if family == "flux":
        engine_args["vae"] = "ae.safetensors"
        engine_args["vae_repo"] = "ffxvs/vae-flux"
        engine_args["vae_file"] = "ae.safetensors"
        if "klein" in repo_id.lower() or "klein" in name.lower():
            engine_args["vae"] = "flux2-vae.safetensors"
            engine_args["vae_repo"] = "Lightricks/LTX-2-comfy"
            engine_args["vae_file"] = "vae/flux2-vae.safetensors"
            if not text_encoders:
                is_9b = "9b" in repo_id.lower() or "9b" in name.lower()
                if is_9b:
                    text_encoders = [
                        {
                            "id": "qwen3_8b_q8_0",
                            "name": "Qwen3-8B Q8_0 (8.7 GB, Максимальная точность)",
                            "repo": "Qwen/Qwen3-8B-GGUF",
                            "file": "Qwen3-8B-Q8_0.gguf",
                            "size_gb": 8.7,
                            "precision": "q8_0",
                            "recommended": True,
                        },
                        {
                            "id": "qwen3_8b_q4_k_m",
                            "name": "Qwen3-8B Q4_K_M (5.0 GB, Быстрый)",
                            "repo": "unsloth/Qwen3-8B-GGUF",
                            "file": "Qwen3-8B-Q4_K_M.gguf",
                            "size_gb": 5.0,
                            "precision": "q4_k_m",
                            "recommended": False,
                        },
                    ]
                else:
                    text_encoders = [
                        {
                            "id": "qwen3vl_4b_q8_0",
                            "name": "Qwen3-VL-4B Q8_0 (4.3 GB, Рекомендуется)",
                            "repo": "Qwen/Qwen3-VL-4B-Instruct-GGUF",
                            "file": "Qwen3VL-4B-Instruct-Q8_0.gguf",
                            "size_gb": 4.3,
                            "precision": "q8_0",
                            "recommended": True,
                        },
                        {
                            "id": "qwen3_4b_q4_k_m",
                            "name": "Qwen3-4B Q4_K_M (2.5 GB, Легкий)",
                            "repo": "unsloth/Qwen3-4B-GGUF",
                            "file": "Qwen3-4B-Q4_K_M.gguf",
                            "size_gb": 2.5,
                            "precision": "q4_k_m",
                            "recommended": False,
                        },
                    ]
                engine_args["llm"] = text_encoders[0]["file"]
                engine_args["llm_repo"] = text_encoders[0]["repo"]
                engine_args["llm_file"] = text_encoders[0]["file"]
    elif family == "qwen":
        engine_args["vae"] = "texture_fix_vae_for_qwen_image_2.1_bf16.safetensors"
        engine_args["vae_repo"] = "madebyollin/texture-fix-vae-for-qwen-image-2.1"
        engine_args["vae_file"] = "texture_fix_vae_for_qwen_image_2.1_bf16.safetensors"
        if not text_encoders:
            text_encoders = [
                {
                    "id": "qwen3vl_8b_q4_k_xl",
                    "name": "Qwen3-VL-8B Q4_K_XL (5.1 GB, Рекомендуется)",
                    "repo": "unsloth/Qwen3-VL-8B-Instruct-GGUF",
                    "file": "Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf",
                    "size_gb": 5.1,
                    "precision": "q4_k_xl",
                    "recommended": True,
                },
                {
                    "id": "qwen3vl_8b_q8_0",
                    "name": "Qwen3-VL-8B Q8_0 (8.7 GB, Высокая точность)",
                    "repo": "unsloth/Qwen3-VL-8B-Instruct-GGUF",
                    "file": "Qwen3-VL-8B-Instruct-Q8_0.gguf",
                    "size_gb": 8.7,
                    "precision": "q8_0",
                    "recommended": False,
                },
            ]
            engine_args["llm"] = text_encoders[0]["file"]
            engine_args["llm_repo"] = text_encoders[0]["repo"]
            engine_args["llm_file"] = text_encoders[0]["file"]

    manifest = {
        "id": model_id,
        "name": f"{name} (HF)",
        "family": family,
        "pipeline": "text_to_image",
        "engine": "sd_cpp",
        "engine_version": "sd_cpp",
        "is_custom": True,
        "default_cfg": cfg,
        "description": data.get("description") or f"Импортировано из Hugging Face: {repo_id} ({len(transformers)} вар.)",
        "sources": {
            "primary": {
                "type": "huggingface",
                "repo": repo_id,
            }
        },
        "base_models": {},
        "engine_args": engine_args,
        "transformers": transformers,
        "text_encoders": text_encoders,
        "requirements": {
            "vram_min_gb": 8,
            "vram_recommended_gb": 12,
            "gpu_min_sm": 75,
            "default_vram_mode": "auto",
        },
        "scheduler_presets": {
            "default": {
                "sampler": "euler",
                "steps": steps,
                "cfg": cfg,
            }
        },
    }

    registry_dir = Path(__file__).parent / "registry" / "models"
    out_json = registry_dir / f"{model_id}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

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
