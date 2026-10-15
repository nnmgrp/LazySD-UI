"""
QuantFunc WebUI — main entry point.

Architecture: gradio.Server + FastAPI + custom index.html (NOT gr.Blocks).

This follows the pattern from the HF Space prithivMLmods/Flux.2-Klein-KV-Edit-Consistency-Ultra-Fast,
which serves a hand-written index.html at / and exposes JSON API endpoints.

The UI is fully custom HTML/CSS/JS — no Gradio components are rendered.
Backend communication happens via REST endpoints defined below.

Run:
    python app.py

Then open: http://127.0.0.1:7860
"""
from __future__ import annotations

import asyncio
import json
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
try:
    import huggingface_hub.constants as _hfc
    _hfc.HF_HUB_DISABLE_XET = True
    _hfc.HF_HUB_ENABLE_HF_TRANSFER = False
except Exception:
    pass

import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# ---------- Backend imports ----------
from backend.cache import CacheManager
from backend.engine import (
    check_engine_status,
    download_engine,
    engine_config_json_path,
    ensure_config_json,
    is_engine_installed,
    list_installed_engine_versions,
    read_engine_api_key,
    verify_engine,
)
from backend.sd_cpp import is_sd_cpp_installed, get_sd_cpp_version
from backend.sdnq_engine import is_sdnq_installed, get_sdnq_version, get_sdnq_path
from backend.gpu import detect_gpu, GPUInfo
from backend.registry import get_registry
from backend.settings import get_settings
from backend.worker import (
    GenerateRequest,
    JobStatus,
    ProgressEvent,
    get_worker,
)

# ---------- Web framework ----------
# NOTE: We use FastAPI + uvicorn directly (no gradio.Blocks).
# The HF Space reference (prithivMLmods/Flux.2-Klein-KV-Edit) uses the same
# pattern: gradio.Server is replaced by a hand-rolled FastAPI app serving
# custom index.html + JSON API endpoints. This avoids gradio's heavy
# dependencies (huggingface_hub version conflicts, etc.).
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


# ============================================================
# State — held in module scope (single-user, global)
# ============================================================
class AppState:
    """Global app state — single-user, single-process."""

    def __init__(self):
        self.gpu_info: Optional[GPUInfo] = None
        self.jobs: dict[str, dict] = {}   # job_id → {status, progress, ...}
        self._job_lock = threading.Lock()

    def register_job(self, job_id: str) -> None:
        with self._job_lock:
            self.jobs[job_id] = {
                "status": JobStatus.QUEUED,
                "progress": 0.0,
                "message": "Queued",
                "started_at": time.time(),
                "image_path": None,
                "error": None,
            }

    def update_job(self, job_id: str, event: ProgressEvent) -> None:
        with self._job_lock:
            if job_id not in self.jobs:
                return
            self.jobs[job_id].update({
                "status": event.status,
                "progress": event.progress,
                "message": event.message,
                "step": event.step,
                "total_steps": event.total_steps,
                "image_path": event.image_path,
                "error": event.error,
            })

    def get_job(self, job_id: str) -> Optional[dict]:
        with self._job_lock:
            return self.jobs.get(job_id)


STATE = AppState()


# ============================================================
# Startup — detect GPU, check engine, load registry
# ============================================================
def init_state() -> None:
    """Run at startup: detect GPU, check engine, load model registry."""
    print("=" * 60)
    print("QuantFunc WebUI — starting up")
    print("=" * 60)

    # Load settings
    settings = get_settings()
    print(f"[startup] Settings loaded from: {settings.settings_file_path}")
    print(f"[startup] Models dir: {settings.models_dir}")
    print(f"[startup] Engine dir: {settings.engine_dir}")
    print(f"[startup] Language: {settings.lang}")

    # Detect GPU
    print("[startup] Detecting GPU...")
    STATE.gpu_info = detect_gpu()
    if STATE.gpu_info:
        g = STATE.gpu_info
        print(f"[startup] GPU: {g.name}")
        print(f"[startup]   SM: {g.sm} -> tier={g.tier}, klein_tier={g.klein_tier}")
        print(f"[startup]   VRAM: {g.vram_mb} MB ({g.vram_gb:.1f} GB)")
        print(f"[startup]   Driver: {g.driver_version}")
        # Persist to settings
        settings.update(
            gpu_name=g.name,
            gpu_sm=g.sm,
            gpu_vram_mb=g.vram_mb,
            gpu_driver_version=g.driver_version,
        )
    else:
        print("[startup] WARNING: No NVIDIA GPU detected. WebUI will start but generation will fail.")

    # Check engine
    print("[startup] Checking engine...")
    engine_status = check_engine_status()
    print(f"[startup] Engine status: {engine_status.status} — {engine_status.message}")

    # Load model registry
    print("[startup] Loading model registry...")
    registry = get_registry()
    print(f"[startup] Loaded {len(registry.models)} models: {list(registry.models.keys())}")

    # Ensure engine auth config exists (bin/<os>/config.json with the bundled
    # free-trial test key). Without it key-protected models fail at
    # checkObfuscation. Best-effort — never blocks startup.
    try:
        cfg_path, cfg_source = ensure_config_json()
        print(f"[startup] Engine auth config: {cfg_path} (source: {cfg_source})")
    except Exception as e:
        print(f"[startup] WARNING: could not ensure engine config.json: {e}")

    print("=" * 60)
    print(f"[startup] Ready. Open http://127.0.0.1:7860")
    print("=" * 60)


# ============================================================
# Build the main HTML page & Engine Status
# ============================================================
def _get_engines_info() -> dict:
    from backend.engine import is_engine_installed, get_engine_paths
    from backend.sd_cpp import is_sd_cpp_installed, get_sd_cpp_cli_path, get_sd_cpp_version

    _, cli17 = get_engine_paths("0.0.17")
    _, cli12 = get_engine_paths("0.0.12")
    cli_sd = get_sd_cpp_cli_path()

    return {
        "quantfunc_0_17": {
            "id": "quantfunc_0_17",
            "engine": "quantfunc",
            "version": "0.0.17",
            "name": "QuantFunc v0.0.17",
            "title": "QuantFunc v0.0.17 (Primary SVDQuant)",
            "installed": is_engine_installed("0.0.17"),
            "path": str(cli17),
            "description": "Primary engine for Qwen-Image-2.1 and standard SVDQuant models.",
        },
        "quantfunc_0_12": {
            "id": "quantfunc_0_12",
            "engine": "quantfunc",
            "version": "0.0.12",
            "name": "QuantFunc v0.0.12",
            "title": "QuantFunc v0.0.12 (Legacy NVFP4)",
            "installed": is_engine_installed("0.0.12"),
            "path": str(cli12),
            "description": "Legacy engine with native NVFP4 support (Flux.2 Klein FP4, Z-Image).",
        },
        "sd_cpp": {
            "id": "sd_cpp",
            "engine": "sd_cpp",
            "version": get_sd_cpp_version(),
            "name": "stable-diffusion.cpp",
            "title": "stable-diffusion.cpp (GGUF Runtime)",
            "installed": is_sd_cpp_installed(),
            "path": str(cli_sd),
            "description": "Standalone C++/CUDA runner for GGUF models (Flux.1 Schnell, SDXL, Unsloth Q4).",
        },
        "sdnq": {
            "id": "sdnq",
            "engine": "sdnq",
            "version": get_sdnq_version(),
            "name": "SDNQ (Diffusers)",
            "title": "SDNQ (SD.Next Quantization + Diffusers)",
            "installed": is_sdnq_installed(),
            "path": get_sdnq_path(),
            "description": "PyTorch + Triton INT8/INT4/FP8 quantization with Hadamard rotation & SVD (Qwen-Image-2.1 Turbo INT8).",
        },
    }


def _api_auth_status() -> dict:
    """Engine auth status for the frontend (config.json + user key)."""
    cfg_present = engine_config_json_path().exists()
    has_cfg_key = read_engine_api_key() is not None
    return {
        "config_present": cfg_present,
        "config_has_key": has_cfg_key,
        "user_key_set": bool(get_settings().quantfunc_api_key and get_settings().quantfunc_api_key.strip()),
    }


def build_index_html() -> str:
    """Read frontend/index.html and inject settings + GPU info as JSON."""
    html_path = PROJECT_ROOT / "frontend" / "index.html"
    if not html_path.exists():
        return "<html><body><h1>frontend/index.html not found</h1></body></html>"

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Inject initial config (so JS doesn't need to fetch on load)
    settings = get_settings()
    gpu_dict = None
    if STATE.gpu_info:
        gpu_dict = {
            "name": STATE.gpu_info.name,
            "sm": STATE.gpu_info.sm,
            "vram_mb": STATE.gpu_info.vram_mb,
            "driver_version": STATE.gpu_info.driver_version,
            "tier": STATE.gpu_info.tier,
            "klein_tier": STATE.gpu_info.klein_tier,
        }

    engine_status = check_engine_status()
    installed_engines = list_installed_engine_versions()
    if is_sd_cpp_installed():
        installed_engines.append("sd_cpp")
    if is_sdnq_installed():
        installed_engines.append("sdnq")

    config_json = json.dumps({
        "settings": {
            "lang": settings.lang,
            "vram_mode": settings.vram_mode,
            "auto_optimize": settings.auto_optimize,
            "models_dir": str(settings.models_dir),
            "output_dir": str(settings.output_dir),
            "engine_version": settings.engine_version,
            "engine_platform": settings.engine_platform,
            "cuda_version": settings.cuda_version,
            "qwen_vae_type": getattr(settings, "qwen_vae_type", "texture_fix"),
        },
        "gpu": gpu_dict,
        "engine_status": {
            "status": engine_status.status,
            "message": engine_status.message,
        },
        "installed_engines": installed_engines,
        "engines": _get_engines_info(),
        "sd_cpp_installed": is_sd_cpp_installed(),
        "sd_cpp_version": get_sd_cpp_version(),
        "sdnq_installed": is_sdnq_installed(),
        "sdnq_version": get_sdnq_version(),
        "api_auth": _api_auth_status(),
    }, default=str)

    # Inject before </head>
    html = html.replace(
        "</head>",
        f'<script>window.QF_INITIAL_CONFIG = {config_json};</script></head>'
    )
    return html


# ============================================================
# FastAPI app — built on top of Gradio
# ============================================================
def _get_token_for_source(source_type: str) -> Optional[str]:
    """Get auth token for a source type from settings."""
    settings = get_settings()
    if source_type == "huggingface":
        return settings.huggingface_token
    elif source_type == "modelscope":
        return settings.modelscope_token
    return None


def create_app() -> FastAPI:
    """Create the FastAPI app with custom routes + Gradio mount."""

    # Use Gradio's FastAPI underpinning
    app = FastAPI(title="QuantFunc WebUI", version="0.1.0")

    # ---------- Static file mounts ----------
    frontend_dir = PROJECT_ROOT / "frontend"
    app.mount("/css", StaticFiles(directory=str(frontend_dir / "css")), name="css")
    app.mount("/js", StaticFiles(directory=str(frontend_dir / "js")), name="js")
    app.mount("/locale", StaticFiles(directory=str(frontend_dir / "locale")), name="locale")

    # Output dir mount (for serving generated images)
    settings = get_settings()
    settings.output_dir.mkdir(parents=True, exist_ok=True)  # Ensure it exists before mount
    app.mount("/outputs", StaticFiles(directory=str(settings.output_dir)), name="outputs")

    def _is_te_installed(te: dict, repo_dir: Optional[Path], cache: CacheManager) -> bool:
        te_f = te.get("file", "")
        if not te_f:
            return False
        if repo_dir:
            if (repo_dir / te_f).exists() or (repo_dir / "text_encoders" / te_f).exists():
                return True
            if te.get("id") == "bf16" and (repo_dir / "qwen3vl_8b_bf16.safetensors").exists():
                return True
        if isinstance(te.get("source"), dict) and te["source"].get("repo_id"):
            sec_dir = cache.repo_dir_for(te["source"]["repo_id"])
            if (sec_dir / te_f).exists() or (sec_dir / "text_encoders" / te_f).exists():
                return True
        return False

    def _model_is_installed(model, cache: CacheManager, gpu_tier: str) -> bool:
        try:
            if getattr(model, "is_custom", False):
                from backend.custom_models import get_custom_models_dir
                custom_dir = get_custom_models_dir()
                ckpt_rel = model.engine_args.get("checkpoint") if getattr(model, "engine_args", None) else None
                if ckpt_rel:
                    p = Path(ckpt_rel)
                    return p.exists() or (custom_dir / ckpt_rel).exists()
                diff_rel = model.engine_args.get("diffusion_model") if getattr(model, "engine_args", None) else None
                if diff_rel:
                    p = Path(diff_rel)
                    return p.exists() or (custom_dir / diff_rel).exists()
                for t in model.transformers:
                    p = Path(t.path)
                    if p.exists() or (custom_dir / t.path).exists():
                        return True
                return False

            if getattr(model, "engine", "quantfunc") == "sd_cpp":
                repo_id = model.sources.primary.repo_id if model.sources else ""
                repo_dir = cache.repo_dir_for(repo_id) if repo_id else get_settings().models_dir
                ckpt_rel = model.engine_args.get("checkpoint")
                if ckpt_rel and (repo_dir / ckpt_rel).exists():
                    return True
                diff_rel = model.engine_args.get("diffusion_model")
                diff_ok = False
                if diff_rel and (repo_dir / diff_rel).exists():
                    diff_ok = True
                else:
                    for t in model.transformers:
                        t_repo = getattr(t, "repo_id", None)
                        if (repo_dir / t.path).exists() or (t_repo and (cache.repo_dir_for(t_repo) / t.path).exists()):
                            diff_ok = True
                            break
                if not diff_ok:
                    return False
                if getattr(model, "text_encoders", None):
                    if not any(_is_te_installed(te, repo_dir, cache) for te in model.text_encoders):
                        return False
                elif model.engine_args.get("llm"):
                    llm_rel = model.engine_args["llm"]
                    if not ((repo_dir / llm_rel).exists() or (repo_dir / "text_encoders" / llm_rel).exists()):
                        return False
                return True

            if getattr(model, "engine", "quantfunc") == "sdnq":
                repo_id = model.sources.primary.repo_id if model.sources else ""
                repo_dir = cache.repo_dir_for(repo_id) if repo_id else get_settings().models_dir
                if not (repo_dir / "model_index.json").exists():
                    return False
                if not (repo_dir / "vae" / "diffusion_pytorch_model.safetensors").exists():
                    return False
                if getattr(model, "text_encoders", None):
                    if not any(_is_te_installed(te, repo_dir, cache) for te in model.text_encoders):
                        return False
                for t in model.transformers:
                    if t.recommended and (repo_dir / t.path).exists():
                        return True
                return False

            # QuantFunc engine models
            if not model.sources:
                return False
            base_path = model.get_base_model_path(gpu_tier) if gpu_tier != "unknown" else None
            if not base_path and model.base_models:
                base_path = list(model.base_models.values())[0]
            if base_path:
                if base_path in ("", "."):
                    local_base = cache.repo_dir_for(model.sources.primary.repo_id)
                else:
                    local_base = cache.repo_dir_for(model.sources.primary.repo_id) / base_path
                if not (local_base / "model_index.json").exists():
                    return False
            if getattr(model, "text_encoders", None):
                repo_dir = cache.repo_dir_for(model.sources.primary.repo_id)
                if not any(_is_te_installed(te, repo_dir, cache) for te in model.text_encoders):
                    return False
            primary_dir = cache.repo_dir_for(model.sources.primary.repo_id)
            for t in model.transformers:
                if primary_dir and (primary_dir / t.path).exists():
                    return True
                t_repo = getattr(t, "repo_id", None)
                if t_repo and (cache.repo_dir_for(t_repo) / t.path).exists():
                    return True
            return False
        except Exception:
            return False

    # ---------- API: Config ----------
    @app.get("/api/config")
    async def api_config():
        """Initial config for frontend."""
        settings = get_settings()
        registry = get_registry()
        cache = CacheManager()
        gpu_tier = STATE.gpu_info.tier if STATE.gpu_info else "unknown"

        # Convert models to JSON-serializable list
        models_list = []
        for m in registry.list_all():
            repo_id = m.sources.primary.repo_id if m.sources else ""
            repo_dir = cache.repo_dir_for(repo_id) if repo_id else None
            default_cfg = m.get_default_cfg()
            models_list.append({
                "id": m.id,
                "name": m.name,
                "family": m.family,
                "pipeline": m.pipeline,
                "engine": getattr(m, "engine", "quantfunc"),
                "engine_version": m.engine_version,
                "is_custom": getattr(m, "is_custom", False),
                "installed": _model_is_installed(m, cache, gpu_tier),
                "description": m.description,
                "key_protected": bool(m.key_protected),
                "default_cfg": default_cfg,
                "requirements": {
                    "vram_min_gb": m.requirements.vram_min_gb,
                    "vram_recommended_gb": m.requirements.vram_recommended_gb,
                    "gpu_min_sm": m.requirements.gpu_min_sm,
                    "default_vram_mode": m.requirements.default_vram_mode,
                },
                "text_encoders": [
                    {
                        "id": te.get("id"),
                        "name": te.get("name"),
                        "file": te.get("file"),
                        "size_gb": te.get("size_gb"),
                        "precision": te.get("precision"),
                        "recommended": te.get("recommended", False),
                        "installed": _is_te_installed(te, repo_dir, cache),
                    }
                    for te in getattr(m, "text_encoders", [])
                ],
                "transformers": [
                    {
                        "id": t.id,
                        "backend": t.backend,
                        "steps": t.steps,
                        "size_gb": t.size_gb,
                        "recommended": t.recommended,
                        "note": t.note,
                    }
                    for t in m.transformers
                ],
            })

        installed_engines = list_installed_engine_versions()
        if is_sd_cpp_installed():
            installed_engines.append("sd_cpp")
        if is_sdnq_installed():
            installed_engines.append("sdnq")

        return {
            "settings": {
                "lang": settings.lang,
                "vram_mode": settings.vram_mode,
                "auto_optimize": settings.auto_optimize,
                "models_dir": str(settings.models_dir),
                "output_dir": str(settings.output_dir),
                "engine_version": settings.engine_version,
                "engine_platform": settings.engine_platform,
                "cuda_version": settings.cuda_version,
                "qwen_vae_type": getattr(settings, "qwen_vae_type", "texture_fix"),
            },
            "gpu": {
                "name": STATE.gpu_info.name,
                "sm": STATE.gpu_info.sm,
                "vram_mb": STATE.gpu_info.vram_mb,
                "driver_version": STATE.gpu_info.driver_version,
                "tier": STATE.gpu_info.tier,
                "klein_tier": STATE.gpu_info.klein_tier,
            } if STATE.gpu_info else None,
            "engine_status": {
                "status": check_engine_status().status,
                "message": check_engine_status().message,
            },
            "installed_engines": installed_engines,
            "engines": _get_engines_info(),
            "sd_cpp_installed": is_sd_cpp_installed(),
            "sd_cpp_version": get_sd_cpp_version(),
            "sdnq_installed": is_sdnq_installed(),
            "sdnq_version": get_sdnq_version(),
            "api_auth": _api_auth_status(),
            "models": models_list,
        }

    # ---------- API: Models ----------
    @app.get("/api/models")
    async def api_models():
        registry = get_registry()
        cache = CacheManager()
        gpu_tier = STATE.gpu_info.tier if STATE.gpu_info else "unknown"
        models_out = []
        for m in registry.list_all():
            repo_id = m.sources.primary.repo_id if m.sources else ""
            repo_dir = cache.repo_dir_for(repo_id) if repo_id else None
            default_cfg = m.get_default_cfg()
            models_out.append({
                "id": m.id,
                "name": m.name,
                "family": m.family,
                "pipeline": m.pipeline,
                "engine": getattr(m, "engine", "quantfunc"),
                "engine_version": m.engine_version,
                "is_custom": getattr(m, "is_custom", False),
                "installed": _model_is_installed(m, cache, gpu_tier),
                "description": m.description,
                "key_protected": bool(m.key_protected),
                "default_cfg": default_cfg,
                "requirements": {
                    "vram_min_gb": m.requirements.vram_min_gb,
                    "vram_recommended_gb": m.requirements.vram_recommended_gb,
                    "gpu_min_sm": m.requirements.gpu_min_sm,
                },
                "text_encoders": [
                    {
                        "id": te.get("id"),
                        "name": te.get("name"),
                        "file": te.get("file"),
                        "size_gb": te.get("size_gb"),
                        "precision": te.get("precision"),
                        "recommended": te.get("recommended", False),
                        "installed": _is_te_installed(te, repo_dir, cache),
                    }
                    for te in getattr(m, "text_encoders", [])
                ],
                "transformers": [
                    {
                        "id": t.id,
                        "backend": t.backend,
                        "steps": t.steps,
                        "size_gb": t.size_gb,
                        "recommended": t.recommended,
                        "note": t.note,
                    }
                    for t in m.transformers
                ],
            })
        return models_out

    @app.get("/api/models/{model_id}/state")
    async def api_model_state(model_id: str):
        """Check cache state for a model's required files.

        If GPU is not detected, returns all transformers (no tier filtering)
        so the UI can still show the model. The user just won't be able to
        generate until GPU is available.
        """
        registry = get_registry()
        try:
            model = registry.get(model_id)
        except KeyError:
            raise HTTPException(404, f"Model not found: {model_id}")

        # Determine GPU tier (or fall back to showing all transformers)
        if STATE.gpu_info:
            gpu_sm = STATE.gpu_info.sm
            gpu_tier = STATE.gpu_info.tier
            transformers_for_gpu = model.get_transformers_for_tier(gpu_sm)
        else:
            # No GPU — show all transformers (UI-only, generation will fail later)
            gpu_sm = 0
            gpu_tier = "unknown"
            transformers_for_gpu = model.transformers

        if getattr(model, "is_custom", False):
            from backend.custom_models import get_custom_models_dir
            custom_dir = get_custom_models_dir()
            result = []
            for t in model.transformers:
                p = Path(t.path)
                f_exists = p.exists() or (custom_dir / t.path).exists()
                result.append({
                    "id": t.id,
                    "backend": t.backend,
                    "steps": t.steps,
                    "prequant": t.prequant,
                    "size_gb": t.size_gb,
                    "tier": t.tier,
                    "klein_tier": t.klein_tier,
                    "recommended": t.recommended,
                    "recommended_for_gpu": t.recommended_for_gpu,
                    "state": "cached" if f_exists else "missing",
                    "state_label": "Local ✓" if f_exists else "Missing",
                })
            te_result = []
            for te in getattr(model, "text_encoders", []):
                p = Path(te.get("file", ""))
                te_exists = p.exists() or (custom_dir / te.get("file", "")).exists()
                te_result.append({
                    "id": te.get("id"),
                    "name": te.get("name"),
                    "file": te.get("file"),
                    "size_gb": te.get("size_gb"),
                    "precision": te.get("precision"),
                    "recommended": te.get("recommended", False),
                    "installed": te_exists,
                })
            return {
                "model_id": model_id,
                "model_name": model.name,
                "gpu_tier": gpu_tier,
                "gpu_sm": gpu_sm,
                "base_model_installed": True,
                "text_encoders": te_result,
                "transformers": result,
            }

        # Check cache state for each
        cache = CacheManager()
        source_spec = model.sources.primary
        repo_id = source_spec.repo_id
        local_repo_dir = cache.repo_dir_for(repo_id)

        result = []
        for t in transformers_for_gpu:
            t_repo = getattr(t, "repo_id", None) or repo_id
            use_repo = t_repo if (cache.repo_dir_for(t_repo) / t.path).exists() else repo_id
            entry = cache.check_file(
                repo_id=use_repo,
                file_path=t.path,
                expected_size=t.size_bytes or 0,
                expected_sha256=t.sha256,
                compute_checksum=False,  # Fast check on first load
            )
            result.append({
                "id": t.id,
                "backend": t.backend,
                "steps": t.steps,
                "prequant": t.prequant,
                "size_gb": t.size_gb,
                "tier": t.tier,
                "klein_tier": t.klein_tier,
                "recommended": t.recommended,
                "recommended_for_gpu": t.recommended_for_gpu,
                "state": entry.state.value,
                "state_label": entry.state_label,
            })

        # Also check base model dir
        base_path = model.get_base_model_path(gpu_tier) if gpu_tier != "unknown" else None
        if not base_path and model.base_models:
            # Fall back to first available base model
            base_path = list(model.base_models.values())[0]
        base_installed = False
        if getattr(model, "engine", "quantfunc") == "sdnq":
            base_installed = (
                (local_repo_dir / "model_index.json").exists()
                and (local_repo_dir / "vae" / "diffusion_pytorch_model.safetensors").exists()
            )
        elif base_path:
            local_base = local_repo_dir / base_path
            base_installed = (local_base / "model_index.json").exists()
        elif getattr(model, "engine", "quantfunc") == "sd_cpp" or not model.base_models:
            base_installed = True

        # Check text encoders state if model specifies them
        te_result = []
        known_te_files = set()
        for te in getattr(model, "text_encoders", []):
            te_f = te.get("file")
            known_te_files.add(te_f)
            known_te_files.add(Path(te_f).name)
            installed = _is_te_installed(te, local_repo_dir, cache)
            te_result.append({
                "id": te.get("id"),
                "name": te.get("name"),
                "file": te_f,
                "size_gb": te.get("size_gb"),
                "precision": te.get("precision"),
                "recommended": te.get("recommended", False),
                "installed": installed,
            })

        # Dynamically discover any other custom/local encoders in the directory (skip for sdnq pipeline folders)
        if getattr(model, "engine", "") != "sdnq" and (getattr(model, "text_encoders", None) or model.family == "qwen" or getattr(model, "engine", "") == "sd_cpp"):
            known_files = set(known_te_files)
            known_files.add("model_index.json")
            for tr in getattr(model, "transformers", []):
                known_files.add(tr.path)
                known_files.add(Path(tr.path).name)

            scan_dirs = [local_repo_dir, local_repo_dir / "text_encoders"]
            seen_extra = set()
            for sdir in scan_dirs:
                if sdir and sdir.exists():
                    for pattern in ("*.gguf", "*.safetensors"):
                        for f in sdir.glob(pattern):
                            fname = f.name
                            if fname not in known_files and fname not in seen_extra:
                                # Skip known non-encoders (VAE or transformer files)
                                if "vae" in fname.lower() or ("base" in fname.lower() and "quantfunc" in fname.lower()):
                                    continue
                                seen_extra.add(fname)
                                sz_gb = round(f.stat().st_size / (1024**3), 2)
                                te_result.append({
                                    "id": fname,
                                    "name": f"{fname} (~{sz_gb} GB, Local)",
                                    "file": fname,
                                    "size_gb": sz_gb,
                                    "precision": "custom",
                                    "recommended": False,
                                    "installed": True,
                                })

        return {
            "model_id": model_id,
            "model_name": model.name,
            "gpu_tier": gpu_tier,
            "gpu_sm": gpu_sm,
            "base_model_installed": base_installed,
            "text_encoders": te_result,
            "transformers": result,
        }

    # ---------- API: Custom Models ----------
    @app.get("/api/custom-models/files")
    async def api_custom_models_files():
        """List local files available in models/custom/ directory."""
        from backend.custom_models import list_custom_files, get_custom_models_dir
        return {
            "custom_dir": str(get_custom_models_dir()),
            "files": list_custom_files(),
        }

    @app.post("/api/custom-models")
    async def api_register_custom_model(body: dict):
        """Register a new custom model profile."""
        from backend.custom_models import register_custom_model
        try:
            res = register_custom_model(body)
            return {"ok": True, "model": res}
        except Exception as e:
            raise HTTPException(400, str(e))

    @app.delete("/api/custom-models/{model_id}")
    async def api_delete_custom_model(model_id: str):
        """Delete an imported custom model profile."""
        from backend.custom_models import delete_custom_model
        try:
            ok = delete_custom_model(model_id)
            if not ok:
                raise HTTPException(404, "Custom model manifest not found")
            return {"ok": True}
        except Exception as e:
            raise HTTPException(400, str(e))

    @app.post("/api/models/{model_id}/download")
    async def api_model_download(model_id: str, body: Optional[dict] = None):
        """Download missing files for a model. Runs in background thread.

        Body (optional):
            {"variant_id": "lighting-50x", "include_base": true, "include_precision": true}
        """
        if not STATE.gpu_info:
            raise HTTPException(400, "GPU not detected")

        registry = get_registry()
        try:
            model = registry.get(model_id)
        except KeyError:
            raise HTTPException(404, f"Model not found: {model_id}")

        # Determine which variant to download
        body = body or {}
        variant_id = body.get("variant_id")
        include_base = body.get("include_base", True)
        include_precision = body.get("include_precision", True)

        gpu_sm = STATE.gpu_info.sm if STATE.gpu_info else 0
        gpu_tier = STATE.gpu_info.tier if STATE.gpu_info else "50x-below"
        transformers_for_gpu = model.get_transformers_for_tier(gpu_sm) if gpu_sm else model.transformers
        selected = None
        if variant_id:
            selected = next((t for t in transformers_for_gpu if t.id == variant_id), None)
        if not selected and transformers_for_gpu:
            # Pick first recommended, or first available
            selected = next((t for t in transformers_for_gpu if t.recommended), None) or transformers_for_gpu[0]
        if not selected:
            raise HTTPException(400, f"No compatible transformer found for GPU SM={gpu_sm}")

        # Create a download job (reuse the job system)
        job_id = str(uuid.uuid4())[:8]
        STATE.register_job(job_id)

        def run_download():
            """Background download thread. ALWAYS updates job state, even on crash."""
            try:
                # Imports inside try so import errors get caught and reported to UI
                from backend.sources import get_source
                from backend.cache import CacheManager

                STATE.update_job(job_id, ProgressEvent(
                    status=JobStatus.DOWNLOADING,
                    progress=0.0,
                    message=f"Preparing to download {model.name} files...",
                ))

                settings = get_settings()
                source_spec = model.sources.primary
                if settings.download_source_priority == "hf_first" and model.sources.mirror and model.sources.mirror.type == "huggingface":
                    source_spec = model.sources.mirror
                elif settings.download_source_priority == "ms_first" and model.sources.mirror and model.sources.mirror.type == "modelscope":
                    source_spec = model.sources.mirror
                source = get_source(source_spec.type, token=_get_token_for_source(source_spec.type))
                repo_id = source_spec.repo_id
                revision = "master" if source_spec.type == "modelscope" else "main"
                cache = CacheManager()
                local_repo_dir = cache.repo_dir_for(repo_id)
                local_repo_dir.mkdir(parents=True, exist_ok=True)

                tasks = []
                if include_base and getattr(model, "engine", "quantfunc") != "sd_cpp" and model.base_models:
                    base_path = model.get_base_model_path(gpu_tier)
                    if base_path:
                        local_base = local_repo_dir / base_path
                        is_sdnq = getattr(model, "engine", "quantfunc") == "sdnq"
                        base_ready = (local_base / "model_index.json").exists() and (
                            not is_sdnq or (local_base / "vae" / "diffusion_pytorch_model.safetensors").exists()
                        )
                        if not base_ready:
                            sub = base_path.rstrip("/")
                            if sub in ("", "."):
                                if is_sdnq:
                                    patterns = [
                                        "model_index.json", "config.json", "*.json",
                                        "*/*.json", "*/*.jinja", "*/*.txt",
                                        "vae/**", "scheduler/**", "tokenizer/**", "processor/**",
                                    ]
                                else:
                                    patterns = [
                                        "model_index.json", "config.json", "*.json",
                                        "text_encoder/**", "text_encoder*/**",
                                        "vae/**", "scheduler/**", "tokenizer/**", "processor/**",
                                    ]
                                tasks.append(("base", ".", patterns))
                            else:
                                tasks.append(("base", sub, [f"{sub}/**"]))
                        else:
                            STATE.update_job(job_id, ProgressEvent(
                                status=JobStatus.DOWNLOADING, progress=0.1,
                                message="Base model already installed ✓",
                            ))
                    else:
                        raise RuntimeError(f"No base model for tier {gpu_tier}")

                # Transformer
                local_transformer = local_repo_dir / selected.path
                if not local_transformer.exists():
                    t_repo = getattr(selected, "repo_id", None) or repo_id
                    tasks.append(("transformer", selected.path, selected.path, t_repo))
                else:
                    STATE.update_job(job_id, ProgressEvent(
                        status=JobStatus.DOWNLOADING, progress=0.5,
                        message="Transformer already installed ✓",
                    ))

                # Precision config
                if include_precision:
                    klein_tier = STATE.gpu_info.klein_tier if STATE.gpu_info else None
                    for pc in model.precision_configs:
                        if pc.tier == gpu_tier:
                            if pc.klein_tier and klein_tier and pc.klein_tier != klein_tier:
                                continue
                            local_pc = local_repo_dir / pc.path
                            if not local_pc.exists():
                                tasks.append(("precision", pc.path, pc.path))
                            break

                # Text encoder (if model specifies text_encoders)
                if getattr(model, "text_encoders", None):
                    te_id = body.get("text_encoder_id")
                    te_spec = model.get_text_encoder(te_id)
                    if te_spec:
                        te_f = te_spec.get("file")
                        local_te = local_repo_dir / te_f
                        local_te_sub = local_repo_dir / "text_encoders" / te_f
                        if not local_te.exists() and not local_te_sub.exists():
                            te_source = te_spec.get("source", {})
                            te_repo = te_source.get("repo_id", repo_id)
                            te_remote = te_source.get("file_path", f"text_encoders/{te_f}")
                            tasks.append(("text_encoder", te_remote, te_remote, te_repo, te_f))

                # VAE (if model specifies engine_args.vae)
                vae_rel = getattr(model, "engine_args", {}).get("vae") if getattr(model, "engine_args", None) else None
                if vae_rel:
                    local_vae = local_repo_dir / vae_rel
                    existing_vae = next((
                        p for p in [
                            settings.models_dir / "Qwen-Image-2.1-4bit" / vae_rel,
                            settings.models_dir / "Qwen-Image-2.1-4bit" / "vae" / vae_rel,
                        ] if p.exists()
                    ), None)
                    if not local_vae.exists():
                        if existing_vae:
                            import shutil
                            local_vae.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(existing_vae, local_vae)
                            print(f"[download] Reused existing VAE from {existing_vae} -> {local_vae}")
                        else:
                            vae_repo = getattr(model, "engine_args", {}).get("vae_repo", "Comfy-Org/Qwen-Image-2.1")
                            vae_remote = getattr(model, "engine_args", {}).get("vae_remote", f"vae/{vae_rel}")
                            tasks.append(("vae", vae_remote, vae_remote, vae_repo, vae_rel))

                total_tasks = len(tasks)
                if total_tasks == 0:
                    STATE.update_job(job_id, ProgressEvent(
                        status=JobStatus.COMPLETE,
                        progress=1.0,
                        message="All files already installed ✓",
                    ))
                    return

                for i, item in enumerate(tasks):
                    kind = item[0]
                    file_path = item[1]
                    pattern = item[2]
                    progress_base = i / total_tasks
                    progress_end = (i + 1) / total_tasks
                    STATE.update_job(job_id, ProgressEvent(
                        status=JobStatus.DOWNLOADING,
                        progress=progress_base,
                        message=f"[{i+1}/{total_tasks}] Downloading {kind}: {file_path}...",
                    ))

                    if kind == "base":
                        # Snapshot download with allow_pattern for recursive dir
                        allow_pat = pattern if isinstance(pattern, list) else [pattern]
                        source.download_snapshot(
                            repo_id=repo_id,
                            local_dir=local_repo_dir,
                            allow_patterns=allow_pat,
                            revision=revision,
                        )
                    elif kind in ("text_encoder", "vae"):
                        te_repo = item[3] if len(item) > 3 else repo_id
                        te_fname = item[4] if len(item) > 4 else Path(file_path).name
                        te_src = get_source("huggingface", token=_get_token_for_source("huggingface"))
                        te_src.download_file(
                            repo_id=te_repo,
                            file_path=file_path,
                            local_dir=local_repo_dir,
                            revision="main",
                        )
                        dl_file = local_repo_dir / file_path
                        base_file = local_repo_dir / te_fname
                        if dl_file.exists() and not base_file.exists() and dl_file.resolve() != base_file.resolve():
                            import shutil
                            base_file.parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(dl_file), str(base_file))
                    else:
                        # Single file download
                        target_repo = item[3] if len(item) > 3 else repo_id
                        target_source = get_source("huggingface", token=_get_token_for_source("huggingface")) if len(item) > 3 else source
                        target_source.download_file(
                            repo_id=target_repo,
                            file_path=file_path,
                            local_dir=local_repo_dir,
                            revision=revision,
                        )

                    STATE.update_job(job_id, ProgressEvent(
                        status=JobStatus.DOWNLOADING,
                        progress=progress_end,
                        message=f"[{i+1}/{total_tasks}] {kind} downloaded ✓",
                    ))

                if model.id == "qwen_image_2_1":
                    try:
                        get_worker()._ensure_qwen_image_2_1_configs(local_repo_dir)
                    except Exception as e:
                        print(f"[download] Warning ensuring Qwen 2.1 configs: {e}")

                STATE.update_job(job_id, ProgressEvent(
                    status=JobStatus.COMPLETE,
                    progress=1.0,
                    message=f"All {total_tasks} files downloaded ✓",
                ))

            except Exception as e:
                # CRITICAL: Always update job state on failure, otherwise
                # frontend will poll forever waiting for a status that never comes.
                import traceback
                error_detail = f"{type(e).__name__}: {e}"
                print(f"[download] Job {job_id} failed: {error_detail}")
                traceback.print_exc()
                STATE.update_job(job_id, ProgressEvent(
                    status=JobStatus.ERROR,
                    error=error_detail,
                    message=f"Download failed: {error_detail}",
                ))

        thread = threading.Thread(target=run_download, daemon=True)
        thread.start()

        return {"job_id": job_id, "status": "downloading", "variant_id": selected.id}

    # ---------- API: Upload reference image for Img2Img ----------
    @app.post("/api/upload")
    async def api_upload(file: UploadFile = File(...)):
        """Upload a reference image for Img2Img / Image Edit mode."""
        if not file.filename:
            raise HTTPException(400, "No file provided")

        ext = Path(file.filename).suffix.lower()
        if ext not in (".png", ".jpg", ".jpeg", ".webp"):
            raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: .png, .jpg, .jpeg, .webp")

        settings = get_settings()
        uploads_dir = settings.output_dir / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)

        from backend.security import validate_filename
        safe_stem = validate_filename(f"ref_{uuid.uuid4().hex[:8]}{ext}")
        target_path = uploads_dir / safe_stem

        try:
            content = await file.read()
            if len(content) > 25 * 1024 * 1024:
                raise HTTPException(400, "File too large. Max 25 MB.")

            with open(target_path, "wb") as f:
                f.write(content)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"Failed to save upload: {e}")

        rel_path = f"uploads/{safe_stem}"
        return {
            "status": "ok",
            "filename": safe_stem,
            "path": rel_path,
            "url": f"/outputs/uploads/{safe_stem}",
        }

    # ---------- API: Generate ----------
    @app.post("/api/generate")
    async def api_generate(body: dict):
        """Start a generation job. Returns job_id for polling."""
        if not STATE.gpu_info:
            raise HTTPException(400, "GPU not detected")

        # Parse request
        try:
            model_id = body["model_id"]
            prompt = body["prompt"]
            variant_id = body.get("variant_id")
            raw_seed = body.get("seed")
            seed = int(raw_seed) if raw_seed not in (None, "", "random") else None
            width = int(body.get("width", 1024))
            height = int(body.get("height", 1024))
            steps = int(body.get("steps", 4))
            backend = body.get("backend", "lighting")
            raw_init_path = body.get("init_image_path")
            init_image_strength = float(body.get("init_image_strength", 0.7))
        except (KeyError, TypeError, ValueError) as e:
            raise HTTPException(400, f"Invalid request: {e}")

        # Normalize init_image_path
        init_image_path = None
        if raw_init_path:
            norm = str(raw_init_path).strip().replace("\\", "/")
            if norm.startswith("/outputs/"):
                norm = norm[len("/outputs/"):]
            elif norm.startswith("outputs/"):
                norm = norm[len("outputs/"):]
            elif norm.startswith("/"):
                norm = norm[1:]
            if norm:
                init_image_path = norm

        # Look up variant to determine backend + use_prequant
        registry = get_registry()
        try:
            model = registry.get(model_id)
        except KeyError:
            raise HTTPException(404, f"Model not found: {model_id}")

        model_engine = getattr(model, "engine", "quantfunc")
        if model_engine == "sd_cpp":
            if not is_sd_cpp_installed():
                raise HTTPException(400, "SD.CPP engine not installed. Go to Settings → Download SD.CPP.")
        elif model_engine == "sdnq":
            if not is_sdnq_installed():
                raise HTTPException(400, "SDNQ runtime not installed. Go to Settings → Install/Verify SDNQ.")
        else:
            if not (is_engine_installed(model.engine_version) or is_engine_installed()):
                raise HTTPException(400, "QuantFunc engine not installed. Go to Settings → Download engine.")

        if model.pipeline == "image_edit" and not init_image_path:
            raise HTTPException(400, "Reference image is required for image_edit models (upload an image or select one from preview/history)")

        # Find the selected transformer
        transformers_for_gpu = model.get_transformers_for_tier(STATE.gpu_info.sm, backend=backend)
        selected = None
        if variant_id:
            selected = next((t for t in transformers_for_gpu if t.id == variant_id), None)
            if not selected:
                selected = next((t for t in model.transformers if t.id == variant_id), None)
        if not selected and transformers_for_gpu:
            selected = transformers_for_gpu[0]
        if not selected:
            raise HTTPException(400, f"No compatible transformer found for backend={backend}")

        # Determine steps and CFG
        effective_steps = int(body["steps"]) if "steps" in body and body["steps"] not in (None, "") else selected.steps
        cfg_val = model.get_default_cfg()
        if "cfg" in body and body["cfg"] not in (None, ""):
            try:
                cfg_val = float(body["cfg"])
            except (ValueError, TypeError):
                pass

        # Determine negative prompt, noise & sampler
        negative_prompt = body.get("negative_prompt", "") or ""
        noise_val = 0.0
        if "noise" in body and body["noise"] not in (None, ""):
            try:
                noise_val = float(body["noise"])
            except (ValueError, TypeError):
                pass

        raw_sampler = str(body.get("sampler") or "").strip().lower()
        allowed_samplers = {
            "euler", "euler_a", "heun", "dpm++2m", "dpm++2m_sde",
            "dpm2", "dpm2_a", "lcm", "unipc"
        }
        sampler = raw_sampler if raw_sampler in allowed_samplers else None

        # Build GenerateRequest
        request = GenerateRequest(
            model_id=model_id,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            width=width,
            height=height,
            steps=effective_steps,
            cfg=cfg_val,
            noise=noise_val,
            backend=selected.backend,
            use_prequant=selected.prequant,
            init_image_path=Path(init_image_path) if init_image_path else None,
            init_image_strength=init_image_strength,
            variant_id=selected.id,
            sampler=sampler,
            text_encoder_id=body.get("text_encoder_id"),
            custom_text_encoder_path=body.get("custom_text_encoder_path"),
            vae=body.get("vae"),
            custom_vae_path=body.get("custom_vae_path"),
        )

        # Create job
        job_id = str(uuid.uuid4())[:8]
        STATE.register_job(job_id)

        # Run in background thread
        def run_job():
            try:
                worker = get_worker()

                def progress_cb(event: ProgressEvent):
                    STATE.update_job(job_id, event)

                result = worker.generate(request, STATE.gpu_info, progress_callback=progress_cb)
                STATE.update_job(job_id, ProgressEvent(
                    status=JobStatus.COMPLETE,
                    progress=1.0,
                    image_path=str(result.image_path),
                    message=f"Done in {result.generation_time_sec:.1f}s",
                ))
            except Exception as e:
                # CRITICAL: Always update job state on failure, otherwise
                # frontend will poll forever waiting for a status that never comes.
                import traceback
                error_detail = f"{type(e).__name__}: {e}"
                print(f"[generate] Job {job_id} failed: {error_detail}")
                traceback.print_exc()
                STATE.update_job(job_id, ProgressEvent(
                    status=JobStatus.ERROR,
                    error=error_detail,
                    message=f"Generation failed: {error_detail}",
                ))

        thread = threading.Thread(target=run_job, daemon=True)
        thread.start()

        return {"job_id": job_id, "status": "queued"}

    @app.get("/api/jobs/{job_id}")
    async def api_job_status(job_id: str):
        job = STATE.get_job(job_id)
        if not job:
            raise HTTPException(404, f"Job not found: {job_id}")
        return job

    @app.post("/api/jobs/{job_id}/cancel")
    async def api_job_cancel(job_id: str):
        worker = get_worker()
        worker.cancel()
        STATE.update_job(job_id, ProgressEvent(
            status=JobStatus.CANCELLED,
            message="Cancelled by user",
        ))
        return {"status": "cancelled"}

    # ---------- API: Settings ----------
    @app.patch("/api/settings")
    async def api_settings_update(body: dict):
        settings = get_settings()
        # Whitelist fields that can be updated
        allowed = {
            "lang", "vram_mode", "auto_optimize",
            "download_source_priority", "auto_update_engine",
            "huggingface_token", "modelscope_token",
            "quantfunc_api_key",
            "fbcache_cond_threshold", "fbcache_uncond_threshold",
            "rotation_block_size", "log_level",
            "worker_restart_on_crash",
            "cuda_bin_path",
            "qwen_vae_type",
        }
        updates = {k: v for k, v in body.items() if k in allowed}
        if updates:
            settings.update(**updates)
            return {"status": "saved", "updated": list(updates.keys())}
        return {"status": "no_changes"}

    # ---------- API: Engine ----------
    @app.get("/api/engine/status")
    async def api_engine_status():
        """Get status of all supported engines."""
        return {
            "status": "ok",
            "engines": _get_engines_info(),
        }

    @app.post("/api/engine/download")
    async def api_engine_download(body: Optional[dict] = None):
        """Download engine binary (QuantFunc v0.0.17 / v0.0.12, SD.CPP, or SDNQ).

        Body (optional):
            {"engine": "quantfunc", "version": "0.0.17"} or
            {"engine": "quantfunc", "version": "0.0.12"} or
            {"engine": "sd_cpp", "force": false} or
            {"engine": "sdnq", "force": false}
        """
        body = body or {}
        engine_type = body.get("engine", "quantfunc")
        version = body.get("version")

        try:
            if engine_type == "sd_cpp":
                from backend.sd_cpp import download_sd_cpp
                binary_path = await asyncio.to_thread(download_sd_cpp, force=bool(body.get("force", False)))
                return {
                    "status": "ok",
                    "engine": "sd_cpp",
                    "message": f"SD.CPP engine installed: {binary_path}",
                    "path": str(binary_path),
                }
            elif engine_type == "sdnq":
                from backend.sdnq_engine import install_sdnq, get_sdnq_version
                sdnq_path = await asyncio.to_thread(install_sdnq, force=bool(body.get("force", False)))
                return {
                    "status": "ok",
                    "engine": "sdnq",
                    "message": f"SDNQ ready ({get_sdnq_version()}): {sdnq_path}",
                    "path": str(sdnq_path),
                }
            else:
                target_ver = version or get_settings().engine_version
                binary_path = await asyncio.to_thread(download_engine, version=target_ver)
                return {
                    "status": "ok",
                    "engine": "quantfunc",
                    "version": target_ver,
                    "message": f"QuantFunc engine v{target_ver} installed and verified: {binary_path}",
                    "path": str(binary_path),
                }
        except Exception as e:
            raise HTTPException(500, f"Engine download failed: {e}")

    @app.post("/api/engine/verify")
    async def api_engine_verify(body: Optional[dict] = None):
        """Verify engine SHA-256 against remote verify.json."""
        body = body or {}
        version = body.get("version")
        target_ver = version or get_settings().engine_version
        result = await asyncio.to_thread(verify_engine, version=target_ver)
        return {
            "status": result.status,
            "version": target_ver,
            "message": result.message,
            "local_sha256": result.local_sha256,
            "remote_sha256": result.remote_sha256,
            "remote_version": result.remote_version,
        }

    @app.post("/api/engine/ensure-config")
    async def api_engine_ensure_config(body: Optional[dict] = None):
        """Install/restore bin/<os>/config.json with the bundled free-trial test key.

        Body (optional): {"force": true} — overwrite existing config.json
        (use when the bundled test key was deleted or corrupted).
        """
        try:
            force = bool((body or {}).get("force", False))
            cfg_path, source = ensure_config_json(force=force)
            return {
                "status": "ok",
                "message": f"Engine auth config ready ({source}): {cfg_path}",
                "path": str(cfg_path),
                "source": source,
            }
        except Exception as e:
            raise HTTPException(500, f"Failed to install config.json: {e}")

    @app.get("/api/cuda/status")
    async def api_cuda_status():
        """Check if CUDA runtime libraries are available."""
        from backend.engine import check_cuda_dependencies
        return check_cuda_dependencies()

    @app.post("/api/cuda/download")
    async def api_cuda_download():
        """Download and extract CUDA dependency bundle from ModelScope (~1 GB).

        Downloads cu13-dep-win32.zip (or equivalent for platform/CUDA version),
        extracts cublas64_XX.dll, cudnn64_9.dll, etc. into engine_dir.
        """
        from backend.engine import download_cuda_deps, check_cuda_dependencies

        job_id = str(uuid.uuid4())[:8]
        STATE.register_job(job_id)

        def run_cuda_download():
            try:
                STATE.update_job(job_id, ProgressEvent(
                    status=JobStatus.DOWNLOADING,
                    progress=0.0,
                    message="Downloading CUDA deps bundle (~1 GB)...",
                ))

                result = download_cuda_deps(progress_callback=lambda d, t, s: {
                    STATE.update_job(job_id, ProgressEvent(
                        status=JobStatus.DOWNLOADING,
                        progress=d / t if t > 0 else 0,
                        downloaded_bytes=d,
                        total_bytes=t,
                        speed_bps=s,
                        message=f"Downloading CUDA deps: {d / 1024 / 1024:.0f} / {t / 1024 / 1024:.0f} MB",
                    ))
                })

                # Verify after extraction
                cuda_status = check_cuda_dependencies()
                if cuda_status["installed"]:
                    STATE.update_job(job_id, ProgressEvent(
                        status=JobStatus.COMPLETE,
                        progress=1.0,
                        message=f"CUDA deps installed: {result['files_count']} files extracted ✓",
                    ))
                else:
                    STATE.update_job(job_id, ProgressEvent(
                        status=JobStatus.ERROR,
                        error=f"CUDA deps downloaded but still missing: {cuda_status['missing']}",
                        message=f"Downloaded but still missing: {cuda_status['missing']}",
                    ))

            except Exception as e:
                import traceback
                error_detail = f"{type(e).__name__}: {e}"
                print(f"[cuda] Download job {job_id} failed: {error_detail}")
                traceback.print_exc()
                STATE.update_job(job_id, ProgressEvent(
                    status=JobStatus.ERROR,
                    error=error_detail,
                    message=f"CUDA deps download failed: {error_detail}",
                ))

        thread = threading.Thread(target=run_cuda_download, daemon=True)
        thread.start()

        return {"job_id": job_id, "status": "downloading"}

    # ---------- API: History ----------
    @app.get("/api/history")
    async def api_history():
        cache = CacheManager()
        outputs = cache.list_outputs(limit=100)
        result = []
        for p in outputs:
            st = p.stat()  # cache: avoid two syscalls per file
            item = {
                "path": str(p),
                "name": p.name,
                "size": st.st_size,
                "modified": st.st_mtime,
                "url": f"/outputs/{p.name}",
                "prompt": "",
                "model_id": "",
                "variant_id": None,
                "seed": None,
                "width": None,
                "height": None,
                "steps": None,
                "cfg": None,
                "noise": None,
                "sampler": None,
                "vae": None,
                "backend": None,
                "generation_time_sec": None,
            }
            sidecar = p.with_suffix(".json")
            if sidecar.exists():
                try:
                    meta = json.loads(sidecar.read_text(encoding="utf-8"))
                    item.update({
                        "prompt": meta.get("prompt", ""),
                        "negative_prompt": meta.get("negative_prompt", ""),
                        "model_id": meta.get("model_id", ""),
                        "variant_id": meta.get("variant_id"),
                        "seed": meta.get("seed"),
                        "width": meta.get("width"),
                        "height": meta.get("height"),
                        "steps": meta.get("steps"),
                        "cfg": meta.get("cfg"),
                        "noise": meta.get("noise"),
                        "sampler": meta.get("sampler"),
                        "vae": meta.get("vae"),
                        "backend": meta.get("backend"),
                        "generation_time_sec": meta.get("generation_time_sec"),
                    })
                except Exception as meta_err:
                    print(f"[history] Failed to parse sidecar for {p.name}: {meta_err}")
            result.append(item)
        return result

    @app.delete("/api/history/{filename}")
    async def api_history_delete(filename: str):
        from backend.security import validate_filename
        try:
            validate_filename(filename)
        except ValueError as e:
            raise HTTPException(400, f"Invalid filename: {e}")

        output_dir = get_settings().output_dir
        target_file = output_dir / filename

        try:
            target_file.unlink()
            sidecar = target_file.with_suffix(".json")
            if sidecar.exists():
                sidecar.unlink()
            return {"status": "ok", "message": f"Deleted {filename}"}
        except FileNotFoundError:
            raise HTTPException(404, f"File not found: {filename}")
        except Exception as e:
            raise HTTPException(500, f"Failed to delete file: {e}")


    # ---------- Root: serve index.html ----------
    @app.get("/", response_class=HTMLResponse)
    async def homepage():
        return build_index_html()

    @app.get("/index.html", response_class=HTMLResponse)
    async def index_html_alias():
        return build_index_html()

    return app


# ============================================================
# Main
# ============================================================
def _free_port_if_in_use(host: str, port: int) -> None:
    """If a previous QuantFunc-WebUI Python instance is still listening on `port`, terminate it cleanly."""
    import os
    import socket
    import time

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        if s.connect_ex((host if host != "0.0.0.0" else "127.0.0.1", port)) != 0:
            return  # Port is already free

    my_pid = os.getpid()
    try:
        import psutil
        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr and conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
                pid = conn.pid
                if pid and pid != my_pid:
                    try:
                        proc = psutil.Process(pid)
                        pname = (proc.name() or "").lower()
                        if "python" in pname:
                            print(f"[startup] Port {port} is held by previous Python process (PID {pid}) — terminating it...")
                            proc.terminate()
                            try:
                                proc.wait(timeout=3.0)
                            except psutil.TimeoutExpired:
                                proc.kill()
                            time.sleep(0.4)
                    except Exception as e:
                        print(f"[startup] Could not terminate PID {pid} on port {port}: {e}")
    except Exception:
        pass


def main():
    """Entry point — initialize state, build app, launch server."""
    import argparse
    from backend.logging_config import setup_logging

    parser = argparse.ArgumentParser(description="QuantFunc WebUI server")
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Show detailed debug output and HTTP access logs in console (default: quiet)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to bind to (default: 7860)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory to store log files (default: logs/)",
    )

    args = parser.parse_args()

    _free_port_if_in_use(args.host, args.port)

    log_file, uvicorn_log_config = setup_logging(verbose=args.verbose, log_dir=args.log_dir)

    init_state()
    app = create_app()

    import uvicorn

    gpu_desc = (
        f"{STATE.gpu_info.name} (SM {STATE.gpu_info.sm}, {STATE.gpu_info.vram_gb:.1f} GB VRAM)"
        if STATE.gpu_info else "No NVIDIA GPU detected"
    )

    print("\n" + "=" * 60)
    print(" QuantFunc WebUI")
    print(f" URL:     http://{args.host}:{args.port}")
    print(f" GPU:     {gpu_desc}")
    print(f" Logs:    {log_file} (all activity logged silently)")
    if args.verbose:
        print(" Mode:    VERBOSE (detailed console logs enabled)")
    else:
        print(" Mode:    QUIET console (use --verbose for detailed HTTP/debug logs)")
    print("=" * 60 + "\n")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_config=uvicorn_log_config,
        log_level="debug" if args.verbose else "info",
        reload=False,
    )


if __name__ == "__main__":
    main()
