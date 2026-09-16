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

import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# ---------- Backend imports ----------
from backend.cache import CacheManager
from backend.engine import (
    check_engine_status,
    download_engine,
    engine_binary_path,
    engine_cli_path,
    engine_config_json_path,
    ensure_config_json,
    is_engine_installed,
    read_engine_api_key,
    verify_engine,
)
from backend.gpu import detect_gpu, GPUInfo
from backend.registry import get_registry, reload_registry
from backend.settings import get_settings, reload_settings
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
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
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
        print(f"[startup]   SM: {g.sm} → tier={g.tier}, klein_tier={g.klein_tier}")
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
# Build the main HTML page
# ============================================================
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
        },
        "gpu": gpu_dict,
        "engine_status": {
            "status": engine_status.status,
            "message": engine_status.message,
        },
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

    # ---------- API: Config ----------
    @app.get("/api/config")
    async def api_config():
        """Initial config for frontend."""
        settings = get_settings()
        registry = get_registry()

        # Convert models to JSON-serializable list
        models_list = []
        for m in registry.list_all():
            models_list.append({
                "id": m.id,
                "name": m.name,
                "family": m.family,
                "pipeline": m.pipeline,
                "description": m.description,
                "key_protected": bool(m.key_protected),
                "requirements": {
                    "vram_min_gb": m.requirements.vram_min_gb,
                    "vram_recommended_gb": m.requirements.vram_recommended_gb,
                    "gpu_min_sm": m.requirements.gpu_min_sm,
                    "default_vram_mode": m.requirements.default_vram_mode,
                },
            })

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
            },
            "gpu": {
                "name": STATE.gpu_info.name if STATE.gpu_info else None,
                "sm": STATE.gpu_info.sm if STATE.gpu_info else None,
                "vram_mb": STATE.gpu_info.vram_mb if STATE.gpu_info else None,
                "driver_version": STATE.gpu_info.driver_version if STATE.gpu_info else None,
                "tier": STATE.gpu_info.tier if STATE.gpu_info else None,
                "klein_tier": STATE.gpu_info.klein_tier if STATE.gpu_info else None,
            },
            "engine_status": {
                "status": check_engine_status().status,
                "message": check_engine_status().message,
            },
            "api_auth": _api_auth_status(),
            "models": models_list,
        }

    # ---------- API: Models ----------
    @app.get("/api/models")
    async def api_models():
        registry = get_registry()
        return [
            {
                "id": m.id,
                "name": m.name,
                "family": m.family,
                "pipeline": m.pipeline,
                "description": m.description,
                "key_protected": bool(m.key_protected),
                "requirements": {
                    "vram_min_gb": m.requirements.vram_min_gb,
                    "vram_recommended_gb": m.requirements.vram_recommended_gb,
                    "gpu_min_sm": m.requirements.gpu_min_sm,
                },
            }
            for m in registry.list_all()
        ]

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

        # Check cache state for each
        cache = CacheManager()
        source_spec = model.sources.primary
        repo_id = source_spec.repo_id

        result = []
        for t in transformers_for_gpu:
            entry = cache.check_file(
                repo_id=repo_id,
                file_path=t.path,
                expected_size=t.size_bytes or (int(t.size_gb * 1024**3) if t.size_gb else 0),
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
        if base_path:
            local_base = cache.repo_dir_for(repo_id) / base_path
            base_installed = (local_base / "model_index.json").exists()

        return {
            "model_id": model_id,
            "model_name": model.name,
            "gpu_tier": gpu_tier,
            "gpu_sm": gpu_sm,
            "base_model_installed": base_installed,
            "transformers": result,
        }

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

        transformers_for_gpu = model.get_transformers_for_tier(STATE.gpu_info.sm)
        selected = None
        if variant_id:
            selected = next((t for t in transformers_for_gpu if t.id == variant_id), None)
        if not selected and transformers_for_gpu:
            # Pick first recommended, or first available
            selected = next((t for t in transformers_for_gpu if t.recommended), None) or transformers_for_gpu[0]
        if not selected:
            raise HTTPException(400, f"No compatible transformer found for GPU SM={STATE.gpu_info.sm}")

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

                source_spec = model.sources.primary
                source = get_source(source_spec.type, token=_get_token_for_source(source_spec.type))
                repo_id = source_spec.repo_id
                revision = "master" if source_spec.type == "modelscope" else "main"
                cache = CacheManager()
                local_repo_dir = cache.repo_dir_for(repo_id)

                tasks = []
                if include_base:
                    base_path = model.get_base_model_path(STATE.gpu_info.tier)
                    if base_path:
                        local_base = local_repo_dir / base_path
                        if not (local_base / "model_index.json").exists():
                            tasks.append(("base", base_path.rstrip("/"), f"{base_path.rstrip('/')}/**"))
                        else:
                            STATE.update_job(job_id, ProgressEvent(
                                status=JobStatus.DOWNLOADING, progress=0.1,
                                message="Base model already installed ✓",
                            ))
                    else:
                        raise RuntimeError(f"No base model for tier {STATE.gpu_info.tier}")

                # Transformer
                local_transformer = local_repo_dir / selected.path
                if not local_transformer.exists():
                    tasks.append(("transformer", selected.path, selected.path))
                else:
                    STATE.update_job(job_id, ProgressEvent(
                        status=JobStatus.DOWNLOADING, progress=0.5,
                        message="Transformer already installed ✓",
                    ))

                # Precision config
                if include_precision:
                    for pc in model.precision_configs:
                        if pc.tier == STATE.gpu_info.tier:
                            local_pc = local_repo_dir / pc.path
                            if not local_pc.exists():
                                tasks.append(("precision", pc.path, pc.path))
                            break

                total_tasks = len(tasks)
                if total_tasks == 0:
                    STATE.update_job(job_id, ProgressEvent(
                        status=JobStatus.COMPLETE,
                        progress=1.0,
                        message="All files already installed ✓",
                    ))
                    return

                for i, (kind, file_path, pattern) in enumerate(tasks):
                    progress_base = i / total_tasks
                    progress_end = (i + 1) / total_tasks
                    STATE.update_job(job_id, ProgressEvent(
                        status=JobStatus.DOWNLOADING,
                        progress=progress_base,
                        message=f"[{i+1}/{total_tasks}] Downloading {kind}: {file_path}...",
                    ))

                    if kind == "base":
                        # Snapshot download with allow_pattern for recursive dir
                        source.download_snapshot(
                            repo_id=repo_id,
                            local_dir=local_repo_dir,
                            allow_patterns=[pattern],
                            revision=revision,
                        )
                    else:
                        # Single file download
                        source.download_file(
                            repo_id=repo_id,
                            file_path=file_path,
                            local_dir=local_repo_dir,
                            revision=revision,
                        )

                    STATE.update_job(job_id, ProgressEvent(
                        status=JobStatus.DOWNLOADING,
                        progress=progress_end,
                        message=f"[{i+1}/{total_tasks}] {kind} downloaded ✓",
                    ))

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

        if not is_engine_installed():
            raise HTTPException(400, "Engine not installed. Go to Settings → Download engine.")

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

        if model.pipeline == "image_edit" and not init_image_path:
            raise HTTPException(400, "Reference image is required for image_edit models (upload an image or select one from preview/history)")

        # Find the selected transformer
        transformers_for_gpu = model.get_transformers_for_tier(STATE.gpu_info.sm, backend=backend)
        selected = None
        if variant_id:
            selected = next((t for t in transformers_for_gpu if t.id == variant_id), None)
        if not selected and transformers_for_gpu:
            selected = transformers_for_gpu[0]
        if not selected:
            raise HTTPException(400, f"No compatible transformer found for backend={backend}")

        # Build GenerateRequest
        request = GenerateRequest(
            model_id=model_id,
            prompt=prompt,
            seed=seed,
            width=width,
            height=height,
            steps=selected.steps,
            backend=selected.backend,
            use_prequant=selected.prequant,
            init_image_path=Path(init_image_path) if init_image_path else None,
            init_image_strength=init_image_strength,
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
        }
        updates = {k: v for k, v in body.items() if k in allowed}
        if updates:
            settings.update(**updates)
            return {"status": "saved", "updated": list(updates.keys())}
        return {"status": "no_changes"}

    # ---------- API: Engine ----------
    @app.post("/api/engine/download")
    async def api_engine_download():
        """Download engine binary from ModelScope (verify-before-replace)."""
        try:
            binary_path = download_engine()
            return {
                "status": "ok",
                "message": f"Engine downloaded and verified: {binary_path}",
                "path": str(binary_path),
            }
        except Exception as e:
            raise HTTPException(500, f"Engine download failed: {e}")

    @app.post("/api/engine/verify")
    async def api_engine_verify():
        """Verify engine SHA-256 against remote verify.json."""
        result = verify_engine()
        return {
            "status": result.status,
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
                "seed": None,
                "width": None,
                "height": None,
                "steps": None,
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
                        "seed": meta.get("seed"),
                        "width": meta.get("width"),
                        "height": meta.get("height"),
                        "steps": meta.get("steps"),
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
def main():
    """Entry point — initialize state, build app, launch server."""
    init_state()

    app = create_app()

    # Launch via uvicorn directly — no gradio dependency needed.
    import uvicorn

    print("\nLaunching QuantFunc WebUI on http://127.0.0.1:7860")
    print("Press Ctrl+C to stop.\n")

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=7860,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
