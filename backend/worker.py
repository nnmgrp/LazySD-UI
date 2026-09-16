"""
Worker — subprocess wrapper for QuantFunc CLI.

Spawns quantfunc / quantfunc.exe as a subprocess, captures stdout/stderr,
parses progress events, returns the generated image.

Architecture (per technical report):
    Gradio UI → backend.worker.generate() → subprocess.run([quantfunc, ...]) → PNG file

CRITICAL: All subprocess arguments go through backend.security.safe_subprocess_args().
NEVER use shell=True or os.system().
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue, Empty
from typing import Callable, Optional

from .cache import CacheManager
from .engine import engine_cli_path, is_engine_installed, has_api_key_available
from .gpu import GPUInfo
from .registry import get_registry, ModelSpec
from .security import (
    safe_subprocess_args,
    validate_int,
    validate_path,
    validate_prompt,
    validate_resolution,
)
from .settings import VRAMMode, get_settings


# ---------- Status enum ----------
class JobStatus:
    QUEUED = "queued"
    LOADING = "loading"
    DOWNLOADING = "downloading"
    INITIALIZING = "initializing"
    GENERATING = "generating"
    SAVING = "saving"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"


# ---------- Progress event ----------
@dataclass
class ProgressEvent:
    """A progress event emitted during generation."""
    status: str                       # JobStatus value
    progress: float = 0.0             # 0.0 to 1.0
    message: str = ""
    step: Optional[int] = None        # Current denoising step
    total_steps: Optional[int] = None  # Total denoising steps
    speed_bps: Optional[float] = None  # Download speed (bytes/sec) — only when downloading
    downloaded_bytes: Optional[int] = None
    total_bytes: Optional[int] = None
    image_path: Optional[str] = None  # Final image path (when complete)
    error: Optional[str] = None       # Error message (when error)


# ---------- Generate request ----------
@dataclass
class GenerateRequest:
    """Input parameters for a generate call."""
    model_id: str
    prompt: str
    negative_prompt: str = ""
    seed: Optional[int] = None        # None = random
    width: int = 1024
    height: int = 1024
    steps: int = 4
    cfg: float = 1.0
    backend: str = "lighting"         # "lighting" | "svdq"
    use_prequant: bool = False
    init_image_path: Optional[Path] = None  # For img2img / image edit
    init_image_strength: float = 0.7
    lora_paths: list[Path] = field(default_factory=list)
    lora_strengths: list[float] = field(default_factory=list)
    output_filename: Optional[str] = None


# ---------- Result ----------
@dataclass
class GenerateResult:
    """Result of a successful generation."""
    image_path: Path
    seed_used: int
    generation_time_sec: float
    model_id: str
    backend: str
    steps: int
    prompt: str


# ---------- Progress parsing ----------
# Engine emits lines like:
#   [step 1/4]
#   [step 2/4]
#   [progress 0.45]
#   [complete] /path/to/output.png
_STEP_RE = re.compile(r"\[step\s+(\d+)/(\d+)\]", re.IGNORECASE)
_PROGRESS_RE = re.compile(r"\[progress\s+([0-9.]+)\]", re.IGNORECASE)
_COMPLETE_RE = re.compile(r"\[complete\]\s*(\S+)", re.IGNORECASE)
_ERROR_RE = re.compile(r"\[error\]\s*(.+)", re.IGNORECASE)

# Engine-side key-protection failure (checkObfuscation in ComponentImpl.cpp).
# Matched against raw output lines so we can give actionable advice.
_KEY_PROTECTED_RE = re.compile(
    r"checkObfuscation|key-protected|API key is required|provide --apikey|"
    r"api[_ ]?key.*(invalid|required|missing)",
    re.IGNORECASE,
)

_KEY_PROTECTED_HELP = (
    "This model is key-protected. Fix it in one of these ways:\n"
    "  1. Settings → Engine → 'Restore bundled test key' — restores "
    "bin/<os>/config.json with the shared free-trial test key (works during "
    "the public beta, no registration needed).\n"
    "  2. Settings → Engine → 'QuantFunc API key' — enter your own key "
    "(get one free at https://www.quantfunc.com)."
)


def parse_engine_output(line: str) -> Optional[ProgressEvent]:
    """Parse a single line of engine stdout. Returns ProgressEvent or None."""
    line = line.strip()
    if not line:
        return None

    # Step progress
    m = _STEP_RE.search(line)
    if m:
        step, total = int(m.group(1)), int(m.group(2))
        progress = step / total if total > 0 else 0.0
        return ProgressEvent(
            status=JobStatus.GENERATING,
            progress=progress,
            step=step,
            total_steps=total,
            message=f"Step {step}/{total}",
        )

    # Explicit progress
    m = _PROGRESS_RE.search(line)
    if m:
        return ProgressEvent(
            status=JobStatus.GENERATING,
            progress=float(m.group(1)),
            message=f"{float(m.group(1)) * 100:.0f}%",
        )

    # Completion
    m = _COMPLETE_RE.search(line)
    if m:
        return ProgressEvent(
            status=JobStatus.COMPLETE,
            progress=1.0,
            image_path=m.group(1),
            message="Complete",
        )

    # Error
    m = _ERROR_RE.search(line)
    if m:
        return ProgressEvent(
            status=JobStatus.ERROR,
            error=m.group(1),
            message=m.group(1),
        )

    # Unrecognized line — treat as info message
    return ProgressEvent(
        status=JobStatus.GENERATING,
        message=line,
    )


# ---------- Worker ----------
class Worker:
    """Subprocess wrapper for QuantFunc CLI. One job at a time."""

    def __init__(self):
        self.cache = CacheManager()
        self._current_process: Optional[subprocess.Popen] = None
        self._cancel_flag = threading.Event()
        self._lock = threading.Lock()

    # ---------- Pre-flight checks ----------
    def _check_engine(self) -> None:
        """Verify engine binary is installed."""
        if not is_engine_installed():
            raise RuntimeError(
                "QuantFunc engine not installed. "
                "Go to Settings → Download engine to install it."
            )

    def _check_cuda_libs(self) -> None:
        """Verify critical CUDA DLLs are available. If missing, give clear instructions.

        cuDNN is OPTIONAL — warn but don't block, engine may not need it
        (or may report the missing DLL in its own error message).
        """
        from .engine import check_cuda_dependencies
        result = check_cuda_dependencies()
        if not result["installed"]:
            missing = result["missing"]
            dep_filename = result.get("dep_bundle_filename", "")
            raise RuntimeError(
                f"CUDA libraries missing: {', '.join(missing)}. "
                f"The QuantFunc engine needs CUDA runtime to work. "
                f"Go to Settings → Download CUDA deps to download {dep_filename} "
                f"(~1 GB, extracts automatically), or install CUDA Toolkit "
                f"from https://developer.nvidia.com/cuda-toolkit"
            )
        # Optional warning for cuDNN (don't block)
        if result.get("missing_optional"):
            missing_opt = result["missing_optional"]
            print(f"[worker] WARNING: Optional CUDA libs missing: {', '.join(missing_opt)} "
                  f"(cuDNN — engine may fail if it needs it)")

    def _check_gpu(self, gpu_info: GPUInfo, model: ModelSpec) -> None:
        """Verify GPU is compatible with model."""
        if gpu_info.sm < model.requirements.gpu_min_sm:
            raise RuntimeError(
                f"GPU SM {gpu_info.sm} is below minimum required {model.requirements.gpu_min_sm} "
                f"for model {model.name}."
            )
        if gpu_info.vram_gb < model.requirements.vram_min_gb:
            raise RuntimeError(
                f"VRAM {gpu_info.vram_gb:.1f} GB is below minimum {model.requirements.vram_min_gb} GB "
                f"for model {model.name}."
            )

    def _resolve_local_paths(
        self,
        model: ModelSpec,
        required_files: dict,
        gpu_info: GPUInfo,
        progress_callback: Optional[Callable[[ProgressEvent], None]] = None,
    ) -> dict:
        """Resolve remote file paths to local cache paths.

        Downloads missing files if needed (lazy download on Generate).
        """
        from .sources import get_source

        def emit(event: ProgressEvent):
            if progress_callback:
                progress_callback(event)

        settings = get_settings()
        source_spec = model.sources.primary
        source = get_source(source_spec.type, token=self._get_token(source_spec.type))

        # Repo ID for cache lookup
        repo_id = source_spec.repo_id
        local_repo_dir = self.cache.repo_dir_for(repo_id)

        # Check each required file
        required_paths = {}
        need_base_download = False
        need_transformer_download = False
        need_precision_download = False

        # Base model — check if directory exists with model_index.json
        base_path = required_files["base_model_path"]
        local_base = local_repo_dir / base_path
        if not (local_base / "model_index.json").exists():
            need_base_download = True
        required_paths["model_dir"] = local_base

        # Transformer file (top-level transformer/ dir, NOT inside base-model)
        transformer_path = required_files["transformer_path"]
        local_transformer = local_repo_dir / transformer_path
        if not local_transformer.exists():
            need_transformer_download = True
        required_paths["transformer"] = local_transformer

        # Optional precision config
        if required_files.get("precision_config_path"):
            precision_path = required_files["precision_config_path"]
            local_precision = local_repo_dir / precision_path
            if not local_precision.exists():
                need_precision_download = True
            required_paths["precision_config"] = local_precision

        # Optional prequant (Klein 9B doesn't have these, but keep for future models)
        if required_files.get("prequant_weights_path"):
            prequant_path = required_files["prequant_weights_path"]
            local_prequant = local_repo_dir / prequant_path
            required_paths["mod_weights"] = local_prequant
            if not local_prequant.exists():
                # Will be included in the download
                pass

        # ---------- Download if needed ----------
        revision = "master" if source_spec.type == "modelscope" else "main"

        # 1. Download base model (entire subdir recursively)
        if need_base_download:
            emit(ProgressEvent(
                status=JobStatus.DOWNLOADING,
                progress=0.1,
                message=f"Downloading base model ({base_path.rstrip('/')})...",
            ))
            base_subdir = base_path.rstrip("/")
            # Use ** pattern for recursive download of the entire base model dir
            allow_patterns = [f"{base_subdir}/**"]
            source.download_snapshot(
                repo_id=repo_id,
                local_dir=local_repo_dir,
                allow_patterns=allow_patterns,
                revision=revision,
            )
            emit(ProgressEvent(
                status=JobStatus.DOWNLOADING,
                progress=0.5,
                message="Base model downloaded ✓",
            ))

        # 2. Download transformer file (single large file)
        if need_transformer_download:
            emit(ProgressEvent(
                status=JobStatus.DOWNLOADING,
                progress=0.5,
                message=f"Downloading transformer ({transformer_path})...",
            ))
            source.download_file(
                repo_id=repo_id,
                file_path=transformer_path,
                local_dir=local_repo_dir,
                revision=revision,
            )
            emit(ProgressEvent(
                status=JobStatus.DOWNLOADING,
                progress=0.9,
                message="Transformer downloaded ✓",
            ))

        # 3. Download precision config (small JSON file)
        if need_precision_download:
            emit(ProgressEvent(
                status=JobStatus.DOWNLOADING,
                progress=0.95,
                message="Downloading precision config...",
            ))
            source.download_file(
                repo_id=repo_id,
                file_path=required_files["precision_config_path"],
                local_dir=local_repo_dir,
                revision=revision,
            )

        return required_paths

    def _get_token(self, source_type: str) -> Optional[str]:
        """Get auth token for a source type."""
        settings = get_settings()
        if source_type == "huggingface":
            return settings.huggingface_token
        elif source_type == "modelscope":
            return settings.modelscope_token
        return None

    # ---------- Generate ----------
    def generate(
        self,
        request: GenerateRequest,
        gpu_info: GPUInfo,
        progress_callback: Optional[Callable[[ProgressEvent], None]] = None,
    ) -> GenerateResult:
        """Run a single text-to-image generation job.

        Args:
            request: GenerateRequest with prompt + model + params
            gpu_info: Detected GPU info
            progress_callback: Optional callback for progress events

        Returns:
            GenerateResult with image path + metadata

        Raises:
            RuntimeError: If engine not installed, GPU incompatible, or inference fails
        """
        self._cancel_flag.clear()

        def emit(event: ProgressEvent):
            if progress_callback:
                progress_callback(event)

        # ----- Phase 1: Pre-flight checks -----
        emit(ProgressEvent(status=JobStatus.LOADING, message="Checking engine..."))
        self._check_engine()

        emit(ProgressEvent(status=JobStatus.LOADING, message="Checking CUDA libraries..."))
        self._check_cuda_libs()

        # Load model from registry
        registry = get_registry()
        model = registry.get(request.model_id)

        emit(ProgressEvent(status=JobStatus.LOADING, message=f"Checking GPU compatibility for {model.name}..."))
        self._check_gpu(gpu_info, model)

        # ----- Phase 1.5: Auth pre-flight (key-protected models) -----
        if getattr(model, "key_protected", False) and not has_api_key_available():
            raise RuntimeError(
                f"{model.name} is key-protected, but no API key is available. "
                + _KEY_PROTECTED_HELP
            )

        # ----- Phase 2: Resolve required files (lazy download) -----
        emit(ProgressEvent(status=JobStatus.LOADING, message="Resolving required files..."))
        required_files = model.resolve_required_files(
            gpu_sm=gpu_info.sm,
            backend=request.backend,
            steps=request.steps,
            use_prequant=request.use_prequant,
        )

        emit(ProgressEvent(status=JobStatus.DOWNLOADING, message="Checking cache for missing files..."))
        local_paths = self._resolve_local_paths(model, required_files, gpu_info, progress_callback=emit)

        # ----- Phase 3: Validate inputs -----
        emit(ProgressEvent(status=JobStatus.INITIALIZING, message="Validating inputs..."))
        prompt = validate_prompt(request.prompt)
        if not prompt:
            raise ValueError("Prompt is empty after validation")

        seed = request.seed if request.seed is not None else int(time.time()) & 0x7FFFFFFF
        seed = validate_int(seed, "seed", 0, 2**31 - 1)

        width, height = validate_resolution(request.width, request.height)

        # Output path
        settings = get_settings()
        if request.output_filename:
            from .security import validate_filename
            validate_filename(request.output_filename)
            output_path = settings.output_dir / request.output_filename
        else:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = settings.output_dir / f"qf_{timestamp}_{seed}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Validate that model_dir and transformer are within cache (path traversal protection)
        validate_path(str(local_paths["model_dir"]), base_dir=settings.models_dir, must_exist=True)
        validate_path(str(local_paths["transformer"]), base_dir=settings.models_dir, must_exist=True)

        # ----- Phase 4: Build subprocess args -----
        cli_path = engine_cli_path()
        vram_budget = self._compute_vram_budget(gpu_info, model)

        args = [
            str(cli_path),
            "--model-dir", str(local_paths["model_dir"]),
            "--transformer", str(local_paths["transformer"]),
            "--model-backend", request.backend,
            "--prompt", prompt,
            "--output", str(output_path),
            "--steps", str(request.steps),
            "--seed", str(seed),
            "--width", str(width),
            "--height", str(height),
        ]

        if settings.auto_optimize:
            args.append("--auto-optimize")

        if vram_budget > 0:
            args.extend(["--vram-budget", str(vram_budget)])

        if local_paths.get("mod_weights"):
            args.extend(["--mod-weights", str(local_paths["mod_weights"])])

        if local_paths.get("precision_config"):
            args.extend(["--precision-config", str(local_paths["precision_config"])])

        if request.init_image_path:
            resolved_init = validate_path(str(request.init_image_path), base_dir=settings.output_dir, must_exist=True)
            from .security import validate_float
            strength = validate_float(request.init_image_strength, "init_image_strength", min_val=0.01, max_val=1.0)
            args.extend([
                "--ref-image", str(resolved_init),
                "--init-img-strength", str(strength),
            ])

        # QuantFunc API key — user key from Settings takes precedence.
        # Without it, key-protected models fail at engine checkObfuscation.
        # If not set here, the engine falls back to bin/<os>/config.json itself.
        if settings.quantfunc_api_key and settings.quantfunc_api_key.strip():
            args.extend(["--apikey", settings.quantfunc_api_key.strip()])

        # LoRA stacking
        for lora_path, lora_strength in zip(request.lora_paths, request.lora_strengths):
            validate_path(str(lora_path), base_dir=settings.models_dir, must_exist=True)
            args.extend(["--lora", f"{lora_path}:{lora_strength}"])

        # Final safety check
        args = safe_subprocess_args(args)

        # ----- Phase 5: Spawn subprocess -----
        emit(ProgressEvent(
            status=JobStatus.GENERATING,
            progress=0.0,
            message=f"Starting inference (seed={seed})",
        ))

        start_time = time.time()

        try:
            # Build environment for subprocess: take current env, prepend CUDA paths
            # so the engine binary can find cublas64_13.dll, cudnn64_9.dll, etc.
            # This is CRITICAL on Windows — the engine binary's DLL loader looks up
            # DLLs by walking PATH in order. Without explicit PATH, CUDA Toolkit
            # install location is invisible to the subprocess.
            import os as _os
            from .engine import get_cuda_search_paths

            env = _os.environ.copy()
            cuda_paths = get_cuda_search_paths()
            existing_path = env.get("PATH", "")
            # Prepend CUDA dirs so they take precedence
            new_path_parts = [str(p) for p in cuda_paths if p.exists()]
            if existing_path:
                new_path_parts.append(existing_path)
            env["PATH"] = _os.pathsep.join(new_path_parts)

            print(f"[worker] Launching QuantFunc engine: width={width}, height={height}, steps={request.steps}, backend={request.backend}")
            print(f"[worker] CLI args: {' '.join(args)}")
            print(f"[worker] subprocess PATH (first 3): {new_path_parts[:3]}")
            print(f"[worker] subprocess cwd: {settings.engine_dir}")

            self._current_process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                bufsize=1,  # Line-buffered
                encoding="utf-8",
                errors="replace",
                cwd=str(settings.engine_dir),  # Engine may need its dir as CWD
                env=env,  # Inject CUDA paths
            )
        except FileNotFoundError as e:
            raise RuntimeError(f"Failed to start engine CLI at {cli_path}: {e}")

        # ----- Phase 6: Read output line by line -----
        full_output_lines = []  # Collect all lines for error reporting
        engine_reported_complete = False
        try:
            for line in self._current_process.stdout:
                if self._cancel_flag.is_set():
                    self._current_process.terminate()
                    emit(ProgressEvent(status=JobStatus.CANCELLED, message="Cancelled by user"))
                    raise RuntimeError("Generation cancelled by user")

                full_output_lines.append(line.rstrip())

                # Detect key-protection failure early and explain the fix
                if _KEY_PROTECTED_RE.search(line):
                    self._current_process.terminate()
                    raise RuntimeError(
                        f"Engine error (key-protected model): {line.strip()}\n"
                        + _KEY_PROTECTED_HELP
                    )

                event = parse_engine_output(line)
                if event:
                    emit(event)
                    if event.status == JobStatus.ERROR:
                        raise RuntimeError(f"Engine error: {event.error}")
                    if event.status == JobStatus.COMPLETE and event.image_path:
                        engine_reported_complete = True
                        output_path = Path(event.image_path)

        finally:
            if self._current_process:
                # Wait for process to finish, then check exit code
                self._current_process.wait(timeout=30)
                return_code = self._current_process.returncode
                self._current_process = None
            else:
                return_code = -1  # Unknown

        # ----- Phase 6.5: Check process exit code -----
        # If the engine crashed (e.g., missing DLL), return_code will be non-zero
        # and no output file will exist. Show the ACTUAL error, not "file not found".
        if return_code != 0 and not engine_reported_complete:
            # Collect last 20 lines of output for diagnosis
            tail = "\n".join(full_output_lines[-20:]) if full_output_lines else "(no output)"
            raise RuntimeError(
                f"Engine process exited with code {return_code} (crashed). "
                f"This usually means a DLL is missing (e.g., cublas64_13.dll, cudnn64_9.dll). "
                f"Go to Settings → Download CUDA deps to install CUDA runtime libraries.\n"
                f"--- Engine output (last 20 lines) ---\n{tail}"
            )

        # ----- Phase 7: Verify output -----
        if not output_path.exists():
            tail = "\n".join(full_output_lines[-20:]) if full_output_lines else "(no output)"
            raise RuntimeError(
                f"Engine reported completion but output file not found: {output_path}\n"
                f"--- Engine output (last 20 lines) ---\n{tail}"
            )

        elapsed = time.time() - start_time

        emit(ProgressEvent(
            status=JobStatus.SAVING,
            message=f"Saving to {output_path.name}...",
        ))

        # Copy to output dir if engine wrote elsewhere
        final_path = output_path
        if output_path.parent != settings.output_dir:
            from .cache import CacheManager
            cm = CacheManager()
            final_path = cm.save_output(output_path)

        # Write JSON sidecar with generation metadata
        json_sidecar = final_path.with_suffix(".json")
        try:
            meta = {
                "filename": final_path.name,
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "model_id": request.model_id,
                "seed": seed,
                "width": width,
                "height": height,
                "steps": request.steps,
                "backend": request.backend,
                "generation_time_sec": round(elapsed, 2),
                "timestamp": int(time.time()),
            }
            json_sidecar.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as meta_err:
            print(f"[worker] Failed to write sidecar metadata {json_sidecar}: {meta_err}")

        emit(ProgressEvent(
            status=JobStatus.COMPLETE,
            progress=1.0,
            image_path=str(final_path),
            message=f"Done in {elapsed:.1f}s",
        ))

        return GenerateResult(
            image_path=final_path,
            seed_used=seed,
            generation_time_sec=elapsed,
            model_id=request.model_id,
            backend=request.backend,
            steps=request.steps,
            prompt=prompt,
        )

    def cancel(self) -> None:
        """Cancel the current generation job."""
        self._cancel_flag.set()
        if self._current_process:
            try:
                self._current_process.terminate()
            except Exception:
                pass

    def _compute_vram_budget(self, gpu_info: GPUInfo, model: ModelSpec) -> int:
        """Compute --vram-budget value in GB based on settings + GPU."""
        settings = get_settings()
        mode = settings.vram_mode

        if mode == VRAMMode.AUTO or gpu_info.vram_gb == 0:
            return 0  # No budget — let engine decide
        elif mode == VRAMMode.CONSERVATIVE:
            return int(gpu_info.vram_gb * 0.5)
        elif mode == VRAMMode.BALANCED:
            return int(gpu_info.vram_gb * 0.75)
        elif mode == VRAMMode.MAXIMUM:
            return int(gpu_info.vram_gb)
        return 0


# ---------- Singleton worker ----------
_worker_instance: Optional[Worker] = None


def get_worker() -> Worker:
    """Get the global Worker singleton."""
    global _worker_instance
    if _worker_instance is None:
        _worker_instance = Worker()
    return _worker_instance
