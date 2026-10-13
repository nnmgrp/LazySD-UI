"""
Integration for SDNQ (SD.Next Quantization) engine with Diffusers.

Provides native PyTorch + Triton/CUDA inference for SDNQ-quantized models
(such as OzzyGT/Qwen_Image_2_1_Turbo_sdnq_dynamic_8bit) using INT8/INT4/FP8
weights with Hadamard rotation and SVD Quant, CPU offload, and VAE tiling.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from .gpu import GPUInfo
from .registry.loader import ModelSpec
from .security import (
    validate_filename,
    validate_int,
    validate_prompt,
    validate_resolution,
)
from .settings import get_settings

logger = logging.getLogger("sdnq")

_ENV_CACHE: Optional[dict] = None
_ENV_LOCK = threading.Lock()

SDNQ_ZIP_URL = "https://github.com/Disty0/sdnq/archive/refs/heads/master.zip"

# Progress regexes for subprocess stdout
_STATUS_RE = re.compile(r"^\[status\]\s*(.+)", re.IGNORECASE)
_STEP_RE = re.compile(r"\[step\s+(\d+)/(\d+)\]", re.IGNORECASE)
_COMPLETE_RE = re.compile(r"^\[complete\]\s*(.+)", re.IGNORECASE)
_ERROR_RE = re.compile(r"^\[error\]\s*(.+)", re.IGNORECASE)


def get_local_sdnq_dir() -> Path:
    """Directory inside engine_dir where standalone sdnq source can be stored."""
    return get_settings().engine_dir / "sdnq"


def _candidate_python_executables() -> list[Path]:
    """Return ordered list of candidate Python executables that may have CUDA torch + diffusers."""
    candidates: list[Path] = []
    settings = get_settings()
    custom_py = getattr(settings, "sdnq_python_path", None)
    if custom_py:
        candidates.append(Path(custom_py))

    # Check common SD.Next / local venv locations across drives
    project_root = Path(__file__).parent.parent.resolve()
    drive_roots = []
    if project_root.drive:
        drive_roots.append(project_root.drive + "\\")
    for d in ("G:\\", "C:\\", "D:\\", "E:\\", "F:\\"):
        if d not in drive_roots:
            drive_roots.append(d)

    for drv in drive_roots:
        candidates.append(Path(drv) / "SDNext" / "sdnext" / "venv" / "Scripts" / "python.exe")
        candidates.append(Path(drv) / "SDNext" / "venv" / "Scripts" / "python.exe")

    candidates.append(project_root / "venv" / "Scripts" / "python.exe")
    candidates.append(project_root / ".venv" / "Scripts" / "python.exe")
    candidates.append(Path(sys.executable))

    seen = set()
    unique: list[Path] = []
    for c in candidates:
        try:
            norm = str(c).lower()
            if norm not in seen and c.exists():
                seen.add(norm)
                unique.append(c)
        except Exception:
            pass
    return unique


def _candidate_sdnq_src_dirs() -> list[Path]:
    """Return candidate directories containing the `sdnq` package."""
    dirs: list[Path] = []
    local_sdnq = get_local_sdnq_dir()
    for sub in (local_sdnq / "src", local_sdnq / "sdnq-master" / "src", local_sdnq):
        if (sub / "sdnq" / "__init__.py").exists():
            dirs.append(sub)

    project_root = Path(__file__).parent.parent.resolve()
    drive_roots = []
    if project_root.drive:
        drive_roots.append(project_root.drive + "\\")
    for d in ("G:\\", "C:\\", "D:\\", "E:\\", "F:\\"):
        if d not in drive_roots:
            drive_roots.append(d)

    for drv in drive_roots:
        for cand in (
            Path(drv) / "SDNext" / "sdnext" / "extensions-builtin" / "sdnq" / "src",
            Path(drv) / "SDNext" / "extensions-builtin" / "sdnq" / "src",
        ):
            if (cand / "sdnq" / "__init__.py").exists():
                dirs.append(cand)

    return dirs


def _read_sdnq_version(sdnq_src: Optional[str]) -> str:
    """Read SDNQ version from pyproject.toml or __init__.py."""
    if not sdnq_src:
        return ""
    src_path = Path(sdnq_src)
    for cand in (src_path.parent / "pyproject.toml", src_path / "pyproject.toml"):
        if cand.exists():
            try:
                m = re.search(r'version\s*=\s*["\']([^"\']+)["\']', cand.read_text(encoding="utf-8", errors="ignore"))
                if m:
                    return m.group(1)
            except Exception:
                pass
    init_py = src_path / "sdnq" / "__init__.py"
    if init_py.exists():
        try:
            m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', init_py.read_text(encoding="utf-8", errors="ignore"))
            if m:
                return m.group(1)
        except Exception:
            pass
        return "installed"
    return ""


def _inspect_python_site_packages(py_exe: Path, default_src: Optional[str]) -> Optional[dict]:
    """Fast filesystem inspection of a Python environment (<1ms, no subprocess cold import)."""
    # Resolve site-packages relative to python.exe (Windows venv: ../Lib/site-packages, System: Lib/site-packages)
    site_dirs = [
        py_exe.parent.parent / "Lib" / "site-packages",
        py_exe.parent / "Lib" / "site-packages",
    ]
    for sp in site_dirs:
        torch_ver_file = sp / "torch" / "version.py"
        diffusers_init = sp / "diffusers" / "__init__.py"
        if not torch_ver_file.exists() or not diffusers_init.exists():
            continue
        try:
            tv_text = torch_ver_file.read_text(encoding="utf-8", errors="ignore")
            m_ver = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", tv_text)
            m_cuda = re.search(r"cuda(?:\s*:\s*[^=]+)?\s*=\s*['\"]([^'\"]+)['\"]", tv_text)
            torch_ver = m_ver.group(1) if m_ver else ""
            has_cuda = bool(m_cuda and m_cuda.group(1)) or ("+cu" in torch_ver)
            if not has_cuda:
                continue

            df_text = diffusers_init.read_text(encoding="utf-8", errors="ignore")
            m_df = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", df_text)
            diffusers_ver = m_df.group(1) if m_df else "installed"
            has_qwen21 = (sp / "diffusers" / "pipelines" / "qwenimage" / "pipeline_qwenimage_2_1.py").exists() or ("QwenImage21Pipeline" in df_text)

            sdnq_src_used = default_src
            if not sdnq_src_used and (sp / "sdnq" / "__init__.py").exists():
                sdnq_src_used = str(sp)
            sdnq_ver = _read_sdnq_version(sdnq_src_used)

            if has_qwen21 and sdnq_ver:
                return {
                    "installed": True,
                    "python_exe": str(py_exe),
                    "sdnq_src": sdnq_src_used,
                    "version": f"v{sdnq_ver} (torch {torch_ver})",
                    "sdnq_version": sdnq_ver,
                    "torch_version": torch_ver,
                    "diffusers_version": diffusers_ver,
                }
        except Exception:
            pass
    return None


def find_sdnq_env(force_refresh: bool = False) -> dict:
    """Locate a Python interpreter with CUDA PyTorch, Diffusers (QwenImage21Pipeline), and SDNQ."""
    global _ENV_CACHE
    with _ENV_LOCK:
        if _ENV_CACHE is not None and not force_refresh:
            return _ENV_CACHE

        sdnq_srcs = _candidate_sdnq_src_dirs()
        default_src = str(sdnq_srcs[0]) if sdnq_srcs else None

        # 1. Fast disk check across candidate Python environments (<1ms)
        for py_exe in _candidate_python_executables():
            fast_info = _inspect_python_site_packages(py_exe, default_src)
            if fast_info and fast_info.get("installed"):
                _ENV_CACHE = fast_info
                return _ENV_CACHE

        # 2. Subprocess fallback if non-standard layout
        probe_code = (
            "import json, sys\n"
            "info = {'cuda': False, 'torch': '', 'diffusers': '', 'sdnq': '', 'qwen21': False}\n"
            "try:\n"
            "    import torch\n"
            "    info['torch'] = torch.__version__\n"
            "    info['cuda'] = bool(torch.cuda.is_available())\n"
            "except Exception:\n"
            "    pass\n"
            "try:\n"
            "    import diffusers\n"
            "    info['diffusers'] = diffusers.__version__\n"
            "    from diffusers import QwenImage21Pipeline\n"
            "    info['qwen21'] = True\n"
            "except Exception:\n"
            "    pass\n"
            "try:\n"
            "    import sdnq\n"
            "    info['sdnq'] = getattr(sdnq, '__version__', 'installed')\n"
            "except Exception:\n"
            "    pass\n"
            "print(json.dumps(info))\n"
        )

        best_fallback: Optional[dict] = None

        for py_exe in _candidate_python_executables():
            env = os.environ.copy()
            if default_src:
                existing_pp = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = f"{default_src}{os.pathsep}{existing_pp}" if existing_pp else default_src
            try:
                proc = subprocess.run(
                    [str(py_exe), "-c", probe_code],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env=env,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    last_line = proc.stdout.strip().splitlines()[-1]
                    data = json.loads(last_line)
                    if data.get("cuda") and data.get("qwen21") and data.get("sdnq"):
                        _ENV_CACHE = {
                            "installed": True,
                            "python_exe": str(py_exe),
                            "sdnq_src": default_src,
                            "version": f"v{data['sdnq']} (torch {data['torch']})",
                            "sdnq_version": data["sdnq"],
                            "torch_version": data["torch"],
                            "diffusers_version": data["diffusers"],
                        }
                        return _ENV_CACHE
                    elif data.get("cuda") and best_fallback is None:
                        best_fallback = {
                            "installed": bool(data.get("sdnq")),
                            "python_exe": str(py_exe),
                            "sdnq_src": default_src,
                            "version": f"v{data['sdnq']}" if data.get("sdnq") else "missing sdnq",
                            "sdnq_version": data.get("sdnq", ""),
                            "torch_version": data.get("torch", ""),
                            "diffusers_version": data.get("diffusers", ""),
                        }
            except Exception as e:
                logger.debug("SDNQ probe failed for %s: %s", py_exe, e)

        if best_fallback:
            _ENV_CACHE = best_fallback
        else:
            _ENV_CACHE = {
                "installed": False,
                "python_exe": sys.executable,
                "sdnq_src": default_src,
                "version": "not installed",
                "sdnq_version": "",
                "torch_version": "",
                "diffusers_version": "",
            }
        return _ENV_CACHE


def is_sdnq_installed() -> bool:
    return bool(find_sdnq_env().get("installed"))


def get_sdnq_version() -> str:
    env = find_sdnq_env()
    if not env.get("installed"):
        return "not installed"
    return env.get("version", "installed")


def get_sdnq_path() -> str:
    env = find_sdnq_env()
    py_exe = env.get("python_exe") or sys.executable
    sdnq_src = env.get("sdnq_src")
    return f"{py_exe} ({sdnq_src})" if sdnq_src else str(py_exe)


def install_sdnq(force: bool = False) -> Path:
    """Ensure SDNQ package is available (either via existing installation or by downloading from GitHub)."""
    env = find_sdnq_env(force_refresh=True)
    if env.get("installed") and not force:
        return Path(env.get("sdnq_src") or env.get("python_exe"))

    target_dir = get_local_sdnq_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    temp_zip = target_dir / "sdnq_master.zip.tmp"

    import zipfile
    from .engine import _download_file_with_progress

    try:
        logger.info("Downloading SDNQ from %s", SDNQ_ZIP_URL)
        _download_file_with_progress(SDNQ_ZIP_URL, temp_zip)
        with zipfile.ZipFile(temp_zip, "r") as z:
            z.extractall(target_dir)
    finally:
        if temp_zip.exists():
            try:
                temp_zip.unlink()
            except OSError:
                pass

    env = find_sdnq_env(force_refresh=True)
    return Path(env.get("sdnq_src") or target_dir)


def convert_qwen21_vae_state_dict(sd: dict) -> dict:
    """Convert single-file ComfyUI/Wan format Qwen-Image-2.1 VAE state_dict to Diffusers AutoencoderKLQwenImage21 format.

    If `sd` is already in Diffusers format (e.g. contains `encoder.conv_in.weight`), returns it unchanged.
    """
    if "encoder.conv_in.weight" in sd and "decoder.conv_in.weight" in sd:
        return sd

    out = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("conv1."):
            nk = "quant_conv." + nk[6:]
        elif nk.startswith("conv2."):
            nk = "post_quant_conv." + nk[6:]
        nk = nk.replace("encoder.conv1.", "encoder.conv_in.").replace("decoder.conv1.", "decoder.conv_in.")
        nk = nk.replace("encoder.head.0.", "encoder.norm_out.").replace("encoder.head.2.", "encoder.conv_out.")
        nk = nk.replace("decoder.head.0.", "decoder.norm_out.").replace("decoder.head.2.", "decoder.conv_out.")
        nk = (
            nk.replace(".middle.0.", ".mid_block.resnets.0.")
            .replace(".middle.1.", ".mid_block.attentions.0.")
            .replace(".middle.2.", ".mid_block.resnets.1.")
        )
        nk = re.sub(r"encoder\.downsamples\.(\d+)\.downsamples\.([01])\.", r"encoder.down_blocks.\1.resnets.\2.", nk)
        nk = re.sub(r"encoder\.downsamples\.(\d+)\.downsamples\.2\.", r"encoder.down_blocks.\1.downsampler.", nk)
        nk = re.sub(r"decoder\.upsamples\.(\d+)\.upsamples\.([012])\.", r"decoder.up_blocks.\1.resnets.\2.", nk)
        nk = re.sub(r"decoder\.upsamples\.(\d+)\.upsamples\.3\.", r"decoder.up_blocks.\1.upsampler.", nk)
        nk = (
            nk.replace(".residual.0.", ".norm1.")
            .replace(".residual.2.", ".conv1.")
            .replace(".residual.3.", ".norm2.")
            .replace(".residual.6.", ".conv2.")
            .replace(".shortcut.", ".conv_shortcut.")
        )
        if hasattr(v, "ndim") and v.ndim == 5 and v.shape[2] == 1:
            v = v.squeeze(2)
        out[nk] = v
    return out


class SDNQWorker:
    """Executes diffusion inference for SDNQ models via CUDA PyTorch + Diffusers subprocess."""

    def __init__(self):
        self._cancel_flag = threading.Event()
        self._current_process: Optional[subprocess.Popen] = None

    def cancel(self):
        """Cancel current running SDNQ process."""
        self._cancel_flag.set()
        if self._current_process:
            try:
                self._current_process.terminate()
            except Exception:
                pass

    def generate(
        self,
        request,
        model: ModelSpec,
        gpu_info: Optional[GPUInfo],
        progress_callback: Optional[Callable] = None,
    ):
        from .worker import GenerateResult, JobStatus, ProgressEvent

        self._cancel_flag.clear()

        def emit(event: ProgressEvent):
            if progress_callback:
                progress_callback(event)

        env_info = find_sdnq_env()
        if not env_info.get("installed"):
            raise RuntimeError(
                "SDNQ runtime environment is not ready (requires CUDA PyTorch + Diffusers QwenImage21Pipeline + SDNQ). "
                "Check Settings → Engines & Runtimes."
            )

        python_exe = env_info["python_exe"]
        sdnq_src = env_info.get("sdnq_src")

        # 1. Validate inputs
        emit(ProgressEvent(status=JobStatus.INITIALIZING, message="Validating inputs for SDNQ..."))
        prompt = validate_prompt(request.prompt)
        if not prompt:
            raise ValueError("Prompt is empty after validation")

        seed = request.seed if request.seed is not None and request.seed >= 0 else int(time.time()) & 0x7FFFFFFF
        seed = validate_int(seed, "seed", 0, 2**31 - 1)
        width, height = validate_resolution(request.width, request.height)

        # 2. Setup output path
        settings = get_settings()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_filename = request.output_filename or f"sdnq_{timestamp}_{seed}.png"
        validate_filename(out_filename)
        output_path = settings.output_dir / out_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 3. Verify model cache directory
        from .cache import CacheManager
        cache = CacheManager()
        primary_repo = model.sources.primary.repo_id if model.sources else ""
        repo_dir = cache.repo_dir_for(primary_repo) if primary_repo else settings.models_dir

        required_check_files = [
            repo_dir / "model_index.json",
            repo_dir / "transformer" / "diffusion_pytorch_model.safetensors",
            repo_dir / "text_encoder" / "model.safetensors",
            repo_dir / "vae" / "diffusion_pytorch_model.safetensors",
        ]
        missing = [str(p.relative_to(repo_dir)) for p in required_check_files if not p.exists()]
        if missing:
            raise RuntimeError(
                f"SDNQ model files are not yet downloaded ({', '.join(missing)}). "
                "Click '⬇ Download' to download the model first."
            )

        # 4. Resolve optional Texture-Fix or custom VAE
        custom_vae_file: Optional[Path] = None
        vae_type = getattr(request, "vae", None) or getattr(settings, "qwen_vae_type", "texture_fix")
        custom_vae_str = getattr(request, "custom_vae_path", None)
        if custom_vae_str:
            custom_vae_p = Path(custom_vae_str)
            if not custom_vae_p.is_file():
                raise RuntimeError(f"Custom VAE file not found: {custom_vae_p}")
            custom_vae_file = custom_vae_p
        elif model.family == "qwen" or model.id.startswith("qwen"):
            if vae_type == "texture_fix":
                vae_rel = "texture_fix_vae_for_qwen_image_2.1_bf16.safetensors"
                candidate_vae = next(
                    (
                        p
                        for p in [
                            repo_dir / vae_rel,
                            settings.models_dir / "Qwen-Image-2.1-4bit" / vae_rel,
                            settings.models_dir / "Qwen-Image-2.1-4bit" / "vae" / vae_rel,
                            settings.models_dir / "Qwen-Image-2.1-GGUF" / vae_rel,
                            settings.models_dir / "vae" / vae_rel,
                        ]
                        if p.exists()
                    ),
                    repo_dir / vae_rel,
                )
                if not candidate_vae.exists():
                    emit(
                        ProgressEvent(
                            status=JobStatus.DOWNLOADING,
                            progress=0.05,
                            message=f"Downloading Texture-Fix VAE ({vae_rel}, 644 MB)...",
                        )
                    )
                    from .sources import HuggingFaceSource

                    hf = HuggingFaceSource(token=settings.huggingface_token)
                    hf.download_file(
                        repo_id="madebyollin/texture-fix-vae-for-qwen-image-2.1",
                        file_path=vae_rel,
                        local_dir=repo_dir,
                    )
                    candidate_vae = repo_dir / vae_rel
                if candidate_vae.exists():
                    custom_vae_file = candidate_vae

        # 5. Resolve optional Img2Img reference image
        ref_image_path: Optional[str] = None
        init_image_strength = float(getattr(request, "init_image_strength", 0.7) or 0.7)
        if getattr(request, "init_image_path", None):
            ref_p = Path(request.init_image_path)
            if not ref_p.is_absolute():
                ref_p = settings.output_dir / ref_p
            if ref_p.exists():
                ref_image_path = str(ref_p)

        # 6. Prepare job config JSON for subprocess
        steps = int(request.steps or 8)
        cfg = float(request.cfg if request.cfg is not None else 1.0)
        noise = float(getattr(request, "noise", 0.0) or 0.0)
        sampler = getattr(request, "sampler", None) or ("euler_a" if noise > 0.0 else "euler")
        neg_prompt = (getattr(request, "negative_prompt", "") or "").strip()

        job_payload = {
            "model_dir": str(repo_dir),
            "prompt": prompt,
            "negative_prompt": neg_prompt,
            "steps": steps,
            "cfg": cfg,
            "noise": noise,
            "sampler": sampler,
            "seed": seed,
            "width": width,
            "height": height,
            "output_path": str(output_path),
            "vae_safetensors_path": str(custom_vae_file) if custom_vae_file else None,
            "init_image_path": ref_image_path,
            "init_image_strength": init_image_strength,
            "sdnq_src": str(sdnq_src) if sdnq_src else None,
        }

        project_root = Path(__file__).parent.parent.resolve()
        env = os.environ.copy()
        pp_parts = [str(project_root)]
        if sdnq_src:
            pp_parts.insert(0, str(sdnq_src))
        if env.get("PYTHONPATH"):
            pp_parts.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pp_parts)
        env["PYTHONIOENCODING"] = "utf-8"

        fd, job_json_path = tempfile.mkstemp(prefix="sdnq_job_", suffix=".json")
        os.close(fd)
        Path(job_json_path).write_text(json.dumps(job_payload, ensure_ascii=False), encoding="utf-8")

        cmd = [str(python_exe), "-u", "-m", "backend.sdnq_engine", "--run-job", job_json_path]
        logger.info(
            "[sdnq] Launching SDNQ worker: %dx%d, steps=%d, cfg=%.1f, sampler=%s, noise=%.2f, seed=%d",
            width, height, steps, cfg, sampler, noise, seed,
        )

        start_time = time.time()
        emit(
            ProgressEvent(
                status=JobStatus.LOADING,
                progress=0.05,
                message="Starting SDNQ pipeline (CPU offload + VAE tiling)...",
            )
        )

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(project_root),
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._current_process = proc
        captured_lines: list[str] = []

        try:
            for line in proc.stdout:
                line_str = line.strip()
                if not line_str:
                    continue
                captured_lines.append(line_str)
                logger.debug("[sdnq] %s", line_str)

                if self._cancel_flag.is_set():
                    proc.terminate()
                    raise RuntimeError("Generation cancelled by user")

                m_step = _STEP_RE.search(line_str)
                if m_step:
                    cur_s = int(m_step.group(1))
                    tot_s = int(m_step.group(2))
                    pct = min(0.95, max(0.10, cur_s / max(tot_s, 1)))
                    emit(
                        ProgressEvent(
                            status=JobStatus.GENERATING,
                            progress=pct,
                            step=cur_s,
                            total_steps=tot_s,
                            message=f"Denoising step {cur_s}/{tot_s} ({int(pct * 100)}%)",
                        )
                    )
                    continue

                m_status = _STATUS_RE.match(line_str)
                if m_status:
                    emit(
                        ProgressEvent(
                            status=JobStatus.LOADING,
                            progress=0.08,
                            message=m_status.group(1).strip(),
                        )
                    )
                    continue

                m_err = _ERROR_RE.match(line_str)
                if m_err:
                    raise RuntimeError(f"SDNQ error: {m_err.group(1).strip()}")

            proc.wait()
        finally:
            self._current_process = None
            try:
                Path(job_json_path).unlink(missing_ok=True)
            except Exception:
                pass

        if proc.returncode != 0:
            tail = "\n".join(captured_lines[-60:]) if captured_lines else "(no output)"
            raise RuntimeError(
                f"SDNQ process exited with code {proc.returncode}.\n"
                f"--- SDNQ output (last 60 lines) ---\n{tail}"
            )

        if not output_path.exists():
            tail = "\n".join(captured_lines[-60:]) if captured_lines else "(no output)"
            raise RuntimeError(
                f"SDNQ finished but output file was not created: {output_path}\n"
                f"--- SDNQ output (last 60 lines) ---\n{tail}"
            )

        elapsed = time.time() - start_time

        # Save JSON sidecar metadata for History tab
        sidecar_path = output_path.with_suffix(".json")
        try:
            meta = {
                "filename": output_path.name,
                "prompt": prompt,
                "negative_prompt": neg_prompt,
                "model_id": request.model_id,
                "variant_id": getattr(request, "variant_id", None),
                "seed": seed,
                "width": width,
                "height": height,
                "steps": steps,
                "cfg": cfg,
                "noise": noise,
                "sampler": sampler,
                "vae": vae_type,
                "backend": "sdnq",
                "generation_time_sec": round(elapsed, 2),
                "timestamp": int(time.time()),
            }
            sidecar_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to write sidecar metadata: %s", e)

        emit(
            ProgressEvent(
                status=JobStatus.COMPLETE,
                progress=1.0,
                image_path=str(output_path),
                message=f"Done in {elapsed:.1f}s",
            )
        )

        return GenerateResult(
            image_path=output_path,
            seed_used=seed,
            generation_time_sec=elapsed,
            model_id=request.model_id,
            backend="sdnq",
            steps=steps,
            prompt=prompt,
        )


def _run_job_cli(job_json_path: str) -> int:
    """Entry point executed inside the CUDA Python subprocess."""
    try:
        import gc

        payload = json.loads(Path(job_json_path).read_text(encoding="utf-8"))
        model_dir = Path(payload["model_dir"])
        prompt = payload["prompt"]
        negative_prompt = payload.get("negative_prompt") or None
        steps = int(payload.get("steps", 8))
        cfg = float(payload.get("cfg", 1.0))
        noise = float(payload.get("noise", 0.0))
        sampler = payload.get("sampler") or "euler"
        seed = int(payload.get("seed", 42))
        width = int(payload.get("width", 1024))
        height = int(payload.get("height", 1024))
        output_path = Path(payload["output_path"])
        vae_safetensors_path = payload.get("vae_safetensors_path")
        init_image_path = payload.get("init_image_path")
        init_image_strength = float(payload.get("init_image_strength", 0.7))
        sdnq_src = payload.get("sdnq_src")

        print("[status] Registering SDNQ quantization backend...", flush=True)
        os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
        os.environ["SDNQ_REGISTER_DIFFUSERS"] = "1"
        os.environ["SDNQ_REGISTER_TRANSFORMERS"] = "1"
        os.environ["DIFFUSERS_SDNQ_TRANSFORMERS"] = "1"
        os.environ["SDNQ_USE_OPENVINO_MM"] = "0"
        os.environ["SDNQ_USE_TORCH_COMPILE"] = "0"
        os.environ["SDNQ_USE_TRITON_MM"] = "0"
        os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"

        # Workaround for safetensors >= 0.8.0 on Windows where `backend="mmap"` triggers
        # 0xC0000005 Access Violation inside `torch/storage.py:__getitem__` on >4GB safetensors files.
        import safetensors
        import safetensors.torch as _st_torch

        _orig_safe_open = safetensors.safe_open

        def _pread_safe_open(filename, framework="pt", device="cpu", **kwargs):
            return _orig_safe_open(filename, framework=framework, device=device, backend="pread")

        safetensors.safe_open = _pread_safe_open
        _st_torch.safe_open = _pread_safe_open
        try:
            import safetensors._safetensors_rust as _st_rust
            _st_rust.safe_open = _pread_safe_open
        except Exception:
            pass

        _orig_load_file = _st_torch.load_file

        def _pread_load_file(filename, device="cpu", **kwargs):
            return _orig_load_file(filename, device=device, backend="pread")

        _st_torch.load_file = _pread_load_file

        import torch
        import diffusers.utils.import_utils as _du_imp
        import sdnq  # noqa: F401 — registers SDNQ quantization method in diffusers & transformers
        from sdnq.loader import load_sdnq_model, apply_sdnq_options_to_model

        # Ensure diffusers' internal metadata check recognizes sdnq even when loaded from PYTHONPATH
        _du_imp._sdnq_available = True
        _du_imp._sdnq_version = getattr(sdnq, "__version__", "0.2.8")
        try:
            import diffusers.quantizers.sdnq.sdnq_quantizer as _dsq
            _dsq.is_sdnq_available = lambda: True
            _dsq.is_sdnq_version = lambda op, ver: False
        except Exception:
            pass

        from diffusers import QwenImage21Pipeline
        from backend.flowmatch_samplers import (
            configure_flowmatch_scheduler,
            interpolate_sample_sigmas,
            prepare_qwen21_img2img_latents,
        )

        # Read model_index.json to check for Turbo sample_sigmas (and rename to _sample_sigmas
        # so Diffusers does not emit unexpected config attribute warnings).
        sample_sigmas = None
        midx_path = model_dir / "model_index.json"
        if midx_path.exists():
            try:
                midx = json.loads(midx_path.read_text(encoding="utf-8"))
                raw_sigmas = midx.get("sample_sigmas") or midx.get("_sample_sigmas")
                if isinstance(raw_sigmas, list):
                    sample_sigmas = [float(x) for x in raw_sigmas]
                if "sample_sigmas" in midx:
                    midx["_sample_sigmas"] = midx.pop("sample_sigmas")
                    midx_path.write_text(json.dumps(midx, indent=2), encoding="utf-8")
            except Exception:
                pass

        # 3-Stage Memory-Isolated Execution for 16GB VRAM:
        # Instead of loading text_encoder (10.03 GB) + transformer (7.19 GB) + vae (0.67 GB)
        # simultaneously and relying on accelerate's `enable_model_cpu_offload()` (which crashes
        # with 0xC0000005 Access Violation inside `text_encoder.to("cpu")` under heavy RAM commit),
        # we load each heavy stage sequentially onto CUDA and release it immediately after use.
        te_dir = model_dir / "text_encoder"
        tr_dir = model_dir / "transformer"
        staged_mode = (te_dir / "config.json").exists() and (tr_dir / "config.json").exists()

        print("[status] Loading Qwen-Image-2.1 Turbo SDNQ INT8 pipeline...", flush=True)
        pipe_kwargs = {"dtype": torch.bfloat16}
        if staged_mode:
            pipe_kwargs["text_encoder"] = None
            pipe_kwargs["transformer"] = None

        pipe = QwenImage21Pipeline.from_pretrained(
            str(model_dir),
            **pipe_kwargs,
        )

        if not staged_mode:
            try:
                if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
                    pipe.text_encoder = apply_sdnq_options_to_model(
                        pipe.text_encoder, dtype=torch.bfloat16, dequantize_fp32=False
                    )
                if hasattr(pipe, "transformer") and pipe.transformer is not None:
                    pipe.transformer = apply_sdnq_options_to_model(
                        pipe.transformer, dtype=torch.bfloat16, dequantize_fp32=False
                    )
                pipe.enable_model_cpu_offload()
            except Exception as opt_err:
                print(f"[status] SDNQ options note: {opt_err}", flush=True)

        if vae_safetensors_path:
            vae_p = Path(vae_safetensors_path)
            if not vae_p.is_file():
                raise FileNotFoundError(f"Custom VAE file not found: {vae_p}")
            print(f"[status] Loading custom/Texture-Fix VAE ({vae_p.name})...", flush=True)
            from safetensors.torch import load_file

            raw_sd = load_file(str(vae_p))
            conv_sd = convert_qwen21_vae_state_dict(raw_sd)
            pipe.vae.load_state_dict(conv_sd, strict=True)

        if hasattr(pipe, "vae") and pipe.vae is not None:
            pipe.vae.to(device="cuda" if staged_mode else "cpu", dtype=torch.bfloat16)
            # IMPORTANT: Do NOT enable default VAE tiling on Qwen-Image-2.1 for <= 1536x1536!
            # Diffusers' AutoencoderKLQwenImage21 inherited `tile_sample_min_width=256` and
            # `tile_sample_stride_width=192` from AutoencoderKLWan (8x VAE). Because Qwen-Image-2.1
            # uses a 16x spatial VAE, 256px is only a 16x16 latent tile with 12-latent (192px) stride,
            # causing severe vertical/horizontal magenta-green seam stripes every 192 pixels.
            # Since Stage 3 runs after text_encoder and transformer are freed from VRAM, full untiled
            # VAE decode takes <2 GB VRAM and produces zero seam lines.
            if width * height > 1536 * 1536 and hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling(
                    tile_sample_min_height=1024,
                    tile_sample_min_width=1024,
                    tile_sample_stride_height=768,
                    tile_sample_stride_width=768,
                )
            elif hasattr(pipe.vae, "disable_tiling"):
                pipe.vae.disable_tiling()

        # Configure FlowMatch sampler & stochastic noise (eta)
        sched_summary = configure_flowmatch_scheduler(
            pipe, sampler_name=sampler, noise=noise, seed=seed, sdnq_src=sdnq_src
        )
        print(f"[status] Active scheduler: {sched_summary}", flush=True)

        multiple_of = pipe.vae_scale_factor * 2
        width = (width // multiple_of) * multiple_of
        height = (height // multiple_of) * multiple_of

        cond_image = None
        if init_image_path and Path(init_image_path).exists():
            from PIL import Image

            cond_image = Image.open(init_image_path).convert("RGB")

        # Handle CFG & Negative Prompt interaction:
        # 1) If user typed a negative prompt while CFG == 1.0, boost effective CFG to 1.5 so QwenImage21Pipeline
        #    activates `do_true_cfg = true_cfg_scale > 1.0 and has_neg_prompt`.
        # 2) If user set CFG > 1.0 without a negative prompt, supply a blank space " " so unconditional pass runs.
        if negative_prompt and cfg <= 1.0:
            cfg = 1.5
            print("[status] Negative prompt provided with CFG=1.0 -> boosting effective CFG to 1.5...", flush=True)
        elif cfg > 1.0 and not negative_prompt:
            negative_prompt = " "

        generator = torch.Generator(device="cpu").manual_seed(seed)

        # Interpolate Turbo's shifted sample_sigmas to any user-requested step count
        if sample_sigmas:
            use_sigmas = interpolate_sample_sigmas(sample_sigmas, steps)
        else:
            use_sigmas = None

        initial_latents = None
        if cond_image is not None and init_image_strength < 0.99:
            base_sched_sigmas = use_sigmas or interpolate_sample_sigmas([], steps)
            print(
                f"[status] Preparing Img2Img latents (strength={init_image_strength:.2f})...",
                flush=True,
            )
            initial_latents, use_sigmas = prepare_qwen21_img2img_latents(
                pipe=pipe,
                cond_image=cond_image,
                width=width,
                height=height,
                full_sigmas=base_sched_sigmas,
                strength=init_image_strength,
                seed=seed,
                dtype=torch.bfloat16,
            )

        total_steps = len(use_sigmas) if use_sigmas else steps

        def on_step_end(pipe_inst, step_idx, timestep, callback_kwargs):
            print(f"[step {step_idx + 1}/{total_steps}]", flush=True)
            return callback_kwargs

        def _load_stage_component(comp_dir: Path, kind: str):
            """Load either an SDNQ-quantized component or a standard BF16/FP16 Diffusers component."""
            cfg_data = {}
            cfg_file = comp_dir / "config.json"
            if cfg_file.exists():
                try:
                    cfg_data = json.loads(cfg_file.read_text(encoding="utf-8"))
                except Exception:
                    pass
            has_qcfg = (comp_dir / "quantization_config.json").exists() or ("quantization_config" in cfg_data)
            if has_qcfg:
                return load_sdnq_model(
                    str(comp_dir),
                    dtype=torch.bfloat16,
                    dequantize_fp32=False,
                ).to("cuda")
            if kind == "text_encoder":
                from transformers import AutoModel, Qwen3VLForConditionalGeneration

                try:
                    return Qwen3VLForConditionalGeneration.from_pretrained(
                        str(comp_dir), dtype=torch.bfloat16
                    ).to("cuda")
                except Exception:
                    return AutoModel.from_pretrained(str(comp_dir), dtype=torch.bfloat16).to("cuda")
            from diffusers.models import QwenImage21Transformer2DModel

            return QwenImage21Transformer2DModel.from_pretrained(
                str(comp_dir), dtype=torch.bfloat16
            ).to("cuda")

        if staged_mode:
            print("[status] Stage 1/3: Loading Qwen3-VL text encoder to GPU...", flush=True)
            pipe.text_encoder = _load_stage_component(te_dir, "text_encoder")

            orig_encode_prompt = pipe.encode_prompt
            expected_encode_calls = 2 if (cfg > 1.0 and negative_prompt is not None) else 1
            encode_call_count = 0

            def _staged_encode_prompt(*args, **kwargs):
                nonlocal encode_call_count
                encode_call_count += 1
                out = orig_encode_prompt(*args, **kwargs)
                if encode_call_count >= expected_encode_calls:
                    print("[status] Releasing Qwen3-VL text encoder from VRAM...", flush=True)
                    pipe.text_encoder = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    print(
                        "[status] Stage 2/3: Loading Qwen-Image-2.1 transformer to GPU...",
                        flush=True,
                    )
                    pipe.transformer = _load_stage_component(tr_dir, "transformer")
                    print(
                        f"[status] Running denoising ({total_steps} steps, cfg={cfg:.1f}, {width}x{height})...",
                        flush=True,
                    )
                return out

            pipe.encode_prompt = _staged_encode_prompt

        call_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "true_cfg_scale": cfg,
            "width": width,
            "height": height,
            "num_inference_steps": total_steps,
            "generator": generator,
            "callback_on_step_end": on_step_end,
            "output_type": "latent" if staged_mode else "pil",
        }
        if use_sigmas is not None:
            call_kwargs["sigmas"] = use_sigmas
        if cond_image is not None:
            call_kwargs["image"] = cond_image
        if initial_latents is not None:
            call_kwargs["latents"] = initial_latents

        if not staged_mode:
            print(f"[status] Running denoising ({total_steps} steps, cfg={cfg:.1f}, {width}x{height})...", flush=True)

        result = pipe(**call_kwargs)

        if staged_mode:
            latents = result.images
            print("[status] Releasing transformer from VRAM...", flush=True)
            pipe.transformer = None
            gc.collect()
            torch.cuda.empty_cache()

            print("[status] Stage 3/3: Decoding latents with VAE...", flush=True)
            with torch.no_grad():
                latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
                latents = latents.to(device=pipe.vae.device, dtype=pipe.vae.dtype)
                latents_mean = (
                    torch.tensor(pipe.vae.config.latents_mean)
                    .view(1, pipe.vae.config.z_dim, 1, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(pipe.vae.config.latents_std)
                    .view(1, pipe.vae.config.z_dim, 1, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents = latents * latents_std + latents_mean
                decoded = pipe.vae.decode(latents, return_dict=False)[0][:, :, 0]
                image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
        else:
            image = result.images[0]

        print("[status] Saving generated image...", flush=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(str(output_path))
        print(f"[complete] {output_path}", flush=True)
        return 0
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"[error] {type(e).__name__}: {e}", flush=True)
        return 1


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--run-job":
        sys.exit(_run_job_cli(sys.argv[2]))
