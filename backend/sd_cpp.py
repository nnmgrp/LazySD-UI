"""
Integration for stable-diffusion.cpp (SD.CPP / sd-cli.exe).

Provides lightweight, standalone C++/CUDA inference for GGUF and safetensors
diffusion models (SDXL, SD 1.5, SD 3.5, Flux.1 GGUF) with quantized text encoders
(T5-XXL Q4, CLIP-L) without PyTorch dependencies.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .gpu import GPUInfo
from .registry.loader import ModelSpec

logger = logging.getLogger("sd_cpp")
from .security import (
    validate_filename,
    validate_int,
    validate_prompt,
    validate_resolution,
)
from .settings import get_settings


# Default SD.CPP binary directory
def get_sd_cpp_dir() -> Path:
    settings = get_settings()
    return settings.engine_dir / "sd_cpp"


def get_sd_cpp_cli_path() -> Path:
    return get_sd_cpp_dir() / "sd-cli.exe"


def is_sd_cpp_installed() -> bool:
    return get_sd_cpp_cli_path().exists()


def get_sd_cpp_version() -> str:
    """Return installed or known SD.CPP release version."""
    if not is_sd_cpp_installed():
        return "not installed"
    return "master-a4a9669 (cu12)"


# Progress matching regex for stable-diffusion.cpp
# Example outputs:
#   "| 3/20 [00:02<00:11,  1.45s/it]"
#   "generating image: 5/20"
#   "sampling: 45%"
_STEP_PROGRESS_RE = re.compile(r"(\d+)\s*/\s*(\d+)")
_PERCENT_PROGRESS_RE = re.compile(r"(\d+)\s*%")


class SDCppWorker:
    """Executes diffusion inference via sd-cli.exe."""

    def __init__(self):
        self._cancel_flag = threading.Event()
        self._current_process: Optional[subprocess.Popen] = None

    def cancel(self):
        """Cancel current running process."""
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
        """Run generation via sd-cli.exe.

        Args:
            request: GenerateRequest instance
            model: ModelSpec for the selected model
            gpu_info: GPUInfo
            progress_callback: Callable taking ProgressEvent

        Returns:
            GenerateResult instance
        """
        from .worker import GenerateResult, JobStatus, ProgressEvent

        self._cancel_flag.clear()

        def emit(event: ProgressEvent):
            if progress_callback:
                progress_callback(event)

        cli_path = get_sd_cpp_cli_path()
        if not cli_path.exists():
            raise RuntimeError(
                f"stable-diffusion.cpp binary not found at {cli_path}. "
                "Ensure bin/windows/sd_cpp/sd-cli.exe is present."
            )

        # 1. Validate inputs
        emit(ProgressEvent(status=JobStatus.INITIALIZING, message="Validating inputs for SD.CPP..."))
        prompt = validate_prompt(request.prompt)
        if not prompt:
            raise ValueError("Prompt is empty after validation")

        seed = request.seed if request.seed is not None and request.seed >= 0 else int(time.time()) & 0x7FFFFFFF
        seed = validate_int(seed, "seed", 0, 2**31 - 1)

        width, height = validate_resolution(request.width, request.height)

        # 2. Setup output path
        settings = get_settings()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_filename = request.output_filename or f"sd_{timestamp}_{seed}.png"
        validate_filename(out_filename)
        output_path = settings.output_dir / out_filename

        # 3. Resolve model paths
        from .cache import CacheManager
        cache = CacheManager()
        primary_repo = model.sources.primary.repo_id if model.sources else ""
        repo_dir = cache.repo_dir_for(primary_repo) if primary_repo else settings.models_dir

        cmd = [str(cli_path), "-M", "img_gen"]

        # Checkpoint vs standalone diffusion-model
        ckpt_rel = model.engine_args.get("checkpoint")
        if ckpt_rel:
            ckpt_path = repo_dir / ckpt_rel
            cmd.extend(["-m", str(ckpt_path)])
        else:
            # Standalone diffusion model (e.g. DiT GGUF)
            diff_rel = model.engine_args.get("diffusion_model")
            if diff_rel:
                diff_path = repo_dir / diff_rel
                cmd.extend(["--diffusion-model", str(diff_path)])

            # Text encoders (GGUF / safetensors)
            t5_rel = model.engine_args.get("t5xxl")
            if t5_rel:
                cmd.extend(["--t5xxl", str(repo_dir / t5_rel)])

            clip_rel = model.engine_args.get("clip_l")
            if clip_rel:
                cmd.extend(["--clip_l", str(repo_dir / clip_rel)])

            # VAE
            vae_rel = model.engine_args.get("vae")
            if vae_rel:
                cmd.extend(["--vae", str(repo_dir / vae_rel)])

        # Img2Img reference image
        if getattr(request, "init_image_path", None):
            ref_path = Path(request.init_image_path)
            if not ref_path.is_absolute():
                ref_path = settings.output_dir / ref_path
            if ref_path.exists():
                cmd.extend(["-i", str(ref_path)])
                strength = getattr(request, "init_image_strength", 0.75) or 0.75
                cmd.extend(["--strength", str(round(strength, 2))])

        # Prompt & Negative prompt
        cmd.extend(["-p", prompt])
        if getattr(request, "negative_prompt", None) and request.negative_prompt.strip():
            cmd.extend(["-n", request.negative_prompt.strip()])

        # Dimensions, steps, seed
        cmd.extend([
            "-W", str(width),
            "-H", str(height),
            "--steps", str(request.steps or 20),
            "-s", str(seed),
            "-o", str(output_path),
        ])

        # Guidance / CFG scale
        guidance = getattr(request, "guidance_scale", None)
        if model.engine_args.get("guidance_type") == "distilled" or model.family in ("flux", "sd3"):
            cfg_val = guidance if guidance is not None else 3.5
            cmd.extend(["--guidance", str(cfg_val)])
        else:
            cfg_val = guidance if guidance is not None else 7.0
            cmd.extend(["--cfg-scale", str(cfg_val)])

        # Sampling method
        if getattr(request, "sampler", None):
            cmd.extend(["--sampling-method", request.sampler])

        # Logging
        logger.info(f"[sd_cpp] Launching SD.CPP: {width}x{height}, steps={request.steps}, seed={seed}")
        logger.debug(f"[sd_cpp] CLI args: {' '.join(cmd)}")

        # 4. Environment PATH
        env = os.environ.copy()
        sd_cpp_dir = get_sd_cpp_dir()
        cuda_dirs = [
            r"G:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
            r"G:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64",
            str(settings.engine_dir),
            str(sd_cpp_dir),
        ]
        valid_cuda = [d for d in cuda_dirs if os.path.exists(d)]
        env["PATH"] = ";".join(valid_cuda + [env.get("PATH", "")])

        start_time = time.time()
        emit(ProgressEvent(
            status=JobStatus.GENERATING,
            progress=0.05,
            message="Starting SD.CPP inference...",
        ))

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(sd_cpp_dir),
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._current_process = proc

        total_steps = request.steps or 20
        captured_lines = []

        try:
            for line in proc.stdout:
                line_str = line.strip()
                if not line_str:
                    continue
                captured_lines.append(line_str)
                logger.debug(f"[sd_cpp] {line_str}")

                # Check cancel
                if self._cancel_flag.is_set():
                    proc.terminate()
                    raise RuntimeError("Generation cancelled by user")

                # Parse progress
                step_match = _STEP_PROGRESS_RE.search(line_str)
                if step_match:
                    try:
                        cur_step = int(step_match.group(1))
                        tot_steps = int(step_match.group(2))
                        pct = min(0.95, max(0.05, cur_step / tot_steps))
                        emit(ProgressEvent(
                            status=JobStatus.GENERATING,
                            progress=pct,
                            step=cur_step,
                            total_steps=tot_steps,
                            message=f"Sampling step {cur_step}/{tot_steps} ({int(pct*100)}%)",
                        ))
                    except ValueError:
                        pass
                else:
                    pct_match = _PERCENT_PROGRESS_RE.search(line_str)
                    if pct_match:
                        try:
                            val = int(pct_match.group(1))
                            emit(ProgressEvent(
                                status=JobStatus.GENERATING,
                                progress=val / 100.0,
                                message=f"Sampling ({val}%)",
                            ))
                        except ValueError:
                            pass

            proc.wait()
        finally:
            self._current_process = None

        if proc.returncode != 0:
            tail = "\n".join(captured_lines[-20:]) if captured_lines else "(no output)"
            raise RuntimeError(
                f"SD.CPP process exited with code {proc.returncode}.\n"
                f"--- Engine output (last 20 lines) ---\n{tail}"
            )

        if not output_path.exists():
            tail = "\n".join(captured_lines[-20:]) if captured_lines else "(no output)"
            raise RuntimeError(
                f"SD.CPP finished but output file not found: {output_path}\n"
                f"--- Engine output (last 20 lines) ---\n{tail}"
            )

        elapsed = time.time() - start_time

        # Save metadata sidecar
        sidecar_path = output_path.with_suffix(".json")
        try:
            meta = {
                "filename": output_path.name,
                "prompt": prompt,
                "negative_prompt": getattr(request, "negative_prompt", None),
                "model_id": request.model_id,
                "seed": seed,
                "width": width,
                "height": height,
                "steps": request.steps,
                "backend": "sd_cpp",
                "engine": "stable-diffusion.cpp",
                "generation_time_sec": round(elapsed, 2),
                "timestamp": int(time.time()),
            }
            sidecar_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"[sd_cpp] Failed to write sidecar metadata: {e}")

        emit(ProgressEvent(
            status=JobStatus.COMPLETE,
            progress=1.0,
            image_path=str(output_path),
            message=f"Done in {elapsed:.1f}s",
        ))

        return GenerateResult(
            image_path=output_path,
            seed_used=seed,
            generation_time_sec=elapsed,
            model_id=request.model_id,
            backend="sd_cpp",
            steps=request.steps,
            prompt=prompt,
        )
