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


SD_CPP_WINDOWS_URL = "https://github.com/leejet/stable-diffusion.cpp/releases/download/master-946-a4a9669/sd-master-a4a9669-bin-win-cuda12-x64.zip"


def download_sd_cpp(
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    force: bool = False,
) -> Path:
    """Download and extract stable-diffusion.cpp standalone CUDA 12 build.

    Extracts to settings.engine_dir / 'sd_cpp'.
    """
    target_dir = get_sd_cpp_dir()
    cli_path = get_sd_cpp_cli_path()
    if cli_path.exists() and not force:
        return cli_path

    target_dir.mkdir(parents=True, exist_ok=True)
    temp_zip = target_dir / "sd_master_cuda12.zip.tmp"

    import zipfile
    from .engine import _download_file_with_progress

    try:
        logger.info("Downloading SD.CPP from %s", SD_CPP_WINDOWS_URL)
        _download_file_with_progress(
            SD_CPP_WINDOWS_URL,
            temp_zip,
            progress_callback=progress_callback,
        )

        logger.info("Extracting SD.CPP zip to %s", target_dir)
        with zipfile.ZipFile(temp_zip, "r") as z:
            z.extractall(target_dir)

        if not cli_path.exists():
            raise RuntimeError(f"sd-cli.exe not found after extracting {temp_zip}")

        logger.info("SD.CPP installed successfully at %s", cli_path)
        return cli_path
    finally:
        if temp_zip.exists():
            try:
                temp_zip.unlink()
            except OSError:
                pass


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

        def resolve_model_file(rel_or_abs: str) -> Path:
            p = Path(rel_or_abs)
            if p.is_absolute() and p.exists():
                return p
            if (repo_dir / rel_or_abs).exists():
                return repo_dir / rel_or_abs
            try:
                from .custom_models import get_custom_models_dir
                custom_candidate = get_custom_models_dir() / rel_or_abs
                if custom_candidate.exists():
                    return custom_candidate
            except Exception:
                pass
            if (repo_dir / "text_encoders" / rel_or_abs).exists():
                return repo_dir / "text_encoders" / rel_or_abs
            if (repo_dir / "vae" / rel_or_abs).exists():
                return repo_dir / "vae" / rel_or_abs
            return p if p.is_absolute() else repo_dir / rel_or_abs

        # Checkpoint vs standalone diffusion-model
        ckpt_rel = model.engine_args.get("checkpoint")
        if ckpt_rel:
            ckpt_path = resolve_model_file(ckpt_rel)
            cmd.extend(["-m", str(ckpt_path)])
        else:
            # Standalone diffusion model (e.g. DiT GGUF)
            diff_rel = model.engine_args.get("diffusion_model")
            if getattr(request, "variant_id", None):
                for t in getattr(model, "transformers", []):
                    if t.id == request.variant_id:
                        diff_rel = t.path
                        break
            if diff_rel:
                diff_path = resolve_model_file(diff_rel)
                if not diff_path.exists():
                    raise RuntimeError(
                        f"Diffusion model file not found: {diff_path.name}. "
                        "Please download it from the Models tab."
                    )
                cmd.extend(["--diffusion-model", str(diff_path)])

            # Text encoders (GGUF / safetensors)
            t5_rel = model.engine_args.get("t5xxl")
            if t5_rel:
                cmd.extend(["--t5xxl", str(resolve_model_file(t5_rel))])

            clip_rel = model.engine_args.get("clip_l")
            if clip_rel:
                cmd.extend(["--clip_l", str(resolve_model_file(clip_rel))])

            # LLM text encoder (for multimodal / Qwen-Image / Flux 2)
            llm_rel = model.engine_args.get("llm")
            if getattr(request, "custom_text_encoder_path", None):
                custom_te = Path(request.custom_text_encoder_path)
                if custom_te.exists():
                    cmd.extend(["--llm", str(custom_te)])
                else:
                    raise RuntimeError(f"Custom text encoder file not found: {custom_te}")
            elif getattr(request, "text_encoder_id", None):
                te_id = request.text_encoder_id
                direct_cand = resolve_model_file(te_id)
                if direct_cand.exists():
                    cmd.extend(["--llm", str(direct_cand)])
                elif getattr(model, "text_encoders", None):
                    te_spec = model.get_text_encoder(te_id)
                    if te_spec and te_spec.get("file"):
                        te_path = resolve_model_file(te_spec["file"])
                        if not te_path.exists():
                            raise RuntimeError(
                                f"Selected text encoder not found: {te_spec['file']}. "
                                "Please download it from the Models tab."
                            )
                        cmd.extend(["--llm", str(te_path)])
                    elif llm_rel:
                        cmd.extend(["--llm", str(resolve_model_file(llm_rel))])
                elif llm_rel:
                    cmd.extend(["--llm", str(resolve_model_file(llm_rel))])
            elif llm_rel:
                llm_path = resolve_model_file(llm_rel)
                if not llm_path.exists() and getattr(model, "text_encoders", None):
                    # Smart fallback: check if another registered text encoder is installed
                    for te in model.text_encoders:
                        cand = resolve_model_file(te.get("file", ""))
                        if cand.exists():
                            logger.info(f"[sd_cpp] Default LLM {llm_rel} not found; falling back to installed {te.get('file')}")
                            llm_path = cand
                            break
                if not llm_path.exists():
                    raise RuntimeError(
                        f"Text encoder file not found: {llm_rel}. "
                        "Please download it from the Models tab."
                    )
                cmd.extend(["--llm", str(llm_path)])

            # VAE
            vae_rel = None
            vae_type = getattr(request, "vae", None) or getattr(settings, "qwen_vae_type", "texture_fix")
            custom_vae_str = getattr(request, "custom_vae_path", None)
            if custom_vae_str:
                custom_vae_p = Path(custom_vae_str)
                if not custom_vae_p.is_file():
                    raise RuntimeError(f"Custom VAE file not found: {custom_vae_p}")
                cmd.extend(["--vae", str(custom_vae_p)])
            else:
                vae_rel = model.engine_args.get("vae")
                if model.family == "qwen" or model.id.startswith("qwen"):
                    if vae_type == "official":
                        vae_rel = "qwen_image_2.1_vae_bf16.safetensors"
                    else:
                        vae_rel = "texture_fix_vae_for_qwen_image_2.1_bf16.safetensors"

            if vae_rel:
                vae_search_paths = [
                    resolve_model_file(vae_rel),
                    settings.models_dir / "Klein-4B-Series" / "klein-4b-series-50x-above-base-model" / "vae" / "diffusion_pytorch_model.safetensors",
                    settings.models_dir / "Klein-9B-Series" / "klein-9b-series-50x-above-base-model" / "vae" / "diffusion_pytorch_model.safetensors",
                    settings.models_dir / "FLUX.1-schnell-gguf" / "ae.safetensors",
                    settings.models_dir / "FLUX.1-schnell-gguf" / vae_rel,
                    settings.models_dir / "Qwen-Image-2.1-4bit" / vae_rel,
                    settings.models_dir / "Qwen-Image-2.1-4bit" / "vae" / vae_rel,
                    settings.models_dir / "Qwen-Image-2.1-GGUF" / vae_rel,
                    settings.models_dir / "vae" / vae_rel,
                ]
                if vae_type == "official":
                    vae_search_paths.append(
                        settings.models_dir / "OzzyGT--Qwen_Image_2_1_Turbo_sdnq_dynamic_8bit" / "vae" / "diffusion_pytorch_model.safetensors"
                    )
                candidate_vae = next((p for p in vae_search_paths if p.exists()), resolve_model_file(vae_rel))

                if not candidate_vae.exists():
                    # Auto-download Qwen VAE if requested and missing
                    if model.family == "qwen" or model.id.startswith("qwen"):
                        from .sources import HuggingFaceSource
                        hf = HuggingFaceSource(token=settings.huggingface_token)
                        if vae_type == "texture_fix":
                            emit(ProgressEvent(
                                status=JobStatus.DOWNLOADING,
                                progress=0.1,
                                message=f"Downloading Texture-Fix VAE ({vae_rel}, 644 MB)...",
                            ))
                            hf.download_file(
                                repo_id="madebyollin/texture-fix-vae-for-qwen-image-2.1",
                                file_path="texture_fix_vae_for_qwen_image_2.1_bf16.safetensors",
                                local_dir=repo_dir,
                            )
                            candidate_vae = repo_dir / vae_rel
                        else:
                            emit(ProgressEvent(
                                status=JobStatus.DOWNLOADING,
                                progress=0.1,
                                message="Downloading Official Qwen-Image-2.1 VAE (644 MB)...",
                            ))
                            hf.download_file(
                                repo_id="OzzyGT/Qwen_Image_2_1_Turbo_sdnq_dynamic_8bit",
                                file_path="vae/diffusion_pytorch_model.safetensors",
                                local_dir=repo_dir,
                            )
                            dl_vae = repo_dir / "vae" / "diffusion_pytorch_model.safetensors"
                            if dl_vae.exists():
                                candidate_vae = repo_dir / vae_rel
                                if not candidate_vae.exists():
                                    dl_vae.rename(candidate_vae)
                    elif model.engine_args.get("vae_repo"):
                        from .sources import HuggingFaceSource
                        hf = HuggingFaceSource(token=settings.huggingface_token)
                        vae_remote = model.engine_args.get("vae_remote", f"vae/{vae_rel}")
                        emit(ProgressEvent(
                            status=JobStatus.DOWNLOADING,
                            progress=0.1,
                            message=f"Downloading VAE ({vae_rel})...",
                        ))
                        hf.download_file(
                            repo_id=model.engine_args["vae_repo"],
                            file_path=vae_remote,
                            local_dir=repo_dir,
                        )
                        target_v = repo_dir / vae_rel
                        dl_v = repo_dir / vae_remote
                        if dl_v.exists() and not target_v.exists() and dl_v != target_v:
                            import shutil
                            target_v.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(dl_v, target_v)
                        candidate_vae = target_v if target_v.exists() else dl_v
                    else:
                        raise RuntimeError(
                            f"VAE file not found: {vae_rel}. "
                            "Please download it from the Models tab."
                        )
                cmd.extend(["--vae", str(candidate_vae)])
                if model.family == "qwen" or model.id.startswith("qwen"):
                    cmd.append("--vae-tiling")

            # VAE latent format override
            if model.engine_args.get("vae_format"):
                cmd.extend(["--vae-format", model.engine_args["vae_format"]])

            # Flow shift
            if model.engine_args.get("flow_shift"):
                cmd.extend(["--flow-shift", str(model.engine_args["flow_shift"])])

        # Img2Img reference image
        if getattr(request, "init_image_path", None):
            ref_path = Path(request.init_image_path)
            if not ref_path.is_absolute():
                ref_path = settings.output_dir / ref_path
            if ref_path.exists():
                cmd.extend(["-i", str(ref_path)])
                strength = getattr(request, "init_image_strength", 0.75) or 0.75
                cmd.extend(["--strength", str(round(strength, 2))])

        # Guidance / CFG scale
        req_cfg = getattr(request, "cfg", None)
        if req_cfg is None:
            req_cfg = getattr(request, "guidance_scale", None)

        if model.engine_args.get("guidance_type") == "distilled" or model.family in ("flux", "sd3"):
            cfg_val = req_cfg if req_cfg is not None else model.get_default_cfg()
            # Always pass --cfg-scale 1.0 alongside --guidance so sd-cli doesn't use default cfg_scale=7.0
            cmd.extend(["--cfg-scale", "1.0", "--guidance", str(cfg_val)])
        else:
            default_cfg = model.get_default_cfg()
            cfg_val = req_cfg if req_cfg is not None else default_cfg
            cmd.extend(["--cfg-scale", str(cfg_val)])

        # Prompt & Negative prompt
        cmd.extend(["-p", prompt])
        neg_str = (getattr(request, "negative_prompt", None) or "").strip()
        if neg_str:
            cmd.extend(["-n", neg_str])
        elif (model.family == "qwen" or model.id.startswith("qwen")) and cfg_val > 1.0:
            # Ensure unconditional pass is active in sd.cpp when CFG > 1.0
            cmd.extend(["-n", " "])

        # Dimensions, steps, seed
        cmd.extend([
            "-W", str(width),
            "-H", str(height),
            "--steps", str(request.steps or 20),
            "-s", str(seed),
            "-o", str(output_path),
        ])

        # Flash Attention for diffusion model
        cmd.append("--diffusion-fa")

        # Sampling method & Noise / Eta
        noise_val = float(getattr(request, "noise", 0.0) or 0.0)
        ui_sampler = getattr(request, "sampler", None)
        if not ui_sampler and getattr(model, "scheduler_presets", None) and "default" in model.scheduler_presets:
            ui_sampler = model.scheduler_presets["default"].get("sampler")
        if not ui_sampler:
            ui_sampler = "euler"
        if noise_val > 0.0 and ui_sampler == "euler":
            ui_sampler = "euler_a"

        sd_cpp_sampler_map = {
            "euler": "euler",
            "euler_a": "euler_a",
            "heun": "heun",
            "dpm++2m": "dpm++2m",
            "dpm++2m_sde": "er_sde",
            "dpm2": "dpm2",
            "dpm2_a": "dpm++2s_a",
            "unipc": "res_multistep",
            "lcm": "lcm",
        }
        mapped_sampler = sd_cpp_sampler_map.get(ui_sampler, "euler")
        cmd.extend(["--sampling-method", mapped_sampler])
        # Always pass --eta explicitly when noise is specified or stochastic sampler is active
        if noise_val > 0.0 or mapped_sampler in ("euler_a", "er_sde", "dpm++2s_a", "lcm"):
            cmd.extend(["--eta", str(noise_val)])

        # Logging
        logger.info(f"[sd_cpp] Launching SD.CPP: {width}x{height}, steps={request.steps}, cfg={cfg_val}, sampler={mapped_sampler}, eta={noise_val}, seed={seed}")
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
                "variant_id": getattr(request, "variant_id", None),
                "seed": seed,
                "width": width,
                "height": height,
                "steps": request.steps,
                "cfg": cfg_val,
                "noise": noise_val,
                "sampler": ui_sampler,
                "vae": vae_type,
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
