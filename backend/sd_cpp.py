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
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

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
    cli_path = get_sd_cpp_cli_path()
    if not cli_path.exists():
        return "not installed"
    try:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        res = subprocess.run(
            [str(cli_path), "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(cli_path.parent),
            creationflags=creationflags,
        )
        out = (res.stdout or res.stderr or "").strip()
        m = re.search(r"commit\s+([0-9a-fA-F]+)", out)
        if m:
            return f"master-{m.group(1)} (cu12)"
    except Exception:
        pass
    return "master-f0aac52 (cu12)"


SD_CPP_WINDOWS_URL = "https://github.com/leejet/stable-diffusion.cpp/releases/download/master-973-f0aac52/sd-master-f0aac52-bin-win-cuda12-x64.zip"
SD_CPP_CUDART_URL = "https://github.com/leejet/stable-diffusion.cpp/releases/download/master-973-f0aac52/cudart-sd-bin-win-cu12-x64.zip"


def _resolve_latest_sd_cpp_url() -> tuple[str, str]:
    """Query GitHub API for the latest leejet/stable-diffusion.cpp CUDA 12 Windows release."""
    try:
        import requests
        r = requests.get(
            "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases/latest",
            timeout=8,
        )
        if r.status_code == 200:
            data = r.json()
            win_url = None
            cudart_url = None
            for asset in data.get("assets", []):
                name = (asset.get("name") or "").lower()
                url = asset.get("browser_download_url")
                if not url:
                    continue
                if "bin-win-cuda12-x64.zip" in name:
                    win_url = url
                elif "cudart-sd-bin-win-cu12-x64.zip" in name:
                    cudart_url = url
            if win_url:
                return win_url, (cudart_url or SD_CPP_CUDART_URL)
    except Exception as e:
        logger.debug("Could not query latest SD.CPP release from GitHub API: %s", e)
    return SD_CPP_WINDOWS_URL, SD_CPP_CUDART_URL


def _sd_cpp_has_working_cuda(cli_path: Path) -> bool:
    """Verify whether sd-cli.exe can load ggml-cuda.dll and see a CUDA device."""
    if not cli_path.exists():
        return False
    try:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        res = subprocess.run(
            [str(cli_path), "--list-devices"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(cli_path.parent),
            creationflags=creationflags,
        )
        return "CUDA" in (res.stdout or "")
    except Exception:
        return False


def download_sd_cpp(
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    force: bool = False,
) -> Path:
    """Download and extract stable-diffusion.cpp standalone CUDA 12 build.

    Also ensures CUDA 12 runtime DLLs (cudart64_12.dll, cublas64_12.dll, cublasLt64_12.dll)
    are present if the host system does not have CUDA Toolkit 12 in PATH (e.g. clean RTX 5070 Ti PC).
    """
    target_dir = get_sd_cpp_dir()
    cli_path = get_sd_cpp_cli_path()

    import zipfile
    from .engine import _download_file_with_progress

    target_dir.mkdir(parents=True, exist_ok=True)
    win_url, cudart_url = _resolve_latest_sd_cpp_url() if force else (SD_CPP_WINDOWS_URL, SD_CPP_CUDART_URL)

    if not cli_path.exists() or force:
        temp_zip = target_dir / "sd_master_cuda12.zip.tmp"
        try:
            logger.info("Downloading SD.CPP from %s", win_url)
            _download_file_with_progress(
                win_url,
                temp_zip,
                progress_callback=progress_callback,
            )

            logger.info("Extracting SD.CPP zip to %s", target_dir)
            with zipfile.ZipFile(temp_zip, "r") as z:
                z.extractall(target_dir)

            if not cli_path.exists():
                raise RuntimeError(f"sd-cli.exe not found after extracting {temp_zip}")
        finally:
            if temp_zip.exists():
                try:
                    temp_zip.unlink()
                except OSError:
                    pass

    # Check if CUDA backend loads properly; if not (missing cudart64_12.dll / cublas64_12.dll on fresh PC),
    # download the official cudart-sd-bin-win-cu12-x64.zip bundle into target_dir.
    if not _sd_cpp_has_working_cuda(cli_path) and not (target_dir / "cublas64_12.dll").exists():
        cudart_zip = target_dir / "cudart_cu12.zip.tmp"
        try:
            logger.info("CUDA 12 runtime DLLs not found in system PATH; downloading %s", SD_CPP_CUDART_URL)
            _download_file_with_progress(
                SD_CPP_CUDART_URL,
                cudart_zip,
                progress_callback=progress_callback,
            )
            with zipfile.ZipFile(cudart_zip, "r") as z:
                z.extractall(target_dir)
        except Exception as e:
            logger.warning("Could not download optional cudart bundle: %s", e)
        finally:
            if cudart_zip.exists():
                try:
                    cudart_zip.unlink()
                except OSError:
                    pass

    logger.info("SD.CPP ready at %s", cli_path)
    return cli_path


def uninstall_sd_cpp() -> dict:
    """Remove the managed SD.CPP directory (binary, CUDA DLLs, downloads)."""
    target_dir = get_sd_cpp_dir()
    removed = target_dir.exists()
    if removed:
        shutil.rmtree(target_dir)
        logger.info("Removed SD.CPP directory: %s", target_dir)
    return {
        "engine": "sd_cpp",
        "removed": removed,
        "path": str(target_dir),
        "message": "SD.CPP удалён" if removed else "SD.CPP уже не установлен",
    }


# Progress matching regex for stable-diffusion.cpp
# Example outputs:
#   "| 3/20 [00:02<00:11,  1.45s/it]"
#   "generating image: 5/20"
#   "sampling: 45%"
_STEP_PROGRESS_RE = re.compile(r"(\d+)\s*/\s*(\d+)")
_PERCENT_PROGRESS_RE = re.compile(r"(\d+)\s*%")


def _ensure_moss_gguf_qkv_fused(gguf_path: Path, emit: Optional[Callable] = None) -> None:
    """Losslessly fuse split layers.{i}.proj.{q,k,v}.weight into layers.{i}.proj.qkv.weight
    for RebelAI MOSS-Image GGUF checkpoints so stable-diffusion.cpp can load them natively."""
    import struct

    try:
        with open(gguf_path, "rb") as f:
            head_probe = f.read(65536)
        if b"moss_image" not in head_probe or b"layers.0.proj.qkv.weight" in head_probe:
            return
        if b"layers.0.proj.q.weight" not in head_probe:
            return
    except Exception:
        return

    logger.info("[sd_cpp] Fusing split Q/K/V tensors in MOSS-Image GGUF: %s", gguf_path.name)
    if emit:
        from .worker import JobStatus, ProgressEvent
        emit(ProgressEvent(
            status=JobStatus.LOADING,
            progress=0.04,
            message=f"Converting {gguf_path.name} Q/K/V tensors for SD.CPP (one-time)...",
        ))

    with open(gguf_path, "rb") as f:
        magic, ver, n_tensors, n_kv = struct.unpack("<4sIQQ", f.read(24))
        if magic != b"GGUF":
            return
        kv_start = 24

        def r_str() -> str:
            (length,) = struct.unpack("<Q", f.read(8))
            return f.read(length).decode("utf-8")

        def skip_val(vtype: int) -> None:
            sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
            if vtype in sz:
                f.read(sz[vtype])
            elif vtype == 8:
                r_str()
            elif vtype == 9:
                et, cnt = struct.unpack("<IQ", f.read(12))
                for _ in range(cnt):
                    skip_val(et)

        for _ in range(n_kv):
            _ = r_str()
            (vt,) = struct.unpack("<I", f.read(4))
            skip_val(vt)
        kv_end = f.tell()
        f.seek(kv_start)
        kv_bytes = f.read(kv_end - kv_start)

        tensors = []
        for _ in range(n_tensors):
            name = r_str()
            (nd,) = struct.unpack("<I", f.read(4))
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
            tt, off = struct.unpack("<IQ", f.read(12))
            tensors.append((name, dims, tt, off))
        header_end = f.tell()
        data_offset = (header_end + 31) // 32 * 32

    file_size = gguf_path.stat().st_size
    sorted_t = sorted(tensors, key=lambda x: x[3])
    sizes = {}
    for idx, (name, dims, tt, off) in enumerate(sorted_t):
        next_off = sorted_t[idx + 1][3] if idx + 1 < len(sorted_t) else (file_size - data_offset)
        sizes[name] = next_off - off

    t_map = {t[0]: t for t in tensors}
    new_entries = []
    consumed = set()

    for name, dims, tt, off in tensors:
        if name in consumed:
            continue
        m = re.match(r"^(layers\.\d+\.proj)\.(q|k|v)\.weight$", name)
        if m:
            prefix = m.group(1)
            q_name = f"{prefix}.q.weight"
            k_name = f"{prefix}.k.weight"
            v_name = f"{prefix}.v.weight"
            if q_name in t_map and k_name in t_map and v_name in t_map:
                consumed.update((q_name, k_name, v_name))
                q_t, k_t, v_t = t_map[q_name], t_map[k_name], t_map[v_name]
                fused_dims = [q_t[1][0], q_t[1][1] + k_t[1][1] + v_t[1][1]]
                new_entries.append((
                    f"{prefix}.qkv.weight",
                    fused_dims,
                    q_t[2],
                    [(q_t[3], sizes[q_name]), (k_t[3], sizes[k_name]), (v_t[3], sizes[v_name])],
                ))
                continue
        new_entries.append((name, dims, tt, [(off, sizes[name])]))

    # Compute new relative data offsets (32-byte aligned)
    cur_rel_off = 0
    header_tensor_parts = []
    entry_Plans = []
    for name, dims, tt, chunks in new_entries:
        aligned_off = (cur_rel_off + 31) // 32 * 32
        pad_before = aligned_off - cur_rel_off
        cur_rel_off = aligned_off
        total_len = sum(length for _, length in chunks)
        name_b = name.encode("utf-8")
        t_hdr = [struct.pack("<Q", len(name_b)), name_b, struct.pack("<I", len(dims))]
        for d in dims:
            t_hdr.append(struct.pack("<Q", d))
        t_hdr.append(struct.pack("<IQ", tt, cur_rel_off))
        header_tensor_parts.append(b"".join(t_hdr))
        entry_Plans.append((pad_before, chunks))
        cur_rel_off += total_len

    new_header = (
        struct.pack("<4sIQQ", b"GGUF", ver, len(new_entries), n_kv)
        + kv_bytes
        + b"".join(header_tensor_parts)
    )
    new_data_offset = (len(new_header) + 31) // 32 * 32
    header_pad = b"\x00" * (new_data_offset - len(new_header))

    tmp_path = gguf_path.with_suffix(".gguf.tmp")
    with open(gguf_path, "rb") as src, open(tmp_path, "wb") as dst:
        dst.write(new_header)
        dst.write(header_pad)
        for pad_before, chunks in entry_Plans:
            if pad_before > 0:
                dst.write(b"\x00" * pad_before)
            for src_rel_off, length in chunks:
                src.seek(data_offset + src_rel_off)
                remaining = length
                while remaining > 0:
                    buf = src.read(min(remaining, 8 * 1024 * 1024))
                    if not buf:
                        break
                    dst.write(buf)
                    remaining -= len(buf)

    tmp_path.replace(gguf_path)
    logger.info("[sd_cpp] Successfully fused Q/K/V tensors in %s (%d tensors)", gguf_path.name, len(new_entries))


class SDCppWorker:
    """Executes diffusion inference via sd-cli.exe."""

    def __init__(self):
        self._cancel_flag = threading.Event()
        self._current_process: Optional[subprocess.Popen] = None

    def cancel(self):
        """Cancel current running process."""
        self._cancel_flag.set()
        proc = self._current_process
        if proc:
            try:
                proc.terminate()
            except Exception:
                pass
            if sys.platform == "win32":
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        capture_output=True,
                        check=False,
                    )
                except Exception:
                    pass
            try:
                proc.kill()
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
            base_name = Path(rel_or_abs).name
            if (repo_dir / rel_or_abs).exists():
                return repo_dir / rel_or_abs
            if (repo_dir / base_name).exists():
                return repo_dir / base_name
            if (repo_dir / "text_encoders" / rel_or_abs).exists():
                return repo_dir / "text_encoders" / rel_or_abs
            if (repo_dir / "text_encoders" / base_name).exists():
                return repo_dir / "text_encoders" / base_name
            if (repo_dir / "diffusion_models" / rel_or_abs).exists():
                return repo_dir / "diffusion_models" / rel_or_abs
            if (repo_dir / "diffusion_models" / base_name).exists():
                return repo_dir / "diffusion_models" / base_name
            if (repo_dir / "vae" / rel_or_abs).exists():
                return repo_dir / "vae" / rel_or_abs
            if (repo_dir / "vae" / base_name).exists():
                return repo_dir / "vae" / base_name
            # Check secondary repos (e.g. text encoders from unsloth/Qwen3-8B-GGUF or variants from akhaliq/...)
            sec_repos = [
                t.repo_id for t in getattr(model, "transformers", []) if getattr(t, "repo_id", None)
            ] + [
                te.get("source", {}).get("repo_id")
                for te in getattr(model, "text_encoders", [])
                if isinstance(te.get("source"), dict) and te.get("source", {}).get("repo_id")
            ]
            for cand_repo in sec_repos:
                alt_dir = cache.repo_dir_for(cand_repo)
                if (alt_dir / rel_or_abs).exists():
                    return alt_dir / rel_or_abs
                if (alt_dir / base_name).exists():
                    return alt_dir / base_name
                if (alt_dir / "text_encoders" / rel_or_abs).exists():
                    return alt_dir / "text_encoders" / rel_or_abs
                if (alt_dir / "text_encoders" / base_name).exists():
                    return alt_dir / "text_encoders" / base_name
                if (alt_dir / "diffusion_models" / rel_or_abs).exists():
                    return alt_dir / "diffusion_models" / rel_or_abs
                if (alt_dir / "diffusion_models" / base_name).exists():
                    return alt_dir / "diffusion_models" / base_name
                if (alt_dir / "vae" / rel_or_abs).exists():
                    return alt_dir / "vae" / rel_or_abs
                if (alt_dir / "vae" / base_name).exists():
                    return alt_dir / "vae" / base_name
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
                if diff_path.suffix.lower() == ".gguf":
                    _ensure_moss_gguf_qkv_fused(diff_path, emit=emit)
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
            is_qwen = model.family == "qwen" or model.id.startswith("qwen")
            is_klein_flux = model.family in ("klein", "flux") or "flux" in model.id.lower() or "klein" in model.id.lower()
            if custom_vae_str:
                custom_vae_p = Path(custom_vae_str)
                if not custom_vae_p.is_file():
                    raise RuntimeError(f"Custom VAE file not found: {custom_vae_p}")
                cmd.extend(["--vae", str(custom_vae_p)])
            else:
                vae_rel = model.engine_args.get("vae")
                if is_qwen:
                    if vae_type == "official":
                        vae_rel = "qwen_image_2.1_vae_bf16.safetensors"
                    elif vae_type == "pid":
                        vae_rel = "qwen_image_2.1_vae_bf16.safetensors"
                    else:
                        vae_rel = "texture_fix_vae_for_qwen_image_2.1_bf16.safetensors"
                elif is_klein_flux:
                    if vae_type == "pid":
                        vae_rel = "flux2-vae.safetensors"

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
            if vae_type == "pid":
                if is_qwen:
                    cmd.extend(["--vae-format", "wan"])
                elif is_klein_flux:
                    cmd.extend(["--vae-format", "flux2"])
            elif model.engine_args.get("vae_format"):
                cmd.extend(["--vae-format", model.engine_args["vae_format"]])

            # Flow shift
            if model.engine_args.get("flow_shift"):
                cmd.extend(["--flow-shift", str(model.engine_args["flow_shift"])])

            # External Tokenizer (required for MOSS-Image / Qwen3.5 and NVIDIA PiD)
            tok_rel = model.engine_args.get("tokenizer")
            is_moss_or_qwen35 = (
                "moss" in model.id.lower()
                or "moss" in (model.family or "").lower()
                or "qwen3.5" in (llm_rel or "").lower()
                or "moss" in (diff_rel or "").lower()
            )
            is_pid_model = bool(
                model.family == "pid"
                or (model.id and "pid" in model.id.lower() and not ("qwen" in model.id.lower() or "flux" in model.id.lower()))
                or (diff_rel and "pid" in diff_rel.lower())
                or (tok_rel and "pid" in tok_rel.lower())
            )
            if tok_rel or is_moss_or_qwen35 or is_pid_model:
                tok_rel = tok_rel or ("tokenizer_gemma2.json" if is_pid_model else "tokenizer.json")
                tok_search = [
                    resolve_model_file(tok_rel),
                    repo_dir / tok_rel,
                    repo_dir / "tokenizer.json",
                    repo_dir / "tokenizer" / "tokenizer.json",
                    repo_dir / "tokenizer_qwen3.5.json",
                    settings.models_dir / "MOSS-Image_GGUFs" / "tokenizer.json",
                    settings.models_dir / "MOSS-Image_GGUFs" / "tokenizer" / "tokenizer.json",
                    settings.models_dir / "PiD" / tok_rel,
                    settings.models_dir / "tokenizers" / tok_rel,
                    get_sd_cpp_dir() / tok_rel,
                ]
                tok_path = next((p for p in tok_search if p.exists() and p.is_file()), None)
                if tok_path is None:
                    from .sources import HuggingFaceSource
                    hf = HuggingFaceSource(token=settings.huggingface_token)
                    default_repo = "unsloth/gemma-2-2b-it" if is_pid_model else "OpenMOSS-Team/MOSS-Image-Base"
                    default_remote = "tokenizer.json" if is_pid_model else "tokenizer/tokenizer.json"
                    tok_repo = model.engine_args.get("tokenizer_repo", default_repo)
                    tok_remote = model.engine_args.get("tokenizer_remote", default_remote)
                    tok_title = "PiD Gemma 2 Tokenizer" if is_pid_model else "Qwen3.5 Tokenizer"
                    emit(ProgressEvent(
                        status=JobStatus.DOWNLOADING,
                        progress=0.12,
                        message=f"Downloading {tok_title} ({tok_rel})...",
                    ))
                    hf.download_file(
                        repo_id=tok_repo,
                        file_path=tok_remote,
                        local_dir=repo_dir,
                    )
                    dl_tok = repo_dir / tok_remote
                    target_tok = repo_dir / tok_rel
                    if dl_tok.exists() and not target_tok.exists() and dl_tok != target_tok:
                        import shutil
                        shutil.copy2(dl_tok, target_tok)
                    tok_path = target_tok if target_tok.exists() else dl_tok
                if tok_path and tok_path.exists():
                    cmd.extend(["--tokenizer", str(tok_path)])

        # ----- Inpaint mask preparation (crop-to-mask) -----
        # SD.CPP applies --mask to the init image (white = regenerate,
        # black = preserve), so a mask always forces classic img2img mode.
        # The engine runs only on a crop around the painted area; the result is
        # pasted back into the original frame afterwards.
        mask_source = getattr(request, "mask_data_url", None) or (
            str(request.mask_path) if getattr(request, "mask_path", None) else None
        )
        mask_ctx = None
        if mask_source:
            from PIL import Image
            from .masking import (
                MaskError,
                crop_array,
                crop_image,
                mask_file_name,
                normalize_crop_mode,
                prepare_mask_arrays,
                resolve_crop_box,
                save_image_file,
                save_mask_file,
            )

            raw_init = getattr(request, "init_image_path", None)
            if not raw_init:
                raise ValueError(
                    "Mask editing requires a source image (init_image_path is missing)"
                )
            source_path = Path(raw_init)
            if not source_path.is_absolute():
                source_path = settings.output_dir / source_path
            if not source_path.exists():
                raise ValueError(f"Source image not found: {source_path.name}")

            with Image.open(source_path) as src_file:
                source_image = src_file.copy()
            try:
                raw_mask, engine_mask = prepare_mask_arrays(
                    mask_source,
                    target_size=source_image.size,
                    invert_mask=bool(getattr(request, "mask_invert", False)),
                    grow=int(getattr(request, "mask_grow", 0) or 0),
                    blur=float(getattr(request, "mask_blur", 0.0) or 0.0),
                )
            except MaskError as exc:
                raise ValueError(str(exc)) from exc

            crop_mode = normalize_crop_mode(getattr(request, "mask_crop_mode", "auto"))
            # sd.cpp requires 64px alignment and 512px min size for optimal
            # latent-space compatibility (64 * 8 = 512 tokens).  Pass engine-
            # specific values so other engines (QuantFunc, Nunchaku) keep
            # their native 16px / 256px defaults.
            sd_align = 64
            sd_min_size = 512
            crop_box = resolve_crop_box(
                engine_mask,
                mode=crop_mode,
                padding=getattr(request, "mask_crop_padding", 64),
                align=sd_align,
                min_size=sd_min_size,
                manual_box=getattr(request, "mask_crop_box", None),
                coverage_mask=raw_mask,
            )
            if crop_box is None:
                raise ValueError("Mask is empty - paint the area to edit")
            crop_w = crop_box[2] - crop_box[0]
            crop_h = crop_box[3] - crop_box[1]
            init_crop = crop_image(source_image, crop_box)
            raw_crop = crop_array(raw_mask, crop_box)
            engine_crop = crop_array(engine_mask, crop_box)

            from .masking import heal_masked_image, scale_crop_for_safe_inference
            safe_init, safe_engine_mask, _ = scale_crop_for_safe_inference(
                init_crop, engine_crop, max_dim=1024
            )

            # Pre-heal the masked area when doing full replacement (strength >= 0.75) so
            # high-contrast SFX/text borders do not leak through VAE convs & self-attention.
            _req_ms = getattr(request, "mask_strength", None)
            _ms_val = 1.0 if _req_ms is None else float(_req_ms)
            if _ms_val >= 0.75:
                safe_init = heal_masked_image(safe_init, safe_engine_mask, dilate_px=8)

            # stable-diffusion.cpp applies sd::ops::round(mask_image_tensor) at 0.5, which
            # clips off soft brush edges (0.04..0.49 -> 0.0). Boost any painted/grown pixels
            # above 0.5 threshold so sd-cli regenerates the entire stroke + soft border.
            safe_engine_mask = np.where(safe_engine_mask > 0.04, 1.0, 0.0).astype(np.float32)

            mask_file = save_mask_file(
                safe_engine_mask,
                settings.output_dir / "masks",
                mask_file_name("sd_cpp_mask"),
            )
            init_file = save_image_file(
                safe_init,
                settings.output_dir / "masks",
                mask_file_name(f"sd_cpp_init_{seed}"),
            )
            mask_ctx = {
                "array": raw_mask,
                "crop_array": raw_crop,
                "box": crop_box,
                "crop_size": safe_init.size,
                "crop_mode": str(crop_mode),
                "crop_padding": int(getattr(request, "mask_crop_padding", 64) or 0),
                "crop_min_size": int(getattr(request, "mask_crop_min_size", 256) or 0),
                "file": mask_file,
                "init_file": init_file,
                "source_image": source_image,
            }
            emit(ProgressEvent(
                status=JobStatus.INITIALIZING,
                message=(
                    f"Mask ready: {crop_w}x{crop_h} crop "
                    f"at ({crop_box[0]},{crop_box[1]}) of "
                    f"{source_image.size[0]}x{source_image.size[1]}"
                ),
            ))

            # Optional: split mask into connected components (islands) and
            # process each separately.  Enabled when request carries
            # mask_split_islands=true.
            _split_islands = getattr(request, "mask_split_islands", False)
            if _split_islands:
                from .masking import split_mask_into_islands
                _island_boxes = split_mask_into_islands(
                    raw_mask,
                    base_box=crop_box,
                    min_area=256,
                    merge_distance=64,
                )
                if len(_island_boxes) > 1:
                    mask_ctx["island_boxes"] = _island_boxes
                    logger.info(
                        "[sd_cpp] Mask split into %d island(s): %s",
                        len(_island_boxes),
                        [_is["box"] for _is in _island_boxes],
                    )
                else:
                    mask_ctx["island_boxes"] = None
            else:
                mask_ctx["island_boxes"] = None

        # ``mask_strength`` may legitimately be 0.0, so ``or`` must NOT be
        # used here — it would silently turn 0 into 1.
        raw_mask_strength = getattr(request, "mask_strength", None)
        mask_strength_value = 1.0 if raw_mask_strength is None else float(raw_mask_strength)

        native_ref = bool(getattr(request, "native_ref_res", False))

        # Reference Image: Native Control/Ref (-r / --control-image / --control-net) vs Classic Noise Img2Img (-i + --strength)
        if getattr(request, "init_image_path", None):
            ref_path = Path(request.init_image_path)
            if not ref_path.is_absolute():
                ref_path = settings.output_dir / ref_path
            if ref_path.exists():
                ref_mode = (getattr(request, "ref_mode", None) or "control_ref").strip().lower()
                strength = float(getattr(request, "init_image_strength", 0.85) or 0.85)
                cn_str = (getattr(request, "control_net_path", None) or "").strip()
                cn_path: Optional[Path] = None
                if cn_str:
                    cand_cn = Path(cn_str)
                    if not cand_cn.is_absolute():
                        for p in (resolve_model_file(cn_str), settings.models_dir / cn_str, settings.models_dir / "controlnet" / cn_str):
                            if p.exists():
                                cand_cn = p
                                break
                    if cand_cn.exists():
                        cn_path = cand_cn

                if mask_ctx is not None:
                    # Masked img2img: SD.CPP only honours --mask together with
                    # --init-img.  Both files are the mask-bbox crop, so the
                    # engine never denoises the full frame.  mask_strength maps
                    # to the denoising strength.
                    cmd.extend(["-i", str(mask_ctx["init_file"])])
                    cmd.extend(["--strength", str(round(max(0.01, min(1.0, mask_strength_value)), 2))])
                    cmd.extend(["--mask", str(mask_ctx["file"])])
                elif ref_mode == "img2img":
                    # Classic noise-based Img2Img (--init-img + --strength)
                    cmd.extend(["-i", str(ref_path)])
                    cmd.extend(["--strength", str(round(strength, 2))])
                else:
                    # Step isolation: NO --init-img (-i) and NO --strength!
                    # Model executes 100% of steps from clean white noise while guided by ControlNet / Reference latents.
                    if cn_path is not None:
                        cmd.extend([
                            "--control-net", str(cn_path),
                            "--control-image", str(ref_path),
                            "--control-strength", str(round(strength, 2)),
                        ])
                        if ref_mode == "controlnet_canny":
                            cmd.extend(["--image-preprocess", "target=control,canny=true"])
                    elif model.family == "qwen" or model.id.startswith("qwen"):
                        # Native Qwen-Image-2.1 Reference / Control conditioning (-r + --llm_vision)
                        mmproj_candidates = [
                            repo_dir / "mmproj-BF16.gguf",
                            repo_dir / "mmproj-F16.gguf",
                            settings.models_dir / "unsloth--Qwen3-VL-8B-Instruct-GGUF" / "mmproj-BF16.gguf",
                            settings.models_dir / "unsloth--Qwen3-VL-8B-Instruct-GGUF" / "mmproj-F16.gguf",
                        ]
                        mmproj_path = next((p for p in mmproj_candidates if p.exists()), None)
                        if mmproj_path is None:
                            from .sources import HuggingFaceSource
                            hf = HuggingFaceSource(token=settings.huggingface_token)
                            emit(ProgressEvent(
                                status=JobStatus.DOWNLOADING,
                                progress=0.12,
                                message="Downloading Qwen3-VL Vision Projector (mmproj-F16.gguf, ~1.1 GB) for Native Control/Reference...",
                            ))
                            hf.download_file(
                                repo_id="unsloth/Qwen3-VL-8B-Instruct-GGUF",
                                file_path="mmproj-F16.gguf",
                                local_dir=repo_dir,
                            )
                            mmproj_path = repo_dir / "mmproj-F16.gguf"
                        # Reference resolution mode:
                        # - Default (False, как на раньше): safe downsampling (max dimension <= 1024),
                        #   allowing ViT to execute in ~2 seconds without 17.3GB VRAM explosion or CPU lag.
                        # - Native (True, toggle ON): full source resolution (1500+ px) with RAM offload.
                        native_ref = getattr(request, "native_ref_res", False) is True
                        ref_w, ref_h = width, height
                        try:
                            from PIL import Image as _PIL_Image
                            with _PIL_Image.open(ref_path) as _ref_im:
                                raw_w, raw_h = _ref_im.size
                            if native_ref:
                                ref_w, ref_h = raw_w, raw_h
                            else:
                                max_dim = 1024
                                if max(raw_w, raw_h) > max_dim:
                                    scale = float(max_dim) / float(max(raw_w, raw_h))
                                    ref_w = max(32, int(round((raw_w * scale) / 32) * 32))
                                    ref_h = max(32, int(round((raw_h * scale) / 32) * 32))
                                else:
                                    ref_w, ref_h = raw_w, raw_h
                        except Exception:
                            pass
                        w32 = max(32, (ref_w // 32) * 32)
                        h32 = max(32, (ref_h // 32) * 32)
                        preproc = f"target=ref,mode=crop-resize,width={w32},height={h32}"
                        if ref_mode == "controlnet_canny":
                            preproc += ",canny=true"
                        cmd.extend([
                            "--llm_vision", str(mmproj_path),
                            "-r", str(ref_path),
                            "--image-preprocess", preproc,
                        ])
                    elif model.id in ("flux_2_klein_base_9b_gguf", "flux_3_image_klein_gguf") or "klein" in model.id or model.family == "flux":
                        # Native Flux.2 / Flux.3 in-context Reference / Control conditioning (-r)
                        cmd.extend(["-r", str(ref_path)])
                        if ref_mode == "controlnet_canny":
                            cmd.extend(["--image-preprocess", "target=ref,canny=true"])
                    else:
                        # Fallback for SDXL without external ControlNet file
                        cmd.extend(["-i", str(ref_path), "--strength", str(round(strength, 2))])

        # Guidance / CFG scale
        req_cfg = getattr(request, "cfg", None)
        if req_cfg is None:
            req_cfg = getattr(request, "guidance_scale", None)

        variant_id = getattr(request, "variant_id", "") or ""
        is_base = ("base" in variant_id.lower()) or ("base" in model.id.lower())

        if not is_base and (model.engine_args.get("guidance_type") == "distilled" or model.family in ("flux", "sd3")):
            cfg_val = req_cfg if req_cfg is not None else model.get_default_cfg()
            # Always pass --cfg-scale 1.0 alongside --guidance so sd-cli doesn't use default cfg_scale=7.0
            cmd.extend(["--cfg-scale", "1.0", "--guidance", str(cfg_val)])
        else:
            default_cfg = model.scheduler_presets.get("base", {}).get("cfg") if is_base and "base" in model.scheduler_presets else model.get_default_cfg()
            cfg_val = req_cfg if req_cfg is not None else default_cfg
            cmd.extend(["--cfg-scale", str(cfg_val)])

        # LoRA injection: append prompt tags <lora:stem:strength> and specify --lora-model-dir
        lora_paths = getattr(request, "lora_paths", None) or []
        lora_strengths = getattr(request, "lora_strengths", None) or []
        if lora_paths:
            from .lora import get_loras_dir, lora_prompt_tags
            lora_tags = lora_prompt_tags(lora_paths, lora_strengths)
            if lora_tags:
                prompt = f"{prompt} {lora_tags}".strip()
            cmd.extend(["--lora-model-dir", str(get_loras_dir())])

        # Prompt & Negative prompt
        cmd.extend(["-p", prompt])
        neg_str = (getattr(request, "negative_prompt", None) or "").strip()
        if neg_str:
            cmd.extend(["-n", neg_str])
        elif (model.family in ("qwen", "klein") or model.id.startswith("qwen") or is_base) and cfg_val > 1.0:
            # Ensure unconditional pass is active in sd.cpp when CFG > 1.0
            cmd.extend(["-n", " "])

        # Dimensions, steps, seed & Highres Fix (Option A)
        hires_fix = bool(getattr(request, "hires_fix", False))
        hires_scale = float(getattr(request, "hires_scale", 2.0) or 2.0)
        hires_strength = float(getattr(request, "hires_denoising_strength", 0.35) or 0.35)
        raw_upscaler = str(getattr(request, "hires_upscaler", "") or "")
        is_pid_hires = (vae_type == "pid") or ("pid" in raw_upscaler.lower())

        user_hires_steps = getattr(request, "hires_steps", None)
        if user_hires_steps is not None and int(user_hires_steps) > 0:
            hires_steps = int(user_hires_steps)
            if "pid" in raw_upscaler.lower():
                hires_steps = min(hires_steps, 4)
        elif is_pid_hires:
            hires_steps = 4
        else:
            hires_steps = 6

        if is_pid_hires:
            hires_upscaler = "Latent (bicubic)"
            if width >= 1536 or height >= 1536:
                hires_fix = True
        else:
            hires_upscaler = raw_upscaler or "Latent (bicubic)"

        valid_builtin_upscalers = {
            "latent": "Latent",
            "latent (bicubic)": "Latent (bicubic)",
            "latent (nearest)": "Latent (nearest)",
            "latent (antialiased)": "Latent (antialiased)",
            "lanczos": "Lanczos",
            "nearest": "Nearest",
        }
        mapped_hires_upscaler = valid_builtin_upscalers.get(
            hires_upscaler.strip().lower(), "Latent (bicubic)"
        )

        # Masked runs operate on the inpainting bbox and avoid hires scaling conflicts
        if mask_ctx is not None:
            hires_fix = False

        if hires_fix:
            if width >= 1536 or height >= 1536:
                target_w = width
                target_h = height
                base_w = max(64, int(round((target_w / hires_scale) / 64) * 64))
                base_h = max(64, int(round((target_h / hires_scale) / 64) * 64))
            else:
                base_w = width
                base_h = height
                target_w = max(64, int(round((base_w * hires_scale) / 64) * 64))
                target_h = max(64, int(round((base_h * hires_scale) / 64) * 64))

            engine_width, engine_height = base_w, base_h
            cmd.extend([
                "-W", str(base_w),
                "-H", str(base_h),
                "--steps", str(request.steps or 20),
                "--hires",
                "--hires-width", str(target_w),
                "--hires-height", str(target_h),
                "--hires-scale", str(hires_scale),
                "--hires-steps", str(hires_steps),
                "--hires-denoising-strength", str(hires_strength),
                "--hires-upscaler", str(mapped_hires_upscaler),
                "-s", str(seed),
                "-o", str(output_path),
            ])
        else:
            engine_width, engine_height = (
                mask_ctx["crop_size"] if mask_ctx is not None else (width, height)
            )
            target_w, target_h = engine_width, engine_height
            cmd.extend([
                "-W", str(engine_width),
                "-H", str(engine_height),
                "--steps", str(request.steps or 20),
                "-s", str(seed),
                "-o", str(output_path),
            ])


        # Flash Attention for diffusion model and text/vision conditioners
        # --flash-attn on enables FlashAttention globally across diffusion, ViT, and LLM,
        # keeping ViT attention memory under 2GB and preventing 17.3GB OOM on high resolutions.
        if "--flash-attn" not in cmd and "--diffusion-fa" not in cmd:
            cmd.extend(["--flash-attn", "on"])

        # Memory optimization & RAM offload (ported from SnowE22)
        gpu_vram = gpu_info.vram_mb if gpu_info else 8192
        ref_dim_max = max(w32 if "w32" in locals() else 0, h32 if "h32" in locals() else 0)
        has_large_res = engine_width > 1024 or engine_height > 1024 or ref_dim_max > 1024 or hires_fix or target_w > 1024 or target_h > 1024

        # VAE spatial tiling: essential for large resolutions (> 1024) or tight VRAM
        if gpu_vram <= 16384 or has_large_res:
            if "--vae-tiling" not in cmd:
                cmd.append("--vae-tiling")

        # Offload text encoder / vision encoder to CPU RAM:
        # Only on GPUs with <= 12GB VRAM (e.g. RTX 3060 12GB) or when explicitly requested via offload_te_cpu.
        # On GPUs with >= 16GB VRAM, --flash-attn on allows ViT + LLM to run entirely on GPU in ~17s without OOM.
        needs_te_cpu = (
            gpu_vram <= 12288
            or getattr(request, "offload_te_cpu", False)
        )
        if needs_te_cpu and "--backend" not in cmd:
            cmd.extend(["--backend", "te=cpu"])

        # Parameter offloading for very constrained VRAM (<= 12GB)
        if gpu_vram <= 12288:
            if "--params-backend" not in cmd:
                cmd.extend(["--params-backend", "cpu"])

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
        crop_msg = f" (crop from {width}x{height} source)" if mask_ctx is not None else ""
        hires_msg = f" -> Hires {target_w}x{target_h} ({hires_upscaler}, steps={hires_steps}, strength={hires_strength})" if hires_fix else ""
        logger.info(f"[sd_cpp] Launching SD.CPP: {engine_width}x{engine_height}{crop_msg}{hires_msg}, steps={request.steps}, cfg={cfg_val}, sampler={mapped_sampler}, eta={noise_val}, seed={seed}")
        logger.debug(f"[sd_cpp] CLI args: {' '.join(cmd)}")

        # 4. Environment PATH
        env = os.environ.copy()
        sd_cpp_dir = get_sd_cpp_dir()
        from .engine import get_cuda_search_paths
        cuda_paths = get_cuda_search_paths()
        cuda_dirs = [
            str(sd_cpp_dir),
            str(settings.engine_dir),
        ] + [str(p) for p in cuda_paths if p.exists()]
        existing_path = env.get("PATH", "")
        if existing_path:
            cuda_dirs.append(existing_path)
        env["PATH"] = os.pathsep.join(cuda_dirs)
        if settings.huggingface_token:
            env["HF_TOKEN"] = settings.huggingface_token
            env["HUGGING_FACE_HUB_TOKEN"] = settings.huggingface_token

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
                    if sys.platform == "win32":
                        try:
                            subprocess.run(
                                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                capture_output=True,
                                check=False,
                            )
                        except Exception:
                            pass
                    proc.kill()
                    raise RuntimeError("Generation cancelled by user")

                # Filter out tensor loading / model weight transfer lines from step progress
                if "MB/s" in line_str or "GB/s" in line_str or "loading tensors" in line_str:
                    match = _STEP_PROGRESS_RE.search(line_str)
                    if match:
                        cur_t = match.group(1)
                        tot_t = match.group(2)
                        emit(ProgressEvent(
                            status=JobStatus.GENERATING,
                            progress=0.05,
                            message=f"Загрузка весов ({cur_t}/{tot_t})...",
                        ))
                    continue

                if "encode_first_stage" in line_str:
                    emit(ProgressEvent(
                        status=JobStatus.GENERATING,
                        progress=0.07,
                        message="Кодирование VAE входного изображения...",
                    ))
                    continue

                if "qwen3vl" in line_str.lower() or "get_learned_condition" in line_str.lower():
                    emit(ProgressEvent(
                        status=JobStatus.GENERATING,
                        progress=0.09,
                        message="Анализ изображения и промпта (Qwen-VL на GPU)...",
                    ))
                    continue

                if "generating image:" in line_str:
                    emit(ProgressEvent(
                        status=JobStatus.GENERATING,
                        progress=0.10,
                        step=1,
                        total_steps=total_steps,
                        message=f"Генерация: шаг 1/{total_steps} (расчет на GPU)...",
                    ))
                    continue

                if "decode_first_stage" in line_str or "decoding" in line_str:
                    emit(ProgressEvent(
                        status=JobStatus.GENERATING,
                        progress=0.96,
                        message="Декодирование VAE...",
                    ))
                    continue

                # Parse real sampling progress (Base pass & Highres Fix pass)
                step_match = _STEP_PROGRESS_RE.search(line_str)
                if step_match and ("s/it" in line_str or "it/s" in line_str or int(step_match.group(2)) in (total_steps, hires_steps)):
                    try:
                        cur_step = int(step_match.group(1))
                        tot_steps = int(step_match.group(2))
                        speed_str = ""
                        speed_match = re.search(r"(\d+(?:\.\d+)?(?:s/it|it/s))", line_str)
                        if speed_match:
                            speed_str = f" · {speed_match.group(1)}"

                        if hires_fix and tot_steps == hires_steps:
                            # Pass 2: Highres Fix pass
                            pass_pct = min(1.0, max(0.0, cur_step / tot_steps))
                            overall_pct = min(0.95, 0.50 + 0.45 * pass_pct)
                            emit(ProgressEvent(
                                status=JobStatus.GENERATING,
                                progress=overall_pct,
                                step=cur_step,
                                total_steps=tot_steps,
                                message=f"Highres Fix: шаг {cur_step}/{tot_steps} ({int(overall_pct*100)}%){speed_str}",
                            ))
                        else:
                            # Pass 1: Base generation
                            pass_pct = min(1.0, max(0.0, cur_step / tot_steps))
                            overall_pct = min(0.50, 0.10 + 0.40 * pass_pct) if hires_fix else min(0.95, 0.10 + 0.85 * pass_pct)
                            prefix = "Базовый кадр: шаг" if hires_fix else "Шаг"
                            emit(ProgressEvent(
                                status=JobStatus.GENERATING,
                                progress=overall_pct,
                                step=cur_step,
                                total_steps=tot_steps,
                                message=f"{prefix} {cur_step}/{tot_steps} ({int(overall_pct*100)}%){speed_str}",
                            ))
                    except ValueError:
                        pass
                else:
                    pct_match = _PERCENT_PROGRESS_RE.search(line_str)
                    if pct_match and ("sampling" in line_str.lower() or "%" in line_str):
                        try:
                            val = int(pct_match.group(1))
                            if val <= 100:
                                emit(ProgressEvent(
                                    status=JobStatus.GENERATING,
                                    progress=val / 100.0,
                                    message=f"Сэмплирование ({val}%)",
                                ))
                        except ValueError:
                            pass

            proc.wait()
        finally:
            self._current_process = None

        if proc.returncode != 0:
            error_lines = [
                l for l in captured_lines
                if any(k in l.lower() for k in ("[e]", "error:", "fatal:", "failed"))
                and not l.strip().lower().startswith("usage:")
                and "cli options:" not in l.lower()
            ]
            err_summary = "\n".join(error_lines[-10:]) if error_lines else ""
            tail = "\n".join(captured_lines[-20:]) if captured_lines else "(no output)"
            msg = f"SD.CPP process exited with code {proc.returncode}."
            if err_summary:
                msg += f"\n--- Engine Error Summary ---\n{err_summary}"
            msg += f"\n--- Engine output (last 20 lines) ---\n{tail}"
            raise RuntimeError(msg)

        if not output_path.exists():
            tail = "\n".join(captured_lines[-20:]) if captured_lines else "(no output)"
            raise RuntimeError(
                f"SD.CPP finished but output file not found: {output_path}\n"
                f"--- Engine output (last 20 lines) ---\n{tail}"
            )

        # ----- Mask compositing (guaranteed preservation) -----
        # The engine ran on the mask-bbox crop; paste the composited crop back
        # into the ORIGINAL frame.  Outside the crop the source stays
        # bit-identical.  The raw engine output is kept as the Inpaint fragment.
        # Soft-edge blending: apply 3px Gaussian blur to the mask before
        # compositing to avoid visible seams at crop boundaries.
        raw_fragment_rel = None
        if mask_ctx is not None:
            from PIL import Image
            from .masking import (
                composite_crop_into_image,
                paste_crop_into_image,
                prepare_compositing_mask,
            )

            try:
                with Image.open(output_path) as gen_file:
                    generated_rgb = gen_file.convert("RGB")
                source_image = mask_ctx["source_image"]

                # Solidify soft-brush interior and feather outer boundary so original SFX never bleeds through
                _blur_radius = float(getattr(request, "mask_composite_blur", 3.5) or 3.5)
                _soft_mask = prepare_compositing_mask(mask_ctx["crop_array"], blur=_blur_radius)

                composite = composite_crop_into_image(
                    generated_rgb,
                    source_image,
                    _soft_mask,
                    mask_ctx["box"],
                )
                composite.save(output_path, format="PNG")

                frag_dir = settings.output_dir / "masks"
                frag_dir.mkdir(parents=True, exist_ok=True)
                frag_path = frag_dir / f"{output_path.stem}_raw.png"
                paste_crop_into_image(
                    generated_rgb, source_image, mask_ctx["box"]
                ).save(frag_path, format="PNG")
                raw_fragment_rel = f"masks/{frag_path.name}"
                logger.info(
                    "[sd_cpp] Mask composite applied -> %s (crop %sx%s, fragment: %s)",
                    output_path.name,
                    mask_ctx["crop_size"][0],
                    mask_ctx["crop_size"][1],
                    raw_fragment_rel,
                )
            except Exception as comp_err:
                raise RuntimeError(f"Mask compositing failed: {comp_err}") from comp_err

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
                "native_ref_res": native_ref,
                "backend": "sd_cpp",
                "engine": "stable-diffusion.cpp",
                "owner_id": getattr(request, "owner_id", "host"),
                "owner_name": getattr(request, "owner_name", "Хост"),
                "shared": getattr(request, "shared", True),
                "generation_time_sec": round(elapsed, 2),
                "timestamp": int(time.time()),
            }
            if hires_fix:
                meta.update({
                    "hires_fix": True,
                    "hires_scale": hires_scale,
                    "hires_steps": hires_steps,
                    "hires_denoising_strength": hires_strength,
                    "hires_upscaler": hires_upscaler,
                    "base_width": base_w,
                    "base_height": base_h,
                    "hires_width": target_w,
                    "hires_height": target_h,
                })
            if mask_ctx is not None:
                meta.update({
                    "mask_applied": True,
                    "mask_strength": mask_strength_value,
                    "mask_grow": int(getattr(request, "mask_grow", 0) or 0),
                    "mask_blur": float(getattr(request, "mask_blur", 0.0) or 0.0),
                    "mask_source": mask_ctx["file"].name,
                    "mask_crop": {
                        "mode": mask_ctx.get("crop_mode", "auto"),
                        "box": list(mask_ctx["box"]),
                        "size": list(mask_ctx["crop_size"]),
                        "padding": mask_ctx.get("crop_padding"),
                        "min_size": mask_ctx.get("crop_min_size"),
                    },
                    "mask_raw_fragment": raw_fragment_rel,
                    "composite": "generated*mask + original*(1-mask)",
                })
            else:
                meta["mask_applied"] = False
            sidecar_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"[sd_cpp] Failed to write sidecar metadata: {e}")

        emit(ProgressEvent(
            status=JobStatus.COMPLETE,
            progress=1.0,
            image_path=str(output_path),
            raw_image_path=raw_fragment_rel,
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
