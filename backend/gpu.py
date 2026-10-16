"""
GPU detection — capability-based, NOT hardcoded GPU list.

Mirrors the logic from ComfyUI-QuantFunc plugin's lib_setup._detect_gpu_sm
(file: lib_setup.py:388-413). We do NOT depend on the plugin — this is a
standalone reimplementation using nvidia-smi as primary, torch as fallback.

The result is a single integer SM (compute capability) in the format
major * 10 + minor. Examples:
    RTX 2060      → 75
    RTX 3070      → 86
    RTX 4090      → 89
    RTX 5060 Ti   → 120
    RTX 5090      → 120
    A100          → 80
    H100          → 90
    B200          → 120

Based on SM, we determine the GPU "tier":
    SM >= 120  → "50x-above"  (Blackwell + future — FP4 native)
    SM < 120   → "50x-below"  (Turing, Ampere, Ada, Hopper, etc.)

For Klein models (Flux.2 Klein 4B/9B), there's an additional per-file
klein_tier: "30x-below" / "40x" / "50x".
"""
from __future__ import annotations

import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------- Tier constants ----------
TIER_50X_ABOVE = "50x-above"
TIER_50X_BELOW = "50x-below"


# ---------- Klein per-file tiers ----------
KLEIN_TIER_30X_BELOW = "30x-below"   # RTX 20, RTX 30 (Turing, Ampere) — SM 75-86
KLEIN_TIER_40X = "40x"               # RTX 40 (Ada Lovelace) — SM 89
KLEIN_TIER_50X = "50x"               # RTX 50+, H100, H200, B100, B200 — SM >= 90


@dataclass
class GPUInfo:
    """Detected GPU information."""
    name: str                           # "NVIDIA GeForce RTX 5060 Ti"
    sm: int                             # 120
    vram_mb: int                        # 16384
    driver_version: str                 # "560.94"
    cuda_version: str                   # "13.0"
    tier: str                           # "50x-above" or "50x-below"
    klein_tier: str                     # "30x-below" / "40x" / "50x"
    is_nvidia: bool = True
    detection_method: str = "nvidia-smi"  # "nvidia-smi" or "torch"

    @property
    def vram_gb(self) -> float:
        return self.vram_mb / 1024.0

    @property
    def tier_label(self) -> str:
        """Human-readable tier label for UI."""
        if self.tier == TIER_50X_ABOVE:
            return "RTX 50-series+ (Blackwell) — FP4 native"
        return "RTX 30/40 + A100/H100 — INT4 / FP4 emulation"

    @property
    def supports_fp4_native(self) -> bool:
        """Blackwell SM >= 120 has native FP4 tensor cores."""
        return self.sm >= 120

    @property
    def supports_bf16(self) -> bool:
        """RTX 20 (Turing, SM 75) does NOT support BF16."""
        return self.sm >= 80  # Ampere+ supports BF16


# ---------- nvidia-smi parsing ----------
_NVSMI_PATH_CANDIDATES = [
    "nvidia-smi",  # In PATH
    # Windows fallbacks
    r"C:\Windows\System32\nvidia-smi.exe",
    r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
]


def _find_nvidia_smi() -> Optional[str]:
    """Find nvidia-smi executable."""
    for candidate in _NVSMI_PATH_CANDIDATES:
        try:
            result = subprocess.run(
                [candidate, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def _parse_sm_from_nvsmi(sm_output: str) -> Optional[int]:
    """Parse '12.0' or '12' → 120."""
    sm_output = sm_output.strip()
    if not sm_output:
        return None
    # Format: "12.0" or "8.9" or "7.5"
    match = re.match(r"^(\d+)\.(\d+)$", sm_output)
    if match:
        major, minor = int(match.group(1)), int(match.group(2))
        return major * 10 + minor
    # Plain integer: "120"
    if sm_output.isdigit():
        return int(sm_output)
    return None


def _detect_via_nvidia_smi() -> Optional[GPUInfo]:
    """Primary detection method — nvidia-smi subprocess."""
    nvsmi = _find_nvidia_smi()
    if not nvsmi:
        return None

    try:
        # Query all needed fields in one call
        result = subprocess.run(
            [nvsmi,
             "--query-gpu=name,compute_cap,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None

        # Output: "NVIDIA GeForce RTX 5060 Ti, 12.0, 16384, 560.94"
        # Note: values are comma-separated, no units because of `nounits`
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) < 4:
            return None

        name = parts[0]
        sm = _parse_sm_from_nvsmi(parts[1])
        vram_mb = int(parts[2]) if parts[2].isdigit() else 0
        driver_version = parts[3]

        # CUDA version from nvidia-smi header (separate call)
        cuda_version = ""
        try:
            header_result = subprocess.run(
                [nvsmi, "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            # The CUDA version appears in the header line of full `nvidia-smi` output
            # We can get it from the standard nvidia-smi command output
            full_result = subprocess.run(
                [nvsmi],
                capture_output=True, text=True, timeout=10,
            )
            cuda_match = re.search(r"CUDA Version:\s*([\d.]+)", full_result.stdout)
            if cuda_match:
                cuda_version = cuda_match.group(1)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        if sm is None:
            return None

        return _build_gpu_info(
            name=name, sm=sm, vram_mb=vram_mb,
            driver_version=driver_version,
            cuda_version=cuda_version,
            detection_method="nvidia-smi",
        )

    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        return None


def _detect_via_torch() -> Optional[GPUInfo]:
    """Fallback detection — PyTorch CUDA. Requires torch installed."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None

        device = torch.cuda.current_device()
        name = torch.cuda.get_device_name(device)
        major, minor = torch.cuda.get_device_capability(device)
        sm = major * 10 + minor

        # VRAM via torch
        props = torch.cuda.get_device_properties(device)
        vram_mb = props.total_memory // (1024 * 1024)

        # Driver version
        driver_version = torch.version.cuda or ""

        return _build_gpu_info(
            name=name, sm=sm, vram_mb=vram_mb,
            driver_version=driver_version,
            cuda_version=driver_version,
            detection_method="torch",
        )
    except ImportError:
        return None
    except Exception:
        return None


def _build_gpu_info(
    name: str, sm: int, vram_mb: int,
    driver_version: str, cuda_version: str,
    detection_method: str,
) -> GPUInfo:
    """Build GPUInfo with computed tier + klein_tier."""
    if sm >= 120:
        tier = TIER_50X_ABOVE
    else:
        tier = TIER_50X_BELOW

    # Klein per-file tier — based on SM ranges
    if sm < 89:
        klein_tier = KLEIN_TIER_30X_BELOW
    elif sm < 90:
        klein_tier = KLEIN_TIER_40X
    else:
        klein_tier = KLEIN_TIER_50X

    return GPUInfo(
        name=name,
        sm=sm,
        vram_mb=vram_mb,
        driver_version=driver_version,
        cuda_version=cuda_version,
        tier=tier,
        klein_tier=klein_tier,
        is_nvidia=True,
        detection_method=detection_method,
    )


# ---------- Public API ----------
def detect_gpu() -> Optional[GPUInfo]:
    """Detect GPU. Tries nvidia-smi first, then torch.

    Returns None if no NVIDIA GPU is detected.
    """
    # Primary: nvidia-smi (no Python deps required)
    info = _detect_via_nvidia_smi()
    if info is not None:
        return info

    # Fallback: torch (requires torch installed)
    info = _detect_via_torch()
    if info is not None:
        return info

    # No GPU detected
    return None


def klein_tier_for_sm(sm: int) -> str:
    """Map SM integer to klein per-file tier."""
    if sm < 89:
        return KLEIN_TIER_30X_BELOW
    elif sm < 90:
        return KLEIN_TIER_40X
    else:
        return KLEIN_TIER_50X


def tier_for_sm(sm: int) -> str:
    """Map SM integer to 50x-above / 50x-below tier."""
    return TIER_50X_ABOVE if sm >= 120 else TIER_50X_BELOW


def is_model_compatible(model_min_sm: int, gpu_sm: int) -> bool:
    """Check if a model's minimum SM requirement is satisfied by current GPU."""
    return gpu_sm >= model_min_sm


_CACHED_NVSMI_EXE: Optional[str] = None


def _cpu_percent() -> tuple[float, int]:
    """Non-blocking system-wide CPU utilisation (%), logical core count."""
    try:
        import psutil
        return float(psutil.cpu_percent(interval=None)), int(psutil.cpu_count(logical=True) or 0)
    except Exception:
        return 0.0, 0


def _collect_system_stats() -> dict:
    """Return live VRAM, System RAM, CPU and GPU load statistics."""
    global _CACHED_NVSMI_EXE
    import ctypes
    import sys

    cpu_pct, cpu_cores = _cpu_percent()
    stats = {
        "vram_used_mb": 0,
        "vram_total_mb": 0,
        "vram_used_gb": 0.0,
        "vram_total_gb": 0.0,
        "vram_percent": 0,
        "gpu_util": 0,
        "gpu_temp_c": None,
        "gpu_power_w": None,
        "cpu_percent": round(cpu_pct, 1),
        "cpu_cores": cpu_cores,
        "ram_used_mb": 0,
        "ram_total_mb": 0,
        "ram_used_gb": 0.0,
        "ram_total_gb": 0.0,
        "ram_percent": 0,
    }

    # 1. System RAM via Windows GlobalMemoryStatusEx (zero external dependencies)
    if sys.platform == "win32":
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            mem = MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
                total_mb = int(mem.ullTotalPhys // (1024 * 1024))
                avail_mb = int(mem.ullAvailPhys // (1024 * 1024))
                used_mb = max(0, total_mb - avail_mb)
                stats["ram_total_mb"] = total_mb
                stats["ram_used_mb"] = used_mb
                stats["ram_total_gb"] = round(total_mb / 1024.0, 1)
                stats["ram_used_gb"] = round(used_mb / 1024.0, 1)
                stats["ram_percent"] = int(mem.dwMemoryLoad)
        except Exception:
            pass
    else:
        try:
            meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
            m_tot = re.search(r"MemTotal:\s+(\d+)\s+kB", meminfo)
            m_av = re.search(r"MemAvailable:\s+(\d+)\s+kB", meminfo)
            if m_tot and m_av:
                total_mb = int(m_tot.group(1)) // 1024
                avail_mb = int(m_av.group(1)) // 1024
                used_mb = max(0, total_mb - avail_mb)
                stats["ram_total_mb"] = total_mb
                stats["ram_used_mb"] = used_mb
                stats["ram_total_gb"] = round(total_mb / 1024.0, 1)
                stats["ram_used_gb"] = round(used_mb / 1024.0, 1)
                stats["ram_percent"] = int(round(used_mb * 100 / max(1, total_mb)))
        except Exception:
            pass

    # 2. GPU VRAM via nvidia-smi
    if _CACHED_NVSMI_EXE is None:
        _CACHED_NVSMI_EXE = _find_nvidia_smi() or ""

    if _CACHED_NVSMI_EXE:
        try:
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
            res = subprocess.run(
                [
                    _CACHED_NVSMI_EXE,
                    "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
                creationflags=creationflags,
            )
            if res.returncode == 0 and res.stdout.strip():
                line = res.stdout.strip().splitlines()[0]
                parts = [p.strip() for p in line.split(",")]

                def _num(i: int):
                    # nvidia-smi prints "[N/A]" / "[Not Supported]" for unsupported fields
                    try:
                        return float(parts[i])
                    except (IndexError, ValueError):
                        return None

                if len(parts) >= 2:
                    v_used = int(_num(0) or 0)
                    v_tot = int(_num(1) or 0)
                    v_util = int(_num(2) or 0)
                    stats["vram_used_mb"] = v_used
                    stats["vram_total_mb"] = v_tot
                    stats["vram_used_gb"] = round(v_used / 1024.0, 1)
                    stats["vram_total_gb"] = round(v_tot / 1024.0, 1)
                    stats["vram_percent"] = int(round(v_used * 100 / max(1, v_tot))) if v_tot > 0 else 0
                    stats["gpu_util"] = v_util
                    temp = _num(3)
                    power = _num(4)
                    stats["gpu_temp_c"] = int(temp) if temp is not None else None
                    stats["gpu_power_w"] = round(power) if power is not None else None
        except Exception:
            pass

    return stats


_STATS_CACHE: dict = {"t": 0.0, "value": None}
_STATS_LOCK = threading.Lock()
_STATS_TTL_SEC = 1.0


def get_system_memory_stats() -> dict:
    """Live VRAM / RAM / CPU / GPU stats (cached ~1s so many pollers stay cheap)."""
    with _STATS_LOCK:
        now = time.monotonic()
        if _STATS_CACHE["value"] is not None and now - _STATS_CACHE["t"] < _STATS_TTL_SEC:
            return dict(_STATS_CACHE["value"])
        value = _collect_system_stats()
        _STATS_CACHE.update(t=now, value=value)
        return dict(value)


# Prime psutil's CPU counter so the first real reading isn't a meaningless 0.0.
_cpu_percent()

