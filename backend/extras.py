"""
Managed "extras" runtime for the PyTorch-based engines that don't ship their own
binaries: **OrbitQuant** and **Nunchaku (vitoom fork)**.

Why a separate venv?  The Nunchaku wheels from ``tonera/vitoom-nunchaku`` are built
against an exact torch minor + CUDA (e.g. torch 2.12 + cu13.0) and won't load into
the SD.Next environment used by SDNQ (a different torch). So we keep an isolated
``envs/extras`` venv; users who already have a suitable environment can point
``settings.extras_python`` at it instead.

Nothing here runs automatically: installs are explicit, cancellable background jobs
with a streaming log.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from .settings import IS_WINDOWS, get_settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENVS_DIR = PROJECT_ROOT / "envs"
VENV_DIR = ENVS_DIR / "extras"
WHEELS_DIR = ENVS_DIR / "wheels"
NUNCHAKU_REPO = "tonera/vitoom-nunchaku"
ORBITQUANT_REQ = "orbitquant[hf,kernels]>=0.6.0"
_WHEEL_CACHE = PROJECT_ROOT / "cache" / "extras_wheels.json"

# Used when the wheel index can't be reached and no plan can be computed.
FALLBACK_TORCH = "2.12"
FALLBACK_CUDA_TAG = "cu130"

_WHEEL_RE = re.compile(r"^nunchaku-([^+]+)\+cu([\d.]+)torch([\d.]+)-(cp\d+)-cp\d+-([\w]+)\.whl$")


# ------------------------------------------------------------
# Environment discovery
# ------------------------------------------------------------
def venv_python() -> Path:
    return VENV_DIR / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def extras_python() -> Optional[Path]:
    """python executable used for OrbitQuant/Nunchaku jobs (override > managed venv)."""
    override = (get_settings().extras_python or "").strip().strip('"')
    if override:
        p = Path(override)
        return p if p.is_file() else None
    p = venv_python()
    return p if p.is_file() else None


_PROBE = r"""
import json, sys
out = {"python": sys.version.split()[0], "py_tag": "cp%d%d" % sys.version_info[:2]}
def _try(name, fn):
    try:
        out[name] = fn()
    except Exception as e:
        out[name] = None
        out[name + "_error"] = type(e).__name__ + ": " + str(e)[:200]
try:
    import torch
    out["torch"] = torch.__version__
    out["cuda"] = torch.version.cuda
    out["cuda_available"] = bool(torch.cuda.is_available())
    if out["cuda_available"]:
        out["device"] = torch.cuda.get_device_name(0)
        out["sm"] = "%d%d" % torch.cuda.get_device_capability(0)
except Exception as e:
    out["torch"] = None
    out["torch_error"] = type(e).__name__ + ": " + str(e)[:200]
def _diff():
    import diffusers
    v = diffusers.__version__
    out["has_flux2_klein"] = hasattr(diffusers, "Flux2KleinPipeline")
    out["has_zimage"] = hasattr(diffusers, "ZImagePipeline")
    return v
_try("diffusers", _diff)
def _nun():
    import nunchaku
    v = nunchaku.__version__
    try:
        from nunchaku.utils import get_precision
        out["nunchaku_precision"] = get_precision()
    except Exception as e:
        out["nunchaku_precision_error"] = str(e)[:120]
    return v
_try("nunchaku", _nun)
def _orb():
    import orbitquant
    return getattr(orbitquant, "__version__", "installed")
_try("orbitquant", _orb)
print("__PROBE__" + json.dumps(out))
"""

_probe_cache: dict = {"key": None, "t": 0.0, "value": None}


def inspect_env(force: bool = False) -> dict:
    """Probe the extras python (cached ~60s). Never raises."""
    py = extras_python()
    base = {
        "python_path": str(py) if py else None,
        "managed": not bool((get_settings().extras_python or "").strip()),
        "venv_exists": venv_python().is_file(),
        "ready": False,
        "torch": None, "cuda": None, "cuda_available": False, "device": None,
        "diffusers": None, "has_flux2_klein": False, "has_zimage": False,
        "nunchaku": None, "nunchaku_precision": None, "orbitquant": None,
        "py_tag": None, "error": None,
    }
    if not py:
        base["error"] = "Extras environment not installed"
        return base

    key = str(py)
    if not force and _probe_cache["key"] == key and time.time() - _probe_cache["t"] < 60 and _probe_cache["value"]:
        return dict(_probe_cache["value"])

    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if IS_WINDOWS else 0
        res = subprocess.run([str(py), "-c", _PROBE], capture_output=True, text=True, timeout=120,
                             creationflags=flags)
        line = next((l for l in res.stdout.splitlines() if l.startswith("__PROBE__")), None)
        if not line:
            base["error"] = (res.stderr or res.stdout or "probe produced no output").strip()[-300:]
        else:
            data = json.loads(line[len("__PROBE__"):])
            base.update({k: v for k, v in data.items() if v is not None or k in base})
            base["ready"] = bool(data.get("torch") and data.get("diffusers") and data.get("cuda_available"))
            if not base["ready"]:
                base["error"] = data.get("torch_error") or (
                    "CUDA is not available in this environment" if data.get("torch") else "torch/diffusers missing")
    except Exception as e:
        base["error"] = f"{type(e).__name__}: {e}"

    _probe_cache.update(key=key, t=time.time(), value=dict(base))
    return base


def invalidate_probe() -> None:
    _probe_cache.update(key=None, t=0.0, value=None)


def is_orbitquant_ready() -> bool:
    e = inspect_env()
    return bool(e["ready"] and e["orbitquant"])


def is_nunchaku_ready() -> bool:
    e = inspect_env()
    return bool(e["ready"] and e["nunchaku"])


# ------------------------------------------------------------
# Nunchaku wheel planning
# ------------------------------------------------------------
def _version_key(ver: str):
    nums = tuple(int(x) for x in re.findall(r"\d+", ver.split(".dev")[0])[:3])
    is_dev = ".dev" in ver
    dev = int(re.search(r"dev(\d+)", ver).group(1)) if is_dev and re.search(r"dev(\d+)", ver) else 0
    return (nums, 0 if is_dev else 1, dev)


def _platform_tag() -> str:
    import platform
    if IS_WINDOWS:
        return "win_amd64"
    return "linux_aarch64" if platform.machine().lower() in ("aarch64", "arm64") else "linux_x86_64"


def list_wheel_files(refresh: bool = False) -> tuple[list[str], Optional[str]]:
    """Wheel filenames in the vitoom-nunchaku repo (cached on disk for offline use)."""
    err = None
    if not refresh and _WHEEL_CACHE.is_file() and time.time() - _WHEEL_CACHE.stat().st_mtime < 6 * 3600:
        try:
            return json.loads(_WHEEL_CACHE.read_text(encoding="utf-8")), None
        except Exception:
            pass
    try:
        from huggingface_hub import HfApi
        files = [f for f in HfApi(token=get_settings().huggingface_token or None).list_repo_files(NUNCHAKU_REPO)
                 if f.endswith(".whl")]
        _WHEEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _WHEEL_CACHE.write_text(json.dumps(files), encoding="utf-8")
        return files, None
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    if _WHEEL_CACHE.is_file():
        try:
            return json.loads(_WHEEL_CACHE.read_text(encoding="utf-8")), err
        except Exception:
            pass
    return [], err


def plan_nunchaku_wheel(py_tag: Optional[str] = None, files: Optional[list[str]] = None) -> dict:
    """Pick the best wheel for this OS + python. Pure function when `files` is given."""
    if py_tag is None:
        env = inspect_env() if extras_python() else {}
        py_tag = env.get("py_tag") or f"cp{sys.version_info.major}{sys.version_info.minor}"
    list_err = None
    if files is None:
        files, list_err = list_wheel_files()

    plat = _platform_tag()
    cands = []
    for f in files:
        m = _WHEEL_RE.match(f)
        if not m:
            continue
        ver, cu, torch_v, py, p = m.groups()
        if py == py_tag and p == plat:
            cands.append((f, ver, cu, torch_v))
    if not cands:
        return {
            "wheel": None, "py_tag": py_tag, "platform": plat,
            "torch_pin": FALLBACK_TORCH, "cuda_tag": FALLBACK_CUDA_TAG,
            "reason": (f"No Nunchaku wheel for {plat} + {py_tag}" +
                       (f" ({list_err})" if list_err else ". Use Python 3.11-3.13 on Windows."))
        }
    cands.sort(key=lambda c: (_version_key(c[1]), tuple(int(x) for x in c[3].split(".")), float(c[2])), reverse=True)
    f, ver, cu, torch_v = cands[0]
    return {
        "wheel": f, "version": ver, "py_tag": py_tag, "platform": plat,
        "torch_pin": torch_v, "cuda_tag": "cu" + cu.replace(".", ""),
        "reason": f"Nunchaku {ver} needs torch {torch_v} + CUDA {cu}",
    }


# ------------------------------------------------------------
# Install jobs
# ------------------------------------------------------------
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()
_PROCS: dict[str, subprocess.Popen] = {}
_MAX_LOG = 500


def _log(job: dict, line: str) -> None:
    job["log"].append(line.rstrip())
    if len(job["log"]) > _MAX_LOG:
        del job["log"][: len(job["log"]) - _MAX_LOG]


class _Cancelled(Exception):
    pass


def _run(job: dict, cmd: list[str], env: Optional[dict] = None) -> int:
    if job["cancel"].is_set():
        raise _Cancelled()
    _log(job, "$ " + " ".join(cmd))
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if IS_WINDOWS else 0
    full_env = {**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PYTHONIOENCODING": "utf-8", **(env or {})}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1, env=full_env, creationflags=flags)
    _PROCS[job["id"]] = proc
    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            _log(job, line)
            if job["cancel"].is_set():
                proc.terminate()
                raise _Cancelled()
        return proc.wait()
    finally:
        _PROCS.pop(job["id"], None)


def _step(job: dict, label: str) -> None:
    job["step"] += 1
    job["step_label"] = label
    _log(job, f"=== [{job['step']}/{job['steps_total']}] {label} ===")


def _pip(job: dict, py: Path, *args: str) -> None:
    if _run(job, [str(py), "-m", "pip", *args]) != 0:
        raise RuntimeError(f"pip {' '.join(args[:3])} ... failed (see log)")


def _ensure_runtime(job: dict, plan: dict) -> Path:
    """Create the venv + torch + diffusers stack if the extras env isn't ready."""
    if (get_settings().extras_python or "").strip():
        py = extras_python()
        if not py:
            raise RuntimeError("settings.extras_python points to a missing file")
        return py

    py = venv_python()
    if not py.is_file():
        _step(job, "Creating virtual environment")
        ENVS_DIR.mkdir(parents=True, exist_ok=True)
        if _run(job, [sys.executable, "-m", "venv", str(VENV_DIR)]) != 0:
            raise RuntimeError("venv creation failed")
        _pip(job, py, "install", "--upgrade", "pip", "wheel", "setuptools")

    env = inspect_env(force=True)
    if not env.get("torch") or not env.get("cuda_available"):
        _step(job, f"Installing PyTorch {plan['torch_pin']} ({plan['cuda_tag']})")
        _pip(job, py, "install", f"torch=={plan['torch_pin']}.*", "torchvision",
             "--index-url", f"https://download.pytorch.org/whl/{plan['cuda_tag']}")
    if not env.get("diffusers"):
        _step(job, "Installing diffusers / transformers / accelerate / peft")
        _pip(job, py, "install", "diffusers", "transformers", "accelerate", "safetensors",
             "sentencepiece", "protobuf", "huggingface_hub", "pillow", "numpy", "peft")
    env = inspect_env(force=True)
    if env.get("diffusers") and not env.get("has_flux2_klein"):
        # FLUX.2 Klein support landed after some releases; fall back to main if git is around.
        _log(job, "diffusers has no Flux2KleinPipeline yet; trying GitHub main (needs git)...")
        if _run(job, [str(py), "-m", "pip", "install", "git+https://github.com/huggingface/diffusers.git"]) != 0:
            _log(job, "WARNING: could not install diffusers from git. FLUX.2 Klein models need a newer diffusers.")
    invalidate_probe()
    return py


def _count_steps(component: str, need_runtime: bool) -> int:
    n = 3 if need_runtime else 0
    return n + {"runtime": 1, "nunchaku": 3, "orbitquant": 2}.get(component, 1)


def _job_main(job: dict, component: str) -> None:
    try:
        plan = plan_nunchaku_wheel()
        job["plan"] = plan
        if component == "nunchaku" and not plan.get("wheel"):
            raise RuntimeError(plan["reason"])

        job["steps_total"] = _count_steps(component, need_runtime=not inspect_env().get("ready"))
        job["state"] = "running"
        py = _ensure_runtime(job, plan)

        if component == "nunchaku":
            _step(job, f"Downloading {plan['wheel']}")
            from huggingface_hub import hf_hub_download
            WHEELS_DIR.mkdir(parents=True, exist_ok=True)
            wheel = hf_hub_download(NUNCHAKU_REPO, plan["wheel"], local_dir=str(WHEELS_DIR),
                                    token=get_settings().huggingface_token or None)
            _step(job, "Installing Nunchaku wheel")
            _pip(job, py, "install", wheel)
            _step(job, "Verifying")
            invalidate_probe()
            env = inspect_env(force=True)
            if not env.get("nunchaku"):
                raise RuntimeError("Nunchaku installed but import failed: " + str(env.get("nunchaku_error")))
            _log(job, f"OK: nunchaku {env['nunchaku']} (precision {env.get('nunchaku_precision')})")

        elif component == "orbitquant":
            _step(job, "Installing OrbitQuant")
            _pip(job, py, "install", ORBITQUANT_REQ)
            _step(job, "Provisioning native kernels (optional)")
            cli = py.parent / ("orbitquant.exe" if IS_WINDOWS else "orbitquant")
            if cli.is_file():
                if _run(job, [str(cli), "kernels-install"]) != 0:
                    _log(job, "Kernel provisioning failed; OrbitQuant will fall back to the Triton path.")
            invalidate_probe()
            env = inspect_env(force=True)
            if not env.get("orbitquant"):
                raise RuntimeError("OrbitQuant installed but import failed: " + str(env.get("orbitquant_error")))
            _log(job, f"OK: orbitquant {env['orbitquant']}")

        else:  # runtime only
            _step(job, "Verifying")
            env = inspect_env(force=True)
            if not env.get("ready"):
                raise RuntimeError(env.get("error") or "Runtime not ready")
            _log(job, f"OK: torch {env['torch']} (CUDA {env['cuda']}), diffusers {env['diffusers']}")

        job["state"] = "done"
    except _Cancelled:
        job["state"] = "cancelled"
        _log(job, "Cancelled by user")
    except Exception as e:
        job["state"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"
        _log(job, "ERROR: " + job["error"])
    finally:
        job["ended"] = time.time()
        invalidate_probe()


def _public(job: dict) -> dict:
    return {k: v for k, v in job.items() if k not in ("cancel",)}


def start_install(component: str) -> dict:
    if component not in ("runtime", "nunchaku", "orbitquant"):
        raise ValueError("component must be runtime | nunchaku | orbitquant")
    with _JOBS_LOCK:
        if any(j["state"] in ("queued", "running") for j in _JOBS.values()):
            raise RuntimeError("Another install is already running")
        job = {
            "id": uuid.uuid4().hex[:8], "component": component, "state": "queued", "step": 0,
            "steps_total": 1, "step_label": "Queued", "log": [], "error": None,
            "started": time.time(), "ended": None, "plan": None, "cancel": threading.Event(),
        }
        _JOBS[job["id"]] = job
    threading.Thread(target=_job_main, args=(job, component), daemon=True).start()
    return _public(job)


def get_job(job_id: str) -> Optional[dict]:
    j = _JOBS.get(job_id)
    return _public(j) if j else None


def cancel_job(job_id: str) -> bool:
    j = _JOBS.get(job_id)
    if not j or j["state"] not in ("queued", "running"):
        return False
    j["cancel"].set()
    p = _PROCS.get(job_id)
    if p and p.poll() is None:
        try:
            p.terminate()
        except Exception:
            pass
    return True


def latest_job() -> Optional[dict]:
    if not _JOBS:
        return None
    return _public(max(_JOBS.values(), key=lambda j: j["started"]))


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
def register_extras_routes(app) -> None:
    import asyncio

    from fastapi import HTTPException

    @app.get("/api/extras/status")
    async def extras_status(refresh: bool = False):
        env = await asyncio.to_thread(inspect_env, refresh)
        plan = await asyncio.to_thread(plan_nunchaku_wheel)
        return {
            "env": env,
            "nunchaku_ready": bool(env.get("nunchaku")),
            "orbitquant_ready": bool(env.get("orbitquant")),
            "nunchaku_plan": plan,
            "job": latest_job(),
        }

    @app.post("/api/extras/install")
    async def extras_install(body: dict):
        try:
            return start_install(str((body or {}).get("component", "")))
        except ValueError as e:
            raise HTTPException(400, str(e))
        except RuntimeError as e:
            raise HTTPException(409, str(e))

    @app.get("/api/extras/jobs/{job_id}")
    async def extras_job(job_id: str):
        j = get_job(job_id)
        if not j:
            raise HTTPException(404, "Job not found")
        return j

    @app.post("/api/extras/jobs/{job_id}/cancel")
    async def extras_job_cancel(job_id: str):
        return {"cancelled": cancel_job(job_id)}

    @app.post("/api/extras/python")
    async def extras_set_python(body: dict):
        """Point the extras runtime at an existing python.exe (or clear with empty path)."""
        path = str((body or {}).get("path") or "").strip().strip('"')
        if path:
            p = Path(path)
            if not p.is_file() or "python" not in p.name.lower():
                raise HTTPException(400, "Path must be an existing python executable")
        get_settings().update(extras_python=path or None)
        invalidate_probe()
        return {"env": await asyncio.to_thread(inspect_env, True)}
