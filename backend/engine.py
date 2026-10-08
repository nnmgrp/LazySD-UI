"""
Engine binary manager — downloads libquantfunc.so / quantfunc.dll from
ModelScope QuantFunc/Plugin, verifies SHA-256 against verify.json,
implements verify-before-replace self-healing logic.

Mirrors the behavior of ComfyUI-QuantFunc plugin's auto_update.py,
but as a standalone module (no ComfyUI dependency).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from .settings import IS_WINDOWS, PLATFORM_NAME, get_settings


# ---------- Constants ----------
MODELSCOPE_PLUGIN_API = "https://www.modelscope.cn/api/v1/models/QuantFunc/Plugin"
MODELSCOPE_PLUGIN_FILE = "https://www.modelscope.cn/api/v1/models/QuantFunc/Plugin/repo"

# ComfyUI-QuantFunc plugin repo — ships bin/<os>/config.json with the shared
# free-trial test API key (see quantfunc.com/docs/model-loading-and-apikey §6.2).
PLUGIN_RAW_BASE = "https://raw.githubusercontent.com/RealJonathanYip/ComfyUI-QuantFunc/main"

# Fallback bundled test config — identical to the plugin's bin/<os>/config.json.
# Used ONLY when the GitHub download fails (offline). During the free trial this
# shared key works out of the box; after the official release register your own
# key at https://www.quantfunc.com and save it in Settings → QuantFunc API key.
BUNDLED_TEST_CONFIG = {
    "server_url": "https://service.quantfunc.com",
    "api_key": "qf_4a256a9dfbfd04fd6fcf6bfc48781557aa1fed478d7a3c62214e91d0fbebba95",
}

# Default engine version — kept in sync with plugin v0.0.07 / ModelScope latest
DEFAULT_ENGINE_VERSION = "0.0.17"

# Mapping (platform, cuda_version) → (binary_filename, cli_filename)
ENGINE_FILENAMES = {
    ("windows", "13"): ("quantfunc.dll", "quantfunc.exe"),
    ("windows", "12"): ("quantfunc-12.dll", "quantfunc-12.exe"),
    ("linux",   "13"): ("libquantfunc.so", "quantfunc"),
    ("linux",   "12"): ("libquantfunc-12.so", "quantfunc-12"),
}

# Mapping (platform, cuda_version) → CUDA dependency bundle filename on ModelScope
# These are ~1-1.6 GB zip files containing cublas64_XX.dll, cudnn64_9.dll, etc.
CUDA_DEP_FILENAMES = {
    ("windows", "13"): "cu13-dep-win32.zip",
    ("windows", "12"): "cu12-dep-win32.zip",
    ("linux",   "13"): "cu13-dep-linux.zip",
    ("linux",   "12"): "cu12-dep-linux.zip",
}

# Critical CUDA DLLs that must be present for the engine to start.
# Uses glob patterns because NVIDIA uses DIFFERENT version suffixes for different
# libraries within the same CUDA Toolkit install:
#   - cublas64_13.dll, cublasLt64_13.dll, cudart64_13.dll  (v13)
#   - curand64_10.dll                                       (v10 — older lib)
#   - cusolver64_12.dll, cusparse64_12.dll                  (v12 — older lib)
#   - nvJitLink_130_0.dll                                   (different naming scheme)
#
# cuDNN is SEPARATE: it's not part of CUDA Toolkit and is OPTIONAL —
# engine will tell us if it really needs it. We warn, but don't block.
CUDA_CRITICAL_DLL_PATTERNS = {
    ("windows", "13"): [
        "cublas64_*.dll",          # cublas64_13.dll
        "cublasLt64_*.dll",        # cublasLt64_13.dll
        "cudart64_*.dll",          # cudart64_13.dll
        "curand64_*.dll",          # curand64_10.dll (yes, _10 in CUDA 13)
        "cusolver64_*.dll",        # cusolver64_12.dll (yes, _12 in CUDA 13)
        "cusparse64_*.dll",        # cusparse64_12.dll
        "nvJitLink_*.dll",         # nvJitLink_130_0.dll (different scheme)
    ],
    ("windows", "12"): [
        "cublas64_*.dll",
        "cublasLt64_*.dll",
        "cudart64_*.dll",
        "curand64_*.dll",
        "cusolver64_*.dll",
        "cusparse64_*.dll",
        "nvJitLink_*.dll",
    ],
    ("linux",   "13"): [
        "libcublas.so.*",
        "libcudart.so.*",
        "libcurand.so.*",
        "libcusolver.so.*",
        "libcusparse.so.*",
    ],
    ("linux",   "12"): [
        "libcublas.so.*",
        "libcudart.so.*",
        "libcurand.so.*",
        "libcusolver.so.*",
        "libcusparse.so.*",
    ],
}

# Optional DLLs — warn if missing but don't block (engine may not need them)
CUDA_OPTIONAL_DLL_PATTERNS = {
    ("windows", "13"): ["cudnn64_*.dll"],     # cuDNN — separate download
    ("windows", "12"): ["cudnn64_*.dll"],
    ("linux",   "13"): ["libcudnn.so.*"],
    ("linux",   "12"): ["libcudnn.so.*"],
}


@dataclass
class VerifyResult:
    """Result of verifying a local engine binary against remote verify.json."""
    status: str  # "ok" | "outdated" | "corrupted" | "missing" | "offline"
    local_sha256: Optional[str] = None
    remote_sha256: Optional[str] = None
    remote_version: Optional[str] = None
    local_path: Optional[Path] = None
    message: str = ""

    @property
    def is_ok(self) -> bool:
        return self.status == "ok"

    @property
    def needs_download(self) -> bool:
        return self.status in ("outdated", "corrupted", "missing")


# ---------- Helpers ----------
def _compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 of a file (chunked, memory-efficient)."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _fetch_plugin_json(rel_path: str, timeout: int = 10) -> Optional[dict]:
    """Fetch a JSON file from the ModelScope QuantFunc/Plugin repo."""
    url = f"{MODELSCOPE_PLUGIN_FILE}?Revision=master&FilePath={rel_path}"
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else None
    except (requests.RequestException, json.JSONDecodeError, ValueError) as e:
        print(f"[engine] Failed to fetch {rel_path}: {e}")
        return None


def _fetch_verify_json(engine_version: str, timeout: int = 10) -> Optional[dict]:
    """Fetch verify.json for a specific engine version from ModelScope."""
    return _fetch_plugin_json(f"{engine_version}/verify.json", timeout=timeout)


def _version_key(version: str) -> tuple:
    """'0.0.17' -> (0, 0, 17). Non-numeric versions sort first."""
    try:
        return tuple(int(part) for part in version.split("."))
    except (ValueError, AttributeError):
        return (-1,)


def _manifest_platform_key() -> str:
    """version.json / verify.json use 'win32' and 'linux' as platform keys."""
    return "win32" if get_settings().engine_platform == "windows" else "linux"


def fetch_engine_manifest(timeout: int = 10) -> Optional[dict]:
    """Fetch version.json — the list of published engine versions per platform."""
    return _fetch_plugin_json("version.json", timeout=timeout)


def get_latest_engine_version(manifest: Optional[dict] = None) -> Optional[str]:
    """Newest engine version published for the current platform (None if offline)."""
    manifest = manifest if manifest is not None else fetch_engine_manifest()
    if not manifest:
        return None
    entries = manifest.get(_manifest_platform_key())
    if not isinstance(entries, dict):
        return None
    versions = [v for v in entries if _version_key(v) != (-1,)]
    return max(versions, key=_version_key) if versions else None


def _platform_hashes(verify_json: dict) -> dict:
    """Return the {relative_path: sha256} map for the current platform.

    verify.json shape: {"schema": 1, "win32": {"sm120a/quantfunc.dll": "<sha>"}, "linux": {...}}
    """
    hashes = verify_json.get(_manifest_platform_key())
    return hashes if isinstance(hashes, dict) else {}


def _select_kernel_set(sets_json: Optional[dict], gpu_sm: Optional[int]) -> Optional[str]:
    """Pick the per-architecture build folder (e.g. 'sm120a') for a GPU SM.

    Engine >= 0.0.13 ships one build per architecture, described by sets.json:
        {"schema": 2, "sets": {"sm75": [75], "sm86": [86], "sm89": [89], "sm120a": [120]}}
    Returns None for legacy flat layouts (<= 0.0.12) that have no sets.json.
    """
    if not sets_json or not isinstance(sets_json.get("sets"), dict):
        return None
    sets = {name: sms for name, sms in sets_json["sets"].items() if isinstance(sms, list) and sms}
    if not sets:
        return None
    if gpu_sm is None:
        raise RuntimeError(
            "GPU compute capability is unknown, but this engine version ships "
            "per-architecture builds. Re-detect the GPU and try again."
        )
    for name, sms in sets.items():
        if gpu_sm in sms:
            return name
    # No exact match — fall back to the newest build not newer than the GPU
    lower = [(max(sms), name) for name, sms in sets.items() if max(sms) <= gpu_sm]
    if lower:
        chosen = max(lower)[1]
        print(f"[engine] WARNING: no exact engine build for SM {gpu_sm}; using '{chosen}'")
        return chosen
    raise RuntimeError(f"No engine build supports SM {gpu_sm}. Available builds: {sets}")


@dataclass
class EngineFile:
    """One engine file to download: remote path (relative to <version>/<platform>/) → local path."""
    remote: str
    local: Path
    sha256: str


def _get_effective_gpu_sm(gpu_sm: Optional[int] = None) -> Optional[int]:
    """Resolve GPU SM from argument, settings.gpu_sm, or hardware auto-detection."""
    if gpu_sm is not None:
        return gpu_sm
    settings = get_settings()
    if settings.gpu_sm:
        return settings.gpu_sm
    try:
        from .gpu import detect_gpu
        info = detect_gpu()
        return info.sm if info else None
    except Exception:
        return None


def _plan_engine_files(
    version: str,
    gpu_sm: Optional[int] = None,
    manifest: Optional[dict] = None,
) -> tuple[list[EngineFile], Optional[str]]:
    """Resolve which files make up engine `version` for this platform/CUDA/GPU.

    Returns (files, kernel_set). Raises RuntimeError if the remote manifests
    can't be fetched or don't list a required file — we never install an
    engine file whose SHA-256 we can't verify.
    """
    settings = get_settings()
    binary_path, cli_path = get_engine_paths()
    effective_sm = _get_effective_gpu_sm(gpu_sm)

    verify_json = _fetch_verify_json(version)
    if not verify_json:
        raise RuntimeError(f"Could not fetch verify.json for engine v{version} (offline?)")
    hashes = _platform_hashes(verify_json)
    if not hashes:
        raise RuntimeError(f"verify.json for v{version} has no '{_manifest_platform_key()}' section")

    sets_json = None
    if "sets.json" in hashes:
        sets_json = _fetch_plugin_json(f"{version}/{settings.engine_platform}/sets.json")
        if sets_json is None:
            raise RuntimeError(f"Could not fetch sets.json for engine v{version}")
    kernel_set = _select_kernel_set(sets_json, effective_sm)

    if manifest is None:
        manifest = fetch_engine_manifest() or {}
    entry = (manifest.get(_manifest_platform_key()) or {}).get(version) or {}
    cuda12 = settings.cuda_version == "12"

    names = [binary_path.name, cli_path.name]
    names += list(entry.get("addons-12" if cuda12 else "addons") or [])
    if entry.get("kernel_so") and settings.engine_platform != "windows":
        # Linux ships GPU kernels as a separate per-arch .so; Windows bakes them into the DLL
        names.append("libquantfunc_kernels-12.so" if cuda12 else "libquantfunc_kernels.so")

    files: list[EngineFile] = []
    for name in names:
        candidates = ([f"{kernel_set}/{name}"] if kernel_set else []) + [name]
        remote = next((c for c in candidates if c in hashes), None)
        if remote is None:
            raise RuntimeError(
                f"verify.json for v{version} does not list {name} "
                f"(build: {kernel_set or 'flat'}) — refusing to install unverified files"
            )
        files.append(EngineFile(remote=remote, local=settings.engine_dir / name, sha256=hashes[remote]))
    return files, kernel_set


# ---------- Public API ----------
def get_engine_paths(version: Optional[str] = None) -> tuple[Path, Path]:
    """Get (binary_path, cli_path) for current platform + CUDA version.

    If `version` is specified and differs from DEFAULT_ENGINE_VERSION:
    looks for `settings.engine_dir / f"v{version}" / ...`.
    If that subfolder exists and contains the binary, returns paths in that subfolder.
    Otherwise returns default paths in `settings.engine_dir`.
    """
    settings = get_settings()
    key = (settings.engine_platform, settings.cuda_version)
    if key not in ENGINE_FILENAMES:
        raise ValueError(f"Unsupported platform/CUDA combo: {key}")

    binary_name, cli_name = ENGINE_FILENAMES[key]

    if version and version != DEFAULT_ENGINE_VERSION:
        for sub in (settings.engine_dir / f"v{version}", settings.engine_dir / version):
            if (sub / cli_name).exists():
                return sub / binary_name, sub / cli_name

    return settings.engine_dir / binary_name, settings.engine_dir / cli_name


def engine_binary_path(version: Optional[str] = None) -> Path:
    return get_engine_paths(version)[0]


def engine_cli_path(version: Optional[str] = None) -> Path:
    return get_engine_paths(version)[1]


def is_engine_installed(version: Optional[str] = None) -> bool:
    """Quick existence check (no SHA verification)."""
    binary, cli = get_engine_paths(version)
    return binary.exists() and cli.exists()


def list_installed_engine_versions() -> list[str]:
    """List all installed engine versions available in bin/<platform>/."""
    versions = []
    settings = get_settings()
    key = (settings.engine_platform, settings.cuda_version)
    if key in ENGINE_FILENAMES:
        _, cli_name = ENGINE_FILENAMES[key]
        if (settings.engine_dir / cli_name).exists():
            versions.append(DEFAULT_ENGINE_VERSION)
        for item in settings.engine_dir.glob("v*"):
            if item.is_dir() and (item / cli_name).exists():
                ver_name = item.name.lstrip("v")
                if ver_name not in versions:
                    versions.append(ver_name)
    return versions


def engine_config_json_path() -> Path:
    """Path to bin/<os>/config.json — engine auth config (server_url + api_key).

    The engine reads this file itself when --apikey is not passed on the CLI.
    The ComfyUI plugin ships it with the shared free-trial test key; our
    downloader previously skipped it, which made key-protected models fail
    with "checkObfuscation failed ... Please provide --apikey".
    """
    settings = get_settings()
    return settings.engine_dir / "config.json"


def read_engine_api_key() -> Optional[str]:
    """Read api_key from bin/<os>/config.json if present and non-empty."""
    path = engine_config_json_path()
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        key = data.get("api_key")
        return key if isinstance(key, str) and key.strip() else None
    except (OSError, json.JSONDecodeError):
        return None


def has_api_key_available() -> bool:
    """True if any API key is available: user settings key OR config.json."""
    settings = get_settings()
    if settings.quantfunc_api_key and settings.quantfunc_api_key.strip():
        return True
    return read_engine_api_key() is not None


def ensure_config_json(force: bool = False, timeout: int = 15) -> tuple[Path, str]:
    """Ensure bin/<os>/config.json exists next to the engine binary.

    Resolution order:
      1. Already present and force=False → keep as-is (never overwrite user edits).
      2. Download from the plugin GitHub repo (bin/<os>/config.json).
      3. Offline fallback — write the embedded BUNDLED_TEST_CONFIG.

    Args:
        force: overwrite an existing file with the bundled test config
               (used by the "Restore test key" action).
        timeout: network timeout in seconds for the GitHub download.

    Returns:
        (path, source) where source is "existing" | "github" | "bundled-fallback".
    """
    settings = get_settings()
    settings.engine_dir.mkdir(parents=True, exist_ok=True)
    config_path = engine_config_json_path()

    if config_path.exists() and not force:
        return config_path, "existing"

    # Try to download the plugin-shipped config (may carry a refreshed test key)
    platform_dir = "windows" if IS_WINDOWS else "linux"
    url = f"{PLUGIN_RAW_BASE}/bin/{platform_dir}/config.json"
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if isinstance(data.get("api_key"), str) and data["api_key"].strip():
            data.setdefault("server_url", BUNDLED_TEST_CONFIG["server_url"])
            config_path.write_text(
                json.dumps(data, indent=4, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"[engine] config.json installed from plugin repo: {config_path}")
            return config_path, "github"
    except (requests.RequestException, json.JSONDecodeError, ValueError) as e:
        print(f"[engine] config.json download failed ({e}) — writing bundled fallback")

    # Offline fallback — bundled test key (same as the plugin ships)
    config_path.write_text(
        json.dumps(BUNDLED_TEST_CONFIG, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[engine] config.json written with bundled test key: {config_path}")
    return config_path, "bundled-fallback"


def verify_engine(gpu_sm: Optional[int] = None) -> VerifyResult:
    """Verify local engine binary and addons SHA-256 against remote verify.json.

    Network access required. If offline, returns status="offline".
    Checks all planned files for current platform, CUDA, and GPU SM.
    """
    settings = get_settings()
    binary_path, _ = get_engine_paths()
    effective_sm = _get_effective_gpu_sm(gpu_sm)

    # Fast existence check first
    if not binary_path.exists():
        return VerifyResult(
            status="missing",
            local_path=binary_path,
            message=f"Engine binary not found at {binary_path}",
        )

    try:
        files, kernel_set = _plan_engine_files(settings.engine_version, effective_sm)
    except Exception as e:
        local_sha = _compute_sha256(binary_path) if binary_path.exists() else None
        return VerifyResult(
            status="offline",
            local_sha256=local_sha,
            local_path=binary_path,
            message=f"Could not fetch remote manifests for engine v{settings.engine_version} ({e})",
        )

    # Check that all planned files exist locally
    for f in files:
        if not f.local.exists():
            return VerifyResult(
                status="missing",
                local_path=f.local,
                remote_sha256=f.sha256,
                remote_version=settings.engine_version,
                message=f"Engine file missing: {f.local.name}",
            )

    # Check SHA-256 for all files
    binary_file = next((f for f in files if f.local.name == binary_path.name), files[0])
    for f in files:
        local_sha = _compute_sha256(f.local)
        if local_sha != f.sha256:
            return VerifyResult(
                status="outdated",
                local_sha256=local_sha,
                remote_sha256=f.sha256,
                remote_version=settings.engine_version,
                local_path=f.local,
                message=f"SHA-256 mismatch for {f.local.name}: local {local_sha[:12]}... != remote {f.sha256[:12]}...",
            )

    return VerifyResult(
        status="ok",
        local_sha256=binary_file.sha256,
        remote_sha256=binary_file.sha256,
        remote_version=settings.engine_version,
        local_path=binary_path,
        message=f"Engine v{settings.engine_version} ({kernel_set or 'flat'}) is up to date",
    )


def download_engine(
    progress_callback=None,
    replace_existing: bool = True,
    gpu_sm: Optional[int] = None,
) -> Path:
    """Download engine binary and addons from ModelScope with verify-before-replace.

    Downloads all files planned for the target engine version and GPU architecture
    (main library, CLI runner, and architecture/CUDA addons like quantfunc_attention.dll).
    Each file is downloaded to a sibling `.tmp` file, SHA-256 verified, and
    atomically renamed into place.

    Args:
        progress_callback: Optional callable(downloaded_bytes, total_bytes, speed_bps)
        replace_existing: If True, replaces existing binary after successful verify.
                         If False, raises if binary already exists.
        gpu_sm: Optional GPU compute capability (e.g. 120 for RTX 5060 Ti).

    Returns:
        Path to the installed engine binary.
    """
    binary_path, cli_path = get_engine_paths()
    settings = get_settings()
    effective_sm = _get_effective_gpu_sm(gpu_sm)

    # Check existing
    if binary_path.exists() and not replace_existing:
        raise FileExistsError(f"Engine binary already exists: {binary_path}")

    version = settings.engine_version
    files, kernel_set = _plan_engine_files(version, effective_sm)

    # Ensure engine_dir exists
    settings.engine_dir.mkdir(parents=True, exist_ok=True)

    temp_files: list[tuple[EngineFile, Path]] = []
    try:
        for f in files:
            temp_file = f.local.with_suffix(f.local.suffix + ".tmp")
            temp_files.append((f, temp_file))
            url = (
                f"{MODELSCOPE_PLUGIN_FILE}?Revision=master"
                f"&FilePath={version}/{settings.engine_platform}/{f.remote}"
            )
            print(f"[engine] Downloading {f.local.name} (from {f.remote}) from ModelScope...")
            _download_file_with_progress(url, temp_file, progress_callback)

            actual_sha = _compute_sha256(temp_file)
            if actual_sha != f.sha256:
                raise SecurityError(
                    f"Engine file {f.local.name} SHA-256 mismatch: expected {f.sha256}, got {actual_sha}"
                )
            print(f"[engine] {f.local.name} SHA-256 verified [OK]")

        # Atomic replace (rename)
        for f, temp_file in temp_files:
            if f.local.exists():
                f.local.unlink()
            temp_file.rename(f.local)

            # On Linux: make CLI executable
            if not IS_WINDOWS and f.local == cli_path:
                os.chmod(f.local, 0o755)

            print(f"[engine] Installed: {f.local}")

        # Install bin/<os>/config.json (engine auth: server_url + api_key).
        try:
            ensure_config_json()
        except Exception as cfg_err:
            print(f"[engine] WARNING: could not install config.json: {cfg_err}")

        return binary_path

    except Exception:
        # Cleanup temp files on failure
        for _, temp_file in temp_files:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass
        raise


def _download_file_with_progress(
    url: str,
    dest_path: Path,
    progress_callback=None,
    chunk_size: int = 1024 * 1024,  # 1 MB
    timeout: int = 600,
) -> None:
    """Download a file with progress reporting and resume support."""
    import time

    # Check for partial download (resume)
    existing_size = dest_path.stat().st_size if dest_path.exists() else 0

    headers = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        mode = "ab"  # append
    else:
        mode = "wb"

    response = requests.get(url, stream=True, headers=headers, timeout=timeout, allow_redirects=True)

    # If server doesn't support range, restart from scratch
    if existing_size > 0 and response.status_code == 200:
        existing_size = 0
        mode = "wb"

    if response.status_code not in (200, 206):
        raise RuntimeError(f"HTTP {response.status_code} downloading {url}")

    total_size = int(response.headers.get("content-length", 0))
    if existing_size > 0 and response.status_code == 206:
        total_size += existing_size  # Total = existing + remaining

    downloaded = existing_size
    start_time = time.time()

    with open(dest_path, mode) as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)

                if progress_callback and total_size > 0:
                    elapsed = time.time() - start_time
                    speed_bps = (downloaded - existing_size) / elapsed if elapsed > 0 else 0
                    progress_callback(downloaded, total_size, speed_bps)

    print(f"[engine] Download complete: {dest_path} ({downloaded / 1024 / 1024:.1f} MB)")


class SecurityError(Exception):
    """Raised when SHA-256 verification fails."""


# ---------- Quick check on import ----------
def check_engine_status() -> VerifyResult:
    """Quick check used by app.py at startup."""
    if not is_engine_installed():
        return VerifyResult(
            status="missing",
            message="Engine binary not installed. Use Settings → Download engine.",
        )

    # Skip network verification on every startup (offline-first)
    # Only verify when user explicitly clicks "Check for updates"
    return VerifyResult(
        status="ok",
        local_path=engine_binary_path(),
        message="Engine binary present (skipped network verify — use Settings → Check for updates)",
    )


# ============================================================
# CUDA dependency management
# ============================================================

def get_cuda_dep_filename() -> Optional[str]:
    """Get the CUDA dep bundle filename for current platform + CUDA version."""
    settings = get_settings()
    key = (settings.engine_platform, settings.cuda_version)
    return CUDA_DEP_FILENAMES.get(key)


def _detect_cuda_bin_paths() -> list[Path]:
    """Auto-detect standard CUDA Toolkit install locations on Windows + Linux.

    Checks (in order):
      1. settings.cuda_bin_path (user override)
      2. CUDA_PATH / CUDA_HOME env vars
      3. Standard Windows install paths (drive letters C..G, CUDA v12.x / v13.x)
      4. Standard Linux install paths (/usr/local/cuda-13, /usr/local/cuda-12)

    Returns:
        List of directories that exist on disk. May include both 64-bit bin
        and bin/x64 subdirectories.
    """
    import os
    settings = get_settings()
    cuda_version = settings.cuda_version
    candidates: list[Path] = []

    # 1. User-specified explicit path
    if settings.cuda_bin_path:
        candidates.append(Path(settings.cuda_bin_path))

    # 2. Environment variables
    for env_var in ("CUDA_PATH", "CUDA_HOME"):
        env_val = os.environ.get(env_var)
        if env_val:
            p = Path(env_val)
            candidates.append(p / "bin")
            candidates.append(p / "bin" / "x64")  # Some installs put DLLs in bin/x64
            candidates.append(p / "lib" / "x64")  # Linux: lib/x64 sometimes
            candidates.append(p / "lib64")       # Linux: lib64

    # 3. Standard Windows install paths
    # Check all common drive letters C, D, E, F, G, H
    if IS_WINDOWS:
        program_files_env = os.environ.get("ProgramFiles", "C:\\Program Files")
        for drive in ("C", "D", "E", "F", "G", "H"):
            for base in (
                f"{drive}:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
                f"{drive}:\\Program Files (x86)\\NVIDIA GPU Computing Toolkit\\CUDA",
            ):
                # Check both v12.x and v13.x (and any other CUDA v*)
                if base.startswith(drive + ":") or drive + ":" in base:
                    try:
                        if Path(base).exists():
                            for sub in Path(base).iterdir():
                                if sub.is_dir() and sub.name.startswith("v"):
                                    candidates.append(sub / "bin")
                                    candidates.append(sub / "bin" / "x64")
                    except (OSError, PermissionError):
                        pass

    # 4. Standard Linux install paths
    if not IS_WINDOWS:
        for v in ("13", "13.0", "13.1", "12", "12.0"):
            candidates.append(Path(f"/usr/local/cuda-{v}"))
            candidates.append(Path(f"/usr/local/cuda-{v}/lib64"))
        candidates.append(Path("/usr/local/cuda"))
        candidates.append(Path("/usr/lib/x86_64-linux-gnu"))

    # Filter to existing dirs only + deduplicate
    seen = set()
    result = []
    for c in candidates:
        try:
            resolved = c.resolve()
            if resolved not in seen and resolved.is_dir():
                seen.add(resolved)
                result.append(resolved)
        except (OSError, RuntimeError):
            continue
    return result


def get_cuda_search_paths() -> list[Path]:
    """All directories where CUDA DLLs might be found.

    Order (most preferred first):
      1. settings.cuda_bin_path (user override)
      2. CUDA_PATH/CUDA_HOME env vars (bin, bin/x64, lib/x64, lib64)
      3. Standard install paths (auto-detected)
      4. engine_dir (dep bundle extracted here)
    """
    settings = get_settings()
    paths = _detect_cuda_bin_paths()
    # Always include engine_dir last (dep bundle)
    engine_dir_resolved = settings.engine_dir.resolve()
    if engine_dir_resolved not in paths and engine_dir_resolved.is_dir():
        paths.append(engine_dir_resolved)
    return paths


def _find_dll_by_pattern(pattern: str, search_paths: list) -> tuple[Optional[str], Optional[str]]:
    """Find a DLL matching a glob pattern across all search paths.

    Args:
        pattern: glob like "cublas64_*.dll"
        search_paths: list of Path objects to search

    Returns:
        (found_filename, found_location) where found_location is one of:
        "engine_dir", "cuda_bin_path", "auto", "system", or None if not found.
    """
    import shutil

    settings = get_settings()
    engine_dir_resolved = settings.engine_dir.resolve()
    cuda_bin_path_resolved = Path(settings.cuda_bin_path).resolve() if settings.cuda_bin_path else None

    for sp in search_paths:
        try:
            for match in sp.glob(pattern):
                # Determine location label
                if sp == engine_dir_resolved:
                    return match.name, "engine_dir"
                elif cuda_bin_path_resolved and sp == cuda_bin_path_resolved:
                    return match.name, "cuda_bin_path"
                else:
                    return match.name, "auto"
        except (OSError, PermissionError):
            continue

    # Fallback: try system PATH via shutil.which
    # For glob patterns, we can't directly use shutil.which — try common variants
    # Strip the wildcard suffix to get the base name
    if pattern.endswith("_*.dll"):
        base = pattern[:-6]  # "cublas64_" from "cublas64_*.dll"
    elif pattern.endswith("*.dll"):
        base = pattern[:-5]
    elif pattern.endswith("_*.so.*"):
        base = pattern[:-6]
    elif pattern.endswith("*.so.*"):
        base = pattern[:-5]
    else:
        base = pattern

    # Try a few version suffixes on system PATH
    for suffix in ("13", "12", "10", "9", "130_0", "120_0", "100_0"):
        candidate = f"{base}{suffix}.dll" if not base.endswith(".dll") else base
        try:
            system_path = shutil.which(candidate)
            if system_path:
                return candidate, "system"
        except Exception:
            pass

    return None, None


def check_cuda_dependencies() -> dict:
    """Check if critical CUDA DLLs are available.

    Uses GLOB PATTERNS (not exact filenames) because NVIDIA uses different
    version suffixes within the same CUDA Toolkit install:
        - cublas64_13.dll (v13)
        - curand64_10.dll (v10 — older lib kept for compat)
        - cusolver64_12.dll (v12)
        - nvJitLink_130_0.dll (different scheme)

    cuDNN is OPTIONAL — separate download from NVIDIA, not in CUDA Toolkit.
    We warn but don't block if missing; engine will tell us if it really needs it.

    Looks in:
      1. settings.cuda_bin_path (user override — highest priority)
      2. CUDA_PATH / CUDA_HOME env vars (bin, bin/x64)
      3. Standard Windows/Linux install paths (auto-detected)
      4. engine_dir (dep bundle extracted here)
      5. System PATH (last resort — shutil.which)
    """
    settings = get_settings()
    key = (settings.engine_platform, settings.cuda_version)
    critical_patterns = CUDA_CRITICAL_DLL_PATTERNS.get(key, [])
    optional_patterns = CUDA_OPTIONAL_DLL_PATTERNS.get(key, [])

    if not critical_patterns:
        return {
            "installed": True,
            "missing": [],
            "missing_optional": [],
            "found_locations": {},
            "search_paths": [],
            "dep_bundle_downloaded": False,
            "message": f"No critical DLL patterns defined for {key}",
        }

    # Build search paths (most preferred first)
    search_paths = get_cuda_search_paths()

    missing = []          # Critical missing
    missing_optional = [] # Optional missing (cuDNN)
    found_locations = {}  # pattern → (found_filename, location)

    # Check critical DLLs
    for pattern in critical_patterns:
        found_name, location = _find_dll_by_pattern(pattern, search_paths)
        if location:
            found_locations[pattern] = {"found": found_name, "location": location}
        else:
            found_locations[pattern] = {"found": None, "location": None}
            missing.append(pattern)

    # Check optional DLLs (cuDNN)
    for pattern in optional_patterns:
        found_name, location = _find_dll_by_pattern(pattern, search_paths)
        if location:
            found_locations[pattern] = {"found": found_name, "location": location}
        else:
            found_locations[pattern] = {"found": None, "location": None}
            missing_optional.append(pattern)

    # Check if dep bundle was downloaded (marker file)
    dep_marker = settings.engine_dir / ".cuda_deps_extracted"
    dep_bundle_downloaded = dep_marker.exists()

    return {
        "installed": len(missing) == 0,
        "missing": missing,
        "missing_optional": missing_optional,
        "found_locations": found_locations,
        "search_paths": [str(p) for p in search_paths],
        "dep_bundle_downloaded": dep_bundle_downloaded,
        "dep_bundle_filename": get_cuda_dep_filename(),
        "cuda_bin_path": settings.cuda_bin_path,
        "engine_dir": str(settings.engine_dir),
    }


def download_cuda_deps(progress_callback=None) -> dict:
    """Download and extract the CUDA dependency bundle from ModelScope.

    The bundle is a ~1-1.6 GB zip file containing cublas64_XX.dll, cudnn64_9.dll, etc.
    Extracted into engine_dir so the engine binary can find them.

    Returns:
        {"status": "ok" | "error", "extracted_dir": str, "files_count": int}
    """
    import zipfile
    import tempfile

    settings = get_settings()
    dep_filename = get_cuda_dep_filename()
    if not dep_filename:
        raise RuntimeError(
            f"No CUDA dep bundle for platform={settings.engine_platform}, "
            f"cuda_version={settings.cuda_version}"
        )

    # Build download URL (dep bundles are at root level, not inside version folder)
    dep_url = f"{MODELSCOPE_PLUGIN_FILE}?Revision=master&FilePath={dep_filename}"

    # Download to a temp file, then extract
    settings.engine_dir.mkdir(parents=True, exist_ok=True)
    temp_zip = settings.engine_dir / dep_filename

    try:
        print(f"[engine] Downloading CUDA deps: {dep_filename} (~1 GB)...")
        _download_file_with_progress(dep_url, temp_zip, progress_callback)

        print(f"[engine] Extracting CUDA deps to {settings.engine_dir}...")
        extracted_count = 0
        with zipfile.ZipFile(temp_zip, 'r') as zf:
            for member in zf.namelist():
                # Skip directories
                if member.endswith('/'):
                    continue
                # Extract, flattening any subdirectory structure
                # (zip might contain files at root or in a subfolder)
                filename = Path(member).name
                if not filename:
                    continue
                target = settings.engine_dir / filename

                # Security: don't extract paths with .. or absolute paths
                try:
                    target.resolve().relative_to(settings.engine_dir.resolve())
                except ValueError:
                    print(f"[engine] Skipping unsafe path in zip: {member}")
                    continue

                # Extract
                with zf.open(member) as src, open(target, 'wb') as dst:
                    dst.write(src.read())
                extracted_count += 1

        # Write marker file
        marker = settings.engine_dir / ".cuda_deps_extracted"
        marker.write_text(f"dep_bundle={dep_filename}\nextracted_count={extracted_count}\n")

        print(f"[engine] CUDA deps extracted: {extracted_count} files → {settings.engine_dir}")

        # Clean up zip file (save disk space)
        try:
            temp_zip.unlink()
        except OSError:
            pass

        return {
            "status": "ok",
            "extracted_dir": str(settings.engine_dir),
            "files_count": extracted_count,
        }

    except Exception as e:
        # Cleanup on failure
        if temp_zip.exists():
            try:
                temp_zip.unlink()
            except OSError:
                pass
        raise RuntimeError(f"Failed to download/extract CUDA deps: {e}") from e
