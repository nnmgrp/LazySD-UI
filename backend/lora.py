"""
LoRA library for LazySD UI.

LoRA files live in ``<models_dir>/loras`` (flat folder). Keeping them inside
``models_dir`` matters: the QuantFunc worker only accepts LoRA paths under
``settings.models_dir`` (see ``validate_path`` in worker.py).

Everything user-facing goes through *names* (file basenames), never raw paths,
so a crafted request can't make the backend read arbitrary files.
"""

from __future__ import annotations

import json
import re
import shutil
import struct
from pathlib import Path
from typing import Optional

from .settings import get_settings

LORA_EXTS = (".safetensors", ".gguf", ".pt", ".ckpt")
MAX_LORAS_PER_JOB = 8
MIN_STRENGTH = -2.0
MAX_STRENGTH = 2.0
_MAX_HEADER_BYTES = 64 * 1024 * 1024  # sanity cap when sniffing safetensors headers

# sd.cpp parses <lora:NAME:WEIGHT>, so ':' / '<' / '>' / whitespace are poison in names.
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._\-\[\]()+]+")


def get_loras_dir() -> Path:
    d = get_settings().models_dir / "loras"
    d.mkdir(parents=True, exist_ok=True)
    return d


def sanitize_lora_filename(name: str) -> str:
    """Return a filesystem- and prompt-tag-safe filename (keeps the extension)."""
    base = Path(str(name)).name
    stem, ext = Path(base).stem, Path(base).suffix.lower()
    stem = _SAFE_NAME_RE.sub("_", stem).strip("._") or "lora"
    return f"{stem}{ext}"


def _read_safetensors_header(path: Path) -> Optional[dict]:
    """Read only the JSON header of a .safetensors file (no tensor data)."""
    try:
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                return None
            (n,) = struct.unpack("<Q", raw)
            if n <= 0 or n > _MAX_HEADER_BYTES:
                return None
            return json.loads(f.read(n).decode("utf-8", errors="replace"))
    except Exception:
        return None


def _family_from_keys(keys: list[str]) -> Optional[str]:
    blob = " ".join(keys[:400]).lower()
    if "noise_refiner" in blob or "context_refiner" in blob:
        return "z_image"
    if "double_blocks" in blob or "single_blocks" in blob or "single_transformer_blocks" in blob:
        return "flux"
    if "transformer_blocks" in blob and ("add_q_proj" in blob or "img_mlp" in blob or "txt_mlp" in blob):
        return "qwen"
    if "lora_te1" in blob or "input_blocks" in blob or "down_blocks" in blob:
        return "sdxl"
    return None


_META_FAMILY_HINTS = (
    ("sdxl", "sdxl"), ("flux", "flux"), ("klein", "flux"),
    ("qwen", "qwen"), ("z-image", "z_image"), ("z_image", "z_image"),
)


def detect_lora_info(path: Path) -> dict:
    """Best-effort metadata (family / rank / trigger words). Never raises."""
    info: dict = {"family": None, "rank": None, "trigger": None, "base": None}
    if path.suffix.lower() != ".safetensors":
        return info
    header = _read_safetensors_header(path)
    if not header:
        return info
    meta = header.get("__metadata__") or {}
    keys = [k for k in header.keys() if k != "__metadata__"]

    base = meta.get("ss_base_model_version") or meta.get("modelspec.architecture") or meta.get("base_model")
    if base:
        info["base"] = str(base)[:80]
        low = str(base).lower()
        for needle, fam in _META_FAMILY_HINTS:
            if needle in low:
                info["family"] = fam
                break
    if not info["family"]:
        info["family"] = _family_from_keys(keys)

    # rank: first lora_down / lora_A weight -> shape[0]
    for k in keys:
        kl = k.lower()
        if "lora_down" in kl or "lora_a" in kl:
            shape = (header[k] or {}).get("shape") or []
            if shape:
                info["rank"] = int(min(shape))
            break

    trig = meta.get("modelspec.trigger_phrase") or meta.get("trigger_word") or meta.get("trigger_words")
    if not trig and meta.get("ss_tag_strings"):
        try:
            tags = json.loads(meta["ss_tag_strings"])
            if isinstance(tags, dict) and tags:
                trig = ", ".join(list(tags.keys())[:3])
        except Exception:
            pass
    if trig:
        info["trigger"] = str(trig)[:160]
    return info


def list_loras() -> list[dict]:
    d = get_loras_dir()
    out = []
    for p in sorted(d.iterdir(), key=lambda x: x.name.lower()):
        if not p.is_file() or p.suffix.lower() not in LORA_EXTS:
            continue
        info = detect_lora_info(p)
        out.append({
            "name": p.name,
            "label": p.stem,
            "size_mb": round(p.stat().st_size / (1024 * 1024), 1),
            "ext": p.suffix.lower().lstrip("."),
            **info,
        })
    return out


def resolve_lora_selection(selection) -> tuple[list[Path], list[float], list[dict]]:
    """Validate a UI selection ``[{name, strength}]`` -> (paths, strengths, meta).

    Raises ValueError on bad input. Names must be plain basenames that exist in
    the LoRA folder.
    """
    if not selection:
        return [], [], []
    if not isinstance(selection, list):
        raise ValueError("loras must be a list")
    if len(selection) > MAX_LORAS_PER_JOB:
        raise ValueError(f"Too many LoRAs (max {MAX_LORAS_PER_JOB})")

    root = get_loras_dir().resolve()
    paths: list[Path] = []
    strengths: list[float] = []
    meta: list[dict] = []
    seen: set[str] = set()
    for item in selection:
        if not isinstance(item, dict):
            raise ValueError("Invalid LoRA entry")
        name = str(item.get("name") or "").strip()
        if not name or name != Path(name).name or name in (".", ".."):
            raise ValueError(f"Invalid LoRA name: {name!r}")
        if name in seen:
            continue
        seen.add(name)
        p = (root / name).resolve()
        if p.parent != root or not p.is_file():
            raise ValueError(f"LoRA not found: {name}")
        try:
            s = float(item.get("strength", 1.0))
        except (TypeError, ValueError):
            raise ValueError(f"Invalid strength for {name}")
        if s != s:  # NaN
            raise ValueError(f"Invalid strength for {name}")
        s = max(MIN_STRENGTH, min(MAX_STRENGTH, s))
        paths.append(p)
        strengths.append(round(s, 3))
        meta.append({"name": name, "strength": round(s, 3)})
    return paths, strengths, meta


def import_local_lora(src: str, rename: Optional[str] = None) -> dict:
    """Copy a LoRA from an arbitrary path on the user's PC into the library."""
    p = Path(str(src).strip().strip('"')).expanduser()
    if not p.is_file():
        raise ValueError(f"File not found: {p}")
    if p.suffix.lower() not in LORA_EXTS:
        raise ValueError(f"Unsupported LoRA format: {p.suffix} (allowed: {', '.join(LORA_EXTS)})")
    dest_name = sanitize_lora_filename(rename or p.name)
    if Path(dest_name).suffix.lower() not in LORA_EXTS:
        dest_name += p.suffix.lower()
    dest = get_loras_dir() / dest_name
    if dest.exists():
        raise ValueError(f"LoRA already exists: {dest_name}")
    shutil.copy2(p, dest)
    return {"name": dest.name, **detect_lora_info(dest)}


def save_uploaded_lora(filename: str, data_stream) -> dict:
    """Persist an uploaded LoRA (file-like stream) into the library."""
    safe = sanitize_lora_filename(filename)
    if Path(safe).suffix.lower() not in LORA_EXTS:
        raise ValueError(f"Unsupported LoRA format: {Path(safe).suffix}")
    dest = get_loras_dir() / safe
    if dest.exists():
        raise ValueError(f"LoRA already exists: {safe}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with open(tmp, "wb") as out:
            shutil.copyfileobj(data_stream, out, length=1024 * 1024)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return {"name": dest.name, **detect_lora_info(dest)}


def download_hf_lora(repo_id: str, filename: Optional[str] = None, token: Optional[str] = None) -> dict:
    """Download a LoRA file from a Hugging Face repo into the library.

    If ``filename`` is omitted, the repo must contain exactly one LoRA-looking
    weight file, otherwise the list of candidates is returned in the error.
    """
    from huggingface_hub import HfApi, hf_hub_download

    repo_id = str(repo_id).strip()
    if not re.fullmatch(r"[\w.\-]+/[\w.\-]+", repo_id):
        raise ValueError("repo_id must look like 'author/repo'")

    api = HfApi(token=token or None)
    files = [
        s.rfilename for s in api.model_info(repo_id, files_metadata=False).siblings
        if Path(s.rfilename).suffix.lower() in LORA_EXTS
    ]
    if not files:
        raise ValueError("No LoRA weight files (.safetensors/.gguf/.pt/.ckpt) found in this repo")
    if filename:
        if filename not in files:
            raise ValueError(f"File not in repo: {filename}")
        chosen = filename
    elif len(files) == 1:
        chosen = files[0]
    else:
        raise ValueError("Multiple files found, pick one: " + ", ".join(files[:12]))

    loras_dir = get_loras_dir()
    dest_name = sanitize_lora_filename(Path(chosen).name)
    dest = loras_dir / dest_name
    if dest.exists():
        raise ValueError(f"LoRA already exists: {dest_name}")
    cached = hf_hub_download(repo_id=repo_id, filename=chosen, token=token or None,
                             local_dir=str(loras_dir / ".hf_tmp"))
    shutil.move(cached, dest)
    shutil.rmtree(loras_dir / ".hf_tmp", ignore_errors=True)
    return {"name": dest.name, **detect_lora_info(dest)}


def delete_lora(name: str) -> None:
    if not name or name != Path(name).name:
        raise ValueError("Invalid LoRA name")
    p = get_loras_dir() / name
    if not p.is_file():
        raise ValueError(f"LoRA not found: {name}")
    p.unlink()


def lora_prompt_tags(paths: list[Path], strengths: list[float]) -> str:
    """sd.cpp-style `<lora:name:weight>` tags (name = file stem)."""
    return " ".join(f"<lora:{p.stem}:{s:g}>" for p, s in zip(paths, strengths))
