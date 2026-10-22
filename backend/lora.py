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


def _get_lora_meta_file() -> Path:
    return get_loras_dir() / ".loras_meta.json"


def load_all_lora_meta() -> dict[str, dict]:
    f = _get_lora_meta_file()
    if not f.exists():
        return {}
    try:
        with open(f, "r", encoding="utf-8") as jf:
            return json.load(jf)
    except Exception:
        return {}


def save_all_lora_meta(meta_map: dict[str, dict]) -> None:
    f = _get_lora_meta_file()
    try:
        with open(f, "w", encoding="utf-8") as jf:
            json.dump(meta_map, jf, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[lora] Failed to save metadata: {e}")


def set_lora_metadata(name: str, meta: dict) -> None:
    if not name:
        return
    data = load_all_lora_meta()
    entry = data.setdefault(name, {})
    for k, v in meta.items():
        if v is not None:
            entry[k] = v
    save_all_lora_meta(data)


def detect_lora_info(path: Path) -> dict:
    """Best-effort metadata (family / rank / trigger words). Never raises."""
    info: dict = {"family": None, "rank": None, "trigger": None, "base": None}
    meta_overrides = load_all_lora_meta().get(path.name, {})

    if path.suffix.lower() == ".safetensors":
        header = _read_safetensors_header(path)
        if header:
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

    # Host-configured metadata overrides take precedence
    if meta_overrides.get("family"):
        info["family"] = meta_overrides["family"]
    if meta_overrides.get("trigger"):
        info["trigger"] = meta_overrides["trigger"]
    if meta_overrides.get("base"):
        info["base"] = meta_overrides["base"]

    return info


def list_loras() -> list[dict]:
    d = get_loras_dir()
    out = []
    meta_map = load_all_lora_meta()
    for p in sorted(d.iterdir(), key=lambda x: x.name.lower()):
        if not p.is_file() or p.suffix.lower() not in LORA_EXTS:
            continue
        info = detect_lora_info(p)
        m_entry = meta_map.get(p.name, {})
        custom_label = m_entry.get("label") or p.stem
        out.append({
            "name": p.name,
            "label": custom_label,
            "size_mb": round(p.stat().st_size / (1024 * 1024), 1),
            "ext": p.suffix.lower().lstrip("."),
            "repo_id": m_entry.get("repo_id") or "",
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


def normalize_hf_repo_lora(text: str) -> str:
    """Normalize a Hugging Face URL or repo string to 'author/repo'."""
    clean = str(text or "").strip().strip('"').strip("'")
    clean = clean.replace("https://huggingface.co/", "").replace("http://huggingface.co/", "").strip("/")
    if "/tree/" in clean:
        clean = clean.split("/tree/")[0]
    if "/blob/" in clean:
        clean = clean.split("/blob/")[0]
    return clean


def inspect_hf_lora(url_or_repo: str, token: Optional[str] = None) -> dict:
    """Inspect a Hugging Face LoRA repo: list weight files, detect family, trigger words, base model."""
    import requests
    clean_repo = normalize_hf_repo_lora(url_or_repo)
    if not clean_repo or "/" not in clean_repo:
        raise ValueError("Укажите корректный Hugging Face репозиторий в формате author/repo")

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    else:
        from .custom_models import _get_hf_token
        t = _get_hf_token()
        if t:
            headers["Authorization"] = f"Bearer {t}"

    url = f"https://huggingface.co/api/models/{clean_repo}?blobs=true"
    resp = requests.get(url, headers=headers, timeout=15)
    if resp.status_code == 404:
        raise ValueError(f"Репозиторий '{clean_repo}' не найден на Hugging Face")
    if resp.status_code == 401:
        raise ValueError(f"Доступ к '{clean_repo}' требует Hugging Face токена авторизации")
    if not resp.ok:
        raise ValueError(f"Ошибка Hugging Face API ({resp.status_code}): {resp.text[:200]}")

    info = resp.json()
    siblings = info.get("siblings") or []
    card_data = info.get("cardData") or {}
    tags = [str(t) for t in (info.get("tags") or [])]

    # Find candidate LoRA files
    files = []
    for s in siblings:
        rfilename = s.get("rfilename") or ""
        lower = rfilename.lower()
        if not lower.endswith(LORA_EXTS):
            continue
        if "mmproj" in lower or lower.startswith("."):
            continue
        size_bytes = s.get("size") or 0
        size_mb = round(size_bytes / (1024 * 1024), 1) if size_bytes else 0.0
        files.append({
            "path": rfilename,
            "name": Path(rfilename).name,
            "size_mb": size_mb,
            "ext": Path(rfilename).suffix.lower().lstrip("."),
        })

    # Family detection
    blob = " ".join([clean_repo, *tags, str(card_data)]).lower()
    suggested_family = "flux"
    if "qwen" in blob:
        suggested_family = "qwen"
    elif "sdxl" in blob:
        suggested_family = "sdxl"
    elif "z_image" in blob or "z-image" in blob:
        suggested_family = "z_image"
    elif "flux" in blob or "klein" in blob:
        suggested_family = "flux"

    # Trigger words detection
    trigger = card_data.get("instance_prompt")
    if not trigger:
        for t in tags:
            if any(k in t.lower() for k in ("trigger", "instance_prompt", "lineart")):
                trigger = t
                break
    if not trigger:
        try:
            raw_url = f"https://huggingface.co/{clean_repo}/raw/main/README.md"
            r_resp = requests.get(raw_url, headers=headers, timeout=5)
            if r_resp.ok:
                m_inst = re.search(r"instance_prompt:\s*([^\r\n]+)", r_resp.text)
                if m_inst:
                    trigger = m_inst.group(1).strip()
        except Exception:
            pass

    base_model = card_data.get("base_model") or ""
    suggested_name = clean_repo.split("/")[-1].replace("-", " ").replace("_", " ")

    return {
        "repo_id": clean_repo,
        "name": suggested_name,
        "files": files,
        "family": suggested_family,
        "suggested_family": suggested_family,
        "trigger": trigger or "",
        "suggested_trigger": trigger or "",
        "base_model": str(base_model) if base_model else "",
        "tags": tags[:15],
    }


def search_hf_loras(query: str, limit: int = 15) -> list[dict]:
    """Search Hugging Face Hub for LoRAs."""
    import requests
    q = (query or "").strip()
    if not q:
        q = "flux lora"
    from .custom_models import _get_hf_token
    token = _get_hf_token()
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        url = f"https://huggingface.co/api/models?search={requests.utils.quote(q)}&filter=lora&limit={limit}&full=true"
        resp = requests.get(url, headers=headers, timeout=12)
        if not resp.ok:
            return []
        items = resp.json()
        out = []
        for item in items:
            repo_id = item.get("id") or item.get("modelId")
            if not repo_id or "/" not in repo_id:
                continue
            card_data = item.get("cardData") or {}
            out.append({
                "repo_id": repo_id,
                "likes": item.get("likes", 0),
                "downloads": item.get("downloads", 0),
                "description": card_data.get("description") or item.get("description") or "",
                "base_model": card_data.get("base_model") or "",
                "tags": (item.get("tags") or [])[:6],
            })
        return out
    except Exception as e:
        print(f"[lora] search_hf_loras failed: {e}")
        return []


def download_hf_lora(
    repo_id: str,
    filename: Optional[str] = None,
    rename: Optional[str] = None,
    family: Optional[str] = None,
    trigger: Optional[str] = None,
    token: Optional[str] = None,
) -> dict:
    """Download a LoRA file from a Hugging Face repo into the library."""
    from huggingface_hub import HfApi, hf_hub_download

    clean_repo = normalize_hf_repo_lora(repo_id)
    if not re.fullmatch(r"[\w.\-]+/[\w.\-]+", clean_repo):
        raise ValueError("repo_id must look like 'author/repo'")

    if not token:
        from .custom_models import _get_hf_token
        token = _get_hf_token()

    api = HfApi(token=token or None)
    files = [
        s.rfilename for s in api.model_info(clean_repo, files_metadata=False).siblings
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
    target_name = rename or Path(chosen).name
    dest_name = sanitize_lora_filename(target_name)
    if Path(dest_name).suffix.lower() not in LORA_EXTS:
        dest_name += Path(chosen).suffix.lower()

    dest = loras_dir / dest_name
    if not dest.exists():
        cached = hf_hub_download(repo_id=clean_repo, filename=chosen, token=token or None,
                                 local_dir=str(loras_dir / ".hf_tmp"))
        shutil.move(cached, dest)
        shutil.rmtree(loras_dir / ".hf_tmp", ignore_errors=True)

    # Auto-detect trigger/family from repo card if not explicitly passed
    if not family or not trigger:
        try:
            insp = inspect_hf_lora(clean_repo, token=token)
            if not family:
                family = insp.get("suggested_family")
            if not trigger:
                trigger = insp.get("suggested_trigger")
        except Exception:
            pass

    meta_payload: dict = {"repo_id": clean_repo}
    if family:
        meta_payload["family"] = family
    if trigger:
        meta_payload["trigger"] = trigger
    set_lora_metadata(dest.name, meta_payload)

    info_out = detect_lora_info(dest)
    meta_map = load_all_lora_meta().get(dest.name, {})
    if meta_map.get("family"):
        info_out["family"] = meta_map["family"]
    if meta_map.get("trigger"):
        info_out["trigger"] = meta_map["trigger"]
    if meta_map.get("repo_id"):
        info_out["repo_id"] = meta_map["repo_id"]
    return {"name": dest.name, **info_out}


def delete_lora(name: str) -> None:
    if not name or name != Path(name).name:
        raise ValueError("Invalid LoRA name")
    p = get_loras_dir() / name
    if not p.is_file():
        raise ValueError(f"LoRA not found: {name}")
    p.unlink()

    meta = load_all_lora_meta()
    if name in meta:
        meta.pop(name, None)
        save_all_lora_meta(meta)


def lora_prompt_tags(paths: list[Path], strengths: list[float]) -> str:
    """sd.cpp-style `<lora:name:weight>` tags (name = file stem)."""
    return " ".join(f"<lora:{p.stem}:{s:g}>" for p, s in zip(paths, strengths))


LORA_SUPPORTED_ENGINES = ("sd_cpp", "quantfunc", "nunchaku", "sdnq")


def normalize_model_family(family: Optional[str]) -> str:
    f = str(family or "").strip().lower()
    if f in ("klein", "flux", "flux2", "flux.2"):
        return "flux"
    if f in ("z-image", "z_image", "zimage"):
        return "z_image"
    if f in ("qwen", "qwen_image", "qwen-image"):
        return "qwen"
    if f in ("sdxl", "sd15", "sd"):
        return "sdxl"
    return f or "any"


def check_lora_compatibility(lora_name: Optional[str], model_spec) -> dict:
    """Check whether a LoRA file is compatible with a given ModelSpec and its runtime engine.

    Returns dict with:
      compatible (bool), supported_by_engine (bool), lora_family (str),
      model_family (str), engine (str), reason (str)
    """
    engine = getattr(model_spec, "engine", "quantfunc") if model_spec else "sd_cpp"
    model_fam = normalize_model_family(getattr(model_spec, "family", "flux") if model_spec else "flux")

    if engine not in LORA_SUPPORTED_ENGINES:
        return {
            "compatible": False,
            "supported_by_engine": False,
            "lora_family": None,
            "model_family": model_fam,
            "engine": engine,
            "reason": (
                f"Движок {engine.upper()} не поддерживает динамическое подключение внешних LoRA-адаптеров. "
                "Используйте модель на движке Nunchaku (vitoom), SD.CPP (GGUF), QuantFunc или SDNQ."
            ),
        }

    if not lora_name:
        return {
            "compatible": True,
            "supported_by_engine": True,
            "lora_family": None,
            "model_family": model_fam,
            "engine": engine,
            "reason": "Движок поддерживает LoRA.",
        }

    safe_name = Path(str(lora_name)).name
    lora_path = (get_loras_dir() / safe_name).resolve()
    if lora_path.parent != get_loras_dir().resolve() or not lora_path.is_file():
        return {
            "compatible": False,
            "supported_by_engine": True,
            "lora_family": None,
            "model_family": model_fam,
            "engine": engine,
            "reason": f"Файл LoRA не найден: {safe_name}",
        }

    info = detect_lora_info(lora_path)
    lora_fam = normalize_model_family(info.get("family") or "any")

    if lora_fam not in ("any", "") and model_fam not in ("any", "") and lora_fam != model_fam:
        return {
            "compatible": False,
            "supported_by_engine": True,
            "lora_family": lora_fam,
            "model_family": model_fam,
            "engine": engine,
            "reason": (
                f"Несовместимая архитектура: LoRA предназначена для «{lora_fam}», "
                f"а выбранная модель относится к семейству «{model_fam}»."
            ),
        }

    return {
        "compatible": True,
        "supported_by_engine": True,
        "lora_family": lora_fam,
        "model_family": model_fam,
        "engine": engine,
        "reason": f"Совместимо ({lora_fam.upper()} на движке {engine.upper()}).",
    }

