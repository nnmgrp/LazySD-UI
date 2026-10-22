"""
Model sniffer — understands the *folder structure* of a model repository
(Hugging Face tree or a local directory) instead of guessing from file names only.

It answers four questions for the importer UI:
  1. What is each weight file?  (transformer / text encoder / VAE / quant sidecar)
  2. What layout is it?          (diffusers folders / flat GGUF / single checkpoint)
  3. Which runtime engine fits?  (SD.CPP / SDNQ / Nunchaku / OrbitQuant / QuantFunc)
  4. What is missing?            (so the host can supply encoder / VAE / base links)

Everything here is pure (no network / no disk writes) except ``scan_local_path``,
which only reads directory metadata.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

WEIGHT_EXTS = (".gguf", ".safetensors")

SUPPORTED_ENGINES: dict[str, str] = {
    "sd_cpp": "SD.CPP (GGUF / safetensors)",
    "sdnq": "SDNQ (Diffusers INT8/INT4)",
    "nunchaku": "Nunchaku (SVDQ)",
    "orbitquant": "OrbitQuant (W4A4 / W4A8)",
    "quantfunc": "QuantFunc engine",
}

# Engines whose models live in a Diffusers folder layout (model_index.json + subfolders).
DIFFUSERS_ENGINES = ("sdnq", "nunchaku", "orbitquant")

# Folder name -> role (checked on every parent segment, case-insensitive).
_ROLE_FOLDERS = {
    "transformer": "diffusion_model",
    "transformers": "diffusion_model",
    "unet": "diffusion_model",
    "diffusion_model": "diffusion_model",
    "diffusion_models": "diffusion_model",
    "dit": "diffusion_model",
    "vae": "vae",
    "text_encoder": "text_encoder",
    "text_encoder_2": "text_encoder",
    "text_encoder_3": "text_encoder",
    "text_encoders": "text_encoder",
    "clip": "text_encoder",
    "llm": "text_encoder",
}

_AUX_NAME_HINTS = ("orbitquant_", "codebook", "rotation", "quant_state", "svdq_meta")
_SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})(?P<ext>\.safetensors|\.gguf)$", re.I)
_IGNORED_DIRS = {".git", ".cache", "__pycache__", "images", "assets", "examples", ".huggingface"}

_QUANT_RE = re.compile(
    r"(q8_0|q6_k|q5_k_m|q5_k_s|q5_0|q4_k_m|q4_k_s|q4_k_xl|q4_0|q3_k_m|q3_k_s|q2_k|iq4_nl|iq4_xs|"
    r"bf16|fp16|f16|fp8|fp4|nvfp4|int8|int4|w4a4|w4a8|w4a16)",
    re.I,
)

_HF_FILE_RE = re.compile(
    r"^(?:https?://(?:www\.)?huggingface\.co/)?(?P<repo>[\w.\-]+/[\w.\-]+)/(?:blob|resolve)/(?P<rev>[^/]+)/(?P<path>[^?#]+)",
    re.I,
)
_HF_REPO_RE = re.compile(r"^(?:https?://(?:www\.)?huggingface\.co/)?(?P<repo>[\w.\-]+/[\w.\-]+?)(?:/(?:tree|blob|resolve)/.*)?/?$", re.I)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def safe_folder_name(text: str, fallback: str = "model") -> str:
    """Single path segment, safe on Windows (letters, digits, dot, dash, underscore, space)."""
    s = re.sub(r"[^\w.\- ]+", "_", str(text or "").strip(), flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip(" ._")
    return (s or fallback)[:80]


def normalize_hf_repo(text: str) -> str:
    """'https://huggingface.co/a/b/tree/main' | 'a/b' -> 'a/b' ('' when not a repo ref)."""
    t = str(text or "").strip().strip('"').strip("'")
    m = _HF_REPO_RE.match(t)
    return m.group("repo") if m else ""


def parse_hf_file_ref(text: str) -> Optional[dict]:
    """Parse an exact file reference.

    Accepted forms:
      https://huggingface.co/owner/repo/blob/main/folder/file.gguf
      https://huggingface.co/owner/repo/resolve/main/file.safetensors
      owner/repo::folder/file.gguf
    Returns {"repo_id", "path", "revision"} or None.
    """
    t = str(text or "").strip().strip('"').strip("'")
    if not t:
        return None
    m = _HF_FILE_RE.match(t)
    if m:
        return {"repo_id": m.group("repo"), "path": m.group("path"), "revision": m.group("rev")}
    if "::" in t:
        repo, _, path = t.partition("::")
        repo = normalize_hf_repo(repo)
        path = path.strip().lstrip("/")
        if repo and path:
            return {"repo_id": repo, "path": path, "revision": "main"}
    return None


def unique_local_name(path: str, taken: set[str]) -> str:
    """Flat local file name for a repo file; disambiguates ``model.safetensors`` in several folders."""
    p = str(path).replace("\\", "/")
    name = p.split("/")[-1]
    if name not in taken:
        taken.add(name)
        return name
    prefix = safe_folder_name("_".join(p.split("/")[:-1]) or "x", "x").replace(" ", "_")
    cand = f"{prefix}__{name}"
    i = 2
    while cand in taken:
        cand = f"{prefix}{i}__{name}"
        i += 1
    taken.add(cand)
    return cand


def _quant_of(name: str) -> str:
    m = _QUANT_RE.search(name)
    return m.group(1).upper() if m else "CUSTOM"


def te_kind(name: str) -> str:
    """Which sd.cpp text-encoder slot a file belongs to: t5xxl | clip_l | llm."""
    low = name.lower()
    if "t5" in low:
        return "t5xxl"
    if "clip" in low:
        return "clip_l"
    return "llm"


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classify_path(path: str, size_bytes: int = 0) -> dict:
    """Classify one weight file by its folder first, then by its name."""
    from .custom_models import detect_file_metadata  # lazy: avoid import cycle

    p = str(path).replace("\\", "/").lstrip("/")
    parts = p.split("/")
    name = parts[-1]
    folder = "/".join(parts[:-1])
    low_parts = [s.lower() for s in parts[:-1]]

    role: Optional[str] = None
    for seg in low_parts:
        if seg in _ROLE_FOLDERS:
            role = _ROLE_FOLDERS[seg]
            break
        if seg.startswith("text_encoder"):
            role = "text_encoder"
            break
    meta = detect_file_metadata(name)
    if role is None:
        low_name = name.lower()
        if low_name.startswith(_AUX_NAME_HINTS) or any(h in low_name for h in _AUX_NAME_HINTS):
            role = "aux"
        else:
            role = meta["role"]
    elif role == "diffusion_model":
        # a quant sidecar inside transformer/ is still a sidecar, not a variant
        low_name = name.lower()
        if low_name.startswith(_AUX_NAME_HINTS) or any(h in low_name for h in _AUX_NAME_HINTS):
            role = "aux"

    shard = _SHARD_RE.match(name)
    ext = Path(name).suffix.lower().lstrip(".")
    return {
        "path": p,
        "name": name,
        "folder": folder,
        "role": role,
        "ext": ext,
        "quant": _quant_of(name),
        "family": meta["family"],
        "steps": meta["steps"],
        "cfg": meta["cfg"],
        "size_gb": round((size_bytes or 0) / (1024 ** 3), 2),
        "shard_count": int(shard.group("total")) if shard else 0,
        "_shard_idx": int(shard.group("idx")) if shard else 0,
        "_shard_stem": (folder + "/" + shard.group("stem")) if shard else "",
    }


def _collapse_shards(entries: list[dict]) -> list[dict]:
    """Show a sharded weight set as ONE row (first shard path, summed size)."""
    out: list[dict] = []
    groups: dict[str, dict] = {}
    for e in entries:
        stem = e.get("_shard_stem")
        if not stem:
            out.append(e)
            continue
        g = groups.get(stem)
        if g is None:
            groups[stem] = e
            out.append(e)
        else:
            g["size_gb"] = round(g["size_gb"] + e["size_gb"], 2)
            if e["_shard_idx"] < g["_shard_idx"]:
                for k in ("path", "name", "_shard_idx"):
                    g[k] = e[k]
    for e in out:
        e.pop("_shard_idx", None)
        e.pop("_shard_stem", None)
    return out


# ---------------------------------------------------------------------------
# Family defaults (suggestions only — the host confirms or edits them)
# ---------------------------------------------------------------------------
def family_defaults(family: str, repo_or_name: str = "") -> dict:
    """Suggested encoder / VAE links for a family. Returned to the UI as *editable* prefill."""
    fam = (family or "").lower()
    low = (repo_or_name or "").lower()
    te: Optional[dict] = None
    vae: Optional[dict] = None
    if fam in ("moss-image", "moss_image", "moss") or "moss" in low:
        te = {"repo_id": "unsloth/Qwen3.5-9B-GGUF", "path": "Qwen3.5-9B-UD-Q4_K_XL.gguf",
              "size_gb": 5.97, "name": "Qwen3.5-9B UD-Q4_K_XL"}
        vae = {"repo_id": "unsloth/FLUX.2-VAE", "path": "split_files/vae/flux2-vae.safetensors", "size_gb": 0.34}
    elif fam == "qwen":
        te = {"repo_id": "unsloth/Qwen3-VL-8B-Instruct-GGUF", "path": "Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf",
              "size_gb": 5.1, "name": "Qwen3-VL-8B Q4_K_XL"}
        vae = {"repo_id": "madebyollin/texture-fix-vae-for-qwen-image-2.1",
               "path": "texture_fix_vae_for_qwen_image_2.1_bf16.safetensors", "size_gb": 0.64}
    elif fam in ("flux", "klein"):
        if "klein" in low:
            vae = {"repo_id": "Lightricks/LTX-2-comfy", "path": "vae/flux2-vae.safetensors", "size_gb": 0.34}
            if "9b" in low:
                te = {"repo_id": "Qwen/Qwen3-8B-GGUF", "path": "Qwen3-8B-Q8_0.gguf", "size_gb": 8.7,
                      "name": "Qwen3-8B Q8_0"}
            else:
                te = {"repo_id": "Qwen/Qwen3-VL-4B-Instruct-GGUF", "path": "Qwen3VL-4B-Instruct-Q8_0.gguf",
                      "size_gb": 4.3, "name": "Qwen3-VL-4B Q8_0"}
        else:
            vae = {"repo_id": "ffxvs/vae-flux", "path": "ae.safetensors", "size_gb": 0.34}
    return {"text_encoder": te, "vae": vae}


# ---------------------------------------------------------------------------
# Core sniffing
# ---------------------------------------------------------------------------
def suggest_engine(repo_id: str, names: list[str], tags: list[str], layout: str,
                   has_gguf: bool) -> tuple[str, list[str]]:
    """Pick the most likely runtime engine and explain why (Russian, shown to the host)."""
    blob = " ".join([repo_id, *names, *tags]).lower()
    if "orbitquant" in blob:
        return "orbitquant", ["В названии/файлах найден маркер OrbitQuant (codebooks/rotations)."]
    if "nunchaku" in blob or "svdq" in blob:
        return "nunchaku", ["В названии/файлах найден маркер Nunchaku / SVDQ."]
    if "sdnq" in blob:
        return "sdnq", ["В названии репозитория найден маркер SDNQ."]
    if has_gguf:
        return "sd_cpp", ["Найдены .gguf веса — их запускает stable-diffusion.cpp (SD.CPP)."]
    if layout == "diffusers":
        return "sdnq", ["Папочная Diffusers-раскладка (model_index.json) без явного маркера движка — "
                        "по умолчанию SDNQ. Проверьте и при необходимости смените движок."]
    return "sd_cpp", ["Одиночный .safetensors чекпоинт — по умолчанию SD.CPP."]


def sniff_files(files: list[dict], repo_id: str = "", tags: Optional[list[str]] = None) -> dict:
    """Analyse a flat list of ``{"path", "size"}`` (all files of a repo / directory).

    Weight files are classified; non-weight files are only used for layout hints.
    """
    tags = tags or []
    all_paths = [str(f.get("path", "")).replace("\\", "/").lstrip("/") for f in files]
    lower_paths = {p.lower() for p in all_paths}
    has_model_index = "model_index.json" in lower_paths

    entries: list[dict] = []
    for f in files:
        p = str(f.get("path", "")).replace("\\", "/").lstrip("/")
        low = p.lower()
        if not low.endswith(WEIGHT_EXTS) or "mmproj" in low:
            continue
        if any(seg.lower() in _IGNORED_DIRS for seg in p.split("/")[:-1]):
            continue
        e = classify_path(p, int(f.get("size") or 0))
        if f.get("abs_path"):
            e["abs_path"] = f["abs_path"]
        entries.append(e)
    entries = _collapse_shards(entries)

    comps: dict[str, list[dict]] = {"diffusion_model": [], "checkpoint": [], "text_encoder": [], "vae": [], "aux": []}
    for e in entries:
        comps.setdefault(e["role"], []).append(e)

    has_gguf = any(e["ext"] == "gguf" and e["role"] in ("diffusion_model", "checkpoint") for e in entries)
    has_folders = any(e["folder"] for e in entries)
    if not entries:
        layout = "empty"
    elif has_model_index:
        layout = "diffusers"
    elif has_gguf:
        layout = "gguf"
    elif has_folders:
        layout = "folders"
    else:
        layout = "flat"

    # Folder summary (what the host sees as the "tree")
    folders: dict[str, dict] = {}
    for e in entries:
        key = e["folder"] or "(корень)"
        row = folders.setdefault(key, {"folder": key, "role": e["role"], "files": 0, "size_gb": 0.0})
        row["files"] += 1
        row["size_gb"] = round(row["size_gb"] + e["size_gb"], 2)

    engine, reasons = suggest_engine(repo_id, [e["name"] for e in entries], tags, layout, has_gguf)

    main_weights = comps["diffusion_model"] + comps["checkpoint"]
    meta_family = (main_weights[0]["family"] if main_weights else "flux")
    fam_low = (repo_id + " " + " ".join(e["name"] for e in main_weights)).lower()
    if "moss" in fam_low:
        meta_family = "moss-image"
    elif "klein" in fam_low:
        meta_family = "flux"

    # What is missing for this layout/engine?
    missing: list[str] = []
    warnings: list[str] = []
    if not main_weights:
        warnings.append("В репозитории не найдено весов модели (.gguf / .safetensors).")
    else:
        needs_te = meta_family in ("flux", "qwen", "klein", "moss-image")
        if layout == "diffusers" or engine in DIFFUSERS_ENGINES:
            if not has_model_index:
                missing.append("base")
            else:
                if not comps["text_encoder"] and needs_te:
                    missing.append("text_encoder")
                if not comps["vae"]:
                    missing.append("vae")
        else:
            if needs_te and not comps["text_encoder"]:
                missing.append("text_encoder")
            if meta_family != "sdxl" and not comps["vae"]:
                missing.append("vae")

        if missing:
            labels = {"text_encoder": "текстовый энкодер", "vae": "VAE", "base": "базовая модель (конфиги, энкодер, VAE)"}
            miss_txt = ", ".join(labels[m] for m in missing)
            if len(main_weights) == 1 and not comps["text_encoder"] and not comps["vae"]:
                warnings.append(
                    "В репозитории найдена только одна модель, без энкодера и VAE. "
                    f"Не хватает: {miss_txt}. Укажите точные ссылки на энкодер, VAE и модель "
                    "и задайте название папки — ниже подставлены рекомендуемые значения, их можно изменить."
                )
            else:
                warnings.append(
                    f"Не хватает: {miss_txt}. Укажите ссылки вручную или выберите готовые файлы "
                    "из дерева репозитория."
                )
        if any(e["shard_count"] for e in main_weights):
            warnings.append(
                "Веса модели разбиты на части (shards). Они скачиваются целиком вместе с папкой модели."
            )
        if len(comps["diffusion_model"]) > 1 and any(e["path"].lower().endswith("/model.safetensors") or
                                                     e["name"].lower() == "model.safetensors" for e in comps["diffusion_model"]):
            warnings.append("Несколько файлов с одним именем в разных папках — смотрите колонку «Папка».")

    return {
        "layout": layout,
        "has_model_index": has_model_index,
        "folders": sorted(folders.values(), key=lambda r: r["folder"]),
        "components": comps,
        "suggested_engine": engine,
        "engine_reasons": reasons,
        "suggested_family": meta_family,
        "missing": missing,
        "needs_manual": bool(missing),
        "warnings": warnings,
        "suggested_folder": safe_folder_name(repo_id.split("/")[-1] if repo_id else ""),
        "defaults": family_defaults(meta_family, repo_id + " " + " ".join(e["name"] for e in main_weights)),
    }


# ---------------------------------------------------------------------------
# Local directories
# ---------------------------------------------------------------------------
def scan_local_path(raw_path: str, max_files: int = 4000) -> dict:
    """Sniff a local file or directory. Entries carry ``abs_path`` for registration."""
    p = Path(str(raw_path or "").strip().strip('"').strip("'")).expanduser()
    if not p.exists():
        raise ValueError(f"Путь не найден: {p}")

    files: list[dict] = []
    if p.is_file():
        root = p.parent
        files.append({"path": p.name, "size": p.stat().st_size, "abs_path": str(p.resolve())})
        # Also discover sibling text_encoder / vae files in the same folder
        try:
            for sib in sorted(root.rglob("*")):
                if not sib.is_file() or sib.resolve() == p.resolve():
                    continue
                rel_parts = sib.relative_to(root).parts
                if any(seg in _IGNORED_DIRS for seg in rel_parts[:-1]):
                    continue
                if sib.suffix.lower() in WEIGHT_EXTS:
                    rel_str = "/".join(rel_parts)
                    cls_info = classify_path(rel_str, sib.stat().st_size)
                    if cls_info["role"] in ("text_encoder", "vae"):
                        files.append({
                            "path": rel_str,
                            "size": sib.stat().st_size,
                            "abs_path": str(sib.resolve()),
                        })
        except OSError:
            pass
    else:
        root = p
        count = 0
        for f in sorted(p.rglob("*")):
            if count >= max_files:
                break
            try:
                if not f.is_file():
                    continue
                rel_parts = f.relative_to(p).parts
                if any(seg in _IGNORED_DIRS for seg in rel_parts[:-1]):
                    continue
                if f.suffix.lower() in WEIGHT_EXTS or f.name.lower() == "model_index.json":
                    files.append({
                        "path": "/".join(rel_parts),
                        "size": f.stat().st_size,
                        "abs_path": str(f.resolve()),
                    })
                    count += 1
            except OSError:
                continue

    info = sniff_files(files, repo_id=root.name)
    info["root"] = str(root.resolve())
    info["suggested_folder"] = safe_folder_name(root.name)
    info["name"] = root.name.replace("_", " ").replace("-", " ").strip() or "Local model"
    return info
