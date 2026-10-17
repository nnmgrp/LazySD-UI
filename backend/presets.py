"""
Presets & Templates Manager for LazySD UI.

Stores factory and user-created generation templates in `models/presets.json`
(or `presets/` directory) and provides CRUD + import/export helpers.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Optional

from .settings import get_settings


FACTORY_PRESETS: list[dict[str, Any]] = [
    {
        "id": "manga_screentone_restore",
        "name": "Реставрация манги и скринтонов (100% шагов)",
        "category": "Manga & Lineart",
        "icon": "✒️",
        "description": "Чистые контуры туши и сохранение скринтонов через Native Control / Reference без зашумления белого фона.",
        "prompt_template": "{prompt}, high-fidelity manga restoration, crisp black ink lineart, authentic screentone patterns, pure white paper background, masterpiece",
        "negative_prompt": "blurry, grey fog, dirty paper, smudged ink, noise artifacts, jpeg artifacts, color fringing, low resolution",
        "settings": {
            "model_id": "flux_3_image_klein_gguf",
            "ref_mode": "control_ref",
            "strength": 0.85,
            "width": 1024,
            "height": 1024,
            "steps": 20,
            "cfg": 4.0,
            "sampler": "euler",
            "noise": 0.0,
        },
        "is_factory": True,
    },
    {
        "id": "lineart_canny_clean",
        "name": "Контурный лайн-арт (ControlNet Canny)",
        "category": "Manga & Lineart",
        "icon": "📐",
        "description": "Строгое повторение контуров туши через препроцессор Canny при 100% шагов с чистого листа.",
        "prompt_template": "{prompt}, clean professional ink lineart, sharp monochrome illustration, pure white background, high contrast",
        "negative_prompt": "blurry lines, broken contours, grey background, artifacts, low quality",
        "settings": {
            "model_id": "flux_2_klein_base_9b_gguf",
            "ref_mode": "controlnet_canny",
            "strength": 0.90,
            "width": 1024,
            "height": 1024,
            "steps": 20,
            "cfg": 4.0,
            "sampler": "euler",
            "noise": 0.0,
        },
        "is_factory": True,
    },
    {
        "id": "anime_cel_qwen",
        "name": "2D Аниме и концепт-арт (Qwen-Image)",
        "category": "Anime & Illustration",
        "icon": "🎨",
        "description": "Яркая аниме-стилистика с чистой заливкой и точным следованием длинному промпту на базе Qwen-Image 2.1.",
        "prompt_template": "{prompt}, vibrant 2D anime illustration, clean cel shading, expressive lighting, detailed character design, key visual",
        "negative_prompt": "bad anatomy, deformed hands, extra fingers, blurry, 3d render, photorealistic, watermark, text",
        "settings": {
            "model_id": "qwen_image_2_1_gguf",
            "ref_mode": "control_ref",
            "strength": 0.85,
            "width": 1024,
            "height": 1024,
            "steps": 20,
            "cfg": 3.0,
            "sampler": "euler",
            "noise": 0.0,
            "vae": "texture_fix",
        },
        "is_factory": True,
    },
    {
        "id": "studio_macro_photoreal",
        "name": "Студийный фотореализм (Macro / Portrait)",
        "category": "Photorealism",
        "icon": "📸",
        "description": "Детализированный фотореализм с естественной текстурой кожи и мягким боке.",
        "prompt_template": "{prompt}, raw photo, 85mm f/1.4 lens, natural skin texture, studio rim lighting, ultra-detailed, 8k uhd",
        "negative_prompt": "plastic skin, airbrushed, cgi, 3d, doll, oversaturated, blurry, deformed",
        "settings": {
            "model_id": "flux_2_klein_base_9b_gguf",
            "width": 1344,
            "height": 768,
            "steps": 24,
            "cfg": 3.5,
            "sampler": "euler_a",
            "noise": 0.05,
        },
        "is_factory": True,
    },
    {
        "id": "fast_draft_4step",
        "name": "Быстрый черновик (4 шага Turbo / Schnell)",
        "category": "Speed / Draft",
        "icon": "⚡",
        "description": "Молниеносная генерация за 4–8 шагов для проверки идеи и композиции.",
        "prompt_template": "{prompt}",
        "negative_prompt": "",
        "settings": {
            "model_id": "flux_3_image_klein_gguf",
            "width": 1024,
            "height": 1024,
            "steps": 4,
            "cfg": 1.0,
            "sampler": "euler",
            "noise": 0.0,
        },
        "is_factory": True,
    },
]


def _get_presets_path() -> Path:
    settings = get_settings()
    presets_dir = settings.models_dir.parent / "presets"
    presets_dir.mkdir(parents=True, exist_ok=True)
    return presets_dir / "presets.json"


def _load_all_presets() -> list[dict[str, Any]]:
    path = _get_presets_path()
    if not path.exists():
        # Initialize with factory presets so that even factory presets can be edited and deleted by the host!
        initial = []
        for p in FACTORY_PRESETS:
            item = dict(p)
            item["owner_id"] = "host"
            item["owner_name"] = "Хост"
            item["is_private"] = False
            item["builtin"] = True
            initial.append(item)
        _save_all_presets(initial)
        return initial
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def _save_all_presets(presets: list[dict[str, Any]]) -> None:
    path = _get_presets_path()
    path.write_text(json.dumps(presets, indent=2, ensure_ascii=False), encoding="utf-8")


def _normalize_preset(p: dict[str, Any], caller_account: Optional[dict] = None) -> dict[str, Any]:
    """Ensure every preset exposes both flat and nested fields for UI & API."""
    out = dict(p)
    st = dict(out.get("settings") or {})
    title = out.get("title") or out.get("name") or "Preset"
    out["title"] = title
    out["name"] = title
    
    owner_id = out.get("owner_id") or ("host" if out.get("is_factory") or out.get("builtin") else "host")
    out["owner_id"] = owner_id
    out["owner_name"] = out.get("owner_name") or ("Хост" if owner_id == "host" else "Гость")
    out["is_private"] = bool(out.get("is_private", False))

    caller_id = (caller_account or {}).get("id", "host")
    is_guest = bool((caller_account or {}).get("is_guest", False))

    if not is_guest or caller_id in ("host", "host_remote"):
        # Host can edit and delete ANY preset (including factory presets)
        out["can_edit"] = True
        out["can_delete"] = True
    else:
        # Guests can only edit/delete their own presets
        out["can_edit"] = (owner_id == caller_id)
        out["can_delete"] = (owner_id == caller_id)

    raw_prompt = out.get("prompt") or out.get("prompt_template") or ""
    clean_prompt = raw_prompt.replace("{prompt}, ", "").replace("{prompt}", "").strip()
    out["prompt"] = clean_prompt
    out["prompt_template"] = out.get("prompt_template") or raw_prompt

    w = st.get("width", 1024)
    h = st.get("height", 1024)
    out["resolution"] = out.get("resolution") or st.get("resolution") or f"{w}x{h}"
    out["model_id"] = out.get("model_id") or st.get("model_id")
    out["variant_id"] = out.get("variant_id") or st.get("variant_id")
    out["ref_mode"] = out.get("ref_mode") or st.get("ref_mode") or "control_ref"
    out["strength"] = out.get("strength") if out.get("strength") is not None else st.get("strength", 0.85)
    out["steps"] = out.get("steps") or st.get("steps") or 20
    out["cfg"] = out.get("cfg") if out.get("cfg") is not None else st.get("cfg", 3.5)
    out["sampler"] = out.get("sampler") or st.get("sampler") or "euler"
    out["noise"] = out.get("noise") if out.get("noise") is not None else st.get("noise", 0.0)
    return out


def list_presets(caller_account: Optional[dict] = None) -> list[dict[str, Any]]:
    """Return filtered presets list based on caller role & ownership."""
    all_presets = _load_all_presets()
    caller = caller_account or {"role": "host", "id": "host", "is_guest": False}
    caller_id = caller.get("id", "host")
    is_guest = bool(caller.get("is_guest", False))

    results = []
    for p in all_presets:
        owner_id = p.get("owner_id") or "host"
        is_private = bool(p.get("is_private", False))

        if is_guest:
            # Guests see all host presets + their own presets. They never see other guests' private presets.
            if owner_id in ("host", "host_remote") or owner_id == caller_id:
                results.append(_normalize_preset(p, caller))
        else:
            # Host sees everything (including guest presets)
            results.append(_normalize_preset(p, caller))
    return results


def save_user_preset(data: dict[str, Any], caller_account: Optional[dict] = None) -> dict[str, Any]:
    """Create or update a preset with permission checks."""
    caller = caller_account or {"role": "host", "id": "host", "username": "Хост", "is_guest": False}
    caller_id = caller.get("id", "host")
    is_guest = bool(caller.get("is_guest", False))

    name = (data.get("title") or data.get("name") or "").strip()
    if not name:
        raise ValueError("Название шаблона обязательно")

    raw_id = (data.get("id") or "").strip()
    all_presets = _load_all_presets()

    # If updating an existing preset
    existing_idx = None
    existing_preset = None
    if raw_id:
        for idx, item in enumerate(all_presets):
            if item.get("id") == raw_id:
                existing_idx = idx
                existing_preset = item
                break

    if existing_preset is not None:
        # Check permission to edit
        owner_id = existing_preset.get("owner_id") or "host"
        if is_guest and owner_id != caller_id:
            raise PermissionError("Гость не может редактировать чужие шаблоны или шаблоны хоста")
        owner_name = existing_preset.get("owner_name") or caller.get("username", "Пользователь")
    else:
        # Creating a new preset
        slug = re.sub(r"[^\w\s-]", "", name.lower())
        slug = re.sub(r"[\s_-]+", "_", slug).strip("_") or "preset"
        raw_id = f"preset_{slug}_{int(time.time()) % 100000}"
        owner_id = caller_id if is_guest else "host"
        owner_name = caller.get("username", "Гость" if is_guest else "Хост")

    st = dict(data.get("settings") or {})
    for key in ("model_id", "variant_id", "ref_mode", "strength", "resolution", "steps", "cfg", "sampler", "noise"):
        if key in data and data[key] is not None:
            st[key] = data[key]

    is_private = bool(data.get("is_private", False))

    preset = {
        "id": raw_id,
        "name": name,
        "title": name,
        "category": (data.get("category") or ("Гостевые" if is_guest else "Пользовательские")).strip(),
        "icon": (data.get("icon") or ("👤" if is_guest else "⭐")).strip(),
        "description": (data.get("description") or "").strip(),
        "prompt": (data.get("prompt") or data.get("prompt_template") or "").strip(),
        "prompt_template": (data.get("prompt_template") or data.get("prompt") or "").strip(),
        "negative_prompt": (data.get("negative_prompt") or "").strip(),
        "settings": st,
        "owner_id": owner_id,
        "owner_name": owner_name,
        "is_private": is_private,
        "is_factory": False,
        "builtin": False,
        "updated_at": int(time.time()),
    }

    if existing_idx is not None:
        all_presets[existing_idx] = preset
    else:
        preset["created_at"] = int(time.time())
        all_presets.insert(0, preset)

    _save_all_presets(all_presets)
    return _normalize_preset(preset, caller)


def delete_user_preset(preset_id: str, caller_account: Optional[dict] = None) -> bool:
    """Delete a preset with permission checks."""
    caller = caller_account or {"role": "host", "id": "host", "is_guest": False}
    caller_id = caller.get("id", "host")
    is_guest = bool(caller.get("is_guest", False))

    all_presets = _load_all_presets()
    target_idx = None
    target_preset = None
    for idx, item in enumerate(all_presets):
        if item.get("id") == preset_id:
            target_idx = idx
            target_preset = item
            break

    if target_preset is None:
        return False

    owner_id = target_preset.get("owner_id") or "host"
    if is_guest and owner_id != caller_id:
        raise PermissionError("Гость не может удалять шаблоны хоста или других пользователей")

    all_presets.pop(target_idx)
    _save_all_presets(all_presets)
    return True


def restore_factory_presets() -> list[dict[str, Any]]:
    """Host helper to reset/repopulate factory presets."""
    all_presets = _load_all_presets()
    existing_ids = {p.get("id") for p in all_presets}
    for p in FACTORY_PRESETS:
        if p["id"] not in existing_ids:
            item = dict(p)
            item["owner_id"] = "host"
            item["owner_name"] = "Хост"
            item["is_private"] = False
            item["builtin"] = True
            all_presets.append(item)
    _save_all_presets(all_presets)
    return list_presets()


