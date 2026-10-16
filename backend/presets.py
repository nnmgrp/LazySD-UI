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


def _get_user_presets_path() -> Path:
    settings = get_settings()
    presets_dir = settings.models_dir.parent / "presets"
    presets_dir.mkdir(parents=True, exist_ok=True)
    return presets_dir / "user_presets.json"


def _load_user_presets() -> list[dict[str, Any]]:
    path = _get_user_presets_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def _save_user_presets(presets: list[dict[str, Any]]) -> None:
    path = _get_user_presets_path()
    path.write_text(json.dumps(presets, indent=2, ensure_ascii=False), encoding="utf-8")


def _normalize_preset(p: dict[str, Any]) -> dict[str, Any]:
    """Ensure every preset exposes both flat and nested fields for UI & API."""
    out = dict(p)
    st = dict(out.get("settings") or {})
    title = out.get("title") or out.get("name") or "Preset"
    out["title"] = title
    out["name"] = title
    builtin = bool(out.get("builtin") if "builtin" in out else out.get("is_factory", False))
    out["builtin"] = builtin
    out["is_factory"] = builtin

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


def list_presets() -> list[dict[str, Any]]:
    """Return combined list of factory presets and user presets."""
    user_presets = _load_user_presets()
    return [_normalize_preset(p) for p in (FACTORY_PRESETS + user_presets)]


def save_user_preset(data: dict[str, Any]) -> dict[str, Any]:
    """Create or update a user preset."""
    name = (data.get("title") or data.get("name") or "").strip()
    if not name:
        raise ValueError("Название шаблона обязательно")

    raw_id = (data.get("id") or "").strip()
    if not raw_id:
        slug = re.sub(r"[^\w\s-]", "", name.lower())
        slug = re.sub(r"[\s_-]+", "_", slug).strip("_") or "preset"
        raw_id = f"user_{slug}_{int(time.time()) % 100000}"

    st = dict(data.get("settings") or {})
    for key in ("model_id", "variant_id", "ref_mode", "strength", "resolution", "steps", "cfg", "sampler", "noise"):
        if key in data and data[key] is not None:
            st[key] = data[key]

    preset = _normalize_preset({
        "id": raw_id,
        "name": name,
        "title": name,
        "category": (data.get("category") or "Пользовательские").strip(),
        "icon": (data.get("icon") or "⭐").strip(),
        "description": (data.get("description") or "").strip(),
        "prompt": (data.get("prompt") or data.get("prompt_template") or "").strip(),
        "prompt_template": (data.get("prompt_template") or data.get("prompt") or "").strip(),
        "negative_prompt": (data.get("negative_prompt") or "").strip(),
        "settings": st,
        "is_factory": False,
        "builtin": False,
        "created_at": int(time.time()),
    })

    user_presets = _load_user_presets()
    replaced = False
    for idx, existing in enumerate(user_presets):
        if existing.get("id") == raw_id:
            user_presets[idx] = preset
            replaced = True
            break
    if not replaced:
        user_presets.insert(0, preset)

    _save_user_presets(user_presets)
    return preset


def delete_user_preset(preset_id: str) -> bool:
    """Delete a user preset by ID."""
    user_presets = _load_user_presets()
    filtered = [p for p in user_presets if p.get("id") != preset_id]
    if len(filtered) == len(user_presets):
        return False
    _save_user_presets(filtered)
    return True

