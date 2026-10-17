"""
Model Visibility Manager — controls host and guest model visibility.

Settings are stored persistently in models/model_visibility.json on the host PC:
{
  "hidden_models": {
    "model_id": {
      "hidden_for_guests": true,
      "hidden_from_host": false
    }
  }
}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .settings import get_settings


def _get_visibility_file() -> Path:
    settings = get_settings()
    p = settings.models_dir.parent / "model_visibility.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_visibility_map() -> Dict[str, Dict[str, bool]]:
    """Return map of model_id -> {"hidden_for_guests": bool, "hidden_from_host": bool}."""
    fpath = _get_visibility_file()
    if not fpath.exists():
        return {}
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("hidden_models", {})
    except Exception as e:
        print(f"[visibility] Error loading {fpath}: {e}")
        return {}


def save_visibility_map(vmap: Dict[str, Dict[str, bool]]) -> None:
    """Save visibility map to disk."""
    fpath = _get_visibility_file()
    try:
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump({"hidden_models": vmap}, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[visibility] Error saving {fpath}: {e}")


def get_model_visibility(model_id: str) -> Dict[str, bool]:
    """Get visibility flags for a specific model."""
    vmap = load_visibility_map()
    return vmap.get(model_id, {"hidden_for_guests": False, "hidden_from_host": False})


def set_model_visibility(
    model_id: str,
    hidden_for_guests: bool = False,
    hidden_from_host: bool = False,
) -> Dict[str, bool]:
    """Update visibility flags for a model."""
    vmap = load_visibility_map()
    if not hidden_for_guests and not hidden_from_host:
        if model_id in vmap:
            del vmap[model_id]
    else:
        vmap[model_id] = {
            "hidden_for_guests": bool(hidden_for_guests),
            "hidden_from_host": bool(hidden_from_host),
        }
    save_visibility_map(vmap)
    return vmap.get(model_id, {"hidden_for_guests": False, "hidden_from_host": False})


def remove_model_from_visibility(model_id: str) -> None:
    """Remove model from visibility map when deleted."""
    vmap = load_visibility_map()
    if model_id in vmap:
        del vmap[model_id]
        save_visibility_map(vmap)
