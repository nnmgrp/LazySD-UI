"""Tests for base-model transformer editing on model cards."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _manifest(model_id: str = "custom_transformer_card") -> dict:
    return {
        "id": model_id,
        "name": "Transformer Card",
        "family": "qwen",
        "pipeline": "text_to_image",
        "engine": "sd_cpp",
        "engine_version": "sd_cpp",
        "is_custom": True,
        "sources": {"primary": {"type": "huggingface", "repo_id": "org/qwen-gguf"}},
        "engine_args": {"diffusion_model": "old.gguf"},
        "text_encoders": [],
        "scheduler_presets": {"default": {"steps": 20, "cfg": 1.0}},
        "default_cfg": 1.0,
    }


def test_normalize_transformers_unique_ids_and_primary_fallback():
    from backend.custom_models import _normalize_transformers

    normalized = _normalize_transformers(
        [
            {"id": "q4", "path": "weights/q4.gguf", "size_gb": 3.91, "steps": 28},
            {
                "id": "q4",
                "path": "weights/q5.gguf",
                "repo_id": "realrebelai/qwen-gguf",
                "size_bytes": 5 * 1024 ** 3,
                "recommended": False,
            },
        ],
        primary_repo="org/qwen-gguf",
        engine="sd_cpp",
    )

    assert [item["id"] for item in normalized] == ["q4", "q4_2"]
    assert normalized[0]["repo_id"] == "org/qwen-gguf"
    assert normalized[1]["repo_id"] == "realrebelai/qwen-gguf"
    assert normalized[0]["recommended"] is True
    assert normalized[1]["recommended"] is False
    assert normalized[1]["size_gb"] == 5.0


def test_normalize_transformers_keeps_local_absolute_paths():
    from backend.custom_models import _normalize_transformers

    local = str(Path("C:/models/custom/base.gguf")).replace("\\", "/")
    normalized = _normalize_transformers([{"path": local}], primary_repo="", engine="sd_cpp")

    assert normalized[0]["path"] == "C:/models/custom/base.gguf"
    assert normalized[0]["repo_id"] == ""
    assert normalized[0]["recommended"] is True


def test_apply_card_updates_syncs_recommended_transformer_to_engine_args():
    from backend.custom_models import _apply_card_updates

    updated = _apply_card_updates(
        _manifest(),
        {
            "transformers": [
                {"path": "weights/q4.gguf", "repo_id": "org/qwen-gguf"},
                {"path": "weights/rebel-q4.gguf", "repo_id": "realrebelai/qwen-gguf", "recommended": True},
            ],
        },
    )

    assert updated["engine_args"]["diffusion_model"] == "weights/rebel-q4.gguf"
    assert all("installed" not in item for item in updated["transformers"])


def test_apply_card_updates_uses_model_key_for_quantfunc_engine():
    from backend.custom_models import _apply_card_updates

    updated = _apply_card_updates(
        _manifest(),
        {
            "engine": "quantfunc",
            "transformers": [{"path": "transformer/model.safetensors", "repo_id": "org/qwen-gguf"}],
        },
    )

    assert updated["engine_args"]["model"] == "transformer/model.safetensors"
    assert "diffusion_model" not in updated["engine_args"]


def test_apply_card_updates_keeps_sdxl_checkpoint_alias_in_sync():
    from backend.custom_models import _apply_card_updates

    manifest = _manifest()
    manifest["family"] = "sdxl"
    updated = _apply_card_updates(
        manifest,
        {"transformers": [{"path": "sdxl-turbo-Q4_0.gguf", "repo_id": "org/qwen-gguf"}]},
    )

    assert updated["engine_args"]["diffusion_model"] == "sdxl-turbo-Q4_0.gguf"
    assert updated["engine_args"]["checkpoint"] == "sdxl-turbo-Q4_0.gguf"
    assert updated["engine_args"]["model"] == "sdxl-turbo-Q4_0.gguf"


def test_get_custom_model_card_reports_transformer_cache_state(tmp_path, monkeypatch):
    import backend.cache as cache_module
    import backend.custom_models as cm

    class FakeCache:
        def __init__(self, *args, **kwargs):
            pass

        def is_installed(self, repo_id, file_path):
            return file_path == "weights/installed.gguf"

        def repo_dir_for(self, repo_id):
            return tmp_path / "missing-repo"

    manifest = _manifest()
    manifest["transformers"] = [
        {"id": "installed", "path": "weights/installed.gguf", "repo_id": "org/qwen-gguf"},
        {"id": "missing", "path": "weights/missing.gguf", "repo_id": "org/qwen-gguf"},
    ]
    (tmp_path / "custom_transformer_card.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    monkeypatch.setattr(cache_module, "CacheManager", FakeCache)
    monkeypatch.setattr(cm, "_card_registry_dir", lambda: tmp_path)

    card = cm.get_custom_model_card("custom_transformer_card")

    assert card["transformers"][0]["installed"] is True
    assert card["transformers"][1]["installed"] is False


def test_update_card_atomically_saves_transformers_without_installed_flag(tmp_path, monkeypatch):
    import backend.custom_models as cm

    manifest = _manifest()
    path = tmp_path / "custom_transformer_card.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    monkeypatch.setattr(cm, "_card_registry_dir", lambda: tmp_path)
    monkeypatch.setattr(cm, "reload_registry", lambda: SimpleNamespace())
    monkeypatch.setattr(cm, "get_registry", lambda: SimpleNamespace(get=lambda mid: SimpleNamespace(id=mid)))

    cm.update_custom_model_card(
        "custom_transformer_card",
        {
            "transformers": [{
                "id": "q4",
                "path": "weights/q4.gguf",
                "repo_id": "org/qwen-gguf",
                "installed": True,
                "recommended": True,
            }],
        },
    )

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["transformers"][0]["id"] == "q4"
    assert "installed" not in saved["transformers"][0]
    assert saved["engine_args"]["diffusion_model"] == "weights/q4.gguf"


def test_sniffer_includes_bin_and_ckpt_transformer_weights():
    from backend.model_sniffer import sniff_files

    result = sniff_files(
        [
            {"path": "model.ckpt", "size": 4_000_000_000},
            {"path": "transformer/model.bin", "size": 2_000_000_000},
            {"path": "text_encoder/pytorch_model.bin", "size": 1_000_000_000},
        ],
        repo_id="org/checkpoint-repo",
    )

    diffusion = {item["path"] for item in result["components"]["diffusion_model"]}
    assert diffusion == {"model.ckpt", "transformer/model.bin"}
    assert [item["path"] for item in result["components"]["text_encoder"]] == ["text_encoder/pytorch_model.bin"]


def test_normalize_transformers_accepts_posix_local_weight_paths():
    from backend.custom_models import _normalize_transformers

    normalized = _normalize_transformers(
        [{"path": "/models/custom/base.gguf", "repo_id": "org/qwen-gguf"}],
        primary_repo="org/qwen-gguf",
        engine="sd_cpp",
    )

    assert normalized[0]["path"] == "/models/custom/base.gguf"
    assert normalized[0]["repo_id"] == "org/qwen-gguf"


@pytest.mark.parametrize("bad_path", ["../escape.gguf", "weights/../../escape.gguf"])
def test_normalize_transformers_rejects_unsafe_repo_paths(bad_path):
    from backend.custom_models import _normalize_transformers

    with pytest.raises(ValueError):
        _normalize_transformers(
            [{"path": bad_path, "repo_id": ""}],
            primary_repo="",
            engine="sd_cpp",
        )


def test_normalize_transformers_rejects_non_weight_paths():
    from backend.custom_models import _normalize_transformers

    with pytest.raises(ValueError):
        _normalize_transformers(
            [{"path": "/etc/passwd", "repo_id": "org/qwen-gguf"}],
            primary_repo="org/qwen-gguf",
            engine="sd_cpp",
        )


def test_loader_accepts_transformer_name_and_extra_fields():
    from backend.registry.loader import ModelSpec

    data = {
        "id": "qwen_test_turbo",
        "name": "Qwen Turbo",
        "family": "qwen",
        "pipeline": "text_to_image",
        "transformers": [
            {
                "id": "q4_k_m",
                "name": "Unsloth Q4_K_M Medium",
                "path": "qwen-turbo.gguf",
                "tier": "universal",
                "backend": "sd_cpp",
                "steps": 8,
                "unknown_key": "safe_to_ignore",
            }
        ],
        "requirements": {"vram_min_gb": 8, "future_feature_flag": True},
    }

    spec = ModelSpec.from_dict(data)
    assert len(spec.transformers) == 1
    assert spec.transformers[0].id == "q4_k_m"
    assert spec.transformers[0].name == "Unsloth Q4_K_M Medium"
