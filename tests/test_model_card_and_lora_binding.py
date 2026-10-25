"""Tests for editable model cards and explicit LoRA-to-model binding.

The model-card API test uses a temporary registry directory, so no real manifest
is modified. LoRA compatibility tests use fake metadata and a temporary library
folder; no weights are touched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _custom_manifest(model_id: str = "custom_test_model") -> dict:
    return {
        "id": model_id,
        "name": "Test Model",
        "family": "flux",
        "pipeline": "text_to_image",
        "engine": "sd_cpp",
        "engine_version": "sd_cpp",
        "is_custom": True,
        "sources": {"primary": {"type": "huggingface", "repo_id": "org/test-model"}},
        "engine_args": {},
        "text_encoders": [],
        "scheduler_presets": {"default": {"steps": 20, "cfg": 1.0}},
        "default_cfg": 1.0,
    }


# ---------------------------------------------------------------------------
# Card normalization
# ---------------------------------------------------------------------------

def test_apply_card_updates_syncs_engine_args_and_presets():
    from backend.custom_models import _apply_card_updates

    manifest = _custom_manifest()
    updated = _apply_card_updates(manifest, {
        "name": "  Renamed model  ",
        "description": "A test card",
        "family": "flux",
        "engine": "orbitquant",
        "steps": 6,
        "cfg": 1.25,
        "license": "cc-by-nc-4.0",
        "base_repo_id": "briaai/Fibo-2.0-edit-turbo",
        "text_encoders": [{
            "id": "te_main",
            "name": "Repo encoder",
            "source": {"repo_id": "org/test-model", "file_path": "text_encoder/model.safetensors"},
            "precision": "bf16",
            "size_gb": 3.19,
        }],
        "vae": {"repo_id": "org/test-model", "path": "vae/diffusion_pytorch_model.safetensors"},
        "aux_files": [{"path": "transformer/orbitquant_codebooks.safetensors"}],
    })

    assert updated["name"] == "Renamed model"
    assert updated["engine"] == "orbitquant"
    assert updated["engine_version"] == "orbitquant"
    assert updated["base_models"] == {"50x-above": ".", "50x-below": "."}
    assert updated["base_repo_id"] == "briaai/Fibo-2.0-edit-turbo"
    assert updated["default_cfg"] == 1.25
    assert updated["scheduler_presets"]["default"]["steps"] == 6
    assert updated["scheduler_presets"]["default"]["cfg"] == 1.25

    encoder = updated["text_encoders"][0]
    assert encoder["repo"] == "org/test-model"
    assert encoder["path"] == "text_encoder/model.safetensors"
    assert encoder["recommended"] is True
    args = updated["engine_args"]
    assert args["llm"] == "text_encoder/model.safetensors"
    assert args["llm_repo"] == "org/test-model"
    assert args["vae"] == "vae/diffusion_pytorch_model.safetensors"
    assert args["vae_repo"] == "org/test-model"
    assert args["aux_files"][0]["repo_id"] == "org/test-model"


def test_apply_card_updates_rejects_unsafe_component_paths():
    from backend.custom_models import _apply_card_updates

    manifest = _custom_manifest()
    with pytest.raises(ValueError, match="за пределы"):
        _apply_card_updates(manifest, {
            "vae": {"repo_id": "org/test-model", "path": "../escape.safetensors"},
        })

    manifest = _custom_manifest()
    with pytest.raises(ValueError, match="относительный"):
        _apply_card_updates(manifest, {
            "text_encoders": [{
                "repo_id": "org/test-model",
                "path": "C:/absolute/model.safetensors",
            }],
        })

    manifest = _custom_manifest()
    with pytest.raises(ValueError, match="относительный"):
        _apply_card_updates(manifest, {
            "text_encoders": [{
                "repo_id": "org/test-model",
                "path": "/etc/passwd",
            }],
        })


def test_apply_card_updates_validates_numeric_ranges():
    from backend.custom_models import _apply_card_updates

    with pytest.raises(ValueError, match="steps"):
        _apply_card_updates(_custom_manifest(), {"steps": 0})
    with pytest.raises(ValueError, match="cfg"):
        _apply_card_updates(_custom_manifest(), {"cfg": 99})
    with pytest.raises(ValueError, match="движок"):
        _apply_card_updates(_custom_manifest(), {"engine": "unknown-engine"})


# ---------------------------------------------------------------------------
# Card HTTP API (temporary registry)
# ---------------------------------------------------------------------------

def _patch_card_registry(monkeypatch, tmp_path):
    import backend.custom_models as cm

    monkeypatch.setattr(cm, "_card_registry_dir", lambda: tmp_path)
    monkeypatch.setattr(cm, "reload_registry", lambda: SimpleNamespace())
    monkeypatch.setattr(cm, "get_registry", lambda: SimpleNamespace(get=lambda mid: SimpleNamespace(id=mid)))


def test_card_api_reads_and_updates_custom_manifest(tmp_path, monkeypatch):
    import app as app_module
    from fastapi.testclient import TestClient

    _patch_card_registry(monkeypatch, tmp_path)
    manifest = _custom_manifest()
    path = tmp_path / "custom_test_model.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    client = TestClient(app_module.create_app())

    read = client.get("/api/models/custom_test_model/card")
    assert read.status_code == 200, read.text
    body = read.json()
    assert body["editable"] is True
    assert body["repo_id"] == "org/test-model"

    patch = client.patch("/api/models/custom_test_model/card", json={
        "name": "Edited card",
        "steps": 8,
        "cfg": 2.5,
        "text_encoders": [{
            "repo_id": "org/test-model",
            "path": "text_encoder/model.safetensors",
            "size_gb": 1.5,
        }],
        "vae": {"repo_id": "org/test-model", "path": "vae/model.safetensors"},
    })
    assert patch.status_code == 200, patch.text
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["name"] == "Edited card"
    assert saved["engine_args"]["llm"] == "text_encoder/model.safetensors"
    assert saved["engine_args"]["vae"] == "vae/model.safetensors"


def test_card_api_protects_built_in_manifests(monkeypatch):
    import app as app_module
    from fastapi.testclient import TestClient

    client = TestClient(app_module.create_app())
    response = client.get("/api/models/flux_2_klein_4b_nunchaku/card")
    assert response.status_code == 404
    assert "защищены" in response.json()["detail"]


@pytest.mark.parametrize("unsafe_path", [
    "../escape.safetensors",
    "C:/absolute/model.safetensors",
    "/etc/passwd",
])
def test_component_download_rejects_unsafe_repository_paths(unsafe_path, monkeypatch):
    import app as app_module
    from fastapi.testclient import TestClient

    model = SimpleNamespace(
        id="custom_unsafe",
        sources=SimpleNamespace(primary=SimpleNamespace(repo_id="org/repo", type="huggingface")),
        engine_args={},
        get_text_encoder=lambda _id: {
            "source": {"repo_id": "org/repo", "file_path": unsafe_path}
        },
    )
    monkeypatch.setattr(app_module, "get_registry", lambda: SimpleNamespace(get=lambda _mid: model))
    client = TestClient(app_module.create_app())

    response = client.post(
        "/api/models/custom_unsafe/component-download",
        json={"role": "text_encoder", "id": "te_main"},
    )
    assert response.status_code == 400
    assert "safe repository/path binding" in response.json()["detail"]


def test_custom_model_card_is_restored_when_registry_reload_fails(tmp_path, monkeypatch):
    import backend.custom_models as cm

    manifest = _custom_manifest()
    path = tmp_path / "custom_test_model.json"
    original = json.dumps(manifest, indent=2)
    path.write_text(original, encoding="utf-8")

    monkeypatch.setattr(cm, "_card_registry_dir", lambda: tmp_path)

    def explode():
        raise RuntimeError("validation failed")

    monkeypatch.setattr(cm, "reload_registry", explode)
    with pytest.raises(ValueError, match="сохранить карточку"):
        cm.update_custom_model_card("custom_test_model", {"name": "Broken"})
    assert path.read_text(encoding="utf-8") == original


# ---------------------------------------------------------------------------
# Explicit LoRA binding
# ---------------------------------------------------------------------------

def _fake_model(model_id="m1", engine="sd_cpp", family="flux"):
    return SimpleNamespace(id=model_id, engine=engine, family=family)


def _patch_lora(tmp_path, monkeypatch, info):
    import backend.lora as lora

    (tmp_path / "test.safetensors").write_bytes(b"x")
    monkeypatch.setattr(lora, "get_loras_dir", lambda: tmp_path)
    monkeypatch.setattr(lora, "detect_lora_info", lambda _path: dict(info))


def test_exact_lora_binding_outranks_architecture_mismatch(tmp_path, monkeypatch):
    import backend.lora as lora

    _patch_lora(tmp_path, monkeypatch, {
        "family": "flux",
        "architecture": "sdxl",
        "compatible_models": ["m1"],
        "engine": [],
    })
    result = lora.check_lora_compatibility("test.safetensors", _fake_model())
    assert result["compatible"] is True
    assert result["match_level"] == "exact"


def test_lora_bound_to_other_models_is_rejected_despite_family_match(tmp_path, monkeypatch):
    import backend.lora as lora

    _patch_lora(tmp_path, monkeypatch, {
        "family": "flux",
        "architecture": "flux",
        "compatible_models": ["other_model"],
        "engine": [],
    })
    result = lora.check_lora_compatibility("test.safetensors", _fake_model())
    assert result["compatible"] is False
    assert "other_model" in result["reason"]


def test_lora_engine_allow_list_is_enforced(tmp_path, monkeypatch):
    import backend.lora as lora

    _patch_lora(tmp_path, monkeypatch, {
        "family": "flux",
        "architecture": "flux",
        "compatible_models": [],
        "engine": ["nunchaku"],
    })
    result = lora.check_lora_compatibility("test.safetensors", _fake_model(engine="sd_cpp"))
    assert result["compatible"] is False
    assert "nunchaku" in result["reason"]

    allowed = lora.check_lora_compatibility("test.safetensors", _fake_model(engine="nunchaku"))
    assert allowed["compatible"] is True


def test_lora_legacy_family_fallback_still_works(tmp_path, monkeypatch):
    import backend.lora as lora

    _patch_lora(tmp_path, monkeypatch, {
        "family": "qwen",
        "architecture": "qwen",
        "compatible_models": [],
        "engine": [],
    })
    result = lora.check_lora_compatibility("test.safetensors", _fake_model(family="flux"))
    assert result["compatible"] is False

    result = lora.check_lora_compatibility("test.safetensors", _fake_model(family="qwen"))
    assert result["compatible"] is True
    assert result["match_level"] == "architecture"


def test_suggest_model_bindings_matches_primary_and_base_repos(monkeypatch):
    import backend.lora as lora
    import backend.registry.loader as loader

    models = [
        SimpleNamespace(id="quantized", sources=SimpleNamespace(
            primary=SimpleNamespace(repo_id="someone/quantized-fibo")),
            base_repo_id="briaai/Fibo-2.0-edit-turbo"),
        SimpleNamespace(id="unrelated", sources=SimpleNamespace(
            primary=SimpleNamespace(repo_id="someone/other")),
            base_repo_id=""),
    ]
    monkeypatch.setattr(loader, "get_registry", lambda: SimpleNamespace(list_all=lambda: models))

    assert lora.suggest_model_bindings("https://huggingface.co/briaai/Fibo-2.0-edit-turbo") == ["quantized"]


def test_update_lora_metadata_normalizes_binding_fields(tmp_path, monkeypatch):
    import backend.lora as lora

    captured = {}
    (tmp_path / "test.safetensors").write_bytes(b"x")
    monkeypatch.setattr(lora, "get_loras_dir", lambda: tmp_path)
    monkeypatch.setattr(lora, "set_lora_metadata", lambda name, meta: captured.update(meta))
    monkeypatch.setattr(lora, "detect_lora_info", lambda _path: {"family": "flux"})

    result = lora.update_lora_metadata("test.safetensors", {
        "compatible_models": "m1, m2;m1",
        "engine": ["nunchaku", "nunchaku"],
        "target_modules": "q_proj, k_proj",
        "architecture": "flux",
        "scale_default": 5,
    })

    assert captured["compatible_models"] == ["m1", "m2"]
    assert captured["engine"] == ["nunchaku"]
    assert captured["target_modules"] == ["q_proj", "k_proj"]
    assert captured["scale_default"] == lora.MAX_STRENGTH
    assert result["name"] == "test.safetensors"


# ---------------------------------------------------------------------------
# Frontend wiring guards
# ---------------------------------------------------------------------------

def test_models_hub_card_editor_and_lora_binding_wiring():
    app_js = (ROOT / "frontend" / "js" / "app.js").read_text(encoding="utf-8")
    index_html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    styles_css = (ROOT / "frontend" / "css" / "styles.css").read_text(encoding="utf-8")

    assert "async loadLoras(notifyIncompatible = true)" in app_js
    assert "model_id=${encodeURIComponent(modelId)}" in app_js
    assert "pruneIncompatibleLoras" in app_js
    assert "loraCompatBadge" in app_js
    assert "openLoraBinding" in app_js
    assert "bindLoraToCurrentModel" in app_js
    assert "clearLoraBinding" in app_js
    assert "await this.loadLoras(true);" in app_js
    assert "downloadModelComponent" in app_js
    assert "/component-download" in app_js

    assert 'id="model-card-editor-modal"' in index_html
    assert 'id="component-picker-modal"' in index_html
    assert 'id="lora-binding-modal"' in index_html
    assert "App.openModelCardEditor(" in app_js
    assert "App.openComponentPicker(" in index_html

    for selector in (
        ".model-card-section",
        ".component-item",
        ".card-editor-grid",
        ".card-editor-check",
        ".lora-compat-badge",
        ".lora-incompat-reason",
        ".hf-result-meta",
    ):
        assert selector in styles_css
