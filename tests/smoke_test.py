"""
Smoke test — validates that all backend modules import cleanly, the model
registry JSONs are valid, and the engine auth config (config.json) works.

Run:
    cd QuantFunc-WebUI
    python tests/smoke_test.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


def test_imports():
    print("\n[1/6] Testing backend imports...")
    try:
        from backend.settings import get_settings, Settings
        from backend.gpu import detect_gpu, GPUInfo, tier_for_sm, klein_tier_for_sm
        from backend.security import validate_path, validate_prompt, validate_filename, safe_subprocess_args
        from backend.cache import CacheManager, FileState, compute_sha256
        from backend.engine import is_engine_installed, verify_engine, check_engine_status
        from backend.registry import get_registry, ModelSpec, TransformerVariant
        from backend.worker import Worker, GenerateRequest, JobStatus, ProgressEvent
        from backend.sources import ModelSource, HuggingFaceSource, ModelScopeSource
        print("  ✓ All backend modules imported successfully")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_settings():
    print("\n[2/7] Testing settings...")
    try:
        from backend.settings import reload_settings
        s = reload_settings()
        assert s.engine_version == "0.0.12", f"Expected 0.0.12, got {s.engine_version}"
        assert s.lang in ("ru", "en", "auto")
        assert s.vram_mode in ("auto", "conservative", "balanced", "maximum")
        assert hasattr(s, "quantfunc_api_key"), "quantfunc_api_key field missing"

        # Clearing behavior: empty string must be stored ("clear the key")
        s.update(quantfunc_api_key="")
        assert s.quantfunc_api_key == "", "empty string should clear the key"
        s.update(quantfunc_api_key="qf_test_value")
        assert s.quantfunc_api_key == "qf_test_value"
        # None must NOT overwrite an existing non-clearable value (vram_mode is not clearable)
        current_mode = s.vram_mode
        s.update(vram_mode=None)
        assert s.vram_mode == current_mode, "None should be ignored for non-clearable fields"
        # Restore
        s.update(quantfunc_api_key="")
        print(f"  ✓ Settings OK: lang={s.lang}, vram_mode={s.vram_mode}, engine=v{s.engine_version}")
        print(f"  ✓ quantfunc_api_key field + clear-via-empty-string OK")
        print(f"  ✓ Models dir: {s.models_dir}")
        print(f"  ✓ Output dir: {s.output_dir}")
        print(f"  ✓ Engine binary: {s.engine_binary_path}")
        print(f"  ✓ Engine CLI: {s.engine_cli_path}")
        return True
    except Exception as e:
        print(f"  ✗ Settings test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu():
    print("\n[3/7] Testing GPU detection...")
    try:
        from backend.gpu import detect_gpu, tier_for_sm, klein_tier_for_sm
        info = detect_gpu()
        if info:
            print(f"  ✓ GPU detected: {info.name}")
            print(f"    SM: {info.sm} → tier={info.tier}, klein_tier={info.klein_tier}")
            print(f"    VRAM: {info.vram_mb} MB ({info.vram_gb:.1f} GB)")
            print(f"    Driver: {info.driver_version}")
        else:
            print("  ⚠ No NVIDIA GPU detected (this is OK on dev machines without GPU)")

        assert tier_for_sm(120) == "50x-above"
        assert tier_for_sm(89) == "50x-below"
        assert klein_tier_for_sm(75) == "30x-below"
        assert klein_tier_for_sm(89) == "40x"
        assert klein_tier_for_sm(120) == "50x"
        print("  ✓ Tier helper functions OK")
        return True
    except Exception as e:
        print(f"  ✗ GPU test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_registry():
    print("\n[4/7] Testing model registry...")
    try:
        from backend.registry import reload_registry
        registry = reload_registry()

        assert "flux_klein_9b" in registry.models
        model = registry.get("flux_klein_9b")

        assert model.name == "Flux.2 Klein 9B"
        assert model.family == "klein"
        assert model.pipeline == "text_to_image"
        assert len(model.transformers) >= 5
        assert len(model.base_models) == 2
        assert model.key_protected is True, "Klein 9B must be key_protected"
        print(f"  ✓ Loaded: {model.name} ({model.id}) — key_protected={model.key_protected}")
        print(f"    Family: {model.family} | Pipeline: {model.pipeline}")
        print(f"    Transformers: {len(model.transformers)}")
        print(f"    Min VRAM: {model.requirements.vram_min_gb} GB")

        # New models in registry (Phase 2.5)
        assert "z_image_turbo" in registry.models, "z_image_turbo missing from registry"
        assert "flux_klein_4b" in registry.models, "flux_klein_4b missing from registry"

        z_image = registry.get("z_image_turbo")
        assert z_image.family == "z-image"
        assert z_image.key_protected is False, "Z-Image should not be key-protected"
        assert len(z_image.transformers) >= 3
        req = z_image.resolve_required_files(gpu_sm=120, backend="lighting", steps=4)
        assert req["base_model_path"] == "z-image-series-50x-above-base-model/"
        assert "z-image-turbo-50x-above" in req["transformer_path"]
        print(f"  ✓ Z-Image Turbo: {len(z_image.transformers)} transformers, resolves for SM 120")
        print(f"    Base: {req['base_model_path']}")
        print(f"    Transformer: {req['transformer_path']}")

        klein4b = registry.get("flux_klein_4b")
        assert klein4b.key_protected is True, "Klein 4B must be key_protected"
        req4b = klein4b.resolve_required_files(gpu_sm=120, backend="lighting", steps=4)
        assert req4b["transformer_path"] == "transformer/klein-4b-50x-lighting.safetensors"
        print(f"  ✓ Klein 4B: {len(klein4b.transformers)} transformers, key_protected={klein4b.key_protected}")

        # RTX 5060 Ti (SM 120) — 50x-above tier
        tier_above = model.get_transformers_for_tier(gpu_sm=120)
        print(f"  ✓ For RTX 5060 Ti (SM 120): {len(tier_above)} compatible transformers (Klein 9B)")
        for t in tier_above:
            print(f"    - {t.id} ({t.backend}, {t.steps} steps, {t.size_gb:.1f} GB)")

        # RTX 3070 (SM 86) — 50x-below tier
        tier_below = model.get_transformers_for_tier(gpu_sm=86)
        print(f"  ✓ For RTX 3070 (SM 86): {len(tier_below)} compatible transformers (Klein 9B)")
        for t in tier_below:
            print(f"    - {t.id} ({t.backend}, {t.steps} steps, {t.size_gb:.1f} GB)")

        # Resolve required files for RTX 5060 Ti + Lighting-4steps
        required = model.resolve_required_files(
            gpu_sm=120, backend="lighting", steps=4, use_prequant=False,
        )
        assert required["base_model_path"] == "klein-9b-series-50x-above-base-model/"
        print(f"  ✓ Resolve required files (RTX 5060 Ti + Lighting-4steps):")
        print(f"    Base: {required['base_model_path']}")
        print(f"    Transformer: {required['transformer_path']}")
        return True
    except Exception as e:
        print(f"  ✗ Registry test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_security():
    print("\n[5/7] Testing security helpers...")
    try:
        from backend.security import (
            validate_path, validate_prompt, validate_filename,
            safe_subprocess_args, SecurityError,
        )
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)

            # Valid path
            p = validate_path("subdir/file.txt", base_dir=base_dir)
            assert str(p).startswith(tmp)

            # Path traversal attempt
            try:
                validate_path("../../../etc/passwd", base_dir=base_dir)
                assert False, "Should have raised"
            except SecurityError:
                pass

            # Null byte injection
            try:
                validate_path("file.txt\x00.exe", base_dir=base_dir)
                assert False
            except SecurityError:
                pass

            # Prompt
            assert validate_prompt("hello world") == "hello world"
            assert validate_prompt("hello\x00world") == "helloworld"

            # Filename
            assert validate_filename("image.png") == "image.png"
            try:
                validate_filename("../etc/passwd")
                assert False
            except SecurityError:
                pass

            # Subprocess args
            args = safe_subprocess_args(["ls", "-la", "/tmp"])
            assert len(args) == 3
            try:
                safe_subprocess_args(["ls", "file\x00name"])
                assert False
            except SecurityError:
                pass

        print("  ✓ Path traversal protection OK")
        print("  ✓ Null byte injection blocked")
        print("  ✓ Prompt sanitization OK")
        print("  ✓ Filename validation OK")
        print("  ✓ Subprocess args validation OK")
        return True
    except Exception as e:
        print(f"  ✗ Security test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_engine_auth():
    print("\n[6/7] Testing engine auth config (config.json)...")
    try:
        from backend.engine import (
            ensure_config_json, engine_config_json_path,
            read_engine_api_key, has_api_key_available,
            BUNDLED_TEST_CONFIG,
        )
        import backend.engine as engine_mod

        # Isolated check: does ensure_config_json produce a valid config?
        # (uses the real settings.engine_dir — acceptable for a smoke test;
        #  an existing config.json is never overwritten without force=True)
        cfg_path, source = ensure_config_json()
        assert cfg_path.exists(), f"config.json not created at {cfg_path}"
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        assert "api_key" in data and data["api_key"].startswith("qf_"), "api_key missing/invalid in config.json"
        assert "server_url" in data, "server_url missing in config.json"
        print(f"  ✓ config.json present (source: {source}): {cfg_path}")
        print(f"  ✓ api_key format OK: {data['api_key'][:12]}...")

        key = read_engine_api_key()
        assert key and key.startswith("qf_"), "read_engine_api_key failed"
        print(f"  ✓ read_engine_api_key OK")
        print(f"  ✓ has_api_key_available: {has_api_key_available()}")

        # Offline fallback path — write bundled config into a temp dir
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            fake_cfg = engine_mod.Path(tmp) / "config.json"
            fake_cfg.write_text(json.dumps(BUNDLED_TEST_CONFIG), encoding="utf-8")
            loaded = json.loads(fake_cfg.read_text(encoding="utf-8"))
            assert loaded["api_key"] == BUNDLED_TEST_CONFIG["api_key"]
        print(f"  ✓ Bundled fallback config valid")

        # Worker key-protection regex
        from backend.worker import _KEY_PROTECTED_RE
        assert _KEY_PROTECTED_RE.search("Flux2Klein: checkObfuscation failed: This model is key-protected.")
        assert _KEY_PROTECTED_RE.search("Please provide --apikey <key>")
        assert not _KEY_PROTECTED_RE.search("Step 2/4 completed")
        print(f"  ✓ Key-protection error detection regex OK")
        return True
    except Exception as e:
        print(f"  ✗ Engine auth test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_engine_status():
    print("\n[7/7] Testing engine status check...")
    try:
        from backend.engine import check_engine_status, is_engine_installed
        status = check_engine_status()
        print(f"  ✓ Engine status: {status.status}")
        print(f"    Message: {status.message}")
        print(f"  ✓ Engine installed: {is_engine_installed()}")
        return True
    except Exception as e:
        print(f"  ✗ Engine status test failed: {e}")
        return False


def main():
    print("=" * 60)
    print("QuantFunc WebUI — Smoke Test")
    print("=" * 60)

    results = [
        ("Imports", test_imports()),
        ("Settings", test_settings()),
        ("GPU", test_gpu()),
        ("Registry", test_registry()),
        ("Security", test_security()),
        ("Engine auth (config.json)", test_engine_auth()),
        ("Engine status", test_engine_status()),
    ]

    print("\n" + "=" * 60)
    print("RESULTS:")
    passed = 0
    for name, ok in results:
        marker = "✓" if ok else "✗"
        print(f"  {marker} {name}")
        if ok:
            passed += 1
    print(f"\n{passed}/{len(results)} tests passed")
    print("=" * 60)

    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
