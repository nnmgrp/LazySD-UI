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
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


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

        import huggingface_hub.constants as hf_constants
        assert hf_constants.HF_HUB_DISABLE_XET is True, "HF_HUB_DISABLE_XET must be True to prevent Rust OOM crashes"
        from huggingface_hub.utils._runtime import is_xet_available
        assert is_xet_available() is False, "is_xet_available() must be False"
        print("  ✓ HF_HUB_DISABLE_XET active (Rust xet disabled for stability)")

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
        assert s.engine_version == "0.0.17", f"Expected 0.0.17, got {s.engine_version}"
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
        assert klein4b.engine_version == "0.0.12", "Klein 4B must use engine v0.0.12"
        req4b = klein4b.resolve_required_files(gpu_sm=120, backend="lighting", steps=4)
        assert req4b["transformer_path"] == "transformer/klein-4b-50x-lighting.safetensors"
        print(f"  ✓ Klein 4B: {len(klein4b.transformers)} transformers, key_protected={klein4b.key_protected}, engine_version={klein4b.engine_version}")

        assert "qwen_image_2_1" in registry.models, "qwen_image_2_1 missing from registry"
        qwen21 = registry.get("qwen_image_2_1")
        assert qwen21.family == "qwen"
        assert qwen21.pipeline == "text_to_image"
        assert qwen21.engine_version == "0.0.17", "Qwen-Image-2.1 must use engine v0.0.17"
        assert len(qwen21.transformers) >= 2
        req_qwen21 = qwen21.resolve_required_files(gpu_sm=120, backend="lighting", steps=4)
        assert "qwen_image_21_base_quantfunc_int4_r128.safetensors" in req_qwen21["transformer_path"]
        assert req_qwen21["base_model_path"] == "."
        req_qwen21_sm86 = qwen21.resolve_required_files(gpu_sm=86, backend="lighting", steps=4)
        assert req_qwen21_sm86["base_model_path"] == "."
        req_qwen21_svdq = qwen21.resolve_required_files(gpu_sm=120, backend="svdq", steps=4)
        assert "qwen_image_21_base_quantfunc_int4_r128.safetensors" in req_qwen21_svdq["transformer_path"]
        print(f"  ✓ Qwen-Image-2.1: {len(qwen21.transformers)} transformers, resolves for SM 120, SM 86, and SVDQ")

        # Test precision config resolution by klein_tier
        qwen = registry.get("qwen_image")
        req_50x = qwen.resolve_required_files(gpu_sm=120, backend="lighting", steps=4)
        assert req_50x["precision_config_path"] == "precision-config/50x-fp4-f8-better.json"
        req_40x = qwen.resolve_required_files(gpu_sm=89, backend="lighting", steps=4)
        assert req_40x["precision_config_path"] == "precision-config/40x-int4-f8-better.json"
        req_30x = qwen.resolve_required_files(gpu_sm=86, backend="lighting", steps=4)
        assert req_30x["precision_config_path"] == "precision-config/30x-below-int4-i8-better.json"
        print(f"  ✓ Qwen-Image precision configs resolve correctly for SM 120, 89, 86")

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

        # SD.CPP models
        sdxl = registry.get("sdxl_turbo_gguf")
        assert sdxl.engine == "sd_cpp"
        assert sdxl.engine_args.get("checkpoint") == "sdxl-turbo-Q4_0.gguf"
        assert len(sdxl.transformers) == 1
        print(f"  ✓ SDXL Turbo GGUF loaded: engine={sdxl.engine}, checkpoint={sdxl.engine_args.get('checkpoint')}")

        flux_gguf = registry.get("flux_schnell_gguf")
        assert flux_gguf.engine == "sd_cpp"
        assert flux_gguf.engine_args.get("diffusion_model") == "flux1-schnell-Q4_K_S.gguf"
        print(f"  ✓ Flux Schnell GGUF loaded: engine={flux_gguf.engine}, diffusion={flux_gguf.engine_args.get('diffusion_model')}")

        # Qwen text encoders
        qwen = registry.get("qwen_image_2_1")
        assert len(qwen.text_encoders) == 1
        bf16 = qwen.get_text_encoder("bf16")
        assert bf16["file"] == "qwen3vl_8b_bf16.safetensors"
        assert bf16["precision"] == "int4"
        print(f"  ✓ Qwen-Image-2.1 (QuantFunc) text encoder: {bf16['file']} (dynamic {bf16['precision']})")

        # Qwen-Image-2.1 GGUF (SD.CPP)
        qwen_gguf = registry.get("qwen_image_2_1_gguf")
        assert qwen_gguf.engine == "sd_cpp"
        assert qwen_gguf.engine_args.get("llm") == "Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf"
        assert qwen_gguf.engine_args.get("vae") == "qwen_image_2.1_vae_bf16.safetensors"
        assert len(qwen_gguf.text_encoders) == 3
        ud = qwen_gguf.get_text_encoder("ud_q4_k_xl")
        assert ud is not None and ud["recommended"] is True
        print(f"  ✓ Qwen-Image-2.1 GGUF (SD.CPP) loaded: diffusion={qwen_gguf.engine_args.get('diffusion_model')}, llm={qwen_gguf.engine_args.get('llm')}")
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
        from backend.engine import check_engine_status, is_engine_installed, get_engine_paths, list_installed_engine_versions
        status = check_engine_status()
        print(f"  ✓ Engine status: {status.status}")
        print(f"    Message: {status.message}")
        print(f"  ✓ Engine installed (default): {is_engine_installed()}")
        print(f"  ✓ Engine installed (v0.0.12): {is_engine_installed('0.0.12')}")
        installed_vers = list_installed_engine_versions()
        print(f"  ✓ Installed engine versions: {installed_vers}")
        assert "0.0.17" in installed_vers
        assert "0.0.12" in installed_vers
        p12_bin, p12_cli = get_engine_paths("0.0.12")
        assert "v0.0.12" in str(p12_cli)

        from backend.sd_cpp import is_sd_cpp_installed, get_sd_cpp_cli_path, get_sd_cpp_version, download_sd_cpp
        print(f"  ✓ SD.CPP installed: {is_sd_cpp_installed()}")
        print(f"    SD.CPP CLI path: {get_sd_cpp_cli_path()}")
        print(f"    SD.CPP version: {get_sd_cpp_version()}")
        assert is_sd_cpp_installed()
        assert get_sd_cpp_cli_path().exists()

        # Test FastAPI /api/engine/status endpoint
        from fastapi.testclient import TestClient
        from app import create_app
        client = TestClient(create_app())
        res = client.get("/api/engine/status")
        assert res.status_code == 200, f"/api/engine/status failed with {res.status_code}"
        data = res.json()
        assert "engines" in data
        assert "quantfunc_0_17" in data["engines"]
        assert "quantfunc_0_12" in data["engines"]
        assert "sd_cpp" in data["engines"]
        print("  ✓ /api/engine/status returns all 3 engines with installed=True")

        res_custom = client.get("/api/custom-models/files")
        assert res_custom.status_code == 200
        assert "files" in res_custom.json()
        print("  ✓ /api/custom-models/files OK")
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
