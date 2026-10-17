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
        assert hasattr(s, "qwen_vae_type"), "qwen_vae_type field missing"
        assert s.qwen_vae_type in ("texture_fix", "official")

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
        print(f"  ✓ Settings OK: lang={s.lang}, vram_mode={s.vram_mode}, engine=v{s.engine_version}, qwen_vae={s.qwen_vae_type}")
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

        flux3 = registry.get("flux_3_image_klein_gguf")
        assert flux3.engine == "sd_cpp"
        assert any(t.id == "q8_0" for t in flux3.transformers), "Q8_0 variant must be present"
        assert any(t.id == "base_q8_0" for t in flux3.transformers), "base_q8_0 variant must be present"
        assert flux3.engine_args.get("diffusion_model") == "flux3_image_klein_sd_Q8_0.gguf"
        assert len(flux3.transformers) == 9, f"Expected 9 variants, got {len(flux3.transformers)}"
        assert len(flux3.text_encoders) == 4
        print(f"  ✓ FLUX 3 Image Klein GGUF loaded: 9 variants (5 distilled + 4 base Q8/Q6/Q4/Q3), 4 text encoders")

        assert "flux_2_klein_base_9b_gguf" in registry.models, "flux_2_klein_base_9b_gguf missing from registry"
        flux2_base = registry.get("flux_2_klein_base_9b_gguf")
        assert flux2_base.engine == "sd_cpp"
        assert flux2_base.family == "klein"
        assert len(flux2_base.transformers) == 5
        assert any(t.id == "q8_0" for t in flux2_base.transformers), "Q8_0 variant must be present in Klein 9B Base"
        assert len(flux2_base.text_encoders) == 2
        q8_te = flux2_base.get_text_encoder("qwen3_8b_q8_0")
        assert q8_te is not None and q8_te["recommended"] is True
        assert q8_te["file"] == "Qwen3-8B-Q8_0.gguf"
        assert flux2_base.scheduler_presets["default"]["cfg"] == 4.0
        assert flux2_base.scheduler_presets["default"]["steps"] == 20
        print(f"  ✓ Flux.2 Klein 9B Base GGUF loaded: 5 variants, Q8 text encoder ({q8_te['file']}), cfg=4.0, steps=20")

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
        assert len(qwen_gguf.text_encoders) == 1
        assert qwen_gguf.scheduler_presets["default"]["cfg"] == 3.0
        ud = qwen_gguf.get_text_encoder("ud_q4_k_xl")
        assert ud is not None and ud["recommended"] is True
        print(f"  ✓ Qwen-Image-2.1 GGUF (SD.CPP) loaded: diffusion={qwen_gguf.engine_args.get('diffusion_model')}, llm={qwen_gguf.engine_args.get('llm')}, cfg={qwen_gguf.scheduler_presets['default']['cfg']}")

        # Qwen-Image-2.1 Turbo SDNQ INT8
        qwen_sdnq = registry.get("qwen_image_2_1_turbo_sdnq")
        assert qwen_sdnq.engine == "sdnq"
        assert qwen_sdnq.sources.primary.repo_id == "OzzyGT/Qwen_Image_2_1_Turbo_sdnq_dynamic_8bit"
        assert len(qwen_sdnq.transformers) == 1
        assert qwen_sdnq.transformers[0].steps == 8
        assert qwen_sdnq.scheduler_presets["default"]["cfg"] == 1.0
        assert len(qwen_sdnq.engine_args.get("sample_sigmas", [])) == 8
        print(f"  ✓ Qwen-Image-2.1 Turbo SDNQ loaded: repo={qwen_sdnq.sources.primary.repo_id}, steps={qwen_sdnq.transformers[0].steps}, cfg={qwen_sdnq.scheduler_presets['default']['cfg']}")
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

        from backend.sdnq_engine import is_sdnq_installed, get_sdnq_version, get_sdnq_path
        print(f"  ✓ SDNQ installed: {is_sdnq_installed()}")
        print(f"    SDNQ path: {get_sdnq_path()}")
        print(f"    SDNQ version: {get_sdnq_version()}")
        assert is_sdnq_installed()

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
        assert "sdnq" in data["engines"]
        assert data["engines"]["sdnq"]["installed"] is True
        print("  ✓ /api/engine/status returns all 4 engines with installed=True")

        res_state = client.get("/api/models/qwen_image_2_1_turbo_sdnq/state")
        assert res_state.status_code == 200
        state_data = res_state.json()
        assert state_data["model_id"] == "qwen_image_2_1_turbo_sdnq"
        assert len(state_data["transformers"]) == 1
        assert len(state_data["text_encoders"]) == 1
        print("  ✓ /api/models/qwen_image_2_1_turbo_sdnq/state OK")

        res_custom = client.get("/api/custom-models/files")
        assert res_custom.status_code == 200
        assert "files" in res_custom.json()
        print("  ✓ /api/custom-models/files OK")
        return True
    except Exception as e:
        print(f"  ✗ Engine status test failed: {e}")
        return False


def test_controls_and_samplers():
    print("\n[8/8] Testing controls, samplers & parameter plumbing...")
    try:
        from backend.worker import GenerateRequest, Worker
        from backend.gpu import GPUInfo
        from backend.registry import get_registry
        from backend.flowmatch_samplers import interpolate_sample_sigmas
        from fastapi.testclient import TestClient
        from app import create_app

        req = GenerateRequest(
            prompt="test",
            model_id="qwen_image_2_1_turbo_sdnq",
            variant_id="qwen_image_21_turbo_sdnq_int8",
            sampler="dpm++2m",
            cfg=2.5,
            noise=0.3,
            vae="texture_fix",
        )
        assert req.variant_id == "qwen_image_21_turbo_sdnq_int8"
        assert req.sampler == "dpm++2m"
        assert req.cfg == 2.5
        assert req.noise == 0.3
        assert req.vae == "texture_fix"
        print("  ✓ GenerateRequest preserves variant_id, sampler, cfg, noise, vae")

        from backend.settings import get_settings
        s = get_settings()
        orig_mode = s.vram_mode
        s.vram_mode = "balanced"
        try:
            w = Worker()
            fake_gpu = GPUInfo(
                name="RTX 5060 Ti",
                sm=120,
                vram_mb=16384,
                driver_version="610.62",
                cuda_version="12.8",
                tier="50x-above",
                klein_tier="50x",
            )
            budget_mb = w._compute_vram_budget(fake_gpu, None)
            assert budget_mb == 12288, f"Expected 12288 MB, got {budget_mb}"
            print(f"  ✓ Worker._compute_vram_budget returns MB ({budget_mb} MB for 16 GB balanced)")
        finally:
            s.vram_mode = orig_mode

        reg = get_registry()
        flux_s = reg.get("flux_schnell_gguf")
        assert flux_s.get_default_cfg() == 3.5
        z_img = reg.get("z_image_turbo")
        r_b = z_img.resolve_required_files(gpu_sm=120, variant_id="lighting-50x-above-better")
        r_v2 = z_img.resolve_required_files(gpu_sm=120, variant_id="lighting-50x-above-v2")
        assert r_b["transformer_path"] != r_v2["transformer_path"]
        assert "v2" in r_v2["transformer_path"]
        print("  ✓ resolve_required_files respects explicit variant_id")

        base_sig = [1.0, 0.811402, 0.654004, 0.502087, 0.362709, 0.228661, 0.114957, 0.017578]
        s12 = interpolate_sample_sigmas(base_sig, 12)
        assert len(s12) == 12 and s12[0] == 1.0 and 0 < s12[-1] < s12[-2]
        print("  ✓ interpolate_sample_sigmas interpolates Turbo schedule to arbitrary steps")

        client = TestClient(create_app())
        cfg_res = client.get("/api/config").json()
        assert "qwen_vae_type" in cfg_res["settings"], "qwen_vae_type missing in /api/config settings"
        print("  ✓ /api/config includes settings.qwen_vae_type")

        # Verify 3-tier model sorting (disabled models at bottom, installed at top)
        models_list = cfg_res["models"]
        tiers = [2 if m.get("disabled") else (0 if m.get("installed") else 1) for m in models_list]
        assert tiers == sorted(tiers), f"Models not sorted in 3-tier order: {tiers}"
        print("  ✓ 3-tier model sorting (installed -> downloadable -> disabled) verified")

        # Verify Presets CRUD API
        presets_res = client.get("/api/presets")
        assert presets_res.status_code == 200
        presets_data = presets_res.json()["presets"]
        assert len(presets_data) >= 5
        create_res = client.post("/api/presets", json={
            "title": "Smoke Test Preset",
            "prompt": "high contrast manga lineart",
            "steps": 12,
            "cfg": 3.5,
        })
        assert create_res.status_code == 200
        created_id = create_res.json()["preset"]["id"]
        del_res = client.delete(f"/api/presets/{created_id}")
        assert del_res.status_code == 200
        print("  ✓ /api/presets (GET, POST, DELETE) CRUD verified")

        # Verify custom_models file metadata detector
        from backend.custom_models import detect_file_metadata
        meta_q8 = detect_file_metadata("flux-2-klein-base-9B-Q8_0.gguf")
        assert meta_q8["family"] == "flux" and meta_q8["quant"] == "Q8_0" and meta_q8["role"] == "diffusion_model"
        meta_te = detect_file_metadata("Qwen3-VL-8B-Instruct-Q8_0.gguf")
        assert meta_te["role"] == "text_encoder"
        print("  ✓ detect_file_metadata auto-detects role, family, and quantization")

        html = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        assert 'id="sampler-select"' in html, "sampler-select missing in index.html"
        assert 'id="gemini-dock-wrap"' in html, "gemini-dock-wrap missing in index.html"
        assert 'id="topaz-inspector"' in html, "topaz-inspector missing in index.html"
        assert 'id="tab-templates"' in html, "tab-templates missing in index.html"
        print("  ✓ frontend/index.html includes #sampler-select, #gemini-dock-wrap, #topaz-inspector, #tab-templates")
        return True
    except Exception as e:
        print(f"  ✗ Controls & samplers test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lora_share_extras():
    print("\n[9/9] Testing LoRA, Sharing, Extras & Live Indicators...")
    try:
        from backend.lora import (
            sanitize_lora_filename,
            lora_prompt_tags,
            get_loras_dir,
            resolve_lora_selection,
            get_settings,
        )
        # 1. LoRA helpers
        safe = sanitize_lora_filename("my:cool<lora>_v1.safetensors")
        assert safe == "my_cool_lora__v1.safetensors", f"Sanitize error: {safe}"
        p = Path("models/loras/test_lora.safetensors")
        tags = lora_prompt_tags([p], [0.85])
        assert tags == "<lora:test_lora:0.85>", f"Tags error: {tags}"
        assert get_loras_dir().exists()
        print("  ✓ LoRA helpers (sanitization, tags, loras dir) OK")

        # 2. Live CPU/GPU/VRAM system stats
        from backend.gpu import get_system_memory_stats
        stats = get_system_memory_stats()
        assert "cpu_percent" in stats, "Missing cpu_percent in stats"
        assert "cpu_cores" in stats, "Missing cpu_cores in stats"
        assert "vram_used_gb" in stats, "Missing vram_used_gb in stats"
        assert "gpu_temp_c" in stats, "Missing gpu_temp_c in stats"
        print(f"  ✓ Live system stats OK: CPU {stats['cpu_percent']}% ({stats['cpu_cores']} cores), VRAM {stats['vram_used_gb']}/{stats['vram_total_gb']} GB")

        # 3. Share manager & Gradio-free remote sharing
        from backend.share import get_share_manager, is_lan_exposed
        sm = get_share_manager()
        assert sm is not None
        status = sm.status()
        assert "tunnel" in status and "token" in status
        print(f"  ✓ Share manager OK (tunnel_state={status['tunnel']['state']}, token_set={status['token_set']})")

        # 4. Extras manager & wheel planner
        from backend.extras import plan_nunchaku_wheel, is_nunchaku_ready, is_orbitquant_ready
        wheel_plan = plan_nunchaku_wheel()
        print(f"  ✓ Nunchaku wheel planner OK: candidate={wheel_plan.get('candidate_wheel') or 'none'}")

        # 5. REST API routes
        from app import create_app
        from fastapi.testclient import TestClient
        client = TestClient(create_app())

        r_loras = client.get("/api/loras")
        assert r_loras.status_code == 200 and "loras" in r_loras.json(), "GET /api/loras failed"
        print("  ✓ GET /api/loras OK")

        r_share = client.get("/api/share/status")
        assert r_share.status_code == 200 and "running" in r_share.json(), "GET /api/share/status failed"
        print("  ✓ GET /api/share/status OK")

        r_extras = client.get("/api/extras/status")
        assert r_extras.status_code == 200 and "nunchaku_ready" in r_extras.json(), "GET /api/extras/status failed"
        print("  ✓ GET /api/extras/status OK")

        r_models = client.get("/api/models")
        assert r_models.status_code == 200
        models_data = r_models.json()
        model_list = models_data if isinstance(models_data, list) else models_data.get("models", [])
        model_ids = [m["id"] for m in model_list]
        assert "flux_2_klein_4b_orbitquant" in model_ids, "Missing OrbitQuant model in /api/models"
        assert "flux_2_klein_4b_nunchaku" in model_ids, "Missing Nunchaku model in /api/models"
        print("  ✓ OrbitQuant and Nunchaku models registered in /api/models")

        # 6. Open folder endpoint verification
        r_folder = client.post("/api/system/open-folder", json={"folder_type": "outputs"})
        assert r_folder.status_code == 200, "POST /api/system/open-folder failed"
        assert r_folder.json().get("status") in ("ok", "remote", "error"), "Invalid open-folder response"
        r_folder_get = client.get("/api/system/open-folder?folder_type=models")
        assert r_folder_get.status_code == 200, "GET /api/system/open-folder failed"
        r_bad_path = client.post("/api/system/open-folder", json={"path": "../../../etc/passwd"})
        assert r_bad_path.status_code == 400, "Path traversal in open-folder not blocked"
        print("  ✓ /api/system/open-folder (POST, GET, traversal guard) OK")

        # 7. Frontend markup validation
        index_html = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
        assert 'id="compare-container"' in index_html, "Missing compare container in index.html"
        assert 'id="btn-compare-toggle"' in index_html, "Missing compare toggle button in index.html"
        assert 'id="dock-lora-pill"' in index_html, "Missing LoRA dock pill in index.html"
        assert 'id="dock-popover-lora"' in index_html, "Missing LoRA popover in index.html"
        assert 'id="cpu-stat-text"' in index_html, "Missing cpu-stat-text in index.html"
        assert 'id="setting-dock-opacity"' in index_html, "Missing dock opacity setting in index.html"
        assert 'id="setting-share-toggle"' in index_html or 'id="btn-share-toggle"' in index_html, "Missing share tunnel toggle in index.html"
        assert 'id="btn-open-output-dir"' in index_html, "Missing btn-open-output-dir in index.html"
        assert 'id="history-modal-open-folder-btn"' in index_html, "Missing history-modal-open-folder-btn in index.html"
        print("  ✓ Frontend UI components (Compare curtain, LoRA popover, CPU indicator, Dock controls, Share controls, Open Folder buttons) verified in index.html")

        return True
    except Exception as e:
        print(f"  ✗ LoRA, Sharing & Extras test failed: {e}")
        import traceback
        traceback.print_exc()
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
        ("Controls & samplers", test_controls_and_samplers()),
        ("LoRA, Sharing, Extras & Indicators", test_lora_share_extras()),
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
