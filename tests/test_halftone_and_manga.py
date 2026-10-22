"""
Comprehensive automated tests for Halftone Studio, Manga Screen AI + Manga Finish,
and LoRA compatibility/isolation in LazySD-UI.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from backend.halftone import (
    HALFTONE_PRESETS,
    HalftoneParams,
    image_to_data_url,
    load_image_safe,
    process_halftone_image,
)
from backend.manga_pipeline import (
    MangaPipelineParams,
    apply_manga_finish,
    apply_manga_levels,
    apply_manga_panel,
    extract_manga_lines,
    process_manga_algorithmic,
)
from backend.lora import (
    lora_prompt_tags,
    check_lora_compatibility,
    list_loras,
    resolve_lora_selection,
)
from backend.registry import get_registry


def _make_sample_rgb(w: int = 160, h: int = 120) -> Image.Image:
    """Create a rich synthetic test image with gradients, pure white, pure black, and sharp lines."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    # Horizontal gradient
    x_grad = np.linspace(0, 255, w, dtype=np.uint8)
    arr[:, :, 0] = x_grad[None, :]
    arr[:, :, 1] = np.flip(x_grad)[None, :]
    arr[:, :, 2] = 128
    # Pure white patch top-left
    arr[5:30, 5:35] = 255
    # Pure black patch bottom-right
    arr[h - 35 : h - 5, w - 40 : w - 5] = 0
    img = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(img)
    draw.line([(10, h - 15), (w - 10, 15)], fill=(0, 0, 0), width=3)
    draw.ellipse([(w // 2 - 20, h // 2 - 20), (w // 2 + 20, h // 2 + 20)], outline=(0, 0, 0), width=2)
    return img


def test_halftone_formats_and_dimensions():
    print("[1/8] Testing Halftone Studio image formats (RGB, RGBA, L, White, Black) & dimensions...")
    rgb = _make_sample_rgb(143, 97)
    rgba = rgb.convert("RGBA")
    alpha = np.full((97, 143), 255, dtype=np.uint8)
    alpha[10:40, 10:40] = 64
    rgba.putalpha(Image.fromarray(alpha, mode="L"))
    gray = rgb.convert("L")
    white = Image.new("RGB", (80, 60), (255, 255, 255))
    black = Image.new("RGB", (80, 60), (0, 0, 0))

    params = HalftoneParams.from_dict({"mode": "manga", "cell_size": 8, "antialias": 2})

    out_rgb = process_halftone_image(rgb, params)
    assert out_rgb.size == (143, 97), f"Expected (143, 97), got {out_rgb.size}"
    assert out_rgb.mode == "RGB"

    out_rgba = process_halftone_image(rgba, params)
    assert out_rgba.size == (143, 97)
    assert out_rgba.mode == "RGBA"
    out_alpha = np.asarray(out_rgba)[:, :, 3]
    assert np.array_equal(out_alpha, alpha), "RGBA alpha channel must be preserved intact"

    out_gray = process_halftone_image(gray, params)
    assert out_gray.size == (143, 97)

    out_white = np.asarray(process_halftone_image(white, params))
    assert out_white.mean() > 250, "Pure white image should remain bright paper"

    out_black = np.asarray(process_halftone_image(black, params))
    assert out_black.mean() < 30, "Pure black image should remain dark ink"
    print("  ✓ Formats & dimensions OK")


def test_halftone_modes_shapes_and_decoupled_radius():
    print("[2/8] Testing Halftone Studio modes, dot shapes, angle, and decoupled cell_size vs max_radius_ratio...")
    rgb = _make_sample_rgb(128, 128)

    # All 5 modes
    for mode in ("manga", "classic", "newspaper", "color", "custom"):
        p = HalftoneParams.from_dict({**HALFTONE_PRESETS[mode], "mode": mode})
        out = process_halftone_image(rgb, p)
        assert out.size == (128, 128)

    # All 4 dot shapes produce distinct patterns
    shape_outputs = {}
    for shape in ("circle", "square", "diamond", "line"):
        p = HalftoneParams.from_dict({
            "mode": "custom", "cell_size": 10, "dot_shape": shape,
            "max_radius_ratio": 0.85, "preserve_lines": False,
        })
        arr = np.asarray(process_halftone_image(rgb, p))
        shape_outputs[shape] = arr
    for s1 in shape_outputs:
        for s2 in shape_outputs:
            if s1 < s2:
                assert not np.array_equal(shape_outputs[s1], shape_outputs[s2]), f"Shapes {s1} and {s2} must differ"

    # Decoupled cell_size vs max_radius_ratio
    p_small_rad = HalftoneParams.from_dict({"mode": "custom", "cell_size": 10, "max_radius_ratio": 0.35, "preserve_lines": False})
    p_large_rad = HalftoneParams.from_dict({"mode": "custom", "cell_size": 10, "max_radius_ratio": 1.15, "preserve_lines": False})
    arr_small = np.asarray(process_halftone_image(rgb, p_small_rad))
    arr_large = np.asarray(process_halftone_image(rgb, p_large_rad))
    assert arr_small.mean() > arr_large.mean(), "Larger max_radius_ratio at fixed cell_size must produce darker coverage"

    # Angle changes pattern
    p_a15 = HalftoneParams.from_dict({"mode": "custom", "cell_size": 10, "angle": 15.0, "preserve_lines": False})
    p_a75 = HalftoneParams.from_dict({"mode": "custom", "cell_size": 10, "angle": 75.0, "preserve_lines": False})
    assert not np.array_equal(
        np.asarray(process_halftone_image(rgb, p_a15)),
        np.asarray(process_halftone_image(rgb, p_a75)),
    ), "Different screen angles must produce different dot grids"
    print("  ✓ Modes, shapes, angles, and decoupled radius OK")


def test_halftone_mask_and_determinism():
    print("[3/8] Testing Halftone Studio mask, invert_mask, determinism, and non-mutation...")
    rgb = _make_sample_rgb(120, 100)
    # Mask: left half white (apply), right half black (keep original)
    mask_arr = np.zeros((100, 120), dtype=np.uint8)
    mask_arr[:, :60] = 255
    mask_img = Image.fromarray(mask_arr, mode="L")

    p_normal = HalftoneParams.from_dict({"mode": "manga", "cell_size": 8, "invert_mask": False})
    out_masked = np.asarray(process_halftone_image(rgb, p_normal, mask_source=mask_img))
    orig_arr = np.asarray(rgb)
    # Right half should match original exactly
    assert np.allclose(out_masked[:, 65:], orig_arr[:, 65:], atol=1), "Unmasked region must preserve original image"
    assert not np.allclose(out_masked[:, :55], orig_arr[:, :55], atol=5), "Masked region must have halftone applied"

    # Invert mask: left half keeps original, right half gets halftone
    p_inv = HalftoneParams.from_dict({"mode": "manga", "cell_size": 8, "invert_mask": True})
    out_inv = np.asarray(process_halftone_image(rgb, p_inv, mask_source=mask_img))
    assert np.allclose(out_inv[:, :55], orig_arr[:, :55], atol=1), "Inverted mask must keep left half original"
    assert not np.allclose(out_inv[:, 65:], orig_arr[:, 65:], atol=5), "Inverted mask must apply halftone to right half"

    # Determinism with seeded noise dithering
    p_seed1 = HalftoneParams.from_dict({"mode": "newspaper", "dither_type": "noise", "seed": 12345})
    p_seed2 = HalftoneParams.from_dict({"mode": "newspaper", "dither_type": "noise", "seed": 99999})
    r1 = np.asarray(process_halftone_image(rgb, p_seed1))
    r2 = np.asarray(process_halftone_image(rgb, p_seed1))
    r3 = np.asarray(process_halftone_image(rgb, p_seed2))
    assert np.array_equal(r1, r2), "Same seed must produce bit-identical output"
    assert not np.array_equal(r1, r3), "Different seed must change noise dither output"

    # Non-mutation of source file on disk
    with tempfile.TemporaryDirectory() as tmpdir:
        src_file = Path(tmpdir) / "orig.png"
        rgb.save(src_file, format="PNG")
        sha_before = hashlib.sha256(src_file.read_bytes()).hexdigest()
        loaded = load_image_safe(src_file)
        _ = process_halftone_image(loaded, p_normal)
        sha_after = hashlib.sha256(src_file.read_bytes()).hexdigest()
        assert sha_before == sha_after, "Original file on disk must never be mutated"
    print("  ✓ Mask, invert_mask, determinism, and non-mutation OK")


def test_halftone_validation():
    print("[4/8] Testing parameter validation...")
    try:
        HalftoneParams.from_dict({"cell_size": 0})
        assert False, "Should reject cell_size=0"
    except ValueError:
        pass

    try:
        HalftoneParams.from_dict({"black_threshold": 0.85, "white_threshold": 0.20})
        assert False, "Should reject black_threshold >= white_threshold"
    except ValueError:
        pass

    try:
        HalftoneParams.from_dict({"dot_shape": "hexagon_invalid"})
        assert False, "Should reject unsupported dot_shape"
    except ValueError:
        pass

    # Verify dither_type='none', GenTonik shapes, satellites, and multi-layer stack
    from backend.halftone import process_halftone_layers
    rgb = _make_sample_rgb(100, 80)
    p_gt = HalftoneParams.from_dict({
        "mode": "manga",
        "dither_type": "none",
        "dot_shape": "hexagon",
        "satellite_enabled": True,
        "satellite_count": 4,
        "satellite_shape": "star",
        "satellite_size_ratio": 0.4,
        "satellite_distance": 0.38,
        "satellite_stretch": 1.4,
        "merge_factor": 0.25,
        "row_offset": 0.25,
        "jitter_pos": 0.1,
        "grad_type": "linear",
        "grad_strength": 0.2,
    })
    out_gt = process_halftone_image(rgb, p_gt)
    assert out_gt.size == (100, 80), "GenTonik satellite + hexagon halftone must preserve dimensions"

    layers_out = process_halftone_layers(rgb, [
        {"visible": True, "opacity": 1.0, "blend_mode": "normal", "params": {"mode": "manga", "cell_size": 8, "dither_type": "none"}},
        {"visible": True, "opacity": 0.7, "blend_mode": "multiply", "params": {"mode": "manga", "cell_size": 10, "angle": 15, "dot_shape": "star", "satellite_enabled": True}},
    ])
    assert layers_out.size == (100, 80), "Multi-layer GenTonik compositing must preserve dimensions"
    print("  ✓ Parameter validation & GenTonik (dither=none, satellites, layers) OK")


def test_manga_pipeline_stages():
    print("[5/8] Testing Manga Screen AI stages (Levels, Line Extraction, Manga Finish, Manga Panel)...")
    rgb = _make_sample_rgb(160, 120)

    # Stage B: Manga Levels
    mp_lite = MangaPipelineParams.from_dict({"mode": "manga_lite", "panel_enabled": False})
    leveled = apply_manga_levels(rgb, mp_lite)
    assert leveled.shape == (120, 160, 3)

    # Stage C: Line Extraction
    lines = extract_manga_lines(leveled, mp_lite)
    assert lines.shape == (120, 160)
    assert lines.dtype == np.float32
    assert 0.0 <= float(lines.min()) and float(lines.max()) <= 1.0

    # Stage D: Manga Finish
    finished = apply_manga_finish(leveled, lines, mp_lite)
    assert finished.shape == (120, 160, 3)

    # Full Manga Lite vs Finish Only
    out_lite = process_manga_algorithmic(rgb, mp_lite)
    assert out_lite.size == (160, 120)

    mp_finish_only = MangaPipelineParams.from_dict({"mode": "finish_only", "panel_enabled": False})
    out_finish = process_manga_algorithmic(rgb, mp_finish_only)
    assert out_finish.size == (160, 120)

    # Stage E: Manga Panel with aspect ratio & margin/border
    mp_panel = MangaPipelineParams.from_dict({
        "mode": "manga_lite",
        "panel_enabled": True,
        "panel_aspect": "1:1",
        "panel_border": 4,
        "panel_margin": 12,
        "panel_paper_tone": "cream",
        "panel_print_bleed": 0.2,
    })
    out_panel = process_manga_algorithmic(rgb, mp_panel)
    assert out_panel.width == out_panel.height, f"1:1 aspect panel with equal margins should be square, got {out_panel.size}"
    print("  ✓ Manga Screen AI algorithmic stages OK")


def test_lora_audit_and_compatibility():
    print("[6/8] Testing backend/lora.py compatibility, architecture detection, and prompt tag isolation...")
    registry = get_registry()
    models = {m.id: m for m in registry.list_all()}

    loras = list_loras()
    if loras and "flux2_klein_gguf" in models:
        flux_lora = next((l for l in loras if l.get("family") in ("flux", "klein")), None)
        if flux_lora:
            c_ok = check_lora_compatibility(flux_lora["name"], models["flux2_klein_gguf"])
            assert c_ok["compatible"] is True, f"Expected Flux LoRA to be compatible with flux2_klein_gguf: {c_ok}"

            if "qwen_image_2_1_gguf" in models:
                c_mismatch = check_lora_compatibility(flux_lora["name"], models["qwen_image_2_1_gguf"])
                assert c_mismatch["compatible"] is False, "Flux LoRA must be flagged incompatible with Qwen model"

            if "qwen_image_2_1_turbo_sdnq" in models:
                c_sdnq = check_lora_compatibility(flux_lora["name"], models["qwen_image_2_1_turbo_sdnq"])
                assert c_sdnq["compatible"] is False and c_sdnq["supported_by_engine"] is False, (
                    "SDNQ engine must explicitly report supported_by_engine=False for LoRA"
                )

    # Verify prompt tag injection and clean restoration when disabled
    base_prompt = "manga illustration of a samurai"
    paths = [Path("models/loras/[FLUX.2.Klein]LineArt_Redmond.safetensors")]
    tags = lora_prompt_tags(paths, [0.85])
    assert "<lora:[FLUX.2.Klein]LineArt_Redmond:0.85>" in tags
    empty_tags = lora_prompt_tags([], [])
    assert empty_tags == "", "Disabling LoRA must produce empty tag string and leave base prompt untouched"
    print("  ✓ LoRA compatibility & isolation OK")


def test_fastapi_endpoints():
    print("[7/8] Testing FastAPI endpoints (/api/halftone/*, /api/manga/*, /api/loras/check-compatibility, /api/history)...")
    from fastapi.testclient import TestClient
    from app import create_app, init_state
    from backend.settings import get_settings

    init_state()
    app = create_app()
    client = TestClient(app)

    # 1. Presets endpoint
    r_presets = client.get("/api/halftone/presets")
    assert r_presets.status_code == 200
    assert "manga" in r_presets.json()["presets"]

    # 2. Create a temporary source image in outputs/uploads
    settings = get_settings()
    uploads_dir = settings.output_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    test_src_path = uploads_dir / "ref_test_halftone_src.png"
    sample_img = _make_sample_rgb(180, 140)
    sample_img.save(test_src_path, format="PNG")
    src_sha_before = hashlib.sha256(test_src_path.read_bytes()).hexdigest()

    saved_files_to_cleanup = [test_src_path]
    try:
        # 3. Preview Halftone (save=False)
        r_prev = client.post("/api/halftone/process", json={
            "image_path": "uploads/ref_test_halftone_src.png",
            "params": {"mode": "manga", "cell_size": 8, "dot_shape": "circle"},
            "save": False,
        })
        assert r_prev.status_code == 200, r_prev.text
        data_prev = r_prev.json()
        assert data_prev["status"] == "ok"
        assert data_prev["preview_data_url"].startswith("data:image/png;base64,")
        assert data_prev["saved"] is None

        # 4. Save Halftone to Gallery (save=True)
        r_save = client.post("/api/halftone/process", json={
            "image_path": "uploads/ref_test_halftone_src.png",
            "params": {"mode": "newspaper", "cell_size": 10, "dot_shape": "diamond"},
            "save": True,
        })
        assert r_save.status_code == 200, r_save.text
        data_save = r_save.json()
        assert data_save["saved"] is not None
        saved_ht_path = Path(data_save["saved"]["path"])
        saved_files_to_cleanup.append(saved_ht_path)
        saved_files_to_cleanup.append(saved_ht_path.with_suffix(".json"))
        assert saved_ht_path.exists()
        assert saved_ht_path.with_suffix(".json").exists()

        # Verify source image on disk was NOT overwritten or mutated
        src_sha_after = hashlib.sha256(test_src_path.read_bytes()).hexdigest()
        assert src_sha_before == src_sha_after, "Source image in outputs/uploads must remain untouched"

        # 5. Save Manga Lite to Gallery (save=True)
        r_manga = client.post("/api/manga/process", json={
            "image_path": "uploads/ref_test_halftone_src.png",
            "params": {
                "mode": "manga_lite",
                "finish_preset": "shonen",
                "finish_dot_size": 6,
                "panel_enabled": True,
                "panel_aspect": "3:4",
            },
            "save": True,
        })
        assert r_manga.status_code == 200, r_manga.text
        data_manga = r_manga.json()
        assert data_manga["saved"] is not None
        saved_manga_path = Path(data_manga["saved"]["path"])
        saved_files_to_cleanup.append(saved_manga_path)
        saved_files_to_cleanup.append(saved_manga_path.with_suffix(".json"))
        assert saved_manga_path.exists()

        # 6. Verify /api/history returns tool & tool_params for both saved items
        r_hist = client.get("/api/history")
        assert r_hist.status_code == 200
        hist_items = {it["name"]: it for it in r_hist.json()}
        assert saved_ht_path.name in hist_items
        assert hist_items[saved_ht_path.name]["tool"] == "halftone_studio"
        assert hist_items[saved_ht_path.name]["tool_params"]["dot_shape"] == "diamond"
        assert saved_manga_path.name in hist_items
        assert hist_items[saved_manga_path.name]["tool"] == "manga_screen_ai"
        assert hist_items[saved_manga_path.name]["tool_params"]["mode"] == "manga_lite"

        # 7. Verify security / invalid input rejection (HTTP 400)
        r_trav = client.post("/api/halftone/process", json={
            "image_path": "../../app.py",
            "params": {"mode": "manga"},
        })
        assert r_trav.status_code == 400, "Path traversal must be rejected with HTTP 400"

        r_bad_param = client.post("/api/halftone/process", json={
            "image_path": "uploads/ref_test_halftone_src.png",
            "params": {"cell_size": -5},
        })
        assert r_bad_param.status_code == 400, "Invalid cell_size must be rejected with HTTP 400"

    finally:
        for p in saved_files_to_cleanup:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

    print("  ✓ FastAPI endpoints & history integration OK")


if __name__ == "__main__":
    test_halftone_formats_and_dimensions()
    test_halftone_modes_shapes_and_decoupled_radius()
    test_halftone_mask_and_determinism()
    test_halftone_validation()
    test_manga_pipeline_stages()
    test_lora_audit_and_compatibility()
    test_fastapi_endpoints()
    print("\n[8/8] ALL HALFTONE STUDIO & MANGA SCREEN AI TESTS PASSED! ✓")
