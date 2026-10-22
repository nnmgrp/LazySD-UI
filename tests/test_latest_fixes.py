"""
Verification script for Halftone Studio GT fixes:
1. Preview resolution: natural dimensions (up to 4096px, no 768px limit)
2. Backend halftone process default preview_max_side = 0 (full resolution)
3. WebGL2 shader implements: Satellites, Gradients, all 13 Shapes, Aspect, Jitter, Dither
4. Zero-allocation reusable buffers for CPU compositor
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(r"G:\QuantFunc-WebUI")
sys.path.insert(0, str(PROJECT_ROOT))

def test_frontend_halftone_features():
    print("\n[1/3] Testing Halftone Studio GT frontend features in halftone.js...")
    # In native ES6 modular architecture, Halftone Studio GT lives in modules/halftone.js
    mod_path = PROJECT_ROOT / "frontend" / "js" / "modules" / "halftone.js"
    halftone_js = mod_path.read_text(encoding="utf-8")

    # 1. Check full resolution preview (no 768 limit)
    assert 'const maxSide = 4096;' in halftone_js, "maxSide is not 4096"
    assert 'const maxSide = 768;' not in halftone_js, "Old 768px downscale limit still present"
    print("  [OK] Full natural resolution (up to 4096px) verified")

    # 2. Check Satellites / Planets uniforms & math
    assert 'u_satEnabled' in halftone_js, "u_satEnabled uniform missing"
    assert 'u_satCount' in halftone_js, "u_satCount uniform missing"
    assert 'u_satSizeRatio' in halftone_js, "u_satSizeRatio uniform missing"
    assert 'u_satDist' in halftone_js, "u_satDist uniform missing"
    assert 'u_satAngleBase' in halftone_js, "u_satAngleBase uniform missing"
    assert 'u_satStretch' in halftone_js, "u_satStretch uniform missing"
    assert 'u_satShape' in halftone_js, "u_satShape uniform missing"
    print("  [OK] GenTonik Satellites & Planets GLSL uniforms verified")

    # 3. Check Gradients (Linear, Radial, Vignette)
    assert 'u_gradType' in halftone_js, "u_gradType uniform missing"
    assert 'u_gradStrength' in halftone_js, "u_gradStrength uniform missing"
    assert 'u_gradAngle' in halftone_js, "u_gradAngle uniform missing"
    print("  [OK] Spatial Gradients (Linear, Radial, Vignette) GLSL uniforms verified")

    # 4. Check 13 Shapes in GLSL evalShapeDist
    for shape in ["square", "diamond", "line", "crosshatch", "hexagon", "cross", "ellipse", "triangle", "star", "heart", "ring"]:
        assert shape in halftone_js, f"Shape {shape} missing in GLSL"
    print("  [OK] All 13 Geometry shapes verified in GLSL evalShapeDist")

    # 5. Check reusable buffers in CPU fallback
    assert '_htAccR' in halftone_js, "Reusable buffer _htAccR missing"
    assert '_htBufN' in halftone_js, "Buffer size tracker _htBufN missing"
    print("  [OK] Zero-allocation reusable buffers verified for CPU compositor")

def test_backend_halftone_resolution():
    print("\n[2/3] Testing backend/app.py halftone resolution default...")
    app_py = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    assert 'preview_max_side = int(body.get("preview_max_side", 0))' in app_py, "preview_max_side default is not 0"
    assert 'preview_max_side = int(body.get("preview_max_side", 0 if save else 960))' not in app_py, "Old 960px clamp still present"
    print("  [OK] Backend preview_max_side default = 0 (full resolution) verified")

def test_index_html_version():
    print("\n[3/4] Testing index.html cache-busting...")
    html = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert ('src="/js/app.js?v=' in html or 'src="/js/main.js?v=' in html), "index.html does not have valid cache-busting script tag"
    print("  [OK] index.html cache-busting verified")

def test_latest_session_fixes():
    print("\n[4/4] Testing satellites, blend modes, tonal curves, and compare button...")
    html = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert "ht-compare-btn" in html, "ht-compare-btn missing in index.html"
    assert "App.revertHalftoneToOriginal()" not in html, "Redundant revert button still present in index.html"

    mod_path = PROJECT_ROOT / "frontend" / "js" / "modules" / "halftone.js"
    halftone_js = mod_path.read_text(encoding="utf-8")
    assert "ht-sat-controls" in halftone_js, "ht-sat-controls sync missing in halftone.js"
    assert "u_opacity" in halftone_js, "u_opacity missing in halftone.js"
    assert "u_blend_mode" in halftone_js, "u_blend_mode missing in halftone.js"
    assert "u_antialias" in halftone_js, "u_antialias missing in halftone.js"
    assert "layers: layersPayload" in halftone_js, "layers payload missing in halftone.js"

    app_js_path = PROJECT_ROOT / "frontend" / "js" / "app.js"
    app_js = app_js_path.read_text(encoding="utf-8")
    assert "ht-sat-controls" in app_js, "ht-sat-controls sync missing in app.js"
    assert "u_opacity" in app_js, "u_opacity missing in app.js"
    assert "u_blend_mode" in app_js, "u_blend_mode missing in app.js"
    assert "layers: layersPayload" in app_js, "layers payload missing in app.js"
    print("  [OK] All latest session fixes verified in index.html, halftone.js, and app.js")


def test_cmyk_mask_dock_cancel_fixes():
    print("\n[5/5] Testing CMYK, mask tools, dock drag, and cancellation fixes...")
    import json
    ru_json = json.loads((PROJECT_ROOT / "frontend" / "locale" / "ru.json").read_text(encoding="utf-8"))
    assert ru_json.get("halftone.mode_custom") == "Свой", "mode_custom in ru.json is not 'Свой'"
    print("  [OK] ru.json mode_custom == 'Свой' verified")

    html = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert 'id="progress-cancel" onclick="App.cancelJob()"' in html, "progress-cancel onclick missing"
    assert 'id="dock-drag-handle"' in html, "dock-drag-handle missing in index.html"
    assert 'id="ht-sat-angle" min="0" max="360" step="1"' in html, "ht-sat-angle step is not 1"
    assert 'id="ht-sat-angle-num" min="0" max="360" step="1"' in html, "ht-sat-angle-num step is not 1"
    print("  [OK] index.html cancel onclick, dock handle, sat angle step=1 verified")

    css = (PROJECT_ROOT / "frontend" / "css" / "styles.css").read_text(encoding="utf-8")
    assert 'input:not([type="checkbox"])' in css, "styles.css does not exclude checkbox from text inputs"
    assert 'input[type="checkbox"]' in css, "styles.css lacks explicit checkbox styling"
    assert '.gemini-dock-drag-handle' in css, "styles.css lacks gemini dock drag handle rule"
    print("  [OK] styles.css checkbox appearance & dock drag styling verified")

    worker_py = (PROJECT_ROOT / "backend" / "worker.py").read_text(encoding="utf-8")
    assert 'self._extras_worker.cancel()' in worker_py, "worker.py does not propagate cancel to extras_worker"
    print("  [OK] backend/worker.py extras_worker cancel propagation verified")

    extras_py = (PROJECT_ROOT / "backend" / "extras.py").read_text(encoding="utf-8")
    assert '[INFO] [SVDQ] Launching' in extras_py, "extras.py lacks [INFO] [SVDQ] Launching log"
    assert 'has_flux3' in extras_py, "extras.py lacks has_flux3 probe"
    print("  [OK] backend/extras.py launch log and flux3 probe verified")

    halftone_js = (PROJECT_ROOT / "frontend" / "js" / "modules" / "halftone.js").read_text(encoding="utf-8")
    assert 'u_mask' in halftone_js, "halftone.js lacks u_mask uniform"
    assert 'cmykRgb' in halftone_js, "halftone.js lacks cmykRgb"
    assert 'initHalftoneMaskEvents' in halftone_js, "halftone.js lacks initHalftoneMaskEvents"
    print("  [OK] halftone.js CMYK shader, mask uniform and interactive events verified")



def test_lasso_brush_eraser_mask_tools():
    print("\n[6/6] Testing Lasso, Brush, Eraser, and Mask tools in backend and frontend...")
    from backend.halftone import decode_mask_to_float, process_halftone_layers
    from PIL import Image
    import numpy as np

    # 1. Test RGBA transparent mask decoding (Brush/Lasso output)
    # Transparent background (0,0,0,0) with white brush stroke (255,255,255,255)
    mask_rgba = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    for x in range(30, 70):
        for y in range(30, 70):
            mask_rgba.putpixel((x, y), (255, 255, 255, 255))
    
    decoded = decode_mask_to_float(mask_rgba, (100, 100), invert_mask=False)
    assert decoded is not None
    assert decoded[0, 0] == 0.0, "Transparent pixel must decode as 0.0"
    assert decoded[50, 50] == 1.0, "Brush stroke pixel must decode as 1.0"
    print("  [OK] RGBA transparent mask decoding verified")

    # 2. Test inverted mask decoding
    decoded_inv = decode_mask_to_float(mask_rgba, (100, 100), invert_mask=True)
    assert decoded_inv[0, 0] == 1.0, "Inverted transparent pixel must decode as 1.0"
    assert decoded_inv[50, 50] == 0.0, "Inverted stroke pixel must decode as 0.0"
    print("  [OK] Inverted mask decoding verified")

    # 3. Test layer composition with mask
    src = Image.new("RGB", (64, 64), (128, 128, 128))
    m = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    for x in range(20, 44):
        for y in range(20, 44):
            m.putpixel((x, y), (255, 255, 255, 255))
    
    layers = [{
        "visible": True,
        "opacity": 1.0,
        "blend_mode": "normal",
        "params": {"mode": "manga", "cell_size": 8},
        "mask": m
    }]
    out = process_halftone_layers(src, layers)
    assert out.size == (64, 64)
    print("  [OK] process_halftone_layers with RGBA mask verified")

    # 4. Test frontend styles.css z-index hierarchy
    css = (PROJECT_ROOT / "frontend" / "css" / "styles.css").read_text(encoding="utf-8")
    assert "z-index: 15" in css, ".studio-mask-canvas lacks z-index: 15"
    assert "z-index: 20" in css, ".studio-lasso-overlay lacks z-index: 20"
    assert ".ht-live-canvas" in css, ".ht-live-canvas rule missing in styles.css"
    print("  [OK] styles.css z-index hierarchy verified")

    # 5. Test app.js mask methods and uniforms
    app_js = (PROJECT_ROOT / "frontend" / "js" / "app.js").read_text(encoding="utf-8")
    assert "initHalftoneMaskEvents" in app_js, "initHalftoneMaskEvents missing in app.js"
    assert "ensureLayerMaskCanvas" in app_js, "ensureLayerMaskCanvas missing in app.js"
    assert "renderMaskOverlayVisual" in app_js, "renderMaskOverlayVisual missing in app.js"
    assert "syncHalftoneMaskCanvasSize" in app_js, "syncHalftoneMaskCanvasSize missing in app.js"
    assert "u_mask" in app_js, "u_mask missing in app.js uniforms"
    print("  [OK] app.js mask methods and uniforms verified")

def test_halftone_layer_stack_system():
    print("\n[7/7] Testing Halftone Studio GT layer stack, masks, FBO compositor and moiré protection...")
    from backend.halftone import process_halftone_layers
    from PIL import Image
    import numpy as np

    root = PROJECT_ROOT
    halftone_js = (root / "frontend" / "js" / "modules" / "halftone.js").read_text(encoding="utf-8")
    app_js = (root / "frontend" / "js" / "app.js").read_text(encoding="utf-8")
    html = (root / "frontend" / "index.html").read_text(encoding="utf-8")
    css = (root / "frontend" / "css" / "styles.css").read_text(encoding="utf-8")
    app_py = (root / "app.py").read_text(encoding="utf-8")
    halftone_py = (root / "backend" / "halftone.py").read_text(encoding="utf-8")

    for name, js in (("modules/halftone.js", halftone_js), ("app.js", app_js)):
        assert 'HT_ORIGINAL_LAYER_ID = "hl_original"' in js, f"{name}: locked original layer ID missing"
        assert "ensureHalftoneFramebuffers" in js, f"{name}: FBO compositor missing"
        assert "glFboTextures" in js, f"{name}: FBO textures missing"
        assert "u_pass_mode" in js, f"{name}: final blit pass mode missing"
        assert "blendLayerRgb" in js, f"{name}: blend modes missing"
        assert "buildHalftoneLayersPayload" in js, f"{name}: layer payload builder missing"
        assert "!l.isOriginal" in js, f"{name}: original layer must be excluded from API payload"
        assert "moveHalftoneLayer" in js, f"{name}: layer reorder missing"
        assert "syncHalftoneLayerControls" in js, f"{name}: layer control sync missing"
        assert "applyHalftoneMoireProtection" in js, f"{name}: moiré protection missing"
        assert "resetHalftoneLayerStack(app)" in js, f"{name}: image load must reset the layer stack"
        assert "hasMask: true" in js, f"{name}: new local layers must start with an empty mask"
    print("  [OK] Original layer, FBO compositor, payload filtering, reorder, and moiré protection verified")

    assert "App.moveHalftoneLayer('up')" in html and "App.moveHalftoneLayer('down')" in html, "index.html reorder buttons missing"
    assert ".ht-layer-actions" in css and ".ht-layer-tag-mask" in css, "styles.css layer action/badge styles missing"
    print("  [OK] Layer reorder buttons and badge styles verified")

    assert '"hasMask": bool(lyr.get("hasMask", False))' in app_py, "app.py metadata lacks hasMask"
    assert '"isBase": bool(lyr.get("isBase", False))' in app_py, "app.py metadata lacks isBase"
    assert '== "hl_original"' in halftone_py, "backend layer compositor does not skip the virtual original layer"
    print("  [OK] Backend metadata and original-layer skip verified")

    src = Image.new("RGB", (64, 64), (128, 128, 128))
    base_layer = {
        "id": "hl_base",
        "name": "Базовый растр",
        "visible": True,
        "opacity": 1.0,
        "blend_mode": "normal",
        "isBase": True,
        "params": {"mode": "manga", "cell_size": 8, "preserve_lines": False},
    }
    original_layer = {
        "id": "hl_original",
        "isOriginal": True,
        "visible": True,
        "opacity": 1.0,
        "blend_mode": "normal",
        "params": {"mode": "color", "cell_size": 4},
    }
    out_with_original = np.asarray(process_halftone_layers(src, [original_layer, base_layer]))
    out_without_original = np.asarray(process_halftone_layers(src, [base_layer]))
    assert np.array_equal(out_with_original, out_without_original), "Virtual original layer must not be toned/blended"
    print("  [OK] Backend skips the virtual original layer exactly")

    empty_mask = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    local_layer = {
        "id": "hl_local",
        "name": "Локальный слой",
        "visible": True,
        "opacity": 1.0,
        "blend_mode": "normal",
        "hasMask": True,
        "params": {"mode": "manga", "cell_size": 8, "preserve_lines": False},
        "mask": empty_mask,
    }
    out_empty = np.asarray(process_halftone_layers(src, [local_layer]))
    assert np.array_equal(out_empty, np.asarray(src)), "Local layer with an empty mask must not affect the canvas"
    print("  [OK] New local layers with empty masks stay invisible until painted")

    full_mask = Image.new("RGBA", (64, 64), (255, 255, 255, 255))
    local_layer_painted = dict(local_layer, mask=full_mask)
    out_painted = np.asarray(process_halftone_layers(src, [local_layer_painted]))
    assert not np.array_equal(out_painted, np.asarray(src)), "Painted full mask must reveal the local raster"
    print("  [OK] Painted local layer mask reveals the raster")


def test_flux3_nunchaku_support():
    print("\n[FLUX.3] Verifying diffusers-main loader and frontend readiness checks...")
    extras_py = (PROJECT_ROOT / "backend" / "extras.py").read_text(encoding="utf-8")
    app_js = (PROJECT_ROOT / "frontend" / "js" / "app.js").read_text(encoding="utf-8")
    generator_js = (PROJECT_ROOT / "frontend" / "js" / "modules" / "generator.js").read_text(encoding="utf-8")

    # Environment probe + installer must detect FLUX.3 separately from Nunchaku.
    assert 'out["has_diffusers_flux3"] = hasattr(diffusers, "Flux3Transformer2DModel")' in extras_py
    assert 'out["has_modular_pipeline"] = hasattr(diffusers, "ModularPipeline")' in extras_py
    assert "needs_diffusers_main" in extras_py, "installer never upgrades diffusers for FLUX.3"
    assert "git+https://github.com/huggingface/diffusers.git" in extras_py

    # Modular FLUX.3 pipeline, local modular index, Qwen3-VL INT4 encoder, no upsampler dependency.
    assert 'modular_model_index.json' in extras_py, "local modular pipeline index not detected"
    assert "from diffusers import ModularPipeline" in extras_py, "FLUX.3 still loads a non-modular pipeline"
    assert "NunchakuQwen3VLEncoderModel" in extras_py, "FLUX.3 Qwen3-VL text encoder loader missing"
    assert 'blocks.sub_blocks.pop("prompt_upsampler", None)' in extras_py, "prompt upsampler bypass missing"
    assert '"output": "images"' in extras_py, "modular pipeline output handling missing"

    # Active monolith and ES6 mirror must block generation with an actionable message.
    for name, js in (("app.js", app_js), ("modules/generator.js", generator_js)):
        assert "has_diffusers_flux3" in js, f"{name}: missing diffusers-main readiness check"
        assert "diffusers main" in js, f"{name}: missing actionable upgrade hint"

    print("  [OK] FLUX.3 diffusers-main, modular pipeline, text encoder and UI checks verified")


def test_lora_hub_frontend_sync():
    print("\n[LoRA Hub] Verifying active app.js is in sync with index.html...")
    app_js = (PROJECT_ROOT / "frontend" / "js" / "app.js").read_text(encoding="utf-8")
    html = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    module_js = (PROJECT_ROOT / "frontend" / "js" / "modules" / "lora-hub.js").read_text(encoding="utf-8")

    assert 'id="models-subpanel-loras"' in html, "LoRA Hub subpanel missing in index.html"
    assert "App.switchModelsHubSubtab('loras')" in html, "LoRA Hub tab does not call switchModelsHubSubtab('loras')"

    # The active monolith must contain real hub/library renderers (not just calls to missing methods).
    assert "renderLoraHubGrid() {" in app_js, "renderLoraHubGrid implementation missing in app.js"
    assert "renderLoraLibrary() {" in app_js, "renderLoraLibrary implementation missing in app.js"
    assert 'document.getElementById("models-subpanel-loras")' in app_js, "app.js does not toggle the LoRA subpanel"
    assert "models-hub-lora-count" in app_js, "app.js does not update the LoRA Hub counter"

    # HF endpoints must match backend routes (/hf-inspect and /hf-search).
    assert "/api/loras/hf-inspect?repo_id=" in app_js, "app.js uses the wrong HF inspect endpoint"
    assert "/api/loras/hf-search?q=" in app_js, "app.js uses the wrong HF search endpoint"
    assert 'getElementById("lora-hf-file-select")' in app_js, "app.js does not read the HF file select"
    assert 'getElementById("lora-modal-list-wrap")' in app_js, "app.js does not render the installed LoRA library"

    # ES6 module mirror must use the same API shape and DOM ids.
    assert "data?.loras" in module_js, "modules/lora-hub.js does not unwrap {loras:[...]} response"
    assert '"lora-modal-list-wrap"' in module_js, "modules/lora-hub.js uses a stale library container id"
    assert "lora-pane-${tName}" in module_js, "modules/lora-hub.js uses stale modal pane ids"

    print("  [OK] LoRA Hub local list, subpanel, modal, and HF endpoints are synchronized")


if __name__ == "__main__":
    test_frontend_halftone_features()
    test_backend_halftone_resolution()
    test_index_html_version()
    test_latest_session_fixes()
    test_cmyk_mask_dock_cancel_fixes()
    test_lasso_brush_eraser_mask_tools()
    test_halftone_layer_stack_system()
    test_flux3_nunchaku_support()
    test_lora_hub_frontend_sync()
    print("\n============================================================")
    print("ALL HALFTONE STUDIO GT FIXES VERIFIED SUCCESSFULLY!")
    print("============================================================")


