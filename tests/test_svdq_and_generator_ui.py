import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, "G:/QuantFunc-WebUI")

from fastapi.testclient import TestClient
from app import create_app

def test_features():
    print("[1] Initializing TestClient...")
    app = create_app()
    client = TestClient(app)

    print("[2] Testing GET /api/config...")
    resp = client.get("/api/config")
    assert resp.status_code == 200, f"GET /api/config failed: {resp.status_code}"
    cfg = resp.json()
    assert "settings" in cfg
    assert "hide_uninstalled_in_generator" in cfg["settings"]
    print(f"  hide_uninstalled_in_generator default: {cfg['settings']['hide_uninstalled_in_generator']}")

    print("[3] Testing PATCH /api/settings for hide_uninstalled_in_generator...")
    patch_resp = client.patch("/api/settings", json={"hide_uninstalled_in_generator": True})
    assert patch_resp.status_code == 200, f"PATCH failed: {patch_resp.text}"
    assert "hide_uninstalled_in_generator" in patch_resp.json().get("updated", [])

    # Verify updated config
    resp2 = client.get("/api/config")
    assert resp2.json()["settings"]["hide_uninstalled_in_generator"] is True
    print("  Successfully set hide_uninstalled_in_generator = True")

    # Restore to False
    client.patch("/api/settings", json={"hide_uninstalled_in_generator": False})
    resp3 = client.get("/api/config")
    assert resp3.json()["settings"]["hide_uninstalled_in_generator"] is False
    print("  Successfully restored hide_uninstalled_in_generator = False")

    print("[4] Checking Nunchaku/SVDQ model states and text encoders...")
    models = cfg.get("models", [])
    model_ids = [m["id"] for m in models]
    print(f"  Total models loaded: {len(models)}")
    
    nunchaku_models = [
        "flux_3_image_klein_nunchaku",
        "qwen_image_2_1_nunchaku",
        "qwen_image_2_1_turbo_nunchaku",
        "flux_2_klein_9b_nunchaku",
        "flux_2_klein_9b_kv_nunchaku",
        "flux_2_klein_4b_nunchaku",
    ]
    for mid in nunchaku_models:
        assert mid in model_ids, f"Model {mid} missing from registry!"
        st_resp = client.get(f"/api/models/{mid}/state")
        assert st_resp.status_code == 200, f"Failed state for {mid}"
        st = st_resp.json()
        print(f"  {mid}: transformers={len(st.get('transformers', []))}, text_encoders={len(st.get('text_encoders', []))}, base_installed={st.get('base_model_installed')}")
        # Verify text encoder is present
        assert len(st.get("text_encoders", [])) > 0, f"No text encoder in {mid}!"

    print("[5] Checking frontend files...")
    html_text = Path("G:/QuantFunc-WebUI/frontend/index.html").read_text(encoding="utf-8")
    assert "id=\"gen-uninstalled-notice\"" in html_text
    assert "App.dismissUninstalledNotice()" in html_text
    assert "id=\"setting-hide-uninstalled-gen\"" in html_text
    assert "/js/app.js?v=" in html_text

    css_text = Path("G:/QuantFunc-WebUI/frontend/css/styles.css").read_text(encoding="utf-8")
    assert ".gen-hub-notice {" in css_text
    assert "top: 18px;" in css_text
    assert ".notice-close-btn" in css_text

    js_text = Path("G:/QuantFunc-WebUI/frontend/js/app.js").read_text(encoding="utf-8")
    assert "dismissUninstalledNotice()" in js_text
    assert "onSettingHideUninstalledChange(" in js_text
    assert "hide_uninstalled_in_generator" in js_text

    print("\n>>> ALL TESTS PASSED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    test_features()
