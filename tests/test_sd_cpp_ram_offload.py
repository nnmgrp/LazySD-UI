"""Tests for SD.CPP reference resolution preservation and RAM offload flags.

Verifies:
1. Native source image resolution is preserved without artificial upscaling to target generation size.
2. RAM offloading (--params-backend cpu, --vae-tiling) is enabled for resolutions > 1024 or <= 16GB GPUs.
3. Heavy ViT vision encoder workloads (>= 1400px) are offloaded to system RAM (--backend te=cpu).
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image

PROJECT_ROOT = Path(r"G:\QuantFunc-WebUI")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.sd_cpp import SDCppWorker
from backend.gpu import GPUInfo


@pytest.fixture
def tmp_source_image(tmp_path):
    img_path = tmp_path / "source_1500.png"
    im = Image.new("RGB", (1500, 1500), color=(200, 50, 50))
    im.save(img_path)
    return img_path


def test_sd_cpp_preserves_ref_resolution_and_enables_ram_offload(tmp_path, tmp_source_image):
    worker = SDCppWorker()
    worker.cli_path = Path("G:/QuantFunc-WebUI/bin/windows/sd_cpp/sd-cli.exe")

    diff_file = tmp_path / "diffusion.gguf"
    diff_file.write_text("dummy")

    # Mock ModelInfo
    mock_model = MagicMock()
    mock_model.id = "qwen_image_edit"
    mock_model.family = "qwen"
    mock_model.sources.primary.repo_id = "test_qwen_repo"
    mock_model.files = {"diffusion": str(diff_file)}
    mock_model.engine_args = {}
    mock_model.scheduler_presets = {}
    mock_model.transformers = []
    mock_model.text_encoders = []
    mock_model.get_default_cfg.return_value = 3.0

    # Mock Request with 2048x2048 generation target and 1500x1500 reference image
    mock_request = MagicMock()
    mock_request.prompt = "A restored comic panel"
    mock_request.output_filename = "test_out.png"
    mock_request.width = 2048
    mock_request.height = 2048
    mock_request.steps = 28
    mock_request.cfg = 3.0
    mock_request.sampler = "euler"
    mock_request.seed = 42
    mock_request.init_image_path = str(tmp_source_image)
    mock_request.ref_mode = "control_ref"
    mock_request.mask_image_path = None
    mock_request.mask_data_url = None
    mock_request.mask_path = None
    mock_request.lora_paths = []
    mock_request.lora_strengths = []
    mock_request.negative_prompt = ""
    mock_request.custom_text_encoder_path = None
    mock_request.custom_vae_path = None
    mock_request.vae = None
    mock_request.text_encoder_id = None
    mock_request.control_net_path = None
    mock_request.variant_id = None
    mock_request.offload_te_cpu = False

    # Mock 16GB RTX 5060 Ti
    gpu_info = GPUInfo(
        name="NVIDIA GeForce RTX 5060 Ti",
        sm=120,
        vram_mb=16311,
        driver_version="610.88",
        cuda_version="13.0",
        tier="50x-above",
        klein_tier="50x",
    )

    captured_cmd = []

    def mock_popen(cmd, *args, **kwargs):
        nonlocal captured_cmd
        captured_cmd = list(cmd)
        mock_proc = MagicMock()
        mock_proc.stdout = iter(["[I] done\n"])
        mock_proc.poll.return_value = 0
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0
        return mock_proc

    from backend.cache import CacheManager
    from backend.settings import get_settings
    settings = get_settings()
    repo_dir = CacheManager().repo_dir_for("test_qwen_repo")
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "mmproj-F16.gguf").write_text("dummy mmproj")

    output_path = settings.output_dir / "test_out.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("dummy output")

    with patch("subprocess.Popen", side_effect=mock_popen):
        worker.generate(mock_request, mock_model, gpu_info=gpu_info)

    assert len(captured_cmd) > 0, "SD.CPP command was not captured"

    # 1. Verify resolution is NOT forcibly upscaled to 2048x2048
    # 1500 aligned to 32 is (1500 // 32) * 32 = 1472
    preproc_idx = -1
    for i, arg in enumerate(captured_cmd):
        if arg == "--image-preprocess":
            preproc_idx = i + 1
            break
    assert preproc_idx != -1, "--image-preprocess must be in command"
    preproc_str = captured_cmd[preproc_idx]
    assert "width=1472,height=1472" in preproc_str, (
        f"Expected width=1472,height=1472 (native 1500px ref), but got: {preproc_str}"
    )
    assert "width=2048,height=2048" not in preproc_str, (
        "Reference was artificially upscaled to 2048x2048!"
    )

    # 2. Verify RAM offload and VAE tiling are active for 16GB GPU / high res
    assert "--params-backend" in captured_cmd, "--params-backend must be present"
    assert captured_cmd[captured_cmd.index("--params-backend") + 1] == "cpu"
    assert "--vae-tiling" in captured_cmd, "--vae-tiling must be present"

    # 3. Verify Vision encoder offload to CPU RAM (--backend te=cpu) is active for large ViT input
    assert "--backend" in captured_cmd, "--backend must be present for large ViT input"
    assert captured_cmd[captured_cmd.index("--backend") + 1] == "te=cpu"
