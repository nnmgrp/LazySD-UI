"""Tests for Highres Fix (Option A) in stable-diffusion.cpp (SD.CPP) engine."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

PROJECT_ROOT = Path(r"G:\QuantFunc-WebUI")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.sd_cpp import SDCppWorker
from backend.gpu import GPUInfo


@pytest.fixture
def mock_gpu_16gb():
    return GPUInfo(
        name="NVIDIA GeForce RTX 5060 Ti",
        sm=120,
        vram_mb=16311,
        driver_version="610.88",
        cuda_version="13.0",
        tier="50x-above",
        klein_tier="50x",
    )


def test_sd_cpp_hires_fix_flags_2048x2048(tmp_path, mock_gpu_16gb):
    worker = SDCppWorker()
    worker.cli_path = Path("G:/QuantFunc-WebUI/bin/windows/sd_cpp/sd-cli.exe")

    diff_file = tmp_path / "diffusion.gguf"
    diff_file.write_text("dummy")

    mock_model = MagicMock()
    mock_model.id = "qwen_image_2_1_gguf"
    mock_model.family = "qwen"
    mock_model.sources.primary.repo_id = "test_qwen_repo"
    mock_model.files = {"diffusion": str(diff_file)}
    mock_model.engine_args = {}
    mock_model.scheduler_presets = {}
    mock_model.transformers = []
    mock_model.text_encoders = []
    mock_model.get_default_cfg.return_value = 3.0

    mock_request = MagicMock()
    mock_request.prompt = "A high-detail 2K landscape"
    mock_request.output_filename = "test_hires.png"
    mock_request.width = 2048
    mock_request.height = 2048
    mock_request.steps = 28
    mock_request.cfg = 3.0
    mock_request.sampler = "euler"
    mock_request.seed = 12345
    mock_request.init_image_path = None
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

    # Highres Fix parameters
    mock_request.hires_fix = True
    mock_request.hires_scale = 2.0
    mock_request.hires_steps = 6
    mock_request.hires_denoising_strength = 0.35
    mock_request.hires_upscaler = "Latent (bicubic)"

    captured_cmd = []

    def mock_popen(cmd, *args, **kwargs):
        nonlocal captured_cmd
        captured_cmd = list(cmd)
        if "-o" in captured_cmd:
            out_p = Path(captured_cmd[captured_cmd.index("-o") + 1])
            out_p.parent.mkdir(parents=True, exist_ok=True)
            from PIL import Image as _PIL_Img
            _PIL_Img.new("RGB", (256, 256), (128, 128, 128)).save(out_p, format="PNG")
        mock_proc = MagicMock()
        mock_proc.stdout = iter(["[I] done\n"])
        mock_proc.poll.return_value = 0
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0
        return mock_proc

    with patch("subprocess.Popen", side_effect=mock_popen):
        res = worker.generate(mock_request, mock_model, mock_gpu_16gb)

    # 1. Base resolution must be 1024x1024 for fast first pass
    assert "-W" in captured_cmd
    w_idx = captured_cmd.index("-W")
    assert captured_cmd[w_idx + 1] == "1024"
    assert "-H" in captured_cmd
    h_idx = captured_cmd.index("-H")
    assert captured_cmd[h_idx + 1] == "1024"

    # 2. Highres Fix flags must be present with target 2048x2048
    assert "--hires" in captured_cmd
    assert "--hires-width" in captured_cmd
    hw_idx = captured_cmd.index("--hires-width")
    assert captured_cmd[hw_idx + 1] == "2048"

    assert "--hires-height" in captured_cmd
    hh_idx = captured_cmd.index("--hires-height")
    assert captured_cmd[hh_idx + 1] == "2048"

    assert "--hires-steps" in captured_cmd
    hs_idx = captured_cmd.index("--hires-steps")
    assert captured_cmd[hs_idx + 1] == "6"

    assert "--hires-denoising-strength" in captured_cmd
    hd_idx = captured_cmd.index("--hires-denoising-strength")
    assert captured_cmd[hd_idx + 1] == "0.35"

    assert "--hires-upscaler" in captured_cmd
    hu_idx = captured_cmd.index("--hires-upscaler")
    assert captured_cmd[hu_idx + 1] == "Latent (bicubic)"

    # 3. VAE tiling must be enabled for 2K
    assert "--vae-tiling" in captured_cmd


def test_sd_cpp_hires_fix_flags_1024x1024_upscale(tmp_path, mock_gpu_16gb):
    worker = SDCppWorker()
    worker.cli_path = Path("G:/QuantFunc-WebUI/bin/windows/sd_cpp/sd-cli.exe")

    diff_file = tmp_path / "diffusion.gguf"
    diff_file.write_text("dummy")

    mock_model = MagicMock()
    mock_model.id = "qwen_image_2_1_gguf"
    mock_model.family = "qwen"
    mock_model.sources.primary.repo_id = "test_qwen_repo"
    mock_model.files = {"diffusion": str(diff_file)}
    mock_model.engine_args = {}
    mock_model.scheduler_presets = {}
    mock_model.transformers = []
    mock_model.text_encoders = []
    mock_model.get_default_cfg.return_value = 3.0

    mock_request = MagicMock()
    mock_request.prompt = "A 1024px base scaled to 2048px"
    mock_request.output_filename = "test_hires_scale.png"
    mock_request.width = 1024
    mock_request.height = 1024
    mock_request.steps = 20
    mock_request.cfg = 3.0
    mock_request.sampler = "euler"
    mock_request.seed = 54321
    mock_request.init_image_path = None
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

    # Highres Fix parameters
    mock_request.hires_fix = True
    mock_request.hires_scale = 2.0
    mock_request.hires_steps = 8
    mock_request.hires_denoising_strength = 0.4
    mock_request.hires_upscaler = "Latent"

    captured_cmd = []

    def mock_popen(cmd, *args, **kwargs):
        nonlocal captured_cmd
        captured_cmd = list(cmd)
        if "-o" in captured_cmd:
            out_p = Path(captured_cmd[captured_cmd.index("-o") + 1])
            out_p.parent.mkdir(parents=True, exist_ok=True)
            from PIL import Image as _PIL_Img
            _PIL_Img.new("RGB", (256, 256), (128, 128, 128)).save(out_p, format="PNG")
        mock_proc = MagicMock()
        mock_proc.stdout = iter(["[I] done\n"])
        mock_proc.poll.return_value = 0
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0
        return mock_proc

    with patch("subprocess.Popen", side_effect=mock_popen):
        res = worker.generate(mock_request, mock_model, mock_gpu_16gb)

    # Base is 1024x1024, target is 2048x2048
    w_idx = captured_cmd.index("-W")
    assert captured_cmd[w_idx + 1] == "1024"
    h_idx = captured_cmd.index("-H")
    assert captured_cmd[h_idx + 1] == "1024"

    hw_idx = captured_cmd.index("--hires-width")
    assert captured_cmd[hw_idx + 1] == "2048"
    hh_idx = captured_cmd.index("--hires-height")
    assert captured_cmd[hh_idx + 1] == "2048"

    hs_idx = captured_cmd.index("--hires-steps")
    assert captured_cmd[hs_idx + 1] == "8"


def test_sd_cpp_hires_bypassed_when_mask_present(tmp_path, mock_gpu_16gb):
    worker = SDCppWorker()
    worker.cli_path = Path("G:/QuantFunc-WebUI/bin/windows/sd_cpp/sd-cli.exe")

    diff_file = tmp_path / "diffusion.gguf"
    diff_file.write_text("dummy")
    src_img = tmp_path / "source.png"
    from PIL import Image
    Image.new("RGB", (1024, 1024), (255, 255, 255)).save(src_img)
    mask_file = tmp_path / "mask.png"
    Image.new("L", (1024, 1024), 255).save(mask_file)

    mock_model = MagicMock()
    mock_model.id = "qwen_image_2_1_gguf"
    mock_model.family = "qwen"
    mock_model.sources.primary.repo_id = "test_qwen_repo"
    mock_model.files = {"diffusion": str(diff_file)}
    mock_model.engine_args = {}
    mock_model.scheduler_presets = {}
    mock_model.transformers = []
    mock_model.text_encoders = []
    mock_model.get_default_cfg.return_value = 3.0

    mock_request = MagicMock()
    mock_request.prompt = "inpaint test"
    mock_request.output_filename = "test_inpaint.png"
    mock_request.width = 1024
    mock_request.height = 1024
    mock_request.steps = 20
    mock_request.cfg = 3.0
    mock_request.sampler = "euler"
    mock_request.seed = 111
    mock_request.init_image_path = str(src_img)
    mock_request.ref_mode = "control_ref"
    mock_request.mask_data_url = None
    mock_request.mask_path = mask_file
    mock_request.mask_strength = 1.0
    mock_request.mask_grow = 6
    mock_request.mask_blur = 0.0
    mock_request.mask_invert = False
    mock_request.mask_crop_mode = "auto"
    mock_request.mask_crop_padding = 64
    mock_request.mask_crop_min_size = 256
    mock_request.mask_crop_box = None
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

    # Hires requested, but mask is present
    mock_request.hires_fix = True
    mock_request.hires_scale = 2.0
    mock_request.hires_steps = 6

    captured_cmd = []

    def mock_popen(cmd, *args, **kwargs):
        nonlocal captured_cmd
        captured_cmd = list(cmd)
        if "-o" in captured_cmd:
            out_p = Path(captured_cmd[captured_cmd.index("-o") + 1])
            out_p.parent.mkdir(parents=True, exist_ok=True)
            from PIL import Image as _PIL_Img
            _PIL_Img.new("RGB", (256, 256), (128, 128, 128)).save(out_p, format="PNG")
        mock_proc = MagicMock()
        mock_proc.stdout = iter(["[I] done\n"])
        mock_proc.poll.return_value = 0
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0
        return mock_proc

    with patch("subprocess.Popen", side_effect=mock_popen):
        res = worker.generate(mock_request, mock_model, mock_gpu_16gb)

    # Highres fix must be disabled when inpaint mask is active
    assert "--hires" not in captured_cmd


def test_sd_cpp_hires_pid_mapping(tmp_path, mock_gpu_16gb):
    worker = SDCppWorker()
    worker.cli_path = Path("G:/QuantFunc-WebUI/bin/windows/sd_cpp/sd-cli.exe")

    diff_file = tmp_path / "diffusion.gguf"
    diff_file.write_text("dummy")

    mock_model = MagicMock()
    mock_model.id = "qwen_image_2_1_gguf"
    mock_model.family = "qwen"
    mock_model.sources.primary.repo_id = "test_qwen_repo"
    mock_model.files = {"diffusion": str(diff_file)}
    mock_model.engine_args = {}
    mock_model.scheduler_presets = {}
    mock_model.transformers = []
    mock_model.text_encoders = []
    mock_model.get_default_cfg.return_value = 3.0

    mock_request = MagicMock()
    mock_request.prompt = "Test PiD highres"
    mock_request.output_filename = "test_pid_out.png"
    mock_request.width = 2048
    mock_request.height = 2048
    mock_request.steps = 28
    mock_request.cfg = 3.0
    mock_request.sampler = "euler"
    mock_request.seed = 42
    mock_request.init_image_path = None
    mock_request.ref_mode = "control_ref"
    mock_request.mask_image_path = None
    mock_request.mask_data_url = None
    mock_request.mask_path = None
    mock_request.mask_crop_box = None
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

    # PiD requested as upscaler
    mock_request.hires_fix = True
    mock_request.hires_scale = 2.0
    mock_request.hires_steps = 6
    mock_request.hires_denoising_strength = 0.35
    mock_request.hires_upscaler = "NVIDIA PiD"

    captured_cmd = []

    def mock_popen(cmd, *args, **kwargs):
        nonlocal captured_cmd
        captured_cmd = list(cmd)
        if "-o" in captured_cmd:
            out_p = Path(captured_cmd[captured_cmd.index("-o") + 1])
            out_p.parent.mkdir(parents=True, exist_ok=True)
            from PIL import Image as _PIL_Img
            _PIL_Img.new("RGB", (256, 256), (128, 128, 128)).save(out_p, format="PNG")
        mock_proc = MagicMock()
        mock_proc.stdout = iter(["[I] done\n"])
        mock_proc.poll.return_value = 0
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0
        return mock_proc

    with patch("subprocess.Popen", side_effect=mock_popen):
        res = worker.generate(mock_request, mock_model, mock_gpu_16gb)

    # 1. Base resolution must be 1024x1024
    assert captured_cmd[captured_cmd.index("-W") + 1] == "1024"
    assert captured_cmd[captured_cmd.index("-H") + 1] == "1024"

    # 2. PiD maps to "from_ldm" in sd-cli and auto-sets steps to 4
    assert "--hires-upscaler" in captured_cmd
    assert captured_cmd[captured_cmd.index("--hires-upscaler") + 1] == "from_ldm"
    assert captured_cmd[captured_cmd.index("--hires-steps") + 1] == "4"
    assert captured_cmd[captured_cmd.index("--hires-width") + 1] == "2048"
