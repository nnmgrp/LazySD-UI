"""
Flow-Matching Samplers, Sigma Interpolation & Img2Img Latent Utilities for SDNQ / Diffusers.

Open-Source Acknowledgements (GPLv3-compatible borrowings):
- vladmandic/sdnext (AGPL-3.0):
  - FlowMatchDPMSolverMultistepScheduler (`modules/schedulers/scheduler_dpm_flowmatch.py`)
  - FlowUniPCMultistepScheduler (`modules/schedulers/scheduler_unipc_flowmatch.py`)
  - FlowMatchHeunDiscreteScheduler `set_timesteps` sigmas/mu patch (`modules/sd_hijack_schedulers.py`)
- huggingface/diffusers (Apache-2.0):
  - Stochastic FlowMatch Euler step (`schedulers/scheduling_flow_match_euler_discrete.py`)
  - QwenImage Img2Img latent scaling (`pipelines/qwenimage/pipeline_qwenimage_img2img.py`)
- modelscope/DiffSynth-Studio & OzzyGT (Apache-2.0):
  - Piecewise Turbo `sample_sigmas` curve interpolation for arbitrary step counts.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def interpolate_sample_sigmas(base_sigmas: list[float], target_steps: int) -> list[float]:
    """Interpolate a reference Turbo sigma schedule (e.g. 8 steps) to any `target_steps`.

    Preserves the non-linear shifted trajectory of distilled FlowMatch models
    instead of falling back to an unshifted linear schedule.
    """
    target_steps = max(1, int(target_steps))
    if not base_sigmas:
        return [round(float(s), 6) for s in np.linspace(1.0, 1.0 / target_steps, target_steps)]
    if target_steps == len(base_sigmas):
        return [float(s) for s in base_sigmas]

    # Ensure terminal 0.0 is included for interpolation endpoints
    full_curve = [float(s) for s in base_sigmas]
    if full_curve[-1] != 0.0:
        full_curve.append(0.0)

    x_old = np.linspace(0.0, 1.0, len(full_curve))
    x_new = np.linspace(0.0, 1.0, target_steps + 1)
    interp = np.interp(x_new, x_old, full_curve)
    # Return the non-zero step sigmas (diffusers appends terminal 0.0 inside set_timesteps)
    return [round(float(max(1e-5, s)), 6) for s in interp[:-1]]


def _try_import_sdnext_schedulers(sdnq_src: Optional[str] = None):
    """Attempt to import FlowMatchDPMSolverMultistepScheduler & FlowUniPCMultistepScheduler from local SD.Next."""
    import types

    # Prevent SD.Next's scheduler_dpm_flowmatch from invoking SD.Next's pip installer on import
    if "installer" not in sys.modules:
        sys.modules["installer"] = types.SimpleNamespace(install=lambda *a, **kw: None)

    candidates = [
        Path(r"G:\SDNext\sdnext"),
    ]
    if sdnq_src:
        # sdnq_src is typically G:\SDNext\sdnext\extensions-builtin\sdnq\src
        try:
            sdnext_root = Path(sdnq_src).resolve().parents[2]
            if sdnext_root.exists() and sdnext_root not in candidates:
                candidates.insert(0, sdnext_root)
        except Exception:
            pass

    for root in candidates:
        sched_dir = root / "modules" / "schedulers"
        if sched_dir.exists():
            root_str = str(root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
            try:
                import importlib

                dpm_mod = importlib.import_module("modules.schedulers.scheduler_dpm_flowmatch")
                unipc_mod = importlib.import_module("modules.schedulers.scheduler_unipc_flowmatch")
                return dpm_mod.FlowMatchDPMSolverMultistepScheduler, unipc_mod.FlowUniPCMultistepScheduler
            except Exception:
                pass
    return None, None


def _patch_heun_flowmatch_scheduler(scheduler, base_cfg: Optional[dict] = None):
    """Patch FlowMatchHeunDiscreteScheduler so it accepts `sigmas`, `mu`, and `set_begin_index`
    (adapted from vladmandic/sdnext modules/sd_hijack_schedulers.py).
    """
    import math
    import torch

    orig_set_timesteps = scheduler.set_timesteps
    sig_params = inspect.signature(orig_set_timesteps).parameters
    use_dyn = bool((base_cfg or {}).get("use_dynamic_shifting", False))
    shift_val = float((base_cfg or {}).get("shift", 1.0))

    if "sigmas" not in sig_params or "mu" not in sig_params:
        def patched_set_timesteps(
            num_inference_steps: Optional[int] = None,
            device=None,
            sigmas: Optional[list[float]] = None,
            mu: Optional[float] = None,
            **kwargs,
        ):
            if sigmas is not None:
                s_arr = np.array(sigmas, dtype=np.float32)
                if use_dyn and mu is not None:
                    # Apply dynamic resolution time_shift matching FlowMatchEulerDiscreteScheduler
                    s_clamped = np.clip(s_arr, 1e-6, 1.0)
                    s_arr = np.where(
                        s_clamped >= 1.0,
                        1.0,
                        math.exp(mu) / (math.exp(mu) + (1.0 / s_clamped - 1.0)),
                    ).astype(np.float32)
                elif shift_val != 1.0:
                    s_arr = (shift_val * s_arr / (1.0 + (shift_val - 1.0) * s_arr)).astype(np.float32)

                timesteps = s_arr * scheduler.config.num_train_timesteps
                s_tensor = torch.from_numpy(s_arr).to(dtype=torch.float32, device=device)
                # Heun is a 2nd-order predictor-corrector: duplicate internal sigmas following diffusers convention
                if len(s_tensor) > 1:
                    s_heun = torch.cat([s_tensor[:1], s_tensor[1:].repeat_interleave(2), torch.zeros(1, device=device, dtype=torch.float32)])
                    t_tensor = torch.from_numpy(timesteps).to(dtype=torch.float32, device=device)
                    t_heun = torch.cat([t_tensor[:1], t_tensor[1:].repeat_interleave(2)])
                else:
                    s_heun = torch.cat([s_tensor, torch.zeros(1, device=device, dtype=torch.float32)])
                    t_heun = torch.from_numpy(timesteps).to(dtype=torch.float32, device=device)
                scheduler.timesteps = t_heun
                scheduler.sigmas = s_heun.to("cpu")
                scheduler.num_inference_steps = len(t_heun)
                scheduler._step_index = None
                scheduler._begin_index = None
                scheduler.prev_derivative = None
                scheduler.dt = None
                scheduler.sample = None
                return
            return orig_set_timesteps(num_inference_steps=num_inference_steps, device=device)

        scheduler.set_timesteps = patched_set_timesteps

    if not hasattr(scheduler, "set_begin_index"):
        def set_begin_index(begin_index: int = 0):
            scheduler._begin_index = begin_index
        scheduler.set_begin_index = set_begin_index

    return scheduler


def _patch_unipc_flowmatch_scheduler(scheduler):
    """Patch FlowUniPCMultistepScheduler to avoid log(0) NaN at sigma=1.0/0.0 and keep float32 timesteps."""
    import torch

    def _safe_sigma_to_alpha_sigma_t(sigma):
        s = torch.clamp(torch.as_tensor(sigma, dtype=torch.float32), min=1e-4, max=1.0 - 1e-4)
        return 1.0 - s, s

    scheduler._sigma_to_alpha_sigma_t = _safe_sigma_to_alpha_sigma_t

    orig_set_timesteps = scheduler.set_timesteps

    def patched_set_timesteps(
        num_inference_steps: Optional[int] = None,
        device=None,
        sigmas: Optional[list[float]] = None,
        mu: Optional[float] = None,
        shift: Optional[float] = None,
    ):
        orig_set_timesteps(
            num_inference_steps=num_inference_steps,
            device=device,
            sigmas=sigmas,
            mu=mu,
            shift=shift,
        )
        # Preserve fractional float32 timesteps (sigmas * 1000) instead of truncated int64
        if hasattr(scheduler, "sigmas") and scheduler.sigmas is not None and len(scheduler.sigmas) > 1:
            t_float = (scheduler.sigmas[:-1].to(dtype=torch.float32) * float(scheduler.config.num_train_timesteps))
            scheduler.timesteps = t_float.to(device=device)

    scheduler.set_timesteps = patched_set_timesteps

    orig_step = scheduler.step

    def patched_step(model_output, timestep, sample, *args, **kwargs):
        return_dict = kwargs.get("return_dict", True)
        if scheduler.step_index is None:
            scheduler._init_step_index(timestep)
        s_idx = scheduler.step_index
        sigma = float(scheduler.sigmas[s_idx].item())
        sigma_next = float(scheduler.sigmas[s_idx + 1].item())
        # On final step where sigma_next == 0, land directly on clean x0 prediction to avoid log(0)
        if sigma_next <= 1e-6:
            prev_sample = (sample.to(torch.float32) - sigma * model_output.to(torch.float32)).to(model_output.dtype)
            scheduler._step_index += 1
            if not return_dict:
                return (prev_sample,)
            from diffusers.schedulers.scheduling_utils import SchedulerOutput
            return SchedulerOutput(prev_sample=prev_sample)
        return orig_step(model_output, timestep, sample, *args, **kwargs)

    scheduler.step = patched_step
    return scheduler


def _wrap_stochastic_euler_step(scheduler, eff_eta: float, seed: int):
    """Wrap any FlowMatch scheduler's `step` to support continuous `eta` (Noise slider 0.0 -> 1.0)
    while preserving 1st-order or 2nd-order multistep trajectories.

    At eff_eta == 0.0: exact deterministic scheduler step (Euler, DPM++ 2M, or UniPC).
    At eff_eta > 0.0: runs the underlying deterministic step to obtain `det_prev` (and update
    multistep history), then blends with the variance-preserving ancestral FlowMatch transition.
    """
    import torch

    eff_eta = float(max(0.0, min(1.0, eff_eta)))
    if eff_eta <= 0.0:
        return scheduler

    orig_step = scheduler.step
    noise_gen = torch.Generator(device="cpu").manual_seed(int(seed) ^ 0x5A5A5A5A)

    def stochastic_step(model_output, timestep, sample, *args, **kwargs):
        return_dict = kwargs.get("return_dict", True)
        if scheduler.step_index is None:
            scheduler._init_step_index(timestep)

        s_idx = scheduler.step_index
        sigma = scheduler.sigmas[s_idx].to(device=sample.device, dtype=torch.float32)
        sigma_next = scheduler.sigmas[s_idx + 1].to(device=sample.device, dtype=torch.float32)
        upcast_sample = sample.to(torch.float32)
        upcast_output = model_output.to(torch.float32)

        # Execute the underlying 1st-order or 2nd-order multistep solver
        kwargs_tuple = dict(kwargs)
        kwargs_tuple["return_dict"] = False
        det_prev = orig_step(model_output, timestep, sample, *args, **kwargs_tuple)[0].to(torch.float32)

        if float(sigma_next.item()) > 1e-6:
            ratio = sigma_next / torch.clamp(sigma, min=1e-6)
            denom = 1.0 - ratio
            if float(torch.abs(denom).item()) > 1e-5:
                x0 = (det_prev - ratio * upcast_sample) / denom
            else:
                x0 = upcast_sample - sigma * upcast_output
            eps = torch.randn(
                upcast_sample.shape,
                generator=noise_gen,
                device="cpu",
                dtype=torch.float32,
            ).to(device=sample.device)
            # Variance-preserving interpolation between deterministic trajectory and ancestral re-noising:
            # At eff_eta=0 -> det_prev; at eff_eta=1 -> (1 - sigma_next)*x0 + sigma_next*eps (Diffusers stochastic_sampling)
            clean_part = (1.0 - sigma_next) * x0
            det_noise_part = det_prev - clean_part
            keep_scale = float((1.0 - eff_eta ** 2) ** 0.5)
            prev_sample = clean_part + keep_scale * det_noise_part + eff_eta * sigma_next * eps
        else:
            prev_sample = det_prev

        prev_sample = prev_sample.to(model_output.dtype)

        if not return_dict:
            return (prev_sample,)
        from diffusers.schedulers.scheduling_utils import SchedulerOutput
        return SchedulerOutput(prev_sample=prev_sample)

    scheduler.step = stochastic_step
    return scheduler


def configure_flowmatch_scheduler(
    pipe,
    sampler_name: Optional[str] = "euler",
    noise: float = 0.0,
    seed: int = 42,
    sdnq_src: Optional[str] = None,
) -> str:
    """Configure `pipe.scheduler` according to user-selected `sampler_name` and `noise` (`eta`).

    Returns a human-readable summary of the active scheduler configuration.
    """
    from diffusers import FlowMatchEulerDiscreteScheduler, FlowMatchHeunDiscreteScheduler

    sampler_key = (sampler_name or "euler").strip().lower()
    noise_val = float(noise or 0.0)

    # Auto-promote deterministic euler to euler_a when user moves Noise slider > 0
    if sampler_key == "euler" and noise_val > 0.0:
        sampler_key = "euler_a"

    base_cfg = dict(pipe.scheduler.config)
    DPMCls, UniPCCls = _try_import_sdnext_schedulers(sdnq_src)

    if sampler_key in ("dpm++2m", "dpm2") and DPMCls is not None:
        pipe.scheduler = DPMCls(
            num_train_timesteps=base_cfg.get("num_train_timesteps", 1000),
            shift=base_cfg.get("shift", 1.0),
            use_dynamic_shifting=base_cfg.get("use_dynamic_shifting", False),
            base_shift=base_cfg.get("base_shift", 0.5),
            max_shift=base_cfg.get("max_shift", 1.15),
            base_image_seq_len=base_cfg.get("base_image_seq_len", 256),
            max_image_seq_len=base_cfg.get("max_image_seq_len", 4096),
            solver_order=2,
            algorithm_type="dpmsolver++2M",
            solver_type="midpoint",
            use_beta_sigmas=False,
        )
        pipe.scheduler.use_beta_sigmas = False
        pipe.scheduler.solver_type = "midpoint"
        if not hasattr(pipe.scheduler, "set_begin_index"):
            pipe.scheduler.set_begin_index = lambda idx=0: setattr(pipe.scheduler, "_begin_index", idx)
        if noise_val > 0.0:
            _wrap_stochastic_euler_step(pipe.scheduler, noise_val, seed)
        return f"FlowMatchDPMSolverMultistepScheduler (DPM++ 2M, noise={noise_val:.2f})"

    if sampler_key in ("dpm++2m_sde", "dpm2_a") and DPMCls is not None:
        s_noise = noise_val if noise_val > 0.0 else 0.5
        pipe.scheduler = DPMCls(
            num_train_timesteps=base_cfg.get("num_train_timesteps", 1000),
            shift=base_cfg.get("shift", 1.0),
            use_dynamic_shifting=base_cfg.get("use_dynamic_shifting", False),
            base_shift=base_cfg.get("base_shift", 0.5),
            max_shift=base_cfg.get("max_shift", 1.15),
            base_image_seq_len=base_cfg.get("base_image_seq_len", 256),
            max_image_seq_len=base_cfg.get("max_image_seq_len", 4096),
            solver_order=2,
            algorithm_type="dpmsolver++2M",
            solver_type="midpoint",
            use_beta_sigmas=False,
        )
        pipe.scheduler.use_beta_sigmas = False
        pipe.scheduler.solver_type = "midpoint"
        if not hasattr(pipe.scheduler, "set_begin_index"):
            pipe.scheduler.set_begin_index = lambda idx=0: setattr(pipe.scheduler, "_begin_index", idx)
        _wrap_stochastic_euler_step(pipe.scheduler, s_noise, seed)
        return f"FlowMatchDPMSolverMultistepScheduler (DPM++ 2M SDE, eta={s_noise:.2f})"

    if sampler_key == "unipc" and UniPCCls is not None:
        pipe.scheduler = UniPCCls(
            num_train_timesteps=base_cfg.get("num_train_timesteps", 1000),
            shift=base_cfg.get("shift", 1.0),
            use_dynamic_shifting=base_cfg.get("use_dynamic_shifting", False),
            base_image_seq_len=base_cfg.get("base_image_seq_len", 256),
            max_image_seq_len=base_cfg.get("max_image_seq_len", 4096),
            solver_order=2,
            solver_type="bh2",
        )
        _patch_unipc_flowmatch_scheduler(pipe.scheduler)
        if not hasattr(pipe.scheduler, "set_begin_index"):
            pipe.scheduler.set_begin_index = lambda idx=0: setattr(pipe.scheduler, "_begin_index", idx)
        if noise_val > 0.0:
            _wrap_stochastic_euler_step(pipe.scheduler, noise_val, seed)
        return f"FlowUniPCMultistepScheduler (UniPC bh2, noise={noise_val:.2f})"

    if sampler_key == "heun":
        heun_cfg = {
            k: v for k, v in base_cfg.items()
            if k in ("num_train_timesteps", "shift", "reverse", " solver_order", "prediction_type")
        }
        heun = FlowMatchHeunDiscreteScheduler.from_config(heun_cfg)
        pipe.scheduler = _patch_heun_flowmatch_scheduler(heun, base_cfg=base_cfg)
        return "FlowMatchHeunDiscreteScheduler (2nd-order predictor-corrector)"

    if sampler_key in ("euler_a", "lcm", "dpm++2m_sde", "dpm2_a"):
        eff_eta = noise_val if noise_val > 0.0 else 0.35
        euler = FlowMatchEulerDiscreteScheduler.from_config(base_cfg, stochastic_sampling=False)
        pipe.scheduler = _wrap_stochastic_euler_step(euler, eff_eta, seed)
        return f"FlowMatchEulerAncestral ({sampler_key}, eta={eff_eta:.2f})"

    # Default deterministic FlowMatch Euler
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(base_cfg, stochastic_sampling=False)
    return "FlowMatchEulerDiscreteScheduler (deterministic)"


def prepare_qwen21_img2img_latents(
    pipe,
    cond_image,
    width: int,
    height: int,
    full_sigmas: list[float],
    strength: float,
    seed: int,
    dtype,
):
    """Prepare truncated `sigmas` and initial noisy `latents` for QwenImage21Pipeline when `strength < 1.0`.

    Ported from huggingface/diffusers `QwenImageImg2ImgPipeline` & `QwenImage21Pipeline`:
    - Encodes `cond_image` resized to `(width, height)` through `pipe.vae`
    - Packs into 2x2 patchified sequence `(1, (H//16)*(W//16), 64)`
    - Slices `full_sigmas` to the last `max(1, round(len(full_sigmas) * strength))` steps
    - Mixes `initial_latents = (1 - sigma_start) * image_latents + sigma_start * rand_noise`
    """
    import torch
    from diffusers.utils.torch_utils import randn_tensor

    strength = float(max(0.05, min(1.0, strength)))
    total_steps = len(full_sigmas)
    num_Verify_steps = max(1, int(round(total_steps * strength)))
    init_step = max(0, total_steps - num_Verify_steps)
    sigmas_slice = list(full_sigmas[init_step:])
    start_sigma = float(sigmas_slice[0])

    multiple_of = pipe.vae_scale_factor * 2
    target_w = (width // multiple_of) * multiple_of
    target_h = (height // multiple_of) * multiple_of

    resized_img = cond_image.resize((target_w, target_h))
    img_tensor = pipe.image_processor.preprocess(resized_img, height=target_h, width=target_w)
    img_tensor = img_tensor.unsqueeze(2)  # (B, C, 1, H, W) for 3D Qwen VAE

    vae_device = next(pipe.vae.parameters()).device
    vae_dtype = next(pipe.vae.parameters()).dtype
    needs_offload_back = str(vae_device) == "cpu" and torch.cuda.is_available()

    if needs_offload_back:
        pipe.vae.to("cuda")
        vae_device = torch.device("cuda")

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    with torch.no_grad():
        encoded = pipe._encode_vae_image(img_tensor.to(device=vae_device, dtype=vae_dtype), gen)

    if needs_offload_back:
        pipe.vae.to("cpu")
        torch.cuda.empty_cache()

    num_channels_latents = pipe.transformer.config.in_channels
    lat_h, lat_w = encoded.shape[3], encoded.shape[4]
    packed_image_latents = pipe._pack_latents(encoded.to(device="cpu", dtype=dtype), 1, num_channels_latents, lat_h, lat_w)

    noise_shape = (1, 1, num_channels_latents, lat_h, lat_w)
    noise = randn_tensor(noise_shape, generator=gen, device=torch.device("cpu"), dtype=dtype)
    packed_noise = pipe._pack_latents(noise, 1, num_channels_latents, lat_h, lat_w)

    initial_latents = (1.0 - start_sigma) * packed_image_latents + start_sigma * packed_noise
    return initial_latents.to(dtype=dtype), sigmas_slice
