"""
Manga Screen AI + Manga Finish Pipeline for LazySD-UI.

Implements modular manga conversion combining:
  - Stage A: AI Edit (optional semantic redraw via existing Img2Img / Reference engines + compatible LoRA)
  - Stage B: Manga Levels (tonal normalization & optional descreen, reusing backend/halftone.py)
  - Stage C: Line Extraction (structural stroke & contour extraction on CPU)
  - Stage D: Manga Finish (4-zone tonal segmentation, halo-free normalized convolution,
             gradation screentone via backend/halftone.py, crisp pen overlay, white line cutouts in solid black)
  - Stage E: Manga Panel (optional aspect framing, border, margin, paper tint, and printed ink texture)

Algorithmic core adapted from ComfyUI-MangaScreen (https://github.com/ashukae/ComfyUI-MangaScreen):
  MIT License — Copyright (c) 2026 ashukae
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software...
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np
from PIL import Image

from .halftone import (
    ALLOWED_SHAPES,
    apply_levels_and_tone,
    compute_perceptual_luminance,
    decode_mask_to_float,
    extract_rgb_alpha,
    load_image_safe,
    render_halftone_screen,
)

MANGA_MODES = ("manga_lite", "finish_only", "ai_manga", "ai_manga_finish")
FINISH_PRESETS: dict[str, dict[str, Any]] = {
    "clean": {
        "screen_lines": 200,
        "white_point": 0.80,
        "black_point": 0.30,
        "light_tone": 0.18,
        "dark_tone": 0.42,
        "gradation": 1.00,
        "pen_threshold": 0.30,
        "white_lines": 0.40,
    },
    "shonen": {
        "screen_lines": 170,
        "white_point": 0.78,
        "black_point": 0.34,
        "light_tone": 0.22,
        "dark_tone": 0.48,
        "gradation": 0.85,
        "pen_threshold": 0.26,
        "white_lines": 0.50,
    },
    "seinen": {
        "screen_lines": 220,
        "white_point": 0.84,
        "black_point": 0.26,
        "light_tone": 0.16,
        "dark_tone": 0.40,
        "gradation": 1.00,
        "pen_threshold": 0.28,
        "white_lines": 0.35,
    },
    "shojo": {
        "screen_lines": 240,
        "white_point": 0.86,
        "black_point": 0.22,
        "light_tone": 0.12,
        "dark_tone": 0.32,
        "gradation": 1.00,
        "pen_threshold": 0.34,
        "white_lines": 0.30,
    },
}
PANEL_FRAMES: dict[str, Optional[float]] = {
    "original": None,
    "4:3": 4.0 / 3.0,
    "4:3 wide": 4.0 / 3.0,
    "16:9": 16.0 / 9.0,
    "16:9 wide": 16.0 / 9.0,
    "3:4": 3.0 / 4.0,
    "3:4 tall": 3.0 / 4.0,
    "1:1": 1.0,
}
PAPER_TINTS: dict[str, tuple[tuple[int, int, int], float]] = {
    "white": ((255, 255, 255), 0.0),
    "off-white": ((251, 250, 247), 1.5),
    "cream": ((246, 243, 236), 3.0),
}
INK_RGB = np.float32((26.0, 25.0, 28.0))

DEFAULT_MANGA_AI_PROMPT_RU = (
    "Преобразуй исходную иллюстрацию в чистую чёрно-белую манга-панель. "
    "Сохрани композицию, позу, персонажей, выражения лиц, одежду и расположение объектов. "
    "Преврати контуры в чёткий чёрный lineart. Светлые участки кожи и белой одежды оставь белой бумагой. "
    "Глубокие тени и действительно тёмные предметы передай сплошным чёрным. "
    "Используй сдержанные серые тональные области как основу для последующего скринтона. "
    "Не добавляй текст, надписи, водяные знаки или новые объекты."
)

DEFAULT_MANGA_AI_PROMPT_EN = (
    "Redraw <image1> as a printed black-and-white manga page panel, as if inked by the original artist "
    "for serialization. Keep the exact composition, pose, character design, face, expression and drawing "
    "style of <image1>. Use confident ink lines with varied line weight, solid black for dark hair and deep "
    "shadows, white paper for skin and light surfaces, ink hatching lines for shadows on cloth, and regular "
    "screentone only where a mid tone is needed. No color and no smooth gradients. Do not add anything that "
    "is not in <image1>."
)


@dataclass
class MangaPipelineParams:
    mode: str = "manga_lite"

    # Stage A: AI Edit (optional)
    ai_enabled: bool = False
    model_id: str = ""
    variant_id: str = ""
    lora_name: str = ""
    lora_strength: float = 0.85
    ai_prompt: str = DEFAULT_MANGA_AI_PROMPT_EN
    ai_negative_prompt: str = "color, blurry, photo, 3d render, watermark, muddy shading"
    ai_denoise_strength: float = 0.65
    ai_steps: int = 0
    ai_cfg: float = 0.0
    ai_ref_mode: str = "control_ref"
    seed: int = 42

    # Stage B: Manga Levels
    levels_enabled: bool = True
    levels_black: float = 0.20
    levels_white: float = 0.70
    levels_gamma: float = 1.30
    levels_contrast: float = 1.00
    levels_brightness: float = 0.0
    levels_descreen: float = 0.0

    # Stage C: Line Extraction
    lines_enabled: bool = True
    lines_strength: float = 1.00
    lines_threshold: float = 0.30
    lines_thickness: int = 1
    lines_invert: bool = False

    # Stage D: Manga Finish
    finish_enabled: bool = True
    finish_preset: str = "clean"
    finish_dot_size: int = 6
    screen_lines: int = 200
    white_point: float = 0.80
    black_point: float = 0.30
    light_tone: float = 0.18
    dark_tone: float = 0.42
    gradation: float = 1.00
    pen_threshold: float = 0.30
    white_lines: float = 0.40
    dot_shape: str = "circle"
    screen_angle: float = 45.0

    # Stage E: Manga Panel
    panel_enabled: bool = False
    panel_frame: str = "original"
    panel_crop_x: float = 0.50
    panel_crop_y: float = 0.40
    panel_border: int = 5
    panel_margin: int = 60
    panel_paper: str = "white"
    panel_ink: float = 0.50

    def validate(self) -> "MangaPipelineParams":
        mode = str(self.mode or "manga_lite").strip().lower()
        if mode not in MANGA_MODES:
            raise ValueError(f"Unsupported manga mode: {mode!r}. Allowed: {', '.join(MANGA_MODES)}")
        self.mode = mode

        self.ai_enabled = bool(self.ai_enabled or mode in ("ai_manga", "ai_manga_finish"))
        self.model_id = str(self.model_id or "").strip()
        self.variant_id = str(self.variant_id or "").strip()
        self.lora_name = str(self.lora_name or "").strip()
        self.lora_strength = max(-2.0, min(2.0, float(self.lora_strength)))
        self.ai_prompt = str(self.ai_prompt or DEFAULT_MANGA_AI_PROMPT_EN).strip()
        self.ai_negative_prompt = str(self.ai_negative_prompt or "").strip()
        self.ai_denoise_strength = max(0.05, min(1.0, float(self.ai_denoise_strength)))
        self.ai_steps = max(0, min(100, int(self.ai_steps)))
        self.ai_cfg = max(0.0, min(20.0, float(self.ai_cfg)))
        self.ai_ref_mode = str(self.ai_ref_mode or "control_ref").strip().lower()
        self.seed = int(self.seed) & 0x7FFFFFFF

        self.levels_enabled = bool(self.levels_enabled)
        self.levels_black = max(0.0, min(0.90, float(self.levels_black)))
        self.levels_white = max(0.10, min(1.0, float(self.levels_white)))
        if self.levels_white <= self.levels_black + 0.02:
            self.levels_white = min(1.0, self.levels_black + 0.05)
        self.levels_gamma = max(0.3, min(3.0, float(self.levels_gamma)))
        self.levels_contrast = max(0.2, min(3.0, float(self.levels_contrast)))
        self.levels_brightness = max(-1.0, min(1.0, float(self.levels_brightness)))
        self.levels_descreen = max(0.0, min(1.0, float(self.levels_descreen)))

        self.lines_enabled = bool(self.lines_enabled)
        self.lines_strength = max(0.0, min(2.0, float(self.lines_strength)))
        self.lines_threshold = max(0.02, min(0.95, float(self.lines_threshold)))
        self.lines_thickness = max(0, min(5, int(self.lines_thickness)))
        self.lines_invert = bool(self.lines_invert)

        self.finish_enabled = bool(self.finish_enabled)
        self.finish_preset = str(self.finish_preset or "clean").strip().lower()
        self.finish_dot_size = max(2, min(32, int(self.finish_dot_size)))
        self.screen_lines = max(40, min(600, int(self.screen_lines)))
        self.white_point = max(0.30, min(1.0, float(self.white_point)))
        self.black_point = max(0.0, min(0.75, float(self.black_point)))
        if self.white_point <= self.black_point + 0.05:
            self.white_point = min(1.0, self.black_point + 0.08)
        self.light_tone = max(0.02, min(0.90, float(self.light_tone)))
        self.dark_tone = max(0.02, min(0.95, float(self.dark_tone)))
        self.gradation = max(0.0, min(1.0, float(self.gradation)))
        self.pen_threshold = max(0.05, min(0.95, float(self.pen_threshold)))
        self.white_lines = max(0.0, min(1.0, float(self.white_lines)))

        shape = str(self.dot_shape or "circle").strip().lower()
        if shape not in ALLOWED_SHAPES:
            raise ValueError(f"Unsupported dot_shape: {shape!r}")
        self.dot_shape = shape
        self.screen_angle = max(-180.0, min(180.0, float(self.screen_angle)))

        self.panel_enabled = bool(self.panel_enabled)
        frame = str(self.panel_frame or "original").strip().lower()
        if frame not in PANEL_FRAMES:
            raise ValueError(f"Unsupported panel_frame: {frame!r}. Allowed: {', '.join(PANEL_FRAMES.keys())}")
        self.panel_frame = frame
        self.panel_crop_x = max(0.0, min(1.0, float(self.panel_crop_x)))
        self.panel_crop_y = max(0.0, min(1.0, float(self.panel_crop_y)))
        self.panel_border = max(0, min(40, int(self.panel_border)))
        self.panel_margin = max(0, min(400, int(self.panel_margin)))

        paper = str(self.panel_paper or "white").strip().lower()
        if paper not in PAPER_TINTS:
            raise ValueError(f"Unsupported panel_paper: {paper!r}. Allowed: {', '.join(PAPER_TINTS.keys())}")
        self.panel_paper = paper
        self.panel_ink = max(0.0, min(1.0, float(self.panel_ink)))

        return self

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]] = None) -> "MangaPipelineParams":
        d = dict(data or {})
        base: dict[str, Any] = {}

        preset = str(d.get("finish_preset") or "clean").strip().lower()
        if preset in FINISH_PRESETS:
            base.update(FINISH_PRESETS[preset])

        # Map UI aliases to canonical MangaPipelineParams fields
        alias_map = {
            "line_strength": "lines_strength",
            "line_threshold": "lines_threshold",
            "line_thickness": "lines_thickness",
            "line_invert": "lines_invert",
            "finish_dot_shape": "dot_shape",
            "finish_angle": "screen_angle",
            "finish_ink_threshold": "pen_threshold",
            "finish_light_tone": "white_point",
            "finish_mid_tone": "dark_tone",
            "finish_black_point": "black_point",
            "finish_gradation": "gradation",
            "finish_white_cut": "white_lines",
            "panel_aspect": "panel_frame",
            "panel_paper_tone": "panel_paper",
            "panel_print_bleed": "panel_ink",
        }
        for ui_key, canonical_key in alias_map.items():
            if ui_key in d and d[ui_key] is not None and canonical_key not in d:
                base[canonical_key] = d[ui_key]

        if "finish_dot_size" in d and d["finish_dot_size"] is not None:
            dot_sz = max(2, min(32, int(d["finish_dot_size"])))
            base["finish_dot_size"] = dot_sz
            if "screen_lines" not in d:
                base["screen_lines"] = max(40, min(600, int(round(1024.0 / dot_sz))))

        for k, v in d.items():
            if v is not None and k in cls.__dataclass_fields__:
                base[k] = v

        return cls(**base).validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ensure_rgb_array(source: Union[np.ndarray, Image.Image, str, Path, bytes]) -> np.ndarray:
    if isinstance(source, np.ndarray):
        if source.ndim == 2:
            return np.stack([source, source, source], axis=2).astype(np.uint8)
        return source[:, :, :3].astype(np.uint8)
    rgb, _ = extract_rgb_alpha(source)
    return rgb


# ---------------------------------------------------------------------------
# Stage B: Manga Levels (Reuses backend/halftone.py apply_levels_and_tone)
# ---------------------------------------------------------------------------
def apply_manga_levels(
    rgb: Union[np.ndarray, Image.Image],
    black: Union[float, MangaPipelineParams] = 0.20,
    white: float = 0.70,
    gamma: float = 1.30,
    contrast: float = 1.00,
    descreen: float = 0.0,
) -> np.ndarray:
    """Convert RGB uint8 [H, W, 3] to normalized grayscale uint8 [H, W, 3] with levels, gamma & descreen."""
    rgb_arr = _ensure_rgb_array(rgb)
    if isinstance(black, MangaPipelineParams):
        p = black
        black_val = p.levels_black
        white_val = p.levels_white
        gamma_val = p.levels_gamma
        contrast_val = p.levels_contrast
        descreen_val = p.levels_descreen
    else:
        black_val = float(black)
        white_val = float(white)
        gamma_val = float(gamma)
        contrast_val = float(contrast)
        descreen_val = float(descreen)

    lum = compute_perceptual_luminance(rgb_arr)
    if descreen_val > 0.02:
        k = max(0.5, lum.shape[1] / 1024.0)
        d = int(round((5 + 8 * float(descreen_val)) * k)) | 1
        lum = cv2.bilateralFilter(lum, d, 0.08 * float(descreen_val) + 0.02, 6.0 * k)

    norm = apply_levels_and_tone(
        lum,
        black_point=black_val,
        white_point=white_val,
        gamma=gamma_val,
        contrast=contrast_val,
        brightness=0.0,
        invert=False,
        pre_blur=0.0,
    )
    g8 = np.clip(np.round(norm * 255.0), 0, 255).astype(np.uint8)
    return np.stack([g8, g8, g8], axis=2)


# ---------------------------------------------------------------------------
# Stage C: Structural Line Extraction (CPU, OpenCV)
# ---------------------------------------------------------------------------
def extract_manga_lines(
    rgb: Union[np.ndarray, Image.Image],
    strength: Union[float, MangaPipelineParams] = 1.0,
    threshold: float = 0.30,
    thickness: int = 1,
    invert: bool = False,
) -> np.ndarray:
    """Extract structural contour lines as a float32 line-strength map [H, W] in [0..1].

    Combines difference-of-Gaussians local stroke detection with connected-component
    shape filtering (preserving elongated strokes while rejecting compact noise/dots)
    and multi-scale gradient ridges.
    """
    rgb_arr = _ensure_rgb_array(rgb)
    if isinstance(strength, MangaPipelineParams):
        p = strength
        strength_val = p.lines_strength
        threshold = p.lines_threshold
        thickness = p.lines_thickness
        invert = p.lines_invert
    else:
        strength_val = float(strength)

    lum = compute_perceptual_luminance(rgb_arr)
    h, w = lum.shape[:2]
    k = max(0.5, w / 1024.0)

    # 1. Local adaptive valley response (dark strokes against local neighborhood)
    lb_fine = cv2.GaussianBlur(lum, (0, 0), 0.65 * k)
    win = int(15 * k) | 1
    local_bg = cv2.blur(lb_fine, (win, win))
    valley = np.clip((local_bg - lb_fine - 0.025) / 0.14, 0.0, 1.0)
    valley = np.where(lb_fine < 0.78, valley, 0.0)

    # 2. Structural stroke filter via connected components (long/thin vs round dots)
    cand = (valley > max(0.08, threshold * 0.65)) & (lb_fine < 0.70)
    _, lab, st, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    cw = st[:, cv2.CC_STAT_WIDTH]
    ch = st[:, cv2.CC_STAT_HEIGHT]
    area = st[:, cv2.CC_STAT_AREA]
    keep = (np.maximum(cw, ch) >= max(5, 8 * k)) & (area >= max(5, 8 * k * k)) & (
        area < 0.68 * cw * ch + 45 * k * k
    )
    keep |= area >= 320 * k * k
    keep[0] = False
    struct_mask = keep[lab].astype(np.float32)

    line_map = np.clip(valley * (0.35 + 0.65 * struct_mask) * float(strength_val), 0.0, 1.0)

    # 3. Optional thickness adjustment
    if thickness > 1:
        kern_sz = int(thickness)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kern_sz, kern_sz))
        line_map = cv2.dilate(line_map, kern)
    elif thickness == 0:
        kern = np.ones((2, 2), np.uint8)
        line_map = cv2.erode(line_map, kern)

    if invert:
        line_map = 1.0 - line_map

    return line_map.astype(np.float32)


# ---------------------------------------------------------------------------
# Stage D: Manga Finish (ashukae/ComfyUI-MangaScreen + shared halftone.py)
# ---------------------------------------------------------------------------
def _ink_layer(L: np.ndarray, k: float) -> np.ndarray:
    """Lines and solid blacks as a crisp boolean mask. k is image scale relative to 1024 px width."""
    solid = cv2.GaussianBlur(L, (0, 0), 1.5 * k) < 0.22
    Lb = cv2.GaussianBlur(L, (0, 0), 0.7 * k)
    blur_sz = max(3, int(15 * k) | 1)
    line = (Lb < cv2.blur(Lb, (blur_sz, blur_sz)) - 0.07) & (Lb < 0.65)
    _, lab, st, _ = cv2.connectedComponentsWithStats(line.astype(np.uint8), connectivity=8)
    w = st[:, cv2.CC_STAT_WIDTH]
    h = st[:, cv2.CC_STAT_HEIGHT]
    area = st[:, cv2.CC_STAT_AREA]
    keep = (np.maximum(w, h) >= max(4, 10 * k)) & (area >= max(4, 12 * k * k)) & (
        area < 0.6 * w * h + 40 * k * k
    )
    keep |= area >= 400 * k * k
    keep[0] = False
    ink = solid | keep[lab]
    _, lab2, st2, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    big = st2[:, cv2.CC_STAT_AREA] >= max(2, 6 * k * k)
    big[0] = False
    return big[lab2]


def _value_map(L: np.ndarray, ink: np.ndarray, k: float) -> np.ndarray:
    """Descreened value with ink masked out (normalized convolution), then edge-preserving flattening."""
    m = (~ink).astype(np.float32)
    s = max(0.8, 2.5 * k)
    num = cv2.GaussianBlur(L * m, (0, 0), s)
    den = np.maximum(cv2.GaussianBlur(m, (0, 0), s), 1e-3)
    V = (num / den).astype(np.float32)
    d = max(3, int(9 * k) | 1)
    for _ in range(2):
        V = cv2.bilateralFilter(V, d, 0.10, max(2.0, 9.0 * k))
    return V


def _tone_plan(V: np.ndarray, k: float, white: float, split: float, black: float, min_region: int = 600) -> np.ndarray:
    """0 = paper white, 1 = light tone, 2 = dark tone, 3 = solid black.
    Small regions under min_region px (at 1024 px width) melt into their surroundings.
    """
    med_k1 = max(3, int(11 * k) | 1)
    level = cv2.medianBlur(np.digitize(-V, [-white, -split, -black]).astype(np.uint8), med_k1)
    med_k2 = max(3, int(21 * k) | 1)
    thresh_area = max(4.0, float(min_region) * k * k)
    for lv in (1, 2, 3, 0):
        _, lab, st, _ = cv2.connectedComponentsWithStats((level == lv).astype(np.uint8), connectivity=4)
        small = st[:, cv2.CC_STAT_AREA] < thresh_area
        small[0] = False
        hole = small[lab]
        if hole.any():
            fill = cv2.medianBlur(np.where(hole, 255, level).astype(np.uint8), med_k2)
            level = np.where(hole & (fill != 255), fill, level).astype(np.uint8)
    return level


def _region_spread(level: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Standard deviation of V inside each flat region of the plan (detects hidden smooth gradients)."""
    out = np.zeros_like(V, dtype=np.float32)
    v_flat = V.ravel()
    v2_flat = (V * V).ravel()
    for lv in range(4):
        n, lab = cv2.connectedComponents((level == lv).astype(np.uint8), connectivity=4)
        lab_flat = lab.ravel()
        cnt = np.maximum(np.bincount(lab_flat, minlength=n), 1)
        mean = np.bincount(lab_flat, v_flat, minlength=n) / cnt
        std = np.sqrt(np.maximum(np.bincount(lab_flat, v2_flat, minlength=n) / cnt - mean * mean, 0.0))
        out = np.where(level == lv, std[lab], out)
    return out.astype(np.float32)


def _white_cut(candidates: np.ndarray, k: float) -> np.ndarray:
    """Long strokes worth keeping in white inside solid black (beta), thinned so folds/seams remain crisp."""
    _, lab, st, _ = cv2.connectedComponentsWithStats(candidates.astype(np.uint8), connectivity=8)
    keep = np.maximum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT]) >= max(8, 40 * k)
    keep[0] = False
    erode_sz = 2 if k < 1.6 else 3
    return cv2.erode(keep[lab].astype(np.uint8), np.ones((erode_sz, erode_sz), np.uint8)) > 0


def apply_manga_finish(
    rgb: Union[np.ndarray, Image.Image],
    screen_lines: Union[int, np.ndarray, MangaPipelineParams] = 200,
    white_point: Union[float, MangaPipelineParams] = 0.80,
    black_point: float = 0.30,
    light_tone: float = 0.18,
    dark_tone: float = 0.42,
    gradation: float = 1.0,
    pen_map: Optional[np.ndarray] = None,
    pen_threshold: float = 0.30,
    white_lines: float = 0.40,
    dot_shape: str = "circle",
    screen_angle: float = 45.0,
) -> np.ndarray:
    """Run Manga Finish on an RGB uint8 image [H, W, 3] and return finished RGB uint8 [H, W, 3].

    Uses the shared `render_halftone_screen` from `backend/halftone.py` for the screentone dots.
    """
    rgb_arr = _ensure_rgb_array(rgb)

    if isinstance(screen_lines, np.ndarray):
        pen_map = screen_lines
        if isinstance(white_point, MangaPipelineParams):
            p = white_point
            screen_lines_val = p.screen_lines
            white_point_val = p.white_point
            black_point = p.black_point
            light_tone = p.light_tone
            dark_tone = p.dark_tone
            gradation = p.gradation
            pen_threshold = p.pen_threshold
            white_lines = p.white_lines
            dot_shape = p.dot_shape
            screen_angle = p.screen_angle
        else:
            screen_lines_val = 200
            white_point_val = float(white_point)
    elif isinstance(screen_lines, MangaPipelineParams):
        p = screen_lines
        screen_lines_val = p.screen_lines
        white_point_val = p.white_point
        black_point = p.black_point
        light_tone = p.light_tone
        dark_tone = p.dark_tone
        gradation = p.gradation
        pen_threshold = p.pen_threshold
        white_lines = p.white_lines
        dot_shape = p.dot_shape
        screen_angle = p.screen_angle
    else:
        screen_lines_val = int(screen_lines)
        white_point_val = float(white_point)

    L = compute_perceptual_luminance(rgb_arr)
    h, w = L.shape[:2]
    k = max(0.35, w / 1024.0)

    ink = _ink_layer(L, k)
    solid_dark = cv2.GaussianBlur(L, (0, 0), max(0.6, 1.5 * k)) < 0.22
    strokes = ink & ~solid_dark

    if pen_map is not None:
        if pen_map.shape[:2] != (h, w):
            pen_map = cv2.resize(pen_map.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
        strokes = strokes | (pen_map > float(pen_threshold))

    V = _value_map(L, ink, k)
    split_pt = (float(white_point_val) + float(black_point)) * 0.5
    level = _tone_plan(V, k, float(white_point_val), split_pt, float(black_point))

    palette = np.float32((0.0, float(light_tone), float(dark_tone), 1.0))
    cover = np.take(palette, level)

    if gradation > 1e-3:
        Vs = cv2.GaussianBlur(V, (0, 0), max(0.8, 3.0 * k))
        spread = _region_spread(level, Vs)
        a = cv2.GaussianBlur(
            np.clip((spread - 0.03) / 0.05, 0.0, 1.0) * float(gradation),
            (0, 0),
            max(1.0, 4.0 * k),
        )
        span = max(float(white_point_val) - float(black_point), 1e-3)
        ramp = np.clip((float(white_point_val) - Vs) / span, 0.0, 1.0)
        cover = cover * (1.0 - a) + ramp * a

    pitch = max(2.5, float(w) / max(40, int(screen_lines_val)))
    screen_dots = (
        render_halftone_screen(
            cover,
            cell_size=pitch,
            angle=float(screen_angle),
            dot_shape=dot_shape,
            max_radius_ratio=1.0,
            antialias=3,
            dither_type="ordered",
            dither_strength=0.0,
            binary_threshold=True,
        )
        > 0.5
    )

    page_ink = screen_dots | ink | strokes

    if white_lines > 1e-3:
        erode_k = max(1, int(3 * k) | 1)
        beta = cv2.erode((level == 3).astype(np.uint8), np.ones((erode_k, erode_k), np.uint8)) > 0
        candidates = strokes if pen_map is None else (pen_map > (1.0 - float(white_lines)))
        page_ink = (page_ink | beta) & ~(beta & _white_cut(candidates, k))

    # ~page_ink: True = white paper (255), False = black ink (0)
    out_u8 = np.where(page_ink, 0, 255).astype(np.uint8)
    return np.stack([out_u8, out_u8, out_u8], axis=2)


# ---------------------------------------------------------------------------
# Stage E: Manga Panel Framing & Print Feel
# ---------------------------------------------------------------------------
def _multiscale_noise(h: int, w: int, scales: list[tuple[int, float]], seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    out = np.zeros((h, w), np.float32)
    for s, amp in scales:
        step = max(1, int(s))
        nh = max(2, h // step)
        nw = max(2, w // step)
        n = rng.standard_normal((nh, nw)).astype(np.float32)
        out += float(amp) * cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
    return out


def apply_manga_panel(
    rgb: np.ndarray,
    frame: str = "original",
    crop_x: float = 0.5,
    crop_y: float = 0.4,
    border: int = 5,
    margin: int = 60,
    paper: str = "white",
    ink: float = 0.5,
) -> np.ndarray:
    """Frame a black-and-white manga page with aspect crop, ink border, paper margin, and ink texture."""
    lum = compute_perceptual_luminance(rgb)
    ink_mask = lum < 0.5
    h, w = ink_mask.shape[:2]
    k = max(0.35, w / 1024.0)

    ratio = PANEL_FRAMES.get(str(frame or "original").strip().lower())
    if ratio is None:
        cw, ch = w, h
    else:
        if (w / float(h)) <= ratio:
            cw, ch = w, int(round(w / ratio))
        else:
            cw, ch = int(round(h * ratio)), h
    cw, ch = min(cw, w), min(ch, h)
    x0 = int(round((w - cw) * float(crop_x)))
    y0 = int(round((h - ch) * float(crop_y)))
    p = ink_mask[y0 : y0 + ch, x0 : x0 + cw].copy()

    b_px = max(0, int(round(border * k)))
    if b_px > 0 and b_px * 2 < min(ch, cw):
        p[:b_px, :] = True
        p[-b_px:, :] = True
        p[:, :b_px] = True
        p[:, -b_px:] = True

    m_px = max(0, int(round(margin * k)))
    if m_px > 0:
        p = np.pad(p, m_px, mode="constant", constant_values=False)

    ph, pw = p.shape[:2]
    a = p.astype(np.float32)
    ink_strength = float(ink)
    if ink_strength > 1e-3:
        a = np.clip((cv2.GaussianBlur(a, (0, 0), max(0.3, 0.6 * k * ink_strength)) - 0.35) / 0.4, 0.0, 1.0)
        mottle = _multiscale_noise(ph, pw, [(max(2, int(64 * k)), 1.0), (max(2, int(16 * k)), 0.5)], seed=1)
        a *= 1.0 - 0.05 * ink_strength * (0.5 + 0.5 * np.tanh(mottle))

    paper_rgb, fibre = PAPER_TINTS.get(paper, PAPER_TINTS["white"])
    base = np.float32(paper_rgb)[None, None, :]
    if fibre > 1e-3:
        grain = _multiscale_noise(ph, pw, [(3, 1.0), (1, 0.6)], seed=2)
        base = base + float(fibre) * grain[:, :, None]

    ink_color = INK_RGB * ink_strength
    out_f = np.clip(base * (1.0 - a[:, :, None]) + ink_color[None, None, :] * a[:, :, None], 0.0, 255.0)
    return np.round(out_f).astype(np.uint8)


# ---------------------------------------------------------------------------
# Coordinator: Run Algorithmic Manga Pipeline (Manga Lite / Finish Only / Post-AI)
# ---------------------------------------------------------------------------
def process_manga_algorithmic(
    source: Union[str, Path, bytes, Image.Image, tuple[np.ndarray, Optional[np.ndarray]]],
    params: Union[MangaPipelineParams, dict[str, Any], None] = None,
    mask: Union[str, Path, bytes, Image.Image, np.ndarray, None] = None,
    max_side: Optional[int] = None,
    *,
    mask_source: Union[str, Path, bytes, Image.Image, np.ndarray, None] = None,
    preview_max_side: Optional[int] = None,
) -> Image.Image:
    """Execute the CPU stages of Manga Screen AI:
    `Source -> [Manga Levels] -> [Line Extraction] -> [Manga Finish] -> [Manga Panel] -> Result`

    Never mutates the input image or requires GPU/diffusion models.
    """
    if isinstance(mask, (int, float)) and max_side is None:
        max_side = int(mask)
        mask = None
    if mask is None and mask_source is not None:
        mask = mask_source
    if max_side is None and preview_max_side is not None:
        max_side = preview_max_side

    if isinstance(params, dict):
        cfg = MangaPipelineParams.from_dict(params)
    elif isinstance(params, MangaPipelineParams):
        cfg = params.validate()
    else:
        cfg = MangaPipelineParams().validate()

    rgb_orig, alpha_orig = extract_rgb_alpha(source)
    orig_h, orig_w = rgb_orig.shape[:2]

    if max_side and max(orig_h, orig_w) > int(max_side):
        scale = float(max_side) / float(max(orig_h, orig_w))
        work_w = max(16, int(round(orig_w * scale)))
        work_h = max(16, int(round(orig_h * scale)))
        rgb = cv2.resize(rgb_orig, (work_w, work_h), interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha_orig, (work_w, work_h), interpolation=cv2.INTER_AREA) if alpha_orig is not None else None
    else:
        rgb = rgb_orig.copy()
        alpha = alpha_orig.copy() if alpha_orig is not None else None

    # Mode-level defaults:
    # - In 'finish_only', Manga Levels is skipped unless explicitly requested in custom stages,
    #   because the input is already a stylized/B&W image.
    run_levels = cfg.levels_enabled and (cfg.mode != "finish_only")
    run_lines = cfg.lines_enabled
    run_finish = cfg.finish_enabled or (cfg.mode in ("manga_lite", "finish_only", "ai_manga_finish"))
    run_panel = cfg.panel_enabled

    work_rgb = rgb

    # Stage C (Line Extraction) is extracted from the pre-levels or post-levels image
    pen_map: Optional[np.ndarray] = None
    if run_lines:
        pen_map = extract_manga_lines(
            work_rgb,
            strength=cfg.lines_strength,
            threshold=cfg.lines_threshold,
            thickness=cfg.lines_thickness,
            invert=cfg.lines_invert,
        )

    # Stage B: Manga Levels
    if run_levels:
        work_rgb = apply_manga_levels(
            work_rgb,
            black=cfg.levels_black,
            white=cfg.levels_white,
            gamma=cfg.levels_gamma,
            contrast=cfg.levels_contrast,
            descreen=cfg.levels_descreen,
        )

    # Stage D: Manga Finish
    if run_finish:
        work_rgb = apply_manga_finish(
            work_rgb,
            screen_lines=cfg.screen_lines,
            white_point=cfg.white_point,
            black_point=cfg.black_point,
            light_tone=cfg.light_tone,
            dark_tone=cfg.dark_tone,
            gradation=cfg.gradation,
            pen_map=pen_map,
            pen_threshold=cfg.pen_threshold,
            white_lines=cfg.white_lines,
            dot_shape=cfg.dot_shape,
            screen_angle=cfg.screen_angle,
        )
    elif pen_map is not None:
        # If Manga Finish is disabled but Line Extraction is enabled, overlay extracted lines on levels
        stroke_mask = pen_map > cfg.pen_threshold
        work_rgb = np.where(stroke_mask[:, :, None], 0, work_rgb).astype(np.uint8)

    # Optional mask blending before panel framing
    if mask is not None and work_rgb.shape[:2] == rgb.shape[:2]:
        mask_f = decode_mask_to_float(mask, (rgb.shape[1], rgb.shape[0]), invert_mask=False)
        if mask_f is not None:
            m3 = mask_f[:, :, None]
            work_rgb = np.clip(
                np.round(rgb.astype(np.float32) * (1.0 - m3) + work_rgb.astype(np.float32) * m3),
                0,
                255,
            ).astype(np.uint8)

    # Stage E: Manga Panel
    if run_panel:
        work_rgb = apply_manga_panel(
            work_rgb,
            frame=cfg.panel_frame,
            crop_x=cfg.panel_crop_x,
            crop_y=cfg.panel_crop_y,
            border=cfg.panel_border,
            margin=cfg.panel_margin,
            paper=cfg.panel_paper,
            ink=cfg.panel_ink,
        )
        # Note: if panel framing changed dimensions (crop or margin), alpha is omitted because paper margin is opaque
        if work_rgb.shape[:2] != (rgb.shape[0], rgb.shape[1]):
            alpha = None

    if alpha is not None and work_rgb.shape[:2] == alpha.shape[:2]:
        return Image.fromarray(np.dstack([work_rgb, alpha]), mode="RGBA")
    return Image.fromarray(work_rgb, mode="RGB")
