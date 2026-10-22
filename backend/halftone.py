"""
Shared Halftone & Screentone Engine for LazySD-UI.

Provides CPU-only, vectorized (NumPy + OpenCV + Pillow) halftone rasterization,
tonal levels normalization, multi-channel CMYK color halftoning, mask blending,
and preset modes (Manga, Classic, Newspaper, Color, Custom).

Used by both:
  1. Halftone Studio (manual artistic halftone & screentone filter)
  2. Manga Screen AI / Manga Finish (screentone rendering & tonal normalization)

Does NOT load any diffusion models or allocate GPU VRAM.
"""

from __future__ import annotations

import base64
import io
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np
from PIL import Image, ImageOps

ALLOWED_MODES = ("manga", "classic", "newspaper", "color", "custom")
ALLOWED_SHAPES = (
    "circle",
    "square",
    "diamond",
    "line",
    "crosshatch",
    "hexagon",
    "cross",
    "ellipse",
    "star",
    "heart",
    "triangle",
    "concentric",
    "ring",
    "stipple",
    "stochastic",
)
ALLOWED_DITHER = ("none", "ordered", "bayer", "noise")
ALLOWED_GRAD_TYPES = ("none", "linear", "radial", "vignette")
ALLOWED_BLEND_MODES = ("normal", "multiply", "screen", "overlay", "darken", "lighten")


@dataclass
class HalftoneParams:
    mode: str = "manga"
    cell_size: int = 10
    max_radius_ratio: float = 0.95
    dot_shape: str = "circle"
    angle: float = 45.0
    contrast: float = 1.25
    gamma: float = 1.05
    brightness: float = 0.0
    black_threshold: float = 0.20
    white_threshold: float = 0.82
    intensity: float = 1.0
    dot_color: str = "#141418"
    bg_color: str = "#ffffff"
    antialias: int = 2
    dither_type: str = "none"
    dither_strength: float = 0.0
    seed: int = 42
    invert: bool = False
    pre_blur: float = 0.0
    preserve_lines: bool = True
    paper_texture: float = 0.0
    cmyk_angles: list[float] = field(default_factory=lambda: [15.0, 75.0, 0.0, 45.0])
    invert_mask: bool = False
    # --- GenTonik (GT) Extended Screentone Controls ---
    satellite_enabled: bool = False
    satellite_count: int = 4
    satellite_size_ratio: float = 0.45
    satellite_distance: float = 0.42
    satellite_angle: float = 0.0
    satellite_shape: str = "circle"
    satellite_stretch: float = 0.0
    merge_factor: float = 0.0
    aspect_x: float = 1.0
    aspect_y: float = 1.0
    row_offset: float = 0.0
    jitter_pos: float = 0.0
    jitter_size: float = 0.0
    grad_type: str = "none"
    grad_angle: float = 45.0
    grad_strength: float = 0.0

    def validate(self) -> "HalftoneParams":
        mode = str(self.mode or "manga").strip().lower()
        if mode not in ALLOWED_MODES:
            raise ValueError(f"Unsupported mode: {mode!r}. Allowed: {', '.join(ALLOWED_MODES)}")
        self.mode = mode

        shape = str(self.dot_shape or "circle").strip().lower()
        if shape not in ALLOWED_SHAPES:
            raise ValueError(f"Unsupported dot_shape: {shape!r}. Allowed: {', '.join(ALLOWED_SHAPES)}")
        self.dot_shape = shape

        dither = str(self.dither_type or "none").strip().lower()
        if dither not in ALLOWED_DITHER:
            raise ValueError(f"Unsupported dither_type: {dither!r}. Allowed: {', '.join(ALLOWED_DITHER)}")
        self.dither_type = dither

        if int(self.cell_size) < 1:
            raise ValueError(f"cell_size must be >= 1, got {self.cell_size}")
        self.cell_size = max(2, min(120, int(self.cell_size)))
        self.max_radius_ratio = max(0.1, min(1.5, float(self.max_radius_ratio)))
        self.angle = max(-180.0, min(180.0, float(self.angle)))
        self.contrast = max(0.1, min(4.0, float(self.contrast)))
        self.gamma = max(0.1, min(5.0, float(self.gamma)))
        self.brightness = max(-1.0, min(1.0, float(self.brightness)))
        raw_black = float(self.black_threshold)
        raw_white = float(self.white_threshold)
        if raw_black >= raw_white:
            raise ValueError(
                f"black_threshold ({raw_black}) must be strictly less than white_threshold ({raw_white})"
            )
        self.black_threshold = max(0.0, min(0.95, raw_black))
        self.white_threshold = max(0.05, min(1.0, raw_white))
        if self.white_threshold <= self.black_threshold + 0.02:
            self.white_threshold = min(1.0, self.black_threshold + 0.05)
        self.intensity = max(0.0, min(1.0, float(self.intensity)))
        self.antialias = max(1, min(4, int(self.antialias)))
        self.dither_strength = max(0.0, min(0.8, float(self.dither_strength)))
        self.seed = int(self.seed) & 0x7FFFFFFF
        self.invert = bool(self.invert)
        self.pre_blur = max(0.0, min(20.0, float(self.pre_blur)))
        self.preserve_lines = bool(self.preserve_lines)
        self.paper_texture = max(0.0, min(1.0, float(self.paper_texture)))
        self.invert_mask = bool(self.invert_mask)

        # Validate GenTonik parameters
        self.satellite_enabled = bool(self.satellite_enabled)
        self.satellite_count = max(1, min(8, int(self.satellite_count)))
        self.satellite_size_ratio = max(0.05, min(1.5, float(self.satellite_size_ratio)))
        self.satellite_distance = max(0.1, min(1.0, float(self.satellite_distance)))
        self.satellite_angle = max(-180.0, min(180.0, float(self.satellite_angle)))
        sat_shape = str(self.satellite_shape or "circle").strip().lower()
        self.satellite_shape = sat_shape if sat_shape in ALLOWED_SHAPES else "circle"
        self.satellite_stretch = max(0.0, min(3.0, float(self.satellite_stretch)))
        self.merge_factor = max(0.0, min(1.0, float(self.merge_factor)))
        self.aspect_x = max(0.25, min(4.0, float(self.aspect_x)))
        self.aspect_y = max(0.25, min(4.0, float(self.aspect_y)))
        self.row_offset = max(0.0, min(0.5, float(self.row_offset)))
        self.jitter_pos = max(0.0, min(1.0, float(self.jitter_pos)))
        self.jitter_size = max(0.0, min(1.0, float(self.jitter_size)))
        g_type = str(self.grad_type or "none").strip().lower()
        self.grad_type = g_type if g_type in ALLOWED_GRAD_TYPES else "none"
        self.grad_angle = max(-360.0, min(360.0, float(self.grad_angle)))
        self.grad_strength = max(-1.0, min(1.0, float(self.grad_strength)))

        # Validate colors
        parse_hex_color(self.dot_color)
        parse_hex_color(self.bg_color)

        if not isinstance(self.cmyk_angles, (list, tuple)) or len(self.cmyk_angles) < 4:
            self.cmyk_angles = [15.0, 75.0, 0.0, 45.0]
        else:
            self.cmyk_angles = [max(-180.0, min(180.0, float(a))) for a in self.cmyk_angles[:4]]

        return self

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]] = None) -> "HalftoneParams":
        d = dict(data or {})
        raw_mode = str(d.get("mode") or "manga").strip().lower()
        base: dict[str, Any] = dict(HALFTONE_PRESETS.get(raw_mode, HALFTONE_PRESETS["manga"]))
        base["mode"] = raw_mode
        for k, v in d.items():
            if v is not None and k in cls.__dataclass_fields__:
                base[k] = v
        return cls(**base).validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


HALFTONE_PRESETS: dict[str, dict[str, Any]] = {
    "manga": {
        "mode": "manga",
        "cell_size": 8,
        "max_radius_ratio": 0.95,
        "dot_shape": "circle",
        "angle": 45.0,
        "contrast": 1.35,
        "gamma": 1.10,
        "brightness": 0.0,
        "black_threshold": 0.22,
        "white_threshold": 0.80,
        "intensity": 1.0,
        "dot_color": "#141418",
        "bg_color": "#ffffff",
        "antialias": 2,
        "dither_type": "none",
        "dither_strength": 0.0,
        "preserve_lines": True,
        "paper_texture": 0.0,
    },
    "classic": {
        "mode": "classic",
        "cell_size": 10,
        "max_radius_ratio": 1.0,
        "dot_shape": "circle",
        "angle": 45.0,
        "contrast": 1.05,
        "gamma": 1.0,
        "brightness": 0.0,
        "black_threshold": 0.03,
        "white_threshold": 0.97,
        "intensity": 1.0,
        "dot_color": "#111111",
        "bg_color": "#ffffff",
        "antialias": 2,
        "dither_type": "none",
        "dither_strength": 0.0,
        "preserve_lines": False,
        "paper_texture": 0.0,
    },
    "newspaper": {
        "mode": "newspaper",
        "cell_size": 13,
        "max_radius_ratio": 1.05,
        "dot_shape": "circle",
        "angle": 45.0,
        "contrast": 1.45,
        "gamma": 0.95,
        "brightness": 0.0,
        "black_threshold": 0.16,
        "white_threshold": 0.86,
        "intensity": 1.0,
        "dot_color": "#22201e",
        "bg_color": "#f4f0e6",
        "antialias": 2,
        "dither_type": "noise",
        "dither_strength": 0.08,
        "preserve_lines": False,
        "paper_texture": 0.35,
    },
    "color": {
        "mode": "color",
        "cell_size": 9,
        "max_radius_ratio": 0.95,
        "dot_shape": "circle",
        "angle": 45.0,
        "contrast": 1.12,
        "gamma": 1.0,
        "brightness": 0.0,
        "black_threshold": 0.05,
        "white_threshold": 0.95,
        "intensity": 1.0,
        "dot_color": "#111111",
        "bg_color": "#ffffff",
        "antialias": 2,
        "dither_type": "none",
        "dither_strength": 0.0,
        "preserve_lines": False,
        "paper_texture": 0.0,
        "cmyk_angles": [15.0, 75.0, 0.0, 45.0],
    },
    "custom": {
        "mode": "custom",
        "cell_size": 8,
        "max_radius_ratio": 0.95,
        "dot_shape": "circle",
        "angle": 45.0,
        "contrast": 1.25,
        "gamma": 1.05,
        "brightness": 0.0,
        "black_threshold": 0.20,
        "white_threshold": 0.82,
        "intensity": 1.0,
        "dot_color": "#141418",
        "bg_color": "#ffffff",
        "antialias": 2,
        "dither_type": "none",
        "dither_strength": 0.0,
        "preserve_lines": True,
        "paper_texture": 0.0,
    },
}


def parse_hex_color(color_str: Union[str, tuple, list]) -> tuple[int, int, int]:
    """Parse '#RRGGBB' or '#RGB' or (R,G,B) into a validated (R, G, B) uint8 tuple."""
    if isinstance(color_str, (tuple, list)) and len(color_str) >= 3:
        return (
            max(0, min(255, int(color_str[0]))),
            max(0, min(255, int(color_str[1]))),
            max(0, min(255, int(color_str[2]))),
        )
    s = str(color_str or "").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6:
        raise ValueError(f"Invalid hex color: {color_str!r}")
    try:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return (r, g, b)
    except ValueError as exc:
        raise ValueError(f"Invalid hex color: {color_str!r}") from exc


def load_image_safe(
    source: Union[str, Path, bytes, Image.Image, tuple[np.ndarray, Optional[np.ndarray]]],
) -> Image.Image:
    """Open an image without mutating the source, respecting EXIF orientation.

    Returns:
        A detached PIL.Image.Image (mode 'RGB' or 'RGBA').
    """
    if isinstance(source, tuple) and len(source) == 2 and isinstance(source[0], np.ndarray):
        rgb_in, alpha_in = source
        if alpha_in is not None:
            return Image.fromarray(np.dstack([rgb_in, alpha_in]).astype(np.uint8), mode="RGBA")
        return Image.fromarray(rgb_in.astype(np.uint8), mode="RGB")
    if isinstance(source, Image.Image):
        img = source.copy()
    elif isinstance(source, (bytes, bytearray)):
        img = Image.open(io.BytesIO(source))
        img.load()
    elif isinstance(source, str) and source.strip().startswith("data:image"):
        _, _, b64_part = source.strip().partition(",")
        raw_bytes = base64.b64decode(b64_part)
        img = Image.open(io.BytesIO(raw_bytes))
        img.load()
    elif isinstance(source, (str, Path)):
        with Image.open(source) as opened:
            opened.load()
            img = opened.copy()
    else:
        raise TypeError(f"Unsupported image source type: {type(source)}")

    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        return img.convert("RGBA")
    return img.convert("RGB")


def extract_rgb_alpha(
    source: Union[str, Path, bytes, Image.Image, tuple[np.ndarray, Optional[np.ndarray]]],
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Extract (rgb_uint8_array [H, W, 3], alpha_uint8_array [H, W] or None) from any image source."""
    if isinstance(source, tuple) and len(source) == 2 and isinstance(source[0], np.ndarray):
        rgb_in, alpha_in = source
        return rgb_in.copy(), (alpha_in.copy() if alpha_in is not None else None)
    img = load_image_safe(source)
    if img.mode == "RGBA":
        arr = np.array(img, dtype=np.uint8)
        return arr[:, :, :3].copy(), arr[:, :, 3].copy()
    return np.array(img.convert("RGB"), dtype=np.uint8), None


def compute_perceptual_luminance(rgb: np.ndarray) -> np.ndarray:
    """Compute perceptual luminance in [0.0, 1.0] from uint8 RGB [H, W, 3]."""
    rgb_f = rgb.astype(np.float32) / 255.0
    return (0.299 * rgb_f[:, :, 0] + 0.587 * rgb_f[:, :, 1] + 0.114 * rgb_f[:, :, 2]).astype(np.float32)


def apply_levels_and_tone(
    lum: np.ndarray,
    black_point: float = 0.0,
    white_point: float = 1.0,
    gamma: float = 1.0,
    contrast: float = 1.0,
    brightness: float = 0.0,
    invert: bool = False,
    pre_blur: float = 0.0,
) -> np.ndarray:
    """Reusable tonal normalization pipeline for both Halftone Studio and Manga Levels.

    Operates on float32 luminance [0..1] and returns normalized float32 [0..1].
    """
    out = lum.astype(np.float32, copy=True)

    if pre_blur > 0.05:
        sigma = float(pre_blur)
        out = cv2.GaussianBlur(out, (0, 0), sigma)

    if invert:
        out = 1.0 - out

    if abs(brightness) > 1e-4:
        out = np.clip(out + float(brightness), 0.0, 1.0)

    # Black / white levels normalization
    span = max(float(white_point) - float(black_point), 1e-3)
    out = np.clip((out - float(black_point)) / span, 0.0, 1.0)

    # Gamma correction (gamma > 1 brightens midtones, matching MangaLevels convention)
    if abs(gamma - 1.0) > 1e-3:
        out = np.power(np.clip(out, 0.0, 1.0), 1.0 / max(float(gamma), 0.1))

    # Contrast around midpoint 0.5
    if abs(contrast - 1.0) > 1e-3:
        out = np.clip((out - 0.5) * float(contrast) + 0.5, 0.0, 1.0)

    return out.astype(np.float32)


def _bayer_matrix_4x4() -> np.ndarray:
    m = np.array(
        [
            [0, 8, 2, 10],
            [12, 4, 14, 6],
            [3, 11, 1, 9],
            [15, 7, 13, 5],
        ],
        dtype=np.float32,
    )
    return (m + 0.5) / 16.0 - 0.5


def _eval_shape_distance(
    du: np.ndarray,
    dv: np.ndarray,
    shape: str,
    max_r: float = 0.95,
) -> tuple[np.ndarray, bool]:
    """Evaluate normalized distance field inside a cell where du, dv are in [-0.5, 0.5].

    Returns:
        (dist_field, is_linear_radius):
          - dist_field: normalized distance where < target_r is inside the dot.
          - is_linear_radius: True if target_r = cov directly (lines/concentric), False if target_r = sqrt(cov).
    """
    half_r = max(0.05, 0.5 * max_r)
    if shape == "square":
        return np.maximum(np.abs(du), np.abs(dv)) / half_r, False
    if shape == "diamond":
        return (np.abs(du) + np.abs(dv)) / (half_r * 1.41421356), False
    if shape == "line":
        return np.abs(dv) / half_r, True
    if shape == "crosshatch":
        return np.minimum(np.abs(du), np.abs(dv)) / half_r, True
    if shape == "hexagon":
        ax = np.abs(du)
        ay = np.abs(dv)
        hex_d = np.maximum(ax * 0.8660254 + ay * 0.5, ay)
        return hex_d / (half_r * 0.95), False
    if shape == "triangle":
        # Equilateral triangle centered in cell
        ty = dv + 0.08 * max_r
        tri_d = np.maximum(np.abs(du) * 1.73205 + ty, -ty * 2.0) / 1.5
        return np.maximum(0.0, tri_d) / (half_r * 0.95), False
    if shape == "star":
        r = np.hypot(du, dv)
        theta = np.arctan2(dv, du)
        # 5-pointed star polar modulation
        star_mod = 0.68 + 0.32 * np.cos(5.0 * theta - np.pi / 2.0)
        return r / (half_r * 1.22 * np.maximum(star_mod, 0.2)), False
    if shape == "heart":
        hx = du / (half_r * 1.15)
        hy = -dv / (half_r * 1.15) + 0.15
        # Implicit heart curve approximation converted to radial scale
        r = np.hypot(hx, hy - 0.35 * np.sqrt(np.abs(hx) + 1e-4))
        return r / 0.85, False
    if shape == "concentric":
        r = np.hypot(du, dv) / (half_r * 1.2)
        ring = np.abs((r * 2.0) - np.floor(r * 2.0) - 0.5) * 2.0
        return ring, True
    if shape == "cross":
        ax = np.abs(du)
        ay = np.abs(dv)
        d = np.minimum(np.maximum(ax * 2.6, ay * 0.85), np.maximum(ax * 0.85, ay * 2.6))
        return d / (half_r * 0.9), False
    if shape == "ellipse":
        return np.hypot(du * 0.82, dv * 1.22) / (half_r * 1.05), False
    if shape == "ring":
        r = np.hypot(du, dv) / (half_r * 1.1)
        return np.abs(r - 0.52) * 2.3, True
    # Default: 'circle' or 'stipple' or 'stochastic'
    return np.hypot(du, dv) / (half_r * 1.128379), False


def _hash_cell_2d(ix: np.ndarray, iy: np.ndarray, salt: int = 0) -> np.ndarray:
    """Fast deterministic 2D integer grid hash returning float32 in [-0.5, 0.5]."""
    h = (ix.astype(np.int64) * 374761393 + iy.astype(np.int64) * 668265263 + int(salt) * 1442695041) & 0x7FFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0x7FFFFFFF
    h = (h ^ (h >> 16)) & 0x7FFFFFFF
    return (h.astype(np.float32) / float(0x7FFFFFFF)) - 0.5


def render_halftone_screen(
    darkness: np.ndarray,
    cell_size: float,
    angle: float = 45.0,
    dot_shape: str = "circle",
    max_radius_ratio: float = 0.95,
    antialias: int = 2,
    dither_type: str = "none",
    dither_strength: float = 0.0,
    seed: int = 42,
    binary_threshold: bool = False,
    *,
    satellite_enabled: bool = False,
    satellite_count: int = 4,
    satellite_size_ratio: float = 0.45,
    satellite_distance: float = 0.42,
    satellite_angle: float = 0.0,
    satellite_shape: str = "circle",
    satellite_stretch: float = 0.0,
    merge_factor: float = 0.0,
    aspect_x: float = 1.0,
    aspect_y: float = 1.0,
    row_offset: float = 0.0,
    jitter_pos: float = 0.0,
    jitter_size: float = 0.0,
    grad_type: str = "none",
    grad_angle: float = 45.0,
    grad_strength: float = 0.0,
) -> np.ndarray:
    """Render a single-channel halftone dot screen (with GenTonik Planets & Satellites, shapes, and distortions)."""
    h, w = darkness.shape[:2]
    s = max(1, min(4, int(antialias)))
    pitch = max(2.0, float(cell_size)) * s
    sh, sw = h * s, w * s

    # Upsample darkness map to supersampled resolution
    if s > 1:
        cov = cv2.resize(darkness.astype(np.float32), (sw, sh), interpolation=cv2.INTER_LINEAR)
    else:
        cov = darkness.astype(np.float32, copy=True)

    # Optional GenTonik gradient modulation across canvas
    if grad_type in ("linear", "radial", "vignette") and abs(grad_strength) > 1e-3:
        gy = np.linspace(-0.5, 0.5, sh, dtype=np.float32)[:, None]
        gx = np.linspace(-0.5, 0.5, sw, dtype=np.float32)[None, :]
        if grad_type == "linear":
            grad_rad = np.deg2rad(float(grad_angle))
            g_map = (gx * float(np.cos(grad_rad)) + gy * float(np.sin(grad_rad))) + 0.5
        elif grad_type == "radial":
            g_map = np.clip(1.0 - np.hypot(gx, gy) * 1.41421356, 0.0, 1.0)
        else:  # vignette
            g_map = np.clip(np.power(np.hypot(gx, gy) * 1.41421356, 1.6) - 0.25, 0.0, 1.0)
        g_map = np.clip(g_map, 0.0, 1.0)
        cov = np.clip(cov * (1.0 - float(grad_strength) * (g_map - 0.5) * 1.5), 0.0, 1.0)

    # Optional deterministic dithering on the coverage map
    if dither_type != "none":
        eff_strength = float(dither_strength) if float(dither_strength) > 1e-4 else (0.10 if dither_type in ("bayer", "noise") else 0.0)
        if eff_strength > 1e-4:
            if dither_type in ("bayer", "ordered"):
                b4 = _bayer_matrix_4x4()
                tiled = np.tile(b4, (sh // 4 + 1, sw // 4 + 1))[:sh, :sw]
                cov = np.clip(cov + tiled * eff_strength, 0.0, 1.0)
            elif dither_type == "noise":
                rng = np.random.default_rng(int(seed))
                noise = rng.uniform(-0.5, 0.5, size=(h, w)).astype(np.float32)
                if s > 1:
                    noise = cv2.resize(noise, (sw, sh), interpolation=cv2.INTER_NEAREST)
                cov = np.clip(cov + noise * eff_strength, 0.0, 1.0)

    # Rotated coordinate grid in cell units around image center (matching GLSL & GenTonik)
    y, x = np.mgrid[0:sh, 0:sw].astype(np.float32)
    cx = sw * 0.5
    cy = sh * 0.5
    dx = x - cx
    dy = y - cy
    rad = np.deg2rad(float(angle))
    cos_a = float(np.cos(rad))
    sin_a = float(np.sin(rad))

    pitch_x = max(1.5, pitch * float(aspect_x))
    pitch_y = max(1.5, pitch * float(aspect_y))
    u = (dx * cos_a + dy * sin_a) / pitch_x
    v = (-dx * sin_a + dy * cos_a) / pitch_y

    # Row offset (brick / hex layout from GenTonik)
    eff_row_offset = 0.5 if dot_shape == "hexagon" and abs(row_offset) < 1e-4 else float(row_offset)
    cell_v = np.floor(v)
    if abs(eff_row_offset) > 1e-4:
        odd_row = (np.abs(cell_v.astype(np.int64)) & 1).astype(np.float32)
        u = u + odd_row * eff_row_offset

    cell_u = np.floor(u)

    # Local coordinates within each cell: [-0.5, 0.5]
    du = (u - cell_u) - 0.5
    dv = (v - cell_v) - 0.5

    # Jitter position & size (for stipple or GenTonik jitter)
    eff_jitter_pos = max(float(jitter_pos), 0.45 if dot_shape == "stipple" else 0.0)
    eff_jitter_size = max(float(jitter_size), 0.35 if dot_shape == "stipple" else 0.0)
    if eff_jitter_pos > 1e-3:
        jx = _hash_cell_2d(cell_u, cell_v, int(seed) + 11) * eff_jitter_pos * 0.55
        jy = _hash_cell_2d(cell_u, cell_v, int(seed) + 29) * eff_jitter_pos * 0.55
        du = du - jx
        dv = dv - jy

    cov_mod = cov
    if eff_jitter_size > 1e-3:
        js = 1.0 + _hash_cell_2d(cell_u, cell_v, int(seed) + 53) * eff_jitter_size * 0.75
        cov_mod = np.clip(cov * js, 0.0, 1.0)

    # Merge factor (GenTonik planet growth)
    max_r = max(0.1, float(max_radius_ratio) * (1.0 + float(merge_factor) * 0.35))

    dist, is_linear = _eval_shape_distance(du, dv, dot_shape, max_r)
    target_r = np.clip(cov_mod, 0.0, 1.0) if is_linear else np.sqrt(np.clip(cov_mod, 0.0, 1.0))
    ink_hi = (dist < target_r).astype(np.float32)

    # GenTonik Planets & Satellites (1..8 slot orbit around each main dot)
    if satellite_enabled and int(satellite_count) > 0:
        sat_dist = max(0.12, float(satellite_distance) * (1.0 - float(merge_factor) * 0.5))
        sat_max_r = max(0.08, max_r * float(satellite_size_ratio) * (1.0 + float(merge_factor) * 0.45))
        sat_rad = np.deg2rad(float(satellite_angle))
        eff_count = max(1, min(8, int(satellite_count)))
        step_ang = (2.0 * np.pi) / float(eff_count)
        stretch = float(satellite_stretch)
        for idx in range(eff_count):
            theta = sat_rad + float(idx) * step_ang
            rsx = sat_dist * float(np.cos(theta))
            rsy = sat_dist * float(np.sin(theta))
            s_du = du - rsx
            s_dv = dv - rsy

            if stretch > 1e-3 and (abs(rsx) > 1e-5 or abs(rsy) > 1e-5):
                # Stretch satellite along axis pointing toward parent planet
                dir_angle = float(np.arctan2(-rsy, -rsx))
                cd = float(np.cos(dir_angle))
                sd = float(np.sin(dir_angle))
                rot_u = (s_du * cd + s_dv * sd) / (1.0 + stretch)
                rot_v = -s_du * sd + s_dv * cd
                s_du = rot_u * cd - rot_v * sd
                s_dv = rot_u * sd + rot_v * cd

            s_dist, s_linear = _eval_shape_distance(s_du, s_dv, satellite_shape, sat_max_r)
            s_target = np.clip(cov_mod, 0.0, 1.0) if s_linear else np.sqrt(np.clip(cov_mod, 0.0, 1.0))
            sat_ink = (s_dist < s_target).astype(np.float32)
            ink_hi = np.maximum(ink_hi, sat_ink)

    # Pure white and solid black overrides:
    ink_hi = np.where(cov <= 1e-4, 0.0, ink_hi)
    if max_r >= 0.92:
        ink_hi = np.where(cov >= 0.995, 1.0, ink_hi)

    if s > 1:
        down = cv2.resize(ink_hi, (w, h), interpolation=cv2.INTER_AREA)
    else:
        down = ink_hi

    if binary_threshold:
        return (down > 0.5).astype(np.float32)
    return np.clip(down, 0.0, 1.0).astype(np.float32)


def _extract_quick_manga_strokes(lum: np.ndarray) -> np.ndarray:
    """Extract clean structural line strokes on CPU so Manga mode in Halftone Studio keeps sharp lines."""
    h, w = lum.shape[:2]
    k = max(0.5, w / 1024.0)
    lb = cv2.GaussianBlur(lum, (0, 0), 0.7 * k)
    ksize = int(15 * k) | 1
    local_mean = cv2.blur(lb, (ksize, ksize))
    line_cand = (lb < (local_mean - 0.065)) & (lb < 0.65)
    _, lab, stats, _ = cv2.connectedComponentsWithStats(line_cand.astype(np.uint8), connectivity=8)
    cw = stats[:, cv2.CC_STAT_WIDTH]
    ch = stats[:, cv2.CC_STAT_HEIGHT]
    area = stats[:, cv2.CC_STAT_AREA]
    keep = (np.maximum(cw, ch) >= max(6, 9 * k)) & (area >= max(6, 10 * k * k))
    keep[0] = False
    return keep[lab].astype(np.float32)


def _apply_paper_grain(rgb_f: np.ndarray, strength: float, seed: int = 42) -> np.ndarray:
    """Apply subtle deterministic newsprint/paper fiber texture."""
    if strength <= 1e-3:
        return rgb_f
    h, w = rgb_f.shape[:2]
    rng = np.random.default_rng(int(seed))
    fine = rng.standard_normal((max(2, h // 2), max(2, w // 2))).astype(np.float32)
    coarse = rng.standard_normal((max(2, h // 16), max(2, w // 16))).astype(np.float32)
    grain = (
        0.65 * cv2.resize(fine, (w, h), interpolation=cv2.INTER_LINEAR)
        + 0.35 * cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    )
    delta = grain[:, :, None] * (float(strength) * 14.0)
    return np.clip(rgb_f + delta, 0.0, 255.0)


def decode_mask_to_float(
    mask_source: Union[str, Path, bytes, Image.Image, np.ndarray, None],
    target_size: tuple[int, int],
    invert_mask: bool = False,
) -> Optional[np.ndarray]:
    """Decode an optional mask (data URL, bytes, Path, PIL Image, or ndarray) into [H, W] float32 in [0..1].

    Semantics (HALFTONE_STUDIO_SPEC.md §6):
      White (1.0) = apply effect
      Black (0.0) = keep original image
    """
    if mask_source is None or mask_source == "":
        return None

    target_w, target_h = target_size
    if isinstance(mask_source, np.ndarray):
        m_arr = mask_source
        if m_arr.ndim == 3:
            if m_arr.shape[2] == 4:
                rgb_part = m_arr[:, :, :3].mean(axis=2)
                alpha_part = m_arr[:, :, 3]
                m_arr = np.maximum(rgb_part, alpha_part)
            else:
                m_arr = m_arr[:, :, :3].mean(axis=2)
        m_f = m_arr.astype(np.float32)
        if m_f.max() > 1.0:
            m_f = m_f / 255.0
    else:
        if isinstance(mask_source, str) and mask_source.startswith("data:image"):
            _, _, b64_part = mask_source.partition(",")
            raw_bytes = base64.b64decode(b64_part)
            opened_img = Image.open(io.BytesIO(raw_bytes))
        elif isinstance(mask_source, (bytes, bytearray)):
            opened_img = Image.open(io.BytesIO(mask_source))
        elif isinstance(mask_source, (str, Path)):
            with Image.open(mask_source) as f_img:
                opened_img = f_img.copy()
        elif isinstance(mask_source, Image.Image):
            opened_img = mask_source.copy()
        else:
            raise ValueError(f"Unsupported mask type: {type(mask_source)}")

        if opened_img.mode in ("RGBA", "LA") or (opened_img.mode == "P" and "transparency" in opened_img.info):
            arr = np.array(opened_img.convert("RGBA"), dtype=np.float32) / 255.0
            rgb_lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
            alpha_chan = arr[:, :, 3]
            m_f = np.maximum(rgb_lum, alpha_chan)
        else:
            m_img = opened_img.convert("L")
            m_f = np.array(m_img, dtype=np.float32) / 255.0

    if m_f.shape[:2] != (target_h, target_w):
        m_f = cv2.resize(m_f, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    m_f = np.clip(m_f, 0.0, 1.0)
    if invert_mask:
        m_f = 1.0 - m_f
    return m_f


def _blend_rgb_layers(base_f: np.ndarray, top_f: np.ndarray, blend_mode: str = "normal") -> np.ndarray:
    """Blend two float32 RGB arrays in [0..255] using GenTonik blend modes."""
    mode = str(blend_mode or "normal").strip().lower()
    if mode == "normal":
        return top_f
    b = np.clip(base_f / 255.0, 0.0, 1.0)
    t = np.clip(top_f / 255.0, 0.0, 1.0)
    if mode == "multiply":
        res = b * t
    elif mode == "screen":
        res = 1.0 - (1.0 - b) * (1.0 - t)
    elif mode == "darken":
        res = np.minimum(b, t)
    elif mode == "lighten":
        res = np.maximum(b, t)
    elif mode == "overlay":
        res = np.where(b < 0.5, 2.0 * b * t, 1.0 - 2.0 * (1.0 - b) * (1.0 - t))
    else:
        res = t
    return np.clip(res * 255.0, 0.0, 255.0)


def process_halftone_image(
    source: Union[str, Path, bytes, Image.Image, tuple[np.ndarray, Optional[np.ndarray]]],
    params: Union[HalftoneParams, dict[str, Any], None] = None,
    mask: Union[str, Path, bytes, Image.Image, np.ndarray, None] = None,
    max_side: Optional[int] = None,
    *,
    mask_source: Union[str, Path, bytes, Image.Image, np.ndarray, None] = None,
    preview_max_side: Optional[int] = None,
) -> Image.Image:
    """Run the complete Halftone Studio GT pipeline on CPU without mutating the source image."""
    if mask is None and mask_source is not None:
        mask = mask_source
    if max_side is None and preview_max_side is not None:
        max_side = preview_max_side

    if isinstance(params, dict):
        cfg = HalftoneParams.from_dict(params)
    elif isinstance(params, HalftoneParams):
        cfg = params.validate()
    else:
        cfg = HalftoneParams.from_dict({"mode": "manga"})

    rgb_orig, alpha_orig = extract_rgb_alpha(source)
    orig_h, orig_w = rgb_orig.shape[:2]

    if max_side and max(orig_h, orig_w) > int(max_side):
        scale = float(max_side) / float(max(orig_h, orig_w))
        work_w = max(16, int(round(orig_w * scale)))
        work_h = max(16, int(round(orig_h * scale)))
        rgb = cv2.resize(rgb_orig, (work_w, work_h), interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha_orig, (work_w, work_h), interpolation=cv2.INTER_AREA) if alpha_orig is not None else None
    else:
        rgb = rgb_orig
        alpha = alpha_orig
        work_h, work_w = orig_h, orig_w

    dot_rgb = np.array(parse_hex_color(cfg.dot_color), dtype=np.float32)
    bg_rgb = np.array(parse_hex_color(cfg.bg_color), dtype=np.float32)

    gt_kwargs = dict(
        satellite_enabled=cfg.satellite_enabled,
        satellite_count=cfg.satellite_count,
        satellite_size_ratio=cfg.satellite_size_ratio,
        satellite_distance=cfg.satellite_distance,
        satellite_angle=cfg.satellite_angle,
        satellite_shape=cfg.satellite_shape,
        satellite_stretch=cfg.satellite_stretch,
        merge_factor=cfg.merge_factor,
        aspect_x=cfg.aspect_x,
        aspect_y=cfg.aspect_y,
        row_offset=cfg.row_offset,
        jitter_pos=cfg.jitter_pos,
        jitter_size=cfg.jitter_size,
        grad_type=cfg.grad_type,
        grad_angle=cfg.grad_angle,
        grad_strength=cfg.grad_strength,
    )

    if cfg.mode == "color":
        # Multi-channel CMYK color halftone with independent channel angles
        rgb_norm = rgb.astype(np.float32) / 255.0
        if cfg.pre_blur > 0.05:
            rgb_norm = cv2.GaussianBlur(rgb_norm, (0, 0), float(cfg.pre_blur))
        if cfg.invert:
            rgb_norm = 1.0 - rgb_norm

        for c in range(3):
            rgb_norm[:, :, c] = apply_levels_and_tone(
                rgb_norm[:, :, c],
                black_point=cfg.black_threshold,
                white_point=cfg.white_threshold,
                gamma=cfg.gamma,
                contrast=cfg.contrast,
                brightness=cfg.brightness,
                invert=False,
                pre_blur=0.0,
            )

        k_chan = 1.0 - np.max(rgb_norm, axis=2)
        denom = np.maximum(1.0 - k_chan, 1e-4)
        c_chan = np.clip((1.0 - rgb_norm[:, :, 0] - k_chan) / denom, 0.0, 1.0)
        m_chan = np.clip((1.0 - rgb_norm[:, :, 1] - k_chan) / denom, 0.0, 1.0)
        y_chan = np.clip((1.0 - rgb_norm[:, :, 2] - k_chan) / denom, 0.0, 1.0)
        k_chan = np.clip(k_chan * 0.85, 0.0, 1.0)

        angles = cfg.cmyk_angles
        c_dots = render_halftone_screen(
            c_chan, cfg.cell_size, angle=angles[0], dot_shape=cfg.dot_shape,
            max_radius_ratio=cfg.max_radius_ratio, antialias=cfg.antialias,
            dither_type=cfg.dither_type, dither_strength=cfg.dither_strength, seed=cfg.seed,
            **gt_kwargs,
        )
        m_dots = render_halftone_screen(
            m_chan, cfg.cell_size, angle=angles[1], dot_shape=cfg.dot_shape,
            max_radius_ratio=cfg.max_radius_ratio, antialias=cfg.antialias,
            dither_type=cfg.dither_type, dither_strength=cfg.dither_strength, seed=cfg.seed + 1,
            **gt_kwargs,
        )
        y_dots = render_halftone_screen(
            y_chan, cfg.cell_size, angle=angles[2], dot_shape=cfg.dot_shape,
            max_radius_ratio=cfg.max_radius_ratio, antialias=cfg.antialias,
            dither_type=cfg.dither_type, dither_strength=cfg.dither_strength, seed=cfg.seed + 2,
            **gt_kwargs,
        )
        k_dots = render_halftone_screen(
            k_chan, cfg.cell_size, angle=angles[3], dot_shape=cfg.dot_shape,
            max_radius_ratio=cfg.max_radius_ratio, antialias=cfg.antialias,
            dither_type=cfg.dither_type, dither_strength=cfg.dither_strength, seed=cfg.seed + 3,
            **gt_kwargs,
        )

        r_out = bg_rgb[0] * (1.0 - c_dots) * (1.0 - k_dots)
        g_out = bg_rgb[1] * (1.0 - m_dots) * (1.0 - k_dots)
        b_out = bg_rgb[2] * (1.0 - y_dots) * (1.0 - k_dots)
        rendered_f = np.stack([r_out, g_out, b_out], axis=2)
        if cfg.preserve_lines:
            lum = compute_perceptual_luminance(rgb)
            strokes = _extract_quick_manga_strokes(lum if not cfg.invert else 1.0 - lum)
            if strokes is not None:
                stroke_mask = (strokes > 0.5)[:, :, None]
                rendered_f = np.where(stroke_mask, 0.0, rendered_f)
    else:
        lum = compute_perceptual_luminance(rgb)
        norm_lum = apply_levels_and_tone(
            lum,
            black_point=cfg.black_threshold,
            white_point=cfg.white_threshold,
            gamma=cfg.gamma,
            contrast=cfg.contrast,
            brightness=cfg.brightness,
            invert=cfg.invert,
            pre_blur=cfg.pre_blur,
        )

        if cfg.mode == "manga":
            strokes = _extract_quick_manga_strokes(lum if not cfg.invert else 1.0 - lum) if cfg.preserve_lines else None
            darkness = 1.0 - norm_lum
            darkness = np.where(norm_lum >= 0.94, 0.0, darkness)
            darkness = np.where(norm_lum <= 0.06, 1.0, darkness)

            dots = render_halftone_screen(
                darkness,
                cell_size=cfg.cell_size,
                angle=cfg.angle,
                dot_shape=cfg.dot_shape,
                max_radius_ratio=cfg.max_radius_ratio,
                antialias=cfg.antialias,
                dither_type=cfg.dither_type,
                dither_strength=cfg.dither_strength,
                seed=cfg.seed,
                binary_threshold=False,
                **gt_kwargs,
            )
            dots = np.where(norm_lum <= 0.04, 1.0, dots)
            if strokes is not None:
                dots = np.maximum(dots, strokes)
        else:
            darkness = 1.0 - norm_lum
            dots = render_halftone_screen(
                darkness,
                cell_size=cfg.cell_size,
                angle=cfg.angle,
                dot_shape=cfg.dot_shape,
                max_radius_ratio=cfg.max_radius_ratio,
                antialias=cfg.antialias,
                dither_type=cfg.dither_type,
                dither_strength=cfg.dither_strength,
                seed=cfg.seed,
                binary_threshold=False,
                **gt_kwargs,
            )
            if cfg.preserve_lines:
                strokes = _extract_quick_manga_strokes(lum if not cfg.invert else 1.0 - lum)
                dots = np.maximum(dots, strokes)

        rendered_f = bg_rgb[None, None, :] * (1.0 - dots[:, :, None]) + dot_rgb[None, None, :] * dots[:, :, None]

    if cfg.paper_texture > 1e-3:
        rendered_f = _apply_paper_grain(rendered_f, cfg.paper_texture, seed=cfg.seed)

    if cfg.intensity < 0.999:
        alpha_blend = float(cfg.intensity)
        rendered_f = rgb.astype(np.float32) * (1.0 - alpha_blend) + rendered_f * alpha_blend

    mask_f = decode_mask_to_float(mask, (work_w, work_h), invert_mask=cfg.invert_mask)
    if mask_f is not None:
        m3 = mask_f[:, :, None]
        rendered_f = rgb.astype(np.float32) * (1.0 - m3) + rendered_f * m3

    out_rgb = np.clip(np.round(rendered_f), 0, 255).astype(np.uint8)

    if alpha is not None:
        out_rgba = np.dstack([out_rgb, alpha])
        return Image.fromarray(out_rgba, mode="RGBA")
    return Image.fromarray(out_rgb, mode="RGB")


def process_halftone_layers(
    source: Union[str, Path, bytes, Image.Image, tuple[np.ndarray, Optional[np.ndarray]]],
    layers: list[dict[str, Any]],
    max_side: Optional[int] = None,
) -> Image.Image:
    """Composite multiple Halftone Studio GT layers over the source image.

    Each layer dict supports:
      - visible (bool, default True)
      - opacity (float 0..1, default 1.0)
      - blend_mode ('normal' | 'multiply' | 'screen' | 'overlay' | 'darken' | 'lighten')
      - params (dict of HalftoneParams)
      - mask (optional data URL / mask source)
    """
    rgb_orig, alpha_orig = extract_rgb_alpha(source)
    orig_h, orig_w = rgb_orig.shape[:2]

    if max_side and max(orig_h, orig_w) > int(max_side):
        scale = float(max_side) / float(max(orig_h, orig_w))
        work_w = max(16, int(round(orig_w * scale)))
        work_h = max(16, int(round(orig_h * scale)))
        rgb_base = cv2.resize(rgb_orig, (work_w, work_h), interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha_orig, (work_w, work_h), interpolation=cv2.INTER_AREA) if alpha_orig is not None else None
    else:
        rgb_base = rgb_orig.copy()
        alpha = alpha_orig.copy() if alpha_orig is not None else None
        work_h, work_w = orig_h, orig_w

    comp_f = rgb_base.astype(np.float32)

    for layer in layers or []:
        if not isinstance(layer, dict):
            continue
        # The virtual original/background layer is represented by the source
        # image itself and must never be toned or blended as a raster layer.
        if layer.get("isOriginal") or str(layer.get("id") or "") == "hl_original":
            continue
        if not layer.get("visible", True):
            continue
        opacity = max(0.0, min(1.0, float(layer.get("opacity", 1.0))))
        if opacity <= 1e-3:
            continue
        blend_mode = str(layer.get("blend_mode") or layer.get("blendMode") or "normal").strip().lower()
        l_params = HalftoneParams.from_dict(layer.get("params") or {})
        l_mask = layer.get("mask")

        # Render this layer from the original source (without mask yet so we can apply blend_mode + mask + opacity cleanly)
        layer_img = process_halftone_image(
            (rgb_base, None),
            l_params,
            mask=None,
            max_side=None,
        )
        layer_rgb_f = np.array(layer_img.convert("RGB"), dtype=np.float32)
        blended_f = _blend_rgb_layers(comp_f, layer_rgb_f, blend_mode=blend_mode)

        mask_f = decode_mask_to_float(l_mask, (work_w, work_h), invert_mask=l_params.invert_mask)
        if mask_f is not None:
            eff_alpha = (mask_f * opacity)[:, :, None]
        else:
            eff_alpha = opacity

        comp_f = comp_f * (1.0 - eff_alpha) + blended_f * eff_alpha

    out_rgb = np.clip(np.round(comp_f), 0, 255).astype(np.uint8)
    if alpha is not None:
        return Image.fromarray(np.dstack([out_rgb, alpha]), mode="RGBA")
    return Image.fromarray(out_rgb, mode="RGB")


def image_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL Image as a base64 data URL."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"
