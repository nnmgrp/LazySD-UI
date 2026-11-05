"""Measured surface description for Inpaint removal prompts.

Why this module exists
----------------------
FLUX 3 PUP (the prompt upsampler) is a **captioner**, not an instruction
following removal assistant.  Verified on the user's own failing case
(``docs/INPAINT_INK_BRUSH_INVESTIGATION.md`` §10):

* it ignored an explicit "answer with AT MOST 12 words" rule and returned 41
  words;
* it captioned the whole crop - including the character standing next to the
  masked area - instead of the area around the mask;
* the hallucinated subject ("a deep black and burgundy character-like torso
  rendered in a gritty, high-contrast ink style ... irregular crimson
  brushwork") was copied verbatim into the diffusion prompt, and the diffusion
  model then painted exactly that: a torso with ink brushwork instead of a
  clean fill.

So for removal tasks the caption must not be the source of truth.  The colour
and tonal facts are **measured** from the pixels around the mask
(``backend.prompt_fallback.analyze_region``), and the caption is only allowed to
contribute surface-material wording through a strict whitelist
(:func:`extract_surface_vocabulary`).  Subject nouns, style adjectives and ink
vocabulary can therefore never reach the diffusion prompt through the caption.
"""
from __future__ import annotations

import colorsys
import re
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageFilter
import numpy as np

# Marker appended to every removal prompt: the fill must continue the surface
# that surrounds the masked area instead of being a flat colour patch.
SURFACE_CONTINUATION_MARKER = (
    "seamless continuation of the surrounding surface, matching its colour, tone and texture"
)

_SURFACE_MARKER_RE = re.compile(
    r",?\s*seamless\s+continuation\s+of\s+the\s+surrounding\s+surface,?\s*"
    r"matching\s+its\s+colour,?\s*tone\s+and\s+texture",
    re.IGNORECASE,
)

# Textual description of the measured surface, used by the removal prompt and
# by the heuristic fallback.
_SURFACE_WORD = "surface"


# --------------------------------------------------------------------------- #
# Colour naming (moved from backend.prompt_fallback so that both the heuristic
# and the VLM path share one implementation).
# --------------------------------------------------------------------------- #
def rgb_to_color_name(rgb: Tuple[int, int, int]) -> str:
    """Convert an RGB tuple (0..255) into a descriptive English colour phrase."""
    r, g, b = (float(c) / 255.0 for c in rgb)
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin
    val = cmax
    sat = 0.0 if cmax <= 1e-5 else (delta / cmax)

    if val > 0.90 and sat < 0.12:
        return "white"

    # Compute hue in degrees [0, 360)
    if delta <= 1e-5:
        hue = 0.0
    elif cmax == r:
        hue = 60.0 * (((g - b) / delta) % 6.0)
    elif cmax == g:
        hue = 60.0 * (((b - r) / delta) + 2.0)
    else:
        hue = 60.0 * (((r - g) / delta) + 4.0)

    def _base_name() -> str:
        if hue < 15 or hue >= 345:
            return "red"
        if hue < 42:
            return "orange" if val > 0.55 else "brown"
        if hue < 68:
            return "yellow" if sat > 0.35 else "warm beige"
        if hue < 160:
            return "green"
        if hue < 200:
            return "teal"
        if hue < 255:
            return "blue"
        if hue < 295:
            return "purple"
        return "crimson" if val < 0.55 else "pink"

    if val < 0.14:
        # A very dark pixel can still be a saturated colour - a navy torso
        # (#010320) must not be called "black", otherwise the fill gets the
        # wrong colour.
        if sat >= 0.25:
            return f"dark {_base_name()}"
        return "black"
    if sat < 0.14:
        if val < 0.38:
            return "dark grey"
        if val > 0.72:
            return "light grey"
        return "neutral grey"

    base = _base_name()
    if val < 0.38 and base not in ("brown", "crimson"):
        return f"dark {base}"
    if val > 0.82 and sat < 0.45 and base not in ("warm beige",):
        return f"pale {base}"
    return base


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    s = (hex_color or "").lstrip("#")
    if len(s) != 6:
        return (255, 255, 255)
    try:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    except ValueError:
        return (255, 255, 255)


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = (max(0, min(255, int(c))) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def extract_dominant_colors(
    image: Image.Image,
    sample_mask: Optional[np.ndarray] = None,
    max_colors: int = 5,
) -> List[str]:
    """Return up to ``max_colors`` dominant hex colours sorted by area."""
    rgb_img = image.convert("RGB")
    small = rgb_img.resize((64, 64), Image.Resampling.BILINEAR)
    arr = np.asarray(small, dtype=np.uint8)

    if sample_mask is not None and sample_mask.ndim == 2 and sample_mask.size > 0:
        mask_pil = Image.fromarray((np.clip(sample_mask, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
        mask_small = np.asarray(mask_pil.resize((64, 64), Image.Resampling.NEAREST), dtype=np.uint8)
        pixels = arr[mask_small > 32]
        if pixels.shape[0] < 16:
            pixels = arr.reshape(-1, 3)
    else:
        pixels = arr.reshape(-1, 3)

    if pixels.shape[0] == 0:
        return ["#ffffff"]

    # Quantise RGB into 16-level buckets (4 bits per channel) and average each cluster
    buckets = (pixels // 24).astype(np.int32)
    keys = (buckets[:, 0] << 8) | (buckets[:, 1] << 4) | buckets[:, 2]
    unique_keys, counts = np.unique(keys, return_counts=True)

    # Filter out tiny noise clusters (< 2% of pixels) unless only one cluster exists
    min_count = max(1, int(pixels.shape[0] * 0.02))
    order = np.argsort(-counts)
    hex_colors: List[str] = []
    for idx in order:
        if len(hex_colors) >= max_colors:
            break
        if counts[idx] < min_count and len(hex_colors) > 0:
            continue
        cluster_pixels = pixels[keys == unique_keys[idx]]
        mean_rgb = tuple(int(round(float(v))) for v in cluster_pixels.mean(axis=0))
        hex_val = rgb_to_hex(mean_rgb)
        if hex_val not in hex_colors:
            hex_colors.append(hex_val)

    return hex_colors or ["#ffffff"]


def _refine_color_name(name: str, hex_color: Optional[str]) -> str:
    """Sharpen a few generic colour names using the actual pixel value."""
    if not hex_color:
        return name
    r, g, b = hex_to_rgb(hex_color)
    _h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    if name == "dark blue" or (name == "blue" and v < 0.45 and s > 0.20):
        return "dark navy blue"
    if name == "blue" and v < 0.62:
        return "deep blue"
    if name == "red" and v < 0.55:
        return "deep crimson red"
    if name == "crimson" and v < 0.45:
        return "deep crimson"
    if name == "purple" and v < 0.45:
        return "deep purple"
    return name


# --------------------------------------------------------------------------- #
# Surface vocabulary whitelist
# --------------------------------------------------------------------------- #
# Only these (regex, adjective, noun) triples may travel from a VLM caption
# into the diffusion prompt.  Everything else - subject nouns ("character",
# "torso", "foliage"), style adjectives ("gritty", "high-contrast"), ink
# vocabulary ("brushwork", "splatter") - is dropped by construction.
_SURFACE_VOCABULARY: Tuple[Tuple[str, str, str], ...] = (
    (r"\bhalftone\b|\bscreentone\b|\bdot\s+pattern\b", "halftone", "dot pattern"),
    (r"\bstippled\b", "stippled", "texture"),
    (r"\bspeckl\w*\b", "speckled", "texture"),
    (r"\bgrainy\b|\bfilm\s+grain\b", "fine", "grain"),
    (r"\bmottled\b|\bmottling\b|\bpatchy\b|\bblotchy\b", "mottled", "shading"),
    (r"\bcloudy\b|\bclouds\b|\bwisp\w*\b", "soft", "cloudy patches"),
    (r"\bshaded\b|\bshadows\b|\bgradient\b", "soft", "shading"),
    (r"\bpainterly\b", "painterly", "texture"),
    (r"\bdense\b", "dense", "texture"),
    (r"\bdistressed\b", "distressed", "texture"),
    (r"\bpaper\b", "paper", "grain"),
    (r"\bwood\b|\bwoodgrain\b", "wood", "grain"),
    (r"\bmarbled\b|\bveined\b", "marbled", "texture"),
    (r"\bconcrete\b", "concrete", "surface"),
    (r"\bfabric\b|\bcloth\b", "fabric", "weave"),
    (r"\bvelvety\b|\bsilky\b", "velvety", "surface"),
    (r"\bmatte\b", "matte", "surface"),
    (r"\blayered\b", "layered", "tonal variation"),
    (r"\buneven\b", "uneven", "tonal variation"),
    (r"\bnoisy\b|\bnoise\b", "fine", "noise"),
    (r"\bhazy\b", "soft", "haze"),
    (r"\bpatches\b", "soft", "patches"),
    (r"\bfragments\b", "angular", "fragments"),
)


def extract_surface_vocabulary(text: str, *, limit: int = 4) -> List[str]:
    """Extract whitelisted surface-material wording from a caption.

    Returns at most ``limit`` short phrases such as ``"dense painterly texture"``.
    Subject nouns, style adjectives and ink vocabulary can never appear here.
    """
    low = (text or "").lower()
    if not low:
        return []

    hits: List[Tuple[int, str, str]] = []
    for pattern, adj, noun in _SURFACE_VOCABULARY:
        m = re.search(pattern, low)
        if m:
            hits.append((m.start(), adj, noun))
    hits.sort(key=lambda item: item[0])

    # Group by noun so that "dense" + "painterly" become one phrase.
    order: List[str] = []
    adjectives: Dict[str, List[str]] = {}
    for _pos, adj, noun in hits:
        if noun not in adjectives:
            adjectives[noun] = []
            order.append(noun)
        if adj and adj not in adjectives[noun]:
            adjectives[noun].append(adj)

    phrases: List[str] = []
    for noun in order:
        adjs = adjectives[noun][:3]
        phrase = " ".join(adjs + [noun]).strip()
        if phrase and phrase not in phrases:
            phrases.append(phrase)
    return phrases[:limit]


# --------------------------------------------------------------------------- #
# Is the caption safe to use verbatim in a removal prompt?
# --------------------------------------------------------------------------- #
# Ink / brush vocabulary - this is what produced the artefacts of round 1.
_REMOVAL_UNSAFE_INK_RE = re.compile(
    r"\b(?:ink|brush\w*|stroke\w*|splatter\w*|splash\w*|droplet\w*|wash(?:es)?|"
    r"streak\w*|marker|graffiti|calligraph\w*)\b",
    re.IGNORECASE,
)

# People / body parts - this is what made the diffusion repaint a character.
_REMOVAL_UNSAFE_SUBJECT_RE = re.compile(
    r"\b(?:character\w*|figures?|persons?|people|torso|body|bodies|faces?|heads?|hair|"
    r"clothing|costumes?|garments?|shoulders?|arms?|hands?|legs?|eyes?|skin|"
    r"silhouettes?|humanoid\w*|portraits?|boys?|girls?|women|men|samurai|ninja|"
    r"robots?|creatures?|monsters?|anime|manga)\b",
    re.IGNORECASE,
)

# Flatness claims - this is what produced the flat colour blob of round 2.
_REMOVAL_UNSAFE_FLAT_RE = re.compile(
    r"\b(?:completely flat|uniform|solid (?:colour|color)|flat,?\s+smooth|even,?\s+uniform)\b",
    re.IGNORECASE,
)

# Harshness / style labels - "gritty, high-contrast ink style" is exactly what
# produced the noisy torso of round 3.
_REMOVAL_UNSAFE_STYLE_RE = re.compile(
    r"\b(?:gritty|harsh|stark|dramatic|chaotic|violent|high[- ]contrast|raw)\b",
    re.IGNORECASE,
)

# Colour families, used to check that the caption agrees with the measurement.
_COLOUR_FAMILIES: Dict[str, str] = {
    "red": "red", "crimson": "red", "scarlet": "red", "burgundy": "red",
    "maroon": "red", "ruby": "red", "wine": "red", "rust": "red", "vermilion": "red",
    "orange": "orange", "amber": "orange", "ochre": "orange", "copper": "orange",
    "yellow": "yellow", "gold": "yellow", "lemon": "yellow", "mustard": "yellow",
    "green": "green", "olive": "green", "emerald": "green", "jade": "green",
    "lime": "green", "mint": "green", "forest": "green",
    "teal": "teal", "cyan": "teal", "turquoise": "teal", "aqua": "teal",
    "blue": "blue", "navy": "blue", "azure": "blue", "cobalt": "blue",
    "indigo": "blue", "sapphire": "blue",
    "purple": "purple", "violet": "purple", "magenta": "purple", "lavender": "purple",
    "pink": "pink", "rose": "pink",
    "brown": "brown", "tan": "brown", "beige": "brown", "sepia": "brown",
    "black": "neutral", "white": "neutral", "grey": "neutral", "gray": "neutral",
    "silver": "neutral", "charcoal": "neutral",
}

_CAPTION_COLOUR_RE = re.compile(
    r"\b(" + "|".join(sorted(_COLOUR_FAMILIES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def colour_family(name: str) -> str:
    """Colour family of a measured or captioned colour name."""
    words = re.findall(r"[a-z]+", (name or "").lower())
    for word in words:
        family = _COLOUR_FAMILIES.get(word)
        if family:
            return family
    return "neutral"


def caption_is_removal_safe(caption: str) -> bool:
    """True when the caption may be used verbatim in a removal prompt."""
    text = caption or ""
    if not text.strip():
        return False
    for pattern in (
        _REMOVAL_UNSAFE_INK_RE,
        _REMOVAL_UNSAFE_SUBJECT_RE,
        _REMOVAL_UNSAFE_FLAT_RE,
        _REMOVAL_UNSAFE_STYLE_RE,
    ):
        if pattern.search(text):
            return False
    return True


def caption_colour_conflicts(caption: str, measured_name: str) -> bool:
    """True when the caption names a colour family other than the measured one.

    A caption that describes a *different* part of the panel ("deep burgundy-red
    mottled texture near the top") would make the diffusion paint that colour
    into the hole, so such a caption is rejected.
    """
    measured_family = colour_family(measured_name)
    if measured_family == "neutral":
        return False
    for match in _CAPTION_COLOUR_RE.finditer(caption or ""):
        word = match.group(1).lower()
        family = _COLOUR_FAMILIES.get(word, "neutral")
        if family not in (measured_family, "neutral"):
            return True
    return False


# --------------------------------------------------------------------------- #
# Measured surface profile
# --------------------------------------------------------------------------- #
def _mask_bbox(mask_array: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask_array > 0.05)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def measure_surface(
    image: Image.Image,
    mask_array: np.ndarray,
    *,
    band_px: int = 8,
) -> Dict[str, Any]:
    """Measure the surface **immediately around** the masked area.

    The fill continues the pixels that touch the mask, not the whole
    neighbourhood: on the user's failing case the mask sat on a dark navy
    character torso, while a wide ring around it also contained red splatter and
    white paper - and the dominant colour of *that* ring was red, which made
    the measured prompt ("deep crimson red surface") describe the wrong surface.
    """
    rgb = image.convert("RGB")
    width, height = rgb.size
    mask = np.clip(np.asarray(mask_array, dtype=np.float32), 0.0, 1.0)
    if mask.shape[:2] != (height, width):
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode="L").resize(
            (width, height), Image.Resampling.NEAREST
        )
        mask = np.asarray(mask_pil, dtype=np.float32) / 255.0

    bbox = _mask_bbox(mask)
    if bbox is None:
        return {
            "primary": "neutral grey",
            "primary_hex": "#808080",
            "color_names": ["neutral grey"],
            "brightness": 0.5,
            "saturation": 0.0,
            "brightness_std": 0.0,
            "edge_density": 0.0,
            "pixel_count": 0,
        }

    pad = max(24, band_px * 3)
    x0 = max(0, bbox[0] - pad)
    y0 = max(0, bbox[1] - pad)
    x1 = min(width, bbox[2] + pad)
    y1 = min(height, bbox[3] + pad)

    crop = rgb.crop((x0, y0, x1, y1))
    crop_mask = mask[y0:y1, x0:x1]
    mask_bin_pil = Image.fromarray(((crop_mask > 0.05).astype(np.uint8) * 255), mode="L")
    # Two MaxFilter(9) passes ~= 8 px of dilation.
    dilated = mask_bin_pil.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.MaxFilter(9))
    dilated_arr = np.asarray(dilated, dtype=np.float32) / 255.0
    band = ((dilated_arr > 0.5) & (crop_mask <= 0.05)).astype(np.float32)

    if float(band.sum()) < 32.0:
        # The mask covers everything around it - fall back to the whole crop.
        band = np.ones_like(crop_mask, dtype=np.float32)

    band_pil = Image.fromarray((band * 255).astype(np.uint8), mode="L")
    small_band = np.asarray(band_pil.resize((64, 64), Image.Resampling.NEAREST), dtype=np.uint8) > 32
    if int(small_band.sum()) < 8:
        small_band = np.ones((64, 64), dtype=bool)

    small_rgb = np.asarray(crop.resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    cmax = small_rgb.max(axis=2)
    cmin = small_rgb.min(axis=2)
    delta = cmax - cmin
    sat_map = np.where(cmax > 1e-5, delta / np.maximum(cmax, 1e-5), 0.0)
    luma_map = 0.299 * small_rgb[:, :, 0] + 0.587 * small_rgb[:, :, 1] + 0.114 * small_rgb[:, :, 2]

    band_luma = luma_map[small_band]
    band_sat = sat_map[small_band]

    # Edge density of the BAND, measured at full resolution (a thumbnail would
    # include the mask border itself and drown the real surface texture).
    crop_arr = np.asarray(crop, dtype=np.float32) / 255.0
    luma_full = 0.299 * crop_arr[:, :, 0] + 0.587 * crop_arr[:, :, 1] + 0.114 * crop_arr[:, :, 2]
    gx = np.abs(np.diff(luma_full, axis=1, prepend=luma_full[:, :1]))
    gy = np.abs(np.diff(luma_full, axis=0, prepend=luma_full[:1, :]))
    grad_full = np.hypot(gx, gy)
    band_pixels_full = band > 0.5
    edge_density = (
        float((grad_full[band_pixels_full] > 0.12).mean()) if bool(band_pixels_full.any()) else 0.0
    )

    # Robust spread: a bright fringe (e.g. the white outline of the removed
    # glyph) must not turn a smooth surface into a "mottled" one.
    if band_luma.size:
        median = float(np.median(band_luma))
        mad = float(np.median(np.abs(band_luma - median)))
        robust_std = 1.4826 * mad
    else:
        robust_std = 0.0

    # Dominant colour of the band, measured on the FULL-resolution band pixels
    # (the 64x64 thumbnail is too coarse for a narrow band).
    band_full = np.asarray(crop, dtype=np.uint8)[band > 0.5]
    hex_colors: List[str] = []
    if band_full.shape[0] > 0:
        # Cluster the band pixels only - the dominant colour of the band is the
        # surface that has to continue into the hole.
        buckets = (band_full // 24).astype(np.int32)
        keys = (buckets[:, 0] << 8) | (buckets[:, 1] << 4) | buckets[:, 2]
        uniq, counts = np.unique(keys, return_counts=True)
        order = np.argsort(-counts)
        hex_colors = []
        min_count = max(1, int(band_full.shape[0] * 0.05))
        for idx in order:
            if len(hex_colors) >= 4:
                break
            if counts[idx] < min_count and hex_colors:
                continue
            cluster = band_full[keys == uniq[idx]]
            mean_rgb = tuple(int(round(float(v))) for v in cluster.mean(axis=0))
            hex_val = rgb_to_hex(mean_rgb)
            if hex_val not in hex_colors:
                hex_colors.append(hex_val)

    color_names: List[str] = []
    for hex_color in hex_colors:
        cname = rgb_to_color_name(hex_to_rgb(hex_color))
        if cname not in color_names:
            color_names.append(cname)

    primary = color_names[0] if color_names else "neutral grey"
    if hex_colors:
        primary = _refine_color_name(primary, hex_colors[0])

    return {
        "primary": primary,
        "primary_hex": hex_colors[0] if hex_colors else "#808080",
        "color_names": color_names,
        "brightness": round(float(band_luma.mean()) if band_luma.size else 0.0, 4),
        "saturation": round(float(band_sat.mean()) if band_sat.size else 0.0, 4),
        "brightness_std": round(float(band_luma.std()) if band_luma.size else 0.0, 4),
        "robust_std": round(float(robust_std), 4),
        "edge_density": round(edge_density, 4),
        "pixel_count": int(band.sum()),
    }


def build_surface_profile(measurement: Dict[str, Any]) -> Dict[str, Any]:
    """Turn measured band statistics into the profile used by the removal prompt."""
    measurement = measurement or {}
    primary = str(measurement.get("primary") or "neutral grey")

    try:
        brightness_std = float(measurement.get("brightness_std") or 0.0)
    except (TypeError, ValueError):
        brightness_std = 0.0
    try:
        robust_std = float(measurement.get("robust_std", brightness_std))
    except (TypeError, ValueError):
        robust_std = brightness_std
    try:
        edge_density = float(measurement.get("edge_density") or 0.0)
    except (TypeError, ValueError):
        edge_density = 0.0

    # Texture class, measured - never guessed from the caption.  The robust
    # spread is used so that a bright fringe around the mask does not make a
    # smooth surface look mottled.
    spread = max(robust_std, 0.0)
    if edge_density >= 0.28:
        texture = "grainy"
    elif spread >= 0.10:
        texture = "mottled"
    elif spread >= 0.035:
        texture = "softly mottled"
    else:
        texture = "smooth"

    phrase = f"{texture} {primary} surface"

    return {
        "primary": primary,
        "primary_hex": measurement.get("primary_hex"),
        "color_names": measurement.get("color_names") or [],
        "texture": texture,
        "brightness": measurement.get("brightness"),
        "saturation": measurement.get("saturation"),
        "brightness_std": measurement.get("brightness_std"),
        "edge_density": measurement.get("edge_density"),
        "phrase": phrase,
    }


def build_removal_prompt(
    profile: Dict[str, Any],
    caption: str = "",
    *,
    flat_fill: bool = False,
) -> str:
    """Build the positive diffusion prompt for a removal / inpaint-fill task.

    * A caption that is safe (no ink, no people, no flatness claim, no harsh
      style label) and agrees with the measured colour is used **verbatim** -
      it carries the texture detail the diffusion model needs.
    * Otherwise the prompt falls back to the **measured** surface phrase plus
      whitelisted surface wording: a hallucinated subject can then never reach
      the diffusion model.
    * The prompt always asks for a continuation of the surrounding surface.
    """
    profile = profile or {}
    primary = str(profile.get("primary") or "neutral grey")
    phrase = str(profile.get("phrase") or f"smooth {primary} surface")

    if flat_fill:
        # Explicitly requested uniform colour patch (opt-in checkbox).
        return (
            f"even, uniform {primary} fill patch, even tone, clean edges, "
            "matching the surrounding colour"
        )

    clean = (caption or "").strip()
    clean = _SURFACE_MARKER_RE.sub("", clean).strip().strip(",").strip()

    use_caption = (
        bool(clean)
        and caption_is_removal_safe(clean)
        and not caption_colour_conflicts(clean, primary)
    )

    if use_caption:
        body = clean
    else:
        parts: List[str] = [phrase]
        vocabulary = extract_surface_vocabulary(clean)
        if vocabulary:
            parts.append(", ".join(vocabulary))
        body = ", ".join(parts)

    if not _SURFACE_MARKER_RE.search(body):
        body = f"{body.rstrip(' .,;')}, {SURFACE_CONTINUATION_MARKER}"
    return body
