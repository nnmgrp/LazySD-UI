"""Regression tests for the "Ink Brush Artefacts" fix.

Background: the diffusion model painted black ink splatter over the masked area
because the VLM prompt literally asked for it.  Root cause analysis and the
full list of gaps are documented in ``docs/INPAINT_INK_BRUSH_INVESTIGATION.md``.

Test IDs map to the acceptance criteria of that document:
  T-1  removal prompt is stripped of ink/brush vocabulary
  T-2  non-removal prompts are NOT modified (gap G2)
  T-3  removal detection gates the artifact stripping
  T-4  negative prompt gains the ink suppression tokens
  T-5  non-text removal does NOT gain ink suppression tokens
  T-6  prose VLM answers are sanitized too (gap G3)
  T-7  no prompt template describes brush/ink artwork (gap G1)
  T-8  the heuristic fallback is removal-aware (gap G5)
  T-9  the prompt cache is invalidated by a template version bump (gap G4)
  T-10 end-to-end through POST /api/inpaint/prompt
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

import app as app_module
from backend import prompt_gen as pg
from backend.prompt_fallback import (
    MANDATORY_INK_NEGATIVES,
    analyze_region,
    generate_heuristic_prompt,
)
from backend.vision_runtime import (
    INPAINT_PUP_REMOVAL_SYSTEM_PROMPT,
    INPAINT_PUP_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    _build_inpaint_request,
    _is_text_sfx_removal_instruction,
    _sanitize_negative_prompt,
    _sanitize_positive_prompt,
    parse_vlm_response_text,
)

# The exact prompt that produced the incident (outputs/nunchaku_20261103_031830_1793657910.json).
INCIDENT_PROMPT = (
    "A seamless dark red and black ink-splash background filling the entire area, with dense "
    "crimson washes, irregular black brush strokes, dry-brush streaks, scattered droplets, and "
    "rough manga-style splatter matching the surrounding artwork. The central region contains "
    "only the continuous painted texture, with no lettering, symbols, characters, masks, or "
    "foreground objects."
)

# Words that describe the REMOVED element.  "texture" is deliberately NOT in this
# list: after the prompt reframe the model describes the surrounding artwork and
# its texture words are exactly what the diffusion needs to continue that surface.
INK_WORDS = ("brush", "ink", "splatter", "streak", "droplet", "wash")


def _assert_no_ink_words(text: str) -> None:
    low = text.lower()
    for word in INK_WORDS:
        assert word not in low, f"'{word}' still present in: {text!r}"


# --------------------------------------------------------------------------- T-1
def test_t1_removal_prompt_is_stripped_of_ink_vocabulary():
    cleaned = _sanitize_positive_prompt(INCIDENT_PROMPT, is_removal=True)
    _assert_no_ink_words(cleaned)
    # The surrounding colour is still described, and the continuation marker is added.
    assert "dark red" in cleaned.lower()
    assert "seamless continuation of the surrounding surface" in cleaned.lower()


def test_t1_sanitizer_is_idempotent():
    once = _sanitize_positive_prompt(INCIDENT_PROMPT, is_removal=True)
    assert _sanitize_positive_prompt(once, is_removal=True) == once


def test_t1_removal_prompt_keeps_the_continuation_marker_once():
    cleaned = _sanitize_positive_prompt(INCIDENT_PROMPT, is_removal=True)
    assert cleaned.lower().count("seamless continuation of the surrounding surface") == 1


def test_t1_removal_prompt_does_not_demand_a_flat_fill():
    """A flat colour patch reads as an obvious blob - the prompt must ask for a continuation."""
    cleaned = _sanitize_positive_prompt(INCIDENT_PROMPT, is_removal=True)
    low = cleaned.lower()
    for word in ("completely flat", "uniform colour field", "solid color fill", "solid colour fill"):
        assert word not in low


# --------------------------------------------------------------------------- T-2
def test_t2_non_removal_prompts_are_not_modified():
    legit = [
        "neon light streaks across a rainy city skyline",
        "water splashes on the shore, golden sunset",
        "ink wash painting of a mountain landscape",
        "manga screentone and ink lineart texture",
    ]
    for prompt in legit:
        assert _sanitize_positive_prompt(prompt, is_removal=False) == prompt


# --------------------------------------------------------------------------- T-3
def test_t3_artifact_stripping_is_gated_on_removal():
    with_ink = "dark background with black brush strokes"
    assert _sanitize_positive_prompt(with_ink, is_removal=False) == with_ink
    _assert_no_ink_words(_sanitize_positive_prompt(with_ink, is_removal=True))


def test_t3_text_sfx_removal_detection():
    assert _is_text_sfx_removal_instruction("delete this Japanese SFX")
    assert _is_text_sfx_removal_instruction("убрать японский текст")
    assert _is_text_sfx_removal_instruction("удали надпись")
    assert not _is_text_sfx_removal_instruction("remove the car")
    assert not _is_text_sfx_removal_instruction("replace with a blue crystal")


# --------------------------------------------------------------------------- T-4
def test_t4_negative_prompt_gains_ink_suppression_tokens():
    neg = _sanitize_negative_prompt("text, blurry", instruction="delete this Japanese SFX")
    for token in MANDATORY_INK_NEGATIVES:
        assert token in neg.lower(), f"missing negative token {token!r} in {neg!r}"
    assert len(neg) <= 500


def test_t4_negative_prompt_has_no_duplicates():
    neg = _sanitize_negative_prompt("text, blurry", instruction="убрать японский текст")
    parts = [p.strip().lower() for p in neg.split(",")]
    assert len(parts) == len(set(parts))


# --------------------------------------------------------------------------- T-5
def test_t5_every_removal_gains_the_ink_suppression_tokens():
    """Ink suppression is applied to EVERY removal task.

    It used to require an explicit text/SFX keyword, so an instruction phrased
    differently ("remove this", "удали это") silently lost the protection and
    the ink artefacts came back.
    """
    for instruction in ("remove the car", "удали это", "delete this Japanese SFX"):
        neg = _sanitize_negative_prompt("text, blurry", instruction=instruction)
        for token in MANDATORY_INK_NEGATIVES:
            assert token in neg.lower(), f"missing {token!r} for {instruction!r} in {neg!r}"


# --------------------------------------------------------------------------- T-6
def test_t6_prose_answer_is_sanitized_and_gets_mandatory_negatives():
    prompt, neg = parse_vlm_response_text(INCIDENT_PROMPT, instruction="delete this Japanese SFX")
    _assert_no_ink_words(prompt)
    assert "ink" in neg.lower()
    assert "brush strokes" in neg.lower()


def test_t6_prose_answer_without_instruction_is_left_alone():
    prompt, _ = parse_vlm_response_text("seamless white speech bubble interior, no text")
    assert "white speech bubble" in prompt


# --------------------------------------------------------------------------- T-7
def test_t7_no_template_describes_brush_or_ink_artwork():
    # Prohibition lists are allowed; descriptive phrases are not.
    forbidden_phrases = (
        "brush texture",
        "brushwork",
        "manga/anime brush",
        "manga brush",
        "and brush",
        "art texture",
        "painted texture",
    )
    templates = {
        "SYSTEM_PROMPT": SYSTEM_PROMPT,
        "INPAINT_PUP_SYSTEM_PROMPT": INPAINT_PUP_SYSTEM_PROMPT,
        "INPAINT_PUP_REMOVAL_SYSTEM_PROMPT": INPAINT_PUP_REMOVAL_SYSTEM_PROMPT,
    }
    for name, text in templates.items():
        low = text.lower()
        for phrase in forbidden_phrases:
            assert phrase not in low, f"{name} still contains {phrase!r}"


def test_t7_removal_system_prompt_forbids_ink_words():
    low = INPAINT_PUP_REMOVAL_SYSTEM_PROMPT.lower()
    for word in ("brush", "ink", "splatter"):
        assert word in low, "removal system prompt must forbid the word explicitly"


def test_t7_removal_system_prompt_treats_the_mask_as_a_placeholder():
    low = INPAINT_PUP_REMOVAL_SYSTEM_PROMPT.lower()
    assert "placeholder" in low
    assert "around the highlighted area" in low
    # It must NOT tell the model to produce a flat / uniform colour field.
    assert "completely flat, clean, smooth, uniform" not in low


def test_t7_inpaint_request_has_no_ink_vocabulary():
    for is_removal in (True, False):
        for instruction in ("", "delete this Japanese SFX", "replace with a blue crystal"):
            req = _build_inpaint_request(
                instruction,
                "mauve, dark grey",
                is_removal=is_removal,
                flat_fill=False,
            )
            low = req.lower()
            for phrase in ("brush texture", "manga/anime brush", "brushwork", "art texture"):
                assert phrase not in low, f"request contains {phrase!r}: {req!r}"


def test_t7_removal_request_asks_for_the_surrounding_surface():
    req = _build_inpaint_request(
        "delete this Japanese SFX",
        "dark red, red",
        is_removal=True,
        flat_fill=False,
    )
    low = req.lower()
    assert "surrounding the highlighted area" in low
    assert "placeholder" in low
    # It must not ask the model to describe a subject.
    assert "characters" in low
    # The default must NOT demand a flat patch.
    assert "flat" not in low
    assert "uniform" not in low


# --------------------------------------------------------------------------- T-8
def _make_mask(width: int, height: int, box: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    x0, y0, x1, y1 = box
    mask[y0:y1, x0:x1] = 1.0
    return mask


def test_t8_heuristic_fallback_is_removal_aware():
    img = Image.new("RGB", (256, 256), (185, 25, 35))
    draw = ImageDraw.Draw(img)
    for y in range(40, 220, 10):
        draw.line((40, y, 220, y + 5), fill=(15, 10, 12), width=4)

    box = (80, 80, 176, 176)
    mask = _make_mask(256, 256, box)
    res = generate_heuristic_prompt(img, mask, box, instruction="убрать японский звук")

    assert res["analysis"]["region_type"] == "colored_art"
    # The fill must continue the surrounding surface, not be a flat colour patch.
    assert "seamless continuation of the surrounding" in res["prompt"]
    assert "manga brush" not in res["prompt"]
    assert "ink lineart texture" not in res["prompt"]
    assert "flat" not in res["prompt"]
    for token in ("ink", "brush strokes", "splatter"):
        assert token in res["negative_prompt"].lower()


def test_t8_heuristic_keeps_texture_for_non_removal_tasks():
    img = Image.new("RGB", (256, 256), (185, 25, 35))
    draw = ImageDraw.Draw(img)
    for y in range(40, 220, 10):
        draw.line((40, y, 220, y + 5), fill=(15, 10, 12), width=4)

    box = (80, 80, 176, 176)
    mask = _make_mask(256, 256, box)
    res = generate_heuristic_prompt(img, mask, box, instruction="восстановить фон")
    assert "manga brush and illustration texture" in res["prompt"]


def test_t8_heuristic_white_bubble_branch_is_preserved():
    img = Image.new("RGB", (256, 256), (250, 250, 250))
    draw = ImageDraw.Draw(img)
    draw.rectangle((110, 110, 146, 146), fill=(20, 20, 20))
    box = (96, 96, 160, 160)
    mask = _make_mask(256, 256, box)
    res = generate_heuristic_prompt(img, mask, box, instruction="убрать японский текст")
    assert res["analysis"]["region_type"] == "white_bubble"
    assert "white speech bubble" in res["prompt"]


# --------------------------------------------------------------------------- T-9
def test_t9_cache_key_contains_the_template_version():
    img = Image.new("RGB", (64, 64), (100, 100, 100))
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[10:30, 10:30] = 1.0
    kwargs = dict(
        resolved_source=Path(__file__).resolve(),
        mask_array=mask,
        instruction="убрать текст",
        vision_model="qwen2.5vl:3b",
        crop_mode="auto",
        crop_box=(0, 0, 64, 64),
        crop_padding=64,
        use_vlm=True,
    )
    key = pg._compute_cache_key(**kwargs)
    # The version is part of the hashed key material.
    other = pg.PROMPT_TEMPLATE_VERSION + "-x"
    original = pg.PROMPT_TEMPLATE_VERSION
    try:
        pg.PROMPT_TEMPLATE_VERSION = other
        changed = pg._compute_cache_key(**kwargs)
    finally:
        pg.PROMPT_TEMPLATE_VERSION = original
    assert key != changed, "cache key must change when the prompt template version changes"


def test_t9_flat_fill_is_part_of_the_cache_key():
    img = Image.new("RGB", (64, 64), (100, 100, 100))
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[10:30, 10:30] = 1.0
    kwargs = dict(
        resolved_source=Path(__file__).resolve(),
        mask_array=mask,
        instruction="убрать текст",
        vision_model="qwen2.5vl:3b",
        crop_mode="auto",
        crop_box=(0, 0, 64, 64),
        crop_padding=64,
        use_vlm=True,
    )
    assert pg._compute_cache_key(**kwargs, flat_fill=True) != pg._compute_cache_key(**kwargs, flat_fill=False)


# --------------------------------------------------------------------------- T-10
def _make_mask_data_url(width: int = 128, height: int = 128) -> str:
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    for x in range(32, 96):
        for y in range(32, 96):
            img.putpixel((x, y), (255, 255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def test_t10_api_end_to_end_removal_prompt():
    pg.clear_prompt_cache()
    root = Path(app_module.__file__).resolve().parent
    uploads = root / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    sample = uploads / "_test_ink_fix_source.png"
    Image.new("RGB", (128, 128), (185, 30, 40)).save(sample, "PNG")
    try:
        client = TestClient(app_module.create_app())
        resp = client.post("/api/inpaint/prompt", json={
            "source_path": "uploads/_test_ink_fix_source.png",
            "mask": _make_mask_data_url(),
            "instruction": "убрать японский текст",
            "use_vlm": False,
            "vision_provider": "heuristic",
            "flat_fill": True,
        })
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["source"] == "heuristic"
        _assert_no_ink_words(data["prompt"])
        assert "ink" in data["negative_prompt"].lower()
    finally:
        sample.unlink(missing_ok=True)


def test_t10_api_accepts_the_flat_fill_flag():
    pg.clear_prompt_cache()
    root = Path(app_module.__file__).resolve().parent
    uploads = root / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    sample = uploads / "_test_ink_fix_source2.png"
    Image.new("RGB", (128, 128), (185, 30, 40)).save(sample, "PNG")
    try:
        client = TestClient(app_module.create_app())
        for flat_fill in (True, False):
            resp = client.post("/api/inpaint/prompt", json={
                "source_path": "uploads/_test_ink_fix_source2.png",
                "mask": _make_mask_data_url(),
                "instruction": "убрать японский текст",
                "use_vlm": False,
                "vision_provider": "heuristic",
                "flat_fill": flat_fill,
            })
            assert resp.status_code == 200, resp.text
            assert resp.json()["source"] == "heuristic"
    finally:
        sample.unlink(missing_ok=True)


# --------------------------------------------------------------------------- T-11
def test_t11_vlm_input_hides_the_removed_element():
    """The VLM must NOT see the raw SFX: under the red highlight it reads as a flat
    red shape and the model then describes *that* instead of the background."""
    from backend.prompt_gen import build_vlm_images

    # Dark red textured background with a bright white glyph in the middle.
    img = Image.new("RGB", (256, 256), (150, 25, 40))
    draw = ImageDraw.Draw(img)
    for y in range(0, 256, 9):
        draw.line((0, y, 256, y + 4), fill=(90, 12, 25), width=3)
    draw.ellipse((80, 80, 176, 176), fill=(240, 240, 240))

    mask = np.zeros((256, 256), dtype=np.float32)
    mask[80:176, 80:176] = 1.0

    _, crop_b64 = build_vlm_images(img, mask, (0, 0, 256, 256), max_dim=256)
    crop = Image.open(io.BytesIO(base64.b64decode(crop_b64))).convert("RGB")
    arr = np.asarray(crop).astype(np.float32)

    # The white glyph must be gone: no near-white pixels may remain inside the mask.
    m = np.asarray(
        Image.fromarray((mask * 255).astype(np.uint8)).resize(crop.size, Image.Resampling.NEAREST)
    ).astype(np.float32) / 255.0
    inside = arr[m > 0.9]
    assert inside.size > 0
    white = inside[(inside[:, 0] > 200) & (inside[:, 1] > 200) & (inside[:, 2] > 200)]
    assert white.shape[0] == 0, "the removed element is still visible to the VLM"

    # The placeholder must stay close to the surrounding colour, not turn into a
    # flat saturated patch of its own.
    ring = arr[m <= 0.05]
    assert ring.size > 0
    colour_delta = abs(inside.mean(axis=0) - ring.mean(axis=0)).max()
    assert colour_delta < 90, f"placeholder colour drifts from the background: {colour_delta:.1f}"


def test_t11_heuristic_prompt_has_no_flat_wording():
    img = Image.new("RGB", (256, 256), (185, 25, 35))
    draw = ImageDraw.Draw(img)
    for y in range(40, 220, 10):
        draw.line((40, y, 220, y + 5), fill=(15, 10, 12), width=4)
    box = (80, 80, 176, 176)
    res = generate_heuristic_prompt(img, _make_mask(256, 256, box), box, instruction="убрать текст")
    assert "seamless continuation" in res["prompt"]
    assert "uniform" not in res["prompt"]


# --------------------------------------------------------------------------- T-12
# Round 3: FLUX 3 PUP is a captioner, not a removal assistant.  It captioned
# the whole crop (including the character next to the mask), ignored the
# "at most 12 words" rule and its hallucinated subject was copied into the
# diffusion prompt - which then literally painted "a character-like torso in
# gritty high-contrast ink style with crimson brushwork".
USER_FAILING_CAPTION = (
    "A deep black and burgundy character-like torso rendered in a gritty, high-contrast "
    "ink style, with irregular crimson brushwork spreading naturally across the torso, "
    "a narrow red vertical form near the top centre, and pale grey-white negative space."
)


def _profile_for(color: str = "dark navy blue", texture: str = "smooth") -> dict:
    return {
        "primary": color,
        "color_names": [color],
        "texture": texture,
        "brightness": 0.2,
        "saturation": 0.4,
        "brightness_std": 0.03,
        "edge_density": 0.05,
        "region_type": "colored_art",
        "phrase": f"{texture} {color} surface",
    }


def test_t12_removal_prompt_drops_the_hallucinated_subject():
    from backend.surface_profile import build_removal_prompt

    prompt = build_removal_prompt(_profile_for(), USER_FAILING_CAPTION)
    low = prompt.lower()
    for banned in (
        "character",
        "torso",
        "gritty",
        "high-contrast",
        "ink",
        "brushwork",
        "brush",
        "foliage",
        "crimson brushwork",
    ):
        assert banned not in low, f"{banned!r} leaked into the removal prompt: {prompt!r}"
    # The measured surface is the source of truth.
    assert "dark navy blue" in low
    assert "seamless continuation of the surrounding surface" in low


def test_t12_surface_vocabulary_keeps_only_surface_wording():
    from backend.surface_profile import extract_surface_vocabulary

    vocabulary = extract_surface_vocabulary(USER_FAILING_CAPTION)
    joined = " ".join(vocabulary).lower()
    for banned in ("character", "torso", "gritty", "ink", "brush", "crimson", "burgundy"):
        assert banned not in joined, f"{banned!r} leaked into {vocabulary!r}"
    # ... but it still keeps the useful texture wording.
    assert extract_surface_vocabulary("a dense painterly field of crimson") == [
        "dense painterly texture"
    ]


def test_t12_removal_prompt_is_short_and_asks_for_a_continuation():
    from backend.surface_profile import build_removal_prompt

    prompt = build_removal_prompt(_profile_for(), USER_FAILING_CAPTION)
    assert len(prompt.split()) <= 30, prompt
    assert "seamless continuation" in prompt
    assert "completely flat" not in prompt.lower()


def test_t12_flat_fill_prompt_is_uniform_and_ink_free():
    from backend.surface_profile import build_removal_prompt

    prompt = build_removal_prompt(_profile_for(), USER_FAILING_CAPTION, flat_fill=True)
    low = prompt.lower()
    assert "uniform" in low
    assert "dark navy blue" in low
    for banned in ("ink", "brush", "character", "torso"):
        assert banned not in low


def test_t12_measured_profile_uses_the_band_next_to_the_mask():
    """The fill continues the pixels that TOUCH the mask, not the whole panel."""
    from backend.surface_profile import build_surface_profile, measure_surface

    # Navy torso in the middle, red splatter far away from the mask.
    img = Image.new("RGB", (256, 256), (160, 20, 30))
    draw = ImageDraw.Draw(img)
    draw.rectangle((60, 60, 196, 196), fill=(1, 3, 32))
    mask = np.zeros((256, 256), dtype=np.float32)
    mask[100:150, 100:150] = 1.0

    profile = build_surface_profile(measure_surface(img, mask))
    assert profile["primary"] == "dark navy blue", profile
    assert profile["phrase"] == "smooth dark navy blue surface", profile


def test_t12_dark_saturated_colour_is_not_called_black():
    from backend.surface_profile import rgb_to_color_name

    # A very dark but saturated navy must not be reported as "black" - the fill
    # would then be painted in the wrong colour.
    assert rgb_to_color_name((1, 3, 32)) == "dark blue"
    assert rgb_to_color_name((10, 10, 10)) == "black"
    assert rgb_to_color_name((250, 250, 250)) == "white"


def test_t12_measured_profile_is_robust_to_a_bright_fringe():
    """The white outline of a removed glyph must not make the surface 'mottled'."""
    from backend.surface_profile import build_surface_profile, measure_surface

    img = Image.new("RGB", (256, 256), (1, 3, 32))
    draw = ImageDraw.Draw(img)
    draw.rectangle((102, 102, 154, 154), fill=(250, 250, 250))  # thin glyph fringe
    mask = np.zeros((256, 256), dtype=np.float32)
    mask[104:152, 104:152] = 1.0

    profile = build_surface_profile(measure_surface(img, mask))
    assert profile["primary"] == "dark navy blue"
    assert profile["texture"] == "smooth", profile


# --------------------------------------------------------------------------- T-13
def test_t13_pup_caption_budget_is_not_truncating():
    from backend.vision_runtime import PUP_MAX_NEW_TOKENS

    # 90 tokens cut the caption mid-sentence ("The central region continues as").
    assert PUP_MAX_NEW_TOKENS >= 160


# --------------------------------------------------------------------------- T-14
def test_t14_heuristic_removal_prompt_is_measured_not_hinted():
    img = Image.new("RGB", (256, 256), (185, 25, 35))
    draw = ImageDraw.Draw(img)
    for y in range(40, 220, 10):
        draw.line((40, y, 220, y + 5), fill=(15, 10, 12), width=4)

    box = (80, 80, 176, 176)
    res = generate_heuristic_prompt(img, _make_mask(256, 256, box), box, instruction="убрать брызги")
    low = res["prompt"].lower()
    assert "ink splash" not in low
    assert "seamless continuation of the surrounding surface" in low
    # Ink suppression always applies to a removal task.
    for token in ("ink", "brush strokes", "splatter"):
        assert token in res["negative_prompt"].lower()


# --------------------------------------------------------------------------- T-15
SAFE_CAPTION = (
    "A continuous deep-crimson textured surface fills the entire image. The surface has "
    "dense, irregular dark red and burgundy mottling, fine grain, scattered near-black "
    "maroon flecks, and uneven painted shading consistent with the surrounding artwork."
)


def test_t15_a_safe_caption_is_used_verbatim():
    """A correct, subject-free caption carries the texture detail the model needs."""
    from backend.surface_profile import build_removal_prompt

    profile = _profile_for(color="deep crimson", texture="softly mottled")
    prompt = build_removal_prompt(profile, SAFE_CAPTION)
    assert "near-black maroon flecks" in prompt
    assert "fine grain" in prompt
    assert "seamless continuation of the surrounding surface" in prompt


def test_t15_a_caption_with_a_conflicting_colour_is_rejected():
    """A caption describing another part of the panel must not paint into the hole."""
    from backend.surface_profile import build_removal_prompt

    profile = _profile_for(color="dark navy blue", texture="smooth")
    caption = (
        "A seamless continuation of the dark navy-blue surface fills the entire area, "
        "with deep burgundy-red mottled texture near the top and narrow white accents."
    )
    prompt = build_removal_prompt(profile, caption)
    assert "burgundy" not in prompt.lower()
    assert "dark navy blue" in prompt.lower()


def test_t15_caption_mentions_of_the_removed_element_are_stripped():
    from backend.vision_runtime import _sanitize_positive_prompt

    cleaned = _sanitize_positive_prompt(
        "A continuous crimson surface fills the image, extending seamlessly through the area "
        "formerly covered by the Japanese sound-effect placeholder. The surface has dense "
        "mottling and fine grain.",
        is_removal=True,
    )
    low = cleaned.lower()
    assert "placeholder" not in low
    assert "sound-effect" not in low
    assert "formerly covered" not in low
    # The useful part survives.
    assert "dense mottling" in low


def test_t15_negation_clauses_are_removed_from_the_positive_prompt():
    from backend.vision_runtime import _sanitize_positive_prompt

    cleaned = _sanitize_positive_prompt(
        "A continuous navy surface fills the composition with no figures, "
        "Japanese sound-effect graphics, red markings or other objects.",
        is_removal=True,
    )
    low = cleaned.lower()
    assert "no figures" not in low
    assert "sound-effect" not in low


# --------------------------------------------------------------------------- T-16
def test_t16_removal_prompt_is_rebuilt_for_every_removal_instruction():
    """The whole point of round 3: a removal prompt must not be a caption copy."""
    from backend.surface_profile import build_removal_prompt

    profile = _profile_for()
    prompt = build_removal_prompt(profile, USER_FAILING_CAPTION)
    low = prompt.lower()
    assert "torso" not in low
    assert "character" not in low
    assert "gritty" not in low
    assert "ink" not in low
    assert "brushwork" not in low


# --------------------------------------------------------------------------- T-17
def test_t17_heuristic_detects_english_delete_as_a_removal():
    """The heuristic fallback must agree with the VLM path on what a removal is.

    "delete this Japanese SFX" was not recognised as a removal by the heuristic,
    so it fell through to the generic branch and produced "manga brush and
    illustration texture" - the very ink trigger the whole fix removes.
    """
    img = Image.new("RGB", (256, 256), (185, 25, 35))
    draw = ImageDraw.Draw(img)
    for y in range(40, 220, 10):
        draw.line((40, y, 220, y + 5), fill=(15, 10, 12), width=4)

    box = (80, 80, 176, 176)
    for instruction in ("delete this Japanese SFX", "remove this SFX", "erase the sound effect"):
        res = generate_heuristic_prompt(
            img, _make_mask(256, 256, box), box, instruction=instruction
        )
        assert "seamless continuation of the surrounding surface" in res["prompt"], res["prompt"]
        assert "manga brush" not in res["prompt"]
        assert "ink" in res["negative_prompt"].lower()


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
