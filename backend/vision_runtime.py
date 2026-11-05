"""Runtime adapter for local Vision-Language Models (VLM) and the official
FLUX 3 Image Klein Prompt Upsampler (``unsloth/flux-3-image-klein-pup-GGUF`` &
``tonera/flux-3-image-klein-pup-nunchaku``).

Handles:
- Checking Ollama availability and version (``/api/version``, ``/api/tags``).
- Running ``tonera/flux-3-image-klein-pup-nunchaku`` natively via ``envs/extras``
  (``NunchakuQwen3_5ForConditionalGeneration``) in an isolated subprocess that
  releases 100% of VRAM upon completion.
- Running ``unsloth/flux-3-image-klein-pup-GGUF`` and general VLMs via Ollama
  ``/api/chat`` with ``keep_alive="0s"``.
- Downloading models either from Hugging Face (for Nunchaku SVDQ & GGUF) or via
  Ollama ``/api/pull`` with real-time progress tracking.
"""
from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

from .prompt_fallback import DEFAULT_NEGATIVE_PROMPT, MANDATORY_INK_NEGATIVES
from .surface_profile import (
    SURFACE_CONTINUATION_MARKER,
    build_removal_prompt,
    build_surface_profile,
)
from .vision_models import (
    clear_download_state,
    get_catalog_entry,
    get_download_state,
    is_local_vision_model_installed,
    set_download_state,
)

logger = logging.getLogger("quantfunc.vision_runtime")

SYSTEM_PROMPT = (
    "You are a prompt engineer for Stable Diffusion / FLUX Inpainting.\n"
    "In both images, the BRIGHT RED highlighted shapes mark the EXACT masked pixels to be repainted.\n"
    "CRITICAL RULES:\n"
    "1. Describe ONLY the background surface that should appear INSIDE the bright red masked shapes "
    "AFTER the edit.\n"
    "2. NEVER describe characters, faces, hair, hats, or clothing that are OUTSIDE the red mask, "
    "even if a character stands next to or between red masked shapes!\n"
    "3. If the user asks to delete/remove/clean text, SFX, sound effects, speech bubbles, or objects: "
    "do NOT mention the removed text/object and do NOT describe nearby characters. The red masked "
    "shapes are only an EMPTY PLACEHOLDER - describe the background surface AROUND them (its colour, "
    "tone, texture and lighting) and state that the same surface continues into them. Never describe "
    "the masked shapes themselves as a flat, smooth, uniform or solid colour field.\n"
    "4. negative_prompt MUST be a comma-separated list of forbidden artifacts (e.g. "
    '"japanese text, sfx, kanji, katakana, letters, speech bubble, watermark, blurry, seam"). '
    'NEVER write "no background", "no colors", "no details", "no style", "no characters", or words starting with "no ".\n'
    '5. Return ONLY valid JSON: {"prompt": "...", "negative_prompt": "..."} (prompt max 40 words).'
)

_LAST_ACTIVE_OLLAMA_MODEL: Optional[str] = None

# Token budget for the FLUX 3 PUP caption.  90 tokens truncated the caption
# mid-sentence ("The central region continues as") whenever the model wrote
# more than ~35 words, which produced half-finished prompts in the diffusion
# model.  160 tokens covers a 40-word caption without cutting it.
PUP_MAX_NEW_TOKENS = 160

_BAD_NEGATIVE_CONCEPTS = {
    "background",
    "backgrounds",
    "color",
    "colors",
    "colour",
    "colours",
    "detail",
    "details",
    "style",
    "styles",
    "character",
    "characters",
    "person",
    "people",
    "art",
    "artwork",
    "illustration",
    "image",
    "shading",
    "lighting",
    "texture",
    "textures",
    "quality",
    "high quality",
    "anime",
    "manga",
    "lineart",
    "lines",
    "scene",
}

_REMOVAL_KEYWORDS_RE = re.compile(
    r"\b(delete|remove|erase|clean|clear|strip|wi?thout|rid\s+of|sfx|text|kanji|katakana|hiragana|subtitle|bubble|watermark|"
    r"убрать|убери|удалить|удали|стереть|сотри|очистить|очисти|замазать|без\s+текста|текст|звук|баббл)\b",
    re.IGNORECASE,
)


def _is_removal_instruction(instruction: str) -> bool:
    return bool(_REMOVAL_KEYWORDS_RE.search(instruction or ""))


# Removal of *text / SFX* specifically (not an arbitrary object).  Only for these
# tasks do we suppress ink/brush artefacts in the negative prompt — forcing
# "no ink" on e.g. a character removal would fight legitimate line art.
_TEXT_SFX_RE = re.compile(
    r"\b(text|sfx|sound\s*effects?|kanji|katakana|hiragana|japanese|letters?|words?|writing|"
    r"captions?|subtitles?|speech\s*bubbles?|thought\s*bubbles?|bubbles?|watermarks?|signatures?|"
    r"текст\w*|надпис\w*|букв\w*|японск\w*|кана\w*|канзи\w*|звук\w*|сфх|бабл\w*|облачк\w*|"
    r"реплик\w*|водян\w*|подпис\w*)\b",
    re.IGNORECASE,
)


def _is_text_sfx_removal_instruction(instruction: str) -> bool:
    """True when the instruction removes text/SFX rather than an arbitrary object."""
    text = (instruction or "").strip()
    if not text or not _is_removal_instruction(text):
        return False
    return bool(_TEXT_SFX_RE.search(text))


# Ink/brush artefacts that must never be generated while filling a removed
# text/SFX area.  Shared with ``backend.prompt_fallback`` (single source of
# truth) — see docs/INPAINT_INK_BRUSH_INVESTIGATION.md.
_MANDATORY_INK_NEGATIVES = MANDATORY_INK_NEGATIVES


def _sanitize_negative_prompt(neg_raw: str, instruction: str = "") -> str:
    tokens: List[str] = []
    seen = set()
    for part in re.split(r"[,;\n]+", neg_raw or ""):
        t = part.strip().strip('."\'')
        if not t:
            continue
        t_clean = re.sub(r"^(?:no|without|not)\s+", "", t, flags=re.IGNORECASE).strip()
        if not t_clean:
            continue
        low = t_clean.lower()
        if low in _BAD_NEGATIVE_CONCEPTS or low in ("none", "n/a", "null", "nothing"):
            continue
        if low not in seen:
            seen.add(low)
            tokens.append(t_clean)

    if _is_removal_instruction(instruction):
        for req in ("text", "japanese sfx", "kanji", "katakana", "letters", "speech bubble", "watermark", "blurry", "visible seam"):
            if req.lower() not in seen:
                seen.add(req.lower())
                tokens.append(req)
        # Ink suppression for ANY removal task.  It used to require an explicit
        # text/SFX keyword, which silently skipped the tokens whenever the
        # instruction was phrased differently ("удали это", "remove this") and
        # let the harsh ink artefacts back in.
        for req in _MANDATORY_INK_NEGATIVES:
            if req.lower() not in seen:
                seen.add(req.lower())
                tokens.append(req)

    if not tokens:
        return DEFAULT_NEGATIVE_PROMPT
    return ", ".join(tokens)[:500]

_FLUX3_PUP_PROMPTS_CACHE: Optional[Dict[str, str]] = None


def get_flux3_pup_system_prompts() -> Dict[str, str]:
    """Load Black Forest Labs' official ``system_prompts.json`` (`t2i`, `i2i`, `t2i_bbox`, `i2i_bbox`)."""
    global _FLUX3_PUP_PROMPTS_CACHE
    if _FLUX3_PUP_PROMPTS_CACHE is not None:
        return _FLUX3_PUP_PROMPTS_CACHE
    prompts_file = Path(__file__).resolve().parent / "registry" / "flux3_pup_system_prompts.json"
    try:
        if prompts_file.is_file():
            _FLUX3_PUP_PROMPTS_CACHE = json.loads(prompts_file.read_text(encoding="utf-8"))
            return _FLUX3_PUP_PROMPTS_CACHE
    except Exception as exc:
        logger.warning("[vision] Could not read flux3_pup_system_prompts.json: %s", exc)
    _FLUX3_PUP_PROMPTS_CACHE = {
        "t2i": (
            "You are the prompt upsampler of a text-to-image model. Expand the user's short "
            "request into ONE detailed, self-contained caption: the exact text the image model "
            "receives. Output only the caption."
        ),
        "i2i": (
            "You are the prompt upsampler of an image-editing model. You see the reference "
            "image(s) and the user's short request. Expand it into ONE detailed, self-contained "
            "instruction as one continuous English prose block. Output only the instruction."
        ),
    }
    return _FLUX3_PUP_PROMPTS_CACHE


def is_flux3_pup_model(model_id: Optional[str]) -> bool:
    if not model_id:
        return False
    m = model_id.lower()
    if "flux-3-image-klein-pup" in m or "flux3-pup" in m:
        return True
    entry = get_catalog_entry(model_id)
    return bool(entry and "flux3_pup" in (entry.get("capabilities") or []))


class VisionRuntimeError(RuntimeError):
    """Base error for vision runtime failures."""

    def __init__(self, message: str, status_code: int = 503) -> None:
        super().__init__(message)
        self.status_code = status_code


class VisionTimeoutError(VisionRuntimeError):
    """Raised when VLM inference times out."""

    def __init__(self, message: str = "Vision model request timed out") -> None:
        super().__init__(message, status_code=504)


def validate_ollama_url(raw_url: str) -> str:
    """Ensure the Ollama base URL points to a valid http/https endpoint."""
    url = (raw_url or "http://127.0.0.1:11434").strip().rstrip("/")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"Invalid Ollama URL: {raw_url!r}")
    return url


def _http_json(
    url: str,
    *,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 5.0,
) -> Dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace").strip()
            if not raw:
                return {}
            return json.loads(raw)
    except (TimeoutError, socket.timeout) as exc:
        raise VisionTimeoutError(f"Ollama timed out after {timeout}s") from exc
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            pass
        if exc.code == 404:
            raise VisionRuntimeError(f"Ollama model or endpoint not found: {body}", status_code=409) from exc
        raise VisionRuntimeError(f"Ollama HTTP {exc.code}: {body or exc.reason}", status_code=503) from exc
    except urllib.error.URLError as exc:
        if isinstance(getattr(exc, "reason", None), (TimeoutError, socket.timeout)):
            raise VisionTimeoutError(f"Ollama timed out after {timeout}s") from exc
        raise VisionRuntimeError(f"Ollama unavailable at {url}: {exc.reason}", status_code=503) from exc
    except Exception as exc:
        raise VisionRuntimeError(f"Ollama error: {exc}", status_code=503) from exc


def check_ollama_status(ollama_url: str, timeout: float = 2.5) -> Dict[str, Any]:
    """Query Ollama ``/api/tags`` (and ``/api/version``) to check availability."""
    base = validate_ollama_url(ollama_url)
    tags: List[str] = []
    version: Optional[str] = None
    try:
        tags_resp = _http_json(f"{base}/api/tags", method="GET", timeout=timeout)
        models = tags_resp.get("models") or []
        for m in models:
            name = m.get("name") or m.get("model")
            if isinstance(name, str) and name.strip():
                tags.append(name.strip())
    except VisionRuntimeError:
        return {
            "ollama_available": False,
            "ollama_version": None,
            "installed_tags": [],
        }

    try:
        ver_resp = _http_json(f"{base}/api/version", method="GET", timeout=timeout)
        version = ver_resp.get("version")
    except Exception:
        version = None

    return {
        "ollama_available": True,
        "ollama_version": version,
        "installed_tags": tags,
    }


# Phrases that describe the REMOVED element (ink / brush artwork).  These are
# stripped ONLY for removal tasks.  Generic vocabulary such as "texture",
# "splatter" or "streaks" is deliberately NOT stripped: after the prompt
# reframe the model describes the surrounding artwork, and its texture words are
# exactly what the diffusion needs in order to continue that surface.
_INPAINT_ARTEFACT_PATTERNS = (
    r"\b(?:irregular\s+|rough\s+|heavy\s+|dense\s+|bold\s+|strong\s+)?black\s+brush\s+strokes?\b",
    r"\b(?:dry[- ]brush|brush[- ]strokes?|brushwork)(?:\s+(?:streaks?|marks?|splashes?))?\b",
    r"\b(?:ink[- ]splash(?:es)?|ink[- ]wash(?:es)?|ink\s+splatter|ink\s+marks?|ink\s+droplets?|ink\s+strokes?)\b",
    r"\b(?:rough|dense|layered)\s+manga[- ]style\s+splatter\b",
    r"\b(?:scattered|flying|sprayed)\s+(?:droplets|ink|splashes)\b",
    r"\b(?:dense\s+|heavy\s+|thick\s+|rich\s+|dark\s+)?"
    r"(?:crimson|scarlet|blood[- ]red|dark\s+red|red)\s+(?:washes?|stains?|pools?)\b",
    r"\b(?:continuous\s+)?painted\s+texture\b",
)

# Meta clauses the VLM likes to append ("contains only the ...", "with no
# lettering").  Diffusion models do not honour negations inside the positive
# prompt, so these are dropped entirely for removal tasks.  Clauses that name
# the REMOVED element are dropped as well - "the area formerly covered by the
# Japanese SFX placeholder" would make the model paint the SFX back in.
_INPAINT_META_PATTERNS = (
    r"\b(?:the\s+)?[\w-]+\s+region\s+contains\s+only\s+the\b[^,.]*",
    r"[^,.]*\bformerly\s+(?:covered|occupied)\s+by\s+[^,.]*",
    r"[^,.]*\breplacing\s+the\s+[^,.]*",
    r",?\s*with\s+no\b[^.]*",
    r",?\s*without\s+any\b[^.]*",
    r",?\s*and\s+no\s+(?:visible\s+)?(?:pattern|text|lettering|figures?|graphics?|symbols?|characters?)\b[^.]*",
    r",?\s*\bno\s+(?:lettering|symbols?|characters?|text|masks?|figures?|graphics?|foreground\s+objects?)\b[^.]*",
    r"\b(?:the\s+)?(?:red\s+|bright\s+)?(?:masked\s+|highlighted\s+)?placeholder\b[,.]?",
)

# Appended to every removal prompt so the diffusion model continues the
# surrounding surface instead of inventing a flat colour patch.
CONTINUATION_MARKER = SURFACE_CONTINUATION_MARKER

_CONTINUATION_MARKER_RE = re.compile(
    r"\b(?:seamless\s+continuation|same\s+surface\s+continues|continuing\s+seamlessly)\b",
    re.IGNORECASE,
)


def _strip_inpaint_artifacts(prompt: str) -> str:
    """Remove ink/brush artwork phrases from a removal-task prompt."""
    p = prompt
    for pat in _INPAINT_META_PATTERNS:
        p = re.sub(pat, "", p, flags=re.IGNORECASE)
    for pat in _INPAINT_ARTEFACT_PATTERNS:
        p = re.sub(pat, "", p, flags=re.IGNORECASE)
    # Clean up list separators, dangling adjectives and conjunctions left behind
    # by removed items.  Run to a fixed point so the result is idempotent.
    for _ in range(3):
        before = p
        p = re.sub(r"\(\s*\)|\[\s*\]", "", p)
        p = re.sub(r",\s*\.", ".", p)
        p = re.sub(r"\b(has|have|is|are|was|were|contains|consists|features)\s*,\s*", r"\1 ", p)
        p = re.sub(r"(?:,\s*){2,}", ", ", p)
        p = re.sub(r"\s+,", ",", p)
        p = re.sub(
            r"\b(?:rough|irregular|heavy|dense|bold|strong|vertical|horizontal|scattered|thick|rich)"
            r"\s*(?=[,.;]|$)",
            "",
            p,
            flags=re.IGNORECASE,
        )
        p = re.sub(r"\b(?:with|and|or|plus)\s*(?=[,.;]|$)", "", p, flags=re.IGNORECASE)
        p = re.sub(r"\bwith\s+(?:and|or)\s+", "", p, flags=re.IGNORECASE)
        # A removed item may leave a dangling preposition: "consists of, mottled".
        p = re.sub(r"\b(of|with|from|into|by)\s*,\s*", r"\1 ", p, flags=re.IGNORECASE)
        p = re.sub(r"\s{2,}", " ", p)
        if p == before:
            break
    p = p.strip().strip(",").strip()
    p = re.sub(r"^(?:and|or|with)\s+", "", p, flags=re.IGNORECASE)
    if not p:
        return CONTINUATION_MARKER
    if not _CONTINUATION_MARKER_RE.search(p):
        p = f"{p.rstrip(' .,;')}, {CONTINUATION_MARKER}"
    return p


def _sanitize_positive_prompt(prompt_raw: str, is_removal: bool = False) -> str:
    p = (prompt_raw or "").strip()
    # Strip any CJK (Japanese/Chinese/Korean) characters so the diffusion model never renders text
    p = re.sub(r'["\'«„“”\(\[]*[\u3000-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff00-\uffef]+["\'»“”\)\]]*', "", p)
    # Strip CSS hex color codes (#030522, #fefefe, etc.) that confuse diffusion text encoders
    p = re.sub(r"#[0-9a-fA-F]{3,8}\b", "", p)
    # Strip multi-image / mask meta-instructions if a prompt upsampler emitted them
    p = re.sub(
        r"^(?:Create|Generate|Render|Paint)\s+a\s+[^,.]*?\bpatch\s+that\s+fills\s+(?:only\s+)?the\s+[^,.]*?(?:from\s+the\s+first\s+image)?(?:,\s*removing\s+[^,.]*?)?[,.]\s*",
        "",
        p,
        flags=re.IGNORECASE,
    )
    p = re.sub(
        r"\bReconstruct\s+the\s+(?:entire\s+)?masked\s+area\s+as\s+a\s+",
        "",
        p,
        flags=re.IGNORECASE,
    )
    p = re.sub(
        r",?\s*using\s+the\s+second\s+image\s+(?:only\s+)?as\s+a\s+guide(?:\s+for\s+[^,.]*)?",
        "",
        p,
        flags=re.IGNORECASE,
    )
    p = re.sub(
        r"\s+from\s+the\s+(?:first|second)\s+image\b",
        "",
        p,
        flags=re.IGNORECASE,
    )
    # Remove any meta-references to the red mask overlay so the diffusion model doesn't paint red shapes
    p = re.sub(
        r"\s*(?:to\s+seamlessly\s+fill|filling|inside|within|in)\s+the\s+(?:bright\s+)?(?:red[-\s]+)?(?:masked|highlighted)\s+(?:shapes|area|region|pixels|zone)s?\.?$",
        "",
        p,
        flags=re.IGNORECASE,
    )
    p = re.sub(r"\bbright\s+red[-\s]+(?:masked|highlighted)\s+(?:vertical\s+|horizontal\s+)?(?:mask\s+)?(?:shapes|area|region|pixels)\b", "surrounding area", p, flags=re.IGNORECASE)
    p = re.sub(r"\bred[-\s]+(?:masked|highlighted|selection)\s+(?:vertical\s+|horizontal\s+)?(?:mask\s+)?(?:shapes|area|region|pixels|border)\b", "surrounding area", p, flags=re.IGNORECASE)
    p = re.sub(r"\s+,", ",", p)
    p = re.sub(r",\s*,+", ", ", p)
    p = re.sub(r"\s{2,}", " ", p)
    p = p.strip().rstrip(",").strip()
    if is_removal:
        p = _strip_inpaint_artifacts(p)
    return p.strip().rstrip(",").strip()


def parse_vlm_response_text(raw_text: str, instruction: str = "") -> Tuple[str, str]:
    """Extract ``(prompt, negative_prompt)`` from VLM or FLUX 3 PUP text output."""
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("Empty response from vision model")

    # Strip <think>...</think> blocks if any model emitted reasoning tags
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()

    # Strip ```json ... ``` fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    candidates = [cleaned]
    brace_match = re.search(r"\{[\s\S]*\}", cleaned)
    if brace_match and brace_match.group(0) != cleaned:
        candidates.append(brace_match.group(0))

    is_removal = _is_removal_instruction(instruction)

    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                prompt = _sanitize_positive_prompt(str(obj.get("prompt") or ""), is_removal=is_removal)
                neg = str(obj.get("negative_prompt") or obj.get("negative") or "").strip()
                if prompt:
                    neg_clean = _sanitize_negative_prompt(neg, instruction=instruction)
                    return prompt[:1000], neg_clean[:500]
        except Exception:
            continue

    # Plain prose caption (e.g. from FLUX 3 Prompt Upsampler)
    prose_lines = [
        ln.strip()
        for ln in cleaned.splitlines()
        if ln.strip() and not ln.strip().startswith(("{", "}"))
    ]
    if prose_lines:
        combined = " ".join(prose_lines)
        combined = re.sub(r"^(prompt\s*:\s*)", "", combined, flags=re.IGNORECASE).strip().strip('"')
        if combined:
            combined = _sanitize_positive_prompt(combined, is_removal=is_removal)
            neg_clean = _sanitize_negative_prompt(DEFAULT_NEGATIVE_PROMPT, instruction=instruction)
            return combined[:1500], neg_clean[:500]

    raise ValueError("Could not extract prompt from vision model response")


# Base rules shared by every PUP request.  Deliberately free of ink/brush
# vocabulary: the previous few-shot example ("rough vertical ink brushwork")
# was the single strongest driver of the ink-splash artefacts (see
# docs/INPAINT_INK_BRUSH_INVESTIGATION.md, §1.4).
_INPAINT_PUP_BASE_RULES = (
    "You write direct, vivid visual scene descriptions for an image inpainting diffusion model.\n"
    "The bright red highlight in the image marks an area that has been emptied out and must be "
    "repainted so it continues the surrounding artwork.\n"
    "CRITICAL RULES:\n"
    "1. Output ONLY a direct visual description (20–40 words) of the background surface.\n"
    "2. NEVER mention 'the first image', 'the second image', 'red highlight', 'mask', 'patch', "
    "'remove', 'delete', 'replace', or CSS hex color codes like #030522.\n"
    "3. NEVER name a subject. No people, characters, faces, heads, hair, clothing, body parts, "
    "plants, animals or objects - not even indirectly ('character-like', 'torso', 'foliage').\n"
    "4. Write ONLY the final visual appearance as a single descriptive phrase (e.g., "
    "'deep crimson painted background with soft mottled shading, darker crimson clouds and "
    "scattered near-black accents, even lighting, seamless surface').\n"
)

# Extra rules appended only for removal tasks.  The model must describe the
# surface AROUND the marked area, not the marked area itself: the marked area is
# only a placeholder (it is red-tinted in the input image), so describing it
# literally is what made the model answer "a completely flat, smooth, uniform
# dark crimson-red background" and the diffusion paint a flat red blob.
#
# Verified limitation (docs/INPAINT_INK_BRUSH_INVESTIGATION.md §10): PUP is a
# captioner and ignores length limits - an explicit "at most 12 words" rule
# still produced 41 words.  Its output is therefore post-filtered down to
# whitelisted surface wording; the colours come from measured pixels.
_INPAINT_PUP_REMOVAL_RULES = (
    "REMOVAL TASK RULES:\n"
    "- Describe the surface AROUND the highlighted area: its colour, tone, texture and lighting.\n"
    "- The highlighted area is an empty placeholder. Never describe it as a flat, smooth or "
    "uniform colour field, and never call it a solid shape.\n"
    "- State that the very same surrounding surface continues into the highlighted area.\n"
    "- Describe only surface materials (texture, grain, shading, pattern). Do NOT invent "
    "subjects, story elements or style labels.\n"
    "- NEVER output the words: 'brush', 'stroke', 'strokes', 'ink', 'splatter', 'spray', "
    "'droplet', 'droplets', 'wash', 'washes', 'streak', 'streaks', 'dry-brush'.\n"
)

INPAINT_PUP_SYSTEM_PROMPT = _INPAINT_PUP_BASE_RULES

INPAINT_PUP_REMOVAL_SYSTEM_PROMPT = _INPAINT_PUP_BASE_RULES + _INPAINT_PUP_REMOVAL_RULES


def _build_inpaint_request(
    instruction: str,
    colors: str,
    *,
    is_removal: bool,
    flat_fill: bool,
) -> str:
    """Build the user request text sent to a PUP prompt upsampler."""
    if is_removal:
        if flat_fill:
            # Explicitly requested uniform colour patch.
            return (
                f"Task: {instruction}. "
                "Write ONLY a direct visual description (15-25 words) of a clean, even, uniform "
                f"background colour patch in the colours ({colors or 'matching surrounding tones'}) "
                "that matches the surrounding area. Do NOT mention the highlight, characters, text "
                "or symbols."
            )
        # The marked area is a placeholder - describe the surface AROUND it so the
        # diffusion can continue that same surface into the hole.  The wording is
        # the one that produced the verified-good caption ("a dense, painterly
        # field of dark crimson ... fills the entire composition, continuing
        # seamlessly through the central area"): it makes PUP describe the
        # surface as a *field* that continues, which is what gives the diffusion
        # model the texture detail it needs.  A caption that names a subject or
        # a foreign colour is rejected downstream (backend/surface_profile.py).
        return (
            f"Task: {instruction}. "
            "Describe the background surface SURROUNDING the highlighted area - its colour, "
            "tone, texture and lighting - so the identical surface can be continued seamlessly "
            f"into it. The surrounding colours are ({colors or 'matching surrounding tones'}). "
            "The highlighted area is an empty placeholder: describe what surrounds it and state "
            "that the same surface continues into it. Do NOT mention the highlight, characters, "
            "text or symbols."
        )
    if instruction:
        return (
            f"Task: {instruction}. "
            "Write ONLY a direct visual description (20-35 words) of what should appear in the marked area, "
            "matching surrounding art style and lighting. Do NOT mention red masks, images, or hex codes."
        )
    return (
        "Write ONLY a direct visual description (15-25 words) of the background surface surrounding "
        f"the highlighted area, in colours ({colors or 'matching surrounding tones'}), stating that "
        "the same surface continues into it. Do NOT mention the highlight, characters, text or symbols."
    )


def run_nunchaku_pup_inference(
    *,
    model_dir: Path,
    quant_file: str,
    prompt_text: str,
    images_b64: Optional[List[str]] = None,
    timeout_sec: int = 120,
    max_new_tokens: int = 180,
    system_prompt: Optional[str] = None,
    inpaint_mode: bool = False,
) -> str:
    """Run ``tonera/flux-3-image-klein-pup-nunchaku`` inside the ``envs/extras`` Python environment."""
    from .extras import extras_python

    py_exe = extras_python()
    if not py_exe or not Path(py_exe).is_file():
        raise VisionRuntimeError("Nunchaku Python environment (envs/extras) is not installed", status_code=503)

    quant_path = (model_dir / quant_file).resolve()
    if not quant_path.is_file():
        raise VisionRuntimeError(f"Nunchaku PUP weights not found: {quant_path}", status_code=409)

    # Ensure system_prompts.json exists in model_dir
    sys_prompts_path = model_dir / "system_prompts.json"
    if not sys_prompts_path.is_file():
        sys_prompts_path.write_text(
            json.dumps(get_flux3_pup_system_prompts(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    runner_code = r"""
import base64, io, json, sys
from pathlib import Path
import torch
from PIL import Image, ImageOps
from transformers import AutoProcessor
from nunchaku import NunchakuQwen3_5ForConditionalGeneration
import nunchaku.models.text_encoders.qwen3_5_pup as _q35_pup

try:
    import triton  # noqa: F401
    _has_triton = True
except ImportError:
    _has_triton = False
    # Windows envs/extras may not have Triton; use native PyTorch CUDA ops for RMSNorm & SiLU*Mul
    def _rms_gain_forward(self, x):
        out = self._norm(x.float()) * self.weight.float()
        return out.type_as(x)
    def _rms_gated_forward(self, hidden_states, gate=None):
        return super(_q35_pup.Qwen3_5RMSNormGatedFused, self).forward(hidden_states, gate)
    def _mlp_gate_up_forward(self, x):
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(torch.nn.functional.silu(gate) * up)
    _q35_pup.Qwen3_5RMSNormGain.forward = _rms_gain_forward
    _q35_pup.Qwen3_5RMSNormGatedFused.forward = _rms_gated_forward
    _q35_pup.Qwen3_5MLPGateUp.forward = _mlp_gate_up_forward

req = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
model_dir = req["model_dir"]
quant_path = req["quant_path"]
prompt = req["prompt"]
images_b64 = req.get("images_b64") or []
max_new_tokens = int(req.get("max_new_tokens") or 512)
custom_sys = req.get("system_prompt")
inpaint_mode = bool(req.get("inpaint_mode"))

sys_prompts = json.loads((Path(model_dir) / "system_prompts.json").read_text(encoding="utf-8"))
processor = AutoProcessor.from_pretrained(model_dir)
upsampler = NunchakuQwen3_5ForConditionalGeneration.from_pretrained(
    quant_path, device="cuda", dtype=torch.bfloat16, decode_cuda_graph=_has_triton
).eval()

pil_images = []
for b64_str in images_b64:
    raw = base64.b64decode(b64_str)
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    pil_images.append(im)

if pil_images:
    if inpaint_mode:
        text = prompt
    else:
        text = (
            "Reference images follow in order (the first image, the second image, ...).\n"
            f"User request: {prompt}\nOutput the caption."
        )
    content = [{"type": "image", "image": im} for im in pil_images] + [{"type": "text", "text": text}]
    sys_msg = custom_sys or sys_prompts["i2i"]
    messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": content}]
else:
    sys_msg = custom_sys or sys_prompts["t2i"]
    messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    enable_thinking=False,
).to("cuda")

with torch.no_grad():
    output = upsampler.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

prompt_tokens = int(inputs["input_ids"].shape[1])
new_tokens = output[0, prompt_tokens:]
caption = processor.decode(new_tokens, skip_special_tokens=True).strip()
print("PUP_RESULT_JSON:" + json.dumps({"caption": caption}, ensure_ascii=True), flush=True)
"""
    with tempfile.TemporaryDirectory(prefix="lazysd_pup_") as tmpdir:
        req_file = Path(tmpdir) / "request.json"
        script_file = Path(tmpdir) / "run_pup.py"
        req_file.write_text(
            json.dumps({
                "model_dir": str(model_dir.resolve()),
                "quant_path": str(quant_path),
                "prompt": prompt_text,
                "images_b64": images_b64 or [],
                "max_new_tokens": int(max_new_tokens or 180),
                "system_prompt": system_prompt,
                "inpaint_mode": inpaint_mode,
            }),
            encoding="utf-8",
        )
        script_file.write_text(runner_code, encoding="utf-8")

        try:
            proc = subprocess.run(
                [str(py_exe), str(script_file), str(req_file)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=max(15, int(timeout_sec)),
            )
        except subprocess.TimeoutExpired as exc:
            raise VisionTimeoutError(f"FLUX.3 Nunchaku Prompt Upsampler timed out after {timeout_sec}s") from exc

        if proc.returncode != 0:
            err_tail = (proc.stderr or proc.stdout or "").strip()[-400:]
            raise VisionRuntimeError(f"Nunchaku PUP failed: {err_tail}", status_code=503)

        for line in (proc.stdout or "").splitlines():
            if line.startswith("PUP_RESULT_JSON:"):
                data = json.loads(line[len("PUP_RESULT_JSON:"):])
                caption = str(data.get("caption") or "").strip()
                if caption:
                    return caption

        raise VisionRuntimeError("Nunchaku PUP did not return a caption", status_code=503)


def generate_vlm_prompt(
    *,
    ollama_url: str,
    model_id: str,
    instruction: str,
    overview_b64: str,
    crop_b64: str,
    analysis_hint: Optional[Dict[str, Any]] = None,
    timeout_sec: int = 120,
    keep_alive: str = "0s",
    models_dir: Optional[Path] = None,
    flat_fill: bool = True,
    surface_profile: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """Generate ``(prompt, negative_prompt)`` for Inpaint via Nunchaku PUP, FLUX 3 PUP GGUF, or Ollama VLM.

    For removal tasks the caption is NOT trusted as the source of truth: it is a
    captioner that describes the whole crop (including characters next to the
    mask) and ignores length limits.  The colour and tonal facts come from the
    measured ``surface_profile``; the caption may only contribute whitelisted
    surface-material wording.  See ``backend/surface_profile.py``.
    """
    entry = get_catalog_entry(model_id)
    is_removal = _is_removal_instruction((instruction or "").strip())
    if surface_profile is None and is_removal and analysis_hint:
        # Degraded fallback: no measured band available (e.g. a direct call).
        surface_profile = build_surface_profile({
            "primary": (analysis_hint.get("color_names") or ["neutral grey"])[0],
            "color_names": analysis_hint.get("color_names") or [],
            "brightness": analysis_hint.get("brightness"),
            "saturation": analysis_hint.get("saturation"),
            "brightness_std": analysis_hint.get("brightness_std"),
            "edge_density": analysis_hint.get("edge_density"),
        })
    # The colours handed to the VLM are the MEASURED ones, not the wide-ring
    # guess: on a mixed panel the wide ring can be dominated by a colour that
    # does not touch the mask at all.
    if surface_profile and surface_profile.get("primary"):
        colors = str(surface_profile["primary"])
    else:
        color_list = (analysis_hint or {}).get("color_names") or []
        colors = ", ".join(color_list[:3])
    if entry and entry.get("provider") == "nunchaku":
        base_models_dir = (models_dir or (Path(__file__).resolve().parent.parent / "models" / "cache")).resolve()
        model_dir = base_models_dir / entry["local_dir"]
        instr_clean = (instruction or "").strip()
        req_text = _build_inpaint_request(
            instr_clean,
            colors,
            is_removal=is_removal,
            flat_fill=flat_fill,
        )
        system_prompt = INPAINT_PUP_REMOVAL_SYSTEM_PROMPT if is_removal else INPAINT_PUP_SYSTEM_PROMPT
        caption = run_nunchaku_pup_inference(
            model_dir=model_dir,
            quant_file=entry["primary_file"],
            prompt_text=req_text,
            images_b64=[crop_b64],
            timeout_sec=timeout_sec,
            # 90 tokens truncated the caption mid-sentence ("The central region
            # continues as") whenever the model wrote more than ~35 words.
            max_new_tokens=PUP_MAX_NEW_TOKENS,
            system_prompt=system_prompt,
            inpaint_mode=True,
        )
        p_out, n_out = parse_vlm_response_text(caption, instruction=instruction)
        if is_removal and surface_profile:
            p_out = build_removal_prompt(surface_profile, p_out, flat_fill=flat_fill)
        return (
            _sanitize_positive_prompt(p_out, is_removal=False),
            _sanitize_negative_prompt(n_out, instruction=instruction),
        )

    global _LAST_ACTIVE_OLLAMA_MODEL
    base = validate_ollama_url(ollama_url)
    use_flux3_pup = is_flux3_pup_model(model_id)
    effective_keep_alive = keep_alive if (keep_alive and keep_alive != "0s") else "5m"
    _LAST_ACTIVE_OLLAMA_MODEL = model_id

    if use_flux3_pup:
        instr_clean = (instruction or "").strip()
        user_content = _build_inpaint_request(
            instr_clean,
            colors,
            is_removal=is_removal,
            flat_fill=flat_fill,
        )
        system_prompt = INPAINT_PUP_REMOVAL_SYSTEM_PROMPT if is_removal else INPAINT_PUP_SYSTEM_PROMPT
        payload = {
            "model": model_id,
            "stream": False,
            "keep_alive": effective_keep_alive,
            "think": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_content,
                    "images": [crop_b64],
                },
            ],
            "options": {
                "temperature": 0.4,
                "top_k": 1,
                "top_p": 0.95,
                "presence_penalty": 0.0,
                "repeat_penalty": 1.0,
                "num_predict": 100,
            },
        }
    else:
        user_parts = []
        instr_clean = (instruction or "").strip()[:500]
        is_removal = _is_removal_instruction(instr_clean)
        rtype = (analysis_hint or {}).get("region_type") or "colored_art"

        if instr_clean:
            user_parts.append(f"User edit instruction for the BRIGHT RED masked pixels: {instr_clean}")
        else:
            user_parts.append("User edit instruction: seamlessly inpaint and restore the background inside the BRIGHT RED masked pixels.")

        if colors:
            user_parts.append(f"Measured boundary colors around the red mask: [{colors}] (region_type={rtype}).")

        if is_removal:
            user_parts.append(
                "IMPORTANT: This is an OBJECT/TEXT/SFX REMOVAL task! "
                "Do NOT describe any character, face, hair, hat, or outfit standing next to the red mask. "
                "The BRIGHT RED masked shapes are only an EMPTY PLACEHOLDER - describe the background "
                "surface AROUND them (colour, tone, texture and lighting) and state that the same "
                f"surface continues into them (matching surrounding {colors or 'colors'}). Never "
                "describe the masked shapes as a flat, smooth, uniform or solid colour field."
            )
        else:
            user_parts.append(
                "IMPORTANT: Focus strictly on the BRIGHT RED highlighted mask shapes in Image 2. "
                "Do NOT caption unmasked characters outside the red highlight."
            )

        user_parts.append(
            'Respond ONLY with JSON: {"prompt": "...", "negative_prompt": "..."}.'
        )
        payload = {
            "model": model_id,
            "stream": False,
            "format": "json",
            "think": False,
            "keep_alive": effective_keep_alive,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "\n".join(user_parts),
                    "images": [overview_b64, crop_b64],
                },
            ],
            "options": {
                "temperature": 0.15,
                "num_predict": 130,
            },
        }

    resp = _http_json(
        f"{base}/api/chat",
        method="POST",
        payload=payload,
        timeout=max(5.0, float(timeout_sec)),
    )
    msg = resp.get("message") or {}
    content = msg.get("content") or resp.get("response") or ""
    p_out, n_out = parse_vlm_response_text(str(content), instruction=instruction)
    if is_removal and surface_profile:
        p_out = build_removal_prompt(surface_profile, p_out, flat_fill=flat_fill)
        n_out = _sanitize_negative_prompt(n_out, instruction=instruction)
    return p_out, n_out


def upsample_flux3_prompt(
    *,
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    images_b64: Optional[List[str]] = None,
    model_id: Optional[str] = None,
    ollama_url: str = "http://127.0.0.1:11434",
    timeout_sec: int = 120,
    keep_alive: str = "0s",
    models_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Upsample a short user prompt into a detailed FLUX 3 Image Klein caption (`t2i` or `i2i`)."""
    clean_prompt = (prompt or "").strip()
    if not clean_prompt:
        raise ValueError("Prompt cannot be empty")
    if len(clean_prompt) > 2000:
        clean_prompt = clean_prompt[:2000]

    base_models_dir = (models_dir or (Path(__file__).resolve().parent.parent / "models" / "cache")).resolve()
    sys_prompts = get_flux3_pup_system_prompts()
    has_images = bool(images_b64)
    target_prefix = f"TARGET CANVAS: {int(width or 1024)}:{int(height or 1024)}\n" if not has_images else ""
    formatted_request = f"{target_prefix}{clean_prompt}"

    # 1. Check if Nunchaku PUP is requested or installed locally
    nunchaku_entry = get_catalog_entry("flux3-pup-nunchaku")
    if (
        (model_id == "flux3-pup-nunchaku" or not model_id)
        and nunchaku_entry
        and is_local_vision_model_installed(nunchaku_entry, models_dir=base_models_dir)
    ):
        caption = run_nunchaku_pup_inference(
            model_dir=base_models_dir / nunchaku_entry["local_dir"],
            quant_file=nunchaku_entry["primary_file"],
            prompt_text=formatted_request,
            images_b64=images_b64,
            timeout_sec=timeout_sec,
        )
        return {
            "prompt": caption,
            "source": "nunchaku_pup",
            "model": "flux3-pup-nunchaku",
        }

    # 2. Try Ollama with the requested model or any installed PUP/vision model
    st = check_ollama_status(ollama_url)
    if st["ollama_available"] and st["installed_tags"]:
        chosen_tag = None
        if model_id and any(t.lower() == model_id.lower() for t in st["installed_tags"]):
            chosen_tag = model_id
        else:
            for tag in st["installed_tags"]:
                if "flux-3-image-klein-pup" in tag.lower() or "qwen" in tag.lower() or "llava" in tag.lower() or "minicpm" in tag.lower():
                    chosen_tag = tag
                    break
            if not chosen_tag:
                chosen_tag = st["installed_tags"][0]

        if chosen_tag:
            base = validate_ollama_url(ollama_url)
            if has_images:
                user_msg: Dict[str, Any] = {
                    "role": "user",
                    "content": (
                        "Reference images follow in order (the first image, the second image, ...).\n"
                        f"User request: {clean_prompt}\nOutput the caption."
                    ),
                    "images": images_b64,
                }
                sys_content = sys_prompts["i2i"]
                pres_pen = 0.0
            else:
                user_msg = {"role": "user", "content": formatted_request}
                sys_content = sys_prompts["t2i"]
                pres_pen = 1.5

            resp = _http_json(
                f"{base}/api/chat",
                method="POST",
                payload={
                    "model": chosen_tag,
                    "stream": False,
                    "keep_alive": keep_alive or "0s",
                    "think": False,
                    "messages": [
                        {"role": "system", "content": sys_content},
                        user_msg,
                    ],
                    "options": {
                        "temperature": 1.0,
                        "top_k": 1,
                        "top_p": 0.95,
                        "presence_penalty": pres_pen,
                        "repeat_penalty": 1.0,
                        "num_predict": 600,
                    },
                },
                timeout=max(5.0, float(timeout_sec)),
            )
            msg = resp.get("message") or {}
            content = str(msg.get("content") or resp.get("response") or "").strip()
            content = re.sub(r"<think>[\s\S]*?</think>", "", content, flags=re.IGNORECASE).strip()
            if content:
                return {
                    "prompt": content[:2000],
                    "source": "ollama_pup",
                    "model": chosen_tag,
                }

    # 3. Deterministic BFL-style local caption expansion fallback (works offline with zero models)
    aspect_hint = (
        "wide horizontal composition"
        if width > height * 1.15
        else ("tall vertical composition" if height > width * 1.15 else "balanced square composition")
    )
    expanded = (
        f"Full-color high-detail image, {aspect_hint}. {clean_prompt.rstrip('.')}. "
        "Crisp subject boundaries, natural coherent lighting, accurate anatomy and proportions, "
        "rich material textures, clean background separation."
    )
    return {
        "prompt": expanded,
        "source": "heuristic_pup",
        "model": None,
        "warnings": [
            "FLUX.3 Prompt Upsampler model is not installed yet — used local template expansion. "
            "You can download FLUX.3 Prompt Upsampler (Nunchaku SVDQ or Unsloth GGUF) in the Inpaint Assistant."
        ],
    }


def unload_vision_model(ollama_url: str, model_id: Optional[str] = None, timeout: float = 3.0) -> bool:
    """Explicitly unload any active VLM / Prompt Upsampler from Ollama VRAM before starting diffusion."""
    global _LAST_ACTIVE_OLLAMA_MODEL
    targets = []
    for cand in (_LAST_ACTIVE_OLLAMA_MODEL, model_id):
        if cand and cand != "flux3-pup-nunchaku" and cand not in targets:
            targets.append(cand)
    unloaded = False
    try:
        base = validate_ollama_url(ollama_url)
        # Also query /api/ps so any running Ollama model (PUP GGUF, Qwen3-VL, Gemma3, etc.) is unloaded
        try:
            ps_resp = _http_json(f"{base}/api/ps", method="GET", timeout=1.5)
            for m in ps_resp.get("models") or []:
                m_name = m.get("name") or m.get("model")
                if isinstance(m_name, str) and m_name.strip() and m_name.strip() not in targets:
                    targets.append(m_name.strip())
        except Exception:
            pass
        for t_model in targets:
            try:
                _http_json(
                    f"{base}/api/generate",
                    method="POST",
                    payload={"model": t_model, "keep_alive": 0},
                    timeout=timeout,
                )
                unloaded = True
            except Exception:
                pass
        return unloaded
    except Exception as exc:
        logger.debug("[vision] Unload skipped or failed: %s", exc)
        return False


def _notify_progress(
    cb: Optional[Callable[..., None]],
    pct: float,
    detail: str,
    downloaded_bytes: Optional[int] = None,
    total_bytes: Optional[int] = None,
    speed_bps: Optional[float] = None,
    status: str = "downloading",
    error: Optional[str] = None,
) -> None:
    if not cb:
        return
    try:
        cb(
            pct,
            detail,
            downloaded_bytes=downloaded_bytes,
            total_bytes=total_bytes,
            speed_bps=speed_bps,
            status=status,
            error=error,
        )
    except TypeError:
        try:
            cb(pct, detail)
        except Exception:
            pass
    except Exception:
        pass


def _download_hf_vision_model(
    entry: Dict[str, Any],
    job_id: str,
    models_dir: Optional[Path] = None,
    hf_token: Optional[str] = None,
    progress_callback: Optional[Callable[..., None]] = None,
) -> None:
    """Download a HuggingFace-hosted vision/PUP model (Nunchaku SVDQ or Unsloth GGUF) into ``models/cache/<local_dir>``."""
    from backend.sources import get_source

    model_id = entry["id"]
    repo_id = entry["repo_id"]
    base_dir = (models_dir or (Path(__file__).resolve().parent.parent / "models" / "cache")).resolve()
    target_dir = base_dir / entry["local_dir"]
    target_dir.mkdir(parents=True, exist_ok=True)

    token = hf_token or os.environ.get("HF_TOKEN") or None
    src = get_source("huggingface", token=token)

    files_to_fetch = list(entry.get("support_files") or []) + [entry["primary_file"]]
    total_files = len(files_to_fetch)

    for idx, fname in enumerate(files_to_fetch, 1):
        base_pct = (idx - 1) / max(1, total_files) * 100.0
        span_pct = 100.0 / max(1, total_files)
        detail = f"Downloading {fname} ({idx}/{total_files})..."
        set_download_state(
            model_id,
            status="downloading",
            job_id=job_id,
            progress=round(base_pct, 1),
            detail=detail,
        )
        _notify_progress(progress_callback, round(base_pct, 1), detail, status="downloading")

        def _file_progress(dl_bytes: int, tot_bytes: int, speed_bps: float, _fname: str = fname, _idx: int = idx, _base: float = base_pct, _span: float = span_pct) -> None:
            frac = (dl_bytes / float(tot_bytes)) if tot_bytes > 0 else 0.0
            cur_pct = min(99.9, round(_base + frac * _span, 1))
            dl_mb = dl_bytes / (1024 * 1024)
            tot_mb = tot_bytes / (1024 * 1024) if tot_bytes > 0 else 0.0
            spd_mb = (speed_bps or 0.0) / (1024 * 1024)
            spd_str = f" · {spd_mb:.1f} MB/s" if spd_mb >= 0.1 else ""
            msg = f"{_fname} ({_idx}/{total_files}): {dl_mb:.1f} / {tot_mb:.1f} MB{spd_str}" if tot_bytes > 0 else f"Downloading {_fname} ({_idx}/{total_files})..."
            set_download_state(
                model_id,
                status="downloading",
                job_id=job_id,
                progress=cur_pct,
                detail=msg,
            )
            _notify_progress(
                progress_callback,
                cur_pct,
                msg,
                downloaded_bytes=dl_bytes,
                total_bytes=tot_bytes,
                speed_bps=speed_bps,
                status="downloading",
            )

        src.download_file(
            repo_id=repo_id,
            file_path=fname,
            local_dir=target_dir,
            revision="main",
            progress_callback=_file_progress,
            job_id=job_id,
        )

    set_download_state(
        model_id,
        status="installed",
        job_id=job_id,
        progress=100.0,
        detail="Installed",
    )
    _notify_progress(progress_callback, 100.0, "Installed ✓", status="complete")


def start_vision_model_download(
    ollama_url: str,
    model_id: str,
    progress_callback: Optional[Callable[..., None]] = None,
    models_dir: Optional[Path] = None,
    hf_token: Optional[str] = None,
    job_id: Optional[str] = None,
    preferred_provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Start (or attach to) an asynchronous download job for a VLM or FLUX.3 PUP model."""
    existing = get_download_state(model_id)
    if existing and existing.get("status") == "downloading":
        return existing

    entry = get_catalog_entry(model_id)
    eff_provider = (preferred_provider or (entry.get("provider") if entry else "ollama") or "ollama").strip().lower()
    if entry and entry.get("provider") == "nunchaku":
        eff_provider = "nunchaku"

    state = set_download_state(
        model_id,
        status="downloading",
        job_id=job_id,
        progress=0.0,
        detail=f"Starting download ({eff_provider})...",
    )
    job_id = state["job_id"]

    def _worker() -> None:
        try:
            # Download via HuggingFace DownloadManager when provider is 'nunchaku' or 'gguf' (local HF cache)
            if entry and eff_provider in ("nunchaku", "gguf", "llamacpp") and entry.get("repo_id") and entry.get("primary_file"):
                _download_hf_vision_model(
                    entry,
                    job_id=job_id,
                    models_dir=models_dir,
                    hf_token=hf_token,
                    progress_callback=progress_callback,
                )
                return

            base = validate_ollama_url(ollama_url)
            url = f"{base}/api/pull"
            data = json.dumps({"name": model_id, "stream": True}).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=600) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except Exception:
                        continue
                    if chunk.get("error"):
                        raise RuntimeError(str(chunk["error"]))
                    status_text = str(chunk.get("status") or "downloading")
                    total = float(chunk.get("total") or 0.0)
                    completed = float(chunk.get("completed") or 0.0)
                    pct = (completed / total * 100.0) if total > 0 else 0.0
                    set_download_state(
                        model_id,
                        status="downloading",
                        job_id=job_id,
                        progress=pct,
                        detail=status_text,
                    )
                    _notify_progress(
                        progress_callback,
                        pct,
                        status_text,
                        downloaded_bytes=int(completed) if completed > 0 else None,
                        total_bytes=int(total) if total > 0 else None,
                        status="downloading",
                    )

            set_download_state(
                model_id,
                status="installed",
                job_id=job_id,
                progress=100.0,
                detail="Installed",
            )
            _notify_progress(progress_callback, 100.0, "Installed ✓", status="complete")
        except Exception as exc:
            logger.warning("[vision] Download failed for %s: %s", model_id, exc)
            is_cancelled = "cancelled by user" in str(exc).lower()
            set_download_state(
                model_id,
                status="not_installed" if is_cancelled else "error",
                job_id=job_id,
                progress=0.0,
                detail="Cancelled" if is_cancelled else "Download failed",
                error=None if is_cancelled else str(exc),
            )
            _notify_progress(
                progress_callback,
                0.0,
                "Cancelled" if is_cancelled else f"Download failed: {exc}",
                status="cancelled" if is_cancelled else "error",
                error=str(exc),
            )

    t = threading.Thread(target=_worker, name=f"vision-dl-{model_id}", daemon=True)
    t.start()
    return state


def delete_vision_model(
    ollama_url: str,
    model_id: str,
    timeout: float = 10.0,
    models_dir: Optional[Path] = None,
) -> bool:
    """Delete a vision/PUP model from local HF cache and/or Ollama."""
    entry = get_catalog_entry(model_id)
    removed_local = False
    if entry and entry.get("local_dir") and entry.get("primary_file"):
        base_dir = (models_dir or (Path(__file__).resolve().parent.parent / "models" / "cache")).resolve()
        primary_path = base_dir / entry["local_dir"] / entry["primary_file"]
        if primary_path.is_file():
            primary_path.unlink(missing_ok=True)
            removed_local = True

    if entry and entry.get("provider") == "nunchaku":
        clear_download_state(model_id)
        return True

    try:
        base = validate_ollama_url(ollama_url)
        _http_json(
            f"{base}/api/delete",
            method="DELETE",
            payload={"name": model_id},
            timeout=timeout,
        )
    except VisionRuntimeError:
        if not removed_local:
            raise
    clear_download_state(model_id)
    return True
