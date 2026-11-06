"""
Manga Degradation Pipeline for Synthetic Restoration Dataset Creation.
Simulates realistic manga scan defects:
- Cheap newsprint paper aging, yellowing & fiber grain
- Ink bleed, diffusion & dot gain (слипание и разбухание туши)
- Line breaks, erosion & fading (разрывы тонких штрихов)
- Screentone moiré & raster degradation (муар сканирования)
- Scanner glass dust, speckles & scratches (пыль и микродефекты)
- Backside show-through (просвечивание обратной стороны)
- Downscaling & aggressive JPEG compression artifacts
"""

import os
import sys
import glob
import random
import argparse
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageEnhance


class MangaDegrader:
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def degrade(self, img_bgr: np.ndarray, severity: str = "medium") -> np.ndarray:
        """Apply a randomized, physically-grounded manga degradation pipeline."""
        h, w = img_bgr.shape[:2]
        out = img_bgr.copy().astype(np.float32)

        # Level presets (strength multiplier)
        mult = {
            "light": 0.6,
            "medium": 1.0,
            "heavy": 1.5,
        }.get(severity, random.uniform(0.7, 1.4))

        # Convert to single-channel luminance / greyscale first (manga is typically B&W)
        if len(out.shape) == 3 and out.shape[2] == 3:
            gray = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = out

        # 1. Backside Show-Through / Bleed-Through (фантомные пятна с обратной стороны)
        if random.random() < 0.6:
            gray = self._apply_show_through(gray, mult)

        # 2. Ink Degradation: Bleed & Diffusion OR Line Breaks / Erosion
        gray = self._apply_ink_defects(gray, mult)

        # 3. Screentone Moiré & Subsampling (сканерный муар растра)
        if random.random() < 0.75:
            gray = self._apply_screentone_moire(gray, mult)

        # 4. Scanner Dust & Speckles (белые точки на черном, черные на белом)
        if random.random() < 0.7:
            gray = self._apply_dust_and_scratches(gray, mult)

        # 5. Paper Tint & Texture (желтизна, грязь, волокна дешевой газетной бумаги)
        out_bgr = self._apply_paper_tone_and_texture(gray, mult)

        # 6. Scanner Non-Uniform Lighting & Vignette (неравномерное освещение сканера)
        out_bgr = self._apply_scanner_lighting(out_bgr, mult)

        # 7. Optical Blur & Defocus (размытие линзы сканера)
        out_bgr = self._apply_optical_blur(out_bgr, mult)

        # 8. Digital Downscale & JPEG Compression Artifacts
        out_bgr = self._apply_jpeg_and_artifacts(out_bgr, mult)

        return np.clip(out_bgr, 0, 255).astype(np.uint8)

    def _apply_show_through(self, gray: np.ndarray, mult: float) -> np.ndarray:
        """Simulate faint ghosting from the reverse side of thin manga paper."""
        h, w = gray.shape[:2]
        # Flip horizontally and blur heavily to simulate backside print
        ghost = cv2.flip(gray, 1)
        ksize = int(max(7, (min(h, w) // 100) | 1))
        ghost = cv2.GaussianBlur(ghost, (ksize, ksize), 0)
        # Shift slightly
        dx = random.randint(-15, 15)
        dy = random.randint(-15, 15)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        ghost = cv2.warpAffine(ghost, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Contrast of the show-through is weak (2% - 8%)
        alpha = random.uniform(0.02, 0.07) * mult
        ghost_mask = (255.0 - ghost) * alpha
        return np.clip(gray - ghost_mask, 0, 255)

    def _apply_ink_defects(self, gray: np.ndarray, mult: float) -> np.ndarray:
        """Simulate ink bleeding into fibers and micro-gaps / broken lines."""
        h, w = gray.shape[:2]
        # Invert so black ink is high (255) and white paper is low (0)
        ink = (255.0 - gray).astype(np.float32)

        # Choice A: Ink bleed / dot gain (чернила расплываются и слипаются)
        if random.random() < 0.65:
            k = random.choice([3, 5]) if mult >= 1.0 else 3
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            bleed = cv2.dilate(ink.astype(np.uint8), kernel, iterations=1).astype(np.float32)
            bleed_blur = cv2.GaussianBlur(bleed, (3, 3), 0.6 * mult)
            blend = random.uniform(0.3, 0.6) * mult
            ink = ink * (1.0 - blend) + bleed_blur * blend

        # Choice B: Line breaks / erosion (тонкие штрихи лица/волос истончаются и рвутся)
        if random.random() < 0.55:
            # Random patchy threshold mask
            noise_map = np.random.normal(0, 15 * mult, (h, w)).astype(np.float32)
            noise_blur = cv2.GaussianBlur(noise_map, (15, 15), 0)
            threshold_cutoff = random.uniform(30, 70)
            # Break fine lines where ink is subtle
            broken = np.where((ink < threshold_cutoff) & (noise_blur > 5), ink * random.uniform(0.2, 0.6), ink)
            ink = broken

        return np.clip(255.0 - ink, 0, 255)

    def _apply_screentone_moire(self, gray: np.ndarray, mult: float) -> np.ndarray:
        """Create authentic scanner moiré by resampling halftone dot lattices."""
        h, w = gray.shape[:2]
        # Rescaling factor with non-integer ratio causes aliasing on dot patterns
        scale = random.uniform(0.60, 0.85)
        inter = random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA])
        small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=inter)
        # Restore with slight phase displacement
        restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        blend = random.uniform(0.5, 0.85)
        return gray * (1.0 - blend) + restored * blend

    def _apply_dust_and_scratches(self, gray: np.ndarray, mult: float) -> np.ndarray:
        """Add white speckles on black ink and black dust flecks on white paper."""
        h, w = gray.shape[:2]
        out = gray.copy()
        num_speckles = int(random.randint(40, 200) * mult)

        for _ in range(num_speckles):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            radius = random.choice([1, 1, 2])
            # If on dark area, make a white dot (speckle / paper flake)
            if out[y, x] < 90 and random.random() < 0.6:
                cv2.circle(out, (x, y), radius, random.uniform(180, 240), -1)
            # If on bright area, make a dark dust particle
            elif out[y, x] > 180 and random.random() < 0.6:
                cv2.circle(out, (x, y), radius, random.uniform(30, 90), -1)

        # Micro-scratches
        if random.random() < 0.4:
            x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
            length = random.randint(15, 80)
            angle = random.uniform(0, 3.14159)
            x2 = int(np.clip(x1 + length * np.cos(angle), 0, w - 1))
            y2 = int(np.clip(y1 + length * np.sin(angle), 0, h - 1))
            val = random.uniform(40, 160)
            cv2.line(out, (x1, y1), (x2, y2), val, 1)

        return out

    def _apply_paper_tone_and_texture(self, gray: np.ndarray, mult: float) -> np.ndarray:
        """Turn pure black/white into cheap yellowed newsprint with grain."""
        h, w = gray.shape[:2]
        # Normalize gray to [0, 1]
        norm = gray / 255.0

        # Colors for cheap pulp newsprint (BGR format):
        # Slightly aged creamy / brownish-yellow base
        paper_b = random.uniform(195, 225)  # blue is lower for warm tone
        paper_g = random.uniform(220, 240)
        paper_r = random.uniform(230, 250)

        # Ink is not pure #000000 on cheap paper; it's dark charcoal blue/brown
        ink_b = random.uniform(20, 45)
        ink_g = random.uniform(20, 40)
        ink_r = random.uniform(20, 40)

        # Interpolate paper tone based on luminance
        b_ch = ink_b + norm * (paper_b - ink_b)
        g_ch = ink_g + norm * (paper_g - ink_g)
        r_ch = ink_r + norm * (paper_r - ink_r)

        out_bgr = cv2.merge([b_ch, g_ch, r_ch])

        # Paper grain / pulp fibers
        grain = np.random.normal(0, 4.5 * mult, (h, w, 1)).astype(np.float32)
        out_bgr = np.clip(out_bgr + grain, 0, 255)

        return out_bgr

    def _apply_scanner_lighting(self, bgr: np.ndarray, mult: float) -> np.ndarray:
        """Simulate uneven scanner illumination, shadow near the book spine."""
        h, w = bgr.shape[:2]
        # Gradient from spine / edge
        edge = random.choice(["left", "right", "top", "bottom"])
        if edge == "left":
            grad = np.linspace(random.uniform(0.75, 0.90), 1.0, w)
            mask = np.tile(grad, (h, 1))[:, :, np.newaxis]
        elif edge == "right":
            grad = np.linspace(1.0, random.uniform(0.75, 0.90), w)
            mask = np.tile(grad, (h, 1))[:, :, np.newaxis]
        else:
            grad = np.linspace(random.uniform(0.85, 0.95), 1.0, h)
            mask = np.tile(grad[:, np.newaxis], (1, w))[:, :, np.newaxis]

        return bgr * mask

    def _apply_optical_blur(self, bgr: np.ndarray, mult: float) -> np.ndarray:
        """Defocus or slight motion blur from handheld scanner / page movement."""
        if random.random() < 0.7:
            k = random.choice([3, 5]) if mult > 1.0 else 3
            sigma = random.uniform(0.4, 0.9) * mult
            bgr = cv2.GaussianBlur(bgr, (k, k), sigma)
        return bgr

    def _apply_jpeg_and_artifacts(self, bgr: np.ndarray, mult: float) -> np.ndarray:
        """Apply aggressive JPEG compression (ringing around lines, 8x8 blocks)."""
        # Quality: lower means uglier ringing around black lines
        base_q = int(round(random.uniform(22, 52) / (1.0 + 0.3 * (mult - 1.0))))
        quality = int(np.clip(base_q, 15, 65))

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, enc = cv2.imencode(".jpg", bgr.astype(np.uint8), encode_param)
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(np.float32)

        return dec


def process_dataset(
    input_dir: str,
    output_dir: str,
    severity: str = "medium",
    generate_captions: bool = True,
    trigger_word: str = "manga_restoration",
    create_side_by_side: bool = True,
):
    inp_p = Path(input_dir)
    out_p = Path(output_dir)

    hq_dir = out_p / "hq"
    lq_dir = out_p / "lq"
    previews_dir = out_p / "previews"

    hq_dir.mkdir(parents=True, exist_ok=True)
    lq_dir.mkdir(parents=True, exist_ok=True)
    if create_side_by_side:
        previews_dir.mkdir(parents=True, exist_ok=True)

    extensions = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
    files = []
    for ext in extensions:
        files.extend(inp_p.glob(ext))

    files = sorted(list(set(files)))
    if not files:
        print(f"[!] No image files found in {input_dir}")
        return

    print(f"[*] Processing {len(files)} manga pages from {input_dir}...")
    degrader = MangaDegrader()

    count = 0
    for idx, f in enumerate(files, 1):
        try:
            # Read image
            im_cv = cv2.imread(str(f))
            if im_cv is None:
                continue

            h, w = im_cv.shape[:2]
            # Ensure divisible by 16 or 32 for optimal DiT/VAE training
            target_w = (w // 32) * 32
            target_h = (h // 32) * 32
            if target_w != w or target_h != h:
                im_cv = cv2.resize(im_cv, (target_w, target_h), interpolation=cv2.INTER_AREA)

            stem = f.stem
            hq_path = hq_dir / f"{stem}.png"
            lq_path = lq_dir / f"{stem}.png"

            # 1. Save pristine HQ ground truth
            cv2.imwrite(str(hq_path), im_cv)

            # 2. Degrade to produce realistic worn paper / bad print LQ
            lq_cv = degrader.degrader = degrader.degrade(im_cv, severity=severity)
            cv2.imwrite(str(lq_path), lq_cv)

            # 3. Create Side-by-Side comparison preview
            if create_side_by_side:
                # Resize if image is huge for preview
                prev_h, prev_w = 1024, int(1024 * (target_w / target_h))
                prev_hq = cv2.resize(im_cv, (prev_w, prev_h), interpolation=cv2.INTER_AREA)
                prev_lq = cv2.resize(lq_cv, (prev_w, prev_h), interpolation=cv2.INTER_AREA)
                sbs = np.hstack([prev_lq, prev_hq])
                # Add text label
                cv2.putText(sbs, "DEGRADED (INPUT / LQ)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.putText(sbs, "ORIGINAL (TARGET / HQ)", (prev_w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imwrite(str(previews_dir / f"{stem}_compare.jpg"), sbs, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

            # 4. Generate AI training captions
            if generate_captions:
                caption_text = (
                    f"{trigger_word}, restore manga page: clean sharp black inking lineart, "
                    f"remove yellow paper tint, remove ink bleed and diffusion, reconstruct broken fine lines, "
                    f"restore crisp halftone screentone dot pattern, pure white paper background, "
                    f"remove scanner dust, speckles and jpeg compression ringing."
                )
                (hq_dir / f"{stem}.txt").write_text(caption_text, encoding="utf-8")
                (lq_dir / f"{stem}.txt").write_text(caption_text, encoding="utf-8")

            count += 1
            print(f"[{idx}/{len(files)}] Done: {stem} -> HQ & LQ saved.")
        except Exception as e:
            print(f"[!] Error processing {f.name}: {e}")

    print(f"\n[OK] Successfully prepared {count} paired manga training sets!")
    print(f"    - HQ Targets: {hq_dir}")
    print(f"    - LQ Inputs:  {lq_dir}")
    if create_side_by_side:
        print(f"    - Comparisons: {previews_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manga Synthetic Degradation Pipeline")
    parser.add_argument("-i", "--input", required=True, help="Folder containing clean HQ manga images")
    parser.add_argument("-o", "--output", required=True, help="Destination folder for paired dataset")
    parser.add_argument("-s", "--severity", default="medium", choices=["light", "medium", "heavy", "random"], help="Degradation severity")
    parser.add_argument("-t", "--trigger", default="manga_restoration", help="Trigger word for LoRA training")
    parser.add_argument("--no-captions", action="store_true", help="Skip creating .txt caption files")
    parser.add_argument("--no-preview", action="store_true", help="Skip side-by-side comparison images")

    args = parser.parse_args()
    process_dataset(
        input_dir=args.input,
        output_dir=args.output,
        severity=args.severity,
        generate_captions=not args.no_captions,
        trigger_word=args.trigger,
        create_side_by_side=not args.no_preview,
    )
