"""Finish a complete frame set from saved cube masks.

ImageGen results are preferred when present. Missing generated edits use the
local LaMa inpainting backend so the batch can finish without altering pixels
outside the detected cube regions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dilate", type=int, default=7)
    parser.add_argument("--lama-max-size", type=int, default=768)
    args = parser.parse_args()

    from PIL import Image
    from simple_lama_inpainting import SimpleLama

    lama = SimpleLama()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated = {
        p.stem: p for p in args.generated_dir.glob("*.png") if p.is_file()
    }
    sources = sorted(args.source_dir.glob("*.png"))
    for index, source_path in enumerate(sources, 1):
        output_path = args.output_dir / source_path.name
        if output_path.exists():
            continue
        mask_path = args.mask_dir / f"{source_path.stem}.png"
        source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
        if source is None:
            continue
        if mask is None or not np.any(mask):
            result = source
        else:
            size = args.dilate * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            edit_mask = cv2.dilate(mask, kernel)
            edit_mask = cv2.GaussianBlur(edit_mask, (0, 0), sigmaX=max(1.0, args.dilate / 2))
            if source_path.stem in generated:
                fill = cv2.imread(str(generated[source_path.stem]), cv2.IMREAD_COLOR)
                fill = cv2.resize(fill, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            else:
                height, width = source.shape[:2]
                scale = min(1.0, args.lama_max_size / max(height, width))
                if scale < 1.0:
                    small_size = (max(1, round(width * scale)), max(1, round(height * scale)))
                    small = cv2.resize(source, small_size, interpolation=cv2.INTER_AREA)
                    small_mask = cv2.resize(edit_mask, small_size, interpolation=cv2.INTER_NEAREST)
                else:
                    small, small_mask = source, edit_mask
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                fill_rgb = np.asarray(lama(Image.fromarray(rgb), Image.fromarray(small_mask, mode="L")))
                fill = cv2.cvtColor(fill_rgb, cv2.COLOR_RGB2BGR)
                if scale < 1.0:
                    fill = cv2.resize(fill, (width, height), interpolation=cv2.INTER_CUBIC)
            if fill.shape[:2] != source.shape[:2]:
                fill = cv2.resize(fill, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_CUBIC)
            alpha = (edit_mask.astype(np.float32) / 255.0)[..., None]
            result = np.round(source.astype(np.float32) * (1.0 - alpha) + fill.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(output_path), result, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        print(f"[{index}/{len(sources)}] {source_path.name}", flush=True)
    print(f"Wrote processed frames to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
