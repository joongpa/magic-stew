"""Composite generated inpaints only inside already-saved cube masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dilate", type=int, default=9)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated_suffixes = {".png", ".jpg", ".jpeg"}
    generated = {
        path.stem: path
        for path in args.generated_dir.iterdir()
        if path.is_file() and path.suffix.lower() in generated_suffixes
    }
    count = 0
    for source_path in sorted(args.source_dir.iterdir()):
        if source_path.suffix.lower() not in generated_suffixes or source_path.stem not in generated:
            continue
        mask_path = args.mask_dir / f"{source_path.stem}.png"
        if not mask_path.exists():
            continue
        source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        generated_image = cv2.imread(str(generated[source_path.stem]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if source is None or generated_image is None or mask is None:
            continue
        generated_image = cv2.resize(
            generated_image,
            (source.shape[1], source.shape[0]),
            interpolation=cv2.INTER_LANCZOS4,
        )
        if args.dilate > 0 and np.any(mask):
            size = args.dilate * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            mask = cv2.dilate(mask, kernel)
            # Feather only the transition at the detected object boundary.
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1.0, args.dilate / 2))
        alpha = (mask.astype(np.float32) / 255.0)[..., None]
        result = np.round(source.astype(np.float32) * (1.0 - alpha) + generated_image.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
        output_path = args.output_dir / f"{source_path.stem}.png"
        cv2.imwrite(str(output_path), result, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        count += 1
        print(f"{count}: {output_path.name}", flush=True)
    print(f"Wrote {count} composited images to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
