"""Keep only generated pixels inside detected physical-cube masks.

Image generation is useful for reconstructing the background behind a cube, but
it may also rewrite logos or text elsewhere in a frame.  This script enforces the
edit boundary: outside the SAM mask, the original source pixels are copied back
exactly.  Generated images may have a different output size/aspect encoding; they
are resized to the source dimensions before compositing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=REPO_ROOT / "sam3.pt")
    parser.add_argument("--prompt", default="physical Rubik's cube or twisty puzzle")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model.exists():
        raise SystemExit(f"SAM 3 model does not exist: {args.model}")
    from live_recognition.detectors import SAM3CubeDetector

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.mask_dir.mkdir(parents=True, exist_ok=True)
    source_paths = sorted(
        path
        for pattern in ("*.png", "*.jpg", "*.jpeg")
        for path in args.source_dir.glob(pattern)
    )
    detector = SAM3CubeDetector(
        model_name=str(args.model),
        confidence_threshold=args.confidence,
        prompt=args.prompt,
        return_mask=True,
    )
    rows: list[dict[str, object]] = []

    for index, source_path in enumerate(source_paths, start=1):
        generated_paths = sorted(args.generated_dir.glob(f"{source_path.stem}.*"))
        generated_paths = [path for path in generated_paths if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        if not generated_paths:
            continue
        output_path = args.output_dir / f"{source_path.stem}.png"
        mask_path = args.mask_dir / f"{source_path.stem}.png"
        if output_path.exists() and mask_path.exists() and not args.overwrite:
            continue

        source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        generated = cv2.imread(str(generated_paths[-1]), cv2.IMREAD_COLOR)
        if source is None or generated is None:
            rows.append({"image": source_path.name, "status": "read_failed"})
            continue
        candidates = detector.detect_all(source)
        union = np.zeros(source.shape[:2], dtype=np.uint8)
        detections = []
        for candidate in candidates:
            if candidate.mask is None:
                continue
            union = cv2.bitwise_or(union, candidate.mask)
            detections.append(
                {
                    "bbox_xyxy": list(candidate.bbox_xyxy),
                    "confidence": candidate.confidence,
                }
            )
        cv2.imwrite(str(mask_path), union)
        generated = cv2.resize(generated, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        result = source.copy()
        result[union > 0] = generated[union > 0]
        cv2.imwrite(str(output_path), result, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        rows.append({"image": source_path.name, "status": "composited", "detections": detections})
        print(f"[{index}/{len(source_paths)}] {source_path.name}: {len(detections)} physical cube(s)")

    manifest = args.output_dir.parent / "composite_manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"Wrote final composited images to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
