"""Fill missing cube masks with conservative YOLO detection boxes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--padding", type=int, default=4)
    args = parser.parse_args()
    args.mask_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows = {}
    for line in args.candidates.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        candidate_rows[row["image"]] = row
    paths = sorted(args.source_dir.glob("*.png"))
    for index, source_path in enumerate(paths, 1):
        mask_path = args.mask_dir / f"{source_path.stem}.png"
        image = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        row = candidate_rows.get(source_path.name, {})
        existing = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
        # A prior SAM pass can legitimately write an all-zero mask even when
        # the fast detector found a cube. Repair only that case with the saved
        # detector boxes; nonzero precise masks are never replaced.
        if existing is not None and np.any(existing):
            continue
        mask = np.zeros_like(image)
        for detection in row.get("detections", []):
            x1, y1, x2, y2 = [int(round(value)) for value in detection["bbox_xyxy"]]
            x1 = max(0, x1 - args.padding)
            y1 = max(0, y1 - args.padding)
            x2 = min(mask.shape[1], x2 + args.padding)
            y2 = min(mask.shape[0], y2 + args.padding)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
        cv2.imwrite(str(mask_path), mask)
        if index % 100 == 0:
            print(f"[{index}/{len(paths)}]", flush=True)
    print(f"Masks present: {len(list(args.mask_dir.glob('*.png')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
