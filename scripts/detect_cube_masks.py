"""Detect physical 3x3 cube regions and save masks for generated inpainting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "sam3.pt")
    parser.add_argument("--prompt", default="physical 3x3 Rubik's cube")
    parser.add_argument("--confidence", type=float, default=0.35)
    args = parser.parse_args()
    if not args.model.exists():
        raise SystemExit(f"missing model: {args.model}")

    from live_recognition.detectors import SAM3CubeDetector

    source_paths = sorted(
        path
        for pattern in ("*.png", "*.jpg", "*.jpeg")
        for path in args.source_dir.glob(pattern)
    )
    args.mask_dir.mkdir(parents=True, exist_ok=True)
    detector = SAM3CubeDetector(
        model_name=str(args.model),
        confidence_threshold=args.confidence,
        prompt=args.prompt,
        return_mask=True,
    )
    manifest = args.mask_dir.parent / "cube_mask_manifest.jsonl"
    existing = {}
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            existing[row.get("image")] = row

    # Resume safely: masks already written are kept, which is important because
    # SAM3 inference is intentionally resumable on a large image collection.
    rows = []
    for index, source_path in enumerate(source_paths, start=1):
        mask_path = args.mask_dir / f"{source_path.stem}.png"
        if mask_path.exists() and source_path.name in existing:
            rows.append(existing[source_path.name])
            continue
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        union = np.zeros(image.shape[:2], dtype=np.uint8)
        detections = []
        for candidate in detector.detect_all(image):
            if candidate.mask is None:
                continue
            union = cv2.bitwise_or(union, candidate.mask)
            detections.append({"bbox_xyxy": list(candidate.bbox_xyxy), "confidence": candidate.confidence})
        cv2.imwrite(str(mask_path), union)
        row = {"image": source_path.name, "detections": detections}
        rows.append(row)
        print(f"[{index}/{len(source_paths)}] {source_path.name}: {len(detections)} cube(s)", flush=True)

    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
