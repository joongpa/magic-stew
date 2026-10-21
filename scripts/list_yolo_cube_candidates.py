"""List source frames likely containing physical cubes using the local YOLO model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "runs/500ExamplesWeights.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()
    from live_recognition.detectors import YOLOCubeDetector

    detector = YOLOCubeDetector(model_name=str(args.model), confidence_threshold=args.confidence)
    paths = sorted(args.source_dir.glob("*.png"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index, path in enumerate(paths, 1):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            detections = detector.detect_all(image) if image is not None else []
            if detections:
                row = {
                    "image": path.name,
                    "detections": [
                        {"bbox_xyxy": list(d.bbox_xyxy), "confidence": d.confidence}
                        for d in detections
                    ],
                }
                handle.write(json.dumps(row) + "\n")
            if index % 100 == 0:
                print(f"[{index}/{len(paths)}]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
