"""Detect physical 3x3 cubes and create a separate inpainted image set.

The source frames are never modified.  SAM 3 is prompted specifically for a
physical 3x3 cube, so cube-themed graphics and logos are left in the image unless
the detector itself identifies them as a cube.  Masks and a manifest are saved for
audit/review.
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
    parser.add_argument("--input-dir", type=Path, default=Path("data/youtube_solve_frames/frames"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/youtube_solve_frames/cleaned"))
    parser.add_argument("--mask-dir", type=Path, default=Path("data/youtube_solve_frames/masks"))
    parser.add_argument("--model", type=Path, default=REPO_ROOT / "sam3.pt")
    parser.add_argument("--prompt", default="physical 3x3x3 Rubik's cube")
    parser.add_argument("--confidence", type=float, default=0.50)
    parser.add_argument("--min-area-ratio", type=float, default=0.003)
    parser.add_argument("--max-area-ratio", type=float, default=0.75)
    parser.add_argument("--dilate", type=int, default=4)
    parser.add_argument("--inpaint-radius", type=float, default=7.0)
    parser.add_argument(
        "--inpaint-backend",
        choices=["lama", "opencv"],
        default="lama",
        help="LaMa handles large masks better; OpenCV is the dependency-light fallback.",
    )
    parser.add_argument(
        "--lama-max-size",
        type=int,
        default=768,
        help="Longest side used by LaMa before the result is resized back to source size.",
    )
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def inpaint_frame(frame: np.ndarray, mask: np.ndarray, radius: float) -> np.ndarray:
    # NS tends to preserve broad table/background gradients better than Telea for
    # the relatively large masks produced by cube segmentation.
    return cv2.inpaint(frame, mask, radius, cv2.INPAINT_NS)


def lama_inpaint_frame(frame: np.ndarray, mask: np.ndarray, lama, max_size: int) -> np.ndarray:
    from PIL import Image

    height, width = frame.shape[:2]
    scale = min(1.0, max_size / max(height, width))
    if scale < 1.0:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        small_frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    else:
        small_frame = frame
        small_mask = mask
    rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    result = lama(Image.fromarray(rgb), Image.fromarray(small_mask, mode="L"))
    clean_rgb = np.asarray(result)
    clean_bgr = cv2.cvtColor(clean_rgb, cv2.COLOR_RGB2BGR)
    if scale < 1.0:
        clean_bgr = cv2.resize(clean_bgr, (width, height), interpolation=cv2.INTER_CUBIC)
    return clean_bgr


def main() -> int:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")
    if not args.model.exists():
        raise SystemExit(f"SAM 3 model does not exist: {args.model}")
    if args.confidence <= 0 or not 0 < args.min_area_ratio < args.max_area_ratio < 1:
        raise SystemExit("Invalid confidence or area-ratio bounds")

    from live_recognition.detectors import SAM3CubeDetector

    os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / ".ultralytics"))
    detector = SAM3CubeDetector(
        model_name=str(args.model),
        confidence_threshold=args.confidence,
        prompt=args.prompt,
        return_mask=True,
    )
    lama = None
    if args.inpaint_backend == "lama":
        try:
            from simple_lama_inpainting import SimpleLama

            lama = SimpleLama()
            print("Loaded LaMa inpainting model")
        except Exception as error:  # pragma: no cover - depends on optional local model
            print(f"Could not load LaMa ({error}); falling back to OpenCV")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.mask_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(
        path
        for pattern in ("*.jpg", "*.jpeg", "*.png")
        for path in args.input_dir.glob(pattern)
    )
    if args.max_images is not None:
        images = images[: args.max_images]
    manifest_path = args.output_dir.parent / "cleaning_manifest.jsonl"
    rows: list[dict[str, object]] = []

    for index, image_path in enumerate(images, start=1):
        output_path = args.output_dir / image_path.name
        mask_path = args.mask_dir / f"{image_path.stem}.png"
        if output_path.exists() and mask_path.exists() and not args.overwrite:
            print(f"[{index}/{len(images)}] skip {image_path.name}")
            continue
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            rows.append({"image": image_path.name, "status": "read_failed"})
            continue

        candidates = detector.detect_all(frame)
        selected = []
        union = np.zeros(frame.shape[:2], dtype=np.uint8)
        frame_area = float(frame.shape[0] * frame.shape[1])
        for candidate in candidates:
            if candidate.mask is None:
                continue
            area_ratio = float(np.count_nonzero(candidate.mask)) / frame_area
            if args.min_area_ratio <= area_ratio <= args.max_area_ratio:
                selected.append(candidate)
                union = cv2.bitwise_or(union, candidate.mask)

        if args.dilate > 0 and np.any(union):
            kernel = np.ones((args.dilate * 2 + 1, args.dilate * 2 + 1), dtype=np.uint8)
            union = cv2.dilate(union, kernel, iterations=1)
        if np.any(union):
            if lama is not None:
                cleaned = lama_inpaint_frame(frame, union, lama, args.lama_max_size)
            else:
                cleaned = inpaint_frame(frame, union, args.inpaint_radius)
            cv2.imwrite(str(output_path), cleaned, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            cleaned = frame
            cv2.imwrite(str(output_path), cleaned, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(mask_path), union)
        row = {
            "image": image_path.name,
            "status": "cleaned" if selected else "no_cube_detected",
            "detections": [
                {
                    "bbox_xyxy": list(candidate.bbox_xyxy),
                    "confidence": candidate.confidence,
                    "mask_area_ratio": float(np.count_nonzero(candidate.mask)) / frame_area,
                }
                for candidate in selected
            ],
        }
        rows.append(row)
        print(f"[{index}/{len(images)}] {image_path.name}: {row['status']} ({len(selected)} cube(s))")

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"Wrote cleaned images to {args.output_dir}")
    print(f"Wrote masks to {args.mask_dir}")
    print(f"Wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
