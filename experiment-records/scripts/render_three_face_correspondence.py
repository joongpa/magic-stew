#!/usr/bin/env python3
"""Render 37-point correspondence labels over their source images.

Examples:

    .venv/bin/python scripts/render_three_face_correspondence.py \
        --stems 000000 000349 \
        --output-dir runs/three_face_correspondence_renders

    .venv/bin/python scripts/render_three_face_correspondence.py \
        --all --show

Each visible point is annotated with its fixed correspondence index.  The
index-to-3-D-coordinate mapping is read from correspondence_manifest.json;
the renderer does not infer or reorder points from the image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data/independent_keypoints_dataset_v3_three_face"
DEFAULT_OUTPUT = ROOT / "runs/three_face_correspondence_renders"
EXPECTED_KEYPOINT_COUNT = 37
FACE_COLORS_BGR = {
    "U": (255, 190, 40),
    "F": (60, 220, 60),
    "R": (60, 90, 255),
}
BLANK_COLOR_BGR = (130, 130, 130)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--stems", nargs="+", help="Image stems to render, such as 000000 000349.")
    selection.add_argument("--all", action="store_true", help="Render every image in the dataset.")
    parser.add_argument("--limit", type=int, default=1, help="Default number of sorted samples to render.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--show-blank", action="store_true", help="Show blank-index text in a lower-left legend.")
    parser.add_argument("--show", action="store_true", help="Display rendered images in an OpenCV window.")
    return parser.parse_args()


def normalize_stem(value: str) -> str:
    value = value.strip()
    if value.endswith(".png"):
        value = value[:-4]
    if not value.isdigit():
        raise ValueError(f"Invalid image stem {value!r}; expected a numeric ID")
    return f"{int(value):06d}"


def load_manifest(dataset: Path) -> list[dict[str, Any]]:
    path = dataset / "correspondence_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("keypoint_count") != EXPECTED_KEYPOINT_COUNT:
        raise ValueError(f"Expected {EXPECTED_KEYPOINT_COUNT} keypoints in {path}")
    keypoints = payload.get("keypoints")
    if not isinstance(keypoints, list) or len(keypoints) != EXPECTED_KEYPOINT_COUNT:
        raise ValueError(f"Manifest has an invalid keypoint list: {path}")
    return keypoints


def read_label(path: Path) -> tuple[float, float, float, float, list[tuple[float, float, int]]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return 0.0, 0.0, 0.0, 0.0, []
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"{path} must contain one pose row, got {len(lines)}")
    fields = lines[0].split()
    expected_fields = 1 + 4 + EXPECTED_KEYPOINT_COUNT * 3
    if len(fields) != expected_fields:
        raise ValueError(f"{path} has {len(fields)} fields; expected {expected_fields}")
    class_id = int(fields[0])
    if class_id != 0:
        raise ValueError(f"{path} has unexpected class {class_id}")
    bbox = tuple(float(value) for value in fields[1:5])
    points = []
    for index in range(EXPECTED_KEYPOINT_COUNT):
        offset = 5 + index * 3
        x, y = float(fields[offset]), float(fields[offset + 1])
        visibility = int(float(fields[offset + 2]))
        if visibility not in (0, 1, 2):
            raise ValueError(f"{path} keypoint {index} has invalid visibility {visibility}")
        points.append((x, y, visibility))
    return *bbox, points


def draw_text(image, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    height, width = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.30, min(width, height) / 256.0 * 0.42)
    thickness = max(1, round(scale * 2.0))
    text_width, text_height = cv2.getTextSize(text, font, scale, thickness)[0]
    text_x = x + 4
    if text_x + text_width >= width:
        text_x = x - text_width - 4
    text_y = y - 4
    if text_y - text_height < 0:
        text_y = y + text_height + 4
    text_x = max(0, min(text_x, width - text_width))
    text_y = max(text_height, min(text_y, height - 1))
    cv2.putText(image, text, (text_x, text_y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, (text_x, text_y), font, scale, color, thickness, cv2.LINE_AA)


def render_sample(dataset: Path, stem: str, output_path: Path, manifest: list[dict[str, Any]], show_blank: bool) -> int:
    image_path = dataset / "images" / f"{stem}.png"
    label_path = dataset / "pose" / f"{stem}.txt"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing label: {label_path}")
    _bbox_x, _bbox_y, _bbox_w, _bbox_h, points = read_label(label_path)
    if not points:
        cv2.putText(image, "NO VISIBLE KEYPOINTS", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 2, cv2.LINE_AA)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"Could not write rendered image: {output_path}")
        return 0

    height, width = image.shape[:2]
    visible_count = 0
    blank_indices = []
    for index, ((x, y, visibility), metadata) in enumerate(zip(points, manifest, strict=True)):
        if visibility == 0:
            blank_indices.append(index)
            continue
        px = max(0, min(width - 1, round(x * width)))
        py = max(0, min(height - 1, round(y * height)))
        faces = metadata.get("faces") or ["U"]
        color = FACE_COLORS_BGR.get(str(faces[0]), (255, 255, 255))
        radius = max(2, round(min(width, height) / 128.0))
        cv2.circle(image, (px, py), radius + 1, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(image, (px, py), radius, color, -1, cv2.LINE_AA)
        draw_text(image, str(index), px, py, color)
        visible_count += 1

    if show_blank and blank_indices:
        legend = "blank: " + ", ".join(str(index) for index in blank_indices)
        cv2.rectangle(image, (3, height - 21), (width - 3, height - 3), (0, 0, 0), -1)
        cv2.putText(image, legend, (6, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.28, BLANK_COLOR_BGR, 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Could not write rendered image: {output_path}")
    return visible_count


def available_stems(dataset: Path) -> list[str]:
    return sorted((path.stem for path in (dataset / "images").glob("*.png")), key=int)


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    dataset = args.dataset
    if not (dataset / "images").is_dir() or not (dataset / "pose").is_dir():
        raise SystemExit(f"Expected images/ and pose/ below {dataset}")
    manifest = load_manifest(dataset)
    stems = available_stems(dataset)
    if args.stems:
        requested = [normalize_stem(value) for value in args.stems]
        missing = [stem for stem in requested if stem not in set(stems)]
        if missing:
            raise SystemExit(f"Image stems not found: {', '.join(missing)}")
        stems = requested
    elif not args.all:
        stems = stems[: args.limit]
    if not stems:
        raise SystemExit(f"No images found below {dataset / 'images'}")

    total = 0
    for stem in stems:
        output_path = args.output_dir / f"{stem}.png"
        count = render_sample(dataset, stem, output_path, manifest, args.show_blank)
        total += count
        print(f"Saved {output_path} ({count} visible points)")
        if args.show:
            image = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
            cv2.imshow("three-face correspondence", image)
            key = cv2.waitKey(0) & 0xFF
            if key in (27, ord("q")):
                break
    if args.show:
        cv2.destroyAllWindows()
    print(f"Rendered {len(stems)} sample(s), {total} visible point(s)")


if __name__ == "__main__":
    main()
