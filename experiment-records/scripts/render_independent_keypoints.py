#!/usr/bin/env python3
"""Render ground-truth independent keypoints over dataset images.

The renderer reads the YOLO pose labels written under ``pose/`` and does not
need a trained model or the original Blender scene. Each pose row represents
one independent keypoint:

    class x_center y_center width height keypoint_x keypoint_y visibility

Examples:

    .venv/bin/python scripts/render_independent_keypoints.py \
        --dataset data/independent_keypoints_dataset_v2 \
        --stems 000000 005000 \
        --output-dir runs/independent_keypoint_renders \
        --labels

    .venv/bin/python scripts/render_independent_keypoints.py \
        --dataset data/independent_keypoints_dataset_v2 \
        --all \
        --output-dir runs/independent_keypoint_renders
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


CLASS_NAMES = {
    0: "corner",
    1: "corner-edge",
    2: "corner-edge-center",
}

# OpenCV uses BGR colors. These match the colors used by the Blender dataset
# generator's independent-keypoint mask renderer.
CLASS_COLORS = {
    0: (255, 255, 255),
    1: (0, 255, 255),
    2: (255, 120, 255),
}
OCCLUDED_COLOR = (0, 0, 255)
UNLABELED_COLOR = (128, 128, 128)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render ground-truth independent keypoints over dataset images."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/independent_keypoints_dataset_v2"),
        help="Dataset directory containing images/ and pose/.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--stems",
        nargs="+",
        help="Image stems to render, for example 000000 005000.",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="Render every image in the dataset.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="When --stems and --all are omitted, render this many sorted samples (default: 1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/independent_keypoint_renders"),
        help="Directory for rendered PNGs.",
    )
    parser.add_argument(
        "--labels",
        action="store_true",
        help="Draw class names and per-class point indices next to keypoints.",
    )
    parser.add_argument(
        "--show-unlabeled",
        action="store_true",
        help="Draw visibility=0 points in gray. They are hidden by default.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display each rendered image in an OpenCV window.",
    )
    return parser.parse_args()


def normalize_stem(value: str) -> str:
    value = value.strip()
    if value.endswith(".png"):
        value = value[:-4]
    if not value.isdigit():
        raise ValueError(f"Invalid image stem {value!r}; expected a numeric ID.")
    return f"{int(value):06d}"


def available_stems(dataset: Path) -> list[str]:
    return sorted(
        (path.stem for path in (dataset / "images").glob("*.png")),
        key=int,
    )


def read_pose_rows(path: Path) -> list[tuple[int, float, float, int]]:
    rows: list[tuple[int, float, float, int]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) < 8:
            raise ValueError(f"{path}:{line_number}: expected at least 8 fields")
        class_id = int(fields[0])
        if class_id not in CLASS_NAMES:
            raise ValueError(f"{path}:{line_number}: unknown independent class {class_id}")
        keypoint_x = float(fields[5])
        keypoint_y = float(fields[6])
        visibility = int(round(float(fields[7])))
        if visibility not in (0, 1, 2):
            raise ValueError(f"{path}:{line_number}: invalid visibility {visibility}")
        rows.append((class_id, keypoint_x, keypoint_y, visibility))
    return rows


def draw_label(image, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    height, width = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.28, min(width, height) / 128.0 * 0.32)
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


def render_sample(dataset: Path, stem: str, output_path: Path, labels: bool, show_unlabeled: bool) -> int:
    image_path = dataset / "images" / f"{stem}.png"
    pose_path = dataset / "pose" / f"{stem}.txt"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    if not pose_path.is_file():
        raise FileNotFoundError(f"Missing pose label: {pose_path}")

    height, width = image.shape[:2]
    class_indices = {class_id: 0 for class_id in CLASS_NAMES}
    rendered = 0
    for class_id, keypoint_x, keypoint_y, visibility in read_pose_rows(pose_path):
        point_index = class_indices[class_id]
        class_indices[class_id] += 1
        if visibility == 0 and not show_unlabeled:
            continue
        x = max(0, min(width - 1, round(keypoint_x * width)))
        y = max(0, min(height - 1, round(keypoint_y * height)))
        if visibility == 0:
            color = UNLABELED_COLOR
        elif visibility == 1:
            color = OCCLUDED_COLOR
        else:
            color = CLASS_COLORS[class_id]
        radius = max(1, round(min(width, height) / 128.0 * 0.8))
        cv2.circle(image, (x, y), radius, color, -1, cv2.LINE_AA)
        if labels:
            draw_label(image, f"{CLASS_NAMES[class_id]}:{point_index}", x, y, color)
        rendered += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Could not write rendered image: {output_path}")
    if show_unlabeled:
        return rendered
    return rendered


def main() -> None:
    args = parse_args()
    dataset = args.dataset
    if not (dataset / "images").is_dir():
        raise SystemExit(f"Dataset images directory not found: {dataset / 'images'}")
    if not (dataset / "pose").is_dir():
        raise SystemExit(f"Dataset pose directory not found: {dataset / 'pose'}")
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    stems = available_stems(dataset)
    if args.stems:
        requested = [normalize_stem(value) for value in args.stems]
        missing = [stem for stem in requested if stem not in set(stems)]
        if missing:
            raise SystemExit(f"Image stems not found in dataset: {', '.join(missing)}")
        stems = requested
    elif not args.all:
        stems = stems[: args.limit]

    if not stems:
        raise SystemExit(f"No PNG images found under {dataset / 'images'}")

    total_points = 0
    for stem in stems:
        output_path = args.output_dir / f"{stem}.png"
        point_count = render_sample(
            dataset,
            stem,
            output_path,
            labels=args.labels,
            show_unlabeled=args.show_unlabeled,
        )
        total_points += point_count
        print(f"Saved {output_path} ({point_count} points)")
        if args.show:
            image = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
            cv2.imshow("independent keypoints", image)
            key = cv2.waitKey(0) & 0xFF
            if key in (27, ord("q")):
                break
    if args.show:
        cv2.destroyAllWindows()
    print(f"Rendered {len(stems)} sample(s), {total_points} visible point(s)")


if __name__ == "__main__":
    main()
