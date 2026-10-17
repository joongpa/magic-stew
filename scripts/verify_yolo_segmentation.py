"""Interactively verify YOLO segmentation labels over their source images.

Example:

    .venv/bin/python scripts/verify_yolo_segmentation.py data/mcp_dataset_v5
    .venv/bin/python scripts/verify_yolo_segmentation.py data/mcp_dataset_v5/images

Controls:
    Left/Right or P/N  previous/next image
    W                 toggle whole-cube labels
    S                 toggle sticker labels
    O                 show original image only
    R                 show both overlays
    +/-               change opacity
    Q or Esc          quit

The argument may be either the dataset root or its images/ directory.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
WHOLE_CUBE_COLORS = {
    0: (255, 0, 255),
}
STICKER_COLORS = {
    0: (255, 255, 255),  # White
    1: (0, 255, 255),    # Yellow
    2: (0, 0, 255),      # Red
    3: (0, 165, 255),    # Orange
    4: (255, 0, 0),      # Blue
    5: (0, 180, 0),      # Green
}


@dataclass(frozen=True)
class PolygonLabel:
    class_id: int
    points: np.ndarray


@dataclass(frozen=True)
class Sample:
    stem: str
    image_path: Path
    whole_label_path: Path | None
    sticker_label_path: Path | None
    unified_label_path: Path | None = None


def load_class_names(path: Path, fallback: tuple[str, ...]) -> tuple[str, ...]:
    if not path.exists():
        return fallback
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return tuple(str(data[str(index)]) for index in range(len(data)))
    except (OSError, TypeError, ValueError, KeyError):
        return fallback


def parse_yolo_label(path: Path, width: int, height: int) -> tuple[list[PolygonLabel], list[str]]:
    if not path.exists():
        return [], [f"missing label: {path}"]

    labels = []
    errors = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7 or (len(parts) - 1) % 2:
            errors.append(f"{path.name}:{line_number} has an invalid polygon")
            continue
        try:
            class_id = int(parts[0])
            values = np.asarray([float(value) for value in parts[1:]], dtype=np.float32)
        except ValueError:
            errors.append(f"{path.name}:{line_number} has non-numeric values")
            continue

        normalized = values.reshape(-1, 2)
        if len(normalized) < 3:
            errors.append(f"{path.name}:{line_number} has fewer than 3 points")
            continue
        if np.any(normalized < 0.0) or np.any(normalized > 1.0):
            errors.append(f"{path.name}:{line_number} has coordinates outside [0, 1]")
        points = np.round(
            normalized * np.asarray((width, height), dtype=np.float32)
        ).astype(np.int32)
        labels.append(PolygonLabel(class_id=class_id, points=points))
    return labels, errors


def normalize_dataset_dir(dataset_dir: Path) -> Path:
    dataset_dir = dataset_dir.expanduser()
    if dataset_dir.name == "images" and (dataset_dir.parent / "labels").is_dir():
        return dataset_dir.parent
    return dataset_dir


def discover_samples(dataset_dir: Path) -> list[Sample]:
    dataset_dir = normalize_dataset_dir(dataset_dir)
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    whole_labels_dir = dataset_dir / "labels" / "whole_cube"
    sticker_labels_dir = dataset_dir / "labels" / "stickers"
    if not images_dir.is_dir():
        raise SystemExit(f"Missing images directory: {images_dir}")

    samples = []
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        unified_label_path = labels_dir / f"{image_path.stem}.txt"
        if unified_label_path.is_file():
            whole_label_path = None
            sticker_label_path = None
        else:
            unified_label_path = None
            whole_candidate = whole_labels_dir / f"{image_path.stem}.txt"
            sticker_candidate = sticker_labels_dir / f"{image_path.stem}.txt"
            whole_label_path = whole_candidate if whole_candidate.exists() else None
            sticker_label_path = sticker_candidate if sticker_candidate.exists() else None
        samples.append(
            Sample(
                stem=image_path.stem,
                image_path=image_path,
                whole_label_path=whole_label_path,
                sticker_label_path=sticker_label_path,
                unified_label_path=unified_label_path,
            )
        )
    if not samples:
        raise SystemExit(f"No images found in: {images_dir}")
    return samples


def draw_labels(
    image: np.ndarray,
    labels: list[PolygonLabel],
    colors: dict[int, tuple[int, int, int]],
    class_names: tuple[str, ...],
    opacity: float,
) -> tuple[np.ndarray, int]:
    overlay = image.copy()
    visible_count = 0
    for label in labels:
        if label.class_id not in colors:
            continue
        points = label.points.reshape(-1, 1, 2)
        color = colors[label.class_id]
        cv2.fillPoly(overlay, [points], color)
        cv2.polylines(image, [points], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        visible_count += 1
    return cv2.addWeighted(overlay, opacity, image, 1.0 - opacity, 0.0), visible_count


def fit_to_window(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, max_width / width, max_height / height)
    if scale == 1.0:
        return image
    return cv2.resize(
        image,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def build_frame(
    sample: Sample,
    show_whole: bool,
    show_stickers: bool,
    opacity: float,
    whole_class_names: tuple[str, ...],
    sticker_class_names: tuple[str, ...],
    unified_class_names: tuple[str, ...] = ("cube", "sticker"),
) -> tuple[np.ndarray, list[str]]:
    image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {sample.image_path}")
    height, width = image.shape[:2]
    frame = image.copy()
    errors = []

    if sample.unified_label_path is not None:
        unified_labels, unified_errors = parse_yolo_label(
            sample.unified_label_path,
            width,
            height,
        )
        errors.extend(unified_errors)
        cube_ids = {
            index
            for index, name in enumerate(unified_class_names)
            if name.strip().lower() == "cube"
        } or {0}
        sticker_ids = {
            index
            for index, name in enumerate(unified_class_names)
            if name.strip().lower() == "sticker"
        } or {1}
        if show_whole:
            frame, _whole_count = draw_labels(
                frame,
                [label for label in unified_labels if label.class_id in cube_ids],
                {class_id: WHOLE_CUBE_COLORS.get(0, (255, 0, 255)) for class_id in cube_ids},
                unified_class_names,
                opacity,
            )
        if show_stickers:
            frame, _sticker_count = draw_labels(
                frame,
                [label for label in unified_labels if label.class_id in sticker_ids],
                {class_id: (0, 220, 0) for class_id in sticker_ids},
                unified_class_names,
                opacity,
            )
        return frame, errors

    if show_whole and sample.whole_label_path is not None:
        whole_labels, whole_errors = parse_yolo_label(sample.whole_label_path, width, height)
        frame, _whole_count = draw_labels(
            frame,
            whole_labels,
            WHOLE_CUBE_COLORS,
            whole_class_names,
            opacity,
        )
        errors.extend(whole_errors)

    if show_stickers and sample.sticker_label_path is not None:
        sticker_labels, sticker_errors = parse_yolo_label(sample.sticker_label_path, width, height)
        frame, _sticker_count = draw_labels(
            frame,
            sticker_labels,
            STICKER_COLORS,
            sticker_class_names,
            opacity,
        )
        errors.extend(sticker_errors)

    return frame, errors


def build_status(
    sample_index: int,
    sample_count: int,
    sample: Sample,
    show_whole: bool,
    show_stickers: bool,
    opacity: float,
    errors: list[str],
) -> str:
    modes = []
    if show_whole:
        modes.append("whole")
    if show_stickers:
        modes.append("stickers")
    mode = "+".join(modes) if modes else "original"
    status = (
        f"{sample_index + 1}/{sample_count}  {sample.stem}  "
        f"[{mode}]  opacity {round(opacity * 100)}%  "
        "W whole | S stickers | O original | R both | ←/→ next | +/- opacity | Q quit"
    )
    if errors:
        status += f"  INVALID: {errors[0]}"
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        type=Path,
        help="Dataset directory containing images/ and labels/.",
    )
    parser.add_argument("--start", type=int, default=0, help="Initial sample index.")
    parser.add_argument("--max-width", type=int, default=1400)
    parser.add_argument("--max-height", type=int, default=900)
    args = parser.parse_args()

    dataset_dir = normalize_dataset_dir(args.dataset)
    samples = discover_samples(dataset_dir)
    sample_index = max(0, min(args.start, len(samples) - 1))
    show_whole = True
    show_stickers = True
    opacity_percent = 45
    window_name = "YOLO segmentation label verifier"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Opacity", window_name, opacity_percent, 100, lambda _value: None)

    whole_class_names = load_class_names(
        dataset_dir / "labels" / "whole_cube" / "classes.json",
        ("cube",),
    )
    sticker_class_names = load_class_names(
        dataset_dir / "labels" / "stickers" / "classes.json",
        ("White", "Yellow", "Red", "Orange", "Blue", "Green"),
    )
    unified_class_names = load_class_names(
        dataset_dir / "labels" / "classes.json",
        ("cube", "sticker"),
    )

    while True:
        sample = samples[sample_index]
        opacity_percent = cv2.getTrackbarPos("Opacity", window_name)
        frame, errors = build_frame(
            sample,
            show_whole,
            show_stickers,
            opacity_percent / 100.0,
            whole_class_names,
            sticker_class_names,
            unified_class_names,
        )
        frame = fit_to_window(frame, args.max_width, args.max_height - 44)
        display = cv2.copyMakeBorder(
            frame,
            44,
            0,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(25, 25, 25),
        )
        status = build_status(
            sample_index,
            len(samples),
            sample,
            show_whole,
            show_stickers,
            opacity_percent / 100.0,
            errors,
        )
        cv2.putText(
            display,
            status,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, display)

        raw_key = cv2.waitKeyEx(0)
        key = raw_key & 0xFF
        if key in (ord("q"), 27):
            break
        if raw_key in (81, 2424832, 65361, 123) or key in (ord("p"), ord("a")):
            sample_index = (sample_index - 1) % len(samples)
        elif raw_key in (83, 2555904, 65363, 124) or key in (ord("n"), ord("d")):
            sample_index = (sample_index + 1) % len(samples)
        elif key == ord("w"):
            show_whole = not show_whole
        elif key == ord("s"):
            show_stickers = not show_stickers
        elif key == ord("o"):
            show_whole = False
            show_stickers = False
        elif key == ord("r"):
            show_whole = True
            show_stickers = True
        elif key in (ord("+"), ord("=")):
            cv2.setTrackbarPos("Opacity", window_name, min(100, opacity_percent + 5))
        elif key in (ord("-"), ord("_")):
            cv2.setTrackbarPos("Opacity", window_name, max(0, opacity_percent - 5))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
