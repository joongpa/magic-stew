"""Interactively verify real ordered-keypoint v2 turned-layer labels.

Examples:

    .venv/bin/python scripts/verify_ordered_keypoint_layer_labels.py \
        data/ordered_pose_dataset_v2
    .venv/bin/python scripts/verify_ordered_keypoint_layer_labels.py \
        data/ordered_pose_dataset_v2 --face F --start 10
    .venv/bin/python scripts/verify_ordered_keypoint_layer_labels.py \
        data/ordered_pose_dataset_v2 --export-dir /private/tmp/turn-overlays \
        --no-display --limit 20

Controls:
    Left/Right or P/N/A/D       previous/next layer-turn image
    Q or Esc                    quit

The filtered face uses the same pivot logic as the training loss: ``mid_turn``
uses ``face``; ``corner_cut`` uses the face whose misalignment has the largest
angle. Visible points are filled, labeled-occluded
points are hollow, and unlabeled points are reported in the status panel rather
than drawn at (0, 0).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from verify_ordered_keypoint_layer_metadata import validate_record


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
VISIBILITY_NAMES = {0: "unlabeled", 1: "occluded", 2: "visible"}
ORANGE = (0, 150, 255)
GREEN = (70, 210, 70)
RED = (70, 70, 235)
WHITE = (245, 245, 245)
DARK = (25, 25, 25)
TURN_LAYER_MODES = frozenset(("mid_turn", "corner_cut"))


def loss_pivot_face(layer_state: dict[str, object]) -> object:
    """Return the layer face used by SymmetryAwarePoseLoss for this state."""
    face = layer_state.get("face")
    if layer_state.get("mode") != "corner_cut":
        return face
    adjacent_face = layer_state.get("adjacent_face")
    try:
        primary_angle = abs(float(layer_state.get("angle_degrees", 0.0)))
        adjacent_angle = abs(float(layer_state.get("adjacent_angle_degrees", 0.0)))
    except (TypeError, ValueError):
        return face
    if adjacent_face in {"U", "D", "F", "B", "R", "L"} and adjacent_angle > primary_angle:
        return adjacent_face
    return face


def pivot_keypoint_names(record: dict[str, object]) -> set[str]:
    """Return the nine structural names belonging to the loss pivot face."""
    state = record.get("layer_state") or {}
    pivot_face = loss_pivot_face(state)
    if not isinstance(pivot_face, str):
        return set()
    ordered = record.get("ordered_keypoints") or {}
    return {
        str(keypoint.get("name"))
        for keypoint in ordered.get("keypoints", [])
        if isinstance(keypoint, dict) and pivot_face in str(keypoint.get("name", ""))
    }


@dataclass(frozen=True)
class Sample:
    stem: str
    image_path: Path
    record: dict[str, object]


def normalize_dataset_dir(dataset_dir: Path) -> Path:
    dataset_dir = dataset_dir.expanduser()
    if dataset_dir.name == "images" and (dataset_dir.parent / "metadata.json").is_file():
        return dataset_dir.parent
    return dataset_dir


def discover_samples(dataset_dir: Path, face: str | None = None) -> list[Sample]:
    dataset_dir = normalize_dataset_dir(dataset_dir)
    images_dir = dataset_dir / "images"
    metadata_path = dataset_dir / "metadata.json"
    if not images_dir.is_dir():
        raise SystemExit(f"Missing images directory: {images_dir}")
    if not metadata_path.is_file():
        raise SystemExit(f"Missing metadata file: {metadata_path}")

    records = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise SystemExit(f"Metadata must contain a list: {metadata_path}")

    samples = []
    for record in records:
        if not isinstance(record, dict):
            continue
        layer_state = record.get("layer_state") or {}
        if layer_state.get("mode") not in TURN_LAYER_MODES:
            continue
        if face is not None and loss_pivot_face(layer_state) != face:
            continue
        image_name = Path(str(record.get("image", ""))).name
        image_path = images_dir / image_name
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS or not image_path.is_file():
            continue
        samples.append(Sample(image_path.stem, image_path, record))
    if not samples:
        suffix = f" for face {face}" if face else ""
        raise SystemExit(f"No layer-turn metadata samples found{suffix} in: {dataset_dir}")
    return samples


def key_direction(raw_key: int) -> int:
    key = raw_key & 0xFF
    if raw_key in (81, 2424832, 65361, 123) or key in (ord("p"), ord("a")):
        return -1
    if raw_key in (83, 2555904, 65363, 124) or key in (ord("n"), ord("d")):
        return 1
    return 0


def put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = WHITE,
    scale: float = 0.48,
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_status_panel(
    image: np.ndarray,
    sample_index: int,
    sample_count: int,
    sample: Sample,
    validation: dict[str, object],
    keypoints: list[dict[str, object]],
) -> np.ndarray:
    """Draw a readable status panel without changing the source image geometry."""
    height, width = image.shape[:2]
    panel_height = 88
    canvas = np.full((height + panel_height, width, 3), DARK, dtype=np.uint8)
    canvas[:height] = image
    state = sample.record.get("layer_state") or {}
    mode = state.get("mode", "?")
    face = state.get("face", "?")
    pivot_face = loss_pivot_face(state) or "?"
    if mode == "corner_cut":
        primary_angle = float(state.get("angle_degrees", 0.0))
        adjacent_angle = float(state.get("adjacent_angle_degrees", 0.0))
        if abs(adjacent_angle) > abs(primary_angle):
            angle = adjacent_angle
            direction = state.get("adjacent_direction", "?")
        else:
            angle = primary_angle
            direction = state.get("direction", "?")
        angle_label = (
            f"pivot angle={angle:.2f} deg  adjacent={state.get('adjacent_face', '?')} "
            f"({adjacent_angle:.2f} deg)  primary={primary_angle:.2f} deg"
        )
    else:
        angle = float(state.get("angle_degrees", 0.0))
        direction = state.get("direction", "?")
        angle_label = f"angle={angle:.2f} deg"
    passed = bool(validation.get("ok"))
    status_color = GREEN if passed else RED
    status = "DERIVATION PASS" if passed else "DERIVATION FAIL"
    put_text(canvas, f"{sample_index + 1}/{sample_count}  {sample.stem}", (6, height + 16), WHITE, 0.48, 1)
    put_text(
        canvas,
        f"{mode}  pivot={pivot_face}  {angle_label}  direction={direction}  {status}",
        (6, height + 36),
        status_color,
        0.46,
        1,
    )
    moved = sorted(pivot_keypoint_names(sample.record))
    moved_text = " ".join(str(name) for name in moved)
    put_text(canvas, f"pivot layer: {moved_text}", (6, height + 56), ORANGE, 0.42, 1)
    visibility_text = " ".join(
        f"{kp['index']}:{VISIBILITY_NAMES.get(int(kp.get('visibility', 0)), '?')}"
        for kp in keypoints
    )
    put_text(canvas, visibility_text, (6, height + 76), WHITE, 0.36, 1)
    return canvas


def fit_to_window(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(4.0, max_width / width, max_height / height)
    if abs(scale - 1.0) < 1e-3:
        return image
    interpolation = cv2.INTER_NEAREST if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=interpolation)


def render_sample(sample: Sample, index: int, count: int, tolerance: float) -> np.ndarray:
    image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {sample.image_path}")
    validation = validate_record(sample.record, tolerance)
    ordered = sample.record.get("ordered_keypoints") or {}
    expected_moved = pivot_keypoint_names(sample.record)
    keypoints = [
        keypoint for keypoint in ordered.get("keypoints", [])
        if keypoint.get("name") in expected_moved
    ]
    keypoints.sort(key=lambda keypoint: int(keypoint.get("index", 0)))

    overlay = image.copy()
    annotation_scale = max(0.5, min(1.0, overlay.shape[1] / 256.0))
    for keypoint in keypoints:
        visibility = int(keypoint.get("visibility", 0))
        pixel = keypoint.get("pixel_point")
        if visibility == 0 or not isinstance(pixel, list) or len(pixel) != 2:
            continue
        x, y = int(round(float(pixel[0]))), int(round(float(pixel[1])))
        if not (0 <= x < overlay.shape[1] and 0 <= y < overlay.shape[0]):
            continue
        base_radius = 5 if keypoint.get("kind") == "center" else 4
        radius = max(1, round(base_radius * annotation_scale))
        cv2.circle(overlay, (x, y), radius + 2, DARK, -1, cv2.LINE_AA)
        if visibility == 2:
            cv2.circle(overlay, (x, y), radius, ORANGE, -1, cv2.LINE_AA)
        else:
            cv2.circle(overlay, (x, y), radius, ORANGE, 2, cv2.LINE_AA)
        label = f"{keypoint['index']}:{keypoint['name']}"
        text_scale = max(0.22, 0.38 * annotation_scale)
        put_text(
            overlay,
            label,
            (min(overlay.shape[1] - 2, x + radius + 2), max(10, y - radius - 2)),
            ORANGE,
            text_scale,
            1,
        )

    legend_height = 25
    legend = np.full((legend_height, overlay.shape[1], 3), DARK, dtype=np.uint8)
    cv2.circle(legend, (10, 12), 5, ORANGE, -1, cv2.LINE_AA)
    put_text(legend, "turned layer only", (20, 17), WHITE, 0.42, 1)
    cv2.circle(legend, (151, 12), 5, ORANGE, 2, cv2.LINE_AA)
    put_text(legend, "occluded", (161, 17), WHITE, 0.42, 1)
    display = np.vstack((legend, overlay))
    display = draw_status_panel(display, index, count, sample, validation, keypoints)
    return display


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Dataset root or its images/ directory.")
    parser.add_argument(
        "--face",
        choices=("U", "D", "F", "B", "R", "L"),
        help="Only show samples whose loss pivot face is this face.",
    )
    parser.add_argument("--start", type=int, default=0, help="Initial filtered sample index.")
    parser.add_argument("--limit", type=int, help="Only inspect the first N filtered records.")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--max-width", type=int, default=1400)
    parser.add_argument("--max-height", type=int, default=900)
    parser.add_argument("--export-dir", type=Path, help="Also save turned-layer overlays here.")
    parser.add_argument("--no-display", action="store_true", help="Export overlays without opening a window.")
    args = parser.parse_args()

    dataset_dir = normalize_dataset_dir(args.dataset)
    samples = discover_samples(dataset_dir, args.face)
    if args.limit is not None:
        samples = samples[: max(0, args.limit)]
    if not samples:
        raise SystemExit("No samples remain after applying --limit.")

    if args.export_dir is not None:
        args.export_dir.mkdir(parents=True, exist_ok=True)
        for index, sample in enumerate(samples):
            overlay = render_sample(sample, index, len(samples), args.tolerance)
            cv2.imwrite(str(args.export_dir / f"{sample.stem}.png"), overlay)
        if args.no_display:
            return

    sample_index = max(0, min(args.start, len(samples) - 1))
    window_name = "Ordered keypoint v2 - turned layer verifier"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        while True:
            display = render_sample(samples[sample_index], sample_index, len(samples), args.tolerance)
            display = fit_to_window(display, args.max_width, args.max_height)
            cv2.setWindowTitle(
                window_name,
                f"{sample_index + 1}/{len(samples)} {samples[sample_index].stem} | P/N/A/D or arrows | Q quit",
            )
            cv2.imshow(window_name, display)
            raw_key = cv2.waitKeyEx(0)
            if (raw_key & 0xFF) in (ord("q"), 27):
                break
            direction = key_direction(raw_key)
            if direction:
                sample_index = (sample_index + direction) % len(samples)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
