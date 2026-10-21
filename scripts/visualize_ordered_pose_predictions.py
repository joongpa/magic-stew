#!/usr/bin/env python3
"""Visualize ordered cube pose predictions against ground truth labels."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from ultralytics import YOLO


KEYPOINT_NAMES = (
    "UBL", "UBR", "UFR", "UFL", "DBL", "DBR", "DFR", "DFL",
    "UB", "UR", "UF", "UL", "DB", "DR", "DF", "DL",
    "BR", "BL", "FR", "FL", "U", "D", "F", "B", "R", "L",
)
FACE_KEYPOINT_NAMES = {
    "U": {name for name in KEYPOINT_NAMES if "U" in name},
    "D": {name for name in KEYPOINT_NAMES if "D" in name},
    "F": {name for name in KEYPOINT_NAMES if "F" in name},
    "B": {name for name in KEYPOINT_NAMES if "B" in name},
    "R": {name for name in KEYPOINT_NAMES if "R" in name},
    "L": {name for name in KEYPOINT_NAMES if "L" in name},
}
GREEN = (70, 220, 70)
RED = (60, 60, 235)
CYAN = (220, 210, 40)
ORANGE = (0, 150, 255)
WHITE = (245, 245, 245)
BLACK = (20, 20, 20)


def read_label(path: Path) -> np.ndarray:
    values = [float(value) for value in path.read_text(encoding="utf-8").split()]
    return np.asarray(values[5:], dtype=np.float32).reshape(26, 3)


def layer_names(state: dict[str, object]) -> set[str]:
    names = set(FACE_KEYPOINT_NAMES.get(str(state.get("face")), ()))
    if state.get("mode") == "corner_cut":
        names.update(FACE_KEYPOINT_NAMES.get(str(state.get("adjacent_face")), ()))
    return names


def best_symmetry_alignment(
    prediction_xy: np.ndarray,
    target: np.ndarray,
    permutations: np.ndarray,
    misaligned_source: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the minimum-coordinate-loss target candidate and source indices.

    The permutation convention is ``permutation[output_index] = source_index``.
    Consequently, the returned target is in model-output order, while
    ``source_indices`` preserves the original designated index for each point.
    """
    visible = target[:, 2] != 0
    if not visible.any():
        identity = np.arange(len(target), dtype=np.int64)
        return target.copy(), identity, float("nan")

    candidate_targets = target[permutations]
    candidate_visible = candidate_targets[..., 2] != 0
    difference = np.abs(prediction_xy[None] - candidate_targets[..., :2])
    smooth_l1 = np.where(difference < 1.0, 0.5 * difference**2, difference - 0.5).sum(axis=-1)

    # Match the loss's equal stationary/misaligned weighting when the caller
    # supplies the source-layer mask.
    point_weights = candidate_visible.astype(np.float32)
    if misaligned_source is not None:
        candidate_misaligned = misaligned_source[permutations] & candidate_visible
        candidate_stationary = candidate_visible & ~candidate_misaligned
        visible_count = candidate_visible.sum(axis=1)
        stationary_count = candidate_stationary.sum(axis=1)
        misaligned_count = candidate_misaligned.sum(axis=1)
        has_both = (stationary_count > 0) & (misaligned_count > 0)
        point_weights = np.where(
            candidate_stationary & has_both[:, None],
            0.5 * visible_count[:, None] / np.maximum(stationary_count[:, None], 1),
            point_weights,
        )
        point_weights = np.where(
            candidate_misaligned & has_both[:, None],
            0.5 * visible_count[:, None] / np.maximum(misaligned_count[:, None], 1),
            point_weights,
        )
    scores = (smooth_l1 * point_weights).sum(axis=1) / np.maximum(candidate_visible.sum(axis=1), 1)
    winner = int(scores.argmin())
    return candidate_targets[winner].copy(), permutations[winner].copy(), float(scores[winner])


def draw_example(
    image: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray | None,
    state: dict[str, object],
    permutations: np.ndarray,
    title: str,
    display_scale: int,
    annotation_scale: float,
) -> np.ndarray:
    display_scale = max(1, int(display_scale))
    height, width = image.shape[:2]
    base = cv2.resize(
        image,
        (width * display_scale, height * display_scale),
        interpolation=cv2.INTER_CUBIC,
    )
    moved = layer_names(state)
    misaligned_source = np.asarray([name in moved for name in KEYPOINT_NAMES], dtype=bool)
    aligned_target = target
    source_indices = np.arange(len(target), dtype=np.int64)
    if prediction is not None:
        aligned_target, source_indices, _ = best_symmetry_alignment(
            prediction,
            target,
            permutations,
            misaligned_source,
        )
    target_xy = aligned_target[:, :2] * np.asarray(
        [width * display_scale, height * display_scale], dtype=np.float32
    )
    prediction_xy = prediction * display_scale if prediction is not None else None

    stationary_errors: list[float] = []
    moved_errors: list[float] = []
    visible = aligned_target[:, 2] != 0
    for index, name in enumerate(KEYPOINT_NAMES):
        if not visible[index]:
            continue
        source_name = KEYPOINT_NAMES[source_indices[index]]
        if prediction_xy is not None:
            pred = tuple(np.round(prediction_xy[index]).astype(int))
            error = float(np.linalg.norm(np.asarray(pred, dtype=np.float32) - target_xy[index]))
            (moved_errors if source_name in moved else stationary_errors).append(error)
    # Use separate GT and prediction panels.  An overlaid 26-point drawing is
    # useful for debugging correspondence, but it obscures the cube in these
    # low-resolution source images.
    gt_canvas = base.copy()
    pred_canvas = base.copy()
    annotation_scale = max(0.1, float(annotation_scale))
    point_radius = max(2, round(3 * display_scale * annotation_scale))
    point_thickness = max(1, round(display_scale * annotation_scale))
    label_scale = max(0.25, 0.22 * display_scale * annotation_scale)
    label_thickness = max(1, round(display_scale * annotation_scale))
    for index in np.flatnonzero(visible):
        gt = tuple(np.round(target_xy[index]).astype(int))
        cv2.circle(gt_canvas, gt, point_radius, GREEN, point_thickness, cv2.LINE_AA)
        label_origin = (gt[0] + 4 * display_scale, gt[1] - 4 * display_scale)
        # The displayed number is the output slot after applying the winning
        # symmetry.  This makes the reordered assignment visible; source_indices
        # is still used above to classify stationary versus misaligned points.
        label = str(index)
        cv2.putText(gt_canvas, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, label_scale, BLACK, label_thickness + 1, cv2.LINE_AA)
        cv2.putText(gt_canvas, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, label_scale, GREEN, label_thickness, cv2.LINE_AA)
        if prediction_xy is not None:
            pred = tuple(np.round(prediction_xy[index]).astype(int))
            cv2.circle(pred_canvas, pred, point_radius, RED, point_thickness, cv2.LINE_AA)
            label_origin = (pred[0] + 4 * display_scale, pred[1] - 4 * display_scale)
            cv2.putText(pred_canvas, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, label_scale, BLACK, label_thickness + 1, cv2.LINE_AA)
            cv2.putText(pred_canvas, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, label_scale, RED, label_thickness, cv2.LINE_AA)

    panel = np.full((70 * display_scale, 2 * width * display_scale, 3), BLACK, dtype=np.uint8)
    mode = state.get("mode", "none")
    face = state.get("face", "-")
    angle = float(state.get("angle_degrees", 0.0))
    if mode == "corner_cut":
        angle_text = f"primary={float(state.get('angle_degrees', 0.0)):.1f} adj={float(state.get('adjacent_angle_degrees', 0.0)):.1f}"
    else:
        angle_text = f"angle={angle:.1f}"
    stationary = np.mean(stationary_errors) if stationary_errors else float("nan")
    misaligned = np.mean(moved_errors) if moved_errors else float("nan")
    cv2.putText(panel, title, (8 * display_scale, 24 * display_scale), cv2.FONT_HERSHEY_SIMPLEX, 0.47 * display_scale, WHITE, max(1, display_scale), cv2.LINE_AA)
    cv2.putText(
        panel,
        f"{mode} face={face} {angle_text}  stationary={stationary:.1f}px  misaligned={misaligned:.1f}px",
        (8 * display_scale, 53 * display_scale),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39 * display_scale,
        WHITE,
        max(1, display_scale),
        cv2.LINE_AA,
    )
    gt_label = np.full((28 * display_scale, width * display_scale, 3), BLACK, dtype=np.uint8)
    pred_label = np.full((28 * display_scale, width * display_scale, 3), BLACK, dtype=np.uint8)
    cv2.putText(gt_label, "GROUND TRUTH (MIN-LOSS ORDER)", (8 * display_scale, 20 * display_scale), cv2.FONT_HERSHEY_SIMPLEX, 0.30 * display_scale, GREEN, max(1, display_scale), cv2.LINE_AA)
    cv2.putText(pred_label, "PREDICTION", (8 * display_scale, 20 * display_scale), cv2.FONT_HERSHEY_SIMPLEX, 0.38 * display_scale, RED, max(1, display_scale), cv2.LINE_AA)
    return np.vstack((panel, np.hstack((gt_label, pred_label)), np.hstack((gt_canvas, pred_canvas))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("dataset", type=Path, help="Prepared dataset directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--stems", nargs="+", required=True, help="Prepared image stems to render")
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--scale", type=int, default=3, help="Display scale applied to each source image")
    parser.add_argument("--annotation-scale", type=float, default=1.0, help="Scale for point circles and numeric labels, independent of image scale")
    parser.add_argument("--columns", type=int, default=2, help="Number of examples per row")
    args = parser.parse_args()

    manifest = json.loads((args.dataset / "layer_state_manifest.json").read_text(encoding="utf-8"))
    permutations = np.asarray(
        json.loads((args.dataset / "ordered_keypoint_symmetries.json").read_text(encoding="utf-8"))["permutations"],
        dtype=np.int64,
    )
    # Training checkpoints pickle the custom model class as __main__ because
    # the trainer script is executed directly. Register those classes before
    # asking Ultralytics to deserialize the checkpoint.
    train_script = Path(__file__).with_name("train_yolo11n_ordered_cube_pose_local.py")
    spec = importlib.util.spec_from_file_location("ordered_pose_train", train_script)
    train_module = importlib.util.module_from_spec(spec)
    sys.modules["ordered_pose_train"] = train_module
    assert spec.loader is not None
    spec.loader.exec_module(train_module)
    sys.modules["__main__"].SymmetryAwarePoseModel = train_module.SymmetryAwarePoseModel
    sys.modules["__main__"].SymmetryAwarePoseLoss = train_module.SymmetryAwarePoseLoss
    model = YOLO(str(args.checkpoint))
    examples: list[np.ndarray] = []
    for stem in args.stems:
        image_path = args.dataset / "images" / args.split / f"{stem}.png"
        label_path = args.dataset / "labels" / args.split / f"{stem}.txt"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        target = read_label(label_path)
        result = model.predict(str(image_path), imgsz=args.imgsz, device="cpu", conf=0.05, max_det=1, verbose=False)[0]
        prediction = None
        if result.keypoints is not None and len(result.keypoints.xy):
            best = int(result.boxes.conf.argmax()) if result.boxes is not None and len(result.boxes.conf) else 0
            prediction = result.keypoints.xy[best].cpu().numpy()
        state = manifest.get(stem, {"mode": "none", "face": None})
        examples.append(draw_example(image, target, prediction, state, permutations, stem, args.scale, args.annotation_scale))

    tile_height = max(example.shape[0] for example in examples)
    tile_width = max(example.shape[1] for example in examples)
    columns = max(1, args.columns)
    rows = (len(examples) + columns - 1) // columns
    sheet = np.full((rows * tile_height, columns * tile_width, 3), BLACK, dtype=np.uint8)
    for index, example in enumerate(examples):
        row, column = divmod(index, columns)
        sheet[row * tile_height : row * tile_height + example.shape[0], column * tile_width : column * tile_width + example.shape[1]] = example
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), sheet)
    print(args.output)


if __name__ == "__main__":
    main()
