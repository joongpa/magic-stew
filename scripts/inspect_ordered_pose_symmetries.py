#!/usr/bin/env python3
"""Print and visualize every ordered-cube symmetry candidate for one image."""

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
GREEN = (70, 220, 70)
RED = (60, 60, 235)
WHITE = (245, 245, 245)
BLACK = (20, 20, 20)


def read_label(path: Path) -> np.ndarray:
    values = [float(value) for value in path.read_text(encoding="utf-8").split()]
    return np.asarray(values[5:], dtype=np.float32).reshape(26, 3)


def candidate_scores(
    prediction_xy: np.ndarray,
    target: np.ndarray,
    permutations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the same unweighted Smooth-L1 coordinate score as the loss."""
    candidate_targets = target[permutations]
    candidate_visible = candidate_targets[..., 2] != 0
    difference = np.abs(prediction_xy[None] - candidate_targets[..., :2])
    smooth_l1 = np.where(difference < 1.0, 0.5 * difference**2, difference - 0.5).sum(axis=-1)
    visible_count = candidate_visible.sum(axis=1)
    scores = (smooth_l1 * candidate_visible).sum(axis=1) / np.maximum(visible_count, 1)
    return scores, candidate_targets


def draw_candidate(
    image: np.ndarray,
    candidate_target: np.ndarray,
    source_indices: np.ndarray,
    prediction_xy: np.ndarray,
    score: float,
    symmetry_index: int,
    state: dict[str, object],
    scale: int,
    annotation_scale: float,
) -> np.ndarray:
    scale = max(1, int(scale))
    annotation_scale = max(0.1, float(annotation_scale))
    height, width = image.shape[:2]
    base = cv2.resize(image, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)
    gt_canvas = base.copy()
    pred_canvas = base.copy()
    target_xy = candidate_target[:, :2] * np.asarray([width * scale, height * scale], dtype=np.float32)
    prediction_xy = prediction_xy * scale
    visible = candidate_target[:, 2] != 0
    point_radius = max(2, round(3 * scale * annotation_scale))
    point_thickness = max(1, round(scale * annotation_scale))
    label_scale = max(0.25, 0.22 * scale * annotation_scale)
    label_thickness = max(1, round(scale * annotation_scale))
    for output_index in np.flatnonzero(visible):
        # Show the candidate's output slot index, not the original source
        # index.  Otherwise every symmetry would display the same labels on
        # the same physical points and the reorder would be invisible.
        label = str(output_index)
        gt = tuple(np.round(target_xy[output_index]).astype(int))
        pred = tuple(np.round(prediction_xy[output_index]).astype(int))
        cv2.circle(gt_canvas, gt, point_radius, GREEN, point_thickness, cv2.LINE_AA)
        cv2.circle(pred_canvas, pred, point_radius, RED, point_thickness, cv2.LINE_AA)
        for canvas, point, color in ((gt_canvas, gt, GREEN), (pred_canvas, pred, RED)):
            origin = (point[0] + 4 * scale, point[1] - 4 * scale)
            cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX, label_scale, BLACK, label_thickness + 1, cv2.LINE_AA)
            cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX, label_scale, color, label_thickness, cv2.LINE_AA)

    panel = np.full((74 * scale, 2 * width * scale, 3), BLACK, dtype=np.uint8)
    mode = state.get("mode", "none")
    face = state.get("face", "-")
    cv2.putText(panel, f"symmetry={symmetry_index:02d}  coordinate_loss={score:.6f}", (8 * scale, 26 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.42 * scale, WHITE, max(1, scale), cv2.LINE_AA)
    cv2.putText(panel, f"{mode} face={face}  permutation[output]=source", (8 * scale, 57 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.34 * scale, WHITE, max(1, scale), cv2.LINE_AA)
    gt_label = np.full((28 * scale, width * scale, 3), BLACK, dtype=np.uint8)
    pred_label = np.full((28 * scale, width * scale, 3), BLACK, dtype=np.uint8)
    cv2.putText(gt_label, "GROUND TRUTH CANDIDATE", (8 * scale, 20 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.31 * scale, GREEN, max(1, scale), cv2.LINE_AA)
    cv2.putText(pred_label, "SAME PREDICTION", (8 * scale, 20 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.34 * scale, RED, max(1, scale), cv2.LINE_AA)
    return np.vstack((panel, np.hstack((gt_label, pred_label)), np.hstack((gt_canvas, pred_canvas))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--stem", default="train_000045")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--annotation-scale", type=float, default=0.35)
    parser.add_argument("--columns", type=int, default=3)
    args = parser.parse_args()

    manifest = json.loads((args.dataset / "layer_state_manifest.json").read_text(encoding="utf-8"))
    train_script = Path(__file__).with_name("train_yolo11n_ordered_cube_pose_local.py")
    spec = importlib.util.spec_from_file_location("ordered_pose_train", train_script)
    train_module = importlib.util.module_from_spec(spec)
    sys.modules["ordered_pose_train"] = train_module
    assert spec.loader is not None
    spec.loader.exec_module(train_module)
    # Use exactly the permutations exposed by the current loss.
    permutations = np.asarray(
        json.loads(
            (args.dataset / "ordered_keypoint_symmetries.json").read_text(encoding="utf-8")
        )["permutations"],
        dtype=np.int64,
    )
    sys.modules["__main__"].SymmetryAwarePoseModel = train_module.SymmetryAwarePoseModel
    sys.modules["__main__"].SymmetryAwarePoseLoss = train_module.SymmetryAwarePoseLoss

    image_path = args.dataset / "images" / args.split / f"{args.stem}.png"
    label_path = args.dataset / "labels" / args.split / f"{args.stem}.txt"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    target = read_label(label_path)
    state = manifest.get(args.stem, {"mode": "none", "face": None})

    model = YOLO(str(args.checkpoint))
    result = model.predict(str(image_path), imgsz=args.imgsz, device="cpu", conf=0.05, max_det=1, verbose=False)[0]
    if result.keypoints is None or not len(result.keypoints.xy):
        raise RuntimeError(f"No pose prediction for {args.stem}")
    best = int(result.boxes.conf.argmax()) if result.boxes is not None and len(result.boxes.conf) else 0
    prediction = result.keypoints.xy[best].cpu().numpy()
    height, width = image.shape[:2]
    prediction_normalized = prediction / np.asarray([width, height], dtype=np.float32)
    scores, candidate_targets = candidate_scores(
        prediction_normalized,
        target,
        permutations,
    )
    order = np.argsort(scores)
    winner = int(order[0])

    lines = [
        f"stem: {args.stem}",
        "score: unweighted visible-point Smooth-L1 coordinate loss used for symmetry selection",
        f"winner: symmetry {winner:02d}",
        "symmetry\tscore\tpermutation[output_index]=source_index",
    ]
    for index in order:
        permutation = " ".join(str(value) for value in permutations[index])
        line = f"{index:02d}\t{scores[index]:.9f}\t{permutation}"
        lines.append(line)
        print(line)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    loss_output = args.output.with_suffix(".txt")
    loss_output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    examples = [
        draw_candidate(
            image,
            candidate_targets[index],
            permutations[index],
            prediction,
            float(scores[index]),
            int(index),
            state,
            args.scale,
            args.annotation_scale,
        )
        for index in range(len(permutations))
    ]
    tile_height = max(example.shape[0] for example in examples)
    tile_width = max(example.shape[1] for example in examples)
    columns = max(1, args.columns)
    rows = (len(examples) + columns - 1) // columns
    sheet = np.full((rows * tile_height, columns * tile_width, 3), BLACK, dtype=np.uint8)
    for index, example in enumerate(examples):
        row, column = divmod(index, columns)
        sheet[row * tile_height : row * tile_height + example.shape[0], column * tile_width : column * tile_width + example.shape[1]] = example
    cv2.imwrite(str(args.output), sheet)
    print(f"winner: symmetry {winner:02d}")
    print(f"loss table: {loss_output}")
    print(f"visualization: {args.output}")


if __name__ == "__main__":
    main()
