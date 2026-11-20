#!/usr/bin/env python3
"""Render the ordered 26-point YOLO cube-pose model.

The runtime path intentionally mirrors the live ROI convention:

1. Run the cube bounding-box detector when no tracked ROI exists.
2. Crop the padded square ROI and run the ordered pose model on that crop.
3. Map the 26 predictions back to full-frame coordinates.
4. Use the model's predicted keypoint bounds to update the ROI tracker.

The keypoint indices are never remapped at inference time. They use the
Blender/training convention directly:

    0-7   corners
    8-19  edge midpoints
    20-25 face centers: U, D, F, B, R, L

Examples:

    .venv/bin/python scripts/render_yolo11n_ordered_cube_pose.py \
        --source data/synthetic_cube_dataset/images/example.png \
        --model runs/pose/runs/ordered_cube_pose_local/runs/ordered_cube_yolo11n_pose/weights/best.pt

    .venv/bin/python scripts/render_yolo11n_ordered_cube_pose.py \
        --source 0 \
        --model runs/pose/runs/ordered_cube_pose_local/runs/ordered_cube_yolo11n_pose/weights/best.pt
"""

from __future__ import annotations

import argparse
import __main__
import importlib.util
import json
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from live_recognition.detectors import YOLOCubeDetector  # noqa: E402
from live_recognition.roi_tracking import (  # noqa: E402
    RoiTracker,
    roi_for_frame as shared_roi_for_frame,
)


CENTER_COLORS_BGR = {
    "U": (245, 245, 245),
    "D": (0, 220, 240),
    "F": (0, 166, 0),
    "B": (200, 110, 20),
    "R": (30, 40, 220),
    "L": (24, 125, 240),
}
CORNER_COUNT = 8
EDGE_END_INDEX = 20
DEFAULT_BBOX_MODEL = ROOT / "runs/500ExamplesWeights.pt"
DEFAULT_SYMMETRY_MANIFEST = ROOT / "data/synthetic_cube_dataset/ordered_keypoint_symmetries.json"
DEFAULT_POSE_MODEL = (
    ROOT
    / "runs/pose/runs/ordered_cube_pose_local/runs/ordered_cube_yolo11n_pose/weights/best.pt"
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
WINDOW_NAME = "ordered cube pose"


def load_keypoint_layout(path: Path) -> tuple[str, ...]:
    """Load and validate the Blender-generated keypoint order.

    The Blender manifest is the source of truth for the model's keypoint
    slots.  Connecting lines are intentionally disabled in this renderer for
    now because the current prediction ordering is still being investigated.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "ordered_cube_keypoint_symmetry_v1":
        raise ValueError(f"Unexpected ordered-keypoint manifest format: {path}")
    if payload.get("permutation_convention") != "permutation[output_index] = source_index":
        raise ValueError(f"Unexpected permutation convention in: {path}")
    names = tuple(str(name) for name in payload.get("keypoint_names", ()))
    if len(names) != 26:
        raise ValueError(f"Expected 26 keypoint names in {path}, got {len(names)}")
    permutations = payload.get("permutations")
    if not isinstance(permutations, list) or len(permutations) != 24:
        raise ValueError(f"Expected 24 Blender symmetry permutations in {path}")
    if tuple(range(26)) not in {tuple(int(value) for value in row) for row in permutations}:
        raise ValueError(f"Blender symmetry manifest is missing the identity permutation: {path}")
    if names[20:] != ("U", "D", "F", "B", "R", "L"):
        raise ValueError(f"Unexpected ordered face-center slots in {path}: {names[20:]}")

    if len(set(names)) != len(names):
        raise ValueError(f"Blender keypoint names are not unique: {path}")
    return names


def register_training_checkpoint_classes() -> None:
    """Make the custom training model class available to Ultralytics loading."""

    training_path = ROOT / "scripts" / "train_yolo11n_ordered_cube_pose_local.py"
    spec = importlib.util.spec_from_file_location(
        "ordered_cube_pose_training_checkpoint_classes",
        training_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load training classes from {training_path}")
    training = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = training
    spec.loader.exec_module(training)
    for name in (
        "SymmetryAwarePoseLoss",
        "SymmetryAwarePoseModel",
        "SymmetryAwarePoseTrainer",
    ):
        if hasattr(training, name):
            setattr(__main__, name, getattr(training, name))


def parse_source(source: str) -> int | str:
    source = source.strip().strip("\"'").strip()
    return int(source) if source.isdigit() else source


def is_image_source(source: int | str) -> bool:
    return isinstance(source, str) and Path(source).suffix.lower() in IMAGE_EXTENSIONS


def tensor_to_numpy(value) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    return array if array.size else None


def resize_to_square(image: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def result_keypoints(result) -> tuple[np.ndarray | None, np.ndarray | None]:
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None:
        return None, None

    xy = tensor_to_numpy(getattr(keypoints, "xy", None))
    confidence = tensor_to_numpy(getattr(keypoints, "conf", None))
    if xy is not None:
        return (
            np.asarray(xy, dtype=np.float32),
            None if confidence is None else np.asarray(confidence, dtype=np.float32),
        )

    data = tensor_to_numpy(getattr(keypoints, "data", None))
    if data is None or data.shape[-1] < 2:
        return None, None
    data = np.asarray(data, dtype=np.float32)
    return data[..., :2], data[..., 2] if data.shape[-1] >= 3 else None


def result_box_confidences(result) -> np.ndarray | None:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return None
    confidence = tensor_to_numpy(getattr(boxes, "conf", None))
    return None if confidence is None else np.asarray(confidence, dtype=np.float32).reshape(-1)


def select_detection_index(result, detection_count: int) -> int | None:
    if detection_count <= 0:
        return None
    confidences = result_box_confidences(result)
    if confidences is None or len(confidences) < detection_count:
        return 0
    return int(np.argmax(confidences[:detection_count]))


def local_to_frame_points(
    local_points: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
    model_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = roi_xyxy
    scale = np.asarray(((x2 - x1) / float(model_size), (y2 - y1) / float(model_size)), dtype=np.float32)
    frame_points = local_points * scale + np.asarray((x1, y1), dtype=np.float32)
    return frame_points, local_points * scale


def point_confidence(confidences: np.ndarray | None, index: int) -> float:
    if confidences is None or index >= len(confidences):
        return 1.0
    value = float(confidences[index])
    return value if np.isfinite(value) else 0.0


def print_keypoint_confidences(
    keypoint_names: tuple[str, ...],
    confidences: np.ndarray | None,
) -> None:
    """Print per-keypoint confidence using the Blender manifest order."""

    print("Keypoint confidences:")
    for index, name in enumerate(keypoint_names):
        if confidences is None or index >= len(confidences):
            value = "n/a"
        else:
            confidence = float(confidences[index])
            value = f"{confidence:.4f}" if np.isfinite(confidence) else "n/a"
        print(f"  {index:02d} {name:>3s}: {value}")


def tracking_bounds(
    local_roi_points: np.ndarray,
    confidences: np.ndarray | None,
    threshold: float,
    roi_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    height, width = roi_shape
    usable = []
    for index, point in enumerate(local_roi_points):
        if point_confidence(confidences, index) < threshold:
            continue
        x, y = float(point[0]), float(point[1])
        if np.isfinite(x) and np.isfinite(y) and 0 <= x < width and 0 <= y < height:
            usable.append((x, y))
    if not usable:
        return None
    points = np.asarray(usable, dtype=np.float32)
    return (
        int(np.floor(points[:, 0].min())),
        int(np.floor(points[:, 1].min())),
        int(np.ceil(points[:, 0].max())),
        int(np.ceil(points[:, 1].max())),
    )


def draw_label(image: np.ndarray, text: str, position: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = position
    cv2.putText(image, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


def draw_ordered_pose(
    image: np.ndarray,
    frame_points: np.ndarray,
    confidences: np.ndarray | None,
    keypoint_names: tuple[str, ...],
    display_threshold: float,
    show_labels: bool,
    show_confidences: bool,
) -> tuple[np.ndarray, int]:
    output = image.copy()

    drawn = 0
    for index, (name, point) in enumerate(zip(keypoint_names, frame_points)):
        confidence = point_confidence(confidences, index)
        if confidence < display_threshold:
            continue
        x, y = np.round(point).astype(np.int32)
        if not (0 <= x < output.shape[1] and 0 <= y < output.shape[0]):
            continue
        if index < CORNER_COUNT:
            color = (255, 255, 255)
            radius = 5
        elif index < EDGE_END_INDEX:
            color = (0, 180, 255)
            radius = 4
        else:
            color = CENTER_COLORS_BGR[name]
            radius = 6
        cv2.circle(output, (int(x), int(y)), radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(output, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
        if show_labels:
            label = f"{index}:{name}"
            if show_confidences:
                label += f" {confidence:.2f}"
            draw_label(output, label, (int(x) + 7, int(y) - 5), color)
        drawn += 1
    return output, drawn


def render_frame(
    frame: np.ndarray,
    bbox_detector: YOLOCubeDetector,
    pose_model,
    roi_tracker: RoiTracker,
    args: argparse.Namespace,
    keypoint_names: tuple[str, ...],
) -> tuple[np.ndarray, str, int, np.ndarray | None]:
    height, width = frame.shape[:2]
    roi_xyxy, candidate, roi_source = shared_roi_for_frame(
        frame,
        bbox_detector,
        roi_tracker,
        args.roi_padding,
    )
    output = frame.copy()
    if roi_xyxy is None:
        cv2.putText(output, "no cube bbox", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return output, "none", 0, None

    x1, y1, x2, y2 = roi_xyxy
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        roi_tracker.reset()
        cv2.putText(output, "empty cube roi", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return output, roi_source, 0, None

    model_roi = resize_to_square(roi, args.view_size)
    predict_kwargs = {} if args.device is None else {"device": args.device}
    results = pose_model.predict(
        source=model_roi,
        conf=args.pose_conf,
        imgsz=args.imgsz,
        max_det=1,
        verbose=False,
        **predict_kwargs,
    )
    result = results[0] if results else None
    local_points_by_detection, confidence_by_detection = (
        result_keypoints(result) if result is not None else (None, None)
    )
    detection_index = (
        select_detection_index(result, len(local_points_by_detection))
        if local_points_by_detection is not None
        else None
    )
    if detection_index is None:
        roi_tracker.reset()
        cv2.rectangle(output, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), 2)
        cv2.putText(output, "no ordered pose", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return output, roi_source, 0, None

    local_points = np.asarray(local_points_by_detection[detection_index], dtype=np.float32)
    if local_points.shape[0] != len(keypoint_names):
        roi_tracker.reset()
        raise RuntimeError(
            f"Expected {len(keypoint_names)} ordered keypoints, got {local_points.shape[0]}"
        )
    confidences = (
        None
        if confidence_by_detection is None
        else np.asarray(confidence_by_detection[detection_index], dtype=np.float32).reshape(-1)
    )
    frame_points, local_roi_points = local_to_frame_points(
        local_points,
        roi_xyxy,
        args.view_size,
    )
    output, point_count = draw_ordered_pose(
        output,
        frame_points,
        confidences,
        keypoint_names,
        args.keypoint_conf,
        args.show_labels,
        args.show_confidences,
    )

    # The next frame uses this model-derived extent; a bbox detector is only
    # used again after this tracking observation fails.
    roi_tracker.observe_tracking_bounds(
        tracking_bounds(
            local_roi_points,
            confidences,
            args.track_conf,
            roi.shape[:2],
        ),
        width,
        height,
        tracking_gain=args.roi_tracking_gain,
        tracking_padding=args.roi_tracking_padding,
    )

    roi_color = (0, 0, 255) if roi_source == "detector" else (40, 220, 40)
    cv2.rectangle(output, (x1, y1), (x2 - 1, y2 - 1), roi_color, 2)
    label = f"{roi_source} ROI | {point_count}/{len(keypoint_names)} points"
    if candidate is not None:
        label += f" | bbox {candidate.confidence:.2f}"
    cv2.putText(output, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, roi_color, 2, cv2.LINE_AA)
    return output, roi_source, point_count, confidences


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the ordered 26-point cube-pose model")
    parser.add_argument("--source", required=True, help="Image, video path, or camera index")
    parser.add_argument("--model", "--checkpoint", dest="model", type=Path, default=DEFAULT_POSE_MODEL)
    parser.add_argument("--bbox-model", type=Path, default=DEFAULT_BBOX_MODEL)
    parser.add_argument(
        "--symmetry-manifest",
        type=Path,
        default=DEFAULT_SYMMETRY_MANIFEST,
        help="Blender-generated ordered_keypoint_symmetries.json",
    )
    parser.add_argument("--device", default=None, help="Ultralytics device, for example cpu, mps, or 0")
    parser.add_argument("--bbox-conf", type=float, default=0.25)
    parser.add_argument("--bbox-imgsz", type=int, default=640)
    parser.add_argument("--pose-conf", type=float, default=0.001)
    parser.add_argument(
        "--keypoint-conf",
        "--min-keypoint-conf",
        "--keypoint-confidence",
        "--min-keypoint-confidence",
        dest="keypoint_conf",
        type=float,
        default=0.05,
        help="Minimum keypoint confidence for rendering points",
    )
    parser.add_argument("--track-conf", type=float, default=0.05, help="Minimum confidence for ROI tracking")
    parser.add_argument("--imgsz", type=int, default=256, help="Pose inference size")
    parser.add_argument("--view-size", type=int, default=256, help="Square size used for the cropped ROI")
    parser.add_argument("--roi-padding", type=float, default=0.08)
    parser.add_argument("--roi-tracking-gain", type=float, default=1.4)
    parser.add_argument("--roi-tracking-padding", type=float, default=0.18)
    parser.add_argument("--show-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--show-confidences",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render each keypoint confidence next to its label",
    )
    parser.add_argument(
        "--print-keypoint-confidences",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Print all keypoint confidences; defaults on for images and --once",
    )
    parser.add_argument("--save", type=Path, help="Output image path, or one frame with --once")
    parser.add_argument("--save-video", type=Path, help="Output path for rendered video")
    parser.add_argument("--once", action="store_true", help="Process one video/camera frame and exit")
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.set_defaults(display=True)
    return parser


def create_video_writer(path: Path, frame: np.ndarray, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    fourcc = "mp4v" if suffix in {".mp4", ".m4v", ".mov"} else "MJPG"
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*fourcc),
        fps if fps > 0 else 30.0,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def main() -> int:
    args = build_parser().parse_args()
    if not args.model.is_file():
        raise SystemExit(f"Could not find ordered pose model: {args.model}")
    if not args.bbox_model.is_file():
        raise SystemExit(f"Could not find bbox model: {args.bbox_model}")
    if not args.symmetry_manifest.is_file():
        raise SystemExit(f"Could not find Blender symmetry manifest: {args.symmetry_manifest}")

    keypoint_names = load_keypoint_layout(args.symmetry_manifest)

    register_training_checkpoint_classes()
    from ultralytics import YOLO

    bbox_detector = YOLOCubeDetector(
        model_name=str(args.bbox_model),
        confidence_threshold=args.bbox_conf,
        image_size=args.bbox_imgsz,
    )
    pose_model = YOLO(str(args.model))
    source = parse_source(args.source)
    roi_tracker = RoiTracker()

    if is_image_source(source):
        frame = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if frame is None:
            raise SystemExit(f"Could not read image: {source}")
        rendered, roi_source, point_count, confidences = render_frame(
            frame,
            bbox_detector,
            pose_model,
            roi_tracker,
            args,
            keypoint_names,
        )
        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.save), rendered):
                raise SystemExit(f"Could not write rendered image: {args.save}")
            print(f"Saved render: {args.save}")
        if args.display:
            cv2.imshow(WINDOW_NAME, rendered)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if args.print_keypoint_confidences is not False:
            print_keypoint_confidences(keypoint_names, confidences)
        print(f"ROI source: {roi_source}; rendered points: {point_count}/{len(keypoint_names)}")
        return 0

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {source}")
    writer = None
    previous = perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rendered, _roi_source, _point_count, confidences = render_frame(
                frame,
                bbox_detector,
                pose_model,
                roi_tracker,
                args,
                keypoint_names,
            )
            now = perf_counter()
            fps = 1.0 / max(now - previous, 1e-6)
            previous = now
            cv2.putText(rendered, f"FPS {fps:.1f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 220, 40), 1, cv2.LINE_AA)
            if args.save_video:
                if writer is None:
                    writer = create_video_writer(args.save_video, rendered, float(capture.get(cv2.CAP_PROP_FPS)))
                writer.write(rendered)
            if args.save and args.once:
                args.save.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(args.save), rendered):
                    raise SystemExit(f"Could not write rendered image: {args.save}")
            if args.print_keypoint_confidences is True or (
                args.print_keypoint_confidences is None and args.once
            ):
                print_keypoint_confidences(keypoint_names, confidences)
            if args.display:
                cv2.imshow(WINDOW_NAME, rendered)
                key = cv2.waitKey(30 if args.once else 1) & 0xFF
                if args.once or key in (ord("q"), 27, 13):
                    break
            elif args.once:
                break
    finally:
        if writer is not None:
            writer.release()
        capture.release()
        if args.display:
            cv2.destroyAllWindows()
    if writer is not None:
        print(f"Saved video: {args.save_video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
