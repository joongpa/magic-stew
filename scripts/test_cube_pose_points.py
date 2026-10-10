"""Display YOLO piece-wise cube pose keypoints."""

from __future__ import annotations

import argparse
import __main__
import importlib.util
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
    square_roi_from_bbox,
)

DEFAULT_BBOX_MODEL = ROOT / "runs/500ExamplesWeights.pt"
DEFAULT_CUBE_POSE_MODEL = (
    ROOT
    / "runs/pose/runs/full_piece_pose_local/runs/cube_piece_keypoint_yolo11n_pose/weights/best.pt"
)
DEFAULT_VIEW_SIZE = 256
POSE_OBJECT_CONFIDENCE = 0.001
POSE_CLASS_NAMES = {"corner", "corner-edge", "corner-edge-center", "center"}
KEYPOINT_COLORS = {
    "corner": (255, 255, 255),
    "edge": (0, 255, 255),
    "center": (255, 120, 255),
}
PIECE_LINE_COLOR = (40, 220, 40)


def register_training_checkpoint_classes() -> None:
    """Expose custom training classes where torch expects them during checkpoint loading."""
    training_path = ROOT / "scripts" / "train_yolo11n_piece_seg_local.py"
    spec = importlib.util.spec_from_file_location("cube_pose_training_checkpoint_classes", training_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load training classes from {training_path}")
    training = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = training
    spec.loader.exec_module(training)

    for name in (
        "PerAnchorKeypointLoss",
        "CubeSymmetricPoseLoss",
        "CubeSymmetricPoseModel",
        "CubeSymmetricPoseTrainer",
    ):
        if hasattr(training, name):
            setattr(__main__, name, getattr(training, name))


def parse_source(source: str) -> int | str:
    source = source.strip().strip("\"'").strip()
    return int(source) if source.isdigit() else source


def clipped_square_roi(
    bbox_xyxy: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    return square_roi_from_bbox(bbox_xyxy, frame_width, frame_height, padding)


def resize_to_square_size(image: np.ndarray, size: int) -> np.ndarray:
    if size <= 0:
        return image
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


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


def class_name_for_id(result, model, class_id: int) -> str:
    names = getattr(result, "names", None) or getattr(model, "names", {}) or {}
    name = str(names.get(class_id, class_id)).lower()
    return name.removesuffix("_sticker")


def keypoint_kind(class_name: str, index: int) -> str:
    if class_name == "corner":
        return "corner"
    if class_name == "corner-edge":
        return "edge"
    return "center"


def result_boxes(result) -> tuple[np.ndarray | None, np.ndarray | None]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return None, None
    classes = tensor_to_numpy(getattr(boxes, "cls", None))
    confidences = tensor_to_numpy(getattr(boxes, "conf", None))
    if classes is None:
        return None, None
    return (
        np.asarray(classes, dtype=np.int32).reshape(-1),
        None if confidences is None else np.asarray(confidences, dtype=np.float32).reshape(-1),
    )


def result_keypoints(result, size: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None:
        return None, None

    xy = tensor_to_numpy(getattr(keypoints, "xy", None))
    conf = tensor_to_numpy(getattr(keypoints, "conf", None))
    if xy is not None:
        return np.asarray(xy, dtype=np.float32), None if conf is None else np.asarray(conf, dtype=np.float32)

    xyn = tensor_to_numpy(getattr(keypoints, "xyn", None))
    if xyn is not None:
        xy = np.asarray(xyn, dtype=np.float32).copy()
        xy[..., 0] *= size
        xy[..., 1] *= size
        return xy, None if conf is None else np.asarray(conf, dtype=np.float32)

    data = tensor_to_numpy(getattr(keypoints, "data", None))
    if data is None:
        return None, None
    data = np.asarray(data, dtype=np.float32)
    if data.shape[-1] >= 2:
        xy = data[..., :2]
        conf = data[..., 2] if data.shape[-1] >= 3 else None
        return xy, conf
    return None, None


def piece_detection_indexes(
    result,
    model,
    classes: np.ndarray | None,
    detection_count: int,
) -> list[tuple[int, str]]:
    indexes = []
    for detection_index in range(detection_count):
        class_id = (
            int(classes[detection_index])
            if classes is not None and detection_index < len(classes)
            else -1
        )
        class_name = class_name_for_id(result, model, class_id)
        if class_name in POSE_CLASS_NAMES:
            indexes.append((detection_index, class_name))
    return indexes


def piece_line_indexes(class_name: str) -> tuple[tuple[int, int], ...]:
    return ()


def draw_keypoints(
    frame: np.ndarray,
    result,
    model,
    xy: np.ndarray | None,
    keypoint_conf: np.ndarray | None,
    *,
    confidence_threshold: float,
    offset_xy: tuple[int, int] = (0, 0),
    scale_xy: tuple[float, float] = (1.0, 1.0),
) -> tuple[np.ndarray, tuple[int, int, int, int] | None, int]:
    output = frame.copy()
    if xy is None or len(xy) == 0:
        cv2.putText(output, "no cube pose", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        return output, None, 0

    classes, _box_conf = result_boxes(result)
    points_by_detection = np.asarray(xy, dtype=np.float32)
    detection_indexes = piece_detection_indexes(
        result,
        model,
        classes,
        len(points_by_detection),
    )
    if not detection_indexes:
        cv2.putText(output, "no cube pieces", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        return output, None, 0

    scores_by_detection = (
        None
        if keypoint_conf is None or len(keypoint_conf) == 0
        else np.asarray(keypoint_conf, dtype=np.float32)
    )
    drawn = 0
    tracking_min_x: int | None = None
    tracking_min_y: int | None = None
    tracking_max_x: int | None = None
    tracking_max_y: int | None = None
    offset_x, offset_y = offset_xy
    scale_x, scale_y = scale_xy
    for detection_index, class_name in detection_indexes:
        points = points_by_detection[detection_index]
        if len(points) == 0:
            continue
        detection_scores = (
            None
            if scores_by_detection is None
            or detection_index >= len(scores_by_detection)
            else np.asarray(scores_by_detection[detection_index], dtype=np.float32).reshape(-1)
        )
        image_points: list[tuple[int, int] | None] = []
        for keypoint_index, point in enumerate(points):
            keypoint_score = (
                None
                if detection_scores is None or keypoint_index >= len(detection_scores)
                else float(detection_scores[keypoint_index])
            )
            if keypoint_score is not None and keypoint_score < confidence_threshold:
                image_points.append(None)
                continue

            roi_x = int(round(float(point[0]) * scale_x))
            roi_y = int(round(float(point[1]) * scale_y))
            x = offset_x + roi_x
            y = offset_y + roi_y
            if not (0 <= x < output.shape[1] and 0 <= y < output.shape[0]):
                image_points.append(None)
                continue
            image_points.append((x, y))

            tracking_min_x = roi_x if tracking_min_x is None else min(tracking_min_x, roi_x)
            tracking_min_y = roi_y if tracking_min_y is None else min(tracking_min_y, roi_y)
            tracking_max_x = roi_x if tracking_max_x is None else max(tracking_max_x, roi_x)
            tracking_max_y = roi_y if tracking_max_y is None else max(tracking_max_y, roi_y)
            drawn += 1

        for start_index, end_index in piece_line_indexes(class_name):
            if start_index >= len(image_points) or end_index >= len(image_points):
                continue
            start = image_points[start_index]
            end = image_points[end_index]
            if start is None or end is None:
                continue
            cv2.line(output, start, end, (0, 0, 0), 4, lineType=cv2.LINE_AA)
            cv2.line(output, start, end, PIECE_LINE_COLOR, 2, lineType=cv2.LINE_AA)

        for keypoint_index, image_point in enumerate(image_points):
            if image_point is None:
                continue
            color = KEYPOINT_COLORS.get(keypoint_kind(class_name, keypoint_index), (255, 255, 255))
            cv2.circle(output, image_point, 4, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(output, image_point, 5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    if drawn == 0:
        cv2.putText(output, "no cube points", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
    tracking_bounds = (
        (tracking_min_x, tracking_min_y, tracking_max_x, tracking_max_y)
        if (
            tracking_min_x is not None
            and tracking_min_y is not None
            and tracking_max_x is not None
            and tracking_max_y is not None
        )
        else None
    )
    return output, tracking_bounds, drawn


def roi_for_frame(
    frame: np.ndarray,
    bbox_detector,
    args: argparse.Namespace,
    roi_tracker: RoiTracker,
) -> tuple[tuple[int, int, int, int] | None, object | None, str]:
    return shared_roi_for_frame(frame, bbox_detector, roi_tracker, args.roi_padding)


def process_frame(
    frame: np.ndarray,
    bbox_detector,
    model,
    args: argparse.Namespace,
    roi_tracker: RoiTracker,
) -> np.ndarray:
    output = frame.copy()
    roi_xyxy, candidate, roi_source = roi_for_frame(frame, bbox_detector, args, roi_tracker)
    if roi_xyxy is None:
        cv2.putText(output, "no cube bbox", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        return output

    x1, y1, x2, y2 = roi_xyxy
    roi = frame[y1:y2, x1:x2]
    model_roi = resize_to_square_size(roi, args.view_size)
    results = model.predict(source=model_roi, conf=POSE_OBJECT_CONFIDENCE, imgsz=args.imgsz, verbose=False)
    result = results[0] if results else None
    xy, conf = result_keypoints(result, args.view_size) if result is not None else (None, None)
    scale_x = (x2 - x1) / float(model_roi.shape[1])
    scale_y = (y2 - y1) / float(model_roi.shape[0])
    output, tracking_bounds_xyxy, point_count = draw_keypoints(
        output,
        result,
        model,
        xy,
        conf,
        confidence_threshold=args.conf,
        offset_xy=(x1, y1),
        scale_xy=(scale_x, scale_y),
    )
    height, width = frame.shape[:2]
    roi_tracker.observe_tracking_bounds(
        tracking_bounds_xyxy,
        width,
        height,
        tracking_gain=args.roi_tracking_gain,
        tracking_padding=args.roi_tracking_padding,
    )
    if roi_tracker.roi_xyxy is not None:
        x1, y1, x2, y2 = roi_tracker.roi_xyxy

    roi_color = (0, 0, 255) if roi_source == "detector" else (40, 220, 40)
    cv2.rectangle(output, (x1, y1), (x2, y2), roi_color, 2)
    if args.debug_overlay:
        label = (
            f"cube {candidate.confidence:.2f} points {point_count}"
            if candidate is not None
            else f"tracked cube points {point_count}"
        )
        cv2.putText(output, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, roi_color, 2, cv2.LINE_AA)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Camera index, video path, or image path")
    parser.add_argument("--model", type=Path, default=DEFAULT_CUBE_POSE_MODEL)
    parser.add_argument("--bbox-model", type=Path, default=DEFAULT_BBOX_MODEL)
    parser.add_argument("--bbox-conf", type=float, default=0.25)
    parser.add_argument("--bbox-imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_VIEW_SIZE)
    parser.add_argument("--view-size", type=int, default=DEFAULT_VIEW_SIZE)
    parser.add_argument("--roi-padding", type=float, default=0.08)
    parser.add_argument(
        "--roi-tracking-gain",
        type=float,
        default=1.4,
        help="Multiplier for ROI movement from detected point bounds-center offset",
    )
    parser.add_argument(
        "--roi-tracking-padding",
        type=float,
        default=0.18,
        help="Extra padding ratio added around tracked point bounds when updating the ROI",
    )
    parser.add_argument("--save", type=Path, help="Save one annotated frame/image")
    parser.add_argument("--debug-overlay", action="store_true")
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.set_defaults(display=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    from ultralytics import YOLO

    bbox_detector = YOLOCubeDetector(
        model_name=str(args.bbox_model),
        confidence_threshold=args.bbox_conf,
        image_size=args.bbox_imgsz,
    )
    register_training_checkpoint_classes()
    model = YOLO(str(args.model))
    source = parse_source(args.source)
    roi_tracker = RoiTracker()

    if isinstance(source, str) and Path(source).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
        frame = cv2.imread(source)
        if frame is None:
            raise SystemExit(f"Could not read image: {source}")
        output = process_frame(frame, bbox_detector, model, args, roi_tracker)
        if args.save:
            cv2.imwrite(str(args.save), output)
        if args.display:
            cv2.imshow("cube pose points", output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return 0

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    previous = perf_counter()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        output = process_frame(frame, bbox_detector, model, args, roi_tracker)
        now = perf_counter()
        fps = 1.0 / max(now - previous, 1e-6)
        previous = now
        cv2.putText(output, f"FPS {fps:.1f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 220, 40), 1, cv2.LINE_AA)
        if args.save:
            cv2.imwrite(str(args.save), output)
            break
        if args.display:
            cv2.imshow("cube pose points", output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    capture.release()
    if args.display:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
