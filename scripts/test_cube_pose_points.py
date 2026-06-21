"""Display YOLO cube-pose Corner and edge keypoints on a centered viewport."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from live_recognition.detectors import YOLOCubeDetector  # noqa: E402

DEFAULT_BBOX_MODEL = ROOT / "runs/500ExamplesWeights.pt"
DEFAULT_CUBE_POSE_MODEL = (
    ROOT
    / "runs/cube_corner_edge_pose_local/runs/cube_corner_edge_yolo11n_pose/weights/best.pt"
)
DEFAULT_VIEW_SIZE = 256
CORNER_CLASS_NAME = "corner"
EDGE_CLASS_NAME = "edge"
CLASS_COLORS = {
    CORNER_CLASS_NAME: (255, 255, 255),
    EDGE_CLASS_NAME: (0, 255, 255),
}


@dataclass
class RoiTracker:
    roi_xyxy: tuple[int, int, int, int] | None = None
    cube_candidate: object | None = None
    base_side: int | None = None

    def reset(self) -> None:
        self.roi_xyxy = None
        self.cube_candidate = None
        self.base_side = None

    def initialize(
        self,
        candidate,
        frame_width: int,
        frame_height: int,
        padding: float,
    ) -> tuple[int, int, int, int]:
        self.cube_candidate = candidate
        self.roi_xyxy = square_roi_from_bbox(
            candidate.bbox_xyxy,
            frame_width,
            frame_height,
            padding,
        )
        x1, y1, x2, y2 = self.roi_xyxy
        self.base_side = min(x2 - x1, y2 - y1)
        return self.roi_xyxy

    def observe_tracking_bounds(
        self,
        tracking_bounds_xyxy: tuple[int, int, int, int] | None,
        frame_width: int,
        frame_height: int,
        tracking_gain: float = 1.0,
        tracking_padding: float = 0.0,
    ) -> None:
        if self.roi_xyxy is None or tracking_bounds_xyxy is None:
            return

        x1, y1, x2, y2 = self.roi_xyxy
        side = min(x2 - x1, y2 - y1)
        base_side = self.base_side if self.base_side is not None else side
        bx1, by1, bx2, by2 = tracking_bounds_xyxy
        tracking_point_x = (bx1 + bx2) / 2.0
        tracking_point_y = (by1 + by2) / 2.0
        roi_center_x = (x1 + x2) / 2.0
        roi_center_y = (y1 + y2) / 2.0
        observed_side = max(bx2 - bx1, by2 - by1)
        target_side = max(base_side, observed_side) * (
            1.0 + max(0.0, float(tracking_padding)) * 2.0
        )
        gain = max(0.0, float(tracking_gain))
        center_x = roi_center_x + (tracking_point_x - roi_center_x) * gain
        center_y = roi_center_y + (tracking_point_y - roi_center_y) * gain
        self.roi_xyxy = square_roi_from_center(
            center_x,
            center_y,
            target_side,
            frame_width,
            frame_height,
        )


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


def square_roi_from_bbox(
    bbox_xyxy: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * (1.0 + max(0.0, padding) * 2.0)
    return square_roi_from_center(center_x, center_y, side, frame_width, frame_height)


def square_roi_from_center(
    center_x: float,
    center_y: float,
    side: float,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    side = max(1, min(int(round(side)), frame_width, frame_height))
    x1 = int(round(center_x - side / 2.0))
    y1 = int(round(center_y - side / 2.0))
    x1 = max(0, min(frame_width - side, x1))
    y1 = max(0, min(frame_height - side, y1))
    return x1, y1, x1 + side, y1 + side


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
    return str(names.get(class_id, class_id)).lower()


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


def draw_keypoints(
    frame: np.ndarray,
    result,
    model,
    xy: np.ndarray | None,
    keypoint_conf: np.ndarray | None,
    *,
    corners_only: bool,
    offset_xy: tuple[int, int] = (0, 0),
    scale_xy: tuple[float, float] = (1.0, 1.0),
) -> tuple[np.ndarray, tuple[int, int, int, int] | None, int]:
    output = frame.copy()
    if xy is None or len(xy) == 0:
        cv2.putText(output, "no cube pose", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        return output, None, 0

    classes, box_conf = result_boxes(result)
    points_by_detection = np.asarray(xy, dtype=np.float32)
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
    for detection_index, points in enumerate(points_by_detection):
        class_id = (
            int(classes[detection_index])
            if classes is not None and detection_index < len(classes)
            else -1
        )
        class_name = class_name_for_id(result, model, class_id)
        if class_name not in {CORNER_CLASS_NAME, EDGE_CLASS_NAME}:
            continue
        if corners_only and class_name != CORNER_CLASS_NAME:
            continue
        if len(points) == 0:
            continue

        point = points[0]
        x = int(round(offset_x + float(point[0]) * scale_x))
        y = int(round(offset_y + float(point[1]) * scale_y))
        if not (0 <= x < output.shape[1] and 0 <= y < output.shape[0]):
            continue

        keypoint_score = (
            None
            if scores_by_detection is None
            or detection_index >= len(scores_by_detection)
            or len(np.asarray(scores_by_detection[detection_index]).reshape(-1)) == 0
            else float(np.asarray(scores_by_detection[detection_index]).reshape(-1)[0])
        )
        detection_score = (
            None
            if box_conf is None or detection_index >= len(box_conf)
            else float(box_conf[detection_index])
        )
        color = CLASS_COLORS.get(class_name, (255, 255, 255))
        if keypoint_score is not None and keypoint_score < 0.35:
            color = (120, 120, 120)

        cv2.circle(output, (x, y), 4, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(output, (x, y), 5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        label = "C" if class_name == CORNER_CLASS_NAME else "E"
        if detection_score is not None:
            label = f"{label} {detection_score:.2f}"
        cv2.putText(output, label, (x + 6, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1, cv2.LINE_AA)
        tracking_min_x = x if tracking_min_x is None else min(tracking_min_x, x)
        tracking_min_y = y if tracking_min_y is None else min(tracking_min_y, y)
        tracking_max_x = x if tracking_max_x is None else max(tracking_max_x, x)
        tracking_max_y = y if tracking_max_y is None else max(tracking_max_y, y)
        drawn += 1
    if drawn == 0:
        cv2.putText(output, "no corner/edge points", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
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
    height, width = frame.shape[:2]
    if roi_tracker.roi_xyxy is not None:
        return roi_tracker.roi_xyxy, roi_tracker.cube_candidate, "tracker"

    candidate = bbox_detector.detect(frame)
    if candidate is None:
        return None, None, "none"
    return (
        roi_tracker.initialize(candidate, width, height, args.roi_padding),
        candidate,
        "detector",
    )


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
    results = model.predict(source=model_roi, conf=args.conf, imgsz=args.imgsz, verbose=False)
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
        corners_only=args.corners_only,
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
    parser.add_argument("--conf", type=float, default=0.25)
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
    parser.add_argument("--corners-only", action="store_true", help="Draw only Corner keypoints")
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
