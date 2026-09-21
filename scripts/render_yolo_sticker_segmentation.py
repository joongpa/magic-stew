"""Display color-coded sticker masks from a YOLO segmentation model.

Examples:

    .venv/bin/python scripts/render_yolo_sticker_segmentation.py \
        --source path/to/cube.png
    .venv/bin/python scripts/render_yolo_sticker_segmentation.py \
        --source path/to/video.mp4

Press `q` or `Esc` in the OpenCV window to close a video or camera stream.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from live_recognition.detectors import YOLOCubeDetector  # noqa: E402

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm"}
DEFAULT_BBOX_MODEL = ROOT / "runs/500ExamplesWeights.pt"
STICKER_COLORS_BGR = {
    "white": (255, 255, 255),
    "yellow": (0, 255, 255),
    "red": (0, 0, 255),
    "orange": (0, 165, 255),
    "blue": (255, 0, 0),
    "green": (0, 180, 0),
}
STICKER_COLORS_BY_ID = tuple(STICKER_COLORS_BGR[name] for name in STICKER_COLORS_BGR)


def default_model_path(segmentation_type: str) -> Path:
    if segmentation_type == "whole-cube":
        candidates = (
            ROOT / "runs/cube_whole_seg_local/runs/cube_whole_yolo11n_seg/weights/best.pt",
            ROOT / "runs/segment/runs/cube_whole_seg_local/runs/cube_whole_yolo11n_seg/weights/best.pt",
        )
    else:
        candidates = (
            ROOT / "runs/segment/runs/cube_sticker_seg_local/runs/cube_sticker_yolo11n_seg/weights/best.pt",
            ROOT / "runs/cube_sticker_seg_local/runs/cube_sticker_yolo11n_seg/weights/best.pt",
        )
    return next((path for path in candidates if path.exists()), candidates[0])


def parse_source(source: str) -> int | str:
    source = source.strip().strip("\"'").strip()
    return int(source) if source.isdigit() else source


def is_image_source(source: int | str) -> bool:
    return isinstance(source, str) and Path(source).suffix.lower() in IMAGE_EXTENSIONS


def square_roi_from_center(
    center_x: float,
    center_y: float,
    side: float,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    side_px = max(1, min(int(round(side)), frame_width, frame_height))
    x1 = max(0, min(frame_width - side_px, int(round(center_x - side_px / 2.0))))
    y1 = max(0, min(frame_height - side_px, int(round(center_y - side_px / 2.0))))
    return x1, y1, x1 + side_px, y1 + side_px


def square_roi_from_bbox(
    bbox_xyxy: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    side = max(x2 - x1, y2 - y1) * (1.0 + padding * 2.0)
    return square_roi_from_center(
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
        side,
        frame_width,
        frame_height,
    )


class RoiTracker:
    """Track the sticker-derived ROI between cube-detector fallbacks."""

    def __init__(self) -> None:
        self.roi_xyxy: tuple[int, int, int, int] | None = None
        self.base_side: int | None = None

    def reset(self) -> None:
        self.roi_xyxy = None
        self.base_side = None

    def initialize(
        self,
        bbox_xyxy: tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
        padding: float,
    ) -> tuple[int, int, int, int]:
        self.roi_xyxy = square_roi_from_bbox(
            bbox_xyxy,
            frame_width,
            frame_height,
            padding,
        )
        x1, y1, x2, y2 = self.roi_xyxy
        self.base_side = min(x2 - x1, y2 - y1)
        return self.roi_xyxy

    def observe_tracking_bounds(
        self,
        bounds_xyxy: tuple[int, int, int, int] | None,
        frame_width: int,
        frame_height: int,
        tracking_gain: float = 1.4,
        tracking_padding: float = 0.18,
    ) -> None:
        if self.roi_xyxy is None:
            return
        if bounds_xyxy is None:
            self.reset()
            return

        x1, y1, x2, y2 = self.roi_xyxy
        side = min(x2 - x1, y2 - y1)
        base_side = self.base_side if self.base_side is not None else side
        bx1, by1, bx2, by2 = bounds_xyxy
        observed_side = max(bx2 - bx1, by2 - by1)
        target_side = max(base_side, observed_side) * (1.0 + tracking_padding * 2.0)
        dx = ((bx1 + bx2) / 2.0 - side / 2.0) * max(0.0, tracking_gain)
        dy = ((by1 + by2) / 2.0 - side / 2.0) * max(0.0, tracking_gain)
        self.roi_xyxy = square_roi_from_center(
            (x1 + x2) / 2.0 + dx,
            (y1 + y2) / 2.0 + dy,
            target_side,
            frame_width,
            frame_height,
        )


def class_name(names: object, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def class_color(names: object, class_id: int, segmentation_type: str) -> tuple[int, int, int]:
    if segmentation_type == "whole-cube":
        return (255, 0, 255)
    name = class_name(names, class_id).strip().lower()
    if name in STICKER_COLORS_BGR:
        return STICKER_COLORS_BGR[name]
    if 0 <= class_id < len(STICKER_COLORS_BY_ID):
        return STICKER_COLORS_BY_ID[class_id]
    return (180, 180, 180)


def tensor_to_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def render_segmentation_overlay(
    image: np.ndarray,
    result: object,
    opacity: float,
    segmentation_type: str,
) -> np.ndarray:
    overlay = image.copy()
    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if masks is None or boxes is None:
        return image.copy()

    mask_data = tensor_to_numpy(masks.data)
    class_ids = tensor_to_numpy(boxes.cls).reshape(-1)
    names = getattr(result, "names", {})
    height, width = image.shape[:2]

    for index, mask in enumerate(mask_data):
        if index >= len(class_ids):
            break
        mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST) >= 0.5
        color = class_color(names, int(class_ids[index]), segmentation_type)
        overlay[mask] = color

    output = cv2.addWeighted(overlay, opacity, image, 1.0 - opacity, 0.0)
    for index, mask in enumerate(mask_data):
        if index >= len(class_ids):
            break
        mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(
            (mask >= 0.5).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(
            output,
            contours,
            -1,
            class_color(names, int(class_ids[index]), segmentation_type),
            2,
        )
    return output


def has_segmentation(result: object) -> bool:
    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if masks is None or boxes is None:
        return False
    mask_data = tensor_to_numpy(masks.data)
    class_ids = tensor_to_numpy(boxes.cls).reshape(-1)
    return mask_data.ndim >= 3 and len(mask_data) > 0 and len(class_ids) > 0


def segmentation_bounds(
    result: object,
    roi_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    if not has_segmentation(result):
        return None
    masks = result.masks
    mask_data = tensor_to_numpy(masks.data)
    roi_height, roi_width = roi_shape
    combined = np.zeros((roi_height, roi_width), dtype=bool)
    for mask in mask_data:
        resized = cv2.resize(
            mask.astype(np.float32),
            (roi_width, roi_height),
            interpolation=cv2.INTER_LINEAR,
        )
        combined |= resized > 0.5
    ys, xs = np.nonzero(combined)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def predict(model: object, image: np.ndarray, args: argparse.Namespace) -> object | None:
    predictions = model.predict(
        source=image,
        conf=args.conf,
        imgsz=args.imgsz,
        verbose=False,
    )
    return predictions[0] if predictions else None


def predict_and_render(
    model: object,
    bbox_detector: YOLOCubeDetector | None,
    roi_tracker: RoiTracker,
    image: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, YOLOCubeDetector | None]:
    height, width = image.shape[:2]
    roi_source = "tracker"
    if roi_tracker.roi_xyxy is None:
        if bbox_detector is None:
            bbox_detector = YOLOCubeDetector(
                model_name=str(args.bbox_model),
                confidence_threshold=args.bbox_conf,
                image_size=args.bbox_imgsz,
            )
        candidate = bbox_detector.detect(image)
        if candidate is None:
            return image.copy(), bbox_detector
        roi_tracker.initialize(candidate.bbox_xyxy, width, height, args.roi_padding)
        roi_source = "detector"

    roi_xyxy = roi_tracker.roi_xyxy
    assert roi_xyxy is not None
    x1, y1, x2, y2 = roi_xyxy
    roi = image[y1:y2, x1:x2]
    model_roi = resize_to_square_size(roi, args.imgsz)
    roi_result = predict(model, model_roi, args)
    if roi_result is None or not has_segmentation(roi_result):
        output = image.copy()
        if args.show_bbox:
            cv2.rectangle(output, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), 2)
        roi_tracker.reset()
        return output, bbox_detector

    roi_tracker.observe_tracking_bounds(
        segmentation_bounds(roi_result, roi.shape[:2]),
        width,
        height,
        tracking_gain=args.roi_tracking_gain,
        tracking_padding=args.roi_tracking_padding,
    )
    rendered_roi = render_segmentation_overlay(
        roi,
        roi_result,
        args.opacity,
        args.segmentation_type,
    )
    output = image.copy()
    output[y1:y2, x1:x2] = rendered_roi
    if args.show_bbox:
        roi_color = (0, 0, 255) if roi_source == "detector" else (40, 220, 40)
        cv2.rectangle(output, (x1, y1), (x2 - 1, y2 - 1), roi_color, 2)
    return output, bbox_detector


def resize_to_square_size(image: np.ndarray, size: int) -> np.ndarray:
    if size <= 0:
        return image
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def create_video_writer(path: Path, frame: np.ndarray, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    fourcc = "mp4v" if suffix in {".mp4", ".m4v", ".mov"} else "MJPG"
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*fourcc),
        fps if fps > 0.0 else 30.0,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Display YOLO segmentation masks over original images")
    parser.add_argument("--source", required=True, help="Image, video path, or camera index")
    parser.add_argument(
        "--segmentation-type",
        choices=("sticker", "whole-cube"),
        default="sticker",
        help="Segmentation model and rendering mode",
    )
    parser.add_argument("--model", type=Path, help="Override the selected segmentation model weights path")
    parser.add_argument("--bbox-model", type=Path, default=DEFAULT_BBOX_MODEL, help="Cube bbox YOLO weights path")
    parser.add_argument("--bbox-conf", type=float, default=0.25, help="Cube bbox confidence threshold")
    parser.add_argument("--bbox-imgsz", type=int, default=640, help="Cube bbox inference image size")
    parser.add_argument("--roi-padding", type=float, default=0.08, help="Padding ratio around the bbox ROI")
    parser.add_argument("--roi-tracking-gain", type=float, default=1.4, help="How strongly segmentation moves the ROI")
    parser.add_argument("--roi-tracking-padding", type=float, default=0.18, help="Padding around tracked segmentation")
    parser.add_argument("--show-bbox", action=argparse.BooleanOptionalAction, default=True, help="Draw the active ROI")
    parser.add_argument("--conf", type=float, default=0.35, help="Segmentation confidence threshold")
    parser.add_argument("--imgsz", type=int, default=256, help="YOLO inference image size")
    parser.add_argument("--opacity", type=float, default=0.45, help="Mask opacity in the range [0, 1]")
    parser.add_argument("--once", action="store_true", help="Process one frame and exit")
    parser.add_argument("--save", type=Path, help="Optional output image path for image or --once mode")
    parser.add_argument("--save-video", type=Path, help="Optional output path for rendered video frames")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0.0 <= args.opacity <= 1.0:
        raise SystemExit("--opacity must be in the range [0, 1]")
    model_path = args.model or default_model_path(args.segmentation_type)
    if not model_path.is_file():
        raise SystemExit(f"Could not find model weights: {model_path}")

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    bbox_detector = None
    roi_tracker = RoiTracker()

    source = parse_source(args.source)
    if is_image_source(source):
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"Could not read image: {source}")
        rendered, bbox_detector = predict_and_render(model, bbox_detector, roi_tracker, image, args)
        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.save), rendered):
                raise SystemExit(f"Could not write rendered overlay: {args.save}")
            print(f"Saved overlay: {args.save}")
        if args.save_video:
            raise SystemExit("--save-video requires a video file or camera source")
        cv2.imshow("sticker color segmentation", rendered)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return 0

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {source}")

    writer = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rendered, bbox_detector = predict_and_render(model, bbox_detector, roi_tracker, frame, args)
            if args.save_video:
                if writer is None:
                    writer = create_video_writer(
                        args.save_video,
                        rendered,
                        float(capture.get(cv2.CAP_PROP_FPS)),
                    )
                writer.write(rendered)
            if args.save and args.once:
                args.save.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(args.save), rendered):
                    raise SystemExit(f"Could not write rendered overlay: {args.save}")
            cv2.imshow("sticker color segmentation", rendered)
            key = cv2.waitKey(0 if args.once else 1) & 0xFF
            if args.once or key in (ord("q"), 27):
                break
    finally:
        if writer is not None:
            writer.release()
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
