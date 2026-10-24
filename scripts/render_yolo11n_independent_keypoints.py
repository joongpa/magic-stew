#!/usr/bin/env python3
"""Render the independent cube-keypoint YOLO model with the live ROI flow.

Per frame, this renderer:

1. Runs the cube bbox detector when no tracked ROI exists.
2. Builds the padded square ROI using the shared live-pipeline helper.
3. Runs the combined cube/sticker segmentation and independent three-class pose models inside that ROI.
4. Maps the independent points and sticker masks back to the full frame.
5. Updates the ROI tracker from the predicted point extent.

Examples:

    .venv/bin/python scripts/render_yolo11n_independent_keypoints.py \
        --source data/independent_keypoints_dataset_v2/images/000000.png \
        --model runs/pose/runs/independent_keypoint_pose_v2/runs/cube_piece_keypoint_yolo11n_pose/weights/best.pt \
        --save runs/independent_keypoints_render.png \
        --no-display

    .venv/bin/python scripts/render_yolo11n_independent_keypoints.py \
        --source 0 \
        --model runs/pose/runs/independent_keypoint_pose_v2/runs/cube_piece_keypoint_yolo11n_pose/weights/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from live_recognition.detectors import (  # noqa: E402
    YOLOCubeDetector,
    default_yolo_device,
)
from live_recognition.independent_pose_thresholds import (  # noqa: E402
    DEFAULT_INDEPENDENT_POSE_CLASS_CONFIDENCES,
    INDEPENDENT_POSE_CLASS_NAMES,
    independent_pose_class_threshold,
    parse_independent_pose_class_confidences,
)
from live_recognition.roi_tracking import (  # noqa: E402
    RoiTracker,
    roi_for_frame as shared_roi_for_frame,
)


CLASS_NAMES = INDEPENDENT_POSE_CLASS_NAMES
# OpenCV uses BGR colors. These match the independent-keypoint generator.
CLASS_COLORS_BGR = (
    (255, 255, 255),
    (0, 255, 255),
    (255, 120, 255),
)
OCCLUDED_COLOR_BGR = (0, 0, 255)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
WINDOW_NAME = "independent cube keypoints"
DEFAULT_BBOX_MODEL = ROOT / "runs/500ExamplesWeights.pt"
DEFAULT_POSE_MODEL = (
    ROOT
    / "runs/pose/runs/independent_keypoint_pose_v2/runs/cube_piece_keypoint_yolo11n_pose/weights/best.pt"
)
DEFAULT_COMBINED_SEGMENTATION_MODEL_CANDIDATES = (
    ROOT / "runs/segment/runs/cube_sticker_combined_seg_v2/runs/cube_sticker_yolo11n_seg/weights/best.pt",
    ROOT / "runs/segment/runs/cube_sticker_combined_seg/runs/cube_sticker_yolo11n_seg/weights/best.pt",
)
DEFAULT_COMBINED_SEGMENTATION_MODEL = next(
    (
        path
        for path in DEFAULT_COMBINED_SEGMENTATION_MODEL_CANDIDATES
        if path.exists()
    ),
    DEFAULT_COMBINED_SEGMENTATION_MODEL_CANDIDATES[0],
)
STICKER_COLORS_BGR = {
    "white": (255, 255, 255),
    "yellow": (0, 255, 255),
    "red": (0, 0, 255),
    "orange": (0, 165, 255),
    "blue": (255, 0, 0),
    "green": (0, 180, 0),
    "sticker": (0, 220, 255),
}
STICKER_BORDER_COLOR_BGR = (0, 0, 255)


def parse_source(source: str) -> int | str:
    source = source.strip().strip("\"'").strip()
    return int(source) if source.isdigit() else source


def is_image_source(source: int | str) -> bool:
    return isinstance(source, str) and Path(source).suffix.lower() in IMAGE_EXTENSIONS


def tensor_to_numpy(value: object) -> np.ndarray | None:
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


def pose_predictions(result: object) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return points, point confidence, class IDs, and box confidence."""

    keypoints = getattr(result, "keypoints", None)
    boxes = getattr(result, "boxes", None)
    if keypoints is None or boxes is None:
        return (np.empty((0, 2), dtype=np.float32),) * 2 + (
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
        )

    points = tensor_to_numpy(getattr(keypoints, "xy", None))
    if points is None:
        return (np.empty((0, 2), dtype=np.float32),) * 2 + (
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
        )
    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 3:
        points = points[:, 0, :]
    if points.ndim != 2 or points.shape[1] < 2:
        raise RuntimeError(f"Unexpected independent pose point shape: {points.shape}")
    points = points[:, :2]

    point_confidence = tensor_to_numpy(getattr(keypoints, "conf", None))
    if point_confidence is None:
        point_confidence = np.ones((len(points),), dtype=np.float32)
    else:
        point_confidence = np.asarray(point_confidence, dtype=np.float32).reshape(len(points), -1)[:, 0]

    class_ids = tensor_to_numpy(getattr(boxes, "cls", None))
    if class_ids is None:
        class_ids = np.zeros((len(points),), dtype=np.int32)
    else:
        class_ids = np.asarray(class_ids, dtype=np.float32).reshape(-1).astype(np.int32)

    box_confidence = tensor_to_numpy(getattr(boxes, "conf", None))
    if box_confidence is None:
        box_confidence = np.ones((len(points),), dtype=np.float32)
    else:
        box_confidence = np.asarray(box_confidence, dtype=np.float32).reshape(-1)

    count = min(len(points), len(point_confidence), len(class_ids), len(box_confidence))
    return points[:count], point_confidence[:count], class_ids[:count], box_confidence[:count]


def class_name(names: object, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), class_id)))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def sticker_color(names: object, class_id: int) -> tuple[int, int, int]:
    name = class_name(names, class_id).strip().lower().replace("_", " ").replace("-", " ")
    return STICKER_COLORS_BGR.get(name, STICKER_COLORS_BGR["sticker"])


def sticker_class_ids(names: object) -> set[int] | None:
    """Return allowed sticker IDs, or None when model metadata is unknown."""

    if isinstance(names, dict):
        class_ids = {
            int(class_id)
            for class_id in names
            if str(class_id).lstrip("-").isdigit()
        }
    elif isinstance(names, (list, tuple)):
        class_ids = set(range(len(names)))
    else:
        return None

    explicit_sticker_ids = {
        class_id
        for class_id in class_ids
        if class_name(names, class_id).strip().lower() == "sticker"
    }
    if explicit_sticker_ids:
        return explicit_sticker_ids

    color_ids = {
        class_id
        for class_id in class_ids
        if class_name(names, class_id).strip().lower() in STICKER_COLORS_BGR
    }
    return color_ids or None


def predict_sticker_result(
    model: object,
    model_roi: np.ndarray,
    args: argparse.Namespace,
) -> object | None:
    predict_kwargs = {} if args.device is None else {"device": args.device}
    results = model.predict(
        source=model_roi,
        conf=args.sticker_mask_conf,
        imgsz=args.imgsz,
        retina_masks=False,
        verbose=False,
        **predict_kwargs,
    )
    return results[0] if results else None


def sticker_masks_from_result(
    result: object | None,
) -> tuple[list[np.ndarray], object, np.ndarray] | None:
    if result is None:
        return None
    masks = getattr(result, "masks", None)
    if masks is None:
        return None
    mask_data = tensor_to_numpy(getattr(masks, "data", None))
    if mask_data is None or mask_data.ndim < 3 or len(mask_data) == 0:
        return None
    boxes = getattr(result, "boxes", None)
    class_ids = tensor_to_numpy(getattr(boxes, "cls", None)) if boxes is not None else None
    if class_ids is None:
        class_ids = np.zeros((len(mask_data),), dtype=np.int32)
    class_ids = np.asarray(class_ids, dtype=np.int32).reshape(-1)
    names = getattr(result, "names", {})
    allowed_sticker_ids = sticker_class_ids(names)
    # The combined model uses a literal "sticker" class. Keep only that
    # class, excluding the model's separate "cube" predictions. Resolve the
    # allowed IDs from the complete class map so cube-only frames do not fall
    # through to the generic fallback.
    if allowed_sticker_ids is not None:
        keep = [
            index
            for index, class_id in enumerate(class_ids[: len(mask_data)])
            if int(class_id) in allowed_sticker_ids
        ]
    else:
        keep = list(range(min(len(mask_data), len(class_ids))))
    return [np.asarray(mask_data[index]) >= 0.5 for index in keep], names, class_ids[keep]


def predict_sticker_masks(
    model: object,
    model_roi: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], object, np.ndarray] | None:
    return sticker_masks_from_result(predict_sticker_result(model, model_roi, args))


def predict_pose_result(
    model: object,
    model_roi: np.ndarray,
    args: argparse.Namespace,
) -> object | None:
    predict_kwargs = {} if args.device is None else {"device": args.device}
    pose_inference_conf = (
        args.pose_conf
        if args.pose_conf is not None
        else min(args.pose_class_conf.values())
    )
    results = model.predict(
        source=model_roi,
        conf=pose_inference_conf,
        imgsz=args.imgsz,
        max_det=args.max_det,
        verbose=False,
        **predict_kwargs,
    )
    return results[0] if results else None


def cuda_stream_device(torch_module: object, device: object) -> object:
    device_text = str(device)
    if device_text.startswith("cuda"):
        return torch_module.device(device_text)
    if device_text.isdigit():
        return torch_module.device(f"cuda:{device_text}")
    return torch_module.device(f"cuda:{torch_module.cuda.current_device()}")


def predict_models_parallel_cuda(
    sticker_seg_model: object | None,
    pose_model: object,
    model_roi: np.ndarray,
    args: argparse.Namespace,
) -> tuple[tuple[list[np.ndarray], object, np.ndarray] | None, object | None]:
    """Launch segmentation and pose on separate CUDA streams, then synchronize."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("--parallel-cuda requires a CUDA-enabled PyTorch installation")

    stream_device = cuda_stream_device(torch, args.device)
    segmentation_stream = torch.cuda.Stream(device=stream_device)
    pose_stream = torch.cuda.Stream(device=stream_device)
    with torch.cuda.device(stream_device):
        with torch.cuda.stream(segmentation_stream):
            sticker_result = (
                predict_sticker_result(sticker_seg_model, model_roi, args)
                if sticker_seg_model is not None and args.run_sticker_segmentation
                else None
            )
        with torch.cuda.stream(pose_stream):
            pose_result = predict_pose_result(pose_model, model_roi, args)
        torch.cuda.synchronize(stream_device)

    return sticker_masks_from_result(sticker_result), pose_result


def render_sticker_masks(
    image: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
    predicted: tuple[list[np.ndarray], object, np.ndarray] | None,
    opacity: float,
    border_thickness: int,
) -> np.ndarray:
    output = image.copy()
    if predicted is None:
        return output
    local_masks, names, class_ids = predicted
    x1, y1, x2, y2 = roi_xyxy
    roi_height, roi_width = y2 - y1, x2 - x1
    roi = output[y1:y2, x1:x2]
    for local_mask, class_id in zip(local_masks, class_ids, strict=True):
        mask = cv2.resize(
            local_mask.astype(np.uint8),
            (roi_width, roi_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        color_layer = np.full_like(roi, sticker_color(names, int(class_id)))
        blended = cv2.addWeighted(roi, 1.0 - opacity, color_layer, opacity, 0.0)
        roi[mask] = blended[mask]
        if border_thickness > 0:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(
                roi,
                contours,
                -1,
                STICKER_BORDER_COLOR_BGR,
                border_thickness,
                cv2.LINE_AA,
            )
    return output


def local_to_frame_points(
    local_points: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
    model_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = roi_xyxy
    scale = np.asarray(
        ((x2 - x1) / float(model_size), (y2 - y1) / float(model_size)),
        dtype=np.float32,
    )
    frame_points = local_points * scale + np.asarray((x1, y1), dtype=np.float32)
    return frame_points, local_points * scale


def tracking_bounds(
    local_roi_points: np.ndarray,
    point_confidences: np.ndarray,
    class_ids: np.ndarray,
    box_confidences: np.ndarray,
    class_confidences: dict[str, float],
    point_threshold: float,
    roi_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    height, width = roi_shape
    usable = [
        (float(point[0]), float(point[1]))
        for point, point_confidence, class_id, box_confidence in zip(
            local_roi_points,
            point_confidences,
            class_ids,
            box_confidences,
            strict=True,
        )
        if 0 <= int(class_id) < len(CLASS_NAMES)
        and float(box_confidence)
        >= independent_pose_class_threshold(CLASS_NAMES[int(class_id)], class_confidences)
        and float(point_confidence) >= point_threshold
        and np.isfinite(point).all()
        and 0 <= float(point[0]) < width
        and 0 <= float(point[1]) < height
    ]
    if not usable:
        return None
    points = np.asarray(usable, dtype=np.float32)
    return (
        int(np.floor(points[:, 0].min())),
        int(np.floor(points[:, 1].min())),
        int(np.ceil(points[:, 0].max())),
        int(np.ceil(points[:, 1].max())),
    )


def draw_label(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    x, y = position
    cv2.putText(
        image,
        text,
        (x + 1, y + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_pose(
    image: np.ndarray,
    frame_points: np.ndarray,
    point_confidences: np.ndarray,
    class_ids: np.ndarray,
    box_confidences: np.ndarray,
    point_threshold: float,
    class_confidences: dict[str, float],
    show_labels: bool,
    show_confidences: bool,
    copy_image: bool = True,
) -> tuple[np.ndarray, int]:
    output = image.copy() if copy_image else image
    drawn = 0
    per_class_index = {class_id: 0 for class_id in range(len(CLASS_NAMES))}
    for point, point_confidence, class_id, box_confidence in zip(
        frame_points,
        point_confidences,
        class_ids,
        box_confidences,
        strict=True,
    ):
        x, y = np.round(point).astype(np.int32)
        if not (0 <= x < output.shape[1] and 0 <= y < output.shape[0]):
            continue
        if not 0 <= int(class_id) < len(CLASS_NAMES):
            continue
        class_name = CLASS_NAMES[int(class_id)]
        class_threshold = independent_pose_class_threshold(class_name, class_confidences)
        if float(box_confidence) < class_threshold or float(point_confidence) < point_threshold:
            continue
        class_index = per_class_index[int(class_id)]
        per_class_index[int(class_id)] += 1
        color = CLASS_COLORS_BGR[int(class_id)]
        radius = max(1, round(min(output.shape[:2]) / 128.0 * 0.8))
        cv2.circle(output, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
        if show_labels:
            label = f"{class_name}:{class_index}"
            if show_confidences:
                label += f" {float(point_confidence):.2f}/{float(box_confidence):.2f}"
            draw_label(output, label, (int(x) + radius + 3, int(y) - 5), color)
        drawn += 1
    return output, drawn


def draw_context(
    image: np.ndarray,
    roi_xyxy: tuple[int, int, int, int] | None,
    candidate: object | None,
    roi_source: str,
    point_count: int,
    show_status: bool,
) -> np.ndarray:
    output = image.copy()
    # The tracker retains the detector candidate so it can keep the original
    # ROI geometry, but that candidate is not a fresh detection. Only show the
    # blue detector box on frames where the detector supplied the ROI.
    if roi_source == "detector" and candidate is not None and hasattr(candidate, "bbox_xyxy"):
        bx1, by1, bx2, by2 = tuple(int(value) for value in candidate.bbox_xyxy)
        cv2.rectangle(output, (bx1, by1), (bx2 - 1, by2 - 1), (255, 120, 0), 1, cv2.LINE_AA)
    if roi_xyxy is not None:
        x1, y1, x2, y2 = roi_xyxy
        cv2.rectangle(output, (x1, y1), (x2 - 1, y2 - 1), (0, 220, 0), 1, cv2.LINE_AA)
        status = f"{roi_source} ROI | {point_count} points"
    else:
        status = "no cube bbox"
    if show_status:
        cv2.putText(output, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 220, 40), 1, cv2.LINE_AA)
    return output


def render_frame(
    frame: np.ndarray,
    bbox_detector: YOLOCubeDetector,
    pose_model: object,
    roi_tracker: RoiTracker,
    args: argparse.Namespace,
    sticker_seg_model: object | None = None,
) -> tuple[np.ndarray, str, int]:
    height, width = frame.shape[:2]
    roi_xyxy, candidate, roi_source = shared_roi_for_frame(
        frame,
        bbox_detector,
        roi_tracker,
        args.roi_padding,
    )
    if roi_xyxy is None:
        return draw_context(frame, None, None, "none", 0, args.show_status), "none", 0

    x1, y1, x2, y2 = roi_xyxy
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        roi_tracker.reset()
        return draw_context(frame, roi_xyxy, candidate, roi_source, 0, args.show_status), roi_source, 0

    model_roi = resize_to_square(roi, args.view_size)
    if args.parallel_cuda:
        sticker_masks, result = predict_models_parallel_cuda(
            sticker_seg_model,
            pose_model,
            model_roi,
            args,
        )
    else:
        sticker_masks = (
            predict_sticker_masks(sticker_seg_model, model_roi, args)
            if sticker_seg_model is not None and args.run_sticker_segmentation
            else None
        )
        result = predict_pose_result(pose_model, model_roi, args)
    output = render_sticker_masks(
        frame,
        roi_xyxy,
        sticker_masks if args.show_sticker_mask else None,
        args.sticker_mask_opacity,
        args.sticker_border_thickness,
    )
    if result is None:
        roi_tracker.reset()
        return draw_context(output, roi_xyxy, candidate, roi_source, 0, args.show_status), roi_source, 0

    local_points, point_confidences, class_ids, box_confidences = pose_predictions(result)
    if len(local_points) == 0:
        roi_tracker.reset()
        return draw_context(output, roi_xyxy, candidate, roi_source, 0, args.show_status), roi_source, 0

    frame_points, local_roi_points = local_to_frame_points(local_points, roi_xyxy, args.view_size)
    output, point_count = draw_pose(
        output,
        frame_points,
        point_confidences,
        class_ids,
        box_confidences,
        args.keypoint_conf,
        args.pose_class_conf,
        args.show_labels,
        args.show_confidences,
        copy_image=False,
    )
    output = draw_context(output, roi_xyxy, candidate, roi_source, point_count, args.show_status)

    # Match the live pipeline: keep using this ROI until the independent pose
    # model fails, then fall back to the bbox detector on the next frame.
    roi_tracker.observe_tracking_bounds(
        tracking_bounds(
            local_roi_points,
            point_confidences,
            class_ids,
            box_confidences,
            args.pose_class_conf,
            args.track_conf,
            roi.shape[:2],
        ),
        width,
        height,
        tracking_gain=args.roi_tracking_gain,
        tracking_padding=args.roi_tracking_padding,
    )
    return output, roi_source, point_count


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


def configure_camera(capture: cv2.VideoCapture, args: argparse.Namespace) -> None:
    """Request a low-latency camera mode and report what OpenCV negotiated."""

    if args.camera_width > 0:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    if args.camera_height > 0:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    if args.camera_fps > 0:
        capture.set(cv2.CAP_PROP_FPS, args.camera_fps)

    try:
        backend = capture.getBackendName()
    except cv2.error:
        backend = "unknown"
    negotiated = (
        f"{capture.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"
        f"{capture.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}@"
        f"{capture.get(cv2.CAP_PROP_FPS):.1f}"
    )
    print(
        "Camera: "
        f"backend={backend}, "
        f"requested={args.camera_width}x{args.camera_height}@{args.camera_fps:g}, "
        f"negotiated={negotiated}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the independent cube-keypoint model with bbox-to-ROI tracking."
    )
    parser.add_argument("--source", required=True, help="Image, video path, or camera index.")
    parser.add_argument("--camera-width", type=int, default=640, help="Requested camera width (0 leaves it unchanged).")
    parser.add_argument("--camera-height", type=int, default=480, help="Requested camera height (0 leaves it unchanged).")
    parser.add_argument("--camera-fps", type=float, default=30.0, help="Requested camera FPS (0 leaves it unchanged).")
    parser.add_argument("--model", type=Path, default=DEFAULT_POSE_MODEL)
    parser.add_argument(
        "--combined-seg-model",
        "--sticker-seg-model",
        dest="sticker_seg_model",
        type=Path,
        default=DEFAULT_COMBINED_SEGMENTATION_MODEL,
        help="Combined cube/sticker segmentation checkpoint; only its sticker class is rendered.",
    )
    parser.add_argument("--bbox-model", type=Path, default=DEFAULT_BBOX_MODEL)
    parser.add_argument("--device", default=None, help="Ultralytics device, for example cpu, mps, or 0.")
    parser.add_argument("--bbox-conf", type=float, default=0.25)
    parser.add_argument("--bbox-imgsz", type=int, default=640)
    parser.add_argument(
        "--pose-conf",
        type=float,
        default=None,
        help="Global model pre-filter; defaults to the lowest per-class threshold.",
    )
    parser.add_argument(
        "--pose-class-conf",
        type=parse_independent_pose_class_confidences,
        default=dict(DEFAULT_INDEPENDENT_POSE_CLASS_CONFIDENCES),
        help=(
            "Per-class boxes.conf thresholds as class=value pairs, for example "
            "corner=0.25,corner-edge=0.30,corner-edge-center=0.25."
        ),
    )
    parser.add_argument(
        "--keypoint-conf",
        "--min-keypoint-conf",
        "--keypoint-confidence",
        "--min-keypoint-confidence",
        dest="keypoint_conf",
        type=float,
        default=0.05,
        help="Minimum independent pose keypoint confidence for drawing points (default: 0.05).",
    )
    parser.add_argument("--track-conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--view-size", type=int, default=256)
    parser.add_argument("--max-det", type=int, default=64)
    parser.add_argument("--sticker-mask-conf", type=float, default=0.25)
    parser.add_argument("--sticker-mask-opacity", type=float, default=0.25)
    parser.add_argument("--sticker-border-thickness", type=int, default=2)
    parser.add_argument(
        "--run-sticker-segmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the combined segmentation model even when its mask is not displayed.",
    )
    parser.add_argument("--show-sticker-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--parallel-cuda",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Experimental: launch sticker segmentation and pose inference on separate CUDA streams.",
    )
    parser.add_argument("--roi-padding", type=float, default=0.08)
    parser.add_argument("--roi-tracking-gain", type=float, default=1.4)
    parser.add_argument("--roi-tracking-padding", type=float, default=0.18)
    parser.add_argument("--show-labels", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--show-confidences", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-status", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--show-fps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save", type=Path, help="Output image path, or one frame with --once.")
    parser.add_argument("--save-video", type=Path, help="Output path for rendered video.")
    parser.add_argument("--once", action="store_true", help="Process one video/camera frame and exit.")
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.set_defaults(display=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.device = args.device or default_yolo_device()
    print(f"Inference device: {args.device}", flush=True)
    if args.parallel_cuda and not (str(args.device).startswith("cuda") or str(args.device).isdigit()):
        raise SystemExit("--parallel-cuda requires --device cuda, cuda:N, or a CUDA device index")
    if args.pose_conf is not None and not 0.0 <= args.pose_conf <= 1.0:
        raise SystemExit("--pose-conf must be in [0, 1]")
    if args.pose_conf is not None and args.pose_conf > min(args.pose_class_conf.values()):
        raise SystemExit("--pose-conf cannot exceed the lowest --pose-class-conf threshold")
    if not 0.0 <= args.sticker_mask_conf <= 1.0:
        raise SystemExit("--sticker-mask-conf must be in [0, 1]")
    if not 0.0 <= args.sticker_mask_opacity <= 1.0:
        raise SystemExit("--sticker-mask-opacity must be in [0, 1]")
    if args.sticker_border_thickness < 0:
        raise SystemExit("--sticker-border-thickness must be non-negative")
    if not args.model.is_file():
        raise SystemExit(f"Could not find independent pose model: {args.model}")
    if args.run_sticker_segmentation and not args.sticker_seg_model.is_file():
        raise SystemExit(f"Could not find sticker segmentation model: {args.sticker_seg_model}")
    if not args.bbox_model.is_file():
        raise SystemExit(f"Could not find bbox model: {args.bbox_model}")

    from ultralytics import YOLO

    bbox_detector = YOLOCubeDetector(
        model_name=str(args.bbox_model),
        confidence_threshold=args.bbox_conf,
        image_size=args.bbox_imgsz,
        device=args.device,
    )
    pose_model = YOLO(str(args.model))
    sticker_seg_model = YOLO(str(args.sticker_seg_model)) if args.run_sticker_segmentation else None
    if sticker_seg_model is not None:
        print(f"Combined segmentation model (sticker class only): {args.sticker_seg_model}", flush=True)
    source = parse_source(args.source)
    roi_tracker = RoiTracker()

    if is_image_source(source):
        frame = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if frame is None:
            raise SystemExit(f"Could not read image: {source}")
        rendered, roi_source, point_count = render_frame(
            frame,
            bbox_detector,
            pose_model,
            roi_tracker,
            args,
            sticker_seg_model,
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
        print(f"ROI source: {roi_source}; rendered points: {point_count}")
        return 0

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {source}")
    if isinstance(source, int):
        configure_camera(capture, args)
    writer = None
    previous_loop_end = perf_counter()
    smoothed_loop_fps = None
    smoothed_render_fps = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            render_start = perf_counter()
            now = render_start
            loop_fps = 1.0 / max(now - previous_loop_end, 1e-6)
            previous_loop_end = now
            smoothed_loop_fps = (
                loop_fps
                if smoothed_loop_fps is None
                else 0.9 * smoothed_loop_fps + 0.1 * loop_fps
            )
            rendered, _roi_source, _point_count = render_frame(
                frame,
                bbox_detector,
                pose_model,
                roi_tracker,
                args,
                sticker_seg_model,
            )
            render_end = perf_counter()
            render_fps = 1.0 / max(render_end - render_start, 1e-6)
            smoothed_render_fps = (
                render_fps
                if smoothed_render_fps is None
                else 0.9 * smoothed_render_fps + 0.1 * render_fps
            )
            if args.show_fps:
                fps_y = 44 if args.show_status else 22
                cv2.putText(
                    rendered,
                    f"PROC {smoothed_render_fps:.1f} | LOOP {smoothed_loop_fps:.1f}",
                    (8, fps_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (40, 220, 40),
                    1,
                    cv2.LINE_AA,
                )
            if args.save_video:
                if writer is None:
                    writer = create_video_writer(args.save_video, rendered, float(capture.get(cv2.CAP_PROP_FPS)))
                writer.write(rendered)
            if args.save and args.once:
                args.save.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(args.save), rendered):
                    raise SystemExit(f"Could not write rendered image: {args.save}")
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
