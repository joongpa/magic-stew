#!/usr/bin/env python3
"""Render RF-DETR or YOLO pose predictions inside the shared cube ROI.

The input flow matches the live pipeline used by the other renderers:

1. Detect the cube with the trained YOLO bounding-box model.
2. Expand that box into a padded square ROI.
3. Run the selected pose model on the square crop.
4. Project the predicted keypoints back onto the full frame.

The three independent keypoint classes use the same colors as the Blender
generator and the existing independent-keypoint renderer:

    corner             white
    corner-edge        yellow
    corner-edge-center pink/magenta

Examples:

    .venv/bin/python scripts/render_rfdetr_keypoints.py \
        --source data/independent_keypoints_dataset_v2/images/000000.png \
        --save runs/rfdetr_keypoint_try.png

    .venv/bin/python scripts/render_rfdetr_keypoints.py \
        --source data/independent_keypoints_dataset_v2/images/000000.png \
        --model runs/pose/runs/rfdetr \
        --device cuda:0 \
        --save runs/rfdetr_keypoint_try.png \
        --no-display

    .venv/bin/python scripts/render_rfdetr_keypoints.py \
        --source data/keypoints_only_dataset/images/000000.png \
        --model runs/yolo11n_keypoints_local/runs/cube_keypoints_yolo11n_pose/weights/best.pt \
        --no-display

The model type is inferred from the checkpoint: ``.pth`` is treated as
RF-DETR and ``.pt`` as Ultralytics YOLO. Use ``--model-type`` to override
that inference when a checkpoint uses a nonstandard extension.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from live_recognition.detectors import YOLOCubeDetector  # noqa: E402
from live_recognition.roi_tracking import (RoiTracker,  # noqa: E402
                                           roi_for_frame)

# This checkpoint was trained from data/keypoints_only_dataset, whose combined
# landmark/sticker manifest contains three geometric landmark classes followed
# by six sticker-color classes.
CLASS_NAMES = (
    "corner",
    "corner-edge",
    "corner-edge-center",
    "white",
    "yellow",
    "orange",
    "red",
    "blue",
    "green",
)
CLASS_LABELS = (
    "C",
    "CE",
    "CEC",
    "W",
    "Y",
    "O",
    "R",
    "B",
    "G",
)
LANDMARK_CLASS_COUNT = 3
# OpenCV uses BGR colors. The first three match the independent-keypoint
# renderer; the sticker colors match STICKER_KEYPOINT_MASK_COLORS in the
# Blender generator.
CLASS_COLORS_BGR = (
    (255, 255, 255),  # corner -> white
    (0, 255, 255),  # corner-edge -> yellow
    (255, 120, 255),  # corner-edge-center -> pink/magenta
    (255, 255, 255),  # white
    (0, 238, 245),  # yellow
    (0, 146, 242),  # orange
    (1, 32, 231),  # red
    (227, 177, 101),  # blue
    (1, 227, 80),  # green
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".avi", ".mkv"}
DEFAULT_CHECKPOINT = Path("runs/pose/runs/rfdetr/checkpoint_best_total.pth")
DEFAULT_BBOX_MODEL = ROOT / "runs/500ExamplesWeights.pt"


def resolve_checkpoint(path: Path) -> Path:
    """Resolve a checkpoint file or a training output directory.

    RF-DETR and YOLO training write different checkpoint extensions and names.
    Prefer conventional best checkpoints, then the newest remaining weight.
    """

    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Pose checkpoint path not found: {path}")

    candidates = [
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in {".pth", ".pt"}
    ]
    if not candidates:
        raise FileNotFoundError(f"No .pth or .pt checkpoint found under: {path}")

    preferred_names = (
        "checkpoint_best_total.pth",
        "checkpoint_best_ema.pth",
        "checkpoint_best_regular.pth",
        "best.pt",
        "last.pt",
    )
    for name in preferred_names:
        preferred = [candidate for candidate in candidates if candidate.name == name]
        if preferred:
            return max(preferred, key=lambda candidate: candidate.stat().st_mtime)
    return max(candidates, key=lambda candidate: candidate.stat().st_mtime)


def infer_model_type(checkpoint: Path, requested: str) -> str:
    """Infer the pose backend from the checkpoint extension when requested."""

    if requested != "auto":
        return requested
    suffix = checkpoint.suffix.lower()
    if suffix == ".pth":
        return "rfdetr"
    if suffix == ".pt":
        return "yolo"
    raise SystemExit(
        f"Cannot infer model type from checkpoint extension: {checkpoint}. "
        "Use --model-type rfdetr or --model-type yolo."
    )


def parse_source(source: str) -> int | Path:
    """Treat a bare integer as a camera index and everything else as a path."""

    stripped = source.strip()
    if stripped.isdigit():
        return int(stripped)
    return Path(source)


def as_numpy(value: Any, dtype: Any = np.float32) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    return np.asarray(value, dtype=dtype)


def prediction_field(predictions: Any, *names: str) -> Any:
    for name in names:
        if hasattr(predictions, name):
            value = getattr(predictions, name)
            if value is not None:
                return value
        data = getattr(predictions, "data", None)
        if isinstance(data, dict) and name in data:
            value = data[name]
            if value is not None:
                return value
    return None


def extract_rfdetr_predictions(
    predictions: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract one local point per RF-DETR detection.

    RF-DETR Keypoint Preview returns a supervision ``KeyPoints`` object with
    ``xy`` shaped ``(detections, keypoints, 2)``. The training setup has one
    keypoint per class, so the first keypoint is the point to draw.
    """

    if isinstance(predictions, (list, tuple)):
        if not predictions:
            return empty_predictions()
        predictions = predictions[0]

    xy = as_numpy(prediction_field(predictions, "xy", "keypoints"))
    if xy is None or xy.size == 0:
        return empty_predictions()
    if xy.ndim == 2 and xy.shape[-1] == 2:
        xy = xy[:, None, :]
    if xy.ndim != 3 or xy.shape[-1] != 2:
        raise ValueError(f"Unexpected RF-DETR keypoint shape: {xy.shape}")

    count = xy.shape[0]
    points = xy[:, 0, :]

    class_ids = as_numpy(prediction_field(predictions, "class_id", "class_ids"), np.int32)
    if class_ids is None:
        class_ids = np.zeros(count, dtype=np.int32)
    class_ids = class_ids.reshape(-1)[:count]

    # RF-DETR/supervision can return the resolved class name in addition to
    # the numeric ID. Prefer that name because some checkpoint generations
    # use different numeric ID conventions (for example, a background slot).
    raw_class_names = prediction_field(predictions, "class_name", "class_names")
    if raw_class_names is None:
        predicted_class_names = np.full(count, "", dtype=object)
    else:
        predicted_class_names = np.asarray(raw_class_names, dtype=object).reshape(-1)[:count]

    detection_confidences = as_numpy(
        prediction_field(predictions, "detection_confidence", "confidence", "conf"),
    )
    if detection_confidences is None:
        # Do not silently draw unscored detections. RF-DETR KeyPoints objects
        # provide detection_confidence; a missing field should fail closed so
        # --confidence remains meaningful.
        detection_confidences = np.zeros(count, dtype=np.float32)
    detection_confidences = detection_confidences.reshape(-1)[:count]

    keypoint_confidences = as_numpy(
        prediction_field(predictions, "keypoint_confidence", "keypoint_confidences"),
    )
    if keypoint_confidences is None:
        keypoint_confidences = np.ones(count, dtype=np.float32)
    if keypoint_confidences.ndim == 2:
        keypoint_confidences = keypoint_confidences[:, 0]
    keypoint_confidences = keypoint_confidences.reshape(-1)[:count]

    return points, class_ids, predicted_class_names, detection_confidences, keypoint_confidences


def extract_yolo_predictions(
    predictions: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract one local point per detection from an Ultralytics YOLO pose result."""

    if isinstance(predictions, (list, tuple)):
        if not predictions:
            return empty_predictions()
        predictions = predictions[0]

    keypoints = getattr(predictions, "keypoints", None)
    xy = as_numpy(getattr(keypoints, "xy", None))
    if xy is None or xy.size == 0:
        return empty_predictions()
    if xy.ndim == 2 and xy.shape[-1] == 2:
        xy = xy[:, None, :]
    if xy.ndim != 3 or xy.shape[-1] != 2:
        raise ValueError(f"Unexpected YOLO keypoint shape: {xy.shape}")

    count = xy.shape[0]
    points = xy[:, 0, :]
    boxes = getattr(predictions, "boxes", None)
    class_ids = as_numpy(getattr(boxes, "cls", None), np.int32)
    if class_ids is None:
        class_ids = np.zeros(count, dtype=np.int32)
    class_ids = class_ids.reshape(-1)[:count]

    box_confidences = as_numpy(getattr(boxes, "conf", None))
    if box_confidences is None:
        box_confidences = np.zeros(count, dtype=np.float32)
    box_confidences = box_confidences.reshape(-1)[:count]

    keypoint_confidences = as_numpy(getattr(keypoints, "conf", None))
    if keypoint_confidences is None:
        keypoint_confidences = np.ones(count, dtype=np.float32)
    if keypoint_confidences.ndim == 2:
        keypoint_confidences = keypoint_confidences[:, 0]
    keypoint_confidences = keypoint_confidences.reshape(-1)[:count]

    names = getattr(predictions, "names", {})
    if isinstance(names, (list, tuple)):
        predicted_class_names = np.asarray(
            [names[int(class_id)] if 0 <= int(class_id) < len(names) else "" for class_id in class_ids],
            dtype=object,
        )
    elif isinstance(names, dict):
        predicted_class_names = np.asarray(
            [names.get(int(class_id), "") for class_id in class_ids],
            dtype=object,
        )
    else:
        predicted_class_names = np.full(count, "", dtype=object)

    return points, class_ids, predicted_class_names, box_confidences, keypoint_confidences


def extract_predictions(
    predictions: Any,
    model_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if model_type == "yolo":
        return extract_yolo_predictions(predictions)
    return extract_rfdetr_predictions(predictions)


def empty_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.empty((0, 2), dtype=np.float32),
        np.empty((0,), dtype=np.int32),
        np.empty((0,), dtype=object),
        np.empty((0,), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
    )


def draw_label(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    height, width = image.shape[:2]
    scale = max(0.35, min(width, height) / 720.0)
    thickness = max(1, round(scale * 2.0))
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_width, text_height = cv2.getTextSize(text, font, scale, thickness)[0]
    text_x = x + 6
    if text_x + text_width >= width:
        text_x = x - text_width - 6
    text_y = y - 6
    if text_y - text_height < 0:
        text_y = y + text_height + 6
    text_x = max(0, min(text_x, width - text_width))
    text_y = max(text_height, min(text_y, height - 1))
    cv2.putText(
        image,
        text,
        (text_x, text_y),
        font,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(image, text, (text_x, text_y), font, scale, color, thickness, cv2.LINE_AA)


def draw_sticker_letter(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    """Draw a centered, colored sticker class letter with a black border."""

    height, width = image.shape[:2]
    scale = min(1.25, max(0.55, min(width, height) / 600.0))
    thickness = max(1, round(scale * 2.0))
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    origin = (
        x - text_width // 2,
        y + (text_height - baseline) // 2,
    )
    cv2.putText(
        image,
        text,
        origin,
        font,
        scale,
        (0, 0, 0),
        thickness + 3,
        cv2.LINE_AA,
    )
    cv2.putText(image, text, origin, font, scale, color, thickness, cv2.LINE_AA)


def render_frame(
    frame: np.ndarray,
    bbox_detector: YOLOCubeDetector,
    pose_model: Any,
    roi_tracker: RoiTracker,
    args: argparse.Namespace,
) -> tuple[np.ndarray, int, str]:
    roi_xyxy, _, roi_source = roi_for_frame(
        frame,
        bbox_detector,
        roi_tracker,
        args.roi_padding,
    )
    output = frame.copy()
    if roi_xyxy is None:
        cv2.putText(
            output,
            "cube not detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return output, 0, roi_source

    x1, y1, x2, y2 = roi_xyxy
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return output, 0, roi_source
    if args.model_type == "yolo":
        # Ultralytics accepts OpenCV's native BGR ndarray input and performs
        # its usual preprocessing internally.
        predict_kwargs: dict[str, Any] = {
            "conf": args.pose_conf,
            "imgsz": args.resolution,
            "verbose": False,
        }
        if args.device is not None:
            predict_kwargs["device"] = args.device
        predictions = pose_model.predict(roi, **predict_kwargs)
    else:
        # RF-DETR's documented inference path expects RGB images. This is
        # especially important for the sticker-color classes; geometry-only
        # landmarks can look reasonable when channels are reversed.
        model_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        try:
            predictions = pose_model.predict(model_roi, threshold=args.pose_conf)
        except TypeError:
            # Older RF-DETR builds accept the image positionally but not the
            # threshold keyword; filter the returned object below instead.
            predictions = pose_model.predict(model_roi)

    (
        local_points,
        class_ids,
        predicted_class_names,
        box_confidences,
        keypoint_confidences,
    ) = extract_predictions(predictions, args.model_type)
    rendered = 0
    tracking_points: list[tuple[float, float]] = []
    class_index_by_name = {name: index for index, name in enumerate(CLASS_NAMES)}
    for point, class_id, predicted_class_name, box_confidence, keypoint_confidence in zip(
        local_points,
        class_ids,
        predicted_class_names,
        box_confidences,
        keypoint_confidences,
    ):
        class_name = str(predicted_class_name).strip().lower()
        class_index = class_index_by_name.get(class_name)
        if class_index is None:
            class_index = int(class_id)
        if not 0 <= class_index < len(CLASS_NAMES):
            continue
        if float(box_confidence) < args.pose_conf:
            continue
        if float(keypoint_confidence) < args.keypoint_conf:
            continue
        if not np.isfinite(point).all() or not (
            0 <= float(point[0]) < roi.shape[1] and 0 <= float(point[1]) < roi.shape[0]
        ):
            continue

        tracking_points.append((float(point[0]), float(point[1])))

        point_x = int(round(float(point[0]) + x1))
        point_y = int(round(float(point[1]) + y1))
        if not (0 <= point_x < output.shape[1] and 0 <= point_y < output.shape[0]):
            continue

        color = CLASS_COLORS_BGR[class_index]
        if class_index < LANDMARK_CLASS_COUNT:
            radius = max(2, round(min(output.shape[:2]) / 240.0))
            outline = np.asarray(
                [
                    (point_x, point_y - radius - 2),
                    (point_x + radius + 2, point_y),
                    (point_x, point_y + radius + 2),
                    (point_x - radius - 2, point_y),
                ],
                dtype=np.int32,
            )
            diamond = np.asarray(
                [
                    (point_x, point_y - radius),
                    (point_x + radius, point_y),
                    (point_x, point_y + radius),
                    (point_x - radius, point_y),
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(output, [outline], (0, 0, 0), lineType=cv2.LINE_AA)
            cv2.fillPoly(output, [diamond], color, lineType=cv2.LINE_AA)
        else:
            draw_sticker_letter(output, CLASS_LABELS[class_index], point_x, point_y, color)
        if args.show_labels:
            draw_label(
                output,
                f"{CLASS_LABELS[class_index]} {float(box_confidence):.2f}",
                point_x,
                point_y,
                color,
            )
        rendered += 1

    if tracking_points:
        tracked = np.asarray(tracking_points, dtype=np.float32)
        roi_tracker.observe_tracking_bounds(
            (
                int(np.floor(tracked[:, 0].min())),
                int(np.floor(tracked[:, 1].min())),
                int(np.ceil(tracked[:, 0].max())),
                int(np.ceil(tracked[:, 1].max())),
            ),
            output.shape[1],
            output.shape[0],
            tracking_gain=args.roi_tracking_gain,
            tracking_padding=args.roi_tracking_padding,
        )
    else:
        # Match the live ROI flow: a pose miss causes the bbox detector to run
        # again on the next frame instead of holding a stale crop forever.
        roi_tracker.reset()

    cv2.rectangle(output, (x1, y1), (x2 - 1, y2 - 1), (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(
        output,
        f"ROI: {roi_source} | points: {rendered}",
        (12, output.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"ROI: {roi_source} | points: {rendered}",
        (12, output.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    return output, rendered, roi_source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Image/video path or camera index.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="RF-DETR .pth or YOLO .pt checkpoint, or a training directory.",
    )
    parser.add_argument(
        "--model-type",
        choices=("auto", "rfdetr", "yolo"),
        default="auto",
        help="Pose backend; auto infers .pth as RF-DETR and .pt as YOLO.",
    )
    parser.add_argument(
        "--bbox-model",
        type=Path,
        default=DEFAULT_BBOX_MODEL,
        help="YOLO cube bounding-box checkpoint.",
    )
    parser.add_argument("--bbox-conf", type=float, default=0.25)
    parser.add_argument("--bbox-imgsz", type=int, default=640)
    parser.add_argument("--roi-padding", type=float, default=0.08)
    parser.add_argument("--roi-tracking-gain", type=float, default=1.4)
    parser.add_argument("--roi-tracking-padding", type=float, default=0.18)
    parser.add_argument(
        "--resolution",
        type=int,
        default=288,
        help="Pose inference resolution; use the resolution used for training.",
    )
    parser.add_argument(
        "--pose-conf",
        "--confidence",
        "--conf",
        dest="pose_conf",
        type=float,
        default=0.25,
        help="Pose detection confidence threshold (default: 0.25).",
    )
    parser.add_argument("--keypoint-conf", type=float, default=0.05)
    parser.add_argument("--device", default=None, help="RF-DETR/YOLO device, e.g. cuda:0 or cpu.")
    parser.add_argument("--save", type=Path, help="Output image or video path.")
    parser.add_argument("--show-labels", action="store_true")
    parser.add_argument("--display", action="store_true", help="Show an OpenCV preview window.")
    parser.add_argument("--no-display", dest="display", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N video frames; 0 means all.")
    # Match the interactive behavior of the other renderers: display unless
    # the caller explicitly supplies --no-display.
    parser.set_defaults(display=True)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for name in ("bbox_conf", "pose_conf", "keypoint_conf"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.roi_padding < 0:
        raise SystemExit("--roi-padding must be nonnegative")
    if args.roi_tracking_gain < 0 or args.roi_tracking_padding < 0:
        raise SystemExit("--roi-tracking-gain and --roi-tracking-padding must be nonnegative")
    if args.resolution <= 0 or args.bbox_imgsz <= 0:
        raise SystemExit("--resolution and --bbox-imgsz must be positive")
    if args.max_frames < 0:
        raise SystemExit("--max-frames must be nonnegative")


def run_image(
    source: Path,
    args: argparse.Namespace,
    bbox_detector: YOLOCubeDetector,
    pose_model: Any,
) -> int:
    frame = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit(f"Could not read image: {source}")
    output, point_count, roi_source = render_frame(
        frame, bbox_detector, pose_model, RoiTracker(), args
    )
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.save), output):
            raise SystemExit(f"Could not write output image: {args.save}")
        print(f"Saved {args.save} ({point_count} points, ROI from {roi_source})")
    else:
        print(f"Rendered {point_count} points (ROI from {roi_source})")
    if args.display:
        cv2.imshow("Pose keypoints", output)
        cv2.waitKey(0)
    return point_count


def run_stream(
    source: int | Path,
    args: argparse.Namespace,
    bbox_detector: YOLOCubeDetector,
    pose_model: Any,
) -> int:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {source}")

    writer = None
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if np.isfinite(fps) and fps > 0 else 30.0
        writer = cv2.VideoWriter(
            str(args.save),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise SystemExit(f"Could not open output video: {args.save}")

    roi_tracker = RoiTracker()
    total_points = 0
    frame_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            output, point_count, _ = render_frame(frame, bbox_detector, pose_model, roi_tracker, args)
            total_points += point_count
            frame_count += 1
            if writer is not None:
                writer.write(output)
            if args.display:
                cv2.imshow("Pose keypoints", output)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
            if args.max_frames and frame_count >= args.max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()
    if args.save:
        print(f"Saved {args.save} ({frame_count} frames, {total_points} rendered points)")
    return total_points


def load_pose_model(checkpoint: Path, args: argparse.Namespace) -> tuple[Any, str]:
    """Load RF-DETR or Ultralytics YOLO pose weights."""

    model_type = infer_model_type(checkpoint, args.model_type)
    if model_type == "rfdetr":
        try:
            from rfdetr import RFDETRKeypointPreview
        except ImportError as exc:
            raise SystemExit(
                'RF-DETR inference requires: python -m pip install "rfdetr[train]"'
            ) from exc

        model_kwargs: dict[str, Any] = {
            "pretrain_weights": str(checkpoint),
            "resolution": args.resolution,
        }
        if args.device is not None:
            model_kwargs["device"] = args.device
        return RFDETRKeypointPreview(**model_kwargs), model_type

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "YOLO inference requires: python -m pip install ultralytics"
        ) from exc

    model = YOLO(str(checkpoint))
    task = getattr(model, "task", None)
    if task is not None and task != "pose":
        raise SystemExit(
            f"The YOLO checkpoint is a {task!r} model, not a pose model: {checkpoint}"
        )
    return model, model_type


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    checkpoint = resolve_checkpoint(args.model)
    if not args.bbox_model.is_file():
        raise SystemExit(f"YOLO bbox model not found: {args.bbox_model}")

    pose_model, args.model_type = load_pose_model(checkpoint, args)
    bbox_detector = YOLOCubeDetector(
        model_name=str(args.bbox_model),
        confidence_threshold=args.bbox_conf,
        image_size=args.bbox_imgsz,
        device=args.device,
    )
    print(f"Pose model type: {args.model_type}")
    print(f"Pose checkpoint: {checkpoint}")
    print(f"BBox checkpoint: {args.bbox_model}")

    source = parse_source(args.source)
    if isinstance(source, Path) and source.suffix.lower() in IMAGE_EXTENSIONS:
        result = run_image(source, args, bbox_detector, pose_model)
    else:
        result = run_stream(source, args, bbox_detector, pose_model)
    if args.display:
        cv2.destroyAllWindows()
    return 0 if result >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
