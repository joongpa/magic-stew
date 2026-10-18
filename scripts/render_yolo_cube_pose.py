"""Render predictions from a trained YOLO cube-pose head.

The renderer overlays the articulated cube wireframe predicted by the custom
pose head and prints the decoded values for each displayed frame.

Examples:

    .venv/bin/python scripts/render_yolo_cube_pose.py \
        --model runs/cube_pose_head_accel_fixed_v2/best.pt \
        --source data/mcp_dataset_v5.3/images/000001.png

    .venv/bin/python scripts/render_yolo_cube_pose.py \
        --model runs/cube_pose_head_accel_fixed_v2/best.pt \
        --source data/mcp_dataset_v5.3/images/000001.png \
        --save runs/cube_pose_render_one.png

Press ``q``, Escape, or Enter to close the OpenCV window. ``--source`` may be
an image, video, or camera index.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from live_recognition.cube_pose import (  # noqa: E402
    YOLOCubePoseModel,
    _project_articulated_lines,
)
from live_recognition.detectors import YOLOCubeDetector  # noqa: E402
from live_recognition.roi_tracking import RoiTracker  # noqa: E402
from live_recognition.roi_tracking import roi_for_frame as shared_roi_for_frame  # noqa: E402
from scripts.render_yolo_sticker_segmentation import (  # noqa: E402
    has_segmentation as has_segmentation_result,
    predict as predict_segmentation,
    segmentation_bounds as combined_segmentation_bounds,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
WINDOW_NAME = "cube pose"
LAYER_COLORS_BGR = (
    (255, 160, 40),
    (50, 220, 80),
    (40, 120, 255),
)
DEFAULT_BBOX_MODEL = ROOT / "runs/500ExamplesWeights.pt"


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def parse_source(source: str) -> int | str:
    source = source.strip().strip("\"'").strip()
    return int(source) if source.isdigit() else source


def is_image_source(source: int | str) -> bool:
    return isinstance(source, str) and Path(source).suffix.lower() in IMAGE_EXTENSIONS


def image_tensor(image_bgr: np.ndarray, image_size: int, device: torch.device) -> torch.Tensor:
    resized = cv2.resize(
        image_bgr,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA,
    )
    image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return (
        torch.from_numpy(image_rgb.copy())
        .permute(2, 0, 1)
        .float()
        .div(255.0)
        .unsqueeze(0)
        .to(device)
    )


def normalized_to_pixels(points: np.ndarray, image_size: int) -> np.ndarray:
    return points * np.asarray((image_size, image_size), dtype=np.float32)


def draw_pose(
    image: np.ndarray,
    lines: np.ndarray,
    prediction: dict[str, float | list[float]],
) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]

    pixel_lines = lines * np.asarray((width, height), dtype=np.float32)
    for line_index, line in enumerate(pixel_lines):
        points = np.round(line).astype(np.int32)
        layer_index = min(line_index // 12, len(LAYER_COLORS_BGR) - 1)
        color = LAYER_COLORS_BGR[layer_index]
        for start, end in zip(points[:-1], points[1:]):
            cv2.line(
                output,
                tuple(start),
                tuple(end),
                color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

    center = np.asarray(prediction["image_center"], dtype=np.float32)
    center_pixels = np.round(center * np.asarray((width, height))).astype(np.int32)
    cv2.drawMarker(
        output,
        tuple(center_pixels),
        (0, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=12,
        thickness=2,
    )

    return output


def draw_prediction_text(
    image: np.ndarray,
    prediction: dict[str, float | list[float]],
) -> np.ndarray:
    """Draw pose values in the full-frame window corner."""
    output = image.copy()
    center = np.asarray(prediction["image_center"], dtype=np.float32)
    layer_angles = prediction["layer_angles"]
    text_lines = (
        f"u={float(prediction['u']):.1f} v={float(prediction['v']):.1f}",
        f"twist={float(prediction['twist']):.1f} "
        f"layers=({layer_angles[0]:.1f},{layer_angles[1]:.1f})",
        f"center=({center[0]:.3f},{center[1]:.3f}) "
        f"scale={float(prediction['image_scale']):.3f}",
        f"perspective={float(prediction['perspective_strength']):.3f}",
    )
    for line_index, text in enumerate(text_lines):
        text_position = (8, 22 + line_index * 20)
        cv2.putText(
            output,
            text,
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            text,
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
    return output


def decode_prediction(
    prediction: dict[str, torch.Tensor],
) -> tuple[dict[str, float | list[float]], np.ndarray]:
    decoded: dict[str, float | list[float]] = {
        "u": float(prediction["u"][0].detach().cpu()),
        "v": float(prediction["v"][0].detach().cpu()),
        "twist": float(prediction["twist"][0].detach().cpu()),
        "image_center": prediction["image_center"][0].detach().cpu().tolist(),
        "image_scale": float(prediction["image_scale"][0].detach().cpu()),
        "perspective_strength": float(
            prediction["perspective_strength"][0].detach().cpu()
        ),
        "layer_angles": prediction["layer_angles"][0].detach().cpu().tolist(),
    }
    lines = _project_articulated_lines(prediction)[0].detach().cpu().numpy()
    return decoded, lines


def render_frame(
    model: YOLOCubePoseModel,
    image: np.ndarray,
    image_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float | list[float]]]:
    with torch.no_grad():
        prediction = model(image_tensor(image, image_size, device))
    decoded, lines = decode_prediction(prediction)
    return draw_pose(image, lines, decoded), decoded


def create_video_writer(path: Path, frame: np.ndarray, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = "mp4v" if path.suffix.lower() in {".mp4", ".m4v", ".mov"} else "MJPG"
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


def predict_and_render(
    pose_model: YOLOCubePoseModel,
    segmentation_model: object,
    bbox_detector: YOLOCubeDetector | None,
    roi_tracker: RoiTracker,
    image: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    image_size: int,
) -> tuple[np.ndarray, YOLOCubeDetector | None, dict[str, float | list[float]] | None]:
    """Run bbox -> ROI -> segmentation -> pose, matching the normal render flow."""
    height, width = image.shape[:2]
    if bbox_detector is None:
        bbox_detector = YOLOCubeDetector(
            model_name=str(args.bbox_model),
            confidence_threshold=args.bbox_conf,
            image_size=args.bbox_imgsz,
        )

    roi_xyxy, _candidate, roi_source = shared_roi_for_frame(
        image,
        bbox_detector,
        roi_tracker,
        args.roi_padding,
    )
    if roi_xyxy is None:
        return image.copy(), bbox_detector, None

    x1, y1, x2, y2 = roi_xyxy
    roi = image[y1:y2, x1:x2]
    segmentation_result = predict_segmentation(segmentation_model, roi, args)
    if segmentation_result is None or not has_segmentation_result(segmentation_result):
        roi_tracker.reset()
        output = image.copy()
        if args.show_bbox:
            cv2.rectangle(output, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), 2)
        return output, bbox_detector, None

    # The pose model has the same combined segmentation backbone/head, so the
    # segmentation check above and the pose forward pass use the same ROI.
    rendered_roi, prediction = render_frame(pose_model, roi, image_size, device)
    roi_tracker.observe_tracking_bounds(
        combined_segmentation_bounds(segmentation_result, roi.shape[:2], "combined"),
        width,
        height,
        tracking_gain=args.roi_tracking_gain,
        tracking_padding=args.roi_tracking_padding,
    )
    output = image.copy()
    output[y1:y2, x1:x2] = rendered_roi
    if args.show_bbox:
        roi_color = (0, 0, 255) if roi_source == "detector" else (40, 220, 40)
        cv2.rectangle(output, (x1, y1), (x2 - 1, y2 - 1), roi_color, 2)
    output = draw_prediction_text(output, prediction)
    return output, bbox_detector, prediction


def main() -> int:
    parser = argparse.ArgumentParser(description="Display YOLO cube-pose predictions")
    parser.add_argument(
        "--model",
        "--checkpoint",
        dest="model",
        type=Path,
        required=True,
        help="Trained cube-pose head checkpoint",
    )
    parser.add_argument("--source", required=True, help="Image, video path, or camera index")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bbox-model", type=Path, default=DEFAULT_BBOX_MODEL)
    parser.add_argument("--bbox-conf", type=float, default=0.25)
    parser.add_argument("--bbox-imgsz", type=int, default=640)
    parser.add_argument("--roi-padding", type=float, default=0.08)
    parser.add_argument("--roi-tracking-gain", type=float, default=1.4)
    parser.add_argument("--roi-tracking-padding", type=float, default=0.18)
    parser.add_argument("--show-bbox", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--conf", type=float, default=0.25, help="Segmentation confidence threshold")
    parser.add_argument("--imgsz", type=int, default=256, help="Segmentation/pose inference size")
    parser.add_argument("--once", action="store_true", help="Process one video/camera frame and exit")
    parser.add_argument("--save", type=Path, help="Output image path for image or --once mode")
    parser.add_argument("--save-video", type=Path, help="Output path for rendered video frames")
    args = parser.parse_args()

    device = resolve_device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        print("MPS is unavailable; falling back to CPU")
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU")
        device = torch.device("cpu")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    base_model_path = checkpoint.get("base_model")
    if base_model_path is not None:
        base_model_path = Path(base_model_path)
        if not base_model_path.is_absolute() and not base_model_path.exists():
            base_model_path = ROOT / base_model_path
    model = YOLOCubePoseModel.from_pose_checkpoint(
        args.model,
        base_model_path=base_model_path,
        map_location="cpu",
    ).to(device)
    model.eval()
    image_size = int(args.imgsz or checkpoint.get("imgsz", 256))
    from ultralytics import YOLO

    if not args.bbox_model.is_file():
        raise SystemExit(f"Could not find bbox model: {args.bbox_model}")
    segmentation_model = YOLO(str(base_model_path))
    bbox_detector = None
    roi_tracker = RoiTracker()
    source = parse_source(args.source)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    if is_image_source(source):
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"Could not read image: {source}")
        rendered, bbox_detector, prediction = predict_and_render(
            model,
            segmentation_model,
            bbox_detector,
            roi_tracker,
            image,
            args,
            device,
            image_size,
        )
        if args.save_video:
            raise SystemExit("--save-video requires a video file or camera source")
        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.save), rendered):
                raise SystemExit(f"Could not write rendered image: {args.save}")
            print(f"Saved render: {args.save}")
        print(json.dumps(prediction, indent=2))
        while True:
            cv2.imshow(WINDOW_NAME, rendered)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27, 13):
                break
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
            rendered, bbox_detector, prediction = predict_and_render(
                model,
                segmentation_model,
                bbox_detector,
                roi_tracker,
                frame,
                args,
                device,
                image_size,
            )
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
                    raise SystemExit(f"Could not write rendered image: {args.save}")
            cv2.imshow(WINDOW_NAME, rendered)
            key = cv2.waitKey(30 if args.once else 1) & 0xFF
            if args.once or key in (ord("q"), 27, 13):
                if args.once:
                    print(json.dumps(prediction, indent=2))
                break
    finally:
        if writer is not None:
            writer.release()
        capture.release()
        cv2.destroyAllWindows()
    if writer is not None:
        print(f"Saved video: {args.save_video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
