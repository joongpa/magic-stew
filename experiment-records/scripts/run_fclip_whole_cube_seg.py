#!/usr/bin/env python3
"""Run a line detector and whole-cube segmentation together on an image or video.

The segmentation model supplies both the displayed whole-cube mask and the
ROI tracker. The selected line detector is evaluated on the tracked square ROI,
and only lines with more than half of their sampled pixels inside a padded copy
of the segmentation mask are drawn.
The padded mask is never displayed and is never used for ROI tracking.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from live_recognition.detectors import YOLOCubeDetector, default_yolo_device  # noqa: E402
from live_recognition.line_detectors import LineDetector, build_line_detector  # noqa: E402
from live_recognition.roi_tracking import RoiTracker, square_roi_from_bbox  # noqa: E402


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_MLSD_MODEL = ROOT / ".models/M-LSD_512_large_fp32.tflite"
DEFAULT_MLSD_REPO = ROOT / ".mlsd_pytorch"
DEFAULT_FCLIP_REPO = ROOT / "F-Clip"
DEFAULT_FCLIP_CONFIG = (
    ROOT / "runs/fclip_cube/261012-174513-cube_mcp_HG1_D2_256/config.yaml"
)
DEFAULT_FCLIP_CHECKPOINT = (
    ROOT / "runs/fclip_cube/261012-174513-cube_mcp_HG1_D2_256/checkpoint_best.pth.tar"
)
DEFAULT_SEGMENTATION_MODEL = (
    ROOT
    / "runs/segment/runs/cube_whole_seg_local/runs/cube_whole_yolo11n_seg-4/weights/best.pt"
)
DEFAULT_BBOX_MODEL = ROOT / "runs/500ExamplesWeights.pt"
WINDOW_NAME = "F-Clip/M-LSD + whole-cube segmentation"


def parse_source(source: str) -> int | str:
    source = source.strip().strip("\"'").strip()
    return int(source) if source.isdigit() else source


def is_image_source(source: int | str) -> bool:
    return isinstance(source, str) and Path(source).suffix.lower() in IMAGE_EXTENSIONS


def tensor_to_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def predict_segmentation(
    model: object,
    image: np.ndarray,
    confidence: float,
    image_size: int,
    device: str | None = None,
):
    predict_kwargs = {} if device in (None, "auto") else {"device": device}
    predictions = model.predict(
        source=image,
        conf=float(confidence),
        imgsz=int(image_size),
        retina_masks=False,
        verbose=False,
        **predict_kwargs,
    )
    return predictions[0] if predictions else None


def largest_segmentation_mask(
    result: object | None,
    output_shape: tuple[int, int],
) -> tuple[np.ndarray | None, float]:
    """Return the largest raw predicted mask at the input image resolution."""

    if result is None:
        return None, 0.0
    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if masks is None or boxes is None:
        return None, 0.0

    mask_data = tensor_to_numpy(masks.data)
    confidences = tensor_to_numpy(boxes.conf).reshape(-1)
    if mask_data.ndim < 3 or len(mask_data) == 0:
        return None, 0.0

    output_height, output_width = output_shape
    candidates: list[tuple[int, np.ndarray, float]] = []
    for index, mask in enumerate(mask_data[: len(confidences)]):
        resized = cv2.resize(
            mask.astype(np.float32),
            (output_width, output_height),
            interpolation=cv2.INTER_NEAREST,
        )
        binary = resized > 0.5
        area = int(binary.sum())
        if area:
            candidates.append((area, binary, float(confidences[index])))
    if not candidates:
        return None, 0.0
    _area, selected, confidence = max(candidates, key=lambda item: item[0])
    return selected, confidence


def mask_bounds(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def padded_filter_mask(mask: np.ndarray, padding_px: int) -> np.ndarray:
    if padding_px <= 0 or not bool(mask.any()):
        return mask
    kernel_size = max(3, int(padding_px) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def lines_inside_mask(lines_xyxy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keep a line only when more than half its sampled points are in the mask."""

    if len(lines_xyxy) == 0 or not bool(mask.any()):
        return np.zeros(len(lines_xyxy), dtype=bool)
    height, width = mask.shape
    keep = np.zeros(len(lines_xyxy), dtype=bool)
    for index, line in enumerate(lines_xyxy):
        x1, y1, x2, y2 = line
        length = float(np.hypot(x2 - x1, y2 - y1))
        sample_count = max(8, min(256, int(math.ceil(length)) + 1))
        xs = np.rint(np.linspace(x1, x2, sample_count)).astype(np.int32)
        ys = np.rint(np.linspace(y1, y2, sample_count)).astype(np.int32)
        valid = (ys >= 0) & (ys < height) & (xs >= 0) & (xs < width)
        if not np.any(valid):
            continue
        inside_fraction = float(np.mean(mask[ys[valid], xs[valid]]))
        keep[index] = inside_fraction > 0.5
    return keep


def crop_mask_to_roi(
    full_mask: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
) -> np.ndarray:
    x1, y1, x2, y2 = roi_xyxy
    return full_mask[y1:y2, x1:x2]


def draw_mask(output: np.ndarray, mask: np.ndarray, opacity: float) -> None:
    """Render exactly the unpadded binary segmentation mask."""

    if not bool(mask.any()):
        return
    color = np.zeros_like(output)
    color[:, :] = (255, 0, 255)
    output[mask] = cv2.addWeighted(output, 1.0 - opacity, color, opacity, 0.0)[mask]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (255, 0, 255), 2, cv2.LINE_AA)


def draw_lines(
    output: np.ndarray,
    lines_xyxy: np.ndarray,
    scores: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
    *,
    show_scores: bool,
    endpoint_radius: int,
) -> None:
    x_offset, y_offset = roi_xyxy[:2]
    for line, score in zip(lines_xyxy, scores):
        x1, y1, x2, y2 = np.rint(line).astype(int)
        start = (int(round(x1 + x_offset)), int(round(y1 + y_offset)))
        end = (int(round(x2 + x_offset)), int(round(y2 + y_offset)))
        cv2.line(output, start, end, (30, 220, 60), 2, cv2.LINE_AA)
        cv2.circle(output, start, endpoint_radius, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(output, end, endpoint_radius, (255, 255, 255), -1, cv2.LINE_AA)
        if show_scores:
            cv2.putText(
                output,
                f"{float(score):.2f}",
                (start[0] + 3, start[1] - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )


def process_frame(
    frame: np.ndarray,
    bbox_detector: YOLOCubeDetector | None,
    segmentation_model: object,
    line_detector: LineDetector,
    tracker: RoiTracker,
    args: argparse.Namespace,
) -> np.ndarray:
    height, width = frame.shape[:2]
    if tracker.roi_xyxy is None:
        candidate = bbox_detector.detect(frame) if bbox_detector is not None else None
        if candidate is not None:
            roi_xyxy = tracker.initialize(candidate, width, height, args.roi_padding)
            roi_source = "bbox-seeded"
        else:
            # Keep a full-frame fallback for sources where the bootstrap bbox
            # model misses; the segmentation mask still owns all tracking.
            roi_xyxy = (0, 0, width, height)
            roi_source = "full-frame-fallback"
    else:
        roi_xyxy = tracker.roi_xyxy
        roi_source = "seg-tracked"
    line_roi_xyxy = roi_xyxy

    x1, y1, x2, y2 = roi_xyxy
    roi = frame[y1:y2, x1:x2]
    seg_result = predict_segmentation(
        segmentation_model,
        roi,
        args.seg_conf,
        args.seg_imgsz,
        args.device,
    )
    local_mask, seg_confidence = largest_segmentation_mask(seg_result, roi.shape[:2])

    output = frame.copy()
    full_mask = np.zeros((height, width), dtype=bool)
    if local_mask is not None:
        full_mask[y1:y2, x1:x2] = local_mask
        draw_mask(output, full_mask, args.mask_opacity)

        local_bounds = mask_bounds(local_mask)
        if local_bounds is not None:
            if roi_source == "full-frame-fallback":
                global_bounds = (
                    x1 + local_bounds[0],
                    y1 + local_bounds[1],
                    x1 + local_bounds[2],
                    y1 + local_bounds[3],
                )
                tracker.roi_xyxy = square_roi_from_bbox(
                    global_bounds,
                    width,
                    height,
                    args.roi_padding,
                )
                tracker.base_side = min(
                    tracker.roi_xyxy[2] - tracker.roi_xyxy[0],
                    tracker.roi_xyxy[3] - tracker.roi_xyxy[1],
                )
                roi_source = "seg-initialized"
                line_roi_xyxy = tracker.roi_xyxy
            else:
                tracker.observe_tracking_bounds(
                    local_bounds,
                    width,
                    height,
                    tracking_gain=args.roi_tracking_gain,
                    tracking_padding=args.roi_tracking_padding,
                )
                roi_source = "mask-tracked"
                # Use the enlarged, updated ROI for line inference on this
                # frame as well as for subsequent frames.
                line_roi_xyxy = tracker.roi_xyxy
    else:
        tracker.reset()

    # Run the selected line detector on the same ROI and filter only against a padded copy of the
    # raw mask. No line geometry participates in ROI tracking.
    if local_mask is not None:
        lx1, ly1, lx2, ly2 = line_roi_xyxy
        line_roi = frame[ly1:ly2, lx1:lx2]
        line_mask = crop_mask_to_roi(full_mask, line_roi_xyxy)
        detections = line_detector.detect(line_roi)
        lines = detections.lines_xyxy
        scores = detections.scores
        filter_mask = padded_filter_mask(line_mask, args.line_mask_padding)
        keep = lines_inside_mask(lines, filter_mask)
        lines, scores = lines[keep], scores[keep]
        draw_lines(
            output,
            lines,
            scores,
            line_roi_xyxy,
            show_scores=args.show_line_scores,
            endpoint_radius=args.endpoint_radius,
        )
        line_count = len(lines)
    else:
        line_count = 0

    if args.show_roi and (tracker.roi_xyxy is not None or local_mask is not None):
        active_roi = tracker.roi_xyxy or roi_xyxy
        ax1, ay1, ax2, ay2 = active_roi
        roi_color = (40, 220, 40) if local_mask is not None else (0, 0, 255)
        cv2.rectangle(output, (ax1, ay1), (ax2 - 1, ay2 - 1), roi_color, 2)

    label = (
        f"seg {seg_confidence:.2f} lines {line_count} ({roi_source})"
        if local_mask is not None
        else f"NO SEGMENTATION MASK — lines skipped ({roi_source})"
    )
    cv2.putText(output, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(output, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 1, cv2.LINE_AA)
    return output


def create_video_writer(path: Path, frame: np.ndarray, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    fourcc = "mp4v" if suffix in {".mp4", ".m4v", ".mov"} else "MJPG"
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*fourcc),
        fps if np.isfinite(fps) and fps > 0.0 else 30.0,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Image, video path, or camera index")
    parser.add_argument(
        "--line-detector",
        choices=("mlsd", "fclip"),
        default="mlsd",
        help="Second-stage line detector",
    )
    parser.add_argument("--mlsd-model", type=Path, default=DEFAULT_MLSD_MODEL)
    parser.add_argument("--mlsd-repo", type=Path, default=DEFAULT_MLSD_REPO)
    parser.add_argument("--mlsd-device", default="auto", help="PyTorch M-LSD device for .pth models")
    parser.add_argument("--fclip-repo", type=Path, default=DEFAULT_FCLIP_REPO)
    parser.add_argument("--fclip-config", type=Path, default=DEFAULT_FCLIP_CONFIG)
    parser.add_argument("--fclip-checkpoint", type=Path, default=DEFAULT_FCLIP_CHECKPOINT)
    parser.add_argument("--fclip-device", default="auto", help="F-Clip device, e.g. auto, cpu, or cuda")
    parser.add_argument(
        "--device",
        default=None,
        help="YOLO device, for example cpu, mps, cuda, or cuda:0; auto-selects MPS when available",
    )
    parser.add_argument("--seg-model", type=Path, default=DEFAULT_SEGMENTATION_MODEL)
    parser.add_argument("--bbox-model", type=Path, default=DEFAULT_BBOX_MODEL)
    parser.add_argument("--bbox-conf", type=float, default=0.25, help="Bootstrap bbox confidence")
    parser.add_argument("--bbox-imgsz", type=int, default=640)
    parser.add_argument("--no-bbox-seed", action="store_true", help="Skip bbox bootstrap and start segmentation full-frame")
    parser.add_argument("--seg-conf", type=float, default=0.25, help="Whole-cube segmentation confidence")
    parser.add_argument("--line-conf", type=float, default=0.10, help="Line score threshold")
    parser.add_argument(
        "--line-imgsz",
        type=int,
        default=None,
        help="Line detector input size; defaults to 512 for TFLite M-LSD, checkpoint size for PyTorch M-LSD, or F-Clip config size",
    )
    parser.add_argument("--line-min-length", type=float, default=5.0)
    parser.add_argument("--max-lines", type=int, default=500)
    parser.add_argument("--seg-imgsz", type=int, default=256)
    parser.add_argument("--line-mask-padding", type=int, default=8, help="Padding in pixels used only for line filtering")
    parser.add_argument(
        "--roi-padding",
        type=float,
        default=0.25,
        help="Padding on each side; 0.25 makes the square ROI 50%% larger than the bbox size",
    )
    parser.add_argument("--roi-tracking-gain", type=float, default=1.0)
    parser.add_argument(
        "--roi-tracking-padding",
        type=float,
        default=0.25,
        help="Padding on each side for newly larger segmentation bounds",
    )
    parser.add_argument("--mask-opacity", type=float, default=0.45)
    parser.add_argument("--show-line-scores", action="store_true")
    parser.add_argument("--endpoint-radius", type=int, default=1)
    parser.add_argument("--show-roi", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save", type=Path, help="Output image path, or one frame when --once is used")
    parser.add_argument("--save-video", type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--display", dest="display", action="store_true")
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.set_defaults(display=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.device in (None, "auto"):
        args.device = default_yolo_device()
    required_paths = [(args.seg_model, "segmentation model")]
    if args.line_detector == "mlsd":
        required_paths.append((args.mlsd_model, "M-LSD model"))
        if args.mlsd_model.suffix.lower() in {".pt", ".pth"} and not args.mlsd_repo.is_dir():
            raise SystemExit(f"Could not find PyTorch M-LSD repository: {args.mlsd_repo}")
    else:
        if not args.fclip_repo.is_dir():
            raise SystemExit(f"Could not find F-Clip repository: {args.fclip_repo}")
        required_paths.extend(
            [
                (args.fclip_config, "F-Clip config"),
                (args.fclip_checkpoint, "F-Clip checkpoint"),
            ]
        )
    for path, label in required_paths:
        if not path.is_file():
            raise SystemExit(f"Could not find {label}: {path}")
    if not args.no_bbox_seed and not args.bbox_model.is_file():
        raise SystemExit(f"Could not find bbox bootstrap model: {args.bbox_model}")
    if not 0.0 <= args.mask_opacity <= 1.0:
        raise SystemExit("--mask-opacity must be between 0 and 1")
    if args.line_mask_padding < 0:
        raise SystemExit("--line-mask-padding must be non-negative")
    if args.endpoint_radius < 0:
        raise SystemExit("--endpoint-radius must be non-negative")

    from ultralytics import YOLO

    bbox_detector = None
    if not args.no_bbox_seed:
        bbox_detector = YOLOCubeDetector(
            model_name=str(args.bbox_model),
            confidence_threshold=args.bbox_conf,
            image_size=args.bbox_imgsz,
            device=args.device,
        )
    if args.line_detector == "mlsd":
        line_detector = build_line_detector(
            "mlsd",
            model_path=args.mlsd_model,
            repo_path=args.mlsd_repo,
            input_size=args.line_imgsz,
            score_threshold=args.line_conf,
            min_length=args.line_min_length,
            max_lines=args.max_lines,
            device=args.mlsd_device,
        )
    else:
        line_detector = build_line_detector(
            "fclip",
            repo_path=args.fclip_repo,
            config_path=args.fclip_config,
            checkpoint_path=args.fclip_checkpoint,
            variant="HG1_D2",
            input_size=args.line_imgsz,
            score_threshold=args.line_conf,
            min_length=args.line_min_length,
            max_lines=args.max_lines,
            device=args.fclip_device,
        )
    segmentation_model = YOLO(str(args.seg_model))
    if args.line_detector == "mlsd":
        print(f"M-LSD model: {args.mlsd_model}")
        if args.mlsd_model.suffix.lower() in {".pt", ".pth"}:
            print(f"M-LSD repository: {args.mlsd_repo}")
            print(f"M-LSD device: {args.mlsd_device}")
    else:
        print(f"F-Clip config: {args.fclip_config}")
        print(f"F-Clip checkpoint: {args.fclip_checkpoint}")
        print(f"F-Clip device: {args.fclip_device}")
    print(f"Segmentation model: {args.seg_model}")
    print(f"YOLO inference device: {args.device}")
    print(f"ROI bootstrap bbox: {args.bbox_model if bbox_detector is not None else 'disabled'}")
    print(f"Thresholds: line={args.line_conf:.2f}, segmentation={args.seg_conf:.2f}")

    tracker = RoiTracker()
    source = parse_source(args.source)
    if is_image_source(source):
        frame = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if frame is None:
            raise SystemExit(f"Could not read image: {source}")
        output = process_frame(frame, bbox_detector, segmentation_model, line_detector, tracker, args)
        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.save), output):
                raise SystemExit(f"Could not write output image: {args.save}")
        if args.display:
            cv2.imshow(WINDOW_NAME, output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return 0

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")
    writer = None
    previous = perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            output = process_frame(frame, bbox_detector, segmentation_model, line_detector, tracker, args)
            if args.save_video:
                if writer is None:
                    writer = create_video_writer(
                        args.save_video,
                        output,
                        float(capture.get(cv2.CAP_PROP_FPS)),
                    )
                writer.write(output)
            now = perf_counter()
            fps = 1.0 / max(now - previous, 1e-6)
            previous = now
            cv2.putText(output, f"FPS {fps:.1f}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(output, f"FPS {fps:.1f}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 1, cv2.LINE_AA)
            if args.save and args.once:
                args.save.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(args.save), output)
            if args.display:
                cv2.imshow(WINDOW_NAME, output)
                key = cv2.waitKey(0 if args.once else 1) & 0xFF
                if key in (ord("q"), 27):
                    break
            if args.once:
                break
    finally:
        if writer is not None:
            writer.release()
        capture.release()
        if args.display:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
