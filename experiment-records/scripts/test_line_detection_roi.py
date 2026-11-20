"""Render a second-stage line detector on a tracked cube ROI.

The flow is shared with the other OpenCV model renderers:

1. Run the cube bbox detector only when the second-stage detector has no usable
   detections from the previous frame.
2. Run M-LSD or LINEA on the current square ROI.
3. Move the ROI from the second-stage detection bounds.
4. Reacquire the cube bbox on the next frame after a second-stage miss.

M-LSD expects an official ``.tflite`` model. LINEA expects a clone of its
official repository plus its config and checkpoint.
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
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from live_recognition.detectors import YOLOCubeDetector  # noqa: E402
from live_recognition.line_detectors import LineDetector, build_line_detector  # noqa: E402
from live_recognition.roi_tracking import RoiTracker, roi_for_frame, square_roi_from_center  # noqa: E402

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_BBOX_MODEL = ROOT / "runs/500ExamplesWeights.pt"
WINDOW_NAME = "cube line detector ROI"


def parse_source(source: str) -> int | str:
    source = source.strip().strip("\"'").strip()
    return int(source) if source.isdigit() else source


def create_video_writer(path: Path, frame: np.ndarray, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    suffix = path.suffix.lower()
    fourcc_name = "mp4v" if suffix in {".mp4", ".m4v", ".mov"} else "MJPG"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*fourcc_name),
        float(fps if np.isfinite(fps) and fps > 0 else 30.0),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    return writer


def draw_lines(
    output: np.ndarray,
    lines_xyxy: np.ndarray,
    scores: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
    *,
    show_scores: bool,
) -> None:
    x_offset, y_offset = roi_xyxy[:2]
    for line, score in zip(lines_xyxy, scores):
        x1, y1, x2, y2 = np.rint(line).astype(int)
        start = (int(x1 + x_offset), int(y1 + y_offset))
        end = (int(x2 + x_offset), int(y2 + y_offset))
        cv2.line(output, start, end, (30, 220, 60), 2, cv2.LINE_AA)
        cv2.circle(output, start, 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(output, end, 3, (255, 255, 255), -1, cv2.LINE_AA)
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
    bbox_detector: YOLOCubeDetector,
    line_detector: LineDetector,
    args: argparse.Namespace,
    tracker: RoiTracker,
) -> np.ndarray:
    output = frame.copy()
    if getattr(args, "disable_bbox", False):
        height, width = frame.shape[:2]
        had_tracker_roi = tracker.roi_xyxy is not None
        if not had_tracker_roi:
            roi_xyxy = square_roi_from_center(
                width / 2.0,
                height / 2.0,
                min(width, height),
                width,
                height,
            )
            tracker.roi_xyxy = roi_xyxy
            tracker.base_side = min(roi_xyxy[2] - roi_xyxy[0], roi_xyxy[3] - roi_xyxy[1])
        else:
            roi_xyxy = tracker.roi_xyxy
        candidate = None
        roi_source = "tracker" if had_tracker_roi else "full-frame"
    else:
        roi_xyxy, candidate, roi_source = roi_for_frame(
            frame,
            bbox_detector,
            tracker,
            args.roi_padding,
        )
    if roi_xyxy is None:
        cv2.putText(
            output,
            "no cube bbox",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return output

    x1, y1, x2, y2 = roi_xyxy
    roi = frame[y1:y2, x1:x2]
    detections = line_detector.detect(roi)
    draw_lines(
        output,
        detections.lines_xyxy,
        detections.scores,
        roi_xyxy,
        show_scores=args.show_scores,
    )

    height, width = frame.shape[:2]
    tracker.observe_tracking_bounds(
        detections.bounds_xyxy,
        width,
        height,
        tracking_gain=args.roi_tracking_gain,
        tracking_padding=args.roi_tracking_padding,
    )
    active_roi = tracker.roi_xyxy or roi_xyxy
    ax1, ay1, ax2, ay2 = active_roi
    roi_color = (0, 0, 255) if roi_source == "detector" else (40, 220, 40)
    if args.show_bbox:
        cv2.rectangle(output, (ax1, ay1), (ax2 - 1, ay2 - 1), roi_color, 2)
    if args.debug_overlay:
        label = (
            f"{args.detector} lines {detections.count} "
            f"{'bbox' if roi_source == 'detector' else 'tracked'}"
        )
        if candidate is not None:
            label += f" cube {candidate.confidence:.2f}"
        cv2.putText(
            output,
            label,
            (ax1, max(24, ay1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            roi_color,
            2,
            cv2.LINE_AA,
        )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="Camera index, video path, or image path")
    parser.add_argument("--detector", choices=("mlsd", "linea"), required=True)
    parser.add_argument("--model", type=Path, required=True, help="M-LSD .tflite or LINEA checkpoint")
    parser.add_argument("--bbox-model", type=Path, default=DEFAULT_BBOX_MODEL)
    parser.add_argument("--bbox-conf", type=float, default=0.25)
    parser.add_argument("--bbox-imgsz", type=int, default=640)
    parser.add_argument("--roi-padding", type=float, default=0.08)
    parser.add_argument("--roi-tracking-gain", type=float, default=1.4)
    parser.add_argument("--roi-tracking-padding", type=float, default=0.18)
    parser.add_argument("--line-imgsz", type=int, default=None)
    parser.add_argument("--line-score", type=float, default=None)
    parser.add_argument("--line-min-length", type=float, default=5.0)
    parser.add_argument("--max-lines", type=int, default=300)
    parser.add_argument("--linea-repo", type=Path, help="Cloned official LINEA repository")
    parser.add_argument("--linea-config", type=Path, help="LINEA config .py file")
    parser.add_argument("--device", default="cpu", help="LINEA torch device")
    parser.add_argument("--show-scores", action="store_true")
    parser.add_argument("--show-bbox", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug-overlay", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--save", type=Path)
    parser.add_argument("--save-video", type=Path)
    parser.add_argument("--display", dest="display", action="store_true")
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.set_defaults(display=True)
    return parser


def build_detector(args: argparse.Namespace) -> LineDetector:
    if not args.model.is_file():
        raise SystemExit(f"Could not find second-stage model: {args.model}")
    if args.detector == "mlsd":
        return build_line_detector(
            "mlsd",
            model_path=args.model,
            input_size=args.line_imgsz or 512,
            score_threshold=args.line_score if args.line_score is not None else 0.10,
            min_length=args.line_min_length,
            max_lines=args.max_lines,
        )
    if args.linea_repo is None or args.linea_config is None:
        raise SystemExit("LINEA requires --linea-repo and --linea-config")
    return build_line_detector(
        "linea",
        repo_path=args.linea_repo,
        config_path=args.linea_config,
        checkpoint_path=args.model,
        input_size=args.line_imgsz or 640,
        score_threshold=args.line_score if args.line_score is not None else 0.40,
        min_length=args.line_min_length,
        max_lines=args.max_lines,
        device=args.device,
    )


def main() -> int:
    args = build_parser().parse_args()
    if not args.bbox_model.is_file():
        raise SystemExit(f"Could not find bbox model: {args.bbox_model}")
    bbox_detector = YOLOCubeDetector(
        model_name=str(args.bbox_model),
        confidence_threshold=args.bbox_conf,
        image_size=args.bbox_imgsz,
    )
    line_detector = build_detector(args)
    tracker = RoiTracker()
    source = parse_source(args.source)

    if isinstance(source, str) and Path(source).suffix.lower() in IMAGE_EXTENSIONS:
        frame = cv2.imread(source, cv2.IMREAD_COLOR)
        if frame is None:
            raise SystemExit(f"Could not read image: {source}")
        output = process_frame(frame, bbox_detector, line_detector, args, tracker)
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
            output = process_frame(frame, bbox_detector, line_detector, args, tracker)
            now = perf_counter()
            if args.debug_overlay:
                fps = 1.0 / max(now - previous, 1e-6)
                cv2.putText(
                    output,
                    f"FPS {fps:.1f}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            previous = now
            if args.save_video:
                if writer is None:
                    writer = create_video_writer(
                        args.save_video,
                        output,
                        float(capture.get(cv2.CAP_PROP_FPS)),
                    )
                writer.write(output)
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
