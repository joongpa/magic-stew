from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from live_recognition.detectors import HeuristicCubeDetector, SAM3CubeDetector, YOLOCubeDetector
from live_recognition.pipeline import CubeOrientationPipeline
from live_recognition.types import FrameResult
from live_recognition.visualization import (
    draw_result,
    lab_ab_samples_from_bbox,
    save_lab_ab_plot,
    save_lab_ab_samples_plot,
)


def _parse_source(source: str) -> int | str:
    source = source.strip().strip("\"'").strip()
    return int(source) if source.isdigit() else source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rubik's Cube bounding-box detector POC")
    parser.add_argument("--source", default="0", help="Camera index, video path, or image path")
    parser.add_argument(
        "--detector",
        choices=["sam3", "heuristic", "yolo"],
        default="sam3",
        help="Detector backend to use",
    )
    parser.add_argument("--model", help="Model weights path")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--prompt", default="3x3x3 rubik's cube", help="SAM 3 text prompt")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument("--mask-overlay", action="store_true", help="Draw SAM mask overlay when available")
    parser.add_argument("--once", action="store_true", help="Process one frame/image and exit")
    parser.add_argument("--save", type=Path, help="Optional output image path for --once mode")
    parser.add_argument("--display", action="store_true", help="Show an OpenCV preview window")
    parser.add_argument(
        "--no-detect",
        action="store_true",
        help="Skip detector/model inference and only render debug overlays that do not need detections",
    )
    parser.add_argument(
        "--color-debug",
        action="store_true",
        help="Show a cube-crop LAB chroma inset with glare/shadow pixels dimmed",
    )
    parser.add_argument(
        "--color-plot",
        type=Path,
        help="Optional output path for a LAB a/b scatter plot of reliable cube-bbox pixels",
    )
    parser.add_argument(
        "--color-plot-frames",
        type=int,
        default=90,
        help="Maximum detected video frames to aggregate for --color-plot",
    )
    parser.add_argument(
        "--color-plot-samples-per-frame",
        type=int,
        default=800,
        help="Maximum LAB a/b samples collected per detected video frame",
    )
    return parser


def _build_pipeline(args: argparse.Namespace) -> CubeOrientationPipeline:
    if args.detector == "heuristic":
        detector = HeuristicCubeDetector()
    elif args.detector == "yolo":
        detector = YOLOCubeDetector(
            model_name=args.model or "runs/500ExamplesWeights.pt",
            confidence_threshold=args.conf,
            image_size=args.imgsz,
        )
    else:
        detector = SAM3CubeDetector(
            model_name=args.model or "sam3.pt",
            confidence_threshold=args.conf,
            prompt=args.prompt,
            return_mask=args.mask_overlay,
        )
    return CubeOrientationPipeline(detector=detector)


def main() -> int:
    args = build_parser().parse_args()
    if args.no_detect and args.color_plot:
        raise SystemExit("--color-plot requires cube detections; remove --no-detect")
    pipeline = None if args.no_detect else _build_pipeline(args)
    source = _parse_source(args.source)

    if isinstance(source, str) and Path(source).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
        frame = cv2.imread(source)
        if frame is None:
            raise SystemExit(f"Could not read image: {source}")
        result = FrameResult(candidate=None, orientations=[])
        if pipeline is not None:
            result = pipeline.process_frame(frame)
        output = draw_result(
            frame,
            result,
            draw_mask_overlay=args.mask_overlay,
            draw_color_debug=args.color_debug,
        )
        if args.color_plot:
            if result.candidate is None:
                raise SystemExit("Could not save color plot because no cube was detected")
            save_lab_ab_plot(frame, result.candidate.bbox_xyxy, str(args.color_plot))
        if args.save:
            cv2.imwrite(str(args.save), output)
        if args.display:
            cv2.imshow("cube bbox", output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return 0

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    previous_frame_time = perf_counter()
    smoothed_fps: float | None = None
    color_plot_ab_batches = []
    color_plot_bgr_batches = []
    color_plot_detected_frames = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        result = FrameResult(candidate=None, orientations=[])
        if pipeline is not None:
            result = pipeline.process_frame(frame)
        now = perf_counter()
        elapsed = now - previous_frame_time
        previous_frame_time = now
        if elapsed > 0:
            instant_fps = 1.0 / elapsed
            smoothed_fps = (
                instant_fps if smoothed_fps is None else 0.9 * smoothed_fps + 0.1 * instant_fps
            )

        output = draw_result(
            frame,
            result,
            draw_mask_overlay=args.mask_overlay,
            fps=smoothed_fps,
            draw_color_debug=args.color_debug,
        )
        if args.save and args.once:
            cv2.imwrite(str(args.save), output)
        if args.color_plot and args.once:
            if result.candidate is None:
                raise SystemExit("Could not save color plot because no cube was detected")
            save_lab_ab_plot(frame, result.candidate.bbox_xyxy, str(args.color_plot))
        elif args.color_plot and result.candidate is not None:
            if color_plot_detected_frames < args.color_plot_frames:
                ab_samples, bgr_samples = lab_ab_samples_from_bbox(
                    frame,
                    result.candidate.bbox_xyxy,
                    max_samples=args.color_plot_samples_per_frame,
                )
                if len(ab_samples) > 0:
                    color_plot_ab_batches.append(ab_samples)
                    color_plot_bgr_batches.append(bgr_samples)
                    color_plot_detected_frames += 1
                    if color_plot_detected_frames >= args.color_plot_frames and not args.display:
                        break

        if args.display:
            cv2.imshow("cube bbox", output)
            key = cv2.waitKey(1 if not args.once else 0) & 0xFF
            if key == ord("q"):
                break

        if args.once:
            break

    capture.release()
    if args.display:
        cv2.destroyAllWindows()
    if args.color_plot and not args.once:
        if not color_plot_ab_batches:
            raise SystemExit("Could not save color plot because no cube detections were collected")
        save_lab_ab_samples_plot(
            np.concatenate(color_plot_ab_batches, axis=0),
            np.concatenate(color_plot_bgr_batches, axis=0),
            str(args.color_plot),
            title=f"Cube bbox reliable pixels across {color_plot_detected_frames} frames",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
