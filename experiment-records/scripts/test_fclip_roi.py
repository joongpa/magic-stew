"""Render the pretrained F-Clip line detector on a tracked cube ROI.

The flow matches ``test_line_detection_roi.py``:

By default this temporary baseline starts from the full-frame square and does
not load the cube bbox detector. F-Clip then tracks its ROI from line bounds.
Pass ``--use-bbox`` to restore the normal bbox fallback flow.

The official F-Clip repository and checkpoint are supplied separately because
the upstream project keeps its model code and weights outside this repository.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from live_recognition.detectors import YOLOCubeDetector  # noqa: E402
from live_recognition.line_detectors import build_line_detector  # noqa: E402
from live_recognition.roi_tracking import RoiTracker  # noqa: E402
from test_line_detection_roi import (  # noqa: E402
    IMAGE_EXTENSIONS,
    WINDOW_NAME,
    create_video_writer,
    parse_source,
    process_frame,
)

DEFAULT_BBOX_MODEL = ROOT / "runs/500ExamplesWeights.pt"
DEFAULT_FCLIP_REPO = ROOT / "F-Clip"
DEFAULT_FCLIP_MODEL = DEFAULT_FCLIP_REPO / "checkpoints/HG1_D2/checkpoint.pth.tar"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="Camera index, video path, or image path")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_FCLIP_MODEL,
        help="F-Clip checkpoint (defaults to the downloaded HG1_D2 weights)",
    )
    parser.add_argument(
        "--fclip-repo",
        type=Path,
        default=DEFAULT_FCLIP_REPO,
        help="Cloned official F-Clip repository",
    )
    parser.add_argument(
        "--fclip-variant",
        choices=("HG1_D2", "HG1_D3", "HG1", "HG2", "HG2_LB", "HR"),
        default="HG1_D2",
    )
    parser.add_argument(
        "--bbox-model",
        type=Path,
        default=DEFAULT_BBOX_MODEL,
        help="Cube bbox model (used only with --use-bbox)",
    )
    parser.add_argument(
        "--use-bbox",
        dest="disable_bbox",
        action="store_false",
        help="Re-enable cube bbox acquisition instead of starting from the full frame",
    )
    parser.add_argument("--bbox-conf", type=float, default=0.25)
    parser.add_argument("--bbox-imgsz", type=int, default=640)
    parser.add_argument("--roi-padding", type=float, default=0.08)
    parser.add_argument("--roi-tracking-gain", type=float, default=1.4)
    parser.add_argument("--roi-tracking-padding", type=float, default=0.18)
    parser.add_argument(
        "--input-size",
        type=int,
        default=128,
        help="F-Clip square input size (temporarily defaults to 128)",
    )
    parser.add_argument("--line-score", type=float, default=0.40)
    parser.add_argument("--line-min-length", type=float, default=5.0)
    parser.add_argument("--max-lines", type=int, default=300)
    parser.add_argument("--device", default="auto", help="F-Clip device: auto, cpu, or cuda")
    parser.add_argument("--show-scores", action="store_true")
    parser.add_argument("--show-bbox", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug-overlay", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--save", type=Path)
    parser.add_argument("--save-video", type=Path)
    parser.add_argument("--display", dest="display", action="store_true")
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.set_defaults(display=True, detector="fclip", disable_bbox=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.disable_bbox and not args.bbox_model.is_file():
        raise SystemExit(f"Could not find bbox model: {args.bbox_model}")
    if not args.fclip_repo.is_dir():
        raise SystemExit(f"Could not find F-Clip repository: {args.fclip_repo}")
    if not args.model.is_file():
        raise SystemExit(f"Could not find F-Clip checkpoint: {args.model}")

    bbox_detector = None
    if not args.disable_bbox:
        bbox_detector = YOLOCubeDetector(
            model_name=str(args.bbox_model),
            confidence_threshold=args.bbox_conf,
            image_size=args.bbox_imgsz,
        )
    line_detector = build_line_detector(
        "fclip",
        repo_path=args.fclip_repo,
        checkpoint_path=args.model,
        variant=args.fclip_variant,
        input_size=args.input_size,
        score_threshold=args.line_score,
        min_length=args.line_min_length,
        max_lines=args.max_lines,
        device=args.device,
    )
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
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            output = process_frame(frame, bbox_detector, line_detector, args, tracker)
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
                if not cv2.imwrite(str(args.save), output):
                    raise SystemExit(f"Could not write output image: {args.save}")
            if args.display:
                cv2.imshow(WINDOW_NAME, output)
                key = cv2.waitKey(0 if args.once else 1) & 0xFF
                if key in (ord("q"), 27):
                    break
            if args.once:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
