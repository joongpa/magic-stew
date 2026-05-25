"""Test piece-type YOLO segmentation on a cube bounding-box ROI.

Pipeline:
1. Detect the whole cube with the existing bbox detector.
2. Crop a padded square ROI around that bbox.
3. Run the new center/edge/corner segmentation model on the ROI.
4. Draw the ROI-space masks back onto the original OpenCV frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from live_recognition.detectors import YOLOCubeDetector  # noqa: E402
from live_recognition.visualization import (  # noqa: E402
    lab_ab_samples_from_bbox,
    save_lab_ab_samples_plot,
    save_rgb_samples_plot,
)


DEFAULT_BBOX_MODEL = ROOT / "runs/500ExamplesWeights.pt"
DEFAULT_PIECE_MODEL = (
    ROOT
    / "runs/segment/runs/cube_piece_seg_local/runs/cube_piece_yolo11n_seg/weights/best.pt"
)
CLASS_COLORS = {
    0: (80, 220, 255),   # center
    1: (80, 255, 120),   # edge
    2: (255, 120, 80),   # corner
}


def parse_source(source: str) -> int | str:
    source = source.strip().strip("\"'").strip()
    return int(source) if source.isdigit() else source


def clipped_square_roi(
    bbox_xyxy: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * (1.0 + padding * 2.0)

    rx1 = int(round(cx - side / 2.0))
    ry1 = int(round(cy - side / 2.0))
    rx2 = int(round(cx + side / 2.0))
    ry2 = int(round(cy + side / 2.0))

    rx1 = max(0, min(frame_width - 1, rx1))
    ry1 = max(0, min(frame_height - 1, ry1))
    rx2 = max(rx1 + 1, min(frame_width, rx2))
    ry2 = max(ry1 + 1, min(frame_height, ry2))
    return rx1, ry1, rx2, ry2


def apply_mask_padding(mask: np.ndarray, padding_px: int) -> np.ndarray:
    if padding_px == 0 or not bool(mask.any()):
        return mask

    radius = abs(padding_px)
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    mask_u8 = mask.astype(np.uint8) * 255
    if padding_px < 0:
        padded = cv2.erode(mask_u8, kernel, iterations=1)
    else:
        padded = cv2.dilate(mask_u8, kernel, iterations=1)
    return padded > 0


def draw_piece_segments(
    frame: np.ndarray,
    roi: np.ndarray,
    piece_model,
    roi_xyxy,
    args,
) -> tuple[int, np.ndarray]:
    x1, y1, x2, y2 = roi_xyxy
    results = piece_model.predict(
        source=roi,
        conf=args.piece_conf,
        imgsz=args.piece_imgsz,
        verbose=False,
    )
    sample_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    if not results:
        return 0, sample_mask

    result = results[0]
    masks = result.masks
    boxes = result.boxes
    if masks is None or boxes is None:
        return 0, sample_mask

    overlay = frame.copy()
    roi_height = y2 - y1
    roi_width = x2 - x1
    count = 0
    confidence_labels = []
    mask_data = masks.data.detach().cpu().numpy()
    sample_roi_mask = sample_mask[y1:y2, x1:x2]
    for index, mask in enumerate(mask_data):
        class_id = int(boxes.cls[index].item())
        color = CLASS_COLORS.get(class_id, (255, 255, 255))

        mask = cv2.resize(
            mask.astype(np.float32),
            (roi_width, roi_height),
            interpolation=cv2.INTER_LINEAR,
        )
        mask_bool = mask > 0.5
        if not mask_bool.any():
            continue

        ys, xs = np.nonzero(mask_bool)
        label_at = (int(xs.mean()) + x1, int(ys.mean()) + y1)
        confidence = float(boxes.conf[index].item())
        confidence_labels.append((label_at, confidence))

        if args.debug_overlay:
            contours, _ = cv2.findContours(
                mask_bool.astype(np.uint8) * 255,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            shifted_contours = []
            for contour in contours:
                contour = contour.copy()
                contour[:, 0, 0] += x1
                contour[:, 0, 1] += y1
                shifted_contours.append(contour)
            for points in shifted_contours:
                if len(points) < 3:
                    continue
                cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)

        roi_overlay = overlay[y1:y2, x1:x2]
        roi_overlay[mask_bool] = color
        sample_area = apply_mask_padding(mask_bool, args.segmentation_padding_px)
        if sample_area.any():
            sample_roi_mask[sample_area] = 255
        count += 1

    mask = cv2.addWeighted(frame, 1.0 - args.mask_opacity, overlay, args.mask_opacity, 0)
    frame[:] = mask
    if args.show_confidence or args.debug_overlay:
        for label_at, confidence in confidence_labels:
            cv2.putText(
                frame,
                f"{confidence:.2f}",
                label_at,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"{confidence:.2f}",
                label_at,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
    return count, sample_mask


def process_frame(
    frame: np.ndarray,
    bbox_detector,
    piece_model,
    args,
) -> tuple[np.ndarray, np.ndarray | None, tuple[int, int, int, int] | None]:
    output = frame.copy()
    candidate = bbox_detector.detect(frame)
    if candidate is None:
        return output, None, None

    height, width = frame.shape[:2]
    roi_xyxy = clipped_square_roi(candidate.bbox_xyxy, width, height, args.roi_padding)
    x1, y1, x2, y2 = roi_xyxy
    roi = frame[y1:y2, x1:x2]
    segment_count, sample_mask = draw_piece_segments(output, roi, piece_model, roi_xyxy, args)

    if args.show_bbox or args.debug_overlay:
        cv2.rectangle(output, (x1, y1), (x2, y2), (40, 220, 40), 2)
    if args.debug_overlay:
        cv2.putText(
            output,
            f"cube {candidate.confidence:.2f} pieces {segment_count}",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (40, 220, 40),
            2,
            cv2.LINE_AA,
        )
    return output, sample_mask, roi_xyxy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Camera index, video path, or image path")
    parser.add_argument("--bbox-model", type=Path, default=DEFAULT_BBOX_MODEL)
    parser.add_argument("--piece-model", type=Path, default=DEFAULT_PIECE_MODEL)
    parser.add_argument("--bbox-conf", type=float, default=0.25)
    parser.add_argument("--piece-conf", type=float, default=0.25)
    parser.add_argument("--bbox-imgsz", type=int, default=640)
    parser.add_argument("--piece-imgsz", type=int, default=256)
    parser.add_argument("--roi-padding", type=float, default=0.08)
    parser.add_argument(
        "--segmentation-padding-px",
        type=int,
        default=-2,
        help="Pixel padding applied to each piece mask before sampling; negative values erode",
    )
    parser.add_argument("--mask-opacity", type=float, default=1.0)
    parser.add_argument("--color-plot", type=Path, help="Output PNG path for LAB a/b samples")
    parser.add_argument("--rgb-color-plot", type=Path, help="Output PNG path for 3D RGB samples")
    parser.add_argument("--color-plot-frames", type=int, default=90)
    parser.add_argument("--color-plot-samples-per-frame", type=int, default=800)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--save", type=Path)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--hide-bbox", dest="show_bbox", action="store_false")
    parser.add_argument("--hide-confidence", dest="show_confidence", action="store_false")
    parser.add_argument(
        "--debug-overlay",
        action="store_true",
        help="Draw ROI box, polygon outlines, labels, and FPS",
    )
    parser.set_defaults(show_bbox=True, show_confidence=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    from ultralytics import YOLO

    bbox_detector = YOLOCubeDetector(
        model_name=str(args.bbox_model),
        confidence_threshold=args.bbox_conf,
        image_size=args.bbox_imgsz,
    )
    piece_model = YOLO(str(args.piece_model))
    source = parse_source(args.source)
    color_plot_ab_batches = []
    color_plot_bgr_batches = []
    color_plot_detected_frames = 0

    if isinstance(source, str) and Path(source).suffix.lower() in {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    }:
        frame = cv2.imread(source)
        if frame is None:
            raise SystemExit(f"Could not read image: {source}")
        output, sample_mask, roi_xyxy = process_frame(frame, bbox_detector, piece_model, args)
        if args.color_plot or args.rgb_color_plot:
            if sample_mask is None or roi_xyxy is None:
                raise SystemExit(
                    "Could not collect color plot samples because no pieces were segmented"
                )
            ab_samples, bgr_samples = lab_ab_samples_from_bbox(
                frame,
                roi_xyxy,
                sample_mask=sample_mask,
                max_samples=args.color_plot_samples_per_frame,
            )
            if args.color_plot:
                save_lab_ab_samples_plot(
                    ab_samples,
                    bgr_samples,
                    str(args.color_plot),
                    title="Piece segmentation reliable pixels in LAB a/b",
                )
            if args.rgb_color_plot:
                save_rgb_samples_plot(
                    bgr_samples,
                    str(args.rgb_color_plot),
                    title="Piece segmentation reliable pixels in RGB",
                )
        if args.save:
            cv2.imwrite(str(args.save), output)
        if args.display:
            cv2.imshow("cube piece segmentation", output)
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
        output, sample_mask, roi_xyxy = process_frame(frame, bbox_detector, piece_model, args)
        now = perf_counter()
        fps = 1.0 / max(now - previous, 1e-6)
        previous = now
        if args.debug_overlay:
            cv2.putText(
                output,
                f"FPS {fps:.1f}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                3,
            )
            cv2.putText(
                output,
                f"FPS {fps:.1f}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (20, 20, 20),
                1,
            )

        if args.save and args.once:
            cv2.imwrite(str(args.save), output)
        if args.color_plot or args.rgb_color_plot:
            if sample_mask is not None and roi_xyxy is not None:
                if color_plot_detected_frames < args.color_plot_frames:
                    ab_samples, bgr_samples = lab_ab_samples_from_bbox(
                        frame,
                        roi_xyxy,
                        sample_mask=sample_mask,
                        max_samples=args.color_plot_samples_per_frame,
                    )
                    if len(ab_samples) > 0:
                        color_plot_ab_batches.append(ab_samples)
                        color_plot_bgr_batches.append(bgr_samples)
                        color_plot_detected_frames += 1
                        if (
                            color_plot_detected_frames >= args.color_plot_frames
                            and not args.display
                        ):
                            break
        if args.display:
            cv2.imshow("cube piece segmentation", output)
            key = cv2.waitKey(0 if args.once else 1) & 0xFF
            if key == ord("q"):
                break
        if args.once:
            break

    capture.release()
    if args.display:
        cv2.destroyAllWindows()
    if args.color_plot or args.rgb_color_plot:
        if not color_plot_ab_batches:
            raise SystemExit("Could not collect color plot samples from any segmented frames")
        ab_samples = np.concatenate(color_plot_ab_batches, axis=0)
        bgr_samples = np.concatenate(color_plot_bgr_batches, axis=0)
        if args.color_plot:
            save_lab_ab_samples_plot(
                ab_samples,
                bgr_samples,
                str(args.color_plot),
                title=(
                    "Piece segmentation reliable pixels "
                    f"across {color_plot_detected_frames} frames"
                ),
            )
        if args.rgb_color_plot:
            save_rgb_samples_plot(
                bgr_samples,
                str(args.rgb_color_plot),
                title=f"Piece segmentation RGB samples across {color_plot_detected_frames} frames",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
