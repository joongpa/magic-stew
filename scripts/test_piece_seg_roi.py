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
    _gray_world_white_balance,
    _reliable_color_mask,
    rgb_origin_regression_rays,
    save_lab_ab_samples_plot,
    save_hsv_samples_interactive_plot,
    save_rgb_samples_interactive_plot,
    save_rgb_samples_plot,
)


DEFAULT_BBOX_MODEL = ROOT / "runs/500ExamplesWeights.pt"
DEFAULT_PIECE_MODEL = (
    ROOT
    / "runs/cube_piece_seg_local/runs/cube_piece_yolo11n_seg-3/weights/best.pt"
)
CLASS_COLORS = {
    0: (80, 220, 255),   # center
    1: (80, 255, 120),   # edge
    2: (255, 120, 80),   # corner
}
CENTER_CLASS_ID = 0


class ColorRayClassifier:
    def __init__(
        self,
        *,
        required_frames: int = 30,
        cluster_count: int = 6,
        min_brightness: float = 40.0,
        raw_density_filter_radius: float = 0.0,
        raw_density_filter_min_neighbors: int = 0,
    ) -> None:
        self.required_frames = max(1, required_frames)
        self.cluster_count = max(1, cluster_count)
        self.min_brightness = min_brightness
        self.raw_density_filter_radius = raw_density_filter_radius
        self.raw_density_filter_min_neighbors = raw_density_filter_min_neighbors
        self.detected_frames = 0
        self._rgb_samples: list[np.ndarray] = []
        self.directions = np.empty((0, 3), dtype=np.float32)
        self.centroids = np.empty((0, 3), dtype=np.float32)

    @property
    def ready(self) -> bool:
        return len(self.directions) > 0

    def observe_frame(self, bgr_samples: np.ndarray) -> None:
        if self.ready or len(bgr_samples) == 0:
            return

        bgr_samples = np.asarray(bgr_samples, dtype=np.uint8).reshape(-1, 3)
        rgb_samples = bgr_samples[:, ::-1]
        self._rgb_samples.extend(rgb_samples)
        self.detected_frames += 1
        if self.detected_frames < self.required_frames:
            return

        samples = np.asarray(self._rgb_samples, dtype=np.uint8)
        self.directions, self.centroids = rgb_origin_regression_rays(
            samples,
            cluster_count=self.cluster_count,
            min_brightness=self.min_brightness,
            raw_density_filter_radius=self.raw_density_filter_radius,
            raw_density_filter_min_neighbors=self.raw_density_filter_min_neighbors,
        )

    def classify_bgr(self, bgr_sample: np.ndarray) -> int | None:
        if not self.ready:
            return None

        rgb = np.asarray(bgr_sample, dtype=np.float32).reshape(3)[::-1]
        if float(np.linalg.norm(rgb)) < self.min_brightness:
            return None

        distances = distance_to_origin_rays(rgb, self.directions)
        if len(distances) == 0:
            return None

        return int(distances.argmin())


def distance_to_origin_rays(
    rgb_sample: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    point = np.asarray(rgb_sample, dtype=np.float32).reshape(3)
    if len(directions) == 0:
        return np.empty((0,), dtype=np.float32)

    projections = np.maximum(directions @ point, 0.0)
    closest_points = directions * projections[:, None]
    return np.linalg.norm(point[None, :] - closest_points, axis=1)


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


def piece_mask_bool(result, index: int, roi_width: int, roi_height: int) -> np.ndarray:
    mask_data = result.masks.data.detach().cpu().numpy()
    mask = cv2.resize(
        mask_data[index].astype(np.float32),
        (roi_width, roi_height),
        interpolation=cv2.INTER_LINEAR,
    )
    return mask > 0.5


def lab_ab_samples_from_piece(
    frame: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
    piece_mask: np.ndarray,
    max_samples: int,
    average_color: bool,
) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = roi_xyxy
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or not bool(piece_mask.any()):
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    reliable_mask = _reliable_color_mask(crop) & piece_mask
    if int(reliable_mask.sum()) < 10:
        reliable_mask = piece_mask.copy()
    if not bool(reliable_mask.any()):
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    balanced = _gray_world_white_balance(crop, reliable_mask)
    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)
    ab_samples = lab[:, :, 1:3][reliable_mask].astype(np.float32)
    bgr_samples = crop[reliable_mask]
    if average_color:
        mean_bgr_sample = (
            bgr_samples.astype(np.float32)
            .mean(axis=0, keepdims=True)
            .clip(0, 255)
            .astype(np.uint8)
        )
        return (
            ab_samples.mean(axis=0, keepdims=True).astype(np.float32),
            mean_bgr_sample,
        )
    if len(ab_samples) > max_samples:
        indexes = np.linspace(0, len(ab_samples) - 1, max_samples, dtype=np.int32)
        ab_samples = ab_samples[indexes]
        bgr_samples = bgr_samples[indexes]
    return ab_samples, bgr_samples


def concatenate_samples(
    ab_batches: list[np.ndarray],
    bgr_batches: list[np.ndarray],
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not ab_batches:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    ab_samples = np.concatenate(ab_batches, axis=0)
    bgr_samples = np.concatenate(bgr_batches, axis=0)
    if len(ab_samples) > max_samples:
        indexes = np.linspace(0, len(ab_samples) - 1, max_samples, dtype=np.int32)
        ab_samples = ab_samples[indexes]
        bgr_samples = bgr_samples[indexes]
    return ab_samples, bgr_samples


def draw_piece_segments(
    frame: np.ndarray,
    roi: np.ndarray,
    piece_model,
    roi_xyxy,
    args,
    color_classifier: ColorRayClassifier | None = None,
) -> tuple[int, np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = roi_xyxy
    results = piece_model.predict(
        source=roi,
        conf=args.piece_conf,
        imgsz=args.piece_imgsz,
        verbose=False,
    )
    ab_batches = []
    bgr_batches = []
    if not results:
        return 0, np.empty((0, 2), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    result = results[0]
    masks = result.masks
    boxes = result.boxes
    if masks is None or boxes is None:
        return 0, np.empty((0, 2), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    overlay = frame.copy()
    roi_height = y2 - y1
    roi_width = x2 - x1
    count = 0
    confidence_labels = []
    center_color_numbers = []
    classifier_bgr_batches = []
    for index in range(len(boxes)):
        class_id = int(boxes.cls[index].item())
        color = CLASS_COLORS.get(class_id, (255, 255, 255))

        mask_bool = piece_mask_bool(result, index, roi_width, roi_height)
        if not mask_bool.any():
            continue

        draw_mask = apply_mask_padding(mask_bool, args.segmentation_padding_px)
        if not draw_mask.any():
            continue

        ys, xs = np.nonzero(draw_mask)
        label_at = (int(xs.mean()) + x1, int(ys.mean()) + y1)
        confidence = float(boxes.conf[index].item())
        confidence_labels.append((label_at, confidence))

        if args.debug_overlay:
            contours, _ = cv2.findContours(
                draw_mask.astype(np.uint8) * 255,
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

        _, center_bgr_samples = (
            lab_ab_samples_from_piece(
                frame,
                roi_xyxy,
                draw_mask,
                max_samples=1,
                average_color=True,
            )
            if color_classifier is not None and class_id == CENTER_CLASS_ID
            else (np.empty((0, 2), dtype=np.float32), np.empty((0, 3), dtype=np.uint8))
        )
        if (
            color_classifier is not None
            and color_classifier.ready
            and getattr(args, "show_center_color_classification", True)
            and class_id == CENTER_CLASS_ID
            and len(center_bgr_samples) > 0
        ):
            cluster_index = color_classifier.classify_bgr(center_bgr_samples[0])
            if cluster_index is not None and cluster_index < len(color_classifier.centroids):
                centroid_rgb = color_classifier.centroids[cluster_index]
                color = tuple(int(value) for value in centroid_rgb[::-1])
                center_color_numbers.append((label_at, cluster_index + 1))

        roi_overlay = overlay[y1:y2, x1:x2]
        roi_overlay[draw_mask] = color
        ab_samples, bgr_samples = lab_ab_samples_from_piece(
            frame,
            roi_xyxy,
            draw_mask,
            args.color_plot_samples_per_frame,
            args.average_mask_color,
        )
        if len(ab_samples) > 0:
            ab_batches.append(ab_samples)
            bgr_batches.append(bgr_samples)
        if color_classifier is not None:
            if args.average_mask_color:
                _, classifier_bgr_samples = lab_ab_samples_from_piece(
                    frame,
                    roi_xyxy,
                    draw_mask,
                    args.color_plot_samples_per_frame,
                    average_color=False,
                )
            else:
                classifier_bgr_samples = bgr_samples
            if len(classifier_bgr_samples) > 0:
                classifier_bgr_batches.append(classifier_bgr_samples)
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
    draw_center_color_numbers(frame, center_color_numbers)
    ab_samples, bgr_samples = concatenate_samples(
        ab_batches,
        bgr_batches,
        args.color_plot_samples_per_frame,
    )
    if color_classifier is not None:
        classifier_bgr_samples = (
            np.concatenate(classifier_bgr_batches, axis=0)
            if classifier_bgr_batches
            else np.empty((0, 3), dtype=np.uint8)
        )
        color_classifier.observe_frame(classifier_bgr_samples)
    return count, ab_samples, bgr_samples


def draw_center_color_numbers(
    frame: np.ndarray,
    center_color_numbers: list[tuple[tuple[int, int], int]],
) -> None:
    for label_at, color_number in center_color_numbers:
        text = str(color_number)
        origin = (label_at[0] - 8, label_at[1] + 8)
        cv2.putText(
            frame,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def process_frame(
    frame: np.ndarray,
    bbox_detector,
    piece_model,
    args,
    color_classifier: ColorRayClassifier | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output = frame.copy()
    candidate = bbox_detector.detect(frame)
    if candidate is None:
        return output, np.empty((0, 2), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    height, width = frame.shape[:2]
    roi_xyxy = clipped_square_roi(candidate.bbox_xyxy, width, height, args.roi_padding)
    x1, y1, x2, y2 = roi_xyxy
    roi = frame[y1:y2, x1:x2]
    segment_count, ab_samples, bgr_samples = draw_piece_segments(
        output,
        roi,
        piece_model,
        roi_xyxy,
        args,
        color_classifier,
    )

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
    return output, ab_samples, bgr_samples


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
    parser.add_argument(
        "--rgb-html-plot",
        type=Path,
        help="Output HTML path for interactive 3D RGB samples",
    )
    parser.add_argument(
        "--hsv-html-plot",
        type=Path,
        help="Output HTML path for interactive 3D HSV samples",
    )
    parser.add_argument(
        "--normalize-rgb-plot",
        action="store_true",
        help="Plot RGB coordinates as R,G,B divided by R+G+B",
    )
    parser.add_argument(
        "--hide-rgb-regressions",
        dest="show_rgb_regressions",
        action="store_false",
        help="Do not draw k-means origin-ray regressions on the interactive RGB plot",
    )
    parser.add_argument(
        "--rgb-regression-clusters",
        type=int,
        default=6,
        help="Number of RGB origin-ray regressions to fit for the interactive plot",
    )
    parser.add_argument(
        "--rgb-regression-min-brightness",
        type=float,
        default=40.0,
        help="Minimum raw RGB vector length used while fitting RGB regressions",
    )
    parser.add_argument(
        "--rgb-regression-raw-density-filter-radius",
        type=float,
        default=0.0,
        help="Reject RGB samples with too few raw-space neighbors inside this radius; 0 disables",
    )
    parser.add_argument(
        "--rgb-regression-raw-density-filter-min-neighbors",
        type=int,
        default=0,
        help="Minimum other samples required inside the raw-density filter radius; 0 disables",
    )
    parser.add_argument(
        "--average-mask-color",
        action="store_true",
        help="Plot one average color per segmented piece mask instead of pixel samples",
    )
    parser.add_argument(
        "--hide-center-color-classification",
        dest="show_center_color_classification",
        action="store_false",
        help="Do not recolor center masks using the learned color rays",
    )
    parser.add_argument(
        "--center-color-frames",
        type=int,
        default=30,
        help="Detected frames of all-mask color samples to collect before recoloring centers",
    )
    parser.add_argument(
        "--center-color-min-brightness",
        type=float,
        default=40.0,
        help="Minimum raw RGB vector length used while fitting color rays",
    )
    parser.add_argument("--color-plot-frames", type=int, default=90)
    parser.add_argument("--color-plot-samples-per-frame", type=int, default=800)
    parser.add_argument("--average-plot-point-size", type=float, default=36.0)
    parser.add_argument(
        "--plotly-point-size",
        type=float,
        default=1.0,
        help="Marker size for interactive Plotly RGB/HSV plots",
    )
    parser.add_argument(
        "--plotly-regression-line-width",
        type=float,
        default=4.0,
        help="Line width for RGB regression rays in the interactive Plotly plot",
    )
    parser.add_argument(
        "--plotly-color-by-cluster",
        action="store_true",
        help="Color interactive Plotly points by learned RGB cluster instead of raw color",
    )
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
    parser.set_defaults(
        show_bbox=True,
        show_confidence=True,
        show_rgb_regressions=True,
        show_center_color_classification=True,
    )
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
    plot_point_size = args.average_plot_point_size if args.average_mask_color else 4
    color_classifier = ColorRayClassifier(
        required_frames=args.center_color_frames,
        cluster_count=args.rgb_regression_clusters,
        min_brightness=args.center_color_min_brightness,
        raw_density_filter_radius=args.rgb_regression_raw_density_filter_radius,
        raw_density_filter_min_neighbors=args.rgb_regression_raw_density_filter_min_neighbors,
    )
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
        output, ab_samples, bgr_samples = process_frame(
            frame,
            bbox_detector,
            piece_model,
            args,
            color_classifier,
        )
        if args.color_plot or args.rgb_color_plot or args.rgb_html_plot or args.hsv_html_plot:
            if len(ab_samples) == 0:
                raise SystemExit(
                    "Could not collect color plot samples because no pieces were segmented"
                )
            if args.color_plot:
                save_lab_ab_samples_plot(
                    ab_samples,
                    bgr_samples,
                    str(args.color_plot),
                    title="Piece segmentation reliable pixels in LAB a/b",
                    point_size=plot_point_size,
                )
            if args.rgb_color_plot:
                save_rgb_samples_plot(
                    bgr_samples,
                    str(args.rgb_color_plot),
                    title="Piece segmentation reliable pixels in RGB",
                    point_size=plot_point_size,
                    normalize_rgb=args.normalize_rgb_plot,
                    color_by_cluster=args.plotly_color_by_cluster,
                    cluster_count=args.rgb_regression_clusters,
                    cluster_min_brightness=args.rgb_regression_min_brightness,
                    raw_density_filter_radius=args.rgb_regression_raw_density_filter_radius,
                    raw_density_filter_min_neighbors=args.rgb_regression_raw_density_filter_min_neighbors,
                )
            if args.rgb_html_plot:
                save_rgb_samples_interactive_plot(
                    bgr_samples,
                    str(args.rgb_html_plot),
                    title="Piece segmentation reliable pixels in RGB",
                    normalize_rgb=args.normalize_rgb_plot,
                    point_size=args.plotly_point_size,
                    draw_regression_lines=args.show_rgb_regressions,
                    regression_cluster_count=args.rgb_regression_clusters,
                    regression_min_brightness=args.rgb_regression_min_brightness,
                    regression_line_width=args.plotly_regression_line_width,
                    color_by_cluster=args.plotly_color_by_cluster,
                    raw_density_filter_radius=args.rgb_regression_raw_density_filter_radius,
                    raw_density_filter_min_neighbors=args.rgb_regression_raw_density_filter_min_neighbors,
                )
            if args.hsv_html_plot:
                save_hsv_samples_interactive_plot(
                    bgr_samples,
                    str(args.hsv_html_plot),
                    title="Piece segmentation reliable pixels in HSV",
                    point_size=args.plotly_point_size,
                    color_by_cluster=args.plotly_color_by_cluster,
                    cluster_count=args.rgb_regression_clusters,
                    cluster_min_brightness=args.rgb_regression_min_brightness,
                    raw_density_filter_radius=args.rgb_regression_raw_density_filter_radius,
                    raw_density_filter_min_neighbors=args.rgb_regression_raw_density_filter_min_neighbors,
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
        output, ab_samples, bgr_samples = process_frame(
            frame,
            bbox_detector,
            piece_model,
            args,
            color_classifier,
        )
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
        if args.color_plot or args.rgb_color_plot or args.rgb_html_plot or args.hsv_html_plot:
            if color_plot_detected_frames < args.color_plot_frames and len(ab_samples) > 0:
                color_plot_ab_batches.append(ab_samples)
                color_plot_bgr_batches.append(bgr_samples)
                color_plot_detected_frames += 1
                if color_plot_detected_frames >= args.color_plot_frames and not args.display:
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
    if args.color_plot or args.rgb_color_plot or args.rgb_html_plot or args.hsv_html_plot:
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
                point_size=plot_point_size,
            )
        if args.rgb_color_plot:
            save_rgb_samples_plot(
                bgr_samples,
                str(args.rgb_color_plot),
                title=f"Piece segmentation RGB samples across {color_plot_detected_frames} frames",
                point_size=plot_point_size,
                normalize_rgb=args.normalize_rgb_plot,
                color_by_cluster=args.plotly_color_by_cluster,
                cluster_count=args.rgb_regression_clusters,
                cluster_min_brightness=args.rgb_regression_min_brightness,
                raw_density_filter_radius=args.rgb_regression_raw_density_filter_radius,
                raw_density_filter_min_neighbors=args.rgb_regression_raw_density_filter_min_neighbors,
            )
        if args.rgb_html_plot:
            save_rgb_samples_interactive_plot(
                bgr_samples,
                str(args.rgb_html_plot),
                title=f"Piece segmentation RGB samples across {color_plot_detected_frames} frames",
                normalize_rgb=args.normalize_rgb_plot,
                point_size=args.plotly_point_size,
                draw_regression_lines=args.show_rgb_regressions,
                regression_cluster_count=args.rgb_regression_clusters,
                regression_min_brightness=args.rgb_regression_min_brightness,
                regression_line_width=args.plotly_regression_line_width,
                color_by_cluster=args.plotly_color_by_cluster,
                raw_density_filter_radius=args.rgb_regression_raw_density_filter_radius,
                raw_density_filter_min_neighbors=args.rgb_regression_raw_density_filter_min_neighbors,
            )
        if args.hsv_html_plot:
            save_hsv_samples_interactive_plot(
                bgr_samples,
                str(args.hsv_html_plot),
                title=f"Piece segmentation HSV samples across {color_plot_detected_frames} frames",
                point_size=args.plotly_point_size,
                color_by_cluster=args.plotly_color_by_cluster,
                cluster_count=args.rgb_regression_clusters,
                cluster_min_brightness=args.rgb_regression_min_brightness,
                raw_density_filter_radius=args.rgb_regression_raw_density_filter_radius,
                raw_density_filter_min_neighbors=args.rgb_regression_raw_density_filter_min_neighbors,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
