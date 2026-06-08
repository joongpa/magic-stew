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
from live_recognition.piece_graph import PieceMask, build_piece_graph, draw_piece_graph  # noqa: E402
from live_recognition.visualization import (  # noqa: E402
    _gray_world_white_balance,
    _reliable_color_mask,
    rgb_origin_regression_clusters,
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
PIECE_TYPE_BY_CLASS_ID = {
    0: "center",
    1: "edge",
    2: "corner",
}
CENTER_CLASS_ID = 0
STICKER_COUNTS_BY_CLASS = {
    0: 1,  # center
    1: 2,  # edge
    2: 3,  # corner
}
COLOR_CLASSIFIER_MODES = ("closest-cluster-point", "linear-regression")


class ColorRayClassifier:
    def __init__(
        self,
        *,
        required_frames: int = 30,
        cluster_count: int = 6,
        min_brightness: float = 40.0,
        raw_density_filter_radius: float = 0.0,
        raw_density_filter_min_neighbors: int = 0,
        mode: str = "closest-cluster-point",
    ) -> None:
        if mode not in COLOR_CLASSIFIER_MODES:
            raise ValueError(f"Unsupported color classifier mode: {mode}")
        self.required_frames = max(1, required_frames)
        self.cluster_count = max(1, cluster_count)
        self.min_brightness = min_brightness
        self.raw_density_filter_radius = raw_density_filter_radius
        self.raw_density_filter_min_neighbors = raw_density_filter_min_neighbors
        self.mode = mode
        self.detected_frames = 0
        self._rgb_samples: list[np.ndarray] = []
        self.directions = np.empty((0, 3), dtype=np.float32)
        self.centroids = np.empty((0, 3), dtype=np.float32)
        self.cluster_samples: list[np.ndarray] = []

    @property
    def ready(self) -> bool:
        return len(self.directions) > 0 or len(self.cluster_samples) > 0

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
        self.directions, self.centroids, self.cluster_samples = rgb_origin_regression_clusters(
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

        if self.mode == "linear-regression":
            distances = distance_to_origin_rays(rgb, self.directions)
        else:
            distances = distance_to_rgb_clusters(rgb, self.cluster_samples)
            if len(distances) == 0:
                distances = distance_to_origin_rays(rgb, self.directions)
        if len(distances) == 0:
            return None

        return int(distances.argmin())


def distance_to_rgb_clusters(
    rgb_sample: np.ndarray,
    cluster_samples: list[np.ndarray],
) -> np.ndarray:
    point = np.asarray(rgb_sample, dtype=np.float32).reshape(3)
    distances = []
    for samples in cluster_samples:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1, 3)
        if len(samples) == 0:
            distances.append(np.inf)
            continue
        distances.append(float(np.min(np.linalg.norm(samples - point[None, :], axis=1))))
    return np.asarray(distances, dtype=np.float32)


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


def sticker_segments_from_piece(
    frame: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
    piece_mask: np.ndarray,
    class_id: int,
    *,
    spatial_weight: float = 1.0,
    max_iterations: int = 30,
    min_segment_ratio: float = 0.25,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    x1, y1, x2, y2 = roi_xyxy
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or not bool(piece_mask.any()):
        return []

    sticker_count = min(
        STICKER_COUNTS_BY_CLASS.get(class_id, 1),
        int(piece_mask.sum()),
    )
    if sticker_count <= 1:
        mean_ab, mean_bgr = lab_ab_samples_from_piece(
            frame,
            roi_xyxy,
            piece_mask,
            max_samples=1,
            average_color=True,
        )
        return [(piece_mask, mean_bgr[0], mean_ab[0])] if len(mean_bgr) else []

    labels = rgbxy_kmeans_labels(
        crop,
        piece_mask,
        cluster_count=sticker_count,
        spatial_weight=spatial_weight,
        max_iterations=max_iterations,
        min_segment_ratio=min_segment_ratio,
    )
    stickers = []
    for cluster_index in sorted(int(label) for label in np.unique(labels[labels >= 0])):
        sticker_mask = piece_mask & (labels == cluster_index)
        if not bool(sticker_mask.any()):
            continue
        mean_ab, mean_bgr = lab_ab_samples_from_piece(
            frame,
            roi_xyxy,
            sticker_mask,
            max_samples=1,
            average_color=True,
        )
        if len(mean_bgr) == 0:
            continue
        stickers.append((sticker_mask, mean_bgr[0], mean_ab[0]))
    return stickers


def rgbxy_kmeans_labels(
    crop_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    cluster_count: int,
    spatial_weight: float = 1.0,
    max_iterations: int = 30,
    min_segment_ratio: float = 0.25,
) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    labels_image = np.full(mask.shape, -1, dtype=np.int32)
    if len(xs) == 0 or cluster_count <= 0:
        return labels_image

    k = min(cluster_count, len(xs))
    rgb = crop_bgr[ys, xs, ::-1].astype(np.float32)
    x_scale = max(mask.shape[1] - 1, 1)
    y_scale = max(mask.shape[0] - 1, 1)
    xy = np.column_stack(
        [
            xs.astype(np.float32) / x_scale * 255.0,
            ys.astype(np.float32) / y_scale * 255.0,
        ]
    )
    features = np.column_stack([rgb, xy * float(spatial_weight)]).astype(np.float32)
    centers = initialize_kmeans_centers(features, k)
    labels = np.zeros(len(features), dtype=np.int32)
    for _ in range(max_iterations):
        distances = np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2)
        next_labels = distances.argmin(axis=1).astype(np.int32)
        next_centers = centers.copy()
        for cluster_index in range(k):
            members = features[next_labels == cluster_index]
            if len(members) > 0:
                next_centers[cluster_index] = members.mean(axis=0)
        if np.array_equal(labels, next_labels) and np.allclose(centers, next_centers):
            labels = next_labels
            break
        centers = next_centers
        labels = next_labels

    labels = merge_small_sticker_labels(
        labels,
        features,
        k,
        min_segment_ratio=min_segment_ratio,
    )
    labels = reorder_sticker_labels(labels, features, int(labels.max()) + 1)
    labels_image[ys, xs] = labels
    return labels_image


def initialize_kmeans_centers(features: np.ndarray, cluster_count: int) -> np.ndarray:
    centers = [features[0]]
    while len(centers) < cluster_count:
        distances = np.min(
            np.linalg.norm(features[:, None, :] - np.asarray(centers)[None, :, :], axis=2),
            axis=1,
        )
        centers.append(features[int(np.argmax(distances))])
    return np.asarray(centers, dtype=np.float32)


def reorder_sticker_labels(
    labels: np.ndarray,
    features: np.ndarray,
    cluster_count: int,
) -> np.ndarray:
    centers = []
    for cluster_index in range(cluster_count):
        members = features[labels == cluster_index]
        if len(members) == 0:
            centers.append((np.inf, np.inf, cluster_index))
            continue
        center = members.mean(axis=0)
        centers.append((float(center[4]), float(center[3]), cluster_index))
    ordered_indexes = [cluster_index for _, _, cluster_index in sorted(centers)]
    remap = {old_index: new_index for new_index, old_index in enumerate(ordered_indexes)}
    return np.asarray([remap[int(label)] for label in labels], dtype=np.int32)


def merge_small_sticker_labels(
    labels: np.ndarray,
    features: np.ndarray,
    cluster_count: int,
    *,
    min_segment_ratio: float,
) -> np.ndarray:
    if min_segment_ratio <= 0 or cluster_count <= 1:
        return labels

    sizes = np.bincount(labels, minlength=cluster_count)
    if len(sizes) == 0:
        return labels

    largest_size = int(sizes.max())
    if largest_size <= 0:
        return labels

    min_size = largest_size * float(min_segment_ratio)
    keep_indexes = np.flatnonzero(sizes >= min_size)
    if len(keep_indexes) == 0:
        keep_indexes = np.array([int(sizes.argmax())], dtype=np.int32)

    centers = []
    for cluster_index in range(cluster_count):
        members = features[labels == cluster_index]
        if len(members) == 0:
            centers.append(np.full(features.shape[1], np.inf, dtype=np.float32))
        else:
            centers.append(members.mean(axis=0))
    centers_array = np.asarray(centers, dtype=np.float32)
    kept_centers = centers_array[keep_indexes]
    kept_index_set = set(int(index) for index in keep_indexes)

    next_labels = labels.copy()
    for cluster_index in range(cluster_count):
        if cluster_index in kept_index_set:
            continue
        member_mask = labels == cluster_index
        if not bool(member_mask.any()):
            continue
        distances = np.linalg.norm(
            features[member_mask, None, :] - kept_centers[None, :, :],
            axis=2,
        )
        nearest_kept = keep_indexes[distances.argmin(axis=1)]
        next_labels[member_mask] = nearest_kept

    compact_indexes = {
        old_index: new_index for new_index, old_index in enumerate(sorted(set(next_labels)))
    }
    return np.asarray(
        [compact_indexes[int(label)] for label in next_labels],
        dtype=np.int32,
    )


def stickers_for_ready_color_classifier(
    frame: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
    stickers: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    color_classifier: ColorRayClassifier | None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, int | None]]:
    classified_stickers = []
    if color_classifier is None or not color_classifier.ready:
        return [(mask, bgr, ab, None) for mask, bgr, ab in stickers]

    grouped_masks: dict[tuple[str, int], list[np.ndarray]] = {}
    group_order = []
    for sticker_index, (sticker_mask, sticker_bgr, _,) in enumerate(stickers):
        cluster_index = (
            color_classifier.classify_bgr(sticker_bgr)
            if len(sticker_bgr) == 3
            else None
        )
        if cluster_index is not None and cluster_index < len(color_classifier.centroids):
            key = ("cluster", int(cluster_index))
        else:
            key = ("raw", sticker_index)
        if key not in grouped_masks:
            grouped_masks[key] = []
            group_order.append(key)
        grouped_masks[key].append(sticker_mask)

    for key in group_order:
        merged_mask = np.zeros_like(grouped_masks[key][0], dtype=bool)
        for sticker_mask in grouped_masks[key]:
            merged_mask |= sticker_mask
        mean_ab, mean_bgr = lab_ab_samples_from_piece(
            frame,
            roi_xyxy,
            merged_mask,
            max_samples=1,
            average_color=True,
        )
        if len(mean_bgr) == 0:
            continue
        cluster_index = key[1] if key[0] == "cluster" else None
        classified_stickers.append((merged_mask, mean_bgr[0], mean_ab[0], cluster_index))
    return classified_stickers


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

    source_frame = frame.copy()
    overlay = frame.copy()
    roi_height = y2 - y1
    roi_width = x2 - x1
    count = 0
    confidence_labels = []
    sticker_color_numbers = []
    classifier_bgr_batches = []
    piece_masks = []
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
        full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = draw_mask.astype(np.uint8) * 255
        piece_masks.append(
            PieceMask(
                piece_id=len(piece_masks),
                class_id=class_id,
                piece_type=PIECE_TYPE_BY_CLASS_ID.get(class_id, str(class_id)),
                confidence=confidence,
                mask=full_mask,
                roi_mask=draw_mask,
                roi_xyxy=roi_xyxy,
            )
        )

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

        raw_stickers = sticker_segments_from_piece(
            frame,
            roi_xyxy,
            draw_mask,
            class_id,
            min_segment_ratio=args.min_sticker_segment_ratio,
        )
        if not raw_stickers:
            raw_stickers = [
                (draw_mask, np.empty((0,), dtype=np.uint8), np.empty((0,), dtype=np.float32))
            ]
        display_stickers = stickers_for_ready_color_classifier(
            frame,
            roi_xyxy,
            raw_stickers,
            color_classifier,
        )

        roi_overlay = overlay[y1:y2, x1:x2]
        for sticker_mask, sticker_bgr, _, cluster_index in display_stickers:
            sticker_color = color
            if (
                color_classifier is not None
                and color_classifier.ready
                and getattr(args, "show_center_color_classification", True)
                and len(sticker_bgr) == 3
            ):
                if cluster_index is not None and cluster_index < len(color_classifier.centroids):
                    centroid_rgb = color_classifier.centroids[cluster_index]
                    sticker_color = tuple(int(value) for value in centroid_rgb[::-1])
                    sticker_ys, sticker_xs = np.nonzero(sticker_mask)
                    sticker_label_at = (
                        int(sticker_xs.mean()) + x1,
                        int(sticker_ys.mean()) + y1,
                    )
                    sticker_color_numbers.append((sticker_label_at, cluster_index + 1))
            roi_overlay[sticker_mask] = sticker_color

        sticker_ab_batches = []
        sticker_bgr_batches = []
        for sticker_mask, _, _, _ in display_stickers:
            sticker_ab_samples, sticker_bgr_samples = lab_ab_samples_from_piece(
                frame,
                roi_xyxy,
                sticker_mask,
                args.color_plot_samples_per_frame,
                args.average_mask_color,
            )
            if len(sticker_ab_samples) > 0:
                sticker_ab_batches.append(sticker_ab_samples)
                sticker_bgr_batches.append(sticker_bgr_samples)
        ab_samples, bgr_samples = concatenate_samples(
            sticker_ab_batches,
            sticker_bgr_batches,
            args.color_plot_samples_per_frame,
        )
        if len(ab_samples) > 0:
            ab_batches.append(ab_samples)
            bgr_batches.append(bgr_samples)
        if color_classifier is not None:
            classifier_bgr_samples = np.asarray(
                [sticker_bgr for _, sticker_bgr, _ in raw_stickers if len(sticker_bgr) == 3],
                dtype=np.uint8,
            ).reshape(-1, 3)
            if len(classifier_bgr_samples) > 0:
                classifier_bgr_batches.append(classifier_bgr_samples)
        count += 1

    mask = cv2.addWeighted(frame, 1.0 - args.mask_opacity, overlay, args.mask_opacity, 0)
    frame[:] = mask
    if args.draw_piece_graph and piece_masks:
        graph = build_piece_graph(
            source_frame,
            piece_masks,
            adjacency_radius_px=args.graph_adjacency_radius,
            min_component_area_px=args.graph_min_sticker_area,
            min_sticker_segment_ratio=args.min_sticker_segment_ratio,
        )
        frame[:] = draw_piece_graph(
            frame,
            graph,
            roi_xyxy=roi_xyxy,
            draw_masks=False,
        )
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
    if args.show_confidence or args.debug_overlay:
        draw_center_color_numbers(frame, sticker_color_numbers)
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
    parser.add_argument(
        "--min-sticker-segment-ratio",
        type=float,
        default=0.25,
        help=(
            "Merge RGBXY sticker segments smaller than this fraction of the largest "
            "segment before recognition and color plotting; 0 disables"
        ),
    )
    parser.add_argument(
        "--draw-piece-graph",
        action="store_true",
        help="Draw piece graph nodes and edges on top of the segmentation overlay",
    )
    parser.add_argument(
        "--adjacency-radius",
        "--graph-adjacency-radius",
        dest="graph_adjacency_radius",
        type=int,
        default=6,
        help="Pixel distance threshold used when connecting nearby graph nodes",
    )
    parser.add_argument(
        "--graph-min-sticker-area",
        type=int,
        default=12,
        help="Minimum sticker area kept as a graph node",
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
    parser.add_argument(
        "--color-classifier-mode",
        choices=COLOR_CLASSIFIER_MODES,
        default="closest-cluster-point",
        help=(
            "Classify recognized sticker colors by nearest retained cluster sample "
            "or by distance to the learned RGB origin-ray regression"
        ),
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
        mode=args.color_classifier_mode,
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
