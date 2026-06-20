"""Test YOLO sticker segmentation on a cube bounding-box ROI.

Pipeline:
1. Detect the whole cube with the existing bbox detector.
2. Crop a padded square ROI around that bbox.
3. Run the sticker segmentation model on the ROI.
4. Draw segmented sticker masks back onto the original OpenCV frame.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from live_recognition.cube_net import (  # noqa: E402
    CubeNetState,
    build_cube_net_from_graph,
    draw_cube_net,
)
from live_recognition.detectors import YOLOCubeDetector  # noqa: E402
from live_recognition.piece_graph import (  # noqa: E402
    PieceGraph,
    StickerNode,
    _legal_mask_adjacencies,
    draw_piece_graph as draw_sticker_graph,
)
from live_recognition.visualization import (  # noqa: E402
    _reliable_color_mask,
    save_hsv_samples_interactive_plot,
    save_rgb_samples_interactive_plot,
    save_rgb_samples_plot,
)


DEFAULT_BBOX_MODEL = ROOT / "runs/500ExamplesWeights.pt"
DEFAULT_STICKER_MODEL = (
    ROOT
    / "runs/segment/runs/cube_face_seg_local/runs/cube_face_sticker_yolo11n_seg/weights/best.pt"
)
DEFAULT_STICKER_MASK_COLOR = (150, 150, 150)
FACE_INSTANCE_MASK_COLORS = (
    (255, 80, 80),
    (80, 210, 120),
    (80, 150, 255),
    (255, 210, 80),
    (220, 120, 255),
    (80, 230, 230),
)
FACE_PARALLELOGRAM_COLORS = (
    (255, 255, 255),
    (40, 255, 255),
    (255, 180, 40),
    (120, 255, 120),
    (255, 120, 255),
    (120, 180, 255),
)
STICKER_CLASS_NAMES = {"face", "sticker", "u_sticker", "f_sticker", "r_sticker"}
FACE_COVERAGE_PRUNE_THRESHOLD = 0.8
MAX_FACE_COUNT = 3
FACE_OUTER_CORNER_MIN_SLIDE_SCALE = 0.72
FACE_OUTER_CORNER_SLIDE_STEPS = 29
PROJECTED_FACE_GRID_COLOR = (255, 255, 255)
STICKER_COLOR_SAMPLE_COUNT = 10
DENSITY_CLUSTER_EPS_RADIANS = 0.08
DENSITY_CLUSTER_MIN_POINTS = 4
CUBE_GRID_FIT_SAMPLE_POINTS = 160
CUBE_GRID_FIT_REFINE_CANDIDATES = 8
CUBE_GRID_FIT_SUPPORTED_LINE_DISTANCE = 0.18
CUBE_GRID_FIT_DIRECTION_TOLERANCE_RADIANS = 0.32
EMPTY_SPACE_LINE_WIDTH_PX = 5
BASIS_VECTOR_ANGLE_BUCKET_DEGREES = 5.0


@dataclass
class RoiTracker:
    roi_xyxy: tuple[int, int, int, int] | None = None
    cube_candidate: object | None = None
    base_side: int | None = None

    def reset(self) -> None:
        self.roi_xyxy = None
        self.cube_candidate = None
        self.base_side = None

    def initialize(
        self,
        candidate,
        frame_width: int,
        frame_height: int,
        padding: float,
    ) -> tuple[int, int, int, int]:
        self.cube_candidate = candidate
        self.roi_xyxy = square_roi_from_bbox(
            candidate.bbox_xyxy,
            frame_width,
            frame_height,
            padding,
        )
        x1, y1, x2, y2 = self.roi_xyxy
        self.base_side = min(x2 - x1, y2 - y1)
        return self.roi_xyxy

    def observe_tracking_bounds(
        self,
        tracking_bounds_xyxy: tuple[int, int, int, int] | None,
        frame_width: int,
        frame_height: int,
        tracking_gain: float = 1.0,
        tracking_padding: float = 0.0,
    ) -> None:
        if self.roi_xyxy is None:
            return
        if tracking_bounds_xyxy is None:
            self.reset()
            return

        x1, y1, x2, y2 = self.roi_xyxy
        side = min(x2 - x1, y2 - y1)
        base_side = self.base_side if self.base_side is not None else side
        bx1, by1, bx2, by2 = tracking_bounds_xyxy
        tracking_point_x = (bx1 + bx2) / 2.0
        tracking_point_y = (by1 + by2) / 2.0
        observed_side = max(bx2 - bx1, by2 - by1)
        target_side = max(base_side, observed_side) * (
            1.0 + max(0.0, float(tracking_padding)) * 2.0
        )
        roi_center_x = side / 2.0
        roi_center_y = side / 2.0
        gain = max(0.0, float(tracking_gain))
        dx = (tracking_point_x - roi_center_x) * gain
        dy = (tracking_point_y - roi_center_y) * gain
        center_x = (x1 + x2) / 2.0 + dx
        center_y = (y1 + y2) / 2.0 + dy
        self.roi_xyxy = square_roi_from_center(
            center_x,
            center_y,
            target_side,
            frame_width,
            frame_height,
        )


@dataclass(frozen=True)
class ProjectedCubeGridFit:
    score: float
    line_edge_distance_score: float
    coverage_ratio: float
    hull_area_ratio: float
    hull_points_xy: np.ndarray
    line_segments_xy: list[tuple[tuple[float, float], tuple[float, float]]]


@dataclass(frozen=True)
class LowIntersectionLine:
    score: float
    intersection_px: int
    length_px: int
    angle_degrees: float
    start_xy: tuple[float, float]
    end_xy: tuple[float, float]


@dataclass(frozen=True)
class EmptySpaceLineOverlay:
    lines: list[LowIntersectionLine]
    sampled_points_xy: np.ndarray
    line_width_px: int = EMPTY_SPACE_LINE_WIDTH_PX
    quad_edge_segments_xy: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2, 2), dtype=np.float32)
    )
    basis_vectors_xy: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32)
    )
    basis_center_xy: tuple[float, float] | None = None


@dataclass(frozen=True)
class StickerEdgePointSample:
    points_xy: np.ndarray
    sticker_indexes: np.ndarray


@dataclass(frozen=True)
class StickerQuadGeometry:
    points_xy: np.ndarray
    sticker_indexes: np.ndarray
    edge_segments_xy: np.ndarray


class ColorDensityClassifier:
    def __init__(
        self,
        *,
        required_frames: int = 30,
        cluster_count: int = 6,
        min_brightness: float = 40.0,
        density_cluster_eps_radians: float = DENSITY_CLUSTER_EPS_RADIANS,
        density_cluster_min_neighbors: int = DENSITY_CLUSTER_MIN_POINTS,
    ) -> None:
        self.required_frames = max(1, required_frames)
        self.cluster_count = max(1, cluster_count)
        self.min_brightness = min_brightness
        self.density_cluster_eps_radians = density_cluster_eps_radians
        self.density_cluster_min_neighbors = max(1, density_cluster_min_neighbors)
        self.detected_frames = 0
        self._rgb_samples: list[np.ndarray] = []
        self.centroids = np.empty((0, 3), dtype=np.float32)
        self.cluster_samples: list[np.ndarray] = []

    @property
    def ready(self) -> bool:
        return len(self.cluster_samples) > 0

    def observe_frame(self, bgr_samples: np.ndarray) -> None:
        if self.ready or len(bgr_samples) == 0:
            return

        bgr_samples = np.asarray(bgr_samples, dtype=np.uint8).reshape(-1, 3)
        self._rgb_samples.extend(bgr_samples[:, ::-1])
        self.detected_frames += 1
        if self.detected_frames < self.required_frames:
            return

        self.centroids, self.cluster_samples = density_rgb_clusters(
            np.asarray(self._rgb_samples, dtype=np.uint8),
            target_clusters=self.cluster_count,
            min_brightness=self.min_brightness,
            eps_radians=self.density_cluster_eps_radians,
            min_points=self.density_cluster_min_neighbors,
        )

    def classify_bgr(self, bgr_sample: np.ndarray) -> int | None:
        if not self.ready:
            return None

        rgb = np.asarray(bgr_sample, dtype=np.float32).reshape(3)[::-1]
        if float(np.linalg.norm(rgb)) < self.min_brightness:
            return None

        distances = angular_distance_to_normalized_rgb_clusters(
            rgb,
            self.cluster_samples,
        )
        if len(distances) == 0 or float(distances.min()) > self.density_cluster_eps_radians:
            return None
        return int(distances.argmin())

    def classify_bgr_samples(self, bgr_samples: np.ndarray) -> int | None:
        if not self.ready:
            return None

        samples = np.asarray(bgr_samples, dtype=np.uint8).reshape(-1, 3)
        votes = [
            cluster_index
            for sample in samples
            if (cluster_index := self.classify_bgr(sample)) is not None
        ]
        if not votes:
            return None
        return int(np.bincount(np.asarray(votes, dtype=np.int32)).argmax())


def normalized_rgb_points(rgb_samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(rgb_samples, dtype=np.float32).reshape(-1, 3)
    brightness = np.maximum(np.linalg.norm(samples, axis=1, keepdims=True), 1.0)
    return samples / brightness


def angular_distance_to_normalized_rgb_clusters(
    rgb_sample: np.ndarray,
    cluster_samples: list[np.ndarray],
) -> np.ndarray:
    point = normalized_rgb_points(np.asarray(rgb_sample, dtype=np.float32).reshape(1, 3))[0]
    distances = []
    for samples in cluster_samples:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1, 3)
        if len(samples) == 0:
            distances.append(np.inf)
            continue
        normalized_samples = normalized_rgb_points(samples)
        max_cosine = float(np.max(np.clip(normalized_samples @ point, -1.0, 1.0)))
        distances.append(float(np.arccos(max_cosine)))
    return np.asarray(distances, dtype=np.float32)


def density_rgb_clusters(
    rgb_samples: np.ndarray,
    *,
    target_clusters: int,
    min_brightness: float,
    eps_radians: float = DENSITY_CLUSTER_EPS_RADIANS,
    min_points: int = DENSITY_CLUSTER_MIN_POINTS,
) -> tuple[np.ndarray, list[np.ndarray]]:
    samples = np.asarray(rgb_samples, dtype=np.float32).reshape(-1, 3)
    if len(samples) == 0 or target_clusters <= 0:
        return np.empty((0, 3), dtype=np.float32), []

    brightness = np.linalg.norm(samples, axis=1)
    usable = samples[brightness >= min_brightness]
    if len(usable) < min_points:
        return np.empty((0, 3), dtype=np.float32), []

    labels = density_bin_labels(
        normalized_rgb_points(usable),
        eps_radians=eps_radians,
        min_points=min_points,
    )

    clusters = []
    for label in sorted(set(int(value) for value in labels if value >= 0)):
        members = usable[labels == label]
        if len(members) >= min_points:
            clusters.append(members.astype(np.float32, copy=True))
    clusters = merge_smallest_clusters_to_nearest_neighbor(
        clusters,
        target_count=target_clusters,
    )

    centroids = []
    retained_clusters = []
    for members in clusters:
        centroid = members.mean(axis=0)
        if float(np.linalg.norm(centroid)) <= 1e-6:
            continue
        centroids.append(centroid)
        retained_clusters.append(members)

    return np.asarray(centroids, dtype=np.float32), retained_clusters


def merge_smallest_clusters_to_nearest_neighbor(
    clusters: list[np.ndarray],
    *,
    target_count: int,
) -> list[np.ndarray]:
    clusters = [np.asarray(cluster, dtype=np.float32).reshape(-1, 3) for cluster in clusters]
    clusters = [cluster for cluster in clusters if len(cluster) > 0]
    while len(clusters) > target_count:
        smallest_index = min(range(len(clusters)), key=lambda index: len(clusters[index]))
        nearest_index = nearest_cluster_index(clusters, smallest_index)
        clusters[nearest_index] = np.concatenate(
            [clusters[nearest_index], clusters[smallest_index]],
            axis=0,
        )
        del clusters[smallest_index]
    clusters.sort(key=len, reverse=True)
    return clusters


def nearest_cluster_index(clusters: list[np.ndarray], source_index: int) -> int:
    source_centroid = normalized_rgb_points(clusters[source_index]).mean(axis=0)
    best_index = 0 if source_index != 0 else 1
    best_distance = np.inf
    for index, cluster in enumerate(clusters):
        if index == source_index:
            continue
        centroid = normalized_rgb_points(cluster).mean(axis=0)
        distance = float(np.linalg.norm(source_centroid - centroid))
        if distance < best_distance:
            best_distance = distance
            best_index = index
    return best_index


def density_bin_labels(
    unit_samples: np.ndarray,
    *,
    eps_radians: float,
    min_points: int,
) -> np.ndarray:
    samples = np.asarray(unit_samples, dtype=np.float32).reshape(-1, 3)
    labels = np.full(len(samples), -1, dtype=np.int32)
    if len(samples) == 0:
        return labels

    eps = max(float(eps_radians), 1e-6)
    chord_radius = float(np.sqrt(max(0.0, 2.0 - 2.0 * np.cos(eps))))
    cell_size = max(chord_radius, 1e-6)
    cell_keys, sample_cell_indexes, cell_counts = np.unique(
        np.floor(samples / cell_size).astype(np.int32),
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    cell_lookup = {
        tuple(int(value) for value in cell_key): cell_index
        for cell_index, cell_key in enumerate(cell_keys)
    }
    cell_neighbor_counts = np.zeros(len(cell_keys), dtype=np.int32)
    for cell_index, cell_key in enumerate(cell_keys):
        base_cell = tuple(int(value) for value in cell_key)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbor_index = cell_lookup.get(
                        (base_cell[0] + dx, base_cell[1] + dy, base_cell[2] + dz)
                    )
                    if neighbor_index is not None:
                        cell_neighbor_counts[cell_index] += cell_counts[neighbor_index]

    dense_cells = cell_neighbor_counts >= int(min_points)
    cell_labels = np.full(len(cell_keys), -1, dtype=np.int32)
    cluster_id = 0
    for start_cell in np.flatnonzero(dense_cells):
        if cell_labels[start_cell] >= 0:
            continue
        queue = [int(start_cell)]
        cell_labels[start_cell] = cluster_id
        while queue:
            current = queue.pop()
            base_cell = tuple(int(value) for value in cell_keys[current])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        neighbor_index = cell_lookup.get(
                            (base_cell[0] + dx, base_cell[1] + dy, base_cell[2] + dz)
                        )
                        if (
                            neighbor_index is None
                            or not dense_cells[neighbor_index]
                            or cell_labels[neighbor_index] >= 0
                        ):
                            continue
                        cell_labels[neighbor_index] = cluster_id
                        queue.append(neighbor_index)
        cluster_id += 1

    for sample_index, cell_index in enumerate(sample_cell_indexes):
        label = cell_labels[cell_index]
        if label >= 0:
            labels[sample_index] = label
    return labels


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


def square_roi_from_bbox(
    bbox_xyxy: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * (1.0 + padding * 2.0)
    return square_roi_from_center(cx, cy, side, frame_width, frame_height)


def square_roi_from_center(
    center_x: float,
    center_y: float,
    side: float,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    side_px = int(round(side))
    side_px = max(1, min(side_px, frame_width, frame_height))
    x1 = int(round(center_x - side_px / 2.0))
    y1 = int(round(center_y - side_px / 2.0))
    x1 = max(0, min(frame_width - side_px, x1))
    y1 = max(0, min(frame_height - side_px, y1))
    return x1, y1, x1 + side_px, y1 + side_px


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


def sticker_mask_bool(
    mask_data: np.ndarray,
    index: int,
    roi_width: int,
    roi_height: int,
) -> np.ndarray:
    mask = cv2.resize(
        mask_data[index].astype(np.float32),
        (roi_width, roi_height),
        interpolation=cv2.INTER_LINEAR,
    )
    return mask > 0.5


def low_intersection_sticker_lines(
    sticker_masks: list[np.ndarray],
    *,
    roi_shape: tuple[int, int],
    line_width_px: int = EMPTY_SPACE_LINE_WIDTH_PX,
    angle_bucket_degrees: float = BASIS_VECTOR_ANGLE_BUCKET_DEGREES,
    **_: object,
) -> EmptySpaceLineOverlay:
    union_mask = _union_sticker_mask(sticker_masks, roi_shape)
    if union_mask is None:
        return EmptySpaceLineOverlay(lines=[], sampled_points_xy=np.empty((0, 2), dtype=np.float32))
    line_width = max(1.0, float(line_width_px))
    quad_geometry = _sticker_quad_geometry(
        sticker_masks,
        roi_shape=roi_shape,
    )
    quad_points = quad_geometry.points_xy
    if len(quad_points) < 2 or len(quad_geometry.edge_segments_xy) == 0:
        return EmptySpaceLineOverlay(
            lines=[],
            sampled_points_xy=quad_points,
            line_width_px=int(round(line_width)),
            quad_edge_segments_xy=quad_geometry.edge_segments_xy,
        )

    lines: list[LowIntersectionLine] = []
    for edge_start, edge_end in quad_geometry.edge_segments_xy:
        candidate = _infinite_line_from_points(edge_start, edge_end)
        if candidate is None:
            continue
        point, direction, _normal = candidate
        segment = _clip_infinite_line_to_roi(point, direction, roi_shape)
        if segment is None:
            continue
        start, end = segment
        length_px = int(round(float(np.linalg.norm(end - start))))
        angle_degrees = float(np.degrees(np.arctan2(direction[1], direction[0])) % 180.0)
        lines.append(
            LowIntersectionLine(
                score=0.0,
                intersection_px=0,
                length_px=length_px,
                angle_degrees=angle_degrees,
                start_xy=(float(start[0]), float(start[1])),
                end_xy=(float(end[0]), float(end[1])),
            )
        )

    basis_vectors = _basis_vectors_from_line_angle_histogram(
        lines,
        bucket_degrees=angle_bucket_degrees,
    )
    roi_height, roi_width = roi_shape
    return EmptySpaceLineOverlay(
        lines=lines,
        sampled_points_xy=quad_points,
        line_width_px=int(round(line_width)),
        quad_edge_segments_xy=quad_geometry.edge_segments_xy,
        basis_vectors_xy=basis_vectors,
        basis_center_xy=((roi_width - 1) / 2.0, (roi_height - 1) / 2.0),
    )


def _basis_vectors_from_line_angle_histogram(
    lines: list[LowIntersectionLine],
    *,
    bucket_degrees: float,
    basis_count: int = 2,
) -> np.ndarray:
    if not lines or basis_count <= 0:
        return np.empty((0, 2), dtype=np.float32)

    bucket_size = max(float(bucket_degrees), 1e-6)
    bucket_count = max(1, int(np.ceil(180.0 / bucket_size)))
    angles = np.asarray([line.angle_degrees % 180.0 for line in lines], dtype=np.float32)
    bucket_indexes = np.minimum(
        np.floor(angles / bucket_size).astype(np.int32),
        bucket_count - 1,
    )
    counts = np.bincount(bucket_indexes, minlength=bucket_count).astype(np.int32)
    peak_scores = (
        counts
        + np.roll(counts, 1)
        + np.roll(counts, -1)
    )

    vectors = []
    suppressed = np.zeros(bucket_count, dtype=bool)
    for _ in range(basis_count):
        available_scores = peak_scores.copy()
        available_scores[suppressed] = -1
        peak_index = int(np.argmax(available_scores))
        if available_scores[peak_index] <= 0:
            break

        center = (peak_index + 0.5) * bucket_size
        lower = center - bucket_size * 1.5
        upper = center + bucket_size * 1.5
        nearby = np.asarray(
            [_angle_in_half_open_axial_window(angle, lower, upper) for angle in angles],
            dtype=bool,
        )
        vector = _average_axial_angle_vector(angles[nearby])
        if vector is not None:
            vectors.append(vector)

        for neighbor_offset in (-1, 0, 1):
            suppressed[(peak_index + neighbor_offset) % bucket_count] = True

    if not vectors:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(vectors, dtype=np.float32)


def _angle_in_half_open_axial_window(angle_degrees: float, lower: float, upper: float) -> bool:
    angle = float(angle_degrees) % 180.0
    width = max(float(upper - lower), 0.0)
    lower = float(lower) % 180.0
    if width >= 180.0:
        return True
    return ((angle - lower) % 180.0) < width


def _average_axial_angle_vector(angles_degrees: np.ndarray) -> np.ndarray | None:
    angles = np.asarray(angles_degrees, dtype=np.float32).reshape(-1)
    if len(angles) == 0:
        return None
    doubled = np.radians(angles * 2.0)
    mean_x = float(np.cos(doubled).mean())
    mean_y = float(np.sin(doubled).mean())
    if float(np.hypot(mean_x, mean_y)) <= 1e-6:
        return None
    angle = 0.5 * float(np.arctan2(mean_y, mean_x))
    return np.asarray((np.cos(angle), np.sin(angle)), dtype=np.float32)


def _mask_convex_hull_points(mask: np.ndarray) -> np.ndarray | None:
    points = cv2.findNonZero((np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255)
    if points is None or len(points) < 3:
        return None
    return cv2.convexHull(points).reshape(-1, 2).astype(np.float32)


def _form_fit_sticker_area_mask(union_mask: np.ndarray) -> np.ndarray | None:
    binary = (np.asarray(union_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    bounds = _mask_bounds_xyxy(union_mask)
    if bounds is None:
        return None

    x1, y1, x2, y2 = bounds
    span = max(x2 - x1, y2 - y1)
    radius = max(4, int(round(span * 0.08)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    form = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    form = cv2.dilate(form, kernel, iterations=1)
    contours, _ = cv2.findContours(form, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    filled = np.zeros_like(binary)
    for contour in contours:
        if cv2.contourArea(contour) > 0:
            cv2.drawContours(filled, [contour], -1, 255, thickness=-1)
    return filled > 0 if bool(filled.any()) else None


def _normalized_sticker_masks(
    sticker_masks: list[np.ndarray],
    roi_shape: tuple[int, int],
) -> list[np.ndarray]:
    roi_height, roi_width = roi_shape
    normalized = []
    for mask in sticker_masks:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (roi_height, roi_width):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (roi_width, roi_height),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        if bool(mask.any()):
            normalized.append(mask)
    return normalized


def _sample_empty_form_points(
    union_mask: np.ndarray,
    form_mask: np.ndarray,
    sticker_masks: list[np.ndarray],
    *,
    sample_count: int,
) -> np.ndarray:
    if sample_count <= 0:
        return np.empty((0, 2), dtype=np.float32)

    empty_mask = _interior_gap_mask(union_mask, form_mask, sticker_masks)
    ys, xs = np.nonzero(empty_mask)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float32)

    seed = _mask_seed(empty_mask)
    rng = np.random.default_rng(seed)
    size = min(sample_count, len(xs))
    indexes = rng.choice(len(xs), size=size, replace=False)
    return np.stack([xs[indexes], ys[indexes]], axis=1).astype(np.float32)


def _interior_gap_mask(
    union_mask: np.ndarray,
    form_mask: np.ndarray,
    sticker_masks: list[np.ndarray],
) -> np.ndarray:
    empty_mask = form_mask & ~union_mask
    if not bool(empty_mask.any()):
        return empty_mask

    form_u8 = form_mask.astype(np.uint8) * 255
    union_u8 = union_mask.astype(np.uint8) * 255
    distance_from_form_edge = cv2.distanceTransform(form_u8, cv2.DIST_L2, 3)
    distance_from_stickers = cv2.distanceTransform(255 - union_u8, cv2.DIST_L2, 3)
    form_edge_margin = max(3.0, min(form_mask.shape[:2]) * 0.035)
    sticker_gap_reach = max(6.0, min(form_mask.shape[:2]) * 0.14)
    multi_sticker_nearby = _near_multiple_stickers_mask(
        sticker_masks,
        max_distance=sticker_gap_reach,
    )
    gap_mask = (
        empty_mask
        & (distance_from_form_edge >= form_edge_margin)
        & (distance_from_stickers <= sticker_gap_reach)
        & multi_sticker_nearby
    )
    if int(gap_mask.sum()) >= 8:
        return gap_mask

    fallback = (
        empty_mask
        & (distance_from_form_edge >= form_edge_margin)
        & (distance_from_stickers <= sticker_gap_reach)
    )
    return fallback if bool(fallback.any()) else empty_mask


def _near_multiple_stickers_mask(
    sticker_masks: list[np.ndarray],
    *,
    max_distance: float,
) -> np.ndarray:
    if not sticker_masks:
        return np.zeros((0, 0), dtype=bool)

    nearby_count = np.zeros(sticker_masks[0].shape, dtype=np.uint8)
    for mask in sticker_masks:
        source = np.where(mask, 0, 255).astype(np.uint8)
        distances = cv2.distanceTransform(source, cv2.DIST_L2, 3)
        nearby_count += (distances <= max_distance).astype(np.uint8)
    return nearby_count >= 2 if len(sticker_masks) >= 2 else nearby_count >= 1


def _mask_seed(mask: np.ndarray) -> int:
    ys, xs = np.nonzero(mask)
    return int(mask.sum() + ys.sum() * 31 + xs.sum() * 131) & 0xFFFFFFFF


def _infinite_line_from_points(
    start_xy: np.ndarray,
    end_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    start = np.asarray(start_xy, dtype=np.float32)
    end = np.asarray(end_xy, dtype=np.float32)
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length <= 1e-6:
        return None
    direction = vector / length
    normal = np.asarray((-direction[1], direction[0]), dtype=np.float32)
    return start, direction, normal


def _sample_segment_integer_points(start_xy: np.ndarray, end_xy: np.ndarray) -> np.ndarray:
    start = np.asarray(start_xy, dtype=np.float32)
    end = np.asarray(end_xy, dtype=np.float32)
    length = int(round(float(np.linalg.norm(end - start))))
    if length <= 0:
        return np.empty((0, 2), dtype=np.int32)
    t = np.linspace(0.0, 1.0, length + 1, dtype=np.float32)
    points = start[None, :] * (1.0 - t[:, None]) + end[None, :] * t[:, None]
    return np.round(points).astype(np.int32)


def _line_intersectionality_profile(
    line_points_xy: np.ndarray,
    *,
    union_mask: np.ndarray,
    edge_distance_image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(line_points_xy, dtype=np.int32).reshape(-1, 2)
    if len(points) == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=bool)

    height, width = union_mask.shape[:2]
    xs = np.clip(points[:, 0], 0, width - 1)
    ys = np.clip(points[:, 1], 0, height - 1)
    intersects = union_mask[ys, xs]
    edge_distances = edge_distance_image[ys, xs]
    normalized_edge_distance = np.minimum(edge_distances / 9.0, 1.0)
    scores = np.where(
        intersects,
        0.35 + 0.65 * normalized_edge_distance,
        0.04 + 0.32 * normalized_edge_distance,
    )
    return scores.astype(np.float32), intersects.astype(bool)


def _best_low_intersectionality_region_score(point_scores: np.ndarray) -> float:
    scores = np.asarray(point_scores, dtype=np.float32).reshape(-1)
    if len(scores) == 0:
        return 1.0

    window = min(len(scores), max(8, int(round(len(scores) * 0.35))))
    if window == len(scores):
        return float(scores.mean())
    cumsum = np.concatenate([[0.0], np.cumsum(scores, dtype=np.float32)])
    means = (cumsum[window:] - cumsum[:-window]) / float(window)
    return float(np.min(means))


def _clip_infinite_line_to_convex_polygon(
    point_xy: np.ndarray,
    direction_xy: np.ndarray,
    polygon_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    direction = np.asarray(direction_xy, dtype=np.float32)
    point = np.asarray(point_xy, dtype=np.float32)
    polygon = np.asarray(polygon_xy, dtype=np.float32).reshape(-1, 2)
    if len(polygon) < 3:
        return None

    intersections = []
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = end - start
        denominator = _cross_2d(direction, edge)
        if abs(denominator) <= 1e-6:
            continue
        delta = start - point
        line_t = _cross_2d(delta, edge) / denominator
        edge_t = _cross_2d(delta, direction) / denominator
        if -1e-4 <= edge_t <= 1.0 + 1e-4:
            intersections.append(point + direction * line_t)

    unique = []
    for intersection in intersections:
        if not any(float(np.linalg.norm(intersection - existing)) < 1e-3 for existing in unique):
            unique.append(intersection)
    if len(unique) < 2:
        return None
    unique.sort(key=lambda value: float(value @ direction))
    return unique[0], unique[-1]


def _clip_infinite_line_to_mask(
    point_xy: np.ndarray,
    direction_xy: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    direction = np.asarray(direction_xy, dtype=np.float32)
    point = np.asarray(point_xy, dtype=np.float32)
    height, width = mask.shape[:2]
    span = float(np.hypot(width, height) * 1.5)
    sample_count = max(8, int(round(span * 2.0)))
    t_values = np.linspace(-span, span, sample_count, dtype=np.float32)
    points = point[None, :] + t_values[:, None] * direction[None, :]
    xs = np.round(points[:, 0]).astype(np.int32)
    ys = np.round(points[:, 1]).astype(np.int32)
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    inside = np.zeros(len(points), dtype=bool)
    inside[valid] = mask[ys[valid], xs[valid]]

    runs = []
    start_index: int | None = None
    for index, is_inside in enumerate(inside):
        if is_inside and start_index is None:
            start_index = index
        elif not is_inside and start_index is not None:
            runs.append((start_index, index - 1))
            start_index = None
    if start_index is not None:
        runs.append((start_index, len(inside) - 1))
    if not runs:
        return None

    best_start, best_end = max(runs, key=lambda run: run[1] - run[0])
    if best_end <= best_start:
        return None
    return points[best_start].astype(np.float32), points[best_end].astype(np.float32)


def _clip_infinite_line_to_roi(
    point_xy: np.ndarray,
    direction_xy: np.ndarray,
    roi_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray] | None:
    roi_height, roi_width = roi_shape
    rect = np.asarray(
        [
            [0.0, 0.0],
            [float(roi_width - 1), 0.0],
            [float(roi_width - 1), float(roi_height - 1)],
            [0.0, float(roi_height - 1)],
        ],
        dtype=np.float32,
    )
    segment = _clip_infinite_line_to_convex_polygon(point_xy, direction_xy, rect)
    if segment is None:
        return None
    start, end = segment
    start = np.asarray(
        (
            np.clip(start[0], 0.0, float(roi_width - 1)),
            np.clip(start[1], 0.0, float(roi_height - 1)),
        ),
        dtype=np.float32,
    )
    end = np.asarray(
        (
            np.clip(end[0], 0.0, float(roi_width - 1)),
            np.clip(end[1], 0.0, float(roi_height - 1)),
        ),
        dtype=np.float32,
    )
    return start, end


def _cross_2d(left: np.ndarray, right: np.ndarray) -> float:
    return float(left[0] * right[1] - left[1] * right[0])


def _diverse_low_intersection_lines(
    candidates: list[LowIntersectionLine],
    *,
    max_lines: int,
) -> list[LowIntersectionLine]:
    selected: list[LowIntersectionLine] = []
    for candidate in candidates:
        if any(
            abs(candidate.angle_degrees - selected_line.angle_degrees) < 1e-6
            and _segment_midpoint_distance(candidate, selected_line) < 8.0
            for selected_line in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= max_lines:
            break
    return selected


def _segment_midpoint_distance(left: LowIntersectionLine, right: LowIntersectionLine) -> float:
    left_midpoint = (
        np.asarray(left.start_xy, dtype=np.float32)
        + np.asarray(left.end_xy, dtype=np.float32)
    ) * 0.5
    right_midpoint = (
        np.asarray(right.start_xy, dtype=np.float32)
        + np.asarray(right.end_xy, dtype=np.float32)
    ) * 0.5
    return float(np.linalg.norm(left_midpoint - right_midpoint))


def draw_low_intersection_lines(
    frame: np.ndarray,
    overlay: EmptySpaceLineOverlay | list[LowIntersectionLine],
    *,
    offset_xy: tuple[int, int],
) -> None:
    offset = np.asarray(offset_xy, dtype=np.float32)
    if isinstance(overlay, EmptySpaceLineOverlay):
        sampled_points = overlay.sampled_points_xy
        lines = overlay.lines
        line_width_px = overlay.line_width_px
        quad_edge_segments = overlay.quad_edge_segments_xy
        basis_vectors = overlay.basis_vectors_xy
        basis_center = overlay.basis_center_xy
    else:
        sampled_points = np.empty((0, 2), dtype=np.float32)
        lines = overlay
        line_width_px = EMPTY_SPACE_LINE_WIDTH_PX
        quad_edge_segments = np.empty((0, 2, 2), dtype=np.float32)
        basis_vectors = np.empty((0, 2), dtype=np.float32)
        basis_center = None

    for segment in np.asarray(quad_edge_segments, dtype=np.float32).reshape(-1, 2, 2):
        start = segment[0] + offset
        end = segment[1] + offset
        start_xy = (int(round(start[0])), int(round(start[1])))
        end_xy = (int(round(end[0])), int(round(end[1])))
        cv2.line(
            frame,
            start_xy,
            end_xy,
            (0, 0, 0),
            4,
            lineType=cv2.LINE_AA,
        )
        cv2.line(
            frame,
            start_xy,
            end_xy,
            (255, 0, 255),
            2,
            lineType=cv2.LINE_AA,
        )

    for index, line in enumerate(lines):
        start = np.asarray(line.start_xy, dtype=np.float32) + offset
        end = np.asarray(line.end_xy, dtype=np.float32) + offset
        color = (0, 255, 255) if index < 8 else (255, 255, 255)
        thickness = max(1, int(round(float(line_width_px))))
        cv2.line(
            frame,
            (int(round(start[0])), int(round(start[1]))),
            (int(round(end[0])), int(round(end[1]))),
            (0, 0, 0),
            thickness + 2,
            lineType=cv2.LINE_AA,
        )
        cv2.line(
            frame,
            (int(round(start[0])), int(round(start[1]))),
            (int(round(end[0])), int(round(end[1]))),
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )

    for point in np.asarray(sampled_points, dtype=np.float32).reshape(-1, 2):
        xy = point + offset
        center = (int(round(xy[0])), int(round(xy[1])))
        cv2.circle(frame, center, 6, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, center, 4, (255, 0, 255), -1, lineType=cv2.LINE_AA)

    if basis_center is not None:
        center = np.asarray(basis_center, dtype=np.float32) + offset
        vector_length = max(18.0, min(frame.shape[:2]) * 0.10)
        for vector, color in zip(
            np.asarray(basis_vectors, dtype=np.float32).reshape(-1, 2),
            [(255, 0, 0), (0, 0, 255)],
        ):
            norm = float(np.linalg.norm(vector))
            if norm <= 1e-6:
                continue
            unit = vector / norm
            start = center - unit * vector_length
            end = center + unit * vector_length
            start_xy = (int(round(start[0])), int(round(start[1])))
            end_xy = (int(round(end[0])), int(round(end[1])))
            cv2.line(frame, start_xy, end_xy, (0, 0, 0), 6, lineType=cv2.LINE_AA)
            cv2.line(frame, start_xy, end_xy, color, 4, lineType=cv2.LINE_AA)

    if len(sampled_points) > 0 or lines:
        label_origin = (int(offset[0]) + 6, int(offset[1]) + 18)
        label = f"quad pts {len(sampled_points)} lines {len(lines)}"
        cv2.putText(
            frame,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def fit_projected_cube_grid(
    sticker_masks: list[np.ndarray],
    *,
    roi_shape: tuple[int, int],
    line_thickness_px: int = 2,
    minimum_coverage: float = 0.88,
) -> ProjectedCubeGridFit | None:
    union_mask = _union_sticker_mask(sticker_masks, roi_shape)
    if union_mask is None:
        return None

    bounds = _mask_bounds_xyxy(union_mask)
    if bounds is None:
        return None

    mask_area = max(float(union_mask.sum()), 1.0)
    mask_points = _sample_mask_xy_points(
        union_mask,
        max_points=CUBE_GRID_FIT_SAMPLE_POINTS,
    )
    edge_mask = _sticker_edge_mask(
        sticker_masks,
        roi_shape=roi_shape,
    )
    if len(mask_points) == 0 or edge_mask is None:
        return None
    edge_distance_image = _distance_image_from_edge_mask(edge_mask)
    edge_directions = _dominant_edge_direction_angles(edge_mask)
    candidate_axes = _projected_cube_axis_candidates_for_edge_directions(edge_directions)

    coarse_fits: list[tuple[float, np.ndarray, np.ndarray, list[tuple[tuple[float, float], tuple[float, float]]], np.ndarray]] = []
    for axes in candidate_axes:
        base_hull, base_lines = _projected_cube_grid_geometry(axes)
        base_min = base_hull.min(axis=0)
        base_max = base_hull.max(axis=0)
        base_size = np.maximum(base_max - base_min, 1e-6)
        bx1, by1, bx2, by2 = bounds
        target_size = np.asarray((bx2 - bx1, by2 - by1), dtype=np.float32) * 1.08
        base_scale = float(np.max(target_size / base_size))
        base_center = (base_min + base_max) / 2.0
        target_center = np.asarray(((bx1 + bx2) / 2.0, (by1 + by2) / 2.0), dtype=np.float32)
        for scale_multiplier in (1.0, 1.15):
            scale = base_scale * scale_multiplier
            hull, lines = _transform_projected_cube_grid(
                base_hull,
                base_lines,
                base_center=base_center,
                scale=scale,
                origin=target_center,
            )
            fit = _score_projected_cube_grid_from_edge_distance(
                edge_distance_image,
                coverage_points_xy=mask_points,
                mask_area=mask_area,
                hull_points_xy=hull,
                line_segments_xy=lines,
                line_thickness_px=line_thickness_px,
                minimum_coverage=minimum_coverage,
            )
            coarse_fits.append((fit.score, base_hull, base_center, base_lines, target_center))

    best_fit: ProjectedCubeGridFit | None = None
    offset_step = max(float(max(target_size)) * 0.045, 1.0)
    offset_candidates = (
        (0.0, 0.0),
        (-offset_step, 0.0),
        (offset_step, 0.0),
        (0.0, -offset_step),
        (0.0, offset_step),
    )
    for _, base_hull, base_center, base_lines, target_center in sorted(
        coarse_fits,
        key=lambda item: item[0],
    )[:CUBE_GRID_FIT_REFINE_CANDIDATES]:
        base_min = base_hull.min(axis=0)
        base_max = base_hull.max(axis=0)
        base_size = np.maximum(base_max - base_min, 1e-6)
        base_scale = float(np.max(target_size / base_size))
        for scale_multiplier in (0.96, 1.08, 1.20):
            scale = base_scale * scale_multiplier
            for offset_x, offset_y in offset_candidates:
                origin = target_center + np.asarray((offset_x, offset_y), dtype=np.float32)
                hull, lines = _transform_projected_cube_grid(
                    base_hull,
                    base_lines,
                    base_center=base_center,
                    scale=scale,
                    origin=origin,
                )
                fit = _score_projected_cube_grid_from_edge_distance(
                    edge_distance_image,
                    coverage_points_xy=mask_points,
                    mask_area=mask_area,
                    hull_points_xy=hull,
                    line_segments_xy=lines,
                    line_thickness_px=line_thickness_px,
                    minimum_coverage=minimum_coverage,
                )
                if best_fit is None or fit.score < best_fit.score:
                    best_fit = fit
    return best_fit


def _transform_projected_cube_grid(
    base_hull: np.ndarray,
    base_lines: list[tuple[tuple[float, float], tuple[float, float]]],
    *,
    base_center: np.ndarray,
    scale: float,
    origin: np.ndarray,
) -> tuple[np.ndarray, list[tuple[tuple[float, float], tuple[float, float]]]]:
    hull = (base_hull - base_center) * scale + origin
    lines = [
        (
            tuple(((np.asarray(start) - base_center) * scale + origin).tolist()),
            tuple(((np.asarray(end) - base_center) * scale + origin).tolist()),
        )
        for start, end in base_lines
    ]
    return hull, lines


def _sample_mask_xy_points(mask: np.ndarray, *, max_points: int) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float32)

    points = np.stack([xs, ys], axis=1).astype(np.float32)
    if len(points) <= max_points:
        return points

    indexes = np.linspace(0, len(points) - 1, max_points, dtype=np.int32)
    return points[indexes]


def _sample_sticker_edge_xy_points(
    sticker_masks: list[np.ndarray],
    *,
    roi_shape: tuple[int, int],
    max_points: int,
) -> np.ndarray:
    del max_points
    return _sticker_quad_geometry(sticker_masks, roi_shape=roi_shape).points_xy


def _sample_sticker_edge_points(
    sticker_masks: list[np.ndarray],
    *,
    roi_shape: tuple[int, int],
    max_points: int,
) -> StickerEdgePointSample:
    del max_points
    geometry = _sticker_quad_geometry(sticker_masks, roi_shape=roi_shape)
    return StickerEdgePointSample(
        points_xy=geometry.points_xy,
        sticker_indexes=geometry.sticker_indexes,
    )


def _sticker_quad_geometry(
    sticker_masks: list[np.ndarray],
    *,
    roi_shape: tuple[int, int],
) -> StickerQuadGeometry:
    roi_height, roi_width = roi_shape
    point_batches = []
    sticker_index_batches = []
    edge_batches = []
    for sticker_index, mask in enumerate(sticker_masks):
        quad = _sticker_mask_quad_points(mask, roi_shape=roi_shape)
        if quad is None:
            continue
        point_batches.append(quad)
        sticker_index_batches.append(
            np.full(len(quad), sticker_index, dtype=np.int32)
        )
        edge_batches.append(np.stack([quad, np.roll(quad, -1, axis=0)], axis=1))

    if not point_batches:
        return StickerQuadGeometry(
            points_xy=np.empty((0, 2), dtype=np.float32),
            sticker_indexes=np.empty(0, dtype=np.int32),
            edge_segments_xy=np.empty((0, 2, 2), dtype=np.float32),
        )

    return StickerQuadGeometry(
        points_xy=np.concatenate(point_batches, axis=0).astype(np.float32),
        sticker_indexes=np.concatenate(sticker_index_batches, axis=0),
        edge_segments_xy=np.concatenate(edge_batches, axis=0).astype(np.float32),
    )


def _sticker_mask_quad_points(
    mask: np.ndarray,
    *,
    roi_shape: tuple[int, int],
) -> np.ndarray | None:
    roi_height, roi_width = roi_shape
    normalized = np.asarray(mask, dtype=bool)
    if normalized.shape != (roi_height, roi_width):
        normalized = cv2.resize(
            normalized.astype(np.uint8),
            (roi_width, roi_height),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
    points = cv2.findNonZero(normalized.astype(np.uint8) * 255)
    if points is None or len(points) < 3:
        return None

    binary = normalized.astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, closed=True)
    if perimeter <= 1e-6:
        return None

    for epsilon_ratio in np.linspace(0.005, 0.12, 32):
        approx = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, closed=True)
        if len(approx) == 4:
            return _order_quad_points_clockwise(approx.reshape(-1, 2).astype(np.float32))

    hull = cv2.convexHull(contour).reshape(-1, 2).astype(np.float32)
    if len(hull) == 4:
        return _order_quad_points_clockwise(hull)
    if len(hull) > 4:
        return _max_area_quad_from_hull_points(hull)
    return None


def _face_parallelogram_points(
    mask: np.ndarray,
    *,
    roi_shape: tuple[int, int],
) -> np.ndarray | None:
    quad = _sticker_mask_quad_points(mask, roi_shape=roi_shape)
    if quad is None:
        return None
    return _parallelogram_from_quad_points(quad)


def _parallelogram_from_quad_points(points_xy: np.ndarray) -> np.ndarray:
    points = _order_quad_points_clockwise(points_xy)
    center = points.mean(axis=0)
    horizontal = ((points[1] + points[2]) - (points[0] + points[3])) / 4.0
    vertical = ((points[2] + points[3]) - (points[0] + points[1])) / 4.0
    return np.asarray(
        [
            center - horizontal - vertical,
            center + horizontal - vertical,
            center + horizontal + vertical,
            center - horizontal + vertical,
        ],
        dtype=np.float32,
    )


def _reconcile_shared_face_parallelogram_sides(
    parallelograms: list[np.ndarray],
    *,
    roi_shape: tuple[int, int],
) -> list[np.ndarray]:
    reconciled = [np.asarray(quad, dtype=np.float32).reshape(4, 2).copy() for quad in parallelograms]
    if len(reconciled) < 2:
        return reconciled

    max_shared_edge_distance = max(10.0, min(roi_shape) * 0.12)
    for left_index, right_index in combinations(range(len(reconciled)), 2):
        match = _closest_compatible_quad_edges(
            reconciled[left_index],
            reconciled[right_index],
            max_distance=max_shared_edge_distance,
        )
        if match is None:
            continue

        left_edge_index, right_edge_index, right_reversed = match
        left_start, left_end = _quad_edge_points(reconciled[left_index], left_edge_index)
        right_start, right_end = _quad_edge_points(reconciled[right_index], right_edge_index)
        if right_reversed:
            right_start, right_end = right_end, right_start

        shared_start = (left_start + right_start) / 2.0
        shared_end = (left_end + right_end) / 2.0
        _set_parallelogram_edge_points(reconciled[left_index], left_edge_index, shared_start, shared_end)
        if right_reversed:
            _set_parallelogram_edge_points(reconciled[right_index], right_edge_index, shared_end, shared_start)
        else:
            _set_parallelogram_edge_points(reconciled[right_index], right_edge_index, shared_start, shared_end)
    return reconciled


def _closest_compatible_quad_edges(
    left_quad: np.ndarray,
    right_quad: np.ndarray,
    *,
    max_distance: float,
) -> tuple[int, int, bool] | None:
    best: tuple[float, int, int, bool] | None = None
    for left_edge_index in range(4):
        left_start, left_end = _quad_edge_points(left_quad, left_edge_index)
        left_vector = left_end - left_start
        left_length = float(np.linalg.norm(left_vector))
        if left_length <= 1e-6:
            continue
        left_unit = left_vector / left_length
        for right_edge_index in range(4):
            right_start, right_end = _quad_edge_points(right_quad, right_edge_index)
            right_vector = right_end - right_start
            right_length = float(np.linalg.norm(right_vector))
            if right_length <= 1e-6:
                continue
            right_unit = right_vector / right_length
            if abs(float(left_unit @ right_unit)) < 0.78:
                continue

            same = float(np.linalg.norm(left_start - right_start) + np.linalg.norm(left_end - right_end))
            reversed_distance = float(np.linalg.norm(left_start - right_end) + np.linalg.norm(left_end - right_start))
            distance = min(same, reversed_distance) / 2.0
            if distance > max_distance:
                continue
            right_reversed = reversed_distance < same
            if best is None or distance < best[0]:
                best = (distance, left_edge_index, right_edge_index, right_reversed)

    if best is None:
        return None
    _distance, left_edge_index, right_edge_index, right_reversed = best
    return left_edge_index, right_edge_index, right_reversed


def _quad_edge_points(quad: np.ndarray, edge_index: int) -> tuple[np.ndarray, np.ndarray]:
    return quad[edge_index], quad[(edge_index + 1) % 4]


def _quad_edge_endpoint_indexes(edge_index: int) -> tuple[int, int]:
    return edge_index, (edge_index + 1) % 4


def _set_parallelogram_edge_points(
    quad: np.ndarray,
    edge_index: int,
    start_xy: np.ndarray,
    end_xy: np.ndarray,
) -> None:
    original = np.asarray(quad, dtype=np.float32).reshape(4, 2).copy()
    start_index = edge_index
    end_index = (edge_index + 1) % 4
    opposite_end_index = (edge_index + 2) % 4
    opposite_start_index = (edge_index - 1) % 4

    start = np.asarray(start_xy, dtype=np.float32)
    end = np.asarray(end_xy, dtype=np.float32)
    opposite_offset = (
        (original[opposite_start_index] - original[start_index])
        + (original[opposite_end_index] - original[end_index])
    ) / 2.0

    quad[start_index] = start
    quad[end_index] = end
    quad[opposite_start_index] = start + opposite_offset
    quad[opposite_end_index] = end + opposite_offset


def face_quad_fits_for_masks(
    face_masks: list[np.ndarray],
    *,
    roi_shape: tuple[int, int],
) -> list[tuple[np.ndarray, np.ndarray]]:
    fitted = [
        (np.asarray(mask, dtype=bool), parallelogram)
        for mask in face_masks
        if (parallelogram := _face_parallelogram_points(mask, roi_shape=roi_shape)) is not None
    ]
    if not fitted:
        return []

    masks = [mask for mask, _quad in fitted]
    quads = _reconcile_shared_face_parallelogram_sides(
        [quad for _mask, quad in fitted],
        roi_shape=roi_shape,
    )
    quads = _adjust_face_triplet_quads_for_shared_corner_and_mask_fit(
        masks,
        quads,
        roi_shape=roi_shape,
    )
    return list(zip(masks, quads))


def _adjust_face_triplet_quads_for_shared_corner_and_mask_fit(
    face_masks: list[np.ndarray],
    quads: list[np.ndarray],
    *,
    roi_shape: tuple[int, int],
) -> list[np.ndarray]:
    adjusted = [np.asarray(quad, dtype=np.float32).reshape(4, 2).copy() for quad in quads]
    if len(adjusted) != MAX_FACE_COUNT:
        return adjusted

    max_shared_edge_distance = max(10.0, min(roi_shape) * 0.12)
    edge_indexes_by_face: list[list[int]] = [[] for _ in range(MAX_FACE_COUNT)]
    matches_by_pair = {}
    for left_index, right_index in combinations(range(MAX_FACE_COUNT), 2):
        match = _closest_compatible_quad_edges(
            adjusted[left_index],
            adjusted[right_index],
            max_distance=max_shared_edge_distance,
        )
        if match is None:
            return adjusted
        left_edge_index, right_edge_index, right_reversed = match
        edge_indexes_by_face[left_index].append(left_edge_index)
        edge_indexes_by_face[right_index].append(right_edge_index)
        matches_by_pair[(left_index, right_index)] = (left_edge_index, right_edge_index, right_reversed)

    common_corner_indexes = []
    for face_index, edge_indexes in enumerate(edge_indexes_by_face):
        if len(edge_indexes) != 2:
            return adjusted
        corner_index = _quad_common_corner_index_for_adjacent_edges(edge_indexes[0], edge_indexes[1])
        if corner_index is None:
            return adjusted
        common_corner_indexes.append(corner_index)

    shared_center = np.mean(
        np.asarray(
            [
                adjusted[face_index][corner_index]
                for face_index, corner_index in enumerate(common_corner_indexes)
            ],
            dtype=np.float32,
        ),
        axis=0,
    )
    for face_index, corner_index in enumerate(common_corner_indexes):
        adjusted[face_index][corner_index] = shared_center

    adjusted = _fit_projected_cube_quads_from_shared_sides(
        adjusted,
        face_masks,
        edge_indexes_by_face,
        matches_by_pair,
        common_corner_indexes,
        shared_center,
        roi_shape=roi_shape,
    )

    return _slide_triplet_outer_quad_corners_with_shared_scale(
        adjusted,
        face_masks,
        common_corner_indexes,
        roi_shape=roi_shape,
    )


def _fit_projected_cube_quads_from_shared_sides(
    quads: list[np.ndarray],
    face_masks: list[np.ndarray],
    edge_indexes_by_face: list[list[int]],
    matches_by_pair: dict[tuple[int, int], tuple[int, int, bool]],
    common_corner_indexes: list[int],
    shared_center: np.ndarray,
    *,
    roi_shape: tuple[int, int],
) -> list[np.ndarray]:
    axis_by_face_edge: list[dict[int, np.ndarray]] = [dict() for _ in quads]
    axis_vectors = []
    for (left_index, right_index), match in matches_by_pair.items():
        left_edge_index, right_edge_index, _right_reversed = match
        left_far = _quad_edge_far_point_from_common_corner(
            quads[left_index],
            left_edge_index,
            common_corner_indexes[left_index],
        )
        right_far = _quad_edge_far_point_from_common_corner(
            quads[right_index],
            right_edge_index,
            common_corner_indexes[right_index],
        )
        if left_far is None or right_far is None:
            return quads
        far_point = (left_far + right_far) / 2.0
        vector = far_point - shared_center
        length = float(np.linalg.norm(vector))
        if length <= 1e-6:
            return quads
        axis_vectors.append(vector)
        axis_by_face_edge[left_index][left_edge_index] = vector
        axis_by_face_edge[right_index][right_edge_index] = vector

    if len(axis_vectors) != MAX_FACE_COUNT:
        return quads

    initial_scale = 1.0
    min_scale = 0.65
    max_scale = 1.35
    best_quads = quads
    best_score = float("-inf")
    for scale in np.linspace(min_scale, max_scale, 57):
        candidate_quads = _projected_cube_quads_for_scale(
            edge_indexes_by_face,
            axis_by_face_edge,
            common_corner_indexes,
            shared_center,
            float(scale),
            fallback_quads=quads,
        )
        score = _orthogonal_quad_mask_fit_score(
            candidate_quads,
            face_masks,
            roi_shape=roi_shape,
        )
        if score > best_score or (
            score == best_score
            and abs(float(scale) - initial_scale) < abs(_projected_cube_scale_from_reference(best_quads, quads) - initial_scale)
        ):
            best_score = score
            best_quads = candidate_quads
    return best_quads


def _projected_cube_quads_for_scale(
    edge_indexes_by_face: list[list[int]],
    axis_by_face_edge: list[dict[int, np.ndarray]],
    common_corner_indexes: list[int],
    shared_center: np.ndarray,
    scale: float,
    *,
    fallback_quads: list[np.ndarray],
) -> list[np.ndarray]:
    fitted = []
    for face_index, edge_indexes in enumerate(edge_indexes_by_face):
        if len(edge_indexes) != 2:
            return fallback_quads
        first_edge, second_edge = edge_indexes
        first_axis = axis_by_face_edge[face_index].get(first_edge)
        second_axis = axis_by_face_edge[face_index].get(second_edge)
        if first_axis is None or second_axis is None:
            return fallback_quads
        quad = _quad_from_common_corner_axes(
            common_corner_index=common_corner_indexes[face_index],
            first_edge_index=first_edge,
            first_axis=first_axis * scale,
            second_edge_index=second_edge,
            second_axis=second_axis * scale,
            shared_center=shared_center,
        )
        fitted.append(quad)
    return fitted


def _quad_from_common_corner_axes(
    *,
    common_corner_index: int,
    first_edge_index: int,
    first_axis: np.ndarray,
    second_edge_index: int,
    second_axis: np.ndarray,
    shared_center: np.ndarray,
) -> np.ndarray:
    quad = np.zeros((4, 2), dtype=np.float32)
    first_far_index = _quad_edge_far_index_from_common_corner(first_edge_index, common_corner_index)
    second_far_index = _quad_edge_far_index_from_common_corner(second_edge_index, common_corner_index)
    if first_far_index is None or second_far_index is None:
        return quad
    outer_index = (common_corner_index + 2) % 4
    quad[common_corner_index] = shared_center
    quad[first_far_index] = shared_center + first_axis
    quad[second_far_index] = shared_center + second_axis
    quad[outer_index] = shared_center + first_axis + second_axis
    return quad


def _quad_edge_far_point_from_common_corner(
    quad: np.ndarray,
    edge_index: int,
    common_corner_index: int,
) -> np.ndarray | None:
    far_index = _quad_edge_far_index_from_common_corner(edge_index, common_corner_index)
    if far_index is None:
        return None
    return quad[far_index]


def _projected_cube_scale_from_reference(
    candidate_quads: list[np.ndarray],
    reference_quads: list[np.ndarray],
) -> float:
    candidate_lengths = [
        _quad_edge_length(quad, edge_index)
        for quad in candidate_quads
        for edge_index in range(4)
    ]
    reference_lengths = [
        _quad_edge_length(quad, edge_index)
        for quad in reference_quads
        for edge_index in range(4)
    ]
    if not candidate_lengths or not reference_lengths:
        return 1.0
    reference_mean = float(np.mean(reference_lengths))
    if reference_mean <= 1e-6:
        return 1.0
    return float(np.mean(candidate_lengths) / reference_mean)


def _quad_edge_far_index_from_common_corner(edge_index: int, common_corner_index: int) -> int | None:
    start_index, end_index = _quad_edge_endpoint_indexes(edge_index)
    if start_index == common_corner_index:
        return end_index
    if end_index == common_corner_index:
        return start_index
    return None

def _slide_outer_quad_corner_toward_center_for_mask_fit(
    quad: np.ndarray,
    face_mask: np.ndarray,
    *,
    common_index: int,
    outer_index: int,
    roi_shape: tuple[int, int],
) -> np.ndarray:
    quad = np.asarray(quad, dtype=np.float32).reshape(4, 2).copy()
    center = quad[common_index].copy()
    original_outer = quad[outer_index].copy()
    vector = original_outer - center
    if float(np.linalg.norm(vector)) <= 1e-6:
        return quad

    best_quad = quad.copy()
    best_score = _quad_mask_fit_score(best_quad, face_mask, roi_shape=roi_shape)
    for scale in _outer_corner_slide_scales():
        candidate = _quad_with_outer_corner_slide(
            quad,
            common_index=common_index,
            outer_index=outer_index,
            scale=float(scale),
        )
        score = _quad_mask_fit_score(candidate, face_mask, roi_shape=roi_shape)
        if score > best_score:
            best_score = score
            best_quad = candidate
    return best_quad


def _slide_triplet_outer_quad_corners_with_shared_scale(
    quads: list[np.ndarray],
    face_masks: list[np.ndarray],
    common_corner_indexes: list[int],
    *,
    roi_shape: tuple[int, int],
) -> list[np.ndarray]:
    best_quads = [np.asarray(quad, dtype=np.float32).reshape(4, 2).copy() for quad in quads]
    best_score = _triplet_quad_mask_fit_score(
        best_quads,
        face_masks,
        roi_shape=roi_shape,
    )
    for scale in _outer_corner_slide_scales():
        candidate_quads = [
            _quad_with_outer_corner_slide(
                quad,
                common_index=common_index,
                outer_index=(common_index + 2) % 4,
                scale=float(scale),
            )
            for quad, common_index in zip(quads, common_corner_indexes)
        ]
        score = _triplet_quad_mask_fit_score(
            candidate_quads,
            face_masks,
            roi_shape=roi_shape,
        )
        if score > best_score:
            best_score = score
            best_quads = candidate_quads
    return best_quads


def _triplet_quad_mask_fit_score(
    quads: list[np.ndarray],
    face_masks: list[np.ndarray],
    *,
    roi_shape: tuple[int, int],
) -> float:
    scores = [
        _quad_mask_fit_score(quad, mask, roi_shape=roi_shape)
        for quad, mask in zip(quads, face_masks)
    ]
    if any(not np.isfinite(score) for score in scores):
        return float("-inf")
    return float(sum(scores))


def _orthogonal_quad_mask_fit_score(
    quads: list[np.ndarray],
    face_masks: list[np.ndarray],
    *,
    roi_shape: tuple[int, int],
) -> float:
    face_scores = [
        _quad_soft_fit_score(quad, mask, roi_shape=roi_shape)
        for quad, mask in zip(quads, face_masks)
    ]
    if not face_scores:
        return float("-inf")
    return float(max(face_scores) * 10.0 + sum(face_scores) * 0.1)


def _quad_soft_fit_score(
    quad: np.ndarray,
    face_mask: np.ndarray,
    *,
    roi_shape: tuple[int, int],
) -> float:
    polygon_mask = _quad_polygon_mask(quad, roi_shape=roi_shape)
    mask = _normalized_bool_mask(face_mask, shape=polygon_mask.shape)
    overlap = int(np.logical_and(polygon_mask, mask).sum())
    spillover = int(np.logical_and(polygon_mask, ~mask).sum())
    missed = int(np.logical_and(mask, ~polygon_mask).sum())
    mask_area = max(1, int(mask.sum()))
    polygon_area = max(1, int(polygon_mask.sum()))
    coverage = float(overlap) / float(mask_area)
    spillover_ratio = float(spillover) / float(polygon_area)
    missed_ratio = float(missed) / float(mask_area)
    return float(coverage * 1000.0 - spillover_ratio * 300.0 - missed_ratio * 900.0)


def _quad_edge_length(quad: np.ndarray, edge_index: int) -> float:
    start, end = _quad_edge_points(quad, edge_index)
    return float(np.linalg.norm(end - start))


def _quad_with_outer_corner_slide(
    quad: np.ndarray,
    *,
    common_index: int,
    outer_index: int,
    scale: float,
) -> np.ndarray:
    candidate = np.asarray(quad, dtype=np.float32).reshape(4, 2).copy()
    center = candidate[common_index].copy()
    vector = candidate[outer_index] - center
    candidate[outer_index] = center + vector * float(scale)
    return candidate


def _outer_corner_slide_scales() -> np.ndarray:
    return np.linspace(
        FACE_OUTER_CORNER_MIN_SLIDE_SCALE,
        1.0,
        FACE_OUTER_CORNER_SLIDE_STEPS,
    )


def _quad_mask_fit_score(
    quad: np.ndarray,
    face_mask: np.ndarray,
    *,
    roi_shape: tuple[int, int],
) -> float:
    polygon_mask = _quad_polygon_mask(quad, roi_shape=roi_shape)
    mask = _normalized_bool_mask(face_mask, shape=polygon_mask.shape)
    missed = int(np.logical_and(mask, ~polygon_mask).sum())
    if missed > 0:
        return float("-inf")
    overlap = int(np.logical_and(polygon_mask, mask).sum())
    spillover = int(np.logical_and(polygon_mask, ~mask).sum())
    return float(overlap - spillover * 3.0)


def _quad_fully_covers_mask(
    quad: np.ndarray,
    face_mask: np.ndarray,
    *,
    roi_shape: tuple[int, int],
) -> bool:
    polygon_mask = _quad_polygon_mask(quad, roi_shape=roi_shape)
    mask = _normalized_bool_mask(face_mask, shape=polygon_mask.shape)
    return not bool(np.logical_and(mask, ~polygon_mask).any())


def _normalized_bool_mask(mask: np.ndarray, *, shape: tuple[int, int]) -> np.ndarray:
    normalized = np.asarray(mask, dtype=bool)
    if normalized.shape != shape:
        normalized = cv2.resize(
            normalized.astype(np.uint8),
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
    return normalized


def _quad_polygon_mask(
    quad: np.ndarray,
    *,
    roi_shape: tuple[int, int],
) -> np.ndarray:
    roi_height, roi_width = roi_shape
    polygon_mask = np.zeros((roi_height, roi_width), dtype=np.uint8)
    points = np.round(np.asarray(quad, dtype=np.float32).reshape(4, 2)).astype(np.int32)
    cv2.fillConvexPoly(polygon_mask, points, 1)
    return polygon_mask > 0


def draw_face_parallelogram_hulls(
    frame: np.ndarray,
    face_masks: list[np.ndarray],
    *,
    roi_shape: tuple[int, int],
    offset_xy: tuple[int, int],
) -> None:
    fitted = face_quad_fits_for_masks(face_masks, roi_shape=roi_shape)
    offset = np.asarray(offset_xy, dtype=np.float32)
    for index, (_mask, quad) in enumerate(fitted):
        points = np.round(quad + offset).astype(np.int32).reshape(-1, 1, 2)
        color = FACE_PARALLELOGRAM_COLORS[index % len(FACE_PARALLELOGRAM_COLORS)]
        cv2.polylines(frame, [points], isClosed=True, color=(0, 0, 0), thickness=5, lineType=cv2.LINE_AA)
        cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)


def face_parallelograms_for_masks(
    face_masks: list[np.ndarray],
    *,
    roi_shape: tuple[int, int],
) -> list[np.ndarray]:
    parallelograms = [
        parallelogram
        for mask in face_masks
        if (parallelogram := _face_parallelogram_points(mask, roi_shape=roi_shape)) is not None
    ]
    return _reconcile_shared_face_parallelogram_sides(parallelograms, roi_shape=roi_shape)


def covered_face_mask_indexes(
    indexed_face_masks: list[tuple[int, np.ndarray]],
    *,
    coverage_threshold: float = FACE_COVERAGE_PRUNE_THRESHOLD,
) -> set[int]:
    face_areas = [
        (index, np.asarray(mask, dtype=bool), int(np.asarray(mask, dtype=bool).sum()))
        for index, mask in indexed_face_masks
    ]
    pruned_indexes: set[int] = set()
    for index, mask, area in face_areas:
        if area <= 0:
            pruned_indexes.add(index)
            continue
        for other_index, other_mask, other_area in face_areas:
            if other_index == index or other_area <= area:
                continue
            covered_fraction = float(np.logical_and(mask, other_mask).sum()) / float(area)
            if covered_fraction >= coverage_threshold:
                pruned_indexes.add(index)
                break
    return pruned_indexes


def constrained_face_masks_by_index(
    indexed_face_masks: list[tuple[int, np.ndarray]],
    *,
    roi_shape: tuple[int, int],
) -> dict[int, np.ndarray]:
    pruned_indexes = covered_face_mask_indexes(indexed_face_masks)
    candidates = [
        (index, np.asarray(mask, dtype=bool), int(np.asarray(mask, dtype=bool).sum()))
        for index, mask in indexed_face_masks
        if index not in pruned_indexes and bool(np.asarray(mask, dtype=bool).any())
    ]
    candidates.sort(key=lambda item: item[2], reverse=True)
    if len(candidates) >= MAX_FACE_COUNT:
        selected = _largest_valid_face_triplet(candidates, roi_shape=roi_shape)
        if not selected:
            return {}
    else:
        selected = candidates[:MAX_FACE_COUNT]

    cleaned = _remove_shared_face_mask_pixels([(index, mask) for index, mask, _area in selected])
    return {
        index: mask
        for index, mask in cleaned
        if bool(mask.any())
    }


def _largest_valid_face_triplet(
    candidates: list[tuple[int, np.ndarray, int]],
    *,
    roi_shape: tuple[int, int],
) -> list[tuple[int, np.ndarray, int]]:
    for triplet in combinations(candidates, MAX_FACE_COUNT):
        if _face_triplet_constraints_met(
            [mask for _index, mask, _area in triplet],
            roi_shape=roi_shape,
        ):
            return list(triplet)
    return []


def _face_triplet_constraints_met(
    face_masks: list[np.ndarray],
    *,
    roi_shape: tuple[int, int],
) -> bool:
    if len(face_masks) != MAX_FACE_COUNT:
        return False
    quads = [
        _face_parallelogram_points(mask, roi_shape=roi_shape)
        for mask in face_masks
    ]
    if any(quad is None for quad in quads):
        return False

    max_shared_edge_distance = max(10.0, min(roi_shape) * 0.12)
    edge_indexes_by_face: list[list[int]] = [[] for _ in range(MAX_FACE_COUNT)]
    for left_index, right_index in combinations(range(MAX_FACE_COUNT), 2):
        match = _closest_compatible_quad_edges(
            quads[left_index],
            quads[right_index],
            max_distance=max_shared_edge_distance,
        )
        if match is None:
            return False
        left_edge_index, right_edge_index, _right_reversed = match
        edge_indexes_by_face[left_index].append(left_edge_index)
        edge_indexes_by_face[right_index].append(right_edge_index)

    common_corners = []
    for face_index, edge_indexes in enumerate(edge_indexes_by_face):
        if len(edge_indexes) != 2:
            return False
        corner = _quad_common_corner_for_adjacent_edges(
            quads[face_index],
            edge_indexes[0],
            edge_indexes[1],
        )
        if corner is None:
            return False
        common_corners.append(corner)

    center = np.mean(np.asarray(common_corners, dtype=np.float32), axis=0)
    max_corner_distance = max(12.0, min(roi_shape) * 0.12)
    return all(float(np.linalg.norm(corner - center)) <= max_corner_distance for corner in common_corners)


def _quad_common_corner_for_adjacent_edges(
    quad: np.ndarray,
    first_edge_index: int,
    second_edge_index: int,
) -> np.ndarray | None:
    corner_index = _quad_common_corner_index_for_adjacent_edges(first_edge_index, second_edge_index)
    if corner_index is None:
        return None
    return quad[corner_index]


def _quad_common_corner_index_for_adjacent_edges(
    first_edge_index: int,
    second_edge_index: int,
) -> int | None:
    if (first_edge_index + 1) % 4 == second_edge_index:
        return second_edge_index
    if (second_edge_index + 1) % 4 == first_edge_index:
        return first_edge_index
    return None


def _remove_shared_face_mask_pixels(
    indexed_face_masks: list[tuple[int, np.ndarray]],
) -> list[tuple[int, np.ndarray]]:
    if len(indexed_face_masks) < 2:
        return [(index, np.asarray(mask, dtype=bool).copy()) for index, mask in indexed_face_masks]

    mask_stack = np.stack([np.asarray(mask, dtype=bool) for _index, mask in indexed_face_masks], axis=0)
    shared_pixels = mask_stack.sum(axis=0) > 1
    return [
        (index, np.logical_and(mask_stack[mask_index], ~shared_pixels))
        for mask_index, (index, _mask) in enumerate(indexed_face_masks)
    ]


def draw_projected_face_grid(
    image: np.ndarray,
    *,
    color: tuple[int, int, int] = PROJECTED_FACE_GRID_COLOR,
) -> None:
    height, width = image.shape[:2]
    for fraction in (1.0 / 3.0, 2.0 / 3.0):
        x = int(round((width - 1) * fraction))
        y = int(round((height - 1) * fraction))
        cv2.line(image, (x, 0), (x, height - 1), (0, 0, 0), 3, lineType=cv2.LINE_8)
        cv2.line(image, (0, y), (width - 1, y), (0, 0, 0), 3, lineType=cv2.LINE_8)
        cv2.line(image, (x, 0), (x, height - 1), color, 1, lineType=cv2.LINE_8)
        cv2.line(image, (0, y), (width - 1, y), color, 1, lineType=cv2.LINE_8)


def face_projection_panel(
    roi_bgr: np.ndarray,
    face_masks: list[np.ndarray],
    *,
    roi_shape: tuple[int, int],
    output_height: int,
) -> np.ndarray | None:
    if not face_masks:
        return None

    fitted = face_quad_fits_for_masks(face_masks, roi_shape=roi_shape)
    if not fitted:
        return None

    padding = 10
    square_size = min(160, max(48, (int(output_height) - padding * (len(fitted) + 1)) // len(fitted)))
    panel_width = square_size + padding * 2
    panel = np.zeros((int(output_height), panel_width, 3), dtype=np.uint8)

    for index, (mask, quad) in enumerate(fitted):
        projection = project_masked_face_to_square(
            roi_bgr,
            mask,
            quad,
            square_size=square_size,
        )
        draw_projected_face_grid(projection)
        y1 = padding + index * (square_size + padding)
        y2 = min(y1 + square_size, panel.shape[0])
        if y1 >= panel.shape[0]:
            break
        panel[y1:y2, padding : padding + square_size] = projection[: y2 - y1]
        color = FACE_PARALLELOGRAM_COLORS[index % len(FACE_PARALLELOGRAM_COLORS)]
        cv2.rectangle(
            panel,
            (padding, y1),
            (padding + square_size - 1, y2 - 1),
            color,
            2,
            lineType=cv2.LINE_AA,
        )
    return panel


def project_masked_face_to_square(
    roi_bgr: np.ndarray,
    face_mask: np.ndarray,
    quad_xy: np.ndarray,
    *,
    square_size: int,
) -> np.ndarray:
    roi_height, roi_width = roi_bgr.shape[:2]
    mask = np.asarray(face_mask, dtype=bool)
    if mask.shape != (roi_height, roi_width):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (roi_width, roi_height),
            interpolation=cv2.INTER_NEAREST,
        ) > 0

    masked_roi = np.zeros_like(roi_bgr)
    masked_roi[mask] = roi_bgr[mask]
    source = np.asarray(quad_xy, dtype=np.float32).reshape(4, 2)
    destination = np.asarray(
        [
            [0.0, 0.0],
            [square_size - 1.0, 0.0],
            [square_size - 1.0, square_size - 1.0],
            [0.0, square_size - 1.0],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(
        masked_roi,
        transform,
        (square_size, square_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def _max_area_quad_from_hull_points(hull_points_xy: np.ndarray) -> np.ndarray | None:
    points = np.asarray(hull_points_xy, dtype=np.float32).reshape(-1, 2)
    if len(points) < 4:
        return None
    if len(points) > 48:
        indexes = np.linspace(0, len(points) - 1, 48, dtype=np.int32)
        points = points[indexes]

    best_quad = None
    best_area = -1.0
    for indexes in combinations(range(len(points)), 4):
        quad = _order_quad_points_clockwise(points[list(indexes)])
        area = float(cv2.contourArea(quad.reshape(-1, 1, 2)))
        if area > best_area:
            best_area = area
            best_quad = quad
    return best_quad


def _order_quad_points_clockwise(points_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    start_index = int(np.lexsort((ordered[:, 0], ordered[:, 1]))[0])
    return np.roll(ordered, -start_index, axis=0).astype(np.float32)


def _sticker_edge_mask(
    sticker_masks: list[np.ndarray],
    *,
    roi_shape: tuple[int, int],
) -> np.ndarray | None:
    roi_height, roi_width = roi_shape
    edge_mask = np.zeros((roi_height, roi_width), dtype=bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    for mask in sticker_masks:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != edge_mask.shape:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (roi_width, roi_height),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        if not bool(mask.any()):
            continue
        mask_u8 = mask.astype(np.uint8) * 255
        eroded = cv2.erode(mask_u8, kernel, iterations=1)
        edge_mask |= mask & (eroded == 0)
    return edge_mask if bool(edge_mask.any()) else None


def _sticker_edge_distance_image(
    sticker_masks: list[np.ndarray],
    *,
    roi_shape: tuple[int, int],
) -> np.ndarray | None:
    edge_mask = _sticker_edge_mask(sticker_masks, roi_shape=roi_shape)
    if edge_mask is None:
        return None
    return _distance_image_from_edge_mask(edge_mask)


def _distance_image_from_edge_mask(edge_mask: np.ndarray) -> np.ndarray:
    distance_source = np.where(edge_mask, 0, 255).astype(np.uint8)
    return cv2.distanceTransform(distance_source, cv2.DIST_L2, 3).astype(np.float32)


def _union_sticker_mask(
    sticker_masks: list[np.ndarray],
    roi_shape: tuple[int, int],
) -> np.ndarray | None:
    roi_height, roi_width = roi_shape
    union_mask = np.zeros((roi_height, roi_width), dtype=bool)
    for mask in sticker_masks:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != union_mask.shape:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (roi_width, roi_height),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        union_mask |= mask
    return union_mask if bool(union_mask.any()) else None


def _mask_bounds_xyxy(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


@lru_cache(maxsize=1)
def _projected_cube_axis_candidates() -> tuple[np.ndarray, ...]:
    candidates_by_key: dict[tuple[int, ...], np.ndarray] = {}
    for yaw_degrees in range(0, 360, 30):
        for pitch_degrees in range(-50, 51, 25):
            for roll_degrees in range(-50, 51, 25):
                rotation = _rotation_matrix_xyz(
                    np.radians(float(yaw_degrees)),
                    np.radians(float(pitch_degrees)),
                    np.radians(float(roll_degrees)),
                )
                axes = _visible_cube_axes_from_rotation(rotation)
                if axes is None:
                    continue
                key = tuple(int(round(value * 1000)) for value in axes.reshape(-1))
                candidates_by_key[key] = axes
    return tuple(candidates_by_key.values())


def _projected_cube_axis_candidates_for_edge_directions(
    edge_direction_angles: np.ndarray,
) -> tuple[np.ndarray, ...]:
    candidates = _projected_cube_axis_candidates()
    observed = np.asarray(edge_direction_angles, dtype=np.float32).reshape(-1)
    if len(observed) < 2:
        return candidates

    scored = [
        (_candidate_direction_alignment_cost(axes, observed), axes)
        for axes in candidates
    ]
    filtered = [
        axes
        for cost, axes in scored
        if cost <= CUBE_GRID_FIT_DIRECTION_TOLERANCE_RADIANS
    ]
    if filtered:
        return tuple(filtered)

    scored.sort(key=lambda item: item[0])
    return tuple(axes for _, axes in scored[: min(12, len(scored))])


def _dominant_edge_direction_angles(edge_mask: np.ndarray, *, max_directions: int = 3) -> np.ndarray:
    edge_u8 = (np.asarray(edge_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    height, width = edge_u8.shape[:2]
    min_line_length = max(6, int(round(min(height, width) / 12.0)))
    lines = cv2.HoughLinesP(
        edge_u8,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(8, min_line_length),
        minLineLength=min_line_length,
        maxLineGap=3,
    )
    if lines is None:
        return np.empty(0, dtype=np.float32)

    angles = []
    weights = []
    for line in lines.reshape(-1, 4):
        x1, y1, x2, y2 = (float(value) for value in line)
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length < min_line_length:
            continue
        angle = float(np.arctan2(dy, dx) % np.pi)
        angles.append(angle)
        weights.append(length)

    if not angles:
        return np.empty(0, dtype=np.float32)
    return _dominant_angles_from_weighted_samples(
        np.asarray(angles, dtype=np.float32),
        np.asarray(weights, dtype=np.float32),
        max_directions=max_directions,
    )


def _dominant_angles_from_weighted_samples(
    angles: np.ndarray,
    weights: np.ndarray,
    *,
    max_directions: int,
) -> np.ndarray:
    bin_count = 36
    bins = np.zeros(bin_count, dtype=np.float32)
    indexes = np.floor((angles % np.pi) / np.pi * bin_count).astype(np.int32) % bin_count
    for index, weight in zip(indexes, weights):
        bins[index] += float(weight)

    selected = []
    suppressed = np.zeros(bin_count, dtype=bool)
    separation_bins = 3
    for _ in range(max_directions):
        available = np.where(suppressed, -1.0, bins)
        best_index = int(np.argmax(available))
        if available[best_index] <= 0.0:
            break
        neighbor_indexes = [(best_index + offset) % bin_count for offset in (-1, 0, 1)]
        neighbor_weights = bins[neighbor_indexes]
        if float(neighbor_weights.sum()) <= 1e-6:
            angle = (best_index + 0.5) / bin_count * np.pi
        else:
            neighbor_angles = np.asarray(
                [(index + 0.5) / bin_count * np.pi for index in neighbor_indexes],
                dtype=np.float32,
            )
            angle = _weighted_undirected_angle_mean(neighbor_angles, neighbor_weights)
        selected.append(angle)
        for offset in range(-separation_bins, separation_bins + 1):
            suppressed[(best_index + offset) % bin_count] = True
    return np.asarray(selected, dtype=np.float32)


def _weighted_undirected_angle_mean(angles: np.ndarray, weights: np.ndarray) -> float:
    doubled = np.asarray(angles, dtype=np.float32) * 2.0
    weights = np.asarray(weights, dtype=np.float32)
    vector = np.asarray(
        [
            float(np.sum(np.cos(doubled) * weights)),
            float(np.sum(np.sin(doubled) * weights)),
        ],
        dtype=np.float32,
    )
    if float(np.linalg.norm(vector)) <= 1e-6:
        return float(angles[0] % np.pi)
    return float((np.arctan2(vector[1], vector[0]) / 2.0) % np.pi)


def _candidate_direction_alignment_cost(axes_xy: np.ndarray, observed_angles: np.ndarray) -> float:
    candidate_angles = _axis_direction_angles(axes_xy)
    observed = np.asarray(observed_angles, dtype=np.float32).reshape(-1)
    if len(observed) == 0:
        return 0.0

    observed_to_candidate = [
        min(_undirected_angle_distance(observed_angle, candidate_angle) for candidate_angle in candidate_angles)
        for observed_angle in observed
    ]
    candidate_to_observed = [
        min(_undirected_angle_distance(candidate_angle, observed_angle) for observed_angle in observed)
        for candidate_angle in candidate_angles
    ]
    if len(observed) >= 3:
        return float(max(max(observed_to_candidate), max(candidate_to_observed)))
    return float(max(observed_to_candidate))


def _axis_direction_angles(axes_xy: np.ndarray) -> np.ndarray:
    axes = np.asarray(axes_xy, dtype=np.float32).reshape(-1, 2)
    return np.asarray([float(np.arctan2(axis[1], axis[0]) % np.pi) for axis in axes], dtype=np.float32)


def _undirected_angle_distance(left: float, right: float) -> float:
    delta = abs((float(left) - float(right)) % np.pi)
    return float(min(delta, np.pi - delta))


def _rotation_matrix_xyz(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
    cos_roll, sin_roll = np.cos(roll), np.sin(roll)
    rotate_x = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_roll, -sin_roll],
            [0.0, sin_roll, cos_roll],
        ],
        dtype=np.float32,
    )
    rotate_y = np.asarray(
        [
            [cos_pitch, 0.0, sin_pitch],
            [0.0, 1.0, 0.0],
            [-sin_pitch, 0.0, cos_pitch],
        ],
        dtype=np.float32,
    )
    rotate_z = np.asarray(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return rotate_z @ rotate_y @ rotate_x


def _visible_cube_axes_from_rotation(rotation: np.ndarray) -> np.ndarray | None:
    axes_3d = np.asarray(rotation, dtype=np.float32).reshape(3, 3).T.copy()
    for index in range(3):
        if axes_3d[index, 2] > 0:
            axes_3d[index] *= -1.0
    depths = -axes_3d[:, 2]
    if float(depths.min()) < 0.22:
        return None
    projected_axes = axes_3d[:, :2]
    lengths = np.linalg.norm(projected_axes, axis=1)
    if float(lengths.min()) < 0.28:
        return None
    axes = projected_axes / float(lengths.max())
    if not _cube_projected_axes_are_possible(axes):
        return None
    return axes.astype(np.float32)


def _cube_projected_axes_are_possible(axes_xy: np.ndarray, tolerance: float = 0.035) -> bool:
    axes = np.asarray(axes_xy, dtype=np.float32).reshape(3, 2)
    lengths = np.linalg.norm(axes, axis=1)
    if float(lengths.min()) <= 1e-6:
        return False
    if not _projected_cube_silhouette_is_hexagonal(axes):
        return False

    edge_lengths_sq = []
    for left_index in range(3):
        for right_index in range(left_index + 1, 3):
            dot_xy = float(axes[left_index] @ axes[right_index])
            if dot_xy >= 0.0:
                return False
            left_length_sq = float(lengths[left_index] * lengths[left_index])
            right_length_sq = float(lengths[right_index] * lengths[right_index])
            discriminant = (
                (left_length_sq + right_length_sq) ** 2
                - 4.0 * (left_length_sq * right_length_sq - dot_xy * dot_xy)
            )
            if discriminant < 0.0:
                return False
            edge_length_sq = (
                left_length_sq
                + right_length_sq
                + float(np.sqrt(discriminant))
            ) / 2.0
            edge_lengths_sq.append(edge_length_sq)

    edge_lengths_sq = np.asarray(edge_lengths_sq, dtype=np.float32)
    mean_edge_length_sq = float(edge_lengths_sq.mean())
    if mean_edge_length_sq <= 1e-6:
        return False
    return bool(
        np.all(np.abs(edge_lengths_sq - mean_edge_length_sq) <= tolerance * mean_edge_length_sq)
    )


def _projected_cube_silhouette_is_hexagonal(axes_xy: np.ndarray) -> bool:
    hull_points, _ = _projected_cube_grid_geometry(axes_xy)
    hull = cv2.convexHull(hull_points.astype(np.float32).reshape(-1, 1, 2)).reshape(-1, 2)
    if len(hull) != 6:
        return False

    edges = np.roll(hull, -1, axis=0) - hull
    edge_lengths = np.linalg.norm(edges, axis=1)
    if float(edge_lengths.min()) < 0.12 * float(edge_lengths.max()):
        return False

    axes = np.asarray(axes_xy, dtype=np.float32).reshape(3, 2)
    face_areas = [
        abs(float(np.cross(axes[left_index], axes[right_index])))
        for left_index, right_index in ((0, 1), (1, 2), (2, 0))
    ]
    return min(face_areas) >= 0.16 * max(face_areas)


def _projected_cube_grid_geometry(
    axes_xy: np.ndarray,
) -> tuple[np.ndarray, list[tuple[tuple[float, float], tuple[float, float]]]]:
    axes = np.asarray(axes_xy, dtype=np.float32).reshape(3, 2)
    origin = np.zeros(2, dtype=np.float32)
    hull_points = np.asarray(
        [
            axes[0],
            axes[0] + axes[1],
            axes[1],
            axes[1] + axes[2],
            axes[2],
            axes[2] + axes[0],
        ],
        dtype=np.float32,
    )

    segments_by_key: dict[tuple[tuple[int, int], tuple[int, int]], tuple[tuple[float, float], tuple[float, float]]] = {}
    for first_index, second_index in ((0, 1), (1, 2), (2, 0)):
        first_axis = axes[first_index]
        second_axis = axes[second_index]
        for subdivision in range(4):
            t = subdivision / 3.0
            first_start = origin + first_axis * t
            first_end = first_start + second_axis
            second_start = origin + second_axis * t
            second_end = second_start + first_axis
            _add_unique_segment(segments_by_key, first_start, first_end)
            _add_unique_segment(segments_by_key, second_start, second_end)

    return hull_points, list(segments_by_key.values())


def _add_unique_segment(
    segments_by_key: dict[
        tuple[tuple[int, int], tuple[int, int]],
        tuple[tuple[float, float], tuple[float, float]],
    ],
    start: np.ndarray,
    end: np.ndarray,
) -> None:
    start_key = tuple(int(round(value * 1000)) for value in start)
    end_key = tuple(int(round(value * 1000)) for value in end)
    key = tuple(sorted((start_key, end_key)))
    segments_by_key[key] = (
        (float(start[0]), float(start[1])),
        (float(end[0]), float(end[1])),
    )


def _score_projected_cube_grid(
    union_mask: np.ndarray,
    *,
    mask_area: float,
    hull_points_xy: np.ndarray,
    line_segments_xy: list[tuple[tuple[float, float], tuple[float, float]]],
    line_thickness_px: int,
    minimum_coverage: float,
) -> ProjectedCubeGridFit:
    hull_mask = np.zeros(union_mask.shape, dtype=np.uint8)
    hull_contour = cv2.convexHull(np.round(hull_points_xy).astype(np.int32).reshape(-1, 1, 2))
    cv2.fillConvexPoly(hull_mask, hull_contour, 255)
    covered_pixels = float(np.count_nonzero(union_mask & (hull_mask > 0)))
    coverage_ratio = covered_pixels / mask_area

    line_mask = np.zeros(union_mask.shape, dtype=np.uint8)
    thickness = max(1, int(line_thickness_px))
    for start, end in line_segments_xy:
        cv2.line(
            line_mask,
            (int(round(start[0])), int(round(start[1]))),
            (int(round(end[0])), int(round(end[1]))),
            255,
            thickness,
            lineType=cv2.LINE_AA,
        )
    line_intersection_ratio = float(np.count_nonzero(union_mask & (line_mask > 0))) / mask_area

    hull_area = max(float(np.count_nonzero(hull_mask)), 1.0)
    hull_area_ratio = hull_area / mask_area
    bounds = _mask_bounds_xyxy(union_mask)
    bbox_fill_penalty = 0.0
    if bounds is not None:
        bx1, by1, bx2, by2 = bounds
        mask_bbox_area = max(float((bx2 - bx1) * (by2 - by1)), 1.0)
        hx, hy, hw, hh = cv2.boundingRect(hull_contour)
        hull_bbox_area = max(float(hw * hh), 1.0)
        bbox_fill = mask_bbox_area / hull_bbox_area
        bbox_fill_penalty = max(0.0, 0.58 - bbox_fill) * 0.9

    coverage_penalty = max(0.0, minimum_coverage - coverage_ratio) * 4.0
    size_penalty = max(0.0, hull_area_ratio - 2.85) * 0.10
    score = line_intersection_ratio + coverage_penalty + bbox_fill_penalty + size_penalty
    return ProjectedCubeGridFit(
        score=float(score),
        line_edge_distance_score=float(line_intersection_ratio),
        coverage_ratio=float(coverage_ratio),
        hull_area_ratio=float(hull_area_ratio),
        hull_points_xy=np.asarray(hull_points_xy, dtype=np.float32),
        line_segments_xy=line_segments_xy,
    )


def _score_projected_cube_grid_from_edge_distance(
    edge_distance_image: np.ndarray,
    *,
    coverage_points_xy: np.ndarray,
    mask_area: float,
    hull_points_xy: np.ndarray,
    line_segments_xy: list[tuple[tuple[float, float], tuple[float, float]]],
    line_thickness_px: int,
    minimum_coverage: float,
) -> ProjectedCubeGridFit:
    coverage_points = np.asarray(coverage_points_xy, dtype=np.float32).reshape(-1, 2)
    hull_contour = cv2.convexHull(np.asarray(hull_points_xy, dtype=np.float32).reshape(-1, 1, 2))
    hull = hull_contour.reshape(-1, 2).astype(np.float32)

    inside_hull = _points_inside_convex_polygon(coverage_points, hull)
    coverage_ratio = float(inside_hull.mean()) if len(coverage_points) else 0.0

    line_scores = _line_edge_distance_scores(
        edge_distance_image,
        line_segments_xy,
        line_thickness_px=line_thickness_px,
    )
    line_distance_score = _robust_line_edge_distance_score(line_scores, line_segments_xy)

    hull_area = max(float(cv2.contourArea(hull_contour)), 1.0)
    hull_area_ratio = hull_area / mask_area
    point_min = coverage_points.min(axis=0) if len(coverage_points) else np.zeros(2, dtype=np.float32)
    point_max = coverage_points.max(axis=0) if len(coverage_points) else np.ones(2, dtype=np.float32)
    mask_bbox_area = max(float(np.prod(point_max - point_min + 1.0)), 1.0)
    hx, hy, hw, hh = cv2.boundingRect(hull_contour)
    hull_bbox_area = max(float(hw * hh), 1.0)
    bbox_fill = mask_bbox_area / hull_bbox_area
    bbox_fill_penalty = max(0.0, 0.58 - bbox_fill) * 0.9

    coverage_penalty = max(0.0, minimum_coverage - coverage_ratio) * 4.0
    size_penalty = max(0.0, hull_area_ratio - 2.85) * 0.10
    score = line_distance_score + coverage_penalty + bbox_fill_penalty + size_penalty
    return ProjectedCubeGridFit(
        score=float(score),
        line_edge_distance_score=float(line_distance_score),
        coverage_ratio=float(coverage_ratio),
        hull_area_ratio=float(hull_area_ratio),
        hull_points_xy=np.asarray(hull_points_xy, dtype=np.float32),
        line_segments_xy=line_segments_xy,
    )


def _line_edge_distance_scores(
    edge_distance_image: np.ndarray,
    line_segments_xy: list[tuple[tuple[float, float], tuple[float, float]]],
    *,
    line_thickness_px: int,
) -> np.ndarray:
    segments = np.asarray(line_segments_xy, dtype=np.float32).reshape(-1, 2, 2)
    if len(segments) == 0:
        return np.empty(0, dtype=np.float32)

    line_radius = max(0.5, float(line_thickness_px) * 0.5 + 0.5)
    scores = []
    for segment in segments:
        length = float(np.linalg.norm(segment[1] - segment[0]))
        sample_count = max(4, min(12, int(round(length / 8.0))))
        t = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)
        points = segment[0][None, :] * (1.0 - t[:, None]) + segment[1][None, :] * t[:, None]
        distances = _distance_image_values(edge_distance_image, points)
        adjusted_distances = np.maximum(distances - line_radius, 0.0)
        scores.append(float(np.mean(np.minimum(adjusted_distances, 12.0)) / 12.0))
    return np.asarray(scores, dtype=np.float32)


def _robust_line_edge_distance_score(
    line_scores: np.ndarray,
    line_segments_xy: list[tuple[tuple[float, float], tuple[float, float]]] | None = None,
) -> float:
    scores = np.asarray(line_scores, dtype=np.float32).reshape(-1)
    if len(scores) == 0:
        return 1.0

    if line_segments_xy is None:
        return _robust_score_subset(scores, minimum_supported=6)

    groups = _line_direction_groups(line_segments_xy)
    if not groups:
        return _robust_score_subset(scores, minimum_supported=6)

    group_scores = []
    for indexes in groups:
        group_scores.append(
            _robust_score_subset(
                scores[np.asarray(indexes, dtype=np.int32)],
                minimum_supported=2,
            )
        )
    return float(np.mean(group_scores))


def _robust_score_subset(line_scores: np.ndarray, *, minimum_supported: int) -> float:
    scores = np.asarray(line_scores, dtype=np.float32).reshape(-1)
    if len(scores) == 0:
        return 1.0

    supported_target = min(len(scores), max(minimum_supported, int(np.ceil(len(scores) * 0.40))))
    best_scores = np.sort(scores)[:supported_target]
    supported_count = int(np.count_nonzero(scores <= CUBE_GRID_FIT_SUPPORTED_LINE_DISTANCE))
    support_penalty = max(0, supported_target - supported_count) / supported_target * 0.35
    return float(np.mean(best_scores) + support_penalty)


def _line_direction_groups(
    line_segments_xy: list[tuple[tuple[float, float], tuple[float, float]]],
) -> list[list[int]]:
    segments = np.asarray(line_segments_xy, dtype=np.float32).reshape(-1, 2, 2)
    groups: list[list[int]] = []
    group_axes: list[np.ndarray] = []
    for index, segment in enumerate(segments):
        vector = segment[1] - segment[0]
        length = float(np.linalg.norm(vector))
        if length <= 1e-6:
            continue
        axis = vector / length
        if axis[0] < 0.0 or (abs(float(axis[0])) < 1e-6 and axis[1] < 0.0):
            axis *= -1.0

        matched = False
        for group_index, group_axis in enumerate(group_axes):
            if abs(float(axis @ group_axis)) >= 0.985:
                groups[group_index].append(index)
                matched = True
                break
        if not matched:
            group_axes.append(axis)
            groups.append([index])
    return groups


def _sample_line_segment_xy_points(
    line_segments_xy: list[tuple[tuple[float, float], tuple[float, float]]],
    *,
    max_points: int,
) -> np.ndarray:
    segments = np.asarray(line_segments_xy, dtype=np.float32).reshape(-1, 2, 2)
    if len(segments) == 0 or max_points <= 0:
        return np.empty((0, 2), dtype=np.float32)

    lengths = np.linalg.norm(segments[:, 1, :] - segments[:, 0, :], axis=1)
    total_length = float(lengths.sum())
    if total_length <= 1e-6:
        return np.empty((0, 2), dtype=np.float32)

    points = []
    for segment, length in zip(segments, lengths):
        sample_count = max(2, int(round(max_points * float(length) / total_length)))
        t = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)
        points.append(segment[0][None, :] * (1.0 - t[:, None]) + segment[1][None, :] * t[:, None])
    sampled = np.concatenate(points, axis=0)
    if len(sampled) <= max_points:
        return sampled
    indexes = np.linspace(0, len(sampled) - 1, max_points, dtype=np.int32)
    return sampled[indexes]


def _distance_image_values(distance_image: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0:
        return np.empty(0, dtype=np.float32)

    height, width = distance_image.shape[:2]
    xs = np.clip(np.round(points[:, 0]).astype(np.int32), 0, width - 1)
    ys = np.clip(np.round(points[:, 1]).astype(np.int32), 0, height - 1)
    return distance_image[ys, xs].astype(np.float32)


def _points_inside_convex_polygon(points_xy: np.ndarray, polygon_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    polygon = np.asarray(polygon_xy, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0 or len(polygon) < 3:
        return np.zeros(len(points), dtype=bool)

    edges = np.roll(polygon, -1, axis=0) - polygon
    point_vectors = points[:, None, :] - polygon[None, :, :]
    crosses = edges[None, :, 0] * point_vectors[:, :, 1] - edges[None, :, 1] * point_vectors[:, :, 0]
    return np.all(crosses >= -1e-4, axis=1) | np.all(crosses <= 1e-4, axis=1)


def _point_to_segment_distances(points_xy: np.ndarray, segments_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    segments = np.asarray(segments_xy, dtype=np.float32).reshape(-1, 2, 2)
    if len(points) == 0 or len(segments) == 0:
        return np.empty((len(points), len(segments)), dtype=np.float32)

    start = segments[:, 0, :]
    end = segments[:, 1, :]
    segment_vectors = end - start
    segment_lengths_sq = np.maximum(np.sum(segment_vectors * segment_vectors, axis=1), 1e-6)
    point_vectors = points[:, None, :] - start[None, :, :]
    projection_t = np.sum(point_vectors * segment_vectors[None, :, :], axis=2) / segment_lengths_sq[None, :]
    projection_t = np.clip(projection_t, 0.0, 1.0)
    closest = start[None, :, :] + projection_t[:, :, None] * segment_vectors[None, :, :]
    deltas = points[:, None, :] - closest
    return np.sqrt(np.sum(deltas * deltas, axis=2))


def draw_projected_cube_grid_fit(
    frame: np.ndarray,
    fit: ProjectedCubeGridFit,
    *,
    offset_xy: tuple[int, int],
) -> None:
    offset = np.asarray(offset_xy, dtype=np.float32)
    hull = np.round(fit.hull_points_xy + offset).astype(np.int32).reshape(-1, 1, 2)
    hull = cv2.convexHull(hull)
    cv2.polylines(frame, [hull], isClosed=True, color=(0, 240, 255), thickness=2)
    for start, end in fit.line_segments_xy:
        start_xy = np.asarray(start, dtype=np.float32) + offset
        end_xy = np.asarray(end, dtype=np.float32) + offset
        cv2.line(
            frame,
            (int(round(start_xy[0])), int(round(start_xy[1]))),
            (int(round(end_xy[0])), int(round(end_xy[1]))),
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.line(
            frame,
            (int(round(start_xy[0])), int(round(start_xy[1]))),
            (int(round(end_xy[0])), int(round(end_xy[1]))),
            (0, 90, 255),
            1,
            lineType=cv2.LINE_AA,
        )


def tensor_to_numpy(value) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def class_name_for_id(result, sticker_model, class_id: int) -> str:
    names = getattr(result, "names", None) or getattr(sticker_model, "names", {}) or {}
    return str(names.get(class_id, class_id)).lower()


def sticker_piece_type(class_name: str) -> str:
    return class_name.removesuffix("_sticker")


def mask_color_for_segment(class_name: str, segment_index: int) -> tuple[int, int, int]:
    if class_name == "face":
        return FACE_INSTANCE_MASK_COLORS[segment_index % len(FACE_INSTANCE_MASK_COLORS)]
    return DEFAULT_STICKER_MASK_COLOR


def roi_color_samples(crop_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reliable_mask = _reliable_color_mask(crop_bgr)
    return crop_bgr, reliable_mask


def random_mask_sample_points(mask: np.ndarray, sample_count: int) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or sample_count <= 0:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)
    if len(xs) <= sample_count:
        return ys.astype(np.int32), xs.astype(np.int32)

    seed = int(mask.sum() + ys.sum() * 31 + xs.sum() * 131) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(xs), size=sample_count, replace=False)
    return ys[selected].astype(np.int32), xs[selected].astype(np.int32)


def sticker_color_sample_points_from_roi(
    color_samples: tuple[np.ndarray, np.ndarray],
    sticker_mask: np.ndarray,
    sample_count: int = STICKER_COLOR_SAMPLE_COUNT,
) -> tuple[np.ndarray, np.ndarray]:
    _, roi_reliable_mask = color_samples
    if not bool(sticker_mask.any()):
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)

    sample_mask = roi_reliable_mask & sticker_mask
    if int(sample_mask.sum()) < min(sample_count, int(sticker_mask.sum())):
        sample_mask = sticker_mask

    return random_mask_sample_points(sample_mask, sample_count)


def sticker_color_samples_from_roi(
    color_samples: tuple[np.ndarray, np.ndarray],
    sticker_mask: np.ndarray,
    sample_count: int = STICKER_COLOR_SAMPLE_COUNT,
) -> np.ndarray:
    crop, _ = color_samples
    ys, xs = sticker_color_sample_points_from_roi(
        color_samples,
        sticker_mask,
        sample_count,
    )
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.uint8)
    return crop[ys, xs].astype(np.uint8)


def sticker_color_samples_at_points(
    color_samples: tuple[np.ndarray, np.ndarray],
    ys: np.ndarray,
    xs: np.ndarray,
) -> np.ndarray:
    crop, _ = color_samples
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.uint8)
    return crop[ys, xs].astype(np.uint8)


def direct_sticker_graph(
    nodes: list[StickerNode],
    *,
    adjacency_radius_px: int,
) -> PieceGraph:
    return PieceGraph(
        nodes=nodes,
        edges=_legal_mask_adjacencies(nodes, adjacency_radius_px=adjacency_radius_px),
    )


def stickers_for_ready_color_classifier(
    frame: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
    stickers: list[tuple[np.ndarray, np.ndarray]],
    color_classifier: ColorDensityClassifier | None,
    color_samples: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[tuple[np.ndarray, np.ndarray, int | None]]:
    if color_classifier is None or not color_classifier.ready:
        return [(mask, bgr, None) for mask, bgr in stickers]

    classified_stickers = []
    for sticker_index, (sticker_mask, sticker_bgr) in enumerate(stickers):
        cluster_index = (
            color_classifier.classify_bgr_samples(sticker_bgr)
            if len(sticker_bgr) > 0
            else None
        )
        if color_samples is None:
            x1, y1, x2, y2 = roi_xyxy
            color_samples = roi_color_samples(frame[y1:y2, x1:x2])
        sample_bgr = sticker_color_samples_from_roi(color_samples, sticker_mask)
        if len(sample_bgr) == 0:
            sample_bgr = sticker_bgr
        classified_stickers.append((sticker_mask, sample_bgr, cluster_index))
    return classified_stickers


def classified_net_colors_from_graph(
    graph,
    color_classifier: ColorDensityClassifier,
) -> tuple[dict[int, str], dict[str, tuple[int, int, int]]]:
    colors_by_node_id = {}
    bgr_by_color = {}
    if not color_classifier.ready:
        return colors_by_node_id, bgr_by_color

    for node in graph.nodes:
        cluster_index = color_classifier.classify_bgr(np.asarray(node.mean_bgr, dtype=np.uint8))
        if cluster_index is None or cluster_index >= len(color_classifier.centroids):
            continue
        color_id = f"cluster-{int(cluster_index) + 1}"
        colors_by_node_id[node.node_id] = color_id
        bgr_by_color[color_id] = tuple(
            int(value) for value in color_classifier.centroids[cluster_index][::-1]
        )
    return colors_by_node_id, bgr_by_color


def parse_net_cluster_faces(value: str) -> dict[str, str]:
    face_by_color = {}
    value = value.strip()
    if not value:
        return face_by_color

    for item in value.split(","):
        cluster_text, separator, face = item.partition("=")
        if not separator:
            raise argparse.ArgumentTypeError(
                "Expected comma-separated cluster=face entries, for example 1=R,2=U"
            )
        cluster_text = cluster_text.strip()
        face = face.strip().upper()
        if not cluster_text.isdigit() or face not in {"U", "D", "F", "B", "R", "L"}:
            raise argparse.ArgumentTypeError(
                "Expected cluster numbers and cube faces U,D,F,B,R,L"
            )
        face_by_color[f"cluster-{int(cluster_text)}"] = face
    return face_by_color


def normalized_rgb_html_plot_path(rgb_html_path: Path) -> Path:
    suffix = rgb_html_path.suffix or ".html"
    return rgb_html_path.with_name(f"{rgb_html_path.stem}_normalized{suffix}")


def color_features_enabled(args) -> bool:
    return int(getattr(args, "color_plot_frames", 0)) > 0


def color_plot_outputs_requested(args) -> bool:
    return color_features_enabled(args) and bool(
        args.rgb_color_plot or args.rgb_html_plot or args.hsv_html_plot
    )


def save_rgb_html_plots(
    bgr_samples: np.ndarray,
    rgb_html_path: Path,
    *,
    title: str,
    normalize_rgb: bool,
    args,
    color_classifier: ColorDensityClassifier | None = None,
) -> None:
    cluster_indexes = None
    cluster_centroids = None
    if color_classifier is not None and color_classifier.ready:
        cluster_indexes = classifier_cluster_indexes_for_bgr_samples(
            color_classifier,
            bgr_samples,
        )
        cluster_centroids = color_classifier.centroids

    save_rgb_samples_interactive_plot(
        bgr_samples,
        str(rgb_html_path),
        title=title,
        normalize_rgb=normalize_rgb,
        point_size=args.plotly_point_size,
        cluster_indexes=cluster_indexes,
        cluster_centroids=cluster_centroids,
    )
    save_rgb_samples_interactive_plot(
        bgr_samples,
        str(normalized_rgb_html_plot_path(rgb_html_path)),
        title=f"{title} normalized to brightness",
        normalize_rgb=True,
        point_size=args.plotly_point_size,
        cluster_indexes=cluster_indexes,
        cluster_centroids=cluster_centroids,
    )


def classifier_cluster_indexes_for_bgr_samples(
    color_classifier: ColorDensityClassifier,
    bgr_samples: np.ndarray,
) -> np.ndarray:
    samples = np.asarray(bgr_samples, dtype=np.uint8).reshape(-1, 3)
    return np.asarray(
        [
            -1
            if (cluster_index := color_classifier.classify_bgr(sample)) is None
            else cluster_index
            for sample in samples
        ],
        dtype=np.int32,
    )


def concatenate_samples(
    bgr_batches: list[np.ndarray],
    max_samples: int,
) -> np.ndarray:
    if not bgr_batches:
        return np.empty((0, 3), dtype=np.uint8)

    bgr_samples = np.concatenate(bgr_batches, axis=0)
    if len(bgr_samples) > max_samples:
        indexes = np.linspace(0, len(bgr_samples) - 1, max_samples, dtype=np.int32)
        bgr_samples = bgr_samples[indexes]
    return bgr_samples


def draw_sticker_segments(
    frame: np.ndarray,
    roi: np.ndarray,
    sticker_model,
    roi_xyxy,
    args,
    color_classifier: ColorDensityClassifier | None = None,
    cube_net_state: CubeNetState | None = None,
) -> tuple[int, np.ndarray, tuple[int, int, int, int] | None, np.ndarray | None]:
    x1, y1, x2, y2 = roi_xyxy
    original_frame = frame.copy()
    model_roi = resize_to_square_size(roi, args.sticker_imgsz)
    results = sticker_model.predict(
        source=model_roi,
        conf=args.sticker_conf,
        imgsz=args.sticker_imgsz,
        verbose=False,
    )
    bgr_batches = []
    if not results:
        return 0, np.empty((0, 3), dtype=np.uint8), None, None

    result = results[0]
    boxes = result.boxes
    if boxes is None:
        return 0, np.empty((0, 3), dtype=np.uint8), None, None

    overlay = np.zeros_like(frame)
    roi_height = y2 - y1
    roi_width = x2 - x1
    result_masks = getattr(result, "masks", None)
    if result_masks is None:
        return 0, np.empty((0, 3), dtype=np.uint8), None, None
    mask_data = result_masks.data.detach().cpu().numpy()

    count = 0
    confidence_labels = []
    sticker_color_numbers = []
    classifier_bgr_batches = []
    sticker_nodes = []
    grid_fit_masks = []
    face_hull_masks = []
    build_sticker_nodes = args.draw_sticker_graph or cube_net_state is not None
    color_samples = roi_color_samples(roi)
    tracking_min_x: int | None = None
    tracking_min_y: int | None = None
    tracking_max_x: int | None = None
    tracking_max_y: int | None = None
    indexed_face_masks = []
    for index in range(len(boxes)):
        class_id = int(boxes.cls[index].item())
        class_name = class_name_for_id(result, sticker_model, class_id)
        if class_name != "face":
            continue
        mask_bool = sticker_mask_bool(mask_data, index, roi_width, roi_height)
        if mask_bool.any():
            indexed_face_masks.append((index, mask_bool))
    constrained_face_masks = constrained_face_masks_by_index(
        indexed_face_masks,
        roi_shape=(roi_height, roi_width),
    )

    for index in range(len(boxes)):
        class_id = int(boxes.cls[index].item())
        class_name = class_name_for_id(result, sticker_model, class_id)
        if class_name not in STICKER_CLASS_NAMES:
            continue
        piece_type = sticker_piece_type(class_name)
        color = mask_color_for_segment(class_name, count)

        mask_bool = sticker_mask_bool(mask_data, index, roi_width, roi_height)
        if class_name == "face":
            mask_bool = constrained_face_masks.get(index)
            if mask_bool is None:
                continue
        if not mask_bool.any():
            continue

        sample_mask = apply_mask_padding(mask_bool, args.segmentation_padding_px)
        if not sample_mask.any():
            sample_mask = mask_bool
        draw_mask = sample_mask
        graph_mask = mask_bool
        if class_name == "face":
            face_hull_masks.append(graph_mask)
        if getattr(args, "draw_cube_grid_fit", False):
            grid_fit_masks.append(graph_mask)

        ys, xs = np.nonzero(graph_mask)
        mask_min_x = int(xs.min())
        mask_max_x = int(xs.max()) + 1
        mask_min_y = int(ys.min())
        mask_max_y = int(ys.max()) + 1
        tracking_min_x = mask_min_x if tracking_min_x is None else min(tracking_min_x, mask_min_x)
        tracking_min_y = mask_min_y if tracking_min_y is None else min(tracking_min_y, mask_min_y)
        tracking_max_x = mask_max_x if tracking_max_x is None else max(tracking_max_x, mask_max_x)
        tracking_max_y = mask_max_y if tracking_max_y is None else max(tracking_max_y, mask_max_y)
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
                cv2.polylines(frame, [points], isClosed=True, color=(220, 220, 220), thickness=2)

        sample_ys, sample_xs = sticker_color_sample_points_from_roi(color_samples, sample_mask)
        sample_bgr = sticker_color_samples_at_points(
            color_samples,
            sample_ys,
            sample_xs,
        )
        raw_stickers = (
            [(draw_mask, sample_bgr)]
            if len(sample_bgr)
            else []
        )
        if (
            build_sticker_nodes
            and len(sample_bgr)
            and int(graph_mask.sum()) >= args.graph_min_sticker_area
        ):
            full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = graph_mask.astype(np.uint8) * 255
            mean_sample_bgr = (
                sample_bgr.astype(np.float32)
                .mean(axis=0)
                .clip(0, 255)
                .astype(np.uint8)
            )
            sticker_nodes.append(
                StickerNode(
                    node_id=len(sticker_nodes),
                    piece_id=len(sticker_nodes),
                    piece_type=piece_type,
                    confidence=confidence,
                    centroid_xy=(float(xs.mean() + x1), float(ys.mean() + y1)),
                    mask=full_mask,
                    mean_bgr=tuple(int(channel) for channel in mean_sample_bgr),
                )
            )
        if not raw_stickers:
            raw_stickers = [
                (
                    draw_mask,
                    np.empty((0, 3), dtype=np.uint8),
                )
            ]
        display_stickers = stickers_for_ready_color_classifier(
            frame,
            roi_xyxy,
            raw_stickers,
            color_classifier,
            color_samples=color_samples,
        )

        roi_overlay = overlay[y1:y2, x1:x2]
        for sticker_mask, sticker_bgr, cluster_index in display_stickers:
            sticker_color = color
            if (
                class_name != "face"
                and
                color_classifier is not None
                and color_classifier.ready
                and getattr(args, "show_sticker_color_classification", True)
                and len(sticker_bgr) > 0
                and cluster_index is not None
                and cluster_index < len(color_classifier.centroids)
            ):
                centroid_rgb = color_classifier.centroids[cluster_index]
                sticker_color = tuple(int(value) for value in centroid_rgb[::-1])
                sticker_ys, sticker_xs = np.nonzero(sticker_mask)
                sticker_label_at = (
                    int(sticker_xs.mean()) + x1,
                    int(sticker_ys.mean()) + y1,
                )
                sticker_color_numbers.append((sticker_label_at, cluster_index + 1))
            roi_overlay[sticker_mask] = sticker_color

        sticker_bgr_batches = []
        for _, sticker_bgr_samples, _ in display_stickers:
            if len(sticker_bgr_samples) > 0:
                sticker_bgr_batches.append(sticker_bgr_samples)
        bgr_samples = concatenate_samples(
            sticker_bgr_batches,
            args.color_plot_samples_per_frame,
        )
        if len(bgr_samples) > 0:
            bgr_batches.append(bgr_samples)
        if color_classifier is not None:
            classifier_sample_batches = [
                sticker_bgr
                for _, sticker_bgr in raw_stickers
                if len(sticker_bgr) > 0
            ]
            classifier_bgr_samples = (
                np.concatenate(classifier_sample_batches, axis=0)
                if classifier_sample_batches
                else np.empty((0, 3), dtype=np.uint8)
            )
            if len(classifier_bgr_samples) > 0:
                classifier_bgr_batches.append(classifier_bgr_samples)
        count += 1

    if getattr(args, "show_original_underlay", False):
        alpha = float(np.clip(getattr(args, "mask_opacity", 1.0), 0.0, 1.0))
        overlay_mask = np.any(overlay != 0, axis=2)
        frame[:] = original_frame
        if bool(overlay_mask.any()):
            blended = (
                original_frame[overlay_mask].astype(np.float32) * (1.0 - alpha)
                + overlay[overlay_mask].astype(np.float32) * alpha
            )
            frame[overlay_mask] = blended.clip(0, 255).astype(np.uint8)
    else:
        frame[:] = overlay
    if face_hull_masks:
        draw_face_parallelogram_hulls(
            frame,
            face_hull_masks,
            roi_shape=(roi_height, roi_width),
            offset_xy=(x1, y1),
        )
    projection_panel = face_projection_panel(
        roi,
        face_hull_masks,
        roi_shape=(roi_height, roi_width),
        output_height=frame.shape[0],
    )
    if getattr(args, "draw_cube_grid_fit", False) and grid_fit_masks:
        line_overlay = low_intersection_sticker_lines(
            grid_fit_masks,
            roi_shape=(roi_height, roi_width),
            line_width_px=getattr(args, "empty_space_line_width", EMPTY_SPACE_LINE_WIDTH_PX),
            angle_bucket_degrees=getattr(
                args,
                "basis_angle_bucket_degrees",
                BASIS_VECTOR_ANGLE_BUCKET_DEGREES,
            ),
        )
        draw_low_intersection_lines(frame, line_overlay, offset_xy=(x1, y1))
    if (args.draw_sticker_graph or cube_net_state is not None) and sticker_nodes:
        graph = direct_sticker_graph(
            sticker_nodes,
            adjacency_radius_px=args.graph_adjacency_radius,
        )
        if cube_net_state is not None and color_classifier is not None and color_classifier.ready:
            classified_colors, classified_bgr = classified_net_colors_from_graph(
                graph,
                color_classifier,
            )
            cube_net_state.replace_with(
                build_cube_net_from_graph(
                    graph,
                    classified_colors_by_node_id=classified_colors,
                    classified_bgr_by_color=classified_bgr,
                    face_by_color=getattr(args, "net_cluster_faces", None),
                )
            )
        if args.draw_sticker_graph:
            frame[:] = draw_sticker_graph(
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
        draw_sticker_color_numbers(frame, sticker_color_numbers)
    bgr_samples = concatenate_samples(
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
    tracking_bounds_xyxy = (
        (tracking_min_x, tracking_min_y, tracking_max_x, tracking_max_y)
        if (
            tracking_min_x is not None
            and tracking_min_y is not None
            and tracking_max_x is not None
            and tracking_max_y is not None
        )
        else None
    )
    return count, bgr_samples, tracking_bounds_xyxy, projection_panel


def draw_sticker_color_numbers(
    frame: np.ndarray,
    sticker_color_numbers: list[tuple[tuple[int, int], int]],
) -> None:
    for label_at, color_number in sticker_color_numbers:
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
    sticker_model,
    args,
    color_classifier: ColorDensityClassifier | None = None,
    cube_net_state: CubeNetState | None = None,
    roi_tracker: RoiTracker | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    output = frame.copy()
    roi_xyxy, candidate, roi_source = roi_for_frame(frame, bbox_detector, args, roi_tracker)
    if roi_xyxy is None:
        return output, np.empty((0, 3), dtype=np.uint8)

    x1, y1, x2, y2 = roi_xyxy
    roi = frame[y1:y2, x1:x2]
    segment_count, bgr_samples, tracking_bounds_xyxy, projection_panel = draw_sticker_segments(
        output,
        roi,
        sticker_model,
        roi_xyxy,
        args,
        color_classifier,
        cube_net_state,
    )
    if roi_tracker is not None:
        height, width = frame.shape[:2]
        roi_tracker.observe_tracking_bounds(
            tracking_bounds_xyxy,
            width,
            height,
            tracking_gain=getattr(args, "roi_tracking_gain", 1.4),
            tracking_padding=getattr(args, "roi_tracking_padding", 0.18),
        )
        if roi_tracker.roi_xyxy is not None:
            roi_xyxy = roi_tracker.roi_xyxy
            x1, y1, x2, y2 = roi_xyxy

    if args.show_bbox or args.debug_overlay:
        roi_color = (0, 0, 255) if roi_source == "detector" else (40, 220, 40)
        cv2.rectangle(output, (x1, y1), (x2, y2), roi_color, 2)
    if args.debug_overlay:
        debug_label = (
            f"cube {candidate.confidence:.2f} stickers {segment_count}"
            if candidate is not None
            else f"sticker viewport stickers {segment_count}"
        )
        cv2.putText(
            output,
            debug_label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (40, 220, 40),
            2,
            cv2.LINE_AA,
        )
    if projection_panel is not None:
        output = np.concatenate([output, projection_panel], axis=1)
    return output, bgr_samples


def roi_for_frame(
    frame: np.ndarray,
    bbox_detector,
    args,
    roi_tracker: RoiTracker | None = None,
) -> tuple[tuple[int, int, int, int] | None, object | None, str]:
    height, width = frame.shape[:2]
    if roi_tracker is not None and roi_tracker.roi_xyxy is not None:
        return roi_tracker.roi_xyxy, roi_tracker.cube_candidate, "tracker"

    candidate = bbox_detector.detect(frame)
    if candidate is None:
        if roi_tracker is not None:
            roi_tracker.reset()
        return None, None, "none"
    if roi_tracker is not None:
        return (
            roi_tracker.initialize(candidate, width, height, args.roi_padding),
            candidate,
            "detector",
        )
    return clipped_square_roi(candidate.bbox_xyxy, width, height, args.roi_padding), candidate, "detector"


def resize_to_square_size(image: np.ndarray, size: int) -> np.ndarray:
    if size <= 0:
        return image
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Camera index, video path, or image path")
    parser.add_argument("--bbox-model", type=Path, default=DEFAULT_BBOX_MODEL)
    parser.add_argument("--sticker-model", type=Path, default=DEFAULT_STICKER_MODEL)
    parser.add_argument("--bbox-conf", type=float, default=0.25)
    parser.add_argument("--sticker-conf", type=float, default=0.25)
    parser.add_argument("--bbox-imgsz", type=int, default=640)
    parser.add_argument("--sticker-imgsz", type=int, default=256)
    parser.add_argument("--roi-padding", type=float, default=0.08)
    parser.add_argument(
        "--roi-tracking-gain",
        type=float,
        default=1.4,
        help="Multiplier for ROI movement from sticker bounds-center offset; higher follows faster",
    )
    parser.add_argument(
        "--roi-tracking-padding",
        type=float,
        default=0.18,
        help="Extra padding ratio added around tracked sticker bounds when updating the ROI",
    )
    parser.add_argument(
        "--segmentation-padding-px",
        type=int,
        default=-2,
        help="Pixel padding applied to each sticker segmentation mask before sampling; negative values erode",
    )
    parser.add_argument(
        "--draw-sticker-graph",
        dest="draw_sticker_graph",
        action="store_true",
        help="Draw sticker graph nodes and edges on top of the sticker overlay",
    )
    parser.add_argument(
        "--adjacency-radius",
        "--graph-adjacency-radius",
        dest="graph_adjacency_radius",
        type=int,
        default=1,
        help="Pixel distance threshold used when connecting nearby graph nodes",
    )
    parser.add_argument(
        "--graph-min-sticker-area",
        type=int,
        default=12,
        help="Minimum sticker area kept as a graph node",
    )
    parser.add_argument(
        "--draw-cube-grid-fit",
        action="store_true",
        help="Draw lines that fit per-sticker quadrilateral hull edges",
    )
    parser.add_argument(
        "--edge-line-width",
        "--empty-space-line-width",
        dest="empty_space_line_width",
        type=int,
        default=EMPTY_SPACE_LINE_WIDTH_PX,
        help="Pixel width used when drawing each fitted sticker-side line",
    )
    parser.add_argument(
        "--basis-angle-bucket-degrees",
        type=float,
        default=BASIS_VECTOR_ANGLE_BUCKET_DEGREES,
        help="Angle histogram bucket size used to derive the two displayed basis vectors",
    )
    parser.add_argument(
        "--cube-grid-fit-line-thickness",
        type=int,
        default=2,
        help="Line thickness used when scoring fitted cube-grid intersections with stickers",
    )
    parser.add_argument(
        "--cube-grid-fit-min-coverage",
        type=float,
        default=0.88,
        help="Minimum sticker-mask coverage expected inside the fitted cube hull",
    )
    parser.add_argument("--mask-opacity", type=float, default=1.0)
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
        help="Plot RGB coordinates normalized by RGB vector length",
    )
    parser.add_argument(
        "--hide-sticker-color-classification",
        dest="show_sticker_color_classification",
        action="store_false",
        help="Do not recolor sticker masks using density-cluster color classification",
    )
    parser.add_argument(
        "--sticker-color-frames",
        type=int,
        default=90,
        help="Detected frames of sticker color samples to collect before classifying stickers",
    )
    parser.add_argument(
        "--sticker-color-min-brightness",
        type=float,
        default=40.0,
        help="Minimum raw RGB vector length used while fitting density color clusters",
    )
    parser.add_argument(
        "--color-cluster-count",
        "--rgb-regression-clusters",
        dest="color_cluster_count",
        type=int,
        default=6,
        help="Target number of density color clusters to keep",
    )
    parser.add_argument(
        "--color-classifier-mode",
        choices=("density-cluster",),
        default="density-cluster",
        help="Color classification mode. Only density-cluster is supported.",
    )
    parser.add_argument(
        "--density-cluster-eps-radians",
        "--cluster-eps-radians",
        dest="density_cluster_eps_radians",
        type=float,
        default=DENSITY_CLUSTER_EPS_RADIANS,
        help="Angular radius for normalized RGB density clustering",
    )
    parser.add_argument(
        "--density-cluster-min-neighbors",
        type=int,
        default=DENSITY_CLUSTER_MIN_POINTS,
        help="Minimum normalized RGB points in the epsilon neighborhood for a density core point",
    )
    parser.add_argument(
        "--color-plot-frames",
        type=int,
        default=90,
        help="Detected frames of color samples to collect for plots; 0 disables color classification/recoloring.",
    )
    parser.add_argument("--color-plot-samples-per-frame", type=int, default=800)
    parser.add_argument("--average-plot-point-size", type=float, default=36.0)
    parser.add_argument(
        "--plotly-point-size",
        type=float,
        default=1.0,
        help="Marker size for interactive Plotly RGB/HSV plots",
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--save", type=Path)
    parser.add_argument(
        "--save-net",
        type=Path,
        help="Output PNG path for the latest cube net built from sticker ROI graphs",
    )
    parser.add_argument(
        "--net-cluster-faces",
        type=parse_net_cluster_faces,
        default={},
        help="Map learned color clusters to cube faces, for example 1=R,2=U",
    )
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--hide-bbox", dest="show_bbox", action="store_false")
    parser.add_argument("--hide-confidence", dest="show_confidence", action="store_false")
    parser.add_argument(
        "--debug-overlay",
        action="store_true",
        help="Draw ROI box, polygon outlines, labels, and FPS",
    )
    parser.add_argument(
        "--show-original-underlay",
        action="store_true",
        help="Draw sticker overlays on top of the original image instead of a black background",
    )
    parser.set_defaults(
        show_bbox=True,
        show_confidence=True,
        show_sticker_color_classification=True,
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
    sticker_model = YOLO(str(args.sticker_model))
    source = parse_source(args.source)
    plot_point_size = args.average_plot_point_size
    color_classifier = (
        ColorDensityClassifier(
            required_frames=args.sticker_color_frames,
            cluster_count=args.color_cluster_count,
            min_brightness=args.sticker_color_min_brightness,
            density_cluster_eps_radians=args.density_cluster_eps_radians,
            density_cluster_min_neighbors=args.density_cluster_min_neighbors,
        )
        if color_features_enabled(args)
        else None
    )
    color_plot_requested = color_plot_outputs_requested(args)
    color_plot_bgr_batches = []
    color_plot_detected_frames = 0
    cube_net_state = CubeNetState() if args.save_net else None
    roi_tracker = RoiTracker()

    if isinstance(source, str) and Path(source).suffix.lower() in {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    }:
        frame = cv2.imread(source)
        if frame is None:
            raise SystemExit(f"Could not read image: {source}")
        output, bgr_samples = process_frame(
            frame,
            bbox_detector,
            sticker_model,
            args,
            color_classifier,
            cube_net_state,
            roi_tracker,
        )
        if color_plot_requested:
            if len(bgr_samples) == 0:
                raise SystemExit(
                    "Could not collect color plot samples because no stickers were segmented"
                )
            if args.rgb_color_plot:
                save_rgb_samples_plot(
                    bgr_samples,
                    str(args.rgb_color_plot),
                    title="Sticker segmentation reliable pixels in RGB",
                    point_size=plot_point_size,
                    normalize_rgb=args.normalize_rgb_plot,
                )
            if args.rgb_html_plot:
                save_rgb_html_plots(
                    bgr_samples,
                    args.rgb_html_plot,
                    title="Sticker segmentation reliable pixels in RGB",
                    normalize_rgb=args.normalize_rgb_plot,
                    args=args,
                    color_classifier=color_classifier,
                )
            if args.hsv_html_plot:
                save_hsv_samples_interactive_plot(
                    bgr_samples,
                    str(args.hsv_html_plot),
                    title="Sticker segmentation reliable pixels in HSV",
                    point_size=args.plotly_point_size,
                )
        if args.save:
            cv2.imwrite(str(args.save), output)
        if args.save_net and cube_net_state is not None:
            args.save_net.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.save_net), draw_cube_net(cube_net_state))
        if args.display:
            cv2.imshow("cube sticker segmentations", output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return 0

    capture = cv2.VideoCapture(source)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    previous = perf_counter()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        output, bgr_samples = process_frame(
            frame,
            bbox_detector,
            sticker_model,
            args,
            color_classifier,
            cube_net_state,
            roi_tracker,
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
        if color_plot_requested:
            if color_plot_detected_frames < args.color_plot_frames and len(bgr_samples) > 0:
                color_plot_bgr_batches.append(bgr_samples)
                color_plot_detected_frames += 1
                if color_plot_detected_frames >= args.color_plot_frames and not args.display:
                    break
        if args.display:
            cv2.imshow("cube sticker segmentations", output)
            key = cv2.waitKey(0 if args.once else 1) & 0xFF
            if key == ord("q"):
                break
        if args.once:
            break

    if args.display:
        cv2.destroyAllWindows()
    capture.release()
    if args.save_net and cube_net_state is not None:
        args.save_net.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_net), draw_cube_net(cube_net_state))
    if color_plot_requested:
        if not color_plot_bgr_batches:
            raise SystemExit("Could not collect color plot samples from any segmented frames")
        bgr_samples = np.concatenate(color_plot_bgr_batches, axis=0)
        if args.rgb_color_plot:
            save_rgb_samples_plot(
                bgr_samples,
                str(args.rgb_color_plot),
                title=f"Sticker segmentation RGB samples across {color_plot_detected_frames} frames",
                point_size=plot_point_size,
                normalize_rgb=args.normalize_rgb_plot,
            )
        if args.rgb_html_plot:
            save_rgb_html_plots(
                bgr_samples,
                args.rgb_html_plot,
                title=f"Sticker segmentation RGB samples across {color_plot_detected_frames} frames",
                normalize_rgb=args.normalize_rgb_plot,
                args=args,
                color_classifier=color_classifier,
            )
        if args.hsv_html_plot:
            save_hsv_samples_interactive_plot(
                bgr_samples,
                str(args.hsv_html_plot),
                title=f"Sticker segmentation HSV samples across {color_plot_detected_frames} frames",
                point_size=args.plotly_point_size,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
