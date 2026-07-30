"""Test YOLO cube pose estimation on a cube bounding-box ROI.

Pipeline:
1. Detect the whole cube with the existing bbox detector.
2. Crop a padded square ROI around that bbox.
3. Run the independent corner/edge pose model on the ROI.
4. Draw predicted pose keypoints and the fitted 55mm cube.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from time import perf_counter

import __main__
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from live_recognition.detectors import YOLOCubeDetector  # noqa: E402
from live_recognition.visualization import _reliable_color_mask  # noqa: E402
from live_recognition.visualization import (save_hsv_samples_interactive_plot,
                                            save_rgb_samples_interactive_plot,
                                            save_rgb_samples_plot)

DEFAULT_BBOX_MODEL = ROOT / "runs/500ExamplesWeights.pt"
DEFAULT_STICKER_MODEL = (
    ROOT
    / "runs/segment/runs/cube_face_seg_local/runs/cube_face_sticker_yolo11n_seg/weights/best.pt"
)
DEFAULT_POSE_MODEL = (
    ROOT
    / "runs/pose/runs/cube_piece_pose_local/runs/cube_corner_edge_yolo11n_pose/weights/best.pt"
)
ENABLE_STICKER_SEGMENTATION = False
ENABLE_POSE_ESTIMATION = True
DRAW_POSE_KEYPOINTS = True
DRAW_PNP_CUBE = True
DISPLAY_WINDOW_NAME = "cube sticker segmentations"
DISPLAY_PREVIEW_BY_DEFAULT = True
SHOW_ORIGINAL_UNDERLAY_BY_DEFAULT = True
POSE_IMAGE_SIZE = 256
POSE_OBJECT_CONFIDENCE = 0.5
POSE_KEYPOINT_CONFIDENCE = 0.1
PNP_BOOTSTRAP_FRAME_COUNT = 12
PNP_FOCAL_SCALE_MIN = 0.7
PNP_FOCAL_SCALE_MAX = 1.8
PNP_FOCAL_SCALE_STEPS = 5
PNP_PRINCIPAL_POINT_OFFSET_RATIO = 0.0
PNP_MAX_REPROJECTION_ERROR_PX = 12.0
PNP_INLIER_THRESHOLD_PX = 10.0
PNP_HYPOTHESIS_POINT_COUNT = 3
PNP_HYPOTHESIS_ATTEMPT_LIMIT = 3
PNP_CORNER_DETECTION_LIMIT = 8
PNP_EDGE_DETECTION_LIMIT = 12
PNP_TEMPORAL_TRACKING_ENABLED = True
PNP_TEMPORAL_MAX_CORNER_SHIFT_PX = 18.0
DEFAULT_STICKER_MASK_COLOR = (150, 150, 150)
FACE_INSTANCE_MASK_COLORS = (
    (255, 80, 80),
    (80, 210, 120),
    (80, 150, 255),
    (255, 210, 80),
    (220, 120, 255),
    (80, 230, 230),
)
STICKER_CLASS_NAMES = {"face", "sticker", "u_sticker", "f_sticker", "r_sticker"}
STICKER_COLOR_SAMPLE_COUNT = 10
DENSITY_CLUSTER_EPS_RADIANS = 0.08
DENSITY_CLUSTER_MIN_POINTS = 4
CUBE_SIZE_MM = 55.0
CUBE_HALF_SIZE_MM = CUBE_SIZE_MM / 2.0
CUBE_CORNER_NAMES = ("UBL", "UBR", "UFR", "UFL", "DBL", "DBR", "DFR", "DFL")
CUBE_CORNER_POINTS_MM = np.array(
    [
        [-CUBE_HALF_SIZE_MM, CUBE_HALF_SIZE_MM, CUBE_HALF_SIZE_MM],
        [CUBE_HALF_SIZE_MM, CUBE_HALF_SIZE_MM, CUBE_HALF_SIZE_MM],
        [CUBE_HALF_SIZE_MM, -CUBE_HALF_SIZE_MM, CUBE_HALF_SIZE_MM],
        [-CUBE_HALF_SIZE_MM, -CUBE_HALF_SIZE_MM, CUBE_HALF_SIZE_MM],
        [-CUBE_HALF_SIZE_MM, CUBE_HALF_SIZE_MM, -CUBE_HALF_SIZE_MM],
        [CUBE_HALF_SIZE_MM, CUBE_HALF_SIZE_MM, -CUBE_HALF_SIZE_MM],
        [CUBE_HALF_SIZE_MM, -CUBE_HALF_SIZE_MM, -CUBE_HALF_SIZE_MM],
        [-CUBE_HALF_SIZE_MM, -CUBE_HALF_SIZE_MM, -CUBE_HALF_SIZE_MM],
    ],
    dtype=np.float32,
)
CUBE_CORNER_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
CUBE_FACES = (
    ((0, 1, 2, 3), (0.0, 0.0, 1.0)),
    ((4, 7, 6, 5), (0.0, 0.0, -1.0)),
    ((0, 4, 5, 1), (0.0, 1.0, 0.0)),
    ((3, 2, 6, 7), (0.0, -1.0, 0.0)),
    ((1, 5, 6, 2), (1.0, 0.0, 0.0)),
    ((0, 3, 7, 4), (-1.0, 0.0, 0.0)),
)


def cube_orientation_corner_permutations() -> tuple[tuple[int, ...], ...]:
    corner_lookup = {
        tuple(np.sign(point).astype(np.int8)): index
        for index, point in enumerate(CUBE_CORNER_POINTS_MM)
    }
    rotations = []
    for axis_order in permutations(range(3)):
        permutation_matrix = np.zeros((3, 3), dtype=np.int8)
        for output_axis, input_axis in enumerate(axis_order):
            permutation_matrix[output_axis, input_axis] = 1
        for signs in ((sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)):
            rotation = permutation_matrix * np.asarray(signs, dtype=np.int8)[:, None]
            if round(float(np.linalg.det(rotation))) != 1:
                continue
            rotations.append(
                tuple(
                    corner_lookup[tuple(rotation @ np.sign(point).astype(np.int8))]
                    for point in CUBE_CORNER_POINTS_MM
                )
            )
    return tuple(sorted(set(rotations)))


def cube_corner_correspondence_triples() -> tuple[tuple[int, int, int], ...]:
    orientations = cube_orientation_corner_permutations()
    representatives = set()
    for corner_triple in permutations(range(len(CUBE_CORNER_POINTS_MM)), PNP_HYPOTHESIS_POINT_COUNT):
        orbit = [
            tuple(orientation[index] for index in corner_triple)
            for orientation in orientations
        ]
        representatives.add(min(orbit))
    return tuple(sorted(representatives))


CUBE_ORIENTATION_CORNER_PERMUTATIONS = cube_orientation_corner_permutations()
CUBE_CORNER_CORRESPONDENCE_TRIPLES = cube_corner_correspondence_triples()
POSE_CLASS_NAMES = {"corner", "edge"}
POSE_KEYPOINT_COLORS = {
    "corner": (255, 255, 255),
    "edge": (0, 255, 255),
    "center": (255, 120, 255),
}
PNP_CUBE_LINE_COLOR = (255, 80, 30)
PNP_CUBE_POINT_COLOR = (30, 180, 255)


def register_training_checkpoint_classes() -> None:
    """Expose custom training classes where torch expects them during checkpoint loading."""
    training_path = ROOT / "scripts" / "train_yolo11n_piece_seg_local.py"
    spec = importlib.util.spec_from_file_location("cube_pose_training_checkpoint_classes", training_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load training classes from {training_path}")
    training = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = training
    spec.loader.exec_module(training)

    for name in (
        "PerAnchorKeypointLoss",
        "CubeSymmetricPoseLoss",
        "CubeSymmetricPoseModel",
        "CubeSymmetricPoseTrainer",
    ):
        setattr(__main__, name, getattr(training, name))


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


@dataclass
class CornerObservation:
    points: np.ndarray
    edge_points: np.ndarray


@dataclass
class CornerPnPResult:
    error: float
    inlier_count: int
    matched_error: float
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    image_points: np.ndarray
    scoring_points: np.ndarray
    scoring_edge_points: np.ndarray
    object_points: np.ndarray
    projected_corners: np.ndarray
    corner_inlier_count: int
    edge_inlier_count: int
    corner_names: tuple[str, ...]
    scoring_corner_indexes: tuple[int, ...]


@dataclass
class CameraBootstrapPnP:
    bootstrap_frames: int = 12
    focal_min: float = 0.7
    focal_max: float = 1.8
    focal_steps: int = 7
    principal_offset: float = 0.08
    reprojection_error: float = 12.0
    inlier_threshold: float = 10.0
    camera_matrix: np.ndarray | None = None
    observations: list[CornerObservation] | None = None
    last_result: CornerPnPResult | None = None

    def __post_init__(self) -> None:
        if self.observations is None:
            self.observations = []

    def solve(
        self,
        image_points: np.ndarray,
        image_size: tuple[int, int],
        *,
        edge_points: np.ndarray | None = None,
    ) -> CornerPnPResult | None:
        if len(image_points) < 3:
            self.last_result = None
            return None

        observations = self.observations if self.observations is not None else []
        if self.camera_matrix is not None and len(observations) >= self.bootstrap_frames:
            result = solve_temporal_or_best_corner_pnp(
                image_points,
                [self.camera_matrix],
                previous_result=self.last_result,
                scoring_edge_points=edge_points,
                inlier_threshold=self.inlier_threshold,
            )
            self.last_result = result
            return result

        candidates = camera_matrix_candidates(
            image_size,
            focal_min=self.focal_min,
            focal_max=self.focal_max,
            focal_steps=self.focal_steps,
            principal_offset=self.principal_offset,
        )
        result = solve_temporal_or_best_corner_pnp(
            image_points,
            candidates,
            previous_result=self.last_result,
            scoring_edge_points=edge_points,
            inlier_threshold=self.inlier_threshold,
        )
        if result is None or result.inlier_count < 3 or result.error > self.reprojection_error:
            self.last_result = None
            return result

        observations.append(
            CornerObservation(
                points=np.asarray(image_points, dtype=np.float32),
                edge_points=(
                    np.empty((0, 2), dtype=np.float32)
                    if edge_points is None
                    else np.asarray(edge_points, dtype=np.float32).reshape(-1, 2)
                ),
            )
        )
        del observations[: max(0, len(observations) - self.bootstrap_frames)]

        best_matrix = best_camera_matrix_for_observations(
            observations,
            candidates,
            inlier_threshold=self.inlier_threshold,
        )
        if best_matrix is not None:
            self.camera_matrix = best_matrix
            result = (
                solve_temporal_or_best_corner_pnp(
                    image_points,
                    [best_matrix],
                    previous_result=result,
                    scoring_edge_points=edge_points,
                    inlier_threshold=self.inlier_threshold,
                )
                or result
            )
        self.last_result = result
        return result

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


def pose_keypoint_kind(class_name: str, index: int) -> str:
    if class_name == "corner" and index == 0:
        return "corner"
    if class_name == "edge" and index < 2:
        return "edge"
    if class_name == "corner" and index < 4:
        return "edge"
    return "center"


def result_keypoints(result, size: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None:
        return None, None

    xy = tensor_to_numpy(getattr(keypoints, "xy", None))
    conf = tensor_to_numpy(getattr(keypoints, "conf", None))
    if xy is not None and xy.size:
        return np.asarray(xy, dtype=np.float32), None if conf is None else np.asarray(conf, dtype=np.float32)

    xyn = tensor_to_numpy(getattr(keypoints, "xyn", None))
    if xyn is not None and xyn.size:
        xy = np.asarray(xyn, dtype=np.float32).copy()
        xy[..., 0] *= size
        xy[..., 1] *= size
        return xy, None if conf is None else np.asarray(conf, dtype=np.float32)

    data = tensor_to_numpy(getattr(keypoints, "data", None))
    if data is None or not data.size:
        return None, None
    data = np.asarray(data, dtype=np.float32)
    if data.shape[-1] < 2:
        return None, None
    xy = data[..., :2]
    conf = data[..., 2] if data.shape[-1] >= 3 else None
    return xy, conf


def result_box_classes(result) -> np.ndarray | None:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return None
    classes = tensor_to_numpy(getattr(boxes, "cls", None))
    if classes is None:
        return None
    return np.asarray(classes, dtype=np.int32).reshape(-1)


def collect_corner_image_points(
    result,
    pose_model,
    xy: np.ndarray | None,
    keypoint_conf: np.ndarray | None,
    *,
    confidence_threshold: float,
    offset_xy: tuple[int, int],
    scale_xy: tuple[float, float],
    point_limit: int,
) -> np.ndarray:
    if xy is None or len(xy) == 0:
        return np.empty((0, 2), dtype=np.float32)

    classes = result_box_classes(result)
    points_by_detection = np.asarray(xy, dtype=np.float32)
    scores_by_detection = (
        None
        if keypoint_conf is None or len(keypoint_conf) == 0
        else np.asarray(keypoint_conf, dtype=np.float32)
    )
    offset_x, offset_y = offset_xy
    scale_x, scale_y = scale_xy
    candidates: list[tuple[float, float, float]] = []
    for detection_index, points in enumerate(points_by_detection):
        class_id = (
            int(classes[detection_index])
            if classes is not None and detection_index < len(classes)
            else -1
        )
        class_name = class_name_for_id(result, pose_model, class_id).removesuffix("_sticker")
        if class_name != "corner" or len(points) == 0:
            continue

        detection_scores = (
            None
            if scores_by_detection is None
            or detection_index >= len(scores_by_detection)
            else np.asarray(scores_by_detection[detection_index], dtype=np.float32).reshape(-1)
        )
        keypoint_score = (
            1.0
            if detection_scores is None or len(detection_scores) == 0
            else float(detection_scores[0])
        )
        if keypoint_score < confidence_threshold:
            continue

        point = points[0]
        candidates.append(
            (
                offset_x + (float(point[0]) * scale_x),
                offset_y + (float(point[1]) * scale_y),
                keypoint_score,
            )
        )

    candidates.sort(key=lambda item: item[2], reverse=True)
    return np.asarray([(x, y) for x, y, _score in candidates[:point_limit]], dtype=np.float32)


def collect_pose_image_points(
    result,
    pose_model,
    xy: np.ndarray | None,
    keypoint_conf: np.ndarray | None,
    *,
    accepted_kinds: set[str],
    confidence_threshold: float,
    offset_xy: tuple[int, int],
    scale_xy: tuple[float, float],
    point_limit: int | None = None,
) -> np.ndarray:
    if xy is None or len(xy) == 0:
        return np.empty((0, 2), dtype=np.float32)

    classes = result_box_classes(result)
    points_by_detection = np.asarray(xy, dtype=np.float32)
    scores_by_detection = (
        None
        if keypoint_conf is None or len(keypoint_conf) == 0
        else np.asarray(keypoint_conf, dtype=np.float32)
    )
    offset_x, offset_y = offset_xy
    scale_x, scale_y = scale_xy
    candidates: list[tuple[float, float, float]] = []
    for detection_index, points in enumerate(points_by_detection):
        class_id = (
            int(classes[detection_index])
            if classes is not None and detection_index < len(classes)
            else -1
        )
        class_name = class_name_for_id(result, pose_model, class_id).removesuffix("_sticker")
        if class_name not in POSE_CLASS_NAMES:
            continue

        detection_scores = (
            None
            if scores_by_detection is None
            or detection_index >= len(scores_by_detection)
            else np.asarray(scores_by_detection[detection_index], dtype=np.float32).reshape(-1)
        )
        for keypoint_index, point in enumerate(points):
            keypoint_score = (
                1.0
                if detection_scores is None or keypoint_index >= len(detection_scores)
                else float(detection_scores[keypoint_index])
            )
            if keypoint_score < confidence_threshold:
                continue
            if pose_keypoint_kind(class_name, keypoint_index) not in accepted_kinds:
                continue
            candidates.append(
                (
                    offset_x + (float(point[0]) * scale_x),
                    offset_y + (float(point[1]) * scale_y),
                    keypoint_score,
                )
            )

    candidates.sort(key=lambda item: item[2], reverse=True)
    if point_limit is not None:
        candidates = candidates[:point_limit]
    return np.asarray([(x, y) for x, y, _score in candidates], dtype=np.float32)


def point_bounds_xyxy(points: np.ndarray, roi_xyxy: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0:
        return None

    x1, y1, x2, y2 = roi_xyxy
    local_x = np.clip(points[:, 0] - x1, 0, max(0, x2 - x1))
    local_y = np.clip(points[:, 1] - y1, 0, max(0, y2 - y1))
    return (
        int(np.floor(local_x.min())),
        int(np.floor(local_y.min())),
        int(np.ceil(local_x.max())),
        int(np.ceil(local_y.max())),
    )


def default_camera_matrix(
    image_size: tuple[int, int],
    *,
    focal_scale: float = 1.2,
    principal_offset_xy: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    width, height = image_size
    focal = max(width, height) * float(focal_scale)
    cx = (width * 0.5) + (width * float(principal_offset_xy[0]))
    cy = (height * 0.5) + (height * float(principal_offset_xy[1]))
    return np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def camera_matrix_candidates(
    image_size: tuple[int, int],
    *,
    focal_min: float,
    focal_max: float,
    focal_steps: int,
    principal_offset: float,
) -> list[np.ndarray]:
    focal_steps = max(1, int(focal_steps))
    focal_scales = np.linspace(float(focal_min), float(focal_max), focal_steps)
    offsets = (-principal_offset, 0.0, principal_offset) if principal_offset > 0.0 else (0.0,)
    return [
        default_camera_matrix(
            image_size,
            focal_scale=float(focal_scale),
            principal_offset_xy=(float(offset_x), float(offset_y)),
        )
        for focal_scale in focal_scales
        for offset_x in offsets
        for offset_y in offsets
    ]


def visible_cube_corner_and_edge_masks(
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
    visible_corners = np.zeros(len(CUBE_CORNER_POINTS_MM), dtype=bool)
    visible_edges = np.zeros(len(CUBE_CORNER_EDGES), dtype=bool)
    edge_lookup = {tuple(sorted(edge)): index for index, edge in enumerate(CUBE_CORNER_EDGES)}

    for face_corners, face_normal in CUBE_FACES:
        face_points = CUBE_CORNER_POINTS_MM[list(face_corners)].astype(np.float64)
        face_center_camera = rotation @ face_points.mean(axis=0) + tvec
        normal_camera = rotation @ np.asarray(face_normal, dtype=np.float64)
        if float(np.dot(normal_camera, -face_center_camera)) <= 0.0:
            continue

        visible_corners[list(face_corners)] = True
        face_cycle = list(face_corners)
        for start, end in zip(face_cycle, face_cycle[1:] + face_cycle[:1]):
            edge_index = edge_lookup.get(tuple(sorted((start, end))))
            if edge_index is not None:
                visible_edges[edge_index] = True

    return visible_corners, visible_edges


def solve_corner_pnp(
    image_points: np.ndarray,
    object_points: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    scoring_points: np.ndarray | None = None,
    scoring_edge_points: np.ndarray | None = None,
    inlier_threshold: float,
) -> CornerPnPResult | None:
    image_points = np.asarray(image_points, dtype=np.float32)
    object_points = np.asarray(object_points, dtype=np.float32)
    if len(image_points) != len(object_points) or len(image_points) < 3:
        return None

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    try:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_SQPNP,
        )
    except cv2.error:
        return None
    if not ok:
        return None

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    projected = projected.reshape(-1, 2)
    matched_error = float(np.mean(np.linalg.norm(projected - image_points, axis=1)))
    projected_corners, _ = cv2.projectPoints(
        CUBE_CORNER_POINTS_MM,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    projected_corners = projected_corners.reshape(-1, 2)
    visible_corner_mask, visible_edge_mask = visible_cube_corner_and_edge_masks(rvec, tvec)
    scoring_points = image_points if scoring_points is None else np.asarray(scoring_points, dtype=np.float32)
    corner_error, corner_inlier_count, scoring_corner_indexes = robust_projected_corner_assignment(
        scoring_points,
        projected_corners,
        allowed_corner_mask=visible_corner_mask,
        inlier_threshold=inlier_threshold,
    )
    scoring_edge_points = (
        np.empty((0, 2), dtype=np.float32)
        if scoring_edge_points is None
        else np.asarray(scoring_edge_points, dtype=np.float32).reshape(-1, 2)
    )
    edge_error, edge_inlier_count = robust_projected_edge_score(
        scoring_edge_points,
        projected_corners,
        allowed_edge_mask=visible_edge_mask,
        inlier_threshold=inlier_threshold,
    )
    inlier_count = corner_inlier_count + edge_inlier_count
    inlier_errors = []
    if corner_inlier_count > 0:
        inlier_errors.append(corner_error)
    if edge_inlier_count > 0:
        inlier_errors.append(edge_error)
    error = float(np.mean(inlier_errors)) if inlier_errors else float("inf")
    return CornerPnPResult(
        error=error,
        inlier_count=inlier_count,
        matched_error=matched_error,
        camera_matrix=np.asarray(camera_matrix, dtype=np.float64),
        dist_coeffs=dist_coeffs,
        rvec=rvec,
        tvec=tvec,
        image_points=image_points,
        scoring_points=scoring_points,
        scoring_edge_points=scoring_edge_points,
        object_points=object_points,
        projected_corners=projected_corners,
        corner_inlier_count=corner_inlier_count,
        edge_inlier_count=edge_inlier_count,
        corner_names=(),
        scoring_corner_indexes=scoring_corner_indexes,
    )


def robust_projected_corner_assignment(
    image_points: np.ndarray,
    projected_corners: np.ndarray,
    *,
    allowed_corner_mask: np.ndarray | None = None,
    inlier_threshold: float,
) -> tuple[float, int, tuple[int, ...]]:
    points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    projected = np.asarray(projected_corners, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0 or len(projected) == 0 or len(points) > len(projected):
        return float("inf"), 0, tuple()
    if allowed_corner_mask is None:
        allowed_corner_mask = np.ones(len(projected), dtype=bool)
    allowed_corner_mask = np.asarray(allowed_corner_mask, dtype=bool).reshape(-1)
    if len(allowed_corner_mask) != len(projected):
        raise ValueError("allowed_corner_mask must match projected corner count")

    distances = np.linalg.norm(points[:, None, :] - projected[None, :, :], axis=2)
    inlier_threshold = float(inlier_threshold)
    best_by_mask: dict[int, tuple[int, float, tuple[int, ...]]] = {0: (0, 0.0, tuple())}
    for point_index in range(len(points)):
        next_by_mask = {
            mask: (count, total_error, assignment + (-1,))
            for mask, (count, total_error, assignment) in best_by_mask.items()
        }
        for mask, (count, total_error, assignment) in best_by_mask.items():
            for corner_index in range(len(projected)):
                if not bool(allowed_corner_mask[corner_index]):
                    continue
                if mask & (1 << corner_index):
                    continue
                distance = float(distances[point_index, corner_index])
                if distance > inlier_threshold:
                    continue
                next_mask = mask | (1 << corner_index)
                candidate = (count + 1, total_error + distance, assignment + (corner_index,))
                existing = next_by_mask.get(next_mask)
                if existing is None or candidate[0] > existing[0] or (
                    candidate[0] == existing[0] and candidate[1] < existing[1]
                ):
                    next_by_mask[next_mask] = candidate
        best_by_mask = next_by_mask

    best_inlier_count, best_total_error, best_assignment = max(
        best_by_mask.values(),
        key=lambda item: (item[0], -item[1]),
    )
    if best_inlier_count == 0:
        return float("inf"), 0, best_assignment
    return best_total_error / best_inlier_count, best_inlier_count, best_assignment


def robust_projected_corner_score(
    image_points: np.ndarray,
    projected_corners: np.ndarray,
    *,
    allowed_corner_mask: np.ndarray | None = None,
    inlier_threshold: float,
) -> tuple[float, int]:
    error, inlier_count, _assignment = robust_projected_corner_assignment(
        image_points,
        projected_corners,
        allowed_corner_mask=allowed_corner_mask,
        inlier_threshold=inlier_threshold,
    )
    return error, inlier_count


def point_to_segments_distances(points: np.ndarray, segments: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    segments = np.asarray(segments, dtype=np.float32).reshape(-1, 2, 2)
    if len(points) == 0 or len(segments) == 0:
        return np.empty((len(points), len(segments)), dtype=np.float32)

    starts = segments[:, 0, :]
    ends = segments[:, 1, :]
    vectors = ends - starts
    lengths_sq = np.maximum(np.sum(vectors * vectors, axis=1), 1e-6)
    point_vectors = points[:, None, :] - starts[None, :, :]
    t = np.sum(point_vectors * vectors[None, :, :], axis=2) / lengths_sq[None, :]
    t = np.clip(t, 0.0, 1.0)
    closest = starts[None, :, :] + t[:, :, None] * vectors[None, :, :]
    return np.linalg.norm(points[:, None, :] - closest, axis=2)


def robust_projected_edge_score(
    image_points: np.ndarray,
    projected_corners: np.ndarray,
    *,
    allowed_edge_mask: np.ndarray | None = None,
    inlier_threshold: float,
) -> tuple[float, int]:
    points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    projected = np.asarray(projected_corners, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0 or len(projected) == 0:
        return float("inf"), 0

    edge_segments = np.asarray(
        [(projected[start], projected[end]) for start, end in CUBE_CORNER_EDGES],
        dtype=np.float32,
    )
    if allowed_edge_mask is not None:
        allowed_edge_mask = np.asarray(allowed_edge_mask, dtype=bool).reshape(-1)
        if len(allowed_edge_mask) != len(edge_segments):
            raise ValueError("allowed_edge_mask must match cube edge count")
        edge_segments = edge_segments[allowed_edge_mask]
        if len(edge_segments) == 0:
            return float("inf"), 0
    distances = point_to_segments_distances(points, edge_segments)
    nearest_distances = distances.min(axis=1)
    inlier_distances = nearest_distances[nearest_distances <= float(inlier_threshold)]
    if len(inlier_distances) == 0:
        return float("inf"), 0
    return float(np.mean(inlier_distances)), int(len(inlier_distances))


def pnp_result_is_better(candidate: CornerPnPResult, best: CornerPnPResult | None) -> bool:
    if best is None:
        return True
    if candidate.inlier_count != best.inlier_count:
        return candidate.inlier_count > best.inlier_count
    if not np.isclose(candidate.error, best.error):
        return candidate.error < best.error
    return candidate.matched_error < best.matched_error


def refine_pnp_with_scored_corner_correspondences(
    result: CornerPnPResult,
    *,
    inlier_threshold: float,
) -> CornerPnPResult:
    scoring_points = np.asarray(result.scoring_points, dtype=np.float32).reshape(-1, 2)
    corner_indexes = np.asarray(result.scoring_corner_indexes, dtype=np.int32)
    if len(scoring_points) == 0 or len(corner_indexes) != len(scoring_points):
        return result

    inlier_mask = corner_indexes >= 0
    if int(inlier_mask.sum()) < 3:
        return result

    refined = solve_corner_pnp(
        scoring_points[inlier_mask],
        CUBE_CORNER_POINTS_MM[corner_indexes[inlier_mask]],
        result.camera_matrix,
        scoring_points=scoring_points,
        scoring_edge_points=result.scoring_edge_points,
        inlier_threshold=inlier_threshold,
    )
    if refined is None:
        return result
    refined.corner_names = tuple(CUBE_CORNER_NAMES[index] for index in corner_indexes[inlier_mask])
    return refined


def temporal_corner_assignment_from_previous_projection(
    image_points: np.ndarray,
    previous_result: CornerPnPResult | None,
    *,
    inlier_threshold: float,
) -> tuple[int, ...]:
    points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    if previous_result is None or len(points) < 3:
        return tuple()

    visible_corner_mask, _visible_edge_mask = visible_cube_corner_and_edge_masks(
        previous_result.rvec,
        previous_result.tvec,
    )
    _error, inlier_count, assignment = robust_projected_corner_assignment(
        points,
        previous_result.projected_corners,
        allowed_corner_mask=visible_corner_mask,
        inlier_threshold=inlier_threshold,
    )
    required_inliers = min(
        len(points),
        max(3, int(previous_result.corner_inlier_count)),
    )
    if inlier_count < required_inliers:
        return tuple()
    return assignment


def solve_temporal_corner_pnp(
    image_points: np.ndarray,
    previous_result: CornerPnPResult | None,
    *,
    scoring_edge_points: np.ndarray | None = None,
    inlier_threshold: float,
    max_corner_shift_px: float = PNP_TEMPORAL_MAX_CORNER_SHIFT_PX,
) -> CornerPnPResult | None:
    scoring_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    if previous_result is None or previous_result.camera_matrix is None:
        return None

    corner_indexes = np.asarray(
        temporal_corner_assignment_from_previous_projection(
            scoring_points,
            previous_result,
            inlier_threshold=max_corner_shift_px,
        ),
        dtype=np.int32,
    )
    if len(corner_indexes) != len(scoring_points):
        return None

    inlier_mask = corner_indexes >= 0
    if int(inlier_mask.sum()) < 3:
        return None

    result = solve_corner_pnp(
        scoring_points[inlier_mask],
        CUBE_CORNER_POINTS_MM[corner_indexes[inlier_mask]],
        previous_result.camera_matrix,
        scoring_points=scoring_points,
        scoring_edge_points=scoring_edge_points,
        inlier_threshold=inlier_threshold,
    )
    if result is None:
        return None
    result.corner_names = tuple(CUBE_CORNER_NAMES[index] for index in corner_indexes[inlier_mask])
    result = refine_pnp_with_scored_corner_correspondences(
        result,
        inlier_threshold=inlier_threshold,
    )
    if result.corner_inlier_count < len(scoring_points):
        return None
    return result


def pnp_hypothesis_point_sets(
    points: np.ndarray,
    *,
    hypothesis_point_count: int = PNP_HYPOTHESIS_POINT_COUNT,
    attempt_limit: int = PNP_HYPOTHESIS_ATTEMPT_LIMIT,
) -> list[np.ndarray]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) < hypothesis_point_count or attempt_limit <= 0:
        return []

    base_indexes = tuple(range(hypothesis_point_count))
    index_sets = [base_indexes]
    replacement_index = hypothesis_point_count
    while len(index_sets) < attempt_limit and replacement_index < len(points):
        index_sets.append(tuple(range(hypothesis_point_count - 1)) + (replacement_index,))
        replacement_index += 1
    return [points[list(indexes)] for indexes in index_sets]


def solve_best_corner_pnp(
    image_points: np.ndarray,
    camera_matrices: list[np.ndarray],
    *,
    scoring_edge_points: np.ndarray | None = None,
    hypothesis_point_count: int = PNP_HYPOTHESIS_POINT_COUNT,
    inlier_threshold: float = 10.0,
) -> CornerPnPResult | None:
    scoring_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    if len(scoring_points) < hypothesis_point_count:
        return None
    hypothesis_point_sets = pnp_hypothesis_point_sets(
        scoring_points,
        hypothesis_point_count=hypothesis_point_count,
        attempt_limit=PNP_HYPOTHESIS_ATTEMPT_LIMIT,
    )

    best: CornerPnPResult | None = None
    for camera_matrix in camera_matrices:
        for hypothesis_points in hypothesis_point_sets:
            for corner_indexes in CUBE_CORNER_CORRESPONDENCE_TRIPLES:
                object_points = CUBE_CORNER_POINTS_MM[list(corner_indexes)]
                result = solve_corner_pnp(
                    hypothesis_points,
                    object_points,
                    camera_matrix,
                    scoring_points=scoring_points,
                    scoring_edge_points=scoring_edge_points,
                    inlier_threshold=inlier_threshold,
                )
                if result is None:
                    continue
                result.corner_names = tuple(CUBE_CORNER_NAMES[index] for index in corner_indexes)
                result = refine_pnp_with_scored_corner_correspondences(
                    result,
                    inlier_threshold=inlier_threshold,
                )
                if pnp_result_is_better(result, best):
                    best = result
    if best is None or best.corner_inlier_count < len(scoring_points):
        return None
    return best


def solve_temporal_or_best_corner_pnp(
    image_points: np.ndarray,
    camera_matrices: list[np.ndarray],
    *,
    previous_result: CornerPnPResult | None,
    scoring_edge_points: np.ndarray | None = None,
    hypothesis_point_count: int = PNP_HYPOTHESIS_POINT_COUNT,
    inlier_threshold: float = 10.0,
) -> CornerPnPResult | None:
    if PNP_TEMPORAL_TRACKING_ENABLED and previous_result is not None:
        temporal_result = solve_temporal_corner_pnp(
            image_points,
            previous_result,
            scoring_edge_points=scoring_edge_points,
            inlier_threshold=inlier_threshold,
        )
        if temporal_result is not None:
            return temporal_result

    return solve_best_corner_pnp(
        image_points,
        camera_matrices,
        scoring_edge_points=scoring_edge_points,
        hypothesis_point_count=hypothesis_point_count,
        inlier_threshold=inlier_threshold,
    )


def best_camera_matrix_for_observations(
    observations: list[CornerObservation],
    camera_matrices: list[np.ndarray],
    *,
    inlier_threshold: float,
) -> np.ndarray | None:
    if not observations:
        return None

    best_matrix: np.ndarray | None = None
    best_inliers = -1
    best_error = float("inf")
    for camera_matrix in camera_matrices:
        errors = []
        inliers = 0
        for observation in observations:
            result = solve_best_corner_pnp(
                observation.points,
                [camera_matrix],
                scoring_edge_points=observation.edge_points,
                inlier_threshold=inlier_threshold,
            )
            if result is not None:
                errors.append(result.error)
                inliers += result.inlier_count
        if not errors:
            continue
        mean_error = float(np.mean(errors))
        if inliers > best_inliers or (inliers == best_inliers and mean_error < best_error):
            best_inliers = inliers
            best_error = mean_error
            best_matrix = camera_matrix
    return best_matrix


def draw_pose_keypoints(
    frame: np.ndarray,
    result,
    pose_model,
    xy: np.ndarray | None,
    keypoint_conf: np.ndarray | None,
    *,
    confidence_threshold: float,
    offset_xy: tuple[int, int],
    scale_xy: tuple[float, float],
) -> int:
    if xy is None or len(xy) == 0:
        return 0

    classes = result_box_classes(result)
    points_by_detection = np.asarray(xy, dtype=np.float32)
    scores_by_detection = (
        None
        if keypoint_conf is None or len(keypoint_conf) == 0
        else np.asarray(keypoint_conf, dtype=np.float32)
    )
    offset_x, offset_y = offset_xy
    scale_x, scale_y = scale_xy
    drawn = 0
    for detection_index, points in enumerate(points_by_detection):
        class_id = (
            int(classes[detection_index])
            if classes is not None and detection_index < len(classes)
            else -1
        )
        class_name = class_name_for_id(result, pose_model, class_id).removesuffix("_sticker")
        if class_name not in POSE_CLASS_NAMES:
            continue

        detection_scores = (
            None
            if scores_by_detection is None
            or detection_index >= len(scores_by_detection)
            else np.asarray(scores_by_detection[detection_index], dtype=np.float32).reshape(-1)
        )
        for keypoint_index, point in enumerate(points):
            keypoint_score = (
                None
                if detection_scores is None or keypoint_index >= len(detection_scores)
                else float(detection_scores[keypoint_index])
            )
            if keypoint_score is not None and keypoint_score < confidence_threshold:
                continue

            x = int(round(offset_x + (float(point[0]) * scale_x)))
            y = int(round(offset_y + (float(point[1]) * scale_y)))
            if not (0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]):
                continue

            kind = pose_keypoint_kind(class_name, keypoint_index)
            color = POSE_KEYPOINT_COLORS.get(kind, (255, 255, 255))
            cv2.circle(frame, (x, y), 4, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (x, y), 5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            drawn += 1
    return drawn


def draw_pnp_cube(frame: np.ndarray, result: CornerPnPResult | None) -> None:
    if result is None:
        return

    projected = np.asarray(result.projected_corners, dtype=np.float32)
    for start, end in CUBE_CORNER_EDGES:
        start_point = tuple(np.round(projected[start]).astype(int))
        end_point = tuple(np.round(projected[end]).astype(int))
        cv2.line(frame, start_point, end_point, (0, 0, 0), 4, lineType=cv2.LINE_AA)
        cv2.line(frame, start_point, end_point, PNP_CUBE_LINE_COLOR, 2, lineType=cv2.LINE_AA)
    for point in projected:
        cv2.circle(frame, tuple(np.round(point).astype(int)), 4, PNP_CUBE_POINT_COLOR, -1, lineType=cv2.LINE_AA)
    for point in result.image_points:
        cv2.circle(frame, tuple(np.round(point).astype(int)), 6, (255, 255, 255), 1, lineType=cv2.LINE_AA)


def draw_pose_estimate(
    frame: np.ndarray,
    roi: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
    pose_model,
    pose_solver: CameraBootstrapPnP | None,
    args,
) -> tuple[int, CornerPnPResult | None, tuple[int, int, int, int] | None]:
    if pose_model is None:
        return 0, None, None

    x1, y1, x2, y2 = roi_xyxy
    model_roi = resize_to_square_size(roi, POSE_IMAGE_SIZE)
    results = pose_model.predict(
        source=model_roi,
        conf=POSE_OBJECT_CONFIDENCE,
        imgsz=POSE_IMAGE_SIZE,
        verbose=False,
    )
    if not results:
        return 0, None, None

    result = results[0]
    xy, conf = result_keypoints(result, POSE_IMAGE_SIZE)
    scale_x = (x2 - x1) / float(model_roi.shape[1])
    scale_y = (y2 - y1) / float(model_roi.shape[0])
    point_count = 0
    if DRAW_POSE_KEYPOINTS:
        point_count = draw_pose_keypoints(
            frame,
            result,
            pose_model,
            xy,
            conf,
            confidence_threshold=POSE_KEYPOINT_CONFIDENCE,
            offset_xy=(x1, y1),
            scale_xy=(scale_x, scale_y),
        )

    corner_points = collect_corner_image_points(
        result,
        pose_model,
        xy,
        conf,
        confidence_threshold=POSE_KEYPOINT_CONFIDENCE,
        offset_xy=(x1, y1),
        scale_xy=(scale_x, scale_y),
        point_limit=PNP_CORNER_DETECTION_LIMIT,
    )
    edge_points = collect_pose_image_points(
        result,
        pose_model,
        xy,
        conf,
        accepted_kinds={"edge"},
        confidence_threshold=POSE_KEYPOINT_CONFIDENCE,
        offset_xy=(x1, y1),
        scale_xy=(scale_x, scale_y),
        point_limit=PNP_EDGE_DETECTION_LIMIT,
    )
    visible_points = (
        np.concatenate([corner_points, edge_points], axis=0)
        if len(corner_points) and len(edge_points)
        else corner_points if len(corner_points) else edge_points
    )
    pose_tracking_bounds_xyxy = point_bounds_xyxy(visible_points, roi_xyxy)
    pnp_result = None
    if ENABLE_POSE_ESTIMATION and pose_solver is not None:
        pnp_result = pose_solver.solve(
            corner_points,
            (frame.shape[1], frame.shape[0]),
            edge_points=edge_points,
        )
        if DRAW_PNP_CUBE:
            draw_pnp_cube(frame, pnp_result)
    return point_count, pnp_result, pose_tracking_bounds_xyxy


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
) -> tuple[int, np.ndarray, tuple[int, int, int, int] | None]:
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
        return 0, np.empty((0, 3), dtype=np.uint8), None

    result = results[0]
    boxes = result.boxes
    if boxes is None:
        return 0, np.empty((0, 3), dtype=np.uint8), None

    overlay = np.zeros_like(frame)
    roi_height = y2 - y1
    roi_width = x2 - x1
    result_masks = getattr(result, "masks", None)
    if result_masks is None:
        return 0, np.empty((0, 3), dtype=np.uint8), None
    mask_data = result_masks.data.detach().cpu().numpy()

    count = 0
    confidence_labels = []
    sticker_color_numbers = []
    classifier_bgr_batches = []
    color_samples = roi_color_samples(roi)
    tracking_min_x: int | None = None
    tracking_min_y: int | None = None
    tracking_max_x: int | None = None
    tracking_max_y: int | None = None

    for index in range(len(boxes)):
        class_id = int(boxes.cls[index].item())
        class_name = class_name_for_id(result, sticker_model, class_id)
        if class_name not in STICKER_CLASS_NAMES:
            continue
        color = mask_color_for_segment(class_name, count)

        mask_bool = sticker_mask_bool(mask_data, index, roi_width, roi_height)
        if not mask_bool.any():
            continue

        sample_mask = apply_mask_padding(mask_bool, args.segmentation_padding_px)
        if not sample_mask.any():
            sample_mask = mask_bool
        draw_mask = sample_mask

        ys, xs = np.nonzero(mask_bool)
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
        raw_stickers = [(draw_mask, sample_bgr)] if len(sample_bgr) else [
            (draw_mask, np.empty((0, 3), dtype=np.uint8))
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

        sticker_bgr_batches = [
            sticker_bgr_samples
            for _, sticker_bgr_samples, _ in display_stickers
            if len(sticker_bgr_samples) > 0
        ]
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
    return count, bgr_samples, tracking_bounds_xyxy


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
    roi_tracker: RoiTracker | None = None,
    pose_model=None,
    pose_solver: CameraBootstrapPnP | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    output = frame.copy()
    roi_xyxy, candidate, roi_source = roi_for_frame(frame, bbox_detector, args, roi_tracker)
    if roi_xyxy is None:
        return output, np.empty((0, 3), dtype=np.uint8)

    x1, y1, x2, y2 = roi_xyxy
    roi = frame[y1:y2, x1:x2]
    segment_count = 0
    bgr_samples = np.empty((0, 3), dtype=np.uint8)
    tracking_bounds_xyxy = None
    if getattr(args, "enable_sticker_segmentation", ENABLE_STICKER_SEGMENTATION) and sticker_model is not None:
        segment_count, bgr_samples, tracking_bounds_xyxy = draw_sticker_segments(
            output,
            roi,
            sticker_model,
            roi_xyxy,
            args,
            color_classifier,
        )
    pose_keypoint_count = 0
    pnp_result = None
    if pose_model is not None:
        if ENABLE_POSE_ESTIMATION and pose_solver is None:
            pose_solver = CameraBootstrapPnP(
                bootstrap_frames=PNP_BOOTSTRAP_FRAME_COUNT,
                focal_min=PNP_FOCAL_SCALE_MIN,
                focal_max=PNP_FOCAL_SCALE_MAX,
                focal_steps=PNP_FOCAL_SCALE_STEPS,
                principal_offset=PNP_PRINCIPAL_POINT_OFFSET_RATIO,
                reprojection_error=PNP_MAX_REPROJECTION_ERROR_PX,
                inlier_threshold=PNP_INLIER_THRESHOLD_PX,
            )
        pose_keypoint_count, pnp_result, pose_tracking_bounds_xyxy = draw_pose_estimate(
            output,
            roi,
            roi_xyxy,
            pose_model,
            pose_solver,
            args,
        )
        if not getattr(args, "enable_sticker_segmentation", ENABLE_STICKER_SEGMENTATION):
            tracking_bounds_xyxy = pose_tracking_bounds_xyxy
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
        pnp_label = "" if pnp_result is None else f" pnp {pnp_result.error:.1f}px"
        debug_label = (
            f"cube {candidate.confidence:.2f} stickers {segment_count} keypoints {pose_keypoint_count}{pnp_label}"
            if candidate is not None
            else f"sticker viewport stickers {segment_count} keypoints {pose_keypoint_count}{pnp_label}"
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
    parser.add_argument("--display", dest="display", action="store_true")
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.add_argument("--hide-bbox", dest="show_bbox", action="store_false")
    parser.add_argument("--hide-confidence", dest="show_confidence", action="store_false")
    parser.add_argument(
        "--debug-overlay",
        action="store_true",
        help="Draw ROI box, polygon outlines, labels, and FPS",
    )
    parser.add_argument(
        "--show-original-underlay",
        dest="show_original_underlay",
        action="store_true",
        help="Draw sticker overlays on top of the original image instead of a black background",
    )
    parser.add_argument(
        "--hide-original-underlay",
        dest="show_original_underlay",
        action="store_false",
        help="Draw only the sticker overlay on a black background",
    )
    parser.set_defaults(
        show_bbox=True,
        show_confidence=True,
        show_sticker_color_classification=True,
        display=DISPLAY_PREVIEW_BY_DEFAULT,
        show_original_underlay=SHOW_ORIGINAL_UNDERLAY_BY_DEFAULT,
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
    sticker_model = YOLO(str(args.sticker_model)) if ENABLE_STICKER_SEGMENTATION else None
    pose_model = None
    if DRAW_POSE_KEYPOINTS or ENABLE_POSE_ESTIMATION:
        register_training_checkpoint_classes()
        pose_model = YOLO(str(DEFAULT_POSE_MODEL))
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
        if ENABLE_STICKER_SEGMENTATION and color_features_enabled(args)
        else None
    )
    color_plot_requested = ENABLE_STICKER_SEGMENTATION and color_plot_outputs_requested(args)
    color_plot_bgr_batches = []
    color_plot_detected_frames = 0
    roi_tracker = RoiTracker()
    pose_solver = (
        CameraBootstrapPnP(
            bootstrap_frames=PNP_BOOTSTRAP_FRAME_COUNT,
            focal_min=PNP_FOCAL_SCALE_MIN,
            focal_max=PNP_FOCAL_SCALE_MAX,
            focal_steps=PNP_FOCAL_SCALE_STEPS,
            principal_offset=PNP_PRINCIPAL_POINT_OFFSET_RATIO,
            reprojection_error=PNP_MAX_REPROJECTION_ERROR_PX,
            inlier_threshold=PNP_INLIER_THRESHOLD_PX,
        )
        if ENABLE_POSE_ESTIMATION
        else None
    )

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
            roi_tracker,
            pose_model,
            pose_solver,
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
        if args.display:
            cv2.imshow(DISPLAY_WINDOW_NAME, output)
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
            roi_tracker,
            pose_model,
            pose_solver,
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
            cv2.imshow(DISPLAY_WINDOW_NAME, output)
            key = cv2.waitKey(0 if args.once else 1) & 0xFF
            if key == ord("q"):
                break
        if args.once:
            break

    if args.display:
        cv2.destroyAllWindows()
    capture.release()
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
