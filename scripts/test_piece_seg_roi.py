"""Test YOLO sticker segmentation on a cube bounding-box ROI.

Pipeline:
1. Detect the whole cube with the existing bbox detector.
2. Crop a padded square ROI around that bbox.
3. Run the sticker segmentation model on the ROI.
4. Draw segmented sticker masks back onto the original OpenCV frame.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Union

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from live_recognition.cube_net import CubeNetState  # noqa: E402
from live_recognition.cube_net import build_cube_net_from_graph, draw_cube_net
from live_recognition.detectors import YOLOCubeDetector  # noqa: E402
from live_recognition.piece_graph import (PieceGraph, StickerNode,
                                          _legal_mask_adjacencies)
from live_recognition.piece_graph import \
    draw_piece_graph as draw_sticker_graph  # noqa: E402
from live_recognition.visualization import _reliable_color_mask  # noqa: E402
from live_recognition.visualization import (save_hsv_samples_interactive_plot,
                                            save_oklab_samples_interactive_plot,
                                            save_rgb_samples_interactive_plot,
                                            save_rgb_samples_plot)

DEFAULT_BBOX_MODEL = ROOT / "runs/500ExamplesWeights.pt"
DEFAULT_STICKER_MODEL = (
    ROOT
    / "runs/segment/runs/cube_face_sticker_seg_local/runs/cube_face_sticker_yolo11n_seg-2/weights/best.pt"
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
DENSITY_CLUSTER_EPS = 20.0
DENSITY_CLUSTER_EPS_REFERENCE_RGB = 150.0
DENSITY_CLUSTER_EPS_ORIGIN = 0.1
DENSITY_CLUSTER_MIN_POINTS = 4
HSV_DENSITY_CLUSTER_EPS = 20.0
HSV_DENSITY_CLUSTER_MIN_POINTS = DENSITY_CLUSTER_MIN_POINTS
OKLAB_GMM_COVARIANCE_REGULARIZATION = 0.0004
OKLAB_GMM_MAX_ITERATIONS = 60
OKLAB_GMM_TOLERANCE = 1e-3
OKLAB_L_COMPRESSION_FACTOR = 3.0
OKLAB_SLICE_STICKER_LINK_LAMBDA = 0.05
OKLAB_SLICE_OVERLAP_LAMBDA = 0.05
DENSITY_WEIGHT_POWER = 0.5
DENSITY_WEIGHT_MAX = 4.0
DENSITY_WEIGHT_MIX = 0.5
POLYLINE_COVERAGE_MIX = 0.5
POLYLINE_COVERAGE_BINS = 24
POLYLINE_INSIDE_PENALTY = 6.0
OKLAB_ANGULAR_MIN_CHROMA = 0.018
OKLAB_ANGULAR_NEUTRAL_CHROMA = 0.035
CLUSTER_MERGE_RAY_DISTANCE_DIVISOR = 4.0
DENSITY_PEAK_K_NEIGHBORS = 16
DENSITY_PEAK_MAX_FIT_SAMPLES = 5000
DENSITY_PEAK_MIN_PEAK_DISTANCE = 18.0
DENSE_RAY_K_NEIGHBORS = 16
DENSE_RAY_MAX_FIT_SAMPLES = 5000
DENSE_RAY_MIN_ANCHOR_DISTANCE = 18.0
DENSE_RAY_MAX_DISTANCE = 18.0
GMM_MAX_MAHALANOBIS_DISTANCE = 9.0
GMM_REGULARIZATION = 36.0
GMM_RAY_VARIANCE_FLOOR = 0.0
GMM_CONSTRAIN_COVARIANCE_TO_RAY = False
GMM_PERPENDICULAR_VARIANCE_FLOOR = 16.0
GMM_PERPENDICULAR_VARIANCE_CEILING = 0.0
GMM_DENSE_CORE_K_NEIGHBORS = 0
GMM_DENSE_CORE_QUANTILE = 0.0
GMM_COLOR_PRIORS = "white=255,255,255;yellow=255,255,0;red=255,0,0;orange=255,120,0;blue=0,0,255;green=0,255,0"
GMM_PRIOR_STRENGTH = 1.0
GMM_MIN_ASSIGNMENT_CONFIDENCE = 0.70
GMM_ASSIGNMENT_BALANCE = 1.0
GMM_BALANCE_ITERATIONS = 30
GMM_MAX_ITERATIONS = 60
GMM_TOLERANCE = 1e-3
RAY_MAX_ANGLE_RADIANS = 0.28
RAY_ASSIGNMENT_SHARPNESS = 80.0
RAY_BALANCE_ITERATIONS = 30
RAY_MAX_ITERATIONS = 30
CONE_MAX_ANGLE_RADIANS = 0.28
CONE_ASSIGNMENT_SHARPNESS = 80.0
CONE_BALANCE_ITERATIONS = 30
CONE_MAX_ITERATIONS = 30
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


class HDBSCANDensityPlaneColorClassifier:
    """Oklab HDBSCAN classifier with recursive 3D density-plane refinement."""

    def __init__(self, *, required_frames=30, cluster_count=6, min_brightness=40.0,
                 max_brightness=None, hdbscan_min_cluster_size=500, hdbscan_min_samples=5,
                 cluster_ambiguity_distance=None,
                 hdbscan_cluster_selection_method="eom", hdbscan_cluster_selection_epsilon=.01,
                 hdbscan_alpha=1.0, hdbscan_metric="euclidean", hdbscan_core_dist_n_jobs=1,
                 hdbscan_leave_noise=True, hdbscan_hierarchy_plot=None,
                 plane_angle_count=36, plane_tilt_count=3,
                 plane_max_tilt_degrees=15.0, plane_offset_count=25, plane_density_grid_size=48,
                 plane_density_smoothing=1.5, plane_bandwidth=0.0, plane_min_child_size=10,
                 plane_min_child_fraction=.15, plane_balance_penalty=.25,
                 plane_relative_valley_weight=1.0, plane_min_side_mass=.12,
                 plane_cut_mass_weight=.5, plane_cluster_size_weight=.25):
        self.required_frames, self.cluster_count = max(1, int(required_frames)), max(1, int(cluster_count))
        self.min_brightness = float(min_brightness)
        self.max_brightness = float(max_brightness) if max_brightness is not None else None
        self.hdbscan_params = dict(
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
            cluster_selection_method=hdbscan_cluster_selection_method,
            cluster_selection_epsilon=hdbscan_cluster_selection_epsilon,
            alpha=hdbscan_alpha,
            metric=hdbscan_metric,
            allow_single_cluster=False,
            core_dist_n_jobs=hdbscan_core_dist_n_jobs,
        )
        self.hdbscan_leave_noise = bool(hdbscan_leave_noise)
        self.hdbscan_hierarchy_plot = Path(hdbscan_hierarchy_plot) if hdbscan_hierarchy_plot else None
        self.plane_params = dict(
            angle_count=plane_angle_count, tilt_count=plane_tilt_count,
            max_tilt_degrees=plane_max_tilt_degrees, offset_count=plane_offset_count,
            density_grid_size=plane_density_grid_size, density_smoothing=plane_density_smoothing,
            plane_bandwidth=plane_bandwidth, min_child_size=plane_min_child_size,
            min_child_fraction=plane_min_child_fraction, balance_penalty=plane_balance_penalty,
            relative_valley_weight=plane_relative_valley_weight,
            min_side_mass=plane_min_side_mass, cut_mass_weight=plane_cut_mass_weight,
            cluster_size_weight=plane_cluster_size_weight,
        )
        self.detected_frames = 0
        self._bgr_samples = []
        self.centroids = np.empty((0, 3), dtype=np.float32)
        self.oklab_centroids = np.empty((0, 3), dtype=np.float32)
        self.cluster_samples = []
        self.split_planes = []
        self._fit_bgr_samples = np.empty((0, 3), dtype=np.uint8)
        self.sample_labels = np.empty(0, dtype=np.int32)

    @property
    def ready(self): return len(self.oklab_centroids) > 0

    def observe_frame(self, bgr_samples):
        if self.ready or len(bgr_samples) == 0: return
        self._bgr_samples.extend(np.asarray(bgr_samples, dtype=np.uint8).reshape(-1, 3))
        self.detected_frames += 1
        if self.detected_frames < self.required_frames: return
        bgr = np.asarray(self._bgr_samples, dtype=np.uint8)
        oklab = bgr_samples_to_oklab(bgr)
        brightness = oklab[:, 0] * 255.0
        mask = brightness >= self.min_brightness
        if self.max_brightness is not None: mask &= brightness <= self.max_brightness
        fit = oklab[mask]
        # Use the shared implementation so the ROI path and standalone
        # color-sample tool have identical selection, noise handling, and
        # plane-splitting behavior.
        from live_recognition.color_clustering import (
            hdbscan_cluster_memberships,
            recursive_density_plane_splits,
        )
        labels, _, _noise_mask = hdbscan_cluster_memberships(
            fit,
            cluster_count=self.cluster_count,
            assign_noise=False,
            hierarchy_plot=self.hdbscan_hierarchy_plot,
            **self.hdbscan_params,
        )
        labels, self.oklab_centroids, self.split_planes = recursive_density_plane_splits(
            fit,
            labels,
            target_clusters=self.cluster_count,
            **self.plane_params,
        )
        if not self.hdbscan_leave_noise and np.any(labels < 0) and len(self.oklab_centroids):
            noise = np.flatnonzero(labels < 0)
            labels[noise] = np.linalg.norm(
                fit[noise, None, :] - self.oklab_centroids[None, :, :], axis=2
            ).argmin(axis=1)
        self.cluster_samples = [fit[labels == i] for i in range(len(self.oklab_centroids))]
        self.centroids = oklab_to_rgb(self.oklab_centroids)
        self._fit_bgr_samples = bgr[mask].copy()
        self.sample_labels = labels.copy()

    def classify_bgr(self, bgr_sample):
        if not self.ready: return None
        lab = bgr_samples_to_oklab(np.asarray(bgr_sample, dtype=np.uint8).reshape(1, 3))[0]
        lightness = float(lab[0] * 255.0)
        if lightness < self.min_brightness or (self.max_brightness is not None and lightness > self.max_brightness): return None
        return int(np.linalg.norm(self.oklab_centroids - lab, axis=1).argmin())

    def classify_bgr_samples(self, bgr_samples):
        votes = [i for sample in np.asarray(bgr_samples, dtype=np.uint8).reshape(-1, 3) if (i := self.classify_bgr(sample)) is not None]
        return int(np.bincount(np.asarray(votes, dtype=np.int32)).argmax()) if votes else None


class PiecewiseLinearColorClassifier(HDBSCANDensityPlaneColorClassifier):
    """Classify RGB samples by distance to black-breakpoint-white polylines."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.polyline_models: list[np.ndarray] = []

    def observe_frame(self, bgr_samples):
        was_ready = self.ready
        super().observe_frame(bgr_samples)
        if was_ready or not self.ready or self.polyline_models:
            return

        oklab_samples = bgr_samples_to_oklab(self._fit_bgr_samples)
        self.polyline_models = [
            _fit_piecewise_oklab_line(oklab_samples[self.sample_labels == cluster_index])
            for cluster_index in range(len(self.centroids))
        ]

    def classify_bgr(self, bgr_sample):
        if not self.ready or not self.polyline_models:
            return None
        oklab = bgr_samples_to_oklab(np.asarray(bgr_sample, dtype=np.uint8).reshape(1, 3))
        lightness = float(oklab[0, 0]) * 255.0
        if lightness < self.min_brightness:
            return None
        if self.max_brightness is not None and lightness > self.max_brightness:
            return None
        distances = [
            float(_piecewise_oklab_line_distances(oklab, breakpoint)[0])
            for breakpoint in self.polyline_models
        ]
        if not distances:
            return None
        distances = np.asarray(distances, dtype=np.float32)
        return int(np.argmin(distances))


class ClusterMembershipColorClassifier(HDBSCANDensityPlaneColorClassifier):
    """Classify only samples that fall within a learned color cluster."""

    def __init__(self, *, cluster_ambiguity_distance: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.cluster_ambiguity_distance = max(float(cluster_ambiguity_distance), 0.0)
        self.cluster_samples_oklab: list[np.ndarray] = []
        self.cluster_membership_radii = np.empty(0, dtype=np.float32)

    def observe_frame(self, bgr_samples):
        was_ready = self.ready
        super().observe_frame(bgr_samples)
        if was_ready or not self.ready or self.cluster_samples_oklab:
            return

        oklab_samples = bgr_samples_to_oklab(self._fit_bgr_samples)
        self.cluster_samples_oklab = [
            oklab_samples[self.sample_labels == cluster_index]
            for cluster_index in range(len(self.centroids))
        ]
        self.cluster_membership_radii = np.asarray(
            [
                max(
                    float(
                        np.linalg.norm(
                            samples - samples.mean(axis=0),
                            axis=1,
                        ).max()
                    ),
                    1e-4,
                )
                if len(samples)
                else 1e-4
                for samples in self.cluster_samples_oklab
            ],
            dtype=np.float32,
        )

    def classify_bgr(self, bgr_sample):
        if not self.ready or not self.cluster_samples_oklab:
            return None
        oklab = bgr_samples_to_oklab(np.asarray(bgr_sample, dtype=np.uint8).reshape(1, 3))
        lightness = float(oklab[0, 0]) * 255.0
        if lightness < self.min_brightness:
            return None
        if self.max_brightness is not None and lightness > self.max_brightness:
            return None

        distances = np.asarray(
            [
                np.linalg.norm(samples - oklab[0], axis=1).min()
                if len(samples)
                else np.inf
                for samples in self.cluster_samples_oklab
            ],
            dtype=np.float32,
        )
        inside = distances <= self.cluster_membership_radii
        if not bool(inside.any()):
            return None
        if self.cluster_ambiguity_distance > 0.0:
            nearest_cluster = int(np.argmin(distances))
            other_distances = np.delete(distances, nearest_cluster)
            if bool(
                np.any(other_distances <= self.cluster_ambiguity_distance)
            ):
                return None
        return int(np.argmin(np.where(inside, distances, np.inf)))


def _fit_piecewise_oklab_line(oklab_samples: np.ndarray) -> np.ndarray:
    points = np.asarray(oklab_samples, dtype=np.float32).reshape(-1, 3)
    if len(points) == 0:
        return np.asarray([[0.67, 0.0, 0.0], [0.33, 0.0, 0.0]], dtype=np.float32)
    ordered = points[np.argsort(points[:, 0])[::-1]]
    candidate_indexes = np.unique(
        np.linspace(0, len(ordered) - 1, min(24, len(ordered)), dtype=np.int32)
    )
    candidates = ordered[candidate_indexes]
    best_pair = np.asarray([candidates[0], candidates[-1]], dtype=np.float32)
    best_score = np.inf
    for first_index in range(len(candidates) - 1):
        for second_index in range(first_index + 1, len(candidates)):
            pair = np.asarray(
                [candidates[first_index], candidates[second_index]],
                dtype=np.float32,
            )
            score = float(np.mean(_piecewise_oklab_line_distances(points, pair) ** 2))
            if score < best_score:
                best_score = score
                best_pair = pair
    return best_pair


def _piecewise_oklab_line_distances(points: np.ndarray, breakpoint: np.ndarray) -> np.ndarray:
    samples = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    pivots = np.asarray(breakpoint, dtype=np.float32).reshape(2, 3)
    white = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    black = np.zeros(3, dtype=np.float32)
    return np.minimum.reduce(
        [
            _oklab_point_to_segment_distances(samples, white, pivots[0]),
            _oklab_point_to_segment_distances(samples, pivots[0], pivots[1]),
            _oklab_point_to_segment_distances(samples, pivots[1], black),
        ]
    )


class ColorDensityClassifier:
    def __init__(
        self,
        *,
        required_frames: int = 30,
        cluster_count: int = 6,
        min_brightness: float = 40.0,
        max_brightness: float | None = None,
        density_cluster_eps: float = DENSITY_CLUSTER_EPS,
        density_cluster_eps_reference_rgb: float = DENSITY_CLUSTER_EPS_REFERENCE_RGB,
        density_cluster_min_neighbors: int = DENSITY_CLUSTER_MIN_POINTS,
        max_cluster_distance: float | None = None,
        cluster_merge_ray_distance_divisor: float = CLUSTER_MERGE_RAY_DISTANCE_DIVISOR,
    ) -> None:
        self.required_frames = max(1, required_frames)
        self.cluster_count = max(1, cluster_count)
        self.min_brightness = min_brightness
        self.max_brightness = float(max_brightness) if max_brightness is not None else None
        self.density_cluster_eps = max(float(density_cluster_eps), 1e-6)
        self.density_cluster_eps_reference_rgb = max(
            float(density_cluster_eps_reference_rgb),
            1e-6,
        )
        self.density_cluster_min_neighbors = max(1, density_cluster_min_neighbors)
        self.cluster_merge_ray_distance_divisor = max(
            float(cluster_merge_ray_distance_divisor),
            1e-6,
        )
        self.max_cluster_distance = (
            float(max_cluster_distance)
            if max_cluster_distance is not None
            else self.density_cluster_eps
        )
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
            max_brightness=self.max_brightness,
            eps=self.density_cluster_eps,
            eps_reference_rgb=self.density_cluster_eps_reference_rgb,
            min_points=self.density_cluster_min_neighbors,
            merge_ray_distance_divisor=self.cluster_merge_ray_distance_divisor,
        )

    def classify_bgr(self, bgr_sample: np.ndarray) -> int | None:
        if not self.ready:
            return None

        rgb = np.asarray(bgr_sample, dtype=np.float32).reshape(3)[::-1]
        brightness = float(np.linalg.norm(rgb))
        if brightness < self.min_brightness:
            return None
        if self.max_brightness is not None and brightness > self.max_brightness:
            return None

        distances = distances_to_rgb_clusters(
            rgb,
            self.cluster_samples,
        )
        if len(distances) == 0 or float(distances.min()) > self.max_cluster_distance:
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


class HSVColorDensityClassifier:
    def __init__(
        self,
        *,
        required_frames: int = 30,
        cluster_count: int = 6,
        min_brightness: float = 40.0,
        max_brightness: float | None = None,
        density_cluster_eps: float = HSV_DENSITY_CLUSTER_EPS,
        density_cluster_min_neighbors: int = HSV_DENSITY_CLUSTER_MIN_POINTS,
        max_cluster_distance: float | None = None,
    ) -> None:
        self.required_frames = max(1, required_frames)
        self.cluster_count = max(1, cluster_count)
        self.min_brightness = min_brightness
        self.max_brightness = float(max_brightness) if max_brightness is not None else None
        self.density_cluster_eps = max(float(density_cluster_eps), 1e-6)
        self.density_cluster_min_neighbors = max(1, int(density_cluster_min_neighbors))
        self.max_cluster_distance = (
            float(max_cluster_distance)
            if max_cluster_distance is not None
            else self.density_cluster_eps
        )
        self.detected_frames = 0
        self._bgr_samples: list[np.ndarray] = []
        self.centroids = np.empty((0, 3), dtype=np.float32)
        self.cluster_samples: list[np.ndarray] = []
        self.hsv_cluster_samples: list[np.ndarray] = []

    @property
    def ready(self) -> bool:
        return len(self.hsv_cluster_samples) > 0

    def observe_frame(self, bgr_samples: np.ndarray) -> None:
        if self.ready or len(bgr_samples) == 0:
            return

        bgr_samples = np.asarray(bgr_samples, dtype=np.uint8).reshape(-1, 3)
        self._bgr_samples.extend(bgr_samples)
        self.detected_frames += 1
        if self.detected_frames < self.required_frames:
            return

        _labels, self.centroids, self.cluster_samples, self.hsv_cluster_samples = (
            density_hsv_cluster_memberships(
                np.asarray(self._bgr_samples, dtype=np.uint8),
                target_clusters=self.cluster_count,
                min_brightness=self.min_brightness,
                max_brightness=self.max_brightness,
                eps=self.density_cluster_eps,
                min_points=self.density_cluster_min_neighbors,
            )
        )

    def classify_bgr(self, bgr_sample: np.ndarray) -> int | None:
        if not self.ready:
            return None

        hsv = bgr_samples_to_hsv(np.asarray(bgr_sample, dtype=np.uint8).reshape(1, 3))[0]
        brightness = float(hsv[2])
        if brightness < self.min_brightness:
            return None
        if self.max_brightness is not None and brightness > self.max_brightness:
            return None

        distances = distances_to_hsv_clusters(hsv, self.hsv_cluster_samples)
        if len(distances) == 0 or float(distances.min()) > self.max_cluster_distance:
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


class OklabGMMColorClassifier:
    """Fit a standard full-covariance GMM in Oklab after sample collection."""

    def __init__(
        self,
        *,
        required_frames: int = 30,
        cluster_count: int = 6,
        min_brightness: float = 40.0,
        max_brightness: float | None = None,
        l_compression_factor: float = OKLAB_L_COMPRESSION_FACTOR,
    ) -> None:
        self.required_frames = max(1, required_frames)
        self.cluster_count = max(1, cluster_count)
        self.min_brightness = float(min_brightness)
        self.max_brightness = float(max_brightness) if max_brightness is not None else None
        self.l_compression_factor = max(float(l_compression_factor), 1e-6)
        self.detected_frames = 0
        self._bgr_samples: list[np.ndarray] = []
        self.centroids = np.empty((0, 3), dtype=np.float32)  # original RGB, for display
        self.oklab_centroids = np.empty((0, 3), dtype=np.float32)
        self.covariances = np.empty((0, 3, 3), dtype=np.float32)
        self.weights = np.empty(0, dtype=np.float32)
        self.cluster_samples: list[np.ndarray] = []
        self.polyline_models: list[dict[str, np.ndarray | float]] = []

    @property
    def ready(self) -> bool:
        return len(self.oklab_centroids) > 0

    def observe_frame(self, bgr_samples: np.ndarray) -> None:
        if self.ready or len(bgr_samples) == 0:
            return
        samples = np.asarray(bgr_samples, dtype=np.uint8).reshape(-1, 3)
        self._bgr_samples.extend(samples)
        self.detected_frames += 1
        if self.detected_frames < self.required_frames:
            return

        bgr = np.asarray(self._bgr_samples, dtype=np.uint8)
        oklab = bgr_samples_to_oklab(bgr)
        compressed_oklab = compress_oklab_l(oklab, self.l_compression_factor)
        (
            _labels,
            self.oklab_centroids,
            self.cluster_samples,
            self.polyline_models,
            self.weights,
        ) = oklab_polyline_cluster_memberships(
            compressed_oklab,
            original_oklab_samples=oklab,
            target_clusters=self.cluster_count,
            min_brightness=self.min_brightness / self.l_compression_factor,
            max_brightness=(
                self.max_brightness / self.l_compression_factor
                if self.max_brightness is not None
                else None
            ),
        )
        self.centroids = oklab_to_rgb(self.oklab_centroids)

    def classify_bgr(self, bgr_sample: np.ndarray) -> int | None:
        if not self.ready:
            return None
        bgr = np.asarray(bgr_sample, dtype=np.uint8).reshape(1, 3)
        oklab = bgr_samples_to_oklab(bgr)[0]
        lightness = float(oklab[0]) * 255.0
        if lightness < self.min_brightness:
            return None
        if self.max_brightness is not None and lightness > self.max_brightness:
            return None
        log_likelihoods = polyline_cluster_log_likelihoods(
            oklab,
            self.polyline_models,
            self.weights,
        )
        if len(log_likelihoods) == 0 or not np.isfinite(log_likelihoods).any():
            return None
        return int(log_likelihoods.argmax())

    def classify_bgr_samples(self, bgr_samples: np.ndarray) -> int | None:
        if not self.ready:
            return None
        votes = [
            cluster_index
            for sample in np.asarray(bgr_samples, dtype=np.uint8).reshape(-1, 3)
            if (cluster_index := self.classify_bgr(sample)) is not None
        ]
        if not votes:
            return None
        return int(np.bincount(np.asarray(votes, dtype=np.int32)).argmax())


class BalancedGMMColorClassifier:
    def __init__(
        self,
        *,
        required_frames: int = 30,
        cluster_count: int = 6,
        min_brightness: float = 40.0,
        max_brightness: float | None = None,
        max_mahalanobis_distance: float = GMM_MAX_MAHALANOBIS_DISTANCE,
        regularization: float = GMM_REGULARIZATION,
        ray_variance_floor: float = GMM_RAY_VARIANCE_FLOOR,
        constrain_covariance_to_ray: bool = GMM_CONSTRAIN_COVARIANCE_TO_RAY,
        perpendicular_variance_floor: float = GMM_PERPENDICULAR_VARIANCE_FLOOR,
        perpendicular_variance_ceiling: float = GMM_PERPENDICULAR_VARIANCE_CEILING,
        dense_core_k_neighbors: int = GMM_DENSE_CORE_K_NEIGHBORS,
        dense_core_quantile: float = GMM_DENSE_CORE_QUANTILE,
        prior_rgb_values: np.ndarray | None = None,
        prior_strength: float = GMM_PRIOR_STRENGTH,
        min_assignment_confidence: float = GMM_MIN_ASSIGNMENT_CONFIDENCE,
        assignment_balance: float = GMM_ASSIGNMENT_BALANCE,
        balance_iterations: int = GMM_BALANCE_ITERATIONS,
        max_iterations: int = GMM_MAX_ITERATIONS,
        tolerance: float = GMM_TOLERANCE,
    ) -> None:
        self.required_frames = max(1, required_frames)
        self.cluster_count = max(1, cluster_count)
        self.min_brightness = min_brightness
        self.max_brightness = float(max_brightness) if max_brightness is not None else None
        self.max_mahalanobis_distance = max(float(max_mahalanobis_distance), 1e-6)
        self.regularization = max(float(regularization), 1e-6)
        self.ray_variance_floor = max(float(ray_variance_floor), 0.0)
        self.constrain_covariance_to_ray = bool(constrain_covariance_to_ray)
        self.perpendicular_variance_floor = max(float(perpendicular_variance_floor), 1e-6)
        self.perpendicular_variance_ceiling = max(float(perpendicular_variance_ceiling), 0.0)
        self.dense_core_k_neighbors = max(0, int(dense_core_k_neighbors))
        self.dense_core_quantile = float(np.clip(dense_core_quantile, 0.0, 0.9999))
        self.prior_rgb_values = _valid_rgb_prior_values(prior_rgb_values)
        self.prior_strength = float(np.clip(prior_strength, 0.0, 1.0))
        self.min_assignment_confidence = float(np.clip(min_assignment_confidence, 0.0, 1.0))
        self.assignment_balance = float(np.clip(assignment_balance, 0.0, 1.0))
        self.balance_iterations = max(1, int(balance_iterations))
        self.max_iterations = max(1, int(max_iterations))
        self.tolerance = max(float(tolerance), 0.0)
        self.detected_frames = 0
        self._rgb_samples: list[np.ndarray] = []
        self.centroids = np.empty((0, 3), dtype=np.float32)
        self.covariances = np.empty((0, 3, 3), dtype=np.float32)
        self.precisions = np.empty((0, 3, 3), dtype=np.float32)
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

        memberships = balanced_gmm_rgb_cluster_memberships(
            np.asarray(self._rgb_samples, dtype=np.uint8),
            target_clusters=self.cluster_count,
            min_brightness=self.min_brightness,
            max_brightness=self.max_brightness,
            max_mahalanobis_distance=self.max_mahalanobis_distance,
            regularization=self.regularization,
            ray_variance_floor=self.ray_variance_floor,
            constrain_covariance_to_ray=self.constrain_covariance_to_ray,
            perpendicular_variance_floor=self.perpendicular_variance_floor,
            perpendicular_variance_ceiling=self.perpendicular_variance_ceiling,
            dense_core_k_neighbors=self.dense_core_k_neighbors,
            dense_core_quantile=self.dense_core_quantile,
            prior_rgb_values=self.prior_rgb_values,
            prior_strength=self.prior_strength,
            min_assignment_confidence=self.min_assignment_confidence,
            assignment_balance=self.assignment_balance,
            balance_iterations=self.balance_iterations,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
        )
        _labels, self.centroids, self.cluster_samples, self.covariances, self.precisions = memberships

    def classify_bgr(self, bgr_sample: np.ndarray) -> int | None:
        if not self.ready:
            return None

        rgb = np.asarray(bgr_sample, dtype=np.float32).reshape(3)[::-1]
        brightness = float(np.linalg.norm(rgb))
        if brightness < self.min_brightness:
            return None
        if self.max_brightness is not None and brightness > self.max_brightness:
            return None

        log_probabilities, _precisions = _gmm_log_probabilities(
            rgb.reshape(1, 3),
            self.centroids,
            self.covariances,
            regularization=self.regularization,
        )
        responsibilities = _gmm_unbalanced_responsibilities(log_probabilities)
        if responsibilities.size == 0:
            return None
        cluster_index = int(responsibilities[0].argmax())
        confidence = float(responsibilities[0, cluster_index])
        if confidence < self.min_assignment_confidence:
            return None

        distances = mahalanobis_distances_to_rgb_clusters(rgb, self.centroids, self.precisions)
        if len(distances) == 0 or float(distances[cluster_index]) > self.max_mahalanobis_distance:
            return None
        return cluster_index

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


class BalancedRayColorClassifier:
    def __init__(
        self,
        *,
        required_frames: int = 30,
        cluster_count: int = 6,
        min_brightness: float = 40.0,
        max_brightness: float | None = None,
        max_angle_radians: float = RAY_MAX_ANGLE_RADIANS,
        assignment_balance: float = GMM_ASSIGNMENT_BALANCE,
        assignment_sharpness: float = RAY_ASSIGNMENT_SHARPNESS,
        balance_iterations: int = RAY_BALANCE_ITERATIONS,
        max_iterations: int = RAY_MAX_ITERATIONS,
    ) -> None:
        self.required_frames = max(1, required_frames)
        self.cluster_count = max(1, cluster_count)
        self.min_brightness = min_brightness
        self.max_brightness = float(max_brightness) if max_brightness is not None else None
        self.max_angle_radians = max(float(max_angle_radians), 1e-6)
        self.assignment_balance = float(np.clip(assignment_balance, 0.0, 1.0))
        self.assignment_sharpness = max(float(assignment_sharpness), 1e-6)
        self.balance_iterations = max(1, int(balance_iterations))
        self.max_iterations = max(1, int(max_iterations))
        self.detected_frames = 0
        self._rgb_samples: list[np.ndarray] = []
        self.centroids = np.empty((0, 3), dtype=np.float32)
        self.ray_directions = np.empty((0, 3), dtype=np.float32)
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

        memberships = balanced_ray_rgb_cluster_memberships(
            np.asarray(self._rgb_samples, dtype=np.uint8),
            target_clusters=self.cluster_count,
            min_brightness=self.min_brightness,
            max_brightness=self.max_brightness,
            max_angle_radians=self.max_angle_radians,
            assignment_balance=self.assignment_balance,
            assignment_sharpness=self.assignment_sharpness,
            balance_iterations=self.balance_iterations,
            max_iterations=self.max_iterations,
        )
        _labels, self.centroids, self.cluster_samples, self.ray_directions = memberships

    def classify_bgr(self, bgr_sample: np.ndarray) -> int | None:
        if not self.ready:
            return None

        rgb = np.asarray(bgr_sample, dtype=np.float32).reshape(3)[::-1]
        brightness = float(np.linalg.norm(rgb))
        if brightness < self.min_brightness:
            return None
        if self.max_brightness is not None and brightness > self.max_brightness:
            return None

        angles = angles_to_rgb_rays(rgb, self.ray_directions)
        if len(angles) == 0 or float(angles.min()) > self.max_angle_radians:
            return None
        return int(angles.argmin())

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


class DensityPeakColorClassifier:
    def __init__(
        self,
        *,
        required_frames: int = 30,
        cluster_count: int = 6,
        min_brightness: float = 40.0,
        max_brightness: float | None = None,
        k_neighbors: int = DENSITY_PEAK_K_NEIGHBORS,
        max_fit_samples: int = DENSITY_PEAK_MAX_FIT_SAMPLES,
        min_peak_distance: float = DENSITY_PEAK_MIN_PEAK_DISTANCE,
        max_cluster_distance: float | None = None,
    ) -> None:
        self.required_frames = max(1, required_frames)
        self.cluster_count = max(1, cluster_count)
        self.min_brightness = min_brightness
        self.max_brightness = float(max_brightness) if max_brightness is not None else None
        self.k_neighbors = max(1, int(k_neighbors))
        self.max_fit_samples = max(1, int(max_fit_samples))
        self.min_peak_distance = max(float(min_peak_distance), 0.0)
        self.max_cluster_distance = (
            float(max_cluster_distance)
            if max_cluster_distance is not None
            else float(min_peak_distance) * 2.0
        )
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

        _labels, self.centroids, self.cluster_samples = density_peak_rgb_cluster_memberships(
            np.asarray(self._rgb_samples, dtype=np.uint8),
            target_clusters=self.cluster_count,
            min_brightness=self.min_brightness,
            max_brightness=self.max_brightness,
            k_neighbors=self.k_neighbors,
            max_fit_samples=self.max_fit_samples,
            min_peak_distance=self.min_peak_distance,
        )

    def classify_bgr(self, bgr_sample: np.ndarray) -> int | None:
        if not self.ready:
            return None

        rgb = np.asarray(bgr_sample, dtype=np.float32).reshape(3)[::-1]
        brightness = float(np.linalg.norm(rgb))
        if brightness < self.min_brightness:
            return None
        if self.max_brightness is not None and brightness > self.max_brightness:
            return None

        distances = distances_to_rgb_clusters(rgb, self.cluster_samples)
        if len(distances) == 0 or float(distances.min()) > self.max_cluster_distance:
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


class DenseRayColorClassifier:
    def __init__(
        self,
        *,
        required_frames: int = 30,
        cluster_count: int = 6,
        min_brightness: float = 40.0,
        max_brightness: float | None = None,
        k_neighbors: int = DENSE_RAY_K_NEIGHBORS,
        max_fit_samples: int = DENSE_RAY_MAX_FIT_SAMPLES,
        min_anchor_distance: float = DENSE_RAY_MIN_ANCHOR_DISTANCE,
        max_ray_distance: float = DENSE_RAY_MAX_DISTANCE,
    ) -> None:
        self.required_frames = max(1, required_frames)
        self.cluster_count = max(1, cluster_count)
        self.min_brightness = min_brightness
        self.max_brightness = float(max_brightness) if max_brightness is not None else None
        self.k_neighbors = max(1, int(k_neighbors))
        self.max_fit_samples = max(1, int(max_fit_samples))
        self.min_anchor_distance = max(float(min_anchor_distance), 0.0)
        self.max_ray_distance = max(float(max_ray_distance), 1e-6)
        self.detected_frames = 0
        self._rgb_samples: list[np.ndarray] = []
        self.centroids = np.empty((0, 3), dtype=np.float32)
        self.ray_directions = np.empty((0, 3), dtype=np.float32)
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

        memberships = dense_ray_rgb_cluster_memberships(
            np.asarray(self._rgb_samples, dtype=np.uint8),
            target_clusters=self.cluster_count,
            min_brightness=self.min_brightness,
            max_brightness=self.max_brightness,
            k_neighbors=self.k_neighbors,
            max_fit_samples=self.max_fit_samples,
            min_anchor_distance=self.min_anchor_distance,
        )
        _labels, self.centroids, self.cluster_samples, self.ray_directions = memberships

    def classify_bgr(self, bgr_sample: np.ndarray) -> int | None:
        if not self.ready:
            return None

        rgb = np.asarray(bgr_sample, dtype=np.float32).reshape(3)[::-1]
        brightness = float(np.linalg.norm(rgb))
        if brightness < self.min_brightness:
            return None
        if self.max_brightness is not None and brightness > self.max_brightness:
            return None

        distances = distances_to_rgb_rays(rgb, self.ray_directions).reshape(-1)
        if len(distances) == 0 or float(distances.min()) > self.max_ray_distance:
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


class BalancedConeColorClassifier(BalancedRayColorClassifier):
    def __init__(
        self,
        *,
        required_frames: int = 30,
        cluster_count: int = 6,
        min_brightness: float = 40.0,
        max_brightness: float | None = None,
        max_angle_radians: float = CONE_MAX_ANGLE_RADIANS,
        assignment_balance: float = GMM_ASSIGNMENT_BALANCE,
        assignment_sharpness: float = CONE_ASSIGNMENT_SHARPNESS,
        balance_iterations: int = CONE_BALANCE_ITERATIONS,
        max_iterations: int = CONE_MAX_ITERATIONS,
    ) -> None:
        super().__init__(
            required_frames=required_frames,
            cluster_count=cluster_count,
            min_brightness=min_brightness,
            max_brightness=max_brightness,
            max_angle_radians=max_angle_radians,
            assignment_balance=assignment_balance,
            assignment_sharpness=assignment_sharpness,
            balance_iterations=balance_iterations,
            max_iterations=max_iterations,
        )


ColorClassifier = HDBSCANDensityPlaneColorClassifier


def distances_to_rgb_clusters(
    rgb_sample: np.ndarray,
    cluster_samples: list[np.ndarray],
) -> np.ndarray:
    point = np.asarray(rgb_sample, dtype=np.float32).reshape(1, 3)
    distances = []
    for samples in cluster_samples:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1, 3)
        if len(samples) == 0:
            distances.append(np.inf)
            continue
        distances.append(float(np.min(np.linalg.norm(samples - point, axis=1))))
    return np.asarray(distances, dtype=np.float32)


def bgr_samples_to_hsv(bgr_samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(bgr_samples, dtype=np.uint8).reshape(-1, 1, 3)
    return cv2.cvtColor(samples, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)


def bgr_samples_to_rgb(bgr_samples: np.ndarray) -> np.ndarray:
    return np.asarray(bgr_samples, dtype=np.uint8).reshape(-1, 3)[:, ::-1].astype(np.float32)


def rgb_to_oklab(rgb_samples: np.ndarray) -> np.ndarray:
    """Convert sRGB 0..255 samples to Oklab (L 0..1, a/b approximately -0.4..0.4)."""
    rgb = np.asarray(rgb_samples, dtype=np.float32).reshape(-1, 3) / 255.0
    rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    lms = rgb @ np.asarray(
        [[0.4122214708, 0.5363325363, 0.0514459929],
         [0.2119034982, 0.6806995451, 0.1073969566],
         [0.0883024619, 0.2817188376, 0.6299787005]], dtype=np.float32
    ).T
    lms_cbrt = np.cbrt(np.maximum(lms, 0.0))
    return (lms_cbrt @ np.asarray(
        [[0.2104542553, 0.7936177850, -0.0040720468],
         [1.9779984951, -2.4285922050, 0.4505937099],
         [0.0259040371, 0.7827717662, -0.8086757660]], dtype=np.float32
    ).T).astype(np.float32)


def oklab_to_rgb(oklab_samples: np.ndarray) -> np.ndarray:
    lab = np.asarray(oklab_samples, dtype=np.float32).reshape(-1, 3)
    lms_cbrt = lab @ np.asarray(
        [[1.0, 0.3963377774, 0.2158037573],
         [1.0, -0.1055613458, -0.0638541728],
         [1.0, -0.0894841775, -1.2914855480]], dtype=np.float32
    ).T
    lms = lms_cbrt ** 3
    rgb_linear = lms @ np.asarray(
        [[4.0767416621, -3.3077115913, 0.2309699292],
         [-1.2684380046, 2.6097574011, -0.3413193965],
         [-0.0041960863, -0.7034186147, 1.7076147010]], dtype=np.float32
    ).T
    rgb = np.where(rgb_linear <= 0.0031308, 12.92 * rgb_linear,
                   1.055 * np.maximum(rgb_linear, 0.0) ** (1.0 / 2.4) - 0.055)
    return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.float32)


def bgr_samples_to_oklab(bgr_samples: np.ndarray) -> np.ndarray:
    return rgb_to_oklab(bgr_samples_to_rgb(bgr_samples))


def oklab_l_slice_knn_outlines(
    oklab_samples: np.ndarray,
    *,
    neighbor_count: int = 6,
    cluster_count: int = 6,
    sticker_ids: np.ndarray | None = None,
    edge_pairs: np.ndarray | None = None,
    cluster_labels: np.ndarray | None = None,
    sticker_link_lambda: float = OKLAB_SLICE_STICKER_LINK_LAMBDA,
    overlap_lambda: float = OKLAB_SLICE_OVERLAP_LAMBDA,
) -> list[list[np.ndarray]]:
    samples = np.asarray(oklab_samples, dtype=np.float32).reshape(-1, 3)
    if sticker_ids is not None:
        sticker_ids = np.asarray(sticker_ids).reshape(-1)
        if len(sticker_ids) != len(samples):
            raise ValueError("sticker_ids must have one entry per Oklab sample")
    if cluster_labels is not None:
        cluster_labels = np.asarray(cluster_labels, dtype=np.int32).reshape(-1)
        if len(cluster_labels) != len(samples):
            raise ValueError("cluster_labels must have one entry per Oklab sample")
    all_ellipses = []
    for slice_index in range(10):
        lower = slice_index / 10.0
        upper = (slice_index + 1) / 10.0
        mask = (samples[:, 0] >= lower) & (
            (samples[:, 0] <= upper) if slice_index == 9 else (samples[:, 0] < upper)
        )
        all_ellipses.append(
            _fit_2d_knn_outlines(
                samples[mask, 1:3],
                neighbor_count=neighbor_count,
                cluster_count=cluster_count,
                sticker_ids=sticker_ids[mask] if sticker_ids is not None else None,
                edge_pairs=_slice_edge_pairs(edge_pairs, np.flatnonzero(mask))
                if edge_pairs is not None else None,
                fixed_labels=cluster_labels[mask] if cluster_labels is not None else None,
                sticker_link_lambda=sticker_link_lambda,
                overlap_lambda=overlap_lambda,
            )
        )
    return all_ellipses


def oklab_graph_cluster_memberships(
    oklab_samples: np.ndarray,
    *,
    cluster_count: int = 6,
    sticker_ids: np.ndarray | None = None,
    edge_pairs: np.ndarray | None = None,
    edge_lambda: float = OKLAB_SLICE_STICKER_LINK_LAMBDA,
    overlap_lambda: float = OKLAB_SLICE_OVERLAP_LAMBDA,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one stable six-way graph-regularized labeling for the 3D plot."""
    values = np.asarray(oklab_samples, dtype=np.float32).reshape(-1, 3)
    if len(values) < max(1, int(cluster_count)):
        return np.full(len(values), -1, dtype=np.int32), np.empty((0, 3), dtype=np.float32)
    groups = None if sticker_ids is None else np.asarray(sticker_ids).reshape(-1)
    if groups is not None and len(groups) != len(values):
        raise ValueError("sticker_ids must have one entry per Oklab sample")
    labels = _graph_regularized_point_labels(
        values,
        groups,
        cluster_count=max(1, int(cluster_count)),
        sticker_link_lambda=float(edge_lambda),
        overlap_lambda=float(overlap_lambda),
        edge_pairs=edge_pairs,
        edge_lambda=float(edge_lambda),
    )
    centers = np.asarray(
        [
            values[labels == index].mean(axis=0)
            if np.any(labels == index) else values.mean(axis=0)
            for index in range(max(1, int(cluster_count)))
        ],
        dtype=np.float32,
    )
    return labels, centers


def _fit_2d_knn_outlines(
    points: np.ndarray,
    *,
    neighbor_count: int,
    cluster_count: int = 6,
    sticker_ids: np.ndarray | None = None,
    edge_pairs: np.ndarray | None = None,
    fixed_labels: np.ndarray | None = None,
    sticker_link_lambda: float = OKLAB_SLICE_STICKER_LINK_LAMBDA,
    overlap_lambda: float = OKLAB_SLICE_OVERLAP_LAMBDA,
) -> list[np.ndarray]:
    values = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    group_ids = None if sticker_ids is None else np.asarray(sticker_ids).reshape(-1)
    if group_ids is not None and len(group_ids) != len(values):
        raise ValueError("sticker_ids must have one entry per point")
    if fixed_labels is not None:
        fixed_labels = np.asarray(fixed_labels, dtype=np.int32).reshape(-1)
        if len(fixed_labels) != len(values):
            raise ValueError("fixed_labels must have one entry per point")
    if edge_pairs is not None:
        edge_pairs = _normalize_edge_pairs(edge_pairs, len(values))
    cluster_count = max(1, int(cluster_count))
    if len(values) < cluster_count:
        return []
    max_points = 2500
    if len(values) > max_points:
        keep_indexes = np.linspace(0, len(values) - 1, max_points, dtype=np.int32)
        values = values[keep_indexes]
        if group_ids is not None:
            group_ids = group_ids[keep_indexes]
        if fixed_labels is not None:
            fixed_labels = fixed_labels[keep_indexes]
        if edge_pairs is not None:
            edge_pairs = _remap_edge_pairs(edge_pairs, keep_indexes)

    # Remove the least locally supported 10% before partitioning the slice.
    # A larger neighbor radius means lower density.  This is intentionally a
    # visualization-only denoising pass; the original samples and classifier
    # inputs are left untouched.
    prefilter_neighbor_count = min(max(1, int(neighbor_count)), len(values) - 1)
    prefilter_distances = np.linalg.norm(
        values[:, None, :] - values[None, :, :], axis=2
    )
    prefilter_distances.sort(axis=1)
    density_radius = prefilter_distances[:, prefilter_neighbor_count]
    keep_count = max(cluster_count, int(math.ceil(len(values) * 0.9)))
    keep_count = min(len(values), keep_count)
    keep_indexes = np.argsort(density_radius, kind="stable")[:keep_count]
    values = values[keep_indexes]
    if group_ids is not None:
        group_ids = group_ids[keep_indexes]
    if fixed_labels is not None:
        fixed_labels = fixed_labels[keep_indexes]
    if edge_pairs is not None:
        edge_pairs = _remap_edge_pairs(edge_pairs, keep_indexes)

    # KNN supplies the local support measure; this small k-means step only
    # prevents support masks from different regions being merged into one
    # contour.  It is deliberately limited to this visualization path.
    if fixed_labels is not None:
        labels = fixed_labels
    elif edge_pairs is not None and len(edge_pairs) and sticker_link_lambda > 0:
        labels = _graph_regularized_point_labels(
            values,
            cluster_count=cluster_count,
            edge_pairs=edge_pairs,
            edge_lambda=sticker_link_lambda,
            overlap_lambda=overlap_lambda,
        )
    elif group_ids is not None and len(np.unique(group_ids)) >= cluster_count and sticker_link_lambda > 0:
        labels = _graph_regularized_point_labels(
            values,
            group_ids,
            cluster_count=cluster_count,
            sticker_link_lambda=sticker_link_lambda,
            overlap_lambda=overlap_lambda,
        )
    else:
        _compactness, labels, _centers = cv2.kmeans(
            values,
            cluster_count,
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4),
            3,
            cv2.KMEANS_PP_CENTERS,
        )
        labels = labels.reshape(-1)
    neighbor_count = max(1, int(neighbor_count))
    outlines = []
    for cluster_index in range(cluster_count):
        cluster_values = values[labels == cluster_index]
        if len(cluster_values) == 0:
            # K-means can leave a center empty when several samples coincide.
            # Reuse a representative point so the requested six visualization
            # slots remain visible rather than silently disappearing.
            cluster_values = values[cluster_index % len(values) : cluster_index % len(values) + 1]
        if len(cluster_values) < 3:
            outlines.append(_convex_hull_outline(cluster_values))
            continue
        local_neighbor_count = min(neighbor_count, len(cluster_values) - 1)
        distances = np.linalg.norm(
            cluster_values[:, None, :] - cluster_values[None, :, :], axis=2
        )
        distances.sort(axis=1)
        kth_distances = distances[:, local_neighbor_count]
        keep = kth_distances <= float(np.quantile(kth_distances, 0.95))
        cluster_values = cluster_values[keep]
        kth_distances = kth_distances[keep]
        if len(cluster_values) < 3:
            outlines.append(_convex_hull_outline(values[labels == cluster_index]))
            continue

        grid_size = 320
        canvas = np.zeros((grid_size, grid_size), dtype=np.uint8)
        pixels = np.clip(
            np.rint((cluster_values + 0.5) * (grid_size - 1)),
            0,
            grid_size - 1,
        ).astype(np.int32)
        coordinate_scale = grid_size - 1
        for pixel, radius in zip(pixels, kth_distances):
            radius_pixels = int(np.clip(radius * coordinate_scale * 1.25, 2.0, 18.0))
            cv2.circle(
                canvas,
                (int(pixel[0]), int(grid_size - 1 - pixel[1])),
                radius_pixels,
                255,
                -1,
            )
        canvas = cv2.GaussianBlur(canvas, (0, 0), 1.5)
        _, binary = cv2.threshold(canvas, 32, 255, cv2.THRESH_BINARY)
        contours, _hierarchy = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contour = max(contours, key=cv2.contourArea, default=None)
        if contour is None or cv2.contourArea(contour) < 4.0:
            outlines.append(_convex_hull_outline(cluster_values))
            continue
        contour = contour.reshape(-1, 2).astype(np.float32)
        outlines.append(
            np.column_stack([
                contour[:, 0] / coordinate_scale - 0.5,
                0.5 - contour[:, 1] / coordinate_scale,
            ]).astype(np.float32)
        )
    return outlines


def _graph_regularized_point_labels(
    values: np.ndarray,
    group_ids: np.ndarray | None = None,
    *,
    cluster_count: int,
    sticker_link_lambda: float,
    overlap_lambda: float,
    edge_pairs: np.ndarray | None = None,
    edge_lambda: float | None = None,
) -> np.ndarray:
    """Cluster points with a compactness plus same-sticker-edge penalty."""
    _compactness, labels, _centers = cv2.kmeans(
        values,
        cluster_count,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4),
        3,
        cv2.KMEANS_PP_CENTERS,
    )
    labels = labels.reshape(-1).astype(np.int32)
    if edge_pairs is None:
        if group_ids is None:
            return labels
        groups = [np.flatnonzero(group_ids == group_id) for group_id in np.unique(group_ids)]
        adjacency = None
        penalty = float(sticker_link_lambda)
    else:
        adjacency = [[] for _ in range(len(values))]
        for left, right in _normalize_edge_pairs(edge_pairs, len(values)):
            adjacency[int(left)].append(int(right))
            adjacency[int(right)].append(int(left))
        groups = []
        penalty = float(sticker_link_lambda if edge_lambda is None else edge_lambda)
    for _ in range(12):
        centers = np.asarray(
            [
                values[labels == cluster_index].mean(axis=0)
                if np.any(labels == cluster_index)
                else values.mean(axis=0)
                for cluster_index in range(cluster_count)
            ],
            dtype=np.float32,
        )
        support_radii = np.asarray(
            [
                np.quantile(
                    np.linalg.norm(values[labels == cluster_index] - centers[cluster_index], axis=1),
                    0.90,
                )
                if np.any(labels == cluster_index)
                else 0.0
                for cluster_index in range(cluster_count)
            ],
            dtype=np.float32,
        )
        center_distances = np.linalg.norm(
            centers[:, None, :] - centers[None, :, :], axis=2
        )
        overlap_amount = np.maximum(
            0.0,
            support_radii[:, None] + support_radii[None, :] - center_distances,
        )
        np.fill_diagonal(overlap_amount, 0.0)
        support_diameter = np.maximum(
            support_radii[:, None] + support_radii[None, :],
            1e-6,
        )
        # Use a dimensionless overlap fraction rather than squared Oklab
        # distances.  This keeps the CLI lambda meaningful across slices.
        overlap_fraction = overlap_amount / support_diameter
        overlap_cost = np.sum(overlap_fraction**2, axis=1)
        changed = False
        for indexes in groups:
            if len(indexes) < 2:
                continue
            current_labels = labels[indexes].copy()
            point_indexes = indexes
            for point_index in point_indexes:
                distances = np.sum((centers - values[point_index]) ** 2, axis=1)
                if adjacency is not None:
                    neighbors = adjacency[int(point_index)]
                    disagreement = np.asarray(
                        [sum(labels[n] != cluster_index for n in neighbors) for cluster_index in range(cluster_count)],
                        dtype=np.float32,
                    )
                else:
                    disagreement = np.asarray(
                        [np.mean(current_labels != cluster_index) for cluster_index in range(cluster_count)],
                        dtype=np.float32,
                    )
                new_label = int(
                    np.argmin(
                        distances
                        + penalty * disagreement
                        + float(overlap_lambda) * overlap_cost
                    )
                )
                local_index = int(np.flatnonzero(indexes == point_index)[0])
                if labels[point_index] != new_label:
                    labels[point_index] = new_label
                    current_labels[local_index] = new_label
                    changed = True
        if not changed:
            break
    return labels


def _normalize_edge_pairs(edge_pairs: np.ndarray, point_count: int) -> np.ndarray:
    edges = np.asarray(edge_pairs, dtype=np.int64).reshape(-1, 2)
    valid = (
        (edges[:, 0] >= 0) & (edges[:, 0] < point_count)
        & (edges[:, 1] >= 0) & (edges[:, 1] < point_count)
        & (edges[:, 0] != edges[:, 1])
    ) if len(edges) else np.zeros(0, dtype=bool)
    edges = edges[valid]
    if len(edges) == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.unique(np.sort(edges, axis=1), axis=0)


def _remap_edge_pairs(edge_pairs: np.ndarray, keep_indexes: np.ndarray) -> np.ndarray:
    remap = np.full(int(np.max(keep_indexes)) + 1, -1, dtype=np.int64)
    remap[keep_indexes] = np.arange(len(keep_indexes), dtype=np.int64)
    edges = _normalize_edge_pairs(edge_pairs, len(remap))
    edges = edges[(remap[edges[:, 0]] >= 0) & (remap[edges[:, 1]] >= 0)]
    return np.column_stack([remap[edges[:, 0]], remap[edges[:, 1]]]) if len(edges) else np.empty((0, 2), dtype=np.int64)


def _slice_edge_pairs(edge_pairs: np.ndarray, slice_indexes: np.ndarray) -> np.ndarray:
    return _remap_edge_pairs(edge_pairs, np.asarray(slice_indexes, dtype=np.int64))


def _convex_hull_outline(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(values) == 0:
        return np.empty((0, 2), dtype=np.float32)
    if len(values) < 3:
        if len(values) == 1:
            center = values[0]
            radius = 0.006
            return np.asarray(
                [
                    [center[0] - radius, center[1] - radius],
                    [center[0] + radius, center[1] - radius],
                    [center[0] + radius, center[1] + radius],
                    [center[0] - radius, center[1] + radius],
                    [center[0] - radius, center[1] - radius],
                ],
                dtype=np.float32,
            )
        return np.vstack([values, values[:1]]).astype(np.float32)
    hull = cv2.convexHull(values.reshape(-1, 1, 2)).reshape(-1, 2)
    return np.vstack([hull, hull[:1]]).astype(np.float32)


def compress_oklab_l(oklab_samples: np.ndarray, factor: float) -> np.ndarray:
    samples = np.asarray(oklab_samples, dtype=np.float32).reshape(-1, 3).copy()
    samples[:, 0] /= max(float(factor), 1e-6)
    return samples


def oklab_curve_points_for_models(
    models: list[dict[str, np.ndarray | float]],
    point_count: int = 160,
) -> list[np.ndarray]:
    curves = []
    for model in models:
        breakpoint = np.asarray(model["breakpoint"], dtype=np.float32).reshape(3)
        half_count = max(2, int(point_count) // 2)
        first = np.linspace(
            np.zeros(3, dtype=np.float32), breakpoint, half_count, dtype=np.float32
        )
        second = np.linspace(
            breakpoint,
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            half_count,
            dtype=np.float32,
        )
        curves.append(np.vstack([first, second[1:]]).astype(np.float32))
    return curves


def distances_to_nearest_oklab_cluster_points(
    oklab_sample: np.ndarray,
    cluster_samples: list[np.ndarray],
) -> np.ndarray:
    point = np.asarray(oklab_sample, dtype=np.float32).reshape(1, 3)
    distances = []
    for samples in cluster_samples:
        members = np.asarray(samples, dtype=np.float32).reshape(-1, 3)
        distances.append(
            float(np.linalg.norm(members - point, axis=1).min())
            if len(members)
            else np.inf
        )
    return np.asarray(distances, dtype=np.float32)


def oklab_gmm_cluster_memberships(
    oklab_samples: np.ndarray,
    *,
    target_clusters: int,
    min_brightness: float,
    max_brightness: float | None = None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    samples = np.asarray(oklab_samples, dtype=np.float32).reshape(-1, 3)
    labels = np.full(len(samples), -1, dtype=np.int32)
    lightness = samples[:, 0] * 255.0
    usable_mask = lightness >= float(min_brightness)
    if max_brightness is not None:
        usable_mask &= lightness <= float(max_brightness)
    usable_indexes = np.flatnonzero(usable_mask)
    if len(usable_indexes) == 0 or target_clusters <= 0:
        empty = np.empty((0, 3), dtype=np.float32)
        empty_cov = np.empty((0, 3, 3), dtype=np.float32)
        return labels, empty, [], empty_cov, np.empty(0, dtype=np.float32)
    points = samples[usable_indexes]
    component_count = min(int(target_clusters), len(points))
    means = _initialize_standard_gmm_means(points, component_count)
    covariance = _regularized_covariance(
        points,
        regularization=OKLAB_GMM_COVARIANCE_REGULARIZATION,
    )
    covariances = np.repeat(covariance[None, :, :], component_count, axis=0)
    weights = np.full(component_count, 1.0 / component_count, dtype=np.float32)
    previous_log_likelihood = -np.inf
    for _ in range(OKLAB_GMM_MAX_ITERATIONS):
        log_probabilities, _precisions = _gmm_log_probabilities(
            points,
            means,
            covariances,
            regularization=OKLAB_GMM_COVARIANCE_REGULARIZATION,
        )
        weighted_log_probabilities = log_probabilities + np.log(
            np.maximum(weights, 1e-12)
        )[None, :]
        log_likelihood = float(np.sum(_logsumexp(weighted_log_probabilities, axis=1)))
        responsibilities = _gmm_unbalanced_responsibilities(weighted_log_probabilities)
        effective_counts = np.maximum(responsibilities.sum(axis=0), 1e-6)
        means = (responsibilities.T @ points) / effective_counts[:, None]
        weights = (effective_counts / len(points)).astype(np.float32)
        deltas = points[:, None, :] - means[None, :, :]
        covariances = np.einsum(
            "nk,nki,nkj->kij", responsibilities, deltas, deltas
        ) / effective_counts[:, None, None]
        covariances += np.eye(3, dtype=np.float32)[None, :, :] * OKLAB_GMM_COVARIANCE_REGULARIZATION
        if abs(log_likelihood - previous_log_likelihood) <= OKLAB_GMM_TOLERANCE * len(points):
            break
        previous_log_likelihood = log_likelihood

    final_log_probabilities, _precisions = _gmm_log_probabilities(
        points,
        means,
        covariances,
        regularization=OKLAB_GMM_COVARIANCE_REGULARIZATION,
    )
    final_log_probabilities += np.log(np.maximum(weights, 1e-12))[None, :]
    usable_labels = np.argmax(final_log_probabilities, axis=1).astype(np.int32)
    labels[usable_indexes] = usable_labels
    cluster_members = [points[usable_labels == index] for index in range(component_count)]
    return labels, means.astype(np.float32), cluster_members, covariances, weights


def _initialize_standard_gmm_means(
    samples: np.ndarray,
    component_count: int,
) -> np.ndarray:
    points = np.asarray(samples, dtype=np.float32).reshape(-1, 3)
    if component_count <= 0 or len(points) == 0:
        return np.empty((0, 3), dtype=np.float32)
    selected = [int(np.argmax(points[:, 0]))]
    while len(selected) < component_count:
        distances = np.min(
            np.linalg.norm(points[:, None, :] - points[selected][None, :, :], axis=2),
            axis=1,
        )
        distances[selected] = -1.0
        selected.append(int(np.argmax(distances)))
    return points[np.asarray(selected, dtype=np.int32)].copy()


def oklab_angular_cluster_memberships(
    compressed_oklab_samples: np.ndarray,
    *,
    original_oklab_samples: np.ndarray,
    target_clusters: int,
    min_brightness: float,
    max_brightness: float | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    list[dict[str, np.ndarray | float | str]],
    np.ndarray,
]:
    fit_samples = np.asarray(compressed_oklab_samples, dtype=np.float32).reshape(-1, 3)
    original = np.asarray(original_oklab_samples, dtype=np.float32).reshape(-1, 3)
    labels = np.full(len(fit_samples), -1, dtype=np.int32)
    lightness = fit_samples[:, 0] * 255.0
    usable_mask = lightness >= float(min_brightness)
    if max_brightness is not None:
        usable_mask &= lightness <= float(max_brightness)
    usable_indexes = np.flatnonzero(usable_mask)
    if len(usable_indexes) == 0 or target_clusters <= 0:
        return labels, np.empty((0, 3), dtype=np.float32), [], [], np.empty(0, dtype=np.float32)

    points = original[usable_indexes]
    chroma_vectors = points[:, 1:3]
    chroma = np.linalg.norm(chroma_vectors, axis=1)
    component_count = min(int(target_clusters), len(points))
    neutral_count = 1 if component_count > 1 else component_count
    chromatic_count = component_count - neutral_count
    neutral_candidates = np.flatnonzero(
        (chroma <= max(OKLAB_ANGULAR_NEUTRAL_CHROMA, float(np.quantile(chroma, 0.20))))
        & (points[:, 0] >= float(np.median(points[:, 0])))
    )
    if len(neutral_candidates) == 0:
        neutral_candidates = np.asarray([int(np.argmin(chroma))], dtype=np.int32)
    labels[usable_indexes[neutral_candidates]] = 0

    chromatic_indexes = np.flatnonzero(labels[usable_indexes] < 0)
    chromatic_indexes = chromatic_indexes[chroma[chromatic_indexes] >= OKLAB_ANGULAR_MIN_CHROMA]
    if chromatic_count > 0 and len(chromatic_indexes) > 0:
        angular_labels = _circular_kmeans_labels(
            chroma_vectors[chromatic_indexes],
            chromatic_count,
        )
        labels[usable_indexes[chromatic_indexes]] = angular_labels + neutral_count

    remaining = np.flatnonzero(labels[usable_indexes] < 0)
    if len(remaining):
        labels[usable_indexes[remaining]] = neutral_count

    for _ in range(8):
        models = _fit_angular_models(points, labels[usable_indexes], component_count)
        scores = angular_cluster_log_likelihoods(points, models)
        next_labels = scores.argmax(axis=1).astype(np.int32)
        if np.array_equal(next_labels, labels[usable_indexes]):
            break
        labels[usable_indexes] = next_labels

    final_labels = labels[usable_indexes]
    models = _fit_angular_models(points, final_labels, component_count)
    clusters = [points[final_labels == index] for index in range(component_count)]
    centroids = np.asarray(
        [cluster.mean(axis=0) if len(cluster) else np.zeros(3) for cluster in clusters],
        dtype=np.float32,
    )
    return (
        labels,
        centroids,
        clusters,
        models,
        np.full(component_count, 1.0 / component_count, dtype=np.float32),
    )


def _circular_kmeans_labels(chroma_vectors: np.ndarray, cluster_count: int) -> np.ndarray:
    vectors = np.asarray(chroma_vectors, dtype=np.float32).reshape(-1, 2)
    unit = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-6)
    selected = [int(np.argmax(np.linalg.norm(vectors, axis=1)))]
    while len(selected) < cluster_count:
        separation = 1.0 - np.max(unit @ unit[selected].T, axis=1)
        separation[selected] = -1.0
        selected.append(int(np.argmax(separation)))
    directions = unit[np.asarray(selected, dtype=np.int32)].copy()
    labels = np.zeros(len(unit), dtype=np.int32)
    for _ in range(20):
        labels = np.argmax(unit @ directions.T, axis=1).astype(np.int32)
        next_directions = directions.copy()
        for index in range(cluster_count):
            members = unit[labels == index]
            if len(members):
                direction = members.mean(axis=0)
                next_directions[index] = direction / max(float(np.linalg.norm(direction)), 1e-6)
        if np.allclose(next_directions, directions, atol=1e-4):
            break
        directions = next_directions
    return labels


def _fit_angular_models(
    points: np.ndarray,
    labels: np.ndarray,
    component_count: int,
) -> list[dict[str, np.ndarray | float | str]]:
    values = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    assignments = np.asarray(labels, dtype=np.int32).reshape(-1)
    models = []
    for index in range(component_count):
        members = values[assignments == index]
        if len(members) == 0:
            members = values
        chroma_vectors = members[:, 1:3]
        chroma = np.linalg.norm(chroma_vectors, axis=1)
        if index == 0:
            models.append({
                "kind": "neutral",
                "lightness": float(np.median(members[:, 0])),
                "sigma_chroma": max(float(np.median(chroma) * 1.4826), 0.012),
                "sigma_lightness": max(float(np.median(np.abs(members[:, 0] - np.median(members[:, 0]))) * 1.4826), 0.04),
            })
            continue
        direction = chroma_vectors.mean(axis=0)
        if float(np.linalg.norm(direction)) <= 1e-6:
            direction = chroma_vectors[int(np.argmax(chroma))]
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        perpendicular = chroma_vectors - (chroma_vectors @ direction)[:, None] * direction[None, :]
        models.append({
            "kind": "chromatic",
            "direction": direction.astype(np.float32),
            "sigma_perpendicular": max(float(np.median(np.linalg.norm(perpendicular, axis=1)) * 1.4826), 0.008),
        })
    return models


def angular_cluster_log_likelihoods(
    samples: np.ndarray,
    models: list[dict[str, np.ndarray | float | str]],
    weights: np.ndarray | None = None,
) -> np.ndarray:
    points = np.asarray(samples, dtype=np.float32).reshape(-1, 3)
    if len(points) == 0 or not models:
        return np.empty((len(points), 0), dtype=np.float32)
    if weights is None:
        weights = np.full(len(models), 1.0 / len(models), dtype=np.float32)
    scores = []
    chroma_vectors = points[:, 1:3]
    chroma = np.linalg.norm(chroma_vectors, axis=1)
    for model, weight in zip(models, np.asarray(weights, dtype=np.float32)):
        if model["kind"] == "neutral":
            sigma_c = float(model["sigma_chroma"])
            sigma_l = float(model["sigma_lightness"])
            scores.append(
                -0.5 * (chroma / sigma_c) ** 2
                -0.5 * ((points[:, 0] - float(model["lightness"])) / sigma_l) ** 2
                -math.log(sigma_c * sigma_l)
                + math.log(max(float(weight), 1e-12))
            )
            continue
        direction = np.asarray(model["direction"], dtype=np.float32)
        projected = chroma_vectors @ direction
        perpendicular = chroma_vectors - projected[:, None] * direction[None, :]
        sigma = float(model["sigma_perpendicular"])
        scores.append(
            -0.5 * np.sum(perpendicular * perpendicular, axis=1) / (sigma * sigma)
            -math.log(sigma)
            + math.log(max(float(weight), 1e-12))
        )
    return np.asarray(scores, dtype=np.float32).T


def oklab_polyline_cluster_memberships(
    compressed_oklab_samples: np.ndarray,
    *,
    original_oklab_samples: np.ndarray,
    target_clusters: int,
    min_brightness: float,
    max_brightness: float | None = None,
    sticker_ids: np.ndarray | None = None,
    edge_pairs: np.ndarray | None = None,
    edge_lambda: float = 0.25,
    max_iterations: int = 12,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    list[dict[str, np.ndarray | float]],
    np.ndarray,
]:
    """Fit density-weighted black-to-white two-segment polylines."""
    fit_samples = np.asarray(compressed_oklab_samples, dtype=np.float32).reshape(-1, 3)
    original = np.asarray(original_oklab_samples, dtype=np.float32).reshape(-1, 3)
    if len(original) != len(fit_samples):
        raise ValueError("original_oklab_samples must align with compressed samples")
    labels = np.full(len(fit_samples), -1, dtype=np.int32)
    lightness = fit_samples[:, 0] * 255.0
    usable_mask = lightness >= float(min_brightness)
    if max_brightness is not None:
        usable_mask &= lightness <= float(max_brightness)
    usable_indexes = np.flatnonzero(usable_mask)
    if len(usable_indexes) == 0 or target_clusters <= 0:
        return labels, np.empty((0, 3), dtype=np.float32), [], [], np.empty(0, dtype=np.float32)

    fit_points = fit_samples[usable_indexes]
    original_points = original[usable_indexes]
    if sticker_ids is not None:
        sticker_ids = np.asarray(sticker_ids).reshape(-1)
        if len(sticker_ids) != len(fit_samples):
            raise ValueError("sticker_ids must align with Oklab samples")
    if edge_pairs is None and sticker_ids is not None:
        edge_pairs = _consecutive_same_group_edges(sticker_ids)
    local_edges = (
        _remap_edge_pairs(edge_pairs, usable_indexes)
        if edge_pairs is not None else np.empty((0, 2), dtype=np.int64)
    )
    adjacency = [[] for _ in range(len(fit_points))]
    for left, right in local_edges:
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))
    component_count = min(int(target_clusters), len(fit_points))
    means = _initialize_standard_gmm_means(fit_points, component_count)
    labels[usable_indexes] = np.linalg.norm(
        fit_points[:, None, :] - means[None, :, :], axis=2
    ).argmin(axis=1)

    density_weights = _local_density_weights(original_points)
    for _ in range(max(1, int(max_iterations))):
        fit_models = _fit_polyline_models(
            original_points,
            labels[usable_indexes],
            component_count,
            density_weights,
        )
        scores = polyline_cluster_log_likelihoods(original_points, fit_models)
        if len(local_edges) and float(edge_lambda) > 0.0:
            for point_index, neighbors in enumerate(adjacency):
                if not neighbors:
                    continue
                neighbor_labels = labels[usable_indexes][neighbors]
                scores[point_index] += float(edge_lambda) * np.asarray(
                    [np.mean(neighbor_labels == cluster_index) for cluster_index in range(component_count)],
                    dtype=np.float32,
                )
        next_labels = scores.argmax(axis=1).astype(np.int32)
        if np.array_equal(next_labels, labels[usable_indexes]):
            break
        labels[usable_indexes] = next_labels

    final_labels = labels[usable_indexes]
    models = _fit_polyline_models(
        original_points,
        final_labels,
        component_count,
        density_weights,
    )
    clusters = [original_points[final_labels == index] for index in range(component_count)]
    centroids = np.asarray(
        [cluster.mean(axis=0) if len(cluster) else np.zeros(3) for cluster in clusters],
        dtype=np.float32,
    )
    counts = np.bincount(final_labels, minlength=component_count).astype(np.float32)
    weights = counts / max(float(counts.sum()), 1.0)
    return labels, centroids, clusters, models, weights


def _fit_polyline_models(
    samples: np.ndarray,
    labels: np.ndarray,
    component_count: int,
    density_weights: np.ndarray,
) -> list[dict[str, np.ndarray | float]]:
    points = np.asarray(samples, dtype=np.float32).reshape(-1, 3)
    assignments = np.asarray(labels, dtype=np.int32).reshape(-1)
    models = []
    for index in range(component_count):
        members = points[assignments == index]
        member_indexes = np.flatnonzero(assignments == index)
        if len(member_indexes) == 0:
            member_indexes = np.arange(len(points), dtype=np.int32)
        members = points[member_indexes]
        member_weights = density_weights[member_indexes]
        density_candidate_indexes = np.argsort(member_weights)[-min(32, len(members)):]
        radial_order = np.argsort(np.linalg.norm(members, axis=1))
        radial_candidate_indexes = radial_order[
            np.linspace(
                0,
                len(radial_order) - 1,
                min(64, len(radial_order)),
                dtype=np.int32,
            )
        ]
        candidate_indexes = np.unique(
            np.concatenate([density_candidate_indexes, radial_candidate_indexes])
        )
        candidates = members[candidate_indexes]
        best_breakpoint = candidates[0]
        best_score = np.inf
        for breakpoint in candidates:
            score = _one_sided_spine_fit_score(members, breakpoint, member_weights)
            if score < best_score:
                best_score = score
                best_breakpoint = breakpoint
        residuals = _polyline_distances(members, best_breakpoint)
        models.append({
            "breakpoint": np.asarray(best_breakpoint, dtype=np.float32),
            "sigma": max(
                float(np.sqrt(np.average(residuals * residuals, weights=member_weights))),
                0.005,
            ),
            "inside_penalty": POLYLINE_INSIDE_PENALTY,
        })
    return models


def polyline_cluster_log_likelihoods(
    samples: np.ndarray,
    models: list[dict[str, np.ndarray | float]],
    weights: np.ndarray | None = None,
) -> np.ndarray:
    points = np.asarray(samples, dtype=np.float32).reshape(-1, 3)
    if len(points) == 0 or not models:
        return np.empty((len(points), 0), dtype=np.float32)
    if weights is None:
        weights = np.full(len(models), 1.0 / len(models), dtype=np.float32)
    scores = []
    for model, weight in zip(models, np.asarray(weights, dtype=np.float32)):
        distances = _polyline_distances(
            points,
            np.asarray(model["breakpoint"], dtype=np.float32),
        )
        sigma = float(model["sigma"])
        radial_gaps = _radial_spine_gaps(
            points,
            np.asarray(model["breakpoint"], dtype=np.float32),
        )
        inside_scale = max(float(model["inside_penalty"]), 1.0)
        distance_score = -0.5 * (distances / sigma) ** 2
        inside_score = np.where(
            radial_gaps < 0.0,
            -0.5 * inside_scale * (radial_gaps / sigma) ** 2,
            0.0,
        )
        scores.append(
            distance_score
            + inside_score
            -math.log(sigma)
            + math.log(max(float(weight), 1e-12))
        )
    return np.asarray(scores, dtype=np.float32).T


def _polyline_distances(points: np.ndarray, breakpoint: np.ndarray) -> np.ndarray:
    source = np.zeros(3, dtype=np.float32)
    destination = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return np.minimum(
        _oklab_point_to_segment_distances(points, source, breakpoint),
        _oklab_point_to_segment_distances(points, breakpoint, destination),
    )


def _polyline_fit_score(
    points: np.ndarray,
    breakpoint: np.ndarray,
    density_weights: np.ndarray,
) -> float:
    distances = _polyline_distances(points, breakpoint)
    point_score = float(
        np.average(distances * distances, weights=density_weights)
    )
    radii = np.linalg.norm(points, axis=1)
    order = np.argsort(radii)
    bin_count = min(POLYLINE_COVERAGE_BINS, len(order))
    coverage_scores = []
    for bin_indexes in np.array_split(order, bin_count):
        if len(bin_indexes):
            coverage_scores.append(float(np.mean(distances[bin_indexes] ** 2)))
    coverage_score = float(np.mean(coverage_scores)) if coverage_scores else point_score
    return (
        (1.0 - POLYLINE_COVERAGE_MIX) * point_score
        + POLYLINE_COVERAGE_MIX * coverage_score
    )


def _one_sided_spine_fit_score(
    points: np.ndarray,
    breakpoint: np.ndarray,
    density_weights: np.ndarray,
) -> float:
    distances = _polyline_distances(points, breakpoint)
    point_score = float(np.average(distances * distances, weights=density_weights))
    gaps = _radial_spine_gaps(points, breakpoint)
    inside = np.minimum(gaps, 0.0)
    outside = np.maximum(gaps, 0.0)
    inside_score = float(np.average(inside * inside, weights=density_weights))
    outside_score = float(np.average(outside * outside, weights=density_weights))
    one_sided_score = POLYLINE_INSIDE_PENALTY * inside_score + outside_score
    coverage_score = _radial_coverage_score(points, breakpoint, distances)
    return (
        (1.0 - POLYLINE_COVERAGE_MIX) * (point_score + one_sided_score)
        + POLYLINE_COVERAGE_MIX * coverage_score
    )


def _radial_coverage_score(
    points: np.ndarray,
    breakpoint: np.ndarray,
    distances: np.ndarray,
) -> float:
    radii = np.linalg.norm(points, axis=1)
    order = np.argsort(radii)
    bin_count = min(POLYLINE_COVERAGE_BINS, len(order))
    coverage_scores = []
    for bin_indexes in np.array_split(order, bin_count):
        if len(bin_indexes):
            gaps = _radial_spine_gaps(points[bin_indexes], breakpoint)
            coverage_scores.append(
                float(np.mean(distances[bin_indexes] ** 2))
                + POLYLINE_INSIDE_PENALTY
                * float(np.mean(np.minimum(gaps, 0.0) ** 2))
            )
    return float(np.mean(coverage_scores)) if coverage_scores else float(np.mean(distances**2))


def _radial_spine_gaps(points: np.ndarray, breakpoint: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    pivot = np.asarray(breakpoint, dtype=np.float32).reshape(3)
    lightness = np.clip(values[:, 0], 0.0, 1.0)
    pivot_lightness = float(np.clip(pivot[0], 1e-4, 1.0 - 1e-4))
    pivot_chroma = float(np.linalg.norm(pivot[1:3]))
    spine_chroma = np.where(
        lightness <= pivot_lightness,
        pivot_chroma * lightness / pivot_lightness,
        pivot_chroma * (1.0 - lightness) / (1.0 - pivot_lightness),
    )
    sample_chroma = np.linalg.norm(values[:, 1:3], axis=1)
    return (sample_chroma - spine_chroma).astype(np.float32)


def _oklab_point_to_segment_distances(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    direction = np.asarray(end, dtype=np.float32) - np.asarray(start, dtype=np.float32)
    denominator = max(float(direction @ direction), 1e-8)
    projection = np.clip(
        ((values - start[None, :]) @ direction) / denominator,
        0.0,
        1.0,
    )
    closest = start[None, :] + projection[:, None] * direction[None, :]
    return np.linalg.norm(values - closest, axis=1).astype(np.float32)


def _local_density_weights(points: np.ndarray) -> np.ndarray:
    """Estimate local ridge density and return capped fitting weights."""
    values = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(values) == 0:
        return np.empty(0, dtype=np.float32)
    quantized = np.floor(values / 0.015).astype(np.int32)
    _unique, inverse, counts = np.unique(
        quantized,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    density = counts[inverse].astype(np.float32)
    density_weights = np.power(density, DENSITY_WEIGHT_POWER)
    density_weights /= max(float(density_weights.mean()), 1e-6)
    weights = (1.0 - DENSITY_WEIGHT_MIX) + DENSITY_WEIGHT_MIX * density_weights
    return np.clip(weights, 1.0 / DENSITY_WEIGHT_MAX, DENSITY_WEIGHT_MAX)


def rgb_brightness_mask(
    rgb_samples: np.ndarray,
    *,
    min_brightness: float,
    max_brightness: float | None,
) -> np.ndarray:
    samples = np.asarray(rgb_samples, dtype=np.float32).reshape(-1, 3)
    brightness = np.linalg.norm(samples, axis=1)
    mask = brightness >= float(min_brightness)
    if max_brightness is not None:
        mask &= brightness <= float(max_brightness)
    return mask


def hsv_value_mask(
    hsv_samples: np.ndarray,
    *,
    min_brightness: float,
    max_brightness: float | None,
) -> np.ndarray:
    samples = np.asarray(hsv_samples, dtype=np.float32).reshape(-1, 3)
    values = samples[:, 2]
    mask = values >= float(min_brightness)
    if max_brightness is not None:
        mask &= values <= float(max_brightness)
    return mask


def hsv_distance_matrix(samples: np.ndarray, centers: np.ndarray) -> np.ndarray:
    points = np.asarray(samples, dtype=np.float32).reshape(-1, 3)
    reference = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
    if len(points) == 0 or len(reference) == 0:
        return np.empty((len(points), len(reference)), dtype=np.float32)
    deltas = np.abs(points[:, None, :] - reference[None, :, :])
    hue_delta = np.minimum(deltas[:, :, 0], 180.0 - deltas[:, :, 0])
    distances = np.sqrt(hue_delta * hue_delta + deltas[:, :, 1] ** 2 + deltas[:, :, 2] ** 2)
    return distances.astype(np.float32)


def distances_to_hsv_clusters(
    hsv_sample: np.ndarray,
    cluster_samples: list[np.ndarray],
) -> np.ndarray:
    point = np.asarray(hsv_sample, dtype=np.float32).reshape(1, 3)
    distances = []
    for samples in cluster_samples:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1, 3)
        if len(samples) == 0:
            distances.append(np.inf)
            continue
        distances.append(float(hsv_distance_matrix(point, samples).min()))
    return np.asarray(distances, dtype=np.float32)


def dense_ray_rgb_cluster_memberships(
    rgb_samples: np.ndarray,
    *,
    target_clusters: int,
    min_brightness: float,
    max_brightness: float | None = None,
    k_neighbors: int = DENSE_RAY_K_NEIGHBORS,
    max_fit_samples: int = DENSE_RAY_MAX_FIT_SAMPLES,
    min_anchor_distance: float = DENSE_RAY_MIN_ANCHOR_DISTANCE,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    samples = np.asarray(rgb_samples, dtype=np.float32).reshape(-1, 3)
    sample_cluster_indexes = np.full(len(samples), -1, dtype=np.int32)
    if len(samples) == 0 or target_clusters <= 0:
        return (
            sample_cluster_indexes,
            np.empty((0, 3), dtype=np.float32),
            [],
            np.empty((0, 3), dtype=np.float32),
        )

    usable_indexes = np.flatnonzero(
        rgb_brightness_mask(
            samples,
            min_brightness=min_brightness,
            max_brightness=max_brightness,
        )
    )
    usable = samples[usable_indexes]
    component_count = min(int(target_clusters), len(usable))
    if component_count <= 0:
        return (
            sample_cluster_indexes,
            np.empty((0, 3), dtype=np.float32),
            [],
            np.empty((0, 3), dtype=np.float32),
        )

    fit_local_indexes = _evenly_spaced_indexes(len(usable), max_fit_samples)
    fit = usable[fit_local_indexes]
    anchors = dense_rgb_anchor_points(
        fit,
        target_clusters=component_count,
        k_neighbors=k_neighbors,
        min_anchor_distance=min_anchor_distance,
    )
    if len(anchors) == 0:
        return (
            sample_cluster_indexes,
            np.empty((0, 3), dtype=np.float32),
            [],
            np.empty((0, 3), dtype=np.float32),
        )

    ray_directions = _normalize_rgb_vectors(anchors)
    distances = distances_to_rgb_rays(usable, ray_directions)
    labels = distances.argmin(axis=1).astype(np.int32)
    sample_cluster_indexes[usable_indexes] = labels

    centroids = []
    cluster_samples = []
    retained_rays = []
    remapped_indexes = np.full(len(samples), -1, dtype=np.int32)
    for cluster_index in range(len(ray_directions)):
        member_indexes = usable_indexes[labels == cluster_index]
        if len(member_indexes) == 0:
            continue
        members = samples[member_indexes]
        centroids.append(members.mean(axis=0))
        cluster_samples.append(members)
        retained_rays.append(ray_directions[cluster_index])
        remapped_indexes[member_indexes] = len(centroids) - 1

    if not centroids:
        return (
            sample_cluster_indexes,
            np.empty((0, 3), dtype=np.float32),
            [],
            np.empty((0, 3), dtype=np.float32),
        )
    return (
        remapped_indexes,
        np.asarray(centroids, dtype=np.float32).reshape(-1, 3),
        cluster_samples,
        np.asarray(retained_rays, dtype=np.float32).reshape(-1, 3),
    )


def dense_rgb_anchor_points(
    rgb_samples: np.ndarray,
    *,
    target_clusters: int,
    k_neighbors: int,
    min_anchor_distance: float,
) -> np.ndarray:
    samples = np.asarray(rgb_samples, dtype=np.float32).reshape(-1, 3)
    if len(samples) == 0 or target_clusters <= 0:
        return np.empty((0, 3), dtype=np.float32)
    if len(samples) == 1:
        return samples.copy()

    distances = _pairwise_rgb_distances(samples)
    neighbor_count = min(max(1, int(k_neighbors)), len(samples) - 1)
    nearest = np.partition(distances, neighbor_count, axis=1)[:, 1 : neighbor_count + 1]
    density = 1.0 / np.maximum(nearest.mean(axis=1), 1e-6)
    selected = []
    for candidate in np.argsort(-density):
        if len(selected) >= target_clusters:
            break
        if selected and min_anchor_distance > 0.0:
            selected_points = samples[np.asarray(selected, dtype=np.int32)]
            nearest_selected = float(
                np.min(np.linalg.norm(selected_points - samples[candidate][None, :], axis=1))
            )
            if nearest_selected < min_anchor_distance:
                continue
        selected.append(int(candidate))

    if len(selected) < target_clusters:
        for candidate in np.argsort(-density):
            if len(selected) >= target_clusters:
                break
            if int(candidate) not in selected:
                selected.append(int(candidate))
    return samples[np.asarray(selected, dtype=np.int32)].astype(np.float32, copy=True)


def distances_to_rgb_rays(
    rgb_samples: np.ndarray,
    ray_directions: np.ndarray,
) -> np.ndarray:
    points = np.asarray(rgb_samples, dtype=np.float32).reshape(-1, 3)
    rays = np.asarray(ray_directions, dtype=np.float32).reshape(-1, 3)
    if len(points) == 0 or len(rays) == 0:
        return np.empty((len(points), len(rays)), dtype=np.float32)

    rays = _normalize_rgb_vectors(rays)
    parallel = points @ rays.T
    projections = parallel[:, :, None] * rays[None, :, :]
    perpendicular = points[:, None, :] - projections
    distances = np.linalg.norm(perpendicular, axis=2)
    return distances.astype(np.float32)


def density_peak_rgb_cluster_memberships(
    rgb_samples: np.ndarray,
    *,
    target_clusters: int,
    min_brightness: float,
    max_brightness: float | None = None,
    k_neighbors: int = DENSITY_PEAK_K_NEIGHBORS,
    max_fit_samples: int = DENSITY_PEAK_MAX_FIT_SAMPLES,
    min_peak_distance: float = DENSITY_PEAK_MIN_PEAK_DISTANCE,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    samples = np.asarray(rgb_samples, dtype=np.float32).reshape(-1, 3)
    sample_cluster_indexes = np.full(len(samples), -1, dtype=np.int32)
    if len(samples) == 0 or target_clusters <= 0:
        return sample_cluster_indexes, np.empty((0, 3), dtype=np.float32), []

    usable_indexes = np.flatnonzero(
        rgb_brightness_mask(
            samples,
            min_brightness=min_brightness,
            max_brightness=max_brightness,
        )
    )
    usable = samples[usable_indexes]
    component_count = min(int(target_clusters), len(usable))
    if component_count <= 0:
        return sample_cluster_indexes, np.empty((0, 3), dtype=np.float32), []

    fit_local_indexes = _evenly_spaced_indexes(len(usable), max_fit_samples)
    fit = usable[fit_local_indexes]
    fit_labels, peak_indexes = _density_peak_fit_labels(
        fit,
        target_clusters=component_count,
        k_neighbors=k_neighbors,
        min_peak_distance=min_peak_distance,
    )
    if len(peak_indexes) == 0:
        return sample_cluster_indexes, np.empty((0, 3), dtype=np.float32), []

    peak_points = fit[np.asarray(peak_indexes, dtype=np.int32)]
    fit_cluster_samples = [
        fit[fit_labels == cluster_index]
        for cluster_index in range(len(peak_points))
    ]
    labels = _assign_samples_to_density_peak_clusters(
        usable,
        peak_points,
        fit_cluster_samples,
    )
    sample_cluster_indexes[usable_indexes] = labels

    centroids = []
    cluster_samples = []
    remapped_indexes = np.full(len(samples), -1, dtype=np.int32)
    for cluster_index in range(len(peak_points)):
        member_indexes = usable_indexes[labels == cluster_index]
        if len(member_indexes) == 0:
            continue
        members = samples[member_indexes]
        centroids.append(members.mean(axis=0))
        cluster_samples.append(members)
        remapped_indexes[member_indexes] = len(centroids) - 1

    if not centroids:
        return sample_cluster_indexes, np.empty((0, 3), dtype=np.float32), []
    return remapped_indexes, np.asarray(centroids, dtype=np.float32).reshape(-1, 3), cluster_samples


def _density_peak_fit_labels(
    samples: np.ndarray,
    *,
    target_clusters: int,
    k_neighbors: int,
    min_peak_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(samples, dtype=np.float32).reshape(-1, 3)
    labels = np.full(len(points), -1, dtype=np.int32)
    if len(points) == 0 or target_clusters <= 0:
        return labels, np.empty(0, dtype=np.int32)
    if len(points) == 1:
        return np.zeros(1, dtype=np.int32), np.zeros(1, dtype=np.int32)

    distances = _pairwise_rgb_distances(points)
    neighbor_count = min(max(1, int(k_neighbors)), len(points) - 1)
    nearest = np.partition(distances, neighbor_count, axis=1)[:, 1 : neighbor_count + 1]
    density = 1.0 / np.maximum(nearest.mean(axis=1), 1e-6)
    peak_indexes = _density_peak_indexes(
        points,
        distances,
        density,
        target_clusters=target_clusters,
        min_peak_distance=min_peak_distance,
    )
    if len(peak_indexes) == 0:
        return labels, peak_indexes

    for cluster_index, peak_index in enumerate(peak_indexes):
        labels[peak_index] = cluster_index

    order = np.argsort(-density)
    for sample_index in order:
        if labels[sample_index] >= 0:
            continue
        higher_density = order[density[order] > density[sample_index]]
        if len(higher_density) == 0:
            labels[sample_index] = int(np.argmin(distances[sample_index, peak_indexes]))
            continue
        nearest_higher = int(higher_density[np.argmin(distances[sample_index, higher_density])])
        labels[sample_index] = labels[nearest_higher]
        if labels[sample_index] < 0:
            labels[sample_index] = int(np.argmin(distances[sample_index, peak_indexes]))
    return labels, peak_indexes


def _density_peak_indexes(
    samples: np.ndarray,
    distances: np.ndarray,
    density: np.ndarray,
    *,
    target_clusters: int,
    min_peak_distance: float,
) -> np.ndarray:
    points = np.asarray(samples, dtype=np.float32).reshape(-1, 3)
    rho = np.asarray(density, dtype=np.float32).reshape(-1)
    order = np.argsort(-rho)
    higher_distance = np.empty(len(points), dtype=np.float32)
    for rank, sample_index in enumerate(order):
        if rank == 0:
            higher_distance[sample_index] = float(distances[sample_index].max())
            continue
        higher = order[:rank]
        higher_distance[sample_index] = float(distances[sample_index, higher].min())

    scores = rho * higher_distance
    peak_indexes = []
    for candidate in np.argsort(-scores):
        if len(peak_indexes) >= target_clusters:
            break
        if peak_indexes and min_peak_distance > 0.0:
            nearest_peak_distance = float(distances[candidate, np.asarray(peak_indexes)].min())
            if nearest_peak_distance < min_peak_distance:
                continue
        peak_indexes.append(int(candidate))

    if len(peak_indexes) < target_clusters:
        for candidate in order:
            if len(peak_indexes) >= target_clusters:
                break
            if int(candidate) not in peak_indexes:
                peak_indexes.append(int(candidate))
    return np.asarray(peak_indexes, dtype=np.int32)


def _assign_samples_to_density_peak_clusters(
    samples: np.ndarray,
    peak_points: np.ndarray,
    fit_cluster_samples: list[np.ndarray],
) -> np.ndarray:
    points = np.asarray(samples, dtype=np.float32).reshape(-1, 3)
    peaks = np.asarray(peak_points, dtype=np.float32).reshape(-1, 3)
    if len(points) == 0 or len(peaks) == 0:
        return np.full(len(points), -1, dtype=np.int32)

    basin_samples = [
        np.asarray(cluster, dtype=np.float32).reshape(-1, 3)
        for cluster in fit_cluster_samples
    ]
    if not any(len(cluster) > 0 for cluster in basin_samples):
        distances = np.linalg.norm(points[:, None, :] - peaks[None, :, :], axis=2)
        return distances.argmin(axis=1).astype(np.int32)

    labels = np.full(len(points), -1, dtype=np.int32)
    chunk_size = 1024
    for start in range(0, len(points), chunk_size):
        chunk = points[start : start + chunk_size]
        best_distances = np.full(len(chunk), np.inf, dtype=np.float32)
        best_labels = np.full(len(chunk), -1, dtype=np.int32)
        for cluster_index, cluster in enumerate(basin_samples):
            if len(cluster) == 0:
                continue
            distances = np.linalg.norm(chunk[:, None, :] - cluster[None, :, :], axis=2)
            nearest = distances.min(axis=1)
            improved = nearest < best_distances
            best_distances[improved] = nearest[improved]
            best_labels[improved] = cluster_index
        labels[start : start + len(chunk)] = best_labels
    return labels


def _pairwise_rgb_distances(samples: np.ndarray) -> np.ndarray:
    points = np.asarray(samples, dtype=np.float32).reshape(-1, 3)
    deltas = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(deltas, axis=2).astype(np.float32)
    np.fill_diagonal(distances, np.inf)
    return distances


def _gmm_dense_core_local_indexes(
    samples: np.ndarray,
    *,
    k_neighbors: int,
    drop_quantile: float,
    min_core_count: int = 1,
) -> np.ndarray:
    points = np.asarray(samples, dtype=np.float32).reshape(-1, 3)
    if len(points) == 0:
        return np.empty(0, dtype=np.int32)
    if k_neighbors <= 0 or drop_quantile <= 0.0 or len(points) <= 2:
        return np.arange(len(points), dtype=np.int32)

    bin_size = max(1, int(k_neighbors))
    bin_coords = np.floor(points / float(bin_size)).astype(np.int32)
    unique_bins, inverse = np.unique(bin_coords, axis=0, return_inverse=True)
    bin_counts = np.bincount(inverse).astype(np.float32)
    bin_lookup = {
        (int(coord[0]), int(coord[1]), int(coord[2])): index
        for index, coord in enumerate(unique_bins)
    }
    bin_density = np.zeros(len(unique_bins), dtype=np.float32)
    for index, coord in enumerate(unique_bins):
        density = 0.0
        base = (int(coord[0]), int(coord[1]), int(coord[2]))
        for red_offset in (-1, 0, 1):
            for green_offset in (-1, 0, 1):
                for blue_offset in (-1, 0, 1):
                    neighbor = (
                        base[0] + red_offset,
                        base[1] + green_offset,
                        base[2] + blue_offset,
                    )
                    neighbor_index = bin_lookup.get(neighbor)
                    if neighbor_index is not None:
                        density += float(bin_counts[neighbor_index])
        bin_density[index] = density
    density = bin_density[inverse]
    keep_fraction = 1.0 - float(np.clip(drop_quantile, 0.0, 0.9999))
    keep_count = int(np.ceil(len(points) * keep_fraction - 1e-9))
    keep_count = max(1, min(len(points), keep_count, len(points)))
    keep_count = max(min(len(points), int(min_core_count)), keep_count)
    order = np.lexsort((np.arange(len(points), dtype=np.int32), -density))
    return order[:keep_count].astype(np.int32)


def _evenly_spaced_indexes(length: int, max_count: int) -> np.ndarray:
    if length <= 0 or max_count <= 0:
        return np.empty(0, dtype=np.int32)
    if length <= max_count:
        return np.arange(length, dtype=np.int32)
    return np.linspace(0, length - 1, int(max_count), dtype=np.int32)


def _valid_rgb_prior_values(prior_rgb_values: np.ndarray | None) -> np.ndarray:
    if prior_rgb_values is None:
        return np.empty((0, 3), dtype=np.float32)
    priors = np.asarray(prior_rgb_values, dtype=np.float32).reshape(-1, 3)
    if len(priors) == 0:
        return np.empty((0, 3), dtype=np.float32)
    brightness = np.linalg.norm(priors, axis=1)
    priors = priors[brightness > 1e-6]
    if len(priors) == 0:
        return np.empty((0, 3), dtype=np.float32)
    return np.clip(priors, 0.0, 255.0).astype(np.float32)


def _rgb_prior_directions(
    prior_rgb_values: np.ndarray | None,
    component_count: int,
) -> np.ndarray:
    priors = _valid_rgb_prior_values(prior_rgb_values)
    if component_count <= 0 or len(priors) == 0:
        return np.empty((0, 3), dtype=np.float32)
    return _normalize_rgb_vectors(priors[:component_count])


def _prior_gmm_centroids(
    samples: np.ndarray,
    prior_directions: np.ndarray,
) -> np.ndarray:
    points = np.asarray(samples, dtype=np.float32).reshape(-1, 3)
    directions = np.asarray(prior_directions, dtype=np.float32).reshape(-1, 3)
    if len(points) == 0 or len(directions) == 0:
        return np.empty((0, 3), dtype=np.float32)

    sample_directions = _normalize_rgb_vectors(points)
    scores = sample_directions @ directions.T
    nearest_prior = scores.argmax(axis=1)
    fallback_radius = float(np.median(np.linalg.norm(points, axis=1)))
    centroids = []
    for index, prior_direction in enumerate(directions):
        projections = points @ prior_direction
        member_projections = projections[nearest_prior == index]
        if len(member_projections) == 0:
            radius = fallback_radius
        else:
            radius = float(np.mean(np.maximum(member_projections, 1e-6)))
        centroids.append(prior_direction * max(radius, 1e-6))
    return np.asarray(centroids, dtype=np.float32).reshape(-1, 3)


def _apply_gmm_prior_to_centroids(
    centroids: np.ndarray,
    prior_directions: np.ndarray,
    *,
    prior_strength: float,
) -> np.ndarray:
    means = np.asarray(centroids, dtype=np.float32).reshape(-1, 3)
    directions = np.asarray(prior_directions, dtype=np.float32).reshape(-1, 3)
    strength = float(np.clip(prior_strength, 0.0, 1.0))
    if strength <= 0.0 or len(means) == 0 or len(directions) == 0:
        return means

    adjusted = means.copy()
    for index in range(min(len(adjusted), len(directions))):
        prior_direction = directions[index]
        radius = float(adjusted[index] @ prior_direction)
        if radius <= 1e-6:
            radius = float(np.linalg.norm(adjusted[index]))
        prior_mean = prior_direction * max(radius, 1e-6)
        adjusted[index] = adjusted[index] * (1.0 - strength) + prior_mean * strength
    return adjusted.astype(np.float32)


def balanced_gmm_rgb_cluster_memberships(
    rgb_samples: np.ndarray,
    *,
    target_clusters: int,
    min_brightness: float,
    max_brightness: float | None = None,
    max_mahalanobis_distance: float = GMM_MAX_MAHALANOBIS_DISTANCE,
    regularization: float = GMM_REGULARIZATION,
    ray_variance_floor: float = GMM_RAY_VARIANCE_FLOOR,
    constrain_covariance_to_ray: bool = GMM_CONSTRAIN_COVARIANCE_TO_RAY,
    perpendicular_variance_floor: float = GMM_PERPENDICULAR_VARIANCE_FLOOR,
    perpendicular_variance_ceiling: float = GMM_PERPENDICULAR_VARIANCE_CEILING,
    dense_core_k_neighbors: int = GMM_DENSE_CORE_K_NEIGHBORS,
    dense_core_quantile: float = GMM_DENSE_CORE_QUANTILE,
    dense_core_outliers_unknown: bool = False,
    prior_rgb_values: np.ndarray | None = None,
    prior_strength: float = GMM_PRIOR_STRENGTH,
    min_assignment_confidence: float = GMM_MIN_ASSIGNMENT_CONFIDENCE,
    assignment_balance: float = GMM_ASSIGNMENT_BALANCE,
    balance_iterations: int = GMM_BALANCE_ITERATIONS,
    max_iterations: int = GMM_MAX_ITERATIONS,
    tolerance: float = GMM_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    memberships = balanced_gmm_rgb_cluster_memberships_with_core_mask(
        rgb_samples,
        target_clusters=target_clusters,
        min_brightness=min_brightness,
        max_brightness=max_brightness,
        max_mahalanobis_distance=max_mahalanobis_distance,
        regularization=regularization,
        ray_variance_floor=ray_variance_floor,
        constrain_covariance_to_ray=constrain_covariance_to_ray,
        perpendicular_variance_floor=perpendicular_variance_floor,
        perpendicular_variance_ceiling=perpendicular_variance_ceiling,
        dense_core_k_neighbors=dense_core_k_neighbors,
        dense_core_quantile=dense_core_quantile,
        dense_core_outliers_unknown=dense_core_outliers_unknown,
        prior_rgb_values=prior_rgb_values,
        prior_strength=prior_strength,
        min_assignment_confidence=min_assignment_confidence,
        assignment_balance=assignment_balance,
        balance_iterations=balance_iterations,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    labels, centroids, cluster_samples, covariances, precisions, _dense_core_mask = memberships
    return labels, centroids, cluster_samples, covariances, precisions


def balanced_gmm_rgb_cluster_memberships_with_core_mask(
    rgb_samples: np.ndarray,
    *,
    target_clusters: int,
    min_brightness: float,
    max_brightness: float | None = None,
    max_mahalanobis_distance: float = GMM_MAX_MAHALANOBIS_DISTANCE,
    regularization: float = GMM_REGULARIZATION,
    ray_variance_floor: float = GMM_RAY_VARIANCE_FLOOR,
    constrain_covariance_to_ray: bool = GMM_CONSTRAIN_COVARIANCE_TO_RAY,
    perpendicular_variance_floor: float = GMM_PERPENDICULAR_VARIANCE_FLOOR,
    perpendicular_variance_ceiling: float = GMM_PERPENDICULAR_VARIANCE_CEILING,
    dense_core_k_neighbors: int = GMM_DENSE_CORE_K_NEIGHBORS,
    dense_core_quantile: float = GMM_DENSE_CORE_QUANTILE,
    dense_core_outliers_unknown: bool = False,
    prior_rgb_values: np.ndarray | None = None,
    prior_strength: float = GMM_PRIOR_STRENGTH,
    min_assignment_confidence: float = GMM_MIN_ASSIGNMENT_CONFIDENCE,
    assignment_balance: float = GMM_ASSIGNMENT_BALANCE,
    balance_iterations: int = GMM_BALANCE_ITERATIONS,
    max_iterations: int = GMM_MAX_ITERATIONS,
    tolerance: float = GMM_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    samples = np.asarray(rgb_samples, dtype=np.float32).reshape(-1, 3)
    sample_cluster_indexes = np.full(len(samples), -1, dtype=np.int32)
    dense_core_sample_mask = np.zeros(len(samples), dtype=bool)
    if len(samples) == 0 or target_clusters <= 0:
        empty_cov = np.empty((0, 3, 3), dtype=np.float32)
        return (
            sample_cluster_indexes,
            np.empty((0, 3), dtype=np.float32),
            [],
            empty_cov,
            empty_cov,
            dense_core_sample_mask,
        )

    usable_indexes = np.flatnonzero(
        rgb_brightness_mask(
            samples,
            min_brightness=min_brightness,
            max_brightness=max_brightness,
        )
    )
    usable = samples[usable_indexes]
    fit_local_indexes = _gmm_dense_core_local_indexes(
        usable,
        k_neighbors=dense_core_k_neighbors,
        drop_quantile=dense_core_quantile,
        min_core_count=target_clusters,
    )
    fit_samples = usable[fit_local_indexes]
    prior_values = _valid_rgb_prior_values(prior_rgb_values)
    component_limit = len(prior_values) if len(prior_values) > 0 else int(target_clusters)
    component_count = min(int(target_clusters), component_limit, len(fit_samples))
    if component_count <= 0:
        empty_cov = np.empty((0, 3, 3), dtype=np.float32)
        return (
            sample_cluster_indexes,
            np.empty((0, 3), dtype=np.float32),
            [],
            empty_cov,
            empty_cov,
            dense_core_sample_mask,
        )
    dense_core_sample_mask[usable_indexes[fit_local_indexes]] = True

    prior_directions = _rgb_prior_directions(prior_values, component_count)
    centroids = (
        _prior_gmm_centroids(fit_samples, prior_directions)
        if len(prior_directions) == component_count
        else _initial_ray_gmm_centroids(fit_samples, component_count)
    )
    covariance_ray_centroids = prior_directions if len(prior_directions) == component_count else None
    covariances = _initial_gmm_covariances(
        fit_samples,
        centroids,
        regularization=regularization,
        ray_variance_floor=ray_variance_floor,
        constrain_covariance_to_ray=constrain_covariance_to_ray,
        perpendicular_variance_floor=perpendicular_variance_floor,
        perpendicular_variance_ceiling=perpendicular_variance_ceiling,
        covariance_ray_centroids=covariance_ray_centroids,
    )
    previous_log_likelihood = -np.inf
    responsibilities = np.full(
        (len(fit_samples), component_count),
        1.0 / float(component_count),
        dtype=np.float32,
    )
    for _ in range(max(1, int(max_iterations))):
        log_probabilities, _precisions = _gmm_log_probabilities(
            fit_samples,
            centroids,
            covariances,
            regularization=regularization,
        )
        responsibilities = _balanced_gmm_responsibilities(
            log_probabilities,
            assignment_balance=assignment_balance,
            balance_iterations=balance_iterations,
        )
        log_likelihood = float(np.sum(_logsumexp(log_probabilities, axis=1)))
        if (
            np.isfinite(previous_log_likelihood)
            and abs(log_likelihood - previous_log_likelihood) <= float(tolerance) * len(fit_samples)
        ):
            break
        previous_log_likelihood = log_likelihood

        effective_counts = np.maximum(responsibilities.sum(axis=0), 1e-6)
        centroids = (responsibilities.T @ fit_samples) / effective_counts[:, None]
        centroids = _apply_gmm_prior_to_centroids(
            centroids,
            prior_directions,
            prior_strength=prior_strength,
        )
        covariances = _responsibility_weighted_covariances(
            fit_samples,
            centroids,
            responsibilities,
            effective_counts,
            regularization=regularization,
            ray_variance_floor=ray_variance_floor,
            constrain_covariance_to_ray=constrain_covariance_to_ray,
            perpendicular_variance_floor=perpendicular_variance_floor,
            perpendicular_variance_ceiling=perpendicular_variance_ceiling,
            covariance_ray_centroids=covariance_ray_centroids,
        )

    final_log_probabilities, _precisions = _gmm_log_probabilities(
        usable,
        centroids,
        covariances,
        regularization=regularization,
    )
    responsibilities = _balanced_gmm_responsibilities(
        final_log_probabilities,
        assignment_balance=assignment_balance,
        balance_iterations=balance_iterations,
    )
    distances = _mahalanobis_distance_matrix(usable, centroids, covariances)
    labels = responsibilities.argmax(axis=1).astype(np.int32)
    labels[distances[np.arange(len(distances)), labels] > max_mahalanobis_distance] = -1
    unbalanced_responsibilities = _gmm_unbalanced_responsibilities(final_log_probabilities)
    valid_labels = labels >= 0
    selected_confidence = np.zeros(len(labels), dtype=np.float32)
    selected_confidence[valid_labels] = unbalanced_responsibilities[
        np.flatnonzero(valid_labels),
        labels[valid_labels],
    ]
    labels[selected_confidence < float(np.clip(min_assignment_confidence, 0.0, 1.0))] = -1
    if dense_core_outliers_unknown:
        dense_core_mask = np.zeros(len(usable), dtype=bool)
        dense_core_mask[fit_local_indexes] = True
        labels[~dense_core_mask] = -1
    sample_cluster_indexes[usable_indexes] = labels

    retained_centroids = []
    retained_covariances = []
    cluster_samples = []
    remapped_indexes = np.full(len(samples), -1, dtype=np.int32)
    for cluster_index in range(component_count):
        member_indexes = usable_indexes[labels == cluster_index]
        if len(member_indexes) == 0:
            continue
        members = samples[member_indexes]
        retained_centroids.append(
            centroids[cluster_index]
            if len(prior_directions) == component_count
            else members.mean(axis=0)
        )
        retained_covariances.append(covariances[cluster_index])
        cluster_samples.append(members)
        remapped_indexes[member_indexes] = len(retained_centroids) - 1

    if not retained_centroids:
        empty_cov = np.empty((0, 3, 3), dtype=np.float32)
        return (
            sample_cluster_indexes,
            np.empty((0, 3), dtype=np.float32),
            [],
            empty_cov,
            empty_cov,
            dense_core_sample_mask,
        )

    centroid_array = np.asarray(retained_centroids, dtype=np.float32).reshape(-1, 3)
    covariance_array = np.asarray(retained_covariances, dtype=np.float32).reshape(-1, 3, 3)
    precision_array = np.asarray(
        [_stable_inverse(covariance, regularization=regularization) for covariance in covariance_array],
        dtype=np.float32,
    )
    return (
        remapped_indexes,
        centroid_array,
        cluster_samples,
        covariance_array,
        precision_array,
        dense_core_sample_mask,
    )


def mahalanobis_distances_to_rgb_clusters(
    rgb_sample: np.ndarray,
    centroids: np.ndarray,
    precisions: np.ndarray,
) -> np.ndarray:
    point = np.asarray(rgb_sample, dtype=np.float32).reshape(3)
    means = np.asarray(centroids, dtype=np.float32).reshape(-1, 3)
    precision_matrices = np.asarray(precisions, dtype=np.float32).reshape(-1, 3, 3)
    if len(means) == 0:
        return np.empty(0, dtype=np.float32)
    deltas = point[None, :] - means
    distances = np.einsum("ki,kij,kj->k", deltas, precision_matrices, deltas)
    return np.maximum(distances, 0.0).astype(np.float32)


def balanced_ray_rgb_cluster_memberships(
    rgb_samples: np.ndarray,
    *,
    target_clusters: int,
    min_brightness: float,
    max_brightness: float | None = None,
    max_angle_radians: float = RAY_MAX_ANGLE_RADIANS,
    assignment_balance: float = GMM_ASSIGNMENT_BALANCE,
    assignment_sharpness: float = RAY_ASSIGNMENT_SHARPNESS,
    balance_iterations: int = RAY_BALANCE_ITERATIONS,
    max_iterations: int = RAY_MAX_ITERATIONS,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    samples = np.asarray(rgb_samples, dtype=np.float32).reshape(-1, 3)
    sample_cluster_indexes = np.full(len(samples), -1, dtype=np.int32)
    if len(samples) == 0 or target_clusters <= 0:
        return (
            sample_cluster_indexes,
            np.empty((0, 3), dtype=np.float32),
            [],
            np.empty((0, 3), dtype=np.float32),
        )

    usable_indexes = np.flatnonzero(
        rgb_brightness_mask(
            samples,
            min_brightness=min_brightness,
            max_brightness=max_brightness,
        )
    )
    usable = samples[usable_indexes]
    component_count = min(int(target_clusters), len(usable))
    if component_count <= 0:
        return (
            sample_cluster_indexes,
            np.empty((0, 3), dtype=np.float32),
            [],
            np.empty((0, 3), dtype=np.float32),
        )

    directions = _normalize_rgb_vectors(usable)
    ray_directions = _initial_ray_directions(usable, component_count)
    responsibilities = np.full(
        (len(usable), component_count),
        1.0 / float(component_count),
        dtype=np.float32,
    )
    for _ in range(max(1, int(max_iterations))):
        cosine_scores = np.clip(directions @ ray_directions.T, -1.0, 1.0)
        responsibilities = _balanced_gmm_responsibilities(
            cosine_scores * float(assignment_sharpness),
            assignment_balance=assignment_balance,
            balance_iterations=balance_iterations,
        )
        next_ray_directions = _weighted_ray_directions(directions, responsibilities)
        if np.allclose(next_ray_directions, ray_directions, atol=1e-4):
            ray_directions = next_ray_directions
            break
        ray_directions = next_ray_directions

    cosine_scores = np.clip(directions @ ray_directions.T, -1.0, 1.0)
    responsibilities = _balanced_gmm_responsibilities(
        cosine_scores * float(assignment_sharpness),
        assignment_balance=assignment_balance,
        balance_iterations=balance_iterations,
    )
    labels = responsibilities.argmax(axis=1).astype(np.int32)
    angles = np.arccos(cosine_scores[np.arange(len(cosine_scores)), labels])
    labels[angles > float(max_angle_radians)] = -1
    sample_cluster_indexes[usable_indexes] = labels

    retained_centroids = []
    retained_directions = []
    cluster_samples = []
    remapped_indexes = np.full(len(samples), -1, dtype=np.int32)
    for cluster_index in range(component_count):
        member_indexes = usable_indexes[labels == cluster_index]
        if len(member_indexes) == 0:
            continue
        members = samples[member_indexes]
        retained_centroids.append(members.mean(axis=0))
        retained_directions.append(ray_directions[cluster_index])
        cluster_samples.append(members)
        remapped_indexes[member_indexes] = len(retained_centroids) - 1

    if not retained_centroids:
        return (
            sample_cluster_indexes,
            np.empty((0, 3), dtype=np.float32),
            [],
            np.empty((0, 3), dtype=np.float32),
        )
    return (
        remapped_indexes,
        np.asarray(retained_centroids, dtype=np.float32).reshape(-1, 3),
        cluster_samples,
        np.asarray(retained_directions, dtype=np.float32).reshape(-1, 3),
    )


def angles_to_rgb_rays(rgb_sample: np.ndarray, ray_directions: np.ndarray) -> np.ndarray:
    point = np.asarray(rgb_sample, dtype=np.float32).reshape(1, 3)
    rays = np.asarray(ray_directions, dtype=np.float32).reshape(-1, 3)
    if len(rays) == 0:
        return np.empty(0, dtype=np.float32)
    direction = _normalize_rgb_vectors(point)[0]
    cosine = np.clip(rays @ direction, -1.0, 1.0)
    return np.arccos(cosine).astype(np.float32)


def _normalize_rgb_vectors(samples: np.ndarray) -> np.ndarray:
    points = np.asarray(samples, dtype=np.float32).reshape(-1, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / np.maximum(norms, 1e-6)


def _initial_ray_directions(samples: np.ndarray, component_count: int) -> np.ndarray:
    centroids = _initial_ray_gmm_centroids(samples, component_count)
    return _normalize_rgb_vectors(centroids)


def _weighted_ray_directions(
    directions: np.ndarray,
    responsibilities: np.ndarray,
) -> np.ndarray:
    weighted = np.asarray(responsibilities, dtype=np.float32).T @ np.asarray(
        directions,
        dtype=np.float32,
    )
    norms = np.linalg.norm(weighted, axis=1, keepdims=True)
    fallback_indexes = np.flatnonzero(norms[:, 0] <= 1e-6)
    if len(fallback_indexes) > 0:
        for index in fallback_indexes:
            weighted[index] = directions[index % len(directions)]
    return weighted / np.maximum(np.linalg.norm(weighted, axis=1, keepdims=True), 1e-6)


def _balanced_gmm_responsibilities(
    log_probabilities: np.ndarray,
    *,
    assignment_balance: float,
    balance_iterations: int,
) -> np.ndarray:
    log_values = np.asarray(log_probabilities, dtype=np.float32)
    if log_values.size == 0:
        return np.empty(log_values.shape, dtype=np.float32)

    log_responsibilities = log_values - _logsumexp(log_values, axis=1)[:, None]
    responsibilities = np.exp(log_responsibilities).astype(np.float32)
    strength = float(np.clip(assignment_balance, 0.0, 1.0))
    if strength <= 0.0 or responsibilities.shape[1] <= 1:
        return responsibilities

    balanced = responsibilities.copy()
    target_column_sum = float(len(balanced)) / float(balanced.shape[1])
    for _ in range(max(1, int(balance_iterations))):
        column_sums = np.maximum(balanced.sum(axis=0, keepdims=True), 1e-12)
        balanced *= target_column_sum / column_sums
        row_sums = np.maximum(balanced.sum(axis=1, keepdims=True), 1e-12)
        balanced /= row_sums

    if strength >= 1.0:
        return balanced.astype(np.float32)
    mixed = responsibilities * (1.0 - strength) + balanced * strength
    mixed /= np.maximum(mixed.sum(axis=1, keepdims=True), 1e-12)
    return mixed.astype(np.float32)


def _gmm_unbalanced_responsibilities(log_probabilities: np.ndarray) -> np.ndarray:
    log_values = np.asarray(log_probabilities, dtype=np.float32)
    if log_values.size == 0:
        return np.empty(log_values.shape, dtype=np.float32)
    log_responsibilities = log_values - _logsumexp(log_values, axis=1)[:, None]
    return np.exp(log_responsibilities).astype(np.float32)


def _initial_ray_gmm_centroids(samples: np.ndarray, component_count: int) -> np.ndarray:
    points = np.asarray(samples, dtype=np.float32).reshape(-1, 3)
    if component_count <= 0:
        return np.empty((0, 3), dtype=np.float32)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    directions = points / np.maximum(norms, 1e-6)
    centroids = [directions[int(np.argmax(norms[:, 0]))]]
    while len(centroids) < component_count:
        existing = np.asarray(centroids, dtype=np.float32)
        angular_distances = 1.0 - np.max(directions @ existing.T, axis=1)
        centroids.append(directions[int(np.argmax(angular_distances))])

    direction_centroids = np.asarray(centroids, dtype=np.float32)
    for _ in range(12):
        labels = np.argmax(directions @ direction_centroids.T, axis=1)
        for index in range(component_count):
            members = directions[labels == index]
            if len(members) == 0:
                continue
            centroid = members.mean(axis=0)
            norm = float(np.linalg.norm(centroid))
            if norm > 1e-6:
                direction_centroids[index] = centroid / norm

    labels = np.argmax(directions @ direction_centroids.T, axis=1)
    raw_centroids = []
    for index in range(component_count):
        members = points[labels == index]
        if len(members) == 0:
            nearest = int(np.argmax(directions @ direction_centroids[index]))
            raw_centroids.append(points[nearest])
        else:
            raw_centroids.append(members.mean(axis=0))
    return np.asarray(raw_centroids, dtype=np.float32).reshape(-1, 3)


def _initial_gmm_covariances(
    samples: np.ndarray,
    centroids: np.ndarray,
    *,
    regularization: float,
    ray_variance_floor: float,
    constrain_covariance_to_ray: bool,
    perpendicular_variance_floor: float,
    perpendicular_variance_ceiling: float,
    covariance_ray_centroids: np.ndarray | None = None,
) -> np.ndarray:
    points = np.asarray(samples, dtype=np.float32).reshape(-1, 3)
    means = np.asarray(centroids, dtype=np.float32).reshape(-1, 3)
    if len(means) == 0:
        return np.empty((0, 3, 3), dtype=np.float32)
    distances = np.linalg.norm(points[:, None, :] - means[None, :, :], axis=2)
    labels = distances.argmin(axis=1)
    covariances = []
    fallback = _regularized_covariance(points, regularization=regularization)
    ray_centroids = (
        np.asarray(covariance_ray_centroids, dtype=np.float32).reshape(-1, 3)
        if covariance_ray_centroids is not None
        else means
    )
    for index in range(len(means)):
        members = points[labels == index]
        ray_centroid = ray_centroids[index] if index < len(ray_centroids) else centroids[index]
        covariances.append(
            _regularized_covariance(
                members,
                ray_centroid,
                regularization=regularization,
                ray_variance_floor=ray_variance_floor,
                constrain_covariance_to_ray=constrain_covariance_to_ray,
                perpendicular_variance_floor=perpendicular_variance_floor,
                perpendicular_variance_ceiling=perpendicular_variance_ceiling,
            )
            if len(members) >= 2
            else _shape_gmm_covariance_for_ray(
                fallback,
                ray_centroid,
                ray_variance_floor=ray_variance_floor,
                constrain_covariance_to_ray=constrain_covariance_to_ray,
                perpendicular_variance_floor=perpendicular_variance_floor,
                perpendicular_variance_ceiling=perpendicular_variance_ceiling,
            )
        )
    return np.asarray(covariances, dtype=np.float32).reshape(-1, 3, 3)


def _responsibility_weighted_covariances(
    samples: np.ndarray,
    centroids: np.ndarray,
    responsibilities: np.ndarray,
    effective_counts: np.ndarray,
    *,
    regularization: float,
    ray_variance_floor: float,
    constrain_covariance_to_ray: bool,
    perpendicular_variance_floor: float,
    perpendicular_variance_ceiling: float,
    covariance_ray_centroids: np.ndarray | None = None,
) -> np.ndarray:
    covariances = []
    means = np.asarray(centroids, dtype=np.float32).reshape(-1, 3)
    ray_centroids = (
        np.asarray(covariance_ray_centroids, dtype=np.float32).reshape(-1, 3)
        if covariance_ray_centroids is not None
        else means
    )
    for index, centroid in enumerate(means):
        deltas = samples - centroid[None, :]
        weighted = deltas * responsibilities[:, index][:, None]
        covariance = (weighted.T @ deltas) / max(float(effective_counts[index]), 1e-6)
        covariance += np.eye(3, dtype=np.float32) * float(regularization)
        ray_centroid = ray_centroids[index] if index < len(ray_centroids) else centroid
        covariance = _shape_gmm_covariance_for_ray(
            covariance,
            ray_centroid,
            ray_variance_floor=ray_variance_floor,
            constrain_covariance_to_ray=constrain_covariance_to_ray,
            perpendicular_variance_floor=perpendicular_variance_floor,
            perpendicular_variance_ceiling=perpendicular_variance_ceiling,
        )
        covariances.append(covariance)
    return np.asarray(covariances, dtype=np.float32).reshape(-1, 3, 3)


def _regularized_covariance(
    samples: np.ndarray,
    centroid: np.ndarray | None = None,
    *,
    regularization: float,
    ray_variance_floor: float = 0.0,
    constrain_covariance_to_ray: bool = False,
    perpendicular_variance_floor: float = GMM_PERPENDICULAR_VARIANCE_FLOOR,
    perpendicular_variance_ceiling: float = GMM_PERPENDICULAR_VARIANCE_CEILING,
) -> np.ndarray:
    points = np.asarray(samples, dtype=np.float32).reshape(-1, 3)
    if len(points) < 2:
        scale = max(float(regularization), 1e-6)
        covariance = np.eye(3, dtype=np.float32) * scale
        return _shape_gmm_covariance_for_ray(
            covariance,
            centroid,
            ray_variance_floor=ray_variance_floor,
            constrain_covariance_to_ray=constrain_covariance_to_ray,
            perpendicular_variance_floor=perpendicular_variance_floor,
            perpendicular_variance_ceiling=perpendicular_variance_ceiling,
        )
    covariance = np.cov(points, rowvar=False).astype(np.float32)
    covariance = covariance + np.eye(3, dtype=np.float32) * float(regularization)
    return _shape_gmm_covariance_for_ray(
        covariance,
        centroid,
        ray_variance_floor=ray_variance_floor,
        constrain_covariance_to_ray=constrain_covariance_to_ray,
        perpendicular_variance_floor=perpendicular_variance_floor,
        perpendicular_variance_ceiling=perpendicular_variance_ceiling,
    )


def _shape_gmm_covariance_for_ray(
    covariance: np.ndarray,
    centroid: np.ndarray | None,
    *,
    ray_variance_floor: float,
    constrain_covariance_to_ray: bool = False,
    perpendicular_variance_floor: float = GMM_PERPENDICULAR_VARIANCE_FLOOR,
    perpendicular_variance_ceiling: float = GMM_PERPENDICULAR_VARIANCE_CEILING,
) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=np.float32).reshape(3, 3).copy()
    if centroid is None:
        return matrix

    center = np.asarray(centroid, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(center))
    if norm <= 1e-6:
        return matrix

    ray = center / norm
    radial_variance = max(float(ray @ matrix @ ray), float(ray_variance_floor), 1e-6)
    if not constrain_covariance_to_ray:
        if float(ray @ matrix @ ray) >= float(ray_variance_floor):
            return matrix
        return matrix + np.outer(ray, ray).astype(np.float32) * (
            radial_variance - float(ray @ matrix @ ray)
        )

    perpendicular_basis = np.eye(3, dtype=np.float32) - np.outer(ray, ray).astype(np.float32)
    perpendicular_variance = float(np.trace(perpendicular_basis @ matrix) / 2.0)
    perpendicular_variance = max(perpendicular_variance, float(perpendicular_variance_floor))
    if perpendicular_variance_ceiling > 0.0:
        perpendicular_variance = min(perpendicular_variance, float(perpendicular_variance_ceiling))
    if perpendicular_variance <= 1e-6:
        perpendicular_variance = 1e-6
    return (
        np.outer(ray, ray).astype(np.float32) * radial_variance
        + perpendicular_basis * perpendicular_variance
    ).astype(np.float32)


def _apply_ray_variance_floor(
    covariance: np.ndarray,
    centroid: np.ndarray | None,
    *,
    ray_variance_floor: float,
) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=np.float32).reshape(3, 3).copy()
    if centroid is None or float(ray_variance_floor) <= 0.0:
        return matrix
    return _shape_gmm_covariance_for_ray(
        matrix,
        centroid,
        ray_variance_floor=ray_variance_floor,
    )


def _gmm_log_probabilities(
    samples: np.ndarray,
    centroids: np.ndarray,
    covariances: np.ndarray,
    *,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(samples, dtype=np.float32).reshape(-1, 3)
    means = np.asarray(centroids, dtype=np.float32).reshape(-1, 3)
    precision_matrices = []
    log_probabilities = []
    for mean, covariance in zip(means, np.asarray(covariances, dtype=np.float32).reshape(-1, 3, 3)):
        stable_covariance = covariance + np.eye(3, dtype=np.float32) * float(regularization) * 1e-3
        precision = _stable_inverse(stable_covariance, regularization=regularization)
        sign, log_determinant = np.linalg.slogdet(stable_covariance)
        if sign <= 0:
            stable_covariance = stable_covariance + np.eye(3, dtype=np.float32) * float(regularization)
            precision = _stable_inverse(stable_covariance, regularization=regularization)
            _sign, log_determinant = np.linalg.slogdet(stable_covariance)
        deltas = points - mean[None, :]
        squared = np.einsum("ni,ij,nj->n", deltas, precision, deltas)
        log_probabilities.append(-0.5 * (squared + float(log_determinant)))
        precision_matrices.append(precision)
    return (
        np.asarray(log_probabilities, dtype=np.float32).T,
        np.asarray(precision_matrices, dtype=np.float32).reshape(-1, 3, 3),
    )


def _mahalanobis_distance_matrix(
    samples: np.ndarray,
    centroids: np.ndarray,
    covariances: np.ndarray,
) -> np.ndarray:
    _log_probabilities, precisions = _gmm_log_probabilities(
        samples,
        centroids,
        covariances,
        regularization=0.0,
    )
    points = np.asarray(samples, dtype=np.float32).reshape(-1, 3)
    means = np.asarray(centroids, dtype=np.float32).reshape(-1, 3)
    distances = []
    for mean, precision in zip(means, precisions):
        deltas = points - mean[None, :]
        distances.append(np.einsum("ni,ij,nj->n", deltas, precision, deltas))
    return np.maximum(np.asarray(distances, dtype=np.float32).T, 0.0)


def _stable_inverse(matrix: np.ndarray, *, regularization: float) -> np.ndarray:
    covariance = np.asarray(matrix, dtype=np.float32).reshape(3, 3)
    try:
        return np.linalg.inv(covariance).astype(np.float32)
    except np.linalg.LinAlgError:
        stabilized = covariance + np.eye(3, dtype=np.float32) * max(float(regularization), 1e-6)
        return np.linalg.pinv(stabilized).astype(np.float32)


def _logsumexp(values: np.ndarray, *, axis: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    max_values = np.max(array, axis=axis, keepdims=True)
    return (
        np.log(np.sum(np.exp(array - max_values), axis=axis, keepdims=True))
        + max_values
    ).squeeze(axis)


def density_rgb_clusters(
    rgb_samples: np.ndarray,
    *,
    target_clusters: int,
    min_brightness: float,
    max_brightness: float | None = None,
    eps: float = DENSITY_CLUSTER_EPS,
    eps_reference_rgb: float = DENSITY_CLUSTER_EPS_REFERENCE_RGB,
    min_points: int = DENSITY_CLUSTER_MIN_POINTS,
    merge_ray_distance_divisor: float = CLUSTER_MERGE_RAY_DISTANCE_DIVISOR,
) -> tuple[np.ndarray, list[np.ndarray]]:
    _, centroids, cluster_samples = density_rgb_cluster_memberships(
        rgb_samples,
        target_clusters=target_clusters,
        min_brightness=min_brightness,
        max_brightness=max_brightness,
        eps=eps,
        eps_reference_rgb=eps_reference_rgb,
        min_points=min_points,
        merge_ray_distance_divisor=merge_ray_distance_divisor,
    )
    return centroids, cluster_samples


def density_rgb_cluster_memberships(
    rgb_samples: np.ndarray,
    *,
    target_clusters: int,
    min_brightness: float,
    max_brightness: float | None = None,
    eps: float = DENSITY_CLUSTER_EPS,
    eps_reference_rgb: float = DENSITY_CLUSTER_EPS_REFERENCE_RGB,
    min_points: int = DENSITY_CLUSTER_MIN_POINTS,
    merge_ray_distance_divisor: float = CLUSTER_MERGE_RAY_DISTANCE_DIVISOR,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    samples = np.asarray(rgb_samples, dtype=np.float32).reshape(-1, 3)
    sample_cluster_indexes = np.full(len(samples), -1, dtype=np.int32)
    if len(samples) == 0 or target_clusters <= 0:
        return sample_cluster_indexes, np.empty((0, 3), dtype=np.float32), []

    usable_indexes = np.flatnonzero(
        rgb_brightness_mask(
            samples,
            min_brightness=min_brightness,
            max_brightness=max_brightness,
        )
    )
    usable = samples[usable_indexes]
    if len(usable) < min_points:
        return sample_cluster_indexes, np.empty((0, 3), dtype=np.float32), []

    labels = density_bin_labels(
        usable,
        eps=eps,
        eps_reference_rgb=eps_reference_rgb,
        min_points=min_points,
    )

    clusters = []
    for label in sorted(set(int(value) for value in labels if value >= 0)):
        member_indexes = usable_indexes[labels == label]
        if len(member_indexes) >= min_points:
            clusters.append(member_indexes.astype(np.int32, copy=True))
    clusters = merge_smallest_index_clusters_to_nearest_neighbor(
        clusters,
        samples=samples,
        target_count=target_clusters,
        ray_distance_divisor=merge_ray_distance_divisor,
    )

    centroids = []
    retained_clusters = []
    for cluster_index, member_indexes in enumerate(clusters):
        members = samples[member_indexes]
        centroid = members.mean(axis=0)
        if float(np.linalg.norm(centroid)) <= 1e-6:
            continue
        centroids.append(centroid)
        retained_clusters.append(members)
        sample_cluster_indexes[member_indexes] = cluster_index

    if len(centroids) == 0:
        return sample_cluster_indexes, np.empty((0, 3), dtype=np.float32), retained_clusters
    return (
        sample_cluster_indexes,
        np.asarray(centroids, dtype=np.float32).reshape(-1, 3),
        retained_clusters,
    )


def density_hsv_clusters(
    bgr_samples: np.ndarray,
    *,
    target_clusters: int,
    min_brightness: float,
    max_brightness: float | None = None,
    eps: float = HSV_DENSITY_CLUSTER_EPS,
    min_points: int = HSV_DENSITY_CLUSTER_MIN_POINTS,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    _labels, centroids, rgb_cluster_samples, hsv_cluster_samples = density_hsv_cluster_memberships(
        bgr_samples,
        target_clusters=target_clusters,
        min_brightness=min_brightness,
        max_brightness=max_brightness,
        eps=eps,
        min_points=min_points,
    )
    return centroids, rgb_cluster_samples, hsv_cluster_samples


def density_hsv_cluster_memberships(
    bgr_samples: np.ndarray,
    *,
    target_clusters: int,
    min_brightness: float,
    max_brightness: float | None = None,
    eps: float = HSV_DENSITY_CLUSTER_EPS,
    min_points: int = HSV_DENSITY_CLUSTER_MIN_POINTS,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    bgr = np.asarray(bgr_samples, dtype=np.uint8).reshape(-1, 3)
    rgb = bgr_samples_to_rgb(bgr)
    hsv = bgr_samples_to_hsv(bgr)
    sample_cluster_indexes = np.full(len(hsv), -1, dtype=np.int32)
    if len(hsv) == 0 or target_clusters <= 0:
        return sample_cluster_indexes, np.empty((0, 3), dtype=np.float32), [], []

    usable_indexes = np.flatnonzero(
        hsv_value_mask(
            hsv,
            min_brightness=min_brightness,
            max_brightness=max_brightness,
        )
    )
    usable_hsv = hsv[usable_indexes]
    if len(usable_hsv) < min_points:
        return sample_cluster_indexes, np.empty((0, 3), dtype=np.float32), [], []

    labels = density_fixed_hsv_labels(
        usable_hsv,
        eps=eps,
        min_points=min_points,
    )
    clusters = []
    for label in sorted(set(int(value) for value in labels if value >= 0)):
        member_indexes = usable_indexes[labels == label]
        if len(member_indexes) >= min_points:
            clusters.append(member_indexes.astype(np.int32, copy=True))
    clusters.sort(key=len, reverse=True)
    clusters = clusters[: max(0, int(target_clusters))]

    centroids = []
    rgb_cluster_samples = []
    hsv_cluster_samples = []
    for cluster_index, member_indexes in enumerate(clusters):
        rgb_members = rgb[member_indexes]
        hsv_members = hsv[member_indexes]
        if len(rgb_members) == 0:
            continue
        centroids.append(rgb_members.mean(axis=0))
        rgb_cluster_samples.append(rgb_members)
        hsv_cluster_samples.append(hsv_members)
        sample_cluster_indexes[member_indexes] = cluster_index

    if len(centroids) == 0:
        return sample_cluster_indexes, np.empty((0, 3), dtype=np.float32), [], []
    return (
        sample_cluster_indexes,
        np.asarray(centroids, dtype=np.float32).reshape(-1, 3),
        rgb_cluster_samples,
        hsv_cluster_samples,
    )


def merge_smallest_clusters_to_nearest_neighbor(
    clusters: list[np.ndarray],
    *,
    target_count: int,
    ray_distance_divisor: float = CLUSTER_MERGE_RAY_DISTANCE_DIVISOR,
) -> list[np.ndarray]:
    clusters = [np.asarray(cluster, dtype=np.float32).reshape(-1, 3) for cluster in clusters]
    clusters = [cluster for cluster in clusters if len(cluster) > 0]
    if target_count <= 0:
        return []
    clusters.sort(key=len, reverse=True)
    if len(clusters) <= target_count:
        return clusters

    anchors = clusters[:target_count]
    extras = clusters[target_count:]
    pending_merges: list[list[np.ndarray]] = [[] for _ in anchors]
    for extra in extras:
        nearest_index = nearest_anchor_cluster_index(
            anchors,
            extra,
            ray_distance_divisor=ray_distance_divisor,
        )
        pending_merges[nearest_index].append(extra)

    merged_clusters = []
    for anchor, assigned_extras in zip(anchors, pending_merges):
        if assigned_extras:
            merged_clusters.append(np.concatenate([anchor, *assigned_extras], axis=0))
        else:
            merged_clusters.append(anchor)
    merged_clusters.sort(key=len, reverse=True)
    return merged_clusters


def merge_smallest_index_clusters_to_nearest_neighbor(
    clusters: list[np.ndarray],
    *,
    samples: np.ndarray,
    target_count: int,
    ray_distance_divisor: float = CLUSTER_MERGE_RAY_DISTANCE_DIVISOR,
) -> list[np.ndarray]:
    clusters = [np.asarray(cluster, dtype=np.int32).reshape(-1) for cluster in clusters]
    clusters = [cluster for cluster in clusters if len(cluster) > 0]
    sample_points = np.asarray(samples, dtype=np.float32).reshape(-1, 3)
    if target_count <= 0:
        return []
    clusters.sort(key=len, reverse=True)
    if len(clusters) <= target_count:
        return clusters

    anchors = clusters[:target_count]
    extras = clusters[target_count:]
    pending_merges: list[list[np.ndarray]] = [[] for _ in anchors]
    for extra in extras:
        nearest_index = nearest_anchor_index_cluster_index(
            anchors,
            extra,
            samples=sample_points,
            ray_distance_divisor=ray_distance_divisor,
        )
        pending_merges[nearest_index].append(extra)

    merged_clusters = []
    for anchor, assigned_extras in zip(anchors, pending_merges):
        if assigned_extras:
            merged_clusters.append(np.concatenate([anchor, *assigned_extras], axis=0))
        else:
            merged_clusters.append(anchor)
    merged_clusters.sort(key=len, reverse=True)
    return merged_clusters


def nearest_cluster_index(
    clusters: list[np.ndarray],
    source_index: int,
    *,
    ray_distance_divisor: float = CLUSTER_MERGE_RAY_DISTANCE_DIVISOR,
) -> int:
    source_centroid = np.asarray(clusters[source_index], dtype=np.float32).mean(axis=0)
    best_index = 0 if source_index != 0 else 1
    best_score = np.inf
    for index, cluster in enumerate(clusters):
        if index == source_index:
            continue
        score = cluster_merge_score_from_origin_ray(
            np.asarray(cluster, dtype=np.float32),
            source_centroid,
            ray_distance_divisor=ray_distance_divisor,
        )
        if score < best_score:
            best_score = score
            best_index = index
    return best_index


def nearest_anchor_cluster_index(
    anchors: list[np.ndarray],
    source_cluster: np.ndarray,
    *,
    ray_distance_divisor: float = CLUSTER_MERGE_RAY_DISTANCE_DIVISOR,
) -> int:
    source_centroid = np.asarray(source_cluster, dtype=np.float32).reshape(-1, 3).mean(axis=0)
    best_index = 0
    best_score = np.inf
    for index, anchor in enumerate(anchors):
        score = cluster_merge_score_from_origin_ray(
            np.asarray(anchor, dtype=np.float32),
            source_centroid,
            ray_distance_divisor=ray_distance_divisor,
        )
        if score < best_score:
            best_score = score
            best_index = index
    return best_index


def nearest_index_cluster_index(
    clusters: list[np.ndarray],
    source_index: int,
    *,
    samples: np.ndarray,
    ray_distance_divisor: float = CLUSTER_MERGE_RAY_DISTANCE_DIVISOR,
) -> int:
    sample_points = np.asarray(samples, dtype=np.float32).reshape(-1, 3)
    source_points = sample_points[np.asarray(clusters[source_index], dtype=np.int32)]
    source_centroid = source_points.mean(axis=0)
    best_index = 0 if source_index != 0 else 1
    best_score = np.inf
    for index, cluster in enumerate(clusters):
        if index == source_index:
            continue
        target_points = sample_points[np.asarray(cluster, dtype=np.int32)]
        score = cluster_merge_score_from_origin_ray(
            target_points,
            source_centroid,
            ray_distance_divisor=ray_distance_divisor,
        )
        if score < best_score:
            best_score = score
            best_index = index
    return best_index


def nearest_anchor_index_cluster_index(
    anchors: list[np.ndarray],
    source_cluster: np.ndarray,
    *,
    samples: np.ndarray,
    ray_distance_divisor: float = CLUSTER_MERGE_RAY_DISTANCE_DIVISOR,
) -> int:
    sample_points = np.asarray(samples, dtype=np.float32).reshape(-1, 3)
    source_points = sample_points[np.asarray(source_cluster, dtype=np.int32)]
    source_centroid = source_points.mean(axis=0)
    best_index = 0
    best_score = np.inf
    for index, anchor in enumerate(anchors):
        target_points = sample_points[np.asarray(anchor, dtype=np.int32)]
        score = cluster_merge_score_from_origin_ray(
            target_points,
            source_centroid,
            ray_distance_divisor=ray_distance_divisor,
        )
        if score < best_score:
            best_score = score
            best_index = index
    return best_index


def cluster_merge_score_from_origin_ray(
    target_cluster: np.ndarray,
    source_centroid: np.ndarray,
    *,
    ray_distance_divisor: float = CLUSTER_MERGE_RAY_DISTANCE_DIVISOR,
) -> float:
    target_points = np.asarray(target_cluster, dtype=np.float32).reshape(-1, 3)
    source = np.asarray(source_centroid, dtype=np.float32).reshape(3)
    if len(target_points) == 0:
        return np.inf

    source_norm = float(np.linalg.norm(source))
    if source_norm <= 1e-6:
        distances = np.linalg.norm(target_points - source[None, :], axis=1)
        return float(distances.min())

    ray_direction = source / source_norm
    deltas = target_points - source[None, :]
    parallel = deltas @ ray_direction
    parallel_vectors = parallel[:, None] * ray_direction[None, :]
    perpendicular_vectors = deltas - parallel_vectors
    perpendicular_distances = np.linalg.norm(perpendicular_vectors, axis=1)
    divisor = max(float(ray_distance_divisor), 1e-6)
    weighted_distances = np.hypot(perpendicular_distances, parallel / divisor)
    return float(weighted_distances.min())


def density_bin_labels(
    rgb_samples: np.ndarray,
    *,
    eps: float,
    eps_reference_rgb: float = DENSITY_CLUSTER_EPS_REFERENCE_RGB,
    min_points: int,
) -> np.ndarray:
    samples = np.asarray(rgb_samples, dtype=np.float32).reshape(-1, 3)
    labels = np.full(len(samples), -1, dtype=np.int32)
    if len(samples) == 0:
        return labels

    reference_radius = max(float(np.linalg.norm([eps_reference_rgb] * 3)), 1e-6)
    sample_radii = np.linalg.norm(samples, axis=1)
    origin_eps = DENSITY_CLUSTER_EPS_ORIGIN
    reference_eps = max(float(eps), 1e-6)
    sample_eps = origin_eps + (reference_eps - origin_eps) * sample_radii / reference_radius
    sample_eps = np.maximum(sample_eps, 1e-6)
    cell_size = max(float(sample_eps.max()), 1e-6)
    sample_cell_keys = np.floor(samples / cell_size).astype(np.int32)
    cell_lookup: dict[tuple[int, int, int], list[int]] = {}
    for sample_index, cell_key in enumerate(sample_cell_keys):
        cell_lookup.setdefault(tuple(int(value) for value in cell_key), []).append(sample_index)

    dense_samples = np.zeros(len(samples), dtype=bool)
    neighbor_indexes_by_sample: list[list[int]] = []
    for sample_index, cell_key in enumerate(sample_cell_keys):
        base_cell = tuple(int(value) for value in cell_key)
        candidate_indexes = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    candidate_indexes.extend(
                        cell_lookup.get(
                            (base_cell[0] + dx, base_cell[1] + dy, base_cell[2] + dz),
                            [],
                        )
                    )
        candidate_indexes_array = np.asarray(candidate_indexes, dtype=np.int32)
        distances = np.linalg.norm(samples[candidate_indexes_array] - samples[sample_index], axis=1)
        density_neighbors = candidate_indexes_array[distances <= sample_eps[sample_index]]
        symmetric_eps = np.maximum(sample_eps[sample_index], sample_eps[candidate_indexes_array])
        connected_neighbors = candidate_indexes_array[distances <= symmetric_eps]
        neighbor_indexes = [int(index) for index in connected_neighbors]
        neighbor_indexes_by_sample.append(neighbor_indexes)
        dense_samples[sample_index] = len(density_neighbors) >= int(min_points)

    cluster_id = 0
    for start_sample in np.flatnonzero(dense_samples):
        if labels[start_sample] >= 0:
            continue
        queue = [int(start_sample)]
        labels[start_sample] = cluster_id
        while queue:
            current = queue.pop()
            for neighbor_index in neighbor_indexes_by_sample[current]:
                if not dense_samples[neighbor_index] or labels[neighbor_index] >= 0:
                    continue
                labels[neighbor_index] = cluster_id
                queue.append(neighbor_index)
        cluster_id += 1

    return labels


def density_fixed_hsv_labels(
    hsv_samples: np.ndarray,
    *,
    eps: float,
    min_points: int,
) -> np.ndarray:
    samples = np.asarray(hsv_samples, dtype=np.float32).reshape(-1, 3)
    labels = np.full(len(samples), -1, dtype=np.int32)
    if len(samples) == 0:
        return labels

    cell_size = max(float(eps), 1e-6)
    hue_cell_count = max(1, int(np.ceil(180.0 / cell_size)))
    sample_cell_keys = np.floor(samples / cell_size).astype(np.int32)
    sample_cell_keys[:, 0] %= hue_cell_count
    cell_lookup: dict[tuple[int, int, int], list[int]] = {}
    for sample_index, cell_key in enumerate(sample_cell_keys):
        cell_lookup.setdefault(tuple(int(value) for value in cell_key), []).append(sample_index)

    dense_samples = np.zeros(len(samples), dtype=bool)
    neighbor_indexes_by_sample: list[list[int]] = []
    for sample_index, cell_key in enumerate(sample_cell_keys):
        base_cell = tuple(int(value) for value in cell_key)
        candidate_indexes = []
        for dh in (-1, 0, 1):
            for ds in (-1, 0, 1):
                for dv in (-1, 0, 1):
                    candidate_indexes.extend(
                        cell_lookup.get(
                            (
                                (base_cell[0] + dh) % hue_cell_count,
                                base_cell[1] + ds,
                                base_cell[2] + dv,
                            ),
                            [],
                        )
                    )
        candidate_indexes_array = np.asarray(candidate_indexes, dtype=np.int32)
        distances = hsv_distance_matrix(
            samples[sample_index].reshape(1, 3),
            samples[candidate_indexes_array],
        ).reshape(-1)
        neighbors = candidate_indexes_array[distances <= cell_size]
        neighbor_indexes = [int(index) for index in neighbors]
        neighbor_indexes_by_sample.append(neighbor_indexes)
        dense_samples[sample_index] = len(neighbor_indexes) >= int(min_points)

    cluster_id = 0
    for start_sample in np.flatnonzero(dense_samples):
        if labels[start_sample] >= 0:
            continue
        queue = [int(start_sample)]
        labels[start_sample] = cluster_id
        while queue:
            current = queue.pop()
            for neighbor_index in neighbor_indexes_by_sample[current]:
                if not dense_samples[neighbor_index] or labels[neighbor_index] >= 0:
                    continue
                labels[neighbor_index] = cluster_id
                queue.append(neighbor_index)
        cluster_id += 1

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
    axis_records = []
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
        axis_records.append((left_index, left_edge_index, right_index, right_edge_index, vector, length))

    if len(axis_records) != MAX_FACE_COUNT:
        return quads

    basis_vectors = _orthographic_projected_cube_basis_vectors(
        [record[4] for record in axis_records],
    )
    if basis_vectors is None:
        return quads

    for record, basis_vector in zip(axis_records, basis_vectors):
        left_index, left_edge_index, right_index, right_edge_index, _vector, _length = record
        axis_by_face_edge[left_index][left_edge_index] = basis_vector
        axis_by_face_edge[right_index][right_edge_index] = basis_vector

    observed_axis_vectors = [record[4] for record in axis_records]
    scale = _projected_cube_anchor_scale(basis_vectors, observed_axis_vectors)
    return _projected_cube_quads_for_scale(
        edge_indexes_by_face,
        axis_by_face_edge,
        common_corner_indexes,
        shared_center,
        scale,
        fallback_quads=quads,
    )


def _projected_cube_anchor_scale(
    basis_vectors: list[np.ndarray],
    observed_axis_vectors: list[np.ndarray],
) -> float:
    numerator = 0.0
    denominator = 0.0
    for basis_vector, observed_vector in zip(basis_vectors, observed_axis_vectors):
        basis_vector = np.asarray(basis_vector, dtype=np.float32).reshape(2)
        observed_vector = np.asarray(observed_vector, dtype=np.float32).reshape(2)
        numerator += float(np.dot(basis_vector, observed_vector))
        denominator += float(np.dot(basis_vector, basis_vector))
    if denominator <= 1e-6:
        return 1.0
    return max(1.0, numerator / denominator)


def _orthographic_projected_cube_basis_vectors(axis_vectors: list[np.ndarray]) -> list[np.ndarray] | None:
    unit_vectors = []
    angles = []
    for vector in axis_vectors:
        vector = np.asarray(vector, dtype=np.float32).reshape(2)
        length = float(np.linalg.norm(vector))
        if length <= 1e-6:
            return None
        unit_vector = vector / length
        unit_vectors.append(unit_vector)
        angles.append(float(np.arctan2(unit_vector[1], unit_vector[0])))

    constraints = np.vstack(
        [
            np.cos(np.asarray(angles, dtype=np.float32) * 2.0),
            np.sin(np.asarray(angles, dtype=np.float32) * 2.0),
        ],
    )
    _u, _s, vh = np.linalg.svd(constraints)
    squared_lengths = vh[-1].astype(np.float32)
    if np.all(squared_lengths < 0):
        squared_lengths = -squared_lengths
    if np.any(squared_lengths <= 1e-5):
        return None

    lengths = np.sqrt(squared_lengths)
    median_length = float(np.median(lengths))
    if median_length <= 1e-6:
        return None
    lengths = lengths / median_length
    return [
        np.asarray(unit_vector * length, dtype=np.float32)
        for unit_vector, length in zip(unit_vectors, lengths)
    ]


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
    return float(max(face_scores) * 20.0 + sum(face_scores) * 0.02)


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
    full_coverage_bonus = 5000.0 if missed == 0 else 0.0
    missed_penalty = missed_ratio * 5000.0 + missed
    return float(full_coverage_bonus + coverage * 1000.0 - spillover_ratio * 700.0 - missed_penalty)


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
    del class_name, segment_index
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
    color_classifier: ColorClassifier | None,
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
    color_classifier: ColorClassifier,
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


def parse_gmm_color_priors(value: str) -> np.ndarray:
    text = value.strip()
    if not text:
        return np.empty((0, 3), dtype=np.float32)

    priors = []
    for item in text.replace("|", ";").split(";"):
        item = item.strip()
        if not item:
            continue
        _name, separator, rgb_text = item.partition("=")
        if not separator:
            rgb_text = item
        parts = [part.strip() for part in rgb_text.split(",")]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                "Expected RGB priors as r,g,b triples separated by semicolons"
            )
        try:
            rgb = [float(part) for part in parts]
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "Expected RGB prior values to be numbers"
            ) from exc
        if any(value < 0.0 or value > 255.0 for value in rgb):
            raise argparse.ArgumentTypeError("Expected RGB prior values in the 0..255 range")
        priors.append(rgb)

    if not priors:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(priors, dtype=np.float32).reshape(-1, 3)


def normalized_rgb_html_plot_path(rgb_html_path: Path) -> Path:
    suffix = rgb_html_path.suffix or ".html"
    return rgb_html_path.with_name(f"{rgb_html_path.stem}_normalized{suffix}")


def color_features_enabled(args) -> bool:
    return int(getattr(args, "color_plot_frames", 0)) > 0


def color_plot_outputs_requested(args) -> bool:
    return color_features_enabled(args) and bool(
        args.rgb_color_plot
        or args.rgb_html_plot
        or args.hsv_html_plot
        or args.oklab_html_plot
        or getattr(args, "color_samples_file", None)
    )


def capture_fps_or_default(capture, default: float = 30.0) -> float:
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        return default
    return fps


def create_video_writer(output_path: Path, frame_shape: tuple[int, ...], fps: float):
    height, width = frame_shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise SystemExit(f"Could not open video writer: {output_path}")
    return writer


def run_two_pass_color_classification_video(
    capture,
    source,
    args,
    bbox_detector,
    sticker_model,
    color_classifier,
    color_plot_requested,
    cube_net_state,
):
    """Collect color samples, rewind a file source, then render classifications."""
    if isinstance(source, int):
        raise SystemExit("--color-classifier-mode two-pass requires a seekable video file")

    frame_limit = 1 if args.once else max(1, int(args.color_plot_frames))
    color_plot_bgr_batches = []
    color_plot_group_batches = []
    color_plot_detected_frames = 0
    roi_tracker = RoiTracker()

    # Pass one deliberately does not display or write frames.  The classifier
    # observes samples inside process_frame and fits when the requested sample
    # frame count has been reached.
    sample_frame_count = 0
    while sample_frame_count < frame_limit:
        ok, frame = capture.read()
        if not ok:
            break
        frame_result = process_frame(
            frame,
            bbox_detector,
            sticker_model,
            args,
            color_classifier,
            None,
            roi_tracker,
            return_sample_groups=color_plot_requested,
        )
        if color_plot_requested:
            _output, bgr_samples, sample_groups = frame_result
            if len(bgr_samples) > 0:
                color_plot_bgr_batches.append(bgr_samples)
                group_offset = (
                    max(
                        (int(groups.max()) for groups in color_plot_group_batches if len(groups)),
                        default=-1,
                    )
                    + 1
                )
                color_plot_group_batches.append(sample_groups + group_offset)
                color_plot_detected_frames += 1
        if args.display:
            collection_output = frame_result[0]
            cv2.putText(
                collection_output,
                f"Collecting color samples {sample_frame_count + 1}/{frame_limit}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                3,
            )
            cv2.putText(
                collection_output,
                f"Collecting color samples {sample_frame_count + 1}/{frame_limit}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (20, 20, 20),
                1,
            )
            cv2.imshow("cube sticker segmentations", collection_output)
            if cv2.waitKey(0 if args.once else 1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                capture.release()
                raise SystemExit("Color sample collection cancelled")
        sample_frame_count += 1

    if not color_classifier.ready:
        raise SystemExit(
            "Could not fit color classifier during the first video pass; "
            "collect more detected color samples or reduce --color-plot-frames"
        )
    if not capture.set(cv2.CAP_PROP_POS_FRAMES, 0):
        raise SystemExit("Could not rewind video for the second color-classification pass")

    roi_tracker.reset()
    if cube_net_state is not None:
        cube_net_state.replace_with(CubeNetState())
    video_writer = None
    video_fps = capture_fps_or_default(capture)
    previous = perf_counter()
    display_frame_count = 0
    while display_frame_count < frame_limit:
        ok, frame = capture.read()
        if not ok:
            break
        frame_result = process_frame(
            frame,
            bbox_detector,
            sticker_model,
            args,
            color_classifier,
            cube_net_state,
            roi_tracker,
            return_sample_groups=False,
        )
        output, _ = frame_result
        now = perf_counter()
        fps = 1.0 / max(now - previous, 1e-6)
        previous = now
        if args.debug_overlay:
            cv2.putText(output, f"FPS {fps:.1f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
            cv2.putText(output, f"FPS {fps:.1f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 1)
        if args.save and args.once:
            cv2.imwrite(str(args.save), output)
        if args.save_video:
            if video_writer is None:
                video_writer = create_video_writer(args.save_video, output.shape, video_fps)
            video_writer.write(output)
        if args.display:
            cv2.imshow("cube sticker segmentations", output)
            key = cv2.waitKey(0 if args.once else 1) & 0xFF
            if key == ord("q"):
                break
        display_frame_count += 1

    if args.display:
        cv2.destroyAllWindows()
    if video_writer is not None:
        video_writer.release()
    capture.release()
    if args.save_net and cube_net_state is not None:
        args.save_net.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_net), draw_cube_net(cube_net_state))

    if color_plot_requested and color_plot_bgr_batches:
        bgr_samples = np.concatenate(color_plot_bgr_batches, axis=0)
        sample_groups = np.concatenate(color_plot_group_batches, axis=0)
        if args.color_samples_file:
            save_color_samples_file(args.color_samples_file, bgr_samples, sample_groups)
            print(f"Saved color sample cache: {args.color_samples_file.resolve()}")
        if args.rgb_color_plot:
            save_rgb_samples_plot(
                bgr_samples, str(args.rgb_color_plot),
                title=f"Sticker segmentation RGB samples across {color_plot_detected_frames} frames",
                point_size=args.average_plot_point_size,
                normalize_rgb=args.normalize_rgb_plot,
                regression_polylines=rgb_piecewise_polylines_for_classifier(color_classifier),
            )
        if args.rgb_html_plot:
            save_rgb_html_plots(
                bgr_samples, args.rgb_html_plot,
                title=f"Sticker segmentation RGB samples across {color_plot_detected_frames} frames",
                normalize_rgb=args.normalize_rgb_plot, args=args,
                color_classifier=color_classifier,
            )
        if args.hsv_html_plot:
            cluster_indexes, cluster_centroids = rgb_plot_clusters_for_samples(bgr_samples, color_classifier)
            save_hsv_samples_interactive_plot(
                bgr_samples, str(args.hsv_html_plot),
                title=f"Sticker segmentation HSV samples across {color_plot_detected_frames} frames",
                point_size=args.plotly_point_size,
                cluster_indexes=cluster_indexes, cluster_centroids=cluster_centroids,
            )
        if args.oklab_html_plot:
            cluster_indexes, cluster_centroids = rgb_plot_clusters_for_samples(bgr_samples, color_classifier)
            save_oklab_samples_interactive_plot(
                bgr_samples, str(args.oklab_html_plot),
                title=f"Sticker segmentation Oklab samples across {color_plot_detected_frames} frames",
                point_size=args.plotly_point_size,
                oklab_samples=bgr_samples_to_oklab(bgr_samples),
                cluster_indexes=cluster_indexes, cluster_centroids=cluster_centroids,
                curve_points=oklab_piecewise_curves_for_classifier(color_classifier),
                split_planes=(color_classifier.split_planes if isinstance(color_classifier, HDBSCANDensityPlaneColorClassifier) and color_classifier.ready else None),
            )
    return 0


def build_color_classifier(args) -> ColorClassifier | None:
    if not color_features_enabled(args):
        return None
    classifier_type = (
        PiecewiseLinearColorClassifier
        if args.color_classifier_mode == "piecewise-linear"
        else ClusterMembershipColorClassifier
        if args.color_classifier_mode == "cluster-membership"
        else HDBSCANDensityPlaneColorClassifier
    )
    return classifier_type(
        cluster_ambiguity_distance=args.color_cluster_ambiguity_distance,
        required_frames=args.color_plot_frames,
        cluster_count=args.color_cluster_count,
        min_brightness=args.sticker_color_min_brightness,
        max_brightness=args.sticker_color_max_brightness,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        hdbscan_min_samples=args.hdbscan_min_samples,
        hdbscan_cluster_selection_method=args.hdbscan_cluster_selection_method,
        hdbscan_cluster_selection_epsilon=args.hdbscan_cluster_selection_epsilon,
        hdbscan_alpha=args.hdbscan_alpha,
        hdbscan_metric=args.hdbscan_metric,
        hdbscan_core_dist_n_jobs=args.hdbscan_core_dist_n_jobs,
        hdbscan_leave_noise=args.hdbscan_leave_noise,
        hdbscan_hierarchy_plot=args.hdbscan_hierarchy_plot,
        plane_angle_count=args.plane_angle_count,
        plane_tilt_count=args.plane_tilt_count,
        plane_max_tilt_degrees=args.plane_max_tilt_degrees,
        plane_offset_count=args.plane_offset_count,
        plane_density_grid_size=args.plane_density_grid_size,
        plane_density_smoothing=args.plane_density_smoothing,
        plane_bandwidth=args.plane_bandwidth,
        plane_min_child_size=args.plane_min_child_size,
        plane_min_child_fraction=args.plane_min_child_fraction,
        plane_balance_penalty=args.plane_balance_penalty,
        plane_relative_valley_weight=args.plane_relative_valley_weight,
        plane_min_side_mass=args.plane_min_side_mass,
        plane_cut_mass_weight=args.plane_cut_mass_weight,
        plane_cluster_size_weight=args.plane_cluster_size_weight,
    )


def save_rgb_html_plots(
    bgr_samples: np.ndarray,
    rgb_html_path: Path,
    *,
    title: str,
    normalize_rgb: bool,
    args,
    color_classifier: ColorClassifier | None = None,
) -> None:
    cluster_indexes, cluster_centroids = rgb_plot_clusters_for_samples(
        bgr_samples,
        color_classifier,
    )
    regression_rays = (
        color_classifier.ray_directions
        if isinstance(color_classifier, (BalancedConeColorClassifier, DenseRayColorClassifier))
        and color_classifier.ready
        else None
    )
    regression_polylines = rgb_piecewise_polylines_for_classifier(color_classifier)
    save_rgb_samples_interactive_plot(
        bgr_samples,
        str(rgb_html_path),
        title=title,
        normalize_rgb=normalize_rgb,
        point_size=args.plotly_point_size,
        cluster_indexes=cluster_indexes,
        cluster_centroids=cluster_centroids,
        regression_rays=regression_rays,
        regression_polylines=regression_polylines,
    )
    print(f"Saved RGB HTML plot: {rgb_html_path.resolve()}")
    normalized_html_path = normalized_rgb_html_plot_path(rgb_html_path)
    save_rgb_samples_interactive_plot(
        bgr_samples,
        str(normalized_html_path),
        title=f"{title} normalized to brightness",
        normalize_rgb=True,
        point_size=args.plotly_point_size,
        cluster_indexes=cluster_indexes,
        cluster_centroids=cluster_centroids,
        regression_rays=regression_rays,
        regression_polylines=regression_polylines,
    )
    print(f"Saved normalized RGB HTML plot: {normalized_html_path.resolve()}")


def rgb_piecewise_polylines_for_classifier(
    color_classifier: ColorClassifier | None,
) -> list[np.ndarray] | None:
    oklab_polylines = oklab_piecewise_curves_for_classifier(color_classifier)
    if oklab_polylines is None:
        return None
    return [oklab_to_rgb(polyline) for polyline in oklab_polylines]


def oklab_piecewise_curves_for_classifier(
    color_classifier: ColorClassifier | None,
) -> list[np.ndarray] | None:
    if not isinstance(color_classifier, PiecewiseLinearColorClassifier) or not color_classifier.ready:
        return None
    return [
        np.vstack(
            [
                np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                breakpoint,
                np.zeros(3, dtype=np.float32),
            ]
        )
        for breakpoint in color_classifier.polyline_models
    ]


def rgb_plot_clusters_for_samples(
    bgr_samples: np.ndarray,
    color_classifier: ColorClassifier | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if color_classifier is None:
        return None, None
    if color_classifier.ready:
        if (
            isinstance(color_classifier, HDBSCANDensityPlaneColorClassifier)
            and len(color_classifier._fit_bgr_samples) == len(bgr_samples)
            and np.array_equal(
                color_classifier._fit_bgr_samples,
                np.asarray(bgr_samples, dtype=np.uint8).reshape(-1, 3),
            )
        ):
            # Preserve HDBSCAN noise as the explicit unknown cluster (-1).
            return color_classifier.sample_labels.copy(), color_classifier.centroids
        return (
            classifier_cluster_indexes_for_bgr_samples(color_classifier, bgr_samples),
            color_classifier.centroids,
        )
    if isinstance(color_classifier, HDBSCANDensityPlaneColorClassifier):
        # Do not run a second provisional classifier just for a preview. The
        # production fit is deliberately performed once after sample capture.
        return None, None

    rgb_samples = np.asarray(bgr_samples, dtype=np.uint8).reshape(-1, 3)[:, ::-1]
    if isinstance(color_classifier, BalancedConeColorClassifier):
        cluster_indexes, centroids, _cluster_samples, _ray_directions = (
            balanced_ray_rgb_cluster_memberships(
                rgb_samples,
                target_clusters=color_classifier.cluster_count,
                min_brightness=color_classifier.min_brightness,
                max_brightness=color_classifier.max_brightness,
                max_angle_radians=color_classifier.max_angle_radians,
                assignment_balance=color_classifier.assignment_balance,
                assignment_sharpness=color_classifier.assignment_sharpness,
                balance_iterations=color_classifier.balance_iterations,
                max_iterations=color_classifier.max_iterations,
            )
        )
    elif isinstance(color_classifier, DenseRayColorClassifier):
        cluster_indexes, centroids, _cluster_samples, _ray_directions = (
            dense_ray_rgb_cluster_memberships(
                rgb_samples,
                target_clusters=color_classifier.cluster_count,
                min_brightness=color_classifier.min_brightness,
                max_brightness=color_classifier.max_brightness,
                k_neighbors=color_classifier.k_neighbors,
                max_fit_samples=color_classifier.max_fit_samples,
                min_anchor_distance=color_classifier.min_anchor_distance,
            )
        )
    elif isinstance(color_classifier, DensityPeakColorClassifier):
        cluster_indexes, centroids, _cluster_samples = density_peak_rgb_cluster_memberships(
            rgb_samples,
            target_clusters=color_classifier.cluster_count,
            min_brightness=color_classifier.min_brightness,
            max_brightness=color_classifier.max_brightness,
            k_neighbors=color_classifier.k_neighbors,
            max_fit_samples=color_classifier.max_fit_samples,
            min_peak_distance=color_classifier.min_peak_distance,
        )
    elif isinstance(color_classifier, BalancedGMMColorClassifier):
        (
            cluster_indexes,
            centroids,
            _cluster_samples,
            _covariances,
            _precisions,
            dense_core_mask,
        ) = (
            balanced_gmm_rgb_cluster_memberships_with_core_mask(
                rgb_samples,
                target_clusters=color_classifier.cluster_count,
                min_brightness=color_classifier.min_brightness,
                max_brightness=color_classifier.max_brightness,
                max_mahalanobis_distance=color_classifier.max_mahalanobis_distance,
                regularization=color_classifier.regularization,
                ray_variance_floor=color_classifier.ray_variance_floor,
                constrain_covariance_to_ray=color_classifier.constrain_covariance_to_ray,
                perpendicular_variance_floor=color_classifier.perpendicular_variance_floor,
                perpendicular_variance_ceiling=color_classifier.perpendicular_variance_ceiling,
                dense_core_k_neighbors=color_classifier.dense_core_k_neighbors,
                dense_core_quantile=color_classifier.dense_core_quantile,
                prior_rgb_values=color_classifier.prior_rgb_values,
                prior_strength=color_classifier.prior_strength,
                min_assignment_confidence=color_classifier.min_assignment_confidence,
                assignment_balance=color_classifier.assignment_balance,
                balance_iterations=color_classifier.balance_iterations,
                max_iterations=color_classifier.max_iterations,
                tolerance=color_classifier.tolerance,
            )
        )
        cluster_indexes = np.asarray(cluster_indexes, dtype=np.int32).copy()
        cluster_indexes[~dense_core_mask] = -1
    elif isinstance(color_classifier, HSVColorDensityClassifier):
        cluster_indexes, centroids, _rgb_cluster_samples, _hsv_cluster_samples = (
            density_hsv_cluster_memberships(
                bgr_samples,
                target_clusters=color_classifier.cluster_count,
                min_brightness=color_classifier.min_brightness,
                max_brightness=color_classifier.max_brightness,
                eps=color_classifier.density_cluster_eps,
                min_points=color_classifier.density_cluster_min_neighbors,
            )
        )
    elif isinstance(color_classifier, OklabGMMColorClassifier):
        (
            cluster_indexes,
            centroids,
            _cluster_samples,
            _angular_models,
            _weights,
        ) = oklab_angular_cluster_memberships(
                compress_oklab_l(
                    bgr_samples_to_oklab(bgr_samples),
                    color_classifier.l_compression_factor,
                ),
                original_oklab_samples=bgr_samples_to_oklab(bgr_samples),
                target_clusters=color_classifier.cluster_count,
                min_brightness=(
                    color_classifier.min_brightness
                    / color_classifier.l_compression_factor
                ),
                max_brightness=(
                    color_classifier.max_brightness
                    / color_classifier.l_compression_factor
                    if color_classifier.max_brightness is not None
                    else None
                ),
        )
    else:
        cluster_indexes, centroids, _cluster_samples = density_rgb_cluster_memberships(
            rgb_samples,
            target_clusters=color_classifier.cluster_count,
            min_brightness=color_classifier.min_brightness,
            max_brightness=color_classifier.max_brightness,
            eps=color_classifier.density_cluster_eps,
            eps_reference_rgb=color_classifier.density_cluster_eps_reference_rgb,
            min_points=color_classifier.density_cluster_min_neighbors,
            merge_ray_distance_divisor=color_classifier.cluster_merge_ray_distance_divisor,
        )
    return cluster_indexes, centroids


def classifier_cluster_indexes_for_bgr_samples(
    color_classifier: ColorClassifier,
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


def concatenate_samples_with_groups(
    bgr_batches: list[np.ndarray],
    group_batches: list[np.ndarray],
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not bgr_batches:
        return np.empty((0, 3), dtype=np.uint8), np.empty((0,), dtype=np.int64)
    bgr_samples = np.concatenate(bgr_batches, axis=0)
    groups = np.concatenate(group_batches, axis=0).astype(np.int64, copy=False)
    if len(bgr_samples) > max_samples:
        indexes = np.linspace(0, len(bgr_samples) - 1, max_samples, dtype=np.int32)
        bgr_samples = bgr_samples[indexes]
        groups = groups[indexes]
    return bgr_samples, groups


def save_color_samples_file(
    output_path: Path,
    bgr_samples: np.ndarray,
    sticker_ids: np.ndarray,
) -> str:
    """Save cached samples and their aligned sticker identities."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        bgr_samples=np.asarray(bgr_samples, dtype=np.uint8).reshape(-1, 3),
        sticker_ids=np.asarray(sticker_ids, dtype=np.int64).reshape(-1),
        edge_pairs=_consecutive_same_group_edges(sticker_ids),
        format_version=np.asarray([2], dtype=np.int32),
    )
    return str(output_path)


def _consecutive_same_group_edges(group_ids: np.ndarray) -> np.ndarray:
    """Create the legacy sample-order graph when no spatial graph is available."""
    groups = np.asarray(group_ids).reshape(-1)
    edges = []
    for group_id in np.unique(groups):
        indexes = np.flatnonzero(groups == group_id)
        if len(indexes) > 1:
            edges.extend(zip(indexes[:-1], indexes[1:]))
    return np.asarray(edges, dtype=np.int64).reshape(-1, 2)


def draw_sticker_segments(
    frame: np.ndarray,
    roi: np.ndarray,
    sticker_model,
    roi_xyxy,
    args,
    color_classifier: ColorClassifier | None = None,
    cube_net_state: CubeNetState | None = None,
    return_sample_groups: bool = False,
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
    bgr_group_batches = []
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
            bgr_group_batches.append(
                np.full(len(bgr_samples), len(bgr_group_batches), dtype=np.int64)
            )
        if color_classifier is not None:
            classifier_sample_batches = [
                sticker_bgr
                for _, sticker_bgr in raw_stickers
                if len(sticker_bgr) > 0
            ]
            classifier_bgr_samples = concatenate_samples(
                classifier_sample_batches,
                args.color_plot_samples_per_frame,
            )
            if len(classifier_bgr_samples) > 0:
                classifier_bgr_batches.append(classifier_bgr_samples)
        count += 1

    side_panel_masks = bool(getattr(args, "show_masks_side_panel", False))
    projection_panel = None
    if side_panel_masks:
        frame[:] = original_frame
        projection_panel = overlay.copy()
    elif getattr(args, "show_original_underlay", False):
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
            graph_frame = frame
            graph_roi_xyxy = roi_xyxy
            if side_panel_masks and projection_panel is not None:
                graph_frame = projection_panel
            graph_frame[:] = draw_sticker_graph(
                graph_frame,
                graph,
                roi_xyxy=graph_roi_xyxy,
                draw_masks=False,
            )
    if args.show_confidence or args.debug_overlay:
        label_frame = projection_panel if side_panel_masks and projection_panel is not None else frame
        for label_at, confidence in confidence_labels:
            cv2.putText(
                label_frame,
                f"{confidence:.2f}",
                label_at,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                label_frame,
                f"{confidence:.2f}",
                label_at,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
    if args.show_confidence or args.debug_overlay:
        label_frame = projection_panel if side_panel_masks and projection_panel is not None else frame
        draw_sticker_color_numbers(label_frame, sticker_color_numbers)
    bgr_samples = concatenate_samples(
        bgr_batches,
        args.color_plot_samples_per_frame,
    )
    if return_sample_groups:
        bgr_samples, sample_groups = concatenate_samples_with_groups(
            bgr_batches,
            bgr_group_batches,
            args.color_plot_samples_per_frame,
        )
    if color_classifier is not None:
        classifier_bgr_samples = concatenate_samples(
            classifier_bgr_batches,
            args.color_plot_samples_per_frame,
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
    if return_sample_groups:
        return count, bgr_samples, tracking_bounds_xyxy, projection_panel, sample_groups
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
    color_classifier: ColorClassifier | None = None,
    cube_net_state: CubeNetState | None = None,
    roi_tracker: RoiTracker | None = None,
    return_sample_groups: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    output = frame.copy()
    roi_xyxy, candidate, roi_source = roi_for_frame(frame, bbox_detector, args, roi_tracker)
    if roi_xyxy is None:
        if return_sample_groups:
            return output, np.empty((0, 3), dtype=np.uint8), np.empty((0,), dtype=np.int64)
        return output, np.empty((0, 3), dtype=np.uint8)

    x1, y1, x2, y2 = roi_xyxy
    roi = frame[y1:y2, x1:x2]
    segment_result = draw_sticker_segments(
        output,
        roi,
        sticker_model,
        roi_xyxy,
        args,
        color_classifier,
        cube_net_state,
        return_sample_groups=return_sample_groups,
    )
    if return_sample_groups:
        if len(segment_result) == 5:
            segment_count, bgr_samples, tracking_bounds_xyxy, projection_panel, sample_groups = segment_result
        else:
            segment_count, bgr_samples, tracking_bounds_xyxy, projection_panel = segment_result
            sample_groups = np.empty((0,), dtype=np.int64)
    else:
        segment_count, bgr_samples, tracking_bounds_xyxy, projection_panel = segment_result
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
    if return_sample_groups:
        return output, bgr_samples, sample_groups
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
        help="Output HTML path for interactive cylindrical HSV samples",
    )
    parser.add_argument(
        "--oklab-html-plot",
        type=Path,
        help="Output HTML path for interactive 3D Oklab samples",
    )
    parser.add_argument(
        "--color-samples-file",
        type=Path,
        help="Save sampled BGR colors and aligned sticker IDs to a compressed NPZ file",
    )
    parser.add_argument(
        "--oklab-l-compression-factor",
        type=float,
        default=OKLAB_L_COMPRESSION_FACTOR,
        help=(
            "Compress Oklab L by this factor in the GMM feature space; "
            "3 reduces brightness influence while plots remain in original Oklab"
        ),
    )
    parser.add_argument(
        "--oklab-slice-sticker-link-lambda",
        type=float,
        default=OKLAB_SLICE_STICKER_LINK_LAMBDA,
        help=(
            "Penalty for splitting samples from the same sticker when grouping "
            "Oklab L-slice visualization points; 0 disables sticker-link regularization"
        ),
    )
    parser.add_argument(
        "--oklab-slice-overlap-lambda",
        type=float,
        default=OKLAB_SLICE_OVERLAP_LAMBDA,
        help=(
            "Penalty for overlapping 90%% support regions between Oklab L-slice "
            "clusters; 0 disables overlap regularization"
        ),
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
        help="Do not recolor sticker masks using learned color classification",
    )
    parser.add_argument(
        "--sticker-color-min-brightness",
        type=float,
        default=40.0,
        help="Minimum raw RGB vector length used while fitting color clusters",
    )
    parser.add_argument(
        "--sticker-color-max-brightness",
        type=float,
        default=None,
        help=(
            "Maximum sample brightness before color samples are treated as unknown; "
            "uses raw RGB vector length for RGB classifiers and HSV value for HSV density"
        ),
    )
    parser.add_argument(
        "--sticker-color-max-cluster-distance",
        type=float,
        default=None,
        help=(
            "Maximum raw RGB distance from a sticker sample to the nearest final "
            "cluster surface before the sample is treated as unknown"
        ),
    )
    parser.add_argument(
        "--color-cluster-count",
        dest="color_cluster_count",
        type=int,
        default=6,
        help="Target number of color clusters to keep",
    )
    parser.add_argument(
        "--color-cluster-ambiguity-distance",
        type=float,
        default=0.01,
        help=(
            "For cluster-membership mode, leave a point unanswered when another "
            "cluster has a sample within this Oklab distance; 0 disables the margin"
        ),
    )
    parser.add_argument(
        "--color-classifier-mode",
        choices=(
            "density-cluster",
            "hsv-density-cluster",
            "balanced-gmm",
            "density-peak",
            "dense-ray",
            "balanced-cone",
            "oklab-gmm",
            "hdbscan-plane",
            "piecewise-linear",
            "cluster-membership",
        ),
        default="hdbscan-plane",
        help="Color recognition algorithm to use",
    )
    parser.add_argument(
        "--color-classifier-pass-mode",
        "--color-pass-mode",
        choices=("single-pass", "two-pass"),
        default="single-pass",
        help=(
            "Video processing strategy; two-pass collects color samples, rewinds "
            "the video, then displays classifications"
        ),
    )
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=500)
    parser.add_argument("--hdbscan-min-samples", type=int, default=5)
    parser.add_argument("--hdbscan-cluster-selection-method", choices=("eom", "leaf"), default="eom")
    parser.add_argument("--hdbscan-cluster-selection-epsilon", type=float, default=0.01)
    parser.add_argument("--hdbscan-alpha", type=float, default=1.0)
    parser.add_argument("--hdbscan-metric", default="euclidean")
    parser.add_argument("--hdbscan-core-dist-n-jobs", type=int, default=1)
    parser.add_argument(
        "--hdbscan-leave-noise", action=argparse.BooleanOptionalAction, default=True,
        help="Keep HDBSCAN noise out of the fit (default); --no-hdbscan-leave-noise attaches it to the nearest cluster",
    )
    parser.add_argument("--hdbscan-hierarchy-plot", type=Path, default=None)
    parser.add_argument("--plane-angle-count", type=int, default=36)
    parser.add_argument("--plane-tilt-count", type=int, default=3)
    parser.add_argument("--plane-max-tilt-degrees", type=float, default=15.0)
    parser.add_argument("--plane-offset-count", type=int, default=25)
    parser.add_argument("--plane-density-grid-size", type=int, default=48)
    parser.add_argument("--plane-density-smoothing", type=float, default=1.5)
    parser.add_argument("--plane-bandwidth", type=float, default=0.0)
    parser.add_argument("--plane-min-child-size", type=int, default=10)
    parser.add_argument("--plane-min-child-fraction", type=float, default=0.15)
    parser.add_argument("--plane-balance-penalty", type=float, default=0.25)
    parser.add_argument("--plane-relative-valley-weight", type=float, default=1.0)
    parser.add_argument("--plane-min-side-mass", type=float, default=0.12)
    parser.add_argument(
        "--plane-cut-mass-weight", type=float, default=0.5,
        help="Blend weight for geometric cut mass; 0 uses density cost, 1 uses point-band mass",
    )
    parser.add_argument(
        "--plane-cluster-size-weight", type=float, default=0.25,
        help="Soft preference for splitting larger unresolved clusters first",
    )
    parser.add_argument(
        "--density-cluster-eps",
        "--cluster-eps",
        dest="density_cluster_eps",
        type=float,
        default=DENSITY_CLUSTER_EPS,
        help=(
            "RGB distance epsilon for density clustering at the reference brightness; "
            "effective epsilon scales by sample distance from the origin"
        ),
    )
    parser.add_argument(
        "--density-cluster-eps-reference-rgb",
        type=float,
        default=DENSITY_CLUSTER_EPS_REFERENCE_RGB,
        help=(
            "RGB channel value used as the epsilon scaling reference; default 150 "
            "means --cluster-eps is measured at RGB=(150,150,150)"
        ),
    )
    parser.add_argument(
        "--density-cluster-min-neighbors",
        type=int,
        default=DENSITY_CLUSTER_MIN_POINTS,
        help="Minimum RGB points in the epsilon neighborhood for a density core point",
    )
    parser.add_argument(
        "--hsv-density-cluster-eps",
        type=float,
        default=HSV_DENSITY_CLUSTER_EPS,
        help="Fixed OpenCV-HSV distance epsilon for HSV density clustering",
    )
    parser.add_argument(
        "--hsv-density-cluster-min-neighbors",
        type=int,
        default=HSV_DENSITY_CLUSTER_MIN_POINTS,
        help="Minimum HSV points in the fixed-epsilon neighborhood for a density core point",
    )
    parser.add_argument(
        "--cluster-merge-ray-distance-divisor",
        type=float,
        default=CLUSTER_MERGE_RAY_DISTANCE_DIVISOR,
        help=(
            "Divisor applied to RGB distance along the origin-to-cluster ray when "
            "merging extra density clusters"
        ),
    )
    parser.add_argument(
        "--density-peak-k-neighbors",
        type=int,
        default=DENSITY_PEAK_K_NEIGHBORS,
        help="Neighbor count used to estimate local RGB density for density-peak clustering",
    )
    parser.add_argument(
        "--density-peak-max-fit-samples",
        type=int,
        default=DENSITY_PEAK_MAX_FIT_SAMPLES,
        help="Maximum RGB samples used for density-peak fitting",
    )
    parser.add_argument(
        "--density-peak-min-peak-distance",
        type=float,
        default=DENSITY_PEAK_MIN_PEAK_DISTANCE,
        help="Minimum RGB distance between selected density peaks",
    )
    parser.add_argument(
        "--dense-ray-k-neighbors",
        type=int,
        default=DENSE_RAY_K_NEIGHBORS,
        help="Neighbor count used to estimate local RGB density for dense-ray anchors",
    )
    parser.add_argument(
        "--dense-ray-max-fit-samples",
        type=int,
        default=DENSE_RAY_MAX_FIT_SAMPLES,
        help="Maximum RGB samples used for dense-ray anchor fitting",
    )
    parser.add_argument(
        "--dense-ray-min-anchor-distance",
        type=float,
        default=DENSE_RAY_MIN_ANCHOR_DISTANCE,
        help="Minimum RGB distance between selected dense-ray anchors",
    )
    parser.add_argument(
        "--dense-ray-max-distance",
        type=float,
        default=DENSE_RAY_MAX_DISTANCE,
        help="Maximum perpendicular RGB distance to a dense ray before a sample is unknown",
    )
    parser.add_argument(
        "--cone-max-angle-radians",
        type=float,
        default=CONE_MAX_ANGLE_RADIANS,
        help="Maximum angular distance from a fitted RGB cone axis before a sample is unknown",
    )
    parser.add_argument(
        "--cone-assignment-balance",
        type=float,
        default=GMM_ASSIGNMENT_BALANCE,
        help=(
            "How strongly balanced-cone responsibilities are pushed toward equal "
            "cluster mass; 0 is vanilla angular assignment, 1 is fully balanced"
        ),
    )
    parser.add_argument(
        "--cone-assignment-sharpness",
        type=float,
        default=CONE_ASSIGNMENT_SHARPNESS,
        help="Cosine-score multiplier used when fitting balanced-cone RGB axes",
    )
    parser.add_argument(
        "--cone-balance-iterations",
        type=int,
        default=CONE_BALANCE_ITERATIONS,
        help="Sinkhorn normalization iterations used for balanced-cone assignment balancing",
    )
    parser.add_argument(
        "--cone-max-iterations",
        type=int,
        default=CONE_MAX_ITERATIONS,
        help="Maximum angular-refinement iterations for balanced-cone fitting",
    )
    parser.add_argument(
        "--gmm-max-mahalanobis-distance",
        type=float,
        default=GMM_MAX_MAHALANOBIS_DISTANCE,
        help="Maximum squared Mahalanobis distance before a balanced-GMM sample is unknown",
    )
    parser.add_argument(
        "--gmm-regularization",
        type=float,
        default=GMM_REGULARIZATION,
        help="Diagonal covariance regularization for balanced full-covariance GMM",
    )
    parser.add_argument(
        "--gmm-ray-variance-floor",
        type=float,
        default=GMM_RAY_VARIANCE_FLOOR,
        help=(
            "Minimum covariance variance along each balanced-GMM component's "
            "origin-to-centroid RGB ray; 0 disables the ray-shape constraint"
        ),
    )
    parser.add_argument(
        "--gmm-constrain-covariance-to-ray",
        action="store_true",
        default=GMM_CONSTRAIN_COVARIANCE_TO_RAY,
        help=(
            "Force each balanced-GMM covariance to align with its origin-to-centroid "
            "ray with no tilted cross-covariance"
        ),
    )
    parser.add_argument(
        "--gmm-perpendicular-variance-floor",
        type=float,
        default=GMM_PERPENDICULAR_VARIANCE_FLOOR,
        help="Minimum variance perpendicular to each constrained GMM ray",
    )
    parser.add_argument(
        "--gmm-perpendicular-variance-ceiling",
        type=float,
        default=GMM_PERPENDICULAR_VARIANCE_CEILING,
        help=(
            "Maximum variance perpendicular to each constrained GMM ray; "
            "0 leaves the measured perpendicular variance uncapped"
        ),
    )
    parser.add_argument(
        "--gmm-dense-core-k-neighbors",
        type=int,
        default=GMM_DENSE_CORE_K_NEIGHBORS,
        help=(
            "When >0, estimate local RGB density with this RGB bin size and fit "
            "balanced-GMM only on dense core samples; larger values smooth more"
        ),
    )
    parser.add_argument(
        "--gmm-dense-core-quantile",
        type=float,
        default=GMM_DENSE_CORE_QUANTILE,
        help=(
            "Lowest density quantile to drop before fitting balanced-GMM; "
            "0 disables dense-core prefiltering"
        ),
    )
    parser.add_argument(
        "--gmm-color-priors",
        type=parse_gmm_color_priors,
        default=parse_gmm_color_priors(GMM_COLOR_PRIORS),
        help=(
            "Semicolon-separated RGB priors for balanced-GMM components, for example "
            "'white=255,255,255;yellow=255,255,0;red=255,0,0;orange=255,120,0;blue=0,0,255;green=0,255,0'"
        ),
    )
    parser.add_argument(
        "--gmm-prior-strength",
        type=float,
        default=GMM_PRIOR_STRENGTH,
        help=(
            "How strongly provided --gmm-color-priors lock component directions; "
            "0 only initializes from priors, 1 keeps means on prior rays"
        ),
    )
    parser.add_argument(
        "--gmm-min-assignment-confidence",
        type=float,
        default=GMM_MIN_ASSIGNMENT_CONFIDENCE,
        help=(
            "Minimum unbalanced GMM posterior confidence for the winning color; "
            "samples below this are treated as unknown"
        ),
    )
    parser.add_argument(
        "--gmm-assignment-balance",
        type=float,
        default=GMM_ASSIGNMENT_BALANCE,
        help=(
            "How strongly balanced-GMM responsibilities are pushed toward equal "
            "cluster mass; 0 is vanilla soft assignment, 1 is fully balanced"
        ),
    )
    parser.add_argument(
        "--gmm-balance-iterations",
        type=int,
        default=GMM_BALANCE_ITERATIONS,
        help="Sinkhorn normalization iterations used for balanced-GMM responsibility balancing",
    )
    parser.add_argument(
        "--gmm-max-iterations",
        type=int,
        default=GMM_MAX_ITERATIONS,
        help="Maximum EM iterations for balanced full-covariance GMM",
    )
    parser.add_argument(
        "--gmm-tolerance",
        type=float,
        default=GMM_TOLERANCE,
        help="Per-sample log-likelihood tolerance used to stop balanced-GMM EM",
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
        "--save-video",
        type=Path,
        help="Output video path for processed frames",
    )
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
    parser.add_argument(
        "--show-masks-side-panel",
        action="store_true",
        help="Append a black side panel showing sticker masks instead of drawing them over the cube",
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
    color_classifier = build_color_classifier(args)
    color_plot_requested = color_plot_outputs_requested(args)
    color_plot_bgr_batches = []
    color_plot_group_batches = []
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
        image_result = process_frame(
            frame,
            bbox_detector,
            sticker_model,
            args,
            color_classifier,
            cube_net_state,
            roi_tracker,
            return_sample_groups=color_plot_requested,
        )
        if color_plot_requested:
            output, image_samples, image_groups = image_result
            if args.color_samples_file:
                save_color_samples_file(args.color_samples_file, image_samples, image_groups)
                print(f"Saved color sample cache: {args.color_samples_file.resolve()}")
        else:
            output, _ = image_result
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

    if args.color_classifier_pass_mode == "two-pass":
        if args.color_plot_frames <= 0:
            raise SystemExit("--color-plot-frames must be positive for two-pass classification")
        return run_two_pass_color_classification_video(
            capture,
            source,
            args,
            bbox_detector,
            sticker_model,
            color_classifier,
            color_plot_requested,
            cube_net_state,
        )

    video_writer = None
    video_fps = capture_fps_or_default(capture)
    previous = perf_counter()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_result = process_frame(
            frame,
            bbox_detector,
            sticker_model,
            args,
            color_classifier,
            cube_net_state,
            roi_tracker,
            return_sample_groups=color_plot_requested,
        )
        if color_plot_requested:
            output, bgr_samples, sample_groups = frame_result
        else:
            output, bgr_samples = frame_result
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
        if args.save_video:
            if video_writer is None:
                video_writer = create_video_writer(args.save_video, output.shape, video_fps)
            video_writer.write(output)
        if color_plot_requested:
            if color_plot_detected_frames < args.color_plot_frames and len(bgr_samples) > 0:
                color_plot_bgr_batches.append(bgr_samples)
                group_offset = (
                    max(
                        (int(groups.max()) for groups in color_plot_group_batches if len(groups)),
                        default=-1,
                    )
                    + 1
                )
                color_plot_group_batches.append(sample_groups + group_offset)
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
    if video_writer is not None:
        video_writer.release()
    capture.release()
    if args.save_net and cube_net_state is not None:
        args.save_net.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_net), draw_cube_net(cube_net_state))
    if color_plot_requested:
        if not color_plot_bgr_batches:
            raise SystemExit("Could not collect color plot samples from any segmented frames")
        bgr_samples = np.concatenate(color_plot_bgr_batches, axis=0)
        sample_groups = np.concatenate(color_plot_group_batches, axis=0)
        if args.color_samples_file:
            save_color_samples_file(args.color_samples_file, bgr_samples, sample_groups)
            print(f"Saved color sample cache: {args.color_samples_file.resolve()}")
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
            cluster_indexes, cluster_centroids = rgb_plot_clusters_for_samples(
                bgr_samples,
                color_classifier,
            )
            save_hsv_samples_interactive_plot(
                bgr_samples,
                str(args.hsv_html_plot),
                title=f"Sticker segmentation HSV samples across {color_plot_detected_frames} frames",
                point_size=args.plotly_point_size,
                cluster_indexes=cluster_indexes,
                cluster_centroids=cluster_centroids,
            )
        if args.oklab_html_plot:
            cluster_indexes, cluster_centroids = rgb_plot_clusters_for_samples(
                bgr_samples,
                color_classifier,
            )
            save_oklab_samples_interactive_plot(
                bgr_samples,
                str(args.oklab_html_plot),
                title=f"Sticker segmentation Oklab samples across {color_plot_detected_frames} frames",
                point_size=args.plotly_point_size,
                oklab_samples=bgr_samples_to_oklab(bgr_samples),
                cluster_indexes=cluster_indexes,
                cluster_centroids=cluster_centroids,
                curve_points=(
                    oklab_curve_points_for_models(color_classifier.polyline_models)
                    if isinstance(color_classifier, OklabGMMColorClassifier)
                    and color_classifier.ready
                    else oklab_piecewise_curves_for_classifier(color_classifier)
                ),
                split_planes=(
                    color_classifier.split_planes
                    if isinstance(color_classifier, HDBSCANDensityPlaneColorClassifier)
                    and color_classifier.ready
                    else None
                ),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
