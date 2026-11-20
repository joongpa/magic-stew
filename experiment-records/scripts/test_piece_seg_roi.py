"""Test YOLO cube pose estimation on a cube bounding-box ROI.

Pipeline:
1. Detect the whole cube with the existing bbox detector.
2. Crop a padded square ROI around that bbox.
3. Run the independent one-point pose model on the ROI.
4. Draw predicted keypoints color-coded by class and the fitted 55mm cube.
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from collections import deque
from dataclasses import dataclass
from itertools import combinations, permutations, product
from pathlib import Path
from time import perf_counter

import __main__
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from live_recognition.detectors import YOLOCubeDetector, default_yolo_device  # noqa: E402
from live_recognition.independent_pose_thresholds import (  # noqa: E402
    DEFAULT_INDEPENDENT_POSE_CLASS_CONFIDENCES,
    independent_pose_class_threshold, parse_independent_pose_class_confidences)
from live_recognition.pose_motion import (  # noqa: E402
    PoseMotionFilter,
    PoseMotionFilterConfig,
)
from live_recognition.visualization import _reliable_color_mask  # noqa: E402
from live_recognition.visualization import (  # noqa: E402
    save_hsv_samples_interactive_plot, save_rgb_samples_interactive_plot,
    save_rgb_samples_plot)

DEFAULT_BBOX_MODEL = ROOT / "runs/500ExamplesWeights.pt"
DEFAULT_STICKER_MODEL = (
    ROOT
    / "runs/segment/runs/cube_face_seg_local/runs/cube_face_sticker_yolo11n_seg/weights/best.pt"
)
DEFAULT_POSE_MODEL = (
    "runs/pose/runs/independent_keypoints_v3_weights.pt"
)
ENABLE_STICKER_SEGMENTATION = False
ENABLE_POSE_ESTIMATION = True
ENABLE_POSE_GRAPH_CORRESPONDENCE = True
DRAW_POSE_KEYPOINTS = True
DRAW_PNP_CUBE = True
DRAW_CORNER_ADJACENCY = False
DRAW_POSE_GRAPH_COORDS = False
DISPLAY_WINDOW_NAME = "cube sticker segmentations"
DISPLAY_PREVIEW_BY_DEFAULT = True
SHOW_ORIGINAL_UNDERLAY_BY_DEFAULT = True
POSE_IMAGE_SIZE = 256
YOLO_DEVICE = default_yolo_device()
POSE_OBJECT_CONFIDENCE = 0.5
POSE_KEYPOINT_CONFIDENCE = 0.25
ADJACENCY_REFERENCE_ROI_SIZE_PX = 600.0
PNP_FOCAL_SCALE_MIN = 0.7
PNP_FOCAL_SCALE_MAX = 1.8
PNP_FOCAL_SCALE_STEPS = 5
PNP_CAMERA_BOOTSTRAP_FRAME_COUNT = 12
PNP_MAX_REPROJECTION_ERROR_PX = 10
PNP_REFINEMENT_REPROJECTION_ERROR_PX = 24
PNP_MAX_REFINEMENT_ATTEMPTS = 16
PNP_INLIER_THRESHOLD_PX = 10.0
PNP_MAX_OBSERVED_OUTLIER_FRACTION = 0.10
PNP_PRUNE_REPROJECTION_ERROR_PX = 18.0
POSE_ANCHOR_RESIDUAL_THRESHOLD_PX = 12.0
POSE_ANCHOR_MIN_POINTS = 3
POSE_MOTION_FILTER_BY_DEFAULT = False
POSE_MOTION_DT_SECONDS = 1.0 / 30.0
POSE_MOTION_MAX_ACCELERATION_UNITS_S2 = 6000.0
POSE_MOTION_MAX_JERK_UNITS_S3 = 120000.0
POSE_MOTION_MAX_ANGULAR_ACCELERATION_RAD_S2 = 12.0
POSE_MOTION_MAX_ANGULAR_JERK_RAD_S3 = 180.0
ADJACENCY_CANNY_BLUR_KERNEL = 1
ADJACENCY_CANNY_LOW = 30
ADJACENCY_CANNY_HIGH = 180
ADJACENCY_MIN_EDGE_LENGTH_PX = 32.0
ADJACENCY_MIN_COVERAGE = 1
ADJACENCY_MAX_UNCOVERED_GAP_RATIO = 0.03
ADJACENCY_MAX_AVERAGE_LENGTH_RATIO = 1.8
ADJACENCY_EDGE_DILATION_PX = 7
ADJACENCY_KEYPOINT_INTERSECTION_DISTANCE_PX = 5
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
CUBE_LAYER_SIZE_MM = CUBE_SIZE_MM / 3.0
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
CUBE_EDGE_MIDPOINT_POINTS_MM = np.asarray(
    [
        (CUBE_CORNER_POINTS_MM[start] + CUBE_CORNER_POINTS_MM[end]) * 0.5
        for start, end in CUBE_CORNER_EDGES
    ],
    dtype=np.float32,
)
CUBE_GRID_MAX_INDEX = 3
CUBE_LATTICE_COORDS = tuple(
    (x, y, z)
    for x in range(CUBE_GRID_MAX_INDEX + 1)
    for y in range(CUBE_GRID_MAX_INDEX + 1)
    for z in range(CUBE_GRID_MAX_INDEX + 1)
    if x in (0, CUBE_GRID_MAX_INDEX)
    or y in (0, CUBE_GRID_MAX_INDEX)
    or z in (0, CUBE_GRID_MAX_INDEX)
)
CUBE_LATTICE_POINTS_MM = np.asarray(
    [
        (
            (coord[0] - (CUBE_GRID_MAX_INDEX * 0.5)) * CUBE_LAYER_SIZE_MM,
            (coord[1] - (CUBE_GRID_MAX_INDEX * 0.5)) * CUBE_LAYER_SIZE_MM,
            (coord[2] - (CUBE_GRID_MAX_INDEX * 0.5)) * CUBE_LAYER_SIZE_MM,
        )
        for coord in CUBE_LATTICE_COORDS
    ],
    dtype=np.float32,
)
CUBE_LATTICE_IDENTITY_TRANSFORM = ((0, 1, 2), (1, 1, 1))
CUBE_LATTICE_AXIS_TRANSFORMS = tuple(
    (axis_order, signs)
    for axis_order in permutations(range(3))
    for signs in product((-1, 1), repeat=3)
)
CUBE_EDGE_KEYPOINT_OFFSET = len(CUBE_CORNER_POINTS_MM)
CUBE_KEYPOINT_EDGE_SEGMENTS = tuple(
    segment
    for edge_index, (start, end) in enumerate(CUBE_CORNER_EDGES)
    for segment in (
        (start, CUBE_EDGE_KEYPOINT_OFFSET + edge_index),
        (CUBE_EDGE_KEYPOINT_OFFSET + edge_index, end),
    )
)
POSE_CLASS_NAMES = {"corner", "corner-edge", "corner-edge-center", "center"}
POSE_KEYPOINT_COLORS = {
    "corner": (255, 255, 255),
    "corner-edge": (0, 255, 255),
    "corner-edge-center": (255, 120, 255),
    "center": (255, 255, 0),
}
PNP_CUBE_LINE_COLOR = (255, 80, 30)
PNP_CUBE_POINT_COLOR = (30, 180, 255)
PNP_CUBE_STATIC_LINE_COLOR = (100, 100, 100)
PNP_CUBE_SEAM_LINE_COLOR = (255, 220, 80)
PNP_OBSERVED_MODEL_DISTANCE_COLOR = (0, 255, 255)
PNP_OBSERVED_MODEL_DISTANCE_ALPHA = 0.35
PNP_REFINEMENT_POINT_COLOR = (255, 0, 255)
PNP_DEBUG_POSE_COLORS = (
    (255, 80, 80),
    (80, 220, 255),
    (80, 255, 120),
    (255, 120, 255),
    (255, 220, 80),
    (180, 120, 255),
)
PNP_DEBUG_POSE_ALPHA = 0.30
POSE_GRAPH_CORRESPONDENCE_LINE_COLOR = (255, 80, 255)
POSE_GRAPH_CORRESPONDENCE_TEXT_COLOR = (255, 255, 255)
POSE_GRAPH_EDGE_STRIP_COLOR = (80, 220, 255)
POSE_GRAPH_EDGE_STRIP_ALPHA = 0.22
PNP_REPROJECTION_RESIDUAL_COLOR = (0, 255, 255)


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
        if hasattr(training, name):
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
        reset_on_missing: bool = True,
    ) -> None:
        if self.roi_xyxy is None:
            return
        if tracking_bounds_xyxy is None:
            if reset_on_missing:
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
class PosePnPResult:
    error: float
    inlier_count: int
    matched_error: float
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    image_points: np.ndarray
    object_points: np.ndarray
    display_points: np.ndarray
    projected_object_points: np.ndarray
    projected_display_points: np.ndarray
    point_names: tuple[str, ...]
    edge_indexes: tuple[tuple[int, int], ...] = tuple()
    display_edge_indexes: tuple[tuple[int, int], ...] | None = None
    pose_label: str = ""
    lattice_transform: tuple[tuple[int, int, int], tuple[int, int, int]] = CUBE_LATTICE_IDENTITY_TRANSFORM
    invisible_point_count: int = 0
    observed_reprojection_error: float | None = None
    correspondence_lattice_coords: tuple[tuple[int, int, int], ...] = tuple()
    correspondence_face: tuple[int, int] | None = None
    correspondence_face_normal_z: float | None = None
    refinement_graph_index: int | None = None
    refinement_lattice_coord: tuple[int, int, int] | None = None
    debug_pose_candidates: tuple["PosePnPResult", ...] = tuple()
    raw_rvec: np.ndarray | None = None
    raw_tvec: np.ndarray | None = None
    motion_diagnostics: object | None = None
    orientation_anchor: "OrientationAnchor | None" = None


@dataclass(frozen=True)
class OrientationAnchor:
    """The cube layers whose motion currently defines the rigid pose."""

    axis: int | None
    layers: tuple[int, ...]
    point_indexes: tuple[int, ...]

    @property
    def label(self) -> str:
        if self.axis is None:
            return "all"
        return f"axis {self.axis} layers {','.join(str(layer) for layer in self.layers)}"


def refresh_pose_pnp_result_projection(result: PosePnPResult) -> None:
    """Reproject the display geometry after temporal pose filtering."""

    if any(
        getattr(result, attribute, None) is None
        for attribute in ("rvec", "tvec", "camera_matrix", "dist_coeffs")
    ):
        return
    try:
        projected_object_points, _ = cv2.projectPoints(
            np.asarray(result.object_points, dtype=np.float32).reshape(-1, 3),
            result.rvec,
            result.tvec,
            result.camera_matrix,
            result.dist_coeffs,
        )
        projected_display_points = None
        if getattr(result, "display_points", None) is not None:
            projected_display_points, _ = cv2.projectPoints(
                np.asarray(result.display_points, dtype=np.float32).reshape(-1, 3),
                result.rvec,
                result.tvec,
                result.camera_matrix,
                result.dist_coeffs,
            )
    except cv2.error:
        return
    result.projected_object_points = projected_object_points.reshape(-1, 2)
    if projected_display_points is not None:
        result.projected_display_points = projected_display_points.reshape(-1, 2)


@dataclass(frozen=True)
class CornerAdjacency:
    start_index: int
    end_index: int
    coverage: float
    max_uncovered_gap: float
    accepted: bool


@dataclass(frozen=True)
class PoseGraphPoint:
    point_index: int
    class_name: str
    xy: tuple[float, float]
    score: float


@dataclass(frozen=True)
class PoseGraphEdge:
    start_index: int
    end_index: int
    coverage: float
    max_uncovered_gap: float
    accepted: bool


@dataclass(frozen=True)
class PoseGraph:
    points: tuple[PoseGraphPoint, ...]
    edges: tuple[PoseGraphEdge, ...]


@dataclass(frozen=True)
class FramePoseStats:
    keypoint_count: int
    valid_pose: bool


@dataclass(frozen=True)
class PoseGraphCorrespondence:
    image_points: np.ndarray
    object_points: np.ndarray
    graph_indexes: tuple[int, ...]
    lattice_coords: tuple[tuple[int, int, int], ...]

    @property
    def key(self) -> tuple[tuple[int, ...], tuple[tuple[int, int, int], ...]]:
        return (tuple(self.graph_indexes), tuple(self.lattice_coords))


def physical_layer_index(lattice_value: int) -> int:
    """Map four lattice samples onto the three physical cube layers."""

    if lattice_value <= 0:
        return 0
    if lattice_value >= CUBE_GRID_MAX_INDEX:
        return 2
    return 1


def all_cube_orientation_anchor(point_count: int) -> OrientationAnchor:
    return OrientationAnchor(
        axis=None,
        layers=(0, 1, 2),
        point_indexes=tuple(range(max(0, int(point_count)))),
    )


def select_orientation_anchor(
    result: PosePnPResult,
    correspondence: PoseGraphCorrespondence,
    motion_filter: PoseMotionFilter,
) -> OrientationAnchor:
    """Choose the largest layer subset consistent with the prior rigid pose.

    The full cube wins when all observed points agree with the prior.  When a
    turn makes one layer disagree, a one- or two-layer subset can retain the
    stable part as the rigid anchor.  No turn angle or turn duration is
    constrained here.
    """

    full_anchor = all_cube_orientation_anchor(len(correspondence.lattice_coords))
    previous_pose = motion_filter.current_pose()
    if previous_pose is None or any(
        getattr(result, attribute, None) is None
        for attribute in ("object_points", "image_points", "camera_matrix", "dist_coeffs")
    ):
        return full_anchor

    prior_rvec, prior_tvec = previous_pose
    try:
        projected_points, _ = cv2.projectPoints(
            np.asarray(result.object_points, dtype=np.float32).reshape(-1, 3),
            prior_rvec,
            prior_tvec,
            result.camera_matrix,
            result.dist_coeffs,
        )
    except cv2.error:
        return full_anchor
    observed_points = np.asarray(result.image_points, dtype=np.float32).reshape(-1, 2)
    projected_points = projected_points.reshape(-1, 2)
    if len(observed_points) != len(projected_points) or len(observed_points) != len(correspondence.lattice_coords):
        return full_anchor
    residuals = np.linalg.norm(projected_points - observed_points, axis=1)
    layer_indexes_by_axis = tuple(
        tuple(
            physical_layer_index(
                transform_lattice_coord(
                    coord,
                    getattr(result, "lattice_transform", CUBE_LATTICE_IDENTITY_TRANSFORM),
                )[axis]
            )
            for coord in correspondence.lattice_coords
        )
        for axis in range(3)
    )

    candidates: list[tuple[tuple[float, ...], OrientationAnchor]] = []
    full_inliers = int(np.count_nonzero(residuals <= POSE_ANCHOR_RESIDUAL_THRESHOLD_PX))
    full_median = float(np.median(residuals)) if len(residuals) else float("inf")
    candidates.append(
        (
            (float(full_inliers), -3.0, -full_median),
            full_anchor,
        )
    )
    for axis in range(3):
        layer_indexes = layer_indexes_by_axis[axis]
        for layer_count in (1, 2):
            for layers in combinations((0, 1, 2), layer_count):
                point_indexes = tuple(
                    index
                    for index, layer_index in enumerate(layer_indexes)
                    if layer_index in layers
                )
                if len(point_indexes) < POSE_ANCHOR_MIN_POINTS:
                    continue
                layer_residuals = residuals[list(point_indexes)]
                inlier_count = int(
                    np.count_nonzero(layer_residuals <= POSE_ANCHOR_RESIDUAL_THRESHOLD_PX)
                )
                if inlier_count < POSE_ANCHOR_MIN_POINTS:
                    continue
                median_error = float(np.median(layer_residuals))
                candidates.append(
                    (
                        (float(inlier_count), float(-layer_count), -median_error),
                        OrientationAnchor(
                            axis=axis,
                            layers=tuple(layers),
                            point_indexes=point_indexes,
                        ),
                    )
                )
    return max(candidates, key=lambda item: item[0])[1]


def refit_pose_to_orientation_anchor(
    result: PosePnPResult,
    correspondence: PoseGraphCorrespondence,
    anchor: OrientationAnchor,
) -> bool:
    """Refit PnP using only points from the selected anchor layers."""

    if anchor.axis is None or len(anchor.point_indexes) < POSE_ANCHOR_MIN_POINTS:
        return anchor.axis is None
    if any(
        getattr(result, attribute, None) is None
        for attribute in ("object_points", "image_points", "camera_matrix", "dist_coeffs")
    ):
        return False
    indexes = np.asarray(anchor.point_indexes, dtype=np.int32)
    object_points = np.asarray(result.object_points, dtype=np.float32).reshape(-1, 3)[indexes]
    image_points = np.asarray(result.image_points, dtype=np.float32).reshape(-1, 2)[indexes]
    if len(object_points) < POSE_ANCHOR_MIN_POINTS:
        return False
    try:
        ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            object_points,
            image_points,
            result.camera_matrix,
            result.dist_coeffs,
            flags=cv2.SOLVEPNP_SQPNP,
        )
    except cv2.error:
        return False
    if not ok or not rvecs or not tvecs:
        return False

    best_pose = None
    best_error = float("inf")
    for rvec, tvec in zip(rvecs, tvecs, strict=True):
        candidate_rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        candidate_tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        if len(object_points) >= 4:
            try:
                candidate_rvec, candidate_tvec = cv2.solvePnPRefineLM(
                    object_points,
                    image_points,
                    result.camera_matrix,
                    result.dist_coeffs,
                    candidate_rvec,
                    candidate_tvec,
                )
            except cv2.error:
                pass
        projected, _ = cv2.projectPoints(
            object_points,
            candidate_rvec,
            candidate_tvec,
            result.camera_matrix,
            result.dist_coeffs,
        )
        error = float(
            np.mean(
                np.linalg.norm(
                    projected.reshape(-1, 2) - image_points,
                    axis=1,
                )
            )
        )
        if error < best_error:
            best_error = error
            best_pose = (candidate_rvec, candidate_tvec)
    if best_pose is None:
        return False
    result.rvec, result.tvec = best_pose
    refresh_pose_pnp_result_projection(result)
    return True


def assign_orientation_anchor(
    result: PosePnPResult,
    correspondence: PoseGraphCorrespondence,
    motion_filter: PoseMotionFilter,
) -> None:
    anchor = select_orientation_anchor(result, correspondence, motion_filter)
    if not refit_pose_to_orientation_anchor(result, correspondence, anchor):
        anchor = all_cube_orientation_anchor(len(correspondence.lattice_coords))
    result.orientation_anchor = anchor


@dataclass(frozen=True)
class PoseGraphNodeAssignment:
    coord: tuple[int, int, int]
    parent_index: int | None
    local_to_global: dict[str, tuple[int, int, int]] | None
    local_to_global_frames: tuple[dict[str, tuple[int, int, int]], ...]
    path_faces: frozenset[tuple[int, int]]
    depth: int
    has_turn: bool


@dataclass
class PoseGraphPnPSolver:
    focal_min: float = 0.7
    focal_max: float = 1.8
    focal_steps: int = 5
    camera_bootstrap_frames: int = PNP_CAMERA_BOOTSTRAP_FRAME_COUNT
    max_reprojection_error_px: float = PNP_MAX_REPROJECTION_ERROR_PX
    refinement_reprojection_error_px: float = PNP_REFINEMENT_REPROJECTION_ERROR_PX
    max_refinement_attempts: int = PNP_MAX_REFINEMENT_ATTEMPTS
    inlier_threshold_px: float = PNP_INLIER_THRESHOLD_PX
    max_observed_outlier_fraction: float = PNP_MAX_OBSERVED_OUTLIER_FRACTION
    prune_reprojection_error_px: float = PNP_PRUNE_REPROJECTION_ERROR_PX
    camera_matrix: np.ndarray | None = None
    lattice_transform: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None
    rejected_pose_result: PosePnPResult | None = None
    accepted_pose_count: int = 0
    motion_filter: PoseMotionFilter | None = None

    def solve_correspondence_candidates(
        self,
        correspondences: tuple[PoseGraphCorrespondence, ...] | list[PoseGraphCorrespondence],
        image_size: tuple[int, int],
        observed_graph: PoseGraph | None = None,
        profile: dict[str, float | int] | None = None,
    ) -> PosePnPResult | None:
        max_seed_attempts = max(0, int(self.max_refinement_attempts))
        refinement_budget = {"remaining": max(0, int(self.max_refinement_attempts))}
        self.rejected_pose_result = None
        for correspondence in tuple(correspondences)[:max_seed_attempts]:
            if profile is not None:
                profile["seed_attempts"] = int(profile.get("seed_attempts", 0)) + 1
            result = self.solve_correspondences(
                correspondence,
                image_size,
                observed_graph=observed_graph,
                profile=profile,
                refinement_budget=refinement_budget,
                clear_rejected=False,
            )
            if result is not None:
                return result
        self.clear_motion_state()
        return None

    def solve_correspondences(
        self,
        correspondence: PoseGraphCorrespondence | None,
        image_size: tuple[int, int],
        observed_graph: PoseGraph | None = None,
        profile: dict[str, float | int] | None = None,
        refinement_budget: dict[str, int] | None = None,
        clear_rejected: bool = True,
    ) -> PosePnPResult | None:
        if clear_rejected:
            self.rejected_pose_result = None
        if correspondence is None:
            return self.miss_pose_result(clear_motion=clear_rejected)

        original_correspondence = correspondence
        camera_matrix = (
            self.camera_matrix
            if self.camera_matrix is not None
            and self.accepted_pose_count >= max(0, int(self.camera_bootstrap_frames))
            else None
        )
        lattice_transforms = (
            (self.lattice_transform,)
            if camera_matrix is not None and self.lattice_transform is not None
            else CUBE_LATTICE_AXIS_TRANSFORMS
        )
        result = solve_best_lattice_graph_pnp_current(
            np.asarray(correspondence.image_points, dtype=np.float32).reshape(-1, 2),
            tuple(correspondence.lattice_coords),
            image_size,
            focal_min=self.focal_min,
            focal_max=self.focal_max,
            focal_steps=self.focal_steps,
            camera_matrix=camera_matrix,
            lattice_transforms=lattice_transforms,
            profile=profile,
        )
        if result is None:
            return self.miss_pose_result(clear_motion=clear_rejected)
        self.remember_rejected_pose_result(result, correspondence)
        backface_only_count = pose_correspondence_backface_only_point_count(
            correspondence,
            result,
        )
        if profile is not None:
            profile["pnp_backface_only_correspondence_points"] = backface_only_count
        if backface_only_count > 0:
            return self.miss_pose_result(clear_motion=clear_rejected)
        if observed_graph is None:
            observed_error, observed_inlier_count, observed_point_count = observed_pose_graph_reprojection_score(
                correspondence,
                result,
                observed_graph,
                all_correspondence=original_correspondence,
                inlier_threshold_px=self.inlier_threshold_px,
            )
        else:
            (
                observed_error,
                observed_inlier_count,
                observed_point_count,
                invisible_observed_count,
            ) = observed_pose_model_nearest_same_class_score(
                result,
                observed_graph,
                inlier_threshold_px=self.inlier_threshold_px,
            )
            if profile is not None:
                profile["pnp_observed_invisible_points"] = invisible_observed_count
            backface_fit_count, _backface_tested_count = observed_pose_model_backface_fit_count(
                result,
                observed_graph,
                inlier_threshold_px=self.inlier_threshold_px,
            )
            if profile is not None:
                profile["pnp_backface_fit_points"] = backface_fit_count
            if backface_fit_count >= 3:
                self.remember_rejected_pose_result(result, correspondence, observed_error)
                return self.miss_pose_result(clear_motion=clear_rejected)
        if observed_point_count < 3:
            return self.miss_pose_result(clear_motion=clear_rejected)
        self.remember_rejected_pose_result(result, correspondence, observed_error)
        if observed_error <= self.max_reprojection_error_px:
            refined = refine_pose_correspondence_with_visible_observed_points(
                correspondence,
                result,
                observed_graph,
                image_size,
                focal_min=self.focal_min,
                focal_max=self.focal_max,
                focal_steps=self.focal_steps,
                camera_matrix=camera_matrix,
                lattice_transforms=lattice_transforms,
                max_attempts=1,
                refinement_budget=refinement_budget,
                accept_error_px=self.max_reprojection_error_px,
                inlier_threshold_px=self.inlier_threshold_px,
                profile=profile,
                return_first_result=True,
            )
            if refined is None:
                return self.accept_pose_result(result, correspondence, observed_error)
            refined_correspondence, refined_result, refined_error = refined
            if refined_error < observed_error:
                return self.accept_pose_result(refined_result, refined_correspondence, refined_error)
            return self.accept_pose_result(result, correspondence, observed_error)
        if observed_error > self.refinement_reprojection_error_px:
            return self.miss_pose_result(clear_motion=clear_rejected)

        refined = refine_pose_correspondence_with_visible_observed_points(
            correspondence,
            result,
            observed_graph,
            image_size,
            focal_min=self.focal_min,
            focal_max=self.focal_max,
            focal_steps=self.focal_steps,
            camera_matrix=camera_matrix,
            lattice_transforms=lattice_transforms,
            max_attempts=self.max_refinement_attempts,
            refinement_budget=refinement_budget,
            accept_error_px=self.max_reprojection_error_px,
            inlier_threshold_px=self.inlier_threshold_px,
            profile=profile,
        )
        if refined is None:
            return self.miss_pose_result(clear_motion=clear_rejected)
        refined_correspondence, refined_result, refined_error = refined
        return self.accept_pose_result(refined_result, refined_correspondence, refined_error)

    def clear_motion_state(self) -> None:
        if self.motion_filter is not None:
            self.motion_filter.reset()

    def miss_pose_result(self, *, clear_motion: bool = True) -> None:
        if clear_motion:
            self.clear_motion_state()
        return None

    def remember_rejected_pose_result(
        self,
        result: PosePnPResult,
        correspondence: PoseGraphCorrespondence,
        observed_error: float | None = None,
    ) -> None:
        result.observed_reprojection_error = observed_error
        result.point_names = tuple(str(coord) for coord in correspondence.lattice_coords)
        result.pose_label = "graph-rejected"
        candidate_error = float(getattr(result, "matched_error", float("inf")))
        current_error = (
            float("inf")
            if self.rejected_pose_result is None
            else float(getattr(self.rejected_pose_result, "matched_error", float("inf")))
        )
        if candidate_error < current_error:
            self.rejected_pose_result = result

    def accept_pose_result(
        self,
        result: PosePnPResult,
        correspondence: PoseGraphCorrespondence,
        observed_error: float,
    ) -> PosePnPResult:
        self.rejected_pose_result = None
        result.observed_reprojection_error = observed_error
        result.point_names = tuple(str(coord) for coord in correspondence.lattice_coords)
        result.pose_label = "graph"
        if self.motion_filter is not None:
            assign_orientation_anchor(result, correspondence, self.motion_filter)
        else:
            result.orientation_anchor = all_cube_orientation_anchor(
                len(correspondence.lattice_coords)
            )
        if self.motion_filter is not None and all(
            getattr(result, attribute, None) is not None
            for attribute in ("rvec", "tvec", "camera_matrix", "dist_coeffs")
        ):
            raw_rvec = np.asarray(result.rvec, dtype=np.float64).reshape(3, 1).copy()
            raw_tvec = np.asarray(result.tvec, dtype=np.float64).reshape(3, 1).copy()
            filtered_rvec, filtered_tvec, diagnostics = self.motion_filter.update(
                raw_rvec,
                raw_tvec,
            )
            result.raw_rvec = raw_rvec
            result.raw_tvec = raw_tvec
            result.motion_diagnostics = diagnostics
            result.rvec = filtered_rvec
            result.tvec = filtered_tvec
            refresh_pose_pnp_result_projection(result)
            result.pose_label = "graph-motion-filtered"
        if getattr(result, "camera_matrix", None) is not None:
            self.camera_matrix = np.asarray(result.camera_matrix, dtype=np.float64).reshape(3, 3).copy()
        self.lattice_transform = getattr(result, "lattice_transform", CUBE_LATTICE_IDENTITY_TRANSFORM)
        self.accepted_pose_count += 1
        return result

    def reset_pose_state(self) -> None:
        self.camera_matrix = None
        self.lattice_transform = None
        self.rejected_pose_result = None
        self.accepted_pose_count = 0
        if self.motion_filter is not None:
            self.motion_filter.reset()


def prune_high_error_pose_graph_correspondence(
    correspondence: PoseGraphCorrespondence,
    result: PosePnPResult,
    *,
    max_error_px: float,
) -> PoseGraphCorrespondence | None:
    image_points = np.asarray(correspondence.image_points, dtype=np.float32).reshape(-1, 2)
    projected_object_points = getattr(result, "projected_object_points", None)
    if projected_object_points is None:
        return None
    projected_points = np.asarray(projected_object_points, dtype=np.float32).reshape(-1, 2)
    if len(image_points) != len(projected_points) or len(image_points) <= 3:
        return None

    reprojection_errors = np.linalg.norm(projected_points - image_points, axis=1)
    keep_mask = reprojection_errors <= float(max_error_px)
    if bool(np.all(keep_mask)):
        return None
    if int(np.count_nonzero(keep_mask)) < 3:
        return None

    return PoseGraphCorrespondence(
        image_points=image_points[keep_mask],
        object_points=np.asarray(correspondence.object_points, dtype=np.float32).reshape(-1, 3)[keep_mask],
        graph_indexes=tuple(
            graph_index
            for graph_index, keep in zip(correspondence.graph_indexes, keep_mask)
            if bool(keep)
        ),
        lattice_coords=tuple(
            coord
            for coord, keep in zip(correspondence.lattice_coords, keep_mask)
            if bool(keep)
        ),
    )


def observed_pose_graph_reprojection_error(
    correspondence: PoseGraphCorrespondence,
    result: PosePnPResult,
    observed_graph: PoseGraph | None,
    all_correspondence: PoseGraphCorrespondence | None = None,
) -> float:
    return observed_pose_graph_reprojection_score(
        correspondence,
        result,
        observed_graph,
        all_correspondence=all_correspondence,
        inlier_threshold_px=PNP_INLIER_THRESHOLD_PX,
    )[0]


def observed_pose_graph_reprojection_score(
    correspondence: PoseGraphCorrespondence,
    result: PosePnPResult,
    observed_graph: PoseGraph | None,
    *,
    all_correspondence: PoseGraphCorrespondence | None = None,
    inlier_threshold_px: float,
) -> tuple[float, int, int]:
    fallback_error = float(getattr(result, "matched_error", float("inf")))
    if observed_graph is None:
        fallback_inlier_count = int(fallback_error <= float(inlier_threshold_px)) * len(
            correspondence.image_points
        )
        return fallback_error, fallback_inlier_count, len(correspondence.image_points)
    if any(
        getattr(result, attribute, None) is None
        for attribute in ("rvec", "tvec", "camera_matrix", "dist_coeffs")
    ):
        return float("inf"), 0, len(correspondence.image_points)

    retained_indexes = set(correspondence.graph_indexes)
    source_correspondence = all_correspondence or correspondence
    graph_indexes = []
    object_points = []
    transformed_object_points = cube_lattice_points_for_coords(
        tuple(source_correspondence.lattice_coords),
        lattice_transform=getattr(result, "lattice_transform", CUBE_LATTICE_IDENTITY_TRANSFORM),
    )
    for graph_index, object_point in zip(
        source_correspondence.graph_indexes,
        transformed_object_points,
    ):
        if graph_index not in retained_indexes:
            continue
        graph_indexes.append(graph_index)
        object_points.append(object_point)
    if not object_points:
        return float("inf"), 0, 0

    try:
        projected_points, _ = cv2.projectPoints(
            np.asarray(object_points, dtype=np.float32).reshape(-1, 3),
            result.rvec,
            result.tvec,
            result.camera_matrix,
            result.dist_coeffs,
        )
    except cv2.error:
        return float("inf"), 0, len(object_points)
    projected_points = projected_points.reshape(-1, 2)

    observed_errors = []
    for graph_index, projected_point in zip(graph_indexes, projected_points):
        if graph_index < 0 or graph_index >= len(observed_graph.points):
            continue
        observed_point = np.asarray(observed_graph.points[graph_index].xy, dtype=np.float32)
        observed_errors.append(float(np.linalg.norm(projected_point - observed_point)))
    if not observed_errors:
        return float("inf"), 0, len(object_points)
    inlier_count = sum(error <= float(inlier_threshold_px) for error in observed_errors)
    return float(np.mean(observed_errors)), int(inlier_count), len(observed_errors)


def refine_pose_correspondence_with_visible_observed_points(
    correspondence: PoseGraphCorrespondence,
    result: PosePnPResult,
    observed_graph: PoseGraph | None,
    image_size: tuple[int, int],
    *,
    focal_min: float,
    focal_max: float,
    focal_steps: int,
    camera_matrix: np.ndarray | None,
    lattice_transforms: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...],
    max_attempts: int,
    refinement_budget: dict[str, int] | None,
    accept_error_px: float,
    inlier_threshold_px: float,
    profile: dict[str, float | int] | None = None,
    return_first_result: bool = False,
) -> tuple[PoseGraphCorrespondence, PosePnPResult, float] | None:
    if observed_graph is None:
        return None

    used_graph_indexes = set(correspondence.graph_indexes)
    used_lattice_coords = set(correspondence.lattice_coords)
    candidate_graph_indexes = [
        index
        for index, point in enumerate(observed_graph.points)
        if index not in used_graph_indexes
        and point.class_name in CUBE_LATTICE_COORDS_BY_CLASS
    ]
    candidate_graph_indexes.sort(
        key=lambda index: (
            -observed_point_distance_to_nearest_correspondence_point(
                observed_graph.points[index],
                correspondence,
            ),
            -observed_graph.points[index].score,
            index,
        )
    )
    remaining_attempts = max(0, int(max_attempts))
    if refinement_budget is not None:
        remaining_attempts = min(
            remaining_attempts,
            max(0, int(refinement_budget.get("remaining", 0))),
        )
    candidate_graph_indexes = candidate_graph_indexes[:remaining_attempts]
    if profile is not None:
        profile["refinement_candidates"] = len(candidate_graph_indexes)

    for graph_index in candidate_graph_indexes:
        if refinement_budget is not None:
            if refinement_budget["remaining"] <= 0:
                break
            refinement_budget["remaining"] -= 1
        observed_point = observed_graph.points[graph_index]
        nearest_coord = nearest_visible_lattice_coord_for_observed_point(
            observed_point,
            result,
            used_lattice_coords=used_lattice_coords,
        )
        if nearest_coord is None:
            continue

        augmented = append_pose_graph_correspondence(
            correspondence,
            graph_index,
            observed_point.xy,
            nearest_coord,
        )
        refined_result = solve_best_lattice_graph_pnp_current(
            np.asarray(augmented.image_points, dtype=np.float32).reshape(-1, 2),
            tuple(augmented.lattice_coords),
            image_size,
            focal_min=focal_min,
            focal_max=focal_max,
            focal_steps=focal_steps,
            camera_matrix=camera_matrix,
            lattice_transforms=lattice_transforms,
            profile=profile,
        )
        if refined_result is None:
            continue
        backface_only_count = pose_correspondence_backface_only_point_count(
            augmented,
            refined_result,
        )
        if profile is not None:
            profile["refinement_backface_only_correspondence_points"] = backface_only_count
        if backface_only_count > 0:
            continue
        refined_error, _inlier_count, refined_point_count, _invisible_count = (
            observed_pose_model_nearest_same_class_score(
                refined_result,
                observed_graph,
                inlier_threshold_px=inlier_threshold_px,
            )
        )
        backface_fit_count, _backface_tested_count = observed_pose_model_backface_fit_count(
            refined_result,
            observed_graph,
            inlier_threshold_px=inlier_threshold_px,
        )
        if profile is not None:
            profile["refinement_attempts"] = int(profile.get("refinement_attempts", 0)) + 1
            profile["refinement_backface_fit_points"] = backface_fit_count
        if backface_fit_count >= 3:
            continue
        if refined_point_count >= 3 and (
            refined_error <= float(accept_error_px) or bool(return_first_result)
        ):
            refined_result.refinement_graph_index = int(graph_index)
            refined_result.refinement_lattice_coord = nearest_coord
            return augmented, refined_result, refined_error
    return None


def observed_point_distance_to_nearest_correspondence_point(
    point: PoseGraphPoint,
    correspondence: PoseGraphCorrespondence,
) -> float:
    image_points = np.asarray(correspondence.image_points, dtype=np.float32).reshape(-1, 2)
    if image_points.size == 0:
        return 0.0
    point_xy = np.asarray(point.xy, dtype=np.float32).reshape(1, 2)
    distances = np.linalg.norm(image_points - point_xy, axis=1)
    return float(np.min(distances))


def append_pose_graph_correspondence(
    correspondence: PoseGraphCorrespondence,
    graph_index: int,
    image_xy: tuple[float, float],
    lattice_coord: tuple[int, int, int],
) -> PoseGraphCorrespondence:
    lattice_coords = (*correspondence.lattice_coords, lattice_coord)
    image_points = np.concatenate(
        [
            np.asarray(correspondence.image_points, dtype=np.float32).reshape(-1, 2),
            np.asarray(image_xy, dtype=np.float32).reshape(1, 2),
        ],
        axis=0,
    )
    return PoseGraphCorrespondence(
        image_points=image_points,
        object_points=cube_lattice_points_for_coords(lattice_coords),
        graph_indexes=(*correspondence.graph_indexes, int(graph_index)),
        lattice_coords=lattice_coords,
    )


def nearest_visible_lattice_coord_for_observed_point(
    observed_point: PoseGraphPoint,
    result: PosePnPResult,
    *,
    used_lattice_coords: set[tuple[int, int, int]],
) -> tuple[int, int, int] | None:
    entries = visible_projected_lattice_points_for_class(result, observed_point.class_name)
    if not entries:
        return None
    observed_xy = np.asarray(observed_point.xy, dtype=np.float32)
    usable_entries = [
        (coord, projected_xy)
        for coord, projected_xy in entries
        if coord not in used_lattice_coords
    ]
    if not usable_entries:
        return None
    coord, _projected = min(
        usable_entries,
        key=lambda entry: float(np.linalg.norm(np.asarray(entry[1], dtype=np.float32) - observed_xy)),
    )
    return coord


def visible_projected_lattice_points_for_class(
    result: PosePnPResult,
    class_name: str,
) -> list[tuple[tuple[int, int, int], np.ndarray]]:
    coords = CUBE_LATTICE_COORDS_BY_CLASS.get(class_name)
    if not coords:
        return []
    if any(
        getattr(result, attribute, None) is None
        for attribute in ("rvec", "tvec", "camera_matrix", "dist_coeffs")
    ):
        return []

    object_points = cube_lattice_points_for_coords(
        coords,
        lattice_transform=getattr(result, "lattice_transform", CUBE_LATTICE_IDENTITY_TRANSFORM),
    )
    visible = [
        (coord, object_point)
        for coord, object_point in zip(coords, object_points)
        if result_object_point_has_visible_surface(object_point, result)
    ]
    if not visible:
        return []
    visible_coords, visible_object_points = zip(*visible)
    try:
        projected_points, _ = cv2.projectPoints(
            np.asarray(visible_object_points, dtype=np.float32).reshape(-1, 3),
            result.rvec,
            result.tvec,
            result.camera_matrix,
            result.dist_coeffs,
        )
    except cv2.error:
        return []
    return [
        (coord, projected_xy)
        for coord, projected_xy in zip(visible_coords, projected_points.reshape(-1, 2))
    ]


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


def roi_side_px(roi_xyxy: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = roi_xyxy
    return float(max(1, max(x2 - x1, y2 - y1)))


def adjacency_roi_scale(roi_side: float, args) -> float:
    reference_side = max(
        1.0,
        float(getattr(args, "adjacency_reference_roi_size", ADJACENCY_REFERENCE_ROI_SIZE_PX)),
    )
    return max(1.0, float(roi_side)) / reference_side


def scaled_adjacency_px(
    value: float,
    roi_side: float,
    args,
    *,
    min_value: float = 0.0,
) -> float:
    return max(float(min_value), float(value) * adjacency_roi_scale(roi_side, args))


def scaled_adjacency_int_px(
    value: float,
    roi_side: float,
    args,
    *,
    min_value: int = 0,
) -> int:
    return max(int(min_value), int(round(scaled_adjacency_px(value, roi_side, args))))


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
    if class_name == "corner":
        return "corner"
    if class_name == "corner-edge":
        return "edge"
    if class_name == "corner-edge-center":
        return "edge-center"
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


def result_box_confidences(result) -> np.ndarray | None:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return None
    confidences = tensor_to_numpy(getattr(boxes, "conf", None))
    if confidences is None:
        return None
    return np.asarray(confidences, dtype=np.float32).reshape(-1)


def collect_pose_graph_points(
    result,
    pose_model,
    xy: np.ndarray | None,
    keypoint_conf: np.ndarray | None,
    *,
    confidence_threshold: float,
    class_confidences: dict[str, float],
    box_confidence: np.ndarray | None,
    offset_xy: tuple[int, int],
    scale_xy: tuple[float, float],
) -> list[PoseGraphPoint]:
    if xy is None or len(xy) == 0:
        return []

    classes = result_box_classes(result)
    box_confidences = (
        None
        if box_confidence is None
        else np.asarray(box_confidence, dtype=np.float32).reshape(-1)
    )
    points_by_detection = np.asarray(xy, dtype=np.float32)
    scores_by_detection = (
        None
        if keypoint_conf is None or len(keypoint_conf) == 0
        else np.asarray(keypoint_conf, dtype=np.float32)
    )
    offset_x, offset_y = offset_xy
    scale_x, scale_y = scale_xy
    observed: list[PoseGraphPoint] = []
    for detection_index, points in enumerate(points_by_detection):
        class_id = (
            int(classes[detection_index])
            if classes is not None and detection_index < len(classes)
            else -1
        )
        class_name = class_name_for_id(result, pose_model, class_id).removesuffix("_sticker")
        if class_name not in {"corner", "corner-edge", "corner-edge-center"}:
            continue

        detection_box_confidence = (
            1.0
            if box_confidences is None or detection_index >= len(box_confidences)
            else float(box_confidences[detection_index])
        )
        if detection_box_confidence < independent_pose_class_threshold(
            class_name,
            class_confidences,
            fallback=confidence_threshold,
        ):
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
            observed.append(
                PoseGraphPoint(
                    point_index=len(observed),
                    class_name=class_name,
                    xy=(
                        offset_x + (float(point[0]) * scale_x),
                        offset_y + (float(point[1]) * scale_y),
                    ),
                    score=keypoint_score,
                )
            )
    return observed


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


def transform_lattice_coord(
    coord: tuple[int, int, int],
    lattice_transform: tuple[tuple[int, int, int], tuple[int, int, int]],
) -> tuple[int, int, int]:
    axis_order, signs = lattice_transform
    return tuple(
        coord[axis] if sign > 0 else CUBE_GRID_MAX_INDEX - coord[axis]
        for axis, sign in zip(axis_order, signs)
    )


def cube_lattice_point_mm(coord: tuple[int, int, int]) -> tuple[float, float, float]:
    return (
        (coord[0] - (CUBE_GRID_MAX_INDEX * 0.5)) * CUBE_LAYER_SIZE_MM,
        (coord[1] - (CUBE_GRID_MAX_INDEX * 0.5)) * CUBE_LAYER_SIZE_MM,
        (coord[2] - (CUBE_GRID_MAX_INDEX * 0.5)) * CUBE_LAYER_SIZE_MM,
    )


def cube_lattice_points_for_coords(
    coords: tuple[tuple[int, int, int], ...],
    *,
    lattice_transform: tuple[tuple[int, int, int], tuple[int, int, int]] = CUBE_LATTICE_IDENTITY_TRANSFORM,
) -> np.ndarray:
    return np.asarray(
        [
            cube_lattice_point_mm(transform_lattice_coord(coord, lattice_transform))
            for coord in coords
        ],
        dtype=np.float32,
    )


def cube_lattice_display_points_for_transform(
    lattice_transform: tuple[tuple[int, int, int], tuple[int, int, int]] = CUBE_LATTICE_IDENTITY_TRANSFORM,
) -> np.ndarray:
    return cube_lattice_points_for_coords(
        CUBE_LATTICE_DISPLAY_CORNER_COORDS,
        lattice_transform=lattice_transform,
    )


def invisible_surface_point_count(
    object_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> int:
    object_points = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
    return sum(
        not object_point_has_visible_surface(point, rvec, tvec)
        for point in object_points
    )


def object_point_has_visible_surface(
    object_point: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray | None = None,
) -> bool:
    if tvec is None:
        return object_point_has_front_facing_boundary_surface(object_point, rvec)
    return not object_point_is_occluded_by_cube(object_point, rvec, tvec)


def result_object_point_has_visible_surface(
    object_point: np.ndarray,
    result: PosePnPResult,
) -> bool:
    tvec = getattr(result, "tvec", None)
    if tvec is None:
        return object_point_has_visible_surface(object_point, result.rvec)
    try:
        return object_point_has_visible_surface(object_point, result.rvec, tvec)
    except TypeError:
        return object_point_has_visible_surface(object_point, result.rvec)


def object_point_is_occluded_by_cube(
    object_point: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> bool:
    point = np.asarray(object_point, dtype=np.float64).reshape(3)
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    camera_origin = (-rotation.T @ np.asarray(tvec, dtype=np.float64).reshape(3, 1)).reshape(3)
    direction = point - camera_origin
    if float(np.linalg.norm(direction)) <= 1e-9:
        return False

    bounds_min = np.full(3, -float(CUBE_HALF_SIZE_MM), dtype=np.float64)
    bounds_max = np.full(3, float(CUBE_HALF_SIZE_MM), dtype=np.float64)
    t_enter = 0.0
    t_exit = 1.0
    for axis in range(3):
        origin_value = float(camera_origin[axis])
        direction_value = float(direction[axis])
        if abs(direction_value) <= 1e-12:
            if origin_value < bounds_min[axis] or origin_value > bounds_max[axis]:
                return False
            continue
        near = (bounds_min[axis] - origin_value) / direction_value
        far = (bounds_max[axis] - origin_value) / direction_value
        if near > far:
            near, far = far, near
        t_enter = max(t_enter, float(near))
        t_exit = min(t_exit, float(far))
        if t_enter > t_exit:
            return False

    return 1e-5 < t_enter < 1.0 - 1e-4


def object_point_has_front_facing_boundary_surface(object_point: np.ndarray, rvec: np.ndarray) -> bool:
    point = np.asarray(object_point, dtype=np.float32).reshape(3)
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    for axis in range(3):
        if abs(float(point[axis]) + CUBE_HALF_SIZE_MM) <= 1e-3:
            normal = np.zeros(3, dtype=np.float64)
            normal[axis] = -1.0
        elif abs(float(point[axis]) - CUBE_HALF_SIZE_MM) <= 1e-3:
            normal = np.zeros(3, dtype=np.float64)
            normal[axis] = 1.0
        else:
            continue
        camera_normal = rotation @ normal
        if float(camera_normal[2]) < 0.0:
            return True
    return False


def lattice_boundary_face_is_visible(
    face: tuple[int, int],
    rvec: np.ndarray,
) -> bool:
    return lattice_boundary_face_camera_normal_z(face, rvec) < 0.0


def lattice_boundary_face_camera_normal_z(
    face: tuple[int, int],
    rvec: np.ndarray,
) -> float:
    axis, value = face
    normal = np.zeros(3, dtype=np.float64)
    normal[int(axis)] = -1.0 if int(value) == 0 else 1.0
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    camera_normal = rotation @ normal
    return float(camera_normal[2])


def visible_lattice_boundary_faces_for_pose(
    rvec: np.ndarray,
) -> frozenset[tuple[int, int]]:
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    visible_faces = set()
    for axis in range(3):
        for value, sign in ((0, -1.0), (CUBE_GRID_MAX_INDEX, 1.0)):
            normal = np.zeros(3, dtype=np.float64)
            normal[axis] = sign
            camera_normal = rotation @ normal
            if float(camera_normal[2]) < 0.0:
                visible_faces.add((axis, value))
    return frozenset(visible_faces)


def pose_correspondence_backface_only_point_count(
    correspondence: PoseGraphCorrespondence,
    result: PosePnPResult,
) -> int:
    if getattr(result, "rvec", None) is None or getattr(result, "tvec", None) is None:
        return 0
    lattice_transform = getattr(result, "lattice_transform", CUBE_LATTICE_IDENTITY_TRANSFORM)
    object_points = cube_lattice_points_for_coords(
        tuple(correspondence.lattice_coords),
        lattice_transform=lattice_transform,
    )
    backface_only_count = 0
    for object_point in object_points:
        if not object_point_has_visible_surface(object_point, result.rvec, result.tvec):
            backface_only_count += 1
    return backface_only_count


def graph_camera_matrix_candidates(
    image_size: tuple[int, int],
    *,
    focal_min: float,
    focal_max: float,
    focal_steps: int,
) -> list[np.ndarray]:
    focal_steps = max(1, int(focal_steps))
    return [
        default_camera_matrix(image_size, focal_scale=float(focal_scale))
        for focal_scale in np.linspace(float(focal_min), float(focal_max), focal_steps)
    ]


def solve_best_lattice_graph_pnp_current(
    image_points: np.ndarray,
    lattice_coords: tuple[tuple[int, int, int], ...],
    image_size: tuple[int, int],
    *,
    focal_min: float,
    focal_max: float,
    focal_steps: int,
    camera_matrix: np.ndarray | None = None,
    lattice_transforms: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...] = CUBE_LATTICE_AXIS_TRANSFORMS,
    profile: dict[str, float | int] | None = None,
) -> PosePnPResult | None:
    image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    if len(image_points) != len(lattice_coords):
        return None

    best_result: PosePnPResult | None = None
    best_score: tuple[float, ...] | None = None
    camera_matrices = (
        [np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)]
        if camera_matrix is not None
        else graph_camera_matrix_candidates(
            image_size,
            focal_min=focal_min,
            focal_max=focal_max,
            focal_steps=focal_steps,
        )
    )
    lattice_transforms = tuple(lattice_transforms)
    if not lattice_transforms:
        return None
    if profile is not None:
        profile["pnp_camera_candidates"] = len(camera_matrices)
        profile["pnp_axis_transforms"] = len(lattice_transforms)
    for camera_matrix in camera_matrices:
        for lattice_transform in lattice_transforms:
            if profile is not None:
                profile["pnp_solve_calls"] = int(profile.get("pnp_solve_calls", 0)) + 1
            result = solve_lattice_graph_pnp(
                image_points,
                cube_lattice_points_for_coords(
                    lattice_coords,
                    lattice_transform=lattice_transform,
                ),
                camera_matrix,
                display_points=cube_lattice_display_points_for_transform(lattice_transform),
                display_edge_indexes=CUBE_LATTICE_DISPLAY_EDGE_INDEXES,
            )
            if result is None:
                continue
            result.lattice_transform = lattice_transform
            invisible_count = float(getattr(result, "invisible_point_count", 0))
            matched_error = float(result.matched_error)
            score = (invisible_count, matched_error)
            if best_score is None or score < best_score:
                best_score = score
                best_result = result
    return best_result


def solve_lattice_graph_pnp(
    image_points: np.ndarray,
    object_points: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    display_points: np.ndarray,
    display_edge_indexes: tuple[tuple[int, int], ...],
) -> PosePnPResult | None:
    candidates = solve_lattice_graph_pnp_candidates(
        image_points,
        object_points,
        camera_matrix,
        display_points=display_points,
        display_edge_indexes=display_edge_indexes,
    )
    if not candidates:
        return None

    best_result = None
    for result in candidates:
        candidate = (
            int(getattr(result, "invisible_point_count", 0)),
            float(result.matched_error),
        )
        if best_result is None:
            best_result = (candidate, result)
        elif candidate < best_result[0]:
            best_result = (candidate, result)
    return None if best_result is None else best_result[1]


def solve_lattice_graph_pnp_candidates(
    image_points: np.ndarray,
    object_points: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    display_points: np.ndarray,
    display_edge_indexes: tuple[tuple[int, int], ...],
) -> list[PosePnPResult]:
    image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    object_points = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
    display_points = np.asarray(display_points, dtype=np.float32).reshape(-1, 3)
    if len(image_points) != len(object_points) or len(image_points) < 3:
        return []

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    try:
        ok, rvecs, tvecs, _reprojection_errors = cv2.solvePnPGeneric(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_SQPNP,
        )
    except cv2.error:
        return []
    if not ok or not rvecs or not tvecs:
        return []

    candidates: list[PosePnPResult] = []
    for rvec, tvec in zip(rvecs, tvecs):
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                np.asarray(rvec, dtype=np.float64).reshape(3, 1),
                np.asarray(tvec, dtype=np.float64).reshape(3, 1),
            )
        except cv2.error:
            pass
        projected_graph_points, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        projected_graph_points = projected_graph_points.reshape(-1, 2)
        reprojection_distances = np.linalg.norm(projected_graph_points - image_points, axis=1)
        matched_error = float(np.mean(reprojection_distances))
        invisible_count = invisible_surface_point_count(object_points, rvec, tvec)

        projected_display_points, _ = cv2.projectPoints(
            display_points,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        projected_display_points = projected_display_points.reshape(-1, 2)
        candidates.append(
            PosePnPResult(
                error=matched_error,
                inlier_count=int(len(image_points)),
                matched_error=matched_error,
                camera_matrix=np.asarray(camera_matrix, dtype=np.float64),
                dist_coeffs=dist_coeffs,
                rvec=rvec,
                tvec=tvec,
                image_points=image_points,
                object_points=object_points,
                display_points=display_points,
                projected_object_points=projected_graph_points,
                projected_display_points=projected_display_points,
                point_names=tuple(),
                edge_indexes=tuple(),
                display_edge_indexes=display_edge_indexes,
                invisible_point_count=int(invisible_count),
            )
        )
    return candidates


def draw_pose_keypoints(
    frame: np.ndarray,
    result,
    pose_model,
    xy: np.ndarray | None,
    keypoint_conf: np.ndarray | None,
    *,
    confidence_threshold: float,
    class_confidences: dict[str, float],
    box_confidence: np.ndarray | None,
    offset_xy: tuple[int, int],
    scale_xy: tuple[float, float],
) -> int:
    if xy is None or len(xy) == 0:
        return 0

    classes = result_box_classes(result)
    box_confidences = (
        None
        if box_confidence is None
        else np.asarray(box_confidence, dtype=np.float32).reshape(-1)
    )
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

        detection_box_confidence = (
            1.0
            if box_confidences is None or detection_index >= len(box_confidences)
            else float(box_confidences[detection_index])
        )
        if detection_box_confidence < independent_pose_class_threshold(
            class_name,
            class_confidences,
            fallback=confidence_threshold,
        ):
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

            color = POSE_KEYPOINT_COLORS.get(class_name, (255, 255, 255))
            cv2.circle(frame, (x, y), 4, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (x, y), 5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            drawn += 1
    return drawn


def draw_pnp_cube(
    frame: np.ndarray,
    result: PosePnPResult | None,
    *,
    alpha: float = 1.0,
    observed_graph: PoseGraph | None = None,
    profile: dict[str, float | int] | None = None,
) -> None:
    if result is None:
        return

    alpha = float(np.clip(alpha, 0.0, 1.0))
    target = frame if alpha >= 1.0 else frame.copy()

    draw_pnp_debug_pose_candidates(target, result)
    if observed_graph is not None:
        draw_observed_model_distance_lines(target, result, observed_graph)
        if DRAW_POSE_GRAPH_COORDS:
            draw_refinement_point_marker(target, result, observed_graph)

    image_points = np.asarray(result.image_points, dtype=np.float32).reshape(-1, 2)
    projected_object_points = np.asarray(result.projected_object_points, dtype=np.float32).reshape(-1, 2)
    residuals = (
        np.linalg.norm(projected_object_points - image_points, axis=1)
        if len(image_points) == len(projected_object_points)
        else np.empty((0,), dtype=np.float32)
    )
    for observed_point, projected_point in zip(image_points, projected_object_points):
        observed = tuple(np.round(observed_point).astype(int))
        projected_match = tuple(np.round(projected_point).astype(int))
        cv2.line(target, observed, projected_match, (0, 0, 0), 4, lineType=cv2.LINE_AA)
        cv2.line(target, observed, projected_match, PNP_REPROJECTION_RESIDUAL_COLOR, 2, lineType=cv2.LINE_AA)

    anchor_segments = projected_orientation_anchor_segments(result)
    if anchor_segments is None:
        projected = np.asarray(result.projected_display_points, dtype=np.float32)
        edge_indexes = result.display_edge_indexes or result.edge_indexes
        anchor_segments = tuple(
            (
                tuple(np.round(projected[start]).astype(int)),
                tuple(np.round(projected[end]).astype(int)),
                PNP_CUBE_LINE_COLOR,
            )
            for start, end in edge_indexes
        )
    for start_point, end_point, color in anchor_segments:
        cv2.line(target, start_point, end_point, (0, 0, 0), 4, lineType=cv2.LINE_AA)
        cv2.line(target, start_point, end_point, color, 2, lineType=cv2.LINE_AA)
    for point in projected_object_points:
        center = tuple(np.round(point).astype(int))
        cv2.circle(target, center, 5, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(target, center, 3, PNP_CUBE_POINT_COLOR, -1, lineType=cv2.LINE_AA)
    if len(residuals):
        focal = float(result.camera_matrix[0, 0])
        axis_order, signs = result.lattice_transform
        observed_error = getattr(result, "observed_reprojection_error", None)
        if observed_graph is None:
            object_error = float(getattr(result, "matched_error", float(np.mean(residuals))))
        else:
            object_error = observed_pose_model_nearest_same_class_score(
                result,
                observed_graph,
                inlier_threshold_px=PNP_INLIER_THRESHOLD_PX,
            )[0]
        display_error = float(np.mean(residuals)) if observed_error is None else float(observed_error)
        fit_label = "" if observed_error is None else f" fit {float(np.mean(residuals)):.1f}"
        face = getattr(result, "correspondence_face", None)
        face_normal_z = getattr(result, "correspondence_face_normal_z", None)
        face_label = "" if face is None or face_normal_z is None else f" face {face} nz {face_normal_z:.2f}"
        orientation_anchor = getattr(result, "orientation_anchor", None)
        anchor_label = "" if orientation_anchor is None else f" anchor {orientation_anchor.label}"
        rejected_label = " rejected" if getattr(result, "pose_label", "") == "graph-rejected" else ""
        attempt_label = pose_attempt_summary(profile)
        label = (
            f"pnp{rejected_label} {attempt_label} obj {object_error:.1f} err {display_error:.1f}{fit_label} max {float(np.max(residuals)):.1f} "
            f"f {focal:.0f}{face_label}{anchor_label} axes {axis_order} signs {signs}"
        )
        cv2.putText(
            target,
            label,
            (12, max(24, frame.shape[0] - 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            target,
            label,
            (12, max(24, frame.shape[0] - 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            PNP_REPROJECTION_RESIDUAL_COLOR,
            2,
            cv2.LINE_AA,
        )
    if alpha < 1.0:
        cv2.addWeighted(target, alpha, frame, 1.0 - alpha, 0.0, dst=frame)


def projected_orientation_anchor_segments(
    result: PosePnPResult,
) -> tuple[tuple[tuple[int, int], tuple[int, int], tuple[int, int, int]], ...] | None:
    """Return projected slab wireframes colored by the current orientation anchor."""

    anchor = getattr(result, "orientation_anchor", None)
    if anchor is None or anchor.axis is None:
        return None
    if any(
        getattr(result, attribute, None) is None
        for attribute in ("rvec", "tvec", "camera_matrix", "dist_coeffs")
    ):
        return None

    segments = []
    corner_bit_values = tuple(product((0, 1), repeat=3))
    for layer in range(3):
        low = layer
        high = layer + 1
        slab_points = []
        for corner_bits in corner_bit_values:
            coord = [0.0, 0.0, 0.0]
            for axis in range(3):
                if axis == anchor.axis:
                    coord[axis] = float(high if corner_bits[axis] else low)
                else:
                    coord[axis] = float(CUBE_GRID_MAX_INDEX if corner_bits[axis] else 0.0)
            slab_points.append(cube_lattice_point_mm(tuple(coord)))
        projected, _ = cv2.projectPoints(
            np.asarray(slab_points, dtype=np.float32),
            result.rvec,
            result.tvec,
            result.camera_matrix,
            result.dist_coeffs,
        )
        projected = projected.reshape(-1, 2)
        for start in range(8):
            for end in range(start + 1, 8):
                if sum(
                    first != second
                    for first, second in zip(
                        corner_bit_values[start],
                        corner_bit_values[end],
                    )
                ) != 1:
                    continue
                start_point = tuple(np.round(projected[start]).astype(int))
                end_point = tuple(np.round(projected[end]).astype(int))
                color = (
                    PNP_CUBE_LINE_COLOR
                    if layer in anchor.layers
                    else PNP_CUBE_STATIC_LINE_COLOR
                )
                segments.append((start_point, end_point, color))
    return tuple(segments)


def draw_observed_model_distance_lines(
    frame: np.ndarray,
    result: PosePnPResult,
    observed_graph: PoseGraph,
) -> None:
    matches = observed_pose_model_nearest_same_class_matches(result, observed_graph)
    if not matches:
        return

    overlay = frame.copy()
    for observed_xy, projected_xy, _distance, _using_invisible_fallback in matches:
        observed = tuple(np.round(observed_xy).astype(int))
        projected = tuple(np.round(projected_xy).astype(int))
        cv2.line(overlay, observed, projected, PNP_OBSERVED_MODEL_DISTANCE_COLOR, 2, lineType=cv2.LINE_AA)
    cv2.addWeighted(
        overlay,
        float(PNP_OBSERVED_MODEL_DISTANCE_ALPHA),
        frame,
        1.0 - float(PNP_OBSERVED_MODEL_DISTANCE_ALPHA),
        0.0,
        dst=frame,
    )


def draw_refinement_point_marker(
    frame: np.ndarray,
    result: PosePnPResult,
    observed_graph: PoseGraph,
) -> None:
    graph_index = getattr(result, "refinement_graph_index", None)
    if graph_index is None:
        return
    graph_index = int(graph_index)
    if graph_index < 0 or graph_index >= len(observed_graph.points):
        return

    point = observed_graph.points[graph_index]
    center = tuple(np.round(np.asarray(point.xy, dtype=np.float32)).astype(int))
    cv2.circle(frame, center, 9, (0, 0, 0), 2, lineType=cv2.LINE_AA)
    cv2.circle(frame, center, 7, PNP_REFINEMENT_POINT_COLOR, 2, lineType=cv2.LINE_AA)

    lattice_coord = getattr(result, "refinement_lattice_coord", None)
    label = "refine" if lattice_coord is None else f"refine {lattice_coord}"
    label_xy = (center[0] + 10, center[1] - 8)
    cv2.putText(
        frame,
        label,
        label_xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        label_xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        PNP_REFINEMENT_POINT_COLOR,
        1,
        cv2.LINE_AA,
    )


def draw_pnp_debug_pose_candidates(frame: np.ndarray, result: PosePnPResult) -> None:
    candidates = tuple(getattr(result, "debug_pose_candidates", tuple()))
    if not candidates:
        return

    overlay = frame.copy()
    for candidate_index, candidate in enumerate(candidates):
        projected = np.asarray(candidate.projected_display_points, dtype=np.float32).reshape(-1, 2)
        edge_indexes = candidate.display_edge_indexes or candidate.edge_indexes
        if len(projected) == 0 or not edge_indexes:
            continue
        color = PNP_DEBUG_POSE_COLORS[candidate_index % len(PNP_DEBUG_POSE_COLORS)]
        for start, end in edge_indexes:
            start_point = tuple(np.round(projected[start]).astype(int))
            end_point = tuple(np.round(projected[end]).astype(int))
            cv2.line(overlay, start_point, end_point, (0, 0, 0), 3, lineType=cv2.LINE_AA)
            cv2.line(overlay, start_point, end_point, color, 1, lineType=cv2.LINE_AA)

        label_point = tuple(np.round(projected[0]).astype(int))
        label = f"{candidate_index + 1}:{float(getattr(candidate, 'observed_reprojection_error', 0.0)):.1f}"
        cv2.putText(
            overlay,
            label,
            (label_point[0] + 4, label_point[1] + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            label,
            (label_point[0] + 4, label_point[1] + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.addWeighted(
        overlay,
        float(PNP_DEBUG_POSE_ALPHA),
        frame,
        1.0 - float(PNP_DEBUG_POSE_ALPHA),
        0.0,
        dst=frame,
    )


def odd_kernel(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def build_adjacency_edge_map(roi: np.ndarray, args, *, roi_side: float | None = None) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_side = float(max(roi.shape[:2])) if roi_side is None else float(roi_side)
    blur_kernel = odd_kernel(
        scaled_adjacency_int_px(
            getattr(args, "adjacency_canny_blur", ADJACENCY_CANNY_BLUR_KERNEL),
            roi_side,
            args,
            min_value=1,
        )
    )
    if blur_kernel > 1:
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    canny_low = max(0, int(getattr(args, "adjacency_canny_low", ADJACENCY_CANNY_LOW)))
    canny_high = max(1, int(getattr(args, "adjacency_canny_high", ADJACENCY_CANNY_HIGH)))
    if canny_low > canny_high:
        canny_low, canny_high = canny_high, canny_low
    return cv2.Canny(gray, canny_low, canny_high)


def merged_interval_coverage(
    intervals: list[tuple[float, float]],
    segment_length: float,
) -> tuple[float, float]:
    if segment_length <= 0.0 or not intervals:
        return 0.0, segment_length

    clipped = [
        (max(0.0, start), min(segment_length, end))
        for start, end in intervals
        if end >= 0.0 and start <= segment_length
    ]
    clipped = [(start, end) for start, end in clipped if end > start]
    if not clipped:
        return 0.0, segment_length

    clipped.sort()
    merged: list[tuple[float, float]] = []
    for start, end in clipped:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    covered = sum(end - start for start, end in merged)
    max_gap = merged[0][0]
    for previous, current in zip(merged, merged[1:]):
        max_gap = max(max_gap, current[0] - previous[1])
    max_gap = max(max_gap, segment_length - merged[-1][1])
    return covered / segment_length, max_gap


POSE_GRAPH_NEIGHBOR_LIMITS = {
    "corner": {"corner-edge": 3},
    "corner-edge": {"corner": 1, "corner-edge": 1, "corner-edge-center": 2},
    "corner-edge-center": {"corner-edge-center": 2, "corner-edge": 2},
}


def pose_graph_candidate_edges(points: list[PoseGraphPoint]) -> list[tuple[int, int]]:
    candidate_edges: set[tuple[int, int]] = set()
    for source_index, source in enumerate(points):
        neighbor_limits = POSE_GRAPH_NEIGHBOR_LIMITS.get(source.class_name, {})
        source_xy = np.asarray(source.xy, dtype=np.float32)
        for neighbor_class, limit in neighbor_limits.items():
            neighbors = [
                (
                    float(np.linalg.norm(np.asarray(target.xy, dtype=np.float32) - source_xy)),
                    target_index,
                )
                for target_index, target in enumerate(points)
                if target_index != source_index and target.class_name == neighbor_class
            ]
            neighbors.sort(key=lambda item: item[0])
            for _distance, target_index in neighbors[: max(0, limit)]:
                candidate_edges.add(tuple(sorted((source_index, target_index))))
    return sorted(candidate_edges)


def edge_support_along_segment(
    edge_map: np.ndarray,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
) -> tuple[float, float]:
    start = np.asarray(start_xy, dtype=np.float32)
    end = np.asarray(end_xy, dtype=np.float32)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length <= 1.0:
        return 0.0, length

    sample_count = max(2, int(np.ceil(length)) + 1)
    t = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)
    samples = start[None, :] + ((end - start)[None, :] * t[:, None])
    xs = np.round(samples[:, 0]).astype(np.int32)
    ys = np.round(samples[:, 1]).astype(np.int32)
    inside = (
        (xs >= 0)
        & (xs < edge_map.shape[1])
        & (ys >= 0)
        & (ys < edge_map.shape[0])
    )
    support = np.zeros(sample_count, dtype=bool)
    support[inside] = edge_map[ys[inside], xs[inside]] > 0

    if not bool(support.any()):
        return 0.0, length

    sample_step = length / max(1, sample_count - 1)
    intervals: list[tuple[float, float]] = []
    run_start: int | None = None
    for sample_index, has_edge in enumerate(support):
        if has_edge and run_start is None:
            run_start = sample_index
        if (not has_edge or sample_index == len(support) - 1) and run_start is not None:
            run_end = sample_index if has_edge else sample_index - 1
            intervals.append(
                (
                    (run_start * sample_step) - sample_step,
                    (run_end * sample_step) + sample_step,
                )
            )
            run_start = None

    return merged_interval_coverage(intervals, length)


def dilated_adjacency_edge_map(
    edge_map: np.ndarray,
    args,
    *,
    roi_side: float | None = None,
) -> np.ndarray:
    roi_side = float(max(edge_map.shape[:2])) if roi_side is None else float(roi_side)
    dilation_px = scaled_adjacency_int_px(
        getattr(args, "adjacency_edge_dilation", ADJACENCY_EDGE_DILATION_PX),
        roi_side,
        args,
    )
    if dilation_px <= 0:
        return edge_map
    kernel_size = (dilation_px * 2) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(edge_map, kernel)


def adjacency_edge_strip_width_px(args, *, roi_side: float) -> int:
    dilation_px = scaled_adjacency_int_px(
        getattr(args, "adjacency_edge_dilation", ADJACENCY_EDGE_DILATION_PX),
        roi_side,
        args,
    )
    return (dilation_px * 2) + 1


def segment_skips_observed_keypoint(
    points: list[PoseGraphPoint],
    start_index: int,
    end_index: int,
    args,
    *,
    roi_side: float,
) -> bool:
    start = np.asarray(points[start_index].xy, dtype=np.float32)
    end = np.asarray(points[end_index].xy, dtype=np.float32)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length <= 1.0:
        return False

    unit = delta / length
    intersection_distance = max(
        0.0,
        scaled_adjacency_px(
            getattr(
                args,
                "adjacency_keypoint_intersection_distance",
                ADJACENCY_KEYPOINT_INTERSECTION_DISTANCE_PX,
            ),
            roi_side,
            args,
        ),
    )
    endpoint_margin = min(intersection_distance, length * 0.25)
    for point_index, point in enumerate(points):
        if point_index in (start_index, end_index):
            continue
        point_xy = np.asarray(point.xy, dtype=np.float32)
        projection = float(np.dot(point_xy - start, unit))
        if projection <= endpoint_margin or projection >= length - endpoint_margin:
            continue
        closest = start + (unit * projection)
        if float(np.linalg.norm(point_xy - closest)) <= intersection_distance:
            return True
    return False


def pose_graph_edge_evidence(
    points: list[PoseGraphPoint],
    roi_xyxy: tuple[int, int, int, int],
    edge_map: np.ndarray,
    args,
) -> list[PoseGraphEdge]:
    if len(points) < 2:
        return []

    roi_offset = np.asarray(roi_xyxy[:2], dtype=np.float32)
    roi_side = roi_side_px(roi_xyxy)
    sampled_edges = dilated_adjacency_edge_map(edge_map, args, roi_side=roi_side)
    min_coverage = max(
        0.0,
        min(1.0, float(getattr(args, "adjacency_min_coverage", ADJACENCY_MIN_COVERAGE))),
    )
    max_gap_ratio = max(
        0.0,
        float(
            getattr(
                args,
                "adjacency_max_uncovered_gap_ratio",
                ADJACENCY_MAX_UNCOVERED_GAP_RATIO,
            )
        ),
    )
    min_edge_length = max(
        0.0,
        scaled_adjacency_px(
            getattr(args, "adjacency_min_edge_length", ADJACENCY_MIN_EDGE_LENGTH_PX),
            roi_side,
            args,
        ),
    )

    graph_edges: list[PoseGraphEdge] = []
    for start_index, end_index in pose_graph_candidate_edges(points):
        start = tuple(np.asarray(points[start_index].xy, dtype=np.float32) - roi_offset)
        end = tuple(np.asarray(points[end_index].xy, dtype=np.float32) - roi_offset)
        length = float(
            np.linalg.norm(np.asarray(end, dtype=np.float32) - np.asarray(start, dtype=np.float32))
        )
        if length < min_edge_length:
            continue
        skips_keypoint = segment_skips_observed_keypoint(
            points,
            start_index,
            end_index,
            args,
            roi_side=roi_side,
        )
        if skips_keypoint:
            coverage, max_gap = 0.0, length
        else:
            coverage, max_gap = edge_support_along_segment(sampled_edges, start, end)
        accepted = (
            not skips_keypoint
            and coverage >= min_coverage
            and max_gap <= (max_gap_ratio * length)
        )
        graph_edges.append(
            PoseGraphEdge(
                start_index=start_index,
                end_index=end_index,
                coverage=coverage,
                max_uncovered_gap=max_gap,
                accepted=accepted,
            )
        )
    return graph_edges


def pose_graph_edge_length(
    points: tuple[PoseGraphPoint, ...] | list[PoseGraphPoint],
    edge: PoseGraphEdge,
) -> float:
    start = np.asarray(points[edge.start_index].xy, dtype=np.float32)
    end = np.asarray(points[edge.end_index].xy, dtype=np.float32)
    return float(np.linalg.norm(end - start))


def prune_long_pose_graph_edges(graph: PoseGraph, args) -> PoseGraph:
    accepted_lengths = [
        pose_graph_edge_length(graph.points, edge)
        for edge in graph.edges
        if edge.accepted
    ]
    if not accepted_lengths:
        return graph

    max_length_ratio = max(
        0.0,
        float(
            getattr(
                args,
                "adjacency_max_average_length_ratio",
                ADJACENCY_MAX_AVERAGE_LENGTH_RATIO,
            )
        ),
    )
    if max_length_ratio <= 0.0:
        return graph

    max_length = float(np.mean(accepted_lengths)) * max_length_ratio
    pruned_edges = tuple(
        (
            PoseGraphEdge(
                start_index=edge.start_index,
                end_index=edge.end_index,
                coverage=edge.coverage,
                max_uncovered_gap=edge.max_uncovered_gap,
                accepted=False,
            )
            if edge.accepted and pose_graph_edge_length(graph.points, edge) > max_length
            else edge
        )
        for edge in graph.edges
    )
    return PoseGraph(points=graph.points, edges=pruned_edges)


def build_pose_graph(
    points: list[PoseGraphPoint],
    roi: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
    args,
) -> PoseGraph:
    if len(points) < 2:
        return PoseGraph(points=tuple(points), edges=tuple())
    profile = getattr(args, "_pose_profile", None)
    started = perf_counter()
    side = roi_side_px(roi_xyxy)
    edges = build_adjacency_edge_map(roi, args, roi_side=side)
    if profile is not None:
        profile["edge_map_ms"] = (perf_counter() - started) * 1000.0
        started = perf_counter()
    graph = PoseGraph(
        points=tuple(points),
        edges=tuple(pose_graph_edge_evidence(points, roi_xyxy, edges, args)),
    )
    if profile is not None:
        profile["edge_evidence_ms"] = (perf_counter() - started) * 1000.0
        profile["candidate_edges"] = len(graph.edges)
        profile["accepted_edges_before_prune"] = sum(edge.accepted for edge in graph.edges)
        started = perf_counter()
    graph = prune_long_pose_graph_edges(graph, args)
    if profile is not None:
        profile["edge_prune_ms"] = (perf_counter() - started) * 1000.0
        profile["accepted_edges"] = sum(edge.accepted for edge in graph.edges)
    return graph


def pose_graph_adjacency(graph: PoseGraph) -> dict[int, list[int]]:
    adjacency = {index: [] for index in range(len(graph.points))}
    for edge in graph.edges:
        if not edge.accepted:
            continue
        adjacency[edge.start_index].append(edge.end_index)
        adjacency[edge.end_index].append(edge.start_index)
    for neighbors in adjacency.values():
        neighbors.sort(key=lambda index: (-graph.points[index].score, index))
    return adjacency


def pose_graph_components(
    graph: PoseGraph,
    adjacency: dict[int, list[int]] | None = None,
) -> list[tuple[int, ...]]:
    adjacency = pose_graph_adjacency(graph) if adjacency is None else adjacency
    seen: set[int] = set()
    components: list[tuple[int, ...]] = []
    for start_index in range(len(graph.points)):
        if start_index in seen:
            continue
        queue = deque([start_index])
        seen.add(start_index)
        component = []
        while queue:
            index = queue.popleft()
            component.append(index)
            for neighbor in adjacency[index]:
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                queue.append(neighbor)
        components.append(tuple(component))
    return components


def cube_lattice_class(coord: tuple[int, int, int]) -> str:
    boundary_count = sum(value in (0, CUBE_GRID_MAX_INDEX) for value in coord)
    if boundary_count == 3:
        return "corner"
    if boundary_count == 2:
        return "corner-edge"
    if boundary_count == 1:
        return "corner-edge-center"
    return "center"


CUBE_LATTICE_COORDS_BY_CLASS = {
    class_name: tuple(coord for coord in CUBE_LATTICE_COORDS if cube_lattice_class(coord) == class_name)
    for class_name in ("corner", "corner-edge", "corner-edge-center")
}


CUBE_LATTICE_POINT_BY_COORD = {
    coord: CUBE_LATTICE_POINTS_MM[index]
    for index, coord in enumerate(CUBE_LATTICE_COORDS)
}
CUBE_LATTICE_DISPLAY_CORNER_COORDS = tuple(
    coord for coord in CUBE_LATTICE_COORDS if cube_lattice_class(coord) == "corner"
)
CUBE_LATTICE_DISPLAY_POINTS_MM = np.asarray(
    [CUBE_LATTICE_POINT_BY_COORD[coord] for coord in CUBE_LATTICE_DISPLAY_CORNER_COORDS],
    dtype=np.float32,
)
CUBE_LATTICE_DISPLAY_EDGE_INDEXES = tuple(
    (start_index, end_index)
    for start_index, start_coord in enumerate(CUBE_LATTICE_DISPLAY_CORNER_COORDS)
    for end_index, end_coord in enumerate(CUBE_LATTICE_DISPLAY_CORNER_COORDS)
    if start_index < end_index
    and sum(abs(start_value - end_value) for start_value, end_value in zip(start_coord, end_coord))
    == CUBE_GRID_MAX_INDEX
)


def shared_lattice_boundary_faces(
    lattice_coords: tuple[tuple[int, int, int], ...],
) -> frozenset[tuple[int, int]]:
    if not lattice_coords:
        return frozenset()
    shared_faces = set(cube_lattice_boundary_faces(lattice_coords[0]))
    for coord in lattice_coords[1:]:
        shared_faces &= set(cube_lattice_boundary_faces(coord))
    return frozenset(shared_faces)


def lattice_correspondence_face_is_visible(
    lattice_coords: tuple[tuple[int, int, int], ...],
    rvec: np.ndarray,
) -> bool:
    shared_faces = shared_lattice_boundary_faces(lattice_coords)
    if not shared_faces:
        return False
    return any(lattice_boundary_face_is_visible(face, rvec) for face in shared_faces)


def projected_lattice_points_by_class(
    result: PosePnPResult,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    projected_by_class: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if any(
        getattr(result, attribute, None) is None
        for attribute in ("rvec", "tvec", "camera_matrix", "dist_coeffs")
    ):
        return projected_by_class

    for class_name in ("corner", "corner-edge", "corner-edge-center"):
        coords = CUBE_LATTICE_COORDS_BY_CLASS[class_name]
        object_points = cube_lattice_points_for_coords(
            coords,
            lattice_transform=getattr(result, "lattice_transform", CUBE_LATTICE_IDENTITY_TRANSFORM),
        )
        try:
            projected_points, _ = cv2.projectPoints(
                object_points,
                result.rvec,
                result.tvec,
                result.camera_matrix,
                result.dist_coeffs,
            )
        except cv2.error:
            continue
        projected_by_class[class_name] = (
            projected_points.reshape(-1, 2).astype(np.float32, copy=False),
            object_points.astype(np.float32, copy=False),
        )
    return projected_by_class


def observed_pose_model_nearest_same_class_score(
    result: PosePnPResult,
    observed_graph: PoseGraph,
    *,
    inlier_threshold_px: float,
) -> tuple[float, int, int, int]:
    matches = observed_pose_model_nearest_same_class_matches(result, observed_graph)
    errors = [distance for _observed_xy, _projected_xy, distance, _fallback in matches]
    observed_point_count = sum(
        1
        for point in observed_graph.points
        if point.class_name in ("corner", "corner-edge", "corner-edge-center")
    )
    invisible_observed_count = sum(1 for *_points, using_invisible_fallback in matches if using_invisible_fallback)
    if not errors:
        return float("inf"), 0, observed_point_count, invisible_observed_count
    inlier_count = sum(error <= float(inlier_threshold_px) for error in errors)
    return float(np.mean(errors)), int(inlier_count), len(errors), invisible_observed_count


def observed_pose_model_backface_fit_count(
    result: PosePnPResult,
    observed_graph: PoseGraph,
    *,
    inlier_threshold_px: float,
) -> tuple[int, int]:
    projected_by_class = projected_lattice_points_by_class(result)
    backface_fit_count = 0
    tested_count = 0
    for class_name in ("corner", "corner-edge", "corner-edge-center"):
        observed_points = [
            point
            for point in observed_graph.points
            if point.class_name == class_name
        ]
        if not observed_points:
            continue
        projected_entry = projected_by_class.get(class_name)
        if projected_entry is None:
            continue
        projected_points, object_points = projected_entry
        if len(projected_points) == 0 or len(projected_points) != len(object_points):
            continue
        visible_mask = np.asarray(
            [
                result_object_point_has_visible_surface(object_point, result)
                for object_point in object_points
            ],
            dtype=bool,
        )
        hidden_projected_points = projected_points[~visible_mask]
        if len(hidden_projected_points) == 0:
            continue
        visible_projected_points = projected_points[visible_mask]
        observed_xy = np.asarray([point.xy for point in observed_points], dtype=np.float32).reshape(-1, 2)
        hidden_distances = np.linalg.norm(
            observed_xy[:, None, :] - hidden_projected_points[None, :, :],
            axis=2,
        )
        nearest_hidden = np.min(hidden_distances, axis=1)
        if len(visible_projected_points) == 0:
            nearest_visible = np.full_like(nearest_hidden, float("inf"), dtype=np.float32)
        else:
            visible_distances = np.linalg.norm(
                observed_xy[:, None, :] - visible_projected_points[None, :, :],
                axis=2,
            )
            nearest_visible = np.min(visible_distances, axis=1)
        tested_count += len(observed_xy)
        backface_fit_count += int(
            np.count_nonzero(
                (nearest_hidden <= float(inlier_threshold_px))
                & (nearest_hidden <= nearest_visible)
            )
        )
    return backface_fit_count, tested_count


def observed_pose_model_nearest_same_class_matches(
    result: PosePnPResult,
    observed_graph: PoseGraph,
) -> list[tuple[np.ndarray, np.ndarray, float, bool]]:
    projected_by_class = projected_lattice_points_by_class(result)
    matches: list[tuple[np.ndarray, np.ndarray, float, bool]] = []
    observed_points_by_class: dict[str, list[PoseGraphPoint]] = {
        class_name: [
            point
            for point in observed_graph.points
            if point.class_name == class_name
        ]
        for class_name in ("corner", "corner-edge", "corner-edge-center")
    }
    for class_name, observed_points in observed_points_by_class.items():
        if not observed_points:
            continue
        projected_entry = projected_by_class.get(class_name)
        if projected_entry is None:
            continue
        projected_points, object_points = projected_entry
        if len(projected_points) == 0 or len(projected_points) != len(object_points):
            continue

        visible_mask = np.asarray(
            [
                result_object_point_has_visible_surface(object_point, result)
                for object_point in object_points
            ],
            dtype=bool,
        )
        candidate_projected_points = projected_points[visible_mask]
        using_invisible_fallback = False
        if len(candidate_projected_points) == 0:
            candidate_projected_points = projected_points
            using_invisible_fallback = True

        observed_xy = np.asarray([point.xy for point in observed_points], dtype=np.float32).reshape(-1, 2)
        distance_matrix = np.linalg.norm(
            observed_xy[:, None, :] - candidate_projected_points[None, :, :],
            axis=2,
        )
        nearest_model_indexes = np.argmin(distance_matrix, axis=1)
        for observed_xy_point, model_index in zip(observed_xy, nearest_model_indexes):
            projected_xy = candidate_projected_points[int(model_index)]
            distance = float(np.linalg.norm(observed_xy_point - projected_xy))
            matches.append((observed_xy_point, projected_xy, distance, using_invisible_fallback))

    return matches


CUBE_LATTICE_BOUNDARY_FACES_BY_COORD = {
    coord: frozenset(
        (axis, value)
        for axis, value in enumerate(coord)
        if value in (0, CUBE_GRID_MAX_INDEX)
    )
    for coord in CUBE_LATTICE_COORDS
}
def cube_lattice_coords_are_neighbors(
    left: tuple[int, int, int],
    right: tuple[int, int, int],
) -> bool:
    return sum(abs(left_value - right_value) for left_value, right_value in zip(left, right)) == 1


VECTOR_GUIDED_START_COORDS = {
    "corner": (0, 0, 0),
    "corner-edge": (1, 0, 0),
    "corner-edge-center": (1, 1, 0),
}
VECTOR_GUIDED_DIRECTION_PRIORITY = (
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (-1, 0, 0),
    (0, -1, 0),
    (0, 0, -1),
)
VECTOR_GUIDED_DIRECTION_RANK = {
    direction: index
    for index, direction in enumerate(VECTOR_GUIDED_DIRECTION_PRIORITY)
}
VECTOR_GUIDED_LOCAL_TURN_ORDER = ("straight", "left", "right", "back")
VECTOR_GUIDED_LOCAL_TURNS = {
    (1, 0, 0): {
        "straight": (1, 0, 0),
        "left": (0, 1, 0),
        "right": (0, 0, 1),
        "back": (-1, 0, 0),
    },
    (0, 1, 0): {
        "straight": (0, 1, 0),
        "left": (0, 0, 1),
        "right": (1, 0, 0),
        "back": (0, -1, 0),
    },
    (0, 0, 1): {
        "straight": (0, 0, 1),
        "left": (1, 0, 0),
        "right": (0, 1, 0),
        "back": (0, 0, -1),
    },
    (-1, 0, 0): {
        "straight": (-1, 0, 0),
        "left": (0, 0, 1),
        "right": (0, 1, 0),
        "back": (1, 0, 0),
    },
    (0, -1, 0): {
        "straight": (0, -1, 0),
        "left": (1, 0, 0),
        "right": (0, 0, 1),
        "back": (0, 1, 0),
    },
    (0, 0, -1): {
        "straight": (0, 0, -1),
        "left": (0, 1, 0),
        "right": (1, 0, 0),
        "back": (0, 0, 1),
    },
}
CUBE_LATTICE_NEIGHBOR_COORDS_BY_CLASS = {
    (coord, class_name): tuple(
        candidate
        for direction in VECTOR_GUIDED_DIRECTION_PRIORITY
        for candidate in (tuple(coord[index] + direction[index] for index in range(3)),)
        if all(0 <= value <= CUBE_GRID_MAX_INDEX for value in candidate)
        and candidate in CUBE_LATTICE_POINT_BY_COORD
        and cube_lattice_class(candidate) == class_name
    )
    for coord in CUBE_LATTICE_COORDS
    for class_name in ("corner", "corner-edge", "corner-edge-center")
}


def normalized_vector(vector: np.ndarray) -> np.ndarray | None:
    length = float(np.linalg.norm(vector))
    if length <= 1e-6:
        return None
    return (vector / length).astype(np.float32)


def cube_lattice_neighbor_coords(
    coord: tuple[int, int, int],
    class_name: str,
    *,
    used_coords: set[tuple[int, int, int]] | None = None,
) -> list[tuple[int, int, int]]:
    used_coords = set() if used_coords is None else used_coords
    return [
        candidate
        for candidate in CUBE_LATTICE_NEIGHBOR_COORDS_BY_CLASS.get((coord, class_name), ())
        if candidate not in used_coords
    ]


def cube_lattice_boundary_faces(coord: tuple[int, int, int]) -> frozenset[tuple[int, int]]:
    return CUBE_LATTICE_BOUNDARY_FACES_BY_COORD[coord]


def cube_lattice_shared_boundary_faces(
    left: tuple[int, int, int],
    right: tuple[int, int, int],
) -> frozenset[tuple[int, int]]:
    return cube_lattice_boundary_faces(left) & cube_lattice_boundary_faces(right)


def axis_delta(axis: int, sign: int) -> tuple[int, int, int]:
    return tuple(sign if index == axis else 0 for index in range(3))


def cross_delta(
    left: tuple[int, int, int],
    right: tuple[int, int, int],
) -> tuple[int, int, int]:
    return (
        (left[1] * right[2]) - (left[2] * right[1]),
        (left[2] * right[0]) - (left[0] * right[2]),
        (left[0] * right[1]) - (left[1] * right[0]),
    )


def face_normal_delta(face: tuple[int, int]) -> tuple[int, int, int]:
    axis, value = face
    return axis_delta(axis, -1 if value == 0 else 1)


def face_local_turn_basis(
    previous_delta: tuple[int, int, int],
    face: tuple[int, int],
) -> dict[str, tuple[int, int, int]] | None:
    normal = face_normal_delta(face)
    if sum(abs(normal[index] * previous_delta[index]) for index in range(3)) != 0:
        return None
    right = cross_delta(normal, previous_delta)
    if sum(abs(value) for value in right) != 1:
        return None
    return {
        "straight": previous_delta,
        "left": negated_delta(right),
        "right": right,
        "back": negated_delta(previous_delta),
    }


def coord_delta_axis_and_sign(
    start: tuple[int, int, int],
    end: tuple[int, int, int],
) -> tuple[int, int] | None:
    delta = tuple(end[index] - start[index] for index in range(3))
    if sum(abs(value) for value in delta) != 1:
        return None
    axis = next(index for index, value in enumerate(delta) if value != 0)
    return axis, delta[axis]


def coord_delta(
    start: tuple[int, int, int],
    end: tuple[int, int, int],
) -> tuple[int, int, int] | None:
    delta = tuple(end[index] - start[index] for index in range(3))
    if sum(abs(value) for value in delta) != 1:
        return None
    return delta


def local_turn_scores(previous_unit: np.ndarray, next_unit: np.ndarray) -> dict[str, float]:
    dot = float(np.dot(previous_unit, next_unit))
    cross = float(
        (previous_unit[0] * next_unit[1])
        - (previous_unit[1] * next_unit[0])
    )
    return {
        "straight": dot,
        "left": -cross,
        "right": cross,
        "back": -dot,
    }


def local_turn_for_delta(
    previous_delta: tuple[int, int, int],
    next_delta: tuple[int, int, int],
) -> str | None:
    local_turns = VECTOR_GUIDED_LOCAL_TURNS.get(previous_delta, {})
    for turn_name, turn_delta in local_turns.items():
        if turn_delta == next_delta:
            return turn_name
    return None


def local_turn_basis_for_delta(delta: tuple[int, int, int]) -> dict[str, tuple[int, int, int]] | None:
    local_turns = VECTOR_GUIDED_LOCAL_TURNS.get(delta)
    return None if local_turns is None else dict(local_turns)


def local_turn_for_basis_delta(
    basis: dict[str, tuple[int, int, int]],
    delta: tuple[int, int, int],
) -> str | None:
    for turn_name, turn_delta in basis.items():
        if turn_delta == delta:
            return turn_name
    return None


def candidate_local_turn_name(
    basis: dict[str, tuple[int, int, int]],
    previous_delta: tuple[int, int, int],
    next_delta: tuple[int, int, int],
    previous_unit: np.ndarray,
    observed_unit: np.ndarray,
    turn_scores: dict[str, float] | None = None,
) -> str | None:
    turn_name = local_turn_for_basis_delta(basis, next_delta)
    if turn_name is not None:
        return turn_name
    return None


def negated_delta(delta: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(-value for value in delta)


def propagated_local_to_global_frame(
    local_to_global: dict[str, tuple[int, int, int]],
    turn_name: str,
) -> dict[str, tuple[int, int, int]]:
    straight = local_to_global[turn_name]
    if turn_name == "straight":
        left = local_to_global["left"]
        right = local_to_global["right"]
    elif turn_name == "left":
        left = local_to_global["right"]
        right = local_to_global["straight"]
    elif turn_name == "right":
        left = local_to_global["straight"]
        right = local_to_global["left"]
    else:
        left = local_to_global["right"]
        right = local_to_global["left"]
    return {
        "straight": straight,
        "left": left,
        "right": right,
        "back": negated_delta(straight),
    }


def unique_local_to_global_frames(
    previous_delta: tuple[int, int, int],
    path_faces: frozenset[tuple[int, int]],
    fallback: dict[str, tuple[int, int, int]],
) -> tuple[dict[str, tuple[int, int, int]], ...]:
    frames = []
    seen = set()
    for face in sorted(path_faces):
        frame = face_local_turn_basis(previous_delta, face)
        if frame is None:
            continue
        key = tuple(sorted(frame.items()))
        if key in seen:
            continue
        frames.append(frame)
        seen.add(key)
    key = tuple(sorted(fallback.items()))
    if key not in seen:
        frames.append(fallback)
    return tuple(frames)


def image_axis_preference(delta: tuple[int, int, int], observed_unit: np.ndarray) -> float:
    axis_and_sign = coord_delta_axis_and_sign((0, 0, 0), delta)
    if axis_and_sign is None:
        return 0.0
    axis, sign = axis_and_sign
    if axis == 0:
        return float(observed_unit[0]) * float(sign)
    if axis == 1:
        return -float(observed_unit[0]) * float(sign)
    return float(observed_unit[1]) * float(sign)


def pose_graph_correspondence_from_assignments(
    graph: PoseGraph,
    assignments: dict[int, PoseGraphNodeAssignment],
) -> PoseGraphCorrespondence | None:
    if len(assignments) < 3 or not any(assignment.has_turn for assignment in assignments.values()):
        return None

    graph_indexes = tuple(
        sorted(
            assignments,
            key=lambda index: (-graph.points[index].score, index),
        )
    )
    lattice_coords = tuple(assignments[index].coord for index in graph_indexes)
    image_points = np.asarray([graph.points[index].xy for index in graph_indexes], dtype=np.float32)
    return PoseGraphCorrespondence(
        image_points=image_points,
        object_points=cube_lattice_points_for_coords(lattice_coords),
        graph_indexes=graph_indexes,
        lattice_coords=lattice_coords,
    )


def mirrored_first_neighbor_lattice_coord(
    start_class_name: str,
    first_neighbor_class_name: str,
    first_neighbor_coord: tuple[int, int, int],
) -> tuple[int, int, int] | None:
    mirror_pairs: tuple[
        tuple[str, str, tuple[int, int, int], tuple[int, int, int]],
        ...
    ] = (
        ("corner-edge", "corner-edge-center", (1, 1, 0), (1, 0, 1)),
        ("corner-edge-center", "corner-edge-center", (2, 1, 0), (1, 2, 0)),
        ("corner-edge-center", "corner-edge", (1, 0, 0), (0, 1, 0)),
    )
    for pair_start_class, pair_neighbor_class, first_coord, second_coord in mirror_pairs:
        if (
            start_class_name != pair_start_class
            or first_neighbor_class_name != pair_neighbor_class
        ):
            continue
        if first_neighbor_coord == first_coord:
            return second_coord
        if first_neighbor_coord == second_coord:
            return first_coord
    return None


def pose_graph_component_has_observed_turn(
    graph: PoseGraph,
    adjacency: tuple[tuple[int, ...], ...],
    component: tuple[int, ...],
    *,
    min_cross: float = 0.15,
) -> bool:
    component_set = set(component)
    for current_index in component:
        current_xy = np.asarray(graph.points[current_index].xy, dtype=np.float32)
        neighbor_units = []
        for neighbor_index in adjacency[current_index]:
            if neighbor_index not in component_set:
                continue
            neighbor_unit = normalized_vector(
                np.asarray(graph.points[neighbor_index].xy, dtype=np.float32) - current_xy
            )
            if neighbor_unit is not None:
                neighbor_units.append(neighbor_unit)
        for first_index, first_unit in enumerate(neighbor_units):
            for second_unit in neighbor_units[first_index + 1:]:
                cross = float(
                    (first_unit[0] * second_unit[1])
                    - (first_unit[1] * second_unit[0])
                )
                if abs(cross) >= float(min_cross):
                    return True
    return False


def weighted_shuffle_start_options(
    start_options: list[tuple[tuple[int, ...], int]],
    point_weights: dict[int, float],
) -> list[tuple[tuple[int, ...], int]]:
    remaining = list(start_options)
    shuffled = []
    while remaining:
        weights = [max(0.0, float(point_weights.get(index, 1.0))) for _component, index in remaining]
        total = sum(weights)
        if total <= 0.0:
            random.shuffle(remaining)
            shuffled.extend(remaining)
            break
        target = random.random() * total
        cumulative = 0.0
        selected_index = len(remaining) - 1
        for option_index, weight in enumerate(weights):
            cumulative += weight
            if target <= cumulative:
                selected_index = option_index
                break
        shuffled.append(remaining.pop(selected_index))
    return shuffled


def _vector_guided_pose_graph_correspondence_candidates_from_start(
    graph: PoseGraph,
    adjacency: tuple[tuple[int, ...], ...],
    component: tuple[int, ...],
    start_index: int,
    *,
    max_candidates: int | None = None,
    point_weights: dict[int, float] | None = None,
    exclude_keys: set[tuple[tuple[int, ...], tuple[tuple[int, int, int], ...]]] | None = None,
) -> tuple[PoseGraphCorrespondence, ...]:
    component_set = set(component)
    start = graph.points[start_index]
    start_coord = VECTOR_GUIDED_START_COORDS.get(start.class_name)
    if start_coord is None:
        return tuple()

    def weighted_point_score(index: int) -> float:
        weight = 1.0 if point_weights is None else float(point_weights.get(index, 1.0))
        return float(graph.points[index].score) * max(0.0, weight)

    start_assignment = PoseGraphNodeAssignment(
        coord=start_coord,
        parent_index=None,
        local_to_global=None,
        local_to_global_frames=tuple(),
        path_faces=cube_lattice_boundary_faces(start_coord),
        depth=0,
        has_turn=False,
    )
    start_used_coords = {start_coord}

    first_neighbor_options = []
    for neighbor_index in adjacency[start_index]:
        if neighbor_index not in component_set:
            continue
        neighbor = graph.points[neighbor_index]
        observed_unit = normalized_vector(
            np.asarray(neighbor.xy, dtype=np.float32)
            - np.asarray(start.xy, dtype=np.float32)
        )
        if observed_unit is None:
            continue
        for neighbor_coord in cube_lattice_neighbor_coords(
            start_coord,
            neighbor.class_name,
            used_coords=start_used_coords,
        ):
            delta = tuple(neighbor_coord[index] - start_coord[index] for index in range(3))
            first_neighbor_options.append(
                (
                    -weighted_point_score(neighbor_index),
                    VECTOR_GUIDED_DIRECTION_RANK.get(delta, len(VECTOR_GUIDED_DIRECTION_PRIORITY)),
                    neighbor_index,
                    neighbor_coord,
                    observed_unit,
                )
            )
    if not first_neighbor_options:
        return tuple()

    correspondences: list[PoseGraphCorrespondence] = []
    correspondence_keys: set[tuple[tuple[int, ...], tuple[tuple[int, int, int], ...]]] = set()

    def add_correspondence(
        assignments: dict[int, PoseGraphNodeAssignment],
    ) -> bool:
        correspondence = pose_graph_correspondence_from_assignments(graph, assignments)
        if correspondence is None:
            return False
        key = correspondence.key
        if key in correspondence_keys or (exclude_keys is not None and key in exclude_keys):
            return False
        correspondence_keys.add(key)
        correspondences.append(correspondence)
        return max_candidates is not None and len(correspondences) >= max_candidates

    def walk_from_first_neighbor(
        first_neighbor_index: int,
        first_neighbor_coord: tuple[int, int, int],
        reference_vector: np.ndarray,
    ) -> bool:
        first_delta = coord_delta(start_coord, first_neighbor_coord)
        if first_delta is None:
            return False
        first_basis = local_turn_basis_for_delta(first_delta)
        if first_basis is None:
            return False
        assignments: dict[int, PoseGraphNodeAssignment] = {
            start_index: start_assignment,
        }
        used_coords = {start_coord}
        first_path_faces = start_assignment.path_faces & cube_lattice_boundary_faces(
            first_neighbor_coord
        )
        assignments[first_neighbor_index] = PoseGraphNodeAssignment(
            coord=first_neighbor_coord,
            parent_index=start_index,
            local_to_global=first_basis,
            local_to_global_frames=unique_local_to_global_frames(
                first_delta,
                first_path_faces,
                first_basis,
            ),
            path_faces=first_path_faces,
            depth=1,
            has_turn=False,
        )
        used_coords.add(first_neighbor_coord)

        queue = deque([start_index, first_neighbor_index])
        queued_indexes = {start_index, first_neighbor_index}

        def enqueue_node(index: int) -> None:
            if index in queued_indexes:
                return
            queue.append(index)
            queued_indexes.add(index)

        while queue:
            current_index = queue.popleft()
            current_assignment = assignments[current_index]
            current_coord = current_assignment.coord
            current_xy = np.asarray(graph.points[current_index].xy, dtype=np.float32)
            if current_assignment.parent_index is None:
                current_local_to_global_frames = (first_basis,)
                previous_unit = reference_vector
            else:
                current_local_to_global_frames = current_assignment.local_to_global_frames
                if not current_local_to_global_frames:
                    continue
                parent_coord = assignments[current_assignment.parent_index].coord
                previous_delta = coord_delta(parent_coord, current_coord)
                if previous_delta is None:
                    continue
                parent_xy = np.asarray(
                    graph.points[current_assignment.parent_index].xy,
                    dtype=np.float32,
                )
                previous_unit = normalized_vector(current_xy - parent_xy)
                if previous_unit is None:
                    continue
            candidate_assignments = []
            for neighbor_index in adjacency[current_index]:
                if neighbor_index not in component_set:
                    continue
                if neighbor_index == current_assignment.parent_index:
                    continue
                if neighbor_index in assignments:
                    continue

                neighbor = graph.points[neighbor_index]
                observed_unit = normalized_vector(np.asarray(neighbor.xy, dtype=np.float32) - current_xy)
                if observed_unit is None:
                    continue
                turn_scores = local_turn_scores(previous_unit, observed_unit)
                for neighbor_coord in cube_lattice_neighbor_coords(
                    current_coord,
                    neighbor.class_name,
                    used_coords=used_coords,
                ):
                    assigned_neighbor_coords = [
                        assignments[assigned_neighbor_index].coord
                        for assigned_neighbor_index in adjacency[neighbor_index]
                        if assigned_neighbor_index in assignments
                    ]
                    conflict_count = sum(
                        not cube_lattice_coords_are_neighbors(neighbor_coord, assigned_neighbor_coord)
                        for assigned_neighbor_coord in assigned_neighbor_coords
                    )
                    if conflict_count > 0:
                        continue
                    next_delta = coord_delta(current_coord, neighbor_coord)
                    if next_delta is None:
                        continue
                    turn_options = []
                    allowed_turns = ("straight", "left", "right")
                    if current_assignment.parent_index is None:
                        allowed_turns = (*allowed_turns, "back")
                    seen_turn_options = set()
                    for current_local_to_global in current_local_to_global_frames:
                        for turn_name in allowed_turns:
                            if current_local_to_global.get(turn_name) != next_delta:
                                continue
                            next_path_faces = (
                                current_assignment.path_faces
                                & cube_lattice_boundary_faces(neighbor_coord)
                            )
                            fallback_frame = propagated_local_to_global_frame(
                                current_local_to_global,
                                turn_name,
                            )
                            next_local_to_global_frames = unique_local_to_global_frames(
                                next_delta,
                                next_path_faces,
                                fallback_frame,
                            )
                            option_key = (
                                turn_name,
                                tuple(tuple(sorted(frame.items())) for frame in next_local_to_global_frames),
                                next_path_faces,
                            )
                            if option_key in seen_turn_options:
                                continue
                            seen_turn_options.add(option_key)
                            turn_options.append(
                                (
                                    turn_name,
                                    next_local_to_global_frames[0],
                                    next_local_to_global_frames,
                                    next_path_faces,
                                    (3.0 if len(assigned_neighbor_coords) >= 2 else 0.0)
                                    + (1.0 if next_path_faces else 0.0),
                                )
                            )

                    if not turn_options and len(assigned_neighbor_coords) >= 2:
                        topology_turn = max(("straight", "left", "right"), key=turn_scores.get)
                        next_path_faces = (
                            current_assignment.path_faces
                            & cube_lattice_boundary_faces(neighbor_coord)
                        )
                        fallback_frame = (
                            local_turn_basis_for_delta(next_delta)
                            or propagated_local_to_global_frame(
                                current_local_to_global_frames[0],
                                topology_turn,
                            )
                        )
                        next_local_to_global_frames = unique_local_to_global_frames(
                            next_delta,
                            next_path_faces,
                            fallback_frame,
                        )
                        turn_options.append(
                            (
                                topology_turn,
                                next_local_to_global_frames[0],
                                next_local_to_global_frames,
                                next_path_faces,
                                3.0,
                            )
                        )

                    if not turn_options:
                        continue
                    (
                        turn_name,
                        next_local_to_global,
                        next_local_to_global_frames,
                        next_path_faces,
                        context_bonus,
                    ) = min(
                        turn_options,
                        key=lambda option: (
                            -turn_scores[option[0]] - option[4],
                            VECTOR_GUIDED_LOCAL_TURN_ORDER.index(option[0]),
                        ),
                    )
                    score = (
                        turn_scores[turn_name]
                        + context_bonus
                    )
                    candidate_assignments.append(
                        (
                            conflict_count,
                            -score,
                            -weighted_point_score(neighbor_index),
                            neighbor_index,
                            neighbor_coord,
                            next_local_to_global,
                            next_local_to_global_frames,
                            next_path_faces,
                            turn_name in ("left", "right"),
                        )
                    )

            candidate_assignments.sort()
            for (
                _conflict_count,
                _direction_score,
                _point_score,
                neighbor_index,
                neighbor_coord,
                next_local_to_global,
                next_local_to_global_frames,
                next_path_faces,
                has_turn,
            ) in candidate_assignments:
                if neighbor_index in assignments or neighbor_coord in used_coords:
                    continue
                node_has_turn = current_assignment.has_turn or bool(has_turn)
                assignments[neighbor_index] = PoseGraphNodeAssignment(
                    coord=neighbor_coord,
                    parent_index=current_index,
                    local_to_global=next_local_to_global,
                    local_to_global_frames=next_local_to_global_frames,
                    path_faces=next_path_faces,
                    depth=current_assignment.depth + 1,
                    has_turn=node_has_turn,
                )
                used_coords.add(neighbor_coord)
                if node_has_turn and len(assignments) >= 3:
                    if add_correspondence(assignments):
                        return True
                enqueue_node(neighbor_index)

        return add_correspondence(assignments)

    _score, _rank, first_neighbor_index, first_neighbor_coord, reference_vector = min(first_neighbor_options)
    first_neighbor = graph.points[first_neighbor_index]
    first_neighbor_coord_options = [first_neighbor_coord]
    mirrored_coord = mirrored_first_neighbor_lattice_coord(
        start.class_name,
        first_neighbor.class_name,
        first_neighbor_coord,
    )
    if mirrored_coord is not None:
        first_neighbor_coord_options.append(mirrored_coord)
    for coord_option in first_neighbor_coord_options:
        if walk_from_first_neighbor(first_neighbor_index, coord_option, reference_vector):
            return tuple(correspondences)
    return tuple(correspondences)


def vector_guided_pose_graph_correspondence_candidates(
    graph: PoseGraph,
    *,
    max_candidates: int | None = None,
    point_weights: dict[int, float] | None = None,
    exclude_keys: set[tuple[tuple[int, ...], tuple[tuple[int, int, int], ...]]] | None = None,
) -> tuple[PoseGraphCorrespondence, ...]:
    if max_candidates is not None:
        max_candidates = max(0, int(max_candidates))
        if max_candidates == 0:
            return tuple()

    adjacency = pose_graph_adjacency(graph)
    components = pose_graph_components(graph, adjacency)
    eligible_components = [
        component
        for component in components
        if len(component) >= 3
        and pose_graph_component_has_observed_turn(graph, adjacency, component)
    ]
    if not eligible_components:
        return tuple()

    start_options = [
        (component, index)
        for component in eligible_components
        for index in component
        if graph.points[index].class_name in VECTOR_GUIDED_START_COORDS
    ]
    if not start_options:
        return tuple()
    if point_weights is None:
        random.shuffle(start_options)
    else:
        start_options = weighted_shuffle_start_options(start_options, point_weights)

    correspondences: list[PoseGraphCorrespondence] = []
    correspondence_keys: set[tuple[tuple[int, ...], tuple[tuple[int, int, int], ...]]] = (
        set() if exclude_keys is None else set(exclude_keys)
    )
    for component, start_index in start_options:
        remaining = (
            None
            if max_candidates is None
            else max(0, max_candidates - len(correspondences))
        )
        if remaining == 0:
            break
        for correspondence in _vector_guided_pose_graph_correspondence_candidates_from_start(
            graph,
            adjacency,
            component,
            start_index,
            max_candidates=remaining,
            point_weights=point_weights,
            exclude_keys=correspondence_keys,
        ):
            key = correspondence.key
            if key in correspondence_keys:
                continue
            correspondence_keys.add(key)
            correspondences.append(correspondence)
            if max_candidates is not None and len(correspondences) >= max_candidates:
                return tuple(correspondences)
    return tuple(correspondences)


def vector_guided_pose_graph_correspondence(graph: PoseGraph) -> PoseGraphCorrespondence | None:
    correspondences = vector_guided_pose_graph_correspondence_candidates(graph, max_candidates=1)
    return None if not correspondences else correspondences[0]


def draw_pose_graph_adjacencies(
    frame: np.ndarray,
    graph: PoseGraph,
    args,
    *,
    roi_side: float | None = None,
) -> int:
    if len(graph.points) < 2:
        return 0

    accepted_count = 0
    roi_side = (
        ADJACENCY_REFERENCE_ROI_SIZE_PX
        if roi_side is None
        else float(roi_side)
    )
    strip_width_px = adjacency_edge_strip_width_px(args, roi_side=roi_side)
    if strip_width_px > 1:
        strip_overlay = frame.copy()
        for adjacency in graph.edges:
            start = tuple(np.round(graph.points[adjacency.start_index].xy).astype(int))
            end = tuple(np.round(graph.points[adjacency.end_index].xy).astype(int))
            cv2.line(
                strip_overlay,
                start,
                end,
                POSE_GRAPH_EDGE_STRIP_COLOR,
                strip_width_px,
                lineType=cv2.LINE_AA,
            )
        cv2.addWeighted(
            strip_overlay,
            float(POSE_GRAPH_EDGE_STRIP_ALPHA),
            frame,
            1.0 - float(POSE_GRAPH_EDGE_STRIP_ALPHA),
            0.0,
            dst=frame,
        )

    for adjacency in graph.edges:
        if not adjacency.accepted:
            continue
        start = tuple(np.round(graph.points[adjacency.start_index].xy).astype(int))
        end = tuple(np.round(graph.points[adjacency.end_index].xy).astype(int))
        cv2.line(frame, start, end, (0, 0, 0), 5, lineType=cv2.LINE_AA)
        cv2.line(frame, start, end, (60, 255, 120), 2, lineType=cv2.LINE_AA)
        if getattr(args, "debug_overlay", False):
            midpoint = tuple(
                np.round(
                    (
                        np.asarray(graph.points[adjacency.start_index].xy, dtype=np.float32)
                        + np.asarray(graph.points[adjacency.end_index].xy, dtype=np.float32)
                    )
                    * 0.5
                ).astype(int)
            )
            cv2.putText(
                frame,
                f"{adjacency.coverage:.0%}",
                midpoint,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        accepted_count += 1
    return accepted_count


def draw_pose_graph_correspondence(
    frame: np.ndarray,
    graph: PoseGraph,
    correspondence: PoseGraphCorrespondence | None,
) -> None:
    if correspondence is None:
        return

    used_indexes = set(correspondence.graph_indexes)
    for edge in graph.edges:
        if not edge.accepted:
            continue
        if edge.start_index not in used_indexes or edge.end_index not in used_indexes:
            continue
        start = tuple(np.round(graph.points[edge.start_index].xy).astype(int))
        end = tuple(np.round(graph.points[edge.end_index].xy).astype(int))
        cv2.line(frame, start, end, (0, 0, 0), 7, lineType=cv2.LINE_AA)
        cv2.line(frame, start, end, POSE_GRAPH_CORRESPONDENCE_LINE_COLOR, 3, lineType=cv2.LINE_AA)

    for graph_index, lattice_coord in zip(correspondence.graph_indexes, correspondence.lattice_coords):
        point = np.asarray(graph.points[graph_index].xy, dtype=np.float32)
        label_at = tuple(np.round(point + np.asarray((8.0, -8.0), dtype=np.float32)).astype(int))
        label = f"{lattice_coord}"
        cv2.putText(
            frame,
            label,
            label_at,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            label,
            label_at,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            POSE_GRAPH_CORRESPONDENCE_TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )


def draw_pose_estimate(
    frame: np.ndarray,
    roi: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
    pose_model,
    pose_solver: PoseGraphPnPSolver | None,
    args,
) -> tuple[int, PosePnPResult | None, tuple[int, int, int, int] | None]:
    if pose_model is None:
        return 0, None, None

    profile = getattr(args, "_pose_profile", None)
    x1, y1, x2, y2 = roi_xyxy
    model_roi = resize_to_square_size(roi, POSE_IMAGE_SIZE)
    started = perf_counter()
    results = pose_model.predict(
        source=model_roi,
        conf=getattr(
            args,
            "pose_conf",
            min(DEFAULT_INDEPENDENT_POSE_CLASS_CONFIDENCES.values()),
        ),
        imgsz=POSE_IMAGE_SIZE,
        device=YOLO_DEVICE,
        verbose=False,
    )
    if profile is not None:
        profile["pose_model_ms"] = (perf_counter() - started) * 1000.0
    if not results:
        reset_pose_solver_state(pose_solver)
        return 0, None, None

    started = perf_counter()
    result = results[0]
    xy, conf = result_keypoints(result, POSE_IMAGE_SIZE)
    box_conf = result_box_confidences(result)
    class_confidences = getattr(
        args,
        "pose_class_conf",
        DEFAULT_INDEPENDENT_POSE_CLASS_CONFIDENCES,
    )
    scale_x = (x2 - x1) / float(model_roi.shape[1])
    scale_y = (y2 - y1) / float(model_roi.shape[0])
    pose_points = collect_pose_graph_points(
        result,
        pose_model,
        xy,
        conf,
        confidence_threshold=POSE_KEYPOINT_CONFIDENCE,
        class_confidences=class_confidences,
        box_confidence=box_conf,
        offset_xy=(x1, y1),
        scale_xy=(scale_x, scale_y),
    )
    if profile is not None:
        profile["collect_points_ms"] = (perf_counter() - started) * 1000.0
        profile["pose_points"] = len(pose_points)
    should_build_pose_graph = (
        getattr(args, "show_corner_adjacency", DRAW_CORNER_ADJACENCY)
        or ENABLE_POSE_GRAPH_CORRESPONDENCE
    )
    started = perf_counter()
    pose_graph = (
        build_pose_graph(pose_points, roi, roi_xyxy, args)
        if should_build_pose_graph
        else PoseGraph(points=tuple(pose_points), edges=tuple())
    )
    if profile is not None:
        profile["graph_total_ms"] = (perf_counter() - started) * 1000.0
    started = perf_counter()
    if getattr(args, "show_corner_adjacency", DRAW_CORNER_ADJACENCY):
        draw_pose_graph_adjacencies(frame, pose_graph, args, roi_side=roi_side_px(roi_xyxy))
    if profile is not None:
        profile["draw_adjacency_ms"] = (perf_counter() - started) * 1000.0

    point_count = 0
    started = perf_counter()
    if DRAW_POSE_KEYPOINTS and (
        ENABLE_POSE_GRAPH_CORRESPONDENCE or not ENABLE_POSE_ESTIMATION
    ):
        point_count = draw_pose_keypoints(
            frame,
            result,
            pose_model,
            xy,
            conf,
            confidence_threshold=POSE_KEYPOINT_CONFIDENCE,
            class_confidences=class_confidences,
            box_confidence=box_conf,
            offset_xy=(x1, y1),
            scale_xy=(scale_x, scale_y),
        )
    if profile is not None:
        profile["draw_keypoints_ms"] = (perf_counter() - started) * 1000.0

    graph_image_points = np.asarray([point.xy for point in pose_graph.points], dtype=np.float32)
    pose_tracking_bounds_xyxy = point_bounds_xyxy(graph_image_points, roi_xyxy)
    pnp_result = None
    if ENABLE_POSE_ESTIMATION and ENABLE_POSE_GRAPH_CORRESPONDENCE and pose_solver is not None:
        started = perf_counter()
        max_correspondence_candidates = max(
            0,
            int(getattr(pose_solver, "max_refinement_attempts", PNP_MAX_REFINEMENT_ATTEMPTS)),
        )
        correspondences: list[PoseGraphCorrespondence] = []
        correspondence = None
        tried_correspondence_keys: set[tuple[tuple[int, ...], tuple[tuple[int, int, int], ...]]] = set()
        point_weights = {index: 1.0 for index in range(len(pose_graph.points))}
        if profile is not None:
            profile["correspondence_generation_ms"] = 0.0
        refinement_budget = {"remaining": max_correspondence_candidates}
        solve_started = perf_counter()
        for _attempt_index in range(max_correspondence_candidates):
            generation_started = perf_counter()
            next_correspondences = vector_guided_pose_graph_correspondence_candidates(
                pose_graph,
                max_candidates=1,
                point_weights=point_weights,
                exclude_keys=tried_correspondence_keys,
            )
            if profile is not None:
                profile["correspondence_generation_ms"] = (
                    float(profile.get("correspondence_generation_ms", 0.0))
                    + (perf_counter() - generation_started) * 1000.0
                )
            if not next_correspondences:
                break
            correspondence = next_correspondences[0]
            correspondences.append(correspondence)
            tried_correspondence_keys.add(correspondence.key)
            if profile is not None:
                profile["seed_attempts"] = int(profile.get("seed_attempts", 0)) + 1
            try:
                pnp_result = pose_solver.solve_correspondences(
                    correspondence,
                    (frame.shape[1], frame.shape[0]),
                    observed_graph=pose_graph,
                    profile=profile,
                    refinement_budget=refinement_budget,
                    clear_rejected=False,
                )
            except TypeError:
                pnp_result = pose_solver.solve_correspondences(
                    correspondence,
                    (frame.shape[1], frame.shape[0]),
                    observed_graph=pose_graph,
                )
            if pnp_result is not None:
                break
            for graph_index in correspondence.graph_indexes:
                point_weights[graph_index] = float(point_weights.get(graph_index, 1.0)) * 0.5
        if pnp_result is None:
            clear_motion_state = getattr(pose_solver, "clear_motion_state", None)
            if callable(clear_motion_state):
                clear_motion_state()
        if profile is not None:
            profile["correspondence_ms"] = (perf_counter() - started) * 1000.0
            profile["solve_ms"] = (perf_counter() - solve_started) * 1000.0
            profile["correspondence_points"] = (
                0 if correspondence is None else len(correspondence.graph_indexes)
            )
            profile["correspondence_candidates"] = len(correspondences)
        if profile is not None:
            profile["pnp_ms"] = (perf_counter() - started) * 1000.0
            profile["pnp_points"] = 0 if pnp_result is None else len(pnp_result.object_points)
        started = perf_counter()
        if DRAW_POSE_GRAPH_COORDS:
            draw_pose_graph_correspondence(frame, pose_graph, correspondence)
        if DRAW_PNP_CUBE:
            rejected_pnp_result = (
                getattr(pose_solver, "rejected_pose_result", None)
                if pnp_result is None
                else None
            )
            draw_pnp_cube(
                frame,
                pnp_result or rejected_pnp_result,
                alpha=0.5 if pnp_result is None else 1.0,
                observed_graph=pose_graph,
                profile=profile,
            )
        if profile is not None:
            profile["draw_pose_ms"] = (perf_counter() - started) * 1000.0
    return point_count, pnp_result, pose_tracking_bounds_xyxy


def reset_pose_solver_state(pose_solver: PoseGraphPnPSolver | None) -> None:
    reset = getattr(pose_solver, "reset_pose_state", None)
    if callable(reset):
        reset()


def pose_motion_filter_from_args(args) -> PoseMotionFilter | None:
    if not getattr(args, "pose_motion_filter", POSE_MOTION_FILTER_BY_DEFAULT):
        return None
    return PoseMotionFilter(
        PoseMotionFilterConfig(
            dt_seconds=getattr(args, "pose_motion_dt", POSE_MOTION_DT_SECONDS),
            max_acceleration_units_s2=getattr(
                args,
                "pose_motion_max_acceleration",
                POSE_MOTION_MAX_ACCELERATION_UNITS_S2,
            ),
            max_jerk_units_s3=getattr(
                args,
                "pose_motion_max_jerk",
                POSE_MOTION_MAX_JERK_UNITS_S3,
            ),
            max_angular_acceleration_rad_s2=getattr(
                args,
                "pose_motion_max_angular_acceleration",
                POSE_MOTION_MAX_ANGULAR_ACCELERATION_RAD_S2,
            ),
            max_angular_jerk_rad_s3=getattr(
                args,
                "pose_motion_max_angular_jerk",
                POSE_MOTION_MAX_ANGULAR_JERK_RAD_S3,
            ),
        )
    )


def pose_solver_from_args(args) -> PoseGraphPnPSolver:
    return PoseGraphPnPSolver(
        focal_min=PNP_FOCAL_SCALE_MIN,
        focal_max=PNP_FOCAL_SCALE_MAX,
        focal_steps=PNP_FOCAL_SCALE_STEPS,
        motion_filter=pose_motion_filter_from_args(args),
    )


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
        device=YOLO_DEVICE,
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


def pose_profile_summary(profile: dict[str, float | int]) -> str:
    timing_parts = [
        f"model {float(profile.get('pose_model_ms', 0.0)):.1f}",
        f"graph {float(profile.get('graph_total_ms', 0.0)):.1f}",
        f"walk {float(profile.get('correspondence_ms', 0.0)):.1f}",
        f"pnp {float(profile.get('pnp_ms', 0.0)):.1f}",
        f"draw {float(profile.get('draw_pose_ms', 0.0)):.1f}",
    ]
    count_parts = [
        f"pts {int(profile.get('pose_points', 0))}",
        f"corr {int(profile.get('correspondence_points', 0))}",
        f"edges {int(profile.get('accepted_edges', 0))}/{int(profile.get('candidate_edges', 0))}",
        f"p1 {int(profile.get('seed_attempts', 0))}",
        f"p2 {int(profile.get('refinement_attempts', 0))}",
        f"pnp_calls {int(profile.get('pnp_solve_calls', 0))}",
    ]
    return "pose ms " + " ".join(timing_parts) + " | " + " ".join(count_parts)


def pose_attempt_summary(profile: dict[str, float | int] | None) -> str:
    profile = {} if profile is None else profile
    return (
        f"p1 {int(profile.get('seed_attempts', 0))} "
        f"p2 {int(profile.get('refinement_attempts', 0))}"
    )


def print_pose_validity_summary(valid_pose_frames: int, invalid_pose_frames: int) -> None:
    keypoint_frames = int(valid_pose_frames) + int(invalid_pose_frames)
    print(
        "pose summary: "
        f"{valid_pose_frames} valid, {invalid_pose_frames} invalid "
        f"out of {keypoint_frames} frames with keypoint detections",
        flush=True,
    )


def source_fps(capture: cv2.VideoCapture, fallback: float = 30.0) -> float:
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 1e-3:
        return float(fallback)
    return fps


def create_video_writer(path: Path, frame_shape: tuple[int, ...], fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame_shape[:2]
    suffix = path.suffix.lower()
    fourcc_name = "mp4v" if suffix in {".mp4", ".m4v", ".mov"} else "MJPG"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*fourcc_name),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    return writer


def process_frame(
    frame: np.ndarray,
    bbox_detector,
    sticker_model,
    args,
    color_classifier: ColorDensityClassifier | None = None,
    roi_tracker: RoiTracker | None = None,
    pose_model=None,
    pose_solver: PoseGraphPnPSolver | None = None,
    return_pose_stats: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, FramePoseStats]:
    output = frame.copy()
    roi_xyxy, candidate, roi_source = roi_for_frame(frame, bbox_detector, args, roi_tracker)
    if roi_xyxy is None:
        bgr_samples = np.empty((0, 3), dtype=np.uint8)
        if return_pose_stats:
            return output, bgr_samples, FramePoseStats(keypoint_count=0, valid_pose=False)
        return output, bgr_samples

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
    pose_profile = {}
    setattr(args, "_pose_profile", pose_profile)
    if pose_model is not None:
        if ENABLE_POSE_ESTIMATION and pose_solver is None:
            pose_solver = pose_solver_from_args(args)
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
        pnp_label = (
            ""
            if pnp_result is None
            else f" pnp {len(pnp_result.object_points)}pts {pnp_result.error:.1f}px"
        )
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
        if pose_profile is not None:
            cv2.putText(
                output,
                pose_profile_summary(pose_profile),
                (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                output,
                pose_profile_summary(pose_profile),
                (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    if getattr(args, "profile_pose", False) and pose_profile is not None:
        print(pose_profile_summary(pose_profile), flush=True)
    if return_pose_stats:
        return (
            output,
            bgr_samples,
            FramePoseStats(
                keypoint_count=pose_keypoint_count,
                valid_pose=pnp_result is not None,
            ),
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
    parser.add_argument(
        "--pose-conf",
        type=float,
        default=min(DEFAULT_INDEPENDENT_POSE_CLASS_CONFIDENCES.values()),
        help="Global independent-pose model pre-filter; keep at or below the lowest class threshold.",
    )
    parser.add_argument(
        "--pose-class-conf",
        type=parse_independent_pose_class_confidences,
        default=dict(DEFAULT_INDEPENDENT_POSE_CLASS_CONFIDENCES),
        help=(
            "Per-class independent-pose boxes.conf thresholds as class=value pairs, for example "
            "corner=0.25,corner-edge=0.30,corner-edge-center=0.25."
        ),
    )
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
    parser.add_argument(
        "--save-video",
        type=Path,
        help="Output path for processed video frames.",
    )
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
        "--profile-pose",
        action="store_true",
        help="Print per-frame pose timing buckets for model, graph, correspondence, PnP, and drawing.",
    )
    parser.add_argument(
        "--pose-motion-filter",
        action="store_true",
        help="Filter accepted rigid pose with a stationary/acceleration/jerk prior.",
    )
    parser.add_argument(
        "--pose-motion-dt",
        type=float,
        default=POSE_MOTION_DT_SECONDS,
        help="Frame interval in seconds used by the pose motion filter.",
    )
    parser.add_argument(
        "--pose-motion-max-acceleration",
        type=float,
        default=POSE_MOTION_MAX_ACCELERATION_UNITS_S2,
        help="Maximum translation acceleration in PnP translation units per second squared.",
    )
    parser.add_argument(
        "--pose-motion-max-jerk",
        type=float,
        default=POSE_MOTION_MAX_JERK_UNITS_S3,
        help="Maximum translation jerk in PnP translation units per second cubed.",
    )
    parser.add_argument(
        "--pose-motion-max-angular-acceleration",
        type=float,
        default=POSE_MOTION_MAX_ANGULAR_ACCELERATION_RAD_S2,
        help="Maximum angular acceleration in radians per second squared.",
    )
    parser.add_argument(
        "--pose-motion-max-angular-jerk",
        type=float,
        default=POSE_MOTION_MAX_ANGULAR_JERK_RAD_S3,
        help="Maximum angular jerk in radians per second cubed.",
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
    parser.add_argument(
        "--hide-corner-adjacency",
        dest="show_corner_adjacency",
        action="store_false",
        help="Do not draw Canny/Hough-derived adjacency lines between detected corner keypoints",
    )
    parser.add_argument(
        "--adjacency-canny-blur",
        type=int,
        default=ADJACENCY_CANNY_BLUR_KERNEL,
        help="Gaussian blur kernel before Canny, in pixels at the reference ROI size",
    )
    parser.add_argument(
        "--adjacency-canny-low",
        type=int,
        default=ADJACENCY_CANNY_LOW,
        help="Lower Canny threshold for corner adjacency evidence",
    )
    parser.add_argument(
        "--adjacency-canny-high",
        type=int,
        default=ADJACENCY_CANNY_HIGH,
        help="Upper Canny threshold for corner adjacency evidence",
    )
    parser.add_argument(
        "--adjacency-min-edge-length",
        type=float,
        default=ADJACENCY_MIN_EDGE_LENGTH_PX,
        help="Ignore candidate pose graph connections shorter than this many pixels at the reference ROI size",
    )
    parser.add_argument(
        "--adjacency-min-coverage",
        type=float,
        default=ADJACENCY_MIN_COVERAGE,
        help="Minimum covered fraction of the corner-pair distance required for adjacency",
    )
    parser.add_argument(
        "--adjacency-max-uncovered-gap-ratio",
        type=float,
        default=ADJACENCY_MAX_UNCOVERED_GAP_RATIO,
        help="Maximum uncovered gap as a fraction of the corner-pair distance",
    )
    parser.add_argument(
        "--adjacency-edge-dilation",
        type=int,
        default=ADJACENCY_EDGE_DILATION_PX,
        help="Half-width in pixels at the reference ROI size around candidate pose graph connections checked for Canny edge support",
    )
    parser.add_argument(
        "--adjacency-keypoint-intersection-distance",
        type=float,
        default=ADJACENCY_KEYPOINT_INTERSECTION_DISTANCE_PX,
        help="Reject candidate pose graph connections that pass within this many reference-ROI pixels of another keypoint",
    )
    parser.add_argument(
        "--adjacency-reference-roi-size",
        type=float,
        default=ADJACENCY_REFERENCE_ROI_SIZE_PX,
        help="ROI side length represented by the adjacency pixel defaults",
    )
    parser.set_defaults(
        show_bbox=True,
        show_confidence=True,
        show_sticker_color_classification=True,
        display=DISPLAY_PREVIEW_BY_DEFAULT,
        show_original_underlay=SHOW_ORIGINAL_UNDERLAY_BY_DEFAULT,
        show_corner_adjacency=DRAW_CORNER_ADJACENCY,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0.0 <= args.pose_conf <= 1.0:
        raise SystemExit("--pose-conf must be in [0, 1]")
    if args.pose_conf > min(args.pose_class_conf.values()):
        raise SystemExit("--pose-conf cannot exceed the lowest --pose-class-conf threshold")

    from ultralytics import YOLO

    bbox_detector = YOLOCubeDetector(
        model_name=str(args.bbox_model),
        confidence_threshold=args.bbox_conf,
        image_size=args.bbox_imgsz,
        device=YOLO_DEVICE,
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
    valid_pose_frames = 0
    invalid_pose_frames = 0
    roi_tracker = RoiTracker()
    pose_solver = (
        pose_solver_from_args(args)
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
        output, bgr_samples, pose_stats = process_frame(
            frame,
            bbox_detector,
            sticker_model,
            args,
            color_classifier,
            roi_tracker,
            pose_model,
            pose_solver,
            return_pose_stats=True,
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
        print_pose_validity_summary(
            1 if pose_stats.keypoint_count > 0 and pose_stats.valid_pose else 0,
            1 if pose_stats.keypoint_count > 0 and not pose_stats.valid_pose else 0,
        )
        return 0

    capture = cv2.VideoCapture(source)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    previous = perf_counter()
    video_writer = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            output, bgr_samples, pose_stats = process_frame(
                frame,
                bbox_detector,
                sticker_model,
                args,
                color_classifier,
                roi_tracker,
                pose_model,
                pose_solver,
                return_pose_stats=True,
            )
            if pose_stats.keypoint_count > 0:
                if pose_stats.valid_pose:
                    valid_pose_frames += 1
                else:
                    invalid_pose_frames += 1
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

            if args.save_video:
                if video_writer is None:
                    video_writer = create_video_writer(
                        args.save_video,
                        output.shape,
                        source_fps(capture),
                    )
                video_writer.write(output)
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
    finally:
        if video_writer is not None:
            video_writer.release()

    if args.display:
        cv2.destroyAllWindows()
    capture.release()
    print_pose_validity_summary(valid_pose_frames, invalid_pose_frames)
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
