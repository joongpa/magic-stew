#!/usr/bin/env python3
"""Render the independent cube-keypoint YOLO model with the live ROI flow.

For video sources, this renderer uses a producer/consumer pipeline:

1. The foreground pass runs the detector and neural models, displays their
   output, and caches the raw frame/ROI/model data.
2. A background worker consumes those cached records for cheap pose-candidate
   generation, keeps the full candidate set, and samples colors while the
   neural producer continues.
3. Pose recovery refines the cached pose candidates and, when requested,
   applies geometric layer-turn fitting; the final pass replays the recovered
   cache for presentation without running neural inference or pose refinement.

Within the foreground pass, the neural work is:

1. Run the cube bbox detector when no tracked ROI exists.
2. Build the padded square ROI using the shared live-pipeline helper.
3. Run the combined cube/sticker segmentation and independent three-class pose models inside that ROI.
4. Map the independent points and sticker masks back to the full frame.
5. Update the ROI tracker from the predicted point extent.

Examples:

    .venv/bin/python scripts/render_yolo11n_independent_keypoints.py \
        --source data/independent_keypoints_dataset_v2/images/000000.png \
        --model runs/pose/runs/independent_keypoint_pose_v2/runs/cube_piece_keypoint_yolo11n_pose/weights/best.pt \
        --save runs/independent_keypoints_render.png \
        --no-display

    .venv/bin/python scripts/render_yolo11n_independent_keypoints.py \
        --source 0 \
        --model runs/pose/runs/independent_keypoint_pose_v2/runs/cube_piece_keypoint_yolo11n_pose/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import combinations, permutations, product
from pathlib import Path
from queue import SimpleQueue
from threading import Lock, Thread
from time import perf_counter

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from live_recognition.detectors import YOLOCubeDetector  # noqa: E402
from live_recognition.detectors import default_yolo_device  # noqa: E402
from live_recognition.independent_pose_thresholds import (  # noqa: E402
    DEFAULT_INDEPENDENT_POSE_CLASS_CONFIDENCES, INDEPENDENT_POSE_CLASS_NAMES,
    independent_pose_class_threshold, parse_independent_pose_class_confidences)
from live_recognition.pose_motion import PoseMotionDiagnostics  # noqa: E402
from live_recognition.pose_motion import PoseMotionFilter  # noqa: E402
from live_recognition.pose_motion import PoseMotionFilterConfig  # noqa: E402
from live_recognition.roi_tracking import RoiTracker  # noqa: E402
from live_recognition.roi_tracking import \
    roi_for_frame as shared_roi_for_frame  # noqa: E402
from live_recognition.visualization import _reliable_color_mask  # noqa: E402

CLASS_NAMES = INDEPENDENT_POSE_CLASS_NAMES
# OpenCV uses BGR colors. These match the independent-keypoint generator.
CLASS_COLORS_BGR = (
    (255, 255, 255),
    (0, 255, 255),
    (255, 120, 255),
)
OCCLUDED_COLOR_BGR = (0, 0, 255)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
QUICKTIME_VIDEO_EXTENSIONS = frozenset({".mp4", ".m4v", ".mov"})
WINDOW_NAME = "independent cube keypoints"
CPU_NEURAL_EXECUTOR = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="cpu-neural",
)
DEFAULT_BBOX_MODEL = ROOT / "runs/500ExamplesWeights.pt"
DEFAULT_POSE_MODEL = (
    ROOT
    / "runs/pose/runs/keypoints-results-20k/weights/best.pt"
)
DEFAULT_COMBINED_SEGMENTATION_MODEL_CANDIDATES = (
    ROOT / "runs/segment/runs/combinedSegmentWeights.pt",
)
DEFAULT_COMBINED_SEGMENTATION_MODEL = next(
    (
        path
        for path in DEFAULT_COMBINED_SEGMENTATION_MODEL_CANDIDATES
        if path.exists()
    ),
    DEFAULT_COMBINED_SEGMENTATION_MODEL_CANDIDATES[0],
)
STICKER_COLORS_BGR = {
    "white": (255, 255, 255),
    "yellow": (0, 255, 255),
    "red": (0, 0, 255),
    "orange": (0, 165, 255),
    "blue": (255, 0, 0),
    "green": (0, 180, 0),
    "sticker": (0, 220, 255),
}
STANDARD_FACE_COLOR_NAMES = {
    "U": "white",
    "D": "yellow",
    "F": "green",
    "B": "blue",
    "R": "red",
    "L": "orange",
}
PURE_COLOR_RGB = {
    "white": (255, 255, 255),
    "yellow": (255, 255, 0),
    "red": (255, 0, 0),
    "orange": (255, 165, 0),
    "blue": (0, 0, 255),
    "green": (0, 255, 0),
}
OPPOSITE_COLOR_PAIRS = frozenset(
    frozenset(pair)
    for pair in (("white", "yellow"), ("red", "orange"), ("blue", "green"))
)
FACE_VECTOR_BY_NAME = {
    "L": np.asarray((-1, 0, 0), dtype=np.int32),
    "R": np.asarray((1, 0, 0), dtype=np.int32),
    "F": np.asarray((0, -1, 0), dtype=np.int32),
    "B": np.asarray((0, 1, 0), dtype=np.int32),
    "D": np.asarray((0, 0, -1), dtype=np.int32),
    "U": np.asarray((0, 0, 1), dtype=np.int32),
}
STICKER_BORDER_COLOR_BGR = (0, 0, 255)
STICKER_TYPE_COLORS_BGR = {
    "corner": (0, 140, 255),
    "edge": (0, 200, 0),
    "center": (255, 0, 0),
    "unknown": (128, 128, 128),
}
STICKER_TYPE_CLASS_IDS = {
    "corner": (0, 1, 2, 1),
    "edge": (1, 1, 2, 2),
    "center": (2, 2, 2, 2),
}
STICKER_TYPE_LATTICE_COORDS = {
    # Canonical near/front face (z=0), ordered clockwise in the image for the
    # default camera convention. The first point is the upper-left corner.
    "corner": ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)),
    "edge": ((1, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0)),
    "center": ((1, 1, 0), (2, 1, 0), (2, 2, 0), (1, 2, 0)),
}
CUBE_SIZE_MM = 55.0
CUBE_HALF_SIZE_MM = CUBE_SIZE_MM / 2.0
CUBE_LAYER_SIZE_MM = CUBE_SIZE_MM / 3.0
CUBE_GRID_MAX_INDEX = 3
CUBE_LATTICE_COORDS = tuple(
    (x, y, z)
    for x, y, z in product(
        range(CUBE_GRID_MAX_INDEX + 1),
        range(CUBE_GRID_MAX_INDEX + 1),
        range(CUBE_GRID_MAX_INDEX + 1),
    )
    if x in (0, CUBE_GRID_MAX_INDEX)
    or y in (0, CUBE_GRID_MAX_INDEX)
    or z in (0, CUBE_GRID_MAX_INDEX)
)
CUBE_LATTICE_OBJECT_POINTS_MM = np.asarray(
    [
        tuple((value - 1.5) * CUBE_LAYER_SIZE_MM for value in coord)
        for coord in CUBE_LATTICE_COORDS
    ],
    dtype=np.float32,
)
CUBE_LATTICE_CLASS_IDS = np.asarray(
    [
        0 if sum(value in (0, CUBE_GRID_MAX_INDEX) for value in coord) == 3
        else 1 if sum(value in (0, CUBE_GRID_MAX_INDEX) for value in coord) == 2
        else 2
        for coord in CUBE_LATTICE_COORDS
    ],
    dtype=np.int32,
)
FACE_NAME_BY_AXIS_SIDE = {
    (0, -1): "L",
    (0, 1): "R",
    (1, -1): "F",
    (1, 1): "B",
    (2, -1): "D",
    (2, 1): "U",
}
FACE_NAME_ORDER = {name: index for index, name in enumerate(("U", "D", "F", "B", "R", "L"))}
CENTER_FACE_ORDER = ("U", "D", "F", "B", "R", "L")
EDGE_CUBIE_ORDER = (
    "UF", "UR", "UB", "UL",
    "DF", "DR", "DB", "DL",
    "FR", "FL", "BR", "BL",
)
CORNER_CUBIE_ORDER = (
    "UFR", "UFL", "UBL", "UBR",
    "DFR", "DFL", "DBL", "DBR",
)
CUBE_LATTICE_BOUNDARY_NORMALS = np.zeros(
    (len(CUBE_LATTICE_COORDS), 3, 3),
    dtype=np.float64,
)
CUBE_LATTICE_BOUNDARY_FACE_MASK = np.zeros(
    (len(CUBE_LATTICE_COORDS), 3),
    dtype=bool,
)
for _lattice_index, _lattice_coord in enumerate(CUBE_LATTICE_COORDS):
    for _face_index, (_axis, _value) in enumerate(
        (
            (axis, value)
            for axis, value in enumerate(_lattice_coord)
            if value in (0, CUBE_GRID_MAX_INDEX)
        )
    ):
        CUBE_LATTICE_BOUNDARY_NORMALS[_lattice_index, _face_index, _axis] = (
            -1.0 if _value == 0 else 1.0
        )
        CUBE_LATTICE_BOUNDARY_FACE_MASK[_lattice_index, _face_index] = True
CUBE_CORNER_POINTS_MM = np.asarray(
    [
        (-CUBE_HALF_SIZE_MM, -CUBE_HALF_SIZE_MM, CUBE_HALF_SIZE_MM),
        (CUBE_HALF_SIZE_MM, -CUBE_HALF_SIZE_MM, CUBE_HALF_SIZE_MM),
        (CUBE_HALF_SIZE_MM, CUBE_HALF_SIZE_MM, CUBE_HALF_SIZE_MM),
        (-CUBE_HALF_SIZE_MM, CUBE_HALF_SIZE_MM, CUBE_HALF_SIZE_MM),
        (-CUBE_HALF_SIZE_MM, -CUBE_HALF_SIZE_MM, -CUBE_HALF_SIZE_MM),
        (CUBE_HALF_SIZE_MM, -CUBE_HALF_SIZE_MM, -CUBE_HALF_SIZE_MM),
        (CUBE_HALF_SIZE_MM, CUBE_HALF_SIZE_MM, -CUBE_HALF_SIZE_MM),
        (-CUBE_HALF_SIZE_MM, CUBE_HALF_SIZE_MM, -CUBE_HALF_SIZE_MM),
    ],
    dtype=np.float32,
)


def _cube_symmetry_matrices() -> tuple[np.ndarray, ...]:
    """Return the 24 proper rotational symmetries of a cube."""

    matrices: list[np.ndarray] = []
    for permutation in permutations(range(3)):
        permutation_matrix = np.zeros((3, 3), dtype=np.int32)
        for row, column in enumerate(permutation):
            permutation_matrix[row, column] = 1
        for signs in product((-1, 1), repeat=3):
            matrix = permutation_matrix * np.asarray(signs, dtype=np.int32)[:, None]
            if int(round(np.linalg.det(matrix))) == 1:
                matrices.append(matrix.astype(np.float64))
    identity = np.eye(3, dtype=np.float64)
    matrices.sort(
        key=lambda matrix: (
            0 if np.array_equal(matrix, identity) else 1,
            tuple(matrix.reshape(-1).tolist()),
        )
    )
    return tuple(matrices)


CUBE_SYMMETRY_MATRICES = _cube_symmetry_matrices()
PNP_CUBE_COLOR_BGR = (255, 80, 30)
PNP_CANDIDATE_CUBE_COLOR_BGR = (0, 165, 255)
PNP_STICKER_COLOR_BGR = (0, 220, 255)
PNP_FINAL_SOLVE_COLOR_BGR = (0, 255, 0)
PNP_ASSIGNMENT_COLOR_BGR = (255, 0, 255)
PNP_ORIENTATION_ANCHOR_COLOR_BGR = (0, 0, 255)
PNP_TURN_MOVING_COLOR_BGR = (0, 0, 255)
PNP_TURN_STATIC_COLOR_BGR = (0, 0, 0)
CLASSIFIED_CUBE_DIAGNOSTIC_DEPTH_MM = 300.0
# OpenCV uses BGR.  These are the conventional X=red, Y=green, Z=blue
# colors used by the orientation gizmo.
ORIENTATION_AXIS_COLORS_BGR = (
    (0, 0, 255),
    (0, 200, 0),
    (255, 0, 0),
)
ORIENTATION_AXIS_NAMES = ("X", "Y", "Z")
MAX_STICKER_PNP_CANDIDATES = 3
# Full-cube assignment mode. ``class-aware`` treats known keypoint classes as
# hard constraints; ``closest`` remains an explicit geometric-only escape hatch
# for experiments with unreliable class predictions.
PNP_ASSIGNMENT_MODE = "class-aware"
PNP_PIVOT_CONSISTENCY_WEIGHT = 1.5
PNP_PIVOT_HUBER_DELTA = 4.0
PNP_PIVOT_P90_WEIGHT = 0.40
PNP_PIVOT_P95_WEIGHT = 0.20
PNP_PIVOT_MAX_WEIGHT = 0.05
PNP_ASSIGNMENT_TAIL_WEIGHT = 0.25
# Sticker segmentation is a geometric cue, not a hard correspondence.  The
# boundary objective therefore uses capped robust distances so a merged mask
# can omit an internal sticker edge without rejecting an otherwise good pose.
# Actual contours are sampled only to keep repeated turn scoring bounded; no
# convex-hull or quadrilateral replacement is applied.
PNP_MASK_BOUNDARY_MAX_CONTOUR_POINTS = 32
PNP_MASK_BOUNDARY_MAX_EDGE_SAMPLES = 4
PNP_MASK_BOUNDARY_DISTANCE_DELTA_FRACTION = 0.012
PNP_MASK_WIRE_TO_BOUNDARY_WEIGHT = 0.45
PNP_MASK_BOUNDARY_TO_WIRE_WEIGHT = 0.55
PNP_MASK_CROSSING_WEIGHT = 0.35
PNP_MASK_MISSING_BOUNDARY_CAP = 5.0
STICKER_MASK_NEUTRAL_COLOR_BGR = (210, 210, 210)
# Segmentation-derived centroid marker.  BGR for a visible pink (RGB
# 255,105,180), kept separate from the three neural keypoint class colors.
STICKER_MASK_CENTROID_COLOR_BGR = (180, 105, 255)
PNP_MASK_POSE_WEIGHT = 1.0
PNP_MASK_POSE_ROTATION_STEP_DEGREES = 0.35
PNP_MASK_POSE_TRANSLATION_STEP_FRACTION = 0.0025
PNP_MASK_POSE_REFINEMENT_PASSES = 2
PNP_MASK_CENTER_HYPOTHESIS_WEIGHT = 1.0
PNP_MASK_CENTER_HYPOTHESIS_HUBER_DELTA = 2.0
PNP_MASK_CENTER_UNMATCHED_PENALTY = 1.0
PNP_MASK_CENTER_SHAPE_PENALTY = 4.0
PNP_MASK_CENTER_MIN_VISIBLE_AREA = 64.0
# Turn fitting uses the same segmentation-derived center observations as
# static fitting, but only for the layer being tested. Boundary/contour
# scoring remains disabled.
PNP_TURN_MASK_CENTER_WEIGHT = 1.0
# Final-solve observations must be close relative to the displayed cube size.
# This is deliberately expressed as a fraction of the projected cube diameter
# so the same setting works at different distances from the camera.
PNP_FINAL_ASSIGNMENT_ERROR_FRACTION = 0.05
# A pose is accepted when its robust, cube-scale-normalized reprojection
# metric is within the same five-percent geometric tolerance used by the
# per-observation assignment gate.
PNP_POSE_REPROJECTION_ERROR_THRESHOLD = 0.05
PNP_LAYER_MISALIGNMENT_MIN_DEGREES = -45.0
PNP_LAYER_MISALIGNMENT_MAX_DEGREES = 45.0
PNP_LAYER_MISALIGNMENT_STEP_DEGREES = 3.0
# Retained for CLI/API compatibility. Move fitting currently evaluates all six
# layer faces; the value no longer limits the face search.
PNP_TURN_FACE_CANDIDATES = 6
PNP_TURN_DEBUG_CANDIDATE_COUNT = 6
PNP_TURN_OVERLAY_CANDIDATE_COUNT = 6
# The final cached pass scores all 24 equivalent labelings using the complete
# sticker-color history. The initial pose pass deliberately does not spend
# time canonicalizing pose symmetry.
POSE_SYMMETRY_RECOVERY_MIN_OBSERVATIONS = 4
POSE_SYMMETRY_RECOVERY_LOW_PREDICTIVE_PROBABILITY = 0.10


@dataclass(frozen=True)
class StickerKeypointMatch:
    """One segmentation mask and its nearby ordered pose keypoints."""

    mask_index: int
    keypoint_indices: tuple[int, ...]
    sticker_type: str
    average_confidence: float
    shape_error: float = float("inf")


@dataclass(frozen=True)
class StickerIdentity:
    """A canonical physical sticker on the 3x3 cube."""

    label: str
    category: str
    cubie_label: str
    face: str
    cubie_index: tuple[int, int, int]
    axis: int
    side: int


@dataclass
class StickerColorTrace:
    """Per-sticker frame-averaged RGB observations for one video."""

    identities: tuple[StickerIdentity, ...]
    observations: dict[str, list[tuple[int, np.ndarray]]]
    observation_weights: dict[str, list[float]]
    frames_seen: int = 0

    @classmethod
    def create(cls) -> "StickerColorTrace":
        identities = canonical_sticker_identities()
        return cls(
            identities=identities,
            observations={identity.label: [] for identity in identities},
            observation_weights={identity.label: [] for identity in identities},
        )

    def observe(
        self,
        frame_index: int,
        observations: dict[str, np.ndarray],
        observation_weights: dict[str, float] | None = None,
    ) -> None:
        self.frames_seen += 1
        for identity in self.identities:
            rgb = observations.get(identity.label)
            if rgb is None:
                continue
            value = np.asarray(rgb, dtype=np.uint8).reshape(3).copy()
            self.observations[identity.label].append((int(frame_index), value))
            weight = 1.0 if observation_weights is None else observation_weights.get(
                identity.label,
                1.0,
            )
            self.observation_weights[identity.label].append(
                float(np.clip(weight, 0.0, 1.0))
            )


@dataclass
class NeuralFrameData:
    """All per-frame data produced by the detector and the two NN models."""

    frame_index: int
    frame: np.ndarray
    roi_xyxy: tuple[int, int, int, int] | None
    roi_source: str
    sticker_masks: tuple[list[np.ndarray], object, np.ndarray, np.ndarray | None] | None
    local_points: np.ndarray
    point_confidences: np.ndarray
    class_ids: np.ndarray
    box_confidences: np.ndarray
    pose_candidate: object | None = None
    # All cheap pose hypotheses retained for the later refinement pass.
    pose_candidates: tuple["StickerPnPCandidate", ...] = ()
    # A rejected pose is retained only so the final diagnostic overlay can
    # report why it was rejected without drawing that pose as valid.
    pose_diagnostic_candidate: object | None = None
    pose_ready: bool = False


def copy_sticker_masks(
    sticker_masks: tuple[list[np.ndarray], object, np.ndarray, np.ndarray | None] | None,
) -> tuple[list[np.ndarray], object, np.ndarray, np.ndarray | None] | None:
    """Detach segmentation arrays from Ultralytics result objects."""

    if sticker_masks is None:
        return None
    masks, names, class_ids, cube_mask = sticker_masks
    return (
        [np.asarray(mask, dtype=bool).copy() for mask in masks],
        names,
        np.asarray(class_ids, dtype=np.int32).copy(),
        None if cube_mask is None else np.asarray(cube_mask, dtype=bool).copy(),
    )


@dataclass(frozen=True)
class StickerColorProfile:
    """A fitted two-segment black-to-color-to-white RGB trajectory."""

    polyline_rgb: np.ndarray
    polyline_oklab: np.ndarray
    residual: float
    sample_count: int


GLOBAL_COLOR_NAMES = tuple(PURE_COLOR_RGB)
COLOR_LIKELIHOOD_SIGMA = 0.08
STICKER_VIEW_WEIGHT_FLOOR = 0.0
PAIRWISE_TRAJECTORY_DISTANCE_THRESHOLD = 0.035
PAIRWISE_TRAJECTORY_MAX_POINTS = 64
PAIRWISE_TRAJECTORY_SATURATION_FLOOR = 0.08
PAIRWISE_TRAJECTORY_SATURATION_WEIGHT = 2.0


@dataclass(frozen=True)
class GlobalColorProfile:
    """A diagnostic color group initialized from the pure-color priors."""

    color_name: str
    prior: StickerColorProfile | None
    fitted: StickerColorProfile | None
    sample_count: int


@dataclass(frozen=True)
class StickerPnPCandidate:
    """A cube pose obtained from one canonical sticker correspondence."""

    sticker_type: str
    sticker_average_confidence: float
    image_points: np.ndarray
    object_points: np.ndarray
    projected_sticker_points: np.ndarray
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    reprojection_error: float
    overall_reprojection_error: float = float("inf")
    assignment_image_points: np.ndarray | None = None
    assignment_projected_points: np.ndarray | None = None
    assignment_object_points: np.ndarray | None = None
    assignment_observed_class_ids: np.ndarray | None = None
    assignment_projected_class_ids: np.ndarray | None = None
    assignment_fixed_mask: np.ndarray | None = None
    sticker_contour: np.ndarray | None = None
    # All available segmentation contours are retained without assigning them
    # to physical stickers.  The later wireframe objective performs only a
    # geometric comparison, so merged same-color masks remain usable.
    sticker_contours: tuple[np.ndarray, ...] = ()
    # Resampled once when the candidate is created.  Mask scoring is called
    # repeatedly by the local pose search, so rebuilding these samples each
    # time is needlessly expensive.
    sticker_boundary_samples: tuple[np.ndarray, ...] = ()
    sticker_mask_centers: tuple[np.ndarray, ...] = ()
    sticker_mask_center_uncertainties: tuple[float, ...] = ()
    mask_fit_error: float = float("inf")
    mask_boundary_sample_count: int = 0
    mask_center_fit_error: float = float("inf")
    mask_center_match_count: int = 0
    image_size: tuple[int, int] | None = None
    layer_rotations: tuple["CubeLayerRotation", ...] = ()
    # Per-candidate diagnostics: (label, global mean score, inliers,
    # keypoint p90, keypoint mean).
    turn_candidate_scores: tuple[tuple[str, float, int, float, float], ...] = ()
    # Full JSON-safe diagnostics for every tested turn configuration when
    # candidate logging is enabled.
    turn_candidate_diagnostics: tuple[dict[str, object], ...] = ()
    turn_candidate_debug: tuple["TurnCandidateDebug", ...] = ()
    # Per-layer pre-search diagnostics: (layer, keypoint, sticker, combined).
    turn_layer_scores: tuple[tuple[str, float, float, float], ...] = ()
    turn_fit_sticker_score: float = float("inf")
    turn_fit_unexplained_keypoints: int = 0
    turn_fit_evaluated: bool = False
    turn_fit_static_scores: tuple[float, float, float] | None = None
    turn_fit_articulated_scores: tuple[float, float, float] | None = None
    final_solve_image_points: np.ndarray | None = None
    pre_final_rvec: np.ndarray | None = None
    pre_final_tvec: np.ndarray | None = None
    raw_rvec: np.ndarray | None = None
    raw_tvec: np.ndarray | None = None
    motion_diagnostics: PoseMotionDiagnostics | None = None
    orientation_anchor: "OrientationAnchor | None" = None
    object_symmetry: np.ndarray | None = None


@dataclass(frozen=True)
class PoseHypothesisFrame:
    """The current frame's retained pose hypotheses."""

    frame_index: int | None
    candidates: tuple[StickerPnPCandidate, ...]


class PoseHypothesisBuffer:
    """Short history of candidate sets, without buffering rendered images."""

    def __init__(self, max_frames: int = 8) -> None:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        self.max_frames = int(max_frames)
        self._frames: deque[PoseHypothesisFrame] = deque(maxlen=self.max_frames)

    def clear(self) -> None:
        self._frames.clear()

    def append(
        self,
        candidates: tuple[StickerPnPCandidate, ...],
        frame_index: int | None,
    ) -> None:
        self._frames.append(
            PoseHypothesisFrame(
                frame_index=frame_index,
                candidates=tuple(candidates),
            )
        )

    @property
    def frames(self) -> tuple[PoseHypothesisFrame, ...]:
        return tuple(self._frames)

    def previous_candidates(self) -> tuple[StickerPnPCandidate, ...]:
        """Return candidates from prior frames, never the current frame."""

        return tuple(
            candidate
            for frame in list(self._frames)[:-1]
            for candidate in frame.candidates
        )

    def best_candidate_frames(self) -> tuple[PoseHypothesisFrame, ...]:
        """Return one best candidate per frame for the cheap graph pass."""

        return tuple(
            PoseHypothesisFrame(frame.frame_index, frame.candidates[:1])
            for frame in self._frames
            if frame.candidates
        )


@dataclass(frozen=True)
class PoseGraphSelection:
    """The best candidate path through the recent hypothesis graph."""

    candidate: StickerPnPCandidate
    path: tuple[StickerPnPCandidate, ...]
    visual_cost: float
    constraint_cost: float
    violates_limits: bool


def _pose_path_deltas(
    path: tuple[StickerPnPCandidate, ...],
    frame_indices: tuple[int | None, ...],
    dt_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return linear/angular velocity and acceleration for one candidate path."""

    positions = [
        np.asarray(candidate.tvec, dtype=np.float64).reshape(3)
        for candidate in path
    ]
    rotations = [
        cv2.Rodrigues(
            np.asarray(candidate.rvec, dtype=np.float64).reshape(3, 1)
        )[0]
        for candidate in path
    ]
    velocities: list[np.ndarray] = []
    angular_velocities: list[np.ndarray] = []
    for index in range(1, len(path)):
        frame_delta = 1
        if frame_indices[index] is not None and frame_indices[index - 1] is not None:
            frame_delta = max(
                int(frame_indices[index]) - int(frame_indices[index - 1]),
                1,
            )
        local_dt = float(dt_seconds) * frame_delta
        velocities.append((positions[index] - positions[index - 1]) / local_dt)
        relative = rotations[index - 1].T @ rotations[index]
        angular_velocities.append(cv2.Rodrigues(relative)[0].reshape(3) / local_dt)
    accelerations = [
        (velocities[index] - velocities[index - 1]) / float(dt_seconds)
        for index in range(1, len(velocities))
    ]
    angular_accelerations = [
        (angular_velocities[index] - angular_velocities[index - 1])
        / float(dt_seconds)
        for index in range(1, len(angular_velocities))
    ]
    return (
        np.asarray(velocities, dtype=np.float64),
        np.asarray(angular_velocities, dtype=np.float64),
        np.asarray(accelerations, dtype=np.float64),
        np.asarray(angular_accelerations, dtype=np.float64),
    )


def _pose_path_constraint_metrics(
    path: tuple[StickerPnPCandidate, ...],
    frame_indices: tuple[int | None, ...],
    config: PoseMotionFilterConfig,
) -> tuple[float, float, bool]:
    """Score acceleration/jerk violations for one graph path."""

    if len(path) < 3:
        return 0.0, 0.0, False
    _velocities, _angular_velocities, accelerations, angular_accelerations = (
        _pose_path_deltas(path, frame_indices, float(config.dt_seconds))
    )
    if len(accelerations) == 0:
        return 0.0, 0.0, False
    acceleration_excess = np.maximum(
        np.linalg.norm(accelerations, axis=1)
        / max(float(config.max_acceleration_units_s2), 1e-9)
        - 1.0,
        0.0,
    )
    angular_acceleration_excess = np.maximum(
        np.linalg.norm(angular_accelerations, axis=1)
        / max(float(config.max_angular_acceleration_rad_s2), 1e-9)
        - 1.0,
        0.0,
    )
    excess = np.concatenate(
        (
            acceleration_excess,
            angular_acceleration_excess,
        )
    )
    constraint_cost = float(np.sum(excess * excess))
    maximum_excess = float(np.max(excess)) if len(excess) else 0.0
    return constraint_cost, maximum_excess, bool(maximum_excess > 0.0)


def _pose_transition_constraint_metrics(
    path: tuple[StickerPnPCandidate, ...],
    frame_indices: tuple[int | None, ...],
    config: PoseMotionFilterConfig,
) -> tuple[float, bool]:
    """Score only the new acceleration/jerk terms at a path extension."""

    if len(path) < 3:
        return 0.0, False
    tail = path[-4:]
    tail_indices = frame_indices[-4:]
    _velocities, _angular_velocities, accelerations, angular_accelerations = (
        _pose_path_deltas(tail[-3:], tail_indices[-3:], float(config.dt_seconds))
    )
    if len(accelerations) == 0:
        return 0.0, False
    excess_values = [
        max(
            float(np.linalg.norm(accelerations[-1]))
            / max(float(config.max_acceleration_units_s2), 1e-9)
            - 1.0,
            0.0,
        ),
        max(
            float(np.linalg.norm(angular_accelerations[-1]))
            / max(float(config.max_angular_acceleration_rad_s2), 1e-9)
            - 1.0,
            0.0,
        )
    ]
    excess = np.asarray(excess_values, dtype=np.float64)
    return float(np.sum(excess * excess)), bool(np.any(excess > 0.0))


def select_pose_graph_path(
    frames: tuple[PoseHypothesisFrame, ...],
    config: PoseMotionFilterConfig,
    predicted_pose: tuple[np.ndarray, np.ndarray] | None = None,
) -> PoseGraphSelection | None:
    """Choose one candidate path using visual fit and bounded derivatives."""

    usable_frames = tuple(frame for frame in frames if frame.candidates)
    if not usable_frames:
        return None
    frame_indices = tuple(frame.frame_index for frame in usable_frames)
    # The local derivative state only needs the last three nodes: the newest
    # acceleration uses three poses and the newest jerk uses four poses. A
    # dynamic-programming state therefore consists of the last three indices,
    # rather than an exhaustive 3^8 path enumeration.
    states: dict[tuple[int, ...], tuple[tuple[StickerPnPCandidate, ...], float, float, bool]] = {}
    for index, candidate in enumerate(usable_frames[0].candidates):
        states[(index,)] = (
            (candidate,),
            0.15 * pose_hypothesis_visual_score(candidate),
            0.0,
            False,
        )
    for frame_number, frame in enumerate(usable_frames[1:], start=1):
        next_states: dict[
            tuple[int, ...], tuple[tuple[StickerPnPCandidate, ...], float, float, bool]
        ] = {}
        for state_key, (path, visual_cost, constraint_cost, violates_limits) in states.items():
            for candidate_index, candidate in enumerate(frame.candidates):
                extended_path = path + (candidate,)
                extended_indices = frame_indices[max(0, frame_number - 3) : frame_number + 1]
                transition_cost, transition_violation = _pose_transition_constraint_metrics(
                    extended_path[-4:],
                    extended_indices,
                    config,
                )
                extended_key = (*state_key, candidate_index)[-3:]
                extended = (
                    extended_path,
                    visual_cost + 0.15 * pose_hypothesis_visual_score(candidate),
                    constraint_cost + transition_cost,
                    violates_limits or transition_violation,
                )
                previous = next_states.get(extended_key)
                if previous is None or (
                    extended[1] + 8.0 * extended[2]
                    < previous[1] + 8.0 * previous[2]
                ):
                    next_states[extended_key] = extended
        states = next_states

    evaluated = []
    for path, visual_cost, constraint_cost, violates_limits in states.values():
        visual_cost = (
            visual_cost
            + 0.85 * pose_hypothesis_visual_score(path[-1])
        )
        if predicted_pose is not None:
            visual_cost += 0.5 * pose_hypothesis_distance(path[-1], predicted_pose)
        evaluated.append(
            PoseGraphSelection(
                candidate=path[-1],
                path=path,
                visual_cost=visual_cost,
                constraint_cost=constraint_cost,
                violates_limits=violates_limits,
            )
        )
    feasible = [selection for selection in evaluated if not selection.violates_limits]
    pool = feasible if feasible else evaluated
    selected = min(
        pool,
        key=lambda selection: (
            selection.visual_cost + 8.0 * selection.constraint_cost,
            selection.visual_cost,
        ),
    )
    return selected


@dataclass(frozen=True)
class CubeLayerRotation:
    """A correction applied to one face direction and layer width."""

    axis: int
    side: int
    angle_degrees: float
    layer_count: int = 1


@dataclass(frozen=True)
class TurnCandidateDebug:
    """Geometry needed to render one ranked layer-turn hypothesis."""

    label: str
    layer_rotations: tuple[CubeLayerRotation, ...]
    assignment_projected_points: np.ndarray
    assignment_object_points: np.ndarray
    assignment_projected_class_ids: np.ndarray


@dataclass(frozen=True)
class OrientationAnchor:
    """The stationary cube layers used as the rigid-pose support."""

    axis: int | None
    layers: tuple[int, ...]

    @property
    def label(self) -> str:
        if self.axis is None:
            return "all"
        return f"{chr(ord('X') + int(self.axis))}:{','.join(str(layer) for layer in self.layers)}"


class CubeSymmetryTracker:
    """Keep equivalent cube poses in one continuous coordinate frame."""

    def __init__(self) -> None:
        self._rotation: np.ndarray | None = None

    def reset(self) -> None:
        self._rotation = None

    def observe(self, rvec: np.ndarray) -> None:
        self._rotation = cv2.Rodrigues(
            np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        )[0]

    @property
    def rotation(self) -> np.ndarray | None:
        return None if self._rotation is None else self._rotation.copy()


def transform_object_points_by_symmetry(
    object_points: np.ndarray,
    symmetry: np.ndarray,
) -> np.ndarray:
    """Relabel object coordinates under one whole-cube symmetry."""

    points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    return (np.asarray(symmetry, dtype=np.float64) @ points.T).T.astype(np.float32)


def transform_layer_rotation_by_symmetry(
    layer_rotation: CubeLayerRotation,
    symmetry: np.ndarray,
) -> CubeLayerRotation:
    """Transform a layer axis and side into a relabeled cube frame."""

    matrix = np.asarray(symmetry, dtype=np.float64).reshape(3, 3)
    source_axis = int(layer_rotation.axis)
    if source_axis not in (0, 1, 2):
        return layer_rotation
    target_axis = int(np.argmax(np.abs(matrix[:, source_axis])))
    axis_sign = int(round(matrix[target_axis, source_axis]))
    return CubeLayerRotation(
        axis=target_axis,
        side=int(layer_rotation.side) * axis_sign,
        angle_degrees=float(layer_rotation.angle_degrees),
        layer_count=int(layer_rotation.layer_count),
    )


def transform_rvec_by_symmetry(
    rvec: np.ndarray,
    symmetry: np.ndarray,
) -> np.ndarray:
    """Return the equivalent camera pose after relabeling object axes."""

    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    transformed_rotation = rotation @ np.asarray(symmetry, dtype=np.float64).T
    return cv2.Rodrigues(transformed_rotation)[0].reshape(3, 1)


def rotation_symmetry_distance(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Return the geodesic distance between two camera rotations."""

    first_matrix, _ = cv2.Rodrigues(np.asarray(first, dtype=np.float64).reshape(3, 1))
    second_matrix, _ = cv2.Rodrigues(np.asarray(second, dtype=np.float64).reshape(3, 1))
    relative = first_matrix.T @ second_matrix
    return float(np.linalg.norm(cv2.Rodrigues(relative)[0]))


def layer_rotation_debug_label(
    layer_rotations: tuple[CubeLayerRotation, ...],
) -> str:
    """Return a compact label for a tested layer-turn hypothesis."""

    if not layer_rotations:
        return "none"
    axis_names = "XYZ"
    parts = []
    for rotation in layer_rotations:
        axis_index = int(rotation.axis)
        axis = axis_names[axis_index] if axis_index in range(3) else "?"
        side = "+" if int(rotation.side) > 0 else "-"
        angle = float(rotation.angle_degrees)
        width = int(rotation.layer_count)
        parts.append(f"{axis}{side}{angle:+.0f}w{width}")
    return "/".join(parts)


def parse_source(source: str) -> int | str:
    source = source.strip().strip("\"'").strip()
    return int(source) if source.isdigit() else source


def is_image_source(source: int | str) -> bool:
    return isinstance(source, str) and Path(source).suffix.lower() in IMAGE_EXTENSIONS


def tensor_to_numpy(value: object) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    return array if array.size else None


def resize_to_square(image: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def pose_predictions(result: object) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return points, point confidence, class IDs, and box confidence."""

    keypoints = getattr(result, "keypoints", None)
    boxes = getattr(result, "boxes", None)
    if keypoints is None or boxes is None:
        return (np.empty((0, 2), dtype=np.float32),) * 2 + (
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
        )

    points = tensor_to_numpy(getattr(keypoints, "xy", None))
    if points is None:
        return (np.empty((0, 2), dtype=np.float32),) * 2 + (
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
        )
    points = np.asarray(points, dtype=np.float32)
    if points.ndim == 3:
        points = points[:, 0, :]
    if points.ndim != 2 or points.shape[1] < 2:
        raise RuntimeError(f"Unexpected independent pose point shape: {points.shape}")
    points = points[:, :2]

    point_confidence = tensor_to_numpy(getattr(keypoints, "conf", None))
    if point_confidence is None:
        point_confidence = np.ones((len(points),), dtype=np.float32)
    else:
        point_confidence = np.asarray(point_confidence, dtype=np.float32).reshape(len(points), -1)[:, 0]

    class_ids = tensor_to_numpy(getattr(boxes, "cls", None))
    if class_ids is None:
        class_ids = np.zeros((len(points),), dtype=np.int32)
    else:
        class_ids = np.asarray(class_ids, dtype=np.float32).reshape(-1).astype(np.int32)

    box_confidence = tensor_to_numpy(getattr(boxes, "conf", None))
    if box_confidence is None:
        box_confidence = np.ones((len(points),), dtype=np.float32)
    else:
        box_confidence = np.asarray(box_confidence, dtype=np.float32).reshape(-1)

    count = min(len(points), len(point_confidence), len(class_ids), len(box_confidence))
    return points[:count], point_confidence[:count], class_ids[:count], box_confidence[:count]


def class_name(names: object, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), class_id)))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def sticker_color(names: object, class_id: int) -> tuple[int, int, int]:
    name = class_name(names, class_id).strip().lower().replace("_", " ").replace("-", " ")
    return STICKER_COLORS_BGR.get(name, STICKER_COLORS_BGR["sticker"])


def sticker_class_ids(names: object) -> set[int] | None:
    """Return allowed sticker IDs, or None when model metadata is unknown."""

    if isinstance(names, dict):
        class_ids = {
            int(class_id)
            for class_id in names
            if str(class_id).lstrip("-").isdigit()
        }
    elif isinstance(names, (list, tuple)):
        class_ids = set(range(len(names)))
    else:
        return None

    explicit_sticker_ids = {
        class_id
        for class_id in class_ids
        if class_name(names, class_id).strip().lower() == "sticker"
    }
    if explicit_sticker_ids:
        return explicit_sticker_ids

    color_ids = {
        class_id
        for class_id in class_ids
        if class_name(names, class_id).strip().lower() in STICKER_COLORS_BGR
    }
    return color_ids or None


def cube_class_ids(names: object) -> set[int]:
    """Return segmentation class IDs whose metadata name is ``cube``."""

    if isinstance(names, dict):
        class_ids = {
            int(class_id)
            for class_id in names
            if str(class_id).lstrip("-").isdigit()
        }
    elif isinstance(names, (list, tuple)):
        class_ids = set(range(len(names)))
    else:
        return set()
    return {
        class_id
        for class_id in class_ids
        if class_name(names, class_id).strip().lower() == "cube"
    }


def predict_sticker_result(
    model: object,
    model_roi: np.ndarray,
    args: argparse.Namespace,
) -> object | None:
    predict_kwargs = {} if args.device is None else {"device": args.device}
    results = model.predict(
        source=model_roi,
        conf=args.sticker_mask_conf,
        imgsz=args.imgsz,
        retina_masks=False,
        verbose=False,
        **predict_kwargs,
    )
    return results[0] if results else None


def sticker_masks_from_result(
    result: object | None,
) -> tuple[list[np.ndarray], object, np.ndarray, np.ndarray | None] | None:
    if result is None:
        return None
    masks = getattr(result, "masks", None)
    if masks is None:
        return None
    mask_data = tensor_to_numpy(getattr(masks, "data", None))
    if mask_data is None or mask_data.ndim < 3 or len(mask_data) == 0:
        return None
    boxes = getattr(result, "boxes", None)
    class_ids = tensor_to_numpy(getattr(boxes, "cls", None)) if boxes is not None else None
    if class_ids is None:
        class_ids = np.zeros((len(mask_data),), dtype=np.int32)
    class_ids = np.asarray(class_ids, dtype=np.int32).reshape(-1)
    names = getattr(result, "names", {})
    allowed_sticker_ids = sticker_class_ids(names)
    allowed_cube_ids = cube_class_ids(names)
    cube_mask = None
    if allowed_cube_ids:
        cube_indices = [
            index
            for index, class_id in enumerate(class_ids[: len(mask_data)])
            if int(class_id) in allowed_cube_ids
        ]
        if cube_indices:
            cube_mask = np.any(
                np.asarray(mask_data[cube_indices]) >= 0.5,
                axis=0,
            )

    # Keep only sticker instances for keypoint matching and rendering, but
    # retain the model's separate cube mask for pose pre-filtering.
    if allowed_sticker_ids is not None:
        keep = [
            index
            for index, class_id in enumerate(class_ids[: len(mask_data)])
            if int(class_id) in allowed_sticker_ids
        ]
    else:
        keep = list(range(min(len(mask_data), len(class_ids))))
    return (
        [np.asarray(mask_data[index]) >= 0.5 for index in keep],
        names,
        class_ids[keep],
        cube_mask,
    )


def erode_binary_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Erode a binary sticker mask by the requested number of pixels."""

    pixels = max(0, int(pixels))
    if pixels == 0:
        return np.asarray(mask, dtype=bool)
    kernel_size = pixels * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.erode(
        np.asarray(mask, dtype=np.uint8),
        kernel,
        iterations=1,
    ).astype(bool)


def predict_sticker_masks(
    model: object,
    model_roi: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], object, np.ndarray, np.ndarray | None] | None:
    return sticker_masks_from_result(predict_sticker_result(model, model_roi, args))


def order_sticker_keypoint_indices(
    mask: np.ndarray,
    points: np.ndarray,
    indices: list[int],
) -> list[int]:
    """Order matched points around a sticker in one deterministic direction.

    Image-space ``atan2`` increases clockwise because image y coordinates point
    down.  The important property here is consistency: the same ordering can
    later be used for the 2-D/3-D correspondence passed to ``solvePnP``.
    """

    ys, xs = np.nonzero(mask)
    if not len(indices) or len(xs) == 0:
        return []
    center_x = float(xs.mean())
    center_y = float(ys.mean())
    ordered = sorted(
        indices,
        key=lambda index: (
            float(np.arctan2(points[index, 1] - center_y, points[index, 0] - center_x)),
            index,
        ),
    )
    if len(ordered) >= 3:
        # In image coordinates a positive 2-D cross product is a right turn:
        # x points right and y points down.  Keep the order explicitly
        # clockwise, rather than relying only on atan2's coordinate-system
        # convention.  This is the direction that must be preserved by the
        # object-point sequence passed to solvePnP.
        turn_sum = 0.0
        for offset in range(len(ordered)):
            first = points[ordered[offset]]
            second = points[ordered[(offset + 1) % len(ordered)]]
            third = points[ordered[(offset + 2) % len(ordered)]]
            first_edge = second - first
            second_edge = third - second
            turn_sum += float(
                first_edge[0] * second_edge[1]
                - first_edge[1] * second_edge[0]
            )
        if turn_sum < 0.0:
            ordered.reverse()
    return ordered


def sticker_shape_error(
    points: np.ndarray,
    indices: list[int] | tuple[int, ...],
) -> float:
    """Return a cheap, scale-normalized shape error for a sticker candidate."""

    selected = np.asarray(points, dtype=np.float64).reshape(-1, 2)[list(indices)]
    if len(selected) == 4:
        edges = np.roll(selected, -1, axis=0) - selected
        scale = max(float(np.mean(np.linalg.norm(edges, axis=1))), 1e-6)
        diagonal_midpoint_error = 0.5 * float(
            np.linalg.norm(selected[0] + selected[2] - selected[1] - selected[3])
        ) / scale
        opposite_edge_error = 0.5 * float(
            np.linalg.norm(edges[0] - edges[2])
            + np.linalg.norm(edges[1] - edges[3])
        ) / scale
        return diagonal_midpoint_error + opposite_edge_error
    return float("inf")


def sticker_match_priority_key(match: StickerKeypointMatch) -> tuple[float, float, int]:
    """Prefer geometrically plausible stickers, then use confidence as a tie-breaker."""

    if len(match.keypoint_indices) != 4:
        return (
            0.0,
            -float(match.average_confidence),
            int(match.mask_index),
        )
    return (
        float(match.shape_error),
        -float(match.average_confidence),
        int(match.mask_index),
    )


def sticker_type_from_keypoint_classes(class_ids: list[int]) -> str:
    """Infer corner-, edge-, or center-sticker type from 3-4 point classes.

    The full patterns are ``0,1,1,2`` for a corner sticker,
    ``1,1,2,2`` for an edge sticker, and ``2,2,2,2`` for a center sticker.
    With only three visible points, a corner/edge distinction can be
    ambiguous when the corner point is missing.  In that case this first-pass
    classifier uses the observed classes: an observed corner wins, all-center
    points are a center sticker, and the remaining valid mixed pattern is an
    edge sticker.
    """

    classes = sorted(int(class_id) for class_id in class_ids)
    counts = {class_id: classes.count(class_id) for class_id in range(len(CLASS_NAMES))}
    if len(classes) == 4:
        if classes == [0, 1, 1, 2]:
            return "corner"
        if classes == [1, 1, 2, 2]:
            return "edge"
        if classes == [2, 2, 2, 2]:
            return "center"
        return "unknown"
    if len(classes) != 3:
        return "unknown"
    if counts[0] == 1 and counts[1] >= 1 and counts[2] <= 1:
        return "corner"
    if counts[0] == 0 and counts[1] >= 1 and counts[2] >= 1:
        return "edge"
    if classes == [2, 2, 2]:
        return "center"
    return "unknown"


def sticker_keypoint_matches(
    mask: np.ndarray,
    points: np.ndarray,
    point_confidences: np.ndarray,
    class_ids: np.ndarray,
    box_confidences: np.ndarray,
    point_threshold: float,
    class_confidences: dict[str, float],
) -> tuple[list[int], str]:
    """Match the nearest 3-4 independent points to one local sticker mask."""

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return [], "unknown"
    contour = max(contours, key=cv2.contourArea)
    _, _, mask_width, mask_height = cv2.boundingRect(contour)
    # The point labels sit at sticker corners.  Keep the radius relative to
    # the sticker, with a small floor for low-resolution masks and predictions.
    boundary_threshold = max(4.0, 0.25 * min(mask_width, mask_height))
    candidates: list[tuple[float, int]] = []
    for index, (point, point_confidence, class_id, box_confidence) in enumerate(
        zip(points, point_confidences, class_ids, box_confidences, strict=True)
    ):
        class_id = int(class_id)
        if not 0 <= class_id < len(CLASS_NAMES):
            continue
        class_name = CLASS_NAMES[class_id]
        if float(box_confidence) < independent_pose_class_threshold(
            class_name, class_confidences
        ) or float(point_confidence) < point_threshold:
            continue
        if not np.isfinite(point).all():
            continue
        distance = abs(
            cv2.pointPolygonTest(
                contour,
                (float(point[0]), float(point[1])),
                True,
            )
        )
        if distance <= boundary_threshold:
            candidates.append((distance, index))

    candidates.sort()
    matched_indices = [index for _distance, index in candidates[:4]]
    ordered_indices = order_sticker_keypoint_indices(mask, points, matched_indices)
    sticker_type = sticker_type_from_keypoint_classes(
        [int(class_ids[index]) for index in ordered_indices]
    )
    return ordered_indices, sticker_type


def sticker_keypoint_matches_for_masks(
    local_masks: list[np.ndarray],
    points: np.ndarray,
    point_confidences: np.ndarray,
    class_ids: np.ndarray,
    box_confidences: np.ndarray,
    point_threshold: float,
    class_confidences: dict[str, float],
) -> tuple[StickerKeypointMatch, ...]:
    """Match every predicted sticker mask and score it by mean point confidence."""

    matches = []
    for mask_index, local_mask in enumerate(local_masks):
        ordered_indices, sticker_type = sticker_keypoint_matches(
            local_mask,
            points,
            point_confidences,
            class_ids,
            box_confidences,
            point_threshold,
            class_confidences,
        )
        average_confidence = (
            float(np.mean(point_confidences[ordered_indices]))
            if ordered_indices
            else 0.0
        )
        shape_error = (
            sticker_shape_error(points, ordered_indices)
            if len(ordered_indices) == 4
            else float("inf")
        )
        matches.append(
            StickerKeypointMatch(
                mask_index=mask_index,
                keypoint_indices=tuple(ordered_indices),
                sticker_type=sticker_type,
                average_confidence=average_confidence,
                shape_error=shape_error,
            )
        )
    return tuple(matches)


def cube_lattice_point_mm(coord: tuple[int, int, int]) -> tuple[float, float, float]:
    """Convert a 4x4x4 surface-lattice coordinate to cube millimetres."""

    return tuple(
        (value - 1.5) * CUBE_LAYER_SIZE_MM
        for value in coord
    )


def cube_lattice_object_points() -> np.ndarray:
    """Return the 56 independent-keypoint locations on the cube surface."""

    return CUBE_LATTICE_OBJECT_POINTS_MM


def cube_lattice_boundary_faces(
    coord: tuple[int, int, int],
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (axis, value)
        for axis, value in enumerate(coord)
        if value in (0, CUBE_GRID_MAX_INDEX)
    )


def cube_lattice_class_id(coord: tuple[int, int, int]) -> int:
    """Return the independent-keypoint class for a surface lattice point."""

    boundary_face_count = len(cube_lattice_boundary_faces(coord))
    if boundary_face_count == 3:
        return 0  # corner
    if boundary_face_count == 2:
        return 1  # corner-edge
    return 2  # corner-edge-center


def _ordered_cubie_label(face_names: tuple[str, ...]) -> str:
    return "".join(sorted(face_names, key=lambda name: FACE_NAME_ORDER[name]))


@lru_cache(maxsize=1)
def canonical_sticker_identities() -> tuple[StickerIdentity, ...]:
    """Return all 54 stickers in stable, human-readable order."""

    by_category: dict[str, list[StickerIdentity]] = {
        "center": [],
        "edge": [],
        "corner": [],
    }
    for x_index in range(3):
        for y_index in range(3):
            for z_index in range(3):
                cubie_index = (x_index, y_index, z_index)
                boundary_faces = tuple(
                    (axis, side)
                    for axis, coordinate in enumerate(cubie_index)
                    for side in (-1, 1)
                    if coordinate == 0 and side == -1
                    or coordinate == 2 and side == 1
                )
                category = {
                    1: "center",
                    2: "edge",
                    3: "corner",
                }.get(len(boundary_faces))
                if category is None:
                    continue
                cubie_label = _ordered_cubie_label(
                    tuple(FACE_NAME_BY_AXIS_SIDE[key] for key in boundary_faces)
                )
                for axis, side in boundary_faces:
                    face = FACE_NAME_BY_AXIS_SIDE[(axis, side)]
                    by_category[category].append(
                        StickerIdentity(
                            label="",
                            category=category,
                            cubie_label=cubie_label,
                            face=face,
                            cubie_index=cubie_index,
                            axis=axis,
                            side=side,
                        )
                    )

    category_order = ("center", "edge", "corner")
    cubie_orders = {
        "center": {face: index for index, face in enumerate(CENTER_FACE_ORDER)},
        "edge": {label: index for index, label in enumerate(EDGE_CUBIE_ORDER)},
        "corner": {label: index for index, label in enumerate(CORNER_CUBIE_ORDER)},
    }
    identities: list[StickerIdentity] = []
    for category in category_order:
        ordered = sorted(
            by_category[category],
            key=lambda identity: (
                cubie_orders[category].get(
                    identity.face if category == "center" else identity.cubie_label,
                    999,
                ),
                FACE_NAME_ORDER[identity.face],
            ),
        )
        prefix = {"center": "C", "edge": "E", "corner": "K"}[category]
        for index, identity in enumerate(ordered, start=1):
            label = (
                f"{prefix}{index:02d} {identity.face}"
                if category == "center"
                else f"{prefix}{index:02d} {identity.cubie_label}/{identity.face}"
            )
            identities.append(replace(identity, label=label))
    if len(identities) != 54:
        raise AssertionError(f"Expected 54 sticker identities, got {len(identities)}")
    return tuple(identities)


def _face_name_from_vector(vector: np.ndarray) -> str:
    vector = np.asarray(vector, dtype=np.int32).reshape(3)
    axis = int(np.argmax(np.abs(vector)))
    side = int(np.sign(vector[axis]))
    return FACE_NAME_BY_AXIS_SIDE[(axis, side)]


def legal_center_color_assignments() -> tuple[dict[str, str], ...]:
    """Return the 24 proper rotations of the standard center color scheme."""

    assignments = []
    for symmetry in CUBE_SYMMETRY_MATRICES:
        assignment = {}
        for face, color_name in STANDARD_FACE_COLOR_NAMES.items():
            target_vector = symmetry @ FACE_VECTOR_BY_NAME[face]
            assignment[_face_name_from_vector(target_vector)] = color_name
        assignments.append(assignment)
    return tuple(assignments)


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    normalized = np.asarray(rgb, dtype=np.float64) / 255.0
    return np.where(
        normalized <= 0.04045,
        normalized / 12.92,
        ((normalized + 0.055) / 1.055) ** 2.4,
    )


def rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Convert one or more sRGB colors to OKLab."""

    linear = _srgb_to_linear(np.asarray(rgb, dtype=np.float64))
    lms = linear @ np.asarray(
        (
            (0.4122214708, 0.5363325363, 0.0514459929),
            (0.2119034982, 0.6806995451, 0.1073969566),
            (0.0883024619, 0.2817188376, 0.6299787005),
        ),
        dtype=np.float64,
    ).T
    lms_cbrt = np.cbrt(np.maximum(lms, 0.0))
    return lms_cbrt @ np.asarray(
        (
            (0.2104542553, 0.7936177850, -0.0040720468),
            (1.9779984951, -2.4285922050, 0.4505937099),
            (0.0259040371, 0.7827717662, -0.8086757660),
        ),
        dtype=np.float64,
    ).T


def oklab_to_rgb(oklab: np.ndarray) -> np.ndarray:
    """Convert one or more OKLab colors to clipped sRGB values in 0..255."""

    oklab = np.asarray(oklab, dtype=np.float64).reshape(-1, 3)
    lightness = oklab[:, 0]
    a_component = oklab[:, 1]
    b_component = oklab[:, 2]
    lms_cbrt = np.column_stack(
        (
            lightness + 0.3963377774 * a_component + 0.2158037573 * b_component,
            lightness - 0.1055613458 * a_component - 0.0638541728 * b_component,
            lightness - 0.0894841775 * a_component - 1.2914855480 * b_component,
        )
    )
    lms = lms_cbrt**3
    linear_rgb = lms @ np.asarray(
        (
            (4.0767416621, -3.3077115913, 0.2309699292),
            (-1.2684380046, 2.6097574011, -0.3413193965),
            (-0.0041960863, -0.7034186147, 1.7076147010),
        ),
        dtype=np.float64,
    ).T
    nonlinear_rgb = np.where(
        linear_rgb <= 0.0031308,
        12.92 * linear_rgb,
        1.055 * np.power(np.maximum(linear_rgb, 0.0), 1.0 / 2.4) - 0.055,
    )
    return np.clip(nonlinear_rgb * 255.0, 0.0, 255.0)


def _oklab_point_to_segment_distances(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    start = np.asarray(start, dtype=np.float64).reshape(3)
    end = np.asarray(end, dtype=np.float64).reshape(3)
    direction = end - start
    denominator = float(np.dot(direction, direction))
    if denominator <= 1e-12:
        return np.linalg.norm(points - start, axis=1)
    projection = np.clip(
        ((points - start) @ direction) / denominator,
        0.0,
        1.0,
    )
    closest = start + projection[:, None] * direction
    return np.linalg.norm(points - closest, axis=1)


def _piecewise_profile_distances(
    points: np.ndarray,
    polyline_oklab: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    polyline = np.asarray(polyline_oklab, dtype=np.float64).reshape(-1, 3)
    if len(polyline) < 2:
        return np.linalg.norm(points - polyline[0], axis=1)
    return np.min(
        np.asarray(
            [
                _oklab_point_to_segment_distances(points, start, end)
                for start, end in zip(polyline[:-1], polyline[1:], strict=True)
            ]
        ),
        axis=0,
    )


def _fit_rgb_inflection_point(rgb_points: np.ndarray) -> np.ndarray:
    """Fit the RGB point where the first segment reaches one channel at 255."""

    points = np.asarray(rgb_points, dtype=np.float64).reshape(-1, 3)
    black = np.zeros(3, dtype=np.float64)
    white = np.full(3, 255.0, dtype=np.float64)
    candidates: list[tuple[int, np.ndarray]] = []
    dominant_channels = np.argmax(points, axis=1)
    for channel in range(3):
        channel_points = points[dominant_channels == channel]
        channel_points = channel_points[channel_points[:, channel] > 1e-6]
        if len(channel_points) == 0:
            continue
        scaled = channel_points * (255.0 / channel_points[:, channel, None])
        scaled = np.clip(scaled, 0.0, 255.0)
        for pivot in scaled[:: max(1, len(scaled) // 8)]:
            pivot = pivot.copy()
            pivot[channel] = 255.0
            candidates.append((channel, pivot))
        median_pivot = np.median(scaled, axis=0)
        median_pivot[channel] = 255.0
        candidates.append((channel, median_pivot))
    if not candidates:
        return white

    def score(pivot: np.ndarray) -> float:
        polyline = np.asarray((black, pivot, white), dtype=np.float64)
        return float(np.mean(_piecewise_profile_distances(points, polyline) ** 2))

    best_channel, best_pivot = min(candidates, key=lambda item: score(item[1]))
    best_score = score(best_pivot)
    other_channels = tuple(index for index in range(3) if index != best_channel)
    for step in (32.0, 16.0, 8.0, 4.0, 2.0, 1.0):
        improved = True
        while improved:
            improved = False
            for channel in other_channels:
                for delta in (-step, step):
                    candidate = best_pivot.copy()
                    candidate[channel] = np.clip(candidate[channel] + delta, 0.0, 255.0)
                    candidate[best_channel] = 255.0
                    candidate_score = score(candidate)
                    if candidate_score < best_score:
                        best_pivot = candidate
                        best_score = candidate_score
                        improved = True
    return best_pivot


def fit_sticker_color_profile(
    trace: StickerColorTrace,
    sticker_label: str,
) -> StickerColorProfile | None:
    """Fit a two-segment RGB profile with a one-channel-255 inflection."""

    entries = trace.observations.get(sticker_label, ())
    if not entries:
        return None
    rgb_points = np.asarray(
        [value for _frame, value in entries],
        dtype=np.float64,
    ).reshape(-1, 3)
    return fit_rgb_color_profile(rgb_points)


def fit_rgb_color_profile(rgb_points: np.ndarray) -> StickerColorProfile | None:
    """Fit the existing constrained two-segment model to RGB observations."""

    rgb_points = np.asarray(rgb_points, dtype=np.float64).reshape(-1, 3)
    if len(rgb_points) == 0:
        return None
    inflection = _fit_rgb_inflection_point(rgb_points)
    polyline_rgb = np.asarray(
        ((0.0, 0.0, 0.0), inflection, (255.0, 255.0, 255.0)),
        dtype=np.float32,
    )
    points_oklab = rgb_to_oklab(rgb_points).astype(np.float32)
    polyline_oklab = rgb_to_oklab(polyline_rgb).astype(np.float32)
    residual = float(
        np.sqrt(np.mean(_piecewise_profile_distances(points_oklab, polyline_oklab) ** 2))
    )
    return StickerColorProfile(
        polyline_rgb,
        polyline_oklab,
        residual,
        len(rgb_points),
    )


def pure_rgb_color_profile(color_name: str) -> StickerColorProfile:
    """Build a weak diagnostic prior from one pure RGB color."""

    if color_name not in PURE_COLOR_RGB:
        raise KeyError(color_name)
    polyline_rgb = np.asarray(
        (
            (0.0, 0.0, 0.0),
            PURE_COLOR_RGB[color_name],
            (255.0, 255.0, 255.0),
        ),
        dtype=np.float32,
    )
    return StickerColorProfile(
        polyline_rgb=polyline_rgb,
        polyline_oklab=rgb_to_oklab(polyline_rgb).astype(np.float32),
        residual=0.0,
        sample_count=0,
    )


def classify_rgb_by_global_prior(
    rgb: np.ndarray,
    prior_profiles: dict[str, StickerColorProfile] | None = None,
) -> tuple[str, float]:
    """Assign one observation to its nearest pure-color prior."""

    prior_profiles = (
        {
            color_name: pure_rgb_color_profile(color_name)
            for color_name in PURE_COLOR_RGB
        }
        if prior_profiles is None
        else prior_profiles
    )
    point_oklab = rgb_to_oklab(np.asarray(rgb, dtype=np.float64).reshape(1, 3))
    distances = {
        color_name: float(
            _piecewise_profile_distances(
                point_oklab,
                profile.polyline_oklab,
            )[0]
        )
        for color_name, profile in prior_profiles.items()
    }
    return min(distances.items(), key=lambda item: item[1])


def build_global_color_profiles(
    trace: StickerColorTrace,
) -> tuple[
    dict[str, GlobalColorProfile],
    dict[str, list[tuple[str, int, np.ndarray]]],
]:
    """Build global profiles by grouping observations with weak pure priors."""

    grouped: dict[str, list[tuple[str, int, np.ndarray]]] = {
        color_name: [] for color_name in GLOBAL_COLOR_NAMES
    }
    prior_profiles = {
        color_name: pure_rgb_color_profile(color_name)
        for color_name in PURE_COLOR_RGB
    }
    for identity in trace.identities:
        for frame_index, rgb in trace.observations.get(identity.label, ()):
            color_name, _distance = classify_rgb_by_global_prior(rgb, prior_profiles)
            grouped[color_name].append(
                (identity.label, int(frame_index), np.asarray(rgb, dtype=np.uint8).copy())
            )

    profiles: dict[str, GlobalColorProfile] = {}
    for color_name in GLOBAL_COLOR_NAMES:
        entries = grouped[color_name]
        values = np.asarray([entry[2] for entry in entries], dtype=np.float64)
        profiles[color_name] = GlobalColorProfile(
            color_name=color_name,
            prior=prior_profiles.get(color_name),
            fitted=fit_rgb_color_profile(values),
            sample_count=len(entries),
        )
    return profiles, grouped


def rgb_observation_color_likelihoods(
    rgb: np.ndarray,
    global_profiles: dict[str, GlobalColorProfile],
) -> dict[str, float]:
    """Return normalized soft likelihoods for one RGB observation."""

    evidence = _rgb_observation_color_evidence(rgb, global_profiles)
    evidence /= max(float(np.sum(evidence)), 1e-12)
    return {
        color_name: float(likelihood)
        for color_name, likelihood in zip(GLOBAL_COLOR_NAMES, evidence, strict=True)
    }


def _rgb_observation_color_evidence(
    rgb: np.ndarray,
    global_profiles: dict[str, GlobalColorProfile],
) -> np.ndarray:
    """Return unnormalized p(observation|color) values for one RGB sample."""

    point_oklab = rgb_to_oklab(np.asarray(rgb, dtype=np.float64).reshape(1, 3))
    distances = np.asarray(
        [
            _piecewise_profile_distances(
                point_oklab,
                _global_profile_reference(global_profiles[color_name]).polyline_oklab,
            )[0]
            for color_name in GLOBAL_COLOR_NAMES
        ],
        dtype=np.float64,
    )
    return np.exp(-0.5 * (distances / COLOR_LIKELIHOOD_SIGMA) ** 2)


def _apply_observation_weight(
    evidence: np.ndarray,
    weight: float,
) -> np.ndarray:
    """Temper one observation's likelihood without changing its color shape."""

    clipped_weight = float(np.clip(weight, 0.0, 1.0))
    safe_evidence = np.clip(np.asarray(evidence, dtype=np.float64), 1e-300, None)
    return np.exp(clipped_weight * np.log(safe_evidence))


def _trace_observation_weight(
    trace: StickerColorTrace,
    label: str,
    observation_index: int,
) -> float:
    weights = trace.observation_weights.get(label, ())
    if observation_index >= len(weights):
        return 1.0
    return float(np.clip(weights[observation_index], 0.0, 1.0))


def build_sticker_observation_likelihoods(
    trace: StickerColorTrace,
    global_profiles: dict[str, GlobalColorProfile] | None = None,
) -> dict[int, dict[str, dict[str, float]]]:
    """Build normalized color posteriors for each sticker identity and frame."""

    if global_profiles is None:
        global_profiles, _global_observations = build_global_color_profiles(trace)
    priors = {
        identity.label: np.full(
            len(GLOBAL_COLOR_NAMES),
            1.0 / len(GLOBAL_COLOR_NAMES),
            dtype=np.float64,
        )
        for identity in trace.identities
    }
    observations_by_frame: dict[int, list[tuple[str, np.ndarray, float]]] = {}
    for identity in trace.identities:
        for observation_index, (frame_index, rgb) in enumerate(
            trace.observations.get(identity.label, ())
        ):
            observations_by_frame.setdefault(int(frame_index), []).append(
                (
                    identity.label,
                    rgb,
                    _trace_observation_weight(
                        trace,
                        identity.label,
                        observation_index,
                    ),
                )
            )
    likelihoods_by_frame: dict[int, dict[str, dict[str, float]]] = {}
    for frame_index in sorted(observations_by_frame):
        for label, rgb, weight in observations_by_frame[frame_index]:
            evidence = _rgb_observation_color_evidence(rgb, global_profiles)
            prior = priors[label]
            posterior = prior * _apply_observation_weight(evidence, weight)
            posterior_sum = float(np.sum(posterior))
            if posterior_sum > 1e-12:
                posterior /= posterior_sum
            else:
                posterior = prior.copy()
            likelihoods_by_frame.setdefault(frame_index, {})[label] = {
                color_name: float(probability)
                for color_name, probability in zip(
                    GLOBAL_COLOR_NAMES,
                    posterior,
                    strict=True,
                )
            }
            priors[label] = posterior
    return likelihoods_by_frame


def classify_sticker_colors_by_final_posterior(
    likelihoods_by_frame: dict[int, dict[str, dict[str, float]]],
) -> dict[str, str]:
    """Assign each sticker its final independent-posterior argmax color."""

    final_posteriors = final_sticker_posteriors(likelihoods_by_frame)
    return {
        label: max(probabilities.items(), key=lambda item: item[1])[0]
        for label, probabilities in final_posteriors.items()
        if probabilities
    }


def final_sticker_posteriors(
    likelihoods_by_frame: dict[int, dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    """Return the latest posterior recorded for each sticker identity."""

    final_posteriors: dict[str, dict[str, float]] = {}
    for frame_index in sorted(likelihoods_by_frame):
        final_posteriors.update(
            {
                label: dict(probabilities)
                for label, probabilities in likelihoods_by_frame[frame_index].items()
            }
        )
    return final_posteriors


def classify_sticker_colors_by_frame_posterior(
    likelihoods_by_frame: dict[int, dict[str, dict[str, float]]],
) -> dict[int, dict[str, str]]:
    """Assign each observed sticker its posterior argmax at every frame."""

    return {
        frame_index: {
            label: max(probabilities.items(), key=lambda item: item[1])[0]
            for label, probabilities in frame_posteriors.items()
            if probabilities
        }
        for frame_index, frame_posteriors in likelihoods_by_frame.items()
    }


def carry_forward_sticker_posteriors(
    likelihoods_by_frame: dict[int, dict[str, dict[str, float]]],
    frame_indices: list[int] | tuple[int, ...],
) -> dict[int, dict[str, dict[str, float]]]:
    """Carry each sticker's latest posterior through frames without a sample."""

    current: dict[str, dict[str, float]] = {}
    carried: dict[int, dict[str, dict[str, float]]] = {}
    for frame_index in sorted(set(int(index) for index in frame_indices)):
        for label, probabilities in likelihoods_by_frame.get(frame_index, {}).items():
            current[label] = dict(probabilities)
        carried[frame_index] = {
            label: dict(probabilities)
            for label, probabilities in current.items()
        }
    return carried


def _trace_observations_by_frame(
    trace: StickerColorTrace,
) -> dict[int, dict[str, np.ndarray]]:
    """Index the raw trace without changing its original sticker labels."""

    observations_by_frame: dict[int, dict[str, np.ndarray]] = {}
    for identity in trace.identities:
        for frame_index, rgb in trace.observations.get(identity.label, ()):
            observations_by_frame.setdefault(int(frame_index), {})[
                identity.label
            ] = np.asarray(rgb, dtype=np.uint8).reshape(3).copy()
    return observations_by_frame


def _trace_observation_weights_by_frame(
    trace: StickerColorTrace,
) -> dict[int, dict[str, float]]:
    """Index per-observation reliability weights using the same frame keys."""

    weights_by_frame: dict[int, dict[str, float]] = {}
    for identity in trace.identities:
        for observation_index, (frame_index, _rgb) in enumerate(
            trace.observations.get(identity.label, ())
        ):
            weights_by_frame.setdefault(int(frame_index), {})[
                identity.label
            ] = _trace_observation_weight(
                trace,
                identity.label,
                observation_index,
            )
    return weights_by_frame


def sticker_identity_label_under_symmetry(
    identity: StickerIdentity,
    symmetry: np.ndarray,
    identities_by_geometry: dict[tuple[tuple[int, int, int], int, int], StickerIdentity]
    | None = None,
) -> str:
    """Return the source label that projects at ``identity`` after relabeling.

    ``transform_candidate_by_symmetry`` uses ``R' = R S^T``.  Consequently,
    target identity ``x`` in the transformed candidate occupies the old image
    location of ``S^T x``.  This helper gives the corresponding raw-trace
    label so color observations can be scored under a candidate symmetry.
    """

    if identities_by_geometry is None:
        identities_by_geometry = {
            (item.cubie_index, item.axis, item.side): item
            for item in canonical_sticker_identities()
        }
    matrix = np.asarray(symmetry, dtype=np.float64).reshape(3, 3)
    target_center = np.asarray(identity.cubie_index, dtype=np.float64) - 1.0
    source_center = np.rint(matrix.T @ target_center).astype(np.int32) + 1
    normal = np.zeros(3, dtype=np.float64)
    normal[int(identity.axis)] = float(identity.side)
    source_normal = matrix.T @ normal
    source_axis = int(np.argmax(np.abs(source_normal)))
    source_side = int(np.sign(source_normal[source_axis]))
    source = identities_by_geometry.get(
        (
            tuple(int(value) for value in source_center),
            source_axis,
            source_side,
        )
    )
    if source is None:
        raise ValueError(
            f"Symmetry did not map sticker {identity.label} to a valid sticker"
        )
    return source.label


def _symmetry_observation_permutations(
    identities: tuple[StickerIdentity, ...],
) -> tuple[tuple[np.ndarray, dict[str, str]], ...]:
    """Precompute target-label -> raw-label permutations for cube symmetries."""

    identities_by_geometry = {
        (identity.cubie_index, identity.axis, identity.side): identity
        for identity in identities
    }
    return tuple(
        (
            symmetry,
            {
                identity.label: sticker_identity_label_under_symmetry(
                    identity,
                    symmetry,
                    identities_by_geometry,
                )
                for identity in identities
            },
        )
        for symmetry in CUBE_SYMMETRY_MATRICES
    )


def _update_sticker_color_posteriors(
    priors: dict[str, np.ndarray],
    observations: dict[str, np.ndarray],
    global_profiles: dict[str, GlobalColorProfile],
    observation_weights: dict[str, float] | None = None,
) -> None:
    """Update identity color posteriors from one corrected frame."""

    for label, rgb in observations.items():
        prior = priors.get(label)
        if prior is None:
            continue
        evidence = _rgb_observation_color_evidence(rgb, global_profiles)
        weight = 1.0 if observation_weights is None else observation_weights.get(label, 1.0)
        posterior = prior * _apply_observation_weight(evidence, weight)
        posterior_sum = float(np.sum(posterior))
        if posterior_sum > 1e-12 and np.isfinite(posterior_sum):
            priors[label] = posterior / posterior_sum


def _score_symmetry_observations(
    raw_observations: dict[str, np.ndarray],
    permutation: dict[str, str],
    priors: dict[str, np.ndarray],
    global_profiles: dict[str, GlobalColorProfile],
    raw_observation_weights: dict[str, float] | None = None,
) -> tuple[float, int, int, dict[str, np.ndarray], dict[str, float]]:
    """Return weighted likelihood scores, observations, and their weights."""

    log_likelihoods: list[float] = []
    low_probability_count = 0
    remapped: dict[str, np.ndarray] = {}
    remapped_weights: dict[str, float] = {}
    for target_label, source_label in permutation.items():
        rgb = raw_observations.get(source_label)
        if rgb is None:
            continue
        prior = priors[target_label]
        evidence = _rgb_observation_color_evidence(rgb, global_profiles)
        weight = (
            1.0
            if raw_observation_weights is None
            else raw_observation_weights.get(source_label, 1.0)
        )
        weighted_evidence = _apply_observation_weight(evidence, weight)
        predictive_probability = float(np.dot(prior, weighted_evidence))
        log_likelihoods.append(np.log(max(predictive_probability, 1e-12)))
        if (
            predictive_probability
            < POSE_SYMMETRY_RECOVERY_LOW_PREDICTIVE_PROBABILITY
        ):
            low_probability_count += 1
        remapped[target_label] = np.asarray(rgb, dtype=np.uint8).copy()
        remapped_weights[target_label] = float(np.clip(weight, 0.0, 1.0))
    if not log_likelihoods:
        return float("-inf"), 0, 0, remapped, remapped_weights
    return (
        float(np.mean(log_likelihoods)),
        len(log_likelihoods),
        low_probability_count,
        remapped,
        remapped_weights,
    )


def recover_cached_pose_symmetries(
    cache: list[NeuralFrameData],
    trace: StickerColorTrace,
    *,
    debug: bool = False,
    progress: bool = False,
    refine_pose: bool = False,
    recover_orientation: bool = True,
    recognize_moves: bool = False,
    max_turn_face_candidates: int = PNP_TURN_FACE_CANDIDATES,
    record_all_turn_candidates: bool = False,
    pose_motion_filter: PoseMotionFilter | None = None,
    turn_candidate_log: Path | None = None,
) -> StickerColorTrace:
    """Recover cached pose candidates and optionally fit layer turns.

    The neural pass and its initial pose/color samples are already complete
    when this runs.  Global color profiles are therefore fixed first, then a
    sequential identity posterior scores all 24 cube symmetries for each cached
    frame. When ``refine_pose`` is enabled, the retained cheap candidates are
    refined here, after the neural and candidate-generation passes have
    completed. Color observations are retained for classification, but do not
    influence geometric pose or turn fitting.
    """

    global_profiles, _global_observations = build_global_color_profiles(trace)
    identities = trace.identities
    permutations = _symmetry_observation_permutations(identities)
    identity_permutation = next(
        permutation
        for symmetry, permutation in permutations
        if np.array_equal(symmetry, np.eye(3, dtype=np.float64))
    )
    identity_symmetry_index = next(
        index
        for index, (symmetry, _permutation) in enumerate(permutations)
        if np.array_equal(symmetry, np.eye(3, dtype=np.float64))
    )
    priors = {
        identity.label: np.full(
            len(GLOBAL_COLOR_NAMES),
            1.0 / len(GLOBAL_COLOR_NAMES),
            dtype=np.float64,
        )
        for identity in identities
    }
    raw_by_frame = _trace_observations_by_frame(trace)
    raw_weights_by_frame = _trace_observation_weights_by_frame(trace)
    corrected_trace = StickerColorTrace.create()
    previous_candidate: StickerPnPCandidate | None = None

    total_records = len(cache)
    for record_index, record in enumerate(cache, start=1):
        if progress:
            print_headless_progress(
                "pose recovery",
                record_index,
                total_records,
            )
        raw_observations = raw_by_frame.get(record.frame_index, {})
        raw_observation_weights = raw_weights_by_frame.get(record.frame_index, {})
        candidate = record.pose_candidate
        if not isinstance(candidate, StickerPnPCandidate):
            corrected_trace.observe(record.frame_index, {})
            if turn_candidate_log is not None and recognize_moves:
                append_turn_candidate_log(turn_candidate_log, None, record.frame_index)
            continue
        candidate_hypotheses = tuple(getattr(record, "pose_candidates", ()))
        if not candidate_hypotheses:
            candidate_hypotheses = (candidate,)
        selected_candidate_index = next(
            (
                index
                for index, hypothesis in enumerate(candidate_hypotheses)
                if hypothesis is candidate
            ),
            0,
        )

        def refine_hypotheses(
            hypotheses: tuple[StickerPnPCandidate, ...],
        ) -> tuple[StickerPnPCandidate, ...]:
            refined_hypotheses: list[StickerPnPCandidate] = []
            for hypothesis in hypotheses:
                refined_hypotheses.append(
                    solve_pnp_from_low_error_assignments(
                        hypothesis,
                        # Keep pose recovery purely static.  A layer-turn
                        # search is applied only after the final static
                        # candidate has been selected below.
                        recover_layer_turns=False,
                        max_turn_face_candidates=max_turn_face_candidates,
                        record_all_turn_candidates=record_all_turn_candidates,
                    )
                )
            return tuple(refined_hypotheses)

        (
            current_log_likelihood,
            observation_count,
            low_probability_count,
            current_observations,
            current_observation_weights,
        ) = (
            _score_symmetry_observations(
                raw_observations,
                identity_permutation,
                priors,
                global_profiles,
                raw_observation_weights,
            )
        )
        if debug and observation_count:
            print(
                "pose symmetry frame "
                f"{record.frame_index}: obs={observation_count} "
                f"current_logp={current_log_likelihood:.4f} "
                f"low_probability_count={low_probability_count}",
                flush=True,
            )

        selected_candidate = candidate
        selected_hypotheses = candidate_hypotheses
        selected_observations = current_observations
        selected_observation_weights = current_observation_weights
        selected_log_likelihood = current_log_likelihood
        if recover_orientation and observation_count >= POSE_SYMMETRY_RECOVERY_MIN_OBSERVATIONS:
            scored = []
            for symmetry_index, (symmetry, permutation) in enumerate(permutations):
                (
                    log_likelihood,
                    count,
                    _low_count,
                    remapped_observations,
                    remapped_observation_weights,
                ) = (
                    _score_symmetry_observations(
                        raw_observations,
                        permutation,
                        priors,
                        global_profiles,
                        raw_observation_weights,
                    )
                )
                if count != observation_count:
                    continue
                transformed_candidate = transform_candidate_by_symmetry(
                    candidate_hypotheses[selected_candidate_index],
                    symmetry,
                )
                continuity_distance = (
                    pose_hypothesis_distance(
                        transformed_candidate,
                        (previous_candidate.rvec, previous_candidate.tvec),
                    )
                    if previous_candidate is not None
                    else 0.0
                )
                scored.append(
                    (
                        symmetry_index,
                        log_likelihood,
                        -continuity_distance,
                        transformed_candidate,
                        remapped_observations,
                        remapped_observation_weights,
                    )
                )
            if scored:
                best_log_likelihood = max(item[1] for item in scored)
                likelihood_ties = [
                    item
                    for item in scored
                    if best_log_likelihood - item[1] <= 1e-9
                ]
                (
                    best_symmetry_index,
                    _best_log_likelihood,
                    _continuity,
                    best_candidate,
                    best_observations,
                    best_observation_weights,
                ) = max(
                    likelihood_ties,
                    key=lambda item: (
                        item[2],
                        int(item[0] == identity_symmetry_index),
                    ),
                )
                if debug:
                    print(
                        "  tested orientations:"
                        f" current_logp={selected_log_likelihood:.4f}",
                        flush=True,
                    )
                    for (
                        symmetry_index,
                        log_likelihood,
                        continuity_score,
                        _tested_candidate,
                        _tested_observations,
                        _tested_observation_weights,
                    ) in sorted(scored, key=lambda item: item[1], reverse=True):
                        print(
                            f"    symmetry={symmetry_index:02d} "
                            f"logp={log_likelihood:.4f} "
                            f"p={np.exp(log_likelihood):.4f} "
                            f"continuity={-continuity_score:.4f}"
                            f"{' *best' if symmetry_index == best_symmetry_index else ''}",
                            flush=True,
                        )
                selected_candidate = best_candidate
                selected_observations = best_observations
                selected_observation_weights = best_observation_weights
                selected_log_likelihood = best_log_likelihood

                # Keep the candidate set aligned with the selected symmetry.
                selected_symmetry = permutations[best_symmetry_index][0]
                selected_hypotheses = tuple(
                    transform_candidate_by_symmetry(hypothesis, selected_symmetry)
                    for hypothesis in candidate_hypotheses
                )

        if refine_pose:
            selected_hypotheses = refine_hypotheses(selected_hypotheses)
            selected_candidate = selected_hypotheses[selected_candidate_index]

        if recognize_moves:
            selected_candidate = solve_pnp_from_low_error_assignments(
                selected_candidate,
                recover_layer_turns=True,
                max_turn_face_candidates=max_turn_face_candidates,
                record_all_turn_candidates=record_all_turn_candidates,
            )
        selected_candidate = replace(
            selected_candidate,
            orientation_anchor=orientation_anchor_for_candidate(selected_candidate),
        )
        selected_candidate = apply_temporal_pose_filter(
            selected_candidate,
            pose_motion_filter,
        )
        record.pose_diagnostic_candidate = None
        if not pose_reprojection_error_is_acceptable(selected_candidate):
            # Keep the color trace available for the independent classifier,
            # but do not let an over-threshold geometric fit become the pose
            # shown in the final pass.
            record.pose_candidate = None
            record.pose_candidates = ()
            record.pose_diagnostic_candidate = selected_candidate
            previous_candidate = None
            corrected_trace.observe(
                record.frame_index,
                selected_observations,
                observation_weights=selected_observation_weights,
            )
            _update_sticker_color_posteriors(
                priors,
                selected_observations,
                global_profiles,
                selected_observation_weights,
            )
            if turn_candidate_log is not None and recognize_moves:
                append_turn_candidate_log(
                    turn_candidate_log,
                    selected_candidate,
                    record.frame_index,
                )
            continue
        record.pose_candidate = selected_candidate
        record.pose_candidates = selected_hypotheses
        previous_candidate = selected_candidate
        corrected_trace.observe(
            record.frame_index,
            selected_observations,
            observation_weights=selected_observation_weights,
        )
        _update_sticker_color_posteriors(
            priors,
            selected_observations,
            global_profiles,
            selected_observation_weights,
        )
        if turn_candidate_log is not None and recognize_moves:
            append_turn_candidate_log(
                turn_candidate_log,
                selected_candidate,
                record.frame_index,
            )
    if progress:
        print()

    return corrected_trace


def sample_profile_polyline(polyline_oklab: np.ndarray, count: int = 48) -> np.ndarray:
    """Sample a polyline by arc length for profile-to-profile comparison."""

    polyline = np.asarray(polyline_oklab, dtype=np.float64).reshape(-1, 3)
    if len(polyline) < 2:
        return polyline
    lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cumulative[-1])
    if total <= 1e-12:
        return np.repeat(polyline[:1], max(1, int(count)), axis=0)
    targets = np.linspace(0.0, total, max(2, int(count)))
    segments = np.clip(np.searchsorted(cumulative, targets, side="right") - 1, 0, len(polyline) - 2)
    fractions = (targets - cumulative[segments]) / np.maximum(lengths[segments], 1e-12)
    return polyline[segments] + fractions[:, None] * (
        polyline[segments + 1] - polyline[segments]
    )


def sticker_color_profile_distance(
    first: StickerColorProfile,
    second: StickerColorProfile,
) -> float:
    """Return a symmetric OKLab distance between two fitted profiles."""

    first_samples = sample_profile_polyline(first.polyline_oklab)
    second_samples = sample_profile_polyline(second.polyline_oklab)
    first_to_second = _piecewise_profile_distances(
        first_samples,
        second.polyline_oklab,
    )
    second_to_first = _piecewise_profile_distances(
        second_samples,
        first.polyline_oklab,
    )
    return float(0.5 * (np.mean(first_to_second) + np.mean(second_to_first)))


def build_pairwise_sticker_color_posteriors(
    trace: StickerColorTrace,
    likelihoods_by_frame: dict[int, dict[str, dict[str, float]]],

) -> dict[str, dict[str, float]]:
    """Build color evidence by grouping corrected sticker trajectories first.

    The global posterior is used only to name each anonymous trajectory group.
    Same-color relationships come from the raw corrected RGB trajectories,
    after normalizing away brightness and retaining saturation as a separate
    cue.  This prevents a bad global posterior for one sticker from being
    propagated directly to its peers.
    """

    final_posteriors = final_sticker_posteriors(likelihoods_by_frame)
    trajectory_by_label = {
        identity.label: _sticker_chromatic_trajectory_points(
            trace,
            identity.label,
        )
        for identity in trace.identities
        if identity.label in final_posteriors
    }
    trajectory_by_label = {
        label: points
        for label, points in trajectory_by_label.items()
        if len(points) > 0
    }
    if len(trajectory_by_label) < 2:
        return {
            label: {
                color_name: float(probability)
                for color_name, probability in probabilities.items()
            }
            for label, probabilities in final_posteriors.items()
        }

    labels = tuple(trajectory_by_label)
    parent = {label: label for label in labels}

    def find(label: str) -> str:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    def union(first: str, second: str) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for first_index, first_label in enumerate(labels):
        for second_label in labels[first_index + 1 :]:
            distance = _sticker_trajectory_distance(
                trajectory_by_label[first_label],
                trajectory_by_label[second_label],
            )
            if distance <= PAIRWISE_TRAJECTORY_DISTANCE_THRESHOLD:
                union(first_label, second_label)

    groups: dict[str, list[str]] = {}
    for label in labels:
        groups.setdefault(find(label), []).append(label)

    posteriors: dict[str, dict[str, float]] = {}
    for group_labels in groups.values():
        group_posterior = np.zeros(len(GLOBAL_COLOR_NAMES), dtype=np.float64)
        posterior_count = 0
        for label in group_labels:
            probabilities = final_posteriors.get(label)
            if not probabilities:
                continue
            group_posterior += np.asarray(
                [
                    probabilities.get(color_name, 0.0)
                    for color_name in GLOBAL_COLOR_NAMES
                ],
                dtype=np.float64,
            )
            posterior_count += 1
        if posterior_count == 0:
            continue
        group_posterior /= float(posterior_count)
        group_sum = float(np.sum(group_posterior))
        if group_sum > 1e-12:
            group_posterior /= group_sum
        group_probabilities = {
            color_name: float(probability)
            for color_name, probability in zip(
                GLOBAL_COLOR_NAMES,
                group_posterior,
                strict=True,
            )
        }
        posteriors.update({label: dict(group_probabilities) for label in group_labels})

    posteriors.update(
        {
            label: {
                color_name: float(probability)
                for color_name, probability in probabilities.items()
            }
            for label, probabilities in final_posteriors.items()
            if label not in posteriors and probabilities
        }
    )
    return posteriors


def classify_sticker_colors_by_pairwise_trajectory(
    trace: StickerColorTrace,
    likelihoods_by_frame: dict[int, dict[str, dict[str, float]]],
) -> dict[str, str]:
    """Assign final colors from pairwise trajectory-grouped posteriors."""

    posteriors = build_pairwise_sticker_color_posteriors(
        trace,
        likelihoods_by_frame,
    )
    return {
        label: max(probabilities.items(), key=lambda item: item[1])[0]
        for label, probabilities in posteriors.items()
        if probabilities
    }


def _sticker_chromatic_trajectory_points(
    trace: StickerColorTrace,
    sticker_label: str,
) -> np.ndarray:
    """Return brightness-normalized, saturation-aware trajectory samples."""

    entries = trace.observations.get(sticker_label, ())
    if not entries:
        return np.empty((0, 4), dtype=np.float64)
    rgb = np.asarray([value for _frame, value in entries], dtype=np.float64)
    brightness = np.sum(rgb, axis=1)
    saturation = (np.max(rgb, axis=1) - np.min(rgb, axis=1)) / np.maximum(
        brightness,
        1.0,
    )
    informative = (brightness > 1.0) & (
        saturation >= PAIRWISE_TRAJECTORY_SATURATION_FLOOR
    )
    if not np.any(informative):
        return np.empty((0, 4), dtype=np.float64)
    chromatic = rgb[informative] / np.maximum(brightness[informative, None], 1.0)
    points = np.column_stack((chromatic, saturation[informative]))
    if len(points) <= PAIRWISE_TRAJECTORY_MAX_POINTS:
        return points
    indices = np.linspace(
        0,
        len(points) - 1,
        PAIRWISE_TRAJECTORY_MAX_POINTS,
        dtype=np.int32,
    )
    return points[indices]


def _sticker_trajectory_distance(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Return a symmetric nearest-neighbor distance between trajectories."""

    first = np.asarray(first, dtype=np.float64).reshape(-1, 4)
    second = np.asarray(second, dtype=np.float64).reshape(-1, 4)
    if len(first) == 0 or len(second) == 0:
        return float("inf")
    scaled_difference = np.asarray((1.0, 1.0, 1.0, PAIRWISE_TRAJECTORY_SATURATION_WEIGHT))
    distances = np.linalg.norm(
        (first[:, None, :] - second[None, :, :]) * scaled_difference,
        axis=2,
    )
    return float(
        0.5
        * (
            np.mean(np.min(distances, axis=1))
            + np.mean(np.min(distances, axis=0))
        )
    )


def _global_profile_reference(profile: GlobalColorProfile) -> StickerColorProfile | None:
    """Return the fitted global profile, falling back to its pure-color prior."""

    return profile.fitted if profile.fitted is not None else profile.prior


def _sticker_to_global_profile_distance(
    sticker_profile: StickerColorProfile,
    global_profile: GlobalColorProfile,
) -> float:
    reference = _global_profile_reference(global_profile)
    if reference is None:
        return float("inf")
    return sticker_color_profile_distance(sticker_profile, reference)


def classify_center_color_names(
    trace: StickerColorTrace,
    global_profiles: dict[str, GlobalColorProfile] | None = None,
) -> dict[str, str]:
    """Assign centers using global profiles under the legal color scheme."""

    if global_profiles is None:
        global_profiles, _global_observations = build_global_color_profiles(trace)

    center_profiles: dict[str, StickerColorProfile] = {}
    center_identities = (
        identity for identity in trace.identities if identity.category == "center"
    )
    for identity in center_identities:
        profile = fit_sticker_color_profile(trace, identity.label)
        if profile is not None:
            center_profiles[identity.face] = profile
    if not center_profiles:
        return {}

    best_assignment = None
    best_cost = float("inf")
    for assignment in legal_center_color_assignments():
        cost = sum(
            _sticker_to_global_profile_distance(
                center_profiles[face],
                global_profiles[assignment[face]],
            )
            for face in center_profiles
        )
        if cost < best_cost:
            best_cost = cost
            best_assignment = assignment
    if best_assignment is None:
        return {}
    return {
        identity.label: best_assignment[identity.face]
        for identity in trace.identities
        if identity.category == "center" and identity.face in center_profiles
    }


def _legal_cubie_color_tuple(colors: tuple[str, ...]) -> bool:
    if len(set(colors)) != len(colors):
        return False
    return all(
        frozenset((first, second)) not in OPPOSITE_COLOR_PAIRS
        for index, first in enumerate(colors)
        for second in colors[index + 1 :]
    )


def legal_edge_color_combinations(
    color_names: tuple[str, ...] = tuple(PURE_COLOR_RGB),
) -> tuple[tuple[str, str], ...]:
    """Return the 12 unique physical edge pieces for the six colors."""

    return tuple(
        combination
        for combination in combinations(color_names, 2)
        if _legal_cubie_color_tuple(combination)
    )


def legal_corner_color_combinations(
    color_names: tuple[str, ...] = tuple(PURE_COLOR_RGB),
) -> tuple[tuple[str, str, str], ...]:
    """Return the 8 unique physical corner pieces for the six colors."""

    return tuple(
        combination
        for combination in combinations(color_names, 3)
        if _legal_cubie_color_tuple(combination)
    )


def corner_color_assignment_is_chiral(
    identities: list[StickerIdentity],
    ordered_colors: tuple[str, ...],
    color_face_by_name: dict[str, str] | None = None,
) -> bool:
    """Return whether a corner assignment preserves proper cube orientation."""

    if len(identities) != 3 or len(ordered_colors) != 3:
        return False
    if color_face_by_name is None:
        color_face_by_name = {
            color_name: face
            for face, color_name in STANDARD_FACE_COLOR_NAMES.items()
        }
    target_face_frame = np.column_stack(
        [FACE_VECTOR_BY_NAME[identity.face] for identity in identities]
    )
    source_faces = [color_face_by_name.get(color_name) for color_name in ordered_colors]
    if any(face is None for face in source_faces):
        # The corner may include a color whose center has not been observed.
        # The distinct/non-opposite check still applies, but chirality cannot
        # be evaluated until all three color-to-face mappings are known.
        return True
    source_color_frame = np.column_stack(
        [FACE_VECTOR_BY_NAME[face] for face in source_faces]
    )
    # The assignment maps the source piece's face-color frame onto this
    # position's sticker-face frame.  A determinant of +1 is a proper cube
    # rotation; -1 is the reflected, impossible chirality.
    return float(np.linalg.det(target_face_frame @ source_color_frame.T)) > 0.0


def _sticker_profile_confidence(profile: StickerColorProfile) -> float:
    sample_factor = min(1.0, np.sqrt(float(profile.sample_count)) / 6.0)
    residual_factor = float(np.exp(-40.0 * max(0.0, profile.residual)))
    return float(np.clip(sample_factor * residual_factor, 0.20, 1.0))


def classify_sticker_color_names(
    trace: StickerColorTrace,
    center_color_names: dict[str, str] | None = None,
) -> dict[str, str]:
    """Classify all observed stickers from global profiles and legal cubies."""

    global_profiles, _global_observations = build_global_color_profiles(trace)

    center_color_names = (
        classify_center_color_names(trace, global_profiles)
        if center_color_names is None
        else dict(center_color_names)
    )
    if not center_color_names:
        return {}
    color_face_by_name = {}
    for identity in trace.identities:
        if identity.category != "center":
            continue
        color_name = center_color_names.get(identity.label)
        if color_name is not None:
            color_face_by_name[color_name] = identity.face
    color_names = tuple(PURE_COLOR_RGB)
    profile_by_label = {
        identity.label: fit_sticker_color_profile(trace, identity.label)
        for identity in trace.identities
        if identity.category != "center"
    }
    profile_distance_by_label = {
        label: {
            color_name: _sticker_to_global_profile_distance(
                profile,
                global_profiles[color_name],
            )
            for color_name in color_names
        }
        for label, profile in profile_by_label.items()
        if profile is not None
    }
    groups_by_category: dict[str, dict[tuple[int, int, int], list[StickerIdentity]]] = {
        "edge": {},
        "corner": {},
    }
    for identity in trace.identities:
        if identity.category != "center":
            groups_by_category[identity.category].setdefault(identity.cubie_index, []).append(identity)

    def solve_inventory(
        category: str,
        piece_combinations: tuple[tuple[str, ...], ...],
    ) -> dict[str, str]:
        groups = [
            identities
            for identities in groups_by_category[category].values()
            if any(
                profile_by_label.get(identity.label) is not None
                for identity in identities
            )
        ]
        for identities in groups:
            identities.sort(key=lambda identity: identity.face)
        states: dict[int, tuple[float, dict[str, str]]] = {0: (0.0, {})}
        for identities in groups:
            next_states: dict[int, tuple[float, dict[str, str]]] = {}
            for used_mask, (base_cost, base_assignment) in states.items():
                for piece_index, piece_colors in enumerate(piece_combinations):
                    piece_bit = 1 << piece_index
                    if used_mask & piece_bit:
                        continue
                    for ordered_colors in permutations(piece_colors):
                        if category == "corner" and not corner_color_assignment_is_chiral(
                            identities,
                            ordered_colors,
                            color_face_by_name,
                        ):
                            continue
                        option_cost = 0.0
                        option_assignment: dict[str, str] = {}
                        for identity, color in zip(
                            identities,
                            ordered_colors,
                            strict=True,
                        ):
                            profile = profile_by_label[identity.label]
                            if profile is None:
                                continue
                            option_cost += _sticker_profile_confidence(profile) * profile_distance_by_label[
                                identity.label
                            ][color]
                            option_assignment[identity.label] = color
                        total_cost = base_cost + option_cost
                        next_mask = used_mask | piece_bit
                        previous = next_states.get(next_mask)
                        if previous is None or total_cost < previous[0]:
                            next_states[next_mask] = (
                                total_cost,
                                {**base_assignment, **option_assignment},
                            )
            states = next_states
        if not states:
            return {}
        best_mask = max(states, key=lambda mask: mask.bit_count())
        return states[best_mask][1]

    edge_assignment = solve_inventory("edge", legal_edge_color_combinations(color_names))
    corner_assignment = solve_inventory("corner", legal_corner_color_combinations(color_names))
    return {**center_color_names, **edge_assignment, **corner_assignment}


def classify_sticker_colors_by_pairwise_cube_rules(
    trace: StickerColorTrace,
    likelihoods_by_frame: dict[int, dict[str, dict[str, float]]],
) -> dict[str, str]:
    """Resolve pairwise trajectory evidence under legal cube constraints.

    Global color profiles still provide the frame-by-frame evidence used for
    pose correction.  Final identity colors come from the pairwise trajectory
    evidence, with center rotations, unique physical pieces, and corner
    chirality enforced only at this final assignment step.
    """

    pairwise_posteriors = build_pairwise_sticker_color_posteriors(
        trace,
        likelihoods_by_frame,
    )
    if not pairwise_posteriors:
        return {}

    color_names = tuple(PURE_COLOR_RGB)
    color_cost_by_label = {
        label: {
            color_name: -float(np.log(max(probabilities.get(color_name, 0.0), 1e-12)))
            for color_name in color_names
        }
        for label, probabilities in pairwise_posteriors.items()
    }

    center_identities = [
        identity for identity in trace.identities if identity.category == "center"
    ]
    best_center_assignment: dict[str, str] | None = None
    best_center_cost = float("inf")
    for assignment in legal_center_color_assignments():
        cost = sum(
            color_cost_by_label[identity.label][assignment[identity.face]]
            for identity in center_identities
            if identity.label in color_cost_by_label
        )
        if cost < best_center_cost:
            best_center_cost = cost
            best_center_assignment = assignment
    if best_center_assignment is None:
        best_center_assignment = legal_center_color_assignments()[0]

    center_color_names = {
        identity.label: best_center_assignment[identity.face]
        for identity in center_identities
        if identity.label in color_cost_by_label
    }
    color_face_by_name = {
        color_name: face
        for face, color_name in best_center_assignment.items()
    }

    groups_by_category: dict[str, dict[tuple[int, int, int], list[StickerIdentity]]] = {
        "edge": {},
        "corner": {},
    }
    for identity in trace.identities:
        if identity.category != "center":
            groups_by_category[identity.category].setdefault(identity.cubie_index, []).append(identity)

    def solve_inventory(
        category: str,
        piece_combinations: tuple[tuple[str, ...], ...],
    ) -> dict[str, str]:
        groups = [
            identities
            for identities in groups_by_category[category].values()
            if any(identity.label in color_cost_by_label for identity in identities)
        ]
        for identities in groups:
            identities.sort(key=lambda identity: identity.face)

        states: dict[int, tuple[float, dict[str, str]]] = {0: (0.0, {})}
        for identities in groups:
            next_states: dict[int, tuple[float, dict[str, str]]] = {}
            for used_mask, (base_cost, base_assignment) in states.items():
                for piece_index, piece_colors in enumerate(piece_combinations):
                    piece_bit = 1 << piece_index
                    if used_mask & piece_bit:
                        continue
                    for ordered_colors in permutations(piece_colors):
                        if category == "corner" and not corner_color_assignment_is_chiral(
                            identities,
                            ordered_colors,
                            color_face_by_name,
                        ):
                            continue
                        option_cost = 0.0
                        option_assignment: dict[str, str] = {}
                        for identity, color in zip(identities, ordered_colors, strict=True):
                            costs = color_cost_by_label.get(identity.label)
                            if costs is None:
                                continue
                            option_cost += costs[color]
                            option_assignment[identity.label] = color
                        total_cost = base_cost + option_cost
                        next_mask = used_mask | piece_bit
                        previous = next_states.get(next_mask)
                        if previous is None or total_cost < previous[0]:
                            next_states[next_mask] = (
                                total_cost,
                                {**base_assignment, **option_assignment},
                            )
            states = next_states
        if not states:
            return {}
        best_mask = max(states, key=lambda mask: mask.bit_count())
        return states[best_mask][1]

    edge_assignment = solve_inventory("edge", legal_edge_color_combinations(color_names))
    corner_assignment = solve_inventory("corner", legal_corner_color_combinations(color_names))
    return {**center_color_names, **edge_assignment, **corner_assignment}


def cube_lattice_point_is_visible(
    object_point: np.ndarray,
    coord: tuple[int, int, int],
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> bool:
    """Return whether a cube-surface lattice point is camera-visible."""

    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    for axis, value in cube_lattice_boundary_faces(coord):
        normal = np.zeros(3, dtype=np.float64)
        normal[axis] = -1.0 if value == 0 else 1.0
        if float((rotation @ normal)[2]) >= 0.0:
            continue

        # A front-facing edge/corner can still be hidden by another cube face.
        # Test the ray from the camera center to the lattice point against the
        # cube interior, in the same object coordinate system as the PnP pose.
        point = np.asarray(object_point, dtype=np.float64).reshape(3)
        camera_origin = (
            -rotation.T @ np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        ).reshape(3)
        direction = point - camera_origin
        if float(np.linalg.norm(direction)) <= 1e-9:
            return True
        bounds_min = np.full(3, -CUBE_HALF_SIZE_MM, dtype=np.float64)
        bounds_max = np.full(3, CUBE_HALF_SIZE_MM, dtype=np.float64)
        t_enter = 0.0
        t_exit = 1.0
        for dimension in range(3):
            origin_value = float(camera_origin[dimension])
            direction_value = float(direction[dimension])
            if abs(direction_value) <= 1e-12:
                if not bounds_min[dimension] <= origin_value <= bounds_max[dimension]:
                    break
                continue
            near = (bounds_min[dimension] - origin_value) / direction_value
            far = (bounds_max[dimension] - origin_value) / direction_value
            if near > far:
                near, far = far, near
            t_enter = max(t_enter, float(near))
            t_exit = min(t_exit, float(far))
            if t_enter > t_exit:
                break
        else:
            if not (1e-5 < t_enter < 1.0 - 1e-4):
                return True
    return False


def visible_cube_lattice_object_points(
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> np.ndarray:
    object_points, _class_ids = visible_cube_lattice_points(rvec, tvec)
    return object_points


def visible_cube_lattice_points(
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    object_points = CUBE_LATTICE_OBJECT_POINTS_MM
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    camera_origin = (
        -rotation.T @ np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    ).reshape(3)

    camera_normals_z = np.einsum(
        "ij,nfj->nfi",
        rotation,
        CUBE_LATTICE_BOUNDARY_NORMALS,
    )[..., 2]
    front_facing = np.any(
        CUBE_LATTICE_BOUNDARY_FACE_MASK & (camera_normals_z < 0.0),
        axis=1,
    )

    points = np.asarray(object_points, dtype=np.float64)
    directions = points - camera_origin
    direction_is_zero = np.linalg.norm(directions, axis=1) <= 1e-9
    safe_directions = np.where(
        np.abs(directions) <= 1e-12,
        1.0,
        directions,
    )
    bounds_min = -CUBE_HALF_SIZE_MM
    bounds_max = CUBE_HALF_SIZE_MM
    near = (bounds_min - camera_origin) / safe_directions
    far = (bounds_max - camera_origin) / safe_directions
    slab_near = np.minimum(near, far)
    slab_far = np.maximum(near, far)
    parallel_inside = (
        (np.abs(directions) > 1e-12)
        | (
            (camera_origin >= bounds_min)
            & (camera_origin <= bounds_max)
        )
    )
    t_enter = np.maximum(0.0, np.max(slab_near, axis=1))
    t_exit = np.minimum(1.0, np.min(slab_far, axis=1))
    ray_intersects_cube = np.all(parallel_inside, axis=1) & (t_enter <= t_exit)
    hidden_by_front_face = (
        ray_intersects_cube
        & ~direction_is_zero
        & (t_enter > 1e-5)
        & (t_enter < 1.0 - 1e-4)
    )
    visible_mask = front_facing & ~hidden_by_front_face
    return (
        object_points[visible_mask],
        CUBE_LATTICE_CLASS_IDS[visible_mask],
    )


def transformed_cube_surface_triangles(
    layer_rotations: tuple["CubeLayerRotation", ...]
    | list["CubeLayerRotation"],
) -> np.ndarray:
    """Return triangles for the transformed exterior cubie surfaces."""

    triangles: list[np.ndarray] = []
    for x_index in range(3):
        for y_index in range(3):
            for z_index in range(3):
                center = cubelet_center_object_point(x_index, y_index, z_index)
                for axis, side in (
                    (axis, side)
                    for axis in range(3)
                    for side in (-1, 1)
                    if (
                        (axis == 0 and ((side == -1 and x_index == 0) or (side == 1 and x_index == 2)))
                        or (axis == 1 and ((side == -1 and y_index == 0) or (side == 1 and y_index == 2)))
                        or (axis == 2 and ((side == -1 and z_index == 0) or (side == 1 and z_index == 2)))
                    )
                ):
                    face = cubelet_face_object_points(center, axis, side)
                    _transformed_center, transformed_face = apply_cubelet_layer_rotations(
                        center,
                        face,
                        tuple(layer_rotations),
                    )
                    triangles.append(
                        np.asarray(
                            (transformed_face[0], transformed_face[1], transformed_face[2]),
                            dtype=np.float64,
                        )
                    )
                    triangles.append(
                        np.asarray(
                            (transformed_face[0], transformed_face[2], transformed_face[3]),
                            dtype=np.float64,
                        )
                    )
    if not triangles:
        return np.empty((0, 3, 3), dtype=np.float64)
    return np.asarray(triangles, dtype=np.float64)


def transformed_surface_occlusion_mask(
    object_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    layer_rotations: tuple["CubeLayerRotation", ...]
    | list["CubeLayerRotation"],
) -> np.ndarray:
    """Return transformed lattice points hidden by another exterior surface.

    This is a small 3-D ray/triangle test rather than a rendered depth pass.
    The target point itself is excluded with a small endpoint margin, so the
    face containing that point does not count as an occluder.
    """

    points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    triangles = transformed_cube_surface_triangles(layer_rotations)
    if len(triangles) == 0:
        return np.zeros(len(points), dtype=bool)

    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    camera_origin = (
        -rotation.T @ np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    ).reshape(3)
    directions = points - camera_origin
    edge_1 = triangles[:, 1] - triangles[:, 0]
    edge_2 = triangles[:, 2] - triangles[:, 0]
    ray_cross_edge_2 = np.cross(directions[:, None, :], edge_2[None, :, :])
    determinant = np.einsum(
        "mj,nmj->nm",
        edge_1,
        ray_cross_edge_2,
    )
    valid = np.abs(determinant) > 1e-9
    inverse_determinant = np.divide(
        1.0,
        determinant,
        out=np.zeros_like(determinant),
        where=valid,
    )
    origin_to_vertex = -triangles[None, :, 0] + camera_origin
    barycentric_u = inverse_determinant * np.sum(
        origin_to_vertex * ray_cross_edge_2,
        axis=2,
    )
    cross_q = np.cross(origin_to_vertex, edge_1[None, :, :])
    barycentric_v = inverse_determinant * np.sum(
        directions[:, None, :] * cross_q,
        axis=2,
    )
    distance = inverse_determinant * np.sum(
        edge_2[None, :, :] * cross_q,
        axis=2,
    )
    triangle_hits = (
        valid
        & (barycentric_u >= -1e-6)
        & (barycentric_v >= -1e-6)
        & (barycentric_u + barycentric_v <= 1.0 + 1e-6)
        & (distance > 1e-5)
        & (distance < 1.0 - 1e-4)
    )
    return np.any(triangle_hits, axis=1)


def visible_transformed_cube_lattice_points(
    rvec: np.ndarray,
    tvec: np.ndarray,
    layer_rotations: tuple["CubeLayerRotation", ...] | list["CubeLayerRotation"],
) -> tuple[np.ndarray, np.ndarray]:
    """Return visible lattice points after applying layer turns.

    Visibility must be evaluated after the layer geometry moves. Reusing the
    pre-turn visible subset can permanently remove an edge/corner target that
    becomes visible because of the turn.
    """

    base_points = np.asarray(CUBE_LATTICE_OBJECT_POINTS_MM, dtype=np.float64)
    points = apply_cube_layer_rotations(base_points, layer_rotations).astype(np.float64)
    boundary_normals = np.asarray(
        CUBE_LATTICE_BOUNDARY_NORMALS,
        dtype=np.float64,
    ).copy()
    for layer_rotation in layer_rotations:
        axis = int(layer_rotation.axis)
        side = int(layer_rotation.side)
        if axis not in (0, 1, 2) or side not in (-1, 1):
            continue
        mask = cube_layer_mask(
            base_points,
            axis,
            side,
            layer_rotation.layer_count,
        )
        rotation_vector = np.zeros((3, 1), dtype=np.float64)
        rotation_vector[axis, 0] = side * np.deg2rad(
            float(layer_rotation.angle_degrees)
        )
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        boundary_normals[mask] = np.einsum(
            "ij,nfj->nfi",
            rotation_matrix,
            boundary_normals[mask],
        )

    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    camera_origin = (
        -rotation.T @ np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    ).reshape(3)
    camera_normals_z = np.einsum(
        "ij,nfj->nfi",
        rotation,
        boundary_normals,
    )[..., 2]
    front_facing = np.any(
        CUBE_LATTICE_BOUNDARY_FACE_MASK & (camera_normals_z < 0.0),
        axis=1,
    )

    directions = points - camera_origin
    direction_is_zero = np.linalg.norm(directions, axis=1) <= 1e-9
    safe_directions = np.where(
        np.abs(directions) <= 1e-12,
        1.0,
        directions,
    )
    bounds_min = -CUBE_HALF_SIZE_MM
    bounds_max = CUBE_HALF_SIZE_MM
    near = (bounds_min - camera_origin) / safe_directions
    far = (bounds_max - camera_origin) / safe_directions
    slab_near = np.minimum(near, far)
    slab_far = np.maximum(near, far)
    parallel_inside = (
        (np.abs(directions) > 1e-12)
        | (
            (camera_origin >= bounds_min)
            & (camera_origin <= bounds_max)
        )
    )
    t_enter = np.maximum(0.0, np.max(slab_near, axis=1))
    t_exit = np.minimum(1.0, np.min(slab_far, axis=1))
    ray_intersects_cube = np.all(parallel_inside, axis=1) & (t_enter <= t_exit)
    if layer_rotations:
        hidden_by_transformed_surface = transformed_surface_occlusion_mask(
            points,
            rvec,
            tvec,
            layer_rotations,
        )
        visible_mask = front_facing & ~hidden_by_transformed_surface
    else:
        hidden_by_front_face = (
            ray_intersects_cube
            & ~direction_is_zero
            & (t_enter > 1e-5)
            & (t_enter < 1.0 - 1e-4)
        )
        visible_mask = front_facing & ~hidden_by_front_face
    return (
        points[visible_mask].astype(np.float32),
        CUBE_LATTICE_CLASS_IDS[visible_mask],
    )


def huber_loss(
    distances: np.ndarray,
    delta: float = 4.0,
) -> np.ndarray:
    distances = np.asarray(distances, dtype=np.float64)
    return np.where(
        distances <= delta,
        0.5 * distances * distances,
        delta * (distances - 0.5 * delta),
    )


def observed_keypoint_reprojection_nearest_assignment(
    observed_points: np.ndarray,
    projected_points: np.ndarray,
    image_size: tuple[int, int],
    outlier_distance: float = 16.0,
    observed_class_ids: np.ndarray | None = None,
    projected_class_ids: np.ndarray | None = None,
    class_mismatch_penalty: float | None = None,
    score_upper_bound: float = float("inf"),
) -> tuple[float, np.ndarray, np.ndarray]:
    """Assign every observation independently to its cheapest cube point.

    Unlike the Hungarian assignment, several observations may select the same
    projected point.  This is useful as a permissive hypothesis score while
    searching pose space; repeated 3-D targets are deduplicated before any
    solvePnP refinement.
    """

    # ``class_mismatch_penalty`` is retained only for API compatibility.  A
    # known class mismatch is now invalid regardless of its former weight.
    del image_size, outlier_distance, class_mismatch_penalty
    observed = np.asarray(observed_points, dtype=np.float64).reshape(-1, 2)
    projected = np.asarray(projected_points, dtype=np.float64).reshape(-1, 2)
    empty = np.empty((0,), dtype=np.int32)
    if len(observed) == 0 or len(projected) == 0:
        return float("inf"), empty, empty
    if not np.isfinite(observed).all() or not np.isfinite(projected).all():
        return float("inf"), empty, empty

    distances = np.linalg.norm(observed[:, None, :] - projected[None, :, :], axis=2)
    costs = huber_loss(distances)
    if observed_class_ids is not None and projected_class_ids is not None:
        observed_classes = np.asarray(observed_class_ids, dtype=np.int32).reshape(-1)
        projected_classes = np.asarray(projected_class_ids, dtype=np.int32).reshape(-1)
        if len(observed_classes) != len(observed) or len(projected_classes) != len(projected):
            return float("inf"), empty, empty
        known_observed = observed_classes[:, None] >= 0
        known_projected = projected_classes[None, :] >= 0
        mismatch = (
            known_observed
            & known_projected
            & (observed_classes[:, None] != projected_classes[None, :])
        )
        # A known keypoint class is an identity constraint, not a preference.
        # Unknown classes remain wildcards so partially classified inputs can
        # still be assigned geometrically.
        costs = np.where(mismatch, float("inf"), costs)

    valid_rows = np.isfinite(costs).any(axis=1)
    if not np.all(valid_rows):
        return float("inf"), empty, empty

    columns = np.argmin(costs, axis=1).astype(np.int32)
    rows = np.arange(len(observed), dtype=np.int32)
    score = float(np.mean(costs[rows, columns]))
    if np.isfinite(score_upper_bound) and score >= float(score_upper_bound):
        return float("inf"), empty, empty
    return score, rows, columns


def pose_hypothesis_score(
    observed_points: np.ndarray,
    projected_points: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    observed_class_ids: np.ndarray | None,
    projected_class_ids: np.ndarray | None,
) -> float:
    """Score all assigned observations, with extra weight on the error tail."""

    observed = np.asarray(observed_points, dtype=np.float64).reshape(-1, 2)
    projected = np.asarray(projected_points, dtype=np.float64).reshape(-1, 2)
    distances = np.linalg.norm(observed[rows] - projected[columns], axis=1)
    if not len(distances):
        return float("inf")
    geometric_score = float(np.mean(huber_loss(distances)))
    tail_score = float(np.percentile(distances, 90))
    if observed_class_ids is not None and projected_class_ids is not None:
        observed_classes = np.asarray(observed_class_ids, dtype=np.int32).reshape(-1)
        projected_classes = np.asarray(projected_class_ids, dtype=np.int32).reshape(-1)
        if (
            len(observed_classes) <= int(np.max(rows, initial=-1))
            or len(projected_classes) <= int(np.max(columns, initial=-1))
        ):
            return float("inf")
        known = (
            (observed_classes[rows] >= 0)
            & (projected_classes[columns] >= 0)
        )
        if np.any(known):
            if np.any(
                observed_classes[rows][known]
                != projected_classes[columns][known]
            ):
                return float("inf")
    return geometric_score + PNP_ASSIGNMENT_TAIL_WEIGHT * tail_score


def pivot_residual_consistency_score(
    observed_points: np.ndarray,
    projected_points: np.ndarray,
    object_points: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    fixed_rows: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    observed_class_ids: np.ndarray | None = None,
    projected_class_ids: np.ndarray | None = None,
    use_class_aware_assignment: bool = False,
) -> float:
    """Score whether assignments fit one pose correction around a pivot.

    The correction is deliberately estimated from the fixed sticker anchors,
    not from the full assignment.  Otherwise a few bad matches can pull the
    correction toward themselves and make the wrong correspondence look
    self-consistent.  Once that anchor-only correction is estimated, every
    assigned point is scored against the same corrected pose.
    """

    if len(fixed_rows) < 3 or len(rows) == 0:
        return 0.0
    assigned_objects = np.asarray(object_points, dtype=np.float32)[columns]
    assigned_observed = np.asarray(observed_points, dtype=np.float64)[rows]
    assigned_projected = np.asarray(projected_points, dtype=np.float64)[columns]
    try:
        _projected, jacobian = cv2.projectPoints(
            assigned_objects,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
    except cv2.error:
        return float("inf")
    if jacobian is None or jacobian.shape[1] < 6:
        return float("inf")
    jacobian = np.asarray(jacobian[:, :6], dtype=np.float64)
    residual = (assigned_observed - assigned_projected).reshape(-1)
    if not np.isfinite(jacobian).all() or not np.isfinite(residual).all():
        return float("inf")

    anchor_mask = np.isin(rows, np.asarray(fixed_rows, dtype=np.int32))
    anchor_indices = np.flatnonzero(anchor_mask)
    if len(anchor_indices) < 3:
        return float("inf")
    anchor_rows = rows[anchor_indices]
    anchor_columns = columns[anchor_indices]
    anchor_object_points = np.asarray(object_points, dtype=np.float32)[anchor_columns]
    anchor_observed_points = np.asarray(observed_points, dtype=np.float32)[anchor_rows]

    # Use the anchor sticker as the pivot.  A local nonlinear correction is
    # more accurate than applying one first-order Jacobian step, especially
    # for a sticker near the edge of the image.  The free assignments never
    # participate in this correction, so an incorrect match cannot drag the
    # candidate pose toward itself.
    try:
        pivot_rvec, pivot_tvec = cv2.solvePnPRefineLM(
            anchor_object_points,
            anchor_observed_points,
            camera_matrix,
            dist_coeffs,
            np.asarray(rvec, dtype=np.float64).reshape(3, 1).copy(),
            np.asarray(tvec, dtype=np.float64).reshape(3, 1).copy(),
        )
        pivot_rvec = np.asarray(pivot_rvec, dtype=np.float64).reshape(3, 1)
        pivot_tvec = np.asarray(pivot_tvec, dtype=np.float64).reshape(3, 1)
        if (
            not np.isfinite(pivot_rvec).all()
            or not np.isfinite(pivot_tvec).all()
            or float(pivot_tvec[2, 0]) <= 0.0
        ):
            raise cv2.error("invalid pivot pose")
        pivot_projected, _ = cv2.projectPoints(
            assigned_objects,
            pivot_rvec,
            pivot_tvec,
            camera_matrix,
            dist_coeffs,
        )
        pivot_projected = pivot_projected.reshape(-1, 2)
        correction_rvec, _ = cv2.Rodrigues(
            cv2.Rodrigues(pivot_rvec)[0] @ cv2.Rodrigues(rvec)[0].T
        )
        correction = np.concatenate(
            (
                np.asarray(correction_rvec, dtype=np.float64).reshape(3),
                (pivot_tvec - np.asarray(tvec, dtype=np.float64).reshape(3, 1)).reshape(3),
            )
        )
    except (cv2.error, np.linalg.LinAlgError):
        # Keep a first-order fallback for degenerate three-point anchors.
        anchor_jacobian = jacobian.reshape(-1, 2, 6)[anchor_indices].reshape(-1, 6)
        anchor_residual = residual.reshape(-1, 2)[anchor_indices].reshape(-1)
        correction = np.zeros(6, dtype=np.float64)
        for _iteration in range(3):
            anchor_residual_vectors = anchor_residual.reshape(-1, 2) - (
                anchor_jacobian @ correction
            ).reshape(-1, 2)
            residual_magnitudes = np.linalg.norm(anchor_residual_vectors, axis=1)
            robust_weights = np.minimum(
                1.0,
                PNP_PIVOT_HUBER_DELTA / np.maximum(residual_magnitudes, 1e-6),
            )
            weights = np.repeat(robust_weights, 2)
            weighted_jacobian = anchor_jacobian * np.sqrt(weights[:, None])
            weighted_residual = anchor_residual * np.sqrt(weights)
            correction = np.linalg.lstsq(
                weighted_jacobian,
                weighted_residual,
                rcond=None,
            )[0]
        residual_vectors = residual.reshape(-1, 2) - (
            jacobian @ correction
        ).reshape(-1, 2)
    else:
        residual_vectors = (
            np.asarray(observed_points, dtype=np.float64)[rows]
            - pivot_projected
        )

    residual_magnitudes = np.linalg.norm(residual_vectors, axis=1)
    if not np.isfinite(residual_magnitudes).all():
        return float("inf")
    consistency_error = float(
        np.mean(huber_loss(residual_magnitudes, PNP_PIVOT_HUBER_DELTA))
        + PNP_PIVOT_P90_WEIGHT * np.percentile(residual_magnitudes, 90)
        + PNP_PIVOT_P95_WEIGHT * np.percentile(residual_magnitudes, 95)
        + PNP_PIVOT_MAX_WEIGHT * np.max(residual_magnitudes)
    )
    if (
        use_class_aware_assignment
        and observed_class_ids is not None
        and projected_class_ids is not None
    ):
        observed_classes = np.asarray(observed_class_ids, dtype=np.int32).reshape(-1)
        projected_classes = np.asarray(projected_class_ids, dtype=np.int32).reshape(-1)
        if (
            len(observed_classes) != len(observed_points)
            or len(projected_classes) != len(projected_points)
        ):
            return float("inf")
        assigned_observed_classes = observed_classes[rows]
        assigned_projected_classes = projected_classes[columns]
        known = (
            (assigned_observed_classes >= 0)
            & (assigned_projected_classes >= 0)
        )
        if np.any(known):
            if np.any(
                assigned_observed_classes[known]
                != assigned_projected_classes[known]
            ):
                return float("inf")
    # A correction that is too large is not a local pivot adjustment anymore,
    # even if the linear model happens to explain the observations.
    correction_size = float(
        np.linalg.norm(correction[:3]) / 0.1
        + np.linalg.norm(correction[3:]) / 20.0
    )
    return consistency_error + max(0.0, correction_size - 1.0) * 8.0


def evaluate_pose_hypothesis(
    rvec: np.ndarray,
    tvec: np.ndarray,
    observed_points: np.ndarray,
    observed_class_ids: np.ndarray | None,
    image_size: tuple[int, int],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    score_upper_bound: float = float("inf"),
    fixed_observed_indices: np.ndarray | None = None,
    fixed_object_points: np.ndarray | None = None,
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Project and assign one pose hypothesis against the full observation set."""

    object_points, projected_class_ids = visible_cube_lattice_points(rvec, tvec)
    try:
        projected_points, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
    except cv2.error:
        empty_2d = np.empty((0, 2), dtype=np.float64)
        empty_3d = np.empty((0, 3), dtype=np.float32)
        empty = np.empty((0,), dtype=np.int32)
        return float("inf"), empty, empty, empty_2d, empty_3d, empty_2d
    projected_points = projected_points.reshape(-1, 2)
    observed = np.asarray(observed_points, dtype=np.float64).reshape(-1, 2)
    use_class_aware_assignment = PNP_ASSIGNMENT_MODE == "class-aware"
    fixed_rows = np.empty((0,), dtype=np.int32)
    fixed_columns = np.empty((0,), dtype=np.int32)
    if fixed_observed_indices is not None or fixed_object_points is not None:
        if fixed_observed_indices is None or fixed_object_points is None:
            return (
                float("inf"),
                fixed_rows,
                fixed_columns,
                projected_points,
                object_points,
                projected_class_ids,
            )
        fixed_rows = np.asarray(fixed_observed_indices, dtype=np.int32).reshape(-1)
        fixed_objects = np.asarray(fixed_object_points, dtype=np.float32).reshape(-1, 3)
        if (
            len(fixed_rows) != len(fixed_objects)
            or len(fixed_rows) == 0
            or np.any(fixed_rows < 0)
            or np.any(fixed_rows >= len(observed))
        ):
            return (
                float("inf"),
                fixed_rows,
                fixed_columns,
                projected_points,
                object_points,
                projected_class_ids,
            )
        fixed_columns_list: list[int] = []
        for fixed_object in fixed_objects:
            distances = np.linalg.norm(
                object_points.astype(np.float64) - fixed_object.astype(np.float64),
                axis=1,
            )
            column = int(np.argmin(distances))
            if float(distances[column]) > 1e-4 or column in fixed_columns_list:
                return (
                    float("inf"),
                    np.empty((0,), dtype=np.int32),
                    np.empty((0,), dtype=np.int32),
                    projected_points,
                    object_points,
                    projected_class_ids,
                )
            fixed_columns_list.append(column)
        fixed_columns = np.asarray(fixed_columns_list, dtype=np.int32)
        if len(np.unique(fixed_rows)) != len(fixed_rows):
            return (
                float("inf"),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
                projected_points,
                object_points,
                projected_class_ids,
            )

    free_rows = np.setdiff1d(
        np.arange(len(observed), dtype=np.int32),
        fixed_rows,
        assume_unique=False,
    )
    free_columns = np.setdiff1d(
        np.arange(len(projected_points), dtype=np.int32),
        fixed_columns,
        assume_unique=False,
    )
    assignment_upper_bound = (
        float("inf") if len(fixed_rows) else score_upper_bound
    )
    if len(free_rows):
        free_observed_classes = (
            np.asarray(observed_class_ids, dtype=np.int32).reshape(-1)[free_rows]
            if use_class_aware_assignment and observed_class_ids is not None
            else None
        )
        _assignment_cost, free_row_indices, free_column_indices = observed_keypoint_reprojection_nearest_assignment(
            observed[free_rows],
            projected_points[free_columns],
            image_size,
            observed_class_ids=free_observed_classes,
            projected_class_ids=(
                projected_class_ids[free_columns]
                if use_class_aware_assignment
                else None
            ),
            score_upper_bound=assignment_upper_bound,
        )
        if not len(free_row_indices):
            return (
                float("inf"),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
                projected_points,
                object_points,
                projected_class_ids,
            )
        rows = np.concatenate((fixed_rows, free_rows[free_row_indices]))
        columns = np.concatenate((fixed_columns, free_columns[free_column_indices]))
    else:
        rows, columns = fixed_rows, fixed_columns
    if not len(rows):
        return (
            float("inf"),
            rows,
            columns,
            projected_points,
            object_points,
            projected_class_ids,
        )
    base_score = pose_hypothesis_score(
        observed_points,
        projected_points,
        rows,
        columns,
        observed_class_ids if use_class_aware_assignment else None,
        projected_class_ids if use_class_aware_assignment else None,
    )
    pivot_score = pivot_residual_consistency_score(
        observed_points,
        projected_points,
        object_points,
        rows,
        columns,
        fixed_rows,
        camera_matrix,
        dist_coeffs,
        rvec,
        tvec,
        observed_class_ids=observed_class_ids,
        projected_class_ids=projected_class_ids,
        use_class_aware_assignment=use_class_aware_assignment,
    )
    return (
        base_score + PNP_PIVOT_CONSISTENCY_WEIGHT * pivot_score,
        rows,
        columns,
        projected_points,
        object_points,
        projected_class_ids,
    )


def unique_object_point_indices(
    object_points: np.ndarray,
    image_points: np.ndarray,
    projected_points: np.ndarray,
) -> np.ndarray:
    """Keep one lowest-error observation for each repeated 3-D point."""

    objects = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    image = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    projected = np.asarray(projected_points, dtype=np.float64).reshape(-1, 2)
    if len(objects) != len(image) or len(objects) != len(projected):
        return np.empty((0,), dtype=np.int32)
    distances = np.linalg.norm(image - projected, axis=1)
    best_by_object: dict[tuple[float, float, float], tuple[float, int]] = {}
    for index, (point, distance) in enumerate(zip(objects, distances, strict=True)):
        key = tuple(float(value) for value in point)
        candidate = (float(distance), index)
        if key not in best_by_object or candidate < best_by_object[key]:
            best_by_object[key] = candidate
    return np.asarray(
        sorted(index for _distance, index in best_by_object.values()),
        dtype=np.int32,
    )


def canonical_sticker_object_points(sticker_type: str) -> np.ndarray:
    """Return canonical 3-D corners for one sticker type on the front face."""

    try:
        lattice_coords = STICKER_TYPE_LATTICE_COORDS[sticker_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported sticker type for PnP: {sticker_type}") from exc
    return np.asarray(
        [cube_lattice_point_mm(coord) for coord in lattice_coords],
        dtype=np.float32,
    )


def default_camera_matrix(
    image_size: tuple[int, int],
    focal_scale: float = 1.2,
) -> np.ndarray:
    """Build the same centered pinhole approximation used by the pose tools."""

    width, height = image_size
    focal = max(width, height) * float(focal_scale)
    return np.asarray(
        [[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def canonical_sticker_is_camera_facing(
    object_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> bool:
    """Reject the planar PnP branch that puts the canonical sticker behind the cube."""

    rotation, _ = cv2.Rodrigues(rvec)
    camera_points = (
        rotation @ np.asarray(object_points, dtype=np.float64).T
        + np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    ).T
    if not np.isfinite(camera_points).all() or np.any(camera_points[:, 2] <= 0.0):
        return False

    # The canonical sticker is on z=-half, so its outward normal is -z. The
    # normal must point toward the camera, which is at the camera origin.
    face_center = camera_points.mean(axis=0)
    outward_normal = rotation @ np.asarray((0.0, 0.0, -1.0), dtype=np.float64)
    toward_camera = -face_center
    return float(np.dot(outward_normal, toward_camera)) > 0.0


def sticker_contour_to_frame(
    local_mask: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
) -> np.ndarray | None:
    """Return the largest local sticker contour in full-frame coordinates."""

    x1, y1, x2, y2 = roi_xyxy
    mask_height, mask_width = y2 - y1, x2 - x1
    mask = cv2.resize(
        local_mask.astype(np.uint8),
        (mask_width, mask_height),
        interpolation=cv2.INTER_NEAREST,
    )
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    contour *= np.asarray((1.0, 1.0), dtype=np.float32)
    contour += np.asarray((x1, y1), dtype=np.float32)
    return contour


def sticker_mask_center_observations_to_frame(
    local_masks: list[np.ndarray] | tuple[np.ndarray, ...] | None,
    roi_xyxy: tuple[int, int, int, int] | None,
    sticker_mask_erosion: int = 0,
) -> tuple[tuple[np.ndarray, ...], tuple[float, ...]]:
    """Return mask centers and conservative per-center uncertainty estimates.

    The masks are currently binary, so uncertainty has to be inferred from
    their geometry.  A centroid that is far from the mask's distance-transform
    peak is likely to be shifted by an asymmetric or merged mask.  We also
    penalize low-solidity/low-extent shapes and very small visible regions,
    which are common when a sticker is barely visible.  The result is used as
    a pixel scale for center residuals, keeping clean full stickers influential
    without allowing partial masks to dominate the pose ranking.
    """

    if not local_masks or roi_xyxy is None:
        return (), ()
    x1, y1, x2, y2 = (int(value) for value in roi_xyxy)
    roi_width = x2 - x1
    roi_height = y2 - y1
    if roi_width <= 0 or roi_height <= 0:
        return (), ()
    centers: list[np.ndarray] = []
    uncertainties: list[float] = []
    for local_mask in local_masks:
        resized = cv2.resize(
            np.asarray(local_mask, dtype=np.uint8),
            (roi_width, roi_height),
            interpolation=cv2.INTER_NEAREST,
        )
        mask = erode_binary_mask(resized > 0, sticker_mask_erosion)
        ys, xs = np.nonzero(mask)
        if not len(xs):
            continue
        local_center = np.asarray(
            (float(np.mean(xs)), float(np.mean(ys))),
            dtype=np.float64,
        )
        center = local_center + np.asarray((x1, y1), dtype=np.float64)

        # Distance-transform peaks are stable for compact stickers and move
        # toward the dominant lobe for merged/asymmetric masks.  The offset
        # is therefore a useful proxy for centroid bias, not just pixel noise.
        distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        peak_y, peak_x = np.unravel_index(int(np.argmax(distance)), distance.shape)
        centroid_peak_offset = float(
            np.linalg.norm(
                local_center - np.asarray((peak_x, peak_y), dtype=np.float64)
            )
        )
        pixel_spread = float(
            np.sqrt(
                np.var(xs.astype(np.float64))
                + np.var(ys.astype(np.float64))
            )
        )
        base_uncertainty = max(
            1.0,
            0.5 * centroid_peak_offset + 0.08 * pixel_spread,
        )
        mask_points = np.column_stack((xs, ys)).astype(np.float32)
        hull_area = float(cv2.contourArea(cv2.convexHull(mask_points)))
        visible_area = float(len(xs))
        solidity = visible_area / max(hull_area, 1.0)
        min_rect = cv2.minAreaRect(mask_points)
        rect_width, rect_height = min_rect[1]
        rectangle_area = float(rect_width * rect_height)
        extent = visible_area / max(rectangle_area, 1.0)
        aspect_ratio = min(rect_width, rect_height) / max(
            max(rect_width, rect_height), 1.0
        )
        compactness = np.clip(
            min(solidity, extent, aspect_ratio),
            0.05,
            1.0,
        )
        shape_penalty = 1.0 + PNP_MASK_CENTER_SHAPE_PENALTY * (
            1.0 - float(compactness)
        )
        size_penalty = max(
            0.0,
            np.sqrt(PNP_MASK_CENTER_MIN_VISIBLE_AREA / max(visible_area, 1.0)) - 1.0,
        )
        uncertainty = max(
            1.0,
            (base_uncertainty + size_penalty) * shape_penalty,
        )
        centers.append(center)
        uncertainties.append(uncertainty)
    return tuple(centers), tuple(uncertainties)


def sticker_mask_centers_to_frame(
    local_masks: list[np.ndarray] | tuple[np.ndarray, ...] | None,
    roi_xyxy: tuple[int, int, int, int] | None,
    sticker_mask_erosion: int = 0,
) -> tuple[np.ndarray, ...]:
    """Return one pixel-center estimate for each usable sticker mask."""

    centers, _uncertainties = sticker_mask_center_observations_to_frame(
        local_masks,
        roi_xyxy,
        sticker_mask_erosion,
    )
    return centers


def sticker_mask_iou(
    projected_points: np.ndarray,
    sticker_contour: np.ndarray,
    image_size: tuple[int, int],
) -> float:
    """Compare a projected sticker quadrilateral with its segmentation mask."""

    width, height = image_size
    projected_mask = np.zeros((height, width), dtype=np.uint8)
    segmentation_mask = np.zeros((height, width), dtype=np.uint8)
    polygon = np.round(projected_points).astype(np.int32)
    if len(polygon) >= 3:
        cv2.fillConvexPoly(projected_mask, polygon, 1)
    contour = np.round(sticker_contour).astype(np.int32).reshape(-1, 1, 2)
    if len(contour) >= 3:
        cv2.fillPoly(segmentation_mask, [contour], 1)
    intersection = np.logical_and(projected_mask, segmentation_mask).sum()
    union = np.logical_or(projected_mask, segmentation_mask).sum()
    return float(intersection / union) if union else 0.0


def resample_closed_boundary(
    contour: np.ndarray,
    max_points: int = PNP_MASK_BOUNDARY_MAX_CONTOUR_POINTS,
) -> np.ndarray:
    """Sample the actual segmentation boundary by arc length.

    Sampling bounds repeated scoring cost while preserving the non-convex
    shape and local boundary deviations of the segmentation result.
    """

    points = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    if len(points) < 2 or not np.isfinite(points).all():
        return np.empty((0, 2), dtype=np.float64)
    segment_vectors = np.roll(points, -1, axis=0) - points
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    perimeter = float(np.sum(segment_lengths))
    if not np.isfinite(perimeter) or perimeter <= 1e-6:
        return np.empty((0, 2), dtype=np.float64)
    sample_count = max(8, min(int(max_points), len(points) * 2))
    distances = np.arange(sample_count, dtype=np.float64) * perimeter / sample_count
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    segment_indices = np.searchsorted(cumulative, distances, side="right") - 1
    segment_indices = np.clip(segment_indices, 0, len(points) - 1)
    local_distance = distances - cumulative[segment_indices]
    fractions = np.divide(
        local_distance,
        segment_lengths[segment_indices],
        out=np.zeros_like(local_distance),
        where=segment_lengths[segment_indices] > 1e-9,
    )
    return points[segment_indices] + (
        segment_vectors[segment_indices] * fractions[:, None]
    )


def projected_wireframe_segments(
    candidate: StickerPnPCandidate,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return the visible projected sticker-edge network for a pose."""

    try:
        projected_faces = projected_sticker_faces(candidate, rvec, tvec)
    except (cv2.error, ValueError, FloatingPointError):
        return tuple()
    return wireframe_segments_from_projected_faces(projected_faces)


def wireframe_segments_from_projected_faces(
    projected_faces: list[tuple[StickerIdentity, float, np.ndarray]],
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Build a deduplicated wireframe from already projected sticker faces."""

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[tuple[float, ...]] = set()
    for _identity, _depth, polygon in projected_faces:
        polygon = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
        if len(polygon) < 3 or not np.isfinite(polygon).all():
            continue
        for index in range(len(polygon)):
            start = polygon[index]
            end = polygon[(index + 1) % len(polygon)]
            if float(np.linalg.norm(end - start)) <= 1e-6:
                continue
            ordered = sorted((start, end), key=lambda point: tuple(point.tolist()))
            key = tuple(np.round(np.concatenate(ordered), 3).tolist())
            if key in seen:
                continue
            seen.add(key)
            segments.append((start, end))
    return tuple(segments)


def sample_wireframe_segments(
    segments: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> np.ndarray:
    """Sample projected wireframe segments with a bounded point count."""

    samples: list[np.ndarray] = []
    for start, end in segments:
        length = float(np.linalg.norm(end - start))
        count = max(
            3,
            min(
                PNP_MASK_BOUNDARY_MAX_EDGE_SAMPLES,
                int(np.ceil(length / 4.0)) + 1,
            ),
        )
        samples.append(np.linspace(start, end, count, dtype=np.float64))
    if not samples:
        return np.empty((0, 2), dtype=np.float64)
    return np.concatenate(samples, axis=0)


def nearest_segment_distances(
    source: np.ndarray,
    segments: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> np.ndarray:
    """Return each source point's distance to the closest line segment."""

    source = np.asarray(source, dtype=np.float64).reshape(-1, 2)
    if len(source) == 0 or not segments:
        return np.full(len(source), float("inf"), dtype=np.float64)
    starts = np.asarray([start for start, _end in segments], dtype=np.float64)
    ends = np.asarray([end for _start, end in segments], dtype=np.float64)
    vectors = ends - starts
    lengths_squared = np.sum(vectors * vectors, axis=1)
    offsets = source[:, None, :] - starts[None, :, :]
    parameters = np.divide(
        np.sum(offsets * vectors[None, :, :], axis=2),
        lengths_squared[None, :],
        out=np.zeros((len(source), len(segments)), dtype=np.float64),
        where=lengths_squared[None, :] > 1e-12,
    )
    parameters = np.clip(parameters, 0.0, 1.0)
    closest = starts[None, :, :] + parameters[..., None] * vectors[None, :, :]
    distances = source[:, None, :] - closest
    return np.sqrt(np.min(np.sum(distances * distances, axis=2), axis=1))


def proper_boundary_crossing_count(
    wireframe_segments: tuple[tuple[np.ndarray, np.ndarray], ...],
    boundaries: tuple[np.ndarray, ...],
) -> float:
    """Count soft evidence that a projected edge crosses a mask boundary.

    Collinear contact is deliberately excluded: a correctly aligned cube edge
    lies on the segmentation boundary and should not be penalized.  A
    transverse crossing receives a fractional penalty based on its angle, so
    a bad pose is discouraged without turning segmentation into a hard gate.
    """

    eps = 1e-5

    wire_starts = np.asarray([start for start, _end in wireframe_segments])
    wire_ends = np.asarray([end for _start, end in wireframe_segments])
    boundary_starts = []
    boundary_ends = []
    for boundary in boundaries:
        if len(boundary) < 2:
            continue
        boundary_starts.append(boundary)
        boundary_ends.append(np.roll(boundary, -1, axis=0))
    if not boundary_starts:
        return 0.0
    boundary_starts = np.concatenate(boundary_starts, axis=0)
    boundary_ends = np.concatenate(boundary_ends, axis=0)
    wire_vectors = wire_ends - wire_starts
    boundary_vectors = boundary_ends - boundary_starts

    def cross(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]

    wire_lengths = np.linalg.norm(wire_vectors, axis=1)
    boundary_lengths = np.linalg.norm(boundary_vectors, axis=1)
    valid = (wire_lengths[:, None] > eps) & (boundary_lengths[None, :] > eps)
    sine = np.divide(
        np.abs(cross(wire_vectors[:, None, :], boundary_vectors[None, :, :])),
        wire_lengths[:, None] * boundary_lengths[None, :],
        out=np.zeros((len(wire_vectors), len(boundary_vectors))),
        where=valid,
    )
    first_side = cross(
        wire_vectors[:, None, :],
        boundary_starts[None, :, :] - wire_starts[:, None, :],
    )
    second_side = cross(
        wire_vectors[:, None, :],
        boundary_ends[None, :, :] - wire_starts[:, None, :],
    )
    third_side = cross(
        boundary_vectors[None, :, :],
        wire_starts[:, None, :] - boundary_starts[None, :, :],
    )
    fourth_side = cross(
        boundary_vectors[None, :, :],
        wire_ends[:, None, :] - boundary_starts[None, :, :],
    )
    return float(
        np.sum(
            sine
            * valid
            * (sine >= 0.25)
            * (first_side * second_side < -eps)
            * (third_side * fourth_side < -eps)
        )
    )


def sticker_mask_wireframe_score(
    candidate: StickerPnPCandidate,
    rvec: np.ndarray | None = None,
    tvec: np.ndarray | None = None,
) -> tuple[float, int]:
    """Score a pose against all available sticker-mask boundaries.

    The score is a soft, symmetric Chamfer-like distance in pixels.  The
    observed-boundary term ensures detected mask edges are explained by the
    projected wireframe.  The reverse term is capped, so an internal edge
    absent from a merged mask is tolerated.  Transverse wireframe/boundary
    crossings add a finite penalty but never make a pose invalid.
    """

    contours = tuple(
        contour
        for contour in (
            candidate.sticker_contours
            if candidate.sticker_contours
            else ((candidate.sticker_contour,) if candidate.sticker_contour is not None else ())
        )
        if contour is not None
    )
    if not contours:
        return float("inf"), 0
    pose_rvec = candidate.rvec if rvec is None else rvec
    pose_tvec = candidate.tvec if tvec is None else tvec
    wireframe_segments = projected_wireframe_segments(candidate, pose_rvec, pose_tvec)
    if not wireframe_segments:
        return float("inf"), 0
    boundary_samples = list(candidate.sticker_boundary_samples)
    if len(boundary_samples) != len(contours):
        boundary_samples = [
            resample_closed_boundary(contour)
            for contour in contours
        ]
        boundary_samples = [sample for sample in boundary_samples if len(sample)]
    if not boundary_samples:
        return float("inf"), 0
    observed_boundary = np.concatenate(boundary_samples, axis=0)
    wireframe = sample_wireframe_segments(wireframe_segments)
    if len(observed_boundary) == 0 or len(wireframe) == 0:
        return float("inf"), 0

    diameter = projected_cube_diameter(candidate, pose_rvec, pose_tvec)
    if not np.isfinite(diameter):
        return float("inf"), 0
    delta = max(1.5, PNP_MASK_BOUNDARY_DISTANCE_DELTA_FRACTION * diameter)
    cap = PNP_MASK_MISSING_BOUNDARY_CAP * delta
    boundary_segments = tuple(
        (boundary[index], boundary[(index + 1) % len(boundary)])
        for boundary in boundary_samples
        for index in range(len(boundary))
    )

    def robust_distance_loss(distances: np.ndarray) -> float:
        distances = np.minimum(np.asarray(distances, dtype=np.float64), cap)
        huber = np.where(
            distances <= delta,
            0.5 * distances * distances / delta,
            distances - 0.5 * delta,
        )
        return float(np.mean(huber))

    wire_to_boundary = robust_distance_loss(
        nearest_segment_distances(wireframe, boundary_segments)
    )
    boundary_to_wire = robust_distance_loss(
        nearest_segment_distances(observed_boundary, wireframe_segments)
    )
    crossing_penalty = (
        PNP_MASK_CROSSING_WEIGHT
        * delta
        * proper_boundary_crossing_count(
            wireframe_segments,
            tuple(boundary_samples),
        )
        / max(1, len(wireframe_segments))
    )
    score = (
        PNP_MASK_WIRE_TO_BOUNDARY_WEIGHT * wire_to_boundary
        + PNP_MASK_BOUNDARY_TO_WIRE_WEIGHT * boundary_to_wire
        + crossing_penalty
    )
    return float(score), int(len(observed_boundary) + len(wireframe))


def sticker_mask_layer_wireframe_score(
    candidate: StickerPnPCandidate,
    axis: int,
    side: int,
    rvec: np.ndarray | None = None,
    tvec: np.ndarray | None = None,
) -> tuple[float, int]:
    """Score only the visible sticker geometry belonging to one cube layer.

    Contours are assigned to the nearest projected sticker-face centroid for
    this diagnostic. This keeps unrelated stationary stickers from dominating
    the layer pre-selection score while still allowing a merged contour to
    support whichever layer it is spatially closest to.
    """

    contours = tuple(
        contour
        for contour in (
            candidate.sticker_contours
            if candidate.sticker_contours
            else (
                (candidate.sticker_contour,)
                if candidate.sticker_contour is not None
                else ()
            )
        )
        if contour is not None
    )
    if not contours:
        return 0.0, 0
    pose_rvec = candidate.rvec if rvec is None else rvec
    pose_tvec = candidate.tvec if tvec is None else tvec
    try:
        projected_faces = projected_sticker_faces(candidate, pose_rvec, pose_tvec)
    except (cv2.error, ValueError, FloatingPointError):
        return 0.0, 0
    target_index = 2 if int(side) > 0 else 0
    layer_faces = [
        face
        for face in projected_faces
        if int(face[0].cubie_index[int(axis)]) == target_index
    ]
    if not layer_faces:
        return 0.0, 0
    all_face_centers = np.asarray(
        [np.mean(np.asarray(face[2], dtype=np.float64), axis=0) for face in projected_faces],
        dtype=np.float64,
    )
    layer_face_indices = {
        index
        for index, face in enumerate(projected_faces)
        if int(face[0].cubie_index[int(axis)]) == target_index
    }
    boundary_samples = list(candidate.sticker_boundary_samples)
    if len(boundary_samples) != len(contours):
        boundary_samples = [resample_closed_boundary(contour) for contour in contours]
    selected_samples = []
    for sample in boundary_samples:
        sample = np.asarray(sample, dtype=np.float64).reshape(-1, 2)
        if len(sample) == 0 or not np.isfinite(sample).all():
            continue
        center = np.mean(sample, axis=0)
        nearest_face = int(
            np.argmin(np.linalg.norm(all_face_centers - center[None, :], axis=1))
        )
        if nearest_face in layer_face_indices:
            selected_samples.append(sample)
    if not selected_samples:
        return 0.0, 0
    wireframe_segments = wireframe_segments_from_projected_faces(layer_faces)
    if not wireframe_segments:
        return 0.0, 0
    observed_boundary = np.concatenate(selected_samples, axis=0)
    wireframe = sample_wireframe_segments(wireframe_segments)
    if len(observed_boundary) == 0 or len(wireframe) == 0:
        return 0.0, 0
    diameter = projected_cube_diameter(candidate, pose_rvec, pose_tvec)
    if not np.isfinite(diameter):
        return 0.0, 0
    delta = max(1.5, PNP_MASK_BOUNDARY_DISTANCE_DELTA_FRACTION * diameter)
    cap = PNP_MASK_MISSING_BOUNDARY_CAP * delta
    boundary_segments = tuple(
        (boundary[index], boundary[(index + 1) % len(boundary)])
        for boundary in selected_samples
        for index in range(len(boundary))
    )

    def robust_distance_loss(distances: np.ndarray) -> float:
        distances = np.minimum(np.asarray(distances, dtype=np.float64), cap)
        huber = np.where(
            distances <= delta,
            0.5 * distances * distances / delta,
            distances - 0.5 * delta,
        )
        return float(np.mean(huber))

    wire_to_boundary = robust_distance_loss(
        nearest_segment_distances(wireframe, boundary_segments)
    )
    boundary_to_wire = robust_distance_loss(
        nearest_segment_distances(observed_boundary, wireframe_segments)
    )
    crossing_penalty = (
        PNP_MASK_CROSSING_WEIGHT
        * delta
        * proper_boundary_crossing_count(wireframe_segments, tuple(selected_samples))
        / max(1, len(wireframe_segments))
    )
    score = (
        PNP_MASK_WIRE_TO_BOUNDARY_WEIGHT * wire_to_boundary
        + PNP_MASK_BOUNDARY_TO_WIRE_WEIGHT * boundary_to_wire
        + crossing_penalty
    )
    return float(score), int(len(observed_boundary) + len(wireframe))


def sticker_correspondence_object_indices(
    sticker_type: str,
    observed_class_ids: list[int],
    observed_points: np.ndarray | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Return canonical-corner index sequences matching an ordered observation.

    We preserve the observed direction and try cyclic starts because the
    image-space ordering has no privileged first corner. For a three-point
    observation, class IDs can leave two or more possible omitted corners.
    When the image points are available, use the largest cyclic image gap to
    locate the missing corner and retain only the matching object sequence.
    """

    canonical_classes = STICKER_TYPE_CLASS_IDS.get(sticker_type)
    if canonical_classes is None or len(observed_class_ids) not in (3, 4):
        return tuple()
    observed = tuple(int(class_id) for class_id in observed_class_ids)
    candidates: list[tuple[int, ...]] = []
    for start in range(4):
        rotated = tuple((start + offset) % 4 for offset in range(4))
        if len(observed) == 4:
            sequences = (rotated,)
        else:
            sequences = tuple(
                rotated[:missing] + rotated[missing + 1 :]
                for missing in range(4)
            )
        for sequence in sequences:
            if tuple(canonical_classes[index] for index in sequence) != observed:
                continue
            if sequence not in candidates:
                candidates.append(sequence)

    if len(observed) == 3 and observed_points is not None and candidates:
        points = np.asarray(observed_points, dtype=np.float64).reshape(-1, 2)
        if len(points) == 3 and np.isfinite(points).all():
            # The missing sticker corner is inside the largest gap between
            # consecutive observed corners.  A canonical sequence represents
            # that same gap with an index jump of two modulo four.
            gap_lengths = np.linalg.norm(
                np.roll(points, -1, axis=0) - points,
                axis=1,
            )
            missing_gap = int(np.argmax(gap_lengths))
            gap_matched = tuple(
                sequence
                for sequence in candidates
                if (
                    sequence[(missing_gap + 1) % 3] - sequence[missing_gap]
                ) % 4
                == 2
            )
            if gap_matched:
                candidates = list(gap_matched)
    return tuple(candidates)


def solve_sticker_pnp(
    frame_points: np.ndarray,
    point_class_ids: np.ndarray,
    match: StickerKeypointMatch,
    image_size: tuple[int, int],
    focal_scale: float,
    sticker_contour: np.ndarray | None = None,
    scoring_points: np.ndarray | None = None,
    sticker_contours: tuple[np.ndarray, ...] | None = None,
    sticker_boundary_samples: tuple[np.ndarray, ...] | None = None,
    sticker_mask_centers: tuple[np.ndarray, ...] | None = None,
    sticker_mask_center_uncertainties: tuple[float, ...] | None = None,
) -> StickerPnPCandidate | None:
    """Solve a cube pose from the selected sticker's ordered keypoints."""

    observed_indices = list(match.keypoint_indices)
    if match.sticker_type == "unknown" or len(observed_indices) not in (3, 4):
        return None
    observed_points = np.asarray(frame_points[observed_indices], dtype=np.float32)
    observed_class_ids = [int(point_class_ids[index]) for index in observed_indices]
    index_sequences = sticker_correspondence_object_indices(
        match.sticker_type,
        observed_class_ids,
        observed_points,
    )
    if not index_sequences:
        return None

    camera_matrix = default_camera_matrix(image_size, focal_scale)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    canonical_object_points = canonical_sticker_object_points(match.sticker_type)
    candidates: list[StickerPnPCandidate] = []
    all_observed_points = (
        np.asarray(scoring_points, dtype=np.float32).reshape(-1, 2)
        if scoring_points is not None
        else observed_points
    )
    all_observed_class_ids = (
        np.asarray(point_class_ids, dtype=np.int32).reshape(-1)
        if len(point_class_ids) == len(all_observed_points)
        else None
    )
    all_sticker_contours = tuple(
        np.asarray(contour, dtype=np.float32).copy()
        for contour in (
            sticker_contours
            if sticker_contours is not None
            else ((sticker_contour,) if sticker_contour is not None else ())
        )
        if contour is not None and len(np.asarray(contour).reshape(-1, 2)) >= 3
    )
    if sticker_boundary_samples is None:
        sticker_boundary_samples = tuple(
            sample
            for contour in all_sticker_contours
            if (sample := resample_closed_boundary(contour)).size
        )
    else:
        sticker_boundary_samples = tuple(
            np.asarray(sample, dtype=np.float64).reshape(-1, 2)
            for sample in sticker_boundary_samples
            if len(np.asarray(sample).reshape(-1, 2))
        )
    all_sticker_centers = tuple(
        np.asarray(center, dtype=np.float64).reshape(2)
        for center in (sticker_mask_centers or ())
        if np.isfinite(np.asarray(center, dtype=np.float64)).all()
    )
    all_sticker_center_uncertainties = tuple(
        max(1.0, float(uncertainty))
        for uncertainty in (sticker_mask_center_uncertainties or ())
        if np.isfinite(float(uncertainty))
    )
    anchor_observed_indices = (
        np.arange(len(observed_points), dtype=np.int32)
        if scoring_points is None
        else np.asarray(observed_indices, dtype=np.int32)
    )
    for index_sequence in index_sequences:
        object_points = canonical_object_points[list(index_sequence)]
        try:
            if len(object_points) >= 4:
                # Four sticker points are planar.  IPPE returns both planar
                # pose branches so the full-cube score can choose between
                # them instead of accepting solvePnP's single branch.
                ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                    object_points,
                    observed_points,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE,
                )
            else:
                # Three-point keypoint hypotheses can have multiple valid
                # SQPNP branches. Keep all of them; contour data is reserved
                # for the later turn-fitting pass and does not choose an
                # initial pose branch.
                ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                    object_points,
                    observed_points,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_SQPNP,
                )
        except cv2.error:
            continue
        if not ok or not rvecs or not tvecs:
            continue
        for rvec, tvec in zip(rvecs, tvecs, strict=True):
            rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
            tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
            if not np.isfinite(tvec).all() or float(tvec[2, 0]) <= 0.0:
                continue
            if not canonical_sticker_is_camera_facing(object_points, rvec, tvec):
                continue
            (
                overall_error,
                assignment_rows,
                assignment_columns,
                projected_lattice_points,
                lattice_object_points,
                _projected_class_ids,
            ) = evaluate_pose_hypothesis(
                rvec,
                tvec,
                all_observed_points,
                all_observed_class_ids,
                image_size,
                camera_matrix,
                dist_coeffs,
                float("inf"),
                anchor_observed_indices,
                object_points,
            )
            if (
                not np.isfinite(overall_error)
                or not canonical_sticker_is_camera_facing(object_points, rvec, tvec)
            ):
                continue
            try:
                projected_points, _ = cv2.projectPoints(
                    object_points,
                    rvec,
                    tvec,
                    camera_matrix,
                    dist_coeffs,
                )
            except cv2.error:
                continue
            projected_points = projected_points.reshape(-1, 2)
            reprojection_error = float(
                np.mean(np.linalg.norm(projected_points - observed_points, axis=1))
            )
            candidates.append(
                StickerPnPCandidate(
                    sticker_type=match.sticker_type,
                    sticker_average_confidence=match.average_confidence,
                    image_points=observed_points,
                    object_points=object_points,
                    projected_sticker_points=projected_points,
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                    rvec=rvec,
                    tvec=tvec,
                    reprojection_error=reprojection_error,
                    overall_reprojection_error=overall_error,
                    assignment_image_points=all_observed_points[assignment_rows],
                    assignment_projected_points=projected_lattice_points[
                        assignment_columns
                    ],
                    assignment_object_points=lattice_object_points[
                        assignment_columns
                    ],
                    assignment_observed_class_ids=(
                        all_observed_class_ids[assignment_rows]
                        if all_observed_class_ids is not None
                        else None
                    ),
                    assignment_projected_class_ids=_projected_class_ids[
                        assignment_columns
                    ],
                    assignment_fixed_mask=np.isin(
                        assignment_rows,
                        anchor_observed_indices,
                    ),
                    sticker_contour=(
                        None
                        if sticker_contour is None
                        else np.asarray(sticker_contour, dtype=np.float32).copy()
                    ),
                    sticker_contours=all_sticker_contours,
                    sticker_boundary_samples=sticker_boundary_samples,
                    sticker_mask_centers=all_sticker_centers,
                    sticker_mask_center_uncertainties=all_sticker_center_uncertainties,
                    image_size=image_size,
                )
            )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            item.overall_reprojection_error,
            item.reprojection_error,
        ),
    )


def assignment_geometric_quality(
    image_points: np.ndarray | None,
    projected_points: np.ndarray | None,
) -> tuple[float, float]:
    """Return mean and tail pixel error for an existing assignment."""

    if image_points is None or projected_points is None:
        return float("inf"), float("inf")
    observed = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    projected = np.asarray(projected_points, dtype=np.float64).reshape(-1, 2)
    if (
        len(observed) == 0
        or len(observed) != len(projected)
        or not np.isfinite(observed).all()
        or not np.isfinite(projected).all()
    ):
        return float("inf"), float("inf")
    distances = np.linalg.norm(observed - projected, axis=1)
    return float(np.mean(distances)), float(np.percentile(distances, 90))


def pose_keypoint_error(
    candidate: StickerPnPCandidate,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> float:
    """Return the robust keypoint reprojection term used by mask refinement."""

    object_points = candidate.assignment_object_points
    image_points = candidate.assignment_image_points
    if object_points is None or image_points is None:
        object_points = candidate.object_points
        image_points = candidate.image_points
    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    if (
        len(object_points) == 0
        or len(object_points) != len(image_points)
        or not np.isfinite(object_points).all()
        or not np.isfinite(image_points).all()
    ):
        return float("inf")
    try:
        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            candidate.camera_matrix,
            candidate.dist_coeffs,
        )
    except cv2.error:
        return float("inf")
    projected = projected.reshape(-1, 2)
    distances = np.linalg.norm(projected - image_points, axis=1)
    finite = np.isfinite(distances)
    if not np.any(finite):
        return float("inf")
    distances = distances[finite]
    diameter = projected_cube_diameter(candidate, rvec, tvec)
    if not np.isfinite(diameter):
        return float("inf")
    delta = max(1.0, 0.01 * diameter)
    huber = np.where(
        distances <= delta,
        0.5 * distances * distances / delta,
        distances - 0.5 * delta,
    )
    return float(np.mean(huber))


def object_points_are_camera_facing(
    object_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> bool:
    """Check facing using the actual (possibly turned) sticker geometry."""

    points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    if len(points) < 3 or not np.isfinite(points).all():
        return False
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    camera_points = (
        rotation @ points.T + np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    ).T
    if not np.isfinite(camera_points).all() or np.any(camera_points[:, 2] <= 0.0):
        return False
    normal = np.cross(camera_points[1] - camera_points[0], camera_points[2] - camera_points[0])
    return float(np.dot(normal, -camera_points.mean(axis=0))) > 0.0


def refresh_candidate_pose_metrics(
    candidate: StickerPnPCandidate,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> StickerPnPCandidate | None:
    """Project stored geometry and recompute the candidate's pose metrics."""

    try:
        projected_sticker_points, _ = cv2.projectPoints(
            candidate.object_points,
            rvec,
            tvec,
            candidate.camera_matrix,
            candidate.dist_coeffs,
        )
        projected_sticker_points = projected_sticker_points.reshape(-1, 2)
        projected_assignment_points = None
        if candidate.assignment_object_points is not None:
            projected_assignment_points, _ = cv2.projectPoints(
                candidate.assignment_object_points,
                rvec,
                tvec,
                candidate.camera_matrix,
                candidate.dist_coeffs,
            )
            projected_assignment_points = projected_assignment_points.reshape(-1, 2)
    except cv2.error:
        return None
    if not np.isfinite(projected_sticker_points).all():
        return None
    sticker_error = float(
        np.mean(
            np.linalg.norm(
                projected_sticker_points
                - np.asarray(candidate.image_points, dtype=np.float64),
                axis=1,
            )
        )
    )
    overall_error = candidate.overall_reprojection_error
    if (
        projected_assignment_points is not None
        and candidate.assignment_image_points is not None
    ):
        assignment_distances = np.linalg.norm(
            projected_assignment_points
            - np.asarray(candidate.assignment_image_points, dtype=np.float64),
            axis=1,
        )
        if np.isfinite(assignment_distances).all():
            overall_error = float(np.mean(huber_loss(assignment_distances)))
    return replace(
        candidate,
        rvec=np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        tvec=np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        projected_sticker_points=projected_sticker_points,
        assignment_projected_points=projected_assignment_points,
        reprojection_error=sticker_error,
        overall_reprojection_error=overall_error,
        mask_fit_error=float("inf"),
        mask_boundary_sample_count=0,
    )


def refine_pose_with_sticker_boundaries(
    candidate: StickerPnPCandidate,
) -> StickerPnPCandidate:
    """Locally refine an existing keypoint pose using soft mask boundaries.

    This is intentionally a small coordinate search around the keypoint pose,
    rather than a new pose hypothesis generator.  The keypoint reprojection
    term remains in the objective, so segmentation can improve precision but
    cannot pull the pose arbitrarily away from the detected geometry.
    """

    if not candidate.sticker_contours:
        return candidate
    initial_mask_error, _ = sticker_mask_wireframe_score(candidate)
    if not np.isfinite(initial_mask_error):
        return candidate
    initial_keypoint_error = pose_keypoint_error(
        candidate,
        candidate.rvec,
        candidate.tvec,
    )
    if not np.isfinite(initial_keypoint_error):
        return candidate
    diameter = projected_cube_diameter(candidate)
    if not np.isfinite(diameter):
        return candidate

    def objective(rvec: np.ndarray, tvec: np.ndarray) -> float:
        keypoint_error = pose_keypoint_error(candidate, rvec, tvec)
        mask_error, _ = sticker_mask_wireframe_score(candidate, rvec, tvec)
        if not np.isfinite(keypoint_error) or not np.isfinite(mask_error):
            return float("inf")
        return keypoint_error + PNP_MASK_POSE_WEIGHT * mask_error

    current_rvec = np.asarray(candidate.rvec, dtype=np.float64).reshape(3, 1).copy()
    current_tvec = np.asarray(candidate.tvec, dtype=np.float64).reshape(3, 1).copy()
    current_score = objective(current_rvec, current_tvec)
    rotation_step = np.deg2rad(PNP_MASK_POSE_ROTATION_STEP_DEGREES)
    translation_step = max(
        0.15,
        abs(float(current_tvec[2, 0])) * PNP_MASK_POSE_TRANSLATION_STEP_FRACTION,
    )
    for _pass in range(PNP_MASK_POSE_REFINEMENT_PASSES):
        improved = False
        for axis in range(3):
            for sign in (-1.0, 1.0):
                trial_rvec = current_rvec.copy()
                trial_rvec[axis, 0] += sign * rotation_step
                if not object_points_are_camera_facing(
                    candidate.object_points,
                    trial_rvec,
                    current_tvec,
                ):
                    continue
                trial_score = objective(trial_rvec, current_tvec)
                if trial_score + 1e-6 < current_score:
                    current_rvec = trial_rvec
                    current_score = trial_score
                    improved = True
            for sign in (-1.0, 1.0):
                trial_tvec = current_tvec.copy()
                trial_tvec[axis, 0] += sign * translation_step
                if trial_tvec[2, 0] <= 0.0:
                    continue
                trial_score = objective(current_rvec, trial_tvec)
                if trial_score + 1e-6 < current_score:
                    current_tvec = trial_tvec
                    current_score = trial_score
                    improved = True
        rotation_step *= 0.5
        translation_step *= 0.5
        if not improved:
            break
    refined = refresh_candidate_pose_metrics(candidate, current_rvec, current_tvec)
    return candidate if refined is None else refined


def assignment_residual_direction_statistics(
    image_points: np.ndarray | None,
    projected_points: np.ndarray | None,
) -> tuple[float, np.ndarray]:
    """Measure whether assigned residual vectors share a common direction."""

    if image_points is None or projected_points is None:
        return 0.0, np.zeros(2, dtype=np.float64)
    observed = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    projected = np.asarray(projected_points, dtype=np.float64).reshape(-1, 2)
    if (
        len(observed) == 0
        or len(observed) != len(projected)
        or not np.isfinite(observed).all()
        or not np.isfinite(projected).all()
    ):
        return 0.0, np.zeros(2, dtype=np.float64)
    residuals = observed - projected
    magnitudes = np.linalg.norm(residuals, axis=1)
    valid = magnitudes > 1e-6
    if not np.any(valid):
        return 1.0, np.zeros(2, dtype=np.float64)
    unit_residuals = residuals[valid] / magnitudes[valid, None]
    mean_unit_residual = np.mean(unit_residuals, axis=0)
    return float(np.linalg.norm(mean_unit_residual)), np.mean(residuals, axis=0)


def projected_cube_diameter(
    candidate: StickerPnPCandidate,
    rvec: np.ndarray | None = None,
    tvec: np.ndarray | None = None,
) -> float:
    """Return the diameter of the smallest image circle containing the cube."""

    pose_rvec = candidate.rvec if rvec is None else rvec
    pose_tvec = candidate.tvec if tvec is None else tvec
    try:
        projected_cube, _ = cv2.projectPoints(
            CUBE_CORNER_POINTS_MM,
            pose_rvec,
            pose_tvec,
            candidate.camera_matrix,
            candidate.dist_coeffs,
        )
    except cv2.error:
        return float("inf")
    projected_cube = projected_cube.reshape(-1, 2)
    if len(projected_cube) < 3 or not np.isfinite(projected_cube).all():
        return float("inf")
    try:
        _center, radius = cv2.minEnclosingCircle(
            np.asarray(projected_cube, dtype=np.float32)
        )
    except cv2.error:
        return float("inf")
    diameter = 2.0 * float(radius)
    return diameter if np.isfinite(diameter) and diameter > 0.0 else float("inf")


def pose_reprojection_fit_metric(
    candidate: StickerPnPCandidate,
    rvec: np.ndarray | None = None,
    tvec: np.ndarray | None = None,
) -> float:
    """Return robust reprojection error as a fraction of projected cube size.

    The median keeps a small number of bad observations from dominating the
    pose decision; the p90 term prevents a pose with a large residual tail from
    passing just because its average is small.  Dividing by the projected cube
    diameter makes the value comparable across frames with different cube
    sizes.
    """

    observed = getattr(candidate, "assignment_image_points", None)
    projected = getattr(candidate, "assignment_projected_points", None)
    if observed is None or projected is None:
        observed = getattr(candidate, "image_points", None)
        projected = getattr(candidate, "projected_sticker_points", None)
    if observed is None or projected is None or not hasattr(candidate, "camera_matrix"):
        return float("inf")
    observed = np.asarray(observed, dtype=np.float64).reshape(-1, 2)
    projected = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
    if (
        len(observed) == 0
        or len(observed) != len(projected)
        or not np.isfinite(observed).all()
        or not np.isfinite(projected).all()
    ):
        return float("inf")
    distances = np.linalg.norm(observed - projected, axis=1)
    finite = np.isfinite(distances)
    if not np.any(finite):
        return float("inf")
    scale = projected_cube_diameter(candidate, rvec, tvec)
    if not np.isfinite(scale) or scale <= 0.0:
        return float("inf")
    normalized = distances[finite] / scale
    return float(np.median(normalized) + 0.5 * np.percentile(normalized, 90))


def pose_reprojection_error_is_acceptable(candidate: object) -> bool:
    """Return whether a candidate has enough geometry to pass the fit gate."""

    if not hasattr(candidate, "camera_matrix"):
        # Lightweight candidates used by graph/continuity code may not carry
        # image geometry. They cannot be evaluated here, so preserve the
        # caller's existing ranking behavior for those objects.
        return True
    metric = pose_reprojection_fit_metric(candidate)
    return np.isfinite(metric) and metric <= PNP_POSE_REPROJECTION_ERROR_THRESHOLD


def cube_layer_mask(
    object_points: np.ndarray,
    axis: int,
    side: int,
    layer_count: int = 1,
) -> np.ndarray:
    """Return surface points belonging to one or more outer cubie layers.

    A turned outer layer contains more than the points on its outside face.
    Including the full cubie thickness is important because points on the
    adjacent side faces move with that layer too. ``layer_count=2`` models a
    wide face turn such as Uw.
    """

    points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    layer_count = int(np.clip(layer_count, 1, 3))
    # A surface lattice plane on the interface between two cubie layers is
    # shared geometrically, but it must belong to the stationary side for a
    # one-layer turn.  Therefore a one-layer turn selects only the outermost
    # lattice plane; a two-layer turn selects the outer two planes.  This is
    # one lattice index farther out than the cubie-center boundary.
    if layer_count >= 3:
        return np.ones(len(points), dtype=bool)
    inner_layer_coordinate = (
        CUBE_HALF_SIZE_MM - (layer_count - 1) * CUBE_LAYER_SIZE_MM
    )
    if side > 0:
        return points[:, axis] >= inner_layer_coordinate - 1e-4
    return points[:, axis] <= -inner_layer_coordinate + 1e-4


def apply_cube_layer_rotations(
    object_points: np.ndarray,
    layer_rotations: tuple[CubeLayerRotation, ...] | list[CubeLayerRotation],
) -> np.ndarray:
    """Apply face-layer rotations to a copy of the cube geometry."""

    base_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    transformed = base_points.copy()
    for layer_rotation in layer_rotations:
        axis = int(layer_rotation.axis)
        side = int(layer_rotation.side)
        if axis not in (0, 1, 2) or side not in (-1, 1):
            continue
        mask = cube_layer_mask(
            base_points,
            axis,
            side,
            layer_rotation.layer_count,
        )
        if not np.any(mask):
            continue
        rotation_vector = np.zeros((3, 1), dtype=np.float64)
        rotation_vector[axis, 0] = (
            side * np.deg2rad(float(layer_rotation.angle_degrees))
        )
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        center = np.zeros(3, dtype=np.float64)
        center[axis] = side * CUBE_HALF_SIZE_MM
        transformed[mask] = (
            (rotation_matrix @ (transformed[mask] - center).T).T + center
        )
    return transformed.astype(np.float32)


def corrected_candidate_object_points(
    candidate: StickerPnPCandidate,
    object_points: np.ndarray,
) -> np.ndarray:
    """Apply the candidate's layer corrections to arbitrary cube points."""

    return apply_cube_layer_rotations(object_points, candidate.layer_rotations)


def all_cube_orientation_anchor() -> OrientationAnchor:
    """Return the anchor representing the entire cube."""

    return OrientationAnchor(axis=None, layers=(0, 1, 2))


def orientation_anchor_for_candidate(
    candidate: StickerPnPCandidate,
) -> OrientationAnchor:
    """Choose the layers unaffected by the selected turn hypothesis.

    The turn search currently produces at most one new layer rotation.  The
    complement of that layer is the rigid support for the global pose.  If no
    meaningful turn was selected, all three layers remain the anchor.
    """

    rotations = tuple(
        rotation
        for rotation in candidate.layer_rotations
        if (
            int(rotation.axis) in (0, 1, 2)
            and int(rotation.side) in (-1, 1)
            and abs(float(rotation.angle_degrees)) > 1e-6
        )
    )
    if not rotations:
        return all_cube_orientation_anchor()
    axes = {int(rotation.axis) for rotation in rotations}
    if len(axes) != 1:
        # A multi-axis hypothesis does not have a single slab-shaped support;
        # do not claim a narrower anchor than the geometry can justify.
        return all_cube_orientation_anchor()
    axis = axes.pop()
    moved_layers: set[int] = set()
    for rotation in rotations:
        layer_count = int(np.clip(rotation.layer_count, 1, 3))
        if int(rotation.side) > 0:
            moved_layers.update(range(3 - layer_count, 3))
        else:
            moved_layers.update(range(layer_count))
    stationary_layers = tuple(layer for layer in range(3) if layer not in moved_layers)
    if not stationary_layers:
        return all_cube_orientation_anchor()
    return OrientationAnchor(axis=axis, layers=stationary_layers)


def _cubelet_layer_index(center: np.ndarray, axis: int) -> int:
    coordinate = float(np.asarray(center, dtype=np.float64).reshape(3)[axis])
    return int(np.clip(round(coordinate / CUBE_LAYER_SIZE_MM) + 1, 0, 2))


def orientation_anchor_assignment_mask(
    candidate: StickerPnPCandidate,
    anchor: OrientationAnchor,
) -> np.ndarray:
    """Return assignment rows whose 3-D targets lie in the anchor layers."""

    object_points = candidate.assignment_object_points
    if object_points is None:
        object_points = candidate.object_points
    points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    if anchor.axis is None or len(points) == 0:
        return np.ones(len(points), dtype=bool)

    transformed_lattice = apply_cube_layer_rotations(
        CUBE_LATTICE_OBJECT_POINTS_MM,
        candidate.layer_rotations,
    ).astype(np.float64)
    distances = np.linalg.norm(
        points[:, None, :] - transformed_lattice[None, :, :],
        axis=2,
    )
    nearest = np.argmin(distances, axis=1)
    valid = np.min(distances, axis=1) <= 1e-3
    layer_indices = np.asarray(
        [
            _cubelet_layer_index(CUBE_LATTICE_OBJECT_POINTS_MM[index], anchor.axis)
            for index in nearest
        ],
        dtype=np.int32,
    )
    return valid & np.isin(layer_indices, np.asarray(anchor.layers, dtype=np.int32))


def refit_pose_to_orientation_anchor(
    candidate: StickerPnPCandidate,
    anchor: OrientationAnchor,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Refit the pose from observations belonging to the stationary support."""

    image_points = candidate.assignment_image_points
    object_points = candidate.assignment_object_points
    if image_points is None or object_points is None:
        image_points = candidate.image_points
        object_points = candidate.object_points
    image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    object_points = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
    if len(image_points) != len(object_points) or len(image_points) < 3:
        return None
    mask = orientation_anchor_assignment_mask(candidate, anchor)
    finite = np.isfinite(image_points).all(axis=1) & np.isfinite(object_points).all(axis=1)
    indexes = np.flatnonzero(mask & finite).astype(np.int32)
    if len(indexes) < 3:
        return None
    _unique_objects, unique_indexes = np.unique(
        object_points[indexes],
        axis=0,
        return_index=True,
    )
    indexes = indexes[np.sort(unique_indexes)]
    if len(indexes) < 3:
        return None
    try:
        if len(indexes) >= 4:
            ok, rvec, tvec = cv2.solvePnP(
                object_points[indexes],
                image_points[indexes],
                candidate.camera_matrix,
                candidate.dist_coeffs,
                rvec=np.asarray(candidate.rvec, dtype=np.float64).reshape(3, 1),
                tvec=np.asarray(candidate.tvec, dtype=np.float64).reshape(3, 1),
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        else:
            ok, rvec, tvec = cv2.solvePnP(
                object_points[indexes],
                image_points[indexes],
                candidate.camera_matrix,
                candidate.dist_coeffs,
                flags=cv2.SOLVEPNP_SQPNP,
            )
    except cv2.error:
        return None
    if not ok or not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
        return None
    return (
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec, dtype=np.float64).reshape(3, 1),
    )


def refresh_candidate_pose_projection(
    candidate: StickerPnPCandidate,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> StickerPnPCandidate:
    """Reproject stored display geometry after temporal pose filtering."""

    projected_sticker_points = candidate.projected_sticker_points
    projected_assignment_points = candidate.assignment_projected_points
    try:
        projected_sticker_points, _ = cv2.projectPoints(
            candidate.object_points,
            rvec,
            tvec,
            candidate.camera_matrix,
            candidate.dist_coeffs,
        )
        projected_sticker_points = projected_sticker_points.reshape(-1, 2)
        if candidate.assignment_object_points is not None:
            projected_assignment_points, _ = cv2.projectPoints(
                candidate.assignment_object_points,
                rvec,
                tvec,
                candidate.camera_matrix,
                candidate.dist_coeffs,
            )
            projected_assignment_points = projected_assignment_points.reshape(-1, 2)
    except cv2.error:
        return candidate
    return replace(
        candidate,
        rvec=np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        tvec=np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        projected_sticker_points=projected_sticker_points,
        assignment_projected_points=projected_assignment_points,
    )


def transform_candidate_by_symmetry(
    candidate: StickerPnPCandidate,
    symmetry: np.ndarray,
) -> StickerPnPCandidate:
    """Transform every pose-bearing field into an equivalent cube frame."""

    transformed_rvec = transform_rvec_by_symmetry(candidate.rvec, symmetry)
    transformed_layer_rotations = tuple(
        transform_layer_rotation_by_symmetry(rotation, symmetry)
        for rotation in candidate.layer_rotations
    )
    transformed_debug = tuple(
        replace(
            debug_candidate,
            label=layer_rotation_debug_label(
                tuple(
                    transform_layer_rotation_by_symmetry(rotation, symmetry)
                    for rotation in debug_candidate.layer_rotations
                )
            ),
            layer_rotations=tuple(
                transform_layer_rotation_by_symmetry(rotation, symmetry)
                for rotation in debug_candidate.layer_rotations
            ),
            assignment_object_points=transform_object_points_by_symmetry(
                debug_candidate.assignment_object_points,
                symmetry,
            ),
        )
        for debug_candidate in candidate.turn_candidate_debug
    )
    transformed_scores = tuple(
        (
            (
                transformed_debug[index].label
                if index < len(transformed_debug)
                else label
            ),
            robust_error,
            inlier_count,
            p90_error,
            mean_error,
        )
        for index, (label, robust_error, inlier_count, p90_error, mean_error)
        in enumerate(candidate.turn_candidate_scores)
    )
    prior_object_symmetry = (
        np.eye(3, dtype=np.float64)
        if candidate.object_symmetry is None
        else np.asarray(candidate.object_symmetry, dtype=np.float64).reshape(3, 3)
    )

    def transform_optional_rvec(value: np.ndarray | None) -> np.ndarray | None:
        return None if value is None else transform_rvec_by_symmetry(value, symmetry)

    return replace(
        candidate,
        object_points=transform_object_points_by_symmetry(
            candidate.object_points,
            symmetry,
        ),
        assignment_object_points=(
            None
            if candidate.assignment_object_points is None
            else transform_object_points_by_symmetry(
                candidate.assignment_object_points,
                symmetry,
            )
        ),
        layer_rotations=transformed_layer_rotations,
        turn_candidate_scores=transformed_scores,
        turn_candidate_debug=transformed_debug,
        rvec=transformed_rvec,
        pre_final_rvec=transform_optional_rvec(candidate.pre_final_rvec),
        raw_rvec=transform_optional_rvec(candidate.raw_rvec),
        orientation_anchor=None,
        object_symmetry=(
            np.asarray(symmetry, dtype=np.float64).reshape(3, 3)
            @ prior_object_symmetry
        ),
    )


def canonicalize_pose_symmetry(
    candidate: StickerPnPCandidate,
    symmetry_tracker: CubeSymmetryTracker,
) -> StickerPnPCandidate:
    """Select the symmetry-equivalent pose closest to the prior frame."""

    reference_rotation = symmetry_tracker.rotation
    if reference_rotation is None:
        # There is no observable preference on the first frame.  Identity is
        # deterministic and preserves the existing canonical convention.
        return transform_candidate_by_symmetry(candidate, CUBE_SYMMETRY_MATRICES[0])

    reference_rvec = cv2.Rodrigues(reference_rotation)[0]
    symmetry = min(
        CUBE_SYMMETRY_MATRICES,
        key=lambda matrix: rotation_symmetry_distance(
            transform_rvec_by_symmetry(candidate.rvec, matrix),
            reference_rvec,
        ),
    )
    return transform_candidate_by_symmetry(candidate, symmetry)


def apply_temporal_pose_filter(
    candidate: StickerPnPCandidate,
    motion_filter: PoseMotionFilter | None,
) -> StickerPnPCandidate:
    """Commit a selected current pose without blending it with prediction."""

    anchor = orientation_anchor_for_candidate(candidate)
    if motion_filter is None:
        # Keep the visualization honest even when temporal filtering is
        # disabled for an A/B run: the selected turn still identifies the
        # stationary support that would anchor the pose.
        return replace(candidate, orientation_anchor=anchor)
    measured_rvec = np.asarray(candidate.rvec, dtype=np.float64).copy()
    measured_tvec = np.asarray(candidate.tvec, dtype=np.float64).copy()
    accepted_rvec, accepted_tvec, diagnostics = motion_filter.accept_measurement(
        measured_rvec,
        measured_tvec,
    )
    accepted = refresh_candidate_pose_projection(
        candidate,
        accepted_rvec,
        accepted_tvec,
    )
    return replace(
        accepted,
        raw_rvec=measured_rvec,
        raw_tvec=measured_tvec,
        motion_diagnostics=diagnostics,
        orientation_anchor=anchor,
    )


def rank_misaligned_layer_faces(
    base_object_points: np.ndarray,
    assignment_object_points: np.ndarray,
    assignment_distances: np.ndarray,
    fixed_mask: np.ndarray | None,
    assignment_error_limit: float,
    max_face_candidates: int,
) -> tuple[tuple[int, int, float], ...]:
    """Rank face directions by the number of explained observations.

    A moved layer is least compatible with the initial rigid-cube pose, so it
    should have the fewest low-error assignments. Mapping the current
    assignment back to the canonical lattice is cheap compared with projecting
    and assigning a complete cube for every angle. The returned entries are
    ``(axis, side, explained_count)``; layer widths are intentionally not
    selected here because both width one and width two are inexpensive once
    the face is shortlisted.
    """

    if max_face_candidates <= 0 or not np.isfinite(assignment_error_limit):
        return tuple()
    base = np.asarray(base_object_points, dtype=np.float64).reshape(-1, 3)
    assigned = np.asarray(assignment_object_points, dtype=np.float64).reshape(-1, 3)
    distances = np.asarray(assignment_distances, dtype=np.float64).reshape(-1)
    if (
        len(base) == 0
        or len(assigned) != len(distances)
        or len(assigned) == 0
        or not np.isfinite(base).all()
    ):
        return tuple()

    finite = np.isfinite(assigned).all(axis=1) & np.isfinite(distances)
    if fixed_mask is None:
        free = np.ones(len(assigned), dtype=bool)
    else:
        fixed = np.asarray(fixed_mask, dtype=bool).reshape(-1)
        if len(fixed) != len(assigned):
            free = np.ones(len(assigned), dtype=bool)
        else:
            free = ~fixed
    usable = finite & free
    if not np.any(usable):
        return tuple()

    nearest = np.argmin(
        np.linalg.norm(assigned[:, None, :] - base[None, :, :], axis=2),
        axis=1,
    )
    # Ignore malformed assignments that are not actually lattice points.
    nearest_error = np.linalg.norm(assigned - base[nearest], axis=1)
    usable &= nearest_error <= 1e-3
    if not np.any(usable):
        return tuple()

    scores: list[tuple[int, int, float]] = []
    for axis in range(3):
        for side in (-1, 1):
            layer_mask = cube_layer_mask(base, axis, side, layer_count=1)
            members = usable & layer_mask[nearest]
            explained_count = int(
                np.count_nonzero(
                    members & (distances <= float(assignment_error_limit))
                )
            )
            scores.append((axis, side, float(explained_count)))

    scores.sort(key=lambda item: (item[2], item[0], item[1]))
    return tuple(scores[:max_face_candidates])


def layer_face_debug_label(axis: int, side: int) -> str:
    """Return the compact label used for a candidate face layer."""

    axis_names = "XYZ"
    axis_index = int(axis)
    axis_name = axis_names[axis_index] if axis_index in range(3) else "?"
    return f"{axis_name}{'+' if int(side) > 0 else '-'}"


def initial_sticker_layer(candidate: StickerPnPCandidate) -> tuple[int, int] | None:
    """Return the cube face layer containing the selected sticker anchor."""

    points = np.asarray(candidate.object_points, dtype=np.float64).reshape(-1, 3)
    if len(points) < 3 or not np.isfinite(points).all():
        return None
    for axis in range(3):
        coordinate = float(np.mean(points[:, axis]))
        if float(np.ptp(points[:, axis])) > 1e-3:
            continue
        if abs(abs(coordinate) - CUBE_HALF_SIZE_MM) > 1e-3:
            continue
        return axis, 1 if coordinate > 0.0 else -1
    return None


def layer_keypoint_fit_score(
    base_object_points: np.ndarray,
    assignment_object_points: np.ndarray,
    assignment_distances: np.ndarray,
    fixed_mask: np.ndarray,
    assignment_error_limit: float,
    axis: int,
    side: int,
) -> tuple[float, int, int]:
    """Score only the assigned observations belonging to one face layer."""

    base = np.asarray(base_object_points, dtype=np.float64).reshape(-1, 3)
    assigned = np.asarray(assignment_object_points, dtype=np.float64).reshape(-1, 3)
    distances = np.asarray(assignment_distances, dtype=np.float64).reshape(-1)
    fixed = np.asarray(fixed_mask, dtype=bool).reshape(-1)
    if (
        len(assigned) != len(distances)
        or len(fixed) != len(distances)
        or len(assigned) == 0
        or not np.isfinite(assigned).all()
        or not np.isfinite(distances).all()
    ):
        return 0.0, 0, 0
    nearest = np.argmin(
        np.linalg.norm(assigned[:, None, :] - base[None, :, :], axis=2),
        axis=1,
    )
    lattice_error = np.linalg.norm(assigned - base[nearest], axis=1)
    layer_mask = cube_layer_mask(base, axis, side, layer_count=1)
    usable = (~fixed) & (lattice_error <= 1e-3) & layer_mask[nearest]
    if not np.any(usable):
        return 0.0, 0, 0
    layer_distances = distances[usable]
    delta = max(1.0, 0.01 * float(assignment_error_limit))
    robust_error = float(
        np.mean(huber_loss(layer_distances, delta=delta))
        + PNP_ASSIGNMENT_TAIL_WEIGHT * np.percentile(layer_distances, 90)
        + 0.10 * np.percentile(layer_distances, 95)
    )
    unexplained_count = int(np.count_nonzero(layer_distances > assignment_error_limit))
    # The pre-selector should strongly prefer a layer that explains fewer
    # observations, while retaining the continuous residual as a tie-breaker.
    score = robust_error + unexplained_count * max(1.0, assignment_error_limit)
    return float(score), unexplained_count, int(len(layer_distances))


def score_turn_layer_faces(
    candidate: StickerPnPCandidate,
    base_object_points: np.ndarray,
    assignment_object_points: np.ndarray,
    assignment_distances: np.ndarray,
    fixed_mask: np.ndarray,
    assignment_error_limit: float,
    excluded_layer: tuple[int, int] | None,
) -> tuple[tuple[tuple[str, float, float, float], ...], int]:
    """Score the five non-anchor layers using local keypoint fits only.

    The sticker columns remain in the diagnostic tuple for compatibility with
    existing logs and overlays, but are zero while sticker scoring is
    disabled.  Sticker masks are visualization-only in this pose-only pass.
    """

    layer_scores: list[tuple[str, float, float, float]] = []
    unexplained_count = 0
    for axis in range(3):
        for side in (-1, 1):
            if excluded_layer == (axis, side):
                continue
            keypoint_score, layer_unexplained, _layer_count = layer_keypoint_fit_score(
                base_object_points,
                assignment_object_points,
                assignment_distances,
                fixed_mask,
                assignment_error_limit,
                axis,
                side,
            )
            sticker_score = 0.0
            combined_score = keypoint_score
            layer_scores.append(
                (
                    layer_face_debug_label(axis, side),
                    float(keypoint_score),
                    float(sticker_score),
                    float(combined_score),
                )
            )
            unexplained_count += layer_unexplained
    layer_scores.sort(key=lambda item: (-item[3], item[0]))
    return tuple(layer_scores), unexplained_count


def _legacy_recover_misaligned_cube_layers(
    candidate: StickerPnPCandidate,
    assignment_error_limit: float,
    max_face_candidates: int = PNP_TURN_FACE_CANDIDATES,
    record_all_turn_candidates: bool = False,
) -> StickerPnPCandidate:
    """Search one likely layer turn using local keypoint geometry.

    The selected sticker is kept rigid.  The other five face layers are scored
    once, and only the worst layer receives the full -45..45 degree search.
    Each angle is ranked using only that layer's keypoint terms.  A turn is
    accepted only when it strictly increases the number of explained
    observations over the static pose.
    """

    image_points = candidate.assignment_image_points
    projected_points = candidate.assignment_projected_points
    object_points = candidate.assignment_object_points
    if image_points is None or projected_points is None or object_points is None:
        return candidate
    image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    projected_points = np.asarray(projected_points, dtype=np.float32).reshape(-1, 2)
    object_points = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
    if (
        len(image_points) < 3
        or len(image_points) != len(projected_points)
        or len(image_points) != len(object_points)
        or not np.isfinite(assignment_error_limit)
    ):
        return candidate

    distances = np.linalg.norm(image_points - projected_points, axis=1)
    finite = (
        np.isfinite(image_points).all(axis=1)
        & np.isfinite(projected_points).all(axis=1)
        & np.isfinite(object_points).all(axis=1)
        & np.isfinite(distances)
    )
    accepted = finite & (distances <= assignment_error_limit)

    base_object_points = np.asarray(
        CUBE_LATTICE_OBJECT_POINTS_MM,
        dtype=np.float32,
    )
    if len(base_object_points) == 0:
        return candidate

    # The selected sticker is the pivot for this local search.  Its own
    # residuals are not evidence of a moved layer, so exclude those rows from
    # the trigger and from the layer pre-selector.
    fixed_mask = candidate.assignment_fixed_mask
    if fixed_mask is not None:
        fixed_mask = np.asarray(fixed_mask, dtype=bool).reshape(-1)
    if fixed_mask is None or len(fixed_mask) != len(image_points):
        fixed_mask = np.zeros(len(image_points), dtype=bool)
        fixed_mask[: min(len(candidate.image_points), len(image_points))] = True
    fixed_positions = np.flatnonzero(fixed_mask).astype(np.int32)
    if len(fixed_positions) != len(candidate.object_points):
        # This should only be reachable for a legacy/malformed candidate.  A
        # partial fixed set is safer than silently binding the wrong points.
        fixed_positions = np.empty((0,), dtype=np.int32)

    unexplained = finite & ~accepted & ~fixed_mask
    unexplained_count = int(np.count_nonzero(unexplained))
    # Sticker scoring is intentionally disabled for this pass.  Keep the
    # diagnostic slot at zero so old consumers can still read the tuple while
    # making it explicit that the score is pose-only.
    sticker_fit_score = 0.0
    static_keypoint_score = float(candidate.overall_reprojection_error)
    if not np.isfinite(static_keypoint_score):
        static_keypoint_score = float("inf")
    static_total_score = static_keypoint_score
    diagnostic_candidate = replace(
        candidate,
        turn_layer_scores=(),
        turn_fit_sticker_score=float(sticker_fit_score),
        turn_fit_unexplained_keypoints=unexplained_count,
        turn_fit_evaluated=True,
        turn_fit_static_scores=(
            static_keypoint_score,
            float(sticker_fit_score),
            float(static_total_score),
        ),
        turn_fit_articulated_scores=None,
    )
    if not unexplained_count:
        return diagnostic_candidate

    observed_classes = candidate.assignment_observed_class_ids
    if observed_classes is not None:
        observed_classes = np.asarray(observed_classes, dtype=np.int32).reshape(-1)
        if len(observed_classes) != len(image_points):
            observed_classes = None
    use_class_aware_assignment = PNP_ASSIGNMENT_MODE == "class-aware"
    existing_projected_classes = candidate.assignment_projected_class_ids
    if existing_projected_classes is not None:
        existing_projected_classes = np.asarray(
            existing_projected_classes,
            dtype=np.int32,
        ).reshape(-1)
        if len(existing_projected_classes) != len(image_points):
            existing_projected_classes = None

    fixed_object_points = np.asarray(
        candidate.object_points,
        dtype=np.float32,
    ).reshape(-1, 3)
    fixed_projected_points = projected_points[fixed_positions].copy()
    fixed_projected_classes = (
        existing_projected_classes[fixed_positions].copy()
        if existing_projected_classes is not None
        else np.full(len(fixed_positions), -1, dtype=np.int32)
    )

    def assign_trial(
        trial_object_points: np.ndarray,
        trial_projected_points: np.ndarray,
        trial_projected_class_ids: np.ndarray,
        trial_fixed_positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return complete assignment arrays and per-row errors.

        Do not deduplicate here. Repeated projected targets are intentional
        during layer-rotation scoring because every observation must be able
        to contribute evidence for or against a hypothesis. Deduplication is
        restricted to the final solvePnP input below.
        """

        assigned_objects = np.empty_like(object_points)
        assigned_projected = np.empty_like(projected_points)
        assigned_classes = np.full(len(image_points), -1, dtype=np.int32)
        preserved_fixed = np.isin(fixed_positions, trial_fixed_positions)
        preserved_positions = fixed_positions[preserved_fixed]
        preserved_slots = np.flatnonzero(preserved_fixed)
        free_positions = np.setdiff1d(
            np.arange(len(image_points), dtype=np.int32),
            preserved_positions,
            assume_unique=False,
        )
        if len(preserved_positions):
            assigned_objects[preserved_positions] = fixed_object_points[preserved_slots]
            assigned_projected[preserved_positions] = fixed_projected_points[preserved_slots]
            assigned_classes[preserved_positions] = fixed_projected_classes[preserved_slots]

        if len(free_positions):
            free_observed_classes = (
                observed_classes[free_positions]
                if use_class_aware_assignment and observed_classes is not None
                else None
            )
            _score, rows, columns = (
                observed_keypoint_reprojection_nearest_assignment(
                    image_points[free_positions],
                    trial_projected_points,
                    (0, 0),
                    observed_class_ids=free_observed_classes,
                    projected_class_ids=(
                        trial_projected_class_ids
                        if use_class_aware_assignment
                        else None
                    ),
                )
            )
            if len(rows) != len(free_positions):
                return (
                    assigned_objects,
                    assigned_projected,
                    assigned_classes,
                    np.full(len(image_points), float("inf"), dtype=np.float64),
                )
            assigned_objects[free_positions[rows]] = trial_object_points[columns]
            assigned_projected[free_positions[rows]] = trial_projected_points[columns]
            assigned_classes[free_positions[rows]] = trial_projected_class_ids[columns]

        assigned_distances = np.linalg.norm(
            image_points - assigned_projected,
            axis=1,
        )
        return (
            assigned_objects,
            assigned_projected,
            assigned_classes,
            assigned_distances,
        )

    excluded_layer = initial_sticker_layer(candidate)
    layer_scores, _layer_unexplained_count = score_turn_layer_faces(
        candidate,
        base_object_points,
        object_points,
        distances,
        fixed_mask,
        assignment_error_limit,
        excluded_layer,
    )
    candidate = replace(
        diagnostic_candidate,
        turn_layer_scores=layer_scores,
    )
    if not layer_scores or max_face_candidates <= 0:
        return candidate
    selected_layer_label = layer_scores[0][0]
    selected_layer = next(
        (
            (axis, side)
            for axis in range(3)
            for side in (-1, 1)
            if layer_face_debug_label(axis, side) == selected_layer_label
        ),
        None,
    )
    if selected_layer is None:
        return candidate
    search_axis, search_side = selected_layer

    def rank_assignment(
        assigned_distances: np.ndarray,
        assigned_classes: np.ndarray,
        assigned_objects: np.ndarray,
        trial_rotations: tuple[CubeLayerRotation, ...],
    ) -> tuple[tuple[float, float, float, float], float, int, float, float]:
        """Rank the selected layer by local keypoint fit only."""

        finite_distances = np.isfinite(assigned_distances)
        if not np.any(finite_distances):
            return (float("inf"),) * 4, float("inf"), 0, float("inf"), float("inf")
        trial_layer_count = (
            1 if not trial_rotations else int(trial_rotations[-1].layer_count)
        )
        trial_layer_mask = cube_layer_mask(
            assigned_objects,
            search_axis,
            search_side,
            layer_count=trial_layer_count,
        )
        layer_finite = finite_distances & trial_layer_mask
        if not np.any(layer_finite):
            return (float("inf"),) * 4, float("inf"), 0, float("inf"), float("inf")
        valid_distances = assigned_distances[layer_finite]
        # Rank/acceptance uses the total number of explained observations,
        # not just the observations on the searched layer.  The layer-local
        # residual below still keeps the inexpensive angle search focused.
        inlier_count = int(
            np.count_nonzero(
                finite_distances & (assigned_distances <= assignment_error_limit)
            )
        )
        keypoint_error = float(
            np.mean(huber_loss(valid_distances))
            + PNP_ASSIGNMENT_TAIL_WEIGHT * np.percentile(valid_distances, 90)
            + 0.10 * np.percentile(valid_distances, 95)
        )
        if (
            use_class_aware_assignment
            and observed_classes is not None
            and len(assigned_classes) == len(observed_classes)
        ):
            known = (
                (observed_classes >= 0)
                & (assigned_classes >= 0)
                & layer_finite
            )
            if np.any(known):
                if np.any(observed_classes[known] != assigned_classes[known]):
                    return (
                        (float("inf"),) * 4,
                        float("inf"),
                        0,
                        float("inf"),
                        float("inf"),
                    )
        mean_error = float(np.mean(valid_distances))
        p90_error = float(np.percentile(valid_distances, 90))
        sticker_error = 0.0
        robust_error = keypoint_error
        rank = (-float(inlier_count), robust_error, p90_error, mean_error)
        return rank, robust_error, inlier_count, keypoint_error, sticker_error

    best = None
    base_rotations = tuple(candidate.layer_rotations)
    shortlisted_faces = (selected_layer,)
    angles = np.arange(
        PNP_LAYER_MISALIGNMENT_MIN_DEGREES,
        PNP_LAYER_MISALIGNMENT_MAX_DEGREES
        + 0.5 * PNP_LAYER_MISALIGNMENT_STEP_DEGREES,
        PNP_LAYER_MISALIGNMENT_STEP_DEGREES,
        dtype=np.float64,
    )

    hypotheses: list[tuple[CubeLayerRotation, ...]] = [base_rotations]
    for axis, side in shortlisted_faces:
        for layer_count in (1, 2):
            layer_mask = cube_layer_mask(
                base_object_points,
                axis,
                side,
                layer_count,
            )
            if int(layer_mask.sum()) < 3:
                continue
            for angle in angles:
                if abs(float(angle)) < 1e-6:
                    continue
                hypotheses.append(
                    base_rotations
                    + (
                        CubeLayerRotation(
                            axis,
                            side,
                            float(angle),
                            layer_count,
                        ),
                    )
                )

    def evaluate_trial(
        trial_rotations: tuple[CubeLayerRotation, ...],
    ) -> dict[str, object] | None:
        # A sticker point on the layer being tested cannot remain fixed: the
        # sticker itself moves with that layer.  Release only those anchor
        # rows; anchors on the other layers remain the pose pivot.
        trial_fixed_positions = fixed_positions.copy()
        if len(trial_fixed_positions):
            unlock_fixed = np.zeros(len(fixed_positions), dtype=bool)
            for layer_rotation in trial_rotations:
                unlock_fixed |= cube_layer_mask(
                    fixed_object_points,
                    layer_rotation.axis,
                    layer_rotation.side,
                    layer_rotation.layer_count,
                )
            trial_fixed_positions = fixed_positions[~unlock_fixed]
        try:
            trial_object_points, trial_projected_class_ids = (
                visible_transformed_cube_lattice_points(
                    candidate.rvec,
                    candidate.tvec,
                    trial_rotations,
                )
            )
        except (cv2.error, ValueError):
            return None
        if len(trial_object_points) == 0:
            return None
        try:
            trial_projected_points, _ = cv2.projectPoints(
                trial_object_points,
                candidate.rvec,
                candidate.tvec,
                candidate.camera_matrix,
                candidate.dist_coeffs,
            )
        except cv2.error:
            return None
        trial_projected_points = trial_projected_points.reshape(-1, 2)
        (
            trial_assignment_objects,
            trial_assignment_projected,
            trial_assignment_classes,
            trial_distances,
        ) = assign_trial(
            trial_object_points,
            trial_projected_points,
            trial_projected_class_ids,
            trial_fixed_positions,
        )
        rank, robust_error, inlier_count, keypoint_error, sticker_error = rank_assignment(
            trial_distances,
            trial_assignment_classes,
            trial_assignment_objects,
            trial_rotations,
        )
        return {
            "is_baseline": trial_rotations == base_rotations,
            "rank": rank,
            "label": layer_rotation_debug_label(trial_rotations),
            "robust_error": robust_error,
            "keypoint_error": keypoint_error,
            "sticker_error": sticker_error,
            "inlier_count": inlier_count,
            "p90_error": float(rank[2]),
            "mean_error": float(rank[3]),
            "layer_rotations": trial_rotations,
            "assignment_projected_points": trial_assignment_projected.copy(),
            "assignment_object_points": trial_assignment_objects.copy(),
            "assignment_projected_class_ids": trial_assignment_classes.copy(),
        }

    ranked_trials = []
    evaluated_labels: set[str] = set()
    for trial_rotations in hypotheses:
        trial = evaluate_trial(trial_rotations)
        if trial is not None:
            ranked_trials.append(trial)
            evaluated_labels.add(str(trial["label"]))

    ordered_trials = sorted(ranked_trials, key=lambda trial: trial["rank"])
    baseline_trial = next(
        (trial for trial in ordered_trials if trial["is_baseline"]),
        None,
    )
    baseline_inlier_count = (
        int(baseline_trial["inlier_count"])
        if baseline_trial is not None
        else int(np.count_nonzero(accepted & ~fixed_mask))
    )
    # The static pose is the default.  A turn hypothesis is eligible only if
    # it explains strictly more observations; lower residual with the same
    # number of observations is not enough to change the pose.
    improving_trials = [
        trial
        for trial in ordered_trials
        if not trial["is_baseline"]
        and int(trial["inlier_count"]) > baseline_inlier_count
    ]
    articulated_scores: tuple[float, float, float] | None = None
    best = None
    if improving_trials:
        best_trial = min(improving_trials, key=lambda trial: trial["rank"])
        articulated_keypoint = float(best_trial["keypoint_error"])
        articulated_scores = (
            articulated_keypoint,
            0.0,
            articulated_keypoint,
        )
        best = (
            best_trial["rank"],
            best_trial["robust_error"],
            best_trial["inlier_count"],
            best_trial["layer_rotations"],
            best_trial["assignment_object_points"],
            best_trial["assignment_projected_points"],
            best_trial["assignment_projected_class_ids"],
        )
    debug_trials = []
    if baseline_trial is not None:
        debug_trials.append(baseline_trial)
    debug_trials.extend(
        trial
        for trial in ordered_trials
        if trial is not baseline_trial
    )
    display_trials = debug_trials[:PNP_TURN_DEBUG_CANDIDATE_COUNT]
    recorded_trials = ordered_trials if record_all_turn_candidates else display_trials
    turn_candidate_scores = tuple(
        (
            trial["label"],
            trial["robust_error"],
            trial["inlier_count"],
            trial["p90_error"],
            trial["mean_error"],
        )
        for trial in recorded_trials
    )
    turn_candidate_debug = tuple(
        TurnCandidateDebug(
            label=trial["label"],
            layer_rotations=trial["layer_rotations"],
            assignment_projected_points=trial["assignment_projected_points"],
            assignment_object_points=trial["assignment_object_points"],
            assignment_projected_class_ids=trial["assignment_projected_class_ids"],
        )
        for trial in recorded_trials
    )

    if best is None:
        return replace(
            candidate,
            turn_candidate_scores=turn_candidate_scores,
            turn_candidate_debug=turn_candidate_debug,
            turn_fit_articulated_scores=articulated_scores,
        )
    (
        _rank,
        robust_error,
        _inlier_count,
        best_rotations,
        best_object_points,
        best_projected_points,
        best_projected_classes,
    ) = best
    best_sticker_object_points = apply_cube_layer_rotations(
        candidate.object_points,
        best_rotations,
    )
    try:
        best_sticker_projected_points, _ = cv2.projectPoints(
            best_sticker_object_points,
            candidate.rvec,
            candidate.tvec,
            candidate.camera_matrix,
            candidate.dist_coeffs,
        )
        best_sticker_projected_points = best_sticker_projected_points.reshape(-1, 2)
    except cv2.error:
        best_sticker_projected_points = candidate.projected_sticker_points
    return replace(
        candidate,
        object_points=best_sticker_object_points,
        projected_sticker_points=best_sticker_projected_points,
        assignment_object_points=best_object_points,
        assignment_projected_points=best_projected_points,
        assignment_projected_class_ids=best_projected_classes,
        overall_reprojection_error=robust_error,
        layer_rotations=best_rotations,
        turn_candidate_scores=turn_candidate_scores,
        turn_candidate_debug=turn_candidate_debug,
        turn_fit_articulated_scores=articulated_scores,
    )


def recover_misaligned_cube_layers(
    candidate: StickerPnPCandidate,
    assignment_error_limit: float,
    max_face_candidates: int = PNP_TURN_FACE_CANDIDATES,
    record_all_turn_candidates: bool = False,
) -> StickerPnPCandidate:
    """Search every single/double layer turn using combined local fit.

    The static pose remains the default. A turn is accepted only when it
    explains more observations than its corresponding static layer hypothesis
    and its selected configuration has a normalized robust combined fit below
    the pose threshold. Configurations are ranked globally by mean normalized
    error; the median-plus-p90 check is applied only after selecting the best
    configuration. Sticker boundaries and contours are deliberately not part
    of this search.
    """

    image_points = candidate.assignment_image_points
    projected_points = candidate.assignment_projected_points
    object_points = candidate.assignment_object_points
    if image_points is None or projected_points is None or object_points is None:
        return candidate
    image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    projected_points = np.asarray(projected_points, dtype=np.float32).reshape(-1, 2)
    object_points = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
    if (
        len(image_points) < 3
        or len(image_points) != len(projected_points)
        or len(image_points) != len(object_points)
        or not np.isfinite(assignment_error_limit)
    ):
        return candidate

    distances = np.linalg.norm(image_points - projected_points, axis=1)
    finite = (
        np.isfinite(image_points).all(axis=1)
        & np.isfinite(projected_points).all(axis=1)
        & np.isfinite(object_points).all(axis=1)
        & np.isfinite(distances)
    )
    accepted = finite & (distances <= assignment_error_limit)
    base_object_points = np.asarray(CUBE_LATTICE_OBJECT_POINTS_MM, dtype=np.float32)
    if len(base_object_points) == 0:
        return candidate

    fixed_mask = candidate.assignment_fixed_mask
    if fixed_mask is not None:
        fixed_mask = np.asarray(fixed_mask, dtype=bool).reshape(-1)
    if fixed_mask is None or len(fixed_mask) != len(image_points):
        fixed_mask = np.zeros(len(image_points), dtype=bool)
        fixed_mask[: min(len(candidate.image_points), len(image_points))] = True
    fixed_positions = np.flatnonzero(fixed_mask).astype(np.int32)
    if len(fixed_positions) != len(candidate.object_points):
        fixed_positions = np.empty((0,), dtype=np.int32)

    unexplained_count = int(np.count_nonzero(finite & ~accepted & ~fixed_mask))
    static_fit = pose_reprojection_fit_metric(candidate)
    static_sticker_fit, static_sticker_matches = sticker_mask_center_fit_metric(
        candidate,
        candidate.rvec,
        candidate.tvec,
        PNP_FINAL_ASSIGNMENT_ERROR_FRACTION * projected_cube_diameter(candidate),
    )
    observed_sticker_count = len(getattr(candidate, "sticker_mask_centers", ()))
    sticker_error = bool(
        observed_sticker_count
        and (
            static_sticker_matches < observed_sticker_count
            or not np.isfinite(static_sticker_fit)
            or static_sticker_fit > PNP_POSE_REPROJECTION_ERROR_THRESHOLD
        )
    )
    diagnostic_candidate = replace(
        candidate,
        turn_layer_scores=(),
        turn_fit_sticker_score=static_sticker_fit,
        turn_fit_unexplained_keypoints=unexplained_count,
        turn_fit_evaluated=True,
        turn_fit_static_scores=(
            static_fit,
            static_sticker_fit,
            static_fit + PNP_TURN_MASK_CENTER_WEIGHT * static_sticker_fit,
        ),
        turn_fit_articulated_scores=None,
    )
    if (not unexplained_count and not sticker_error) or max_face_candidates <= 0:
        return diagnostic_candidate

    cube_scale = projected_cube_diameter(candidate)
    if not np.isfinite(cube_scale) or cube_scale <= 0.0:
        return diagnostic_candidate

    observed_classes = candidate.assignment_observed_class_ids
    if observed_classes is not None:
        observed_classes = np.asarray(observed_classes, dtype=np.int32).reshape(-1)
        if len(observed_classes) != len(image_points):
            observed_classes = None
    use_class_aware_assignment = PNP_ASSIGNMENT_MODE == "class-aware"
    existing_projected_classes = candidate.assignment_projected_class_ids
    if existing_projected_classes is not None:
        existing_projected_classes = np.asarray(
            existing_projected_classes,
            dtype=np.int32,
        ).reshape(-1)
        if len(existing_projected_classes) != len(image_points):
            existing_projected_classes = None

    fixed_object_points = np.asarray(candidate.object_points, dtype=np.float32).reshape(-1, 3)
    fixed_projected_points = projected_points[fixed_positions].copy()
    fixed_projected_classes = (
        existing_projected_classes[fixed_positions].copy()
        if existing_projected_classes is not None
        else np.full(len(fixed_positions), -1, dtype=np.int32)
    )

    def assign_trial(
        trial_object_points: np.ndarray,
        trial_projected_points: np.ndarray,
        trial_projected_class_ids: np.ndarray,
        trial_fixed_positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assigned_objects = np.empty_like(object_points)
        assigned_projected = np.empty_like(projected_points)
        assigned_classes = np.full(len(image_points), -1, dtype=np.int32)
        preserved_fixed = np.isin(fixed_positions, trial_fixed_positions)
        preserved_positions = fixed_positions[preserved_fixed]
        preserved_slots = np.flatnonzero(preserved_fixed)
        free_positions = np.setdiff1d(
            np.arange(len(image_points), dtype=np.int32),
            preserved_positions,
            assume_unique=False,
        )
        if len(preserved_positions):
            assigned_objects[preserved_positions] = fixed_object_points[preserved_slots]
            assigned_projected[preserved_positions] = fixed_projected_points[preserved_slots]
            assigned_classes[preserved_positions] = fixed_projected_classes[preserved_slots]
        if len(free_positions):
            free_observed_classes = (
                observed_classes[free_positions]
                if use_class_aware_assignment and observed_classes is not None
                else None
            )
            _score, rows, columns = observed_keypoint_reprojection_nearest_assignment(
                image_points[free_positions],
                trial_projected_points,
                (0, 0),
                observed_class_ids=free_observed_classes,
                projected_class_ids=(
                    trial_projected_class_ids
                    if use_class_aware_assignment
                    else None
                ),
            )
            if len(rows) != len(free_positions):
                return (
                    assigned_objects,
                    assigned_projected,
                    assigned_classes,
                    np.full(len(image_points), float("inf"), dtype=np.float64),
                )
            assigned_objects[free_positions[rows]] = trial_object_points[columns]
            assigned_projected[free_positions[rows]] = trial_projected_points[columns]
            assigned_classes[free_positions[rows]] = trial_projected_class_ids[columns]
        assigned_distances = np.linalg.norm(
            image_points - assigned_projected,
            axis=1,
        )
        return assigned_objects, assigned_projected, assigned_classes, assigned_distances

    def rank_assignment(
        assigned_distances: np.ndarray,
        assigned_classes: np.ndarray,
        assigned_objects: np.ndarray,
        layer_axis: int,
        layer_side: int,
        layer_count: int,
    ) -> tuple[
        tuple[float, float, float, float],
        float,
        float,
        int,
        float,
        float,
        int,
    ]:
        finite_distances = np.isfinite(assigned_distances)
        layer_mask = cube_layer_mask(
            assigned_objects,
            layer_axis,
            layer_side,
            layer_count=layer_count,
        )
        layer_finite = finite_distances & layer_mask
        if not np.any(layer_finite):
            return (
                (float("inf"),) * 4,
                float("inf"),
                float("inf"),
                0,
                float("inf"),
                float("inf"),
                0,
            )
        if (
            use_class_aware_assignment
            and observed_classes is not None
            and len(assigned_classes) == len(observed_classes)
        ):
            known = (
                (observed_classes >= 0)
                & (assigned_classes >= 0)
                & layer_finite
            )
            if np.any(known) and np.any(
                observed_classes[known] != assigned_classes[known]
            ):
                return (
                    (float("inf"),) * 4,
                    float("inf"),
                    float("inf"),
                    0,
                    float("inf"),
                    float("inf"),
                    0,
                )
        global_normalized = np.asarray(
            assigned_distances[finite_distances],
            dtype=np.float64,
        ) / cube_scale
        total_inlier_count = int(
            np.count_nonzero(
                finite_distances & (assigned_distances <= assignment_error_limit)
            )
        )
        layer_inlier_count = int(
            np.count_nonzero(
                layer_finite & (assigned_distances <= assignment_error_limit)
            )
        )
        mean_error = float(np.mean(global_normalized))
        robust_error = float(
            np.median(global_normalized)
            + 0.5 * np.percentile(global_normalized, 90)
        )
        p90_error = float(np.percentile(global_normalized, 90))
        rank = (
            mean_error,
            -float(total_inlier_count),
            p90_error,
            -float(layer_inlier_count),
        )
        return (
            rank,
            mean_error,
            robust_error,
            total_inlier_count,
            mean_error,
            p90_error,
            layer_inlier_count,
        )

    def evaluate_trial(
        trial_rotations: tuple[CubeLayerRotation, ...],
        layer_axis: int,
        layer_side: int,
        layer_count: int,
        is_baseline: bool,
    ) -> dict[str, object] | None:
        trial_fixed_positions = fixed_positions.copy()
        if len(trial_fixed_positions):
            unlock_fixed = np.zeros(len(fixed_positions), dtype=bool)
            for layer_rotation in trial_rotations:
                unlock_fixed |= cube_layer_mask(
                    fixed_object_points,
                    layer_rotation.axis,
                    layer_rotation.side,
                    layer_rotation.layer_count,
                )
            trial_fixed_positions = fixed_positions[~unlock_fixed]
        try:
            trial_object_points, trial_projected_class_ids = visible_transformed_cube_lattice_points(
                candidate.rvec,
                candidate.tvec,
                trial_rotations,
            )
            trial_projected_points, _ = cv2.projectPoints(
                trial_object_points,
                candidate.rvec,
                candidate.tvec,
                candidate.camera_matrix,
                candidate.dist_coeffs,
            )
        except (cv2.error, ValueError):
            return None
        if len(trial_object_points) == 0:
            return None
        trial_projected_points = trial_projected_points.reshape(-1, 2)
        (
            trial_assignment_objects,
            trial_assignment_projected,
            trial_assignment_classes,
            trial_distances,
        ) = assign_trial(
            trial_object_points,
            trial_projected_points,
            trial_projected_class_ids,
            trial_fixed_positions,
        )
        (
            rank,
            keypoint_mean_error,
            keypoint_robust_error,
            total_inlier_count,
            mean_error,
            p90_error,
            layer_inlier_count,
        ) = rank_assignment(
            trial_distances,
            trial_assignment_classes,
            trial_assignment_objects,
            layer_axis,
            layer_side,
            layer_count,
        )
        sticker_mean_error, sticker_robust_error, sticker_match_count = (
            sticker_mask_center_turn_fit_metrics(
                candidate,
                candidate.rvec,
                candidate.tvec,
                trial_rotations,
                PNP_FINAL_ASSIGNMENT_ERROR_FRACTION * cube_scale,
            )
        )
        combined_mean_error = keypoint_mean_error + (
            PNP_TURN_MASK_CENTER_WEIGHT * sticker_mean_error
            if np.isfinite(sticker_mean_error)
            else float("inf")
        )
        combined_robust_error = keypoint_robust_error + (
            PNP_TURN_MASK_CENTER_WEIGHT * sticker_robust_error
            if np.isfinite(sticker_robust_error)
            else float("inf")
        )
        rank = (
            combined_mean_error,
            -float(total_inlier_count),
            p90_error,
            -float(layer_inlier_count),
        )
        return {
            "is_baseline": is_baseline,
            "rank": rank,
            "label": (
                f"static {layer_face_debug_label(layer_axis, layer_side)}"
                if is_baseline
                else layer_rotation_debug_label(trial_rotations)
            ),
            "robust_error": combined_robust_error,
            "keypoint_error": keypoint_mean_error,
            "sticker_error": sticker_mean_error,
            "combined_mean_error": combined_mean_error,
            "keypoint_robust_error": keypoint_robust_error,
            "sticker_robust_error": sticker_robust_error,
            "sticker_match_count": sticker_match_count,
            "inlier_count": total_inlier_count,
            "layer_inlier_count": layer_inlier_count,
            "p90_error": p90_error,
            "mean_error": mean_error,
            "layer_axis": layer_axis,
            "layer_side": layer_side,
            "layer_count": layer_count,
            "layer_rotations": trial_rotations,
            "assignment_projected_points": trial_assignment_projected.copy(),
            "assignment_object_points": trial_assignment_objects.copy(),
            "assignment_projected_class_ids": trial_assignment_classes.copy(),
        }

    base_rotations = tuple(candidate.layer_rotations)
    angles = np.arange(
        PNP_LAYER_MISALIGNMENT_MIN_DEGREES,
        PNP_LAYER_MISALIGNMENT_MAX_DEGREES
        + 0.5 * PNP_LAYER_MISALIGNMENT_STEP_DEGREES,
        PNP_LAYER_MISALIGNMENT_STEP_DEGREES,
        dtype=np.float64,
    )
    ranked_trials: list[dict[str, object]] = []
    for axis in range(3):
        for side in (-1, 1):
            for layer_count in (1, 2):
                baseline = evaluate_trial(
                    base_rotations,
                    axis,
                    side,
                    layer_count,
                    True,
                )
                if baseline is not None:
                    ranked_trials.append(baseline)
                if int(cube_layer_mask(base_object_points, axis, side, layer_count).sum()) < 3:
                    continue
                for angle in angles:
                    if abs(float(angle)) < 1e-6:
                        continue
                    rotation = CubeLayerRotation(
                        axis,
                        side,
                        float(angle),
                        layer_count,
                    )
                    trial = evaluate_trial(
                        base_rotations + (rotation,),
                        axis,
                        side,
                        layer_count,
                        False,
                    )
                    if trial is not None:
                        ranked_trials.append(trial)

    ordered_trials = sorted(ranked_trials, key=lambda trial: trial["rank"])
    static_inlier_count = int(np.count_nonzero(accepted))
    improving_trials = [
        trial
        for trial in ordered_trials
        if not bool(trial["is_baseline"])
        and int(trial["inlier_count"])
        > static_inlier_count
    ]
    articulated_scores: tuple[float, float, float] | None = None
    best: dict[str, object] | None = None
    if improving_trials:
        best = min(improving_trials, key=lambda trial: trial["rank"])
        articulated_keypoint = float(best["keypoint_error"])
        articulated_sticker = float(best["sticker_error"])
        articulated_scores = (
            articulated_keypoint,
            articulated_sticker,
            float(best["combined_mean_error"]),
        )
        if (
            not np.isfinite(float(best["robust_error"]))
            or float(best["robust_error"]) > PNP_POSE_REPROJECTION_ERROR_THRESHOLD
        ):
            best = None

    layer_scores = tuple(
        (
            str(trial["label"]),
            float(trial["keypoint_error"]),
            float(trial["sticker_error"]),
            float(trial["combined_mean_error"]),
        )
        for trial in ordered_trials
        if bool(trial["is_baseline"]) and int(trial["layer_count"]) == 1
    )
    recorded_trials = (
        ordered_trials
        if record_all_turn_candidates
        else ordered_trials[:PNP_TURN_DEBUG_CANDIDATE_COUNT]
    )
    turn_candidate_scores = tuple(
        (
            str(trial["label"]),
            float(trial["robust_error"]),
            int(trial["inlier_count"]),
            float(trial["p90_error"]),
            float(trial["mean_error"]),
        )
        for trial in recorded_trials
    )
    turn_candidate_debug = tuple(
        TurnCandidateDebug(
            label=str(trial["label"]),
            layer_rotations=trial["layer_rotations"],
            assignment_projected_points=trial["assignment_projected_points"],
            assignment_object_points=trial["assignment_object_points"],
            assignment_projected_class_ids=trial["assignment_projected_class_ids"],
        )
        for trial in recorded_trials
    )
    turn_candidate_diagnostics = tuple(
        {
            "label": str(trial["label"]),
            "is_baseline": bool(trial["is_baseline"]),
            "layer_axis": int(trial["layer_axis"]),
            "layer_side": int(trial["layer_side"]),
            "layer_count": int(trial["layer_count"]),
            "layer_rotations": [
                {
                    "axis": int(rotation.axis),
                    "side": int(rotation.side),
                    "angle_degrees": float(rotation.angle_degrees),
                    "layer_count": int(rotation.layer_count),
                }
                for rotation in trial["layer_rotations"]
            ],
            "global_mean_error": float(trial["combined_mean_error"]),
            "global_robust_error": float(trial["robust_error"]),
            "keypoint_mean_error": float(trial["keypoint_error"]),
            "keypoint_robust_error": float(trial["keypoint_robust_error"]),
            "keypoint_p90_error": float(trial["p90_error"]),
            "heatmap_mean_error": float(trial["sticker_error"]),
            "heatmap_robust_error": float(trial["sticker_robust_error"]),
            "heatmap_match_count": int(trial["sticker_match_count"]),
            "inlier_count": int(trial["inlier_count"]),
            "layer_inlier_count": int(trial["layer_inlier_count"]),
        }
        for trial in recorded_trials
    )
    diagnostic_candidate = replace(
        diagnostic_candidate,
        turn_layer_scores=layer_scores,
        turn_candidate_scores=turn_candidate_scores,
        turn_candidate_debug=turn_candidate_debug,
        turn_candidate_diagnostics=turn_candidate_diagnostics,
        turn_fit_articulated_scores=articulated_scores,
    )
    if best is None:
        return diagnostic_candidate

    best_rotations = best["layer_rotations"]
    best_sticker_object_points = apply_cube_layer_rotations(
        candidate.object_points,
        best_rotations,
    )
    try:
        best_sticker_projected_points, _ = cv2.projectPoints(
            best_sticker_object_points,
            candidate.rvec,
            candidate.tvec,
            candidate.camera_matrix,
            candidate.dist_coeffs,
        )
        best_sticker_projected_points = best_sticker_projected_points.reshape(-1, 2)
    except cv2.error:
        best_sticker_projected_points = candidate.projected_sticker_points
    return replace(
        diagnostic_candidate,
        object_points=best_sticker_object_points,
        projected_sticker_points=best_sticker_projected_points,
        assignment_object_points=best["assignment_object_points"],
        assignment_projected_points=best["assignment_projected_points"],
        assignment_projected_class_ids=best["assignment_projected_class_ids"],
        layer_rotations=best_rotations,
        turn_fit_articulated_scores=articulated_scores,
    )


def sticker_identity_object_center(identity: StickerIdentity) -> np.ndarray:
    """Return the 3-D center of one physical sticker face."""

    center = cubelet_center_object_point(*identity.cubie_index).astype(np.float64)
    center[int(identity.axis)] += int(identity.side) * CUBE_LAYER_SIZE_MM / 2.0
    return center


def matched_sticker_mask_center_correspondences(
    candidate: StickerPnPCandidate,
    rvec: np.ndarray,
    tvec: np.ndarray,
    max_error: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Match mask centers to visible projected sticker centers within a gate."""

    observed = np.asarray(candidate.sticker_mask_centers, dtype=np.float64).reshape(-1, 2)
    if len(observed) == 0 or not np.isfinite(observed).all() or not np.isfinite(max_error):
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )
    try:
        projected_faces = projected_sticker_faces(candidate, rvec, tvec)
    except (cv2.error, ValueError, FloatingPointError):
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )
    if not projected_faces:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )
    projected = np.asarray(
        [np.mean(np.asarray(face[2], dtype=np.float64), axis=0) for face in projected_faces],
        dtype=np.float64,
    )
    object_centers = np.asarray(
        [sticker_identity_object_center(face[0]) for face in projected_faces],
        dtype=np.float64,
    )
    distances = np.linalg.norm(observed[:, None, :] - projected[None, :, :], axis=2)
    pair_indices = [
        (float(distances[mask_index, face_index]), mask_index, face_index)
        for mask_index in range(len(observed))
        for face_index in range(len(projected))
    ]
    pair_indices.sort(key=lambda item: (item[0], item[1], item[2]))
    used_masks: set[int] = set()
    used_faces: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for distance, mask_index, face_index in pair_indices:
        if distance > max_error:
            break
        if mask_index in used_masks or face_index in used_faces:
            continue
        used_masks.add(mask_index)
        used_faces.add(face_index)
        matches.append((mask_index, face_index, distance))
    if not matches:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )
    return (
        np.asarray([observed[mask_index] for mask_index, _face_index, _distance in matches]),
        np.asarray([object_centers[face_index] for _mask_index, face_index, _distance in matches]),
        np.asarray([distance for _mask_index, _face_index, distance in matches]),
        np.asarray(
            [
                getattr(candidate, "sticker_mask_center_uncertainties", ())[mask_index]
                if mask_index
                < len(getattr(candidate, "sticker_mask_center_uncertainties", ()))
                else 1.0
                for mask_index, _face_index, _distance in matches
            ],
            dtype=np.float64,
        ),
    )


def uncertainty_weighted_center_fit_metric(
    distances: np.ndarray,
    uncertainties: np.ndarray,
    scale: float,
) -> float:
    """Return a robust cube-scale center residual with uncertainty weights."""

    _mean, robust = uncertainty_weighted_center_fit_metrics(
        distances,
        uncertainties,
        scale,
    )
    return robust


def uncertainty_weighted_center_fit_metrics(
    distances: np.ndarray,
    uncertainties: np.ndarray,
    scale: float,
) -> tuple[float, float]:
    """Return mean and robust cube-scale center residuals."""

    distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    uncertainties = np.asarray(uncertainties, dtype=np.float64).reshape(-1)
    if (
        len(distances) == 0
        or len(distances) != len(uncertainties)
        or not np.isfinite(scale)
        or scale <= 0.0
    ):
        return float("inf"), float("inf")
    uncertainties = np.maximum(uncertainties, 1.0)
    normalized = distances / scale
    weights = 1.0 / np.square(uncertainties)
    finite = np.isfinite(normalized) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(finite):
        return float("inf"), float("inf")
    normalized = normalized[finite]
    weights = weights[finite]
    mean = float(np.sum(weights * normalized) / max(float(np.sum(weights)), 1e-9))
    order = np.argsort(normalized, kind="stable")
    sorted_values = normalized[order]
    cumulative = np.cumsum(weights[order])
    total_weight = float(cumulative[-1])
    median_index = int(np.searchsorted(cumulative, 0.50 * total_weight, side="left"))
    p90_index = int(np.searchsorted(cumulative, 0.90 * total_weight, side="left"))
    median_index = min(median_index, len(sorted_values) - 1)
    p90_index = min(p90_index, len(sorted_values) - 1)
    robust = float(sorted_values[median_index] + 0.5 * sorted_values[p90_index])
    return mean, robust


def sticker_mask_center_fit_metric(
    candidate: StickerPnPCandidate,
    rvec: np.ndarray,
    tvec: np.ndarray,
    max_error: float,
) -> tuple[float, int]:
    """Score all heatmap centers against the current projected pose."""

    observed = np.asarray(
        getattr(candidate, "sticker_mask_centers", ()),
        dtype=np.float64,
    ).reshape(-1, 2)
    if len(observed) == 0:
        return 0.0, 0
    if not np.isfinite(max_error) or max_error <= 0.0:
        return float("inf"), 0
    (
        _image_points,
        _object_points,
        distances,
        uncertainties,
    ) = matched_sticker_mask_center_correspondences(
        candidate,
        rvec,
        tvec,
        max_error,
    )
    if len(distances) == 0:
        return float("inf"), 0
    scale = projected_cube_diameter(candidate, rvec, tvec)
    metric = uncertainty_weighted_center_fit_metric(
        distances,
        uncertainties,
        scale,
    )
    return metric, len(distances)


def sticker_mask_center_turn_fit_metrics(
    candidate: StickerPnPCandidate,
    rvec: np.ndarray,
    tvec: np.ndarray,
    layer_rotations: tuple[CubeLayerRotation, ...],
    max_error: float,
) -> tuple[float, float, int]:
    """Return global mean/robust heatmap fit for one turn configuration."""

    observed = np.asarray(
        getattr(candidate, "sticker_mask_centers", ()),
        dtype=np.float64,
    ).reshape(-1, 2)
    if len(observed) == 0:
        return 0.0, 0.0, 0
    if not np.isfinite(max_error) or max_error <= 0.0:
        return float("inf"), float("inf"), 0
    trial_candidate = replace(candidate, layer_rotations=layer_rotations)
    (
        _image_points,
        _object_points,
        distances,
        uncertainties,
    ) = matched_sticker_mask_center_correspondences(
        trial_candidate,
        rvec,
        tvec,
        max_error,
    )
    if len(distances) == 0:
        return float("inf"), float("inf"), 0
    scale = projected_cube_diameter(candidate, rvec, tvec)
    mean, robust = uncertainty_weighted_center_fit_metrics(
        distances,
        uncertainties,
        scale,
    )
    return mean, robust, len(distances)


def sticker_mask_center_layer_fit_metric(
    candidate: StickerPnPCandidate,
    rvec: np.ndarray,
    tvec: np.ndarray,
    layer_rotations: tuple[CubeLayerRotation, ...],
    layer_axis: int,
    layer_side: int,
    layer_count: int,
    max_error: float,
) -> tuple[float, int]:
    """Return the uncertainty-weighted heatmap-center fit for one layer.

    This intentionally scores only projected sticker faces belonging to the
    layer currently being tested.  It uses the same one-to-one gated center
    matching as static fitting, but leaves the center observations as fuzzy
    measurements: low-uncertainty centers carry more ranking weight while a
    broad/irregular mask cannot dominate the turn decision.
    """

    observed = np.asarray(
        getattr(candidate, "sticker_mask_centers", ()),
        dtype=np.float64,
    ).reshape(-1, 2)
    if (
        len(observed) == 0
        or not np.isfinite(observed).all()
        or not np.isfinite(max_error)
        or max_error <= 0.0
    ):
        return 0.0, 0
    trial_candidate = replace(candidate, layer_rotations=layer_rotations)
    try:
        projected_faces = projected_sticker_faces(trial_candidate, rvec, tvec)
    except (cv2.error, ValueError, FloatingPointError):
        return 0.0, 0
    if not projected_faces:
        return 0.0, 0

    layer_faces = []
    for identity, _depth, polygon in projected_faces:
        # The lattice-layer helper operates on boundary points, whereas a
        # sticker identity carries a cubie index. Use that discrete index so
        # side stickers on an outer-layer cubie are included as well.
        cubie_coordinate = identity.cubie_index[layer_axis]
        in_layer = (
            cubie_coordinate >= 3 - layer_count
            if layer_side > 0
            else cubie_coordinate < layer_count
        )
        if in_layer:
            layer_faces.append((identity, polygon))
    if not layer_faces:
        return 0.0, 0

    projected_centers = np.asarray(
        [np.mean(np.asarray(polygon, dtype=np.float64), axis=0) for _identity, polygon in layer_faces],
        dtype=np.float64,
    )
    if not np.isfinite(projected_centers).all():
        return 0.0, 0
    distances = np.linalg.norm(
        observed[:, None, :] - projected_centers[None, :, :],
        axis=2,
    )
    pair_indices = [
        (float(distances[mask_index, face_index]), mask_index, face_index)
        for mask_index in range(len(observed))
        for face_index in range(len(layer_faces))
    ]
    pair_indices.sort(key=lambda item: (item[0], item[1], item[2]))
    used_masks: set[int] = set()
    used_faces: set[int] = set()
    matched: list[tuple[int, float]] = []
    for distance, mask_index, face_index in pair_indices:
        if distance > max_error:
            break
        if mask_index in used_masks or face_index in used_faces:
            continue
        used_masks.add(mask_index)
        used_faces.add(face_index)
        matched.append((mask_index, distance))
    if not matched:
        return 0.0, 0

    uncertainties = np.asarray(
        getattr(candidate, "sticker_mask_center_uncertainties", ()),
        dtype=np.float64,
    ).reshape(-1)
    if len(uncertainties) != len(observed):
        uncertainties = np.ones(len(observed), dtype=np.float64)
    scale = projected_cube_diameter(candidate, rvec, tvec)
    metric = uncertainty_weighted_center_fit_metric(
        np.asarray([distance for _mask_index, distance in matched], dtype=np.float64),
        np.asarray(
            [uncertainties[mask_index] for mask_index, _distance in matched],
            dtype=np.float64,
        ),
        scale,
    )
    if not np.isfinite(metric):
        return 0.0, 0
    return metric, len(matched)


def draw_sticker_mask_centroids(
    image: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
    predicted: tuple[list[np.ndarray], object, np.ndarray, np.ndarray | None]
    | tuple[list[np.ndarray], object, np.ndarray]
    | None,
    sticker_mask_erosion: int = 0,
    show_sticker_identities: bool = True,
) -> np.ndarray:
    """Draw segmentation-derived sticker centers and optional IDs."""

    output = image.copy()
    if predicted is None:
        return output
    if len(predicted) == 4:
        local_masks = predicted[0]
    else:
        local_masks = predicted[0]
    centers = sticker_mask_centers_to_frame(
        local_masks,
        roi_xyxy,
        sticker_mask_erosion,
    )
    for index, center in enumerate(centers):
        x, y = np.round(center).astype(np.int32)
        if not (0 <= x < output.shape[1] and 0 <= y < output.shape[0]):
            continue
        cv2.circle(output, (int(x), int(y)), 6, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(
            output,
            (int(x), int(y)),
            4,
            STICKER_MASK_CENTROID_COLOR_BGR,
            -1,
            cv2.LINE_AA,
        )
        if show_sticker_identities:
            cv2.putText(
                output,
                f"M{index}",
                (int(x) + 6, int(y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                STICKER_MASK_CENTROID_COLOR_BGR,
                1,
                cv2.LINE_AA,
            )
    return output


def refine_pose_with_sticker_centers(
    candidate: StickerPnPCandidate,
    keypoint_object_points: np.ndarray,
    keypoint_image_points: np.ndarray,
    solve_object_points: np.ndarray,
    solve_image_points: np.ndarray,
    valid_indices: np.ndarray,
    pre_solve_quality: tuple[float, float],
    assignment_error_limit: float,
    validate_sticker_facing: bool,
) -> StickerPnPCandidate:
    """Add gated segmentation centers to the static pose refinement solve."""

    center_limit = PNP_FINAL_ASSIGNMENT_ERROR_FRACTION * projected_cube_diameter(
        candidate
    )
    if not np.isfinite(center_limit):
        center_limit = assignment_error_limit
    (
        center_image_points,
        center_object_points,
        pre_center_distances,
        _center_uncertainties,
    ) = (
        matched_sticker_mask_center_correspondences(
            candidate,
            candidate.rvec,
            candidate.tvec,
            center_limit,
        )
    )
    if len(center_image_points) == 0:
        return replace(
            candidate,
            mask_center_fit_error=float("inf"),
            mask_center_match_count=0,
        )
    try:
        ok, center_rvec, center_tvec = cv2.solvePnP(
            np.concatenate((solve_object_points, center_object_points), axis=0),
            np.concatenate((solve_image_points, center_image_points), axis=0),
            candidate.camera_matrix,
            candidate.dist_coeffs,
            rvec=np.asarray(candidate.rvec, dtype=np.float64).reshape(3, 1),
            tvec=np.asarray(candidate.tvec, dtype=np.float64).reshape(3, 1),
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        return candidate
    if not ok:
        return candidate
    center_rvec = np.asarray(center_rvec, dtype=np.float64).reshape(3, 1)
    center_tvec = np.asarray(center_tvec, dtype=np.float64).reshape(3, 1)
    if (
        not np.isfinite(center_rvec).all()
        or not np.isfinite(center_tvec).all()
        or float(center_tvec[2, 0]) <= 0.0
        or (
            validate_sticker_facing
            and not canonical_sticker_is_camera_facing(
                candidate.object_points,
                center_rvec,
                center_tvec,
            )
        )
    ):
        return candidate
    try:
        projected_sticker_points, _ = cv2.projectPoints(
            candidate.object_points,
            center_rvec,
            center_tvec,
            candidate.camera_matrix,
            candidate.dist_coeffs,
        )
        projected_assignment_points, _ = cv2.projectPoints(
            keypoint_object_points,
            center_rvec,
            center_tvec,
            candidate.camera_matrix,
            candidate.dist_coeffs,
        )
        projected_centers, _ = cv2.projectPoints(
            center_object_points,
            center_rvec,
            center_tvec,
            candidate.camera_matrix,
            candidate.dist_coeffs,
        )
    except cv2.error:
        return candidate
    projected_sticker_points = projected_sticker_points.reshape(-1, 2)
    projected_assignment_points = projected_assignment_points.reshape(-1, 2)
    projected_centers = projected_centers.reshape(-1, 2)
    post_assignment_distances = np.linalg.norm(
        projected_assignment_points - keypoint_image_points,
        axis=1,
    )
    post_solve_distances = post_assignment_distances[valid_indices]
    post_solve_mean_error = float(np.mean(post_solve_distances))
    post_solve_p90_error = float(np.percentile(post_solve_distances, 90))
    if (
        post_solve_mean_error > pre_solve_quality[0] + 1e-6
        or post_solve_p90_error > pre_solve_quality[1] + 1e-6
    ):
        return candidate
    post_center_distances = np.linalg.norm(
        projected_centers - center_image_points,
        axis=1,
    )
    diameter = projected_cube_diameter(candidate, center_rvec, center_tvec)
    center_delta = max(1.5, PNP_MASK_BOUNDARY_DISTANCE_DELTA_FRACTION * diameter)
    pre_center_score = float(np.mean(huber_loss(pre_center_distances, center_delta)))
    post_center_score = float(np.mean(huber_loss(post_center_distances, center_delta)))
    baseline_keypoint_distances = np.linalg.norm(
        np.asarray(
            candidate.assignment_projected_points,
            dtype=np.float64,
        ).reshape(-1, 2)
        - keypoint_image_points,
        axis=1,
    )
    baseline_keypoint_score = float(
        np.mean(huber_loss(baseline_keypoint_distances, center_delta))
    )
    post_keypoint_score = float(
        np.mean(huber_loss(post_assignment_distances, center_delta))
    )
    if post_keypoint_score + post_center_score >= (
        baseline_keypoint_score + pre_center_score - 1e-6
    ):
        return replace(
            candidate,
            mask_center_fit_error=pre_center_score,
            mask_center_match_count=len(center_image_points),
        )
    return replace(
        candidate,
        rvec=center_rvec,
        tvec=center_tvec,
        projected_sticker_points=projected_sticker_points,
        reprojection_error=float(
            np.mean(
                np.linalg.norm(
                    projected_sticker_points - candidate.image_points,
                    axis=1,
                )
            )
        ),
        overall_reprojection_error=float(np.mean(huber_loss(post_assignment_distances))),
        assignment_projected_points=projected_assignment_points,
        pre_final_rvec=np.asarray(candidate.rvec, dtype=np.float64).copy(),
        pre_final_tvec=np.asarray(candidate.tvec, dtype=np.float64).copy(),
        mask_center_fit_error=post_center_score,
        mask_center_match_count=len(center_image_points),
    )


def solve_pnp_from_low_error_assignments(
    candidate: StickerPnPCandidate,
    validate_sticker_facing: bool = True,
    max_turn_face_candidates: int = PNP_TURN_FACE_CANDIDATES,
    record_all_turn_candidates: bool = False,
    recover_layer_turns: bool = False,
) -> StickerPnPCandidate:
    """Run one final PnP solve using the candidate's full assignment.

    The static-pose path deliberately leaves ``recover_layer_turns`` off:
    sticker selection and full-cube refinement are useful before color
    recognition, while layer-turn recovery is move recognition and can make
    the displayed pose jump between different cube geometries.
    """

    # Static fitting is keypoint-first. Mask centers are added only after this
    # keypoint solve; articulated turn trials use the resulting heatmap-center
    # observations as a separate layer-local ranking cue.
    # Clear stale boundary diagnostics without evaluating mask geometry.
    candidate = replace(
        candidate,
        mask_fit_error=float("inf"),
        mask_boundary_sample_count=0,
    )

    image_points = candidate.assignment_image_points
    object_points = candidate.assignment_object_points
    if image_points is None or object_points is None:
        return candidate
    image_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    object_points = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
    if len(image_points) != len(object_points) or len(image_points) < 3:
        return candidate

    projected_points = candidate.assignment_projected_points
    if projected_points is None:
        return candidate
    projected_points = np.asarray(projected_points, dtype=np.float32).reshape(-1, 2)
    if len(projected_points) != len(image_points):
        return candidate
    cube_diameter = projected_cube_diameter(candidate)
    assignment_error_limit = (
        PNP_FINAL_ASSIGNMENT_ERROR_FRACTION * cube_diameter
    )
    if recover_layer_turns:
        candidate = recover_misaligned_cube_layers(
            candidate,
            assignment_error_limit,
            max_face_candidates=max_turn_face_candidates,
            record_all_turn_candidates=record_all_turn_candidates,
        )
    else:
        # Make the static-pose contract explicit even if a caller passes a
        # candidate that was previously used by the turn-recovery path.
        candidate = replace(
            candidate,
            layer_rotations=(),
            turn_candidate_scores=(),
            turn_candidate_debug=(),
        )
    image_points = np.asarray(candidate.assignment_image_points, dtype=np.float32).reshape(-1, 2)
    object_points = np.asarray(candidate.assignment_object_points, dtype=np.float32).reshape(-1, 3)
    projected_points = np.asarray(candidate.assignment_projected_points, dtype=np.float32).reshape(-1, 2)
    assignment_distances = np.linalg.norm(
        projected_points - image_points,
        axis=1,
    )
    finite = (
        np.isfinite(image_points).all(axis=1)
        & np.isfinite(object_points).all(axis=1)
        & np.isfinite(projected_points).all(axis=1)
        & np.isfinite(assignment_distances)
    )
    if np.isfinite(assignment_error_limit):
        finite &= assignment_distances <= assignment_error_limit
    if int(finite.sum()) < 3:
        return candidate
    valid_indices = np.flatnonzero(finite).astype(np.int32)
    # The final solve is intentionally judged only on the observations that
    # passed the scale-relative gate.  Rejected observations should not be
    # able to veto a solve that is supported by the inlier set.
    pre_solve_quality = assignment_geometric_quality(
        image_points[valid_indices],
        projected_points[valid_indices],
    )
    unique_valid_indices = unique_object_point_indices(
        object_points[valid_indices],
        image_points[valid_indices],
        projected_points[valid_indices],
    )
    # Naïve assignments intentionally remain many-to-one during hypothesis
    # scoring. Only the final PnP input needs unique 3-D targets.
    solve_indices = valid_indices[unique_valid_indices]
    if len(solve_indices) < 3:
        return candidate
    # Keep the solve input visible even if the optimized pose is later
    # rejected by the quality gate.  These markers answer which observations
    # were actually passed to solvePnP; ``pre_final_rvec`` remains the signal
    # that the refined pose itself was accepted for display.
    candidate = replace(
        candidate,
        final_solve_image_points=image_points[solve_indices].copy(),
    )

    try:
        if len(solve_indices) >= 4:
            ok, rvec, tvec = cv2.solvePnP(
                object_points[solve_indices],
                image_points[solve_indices],
                candidate.camera_matrix,
                candidate.dist_coeffs,
                rvec=np.asarray(candidate.rvec, dtype=np.float64).reshape(3, 1),
                tvec=np.asarray(candidate.tvec, dtype=np.float64).reshape(3, 1),
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        else:
            ok, rvec, tvec = cv2.solvePnP(
                object_points[solve_indices],
                image_points[solve_indices],
                candidate.camera_matrix,
                candidate.dist_coeffs,
                flags=cv2.SOLVEPNP_SQPNP,
            )
    except cv2.error:
        return candidate
    if not ok:
        return candidate
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    if (
        not np.isfinite(rvec).all()
        or not np.isfinite(tvec).all()
        or float(tvec[2, 0]) <= 0.0
        or (
            validate_sticker_facing
            and not canonical_sticker_is_camera_facing(
                candidate.object_points,
                rvec,
                tvec,
            )
        )
    ):
        return candidate

    try:
        projected_sticker_points, _ = cv2.projectPoints(
            candidate.object_points,
            rvec,
            tvec,
            candidate.camera_matrix,
            candidate.dist_coeffs,
        )
        projected_assignment_points, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            candidate.camera_matrix,
            candidate.dist_coeffs,
        )
    except cv2.error:
        return candidate
    projected_sticker_points = projected_sticker_points.reshape(-1, 2)
    projected_assignment_points = projected_assignment_points.reshape(-1, 2)
    sticker_error = float(
        np.mean(
            np.linalg.norm(
                projected_sticker_points - candidate.image_points,
                axis=1,
            )
        )
    )
    assignment_distances = np.linalg.norm(
        projected_assignment_points - image_points,
        axis=1,
    )
    post_solve_distances = assignment_distances[valid_indices]
    post_solve_mean_error = float(np.mean(post_solve_distances))
    post_solve_p90_error = float(np.percentile(post_solve_distances, 90))
    # Keep the final solve only when it improves the threshold-passing set.
    if (
        post_solve_mean_error > pre_solve_quality[0] + 1e-6
        or post_solve_p90_error > pre_solve_quality[1] + 1e-6
    ):
        # The mask stage only follows an accepted keypoint/PnP refinement.
        # Do not let it rescue or replace a rejected original pose solve.
        return candidate
    refined_candidate = replace(
        candidate,
        rvec=rvec,
        tvec=tvec,
        projected_sticker_points=projected_sticker_points,
        reprojection_error=sticker_error,
        overall_reprojection_error=float(
            np.mean(huber_loss(assignment_distances))
        ),
        assignment_projected_points=projected_assignment_points,
        pre_final_rvec=np.asarray(candidate.rvec, dtype=np.float64).copy(),
        pre_final_tvec=np.asarray(candidate.tvec, dtype=np.float64).copy(),
    )
    if recover_layer_turns:
        return refined_candidate
    return refine_pose_with_sticker_centers(
        refined_candidate,
        object_points,
        image_points,
        object_points[solve_indices],
        image_points[solve_indices],
        valid_indices,
        pre_solve_quality,
        assignment_error_limit,
        validate_sticker_facing,
    )


def best_sticker_pnp_candidates(
    frame_points: np.ndarray,
    point_class_ids: np.ndarray,
    matches: tuple[StickerKeypointMatch, ...],
    image_size: tuple[int, int],
    focal_scale: float,
    local_masks: list[np.ndarray] | None = None,
    roi_xyxy: tuple[int, int, int, int] | None = None,
    scoring_points: np.ndarray | None = None,
    sticker_mask_erosion: int = 0,
) -> tuple[StickerPnPCandidate, ...]:
    """Solve prioritized stickers and retain the best full-cube hypotheses.

    Up to three four-point stickers are tried in descending confidence order.
    All successful candidates are returned in score order so temporal logic
    can keep alternatives alive. If none of the four-point stickers yields a
    pose, the same process falls back to three-point stickers.
    """

    valid_matches = tuple(
        match
        for match in matches
        if match.sticker_type != "unknown" and len(match.keypoint_indices) in (3, 4)
    )
    if not valid_matches:
        return tuple()
    four_point_matches = tuple(
        match for match in valid_matches if len(match.keypoint_indices) == 4
    )
    sticker_contours_by_mask = (
        tuple(
            sticker_contour_to_frame(local_mask, roi_xyxy)
            for local_mask in (local_masks or ())
        )
        if local_masks is not None and roi_xyxy is not None
        else tuple()
    )
    sticker_contours = tuple(
        contour for contour in sticker_contours_by_mask if contour is not None
    )
    sticker_boundary_samples = tuple(
        sample
        for contour in sticker_contours
        if (sample := resample_closed_boundary(contour)).size
    )
    sticker_mask_centers, sticker_mask_center_uncertainties = (
        sticker_mask_center_observations_to_frame(
            local_masks,
            roi_xyxy,
            sticker_mask_erosion,
        )
    )

    def solve_matches(
        prioritized_matches: tuple[StickerKeypointMatch, ...],
    ) -> tuple[StickerPnPCandidate, ...]:
        candidates: list[StickerPnPCandidate] = []
        prioritized = sorted(prioritized_matches, key=sticker_match_priority_key)[
            :MAX_STICKER_PNP_CANDIDATES
        ]
        for match in prioritized:
            sticker_contour = None
            if (
                sticker_contours_by_mask
                and 0 <= match.mask_index < len(local_masks or ())
            ):
                sticker_contour = sticker_contours_by_mask[match.mask_index]
            candidate = solve_sticker_pnp(
                frame_points,
                point_class_ids,
                match,
                image_size,
                focal_scale,
                sticker_contour,
                scoring_points,
                sticker_contours,
                sticker_boundary_samples,
                sticker_mask_centers,
                sticker_mask_center_uncertainties,
            )
            if candidate is not None:
                candidates.append(candidate)
        candidates.sort(
            key=lambda candidate: (
                pose_hypothesis_visual_score(candidate),
                -candidate.sticker_average_confidence,
            )
        )
        return tuple(candidates)

    candidates = solve_matches(four_point_matches)
    if candidates:
        return candidates
    return solve_matches(
        tuple(match for match in valid_matches if len(match.keypoint_indices) == 3)
    )


def best_sticker_pnp_candidate(
    frame_points: np.ndarray,
    point_class_ids: np.ndarray,
    matches: tuple[StickerKeypointMatch, ...],
    image_size: tuple[int, int],
    focal_scale: float,
    local_masks: list[np.ndarray] | None = None,
    roi_xyxy: tuple[int, int, int, int] | None = None,
    scoring_points: np.ndarray | None = None,
    sticker_mask_erosion: int = 0,
) -> StickerPnPCandidate | None:
    """Compatibility wrapper returning the best retained candidate."""

    candidates = best_sticker_pnp_candidates(
        frame_points,
        point_class_ids,
        matches,
        image_size,
        focal_scale,
        local_masks,
        roi_xyxy,
        scoring_points,
        sticker_mask_erosion,
    )
    return candidates[0] if candidates else None


def pose_hypothesis_distance(
    candidate: StickerPnPCandidate,
    predicted_pose: tuple[np.ndarray, np.ndarray] | None,
) -> float:
    """Return a scale-normalized distance from a predicted pose."""

    if predicted_pose is None:
        return 0.0
    predicted_rvec, predicted_tvec = predicted_pose
    translation_scale = max(float(abs(np.asarray(candidate.tvec).reshape(3)[2])), 1.0)
    translation_distance = float(
        np.linalg.norm(
            np.asarray(candidate.tvec, dtype=np.float64).reshape(3)
            - np.asarray(predicted_tvec, dtype=np.float64).reshape(3)
        )
    ) / translation_scale
    rotation_distance = rotation_symmetry_distance(candidate.rvec, predicted_rvec)
    return translation_distance + rotation_distance


def pose_hypothesis_visual_score(candidate: StickerPnPCandidate) -> float:
    """Return normalized keypoint quality plus mask-center quality."""

    overall = pose_reprojection_fit_metric(candidate)
    if not np.isfinite(overall):
        overall = float(candidate.overall_reprojection_error)
    sticker = float(candidate.reprojection_error)
    if hasattr(candidate, "camera_matrix"):
        sticker_scale = projected_cube_diameter(candidate)
        if np.isfinite(sticker_scale) and sticker_scale > 0.0:
            sticker /= sticker_scale
    if not np.isfinite(overall):
        overall = 1e6
    if not np.isfinite(sticker):
        sticker = 1e6
    return (
        overall
        + 0.25 * sticker
        + PNP_MASK_CENTER_HYPOTHESIS_WEIGHT
        * sticker_mask_center_hypothesis_score(candidate)
    )


def sticker_mask_center_hypothesis_score(candidate: StickerPnPCandidate) -> float:
    """Score projected sticker centers for initial hypothesis ranking.

    This is deliberately a ranking-only cue.  The initial pose is still
    generated from the keypoint anchor; mask centers are matched afterward
    within the same 5-percent projected-cube gate.  Residuals are normalized
    by each mask's uncertainty, and the inverse-variance weighting prevents an
    irregular or merged mask from outweighing clean centers.
    """

    observed_centers = tuple(getattr(candidate, "sticker_mask_centers", ()))
    observed_count = len(observed_centers)
    if observed_count == 0:
        return 0.0
    diameter = projected_cube_diameter(candidate)
    if not np.isfinite(diameter) or diameter <= 0.0:
        return PNP_MASK_CENTER_UNMATCHED_PENALTY
    center_limit = PNP_FINAL_ASSIGNMENT_ERROR_FRACTION * diameter
    (
        _image_points,
        _object_points,
        distances,
        uncertainties,
    ) = matched_sticker_mask_center_correspondences(
        candidate,
        candidate.rvec,
        candidate.tvec,
        center_limit,
    )
    matched_count = len(distances)
    if matched_count == 0:
        return PNP_MASK_CENTER_UNMATCHED_PENALTY
    uncertainties = np.maximum(np.asarray(uncertainties, dtype=np.float64), 1.0)
    normalized_distances = np.asarray(distances, dtype=np.float64) / uncertainties
    weights = 1.0 / np.square(uncertainties)
    weighted_error = float(
        np.sum(weights * huber_loss(
            normalized_distances,
            delta=PNP_MASK_CENTER_HYPOTHESIS_HUBER_DELTA,
        ))
        / max(float(np.sum(weights)), 1e-9)
    )
    all_uncertainties = np.asarray(
        getattr(candidate, "sticker_mask_center_uncertainties", ()),
        dtype=np.float64,
    ).reshape(-1)
    if len(all_uncertainties) != observed_count:
        all_uncertainties = np.ones(observed_count, dtype=np.float64)
    all_uncertainties = np.maximum(all_uncertainties, 1.0)
    total_weight = float(np.sum(1.0 / np.square(all_uncertainties)))
    unmatched_weight = max(0.0, total_weight - float(np.sum(weights)))
    unmatched_fraction = unmatched_weight / max(total_weight, 1e-9)
    return weighted_error + PNP_MASK_CENTER_UNMATCHED_PENALTY * unmatched_fraction


def select_pose_hypothesis(
    candidates: tuple[StickerPnPCandidate, ...],
    predicted_pose: tuple[np.ndarray, np.ndarray] | None,
    previous_candidates: tuple[StickerPnPCandidate, ...] = (),
) -> StickerPnPCandidate | None:
    """Select one current measurement without averaging it with prediction.

    Reprojection quality is primary. Prediction and the recent hypothesis
    history only break close quality ties; they cannot force a good current
    measurement to move toward an old pose. This keeps one-frame turns
    observable while still making equivalent candidates temporally stable.
    """

    if not candidates:
        return None
    scored = []
    for candidate in candidates:
        visual_score = pose_hypothesis_visual_score(candidate)
        prediction_distance = pose_hypothesis_distance(candidate, predicted_pose)
        history_distance = (
            min(
                pose_hypothesis_distance(candidate, (prior.rvec, prior.tvec))
                for prior in previous_candidates
            )
            if previous_candidates
            else 0.0
        )
        scored.append((candidate, visual_score, prediction_distance, history_distance))
    accepted = [
        item for item in scored
        if pose_reprojection_error_is_acceptable(item[0])
    ]
    if not accepted:
        return None
    scored = accepted
    best_visual = min(item[1] for item in scored)
    # Only candidates within a small geometric-quality band are allowed to
    # use temporal continuity as a tie-breaker. The best visual candidate is
    # otherwise kept, which is important when a turn is genuinely abrupt.
    quality_band = max(1.0, 0.15 * best_visual)
    eligible = [item for item in scored if item[1] <= best_visual + quality_band]
    return min(
        eligible,
        key=lambda item: (item[2] + 0.5 * item[3], item[1]),
    )[0]


def predict_pose_result(
    model: object,
    model_roi: np.ndarray,
    args: argparse.Namespace,
) -> object | None:
    predict_kwargs = {} if args.device is None else {"device": args.device}
    pose_inference_conf = (
        args.pose_conf
        if args.pose_conf is not None
        else min(args.pose_class_conf.values())
    )
    results = model.predict(
        source=model_roi,
        conf=pose_inference_conf,
        imgsz=args.imgsz,
        max_det=args.max_det,
        verbose=False,
        **predict_kwargs,
    )
    return results[0] if results else None


def is_cpu_device(device: object) -> bool:
    """Return whether Ultralytics should run inference on the CPU."""

    device_text = "" if device is None else str(device).lower()
    return device is None or device_text == "cpu" or device_text.startswith("cpu:")


def predict_models_parallel_cpu(
    sticker_seg_model: object | None,
    pose_model: object,
    model_roi: np.ndarray,
    args: argparse.Namespace,
) -> tuple[tuple[list[np.ndarray], object, np.ndarray, np.ndarray | None] | None, object | None]:
    """Run the independent CPU model calls concurrently."""

    sticker_future = (
        CPU_NEURAL_EXECUTOR.submit(predict_sticker_masks, sticker_seg_model, model_roi, args)
        if sticker_seg_model is not None and args.run_sticker_segmentation
        else None
    )
    pose_future = CPU_NEURAL_EXECUTOR.submit(predict_pose_result, pose_model, model_roi, args)
    sticker_masks = sticker_future.result() if sticker_future is not None else None
    return sticker_masks, pose_future.result()


def cuda_stream_device(torch_module: object, device: object) -> object:
    device_text = str(device)
    if device_text.startswith("cuda"):
        return torch_module.device(device_text)
    if device_text.isdigit():
        return torch_module.device(f"cuda:{device_text}")
    return torch_module.device(f"cuda:{torch_module.cuda.current_device()}")


def predict_models_parallel_cuda(
    sticker_seg_model: object | None,
    pose_model: object,
    model_roi: np.ndarray,
    args: argparse.Namespace,
) -> tuple[tuple[list[np.ndarray], object, np.ndarray, np.ndarray | None] | None, object | None]:
    """Launch segmentation and pose on separate CUDA streams, then synchronize."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("--parallel-cuda requires a CUDA-enabled PyTorch installation")

    stream_device = cuda_stream_device(torch, args.device)
    segmentation_stream = torch.cuda.Stream(device=stream_device)
    pose_stream = torch.cuda.Stream(device=stream_device)
    with torch.cuda.device(stream_device):
        with torch.cuda.stream(segmentation_stream):
            sticker_result = (
                predict_sticker_result(sticker_seg_model, model_roi, args)
                if sticker_seg_model is not None and args.run_sticker_segmentation
                else None
            )
        with torch.cuda.stream(pose_stream):
            pose_result = predict_pose_result(pose_model, model_roi, args)
        torch.cuda.synchronize(stream_device)

    return sticker_masks_from_result(sticker_result), pose_result


def render_sticker_masks(
    image: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
    predicted: tuple[list[np.ndarray], object, np.ndarray, np.ndarray | None] | tuple[
        list[np.ndarray], object, np.ndarray
    ] | None,
    opacity: float,
    border_thickness: int,
    pose_predictions: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    point_threshold: float = 0.0,
    class_confidences: dict[str, float] | None = None,
    show_class_labels: bool = False,
    sticker_matches: tuple[StickerKeypointMatch, ...] | None = None,
    sticker_mask_erosion: int = 0,
) -> np.ndarray:
    output = image.copy()
    if predicted is None:
        return output
    if len(predicted) == 4:
        local_masks, names, class_ids, _cube_mask = predicted
    else:
        local_masks, names, class_ids = predicted
    x1, y1, x2, y2 = roi_xyxy
    roi_height, roi_width = y2 - y1, x2 - x1
    roi = output[y1:y2, x1:x2]
    local_points = point_confidences = point_class_ids = box_confidences = None
    if pose_predictions is not None:
        (
            local_points,
            point_confidences,
            point_class_ids,
            box_confidences,
        ) = pose_predictions
        class_confidences = class_confidences or dict(
            DEFAULT_INDEPENDENT_POSE_CLASS_CONFIDENCES
        )
        if sticker_matches is None:
            sticker_matches = sticker_keypoint_matches_for_masks(
                local_masks,
                local_points,
                point_confidences,
                point_class_ids,
                box_confidences,
                point_threshold,
                class_confidences,
            )

    for mask_index, (local_mask, _class_id) in enumerate(zip(local_masks, class_ids, strict=True)):
        sticker_type = "unknown"
        ordered_indices: list[int] = []
        if sticker_matches is not None and mask_index < len(sticker_matches):
            sticker_match = sticker_matches[mask_index]
            sticker_type = sticker_match.sticker_type
            ordered_indices = list(sticker_match.keypoint_indices)
        mask = cv2.resize(
            local_mask.astype(np.uint8),
            (roi_width, roi_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        mask = erode_binary_mask(mask, sticker_mask_erosion)
        # Keep the mask visualization class-independent.  Boundary color is
        # assigned later from distance to the projected cube wireframe.
        color = STICKER_MASK_NEUTRAL_COLOR_BGR
        color_layer = np.full_like(roi, color)
        blended = cv2.addWeighted(roi, 1.0 - opacity, color_layer, opacity, 0.0)
        roi[mask] = blended[mask]
        if border_thickness > 0:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(
                roi,
                contours,
                -1,
                STICKER_MASK_NEUTRAL_COLOR_BGR,
                border_thickness,
                cv2.LINE_AA,
            )
        if pose_predictions is not None and show_class_labels and ordered_indices:
            ys, xs = np.nonzero(mask)
            if len(xs):
                label = f"{sticker_type} ({len(ordered_indices)})"
                draw_label(
                    output,
                    label,
                    (int(round(xs.mean() + x1)), int(round(ys.mean() + y1))),
                    STICKER_MASK_NEUTRAL_COLOR_BGR,
                )
    return output


def draw_sticker_boundaries(
    image: np.ndarray,
    candidate: StickerPnPCandidate | None,
    *,
    thickness: int = 2,
) -> np.ndarray:
    """Draw the original segmentation contours retained by the pose fit."""

    output = image.copy()
    if candidate is None:
        return output
    line_thickness = max(1, int(thickness))
    boundaries = candidate.sticker_contours or (
        (candidate.sticker_contour,)
        if candidate.sticker_contour is not None
        else ()
    )
    for boundary in boundaries:
        points = np.asarray(boundary, dtype=np.float64).reshape(-1, 2)
        if len(points) < 3 or not np.isfinite(points).all():
            continue
        polygon = np.round(points).astype(np.int32).reshape(-1, 1, 2)
        # Keep the actual boundary visible over both the raw mask tint and the
        # underlying video, including along bright sticker boundaries.
        cv2.polylines(
            output,
            [polygon],
            True,
            (0, 0, 0),
            line_thickness + 2,
            cv2.LINE_AA,
        )
        cv2.polylines(
            output,
            [polygon],
            True,
            PNP_STICKER_COLOR_BGR,
            line_thickness,
            cv2.LINE_AA,
        )
    return output


def local_to_frame_points(
    local_points: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
    model_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = roi_xyxy
    scale = np.asarray(
        ((x2 - x1) / float(model_size), (y2 - y1) / float(model_size)),
        dtype=np.float32,
    )
    frame_points = local_points * scale + np.asarray((x1, y1), dtype=np.float32)
    return frame_points, local_points * scale


def tracking_bounds(
    local_roi_points: np.ndarray,
    point_confidences: np.ndarray,
    class_ids: np.ndarray,
    box_confidences: np.ndarray,
    class_confidences: dict[str, float],
    point_threshold: float,
    roi_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    height, width = roi_shape
    usable = [
        (float(point[0]), float(point[1]))
        for point, point_confidence, class_id, box_confidence in zip(
            local_roi_points,
            point_confidences,
            class_ids,
            box_confidences,
            strict=True,
        )
        if 0 <= int(class_id) < len(CLASS_NAMES)
        and float(box_confidence)
        >= independent_pose_class_threshold(CLASS_NAMES[int(class_id)], class_confidences)
        and float(point_confidence) >= point_threshold
        and np.isfinite(point).all()
        and 0 <= float(point[0]) < width
        and 0 <= float(point[1]) < height
    ]
    if not usable:
        return None
    points = np.asarray(usable, dtype=np.float32)
    return (
        int(np.floor(points[:, 0].min())),
        int(np.floor(points[:, 1].min())),
        int(np.ceil(points[:, 0].max())),
        int(np.ceil(points[:, 1].max())),
    )


def draw_label(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    x, y = position
    cv2.putText(
        image,
        text,
        (x + 1, y + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_pose(
    image: np.ndarray,
    frame_points: np.ndarray,
    point_confidences: np.ndarray,
    class_ids: np.ndarray,
    box_confidences: np.ndarray,
    point_threshold: float,
    class_confidences: dict[str, float],
    show_labels: bool,
    show_confidences: bool,
    copy_image: bool = True,
) -> tuple[np.ndarray, int]:
    output = image.copy() if copy_image else image
    drawn = 0
    per_class_index = {class_id: 0 for class_id in range(len(CLASS_NAMES))}
    for point, point_confidence, class_id, box_confidence in zip(
        frame_points,
        point_confidences,
        class_ids,
        box_confidences,
        strict=True,
    ):
        x, y = np.round(point).astype(np.int32)
        if not (0 <= x < output.shape[1] and 0 <= y < output.shape[0]):
            continue
        if not 0 <= int(class_id) < len(CLASS_NAMES):
            continue
        class_name = CLASS_NAMES[int(class_id)]
        class_threshold = independent_pose_class_threshold(class_name, class_confidences)
        if float(box_confidence) < class_threshold or float(point_confidence) < point_threshold:
            continue
        class_index = per_class_index[int(class_id)]
        per_class_index[int(class_id)] += 1
        color = CLASS_COLORS_BGR[int(class_id)]
        radius = max(1, round(min(output.shape[:2]) / 128.0 * 0.8))
        cv2.circle(output, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
        if show_labels:
            label = f"{class_name}:{class_index}"
            if show_confidences:
                label += f" {float(point_confidence):.2f}/{float(box_confidence):.2f}"
            draw_label(output, label, (int(x) + radius + 3, int(y) - 5), color)
        drawn += 1
    return output, drawn


def cubelet_center_object_point(
    x_index: int,
    y_index: int,
    z_index: int,
) -> np.ndarray:
    """Return the center of one of the cube's 27 cubies."""

    return np.asarray(
        (
            (x_index - 1) * CUBE_LAYER_SIZE_MM,
            (y_index - 1) * CUBE_LAYER_SIZE_MM,
            (z_index - 1) * CUBE_LAYER_SIZE_MM,
        ),
        dtype=np.float32,
    )


def cubelet_face_object_points(
    center: np.ndarray,
    axis: int,
    side: int,
) -> np.ndarray:
    """Return the four corners of one outer face of a cubie."""

    face_axes = {
        0: (1, 2),
        1: (0, 2),
        2: (0, 1),
    }
    u_axis, v_axis = face_axes[axis]
    half_layer = CUBE_LAYER_SIZE_MM / 2.0
    points = []
    for v_sign, u_sign in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
        point = np.asarray(center, dtype=np.float64).copy()
        point[axis] += side * half_layer
        point[u_axis] += u_sign * half_layer
        point[v_axis] += v_sign * half_layer
        points.append(point)
    points = np.asarray(points, dtype=np.float32)
    expected_normal = np.zeros(3, dtype=np.float64)
    expected_normal[axis] = side
    normal = np.cross(points[1] - points[0], points[2] - points[0])
    if float(np.dot(normal, expected_normal)) < 0.0:
        points = points[[0, 3, 2, 1]]
    return points


def apply_cubelet_layer_rotations(
    center: np.ndarray,
    face_points: np.ndarray,
    layer_rotations: tuple[CubeLayerRotation, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply layer turns to one cubie center and one of its outer faces."""

    original_center = np.asarray(center, dtype=np.float64).reshape(3)
    transformed_center = original_center.copy()
    transformed_face = np.asarray(face_points, dtype=np.float64).reshape(-1, 3).copy()
    for layer_rotation in layer_rotations:
        axis = int(layer_rotation.axis)
        side = int(layer_rotation.side)
        if axis not in (0, 1, 2) or side not in (-1, 1):
            continue
        layer_count = int(np.clip(layer_rotation.layer_count, 1, 3))
        layer_boundary = (
            CUBE_HALF_SIZE_MM
            - layer_count * CUBE_LAYER_SIZE_MM
            + CUBE_LAYER_SIZE_MM / 2.0
        )
        if side * original_center[axis] < layer_boundary - 1e-4:
            continue
        rotation_vector = np.zeros((3, 1), dtype=np.float64)
        rotation_vector[axis, 0] = (
            side * np.deg2rad(float(layer_rotation.angle_degrees))
        )
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pivot = np.zeros(3, dtype=np.float64)
        pivot[axis] = side * CUBE_HALF_SIZE_MM
        transformed_center = (
            rotation_matrix @ (transformed_center - pivot)
        ) + pivot
        transformed_face = (
            (rotation_matrix @ (transformed_face - pivot).T).T + pivot
        )
    return (
        transformed_center.astype(np.float32),
        transformed_face.astype(np.float32),
    )


def projected_sticker_faces(
    candidate: StickerPnPCandidate,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> list[tuple[StickerIdentity, float, np.ndarray]]:
    """Project visible stickers, retaining their persistent physical identities."""

    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    identities = {
        (identity.cubie_index, identity.axis, identity.side): identity
        for identity in canonical_sticker_identities()
    }
    visible_faces: list[tuple[StickerIdentity, float, np.ndarray]] = []
    for x_index in range(3):
        for y_index in range(3):
            for z_index in range(3):
                cubie_index = (x_index, y_index, z_index)
                canonical_center = cubelet_center_object_point(x_index, y_index, z_index)
                for axis, side in (
                    (axis, side)
                    for axis in range(3)
                    for side in (-1, 1)
                    if (
                        (axis == 0 and ((side == -1 and x_index == 0) or (side == 1 and x_index == 2)))
                        or (axis == 1 and ((side == -1 and y_index == 0) or (side == 1 and y_index == 2)))
                        or (axis == 2 and ((side == -1 and z_index == 0) or (side == 1 and z_index == 2)))
                    )
                ):
                    identity = identities[(cubie_index, axis, side)]
                    face_points = cubelet_face_object_points(canonical_center, axis, side)
                    transformed_center, transformed_face = apply_cubelet_layer_rotations(
                        canonical_center,
                        face_points,
                        candidate.layer_rotations,
                    )
                    normal = np.cross(
                        transformed_face[1] - transformed_face[0],
                        transformed_face[2] - transformed_face[0],
                    )
                    center_camera = (
                        rotation @ transformed_center.astype(np.float64)
                        + np.asarray(tvec, dtype=np.float64).reshape(3)
                    )
                    normal_camera = rotation @ normal
                    if float(np.dot(normal_camera, -center_camera)) <= 0.0:
                        continue
                    visible_faces.append(
                        (identity, float(center_camera[2]), transformed_face)
                    )

    if not visible_faces:
        return []
    try:
        all_face_points = np.concatenate(
            [face for _identity, _depth, face in visible_faces],
            axis=0,
        )
        projected_points, _ = cv2.projectPoints(
            all_face_points,
            rvec,
            tvec,
            candidate.camera_matrix,
            candidate.dist_coeffs,
        )
    except cv2.error:
        return []
    projected_points = projected_points.reshape(-1, 4, 2)
    projected = [
        (identity, depth, polygon)
        for (identity, depth, _face), polygon in zip(
            visible_faces,
            projected_points,
            strict=True,
        )
        if np.isfinite(polygon).all()
    ]
    projected.sort(key=lambda item: item[1], reverse=True)
    return projected


def sticker_segmentation_intersection_mask(
    predicted: tuple[list[np.ndarray], object, np.ndarray, np.ndarray | None],
    roi_shape: tuple[int, int],
    sticker_mask_erosion: int = 0,
) -> np.ndarray | None:
    """Return the sticker/cube intersection at the current ROI resolution."""

    local_masks, _names, _class_ids, cube_mask = predicted
    roi_height, roi_width = roi_shape
    sticker_union = np.zeros((roi_height, roi_width), dtype=bool)
    for local_mask in local_masks:
        resized = cv2.resize(
            np.asarray(local_mask, dtype=np.uint8),
            (roi_width, roi_height),
            interpolation=cv2.INTER_NEAREST,
        )
        sticker_union |= erode_binary_mask(resized > 0, sticker_mask_erosion)
    if not bool(sticker_union.any()):
        return None
    if cube_mask is None:
        return sticker_union
    resized_cube = cv2.resize(
        np.asarray(cube_mask, dtype=np.uint8),
        (roi_width, roi_height),
        interpolation=cv2.INTER_NEAREST,
    )
    return sticker_union & (resized_cube > 0)


def sticker_segmentation_masks_in_roi(
    predicted: tuple[list[np.ndarray], object, np.ndarray, np.ndarray | None],
    roi_shape: tuple[int, int],
    sticker_mask_erosion: int = 0,
) -> list[np.ndarray]:
    """Resize each sticker instance and intersect it with the cube mask."""

    if len(predicted) == 4:
        local_masks, _names, _class_ids, cube_mask = predicted
    else:
        local_masks, _names, _class_ids = predicted
        cube_mask = None
    roi_height, roi_width = roi_shape
    resized_cube = None
    if cube_mask is not None:
        resized_cube = cv2.resize(
            np.asarray(cube_mask, dtype=np.uint8),
            (roi_width, roi_height),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
    masks: list[np.ndarray] = []
    for local_mask in local_masks:
        resized = cv2.resize(
            np.asarray(local_mask, dtype=np.uint8),
            (roi_width, roi_height),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
        resized = erode_binary_mask(resized, sticker_mask_erosion)
        if resized_cube is not None:
            resized &= resized_cube
        if bool(resized.any()):
            masks.append(resized)
    return masks


def sticker_view_reliability_weight(
    candidate: StickerPnPCandidate,
    identity: StickerIdentity,
    rotation: np.ndarray | None = None,
) -> float:
    """Return a lower evidence weight for nearly edge-on sticker views.

    The reliability is the absolute cosine between the sticker normal and
    the camera-facing direction.  It therefore tends to zero as the viewing
    angle approaches 90 degrees, where projected sticker locations are most
    sensitive to pose error.
    """

    if rotation is None:
        rotation, _ = cv2.Rodrigues(
            np.asarray(candidate.rvec, dtype=np.float64).reshape(3, 1)
        )
    center = cubelet_center_object_point(*identity.cubie_index)
    face = cubelet_face_object_points(center, identity.axis, identity.side)
    transformed_center, transformed_face = apply_cubelet_layer_rotations(
        center,
        face,
        candidate.layer_rotations,
    )
    normal = np.cross(
        transformed_face[1] - transformed_face[0],
        transformed_face[2] - transformed_face[0],
    )
    center_camera = (
        rotation @ np.asarray(transformed_center, dtype=np.float64)
        + np.asarray(candidate.tvec, dtype=np.float64).reshape(3)
    )
    normal_camera = rotation @ normal
    denominator = float(np.linalg.norm(normal_camera) * np.linalg.norm(center_camera))
    if denominator <= 1e-12:
        return STICKER_VIEW_WEIGHT_FLOOR
    normal_alignment = abs(float(np.dot(normal_camera, center_camera))) / denominator
    return float(max(STICKER_VIEW_WEIGHT_FLOOR, normal_alignment))


def sample_projected_sticker_colors(
    frame_bgr: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
    predicted: tuple[list[np.ndarray], object, np.ndarray, np.ndarray | None] | None,
    candidate: StickerPnPCandidate | None,
    *,
    max_samples_per_sticker: int = 128,
    min_samples_per_sticker: int = 4,
    sticker_mask_erosion: int = 0,
) -> dict[str, np.ndarray]:
    """Average reliable pixels for each projected sticker identity in one frame."""

    if predicted is None or candidate is None:
        return {}
    x1, y1, x2, y2 = roi_xyxy
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return {}
    segmentation_masks = sticker_segmentation_masks_in_roi(
        predicted,
        roi.shape[:2],
        sticker_mask_erosion,
    )
    if not segmentation_masks:
        return {}
    segmentation_mask = np.logical_or.reduce(segmentation_masks)
    reliable_mask = _reliable_color_mask(roi)
    observations: dict[str, np.ndarray] = {}
    projected_faces = projected_sticker_faces(
        candidate,
        candidate.rvec,
        candidate.tvec,
    )
    # The face list is far-to-near.  Assigning each pixel in that order makes
    # the nearer projected face own any overlap, so a projected polygon cannot
    # borrow visible pixels from another sticker behind it.
    projected_face_owners = np.full(roi.shape[:2], -1, dtype=np.int32)
    for face_index, (_identity, _depth, polygon) in enumerate(projected_faces):
        polygon_roi = np.round(polygon - np.asarray((x1, y1), dtype=np.float32)).astype(np.int32)
        projected_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(projected_mask, polygon_roi, 1)
        projected_face_owners[projected_mask > 0] = face_index

    # A combined segmentation model normally returns one instance per visible
    # sticker. Match those instances to projected faces before sampling so a
    # small pose error cannot mix a neighboring sticker's color into this one.
    # Keep the union fallback for a model that returns one mask covering all
    # stickers rather than separate instances.
    face_to_segmentation: dict[int, np.ndarray] = {}
    if len(segmentation_masks) > 1:
        pairings = []
        for face_index, _face in enumerate(projected_faces):
            face_mask = projected_face_owners == face_index
            face_area = int(face_mask.sum())
            if face_area == 0:
                continue
            for mask_index, segmentation_instance in enumerate(segmentation_masks):
                overlap = int(np.count_nonzero(face_mask & segmentation_instance))
                if overlap < max(1, int(min_samples_per_sticker)):
                    continue
                union = face_area + int(segmentation_instance.sum()) - overlap
                pairings.append(
                    (
                        overlap / max(1, union),
                        overlap,
                        face_index,
                        mask_index,
                    )
                )
        used_faces: set[int] = set()
        used_masks: set[int] = set()
        for _iou, _overlap, face_index, mask_index in sorted(
            pairings,
            key=lambda pairing: (-pairing[0], -pairing[1], pairing[2], pairing[3]),
        ):
            if face_index in used_faces or mask_index in used_masks:
                continue
            used_faces.add(face_index)
            used_masks.add(mask_index)
            face_to_segmentation[face_index] = segmentation_masks[mask_index]

    for face_index, (identity, _depth, _polygon) in enumerate(projected_faces):
        if len(segmentation_masks) > 1:
            sticker_mask = face_to_segmentation.get(face_index)
            if sticker_mask is None:
                continue
        else:
            sticker_mask = segmentation_mask
        sample_mask = (projected_face_owners == face_index) & sticker_mask & reliable_mask
        ys, xs = np.nonzero(sample_mask)
        if len(xs) < max(1, int(min_samples_per_sticker)):
            continue
        if len(xs) > max(1, int(max_samples_per_sticker)):
            selected = np.linspace(
                0,
                len(xs) - 1,
                max(1, int(max_samples_per_sticker)),
                dtype=np.int32,
            )
            ys = ys[selected]
            xs = xs[selected]
        mean_bgr = roi[ys, xs].astype(np.float32).mean(axis=0)
        observations[identity.label] = np.round(mean_bgr[::-1]).clip(0, 255).astype(np.uint8)
    return observations


def save_sticker_rgb_html_plot(
    trace: StickerColorTrace,
    output_path: Path,
    *,
    title: str,
    normalize_rgb: bool = False,
    point_size: float = 5.0,
) -> None:
    """Save one interactive RGB trajectory plot with one trace per sticker."""

    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError(
            "Plotly is required for --rgb-html-plot. Install it in the Python environment."
        ) from exc

    figure = go.Figure()
    category_titles = {"center": "Centers", "edge": "Edges", "corner": "Corners"}
    category_seen: set[str] = set()
    raw_trace_indexes: list[int] = []
    regression_trace_indexes: list[int] = []
    global_raw_trace_indexes: dict[str, int] = {}
    global_profile_trace_indexes: dict[str, int] = {}
    profiles = {
        identity.label: fit_sticker_color_profile(trace, identity.label)
        for identity in trace.identities
    }
    global_profiles, global_observations = build_global_color_profiles(trace)
    for identity in trace.identities:
        entries = trace.observations.get(identity.label, ())
        frames = np.asarray([entry[0] for entry in entries], dtype=np.int32)
        rgb = np.asarray([entry[1] for entry in entries], dtype=np.uint8).reshape(-1, 3)
        plot_rgb = rgb.astype(np.float32)
        if normalize_rgb:
            plot_rgb /= np.maximum(np.linalg.norm(plot_rgb, axis=1, keepdims=True), 1.0)
        customdata = np.column_stack((frames, rgb))
        raw_trace_indexes.append(len(figure.data))
        figure.add_trace(
            go.Scatter3d(
                x=plot_rgb[:, 0],
                y=plot_rgb[:, 1],
                z=plot_rgb[:, 2],
                mode="markers",
                marker={
                    "size": point_size,
                    "color": [
                        f"rgb({int(red)},{int(green)},{int(blue)})"
                        for red, green, blue in rgb
                    ],
                    "opacity": 0.82,
                },
                customdata=customdata,
                hovertemplate=(
                    f"{identity.label}<br>"
                    "frame=%{customdata[0]}<br>"
                    "R=%{customdata[1]} G=%{customdata[2]} B=%{customdata[3]}<br>"
                    "x=%{x:.3f} y=%{y:.3f} z=%{z:.3f}<extra></extra>"
                ),
                name=identity.label,
                legendgroup=identity.category,
                legendgrouptitle=(
                    {"text": category_titles[identity.category]}
                    if identity.category not in category_seen
                    else None
                ),
            )
        )
        profile = profiles[identity.label]
        if profile is None:
            regression_rgb = np.empty((0, 3), dtype=np.float32)
        else:
            regression_rgb = sample_profile_polyline(profile.polyline_rgb, count=96).astype(
                np.float32
            )
            if normalize_rgb:
                regression_rgb /= np.maximum(
                    np.linalg.norm(regression_rgb, axis=1, keepdims=True),
                    1.0,
                )
        regression_trace_indexes.append(len(figure.data))
        figure.add_trace(
            go.Scatter3d(
                x=regression_rgb[:, 0],
                y=regression_rgb[:, 1],
                z=regression_rgb[:, 2],
                mode="lines",
                line={"color": "rgba(0,0,0,0.9)", "width": 5},
                hovertemplate=(
                    f"{identity.label} fitted regression<br>"
                    "R=%{x:.3f} G=%{y:.3f} B=%{z:.3f}<extra></extra>"
                ),
                name=f"{identity.label} regression",
                legendgroup=identity.category,
                showlegend=False,
            )
        )
        category_seen.add(identity.category)

    def global_plot_color(color_name: str) -> str:
        red, green, blue = PURE_COLOR_RGB[color_name]
        # A pure white line disappears on Plotly's white background.
        if color_name == "white":
            return "rgb(80,80,80)"
        return f"rgb({red},{green},{blue})"

    for color_name in GLOBAL_COLOR_NAMES:
        entries = global_observations[color_name]
        if entries:
            rgb = np.asarray([entry[2] for entry in entries], dtype=np.uint8).reshape(-1, 3)
            plot_rgb = rgb.astype(np.float32)
            if normalize_rgb:
                plot_rgb /= np.maximum(np.linalg.norm(plot_rgb, axis=1, keepdims=True), 1.0)
            customdata = [
                [entry[0], entry[1], int(red), int(green), int(blue)]
                for entry, (red, green, blue) in zip(entries, rgb, strict=True)
            ]
        else:
            rgb = np.empty((0, 3), dtype=np.uint8)
            plot_rgb = np.empty((0, 3), dtype=np.float32)
            customdata = []
        global_raw_trace_indexes[color_name] = len(figure.data)
        figure.add_trace(
            go.Scatter3d(
                x=plot_rgb[:, 0],
                y=plot_rgb[:, 1],
                z=plot_rgb[:, 2],
                mode="markers",
                marker={
                    "size": point_size,
                    "color": [
                        f"rgb({int(red)},{int(green)},{int(blue)})"
                        for red, green, blue in rgb
                    ],
                    "opacity": 0.88,
                    "line": {"color": "black", "width": 1},
                },
                customdata=customdata,
                hovertemplate=(
                    f"{color_name} global group<br>"
                    "sticker=%{customdata[0]}<br>"
                    "frame=%{customdata[1]}<br>"
                    "R=%{customdata[2]} G=%{customdata[3]} B=%{customdata[4]}"
                    "<extra></extra>"
                ),
                name=f"{color_name} observations",
                legendgroup="global-colors",
            )
        )
        fitted = global_profiles[color_name].fitted
        if fitted is None:
            profile_rgb = np.empty((0, 3), dtype=np.float32)
        else:
            profile_rgb = sample_profile_polyline(fitted.polyline_rgb, count=96).astype(
                np.float32
            )
            if normalize_rgb:
                profile_rgb /= np.maximum(
                    np.linalg.norm(profile_rgb, axis=1, keepdims=True),
                    1.0,
                )
        global_profile_trace_indexes[color_name] = len(figure.data)
        figure.add_trace(
            go.Scatter3d(
                x=profile_rgb[:, 0],
                y=profile_rgb[:, 1],
                z=profile_rgb[:, 2],
                mode="lines",
                line={
                    "color": global_plot_color(color_name),
                    "width": 7,
                },
                name=f"{color_name} global profile",
                legendgroup="global-colors",
                showlegend=False,
                hovertemplate=(
                    f"{color_name} global profile<br>"
                    "R=%{x:.3f} G=%{y:.3f} B=%{z:.3f}<extra></extra>"
                ),
            )
        )

    sticker_count = len(trace.identities)
    plot_trace_count = len(figure.data)

    all_sticker_visibility = [False] * plot_trace_count
    for index in range(len(trace.identities)):
        all_sticker_visibility[raw_trace_indexes[index]] = True
        all_sticker_visibility[regression_trace_indexes[index]] = True

    def global_visibility(color_names: set[str]) -> list[bool]:
        visibility = [False] * plot_trace_count
        for color_name in color_names:
            visibility[global_raw_trace_indexes[color_name]] = True
            visibility[global_profile_trace_indexes[color_name]] = True
        return visibility

    def visibility_for(predicate) -> list[bool]:
        visibility = [False] * plot_trace_count
        for index, identity in enumerate(trace.identities):
            if predicate(identity):
                visibility[raw_trace_indexes[index]] = True
                visibility[regression_trace_indexes[index]] = True
        return visibility

    selector_buttons = [
        {
            "label": "All stickers",
            "method": "update",
            "args": [
                {"visible": all_sticker_visibility},
                {"title": f"{title} (all {sticker_count} stickers)"},
            ],
        },
        {
            "label": "All colors",
            "method": "update",
            "args": [
                {"visible": global_visibility(set(GLOBAL_COLOR_NAMES))},
                {"title": f"{title} — All colors"},
            ],
        },
        {
            "label": "Centers",
            "method": "update",
            "args": [
                {
                    "visible": visibility_for(
                        lambda identity: identity.category == "center"
                    )
                },
                {"title": f"{title} — Centers"},
            ],
        },
        {
            "label": "Edges",
            "method": "update",
            "args": [
                {
                    "visible": visibility_for(
                        lambda identity: identity.category == "edge"
                    )
                },
                {"title": f"{title} — Edges"},
            ],
        },
        {
            "label": "Corners",
            "method": "update",
            "args": [
                {
                    "visible": visibility_for(
                        lambda identity: identity.category == "corner"
                    )
                },
                {"title": f"{title} — Corners"},
            ],
        },
    ]
    for color_name in GLOBAL_COLOR_NAMES:
        selector_buttons.append(
            {
                "label": color_name,
                "method": "update",
                "args": [
                    {"visible": global_visibility({color_name})},
                    {"title": f"{title} — {color_name}"},
                ],
            }
        )
    for trace_index, identity in enumerate(trace.identities):
        sticker_visibility = [False] * plot_trace_count
        sticker_visibility[raw_trace_indexes[trace_index]] = True
        sticker_visibility[regression_trace_indexes[trace_index]] = True
        selector_buttons.append(
            {
                "label": identity.label,
                "method": "update",
                "args": [
                    {"visible": sticker_visibility},
                    {"title": f"{title} — {identity.label}"},
                ],
            }
        )

    axis_range = [0.0, 1.0] if normalize_rgb else [0.0, 255.0]
    axis_title = "normalized RGB" if normalize_rgb else "RGB"
    figure.update_layout(
        title=f"{title} ({sum(bool(trace.observations.get(identity.label)) for identity in trace.identities)} stickers)",
        scene={
            "xaxis": {"title": f"R ({axis_title})", "range": axis_range},
            "yaxis": {"title": f"G ({axis_title})", "range": axis_range},
            "zaxis": {"title": f"B ({axis_title})", "range": axis_range},
            "aspectmode": "cube",
        },
        updatemenus=[
            {
                "active": 0,
                "buttons": selector_buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.0,
                "xanchor": "left",
                "y": 1.16,
                "yanchor": "top",
            }
        ],
        clickmode="event+select",
        margin={"l": 0, "r": 0, "b": 0, "t": 86},
        legend={
            "itemsizing": "constant",
            "groupclick": "toggleitem",
            "itemclick": "toggle",
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(output_path), include_plotlyjs="cdn", full_html=True)


def normalized_rgb_html_plot_path(rgb_html_path: Path) -> Path:
    suffix = rgb_html_path.suffix or ".html"
    return rgb_html_path.with_name(f"{rgb_html_path.stem}_normalized{suffix}")


def raw_rgb_html_plot_path(rgb_html_path: Path) -> Path:
    suffix = rgb_html_path.suffix or ".html"
    return rgb_html_path.with_name(f"{rgb_html_path.stem}_raw{suffix}")


def save_sticker_color_plots(
    trace: StickerColorTrace,
    output_path: Path,
    *,
    title: str,
    primary_normalized: bool,
    point_size: float,
) -> None:
    save_sticker_rgb_html_plot(
        trace,
        output_path,
        title=title,
        normalize_rgb=primary_normalized,
        point_size=point_size,
    )
    companion_path = (
        raw_rgb_html_plot_path(output_path)
        if primary_normalized
        else normalized_rgb_html_plot_path(output_path)
    )
    save_sticker_rgb_html_plot(
        trace,
        companion_path,
        title=f"{title} {'raw' if primary_normalized else 'normalized'}",
        normalize_rgb=not primary_normalized,
        point_size=point_size,
    )


def projected_cubelet_faces(
    candidate: StickerPnPCandidate,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> list[tuple[float, np.ndarray, bool]]:
    """Project visible cubie faces with anchor and turn metadata."""

    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    projected_faces: list[tuple[float, np.ndarray, bool]] = []
    orientation_anchor = getattr(candidate, "orientation_anchor", None)
    if orientation_anchor is None:
        orientation_anchor = all_cube_orientation_anchor()
    for x_index in range(3):
        for y_index in range(3):
            for z_index in range(3):
                center = cubelet_center_object_point(x_index, y_index, z_index)
                for axis, side in (
                    (axis, side)
                    for axis in range(3)
                    for side in (-1, 1)
                    if (
                        (axis == 0 and ((side == -1 and x_index == 0) or (side == 1 and x_index == 2)))
                        or (axis == 1 and ((side == -1 and y_index == 0) or (side == 1 and y_index == 2)))
                        or (axis == 2 and ((side == -1 and z_index == 0) or (side == 1 and z_index == 2)))
                    )
                ):
                    face_points = cubelet_face_object_points(center, axis, side)
                    transformed_center, transformed_face = apply_cubelet_layer_rotations(
                        center,
                        face_points,
                        candidate.layer_rotations,
                    )
                    normal = np.cross(
                        transformed_face[1] - transformed_face[0],
                        transformed_face[2] - transformed_face[0],
                    )
                    center_camera = (
                        rotation @ transformed_center.astype(np.float64)
                        + np.asarray(tvec, dtype=np.float64).reshape(3)
                    )
                    normal_camera = rotation @ normal
                    if float(np.dot(normal_camera, -center_camera)) <= 0.0:
                        continue
                    projected_face, _ = cv2.projectPoints(
                        transformed_face,
                        rvec,
                        tvec,
                        candidate.camera_matrix,
                        candidate.dist_coeffs,
                    )
                    anchor = orientation_anchor.axis is None or (
                        _cubelet_layer_index(center, orientation_anchor.axis)
                        in orientation_anchor.layers
                    )
                    projected_faces.append(
                        (
                            float(center_camera[2]),
                            projected_face.reshape(-1, 2),
                            anchor,
                        )
                    )
    projected_faces.sort(key=lambda item: item[0], reverse=True)
    return projected_faces


def projected_cube_layer_color(
    candidate: StickerPnPCandidate,
    anchor: bool,
    fallback_color: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Return the turn overlay color for a projected cube face.

    For a single-layer turn the moving layer is red and the stationary
    support is black.  For a double-layer turn the visual roles are reversed:
    the larger moving slab is black and the remaining stationary layer is
    red.  Candidates without a turn keep their caller-supplied color.
    """

    rotations = tuple(
        rotation
        for rotation in candidate.layer_rotations
        if (
            int(rotation.axis) in (0, 1, 2)
            and int(rotation.side) in (-1, 1)
            and abs(float(rotation.angle_degrees)) > 1e-6
        )
    )
    if not rotations:
        return fallback_color
    double_layer = max(int(rotation.layer_count) for rotation in rotations) >= 2
    if double_layer:
        return PNP_TURN_STATIC_COLOR_BGR if not anchor else PNP_TURN_MOVING_COLOR_BGR
    return PNP_TURN_MOVING_COLOR_BGR if not anchor else PNP_TURN_STATIC_COLOR_BGR


def draw_projected_cube_pose(
    image: np.ndarray,
    candidate: StickerPnPCandidate,
    rvec: np.ndarray,
    tvec: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
    outline: bool = False,
    image_offset: tuple[int, int] = (0, 0),
) -> bool:
    """Draw a corrected 3x3 Rubik's-cube wireframe."""

    try:
        projected_faces = projected_cubelet_faces(candidate, rvec, tvec)
    except cv2.error:
        return False
    if not projected_faces:
        return False
    image_offset_array = np.asarray(image_offset, dtype=np.float32).reshape(1, 2)
    for _depth, projected_face, anchor in projected_faces:
        projected_face = np.round(projected_face + image_offset_array).astype(np.int32)
        face_color = projected_cube_layer_color(candidate, anchor, color)
        for index in range(4):
            start = tuple(int(value) for value in projected_face[index])
            end = tuple(int(value) for value in projected_face[(index + 1) % 4])
            if outline:
                cv2.line(image, start, end, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.line(image, start, end, face_color, thickness, cv2.LINE_AA)
    return True


def project_cube_orientation_axes(
    candidate: StickerPnPCandidate,
    rvec: np.ndarray,
    tvec: np.ndarray,
    axis_length: float,
) -> np.ndarray | None:
    """Project the cube-centered positive X/Y/Z axes into the image."""

    if not np.isfinite(axis_length) or float(axis_length) <= 0.0:
        return None
    object_points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (float(axis_length), 0.0, 0.0),
            (0.0, float(axis_length), 0.0),
            (0.0, 0.0, float(axis_length)),
        ),
        dtype=np.float32,
    )
    try:
        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            candidate.camera_matrix,
            candidate.dist_coeffs,
        )
    except cv2.error:
        return None
    projected = projected.reshape(-1, 2)
    return projected if np.isfinite(projected).all() else None


def draw_orientation_axes(
    image: np.ndarray,
    candidate: StickerPnPCandidate,
    rvec: np.ndarray,
    tvec: np.ndarray,
    axis_length: float = 24.0,
    image_offset: tuple[int, int] = (0, 0),
) -> bool:
    """Draw a cube-attached 3-axis orientation gizmo.

    The axes are projected from the same object frame as the pose wireframe,
    so the gizmo follows the symmetry-canonicalized pose continuously across
    a video rather than being a separate screen-space indicator.
    """

    projected = project_cube_orientation_axes(
        candidate,
        rvec,
        tvec,
        axis_length,
    )
    if projected is None:
        return False
    projected = projected + np.asarray(image_offset, dtype=np.float64).reshape(1, 2)
    origin = tuple(int(value) for value in np.round(projected[0]).astype(np.int32))
    for axis_index, (color, name) in enumerate(
        zip(ORIENTATION_AXIS_COLORS_BGR, ORIENTATION_AXIS_NAMES, strict=True)
    ):
        endpoint = tuple(
            int(value)
            for value in np.round(projected[axis_index + 1]).astype(np.int32)
        )
        cv2.arrowedLine(
            image,
            origin,
            endpoint,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
            tipLength=0.18,
        )
        cv2.arrowedLine(
            image,
            origin,
            endpoint,
            color,
            3,
            cv2.LINE_AA,
            tipLength=0.18,
        )
        draw_label(
            image,
            name,
            (endpoint[0] + 4, endpoint[1] - 4),
            color,
        )
    cv2.circle(image, origin, 5, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(image, origin, 3, (255, 255, 255), -1, cv2.LINE_AA)
    return True


def append_turn_candidate_log(
    path: Path,
    candidate: StickerPnPCandidate | None,
    frame_index: int | None,
) -> None:
    """Append one complete turn-search record for one rendered frame."""

    def json_safe(value: object) -> object:
        if isinstance(value, dict):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if isinstance(value, (float, np.floating)):
            numeric = float(value)
            return numeric if np.isfinite(numeric) else None
        if isinstance(value, (int, np.integer, str, bool)) or value is None:
            return value
        return str(value)

    scores = getattr(candidate, "turn_candidate_scores", ()) if candidate else ()
    diagnostics = (
        getattr(candidate, "turn_candidate_diagnostics", ()) if candidate else ()
    )
    if diagnostics:
        candidates = list(diagnostics)
    else:
        candidates = [
            {
                "label": label,
                "global_mean_error": float(mean_score),
                "inlier_count": int(inlier_count),
                "keypoint_p90_error": float(p90_error),
                "keypoint_mean_error": float(keypoint_mean_error),
            }
            for label, mean_score, inlier_count, p90_error, keypoint_mean_error in scores
        ]
    static_scores = (
        list(getattr(candidate, "turn_fit_static_scores", ()))
        if candidate is not None
        and getattr(candidate, "turn_fit_static_scores", None) is not None
        else None
    )
    articulated_scores = (
        list(getattr(candidate, "turn_fit_articulated_scores", ()))
        if candidate is not None
        and getattr(candidate, "turn_fit_articulated_scores", None) is not None
        else None
    )
    selected_rotations = (
        [
            {
                "axis": int(rotation.axis),
                "side": int(rotation.side),
                "angle_degrees": float(rotation.angle_degrees),
                "layer_count": int(rotation.layer_count),
            }
            for rotation in candidate.layer_rotations
        ]
        if candidate is not None
        else []
    )
    record = {
        "frame_index": frame_index,
        "has_pose": candidate is not None,
        "turn_fit_evaluated": bool(
            getattr(candidate, "turn_fit_evaluated", False)
            if candidate is not None
            else False
        ),
        "turn_fit_unexplained_keypoints": int(
            getattr(candidate, "turn_fit_unexplained_keypoints", 0)
            if candidate is not None
            else 0
        ),
        "turn_fit_sticker_score": float(
            getattr(candidate, "turn_fit_sticker_score", float("inf"))
            if candidate is not None
            else float("inf")
        ),
        "static_fit_scores": static_scores,
        "selected_fit_scores": articulated_scores,
        "selected": (
            layer_rotation_debug_label(candidate.layer_rotations)
            if candidate is not None
            else None
        ),
        "selected_layer_rotations": selected_rotations,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log_file:
        json.dump(json_safe(record), log_file, separators=(",", ":"))
        log_file.write("\n")


def draw_projected_sticker_identity_labels(
    image: np.ndarray,
    candidate: StickerPnPCandidate,
    rvec: np.ndarray,
    tvec: np.ndarray,
    image_offset: tuple[int, int] = (0, 0),
) -> int:
    """Draw the short identity number at the center of each visible sticker."""

    try:
        projected_faces = projected_sticker_faces(candidate, rvec, tvec)
    except cv2.error:
        return 0
    height, width = image.shape[:2]
    image_offset_array = np.asarray(image_offset, dtype=np.float32).reshape(1, 2)
    drawn = 0
    for identity, _depth, polygon in projected_faces:
        center = np.mean(polygon, axis=0) + image_offset_array[0]
        if not np.isfinite(center).all():
            continue
        x, y = np.round(center).astype(np.int32)
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        label = identity.label.split(" ", maxsplit=1)[0]
        text_size, baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            1,
        )
        origin = (
            int(x - text_size[0] / 2),
            int(y + (text_size[1] + baseline) / 2),
        )
        top_left = (origin[0] - 2, origin[1] - text_size[1] - 2)
        bottom_right = (origin[0] + text_size[0] + 2, origin[1] + baseline + 2)
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.putText(
            image,
            label,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        drawn += 1
    return drawn


def draw_projected_sticker_colors(
    image: np.ndarray,
    candidate: StickerPnPCandidate,
    sticker_color_names: dict[str, str],
    rvec: np.ndarray,
    tvec: np.ndarray,
    opacity: float = 0.72,
    image_offset: tuple[int, int] = (0, 0),
    projected_faces: list[tuple[StickerIdentity, float, np.ndarray]] | None = None,
) -> int:
    """Fill classified, visible projected stickers with their colors."""

    if not sticker_color_names:
        return 0
    opacity = float(np.clip(opacity, 0.0, 1.0))
    image_offset_array = np.asarray(image_offset, dtype=np.float32).reshape(1, 2)
    drawn = 0
    if projected_faces is None:
        try:
            projected_faces = projected_sticker_faces(candidate, rvec, tvec)
        except cv2.error:
            return 0
    for identity, _depth, polygon in projected_faces:
        color_name = sticker_color_names.get(identity.label)
        if color_name not in PURE_COLOR_RGB:
            continue
        polygon_int = np.round(polygon + image_offset_array).astype(np.int32)
        height, width = image.shape[:2]
        min_x = max(0, int(np.min(polygon_int[:, 0])))
        max_x = min(width, int(np.max(polygon_int[:, 0])) + 1)
        min_y = max(0, int(np.min(polygon_int[:, 1])))
        max_y = min(height, int(np.max(polygon_int[:, 1])) + 1)
        if min_x >= max_x or min_y >= max_y:
            continue
        local_polygon = polygon_int - np.asarray((min_x, min_y), dtype=np.int32)
        mask = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
        cv2.fillConvexPoly(mask, local_polygon, 1)
        color_bgr = tuple(int(value) for value in PURE_COLOR_RGB[color_name][::-1])
        region = image[min_y:max_y, min_x:max_x]
        selected = mask > 0
        if np.any(selected):
            pixels = region[selected].astype(np.float32)
            color = np.asarray(color_bgr, dtype=np.float32)
            region[selected] = np.clip(
                pixels * (1.0 - opacity) + color * opacity,
                0.0,
                255.0,
            ).astype(np.uint8)
        cv2.polylines(image, [polygon_int], True, (0, 0, 0), 2, cv2.LINE_AA)
        drawn += 1
    return drawn


def draw_projected_center_colors(
    image: np.ndarray,
    candidate: StickerPnPCandidate,
    center_color_names: dict[str, str],
    rvec: np.ndarray,
    tvec: np.ndarray,
    opacity: float = 0.72,
) -> int:
    """Fill classified, visible projected center stickers with their colors."""

    center_names = {
        label: color
        for label, color in center_color_names.items()
        if label.startswith("C")
    }
    return draw_projected_sticker_colors(
        image,
        candidate,
        center_names,
        rvec,
        tvec,
        opacity,
    )


def draw_projected_sticker_likelihoods(
    image: np.ndarray,
    candidate: StickerPnPCandidate,
    sticker_color_names: dict[str, str],
    sticker_likelihoods: dict[str, dict[str, float]],
    rvec: np.ndarray,
    tvec: np.ndarray,
    image_offset: tuple[int, int] = (0, 0),
    projected_faces: list[tuple[StickerIdentity, float, np.ndarray]] | None = None,
) -> int:
    """Draw compact confidence values and a swatch for the strongest alternative."""

    if not sticker_likelihoods:
        return 0
    if projected_faces is None:
        try:
            projected_faces = projected_sticker_faces(candidate, rvec, tvec)
        except cv2.error:
            return 0
    drawn = 0
    height, width = image.shape[:2]
    offset = np.asarray(image_offset, dtype=np.float32)
    for identity, _depth, polygon in projected_faces:
        probabilities = sticker_likelihoods.get(identity.label)
        assigned_color = sticker_color_names.get(identity.label)
        if probabilities is None or assigned_color is None:
            continue
        confidence = float(probabilities.get(assigned_color, 0.0))
        alternatives = [
            (color_name, float(probability))
            for color_name, probability in probabilities.items()
            if color_name != assigned_color
        ]
        if not alternatives:
            continue
        second_color, second_probability = max(
            alternatives,
            key=lambda item: item[1],
        )
        center = np.mean(polygon, axis=0) + offset
        if not np.isfinite(center).all():
            continue
        x, y = np.round(center).astype(np.int32)
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        text = f"{confidence:.2f}, {second_probability:.2f}"
        font_scale = 0.28
        thickness = 1
        text_size, baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )
        swatch_size = max(5, int(round(text_size[1] * 0.75)))
        swatch_gap = 3
        block_width = text_size[0] + swatch_gap + swatch_size
        block_height = max(text_size[1] + baseline, swatch_size)
        origin_x = int(x - block_width / 2)
        text_y = int(y - block_height / 2 + text_size[1])
        top_left = (origin_x - 2, int(y - block_height / 2) - 2)
        bottom_right = (origin_x + block_width + 2, int(y + block_height / 2) + 2)
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.putText(
            image,
            text,
            (origin_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        swatch_x = origin_x + text_size[0] + swatch_gap
        swatch_y = int(y - swatch_size / 2)
        swatch_color_bgr = tuple(
            int(value) for value in PURE_COLOR_RGB[second_color][::-1]
        )
        cv2.rectangle(
            image,
            (swatch_x, swatch_y),
            (swatch_x + swatch_size - 1, swatch_y + swatch_size - 1),
            swatch_color_bgr,
            -1,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            image,
            (swatch_x, swatch_y),
            (swatch_x + swatch_size - 1, swatch_y + swatch_size - 1),
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        drawn += 1
    return drawn


def draw_classified_cube_diagnostic_panel(
    image: np.ndarray,
    candidate: StickerPnPCandidate | None,
    sticker_color_names: dict[str, str],
    sticker_likelihoods: dict[str, dict[str, float]] | None = None,
    rvec: np.ndarray | None = None,
    orientation_axis_length: float = 24.0,
    show_sticker_identities: bool = True,
) -> np.ndarray:
    """Append a classified projected cube beside the original diagnostic image."""

    height, width = image.shape[:2]
    panel = np.full_like(image, 32)
    output = np.concatenate((image, panel), axis=1)
    image_offset = (width, 0)
    cv2.line(
        output,
        (width, 0),
        (width, height - 1),
        (120, 120, 120),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        "classified cube | color confidence",
        (width + 10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    if candidate is None:
        return output
    display_rvec = candidate.rvec if rvec is None else rvec
    # Keep the diagnostic cube stationary in the side panel.  Only the
    # current orientation is transferred from the tracked cube.
    display_tvec = np.asarray(
        [[0.0], [0.0], [CLASSIFIED_CUBE_DIAGNOSTIC_DEPTH_MM]],
        dtype=np.float64,
    )
    try:
        projected_faces = projected_sticker_faces(candidate, display_rvec, display_tvec)
    except cv2.error:
        projected_faces = []
    draw_projected_sticker_colors(
        output,
        candidate,
        sticker_color_names,
        display_rvec,
        display_tvec,
        image_offset=image_offset,
        projected_faces=projected_faces,
    )
    if sticker_likelihoods is not None:
        draw_projected_sticker_likelihoods(
            output,
            candidate,
            sticker_color_names,
            sticker_likelihoods,
            display_rvec,
            display_tvec,
            image_offset=image_offset,
            projected_faces=projected_faces,
        )
    draw_projected_cube_pose(
        output,
        candidate,
        display_rvec,
        display_tvec,
        (0, 0, 0),
        2,
        outline=False,
        image_offset=image_offset,
    )
    if show_sticker_identities:
        draw_projected_sticker_identity_labels(
            output,
            candidate,
            display_rvec,
            display_tvec,
            image_offset=image_offset,
        )
    draw_orientation_axes(
        output,
        candidate,
        display_rvec,
        display_tvec,
        orientation_axis_length,
        image_offset=image_offset,
    )
    return output


def save_turn_candidate_images(
    base_image: np.ndarray,
    candidate: StickerPnPCandidate,
    output_dir: Path,
    frame_index: int | None,
) -> None:
    """Save annotated images for the ranked layer-turn hypotheses."""

    debug_candidates = getattr(candidate, "turn_candidate_debug", ())
    if not debug_candidates:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_label = "unknown" if frame_index is None else f"{frame_index:06d}"
    for rank, debug_candidate in enumerate(debug_candidates):
        rendered_candidate = replace(
            candidate,
            layer_rotations=debug_candidate.layer_rotations,
            assignment_projected_points=debug_candidate.assignment_projected_points,
            assignment_object_points=debug_candidate.assignment_object_points,
            assignment_projected_class_ids=debug_candidate.assignment_projected_class_ids,
            turn_candidate_scores=(),
            turn_candidate_debug=(),
            final_solve_image_points=None,
            pre_final_rvec=None,
            pre_final_tvec=None,
        )
        rendered = draw_sticker_pnp_candidate(
            base_image,
            rendered_candidate,
            show_status=False,
            show_final_pose=False,
        )
        if rank < len(candidate.turn_candidate_scores):
            _label, mean_score, inlier_count, p90_error, mean_error = (
                candidate.turn_candidate_scores[rank]
            )
            cv2.putText(
                rendered,
                (
                    f"turn rank {rank} {debug_candidate.label} | "
                    f"in {inlier_count} | mean {mean_score:.3f} | "
                    f"kp p90 {p90_error:.3f} | kp mean {mean_error:.3f}"
                ),
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                PNP_ASSIGNMENT_COLOR_BGR,
                1,
                cv2.LINE_AA,
            )
        safe_label = "".join(
            character if character.isalnum() else "_"
            for character in debug_candidate.label
        ).strip("_")
        image_path = output_dir / f"frame_{frame_label}_rank_{rank:02d}_{safe_label}.png"
        cv2.imwrite(str(image_path), rendered)


def draw_sticker_pnp_candidate(
    image: np.ndarray,
    candidate: StickerPnPCandidate | None,
    show_status: bool,
    show_final_pose: bool = False,
    show_orientation_axes: bool = True,
    orientation_axis_length: float = 24.0,
    turn_candidate_log: Path | None = None,
    turn_candidate_image_dir: Path | None = None,
    frame_index: int | None = None,
    show_sticker_identities: bool = False,
    sticker_color_names: dict[str, str] | None = None,
) -> np.ndarray:
    """Draw the canonical cube pose and selected sticker reprojection."""

    output = image.copy()
    if candidate is None:
        return output
    draw_reprojection_error_overlay(output, candidate)
    has_pre_final_pose = (
        candidate.pre_final_rvec is not None
        and candidate.pre_final_tvec is not None
    )
    has_final_pose = show_final_pose and has_pre_final_pose
    if has_final_pose:
        draw_projected_cube_pose(
            output,
            candidate,
            candidate.pre_final_rvec,
            candidate.pre_final_tvec,
            (0, 0, 0),
            1,
        )
    # The candidate stored in the buffer is always the current finalized
    # measurement. ``show_final_pose`` controls whether the pre-final pose is
    # shown as an additional diagnostic, never which pose is displayed as the
    # main result.
    display_rvec = candidate.rvec
    display_tvec = candidate.tvec
    if not draw_projected_cube_pose(
        output,
        candidate,
        display_rvec,
        display_tvec,
        (0, 0, 0),
        2,
        outline=True,
    ):
        return output
    if show_orientation_axes:
        draw_orientation_axes(
            output,
            candidate,
            display_rvec,
            display_tvec,
            orientation_axis_length,
        )

    for point in candidate.image_points:
        x, y = np.round(point).astype(np.int32)
        cv2.circle(output, (int(x), int(y)), 6, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(output, (int(x), int(y)), 4, PNP_STICKER_COLOR_BGR, -1, cv2.LINE_AA)
    projected_sticker_points = candidate.projected_sticker_points
    assignment_projected_points = candidate.assignment_projected_points
    for point in projected_sticker_points:
            x, y = np.round(point).astype(np.int32)
            cv2.drawMarker(
                output,
                (int(x), int(y)),
                PNP_STICKER_COLOR_BGR,
                cv2.MARKER_CROSS,
                10,
                2,
                cv2.LINE_AA,
            )

    # Draw the assignment layer last so the complete segment remains visible
    # instead of being hidden under the observed/projected point markers.
    assignment_image_points = candidate.assignment_image_points
    if (
        assignment_image_points is not None
        and assignment_projected_points is not None
        and len(assignment_image_points) == len(assignment_projected_points)
    ):
        assignment_overlay = output.copy()
        for observed, projected in zip(
            assignment_image_points,
            assignment_projected_points,
            strict=True,
        ):
            start = tuple(int(value) for value in np.round(observed).astype(np.int32))
            end = tuple(int(value) for value in np.round(projected).astype(np.int32))
            cv2.line(
                assignment_overlay,
                start,
                end,
                PNP_ASSIGNMENT_COLOR_BGR,
                2,
                cv2.LINE_AA,
            )
            # Assignment targets are surface-lattice points, so most are
            # inside a projected cube face rather than on its eight-edge
            # wireframe. Mark them explicitly at the line endpoint.
            cv2.drawMarker(
                assignment_overlay,
                end,
                PNP_ASSIGNMENT_COLOR_BGR,
                cv2.MARKER_CROSS,
                7,
                1,
                cv2.LINE_AA,
            )
        output = cv2.addWeighted(output, 0.55, assignment_overlay, 0.45, 0.0)

    # Highlight the observed points that were retained for the final PnP
    # solve. These are intentionally drawn last so they remain distinguishable
    # from the original sticker-anchor points and assignment lines.
    if candidate.final_solve_image_points is not None:
        for point in candidate.final_solve_image_points:
            x, y = np.round(point).astype(np.int32)
            cv2.circle(output, (int(x), int(y)), 5, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(
                output,
                (int(x), int(y)),
                3,
                PNP_FINAL_SOLVE_COLOR_BGR,
                -1,
                cv2.LINE_AA,
            )

    if show_sticker_identities:
        draw_projected_sticker_identity_labels(
            output,
            candidate,
            display_rvec,
            display_tvec,
        )
    if sticker_color_names is not None:
        draw_projected_sticker_colors(
            output,
            candidate,
            sticker_color_names,
            display_rvec,
            display_tvec,
        )

    turn_candidate_scores = getattr(candidate, "turn_candidate_scores", ())
    if show_status:
        pose_label = "final" if has_final_pose else "candidate"
        fit_error = pose_reprojection_fit_metric(candidate)
        fit_text = f"{fit_error:.3f}" if np.isfinite(fit_error) else "inf"
        label = (
            f"{pose_label} sticker {candidate.sticker_type} | "
            f"kp {candidate.sticker_average_confidence:.2f} | "
            f"fit {fit_text}/{PNP_POSE_REPROJECTION_ERROR_THRESHOLD:.3f} | "
            f"sticker {candidate.reprojection_error:.1f}px | "
            f"all {candidate.overall_reprojection_error:.1f} | "
            f"anchor {getattr(candidate.orientation_anchor, 'label', 'all')}"
        )
        cv2.putText(
            output,
            label,
            (8, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            PNP_CUBE_COLOR_BGR,
            1,
            cv2.LINE_AA,
        )
        direction_consistency, mean_residual = assignment_residual_direction_statistics(
            candidate.assignment_image_points,
            candidate.assignment_projected_points,
        )
        direction_label = (
            f"res dir {direction_consistency:.2f} | "
            f"mean ({mean_residual[0]:+.1f}, {mean_residual[1]:+.1f})"
        )
        cv2.putText(
            output,
            direction_label,
            (8, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            PNP_ASSIGNMENT_COLOR_BGR,
            1,
            cv2.LINE_AA,
        )
    # Turn diagnostics are independent of the general status overlay.  The
    # latter is off by default, but the turn ranking is useful while debugging
    # PnP behavior and should remain visible whenever a turn search ran.
    if turn_candidate_scores:
        selected_turn_label = layer_rotation_debug_label(candidate.layer_rotations)
        turn_y = 110 if show_status else 66
        for index, (label, mean_score, inlier_count, p90_error, mean_error) in enumerate(
            turn_candidate_scores[:PNP_TURN_OVERLAY_CANDIDATE_COUNT]
        ):
            selected_marker = "*" if label == selected_turn_label else " "
            turn_label = (
                f"turn {selected_marker}{label} | in {inlier_count} | "
                f"mean {mean_score:.3f} | kp p90 {p90_error:.3f} | "
                f"kp mean {mean_error:.3f}"
            )
            cv2.putText(
                output,
                turn_label,
                (8, turn_y + 16 * index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                PNP_ASSIGNMENT_COLOR_BGR,
                1,
                cv2.LINE_AA,
            )
    if getattr(candidate, "turn_fit_evaluated", False):
        def score_text(scores: tuple[float, float, float] | None) -> str:
            if scores is None:
                return "none"
            return ",".join(
                f"{value:.3f}" if np.isfinite(value) else "inf"
                for value in scores
            )

        static_scores = getattr(candidate, "turn_fit_static_scores", None)
        articulated_scores = getattr(candidate, "turn_fit_articulated_scores", None)
        score_y = (
            (110 if show_status else 66)
            + 16 * min(
                len(turn_candidate_scores),
                PNP_TURN_OVERLAY_CANDIDATE_COUNT,
            )
            + 18
        )

        def draw_fit_score_line(
            label: str,
            scores: tuple[float, float, float] | None,
            y: int,
        ) -> None:
            if scores is None:
                return
            values = score_text(scores).split(",")
            cv2.putText(
                output,
                f"{label} kp {values[0]} heat {values[1]} sum {values[2]}",
                (8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                PNP_ASSIGNMENT_COLOR_BGR,
                1,
                cv2.LINE_AA,
            )

        draw_fit_score_line("fit static", static_scores, score_y)
        draw_fit_score_line("fit move", articulated_scores, score_y + 16)
    if turn_candidate_image_dir is not None:
        save_turn_candidate_images(
            image,
            candidate,
            turn_candidate_image_dir,
            frame_index,
        )
    return output


def draw_reprojection_error_overlay(
    image: np.ndarray,
    candidate: StickerPnPCandidate,
) -> np.ndarray:
    """Draw pose reprojection diagnostics without writing to the console."""

    overall_error = float(candidate.overall_reprojection_error)
    fit_error = pose_reprojection_fit_metric(candidate)
    fit_text = f"{fit_error:.3f}" if np.isfinite(fit_error) else "inf"
    anchor_error = float(candidate.reprojection_error)
    overall_text = (
        f"{overall_error:.1f}px"
        if np.isfinite(overall_error)
        else "inf"
    )
    anchor_text = (
        f"{anchor_error:.1f}px"
        if np.isfinite(anchor_error)
        else "inf"
    )
    label = (
        f"REPROJ fit {fit_text}/{PNP_POSE_REPROJECTION_ERROR_THRESHOLD:.3f}"
        f" | raw {overall_text} | anchor {anchor_text}"
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        font,
        scale,
        thickness,
    )
    height, width = image.shape[:2]
    origin = (
        max(8, width - text_width - 8),
        max(text_height + baseline + 4, 22),
    )
    # A dark backing keeps the diagnostic readable over the video without
    # changing the underlying pose or mask rendering.
    x, y = origin
    cv2.rectangle(
        image,
        (x - 4, y - text_height - 4),
        (min(width - 1, x + text_width + 4), min(height - 1, y + baseline + 2)),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        image,
        label,
        origin,
        font,
        scale,
        (40, 220, 40),
        thickness,
        cv2.LINE_AA,
    )
    return image


def draw_translucent_sticker_pnp_candidate(
    image: np.ndarray,
    candidate: StickerPnPCandidate,
    opacity: float = 0.45,
) -> np.ndarray:
    """Draw a rejected pose translucently while preserving the video below."""

    opacity = float(np.clip(opacity, 0.0, 1.0))
    rendered = draw_sticker_pnp_candidate(
        image,
        candidate,
        show_status=False,
        show_final_pose=False,
        show_orientation_axes=True,
    )
    changed = np.any(rendered != image, axis=2)
    if not np.any(changed) or opacity <= 0.0:
        return image
    output = image.copy().astype(np.float32)
    base_pixels = image[changed].astype(np.float32)
    pose_pixels = rendered[changed].astype(np.float32)
    output[changed] = (1.0 - opacity) * base_pixels + opacity * pose_pixels
    return np.clip(output, 0.0, 255.0).astype(np.uint8)


def draw_context(
    image: np.ndarray,
    roi_xyxy: tuple[int, int, int, int] | None,
    candidate: object | None,
    roi_source: str,
    point_count: int,
    show_status: bool,
) -> np.ndarray:
    output = image.copy()
    # The tracker retains the detector candidate so it can keep the original
    # ROI geometry, but that candidate is not a fresh detection. Only show the
    # blue detector box on frames where the detector supplied the ROI.
    if roi_source == "detector" and candidate is not None and hasattr(candidate, "bbox_xyxy"):
        bx1, by1, bx2, by2 = tuple(int(value) for value in candidate.bbox_xyxy)
        cv2.rectangle(output, (bx1, by1), (bx2 - 1, by2 - 1), (255, 120, 0), 1, cv2.LINE_AA)
    if roi_xyxy is not None:
        x1, y1, x2, y2 = roi_xyxy
        cv2.rectangle(output, (x1, y1), (x2 - 1, y2 - 1), (0, 220, 0), 1, cv2.LINE_AA)
        status = f"{roi_source} ROI | {point_count} points"
    else:
        status = "no cube bbox"
    if show_status:
        cv2.putText(output, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 220, 40), 1, cv2.LINE_AA)
    return output


def reset_pose_temporal_state(
    pose_motion_filter: PoseMotionFilter | None,
    symmetry_tracker: CubeSymmetryTracker | None,
    pose_hypothesis_buffer: PoseHypothesisBuffer | None,
) -> None:
    """Drop temporal state after the tracked cube is no longer observable."""

    if pose_motion_filter is not None:
        pose_motion_filter.reset()
    if symmetry_tracker is not None:
        symmetry_tracker.reset()
    if pose_hypothesis_buffer is not None:
        pose_hypothesis_buffer.clear()


def prepare_pose_hypotheses(
    candidates: tuple[StickerPnPCandidate, ...],
    symmetry_tracker: CubeSymmetryTracker | None,
) -> tuple[StickerPnPCandidate, ...]:
    """Canonicalize the cheap current-frame hypotheses before ranking them."""

    prepared: list[StickerPnPCandidate] = []
    for candidate in candidates:
        finalized = candidate
        if symmetry_tracker is not None:
            finalized = canonicalize_pose_symmetry(finalized, symmetry_tracker)
        prepared.append(
            replace(
                finalized,
                orientation_anchor=orientation_anchor_for_candidate(finalized),
            )
        )
    return tuple(prepared)


def render_frame(
    frame: np.ndarray,
    bbox_detector: YOLOCubeDetector,
    pose_model: object,
    roi_tracker: RoiTracker,
    args: argparse.Namespace,
    sticker_seg_model: object | None = None,
    turn_candidate_log: Path | None = None,
    turn_candidate_image_dir: Path | None = None,
    frame_index: int | None = None,
    pose_motion_filter: PoseMotionFilter | None = None,
    symmetry_tracker: CubeSymmetryTracker | None = None,
    pose_hypothesis_buffer: PoseHypothesisBuffer | None = None,
    sticker_color_trace: StickerColorTrace | None = None,
    sticker_color_names: dict[str, str] | None = None,
    sticker_likelihoods: dict[str, dict[str, float]] | None = None,
    cached_final_only: bool = False,
    *,
    neural_cache: list[NeuralFrameData] | None = None,
    neural_data: NeuralFrameData | None = None,
    neural_only: bool = False,
    candidate_only: bool = False,
    collect_color_observations: bool = True,
) -> tuple[np.ndarray, str, int]:
    height, width = frame.shape[:2]

    def log_turn_frame(candidate_for_log: StickerPnPCandidate | None) -> None:
        # Candidate generation and the neural-only foreground pass do not run
        # turn fitting. Log only the final/live pose pass so each video frame
        # contributes exactly one JSONL record.
        if (
            turn_candidate_log is not None
            and bool(getattr(args, "recognize_moves", False))
            and not neural_only
            and not candidate_only
        ):
            append_turn_candidate_log(
                turn_candidate_log,
                candidate_for_log,
                frame_index,
            )

    if neural_data is not None:
        roi_xyxy = neural_data.roi_xyxy
        candidate = None
        roi_source = neural_data.roi_source
    else:
        roi_xyxy, candidate, roi_source = shared_roi_for_frame(
            frame,
            bbox_detector,
            roi_tracker,
            args.roi_padding,
        )
    if roi_xyxy is None:
        if neural_cache is not None:
            neural_cache.append(
                NeuralFrameData(
                    frame_index=0 if frame_index is None else frame_index,
                    frame=frame.copy(),
                    roi_xyxy=None,
                    roi_source="none",
                    sticker_masks=None,
                    local_points=np.empty((0, 2), dtype=np.float32),
                    point_confidences=np.empty((0,), dtype=np.float32),
                    class_ids=np.empty((0,), dtype=np.int32),
                    box_confidences=np.empty((0,), dtype=np.float32),
                )
            )
        reset_pose_temporal_state(
            pose_motion_filter,
            symmetry_tracker,
            pose_hypothesis_buffer,
        )
        output = draw_context(frame, None, None, "none", 0, args.show_status)
        if sticker_color_names is not None:
            output = draw_classified_cube_diagnostic_panel(
                output,
                None,
                sticker_color_names,
                sticker_likelihoods=sticker_likelihoods,
                orientation_axis_length=args.orientation_axis_length,
                show_sticker_identities=getattr(args, "show_sticker_identities", True),
            )
        log_turn_frame(None)
        return output, "none", 0

    x1, y1, x2, y2 = roi_xyxy
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        if neural_cache is not None:
            neural_cache.append(
                NeuralFrameData(
                    frame_index=0 if frame_index is None else frame_index,
                    frame=frame.copy(),
                    roi_xyxy=roi_xyxy,
                    roi_source=roi_source,
                    sticker_masks=None,
                    local_points=np.empty((0, 2), dtype=np.float32),
                    point_confidences=np.empty((0,), dtype=np.float32),
                    class_ids=np.empty((0,), dtype=np.int32),
                    box_confidences=np.empty((0,), dtype=np.float32),
                )
            )
        roi_tracker.reset()
        reset_pose_temporal_state(
            pose_motion_filter,
            symmetry_tracker,
            pose_hypothesis_buffer,
        )
        output = draw_context(frame, roi_xyxy, candidate, roi_source, 0, args.show_status)
        if sticker_color_names is not None:
            output = draw_classified_cube_diagnostic_panel(
                output,
                None,
                sticker_color_names,
                sticker_likelihoods=sticker_likelihoods,
                orientation_axis_length=args.orientation_axis_length,
                show_sticker_identities=getattr(args, "show_sticker_identities", True),
            )
        log_turn_frame(None)
        return output, roi_source, 0

    if neural_data is None:
        model_roi = resize_to_square(roi, args.view_size)
        if args.parallel_cuda:
            sticker_masks, result = predict_models_parallel_cuda(
                sticker_seg_model,
                pose_model,
                model_roi,
                args,
            )
        elif is_cpu_device(args.device):
            sticker_masks, result = predict_models_parallel_cpu(
                sticker_seg_model,
                pose_model,
                model_roi,
                args,
            )
        else:
            sticker_masks = (
                predict_sticker_masks(sticker_seg_model, model_roi, args)
                if sticker_seg_model is not None and args.run_sticker_segmentation
                else None
            )
            result = predict_pose_result(pose_model, model_roi, args)
        if result is None:
            local_points = np.empty((0, 2), dtype=np.float32)
            point_confidences = np.empty((0,), dtype=np.float32)
            class_ids = np.empty((0,), dtype=np.int32)
            box_confidences = np.empty((0,), dtype=np.float32)
        else:
            local_points, point_confidences, class_ids, box_confidences = pose_predictions(result)
        if neural_cache is not None:
            neural_cache.append(
                NeuralFrameData(
                    frame_index=0 if frame_index is None else frame_index,
                    frame=frame.copy(),
                    roi_xyxy=roi_xyxy,
                    roi_source=roi_source,
                    sticker_masks=copy_sticker_masks(sticker_masks),
                    local_points=np.asarray(local_points, dtype=np.float32).copy(),
                    point_confidences=np.asarray(point_confidences, dtype=np.float32).copy(),
                    class_ids=np.asarray(class_ids, dtype=np.int32).copy(),
                    box_confidences=np.asarray(box_confidences, dtype=np.float32).copy(),
                )
            )
    else:
        sticker_masks = neural_data.sticker_masks
        local_points = np.asarray(neural_data.local_points)
        point_confidences = np.asarray(neural_data.point_confidences)
        class_ids = np.asarray(neural_data.class_ids)
        box_confidences = np.asarray(neural_data.box_confidences)
    if cached_final_only and neural_data is not None and neural_data.pose_ready:
        pnp_candidate = neural_data.pose_candidate
        diagnostic_candidate = getattr(
            neural_data,
            "pose_diagnostic_candidate",
            None,
        )
        output = frame.copy()
        point_count = int(len(local_points))
        if args.show_sticker_mask and sticker_masks is not None and roi_xyxy is not None:
            sticker_matches = sticker_keypoint_matches_for_masks(
                sticker_masks[0],
                local_points,
                point_confidences,
                class_ids,
                box_confidences,
                args.keypoint_conf,
                args.pose_class_conf,
            )
            output = render_sticker_masks(
                output,
                roi_xyxy,
                sticker_masks,
                args.sticker_mask_opacity,
                args.sticker_border_thickness,
                pose_predictions=(
                    local_points,
                    point_confidences,
                    class_ids,
                    box_confidences,
                ),
                point_threshold=args.keypoint_conf,
                class_confidences=args.pose_class_conf,
                show_class_labels=args.show_labels,
                sticker_matches=sticker_matches,
                sticker_mask_erosion=args.sticker_mask_erosion,
            )
        if roi_xyxy is not None:
            frame_points, _ = local_to_frame_points(
                local_points,
                roi_xyxy,
                args.view_size,
            )
            output, _ = draw_pose(
                output,
                frame_points,
                point_confidences,
                class_ids,
                box_confidences,
                args.keypoint_conf,
                args.pose_class_conf,
                args.show_labels,
                args.show_confidences,
                copy_image=False,
            )
        output = draw_context(
            output,
            roi_xyxy,
            pnp_candidate,
            roi_source,
            point_count,
            args.show_status,
        )
        if args.show_pnp:
            if pnp_candidate is not None:
                output = draw_sticker_pnp_candidate(
                    output,
                    pnp_candidate,
                    args.show_status,
                    getattr(args, "show_final_pnp_pose", False),
                    getattr(args, "show_orientation_axes", True),
                    getattr(args, "orientation_axis_length", 24.0),
                    turn_candidate_log if getattr(args, "recognize_moves", False) else None,
                    turn_candidate_image_dir if getattr(args, "recognize_moves", False) else None,
                    frame_index,
                    getattr(args, "show_sticker_identities", False),
                    None,
                )
            elif diagnostic_candidate is not None:
                output = draw_translucent_sticker_pnp_candidate(
                    output,
                    diagnostic_candidate,
                )
                output = draw_reprojection_error_overlay(
                    output,
                    diagnostic_candidate,
                )
        output = draw_sticker_mask_centroids(
            output,
            roi_xyxy,
            sticker_masks,
            sticker_mask_erosion=getattr(args, "sticker_mask_erosion", 0),
            show_sticker_identities=getattr(args, "show_sticker_identities", True),
        )
        if sticker_color_names is not None:
            output = draw_classified_cube_diagnostic_panel(
                output,
                pnp_candidate,
                sticker_color_names,
                sticker_likelihoods=sticker_likelihoods,
                orientation_axis_length=args.orientation_axis_length,
                show_sticker_identities=getattr(args, "show_sticker_identities", True),
            )
        log_turn_frame(pnp_candidate if pnp_candidate is not None else diagnostic_candidate)
        return output, roi_source, point_count
    if len(local_points) == 0:
        roi_tracker.reset()
        reset_pose_temporal_state(
            pose_motion_filter,
            symmetry_tracker,
            pose_hypothesis_buffer,
        )
        output = render_sticker_masks(
            frame,
            roi_xyxy,
            sticker_masks if args.show_sticker_mask else None,
            args.sticker_mask_opacity,
            args.sticker_border_thickness,
            sticker_mask_erosion=args.sticker_mask_erosion,
        )
        output = draw_context(output, roi_xyxy, candidate, roi_source, 0, args.show_status)
        output = draw_sticker_mask_centroids(
            output,
            roi_xyxy,
            sticker_masks,
            sticker_mask_erosion=getattr(args, "sticker_mask_erosion", 0),
            show_sticker_identities=getattr(args, "show_sticker_identities", True),
        )
        if sticker_color_names is not None:
            output = draw_classified_cube_diagnostic_panel(
                output,
                None,
                sticker_color_names,
                orientation_axis_length=args.orientation_axis_length,
                show_sticker_identities=getattr(args, "show_sticker_identities", True),
            )
        log_turn_frame(None)
        return output, roi_source, 0

    sticker_matches = (
        tuple()
        if neural_only
        else (
            sticker_keypoint_matches_for_masks(
                sticker_masks[0],
                local_points,
                point_confidences,
                class_ids,
                box_confidences,
                args.keypoint_conf,
                args.pose_class_conf,
            )
            if sticker_masks is not None
            else tuple()
        )
    )
    output = render_sticker_masks(
        frame,
        roi_xyxy,
        sticker_masks if args.show_sticker_mask else None,
        args.sticker_mask_opacity,
        args.sticker_border_thickness,
        pose_predictions=(local_points, point_confidences, class_ids, box_confidences),
        point_threshold=args.keypoint_conf,
        class_confidences=args.pose_class_conf,
        show_class_labels=args.show_labels,
        sticker_matches=sticker_matches,
        sticker_mask_erosion=args.sticker_mask_erosion,
    )

    frame_points, local_roi_points = local_to_frame_points(local_points, roi_xyxy, args.view_size)
    output, point_count = draw_pose(
        output,
        frame_points,
        point_confidences,
        class_ids,
        box_confidences,
        args.keypoint_conf,
        args.pose_class_conf,
        args.show_labels,
        args.show_confidences,
        copy_image=False,
    )
    output = draw_context(output, roi_xyxy, candidate, roi_source, point_count, args.show_status)
    if neural_only:
        # The foreground pass is deliberately limited to detector/model
        # output. Pose candidates are selected by the asynchronous consumer.
        output = draw_sticker_mask_centroids(
            output,
            roi_xyxy,
            sticker_masks,
            sticker_mask_erosion=getattr(args, "sticker_mask_erosion", 0),
            show_sticker_identities=getattr(args, "show_sticker_identities", True),
        )
        roi_tracker.observe_tracking_bounds(
            tracking_bounds(
                local_roi_points,
                point_confidences,
                class_ids,
                box_confidences,
                args.pose_class_conf,
                args.track_conf,
                roi.shape[:2],
            ),
            width,
            height,
            tracking_gain=args.roi_tracking_gain,
            tracking_padding=args.roi_tracking_padding,
        )
        return output, roi_source, point_count
    pnp_candidate = (
        neural_data.pose_candidate
        if neural_data is not None and neural_data.pose_ready
        else None
    )
    candidate_hypotheses = (
        tuple(getattr(neural_data, "pose_candidates", ()))
        if neural_data is not None and neural_data.pose_ready
        else ()
    )
    if not candidate_hypotheses and pnp_candidate is not None:
        candidate_hypotheses = (pnp_candidate,)
    recognize_moves = bool(getattr(args, "recognize_moves", False))

    def record_color_observations(candidate: StickerPnPCandidate) -> None:
        if not collect_color_observations or sticker_color_trace is None:
            return
        sticker_observations = sample_projected_sticker_colors(
            frame,
            roi_xyxy,
            sticker_masks,
            candidate,
            max_samples_per_sticker=args.color_samples_per_sticker,
            min_samples_per_sticker=args.color_min_samples_per_sticker,
            sticker_mask_erosion=args.sticker_mask_erosion,
        )
        candidate_rotation, _ = cv2.Rodrigues(
            np.asarray(candidate.rvec, dtype=np.float64).reshape(3, 1)
        )
        sticker_observation_weights = {
            identity.label: sticker_view_reliability_weight(
                candidate,
                identity,
                candidate_rotation,
            )
            for identity in sticker_color_trace.identities
            if identity.label in sticker_observations
        }
        sticker_color_trace.observe(
            0 if frame_index is None else frame_index,
            sticker_observations,
            observation_weights=sticker_observation_weights,
        )

    should_generate_candidate = pnp_candidate is None and (
        args.show_pnp or candidate_only
    )
    if should_generate_candidate:
        current_candidates = best_sticker_pnp_candidates(
            frame_points,
            class_ids,
            sticker_matches,
            (width, height),
            args.pnp_focal_scale,
            sticker_masks[0] if sticker_masks is not None else None,
            roi_xyxy,
            frame_points,
            sticker_mask_erosion=args.sticker_mask_erosion,
        )
        prepared_candidates = prepare_pose_hypotheses(
            current_candidates,
            symmetry_tracker,
        )
        # Move recognition must not change the static candidate selected from
        # the current frame.  Turn fitting happens only after static PnP
        # refinement and hypothesis selection below.
        pnp_candidate = prepared_candidates[0] if prepared_candidates else None
        candidate_hypotheses = tuple(prepared_candidates)
    if candidate_only:
        if neural_data is not None:
            neural_data.pose_candidate = pnp_candidate
            neural_data.pose_candidates = candidate_hypotheses
            neural_data.pose_ready = True
        if pnp_candidate is not None:
            # Keep the inexpensive color history used by pose symmetry and
            # later turn scoring; defer only the expensive pose refinement.
            record_color_observations(pnp_candidate)
        output = draw_sticker_mask_centroids(
            output,
            roi_xyxy,
            sticker_masks,
            sticker_mask_erosion=getattr(args, "sticker_mask_erosion", 0),
            show_sticker_identities=getattr(args, "show_sticker_identities", True),
        )
        roi_tracker.observe_tracking_bounds(
            tracking_bounds(
                local_roi_points,
                point_confidences,
                class_ids,
                box_confidences,
                args.pose_class_conf,
                args.track_conf,
                roi.shape[:2],
            ),
            width,
            height,
            tracking_gain=args.roi_tracking_gain,
            tracking_padding=args.roi_tracking_padding,
        )
        return output, roi_source, point_count
    refined_candidates: tuple[StickerPnPCandidate, ...] = ()
    diagnostic_candidate: StickerPnPCandidate | None = None
    if pnp_candidate is not None and args.show_pnp:
        hypotheses_to_refine = (
            candidate_hypotheses
            if cached_final_only and candidate_hypotheses
            else (pnp_candidate,)
        )
        refined: list[StickerPnPCandidate] = []
        for hypothesis in hypotheses_to_refine:
            refined_candidate = solve_pnp_from_low_error_assignments(
                hypothesis,
                # Refine every retained hypothesis as a static cube first.
                recover_layer_turns=False,
                max_turn_face_candidates=args.pnp_turn_face_candidates,
                record_all_turn_candidates=False,
            )
            if symmetry_tracker is not None:
                # Apply the continuity choice after refinement too.  This is
                # normally a no-op, but protects the displayed axes if a
                # refinement crosses a Rodrigues branch.
                refined_candidate = canonicalize_pose_symmetry(
                    refined_candidate,
                    symmetry_tracker,
                )
            refined.append(refined_candidate)
        refined_candidates = tuple(refined)
        selected_candidate = select_pose_hypothesis(
            refined_candidates,
            pose_motion_filter.predict_pose()
            if pose_motion_filter is not None
            else None,
            pose_hypothesis_buffer.previous_candidates()
            if pose_hypothesis_buffer is not None
            else (),
        )
        if selected_candidate is not None:
            if recognize_moves:
                # This is the only move-search entry point in the live path:
                # the static pose is already selected and finalized.
                selected_candidate = solve_pnp_from_low_error_assignments(
                    selected_candidate,
                    recover_layer_turns=True,
                    max_turn_face_candidates=args.pnp_turn_face_candidates,
                    record_all_turn_candidates=turn_candidate_log is not None,
                )
            pnp_candidate = apply_temporal_pose_filter(
                selected_candidate,
                pose_motion_filter,
            )
            if symmetry_tracker is not None:
                symmetry_tracker.observe(pnp_candidate.rvec)
        else:
            # Do not display or classify a pose that fails the normalized
            # reprojection acceptance gate.
            diagnostic_candidate = min(
                refined_candidates,
                key=pose_hypothesis_visual_score,
                default=None,
            )
            pnp_candidate = None
            if neural_data is not None:
                neural_data.pose_candidate = None
                neural_data.pose_candidates = ()
        if pnp_candidate is not None:
            record_color_observations(pnp_candidate)
            output = draw_sticker_pnp_candidate(
                output,
                pnp_candidate,
                args.show_status,
                getattr(args, "show_final_pnp_pose", False),
                getattr(args, "show_orientation_axes", True),
                getattr(args, "orientation_axis_length", 24.0),
                turn_candidate_log if recognize_moves else None,
                turn_candidate_image_dir if recognize_moves else None,
                frame_index,
                getattr(args, "show_sticker_identities", False),
                None if sticker_color_names is not None else sticker_color_names,
            )
            if sticker_color_names is not None:
                output = draw_classified_cube_diagnostic_panel(
                    output,
                    pnp_candidate,
                    sticker_color_names,
                    sticker_likelihoods=sticker_likelihoods,
                    orientation_axis_length=args.orientation_axis_length,
                    show_sticker_identities=getattr(args, "show_sticker_identities", True),
                )
        elif diagnostic_candidate is not None:
            output = draw_translucent_sticker_pnp_candidate(
                output,
                diagnostic_candidate,
            )
            output = draw_reprojection_error_overlay(
                output,
                diagnostic_candidate,
            )
    elif args.show_pnp and neural_data is not None and neural_data.pose_ready:
        output = draw_sticker_pnp_candidate(
            output,
            pnp_candidate,
            args.show_status,
            getattr(args, "show_final_pnp_pose", False),
            getattr(args, "show_orientation_axes", True),
            getattr(args, "orientation_axis_length", 24.0),
            turn_candidate_log if recognize_moves else None,
            turn_candidate_image_dir if recognize_moves else None,
            frame_index,
            getattr(args, "show_sticker_identities", False),
            None if sticker_color_names is not None else sticker_color_names,
        )
        if sticker_color_names is not None:
            output = draw_classified_cube_diagnostic_panel(
                output,
                pnp_candidate,
                sticker_color_names,
                sticker_likelihoods=sticker_likelihoods,
                orientation_axis_length=args.orientation_axis_length,
                show_sticker_identities=getattr(args, "show_sticker_identities", True),
            )
    output = draw_sticker_mask_centroids(
        output,
        roi_xyxy,
        sticker_masks,
        sticker_mask_erosion=getattr(args, "sticker_mask_erosion", 0),
        show_sticker_identities=getattr(args, "show_sticker_identities", True),
    )
    if neural_data is not None:
        neural_data.pose_candidate = pnp_candidate
        if refined_candidates:
            neural_data.pose_candidates = refined_candidates
        neural_data.pose_ready = True
    log_turn_frame(pnp_candidate if pnp_candidate is not None else diagnostic_candidate)
    # Match the live pipeline: keep using this ROI until the independent pose
    # model fails, then fall back to the bbox detector on the next frame.
    roi_tracker.observe_tracking_bounds(
        tracking_bounds(
            local_roi_points,
            point_confidences,
            class_ids,
            box_confidences,
            args.pose_class_conf,
            args.track_conf,
            roi.shape[:2],
        ),
        width,
        height,
        tracking_gain=args.roi_tracking_gain,
        tracking_padding=args.roi_tracking_padding,
    )
    return output, roi_source, point_count


def create_video_writer(path: Path, frame: np.ndarray, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    fourcc = "mp4v" if suffix in {".mp4", ".m4v", ".mov"} else "MJPG"
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*fourcc),
        fps if fps > 0 else 30.0,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def add_quicktime_frame_counter(path: Path) -> bool:
    """Add a QuickTime-readable frame counter track without re-encoding video.

    QuickTime only enables its frame/timecode display for movies that expose a
    timecode track. OpenCV's ``VideoWriter`` does not create one, so finalize
    QuickTime-compatible outputs with FFmpeg after the writer is released.
    """

    if path.suffix.lower() not in QUICKTIME_VIDEO_EXTENSIONS:
        return False
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "Could not add the QuickTime frame counter because ffmpeg was not "
            "found on PATH. Install ffmpeg or use a .mp4, .m4v, or .mov output "
            "after making ffmpeg available."
        )
    if not path.is_file():
        raise RuntimeError(f"Could not add QuickTime frame counter; video is missing: {path}")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.stem}.",
            suffix=path.suffix,
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        completed = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(path),
                "-map",
                "0",
                "-c",
                "copy",
                "-timecode",
                "00:00:00:00",
                str(temporary_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            details = (completed.stderr or "").strip()
            raise RuntimeError(
                f"ffmpeg could not add a QuickTime frame counter to {path}"
                + (f": {details}" if details else ".")
            )
        if temporary_path.stat().st_size <= 0:
            raise RuntimeError(
                f"ffmpeg produced an empty QuickTime timecode file for {path}"
            )
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return True


def draw_fps_counter(
    image: np.ndarray,
    args: argparse.Namespace,
    processing_fps: float,
    loop_fps: float,
) -> np.ndarray:
    """Draw the processing/loop FPS indicator in the upper-left corner."""

    if not args.show_fps:
        return image
    fps_y = 44 if args.show_status else 22
    cv2.putText(
        image,
        f"PROC {processing_fps:.1f} | LOOP {loop_fps:.1f}",
        (8, fps_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (40, 220, 40),
        1,
        cv2.LINE_AA,
    )
    return image


def headless_progress_message(label: str, completed: int, total: int | None) -> str:
    if total is not None and total > 0:
        bounded_completed = min(max(int(completed), 0), int(total))
        percent = 100.0 * bounded_completed / float(total)
        return f"{label}: {bounded_completed}/{total} ({percent:5.1f}%)"
    return f"{label}: {int(completed)} frames"


def print_headless_progress(label: str, completed: int, total: int | None) -> None:
    """Update one in-place console progress line for non-display runs."""

    message = headless_progress_message(label, completed, total)
    print(f"\r{message:<52}", end="", flush=True)


class CombinedHeadlessProgress:
    """Render progress for concurrent neural and pose passes on one line."""

    def __init__(self, labels: tuple[str, ...]) -> None:
        self._labels = labels
        self._completed = {label: 0 for label in labels}
        self._totals: dict[str, int | None] = {label: None for label in labels}
        self._lock = Lock()

    def update(self, label: str, completed: int, total: int | None) -> None:
        with self._lock:
            self._completed[label] = int(completed)
            self._totals[label] = total
            message = " | ".join(
                headless_progress_message(
                    current_label,
                    self._completed[current_label],
                    self._totals[current_label],
                )
                for current_label in self._labels
            )
            print(f"\r{message:<110}", end="", flush=True)

    def finish(self) -> None:
        with self._lock:
            print()


def make_pose_motion_filter(args: argparse.Namespace) -> PoseMotionFilter | None:
    if not args.pose_motion_filter:
        return None
    return PoseMotionFilter(
        PoseMotionFilterConfig(
            dt_seconds=args.pose_motion_dt,
            max_acceleration_units_s2=args.pose_motion_max_acceleration,
            max_jerk_units_s3=args.pose_motion_max_jerk,
            max_angular_acceleration_rad_s2=args.pose_motion_max_angular_acceleration,
            max_angular_jerk_rad_s3=args.pose_motion_max_angular_jerk,
        )
    )


def consume_neural_cache_for_pose(
    records: SimpleQueue[NeuralFrameData | None],
    bbox_detector: YOLOCubeDetector | None,
    pose_model: object | None,
    sticker_seg_model: object | None,
    args: argparse.Namespace,
    sticker_color_trace: StickerColorTrace | None,
    turn_candidate_log: Path | None,
    turn_candidate_image_dir: Path | None,
    progress: CombinedHeadlessProgress | None = None,
    total_frames: int | None = None,
) -> None:
    """Select cheap pose candidates without holding up neural inference."""

    roi_tracker = RoiTracker()
    pose_hypothesis_buffer = PoseHypothesisBuffer(max_frames=8)
    pose_motion_filter = make_pose_motion_filter(args)
    processed_frames = 0
    while True:
        record = records.get()
        if record is None:
            return
        render_frame(
            record.frame,
            bbox_detector,
            pose_model,
            roi_tracker,
            args,
            sticker_seg_model,
            turn_candidate_log,
            turn_candidate_image_dir,
            record.frame_index,
            pose_motion_filter,
            None,
            pose_hypothesis_buffer,
            sticker_color_trace,
            None,
            neural_data=record,
            candidate_only=True,
        )
        processed_frames += 1
        if progress is not None:
            progress.update("pose pass", processed_frames, total_frames)


def run_neural_video_pass(
    source: int | str,
    bbox_detector: YOLOCubeDetector,
    pose_model: object,
    sticker_seg_model: object | None,
    args: argparse.Namespace,
    *,
    sticker_color_trace: StickerColorTrace | None = None,
    output_path: Path | None = None,
    turn_candidate_log: Path | None = None,
    turn_candidate_image_dir: Path | None = None,
) -> tuple[list[NeuralFrameData], int, bool, float]:
    """Collect/cache every NN result while a worker consumes pose results."""

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {source}")
    if isinstance(source, int):
        configure_camera(capture, args)
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = capture_frame_count if capture_frame_count > 0 else None
    cache: list[NeuralFrameData] = []
    records: SimpleQueue[NeuralFrameData | None] = SimpleQueue()
    progress = (
        CombinedHeadlessProgress(("neural pass", "pose pass"))
        if not args.display
        else None
    )
    worker = Thread(
        target=consume_neural_cache_for_pose,
        args=(
            records,
            None,
            None,
            None,
            args,
            sticker_color_trace,
            turn_candidate_log,
            turn_candidate_image_dir,
            progress,
            total_frames,
        ),
        name="initial-pose-consumer",
        daemon=True,
    )
    worker.start()
    writer = None
    roi_tracker = RoiTracker()
    previous_loop_end = None
    smoothed_loop_fps = None
    smoothed_processing_fps = None
    frame_index = 0
    completed = True
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            render_start = perf_counter()
            now = render_start
            loop_fps = (
                None
                if previous_loop_end is None
                else 1.0 / max(now - previous_loop_end, 1e-6)
            )
            previous_loop_end = now
            rendered, _roi_source, _point_count = render_frame(
                frame,
                bbox_detector,
                pose_model,
                roi_tracker,
                args,
                sticker_seg_model,
                None,
                None,
                frame_index,
                None,
                None,
                None,
                None,
                None,
                neural_cache=cache,
                neural_only=True,
            )
            render_end = perf_counter()
            processing_fps = 1.0 / max(render_end - render_start, 1e-6)
            if loop_fps is None:
                loop_fps = processing_fps
            smoothed_loop_fps = (
                loop_fps
                if smoothed_loop_fps is None
                else 0.9 * smoothed_loop_fps + 0.1 * loop_fps
            )
            smoothed_processing_fps = (
                processing_fps
                if smoothed_processing_fps is None
                else 0.9 * smoothed_processing_fps + 0.1 * processing_fps
            )
            rendered = draw_fps_counter(
                rendered,
                args,
                smoothed_processing_fps,
                smoothed_loop_fps,
            )
            records.put(cache[-1])
            frame_index += 1
            if progress is not None:
                progress.update("neural pass", frame_index, total_frames)
            if output_path is not None:
                if writer is None:
                    writer = create_video_writer(
                        output_path,
                        rendered,
                        float(capture.get(cv2.CAP_PROP_FPS)),
                    )
                writer.write(rendered)
            if args.display:
                cv2.imshow(WINDOW_NAME, rendered)
                key = cv2.waitKey(30 if args.once else 1) & 0xFF
                if args.once or key in (ord("q"), 27, 13):
                    completed = args.once
                    break
            elif args.once:
                break
    finally:
        if writer is not None:
            writer.release()
        capture.release()
        records.put(None)
        worker.join()
        if progress is not None:
            progress.finish()
    if output_path is not None and writer is not None:
        if add_quicktime_frame_counter(output_path):
            print(f"Added QuickTime frame counter: {output_path}", flush=True)
        print(f"Saved neural output pass: {output_path}", flush=True)
    return cache, frame_index, completed, source_fps


def run_cached_video_pass(
    cache: list[NeuralFrameData],
    args: argparse.Namespace,
    *,
    sticker_color_trace: StickerColorTrace | None = None,
    sticker_color_names: dict[str, str] | None = None,
    sticker_color_names_by_frame: dict[int, dict[str, str]] | None = None,
    sticker_likelihoods_by_frame: dict[int, dict[str, dict[str, float]]] | None = None,
    output_path: Path | None = None,
    fps: float = 30.0,
) -> bool:
    """Replay already-recovered frames for presentation and video writing."""

    writer = None
    roi_tracker = RoiTracker()
    previous_loop_end = None
    smoothed_loop_fps = None
    smoothed_processing_fps = None
    completed = True
    try:
        total_frames = len(cache)
        for record_index, record in enumerate(cache, start=1):
            render_start = perf_counter()
            now = render_start
            loop_fps = (
                None
                if previous_loop_end is None
                else 1.0 / max(now - previous_loop_end, 1e-6)
            )
            previous_loop_end = now
            frame_sticker_color_names = (
                sticker_color_names_by_frame.get(record.frame_index, {})
                if sticker_color_names_by_frame is not None
                else sticker_color_names
            )
            rendered, _roi_source, _point_count = render_frame(
                record.frame,
                None,
                None,
                roi_tracker,
                args,
                None,
                None,
                None,
                record.frame_index,
                None,
                None,
                None,
                sticker_color_trace,
                frame_sticker_color_names,
                sticker_likelihoods=(
                    sticker_likelihoods_by_frame.get(record.frame_index, {})
                    if sticker_likelihoods_by_frame is not None
                    else None
                ),
                # Pose recovery already selected and cached the pose result;
                # this replay only presents it.
                cached_final_only=True,
                neural_data=record,
            )
            render_end = perf_counter()
            processing_fps = 1.0 / max(render_end - render_start, 1e-6)
            if loop_fps is None:
                loop_fps = processing_fps
            smoothed_loop_fps = (
                loop_fps
                if smoothed_loop_fps is None
                else 0.9 * smoothed_loop_fps + 0.1 * loop_fps
            )
            smoothed_processing_fps = (
                processing_fps
                if smoothed_processing_fps is None
                else 0.9 * smoothed_processing_fps + 0.1 * processing_fps
            )
            rendered = draw_fps_counter(
                rendered,
                args,
                smoothed_processing_fps,
                smoothed_loop_fps,
            )
            if output_path is not None:
                if writer is None:
                    writer = create_video_writer(
                        output_path,
                        rendered,
                        fps,
                    )
                writer.write(rendered)
            if not args.display:
                print_headless_progress("final pass", record_index, total_frames)
            if args.display:
                cv2.imshow(WINDOW_NAME, rendered)
                key = cv2.waitKey(30 if args.once else 1) & 0xFF
                if args.once or key in (ord("q"), 27, 13):
                    completed = args.once
                    break
            elif args.once:
                break
    finally:
        if writer is not None:
            writer.release()
        if not args.display:
            print()
    if output_path is not None and writer is not None:
        if add_quicktime_frame_counter(output_path):
            print(f"Added QuickTime frame counter: {output_path}", flush=True)
        print(f"Saved final output pass: {output_path}", flush=True)
    return completed


def run_video_pass(
    source: int | str,
    bbox_detector: YOLOCubeDetector,
    pose_model: object,
    sticker_seg_model: object | None,
    args: argparse.Namespace,
    *,
    sticker_color_trace: StickerColorTrace | None = None,
    sticker_color_names: dict[str, str] | None = None,
    output_path: Path | None = None,
    turn_candidate_log: Path | None = None,
    turn_candidate_image_dir: Path | None = None,
) -> tuple[int, bool]:
    """Run one complete video pass, optionally collecting or drawing colors."""

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {source}")
    if isinstance(source, int):
        configure_camera(capture, args)
    roi_tracker = RoiTracker()
    symmetry_tracker = CubeSymmetryTracker()
    pose_hypothesis_buffer = PoseHypothesisBuffer(max_frames=8)
    pose_motion_filter = make_pose_motion_filter(args)
    writer = None
    previous_loop_end = perf_counter()
    smoothed_loop_fps = None
    smoothed_render_fps = None
    frame_index = 0
    completed = True
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            render_start = perf_counter()
            now = render_start
            loop_fps = 1.0 / max(now - previous_loop_end, 1e-6)
            previous_loop_end = now
            smoothed_loop_fps = (
                loop_fps
                if smoothed_loop_fps is None
                else 0.9 * smoothed_loop_fps + 0.1 * loop_fps
            )
            rendered, _roi_source, _point_count = render_frame(
                frame,
                bbox_detector,
                pose_model,
                roi_tracker,
                args,
                sticker_seg_model,
                turn_candidate_log,
                turn_candidate_image_dir,
                frame_index,
                pose_motion_filter,
                symmetry_tracker,
                pose_hypothesis_buffer,
                sticker_color_trace,
                sticker_color_names,
            )
            frame_index += 1
            render_end = perf_counter()
            render_fps = 1.0 / max(render_end - render_start, 1e-6)
            smoothed_render_fps = (
                render_fps
                if smoothed_render_fps is None
                else 0.9 * smoothed_render_fps + 0.1 * render_fps
            )
            if args.show_fps:
                fps_y = 44 if args.show_status else 22
                cv2.putText(
                    rendered,
                    f"PROC {smoothed_render_fps:.1f} | LOOP {smoothed_loop_fps:.1f}",
                    (8, fps_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (40, 220, 40),
                    1,
                    cv2.LINE_AA,
                )
            if output_path:
                if writer is None:
                    writer = create_video_writer(
                        output_path,
                        rendered,
                        float(capture.get(cv2.CAP_PROP_FPS)),
                    )
                writer.write(rendered)
            if args.display:
                cv2.imshow(WINDOW_NAME, rendered)
                key = cv2.waitKey(30 if args.once else 1) & 0xFF
                if args.once or key in (ord("q"), 27, 13):
                    completed = args.once
                    break
            elif args.once:
                break
    finally:
        if writer is not None:
            writer.release()
        capture.release()
    if output_path is not None and writer is not None:
        if add_quicktime_frame_counter(output_path):
            print(f"Added QuickTime frame counter: {output_path}", flush=True)
        print(f"Saved video pass: {output_path}", flush=True)
    return frame_index, completed


def configure_camera(capture: cv2.VideoCapture, args: argparse.Namespace) -> None:
    """Request a low-latency camera mode and report what OpenCV negotiated."""

    if args.camera_width > 0:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    if args.camera_height > 0:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    if args.camera_fps > 0:
        capture.set(cv2.CAP_PROP_FPS, args.camera_fps)

    try:
        backend = capture.getBackendName()
    except cv2.error:
        backend = "unknown"
    negotiated = (
        f"{capture.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"
        f"{capture.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}@"
        f"{capture.get(cv2.CAP_PROP_FPS):.1f}"
    )
    print(
        "Camera: "
        f"backend={backend}, "
        f"requested={args.camera_width}x{args.camera_height}@{args.camera_fps:g}, "
        f"negotiated={negotiated}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the independent cube-keypoint model with bbox-to-ROI tracking."
    )
    parser.add_argument("--source", required=True, help="Image, video path, or camera index.")
    parser.add_argument("--camera-width", type=int, default=640, help="Requested camera width (0 leaves it unchanged).")
    parser.add_argument("--camera-height", type=int, default=480, help="Requested camera height (0 leaves it unchanged).")
    parser.add_argument("--camera-fps", type=float, default=30.0, help="Requested camera FPS (0 leaves it unchanged).")
    parser.add_argument("--model", type=Path, default=DEFAULT_POSE_MODEL)
    parser.add_argument(
        "--combined-seg-model",
        "--sticker-seg-model",
        dest="sticker_seg_model",
        type=Path,
        default=DEFAULT_COMBINED_SEGMENTATION_MODEL,
        help="Combined cube/sticker segmentation checkpoint; only its sticker class is rendered.",
    )
    parser.add_argument("--bbox-model", type=Path, default=DEFAULT_BBOX_MODEL)
    parser.add_argument("--device", default=None, help="Ultralytics device, for example cpu, mps, or 0.")
    parser.add_argument("--bbox-conf", type=float, default=0.25)
    parser.add_argument("--bbox-imgsz", type=int, default=640)
    parser.add_argument(
        "--pose-conf",
        type=float,
        default=None,
        help="Global model pre-filter; defaults to the lowest per-class threshold.",
    )
    parser.add_argument(
        "--pose-class-conf",
        type=parse_independent_pose_class_confidences,
        default=dict(DEFAULT_INDEPENDENT_POSE_CLASS_CONFIDENCES),
        help=(
            "Per-class boxes.conf thresholds as class=value pairs, for example "
            "corner=0.25,corner-edge=0.30,corner-edge-center=0.25."
        ),
    )
    parser.add_argument(
        "--keypoint-conf",
        "--min-keypoint-conf",
        "--keypoint-confidence",
        "--min-keypoint-confidence",
        dest="keypoint_conf",
        type=float,
        default=0.05,
        help="Minimum independent pose keypoint confidence for drawing points (default: 0.05).",
    )
    parser.add_argument("--track-conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--view-size", type=int, default=256)
    parser.add_argument("--max-det", type=int, default=64)
    parser.add_argument("--sticker-mask-conf", type=float, default=0.25)
    parser.add_argument("--sticker-mask-opacity", type=float, default=0.25)
    parser.add_argument("--sticker-border-thickness", type=int, default=2)
    parser.add_argument(
        "--sticker-mask-erosion",
        type=int,
        default=0,
        help="Erode each sticker mask by this many ROI pixels before rendering and sampling.",
    )
    parser.add_argument(
        "--run-sticker-segmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the combined segmentation model even when its mask is not displayed.",
    )
    parser.add_argument("--show-sticker-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--parallel-cuda",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Experimental: launch sticker segmentation and pose inference on separate CUDA streams.",
    )
    parser.add_argument("--roi-padding", type=float, default=0.08)
    parser.add_argument("--roi-tracking-gain", type=float, default=1.4)
    parser.add_argument("--roi-tracking-padding", type=float, default=0.18)
    parser.add_argument("--show-labels", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--show-confidences", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-status", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--show-pnp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw the cube pose during the final cached pass.",
    )
    parser.add_argument(
        "--show-final-pnp-pose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw the final low-error solvePnP pose over the candidate pose.",
    )
    parser.add_argument(
        "--show-orientation-axes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw the projected positive X/Y/Z axes at the cube center.",
    )
    parser.add_argument(
        "--orientation-axis-length",
        type=float,
        default=24.0,
        help="Orientation gizmo axis length in cube model millimetres.",
    )
    parser.add_argument(
        "--show-sticker-identities",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw stable C/E/K sticker identity numbers on visible stickers.",
    )
    parser.add_argument(
        "--recognize-moves",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable final layer-turn recovery after static pose selection. "
            "Static candidate selection and full-cube refinement remain "
            "independent of move recognition."
        ),
    )
    parser.add_argument(
        "--recover-pose-symmetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Recover whole-cube symmetry changes from cached sticker colors "
            "after the neural pass. This is independent of move recognition."
        ),
    )
    parser.add_argument(
        "--debug-pose-symmetry",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print per-frame pose-symmetry trigger and tested-orientation scores.",
    )
    parser.add_argument(
        "--pose-motion-filter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Temporally filter pose using the stationary layers of the selected "
            "turn hypothesis (disabled by default)."
        ),
    )
    parser.add_argument("--pose-motion-dt", type=float, default=1.0 / 30.0)
    parser.add_argument(
        "--pose-motion-max-acceleration",
        type=float,
        default=20000.0,
        help="Maximum linear acceleration in mm/s^2 (20 m/s^2 by default).",
    )
    parser.add_argument(
        "--pose-motion-max-jerk",
        type=float,
        default=float("inf"),
        help="Linear jerk limit; disabled by default.",
    )
    parser.add_argument(
        "--pose-motion-max-angular-acceleration",
        type=float,
        default=12.0,
    )
    parser.add_argument(
        "--pose-motion-max-angular-jerk",
        type=float,
        default=float("inf"),
        help="Angular jerk limit; disabled by default.",
    )
    parser.add_argument(
        "--turn-candidate-log",
        type=Path,
        default=None,
        help=(
            "Output directory for turn scores and images; logging retains every "
            "evaluated turn permutation (disabled by default)."
        ),
    )
    parser.add_argument(
        "--turn-candidate-images",
        type=Path,
        default=None,
        help="Directory for annotated candidate images (default: same directory as the log).",
    )
    parser.add_argument(
        "--pnp-assignment-mode",
        choices=("closest", "class-aware"),
        default=PNP_ASSIGNMENT_MODE,
        help=(
            "Full-cube assignment strategy: class-aware uses absolute known "
            "class matches; closest is geometric-only."
        ),
    )
    parser.add_argument(
        "--pnp-turn-face-candidates",
        type=int,
        default=PNP_TURN_FACE_CANDIDATES,
        help=(
            "Legacy compatibility option; the current search tests all six "
            "layer faces (default: "
            f"{PNP_TURN_FACE_CANDIDATES})."
        ),
    )
    parser.add_argument(
        "--pnp-focal-scale",
        type=float,
        default=1.2,
        help="Approximate focal length as a multiple of max(image width, height).",
    )
    parser.add_argument("--show-fps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--rgb-html-plot",
        type=Path,
        help=(
            "Output an interactive grouped RGB plot. Each visible sticker contributes "
            "one averaged point per frame when the projected sticker intersects the "
            "sticker/cube segmentation mask."
        ),
    )
    parser.add_argument(
        "--color-samples-per-sticker",
        type=int,
        default=128,
        help="Maximum reliable pixels averaged for each sticker in each frame.",
    )
    parser.add_argument(
        "--color-min-samples-per-sticker",
        type=int,
        default=4,
        help="Minimum reliable pixels required before plotting a sticker for a frame.",
    )
    parser.add_argument(
        "--normalize-rgb-plot",
        action="store_true",
        help="Use brightness-normalized RGB coordinates for the primary HTML plot.",
    )
    parser.add_argument(
        "--plotly-point-size",
        type=float,
        default=5.0,
        help="Marker size in the grouped RGB HTML plot.",
    )
    parser.add_argument("--save", type=Path, help="Output image path, or one frame with --once.")
    parser.add_argument(
        "--save-video",
        type=Path,
        help="Output path for the first pass containing neural outputs only.",
    )
    parser.add_argument(
        "--show-classified-colors",
        "--show-classified-stickers",
        "--show-classified-centers",
        dest="show_classified_colors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Color visible projected stickers during the final cached pass "
            "using the legal standard cube color scheme."
        ),
    )
    parser.add_argument(
        "--debug-independent-color-posteriors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Color each sticker by its final independent posterior argmax; "
            "disable legal cube color assignments while retaining pose recovery."
        ),
    )
    parser.add_argument(
        "--save-final-video",
        type=Path,
        dest="save_final_video",
        help=(
            "Optional output path for the final cached video pass, including pose "
            "output even when color classification is disabled."
        ),
    )
    parser.add_argument(
        "--save-colored-video",
        type=Path,
        dest="save_final_video",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--once", action="store_true", help="Process one video/camera frame and exit.")
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.set_defaults(display=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    global PNP_ASSIGNMENT_MODE
    PNP_ASSIGNMENT_MODE = args.pnp_assignment_mode
    if args.debug_independent_color_posteriors:
        args.show_classified_colors = True
        print(
            "Debug color mode: independent final posteriors; "
            "legal cube color assignments disabled; pose recovery retained",
            flush=True,
        )
    args.device = args.device or default_yolo_device()
    if args.turn_candidate_log is not None:
        turn_candidate_output_dir = args.turn_candidate_log
        args.turn_candidate_log = turn_candidate_output_dir / "turn_candidate_scores.jsonl"
        if args.turn_candidate_images is None:
            args.turn_candidate_images = turn_candidate_output_dir
        turn_candidate_output_dir.mkdir(parents=True, exist_ok=True)
        args.turn_candidate_log.write_text("", encoding="utf-8")
        print(f"Turn candidate log: {args.turn_candidate_log}", flush=True)
    if args.turn_candidate_images is not None:
        args.turn_candidate_images.mkdir(parents=True, exist_ok=True)
        print(f"Turn candidate images: {args.turn_candidate_images}", flush=True)
    print(f"Inference device: {args.device}", flush=True)
    if args.parallel_cuda and not (str(args.device).startswith("cuda") or str(args.device).isdigit()):
        raise SystemExit("--parallel-cuda requires --device cuda, cuda:N, or a CUDA device index")
    if args.pose_conf is not None and not 0.0 <= args.pose_conf <= 1.0:
        raise SystemExit("--pose-conf must be in [0, 1]")
    if args.pose_conf is not None and args.pose_conf > min(args.pose_class_conf.values()):
        raise SystemExit("--pose-conf cannot exceed the lowest --pose-class-conf threshold")
    if not 0.0 <= args.sticker_mask_conf <= 1.0:
        raise SystemExit("--sticker-mask-conf must be in [0, 1]")
    if not 0.0 <= args.sticker_mask_opacity <= 1.0:
        raise SystemExit("--sticker-mask-opacity must be in [0, 1]")
    if args.pnp_focal_scale <= 0.0:
        raise SystemExit("--pnp-focal-scale must be positive")
    if args.orientation_axis_length <= 0.0:
        raise SystemExit("--orientation-axis-length must be positive")
    if not 1 <= args.pnp_turn_face_candidates <= 6:
        raise SystemExit("--pnp-turn-face-candidates must be between 1 and 6")
    if args.sticker_border_thickness < 0:
        raise SystemExit("--sticker-border-thickness must be non-negative")
    if args.sticker_mask_erosion < 0:
        raise SystemExit("--sticker-mask-erosion must be non-negative")
    if args.color_samples_per_sticker < 1:
        raise SystemExit("--color-samples-per-sticker must be positive")
    if args.color_min_samples_per_sticker < 1:
        raise SystemExit("--color-min-samples-per-sticker must be positive")
    if args.plotly_point_size <= 0.0:
        raise SystemExit("--plotly-point-size must be positive")
    if args.rgb_html_plot is not None and not args.run_sticker_segmentation:
        raise SystemExit("--rgb-html-plot requires --run-sticker-segmentation")
    if args.rgb_html_plot is not None and not args.show_pnp:
        raise SystemExit("--rgb-html-plot requires --show-pnp")
    if not args.model.is_file():
        raise SystemExit(f"Could not find independent pose model: {args.model}")
    if args.run_sticker_segmentation and not args.sticker_seg_model.is_file():
        raise SystemExit(f"Could not find sticker segmentation model: {args.sticker_seg_model}")
    if not args.bbox_model.is_file():
        raise SystemExit(f"Could not find bbox model: {args.bbox_model}")

    from ultralytics import YOLO

    bbox_detector = YOLOCubeDetector(
        model_name=str(args.bbox_model),
        confidence_threshold=args.bbox_conf,
        image_size=args.bbox_imgsz,
        device=args.device,
    )
    pose_model = YOLO(str(args.model))
    sticker_seg_model = YOLO(str(args.sticker_seg_model)) if args.run_sticker_segmentation else None
    if sticker_seg_model is not None:
        print(f"Combined segmentation model (sticker class only): {args.sticker_seg_model}", flush=True)
    source = parse_source(args.source)
    if args.show_classified_colors and isinstance(source, int):
        raise SystemExit("--show-classified-colors requires a replayable image or video source")
    if args.show_classified_colors and not args.run_sticker_segmentation:
        raise SystemExit("--show-classified-colors requires --run-sticker-segmentation")
    if args.show_classified_colors and not args.show_pnp:
        raise SystemExit("--show-classified-colors requires --show-pnp")
    sticker_color_trace = (
        StickerColorTrace.create()
        if (
            args.recover_pose_symmetry
            or args.rgb_html_plot is not None
            or args.show_classified_colors
        )
        else None
    )

    if is_image_source(source):
        frame = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if frame is None:
            raise SystemExit(f"Could not read image: {source}")
        image_cache: list[NeuralFrameData] = []
        rendered, roi_source, point_count = render_frame(
            frame,
            bbox_detector,
            pose_model,
            RoiTracker(),
            args,
            sticker_seg_model,
            None,
            None,
            0,
            None,
            None,
            None,
            None,
            None,
            neural_cache=image_cache,
            neural_only=True,
        )
        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.save), rendered):
                raise SystemExit(f"Could not write rendered image: {args.save}")
            print(f"Saved render: {args.save}")
        if args.display:
            cv2.imshow(WINDOW_NAME, rendered)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        print(f"ROI source: {roi_source}; rendered points: {point_count}")
        # Select and cache all cheap pose candidates. Pose/orientation/layer
        # recovery happens below as one joint pass.
        render_frame(
            frame,
            None,
            None,
            RoiTracker(),
            args,
            None,
            args.turn_candidate_log,
            args.turn_candidate_images,
            0,
            make_pose_motion_filter(args),
            None,
            PoseHypothesisBuffer(max_frames=8),
            sticker_color_trace,
            None,
            neural_data=image_cache[0],
            candidate_only=True,
        )
        recovery_trace = (
            sticker_color_trace
            if sticker_color_trace is not None
            else StickerColorTrace.create()
        )
        if args.show_pnp:
            sticker_color_trace = recover_cached_pose_symmetries(
                image_cache,
                recovery_trace,
                debug=args.debug_pose_symmetry,
                progress=not args.display,
                refine_pose=True,
                recover_orientation=(
                    args.recover_pose_symmetry
                    and sticker_color_trace is not None
                ),
                recognize_moves=args.recognize_moves,
                max_turn_face_candidates=args.pnp_turn_face_candidates,
                record_all_turn_candidates=args.turn_candidate_log is not None,
                pose_motion_filter=make_pose_motion_filter(args),
                turn_candidate_log=args.turn_candidate_log,
            )
        sticker_likelihoods_by_frame = (
            build_sticker_observation_likelihoods(sticker_color_trace)
            if args.show_classified_colors and sticker_color_trace is not None
            else {}
        )
        if args.debug_independent_color_posteriors:
            sticker_likelihoods_by_frame = carry_forward_sticker_posteriors(
                sticker_likelihoods_by_frame,
                (0,),
            )
        sticker_color_names_by_frame = None
        if args.show_classified_colors and sticker_color_trace is not None:
            # Keep the independent debug view unconstrained; normal replay
            # applies legal cube rules after pairwise trajectory grouping.
            classifier = (
                classify_sticker_colors_by_pairwise_trajectory
                if args.debug_independent_color_posteriors
                else classify_sticker_colors_by_pairwise_cube_rules
            )
            sticker_color_names = classifier(
                sticker_color_trace,
                sticker_likelihoods_by_frame,
            )
            center_color_names = {
                identity.label: sticker_color_names[identity.label]
                for identity in sticker_color_trace.identities
                if identity.category == "center" and identity.label in sticker_color_names
            }
        else:
            center_color_names = (
                classify_center_color_names(sticker_color_trace)
                if args.show_classified_colors and sticker_color_trace is not None
                else {}
            )
            sticker_color_names = classify_sticker_color_names(
                sticker_color_trace,
                center_color_names,
            ) if args.show_classified_colors and sticker_color_trace is not None else {}
        if center_color_names:
            print(
                "Classified centers: "
                + ", ".join(
                    f"{label}={color}"
                    for label, color in sorted(center_color_names.items())
                ),
                flush=True,
            )
        if (args.show_classified_colors or args.show_pnp) and args.display:
            colored_rendered, _colored_roi_source, _colored_point_count = render_frame(
                frame,
                None,
                None,
                RoiTracker(),
                args,
                None,
                None,
                None,
                0,
                make_pose_motion_filter(args),
                CubeSymmetryTracker(),
                PoseHypothesisBuffer(max_frames=8),
                None,
                (
                    sticker_color_names_by_frame.get(0, {})
                    if sticker_color_names_by_frame is not None
                    else sticker_color_names
                ) if args.show_classified_colors else None,
                sticker_likelihoods=sticker_likelihoods_by_frame.get(0, {}),
                neural_data=image_cache[0],
            )
            cv2.imshow(WINDOW_NAME, colored_rendered)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if sticker_color_trace is not None and args.rgb_html_plot is not None:
            save_sticker_color_plots(
                sticker_color_trace,
                args.rgb_html_plot,
                title="Independent-keypoint sticker RGB observations",
                primary_normalized=args.normalize_rgb_plot,
                point_size=args.plotly_point_size,
            )
            print(f"Saved sticker color plots: {args.rgb_html_plot}")
        return 0

    cache, frame_count, first_pass_completed, source_fps = run_neural_video_pass(
        source,
        bbox_detector,
        pose_model,
        sticker_seg_model,
        args,
        sticker_color_trace=sticker_color_trace,
        output_path=args.save_video,
        turn_candidate_log=args.turn_candidate_log,
        turn_candidate_image_dir=args.turn_candidate_images,
    )
    recovery_trace = (
        sticker_color_trace
        if sticker_color_trace is not None
        else StickerColorTrace.create()
    )
    if args.show_pnp:
        sticker_color_trace = recover_cached_pose_symmetries(
            cache,
            recovery_trace,
            debug=args.debug_pose_symmetry,
            progress=not args.display,
            refine_pose=True,
            recover_orientation=(
                args.recover_pose_symmetry
                and sticker_color_trace is not None
            ),
            recognize_moves=args.recognize_moves,
            max_turn_face_candidates=args.pnp_turn_face_candidates,
            record_all_turn_candidates=args.turn_candidate_log is not None,
            pose_motion_filter=make_pose_motion_filter(args),
            turn_candidate_log=args.turn_candidate_log,
        )
    sticker_likelihoods_by_frame = (
        build_sticker_observation_likelihoods(sticker_color_trace)
        if args.show_classified_colors and sticker_color_trace is not None
        else {}
    )
    if args.debug_independent_color_posteriors:
        sticker_likelihoods_by_frame = carry_forward_sticker_posteriors(
            sticker_likelihoods_by_frame,
            tuple(record.frame_index for record in cache),
        )
    sticker_color_names_by_frame = None
    if args.show_classified_colors and sticker_color_trace is not None:
        # Keep the independent debug view unconstrained; normal replay
        # applies legal cube rules after pairwise trajectory grouping.
        classifier = (
            classify_sticker_colors_by_pairwise_trajectory
            if args.debug_independent_color_posteriors
            else classify_sticker_colors_by_pairwise_cube_rules
        )
        sticker_color_names = classifier(
            sticker_color_trace,
            sticker_likelihoods_by_frame,
        )
        center_color_names = {
            identity.label: sticker_color_names[identity.label]
            for identity in sticker_color_trace.identities
            if identity.category == "center" and identity.label in sticker_color_names
        }
    else:
        center_color_names = (
            classify_center_color_names(sticker_color_trace)
            if args.show_classified_colors and sticker_color_trace is not None
            else {}
        )
        sticker_color_names = classify_sticker_color_names(
            sticker_color_trace,
            center_color_names,
        ) if args.show_classified_colors and sticker_color_trace is not None else {}
    if center_color_names:
        print(
            "Classified centers: "
            + ", ".join(
                f"{label}={color}"
                for label, color in sorted(center_color_names.items())
            ),
            flush=True,
        )
    if (
        first_pass_completed
        and (args.display or args.save_final_video is not None)
    ):
        run_cached_video_pass(
            cache,
            args,
            sticker_color_trace=sticker_color_trace,
            sticker_color_names=sticker_color_names if args.show_classified_colors else None,
            sticker_color_names_by_frame=sticker_color_names_by_frame,
            sticker_likelihoods_by_frame=sticker_likelihoods_by_frame,
            output_path=args.save_final_video,
            fps=source_fps,
        )
    if sticker_color_trace is not None and args.rgb_html_plot is not None:
        save_sticker_color_plots(
            sticker_color_trace,
            args.rgb_html_plot,
            title=(
                "Independent-keypoint sticker RGB observations "
                f"across {frame_count} frames"
            ),
            primary_normalized=args.normalize_rgb_plot,
            point_size=args.plotly_point_size,
        )
        print(f"Saved sticker color plots: {args.rgb_html_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
