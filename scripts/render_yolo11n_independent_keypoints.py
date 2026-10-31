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
        --initial-scramble "" \
        --model runs/pose/runs/independent_keypoint_pose_v2/runs/cube_piece_keypoint_yolo11n_pose/weights/best.pt \
        --save runs/independent_keypoints_render.png \
        --no-display

    .venv/bin/python scripts/render_yolo11n_independent_keypoints.py \
        --source 0 --initial-scramble "" \
        --model runs/pose/runs/independent_keypoint_pose_v2/runs/cube_piece_keypoint_yolo11n_pose/weights/best.pt
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from functools import lru_cache
from itertools import combinations, permutations, product
from pathlib import Path
from queue import SimpleQueue
from threading import Lock, Thread
from time import perf_counter
from typing import Callable

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from live_recognition.detectors import YOLOCubeDetector  # noqa: E402
from live_recognition.cube_state import CubeState  # noqa: E402
from live_recognition.cube_state import parse_scramble_moves  # noqa: E402
from live_recognition.independent_pose_thresholds import (  # noqa: E402
    DEFAULT_INDEPENDENT_POSE_CLASS_CONFIDENCES, INDEPENDENT_POSE_CLASS_NAMES,
    independent_pose_class_threshold, parse_independent_pose_class_confidences)
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
PURE_COLOR_RGB = {
    "white": (255, 255, 255),
    "yellow": (255, 255, 0),
    "red": (255, 0, 0),
    "orange": (255, 165, 0),
    "blue": (0, 0, 255),
    "green": (0, 255, 0),
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
PNP_STICKER_COLOR_BGR = (0, 220, 255)
PNP_FINAL_SOLVE_COLOR_BGR = (0, 255, 0)
PNP_ASSIGNMENT_COLOR_BGR = (255, 0, 255)
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
PNP_MASK_BOUNDARY_MAX_CONTOUR_POINTS = 32
PNP_MASK_BOUNDARY_DISTANCE_DELTA_FRACTION = 0.012
# Segmentation-derived centroid marker.  BGR for a visible pink (RGB
# 255,105,180), kept separate from the three neural keypoint class colors.
STICKER_MASK_CENTROID_COLOR_BGR = (180, 105, 255)
PNP_MASK_CENTER_HYPOTHESIS_WEIGHT = 1.0
PNP_MASK_CENTER_HYPOTHESIS_HUBER_DELTA = 2.0
PNP_MASK_CENTER_UNMATCHED_PENALTY = 1.0
PNP_MASK_CENTER_SHAPE_PENALTY = 4.0
PNP_MASK_CENTER_MIN_VISIBLE_AREA = 64.0
# A centroid at the distance-transform peak is a stronger center observation
# than one sitting in a low-heat region. Keep a floor so weak masks remain
# usable as soft evidence instead of becoming hard rejects.
PNP_MASK_CENTER_HEATMAP_CONFIDENCE_FLOOR = 0.20
PNP_MASK_CENTER_SOLVE_WEIGHT = 0.50
# Suppress duplicate/partial sticker detections mostly covered by a larger
# sticker mask, before erosion and before the masks are used by any pass.
STICKER_MASK_CONTAINMENT_THRESHOLD = 0.90
# Turn ranking uses all segmentation-derived center observations. Its bounded
# coverage penalty is converted to cube-normalized distance units before it is
# combined with keypoint reprojection error.
PNP_TURN_MASK_CENTER_WEIGHT = 1.0
# Color contributes the same weight as each geometric term in the final turn
# ranking. Geometry is keypoint error plus the sticker-centroid term, while
# color is the negative log predictive likelihood p(observation | prior).
PNP_TURN_COLOR_WEIGHT = 1.0
MIN_TURN_LAYER_OBSERVATIONS = 3
# A turn is eligible against static when it has more unified inliers or its
# predictive color error is at least 2x better than the matching baseline.
PNP_TURN_COLOR_RESCUE_MAX_RATIO = 0.5
PNP_TURN_CENTER_PENALTY_SCALE = 0.05
# Search multiple state sequences over the next ten frames. Scores use two
# uniform blocks: the front half gets 30% and the back half gets 70%.
PNP_TURN_LOOKAHEAD_FRAMES = 10
# Scheduling batch size, not a limit on retained states or branching.
PNP_TURN_LOOKAHEAD_STATE_BATCH_SIZE = 32
# Keep the best hold plus the top color-ranked changes at every parent.
PNP_TURN_LOOKAHEAD_STATE_CHANGE_LIMIT = 10
# Bound total frame/state evaluations per decision, including complete-path seeds.
PNP_TURN_LOOKAHEAD_STATE_BUDGET = 1024
# Minimum whole-window fit improvement needed per additional turn. Charge
# this outside temporal weights so later moves do not become cheaper and
# future evidence can outweigh a weak current-frame fit without a hard gate.
PNP_TURN_LOOKAHEAD_MOVE_COST = 0.01
# Final-solve observations must be close relative to the displayed cube size.
# This is deliberately expressed as a fraction of the projected cube diameter
# so the same setting works at different distances from the camera. Keep this
# separate from the acceptance threshold while the turn score is being tuned.
PNP_FINAL_ASSIGNMENT_ERROR_FRACTION = 0.05
# This is the normalized fit threshold used for temporal candidate
# tie-breaking. Outright acceptance rejection remains disabled while this
# threshold is being evaluated on live footage.
PNP_POSE_REPROJECTION_ERROR_THRESHOLD = 0.12
PNP_POSE_REPROJECTION_ERROR_REJECTION_ENABLED = False
PNP_LAYER_MISALIGNMENT_MIN_DEGREES = -90.0
PNP_LAYER_MISALIGNMENT_MAX_DEGREES = 90.0
PNP_LAYER_MISALIGNMENT_STEP_DEGREES = 6.0
POSE_MOTION_DISPLACEMENT_TOLERANCE_MM = 40.0
# Temporarily keep isolated translation jumps instead of rejecting the frame.
POSE_TWO_SIDED_MOTION_REJECTION_ENABLED = False
# Retained for CLI/API compatibility. Move fitting currently evaluates all six
# layer faces; the value no longer limits the face search.
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
# Use a finite RGB profile kernel so nearby color regressions retain
# meaningful likelihood instead of being treated as hard assignments.
COLOR_LIKELIHOOD_SIGMA = 0.08
STICKER_VIEW_WEIGHT_FLOOR = 0.0
# Global profiles are estimated without trusting any sticker's color label.
# These settings make that estimate a small, regularized EM fit: ambiguous
# samples contribute softly, while the pure-color profiles remain weak anchors.
GLOBAL_PROFILE_EM_ITERATIONS = 6
GLOBAL_PROFILE_EM_TEMPERATURE = 0.025
GLOBAL_PROFILE_PRIOR_WEIGHT = 6.0
GLOBAL_PROFILE_MIN_EFFECTIVE_WEIGHT = 0.5
INITIAL_CALIBRATION_MOVE_PATIENCE_FRAMES = 20
LABELED_SAMPLE_MAX_POINTS = 256
LABELED_SAMPLE_NEIGHBORS = 3
LABELED_SAMPLE_DISTANCE_WEIGHT = 0.3
LABELED_SAMPLE_COVERAGE_SCALE = 0.15


@dataclass(frozen=True)
class GlobalColorProfile:
    """A diagnostic color group initialized from the pure-color priors."""

    color_name: str
    prior: StickerColorProfile | None
    fitted: StickerColorProfile | None
    sample_count: int
    labeled_samples: tuple[tuple[float, float, float], ...] = ()
    labeled_sample_weights: tuple[float, ...] = ()


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
    sticker_mask_index: int | None = None
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
    # Per-mask RGB observations used by the color-assisted turn scorer.
    # Entries follow ``sticker_mask_centers`` ordering.
    sticker_mask_colors: tuple[np.ndarray, ...] = ()
    mask_fit_error: float = float("inf")
    mask_boundary_sample_count: int = 0
    mask_center_fit_error: float = float("inf")
    mask_center_match_count: int = 0
    image_size: tuple[int, int] | None = None
    layer_rotations: tuple["CubeLayerRotation", ...] = ()
    # Physical stickers include the selected color offset. Keypoint geometry
    # retains only the partial angle because quarter turns permute its lattice.
    # None preserves the legacy shared geometry/sticker rotation behavior.
    sticker_layer_rotations: tuple["CubeLayerRotation", ...] | None = None
    # Layer selected by the fast removal-based move-recognition mode.  This is
    # deliberately separate from ``layer_rotations``: the selected layer is
    # highlighted, but the displayed cube is not turned.
    # Per-candidate diagnostics: (label, global mean score, inliers,
    # keypoint p90, keypoint mean).
    turn_candidate_scores: tuple[tuple[str, float, int, float, float], ...] = ()
    # Full JSON-safe diagnostics for every tested turn configuration when
    # candidate logging is enabled.
    turn_candidate_diagnostics: tuple[dict[str, object], ...] = ()
    # Exact static pose inputs used while evaluating the turn configurations.
    # These are kept separate because temporal filtering may change the pose
    # after the search has completed.
    turn_fit_search_rvec: np.ndarray | None = None
    turn_fit_search_tvec: np.ndarray | None = None
    turn_candidate_debug: tuple["TurnCandidateDebug", ...] = ()
    # Per-layer pre-search diagnostics: (layer, keypoint, sticker, combined).
    turn_layer_scores: tuple[tuple[str, float, float, float], ...] = ()
    turn_fit_sticker_score: float = float("inf")
    turn_fit_color_score: float = float("inf")
    turn_color_unreliable: bool = False
    # Per-observation p(observation | prior), indexed as
    # [canonical sticker identity, observed mask].
    turn_color_likelihoods: np.ndarray | None = None
    turn_fit_unexplained_keypoints: int = 0
    turn_fit_evaluated: bool = False
    turn_fit_static_scores: tuple[float, float, float] | None = None
    turn_fit_articulated_scores: tuple[float, float, float] | None = None
    # Optional action-vs-static lookahead diagnostics.  The future scores are
    # equal-weight geometric-plus-color scores evaluated under the two
    # competing states.
    turn_fit_lookahead_action_score: float = float("inf")
    turn_fit_lookahead_static_score: float = float("inf")
    turn_fit_lookahead_action_future_scores: tuple[float, ...] = ()
    turn_fit_lookahead_static_future_scores: tuple[float, ...] = ()
    turn_fit_lookahead_rejected: bool = False
    turn_search_diagnostics: dict[str, object] | None = None
    turn_selected_trial_id: str | None = None
    # Explicit discrete state update chosen by the color/state rollout.  None
    # preserves the legacy angle-to-move behavior; an empty tuple explicitly
    # means that this new-format candidate makes no state change.
    turn_state_moves: tuple[str, ...] | None = None
    turn_color_offset_degrees: float = 0.0
    final_solve_image_points: np.ndarray | None = None
    pre_final_rvec: np.ndarray | None = None
    pre_final_tvec: np.ndarray | None = None
    orientation_anchor: "OrientationAnchor | None" = None
    object_symmetry: np.ndarray | None = None
    # Committed physical sticker state before this candidate's incremental
    # layer rotation is applied.  This is part of fitting, not display-only
    # metadata, and is copied into every candidate entering turn search.
    cube_geometry_state: "CubeGeometryState | None" = None


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


    def previous_candidates(self) -> tuple[StickerPnPCandidate, ...]:
        """Return candidates from prior frames, never the current frame."""

        return tuple(
            candidate
            for frame in list(self._frames)[:-1]
            for candidate in frame.candidates
        )



@dataclass(frozen=True)
class CubeLayerRotation:
    """A correction applied to one face direction and layer width."""

    axis: int
    side: int
    angle_degrees: float
    layer_count: int = 1


@dataclass(frozen=True)
class CubeGeometryState:
    """The committed physical sticker locations used by move fitting.

    ``CubeState`` stores colors at fixed facelet locations.  Move fitting also
    needs the inverse view: where each persistent sticker identity currently
    lives.  Keeping that mapping in the cube model prevents the renderer's
    display accumulator and the scorer from silently using different states.
    """

    facelets: dict[tuple[tuple[int, int, int], int, int], str]

    @classmethod
    def identity(cls) -> "CubeGeometryState":
        return cls(
            {
                (identity.cubie_index, identity.axis, identity.side): identity.label
                for identity in canonical_sticker_identities()
            }
        )

    def copy(self) -> "CubeGeometryState":
        return type(self)(dict(self.facelets))

    def signature(self) -> tuple[tuple[str, tuple[tuple[int, int, int], int, int]], ...]:
        return tuple(
            sorted(
                (label, key)
                for key, label in self.facelets.items()
            )
        )

    def position_for(self, label: str) -> tuple[tuple[int, int, int], int, int]:
        for key, value in self.facelets.items():
            if value == label:
                return key
        raise KeyError(f"Unknown sticker identity {label!r}")

    def apply_move(self, move: str) -> "CubeGeometryState":
        return type(self)({
            target: self.facelets[source]
            for target, source in _cube_geometry_move_permutation(move)
        })

    def apply_moves(self, moves: tuple[str, ...] | list[str]) -> "CubeGeometryState":
        state = self
        for move in moves:
            state = state.apply_move(move)
        return state

    def to_json(self) -> dict[str, list[object]]:
        return {
            label: [list(position), int(axis), int(side)]
            for key, label in self.facelets.items()
            for position, axis, side in (key,)
        }


@lru_cache(maxsize=128)
def _cube_geometry_move_permutation(move: str) -> tuple[tuple, ...]:
    """Compute exact move geometry once; state search only needs label permutations."""

    identity = CubeGeometryState.identity()
    source_by_label = {label: key for key, label in identity.facelets.items()}
    moved = CubeState(dict(identity.facelets))
    moved.apply_move(move)
    return tuple((target, source_by_label[label]) for target, label in moved.colors.items())


@lru_cache(maxsize=4096)
def cube_moves_cancel(previous_move: str, move: str) -> bool:
    """Recognize exact inverses, including wide-turn aliases and half turns."""

    previous_sources = dict(_cube_geometry_move_permutation(previous_move))
    return all(
        previous_sources[source] == target
        for target, source in _cube_geometry_move_permutation(move)
    )


def move_sequence_has_immediate_undo(
    previous_move: str | None,
    moves: tuple[str, ...],
) -> bool:
    """Check consecutive committed moves; neutral frames do not reset history."""

    for move in moves:
        if previous_move is not None and cube_moves_cancel(previous_move, move):
            return True
        previous_move = move
    return False


def advance_without_state_revisit(
    state: CubeGeometryState,
    moves: tuple[str, ...],
    visited_states: frozenset[tuple],
) -> tuple[CubeGeometryState, frozenset[tuple]] | None:
    """Reject cycles such as four same-direction turns; holding is allowed."""

    for move in moves:
        state = state.apply_move(move)
        signature = state.signature()
        if signature in visited_states:
            return None
        visited_states = visited_states | {signature}
    return state, visited_states


def cube_rotation_to_notation(rotation: CubeLayerRotation) -> str | None:
    """Convert a completed quarter/half turn to cube notation.

    The move accumulator must not turn an intermediate fitted angle into a
    completed move.  Search angles therefore have to land on an exact
    quarter-turn boundary before they can be emitted.
    """

    angle = float(rotation.angle_degrees)
    quarter_turns = int(round(angle / 90.0))
    if not np.isclose(angle, quarter_turns * 90.0, rtol=0.0, atol=1e-6):
        return None
    quarter_turns %= 4
    if quarter_turns == 0:
        return None
    layer_count = int(rotation.layer_count)
    normal_quarter_turn = -1
    if layer_count >= 3:
        whole_cube_names = {0: "x", 1: "z", 2: "y"}
        base = whole_cube_names.get(int(rotation.axis))
        if base is None:
            return None
        # CubeState's standard whole-cube z rotation uses +Y in this
        # renderer's coordinate frame; x and y use negative X/Z.
        normal_quarter_turn = 1 if int(rotation.axis) == 1 else -1
    else:
        base = FACE_NAME_BY_AXIS_SIDE.get((int(rotation.axis), int(rotation.side)))
        if base is None:
            return None
        if layer_count == 2:
            base += "w"
    if quarter_turns == 2:
        return f"{base}2"
    return base if quarter_turns == normal_quarter_turn % 4 else f"{base}'"


def discrete_rotation_notation(rotation: CubeLayerRotation) -> str | None:
    """Return the discrete state update represented by one layer fit."""

    if int(rotation.layer_count) >= 3:
        return None
    notation = cube_rotation_to_notation(rotation)
    if notation is not None:
        return notation
    angle = float(rotation.angle_degrees)
    if abs(angle) <= 45.0:
        return None
    quarter_turns = max(1, int(np.floor(abs(angle) / 90.0 + 0.5)))
    snapped_angle = float(np.sign(angle) * quarter_turns * 90.0)
    return cube_rotation_to_notation(
        replace(rotation, angle_degrees=snapped_angle)
    )


def completed_candidate_moves(
    candidate: StickerPnPCandidate | object | None,
) -> tuple[str, ...]:
    """Return discrete face/wide state updates represented by a candidate.

    Exact quarter/half turns are emitted directly. Once a fitted layer angle
    passes 45 degrees, the discrete cube model treats it as the nearest
    completed quarter/half turn for the next frame. The candidate itself is
    not rewritten, so diagnostics can still show the raw fitted angle.
    """

    if candidate is None:
        return ()
    explicit_moves = getattr(candidate, "turn_state_moves", None)
    if explicit_moves is not None:
        return tuple(str(move) for move in explicit_moves)
    return tuple(
        notation
        for notation in (
            discrete_rotation_notation(rotation)
            for rotation in getattr(candidate, "layer_rotations", ())
        )
        if notation is not None
    )


def move_lookahead_weights(frame_count: int) -> tuple[float, ...]:
    """Return normalized front-half/back-half weights for a forecast window."""

    if frame_count < 1:
        raise ValueError("Lookahead frame count must be positive")
    if frame_count == 1:
        return (1.0,)
    front_count = frame_count // 2
    back_count = frame_count - front_count
    front_weight = 0.30 / front_count
    back_weight = 0.70 / back_count
    return (front_weight,) * front_count + (back_weight,) * back_count


def move_lookahead_weighted_score(
    current_score: float,
    future_scores: tuple[float, ...] | list[float],
) -> float:
    """Combine one current-frame score with the available future scores.

    The future contribution is normalized over the available frames. An
    unavailable frame is omitted by the caller, while a nonfinite score
    invalidates the path: failing to explain an observed frame is not missing
    data. With no future frames, use only the current-frame score.
    """

    current = float(current_score)
    if not np.isfinite(current):
        return float("inf")
    future = [float(score) for score in future_scores]
    if not all(np.isfinite(score) for score in future):
        return float("inf")
    if not future:
        return current
    scores = (current, *future)
    weights = move_lookahead_weights(len(scores))
    return float(sum(weight * score for weight, score in zip(weights, scores)))


@dataclass(frozen=True)
class MoveLookaheadPath:
    state: CubeGeometryState
    first_trial: dict[str, object]
    future_scores: tuple[float, ...] = ()
    last_move: str | None = None
    future_trials: tuple[dict[str, object], ...] = ()
    move_count: int = 0


def move_lookahead_path_score(path: MoveLookaheadPath) -> float:
    """Combine observation fit with a sparse, whole-sequence move prior."""

    return move_lookahead_weighted_score(
        float(path.first_trial['state_score']), path.future_scores,
    ) + PNP_TURN_LOOKAHEAD_MOVE_COST * path.move_count


def turn_trial_log_payload(trial: dict[str, object]) -> dict[str, object]:
    """Describe a trial without copying its projection arrays into the log."""

    names = (
        'trial_id', 'label', 'is_baseline', 'geometry_option_index', 'layer_axis',
        'layer_side', 'layer_count', 'color_offset', 'state_moves', 'state_score',
        'color_error', 'color_match_count', 'inlier_count', 'layer_inlier_count',
        'keypoint_inlier_count', 'keypoint_layer_inlier_count',
    )
    result = {name: trial[name] for name in names if name in trial}
    for output, source in (
        ('global_mean_error', 'combined_mean_error'), ('global_robust_error', 'robust_error'),
        ('keypoint_mean_error', 'keypoint_error'), ('heatmap_mean_error', 'sticker_error'),
        ('heatmap_match_count', 'sticker_match_count'), ('keypoint_p90_error', 'p90_error'),
    ):
        result[output] = trial.get(source)
    result['layer_rotations'] = [
        dict(axis=rotation.axis, side=rotation.side, angle_degrees=rotation.angle_degrees,
             layer_count=rotation.layer_count)
        for rotation in trial.get('layer_rotations', ())
    ]
    return result


def turn_trial_has_invalid_color(trial: dict[str, object]) -> bool:
    """Missing color is neutral; an available nonfinite color fit is invalid."""

    return 'color_error' in trial and not np.isfinite(float(trial['color_error']))


def search_move_lookahead_paths(
    state_before: CubeGeometryState,
    current_trials: tuple[dict[str, object], ...],
    future_frame_indices: tuple[int, ...],
    trials_for_states: Callable[
        [int, tuple[CubeGeometryState, ...]],
        tuple[tuple[dict[str, object], ...] | None, ...],
    ],
    *,
    state_batch_size: int = PNP_TURN_LOOKAHEAD_STATE_BATCH_SIZE,
    state_change_limit: int | None = PNP_TURN_LOOKAHEAD_STATE_CHANGE_LIMIT,
    previous_move: str | None = None,
    diagnostics: dict[str, object] | None = None,
    previous_states: tuple[tuple, ...] = (),
    frame_score_lower_bounds: tuple[float, ...] | None = None,
    state_evaluation_budget: int | None = None,
    warm_start_moves: tuple[tuple[str, ...] | None, ...] | None = None,
    seed_greedy: bool = False,
    retain_forecast: bool = False,
) -> tuple[MoveLookaheadPath, ...]:
    """Search color-shortlisted state changes with bounds and no beam cutoff.

    Collapse the hold geometries before branching. A complete no-turn rollout
    supplies the initial upper bound. Explore optimistic completions first;
    color orders equally promising children. Merge prefixes by (depth, state),
    and discard one only when dominated or unable to improve the incumbent.
    At every parent keep the best hold plus the top color-ranked state
    changes; None disables that shortlist. Batch size only limits GPU work
    per call. Bounds prove optimality within the shortlisted search space.
    Supplied bounds must be valid for every state. Missing frame bounds
    disable bound pruning across those unknown frames. An optional total
    frame/state budget includes the current state and seed rollouts; a budget
    stop returns only complete forecasts and reports an unresolved bound gap.
    A state-changing trial that returns to this window's starting state is
    ineligible at every depth. Holds remain valid. Warm-start and greedy
    rollouts obey the same rule and per-parent shortlist. No history before
    this window or intermediate-state revisit restriction is imposed.
    """

    if state_batch_size < 1:
        raise ValueError('State batch size must be positive')
    if state_change_limit is not None and state_change_limit < 0:
        raise ValueError('State-change limit must be nonnegative or None')
    count = len(future_frame_indices) + 1
    starting_signature = state_before.signature()
    if state_evaluation_budget is not None and state_evaluation_budget < count:
        raise ValueError('State budget must cover at least one state per search frame')
    if warm_start_moves is not None and len(warm_start_moves) != count:
        raise ValueError('Warm start must provide one move choice per search frame')
    weights = move_lookahead_weights(count)
    bounds = tuple(frame_score_lower_bounds) if frame_score_lower_bounds is not None else (-float('inf'),) * count
    if len(bounds) != count or any(np.isnan(value) for value in bounds):
        raise ValueError('Provide one non-NaN lower bound per search frame')
    # Treat unsupported/nonfinite bounds as unknown, rather than assuming
    # zero: predictive likelihood densities can produce negative NLLs.
    suffix = [0.0] * (count + 1)
    for depth in range(count - 1, -1, -1):
        suffix[depth] = (weights[depth] * bounds[depth] + suffix[depth + 1]
                         if np.isfinite(bounds[depth]) else -float('inf'))
    roots = {}
    if diagnostics is not None:
        roots = {id(trial): dict(trial_id=trial.get('trial_id', f'root-{i}'),
                                trial=turn_trial_log_payload(trial), status='eligible',
                                last_retained_depth=None, pruning_counts={})
                 for i, trial in enumerate(current_trials)}
        diagnostics.update(roots=list(roots.values()), depths=[],
                           frame_score_lower_bounds=list(bounds), incumbent_updates=[],
                           state_change_limit=state_change_limit, prune_root_state_returns=True)
    stats = [dict(depth=d, attempted_trials=0, invalid_score=0, unavailable=0,
                  immediate_undo=0, state_revisit=0, root_return=0, input_paths=0, scored_states=0,
                  expanded_paths=0, merged_paths=0, collapsed_holds=0,
                  bound_pruned_paths=0, color_pruned_paths=0, invalid_color=0,
                  unreliable_color_states=0, completed_paths=0, queued_paths=0)
             for d in range(count)]
    trial_cache: list[dict[tuple, tuple[dict[str, object], ...] | None]] = [{} for _ in range(count)]
    trial_cache[0][state_before.signature()] = current_trials
    stats[0]['scored_states'] = 1
    seen: list[dict[tuple, tuple]] = [{} for _ in range(count)]
    completions: dict[int, MoveLookaheadPath] = {}
    incumbent: MoveLookaheadPath | None = None
    queue: list[tuple] = []
    serial = 0
    evaluated_states = 1  # Current-frame trials are supplied by the caller.
    budget_exhausted = False
    pending_lower_bound = float('inf')
    seed_results = []

    class BudgetExhausted(Exception):
        pass

    def prefix_rank(path: MoveLookaheadPath) -> tuple:
        scores = (float(path.first_trial['state_score']), *path.future_scores)
        value = sum(weight * score for weight, score in zip(weights, scores))
        return value + PNP_TURN_LOOKAHEAD_MOVE_COST * path.move_count, path.first_trial['rank']

    def record(path: MoveLookaheadPath, reason: str, depth: int) -> None:
        stats[depth][reason] += 1
        if diagnostics is not None:
            root = roots[id(path.first_trial)]
            root['pruning_counts'][reason] = root['pruning_counts'].get(reason, 0) + 1
            root['eliminated_depth'] = depth
            if reason == 'bound_pruned_paths':
                examples = root.setdefault('pruning_examples', [])
                if len(examples) < 3:
                    prefix = prefix_rank(path)[0]
                    examples.append(dict(depth=depth, prefix_cost=prefix,
                                         remaining_lower_bound=suffix[depth + 1],
                                         lower_bound=prefix + suffix[depth + 1],
                                         incumbent_score=prefix_rank(incumbent)[0] if incumbent is not None else None))

    def complete(path: MoveLookaheadPath) -> None:
        nonlocal incumbent
        key = id(path.first_trial)
        previous = completions.get(key)
        if previous is None or prefix_rank(path) < prefix_rank(previous):
            completions[key] = path
        stats[-1]['completed_paths'] += 1
        if incumbent is None or prefix_rank(path) < prefix_rank(incumbent):
            incumbent = path
            if diagnostics is not None:
                diagnostics['incumbent_updates'].append(dict(
                    score=move_lookahead_path_score(path), move_count=path.move_count,
                    trial_id=path.first_trial.get('trial_id')))

    def evaluate(depth: int, states: tuple[CubeGeometryState, ...]):
        nonlocal evaluated_states, budget_exhausted
        cache = trial_cache[depth]
        missing = {state.signature(): state for state in states if state.signature() not in cache}
        if missing:
            if state_evaluation_budget is not None and evaluated_states + len(missing) > state_evaluation_budget:
                budget_exhausted = True
                raise BudgetExhausted
            evaluated_states += len(missing)
            values = trials_for_states(future_frame_indices[depth - 1], tuple(missing.values()))
            cache.update(zip(missing, values, strict=True))
            stats[depth]['scored_states'] += len(missing)
        return tuple(cache[state.signature()] for state in states)

    def hold(trials):
        return min((trial for trial in trials or ()
                    if not trial['state_moves'] and np.isfinite(float(trial['state_score']))
                    and not turn_trial_has_invalid_color(trial)),
                   key=lambda trial: trial['rank'], default=None)

    # Establish a complete solution before examining alternative states.
    first_hold = hold(current_trials)
    no_turn = None
    if first_hold is not None:
        no_turn = MoveLookaheadPath(state_before, first_hold, last_move=previous_move)
        for depth in range(1, count):
            trial = hold(evaluate(depth, (state_before,))[0])
            if trial is None:
                no_turn = None
                break
            no_turn = replace(no_turn, future_scores=(*no_turn.future_scores, float(trial['state_score'])),
                              future_trials=(*no_turn.future_trials, trial) if diagnostics is not None or retain_forecast else ())
        if no_turn is not None:
            complete(no_turn)
    baseline = min((trial for trial in current_trials if trial.get('is_baseline')
                    and np.isfinite(float(trial['state_score']))
                    and not turn_trial_has_invalid_color(trial)), key=lambda trial: trial['rank'], default=None)
    baseline_comparison = None
    if no_turn is not None and baseline is not None:
        baseline_comparison = no_turn if baseline is first_hold else replace(no_turn, first_trial=baseline)
    if baseline_comparison is not None and baseline_comparison is not no_turn:
        complete(baseline_comparison)

    def rollout(moves_by_depth, name):
        """Re-score a complete seed within the same per-parent shortlist."""
        before = state_before
        path = None
        repaired = 0
        root_return_pruned = 0
        for depth in range(count):
            trials = current_trials if depth == 0 else evaluate(depth, (before,))[0]
            unique = {}
            for trial in trials or ():
                if not np.isfinite(float(trial['state_score'])) or turn_trial_has_invalid_color(trial):
                    continue
                after = before.apply_moves(tuple(trial['state_moves']))
                key = after.signature()
                if trial['state_moves'] and key == starting_signature:
                    root_return_pruned += 1
                    continue
                edge = weights[depth] * float(trial['state_score']) + PNP_TURN_LOOKAHEAD_MOVE_COST * len(trial['state_moves'])
                previous = unique.get(key)
                if previous is None or (edge, trial['rank']) < previous[0]:
                    unique[key] = ((edge, trial['rank']), trial, after)
            holds = [item for item in unique.values() if not item[1]['state_moves']]
            changes = sorted((item for item in unique.values() if item[1]['state_moves']),
                             key=lambda item: (float(item[1].get('color_error', item[1]['state_score'])), item[1]['rank']))
            choices = holds + (changes if state_change_limit is None else changes[:state_change_limit])
            desired = moves_by_depth[depth]
            matching = [item for item in choices if desired is not None and tuple(item[1]['state_moves']) == desired]
            if desired is not None and not matching:
                repaired += 1
            choice = min(matching or choices, key=lambda item: item[0], default=None)
            if choice is None:
                seed_results.append(dict(source=name, status='no_legal_continuation', depth=depth,
                                         root_return_pruned=root_return_pruned))
                return
            _rank, trial, after = choice
            moves = tuple(trial['state_moves'])
            path = (MoveLookaheadPath(after, trial, last_move=moves[-1] if moves else previous_move,
                                      move_count=len(moves)) if path is None else
                    MoveLookaheadPath(after, path.first_trial, (*path.future_scores, float(trial['state_score'])),
                                      moves[-1] if moves else path.last_move,
                                      (*path.future_trials, trial), path.move_count + len(moves)))
            before = after
        complete(path)
        seed_results.append(dict(source=name, status='complete', score=move_lookahead_path_score(path),
                                 repaired_steps=repaired, root_return_pruned=root_return_pruned))

    # A valid lower bound even when a seed spends the budget before the queue
    # is initialized. Once exploration starts, use the tighter pending frontier.
    initial_lower_bound = min((weights[0] * float(trial['state_score'])
                               + PNP_TURN_LOOKAHEAD_MOVE_COST * len(trial['state_moves'])
                               + suffix[1] for trial in current_trials
                               if np.isfinite(float(trial['state_score']))
                               and not turn_trial_has_invalid_color(trial)), default=float('inf'))
    try:
        if warm_start_moves is not None:
            rollout(warm_start_moves, 'previous_forecast')
        if seed_greedy:
            rollout((None,) * count, 'greedy')
    except BudgetExhausted:
        pending_lower_bound = initial_lower_bound
        seed_results.append(dict(source='seeding', status='budget_exhausted'))

    def offer(path: MoveLookaheadPath, depth: int, color: float) -> None:
        nonlocal serial
        stats[depth]['expanded_paths'] += 1
        if depth == count - 1:
            complete(path)
            return
        signature = path.state.signature()
        rank = prefix_rank(path)
        previous = seen[depth].get(signature)
        if previous is not None and previous <= rank:
            record(path, 'merged_paths', depth)
            return
        seen[depth][signature] = rank
        optimistic = rank[0] + suffix[depth + 1]
        if incumbent is not None and optimistic > prefix_rank(incumbent)[0]:
            record(path, 'bound_pruned_paths', depth)
            return
        serial += 1
        heapq.heappush(queue, (optimistic, color, -depth, serial, path))
        stats[depth]['queued_paths'] += 1
        if diagnostics is not None:
            roots[id(path.first_trial)]['last_retained_depth'] = depth

    def expand(parent: MoveLookaheadPath | None, trials, depth: int) -> None:
        before = state_before if parent is None else parent.state
        stats[depth]['input_paths'] += 1
        if trials is None:
            if parent is not None:
                record(parent, 'unavailable', depth)
            return
        # At most one hold and 24 changes for the standard 12 geometries.
        # Deduplicate any equivalent move aliases as well, using the actual
        # whole-window edge cost rather than the color ordering heuristic.
        choices: dict[tuple, tuple[dict[str, object], CubeGeometryState]] = {}
        hold_count = 0
        invalid_color_count = 0
        for trial in trials:
            stats[depth]['attempted_trials'] += 1
            score = float(trial['state_score'])
            if not np.isfinite(score):
                if parent is None:
                    stats[depth]['invalid_score'] += 1
                else:
                    record(parent, 'invalid_score', depth)
                if parent is None and diagnostics is not None:
                    roots[id(trial)].update(status='invalid_score', eliminated_depth=0)
                continue
            if turn_trial_has_invalid_color(trial):
                invalid_color_count += 1
                if parent is None:
                    stats[depth]['invalid_color'] += 1
                    if diagnostics is not None:
                        roots[id(trial)].update(status='invalid_color', eliminated_depth=depth)
                else:
                    record(parent, 'invalid_color', depth)
                continue
            moves = tuple(trial['state_moves'])
            hold_count += not bool(moves)
            after = before.apply_moves(moves)
            key = after.signature()
            if moves and key == starting_signature:
                if parent is None:
                    stats[depth]['root_return'] += 1
                    if diagnostics is not None:
                        roots[id(trial)].update(status='root_return', eliminated_depth=depth)
                else:
                    record(parent, 'root_return', depth)
                    if diagnostics is not None:
                        examples = roots[id(parent.first_trial)].setdefault('root_return_examples', [])
                        if len(examples) < 3:
                            examples.append(dict(depth=depth, state_moves=moves, trial_id=trial.get('trial_id')))
                continue
            previous = choices.get(key)
            edge_rank = (weights[depth] * score + PNP_TURN_LOOKAHEAD_MOVE_COST * len(moves), trial['rank'])
            if previous is None or edge_rank < (
                weights[depth] * float(previous[0]['state_score'])
                + PNP_TURN_LOOKAHEAD_MOVE_COST * len(previous[0]['state_moves']), previous[0]['rank'],
            ):
                choices[key] = (trial, after)
        if not choices and invalid_color_count:
            stats[depth]['unreliable_color_states'] += 1
        stats[depth]['collapsed_holds'] += max(0, hold_count - 1)
        if parent is None and diagnostics is not None:
            chosen = {id(trial) for trial, _ in choices.values()}
            for trial in trials:
                if id(trial) not in chosen and roots[id(trial)]['status'] == 'eligible':
                    roots[id(trial)].update(status='merged', eliminated_depth=0)
        ordered = sorted(choices.values(), key=lambda item: (
            bool(item[0]['state_moves']), float(item[0].get('color_error', item[0]['state_score'])), item[0]['rank']))
        if state_change_limit is not None:
            holds = [item for item in ordered if not item[0]['state_moves']]
            changes = [item for item in ordered if item[0]['state_moves']]
            for trial, _after in changes[state_change_limit:]:
                if parent is None:
                    stats[depth]['color_pruned_paths'] += 1
                    if diagnostics is not None:
                        roots[id(trial)].update(status='color_pruned', eliminated_depth=depth)
                else:
                    record(parent, 'color_pruned_paths', depth)
            ordered = holds + changes[:state_change_limit]
        for trial, after in ordered:
            moves = tuple(trial['state_moves'])
            path = (MoveLookaheadPath(after, trial, last_move=moves[-1] if moves else previous_move,
                                      move_count=len(moves)) if parent is None else
                    MoveLookaheadPath(after, parent.first_trial,
                                      (*parent.future_scores, float(trial['state_score'])),
                                      moves[-1] if moves else parent.last_move,
                                      (*parent.future_trials, trial) if diagnostics is not None or retain_forecast else (),
                                      parent.move_count + len(moves)))
            offer(path, depth, float(trial.get('color_error', trial['state_score'])))

    try:
        expand(None, current_trials, 0)
        peak_queue = len(queue)
        while queue and not budget_exhausted:
            batch = []
            while queue and len(batch) < state_batch_size:
                optimistic, _color, _neg_depth, _serial, path = heapq.heappop(queue)
                depth = len(path.future_scores)
                if seen[depth].get(path.state.signature()) != prefix_rank(path):
                    record(path, 'merged_paths', depth)
                    continue
                if incumbent is not None and optimistic > prefix_rank(incumbent)[0]:
                    record(path, 'bound_pruned_paths', depth)
                    continue
                batch.append(path)
            pending_lower_bound = min((prefix_rank(path)[0] + suffix[len(path.future_scores) + 1]
                                       for path in batch), default=float('inf'))
            by_depth: dict[int, list[MoveLookaheadPath]] = {}
            for path in batch:
                by_depth.setdefault(len(path.future_scores) + 1, []).append(path)
            for depth, parents in by_depth.items():
                viable = []
                for parent in parents:
                    if incumbent is not None and prefix_rank(parent)[0] + suffix[depth] > prefix_rank(incumbent)[0]:
                        record(parent, 'bound_pruned_paths', depth - 1)
                    else:
                        viable.append(parent)
                parents = viable
                if not parents:
                    continue
                values = evaluate(depth, tuple(path.state for path in parents))
                for parent, trials in zip(parents, values, strict=True):
                    # An earlier completion in this batch may tighten the bound.
                    if incumbent is not None and prefix_rank(parent)[0] + suffix[depth] > prefix_rank(incumbent)[0]:
                        record(parent, 'bound_pruned_paths', depth - 1)
                        continue
                    expand(parent, trials, depth)
            peak_queue = max(peak_queue, len(queue))

    except BudgetExhausted:
        pass

    result = sorted(completions.values(), key=prefix_rank)
    for comparison in (no_turn, baseline_comparison):
        if comparison is not None and not any(
            path.first_trial is comparison.first_trial and path.future_scores == comparison.future_scores
            and path.move_count == comparison.move_count for path in result
        ):
            result.append(comparison)
    if diagnostics is not None:
        shortlisted = any(item['color_pruned_paths'] for item in stats)
        incumbent_score = move_lookahead_path_score(incumbent) if incumbent is not None else None
        lower_bound = (min(pending_lower_bound, *(item[0] for item in queue),
                           incumbent_score if incumbent_score is not None else float('inf'))
                       if budget_exhausted else incumbent_score)
        diagnostics.update(optimality=('budget_exhausted' if incumbent is not None else 'budget_exhausted_no_path')
                           if budget_exhausted else
                           ('proven_within_shortlist' if shortlisted else 'proven') if incumbent is not None else 'no_finite_path',
                           termination_reason='state_budget' if budget_exhausted else 'search_complete',
                           state_evaluation_budget=state_evaluation_budget, evaluated_states=evaluated_states,
                           budget_exhausted=budget_exhausted, seeds=seed_results,
                           search_space='color_shortlist' if shortlisted else 'all_state_changes',
                           best_lower_bound=lower_bound,
                           optimality_gap=max(0.0, incumbent_score - lower_bound)
                           if incumbent_score is not None and lower_bound is not None else None,
                           peak_queued_paths=peak_queue,
                           initial_no_turn_score=move_lookahead_path_score(no_turn) if no_turn is not None else None)
        for path in result:
            root = roots[id(path.first_trial)]
            root.update(status='complete', last_retained_depth=count - 1)
            root.pop('eliminated_depth', None)
            root['best_sequence_score'] = min(root.get('best_sequence_score', float('inf')), move_lookahead_path_score(path))
        for root in roots.values():
            if root['status'] == 'eligible':
                reasons = root['pruning_counts']
                root.update(status='budget_unfinished' if budget_exhausted else
                            'bound_pruned' if reasons.get('bound_pruned_paths') else
                            'merged' if reasons.get('merged_paths') else
                            'root_return' if reasons.get('root_return') else
                            'invalid_color' if reasons.get('invalid_color') else 'no_legal_continuation')
        for depth, item in enumerate(stats):
            item.update(distinct_states=len(trial_cache[depth]),
                        remaining_lower_bound=suffix[depth + 1],
                        beam_pruned_paths=0, retained_paths=0,
                        baseline_extra=False, no_turn_extra=False)
        diagnostics['depths'] = stats
    return tuple(result)


def move_lookahead_path_log(path: MoveLookaheadPath, frames: tuple[int, ...]) -> dict[str, object]:
    """Record the forecast as steps, without retaining full cube-state copies."""

    trials = (path.first_trial, *path.future_trials)
    future_count = len(path.future_scores)
    weights = move_lookahead_weights(future_count + 1) if future_count else (1.0,)
    steps = []
    for depth, (frame, trial) in enumerate(zip(frames, trials, strict=True)):
        weight = weights[depth]
        steps.append(dict(frame_index=frame, depth=depth, **turn_trial_log_payload(trial),
                          weight=weight, weighted_fit_score=weight * float(trial['state_score']),
                          move_cost=PNP_TURN_LOOKAHEAD_MOVE_COST * len(trial['state_moves']),
                          weighted_score=weight * float(trial['state_score'])
                          + PNP_TURN_LOOKAHEAD_MOVE_COST * len(trial['state_moves'])))
    return dict(trial_id=path.first_trial.get('trial_id'),
                total_score=move_lookahead_path_score(path),
                observation_score=move_lookahead_weighted_score(float(path.first_trial['state_score']), path.future_scores),
                move_cost=PNP_TURN_LOOKAHEAD_MOVE_COST * path.move_count,
                move_count=path.move_count,
                current_score=float(path.first_trial['state_score']),
                future_mean=float(np.mean(path.future_scores)) if future_count else None,
                steps=steps)

@dataclass
class MoveAccumulator:
    """Keep the discrete fitted moves visible while frames are replayed."""

    moves: list[str] = field(default_factory=list)
    current_fit: str = "none"
    displacement_mm: float | None = None
    _last_emitted_signature: tuple[str, ...] | None = None
    _last_core_tvec: np.ndarray | None = None
    _state: CubeState | None = None
    _observation_index: int = 0
    _recent_states: list[tuple[int, tuple]] = field(default_factory=list)
    _geometry_state: CubeGeometryState = field(
        default_factory=CubeGeometryState.identity
    )

    def recent_geometry_states(self) -> tuple[tuple, ...]:
        """States departed within the lookback window before the next frame."""

        return tuple(signature for index, signature in self._recent_states
                     if self._observation_index + 1 - index <= PNP_TURN_LOOKAHEAD_FRAMES)

    def _initialize_state(self, sticker_color_names: dict[str, str] | None) -> None:
        if self._state is not None or not sticker_color_names:
            return
        identities = canonical_sticker_identities()
        if any(identity.label not in sticker_color_names for identity in identities):
            return
        self._state = CubeState(
            {
                (identity.cubie_index, identity.axis, identity.side): sticker_color_names[
                    identity.label
                ]
                for identity in identities
            }
        )

    def current_color_names(
        self,
        sticker_color_names: dict[str, str] | None = None,
    ) -> dict[str, str] | None:
        """Return colors for the currently committed cube state.

        Initialize from ``sticker_color_names`` before the current candidate
        is observed, because that candidate still carries the pre-turn
        geometry state while its layer rotation is rendered.
        """

        self._initialize_state(sticker_color_names)

        if self._state is None:
            return None
        return {
            identity.label: self._state.color_at(
                self._geometry_state.position_for(identity.label)
            )
            for identity in canonical_sticker_identities()
        }

    def observe(
        self,
        candidate: StickerPnPCandidate | None,
        *,
        displacement_candidate: StickerPnPCandidate | None = None,
        sticker_color_names: dict[str, str] | None = None,
    ) -> None:
        self._observation_index += 1
        self._recent_states[:] = [entry for entry in self._recent_states
                                 if self._observation_index - entry[0] <= PNP_TURN_LOOKAHEAD_FRAMES]
        self._initialize_state(sticker_color_names)
        if candidate is None:
            self.current_fit = "rejected" if displacement_candidate is not None else "none"
            candidate_for_displacement = displacement_candidate
        else:
            self.current_fit = candidate_turn_debug_label(candidate)
            candidate_for_displacement = candidate
        if candidate_for_displacement is None:
            self.displacement_mm = None
            return
        current_tvec = np.asarray(
            candidate_for_displacement.tvec,
            dtype=np.float64,
        ).reshape(3)
        if self._last_core_tvec is None:
            self.displacement_mm = 0.0
        else:
            self.displacement_mm = float(
                np.linalg.norm(current_tvec - self._last_core_tvec)
            )
        if candidate is not None:
            self._last_core_tvec = current_tvec.copy()
        if candidate is None:
            return
        candidate_geometry = getattr(candidate, "cube_geometry_state", None)
        if candidate_geometry is not None:
            if self._geometry_state.signature() != candidate_geometry.signature():
                # A candidate produced from a different committed state must
                # never mutate the model or its displayed move history.
                self.current_fit = "state-mismatch"
                return
        notations = completed_candidate_moves(candidate)
        if not notations:
            return
        signature = notations
        # The turn fit is an estimate of the current cube state.  Do not emit
        # a duplicate just because an intervening frame was neutral/rejected.
        # Stateful candidates are generated from the committed model.  A
        # repeated notation can therefore be a real second turn; suppressing
        # it by text alone would desynchronize the display model from fitting.
        # Legacy lightweight candidates retain the old signature de-duplication
        # behavior used by the standalone accumulator tests.
        if candidate_geometry is not None or signature != self._last_emitted_signature:
            self.moves.extend(notations)
            self._last_emitted_signature = signature
            for notation in notations:
                self._recent_states.append((self._observation_index, self._geometry_state.signature()))
                self._geometry_state = self._geometry_state.apply_move(notation)
            if self._state is not None:
                for notation in notations:
                    self._state.apply_move(notation)


def draw_move_accumulator_overlay(
    image: np.ndarray,
    accumulator: MoveAccumulator,
) -> np.ndarray:
    """Draw the accumulated fitted moves and current fit in the output."""

    history = " ".join(accumulator.moves[-24:]) or "-"
    current = accumulator.current_fit
    displacement = (
        "--"
        if accumulator.displacement_mm is None
        else f"{accumulator.displacement_mm:.1f}mm"
    )
    label = f"moves {history} | fit {current} | core displacement {displacement}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
    x = max(8, image.shape[1] - text_width - 8)
    y = image.shape[0] - 10
    cv2.rectangle(
        image,
        (x - 4, y - text_height - 4),
        (min(image.shape[1] - 1, x + text_width + 4), y + baseline + 2),
        (0, 0, 0),
        -1,
    )
    cv2.putText(image, label, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return image


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


def turn_search_angles() -> np.ndarray:
    """Return the inclusive exhaustive turn-angle grid."""

    return np.arange(
        PNP_LAYER_MISALIGNMENT_MIN_DEGREES,
        PNP_LAYER_MISALIGNMENT_MAX_DEGREES
        + PNP_LAYER_MISALIGNMENT_STEP_DEGREES,
        PNP_LAYER_MISALIGNMENT_STEP_DEGREES,
        dtype=np.float64,
    )


def turn_geometric_search_angles() -> np.ndarray:
    """Return the partial-turn grid used by the stateful move search."""

    # Keep the endpoints requested by the state rollout and explicitly add
    # zero because a six-degree grid starting at -45 does not land on it.
    return np.unique(
        np.concatenate(
            (
                np.arange(-45.0, 45.0 + 6.0, 6.0, dtype=np.float64),
                np.asarray((0.0,), dtype=np.float64),
            )
        )
    )


def candidate_turn_debug_label(candidate: StickerPnPCandidate | None) -> str:
    """Return the displayed label for a fitted turn."""

    return layer_rotation_debug_label(
        tuple() if candidate is None else candidate.layer_rotations
    )


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


def retained_sticker_mask_indices(masks: np.ndarray) -> list[int]:
    """Keep nonempty masks unless larger retained masks cover most of them.

    Coverage is measured against the union of all larger masks already
    retained, so several partial detections can collectively suppress a
    duplicate. Process largest first and break equal-area ties by original
    detection order. Return surviving indices in original order to preserve
    alignment with segmentation class IDs and downstream sticker references.
    """

    binary = np.asarray(masks, dtype=bool)
    if binary.ndim != 3 or not len(binary):
        return []
    areas = np.count_nonzero(binary, axis=(1, 2))
    retained: list[int] = []
    retained_union = np.zeros(binary.shape[1:], dtype=bool)
    for index in sorted(range(len(binary)), key=lambda item: (-int(areas[item]), item)):
        area = int(areas[index])
        if area == 0:
            continue
        covered_area = np.count_nonzero(binary[index] & retained_union)
        if covered_area >= STICKER_MASK_CONTAINMENT_THRESHOLD * area:
            continue
        retained.append(index)
        retained_union |= binary[index]
    return sorted(retained)


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
    selected_masks = np.asarray(mask_data[keep]) >= 0.5
    keep = [keep[index] for index in retained_sticker_mask_indices(selected_masks)]
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


def _piecewise_rgb_profile_distances(
    points_rgb: np.ndarray,
    polyline_rgb: np.ndarray,
) -> np.ndarray:
    """Return profile distances in RGB coordinates scaled to 0..1."""

    return _piecewise_profile_distances(
        np.asarray(points_rgb, dtype=np.float64).reshape(-1, 3) / 255.0,
        np.asarray(polyline_rgb, dtype=np.float64).reshape(-1, 3) / 255.0,
    )


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Return a finite weighted median for one-dimensional samples."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return float(np.median(values)) if len(values) else 0.0
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    return float(values[np.searchsorted(np.cumsum(weights), 0.5 * np.sum(weights))])


def _fit_rgb_inflection_point(
    rgb_points: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Fit the RGB point where the first segment reaches one channel at 255."""

    points = np.asarray(rgb_points, dtype=np.float64).reshape(-1, 3)
    if weights is None:
        point_weights = np.ones(len(points), dtype=np.float64)
    else:
        point_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(point_weights) != len(points):
            raise ValueError("RGB point weights must match the point count.")
    valid = (
        np.all(np.isfinite(points), axis=1)
        & np.isfinite(point_weights)
        & (point_weights > 0.0)
    )
    points = points[valid]
    point_weights = point_weights[valid]
    if len(points) == 0 or float(np.sum(point_weights)) <= 1e-12:
        return np.full(3, 255.0, dtype=np.float64)
    black = np.zeros(3, dtype=np.float64)
    white = np.full(3, 255.0, dtype=np.float64)
    candidates: list[tuple[int, np.ndarray]] = []
    dominant_channels = np.argmax(points, axis=1)
    for channel in range(3):
        channel_mask = dominant_channels == channel
        channel_points = points[channel_mask]
        channel_weights = point_weights[channel_mask]
        channel_points = channel_points[channel_points[:, channel] > 1e-6]
        channel_weights = channel_weights[points[channel_mask][:, channel] > 1e-6]
        if len(channel_points) == 0:
            continue
        scaled = channel_points * (255.0 / channel_points[:, channel, None])
        scaled = np.clip(scaled, 0.0, 255.0)
        for pivot in scaled[:: max(1, len(scaled) // 8)]:
            pivot = pivot.copy()
            pivot[channel] = 255.0
            candidates.append((channel, pivot))
        median_pivot = np.asarray(
            [
                _weighted_median(scaled[:, dimension], channel_weights)
                for dimension in range(3)
            ],
            dtype=np.float64,
        )
        median_pivot[channel] = 255.0
        candidates.append((channel, median_pivot))
    if not candidates:
        return white

    def score(pivot: np.ndarray) -> float:
        polyline = np.asarray((black, pivot, white), dtype=np.float64)
        distances = _piecewise_profile_distances(points, polyline) ** 2
        return float(np.average(distances, weights=point_weights))

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


def fit_rgb_color_profile(
    rgb_points: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    sample_count: int | None = None,
) -> StickerColorProfile | None:
    """Fit the constrained two-segment model to RGB observations.

    ``weights`` is used by the global-profile estimator. The ordinary
    per-sticker fit remains unchanged when weights are omitted.
    """

    rgb_points = np.asarray(rgb_points, dtype=np.float64).reshape(-1, 3)
    if len(rgb_points) == 0:
        return None
    if weights is None:
        point_weights = np.ones(len(rgb_points), dtype=np.float64)
    else:
        point_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(point_weights) != len(rgb_points):
            raise ValueError("RGB point weights must match the point count.")
    point_weights = np.where(np.isfinite(point_weights), point_weights, 0.0)
    point_weights = np.clip(point_weights, 0.0, None)
    if float(np.sum(point_weights)) <= 1e-12:
        return None
    inflection = _fit_rgb_inflection_point(rgb_points, point_weights)
    polyline_rgb = np.asarray(
        ((0.0, 0.0, 0.0), inflection, (255.0, 255.0, 255.0)),
        dtype=np.float32,
    )
    polyline_oklab = rgb_to_oklab(polyline_rgb).astype(np.float32)
    residual = float(
        np.sqrt(
            np.average(
                _piecewise_rgb_profile_distances(rgb_points, polyline_rgb) ** 2,
                weights=point_weights,
            )
        )
    )
    return StickerColorProfile(
        polyline_rgb,
        polyline_oklab,
        residual,
        len(rgb_points) if sample_count is None else int(sample_count),
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


def _global_profile_distance_matrix(
    rgb_points: np.ndarray,
    profiles: dict[str, StickerColorProfile],
) -> np.ndarray:
    """Return one RGB profile distance per sample and global color."""

    points_rgb = np.asarray(rgb_points, dtype=np.float64).reshape(-1, 3)
    return np.column_stack(
        [
            _piecewise_rgb_profile_distances(
                points_rgb,
                profiles[color_name].polyline_rgb,
            )
            for color_name in GLOBAL_COLOR_NAMES
        ]
    )


def _global_profile_soft_responsibilities(distances: np.ndarray) -> np.ndarray:
    """Convert profile distances into normalized soft color responsibilities."""

    distances = np.asarray(distances, dtype=np.float64)
    scaled = -distances / max(GLOBAL_PROFILE_EM_TEMPERATURE, 1e-6)
    scaled -= np.max(scaled, axis=1, keepdims=True)
    responsibilities = np.exp(scaled)
    return responsibilities / np.maximum(
        np.sum(responsibilities, axis=1, keepdims=True),
        1e-12,
    )


def _global_profile_information_weights(
    rgb_points: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    """Weight samples by how much reliable hue information they contain.

    Samples near black or white, and samples with little chroma, cannot
    reliably distinguish red from orange. They remain available to the later
    posterior pass, but should not define the shared profile hue.
    """

    rgb = np.asarray(rgb_points, dtype=np.float64).reshape(-1, 3) / 255.0
    brightness = np.mean(rgb, axis=1)
    chroma = (np.max(rgb, axis=1) - np.min(rgb, axis=1)) / np.maximum(
        np.sum(rgb, axis=1),
        1.0 / 255.0,
    )
    midtone = np.clip(4.0 * brightness * (1.0 - brightness), 0.0, 1.0)
    ordered = np.sort(np.asarray(distances, dtype=np.float64), axis=1)
    margin = np.maximum(ordered[:, 1] - ordered[:, 0], 0.0)
    separation = 1.0 - np.exp(
        -margin / max(GLOBAL_PROFILE_EM_TEMPERATURE, 1e-6)
    )
    chroma_weight = np.clip(chroma / 0.08, 0.0, 1.0)
    return chroma_weight * (0.25 + 0.75 * midtone) * separation


def build_global_color_profiles_from_samples(
    entries: list[tuple[str, int, np.ndarray]],
    source_weights: list[float] | np.ndarray | None = None,
) -> tuple[
    dict[str, GlobalColorProfile],
    dict[str, list[tuple[str, int, np.ndarray]]],
]:
    """Build regularized global profiles from pose-free RGB samples.

    The old implementation assigned every observation to the nearest pure
    prior before fitting. That made a red/orange mistake self-reinforcing. The
    estimator below starts from the same priors, gives every sample a soft
    responsibility for every color, and refits each trajectory using those
    responsibilities. A small pure-color anchor keeps the named profiles
    from swapping or collapsing when the video is short or unbalanced.
    """

    prior_profiles = {
        color_name: pure_rgb_color_profile(color_name)
        for color_name in PURE_COLOR_RGB
    }
    entries = [
        (label, int(frame_index), np.asarray(rgb, dtype=np.uint8).reshape(3).copy())
        for label, frame_index, rgb in entries
    ]
    trace_weights = (
        np.ones(len(entries), dtype=np.float64)
        if source_weights is None
        else np.asarray(source_weights, dtype=np.float64).reshape(-1)
    )
    if len(trace_weights) != len(entries):
        raise ValueError("Global color sample weights must match the sample count.")
    profiles: dict[str, GlobalColorProfile] = {
        color_name: GlobalColorProfile(
            color_name=color_name,
            prior=prior_profiles[color_name],
            fitted=None,
            sample_count=0,
        )
        for color_name in GLOBAL_COLOR_NAMES
    }
    if not entries:
        return profiles, {color_name: [] for color_name in GLOBAL_COLOR_NAMES}

    values = np.asarray([entry[2] for entry in entries], dtype=np.float64).reshape(-1, 3)
    source_weights = np.asarray(trace_weights, dtype=np.float64)
    for _iteration in range(GLOBAL_PROFILE_EM_ITERATIONS):
        references = {
            color_name: _global_profile_reference(profiles[color_name])
            for color_name in GLOBAL_COLOR_NAMES
        }
        distances = _global_profile_distance_matrix(
            values,
            {
                color_name: reference
                for color_name, reference in references.items()
                if reference is not None
            },
        )
        responsibilities = _global_profile_soft_responsibilities(distances)
        information_weights = _global_profile_information_weights(values, distances)
        next_profiles: dict[str, GlobalColorProfile] = {}
        for color_index, color_name in enumerate(GLOBAL_COLOR_NAMES):
            data_weights = (
                source_weights
                * information_weights
                * responsibilities[:, color_index] ** 2
            )
            effective_weight = float(np.sum(data_weights))
            fitted = None
            if effective_weight >= GLOBAL_PROFILE_MIN_EFFECTIVE_WEIGHT:
                prior_point = np.asarray(
                    [PURE_COLOR_RGB[color_name]],
                    dtype=np.float64,
                )
                fit_values = np.vstack((values, prior_point))
                fit_weights = np.concatenate(
                    (
                        data_weights,
                        np.asarray((GLOBAL_PROFILE_PRIOR_WEIGHT,), dtype=np.float64),
                    )
                )
                fitted = fit_rgb_color_profile(
                    fit_values,
                    fit_weights,
                    sample_count=int(np.count_nonzero(data_weights > 1e-6)),
                )
            next_profiles[color_name] = GlobalColorProfile(
                color_name=color_name,
                prior=prior_profiles[color_name],
                fitted=fitted,
                sample_count=0,
            )
        profiles = next_profiles

    final_references = {
        color_name: _global_profile_reference(profiles[color_name])
        for color_name in GLOBAL_COLOR_NAMES
    }
    final_distances = _global_profile_distance_matrix(
        values,
        {
            color_name: reference
            for color_name, reference in final_references.items()
            if reference is not None
        },
    )
    grouped: dict[str, list[tuple[str, int, np.ndarray]]] = {
        color_name: [] for color_name in GLOBAL_COLOR_NAMES
    }
    for entry, color_index in zip(
        entries,
        np.argmin(final_distances, axis=1),
        strict=True,
    ):
        grouped[GLOBAL_COLOR_NAMES[int(color_index)]].append(entry)
    profiles = {
        color_name: replace(
            profiles[color_name],
            sample_count=len(grouped[color_name]),
        )
        for color_name in GLOBAL_COLOR_NAMES
    }
    return profiles, grouped


def build_global_color_profiles(
    trace: StickerColorTrace,
) -> tuple[
    dict[str, GlobalColorProfile],
    dict[str, list[tuple[str, int, np.ndarray]]],
]:
    """Build global profiles from identity-associated samples."""

    entries: list[tuple[str, int, np.ndarray]] = []
    trace_weights: list[float] = []
    for identity in trace.identities:
        for observation_index, (frame_index, rgb) in enumerate(
            trace.observations.get(identity.label, ())
        ):
            entries.append((identity.label, int(frame_index), rgb))
            trace_weights.append(
                _trace_observation_weight(
                    trace,
                    identity.label,
                    observation_index,
                )
            )
    return build_global_color_profiles_from_samples(entries, trace_weights)


def refine_global_color_profiles_from_initial_state(
    trace: StickerColorTrace,
    known_initial_colors: dict[str, str],
    profiles: dict[str, GlobalColorProfile],
    first_turn_frame: int | None,
) -> tuple[dict[str, GlobalColorProfile], dict[str, list[tuple[str, int, np.ndarray]]]]:
    """Fit labeled initial-state samples, retaining the original fit if sparse."""
    grouped = {color: [] for color in GLOBAL_COLOR_NAMES}
    weights = {color: [] for color in GLOBAL_COLOR_NAMES}
    for identity in trace.identities:
        color = known_initial_colors.get(identity.label)
        if color not in grouped:
            continue
        for index, (frame, rgb) in enumerate(trace.observations.get(identity.label, ())):
            if first_turn_frame is not None and frame >= first_turn_frame:
                continue
            grouped[color].append((identity.label, frame, rgb))
            weights[color].append(_trace_observation_weight(trace, identity.label, index))

    refined = dict(profiles)
    for color in GLOBAL_COLOR_NAMES:
        data_weights = np.asarray(weights[color], dtype=np.float64)
        if np.sum(data_weights) < GLOBAL_PROFILE_MIN_EFFECTIVE_WEIGHT:
            continue
        values = np.asarray([entry[2] for entry in grouped[color]], dtype=np.float64)
        fitted = fit_rgb_color_profile(
            np.vstack((values, np.asarray(PURE_COLOR_RGB[color], dtype=np.float64))),
            np.append(data_weights, GLOBAL_PROFILE_PRIOR_WEIGHT),
            sample_count=len(values),
        )
        if fitted is not None:
            valid = np.flatnonzero(np.isfinite(data_weights) & (data_weights > 0.0))
            if len(valid) > LABELED_SAMPLE_MAX_POINTS:
                valid = valid[np.linspace(0, len(valid) - 1, LABELED_SAMPLE_MAX_POINTS, dtype=int)]
            refined[color] = replace(
                profiles[color], fitted=fitted, sample_count=len(values),
                labeled_samples=tuple(tuple(float(component) for component in values[index]) for index in valid),
                labeled_sample_weights=tuple(float(data_weights[index]) for index in valid),
            )
    return refined, grouped


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

    point_rgb = np.asarray(rgb, dtype=np.float64).reshape(1, 3)
    distances = np.asarray(
        [
            _piecewise_rgb_profile_distances(
                point_rgb,
                _global_profile_reference(global_profiles[color_name]).polyline_rgb,
            )[0]
            for color_name in GLOBAL_COLOR_NAMES
        ],
        dtype=np.float64,
    )
    for index, color_name in enumerate(GLOBAL_COLOR_NAMES):
        profile = global_profiles[color_name]
        if not profile.labeled_samples:
            continue
        samples = np.asarray(profile.labeled_samples, dtype=np.float64).reshape(-1, 3)
        weights = np.asarray(profile.labeled_sample_weights, dtype=np.float64)
        sample_distances = np.linalg.norm((samples - point_rgb) / 255.0, axis=1)
        count = min(LABELED_SAMPLE_NEIGHBORS, len(sample_distances))
        nearest = np.argpartition(sample_distances, count - 1)[:count]
        neighbor_weights = weights[nearest]
        total_weight = float(np.sum(neighbor_weights))
        if total_weight <= 1e-12:
            continue
        sample_distance = float(np.average(sample_distances[nearest], weights=neighbor_weights))
        coverage = min(total_weight / LABELED_SAMPLE_NEIGHBORS, 1.0) * np.exp(
            -(sample_distance / LABELED_SAMPLE_COVERAGE_SCALE) ** 2
        )
        blend = LABELED_SAMPLE_DISTANCE_WEIGHT * coverage
        distances[index] = (1.0 - blend) * distances[index] + blend * sample_distance
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


def build_sticker_global_profile_likelihoods(
    trace: StickerColorTrace,
    global_profiles: dict[str, GlobalColorProfile] | None = None,
) -> dict[int, dict[str, dict[str, float]]]:
    """Build independent per-frame probabilities from global profiles only.

    Each observation is normalized independently. There is deliberately no
    temporal prior, trajectory grouping, or cube-inventory constraint, so the
    probabilities match the nearest-global-profile debug classification.
    """

    if global_profiles is None:
        global_profiles, _global_observations = build_global_color_profiles(trace)
    likelihoods_by_frame: dict[int, dict[str, dict[str, float]]] = {}
    for identity in trace.identities:
        for frame_index, rgb in trace.observations.get(identity.label, ()):
            likelihoods_by_frame.setdefault(int(frame_index), {})[identity.label] = (
                rgb_observation_color_likelihoods(rgb, global_profiles)
            )
    return likelihoods_by_frame


def classify_sticker_colors_by_frame_global_profile(
    trace: StickerColorTrace,
    global_profiles: dict[str, GlobalColorProfile],
) -> dict[int, dict[str, str]]:
    """Assign each observed sticker to its nearest global profile per frame.

    Unlike posterior replay, this intentionally has no temporal prior or
    trajectory grouping: every RGB sample is classified independently using
    only its distance to the fitted global profiles.
    """

    names_by_frame: dict[int, dict[str, str]] = {}
    profile_references = {
        color_name: _global_profile_reference(global_profiles[color_name])
        for color_name in GLOBAL_COLOR_NAMES
    }
    for identity in trace.identities:
        for frame_index, rgb in trace.observations.get(identity.label, ()):
            point_distances = _global_profile_distance_matrix(
                np.asarray([rgb], dtype=np.float64),
                {
                    color_name: reference
                    for color_name, reference in profile_references.items()
                    if reference is not None
                },
            )[0]
            closest_color = GLOBAL_COLOR_NAMES[int(np.argmin(point_distances))]
            names_by_frame.setdefault(int(frame_index), {})[identity.label] = (
                closest_color
            )
    return names_by_frame


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
    expected_color_by_label: dict[str, str] | None = None,
) -> tuple[float, int, int, dict[str, np.ndarray], dict[str, float]]:
    """Return weighted likelihood scores, observations, and their weights.

    When ``expected_color_by_label`` is supplied, score each candidate against
    the known color at that canonical sticker location instead of a learned
    per-sticker prior.  This is used only to anchor the initial orientation
    from a known scramble.
    """

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
        if expected_color_by_label is None:
            predictive_probability = float(np.dot(prior, weighted_evidence))
        else:
            expected_color = expected_color_by_label.get(target_label)
            if expected_color not in GLOBAL_COLOR_NAMES:
                raise ValueError(
                    f"Unknown expected color {expected_color!r} for {target_label}"
                )
            expected_index = GLOBAL_COLOR_NAMES.index(expected_color)
            predictive_probability = float(
                weighted_evidence[expected_index]
                / max(float(np.sum(weighted_evidence)), 1e-12)
            )
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
    record_all_turn_candidates: bool = False,
    turn_candidate_log: Path | None = None,
    global_profiles: dict[str, GlobalColorProfile] | None = None,
    sticker_mask_erosion: int = 0,
    known_initial_colors: dict[str, str] | None = None,
    pose_prepared: bool = False,
    require_orientation: bool = False,
    color_samples_per_sticker: int = 128,
    color_min_samples_per_sticker: int = 4,
    calibrate_initial_colors: bool = False,
    calibration_observations: dict[str, list[tuple[str, int, np.ndarray]]] | None = None,
) -> StickerColorTrace:
    """Recover cached pose candidates and optionally fit layer turns.

    The neural pass and its initial pose/color samples are already complete
    when this runs. When ``recover_orientation`` is false, this is the
    geometry-only pose pass: it refines cached candidates and preserves the
    collected color observations without building profiles, scoring colors, or
    classifying identities. When ``recover_orientation`` is true, it applies
    the separate color-based symmetry recovery to an already-fitted trace.
    When move recognition is enabled, each frame first caches geometric
    partial-turn options. Color offsets then drive a bounded ten-frame state
    search with adaptive branch-and-bound; static pose refinement remains keypoint-first.
    """

    if (recover_orientation or recognize_moves) and global_profiles is None:
        global_profiles, _global_observations = build_global_color_profiles(trace)
    identities = trace.identities
    if recover_orientation:
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
    else:
        permutations = ()
        identity_permutation = None
        identity_symmetry_index = -1
        priors = {}
    turn_color_priors = {
        identity.label: (
            np.eye(len(GLOBAL_COLOR_NAMES), dtype=np.float64)[
                GLOBAL_COLOR_NAMES.index(known_initial_colors[identity.label])
            ]
            if known_initial_colors is not None
            and known_initial_colors.get(identity.label) in GLOBAL_COLOR_NAMES
            else np.full(
                len(GLOBAL_COLOR_NAMES),
                1.0 / len(GLOBAL_COLOR_NAMES),
                dtype=np.float64,
            )
        )
        for identity in identities
    }
    raw_by_frame = _trace_observations_by_frame(trace)
    raw_weights_by_frame = _trace_observation_weights_by_frame(trace)
    corrected_trace = StickerColorTrace.create()
    initial_calibration_trace = StickerColorTrace.create()
    calibration_done = False
    accepted_initial_pose = False

    def calibrate_initial_profiles(first_move_frame: int | None = None) -> None:
        nonlocal calibration_done
        if not calibrate_initial_colors or calibration_done:
            return
        boundary = (
            f"first committed move at frame {first_move_frame}"
            if first_move_frame is not None else "end of input without a lasting committed move"
        )
        if not accepted_initial_pose:
            print(f"Color calibration stopped: {boundary}; no stationary samples available.", flush=True)
            calibration_done = True
            return
        refined, grouped = refine_global_color_profiles_from_initial_state(
            initial_calibration_trace, known_initial_colors or {}, global_profiles, None,
        )
        global_profiles.update(refined)
        # Future frames may already have been scored with the old profiles.
        lookahead_color_trial_cache.clear()
        lookahead_frame_lower_bound_cache.clear()
        if calibration_observations is not None and any(grouped.values()):
            calibration_observations.clear()
            calibration_observations.update(grouped)
        sample_count = sum(len(entries) for entries in grouped.values())
        print(f"Color calibration stopped: {boundary}; {sample_count} labeled samples collected.", flush=True)
        calibration_done = True
    previous_continuity_candidate: StickerPnPCandidate | None = None
    continuity_symmetry_tracker = CubeSymmetryTracker()
    cube_geometry_state = CubeGeometryState.identity()
    last_committed_move: str | None = None
    committed_state_history: list[tuple[int, tuple]] = []
    pending_calibration_move: tuple[int, tuple] | None = None
    locked_orientation_symmetry: np.ndarray | None = None
    locked_orientation_permutation: dict[str, str] | None = None
    lookahead_frame_lower_bound_cache: dict[int, float] = {}
    lookahead_geometry_cache: dict[
        int,
        tuple[StickerPnPCandidate, dict[str, object]] | None,
    ] = {}
    lookahead_color_trial_cache: dict[
        tuple[int, tuple],
        tuple[StickerPnPCandidate, tuple[dict[str, object], ...]] | None,
    ] = {}
    lookahead_scoring_device = _turn_torch_scoring_device("mps")
    lookahead_forecast: dict[int, tuple[str, ...]] = {}
    lookahead_forecast_expected_state: tuple | None = None

    def prepare_lookahead_geometry(
        cache_index: int,
        candidate_override: StickerPnPCandidate | None = None,
    ) -> tuple[StickerPnPCandidate, dict[str, object]] | None:
        """Prepare one frame's geometry once, independent of cube state."""

        if cache_index in lookahead_geometry_cache:
            cached = lookahead_geometry_cache[cache_index]
            if candidate_override is None:
                return cached
            if cached is not None:
                return candidate_override, cached[1]
        if cache_index < 0 or cache_index >= total_records:
            return None
        record = cache[cache_index]
        frame_candidate = candidate_override
        if frame_candidate is None:
            frame_candidate = getattr(record, "pose_candidate", None)
            if not isinstance(frame_candidate, StickerPnPCandidate):
                lookahead_geometry_cache[cache_index] = None
                return None
            if recover_orientation:
                if locked_orientation_symmetry is None:
                    return None
                frame_candidate = transform_candidate_by_symmetry(
                    frame_candidate,
                    locked_orientation_symmetry,
                )
        if not isinstance(frame_candidate, StickerPnPCandidate):
            return None
        if (
            not getattr(frame_candidate, "sticker_mask_colors", ())
            and getattr(record, "sticker_masks", None) is not None
            and getattr(record, "roi_xyxy", None) is not None
        ):
            mask_color_entries = sticker_mask_rgb_observations(
                record.frame,
                record.roi_xyxy,
                record.sticker_masks,
                sticker_mask_erosion,
            )
            frame_candidate = replace(
                frame_candidate,
                sticker_mask_colors=tuple(
                    rgb for _mask_index, rgb in mask_color_entries
                ),
            )
        if lookahead_scoring_device is None:
            return None
        frame_candidate = replace(
            frame_candidate,
            layer_rotations=(),
            sticker_layer_rotations=None,
            turn_state_moves=None,
        )
        assignment_error_limit = (
            PNP_FINAL_ASSIGNMENT_ERROR_FRACTION
            * projected_cube_diameter(frame_candidate)
        )
        geometry_frame = _build_turn_geometry_frame(
            frame_candidate,
            assignment_error_limit,
            lookahead_scoring_device,
        )
        if geometry_frame is None:
            result = None
        else:
            result = (frame_candidate, geometry_frame)
        lookahead_geometry_cache[cache_index] = result
        return result

    def color_trials_for_state(
        cache_index: int,
        geometry_state: CubeGeometryState,
        candidate_override: StickerPnPCandidate | None = None,
    ) -> tuple[StickerPnPCandidate, tuple[dict[str, object], ...]] | None:
        """Return every color-offset option for one frame/state pair."""

        prepared = prepare_lookahead_geometry(cache_index, candidate_override)
        if prepared is None or lookahead_scoring_device is None:
            return None
        frame_candidate, geometry_frame = prepared
        cache_key = (cache_index, geometry_state.signature())
        if candidate_override is None and cache_key in lookahead_color_trial_cache:
            return lookahead_color_trial_cache[cache_key]
        state_candidate = replace(
            frame_candidate,
            cube_geometry_state=geometry_state.copy(),
            layer_rotations=(),
            sticker_layer_rotations=None,
            turn_state_moves=None,
        )
        color_likelihoods = turn_color_predictive_likelihoods(
            state_candidate,
            turn_color_priors,
            global_profiles,
            None,
        )
        if color_likelihoods is None:
            color_likelihoods = getattr(
                state_candidate,
                "turn_color_likelihoods",
                None,
            )
        assignment_error_limit = (
            PNP_FINAL_ASSIGNMENT_ERROR_FRACTION
            * projected_cube_diameter(state_candidate)
        )
        trials = _score_turn_color_offsets_on_gpu(
            state_candidate,
            tuple(geometry_frame["options"]),
            assignment_error_limit,
            lookahead_scoring_device,
            color_likelihoods,
            projection_cache=geometry_frame.setdefault("color_projection_cache", {}),
        )
        if not trials:
            result = None
        else:
            # Every scored option participates in the rollout.  There is no
            # inlier-count, color-rescue, or static-baseline acceptance gate;
            # the lowest lookahead score is the decision criterion.
            result = (state_candidate, tuple(trials))
        if candidate_override is None:
            lookahead_color_trial_cache[cache_key] = result
        return result

    def color_trials_for_states(
        cache_index: int,
        states: tuple[CubeGeometryState, ...],
    ) -> tuple[tuple[dict[str, object], ...] | None, ...]:
        """Score uncached frontier states together using one GPU batch."""

        prepared = prepare_lookahead_geometry(cache_index)
        if prepared is None:
            return (None,) * len(states)
        frame_candidate, geometry_frame = prepared
        missing = tuple(
            state for state in states
            if (cache_index, state.signature()) not in lookahead_color_trial_cache
        )
        if missing:
            color_likelihoods = turn_color_predictive_likelihoods(
                frame_candidate, turn_color_priors, global_profiles, None,
            )
            if color_likelihoods is None:
                color_likelihoods = getattr(frame_candidate, "turn_color_likelihoods", None)
            trials_by_state = _score_turn_color_offsets_for_states_on_gpu(
                frame_candidate, tuple(geometry_frame["options"]),
                PNP_FINAL_ASSIGNMENT_ERROR_FRACTION * projected_cube_diameter(frame_candidate),
                lookahead_scoring_device, color_likelihoods, missing,
                projection_cache=geometry_frame.setdefault("color_projection_cache", {}),
            )
            for index, state in enumerate(missing):
                trials = trials_by_state[index] if trials_by_state is not None else None
                lookahead_color_trial_cache[(cache_index, state.signature())] = (
                    (frame_candidate, tuple(trials)) if trials else None
                )
        results = []
        for state in states:
            cached = lookahead_color_trial_cache[(cache_index, state.signature())]
            results.append(cached[1] if cached is not None else None)
        return tuple(results)

    def frame_score_lower_bound(cache_index: int) -> float:
        if cache_index not in lookahead_frame_lower_bound_cache:
            prepared = prepare_lookahead_geometry(cache_index)
            if prepared is None or lookahead_scoring_device is None:
                return -float('inf')
            frame_candidate, geometry_frame = prepared
            likelihoods = turn_color_predictive_likelihoods(
                frame_candidate, turn_color_priors, global_profiles, None,
            )
            if likelihoods is None:
                likelihoods = getattr(frame_candidate, 'turn_color_likelihoods', None)
            lookahead_frame_lower_bound_cache[cache_index] = _turn_color_frame_score_lower_bound(
                frame_candidate, tuple(geometry_frame['options']),
                PNP_FINAL_ASSIGNMENT_ERROR_FRACTION * projected_cube_diameter(frame_candidate),
                lookahead_scoring_device, likelihoods,
                projection_cache=geometry_frame.setdefault("color_projection_cache", {}),
            )
        return lookahead_frame_lower_bound_cache[cache_index]

    def apply_move_lookahead(
        selected: StickerPnPCandidate,
        static_candidate: StickerPnPCandidate,
        state_before: CubeGeometryState,
        current_record_index: int,
    ) -> StickerPnPCandidate:
        """Choose a transition with cached, warm-started, budgeted lookahead."""

        nonlocal lookahead_forecast, lookahead_forecast_expected_state
        fallback = selected
        previous_states = tuple(signature for _index, signature in committed_state_history)
        started = perf_counter()
        search_debug = (
            dict(version=11, algorithm="adaptive_branch_and_bound", status="not_started",
                 previous_move=last_committed_move, beam_width=None,
                 state_batch_size=PNP_TURN_LOOKAHEAD_STATE_BATCH_SIZE,
                 state_change_limit=PNP_TURN_LOOKAHEAD_STATE_CHANGE_LIMIT,
                 recent_committed_states=len(previous_states), prune_state_revisits=False,
                 prune_root_state_returns=True,
                 prune_immediate_undo=False, merge_key="cube_state", preserve_no_turn_path=True,
                 move_cost_per_turn=PNP_TURN_LOOKAHEAD_MOVE_COST,
                 requested_future_frames=PNP_TURN_LOOKAHEAD_FRAMES,
                 weights=dict(schedule="front_back_uniform", front=0.30, back=0.70),
                 fallback_undo_blocked=False)
            if turn_candidate_log is not None else None
        )

        def fallback_with_diagnostics(reason: str) -> StickerPnPCandidate:
            # Failed/nonfinite fits cannot authorize an unfiltered local move.
            known_bad_color = (bool(getattr(selected, 'turn_fit_evaluated', False))
                               and not np.isfinite(selected.turn_fit_color_score))
            unreliable = reason == 'no_surviving_path' or known_bad_color or selected.turn_color_unreliable
            result = _unreliable_color_hold(selected, static_candidate) if unreliable else fallback
            if search_debug is not None:
                search_debug.update(status="fallback", fallback_reason=reason,
                                    color_evidence_unreliable=bool(unreliable),
                                    elapsed_ms=1000.0 * (perf_counter() - started))
            return replace(result, turn_search_diagnostics=search_debug)
        current_cache_index = current_record_index - 1
        evaluation = color_trials_for_state(
            current_cache_index,
            state_before,
            static_candidate,
        )
        if evaluation is None or not evaluation[1]:
            return fallback_with_diagnostics("current_frame_unavailable")
        _current_candidate, current_trials = evaluation
        score_frontier = color_trials_for_states
        if search_debug is not None:
            geometry = lookahead_geometry_cache[current_cache_index][1]
            search_debug["geometry"] = dict(
                tested_configurations=len(geometry.get("trials", ())),
                retained_options=len(geometry.get("options", ())),
                options=[turn_trial_log_payload(trial) for trial in geometry.get("options", ())],
            )
            evaluation_stats = dict(future_state_requests=0, cache_hits=0, states_scored=0, gpu_batches=0)
            search_debug["evaluation_cache"] = evaluation_stats

            def score_frontier(index: int, states: tuple[CubeGeometryState, ...]):
                hits = sum((index, state.signature()) in lookahead_color_trial_cache for state in states)
                evaluation_stats["future_state_requests"] += len(states)
                evaluation_stats["cache_hits"] += hits
                evaluation_stats["states_scored"] += len(states) - hits
                evaluation_stats["gpu_batches"] += int(hits < len(states))
                return color_trials_for_states(index, states)
        current_has_accepted_trial = any(
            np.isfinite(float(trial['state_score'])) and not turn_trial_has_invalid_color(trial)
            for trial in current_trials
        )
        future_indices = tuple(
            index for index in range(
                current_cache_index + 1,
                min(total_records, current_cache_index + PNP_TURN_LOOKAHEAD_FRAMES + 1),
            )
            if prepare_lookahead_geometry(index) is not None
        ) if current_has_accepted_trial else ()
        frames = (cache[current_cache_index].frame_index, *(cache[index].frame_index for index in future_indices))
        if search_debug is not None:
            search_debug.update(
                frames=list(frames),
                skipped_frames=[
                    cache[index].frame_index for index in range(
                        current_cache_index + 1,
                        min(total_records, current_cache_index + PNP_TURN_LOOKAHEAD_FRAMES + 1),
                    ) if index not in future_indices
                ],
                weights=dict(
                    schedule="front_back_uniform",
                    front=0.30,
                    back=0.70,
                    normalized=list(move_lookahead_weights(len(frames))),
                ),
            )
        reused_forecast = lookahead_forecast_expected_state == state_before.signature()
        run_debug = search_debug if search_debug is not None else {}
        seed_moves = tuple(lookahead_forecast.get(index) for index in (current_cache_index, *future_indices)) if reused_forecast else None
        search_paths = search_move_lookahead_paths(
            state_before, current_trials, future_indices, score_frontier,
            previous_move=last_committed_move, diagnostics=run_debug,
            state_evaluation_budget=PNP_TURN_LOOKAHEAD_STATE_BUDGET,
            warm_start_moves=seed_moves, seed_greedy=True, retain_forecast=True,
            previous_states=previous_states,
            frame_score_lower_bounds=(-float('inf'), *(frame_score_lower_bound(index) for index in future_indices)),
        )
        frames = (cache[current_cache_index].frame_index, *(cache[index].frame_index for index in future_indices))
        if search_debug is not None:
            projection_caches = [lookahead_geometry_cache[index][1].get('color_projection_cache', {})
                                 for index in (current_cache_index, *future_indices)]
            search_debug.update(
                state_evaluation_budget=PNP_TURN_LOOKAHEAD_STATE_BUDGET,
                decision_evaluated_states=run_debug['evaluated_states'],
                reused_previous_forecast=reused_forecast,
                projection_cache=dict(builds=sum(item.get('builds', 0) for item in projection_caches),
                                      hits=sum(item.get('hits', 0) for item in projection_caches),
                                      assignment_builds=sum(item.get('assignments', {}).get('builds', 0) for item in projection_caches),
                                      assignment_hits=sum(item.get('assignments', {}).get('hits', 0) for item in projection_caches)),
                frames=list(frames),
                weights=dict(schedule='front_back_uniform', front=0.30, back=0.70,
                             normalized=list(move_lookahead_weights(len(frames)))),
            )
            for depth in search_debug['depths']:
                depth['frame_index'] = frames[depth['depth']]
        paths = [
            (move_lookahead_path_score(path), path.first_trial, path.future_scores)
            for path in search_paths
        ]
        if not paths:
            lookahead_forecast = {}
            lookahead_forecast_expected_state = None
            return fallback_with_diagnostics('state_budget_no_complete_path' if run_debug.get('budget_exhausted') else 'no_surviving_path')
        winner = min(search_paths, key=lambda path: (move_lookahead_path_score(path), path.first_trial['rank']))
        lookahead_forecast = {index: tuple(trial['state_moves'])
                              for index, trial in zip((current_cache_index, *future_indices),
                                                      (winner.first_trial, *winner.future_trials), strict=True)}
        lookahead_forecast_expected_state = state_before.apply_moves(tuple(winner.first_trial['state_moves'])).signature()
        chosen_score, chosen_trial, chosen_future_scores = min(
            paths,
            key=lambda path: (path[0], path[1]["rank"]),
        )
        static_paths = [
            path for path in paths if bool(path[1]["is_baseline"])
        ]
        static_path = min(
            static_paths,
            key=lambda path: (path[0], path[1]["rank"]),
            default=None,
        )
        annotated = _candidate_with_turn_trial(
            selected,
            static_candidate,
            chosen_trial,
        )
        if search_debug is not None:
            ordered_paths = sorted(search_paths, key=lambda path: (
                move_lookahead_path_score(path),
                path.first_trial["rank"],
            ))
            best_by_root = {}
            for path in ordered_paths:
                best_by_root.setdefault(id(path.first_trial), path)
            finalists = list(best_by_root.values())[:8]
            best_static = next((path for path in ordered_paths if path.first_trial["is_baseline"]), None)
            if best_static is not None and all(path is not best_static for path in finalists):
                finalists.append(best_static)
            eligible = [trial for trial in current_trials if np.isfinite(float(trial["state_score"]))
                        and not turn_trial_has_invalid_color(trial)]
            no_turn_path = next((path for path in ordered_paths if path.move_count == 0), None)
            local_best = min(eligible, key=lambda trial: (
                float(trial["state_score"]) + PNP_TURN_LOOKAHEAD_MOVE_COST * len(trial["state_moves"]),
                trial["rank"],
            ), default=None)
            search_debug.update(
                status="searched",
                selected_trial_id=chosen_trial.get("trial_id"),
                local_best_trial_id=local_best.get("trial_id") if local_best is not None else None,
                changed_local_choice=local_best is not chosen_trial,
                selected_path=move_lookahead_path_log(ordered_paths[0], frames),
                no_turn_path=move_lookahead_path_log(no_turn_path, frames) if no_turn_path is not None else None,
                improvement_over_no_turn=(move_lookahead_path_score(no_turn_path) - chosen_score) if no_turn_path is not None else None,
                finalists=[move_lookahead_path_log(path, frames) for path in finalists],
                elapsed_ms=1000.0 * (perf_counter() - started),
            )
        return replace(
            annotated,
            turn_search_diagnostics=search_debug,
            turn_fit_lookahead_action_score=float(chosen_score),
            turn_fit_lookahead_static_score=(
                float(static_path[0]) if static_path is not None else float("inf")
            ),
            turn_fit_lookahead_action_future_scores=tuple(chosen_future_scores),
            turn_fit_lookahead_static_future_scores=(
                tuple(static_path[2]) if static_path is not None else ()
            ),
            # Static is a valid winning option, not a rejected action.  Keep
            # the legacy diagnostic field false now that no gate is applied.
            turn_fit_lookahead_rejected=False,
        )

    # Choose the initial offset before replaying any turns. This also applies
    # the same offset to earlier frames with too few visible color samples.
    if pose_prepared and recover_orientation and known_initial_colors is not None:
        for record in cache:
            observations = raw_by_frame.get(record.frame_index, {})
            weights = raw_weights_by_frame.get(record.frame_index, {})
            if not isinstance(record.pose_candidate, StickerPnPCandidate) or len(observations) < POSE_SYMMETRY_RECOVERY_MIN_OBSERVATIONS:
                continue
            scores = []
            for index, (symmetry, permutation) in enumerate(permutations):
                score, count, _, _, _ = _score_symmetry_observations(
                    observations, permutation, priors, global_profiles, weights,
                    expected_color_by_label=known_initial_colors,
                )
                if count == len(observations):
                    scores.append((score, index == identity_symmetry_index, index))
            if scores:
                best_score = max(item[0] for item in scores)
                tied = [item for item in scores if best_score - item[0] <= 1e-9]
                _, _, index = max(tied, key=lambda item: item[1])
                locked_orientation_symmetry = np.asarray(permutations[index][0], dtype=np.float64).copy()
                locked_orientation_permutation = dict(permutations[index][1])
                break
        if require_orientation and locked_orientation_symmetry is None and any(
            isinstance(record.pose_candidate, StickerPnPCandidate) for record in cache
        ):
            raise RuntimeError("Cannot recognize moves: too few color observations to determine the initial cube orientation")

    progress_label = (
        "pose refinement and continuity" if refine_pose
        else "orientation matching" if recover_orientation
        else "move search" if recognize_moves
        else "pose recovery"
    )
    total_records = len(cache)
    for record_index, record in enumerate(cache, start=1):
        committed_state_history[:] = [
            entry for entry in committed_state_history
            if record_index - 1 - entry[0] <= PNP_TURN_LOOKAHEAD_FRAMES
        ]
        # Adaptive search visits many states. Retain only frames that a future
        # decision can reuse, rather than growing these caches with clip length.
        for past_index in tuple(lookahead_geometry_cache):
            if past_index < record_index - 1:
                del lookahead_geometry_cache[past_index]
                lookahead_frame_lower_bound_cache.pop(past_index, None)
        for key in tuple(lookahead_color_trial_cache):
            if key[0] < record_index - 1:
                del lookahead_color_trial_cache[key]
        if (pending_calibration_move is not None
                and record.frame_index - pending_calibration_move[0] > INITIAL_CALIBRATION_MOVE_PATIENCE_FRAMES):
            calibrate_initial_profiles(pending_calibration_move[0])
            pending_calibration_move = None
        if progress:
            print_headless_progress(
                progress_label,
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
        if refine_pose:
            local_masks = (
                getattr(record, "sticker_masks")[0]
                if getattr(record, "sticker_masks", None) is not None
                else None
            )
            mask_geometry = sticker_mask_pose_geometry(
                local_masks,
                getattr(record, "roi_xyxy", None),
                sticker_mask_erosion,
            )
            candidate_hypotheses = tuple(
                restore_sticker_mask_pose_geometry(
                    hypothesis,
                    local_masks,
                    getattr(record, "roi_xyxy", None),
                    sticker_mask_erosion,
                    geometry=mask_geometry,
                )
                for hypothesis in candidate_hypotheses
            )
            candidate = candidate_hypotheses[selected_candidate_index]

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
                        record_all_turn_candidates=record_all_turn_candidates,
                    )
                )
            return tuple(refined_hypotheses)

        if recover_orientation:
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
                    expected_color_by_label=(
                        known_initial_colors
                        if known_initial_colors is not None
                        else None
                    ),
                )
            )
        else:
            current_log_likelihood = float("-inf")
            observation_count = len(raw_observations)
            low_probability_count = 0
            current_observations = {
                label: np.asarray(rgb, dtype=np.uint8).copy()
                for label, rgb in raw_observations.items()
            }
            current_observation_weights = {
                label: float(np.clip(weight, 0.0, 1.0))
                for label, weight in raw_observation_weights.items()
            }
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
        if refine_pose:
            selected_hypotheses = refine_hypotheses(selected_hypotheses)
            selected_candidate = selected_hypotheses[selected_candidate_index]

        if pose_prepared:
            motion_rejected = False
            continuity_reference_candidate = selected_candidate
        else:
            # Preserve the existing symmetry and refinement order. Continuity is
            # applied only after the best pose has been selected and refined.
            continuity_candidate = select_pose_hypothesis(
                selected_hypotheses, None,
                (previous_continuity_candidate,) if previous_continuity_candidate is not None else (),
            )
            if continuity_candidate is not None:
                selected_candidate = continuity_candidate
                selected_candidate_index = next(
                    index for index, hypothesis in enumerate(selected_hypotheses)
                    if hypothesis is continuity_candidate
                )
            next_candidate = cache[record_index].pose_candidate if record_index < total_records else None
            motion_rejected = False
            if (POSE_TWO_SIDED_MOTION_REJECTION_ENABLED
                    and previous_continuity_candidate is not None
                    and isinstance(next_candidate, StickerPnPCandidate)):
                current = np.asarray(selected_candidate.tvec, dtype=np.float64).reshape(3)
                previous = np.asarray(previous_continuity_candidate.tvec, dtype=np.float64).reshape(3)
                following = np.asarray(next_candidate.tvec, dtype=np.float64).reshape(3)
                motion_rejected = (
                    np.linalg.norm(current - previous) > POSE_MOTION_DISPLACEMENT_TOLERANCE_MM
                    and np.linalg.norm(following - current) > POSE_MOTION_DISPLACEMENT_TOLERANCE_MM
                )

            # Keep the pose in one continuous cube coordinate frame before color
            # recovery changes that frame.  The cached path must use the same
            # symmetry canonicalization as the live path: the PnP roots can be
            # physically equivalent while assigning the cube axes differently.
            # Color recovery is then applied to this already-continuous pose and
            # its fixed mapping is reused on later frames.
            selected_candidate = canonicalize_pose_symmetry(
                selected_candidate,
                continuity_symmetry_tracker,
            )
            selected_hypotheses = tuple(
                selected_candidate if index == selected_candidate_index else hypothesis
                for index, hypothesis in enumerate(selected_hypotheses)
            )
            # This is the continuity state for the next frame. Keep it before
            # color orientation recovery and move fitting so neither display
            # operation can change root matching on the next frame.
            continuity_reference_candidate = selected_candidate

        if refine_pose and not motion_rejected and getattr(record, "sticker_masks", None) is not None and getattr(record, "roi_xyxy", None) is not None:
            selected_observations = sample_projected_sticker_colors(
                record.frame, record.roi_xyxy, record.sticker_masks, selected_candidate,
                max_samples_per_sticker=color_samples_per_sticker,
                min_samples_per_sticker=color_min_samples_per_sticker,
                sticker_mask_erosion=sticker_mask_erosion,
            )
            rotation, _ = cv2.Rodrigues(np.asarray(selected_candidate.rvec, dtype=np.float64).reshape(3, 1))
            selected_observation_weights = {
                identity.label: sticker_view_reliability_weight(selected_candidate, identity, rotation)
                for identity in identities if identity.label in selected_observations
            }

        # Orientation recovery is deliberately the final pose operation. The
        # candidate root, temporal continuity choice, motion rejection, and
        # displacement checks must finish before cube symmetries are
        # applied to the accepted pose. Color matching establishes this
        # whole-cube frame once; later frames reuse the same center-to-axis
        # mapping and never re-rank the 24 symmetries.
        scored = []
        if (
            recover_orientation
            and locked_orientation_symmetry is None
            and observation_count >= POSE_SYMMETRY_RECOVERY_MIN_OBSERVATIONS
        ):
            for symmetry_index, (symmetry, permutation) in enumerate(permutations):
                (
                    log_likelihood,
                    count,
                    _low_count,
                    remapped_observations,
                    remapped_observation_weights,
                ) = _score_symmetry_observations(
                    raw_observations,
                    permutation,
                    priors,
                    global_profiles,
                    raw_observation_weights,
                    expected_color_by_label=(
                        known_initial_colors
                        if known_initial_colors is not None
                        else None
                    ),
                )
                if count != observation_count:
                    continue
                transformed_candidate = transform_candidate_by_symmetry(
                    selected_candidate,
                    symmetry,
                )
                continuity_distance = (
                    pose_hypothesis_distance(
                        transformed_candidate,
                        (
                            previous_continuity_candidate.rvec,
                            previous_continuity_candidate.tvec,
                        ),
                    )
                    if previous_continuity_candidate is not None
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
                # Candidate/root continuity has already been resolved above.
                # Color matching chooses the initial whole-cube frame exactly
                # once; it is never re-ranked on later frames.
                best_log_likelihood = max(item[1] for item in scored)
                color_ties = [
                    item
                    for item in scored
                    if best_log_likelihood - item[1] <= 1e-9
                ]
                (
                    best_symmetry_index,
                    best_symmetry_log_likelihood,
                    _best_continuity_score,
                    best_candidate,
                    best_observations,
                    best_observation_weights,
                ) = max(
                    color_ties,
                    key=lambda item: int(item[0] == identity_symmetry_index),
                )
                locked_orientation_symmetry = np.asarray(
                    permutations[best_symmetry_index][0],
                    dtype=np.float64,
                ).copy()
                locked_orientation_permutation = dict(
                    permutations[best_symmetry_index][1]
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
                    ) in sorted(
                        scored,
                        key=lambda item: (-item[2], -item[1]),
                    ):
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
                selected_log_likelihood = best_symmetry_log_likelihood
                selected_hypotheses = tuple(
                    selected_candidate if index == selected_candidate_index else hypothesis
                    for index, hypothesis in enumerate(selected_hypotheses)
                )
        elif recover_orientation and locked_orientation_symmetry is not None:
            # Keep the fixed orientation even when this frame has too few
            # observations to participate in color matching. The permutation
            # is reused directly, so no color likelihood is evaluated again.
            selected_candidate = transform_candidate_by_symmetry(
                selected_candidate,
                locked_orientation_symmetry,
            )
            selected_observations = {
                target_label: np.asarray(raw_observations[source_label], dtype=np.uint8).copy()
                for target_label, source_label in (
                    locked_orientation_permutation or {}
                ).items()
                if source_label in raw_observations
            }
            selected_observation_weights = {
                target_label: float(
                    np.clip(raw_observation_weights.get(source_label, 1.0), 0.0, 1.0)
                )
                for target_label, source_label in (
                    locked_orientation_permutation or {}
                ).items()
                if source_label in raw_observations
            }
            selected_hypotheses = tuple(
                selected_candidate if index == selected_candidate_index else hypothesis
                for index, hypothesis in enumerate(selected_hypotheses)
            )

        if recognize_moves:
            # Turn fitting is incremental: the candidate's persistent sticker
            # identities must be projected from the last committed cube model
            # before this frame's move is tested.
            selected_candidate = replace(
                selected_candidate,
                cube_geometry_state=cube_geometry_state.copy(),
            )
            if (
                not getattr(selected_candidate, "sticker_mask_colors", ())
                and getattr(record, "sticker_masks", None) is not None
            ):
                mask_color_entries = sticker_mask_rgb_observations(
                    record.frame,
                    record.roi_xyxy,
                    record.sticker_masks,
                    sticker_mask_erosion,
                )
                selected_candidate = replace(
                    selected_candidate,
                    sticker_mask_colors=tuple(
                        rgb for _mask_index, rgb in mask_color_entries
                    ),
                )
            static_candidate_before_turn = selected_candidate
            current_geometry = prepare_lookahead_geometry(
                record_index - 1,
                static_candidate_before_turn,
            )
            selected_candidate = solve_pnp_from_low_error_assignments(
                selected_candidate,
                recover_layer_turns=True,
                turn_color_priors=turn_color_priors,
                turn_global_profiles=global_profiles,
                turn_expected_color_by_label=None,
                record_all_turn_candidates=record_all_turn_candidates,
                turn_geometry_frame=(
                    current_geometry[1] if current_geometry is not None else None
                ),
                turn_previous_move=last_committed_move,
                turn_previous_states=tuple(signature for _index, signature in committed_state_history),
            )
            selected_candidate = apply_move_lookahead(
                selected_candidate,
                static_candidate_before_turn,
                cube_geometry_state,
                record_index,
            )
        selected_candidate = replace(
            selected_candidate,
            orientation_anchor=orientation_anchor_for_candidate(selected_candidate),
        )
        record.pose_diagnostic_candidate = None
        if motion_rejected or not pose_reprojection_error_is_acceptable(selected_candidate):
            # Keep the color trace available for the independent classifier,
            # but do not let an over-threshold geometric fit become the pose
            # shown in the final pass.
            record.pose_candidate = None
            record.pose_candidates = ()
            record.pose_diagnostic_candidate = selected_candidate
            # Preserve the last accepted pose as the continuity reference.
            # A rejected frame must not become the starting point for the
            # next refinement attempt.
            corrected_trace.observe(
                record.frame_index,
                selected_observations,
                observation_weights=selected_observation_weights,
            )
            if recover_orientation and known_initial_colors is None:
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
        previous_continuity_candidate = continuity_reference_candidate
        continuity_symmetry_tracker.observe(continuity_reference_candidate.rvec)
        if recognize_moves:
            committed_moves = completed_candidate_moves(selected_candidate)
            if committed_moves:
                prior_signature = cube_geometry_state.signature()
                for move in committed_moves:
                    committed_state_history.append((record_index - 1, cube_geometry_state.signature()))
                    cube_geometry_state = cube_geometry_state.apply_move(move)
                last_committed_move = committed_moves[-1]
                if calibrate_initial_colors and not calibration_done:
                    if pending_calibration_move is None:
                        pending_calibration_move = (record.frame_index, prior_signature)
                        print(f"Color calibration paused at frame {record.frame_index}; watching {INITIAL_CALIBRATION_MOVE_PATIENCE_FRAMES} frames for a reversal.", flush=True)
                    if cube_geometry_state.signature() == pending_calibration_move[1]:
                        print(f"Color calibration resumed at frame {record.frame_index}: moves canceled.", flush=True)
                        pending_calibration_move = None
            elif (not any(abs(float(rotation.angle_degrees)) > 1e-6
                          for rotation in selected_candidate.layer_rotations)
                  and pending_calibration_move is None and not selected_candidate.turn_color_unreliable):
                # Partial turns that return to rest do not commit a move, but
                # their moving frames are not reliable calibration samples.
                accepted_initial_pose = True
                if calibrate_initial_colors and not calibration_done:
                    initial_calibration_trace.observe(
                        record.frame_index, selected_observations,
                        observation_weights=selected_observation_weights,
                    )
        if (recognize_moves and global_profiles is not None and known_initial_colors is None
                and not selected_candidate.turn_color_unreliable):
            turn_observations = selected_observations
            if (
                getattr(record, "sticker_masks", None) is not None
                and record.roi_xyxy is not None
            ):
                sampled_observations = sample_projected_sticker_colors(
                    record.frame,
                    record.roi_xyxy,
                    record.sticker_masks,
                    selected_candidate,
                    sticker_mask_erosion=sticker_mask_erosion,
                )
                if sampled_observations:
                    turn_observations = sampled_observations
            turn_rotation, _ = cv2.Rodrigues(
                np.asarray(selected_candidate.rvec, dtype=np.float64).reshape(3, 1)
            )
            turn_observation_weights = {
                identity.label: sticker_view_reliability_weight(
                    selected_candidate,
                    identity,
                    turn_rotation,
                )
                for identity in identities
                if identity.label in turn_observations
            }
            _update_sticker_color_posteriors(
                turn_color_priors,
                turn_observations,
                global_profiles,
                turn_observation_weights,
            )
            # The frame/state key alone cannot identify scores after learning.
            lookahead_color_trial_cache.clear()
            lookahead_frame_lower_bound_cache.clear()
        corrected_trace.observe(
            record.frame_index,
            selected_observations,
            observation_weights=selected_observation_weights,
        )
        if recover_orientation and known_initial_colors is None:
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
    calibrate_initial_profiles(pending_calibration_move[0] if pending_calibration_move else None)
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


def _global_profile_reference(profile: GlobalColorProfile) -> StickerColorProfile | None:
    """Return the fitted global profile, falling back to its pure-color prior."""

    return profile.fitted if profile.fitted is not None else profile.prior


def scramble_from_classified_sticker_colors(
    trace: StickerColorTrace | None,
    sticker_color_names: dict[str, str],
    *,
    solve=None,
    diagnostic: list[str] | None = None,
    normalize_centers: bool = True,
) -> str | None:
    """Return a solved-to-recognized scramble when all colors form a valid cube."""

    if trace is None or len(sticker_color_names) < len(trace.identities):
        return None

    colors = {}
    for identity in trace.identities:
        color = sticker_color_names.get(identity.label)
        if color is None:
            return None
        colors[(identity.cubie_index, identity.axis, identity.side)] = color

    try:
        state = CubeState(colors)
        if normalize_centers:
            state = state.reoriented_to_standard_centers()
        return state.scramble_from_solved(solve=solve)
    except (RuntimeError, ValueError) as exc:
        # Classification may produce an incomplete or physically impossible
        # cube. In that case there is no scramble to report.
        if diagnostic is not None:
            diagnostic.append(str(exc))
        return None


def report_classified_scramble(
    trace: StickerColorTrace | None,
    sticker_color_names: dict[str, str],
    *,
    normalize_centers: bool = True,
) -> None:
    """Print the recognized scramble or explain why it is unavailable."""

    if trace is None:
        print(
            "Recognized scramble unavailable: no sticker color trace was collected.",
            flush=True,
        )
        return
    classified_count = len(sticker_color_names)
    total_count = len(trace.identities)
    if classified_count < total_count:
        print(
            "Recognized scramble unavailable: "
            f"{classified_count}/{total_count} stickers classified; all 54 are required.",
            flush=True,
        )
        return
    diagnostic: list[str] = []
    recognized_scramble = scramble_from_classified_sticker_colors(
        trace,
        sticker_color_names,
        diagnostic=diagnostic,
        normalize_centers=normalize_centers,
    )
    if recognized_scramble is None:
        message = (
            "Recognized scramble unavailable: classified colors do not form a "
            "valid Kociemba cube state."
        )
        if diagnostic:
            message += f" Kociemba: {diagnostic[0]}"
        print(message, flush=True)
        return
    print(
        "Recognized scramble: "
        + (recognized_scramble or "(solved)"),
        flush=True,
    )


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
        peak_distance = float(distance[peak_y, peak_x])
        centroid_distance = float(
            distance[
                int(np.clip(round(local_center[1]), 0, distance.shape[0] - 1)),
                int(np.clip(round(local_center[0]), 0, distance.shape[1] - 1)),
            ]
        )
        heatmap_confidence = (
            centroid_distance / peak_distance if peak_distance > 0.0 else 0.0
        )
        heatmap_confidence = float(
            np.clip(
                heatmap_confidence,
                PNP_MASK_CENTER_HEATMAP_CONFIDENCE_FLOOR,
                1.0,
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
        # A centroid in a low-intensity heatmap region is less trustworthy
        # even when its overall mask area is large. This directly reduces its
        # inverse-variance weight in center scoring and turn ranking.
        uncertainty /= np.sqrt(heatmap_confidence)
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


def solve_sticker_pnp_candidates(
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
    sticker_mask_colors: tuple[np.ndarray, ...] | None = None,
) -> tuple[StickerPnPCandidate, ...]:
    """Return every valid PnP pose from the selected sticker correspondence."""

    observed_indices = list(match.keypoint_indices)
    if match.sticker_type == "unknown" or len(observed_indices) not in (3, 4):
        return tuple()
    observed_points = np.asarray(frame_points[observed_indices], dtype=np.float32)
    observed_class_ids = [int(point_class_ids[index]) for index in observed_indices]
    index_sequences = sticker_correspondence_object_indices(
        match.sticker_type,
        observed_class_ids,
        observed_points,
    )
    if not index_sequences:
        return tuple()

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
    all_sticker_colors = tuple(
        np.asarray(color, dtype=np.uint8).reshape(3).copy()
        for color in (sticker_mask_colors or ())
        if np.isfinite(np.asarray(color, dtype=np.float64)).all()
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
                # Three-point hypotheses are coplanar and can have multiple
                # valid camera poses. Preserve every SQPNP root, including in
                # the initial pass where contour geometry is unavailable.
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
                    sticker_mask_index=match.mask_index,
                    sticker_contours=all_sticker_contours,
                    sticker_boundary_samples=sticker_boundary_samples,
                    sticker_mask_centers=all_sticker_centers,
                    sticker_mask_center_uncertainties=all_sticker_center_uncertainties,
                    sticker_mask_colors=all_sticker_colors,
                    image_size=image_size,
                )
            )
    if not candidates:
        return tuple()
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.overall_reprojection_error,
                item.reprojection_error,
            ),
        )
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
    return np.isfinite(metric) and (
        not PNP_POSE_REPROJECTION_ERROR_REJECTION_ENABLED
        or metric <= PNP_POSE_REPROJECTION_ERROR_THRESHOLD
    )


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
        # A width-three rotation is a whole-cube orientation change, so it
        # rotates about the cube center rather than an outer face pivot.
        center = np.zeros(3, dtype=np.float64)
        if int(layer_rotation.layer_count) < 3:
            center[axis] = side * CUBE_HALF_SIZE_MM
        transformed[mask] = (
            (rotation_matrix @ (transformed[mask] - center).T).T + center
        )
    return transformed.astype(np.float32)


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

    transformed = replace(
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
        sticker_layer_rotations=(
            None if candidate.sticker_layer_rotations is None else tuple(
                transform_layer_rotation_by_symmetry(rotation, symmetry)
                for rotation in candidate.sticker_layer_rotations
            )
        ),
        turn_candidate_scores=transformed_scores,
        turn_candidate_debug=transformed_debug,
        rvec=transformed_rvec,
        pre_final_rvec=transform_optional_rvec(candidate.pre_final_rvec),
        orientation_anchor=None,
        object_symmetry=(
            np.asarray(symmetry, dtype=np.float64).reshape(3, 3)
            @ prior_object_symmetry
        ),
    )
    # Symmetry changes both the cube-frame object points and the camera pose.
    # Recompute cached projections so sticker centers, assignment residuals,
    # and color sampling remain aligned after final orientation recovery.
    return refresh_candidate_pose_projection(
        transformed,
        transformed.rvec,
        transformed.tvec,
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


def layer_face_debug_label(axis: int, side: int) -> str:
    """Return the compact label used for a candidate face layer."""

    axis_names = "XYZ"
    axis_index = int(axis)
    axis_name = axis_names[axis_index] if axis_index in range(3) else "?"
    return f"{axis_name}{'+' if int(side) > 0 else '-'}"


def _turn_torch_scoring_device(device: object | None) -> object | None:
    """Return a CUDA/MPS Torch device when turn scoring is requested and usable."""

    if device is None:
        return None
    try:
        import torch
    except ImportError:
        return None
    device_text = str(device).lower()
    if device_text.startswith("cuda"):
        return torch.device(device_text) if torch.cuda.is_available() else None
    if device_text.isdigit():
        return torch.device(f"cuda:{device_text}") if torch.cuda.is_available() else None
    if device_text.startswith("mps"):
        mps_backend = getattr(torch.backends, "mps", None)
        return torch.device("mps") if mps_backend is not None and mps_backend.is_available() else None
    return None


@lru_cache(maxsize=1)
def _turn_gpu_static_geometry() -> tuple[np.ndarray, ...]:
    """Build the fixed cube geometry used by the batched turn scorer."""

    triangles: list[np.ndarray] = []
    triangle_centers: list[np.ndarray] = []
    sticker_faces: list[np.ndarray] = []
    sticker_centers: list[np.ndarray] = []
    sticker_cubie_centers: list[np.ndarray] = []
    sticker_normals: list[np.ndarray] = []
    identity_indices_by_geometry = {
        (identity.cubie_index, identity.axis, identity.side): index
        for index, identity in enumerate(canonical_sticker_identities())
    }
    sticker_identity_indices: list[int] = []
    for x_index in range(3):
        for y_index in range(3):
            for z_index in range(3):
                center = cubelet_center_object_point(x_index, y_index, z_index)
                outer_faces = tuple(
                    (axis, side)
                    for axis in range(3)
                    for side in (-1, 1)
                    if (
                        (axis == 0 and ((side == -1 and x_index == 0) or (side == 1 and x_index == 2)))
                        or (axis == 1 and ((side == -1 and y_index == 0) or (side == 1 and y_index == 2)))
                        or (axis == 2 and ((side == -1 and z_index == 0) or (side == 1 and z_index == 2)))
                    )
                )
                for axis, side in outer_faces:
                    face = cubelet_face_object_points(center, axis, side).astype(np.float64)
                    sticker_faces.append(face)
                    sticker_centers.append(np.mean(face, axis=0))
                    sticker_cubie_centers.append(np.asarray(center, dtype=np.float64))
                    sticker_normals.append(np.cross(face[1] - face[0], face[2] - face[0]))
                    sticker_identity_indices.append(
                        identity_indices_by_geometry[(
                            (x_index, y_index, z_index),
                            axis,
                            side,
                        )]
                    )
                    triangles.append(np.asarray((face[0], face[1], face[2]), dtype=np.float64))
                    triangles.append(np.asarray((face[0], face[2], face[3]), dtype=np.float64))
                    triangle_centers.extend((center, center))
    return (
        np.asarray(triangles, dtype=np.float64),
        np.asarray(triangle_centers, dtype=np.float64),
        np.asarray(sticker_faces, dtype=np.float64),
        np.asarray(sticker_centers, dtype=np.float64),
        np.asarray(sticker_cubie_centers, dtype=np.float64),
        np.asarray(sticker_normals, dtype=np.float64),
        np.asarray(sticker_identity_indices, dtype=np.int64),
    )


@lru_cache(maxsize=4)
def _turn_gpu_static_tensors(device: str, dtype: object) -> tuple[object, ...]:
    """Keep immutable cube geometry on the scoring device between frames."""

    import torch

    triangles, triangle_centers, sticker_faces, _, sticker_centers, sticker_normals, _ = (
        _turn_gpu_static_geometry()
    )
    return tuple(
        torch.as_tensor(value, dtype=dtype, device=device)
        for value in (
            CUBE_LATTICE_OBJECT_POINTS_MM,
            CUBE_LATTICE_BOUNDARY_NORMALS,
            triangles,
            triangle_centers,
            sticker_faces,
            sticker_centers,
            sticker_normals,
        )
    ) + (
        torch.as_tensor(CUBE_LATTICE_BOUNDARY_FACE_MASK, dtype=torch.bool, device=device),
        torch.as_tensor(CUBE_LATTICE_CLASS_IDS, dtype=torch.long, device=device),
    )


def _turn_gpu_identity_indices_for_state(
    geometry_state: CubeGeometryState | None,
    canonical_geometry_identity_indices: np.ndarray,
) -> np.ndarray:
    """Map each fixed surface position to its current physical identity."""

    canonical_indices = np.asarray(
        canonical_geometry_identity_indices,
        dtype=np.int64,
    ).reshape(-1)
    if geometry_state is None:
        return canonical_indices.copy()

    identities = canonical_sticker_identities()
    identity_index_by_label = {
        identity.label: index
        for index, identity in enumerate(identities)
    }
    remapped: list[int] = []
    for canonical_index in canonical_indices:
        if canonical_index < 0 or canonical_index >= len(identities):
            raise ValueError("Invalid canonical sticker identity index")
        canonical_identity = identities[int(canonical_index)]
        position = (
            canonical_identity.cubie_index,
            canonical_identity.axis,
            canonical_identity.side,
        )
        label = geometry_state.facelets.get(position)
        if label is None or label not in identity_index_by_label:
            raise ValueError("Incomplete cube geometry state")
        remapped.append(identity_index_by_label[label])
    return np.asarray(remapped, dtype=np.int64)


def _torch_axis_rotation_matrices(torch_module: object, axes: object, angles: object) -> object:
    """Return one Rodrigues matrix per axis/angle pair."""

    torch = torch_module
    count = int(axes.shape[0])
    zeros = torch.zeros(count, dtype=angles.dtype, device=angles.device)
    ones = torch.ones(count, dtype=angles.dtype, device=angles.device)
    cosine = torch.cos(angles)
    sine = torch.sin(angles)
    rotation_x = torch.stack(
        (
            torch.stack((ones, zeros, zeros), dim=1),
            torch.stack((zeros, cosine, -sine), dim=1),
            torch.stack((zeros, sine, cosine), dim=1),
        ),
        dim=1,
    )
    rotation_y = torch.stack(
        (
            torch.stack((cosine, zeros, sine), dim=1),
            torch.stack((zeros, ones, zeros), dim=1),
            torch.stack((-sine, zeros, cosine), dim=1),
        ),
        dim=1,
    )
    rotation_z = torch.stack(
        (
            torch.stack((cosine, -sine, zeros), dim=1),
            torch.stack((sine, cosine, zeros), dim=1),
            torch.stack((zeros, zeros, ones), dim=1),
        ),
        dim=1,
    )
    rotations = torch.stack((rotation_x, rotation_y, rotation_z), dim=1)
    return rotations[torch.arange(count, device=axes.device), axes]


def _torch_project_turn_points(
    torch_module: object,
    points: object,
    camera_rotation: object,
    tvec: object,
    camera_matrix: object,
    dist_coeffs: object,
) -> object:
    """Project batched 3-D points with the OpenCV pinhole distortion model."""

    torch = torch_module
    camera_points = torch.einsum("ij,b...j->b...i", camera_rotation, points) + tvec
    z = camera_points[..., 2]
    safe_z = torch.where(torch.abs(z) > 1e-12, z, torch.ones_like(z))
    x = camera_points[..., 0] / safe_z
    y = camera_points[..., 1] / safe_z
    coefficients = torch.zeros(5, dtype=points.dtype, device=points.device)
    flattened_coefficients = dist_coeffs.reshape(-1)
    coefficient_count = min(5, int(flattened_coefficients.shape[0]))
    if coefficient_count:
        coefficients[:coefficient_count] = flattened_coefficients[:coefficient_count]
    k1, k2, p1, p2, k3 = coefficients
    radius_squared = x * x + y * y
    radial = 1.0 + radius_squared * (
        k1 + radius_squared * (k2 + radius_squared * k3)
    )
    distorted_x = x * radial + 2.0 * p1 * x * y + p2 * (
        radius_squared + 2.0 * x * x
    )
    distorted_y = y * radial + p1 * (radius_squared + 2.0 * y * y) + 2.0 * p2 * x * y
    return torch.stack(
        (
            camera_matrix[0, 0] * distorted_x + camera_matrix[0, 2],
            camera_matrix[1, 1] * distorted_y + camera_matrix[1, 2],
        ),
        dim=-1,
    )


def _gpu_sticker_center_fit_metrics(
    candidate: StickerPnPCandidate,
    torch_module: object,
    projected_face_centers: object,
    visible_faces: object,
    scale: float,
    max_error: float,
    *,
    return_tensors: bool = False,
    layer_mask: object | None = None,
    return_layer_match_counts: bool = False,
) -> tuple[object, ...]:
    """Score batched sticker centers with accelerator-resident nearest matching.

    The mask centroids are intentionally allowed to share a projected face.
    This avoids the CPU-only Hungarian assignment while retaining the same
    uncertainty-weighted, gated center penalty for every configuration.
    """

    torch = torch_module
    observed = np.asarray(getattr(candidate, "sticker_mask_centers", ()), dtype=np.float64).reshape(-1, 2)
    if len(observed) == 0:
        count = int(projected_face_centers.shape[0])
        if return_tensors:
            zero = projected_face_centers.new_zeros(count)
            result = (zero, zero, zero.to(dtype=torch.int32))
            if return_layer_match_counts:
                return result + (zero.to(dtype=torch.int32),)
            return result
        result = (np.zeros(count), np.zeros(count), np.zeros(count, dtype=np.int32))
        if return_layer_match_counts:
            return result + (np.zeros(count, dtype=np.int32),)
        return result
    uncertainties = np.asarray(
        getattr(candidate, "sticker_mask_center_uncertainties", ()),
        dtype=np.float64,
    ).reshape(-1)
    if len(uncertainties) != len(observed):
        uncertainties = np.ones(len(observed), dtype=np.float64)
    uncertainties = np.where(
        np.isfinite(uncertainties), np.maximum(uncertainties, 1.0), np.inf
    )
    weights = 1.0 / np.square(uncertainties)
    observed_t = torch.as_tensor(observed, dtype=projected_face_centers.dtype, device=projected_face_centers.device)
    uncertainty_t = torch.as_tensor(uncertainties, dtype=projected_face_centers.dtype, device=projected_face_centers.device)
    weight_t = torch.as_tensor(weights, dtype=projected_face_centers.dtype, device=projected_face_centers.device)
    distances = torch.linalg.vector_norm(
        observed_t[None, :, None, :] - projected_face_centers[:, None, :, :],
        dim=-1,
    )
    distances = torch.where(visible_faces[:, None, :], distances, torch.full_like(distances, float("inf")))
    gate = max_error / scale
    tolerance = torch.minimum(
        0.02 + torch.minimum(uncertainty_t / scale, torch.as_tensor(0.01, dtype=uncertainty_t.dtype, device=uncertainty_t.device)),
        torch.as_tensor(gate - 1e-9, dtype=uncertainty_t.dtype, device=uncertainty_t.device),
    )
    fraction = torch.clamp(
        (distances / scale - tolerance[None, :, None])
        / torch.clamp(gate - tolerance[None, :, None], min=1e-12),
        0.0,
        1.0,
    )
    penalties = fraction * fraction * (3.0 - 2.0 * fraction)
    nearest_distances, nearest_faces = torch.min(distances, dim=2)
    nearest_penalties = torch.gather(
        penalties,
        2,
        nearest_faces[:, :, None],
    ).squeeze(2)
    finite = torch.isfinite(nearest_distances) & (weight_t[None, :] > 0.0)
    safe_weights = torch.where(
        finite,
        weight_t[None, :].expand_as(nearest_distances),
        torch.zeros_like(nearest_distances),
    )
    total_weight_t = torch.sum(safe_weights, dim=1)
    means_t = torch.where(
        total_weight_t > 0.0,
        PNP_TURN_CENTER_PENALTY_SCALE
        * torch.sum(nearest_penalties * safe_weights, dim=1)
        / torch.clamp(total_weight_t, min=1e-12),
        torch.full_like(total_weight_t, float("inf")),
    )

    # Compute weighted median and p90 on the accelerator.  Invalid matches
    # have zero weight and sort to the end, so no per-configuration Python
    # loop or host-side assignment is needed.
    normalized = torch.where(
        finite,
        nearest_distances / scale,
        torch.full_like(nearest_distances, float("inf")),
    )
    sorted_values, order = torch.sort(normalized, dim=1)
    sorted_weights = torch.gather(safe_weights, 1, order)
    cumulative = torch.cumsum(sorted_weights, dim=1)
    median_targets = 0.50 * total_weight_t[:, None]
    p90_targets = 0.90 * total_weight_t[:, None]
    median_reached = cumulative >= median_targets
    p90_reached = cumulative >= p90_targets
    median_indices = torch.argmax(median_reached.to(torch.int32), dim=1)
    p90_indices = torch.argmax(p90_reached.to(torch.int32), dim=1)
    median_values = torch.gather(sorted_values, 1, median_indices[:, None]).squeeze(1)
    p90_values = torch.gather(sorted_values, 1, p90_indices[:, None]).squeeze(1)
    robust_t = torch.where(
        total_weight_t > 0.0,
        median_values + 0.5 * p90_values,
        torch.full_like(total_weight_t, float("inf")),
    )
    match_counts_t = torch.count_nonzero(
        finite & (nearest_distances < max_error),
        dim=1,
    )
    layer_match_counts_t = None
    if layer_mask is not None:
        layer_distances = torch.where(
            visible_faces[:, None, :] & layer_mask[:, None, :],
            distances,
            torch.full_like(distances, float("inf")),
        )
        nearest_layer_distances = torch.min(layer_distances, dim=2).values
        layer_match_counts_t = torch.count_nonzero(
            torch.isfinite(nearest_layer_distances)
            & (nearest_layer_distances < max_error),
            dim=1,
        )
    if return_tensors:
        result = (means_t, robust_t, match_counts_t)
        if return_layer_match_counts:
            if layer_match_counts_t is None:
                layer_match_counts_t = torch.zeros_like(match_counts_t)
            return result + (layer_match_counts_t,)
        return result
    result = (
        means_t.detach().cpu().numpy().astype(np.float64, copy=False),
        robust_t.detach().cpu().numpy().astype(np.float64, copy=False),
        match_counts_t.detach().cpu().numpy().astype(np.int32, copy=False),
    )
    if return_layer_match_counts:
        if layer_match_counts_t is None:
            layer_match_counts_t = torch.zeros_like(match_counts_t)
        return result + (
            layer_match_counts_t.detach().cpu().numpy().astype(np.int32, copy=False),
        )
    return result


def _gpu_sticker_color_fit_metrics(
    candidate: StickerPnPCandidate,
    torch_module: object,
    projected_face_centers: object,
    projected_face_polygons: object,
    visible_faces: object,
    sticker_identity_indices: np.ndarray,
    scale: float,
    max_error: float,
    color_likelihoods: np.ndarray | None,
    *,
    return_tensors: bool = False,
    assignment_cache: dict | None = None,
) -> tuple[object, object]:
    """Score mask colors against moving physical sticker identities on-device."""

    torch = torch_module
    colors = tuple(getattr(candidate, "sticker_mask_colors", ()))
    centers = tuple(getattr(candidate, "sticker_mask_centers", ()))
    if color_likelihoods is None or not colors or not centers:
        count = int(projected_face_centers.shape[0])
        if return_tensors:
            zero = projected_face_centers.new_zeros(count)
            return zero, zero.to(dtype=torch.int32)
        return np.zeros(count, dtype=np.float64), np.zeros(count, dtype=np.int32)
    count = min(len(colors), len(centers), int(color_likelihoods.shape[1]))
    if count <= 0:
        batch = int(projected_face_centers.shape[0])
        if return_tensors:
            zero = projected_face_centers.new_zeros(batch)
            return zero, zero.to(dtype=torch.int32)
        return np.zeros(batch, dtype=np.float64), np.zeros(batch, dtype=np.int32)
    assignment_key = (count, float(max_error),
                      np.asarray(centers[:count], dtype=np.float64).tobytes(),
                      np.asarray(getattr(candidate, 'sticker_mask_center_uncertainties', ()), dtype=np.float64).tobytes())
    cached = assignment_cache.get('value') if assignment_cache is not None and assignment_cache.get('key') == assignment_key else None
    if cached is None:
        observed = np.asarray(centers[:count], dtype=np.float64).reshape(-1, 2)
        uncertainties = np.asarray(
            getattr(candidate, "sticker_mask_center_uncertainties", ()),
            dtype=np.float64,
        ).reshape(-1)
        if len(uncertainties) != count or not np.isfinite(uncertainties[:count]).all():
            uncertainties = np.ones(count, dtype=np.float64)
        else:
            uncertainties = np.maximum(uncertainties[:count], 1.0)
        weights = 1.0 / np.square(uncertainties)
        observed_t = torch.as_tensor(
            observed,
            dtype=projected_face_centers.dtype,
            device=projected_face_centers.device,
        )
        distances = torch.linalg.vector_norm(
            observed_t[None, :, None, :] - projected_face_centers[:, None, :, :],
            dim=-1,
        )
        polygons = projected_face_polygons
        edges = torch.roll(polygons, shifts=-1, dims=2) - polygons
        relative = observed_t[None, None, :, None, :] - polygons[:, :, None, :, :]
        cross = (
            edges[:, :, None, :, 0] * relative[:, :, :, :, 1]
            - edges[:, :, None, :, 1] * relative[:, :, :, :, 0]
        )
        polygon_tolerance = torch.as_tensor(
            1e-4,
            dtype=polygons.dtype,
            device=polygons.device,
        )
        inside_positive = torch.all(cross >= -polygon_tolerance, dim=3)
        inside_negative = torch.all(cross <= polygon_tolerance, dim=3)
        inside = (inside_positive | inside_negative).permute(0, 2, 1)
        distances = torch.where(
            visible_faces[:, None, :] & inside,
            distances,
            torch.full_like(distances, float("inf")),
        )
        nearest_distances, nearest_faces = torch.min(distances, dim=2)
        finite = torch.isfinite(nearest_distances)
        safe_weights = torch.as_tensor(
            weights,
            dtype=projected_face_centers.dtype,
            device=projected_face_centers.device,
        )[None, :].expand_as(nearest_distances)
        safe_weights = torch.where(
            finite,
            safe_weights,
            torch.zeros_like(safe_weights),
        )
        match_counts = torch.count_nonzero(
            finite & (nearest_distances <= max_error),
            dim=1,
        )
        cached = (nearest_faces, safe_weights, match_counts)
        if assignment_cache is not None:
            assignment_cache.update(key=assignment_key, value=cached,
                                    builds=assignment_cache.get('builds', 0) + 1)
    elif assignment_cache is not None:
        assignment_cache['hits'] = assignment_cache.get('hits', 0) + 1
    nearest_faces, safe_weights, match_counts = cached
    identity_indices = torch.as_tensor(
        sticker_identity_indices,
        dtype=torch.long,
        device=projected_face_centers.device,
    )
    # The frontier shares projected geometry; only the identity at each fixed face
    # differs between states. Gather every state's likelihoods in one batch.
    selected_identity_indices = (
        identity_indices[nearest_faces] if identity_indices.ndim == 1
        else identity_indices[:, nearest_faces]
    )
    likelihoods_t = torch.as_tensor(
        np.asarray(color_likelihoods[:, :count], dtype=np.float64),
        dtype=projected_face_centers.dtype,
        device=projected_face_centers.device,
    )
    observation_indices = torch.arange(count, dtype=torch.long, device=likelihoods_t.device)
    selected_likelihoods = likelihoods_t[
        selected_identity_indices,
        observation_indices[None, :],
    ]
    log_likelihoods = torch.log(torch.clamp(selected_likelihoods, min=1e-30))
    total_weight = torch.sum(safe_weights, dim=1)
    color_error = torch.where(
        total_weight > 0.0,
        -torch.sum(safe_weights * log_likelihoods, dim=-1)
        / torch.clamp(total_weight, min=1e-12),
        torch.full_like(total_weight, float("inf")),
    )
    if identity_indices.ndim > 1:
        match_counts = match_counts[None, :].expand(identity_indices.shape[0], -1)
    if return_tensors:
        return color_error, match_counts
    return (
        color_error.detach().cpu().numpy().astype(np.float64, copy=False),
        match_counts.detach().cpu().numpy().astype(np.int32, copy=False),
    )


def _gpu_keypoint_fit_metrics(
    torch: object,
    distances: object,
    objects: object,
    classes: object,
    axes: object,
    sides: object,
    layer_counts: object,
    observed_classes: object | None,
    scale: float,
    assignment_error_limit: float,
) -> tuple[object, object, object, object, object, object]:
    """Compute the existing keypoint ranking statistics for all trials."""

    finite = torch.isfinite(distances)
    coordinates = torch.gather(
        objects, 2, axes[:, None, None].expand(-1, objects.shape[1], 1)
    ).squeeze(-1)
    boundary = CUBE_HALF_SIZE_MM - (layer_counts - 1.0) * CUBE_LAYER_SIZE_MM
    layer_mask = torch.where(
        sides[:, None] > 0.0,
        coordinates >= boundary[:, None] - 1e-4,
        coordinates <= -boundary[:, None] + 1e-4,
    )
    layer_finite = finite & layer_mask
    valid_trial = torch.any(layer_finite, dim=1)
    if observed_classes is not None:
        known = (observed_classes[None, :] >= 0) & (classes >= 0) & layer_finite
        valid_trial = valid_trial & ~torch.any(
            known & (observed_classes[None, :] != classes), dim=1
        )

    normalized = distances / scale
    finite_count = torch.count_nonzero(finite, dim=1)
    mean = torch.sum(torch.where(finite, normalized, torch.zeros_like(normalized)), dim=1) / torch.clamp(finite_count, min=1)
    sorted_values, _ = torch.sort(
        torch.where(finite, normalized, torch.full_like(normalized, float("inf"))), dim=1
    )

    def percentile(fraction: float) -> object:
        position = (torch.clamp(finite_count, min=1) - 1).to(dtype=distances.dtype) * fraction
        lower = torch.floor(position).to(dtype=torch.long)
        upper = torch.ceil(position).to(dtype=torch.long)
        low_value = torch.gather(sorted_values, 1, lower[:, None]).squeeze(1)
        high_value = torch.gather(sorted_values, 1, upper[:, None]).squeeze(1)
        weight = position - lower.to(dtype=distances.dtype)
        # Avoid inf - inf on trials with no finite points.
        return torch.where(lower == upper, low_value, low_value + (high_value - low_value) * weight)

    median = percentile(0.5)
    p90 = percentile(0.9)
    infinity = torch.full_like(mean, float("inf"))
    mean = torch.where(valid_trial, mean, infinity)
    robust = torch.where(valid_trial, median + 0.5 * p90, infinity)
    p90 = torch.where(valid_trial, p90, infinity)
    inliers = torch.where(valid_trial, torch.count_nonzero(finite & (distances <= assignment_error_limit), dim=1), 0)
    layer_inliers = torch.where(valid_trial, torch.count_nonzero(layer_finite & (distances <= assignment_error_limit), dim=1), 0)
    return mean, robust, p90, inliers, layer_inliers, valid_trial


def _score_turn_configurations_on_gpu(
    candidate: StickerPnPCandidate,
    assignment_error_limit: float,
    scoring_device: object,
    color_likelihoods: np.ndarray | None = None,
    *,
    materialize_projections: bool = True,
    angle_values: np.ndarray | None = None,
    enforce_layer_observation_gate: bool = True,
    enforce_validity_gate: bool = True,
) -> list[dict[str, object]] | None:
    """Batch exhaustive turn projection and scoring on a Torch device.

    Mask centroids use nearest visible projected centers, so the complete
    numeric configuration score stays on the accelerator. All configuration
    geometry, projection, class-aware nearest assignment, and residual
    calculation are batched on the GPU. Return ``None`` if the input or
    accelerator cannot be scored.
    """

    try:
        import torch

        geometry_state = getattr(candidate, "cube_geometry_state", None)
        base_rotations = tuple(candidate.layer_rotations)
        if base_rotations:
            return None
        image_points = np.asarray(candidate.assignment_image_points, dtype=np.float64).reshape(-1, 2)
        projected_points = np.asarray(candidate.assignment_projected_points, dtype=np.float64).reshape(-1, 2)
        object_points = np.asarray(candidate.assignment_object_points, dtype=np.float64).reshape(-1, 3)
        if (
            len(image_points) < 3
            or len(image_points) != len(projected_points)
            or len(image_points) != len(object_points)
            or not np.isfinite(assignment_error_limit)
        ):
            return None
        scale = projected_cube_diameter(candidate)
        if not np.isfinite(scale) or scale <= 0.0:
            return None
        base_object_points = np.asarray(CUBE_LATTICE_OBJECT_POINTS_MM, dtype=np.float64)
        sticker_identity_indices = _turn_gpu_static_geometry()[-1]
        sticker_identity_indices = _turn_gpu_identity_indices_for_state(
            geometry_state,
            sticker_identity_indices,
        )
        angles = (
            turn_search_angles()
            if angle_values is None
            else np.asarray(angle_values, dtype=np.float64).reshape(-1)
        )
        eligible_layers = {
            (axis, side, layer_count): turn_layer_observation_count(
                candidate,
                axis,
                side,
                layer_count,
            )
            for axis in range(3)
            for side in (-1, 1)
            for layer_count in (1, 2)
        }
        configurations: list[tuple[int, int, int, float, bool, tuple[CubeLayerRotation, ...], str]] = []
        for axis in range(3):
            for side in (-1, 1):
                for layer_count in (1, 2):
                    configurations.append(
                        (
                            axis,
                            side,
                            layer_count,
                            0.0,
                            True,
                            (),
                            f"static {layer_face_debug_label(axis, side)}",
                        )
                    )
                    if enforce_layer_observation_gate and (
                        int(cube_layer_mask(base_object_points, axis, side, layer_count).sum()) < 3
                        or eligible_layers[(axis, side, layer_count)]
                        < MIN_TURN_LAYER_OBSERVATIONS
                    ):
                        continue
                    for angle in angles:
                        if abs(float(angle)) < 1e-6:
                            continue
                        rotation = CubeLayerRotation(axis, side, float(angle), layer_count)
                        configurations.append(
                            (
                                axis,
                                side,
                                layer_count,
                                float(angle),
                                False,
                                (rotation,),
                                layer_rotation_debug_label((rotation,)),
                            )
                        )
        if not configurations:
            return []

        device = scoring_device
        dtype = torch.float32 if str(device).startswith("mps") else torch.float64
        count = len(configurations)
        axes = torch.as_tensor([item[0] for item in configurations], dtype=torch.long, device=device)
        sides = torch.as_tensor([item[1] for item in configurations], dtype=dtype, device=device)
        layer_counts = torch.as_tensor([item[2] for item in configurations], dtype=dtype, device=device)
        angles_t = torch.as_tensor([item[3] for item in configurations], dtype=dtype, device=device)
        active = torch.as_tensor([not item[4] for item in configurations], dtype=torch.bool, device=device)
        (
            base_points_t, base_normals_t, triangles_t, triangle_centers_t,
            sticker_faces_t, sticker_face_centers_t, sticker_normals_t,
            base_face_mask_t, trial_classes,
        ) = _turn_gpu_static_tensors(str(device), dtype)
        camera_rotation_np, _ = cv2.Rodrigues(np.asarray(candidate.rvec, dtype=np.float64).reshape(3, 1))
        camera_rotation = torch.as_tensor(camera_rotation_np, dtype=dtype, device=device)
        tvec = torch.as_tensor(np.asarray(candidate.tvec, dtype=np.float64).reshape(1, 1, 3), dtype=dtype, device=device)
        camera_matrix = torch.as_tensor(np.asarray(candidate.camera_matrix, dtype=np.float64), dtype=dtype, device=device)
        dist_coeffs = torch.as_tensor(np.asarray(candidate.dist_coeffs, dtype=np.float64), dtype=dtype, device=device)
        rotation = _torch_axis_rotation_matrices(torch, axes, sides * angles_t * (np.pi / 180.0))
        point_coordinates = base_points_t[:, axes].T
        layer_boundary = CUBE_HALF_SIZE_MM - (layer_counts - 1.0) * CUBE_LAYER_SIZE_MM
        point_layer_mask = torch.where(
            sides[:, None] > 0.0,
            point_coordinates >= layer_boundary[:, None] - 1e-4,
            point_coordinates <= -layer_boundary[:, None] + 1e-4,
        ) & active[:, None]
        point_center = torch.zeros((count, 3), dtype=dtype, device=device)
        point_center[torch.arange(count, device=device), axes] = sides * CUBE_HALF_SIZE_MM
        expanded_points = base_points_t[None, :, :].expand(count, -1, -1)
        rotated_points = torch.matmul(expanded_points - point_center[:, None, :], rotation.transpose(1, 2)) + point_center[:, None, :]
        transformed_points = torch.where(point_layer_mask[:, :, None], rotated_points, expanded_points)

        expanded_normals = base_normals_t[None, :, :, :].expand(count, -1, -1, -1)
        rotated_normals = torch.einsum("bij,bnfj->bnfi", rotation, expanded_normals)
        transformed_normals = torch.where(point_layer_mask[:, :, None, None], rotated_normals, expanded_normals)
        front_facing = (
            base_face_mask_t[None, :, :, None]
            & (torch.einsum("ij,bnfj->bnfi", camera_rotation, transformed_normals)[..., 2:3] < 0.0)
        ).any(dim=2).squeeze(-1)

        triangle_coordinates = triangle_centers_t[:, axes].T
        triangle_boundary = CUBE_HALF_SIZE_MM - layer_counts * CUBE_LAYER_SIZE_MM + CUBE_LAYER_SIZE_MM / 2.0
        triangle_layer_mask = torch.where(
            sides[:, None] > 0.0,
            triangle_coordinates >= triangle_boundary[:, None] - 1e-4,
            triangle_coordinates <= -triangle_boundary[:, None] + 1e-4,
        ) & active[:, None]
        expanded_triangles = triangles_t[None, :, :, :].expand(count, -1, -1, -1)
        triangle_center_points = point_center[:, None, :]
        rotated_triangles = torch.matmul(
            expanded_triangles - triangle_center_points[:, :, None, :],
            rotation[:, None, :, :].transpose(2, 3),
        ) + triangle_center_points[:, :, None, :]
        transformed_triangles = torch.where(
            triangle_layer_mask[:, :, None, None],
            rotated_triangles,
            expanded_triangles,
        )
        camera_origin = -camera_rotation.transpose(0, 1) @ tvec.reshape(3, 1)
        camera_origin = camera_origin.reshape(1, 1, 3)
        directions = transformed_points - camera_origin
        safe_directions = torch.where(torch.abs(directions) <= 1e-12, 1.0, directions)
        bounds_min = -CUBE_HALF_SIZE_MM
        bounds_max = CUBE_HALF_SIZE_MM
        near = (bounds_min - camera_origin) / safe_directions
        far = (bounds_max - camera_origin) / safe_directions
        slab_near = torch.minimum(near, far)
        slab_far = torch.maximum(near, far)
        parallel_inside = (torch.abs(directions) > 1e-12) | (
            (camera_origin >= bounds_min) & (camera_origin <= bounds_max)
        )
        t_enter = torch.maximum(torch.zeros_like(slab_near[..., 0]), torch.amax(slab_near, dim=2))
        t_exit = torch.minimum(torch.ones_like(slab_far[..., 0]), torch.amin(slab_far, dim=2))
        ray_intersects_cube = torch.all(parallel_inside, dim=2) & (t_enter <= t_exit)
        direction_is_zero = torch.linalg.vector_norm(directions, dim=2) <= 1e-9
        hidden_by_cube = ray_intersects_cube & ~direction_is_zero & (t_enter > 1e-5) & (t_enter < 1.0 - 1e-4)

        edge_1 = transformed_triangles[:, :, 1] - transformed_triangles[:, :, 0]
        edge_2 = transformed_triangles[:, :, 2] - transformed_triangles[:, :, 0]
        ray_cross_edge_2 = torch.cross(directions[:, :, None, :], edge_2[:, None, :, :], dim=3)
        determinant = torch.sum(edge_1[:, None, :, :] * ray_cross_edge_2, dim=3)
        valid_determinant = torch.abs(determinant) > 1e-9
        inverse_determinant = torch.where(valid_determinant, 1.0 / determinant, torch.zeros_like(determinant))
        origin_to_vertex = -transformed_triangles[:, None, :, 0, :] + camera_origin[:, :, None, :]
        barycentric_u = inverse_determinant * torch.sum(origin_to_vertex * ray_cross_edge_2, dim=3)
        cross_q = torch.cross(origin_to_vertex, edge_1[:, None, :, :], dim=3)
        barycentric_v = inverse_determinant * torch.sum(directions[:, :, None, :] * cross_q, dim=3)
        distance = inverse_determinant * torch.sum(edge_2[:, None, :, :] * cross_q, dim=3)
        triangle_hits = (
            valid_determinant
            & (barycentric_u >= -1e-6)
            & (barycentric_v >= -1e-6)
            & (barycentric_u + barycentric_v <= 1.0 + 1e-6)
            & (distance > 1e-5)
            & (distance < 1.0 - 1e-4)
        )
        hidden_by_surface = torch.any(triangle_hits, dim=2)
        visible_points = front_facing & torch.where(active[:, None], ~hidden_by_surface, ~hidden_by_cube)
        projected = _torch_project_turn_points(
            torch,
            transformed_points,
            camera_rotation,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        observed = torch.as_tensor(image_points, dtype=dtype, device=device)
        distances = torch.linalg.vector_norm(observed[None, :, None, :] - projected[:, None, :, :], dim=3)
        distances = torch.where(visible_points[:, None, :], distances, torch.full_like(distances, float("inf")))

        fixed_mask = candidate.assignment_fixed_mask
        if fixed_mask is not None:
            fixed_mask = np.asarray(fixed_mask, dtype=bool).reshape(-1)
        if fixed_mask is None or len(fixed_mask) != len(image_points):
            fixed_mask = np.zeros(len(image_points), dtype=bool)
            fixed_mask[: min(len(candidate.image_points), len(image_points))] = True
        fixed_positions = np.flatnonzero(fixed_mask).astype(np.int32)
        if len(fixed_positions) != len(candidate.object_points):
            fixed_positions = np.empty((0,), dtype=np.int32)
        fixed_object_points = np.asarray(candidate.object_points, dtype=np.float64).reshape(-1, 3)
        fixed_projected_points = projected_points[fixed_positions].copy()
        existing_projected_classes = candidate.assignment_projected_class_ids
        if existing_projected_classes is not None:
            existing_projected_classes = np.asarray(existing_projected_classes, dtype=np.int32).reshape(-1)
            if len(existing_projected_classes) != len(image_points):
                existing_projected_classes = None
        fixed_projected_classes = (
            existing_projected_classes[fixed_positions].copy()
            if existing_projected_classes is not None
            else np.full(len(fixed_positions), -1, dtype=np.int32)
        )
        observed_classes = candidate.assignment_observed_class_ids
        if observed_classes is not None:
            observed_classes = np.asarray(observed_classes, dtype=np.int32).reshape(-1)
            if len(observed_classes) != len(image_points):
                observed_classes = None
        use_class_aware_assignment = PNP_ASSIGNMENT_MODE == "class-aware"
        if use_class_aware_assignment and observed_classes is not None:
            observed_class_t = torch.as_tensor(observed_classes, dtype=torch.long, device=device)
            mismatch = (
                (observed_class_t[None, :, None] >= 0)
                & (trial_classes[None, None, :] >= 0)
                & (observed_class_t[None, :, None] != trial_classes[None, None, :])
            )
            distances = torch.where(mismatch, torch.full_like(distances, float("inf")), distances)

        fixed_object_t = torch.as_tensor(fixed_object_points, dtype=dtype, device=device)
        fixed_layer_coordinates = fixed_object_t[:, axes].T if len(fixed_object_points) else torch.empty((count, 0), dtype=dtype, device=device)
        fixed_layer_mask = torch.where(
            sides[:, None] > 0.0,
            fixed_layer_coordinates >= layer_boundary[:, None] - 1e-4,
            fixed_layer_coordinates <= -layer_boundary[:, None] + 1e-4,
        ) & active[:, None] if len(fixed_object_points) else torch.empty((count, 0), dtype=torch.bool, device=device)
        preserve_fixed = torch.zeros((count, len(image_points)), dtype=torch.bool, device=device)
        if len(fixed_positions):
            preserve_fixed[:, fixed_positions] = ~fixed_layer_mask
        min_distances, columns = torch.min(distances, dim=2)
        selected_projected = torch.gather(projected, 1, columns[:, :, None].expand(-1, -1, 2))
        selected_objects = torch.gather(transformed_points, 1, columns[:, :, None].expand(-1, -1, 3))
        selected_classes = trial_classes[columns]
        if len(fixed_positions):
            fixed_row_mask = np.full(len(image_points), -1, dtype=np.int32)
            fixed_row_mask[fixed_positions] = np.arange(len(fixed_positions), dtype=np.int32)
            fixed_row_t = torch.as_tensor(np.maximum(fixed_row_mask, 0), dtype=torch.long, device=device)
            fixed_objects_by_row = fixed_object_t[fixed_row_t]
            fixed_projected_by_row_np = np.zeros((len(image_points), 2), dtype=np.float64)
            fixed_projected_by_row_np[fixed_positions] = fixed_projected_points
            fixed_projected_by_row = torch.as_tensor(
                fixed_projected_by_row_np,
                dtype=dtype,
                device=device,
            )
            fixed_classes_by_row_np = np.full(len(image_points), -1, dtype=np.int32)
            fixed_classes_by_row_np[fixed_positions] = fixed_projected_classes
            fixed_classes_by_row = torch.as_tensor(
                fixed_classes_by_row_np,
                dtype=torch.long,
                device=device,
            )
            fixed_distances_by_row = torch.linalg.vector_norm(
                observed - fixed_projected_by_row,
                dim=1,
            )
            assigned_objects = torch.where(preserve_fixed[:, :, None], fixed_objects_by_row[None, :, :], selected_objects)
            assigned_projected = torch.where(preserve_fixed[:, :, None], fixed_projected_by_row[None, :, :], selected_projected)
            assigned_classes = torch.where(preserve_fixed, fixed_classes_by_row[None, :], selected_classes)
            assigned_distances = torch.where(preserve_fixed, fixed_distances_by_row[None, :], min_distances)
        else:
            assigned_objects = selected_objects
            assigned_projected = selected_projected
            assigned_classes = selected_classes
            assigned_distances = min_distances

        sticker_coordinates = sticker_face_centers_t[:, axes].T
        sticker_boundary = CUBE_HALF_SIZE_MM - layer_counts * CUBE_LAYER_SIZE_MM + CUBE_LAYER_SIZE_MM / 2.0
        sticker_layer_mask = torch.where(
            sides[:, None] > 0.0,
            sticker_coordinates >= sticker_boundary[:, None] - 1e-4,
            sticker_coordinates <= -sticker_boundary[:, None] + 1e-4,
        ) & active[:, None]
        # Use the un-gated layer mask for observation accounting so static
        # baselines and turn trials treat centroid observations identically.
        sticker_observation_layer_mask = torch.where(
            sides[:, None] > 0.0,
            sticker_coordinates >= sticker_boundary[:, None] - 1e-4,
            sticker_coordinates <= -sticker_boundary[:, None] + 1e-4,
        )
        expanded_faces = sticker_faces_t[None, :, :, :].expand(count, -1, -1, -1)
        rotated_faces = torch.matmul(
            expanded_faces - point_center[:, None, None, :],
            rotation[:, None, :, :].transpose(2, 3),
        ) + point_center[:, None, None, :]
        transformed_faces = torch.where(sticker_layer_mask[:, :, None, None], rotated_faces, expanded_faces)
        projected_face_polygons = _torch_project_turn_points(
            torch,
            transformed_faces,
            camera_rotation,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        projected_face_centers = projected_face_polygons.mean(dim=2)
        transformed_face_normals = torch.einsum(
            "bij,bfj->bfi",
            rotation,
            sticker_normals_t[None, :, :].expand(count, -1, -1),
        )
        transformed_face_normals = torch.where(
            sticker_layer_mask[:, :, None],
            transformed_face_normals,
            sticker_normals_t[None, :, :].expand(count, -1, -1),
        )
        face_centers_camera = torch.einsum("ij,bfj->bfi", camera_rotation, transformed_faces.mean(dim=2)) + tvec
        face_normals_camera = torch.einsum("ij,bfj->bfi", camera_rotation, transformed_face_normals)
        visible_faces = torch.sum(face_normals_camera * -face_centers_camera, dim=2) > 0.0
        (
            sticker_mean, sticker_robust, sticker_matches,
            sticker_layer_matches,
        ) = _gpu_sticker_center_fit_metrics(
            candidate,
            torch,
            projected_face_centers,
            visible_faces,
            float(scale),
            float(assignment_error_limit),
            return_tensors=True,
            layer_mask=sticker_observation_layer_mask,
            return_layer_match_counts=True,
        )
        color_errors, color_matches = _gpu_sticker_color_fit_metrics(
            candidate,
            torch,
            projected_face_centers,
            projected_face_polygons,
            visible_faces,
            sticker_identity_indices,
            float(scale),
            float(assignment_error_limit),
            color_likelihoods,
            return_tensors=True,
        )
        (
            keypoint_mean_t, keypoint_robust_t, p90_t,
            inliers_t, layer_inliers_t, keypoint_valid_t,
        ) = (
            _gpu_keypoint_fit_metrics(
                torch, assigned_distances, assigned_objects, assigned_classes,
                axes, sides, layer_counts,
                observed_class_t if use_class_aware_assignment and observed_classes is not None else None,
                float(scale), float(assignment_error_limit),
            )
        )
        # All configuration statistics are computed on-device. Transfer the
        # compact vectors once; Python retains the existing ordering rules.
        score_vectors = torch.stack((
            sticker_mean, sticker_robust, sticker_matches.to(dtype=dtype),
            sticker_layer_matches.to(dtype=dtype),
            color_errors, color_matches.to(dtype=dtype),
            keypoint_mean_t, keypoint_robust_t, p90_t,
            inliers_t.to(dtype=dtype), layer_inliers_t.to(dtype=dtype),
            keypoint_valid_t.to(dtype=dtype),
        )).detach().cpu().numpy()
        sticker_mean = score_vectors[0].astype(np.float64, copy=False)
        sticker_robust = score_vectors[1].astype(np.float64, copy=False)
        sticker_matches = score_vectors[2].astype(np.int32, copy=False)
        sticker_layer_matches = score_vectors[3].astype(np.int32, copy=False)
        color_errors = score_vectors[4].astype(np.float64, copy=False)
        color_matches = score_vectors[5].astype(np.int32, copy=False)
        keypoint_means = score_vectors[6]
        keypoint_robust = score_vectors[7]
        p90_values = score_vectors[8]
        keypoint_inlier_counts = score_vectors[9].astype(np.int32, copy=False)
        keypoint_layer_inlier_counts = score_vectors[10].astype(np.int32, copy=False)
        keypoint_valid = score_vectors[11].astype(bool, copy=False)
        assigned_projected_cpu = (
            assigned_projected.detach().cpu().numpy()
            if materialize_projections else None
        )
        assigned_objects_cpu = (
            assigned_objects.detach().cpu().numpy()
            if materialize_projections else None
        )
        assigned_classes_cpu = (
            assigned_classes.detach().cpu().numpy()
            if materialize_projections else None
        )
        results: list[dict[str, object]] = []
        for trial_index, configuration in enumerate(configurations):
            axis, side, layer_count, _angle, is_baseline, rotations, label = configuration
            keypoint_error = float(keypoint_means[trial_index])
            robust_error = float(keypoint_robust[trial_index])
            p90_error = float(p90_values[trial_index])
            keypoint_inlier_count = int(keypoint_inlier_counts[trial_index])
            keypoint_layer_inlier_count = int(keypoint_layer_inlier_counts[trial_index])
            centroid_layer_match_count = int(sticker_layer_matches[trial_index])
            combined_valid = bool(keypoint_valid[trial_index]) or centroid_layer_match_count > 0
            # A missing keypoint component is allowed when centroid
            # observations validate the same layer; the available observation
            # terms then contribute identically to the combined score.
            if not np.isfinite(keypoint_error):
                keypoint_error = 0.0
            if not np.isfinite(robust_error):
                robust_error = 0.0
            if not np.isfinite(p90_error):
                p90_error = 0.0
            mean_error = keypoint_error
            inlier_count = keypoint_inlier_count + int(sticker_matches[trial_index])
            layer_inlier_count = keypoint_layer_inlier_count + centroid_layer_match_count
            combined_mean_error = keypoint_error + (
                PNP_TURN_MASK_CENTER_WEIGHT * float(sticker_mean[trial_index])
                if np.isfinite(sticker_mean[trial_index])
                else float("inf")
            )
            combined_mean_error += (
                PNP_TURN_COLOR_WEIGHT * float(color_errors[trial_index])
                if np.isfinite(color_errors[trial_index])
                else 0.0
            )
            combined_robust_error = robust_error + (
                PNP_TURN_MASK_CENTER_WEIGHT * float(sticker_robust[trial_index])
                if np.isfinite(sticker_robust[trial_index])
                else float("inf")
            )
            combined_robust_error += (
                PNP_TURN_COLOR_WEIGHT * float(color_errors[trial_index])
                if np.isfinite(color_errors[trial_index])
                else 0.0
            )
            if enforce_validity_gate and not combined_valid:
                combined_mean_error = float("inf")
                combined_robust_error = float("inf")
                p90_error = float("inf")
            results.append(
                {
                    "is_baseline": is_baseline,
                    "rank": (combined_mean_error, -float(inlier_count), p90_error, -float(layer_inlier_count)),
                    "label": label,
                    "robust_error": combined_robust_error,
                    "keypoint_error": keypoint_error,
                    "sticker_error": float(sticker_mean[trial_index]),
                    "combined_mean_error": combined_mean_error,
                    "keypoint_robust_error": robust_error,
                    "sticker_robust_error": float(sticker_robust[trial_index]),
                    "sticker_match_count": int(sticker_matches[trial_index]),
                    "sticker_layer_match_count": centroid_layer_match_count,
                    "color_error": float(color_errors[trial_index]),
                    "color_match_count": int(color_matches[trial_index]),
                    "keypoint_inlier_count": keypoint_inlier_count,
                    "keypoint_layer_inlier_count": keypoint_layer_inlier_count,
                    "inlier_count": inlier_count,
                    "layer_inlier_count": layer_inlier_count,
                    "p90_error": p90_error,
                    "mean_error": mean_error,
                    "layer_axis": axis,
                    "layer_side": side,
                    "layer_count": layer_count,
                    "layer_rotations": rotations,
                    "assignment_projected_points": (
                        assigned_projected_cpu[trial_index].copy()
                        if assigned_projected_cpu is not None
                        else assigned_projected[trial_index]
                    ),
                    "assignment_object_points": (
                        assigned_objects_cpu[trial_index].copy()
                        if assigned_objects_cpu is not None else assigned_objects[trial_index]
                    ),
                    "assignment_projected_class_ids": (
                        assigned_classes_cpu[trial_index].copy()
                        if assigned_classes_cpu is not None else assigned_classes[trial_index]
                    ),
                }
            )
        return results


    except (RuntimeError, ValueError, TypeError, IndexError, cv2.error):
        return None


def _build_turn_geometry_frame(
    candidate: StickerPnPCandidate,
    assignment_error_limit: float,
    scoring_device: object,
) -> dict[str, object] | None:
    """Cache the best partial geometric fit for each of the 12 layer choices."""

    trials = _score_turn_configurations_on_gpu(
        candidate,
        assignment_error_limit,
        scoring_device,
        color_likelihoods=None,
        materialize_projections=False,
        angle_values=turn_geometric_search_angles(),
        enforce_layer_observation_gate=False,
        enforce_validity_gate=False,
    )
    if not trials:
        return None
    options_by_layer: dict[tuple[int, int, int], dict[str, object]] = {}
    static_trials: list[dict[str, object]] = []
    for trial in trials:
        key = (
            int(trial["layer_axis"]),
            int(trial["layer_side"]),
            int(trial["layer_count"]),
        )
        if bool(trial["is_baseline"]):
            static_trials.append(trial)
        previous = options_by_layer.get(key)
        if previous is None or trial["rank"] < previous["rank"]:
            options_by_layer[key] = trial
    materialized_options: list[dict[str, object]] = []
    for key in sorted(options_by_layer):
        option = dict(options_by_layer[key])
        for field_name in (
            "assignment_projected_points",
            "assignment_object_points",
            "assignment_projected_class_ids",
        ):
            value = option.get(field_name)
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            option[field_name] = value
        materialized_options.append(option)
    options = tuple(materialized_options)
    if len(options) != 12 or not static_trials:
        return None
    return {
        "trials": tuple(trials),
        "options": options,
        "static_trials": tuple(static_trials),
    }


def _turn_color_frame_score_lower_bound(
    candidate: StickerPnPCandidate,
    geometry_options: tuple[dict[str, object], ...],
    assignment_error_limit: float,
    scoring_device: object,
    color_likelihoods: np.ndarray | None,
    *, projection_cache: dict | None = None,
) -> float:
    """Relax sticker permutation constraints to bound every state/offset.

    Each observation receives its maximum likelihood over all identities.
    Keep the real per-option coverage, uncertainty weights and geometric cost.
    Every feasible cube assignment is at least as costly as this relaxation.
    """

    if color_likelihoods is None:
        return min((float(option.get('combined_mean_error', -float('inf')))
                    for option in geometry_options), default=-float('inf'))
    likelihoods = np.asarray(color_likelihoods, dtype=np.float64)
    if likelihoods.ndim != 2 or not likelihoods.size or not np.isfinite(likelihoods).all():
        return -float('inf')
    optimistic = np.broadcast_to(likelihoods.max(axis=0, keepdims=True), likelihoods.shape).copy()
    trials = _score_turn_color_offsets_on_gpu(
        candidate, geometry_options, assignment_error_limit, scoring_device, optimistic, projection_cache=projection_cache,
    )
    if not trials:
        return -float('inf')
    lower = min(float(trial['state_score']) for trial in trials)
    if not np.isfinite(lower):
        return -float('inf')
    # Round down conservatively across accelerator dtypes/reduction order.
    slack = 1e-6 if str(scoring_device).startswith('mps') else 1e-10
    return lower - slack * (1.0 + abs(lower))


def _score_turn_color_offsets_on_gpu(
    candidate: StickerPnPCandidate,
    geometry_options: tuple[dict[str, object], ...],
    assignment_error_limit: float,
    scoring_device: object,
    color_likelihoods: np.ndarray | None,
    *, projection_cache: dict | None = None,
) -> list[dict[str, object]] | None:
    """Score only the -90/0/+90 color offsets for cached geometry options."""

    results = _score_turn_color_offsets_for_states_on_gpu(
        candidate, geometry_options, assignment_error_limit, scoring_device,
        color_likelihoods, (getattr(candidate, "cube_geometry_state", None),), projection_cache=projection_cache,
    )
    return results[0] if results is not None else None


def _score_turn_color_offsets_for_states_on_gpu(
    candidate: StickerPnPCandidate,
    geometry_options: tuple[dict[str, object], ...],
    assignment_error_limit: float,
    scoring_device: object,
    color_likelihoods: np.ndarray | None,
    geometry_states: tuple[CubeGeometryState | None, ...],
    *, projection_cache: dict | None = None,
) -> list[list[dict[str, object]]] | None:
    """Project each option once and score all frontier states on the GPU."""

    if not geometry_states:
        return []
    if not geometry_options:
        return [[] for _state in geometry_states]
    try:
        import torch

        canonical_identity_indices = _turn_gpu_static_geometry()[-1]
        sticker_identity_indices = np.stack([
            _turn_gpu_identity_indices_for_state(state, canonical_identity_indices)
            for state in geometry_states
        ])
        device = scoring_device
        dtype = torch.float32 if str(device).startswith("mps") else torch.float64
        configurations: list[tuple[int, float, int, int, int, float]] = []
        for option_index, option in enumerate(geometry_options):
            axis = int(option["layer_axis"])
            side = int(option["layer_side"])
            layer_count = int(option["layer_count"])
            rotations = tuple(option.get("layer_rotations", ()))
            base_angle = (
                float(rotations[0].angle_degrees)
                if rotations
                else 0.0
            )
            for offset in (-90.0, 0.0, 90.0):
                configurations.append(
                    (
                        option_index,
                        offset,
                        axis,
                        side,
                        layer_count,
                        base_angle + offset,
                    )
                )

        count = len(configurations)
        projection_key = (str(device), tuple(configurations), *(np.asarray(value, dtype=np.float64).tobytes()
                          for value in (candidate.rvec, candidate.tvec, candidate.camera_matrix, candidate.dist_coeffs)))
        projected = projection_cache.get('projected') if projection_cache is not None and projection_cache.get('key') == projection_key else None
        if projected is None:
            axes = torch.as_tensor(
                [item[2] for item in configurations],
                dtype=torch.long,
                device=device,
            )
            sides = torch.as_tensor(
                [item[3] for item in configurations],
                dtype=dtype,
                device=device,
            )
            layer_counts = torch.as_tensor(
                [item[4] for item in configurations],
                dtype=dtype,
                device=device,
            )
            angles = torch.as_tensor(
                [item[5] for item in configurations],
                dtype=dtype,
                device=device,
            )
            active = torch.as_tensor(
                [abs(item[5]) > 1e-6 for item in configurations],
                dtype=torch.bool,
                device=device,
            )
            (
                _base_points_t,
                _base_normals_t,
                _triangles_t,
                _triangle_centers_t,
                sticker_faces_t,
                sticker_face_centers_t,
                sticker_normals_t,
                _base_face_mask_t,
                _trial_classes,
            ) = _turn_gpu_static_tensors(str(device), dtype)
            camera_rotation_np, _ = cv2.Rodrigues(
                np.asarray(candidate.rvec, dtype=np.float64).reshape(3, 1)
            )
            camera_rotation = torch.as_tensor(
                camera_rotation_np,
                dtype=dtype,
                device=device,
            )
            tvec = torch.as_tensor(
                np.asarray(candidate.tvec, dtype=np.float64).reshape(1, 1, 3),
                dtype=dtype,
                device=device,
            )
            camera_matrix = torch.as_tensor(
                np.asarray(candidate.camera_matrix, dtype=np.float64),
                dtype=dtype,
                device=device,
            )
            dist_coeffs = torch.as_tensor(
                np.asarray(candidate.dist_coeffs, dtype=np.float64),
                dtype=dtype,
                device=device,
            )
            rotation = _torch_axis_rotation_matrices(
                torch,
                axes,
                sides * angles * (np.pi / 180.0),
            )
            sticker_coordinates = sticker_face_centers_t[:, axes].T
            sticker_boundary = (
                CUBE_HALF_SIZE_MM
                - layer_counts * CUBE_LAYER_SIZE_MM
                + CUBE_LAYER_SIZE_MM / 2.0
            )
            sticker_layer_mask = torch.where(
                sides[:, None] > 0.0,
                sticker_coordinates >= sticker_boundary[:, None] - 1e-4,
                sticker_coordinates <= -sticker_boundary[:, None] + 1e-4,
            ) & active[:, None]
            expanded_faces = sticker_faces_t[None, :, :, :].expand(
                count, -1, -1, -1
            )
            point_center = torch.zeros((count, 3), dtype=dtype, device=device)
            point_center[torch.arange(count, device=device), axes] = (
                sides * CUBE_HALF_SIZE_MM
            )
            rotated_faces = torch.matmul(
                expanded_faces - point_center[:, None, None, :],
                rotation[:, None, :, :].transpose(2, 3),
            ) + point_center[:, None, None, :]
            transformed_faces = torch.where(
                sticker_layer_mask[:, :, None, None],
                rotated_faces,
                expanded_faces,
            )
            projected_face_polygons = _torch_project_turn_points(
                torch,
                transformed_faces,
                camera_rotation,
                tvec,
                camera_matrix,
                dist_coeffs,
            )
            projected_face_centers = projected_face_polygons.mean(dim=2)
            transformed_face_normals = torch.einsum(
                "bij,bfj->bfi",
                rotation,
                sticker_normals_t[None, :, :].expand(count, -1, -1),
            )
            transformed_face_normals = torch.where(
                sticker_layer_mask[:, :, None],
                transformed_face_normals,
                sticker_normals_t[None, :, :].expand(count, -1, -1),
            )
            face_centers_camera = (
                torch.einsum(
                    "ij,bfj->bfi",
                    camera_rotation,
                    transformed_faces.mean(dim=2),
                )
                + tvec
            )
            face_normals_camera = torch.einsum(
                "ij,bfj->bfi",
                camera_rotation,
                transformed_face_normals,
            )
            visible_faces = torch.sum(
                face_normals_camera * -face_centers_camera,
                dim=2,
            ) > 0.0
            scale = projected_cube_diameter(candidate)
            if not np.isfinite(scale) or scale <= 0.0:
                return None
            projected = (projected_face_centers, projected_face_polygons, visible_faces, scale)
            if projection_cache is not None:
                projection_cache.update(key=projection_key, projected=projected, assignments={},
                                        builds=projection_cache.get('builds', 0) + 1)
        elif projection_cache is not None:
            projection_cache['hits'] = projection_cache.get('hits', 0) + 1
        projected_face_centers, projected_face_polygons, visible_faces, scale = projected
        color_errors, color_matches = _gpu_sticker_color_fit_metrics(
            candidate,
            torch,
            projected_face_centers,
            projected_face_polygons,
            visible_faces,
            sticker_identity_indices,
            float(scale),
            float(assignment_error_limit),
            color_likelihoods,
            return_tensors=True,
            assignment_cache=projection_cache.get("assignments") if projection_cache is not None else None,
        )
        color_vectors = torch.stack(
            (color_errors, color_matches.to(dtype=dtype))
        ).detach().cpu().numpy()
        color_errors_np = color_vectors[0].astype(np.float64, copy=False)
        color_matches_np = color_vectors[1].astype(np.int32, copy=False)
        # Without color inputs the shared metric returns one neutral vector.
        if color_errors_np.ndim == 1:
            color_errors_np = np.broadcast_to(color_errors_np, (len(geometry_states), count))
            color_matches_np = np.broadcast_to(color_matches_np, (len(geometry_states), count))

        results_by_state: list[list[dict[str, object]]] = []
        for state_index in range(len(geometry_states)):
            results: list[dict[str, object]] = []
            for index, (option_index, offset, axis, side, layer_count, _angle) in enumerate(configurations):
                option = geometry_options[option_index]
                geometry_rotations = tuple(option.get("layer_rotations", ()))
                state_moves: tuple[str, ...] = ()
                if abs(offset) > 1e-6:
                    notation = cube_rotation_to_notation(
                        CubeLayerRotation(axis, side, offset, layer_count)
                    )
                    if notation is not None:
                        state_moves = (notation,)
                color_error = float(color_errors_np[state_index, index])
                # The metric returns zero when color inputs are absent, but inf
                # when available observations cannot be matched to this trial.
                # Only the former is neutral evidence.
                color_cost = (
                    PNP_TURN_COLOR_WEIGHT * color_error
                    if np.isfinite(color_error)
                    else float("inf")
                )
                geometry_score = float(option["combined_mean_error"])
                combined_score = geometry_score + color_cost
                # Geometry and color are searched in separate stages, but the
                # resulting per-frame score used by state selection/lookahead
                # remains equally weighted.
                state_score = combined_score
                inlier_count = int(option.get("inlier_count", 0))
                layer_inlier_count = int(option.get("layer_inlier_count", 0))
                results.append(
                    {
                        "trial_id": f"{option_index}:{offset:+g}",
                        "is_baseline": bool(
                            option.get("is_baseline", False)
                            and not state_moves
                        ),
                        "geometry_option_index": option_index,
                        "layer_axis": axis,
                        "layer_side": side,
                        "layer_count": layer_count,
                        "color_offset": float(offset),
                        "state_moves": state_moves,
                        "layer_rotations": geometry_rotations,
                        "label": (
                            f"{option['label']} offset {offset:+.0f}"
                        ),
                        "global_mean_error": combined_score,
                        "combined_mean_error": combined_score,
                        # State selection and lookahead use this equal-weight
                        # geometric-plus-color score. The geometry and color
                        # calculations themselves remain decoupled.
                        "state_score": state_score,
                        "rank": (
                            state_score,
                            -float(inlier_count),
                            float(option.get("p90_error", float("inf"))),
                            -float(layer_inlier_count),
                        ),
                        "robust_error": float(option.get("robust_error", float("inf")))
                        + color_cost,
                        "keypoint_error": float(option.get("keypoint_error", float("inf"))),
                        "sticker_error": float(option.get("sticker_error", float("inf"))),
                        "keypoint_robust_error": float(option.get("keypoint_robust_error", float("inf"))),
                        "sticker_robust_error": float(option.get("sticker_robust_error", float("inf"))),
                        "sticker_match_count": int(option.get("sticker_match_count", 0)),
                        "sticker_layer_match_count": int(option.get("sticker_layer_match_count", 0)),
                        "color_error": color_error,
                        "color_match_count": int(color_matches_np[state_index, index]),
                        "keypoint_inlier_count": int(option.get("keypoint_inlier_count", 0)),
                        "keypoint_layer_inlier_count": int(option.get("keypoint_layer_inlier_count", 0)),
                        "inlier_count": inlier_count,
                        "layer_inlier_count": layer_inlier_count,
                        "p90_error": float(option.get("p90_error", float("inf"))),
                        "mean_error": float(option.get("mean_error", float("inf"))),
                        "assignment_projected_points": option["assignment_projected_points"],
                        "assignment_object_points": option["assignment_object_points"],
                        "assignment_projected_class_ids": option["assignment_projected_class_ids"],
                    }
                )
            results_by_state.append(results)
        return results_by_state
    except (RuntimeError, ValueError, TypeError, IndexError, cv2.error):
        return None


def turn_trial_exceeds_static_baseline(
    trial: dict[str, object],
    static_baseline_by_layer: dict[tuple[int, int, int], object],
) -> bool:
    """Return whether a turn is eligible against its static baseline.

    The pre-search assignment can contain observations that the exhaustive
    static rematch later recovers.  It is therefore not a valid baseline for
    accepting a turn.  Compare each tested turn with the static trial for the
    same axis, side, and layer width instead.

    A turn configuration must either explain strictly more unified inliers
    than static or provide a sufficiently strong color-likelihood improvement.
    Keypoint/centroid validity is still enforced by the fit scorer before a
    candidate can be selected.
    The 45-degree rule is applied only after selection, when deciding whether
    the selected fit commits a discrete cube-state update for the next frame.
    """

    if bool(trial["is_baseline"]):
        return False
    key = (
        int(trial["layer_axis"]),
        int(trial["layer_side"]),
        int(trial["layer_count"]),
    )
    baseline = static_baseline_by_layer.get(key)
    if baseline is None:
        return False

    # More unified observations waive the color-improvement requirement.
    # `inlier_count` includes both keypoint and sticker-centroid matches.
    baseline_inliers = int(
        baseline.get("inlier_count", 0) if isinstance(baseline, dict) else baseline
    )
    if int(trial.get("inlier_count", 0)) > baseline_inliers:
        return True

    # Otherwise compare negative-log likelihood, where lower is better: a
    # ratio of 0.5 means the turn's color error is at least 2x better.
    if not isinstance(baseline, dict):
        return False
    trial_color_error = float(trial.get("color_error", float("inf")))
    baseline_color_error = float(baseline.get("color_error", float("inf")))
    return (
        np.isfinite(trial_color_error)
        and np.isfinite(baseline_color_error)
        and baseline_color_error > 0.0
        and trial_color_error
        <= PNP_TURN_COLOR_RESCUE_MAX_RATIO * baseline_color_error
    )


def recover_misaligned_cube_layers(
    candidate: StickerPnPCandidate,
    assignment_error_limit: float,
    record_all_turn_candidates: bool = False,
    color_priors: dict[str, np.ndarray] | None = None,
    global_profiles: dict[str, GlobalColorProfile] | None = None,
    expected_color_by_label: dict[str, str] | None = None,
) -> StickerPnPCandidate:
    """Search layer turns using the combined local fit.

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
    search_rvec = np.asarray(candidate.rvec, dtype=np.float64).copy()
    search_tvec = np.asarray(candidate.tvec, dtype=np.float64).copy()
    color_likelihoods = turn_color_predictive_likelihoods(
        candidate,
        color_priors,
        global_profiles,
        expected_color_by_label,
    )
    if color_likelihoods is None:
        color_likelihoods = getattr(candidate, "turn_color_likelihoods", None)
    static_sticker_fit, _static_sticker_matches = sticker_mask_center_fit_metric(
        candidate,
        candidate.rvec,
        candidate.tvec,
        PNP_FINAL_ASSIGNMENT_ERROR_FRACTION * projected_cube_diameter(candidate),
    )
    static_color_fit, _static_color_matches = sticker_mask_color_turn_fit_metrics(
        candidate,
        candidate.rvec,
        candidate.tvec,
        (),
        PNP_FINAL_ASSIGNMENT_ERROR_FRACTION * projected_cube_diameter(candidate),
        color_likelihoods,
    )
    static_color_term = (
        PNP_TURN_COLOR_WEIGHT * static_color_fit
        if np.isfinite(static_color_fit)
        else 0.0
    )
    diagnostic_candidate = replace(
        candidate,
        turn_layer_scores=(),
        turn_fit_sticker_score=static_sticker_fit,
        turn_fit_color_score=static_color_fit,
        turn_color_likelihoods=color_likelihoods,
        turn_fit_unexplained_keypoints=unexplained_count,
        turn_fit_evaluated=True,
        turn_fit_static_scores=(
            static_fit,
            static_sticker_fit,
            static_fit
            + PNP_TURN_MASK_CENTER_WEIGHT * static_sticker_fit
            + static_color_term,
        ),
        turn_fit_articulated_scores=None,
        turn_fit_search_rvec=search_rvec,
        turn_fit_search_tvec=search_tvec,
    )
    # Always evaluate the complete turn configuration set for a valid pose.
    cube_scale = projected_cube_diameter(candidate)
    if not np.isfinite(cube_scale) or cube_scale <= 0.0:
        return diagnostic_candidate

    torch_device = _turn_torch_scoring_device("mps")
    if torch_device is None:
        raise RuntimeError("Move recognition requires an available Apple MPS device")
    ranked_trials = _score_turn_configurations_on_gpu(
        candidate, assignment_error_limit, torch_device,
        color_likelihoods=color_likelihoods,
        materialize_projections=record_all_turn_candidates,
    )
    if ranked_trials is None:
        raise RuntimeError("MPS move search could not score this frame")

    ordered_trials = sorted(ranked_trials, key=lambda trial: trial["rank"])
    static_baseline_by_layer = {
        (
            int(trial["layer_axis"]),
            int(trial["layer_side"]),
            int(trial["layer_count"]),
        ): trial
        for trial in ordered_trials
        if bool(trial["is_baseline"])
    }
    improving_trials = [
        trial
        for trial in ordered_trials
        if turn_trial_exceeds_static_baseline(trial, static_baseline_by_layer)
        and not turn_trial_has_invalid_color(trial)
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
            or (
                PNP_POSE_REPROJECTION_ERROR_REJECTION_ENABLED
                and
                PNP_POSE_REPROJECTION_ERROR_THRESHOLD is not None
                and float(best["robust_error"]) > PNP_POSE_REPROJECTION_ERROR_THRESHOLD
            )
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
        if bool(trial["is_baseline"])
        and int(trial["layer_count"]) == 1
    )
    recorded_trials = (
        ordered_trials
        if record_all_turn_candidates
        else ordered_trials[:PNP_TURN_DEBUG_CANDIDATE_COUNT]
    )
    if not record_all_turn_candidates:
        # Ranking does not need the projected coordinates. Transfer only the
        # displayed candidates and the selected fit, together in one batch.
        import torch

        needed_trials = list(recorded_trials)
        if best is not None and all(best is not trial for trial in needed_trials):
            needed_trials.append(best)
        if needed_trials:
            for key in (
                "assignment_projected_points", "assignment_object_points",
                "assignment_projected_class_ids",
            ):
                values = torch.stack(
                    [trial[key] for trial in needed_trials]
                ).detach().cpu().numpy()
                for trial, value in zip(needed_trials, values, strict=True):
                    trial[key] = value.copy()
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
            "trial_id": trial.get("trial_id"),
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
            "heatmap_layer_match_count": int(trial["sticker_layer_match_count"]),
            "color_error": float(trial["color_error"]),
            "color_match_count": int(trial["color_match_count"]),
            "keypoint_inlier_count": int(trial["keypoint_inlier_count"]),
            "keypoint_layer_inlier_count": int(trial["keypoint_layer_inlier_count"]),
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
        return replace(diagnostic_candidate, turn_color_unreliable=bool(
            ranked_trials and all(turn_trial_has_invalid_color(trial) for trial in ranked_trials)))

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
        turn_fit_color_score=float(best["color_error"]),
        turn_color_unreliable=False,
        turn_fit_articulated_scores=articulated_scores,
    )


def recover_misaligned_cube_layers_from_geometry(
    candidate: StickerPnPCandidate,
    assignment_error_limit: float,
    geometry_frame: dict[str, object],
    record_all_turn_candidates: bool = False,
    color_priors: dict[str, np.ndarray] | None = None,
    global_profiles: dict[str, GlobalColorProfile] | None = None,
    expected_color_by_label: dict[str, str] | None = None,
    previous_move: str | None = None,
    previous_states: tuple[tuple, ...] = (),
) -> StickerPnPCandidate:
    """Choose state offsets using a frame's already-cached geometry fits."""

    options = tuple(geometry_frame.get("options", ()))
    static_trials = tuple(geometry_frame.get("static_trials", ()))
    if not options or not static_trials:
        return replace(
            candidate,
            layer_rotations=(),
            turn_state_moves=(),
            turn_fit_articulated_scores=None,
            turn_fit_evaluated=False,
        )
    color_likelihoods = turn_color_predictive_likelihoods(
        candidate,
        color_priors,
        global_profiles,
        expected_color_by_label,
    )
    if color_likelihoods is None:
        color_likelihoods = getattr(candidate, "turn_color_likelihoods", None)
    scoring_device = _turn_torch_scoring_device("mps")
    if scoring_device is None:
        raise RuntimeError("Move recognition requires an available Apple MPS device")
    color_trials = _score_turn_color_offsets_on_gpu(
        candidate,
        options,
        assignment_error_limit,
        scoring_device,
        color_likelihoods,
    )
    if not color_trials:
        return replace(
            candidate,
            turn_color_likelihoods=color_likelihoods,
            turn_fit_evaluated=True,
            turn_state_moves=(),
            turn_fit_articulated_scores=None,
        )

    # History does not exclude a finite single-frame option or fallback.
    best_trial = min(
        (trial for trial in color_trials if np.isfinite(float(trial["state_score"]))
         and not turn_trial_has_invalid_color(trial)),
        key=lambda trial: (float(trial["state_score"])
                           + PNP_TURN_LOOKAHEAD_MOVE_COST * len(trial["state_moves"]), trial["rank"]),
        default=None,
    )
    if best_trial is None:
        return replace(
            candidate,
            layer_rotations=(),
            sticker_layer_rotations=(),
            turn_selected_trial_id=None,
            turn_color_offset_degrees=0.0,
            turn_color_unreliable=any(turn_trial_has_invalid_color(trial) for trial in color_trials),
            turn_candidate_diagnostics=tuple(dict(turn_trial_log_payload(trial),
                invalid_color=turn_trial_has_invalid_color(trial)) for trial in color_trials),
            turn_color_likelihoods=color_likelihoods,
            turn_fit_evaluated=True,
            turn_state_moves=(),
            turn_fit_articulated_scores=None,
        )
    best = None if bool(best_trial["is_baseline"]) else best_trial
    static_trial = min(
        (
            trial
            for trial in color_trials
            if bool(trial["is_baseline"])
        ),
        key=lambda trial: trial["rank"],
        default=None,
    )
    if static_trial is None:
        return replace(
            candidate,
            turn_color_likelihoods=color_likelihoods,
            turn_fit_evaluated=True,
            turn_state_moves=(),
            turn_fit_articulated_scores=None,
        )
    static_scores = (
        float(static_trial["keypoint_error"]),
        float(static_trial["sticker_error"]),
        float(static_trial["combined_mean_error"]),
    )
    layer_scores = tuple(
        (
            str(trial["label"]),
            float(trial["keypoint_error"]),
            float(trial["sticker_error"]),
            float(trial["combined_mean_error"]),
        )
        for trial in static_trials
    )
    articulated_scores = (
        (
            float(best["keypoint_error"]),
            float(best["sticker_error"]),
            float(best["combined_mean_error"]),
        )
        if best is not None
        else None
    )
    ordered_trials = sorted(color_trials, key=lambda trial: trial["rank"])
    recorded_trials = (
        ordered_trials
        if record_all_turn_candidates
        else ordered_trials[:PNP_TURN_DEBUG_CANDIDATE_COUNT]
    )
    if best is not None and all(best is not trial for trial in recorded_trials):
        recorded_trials.append(best)
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
            "trial_id": trial.get("trial_id"),
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
            "color_offset": float(trial["color_offset"]),
            "state_moves": list(trial["state_moves"]),
            "global_mean_error": float(trial["combined_mean_error"]),
            "state_score": float(trial["state_score"]),
            "global_robust_error": float(trial["robust_error"]),
            "keypoint_mean_error": float(trial["keypoint_error"]),
            "keypoint_robust_error": float(trial["keypoint_robust_error"]),
            "heatmap_mean_error": float(trial["sticker_error"]),
            "heatmap_robust_error": float(trial["sticker_robust_error"]),
            "heatmap_match_count": int(trial["sticker_match_count"]),
            "heatmap_layer_match_count": int(trial["sticker_layer_match_count"]),
            "color_error": float(trial["color_error"]),
            "color_match_count": int(trial["color_match_count"]),
            "keypoint_inlier_count": int(trial["keypoint_inlier_count"]),
            "keypoint_layer_inlier_count": int(trial["keypoint_layer_inlier_count"]),
            "inlier_count": int(trial["inlier_count"]),
            "layer_inlier_count": int(trial["layer_inlier_count"]),
        }
        for trial in recorded_trials
    )
    diagnostic_candidate = replace(
        candidate,
        turn_layer_scores=layer_scores,
        turn_candidate_scores=turn_candidate_scores,
        turn_candidate_debug=turn_candidate_debug,
        turn_candidate_diagnostics=turn_candidate_diagnostics,
        turn_fit_sticker_score=float(static_trial["sticker_error"]),
        turn_fit_color_score=float(static_trial["color_error"]),
        turn_color_likelihoods=color_likelihoods,
        turn_fit_unexplained_keypoints=int(
            getattr(candidate, "turn_fit_unexplained_keypoints", 0)
        ),
        turn_fit_evaluated=True,
        turn_fit_static_scores=static_scores,
        turn_fit_articulated_scores=articulated_scores,
        turn_fit_search_rvec=np.asarray(candidate.rvec, dtype=np.float64).copy(),
        turn_fit_search_tvec=np.asarray(candidate.tvec, dtype=np.float64).copy(),
        turn_state_moves=(),
        turn_color_unreliable=False,
        turn_color_offset_degrees=0.0,
    )
    if best is None:
        return diagnostic_candidate

    return _candidate_with_turn_trial(diagnostic_candidate, candidate, best)


def _candidate_with_turn_trial(
    diagnostic_candidate: StickerPnPCandidate,
    static_candidate: StickerPnPCandidate,
    trial: dict[str, object],
) -> StickerPnPCandidate:
    """Apply one cached geometry/color trial to a diagnostic candidate."""

    rotations = tuple(trial.get("layer_rotations", ()))
    sticker_rotations = rotations
    offset = float(trial.get("color_offset", 0.0))
    if abs(offset) > 1e-6:
        # Color scoring rotates the identities by partial angle + offset.
        # Keep the committed state before this move for the accumulator, and
        # render those identities with that same full rotation on this frame.
        base_angle = float(rotations[0].angle_degrees) if rotations else 0.0
        sticker_rotations = (CubeLayerRotation(
            int(trial["layer_axis"]), int(trial["layer_side"]),
            base_angle + offset, int(trial["layer_count"]),
        ),)
    object_points = apply_cube_layer_rotations(
        static_candidate.object_points,
        rotations,
    )
    try:
        projected_sticker_points, _ = cv2.projectPoints(
            object_points,
            static_candidate.rvec,
            static_candidate.tvec,
            static_candidate.camera_matrix,
            static_candidate.dist_coeffs,
        )
        projected_sticker_points = projected_sticker_points.reshape(-1, 2)
    except cv2.error:
        projected_sticker_points = static_candidate.projected_sticker_points
    articulated_scores = None if bool(trial.get("is_baseline", False)) else (
        float(trial.get("keypoint_error", float("inf"))),
        float(trial.get("sticker_error", float("inf"))),
        float(trial.get("combined_mean_error", float("inf"))),
    )
    return replace(
        diagnostic_candidate,
        # A different trial may have refined the diagnostic pose already.
        # This trial's cached assignments and color score use the static pose.
        rvec=np.asarray(static_candidate.rvec, dtype=np.float64).copy(),
        tvec=np.asarray(static_candidate.tvec, dtype=np.float64).copy(),
        pre_final_rvec=static_candidate.pre_final_rvec,
        pre_final_tvec=static_candidate.pre_final_tvec,
        final_solve_image_points=static_candidate.final_solve_image_points,
        object_points=object_points,
        projected_sticker_points=projected_sticker_points,
        assignment_object_points=trial.get("assignment_object_points"),
        assignment_projected_points=trial.get("assignment_projected_points"),
        assignment_projected_class_ids=trial.get("assignment_projected_class_ids"),
        layer_rotations=rotations,
        sticker_layer_rotations=sticker_rotations,
        turn_state_moves=tuple(trial.get("state_moves", ())),
        turn_color_unreliable=False,
        turn_fit_color_score=float(trial.get("color_error", diagnostic_candidate.turn_fit_color_score)),
        turn_selected_trial_id=trial.get("trial_id"),
        turn_color_offset_degrees=float(trial.get("color_offset", 0.0)),
        turn_fit_articulated_scores=articulated_scores,
    )


def _unreliable_color_hold(
    diagnostic_candidate: StickerPnPCandidate,
    static_candidate: StickerPnPCandidate,
) -> StickerPnPCandidate:
    """Keep static geometry/state when available color scoring fails."""

    held = _candidate_with_turn_trial(diagnostic_candidate, static_candidate, dict(
        trial_id=None, is_baseline=True, layer_rotations=(), state_moves=(), color_offset=0.0,
        assignment_object_points=static_candidate.assignment_object_points,
        assignment_projected_points=static_candidate.assignment_projected_points,
        assignment_projected_class_ids=static_candidate.assignment_projected_class_ids,
    ))
    return replace(held, turn_color_unreliable=True)


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


def turn_color_predictive_likelihoods(
    candidate: StickerPnPCandidate,
    color_priors: dict[str, np.ndarray] | None,
    global_profiles: dict[str, GlobalColorProfile] | None,
    expected_color_by_label: dict[str, str] | None = None,
) -> np.ndarray | None:
    """Return ``p(mask RGB | sticker identity prior)`` for turn scoring."""

    colors = tuple(getattr(candidate, "sticker_mask_colors", ()))
    centers = tuple(getattr(candidate, "sticker_mask_centers", ()))
    if not colors or not centers or global_profiles is None:
        return None
    count = min(len(colors), len(centers))
    if count <= 0:
        return None
    evidence = np.asarray(
        [
            _rgb_observation_color_evidence(color, global_profiles)
            for color in colors[:count]
        ],
        dtype=np.float64,
    )
    if not np.isfinite(evidence).all():
        return None
    evidence = np.clip(evidence, 1e-300, None)
    identities = canonical_sticker_identities()
    uniform_prior = np.full(
        len(GLOBAL_COLOR_NAMES),
        1.0 / len(GLOBAL_COLOR_NAMES),
        dtype=np.float64,
    )
    likelihoods = np.empty((len(identities), count), dtype=np.float64)
    for identity_index, identity in enumerate(identities):
        if expected_color_by_label is not None:
            expected = expected_color_by_label.get(identity.label)
            if expected in GLOBAL_COLOR_NAMES:
                expected_index = GLOBAL_COLOR_NAMES.index(expected)
                likelihoods[identity_index] = evidence[:, expected_index]
                continue
        prior = uniform_prior
        if color_priors is not None:
            prior_value = np.asarray(
                color_priors.get(identity.label, uniform_prior),
                dtype=np.float64,
            ).reshape(-1)
            if len(prior_value) == len(GLOBAL_COLOR_NAMES) and np.isfinite(prior_value).all():
                prior_total = float(np.sum(prior_value))
                if prior_total > 1e-12:
                    prior = prior_value / prior_total
        likelihoods[identity_index] = evidence @ prior
    return np.clip(likelihoods, 1e-300, None)


def sticker_mask_color_turn_fit_metrics(
    candidate: StickerPnPCandidate,
    rvec: np.ndarray,
    tvec: np.ndarray,
    layer_rotations: tuple[CubeLayerRotation, ...],
    max_error: float,
    color_likelihoods: np.ndarray | None,
) -> tuple[float, int]:
    """Score mask colors against identities carried by a tested turn."""

    if color_likelihoods is None:
        return 0.0, 0
    observed = tuple(getattr(candidate, "sticker_mask_centers", ()))
    uncertainties = np.asarray(
        getattr(candidate, "sticker_mask_center_uncertainties", ()),
        dtype=np.float64,
    ).reshape(-1)
    colors = tuple(getattr(candidate, "sticker_mask_colors", ()))
    count = min(len(observed), len(colors), len(color_likelihoods[0]))
    if count <= 0 or not np.isfinite(max_error) or max_error <= 0.0:
        return 0.0, 0
    trial_candidate = replace(
        candidate, layer_rotations=layer_rotations, sticker_layer_rotations=None,
    )
    try:
        projected_faces = projected_sticker_faces(trial_candidate, rvec, tvec)
    except (cv2.error, ValueError, FloatingPointError):
        return float("inf"), 0
    if not projected_faces:
        return float("inf"), 0
    projected_centers = np.asarray(
        [np.mean(np.asarray(face[2], dtype=np.float64), axis=0) for face in projected_faces],
        dtype=np.float64,
    )
    if not np.isfinite(projected_centers).all():
        return float("inf"), 0
    observed_points = np.asarray(observed[:count], dtype=np.float64).reshape(-1, 2)
    distances = np.linalg.norm(
        observed_points[:, None, :] - projected_centers[None, :, :],
        axis=2,
    )
    containing_faces: list[list[int]] = []
    for observed_point in observed_points:
        containing_faces.append(
            [
                face_index
                for face_index, (_identity, _depth, polygon) in enumerate(projected_faces)
                if cv2.pointPolygonTest(
                    np.asarray(polygon, dtype=np.float32),
                    (float(observed_point[0]), float(observed_point[1])),
                    False,
                ) >= 0.0
            ]
        )
    nearest_faces = np.full(count, -1, dtype=np.int32)
    nearest_distances = np.full(count, float("inf"), dtype=np.float64)
    for observation_index, face_indices in enumerate(containing_faces):
        if not face_indices:
            continue
        face_distances = distances[observation_index, face_indices]
        selected_offset = int(np.argmin(face_distances))
        nearest_faces[observation_index] = face_indices[selected_offset]
        nearest_distances[observation_index] = face_distances[selected_offset]
    finite = np.isfinite(nearest_distances) & (nearest_faces >= 0)
    if not np.any(finite):
        return float("inf"), 0
    identity_indices = {
        (identity.cubie_index, identity.axis, identity.side): index
        for index, identity in enumerate(canonical_sticker_identities())
    }
    face_identity_indices = np.asarray(
        [
            identity_indices[
                (face[0].cubie_index, face[0].axis, face[0].side)
            ]
            for face in projected_faces
        ],
        dtype=np.int32,
    )
    selected_identity_indices = np.zeros(count, dtype=np.int32)
    selected_identity_indices[finite] = face_identity_indices[nearest_faces[finite]]
    likelihood_matrix = np.asarray(color_likelihoods, dtype=np.float64)
    selected_likelihoods = np.full(count, np.nan, dtype=np.float64)
    selected_likelihoods[finite] = likelihood_matrix[
        selected_identity_indices[finite],
        np.flatnonzero(finite),
    ]
    if len(uncertainties) != count or not np.isfinite(uncertainties[:count]).all():
        uncertainties = np.ones(count, dtype=np.float64)
    else:
        uncertainties = np.maximum(uncertainties[:count], 1.0)
    weights = 1.0 / np.square(uncertainties)
    finite &= np.isfinite(selected_likelihoods) & (selected_likelihoods > 0.0)
    if not np.any(finite):
        return float("inf"), 0
    log_likelihoods = np.log(np.clip(selected_likelihoods, 1e-300, None))
    color_error = -float(
        np.sum(weights[finite] * log_likelihoods[finite])
        / max(float(np.sum(weights[finite])), 1e-12)
    )
    match_count = int(
        np.count_nonzero(finite & (nearest_distances <= max_error))
    )
    return color_error, match_count


def turn_layer_observation_count(
    candidate: StickerPnPCandidate,
    axis: int,
    side: int,
    layer_count: int,
) -> int:
    """Count reliable keypoint and centroid observations fully in a layer."""

    keypoint_count = 0
    image_points = getattr(candidate, "assignment_image_points", None)
    projected_points = getattr(candidate, "assignment_projected_points", None)
    object_points = getattr(candidate, "assignment_object_points", None)
    if (
        image_points is not None
        and projected_points is not None
        and object_points is not None
    ):
        image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
        projected_points = np.asarray(projected_points, dtype=np.float64).reshape(-1, 2)
        object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
        if (
            len(image_points) == len(projected_points) == len(object_points)
            and len(image_points)
            and np.isfinite(image_points).all()
            and np.isfinite(projected_points).all()
            and np.isfinite(object_points).all()
        ):
            distances = np.linalg.norm(image_points - projected_points, axis=1)
            keypoint_count = int(
                np.count_nonzero(
                    cube_layer_mask(object_points, axis, side, layer_count)
                    & np.isfinite(distances)
                )
            )

    observed_centers = np.asarray(
        getattr(candidate, "sticker_mask_centers", ()),
        dtype=np.float64,
    ).reshape(-1, 2)
    centroid_count = 0
    if len(observed_centers) and np.isfinite(observed_centers).all():
        trial_candidate = replace(
            candidate,
            layer_rotations=tuple(getattr(candidate, "layer_rotations", ())),
        )
        try:
            projected_faces = projected_sticker_faces(
                trial_candidate,
                candidate.rvec,
                candidate.tvec,
            )
        except (cv2.error, ValueError, FloatingPointError):
            projected_faces = []
        layer_faces = [
            (face_index, face)
            for face_index, face in enumerate(projected_faces)
            if (
                sticker_geometry_key(candidate, face[0])[0][int(axis)] >= 3 - layer_count
                if side > 0
                else sticker_geometry_key(candidate, face[0])[0][int(axis)] < layer_count
            )
        ]
        pairs: list[tuple[float, int, int]] = []
        for observation_index, center in enumerate(observed_centers):
            for face_index, (_identity, _depth, polygon) in layer_faces:
                if cv2.pointPolygonTest(
                    np.asarray(polygon, dtype=np.float32),
                    (float(center[0]), float(center[1])),
                    False,
                ) >= 0.0:
                    projected_center = np.mean(
                        np.asarray(polygon, dtype=np.float64),
                        axis=0,
                    )
                    pairs.append(
                        (
                            float(np.linalg.norm(center - projected_center)),
                            observation_index,
                            face_index,
                        )
                    )
        used_observations: set[int] = set()
        used_faces: set[int] = set()
        for _distance, observation_index, face_index in sorted(pairs):
            if observation_index in used_observations or face_index in used_faces:
                continue
            used_observations.add(observation_index)
            used_faces.add(face_index)
        centroid_count = len(used_observations)
    return keypoint_count + int(centroid_count)


def draw_sticker_mask_heatmap(
    image: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
    predicted: tuple[list[np.ndarray], object, np.ndarray, np.ndarray | None]
    | tuple[list[np.ndarray], object, np.ndarray]
    | None,
    sticker_mask_erosion: int = 0,
    show_sticker_identities: bool = True,
    show_sticker_centroids: bool = True,
    show_sticker_mask: bool = True,
) -> np.ndarray:
    """Draw a transparent red heatmap centered on each sticker-mask peak."""

    output = image.copy()
    if predicted is None:
        return output
    local_masks = predicted[0]
    x1, y1, x2, y2 = (int(value) for value in roi_xyxy)
    roi_height = y2 - y1
    roi_width = x2 - x1
    if roi_height <= 0 or roi_width <= 0:
        return output

    if show_sticker_mask:
        heatmap = np.zeros((roi_height, roi_width), dtype=np.float32)
        for mask in sticker_segmentation_masks_in_roi(
            predicted,
            (roi_height, roi_width),
            sticker_mask_erosion,
        ):
            distance = cv2.distanceTransform(
                np.asarray(mask, dtype=np.uint8),
                cv2.DIST_L2,
                5,
            )
            peak = float(distance.max())
            if peak > 0.0:
                heatmap = np.maximum(heatmap, distance / peak)

        if bool(np.any(heatmap > 0.0)):
            # Keep the boundary nearly transparent and make the deepest interior
            # pixels visibly red. The power curve concentrates color near the
            # distance-transform peak instead of tinting the whole mask uniformly.
            alpha = np.square(heatmap) * 0.82
            roi = output[y1:y2, x1:x2]
            red = np.zeros_like(roi, dtype=np.float32)
            red[..., 2] = 255.0
            blended = (
                roi.astype(np.float32) * (1.0 - alpha[..., None])
                + red * alpha[..., None]
            )
            output[y1:y2, x1:x2] = np.clip(blended, 0.0, 255.0).astype(np.uint8)

    if show_sticker_centroids:
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


def sticker_mask_rgb_observations(
    image: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
    predicted: tuple[list[np.ndarray], object, np.ndarray, np.ndarray | None]
    | tuple[list[np.ndarray], object, np.ndarray]
    | None,
    sticker_mask_erosion: int = 0,
) -> list[tuple[int, np.ndarray]]:
    """Return one robust RGB sample for each segmentation mask."""

    if predicted is None:
        return []
    x1, y1, x2, y2 = (int(value) for value in roi_xyxy)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return []
    masks = sticker_segmentation_masks_in_roi(
        predicted,
        roi.shape[:2],
        sticker_mask_erosion,
    )
    reliable_pixels = _reliable_color_mask(roi)
    observations: list[tuple[int, np.ndarray]] = []
    for mask_index, mask in enumerate(masks):
        sample_mask = mask & reliable_pixels
        if not bool(sample_mask.any()):
            sample_mask = mask
        pixels = roi[sample_mask]
        if len(pixels) == 0:
            continue
        rgb = np.median(pixels[:, ::-1].astype(np.float64), axis=0)
        observations.append((mask_index, np.round(rgb).clip(0, 255).astype(np.uint8)))
    return observations


def draw_classified_sticker_masks(
    image: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
    predicted: tuple[list[np.ndarray], object, np.ndarray, np.ndarray | None]
    | tuple[list[np.ndarray], object, np.ndarray]
    | None,
    global_profiles: dict[str, GlobalColorProfile] | None = None,
    sticker_mask_erosion: int = 0,
    opacity: float = 0.72,
    show_sticker_identities: bool = True,
    show_sticker_centroids: bool = True,
    source_image: np.ndarray | None = None,
) -> np.ndarray:
    """Color each segmentation mask from its own image pixels.

    This intentionally does not use projected sticker identities or pose. It
    is a diagnostic view of the segmentation masks themselves.
    """

    if predicted is None:
        return image.copy()
    x1, y1, x2, y2 = (int(value) for value in roi_xyxy)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return image.copy()
    masks = sticker_segmentation_masks_in_roi(
        predicted,
        roi.shape[:2],
        sticker_mask_erosion,
    )
    if not masks:
        return image.copy()
    if global_profiles is None:
        global_profiles = {
            color_name: GlobalColorProfile(
                color_name=color_name,
                prior=pure_rgb_color_profile(color_name),
                fitted=None,
                sample_count=0,
            )
            for color_name in GLOBAL_COLOR_NAMES
        }

    output = image.copy()
    output_roi = output[y1:y2, x1:x2]
    opacity = float(np.clip(opacity, 0.0, 1.0))
    mask_observations = dict(
        sticker_mask_rgb_observations(
            image if source_image is None else source_image,
            roi_xyxy,
            predicted,
            sticker_mask_erosion,
        )
    )
    for mask_index, mask in enumerate(masks):
        rgb = mask_observations.get(mask_index)
        if rgb is None:
            continue
        probabilities = rgb_observation_color_likelihoods(rgb, global_profiles)
        color_name = max(probabilities, key=probabilities.get)
        color_bgr = np.asarray(PURE_COLOR_RGB[color_name][::-1], dtype=np.float32)
        blended = (
            output_roi.astype(np.float32) * (1.0 - opacity)
            + color_bgr * opacity
        ).astype(np.uint8)
        output_roi[mask] = blended[mask]

        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(output_roi, contours, -1, (0, 0, 0), 2, cv2.LINE_AA)
        if show_sticker_centroids and show_sticker_identities:
            ys, xs = np.nonzero(mask)
            if len(xs):
                center = (int(round(float(xs.mean()) + x1)), int(round(float(ys.mean()) + y1)))
                confidence = probabilities[color_name]
                cv2.putText(
                    output,
                    f"M{mask_index} {color_name} {confidence:.2f}",
                    (center[0] + 5, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    output,
                    f"M{mask_index} {color_name} {confidence:.2f}",
                    (center[0] + 5, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
    return output


def draw_sticker_mask_overlay(
    image: np.ndarray,
    roi_xyxy: tuple[int, int, int, int],
    predicted: tuple[list[np.ndarray], object, np.ndarray, np.ndarray | None]
    | tuple[list[np.ndarray], object, np.ndarray]
    | None,
    *,
    global_profiles: dict[str, GlobalColorProfile] | None = None,
    debug_color_masks: bool = False,
    sticker_mask_erosion: int = 0,
    show_sticker_identities: bool = True,
    show_sticker_centroids: bool = True,
    show_sticker_mask: bool = True,
    source_image: np.ndarray | None = None,
) -> np.ndarray:
    """Choose direct mask colors or the ordinary segmentation heatmap."""

    if debug_color_masks:
        return draw_classified_sticker_masks(
            image,
            roi_xyxy,
            predicted,
            global_profiles=global_profiles,
            sticker_mask_erosion=sticker_mask_erosion,
            show_sticker_identities=show_sticker_identities,
            show_sticker_centroids=show_sticker_centroids,
            source_image=source_image,
        )
    return draw_sticker_mask_heatmap(
        image,
        roi_xyxy,
        predicted,
        sticker_mask_erosion=sticker_mask_erosion,
        show_sticker_identities=show_sticker_identities,
        show_sticker_centroids=show_sticker_centroids,
        show_sticker_mask=show_sticker_mask,
    )


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
    observed_center_image_points = np.asarray(
        center_image_points,
        dtype=np.float64,
    ).copy()
    try:
        projected_center_points, _ = cv2.projectPoints(
            center_object_points,
            candidate.rvec,
            candidate.tvec,
            candidate.camera_matrix,
            candidate.dist_coeffs,
        )
    except cv2.error:
        return candidate
    projected_center_points = projected_center_points.reshape(-1, 2)
    center_uncertainties = np.maximum(
        np.asarray(_center_uncertainties, dtype=np.float64).reshape(-1),
        1.0,
    )
    if len(center_uncertainties) != len(center_image_points):
        center_uncertainties = np.ones(len(center_image_points), dtype=np.float64)
    center_weights = np.clip(
        PNP_MASK_CENTER_SOLVE_WEIGHT / np.square(center_uncertainties),
        0.0,
        PNP_MASK_CENTER_SOLVE_WEIGHT,
    )
    # Feed solvePnP a tempered observation: a clean heatmap center retains
    # useful pull, while an unreliable centroid is moved toward the current
    # pose prediction and cannot drag the solve as strongly.
    center_image_points = projected_center_points + center_weights[:, None] * (
        center_image_points - projected_center_points
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
        projected_centers - observed_center_image_points,
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
    record_all_turn_candidates: bool = False,
    recover_layer_turns: bool = False,
    turn_color_priors: dict[str, np.ndarray] | None = None,
    turn_global_profiles: dict[str, GlobalColorProfile] | None = None,
    turn_expected_color_by_label: dict[str, str] | None = None,
    turn_geometry_frame: dict[str, object] | None = None,
    turn_previous_move: str | None = None,
    turn_previous_states: tuple[tuple, ...] = (),
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
        if turn_geometry_frame is None:
            torch_device = _turn_torch_scoring_device("mps")
            if torch_device is None:
                raise RuntimeError(
                    "Move recognition requires an available Apple MPS device"
                )
            turn_geometry_frame = _build_turn_geometry_frame(
                candidate,
                assignment_error_limit,
                torch_device,
            )
        if turn_geometry_frame is None:
            return candidate
        candidate = recover_misaligned_cube_layers_from_geometry(
            candidate,
            assignment_error_limit,
            turn_geometry_frame,
            record_all_turn_candidates=record_all_turn_candidates,
            color_priors=turn_color_priors,
            global_profiles=turn_global_profiles,
            expected_color_by_label=turn_expected_color_by_label,
            previous_move=turn_previous_move,
            previous_states=turn_previous_states,
        )
    else:
        # Make the static-pose contract explicit even if a caller passes a
        # candidate that was previously used by the turn-recovery path.
        candidate = replace(
            candidate,
            layer_rotations=(),
            sticker_layer_rotations=None,
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
                rvec=np.asarray(candidate.rvec, dtype=np.float64).reshape(3, 1).copy(),
                tvec=np.asarray(candidate.tvec, dtype=np.float64).reshape(3, 1).copy(),
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


def sticker_mask_pose_geometry(
    local_masks: list[np.ndarray] | tuple[np.ndarray, ...] | None,
    roi_xyxy: tuple[int, int, int, int] | None,
    sticker_mask_erosion: int = 0,
) -> tuple[
    tuple[np.ndarray | None, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[float, ...],
]:
    """Build mask geometry needed by the later pose-refinement stages."""

    if not local_masks or roi_xyxy is None:
        return (), (), (), (), ()
    contours_by_mask = tuple(
        sticker_contour_to_frame(local_mask, roi_xyxy)
        for local_mask in local_masks
    )
    contours = tuple(
        contour for contour in contours_by_mask if contour is not None
    )
    boundary_samples = tuple(
        sample
        for contour in contours
        if (sample := resample_closed_boundary(contour)).size
    )
    centers, uncertainties = sticker_mask_center_observations_to_frame(
        local_masks,
        roi_xyxy,
        sticker_mask_erosion,
    )
    return contours_by_mask, contours, boundary_samples, centers, uncertainties


def restore_sticker_mask_pose_geometry(
    candidate: StickerPnPCandidate,
    local_masks: list[np.ndarray] | tuple[np.ndarray, ...] | None,
    roi_xyxy: tuple[int, int, int, int] | None,
    sticker_mask_erosion: int = 0,
    geometry: tuple[
        tuple[np.ndarray | None, ...],
        tuple[np.ndarray, ...],
        tuple[np.ndarray, ...],
        tuple[np.ndarray, ...],
        tuple[float, ...],
    ] | None = None,
) -> StickerPnPCandidate:
    """Attach deferred mask geometry before cached pose refinement."""

    if geometry is None:
        geometry = sticker_mask_pose_geometry(
            local_masks,
            roi_xyxy,
            sticker_mask_erosion,
        )
    (
        contours_by_mask,
        contours,
        boundary_samples,
        centers,
        uncertainties,
    ) = geometry
    if not contours and not centers:
        return candidate
    mask_index = candidate.sticker_mask_index
    sticker_contour = (
        contours_by_mask[mask_index]
        if mask_index is not None
        and 0 <= mask_index < len(contours_by_mask)
        else candidate.sticker_contour
    )
    return replace(
        candidate,
        sticker_contour=sticker_contour,
        sticker_contours=contours,
        sticker_boundary_samples=boundary_samples,
        sticker_mask_centers=centers,
        sticker_mask_center_uncertainties=uncertainties,
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
    prepare_mask_geometry: bool = True,
    sticker_mask_colors: tuple[np.ndarray, ...] | None = None,
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
    if prepare_mask_geometry:
        (
            sticker_contours_by_mask,
            sticker_contours,
            sticker_boundary_samples,
            sticker_mask_centers,
            sticker_mask_center_uncertainties,
        ) = sticker_mask_pose_geometry(
            local_masks,
            roi_xyxy,
            sticker_mask_erosion,
        )
    else:
        sticker_contours_by_mask = ()
        sticker_contours = ()
        sticker_boundary_samples = ()
        sticker_mask_centers = ()
        sticker_mask_center_uncertainties = ()

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
            solved_candidates = solve_sticker_pnp_candidates(
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
                sticker_mask_colors,
            )
            candidates.extend(solved_candidates)
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
    max_core_displacement_mm: float = POSE_MOTION_DISPLACEMENT_TOLERANCE_MM,
) -> StickerPnPCandidate | None:
    """Select a pose using the shared fit gate and translation continuity.

    Finite-error candidates remain eligible while the outright acceptance
    gate is disabled. If any candidates meet the shared fit threshold, only
    those candidates participate in temporal tie-breaking; the pose with the
    smallest translation delta from the previous accepted frame wins. Visual
    fit remains the deterministic fallback when no previous pose is available
    and breaks exact translation ties.
    """

    if not candidates:
        return None
    scored = [
        (candidate, pose_hypothesis_visual_score(candidate))
        for candidate in candidates
    ]
    accepted = [
        item for item in scored
        if pose_reprojection_error_is_acceptable(item[0])
    ]
    if not accepted:
        return None
    if predicted_pose is not None:
        _predicted_rvec, predicted_tvec = predicted_pose
        predicted_core = np.asarray(predicted_tvec, dtype=np.float64).reshape(3)
        tolerance = max(float(max_core_displacement_mm), 0.0)
        accepted = [
            item
            for item in accepted
            if np.linalg.norm(
                np.asarray(item[0].tvec, dtype=np.float64).reshape(3)
                - predicted_core
            )
            <= tolerance
        ]
        if not accepted:
            return None
    thresholded = [
        item
        for item in accepted
        if pose_reprojection_fit_metric(item[0])
        <= PNP_POSE_REPROJECTION_ERROR_THRESHOLD
    ]
    # The threshold narrows temporal tie-breaking when possible, but does not
    # reject the frame while the outright acceptance gate is disabled.
    ranking_pool = thresholded or accepted
    previous_pose = None
    if previous_candidates:
        previous_candidate = previous_candidates[-1]
        if hasattr(previous_candidate, "tvec"):
            previous_pose = (
                getattr(previous_candidate, "rvec", np.zeros((3, 1))),
                previous_candidate.tvec,
            )
    if previous_pose is None:
        previous_pose = predicted_pose
    if previous_pose is None:
        return min(ranking_pool, key=lambda item: item[1])[0]
    _previous_rvec, previous_tvec = previous_pose
    previous_translation = np.asarray(previous_tvec, dtype=np.float64).reshape(3)
    return min(
        ranking_pool,
        key=lambda item: (
            float(
                np.linalg.norm(
                    np.asarray(item[0].tvec, dtype=np.float64).reshape(3)
                    - previous_translation
                )
            ),
            item[1],
        ),
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
        if layer_count < 3 and side * original_center[axis] < layer_boundary - 1e-4:
            continue
        rotation_vector = np.zeros((3, 1), dtype=np.float64)
        rotation_vector[axis, 0] = (
            side * np.deg2rad(float(layer_rotation.angle_degrees))
        )
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pivot = np.zeros(3, dtype=np.float64)
        if layer_count < 3:
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


def sticker_geometry_key(
    candidate: StickerPnPCandidate,
    identity: StickerIdentity,
) -> tuple[tuple[int, int, int], int, int]:
    """Return an identity's committed location before candidate rotations."""

    geometry_state = getattr(candidate, "cube_geometry_state", None)
    if geometry_state is not None:
        try:
            return geometry_state.position_for(identity.label)
        except KeyError:
            pass
    return (identity.cubie_index, identity.axis, identity.side)


def physical_sticker_rotations(candidate: StickerPnPCandidate) -> tuple[CubeLayerRotation, ...]:
    """Return rotations carrying identities, including any discrete offset."""

    rotations = getattr(candidate, "sticker_layer_rotations", None)
    return tuple(candidate.layer_rotations) if rotations is None else tuple(rotations)


def projected_sticker_faces(
    candidate: StickerPnPCandidate,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> list[tuple[StickerIdentity, float, np.ndarray]]:
    """Project visible stickers, retaining their persistent physical identities."""

    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    visible_faces: list[tuple[StickerIdentity, float, np.ndarray]] = []
    identities = canonical_sticker_identities()
    identity_by_key = {
        (identity.cubie_index, identity.axis, identity.side): identity
        for identity in identities
    }
    geometry_state = getattr(candidate, "cube_geometry_state", None)
    if geometry_state is None:
        # Preserve the historical deterministic traversal order for callers
        # that do not participate in stateful fitting.
        entries = (
            (
                identity_by_key[(cubie_index, axis, side)],
                (cubie_index, axis, side),
            )
            for x_index in range(3)
            for y_index in range(3)
            for z_index in range(3)
            for cubie_index in ((x_index, y_index, z_index),)
            for axis in range(3)
            for side in (-1, 1)
            if cubie_index[axis] == (0 if side < 0 else 2)
        )
    else:
        entries = (
            (identity, sticker_geometry_key(candidate, identity))
            for identity in identities
        )
    for identity, current_key in entries:
        cubie_index, axis, side = current_key
        current_center = cubelet_center_object_point(*cubie_index)
        face_points = cubelet_face_object_points(current_center, axis, side)
        transformed_center, transformed_face = apply_cubelet_layer_rotations(
            current_center,
            face_points,
            physical_sticker_rotations(candidate),
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
    cubie_index, axis, side = sticker_geometry_key(candidate, identity)
    center = cubelet_center_object_point(*cubie_index)
    face = cubelet_face_object_points(center, axis, side)
    transformed_center, transformed_face = apply_cubelet_layer_rotations(
        center,
        face,
        physical_sticker_rotations(candidate),
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
    intersection_masks: dict[str, np.ndarray] | None = None,
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
        intersection = (projected_face_owners == face_index) & sticker_mask
        sample_mask = intersection & reliable_mask
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
        if intersection_masks is not None:
            intersection_masks[identity.label] = intersection
    return observations


def draw_projected_classified_masks(
    image: np.ndarray, source_image: np.ndarray,
    roi_xyxy: tuple[int, int, int, int], predicted: object,
    candidate: StickerPnPCandidate | None,
    global_profiles: dict[str, GlobalColorProfile],
    *, sticker_mask_erosion: int = 0, opacity: float = 0.72,
    max_samples_per_sticker: int = 128, min_samples_per_sticker: int = 4,
) -> np.ndarray:
    """Color projected sticker/segmentation intersections without identity priors."""
    masks: dict[str, np.ndarray] = {}
    samples = sample_projected_sticker_colors(
        source_image, roi_xyxy, predicted, candidate,
        max_samples_per_sticker=max_samples_per_sticker,
        min_samples_per_sticker=min_samples_per_sticker,
        sticker_mask_erosion=sticker_mask_erosion, intersection_masks=masks,
    )
    output = image.copy()
    x1, y1, x2, y2 = roi_xyxy
    output_roi = output[y1:y2, x1:x2]
    opacity = float(np.clip(opacity, 0.0, 1.0))
    for label, rgb in samples.items():
        likelihoods = rgb_observation_color_likelihoods(rgb, global_profiles)
        color = max(likelihoods, key=likelihoods.get)
        mask = masks[label]
        bgr = np.asarray(PURE_COLOR_RGB[color][::-1], dtype=np.float32)
        output_roi[mask] = np.clip(
            output_roi[mask].astype(np.float32) * (1.0 - opacity) + bgr * opacity,
            0, 255,
        ).astype(np.uint8)
    return output


def save_sticker_rgb_html_plot(
    trace: StickerColorTrace,
    output_path: Path,
    *,
    title: str,
    normalize_rgb: bool = False,
    point_size: float = 5.0,
    global_profiles: dict[str, GlobalColorProfile] | None = None,
    global_observations: dict[str, list[tuple[str, int, np.ndarray]]] | None = None,
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
    if global_profiles is None or global_observations is None:
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
                legendgrouptitle={"text": "Whole colors and regressions"},
            )
        )
        fitted = _global_profile_reference(global_profiles[color_name])
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
                name=(f"{color_name} global profile regression" if global_profiles[color_name].fitted is not None
                      else f"{color_name} global profile prior"),
                legendgroup="global-colors",
                legendgrouptitle={"text": "Whole colors and regressions"},
                showlegend=True,
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
            "label": "All stickers and colors",
            "method": "update",
            "args": [
                {"visible": [True] * plot_trace_count},
                {"title": f"{title} — All stickers and colors"},
            ],
        },
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
    global_profiles: dict[str, GlobalColorProfile] | None = None,
    global_observations: dict[str, list[tuple[str, int, np.ndarray]]] | None = None,
) -> None:
    save_sticker_rgb_html_plot(
        trace,
        output_path,
        title=title,
        normalize_rgb=primary_normalized,
        point_size=point_size,
        global_profiles=global_profiles,
        global_observations=global_observations,
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
        global_profiles=global_profiles,
        global_observations=global_observations,
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
        if isinstance(value, np.ndarray):
            return value.tolist()
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
    search_diagnostics = getattr(candidate, "turn_search_diagnostics", None) if candidate is not None else None
    if isinstance(search_diagnostics, dict) and search_diagnostics.get("roots"):
        # These are the actual root trials the search considered, even if the
        # single-frame fit returned early or kept only a diagnostic subset.
        candidates = [dict(
            root["trial"], trial_id=root["trial_id"],
            search_status=root["status"], last_retained_depth=root["last_retained_depth"],
            eliminated_depth=root.get("eliminated_depth"),
            best_sequence_score=root.get("best_sequence_score"),
            pruning_counts=root["pruning_counts"],
            pruning_examples=root.get("pruning_examples", []),
            root_return_examples=root.get("root_return_examples", []),
        ) for root in search_diagnostics["roots"]]
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
    color_likelihoods = (
        getattr(candidate, "turn_color_likelihoods", None)
        if candidate is not None
        else None
    )
    color_identity_labels = (
        [identity.label for identity in canonical_sticker_identities()]
        if color_likelihoods is not None
        else None
    )
    color_observation_centers = (
        getattr(candidate, "sticker_mask_centers", ())
        if candidate is not None and color_likelihoods is not None
        else None
    )
    rotations_for_log = tuple(candidate.layer_rotations) if candidate is not None else ()
    selected_rotations = (
        [
            {
                "axis": int(rotation.axis),
                "side": int(rotation.side),
                "angle_degrees": float(rotation.angle_degrees),
                "layer_count": int(rotation.layer_count),
            }
            for rotation in rotations_for_log
        ]
        if candidate is not None
        else []
    )
    search_rvec = (
        getattr(candidate, "turn_fit_search_rvec", None)
        if candidate is not None
        else None
    )
    search_tvec = (
        getattr(candidate, "turn_fit_search_tvec", None)
        if candidate is not None
        else None
    )
    search_pose = (
        {
            "rvec": search_rvec,
            "tvec": search_tvec,
            "camera_matrix": getattr(candidate, "camera_matrix", None),
            "dist_coeffs": getattr(candidate, "dist_coeffs", None),
            "image_size": getattr(candidate, "image_size", None),
            "sticker_type": getattr(candidate, "sticker_type", None),
            "assignment_image_points": getattr(
                candidate,
                "assignment_image_points",
                None,
            ),
            "assignment_observed_class_ids": getattr(
                candidate,
                "assignment_observed_class_ids",
                None,
            ),
            "assignment_fixed_mask": getattr(
                candidate,
                "assignment_fixed_mask",
                None,
            ),
            "sticker_mask_centers": getattr(
                candidate,
                "sticker_mask_centers",
                (),
            ),
            "sticker_mask_center_uncertainties": getattr(
                candidate,
                "sticker_mask_center_uncertainties",
                (),
            ),
            "cube_geometry_state": (
                getattr(candidate, "cube_geometry_state").to_json()
                if getattr(candidate, "cube_geometry_state", None) is not None
                else None
            ),
        }
        if candidate is not None
        else None
    )
    selected_pose = None
    if candidate is not None and isinstance(search_pose, dict):
        selected_pose = dict(search_pose)
        selected_pose.update(
            {
                "rvec": getattr(candidate, "rvec", search_pose.get("rvec")),
                "tvec": getattr(candidate, "tvec", search_pose.get("tvec")),
                "assignment_projected_points": getattr(
                    candidate,
                    "assignment_projected_points",
                    None,
                ),
            }
        )
    record = {
        "log_schema_version": 2,
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
        "turn_fit_color_score": float(
            getattr(candidate, "turn_fit_color_score", float("inf"))
            if candidate is not None
            else float("inf")
        ),
        "turn_color_identity_labels": color_identity_labels,
        "turn_color_observation_centers": color_observation_centers,
        "turn_color_likelihoods": color_likelihoods,
        "static_fit_scores": static_scores,
        "selected_fit_scores": articulated_scores,
        "lookahead_action_score": float(
            getattr(candidate, "turn_fit_lookahead_action_score", float("inf"))
            if candidate is not None
            else float("inf")
        ),
        "lookahead_static_score": float(
            getattr(candidate, "turn_fit_lookahead_static_score", float("inf"))
            if candidate is not None
            else float("inf")
        ),
        "lookahead_action_future_scores": (
            list(getattr(candidate, "turn_fit_lookahead_action_future_scores", ()))
            if candidate is not None
            else []
        ),
        "lookahead_static_future_scores": (
            list(getattr(candidate, "turn_fit_lookahead_static_future_scores", ()))
            if candidate is not None
            else []
        ),
        "lookahead_rejected": bool(
            getattr(candidate, "turn_fit_lookahead_rejected", False)
            if candidate is not None
            else False
        ),
        "lookahead_frames": PNP_TURN_LOOKAHEAD_FRAMES,
        "lookahead_beam_width": None,
        "lookahead_state_batch_size": PNP_TURN_LOOKAHEAD_STATE_BATCH_SIZE,
        "lookahead_state_change_limit": PNP_TURN_LOOKAHEAD_STATE_CHANGE_LIMIT,
        "lookahead_state_evaluation_budget": PNP_TURN_LOOKAHEAD_STATE_BUDGET,
        "color_evidence_unreliable": bool(getattr(candidate, "turn_color_unreliable", False)),
        "search": search_diagnostics,
        "selected_trial_id": getattr(candidate, "turn_selected_trial_id", None) if candidate is not None else None,
        "selected": (
            next((trial["label"] for trial in candidates
                  if trial.get("trial_id") is not None
                  and trial["trial_id"] == getattr(candidate, "turn_selected_trial_id", None)),
                 candidate_turn_debug_label(candidate))
            if candidate is not None
            else None
        ),
        "turn_recovery_mode": "geometric-options-color-rollout",
        "selected_layer": None,
        "selected_layer_rotations": selected_rotations,
        "selected_sticker_layer_rotations": [
            {
                "axis": int(rotation.axis),
                "side": int(rotation.side),
                "angle_degrees": float(rotation.angle_degrees),
                "layer_count": int(rotation.layer_count),
            }
            for rotation in (physical_sticker_rotations(candidate) if candidate is not None else ())
        ],
        "selected_committed_moves": list(completed_candidate_moves(candidate)),
        "selected_color_offset_degrees": float(
            getattr(candidate, "turn_color_offset_degrees", 0.0)
            if candidate is not None
            else 0.0
        ),
        "cube_geometry_state_before": (
            getattr(candidate, "cube_geometry_state").to_json()
            if candidate is not None
            and getattr(candidate, "cube_geometry_state", None) is not None
            else None
        ),
        "search_pose": search_pose,
        "selected_pose": selected_pose,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log_file:
        json.dump(json_safe(record), log_file, separators=(",", ":"))
        log_file.write("\n")


def projected_turn_color_likelihoods_by_identity(
    candidate: StickerPnPCandidate,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> dict[str, float]:
    """Map contained observed mask likelihoods onto projected identities."""

    likelihoods = getattr(candidate, "turn_color_likelihoods", None)
    centers = tuple(getattr(candidate, "sticker_mask_centers", ()))
    if likelihoods is None or not centers:
        return {}
    likelihoods = np.asarray(likelihoods, dtype=np.float64)
    if likelihoods.ndim != 2 or likelihoods.shape[1] == 0:
        return {}
    try:
        projected_faces = projected_sticker_faces(candidate, rvec, tvec)
    except cv2.error:
        return {}
    if not projected_faces:
        return {}
    projected_centers = np.asarray(
        [np.mean(np.asarray(face[2], dtype=np.float64), axis=0) for face in projected_faces],
        dtype=np.float64,
    )
    observed_centers = np.asarray(
        centers[: likelihoods.shape[1]],
        dtype=np.float64,
    ).reshape(-1, 2)
    if len(observed_centers) == 0 or not np.isfinite(observed_centers).all():
        return {}
    identity_indices = {
        (identity.cubie_index, identity.axis, identity.side): index
        for index, identity in enumerate(canonical_sticker_identities())
    }
    values: dict[str, tuple[float, float]] = {}
    for face_index, (identity, _depth, polygon) in enumerate(projected_faces):
        identity_index = identity_indices.get(
            (identity.cubie_index, identity.axis, identity.side)
        )
        if identity_index is None:
            continue
        face_center = projected_centers[face_index]
        distances = np.linalg.norm(observed_centers - face_center, axis=1)
        inside_indices = [
            observation_index
            for observation_index, observed_center in enumerate(observed_centers)
            if cv2.pointPolygonTest(
                np.asarray(polygon, dtype=np.float32),
                tuple(float(value) for value in observed_center),
                False,
            ) >= 0.0
        ]
        if not inside_indices:
            continue
        candidate_indices = inside_indices
        observation_index = min(
            candidate_indices,
            key=lambda index: float(distances[index]),
        )
        value = float(likelihoods[identity_index, observation_index])
        distance = float(distances[observation_index])
        if not np.isfinite(value) or not np.isfinite(distance):
            continue
        previous = values.get(identity.label)
        if previous is None or distance < previous[0]:
            values[identity.label] = (distance, value)
    return {label: value for label, (_distance, value) in values.items()}


def draw_projected_sticker_identity_labels(
    image: np.ndarray,
    candidate: StickerPnPCandidate,
    rvec: np.ndarray,
    tvec: np.ndarray,
    image_offset: tuple[int, int] = (0, 0),
    show_observation_likelihood: bool = True,
    center_offset: tuple[float, float] = (0.0, 0.0),
    likelihood_by_identity: dict[str, float] | None = None,
) -> int:
    """Draw each visible identity and its current color likelihood."""

    try:
        projected_faces = projected_sticker_faces(candidate, rvec, tvec)
    except cv2.error:
        return 0
    height, width = image.shape[:2]
    image_offset_array = np.asarray(image_offset, dtype=np.float32).reshape(1, 2)
    label_offset = np.asarray(center_offset, dtype=np.float32).reshape(2)
    displayed_likelihoods = (
        likelihood_by_identity
        if likelihood_by_identity is not None
        else (
            projected_turn_color_likelihoods_by_identity(candidate, rvec, tvec)
            if show_observation_likelihood
            else {}
        )
    )
    drawn = 0
    for identity, _depth, polygon in projected_faces:
        center = np.mean(polygon, axis=0) + image_offset_array[0] + label_offset
        if not np.isfinite(center).all():
            continue
        x, y = np.round(center).astype(np.int32)
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        label = identity.label.split(" ", maxsplit=1)[0]
        likelihood = displayed_likelihoods.get(identity.label)
        lines = [
            f"{label} {likelihood:.2f}"
            if show_observation_likelihood and likelihood is not None
            else label
        ]
        text_sizes = [
            cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            for line in lines
        ]
        text_width = max(size[0][0] for size in text_sizes)
        text_height = sum(size[0][1] + size[1] for size in text_sizes)
        origin = (
            int(x - text_width / 2),
            int(y - text_height / 2 + text_sizes[0][0][1]),
        )
        top_left = (origin[0] - 2, origin[1] - text_sizes[0][0][1] - 2)
        bottom_right = (origin[0] + text_width + 2, origin[1] + text_height + 2)
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), -1, cv2.LINE_AA)
        line_y = origin[1]
        for line, ((_, line_height), line_baseline) in zip(lines, text_sizes, strict=True):
            cv2.putText(
                image,
                line,
                (origin[0], line_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            line_y += line_height + line_baseline
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
    # Mask centers are recorded in the source camera frame.  Assign them to
    # physical identities using the source pose first; the diagnostic cube is
    # projected at a different depth only after that identity mapping.
    turn_likelihoods_by_identity = projected_turn_color_likelihoods_by_identity(
        candidate,
        candidate.rvec,
        candidate.tvec,
    )
    # Only display the turn-search value here.  It is the requested
    # p(observation | identity prior), rather than the separate classifier
    # quantity p(observation | color) stored in sticker_likelihoods.
    displayed_likelihoods = turn_likelihoods_by_identity or None
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
            show_observation_likelihood=displayed_likelihoods is not None,
            likelihood_by_identity=displayed_likelihoods,
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

    # Keep identity/probability annotations confined to the classified
    # projected-cube panel.  The camera-view pose should remain unobstructed.
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
        threshold_text = (
            f"{PNP_POSE_REPROJECTION_ERROR_THRESHOLD:.3f}"
            if PNP_POSE_REPROJECTION_ERROR_THRESHOLD is not None
            else "off"
        )
        label = (
            f"{pose_label} sticker {candidate.sticker_type} | "
            f"kp {candidate.sticker_average_confidence:.2f} | "
            f"fit {fit_text}/{threshold_text} | "
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
        selected_turn_label = candidate_turn_debug_label(candidate)
        turn_y = 110 if show_status else 66
        turn_diagnostics = getattr(candidate, "turn_candidate_diagnostics", ())
        if turn_diagnostics:
            displayed_trials = turn_diagnostics[:PNP_TURN_OVERLAY_CANDIDATE_COUNT]
            for index, trial in enumerate(displayed_trials):
                label = str(trial.get("label", index))
                selected_marker = "*" if label == selected_turn_label else " "
                rank_score = float(trial.get("global_mean_error", float("inf")))
                accept_score = float(trial.get("global_robust_error", float("inf")))
                inlier_count = int(trial.get("inlier_count", 0))
                rank_text = f"{rank_score:.3f}" if np.isfinite(rank_score) else "inf"
                accept_text = (
                    f"{accept_score:.3f}" if np.isfinite(accept_score) else "inf"
                )
                turn_label = (
                    f"turn {selected_marker}{label} | in {inlier_count} | "
                    f"rank {rank_text} | accept {accept_text}"
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
        else:
            displayed_trials = turn_candidate_scores[:PNP_TURN_OVERLAY_CANDIDATE_COUNT]
            for index, (label, mean_score, inlier_count, p90_error, mean_error) in enumerate(
                displayed_trials
            ):
                selected_marker = "*" if label == selected_turn_label else " "
                turn_label = (
                    f"turn {selected_marker}{label} | in {inlier_count} | "
                    f"accept {mean_score:.3f} | kp p90 {p90_error:.3f} | "
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
    threshold_text = (
        f"{PNP_POSE_REPROJECTION_ERROR_THRESHOLD:.3f}"
        if PNP_POSE_REPROJECTION_ERROR_THRESHOLD is not None
        else "off"
    )
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
        f"REPROJ fit {fit_text}/{threshold_text}"
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
        output_image = image.copy()
    else:
        output = image.copy().astype(np.float32)
        base_pixels = image[changed].astype(np.float32)
        pose_pixels = rendered[changed].astype(np.float32)
        output[changed] = (1.0 - opacity) * base_pixels + opacity * pose_pixels
        output_image = np.clip(output, 0.0, 255.0).astype(np.uint8)
    label = "REJECTED POSE (diagnostic only)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    cv2.putText(
        output_image,
        label,
        (8, 92),
        font,
        scale,
        (0, 0, 255),
        thickness,
        cv2.LINE_AA,
    )
    return output_image


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
    symmetry_tracker: CubeSymmetryTracker | None,
    pose_hypothesis_buffer: PoseHypothesisBuffer | None,
) -> None:
    """Drop temporal state after the tracked cube is no longer observable."""

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
    global_profiles: dict[str, GlobalColorProfile] | None = None,
    global_color_samples: list[tuple[str, int, np.ndarray]] | None = None,
    move_accumulator: MoveAccumulator | None = None,
) -> tuple[np.ndarray, str, int]:
    height, width = frame.shape[:2]

    def overlay_sticker_masks(output: np.ndarray, pose: StickerPnPCandidate | None = None) -> np.ndarray:
        if getattr(args, "show_colored_masks", False) and cached_final_only and global_profiles is not None:
            return draw_projected_classified_masks(
                output, frame, roi_xyxy, sticker_masks, pose, global_profiles,
                sticker_mask_erosion=getattr(args, "sticker_mask_erosion", 0),
                max_samples_per_sticker=getattr(args, "color_samples_per_sticker", 128),
                min_samples_per_sticker=getattr(args, "color_min_samples_per_sticker", 4),
            )
        return draw_sticker_mask_overlay(
            output, roi_xyxy, sticker_masks, global_profiles=global_profiles,
            debug_color_masks=getattr(args, "debug_color_masks", False),
            sticker_mask_erosion=getattr(args, "sticker_mask_erosion", 0),
            show_sticker_identities=getattr(args, "show_sticker_identities", True),
            show_sticker_centroids=getattr(args, "show_keypoints", True),
            show_sticker_mask=getattr(args, "show_sticker_mask", False),
            source_image=frame,
        )

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
        if candidate_only:
            return frame, "none", 0
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
        if candidate_only:
            return frame, roi_source, 0
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
        sticker_masks, result = predict_models_parallel_cpu(
            sticker_seg_model, pose_model, model_roi, args,
        )
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
    if (
        global_color_samples is not None
        and getattr(args, "debug_color_masks", False)
        and not cached_final_only
    ):
        sample_frame_index = 0 if frame_index is None else int(frame_index)
        for mask_index, rgb in sticker_mask_rgb_observations(
            frame,
            roi_xyxy,
            sticker_masks,
            getattr(args, "sticker_mask_erosion", 0),
        ):
            global_color_samples.append(
                (f"M{mask_index}", sample_frame_index, rgb)
            )
    if candidate_only and len(local_points) == 0:
        return frame, roi_source, 0
    if cached_final_only and neural_data is not None and neural_data.pose_ready:
        pnp_candidate = neural_data.pose_candidate
        diagnostic_candidate = getattr(
            neural_data,
            "pose_diagnostic_candidate",
            None,
        )
        output = frame.copy()
        point_count = int(len(local_points))
        if roi_xyxy is not None:
            frame_points, _ = local_to_frame_points(
                local_points,
                roi_xyxy,
                args.view_size,
            )
            if getattr(args, "show_keypoints", True):
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
        display_sticker_color_names = sticker_color_names
        if move_accumulator is not None:
            display_sticker_color_names = (
                move_accumulator.current_color_names(sticker_color_names)
                or display_sticker_color_names
            )
            move_accumulator.observe(
                pnp_candidate,
                displacement_candidate=diagnostic_candidate,
                sticker_color_names=sticker_color_names,
            )
        if args.show_pnp and pnp_candidate is not None:
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
        elif diagnostic_candidate is not None and (
            args.show_pnp or getattr(args, "recognize_moves", False)
        ):
            output = draw_translucent_sticker_pnp_candidate(
                output,
                diagnostic_candidate,
            )
            output = draw_reprojection_error_overlay(
                output,
                diagnostic_candidate,
            )
        output = overlay_sticker_masks(output, pnp_candidate)
        if sticker_color_names is not None:
            output = draw_classified_cube_diagnostic_panel(
                output,
                pnp_candidate,
                display_sticker_color_names or sticker_color_names,
                sticker_likelihoods=sticker_likelihoods,
                orientation_axis_length=args.orientation_axis_length,
                show_sticker_identities=getattr(args, "show_sticker_identities", True),
            )
        if move_accumulator is not None:
            # The accumulator was updated once above, before the diagnostic
            # drawing.  Do not observe the same frame again: a second observe
            # would compare the pose with itself and overwrite the displayed
            # displacement with 0 mm.
            output = draw_move_accumulator_overlay(output, move_accumulator)
        log_turn_frame(pnp_candidate)
        return output, roi_source, point_count
    if len(local_points) == 0:
        roi_tracker.reset()
        reset_pose_temporal_state(
            symmetry_tracker,
            pose_hypothesis_buffer,
        )
        output = frame.copy()
        output = draw_context(output, roi_xyxy, candidate, roi_source, 0, args.show_status)
        output = overlay_sticker_masks(output)
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
    frame_points, local_roi_points = local_to_frame_points(local_points, roi_xyxy, args.view_size)
    output = frame if candidate_only else frame.copy()
    point_count = int(len(frame_points))
    if not candidate_only:
        if getattr(args, "show_keypoints", True):
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
        output = draw_context(
            output,
            roi_xyxy,
            candidate,
            roi_source,
            point_count,
            args.show_status,
        )
    if neural_only:
        # The foreground pass is deliberately limited to detector/model
        # output. Pose candidates are selected by the asynchronous consumer.
        output = overlay_sticker_masks(output)
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

    # Move recognition is a per-frame computation.  It must still run when
    # pose drawing is disabled, because ``--recognize-moves`` is an analysis
    # request rather than a rendering option.
    should_generate_candidate = pnp_candidate is None and (
        args.show_pnp or recognize_moves or candidate_only
    )
    if should_generate_candidate:
        mask_color_entries = sticker_mask_rgb_observations(
            frame,
            roi_xyxy,
            sticker_masks,
            args.sticker_mask_erosion,
        )
        sticker_mask_colors = tuple(
            rgb for _mask_index, rgb in mask_color_entries
        )
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
            prepare_mask_geometry=not candidate_only,
            sticker_mask_colors=sticker_mask_colors,
        )
        prepared_candidates = prepare_pose_hypotheses(
            current_candidates,
            None,
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
        return output, roi_source, point_count
    refined_candidates: tuple[StickerPnPCandidate, ...] = ()
    diagnostic_candidate: StickerPnPCandidate | None = None
    if pnp_candidate is not None and (args.show_pnp or recognize_moves):
        hypotheses_to_refine = (
            candidate_hypotheses
            if candidate_hypotheses
            else (pnp_candidate,)
        )
        refined: list[StickerPnPCandidate] = []
        for hypothesis in hypotheses_to_refine:
            refined_candidate = solve_pnp_from_low_error_assignments(
                hypothesis,
                # Refine every retained hypothesis as a static cube first.
                recover_layer_turns=False,
                record_all_turn_candidates=False,
            )
            refined.append(refined_candidate)
        refined_candidates = tuple(refined)
        selected_candidate = select_pose_hypothesis(
            refined_candidates,
            None,
            pose_hypothesis_buffer.previous_candidates()
            if pose_hypothesis_buffer is not None
            else (),
            max_core_displacement_mm=POSE_MOTION_DISPLACEMENT_TOLERANCE_MM,
        )
        if selected_candidate is not None:
            if recognize_moves:
                geometry_state = (
                    move_accumulator._geometry_state.copy()
                    if move_accumulator is not None
                    else CubeGeometryState.identity()
                )
                selected_candidate = replace(
                    selected_candidate,
                    cube_geometry_state=geometry_state,
                )
                # This is the only move-search entry point in the live path:
                # the static pose is already selected and finalized.
                selected_candidate = solve_pnp_from_low_error_assignments(
                    selected_candidate,
                    recover_layer_turns=True,
                    record_all_turn_candidates=turn_candidate_log is not None,
                    turn_previous_move=(
                        move_accumulator.moves[-1]
                        if move_accumulator is not None and move_accumulator.moves else None
                    ),
                    turn_previous_states=(
                        move_accumulator.recent_geometry_states() if move_accumulator is not None else ()
                    ),
                )
            pnp_candidate = replace(
                selected_candidate, orientation_anchor=orientation_anchor_for_candidate(selected_candidate)
            )
            if symmetry_tracker is not None:
                # Recover orientation only after root selection, move fitting,
                # and pose selection have completed.
                pnp_candidate = canonicalize_pose_symmetry(
                    pnp_candidate,
                    symmetry_tracker,
                )
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
        display_sticker_color_names = sticker_color_names
        if move_accumulator is not None:
            display_sticker_color_names = (
                move_accumulator.current_color_names(sticker_color_names)
                or display_sticker_color_names
            )
            move_accumulator.observe(
                pnp_candidate,
                displacement_candidate=diagnostic_candidate,
                sticker_color_names=sticker_color_names,
            )
        if pnp_candidate is not None:
            record_color_observations(pnp_candidate)
            if args.show_pnp:
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
                        display_sticker_color_names or sticker_color_names,
                        sticker_likelihoods=sticker_likelihoods,
                        orientation_axis_length=args.orientation_axis_length,
                        show_sticker_identities=getattr(args, "show_sticker_identities", True),
                    )
        elif diagnostic_candidate is not None and (
            args.show_pnp or recognize_moves
        ):
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
    output = overlay_sticker_masks(output, pnp_candidate)
    if neural_data is not None:
        neural_data.pose_candidate = pnp_candidate
        if refined_candidates:
            neural_data.pose_candidates = refined_candidates
        neural_data.pose_ready = True
    if move_accumulator is not None:
        # Rejected diagnostic fits are for visualization only.  They must not
        # advance the accumulated move history or become the next fit prior.
        output = draw_move_accumulator_overlay(output, move_accumulator)
    if pose_hypothesis_buffer is not None:
        # Keep only the accepted pose for temporal tie-breaking.  The buffer
        # still records empty frames so the last accepted pose remains the
        # prior when a frame is temporarily rejected.
        pose_hypothesis_buffer.append(
            (pnp_candidate,) if pnp_candidate is not None else (),
            frame_index,
        )
    log_turn_frame(pnp_candidate)
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
            None,
            None,
            sticker_color_trace,
            None,
            neural_data=record,
            candidate_only=True,
        )
        processed_frames += 1
        if progress is not None:
            progress.update("initial pose candidates", processed_frames, total_frames)


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
    global_color_samples: list[tuple[str, int, np.ndarray]] | None = None,
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
        CombinedHeadlessProgress(("neural inference", "initial pose candidates"))
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
                neural_cache=cache,
                neural_only=True,
                global_color_samples=global_color_samples,
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
                progress.update("neural inference", frame_index, total_frames)
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
    global_profiles: dict[str, GlobalColorProfile] | None = None,
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
    move_accumulator = MoveAccumulator()
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
                global_profiles=global_profiles,
                move_accumulator=move_accumulator,
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


def parse_turn_search_budget(value: str) -> int | None:
    if value.lower() in ('none', 'unlimited'):
        return None
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError('Use an integer >= 11, or unlimited') from error
    if result < PNP_TURN_LOOKAHEAD_FRAMES + 1:
        raise argparse.ArgumentTypeError('State budget must be at least 11, or unlimited')
    return result


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
    parser.add_argument("--device", choices=("cpu",), default="cpu", help="Neural inference runs on CPU.")
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
    parser.add_argument(
        "--show-sticker-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show a transparent red sticker heatmap in final output.",
    )
    parser.add_argument(
        "--show-colored-masks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=("Color projected sticker/segmentation intersection masks in the final video using "
              "global color profiles independently for each mask."),
    )
    parser.add_argument("--roi-padding", type=float, default=0.08)
    parser.add_argument("--roi-tracking-gain", type=float, default=1.4)
    parser.add_argument("--roi-tracking-padding", type=float, default=0.18)
    parser.add_argument(
        "--show-keypoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw detected pose keypoints in the final output.",
    )
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
        "--recognize-moves", action=argparse.BooleanOptionalAction, default=False,
        help="Enable layer-turn search after pose and orientation recovery.",
    )
    parser.add_argument(
        "--turn-search-state-budget", type=parse_turn_search_budget,
        default=PNP_TURN_LOOKAHEAD_STATE_BUDGET,
        help="Maximum frame/state evaluations per decision, including seeds and retries (default: 1024; unlimited disables).",
    )
    parser.add_argument(
        "--initial-scramble",
        "--known-scramble",
        type=parse_scramble_moves,
        required=True,
        help=(
            "Known starting scramble in standard notation, for example "
            '"R U2 F\'". The fixed frame is white U/top and green F/front; '
            "it fixes sticker colors and makes color matching orientation-only."
        ),
    )
    parser.add_argument(
        "--debug-pose-symmetry",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print per-frame pose-symmetry trigger and tested-orientation scores.",
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
        "--recognize-scramble",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Report a solved-to-recognized scramble after color classification "
            "(disabled by default)."
        ),
    )
    parser.add_argument(
        "--debug-independent-color-posteriors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Color each observed sticker per frame by its nearest global profile; "
            "disable temporal and legal cube color assignments while retaining pose recovery."
        ),
    )
    parser.add_argument(
        "--debug-color-masks",
        "--debug-sticker-mask-colors",
        dest="debug_color_masks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Color each segmentation mask directly from its image pixels; requires "
            "--debug-independent-color-posteriors and --no-show-pnp, and disables the heatmap."
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


def process_cached_pose_and_profiles(
    cache: list[NeuralFrameData], trace: StickerColorTrace | None,
    args: argparse.Namespace, known_initial_colors: dict[str, str],
    global_color_samples: list[tuple[str, int, np.ndarray]] | None,
) -> tuple[StickerColorTrace, dict[str, GlobalColorProfile], dict[str, list[tuple[str, int, np.ndarray]]]]:
    """Orient with unsupervised profiles, then calibrate before recognizing moves."""
    initial_trace = trace if trace is not None else StickerColorTrace.create()
    prepared_trace = initial_trace
    needs_pose = args.show_pnp or args.recognize_moves or getattr(args, "show_colored_masks", False)
    if needs_pose:
        print("Refining pose candidates and establishing frame-to-frame continuity...", flush=True)
        prepared_trace = recover_cached_pose_symmetries(
            cache, initial_trace, progress=not args.display, refine_pose=True,
            recover_orientation=False, sticker_mask_erosion=args.sticker_mask_erosion,
            color_samples_per_sticker=args.color_samples_per_sticker,
            color_min_samples_per_sticker=args.color_min_samples_per_sticker,
        )
    print("Building global color profiles from collected samples...", flush=True)
    if args.debug_color_masks:
        profiles, observations = build_global_color_profiles_from_samples(global_color_samples or [])
    else:
        profiles, observations = build_global_color_profiles(initial_trace)
    if needs_pose:
        print("Matching cube orientation to the initial scramble and observed colors...", flush=True)
        prepared_trace = recover_cached_pose_symmetries(
            cache, prepared_trace, debug=args.debug_pose_symmetry,
            progress=not args.display, pose_prepared=True,
            recover_orientation=True,
            require_orientation=args.recognize_moves,
            recognize_moves=False, global_profiles=profiles,
            sticker_mask_erosion=args.sticker_mask_erosion,
            known_initial_colors=known_initial_colors,
        )
    if args.recognize_moves:
        print("Searching for layer moves and updating the cube state...", flush=True)
        prepared_trace = recover_cached_pose_symmetries(
            cache, prepared_trace, progress=not args.display, pose_prepared=True,
            recover_orientation=False, recognize_moves=True,
            record_all_turn_candidates=args.turn_candidate_log is not None,
            turn_candidate_log=args.turn_candidate_log, global_profiles=profiles,
            sticker_mask_erosion=args.sticker_mask_erosion,
            known_initial_colors=known_initial_colors,
            calibrate_initial_colors=True, calibration_observations=observations,
        )
    return prepared_trace, profiles, observations


def classify_cached_colors(
    trace: StickerColorTrace, args: argparse.Namespace,
    profiles: dict[str, GlobalColorProfile], known_initial_colors: dict[str, str],
    *, image: bool,
) -> tuple[dict[int, dict[str, dict[str, float]]], dict[int, dict[str, str]] | None, dict[str, str], dict[str, str]]:
    """Prepare the normal known-state view or independent debug color view."""
    likelihoods: dict[int, dict[str, dict[str, float]]] = {}
    names_by_frame = None
    names: dict[str, str] = {}
    centers: dict[str, str] = {}
    if not args.show_classified_colors:
        return likelihoods, names_by_frame, names, centers
    if args.debug_independent_color_posteriors:
        likelihoods = build_sticker_global_profile_likelihoods(trace, profiles)
        names_by_frame = classify_sticker_colors_by_frame_global_profile(trace, profiles)
        frame_index = 0 if image else max(names_by_frame, default=None)
        names = names_by_frame.get(frame_index, {})
    else:
        names = dict(known_initial_colors)
    centers = {
        identity.label: names[identity.label]
        for identity in trace.identities
        if identity.category == "center" and identity.label in names
    }
    return likelihoods, names_by_frame, names, centers


def main() -> int:
    args = build_parser().parse_args()
    global PNP_ASSIGNMENT_MODE, PNP_TURN_LOOKAHEAD_STATE_BUDGET
    PNP_ASSIGNMENT_MODE = args.pnp_assignment_mode
    PNP_TURN_LOOKAHEAD_STATE_BUDGET = args.turn_search_state_budget
    initial_state = CubeState.from_scramble(args.initial_scramble)
    known_initial_colors = {
        identity.label: initial_state.color_at(
            (identity.cubie_index, identity.axis, identity.side)
        )
        for identity in canonical_sticker_identities()
    }
    print(
        "Known initial scramble: "
        + (" ".join(args.initial_scramble) if args.initial_scramble else "(solved)"),
        flush=True,
    )
    if args.debug_independent_color_posteriors:
        args.show_classified_colors = True
        print(
            "Debug color mode: per-frame nearest global profiles; "
            "temporal and legal cube color assignments disabled; pose recovery retained",
            flush=True,
        )
        if not args.show_pnp:
            # Without a pose there is no identity projection to display, so
            # debug color mode automatically switches to direct mask coloring.
            args.debug_color_masks = True
            args.show_sticker_mask = False
    if args.debug_color_masks:
        if not args.debug_independent_color_posteriors:
            raise SystemExit(
                "--debug-color-masks requires --debug-independent-color-posteriors"
            )
        if args.show_pnp:
            raise SystemExit("--debug-color-masks requires --no-show-pnp")
        args.show_classified_colors = True
        args.show_sticker_mask = False
        print(
            "Debug mask color mode: segmentation masks classified directly; "
            "sticker heatmap disabled",
            flush=True,
        )
    args.device = args.device or "cpu"
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
    if args.recognize_moves and _turn_torch_scoring_device("mps") is None:
        raise SystemExit("Move recognition requires an available Apple MPS device")
    if args.pose_conf is not None and not 0.0 <= args.pose_conf <= 1.0:
        raise SystemExit("--pose-conf must be in [0, 1]")
    if args.pose_conf is not None and args.pose_conf > min(args.pose_class_conf.values()):
        raise SystemExit("--pose-conf cannot exceed the lowest --pose-class-conf threshold")
    if not 0.0 <= args.sticker_mask_conf <= 1.0:
        raise SystemExit("--sticker-mask-conf must be in [0, 1]")
    if not 0.0 <= args.sticker_mask_opacity <= 1.0:
        raise SystemExit("--sticker-mask-opacity must be in [0, 1]")
    if args.show_colored_masks and not args.run_sticker_segmentation:
        raise SystemExit("--show-colored-masks requires --run-sticker-segmentation")
    if args.show_colored_masks and args.debug_color_masks:
        raise SystemExit("--show-colored-masks cannot be combined with --debug-color-masks")
    if args.pnp_focal_scale <= 0.0:
        raise SystemExit("--pnp-focal-scale must be positive")
    if args.orientation_axis_length <= 0.0:
        raise SystemExit("--orientation-axis-length must be positive")
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
    if (
        args.show_classified_colors
        and not args.show_pnp
        and not args.debug_independent_color_posteriors
    ):
        raise SystemExit("--show-classified-colors requires --show-pnp")
    sticker_color_trace = None if args.debug_color_masks else StickerColorTrace.create()
    global_color_samples = [] if args.debug_color_masks else None

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
            neural_cache=image_cache,
            neural_only=True,
            global_color_samples=global_color_samples,
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
            None,
            None,
            sticker_color_trace,
            None,
            neural_data=image_cache[0],
            candidate_only=True,
        )
        sticker_color_trace, global_profiles, global_observations = process_cached_pose_and_profiles(
            image_cache, sticker_color_trace, args, known_initial_colors, global_color_samples
        )
        sticker_likelihoods_by_frame, sticker_color_names_by_frame, sticker_color_names, center_color_names = classify_cached_colors(
            sticker_color_trace, args, global_profiles, known_initial_colors, image=True
        )
        if center_color_names:
            print(
                "Classified centers: "
                + ", ".join(
                    f"{label}={color}"
                    for label, color in sorted(center_color_names.items())
                ),
                flush=True,
            )
        if args.show_classified_colors and args.recognize_scramble:
            report_classified_scramble(
                sticker_color_trace,
                sticker_color_names,
            )
        if (args.show_classified_colors or args.show_pnp or args.show_colored_masks) and args.display:
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
                CubeSymmetryTracker(),
                PoseHypothesisBuffer(max_frames=8),
                None,
                (
                    sticker_color_names_by_frame.get(0, {})
                    if sticker_color_names_by_frame is not None
                    else sticker_color_names
                ) if args.show_classified_colors else None,
                sticker_likelihoods=sticker_likelihoods_by_frame.get(0, {}),
                cached_final_only=True,
                neural_data=image_cache[0],
                global_profiles=global_profiles,
                move_accumulator=MoveAccumulator(),
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
                global_profiles=global_profiles,
                global_observations=global_observations,
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
        global_color_samples=global_color_samples,
    )
    sticker_color_trace, global_profiles, global_observations = process_cached_pose_and_profiles(
        cache, sticker_color_trace, args, known_initial_colors, global_color_samples
    )
    sticker_likelihoods_by_frame, sticker_color_names_by_frame, sticker_color_names, center_color_names = classify_cached_colors(
        sticker_color_trace, args, global_profiles, known_initial_colors, image=False
    )
    if center_color_names:
        print(
            "Classified centers: "
            + ", ".join(
                f"{label}={color}"
                for label, color in sorted(center_color_names.items())
            ),
            flush=True,
        )
    if args.show_classified_colors and args.recognize_scramble:
        report_classified_scramble(
            sticker_color_trace,
            sticker_color_names,
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
            global_profiles=global_profiles,
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
            global_profiles=global_profiles,
            global_observations=global_observations,
        )
        print(f"Saved sticker color plots: {args.rgb_html_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
