#!/usr/bin/env python3
"""Render the independent cube-keypoint YOLO model with the live ROI flow.

The foreground pass runs a cube bbox detector when needed, predicts geometric
landmarks and sticker colors with one pose model inside the tracked ROI, and
caches the raw predictions. A background worker seeds sticker poses from each
color point's nearest class-valid three/four landmarks, ranked by diagonal-center
alignment and excluding other sticker points. Cached recovery then
refines geometry, matches all observed colors directly to the known cube state,
and optionally searches layer moves. No segmentation or pixel color sampling
is performed.

Examples:

    .venv/bin/python scripts/render_yolo11n_independent_keypoints.py \
        --source data/independent_keypoints_dataset_v2/images/000000.png \
        --initial-scramble "" \
        --model runs/pose/runs/yoloPoseMedium/weights/best.pt \
        --save runs/independent_keypoints_render.png \
        --no-display

    .venv/bin/python scripts/render_yolo11n_independent_keypoints.py \
        --source 0 --initial-scramble "" \
        --model runs/pose/runs/yoloPoseMedium/weights/best.pt
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

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from live_recognition.cube_state import CubeState  # noqa: E402
from live_recognition.cube_state import parse_scramble_moves  # noqa: E402
from live_recognition.detectors import YOLOCubeDetector  # noqa: E402
from live_recognition.independent_pose_thresholds import (  # noqa: E402
    DEFAULT_INDEPENDENT_POSE_CLASS_CONFIDENCES, INDEPENDENT_POSE_CLASS_NAMES,
    independent_pose_class_threshold, parse_independent_pose_class_confidences)
from live_recognition.roi_tracking import RoiTracker  # noqa: E402
from live_recognition.roi_tracking import \
    roi_for_frame as shared_roi_for_frame  # noqa: E402

CLASS_NAMES = INDEPENDENT_POSE_CLASS_NAMES
STICKER_KEYPOINT_MIN_CONFIDENCE = 0.25
# OpenCV uses BGR colors. These match the independent-keypoint generator.
CLASS_COLORS_BGR = (
    (255, 255, 255),
    (0, 255, 255),
    (255, 120, 255),
    (255, 255, 255), (0, 255, 255), (0, 165, 255),
    (0, 0, 255), (255, 0, 0), (0, 180, 0),
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
QUICKTIME_VIDEO_EXTENSIONS = frozenset({".mp4", ".m4v", ".mov"})
WINDOW_NAME = "independent cube keypoints"
DEFAULT_BBOX_MODEL = ROOT / "runs/500ExamplesWeights.pt"
DEFAULT_POSE_MODEL = (
    ROOT
    / "runs/pose/runs/yoloPoseMedium/weights/best.pt"
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
# Display axes follow center colors; model/move coordinates remain unchanged.
# OpenCV uses BGR: X+ red, Y+ white, Z+ green.
ORIENTATION_AXIS_COLORS_BGR = (
    (0, 0, 255),
    (255, 255, 255),
    (0, 200, 0),
)
ORIENTATION_AXIS_NAMES = ("X+", "Y+", "Z+")
ORIENTATION_AXIS_CENTER_COLORS = ("red", "white", "green")
ORIENTATION_AXIS_DEFAULT_FACES = ("R", "U", "F")
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
# Turn ranking uses all pose-model sticker color points. Its bounded
# coverage penalty is converted to cube-normalized distance units before it is
# combined with keypoint reprojection error.
PNP_TURN_MASK_CENTER_WEIGHT = 1.0
# Color contributes the same weight as each geometric term in the final turn
# ranking. Geometry is landmark error plus the sticker-point term; color is
# one minus the fraction of observed projected stickers with matching colors.
PNP_TURN_COLOR_WEIGHT = 1.0
PNP_TURN_GEOMETRY_SCORE_VERSION = 2  # Whole-cube validity; invalid landmark costs remain invalid.
# Check the unchanged state's colors before generating move alternatives.
# Subtraction matches the categorical scorer exactly at the 2/3 boundary.
PNP_TURN_ORIENTATION_RESCUE_COLOR_ERROR = 1.0 - 2.0 / 3.0
MIN_TURN_LAYER_OBSERVATIONS = 3
# A turn is eligible against static when it has more unified inliers or its
# predictive color error is at least 2x better than the matching baseline.
PNP_TURN_COLOR_RESCUE_MAX_RATIO = 0.5
PNP_TURN_CENTER_PENALTY_SCALE = 0.05
# Search multiple state sequences over the next ten frames at the 30 FPS
# reference rate. The runtime window scales linearly with source FPS. Scores
# use two uniform blocks: the front half gets 30% and the back half gets 70%.
PNP_TURN_LOOKAHEAD_FRAMES = 10
PNP_TURN_LOOKAHEAD_REFERENCE_FPS = 30.0
# Scheduling batch size, not a limit on retained states or branching.
PNP_TURN_LOOKAHEAD_STATE_BATCH_SIZE = 32
# Keep the best hold plus the top geometry-plus-color changes at every parent.
PNP_TURN_LOOKAHEAD_STATE_CHANGE_LIMIT = 10
# Bound total frame/state evaluations per decision, including complete-path seeds.
PNP_TURN_LOOKAHEAD_STATE_BUDGET = 4096
# Scale the spacing to the source rate: adjacent moves are valid at <=30 FPS,
# while 60 FPS needs an intervening frame between different-layer updates.
PNP_TURN_LOOKAHEAD_MAX_STATE_UPDATES_PER_SECOND = 30.0
# Minimum whole-window fit improvement needed per additional turn. Charge
# this outside temporal weights so later moves do not become cheaper and
# future evidence can outweigh a weak current-frame fit without a hard gate.
PNP_TURN_LOOKAHEAD_MOVE_COST = 0.01


def turn_lookahead_frame_count(source_fps: float | None) -> int:
    """Scale the 30 FPS baseline window to the source video rate."""

    fps = float(source_fps) if source_fps is not None else PNP_TURN_LOOKAHEAD_REFERENCE_FPS
    if not np.isfinite(fps) or fps <= 0.0:
        fps = PNP_TURN_LOOKAHEAD_REFERENCE_FPS
    return max(1, int(round(PNP_TURN_LOOKAHEAD_FRAMES * fps / PNP_TURN_LOOKAHEAD_REFERENCE_FPS)))


def turn_state_update_gap_frames(source_fps: float | None) -> int:
    """Apply the move-rate policy in seconds rather than a fixed frame count."""

    fps = float(source_fps) if source_fps is not None else PNP_TURN_LOOKAHEAD_REFERENCE_FPS
    if not np.isfinite(fps) or fps <= 0.0:
        fps = PNP_TURN_LOOKAHEAD_REFERENCE_FPS
    return max(1, int(np.ceil(fps / PNP_TURN_LOOKAHEAD_MAX_STATE_UPDATES_PER_SECOND)))
# Final-solve observations must be close relative to the displayed cube size.
# This is deliberately expressed as a fraction of the projected cube diameter
# so the same setting works at different distances from the camera. Keep this
# separate from the acceptance threshold while the turn score is being tuned.
PNP_FINAL_ASSIGNMENT_ERROR_FRACTION = 0.05
# Reject pose candidates whose normalized reprojection fit exceeds this
# threshold, including when selecting among temporal hypotheses.
PNP_POSE_REPROJECTION_ERROR_THRESHOLD = 0.1
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
@dataclass(frozen=True)
class StickerKeypointMatch:
    """One sticker color point and its nearby ordered geometric landmarks."""

    mask_index: int
    keypoint_indices: tuple[int, ...]
    sticker_type: str
    average_confidence: float
    shape_error: float = float("inf")
    center_error: float | None = None
    # Restrict three-point PnP to the completion used for center ranking.
    # Symmetric cyclic starts for that same completion remain valid roots.
    object_index_sequences: tuple[tuple[int, ...], ...] | None = None


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
    """Per-frame categorical color observations grouped by physical sticker."""
    identities: tuple[StickerIdentity, ...]
    color_observations: dict[int, dict[str, tuple[str, ...]]] = field(default_factory=dict)
    frames_seen: int = 0

    @classmethod
    def create(cls) -> "StickerColorTrace":
        return cls(identities=canonical_sticker_identities())

    def observe_colors(self, frame_index: int, observations: dict[str, tuple[str, ...]]) -> None:
        self.frames_seen += 1
        self.color_observations[int(frame_index)] = dict(observations)


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


GLOBAL_COLOR_NAMES = tuple(PURE_COLOR_RGB)
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
    # Direct pose-model color classes, aligned with sticker_mask_centers.
    sticker_color_names: tuple[str, ...] = ()
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
    # Binary class equality [canonical sticker identity, observed color point].
    turn_color_matches: np.ndarray | None = None
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
    # Historical continuity decision, kept in its original coordinate frame
    # even when later color matching relabels the candidate's cube axes.
    pose_continuity_diagnostics: dict[str, object] | None = None
    # Committed physical sticker state before this candidate's incremental
    # layer rotation is applied.  This is part of fitting, not display-only
    # metadata, and is copied into every candidate entering turn search.
    cube_geometry_state: "CubeGeometryState | None" = None
    # Persistent center identities for display X+/Y+/Z+, including a rotated
    # initial scramble's color layout. These do not change solver coordinates.
    orientation_axis_center_labels: tuple[str, str, str] | None = None


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

    def is_solved(self, sticker_colors: dict[str, str] | None) -> bool:
        """Check the current physical colors, not the original identity layout."""

        if not sticker_colors or any(label not in sticker_colors for label in self.facelets.values()):
            return False
        return CubeState({key: sticker_colors[label] for key, label in self.facelets.items()}).is_solved()

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
    last_state_update_frame: int | None = None
    last_state_update_key: tuple | None = None
    state_update_streak: int = 0


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
        'orientation_symmetry_index', 'orientation_symmetry', 'orientation_rescue',
        'score_components',
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


def state_update_layer_direction(value: dict[str, object] | object) -> tuple | None:
    """Identify a state update's layer and direction for double-turn continuation."""

    if isinstance(value, dict):
        moves = tuple(str(move) for move in value.get("state_moves", ()))
        axis = value.get("layer_axis")
        side = value.get("layer_side")
        layer_count = value.get("layer_count")
    else:
        moves = completed_candidate_moves(value)
        rotations = tuple(getattr(value, "layer_rotations", ()))
        rotation = rotations[0] if rotations else None
        axis = getattr(rotation, "axis", None)
        side = getattr(rotation, "side", None)
        layer_count = getattr(rotation, "layer_count", None)
    if not moves:
        return None
    directions = tuple(
        2 if move.endswith("2") else -1 if move.endswith("'") else 1
        for move in moves
    )
    if axis is None or side is None or layer_count is None:
        # Synthetic callers and older cached trials may not have geometry
        # metadata; exact notation is the safest compatibility fallback.
        return ("notation", moves)
    return ("layer", int(axis), int(side), int(layer_count), directions)


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
    frame_indices: tuple[int, ...] | None = None,
    previous_state_update_frame: int | None = None,
    previous_state_update_key: tuple | None = None,
    previous_state_update_streak: int = 0,
    # Keep the helper's historical unrestricted behavior unless the caller
    # explicitly enables the video-rate policy below.
    min_state_update_gap_frames: int = 1,
) -> tuple[MoveLookaheadPath, ...]:
    """Search combined-fit-shortlisted changes with bounds and no beam cutoff.

    Collapse the hold geometries before branching. A complete no-turn rollout
    supplies the initial upper bound. Explore optimistic completions first;
    combined current fit orders equally promising children. Merge prefixes by (depth, state),
    and discard one only when dominated or unable to improve the incumbent.
    At every parent keep the best hold plus the top combined-fit state
    changes; None disables that shortlist. Batch size only limits GPU work
    per call. Bounds prove optimality within the shortlisted search space.
    Supplied bounds must be valid for every state. Missing frame bounds
    disable bound pruning across those unknown frames. An optional total
    frame/state budget includes the current state and seed rollouts; a budget
    stop returns only complete forecasts and reports an unresolved bound gap.
    A state-changing trial that returns to this window's starting state is
    ineligible at every depth. Holds remain valid. State-changing frames must
    be separated by 'min_state_update_gap_frames' actual frame indices,
    including across the boundary from the previous search window. A
    consecutive same-layer, same-direction update is allowed as the second
    quarter-turn of a double turn. Warm-start and greedy rollouts obey the same
    rule and per-parent shortlist. No intermediate-state revisit restriction is
    imposed.
    """

    if state_batch_size < 1:
        raise ValueError('State batch size must be positive')
    if state_change_limit is not None and state_change_limit < 0:
        raise ValueError('State-change limit must be nonnegative or None')
    if min_state_update_gap_frames < 1:
        raise ValueError('Minimum state-update gap must be positive')
    count = len(future_frame_indices) + 1
    if frame_indices is None:
        frame_indices = tuple(range(count))
    else:
        frame_indices = tuple(int(frame) for frame in frame_indices)
        if len(frame_indices) != count:
            raise ValueError('Frame indices must cover the current and future search frames')
        if any(after <= before for before, after in zip(frame_indices, frame_indices[1:])):
            raise ValueError('Frame indices must be strictly increasing')
    starting_signature = state_before.signature()
    if state_evaluation_budget is not None and state_evaluation_budget < count:
        raise ValueError('State budget must cover at least one state per search frame')
    if warm_start_moves is not None and len(warm_start_moves) != count:
        raise ValueError('Warm start must provide one move choice per search frame')
    weights = move_lookahead_weights(count)
    bounds = tuple(frame_score_lower_bounds) if frame_score_lower_bounds is not None else (-float('inf'),) * count
    if len(bounds) != count or any(np.isnan(value) for value in bounds):
        raise ValueError('Provide one non-NaN lower bound per search frame')
    # Treat unsupported/nonfinite bounds as unknown rather than assuming zero.
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
                           state_change_limit=state_change_limit, shortlist_rank='geometry_plus_color',
                           prune_root_state_returns=True,
                           min_state_update_gap_frames=min_state_update_gap_frames,
                           previous_state_update_frame=previous_state_update_frame,
                           previous_state_update_key=previous_state_update_key,
                           previous_state_update_streak=previous_state_update_streak,
                           frame_indices=list(frame_indices))
    stats = [dict(depth=d, attempted_trials=0, invalid_score=0, unavailable=0,
                  immediate_undo=0, state_revisit=0, root_return=0, input_paths=0, scored_states=0,
                  expanded_paths=0, merged_paths=0, collapsed_holds=0,
                  bound_pruned_paths=0, shortlist_pruned_paths=0, invalid_color=0,
                  unreliable_color_states=0, completed_paths=0, queued_paths=0,
                  state_update_too_soon=0)
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

    def state_update_allowed(
        depth: int,
        last_update_frame: int | None,
        last_update_key: tuple | None,
        last_update_streak: int,
        moves: tuple[str, ...],
        update_key: tuple | None,
    ) -> bool:
        return (
            not moves
            or last_update_frame is None
            or frame_indices[depth] - last_update_frame >= min_state_update_gap_frames
            or (
                frame_indices[depth] - last_update_frame == 1
                and last_update_streak < 2
                and update_key is not None
                and update_key == last_update_key
            )
        )

    def next_update_state(
        depth: int,
        last_update_frame: int | None,
        last_update_key: tuple | None,
        last_update_streak: int,
        moves: tuple[str, ...],
        update_key: tuple | None,
    ) -> tuple[int | None, tuple | None, int]:
        return (
            (
                frame_indices[depth],
                update_key,
                last_update_streak + 1
                if (
                    last_update_frame is not None
                    and frame_indices[depth] - last_update_frame == 1
                    and update_key == last_update_key
                )
                else 1,
            )
            if moves
            else (last_update_frame, last_update_key, last_update_streak)
        )

    def path_frontier_key(path: MoveLookaheadPath) -> tuple:
        if min_state_update_gap_frames <= 1:
            return (path.state.signature(),)
        return (
            path.state.signature(),
            path.last_state_update_frame,
            path.last_state_update_key,
            path.state_update_streak,
        )

    # Establish a complete solution before examining alternative states.
    first_hold = hold(current_trials)
    no_turn = None
    if first_hold is not None:
        no_turn = MoveLookaheadPath(
            state_before, first_hold, last_move=previous_move,
            last_state_update_frame=previous_state_update_frame,
            last_state_update_key=previous_state_update_key,
            state_update_streak=previous_state_update_streak,
        )
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
        last_update_frame = previous_state_update_frame
        last_update_key = previous_state_update_key
        last_update_streak = previous_state_update_streak
        for depth in range(count):
            trials = current_trials if depth == 0 else evaluate(depth, (before,))[0]
            unique = {}
            for trial in trials or ():
                if not np.isfinite(float(trial['state_score'])) or turn_trial_has_invalid_color(trial):
                    continue
                moves = tuple(trial['state_moves'])
                update_key = state_update_layer_direction(trial)
                if not state_update_allowed(
                    depth, last_update_frame, last_update_key, last_update_streak,
                    moves, update_key,
                ):
                    continue
                after = before.apply_moves(moves)
                key = after.signature()
                if moves and key == starting_signature:
                    root_return_pruned += 1
                    continue
                edge = weights[depth] * float(trial['state_score']) + PNP_TURN_LOOKAHEAD_MOVE_COST * len(moves)
                previous = unique.get(key)
                if previous is None or (edge, trial['rank']) < previous[0]:
                    unique[key] = ((edge, trial['rank']), trial, after)
            holds = [item for item in unique.values() if not item[1]['state_moves']]
            changes = sorted((item for item in unique.values() if item[1]['state_moves']),
                             key=lambda item: (float(item[1]['state_score']), item[1]['rank']))
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
            update_key = state_update_layer_direction(trial)
            next_frame, next_key, next_streak = next_update_state(
                depth, last_update_frame, last_update_key, last_update_streak,
                moves, update_key,
            )
            path = (MoveLookaheadPath(after, trial, last_move=moves[-1] if moves else previous_move,
                                      move_count=len(moves),
                                      last_state_update_frame=next_frame,
                                      last_state_update_key=next_key,
                                      state_update_streak=next_streak) if path is None else
                    MoveLookaheadPath(after, path.first_trial, (*path.future_scores, float(trial['state_score'])),
                                      moves[-1] if moves else path.last_move,
                                      (*path.future_trials, trial), path.move_count + len(moves),
                                      next_frame, next_key, next_streak))
            last_update_frame, last_update_key, last_update_streak = (
                next_frame, next_key, next_streak
            )
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

    def offer(path: MoveLookaheadPath, depth: int, current_fit: float) -> None:
        nonlocal serial
        stats[depth]['expanded_paths'] += 1
        if depth == count - 1:
            complete(path)
            return
        rank = prefix_rank(path)
        frontier_key = path_frontier_key(path)
        previous = seen[depth].get(frontier_key)
        if previous is not None and previous <= rank:
            record(path, 'merged_paths', depth)
            return
        seen[depth][frontier_key] = rank
        optimistic = rank[0] + suffix[depth + 1]
        if incumbent is not None and optimistic > prefix_rank(incumbent)[0]:
            record(path, 'bound_pruned_paths', depth)
            return
        serial += 1
        heapq.heappush(queue, (optimistic, current_fit, -depth, serial, path))
        stats[depth]['queued_paths'] += 1
        if diagnostics is not None:
            roots[id(path.first_trial)]['last_retained_depth'] = depth

    def expand(parent: MoveLookaheadPath | None, trials, depth: int) -> None:
        before = state_before if parent is None else parent.state
        last_update_frame = (
            previous_state_update_frame if parent is None else parent.last_state_update_frame
        )
        last_update_key = (
            previous_state_update_key if parent is None else parent.last_state_update_key
        )
        last_update_streak = (
            previous_state_update_streak if parent is None else parent.state_update_streak
        )
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
            update_key = state_update_layer_direction(trial)
            if not state_update_allowed(
                depth, last_update_frame, last_update_key, last_update_streak,
                moves, update_key,
            ):
                stats[depth]['state_update_too_soon'] += 1
                if parent is None:
                    if diagnostics is not None:
                        roots[id(trial)].update(status='state_update_too_soon', eliminated_depth=depth)
                else:
                    record(parent, 'state_update_too_soon', depth)
                continue
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
            bool(item[0]['state_moves']), float(item[0]['state_score']), item[0]['rank']))
        if state_change_limit is not None:
            holds = [item for item in ordered if not item[0]['state_moves']]
            changes = [item for item in ordered if item[0]['state_moves']]
            for trial, _after in changes[state_change_limit:]:
                if parent is None:
                    stats[depth]['shortlist_pruned_paths'] += 1
                    if diagnostics is not None:
                        roots[id(trial)].update(status='shortlist_pruned', eliminated_depth=depth)
                else:
                    record(parent, 'shortlist_pruned_paths', depth)
            ordered = holds + changes[:state_change_limit]
        for trial, after in ordered:
            moves = tuple(trial['state_moves'])
            update_key = state_update_layer_direction(trial)
            next_frame, next_key, next_streak = next_update_state(
                depth, last_update_frame, last_update_key, last_update_streak,
                moves, update_key,
            )
            path = (MoveLookaheadPath(after, trial, last_move=moves[-1] if moves else previous_move,
                                      move_count=len(moves),
                                      last_state_update_frame=next_frame,
                                      last_state_update_key=next_key,
                                      state_update_streak=next_streak) if parent is None else
                    MoveLookaheadPath(after, parent.first_trial,
                                      (*parent.future_scores, float(trial['state_score'])),
                                      moves[-1] if moves else parent.last_move,
                                      (*parent.future_trials, trial) if diagnostics is not None or retain_forecast else (),
                                      parent.move_count + len(moves),
                                      next_frame, next_key, next_streak))
            offer(path, depth, float(trial['state_score']))

    try:
        expand(None, current_trials, 0)
        peak_queue = len(queue)
        while queue and not budget_exhausted:
            batch = []
            while queue and len(batch) < state_batch_size:
                optimistic, _current_fit, _neg_depth, _serial, path = heapq.heappop(queue)
                depth = len(path.future_scores)
                frontier_key = path_frontier_key(path)
                if seen[depth].get(frontier_key) != prefix_rank(path):
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
        shortlisted = any(item['shortlist_pruned_paths'] for item in stats)
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
                           search_space='combined_score_shortlist' if shortlisted else 'all_state_changes',
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
    solved_frame_index: int | None = None
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

    def stop_if_solved(
        self, candidate: StickerPnPCandidate | None, frame_index: int | None,
    ) -> bool:
        """Latch a solved accepted current state, never a forecast state."""

        if (self.solved_frame_index is None and candidate is not None
                and self._state is not None and self._state.is_solved()
                and not candidate.turn_color_unreliable
                and candidate_layers_are_aligned(candidate)
                and pose_reprojection_error_is_acceptable(candidate)):
            self.solved_frame_index = 0 if frame_index is None else frame_index
        return self.solved_frame_index is not None

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
        search = getattr(candidate, "turn_search_diagnostics", None) or {}
        if search.get("stop_reason") == "cube_solved":
            self.solved_frame_index = search["solved_frame_index"]
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
    current = "solved" if accumulator.solved_frame_index is not None else accumulator.current_fit
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
        self.frame_index: int | None = None

    def reset(self) -> None:
        self._rotation = None
        self.frame_index = None

    def observe(self, rvec: np.ndarray, frame_index: int | None = None) -> None:
        self._rotation = cv2.Rodrigues(
            np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        )[0]
        self.frame_index = frame_index

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
    names = getattr(result, "names", None)
    if names is not None:
        class_ids = np.asarray([
            CLASS_NAMES.index(name) if (name := class_name(names, int(cid))) in CLASS_NAMES else -1
            for cid in class_ids
        ], dtype=np.int32)
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
    """Rank rhombus-like quads by compactness plus parallelogram error.

    Perimeter squared / (16 * area) penalizes thin or stretched groups, with
    a square at zero after subtracting one. The normalized distance between
    diagonal midpoints measures failure to form a parallelogram. Both terms
    are invariant to translation, rotation and image scale.
    """

    selected = np.asarray(points, dtype=np.float64).reshape(-1, 2)[list(indices)]
    if len(selected) != 4 or not np.isfinite(selected).all():
        return float("inf")
    edges = np.roll(selected, -1, axis=0) - selected
    scale = float(np.mean(np.linalg.norm(edges, axis=1)))
    if not np.isfinite(scale) or scale <= np.finfo(np.float64).eps:
        return float("inf")
    quad = (selected - selected.mean(axis=0)) / scale
    if not cv2.isContourConvex(quad.astype(np.float32)):
        return float("inf")
    following = np.roll(quad, -1, axis=0)
    area = 0.5 * abs(float(np.sum(quad[:, 0] * following[:, 1] - quad[:, 1] * following[:, 0])))
    if area <= 1e-10:
        return float("inf")
    # Normalization makes the mean side length one and perimeter four.
    compactness_error = max(0.0, 1.0 / area - 1.0)
    parallelogram_error = 0.5 * float(np.linalg.norm(quad[0] + quad[2] - quad[1] - quad[3]))
    return compactness_error + parallelogram_error


def sticker_match_priority_key(match: StickerKeypointMatch) -> tuple[float, int, float, int]:
    """Prefer geometrically plausible stickers, then use confidence as a tie-breaker."""

    if match.center_error is not None:
        return (
            float(match.center_error),
            -len(match.keypoint_indices),
            -float(match.average_confidence),
            int(match.mask_index),
        )
    if len(match.keypoint_indices) != 4:
        return (
            0.0,
            -len(match.keypoint_indices),
            -float(match.average_confidence),
            int(match.mask_index),
        )
    return (
        float(match.shape_error),
        -len(match.keypoint_indices),
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


def pose_keypoint_confidence_threshold(class_id: int, point_threshold: float) -> float:
    """Keep a separate confidence floor for the six sticker color classes."""
    return max(point_threshold, STICKER_KEYPOINT_MIN_CONFIDENCE) if class_id >= 3 else point_threshold


def usable_pose_points(
    points: np.ndarray, point_confidences: np.ndarray, class_ids: np.ndarray,
    box_confidences: np.ndarray, point_threshold: float,
    class_confidences: dict[str, float],
) -> np.ndarray:
    """Apply the same confidence and finite-coordinate checks to every class."""
    return np.asarray([
        np.isfinite(point).all() and np.isfinite(pc) and np.isfinite(bc)
        and 0 <= int(cid) < len(CLASS_NAMES)
        and pc >= pose_keypoint_confidence_threshold(int(cid), point_threshold)
        and bc >= independent_pose_class_threshold(CLASS_NAMES[int(cid)], class_confidences)
        for point, pc, cid, bc in zip(
            points, point_confidences, class_ids, box_confidences, strict=True,
        )
    ], dtype=bool)


def sticker_keypoint_matches_for_colors(
    landmark_points: np.ndarray, landmark_confidences: np.ndarray,
    landmark_class_ids: np.ndarray, sticker_points: np.ndarray,
) -> tuple[StickerKeypointMatch, ...]:
    """Find each color point's nearest class-valid three/four-landmark groups.

    Grow the nearest-landmark neighborhood only until a valid group exists,
    then prefer its smallest total distance. Rank groups by the RMS distance
    from the color point to both diagonal midpoints, in unnormalized pixels.
    For three observations, test every class-consistent parallelogram
    completion and keep the best center alignment. Never add the inferred
    point to the measurements passed to PnP. Other sticker points must stay
    outside the full real/inferred quadrilateral.
    """
    points = np.asarray(landmark_points, dtype=np.float64).reshape(-1, 2)
    confidences = np.asarray(landmark_confidences)
    classes = np.asarray(landmark_class_ids)
    sticker_points = np.asarray(sticker_points, dtype=np.float64).reshape(-1, 2)
    matches: dict[tuple[tuple[int, ...], str], StickerKeypointMatch] = {}
    for sticker_index, center in enumerate(sticker_points):
        if not np.isfinite(center).all():
            continue
        other_stickers = sticker_points[(np.arange(len(sticker_points)) != sticker_index)
                                       & np.isfinite(sticker_points).all(axis=1)]
        blocked_pairs = sticker_blocked_landmark_pairs(points, center, other_stickers)
        distances = np.linalg.norm(points - center, axis=1)
        nearest = np.argsort(distances, kind="stable")
        nearest = [int(i) for i in nearest if np.isfinite(points[i]).all() and 0 <= classes[i] < 3]
        for count in (4, 3):
            for size in range(count, len(nearest) + 1):
                groups = []
                # Only new combinations containing the newly added point.
                compatible_nearby = [i for i in nearest[:size - 1]
                                     if not blocked_pairs[i, nearest[size - 1]]]
                for preceding in combinations(compatible_nearby, count - 1):
                    indices = (*preceding, nearest[size - 1])
                    if any(blocked_pairs[i, j] for i, j in combinations(preceding, 2)):
                        continue
                    if sticker_type_from_keypoint_classes(classes[list(indices)].tolist()) == "unknown":
                        continue
                    group = sticker_center_alignment_matches(
                        points, classes, confidences, indices, center, sticker_index, other_stickers,
                    )
                    if group:
                        groups.append((float(np.sum(distances[list(indices)])), group))
                if not groups:
                    continue
                _, group = min(groups, key=lambda item: (
                    item[0], sticker_match_priority_key(item[1][0]),
                ))
                for match in group:
                    # Repeated color detections must not use multiple seed slots.
                    key = (tuple(sorted(match.keypoint_indices)), match.sticker_type)
                    previous = matches.get(key)
                    if previous is None or sticker_match_priority_key(match) < sticker_match_priority_key(previous):
                        matches[key] = match
                break
    return tuple(sorted(matches.values(), key=sticker_match_priority_key))


def sticker_blocked_landmark_pairs(
    points: np.ndarray, center: np.ndarray, other_stickers: np.ndarray,
) -> np.ndarray:
    """Prune pairs whose triangle with the seed contains another sticker.

    Every enclosing convex sticker using that pair contains this triangle,
    so no third/fourth landmark can rescue it. Batch all tests before the
    combination search, rather than repeatedly building invalid quads.
    """
    blocked = np.zeros((len(points), len(points)), dtype=bool)
    if len(other_stickers) == 0:
        return blocked
    first, second = np.triu_indices(len(points), k=1)
    a, b = points[first] - center, points[second] - center
    q = other_stickers - center

    def cross(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    area = cross(a, b)
    sides = np.stack((
        cross(a[:, None], q[None]),
        cross((b - a)[:, None], q[None] - a[:, None]),
        cross(-b[:, None], q[None] - b[:, None]),
    ), axis=-1)
    inside = np.all(sides >= 0.0, axis=-1) | np.all(sides <= 0.0, axis=-1)
    invalid = np.any(inside, axis=1) & (np.abs(area) > 1e-6)
    blocked[first, second] = invalid
    blocked[second, first] = invalid
    return blocked


def sticker_center_alignment_matches(
    points: np.ndarray, class_ids: np.ndarray, confidences: np.ndarray,
    indices: tuple[int, ...], center: np.ndarray, sticker_index: int,
    other_stickers: np.ndarray | None = None,
) -> tuple[StickerKeypointMatch, ...]:
    """Validate cyclic profiles and score real/inferred diagonal centers."""
    selected = points[list(indices)]
    if not np.isfinite(selected).all():
        return ()
    centroid = selected.mean(axis=0)
    order = np.argsort(np.arctan2(selected[:, 1] - centroid[1], selected[:, 0] - centroid[0]), kind="stable")
    indices = tuple(indices[int(i)] for i in order)
    selected = points[list(indices)]
    contour = selected.astype(np.float32)
    if not cv2.isContourConvex(contour) or abs(cv2.contourArea(contour)) <= 1e-6:
        return ()
    observed_classes = class_ids[list(indices)].tolist()
    options = []
    for sticker_type in STICKER_TYPE_CLASS_IDS:
        sequences = sticker_correspondence_object_indices(sticker_type, observed_classes)
        completions: dict[tuple[int, int] | None, list[tuple[int, ...]]] = {}
        for sequence in sequences:
            # A three-point sequence has exactly one observed opposite pair.
            opposite = (next((i, j) for i, j in combinations(range(3), 2)
                             if (sequence[i] - sequence[j]) % 4 == 2)
                        if len(indices) == 3 else None)
            completions.setdefault(opposite, []).append(sequence)
        for opposite, compatible in completions.items():
            if opposite is None:
                quad = selected
            else:
                i, j = opposite
                other = 3 - i - j
                inferred = selected[i] + selected[j] - selected[other]
                quad = np.empty((4, 2), dtype=np.float64)
                quad[list(compatible[0])] = selected
                missing = next(i for i in range(4) if i not in compatible[0])
                quad[missing] = inferred
            # Keep local enclosing stickers, never a remote extrapolation.
            if cv2.pointPolygonTest(quad.astype(np.float32), tuple(float(v) for v in center), False) < 0:
                continue
            if other_stickers is not None and any(
                cv2.pointPolygonTest(quad.astype(np.float32), tuple(float(v) for v in point), False) >= 0
                for point in other_stickers
            ):
                continue
            midpoints = (quad + np.roll(quad, -2, axis=0))[:2] / 2.0
            error = float(np.sqrt(np.mean(np.sum((midpoints - center) ** 2, axis=1))))
            options.append(StickerKeypointMatch(
                sticker_index, indices, sticker_type,
                float(np.mean(confidences[list(indices)])),
                center_error=error, object_index_sequences=tuple(compatible),
            ))
    if not options:
        return ()
    best_error = min(float(match.center_error) for match in options)
    # Keep ambiguous physical sticker types only for the same best completion.
    return tuple(match for match in options if abs(float(match.center_error) - best_error) <= 1e-9)


def attach_color_keypoints(
    candidate: StickerPnPCandidate, centers: np.ndarray, colors: tuple[str, ...],
) -> StickerPnPCandidate:
    """Use pose color points as the center observations for existing geometry fits."""
    return replace(
        candidate, sticker_mask_centers=tuple(np.asarray(center).copy() for center in centers),
        sticker_mask_center_uncertainties=(1.0,) * len(centers),
        sticker_color_names=colors, sticker_mask_colors=(),
        sticker_contour=None, sticker_contours=(), sticker_boundary_samples=(),
    )


def projected_color_observations(candidate: StickerPnPCandidate) -> dict[str, tuple[str, ...]]:
    """Group color predictions by containing projected sticker, excluding the background.

    Each point belongs to its nearest containing sticker if projections share
    a boundary. Preserve all colors in a group so a duplicate can still match.
    """
    faces = projected_sticker_faces(candidate, candidate.rvec, candidate.tvec)
    groups: dict[str, set[str]] = {}
    for point, color in zip(candidate.sticker_mask_centers, candidate.sticker_color_names, strict=True):
        containing = [face for face in faces if cv2.pointPolygonTest(
            np.asarray(face[2], dtype=np.float32), tuple(float(v) for v in point), False,
        ) >= 0]
        if not containing:
            continue
        face = min(containing, key=lambda face: float(np.linalg.norm(np.mean(face[2], axis=0) - point)))
        groups.setdefault(face[0].label, set()).add(color)
    return {label: tuple(sorted(colors)) for label, colors in groups.items()}


def match_orientation_to_colors(
    candidate: StickerPnPCandidate, expected_colors: dict[str, str],
) -> StickerPnPCandidate:
    """Choose among all cube symmetries by direct sticker-color agreement only."""
    best = candidate
    best_score = (-1, -1, False)
    for symmetry in CUBE_SYMMETRY_MATRICES:
        transformed = transform_candidate_by_symmetry(candidate, symmetry)
        observations = projected_color_observations(transformed)
        correct = sum(expected_colors.get(label) in colors for label, colors in observations.items())
        score = (correct, len(observations), bool(np.array_equal(symmetry, np.eye(3))))
        if score > best_score:
            best, best_score = transformed, score
    center_by_color = {
        expected_colors[identity.label]: identity.label
        for identity in canonical_sticker_identities()
        if identity.category == "center" and identity.label in expected_colors
    }
    axis_centers = (tuple(center_by_color[color] for color in ORIENTATION_AXIS_CENTER_COLORS)
                    if all(color in center_by_color for color in ORIENTATION_AXIS_CENTER_COLORS) else None)
    return replace(best, pose_continuity_diagnostics=None, orientation_axis_center_labels=axis_centers)


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


def recover_cached_pose_symmetries(
    cache: list[NeuralFrameData],
    trace: StickerColorTrace,
    *,
    progress: bool = False,
    refine_pose: bool = False,
    recover_orientation: bool = True,
    recognize_moves: bool = False,
    source_fps: float = PNP_TURN_LOOKAHEAD_REFERENCE_FPS,
    turning_start_frame: int | None = None,
    record_all_turn_candidates: bool = False,
    turn_candidate_log: Path | None = None,
    known_initial_colors: dict[str, str] | None = None,
    pose_prepared: bool = False,
) -> StickerColorTrace:
    """Refine geometry, match categorical colors, and replay the existing move rollout.

    Orientation is selected independently from observed color classes. No RGB
    sampling, learned color profiles, or temporal orientation gate is involved.
    """
    if turning_start_frame is not None and turning_start_frame < 0:
        raise ValueError("Turning start frame must be nonnegative or None")
    lookahead_frames = turn_lookahead_frame_count(source_fps)
    state_update_gap_frames = turn_state_update_gap_frames(source_fps)
    turn_color_priors = {
        label: np.eye(len(GLOBAL_COLOR_NAMES))[GLOBAL_COLOR_NAMES.index(color)]
        for label, color in (known_initial_colors or {}).items() if color in GLOBAL_COLOR_NAMES
    }
    corrected_trace = StickerColorTrace.create()
    cube_geometry_state = CubeGeometryState.identity()
    solved_frame_index = None
    last_committed_move = None
    last_committed_state_update_frame = None
    last_committed_state_update_key = None
    last_committed_state_update_streak = 0
    committed_state_history = []
    lookahead_frame_lower_bound_cache = {}
    lookahead_geometry_cache = {}
    lookahead_color_trial_cache = {}
    lookahead_scoring_device = _turn_torch_scoring_device("mps")
    lookahead_forecast = {}
    lookahead_forecast_expected_state = None
    total_records = len(cache)
    previous_candidate = None
    move_search_started = False

    def prepare_lookahead_geometry(
        cache_index: int,
        candidate_override: StickerPnPCandidate | None = None,
    ) -> tuple[StickerPnPCandidate, dict[str, object]] | None:
        """Prepare one frame's geometry once, independent of cube state."""

        if cache_index < 0 or cache_index >= total_records:
            return None
        if _frame_forces_static_hold(cache[cache_index].frame_index, turning_start_frame):
            return None
        if cache_index in lookahead_geometry_cache:
            cached = lookahead_geometry_cache[cache_index]
            if candidate_override is None:
                return cached
            if (cached is not None
                    and np.allclose(cached[0].rvec, candidate_override.rvec, rtol=0, atol=1e-9)
                    and np.allclose(cached[0].tvec, candidate_override.tvec, rtol=0, atol=1e-9)):
                return candidate_override, cached[1]
            # Color matching can change this frame's cube axes after it was
            # visited by a prior lookahead. Its geometry must follow that pose.
            lookahead_frame_lower_bound_cache.pop(cache_index, None)
            for key in tuple(lookahead_color_trial_cache):
                if key[0] == cache_index:
                    del lookahead_color_trial_cache[key]
        record = cache[cache_index]
        frame_candidate = candidate_override
        if frame_candidate is None:
            frame_candidate = getattr(record, "pose_candidate", None)
            if not isinstance(frame_candidate, StickerPnPCandidate):
                lookahead_geometry_cache[cache_index] = None
                return None
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
        color_matches = turn_color_match_matrix(
            state_candidate, turn_color_priors,
        )
        if color_matches is None:
            color_matches = getattr(
                state_candidate,
                "turn_color_matches",
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
            color_matches,
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
            color_matches = turn_color_match_matrix(
                frame_candidate, turn_color_priors,
            )
            if color_matches is None:
                color_matches = getattr(frame_candidate, "turn_color_matches", None)
            trials_by_state = _score_turn_color_offsets_with_orientation_rescue(
                frame_candidate, tuple(geometry_frame["options"]),
                PNP_FINAL_ASSIGNMENT_ERROR_FRACTION * projected_cube_diameter(frame_candidate),
                lookahead_scoring_device, color_matches, missing,
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
            likelihoods = turn_color_match_matrix(
                frame_candidate, turn_color_priors,
            )
            if likelihoods is None:
                likelihoods = getattr(frame_candidate, 'turn_color_matches', None)
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
            dict(version=15, algorithm="adaptive_branch_and_bound", status="not_started",
                 orientation_rescue_color_threshold=PNP_TURN_ORIENTATION_RESCUE_COLOR_ERROR,
                 turning_start_frame=turning_start_frame,
                 previous_move=last_committed_move, beam_width=None,
                 state_batch_size=PNP_TURN_LOOKAHEAD_STATE_BATCH_SIZE,
                 state_change_limit=PNP_TURN_LOOKAHEAD_STATE_CHANGE_LIMIT,
                 recent_committed_states=len(previous_states), prune_state_revisits=False,
                 prune_root_state_returns=True,
                 prune_immediate_undo=False,
                 min_state_update_gap_frames=state_update_gap_frames,
                 previous_state_update_frame=last_committed_state_update_frame,
                 previous_state_update_key=last_committed_state_update_key,
                 previous_state_update_streak=last_committed_state_update_streak,
                 merge_key="cube_state", preserve_no_turn_path=True,
                 move_cost_per_turn=PNP_TURN_LOOKAHEAD_MOVE_COST,
                 source_fps=float(source_fps),
                 requested_future_frames=lookahead_frames,
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
                min(total_records, current_cache_index + lookahead_frames + 1),
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
                        min(total_records, current_cache_index + lookahead_frames + 1),
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
        # Do not collect the branch-and-bound diagnostics unless the caller
        # requested a turn-candidate log.  An empty dict is still treated as
        # enabled by search_move_lookahead_paths, so passing one here would
        # retain most of the bookkeeping cost even when its contents are
        # discarded.
        run_debug = search_debug
        seed_moves = tuple(lookahead_forecast.get(index) for index in (current_cache_index, *future_indices)) if reused_forecast else None
        search_paths = search_move_lookahead_paths(
            state_before, current_trials, future_indices, score_frontier,
            previous_move=last_committed_move, diagnostics=run_debug,
            state_evaluation_budget=PNP_TURN_LOOKAHEAD_STATE_BUDGET,
            warm_start_moves=seed_moves, seed_greedy=True, retain_forecast=True,
            previous_states=previous_states,
            frame_indices=frames,
            previous_state_update_frame=last_committed_state_update_frame,
            previous_state_update_key=last_committed_state_update_key,
            previous_state_update_streak=last_committed_state_update_streak,
            min_state_update_gap_frames=state_update_gap_frames,
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
                                      projected_poses=sum(item.get('projection_count', 0) for item in projection_caches),
                                      color_configurations=sum(item.get('configuration_count', 0) for item in projection_caches),
                                      assignment_builds=sum(item.get('assignments', {}).get('builds', 0) for item in projection_caches),
                                      assignment_hits=sum(item.get('assignments', {}).get('hits', 0) for item in projection_caches),
                                      ),
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
            return fallback_with_diagnostics(
                'state_budget_no_complete_path'
                if run_debug is not None and run_debug.get('budget_exhausted')
                else 'no_surviving_path'
            )
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

    label = "pose refinement" if refine_pose else "move search" if recognize_moves else "orientation matching"
    for record_index, record in enumerate(cache, start=1):
        if progress:
            frame_label = label
            if recognize_moves:
                if _frame_forces_static_hold(record.frame_index, turning_start_frame):
                    frame_label = "static holds"
                elif solved_frame_index is not None:
                    frame_label = "solved replay"
            print_headless_progress(frame_label, record_index, total_records)
        committed_state_history[:] = [entry for entry in committed_state_history
                                      if record_index - 1 - entry[0] <= lookahead_frames]
        for past_index in tuple(lookahead_geometry_cache):
            if past_index < record_index - 1:
                del lookahead_geometry_cache[past_index]
                lookahead_frame_lower_bound_cache.pop(past_index, None)
        for key in tuple(lookahead_color_trial_cache):
            if key[0] < record_index - 1:
                del lookahead_color_trial_cache[key]
        candidate = record.pose_candidate
        if not isinstance(candidate, StickerPnPCandidate):
            corrected_trace.observe_colors(record.frame_index, {})
            if turn_candidate_log is not None and recognize_moves:
                append_turn_candidate_log(turn_candidate_log, None, record.frame_index)
            continue
        hypotheses = tuple(getattr(record, "pose_candidates", ())) or (candidate,)
        if refine_pose:
            hypotheses = tuple(solve_pnp_from_low_error_assignments(
                hypothesis, recover_layer_turns=False,
                record_all_turn_candidates=record_all_turn_candidates,
            ) for hypothesis in hypotheses)
        if refine_pose or not pose_prepared:
            candidate = select_pose_hypothesis(
                hypotheses, None, (previous_candidate,) if previous_candidate is not None else (),
            )
            if candidate is None:
                record.pose_diagnostic_candidate = min(hypotheses, key=pose_hypothesis_visual_score)
                record.pose_candidate = None
                record.pose_candidates = ()
                corrected_trace.observe_colors(record.frame_index, {})
                continue
        motion_rejected = False
        next_candidate = cache[record_index].pose_candidate if record_index < total_records else None
        if (not pose_prepared and POSE_TWO_SIDED_MOTION_REJECTION_ENABLED
                and previous_candidate is not None and isinstance(next_candidate, StickerPnPCandidate)):
            current = np.asarray(candidate.tvec).reshape(3)
            motion_rejected = (
                np.linalg.norm(current - np.asarray(previous_candidate.tvec).reshape(3))
                > POSE_MOTION_DISPLACEMENT_TOLERANCE_MM
                and np.linalg.norm(np.asarray(next_candidate.tvec).reshape(3) - current)
                > POSE_MOTION_DISPLACEMENT_TOLERANCE_MM
            )
        candidate = replace(candidate, cube_geometry_state=cube_geometry_state.copy())
        if (recover_orientation or recognize_moves) and known_initial_colors:
            candidate = match_orientation_to_colors(candidate, known_initial_colors)
        if recognize_moves:
            static_candidate = candidate
            if solved_frame_index is not None:
                candidate = _hold_after_solved(candidate, record.frame_index, solved_frame_index)
            elif _frame_forces_static_hold(record.frame_index, turning_start_frame):
                candidate = _hold_before_turning_start(candidate, record.frame_index, turning_start_frame)
            else:
                if not move_search_started:
                    if progress:
                        print()
                    print(
                        f"Starting move search at frame {record.frame_index} "
                        "using direct sticker-color matches...", flush=True,
                    )
                    move_search_started = True
                current_geometry = prepare_lookahead_geometry(record_index - 1, candidate)
                candidate = solve_pnp_from_low_error_assignments(
                    candidate, recover_layer_turns=True,
                    turn_expected_color_by_label=known_initial_colors,
                    turn_color_priors=turn_color_priors,
                    record_all_turn_candidates=record_all_turn_candidates,
                    turn_geometry_frame=current_geometry[1] if current_geometry is not None else None,
                    turn_previous_move=last_committed_move,
                    turn_previous_states=tuple(signature for _, signature in committed_state_history),
                )
                current_state = cube_geometry_state.apply_moves(completed_candidate_moves(candidate))
                if (pose_reprojection_error_is_acceptable(candidate)
                        and not candidate.turn_color_unreliable and candidate_layers_are_aligned(candidate)
                        and current_state.is_solved(known_initial_colors)):
                    candidate = _annotate_solved_search(candidate, record.frame_index)
                else:
                    candidate = apply_move_lookahead(candidate, static_candidate, cube_geometry_state, record_index)
        candidate = replace(candidate, orientation_anchor=orientation_anchor_for_candidate(candidate),
                            pose_continuity_diagnostics=None)
        record.pose_diagnostic_candidate = None
        if motion_rejected or not pose_reprojection_error_is_acceptable(candidate):
            record.pose_candidate = None
            record.pose_candidates = ()
            record.pose_diagnostic_candidate = candidate
        else:
            previous_candidate = candidate
            record.pose_candidate = candidate
            record.pose_candidates = (candidate,)
            if recognize_moves:
                moves = completed_candidate_moves(candidate)
                for move in moves:
                    committed_state_history.append((record_index - 1, cube_geometry_state.signature()))
                    cube_geometry_state = cube_geometry_state.apply_move(move)
                if moves:
                    last_committed_move = moves[-1]
                    update_key = state_update_layer_direction(candidate)
                    if (last_committed_state_update_frame is not None
                            and record.frame_index - last_committed_state_update_frame == 1
                            and update_key == last_committed_state_update_key):
                        last_committed_state_update_streak += 1
                    else:
                        last_committed_state_update_streak = 1
                    last_committed_state_update_frame = record.frame_index
                    last_committed_state_update_key = update_key
                if (solved_frame_index is None and not candidate.turn_color_unreliable
                        and candidate_layers_are_aligned(candidate)
                        and cube_geometry_state.is_solved(known_initial_colors)):
                    solved_frame_index = record.frame_index
                    lookahead_forecast.clear()
                    lookahead_forecast_expected_state = None
                    lookahead_color_trial_cache.clear()
                    lookahead_geometry_cache.clear()
                    lookahead_frame_lower_bound_cache.clear()
                    candidate = _annotate_solved_search(candidate, solved_frame_index)
                    record.pose_candidate = candidate
                    action = "stopping move search" if move_search_started else "move search is not needed"
                    print(f"Cube solved at frame {solved_frame_index}; {action}.", flush=True)
        corrected_trace.observe_colors(record.frame_index, projected_color_observations(candidate))
        if turn_candidate_log is not None and recognize_moves:
            append_turn_candidate_log(turn_candidate_log, candidate, record.frame_index)
    if progress:
        print()
    return corrected_trace


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


def visible_cube_lattice_points(
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    object_points = CUBE_LATTICE_OBJECT_POINTS_MM
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    camera_origin = (
        -rotation.T @ np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    ).reshape(3)

    # Facing depends on the ray to this off-axis cube, not the optical axis.
    toward_camera = camera_origin[None, :] - np.asarray(object_points, dtype=np.float64)
    face_dot = np.einsum("nfj,nj->nf", CUBE_LATTICE_BOUNDARY_NORMALS, toward_camera)
    front_facing = np.any(
        CUBE_LATTICE_BOUNDARY_FACE_MASK & (face_dot > 0.0),
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


def three_point_sticker_pose_roots(
    object_points: np.ndarray, image_points: np.ndarray,
    camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Union SQPnP and perspective-three-point solutions without duplicates.

    SQPnP can return only one of the exact three-point solutions. P3P/AP3P
    expose the other branches so the full observation set can rank them.
    Solver failures are independent; an unstable solver cannot discard the
    valid roots from the others. Deduplicate within one correspondence only.
    """
    roots: list[tuple[np.ndarray, np.ndarray]] = []
    rotations: list[np.ndarray] = []
    for flag in (cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_P3P, cv2.SOLVEPNP_AP3P):
        try:
            if flag == cv2.SOLVEPNP_SQPNP:
                ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                    object_points, image_points, camera_matrix, dist_coeffs, flags=flag,
                )
            else:
                ok, rvecs, tvecs = cv2.solveP3P(
                    object_points, image_points, camera_matrix, dist_coeffs, flags=flag,
                )
        except cv2.error:
            continue
        if not ok:
            continue
        for rvec, tvec in zip(rvecs, tvecs, strict=True):
            rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
            tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
            if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
                continue
            rotation, _ = cv2.Rodrigues(rvec)
            # Merge numerical copies, preserving distinct physical branches.
            if any(np.linalg.norm(rotation - previous_rotation) <= 1e-5
                   and np.linalg.norm(tvec - previous_tvec) <= 1e-4
                   for previous_rotation, (_, previous_tvec) in zip(rotations, roots, strict=True)):
                continue
            roots.append((rvec.copy(), tvec.copy()))
            rotations.append(rotation)
    return tuple(roots)


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
    index_sequences = match.object_index_sequences if match.object_index_sequences is not None else sticker_correspondence_object_indices(
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
                roots = three_point_sticker_pose_roots(
                    object_points, observed_points, camera_matrix, dist_coeffs,
                )
                if not roots:
                    continue
                rvecs, tvecs = zip(*roots, strict=True)
                ok = True
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
        if len(candidate.object_points):
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


@lru_cache(maxsize=8)
def _turn_gpu_identity_layout(canonical_indices: tuple[int, ...]) -> tuple:
    """Keep the fixed face-position and identity lookup out of state scoring."""

    identities = canonical_sticker_identities()
    if any(index < 0 or index >= len(identities) for index in canonical_indices):
        raise ValueError("Invalid canonical sticker identity index")
    return (
        tuple((identities[index].cubie_index, identities[index].axis, identities[index].side)
              for index in canonical_indices),
        {identity.label: index for index, identity in enumerate(identities)},
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

    positions, identity_index_by_label = _turn_gpu_identity_layout(tuple(canonical_indices))
    remapped: list[int] = []
    for position in positions:
        label = geometry_state.facelets.get(position)
        if label is None or label not in identity_index_by_label:
            raise ValueError("Incomplete cube geometry state")
        remapped.append(identity_index_by_label[label])
    return np.asarray(remapped, dtype=np.int64)


@lru_cache(maxsize=128)
def _turn_color_projection_plan(configurations: tuple[tuple, ...]) -> tuple:
    """Project partial poses once; discrete offsets only permute identities.

    Rows map full color configurations to unique projected surfaces. Each
    permutation maps a base-pose face back to its source before the offset.
    Keep explicit projection support for non-quarter-turn offsets as well.
    """

    canonical = _turn_gpu_static_geometry()[-1]
    geometry_from_canonical = np.argsort(canonical)
    identity = CubeGeometryState.identity()
    projections, rows, sources = [], [], []
    unique, permutations, groups = {}, {}, {}
    group_ids = []
    for option, offset, axis, side, width, angle in configurations:
        group_ids.append(groups.setdefault((axis, side, width, offset), len(groups)))
        discrete = abs(offset / 90.0 - round(offset / 90.0)) < 1e-9
        base_angle = angle - offset if discrete else angle
        key = (axis, side, width, base_angle) if abs(base_angle) > 1e-6 else None
        if key not in unique:
            unique[key] = len(projections)
            projections.append((option, 0., axis, side, width, base_angle))
        rows.append(unique[key])
        move = (cube_rotation_to_notation(CubeLayerRotation(axis, side, offset, width))
                if discrete and abs(offset) > 1e-6 else None)
        if move not in permutations:
            moved = identity.apply_moves((move,)) if move is not None else identity
            permutations[move] = geometry_from_canonical[
                _turn_gpu_identity_indices_for_state(moved, canonical)]
        sources.append(permutations[move])
    return (tuple(projections), np.asarray(rows, dtype=np.int64), np.asarray(sources),
            np.asarray(group_ids))


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
    candidate: StickerPnPCandidate, torch_module: object,
    projected_face_centers: object, projected_face_polygons: object, visible_faces: object,
    sticker_identity_indices: np.ndarray, scale: float, max_error: float,
    color_matches: np.ndarray | None, *, return_tensors: bool = False,
    assignment_cache: dict | None = None,
    configuration_mapping: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[object, object]:
    """Batch exact class matches, with one vote per observed projected sticker.

    Background points are excluded. Multiple color points assigned to the same
    sticker contribute its best match, independently for each tested state.
    """
    torch = torch_module
    centers = tuple(getattr(candidate, "sticker_mask_centers", ()))
    batch = (len(configuration_mapping[0]) if configuration_mapping is not None
             else int(projected_face_centers.shape[0]))
    if color_matches is None or not centers:
        errors = projected_face_centers.new_zeros(batch)
        counts = errors.to(dtype=torch.int32)
    else:
        count = min(len(centers), color_matches.shape[1])
        observed = np.asarray(centers[:count], dtype=np.float64).reshape(-1, 2)
        key = (observed.tobytes(), count)
        cached = assignment_cache.get("value") if assignment_cache is not None and assignment_cache.get("key") == key else None
        if cached is None:
            observed_t = torch.as_tensor(observed, dtype=projected_face_centers.dtype,
                                         device=projected_face_centers.device)
            polygons = projected_face_polygons
            edges = torch.roll(polygons, shifts=-1, dims=2) - polygons
            relative = observed_t[None, None, :, None, :] - polygons[:, :, None, :, :]
            cross = (edges[:, :, None, :, 0] * relative[..., 1]
                     - edges[:, :, None, :, 1] * relative[..., 0])
            inside = (torch.all(cross >= -1e-4, dim=3)
                      | torch.all(cross <= 1e-4, dim=3)).permute(0, 2, 1)
            distances = torch.linalg.vector_norm(
                observed_t[None, :, None, :] - projected_face_centers[:, None, :, :], dim=-1,
            )
            distances = torch.where(inside & visible_faces[:, None, :], distances,
                                    torch.full_like(distances, float("inf")))
            nearest_distance, nearest = torch.min(distances, dim=2)
            valid = torch.isfinite(nearest_distance)
            if configuration_mapping is not None:
                rows, sources = configuration_mapping
                rows_t = torch.as_tensor(rows, dtype=torch.long, device=nearest.device)
                sources_t = torch.as_tensor(sources, dtype=torch.long, device=nearest.device)
                nearest = torch.gather(sources_t, 1, nearest[rows_t])
                valid = valid[rows_t]
            per_face = projected_face_centers.new_zeros((batch, projected_face_centers.shape[1]))
            per_face = per_face.scatter_add(1, nearest, valid.to(dtype=per_face.dtype))
            occupied = per_face > 0
            totals = torch.count_nonzero(occupied, dim=1)
            cached = (nearest, valid, totals)
            if assignment_cache is not None:
                assignment_cache.update(key=key, value=cached, builds=assignment_cache.get("builds", 0) + 1)
        elif assignment_cache is not None:
            assignment_cache["hits"] = assignment_cache.get("hits", 0) + 1
        nearest, valid, totals = cached
        identities = torch.as_tensor(sticker_identity_indices, dtype=torch.long,
                                      device=projected_face_centers.device)
        matrix = torch.as_tensor(color_matches[:, :count], dtype=projected_face_centers.dtype,
                                 device=projected_face_centers.device)
        indices = identities[nearest] if identities.ndim == 1 else identities[:, nearest]
        observations = torch.arange(count, device=matrix.device)
        matches = matrix[indices, observations] * valid
        if identities.ndim == 1:
            face_matches = projected_face_centers.new_zeros((batch, projected_face_centers.shape[1]))
            face_matches = face_matches.scatter_add(1, nearest, matches)
            correct = torch.sum(torch.clamp(face_matches, max=1), dim=-1)
            counts = totals
        else:
            face_matches = projected_face_centers.new_zeros((identities.shape[0], batch, projected_face_centers.shape[1]))
            face_matches = face_matches.scatter_add(2, nearest[None].expand_as(matches), matches)
            correct = torch.sum(torch.clamp(face_matches, max=1), dim=-1)
            counts = totals[None].expand(identities.shape[0], -1)
        errors = torch.where(counts > 0, 1.0 - correct / torch.clamp(counts, min=1),
                             torch.full_like(correct, float("inf")))
    if return_tensors:
        return errors, counts
    return (errors.detach().cpu().numpy().astype(np.float64, copy=False),
            counts.detach().cpu().numpy().astype(np.int32, copy=False))


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
    """Score whole-cube matches; layer membership affects diagnostics only."""

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
    valid_trial = torch.any(finite, dim=1)
    if observed_classes is not None:
        known = (observed_classes[None, :] >= 0) & (classes >= 0) & finite
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
    color_matches: np.ndarray | None = None,
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
        camera_origin = -camera_rotation.transpose(0, 1) @ tvec.reshape(3, 1)
        camera_origin = camera_origin.reshape(1, 1, 3)
        face_dot = torch.sum(
            transformed_normals * (camera_origin - transformed_points)[:, :, None, :], dim=3,
        )
        front_facing = (
            base_face_mask_t[None, :, :] & (face_dot > 0.0)
        ).any(dim=2)

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
            color_matches,
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
            # Sticker/layer support cannot turn a failed global landmark
            # measurement into a perfect fit. Preserve nonfinite errors even
            # when retaining all configurations for the full-search cache.
            combined_valid = bool(keypoint_valid[trial_index])
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
    """Keep all partial angles until the state's color evidence is scored."""

    trials = _score_turn_configurations_on_gpu(
        candidate,
        assignment_error_limit,
        scoring_device,
        color_matches=None,
        # Transfer all assignments together, rather than synchronizing once
        # for every retained angle when constructing the cached options.
        materialize_projections=True,
        angle_values=turn_geometric_search_angles(),
        enforce_layer_observation_gate=False,
        enforce_validity_gate=False,
    )
    if not trials:
        return None
    layers: set[tuple[int, int, int]] = set()
    static_trials: list[dict[str, object]] = []
    for trial in trials:
        key = (
            int(trial["layer_axis"]),
            int(trial["layer_side"]),
            int(trial["layer_count"]),
        )
        if bool(trial["is_baseline"]):
            static_trials.append(trial)
        layers.add(key)
    materialized_options: list[dict[str, object]] = []
    for trial in trials:
        option = dict(trial)
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
    if len(layers) != 12 or not static_trials:
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
    color_matches: np.ndarray | None,
    *, projection_cache: dict | None = None,
) -> float:
    """Relax sticker permutation constraints to bound every state/offset.

    Each observation receives its maximum likelihood over all identities.
    Keep the real per-option coverage, uncertainty weights and geometric cost.
    Every feasible cube assignment is at least as costly as this relaxation.
    """

    if color_matches is None:
        return min((float(option.get('combined_mean_error', -float('inf')))
                    for option in geometry_options), default=-float('inf'))
    likelihoods = np.asarray(color_matches, dtype=np.float64)
    if likelihoods.ndim != 2 or not likelihoods.size or not np.isfinite(likelihoods).all():
        return -float('inf')
    optimistic = np.broadcast_to(likelihoods.max(axis=0, keepdims=True), likelihoods.shape).copy()
    batches = _score_turn_color_offsets_for_states_on_gpu(
        candidate, geometry_options, assignment_error_limit, scoring_device, optimistic,
        (getattr(candidate, 'cube_geometry_state', None),), projection_cache=projection_cache,
    )
    trials = batches[0] if batches else None
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
    color_matches: np.ndarray | None,
    *, projection_cache: dict | None = None,
) -> list[dict[str, object]] | None:
    """Score only the -90/0/+90 color offsets for cached geometry options."""

    results = _score_turn_color_offsets_with_orientation_rescue(
        candidate, geometry_options, assignment_error_limit, scoring_device,
        color_matches, (getattr(candidate, "cube_geometry_state", None),), projection_cache=projection_cache,
    )
    return results[0] if results is not None else None


@lru_cache(maxsize=24)
def _geometry_symmetry_permutation(symmetry_key: tuple[int, ...]) -> tuple[tuple, ...]:
    """Relabel facelet coordinates without changing physical identities."""

    symmetry = np.asarray(symmetry_key, dtype=np.int32).reshape(3, 3)
    entries = []
    for position, axis, side in CubeGeometryState.identity().facelets:
        target_position = tuple((symmetry @ (np.asarray(position) - 1) + 1).tolist())
        target_axis = int(np.argmax(np.abs(symmetry[:, axis])))
        target_side = int(side * symmetry[target_axis, axis])
        entries.append(((target_position, target_axis, target_side), (position, axis, side)))
    return tuple(entries)


def _geometry_state_under_symmetry(
    state: CubeGeometryState, symmetry: np.ndarray,
) -> CubeGeometryState:
    permutation = _geometry_symmetry_permutation(tuple(int(v) for v in symmetry.ravel()))
    return CubeGeometryState({target: state.facelets[source] for target, source in permutation})


def _turn_trial_under_pose_symmetry(
    trial: dict[str, object], symmetry: np.ndarray, symmetry_index: int,
) -> dict[str, object]:
    """Express a cached projection's move/geometry in the alternative pose frame."""

    layer = transform_layer_rotation_by_symmetry(CubeLayerRotation(
        int(trial['layer_axis']), int(trial['layer_side']), 0., int(trial['layer_count']),
    ), symmetry)
    rotations = tuple(transform_layer_rotation_by_symmetry(r, symmetry)
                      for r in trial['layer_rotations'])
    offset = float(trial['color_offset'])
    move = cube_rotation_to_notation(replace(layer, angle_degrees=offset))
    geometry_label = (layer_rotation_debug_label(rotations) if rotations else
                      f'static {layer_face_debug_label(layer.axis, layer.side)}')
    transformed = dict(
        trial, trial_id=f"{trial['trial_id']}@pose{symmetry_index}",
        layer_axis=layer.axis, layer_side=layer.side, layer_count=layer.layer_count,
        layer_rotations=rotations, state_moves=(move,) if move is not None else (),
        label=f'{geometry_label} offset {offset:+.0f} [pose {symmetry_index}]',
        orientation_symmetry_index=symmetry_index,
        orientation_symmetry=symmetry.astype(int).tolist(),
    )
    if trial.get('assignment_object_points') is not None:
        transformed['assignment_object_points'] = transform_object_points_by_symmetry(
            trial['assignment_object_points'], symmetry,
        )
    return transformed


def _orientation_rescue_diagnostic(inherited_error: float, selected_error: float, tested: int) -> dict:
    return dict(threshold=PNP_TURN_ORIENTATION_RESCUE_COLOR_ERROR,
                minimum_color_accuracy=2.0 / 3.0, phase="before_move_search",
                inherited_color_error=inherited_error, selected_color_error=selected_error,
                tested_orientations=tested)


def balanced_geometry_cost(keypoint_error, sticker_error):
    """Map the two geometry components to equal halves of a 0..1 cost."""
    keypoint = np.asarray(keypoint_error, dtype=np.float64)
    sticker = np.asarray(sticker_error, dtype=np.float64)
    return np.where(np.isfinite(keypoint) & np.isfinite(sticker),
                    .5 * np.clip(keypoint / PNP_FINAL_ASSIGNMENT_ERROR_FRACTION, 0., 1.)
                    + .5 * np.clip(sticker / PNP_TURN_CENTER_PENALTY_SCALE, 0., 1.), float("inf"))


def balanced_robust_cost(geometry_robust_error, color_error):
    """Normalize the aggregate robust geometry diagnostic to the same budget."""
    geometry = np.asarray(geometry_robust_error, dtype=np.float64)
    geometry = np.where(np.isfinite(geometry), np.clip(
        geometry / (PNP_FINAL_ASSIGNMENT_ERROR_FRACTION + PNP_TURN_CENTER_PENALTY_SCALE), 0., 1.), float("inf"))
    return .5 * geometry + .5 * color_error


def _score_turn_color_offsets_with_orientation_rescue(
    candidate: StickerPnPCandidate,
    geometry_options: tuple[dict[str, object], ...],
    assignment_error_limit: float,
    scoring_device: object,
    color_matches: np.ndarray | None,
    geometry_states: tuple[CubeGeometryState | None, ...],
    *, projection_cache: dict | None = None, balance_geometry_color: bool = False,
    minimum_color_accuracy: float | None = None,
) -> list[list[dict[str, object]]] | None:
    """Rescue each unchanged state's orientation before generating its moves.

    Below 2/3 color accuracy, choose one of the 24 cube orientations using
    only the unchanged state's color agreement. Then evaluate moves in that
    orientation and map them back to the physical state's coordinates.
    A pose correction never commits a move or changes the physical state.
    """
    static_options = tuple(option for option in geometry_options
                           if option.get("is_baseline") and not option.get("layer_rotations"))[:1]
    if color_matches is None or not static_options:
        return _score_turn_color_offsets_for_states_on_gpu(
            candidate, geometry_options, assignment_error_limit, scoring_device,
            color_matches, geometry_states, projection_cache=projection_cache,
            balance_geometry_color=balance_geometry_color, minimum_color_accuracy=minimum_color_accuracy,
        )
    static = _score_turn_color_offsets_for_states_on_gpu(
        candidate, static_options, assignment_error_limit, scoring_device,
        color_matches, geometry_states, projection_cache=projection_cache, color_offsets=(0.,),
    )
    if static is None:
        return None
    rescue_indices, inherited_errors, best_errors, selected = [], {}, {}, {}
    for index, trials in enumerate(static):
        observed = next((trial for trial in trials if int(trial.get("color_match_count", 0)) > 0), None)
        if observed is None:
            continue
        error = float(observed["color_error"])
        if np.isfinite(error) and error > PNP_TURN_ORIENTATION_RESCUE_COLOR_ERROR:
            rescue_indices.append(index)
            inherited_errors[index] = best_errors[index] = error
    tested = 1
    for symmetry_index, symmetry in enumerate(CUBE_SYMMETRY_MATRICES):
        if not rescue_indices or np.array_equal(symmetry, np.eye(3)):
            continue
        relabeled = tuple(_geometry_state_under_symmetry(
            geometry_states[index] if geometry_states[index] is not None else CubeGeometryState.identity(),
            symmetry.T,
        ) for index in rescue_indices)
        alternatives = _score_turn_color_offsets_for_states_on_gpu(
            candidate, static_options, assignment_error_limit, scoring_device,
            color_matches, relabeled, projection_cache=projection_cache, color_offsets=(0.,),
        )
        if alternatives is None:
            continue
        tested += 1
        for index, trials in zip(rescue_indices, alternatives, strict=True):
            error = min((float(trial["color_error"]) for trial in trials
                         if int(trial.get("color_match_count", 0)) > 0), default=float("inf"))
            # Ties retain the inherited orientation and deterministic symmetry order.
            if error < best_errors[index]:
                best_errors[index] = error
                selected[index] = (symmetry_index, symmetry)
    oriented_states = list(geometry_states)
    for index, (_symmetry_index, symmetry) in selected.items():
        oriented_states[index] = _geometry_state_under_symmetry(
            geometry_states[index] if geometry_states[index] is not None else CubeGeometryState.identity(), symmetry.T,
        )
    results = _score_turn_color_offsets_for_states_on_gpu(
        candidate, geometry_options, assignment_error_limit, scoring_device,
        color_matches, tuple(oriented_states), projection_cache=projection_cache,
        balance_geometry_color=balance_geometry_color, minimum_color_accuracy=minimum_color_accuracy,
    )
    if results is None:
        return None
    for index in rescue_indices:
        diagnostic = _orientation_rescue_diagnostic(inherited_errors[index], best_errors[index], tested)
        results[index] = [dict(
            _turn_trial_under_pose_symmetry(trial, selected[index][1], selected[index][0])
            if index in selected else trial, orientation_rescue=diagnostic,
        ) for trial in results[index]]
    return results


def _score_turn_color_offsets_for_states_on_gpu(
    candidate: StickerPnPCandidate,
    geometry_options: tuple[dict[str, object], ...],
    assignment_error_limit: float,
    scoring_device: object,
    color_matches: np.ndarray | None,
    geometry_states: tuple[CubeGeometryState | None, ...],
    *, projection_cache: dict | None = None,
    color_offsets: tuple[float, ...] = (-90.0, 0.0, 90.0),
    balance_geometry_color: bool = False,
    minimum_color_accuracy: float | None = None,
) -> list[list[dict[str, object]]] | None:
    """Reuse partial-pose assignments and score frontier states by identity gathers."""

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
            for offset in color_offsets:
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
        projection_configurations, projection_rows, source_permutations, group_ids = _turn_color_projection_plan(
            tuple(configurations))
        projection_count = len(projection_configurations)
        if projected is None:
            axes = torch.as_tensor(
                [item[2] for item in projection_configurations],
                dtype=torch.long,
                device=device,
            )
            sides = torch.as_tensor(
                [item[3] for item in projection_configurations],
                dtype=dtype,
                device=device,
            )
            layer_counts = torch.as_tensor(
                [item[4] for item in projection_configurations],
                dtype=dtype,
                device=device,
            )
            angles = torch.as_tensor(
                [item[5] for item in projection_configurations],
                dtype=dtype,
                device=device,
            )
            active = torch.as_tensor(
                [abs(item[5]) > 1e-6 for item in projection_configurations],
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
                projection_count, -1, -1, -1
            )
            point_center = torch.zeros((projection_count, 3), dtype=dtype, device=device)
            point_center[torch.arange(projection_count, device=device), axes] = (
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
                sticker_normals_t[None, :, :].expand(projection_count, -1, -1),
            )
            transformed_face_normals = torch.where(
                sticker_layer_mask[:, :, None],
                transformed_face_normals,
                sticker_normals_t[None, :, :].expand(projection_count, -1, -1),
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
                                        projection_count=projection_count, configuration_count=count,
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
            color_matches,
            return_tensors=True,
            assignment_cache=projection_cache.get("assignments") if projection_cache is not None else None,
            configuration_mapping=(projection_rows, source_permutations),
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

        # For the same layer and discrete offset, different partial angles
        # lead to the same next state. Choose their best combined fit for
        # each input state before building Python trial dictionaries. This
        # keeps frontier/cache size bounded while retaining angle alternatives
        # through color scoring (e.g. Y--15 versus Y-+45).
        configured_options = [geometry_options[item[0]] for item in configurations]
        geometry_scores = np.asarray([float(option["combined_mean_error"]) for option in configured_options])
        if balance_geometry_color:
            geometry_scores = np.where(np.isfinite(geometry_scores), balanced_geometry_cost(
                [option["keypoint_error"] for option in configured_options],
                [option["sticker_error"] for option in configured_options]), float("inf"))
        inliers = np.asarray([int(option.get("inlier_count", 0)) for option in configured_options])
        p90s = np.asarray([float(option.get("p90_error", float("inf"))) for option in configured_options])
        layer_inliers = np.asarray([int(option.get("layer_inlier_count", 0)) for option in configured_options])
        results_by_state: list[list[dict[str, object]]] = []
        for state_index in range(len(geometry_states)):
            results: list[dict[str, object]] = []
            errors = color_errors_np[state_index]
            costs = geometry_scores + np.where(np.isfinite(errors), PNP_TURN_COLOR_WEIGHT * errors, float("inf"))
            if balance_geometry_color:
                costs = .5 * geometry_scores + .5 * errors
            if minimum_color_accuracy is not None:
                costs = np.where(errors <= 1. - minimum_color_accuracy, costs, float("inf"))
            eligible = (np.flatnonzero(np.isfinite(costs)) if minimum_color_accuracy is not None
                        else np.arange(len(costs)))
            if not len(eligible):
                results_by_state.append([])
                continue
            order = eligible[np.lexsort((-layer_inliers[eligible], p90s[eligible], -inliers[eligible],
                                         costs[eligible], group_ids[eligible]))]
            ordered_groups = group_ids[order]
            best_indices = np.sort(order[np.r_[True, ordered_groups[1:] != ordered_groups[:-1]]])
            for index in best_indices:
                option_index, offset, axis, side, layer_count, _angle = configurations[index]
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
                combined_score = float(costs[index])
                # Both geometry and color contribute before choosing a
                # partial angle for this discrete state change.
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
                        "robust_error": (float(balanced_robust_cost(option.get("robust_error", float("inf")), color_error))
                                         if balance_geometry_color else
                                         float(option.get("robust_error", float("inf"))) + color_cost),
                        **({"score_components": dict(geometry=float(geometry_scores[index]), color=color_error,
                                                     geometry_weight=.5, color_weight=.5)}
                           if balance_geometry_color else {}),
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

    # Otherwise compare color mismatch fractions, where lower is better: a
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
    color_matches = turn_color_match_matrix(
        candidate, color_priors, expected_color_by_label,
    )
    if color_matches is None:
        color_matches = getattr(candidate, "turn_color_matches", None)
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
        color_matches,
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
        turn_color_matches=color_matches,
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
        color_matches=color_matches,
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
    color_matches = turn_color_match_matrix(
        candidate, color_priors, expected_color_by_label,
    )
    if color_matches is None:
        color_matches = getattr(candidate, "turn_color_matches", None)
    scoring_device = _turn_torch_scoring_device("mps")
    if scoring_device is None:
        raise RuntimeError("Move recognition requires an available Apple MPS device")
    color_trials = _score_turn_color_offsets_on_gpu(
        candidate,
        options,
        assignment_error_limit,
        scoring_device,
        color_matches,
    )
    if not color_trials:
        return replace(
            candidate,
            turn_color_matches=color_matches,
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
            turn_color_matches=color_matches,
            turn_fit_evaluated=True,
            turn_state_moves=(),
            turn_fit_articulated_scores=None,
        )
    best = (None if bool(best_trial["is_baseline"])
            and best_trial.get('orientation_symmetry') is None else best_trial)
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
            turn_color_matches=color_matches,
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
            **{key: trial[key] for key in (
                'orientation_symmetry_index', 'orientation_symmetry', 'orientation_rescue',
            ) if key in trial},
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
        turn_color_matches=color_matches,
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

    if trial.get('orientation_symmetry') is not None:
        static_candidate = transform_candidate_by_symmetry(
            static_candidate, np.asarray(trial['orientation_symmetry'], dtype=np.float64),
        )
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
        if len(object_points):
            projected_sticker_points, _ = cv2.projectPoints(
                object_points,
                static_candidate.rvec,
                static_candidate.tvec,
                static_candidate.camera_matrix,
                static_candidate.dist_coeffs,
            )
            projected_sticker_points = projected_sticker_points.reshape(-1, 2)
        else:
            projected_sticker_points = np.empty((0, 2))
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
        object_symmetry=static_candidate.object_symmetry,
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


def _frame_forces_static_hold(frame_index: int | None, turning_start_frame: int | None) -> bool:
    """Treat the configured zero-based boundary as the last static frame."""

    return turning_start_frame is not None and (0 if frame_index is None else frame_index) <= turning_start_frame


def _hold_before_turning_start(
    candidate: StickerPnPCandidate,
    frame_index: int | None,
    turning_start_frame: int,
) -> StickerPnPCandidate:
    """Keep the static pose with no partial turns, offsets, or state updates."""

    return _hold_without_turn_search(candidate, dict(
        version=13, status='forced_static', reason='turning_start_frame',
        turning_start_frame=turning_start_frame, frames=[0 if frame_index is None else frame_index],
        evaluated_states=0,
    ))


def candidate_layers_are_aligned(candidate: StickerPnPCandidate) -> bool:
    """A solved discrete state is insufficient while a layer is still turning."""

    return all(
        np.isfinite(rotation.angle_degrees)
        and np.isclose(rotation.angle_degrees / 90.0,
                       round(rotation.angle_degrees / 90.0), rtol=0.0, atol=1e-8)
        for rotation in physical_sticker_rotations(candidate)
    )


def _annotate_solved_search(
    candidate: StickerPnPCandidate, solved_frame_index: int,
) -> StickerPnPCandidate:
    """Preserve the final solving move and its current-frame search evidence."""

    diagnostics = dict(version=13, status="stopped_solved", frames=[solved_frame_index],
                       evaluated_states=0)
    diagnostics.update(candidate.turn_search_diagnostics or {})
    diagnostics.update(stopped=True, stop_reason="cube_solved", solved_frame_index=solved_frame_index)
    return replace(candidate, turn_search_diagnostics=diagnostics)


def _hold_after_solved(
    candidate: StickerPnPCandidate, frame_index: int | None, solved_frame_index: int,
) -> StickerPnPCandidate:
    return _hold_without_turn_search(candidate, dict(
        version=13, status="stopped_solved", stopped=True, stop_reason="cube_solved",
        solved_frame_index=solved_frame_index, frames=[0 if frame_index is None else frame_index],
        evaluated_states=0,
    ))


def _hold_without_turn_search(
    candidate: StickerPnPCandidate, diagnostics: dict[str, object],
) -> StickerPnPCandidate:
    return replace(
        candidate,
        layer_rotations=(), sticker_layer_rotations=(), turn_state_moves=(),
        turn_color_offset_degrees=0.0, turn_selected_trial_id=None,
        turn_candidate_scores=(), turn_candidate_diagnostics=(), turn_candidate_debug=(),
        turn_layer_scores=(), turn_fit_articulated_scores=None, turn_fit_evaluated=False,
        turn_fit_color_score=float('inf'), turn_color_matches=None, turn_color_unreliable=False,
        turn_fit_search_rvec=None, turn_fit_search_tvec=None,
        turn_fit_lookahead_action_score=float('inf'), turn_fit_lookahead_static_score=float('inf'),
        turn_fit_lookahead_action_future_scores=(), turn_fit_lookahead_static_future_scores=(),
        turn_fit_lookahead_rejected=False,
        turn_search_diagnostics=diagnostics,
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


def turn_color_match_matrix(
    candidate: StickerPnPCandidate,
    color_priors: dict[str, np.ndarray] | None = None,
    expected_color_by_label: dict[str, str] | None = None,
) -> np.ndarray | None:
    """Return binary color equality [physical sticker, observed color point].

    No probability, RGB observation, or color regression participates. Priors
    are accepted only as the existing one-hot encoding of the known scramble.
    """
    colors = tuple(getattr(candidate, "sticker_color_names", ()))
    if not colors:
        return None
    expected = dict(expected_color_by_label or {})
    if color_priors:
        for label, values in color_priors.items():
            values = np.asarray(values)
            if values.shape == (len(GLOBAL_COLOR_NAMES),) and np.count_nonzero(values == 1) == 1:
                expected.setdefault(label, GLOBAL_COLOR_NAMES[int(np.argmax(values))])
    if not expected:
        return None
    return np.asarray([
        [expected.get(identity.label) == color for color in colors]
        for identity in canonical_sticker_identities()
    ], dtype=np.float64)


def sticker_mask_color_turn_fit_metrics(
    candidate: StickerPnPCandidate, rvec: np.ndarray, tvec: np.ndarray,
    layer_rotations: tuple[CubeLayerRotation, ...], max_error: float,
    color_matches: np.ndarray | None,
) -> tuple[float, int]:
    """Return 1 - correct/observed, counting each containing sticker once."""
    if color_matches is None:
        return 0.0, 0
    trial = replace(candidate, layer_rotations=layer_rotations, sticker_layer_rotations=None)
    faces = projected_sticker_faces(trial, rvec, tvec)
    identities = {identity.label: index for index, identity in enumerate(canonical_sticker_identities())}
    groups = {}
    for index, point in enumerate(candidate.sticker_mask_centers):
        containing = [face for face in faces if cv2.pointPolygonTest(
            np.asarray(face[2], dtype=np.float32), tuple(float(v) for v in point), False,
        ) >= 0]
        if not containing:
            continue
        face = min(containing, key=lambda face: float(np.linalg.norm(np.mean(face[2], axis=0) - point)))
        label = face[0].label
        groups[label] = max(groups.get(label, 0.0), float(color_matches[identities[label], index]))
    if not groups:
        return float("inf"), 0
    return 1.0 - sum(groups.values()) / len(groups), len(groups)


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


def refine_static_pose_from_all_observations(
    candidate: StickerPnPCandidate, validate_sticker_facing: bool = True,
) -> StickerPnPCandidate:
    """Try bounded full-observation PnP/rematching before the inlier solve.

    A small planar seed can miss an entire visible face with the 5% inlier
    gate. Rematching all measured landmarks lets that face correct the pose.
    Accept only a globally improved fit that puts *every* observation inside
    the existing gate; partial turns/outliers retain the robust inlier path.
    This is an optional rigid proposal, never a frame-acceptance requirement:
    an unsuccessful attempt returns the input pose for inlier refinement.
    The measured sticker anchors remain fixed throughout rematching.
    """
    if (candidate.assignment_image_points is None or candidate.assignment_object_points is None
            or candidate.assignment_projected_points is None or candidate.assignment_observed_class_ids is None):
        return candidate
    observed = np.asarray(candidate.assignment_image_points, dtype=np.float32).reshape(-1, 2)
    objects = np.asarray(candidate.assignment_object_points, dtype=np.float32).reshape(-1, 3).copy()
    projected = np.asarray(candidate.assignment_projected_points, dtype=np.float32).reshape(-1, 2).copy()
    classes = np.asarray(candidate.assignment_observed_class_ids, dtype=np.int32).reshape(-1)
    if (len(observed) < 4 or not (len(observed) == len(objects) == len(projected) == len(classes))
            or not np.isfinite(observed).all() or not np.isfinite(objects).all() or not np.isfinite(projected).all()):
        return candidate
    fixed = candidate.assignment_fixed_mask
    if fixed is None:
        fixed = np.arange(len(observed)) < len(candidate.image_points)
    fixed = np.asarray(fixed, dtype=bool).reshape(-1)
    if len(fixed) != len(observed) or int(fixed.sum()) != len(candidate.object_points):
        return candidate
    anchor_indices = np.flatnonzero(fixed)
    baseline = assignment_geometric_quality(observed, projected)
    best = candidate
    best_mean = baseline[0]
    rvec, tvec = candidate.rvec.copy(), candidate.tvec.copy()
    for _ in range(6):
        unique = unique_object_point_indices(objects, observed, projected)
        if len(unique) < 4:
            break
        try:
            ok, next_rvec, next_tvec = cv2.solvePnP(
                objects[unique], observed[unique], candidate.camera_matrix, candidate.dist_coeffs,
                rvec=rvec.copy(), tvec=tvec.copy(), useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            break
        if (not ok or not np.isfinite(next_rvec).all() or not np.isfinite(next_tvec).all()
                or float(next_tvec[2, 0]) <= 0.0
                or (validate_sticker_facing and not canonical_sticker_is_camera_facing(
                    candidate.object_points, next_rvec, next_tvec))):
            break
        score, rows, columns, lattice_projected, lattice_objects, lattice_classes = evaluate_pose_hypothesis(
            next_rvec, next_tvec, observed, classes, candidate.image_size,
            candidate.camera_matrix, candidate.dist_coeffs, float("inf"), anchor_indices, candidate.object_points,
        )
        if not np.isfinite(score) or len(rows) != len(observed):
            break
        objects[rows] = lattice_objects[columns]
        projected[rows] = lattice_projected[columns]
        residuals = np.linalg.norm(observed - projected, axis=1)
        mean, p90 = float(np.mean(residuals)), float(np.percentile(residuals, 90))
        proposal = replace(candidate, rvec=next_rvec.copy(), tvec=next_tvec.copy())
        limit = PNP_FINAL_ASSIGNMENT_ERROR_FRACTION * projected_cube_diameter(proposal)
        if (np.isfinite(limit) and np.all(residuals <= limit) and mean < best_mean - 1e-6
                and p90 <= baseline[1] + 1e-6):
            anchor_projected, _ = cv2.projectPoints(
                candidate.object_points, next_rvec, next_tvec, candidate.camera_matrix, candidate.dist_coeffs,
            )
            anchor_projected = anchor_projected.reshape(-1, 2)
            assigned_classes = np.empty_like(classes)
            assigned_classes[rows] = lattice_classes[columns]
            best = replace(
                proposal, projected_sticker_points=anchor_projected,
                reprojection_error=float(np.linalg.norm(anchor_projected - candidate.image_points, axis=1).mean()),
                overall_reprojection_error=float(np.mean(huber_loss(residuals))),
                assignment_object_points=objects.copy(), assignment_projected_points=projected.copy(),
                assignment_projected_class_ids=assigned_classes,
                final_solve_image_points=observed[unique].copy(),
                pre_final_rvec=candidate.rvec.copy(), pre_final_tvec=candidate.tvec.copy(),
            )
            best_mean = mean
        converged = (np.linalg.norm(next_tvec - tvec) <= 1e-5
                     and np.linalg.norm(next_rvec - rvec) <= 1e-6)
        rvec, tvec = next_rvec, next_tvec
        if converged:
            break
    return best


def solve_pnp_from_low_error_assignments(
    candidate: StickerPnPCandidate,
    validate_sticker_facing: bool = True,
    record_all_turn_candidates: bool = False,
    recover_layer_turns: bool = False,
    turn_color_priors: dict[str, np.ndarray] | None = None,
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
        candidate = refine_static_pose_from_all_observations(candidate, validate_sticker_facing)
        assignment_error_limit = PNP_FINAL_ASSIGNMENT_ERROR_FRACTION * projected_cube_diameter(candidate)
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

    Color-point seeds use pixel diagonal-center error, preferring four
    measurements on ties, then confidence. Up to three distinct landmark
    groups are fitted, retaining ambiguous class-consistent sticker types.
    All successful candidates are returned in score order so temporal logic
    can keep alternatives alive. If none of the four-point stickers yields a
    pose, legacy mask seeds fall back to three-point stickers.
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
        color_seeds = all(match.center_error is not None for match in prioritized_matches)
        prioritized = sorted(prioritized_matches, key=sticker_match_priority_key)
        if color_seeds:
            groups: dict[tuple[int, ...], list[StickerKeypointMatch]] = {}
            for match in prioritized:
                group = groups.setdefault(tuple(sorted(match.keypoint_indices)), [])
                if not any(existing.sticker_type == match.sticker_type for existing in group):
                    group.append(match)
            prioritized_groups = list(groups.values())
        else:
            prioritized_groups = [[match] for match in prioritized[:MAX_STICKER_PNP_CANDIDATES]]
        solved_groups = 0
        for group in prioritized_groups:
            before = len(candidates)
            for match in group:
                candidates.extend(solve_match(match))
            if len(candidates) > before:
                solved_groups += 1
            if solved_groups >= MAX_STICKER_PNP_CANDIDATES:
                break
        candidates.sort(
            key=lambda candidate: (
                pose_hypothesis_visual_score(candidate),
                -candidate.sticker_average_confidence,
            )
        )
        return tuple(candidates)

    def solve_match(match: StickerKeypointMatch) -> tuple[StickerPnPCandidate, ...]:
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
        return solved_candidates

    if all(match.center_error is not None for match in valid_matches):
        return solve_matches(valid_matches)
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
    # Keep the thresholded pool explicit for callers that provide lightweight
    # candidates without full reprojection geometry; normal pose candidates
    # have already passed the same hard acceptance gate above.
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
        and float(point_confidence) >= pose_keypoint_confidence_threshold(int(class_id), point_threshold)
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
        if (float(box_confidence) < class_threshold
                or float(point_confidence) < pose_keypoint_confidence_threshold(int(class_id), point_threshold)):
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
    """Project display X+/Y+/Z+ toward the red/white/green center stickers."""

    if not np.isfinite(axis_length) or float(axis_length) <= 0.0:
        return None
    centers = {identity.label: identity for identity in canonical_sticker_identities()
               if identity.category == "center"}
    labels = getattr(candidate, "orientation_axis_center_labels", None)
    if labels is None:
        by_face = {identity.face: identity.label for identity in centers.values()}
        labels = tuple(by_face[face] for face in ORIENTATION_AXIS_DEFAULT_FACES)
    rotations = getattr(candidate, "sticker_layer_rotations", None)
    if rotations is None:
        rotations = getattr(candidate, "layer_rotations", ())
    object_points = [np.zeros(3, dtype=np.float64)]
    for label in labels:
        identity = centers[label]
        cubie_index, axis, side = sticker_geometry_key(candidate, identity)
        cubie_center = cubelet_center_object_point(*cubie_index)
        face_center = cubie_center.copy()
        face_center[axis] += side * CUBE_LAYER_SIZE_MM / 2.0
        _, transformed = apply_cubelet_layer_rotations(cubie_center, face_center[None, :], tuple(rotations))
        direction = transformed[0]
        object_points.append(direction * (float(axis_length) / np.linalg.norm(direction)))
    object_points = np.asarray(object_points, dtype=np.float32)
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

    Display X+/Y+/Z+ follow the red/white/green centers, including committed
    center permutations and partial wide turns, in the wireframe's pose.
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
    candidates = [dict(trial) for trial in candidates]
    for trial in candidates:
        observed = trial.get("color_match_count", 0)
        error = trial.get("color_error")
        if observed and isinstance(error, (float, int)) and np.isfinite(error) and 0 <= error <= 1:
            trial.update(
                color_observed_count=int(observed),
                color_correct_count=int(round((1.0 - error) * observed)),
                color_score=1.0 - error,
            )
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
    color_matches = (
        getattr(candidate, "turn_color_matches", None)
        if candidate is not None
        else None
    )
    color_identity_labels = (
        [identity.label for identity in canonical_sticker_identities()]
        if color_matches is not None
        else None
    )
    color_observation_centers = (
        getattr(candidate, "sticker_mask_centers", ())
        if candidate is not None
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
            "sticker_color_names": getattr(candidate, "sticker_color_names", ()),
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
        "log_schema_version": 3,
        "geometry_scoring_version": PNP_TURN_GEOMETRY_SCORE_VERSION,
        "color_scoring": "direct_match_fraction",
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
        "turn_color_observation_names": getattr(candidate, "sticker_color_names", ()) if candidate else (),
        "turn_color_matches": color_matches,
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
        "lookahead_frames": (
            search_diagnostics.get("requested_future_frames", PNP_TURN_LOOKAHEAD_FRAMES)
            if isinstance(search_diagnostics, dict)
            else PNP_TURN_LOOKAHEAD_FRAMES
        ),
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
        "pose_continuity": getattr(candidate, "pose_continuity_diagnostics", None),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log_file:
        json.dump(json_safe(record), log_file, separators=(",", ":"))
        log_file.write("\n")


def projected_turn_color_matches_by_identity(
    candidate: StickerPnPCandidate,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> dict[str, float]:
    """Map each projected sticker to its best contained color-class match."""
    matrix = getattr(candidate, "turn_color_matches", None)
    if matrix is None:
        return {}
    faces = projected_sticker_faces(candidate, rvec, tvec)
    identity_indices = {identity.label: index for index, identity in enumerate(canonical_sticker_identities())}
    result = {}
    for observation_index, point in enumerate(candidate.sticker_mask_centers):
        containing = [face for face in faces if cv2.pointPolygonTest(
            np.asarray(face[2], dtype=np.float32), tuple(float(v) for v in point), False,
        ) >= 0]
        if not containing:
            continue
        face = min(containing, key=lambda face: float(np.linalg.norm(np.mean(face[2], axis=0) - point)))
        label = face[0].label
        result[label] = max(result.get(label, 0.), float(matrix[identity_indices[label], observation_index]))
    return result


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
            projected_turn_color_matches_by_identity(candidate, rvec, tvec)
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
    rvec: np.ndarray | None = None,
    orientation_axis_length: float = 24.0,
    show_sticker_identities: bool = True,
    show_orientation_axes: bool = True,
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
        "virtual cube | sticker color match" if show_sticker_identities else "virtual cube",
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
    turn_likelihoods_by_identity = projected_turn_color_matches_by_identity(
        candidate,
        candidate.rvec,
        candidate.tvec,
    )
    # The displayed value is binary agreement with the expected sticker color.
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
    if show_orientation_axes:
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
    # Draw only the finalized measurement, without a pre-refinement wireframe.
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
        pose_label = "final"
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
    anchor_points = getattr(candidate, "image_points", None)
    anchor_text = ("--" if anchor_points is not None and not len(anchor_points) else
                   f"{anchor_error:.1f}px" if np.isfinite(anchor_error) else "inf")
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
    turn_candidate_log: Path | None = None,
    turn_candidate_image_dir: Path | None = None,
    frame_index: int | None = None,
    symmetry_tracker: CubeSymmetryTracker | None = None,
    pose_hypothesis_buffer: PoseHypothesisBuffer | None = None,
    sticker_color_names: dict[str, str] | None = None,
    cached_final_only: bool = False,
    *,
    neural_cache: list[NeuralFrameData] | None = None,
    neural_data: NeuralFrameData | None = None,
    neural_only: bool = False,
    candidate_only: bool = False,
    move_accumulator: MoveAccumulator | None = None,
) -> tuple[np.ndarray, str, int]:
    height, width = frame.shape[:2]

    def overlay_sticker_masks(output: np.ndarray, pose: StickerPnPCandidate | None = None) -> np.ndarray:
        if not getattr(args, "show_colored_masks", False) or pose is None:
            return output
        observations = projected_color_observations(pose)
        overlay = output.copy()
        for identity, _depth, polygon in projected_sticker_faces(pose, pose.rvec, pose.tvec):
            colors = observations.get(identity.label, ())
            if colors:
                expected = (sticker_color_names or {}).get(identity.label)
                color = expected if expected in colors else colors[0]
                cv2.fillConvexPoly(overlay, np.round(polygon).astype(np.int32), STICKER_COLORS_BGR[color])
        return cv2.addWeighted(overlay, 0.5, output, 0.5, 0)

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
        log_turn_frame(None)
        return output, roi_source, 0

    if neural_data is None:
        model_roi = resize_to_square(roi, args.view_size)
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
                    sticker_masks=None,
                    local_points=np.asarray(local_points, dtype=np.float32).copy(),
                    point_confidences=np.asarray(point_confidences, dtype=np.float32).copy(),
                    class_ids=np.asarray(class_ids, dtype=np.int32).copy(),
                    box_confidences=np.asarray(box_confidences, dtype=np.float32).copy(),
                )
            )
    else:
        local_points = np.asarray(neural_data.local_points)
        point_confidences = np.asarray(neural_data.point_confidences)
        class_ids = np.asarray(neural_data.class_ids)
        box_confidences = np.asarray(neural_data.box_confidences)
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
        if move_accumulator is not None:
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
        log_turn_frame(None)
        return output, roi_source, 0

    usable = usable_pose_points(local_points, point_confidences, class_ids,
                                box_confidences, args.keypoint_conf, args.pose_class_conf)
    landmark_mask = usable & (class_ids >= 0) & (class_ids < 3)
    color_mask = usable & (class_ids >= 3)
    sticker_matches = (() if neural_only else sticker_keypoint_matches_for_colors(
        local_points[landmark_mask], point_confidences[landmark_mask],
        class_ids[landmark_mask], local_points[color_mask],
    ))
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

    # Move recognition is a per-frame computation.  It must still run when
    # pose drawing is disabled, because ``--recognize-moves`` is an analysis
    # request rather than a rendering option.
    should_generate_candidate = pnp_candidate is None and (
        args.show_pnp or recognize_moves or candidate_only
    )
    if should_generate_candidate:
        current_candidates = best_sticker_pnp_candidates(
            frame_points[landmark_mask], class_ids[landmark_mask], sticker_matches,
            (width, height), args.pnp_focal_scale,
            scoring_points=frame_points[landmark_mask], prepare_mask_geometry=False,
        )
        current_candidates = tuple(attach_color_keypoints(
            hypothesis, frame_points[color_mask],
            tuple(CLASS_NAMES[int(cid)] for cid in class_ids[color_mask]),
        ) for hypothesis in current_candidates)
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
            if pose_hypothesis_buffer is not None else (),
            max_core_displacement_mm=POSE_MOTION_DISPLACEMENT_TOLERANCE_MM,
        )
        if selected_candidate is not None:
            initial_state = CubeState.from_scramble(getattr(args, "initial_scramble", ()))
            expected_colors = {
                identity.label: initial_state.color_at((identity.cubie_index, identity.axis, identity.side))
                for identity in canonical_sticker_identities()
            }
            geometry_state = (move_accumulator._geometry_state.copy()
                              if move_accumulator is not None else CubeGeometryState.identity())
            selected_candidate = match_orientation_to_colors(
                replace(selected_candidate, cube_geometry_state=geometry_state), expected_colors,
            )
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
                turning_start_frame = getattr(args, 'turning_start_frame', None)
                if (move_accumulator is not None and move_accumulator._state is None
                        and hasattr(args, "initial_scramble")):
                    initial_state = CubeState.from_scramble(args.initial_scramble)
                    move_accumulator._initialize_state({
                        identity.label: initial_state.color_at(
                            (identity.cubie_index, identity.axis, identity.side)
                        ) for identity in canonical_sticker_identities()
                    })
                if (move_accumulator is not None
                        and move_accumulator.solved_frame_index is not None):
                    selected_candidate = _hold_after_solved(
                        selected_candidate, frame_index, move_accumulator.solved_frame_index,
                    )
                elif _frame_forces_static_hold(frame_index, turning_start_frame):
                    selected_candidate = _hold_before_turning_start(
                        selected_candidate, frame_index, turning_start_frame,
                    )
                else:
                    selected_candidate = solve_pnp_from_low_error_assignments(
                        selected_candidate,
                        recover_layer_turns=True,
                        turn_expected_color_by_label=expected_colors,
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
        if move_accumulator is not None:
            move_accumulator.observe(
                pnp_candidate,
                displacement_candidate=diagnostic_candidate,
                sticker_color_names=sticker_color_names,
            )
            if (recognize_moves and pnp_candidate is not None
                    and move_accumulator.stop_if_solved(pnp_candidate, frame_index)):
                pnp_candidate = _annotate_solved_search(
                    pnp_candidate, move_accumulator.solved_frame_index,
                )
        if pnp_candidate is not None:
            if args.show_pnp:
                output = draw_sticker_pnp_candidate(
                    output,
                    pnp_candidate,
                    args.show_status,
                    getattr(args, "show_orientation_axes", True),
                    getattr(args, "orientation_axis_length", 24.0),
                    turn_candidate_log if recognize_moves else None,
                    turn_candidate_image_dir if recognize_moves else None,
                    frame_index,
                    getattr(args, "show_sticker_identities", False),
                    None,
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
            getattr(args, "show_orientation_axes", True),
            getattr(args, "orientation_axis_length", 24.0),
            turn_candidate_log if recognize_moves else None,
            turn_candidate_image_dir if recognize_moves else None,
            frame_index,
            getattr(args, "show_sticker_identities", False),
            None,
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
    args: argparse.Namespace,
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
            turn_candidate_log=turn_candidate_log,
            turn_candidate_image_dir=turn_candidate_image_dir,
            frame_index=record.frame_index,
            symmetry_tracker=None,
            pose_hypothesis_buffer=None,
            sticker_color_names=None,
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
    args: argparse.Namespace,
    *,
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
            args,
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
                turn_candidate_log=None,
                turn_candidate_image_dir=None,
                frame_index=frame_index,
                symmetry_tracker=None,
                pose_hypothesis_buffer=None,
                sticker_color_names=None,
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
    sticker_color_names: dict[str, str] | None = None,
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
            rendered, _roi_source, _point_count = render_frame(
                record.frame,
                None,
                None,
                roi_tracker,
                args,
                turn_candidate_log=None,
                turn_candidate_image_dir=None,
                frame_index=record.frame_index,
                symmetry_tracker=None,
                pose_hypothesis_buffer=None,
                sticker_color_names=sticker_color_names,
                cached_final_only=True,
                neural_data=record,
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


def parse_turning_start_frame(value: str) -> int:
    """Parse the inclusive last-static-frame index."""

    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError('Turning start frame must be a nonnegative integer') from error
    if result < 0:
        raise argparse.ArgumentTypeError('Turning start frame must be a nonnegative integer')
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
        help="Minimum landmark keypoint confidence (default: 0.05); sticker color keypoints have a 0.25 floor.",
    )
    parser.add_argument("--track-conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--view-size", type=int, default=256)
    parser.add_argument("--max-det", type=int, default=128)
    parser.add_argument(
        "--show-colored-masks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Color observed projected stickers using the pose model's color classes.",
    )
    parser.add_argument("--roi-padding", type=float, default=0.4)
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
        help="Maximum frame/state evaluations per decision, including seeds and retries (default: 4096; unlimited disables).",
    )
    parser.add_argument(
        "--turning-start-frame", type=parse_turning_start_frame, default=None,
        help="Force static holds with no layer offsets through this zero-based frame, inclusive; moves are allowed afterward (default: unrestricted).",
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
    parser.add_argument("--save", type=Path, help="Output image path, or one frame with --once.")
    parser.add_argument(
        "--save-video",
        type=Path,
        help="Output path for the first pass containing neural outputs only.",
    )
    parser.add_argument(
        "--save-final-video",
        type=Path,
        dest="save_final_video",
        help=(
            "Output path for the final cached video pass with pose and move overlays."
        ),
    )
    parser.add_argument("--once", action="store_true", help="Process one video/camera frame and exit.")
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.set_defaults(display=True)
    return parser


def process_cached_poses(
    cache: list[NeuralFrameData], args: argparse.Namespace,
    known_initial_colors: dict[str, str],
    source_fps: float = PNP_TURN_LOOKAHEAD_REFERENCE_FPS,
) -> None:
    """Recover poses and optional moves directly from the cached model predictions."""
    trace = StickerColorTrace.create()
    needs_pose = args.show_pnp or args.recognize_moves or args.show_colored_masks
    if needs_pose:
        print("Refining poses and matching observed sticker colors to the scramble...", flush=True)
        trace = recover_cached_pose_symmetries(
            cache,
            trace,
            progress=not args.display,
            refine_pose=True,
            recover_orientation=True,
            known_initial_colors=known_initial_colors,
            source_fps=source_fps,
        )
    if args.recognize_moves:
        if args.turning_start_frame is not None:
            print(
                f"Holding cube state static through frame {args.turning_start_frame} (inclusive).",
                flush=True,
            )
        recover_cached_pose_symmetries(
            cache,
            trace,
            progress=not args.display,
            pose_prepared=True,
            recover_orientation=False,
            recognize_moves=True,
            source_fps=source_fps,
            known_initial_colors=known_initial_colors,
            turning_start_frame=args.turning_start_frame,
            record_all_turn_candidates=args.turn_candidate_log is not None,
            turn_candidate_log=args.turn_candidate_log,
        )


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
    if args.pnp_focal_scale <= 0.0:
        raise SystemExit("--pnp-focal-scale must be positive")
    if args.orientation_axis_length <= 0.0:
        raise SystemExit("--orientation-axis-length must be positive")
    if not args.model.is_file():
        raise SystemExit(f"Could not find independent pose model: {args.model}")
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
    if not set(CLASS_NAMES).issubset(set(pose_model.names.values())):
        raise SystemExit("Pose checkpoint must predict the three landmark classes and six sticker colors")
    source = parse_source(args.source)
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
        render_frame(
            frame,
            None,
            None,
            RoiTracker(),
            args,
            turn_candidate_log=args.turn_candidate_log,
            turn_candidate_image_dir=args.turn_candidate_images,
            frame_index=0,
            neural_data=image_cache[0],
            candidate_only=True,
        )
        process_cached_poses(image_cache, args, known_initial_colors)
        if args.show_pnp or args.show_colored_masks:
            rendered, _, _ = render_frame(
                frame,
                None,
                None,
                RoiTracker(),
                args,
                frame_index=0,
                sticker_color_names=known_initial_colors,
                cached_final_only=True,
                neural_data=image_cache[0],
                move_accumulator=MoveAccumulator(),
            )
            if args.save:
                if not cv2.imwrite(str(args.save), rendered):
                    raise SystemExit(f"Could not write rendered image: {args.save}")
            if args.display:
                cv2.imshow(WINDOW_NAME, rendered)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        return 0

    cache, _, first_pass_completed, source_fps = run_neural_video_pass(
        source,
        bbox_detector,
        pose_model,
        args,
        output_path=args.save_video,
        turn_candidate_log=args.turn_candidate_log,
        turn_candidate_image_dir=args.turn_candidate_images,
    )
    process_cached_poses(cache, args, known_initial_colors, source_fps=source_fps)
    if first_pass_completed and (args.display or args.save_final_video is not None):
        run_cached_video_pass(
            cache,
            args,
            sticker_color_names=known_initial_colors,
            output_path=args.save_final_video,
            fps=source_fps,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
