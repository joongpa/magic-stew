#!/usr/bin/env python3
"""Render the independent cube-keypoint YOLO model with the live ROI flow.

Per frame, this renderer:

1. Runs the cube bbox detector when no tracked ROI exists.
2. Builds the padded square ROI using the shared live-pipeline helper.
3. Runs the combined cube/sticker segmentation and independent three-class pose models inside that ROI.
4. Maps the independent points and sticker masks back to the full frame.
5. Updates the ROI tracker from the predicted point extent.

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
import sys
from collections import deque
from dataclasses import dataclass, replace
from itertools import combinations, permutations, product
from pathlib import Path
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
from live_recognition.pose_motion import (  # noqa: E402
    PoseMotionFilter,
    PoseMotionFilterConfig,
)
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
WINDOW_NAME = "independent cube keypoints"
DEFAULT_BBOX_MODEL = ROOT / "runs/500ExamplesWeights.pt"
DEFAULT_POSE_MODEL = (
    ROOT
    / "runs/pose/runs/independent_keypoints_v3_weights.pt"
)
DEFAULT_COMBINED_SEGMENTATION_MODEL_CANDIDATES = (
    ROOT / "runs/segment/runs/combinedSegmentWeights.pt",
    ROOT / "runs/segment/runs/cube_sticker_combined_seg/runs/cube_sticker_yolo11n_seg/weights/best.pt",
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
# A class is correct about 90% of the time.  Make a mismatch expensive enough
# to prevent a one-cubie row/column slip, while keeping it finite so an
# occasional misclassified point can still use a much better geometric match.
PNP_CLASS_MISMATCH_PENALTY = 48.0
# Full-cube assignment mode. ``closest`` ignores class labels, while
# ``class-aware`` adds a finite class-mismatch cost. The separate assignment
# constraint can use either naïve independent nearest-neighbor matching or a
# globally one-to-one Hungarian assignment.
PNP_ASSIGNMENT_MODE = "closest"
PNP_PIVOT_CONSISTENCY_WEIGHT = 1.5
PNP_PIVOT_HUBER_DELTA = 4.0
PNP_PIVOT_P90_WEIGHT = 0.40
PNP_PIVOT_P95_WEIGHT = 0.20
PNP_PIVOT_MAX_WEIGHT = 0.05
PNP_ASSIGNMENT_TAIL_WEIGHT = 0.25
# Final-solve observations must be close relative to the displayed cube size.
# This is deliberately expressed as a fraction of the projected cube diameter
# so the same setting works at different distances from the camera.
PNP_FINAL_ASSIGNMENT_ERROR_FRACTION = 0.05
PNP_LAYER_MISALIGNMENT_MIN_DEGREES = -45.0
PNP_LAYER_MISALIGNMENT_MAX_DEGREES = 45.0
PNP_LAYER_MISALIGNMENT_STEP_DEGREES = 3.0
# A moved layer is likely to have the fewest low-error assignments under the
# initial rigid-cube pose. Only these face directions need the expensive angle
# sweep; callers can set this to 6 for exhaustive diagnostics.
PNP_TURN_FACE_CANDIDATES = 2
PNP_TURN_DEBUG_CANDIDATE_COUNT = 6
PNP_TURN_OVERLAY_CANDIDATE_COUNT = 6


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
    frames_seen: int = 0

    @classmethod
    def create(cls) -> "StickerColorTrace":
        identities = canonical_sticker_identities()
        return cls(
            identities=identities,
            observations={identity.label: [] for identity in identities},
        )

    def observe(
        self,
        frame_index: int,
        observations: dict[str, np.ndarray],
    ) -> None:
        self.frames_seen += 1
        for identity in self.identities:
            rgb = observations.get(identity.label)
            if rgb is None:
                continue
            value = np.asarray(rgb, dtype=np.uint8).reshape(3).copy()
            self.observations[identity.label].append((int(frame_index), value))


@dataclass(frozen=True)
class StickerColorProfile:
    """A fitted black-to-color-to-white RGB trajectory in OKLab."""

    polyline_oklab: np.ndarray
    residual: float
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
    image_size: tuple[int, int] | None = None
    layer_rotations: tuple["CubeLayerRotation", ...] = ()
    turn_candidate_scores: tuple[tuple[str, float, int, float, float], ...] = ()
    turn_candidate_debug: tuple["TurnCandidateDebug", ...] = ()
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
        ),
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


def fit_sticker_color_profile(
    trace: StickerColorTrace,
    sticker_label: str,
) -> StickerColorProfile | None:
    """Fit a piecewise OKLab profile to one sticker's RGB observations."""

    entries = trace.observations.get(sticker_label, ())
    if not entries:
        return None
    points = rgb_to_oklab(
        np.asarray([value for _frame, value in entries], dtype=np.float64)
    ).astype(np.float32)
    black = np.zeros(3, dtype=np.float32)
    white = np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
    if len(points) == 1:
        polyline = np.asarray((black, points[0], white), dtype=np.float32)
        residual = float(_piecewise_profile_distances(points, polyline).mean())
        return StickerColorProfile(polyline, residual, len(points))

    ordered = points[np.argsort(points[:, 0])[::-1]]
    candidate_indexes = np.unique(
        np.linspace(0, len(ordered) - 1, min(24, len(ordered)), dtype=np.int32)
    )
    candidates = ordered[candidate_indexes]
    best_pair = np.asarray((candidates[0], candidates[-1]), dtype=np.float32)
    best_score = float("inf")
    for first_index in range(len(candidates) - 1):
        for second_index in range(first_index + 1, len(candidates)):
            high_pivot = candidates[first_index]
            low_pivot = candidates[second_index]
            polyline = np.asarray(
                (black, low_pivot, high_pivot, white),
                dtype=np.float32,
            )
            score = float(
                np.mean(_piecewise_profile_distances(points, polyline) ** 2)
            )
            if score < best_score:
                best_score = score
                best_pair = np.asarray((high_pivot, low_pivot), dtype=np.float32)
    polyline = np.asarray(
        (black, best_pair[1], best_pair[0], white),
        dtype=np.float32,
    )
    residual = float(np.sqrt(np.mean(_piecewise_profile_distances(points, polyline) ** 2)))
    return StickerColorProfile(polyline, residual, len(points))


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


def classify_center_color_names(trace: StickerColorTrace) -> dict[str, str]:
    """Assign observed centers to the lowest-cost legal standard color scheme."""

    center_observations: dict[str, np.ndarray] = {}
    center_identities = (
        identity for identity in trace.identities if identity.category == "center"
    )
    for identity in center_identities:
        entries = trace.observations.get(identity.label, ())
        if not entries:
            continue
        values = np.asarray([value for _frame, value in entries], dtype=np.float64)
        center_observations[identity.face] = np.median(values, axis=0)
    if not center_observations:
        return {}

    observed_faces = tuple(sorted(center_observations))
    observed_oklab = rgb_to_oklab(
        np.asarray([center_observations[face] for face in observed_faces])
    )
    target_names = tuple(PURE_COLOR_RGB)
    target_oklab = rgb_to_oklab(
        np.asarray([PURE_COLOR_RGB[name] for name in target_names])
    )
    best_assignment = None
    best_cost = float("inf")
    for assignment in legal_center_color_assignments():
        target_indices = [target_names.index(assignment[face]) for face in observed_faces]
        cost = float(
            np.sum(
                np.linalg.norm(
                    observed_oklab - target_oklab[target_indices],
                    axis=1,
                )
                ** 2
            )
        )
        if cost < best_cost:
            best_cost = cost
            best_assignment = assignment
    if best_assignment is None:
        return {}
    return {
        identity.label: best_assignment[identity.face]
        for identity in trace.identities
        if identity.category == "center" and identity.face in center_observations
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
    try:
        source_color_frame = np.column_stack(
            [FACE_VECTOR_BY_NAME[color_face_by_name[color_name]] for color_name in ordered_colors]
        )
    except KeyError:
        return False
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
    """Classify observed edge/corner stickers with legal cubie constraints."""

    center_color_names = (
        classify_center_color_names(trace)
        if center_color_names is None
        else dict(center_color_names)
    )
    if not center_color_names:
        return {}
    center_profiles: dict[str, StickerColorProfile] = {}
    for identity in trace.identities:
        if identity.category != "center":
            continue
        color_name = center_color_names.get(identity.label)
        if color_name is None:
            continue
        profile = fit_sticker_color_profile(trace, identity.label)
        if profile is not None:
            center_profiles[color_name] = profile
    if not center_profiles:
        return dict(center_color_names)

    if len(center_profiles) != len(PURE_COLOR_RGB):
        # Without all six center references there is no reliable way to score
        # a complete physical piece inventory.
        return dict(center_color_names)
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
        groups = list(groups_by_category[category].values())
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
                            option_cost += _sticker_profile_confidence(profile) * sticker_color_profile_distance(
                                profile,
                                center_profiles[color],
                            )
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
    class_mismatch_penalty: float = PNP_CLASS_MISMATCH_PENALTY,
    score_upper_bound: float = float("inf"),
) -> tuple[float, np.ndarray, np.ndarray]:
    """Assign every observation independently to its cheapest cube point.

    Unlike the Hungarian assignment, several observations may select the same
    projected point.  This is useful as a permissive hypothesis score while
    searching pose space; repeated 3-D targets are deduplicated before any
    solvePnP refinement.
    """

    del image_size, outlier_distance
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
        costs = costs + mismatch * float(class_mismatch_penalty)

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
    class_score = 0.0
    if observed_class_ids is not None and projected_class_ids is not None:
        observed_classes = np.asarray(observed_class_ids, dtype=np.int32).reshape(-1)
        projected_classes = np.asarray(projected_class_ids, dtype=np.int32).reshape(-1)
        known = (
            (observed_classes[rows] >= 0)
            & (projected_classes[columns] >= 0)
        )
        if np.any(known):
            class_score = float(
                np.mean(
                    observed_classes[rows][known]
                    != projected_classes[columns][known]
                )
                * PNP_CLASS_MISMATCH_PENALTY
            )
    return geometric_score + PNP_ASSIGNMENT_TAIL_WEIGHT * tail_score + class_score


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
            consistency_error += float(
                np.mean(
                    assigned_observed_classes[known]
                    != assigned_projected_classes[known]
                )
                * PNP_CLASS_MISMATCH_PENALTY
            )
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
    candidates: list[tuple[StickerPnPCandidate, float]] = []
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
            elif sticker_contour is not None:
                ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                    object_points,
                    observed_points,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_SQPNP,
                )
            else:
                ok, rvec, tvec = cv2.solvePnP(
                    object_points,
                    observed_points,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_SQPNP,
                )
                rvecs, tvecs = (rvec,), (tvec,)
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
                projected_full_points, _ = cv2.projectPoints(
                    canonical_object_points,
                    rvec,
                    tvec,
                    camera_matrix,
                    dist_coeffs,
                )
            except cv2.error:
                continue
            projected_points = projected_points.reshape(-1, 2)
            projected_full_points = projected_full_points.reshape(-1, 2)
            reprojection_error = float(
                np.mean(np.linalg.norm(projected_points - observed_points, axis=1))
            )
            mask_score = (
                sticker_mask_iou(projected_full_points, sticker_contour, image_size)
                if sticker_contour is not None
                else 0.0
            )
            candidates.append(
                (
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
                        image_size=image_size,
                    ),
                    mask_score,
                )
            )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            item[0].overall_reprojection_error,
            -item[1],
            item[0].reprojection_error,
        ),
    )[0]


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


def recover_misaligned_cube_layers(
    candidate: StickerPnPCandidate,
    assignment_error_limit: float,
    max_face_candidates: int = PNP_TURN_FACE_CANDIDATES,
    record_all_turn_candidates: bool = False,
) -> StickerPnPCandidate:
    """Search layer turns using a complete, robust assignment score.

    The old search scored only the currently unexplained observations.  That
    made a bad layer win when it happened to explain one stray point, even if
    it displaced a large number of already-good observations.  Every trial
    below instead reassigns all non-anchor observations to the whole cube and
    is ranked by inlier coverage first, then by robust error over the complete
    assignment.
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
    if not np.any(finite & ~accepted):
        return candidate

    base_object_points = np.asarray(
        CUBE_LATTICE_OBJECT_POINTS_MM,
        dtype=np.float32,
    )
    if len(base_object_points) == 0:
        return candidate

    observed_classes = candidate.assignment_observed_class_ids
    if observed_classes is not None:
        observed_classes = np.asarray(observed_classes, dtype=np.int32).reshape(-1)
        if len(observed_classes) != len(image_points):
            observed_classes = None
    use_class_aware_assignment = PNP_ASSIGNMENT_MODE == "class-aware"
    # The selected sticker is the pivot for this local search.  New layer
    # hypotheses may not replace those correspondences.  Older candidates
    # created before assignment_fixed_mask existed stored the anchor rows
    # first, so retain that layout as a compatibility fallback.
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

    def rank_assignment(
        assigned_distances: np.ndarray,
        assigned_classes: np.ndarray,
        trial_rotations: tuple[CubeLayerRotation, ...],
    ) -> tuple[tuple[float, float, float, float], float, int]:
        """Rank coverage first and robust error second."""

        finite_distances = np.isfinite(assigned_distances)
        if not np.any(finite_distances):
            return (float("inf"),) * 4, float("inf"), 0
        valid_distances = assigned_distances[finite_distances]
        inlier_count = int(
            np.count_nonzero(valid_distances <= assignment_error_limit)
        )
        robust_error = float(
            np.mean(huber_loss(valid_distances))
            + PNP_ASSIGNMENT_TAIL_WEIGHT * np.percentile(valid_distances, 90)
            + 0.10 * np.percentile(valid_distances, 95)
        )
        if (
            use_class_aware_assignment
            and observed_classes is not None
            and len(assigned_classes) == len(observed_classes)
        ):
            known = (observed_classes >= 0) & (assigned_classes >= 0)
            if np.any(known):
                robust_error += float(
                    np.mean(observed_classes[known] != assigned_classes[known])
                    * PNP_CLASS_MISMATCH_PENALTY
                )
        mean_error = float(np.mean(valid_distances))
        p90_error = float(np.percentile(valid_distances, 90))
        rank = (-float(inlier_count), robust_error, p90_error, mean_error)
        return rank, robust_error, inlier_count

    best = None
    base_rotations = tuple(candidate.layer_rotations)
    shortlisted_faces = rank_misaligned_layer_faces(
        base_object_points,
        object_points,
        distances,
        fixed_mask,
        assignment_error_limit,
        max_face_candidates,
    )
    # Preserve the old exhaustive behavior when explicitly requested, and use
    # all faces if the cheap pre-filter cannot map enough assignments back to
    # the canonical lattice.
    if not shortlisted_faces or max_face_candidates >= 6:
        shortlisted_faces = tuple(
            (axis, side, 0.0)
            for axis in range(3)
            for side in (-1, 1)
        )

    angles = np.arange(
        PNP_LAYER_MISALIGNMENT_MIN_DEGREES,
        PNP_LAYER_MISALIGNMENT_MAX_DEGREES
        + 0.5 * PNP_LAYER_MISALIGNMENT_STEP_DEGREES,
        PNP_LAYER_MISALIGNMENT_STEP_DEGREES,
        dtype=np.float64,
    )

    hypotheses: list[tuple[CubeLayerRotation, ...]] = [base_rotations]
    for axis, side, _score in shortlisted_faces:
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

    ranked_trials = []
    for trial_rotations in hypotheses:
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
            continue
        if len(trial_object_points) == 0:
            continue
        try:
            trial_projected_points, _ = cv2.projectPoints(
                trial_object_points,
                candidate.rvec,
                candidate.tvec,
                candidate.camera_matrix,
                candidate.dist_coeffs,
            )
        except cv2.error:
            continue
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
        rank, robust_error, inlier_count = rank_assignment(
            trial_distances,
            trial_assignment_classes,
            trial_rotations,
        )
        ranked_trials.append(
            {
                "is_baseline": trial_rotations == base_rotations,
                "rank": rank,
                "label": layer_rotation_debug_label(trial_rotations),
                "robust_error": robust_error,
                "inlier_count": inlier_count,
                "p90_error": float(rank[2]),
                "mean_error": float(rank[3]),
                "layer_rotations": trial_rotations,
                "assignment_projected_points": trial_assignment_projected.copy(),
                "assignment_object_points": trial_assignment_objects.copy(),
                "assignment_projected_class_ids": trial_assignment_classes.copy(),
            }
        )
        if best is None or rank < best[0]:
            best = (
                rank,
                robust_error,
                inlier_count,
                trial_rotations,
                trial_assignment_objects,
                trial_assignment_projected,
                trial_assignment_classes,
            )

    ordered_trials = sorted(ranked_trials, key=lambda trial: trial["rank"])
    baseline_trial = next(
        (trial for trial in ordered_trials if trial["is_baseline"]),
        None,
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
    return replace(
        candidate,
        assignment_object_points=best_object_points,
        assignment_projected_points=best_projected_points,
        assignment_projected_class_ids=best_projected_classes,
        overall_reprojection_error=robust_error,
        layer_rotations=best_rotations,
        turn_candidate_scores=turn_candidate_scores,
        turn_candidate_debug=turn_candidate_debug,
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
        return candidate
    return replace(
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


def best_sticker_pnp_candidates(
    frame_points: np.ndarray,
    point_class_ids: np.ndarray,
    matches: tuple[StickerKeypointMatch, ...],
    image_size: tuple[int, int],
    focal_scale: float,
    local_masks: list[np.ndarray] | None = None,
    roi_xyxy: tuple[int, int, int, int] | None = None,
    scoring_points: np.ndarray | None = None,
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
                local_masks is not None
                and roi_xyxy is not None
                and 0 <= match.mask_index < len(local_masks)
            ):
                sticker_contour = sticker_contour_to_frame(
                    local_masks[match.mask_index],
                    roi_xyxy,
                )
            candidate = solve_sticker_pnp(
                frame_points,
                point_class_ids,
                match,
                image_size,
                focal_scale,
                sticker_contour,
                scoring_points,
            )
            if candidate is not None:
                candidates.append(candidate)
        candidates.sort(
            key=lambda candidate: (
                candidate.overall_reprojection_error,
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
    """Return the candidate's direct geometric quality score."""

    overall = float(candidate.overall_reprojection_error)
    sticker = float(candidate.reprojection_error)
    if not np.isfinite(overall):
        overall = 1e6
    if not np.isfinite(sticker):
        sticker = 1e6
    return overall + 0.25 * sticker


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

    for mask_index, (local_mask, class_id) in enumerate(zip(local_masks, class_ids, strict=True)):
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
        color = (
            STICKER_TYPE_COLORS_BGR[sticker_type]
            if pose_predictions is not None
            else sticker_color(names, int(class_id))
        )
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
                color if pose_predictions is not None else STICKER_BORDER_COLOR_BGR,
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
                    color,
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
    """Project visible canonical stickers, retaining their physical identities."""

    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    identities = {
        (identity.cubie_index, identity.axis, identity.side): identity
        for identity in canonical_sticker_identities()
    }
    projected: list[tuple[StickerIdentity, float, np.ndarray]] = []
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
                    polygon, _ = cv2.projectPoints(
                        transformed_face,
                        rvec,
                        tvec,
                        candidate.camera_matrix,
                        candidate.dist_coeffs,
                    )
                    polygon = polygon.reshape(-1, 2)
                    if np.isfinite(polygon).all():
                        projected.append((identity, float(center_camera[2]), polygon))
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

    local_masks, _names, _class_ids, cube_mask = predicted
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
    profiles = {
        identity.label: fit_sticker_color_profile(trace, identity.label)
        for identity in trace.identities
    }
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
            regression_oklab = sample_profile_polyline(profile.polyline_oklab, count=96)
            regression_rgb = oklab_to_rgb(regression_oklab).astype(np.float32)
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

    sticker_count = len(trace.identities)
    plot_trace_count = len(figure.data)

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
                {"visible": [True] * plot_trace_count},
                {"title": f"{title} (all {sticker_count} stickers)"},
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
        face_color = PNP_ORIENTATION_ANCHOR_COLOR_BGR if anchor else color
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
    candidate: StickerPnPCandidate,
    frame_index: int | None,
) -> None:
    """Append the tested layer-turn scores for one rendered frame as JSONL."""

    scores = getattr(candidate, "turn_candidate_scores", ())
    if not scores:
        return
    record = {
        "frame_index": frame_index,
        "selected": layer_rotation_debug_label(candidate.layer_rotations),
        "candidates": [
            {
                "label": label,
                "robust_score": float(robust_error),
                "inlier_count": int(inlier_count),
                "p90_error": float(p90_error),
                "mean_error": float(mean_error),
            }
            for label, robust_error, inlier_count, p90_error, mean_error in scores
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log_file:
        json.dump(record, log_file, separators=(",", ":"))
        log_file.write("\n")


def draw_projected_sticker_identity_labels(
    image: np.ndarray,
    candidate: StickerPnPCandidate,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> int:
    """Draw the short identity number at the center of each visible sticker."""

    try:
        projected_faces = projected_sticker_faces(candidate, rvec, tvec)
    except cv2.error:
        return 0
    height, width = image.shape[:2]
    drawn = 0
    for identity, _depth, polygon in projected_faces:
        center = np.mean(polygon, axis=0)
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
) -> int:
    """Fill classified, visible projected stickers with their colors."""

    if not sticker_color_names:
        return 0
    opacity = float(np.clip(opacity, 0.0, 1.0))
    image_offset_array = np.asarray(image_offset, dtype=np.float32).reshape(1, 2)
    drawn = 0
    try:
        projected_faces = projected_sticker_faces(candidate, rvec, tvec)
    except cv2.error:
        return 0
    for identity, _depth, polygon in projected_faces:
        color_name = sticker_color_names.get(identity.label)
        if color_name not in PURE_COLOR_RGB:
            continue
        polygon_int = np.round(polygon + image_offset_array).astype(np.int32)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, polygon_int, 1)
        color_bgr = tuple(int(value) for value in PURE_COLOR_RGB[color_name][::-1])
        color_layer = np.empty_like(image)
        color_layer[...] = color_bgr
        blended = cv2.addWeighted(image, 1.0 - opacity, color_layer, opacity, 0.0)
        image[mask > 0] = blended[mask > 0]
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


def draw_classified_cube_diagnostic_panel(
    image: np.ndarray,
    candidate: StickerPnPCandidate | None,
    sticker_color_names: dict[str, str],
    rvec: np.ndarray | None = None,
    orientation_axis_length: float = 24.0,
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
        "classified cube",
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
    draw_projected_sticker_colors(
        output,
        candidate,
        sticker_color_names,
        display_rvec,
        display_tvec,
        image_offset=image_offset,
    )
    draw_projected_cube_pose(
        output,
        candidate,
        display_rvec,
        display_tvec,
        PNP_CUBE_COLOR_BGR,
        2,
        outline=True,
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
            _label, robust_error, inlier_count, p90_error, mean_error = (
                candidate.turn_candidate_scores[rank]
            )
            cv2.putText(
                rendered,
                (
                    f"turn rank {rank} {debug_candidate.label} | "
                    f"in {inlier_count} | score {robust_error:.1f} | "
                    f"p90 {p90_error:.1f} | mean {mean_error:.1f}"
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
            PNP_CANDIDATE_CUBE_COLOR_BGR,
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
        PNP_CUBE_COLOR_BGR,
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
        label = (
            f"{pose_label} sticker {candidate.sticker_type} | "
            f"kp {candidate.sticker_average_confidence:.2f} | "
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
        for index, (label, robust_error, inlier_count, p90_error, mean_error) in enumerate(
            turn_candidate_scores[:PNP_TURN_OVERLAY_CANDIDATE_COUNT]
        ):
            selected_marker = "*" if label == selected_turn_label else " "
            turn_label = (
                f"turn {selected_marker}{label} | in {inlier_count} | "
                f"score {robust_error:.1f} | p90 {p90_error:.1f} | "
                f"mean {mean_error:.1f}"
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
    if turn_candidate_log is not None:
        append_turn_candidate_log(turn_candidate_log, candidate, frame_index)
    if turn_candidate_image_dir is not None:
        save_turn_candidate_images(
            image,
            candidate,
            turn_candidate_image_dir,
            frame_index,
        )
    return output


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
) -> tuple[np.ndarray, str, int]:
    height, width = frame.shape[:2]
    roi_xyxy, candidate, roi_source = shared_roi_for_frame(
        frame,
        bbox_detector,
        roi_tracker,
        args.roi_padding,
    )
    if roi_xyxy is None:
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
                orientation_axis_length=args.orientation_axis_length,
            )
        return output, "none", 0

    x1, y1, x2, y2 = roi_xyxy
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
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
                orientation_axis_length=args.orientation_axis_length,
            )
        return output, roi_source, 0

    model_roi = resize_to_square(roi, args.view_size)
    if args.parallel_cuda:
        sticker_masks, result = predict_models_parallel_cuda(
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
        if sticker_color_names is not None:
            output = draw_classified_cube_diagnostic_panel(
                output,
                None,
                sticker_color_names,
                orientation_axis_length=args.orientation_axis_length,
            )
        return output, roi_source, 0

    local_points, point_confidences, class_ids, box_confidences = pose_predictions(result)
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
        if sticker_color_names is not None:
            output = draw_classified_cube_diagnostic_panel(
                output,
                None,
                sticker_color_names,
                orientation_axis_length=args.orientation_axis_length,
            )
        return output, roi_source, 0

    sticker_matches = (
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
    pnp_candidate = None
    if args.show_pnp:
        current_candidates = best_sticker_pnp_candidates(
            frame_points,
            class_ids,
            sticker_matches,
            (width, height),
            args.pnp_focal_scale,
            sticker_masks[0] if sticker_masks is not None else None,
            roi_xyxy,
            frame_points,
        )
        prepared_candidates = prepare_pose_hypotheses(
            current_candidates,
            symmetry_tracker,
        )
        recognize_moves = bool(getattr(args, "recognize_moves", False))
        pose_graph_selection = None
        if recognize_moves:
            if pose_hypothesis_buffer is not None:
                pose_hypothesis_buffer.append(prepared_candidates, frame_index)
                graph_frames = pose_hypothesis_buffer.best_candidate_frames()
            else:
                graph_frames = (
                    PoseHypothesisFrame(frame_index, prepared_candidates),
                )
            predicted_pose = (
                pose_motion_filter.predict_pose()
                if pose_motion_filter is not None
                else None
            )
            pose_graph_selection = (
                select_pose_graph_path(
                    graph_frames,
                    pose_motion_filter.config
                    if pose_motion_filter is not None
                    else PoseMotionFilterConfig(),
                    predicted_pose,
                )
                if prepared_candidates
                else None
            )
            if (
                pose_graph_selection is not None
                and pose_graph_selection.violates_limits
                and pose_hypothesis_buffer is not None
            ):
                # The common path has already used one candidate per frame.
                # Only expand to retained alternatives after the best path
                # breaks the motion envelope.
                reconsidered_selection = select_pose_graph_path(
                    pose_hypothesis_buffer.frames,
                    pose_motion_filter.config
                    if pose_motion_filter is not None
                    else PoseMotionFilterConfig(),
                    predicted_pose,
                )
                if reconsidered_selection is not None:
                    pose_graph_selection = reconsidered_selection
        pnp_candidate = (
            pose_graph_selection.candidate
            if pose_graph_selection is not None
            else prepared_candidates[0] if prepared_candidates else None
        )
        if pnp_candidate is not None:
            # Static mode always performs the sticker-selected full-cube
            # refinement.  Move mode retains its existing motion-triggered
            # refinement and layer-turn recovery behavior.
            should_refine = (
                not recognize_moves
                or (
                    pose_graph_selection is not None
                    and pose_graph_selection.violates_limits
                )
            )
            if should_refine:
                pnp_candidate = solve_pnp_from_low_error_assignments(
                    pnp_candidate,
                    recover_layer_turns=recognize_moves,
                    max_turn_face_candidates=args.pnp_turn_face_candidates,
                    record_all_turn_candidates=(
                        recognize_moves and turn_candidate_log is not None
                    ),
                )
            if symmetry_tracker is not None:
                # Apply the continuity choice after refinement too.  This is
                # normally a no-op, but protects the displayed axes if a
                # refinement crosses a Rodrigues branch.
                pnp_candidate = canonicalize_pose_symmetry(
                    pnp_candidate,
                    symmetry_tracker,
                )
            pnp_candidate = apply_temporal_pose_filter(
                pnp_candidate,
                pose_motion_filter,
            )
            if symmetry_tracker is not None:
                symmetry_tracker.observe(pnp_candidate.rvec)
        if (
            sticker_color_trace is not None
            and pnp_candidate is not None
            and (frame_index is None or frame_index < args.color_plot_frames)
        ):
            sticker_color_trace.observe(
                0 if frame_index is None else frame_index,
                sample_projected_sticker_colors(
                    frame,
                    roi_xyxy,
                    sticker_masks,
                    pnp_candidate,
                    max_samples_per_sticker=args.color_samples_per_sticker,
                    min_samples_per_sticker=args.color_min_samples_per_sticker,
                    sticker_mask_erosion=args.sticker_mask_erosion,
                ),
            )
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
                orientation_axis_length=args.orientation_axis_length,
            )
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
            if (
                sticker_color_trace is not None
                and frame_index >= args.color_plot_frames
                and not args.display
            ):
                break
    finally:
        if writer is not None:
            writer.release()
        capture.release()
    if output_path is not None:
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
        help="Draw the cube pose from sticker initialization and final refinement.",
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
        default=False,
        help="Draw stable C/E/K sticker identity numbers on visible stickers.",
    )
    parser.add_argument(
        "--recognize-moves",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable motion-graph and layer-turn recovery. By default the "
            "renderer stays on static sticker selection plus full-cube refinement."
        ),
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
        help="Full-cube assignment strategy: geometric closest or class-aware.",
    )
    parser.add_argument(
        "--pnp-turn-face-candidates",
        type=int,
        default=PNP_TURN_FACE_CANDIDATES,
        help=(
            "Number of face directions retained by the cheap turn pre-filter "
            "before the angle sweep (1-6; 6 is exhaustive, default: "
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
        "--color-plot-frames",
        type=int,
        default=90,
        help="Maximum video frames to collect for --rgb-html-plot (default: 90).",
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
    parser.add_argument("--save-video", type=Path, help="Output path for rendered video.")
    parser.add_argument(
        "--show-classified-colors",
        "--show-classified-stickers",
        "--show-classified-centers",
        dest="show_classified_colors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run a second pass and color visible projected stickers using the "
            "legal standard cube color scheme."
        ),
    )
    parser.add_argument(
        "--save-colored-video",
        type=Path,
        help="Optional output path for the second, sticker-colored video pass.",
    )
    parser.add_argument("--once", action="store_true", help="Process one video/camera frame and exit.")
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.set_defaults(display=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    global PNP_ASSIGNMENT_MODE
    PNP_ASSIGNMENT_MODE = args.pnp_assignment_mode
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
    if args.color_plot_frames < 1:
        raise SystemExit("--color-plot-frames must be positive")
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
    roi_tracker = RoiTracker()
    symmetry_tracker = CubeSymmetryTracker()
    pose_hypothesis_buffer = PoseHypothesisBuffer(max_frames=8)
    sticker_color_trace = (
        StickerColorTrace.create()
        if args.rgb_html_plot is not None or args.show_classified_colors
        else None
    )
    pose_motion_filter = None
    if args.pose_motion_filter:
        pose_motion_filter = PoseMotionFilter(
            PoseMotionFilterConfig(
                dt_seconds=args.pose_motion_dt,
                max_acceleration_units_s2=args.pose_motion_max_acceleration,
                max_jerk_units_s3=args.pose_motion_max_jerk,
                max_angular_acceleration_rad_s2=args.pose_motion_max_angular_acceleration,
                max_angular_jerk_rad_s3=args.pose_motion_max_angular_jerk,
            )
        )

    if is_image_source(source):
        frame = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if frame is None:
            raise SystemExit(f"Could not read image: {source}")
        rendered, roi_source, point_count = render_frame(
            frame,
            bbox_detector,
            pose_model,
            roi_tracker,
            args,
            sticker_seg_model,
            args.turn_candidate_log,
            args.turn_candidate_images,
            0,
            pose_motion_filter,
            symmetry_tracker,
            pose_hypothesis_buffer,
            sticker_color_trace,
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
        center_color_names = (
            classify_center_color_names(sticker_color_trace)
            if args.show_classified_colors and sticker_color_trace is not None
            else {}
        )
        sticker_color_names = classify_sticker_color_names(
            sticker_color_trace,
            center_color_names,
        ) if sticker_color_trace is not None else {}
        if center_color_names:
            print(
                "Classified centers: "
                + ", ".join(
                    f"{label}={color}"
                    for label, color in sorted(center_color_names.items())
                ),
                flush=True,
            )
        if args.show_classified_colors and (args.display or sticker_color_names):
            colored_roi_tracker = RoiTracker()
            colored_symmetry_tracker = CubeSymmetryTracker()
            colored_pose_hypothesis_buffer = PoseHypothesisBuffer(max_frames=8)
            colored_pose_motion_filter = make_pose_motion_filter(args)
            colored_rendered, _colored_roi_source, _colored_point_count = render_frame(
                frame,
                bbox_detector,
                pose_model,
                colored_roi_tracker,
                args,
                sticker_seg_model,
                None,
                None,
                0,
                colored_pose_motion_filter,
                colored_symmetry_tracker,
                colored_pose_hypothesis_buffer,
                None,
                sticker_color_names,
            )
            if args.display:
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

    frame_count, first_pass_completed = run_video_pass(
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
    center_color_names = (
        classify_center_color_names(sticker_color_trace)
        if args.show_classified_colors and sticker_color_trace is not None
        else {}
    )
    sticker_color_names = classify_sticker_color_names(
        sticker_color_trace,
        center_color_names,
    ) if sticker_color_trace is not None else {}
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
        args.show_classified_colors
        and first_pass_completed
        and (args.display or args.save_colored_video is not None)
    ):
        run_video_pass(
            source,
            bbox_detector,
            pose_model,
            sticker_seg_model,
            args,
        sticker_color_names=sticker_color_names,
            output_path=args.save_colored_video,
        )
    if sticker_color_trace is not None and args.rgb_html_plot is not None:
        save_sticker_color_plots(
            sticker_color_trace,
            args.rgb_html_plot,
            title=(
                "Independent-keypoint sticker RGB observations "
                f"across {min(frame_count, args.color_plot_frames)} frames"
            ),
            primary_normalized=args.normalize_rgb_plot,
            point_size=args.plotly_point_size,
        )
        print(f"Saved sticker color plots: {args.rgb_html_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
