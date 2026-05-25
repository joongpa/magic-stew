from __future__ import annotations

from itertools import product

from live_recognition.types import Axis, Face, OrientationHypothesis

_AXIS_VECTOR: dict[Axis, tuple[int, int, int]] = {
    "+x": (1, 0, 0),
    "-x": (-1, 0, 0),
    "+y": (0, 1, 0),
    "-y": (0, -1, 0),
    "+z": (0, 0, 1),
    "-z": (0, 0, -1),
}

_VECTOR_AXIS = {value: key for key, value in _AXIS_VECTOR.items()}

_BASE_FACE_VECTOR: dict[Face, tuple[int, int, int]] = {
    "R": (1, 0, 0),
    "L": (-1, 0, 0),
    "U": (0, 1, 0),
    "D": (0, -1, 0),
    "F": (0, 0, 1),
    "B": (0, 0, -1),
}


def _cross(a: tuple[int, int, int], b: tuple[int, int, int]) -> tuple[int, int, int]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def all_cube_orientations() -> list[dict[Face, Axis]]:
    """Return the 24 proper rotations of a cube as face-to-camera-axis mappings."""

    orientations: list[dict[Face, Axis]] = []
    axes = list(_AXIS_VECTOR.values())

    for x_axis, y_axis in product(axes, axes):
        if sum(a * b for a, b in zip(x_axis, y_axis, strict=True)) != 0:
            continue

        z_axis = _cross(x_axis, y_axis)
        rotation = {
            (1, 0, 0): x_axis,
            (-1, 0, 0): tuple(-v for v in x_axis),
            (0, 1, 0): y_axis,
            (0, -1, 0): tuple(-v for v in y_axis),
            (0, 0, 1): z_axis,
            (0, 0, -1): tuple(-v for v in z_axis),
        }
        orientations.append(
            {
                face: _VECTOR_AXIS[rotation[vector]]
                for face, vector in _BASE_FACE_VECTOR.items()
            }
        )

    return orientations


def orientation_hypotheses(
    cube_confidence: float,
    *,
    max_hypotheses: int = 8,
) -> list[OrientationHypothesis]:
    """Create a calibrated placeholder distribution over cube orientations.

    Until a learned keypoint/face model is available, this keeps the pipeline probabilistic:
    stronger cube detections concentrate more mass in a few stable hypotheses, while weaker
    detections flatten the distribution.
    """

    orientations = all_cube_orientations()
    max_hypotheses = max(1, min(max_hypotheses, len(orientations)))
    sharpness = min(max(cube_confidence, 0.0), 1.0)

    weights = []
    for index in range(max_hypotheses):
        rank_weight = 1.0 / (index + 1)
        weights.append((1.0 - sharpness) + sharpness * rank_weight)

    total = sum(weights)
    return [
        OrientationHypothesis(face_to_axis=orientations[index], probability=weight / total)
        for index, weight in enumerate(weights)
    ]

