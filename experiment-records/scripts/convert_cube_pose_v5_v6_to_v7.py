"""Convert compact cube-pose v5/v6 labels into direct custom-head v7 labels.

The source directory is read only.  Every converted JSON and the v7 schema are
written below a separate output directory.

Example:

    .venv/bin/python scripts/convert_cube_pose_v5_v6_to_v7.py \
        data/mcp_dataset_v5/pose data/mcp_dataset_v7/pose

v7 stores the quantities the custom head trains on directly:

* ``cube_pose.u_colatitude_degrees`` in ``[0, 90)`` for the selected canonical
  stack axis,
* ``cube_pose.v_azimuth_degrees`` in ``[-180, 180)``,
* ``cube_pose.twist_degrees`` in ``[-45, 45)``,
* normalized image center and scale,
* perspective strength, and
* the two outer-layer angles in ``[-45, 45)``.

No quaternion is written to v7.  The v5/v6 quaternion is used only during
conversion and can be reconstructed from the v7 head values when loading.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import permutations, product
from pathlib import Path
from typing import Any

import numpy as np


IMAGE_ZENITH = np.asarray((0.0, -1.0, 0.0), dtype=np.float64)
IMAGE_RIGHT = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
FALLBACK_RIGHT = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
ANGLE_PERIOD_DEGREES = 90.0
FACE_AXES = {
    "R": np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
    "L": np.asarray((-1.0, 0.0, 0.0), dtype=np.float64),
    "U": np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
    "D": np.asarray((0.0, 0.0, -1.0), dtype=np.float64),
    "F": np.asarray((0.0, -1.0, 0.0), dtype=np.float64),
    "B": np.asarray((0.0, 1.0, 0.0), dtype=np.float64),
}


def cube_symmetry_matrices() -> tuple[np.ndarray, ...]:
    matrices = []
    for permutation in permutations(range(3)):
        for signs in product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float64)
            for row, column in enumerate(permutation):
                matrix[row, column] = signs[row]
            if np.linalg.det(matrix) > 0.5:
                matrices.append(matrix)
    return tuple(matrices)


SYMMETRIES = cube_symmetry_matrices()


def wrap_degrees(value: float, period: float) -> float:
    return (float(value) + period / 2.0) % period - period / 2.0


def quaternion_matrix(values: Any) -> np.ndarray:
    quaternion = np.asarray(values, dtype=np.float64).reshape(4)
    quaternion /= max(float(np.linalg.norm(quaternion)), 1e-12)
    x, y, z, w = quaternion
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _projected_image_right(stack: np.ndarray) -> np.ndarray:
    reference = FALLBACK_RIGHT if abs(float(IMAGE_RIGHT @ stack)) > 0.95 else IMAGE_RIGHT
    projected = reference - float(reference @ stack) * stack
    return projected / max(float(np.linalg.norm(projected)), 1e-12)


def head_parameters(rotation: np.ndarray) -> tuple[float, float, float]:
    stack = rotation[:, 2]
    colatitude = math.degrees(math.acos(float(np.clip(IMAGE_ZENITH @ stack, -1.0, 1.0))))
    azimuth = math.degrees(math.atan2(float(stack[0]), float(stack[2])))
    local_x = _projected_image_right(stack)
    local_y = np.cross(stack, local_x)
    rotation_x = rotation[:, 0]
    twist = math.degrees(math.atan2(float(rotation_x @ local_y), float(rotation_x @ local_x)))
    return (
        colatitude,
        wrap_degrees(azimuth, 360.0),
        wrap_degrees(twist, ANGLE_PERIOD_DEGREES),
    )


def rotation_angle(rotation: np.ndarray) -> float:
    return math.acos(float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))


def canonicalize_no_turn(rotation: np.ndarray) -> np.ndarray:
    """Choose the cube axis closest to image zenith for a no-turn pose."""
    candidates = []
    for symmetry in SYMMETRIES:
        candidate = rotation @ symmetry
        u, _v, twist = head_parameters(candidate)
        tie_break = tuple(np.round(candidate.reshape(-1), 9))
        candidates.append((u, abs(twist), tie_break, candidate))
    return min(candidates, key=lambda item: item[:3])[3]


def canonicalize_turn(rotation: np.ndarray, face: str) -> tuple[np.ndarray, int]:
    """Align a named v5 turning face with the v7 local stack axis."""
    source_axis = FACE_AXES[face]
    candidates = []
    for symmetry in SYMMETRIES:
        transformed_axis = symmetry.T @ source_axis
        if abs(abs(float(transformed_axis[2])) - 1.0) > 1e-6:
            continue
        candidate = rotation @ symmetry
        tie_break = tuple(np.round(candidate.reshape(-1), 9))
        candidate_stack = candidate[:, 2]
        colatitude = math.degrees(
            math.acos(float(np.clip(IMAGE_ZENITH @ candidate_stack, -1.0, 1.0)))
        )
        candidates.append((colatitude, abs(float(transformed_axis[2])), tie_break, symmetry, candidate, transformed_axis[2]))
    if not candidates:
        raise ValueError(f"Could not align turning face {face!r} with the canonical stack axis")
    _colatitude, _sign, _tie_break, _symmetry, candidate, stack_sign = min(
        candidates,
        key=lambda item: (item[0], item[1], item[2]),
    )
    return candidate, 2 if stack_sign > 0.0 else 0


def canonicalize_v6_turn(
    rotation: np.ndarray,
    angles: list[float],
) -> tuple[np.ndarray, list[float]]:
    """Choose the visible sign without changing an already adjusted turn."""
    if float(IMAGE_ZENITH @ rotation[:, 2]) >= 0.0:
        return rotation, list(angles)

    candidates = []
    for symmetry in SYMMETRIES:
        transformed_axis = symmetry[:, 2]
        if abs(abs(float(transformed_axis[2])) - 1.0) > 1e-6:
            continue
        if float(transformed_axis[2]) > 0.0:
            continue
        candidate = rotation @ symmetry
        tie_break = tuple(np.round(candidate.reshape(-1), 9))
        candidates.append(
            (rotation_angle(candidate), tie_break, candidate)
        )
    if not candidates:
        raise ValueError("Could not canonicalize the v6 turning stack axis")
    _angle, _tie_break, candidate = min(
        candidates,
        key=lambda item: (item[0], item[1]),
    )
    canonical_angles = list(angles)
    canonical_angles.reverse()
    return candidate, canonical_angles


def layer_angles_from_payload(payload: dict[str, Any]) -> list[float]:
    articulation = payload.get("articulation")
    if not isinstance(articulation, dict):
        raise ValueError("Pose label is missing articulation")
    layer_angles = articulation.get("layer_angles_degrees")
    if isinstance(layer_angles, list) and len(layer_angles) == 2:
        angles = [wrap_degrees(float(value), ANGLE_PERIOD_DEGREES) for value in layer_angles]
        return angles

    turn = articulation.get("turn")
    if turn is None:
        return [0.0, 0.0]
    if not isinstance(turn, dict):
        raise ValueError("articulation.turn must be an object or null")
    layer_index = int(turn.get("layer_index", -1))
    if layer_index not in (0, 2):
        raise ValueError(f"Expected an outer layer index, got {layer_index}")
    angle = wrap_degrees(float(turn.get("signed_angle_degrees", 0.0)), ANGLE_PERIOD_DEGREES)
    result = [0.0, 0.0]
    result[0 if layer_index == 0 else 1] = angle
    return result


def convert_payload(payload: dict[str, Any], source_name: str) -> dict[str, Any]:
    source_format = payload.get("format")
    if source_format not in {"cube_pose_v5", "cube_pose_v6"}:
        raise ValueError(f"{source_name}: expected cube_pose_v5 or cube_pose_v6, got {source_format!r}")
    rotation_payload = payload.get("cube_rotation_camera")
    projection = payload.get("image_projection")
    if not isinstance(rotation_payload, dict) or not isinstance(projection, dict):
        raise ValueError(f"{source_name}: missing cube_rotation_camera or image_projection")
    rotation = quaternion_matrix(rotation_payload.get("quaternion_xyzw"))
    angles = layer_angles_from_payload(payload)

    if source_format == "cube_pose_v5":
        articulation = payload["articulation"]
        turn = articulation.get("turn") if isinstance(articulation, dict) else None
        if isinstance(turn, dict):
            face = str(turn.get("face", ""))
            if face in FACE_AXES:
                rotation, canonical_layer_index = canonicalize_turn(rotation, face)
                source_angle = next(
                    (value for value in angles if abs(value) > 1e-8),
                    0.0,
                )
                angles = [0.0, 0.0]
                angles[0 if canonical_layer_index == 0 else 1] = source_angle
            else:
                # Early v5 records can omit face metadata.  Their quaternion
                # is already adjusted independently of the face notation.
                rotation, angles = canonicalize_v6_turn(rotation, angles)
        else:
            rotation = canonicalize_no_turn(rotation)
    elif any(abs(value) > 1e-8 for value in angles):
        rotation, angles = canonicalize_v6_turn(rotation, angles)
    else:
        # v6 carries a canonical quaternion, but older no-turn batches used a
        # cube-symmetry choice that was not based on image zenith.  Re-select
        # the stack axis here so all v7 no-turn labels use one head convention.
        rotation = canonicalize_no_turn(rotation)

    u, v, twist = head_parameters(rotation)
    center = projection.get("cube_center_normalized")
    if not isinstance(center, list) or len(center) != 2:
        raise ValueError(f"{source_name}: invalid cube_center_normalized")
    scale = float(projection.get("image_scale_normalized"))
    perspective = float(projection.get("perspective_strength"))
    return {
        "format": "cube_pose_v7",
        "cube_pose": {
            "u_colatitude_degrees": float(u),
            "v_azimuth_degrees": float(v),
            "twist_degrees": float(twist),
        },
        "image_projection": {
            "model": "scaled_perspective",
            "cube_center_normalized": [float(value) for value in center],
            "image_scale_normalized": scale,
            "perspective_strength": perspective,
        },
        "articulation": {
            "layer_angles_degrees": [float(value) for value in angles],
        },
    }


def convert_directory(source_dir: Path, output_dir: Path, *, overwrite: bool = False) -> int:
    source_dir = source_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if source_dir == output_dir:
        raise ValueError("Source and output directories must be different")
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    converted = 0
    for source_path in sorted(source_dir.glob("*.json")):
        if source_path.name == "schema.json":
            continue
        output_path = output_dir / source_path.name
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {output_path}; pass --overwrite")
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{source_path}: expected a JSON object")
        converted_payload = convert_payload(payload, str(source_path))
        output_path.write_text(json.dumps(converted_payload, indent=2) + "\n", encoding="utf-8")
        converted += 1
    schema = {
        "format": "cube_pose_v7",
        "description": "Direct image-space labels for the custom cube pose head.",
        "cube_pose": {
            "u_colatitude_degrees": "Colatitude from image zenith in [0, 90).",
            "v_azimuth_degrees": "Azimuth around image zenith in [-180, 180).",
            "twist_degrees": "Twist around the stack axis in [-45, 45).",
        },
        "image_projection": "Normalized cube center, image scale, and perspective strength.",
        "articulation": "Negative-Z and positive-Z outer-layer angles in [-45, 45), signed around each layer's outward normal.",
        "source": "Converted from cube_pose_v5 or cube_pose_v6 without modifying source files.",
    }
    schema_path = output_dir / "schema.json"
    if schema_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {schema_path}; pass --overwrite")
    schema_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    return converted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_pose_dir", type=Path)
    parser.add_argument("output_pose_dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    converted = convert_directory(
        args.source_pose_dir,
        args.output_pose_dir,
        overwrite=args.overwrite,
    )
    print(f"converted {converted} labels to {args.output_pose_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
