"""Migrate compact cube_pose_v5 files to the fixed-axis cube_pose_v6 schema."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


AXES = {
    "R": np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
    "L": np.asarray((-1.0, 0.0, 0.0), dtype=np.float64),
    "U": np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
    "D": np.asarray((0.0, 0.0, -1.0), dtype=np.float64),
    "F": np.asarray((0.0, -1.0, 0.0), dtype=np.float64),
    "B": np.asarray((0.0, 1.0, 0.0), dtype=np.float64),
}


def symmetry_matrices() -> tuple[np.ndarray, ...]:
    matrices = []
    for permutation in __import__("itertools").permutations(range(3)):
        for signs in __import__("itertools").product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float64)
            for row, column in enumerate(permutation):
                matrix[row, column] = signs[row]
            if np.linalg.det(matrix) > 0.5:
                matrices.append(matrix)
    return tuple(matrices)


SYMMETRIES = symmetry_matrices()


def quaternion_matrix(values: list[float]) -> np.ndarray:
    x, y, z, w = np.asarray(values, dtype=np.float64) / max(np.linalg.norm(values), 1e-12)
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def matrix_quaternion(matrix: np.ndarray) -> list[float]:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 1e-12)) * 2.0
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
            w = (matrix[2, 1] - matrix[1, 2]) / scale
        elif index == 1:
            scale = math.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 1e-12)) * 2.0
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
            w = (matrix[0, 2] - matrix[2, 0]) / scale
        else:
            scale = math.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 1e-12)) * 2.0
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
            w = (matrix[1, 0] - matrix[0, 1]) / scale
    values = np.asarray((x, y, z, w), dtype=np.float64)
    values /= max(np.linalg.norm(values), 1e-12)
    if values[3] < 0.0:
        values *= -1.0
    return [float(value) for value in values]


def turn_symmetry(rotation: np.ndarray, face: str) -> tuple[np.ndarray, int]:
    source_axis = AXES[face]
    candidates = []
    for symmetry in SYMMETRIES:
        transformed_axis = symmetry.T @ source_axis
        if abs(abs(float(transformed_axis[2])) - 1.0) > 1e-6:
            continue
        candidate = rotation @ symmetry
        angle = math.acos(np.clip((np.trace(candidate) - 1.0) / 2.0, -1.0, 1.0))
        tie_break = tuple(np.round(candidate.reshape(-1), 9))
        candidates.append((angle, tie_break, symmetry, 2 if transformed_axis[2] > 0 else 0))
    if not candidates:
        raise ValueError(f"Could not align turn face {face!r} to the canonical stack axis")
    _angle, _tie_break, symmetry, layer_index = min(candidates, key=lambda item: (item[0], item[1]))
    return symmetry, layer_index


def migrate_payload(payload: dict[str, object]) -> dict[str, object]:
    if payload.get("format") == "cube_pose_v6":
        return payload
    if payload.get("format") != "cube_pose_v5":
        raise ValueError(f"Unsupported pose format: {payload.get('format')!r}")
    rotation_payload = payload["cube_rotation_camera"]
    projection = payload["image_projection"]
    articulation = payload["articulation"]
    rotation = quaternion_matrix(rotation_payload["quaternion_xyzw"])
    layer_angles = [0.0, 0.0]
    turn = articulation.get("turn")
    if isinstance(turn, dict):
        symmetry, layer_index = turn_symmetry(rotation, str(turn["face"]))
        rotation = rotation @ symmetry
        layer_angles[0 if layer_index == 0 else 1] = ((
            float(turn["signed_angle_degrees"]) + 45.0
        ) % 90.0) - 45.0
        if abs(layer_angles[0 if layer_index == 0 else 1]) < 1e-8:
            layer_angles[0 if layer_index == 0 else 1] = 0.0
    return {
        "format": "cube_pose_v6",
        "cube_rotation_camera": {
            "coordinate_convention": "opencv",
            "quaternion_xyzw": matrix_quaternion(rotation),
        },
        "image_projection": projection,
        "articulation": {"layer_angles_degrees": layer_angles},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pose_dir", type=Path)
    args = parser.parse_args()
    paths = sorted(path for path in args.pose_dir.glob("*.json") if path.name != "schema.json")
    converted = 0
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        migrated = migrate_payload(payload)
        if migrated != payload:
            path.write_text(json.dumps(migrated, indent=2) + "\n", encoding="utf-8")
            converted += 1
    schema = args.pose_dir / "schema.json"
    schema.write_text(
        json.dumps(
            {
                "format": "cube_pose_v6",
                "description": "Compact image-space pose for a canonicalized three-layer 1x1x3 articulated cube.",
                "rotation": "cube_rotation_camera.quaternion_xyzw in the OpenCV camera convention, canonicalized with the active layer axis on Z.",
                "projection": "image_projection contains normalized center, normalized image scale, and perspective strength.",
                "articulation": "articulation.layer_angles_degrees contains the negative and positive outer-layer angles, modulo 90 degrees in [-45, 45).",
                "per_image": "{stem}.json",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"converted {converted} pose labels in {args.pose_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

