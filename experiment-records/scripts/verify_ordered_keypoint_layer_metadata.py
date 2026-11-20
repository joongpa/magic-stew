#!/usr/bin/env python3
"""Verify ordered-keypoint v2 layer-state metadata.

This checker deliberately mirrors the geometry in
``apply_blender_cube_moves.py`` without importing Blender.  It verifies that
the stored ``local_point`` for every ordered keypoint can be derived from:

  * its canonical name (for example ``DFR`` or ``F``), and
  * the recorded ``layer_state`` (face, angle, direction, and optional second
    face for a corner cut).

The checker does not use visibility to decide whether a point is valid.  A
point may be hidden or unlabeled in the rendered image while its geometric
target must still be present in the metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_NAMES = (
    "UBL", "UBR", "UFR", "UFL",
    "DBL", "DBR", "DFR", "DFL",
    "UB", "UR", "UF", "UL",
    "DB", "DR", "DF", "DL",
    "BR", "BL", "FR", "FL",
    "U", "D", "F", "B", "R", "L",
)

FACE_AXES: dict[str, tuple[float, float, float]] = {
    "R": (1.0, 0.0, 0.0),
    "L": (-1.0, 0.0, 0.0),
    "U": (0.0, 0.0, 1.0),
    "D": (0.0, 0.0, -1.0),
    # This matches the generator's default FRONT_AXIS of -Y.
    "F": (0.0, -1.0, 0.0),
    "B": (0.0, 1.0, 0.0),
}

FACE_COORDINATES: dict[str, tuple[int, float]] = {
    "R": (0, 1.5),
    "L": (0, -1.5),
    "B": (1, 1.5),
    "F": (1, -1.5),
    "U": (2, 1.5),
    "D": (2, -1.5),
}


def add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(x + y for x, y in zip(a, b))  # type: ignore[return-value]


def sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(x - y for x, y in zip(a, b))  # type: ignore[return-value]


def scale(a: tuple[float, float, float], factor: float) -> tuple[float, float, float]:
    return tuple(factor * x for x in a)  # type: ignore[return-value]


def dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def rotate(
    point: tuple[float, float, float],
    axis: tuple[float, float, float],
    angle_radians: float,
) -> tuple[float, float, float]:
    """Rotate a vector using the same right-handed convention as Blender."""
    cosine = math.cos(angle_radians)
    sine = math.sin(angle_radians)
    return add(
        add(scale(point, cosine), scale(cross(axis, point), sine)),
        scale(axis, dot(axis, point) * (1.0 - cosine)),
    )


def cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def canonical_point(name: str) -> tuple[float, float, float]:
    """Reproduce ordered_cube_keypoint_specs() from its face names."""
    point = (0.0, 0.0, 0.0)
    for face in name:
        point = add(point, scale(FACE_AXES[face], 1.5))
    return point


def source_cubie_center(name: str) -> tuple[float, float, float]:
    """Reproduce ordered_keypoint_source_cubie_center()."""
    center = [0.0, 0.0, 0.0]
    for face in name:
        axis, value = FACE_COORDINATES[face]
        center[axis] = 1.0 if value > 1.0 else -1.0
    return tuple(center)  # type: ignore[return-value]


def transformed_point(
    name: str,
    layer_state: dict[str, Any],
) -> tuple[float, float, float]:
    """Reproduce the point portion of dimension_cubie_transforms()."""
    mode = layer_state.get("mode", "none")
    base = canonical_point(name)
    if mode == "none":
        return base
    if mode not in {"mid_turn", "corner_cut"}:
        raise ValueError(f"Unsupported layer mode {mode!r}")

    source = source_cubie_center(name)
    face = layer_state.get("face")
    if face not in FACE_AXES:
        raise ValueError(f"Invalid layer-state face {face!r}")

    def apply_turn(
        center: tuple[float, float, float],
        point: tuple[float, float, float],
        turn_face: str,
        angle_degrees: float,
        direction: float,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        axis = FACE_AXES[turn_face]
        # This is equivalent to the generator's
        # ``turn @ initial_center`` and ``center + turn @ (point-center)``.
        new_center = rotate(center, axis, math.radians(direction * angle_degrees))
        new_point = add(
            new_center,
            rotate(sub(point, center), axis, math.radians(direction * angle_degrees)),
        )
        return new_center, new_point

    primary_face = str(face)
    primary_axis = FACE_AXES[primary_face]
    center = source
    point = base
    if round(dot(source, primary_axis)) == 1:
        center, point = apply_turn(
            center,
            point,
            primary_face,
            float(layer_state["angle_degrees"]),
            float(layer_state["direction"]),
        )

    if mode == "corner_cut":
        second_face = layer_state.get("adjacent_face")
        if second_face not in FACE_AXES:
            raise ValueError(f"Invalid corner-cut adjacent face {second_face!r}")
        # The generator selects the second layer using the already-transformed
        # center, then applies the second rotation to that cubie's center and
        # orientation.
        if round(dot(center, FACE_AXES[str(second_face)])) == 1:
            center, point = apply_turn(
                center,
                point,
                str(second_face),
                float(layer_state["adjacent_angle_degrees"]),
                float(layer_state["adjacent_direction"]),
            )
    return point


def distance(a: Any, b: Any) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def expected_moved_names(layer_state: dict[str, Any]) -> set[str]:
    mode = layer_state.get("mode", "none")
    if mode == "none":
        return set()
    face = layer_state.get("face")
    if face not in FACE_AXES:
        raise ValueError(f"Invalid layer-state face {face!r}")
    names = set()
    for name in EXPECTED_NAMES:
        if round(dot(source_cubie_center(name), FACE_AXES[str(face)])) == 1:
            names.add(name)
    return names


def validate_record(record: dict[str, Any], tolerance: float) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    image = str(record.get("image", "<unknown>"))
    layer_state = record.get("layer_state") or {"mode": "none"}
    ordered = record.get("ordered_keypoints") or {}
    keypoints = ordered.get("keypoints") or []
    names = ordered.get("keypoint_names")
    mode = layer_state.get("mode", "none")

    if tuple(names or ()) != EXPECTED_NAMES:
        errors.append("keypoint_names does not match the v2 convention")
    if ordered.get("keypoint_shape") != [26, 3]:
        errors.append(f"keypoint_shape is {ordered.get('keypoint_shape')!r}, expected [26, 3]")
    if ordered.get("layer_mode", "none") != mode:
        errors.append("ordered_keypoints.layer_mode disagrees with layer_state.mode")

    by_index = {kp.get("index"): kp for kp in keypoints if isinstance(kp, dict)}
    if len(keypoints) != 26:
        errors.append(f"stored {len(keypoints)} keypoints, expected 26")
    if set(by_index) != set(range(26)):
        errors.append("keypoint indices are missing or duplicated")

    expected_moved = expected_moved_names(layer_state)
    expected_displaced: set[str] = set()
    actual_displaced: set[str] = set()
    max_error = 0.0
    max_error_name = None
    geometry_missing: list[str] = []
    geometry_errors: list[dict[str, Any]] = []
    projection_errors: list[str] = []

    image_size = record.get("image_size") or [None, None]
    width, height = image_size if len(image_size) == 2 else (None, None)
    for index, expected_name in enumerate(EXPECTED_NAMES):
        kp = by_index.get(index)
        if kp is None:
            continue
        if kp.get("name") != expected_name:
            errors.append(f"index {index} is named {kp.get('name')!r}, expected {expected_name!r}")
        if "local_point" not in kp:
            geometry_missing.append(expected_name)
            continue
        expected = transformed_point(expected_name, layer_state)
        actual = kp["local_point"]
        point_error = distance(actual, expected)
        if distance(expected, canonical_point(expected_name)) > tolerance:
            expected_displaced.add(expected_name)
        if point_error > tolerance:
            geometry_errors.append({
                "index": index,
                "name": expected_name,
                "error": point_error,
                "expected": list(expected),
                "actual": actual,
            })
        if point_error > max_error:
            max_error = point_error
            max_error_name = expected_name
        if distance(actual, canonical_point(expected_name)) > tolerance:
            actual_displaced.add(expected_name)

        if width and height and "pixel_point" in kp and "normalized_point" in kp:
            pixel = kp["pixel_point"]
            normalized = kp["normalized_point"]
            if kp.get("visibility", 0) == 0:
                continue
            expected_normalized = (float(pixel[0]) / float(width), float(pixel[1]) / float(height))
            if distance(normalized, expected_normalized) > 2e-5:
                projection_errors.append(expected_name)

    if geometry_missing:
        errors.append(f"missing local_point for {geometry_missing}")
    if geometry_errors:
        errors.append(f"{len(geometry_errors)} local_point values do not match derived geometry")
    if projection_errors:
        errors.append(f"pixel/normalized metadata mismatch for {projection_errors}")

    if mode == "mid_turn":
        objects = layer_state.get("objects") or []
        if len(objects) != 9:
            errors.append(f"mid_turn lists {len(objects)} moved cubies, expected 9")
    if mode == "corner_cut":
        if layer_state.get("adjacent_face") not in FACE_AXES:
            errors.append("corner_cut is missing adjacent_face")

    if actual_displaced != expected_displaced:
        errors.append(
            "displaced keypoints do not match the derived layer geometry "
            f"(expected {sorted(expected_displaced)}, actual {sorted(actual_displaced)})"
        )

    visible_moved = sum(
        1
        for kp in keypoints
        if kp.get("name") in expected_moved and kp.get("visibility", 0) != 0
    )
    return {
        "image": image,
        "mode": mode,
        "face": layer_state.get("face"),
        "angle_degrees": layer_state.get("angle_degrees", 0.0),
        "expected_moved": sorted(expected_moved),
        "expected_displaced": sorted(expected_displaced),
        "actual_displaced": sorted(actual_displaced),
        "visible_moved_count": visible_moved,
        "max_geometry_error": max_error,
        "max_geometry_error_name": max_error_name,
        "geometry_errors": geometry_errors,
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/ordered_pose_dataset_v2"))
    parser.add_argument("--metadata", type=Path, help="Override source-dir/metadata.json")
    parser.add_argument("--mode", choices=("mid_turn", "corner_cut", "none", "all"), default="mid_turn")
    parser.add_argument("--image-id", action="append", help="Check only this image stem; may be repeated")
    parser.add_argument("--limit", type=int, help="Check at most this many matching records")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--json-out", type=Path, help="Write the detailed report as JSON")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit 1 if any record fails")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata_path = args.metadata or args.source_dir / "metadata.json"
    with metadata_path.open() as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise SystemExit(f"Expected a list in {metadata_path}")

    requested_ids = set(args.image_id or ())
    selected = []
    for record in records:
        layer_state = record.get("layer_state") or {"mode": "none"}
        mode = layer_state.get("mode", "none")
        image_id = Path(str(record.get("image", ""))).stem
        if args.mode != "all" and mode != args.mode:
            continue
        if requested_ids and image_id not in requested_ids:
            continue
        selected.append(record)
        if args.limit is not None and len(selected) >= args.limit:
            break

    results = [validate_record(record, args.tolerance) for record in selected]
    error_count = sum(not result["ok"] for result in results)
    by_mode = Counter(result["mode"] for result in results)
    by_face = Counter(result["face"] for result in results if result["face"])
    max_error = max((result["max_geometry_error"] for result in results), default=0.0)

    print(f"Metadata: {metadata_path}")
    print(f"Checked: {len(results)} records | modes: {dict(sorted(by_mode.items()))}")
    if by_face:
        print(f"Faces: {dict(sorted(by_face.items()))}")
    print(f"Geometry tolerance: {args.tolerance:g}")
    print(f"Maximum derived local-point error: {max_error:.3e}")
    print(f"Result: {'PASS' if error_count == 0 else f'{error_count} FAILED RECORD(S)'}")

    for result in results:
        if result["errors"]:
            print(f"\nFAIL {result['image']}")
            for error in result["errors"]:
                print(f"  - {error}")
    if args.json_out:
        report = {
            "metadata": str(metadata_path),
            "mode_filter": args.mode,
            "record_count": len(results),
            "failed_count": error_count,
            "results": results,
        }
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Detailed report: {args.json_out}")

    if args.fail_on_error and error_count:
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, TypeError, ValueError) as exc:
        print(f"Metadata validation could not run: {exc}", file=sys.stderr)
        raise SystemExit(2)
