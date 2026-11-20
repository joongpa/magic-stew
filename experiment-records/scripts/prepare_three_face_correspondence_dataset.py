#!/usr/bin/env python3
"""Prepare a fixed 37-point correspondence dataset from independent v3 labels.

The independent dataset stores visible 3-D points attached to physical cubies.
This converter removes the recorded global cube transform and then undoes any
recorded partial layer turns.  The result is assigned to the corresponding
point on the canonical cube boundary lattice.

The target is the union of three camera-facing faces mapped into a canonical
U, F, R frame.  Each face is a 4x4 grid of boundary-lattice points.  Shared
edges and corners are emitted once, so the target contains 16 + 12 + 9 = 37
unique points.  The canonical order is:

    U face, top-to-bottom / left-to-right;
    F face, top-to-bottom / left-to-right, omitting points already in U;
    R face, top-to-bottom / left-to-right, omitting points already in U/F.

For each source image, the three outward normals facing the recorded camera
are mapped to canonical U/F/R by a proper cube rotation.  This keeps the
network target fixed even when the source renders a different physical face
triple.

Missing observations use YOLO pose visibility 0 and coordinates 0, 0.

The source v3 metadata currently has a truncated final record.  The parser is
deliberately tolerant of a truncated JSON-array tail and records how many
complete records were converted.  The orphaned image is not copied.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/independent_keypoints_dataset_v3_part2"
DEFAULT_OUTPUT = ROOT / "data/independent_keypoints_dataset_v3_three_face"

FACE_AXES: dict[str, tuple[int, int, int]] = {
    "R": (1, 0, 0),
    "L": (-1, 0, 0),
    "U": (0, 0, 1),
    "D": (0, 0, -1),
    "F": (0, -1, 0),
    "B": (0, 1, 0),
}
FACE_SET = frozenset(FACE_AXES)
LATTICE_VALUES = (-1.5, -0.5, 0.5, 1.5)
LATTICE_INDEX_BY_VALUE = {value: index for index, value in enumerate(LATTICE_VALUES)}
MAX_CANONICALIZATION_ERROR = 1e-3
MAX_TURN_ANGLE_DEGREES = 45.0
YOLO_VISIBILITY_BLANK = 0
YOLO_VISIBILITY_VISIBLE = 2


@dataclass(frozen=True)
class PointSlot:
    """One unique lattice point in the emitted target order."""

    index: int
    coordinate: tuple[int, int, int]
    faces: tuple[str, ...]
    name: str


@dataclass(frozen=True)
class LayerOperation:
    face: str
    angle_degrees: float
    direction: float
    objects: frozenset[str]


def _skip_space(text: str, position: int) -> int:
    while position < len(text) and text[position].isspace():
        position += 1
    return position


def iter_metadata_records(path: Path) -> tuple[Iterator[dict[str, Any]], dict[str, int]]:
    """Return a tolerant iterator and parser diagnostics for a JSON array.

    ``json.load`` cannot recover any records from the v3 file because its last
    object is truncated.  ``JSONDecoder.raw_decode`` can still parse every
    complete object before that tail.
    """

    text = path.read_text(encoding="utf-8")
    diagnostics = {
        "bytes": len(text.encode("utf-8")),
        "complete_records": 0,
        "truncated_tail": 0,
    }

    def iterator() -> Iterator[dict[str, Any]]:
        decoder = json.JSONDecoder()
        position = _skip_space(text, 0)
        if position >= len(text) or text[position] != "[":
            raise ValueError(f"Expected a JSON array in {path}")
        position += 1
        while True:
            position = _skip_space(text, position)
            if position >= len(text):
                diagnostics["truncated_tail"] = 1
                return
            if text[position] == "]":
                return
            try:
                record, end = decoder.raw_decode(text, position)
            except json.JSONDecodeError:
                diagnostics["truncated_tail"] = 1
                return
            if not isinstance(record, dict):
                raise ValueError(f"Metadata record {diagnostics['complete_records']} is not an object")
            diagnostics["complete_records"] += 1
            yield record
            position = _skip_space(text, end)
            if position < len(text) and text[position] == ",":
                position += 1
                continue
            if position < len(text) and text[position] == "]":
                return
            diagnostics["truncated_tail"] = 1
            return

    return iterator(), diagnostics


def face_grid(face: str) -> tuple[tuple[int, int, int], ...]:
    """Return a face's 4x4 grid in an intuitive fixed orientation.

    U is viewed from above with B->F rows and L->R columns.  F is viewed from
    the front with U->D rows and L->R columns.  R is viewed from the right
    with U->D rows and F->B columns.  Values are lattice indices 0..3.
    """

    if face not in FACE_SET:
        raise ValueError(f"Unsupported face {face!r}; expected one of {sorted(FACE_SET)}")
    if face == "U":
        return tuple((x, y, 3) for y in (3, 2, 1, 0) for x in range(4))
    if face == "F":
        return tuple((x, 0, z) for z in (3, 2, 1, 0) for x in range(4))
    if face == "R":
        return tuple((3, y, z) for z in (3, 2, 1, 0) for y in range(4))

    # These orientations are included so callers can request another adjacent
    # triple without changing the layout machinery.
    if face == "D":
        return tuple((x, y, 0) for y in range(4) for x in range(4))
    if face == "B":
        return tuple((x, 3, z) for z in (3, 2, 1, 0) for x in (3, 2, 1, 0))
    return tuple((0, y, z) for z in (3, 2, 1, 0) for y in (3, 2, 1, 0))


def validate_face_order(faces: str | Iterable[str]) -> tuple[str, str, str]:
    values = tuple(faces) if isinstance(faces, str) else tuple(faces)
    if len(values) != 3 or len(set(values)) != 3 or any(face not in FACE_SET for face in values):
        raise ValueError(f"Expected three distinct cube faces, got {values!r}")
    normals = [FACE_AXES[face] for face in values]
    axes = [max(range(3), key=lambda index: abs(normal[index])) for normal in normals]
    for first in range(3):
        for second in range(first + 1, 3):
            if axes[first] == axes[second]:
                raise ValueError(f"Faces must be adjacent, got {values!r}")
    return values  # type: ignore[return-value]


def build_point_slots(faces: tuple[str, str, str] = ("U", "F", "R")) -> tuple[PointSlot, ...]:
    faces = validate_face_order(faces)
    membership: dict[tuple[int, int, int], list[str]] = {}
    ordered_coordinates: list[tuple[int, int, int]] = []
    for face in faces:
        for coordinate in face_grid(face):
            if coordinate not in membership:
                ordered_coordinates.append(coordinate)
                membership[coordinate] = []
            membership[coordinate].append(face)
    return tuple(
        PointSlot(
            index=index,
            coordinate=coordinate,
            faces=tuple(membership[coordinate]),
            name="/".join(membership[coordinate]) + ":" + ",".join(str(value) for value in coordinate),
        )
        for index, coordinate in enumerate(ordered_coordinates)
    )


def lattice_point_from_index(coordinate: tuple[int, int, int]) -> np.ndarray:
    return np.asarray([LATTICE_VALUES[index] for index in coordinate], dtype=np.float64)


def _rotation_matrix(axis: tuple[int, int, int], angle_degrees: float) -> np.ndarray:
    unit_axis = np.asarray(axis, dtype=np.float64)
    unit_axis /= np.linalg.norm(unit_axis)
    angle = math.radians(angle_degrees)
    skew = np.asarray(
        [
            [0.0, -unit_axis[2], unit_axis[1]],
            [unit_axis[2], 0.0, -unit_axis[0]],
            [-unit_axis[1], unit_axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return (
        math.cos(angle) * np.eye(3)
        + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * np.outer(unit_axis, unit_axis)
    )


def layer_operations(layer_state: dict[str, Any] | None) -> tuple[LayerOperation, ...]:
    if not isinstance(layer_state, dict) or layer_state.get("mode") not in {"mid_turn", "corner_cut"}:
        return ()
    raw = [("face", "angle_degrees", "direction", "objects")]
    if layer_state.get("mode") == "corner_cut":
        raw.append(("adjacent_face", "adjacent_angle_degrees", "adjacent_direction", "adjacent_objects"))
    operations: list[LayerOperation] = []
    for face_key, angle_key, direction_key, objects_key in raw:
        face = layer_state.get(face_key)
        angle = layer_state.get(angle_key)
        direction = layer_state.get(direction_key, 1.0)
        objects = layer_state.get(objects_key)
        if face not in FACE_SET or not isinstance(angle, (int, float)) or not isinstance(objects, list):
            raise ValueError(f"Incomplete layer operation in {layer_state!r}")
        angle = float(angle)
        if abs(angle) > MAX_TURN_ANGLE_DEGREES + 1e-6:
            raise ValueError(f"Turn angle exceeds 45 degrees: {angle}")
        if not isinstance(direction, (int, float)):
            direction = 1.0
        operations.append(LayerOperation(face, angle, float(direction), frozenset(str(name) for name in objects)))
    return tuple(operations)


def canonicalize_world_point(
    world_point: Iterable[float],
    cube_transform: Iterable[Iterable[float]],
    source_object: str,
    layer_state: dict[str, Any] | None,
    canonical_frame: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Undo global pose and source-cubie partial turns, returning snap error."""

    transform = np.asarray(cube_transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 cube transform, got {transform.shape}")
    local = (np.linalg.inv(transform) @ np.r_[np.asarray(tuple(world_point), dtype=np.float64), 1.0])[:3]
    for operation in reversed(layer_operations(layer_state)):
        if source_object not in operation.objects:
            continue
        inverse_rotation = _rotation_matrix(
            FACE_AXES[operation.face],
            -operation.direction * operation.angle_degrees,
        )
        local = inverse_rotation @ local
    if canonical_frame is not None:
        local = canonical_frame @ local
    nearest = np.rint(local + 1.5) - 1.5
    return local, float(np.linalg.norm(local - nearest))


def camera_facing_frame(
    record: dict[str, Any],
    target_faces: tuple[str, str, str] = ("U", "F", "R"),
) -> tuple[np.ndarray, tuple[str, str, str]]:
    """Map this sample's camera-facing face basis to the target face frame."""

    cube_pose = record.get("cube_pose") or {}
    transform = np.asarray(cube_pose.get("transform"), dtype=np.float64)
    camera = record.get("camera") or {}
    camera_location = np.asarray(camera.get("location"), dtype=np.float64)
    if transform.shape != (4, 4) or camera_location.shape != (3,):
        raise ValueError(f"Record {record.get('image')} lacks pose/camera metadata")
    rotation = transform[:3, :3]
    center = transform[:3, 3]
    camera_direction = camera_location - center
    if np.linalg.norm(camera_direction) <= 1e-9:
        raise ValueError(f"Record {record.get('image')} has camera at cube center")
    camera_direction /= np.linalg.norm(camera_direction)

    candidates: list[tuple[str, np.ndarray, float]] = []
    for face, local_normal in FACE_AXES.items():
        world_normal = rotation @ np.asarray(local_normal, dtype=np.float64)
        world_normal /= np.linalg.norm(world_normal)
        candidates.append((face, np.asarray(local_normal, dtype=np.float64), float(world_normal @ camera_direction)))
    visible = [candidate for candidate in candidates if candidate[2] > 0.0]
    if len(visible) != 3:
        # A camera exactly on a face boundary can make one dot product zero.
        # Select the strongest face from each axis pair in that case.
        visible = []
        for axis in range(3):
            pair = [candidate for candidate in candidates if abs(candidate[1][axis]) > 0.5]
            visible.append(max(pair, key=lambda candidate: candidate[2]))
        if len({candidate[0] for candidate in visible}) != 3:
            raise ValueError(f"Could not identify three camera-facing faces for {record.get('image')}")

    world_up = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    top = max(visible, key=lambda candidate: float((rotation @ candidate[1]) @ world_up))
    sides = [candidate for candidate in visible if candidate[0] != top[0]]
    if len(sides) != 2:
        raise ValueError(f"Could not identify two side faces for {record.get('image')}")
    side_a = sides[0][1]
    side_b = sides[1][1]
    if np.dot(np.cross(top[1], side_a), side_b) > 0.5:
        front, right = side_a, side_b
        front_name, right_name = sides[0][0], sides[1][0]
    else:
        front, right = side_b, side_a
        front_name, right_name = sides[1][0], sides[0][0]

    source_basis = np.stack((top[1], front, right), axis=1)
    target_basis = np.stack(
        (
            np.asarray(FACE_AXES[target_faces[0]], dtype=np.float64),
            np.asarray(FACE_AXES[target_faces[1]], dtype=np.float64),
            np.asarray(FACE_AXES[target_faces[2]], dtype=np.float64),
        ),
        axis=1,
    )
    frame = target_basis @ source_basis.T
    if np.linalg.det(frame) < 0.5:
        raise ValueError(f"Camera-facing frame is not a proper rotation for {record.get('image')}")
    return frame, (top[0], front_name, right_name)


def lattice_coordinate(local_point: np.ndarray) -> tuple[int, int, int]:
    nearest = np.rint(local_point + 1.5).astype(int)
    if np.any(nearest < 0) or np.any(nearest > 3):
        raise ValueError(f"Point lies outside the cube lattice: {local_point.tolist()}")
    coordinate = tuple(int(value) for value in nearest)
    if not any(value in (0, 3) for value in coordinate):
        raise ValueError(f"Point is not on the cube boundary: {coordinate}")
    return coordinate


def _point_records(record: dict[str, Any]) -> Iterator[dict[str, Any]]:
    independent = record.get("independent_keypoints") or {}
    for object_record in independent.get("objects") or []:
        for point in object_record.get("keypoints") or []:
            yield point


def _bbox(points: list[tuple[float, float]], width: int, height: int) -> tuple[float, float, float, float]:
    radius = max(3.0, round(min(width, height) * 0.02))
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x1 = max(0.0, min(xs) - radius)
    y1 = max(0.0, min(ys) - radius)
    x2 = min(float(width), max(xs) + radius)
    y2 = min(float(height), max(ys) + radius)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, max(1.0, x2 - x1), max(1.0, y2 - y1)


def make_pose_label(
    record: dict[str, Any],
    slots: tuple[PointSlot, ...],
    target_faces: tuple[str, str, str] = ("U", "F", "R"),
) -> tuple[str, dict[str, Any]]:
    image_size = record.get("image_size") or {}
    width = int(image_size.get("width", 256))
    height = int(image_size.get("height", 256))
    cube_pose = record.get("cube_pose") or {}
    transform = cube_pose.get("transform")
    if transform is None:
        raise ValueError(f"Record {record.get('image')} is missing cube_pose.transform")
    slot_by_coordinate = {slot.coordinate: slot for slot in slots}
    canonical_frame, source_faces = camera_facing_frame(record, target_faces)
    observations: dict[int, tuple[float, float, int, float]] = {}
    max_error = 0.0
    source_point_count = 0
    for point in _point_records(record):
        source_object = point.get("source_object")
        world_point = point.get("world_point")
        pixel_point = point.get("pixel_point")
        if not isinstance(source_object, str) or not isinstance(world_point, list) or not isinstance(pixel_point, list):
            continue
        source_point_count += 1
        local, error = canonicalize_world_point(
            world_point,
            transform,
            source_object,
            record.get("layer_state"),
            canonical_frame=canonical_frame,
        )
        max_error = max(max_error, error)
        if error > MAX_CANONICALIZATION_ERROR:
            raise ValueError(
                f"{record.get('image')} point {source_object} cannot be canonicalized "
                f"(snap error {error:.6f})"
            )
        coordinate = lattice_coordinate(local)
        slot = slot_by_coordinate.get(coordinate)
        if slot is None:
            continue
        x, y = float(pixel_point[0]), float(pixel_point[1])
        visibility = int(point.get("visibility", YOLO_VISIBILITY_VISIBLE))
        candidate = (x, y, visibility, error)
        existing = observations.get(slot.index)
        if existing is not None:
            # A seam can theoretically produce two metadata points for one
            # canonical slot. Prefer visible evidence, then the tighter snap.
            if (visibility > existing[2]) or (visibility == existing[2] and error < existing[3]):
                observations[slot.index] = candidate
        else:
            observations[slot.index] = candidate

    visible_points = [(value[0], value[1]) for value in observations.values() if value[2] != YOLO_VISIBILITY_BLANK]
    if not visible_points:
        # The source renderer can legitimately produce a frame with no
        # visible independent point (for example, a hand completely blocks
        # the cube).  Preserve it as an empty pose label rather than inventing
        # a bounding box.  Ultralytics treats this as a negative sample.
        return "", {
            "source_image": record.get("image"),
            "layer_mode": (record.get("layer_state") or {}).get("mode", "none"),
            "source_face_order": list(source_faces),
            "source_point_count": source_point_count,
            "observed_point_count": 0,
            "blank_point_count": len(slots),
            "max_canonicalization_error": max_error,
            "keypoints": [
                {"index": slot.index, "name": slot.name, "coordinate": list(slot.coordinate), "visibility": YOLO_VISIBILITY_BLANK}
                for slot in slots
            ],
        }
    bbox = _bbox(visible_points, width, height)
    values = ["0", *(f"{value:.6f}" for value in (bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height))]
    keypoint_metadata = []
    for slot in slots:
        observation = observations.get(slot.index)
        if observation is None:
            values.extend(("0.000000", "0.000000", str(YOLO_VISIBILITY_BLANK)))
            keypoint_metadata.append({"index": slot.index, "name": slot.name, "coordinate": list(slot.coordinate), "visibility": YOLO_VISIBILITY_BLANK})
            continue
        x, y, visibility, error = observation
        values.extend((f"{x / width:.6f}", f"{y / height:.6f}", str(visibility)))
        keypoint_metadata.append({"index": slot.index, "name": slot.name, "coordinate": list(slot.coordinate), "pixel_point": [x, y], "visibility": visibility, "canonicalization_error": error})
    return " ".join(values), {
        "source_image": record.get("image"),
        "layer_mode": (record.get("layer_state") or {}).get("mode", "none"),
        "source_face_order": list(source_faces),
        "source_point_count": source_point_count,
        "observed_point_count": len(observations),
        "blank_point_count": len(slots) - len(observations),
        "max_canonicalization_error": max_error,
        "keypoints": keypoint_metadata,
    }


def manifest_for_slots(slots: tuple[PointSlot, ...], faces: tuple[str, str, str]) -> dict[str, Any]:
    return {
        "format": "three_face_correspondence_v1",
        "source_keypoint_dataset": "independent_keypoints_dataset_v3_part2",
        "faces": list(faces),
        "keypoint_count": len(slots),
        "keypoint_shape": [len(slots), 3],
        "blank_visibility": 0,
        "visible_visibility": 2,
        "coordinate_convention": "cube-local boundary lattice indices 0..3; metric coordinate = index - 1.5",
        "ordering": "camera-facing source faces mapped to canonical U/F/R; face-major row-major; shared edges and corners emitted once",
        "partial_turn_policy": "inverse recorded layer turns, then assign to the pre-turn lattice coordinate",
        "keypoints": [
            {"index": slot.index, "name": slot.name, "coordinate": list(slot.coordinate), "faces": list(slot.faces)}
            for slot in slots
        ],
    }


def prepare_dataset(source: Path, output: Path, faces: tuple[str, str, str], force: bool) -> dict[str, Any]:
    metadata_path = source / "metadata.json"
    images_dir = source / "images"
    if not metadata_path.is_file() or not images_dir.is_dir():
        raise FileNotFoundError(f"Expected metadata.json and images/ below {source}")
    if output.exists():
        if not force:
            raise FileExistsError(f"Output exists: {output}; pass --force to replace it")
        shutil.rmtree(output)
    (output / "images").mkdir(parents=True)
    (output / "pose").mkdir()
    slots = build_point_slots(faces)
    records, diagnostics = iter_metadata_records(metadata_path)
    output_records = []
    layer_counts: dict[str, int] = {}
    copied_stems: set[str] = set()
    for record in records:
        source_image = Path(str(record.get("image", "")))
        image_path = source / "images" / source_image.name
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing source image for metadata record: {image_path}")
        label, compact_metadata = make_pose_label(record, slots, faces)
        stem = image_path.stem
        shutil.copy2(image_path, output / "images" / image_path.name)
        (output / "pose" / f"{stem}.txt").write_text(
            label + ("\n" if label else ""),
            encoding="utf-8",
        )
        copied_stems.add(stem)
        mode = str(compact_metadata["layer_mode"])
        layer_counts[mode] = layer_counts.get(mode, 0) + 1
        output_records.append({"stem": stem, **compact_metadata})

    all_source_stems = {path.stem for path in images_dir.glob("*.png")}
    orphan_stems = sorted(all_source_stems - copied_stems)
    (output / "correspondence_manifest.json").write_text(
        json.dumps(manifest_for_slots(slots, faces), indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "metadata.json").write_text(json.dumps(output_records, indent=2) + "\n", encoding="utf-8")
    (output / "conversion_summary.json").write_text(
        json.dumps({
            "source": str(source),
            "output": str(output),
            "records": len(output_records),
            "images_copied": len(copied_stems),
            "orphan_source_images": orphan_stems,
            "metadata_parser": diagnostics,
            "layer_modes": layer_counts,
            "keypoint_count": len(slots),
            "blank_counts": {
                "min": min(record["blank_point_count"] for record in output_records),
                "max": max(record["blank_point_count"] for record in output_records),
                "mean": sum(record["blank_point_count"] for record in output_records) / len(output_records),
            },
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "output": str(output),
        "records": len(output_records),
        "orphan_source_images": orphan_stems,
        "layer_modes": layer_counts,
        "keypoint_count": len(slots),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--faces", default="UFR", help="Three adjacent faces in output order; default UFR")
    parser.add_argument("--force", action="store_true", help="Replace an existing output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    faces = validate_face_order(args.faces)
    summary = prepare_dataset(args.source, args.output, faces, args.force)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
