#!/usr/bin/env python3
"""Build and visually verify a small direct-pose target dataset.

    .venv/bin/python scripts/prepare_direct_cube_pose_targets.py \
        --source-dir data/keypoints_only_dataset_part7 \
        --output-dir runs/direct_pose_targets_verification --limit 30

Reads only metadata/<stem>.json and existing images/pose labels. Source files
are never edited. No Blender, model inference, dense maps or cube-core class
is needed. All generator articulation modes are supported. Slice turns rebase
the global color frame and compensate with opposite outer-face offsets.
Corner cuts use only the larger angle. The exact two-operation geometry is
also reconstructed as a diagnostic, making the approximation error explicit.
Angles are clockwise looking at the named color face, with
output order U,D,F,B,R,L. Optional --audit-all checks every metadata record
while exporting images/targets/overlays only for the selected subset.

The source camera uses square pixels, horizontal sensor fit, no distortion,
and a 36 mm sensor width (override if the source blend differs). This assumption
is checked against stored landmark world/image correspondences. Source sensor
shifts are preserved in the exact projector. A separate diagnostic measures
the earlier center + scale*XY/(1 + perspective*Z/3) approximation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations, product
import json
from pathlib import Path
import random
import shutil

import cv2
import numpy as np

FACE_ORDER = ("U", "D", "F", "B", "R", "L")
FACE_AXES = np.asarray(((0, 0, 1), (0, 0, -1), (0, -1, 0),
                        (0, 1, 0), (1, 0, 0), (-1, 0, 0)), dtype=np.float64)
FACE_COLORS = ("white", "yellow", "green", "blue", "red", "orange")
FACE_BGR = ((255, 255, 255), (0, 240, 255), (40, 220, 40),
            (255, 130, 40), (50, 50, 240), (0, 150, 255))
CLASS_BGR = ((0, 160, 255), (255, 230, 0), (240, 130, 255))
SUPPORTED_MODES = {"none", "mid_turn", "simultaneous_turn", "slice_turn", "corner_cut"}
SLICE_REFERENCE_FACE = {"M": "L", "E": "D", "S": "F"}
CORNER_NAMES = ("UBL", "UBR", "UFR", "UFL", "DBL", "DBR", "DFR", "DFL")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


@dataclass
class Sample:
    root: Path
    metadata_path: Path
    image_path: Path
    record: dict
    angles: np.ndarray

    @property
    def mode(self):
        return self.record.get("layer_state", {}).get("mode", "none")

    @property
    def key(self):
        return f"{self.root.name}__{self.metadata_path.stem}"


def rotation(axis, angle_degrees):
    return cv2.Rodrigues(np.asarray(axis, dtype=np.float64) * np.deg2rad(angle_degrees))[0]


def scramble_rotation(record):
    raw = np.asarray(record["cube_pose"]["wireframe_scramble_transform"], dtype=np.float64)[:3, :3]
    snapped = np.rint(raw)
    if not np.allclose(raw, snapped, atol=1e-4) or not np.array_equal(snapped.T @ snapped, np.eye(3)):
        raise ValueError("Whole-cube scramble transform is not a signed axis permutation")
    if np.linalg.det(snapped) < 0:
        raise ValueError("Whole-cube scramble transform reflects the cube")
    return snapped


def articulation_parameters(record, *, dominant_corner=True):
    state = record.get("layer_state", {})
    mode = state.get("mode", "none")
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"unsupported articulation: {mode}")
    angles = np.zeros(6)
    operations = state.get("operations")
    if operations is None:
        operations = [] if mode == "none" else [state]
    expected = {"none": 0, "mid_turn": 1, "simultaneous_turn": 2,
                "slice_turn": 1, "corner_cut": 2}[mode]
    if len(operations) != expected:
        raise ValueError(f"{mode} requires {expected} operations")
    scramble = scramble_rotation(record)
    selected = []
    ordered = []
    rebase = np.eye(3)
    for operation in operations:
        move = operation.get("move", operation.get("face"))
        is_slice = move in SLICE_REFERENCE_FACE
        if is_slice != (mode == "slice_turn"):
            raise ValueError("Slice moves are only valid in slice_turn records")
        reference_face = SLICE_REFERENCE_FACE.get(move, move)
        if reference_face not in FACE_ORDER:
            raise ValueError(f"unsupported outer-layer move: {move}")
        axis = scramble.T @ FACE_AXES[FACE_ORDER.index(reference_face)]
        face = int(np.argmax(FACE_AXES @ axis))
        selected.append(face)
        # The generator multiplies BOTH fields, and angle_degrees can be negative.
        # Its right-hand rotation becomes negative in clockwise face notation.
        angle = -float(operation["direction"]) * float(operation["angle_degrees"])
        if not np.isfinite(angle) or not -45 <= angle <= 45:
            raise ValueError(f"Unexpected layer offset outside [-45,45]: {angle}")
        if is_slice:
            # Rotate the reference frame with the middle slice's four centers.
            # Undo that global rotation on both outer layers. This is exactly
            # the same geometry, with no extra slice-angle output required.
            opposite = int(np.argmax(FACE_AXES @ -axis))
            rebase = rotation(axis, -angle)
            angles[face], angles[opposite] = -angle, angle
            ordered.extend([{"face": FACE_ORDER[face], "angle_degrees": -angle},
                            {"face": FACE_ORDER[opposite], "angle_degrees": angle}])
        else:
            angles[face] = angle
            ordered.append({"face": FACE_ORDER[face], "angle_degrees": angle})
    if mode == "simultaneous_turn" and not np.array_equal(FACE_AXES[selected[0]], -FACE_AXES[selected[1]]):
        raise ValueError("Simultaneous operations must select distinct opposite faces")
    if mode == "corner_cut" and abs(FACE_AXES[selected[0]] @ FACE_AXES[selected[1]]) > 1e-8:
        raise ValueError("Corner-cut operations must select adjacent faces")
    if mode == "corner_cut" and dominant_corner:
        dominant = max(ordered, key=lambda operation: abs(operation["angle_degrees"]))
        angles[:] = 0
        angles[FACE_ORDER.index(dominant["face"])] = dominant["angle_degrees"]
        ordered = [dominant]
    return angles, ordered, rebase


def layer_angles(record):
    return articulation_parameters(record)[0]


def camera_parameters(record, sensor_width_mm=36.0):
    size = record["image_size"]
    width, height = int(size["width"]), int(size["height"])
    if width != height:
        raise ValueError("Initial verifier supports square source images only")
    camera = record["camera"]
    location = np.asarray(camera["location"], dtype=np.float64)
    forward = np.asarray(camera["target"], dtype=np.float64) - location
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, (0, 0, 1))
    if np.linalg.norm(right) < 1e-8:
        raise ValueError("Camera look-at direction is parallel to the world up axis")
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    angle = np.deg2rad(float(camera["roll_degrees"]))
    rolled_right = np.cos(angle) * right - np.sin(angle) * up
    rolled_up = np.sin(angle) * right + np.cos(angle) * up
    world_to_camera = np.stack((rolled_right, -rolled_up, forward))
    focal = float(camera["lens_mm"]) * width / sensor_width_mm
    shifts = camera["roi_crop"]
    cx = width * (0.5 - float(shifts["shift_x"]))
    cy = height * 0.5 + width * float(shifts["shift_y"])
    intrinsics = np.asarray(((focal, 0, cx), (0, focal, cy), (0, 0, 1)))
    return world_to_camera, location, intrinsics, (width, height)


def project_camera(points, intrinsics):
    points = np.asarray(points, dtype=np.float64)
    if not np.isfinite(points).all() or np.any(points[:, 2] <= 0):
        raise ValueError("Projection contains nonfinite points or points behind the camera")
    projected = points @ intrinsics.T
    return projected[:, :2] / projected[:, 2:]


def project_target(points, target, *, compact=False):
    projection = target["projection"]
    rotated = np.asarray(points) @ np.asarray(target["rotation_matrix"]).T
    center = np.asarray(projection["center_xy_normalized"])
    denominator = 1 + projection["perspective_strength"] * rotated[:, 2:] / 3
    if np.any(denominator <= 0):
        raise ValueError("Projection denominator is nonpositive")
    scaled = projection["image_scale_normalized"] * rotated[:, :2]
    if compact:
        normalized = center + scaled / denominator
    else:
        principal = np.asarray(projection["principal_point_normalized"])
        normalized = principal + (center - principal + scaled) / denominator
    return normalized * np.asarray(target["image_size"])


def articulate(points, owners, angles, *, operations=None):
    result = np.asarray(points, dtype=np.float64).copy()
    centers = np.asarray(owners, dtype=np.float64).copy()
    if operations is None:
        operations = [{"face": face, "angle_degrees": angle}
                      for face, angle in zip(FACE_ORDER, angles)]
    for operation in operations:
        axis = FACE_AXES[FACE_ORDER.index(operation["face"])]
        angle = operation["angle_degrees"]
        if abs(angle) > 1e-10:
            # Match the generator's round(current_center.dot(axis)) == 1.
            selected = np.rint(centers @ axis) == 1
            turn = rotation(axis, -angle)
            result[selected] = result[selected] @ turn.T
            centers[selected] = centers[selected] @ turn.T
    return result


def landmark_template():
    points, classes, names = [], [], []
    for outer_count, class_id in ((3, 0), (2, 1), (1, 2)):
        for outer_axes in combinations(range(3), outer_count):
            for signs in product((-1, 1), repeat=3):
                point = [signs[i] * (1.5 if i in outer_axes else 0.5) for i in range(3)]
                points.append(point)
                classes.append(class_id)
                names.append(f"landmark_{class_id}_{len(names):02d}")
    points = np.asarray(points)
    owners = np.where(np.abs(points) > 1, np.sign(points), 0)
    return points, owners, np.asarray(classes), names


def surface_quads(angles, *, operations=None):
    points, owners, normals = [], [], []
    for center in product((-1, 0, 1), repeat=3):
        for axis_index in range(3):
            if center[axis_index] == 0:
                continue
            normal = np.eye(3)[axis_index] * center[axis_index]
            tangent = [i for i in range(3) if i != axis_index]
            for first, second in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                point = np.asarray(center, dtype=np.float64) + normal * 0.5
                point[tangent[0]] += first * 0.5
                point[tangent[1]] += second * 0.5
                points.append(point)
                owners.append(center)
                normals.append(normal)
    points = articulate(points, owners, angles, operations=operations)
    normals = articulate(normals, owners, angles, operations=operations)
    return points.reshape(-1, 4, 3), normals.reshape(-1, 4, 3)[:, 0]


def error_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {"count": int(values.size), "mean": float(values.mean()),
            "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def build_target(record, *, sensor_width_mm=36.0):
    angles, operations, rebase = articulation_parameters(record)
    camera_rotation, camera_location, intrinsics, size = camera_parameters(record, sensor_width_mm)
    transform = np.asarray(record["cube_pose"]["transform"], dtype=np.float64)
    core_to_world = transform[:3, :3] @ scramble_rotation(record) @ rebase
    # Normalize small accumulated Blender float errors into a proper rotation.
    u, _, vt = np.linalg.svd(core_to_world)
    core_to_world = u @ vt
    core_rotation = camera_rotation @ core_to_world
    translation = camera_rotation @ (transform[:3, 3] - camera_location)
    if translation[2] <= 3:
        raise ValueError("Core origin is too close to/behind the camera")
    center = project_camera(translation[None, :], intrinsics)[0] / size
    target = {
        "format": "direct_cube_projection_v2",
        "image_size": list(size),
        "coordinate_frame": "X right/red, Y back/blue, Z up/white; F green is -Y",
        "cube_side_length": 3.0,
        "rotation_matrix": core_rotation.tolist(),
        "rotation_6d_columns": np.concatenate((core_rotation[:, 0], core_rotation[:, 1])).tolist(),
        "rotation_9d_row_major": core_rotation.ravel().tolist(),
        "projection": {
            "center_xy_normalized": center.tolist(),
            "image_scale_normalized": float(intrinsics[0, 0] / size[0] / translation[2]),
            "perspective_strength": float(3 / translation[2]),
            "principal_point_normalized": (intrinsics[:2, 2] / size).tolist(),
            "formula": "principal + (center - principal + scale * (R @ X).xy) / (1 + perspective * (R @ X).z / 3)",
        },
        "layer_face_order": list(FACE_ORDER),
        "layer_angles_degrees": angles.tolist(),
        "layer_operations": operations,
        "corner_cut_policy": "dominant_angle_only",
        "color_frame_rebase_rotation": rebase.tolist(),
        "source_layer_state": record.get("layer_state", {}),
        "layer_angle_convention": "clockwise when viewed from outside the named color face; zero aligned",
        "layer_state_mode": record.get("layer_state", {}).get("mode", "none"),
        "reference_camera": {"intrinsics": intrinsics.tolist(),
                             "translation_in_cubie_units": translation.tolist(),
                             "sensor_width_mm": sensor_width_mm,
                             "used_at_inference": False},
    }
    landmarks = record["sticker_keypoints"]["landmark_keypoints"]
    if not landmarks:
        raise ValueError("At least one saved landmark is required for verification")
    world = np.asarray([point["world_point"] for point in landmarks])
    observed = np.asarray([point["pixel_point"] for point in landmarks])
    camera_points = (world - camera_location) @ camera_rotation.T
    reference_pixels = project_camera(camera_points, intrinsics)
    local = (world - transform[:3, 3]) @ core_to_world
    base, owners, classes, names = landmark_template()
    articulated = articulate(base, owners, angles, operations=operations)
    projected = project_target(articulated, target)
    _, reference_operations, _ = articulation_parameters(record, dominant_corner=False)
    reference_articulated = articulate(base, owners, angles, operations=reference_operations)
    reference_projected = project_target(reference_articulated, target)
    target["reference_layer_operations"] = reference_operations
    matched = []
    geometry_errors, pixel_errors, compact_errors = [], [], []
    reference_geometry_errors, reference_pixel_errors = [], []
    for observation, point, image_point in zip(landmarks, local, observed):
        candidates = np.flatnonzero(classes == int(observation["class_id"]))
        index = int(candidates[np.argmin(np.linalg.norm(reference_articulated[candidates] - point, axis=1))])
        geometry_errors.append(np.linalg.norm(articulated[index] - point))
        pixel_errors.append(np.linalg.norm(projected[index] - image_point))
        reference_geometry_errors.append(np.linalg.norm(reference_articulated[index] - point))
        reference_pixel_errors.append(np.linalg.norm(reference_projected[index] - image_point))
        compact_pixel = project_target(articulated[index:index + 1], target, compact=True)[0]
        compact_errors.append(np.linalg.norm(compact_pixel - image_point))
        matched.append({"anchor": names[index], "class_id": int(classes[index]),
                        "source_object": observation["source_object"],
                        "observed_pixel": image_point.tolist(),
                        "projected_pixel": projected[index].tolist(),
                        "visibility": int(observation["visibility"])})
    target["landmark_anchors"] = [
        {"name": name, "class_id": int(cls), "rest_point": rest.tolist(),
         "articulated_point": point.tolist(), "pixel_point": pixel.tolist()}
        for name, cls, rest, point, pixel in zip(names, classes, base, articulated, projected)
    ]
    virtual_names = (*CORNER_NAMES, *FACE_ORDER)
    virtual_points = np.asarray([
        sum((FACE_AXES[FACE_ORDER.index(face)] for face in name), np.zeros(3)) * 1.5
        for name in virtual_names
    ])
    virtual_pixels = project_target(virtual_points, target)
    target["virtual_core_anchors"] = [
        {"name": name, "core_point": point.tolist(), "pixel_point": pixel.tolist(),
         "in_frame": bool(np.all(pixel >= 0) and np.all(pixel < size)),
         "physical_visibility": None}
        for name, point, pixel in zip(virtual_names, virtual_points, virtual_pixels)
    ]
    target["matched_observations"] = matched
    target["sticker_observations"] = record["sticker_keypoints"]["keypoints"]
    target["verification"] = {
        "source_camera_error_px": error_summary(np.linalg.norm(reference_pixels - observed, axis=1)),
        "articulated_geometry_error_cubie_units": error_summary(geometry_errors),
        "target_projection_error_px": error_summary(pixel_errors),
        "old_compact_projection_error_px": error_summary(compact_errors),
        "full_source_geometry_error_cubie_units": error_summary(reference_geometry_errors),
        "full_source_projection_error_px": error_summary(reference_pixel_errors),
    }
    return target


def discover_samples(roots):
    samples, excluded, inventory = [], Counter(), []
    for root in roots:
        root = root.expanduser().resolve()
        metadata_paths = sorted((root / "metadata").glob("*.json"))
        if not metadata_paths:
            raise ValueError(f"No per-image JSON under {root / 'metadata'}; aggregate metadata is not supported")
        image_stems = {p.stem for p in (root / "images").iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
        metadata_stems = {p.stem for p in metadata_paths}
        inventory.append({"source": str(root), "images": len(image_stems),
                          "per_image_metadata": len(metadata_paths),
                          "images_without_per_image_metadata": len(image_stems - metadata_stems)})
        for path in metadata_paths:
            record = json.loads(path.read_text())
            mode = record.get("layer_state", {}).get("mode", "none")
            if mode not in SUPPORTED_MODES:
                excluded[f"unsupported_articulation:{mode}"] += 1
                continue
            try:
                angles = layer_angles(record)
            except (KeyError, ValueError, TypeError) as error:
                excluded[f"invalid_layer_metadata:{error}"] += 1
                continue
            image_path = next((root / "images" / (path.stem + ext) for ext in IMAGE_EXTENSIONS
                               if (root / "images" / (path.stem + ext)).is_file()), None)
            if image_path is None or not (root / "pose" / (path.stem + ".txt")).is_file():
                excluded["missing_image_or_yolo_label"] += 1
                continue
            samples.append(Sample(root, path, image_path, record, angles))
    return samples, dict(excluded), inventory


def select_samples(samples, limit, seed):
    groups = defaultdict(list)
    for sample in samples:
        faces = tuple(FACE_ORDER[i] for i in np.flatnonzero(np.abs(sample.angles) > 1e-8))
        source_moves = tuple(op.get("move", op.get("face")) for op in
                             sample.record.get("layer_state", {}).get("operations", []))
        slice_kind = source_moves if sample.mode == "slice_turn" else ()
        groups[(sample.mode, faces, slice_kind)].append(sample)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    queues = defaultdict(list)
    for key in sorted(groups):
        queues[key[0]].append(groups[key])
    selected = []
    while queues and len(selected) < limit:
        for mode in ("none", "mid_turn", "simultaneous_turn", "slice_turn", "corner_cut"):
            if mode not in queues:
                continue
            group = queues[mode].pop(0)
            selected.append(group.pop())
            if group:
                queues[mode].append(group)
            if not queues[mode]:
                del queues[mode]
            if len(selected) == limit:
                break
    return selected


def read_image(path):
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def save_image(path, image):
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise RuntimeError(f"Cannot encode image: {path}")
    path.write_bytes(encoded.tobytes())


def text(image, value, point, color=(240, 240, 240), scale=0.48):
    cv2.putText(image, value, point, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, value, point, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def render_overlay(image, target, sample):
    tile = 432
    factor = tile / image.shape[1]
    left = cv2.resize(image, (tile, tile), interpolation=cv2.INTER_NEAREST)
    right = left.copy()
    for observation in target["matched_observations"]:
        point = tuple(np.rint(np.asarray(observation["observed_pixel"]) * factor).astype(int))
        cv2.circle(left, point, 5, CLASS_BGR[observation["class_id"]], 1, cv2.LINE_AA)
    quads, normals = surface_quads(sample.angles, operations=target["layer_operations"])
    rot = np.asarray(target["rotation_matrix"])
    translation = np.asarray(target["reference_camera"]["translation_in_cubie_units"])
    camera_in_core = -rot.T @ translation
    visible = np.sum(normals * (camera_in_core - quads.mean(axis=1)), axis=1) > 0
    for quad in quads[visible]:
        pixels = np.rint(project_target(quad, target) * factor).astype(np.int32)
        cv2.polylines(right, [pixels], True, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.polylines(right, [pixels], True, (255, 200, 60), 1, cv2.LINE_AA)
    for observation in target["matched_observations"]:
        actual = tuple(np.rint(np.asarray(observation["observed_pixel"]) * factor).astype(int))
        predicted = tuple(np.rint(np.asarray(observation["projected_pixel"]) * factor).astype(int))
        if np.linalg.norm(np.asarray(actual) - predicted) > 1:
            cv2.line(right, actual, predicted, (0, 140, 255), 1, cv2.LINE_AA)
        cv2.circle(right, actual, 5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.drawMarker(right, predicted, (30, 255, 70), cv2.MARKER_CROSS, 7, 1, cv2.LINE_AA)
    for index, face in enumerate(FACE_ORDER):
        center = FACE_AXES[index] * 1.5
        if FACE_AXES[index] @ (camera_in_core - center) <= 0:
            continue
        point = tuple(np.rint(project_target(center[None], target)[0] * factor).astype(int))
        cv2.circle(right, point, 4, FACE_BGR[index], -1, cv2.LINE_AA)
        text(right, f"{face} {FACE_COLORS[index]}", (point[0] + 6, point[1]), FACE_BGR[index], .42)
    canvas = np.full((tile + 112, tile * 2, 3), 24, dtype=np.uint8)
    canvas[:tile, :tile] = left
    canvas[:tile, tile:] = right
    text(canvas, "Saved landmark observations", (10, 21))
    text(canvas, "Reconstructed geometry; ring=stored, +=projected", (tile + 8, 21), scale=.40)
    metrics = target["verification"]
    turns = " then ".join(f"{op['face']}={op['angle_degrees']:+.2f}"
                          for op in target["layer_operations"]) or "all faces aligned"
    if sample.mode == "slice_turn":
        move = sample.record["layer_state"].get("operations", [sample.record["layer_state"]])[0]
        turns = f"Source {move.get('move', move.get('face'))} slice; rebased offsets: {turns}"
    if sample.mode == "corner_cut":
        ignored = [op for op in target["reference_layer_operations"] if op not in target["layer_operations"]]
        turns += "; ignored " + ", ".join(f"{op['face']}={op['angle_degrees']:+.2f}" for op in ignored)
    text(canvas, f"{sample.key}  |  {sample.mode}  |  {sample.record.get('hard_case', 'none')}", (10, tile + 23), scale=.44)
    text(canvas, f"Clockwise offsets (deg): {turns}", (10, tile + 47), scale=.44)
    label = "Dominant-angle approximation" if sample.mode == "corner_cut" else "Projection"
    color = (80, 255, 100) if metrics["target_projection_error_px"]["max"] < .05 else (0, 190, 255)
    text(canvas, f"{label} max error: {metrics['target_projection_error_px']['max']:.6f} px; "
         f"3D max error: {metrics['articulated_geometry_error_cubie_units']['max']:.6f}", (10, tile + 72), color, .44)
    text(canvas, f"Compact projector error: {metrics['old_compact_projection_error_px']['max']:.3f} px; "
         f"full-source reconstruction: {metrics['full_source_projection_error_px']['max']:.5f} px", (10, tile + 96))
    return canvas


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, nargs="+", default=[Path("data/keypoints_only_dataset_part7")])
    parser.add_argument("--output-dir", type=Path, default=Path("runs/direct_pose_targets_all_types_verification"))
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--audit-all", action="store_true", help="Verify every eligible record; export only --limit samples.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--sensor-width-mm", type=float, default=36.0)
    parser.add_argument("--pixel-tolerance", type=float, default=.05)
    parser.add_argument("--geometry-tolerance", type=float, default=1e-3)
    return parser


def main():
    args = build_parser().parse_args()
    if args.limit < 1 or args.sensor_width_mm <= 0 or args.pixel_tolerance <= 0 or args.geometry_tolerance <= 0:
        raise SystemExit("Limit, sensor width and tolerances must be positive")
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output directory must be empty: {output}")
    samples, excluded, inventory = discover_samples(args.source_dir)
    selected = select_samples(samples, args.limit, args.seed)
    if not selected:
        raise SystemExit("No supported samples found")
    for directory in ("images", "pose", "targets", "overlays"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    report = {"format": "direct_cube_projection_verification_v2", "sources": inventory,
              "eligible_samples": len(samples), "excluded_samples": excluded,
              "seed": args.seed, "pixel_tolerance": args.pixel_tolerance,
              "geometry_tolerance": args.geometry_tolerance,
              "sensor_width_mm": args.sensor_width_mm,
              "corner_cut_policy": "dominant_angle_only", "samples": []}
    previews = {}
    selected_keys = {sample.key for sample in selected}
    for sample in (samples if args.audit_all else selected):
        target = build_target(sample.record, sensor_width_mm=args.sensor_width_mm)
        target["source"] = {"metadata": str(sample.metadata_path), "image": str(sample.image_path)}
        metrics = target["verification"]
        passed = (metrics["source_camera_error_px"]["max"] <= args.pixel_tolerance
                  and metrics["target_projection_error_px"]["max"] <= args.pixel_tolerance
                  and metrics["articulated_geometry_error_cubie_units"]["max"] <= args.geometry_tolerance)
        reference_passed = (metrics["source_camera_error_px"]["max"] <= args.pixel_tolerance
                            and metrics["full_source_projection_error_px"]["max"] <= args.pixel_tolerance
                            and metrics["full_source_geometry_error_cubie_units"]["max"] <= args.geometry_tolerance)
        target["verification"]["passed"] = passed
        target["verification"]["reference_passed"] = reference_passed
        if sample.key in selected_keys:
            (output / "targets" / f"{sample.key}.json").write_text(json.dumps(target, indent=2) + "\n")
            shutil.copy2(sample.image_path, output / "images" / (sample.key + sample.image_path.suffix))
            shutil.copy2(sample.root / "pose" / f"{sample.metadata_path.stem}.txt", output / "pose" / f"{sample.key}.txt")
            overlay = render_overlay(read_image(sample.image_path), target, sample)
            save_image(output / "overlays" / f"{sample.key}.png", overlay)
            previews[sample.key] = cv2.resize(overlay, (648, 408), interpolation=cv2.INTER_AREA)
        report["samples"].append({"key": sample.key, "mode": sample.mode,
                                   "layer_angles_degrees": sample.angles.tolist(),
                                   "exported": sample.key in selected_keys, **target["verification"]})
        if sample.key in selected_keys or not reference_passed:
            status = "PASS" if passed else ("APPROX" if reference_passed and sample.mode == "corner_cut" else "FAIL")
            print(f"{status} {sample.key} ({sample.mode}): "
                  f"{metrics['target_projection_error_px']['max']:.6f} px, "
                  f"full-source {metrics['full_source_projection_error_px']['max']:.6f} px")
    report["passed"] = all(sample["passed"] for sample in report["samples"])
    report["reference_passed"] = all(sample["reference_passed"] for sample in report["samples"])
    report["selected_modes"] = dict(Counter(sample.mode for sample in selected))
    report["audited_modes"] = dict(Counter(sample["mode"] for sample in report["samples"]))
    report["by_mode"] = {}
    for mode in sorted(SUPPORTED_MODES):
        records = [sample for sample in report["samples"] if sample["mode"] == mode]
        if not records:
            continue
        total = sum(sample["target_projection_error_px"]["count"] for sample in records)
        report["by_mode"][mode] = {
            "samples": len(records), "landmarks": total,
            "mean_projection_error_px": sum(sample["target_projection_error_px"]["mean"]
                                             * sample["target_projection_error_px"]["count"] for sample in records) / total,
            "worst_projection_error_px": max(sample["target_projection_error_px"]["max"] for sample in records),
            "worst_full_source_error_px": max(sample["full_source_projection_error_px"]["max"] for sample in records),
            "reference_failures": sum(not sample["reference_passed"] for sample in records),
            "projection_tolerance_failures": sum(not sample["passed"] for sample in records),
        }
    report["worst_projection_error_px"] = max(sample["target_projection_error_px"]["max"] for sample in report["samples"])
    report["worst_compact_projection_error_px"] = max(sample["old_compact_projection_error_px"]["max"] for sample in report["samples"])
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    # A short, readable gallery; every selected sample also has a full overlay.
    ordered_previews = [previews[sample.key] for sample in selected]
    count = min(10, len(ordered_previews))
    sheet = np.full((((count + 1) // 2) * 408, 1296, 3), 24, dtype=np.uint8)
    for index, preview in enumerate(ordered_previews[:count]):
        y, x = (index // 2) * 408, (index % 2) * 648
        sheet[y:y + 408, x:x + 648] = preview
    save_image(output / "contact_sheet.jpg", sheet)
    print(json.dumps(report["by_mode"], indent=2))
    print(f"Audited {len(report['samples'])} samples, exported {len(selected)}; report and overlays: {output}")
    if not report["reference_passed"] or any(not s["passed"] and s["mode"] != "corner_cut" for s in report["samples"]):
        raise SystemExit("Source/geometry verification failed; inspect report.json before using these targets")


if __name__ == "__main__":
    main()
