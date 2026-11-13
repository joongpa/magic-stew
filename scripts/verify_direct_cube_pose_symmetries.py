#!/usr/bin/env python3
"""Verify 24/4 pose-and-turn alternatives on existing, unmodified images.

    .venv/bin/python scripts/verify_direct_cube_pose_symmetries.py

The default examples were visually checked at projected center positions:
001978 has its centers covered; 001192 retains a partially visible white
center (visibility 1). Both visibility 1 and 2 count as orientation evidence.
Existing exporter angle and coverage filtering is retained. Missing center
labels alone are NOT used to establish image ambiguity: the older exporter
can omit stickers through incorrect surface selection.
"""

from __future__ import annotations

import argparse
import copy
from itertools import permutations, product
import json
from pathlib import Path
import re
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import prepare_direct_cube_pose_targets as geometry  # noqa: E402


def cube_rotations(visible_faces=()):
    if len(visible_faces) == 1:
        axis = geometry.FACE_AXES[geometry.FACE_ORDER.index(visible_faces[0])]
        return [np.rint(geometry.rotation(axis, angle)).astype(int) for angle in (0, 90, 180, 270)]
    matrices = []
    for order in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            matrix = np.zeros((3, 3), dtype=int)
            matrix[np.arange(3), order] = signs
            if round(np.linalg.det(matrix)) != 1:
                continue
            if any(not np.array_equal(matrix @ geometry.FACE_AXES[geometry.FACE_ORDER.index(face)],
                                      geometry.FACE_AXES[geometry.FACE_ORDER.index(face)]) for face in visible_faces):
                continue
            matrices.append(matrix)
    return sorted(matrices, key=lambda matrix: (not np.array_equal(matrix, np.eye(3)), tuple(matrix.ravel())))


def face_permutation(matrix):
    """Q maps candidate coordinates to reference coordinates; axes map by Q.T."""
    return {face: geometry.FACE_ORDER[int(np.argmax(geometry.FACE_AXES @ (matrix.T @ axis)))]
            for face, axis in zip(geometry.FACE_ORDER, geometry.FACE_AXES)}


def permute_points(points, matrix):
    changed = np.asarray(points) @ matrix
    distances = np.linalg.norm(changed[:, None] - np.asarray(points)[None], axis=2)
    indexes = distances.argmin(axis=1)
    if np.max(distances[np.arange(len(points)), indexes]) > 1e-9 or len(set(indexes)) != len(points):
        raise ValueError("Symmetry did not produce a one-to-one anchor permutation")
    return indexes


def transform_target(reference, matrix):
    candidate = copy.deepcopy(reference)
    mapping = face_permutation(matrix)
    rotation = np.asarray(reference["rotation_matrix"]) @ matrix
    candidate["rotation_matrix"] = rotation.tolist()
    candidate["rotation_6d_columns"] = np.r_[rotation[:, 0], rotation[:, 1]].tolist()
    candidate["rotation_9d_row_major"] = rotation.ravel().tolist()
    candidate["color_frame_rebase_rotation"] = (np.asarray(reference["color_frame_rebase_rotation"]) @ matrix).tolist()
    angles = np.zeros(6)
    for face, angle in zip(geometry.FACE_ORDER, reference["layer_angles_degrees"]):
        angles[geometry.FACE_ORDER.index(mapping[face])] = angle
    candidate["layer_angles_degrees"] = angles.tolist()
    for field in ("layer_operations", "reference_layer_operations"):
        candidate[field] = [{**op, "face": mapping[op["face"]]} for op in reference[field]]
    # Rebuild all cached named keypoints so their identities change together
    # with the pose/turn channels. The physical source observations stay fixed.
    base, owners, classes, names = geometry.landmark_template()
    permutation = permute_points(base, matrix)
    articulated = geometry.articulate(base, owners, angles, operations=candidate["layer_operations"])
    pixels = geometry.project_target(articulated, candidate)
    candidate["landmark_anchors"] = [
        {"name": name, "class_id": int(cls), "rest_point": rest.tolist(),
         "articulated_point": point.tolist(), "pixel_point": pixel.tolist()}
        for name, cls, rest, point, pixel in zip(names, classes, base, articulated, pixels)
    ]
    old_indexes = {name: index for index, name in enumerate(names)}
    for observation in candidate["matched_observations"]:
        index = int(permutation[old_indexes[observation["anchor"]]])
        observation["anchor"] = names[index]
        observation["projected_pixel"] = pixels[index].tolist()
    for anchor in candidate["virtual_core_anchors"]:
        pixel = geometry.project_target(np.asarray(anchor["core_point"])[None], candidate)[0]
        anchor["pixel_point"] = pixel.tolist()
        anchor["in_frame"] = bool(np.all(pixel >= 0) and np.all(pixel < candidate["image_size"]))
    candidate["symmetry_candidate_to_reference"] = matrix.tolist()
    candidate["reference_face_to_candidate_face"] = mapping
    return candidate


def verify_candidate(reference, candidate, matrix, visible_faces=()):
    base, owners, _, _ = geometry.landmark_template()
    indexes = permute_points(base, matrix)
    errors = []
    local_errors = []
    for operation_field in ("layer_operations", "reference_layer_operations"):
        original = geometry.articulate(base, owners, reference["layer_angles_degrees"], operations=reference[operation_field])
        changed = geometry.articulate(base, owners, candidate["layer_angles_degrees"], operations=candidate[operation_field])
        local_errors.extend(np.linalg.norm(changed[indexes] - original @ matrix, axis=1))
        errors.extend(np.linalg.norm(geometry.project_target(changed[indexes], candidate)
                                     - geometry.project_target(original, reference), axis=1))
    virtual = np.asarray([point["core_point"] for point in reference["virtual_core_anchors"]])
    virtual_indexes = permute_points(virtual, matrix)
    errors.extend(np.linalg.norm(geometry.project_target(virtual[virtual_indexes], candidate)
                                 - geometry.project_target(virtual, reference), axis=1))
    constrained_errors = []
    for face in visible_faces:
        point = geometry.FACE_AXES[geometry.FACE_ORDER.index(face)][None] * 1.5
        constrained_errors.append(float(np.linalg.norm(geometry.project_target(point, candidate)
                                                       - geometry.project_target(point, reference))))
    return {"max_projection_change_px": float(max(errors)),
            "max_articulation_change_cubie_units": float(max(local_errors)),
            "max_visible_center_change_px": max(constrained_errors, default=0.0)}


def retained_center_labels(record, center_objects):
    result = []
    for point in record["sticker_keypoints"]["keypoints"]:
        owner = point.get("source_cubie")
        if owner is None:
            match = re.search(r"(cubie(?:\.\d{3})?)(?=[.:]|$)", point["source_object"])
            owner = match.group(1) if match else None
        if owner in center_objects and point["visibility"] in (1, 2):
            result.append(point)
    return result


def draw_candidate(image, target, index, visible_faces, tile):
    factor = tile / image.shape[1]
    picture = cv2.resize(image, (tile, tile), interpolation=cv2.INTER_NEAREST)
    angles = target["layer_angles_degrees"]
    quads, normals = geometry.surface_quads(angles, operations=target["layer_operations"])
    rest_quads, _ = geometry.surface_quads(np.zeros(6))
    rotation = np.asarray(target["rotation_matrix"])
    camera = -rotation.T @ np.asarray(target["reference_camera"]["translation_in_cubie_units"])
    for quad, normal, rest_quad in zip(quads, normals, rest_quads):
        if normal @ (camera - quad.mean(axis=0)) <= 0:
            continue
        color = (220, 220, 220)
        owner = np.clip(np.rint(rest_quad.mean(axis=0)), -1, 1)
        for face_index in np.argsort(np.abs(angles)):
            if abs(angles[face_index]) > 1e-8 and owner @ geometry.FACE_AXES[face_index] > .5:
                color = geometry.FACE_BGR[face_index]
        pixels = np.rint(geometry.project_target(quad, target) * factor).astype(np.int32)
        cv2.polylines(picture, [pixels], True, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.polylines(picture, [pixels], True, color, 1, cv2.LINE_AA)
    for face, axis, color, name in zip(geometry.FACE_ORDER, geometry.FACE_AXES, geometry.FACE_BGR, geometry.FACE_COLORS):
        if axis @ (camera - axis * 1.5) <= 0:
            continue
        pixel = tuple(np.rint(geometry.project_target((axis * 1.5)[None], target)[0] * factor).astype(int))
        cv2.circle(picture, pixel, 4, color, -1, cv2.LINE_AA)
        geometry.text(picture, face + (" fixed" if face in visible_faces else "?"),
                      (pixel[0] + 5, pixel[1]), color, .42)
    panel = np.full((tile + 78, tile, 3), 24, np.uint8)
    panel[:tile] = picture
    geometry.text(panel, f"Candidate {index:02d}", (8, tile + 20), scale=.48)
    turns = ", ".join(f"{face}{angle:+.2f}" for face, angle in zip(geometry.FACE_ORDER, angles) if abs(angle) > 1e-8)
    geometry.text(panel, turns or "aligned", (8, tile + 43), scale=.43)
    geometry.text(panel, "? = hypothetical face identity", (8, tile + 65), scale=.34)
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, default=Path("data/keypoints_only_dataset_part7"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/direct_pose_symmetry_verification"))
    parser.add_argument("--zero-stem", default="001978")
    parser.add_argument("--one-stem", default="001192")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    center_objects = set()
    for path in sorted((args.source_dir / "metadata").glob("*.json")):
        record = json.loads(path.read_text())
        center_objects.update(point["source_object"] for point in record["sticker_keypoints"]["landmark_keypoints"]
                              if point.get("source_piece_type") == "center")
    if len(center_objects) != 6:
        raise SystemExit(f"Could not identify exactly six center cubies: {center_objects}")
    report = {"visibility_rule": "retained center-sticker labels with visibility 1 OR 2; source angle and coverage filters unchanged",
              "selection_note": "Default source images also visually inspected at projected center locations; absence of labels alone is insufficient",
              "center_objects": sorted(center_objects), "cases": []}
    for stem, expected in ((args.zero_stem, 0), (args.one_stem, 1)):
        path = args.source_dir / "metadata" / f"{stem}.json"
        record = json.loads(path.read_text())
        observations = retained_center_labels(record, center_objects)
        if len(observations) != expected:
            raise SystemExit(f"{stem}: expected {expected} retained centers, found {len(observations)}")
        faces = [geometry.FACE_ORDER[geometry.FACE_COLORS.index(point["class"])] for point in observations]
        reference = geometry.build_target(record)
        image = geometry.read_image(args.source_dir / "images" / f"{stem}.png")
        case = {"stem": stem, "retained_centers": observations, "visible_faces": faces, "candidates": []}
        panels = []
        directory = output / ("zero_centers" if expected == 0 else "one_center")
        directory.mkdir()
        for index, matrix in enumerate(cube_rotations(faces)):
            candidate = transform_target(reference, matrix)
            metrics = verify_candidate(reference, candidate, matrix, faces)
            if max(metrics.values()) > 1e-8:
                raise RuntimeError(f"Symmetry failed for {stem}/{index}: {metrics}")
            case["candidates"].append({"index": index, "face_mapping": face_permutation(matrix),
                                       "angles": dict(zip(geometry.FACE_ORDER, candidate["layer_angles_degrees"])), **metrics})
            (directory / f"candidate_{index:02d}.json").write_text(json.dumps(candidate, indent=2) + "\n")
            panels.append(draw_candidate(image, candidate, index, faces, 360 if expected else 260))
        cols = 2 if expected else 4
        height, width = panels[0].shape[:2]
        sheet = np.full((((len(panels) + cols - 1) // cols) * height + 54, cols * width, 3), 24, np.uint8)
        title = f"{stem}: {len(faces)} retained center(s); {len(panels)} alternatives"
        geometry.text(sheet, title, (9, 23), scale=.55)
        geometry.text(sheet, "Unmodified dataset image; colored wire/labels describe each hypothetical orientation", (9, 45), scale=.35)
        for index, panel in enumerate(panels):
            y, x = 54 + (index // cols) * height, (index % cols) * width
            sheet[y:y + height, x:x + width] = panel
        geometry.save_image(directory / "candidates.jpg", sheet)
        annotated = cv2.resize(image, (576, 576), interpolation=cv2.INTER_NEAREST)
        for point in observations:
            pixel = tuple(np.rint(np.asarray(point["pixel_point"]) * 4).astype(int))
            cv2.circle(annotated, pixel, 8, (255, 255, 255), 2, cv2.LINE_AA)
            geometry.text(annotated, f"{point['class']} v={point['visibility']} area={point['visible_area_ratio']:.2f}",
                          (10, 28), scale=.55)
        geometry.save_image(directory / "source.jpg", annotated)
        report["cases"].append(case)
        print(stem, f"{len(panels)} candidates", "retained", [(p["class"], p["visibility"]) for p in observations],
              "max projection change", max(item["max_projection_change_px"] for item in case["candidates"]))
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Saved", output)


if __name__ == "__main__":
    main()
