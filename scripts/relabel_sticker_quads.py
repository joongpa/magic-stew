#!/usr/bin/env python3
"""Create an 18-class four-corner dataset from direct-pose per-image targets.

Sources remain unchanged. Reads targets/*.json, not legacy monolithic metadata.
Retains only originally labeled stickers (including visibility 1). Reconstructs
exact reference articulation, including both corner-cut operations. Corner
targets cover all four in-frame vertices, including turned-layer seams.
Saved landmark visibility flags are preserved. Other in-frame corners use
conservative visibility 1 (amodal coordinates; hand visibility unverified).
Out-of-frame corners use 0. Original sticker angle/coverage selection remains
authoritative; this does not introduce previously excluded sticker objects.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from live_recognition.sticker_keypoint_order import clockwise_sticker_order
from live_recognition.sticker_quad_schema import class_id, manifest, surface_template, yolo_row
from scripts.prepare_direct_cube_pose_targets import articulate, project_target

TEMPLATE = surface_template()
CORNER_TARGET_POLICY = "amodal_sticker_corners_v2"
VISIBILITY_POLICY = (
    "all four in-frame corners supervised; saved landmark visibility preserved; "
    "reconstructed corners use 1 with unverified hand visibility; outside image uses 0; "
    "original sticker angle/coverage selection preserved"
)


def corner_targets(pixels, roles, observations, size, tolerance):
    """Full quad targets: missing old landmarks do not erase seam vertices.

    Flag 1 conservatively labels reconstructed coordinates without claiming
    the point is visibly observed. YOLO supervises XY for both 1 and 2.
    This is not a new per-point hand-occlusion measurement.
    """
    observed = np.asarray([o["observed_pixel"] for o in observations], float).reshape(-1, 2)
    observed_classes = np.asarray([o["class_id"] for o in observations], int)
    corners = []
    for pixel, role in zip(pixels, roles):
        matched = None
        if np.all(pixel >= 0) and np.all(pixel < size):
            visibility, evidence = 1, "amodal_reconstruction"
            if len(observed):
                distances = np.linalg.norm(observed - pixel, axis=-1)
                distances[observed_classes != role] = np.inf
                nearest = int(distances.argmin())
                if distances[nearest] <= tolerance:
                    matched = observations[nearest]
                    visibility, evidence = int(matched["visibility"]), "saved_landmark"
        else:
            visibility, evidence = 0, "outside_image"
        corners.append(
            {
                "pixel_point": pixel.tolist(),
                "visibility": visibility,
                "landmark_class_id": int(role),
                "matched_landmark": matched,
                "visibility_evidence": evidence,
            }
        )
    return corners


def sticker_owner(sticker):
    if sticker.get("source_cubie"):
        return sticker["source_cubie"]
    match = re.search(r"cubie(?:\.\d+)?", sticker["source_object"])
    return match.group() if match else sticker["source_object"].split(":")[0]


def infer_owner_types(source):
    """Physical piece type is invariant even when the cube is scrambled."""
    groups = {}
    for path in sorted((source / "targets").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        mapping = groups.setdefault(path.stem.split("__")[0], {})
        for observation in payload["matched_observations"]:
            owner = observation["source_object"]
            kind = ("corner", "edge", "center")[int(observation["class_id"])]
            previous = mapping.setdefault(owner, kind)
            if previous != kind:
                raise ValueError(f"Inconsistent physical piece type for {owner} in {path}")
    return groups


def relabel_target(target, tolerance_at_128=0.75, owner_types=None):
    stickers = [s for s in target["sticker_observations"] if s["visibility"] in (1, 2)]
    if not stickers:
        # The direct export intentionally includes negative images whose
        # pose/camera fields are null because no observations survived.
        return [], {"retained_stickers": 0, "unmatched_stickers": 0}
    quads, owners, normals, roles, kinds = TEMPLATE
    angles = target["layer_angles_degrees"]
    operations = target.get("reference_layer_operations")
    if operations is None:
        raise ValueError("Missing exact reference_layer_operations")
    size = np.asarray(target["image_size"], float)
    tolerance = tolerance_at_128 * min(size) / 128
    points = articulate(
        quads.reshape(-1, 3), np.repeat(owners, 4, axis=0), angles, operations=operations
    )
    pixels = project_target(points, target).reshape(-1, 4, 2)
    centers = articulate(quads.mean(1), owners, angles, operations=operations)
    projected_centers = project_target(centers, target)
    turned_normals = articulate(normals, owners, angles, operations=operations)
    camera_centers = centers @ np.asarray(target["rotation_matrix"]).T
    camera_centers += np.asarray(target["reference_camera"]["translation_in_cubie_units"])
    camera_normals = turned_normals @ np.asarray(target["rotation_matrix"]).T
    facing = np.sum(camera_centers * camera_normals, axis=-1) < 0
    candidates = np.flatnonzero(facing)
    observations = target["matched_observations"]
    cost = np.linalg.norm(
        np.asarray([s["pixel_point"] for s in stickers])[:, None] - projected_centers[candidates],
        axis=-1,
    )
    geometric_cost = cost.copy()
    known_types = dict(owner_types or {})
    known_positions = {}
    anchors = {a["name"]: a for a in target["landmark_anchors"]}
    for observation in observations:
        owner = observation["source_object"]
        known_types[owner] = ("corner", "edge", "center")[int(observation["class_id"])]
        rest = np.asarray(anchors[observation["anchor"]]["rest_point"])
        position = tuple(np.where(abs(rest) > 1, np.sign(rest), 0).astype(int))
        known_positions.setdefault(owner, set()).add(position)
    for row, sticker in enumerate(stickers):
        owner = sticker_owner(sticker)
        if owner in known_types:
            cost[row, [kinds[i] != known_types[owner] for i in candidates]] = 1e6
        positions = known_positions.get(owner, set())
        if len(positions) == 1:
            cost[row, np.any(owners[candidates] != next(iter(positions)), axis=-1)] = 1e6
    rows, cols = linear_sum_assignment(cost)
    matches = {
        int(row): int(candidates[col])
        for row, col in zip(rows, cols)
        if cost[row, col] <= tolerance
    }
    result = []
    unmatched = []
    for index, sticker in enumerate(stickers):
        if index not in matches:
            unmatched.append(
                {
                    "source_object": sticker["source_object"],
                    "color": sticker["class"],
                    "visibility": sticker["visibility"],
                    "nearest_front_face_center_error_px": float(geometric_cost[index].min()),
                    "known_piece_type": known_types.get(sticker_owner(sticker)),
                    "tolerance_px": float(tolerance),
                }
            )
            continue
        face = matches[index]
        q, r, kind = pixels[face], roles[face], kinds[face]
        order = clockwise_sticker_order(q, r, kind)
        q, r = q[order], r[order]
        corners = corner_targets(q, r, observations, size, tolerance)
        color = sticker.get("color", sticker["class"]).lower()
        label_class = class_id(color, kind)
        result.append(
            {
                "class_id": label_class,
                "color": color,
                "sticker_type": kind,
                "canonical_owner": owners[face].tolist(),
                "source_sticker": sticker,
                "corners": corners,
                "center_error_px": float(
                    np.linalg.norm(projected_centers[face] - sticker["pixel_point"])
                ),
                "row": yolo_row(label_class, q, [c["visibility"] for c in corners], size),
            }
        )
    return result, {
        "retained_stickers": len(stickers),
        "unmatched_stickers": len(stickers) - len(result),
        "unmatched_details": unmatched,
    }


def link_or_copy(source, dest):
    try:
        os.link(source, dest)
    except OSError:
        shutil.copy2(source, dest)


def preview(image, objects, output):
    canvas = cv2.imread(str(image))
    canvas = cv2.resize(canvas, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    colors = {"corner": (30, 130, 245), "edge": (255, 180, 30), "center": (230, 80, 230)}
    for obj in objects:
        q = np.asarray([c["pixel_point"] for c in obj["corners"]]) * 3
        cv2.polylines(
            canvas, [q.round().astype(np.int32)], True, colors[obj["sticker_type"]], 1, cv2.LINE_AA
        )
        for i, (xy, corner) in enumerate(zip(q, obj["corners"])):
            if corner["visibility"]:
                xy = tuple(xy.round().astype(int))
                cv2.circle(canvas, xy, 3, colors[obj["sticker_type"]], -1)
                cv2.putText(
                    canvas, str(i), xy, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA
                )
                cv2.putText(
                    canvas,
                    str(i),
                    xy,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
    cv2.imwrite(str(output), canvas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path, nargs="+", default=[ROOT / "data/direct_cube_pose_dataset"]
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/sticker_quad_dataset")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--preview-count", type=int, default=12)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Use a new or empty output directory; existing datasets are never overwritten")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if any(not (s / "targets").is_dir() for s in args.source_dir):
        parser.error("Every source must contain direct-pose per-image targets/*.json")
    for subdir in ("images", "pose", "metadata", "previews"):
        (out / subdir).mkdir(parents=True, exist_ok=True)
    counts, classes, visibility, evidence = Counter(), Counter(), Counter(), Counter()
    errors, seen = [], set()
    with (
        (out / "manifest.jsonl").open("w", encoding="utf-8") as manifest_file,
        (out / "unmatched_stickers.jsonl").open("w", encoding="utf-8") as unmatched_file,
    ):
        for source in args.source_dir:
            owner_types = infer_owner_types(source)
            for image in sorted((source / "images").iterdir()):
                if image.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                    continue
                if args.limit is not None and counts["images"] >= args.limit:
                    break
                target_path = source / "targets" / f"{image.stem}.json"
                if not target_path.is_file():
                    counts["images_without_metadata"] += 1
                    continue
                if image.stem in seen:
                    raise ValueError(f"Duplicate image key across sources: {image.stem}")
                target = json.loads(target_path.read_text(encoding="utf-8"))
                objects, diagnostics = relabel_target(
                    target, owner_types=owner_types.get(image.stem.split("__")[0])
                )
                counts.update({k: v for k, v in diagnostics.items() if k != "unmatched_details"})
                for detail in diagnostics.get("unmatched_details", []):
                    unmatched_file.write(
                        json.dumps(
                            {"key": image.stem, "mode": target["layer_state_mode"], **detail}
                        )
                        + "\n"
                    )
                key = image.stem
                seen.add(key)
                link_or_copy(image, out / "images" / image.name)
                (out / "pose" / f"{key}.txt").write_text(
                    "\n".join(o["row"] for o in objects) + ("\n" if objects else ""),
                    encoding="utf-8",
                )
                (out / "metadata" / f"{key}.json").write_text(
                    json.dumps(
                        {
                            **manifest(),
                            "source_target": str(target_path.resolve()),
                            "image_size": target["image_size"],
                            "layer_state_mode": target["layer_state_mode"],
                            "objects": [
                                {k: v for k, v in o.items() if k != "row"} for o in objects
                            ],
                            "corner_target_policy": CORNER_TARGET_POLICY,
                            "visibility_policy": VISIBILITY_POLICY,
                            **diagnostics,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                manifest_file.write(
                    json.dumps(
                        {
                            "key": key,
                            "image": f"images/{image.name}",
                            "pose": f"pose/{key}.txt",
                            "metadata": f"metadata/{key}.json",
                            "source_target": str(target_path.resolve()),
                        }
                    )
                    + "\n"
                )
                for o in objects:
                    classes[o["class_id"]] += 1
                    visibility.update(c["visibility"] for c in o["corners"])
                    evidence.update(c["visibility_evidence"] for c in o["corners"])
                    errors.append(o["center_error_px"])
                if counts["images"] < args.preview_count:
                    preview(image, objects, out / "previews" / f"{key}.png")
                counts["images"] += 1
                if counts["images"] % 1000 == 0:
                    print(f"Relabeled {counts['images']} images", flush=True)
    (out / "sticker_keypoint_manifest.json").write_text(
        json.dumps(
            {
                **manifest(),
                "corner_target_policy": CORNER_TARGET_POLICY,
                "visibility_policy": VISIBILITY_POLICY,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    import yaml

    (out / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(out),
                "train": "images",
                "val": "images",
                "pose": "pose",
                "kpt_shape": [4, 3],
                "names": manifest()["class_names"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    report = {
        **counts,
        "classes": {manifest()["class_names"][k]: v for k, v in sorted(classes.items())},
        "corner_visibility": dict(visibility),
        "corner_visibility_evidence": dict(evidence),
        "corner_target_policy": CORNER_TARGET_POLICY,
        "center_matching_max_px": max(errors, default=None),
        "sources": [str(p.resolve()) for p in args.source_dir],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Dataset: {out}")


if __name__ == "__main__":
    main()
