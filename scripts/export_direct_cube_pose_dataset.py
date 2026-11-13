#!/usr/bin/env python3
"""Export all metadata-backed images with six-center pose and articulation targets.

    .venv/bin/python scripts/export_direct_cube_pose_dataset.py

Sources are unchanged. Per-file metadata takes precedence over an aggregate
metadata.json; complete aggregate records are streamed, including when a run
left a truncated final record. Images without a complete record are omitted.
The original nine independent classes remain in pose/*.txt. targets/*.json
adds six ordered, rigid face-center anchors, rotation/projection parameters,
six outer-face offsets, physical landmark correspondences, and allowed cube
symmetries. Corner cuts retain the previously verified dominant-angle policy.

Center evidence comes from existing visibility 1 AND 2 labels, preserving the
exporter's angle/coverage filtering. Old missing-label bugs mean these symmetry
sets are conservative alternatives, not proof of actual image ambiguity.
Geometric center targets are supplied even when hidden; missing sticker labels
are never converted into fabricated visibility or presence targets.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import prepare_direct_cube_pose_targets as geometry  # noqa: E402
from scripts import verify_direct_cube_pose_symmetries as symmetry  # noqa: E402

DEFAULT_SOURCES = [Path("data/keypoints_only_dataset"), *(
    Path(f"data/keypoints_only_dataset_part{part}") for part in (3, 5, 7)
)]


def stream_array(path, diagnostics, chunk_size=1024 * 1024):
    """Read one object at a time, reporting an incomplete final JSON record."""
    decoder = json.JSONDecoder()
    diagnostics.update(complete_records=0, truncated_tail=False)
    with path.open(encoding="utf-8") as handle:
        buffer = ""
        eof = False

        def fill():
            nonlocal buffer, eof
            chunk = handle.read(chunk_size)
            buffer += chunk
            eof = not chunk

        def whitespace():
            nonlocal buffer
            buffer = buffer.lstrip()
            while not buffer and not eof:
                fill()
                buffer = buffer.lstrip()

        whitespace()
        if not buffer.startswith("["):
            raise ValueError(f"Expected a JSON array: {path}")
        buffer = buffer[1:]
        after_comma = False
        while True:
            whitespace()
            if buffer.startswith("]"):
                if after_comma:
                    raise ValueError(f"Trailing comma in {path}")
                if (buffer[1:] + handle.read()).strip():
                    raise ValueError(f"Unexpected content after JSON array: {path}")
                return
            if not buffer and eof:
                diagnostics["truncated_tail"] = True
                return
            while True:
                try:
                    record, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    if eof:
                        diagnostics["truncated_tail"] = True
                        return
                    fill()
            if not isinstance(record, dict):
                raise ValueError(f"Metadata entry is not an object: {path}")
            diagnostics["complete_records"] += 1
            yield record
            buffer = buffer[end:]
            whitespace()
            if buffer.startswith(","):
                buffer = buffer[1:]
                after_comma = True
            elif buffer.startswith("]"):
                after_comma = False
            elif not buffer and eof:
                diagnostics["truncated_tail"] = True
                return
            else:
                raise ValueError(f"Missing metadata record separator: {path}")


def metadata_records(root, diagnostics):
    """Yield (stem, record, provenance), preferring newer per-file records."""
    paths = sorted((root / "metadata").glob("*.json"))
    per_file_stems = {path.stem for path in paths}
    diagnostics["per_file_records"] = len(paths)
    for path in paths:
        record = json.loads(path.read_text())
        if Path(record["image"]).stem != path.stem:
            raise ValueError(f"Image/metadata stem mismatch: {path}")
        yield path.stem, record, {"path": str(path), "storage": "per_file"}
    aggregate = root / "metadata.json"
    if aggregate.is_file():
        state = diagnostics.setdefault("aggregate", {})
        seen = set()
        for index, record in enumerate(stream_array(aggregate, state)):
            stem = Path(record["image"]).stem
            if stem in seen:
                raise ValueError(f"Duplicate metadata image {stem}: {aggregate}")
            seen.add(stem)
            if stem not in per_file_stems:
                yield stem, record, {"path": str(aggregate), "storage": "aggregate", "record_index": index}


def inventory_source(root):
    images = {path.stem: path for path in sorted((root / "images").iterdir())
              if path.suffix.lower() in geometry.IMAGE_EXTENSIONS}
    metadata_stems, centers = set(), set()
    modes, diagnostics = Counter(), {}
    missing_labels, no_landmarks = [], []
    for stem, record, _ in metadata_records(root, diagnostics):
        metadata_stems.add(stem)
        if stem not in images:
            continue
        modes[record.get("layer_state", {}).get("mode", "none")] += 1
        if not (root / "pose" / f"{stem}.txt").is_file():
            missing_labels.append(stem)
        landmarks = record.get("sticker_keypoints", {}).get("landmark_keypoints", [])
        if not landmarks:
            no_landmarks.append(stem)
        centers.update(point["source_object"] for point in landmarks
                       if point.get("source_piece_type") == "center")
    report = {"source": str(root), "images": len(images), "metadata_records": len(metadata_stems),
              "images_without_metadata": sorted(set(images) - metadata_stems),
              "metadata_without_images": sorted(metadata_stems - set(images)),
              "missing_pose_labels": missing_labels, "records_without_landmarks": no_landmarks,
              "modes": dict(modes), "center_objects": sorted(centers), "metadata_storage": diagnostics}
    return images, report


def symmetry_manifest():
    base, _, _, names = geometry.landmark_template()
    centers = geometry.FACE_AXES * 1.5
    candidates = []
    for index, matrix in enumerate(symmetry.cube_rotations()):
        candidates.append({"id": index, "candidate_to_reference": matrix.tolist(),
                           "reference_face_to_candidate_face": symmetry.face_permutation(matrix),
                           "reference_to_candidate_landmark_index": symmetry.permute_points(base, matrix).tolist(),
                           "reference_to_candidate_center_index": symmetry.permute_points(centers, matrix).tolist()})
    return {"format": "direct_cube_pose_symmetries_v1", "face_order": list(geometry.FACE_ORDER),
            "center_color_order": list(geometry.FACE_COLORS), "landmark_names": names,
            "convention": "Q maps candidate to reference; R_candidate=R_reference@Q; X_candidate=Q.T@X_reference; remap face channels without changing clockwise angle signs",
            "loss_policy": "Choose ONE allowed candidate jointly for pose, ordered centers, landmarks, and turn channels; never minimize each output independently",
            "candidates": candidates}


def add_center_targets(target, record, center_objects, manifest):
    observations = symmetry.retained_center_labels(record, center_objects)
    by_color = {point["class"]: point for point in observations}
    faces = [face for face, color in zip(geometry.FACE_ORDER, geometry.FACE_COLORS) if color in by_color]
    target["format"] = "direct_cube_centers_projection_v3"
    # These are geometric targets, not another independent detection class.
    centers = {point["name"]: point for point in target["virtual_core_anchors"]}
    target["virtual_core_anchors"] = []
    for face, color in zip(geometry.FACE_ORDER, geometry.FACE_COLORS):
        anchor = centers[face]
        observation = by_color.get(color)
        anchor.update(color=color,
                      normalized_point=(np.asarray(anchor["pixel_point"]) / target["image_size"]).tolist(),
                      source_label_visibility=observation["visibility"] if observation else None,
                      source_observed_pixel=observation["pixel_point"] if observation else None)
        target["virtual_core_anchors"].append(anchor)
    target["pose_anchor_order"] = list(geometry.FACE_ORDER)
    target["pose_anchor_color_order"] = list(geometry.FACE_COLORS)
    allowed = []
    for item in manifest["candidates"]:
        matrix = np.asarray(item["candidate_to_reference"])
        if all(np.array_equal(matrix @ geometry.FACE_AXES[geometry.FACE_ORDER.index(face)],
                              geometry.FACE_AXES[geometry.FACE_ORDER.index(face)]) for face in faces):
            allowed.append(item["id"])
    target["symmetry"] = {"manifest": "../symmetries.json", "allowed_ids": allowed,
                          "retained_center_faces": faces, "retained_center_count": len(faces),
                          "evidence": "existing center-sticker labels, visibility 1 or 2; original angle/coverage filtering preserved",
                          "missing_label_caveat": "Allowed IDs are conservative: historical sticker omissions can hide orientation evidence; count is not verified image visibility"}
    return target


def check_verification(target, pixel_tolerance=.05, geometry_tolerance=1e-3):
    metrics = target["verification"]
    if not metrics["source_camera_error_px"]["count"]:
        metrics.update(reference_passed=None, passed=None, status="unavailable_no_saved_landmarks")
        return True  # Metadata still fully specifies geometry; no empirical check is claimed.
    source_pass = (metrics["source_camera_error_px"]["max"] <= pixel_tolerance
                   and metrics["full_source_projection_error_px"]["max"] <= pixel_tolerance
                   and metrics["full_source_geometry_error_cubie_units"]["max"] <= geometry_tolerance)
    target_pass = (metrics["target_projection_error_px"]["max"] <= pixel_tolerance
                   and metrics["articulated_geometry_error_cubie_units"]["max"] <= geometry_tolerance)
    metrics.update(reference_passed=source_pass, passed=target_pass)
    metrics["status"] = "verified" if source_pass else "failed"
    return source_pass and (target_pass or target["layer_state_mode"] == "corner_cut")


def build_export_target(record, centers, manifest):
    saved = record["sticker_keypoints"]
    has_evidence = any(point["visibility"] in (1, 2)
                       for field in ("landmark_keypoints", "keypoints") for point in saved[field])
    if not has_evidence:
        # Empty scenes and fully hidden cubes provide no pose evidence. Retain
        # the image/empty independent labels, but never fabricate pose targets
        # (one historical record even has its nominal core behind the camera).
        return {"format": "direct_cube_centers_projection_v3",
                "image_size": [record["image_size"]["width"], record["image_size"]["height"]],
                "pose_supervision": {"enabled": False, "reason": "no_retained_sticker_or_landmark_observations"},
                "pose_anchor_order": list(geometry.FACE_ORDER), "pose_anchor_color_order": list(geometry.FACE_COLORS),
                "rotation_matrix": None, "projection": None, "layer_angles_degrees": None,
                "virtual_core_anchors": [], "landmark_anchors": [], "matched_observations": [],
                "sticker_observations": saved["keypoints"], "source_layer_state": record.get("layer_state", {}),
                "layer_state_mode": record.get("layer_state", {}).get("mode", "none"),
                "symmetry": {"manifest": "../symmetries.json", "allowed_ids": list(range(24)),
                             "retained_center_faces": [], "retained_center_count": 0},
                "verification": {**{name: geometry.error_summary([]) for name in (
                    "source_camera_error_px", "full_source_projection_error_px", "full_source_geometry_error_cubie_units",
                    "target_projection_error_px", "articulated_geometry_error_cubie_units", "old_compact_projection_error_px")},
                    "reference_passed": None, "passed": None, "status": "pose_supervision_masked"}}
    target = geometry.build_target(record, allow_unobserved=True)
    if not check_verification(target):
        raise ValueError(f"Source geometry verification failed: {target['verification']}")
    target = add_center_targets(target, record, centers, manifest)
    target["pose_supervision"] = {"enabled": True}
    return target


def export_dataset(roots, output, inventory_only=False):
    roots = [root.expanduser().resolve() for root in roots]
    if len({root.name for root in roots}) != len(roots):
        raise ValueError("Source directory names must be unique")
    inventories = [inventory_source(root) for root in roots]
    if inventory_only:
        print(json.dumps([report for _, report in inventories], indent=2))
        return
    output = output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    for directory in ("images", "pose", "targets"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    manifest = symmetry_manifest()
    (output / "symmetries.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = {"format": "direct_cube_centers_dataset_v1", "sources": [], "exported": 0,
              "pose_anchor_order": list(geometry.FACE_ORDER), "pose_anchor_color_order": list(geometry.FACE_COLORS),
              "corner_cut_policy": "dominant_angle_only", "failures": [],
              "source_files_modified": False}
    with (output / "manifest.jsonl").open("w") as index:
        for root, (images, source_report) in zip(roots, inventories):
            centers = set(source_report["center_objects"])
            if len(centers) != 6:
                raise ValueError(f"Expected six center owners under {root}, found {centers}")
            source_report.update(exported=0, exported_modes={}, symmetry_counts={}, worst_full_source_error_px=0.0,
                                 worst_target_error_px=0.0, without_landmark_verification=0, pose_supervision_masked=0, failures=0)
            mode_counts, symmetry_counts = Counter(), Counter()
            for stem, record, provenance in metadata_records(root, {}):
                if stem not in images:
                    continue
                key = f"{root.name}__{stem}"
                if not (root / "pose" / f"{stem}.txt").is_file():
                    report["failures"].append({"key": key, "reason": "missing_pose_label"})
                    source_report["failures"] += 1
                    continue
                try:
                    target = build_export_target(record, centers, manifest)
                    if geometry.read_image(images[stem]).shape[:2] != tuple(reversed(target["image_size"])):
                        raise ValueError("Image dimensions disagree with metadata")
                except (KeyError, ValueError, TypeError) as error:
                    report["failures"].append({"key": key, "reason": str(error)})
                    source_report["failures"] += 1
                    continue
                target["source"] = {"image": str(images[stem]), "pose_label": str(root / "pose" / f"{stem}.txt"),
                                    "metadata": provenance}
                image_name = key + images[stem].suffix
                (output / "targets" / f"{key}.json").write_text(json.dumps(target, separators=(",", ":"), allow_nan=False) + "\n")
                shutil.copy2(images[stem], output / "images" / image_name)
                shutil.copy2(root / "pose" / f"{stem}.txt", output / "pose" / f"{key}.txt")
                index.write(json.dumps({"key": key, "image": f"images/{image_name}", "pose": f"pose/{key}.txt",
                                        "target": f"targets/{key}.json", "source_dataset": root.name,
                                        "mode": target["layer_state_mode"], "pose_supervision": target["pose_supervision"]["enabled"],
                                        "symmetry_count": len(target["symmetry"]["allowed_ids"])}) + "\n")
                source_report["exported"] += 1
                report["exported"] += 1
                mode_counts[target["layer_state_mode"]] += 1
                symmetry_counts[len(target["symmetry"]["allowed_ids"])] += 1
                if target["verification"]["reference_passed"] is None:
                    source_report["without_landmark_verification"] += 1
                if not target["pose_supervision"]["enabled"]:
                    source_report["pose_supervision_masked"] += 1
                for field, metric in (("worst_full_source_error_px", "full_source_projection_error_px"),
                                      ("worst_target_error_px", "target_projection_error_px")):
                    value = target["verification"][metric]["max"]
                    if value is not None:
                        source_report[field] = max(source_report[field], value)
                if source_report["exported"] % 500 == 0:
                    print(f"{root.name}: {source_report['exported']} exported", flush=True)
            source_report["exported_modes"] = dict(mode_counts)
            source_report["symmetry_counts"] = dict(symmetry_counts)
            report["sources"].append(source_report)
            print(f"{root.name}: {source_report['exported']} exported; {len(source_report['images_without_metadata'])} without metadata; {source_report['failures']} failures", flush=True)
    report["complete"] = not report["failures"]
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "README.txt").write_text(
        "Six-center direct cube pose targets\n\n"
        "manifest.jsonl lists exactly the included images and their original independent pose labels.\n"
        "Check pose_supervision.enabled FIRST: empty/fully hidden scenes remain included, with pose fields null/empty.\n"
        "No object confidence target is inferred from that pose mask. No train/val split is imposed.\n"
        "Each enabled targets/*.json has six virtual_core_anchors ordered U,D,F,B,R,L\n"
        "(white,yellow,green,blue,red,orange), rotation/projection parameters and six face offsets.\n"
        "Centers supervise rigid pose; retained sticker/landmark labels and landmark_anchors supervise articulation.\n"
        "Hidden center coordinates are geometric targets; physical_visibility remains null when unknown.\n"
        "source_label_visibility null means no retained label, not known absence. Values 1 and 2 both supply evidence.\n"
        "symmetry.allowed_ids references symmetries.json. Pick one candidate jointly for all pose/turn/anchor losses.\n"
        "Old sticker-label omissions make candidate sets conservative, not verified image ambiguity.\n"
        "Corner cuts use only the larger angle; reference_layer_operations preserves the full source geometry.\n"
        "Metadata-backed images without saved landmarks remain included; their verification status is unavailable.\n"
        "report.json records omissions, provenance, verification errors and counts. Source files are unchanged.\n"
    )
    print(f"Exported {report['exported']} images to {output}", flush=True)
    if report["failures"]:
        raise ValueError(f"{len(report['failures'])} metadata-backed images failed validation; see report.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, nargs="+", default=DEFAULT_SOURCES)
    parser.add_argument("--output-dir", type=Path, default=Path("data/direct_cube_pose_dataset"))
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    export_dataset(args.source_dir, args.output_dir, args.inventory_only)


if __name__ == "__main__":
    main()
