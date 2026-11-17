#!/usr/bin/env python3
"""Copy direct_cube_pose_dataset to v2, changing only sticker class IDs.

Reads per-image targets/*.json. Every original row is retained, including
byte-identical landmark rows. Sticker centers, boxes, visibility and row order
are unchanged. Physical cubie types come from metadata identities, never pose
projection or nearest geometric matching. Missing metadata omits an image;
unresolvable row/metadata inconsistencies fail rather than silently drop labels.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from live_recognition.typed_sticker_keypoints import (
    CLASS_NAMES,
    COLORS,
    PIECE_TYPES,
    manifest,
    sticker_class_id,
    sticker_owner,
)


def owner_types(source):
    groups = {}
    for path in sorted((source / "targets").glob("*.json")):
        target = json.loads(path.read_text(encoding="utf-8"))
        mapping = groups.setdefault(path.stem.split("__")[0], {})
        for point in target["matched_observations"]:
            owner, kind = point["source_object"], PIECE_TYPES[int(point["class_id"])]
            if mapping.setdefault(owner, kind) != kind:
                raise ValueError(f"Conflicting cubie types for {owner} in {path}")
    return groups


def relabel_rows(text, target, types):
    """Return text and updated sticker metadata, preserving every row's tail."""
    lines = text.splitlines(keepends=True)
    metadata = target["sticker_observations"]
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        values = np.asarray([float(x) for x in line.split()])
        if len(values) != 8 or not np.isfinite(values).all():
            raise ValueError(f"Expected finite, single-point 9-class row: {line!r}")
        cls = int(values[0])
        if values[0] != cls or not 0 <= cls < 9:
            raise ValueError(f"Expected original class ID 0..8: {line!r}")
        if cls >= 3:
            rows.append((index, cls - 3, values[5:7], int(values[7])))
    if len(rows) != len(metadata):
        raise ValueError(f"Sticker row/metadata count differs: {len(rows)} vs {len(metadata)}")
    updated = [dict(sticker) for sticker in metadata]
    if rows:
        size = np.asarray(target["image_size"], float)
        observed = np.asarray([s["pixel_point"] for s in metadata]) / size
        row_points = np.asarray([r[2] for r in rows])
        distances = np.linalg.norm(row_points[:, None] - observed, axis=-1)
        cost = distances.copy()
        for i, (_, color, _, visibility) in enumerate(rows):
            for j, sticker in enumerate(metadata):
                if sticker["class"] != COLORS[color] or int(sticker["visibility"]) != visibility:
                    cost[i, j] = 1e6
        indices, matches = linear_sum_assignment(cost)
        for i, j in zip(indices, matches):
            if cost[i, j] > 2e-6:
                raise ValueError(
                    f"Sticker row has no exact metadata match (normalized error={cost[i, j]})"
                )
            line_index, color, _, _ = rows[i]
            sticker = metadata[j]
            kind = sticker.get("sticker_type") or types.get(sticker_owner(sticker))
            if kind not in PIECE_TYPES:
                raise ValueError(f"No physical piece type for {sticker['source_object']}")
            new_id = sticker_class_id(COLORS[color], kind)
            line = lines[line_index]
            stripped = line.lstrip()
            old_token = stripped.split(None, 1)[0]
            leading = line[: len(line) - len(stripped)]
            lines[line_index] = leading + str(new_id) + stripped[len(old_token) :]
            updated[j].update(
                class_id=new_id,
                source_class_id=int(sticker["class_id"]),
                color=COLORS[color],
                sticker_type=kind,
                **{"class": CLASS_NAMES[new_id]},
            )
    result = "".join(lines)
    assert len(result.splitlines()) == len(text.splitlines())
    return result, updated


def link_or_copy(source, dest):
    try:
        os.link(source, dest)
    except OSError:
        shutil.copy2(source, dest)


def convert(source, output, limit=None):
    source, output = source.resolve(), output.resolve()
    if not (source / "targets").is_dir():
        raise ValueError("Source must contain per-image targets/*.json")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be new or empty")
    for folder in ("images", "pose", "targets"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    types = owner_types(source)
    counts, classes, modes = Counter(), Counter(), Counter()
    source_entries = {}
    if (source / "manifest.jsonl").is_file():
        source_entries = {
            e["key"]: e
            for e in map(json.loads, (source / "manifest.jsonl").read_text().splitlines())
        }
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as records:
        for image in sorted((source / "images").iterdir()):
            if image.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            if limit is not None and counts["images"] >= limit:
                break
            key = image.stem
            target_path, label_path = (
                source / "targets" / f"{key}.json",
                source / "pose" / f"{key}.txt",
            )
            if not target_path.is_file():
                counts["omitted_without_metadata"] += 1
                continue
            if not label_path.is_file():
                raise ValueError(f"Missing source label: {label_path}")
            target = json.loads(target_path.read_text(encoding="utf-8"))
            with label_path.open("r", encoding="utf-8", newline="") as f:
                before = f.read()
            after, stickers = relabel_rows(before, target, types[key.split("__")[0]])
            for a, b in zip(before.splitlines(), after.splitlines()):
                if not a.strip():
                    assert a == b
                    continue
                old_class = int(a.split()[0])
                new_class = int(b.split()[0])
                assert a.split()[1:] == b.split()[1:]
                if old_class < 3:
                    assert a == b
                    counts["unchanged_landmark_rows"] += 1
                else:
                    counts["reclassified_sticker_rows"] += 1
                classes[new_class] += 1
            counts["images"] += 1
            modes[target["layer_state_mode"]] += 1
            link_or_copy(image, output / "images" / image.name)
            with (output / "pose" / f"{key}.txt").open("w", encoding="utf-8", newline="") as f:
                f.write(after)
            target.update(
                source_target_format=target["format"],
                format="direct_cube_typed_stickers_v1",
                class_names=list(CLASS_NAMES),
                keypoint_shape=[1, 3],
                sticker_observations=stickers,
                label_conversion={
                    "source_pose": str(label_path),
                    "policy": "sticker class IDs only; all rows and numeric fields preserved",
                },
            )
            (output / "targets" / f"{key}.json").write_text(
                json.dumps(target) + "\n", encoding="utf-8"
            )
            entry = {
                **source_entries.get(key, {}),
                "key": key,
                "image": f"images/{image.name}",
                "pose": f"pose/{key}.txt",
                "target": f"targets/{key}.json",
            }
            records.write(json.dumps(entry) + "\n")
            if counts["images"] % 1000 == 0:
                print(f"Converted {counts['images']} images (row counts preserved)", flush=True)
    for name in ("symmetries.json",):
        if (source / name).exists():
            shutil.copy2(source / name, output / name)
    (output / "sticker_keypoint_manifest.json").write_text(
        json.dumps(manifest(), indent=2) + "\n", encoding="utf-8"
    )
    import yaml

    (output / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(output),
                "train": "images",
                "val": "images",
                "pose": "pose",
                "kpt_shape": [1, 3],
                "names": list(CLASS_NAMES),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    report = {
        **counts,
        "class_objects": {name: classes[i] for i, name in enumerate(CLASS_NAMES)},
        "modes": dict(modes),
        "source": str(source),
        "label_count_preserved": True,
        "landmark_rows_byte_identical": True,
        "coordinates_boxes_visibility_preserved": True,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data/direct_cube_pose_dataset")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/direct_cube_pose_dataset_v2"
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    convert(args.source_dir, args.output_dir, args.limit)


if __name__ == "__main__":
    main()
