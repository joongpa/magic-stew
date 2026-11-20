"""Measure the cost/geometry tradeoff of skipping low-error temporal refits.

This is a counterfactual on the saved paired CPU fit_frame workloads, not a
tracker/move-search rollout. The CPU outputs determine gates and geometry
changes; repeated WASM replays supply fitting-only timing estimates. Normal
search/solver behavior is unchanged and no gate is enabled in the app.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_cpu_pose_port import Server, compare  # noqa: E402


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def diameter(geometry, workload, fit):
    camera, x = workload["camera"], np.asarray(fit["params"], dtype=float)
    image, _ = cv2.projectPoints(
        np.asarray(geometry["corners"], dtype=np.float32),
        x[:3],
        x[3:6],
        np.asarray(camera["k"]),
        np.asarray(camera["dist"]),
    )
    return 2 * cv2.minEnclosingCircle(image.reshape(-1, 2).astype(np.float32))[1]


from scripts.pose_assignment_comparison import compare_sticker_assignments


def sticker_assignment_features(geometry, workload, initial, final, cube_diameter):
    first, valid_first = np.asarray(initial["nearest"]), np.asarray(initial["valid"])
    comparison = compare_sticker_assignments(geometry, initial, final)

    # A cheap, pre-refit feature: distance of each assigned sticker observation
    # from the boundary of its own projected quad, normalized like reprojection.
    polygons = np.asarray(geometry["faces"]).copy()
    layers = np.asarray(geometry["axes"][initial["axis"]]["face_layers"])
    for layer, angle in enumerate(initial["angles"]):
        rotation = cv2.Rodrigues(np.eye(3)[initial["axis"]] * np.deg2rad(angle))[0]
        polygons[layers == layer] = polygons[layers == layer] @ rotation.T
    x, camera = np.asarray(initial["params"]), workload["camera"]
    projected = cv2.projectPoints(
        polygons.reshape(-1, 3), x[:3], x[3:6], np.asarray(camera["k"]), np.asarray(camera["dist"])
    )[0]
    projected = projected.reshape(-1, 4, 2).astype(np.float32)
    margins = [
        cv2.pointPolygonTest(projected[face], tuple(point), True)
        for face, point, valid in zip(first, workload["observations"]["stickers"], valid_first)
        if valid
    ]
    return dict(
        sticker_count=len(first),
        initial_all_stickers_valid=bool(valid_first.all()),
        initial_min_sticker_edge_margin=min(margins, default=0.0) / cube_diameter,
        **comparison,
    )


def summarize(rows, threshold, guard=None):
    skipped = [
        r
        for r in rows
        if (threshold is None or r["initial_error"] <= threshold) and (guard is None or guard(r))
    ]
    saved = sum(r["wasm_temporal_ms"] for r in skipped)
    total = sum(r["wasm_initial_ms"] + r["wasm_temporal_ms"] for r in rows)
    before = sum(r["cpu_initial_ms"] + r["cpu_temporal_ms"] for r in rows)
    worst = sorted(skipped, key=lambda r: r["point_max_px"], reverse=True)
    return dict(
        threshold=threshold,
        threshold_percent_diameter=100 * threshold if threshold is not None else None,
        frames=len(rows),
        skipped=len(skipped),
        skipped_fraction=len(skipped) / len(rows),
        wasm_baseline_s=total / 1000,
        wasm_saved_s=saved / 1000,
        wasm_after_s=(total - saved) / 1000,
        wasm_saved_fraction=saved / total,
        wasm_baseline_ms_per_frame=total / len(rows),
        wasm_after_ms_per_frame=(total - saved) / len(rows),
        cpu_saved_fraction=sum(r["cpu_temporal_ms"] for r in skipped) / before,
        skipped_changed_winner=sum(not r["winner_unchanged"] for r in skipped),
        skipped_changed_bank=sum(not r["bank_unchanged"] for r in skipped),
        skipped_changed_sticker_assignments=sum(
            r["sticker_assignment_changes"] > 0 for r in skipped
        ),
        skipped_changed_sticker_assignments_raw=sum(
            r["sticker_assignment_raw_changes"] > 0 for r in skipped
        ),
        changed_sticker_keypoints=sum(r["sticker_assignment_changes"] for r in skipped),
        changed_sticker_keypoints_raw=sum(r["sticker_assignment_raw_changes"] for r in skipped),
        tested_sticker_keypoints=sum(r["sticker_count"] for r in skipped),
        changed_sticker_validity=sum(r["sticker_validity_changes"] for r in skipped),
        sticker_assignment_change_examples=[
            {
                k: r[k]
                for k in ("name", "frame", "sticker_assignment_changes", "sticker_validity_changes")
            }
            for r in skipped
            if r["sticker_assignment_changes"] > 0
        ],
        skipped_move_over_0_1px=sum(r["point_max_px"] > 0.1 for r in skipped),
        skipped_move_over_1px=sum(r["point_max_px"] > 1 for r in skipped),
        largest_point_change_px=max((r["point_max_px"] for r in skipped), default=0.0),
        largest_score_gain=max((r["score_gain"] for r in skipped), default=0.0),
        worst_skipped_frames=[
            {
                k: r[k]
                for k in (
                    "name",
                    "frame",
                    "initial_error",
                    "initial_error_px",
                    "point_max_px",
                    "point_rms_px",
                    "score_gain",
                )
            }
            for r in worst[:5]
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=ROOT / "pose-fitter/results/sequence-parity"
    )
    parser.add_argument(
        "--timings", type=Path, default=ROOT / "pose-fitter/results/full-wasm-timings.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "pose-fitter/benchmarks/refit-gate.json"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reuse-timings", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    names = ("matty", "yiheng", "tommy")
    wasm_path = ROOT / "pose-fitter/target/wasm32-unknown-unknown/release/cube_pose_fitter.wasm"
    wasm_hash = hashlib.sha256(wasm_path.read_bytes()).hexdigest()
    timing = (
        json.loads(args.timings.read_text())
        if args.reuse_timings
        else dict(
            scope="Warmed Node WASM fitting only; JSON input/output excluded",
            wasm_sha256=wasm_hash,
            repeats=args.repeats,
            cases={},
        )
    )
    assert timing["wasm_sha256"] == wasm_hash
    rows = []
    for name in names:
        path = args.input_dir / f"{name}-fixtures.json"
        fixture_bytes = path.read_bytes()
        corpus = json.loads(fixture_bytes)
        workloads = corpus["workloads"]
        fixture_hash = hashlib.sha256(fixture_bytes).hexdigest()
        if not args.reuse_timings:
            server = Server(["node", "pose-fitter/tools/benchmark-wasm.mjs", str(path), "--serve"])
            try:
                # Prime both initial and temporal paths before measured passes.
                for stage in ("initial_geometry", "temporal_neighbors"):
                    indices = [i for i, w in enumerate(workloads) if w["stage"] == stage]
                    for i in indices[:3]:
                        server.run(i)
                times = [[] for _ in workloads]
                for pass_index in range(args.repeats):
                    indices = (
                        range(len(workloads))
                        if pass_index % 2 == 0
                        else range(len(workloads) - 1, -1, -1)
                    )
                    for count, i in enumerate(indices):
                        result = server.run(i)
                        times[i].append(result["ms"])
                        if count % 240 == 0:
                            print(
                                name,
                                "timing pass",
                                pass_index + 1,
                                "call",
                                count + 1,
                                "/",
                                len(workloads),
                                flush=True,
                            )
                timing["cases"][name] = dict(
                    fixture_sha256=fixture_hash,
                    runtime=server.metadata,
                    calls=[
                        dict(
                            frame=w["frame"],
                            stage=w["stage"],
                            times_ms=t,
                            median_ms=statistics.median(t),
                        )
                        for w, t in zip(workloads, times)
                    ],
                )
                dump(args.timings, timing)
            finally:
                server.close()
        case_timing = timing["cases"][name]
        assert case_timing["fixture_sha256"] == fixture_hash
        assert len(case_timing["calls"]) == len(workloads)
        paired = {}
        for w, t in zip(workloads, case_timing["calls"]):
            assert (w["frame"], w["stage"]) == (t["frame"], t["stage"])
            paired.setdefault(w["frame"], {})[w["stage"]] = (w, t["median_ms"])
        for frame, pair in paired.items():
            initial, initial_ms = pair["initial_geometry"]
            temporal, temporal_ms = pair["temporal_neighbors"]
            a, b = initial["expected"], temporal["expected"]
            first, last = a[0], b[0]
            consensus = [
                compare_sticker_assignments(corpus["geometry"], first, alternative)
                for alternative in a
            ]
            cube_diameter = diameter(corpus["geometry"], initial, first)
            distances = np.linalg.norm(
                np.asarray(first["assignment_projected"]) - last["assignment_projected"], axis=1
            )
            rows.append(
                dict(
                    name=name,
                    frame=frame,
                    initial_error=first["metrics"]["keypoint_error"],
                    initial_error_px=first["metrics"]["keypoint_error"] * cube_diameter,
                    initial_p90=first["metrics"]["p90_error"],
                    sticker_error=first["metrics"]["sticker_error"],
                    landmark_count=len(initial["observations"]["points"]),
                    score_gap=a[1]["score"] - first["score"] if len(a) > 1 else None,
                    initial_score=first["score"],
                    final_score=last["score"],
                    score_gain=first["score"] - last["score"],
                    winner_unchanged=compare(a[:1], b[:1])["passed"],
                    bank_unchanged=compare(a, b)["passed"],
                    point_max_px=float(distances.max()),
                    point_rms_px=float(np.sqrt(np.mean(distances**2))),
                    cpu_initial_ms=initial["cpu_ms"],
                    cpu_temporal_ms=temporal["cpu_ms"],
                    wasm_initial_ms=initial_ms,
                    wasm_temporal_ms=temporal_ms,
                    initial_top2_sticker_agreement=all(
                        c["sticker_assignment_changes"] == 0 for c in consensus[:2]
                    ),
                    initial_top2_sticker_agreement_raw=all(
                        c["sticker_assignment_raw_changes"] == 0 for c in consensus[:2]
                    ),
                    initial_top3_sticker_agreement=all(
                        c["sticker_assignment_changes"] == 0 for c in consensus[:3]
                    ),
                    initial_bank_sticker_agreement=all(
                        c["sticker_assignment_changes"] == 0 for c in consensus
                    ),
                    **sticker_assignment_features(
                        corpus["geometry"], initial, first, last, cube_diameter
                    ),
                )
            )
    cutoffs = (0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02)
    report = dict(
        scope=__doc__,
        metric="Initial rank-0 mean landmark error divided by rigid projected cube diameter.",
        wasm_sha256=wasm_hash,
        timing_source=str(args.timings.relative_to(ROOT)),
        repeats=timing["repeats"],
        frames=len(rows),
        aggregate=[summarize(rows, t) for t in cutoffs],
        cases={
            name: [summarize([r for r in rows if r["name"] == name], t) for t in cutoffs]
            for name in names
        },
        assignment_guard=dict(
            definition="Initial mean landmark error below cutoff, every sticker assigned, and minimum sticker-to-quad-edge margin >= 2 * landmark p90 error (all normalized by cube diameter).",
            aggregate=[
                summarize(
                    rows,
                    t,
                    guard=lambda r: (
                        r["initial_all_stickers_valid"]
                        and r["initial_min_sticker_edge_margin"] >= 2 * r["initial_p90"]
                    ),
                )
                for t in cutoffs
            ],
        ),
        assignment_consensus={
            key: [summarize(rows, t, guard=lambda r, key=key: r[key]) for t in cutoffs]
            for key in (
                "initial_top2_sticker_agreement",
                "initial_top2_sticker_agreement_raw",
                "initial_top3_sticker_agreement",
                "initial_bank_sticker_agreement",
            )
        },
        candidate_policies={
            "low_error_and_top2_agreement": summarize(
                rows, 0.01, guard=lambda r: r["initial_top2_sticker_agreement"]
            ),
            "unanimous_initial_bank": summarize(
                rows, None, guard=lambda r: r["initial_bank_sticker_agreement"]
            ),
        },
        notes=[
            "Gate evaluated after the full initial fit; kept frames retain the complete original bank.",
            "Saving is the sum of measured costs of omitted temporal calls; gated execution is modeled, not rerun.",
            "WASM measurements are Node/V8, not browser or full-app FPS.",
            "Geometry changes are differences from the CPU temporal result, not ground-truth accuracy loss.",
            "Saved workloads use fixed cached cameras and retained banks, not an autonomous sequence rollout.",
            "Sticker assignments compare each observed sticker's face ID and valid/unassigned status; arbitrary nearest indices for two unassigned observations are ignored.",
            "Common-frame counts map refined face IDs through the closest cube symmetry to the initial rigid rotation; raw ID counts are also retained.",
            "The extra assignment margin guard is exploratory on these same runs, not validated on held-out sequences.",
        ],
    )
    dump(args.output, report)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for entry in report["aggregate"]:
        print(
            entry["threshold"],
            "skip",
            entry["skipped"],
            "saved %",
            round(100 * entry["wasm_saved_fraction"], 2),
            "ms/frame",
            round(entry["wasm_after_ms_per_frame"], 2),
            "changed sticker assignments",
            entry["skipped_changed_sticker_assignments"],
            flush=True,
        )
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
