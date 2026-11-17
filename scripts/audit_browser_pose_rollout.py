"""Audit autonomous browser tracking against the saved Python geometry render.

Replay already-filtered detections, with no saved cameras fed to the browser
rollout. Warm up from the available clip prefix, sample by source time exactly
as FrameSelector does, and measure the solve window. The fixed-camera control
and isolated neighbor-pass experiment are explicitly separate from the app.
This compares with logged geometry option zero, not ground truth or move-search
selection. Missing saved observations are reported, not reconstructed.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_orientation_motion import REFERENCE_SOLUTIONS
from scripts.pose_assignment_comparison import compare_sticker_assignments
from scripts.benchmark_temporal_pose_reuse import saved_candidate
from scripts.benchmark_cpu_pose_port import hydrate
from scripts.export_cpu_pose_fixtures import pack
from scripts import render_yolo11n_independent_keypoints as renderer


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, allow_nan=False, separators=(",", ":")) + "\n")


def logged_fit(option):
    rotations = tuple(renderer.CubeLayerRotation(**r) for r in option["layer_rotations"])
    angles = renderer.parallel_layer_angles(rotations)
    return dict(axis=rotations[0].axis if rotations else 0, angles=list(angles),
                params=[*np.asarray(option["pose_rvec"]).ravel(),
                        *np.asarray(option["pose_tvec"]).ravel(), angles[0], angles[2]],
                nearest=option["cached_sticker_indices"], valid=option["cached_sticker_valid"])


def render_examples(output_dir, report):
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    examples = []
    for case in report["cases"]:
        worst = case["stages"]["browser"]["worst"][0]
        if worst["sticker_assignment_changes"]:
            examples.append((case, worst["frame"]))
    fig, axes = plt.subplots(len(examples), 3, figsize=(12, 3.5 * len(examples)), squeeze=False, layout="constrained")
    for index, (case, frame) in enumerate(examples):
        inputs = json.loads((output_dir / f"{case['name']}-inputs.json").read_text())
        item = next(w for w in inputs["workloads"] if w["frame"] == frame)
        rollout = json.loads((output_dir / f"{case['name']}-rollout.json").read_text())
        row = next(r for r in rollout["rows"] if r["frame"] == frame)
        base, _ = hydrate(item)
        meta = json.loads(Path(case["log"]).with_suffix(".meta.json").read_text())
        video = cv2.VideoCapture(meta["source"])
        video.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = video.read()
        video.release()
        assert ok, (case["name"], frame)
        assert (image.shape[1], image.shape[0]) == tuple(item["image_size"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        points = np.asarray(item["observations"]["points"] + item["observations"]["stickers"])
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        radius = max(np.ptp(points, axis=0)) * .7
        for column, label in enumerate(("Saved Python geometry", "Current browser rollout", "Isolated neighbor pass")):
            ax = axes[index, column]
            ax.imshow(image)
            ax.set_xlim(center[0] - radius, center[0] + radius)
            ax.set_ylim(center[1] + radius, center[1] - radius)
            ax.axis("off")
            fit = item["reference_fit"] if column == 0 else (row["browser" if column == 1 else "revised"]["bank"] or [None])[0]
            if fit:
                rotations = renderer.parallel_layer_rotations(fit["axis"], fit["angles"])
                candidate = replace(base, rvec=np.asarray(fit["params"][:3]).reshape(3, 1),
                    tvec=np.asarray(fit["params"][3:6]).reshape(3, 1), layer_rotations=rotations,
                    sticker_layer_rotations=rotations)
                for _, polygon, _ in renderer.projected_cubelet_faces(candidate, candidate.rvec, candidate.tvec):
                    polygon = np.r_[polygon, polygon[:1]]
                    ax.plot(polygon[:, 0], polygon[:, 1], color=("#26efc0" if column == 0 else "#ffa638"), linewidth=1)
                comparison = compare_sticker_assignments(inputs["geometry"], item["reference_fit"], fit)
                label += f"\n{comparison['sticker_assignment_changes']} differing sticker assignments"
            else:
                label += "\nNo fit"
            stickers = np.asarray(item["observations"]["stickers"])
            ax.scatter(stickers[:, 0], stickers[:, 1], s=8, c="white", edgecolors="black", linewidths=.4)
            ax.set_title(f"{case['name'].title()} · frame {frame} · {label}", fontsize=10)
    fig.suptitle("Identical saved detections; dots are observed sticker keypoints", fontsize=13)
    fig.savefig(output_dir / "examples.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=list(REFERENCE_SOLUTIONS))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "pose-fitter/results/browser-rollout-audit")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    geometry = json.loads((ROOT / "pose-fitter/tests/fixtures/cpu-golden.json").read_text())["geometry"]
    report = dict(scope=__doc__, selection="FrameSelector source-time buckets, 30 fps cap, clip origin frame 0",
        assignment_comparison="Sticker-face IDs aligned by the closest global cube symmetry, including assigned/unassigned status; not raw IDs or a pixel-distance gate",
        limitations=["Reference is logged geometry option zero, not final move-search selection or ground truth",
            "Replay uses saved filtered detections, not fresh detector inference or ROI-reset events",
            "Unavailable saved observations are skipped and reported",
            "Neighbor experiment is isolated and not connected to the app",
            "Fixed-camera control changes the seed only; it is not a Python/Rust solver parity test"],
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            ROOT / "pose-fitter/src/browser.rs", ROOT / "pose-fitter/src/search.rs",
            ROOT / "web/src/lib/pose.worker.ts", ROOT / "web/src/lib/frame-stream.ts"]}, cases=[])
    for name in args.cases:
        solution = json.loads((ROOT / REFERENCE_SOLUTIONS[name]).read_text())
        log = Path(solution["config"]["log"])
        meta = json.loads(log.with_suffix(".meta.json").read_text())
        fps = float(meta.get("source_fps", meta["fps"]))
        start = meta.get("turning_start_frame") or 0
        end = solution.get("solved_frame_index") or max(s["frame_index"] for s in solution["steps"])
        records = {r["frame_index"]: r for r in map(json.loads, log.open())}
        corpus = dict(geometry=geometry, workloads=[], probes=[], sweep_probes=[])
        unavailable, last_bucket = [], -1
        for frame in sorted(records):
            if frame > end:
                continue
            bucket = int(np.floor(frame / fps * 30 + .0001))
            if fps > 30.001 and bucket <= last_bucket:
                continue
            last_bucket = bucket
            record = records[frame]
            candidate = saved_candidate(record)
            if candidate is None:
                unavailable.append(frame)
                continue
            item = pack(name, frame, "initial_geometry", candidate, None, (), ())
            item["reference_fit"] = logged_fit(record["search"]["geometry"]["options"][0])
            corpus["workloads"].append(item)
        fixture_path = args.output_dir / f"{name}-inputs.json"
        rollout_path = args.output_dir / f"{name}-rollout.json"
        write(fixture_path, corpus)
        if not args.reuse:
            print(name, "replaying", len(corpus["workloads"]), "prefix + solve frames", flush=True)
            subprocess.run(["node", str(ROOT / "pose-fitter/tools/audit-browser-rollout.mjs"),
                            str(fixture_path), str(rollout_path), "1"], check=True, cwd=ROOT)
        data = json.loads(rollout_path.read_text())
        assert data["fixture_sha256"] == hashlib.sha256(fixture_path.read_bytes()).hexdigest()
        inputs = {w["frame"]: w for w in corpus["workloads"]}
        case = dict(name=name, log=str(log), log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
                    source_fps=fps, start=start, end=end,
                    available_prefix_frames=sum(w["frame"] < start for w in corpus["workloads"]),
                    unavailable_saved_observation_frames=unavailable, wasm_sha256=data["wasm_sha256"],
                    fixture_sha256=data["fixture_sha256"], stages={})
        for stage in ("browser", "revised", "fixed"):
            rows = []
            for row in data["rows"]:
                if row["frame"] < start:
                    continue
                expected = inputs[row["frame"]]["reference_fit"]
                bank = row[stage] if stage == "fixed" else row[stage]["bank"]
                comparisons = [compare_sticker_assignments(geometry, expected, f) for f in bank]
                count = len(expected["nearest"])
                first = comparisons[0] if comparisons else dict(sticker_assignment_changes=count, sticker_validity_changes=count)
                rows.append(dict(frame=row["frame"], sticker_count=count, missing_fit=not bank,
                                 matching_assignment_in_bank=any(c["sticker_assignment_changes"] == 0 for c in comparisons),
                                 **first))
            case["stages"][stage] = dict(frames=len(rows), missing_fits=sum(r["missing_fit"] for r in rows),
                differing_frames=sum(r["sticker_assignment_changes"] > 0 for r in rows),
                differing_keypoints=sum(r["sticker_assignment_changes"] for r in rows),
                assignment_absent_from_bank=sum(not r["matching_assignment_in_bank"] for r in rows),
                worst=sorted(rows, key=lambda r: -r["sticker_assignment_changes"])[:10], rows=rows)
        report["cases"].append(case)
        write(args.output_dir / "report.json", report)
        print(name, {k: {a: b for a, b in v.items() if a not in ("worst", "rows")} for k, v in case["stages"].items()}, flush=True)
    print(args.output_dir / "report.json")
    if args.render:
        render_examples(args.output_dir, report)


if __name__ == "__main__":
    main()
