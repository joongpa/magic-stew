"""Compare every eligible saved-frame geometry call for complete solve windows.

Both ports receive the CPU's exact saved-camera inputs and retained banks. This
isolates numerical/search parity; it is not an autonomous tracker rollout or a
test of the orientation/move-search pipeline, which has not been ported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import independent_keypoint_pose_geometry as fitter  # noqa: E402
from scripts import render_yolo11n_independent_keypoints as renderer  # noqa: E402
from scripts.analyze_orientation_motion import REFERENCE_SOLUTIONS  # noqa: E402
from scripts.benchmark_temporal_pose_reuse import saved_candidate  # noqa: E402
from scripts.benchmark_cpu_pose_port import Server, compare  # noqa: E402
from scripts.export_cpu_pose_fixtures import fit, pack  # noqa: E402


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["matty", "yiheng", "tommy"])
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "pose-fitter/results/sequence-parity"
    )
    parser.add_argument(
        "--fixture-template", type=Path, default=ROOT / "pose-fitter/tests/fixtures/cpu-golden.json"
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    template = json.loads(args.fixture_template.read_text())
    report = dict(
        scope=__doc__,
        cases=[],
        passed=True,
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                ROOT / "scripts/independent_keypoint_pose_geometry.py",
                ROOT / "scripts/render_yolo11n_independent_keypoints.py",
            ]
        },
    )
    for name in args.cases:
        solution = json.loads((ROOT / REFERENCE_SOLUTIONS[name]).read_text())
        log = Path(solution["config"]["log"])
        metadata = json.loads(log.with_suffix(".meta.json").read_text())
        start = metadata.get("turning_start_frame") or 0
        end = solution.get("solved_frame_index") or max(s["frame_index"] for s in solution["steps"])
        observations, skipped = [], []
        for line in log.open():
            record = json.loads(line)
            if not start <= record["frame_index"] <= end:
                continue
            c = saved_candidate(record)
            if c is None or len(c.assignment_image_points) < 3 or not c.sticker_mask_centers:
                skipped.append(record["frame_index"])
            else:
                observations.append((record["frame_index"], c))
        print(
            name,
            "window",
            start,
            end,
            "eligible",
            len(observations),
            "skipped",
            len(skipped),
            flush=True,
        )
        corpus = dict(geometry=template["geometry"], workloads=[], probes=[], sweep_probes=[])
        banks = []
        for index, (frame, c) in enumerate(observations):
            prior = observations[index - 1][1] if index else None
            started = perf_counter()
            bank = fitter.fit_frame(renderer, c, prior=prior)
            elapsed = 1000 * (perf_counter() - started)
            banks.append(bank)
            item = pack(name, frame, "initial_geometry", c, prior, (), ())
            item.update(expected=[fit(f) for f in bank], cpu_ms=elapsed)
            corpus["workloads"].append(item)
            if index % 40 == 0 or index + 1 == len(observations):
                print(
                    name,
                    "CPU initial",
                    index + 1,
                    "/",
                    len(observations),
                    "frame",
                    frame,
                    flush=True,
                )
        for index, (frame, _) in enumerate(observations):
            bank = banks[index]
            if not bank:
                continue
            neighbors = [
                banks[i][0].candidate
                for i in (index - 1, index + 1)
                if 0 <= i < len(banks) and banks[i]
            ]
            if not neighbors:
                continue
            c, prior = bank[0].candidate, neighbors[0]
            started = perf_counter()
            revised = fitter.fit_frame(renderer, c, prior=prior, neighbors=neighbors, retained=bank)
            elapsed = 1000 * (perf_counter() - started)
            item = pack(name, frame, "temporal_neighbors", c, prior, neighbors, bank)
            item.update(expected=[fit(f) for f in revised], cpu_ms=elapsed)
            corpus["workloads"].append(item)
            if index % 40 == 0 or index + 1 == len(observations):
                print(
                    name,
                    "CPU temporal",
                    index + 1,
                    "/",
                    len(observations),
                    "frame",
                    frame,
                    flush=True,
                )
        fixture_path = args.output_dir / f"{name}-fixtures.json"
        write_json(fixture_path, corpus)
        summary = dict(
            name=name,
            log=str(log),
            start=start,
            end=end,
            eligible_frames=len(observations),
            skipped_frames=skipped,
            fixture_sha256=hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
            calls=len(corpus["workloads"]),
            rows=[],
            passed=True,
        )
        servers = {}
        try:
            servers["native"] = Server(
                [
                    str(ROOT / "pose-fitter/target/release/pose-benchmark"),
                    str(fixture_path),
                    "--serve",
                ]
            )
            servers["wasm"] = Server(
                ["node", "pose-fitter/tools/benchmark-wasm.mjs", str(fixture_path), "--serve"]
            )
            for index, item in enumerate(corpus["workloads"]):
                row = dict(
                    frame=item["frame"],
                    stage=item["stage"],
                    cpu_bank_size=len(item["expected"]),
                    parity={},
                )
                for runtime, server in servers.items():
                    actual = server.run(index)
                    parity = compare(item["expected"], actual["bank"])
                    row["parity"][runtime] = parity
                    if not parity["passed"]:
                        summary["passed"] = False
                        write_json(
                            args.output_dir
                            / "differences"
                            / f"{name}-{item['frame']}-{item['stage']}-{runtime}.json",
                            dict(workload=item, actual=actual, parity=parity),
                        )
                summary["rows"].append(row)
                if index % 80 == 0 or index + 1 == len(corpus["workloads"]):
                    print(
                        name,
                        "port comparison",
                        index + 1,
                        "/",
                        len(corpus["workloads"]),
                        "differing calls",
                        sum(
                            any(not p["passed"] for p in r["parity"].values())
                            for r in summary["rows"]
                        ),
                        flush=True,
                    )
        finally:
            for server in servers.values():
                server.close()
        report["cases"].append(summary)
        report["passed"] &= summary["passed"]
        write_json(args.output_dir / "report.json", report)
        print(name, "complete", "passed", summary["passed"], flush=True)
    print(args.output_dir / "report.json", flush=True)


if __name__ == "__main__":
    main()
