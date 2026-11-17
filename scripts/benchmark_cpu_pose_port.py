"""Interleave CPU, native Rust and Node WASM fits on the same saved inputs.

Numerical work only: loading, fixture reconstruction and JSON transfer are outside
timers. The cached cameras are inputs to every runtime, including Python. This
does not measure detection, PnP initialization, orientation or sequence recovery.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter

import numpy as np
import scipy
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import independent_keypoint_pose_geometry as fitter  # noqa: E402
from scripts import render_yolo11n_independent_keypoints as renderer  # noqa: E402
from scripts.export_cpu_pose_fixtures import fit as pack_fit  # noqa: E402


def hydrate(item):
    camera, obs = item["camera"], item["observations"]
    base = renderer.StickerPnPCandidate(
        sticker_type="fixture",
        sticker_average_confidence=1.0,
        image_points=np.empty((0, 2)),
        object_points=np.empty((0, 3)),
        projected_sticker_points=np.empty((0, 2)),
        reprojection_error=float("nan"),
        camera_matrix=np.asarray(camera["k"]),
        dist_coeffs=np.asarray(camera["dist"]),
        rvec=np.asarray(camera["rvec"]).reshape(3, 1),
        tvec=np.asarray(camera["tvec"]).reshape(3, 1),
        image_size=tuple(item["image_size"]),
        assignment_image_points=np.asarray(obs["points"]).reshape(-1, 2),
        assignment_observed_class_ids=np.asarray(obs["classes"]),
        sticker_mask_centers=tuple(np.asarray(p) for p in obs["stickers"]),
        sticker_mask_center_uncertainties=tuple(obs["uncertainties"]),
    )

    def camera_pose(p):
        return replace(
            base, rvec=np.asarray(p["rvec"]).reshape(3, 1), tvec=np.asarray(p["tvec"]).reshape(3, 1)
        )

    retained = []
    for saved in item["retained"]:
        f = fitter._materialize(renderer, base, saved["axis"], saved["params"])
        # Preserve the original retained measurements, as the temporal CPU path does.
        f.score, f.proposal_source = saved["score"], saved["source"]
        f.flat_fallback = saved.get("flat_fallback", False)
        f.option.update(saved["metrics"])
        retained.append(f)
    return base, dict(
        static=item["static_mode"],
        prior=None if item["prior"] is None else camera_pose(item["prior"]),
        neighbors=tuple(camera_pose(p) for p in item["neighbors"]),
        retained=retained,
    )


# Per-component units: radians/mm/degrees for params, pixels for projected points.
TOLERANCES = {
    "params": 1e-4,
    "angles": 1e-4,
    "score": 1e-6,
    "metrics": 1e-6,
    "assignment_objects": 1e-4,
    "assignment_projected": 1e-4,
}
DISCRETE = ("axis", "source", "nearest", "valid", "assignment_classes", "flat_fallback")


def compare(expected, actual):
    errors = {k: 0.0 for k in TOLERANCES}
    issues = []
    if len(expected) != len(actual):
        issues.append(f"bank length {len(actual)} != {len(expected)}")
    for i, (a, b) in enumerate(zip(expected, actual)):
        for key in DISCRETE:
            if a.get(key, False) != b.get(key, False):
                issues.append(f"bank[{i}].{key}")
        for key in TOLERANCES:
            av, bv = a[key], b[key]
            if key == "metrics":
                for metric in av:
                    if metric.endswith("count") or metric == "layer_support":
                        if not np.array_equal(av[metric], bv[metric]):
                            issues.append(f"bank[{i}].metrics.{metric}")
                    else:
                        errors[key] = max(errors[key], abs(av[metric] - bv[metric]))
            else:
                aa, bb = np.asarray(av), np.asarray(bv)
                if aa.shape != bb.shape:
                    issues.append(f"bank[{i}].{key} shape")
                elif aa.size:
                    errors[key] = max(errors[key], float(np.max(np.abs(aa - bb))))
    issues.extend(
        f"{key} error {value} > {TOLERANCES[key]}"
        for key, value in errors.items()
        if value > TOLERANCES[key] or not np.isfinite(value)
    )
    return dict(passed=not issues, max_abs=errors, issues=issues)


def seed_audit(reference, actual):
    differences = []
    for index, (a, b) in enumerate(zip(reference, actual)):
        delta = float(np.max(np.abs(np.asarray(a["seed"]) - b["seed"])))
        if a["axis"] != b["axis"] or delta > 1e-8:
            differences.append(
                dict(
                    index=index,
                    axis=a["axis"],
                    cpu_seed=a["seed"],
                    port_seed=b["seed"],
                    parameter_max_abs=delta,
                    score_gap=None
                    if a["seed_score"] is None
                    else abs(a["seed_score"] - b["seed_score"]),
                )
            )
    return dict(cpu_count=len(reference), port_count=len(actual), differences=differences)


def check_numerics(probes):
    projection, solver, sweep = (probes[k] for k in ("projection", "solver", "sweep"))
    return (
        projection["probes"] > 0
        and projection["residual_max_abs"] < 2e-5
        and projection["jacobian_max_abs"] < 2e-5
        and projection["diameter_max_abs"] < 1e-4
        and solver["solves"] > 0
        and solver["parameter_max_abs"] < 1e-8
        and solver["cost_max_abs"] < 1e-8
        and solver["stopping_differences"] == 0
        and sweep["rows"] > 0
        and sweep["score_max_abs"] < 1e-6
        and sweep["length_differences"] == 0
    )


class Server:
    def __init__(self, command):
        self.process = subprocess.Popen(
            command, cwd=ROOT, text=True, bufsize=1, stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )
        self.metadata = json.loads(self.process.stdout.readline())
        if not self.metadata.get("ready"):
            raise RuntimeError(self.metadata)

    def run(self, index):
        self.process.stdin.write(json.dumps(dict(index=index)) + "\n")
        self.process.stdin.flush()
        return json.loads(self.process.stdout.readline())

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures", type=Path, default=ROOT / "pose-fitter/fixtures/cpu-reference.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "pose-fitter/results/interleaved.json"
    )
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    torch.set_num_threads(2)
    fixture_bytes = args.fixtures.read_bytes()
    data = json.loads(fixture_bytes)
    for name, digest in data["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    hydrated = [hydrate(w) for w in data["workloads"]]
    servers = {}
    try:
        servers["native"] = Server(
            [str(ROOT / "pose-fitter/target/release/pose-benchmark"), str(args.fixtures), "--serve"]
        )
        servers["wasm_node"] = Server(
            ["node", "pose-fitter/tools/benchmark-wasm.mjs", str(args.fixtures), "--serve"]
        )
        report = dict(
            scope=__doc__,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            fixture_sha256=hashlib.sha256(fixture_bytes).hexdigest(),
            port_source_sha256={
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((ROOT / "pose-fitter/src").glob("*.rs"))
            },
            wasm_sha256=hashlib.sha256(
                (
                    ROOT / "pose-fitter/target/wasm32-unknown-unknown/release/cube_pose_fitter.wasm"
                ).read_bytes()
            ).hexdigest(),
            platform=platform.platform(),
            python=platform.python_version(),
            numpy=np.__version__,
            scipy=scipy.__version__,
            torch=torch.__version__,
            torch_threads=torch.get_num_threads(),
            repeats=args.repeats,
            runtimes={k: s.metadata for k, s in servers.items()},
            tolerances=TOLERANCES,
            cases=[],
        )
        for index, (item, (base, options)) in enumerate(zip(data["workloads"], hydrated)):
            times = {k: [] for k in ("cpu", *servers)}
            results = {}
            # One full warm-up per runtime per workload; subsequent passes rotate order.
            for pass_index in range(-1, args.repeats):
                order = list(times)
                rotation = (index + pass_index + 1) % len(order)
                for runtime in order[rotation:] + order[:rotation]:
                    if runtime == "cpu":
                        start = perf_counter()
                        bank = fitter.fit_frame(renderer, base, **options)
                        elapsed = 1000 * (perf_counter() - start)
                        results[runtime] = dict(bank=[pack_fit(f) for f in bank])
                    else:
                        results[runtime] = servers[runtime].run(index)
                        elapsed = results[runtime]["ms"]
                    if pass_index >= 0:
                        times[runtime].append(elapsed)
            case = dict(
                index=index,
                name=item["name"],
                frame=item["frame"],
                stage=item["stage"],
                times_ms=times,
                medians_ms={k: statistics.median(v) for k, v in times.items()},
                parity={k: compare(item["expected"], v["bank"]) for k, v in results.items()},
                seed_audit={
                    k: seed_audit(item["refinement_traces"], results[k]["refinements"])
                    for k in servers
                },
                results=results,
            )
            report["cases"].append(case)
            print(
                item["name"],
                item["stage"],
                {k: round(v, 2) for k, v in case["medians_ms"].items()},
                {k: v["passed"] for k, v in case["parity"].items()},
                flush=True,
            )
        report["passed"] = all(v["passed"] for c in report["cases"] for v in c["parity"].values())
        report["passed"] &= all(check_numerics(s.metadata["probes"]) for s in servers.values())
        report["passed"] &= all(
            a["cpu_count"] == a["port_count"]
            for c in report["cases"]
            for a in c["seed_audit"].values()
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, allow_nan=False, separators=(",", ":")) + "\n")
        print(args.output, flush=True)
        if not report["passed"]:
            raise SystemExit("Numerical parity failed; see report")
    finally:
        for server in servers.values():
            server.close()


if __name__ == "__main__":
    main()
