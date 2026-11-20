"""Validate browser output and summarize the saved CPU/Rust/WASM benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_cpu_pose_port import compare, seed_audit, check_numerics  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures", type=Path, default=ROOT / "pose-fitter/fixtures/cpu-reference.json"
    )
    parser.add_argument(
        "--baseline", type=Path, default=ROOT / "pose-fitter/results/interleaved.json"
    )
    parser.add_argument("--browser", type=Path, default=ROOT / "pose-fitter/results/browser.json")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "pose-fitter/benchmarks/cpu-port.json"
    )
    args = parser.parse_args()
    fixture_bytes = args.fixtures.read_bytes()
    fixture = json.loads(fixture_bytes)
    baseline, browser = (json.loads(p.read_text()) for p in (args.baseline, args.browser))
    digest = hashlib.sha256(fixture_bytes).hexdigest()
    assert digest == baseline["fixture_sha256"] == browser["fixture_sha256"], "Fixture mismatch"
    assert baseline["passed"], "Baseline numerical checks failed"
    assert check_numerics(browser["probes"]), browser["probes"]
    assert len(browser["cases"]) == len(fixture["workloads"])
    parity = []
    for index, (workload, result) in enumerate(zip(fixture["workloads"], browser["cases"])):
        assert result["index"] == index
        bank = compare(workload["expected"], result["bank"])
        assert bank["passed"], (index, bank)
        audit = seed_audit(workload["refinement_traces"], result["refinements"])
        assert audit["cpu_count"] == audit["port_count"]
        parity.append(
            dict(
                index=index,
                name=workload["name"],
                stage=workload["stage"],
                bank=bank,
                seed_audit=audit,
            )
        )
    stages = {}
    for stage in ("initial_geometry", "temporal_neighbors", "static_geometry"):
        indices = [i for i, w in enumerate(fixture["workloads"]) if w["stage"] == stage]
        timings = {
            runtime: [baseline["cases"][i]["medians_ms"][runtime] for i in indices]
            for runtime in ("cpu", "native", "wasm_node")
        }
        timings["wasm_browser"] = [browser["cases"][i]["median_ms"] for i in indices]
        stages[stage] = {
            runtime: dict(
                median_ms=statistics.median(values), min_ms=min(values), max_ms=max(values)
            )
            for runtime, values in timings.items()
        }
        print(stage, {k: round(v["median_ms"], 3) for k, v in stages[stage].items()})
    maximum = {
        key: max(p["bank"]["max_abs"][key] for p in parity) for key in parity[0]["bank"]["max_abs"]
    }
    seed_differences = [
        dict(workload=p["index"], **d) for p in parity for d in p["seed_audit"]["differences"]
    ]
    summary = dict(
        passed=True,
        scope=baseline["scope"],
        timestamp_utc=baseline["timestamp_utc"],
        fixture_sha256=digest,
        source_sha256=fixture["source_sha256"],
        port_source_sha256=baseline["port_source_sha256"],
        wasm_sha256=baseline["wasm_sha256"],
        platform=baseline["platform"],
        python=baseline["python"],
        numpy=baseline["numpy"],
        scipy=baseline["scipy"],
        torch=baseline["torch"],
        torch_threads=baseline["torch_threads"],
        node_runtime={k: v for k, v in baseline["runtimes"]["wasm_node"].items() if k != "probes"},
        browser={
            k: browser[k] for k in ("userAgent", "hardwareConcurrency", "wasmBytes", "init_ms")
        },
        workloads=len(parity),
        repeats=baseline["repeats"],
        browser_repeats=browser["repeats"],
        timings=stages,
        browser_bank_max_abs=maximum,
        tolerances=baseline["tolerances"],
        browser_probes=browser["probes"],
        native_probes=baseline["runtimes"]["native"]["probes"],
        node_probes=baseline["runtimes"]["wasm_node"]["probes"],
        browser_seed_differences=seed_differences,
        output_serialization_mean_ms=statistics.mean(c["mean_output_ms"] for c in browser["cases"]),
        notes=[
            "Five independent saved runs; cached cameras are provided to every runtime.",
            "CPU/native/Node measurements are interleaved; browser measurements are a separate pass.",
            "Bank order, axes, sources, integer metrics and assignments match within declared float tolerances.",
            "Coarse tie choices can differ; exact seed solver probes are separate from normal search.",
            "No app integration, whole-video rollout or ground-truth accuracy claim.",
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print("Browser bank max errors", maximum)
    print("Coarse seed differences", len(seed_differences))
    print(args.output)


if __name__ == "__main__":
    main()
