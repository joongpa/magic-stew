"""Rank full-window CPU/WASM winning-fit differences and render video overlays."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import render_yolo11n_independent_keypoints as renderer  # noqa: E402
from scripts.benchmark_cpu_pose_port import Server, compare, hydrate  # noqa: E402

CPU_COLOR = "#20e2c1"
WASM_COLOR = "#ffad47"


def measure(expected, actual):
    a, b = np.asarray(expected["assignment_projected"]), np.asarray(actual["assignment_projected"])
    assert a.shape == b.shape
    distance = np.linalg.norm(a - b, axis=1)
    return dict(
        point_max_px=float(distance.max(initial=0)),
        point_rms_px=float(np.sqrt(np.mean(distance**2))) if len(distance) else 0.0,
        score_delta=actual["score"] - expected["score"],
        cpu_score=expected["score"],
        wasm_score=actual["score"],
        cpu_axis=expected["axis"],
        wasm_axis=actual["axis"],
    )


def render_panel(rows, output, title):
    fig, axes = plt.subplots(
        len(rows),
        3,
        figsize=(12, 3.8 * len(rows)),
        squeeze=False,
        facecolor="#101820",
        layout="constrained",
    )
    for number, row in enumerate(rows):
        item, a, b = row["workload"], row["cpu"], row["wasm"]
        base, _ = hydrate(item)
        metadata = json.loads(Path(row["log"]).with_suffix(".meta.json").read_text())
        cap = cv2.VideoCapture(metadata["source"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, row["frame"])
        ok, image = cap.read()
        cap.release()
        assert ok, (row["name"], row["frame"])
        assert (image.shape[1], image.shape[0]) == tuple(item["image_size"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        def faces(fit):
            rotations = renderer.parallel_layer_rotations(fit["axis"], fit["angles"])
            pose = replace(
                base,
                rvec=np.asarray(fit["params"][:3]).reshape(3, 1),
                tvec=np.asarray(fit["params"][3:6]).reshape(3, 1),
                layer_rotations=rotations,
                sticker_layer_rotations=rotations,
            )
            return [
                polygon
                for _, polygon, _ in renderer.projected_cubelet_faces(pose, pose.rvec, pose.tvec)
            ]

        cpu_faces, wasm_faces = faces(a), faces(b)
        all_points = np.concatenate(cpu_faces + wasm_faces)
        lower, upper = all_points.min(axis=0), all_points.max(axis=0)
        center = (lower + upper) / 2
        radius = max(upper - lower) * 0.68
        bounds = (
            max(0, center[0] - radius),
            min(image.shape[1], center[0] + radius),
            min(image.shape[0], center[1] + radius),
            max(0, center[1] - radius),
        )
        for column, ax in enumerate(axes[number]):
            ax.imshow(image)
            ax.set_xlim(bounds[:2])
            ax.set_ylim(bounds[2:])
            ax.axis("off")
            if column < 2:
                for polygon in cpu_faces if column == 0 else wasm_faces:
                    closed = np.r_[polygon, polygon[:1]]
                    ax.plot(
                        closed[:, 0],
                        closed[:, 1],
                        color=CPU_COLOR if column == 0 else WASM_COLOR,
                        linewidth=1.1,
                        alpha=0.9,
                    )
            else:
                p, q = np.asarray(a["assignment_projected"]), np.asarray(b["assignment_projected"])
                for first, second in zip(p, q):
                    ax.plot(
                        [first[0], second[0]], [first[1], second[1]], color="#ff5fa2", linewidth=1.2
                    )
                ax.scatter(p[:, 0], p[:, 1], s=16, color=CPU_COLOR, marker="o", label="CPU")
                ax.scatter(q[:, 0], q[:, 1], s=24, color=WASM_COLOR, marker="x", label="WASM")
            label = ("CPU wireframe", "WASM wireframe", "Assigned landmark projections")[column]
            ax.set_title(label, color="#c9d4df", fontsize=10)
        axes[number, 0].text(
            0,
            1.16,
            f"{row['name'].title()} · frame {row['frame']} · {row['stage'].replace('_', ' ')}\n"
            f"Max point distance {row['point_max_px']:.6g} px · RMS {row['point_rms_px']:.6g} px",
            transform=axes[number, 0].transAxes,
            color="white",
            fontsize=11,
            va="bottom",
        )
    fig.suptitle(
        title + "\nTeal = CPU · Orange = WASM · Magenta = displacement", color="white", fontsize=15
    )
    fig.savefig(output, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=ROOT / "pose-fitter/results/sequence-parity"
    )
    parser.add_argument(
        "--summary", type=Path, default=ROOT / "pose-fitter/benchmarks/sequence-parity.json"
    )
    args = parser.parse_args()
    report = json.loads((args.input_dir / "report.json").read_text())
    assert {c["name"] for c in report["cases"]} == {"matty", "yiheng", "tommy"}
    ranked, cases = [], []
    for case in report["cases"]:
        path = args.input_dir / f"{case['name']}-fixtures.json"
        corpus = json.loads(path.read_text())
        server = Server(["node", "pose-fitter/tools/benchmark-wasm.mjs", str(path), "--serve"])
        case_rows = []
        try:
            for index, item in enumerate(corpus["workloads"]):
                output = server.run(index)
                a, b = item["expected"], output["bank"]
                if not a or not b:
                    assert not a and not b, "Empty/nonempty winner mismatch"
                    continue
                row = dict(
                    name=case["name"],
                    log=case["log"],
                    frame=item["frame"],
                    stage=item["stage"],
                    winner_passed=compare(a[:1], b[:1])["passed"],
                    bank_passed=compare(a, b)["passed"],
                    cpu_bank_size=len(a),
                    wasm_bank_size=len(b),
                    **measure(a[0], b[0]),
                )
                case_rows.append(row)
                ranked.append({**row, "workload": item, "cpu": a[0], "wasm": b[0]})
                if index % 160 == 0:
                    print(
                        case["name"],
                        "ranking",
                        index + 1,
                        "/",
                        len(corpus["workloads"]),
                        flush=True,
                    )
        finally:
            server.close()
        worst = max(case_rows, key=lambda r: r["point_max_px"])
        cases.append(
            dict(
                name=case["name"],
                eligible_frames=case["eligible_frames"],
                calls=case["calls"],
                skipped_frames=case["skipped_frames"],
                bank_differences=sum(not r["bank_passed"] for r in case_rows),
                winner_differences=sum(not r["winner_passed"] for r in case_rows),
                worst_winning_fit=worst,
            )
        )
        print(cases[-1], flush=True)
    ranked.sort(key=lambda r: r["point_max_px"], reverse=True)
    keys = [k for k in ranked[0] if k not in ("workload", "cpu", "wasm", "log")]
    with (args.input_dir / "winning-fit-differences.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: r[k] for k in keys} for r in ranked)

    def compact(row):
        return {key: row[key] for key in keys}

    summary = dict(
        scope=report["scope"],
        runtime="Same standalone WASM module under Node/V8",
        source_sha256=report["source_sha256"],
        wasm_sha256=hashlib.sha256(
            (
                ROOT / "pose-fitter/target/wasm32-unknown-unknown/release/cube_pose_fitter.wasm"
            ).read_bytes()
        ).hexdigest(),
        port_source_sha256={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "pose-fitter/src").glob("*.rs"))
        },
        cases=cases,
        metric="Euclidean distance between CPU/WASM assigned projections for the same observed landmark; original source pixels.",
        top_winning_differences=[compact(r) for r in ranked[:12]],
        notes=[
            "Complete ordered banks are compared under the original benchmark tolerances.",
            "Frame numbers are zero-based source-video indices.",
            "Both overlays use the existing CPU renderer with the returned pose/angles; no refitting.",
            "Raw angle differences can be 90 degrees for equivalent surfaces; ranking uses image-space distances.",
        ],
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    worst_by_case = [
        max((r for r in ranked if r["name"] == name), key=lambda r: r["point_max_px"])
        for name in ("matty", "yiheng", "tommy")
    ]
    render_panel(
        worst_by_case,
        args.input_dir / "worst-per-solve.png",
        "Largest winning-fit difference in each solve",
    )
    unique = []
    for row in ranked:
        if (row["name"], row["frame"]) not in {(r["name"], r["frame"]) for r in unique}:
            unique.append(row)
        if len(unique) == 4:
            break
    render_panel(
        unique,
        args.input_dir / "largest-winning-differences.png",
        "Largest winning-fit differences across all three solves",
    )
    print(args.summary, flush=True)


if __name__ == "__main__":
    main()
