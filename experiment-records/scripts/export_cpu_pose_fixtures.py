"""Export identical saved-input CPU joint-fitter workloads for native/WASM parity.

Scope is fit_frame, matching benchmark_pose_geometry_mechanical.py. Cached input
cameras are supplied to BOTH implementations, never counted as a cold-start test.
No detector, PnP seed generation, orientation matching or move search is timed.
"""

from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import independent_keypoint_pose_geometry as fitter  # noqa: E402
from scripts import render_yolo11n_independent_keypoints as r  # noqa: E402
from scripts.analyze_orientation_motion import REFERENCE_SOLUTIONS  # noqa: E402
from scripts.benchmark_temporal_pose_reuse import saved_candidate  # noqa: E402


def pose(p):
    return dict(rvec=p.rvec.ravel().tolist(), tvec=p.tvec.ravel().tolist())


def fit(f):
    c, o = f.candidate, f.option
    return dict(
        axis=f.axis,
        params=[*c.rvec.ravel(), *c.tvec.ravel(), f.angles[0], f.angles[2]],
        angles=list(f.angles),
        score=f.score,
        source=f.proposal_source,
        flat_fallback=f.flat_fallback,
        metrics={
            key: o[key]
            for key in (
                "keypoint_error",
                "sticker_error",
                "combined_mean_error",
                "robust_error",
                "p90_error",
                "inlier_count",
                "layer_inlier_count",
                "keypoint_inlier_count",
                "keypoint_layer_inlier_count",
                "sticker_match_count",
                "layer_support",
            )
        },
        nearest=f.nearest.tolist(),
        valid=f.valid.tolist(),
        assignment_objects=c.assignment_object_points.tolist(),
        assignment_projected=c.assignment_projected_points.tolist(),
        assignment_classes=c.assignment_projected_class_ids.tolist(),
    )


def pack(name, frame, stage, c, prior, neighbors, retained):
    return dict(
        name=name,
        frame=frame,
        stage=stage,
        camera={**pose(c), "k": c.camera_matrix.tolist(), "dist": c.dist_coeffs.ravel().tolist()},
        observations=dict(
            points=c.assignment_image_points.tolist(),
            classes=c.assignment_observed_class_ids.tolist(),
            stickers=np.asarray(c.sticker_mask_centers).reshape(-1, 2).tolist(),
            uncertainties=list(c.sticker_mask_center_uncertainties),
        ),
        prior=None if prior is None else pose(prior),
        neighbors=[pose(n) for n in neighbors],
        retained=[fit(f) for f in retained],
        static_mode=stage == "static_geometry",
        image_size=list(c.image_size),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "pose-fitter/fixtures/cpu-reference.json"
    )
    parser.add_argument("--samples-per-case", type=int, default=2)
    parser.add_argument("--cases", nargs="+", default=list(REFERENCE_SOLUTIONS))
    parser.add_argument(
        "--golden-output", type=Path, help="Write a compact, deterministic Rust regression fixture"
    )
    args = parser.parse_args()
    faces = r._turn_gpu_static_geometry()
    geometry = dict(
        faces=faces[2].tolist(),
        face_normals=faces[5].tolist(),
        corners=r.CUBE_CORNER_POINTS_MM.tolist(),
        symmetries=np.asarray(r.CUBE_SYMMETRY_MATRICES).tolist(),
        cube_size=r.CUBE_SIZE_MM,
        assignment_fraction=r.PNP_FINAL_ASSIGNMENT_ERROR_FRACTION,
        sticker_scale=r.PNP_TURN_CENTER_PENALTY_SCALE,
        axes=[],
    )
    for axis in range(3):
        layers, vertices, face_layers, point_aliases, face_aliases = r._parallel_slab_geometry(axis)
        geometry["axes"].append(
            dict(
                points=r.CUBE_LATTICE_OBJECT_POINTS_MM[vertices].tolist(),
                normals=r.CUBE_LATTICE_BOUNDARY_NORMALS[vertices].tolist(),
                masks=r.CUBE_LATTICE_BOUNDARY_FACE_MASK[vertices].tolist(),
                classes=r.CUBE_LATTICE_CLASS_IDS[vertices].tolist(),
                layers=layers.tolist(),
                face_layers=face_layers.tolist(),
                point_aliases=point_aliases.tolist(),
                face_aliases=face_aliases.tolist(),
            )
        )
    result = dict(
        version=1,
        scope=__doc__,
        geometry=geometry,
        workloads=[],
        probes=[],
        sweep_probes=[],
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
        meta = json.loads(log.with_suffix(".meta.json").read_text())
        start = meta.get("turning_start_frame") or 0
        end = solution.get("solved_frame_index") or max(s["frame_index"] for s in solution["steps"])
        observations = []
        for line in log.open():
            rec = json.loads(line)
            if start <= rec["frame_index"] <= end:
                c = saved_candidate(rec)
                if (
                    c is not None
                    and len(c.assignment_image_points) >= 3
                    and len(c.sticker_mask_centers)
                ):
                    observations.append((rec["frame_index"], c))
        indices = np.unique(
            np.linspace(1, len(observations) - 2, args.samples_per_case + 2, dtype=int)[1:-1]
        )
        cases = []
        for index in indices:
            frame, c = observations[index]
            cases.append((frame, "initial_geometry", c, observations[index - 1][1], (), ()))
        index = int(indices[0])
        frame, c = observations[index]
        retained = fitter.fit_frame(r, c, prior=observations[index - 1][1])
        neighbors = [
            fitter.fit_frame(r, observations[i][1])[0].candidate for i in (index - 1, index + 1)
        ]
        cases.append(
            (frame, "temporal_neighbors", retained[0].candidate, neighbors[0], neighbors, retained)
        )
        cases.append((frame, "static_geometry", c, observations[index - 1][1], (), ()))
        for frame, stage, c, prior, neighbors, retained in cases:
            item = pack(name, frame, stage, c, prior, neighbors, retained)
            traces = []
            seed_scores = []
            original_shortlist = fitter.diverse_shortlist

            def shortlist(*a, **kw):
                selected = original_shortlist(*a, **kw)
                if kw.get("size") == 2:
                    seed_scores.extend(f.score for f in selected[:2])
                return selected

            original_refine, original_lsq = fitter._refine, fitter.least_squares
            last_solve = {}

            def lsq(*a, **kw):
                result = original_lsq(*a, **kw)
                last_solve.update(nfev=result.nfev, status=result.status, cost=result.cost)
                return result

            def refine(model, seed, **kw):
                output = original_refine(model, seed, **kw)
                traces.append(
                    dict(
                        axis=model.axis,
                        seed_score=seed_scores[len(traces)] if seed_scores else None,
                        seed=np.asarray(seed).tolist(),
                        output=output.tolist(),
                        prior=np.asarray(kw.get("prior", np.asarray(seed)[:6])).tolist(),
                        mask=[False, False]
                        if kw.get("fixed_angles") is not None
                        else (
                            [kw["free_angle"] == 6, kw["free_angle"] == 7]
                            if kw.get("free_angle") is not None
                            else [True, True]
                        ),
                        **last_solve,
                    )
                )
                return output

            fitter._refine, fitter.least_squares = refine, lsq
            fitter.diverse_shortlist = shortlist
            try:
                started = perf_counter()
                bank = fitter.fit_frame(
                    r,
                    c,
                    static=item["static_mode"],
                    prior=prior,
                    neighbors=neighbors,
                    retained=retained,
                )
            finally:
                fitter._refine, fitter.least_squares = original_refine, original_lsq
                fitter.diverse_shortlist = original_shortlist
            item["refinement_traces"] = traces
            item["cpu_ms"] = 1000 * (perf_counter() - started)
            item["expected"] = [fit(f) for f in bank]
            result["workloads"].append(item)
            if not item["static_mode"]:
                cameras = neighbors if retained else (c, *neighbors)
                for camera in cameras:
                    swept = fitter.swept_geometries(r, c, camera)
                    result["sweep_probes"].append(
                        dict(
                            workload=len(result["workloads"]) - 1,
                            camera=pose(camera),
                            scores=[f.score for f in swept],
                        )
                    )
            print(name, stage, frame, len(bank), round(item["cpu_ms"], 2), flush=True)
        # Independent low-level checks include nonzero distortion and zero rotation.
        for axis in range(3):
            for variant in range(3):
                camera = c
                x = np.r_[c.rvec.ravel(), c.tvec.ravel(), [13.0, -21.0]]
                if variant == 1:
                    x[:3] = 0.0
                if variant == 2:
                    camera = replace(
                        c,
                        dist_coeffs=np.array(
                            [
                                0.03,
                                -0.01,
                                0.001,
                                -0.002,
                                0.002,
                                0.001,
                                -0.001,
                                0.0001,
                                0.0002,
                                -0.0001,
                                0.0001,
                                0.0002,
                            ]
                        ),
                    )
                model = fitter.ArticulatedProjection(r, camera, axis)
                values, jac = model.evaluate(x)
                result["probes"].append(
                    dict(
                        workload=len(result["workloads"]) - 1,
                        axis=axis,
                        params=x.tolist(),
                        dist=camera.dist_coeffs.ravel().tolist(),
                        residual=values.tolist(),
                        jacobian=jac.tolist(),
                        diameter=r.projected_cube_diameter(camera, rvec=x[:3], tvec=x[3:6]),
                        corners=__import__("cv2")
                        .projectPoints(
                            r.CUBE_CORNER_POINTS_MM,
                            x[:3],
                            x[3:6],
                            camera.camera_matrix,
                            camera.dist_coeffs,
                        )[0]
                        .reshape(-1, 2)
                        .tolist(),
                    )
                )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, allow_nan=False, separators=(",", ":")) + "\n")
    print(args.output, flush=True)
    if args.golden_output:
        selected = [
            i
            for i, w in enumerate(result["workloads"])
            if w["static_mode"]
            or (w["name"] == "matty" and w["stage"] == "initial_geometry" and w["frame"] == 445)
            or (w["name"] == "yiheng" and w["stage"] == "temporal_neighbors")
        ]
        remap = {old: new for new, old in enumerate(selected)}
        golden = {**result, "workloads": [result["workloads"][i] for i in selected]}
        for key in ("probes", "sweep_probes"):
            golden[key] = [
                {**p, "workload": remap[p["workload"]]}
                for p in result[key]
                if p["workload"] in remap
            ]
        args.golden_output.parent.mkdir(parents=True, exist_ok=True)
        args.golden_output.write_text(
            json.dumps(golden, allow_nan=False, separators=(",", ":")) + "\n"
        )


if __name__ == "__main__":
    import torch

    torch.set_num_threads(2)
    main()
