"""Measure previous-pose initialization without changing the production fitter.

One-step trials use a saved previous-frame pose (an optimistic initialization
experiment). Rollouts use their own previous outputs after one saved bootstrap.
Saved current-frame fits are evaluation references only, never tracker inputs.
Agreement with these fits is not ground-truth accuracy. No inference or move
search runs, and already filtered detections are not thresholded again.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from itertools import product
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import independent_keypoint_pose_geometry as geometry
from scripts import render_yolo11n_independent_keypoints as renderer


DEFAULT_CASES = (
    ('matty', 'matty-exhaustive-middle-frame-wireframe', 560),
    ('nahm', 'split-tps-40-50-validation/nahm/geometry-render', 767),
    ('xuanyi', 'split-tps-40-50-validation/xuanyi', 234),
    ('tommy', 'split-tps-40-50-validation/tommy-short-fresh/geometry-render', 441),
    ('yiheng', 'yihengCropped-wireframe', 125),
)


def saved_candidate(record):
    """Read a cached pose and the current frame's already filtered observations."""
    options = ((record.get('search') or {}).get('geometry') or {}).get('options', ())
    payload = record.get('search_pose') or record.get('selected_pose') or {}
    if not options or not payload or not record.get('has_pose'):
        return None
    option = options[0]
    return renderer.StickerPnPCandidate(
        sticker_type=payload['sticker_type'], sticker_average_confidence=1.,
        image_points=np.empty((0, 2)), object_points=np.empty((0, 3)),
        projected_sticker_points=np.empty((0, 2)), reprojection_error=float('nan'),
        camera_matrix=np.asarray(payload['camera_matrix']),
        dist_coeffs=np.asarray(payload['dist_coeffs']), image_size=tuple(payload['image_size']),
        rvec=np.asarray(option['pose_rvec']), tvec=np.asarray(option['pose_tvec']),
        layer_rotations=tuple(renderer.CubeLayerRotation(**r) for r in option['layer_rotations']),
        assignment_image_points=np.asarray(payload['assignment_image_points']).reshape(-1, 2),
        assignment_observed_class_ids=np.asarray(payload['assignment_observed_class_ids']),
        assignment_projected_points=np.asarray(option['assignment_projected_points']),
        sticker_mask_centers=tuple(np.asarray(p) for p in payload['sticker_mask_centers']),
        sticker_mask_center_uncertainties=tuple(payload['sticker_mask_center_uncertainties']),
        sticker_color_names=tuple(payload['sticker_color_names']),
    )


def reuse_previous(observations, previous, *, iterations=12, pose_radius=None, angle_radius=None):
    """Try one local solve per axis, using only previous pose and new detections.

    Keep the existing soft motion prior and image objective. This first experiment
    deliberately does not impose hard physical angle limits on folded slab
    offsets: those offsets are modulo 90 degrees, not unwrapped physical travel.
    """
    base = replace(observations, rvec=previous.rvec.copy(), tvec=previous.tvec.copy(),
        layer_rotations=(), sticker_layer_rotations=None,
        assignment_object_points=None, assignment_projected_points=None,
        assignment_projected_class_ids=None, joint_geometry_candidates=())
    previous_axis = previous.layer_rotations[0].axis if previous.layer_rotations else None
    pose = np.r_[previous.rvec.ravel(), previous.tvec.ravel()]
    fits = []
    for axis in range(3):
        angles = (np.asarray(renderer.parallel_layer_angles(previous.layer_rotations))[[0, 2]]
                  if axis == previous_axis else np.zeros(2))
        seed = np.r_[pose, np.clip(angles, -45., 45.)]
        model = geometry.ArticulatedProjection(renderer, base, axis)
        parameters = geometry._refine(model, seed, prior=pose, iterations=iterations,
                                     pose_radius=pose_radius, angle_radius=angle_radius)
        fit = geometry._materialize(renderer, base, axis, parameters)
        if fit is not None:
            fit.proposal_source = 'previous_frame_local'
            fits.append(fit)
    return sorted(fits, key=lambda fit: (fit.score, fit.axis, fit.angles))


def local_radii(previous_error):
    """Numerical search bounds, not a physical angular-speed measurement.

    Rotation-vector component boxes are only local optimizer bounds. Folded
    articulation and cube-symmetry changes require separate physical bookkeeping.
    """
    if previous_error < .015:
        degrees, translation, angle = 5., (2., 2., 6.), 15.
    elif previous_error < .025:
        degrees, translation, angle = 10., (4., 4., 12.), 25.
    else:
        degrees, translation, angle = 20., (8., 8., 25.), 45.
    return np.r_[np.full(3, np.deg2rad(degrees)), translation], angle


def acceptable(fits, threshold, geometry_threshold):
    return bool(fits and renderer.pose_reprojection_fit_metric(fits[0].candidate) <= threshold
                and fits[0].score <= geometry_threshold)


def restricted_sweep(observations, previous, *, iterations=12):
    """Search outer angles at the previous middle-slab pose, then refine six seeds.

    This reduces 343 configurations per axis to 49. Unlike the full sweep, the
    middle-slab orientation is initially held at the previous frame's value;
    local refinement may subsequently move it.
    """
    base = replace(observations, rvec=previous.rvec.copy(), tvec=previous.tvec.copy(),
                   layer_rotations=(), sticker_layer_rotations=None)
    grid = np.array([(a, 0., b) for a, b in product(range(-45, 46, 15), repeat=2)])
    limit = renderer.PNP_FINAL_ASSIGNMENT_ERROR_FRACTION * renderer.projected_cube_diameter(base)
    prior = np.r_[base.rvec.ravel(), base.tvec.ravel()]
    fits = []
    for axis in range(3):
        options, _, _ = renderer.parallel_turn_frame_geometry(base, limit, axis, angle_grid=grid)
        options += renderer.parallel_turn_frame_geometry(base, limit, axis, angles=(0., 0., 0.))[0]
        ranked = sorted(options, key=lambda option: float(renderer.balanced_geometry_cost(
            option['keypoint_error'], option['sticker_error'])))
        model = geometry.ArticulatedProjection(renderer, base, axis)
        for option in ranked[:2]:
            angles = np.asarray(renderer.parallel_layer_angles(option['layer_rotations']))[[0, 2]]
            seed = np.r_[prior, angles]
            swept = geometry._materialize(renderer, base, axis, seed)
            if swept is not None:
                fits.append(swept)
            active = np.flatnonzero(angles != 0.)+6
            constraints = (dict(fixed_angles=(0., 0.)) if len(active) == 0 else
                           dict(free_angle=int(active[0])) if len(active) == 1 else {})
            parameters = geometry._refine(model, seed, prior=prior, iterations=iterations,
                                          **constraints)
            fit = geometry._materialize(renderer, base, axis, parameters)
            if fit is not None:
                fit.proposal_source = 'previous_camera_restricted_sweep'
                fits.append(fit)
    return sorted(fits, key=lambda fit: (fit.score, fit.axis, fit.angles))


def adaptive_reuse(observations, previous, *, iterations=12, threshold=.025,
                   geometry_threshold=.15):
    """Expand from local solves to exhaustive recovery using current evidence.

    The exact displayed REPROJ fit metric is the reprojection gate. A separate
    geometry-cost gate includes stickers, which REPROJ fit does not measure.
    Neither gate has access to the saved current-frame reference fit.
    """
    radius, angle_radius = local_radii(renderer.pose_reprojection_fit_metric(previous))
    fits = reuse_previous(observations, previous, iterations=iterations,
                          pose_radius=radius, angle_radius=angle_radius)
    if acceptable(fits, threshold, geometry_threshold):
        return fits, 'narrow_local'
    wide = reuse_previous(observations, previous, iterations=iterations)
    fits = sorted([*fits, *wide], key=lambda fit: (fit.score, fit.axis, fit.angles))
    if acceptable(fits, threshold, geometry_threshold):
        return fits, 'wide_local'
    swept = restricted_sweep(observations, previous, iterations=iterations)
    fits = sorted([*fits, *swept], key=lambda fit: (fit.score, fit.axis, fit.angles))
    if acceptable(fits, threshold, geometry_threshold):
        return fits, 'restricted_sweep'
    # Current cached camera/angles are deliberately discarded, including here.
    base = replace(observations, rvec=previous.rvec.copy(), tvec=previous.tvec.copy(),
                   layer_rotations=(), sticker_layer_rotations=None)
    expanded = geometry.fit_frame(renderer, base, prior=previous)
    combined = sorted([*fits, *expanded], key=lambda fit: (fit.score, fit.axis, fit.angles))
    return combined, 'exhaustive_previous_camera'


def rotation_degrees(first, second):
    a, b = cv2.Rodrigues(first.rvec)[0], cv2.Rodrigues(second.rvec)[0]
    return float(np.rad2deg(np.linalg.norm(cv2.Rodrigues(b @ a.T)[0])))


def describe(rows):
    successful = [r for r in rows if r['score'] is not None]
    times = np.asarray([r['milliseconds'] for r in rows])
    differences = np.asarray([r['score'] - r['reference_score'] for r in successful])
    return dict(frames=len(rows), failed_frames=len(rows)-len(successful),
        mean_ms=float(times.mean()), median_ms=float(np.median(times)),
        p95_ms=float(np.percentile(times, 95)),
        within_reference_plus_005=int(np.count_nonzero(differences <= .05)),
        within_reference_plus_010=int(np.count_nonzero(differences <= .10)),
        median_score_delta=float(np.median(differences)) if len(differences) else None,
        p95_score_delta=float(np.percentile(differences, 95)) if len(differences) else None,
        stages={stage: sum(r['stage'] == stage for r in rows)
                for stage in sorted({r['stage'] for r in rows})},
        quality_gate_passed_frames=sum(r['quality_gate_passed'] for r in rows))


def measure(current, previous, frame, iterations, *, adaptive=False, threshold=.025,
            geometry_threshold=.15):
    started = perf_counter()
    if adaptive:
        fits, stage = adaptive_reuse(current, previous, iterations=iterations,
            threshold=threshold, geometry_threshold=geometry_threshold)
    else:
        fits, stage = reuse_previous(current, previous, iterations=iterations), 'local'
    elapsed = 1000. * (perf_counter()-started)
    # Evaluation happens after timing and does not feed back into the tracker.
    rotations = current.layer_rotations
    axis = rotations[0].axis if rotations else 0
    x = np.r_[current.rvec.ravel(), current.tvec.ravel(),
              np.asarray(renderer.parallel_layer_angles(rotations))[[0, 2]]]
    reference = geometry._materialize(renderer, current, axis, x)
    best = fits[0] if fits else None
    row = dict(frame=frame, milliseconds=elapsed, reference_score=reference.score,
        score=None if best is None else best.score,
        stage=stage,
        quality_gate_passed=acceptable(fits, threshold, geometry_threshold),
        reprojection_fit=None if best is None else renderer.pose_reprojection_fit_metric(best.candidate),
        reference_reprojection_fit=renderer.pose_reprojection_fit_metric(reference.candidate),
        pose_rvec=None if best is None else best.candidate.rvec.tolist(),
        pose_tvec=None if best is None else best.candidate.tvec.tolist(),
        layer_rotations=[] if best is None else [renderer.layer_rotation_payload(r)
                                               for r in best.candidate.layer_rotations],
        reference_translation_step_mm=float(np.linalg.norm(current.tvec-previous.tvec)),
        reference_rotation_step_degrees=rotation_degrees(previous, current))
    return row, None if best is None else best.candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--samples', type=int, default=60)
    parser.add_argument('--iterations', type=int, default=12)
    parser.add_argument('--rollout', action='store_true')
    parser.add_argument('--adaptive', action='store_true')
    parser.add_argument('--reprojection-threshold', type=float, default=.025)
    parser.add_argument('--geometry-threshold', type=float, default=.15)
    parser.add_argument('--cases', nargs='+', choices=[c[0] for c in DEFAULT_CASES])
    args = parser.parse_args()
    if args.samples < 1 or args.iterations < 1:
        parser.error('--samples and --iterations must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    reports = []
    for name, directory, end in DEFAULT_CASES:
        if args.cases and name not in args.cases:
            continue
        path = ROOT/'runs'/directory/'observations.jsonl'
        meta = json.loads(path.with_suffix('.meta.json').read_text())
        start = meta.get('turning_start_frame') or 0
        rows = [json.loads(line) for line in path.open()]
        selected = [(r['frame_index'], saved_candidate(r)) for r in rows
                    if start <= r['frame_index'] <= end]
        pairs = [(previous, current) for previous, current in zip(selected, selected[1:])
                 if previous[1] is not None and current[1] is not None
                 and current[0] == previous[0]+1]
        indexes = np.unique(np.linspace(0, len(pairs)-1, min(args.samples, len(pairs)), dtype=int))
        measured = []
        for index in indexes:
            previous, current = pairs[index]
            row, _ = measure(current[1], previous[1], current[0], args.iterations,
                adaptive=args.adaptive, threshold=args.reprojection_threshold,
                geometry_threshold=args.geometry_threshold)
            measured.append(row)
        report = dict(name=name, log=str(path), first_frame=start, last_frame=end,
            source_fps=meta.get('source_fps', meta['fps']), one_step=describe(measured),
            one_step_rows=measured)
        print(json.dumps({k: v for k, v in report.items() if not k.endswith('_rows')}), flush=True)
        if args.rollout:
            rollout, previous = [], None
            bootstraps = []
            for frame, observations in selected:
                if observations is None:
                    continue
                if previous is None:
                    previous = observations
                    bootstraps.append(frame)
                    continue
                row, proposed = measure(observations, previous, frame, args.iterations,
                    adaptive=args.adaptive, threshold=args.reprojection_threshold,
                    geometry_threshold=args.geometry_threshold)
                rollout.append(row)
                previous = proposed or previous
            report.update(rollout=describe(rollout), rollout_rows=rollout,
                          bootstrap_frames=bootstraps)
            print(json.dumps(dict(name=name, rollout=report['rollout'],
                                  bootstrap_frames=bootstraps)), flush=True)
        reports.append(report)
        (args.output_dir/'report.json').write_text(json.dumps(dict(
            experiment='adaptive_previous_pose_search' if args.adaptive else 'three_local_solves_from_previous_pose',
            iterations=args.iterations,
            adaptive=args.adaptive, reprojection_threshold=args.reprojection_threshold,
            geometry_threshold=args.geometry_threshold,
            current_reference_pose_used_by_tracker=False,
            hard_angular_gate_enabled=False, inference_and_initialization_timed=False,
            accuracy_note='Geometry-cost agreement with saved fits, not ground truth. '
                'One-step trials use saved previous poses; rollouts do not.', cases=reports), indent=2)+'\n')


if __name__ == '__main__':
    import torch

    torch.set_num_threads(2)
    main()
