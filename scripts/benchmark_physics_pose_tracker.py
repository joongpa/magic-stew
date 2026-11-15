"""Evaluate a causal five-geometry tracker; reference fits never enter updates."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_temporal_pose_reuse import DEFAULT_CASES, saved_candidate
from scripts.independent_keypoint_physics_tracker import PhysicsPoseTracker
from scripts import independent_keypoint_pose_geometry as geometry
from scripts import render_yolo11n_independent_keypoints as renderer
from scripts.analyze_orientation_motion import REFERENCE_SOLUTIONS, selected_rotation
from scripts.independent_keypoint_color_tracks import colored_center_orientation, tracked_center_orientation
from scripts.pose_wireframe_metrics import center_pixels, wire_mask, wire_error


def reference_sticker_ids(record, step, state, initial_colors):
    """Evaluation-only physical IDs from the final color-matched solution."""
    from scripts import render_independent_keypoint_search as playback
    candidate = playback.candidate_for_step(renderer, record, step, state, initial_colors)
    if candidate is None:
        return []
    faces = renderer.projected_sticker_faces(candidate, candidate.rvec, candidate.tvec)
    directions, labels = [], []
    rotation = cv2.Rodrigues(candidate.rvec)[0]
    for identity in renderer.canonical_sticker_identities():
        if identity.category != 'center':
            continue
        cubie, axis, side = renderer.sticker_geometry_key(candidate, identity)
        center = renderer.cubelet_center_object_point(*cubie)
        polygon = renderer.cubelet_face_object_points(center, axis, side)
        _, transformed = renderer.apply_cubelet_layer_rotations(center, polygon,
            renderer.physical_sticker_rotations(candidate))
        directions.append(rotation @ transformed.mean(axis=0))
        labels.append(initial_colors[identity.label])
    physical_orientation = colored_center_orientation(directions, labels)
    record['_reference_center_orientation'] = physical_orientation.tolist() if physical_orientation is not None else None
    ids = []
    for point, color in zip(candidate.sticker_mask_centers, candidate.sticker_color_names, strict=True):
        choices = [(np.linalg.norm(np.mean(polygon, axis=0)-point), identity.label)
            for identity, _, polygon in faces
            if initial_colors[identity.label] == color and cv2.pointPolygonTest(
                np.asarray(polygon, np.float32), tuple(map(float, point)), False) >= 0]
        ids.append(min(choices)[1] if choices else None)
    return ids


def saved_fits(record):
    observation = saved_candidate(record)
    if observation is None:
        return []
    options = record['search']['geometry']['options']
    fits = []
    for option in options[:5]:
        rotations = tuple(renderer.CubeLayerRotation(**r) for r in option['layer_rotations'])
        axis = option['layer_axis']
        parameters = np.r_[np.asarray(option['pose_rvec']).ravel(),
            np.asarray(option['pose_tvec']).ravel(),
            np.asarray(renderer.parallel_layer_angles(rotations))[[0, 2]]]
        fit = geometry._materialize(renderer, observation, axis, parameters)
        if fit is not None:
            fits.append(fit)
    return fits


def final_selected_fit(record):
    """Evaluation target follows the solved path, including its axis correction."""
    step = record.get('_reference_step')
    if step is None:
        return saved_fits(record)[0]
    options = record['search']['geometry']['options']
    option = options[step.get('geometry_option_index', 0)]
    rotation = selected_rotation(option, step)
    layers = tuple(renderer.CubeLayerRotation(**r) for r in step.get('layer_rotations', option['layer_rotations']))
    axis = step.get('layer_axis', option['layer_axis'])
    angles = np.asarray(renderer.parallel_layer_angles(layers))
    rotation = rotation @ cv2.Rodrigues(np.eye(3)[axis]*np.deg2rad(angles[1]))[0]
    seed = np.r_[cv2.Rodrigues(rotation)[0].ravel(),
        np.asarray(step.get('pose_tvec', option['pose_tvec'])).ravel(),
        angles[[0, 2]]-angles[1]]
    return geometry._materialize(renderer, saved_candidate(record), axis, seed)


def bootstrap_fits(record):
    fits = saved_fits(record)
    if record.get('_reference_step') is not None:
        selected = final_selected_fit(record)
        fits = [selected, *fits]
    return fits


def evaluate_frame(tracker, record, time, frame=None, previous_reference_ids=None):
    observations = saved_candidate(record)
    active_quarter_branches = sum(b.quarter is not None and b.quarter.ready for b in tracker.branches)
    started = perf_counter()
    fits = tracker.update(observations, time, frame=frame)
    milliseconds = 1000.*(perf_counter()-started)
    color_stats = dict(tracker.last_color_diagnostics)
    current_ids = record.get('_reference_sticker_ids')
    previous_ids = previous_reference_ids
    if tracker.color_tracking and current_ids is not None and previous_ids is not None:
        correct = evaluated = 0
        for i, j in zip(tracker.last_color_matches.previous, tracker.last_color_matches.current, strict=True):
            if previous_ids[i] is not None and current_ids[j] is not None:
                evaluated += 1
                correct += previous_ids[i] == current_ids[j]
        color_stats.update(reference_evaluable_matches=evaluated, reference_correct_matches=correct)
    physical_errors = []
    if tracker.color_tracking and record.get('_reference_center_orientation') is not None:
        target = np.asarray(record['_reference_center_orientation'])
        for branch in tracker.branches:
            predicted = tracked_center_orientation(renderer, branch, tracker.color_tracker)
            if predicted is not None:
                physical_errors.append(float(np.rad2deg(np.linalg.norm(cv2.Rodrigues(predicted @ target.T)[0]))))
    color_stats['center_orientation_best_error_degrees'] = min(physical_errors) if physical_errors else None
    references = saved_fits(record)
    reference = final_selected_fit(record)
    references = [reference, *references]
    diameter = renderer.projected_cube_diameter(reference.candidate)
    distances = []
    primary_errors = []
    for target in references:
        errors = [np.sqrt(np.mean(np.sum((fit.candidate.assignment_projected_points-
            target.candidate.assignment_projected_points)**2, axis=1)))/diameter for fit in fits]
        distances.append(min(errors, default=float('inf')))
        if not primary_errors:
            primary_errors = errors
    low_error_reference_agreement = any(error <= .03 and renderer.pose_reprojection_fit_metric(fit.candidate) <= .035
        for fit, error in zip(fits, primary_errors, strict=True))
    # Compare complete rendered geometry after update timing. Assigned-point
    # agreement alone can miss visibly wrong orientation/articulation.
    target_mask = wire_mask(reference.candidate, center_pixels(reference.candidate, reference.candidate.tvec), diameter)
    wire_errors = [wire_error(fit.candidate, reference.candidate, target_mask) for fit in fits]
    return dict(frame=record['frame_index'], milliseconds=milliseconds, geometries=len(fits),
        best_score=fits[0].score if fits else None, reference_score=reference.score,
        reference_best_projected_rms_fraction=distances[0],
        reference_low_error_bank_agreement=low_error_reference_agreement,
        reference_wire_p90_best_scored=wire_errors[0] if wire_errors else None,
        reference_wire_p90_best_bank=min(wire_errors, default=None),
        reference_best_scored_projected_rms_fraction=primary_errors[0] if primary_errors else None,
        reference_bank_projected_rms_fractions=distances,
        reprojection_fit=renderer.pose_reprojection_fit_metric(fits[0].candidate) if fits else None,
        quarter_turn_prior=tracker.quarter_turn_prior,
        motion_prediction=dict(getattr(tracker, 'last_prediction_diagnostics', {})),
        active_quarter_branches=active_quarter_branches,
        color_tracking=color_stats,
        pose_bank=[dict(axis=f.axis, angles=list(f.angles), score=f.score,
                       rvec=f.candidate.rvec.tolist(), tvec=f.candidate.tvec.tolist()) for f in fits])


def summarize(rows):
    elapsed = np.asarray([r['milliseconds'] for r in rows])
    evaluated = sum(r.get('color_tracking', {}).get('reference_evaluable_matches', 0) for r in rows)
    correct = sum(r.get('color_tracking', {}).get('reference_correct_matches', 0) for r in rows)
    orientation_errors = [r['color_tracking']['center_orientation_best_error_degrees'] for r in rows
        if r.get('color_tracking', {}).get('center_orientation_best_error_degrees') is not None]
    return dict(frames=len(rows), mean_ms=float(elapsed.mean()), median_ms=float(np.median(elapsed)),
        p95_ms=float(np.percentile(elapsed, 95)), min_retained_geometries=min(r['geometries'] for r in rows),
        active_quarter_frames=sum(r['active_quarter_branches'] > 0 for r in rows),
        reference_full_wire_best_scored_within_002=sum(r.get('reference_wire_p90_best_scored', float('inf')) <= .02 for r in rows),
        reference_full_wire_best_bank_within_002=sum(r.get('reference_wire_p90_best_bank', float('inf')) <= .02 for r in rows),
        full_low_reprojection_frames=int(sum(r.get('reprojection_fit') is not None and r['reprojection_fit'] <= .035 for r in rows)),
        low_reprojection_reference_agreement_frames=sum(r.get('reference_low_error_bank_agreement', False) for r in rows),
        slab_final_accepted_frames=sum(r.get('motion_prediction', {}).get('final_full_fit_accepted', False) for r in rows),
        mean_static_search_ms=float(np.mean([r.get('motion_prediction', {}).get('static_ms', 0.) for r in rows])),
        mean_articulation_ms=float(np.mean([r.get('motion_prediction', {}).get('articulation_ms', 0.) for r in rows])),
        motion_recovery_frames=sum(r.get('motion_prediction', {}).get('recovery', False) for r in rows),
        mean_motion_refinements=float(np.mean([r.get('motion_prediction', {}).get('refinements', 0) for r in rows])),
        mean_motion_projection_evaluations=float(np.mean([r.get('motion_prediction', {}).get('projection_evaluations', 0) for r in rows])),
        color_fast_frames=sum(r.get('color_tracking', {}).get('fast_lane', False) for r in rows),
        mean_color_matches=float(np.mean([r.get('color_tracking', {}).get('matched_observations', 0) for r in rows])),
        mean_tracking_and_pnp_ms=float(np.mean([r.get('color_tracking', {}).get('tracking_and_pnp_ms', 0.) for r in rows])),
        reference_evaluable_color_matches=evaluated,
        reference_correct_color_matches=correct,
        reference_color_match_precision=correct/evaluated if evaluated else None,
        center_orientation_available_frames=len(orientation_errors),
        center_orientation_within_15_frames=int(np.count_nonzero(np.asarray(orientation_errors) <= 15.)),
        center_orientation_median_error_degrees=float(np.median(orientation_errors)) if orientation_errors else None,
        within_reference_cost_plus_005=int(sum(r['best_score'] <= r['reference_score']+.05 for r in rows)),
        reference_best_projected_rms_within_003=int(sum(r['reference_best_projected_rms_fraction'] <= .03 for r in rows)),
        reference_bank_coverage_003=float(np.mean([np.mean(np.asarray(r['reference_bank_projected_rms_fractions']) <= .03)
                                                 for r in rows])))


def limit_update_rate(records, source_fps, maximum_fps):
    """Keep real sampled timestamps, without inventing interpolated frames."""
    if maximum_fps <= 0:
        return records
    kept = []
    # Half a source-frame tolerance avoids float rounding selecting three
    # frames at 59.95 Hz when a two-frame interval is effectively 30 Hz.
    minimum_gap = 1./maximum_fps-.5/source_fps
    for record in records:
        if not kept or (record['frame_index']-kept[-1]['frame_index'])/source_fps >= minimum_gap:
            kept.append(record)
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=2)
    parser.add_argument('--extra-proposals', type=int, default=3)
    parser.add_argument('--reversion', type=float, default=2.)
    parser.add_argument('--settle-seconds', type=float, default=.75)
    parser.add_argument('--tail-seconds', type=float, default=.5)
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--rollout', action='store_true')
    parser.add_argument('--cases', nargs='+', choices=[c[0] for c in DEFAULT_CASES])
    parser.add_argument('--quarter-turn-prior', action='store_true')
    parser.add_argument('--quarter-sigma-degrees', type=float, default=25.)
    parser.add_argument('--quarter-weight', type=float, default=4.)
    parser.add_argument('--reference-source', choices=('final-selected', 'bank-rank0'), default='final-selected')
    parser.add_argument('--color-tracking', action='store_true')
    parser.add_argument('--color-fast-lane', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--motion-prediction', action='store_true', help='Adaptive continuous-motion search; no color/discrete prior')
    parser.add_argument('--slab-hierarchy', action='store_true', help='Separate static best-slab hierarchy and fresh articulation search')
    parser.add_argument('--static-beam', type=int, default=8)
    parser.add_argument('--slab-all-axes', action='store_true', help='Decouple static slab ranking from articulated axis/anchor selection')
    parser.add_argument('--prediction-damping', type=float, default=.25)
    parser.add_argument('--max-update-fps', type=float, default=0., help='Subsample observations to this approximate rate; zero keeps all')
    args = parser.parse_args()
    if args.motion_prediction and (args.color_tracking or args.quarter_turn_prior):
        parser.error('Motion prediction is evaluated independently of color and quarter-turn priors')
    if args.slab_hierarchy and (args.motion_prediction or args.color_tracking or args.quarter_turn_prior):
        parser.error('Slab hierarchy is evaluated separately from other tracker modes')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(parameters=vars(args) | dict(output_dir=str(args.output_dir)),
        tracker_source_sha256=hashlib.sha256((ROOT/'scripts/independent_keypoint_physics_tracker.py').read_bytes()).hexdigest(),
        color_tracker_source_sha256=hashlib.sha256((ROOT/'scripts/independent_keypoint_color_tracks.py').read_bytes()).hexdigest(),
        motion_tracker_source_sha256=hashlib.sha256((ROOT/'scripts/independent_keypoint_motion_prediction.py').read_bytes()).hexdigest(),
        slab_tracker_source_sha256=hashlib.sha256((ROOT/'scripts/independent_keypoint_slab_search.py').read_bytes()).hexdigest(),
        scope='Conservative central solve windows. Fitted references are not ground-truth poses.',
        timing='Pose update only; inference, cached bootstrap, evaluation, and move search excluded.',
        beam_policy='Five distinct geometries using the existing score and diversity test; hard cap five.',
        current_reference_pose_used_by_update=False, cases=[])
    report['wireframe_evaluation'] = 'After update only: symmetric p90 full rendered edge distance at 256px, normalized by reference diameter; no color identity.'
    for name, directory, end in DEFAULT_CASES:
        if args.cases and name not in args.cases:
            continue
        solution = (json.loads((ROOT/REFERENCE_SOLUTIONS[name]).read_text())
                    if args.reference_source == 'final-selected' else None)
        path = Path(solution['config']['log']) if solution else ROOT/'runs'/directory/'observations.jsonl'
        meta = json.loads(path.with_suffix('.meta.json').read_text())
        fps = meta.get('source_fps', meta['fps'])
        start = (meta.get('turning_start_frame') or 0)+int(np.ceil(args.settle_seconds*fps))
        if solution:
            end = min(end, solution.get('solved_frame_index') or max(s['frame_index'] for s in solution['steps']))
        end -= int(np.ceil(args.tail_seconds*fps))
        records = [json.loads(line) for line in path.open()]
        records = [r for r in records if start <= r['frame_index'] <= end and saved_candidate(r) is not None]
        if solution:
            steps = {s['frame_index']:s for s in solution['steps']}
            records = [dict(r, _reference_step=steps[r['frame_index']]) for r in records
                       if r['frame_index'] in steps]
        records = limit_update_rate(records, fps, args.max_update_fps)
        if solution:
            if args.color_tracking:
                from scripts.search_independent_keypoint_full_solve import initial_sticker_colors
                initial_colors = initial_sticker_colors(renderer, solution['config']['initial_scramble'])
                state = renderer.CubeGeometryState.identity()
                by_frame = {r['frame_index']:r for r in records}
                for step in solution['steps']:
                    record = by_frame.get(step['frame_index'])
                    if record is not None:
                        record['_reference_sticker_ids'] = reference_sticker_ids(record, step, state, initial_colors)
                    state = state.apply_moves(tuple(step.get('state_moves', ())))
        capture = cv2.VideoCapture(meta['source']) if args.color_tracking else None
        video_position = -1

        def source_frame(record):
            nonlocal video_position
            if capture is None:
                return None
            target = record['frame_index']
            if not 0 < target-video_position <= 5:
                capture.set(cv2.CAP_PROP_POS_FRAMES, target)
                video_position = target-1
            frame = None
            while video_position < target:
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f'Cannot read source frame {target}')
                video_position += 1
            return frame

        def new_tracker():
            if args.slab_hierarchy:
                from scripts.independent_keypoint_slab_search import SlabHierarchyTracker
                return SlabHierarchyTracker(renderer, beam=args.static_beam, iterations=args.iterations,
                    damping=args.prediction_damping, all_axes=args.slab_all_axes)
            if args.motion_prediction:
                from scripts.independent_keypoint_motion_prediction import PredictivePoseTracker
                return PredictivePoseTracker(renderer, iterations=args.iterations, damping=args.prediction_damping)
            return PhysicsPoseTracker(renderer, reversion=args.reversion,
                iterations=args.iterations, proposals_to_refine=args.extra_proposals,
                quarter_turn_prior=args.quarter_turn_prior,
                quarter_sigma_degrees=args.quarter_sigma_degrees, quarter_weight=args.quarter_weight,
                color_tracking=args.color_tracking, color_fast_lane=args.color_fast_lane)

        samples = []
        indices = np.unique(np.linspace(1, len(records)-1, min(args.samples, len(records)-1), dtype=int))
        for i in indices:
            previous, current = records[i-1:i+1]
            fits = bootstrap_fits(previous)
            if len(geometry.diverse_shortlist(renderer, fits)) < 5:
                continue
            tracker = new_tracker()
            tracker.initialize(fits, previous['frame_index']/fps, frame=source_frame(previous))
            samples.append(evaluate_frame(tracker, current, current['frame_index']/fps,
                frame=source_frame(current), previous_reference_ids=previous.get('_reference_sticker_ids')))
        case = dict(name=name, fps=fps, source_log=str(path),
            effective_update_fps=float(1./np.median(np.diff([r['frame_index']/fps for r in records]))),
            source_solution=REFERENCE_SOLUTIONS[name] if solution else None,
            first_frame=records[0]['frame_index'],
            last_frame=records[-1]['frame_index'], one_step=summarize(samples), one_step_rows=samples)
        print(json.dumps(dict(name=name, one_step=case['one_step'])), flush=True)
        if args.rollout:
            tracker, rows, bootstrap, previous_ids = new_tracker(), [], None, None
            for record in records:
                time = record['frame_index']/fps
                if bootstrap is None:
                    fits = bootstrap_fits(record)
                    if len(geometry.diverse_shortlist(renderer, fits)) < 5:
                        continue
                    tracker.initialize(fits, time, frame=source_frame(record))
                    previous_ids = record.get('_reference_sticker_ids')
                    bootstrap = record['frame_index']
                    continue
                rows.append(evaluate_frame(tracker, record, time, frame=source_frame(record),
                    previous_reference_ids=previous_ids))
                previous_ids = record.get('_reference_sticker_ids')
            case.update(bootstrap_frame=bootstrap, rollout=summarize(rows), rollout_rows=rows)
            print(json.dumps(dict(name=name, bootstrap_frame=bootstrap, rollout=case['rollout'])), flush=True)
        report['cases'].append(case)
        if capture is not None:
            capture.release()
        (args.output_dir/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    import torch

    torch.set_num_threads(2)
    main()
