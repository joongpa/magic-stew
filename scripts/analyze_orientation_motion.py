"""Orientation trends in trusted, color-matched reference trajectories.

The primary analysis preserves the recorded physical/color frame: no cube
symmetry relabeling or layer-angle correction is applied to its rotations.
Cube-symmetry reduction is a separately labeled geometry-only diagnostic.
All forecast features are causal; held-out parameters use the other four clips.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_temporal_pose_reuse import DEFAULT_CASES
from scripts import render_yolo11n_independent_keypoints as renderer


REFERENCE_SOLUTIONS = {
    'matty': 'runs/search-throughput-20261113/cross-solve-54-54/matty/result.json',
    'nahm': 'runs/search-throughput-20261113/cross-solve-54-54/nahm/result.json',
    'xuanyi': 'runs/search-throughput-20261113/cross-solve-54-54/xuanyi/result.json',
    'tommy': 'runs/split-tps-40-50-validation/tommy-short-fresh/search-30fps/result.json',
    'yiheng': 'runs/yihengCropped-independent-keypoint-search-118-tps54-shared54-sparse3-300s/'
        'corrected-26-27-87-94/result.json',
}


def rotation_mean(matrices):
    """Proper chordal mean, preserving the supplied color/axis labels."""
    u, _, vt = np.linalg.svd(np.asarray(matrices).sum(axis=0))
    return u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt


def rotation_error(predicted, actual):
    relative = np.asarray(actual) @ np.swapaxes(np.asarray(predicted), -1, -2)
    return np.rad2deg(np.linalg.norm(Rotation.from_matrix(relative).as_rotvec(), axis=-1))


def symmetry_align(matrices, reference):
    """Geometry-only diagnostic. Never used in physical-frame predictions."""
    matrices = np.asarray(matrices)
    scalar = matrices.ndim == 2
    matrices = matrices.reshape(-1, 3, 3)
    references = np.broadcast_to(reference, matrices.shape)
    alternatives = matrices[:, None] @ np.asarray(renderer.CUBE_SYMMETRY_MATRICES)[None]
    scores = np.einsum('nsij,nij->ns', alternatives, references)
    selected = scores.argmax(axis=1)
    aligned = alternatives[np.arange(len(matrices)), selected]
    return (aligned[0] if scalar else aligned), selected


def selected_rotation(option, step):
    """Use the same selected pose and right-axis correction as saved replay."""
    rotation = cv2.Rodrigues(np.asarray(step.get('pose_rvec', option['pose_rvec'])))[0]
    if step.get('orientation_symmetry') is not None:
        rotation = rotation @ np.asarray(step['orientation_symmetry']).T
    return rotation


def load_case(directory, end, settle_seconds, tail_seconds, solution_path=None):
    solution = json.loads(Path(solution_path).read_text()) if solution_path else None
    path = Path(solution['config']['log']) if solution else ROOT/'runs'/directory/'observations.jsonl'
    meta = json.loads(path.with_suffix('.meta.json').read_text())
    fps = meta.get('source_fps', meta['fps'])
    start = (meta.get('turning_start_frame') or 0)+int(np.ceil(settle_seconds*fps))
    if solution:
        end = min(end, solution.get('solved_frame_index') or max(s['frame_index'] for s in solution['steps']))
    end -= int(np.ceil(tail_seconds*fps))
    frames, rotations, axes, angles = [], [], [], []
    steps = {s['frame_index']:s for s in solution['steps']} if solution else {}
    for line in path.open():
        record = json.loads(line)
        options = ((record.get('search') or {}).get('geometry') or {}).get('options') or []
        if not start <= record['frame_index'] <= end or not options:
            continue
        if solution and record['frame_index'] not in steps:
            continue
        step = steps.get(record['frame_index'], {})
        best = options[step.get('geometry_option_index', 0)]
        rotations.append(selected_rotation(best, step))
        frames.append(record['frame_index'])
        layers = tuple(renderer.CubeLayerRotation(**r) for r in step.get('layer_rotations', best['layer_rotations']))
        offsets = np.asarray(renderer.parallel_layer_angles(layers))[[0, 2]]
        angles.append(offsets)
        axes.append(step.get('layer_axis', best['layer_axis']) if np.max(np.abs(offsets)) >= 5. else -1)
    return dict(frames=np.asarray(frames), times=np.asarray(frames)/fps,
        rotations=np.asarray(rotations), axes=np.asarray(axes), angles=np.asarray(angles), fps=fps,
        source_log=str(path), source_solution=str(solution_path) if solution_path else None)


def causal_features(case):
    times, rotations = case['times'], case['rotations']
    rows = []
    initial = rotation_mean(rotations[:5])
    for i in range(1, len(times)-1):
        dt_previous, dt = times[i]-times[i-1], times[i+1]-times[i]
        if dt_previous > .08 or dt > .08:
            continue
        start = int(np.searchsorted(times, times[i]-.25))
        anchor = rotation_mean(rotations[start:i+1])
        short = max(0, int(np.searchsorted(times, times[i]-.08)))
        if short == i:
            short = i-1
        velocities = Rotation.from_matrix(rotations[short+1:i+1] @
            rotations[short:i].transpose(0, 2, 1)).as_rotvec()/np.diff(times[short:i+1])[:, None]
        # The initial anchor must be causal even during its five-pose warmup.
        initial_now = initial if i >= 4 else rotation_mean(rotations[:i+1])
        # Learn the lattice of preferred quarter-turn orientations causally.
        quarter_anchor = rotations[0]
        for _ in range(3):
            quarter_anchor = rotation_mean(symmetry_align(rotations[:i+1], quarter_anchor)[0])
        wells = quarter_anchor @ np.asarray(renderer.CUBE_SYMMETRY_MATRICES)
        nearest = wells[int(np.argmin(rotation_error(wells, rotations[i])))]
        rows.append(dict(index=i, target=i+1, dt=dt, rotation=rotations[i], actual=rotations[i+1],
            velocity=Rotation.from_matrix(rotations[i] @ rotations[i-1].T).as_rotvec()/dt_previous,
            smooth_velocity=np.median(velocities, axis=0),
            local_return=Rotation.from_matrix(anchor @ rotations[i].T).as_rotvec(),
            initial_return=Rotation.from_matrix(initial_now @ rotations[i].T).as_rotvec(),
            quarter_return=Rotation.from_matrix(nearest @ rotations[i].T).as_rotvec()))
    return {key: np.asarray([row[key] for row in rows]) for key in rows[0]}


def prediction(features, damping=0., reversion=0., center='local', smooth=False):
    velocity = features['smooth_velocity' if smooth else 'velocity']
    tangent = damping*features['dt'][:, None]*velocity
    tangent += (1.-np.exp(-reversion*features['dt']))[:, None]*features[center+'_return']
    return Rotation.from_rotvec(tangent).as_matrix() @ features['rotation']


def errors(features, config):
    return rotation_error(prediction(features, **config), features['actual'])


def select_config(training):
    candidates = [dict(damping=a, reversion=k, center=center, smooth=smooth)
        for a in (0., .25, .5, .75, 1.) for k in (0., 1., 2., 4., 8., 16.)
        for center in ('local', 'initial', 'quarter') for smooth in (False, True)]
    return min(candidates, key=lambda cfg: np.mean([errors(features, cfg).mean() for features in training]))


def distribution(values):
    values = np.asarray(values)
    return dict(mean=float(np.mean(values)), median=float(np.median(values)),
        p90=float(np.percentile(values, 90)), p95=float(np.percentile(values, 95)),
        p99=float(np.percentile(values, 99)), maximum=float(np.max(values)))


def analyze(name, case, features, training):
    rotations, times = case['rotations'], case['times']
    dt = np.diff(times)
    contiguous = (dt > 0) & (dt <= .08)
    changes = rotation_error(rotations[:-1], rotations[1:])
    aligned, mode = symmetry_align(rotations[1:], rotations[:-1])
    geometry_changes = rotation_error(rotations[:-1], aligned)
    anchor = rotation_mean(rotations)
    deviation = rotation_error(anchor, rotations)
    initial = rotation_mean(rotations[:5])
    to_initial = rotation_error(initial, rotations)
    # Estimate the most populated geometric working orientation separately;
    # its 24 color-frame modes are reported, not collapsed in primary metrics.
    geometry_anchor = rotations[0]
    for _ in range(5):
        geometry_anchor = rotation_mean(symmetry_align(rotations, geometry_anchor)[0])
    aligned, modes = symmetry_align(rotations, geometry_anchor)
    geometry_deviation = rotation_error(geometry_anchor, aligned)
    counts = sorted(np.bincount(modes, minlength=24).tolist(), reverse=True)
    velocity = Rotation.from_matrix(rotations[1:] @ rotations[:-1].transpose(0, 2, 1)).as_rotvec()/dt[:, None]
    v0, v1 = velocity[:-1], velocity[1:]
    nonzero = (np.linalg.norm(v0, axis=1) > 1e-6) & (np.linalg.norm(v1, axis=1) > 1e-6)
    cosine = np.sum(v0[nonzero]*v1[nonzero], axis=1)/(np.linalg.norm(v0[nonzero], axis=1)*np.linalg.norm(v1[nonzero], axis=1))
    covariance = np.cov(velocity.T)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    switch = (case['axes'][1:] != case['axes'][:-1])
    seam = ((case['axes'][1:] == case['axes'][:-1]) &
        np.any(np.abs(np.diff(case['angles'], axis=0)) > 60., axis=1))
    config = select_config(training)
    methods = dict(stationary=dict(damping=0.), constant_velocity=dict(damping=1.),
        half_velocity=dict(damping=.5), held_out_return=config)
    forecast = {method: distribution(errors(features, parameters)) for method, parameters in methods.items()}
    # Return direction at 100 ms is measured relative to a past-only local anchor.
    inward, eligible = [], 0
    for i in range(1, len(times)):
        target = int(np.searchsorted(times, times[i]+.1))
        if target >= len(times):
            continue
        start = int(np.searchsorted(times, times[i]-.25))
        local_anchor = rotation_mean(rotations[start:i+1])
        current_distance = float(rotation_error(local_anchor, rotations[i]))
        if current_distance >= 5.:
            eligible += 1
            inward.append(float(rotation_error(local_anchor, rotations[target])) < current_distance)
    large = changes > 20.
    result = dict(name=name, first_frame=int(case['frames'][0]), last_frame=int(case['frames'][-1]),
        frames=len(rotations), fps=case['fps'],
        observed_step_rate_hz=float(1./np.median(dt)),
        source_log=case.get('source_log'), source_solution=case.get('source_solution'),
        physical_color_frame_step_degrees=distribution(changes[contiguous]),
        physical_step_as_30fps_degrees=distribution((changes/(30.*dt))[contiguous]),
        physical_step_exceeds_90deg_at_30fps_count=int(np.count_nonzero(changes > 2700.*dt)),
        physical_working_mean_deviation_degrees=distribution(deviation),
        physical_initial_working_deviation_degrees=distribution(to_initial),
        geometry_only_symmetry_step_degrees=distribution(geometry_changes[contiguous]),
        geometry_only_working_deviation_degrees=distribution(geometry_deviation),
        geometry_working_mode_top1_fraction=counts[0]/len(rotations),
        geometry_working_mode_top3_fraction=sum(counts[:3])/len(rotations),
        quarter_turn_mode_residual_degrees=distribution(geometry_deviation),
        quarter_turn_mode_within_degrees={str(limit):float(np.mean(geometry_deviation <= limit))
                                         for limit in (10, 15, 20, 25, 30, 40, 45)},
        quarter_turn_mode_occupancy_counts=counts,
        angular_velocity_next_step_cosine_median=float(np.median(cosine)),
        angular_velocity_reversal_fraction=float(np.mean(cosine < 0)),
        angular_motion_principal_variance_fractions=(eigenvalues/eigenvalues.sum()).tolist(),
        large_step_count=int(np.count_nonzero(large)),
        large_steps_at_axis_or_seam_transition=int(np.count_nonzero(large & (switch | seam))),
        axis_or_seam_transition_count=int(np.count_nonzero(switch | seam)),
        closer_to_past_working_orientation_after_100ms_fraction=float(np.mean(inward)) if inward else None,
        return_direction_eligible_frames=eligible, selected_forecast=config, forecast=forecast,
        held_out_forecast_mean_improvement_fraction=1.-forecast['held_out_return']['mean']/forecast['stationary']['mean'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--settle-seconds', type=float, default=.75)
    parser.add_argument('--tail-seconds', type=float, default=.5)
    parser.add_argument('--source', choices=('final-selected', 'bank-rank0'), default='final-selected')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cases = {name: load_case(directory, end, args.settle_seconds, args.tail_seconds,
                            ROOT/REFERENCE_SOLUTIONS[name] if args.source == 'final-selected' else None)
             for name, directory, end in DEFAULT_CASES}
    features = {name: causal_features(case) for name, case in cases.items()}
    report = dict(source=args.source,
        primary_frame='Final selected pose including saved orientation correction; no symmetry relabeling.',
        diagnostic_frame='24-fold geometry-only symmetry reductions are reported separately.',
        phase_selection='Conservative central windows; not annotated solve boundaries.',
        forecast_validation='Causal features, leave-one-clip-out parameter selection.', cases=[])
    # For radii <=45 degrees, the 24 SO(3) balls are disjoint. This is a
    # geometric null comparison, not a speed or pose-accuracy measurement.
    report['uniform_orientation_quarter_mode_within_degrees'] = {
        str(degrees):float(24.*(np.deg2rad(degrees)-np.sin(np.deg2rad(degrees)))/np.pi)
        for degrees in (10, 15, 20, 25, 30, 40, 45)}
    for name, case in cases.items():
        result = analyze(name, case, features[name], [f for key, f in features.items() if key != name])
        report['cases'].append(result)
        print(json.dumps(result), flush=True)
    (args.output_dir/'analysis.json').write_text(json.dumps(report, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(5, 2, figsize=(12, 12))
    for row, (name, case) in enumerate(cases.items()):
        times = case['times']-case['times'][0]
        rotations = case['rotations']
        axes[row, 0].plot(times[1:], rotation_error(rotations[:-1], rotations[1:]),
                          color='#2b658c', linewidth=1.4)
        axes[row, 0].set_title(f'{name}: rotation between final selected poses')
        axes[row, 0].set_ylabel('step (degrees)')
        f = features[name]
        axes[row, 1].plot(times[f['target']], errors(f, dict(damping=0.)), label='hold orientation', alpha=.8)
        cfg = report['cases'][row]['selected_forecast']
        axes[row, 1].plot(times[f['target']], errors(f, cfg), label='held-out predictor', alpha=.8)
        axes[row, 1].set_title(f'{name}: next-frame physical-frame error')
        axes[row, 1].set_ylabel('error (degrees)')
        axes[row, 1].legend(fontsize=8)
        for ax in axes[row]:
            ax.set_xlabel('seconds within selected solve window')
            ax.grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(args.output_dir/'orientation_motion.png', dpi=150)


if __name__ == '__main__':
    main()
