"""Causal, leave-one-clip-out tests of damped core-position prediction.

Saved fitted translations are reference estimates, not motion-capture truth.
Every prediction uses only positions at or before the source frame. Parameters
for a held-out clip are chosen using the other clips, never its future poses.
"""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_temporal_pose_reuse import DEFAULT_CASES


def load_trajectory(directory, end, settle_seconds=0., tail_seconds=0.):
    path = ROOT/'runs'/directory/'observations.jsonl'
    meta = json.loads(path.with_suffix('.meta.json').read_text())
    original_start = meta.get('turning_start_frame') or 0
    fps = meta.get('source_fps', meta['fps'])
    start = original_start + int(np.ceil(settle_seconds*fps))
    end = end-int(np.ceil(tail_seconds*fps))
    frames, positions, image_positions, errors = [], [], [], []
    for line in path.open():
        record = json.loads(line)
        options = ((record.get('search') or {}).get('geometry') or {}).get('options') or []
        if not start <= record['frame_index'] <= end or not options:
            continue
        best = options[0]
        translation = np.asarray(best['pose_tvec']).reshape(3)
        pose = record.get('search_pose') or record['selected_pose']
        uv = cv2.projectPoints(np.zeros((1, 3)), np.asarray(best['pose_rvec']), translation,
            np.asarray(pose['camera_matrix']), np.asarray(pose['dist_coeffs']))[0].reshape(2)
        frames.append(record['frame_index'])
        positions.append(translation)
        image_positions.append(uv)
        errors.append(best.get('keypoint_mean_error'))
    frames = np.asarray(frames)
    return dict(frames=frames, times=frames/meta.get('source_fps', meta['fps']),
        positions=np.asarray(positions), image_positions=np.asarray(image_positions),
        fps=fps, path=str(path), original_start_frame=original_start,
        analyzed_first_frame=int(frames[0]), analyzed_last_frame=int(frames[-1]))


def causal_features(times, positions, horizon_frames=1):
    """Past-only initial/local anchors and robust recent velocities."""
    result = []
    for i in range(1, len(times)-horizon_frames):
        target = i+horizon_frames
        dt_previous, dt = times[i]-times[i-1], times[target]-times[i]
        if dt_previous > .08 or dt > .08*horizon_frames:
            continue
        history = positions[np.maximum(0, np.searchsorted(times, times[i]-.5)):i+1]
        short_start = max(0, np.searchsorted(times, times[i]-.08))
        short_t = times[short_start:i+1]-times[i]
        short = positions[short_start:i+1]
        slope = np.linalg.lstsq(np.column_stack((np.ones(len(short_t)), short_t)), short,
                                rcond=None)[0][1]
        result.append(dict(index=i, target=target, dt=dt, position=positions[i],
            velocity=(positions[i]-positions[i-1])/dt_previous,
            smooth_velocity=slope, local_center=np.median(history, axis=0),
            initial_center=np.median(positions[:min(i+1, 5)], axis=0), actual=positions[target]))
    return result


def predict(row, damping, reversion, center='local_center', smooth=False):
    velocity = row['smooth_velocity' if smooth else 'velocity']
    return (row['position'] + damping*row['dt']*velocity
            + (1.-np.exp(-reversion*row['dt']))*(row[center]-row['position']))


def errors(rows, config):
    return np.asarray([np.linalg.norm(predict(row, **config)-row['actual']) for row in rows])


def fit_config(training):
    candidates = [dict(damping=a, reversion=k, center=center, smooth=smooth)
        for a, k, center, smooth in product((0., .25, .5, .75, 1.), (0., 1., 2., 4., 8., 16.),
                                          ('local_center', 'initial_center'), (False, True))]
    return min(candidates, key=lambda cfg: np.mean([errors(rows, cfg).mean() for rows in training]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--settle-seconds', type=float, default=0.)
    parser.add_argument('--tail-seconds', type=float, default=0.)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    data = {name: load_trajectory(directory, end, args.settle_seconds, args.tail_seconds)
            for name, directory, end in DEFAULT_CASES}
    features = {name: causal_features(d['times'], d['positions']) for name, d in data.items()}
    report = dict(settle_seconds=args.settle_seconds, tail_seconds=args.tail_seconds,
        segmentation='Conservative central-window sensitivity test, not annotated solve start/end.',
        interpretation='Core translations are fitted estimates. Predictive mean reversion '
        'can reflect pose noise as well as real hand/cube motion. Held-out evaluation uses only past poses.',
        predictor='p_next = p + a*dt*v + (1-exp(-k*dt))*(center-p)', cases=[])
    for name, trajectory in data.items():
        rows = features[name]
        config = fit_config([features[other] for other in features if other != name])
        baselines = dict(stationary=dict(damping=0., reversion=0.),
            constant_velocity=dict(damping=1., reversion=0.),
            damped_velocity=dict(damping=.5, reversion=0.),
            held_out_physics=config)
        metrics = {}
        for model, cfg in baselines.items():
            distances = errors(rows, cfg)
            metrics[model] = dict(mean_mm=float(distances.mean()), median_mm=float(np.median(distances)),
                p90_mm=float(np.percentile(distances, 90)), p95_mm=float(np.percentile(distances, 95)))
        stationary = errors(rows, baselines['stationary'])
        physics = errors(rows, config)
        position = trajectory['positions']
        initial = np.median(position[:5], axis=0)
        inward = []
        for i, t in enumerate(trajectory['times']):
            j = np.searchsorted(trajectory['times'], t+.1)
            if j < len(position):
                inward.append(np.linalg.norm(position[j]-initial) < np.linalg.norm(position[i]-initial))
        row = dict(name=name, frames=len(position), transitions=len(rows), source_fps=trajectory['fps'],
            first_frame=trajectory['analyzed_first_frame'], last_frame=trajectory['analyzed_last_frame'],
            initial_position_mm=initial.tolist(),
            initial_distance_p50_p90_mm=np.percentile(np.linalg.norm(position-initial, axis=1), [50, 90]).tolist(),
            coordinate_span_p05_p95_mm=(np.percentile(position, 95, axis=0)-np.percentile(position, 5, axis=0)).tolist(),
            closer_to_initial_after_100ms_fraction=float(np.mean(inward)),
            selected_config=config, prediction=metrics,
            physics_better_than_stationary_fraction=float(np.mean(physics < stationary)),
            physics_mean_error_reduction_fraction=float(1.-physics.mean()/stationary.mean()))
        report['cases'].append(row)
        print(json.dumps(row), flush=True)
    (args.output_dir/'analysis.json').write_text(json.dumps(report, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(5, 2, figsize=(13, 13))
    for row, (name, trajectory) in enumerate(data.items()):
        times = trajectory['times']-trajectory['times'][0]
        position = trajectory['positions']
        initial = np.median(position[:5], axis=0)
        for axis, label in enumerate(('camera X', 'camera Y', 'depth Z')):
            axes[row, 0].plot(times, position[:, axis]-initial[axis], label=label, linewidth=1.)
        axes[row, 0].set_title(f'{name}: core offset from first five poses')
        axes[row, 0].set_ylabel('offset (mm)')
        axes[row, 0].legend(fontsize=8, ncol=3)
        features_row = features[name]
        config = report['cases'][row]['selected_config']
        target_times = [trajectory['times'][r['target']]-trajectory['times'][0] for r in features_row]
        axes[row, 1].plot(target_times, errors(features_row, dict(damping=0., reversion=0.)),
                          label='previous position', alpha=.7, linewidth=1.)
        axes[row, 1].plot(target_times, errors(features_row, config),
                          label='held-out physics', alpha=.8, linewidth=1.)
        axes[row, 1].set_title(f'{name}: next-frame prediction error')
        axes[row, 1].set_ylabel('error (mm)')
        axes[row, 1].legend(fontsize=8)
        for ax in axes[row]:
            ax.set_xlabel('seconds after active segment begins')
            ax.grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(args.output_dir/'core_motion.png', dpi=150)


if __name__ == '__main__':
    main()
