"""Measure peak rolling angular-travel TPS in saved Matty and Yiheng paths.

One window frame means one source-frame interval (N intervals, N+1 endpoints).
Uses the search cap's pooled travel metric, including residual motion, rather
than counting committed move tokens. Does not modify the saved reconstructions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_PATHS = {
    'matty': ROOT/'runs/search-throughput-20261113/cross-solve-54-54/matty/result.json',
    'yiheng': ROOT/'runs/yihengCropped-independent-keypoint-search-118-tps54-shared54-sparse3-300s/'
        'corrected-26-27-87-94/result.json',
}


def load_intervals(path):
    from scripts import render_yolo11n_independent_keypoints as renderer
    from scripts.independent_keypoint_rotation_budget import FLAT, RotationGeometry, routes_between

    data = json.loads(path.read_text())
    config = data['config']
    fps = float(config.get('observation_input', {}).get('source_fps', config['fps']))
    if data['status'] != 'solved' or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f'Expected a solved reconstruction and positive source FPS: {path}')
    if config['max_tps'] != config['max_double_tps']:
        raise ValueError('Route audit currently requires equal single/double TPS caps')
    start, end = config['turning_start_frame'], data['solved_frame_index']
    steps = {step['frame_index']: step for step in data['steps']}
    if len(steps) != len(data['steps']):
        raise ValueError('Duplicate frame indices')
    # Reject skipped observations instead of treating a longer interval as one frame.
    if any(frame not in steps for frame in range(start, end+1)):
        raise ValueError('Analysis requires consecutive source-frame observations')
    # Check the actual source indices, including logs that renumber sampled frames.
    log = Path(config['log'])
    for line in log.open():
        record = json.loads(line)
        frame = record['frame_index']
        if start <= frame <= end and record.get('original_source_frame', frame) != frame:
            raise ValueError('Renumbered source frames require explicit time resampling')
    previous = FLAT
    intervals, restored = [], []
    for frame in range(start+1, end+1):
        step = steps[frame]
        if step.get('orientation_symmetry') is not None:
            raise ValueError('Route audit requires a consistent physical coordinate frame')
        rotations = tuple(renderer.CubeLayerRotation(**r) for r in step['layer_rotations'])
        current = RotationGeometry.make(step['layer_axis'], renderer.parallel_layer_angles(rotations))
        limit = 90.*config['max_tps']/fps
        matches = []
        for route in routes_between(previous, current, limit):
            moves = tuple(renderer.cube_rotation_to_notation(rotation)
                for axis, offsets in route.groups
                for rotation in renderer.parallel_layer_rotations(axis, tuple(90.*q for q in offsets)))
            if moves == tuple(step['state_moves']):
                matches.append(route.degrees)
        if not matches:
            raise ValueError(f'No legal route for selected moves at frame {frame}')
        degrees = min(matches)
        if 'angular_travel_degrees' in step:
            if not math.isclose(degrees, step['angular_travel_degrees'], abs_tol=1e-6):
                raise ValueError(f'Saved travel disagrees with reconstructed route at {frame}')
            degrees = float(step['angular_travel_degrees'])
            origin = 'saved'
        elif step.get('manual_correction'):
            restored.append(dict(frame=frame, angular_travel_degrees=degrees))
            origin = 'recomputed_manual_correction'
        else:
            raise ValueError(f'Unexplained missing travel at frame {frame}')
        intervals.append(dict(start_frame=frame-1, end_frame=frame,
            angular_travel_degrees=degrees, tps=degrees*fps/90., origin=origin))
        previous = current
    return dict(source_result=str(path.resolve()), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_fps=fps, start_frame=start, end_frame=end,
        original_cap_tps=config['max_tps'], audited_intervals=len(intervals),
        restored_manual_correction_travel=restored, intervals=intervals)


def rolling_peaks(intervals, fps, max_frames):
    if not 1 <= max_frames <= len(intervals):
        raise ValueError('Window range must fit within the available trajectory')
    prefix = [0.]
    for interval in intervals:
        prefix.append(prefix[-1]+interval['angular_travel_degrees'])
    peaks = []
    for count in range(1, max_frames+1):
        sums = [prefix[i+count]-prefix[i] for i in range(len(intervals)-count+1)]
        # Independently enumerate every window to check indexing and prefix sums.
        direct = [math.fsum(row['angular_travel_degrees'] for row in intervals[i:i+count])
                  for i in range(len(sums))]
        if not all(math.isclose(a, b, abs_tol=1e-8) for a, b in zip(sums, direct)):
            raise AssertionError('Rolling window cross-check failed')
        best = max(range(len(sums)), key=sums.__getitem__)
        degrees = direct[best]
        peaks.append(dict(window_frames=count, duration_ms=1000.*count/fps,
            peak_tps=degrees*fps/(90.*count), angular_travel_degrees=degrees,
            quarter_turn_equivalents=degrees/90., start_frame=intervals[best]['start_frame'],
            end_frame=intervals[best+count-1]['end_frame'], eligible_windows=len(sums)))
    return peaks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    for name in DEFAULT_PATHS:
        parser.add_argument(f'--{name}-result', type=Path, default=DEFAULT_PATHS[name])
        parser.add_argument(f'--{name}-max-frames', type=int, default=60 if name == 'matty' else 30)
    args = parser.parse_args()
    cases = {}
    for name in DEFAULT_PATHS:
        case = load_intervals(getattr(args, f'{name}_result'))
        case['peaks'] = rolling_peaks(case['intervals'], case['source_fps'],
                                     getattr(args, f'{name}_max_frames'))
        cases[name] = case
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    method = (
        'TPS = sum(selected route angular_travel_degrees) / 90 / elapsed seconds. '
        'N frames means N source-frame intervals, with N+1 endpoint poses. '
        'Travel pools absolute outer-slab angles relative to the middle slab; '
        'simultaneous relative-slab travel adds, and rigid cube rotation is excluded. '
        'This matches the shared 54 TPS search cap, not HTM move-event TPS. '
        'All saved interval costs were independently checked against legal routes. '
        'Four missing costs in the corrected Yiheng path were recomputed from its edited moves. '
        'These are maxima in fitted, already cap-constrained paths; geometry noise can add travel. '
        'They do not establish unconstrained human-speed limits or safe caps for other solves.'
    )
    (output/'analysis.json').write_text(json.dumps(dict(method=method, cases=cases), indent=2)+'\n')
    for filename, field in [('peak-tps.csv', 'peaks'), ('interval-tps.csv', 'intervals')]:
        with (output/filename).open('w', newline='') as stream:
            columns = ['solve', *next(iter(cases.values()))[field][0]]
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for name, case in cases.items():
                writer.writerows(dict(solve=name, **row) for row in case[field])
    lines = ['Peak rolling angular-travel TPS', '', method, '']
    for name, case in cases.items():
        lines.extend([f"{name.title()}: {case['source_fps']:.8f} fps; frames "
                      f"{case['start_frame']} -> {case['end_frame']}",
                      'Window  Duration(ms)  Peak TPS  Endpoint frames'])
        lines.extend(f"{p['window_frames']:>6}  {p['duration_ms']:>12.3f}  {p['peak_tps']:>8.3f}  "
                     f"{p['start_frame']} -> {p['end_frame']}" for p in case['peaks'])
        lines.append('')
    (output/'report.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for name, case in cases.items():
        peaks = case['peaks']
        label = f"{name.title()} ({case['source_fps']:.2f} fps)"
        y = [p['peak_tps'] for p in peaks]
        axes[0].plot([p['window_frames'] for p in peaks], y, 'o-', label=label, markersize=4)
        axes[1].plot([p['duration_ms'] for p in peaks], y, 'o-', label=label, markersize=4)
    for ax, xlabel in zip(axes, ('Window length (source-frame intervals)', 'Window duration (ms)')):
        ax.axhline(54., color='#777777', linestyle='--', linewidth=1, label='Existing cap: 54 TPS')
        ax.set(xlabel=xlabel, ylabel='Peak angular-travel TPS', ylim=(0, 60))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    longest = max(len(case['peaks']) for case in cases.values())
    tick_step = max(1, math.ceil(longest/6))
    axes[0].set_xticks(sorted({1, longest, *range(tick_step, longest+1, tick_step)}))
    fig.suptitle('Maximum rolling TPS in saved reconstructions')
    fig.savefig(output/'peak-tps.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
