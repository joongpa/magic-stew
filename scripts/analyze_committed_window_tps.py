"""Compare committed quarter-turn counts with rolling angular-travel TPS."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def committed_qtm(move):
    # These reconstructions contain face and wide turns only. Reject other
    # notation instead of silently choosing a slice/rotation counting policy.
    if re.fullmatch(r"[UDLRFB]w?(?:2|')?", move) is None:
        raise ValueError(f'Unsupported committed move: {move}')
    return 2 if move.endswith('2') else 1


def committed_peaks(intervals, fps, max_frames):
    if not 1 <= max_frames <= len(intervals):
        raise ValueError('Window range must fit within the trajectory')
    prefix = [0]
    for row in intervals:
        prefix.append(prefix[-1]+row['committed_qtm'])
    result = []
    for count in range(1, max_frames+1):
        sums = [prefix[i+count]-prefix[i] for i in range(len(intervals)-count+1)]
        assert sums == [sum(r['committed_qtm'] for r in intervals[i:i+count])
                        for i in range(len(sums))]
        best = max(range(len(sums)), key=sums.__getitem__)
        window = intervals[best:best+count]
        angular_tps = sum(r['angular_travel_degrees'] for r in window)*fps/(90.*count)
        result.append(dict(window_frames=count, duration_ms=1000.*count/fps,
            committed_qtm=sums[best], peak_committed_tps=sums[best]*fps/count,
            start_frame=window[0]['start_frame'], end_frame=window[-1]['end_frame'],
            angular_tps_in_committed_peak_window=angular_tps,
            eligible_windows=len(sums)))
    return result


def main():
    from scripts.analyze_peak_window_tps import DEFAULT_PATHS, load_intervals, rolling_peaks

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    cases = {}
    for name, path in DEFAULT_PATHS.items():
        case = load_intervals(path)
        data = json.loads(path.read_text())
        steps = {s['frame_index']: s for s in data['steps']}
        tokens = Counter()
        for interval in case['intervals']:
            moves = steps[interval['end_frame']]['state_moves']
            interval.update(committed_moves=moves,
                committed_qtm=sum(committed_qtm(move) for move in moves))
            tokens.update(moves)
        fps = case['source_fps']
        max_frames = 60 if name == 'matty' else 30
        peaks = committed_peaks(case['intervals'], fps, max_frames)
        angular_peaks = rolling_peaks(case['intervals'], fps, max_frames)
        for committed, angular in zip(peaks, angular_peaks, strict=True):
            committed.update(peak_angular_tps=angular['peak_tps'],
                angular_peak_start_frame=angular['start_frame'],
                angular_peak_end_frame=angular['end_frame'])
        qtm = sum(i['committed_qtm'] for i in case['intervals'])
        assert qtm == sum(committed_qtm(move)*count for move, count in tokens.items())
        # Include the entire analyzed span, matching the angular study exactly.
        seconds = (case['end_frame']-case['start_frame'])/fps
        angular_total = sum(i['angular_travel_degrees'] for i in case['intervals'])/90.
        case.update(peaks=peaks, totals=dict(committed_tokens=sum(tokens.values()),
            committed_qtm=qtm, angular_quarter_turn_equivalents=angular_total,
            angular_excess_equivalents=angular_total-qtm, duration_seconds=seconds,
            committed_average_tps=qtm/seconds, angular_average_tps=angular_total/seconds,
            move_histogram=dict(tokens)))
        cases[name] = case
    method = (
        'Count saved state_moves at their commit frames: quarter and inverse turns = 1, '
        'half turns = 2, wide quarter turns = 1. No move cancellation, merging, smoothing, '
        'or relocation in time. Split quarter commits are each counted once. '
        'These paths contain only quarter face/wide tokens; no slice or whole-cube tokens. '
        'For interval (a,b], count commits at frames a+1 through b and divide by (b-a)/fps. '
        'Angular-travel comparisons use the identical saved paths and frame boundaries. '
        'Each metric selects its own peak window; same-window angular rates are also saved. '
        'Discrete commit rates can exceed 54 TPS over short windows because physical '
        'travel is distributed across neighboring frames. Fitted angular travel also '
        'includes residual motion and reversals that need not produce a committed move.'
    )
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    (output/'analysis.json').write_text(json.dumps(dict(method=method, cases=cases), indent=2)+'\n')
    with (output/'peak-comparison.csv').open('w', newline='') as stream:
        columns = ['solve', *next(iter(cases.values()))['peaks'][0]]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for name, case in cases.items():
            writer.writerows(dict(solve=name, **row) for row in case['peaks'])
    lines = ['Committed QTM versus angular-travel TPS', '', method, '']
    for name, case in cases.items():
        lines.extend([name.title(), json.dumps(case['totals'], indent=2),
            'Frames  Duration(ms)  Committed TPS  Angular peak TPS  Committed peak endpoints'])
        lines.extend(f"{p['window_frames']:>6}  {p['duration_ms']:>12.3f}  "
            f"{p['peak_committed_tps']:>13.3f}  {p['peak_angular_tps']:>16.3f}  "
            f"{p['start_frame']} -> {p['end_frame']}" for p in case['peaks'])
        lines.append('')
    (output/'report.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for ax, (name, case) in zip(axes, cases.items(), strict=True):
        peaks = case['peaks']
        durations = [p['duration_ms'] for p in peaks]
        ax.plot(durations, [p['peak_committed_tps'] for p in peaks], 'o-',
                markersize=3, label='Committed QTM/s')
        ax.plot(durations, [p['peak_angular_tps'] for p in peaks], '--',
                linewidth=1.8, label='Angular-travel equivalents/s')
        ax.set(title=f"{name.title()} ({case['source_fps']:.2f} fps)",
            xlabel='Window duration (ms)', ylabel='Maximum rolling TPS', ylim=(0, 65))
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    fig.suptitle('Committed quarter-turn counts versus fitted angular travel')
    fig.savefig(output/'committed-vs-angular-tps.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
