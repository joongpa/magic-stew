"""Counterfactual rolling TPS gates on saved paths; does not rerun or change search."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ['0.25s', '8frames', '0.5s', '1s', '2s']
EPS = 1e-8


def padded_cap(phase_tps, additive=2.0):
    return min(54., max(0., 1.25 * phase_tps + additive))


def burst_cap(phase_tps, seconds, quarters, additive=2.0):
    """A fixed travel allowance, converted to the evaluated window's TPS units."""
    return min(54., padded_cap(phase_tps, additive) + quarters / seconds)


def gate_summary(rows, cut_key):
    """A single violating window cuts the saved path; windows are not independent."""
    cases = sorted({r['case'] for r in rows})
    cuts = [r for r in rows if r[cut_key]]
    return {'evaluated_windows': len(rows), 'violating_windows': len(cuts),
            'captures_with_a_saved_path_cut': sorted({r['case'] for r in cuts}),
            'saved_capture_paths_preserved': len(cases) - len({r['case'] for r in cuts}),
            'saved_capture_paths_total': len(cases)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('curves', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--burst-turns', type=float, default=0.)
    args = parser.parse_args()
    if args.burst_turns < 0:
        parser.error('--burst-turns must be nonnegative')
    manifest = json.loads(args.manifest.read_text())
    curves = list(csv.DictReader(args.curves.open()))
    edges, durations, family_by_case = {}, {}, {}
    for item in manifest['cases']:
        data = json.loads((ROOT / item['input']).read_text())
        config, steps = data['config'], data['result']['steps']
        fps = config['fps']
        hold = config.get('turning_start_frame')
        start = max(steps[0]['frame_index'] - 1, hold if hold is not None else -float('inf')) / fps
        previous = start
        previous_flagged = False
        frames = {f['frame']: f for f in data['frames']}
        for step in steps:
            end = step['frame_index'] / fps
            if end <= start:
                continue
            frame = frames[step['frame_index']]
            if frame['bank']:
                edges[(item['name'], step['frame_index'])] = {
                    'interval_tps': step['angular_travel_degrees'] / 90 / (end - previous),
                    'geometry_guard': previous_flagged or frame['geometry_flagged'],
                }
                previous = end
                previous_flagged = frame['geometry_flagged']
            else:
                edges[(item['name'], step['frame_index'])] = {'interval_tps': 0., 'geometry_guard': True}
        durations[item['name']] = steps[-1]['frame_index'] / fps - start
        family_by_case[item['name']] = item['family']
    evaluated = []
    for r in curves:
        if r['window'] not in WINDOWS or r['full_window'] != 'True':
            continue
        frame = int(float(r['frame']))
        edge = edges[(r['case'], frame)]
        phase = float(r['primary_snapped6_travel'])
        reference = float(r['reference_angular'])
        cap = burst_cap(phase, float(r['duration_seconds']), args.burst_turns)
        guard_cap = 54. if edge['geometry_guard'] else cap
        evaluated.append({'case': r['case'], 'family': r['family'], 'frame': frame,
            'window': r['window'], 'end_seconds': float(r['end_seconds']),
            'duration_seconds': float(r['duration_seconds']), 'phase_tps': phase,
            'reference_window_tps': reference, 'reference_interval_tps': edge['interval_tps'],
            'window_cap_tps': cap, 'rolling_window_violation': reference > cap + EPS,
            'per_interval_violation': edge['interval_tps'] > cap + EPS,
            'guarded_rolling_window_violation': reference > guard_cap + EPS,
            'guarded_per_interval_violation': edge['interval_tps'] > guard_cap + EPS,
            'geometry_guard': edge['geometry_guard'],
            'headroom_tps': cap - reference})
    report = {'scope': __doc__, 'rule': 'min(54, 1.25 * same-window >6-degree phase-change TPS + 2 + burst_turns/window_seconds)',
        'burst_quarter_turns': args.burst_turns,
        'reference': 'Saved successful path angular travel, not manually labeled physical TPS. Cutting one saved path does not prove the search cannot find another valid path.',
        'warmup': 'No dynamic gate until a complete window exists; retain the original 54 TPS ceiling during warmup.',
        'mechanical_interpretation': 'Rolling constraints apply to accumulated angular travel over the same window. A rolling average is not a valid replacement for the per-interval 54 TPS ceiling.',
        'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        'curves_sha256': hashlib.sha256(args.curves.read_bytes()).hexdigest(),
        'windows': {}, 'cross_validation': {}}
    for window in WINDOWS:
        sample = [r for r in evaluated if r['window'] == window]
        by_family = defaultdict(list)
        by_case = defaultdict(list)
        for row in sample:
            by_family[row['family']].append(row)
            by_case[row['case']].append(row)
        active_coverage = []
        for cases in [{k: v for k, v in by_case.items() if family_by_case[k] == family} for family in by_family]:
            per_case = []
            for rs in cases.values():
                active = [r for r in rs if r['reference_window_tps'] >= 1.]
                per_case.append(statistics.mean(not r['rolling_window_violation'] for r in active))
            active_coverage.append(statistics.mean(per_case))
        warmup = {name: max(0., durations[name] - rs[0]['duration_seconds']) / durations[name] for name, rs in by_case.items()}
        usable_by_family = defaultdict(list)
        for name, fraction in warmup.items():
            usable_by_family[family_by_case[name]].append(fraction)
        out = {key: gate_summary(sample, key) for key in [
            'rolling_window_violation', 'per_interval_violation',
            'guarded_rolling_window_violation', 'guarded_per_interval_violation']}
        out.update({'active_window_coverage_equal_family': statistics.mean(active_coverage),
            'mean_cap_tps_equal_family': statistics.mean(statistics.mean(r['window_cap_tps'] for r in rs) for rs in by_family.values()),
            'maximum_window_shortfall_tps': max(0., max(-r['headroom_tps'] for r in sample)),
            'mean_fraction_solve_after_warmup_equal_family': statistics.mean(statistics.mean(v) for v in usable_by_family.values()),
            'median_window_duration_seconds': statistics.median(r['duration_seconds'] for r in sample),
            'ideal_boxcar_half_step_response_seconds': statistics.median(r['duration_seconds'] for r in sample) / 2,
            'violation_examples': sorted([r for r in sample if r['rolling_window_violation']], key=lambda r: r['headroom_tps'])[:8]})
        report['windows'][window] = out
        # Test whether a margin covering every other family's observed windows
        # transfers. The held family's labels never select its additive margin.
        folds = []
        for family in by_family:
            train = [r for r in sample if r['family'] != family]
            test = [dict(r) for r in sample if r['family'] == family]
            margin = max(2., max(r['reference_window_tps'] - 1.25 * r['phase_tps'] - args.burst_turns / r['duration_seconds'] for r in train))
            for r in test:
                r['held_out_cut'] = r['reference_window_tps'] > burst_cap(r['phase_tps'], r['duration_seconds'], args.burst_turns, margin) + EPS
            folds.append({'held_out_family': family, 'training_max_additive_margin': margin,
                          **gate_summary(test, 'held_out_cut')})
        report['cross_validation'][window] = folds
        print(window, 'rolling paths', out['rolling_window_violation']['saved_capture_paths_preserved'], '/10',
            'violations', out['rolling_window_violation']['violating_windows'],
            'direct interval paths', out['per_interval_violation']['saved_capture_paths_preserved'], '/10',
            'mean cap', round(out['mean_cap_tps_equal_family'], 1),
            'usable fraction', round(out['mean_fraction_solve_after_warmup_equal_family'], 2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    with args.output.with_suffix('.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(evaluated[0])); writer.writeheader(); writer.writerows(evaluated)


if __name__ == '__main__':
    main()
