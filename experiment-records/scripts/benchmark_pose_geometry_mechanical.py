"""Time the established pose fitter with exact complete-bank equivalence checks.

Load a saved pre-change module, interleave baseline/optimized timing on identical
real inputs, and compare every exported value (including floating-point bytes).
No search pruning, iteration budget or pose-quality objective is changed.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import independent_keypoint_pose_geometry as optimized
from scripts import render_yolo11n_independent_keypoints as renderer
from scripts.analyze_orientation_motion import REFERENCE_SOLUTIONS
from scripts.benchmark_temporal_pose_reuse import saved_candidate


def canonical(value):
    if isinstance(value, np.ndarray):
        return dict(dtype=value.dtype.str, shape=value.shape, bytes=value.tobytes().hex())
    if isinstance(value, np.generic):
        return canonical(value.item())
    if dataclasses.is_dataclass(value):
        return {field.name:canonical(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key):canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [canonical(item) for item in value]
    if isinstance(value, float):
        return value.hex()
    return value


def fingerprint(bank):
    encoded = json.dumps(canonical(bank), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(encoded.encode()).hexdigest()


def load_baseline(path):
    spec = importlib.util.spec_from_file_location('pose_geometry_mechanical_baseline', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--samples-per-case', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    baseline = load_baseline(args.baseline_source)
    report = dict(baseline_source=str(args.baseline_source.resolve()),
        baseline_sha256=hashlib.sha256(args.baseline_source.read_bytes()).hexdigest(),
        optimized_sha256=hashlib.sha256((ROOT/'scripts/independent_keypoint_pose_geometry.py').read_bytes()).hexdigest(),
        scope='Established joint geometry and temporal-neighbor fitting on fixed real-frame inputs; not end-to-end detector/solve throughput.',
        policy='Same exhaustive sweep, constraints, scoring, diversity/ties, iteration budgets and candidate order. No hypothesis pruning.',
        verification='SHA256 of complete PoseFit dataclasses and nested candidate/option fields; arrays include dtype, shape and exact bytes.',
        repeats=args.repeats, cases=[])
    for name, solution_path in REFERENCE_SOLUTIONS.items():
        solution = json.loads((ROOT/solution_path).read_text())
        log = Path(solution['config']['log'])
        metadata = json.loads(log.with_suffix('.meta.json').read_text())
        start = metadata.get('turning_start_frame') or 0
        end = solution.get('solved_frame_index') or max(step['frame_index'] for step in solution['steps'])
        observations = []
        for line in log.open():
            record = json.loads(line)
            if not start <= record['frame_index'] <= end:
                continue
            candidate = saved_candidate(record)
            if candidate is not None and len(candidate.assignment_image_points) >= 3 and len(candidate.sticker_mask_centers):
                observations.append((record['frame_index'], candidate))
        indices = np.unique(np.linspace(1, len(observations)-2, args.samples_per_case+2, dtype=int)[1:-1])
        workloads = []
        for index in indices:
            frame, candidate = observations[index]
            prior = observations[index-1][1]
            workloads.append(dict(stage='initial_geometry', frame=frame, candidate=candidate,
                                  kwargs=dict(prior=prior)))
        # The temporal pass preserves already-fitted source candidates and
        # sweeps only neighboring cameras, exactly as the existing pipeline.
        index = int(indices[0])
        frame, candidate = observations[index]
        retained = baseline.fit_frame(renderer, candidate, prior=observations[index-1][1])
        neighbors = [baseline.fit_frame(renderer, observations[i][1])[0].candidate for i in (index-1, index+1)]
        workloads.append(dict(stage='temporal_neighbors', frame=frame, candidate=retained[0].candidate,
            kwargs=dict(prior=neighbors[0], neighbors=neighbors, retained=retained)))
        measured = []
        for workload in workloads:
            # Warm both implementations before measuring this fixed workload.
            expected = baseline.fit_frame(renderer, workload['candidate'], **workload['kwargs'])
            actual = optimized.fit_frame(renderer, workload['candidate'], **workload['kwargs'])
            expected_hash = fingerprint(expected)
            assert fingerprint(actual) == expected_hash, (name, workload['stage'], workload['frame'])
            times = {'baseline':[], 'optimized':[]}
            for repeat in range(args.repeats):
                order = [('baseline', baseline), ('optimized', optimized)]
                if repeat % 2:
                    order.reverse()
                for label, module in order:
                    started = perf_counter()
                    bank = module.fit_frame(renderer, workload['candidate'], **workload['kwargs'])
                    times[label].append(1000.*(perf_counter()-started))
                    # Serialization and correctness checks are outside timing.
                    assert fingerprint(bank) == expected_hash, (name, workload['stage'], workload['frame'], label, repeat)
            before, after = statistics.median(times['baseline']), statistics.median(times['optimized'])
            row = dict(stage=workload['stage'], frame=workload['frame'], retained_candidates=len(expected),
                exact_bank_equal=True, bank_sha256=expected_hash, baseline_ms=times['baseline'], optimized_ms=times['optimized'],
                baseline_median_ms=before, optimized_median_ms=after, reduction_fraction=1.-after/before)
            measured.append(row)
            print(json.dumps(dict(name=name, **{k:v for k,v in row.items() if k not in ('bank_sha256','baseline_ms','optimized_ms')})), flush=True)
        report['cases'].append(dict(name=name, source_log=str(log), measurements=measured))
        (args.output_dir/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    rows = [row for case in report['cases'] for row in case['measurements']]
    report['summary'] = dict(workloads=len(rows), all_banks_exactly_equal=all(row['exact_bank_equal'] for row in rows),
        total_baseline_median_ms=sum(row['baseline_median_ms'] for row in rows),
        total_optimized_median_ms=sum(row['optimized_median_ms'] for row in rows),
        aggregate_reduction_fraction=1.-sum(row['optimized_median_ms'] for row in rows)/sum(row['baseline_median_ms'] for row in rows))
    (args.output_dir/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    import torch
    torch.set_num_threads(2)
    main()
