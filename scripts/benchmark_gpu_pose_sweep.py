"""Benchmark the exhaustive GPU sweep and unchanged CPU refinement on five clips.

Synchronized wall time includes transfers and Python option construction.
Full bank/wireframe comparisons run outside timing. No detector inference or
move reconstruction is included, and the CPU path remains the default.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import independent_keypoint_pose_geometry as geometry
from scripts import render_yolo11n_independent_keypoints as renderer
from scripts.analyze_orientation_motion import REFERENCE_SOLUTIONS
from scripts.benchmark_temporal_pose_reuse import saved_candidate
from scripts.benchmark_pose_geometry_mechanical import fingerprint
from scripts.independent_keypoint_gpu_sweep import parallel_turn_sweeps
from scripts.pose_wireframe_metrics import center_pixels, wire_mask


def synchronize(device):
    if device.startswith('mps'):
        torch.mps.synchronize()
    else:
        torch.cuda.synchronize()


def compare_sweeps(cpu, gpu):
    float_fields = ('keypoint_error', 'sticker_error', 'robust_error', 'p90_error')
    integer_fields = ('inlier_count', 'layer_inlier_count', 'keypoint_inlier_count',
                      'keypoint_layer_inlier_count', 'sticker_match_count', 'layer_support')
    stats = dict(configurations=0, finite_score_disagreements=0, diagnostic_count_disagreements=0,
                 valid_assignment_disagreements=0, nearest_assignment_disagreements=0,
                 max_balanced_score_difference=0., max_float_field_difference=0.)
    for expected, actual in zip(cpu, gpu, strict=True):
        options, nearest, valid = expected
        other, other_nearest, other_valid = actual
        assert len(options) == len(other) == 342
        stats['configurations'] += len(options)
        a = np.asarray([renderer.balanced_geometry_cost(o['keypoint_error'], o['sticker_error']) for o in options])
        b = np.asarray([renderer.balanced_geometry_cost(o['keypoint_error'], o['sticker_error']) for o in other])
        stats['finite_score_disagreements'] += int(np.count_nonzero(np.isfinite(a) != np.isfinite(b)))
        finite = np.isfinite(a) & np.isfinite(b)
        stats['max_balanced_score_difference'] = max(stats['max_balanced_score_difference'], float(np.max(np.abs(a[finite]-b[finite]), initial=0.)))
        for field in float_fields:
            x, y = np.asarray([o[field] for o in options]), np.asarray([o[field] for o in other])
            finite = np.isfinite(x) & np.isfinite(y)
            stats['max_float_field_difference'] = max(stats['max_float_field_difference'], float(np.max(np.abs(x[finite]-y[finite]), initial=0.)))
        for field in integer_fields:
            stats['diagnostic_count_disagreements'] += int(np.count_nonzero(np.asarray([o[field] for o in options]) != np.asarray([o[field] for o in other])))
        for x, y in zip(options, other, strict=True):
            assert x['layer_rotations'] == y['layer_rotations']
        stats['valid_assignment_disagreements'] += int(np.count_nonzero(valid != other_valid))
        stats['nearest_assignment_disagreements'] += int(np.count_nonzero((nearest != other_nearest) & valid & other_valid))
    return stats


def compare_banks(cpu, gpu):
    reference = cpu[0].candidate
    diameter = renderer.projected_cube_diameter(reference)
    center = center_pixels(reference, reference.tvec)
    cpu_masks = [wire_mask(f.candidate, center, diameter)[0] for f in cpu]
    gpu_masks = [wire_mask(f.candidate, center, diameter)[0] for f in gpu]
    cpu_distances = [cv2.distanceTransform(1-mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) for mask in cpu_masks]
    gpu_distances = [cv2.distanceTransform(1-mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) for mask in gpu_masks]
    errors = np.empty((len(cpu), len(gpu)))
    for i, a in enumerate(cpu_masks):
        for j, b in enumerate(gpu_masks):
            errors[i, j] = (max(np.percentile(gpu_distances[j][a.astype(bool)], 90),
                np.percentile(cpu_distances[i][b.astype(bool)], 90))/(256./2.4)
                if a.any() and b.any() else 1.)
    def parameters(fit):
        return np.r_[fit.candidate.rvec.ravel(), fit.candidate.tvec.ravel(), fit.angles]
    return dict(cpu_candidates=len(cpu), gpu_candidates=len(gpu), exact_bank_equal=fingerprint(cpu) == fingerprint(gpu),
        best_scored_wire_p90_fraction=float(errors[0, 0]),
        cpu_bank_coverage_within_002=float(np.mean(errors.min(axis=1) <= .02)),
        gpu_bank_coverage_within_002=float(np.mean(errors.min(axis=0) <= .02)),
        max_cpu_bank_best_wire_p90=float(errors.min(axis=1).max()),
        max_gpu_bank_best_wire_p90=float(errors.min(axis=0).max()),
        best_score_absolute_difference=abs(cpu[0].score-gpu[0].score),
        best_scored_pose_parameters_exact=bool(np.array_equal(parameters(cpu[0]), parameters(gpu[0]))))


def timed_pair(cpu_call, gpu_call, repeats, device):
    timings = {'cpu':[], 'gpu':[]}
    last = {}
    for repeat in range(repeats):
        order = [('cpu', cpu_call), ('gpu', gpu_call)]
        if repeat % 2:
            order.reverse()
        for name, call in order:
            synchronize(device)
            started = perf_counter()
            last[name] = call()
            synchronize(device)
            timings[name].append(1000.*(perf_counter()-started))
    a, b = statistics.median(timings['cpu']), statistics.median(timings['gpu'])
    return dict(cpu_ms=timings['cpu'], gpu_ms=timings['gpu'], cpu_median_ms=a, gpu_median_ms=b,
                speedup=a/b, reduction_fraction=1.-b/a), last


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('mps', 'cuda'), default='mps')
    parser.add_argument('--sweep-samples', type=int, default=12)
    parser.add_argument('--fit-samples', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable in this process; GPU benchmark must run outside the sandbox')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(parameters=vars(args) | dict(output_dir=str(args.output_dir)),
        source_sha256={path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in (
            'scripts/independent_keypoint_gpu_sweep.py', 'scripts/independent_keypoint_pose_geometry.py',
            'scripts/render_yolo11n_independent_keypoints.py')}, torch_version=torch.__version__,
        device=args.device, gpu_dtype='float32' if args.device == 'mps' else 'float64',
        scope='Fixed cached camera/observation workloads, exhaustive sweeps and CPU continuous refinement; no detector or move search.',
        timing='Synchronized, warmed, interleaved wall times include transfers and Python result construction. Quality checks excluded.', cases=[])
    if args.device == 'mps' and hasattr(torch.backends.mps, 'get_name'):
        report['gpu_name'] = torch.backends.mps.get_name()
    for name, solution_path in REFERENCE_SOLUTIONS.items():
        solution = json.loads((ROOT/solution_path).read_text())
        log = Path(solution['config']['log'])
        metadata = json.loads(log.with_suffix('.meta.json').read_text())
        start = metadata.get('turning_start_frame') or 0
        end = solution.get('solved_frame_index') or max(s['frame_index'] for s in solution['steps'])
        steps = {s['frame_index'] for s in solution['steps']}
        observations = []
        for line in log.open():
            record = json.loads(line)
            if not start <= record['frame_index'] <= end or record['frame_index'] not in steps:
                continue
            candidate = saved_candidate(record)
            if candidate is not None and len(candidate.assignment_image_points) >= 3 and len(candidate.sticker_mask_centers):
                observations.append((record['frame_index'], candidate))
        case = dict(name=name, source_log=str(log), sweeps=[], fits=[])
        indices = np.unique(np.linspace(1, len(observations)-2, args.sweep_samples, dtype=int))
        for index in indices:
            frame, candidate = observations[index]
            limit = renderer.PNP_FINAL_ASSIGNMENT_ERROR_FRACTION*renderer.projected_cube_diameter(candidate)
            cpu_call = lambda: [renderer.parallel_turn_frame_geometry(candidate, limit, a, middle_frame=True) for a in range(3)]
            gpu_call = lambda: parallel_turn_sweeps(renderer, candidate, limit, middle_frame=True, device=args.device)
            cpu, gpu = cpu_call(), gpu_call()
            quality = compare_sweeps(cpu, gpu)
            timings, _ = timed_pair(cpu_call, gpu_call, args.repeats, args.device)
            case['sweeps'].append(dict(frame=frame, **timings, **quality))
        fit_indices = np.unique(np.linspace(1, len(observations)-2, args.fit_samples+2, dtype=int)[1:-1])
        workloads = [(observations[i][0], observations[i][1], dict(prior=observations[i-1][1]), 'initial_geometry') for i in fit_indices]
        index = int(fit_indices[0])
        frame, candidate = observations[index]
        retained = geometry.fit_frame(renderer, candidate, prior=observations[index-1][1])
        neighbors = [geometry.fit_frame(renderer, observations[i][1])[0].candidate for i in (index-1, index+1)]
        workloads.append((frame, retained[0].candidate, dict(prior=neighbors[0], neighbors=neighbors, retained=retained), 'temporal_neighbors'))
        for frame, candidate, kwargs, stage in workloads:
            cpu_call = lambda: geometry.fit_frame(renderer, candidate, **kwargs)
            gpu_call = lambda: geometry.fit_frame(renderer, candidate, sweep_device=args.device, **kwargs)
            cpu, gpu = cpu_call(), gpu_call()
            quality = compare_banks(cpu, gpu)
            timings, _ = timed_pair(cpu_call, gpu_call, args.repeats, args.device)
            case['fits'].append(dict(frame=frame, stage=stage, **timings, **quality))
        report['cases'].append(case)
        (args.output_dir/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(dict(name=name,
            sweep_speedup=sum(r['cpu_median_ms'] for r in case['sweeps'])/sum(r['gpu_median_ms'] for r in case['sweeps']),
            fit_speedup=sum(r['cpu_median_ms'] for r in case['fits'])/sum(r['gpu_median_ms'] for r in case['fits']),
            best_wire_max=max(r['best_scored_wire_p90_fraction'] for r in case['fits']),
            exact_banks=sum(r['exact_bank_equal'] for r in case['fits']), fit_workloads=len(case['fits']))), flush=True)
    summary = {}
    for key in ('sweeps', 'fits'):
        rows = [row for case in report['cases'] for row in case[key]]
        a, b = sum(r['cpu_median_ms'] for r in rows), sum(r['gpu_median_ms'] for r in rows)
        summary[key] = dict(workloads=len(rows), cpu_median_total_ms=a, gpu_median_total_ms=b, speedup=a/b, reduction_fraction=1.-b/a)
    report['summary'] = summary
    (args.output_dir/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
