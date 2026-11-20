"""Profile sweep, continuous refinement and materialization on reference inputs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import independent_keypoint_pose_geometry as geometry
from scripts import render_yolo11n_independent_keypoints as renderer
from scripts.benchmark_gpu_pose_sweep import synchronize
from scripts.benchmark_temporal_pose_reuse import saved_candidate


def profile(candidate, kwargs, device):
    totals, counts = defaultdict(float), defaultdict(int)
    names = {'sweep':'swept_geometries', 'refinement':'_refine', 'materialization':'_materialize'}
    originals = {label:getattr(geometry, name) for label, name in names.items()}
    def timed(label, function):
        def call(*args, **kwargs):
            started = perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                totals[label] += 1000.*(perf_counter()-started)
                counts[label] += 1
        return call
    synchronize('mps')
    started = perf_counter()
    try:
        for label, name in names.items():
            setattr(geometry, name, timed(label, originals[label]))
        result = geometry.fit_frame(renderer, candidate, sweep_device=device, **kwargs)
        synchronize('mps')
    finally:
        for label, name in names.items():
            setattr(geometry, name, originals[label])
    elapsed = 1000.*(perf_counter()-started)
    return dict(total_ms=elapsed, **{k+'_ms':v for k,v in totals.items()},
                other_ms=elapsed-sum(totals.values()), calls=dict(counts), candidates=len(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS must be available to profile the implemented GPU path')
    source = json.loads(args.benchmark.read_text())
    report = dict(benchmark=str(args.benchmark), repeats=args.repeats,
        scope='Direct non-overlapping wall times in fit_frame; fixed initial and temporal workloads per reference. No detector, orientation, depth recovery or move search.',
        prior_source='Immediately previous/next available cached observation in each log; frames selected from benchmark. These additional profile workloads use adjacent-log priors, rather than the selected-solution-step priors in the timing benchmark.',
        cases=[])
    for case in source['cases']:
        observations = []
        for record in map(json.loads, Path(case['source_log']).open()):
            candidate = saved_candidate(record)
            if candidate is not None:
                observations.append((record['frame_index'], candidate))
        by_frame = {frame:i for i,(frame,candidate) in enumerate(observations)}
        initial = next(f for f in case['fits'] if f['stage']=='initial_geometry')
        i = by_frame[initial['frame']]
        candidate = observations[i][1]
        prior = observations[i-1][1]
        retained = geometry.fit_frame(renderer, candidate, prior=prior)
        neighbors = [geometry.fit_frame(renderer, observations[j][1])[0].candidate for j in (i-1,i+1)]
        workloads = [('initial_geometry',candidate,dict(prior=prior)),
            ('temporal_neighbors',retained[0].candidate,dict(prior=neighbors[0],neighbors=neighbors,retained=retained))]
        rows = []
        for stage,candidate,kwargs in workloads:
            for device in ('cpu','mps'):
                geometry.fit_frame(renderer,candidate,sweep_device=device,**kwargs)
            samples = defaultdict(list)
            for repeat in range(args.repeats):
                for device in (('cpu','mps') if repeat%2==0 else ('mps','cpu')):
                    samples[device].append(profile(candidate,kwargs,device))
            for device,measurements in samples.items():
                row = dict(stage=stage,frame=initial['frame'],device=device,measurements=measurements)
                row['medians'] = {field:statistics.median(m[field] for m in measurements)
                    for field in ('total_ms','sweep_ms','refinement_ms','materialization_ms','other_ms')}
                rows.append(row)
        report['cases'].append(dict(name=case['name'],profiles=rows))
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(case['name'],json.dumps([dict(stage=r['stage'],device=r['device'],**r['medians']) for r in rows]),flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
