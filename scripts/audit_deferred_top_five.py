"""Check strict top-five reference geometry recall separately from cutoff ties."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import render_yolo11n_independent_keypoints as renderer
from scripts.analyze_orientation_motion import REFERENCE_SOLUTIONS
from scripts.benchmark_temporal_pose_reuse import saved_candidate
from scripts.benchmark_physics_pose_tracker import final_selected_fit
from scripts.pose_wireframe_metrics import center_pixels,wire_mask


def evaluate_bank(base,bank,reference):
    diameter=renderer.projected_cube_diameter(reference)
    center=center_pixels(reference,reference.tvec)
    target,scale=wire_mask(reference,center,diameter)
    to_target=cv2.distanceTransform(1-target,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    errors=[]
    for p in bank:
        rotations=renderer.parallel_layer_rotations(p['axis'],p['angles'])
        candidate=replace(base,rvec=np.asarray(p['rvec']).reshape(3,1),tvec=np.asarray(p['tvec']).reshape(3,1),
            layer_rotations=rotations,sticker_layer_rotations=rotations)
        actual,_=wire_mask(candidate,center,diameter)
        if not actual.any() or not target.any():error=1.
        else:
            to_actual=cv2.distanceTransform(1-actual,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            error=float(max(np.percentile(to_actual[target.astype(bool)],90),np.percentile(to_target[actual.astype(bool)],90)))/(scale*diameter)
        errors.append(error)
    ranks=[i+1 for i,e in enumerate(errors) if e<=.02]
    return dict(errors=errors,top_five_match=any(e<=.02 for e in errors[:5]),
        top_five_within_001=any(e<=.01 for e in errors[:5]),top_five_raster_exact=any(e==0 for e in errors[:5]),
        all_retained_match=bool(ranks),first_matching_rank=min(ranks) if ranks else None,
        closest_top_five_error=min(errors[:5],default=None),closest_retained_error=min(errors,default=None))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=json.loads(args.benchmark.read_text())
    review=json.loads((args.benchmark.parent/'regressions/review.json').read_text())
    flagged={(r['name'],r['frame']) for r in review['regressions'] if r['material_best_regression'] or r['bank_candidate_lost']}
    result=dict(criterion='Symmetric p90 full wireframe distance <= 2% selected-reference diameter. Strict first five in existing score order, separately from cutoff ties. Geometry only; no physical color-identity evaluation.',cases=[])
    for case in report['cases']:
        name=case['name'];solution=json.loads((ROOT/REFERENCE_SOLUTIONS[name]).read_text())
        records={r['frame_index']:r for r in map(json.loads,Path(case['source_log']).open())}
        steps={s['frame_index']:s for s in solution['steps']}
        rows=[]
        for row in map(json.loads,(args.benchmark.parent/name/'frames.jsonl').open()):
            if not row['reference_available']:continue
            record=records[row['frame']];base=saved_candidate(record)
            target=final_selected_fit(dict(record,_reference_step=steps[row['frame']])).candidate
            output=dict(frame=row['frame'],flagged=(name,row['frame']) in flagged)
            for label in ['baseline','deferred']:
                bank=row[label+'_bank']
                assert all(bank[i]['score']<=bank[i+1]['score']+1e-12 for i in range(len(bank)-1))
                output[label]=evaluate_bank(base,bank,target)
                expected=row[label]['closest_wire'];actual=output[label]['closest_retained_error']
                if expected is not None:np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=1e-7)
            rows.append(output)
        def summary(subset,label):
            return dict(frames=len(subset),top_five_matches=sum(r[label]['top_five_match'] for r in subset),
                top_five_within_001=sum(r[label]['top_five_within_001'] for r in subset),
                top_five_raster_exact=sum(r[label]['top_five_raster_exact'] for r in subset),
                all_retained_matches=sum(r[label]['all_retained_match'] for r in subset),
                top_five_misses=[r['frame'] for r in subset if not r[label]['top_five_match']],
                tied_only_matches=[r['frame'] for r in subset if not r[label]['top_five_match'] and r[label]['all_retained_match']])
        entry=dict(name=name,all_frames={k:summary(rows,k) for k in ['baseline','deferred']},
            flagged_frames={k:summary([r for r in rows if r['flagged']],k) for k in ['baseline','deferred']},frames=rows)
        result['cases'].append(entry)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(name=name,all_frames=entry['all_frames'],flagged_frames=entry['flagged_frames'])),flush=True)


if __name__=='__main__':
    import torch
    torch.set_num_threads(2)
    main()
