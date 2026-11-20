"""Export current-frame CPU startup from identical already-filtered detections."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import render_yolo11n_independent_keypoints as r

def reference(item):
    obs=item['observations']
    points=np.asarray(obs['points'],np.float32).reshape(-1,2)
    classes=np.asarray(obs['classes'],np.int32)
    presence=np.ones(len(points))
    matches=r.sticker_keypoint_matches_for_colors(points,presence,classes,np.asarray(obs['stickers']))
    candidates=r.best_sticker_pnp_candidates(points,classes,matches,tuple(item['image_size']),1.2,
        scoring_points=points,prepare_mask_geometry=False)
    return dict(name=item['name'],frame=item['frame'],input=dict(width=item['image_size'][0],
        height=item['image_size'][1],observations=obs,landmark_presence=presence.tolist()),
        matches=[dict(sticker_index=m.mask_index,indices=list(m.keypoint_indices),
            sticker_type=['corner','edge','center'].index(m.sticker_type),center_error=m.center_error,
            mean_presence=m.average_confidence,sequences=[list(s) for s in m.object_index_sequences]) for m in matches],
        candidates=[dict(pose=dict(rvec=c.rvec.ravel().tolist(),tvec=c.tvec.ravel().tolist()),
            visual_score=r.pose_hypothesis_visual_score(c),overall_score=c.overall_reprojection_error,
            anchor_error=c.reprojection_error,mean_presence=c.sticker_average_confidence,
            sticker_type=['corner','edge','center'].index(c.sticker_type)) for c in candidates])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,default=ROOT/'pose-fitter/results/browser-rollout-audit')
    p.add_argument('--output',type=Path,default=ROOT/'pose-fitter/fixtures/startup-reference.json')
    p.add_argument('--samples',type=int,default=5)
    a=p.parse_args();rows=[]
    for name in ('matty','nahm','xuanyi','tommy','yiheng'):
        corpus=json.loads((a.input_dir/f'{name}-inputs.json').read_text())
        indices=np.unique(np.linspace(0,len(corpus['workloads'])-1,a.samples,dtype=int))
        for index in indices:
            item=corpus['workloads'][index];rows.append(reference(item));print(name,item['frame'],len(rows[-1]['matches']),len(rows[-1]['candidates']),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(geometry=corpus['geometry'],workloads=[],probes=[],cases=rows),allow_nan=False,separators=(',',':'))+'\n')

if __name__=='__main__':main()
