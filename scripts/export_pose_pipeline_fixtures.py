"""Re-run the complete CPU joint pipeline on the browser's saved frame sequence.

No cached camera is supplied. The same filtered points, presence values,
sticker colors, pixel scale, frame indices and static-hold configuration become
inputs to both runtimes. Historical render logs remain a separate comparison.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import sys
from time import perf_counter
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import render_yolo11n_independent_keypoints as r
from scripts import independent_keypoint_pose_geometry as f
from scripts.analyze_orientation_motion import REFERENCE_SOLUTIONS
from scripts.export_cpu_pose_fixtures import fit,pose,pack

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',nargs='+',default=list(REFERENCE_SOLUTIONS))
    p.add_argument('--input-dir',type=Path,default=ROOT/'pose-fitter/results/browser-rollout-audit')
    p.add_argument('--output-dir',type=Path,default=ROOT/'pose-fitter/fixtures/full-pipeline')
    p.add_argument('--trace',action='store_true',help='Save independent stage checkpoints and fixed-input recovery solves')
    a=p.parse_args();torch.set_num_threads(2)
    for name in a.cases:
        data=json.loads((a.input_dir/f'{name}-inputs.json').read_text())
        solution=json.loads((ROOT/REFERENCE_SOLUTIONS[name]).read_text())
        log=Path(solution['config']['log']);meta=json.loads(log.with_suffix('.meta.json').read_text())
        saved={x['frame_index']:x for x in map(json.loads,log.open())}
        scramble=solution['config'].get('initial_scramble') or ''
        state=r.CubeState.from_scramble(scramble.split() if isinstance(scramble,str) else scramble)
        expected_colors={i.label:state.color_at((i.cubie_index,i.axis,i.side)) for i in r.canonical_sticker_identities()}
        centers=[i for i in r.canonical_sticker_identities() if i.category=='center']
        out=dict(geometry=data['geometry'],workloads=[],probes=[],name=name,
            fps=meta.get('source_fps',meta['fps']),turning_start_frame=meta.get('turning_start_frame'),
            expected_center_colors=[dict(center=[(i.cubie_index[k]-1)*r.CUBE_LAYER_SIZE_MM for k in range(3)],
                axis=i.axis,side=i.side,color=expected_colors[i.label]) for i in centers],frames=[],
            scope=__doc__,pixel_scale_source='Saved camera diameter / 220 fallback; source ROI absent from historical logs',
            source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [
                ROOT/'scripts/independent_keypoint_pose_geometry.py',ROOT/'scripts/render_yolo11n_independent_keypoints.py']})
        cache=[]
        for item in data['workloads']:
            obs=item['observations'];points=np.asarray(obs['points'],np.float32);classes=np.asarray(obs['classes'],np.int32)
            stickers=np.asarray(obs['stickers'],float);presence=np.ones(len(points))
            matches=r.sticker_keypoint_matches_for_colors(points,presence,classes,stickers)
            candidates=r.best_sticker_pnp_candidates(points,classes,matches,tuple(item['image_size']),1.2,
                scoring_points=points,prepare_mask_geometry=False)
            payload=saved[item['frame']].get('search_pose') or saved[item['frame']].get('selected_pose')
            colors=tuple(payload['sticker_color_names']);assert len(colors)==len(stickers)
            candidates=tuple(r.attach_color_keypoints(c,stickers,colors) for c in candidates)
            # Preserve the renderer's documented candidate-only phase. No final
            # PnP or articulation is performed before process_pose_geometry.
            candidate=candidates[0] if candidates else None
            from scripts.benchmark_cpu_pose_port import hydrate
            saved_camera,_=hydrate(item)
            pixel_scale=max(r.projected_cube_diameter(saved_camera)/220.,.3)
            record=SimpleNamespace(frame_index=item['frame'],pose_candidate=candidate,pose_candidates=candidates,
                pose_ready=True,frame=None,audit_observations=r.RigidPoseObservations(points,classes,stickers,colors,pixel_scale))
            cache.append(record)
            out['frames'].append(dict(frame=item['frame'],width=item['image_size'][0],height=item['image_size'][1],
                observations=obs,landmark_presence=presence.tolist(),colors=colors,pixel_scale=pixel_scale,
                expected_startup=None if candidate is None else pose(candidate)))
        by_option={};original_option=f.projection_option;original_observations=r._rigid_pose_observations
        original_fit=f.fit_frame;original_orient=f.orient_fits;original_targets=f.depth_trend_targets;original_depth=f._refine_depth_recovery
        active=[None,0];traces={};depth_probes=[]
        def fit_call(renderer,candidate,**kw):
            bank=original_fit(renderer,candidate,**kw)
            if a.trace:
                label='initial' if active[0]=='joint pose/geometry refinement' else 'neighbors'
                traces.setdefault(label,{})[active[1]]=[fit(v) for v in bank]
            return bank
        def orient_call(*args,**kw):
            bank=original_orient(*args,**kw)
            if a.trace:traces.setdefault('orientation',{})[active[1]]=[fit(v) for v in bank]
            return bank
        def targets_call(*args,**kw):
            values=original_targets(*args,**kw)
            if a.trace:out['expected_depth_targets']=values
            return values
        def depth_call(model,seed,target,*args,**kw):
            value=original_depth(model,seed,target,*args,**kw)
            if a.trace:depth_probes.append(dict(workload=pack(name,0,'depth_recovery',model.base,None,(),()),
                axis=model.axis,seed=np.asarray(seed).tolist(),target=target,expected=value.tolist()))
            return value
        def option(renderer,fit):
            value=original_option(renderer,fit);by_option[id(value)]=fit;return value
        r._rigid_pose_observations=lambda record,args:record.audit_observations
        f.projection_option=option
        f.fit_frame=fit_call;f.orient_fits=orient_call;f.depth_trend_targets=targets_call;f._refine_depth_recovery=depth_call
        previous_message=[None,0]
        def progress(label,i,total):
            active[:]=[label,i]
            if label!=previous_message[0] or i>=previous_message[1]+30 or i==total:
                print(name,label,i,'/',total,flush=True);previous_message[:]=[label,i]
        started=perf_counter()
        try:
            f.process_pose_geometry(r,cache,SimpleNamespace(turning_start_frame=out['turning_start_frame'],
                pose_smoothing=True,joint_depth_fallback=True,geometry_sweep_device='cpu',geometry_refinement_device='cpu'),
                expected_colors,out['fps'],progress=progress)
        finally:
            r._rigid_pose_observations=original_observations;f.projection_option=original_option
            f.fit_frame=original_fit;f.orient_fits=original_orient;f.depth_trend_targets=original_targets;f._refine_depth_recovery=original_depth
        for frame,record in zip(out['frames'],cache):
            opts=record.pose_candidate.joint_geometry_candidates if record.pose_candidate else ()
            frame['expected']=[fit(by_option[id(option)]) for option in opts]
            frame['expected_depth_recovery']=bool(record.pose_candidate and record.pose_candidate.joint_pose_diagnostics and
                'trigger' in record.pose_candidate.joint_pose_diagnostics)
        out['cpu_seconds']=perf_counter()-started
        if a.trace:
            out['expected_stages']={label:[values.get(i,[]) for i in range(len(cache))] for label,values in traces.items()}
            out['depth_probes']=depth_probes
        a.output_dir.mkdir(parents=True,exist_ok=True)
        path=a.output_dir/f'{name}.json';path.write_text(json.dumps(out,allow_nan=False,separators=(',',':'))+'\n')
        print('saved',path,'seconds',out['cpu_seconds'],flush=True)

if __name__=='__main__':main()
