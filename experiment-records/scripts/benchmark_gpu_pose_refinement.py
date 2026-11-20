"""Compare batched refinement with the existing GPU-sweep/CPU-refinement fitter."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from scripts import independent_keypoint_pose_geometry as geometry
from scripts import render_yolo11n_independent_keypoints as renderer
from scripts.analyze_orientation_motion import REFERENCE_SOLUTIONS
from scripts.benchmark_temporal_pose_reuse import saved_candidate
from scripts.benchmark_gpu_pose_sweep import compare_banks,timed_pair,synchronize
from scripts.independent_keypoint_gpu_refine import refine_batch
from scripts.benchmark_physics_pose_tracker import final_selected_fit
from scripts.pose_wireframe_metrics import center_pixels,wire_error



def save_preview(path,metadata,frame,target,cpu,gpu,errors):
    capture=cv2.VideoCapture(str(metadata['source']))
    capture.set(cv2.CAP_PROP_POS_FRAMES,frame)
    ok,image=capture.read();capture.release()
    if not ok:raise RuntimeError(f'Could not read source frame {frame}')
    diameter=renderer.projected_cube_diameter(target)
    center=center_pixels(target,target.tvec)
    size=360
    zoom=size/(1.8*diameter)
    transform=np.array([[zoom,0,size/2-center[0]*zoom],[0,zoom,size/2-center[1]*zoom]])
    clean=cv2.warpAffine(image,transform,(size,size))
    def panel(candidate,label,color):
        canvas=clean.copy()
        for _,polygon,_ in renderer.projected_cubelet_faces(candidate,candidate.rvec,candidate.tvec):
            points=np.rint(np.c_[polygon,np.ones(len(polygon))]@transform.T).astype(np.int32)
            cv2.polylines(canvas,[points],True,(0,0,0),3,cv2.LINE_AA)
            cv2.polylines(canvas,[points],True,color,1,cv2.LINE_AA)
        cv2.rectangle(canvas,(0,0),(size,27),(25,25,25),-1)
        cv2.putText(canvas,label,(8,19),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
        return canvas
    ref=panel(target,'Trusted selected reference',(255,220,70))
    first=np.hstack((ref,panel(cpu[0].candidate,'CPU best geometry score',(70,255,70)),
        panel(gpu[0].candidate,'Metal best geometry score',(70,70,255))))
    second=np.hstack((ref,panel(cpu[int(np.argmin(errors['cpu']))].candidate,'CPU closest bank member',(70,255,70)),
        panel(gpu[int(np.argmin(errors['gpu']))].candidate,'Metal closest bank member',(70,70,255))))
    cv2.imwrite(str(path),np.vstack((first,second)),[cv2.IMWRITE_JPEG_QUALITY,92])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-benchmark',type=Path,default=Path('runs/gpu-exhaustive-sweep-mps-v3/report.json'))
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--cases',nargs='+',choices=tuple(REFERENCE_SOLUTIONS),default=list(REFERENCE_SOLUTIONS))
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--fit-limit',type=int,default=0)
    parser.add_argument('--previews',action='store_true')
    args=parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; this benchmark requires the actual GPU')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    previous=json.loads(args.reference_benchmark.read_text())
    report=dict(parameters=vars(args)|dict(reference_benchmark=str(args.reference_benchmark),output_dir=str(args.output_dir)),
        scope='Same cached workloads as the GPU sweep benchmark. Both compared fitting paths use the same MPS sweep; only refinement changes. Final scoring/materialization are CPU.',
        timing='Warmed, synchronized, interleaved wall times including transfers. Quality checks excluded.',
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in [
            'scripts/independent_keypoint_gpu_refine.py','scripts/independent_keypoint_pose_geometry.py',
            'scripts/independent_keypoint_gpu_sweep.py','scripts/independent_keypoint_metal_refine.py',
            'scripts/independent_keypoint_refine.metal','scripts/benchmark_gpu_pose_refinement.py']},cases=[])
    for case in previous['cases']:
        name=case['name']
        if name not in args.cases: continue
        solution=json.loads((ROOT/REFERENCE_SOLUTIONS[name]).read_text())
        metadata=json.loads(Path(case['source_log']).with_suffix('.meta.json').read_text())
        start=metadata.get('turning_start_frame') or 0
        end=solution.get('solved_frame_index') or max(s['frame_index'] for s in solution['steps'])
        step_by_frame={s['frame_index']:s for s in solution['steps']}
        steps=set(step_by_frame)
        observations=[]
        records={}
        for record in map(json.loads,Path(case['source_log']).open()):
            if not start<=record['frame_index']<=end or record['frame_index'] not in steps:continue
            candidate=saved_candidate(record)
            if candidate is not None and len(candidate.assignment_image_points)>=3 and len(candidate.sticker_mask_centers):
                observations.append((record['frame_index'],candidate))
                records[record['frame_index']]=record|dict(_reference_step=step_by_frame[record['frame_index']])
        by_frame={frame:i for i,(frame,c) in enumerate(observations)}
        rows=[]
        for workload in case['fits'][:args.fit_limit or None]:
            frame,stage=workload['frame'],workload['stage']
            i=by_frame[frame];candidate=observations[i][1]
            kwargs=dict(prior=observations[i-1][1])
            if stage=='temporal_neighbors':
                retained=geometry.fit_frame(renderer,candidate,**kwargs)
                neighbors=[geometry.fit_frame(renderer,observations[j][1])[0].candidate for j in (i-1,i+1)]
                candidate=retained[0].candidate
                kwargs=dict(prior=neighbors[0],neighbors=neighbors,retained=retained)
            starts=[]
            original=geometry._refine
            def collect(model,seed,**kw):
                starts.append((model,seed.copy(),kw))
                return original(model,seed,**kw)
            geometry._refine=collect
            try:
                cpu=geometry.fit_frame(renderer,candidate,sweep_device='mps',**kwargs)
            finally:
                geometry._refine=original
            base=starts[0][0].base
            axes=[model.axis for model,seed,kw in starts]
            seeds=np.stack([seed for model,seed,kw in starts])
            prior=starts[0][2]['prior']
            cpu_refine=lambda:[original(model,seed,**kw) for model,seed,kw in starts]
            gpu_refine=lambda:refine_batch(renderer,base,axes,seeds,prior=prior)
            cpu_refine();gpu_parameters=gpu_refine()
            refinement_times,_=timed_pair(cpu_refine,gpu_refine,args.repeats,'mps')
            cpu_call=lambda:geometry.fit_frame(renderer,candidate,sweep_device='mps',**kwargs)
            gpu_call=lambda:geometry.fit_frame(renderer,candidate,sweep_device='mps',refinement_device='mps',**kwargs)
            gpu=gpu_call()
            quality=compare_banks(cpu,gpu)
            quality['cpu_best_score']=cpu[0].score;quality['gpu_best_score']=gpu[0].score
            quality['best_score_change']=gpu[0].score-cpu[0].score
            target=final_selected_fit(records[frame]).candidate
            reference_errors={label:[wire_error(f.candidate,target) for f in bank] for label,bank in [('cpu',cpu),('gpu',gpu)]}
            quality['reference_best_bank_wire_cpu']=min(reference_errors['cpu'])
            quality['reference_best_bank_wire_gpu']=min(reference_errors['gpu'])
            def pose(f):
                return dict(axis=f.axis,angles=list(map(float,f.angles)),rvec=f.candidate.rvec.ravel().tolist(),
                    tvec=f.candidate.tvec.ravel().tolist(),score=f.score)
            bank_parameters=dict(cpu=[pose(f) for f in cpu],gpu=[pose(f) for f in gpu])
            if args.previews:
                save_preview(args.output_dir/f'{name}-{frame}-{stage}.jpg',metadata,frame,target,cpu,gpu,reference_errors)

            times,_=timed_pair(cpu_call,gpu_call,args.repeats,'mps')
            row=dict(frame=frame,stage=stage,seeds=len(starts),refinement=refinement_times,fitting=times,quality=quality,bank_parameters=bank_parameters)
            rows.append(row)
            print(json.dumps(dict(name=name,frame=frame,stage=stage,seeds=len(starts),
                refinement_speedup=refinement_times['speedup'],fitting_speedup=times['speedup'],
                wire=quality['best_scored_wire_p90_fraction'],score_change=quality['best_score_change'])),flush=True)
        report['cases'].append(dict(name=name,workloads=rows))
        (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    summary={}
    for stage in ['refinement','fitting']:
        rows=[w[stage] for c in report['cases'] for w in c['workloads']]
        a=sum(w['cpu_median_ms'] for w in rows);b=sum(w['gpu_median_ms'] for w in rows)
        summary[stage]=dict(cpu_total_median_ms=a,gpu_total_median_ms=b,speedup=a/b,reduction_fraction=1-b/a)
    report['summary']=summary
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    main()
