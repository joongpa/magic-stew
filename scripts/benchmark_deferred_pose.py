"""Full solve-window comparison of partial-first and established GPU fitting.

Both paths receive the same newly generated rigid candidates. Saved poses are
used only for evaluation and display, never initialization or temporal feedback.
All timings exclude detection, evaluation, video I/O and rendering.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import render_yolo11n_independent_keypoints as renderer
from scripts import independent_keypoint_pose_geometry as geometry
from scripts.analyze_orientation_motion import REFERENCE_SOLUTIONS
from scripts.benchmark_temporal_pose_reuse import saved_candidate
from scripts.benchmark_physics_pose_tracker import final_selected_fit
from scripts.independent_keypoint_deferred_pose import static_candidates,observation_template,fit_buffered,buffered_cameras
from scripts.pose_wireframe_metrics import center_pixels,wire_error


def measured(call):
    torch.mps.synchronize();start=perf_counter()
    value=call();torch.mps.synchronize()
    return value,1000*(perf_counter()-start)


def pose(fit):
    return dict(axis=fit.axis,angles=list(map(float,fit.angles)),score=fit.score,
        rvec=fit.candidate.rvec.ravel().tolist(),tvec=fit.candidate.tvec.ravel().tolist())


def quality(bank,target):
    if not bank:return dict(candidates=0,best_score=None,reprojection_fit=None,best_wire=None,closest_wire=None)
    errors=[wire_error(f.candidate,target) for f in bank] if target is not None else []
    return dict(candidates=len(bank),best_score=bank[0].score,
        reprojection_fit=float(renderer.pose_reprojection_fit_metric(bank[0].candidate)),
        best_wire=errors[0] if errors else None,closest_wire=min(errors) if errors else None)


def partial_recall(bank,target):
    if target is None or not bank:return None
    R=cv2.Rodrigues(target.rvec)[0]
    layers=target.layer_rotations
    axis=layers[0].axis if layers else 0
    angles=renderer.parallel_layer_angles(layers) if layers else (0.,0.,0.)
    alternatives=np.concatenate([R@cv2.Rodrigues(np.eye(3)[axis]*np.deg2rad(a))[0]@np.asarray(renderer.CUBE_SYMMETRY_MATRICES) for a in angles])
    diameter=renderer.projected_cube_diameter(target)
    center=center_pixels(target,target.tvec)
    best=999.
    for proposal in bank:
        x=proposal.parameters;r=cv2.Rodrigues(x[:3])[0]
        traces=np.einsum('nij,ij->n',alternatives,r)
        angle=np.rad2deg(np.arccos(np.clip((traces.max()-1)/2,-1,1)))
        core=center_pixels(target,x[3:6])
        if np.linalg.norm(core-center)/diameter<=.03 and abs(x[5]-target.tvec.ravel()[2])/target.tvec.ravel()[2]<=.05:
            best=min(best,float(angle))
    return best<=6.


def panel(image,candidate,center,diameter,title,color,size=400):
    zoom=size/(2.1*diameter)
    transform=np.array([[zoom,0,size/2-center[0]*zoom],[0,zoom,size/2-center[1]*zoom]])
    result=cv2.warpAffine(image,transform,(size,size))
    if candidate is not None:
        for _,polygon,_ in renderer.projected_cubelet_faces(candidate,candidate.rvec,candidate.tvec):
            xy=np.rint(np.c_[polygon,np.ones(len(polygon))]@transform.T).astype(np.int32)
            cv2.polylines(result,[xy],True,(0,0,0),3,cv2.LINE_AA)
            cv2.polylines(result,[xy],True,color,1,cv2.LINE_AA)
    cv2.rectangle(result,(0,0),(size,30),(20,20,20),-1)
    cv2.putText(result,title,(7,21),cv2.FONT_HERSHEY_SIMPLEX,.47,(255,255,255),1,cv2.LINE_AA)
    return result


def draw_comparison(image,frame,metadata,target,old,new,row,closest=False):
    candidate=target or (new[0].candidate if new else old[0].candidate if old else None)
    if candidate is not None:
        center=center_pixels(candidate,candidate.tvec);diameter=renderer.projected_cube_diameter(candidate)
    else:
        center=np.array([image.shape[1]/2,image.shape[0]/2]);diameter=min(image.shape[:2])*.6
    def choice(bank):
        if not bank:return None
        if closest and target is not None:return min(bank,key=lambda f:wire_error(f.candidate,target)).candidate
        return bank[0].candidate
    mode='closest bank' if closest else 'best score'
    tiles=[panel(image,target,center,diameter,'Saved selected reference',(255,220,70)),
        panel(image,choice(old),center,diameter,f'Existing GPU - {mode}',(70,255,70)),
        panel(image,choice(new),center,diameter,f'Deferred GPU - {mode}',(70,70,255))]
    result=np.hstack(tiles)
    footer=np.full((48,result.shape[1],3),20,np.uint8)
    olderror=row['baseline']['best_wire'];newerror=row['deferred']['best_wire']
    text=f"{row['name'].upper()} | frame {frame} | {frame/metadata.get('source_fps',metadata['fps']):.2f}s | "
    text+=f"wire p90 / diameter: existing {olderror:.3f}, deferred {newerror:.3f}" if olderror is not None and newerror is not None else 'missing fit or selected reference'
    cv2.putText(footer,text,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.52,(240,240,240),1,cv2.LINE_AA)
    cv2.putText(footer,'Geometry comparison; colors identify algorithms, not detector confidence.',(8,40),cv2.FONT_HERSHEY_SIMPLEX,.44,(210,210,210),1,cv2.LINE_AA)
    return np.vstack((result,footer))


def case_run(name,args):
    solution=json.loads((ROOT/REFERENCE_SOLUTIONS[name]).read_text())
    log=Path(solution['config']['log']);metadata=json.loads(log.with_suffix('.meta.json').read_text())
    records={r['frame_index']:r for r in map(json.loads,log.open())}
    steps={s['frame_index']:s for s in solution['steps']}
    start=metadata.get('turning_start_frame') or 0
    end=solution.get('solved_frame_index') or max(steps)
    if name=='matty':end=min(end,560) # User excludes post-solve tail.
    fps=metadata.get('source_fps',metadata['fps']);stride=max(1,int(np.ceil(fps/args.max_fps-1e-6)))
    frames=list(range(start,end+1,stride))
    if args.limit:frames=frames[:args.limit]
    directory=args.output_dir/name;directory.mkdir()
    observations=[];banks=[];static_stats=[]
    last=perf_counter()
    for i,frame in enumerate(frames):
        c=saved_candidate(records[frame]) if frame in records else None
        c=observation_template(c) if c is not None else None
        observations.append(c)
        if c is not None and len(c.assignment_image_points)>=3 and len(c.sticker_mask_centers):
            bank,stats=static_candidates(renderer,c,beam=args.static_beam)
        else:bank,stats=[],dict(milliseconds=0.,retained=0,raw_roots=0,status='missing_observations')
        banks.append(bank);static_stats.append(stats)
        if perf_counter()-last>5:
            print(f'{name}: static proposals {i+1}/{len(frames)}',flush=True);last=perf_counter()
    baseline_initial=[];initial_ms=[];previous=None
    # Compile and warm kernels with generated current-detection proposals only.
    warm=next((i for i,b in enumerate(banks) if b),None)
    if warm is not None:
        x=banks[warm][0].parameters
        base=replace(observations[warm],rvec=x[:3].reshape(3,1),tvec=x[3:6].reshape(3,1))
        geometry.fit_frame(renderer,base,sweep_device='mps',refinement_device='mps')
        fit_buffered(renderer,observations[warm],banks[warm],device='mps',camera_budget=args.camera_budget)
    last=perf_counter()
    for i,c in enumerate(observations):
        if c is None or not banks[i]:fits,elapsed=[],0.
        else:
            x=banks[i][0].parameters;base=replace(c,rvec=x[:3].reshape(3,1),tvec=x[3:6].reshape(3,1))
            fits,elapsed=measured(lambda:geometry.fit_frame(renderer,base,prior=previous,sweep_device='mps',refinement_device='mps'))
        baseline_initial.append(fits);initial_ms.append(elapsed)
        if fits:previous=fits[0].candidate
        if perf_counter()-last>5:
            print(f'{name}: existing initial fits {i+1}/{len(frames)}',flush=True);last=perf_counter()
    selected=[f[0].candidate if f else None for f in baseline_initial]
    targets=[]
    for frame in frames:
        r=records.get(frame)
        target=(final_selected_fit(dict(r,_reference_step=steps[frame])).candidate
                if r is not None and frame in steps and saved_candidate(r) is not None else None)
        targets.append(target)
    old_banks=[];new_banks=[];rows=[]
    last=perf_counter()
    stream=(directory/'frames.jsonl').open('w')
    try:
        for i,(frame,c) in enumerate(zip(frames,observations,strict=True)):
            neighbors=[selected[j] for j in (i-1,i+1) if 0<=j<len(frames) and selected[j] is not None]
            retained=baseline_initial[i]
            def baseline():
                if not retained or not neighbors:return retained
                base=retained[0].candidate;ref=cv2.Rodrigues(base.rvec)[0]
                rotations=[renderer._align_rigid_rotation(n.rvec,ref) for n in neighbors]
                mean=np.mean([cv2.Rodrigues(r@ref.T)[0].ravel() for r in rotations],axis=0)
                rv=cv2.Rodrigues(cv2.Rodrigues(mean)[0]@ref)[0]
                prior=replace(base,rvec=rv,tvec=np.mean([n.tvec for n in neighbors],axis=0))
                return geometry.fit_frame(renderer,base,prior=prior,neighbors=neighbors,retained=retained,
                    sweep_device='mps',refinement_device='mps')
            def deferred():
                if c is None:return [],dict(status='missing_observations')
                return fit_buffered(renderer,c,banks[i],banks[i-1] if i else (),banks[i+1] if i+1<len(frames) else (),
                    device='mps',camera_budget=args.camera_budget)
            if i%2==0:
                old,old_ms=measured(baseline);(new,diagnostics),new_ms=measured(deferred)
            else:
                (new,diagnostics),new_ms=measured(deferred);old,old_ms=measured(baseline)
            target=targets[i]
            candidate_cameras=buffered_cameras(renderer,banks[i],banks[i-1] if i else (),banks[i+1] if i+1<len(frames) else (),budget=args.camera_budget)
            row=dict(name=name,frame=frame,static=static_stats[i],initial_ms=initial_ms[i],smoothing_ms=old_ms,
                deferred_joint_ms=new_ms,baseline_total_ms=static_stats[i]['milliseconds']+initial_ms[i]+old_ms,
                deferred_total_ms=static_stats[i]['milliseconds']+new_ms,diagnostics=diagnostics,
                reference_available=target is not None,partial_bank_recall=partial_recall(banks[i],target),
                selected_camera_recall=partial_recall(candidate_cameras,target),baseline=quality(old,target),deferred=quality(new,target),
                baseline_bank=[pose(f) for f in old],deferred_bank=[pose(f) for f in new])
            rows.append(row);old_banks.append(old);new_banks.append(new)
            stream.write(json.dumps(row,separators=(',',':'))+'\n');stream.flush()
            if perf_counter()-last>5:
                print(f'{name}: buffered joint fits {i+1}/{len(frames)}',flush=True);last=perf_counter()
    finally:stream.close()
    evaluated=[r for r in rows if r['reference_available']]
    def summarize(label):
        wires=[r[label]['best_wire'] for r in evaluated if r[label]['best_wire'] is not None]
        closest=[r[label]['closest_wire'] for r in evaluated if r[label]['closest_wire'] is not None]
        total=[r[f'{label}_total_ms'] for r in rows]
        return dict(mean_ms=float(np.mean(total)),median_ms=float(np.median(total)),p90_ms=float(np.percentile(total,90)),
            fitting_seconds=sum(total)/1000.,missing_fits=sum(not r[label]['candidates'] for r in rows),
            best_wire_within_002=sum(e<=.02 for e in wires),bank_wire_within_002=sum(e<=.02 for e in closest),
            evaluated_frames=len(evaluated),median_best_wire=float(np.median(wires)) if wires else None,
            p90_best_wire=float(np.percentile(wires,90)) if wires else None,min_bank_size=min(r[label]['candidates'] for r in rows))
    result=dict(name=name,source=str(metadata['source']),source_log=str(log),source_fps=fps,stride=stride,
        start_frame=start,end_frame=frames[-1],frames=len(frames),baseline=summarize('baseline'),deferred=summarize('deferred'),
        static_mean_ms=float(np.mean([r['static']['milliseconds'] for r in rows])),
        deferred_joint_mean_ms=float(np.mean([r['deferred_joint_ms'] for r in rows])),
        partial_bank_recall=sum(r['partial_bank_recall'] is True for r in evaluated),
        selected_camera_recall=sum(r['selected_camera_recall'] is True for r in evaluated),
        baseline_input='Same independently generated partial-static bank: best raw partial candidate, then established initial plus temporal GPU fitting.',
        temporal_buffer_frames=1,reference_used_by_fitting=False)
    (directory/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    # All render/evaluation work happens after timing.
    capture=cv2.VideoCapture(str(metadata['source']))
    video=directory/'wireframes.mp4'
    command=['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s','1200x448',
        '-r',str(fps/stride),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    encoder=subprocess.Popen(command,stdin=subprocess.PIPE)
    worst=sorted(range(len(rows)),key=lambda i:rows[i]['deferred']['best_wire'] or 0.,reverse=True)[:3]
    previews=set(np.linspace(0,len(frames)-1,5,dtype=int))|set(worst)
    try:
        for i,frame in enumerate(frames):
            capture.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=capture.read()
            if not ok:raise RuntimeError(f'{name}: could not read source frame {frame}')
            drawing=draw_comparison(image,frame,metadata,targets[i],old_banks[i],new_banks[i],rows[i])
            encoder.stdin.write(drawing.tobytes())
            if i in previews:
                closest=draw_comparison(image,frame,metadata,targets[i],old_banks[i],new_banks[i],rows[i],closest=True)
                cv2.imwrite(str(directory/f'frame-{frame}.jpg'),np.vstack((drawing,closest)))
    finally:
        capture.release();encoder.stdin.close();code=encoder.wait()
    if code:raise RuntimeError(f'{name}: encoder exited {code}')
    result['video']=str(video.resolve());result['previews']=[str(p.resolve()) for p in sorted(directory.glob('frame-*.jpg'))]
    (directory/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--cases',nargs='+',choices=list(REFERENCE_SOLUTIONS),default=list(REFERENCE_SOLUTIONS))
    parser.add_argument('--static-beam',type=int,default=16)
    parser.add_argument('--camera-budget',type=int,default=6)
    parser.add_argument('--max-fps',type=float,default=30.)
    parser.add_argument('--limit',type=int,default=0,help='Debug only: limit to initial frames of each solve window.')
    args=parser.parse_args()
    if not torch.backends.mps.is_available():raise RuntimeError('Actual MPS access required; no CPU fallback')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    sources=['scripts/independent_keypoint_deferred_pose.py','scripts/benchmark_deferred_pose.py',
        'scripts/independent_keypoint_slab_search.py','scripts/independent_keypoint_pose_geometry.py',
        'scripts/independent_keypoint_gpu_sweep.py','scripts/independent_keypoint_refine.metal']
    report=dict(parameters=vars(args)|dict(output_dir=str(args.output_dir)),
        scope='Complete sampled solve windows from solve starts; max 30 fps; independent current-detection PnP proposals; one-frame buffer; no current reference initialization.',
        timing='One synchronized full-window run per implementation. Same static stage charged to both; baseline initial fits followed by interleaved baseline-smoothing/deferred-joint fits. Detection, evaluation and rendering excluded.',
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},cases=[])
    for name in args.cases:
        report['cases'].append(case_run(name,args))
        (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    if len(report['cases'])>1:
        command=['ffmpeg','-hide_banner','-loglevel','error','-n']
        for case in report['cases']:command+=['-i',case['video']]
        filters=';'.join(f'[{i}:v]fps=30,settb=AVTB,setpts=PTS-STARTPTS[v{i}]' for i in range(len(report['cases'])))
        filters+=';'+''.join(f'[v{i}]' for i in range(len(report['cases'])))+f"concat=n={len(report['cases'])}:v=1:a=0[out]"
        command+=['-filter_complex',filters,'-map','[out]','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(args.output_dir/'all-five-wireframes.mp4')]
        subprocess.run(command,check=True)
    print(json.dumps(dict(completed_cases=len(report['cases']),report=str(args.output_dir/'report.json'))),flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    main()
