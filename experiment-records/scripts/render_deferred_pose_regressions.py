"""Render only paired regressions from a completed deferred-pose benchmark."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import independent_keypoint_pose_geometry as geometry
from scripts import render_yolo11n_independent_keypoints as renderer
from scripts.analyze_orientation_motion import REFERENCE_SOLUTIONS
from scripts.benchmark_temporal_pose_reuse import saved_candidate
from scripts.benchmark_physics_pose_tracker import final_selected_fit
from scripts.benchmark_deferred_pose import draw_comparison


def restore(base,payload):
    x=np.r_[payload['rvec'],payload['tvec'],np.asarray(payload['angles'])[[0,2]]]
    return geometry._materialize(renderer,base,payload['axis'],x)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    report=json.loads(args.benchmark.read_text())
    index=[];slides=[];contact=[]
    for case in report['cases']:
        name=case['name']
        solution=json.loads((ROOT/REFERENCE_SOLUTIONS[name]).read_text())
        log=Path(case['source_log']);metadata=json.loads(log.with_suffix('.meta.json').read_text())
        records={r['frame_index']:r for r in map(json.loads,log.open())}
        steps={s['frame_index']:s for s in solution['steps']}
        rows=[json.loads(l) for l in (args.benchmark.parent/name/'frames.jsonl').open()]
        regressions=[]
        for row in rows:
            a,b=row['baseline'],row['deferred']
            if a['best_wire'] is None or b['best_wire'] is None:continue
            delta=b['best_wire']-a['best_wire']
            worse=delta>1e-7
            bank_lost=a['closest_wire']<=.02<b['closest_wire']
            material=worse and (delta>=.01 or a['best_wire']<=.02<b['best_wire'])
            if worse or bank_lost:regressions.append((row,delta,material,bank_lost))
        capture=cv2.VideoCapture(str(metadata['source']));case_images=[]
        for row,delta,material,bank_lost in regressions:
            frame=row['frame'];record=records[frame];base=saved_candidate(record)
            target=final_selected_fit(dict(record,_reference_step=steps[frame])).candidate
            old=[restore(base,p) for p in row['baseline_bank']]
            new=[restore(base,p) for p in row['deferred_bank']]
            old=[p for p in old if p is not None];new=[p for p in new if p is not None]
            capture.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=capture.read()
            if not ok:raise RuntimeError(f'Could not read {name} frame {frame}')
            directory=args.output_dir/name;directory.mkdir(exist_ok=True)
            path=directory/f'frame-{frame}.jpg'
            primary=draw_comparison(image,frame,metadata,target,old,new,row)
            cv2.imwrite(str(path),primary,[cv2.IMWRITE_JPEG_QUALITY,94])
            data=dict(name=name,frame=frame,best_wire_existing=row['baseline']['best_wire'],
                best_wire_deferred=row['deferred']['best_wire'],increase=delta,
                material_best_regression=material,bank_candidate_lost=bank_lost,image=str(path.resolve()))
            if material:
                slides.append(primary)
                case_images.append(primary)
            if bank_lost:
                edited=dict(row,baseline=dict(row['baseline'],best_wire=row['baseline']['closest_wire']),
                    deferred=dict(row['deferred'],best_wire=row['deferred']['closest_wire']))
                comparison=draw_comparison(image,frame,metadata,target,old,new,edited,closest=True)
                cv2.rectangle(comparison,(0,400),(1200,448),(20,20,20),-1)
                text=f"{name.upper()} frame {frame} - BANK CANDIDATE LOST | closest wire: {row['baseline']['closest_wire']:.3f} -> {row['deferred']['closest_wire']:.3f}"
                cv2.putText(comparison,text,(8,426),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
                bank_path=directory/f'frame-{frame}-bank-loss.jpg'
                cv2.imwrite(str(bank_path),comparison,[cv2.IMWRITE_JPEG_QUALITY,94])
                data['bank_image']=str(bank_path.resolve())
                slides.append(comparison)
                if not material:case_images.append(comparison)
            index.append(data)
        capture.release()
        if case_images:
            contact_path=args.output_dir/f'{name}-regressions.jpg'
            cv2.imwrite(str(contact_path),np.vstack(case_images),[cv2.IMWRITE_JPEG_QUALITY,93])
            contact.append(dict(name=name,image=str(contact_path.resolve()),frames=len(case_images)))
    review=dict(criteria='All positive best-scored symmetric-p90 wire differences are indexed. Main slides show increases >=1% cube diameter, crossings of the 2% inspection threshold, and losses of reference-matching bank members. Numeric flags are for visual review, not independent ground truth.',
        regressions=index,contacts=contact,slides=len(slides))
    (args.output_dir/'review.json').write_text(json.dumps(review,indent=2)+'\n')
    if slides:
        video=args.output_dir/'regressions-only.mp4'
        encoder=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','bgr24',
            '-s','1200x448','-r','30','-i','-','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
        try:
            for slide in slides:
                for _ in range(60):encoder.stdin.write(slide.tobytes())
        finally:encoder.stdin.close();code=encoder.wait()
        if code:raise RuntimeError(f'Encoder exited {code}')
    lines=['# Regression frames only','',review['criteria'],'',
        'Panels: saved selected reference (cyan), existing GPU algorithm (green), deferred GPU algorithm (red).',
        'The controls use the same freshly generated static proposals; they are not replays of the saved final reference fit.','']
    for case in report['cases']:
        name=case['name'];rows=[r for r in index if r['name']==name]
        lines += [f'## {name.title()}','',f'{len(rows)} frames with a numeric best-fit regression or a lost bank candidate.','',
            '| Frame | Existing wire % | Deferred wire % | Review flag |','|---|---:|---:|---|']
        for row in rows:
            flags=[]
            if row['material_best_regression']:flags.append('larger best-fit difference')
            if row['bank_candidate_lost']:flags.append('bank candidate lost')
            if not flags:flags.append('small raster difference')
            lines.append(f"| [{row['frame']}]({row['image']}) | {100*row['best_wire_existing']:.2f} | {100*row['best_wire_deferred']:.2f} | {', '.join(flags)} |")
        lines.append('')
    (args.output_dir/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(index=str(args.output_dir/'README.md'),slides=len(slides),flagged_frames=len(index),contacts=contact)),flush=True)


if __name__=='__main__':
    import torch
    torch.set_num_threads(2)
    main()
