"""Show ranked CPU and WASM geometry banks from saved diagnostic outputs only."""
from pathlib import Path
from dataclasses import replace
import argparse,json,sys
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import render_yolo11n_independent_keypoints as r
from scripts.analyze_orientation_motion import REFERENCE_SOLUTIONS
from scripts.pose_assignment_comparison import compare_sticker_assignments
from scripts.benchmark_cpu_pose_port import hydrate

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'pose-fitter/results/full-pipeline-review/differences.json')
    a=p.parse_args();data=json.loads(a.input.read_text());g=data['geometry'];summary=[]
    for row in data['rows']:
        item=row['input'];reference=item['expected'][0]
        solution=json.loads((ROOT/REFERENCE_SOLUTIONS[row['name']]).read_text());log=Path(solution['config']['log']);meta=json.loads(log.with_suffix('.meta.json').read_text())
        cap=cv2.VideoCapture(meta['source']);cap.set(cv2.CAP_PROP_POS_FRAMES,item['frame']);ok,image=cap.read();cap.release();assert ok
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        base,_=hydrate(dict(camera=dict(rvec=reference['params'][:3],tvec=reference['params'][3:6],k=r.default_camera_matrix((item['width'],item['height'])).tolist(),dist=[0.]*4),observations=item['observations'],image_size=[item['width'],item['height']],retained=[],prior=None,neighbors=[],static_mode=False))
        stickers=np.asarray(item['observations']['stickers']);cloud=np.asarray(item['observations']['points']+item['observations']['stickers']);center=(cloud.min(axis=0)+cloud.max(axis=0))/2;radius=max(np.ptp(cloud,axis=0))*.65
        fig,axes=plt.subplots(2,5,figsize=(20,8.4),layout='constrained')
        entry=dict(name=row['name'],frame=item['frame'],rows=[])
        for which,bank in enumerate((item['expected'],row['actual']['bank'])):
            ordinary=[fit for fit in bank if not fit.get('flat_fallback')][:5]
            counts=[]
            for rank,ax in enumerate(axes[which]):
                ax.axis('off')
                if rank>=len(ordinary):ax.set_title('No additional unique assignment');continue
                fit=ordinary[rank];counts.append(compare_sticker_assignments(g,reference,fit)['sticker_assignment_changes'])
                ax.imshow(image);ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]+radius,center[1]-radius)
                rotations=r.parallel_layer_rotations(fit['axis'],fit['angles']);candidate=replace(base,rvec=np.asarray(fit['params'][:3]).reshape(3,1),tvec=np.asarray(fit['params'][3:6]).reshape(3,1),layer_rotations=rotations,sticker_layer_rotations=rotations)
                for _,poly,_ in r.projected_cubelet_faces(candidate,candidate.rvec,candidate.tvec):
                    poly=np.r_[poly,poly[:1]];ax.plot(poly[:,0],poly[:,1],color=('#00eac1' if which==0 else '#ff9b26'),linewidth=1.15)
                ax.scatter(stickers[:,0],stickers[:,1],s=7,c='white',edgecolors='black',linewidths=.4)
                source='Python CPU' if which==0 else 'Rust/WASM'
                ax.set_title(f"{source} #{rank+1} · cost {fit['score']:.5f}\n{counts[-1]}/{len(stickers)} assignments differ from Python #1\n{'XYZ'[fit['axis']]} outer angles [{fit['angles'][0]:+.1f}, {fit['angles'][2]:+.1f}]°",fontsize=9)
            entry['rows'].append(dict(runtime='python' if which==0 else 'wasm',assignment_changes_vs_python_winner=counts))
        fig.suptitle(f"{row['name'].title()} · frame {item['frame']} · top five unique-assignment options\nGeometry cost: lower is better. Separate reserved flat fallback is outside these five.",fontsize=13)
        output=a.input.parent/f"{row['name']}-{item['frame']}-top-five.png";fig.savefig(output,dpi=150);plt.close(fig);entry['image']=str(output);summary.append(entry);print(output,flush=True)
    (a.input.parent/'top-five-summary.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':main()
