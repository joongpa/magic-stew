"""Render saved CPU/WASM pipeline differences without running inference or fitting."""
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
from scripts.benchmark_cpu_pose_port import hydrate

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'pose-fitter/results/full-pipeline-review/differences.json')
    a=p.parse_args();data=json.loads(a.input.read_text());g=data['geometry'];rows=data['rows']
    for page in range((len(rows)+2)//3):
        selected=rows[page*3:page*3+3]
        fig,axes=plt.subplots(len(selected),2,figsize=(11,4.5*len(selected)),squeeze=False,layout='constrained')
        for index,row in enumerate(selected):
            item=row['input'];cpu=item['expected'][0];wasm=row['actual']['bank'][0]
            solution=json.loads((ROOT/REFERENCE_SOLUTIONS[row['name']]).read_text());log=Path(solution['config']['log']);meta=json.loads(log.with_suffix('.meta.json').read_text())
            cap=cv2.VideoCapture(meta['source']);cap.set(cv2.CAP_PROP_POS_FRAMES,item['frame']);ok,image=cap.read();cap.release();assert ok
            image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            base,_=hydrate(dict(camera=dict(rvec=cpu['params'][:3],tvec=cpu['params'][3:6],k=r.default_camera_matrix((item['width'],item['height'])).tolist(),dist=[0.]*4),observations=item['observations'],image_size=[item['width'],item['height']],retained=[],prior=None,neighbors=[],static_mode=False))
            stickers=np.asarray(item['observations']['stickers']);cloud=np.asarray(item['observations']['points']+item['observations']['stickers']);center=(cloud.min(axis=0)+cloud.max(axis=0))/2;radius=max(np.ptp(cloud,axis=0))*.65
            r1=cv2.Rodrigues(np.asarray(cpu['params'][:3]))[0];r2=cv2.Rodrigues(np.asarray(wasm['params'][:3]))[0];sym=np.asarray(g['symmetries']);which=np.einsum('nij,ij->n',r2@sym.transpose(0,2,1),r1).argmax();centers=np.asarray(g['faces']).mean(axis=1);permutation=np.linalg.norm((centers@sym[which].T)[:,None]-centers[None],axis=2).argmin(axis=1)
            va=np.asarray(cpu['valid']);vb=np.asarray(wasm['valid']);changed=(va!=vb)|(va&vb&(np.asarray(cpu['nearest'])!=permutation[np.asarray(wasm['nearest'])]))
            for column,fit in enumerate((cpu,wasm)):
                ax=axes[index,column];ax.imshow(image);ax.set_xlim(center[0]-radius,center[0]+radius);ax.set_ylim(center[1]+radius,center[1]-radius);ax.axis('off')
                rotations=r.parallel_layer_rotations(fit['axis'],fit['angles']);candidate=replace(base,rvec=np.asarray(fit['params'][:3]).reshape(3,1),tvec=np.asarray(fit['params'][3:6]).reshape(3,1),layer_rotations=rotations,sticker_layer_rotations=rotations)
                for _,poly,_ in r.projected_cubelet_faces(candidate,candidate.rvec,candidate.tvec):
                    poly=np.r_[poly,poly[:1]];ax.plot(poly[:,0],poly[:,1],color=('#00eac1' if column==0 else '#ff9b26'),linewidth=1.25)
                ax.scatter(stickers[:,0],stickers[:,1],s=10,c='white',edgecolors='black',linewidths=.5)
                ax.scatter(stickers[changed,0],stickers[changed,1],s=90,facecolors='none',edgecolors='#f32659',linewidths=1.5)
                for i in np.flatnonzero(changed):ax.annotate(str(i),stickers[i],xytext=(5,5),textcoords='offset points',color='#f32659',fontsize=8,bbox=dict(facecolor='white',alpha=.75,edgecolor='none',pad=.5))
                ax.set_title(f"{row['name'].title()} · frame {item['frame']} · {'Python CPU' if column==0 else 'Rust/WASM'}\n{changed.sum()} / {len(stickers)} sticker assignments differ",fontsize=11)
        fig.suptitle('Same saved detections and frame sequence · red circles mark changed assignments',fontsize=12)
        output=a.input.parent/f'differences-{page+1}.png';fig.savefig(output,dpi=150);plt.close(fig);print(output)

if __name__=='__main__':main()
