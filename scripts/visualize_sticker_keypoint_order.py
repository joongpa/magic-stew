#!/usr/bin/env python3
"""Verify proposed sticker outline ordering on diagrams and real dataset images."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from live_recognition.sticker_keypoint_order import clockwise_sticker_order, center_cyclic_orders
from live_recognition.direct_cube_pose import articulated_landmarks, project_points
from live_recognition.direct_cube_pose_data import target_tensors
from live_recognition.staged_pose.geometry import ClassNearestGeometry

ROLE_NAMES = ("C", "CE", "CEC")
ROLE_COLORS = ("#ffffff", "#ffd447", "#df83f4")
TYPE_COLORS = dict(corner="#da6b2b", edge="#2875b5", center="#8651af")
SEQUENCES = dict(corner="C → CE → CEC → CE", edge="CE → CE → CEC → CEC", center="CEC → CEC → CEC → CEC")


def grid_cell(x, y):
    q = np.array([(x,y),(x+1,y),(x+1,y+1),(x,y+1)], dtype=float)
    roles = np.array([0 if (a in (0,3) and b in (0,3)) else
                      1 if (a in (0,3) or b in (0,3)) else 2 for a,b in q])
    kind = "corner" if 0 in roles else "edge" if 1 in roles else "center"
    order = clockwise_sticker_order(q, roles, kind)
    return q[order], roles[order], kind


def draw_quad(ax, q, roles, *, detail=True, star=False, color="#147f62", number_size=15):
    middle = q.mean(0)
    ax.fill(q[:,0], q[:,1], color=color, alpha=.20, zorder=1)
    ax.plot(*np.vstack((q,q[0])).T, color=color, lw=2, zorder=2)
    if detail:
        ax.scatter(q[:,0],q[:,1], s=190, c=[ROLE_COLORS[i] for i in roles],
                   edgecolors="#1f2937", linewidths=1.2, zorder=4)
    for i in range(4):
        a = middle + .72*(q[i]-middle)
        b = middle + .72*(q[(i+1)%4]-middle)
        direction = b-a
        ax.annotate("", xy=b-.08*direction, xytext=a+.12*direction,
            arrowprops=dict(arrowstyle="-|>",color=color,lw=1.6,mutation_scale=13), zorder=3)
        position = middle + .69*(q[i]-middle)
        label = str(i) + ("*" if star and i==0 else "")
        if detail: label += "\n"+ROLE_NAMES[roles[i]]
        ax.text(*position,label,ha="center",va="center",fontsize=number_size,
            fontweight="bold",color="#b22b2b" if i==0 else "#16212b", zorder=5,
            bbox=dict(facecolor="white",edgecolor="none",alpha=.95,pad=1.4))


def diagrams(output):
    fig, axes = plt.subplots(1,3,figsize=(13.5,4.7),layout="constrained")
    for ax,(x,y,kind) in zip(axes,[(0,0,"corner"),(1,0,"edge"),(1,1,"center")]):
        q,roles,_=grid_cell(x,y)
        draw_quad(ax,q,roles,star=kind=="center")
        if kind=="corner":
            ax.plot([x+1,x,x],[y,y,y+1],color="#222",lw=5,zorder=0)
        elif kind=="edge":
            ax.plot([x,x+1],[y,y],color="#222",lw=5,zorder=0)
        ax.set(xlim=(x-.13,x+1.13),ylim=(y+1.16,y-.13),aspect="equal")
        ax.axis("off")
        ax.set_title(kind.capitalize()+" sticker\n"+SEQUENCES[kind],fontsize=15,pad=16)
        ax.text(x+.5,y+1.12,"four equivalent starts" if kind=="center" else
            "0 = cube corner" if kind=="corner" else "0 → 1 = outside edge",
            ha="center",va="top",fontsize=12)
    fig.suptitle("Proposed keypoint order: clockwise in the image",fontsize=19)
    fig.supxlabel("Black thick lines: outer cube-face boundary. Colored dots: existing landmark roles.\n* Center point 0 is arbitrary; cyclic shifts are equivalent. These are cell/grid corners, not inset printed-sticker corners.",fontsize=10)
    fig.savefig(output/'ordering_rules.png',dpi=170);plt.close(fig)

    fig=plt.figure(figsize=(13.5,7.5),layout="constrained")
    gs=fig.add_gridspec(2,4,width_ratios=[1.45,1.45,1,1])
    ax=fig.add_subplot(gs[:,:2])
    for y in range(3):
        for x in range(3):
            q,roles,kind=grid_cell(x,y)
            draw_quad(ax,q,roles,detail=False,star=kind=='center',number_size=14)
            ax.text(x+.5,y+.5,kind,ha='center',va='center',fontsize=12)
    for y in range(4):
        for x in range(4):
            role=0 if x in (0,3) and y in (0,3) else 1 if x in (0,3) or y in (0,3) else 2
            ax.scatter([x],[y],s=120,c=[ROLE_COLORS[role]],edgecolors='#222',zorder=6)
    ax.plot([0,3,3,0,0],[0,0,3,3,0],color='#222',lw=3)
    ax.set(xlim=(-.1,3.1),ylim=(3.1,-.1),aspect='equal');ax.axis('off')
    ax.set_title('All nine cells: start follows the geometric role\n36 keypoint occurrences share 16 grid positions',fontsize=14)
    base,roles,_=grid_cell(1,1)
    for k,indices in enumerate(center_cyclic_orders()):
        ax=fig.add_subplot(gs[k//2,2+k%2])
        draw_quad(ax,base[indices],roles[indices],detail=False,number_size=15)
        ax.set(xlim=(.85,2.15),ylim=(2.15,.85),aspect='equal');ax.axis('off')
        ax.set_title(f'Valid center target {k+1}',fontsize=12)
    fig.suptitle('Corner/edge starts are fixed; center has four clockwise targets',fontsize=18)
    fig.supxlabel('Center loss: select the lowest-loss cyclic target. Rotate coordinates AND visibility together.\nClockwise winding excludes reflected/reversed orders. Do not average the four target coordinates.',fontsize=11)
    fig.savefig(output/'face_and_center_symmetry.png',dpi=170);plt.close(fig)


@torch.inference_mode()
def dataset_examples(root, output, keys):
    g=ClassNearestGeometry()
    fig,axes=plt.subplots(len(keys),4,figsize=(15,4.2*len(keys)),layout='constrained',squeeze=False)
    report=[]
    for row,key in enumerate(keys):
        payload=json.loads((root/'targets'/f'{key}.json').read_text())
        target=target_tensors(payload)
        p={k:target['rotation' if k=='rotation_matrix' else k][None] for k in
           ['rotation_matrix','center','scale','perspective','principal','angles']}
        family=p['angles'].abs().reshape(1,3,2).sum(-1).argmax(-1)
        _,_,_,info=g.project(p,family)
        xyz=articulated_landmarks(g.quads.flatten(0,1),g.sticker_owners.repeat_interleave(4,0),p['angles'])
        width,height=payload['image_size']
        uv=project_points(xyz,p)[0].reshape(54,4,2).numpy()*[width,height]
        image=Image.open(root/'images'/f'{key}.png').convert('RGB')
        observations=payload['matched_observations']
        observed=np.asarray([o['observed_pixel'] for o in observations],float).reshape(-1,2)
        observed_types=np.asarray([o['class_id'] for o in observations])
        selected={}
        for index in range(54):
            if not bool(info['sticker_visible'][0,index]):continue
            q=uv[index]
            if (q<0).any() or (q[:,0]>=width).any() or (q[:,1]>=height).any():continue
            roles=(g.quads[index].abs()>1.49).sum(-1).numpy()
            roles=3-roles
            count=int((g.sticker_owners[index].abs()>.5).sum())
            kind={1:'center',2:'edge',3:'corner'}[count]
            order=clockwise_sticker_order(q,roles,kind);q=q[order];roles=roles[order]
            # Choose examples with direct visible landmark support for every
            # corner, rather than displaying hidden projections as annotations.
            distance=np.linalg.norm(q[:,None]-observed[None],axis=-1)
            distance=np.where(roles[:,None]==observed_types[None],distance,np.inf)
            if (distance.min(-1)>1.).any():continue
            area=.5*abs(np.sum(q[:,0]*np.roll(q[:,1],-1)-q[:,1]*np.roll(q[:,0],-1)))
            if kind not in selected or area>selected[kind]['area']:
                selected[kind]=dict(index=index,q=q,roles=roles,area=float(area))
        assert set(selected)=={'corner','edge','center'},(key,selected.keys())
        overview=axes[row,0];overview.imshow(image);overview.axis('off')
        overview.set_title(payload['layer_state_mode']+'\n'+key.replace('keypoints_only_dataset','dataset'),fontsize=10)
        for column,kind in enumerate(['corner','edge','center'],1):
            item=selected[kind];q=item['q'];roles=item['roles'];color=TYPE_COLORS[kind]
            overview.plot(*np.vstack((q,q[0])).T,color=color,lw=2)
            ax=axes[row,column];ax.imshow(image)
            ax.plot(*np.vstack((q,q[0])).T,color=color,lw=2)
            center=q.mean(0);side=np.ptp(q,axis=0).max()*1.7
            ax.set(xlim=(center[0]-side/2,center[0]+side/2),ylim=(center[1]+side/2,center[1]-side/2))
            ax.axis('off');ax.set_title(kind.capitalize()+'\n'+SEQUENCES[kind],fontsize=12)
            for i in range(4):
                label=f'{i}'+('*' if kind=='center' and i==0 else '')+'\n'+ROLE_NAMES[roles[i]]
                ax.scatter(*q[i],s=85,c=ROLE_COLORS[roles[i]],edgecolors='#111',zorder=4)
                shift=(q[i]-center);shift=shift/(np.linalg.norm(shift)+1e-9)*(side*.10)
                ax.text(*(q[i]+shift),label,ha='center',va='center',fontsize=12,fontweight='bold',
                    color='white',path_effects=[pe.withStroke(linewidth=3,foreground='black')],zorder=5)
                a=center+.75*(q[i]-center);b=center+.75*(q[(i+1)%4]-center)
                ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',lw=2,color='cyan'),zorder=3)
            report.append(dict(key=key,type=kind,template_index=item['index'],points=q.tolist(),
                old_classes=roles.tolist(),clockwise_pixel_area=item['area'],all_corners_match_visible_labels_within_px=1.))
    fig.suptitle('Ordering on existing dataset images — same rule after articulation',fontsize=18)
    fig.supxlabel('Corners follow their original cubie-face ownership. Every displayed corner is supported by an existing visible landmark label.\nCenter numbering shown is one of four equivalent targets. This preview does not change training labels.',fontsize=10)
    fig.savefig(output/'dataset_examples.png',dpi=170);plt.close(fig)
    (output/'example_points.json').write_text(json.dumps(report,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT/'runs/sticker_quad_ordering_verification')
    p.add_argument('--dataset',type=Path,default=ROOT/'data/direct_cube_pose_dataset')
    p.add_argument('--keys',nargs='+',default=['keypoints_only_dataset__000051','keypoints_only_dataset__000074'])
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    diagrams(args.output_dir);dataset_examples(args.dataset,args.output_dir,args.keys)
    print(args.output_dir)


if __name__=='__main__':main()
