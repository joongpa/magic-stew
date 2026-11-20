"""Experimental partial rigid proposals, then one buffered articulated fit.

Current saved cameras/angles are never read by static_candidates. Partial slab
labels are scoring metadata, not fixed anchors or articulation-axis constraints.
The second stage allows all axes and jointly refines rigid pose and outer angles.
"""
from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from time import perf_counter

import cv2
import numpy as np
import torch

from scripts import independent_keypoint_pose_geometry as geometry
from scripts.independent_keypoint_slab_search import SlabScorer, static_shortlist
from scripts.independent_keypoint_gpu_sweep import static_tensors, stacked_static_tensors, _tensor_sweep, _export_options
from scripts.independent_keypoint_gpu_refine import refine_batch


def observation_template(candidate):
    """Calibration and detections only; erase saved pose and all assignments."""
    return replace(candidate,rvec=np.zeros((3,1)),tvec=np.array([[0.],[0.],[300.]]),
        layer_rotations=(),sticker_layer_rotations=None,assignment_object_points=None,
        assignment_projected_points=None,assignment_projected_class_ids=None,joint_geometry_candidates=(),
        image_points=np.empty((0,2)),object_points=np.empty((0,3)),projected_sticker_points=np.empty((0,2)))


def static_candidates(renderer, observations, *, beam=16, neighborhoods=8, groups=6):
    """Fresh per-frame sticker PnP followed only by rigid subset fitting."""
    started=perf_counter()
    base=observation_template(observations)
    points=np.asarray(base.assignment_image_points).reshape(-1,2)
    classes=np.asarray(base.assignment_observed_class_ids)
    stickers=np.asarray(base.sticker_mask_centers).reshape(-1,2)
    matches={}
    # Confidence scores are absent from these already-filtered logs. NaN marks
    # them missing in the existing geometric group validator; ranking below
    # uses only geometry and never thresholds or interprets these values.
    absent_scores=np.full(len(points),np.nan)
    for sticker,center in enumerate(stickers):
        nearest=np.argsort(np.linalg.norm(points-center,axis=1),kind='stable')[:neighborhoods]
        others=np.delete(stickers,sticker,axis=0)
        blocked=renderer.sticker_blocked_landmark_pairs(points,center,others)
        for count in (4,3):
            choices=[]
            for size in range(count,len(nearest)+1):
                newest=int(nearest[size-1])
                for preceding in combinations(map(int,nearest[:size-1]),count-1):
                    indices=(*preceding,newest)
                    if any(blocked[i,j] for i,j in combinations(indices,2)):continue
                    if renderer.sticker_type_from_keypoint_classes(classes[list(indices)].tolist())=='unknown':continue
                    found=renderer.sticker_center_alignment_matches(points,classes,absent_scores,indices,center,sticker,others)
                    for match in found:
                        choices.append(match)
                if choices:break
            if choices:
                best=min(choices,key=lambda m:(m.center_error,sum(np.linalg.norm(points[list(m.keypoint_indices)]-center,axis=1)),m.sticker_type))
                # Preserve ambiguous sticker types for the same real landmarks.
                for match in choices:
                    if set(match.keypoint_indices)==set(best.keypoint_indices) and abs(match.center_error-best.center_error)<1e-9:
                        matches[(tuple(match.keypoint_indices),match.sticker_type)]=match
    ordered=sorted(matches.values(),key=lambda m:(m.center_error,-len(m.keypoint_indices),m.sticker_type))
    selected=[];used=[]
    for match in ordered:
        key=tuple(sorted(match.keypoint_indices))
        if key not in used:
            if len(used)>=groups:continue
            used.append(key)
        selected.append(match)
    roots=[]
    for match in selected:
        measured=points[list(match.keypoint_indices)].astype(float)
        canonical=renderer.canonical_sticker_object_points(match.sticker_type)
        for indices in match.object_index_sequences:
            objects=np.asarray(canonical[list(indices)],float)
            if len(measured)==3:
                solutions=renderer.three_point_sticker_pose_roots(objects,measured,base.camera_matrix,base.dist_coeffs)
            else:
                try:
                    ok,rvecs,tvecs,_=cv2.solvePnPGeneric(objects,measured,base.camera_matrix,base.dist_coeffs,flags=cv2.SOLVEPNP_IPPE)
                    solutions=tuple(zip(rvecs,tvecs)) if ok else ()
                except cv2.error:solutions=()
            for rvec,tvec in solutions:
                if (np.isfinite(rvec).all() and np.isfinite(tvec).all() and tvec[2,0]>renderer.CUBE_SIZE_MM+1.
                    and renderer.canonical_sticker_is_camera_facing(objects,rvec,tvec)):
                    roots.append(np.r_[rvec.ravel(),tvec.ravel()])
    if not roots:
        return [],dict(milliseconds=1000*(perf_counter()-started),raw_roots=0,groups=len(used),retained=0)
    base=replace(base,rvec=roots[0][:3].reshape(3,1),tvec=roots[0][3:6].reshape(3,1))
    scorer=SlabScorer(renderer,base)
    # Determine support scale from current detections' PnP roots, not a saved fit.
    diameters=[renderer.projected_cube_diameter(base,rvec=x[:3].reshape(3,1),tvec=x[3:6].reshape(3,1)) for x in roots]
    scorer.diameter=float(np.median([d for d in diameters if np.isfinite(d) and d>0]))
    ranked=scorer.score(roots,[-1]*len(roots))
    supported=[p for p in ranked if p.landmark_support>=4 and p.sticker_support>=1 and p.spatial_support]
    proposals=static_shortlist(renderer,supported or ranked,count=beam,separation_degrees=3.)
    proposals=[scorer.refine(p,iterations=2) for p in proposals]
    proposals=static_shortlist(renderer,proposals,count=beam,separation_degrees=3.)
    return proposals,dict(milliseconds=1000*(perf_counter()-started),raw_roots=len(roots),groups=len(used),
        retained=len(proposals),supported=len(supported),best_partial_score=proposals[0].score,
        best_landmark_support=proposals[0].landmark_support,best_sticker_support=proposals[0].sticker_support)


def link_cost(renderer, first, second):
    a,b=first.parameters,second.parameters
    r1,r2=cv2.Rodrigues(a[:3])[0],cv2.Rodrigues(b[:3])[0]
    trace=np.einsum('nij,ij->n',r2@np.asarray(renderer.CUBE_SYMMETRY_MATRICES),r1).max()
    angle=np.rad2deg(np.arccos(np.clip((trace-1)/2,-1,1)))
    return .006*(min(np.linalg.norm(a[3:6]-b[3:6])/15.,4.)+min(angle/30.,3.))


def buffered_cameras(renderer, current, previous=(), following=(), *, budget=6):
    """One-frame lookahead; retain alternatives instead of averaging rotations."""
    def ordered(bank,nearby):
        ranked=[]
        for proposal in bank:
            score=proposal.score
            for other in nearby:
                if other:
                    best=min(p.score for p in other)
                    score+=min(link_cost(renderer,proposal,p)+.5*(p.score-best) for p in other)
            ranked.append((score,proposal))
        return [p for _,p in sorted(ranked,key=lambda x:x[0])]
    center=ordered(current,[previous,following])
    origins=list(center[:max(1,budget-2)])
    if previous:origins+=ordered(previous,[current])[:1]
    if following:origins+=ordered(following,[current])[:1]
    if len(origins)<budget:origins+=center[max(1,budget-2):budget]
    # Different partial slabs can score one camera; remove only near-exact poses.
    return static_shortlist(renderer,[replace(p,score=float(i)) for i,p in enumerate(origins)],
        count=budget,separation_degrees=2.)


def _camera_grid(renderer, base, cameras, *, device='mps', per_group=10):
    """All cameras and axes in one tensor sweep; construct only shortlisted rows."""
    dtype=torch.float32 if str(device).startswith('mps') else torch.float64
    tensor=lambda v,kind=dtype:torch.as_tensor(np.asarray(v),dtype=kind,device=device)
    axes=tuple(axis for camera in cameras for axis in range(3))
    models=[static_tensors(renderer,axis,device,dtype) for axis in axes]
    # Index the three cached model tables; avoid rebuilding them per frame.
    cache=stacked_static_tensors(renderer,(0,1,2),device,dtype)
    index=tensor(axes,torch.long)
    data={k:v[index] for k,v in cache.items()}
    rotations=[];translations=[];diameter_rows=[];scales=[];poses=[]
    reference=cv2.Rodrigues(base.rvec)[0]
    for proposal in cameras:
        x=proposal.parameters
        r=renderer._align_rigid_rotation(x[:3].reshape(3,1),reference)
        pose=replace(base,rvec=cv2.Rodrigues(r)[0],tvec=x[3:6].reshape(3,1))
        scale=renderer.projected_cube_diameter(pose)
        poses.append(pose)
        for axis in range(3):
            rotations.append(r);translations.append(x[3:6]);scales.append(scale)
            model=models[len(rotations)-1]
            middle=model['values'][model['grid_numpy'][:,1]]
            row=np.empty(len(middle))
            for value in np.unique(middle):
                rv=cv2.Rodrigues(r@cv2.Rodrigues(np.eye(3)[axis]*np.deg2rad(value))[0])[0]
                row[middle==value]=renderer.projected_cube_diameter(pose,rvec=rv)
            diameter_rows.append(row)
    R,T=tensor(rotations),tensor(translations)
    camera=-(R.transpose(-1,-2)@T[:,:,None])[:,:,0]
    K=tensor(base.camera_matrix)
    coeff=np.asarray(base.dist_coeffs).ravel()
    if len(coeff)>12 and np.any(coeff[12:]):raise ValueError('Sensor tilt unsupported')
    distortion=tensor(np.pad(coeff[:12],(0,max(0,12-len(coeff))))) if np.any(coeff) else None
    observed=tensor(np.asarray(base.assignment_image_points).reshape(-1,2))
    stickers_np=np.asarray(base.sticker_mask_centers).reshape(-1,2)
    stickers=tensor(stickers_np)
    classes=(tensor(base.assignment_observed_class_ids,torch.long) if base.assignment_observed_class_ids is not None
        and renderer.PNP_ASSIGNMENT_MODE=='class-aware' else None)
    uncertainty=np.asarray(base.sticker_mask_center_uncertainties,float)
    if len(uncertainty)!=len(stickers_np):uncertainty=np.ones(len(stickers_np))
    uncertainty=np.where(np.isfinite(uncertainty),np.maximum(uncertainty,1.),np.inf)
    uncertainty,weights=tensor(uncertainty),tensor(1/uncertainty**2)
    normalized=cv2.undistortPoints(stickers_np,base.camera_matrix,base.dist_coeffs).reshape(-1,2)
    rays=tensor(np.c_[normalized,np.ones(len(normalized))])[None]@R
    limits=tensor(scales)*renderer.PNP_FINAL_ASSIGNMENT_ERROR_FRACTION
    def sweep(model,r,t,c,ray,diameter,limit,scale):
        inputs=(r,t,c,K,distortion,observed,stickers,classes,uncertainty,weights,ray,limit,scale)
        return _tensor_sweep(renderer,model,inputs,diameter,device,dtype)
    exported=torch.vmap(sweep)(data,R,T,camera,rays,tensor(diameter_rows),limits,tensor(scales)).cpu().numpy()
    scores=renderer.balanced_geometry_cost(exported[:,:,0],exported[:,:,5])
    grid=models[0]['values'][models[0]['grid_numpy']]
    relative=(grid-grid[:,1,None]+45.)%90.-45.
    masks=relative[:,[0,2]]!=0
    chosen=set()
    for axis in range(3):
        model_ids=np.arange(axis,len(axes),3)
        for mask in [(False,False),(True,False),(False,True),(True,True)]:
            row_ids=np.flatnonzero(np.all(masks==mask,axis=1))
            flat=scores[np.ix_(model_ids,row_ids)].ravel()
            for flat_id in np.argsort(flat,kind='stable')[:per_group]:
                chosen.add((int(model_ids[flat_id//len(row_ids)]),int(row_ids[flat_id%len(row_ids)])))
    # Protect globally good unrefined measurements independently of seed groups.
    for flat_id in np.argsort(scores,axis=None,kind='stable')[:32]:chosen.add((int(flat_id//342),int(flat_id%342)))
    fits=[]
    for model_id in sorted(set(i for i,j in chosen)):
        ids=np.array(sorted(j for i,j in chosen if i==model_id))
        axis=axes[model_id];model=models[model_id]
        subset=dict(values=model['values'],grid_numpy=model['grid_numpy'][ids])
        options,_,_=_export_options(renderer,axis,subset,exported[model_id,ids],len(stickers_np))
        pose=poses[model_id//3];r=rotations[model_id]
        for option in options:
            angles=np.asarray(renderer.parallel_layer_angles(option['layer_rotations']))
            rel=(angles-angles[1]+45.)%90.-45.
            rel[(rel==-45.)&(angles-angles[1]>0)]=45.
            rv=cv2.Rodrigues(r@cv2.Rodrigues(np.eye(3)[axis]*np.deg2rad(angles[1]))[0])[0]
            rotations_layer=renderer.parallel_layer_rotations(axis,rel)
            candidate=replace(pose,rvec=rv,layer_rotations=rotations_layer)
            fits.append(geometry.PoseFit(candidate,axis,tuple(rel),option,None,None,
                float(renderer.balanced_geometry_cost(option['keypoint_error'],option['sticker_error']))))
    for pose in poses:
        for axis in range(3):
            limit=renderer.PNP_FINAL_ASSIGNMENT_ERROR_FRACTION*renderer.projected_cube_diameter(pose)
            option=renderer.parallel_turn_frame_geometry(pose,limit,axis,angles=(0.,0.,0.))[0][0]
            fits.append(geometry.PoseFit(pose,axis,(0.,0.,0.),option,None,None,
                float(renderer.balanced_geometry_cost(option['keypoint_error'],option['sticker_error']))))
    return fits,dict(cameras=len(cameras),configurations=len(cameras)*1026,constructed_proposals=len(fits))


def fit_buffered(renderer, observations, current, previous=(), following=(), *, device='mps',camera_budget=6):
    started=perf_counter()
    cameras=buffered_cameras(renderer,current,previous,following,budget=camera_budget)
    if not cameras:return [],dict(milliseconds=1000*(perf_counter()-started),status='no_static_proposals')
    base=observation_template(observations)
    origin=cameras[0].parameters
    base=replace(base,rvec=origin[:3].reshape(3,1),tvec=origin[3:6].reshape(3,1))
    proposals,diagnostics=_camera_grid(renderer,base,cameras,device=device)
    seeds=[]
    for axis in range(3):
        for mask in [(False,False),(True,False),(False,True),(True,True)]:
            group=[f for f in proposals if f.axis==axis and tuple(np.asarray(f.angles)[[0,2]]!=0)==mask]
            seeds.extend(geometry.diverse_shortlist(renderer,group,size=2)[:2])
    starts=[np.r_[f.candidate.rvec.ravel(),f.candidate.tvec.ravel(),np.asarray(f.angles)[[0,2]]] for f in seeds]
    parameters=refine_batch(renderer,base,[f.axis for f in seeds],starts,device=device,prior=origin)
    fits=[]
    for f in geometry.diverse_shortlist(renderer,proposals):
        fit=geometry._materialize(renderer,base,f.axis,np.r_[f.candidate.rvec.ravel(),f.candidate.tvec.ravel(),np.asarray(f.angles)[[0,2]]])
        if fit is not None:fits.append(fit)
    for f,x in zip(seeds,parameters,strict=True):
        fit=geometry._materialize(renderer,base,f.axis,x)
        if fit is not None:fits.append(replace(fit,proposal_source='deferred_partial_then_joint'))
    fits=geometry.diverse_shortlist(renderer,fits)
    diagnostics.update(milliseconds=1000*(perf_counter()-started),optimizer_starts=len(seeds),retained=len(fits),
        status='fitted',best_score=fits[0].score if fits else None)
    return fits,diagnostics
