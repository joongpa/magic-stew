"""Fused Metal kernels for experimental batched articulated refinement."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch


@lru_cache(maxsize=1)
def shaders():
    return torch.mps.compile_shader(Path(__file__).with_name('independent_keypoint_refine.metal').read_text())


class MetalArticulatedProjection:
    def __init__(self, renderer, candidate, axes):
        self.axes=np.asarray(axes,int)
        self.base=candidate
        self.renderer=renderer
        faces=renderer._turn_gpu_static_geometry()
        models=[renderer._parallel_slab_geometry(int(axis)) for axis in axes]
        self.point_count=len(models[0][0])
        self.face_count=len(faces[2])
        count=self.point_count+self.face_count
        objects=np.zeros((len(axes),count,30),np.float32)
        ids=np.zeros((len(axes),count,2),np.int32)
        for i,(layers,vertices,face_layers,_,_) in enumerate(models):
            objects[i,:self.point_count,:3]=renderer.CUBE_LATTICE_OBJECT_POINTS_MM[vertices]
            normals=renderer.CUBE_LATTICE_BOUNDARY_NORMALS[vertices]*renderer.CUBE_LATTICE_BOUNDARY_FACE_MASK[vertices,:,None]
            objects[i,:self.point_count,12:12+normals.shape[1]*3]=normals.reshape(self.point_count,-1)
            objects[i,self.point_count:,:12]=faces[2].reshape(self.face_count,12)
            objects[i,self.point_count:,12:15]=faces[5]
            ids[i,:self.point_count,0]=layers
            ids[i,:self.point_count,1]=renderer.CUBE_LATTICE_CLASS_IDS[vertices]
            ids[i,self.point_count:,0]=face_layers
        self.tensor=lambda value:torch.as_tensor(np.array(value,copy=True),device='mps')
        self.objects=self.tensor(objects)
        self.ids=self.tensor(ids)
        self.axis_tensor=self.tensor(self.axes.astype(np.int32))
        points=np.asarray(candidate.assignment_image_points,float).reshape(-1,2)
        centers=np.asarray(candidate.sticker_mask_centers,float).reshape(-1,2)
        self.observations=self.tensor(np.vstack((points,centers)).astype(np.float32))
        classes=np.full(len(points),-1) if candidate.assignment_observed_class_ids is None else candidate.assignment_observed_class_ids
        self.classes=self.tensor(np.asarray(classes,np.int32))
        self.config=self.tensor(np.array([len(axes),self.point_count,self.face_count,len(points),len(centers)],np.int32))
        k=candidate.camera_matrix
        dist=np.asarray(candidate.dist_coeffs).ravel()
        if len(dist)>12 and np.any(dist[12:]):raise ValueError('Metal refinement does not support sensor tilt')
        calibration=np.r_[k[0,0],k[1,1],k[0,2],k[1,2],np.pad(dist[:12],(0,max(0,12-len(dist)))),renderer.CUBE_HALF_SIZE_MM].astype(np.float32)
        self.calibration=self.tensor(calibration)
        self.scale=max(renderer.projected_cube_diameter(candidate)/220.,.3)
        self.projections=torch.empty((len(axes),count,19),device='mps')
        self.residuals=torch.empty((len(axes),len(points)+len(centers),18),device='mps')
        self.library=shaders()

    def evaluate(self,parameters):
        self.library.geometry_project(parameters,self.objects,self.ids,self.axis_tensor,self.calibration,self.config,self.projections,
            threads=len(self.axes)*(self.point_count+self.face_count))
        self.library.match_residuals(self.projections,self.ids,self.observations,self.classes,self.config,self.scale,self.residuals,
            threads=len(self.axes)*self.observations.shape[0])
        return self.residuals


def refine_batch_metal(renderer,candidate,axes,seeds,*,prior=None,iterations=18,return_diagnostics=False):
    model=MetalArticulatedProjection(renderer,candidate,axes)
    seeds_np=np.asarray(seeds,np.float32)
    parameters=model.tensor(seeds_np)
    seeds_tensor=parameters.clone()
    priors=np.zeros_like(seeds_np)
    priors[:,:6]=seeds_np[:,:6] if prior is None else prior
    priors=model.tensor(priors)
    states=torch.zeros((len(seeds_np),83),device='mps')
    for iteration in range(iterations):
        model.evaluate(parameters)
        model.library.optimizer_step(parameters,seeds_tensor,priors,model.residuals,model.config,iteration,
            float(renderer.CUBE_SIZE_MM+1.),states,threads=len(seeds_np))
    exported=states.cpu().numpy()
    result=exported[:,:8].astype(float)
    if return_diagnostics:
        return result,dict(initial_cost=exported[:,82],final_cost=exported[:,8])
    return result
