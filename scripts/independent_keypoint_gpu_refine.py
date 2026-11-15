"""Experimental batched articulated refinement; CPU scoring remains authoritative.

The tensor evaluator mirrors ArticulatedProjection's analytic residual/Jacobian.
Each seed has independent bounds, free parameters and damping. No candidate
pruning or detector-threshold changes are introduced.
"""
from __future__ import annotations

import numpy as np
import torch

from scripts.independent_keypoint_gpu_sweep import local_slab_entry


def skew(vector):
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*vector.shape[:-1], 3, 3)


def rotation_and_right_jacobian(vector):
    square = (vector*vector).sum(-1)
    angle = square.clamp(min=1e-16).sqrt()
    small = square < 1e-8
    a = torch.where(small, 1-square/6+square*square/120, angle.sin()/angle)
    b = torch.where(small, .5-square/24+square*square/720, (1-angle.cos())/square.clamp(min=1e-16))
    c = torch.where(small, 1/6-square/120+square*square/5040, (angle-angle.sin())/(square.clamp(min=1e-16)*angle))
    cross = skew(vector)
    cross2 = cross @ cross
    eye = torch.eye(3, dtype=vector.dtype, device=vector.device)
    return eye+a[..., None, None]*cross+b[..., None, None]*cross2, eye-b[..., None, None]*cross+c[..., None, None]*cross2


class BatchedArticulatedProjection:
    def __init__(self, renderer, candidate, axes, *, device='mps', dtype=None):
        self.renderer, self.base, self.axes = renderer, candidate, np.asarray(axes, int)
        self.device = str(device)
        if self.device.startswith('mps') and not torch.backends.mps.is_available():
            raise RuntimeError('MPS is unavailable in this process')
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable in this process')
        self.dtype = dtype or (torch.float32 if self.device.startswith('mps') else torch.float64)
        self.tensor = lambda value, kind=None: torch.as_tensor(np.array(value, copy=True), dtype=kind or self.dtype, device=self.device)
        t = self.tensor
        geometry = [renderer._parallel_slab_geometry(int(axis)) for axis in axes]
        faces = renderer._turn_gpu_static_geometry()
        self.layers = t(np.stack([g[0] for g in geometry]), torch.long)
        vertices = np.stack([g[1] for g in geometry])
        self.face_layers = t(np.stack([g[2] for g in geometry]), torch.long)
        self.points = t(renderer.CUBE_LATTICE_OBJECT_POINTS_MM[vertices])
        self.normals = t(renderer.CUBE_LATTICE_BOUNDARY_NORMALS[vertices])
        self.face_mask = t(renderer.CUBE_LATTICE_BOUNDARY_FACE_MASK[vertices], torch.bool)
        self.polygons = t(np.broadcast_to(faces[2], (len(axes), *faces[2].shape)))
        self.face_normals = t(np.broadcast_to(faces[5], (len(axes), *faces[5].shape)))
        self.unit_axes = t(np.eye(3)[axes])
        self.observed = t(np.asarray(candidate.assignment_image_points).reshape(-1, 2))
        self.stickers = t(np.asarray(candidate.sticker_mask_centers).reshape(-1, 2))
        observed_classes = (np.full(len(self.observed), -1) if candidate.assignment_observed_class_ids is None
                            else np.asarray(candidate.assignment_observed_class_ids))
        classes = renderer.CUBE_LATTICE_CLASS_IDS[vertices]
        self.compatible = t((observed_classes[None, :, None] < 0) | (classes[:, None] < 0)
                            | (observed_classes[None, :, None] == classes[:, None]), torch.bool)
        self.scale = max(renderer.projected_cube_diameter(candidate)/220., .3)
        self.camera_matrix = t(candidate.camera_matrix)
        coefficients = np.asarray(candidate.dist_coeffs).ravel()
        if len(coefficients) > 12 and np.any(coefficients[12:]):
            raise ValueError('GPU refinement does not support sensor tilt')
        self.distortion = t(np.pad(coefficients[:12], (0, max(0, 12-len(coefficients))))) if np.any(coefficients) else None
        lower = np.full((len(axes), 3, 3), -renderer.CUBE_HALF_SIZE_MM)
        upper = -lower.copy()
        for i, axis in enumerate(axes):
            lower[i, :, axis] += np.arange(3)*renderer.CUBE_LAYER_SIZE_MM
            upper[i, :, axis] = -renderer.CUBE_HALF_SIZE_MM+np.arange(1, 4)*renderer.CUBE_LAYER_SIZE_MM
        self.lower, self.upper = t(lower), t(upper)
        self.batch = torch.arange(len(axes), device=self.device)[:, None]

    def rotate_layers(self, objects, angles):
        # All slabs share an axis, so their relative rotation is the angle difference.
        axis = self.unit_axes.reshape(len(self.axes), *([1]*(objects.ndim-2)), 3)
        a = angles*(np.pi/180.)
        co, si = a.cos()[..., None], a.sin()[..., None]
        return objects*co+torch.linalg.cross(axis.expand_as(objects), objects)*si+axis*(axis*objects).sum(-1, keepdim=True)*(1-co)

    def project(self, objects, rotation, right_jacobian, translation):
        camera_objects = objects @ rotation.transpose(-1, -2)+translation[:, None]
        x, y, z = camera_objects.unbind(-1)
        xn, yn = x/z, y/z
        one, zero = torch.ones_like(xn), torch.zeros_like(xn)
        u, v = xn, yn
        dudx, dudy, dvdx, dvdy = one, zero, zero, one
        if self.distortion is not None:
            k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4 = self.distortion.unbind()
            r2 = xn*xn+yn*yn
            r4,r6 = r2*r2,r2*r2*r2
            numerator,denominator = 1+k1*r2+k2*r4+k3*r6,1+k4*r2+k5*r4+k6*r6
            radial = numerator/denominator
            derivative = ((k1+2*k2*r2+3*k3*r4)*denominator-numerator*(k4+2*k5*r2+3*k6*r4))/denominator**2
            u = xn*radial+2*p1*xn*yn+p2*(r2+2*xn*xn)+s1*r2+s2*r4
            v = yn*radial+p1*(r2+2*yn*yn)+2*p2*xn*yn+s3*r2+s4*r4
            dudx = radial+2*xn*xn*derivative+2*p1*yn+6*p2*xn+2*s1*xn+4*s2*r2*xn
            dudy = 2*xn*yn*derivative+2*p1*xn+2*p2*yn+2*s1*yn+4*s2*r2*yn
            dvdx = 2*xn*yn*derivative+2*p1*xn+2*p2*yn+2*s3*xn+4*s4*r2*xn
            dvdy = radial+2*yn*yn*derivative+6*p1*yn+2*p2*xn+2*s3*yn+4*s4*r2*yn
        fx,fy,cx,cy = self.camera_matrix[0,0],self.camera_matrix[1,1],self.camera_matrix[0,2],self.camera_matrix[1,2]
        pixels = torch.stack((fx*u+cx,fy*v+cy),-1)
        translation_jac = torch.stack((fx*dudx/z,fx*dudy/z,-fx*(dudx*xn+dudy*yn)/z,
            fy*dvdx/z,fy*dvdy/z,-fy*(dvdx*xn+dvdy*yn)/z),-1).reshape(*objects.shape[:-1],2,3)
        rotation_jac = -rotation[:,None] @ skew(objects) @ right_jacobian[:,None]
        rigid_jac = torch.cat((translation_jac @ rotation_jac,translation_jac),-1)
        axis = self.unit_axes[:,None].expand_as(objects)
        angle_derivative = torch.linalg.cross(axis, objects)*(np.pi/180.)
        angle_jac = (translation_jac @ (rotation[:,None] @ angle_derivative[...,None]))[...,0]
        return pixels,rigid_jac,angle_jac

    @torch.no_grad()
    def evaluate(self, parameters):
        x = parameters
        rotation,right_jacobian = rotation_and_right_jacobian(x[:,:3])
        camera = -(rotation.transpose(-1,-2) @ x[:,3:6,None])[...,0]
        angles = torch.stack((x[:,6],torch.zeros_like(x[:,6]),x[:,7]),-1)
        point_angles = angles.gather(1,self.layers)
        face_angles = angles.gather(1,self.face_layers)
        objects = self.rotate_layers(self.points,point_angles)
        polygons = self.rotate_layers(self.polygons,face_angles[:,:,None])
        normals = self.rotate_layers(self.normals,point_angles[:,:,None])
        face_objects = polygons.mean(-2)
        face_normals = self.rotate_layers(self.face_normals,face_angles)
        visible = (self.face_mask & ((normals*(camera[:,None,None]-objects[:,:,None])).sum(-1)>0)).any(-1)
        face_visible = (face_normals*(camera[:,None]-face_objects)).sum(-1)>0
        for layer in range(3):
            origin = self.rotate_layers(camera,-angles[:,layer])[:,None]
            local_points = self.rotate_layers(self.points,point_angles-angles[:,layer,None])
            local_centers = self.rotate_layers(self.polygons.mean(-2),face_angles-angles[:,layer,None])
            endpoints = torch.cat((local_points,local_centers),1)
            clear = local_slab_entry(origin,endpoints-origin,self.lower[:,layer,None],self.upper[:,layer,None])>=1.-1e-5
            visible = visible & clear[:,:objects.shape[1]]
            face_visible = face_visible & clear[:,objects.shape[1]:]
        points,point_jac,point_angle_jac = self.project(objects,rotation,right_jacobian,x[:,3:6])
        b,f,c,_ = polygons.shape
        pixels,face_jac,face_angle_jac = self.project(polygons.reshape(b,f*c,3),rotation,right_jacobian,x[:,3:6])
        centers = pixels.reshape(b,f,c,2).mean(2)
        face_jac = face_jac.reshape(b,f,c,2,6).mean(2)
        face_angle_jac = face_angle_jac.reshape(b,f,c,2).mean(2)
        distance = ((self.observed[None,:,None]-points[:,None])**2).sum(-1)
        distance = torch.where(self.compatible & visible[:,None],distance,torch.inf)
        minimum,columns = distance.min(-1)
        sd = ((self.stickers[None,:,None]-centers[:,None])**2).sum(-1)
        sd = torch.where(face_visible[:,None],sd,torch.inf)
        sticker_minimum,sticker_columns = sd.min(-1)
        residual = torch.cat((points[self.batch,columns]-self.observed,centers[self.batch,sticker_columns]-self.stickers),1)
        rigid_jac = torch.cat((point_jac[self.batch,columns],face_jac[self.batch,sticker_columns]),1)
        assigned_layers = torch.cat((self.layers.gather(1,columns),self.face_layers.gather(1,sticker_columns)),1)
        angle_jac = torch.cat((point_angle_jac[self.batch,columns],face_angle_jac[self.batch,sticker_columns]),1)
        jac = torch.cat((rigid_jac,angle_jac[:,:,:,None]*(assigned_layers[:,:,None,None]==self.tensor([0,2],torch.long))),-1)
        usable = torch.cat((torch.isfinite(minimum),torch.isfinite(sticker_minimum)),1)
        residual = torch.where(usable[:,:,None],residual/self.scale,40.)
        jac = torch.where(usable[:,:,None,None],jac/self.scale,0.)
        return residual.flatten(1),jac.flatten(1,2)


def refine_batch_tensor(renderer, candidate, axes, seeds, *, device='mps', prior=None, iterations=18, dtype=None,
                 return_diagnostics=False):
    """Projected, independently damped LM with the original robust objective."""
    model = BatchedArticulatedProjection(renderer,candidate,axes,device=device,dtype=dtype)
    t = model.tensor
    seed = t(seeds)
    free = torch.cat((torch.ones_like(seed[:,:6],dtype=torch.bool),seed[:,6:]!=0.),1)
    lower = seed-t([.8,.8,.8,40.,40.,100.,0.,0.])
    upper = seed+t([.8,.8,.8,40.,40.,100.,0.,0.])
    lower[:,6:],upper[:,6:] = -45.,45.
    lower[:,5] = lower[:,5].clamp(min=renderer.CUBE_SIZE_MM+1.)
    scales = t([.25,.25,.25,8.,8.,25.,15.,15.])
    sigmas = t([.6,.6,.6,25.,25.,70.,180.,180.])
    prior = torch.cat((seed[:,:6] if prior is None else t(np.broadcast_to(prior,(len(seed),6))),torch.zeros_like(seed[:,6:])),1)
    regularizer = torch.diag(1/sigmas).expand(len(seed),-1,-1)
    def evaluate(x):
        residual,jac = model.evaluate(x)
        residual = torch.cat((residual,(x-prior)/sigmas),1)
        jac = torch.cat((jac,regularizer),1)*free[:,None]
        root = torch.sqrt(1+(residual/3.)**2)
        cost = (9.*(root-1)).sum(1)
        # SciPy soft_l1's robust Gauss-Newton Jacobian scaling.
        weight = root**-1.5
        weighted_jac = jac*weight[:,:,None]
        gradient = torch.einsum('bri,br->bi',jac,residual/root)*scales
        scaled_jac = weighted_jac*scales
        hessian = scaled_jac.transpose(1,2) @ scaled_jac
        return cost,gradient,hessian
    x = seed.clone()
    cost,gradient,hessian = evaluate(x)
    initial_cost = cost.clone()
    damping = torch.full((len(seed),),1e-2,dtype=model.dtype,device=device)
    accepted = torch.zeros(len(seed),dtype=torch.long,device=device)
    for _ in range(iterations-1):
        diagonal = hessian.diagonal(dim1=-2,dim2=-1).clamp(min=1e-8)
        system = hessian+torch.diag_embed(damping[:,None]*diagonal+(~free).to(model.dtype))
        delta = torch.linalg.solve(system,-gradient[:,:,None])[...,0]*scales
        proposal = torch.maximum(lower,torch.minimum(upper,x+delta))
        proposal = torch.where(free,proposal,seed)
        new_cost,new_gradient,new_hessian = evaluate(proposal)
        good = torch.isfinite(new_cost)&(new_cost<cost)
        step = (proposal-x)/scales
        predicted = -(gradient*step).sum(1)-.5*torch.einsum('bi,bij,bj->b',step,hessian,step)
        ratio = (cost-new_cost)/predicted.clamp(min=1e-12)
        damping = torch.where(good & (ratio>.75),damping/3.,torch.where(~good | (ratio<.25),damping*4.,damping)).clamp(1e-8,1e8)
        x = torch.where(good[:,None],proposal,x)
        cost = torch.where(good,new_cost,cost)
        gradient = torch.where(good[:,None],new_gradient,gradient)
        hessian = torch.where(good[:,None,None],new_hessian,hessian)
        accepted += good
    result = x.cpu().numpy()
    if return_diagnostics:
        return result,dict(initial_cost=initial_cost.cpu().numpy(),final_cost=cost.cpu().numpy(),accepted_steps=accepted.cpu().numpy())
    return result


def refine_batch(renderer, candidate, axes, seeds, *, device='mps', prior=None, iterations=18, dtype=None,
                 return_diagnostics=False):
    if str(device).startswith('mps'):
        from scripts.independent_keypoint_metal_refine import refine_batch_metal
        return refine_batch_metal(renderer,candidate,axes,seeds,prior=prior,iterations=iterations,
                                  return_diagnostics=return_diagnostics)
    return refine_batch_tensor(renderer,candidate,axes,seeds,device=device,prior=prior,iterations=iterations,
                               dtype=dtype,return_diagnostics=return_diagnostics)
