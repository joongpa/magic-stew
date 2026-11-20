"""Opt-in tensor implementation of the established exhaustive slab sweep.

Projection, visibility, assignment and scoring remain on one Torch device.
Only immutable model geometry and small per-camera calibration are prepared on
CPU. Refinement and final explicit-angle materialization remain unchanged.
MPS uses float32; CPU float64 is available for numerical port validation.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import product

import cv2
import numpy as np
import torch


@lru_cache(maxsize=12)
def static_tensors(renderer, axis, device, dtype):
    values = np.arange(-45., 46., 15.)
    matrices = np.asarray([cv2.Rodrigues(np.eye(3)[axis]*np.deg2rad(a))[0] for a in values[:-1]])
    endpoint = cv2.Rodrigues(np.eye(3)[axis]*np.deg2rad(values[-1]))[0]
    layers, vertices, face_layers, vertex_sources, face_sources = renderer._parallel_slab_geometry(axis)
    faces = renderer._turn_gpu_static_geometry()
    points = renderer.CUBE_LATTICE_OBJECT_POINTS_MM[vertices][None] @ matrices.transpose(0, 2, 1)
    polygons = np.einsum('aij,nkj->anki', matrices, faces[2])
    grid = np.asarray(list(product(range(7), repeat=3)))
    grid = grid[np.any(values[grid] != 0., axis=1)]
    def tensor(value, kind=dtype):
        return torch.as_tensor(np.array(value, copy=True), dtype=kind, device=device)
    return dict(values=values, value_tensor=tensor(values), grid_numpy=grid, grid=tensor(grid, torch.long),
        shape_rows=tensor([0, 1, 2, 3, 4, 5, 0], torch.long), matrices=tensor(matrices),
        points=tensor(points), endpoint_points=tensor(renderer.CUBE_LATTICE_OBJECT_POINTS_MM[vertices] @ endpoint.T),
        polygons=tensor(polygons), centers=tensor(polygons.mean(axis=2)),
        local_points=tensor(np.einsum('api,bij->bapj', points, matrices)),
        local_centers=tensor(np.einsum('api,bij->bapj', polygons.mean(axis=2), matrices)),
        normals=tensor(np.einsum('aij,nfj->anfi', matrices, renderer.CUBE_LATTICE_BOUNDARY_NORMALS[vertices])),
        face_normals=tensor(np.einsum('aij,nj->ani', matrices, faces[5])),
        face_mask=tensor(renderer.CUBE_LATTICE_BOUNDARY_FACE_MASK[vertices], torch.bool),
        classes=tensor(renderer.CUBE_LATTICE_CLASS_IDS[vertices], torch.long),
        layers=tensor(layers, torch.long), face_layers=tensor(face_layers, torch.long),
        vertex_sources=tensor(vertex_sources, torch.long), face_sources=tensor(face_sources, torch.long),
        lower=tensor([[-renderer.CUBE_HALF_SIZE_MM+(layer*renderer.CUBE_LAYER_SIZE_MM if a == axis else 0.)
                       for a in range(3)] for layer in range(3)]),
        upper=tensor([[(-renderer.CUBE_HALF_SIZE_MM+(layer+1)*renderer.CUBE_LAYER_SIZE_MM)
                       if a == axis else renderer.CUBE_HALF_SIZE_MM for a in range(3)] for layer in range(3)]))


def slab_entry(camera, directions, matrices, lower, upper):
    origin = torch.einsum('i,bij->bj', camera, matrices)[:, None]
    rays = torch.einsum('bni,bij->bnj', directions, matrices)
    return local_slab_entry(origin, rays, lower, upper)


def local_slab_entry(origin, rays, lower, upper):
    parallel = rays.abs() < 1e-12
    inside = (origin >= lower-1e-7) & (origin <= upper+1e-7)
    safe = torch.where(parallel, torch.ones_like(rays), rays)
    first, last = (lower-origin)/safe, (upper-origin)/safe
    entry = torch.clamp(torch.where(parallel, -torch.inf, torch.minimum(first, last)).amax(dim=-1), min=0.)
    exit = torch.where(parallel, torch.inf, torch.maximum(first, last)).amin(dim=-1)
    return torch.where((entry <= exit+1e-7) & torch.all(~parallel | inside, dim=-1), entry, torch.inf)


def project(objects, rotation, translation, camera_matrix, distortion):
    xyz = objects @ rotation.T+translation
    xy = xyz[..., :2]/xyz[..., 2:]
    if distortion is not None:
        x, y = xy.unbind(-1)
        r2 = x*x+y*y
        r4, r6 = r2*r2, r2*r2*r2
        k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4 = distortion.unbind()
        radial = (1+k1*r2+k2*r4+k3*r6)/(1+k4*r2+k5*r4+k6*r6)
        xy = torch.stack((x*radial+2*p1*x*y+p2*(r2+2*x*x)+s1*r2+s2*r4,
                          y*radial+p1*(r2+2*y*y)+2*p2*x*y+s3*r2+s4*r4), dim=-1)
    return xy*torch.stack((camera_matrix[0, 0], camera_matrix[1, 1]))+camera_matrix[:2, 2]


def percentile(values, count, fraction):
    position = (torch.clamp(count, min=1)-1).to(values.dtype)*fraction
    low, high = position.floor().long(), position.ceil().long()
    a, b = values.gather(1, low[:, None])[:, 0], values.gather(1, high[:, None])[:, 0]
    return torch.where(low == high, a, a+(b-a)*(position-low))


@lru_cache(maxsize=4)
def stacked_static_tensors(renderer, axes, device, dtype):
    models = [static_tensors(renderer, axis, device, dtype) for axis in axes]
    return {key: torch.stack([model[key] for model in models]) for key, value in models[0].items()
            if isinstance(value, torch.Tensor)}


@torch.no_grad()
def parallel_turn_sweeps(renderer, candidate, limit, *, axes=(0, 1, 2), device='mps', middle_frame=False, dtype=None):
    """Evaluate all requested axes together, with one result transfer to CPU."""
    device = str(device)
    axes = tuple(axes)
    if device.startswith('mps') and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable; run the GPU benchmark outside the filesystem sandbox')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    dtype = dtype or (torch.float32 if device.startswith('mps') else torch.float64)
    observed_np = np.asarray(candidate.assignment_image_points, float).reshape(-1, 2)
    stickers_np = np.asarray(candidate.sticker_mask_centers, float).reshape(-1, 2)
    scale = renderer.projected_cube_diameter(candidate)
    if len(observed_np) < 3 or not len(stickers_np) or not np.isfinite(scale) or scale <= 0.:
        return [([], np.empty((0, len(stickers_np)), np.uint8), np.empty((0, len(stickers_np)), bool)) for axis in axes]
    def tensor(value, kind=dtype):
        return torch.as_tensor(np.asarray(value), dtype=kind, device=device)
    rotation_np = cv2.Rodrigues(candidate.rvec)[0]
    rotation, translation = tensor(rotation_np), tensor(np.asarray(candidate.tvec).reshape(3))
    camera = -rotation.T @ translation
    camera_matrix = tensor(candidate.camera_matrix)
    coeffs = np.asarray(candidate.dist_coeffs).ravel()
    if len(coeffs) > 12 and np.any(coeffs[12:]):
        raise ValueError('GPU sweep does not support tilted-sensor distortion')
    distortion = tensor(np.pad(coeffs[:12], (0, max(0, 12-len(coeffs))))) if np.any(coeffs) else None
    observed, stickers = tensor(observed_np), tensor(stickers_np)
    observed_classes = (tensor(candidate.assignment_observed_class_ids, torch.long).reshape(-1)
        if renderer.PNP_ASSIGNMENT_MODE == 'class-aware' and candidate.assignment_observed_class_ids is not None else None)
    uncertainty_np = np.asarray(candidate.sticker_mask_center_uncertainties, float).reshape(-1)
    if len(uncertainty_np) != len(stickers_np):
        uncertainty_np = np.ones(len(stickers_np))
    uncertainty_np = np.where(np.isfinite(uncertainty_np), np.maximum(uncertainty_np, 1.), np.inf)
    uncertainty, weight = tensor(uncertainty_np), tensor(1./uncertainty_np**2)
    normalized_rays = cv2.undistortPoints(stickers_np, candidate.camera_matrix, candidate.dist_coeffs).reshape(-1, 2)
    rays = tensor(np.column_stack((normalized_rays, np.ones(len(normalized_rays))))) @ rotation
    inputs = (rotation, translation, camera, camera_matrix, distortion, observed, stickers,
              observed_classes, uncertainty, weight, rays, limit, scale)
    models = [static_tensors(renderer, axis, device, dtype) for axis in axes]
    diameter_rows = []
    for axis, data in zip(axes, models, strict=True):
        diameters = np.full(len(data['grid_numpy']), scale)
        if middle_frame:
            middle_values = data['values'][data['grid_numpy'][:, 1]]
            for value in np.unique(middle_values):
                rvec = cv2.Rodrigues(rotation_np @ cv2.Rodrigues(np.eye(3)[axis]*np.deg2rad(value))[0])[0]
                diameters[middle_values == value] = renderer.projected_cube_diameter(candidate, rvec=rvec)
        diameter_rows.append(diameters)
    diameters = tensor(np.asarray(diameter_rows))
    if len(axes) == 1:
        exported = _tensor_sweep(renderer, models[0], inputs, diameters[0], device, dtype)[None]
    else:
        exported = torch.vmap(lambda data, diameter: _tensor_sweep(renderer, data, inputs, diameter, device, dtype))(
            stacked_static_tensors(renderer, axes, device, dtype), diameters)
    exported = exported.cpu().numpy()
    return [_export_options(renderer, axis, data, row, len(stickers_np))
            for axis, data, row in zip(axes, models, exported, strict=True)]


def parallel_turn_sweep(renderer, candidate, limit, axis, **kwargs):
    """Single-axis compatibility wrapper for port validation and callers."""
    return parallel_turn_sweeps(renderer, candidate, limit, axes=(axis,), **kwargs)[0]


def _tensor_sweep(renderer, data, inputs, diameters, device, dtype):
    grid, layers, face_layers = data['grid'], data['layers'], data['face_layers']
    row = torch.arange(len(grid), device=device)[:, None]
    rotation, translation, camera, camera_matrix, distortion, observed, stickers, observed_classes, uncertainty, weight, rays, limit, scale = inputs
    tensor = lambda value: torch.as_tensor(value, dtype=dtype, device=device)
    def projection(objects):
        return project(objects, rotation, translation, camera_matrix, distortion)
    def aliases(table, sources):
        return torch.cat((table, table[0, sources][None]))
    def assemble(table, ids):
        return table[grid[:, ids], torch.arange(len(ids), device=device)]
    def visibility(objects, local_objects, ids, sources):
        count, points = objects.shape[:2]
        # Transform each endpoint before subtracting the camera. Rotating the
        # long camera-to-point vector loses precision for grazing self-rays.
        # Immutable local endpoints are prepared in float64 before upload.
        origins = torch.einsum('i,bij->bj', camera, data['matrices']).repeat_interleave(count, dim=0)[:, None]
        rays = local_objects.reshape(count*count, points, 3)-origins
        visible = torch.ones((len(grid), points), dtype=torch.bool, device=device)
        for layer in range(3):
            entries = local_slab_entry(origins, rays, data['lower'][layer], data['upper'][layer])
            clear = entries.reshape(count, count, points) >= 1.-1e-5
            clear = torch.cat((clear, clear[:, 0, sources][:, None]), dim=1)
            visible = visible & clear[data['shape_rows'][grid[:, layer, None]], grid[:, ids], torch.arange(points, device=device)]
        return visible
    points = data['points']
    projected = projection(points)
    point_distances = aliases(torch.linalg.vector_norm(projected[:, :, None]-observed[None, None], dim=-1), data['vertex_sources'])
    point_distances = torch.cat((point_distances[:-1], torch.linalg.vector_norm(projection(data['endpoint_points'])[:, None]-observed[None], dim=-1)[None]))
    distances = assemble(point_distances, layers).transpose(1, 2)
    front = (data['face_mask'][None] & ((data['normals']*(camera-points)[:, :, None]).sum(dim=-1) > 0)).any(dim=-1)
    visible = visibility(points, data['local_points'], layers, data['vertex_sources']) & assemble(aliases(front, data['vertex_sources']), layers)
    distances = torch.where(visible[:, None], distances, torch.inf)
    if observed_classes is not None:
        mismatch = (observed_classes[:, None] >= 0) & (data['classes'][None] >= 0) & (observed_classes[:, None] != data['classes'][None])
        distances = torch.where(mismatch[None], torch.inf, distances)
    distances, columns = distances.min(dim=-1)
    assigned_layers = layers[columns]
    values = data['value_tensor']
    active = values[grid] != 0.
    face_polygons = projection(data['polygons'])
    projected_centers = face_polygons.mean(dim=2)
    front_faces = (data['face_normals']*(camera-data['centers'])).sum(dim=-1) > 0
    visible_centers = assemble(aliases(front_faces, data['face_sources']), face_layers) & visibility(data['centers'], data['local_centers'], face_layers, data['face_sources'])
    center_distances = assemble(aliases(torch.linalg.vector_norm(projected_centers[:, :, None]-stickers[None, None], dim=-1), data['face_sources']), face_layers).transpose(1, 2)
    center_distances = torch.where(visible_centers[:, None], center_distances, torch.inf)
    error_limits = diameters*(limit/scale)
    finite = torch.isfinite(distances)
    count = finite.sum(dim=1)
    normalized = distances/diameters[:, None]
    sorted_keys = torch.sort(torch.where(finite, normalized, torch.inf), dim=1).values
    key_mean = torch.where(count > 0, torch.where(finite, normalized, 0.).sum(dim=1)/count.clamp(min=1), torch.inf)
    p90 = percentile(sorted_keys, count, .9)
    key_robust = percentile(sorted_keys, count, .5)+.5*p90
    key_inliers = (finite & (distances <= error_limits[:, None])).sum(dim=1)
    key_layer_inliers = (finite & active[row, assigned_layers] & (distances <= error_limits[:, None])).sum(dim=1)
    nearest_distances, nearest_faces = center_distances.min(dim=2)
    gate = limit/scale
    tolerance = torch.minimum(.02+torch.minimum(uncertainty[None]/diameters[:, None], tensor(.01)), tensor(gate-1e-9))
    fraction = ((nearest_distances/diameters[:, None]-tolerance)/(gate-tolerance).clamp(min=1e-12)).clamp(0., 1.)
    penalties = fraction*fraction*(3.-2.*fraction)
    usable = torch.isfinite(nearest_distances) & (weight[None] > 0.)
    safe_weights = torch.where(usable, weight[None].expand_as(nearest_distances), 0.)
    total_weight = safe_weights.sum(dim=1)
    sticker_mean = torch.where(total_weight > 0., renderer.PNP_TURN_CENTER_PENALTY_SCALE*(penalties*safe_weights).sum(dim=1)/total_weight.clamp(min=1e-12), torch.inf)
    sticker_normalized = torch.where(usable, nearest_distances/diameters[:, None], torch.inf)
    sorted_stickers, order = sticker_normalized.sort(dim=1)
    cumulative = safe_weights.gather(1, order).cumsum(dim=1)
    median_indices = (cumulative >= .5*total_weight[:, None]).to(torch.int32).argmax(dim=1)
    p90_indices = (cumulative >= .9*total_weight[:, None]).to(torch.int32).argmax(dim=1)
    sticker_robust = torch.where(total_weight > 0., sorted_stickers.gather(1, median_indices[:, None])[:, 0]+.5*sorted_stickers.gather(1, p90_indices[:, None])[:, 0], torch.inf)
    sticker_matches = (usable & (nearest_distances < error_limits[:, None])).sum(dim=1)
    layer_distances = torch.where(active[:, None, face_layers], center_distances, torch.inf).min(dim=2).values
    sticker_layer_matches = (torch.isfinite(layer_distances) & (layer_distances < error_limits[:, None])).sum(dim=1)
    support = []
    for layer in range(3):
        support.append(((assigned_layers == layer) & (distances <= limit)).sum(dim=1)+
            ((center_distances < limit) & (face_layers[None, None] == layer)).any(dim=2).sum(dim=1))
    metric_values = torch.stack([key_mean, key_robust, p90, key_inliers, key_layer_inliers,
                                sticker_mean, sticker_robust, sticker_matches, sticker_layer_matches, *support], dim=1)
    # Exact same sticker containment/occlusion assignment contract as CPU.
    depths = []
    for layer in range(3):
        entry = slab_entry(camera, rays[None].expand(6, -1, -1), data['matrices'], data['lower'][layer], data['upper'][layer])
        depths.append(entry[data['shape_rows'][grid[:, layer]]])
    depths = torch.stack(depths, dim=-1)
    exposed = torch.isfinite(depths) & (depths <= depths.min(dim=-1, keepdim=True).values+1e-6)
    assignments = renderer._torch_sticker_assignment_distances(torch, stickers, projected_centers, face_polygons, front_faces)
    assignments = assemble(aliases(assignments.transpose(1, 2), data['face_sources']), face_layers).transpose(1, 2)
    assignments = torch.where(exposed[:, :, face_layers], assignments, torch.inf)
    minimum, nearest = assignments.min(dim=2)
    # One transfer after all device-side scoring/assignment work completes.
    return torch.cat((metric_values, nearest.to(dtype), torch.isfinite(minimum).to(dtype)), dim=1)


def _export_options(renderer, axis, data, exported, sticker_count):
    metrics = exported[:, :12]
    nearest_np = exported[:, 12:12+sticker_count].astype(np.uint8)
    valid_np = exported[:, 12+sticker_count:].astype(bool)
    options = []
    for index, residuals in enumerate(data['values'][data['grid_numpy']]):
        key_mean, key_robust, p90, key_inliers, key_layer_inliers, paint_mean, paint_robust, paint_matches, paint_layer_matches, *layer_support = metrics[index]
        rotations = renderer.parallel_layer_rotations(axis, residuals)
        options.append(dict(parallel_layers=True, layer_axis=axis, layer_side=1, layer_count=1,
            layer_rotations=rotations, label=renderer.layer_rotation_debug_label(rotations), is_baseline=not rotations,
            layer_support=tuple(int(v) for v in layer_support), keypoint_error=float(key_mean), sticker_error=float(paint_mean),
            combined_mean_error=float(key_mean+paint_mean), robust_error=float(key_robust)+renderer.PNP_TURN_MASK_CENTER_WEIGHT*float(paint_robust),
            inlier_count=int(key_inliers+paint_matches), p90_error=float(p90), layer_inlier_count=int(key_layer_inliers+paint_layer_matches),
            keypoint_inlier_count=int(key_inliers), keypoint_layer_inlier_count=int(key_layer_inliers), sticker_match_count=int(paint_matches)))
    return options, nearest_np, valid_np
