"""Articulation-aware middle-frame pose fitting and retained projection banks.

No cube moves or sticker states are inferred here. Each axis gets independent
pose/angle refinement before the geometry-only shortlist is selected.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
import math

import cv2
import numpy as np
from scipy.optimize import least_squares


@dataclass
class PoseFit:
    candidate: object
    axis: int
    angles: tuple
    option: dict
    nearest: np.ndarray | None
    valid: np.ndarray | None
    score: float
    proposal_source: str = 'angle_sweep'
    flat_fallback: bool = False


class ArticulatedProjection:
    """Class-aware projection residuals with an analytic pose/angle Jacobian."""

    def __init__(self, renderer, candidate, axis, *, cache_sticker_jacobians=False):
        self.m, self.base, self.axis = renderer, candidate, axis
        self.observed = np.asarray(candidate.assignment_image_points, float).reshape(-1, 2)
        self.classes = (np.full(len(self.observed), -1) if
            candidate.assignment_observed_class_ids is None else
            np.asarray(candidate.assignment_observed_class_ids, int))
        self.stickers = np.asarray(candidate.sticker_mask_centers, float).reshape(-1, 2)
        self.layers, self.vertices, self.face_layers, _, _ = renderer._parallel_slab_geometry(axis)
        self.points = renderer.CUBE_LATTICE_OBJECT_POINTS_MM[self.vertices]
        self.normals = renderer.CUBE_LATTICE_BOUNDARY_NORMALS[self.vertices]
        self.face_mask = renderer.CUBE_LATTICE_BOUNDARY_FACE_MASK[self.vertices]
        faces = renderer._turn_gpu_static_geometry()
        self.polygons, self.face_normals = faces[2], faces[5]
        pc = renderer.CUBE_LATTICE_CLASS_IDS[self.vertices]
        self.compatible = ((self.classes[:, None] < 0) | (pc[None] < 0)
                           | (self.classes[:, None] == pc[None]))
        self.scale = max(renderer.projected_cube_diameter(candidate) / 220., .3)
        self._last = None
        self._articulation_key = None
        self._articulation_geometry = None
        self.cache_sticker_jacobians = cache_sticker_jacobians

    def evaluate(self, parameters):
        x = np.asarray(parameters)
        if self._last is not None and np.array_equal(x, self._last[0]):
            return self._last[1:]
        rvec, tvec = x[:3].reshape(3, 1), x[3:6].reshape(3, 1)
        # Rigid-only optimizer steps leave the articulated object geometry
        # unchanged. This one-entry cache is local to this frame and axis.
        articulation_key = (x.dtype.str, x[6:].tobytes())
        if articulation_key != self._articulation_key:
            angles = (x[6], 0., x[7])
            matrices = np.asarray([cv2.Rodrigues(np.eye(3)[self.axis] * np.deg2rad(a))[0]
                                   for a in angles])
            objects = np.einsum('nij,nj->ni', matrices[self.layers], self.points)
            polygons = np.einsum('nij,nkj->nki', matrices[self.face_layers], self.polygons)
            normals = np.einsum('nij,nfj->nfi', matrices[self.layers], self.normals)
            face_centers = polygons.mean(axis=1)
            face_normals = np.einsum('nij,nj->ni', matrices[self.face_layers], self.face_normals)
            unit_axis = np.eye(3)[self.axis]
            point_derivative = np.cross(unit_axis, objects) * (np.pi / 180.)
            face_derivative = np.cross(unit_axis, polygons) * (np.pi / 180.)
            self._articulation_geometry = (matrices, objects, polygons, normals, face_centers,
                                           face_normals, point_derivative, face_derivative)
            self._articulation_key = articulation_key
        (matrices, objects, polygons, normals, face_centers, face_normals,
         point_derivative, face_derivative) = self._articulation_geometry
        rotation = cv2.Rodrigues(rvec)[0]
        camera = -rotation.T @ tvec.ravel()
        visible = np.any(self.face_mask & (np.sum(
            normals * (camera - objects)[:, None], axis=-1) > 0), axis=1)
        face_visible = np.sum(face_normals * (camera - face_centers), axis=1) > 0
        # Both sets use the same camera and slab blockers. Batch their rays
        # without changing any per-ray arithmetic or visibility thresholds.
        rays = np.concatenate((objects-camera, face_centers-camera))[None]
        point_count = len(objects)
        for layer in range(3):
            clear = self.m._parallel_slab_entry(camera, rays,
                matrices[layer:layer+1], self.axis, layer)[0] >= 1. - 1e-5
            visible &= clear[:point_count]
            face_visible &= clear[point_count:]

        points, point_jac = cv2.projectPoints(objects, rvec, tvec,
            self.base.camera_matrix, self.base.dist_coeffs)
        points = points.reshape(-1, 2)
        point_jac = point_jac[:, :6].reshape(-1, 2, 6)
        pixels, face_jac = cv2.projectPoints(polygons.reshape(-1, 3), rvec, tvec,
            self.base.camera_matrix, self.base.dist_coeffs)
        pixels = pixels.reshape(-1, 4, 2)
        centers = pixels.mean(axis=1)
        face_jac = face_jac[:, :6].reshape(-1, 4, 2, 6)
        distance = np.linalg.norm(self.observed[:, None] - points[None], axis=-1)
        distance[~(self.compatible & visible[None])] = np.inf
        columns = distance.argmin(axis=1)
        matched = np.isfinite(distance[np.arange(len(distance)), columns])
        sd = np.linalg.norm(self.stickers[:, None] - centers[None], axis=-1)
        sd[:, ~face_visible] = np.inf
        sticker_columns = sd.argmin(axis=1)
        sticker_matched = np.isfinite(sd[np.arange(len(sd)), sticker_columns])

        residual = np.vstack((points[columns]-self.observed,
                              centers[sticker_columns]-self.stickers))
        jacobian = np.zeros((len(residual), 2, 8))
        jacobian[:len(columns), :, :6] = point_jac[columns]
        jacobian[len(columns):, :, :6] = face_jac[sticker_columns].mean(axis=1)
        # d(pixel)/d(object) = d(pixel)/d(camera translation) @ camera rotation.
        point_angle = np.einsum('nij,jk,nk->ni', point_jac[:, :, 3:6],
                                rotation, point_derivative)
        face_angle = np.einsum('nfij,jk,nfk->nfi', face_jac[:, :, :, 3:6],
                               rotation, face_derivative).mean(axis=1)
        for parameter, layer in ((6, 0), (7, 2)):
            jacobian[:len(columns), :, parameter] = point_angle[columns] * (
                self.layers[columns] == layer)[:, None]
            jacobian[len(columns):, :, parameter] = face_angle[sticker_columns] * (
                self.face_layers[sticker_columns] == layer)[:, None]
        if self.cache_sticker_jacobians:
            derivatives = np.zeros((len(centers), 2, 8))
            derivatives[:, :, :6] = face_jac.mean(axis=1)
            for parameter, layer in ((6, 0), (7, 2)):
                derivatives[:, :, parameter] = face_angle*(self.face_layers == layer)[:, None]
            self.sticker_projection_cache = centers, derivatives
        usable = np.r_[matched, sticker_matched]
        # Optional correspondence access for the experimental causal PnP seed.
        # This uses only real current-frame landmarks, never inferred corners.
        self.landmark_correspondences = objects[columns][matched], self.observed[matched]
        residual[~usable] = 40. * self.scale
        jacobian[~usable] = 0.
        residual, jacobian = residual.ravel()/self.scale, jacobian.reshape(-1, 8)/self.scale
        self._last = x.copy(), residual, jacobian
        return residual, jacobian


def _refine(model, seed, *, fixed_angles=None, free_angle=None, prior=None, iterations=18,
            pose_radius=None, angle_radius=None):
    """Refine a swept geometry while preserving its inactive outer slabs."""
    seed = np.asarray(seed, float)
    origin = seed[:6]
    radius = np.asarray([.8, .8, .8, 40., 40., 100.] if pose_radius is None else pose_radius,
                        dtype=float)
    if radius.shape != (6,) or not np.isfinite(radius).all() or np.any(radius <= 0):
        raise ValueError('pose_radius must contain six finite positive values')
    lower = np.r_[origin-radius, -45., -45.]
    upper = np.r_[origin+radius, 45., 45.]
    if angle_radius is not None:
        if not np.isfinite(angle_radius) or angle_radius <= 0:
            raise ValueError('angle_radius must be finite and positive')
        lower[6:] = np.maximum(lower[6:], seed[6:]-angle_radius)
        upper[6:] = np.minimum(upper[6:], seed[6:]+angle_radius)
    lower[5] = max(lower[5], model.m.CUBE_SIZE_MM+1.)
    if prior is None:
        prior = origin
    prior = np.asarray(prior)
    sigmas = np.array([.6, .6, .6, 25., 25., 70.])
    free = (np.r_[np.arange(6), free_angle] if free_angle is not None else
            np.arange(8 if fixed_angles is None else 6))

    def unpack(z):
        x = seed.copy()
        if fixed_angles is not None:
            x[6:] = fixed_angles
        x[free] = z
        return x

    def residual(z):
        x = unpack(z)
        values, _ = model.evaluate(x)
        return np.r_[values, (x[:6]-prior)/sigmas, x[6:]/180.]

    def jacobian(z):
        x = unpack(z)
        _, jac = model.evaluate(x)
        regularizer = np.zeros((8, 8))
        regularizer[np.arange(6), np.arange(6)] = 1./sigmas
        regularizer[6, 6] = regularizer[7, 7] = 1./180.
        return np.vstack((jac, regularizer))[:, free]

    result = least_squares(residual, seed[free], jac=jacobian,
        bounds=(lower[free], upper[free]), loss='soft_l1', f_scale=3.,
        x_scale=np.array([.25, .25, .25, 8., 8., 25., 15., 15.])[free],
        max_nfev=iterations)
    return unpack(result.x)


def capped_projection_residuals(values, jacobian, landmark_count, expansion=1.):
    """Radial capped loss: distant detections have zero pull on the pose."""
    values = values.reshape(-1, 2)
    jacobian = jacobian.reshape(-1, 2, 8)
    radii = np.linalg.norm(values, axis=1)
    caps = np.r_[np.full(landmark_count, 8.), np.full(len(values)-landmark_count, 6.)] * expansion
    denominators = np.r_[np.full(landmark_count, 4.), np.full(len(values)-landmark_count, 5.)]
    clipped = np.minimum(radii, caps)
    ratio = clipped/caps
    polynomial = np.sqrt(1.-ratio**2+ratio**4/3.)
    residuals = clipped*polynomial/denominators
    derivative = np.where(radii < caps, (1.-ratio**2)**2/polynomial/denominators, 0.)
    radial_jacobian = np.einsum('ni,nij->nj', values/np.maximum(radii[:, None], 1e-12), jacobian)
    return residuals, radial_jacobian*derivative[:, None]


def _refine_depth_recovery(model, seed, target, iterations=18):
    """Recover camera depth at fixed slab angles for sparse outlier frames."""
    seed = np.asarray(seed, float)
    origin = seed[:6]
    prior = origin.copy()
    diameter = model.m.projected_cube_diameter(model.base)
    sigmas = np.array([.6, .6, .6, max(3., .025*diameter), max(3., .025*diameter), max(6., .012*target)])
    # Preserve the current frame's image position while depth follows a robust
    # neighborhood trend. Cartesian x/y would couple translation to depth.
    prior[3:5] = model.m._rigid_core_coordinates(model.base)[:2]
    prior[5] = target

    def core(x):
        uv, jac = cv2.projectPoints(np.zeros((1, 3)), x[:3], x[3:6],
                                   model.base.camera_matrix, model.base.dist_coeffs)
        result = np.zeros((6, 8))
        result[:3, :3] = np.eye(3)
        result[3:5, :6] = jac[:, :6]
        result[5, 5] = 1.
        return np.r_[x[:3], uv.ravel(), x[5]], result

    lower = np.r_[origin-np.array([.8, .8, .8, 40., 40., 100.]), -45., -45.]
    upper = np.r_[origin+np.array([.8, .8, .8, 40., 40., 100.]), 45., 45.]
    lower[5] = max(model.m.CUBE_SIZE_MM+1., target-4.*sigmas[5])
    upper[5] = target+4.*sigmas[5]
    free = np.arange(6)

    def unpack(z):
        x = seed.copy()
        x[free] = z
        return x

    def residual(z):
        x = unpack(z)
        values, jacobian = model.evaluate(x)
        values, _ = capped_projection_residuals(values, jacobian, len(model.observed), expansion)
        return np.r_[values, (core(x)[0]-prior)/sigmas, x[6:]/180.]

    def jacobian(z):
        x = unpack(z)
        values, jac = model.evaluate(x)
        _, jac = capped_projection_residuals(values, jac, len(model.observed), expansion)
        regularizer = np.zeros((8, 8))
        regularizer[:6] = core(x)[1]/sigmas[:, None]
        regularizer[6, 6] = regularizer[7, 7] = 1./180.
        return np.vstack((jac, regularizer))[:, free]

    z = np.clip(seed[free], lower[free]+1e-8, upper[free]-1e-8)
    # A modest capture range first lets genuine articulated poses recover from
    # a tilted rigid seed. Both stages cap distant background detections.
    for expansion in ((1.5, 1.) if iterations > 6 else (1.,)):
        result = least_squares(residual, z, jac=jacobian,
            bounds=(lower[free], upper[free]), loss='linear',
            x_scale=np.array([.25, .25, .25, 8., 8., 25., 15., 15.])[free],
            max_nfev=iterations)
        z = result.x
    return unpack(z)


def depth_trend_targets(renderer, cache, args, fps):
    """Reuse the rigid fitter's supported anchors and robust time-based depth trend."""
    fps = fps if np.isfinite(fps) and fps > 0 else renderer.PNP_TURN_LOOKAHEAD_REFERENCE_FPS
    times = np.asarray([record.frame_index/fps for record in cache])
    coordinates, supported = {}, []
    for index, record in enumerate(cache):
        pose = record.pose_candidate
        if pose is None or not np.isfinite(pose.tvec).all() or pose.tvec[2, 0] <= renderer.CUBE_SIZE_MM:
            continue
        coordinates[index] = renderer._rigid_core_coordinates(pose)
        obs = renderer._rigid_pose_observations(record, args)
        if obs is None:
            if pose.assignment_image_points is None:
                continue
            classes = (np.full(len(pose.assignment_image_points), -1) if pose.assignment_observed_class_ids is None
                       else np.asarray(pose.assignment_observed_class_ids))
            obs = renderer.RigidPoseObservations(np.asarray(pose.assignment_image_points),
                classes, np.asarray(pose.sticker_mask_centers).reshape(-1, 2),
                tuple(pose.sticker_color_names), max(renderer.projected_cube_diameter(pose)/220., .3))
        matches = renderer._match_rigid_pose(obs, pose.rvec, pose.tvec, pose)
        if renderer._rigid_anchor_supported(matches, obs.pixel_scale):
            supported.append(index)
    anchors = np.asarray(supported, dtype=int)
    targets = []
    for index in range(len(cache)):
        target = renderer._local_rigid_depth(index, anchors, times, coordinates) if len(anchors) >= 2 else None
        if target is None and index in coordinates:
            target = float(coordinates[index][2])
        targets.append(target)
    return targets


def reject_spatial_outliers(renderer, candidate, depth):
    """Reject rays outside the sphere enclosing every possible slab rotation.

    Rotation about a cube axis preserves distance from the middle-frame origin.
    This bound therefore admits genuine articulation on all three axes.
    """
    if candidate.assignment_image_points is None or not len(candidate.assignment_image_points):
        return candidate
    points = np.asarray(candidate.assignment_image_points)
    center = renderer._rigid_core_coordinates(candidate)
    center[2] = depth
    translation = renderer._rigid_translation(center, candidate).ravel()
    normalized = cv2.undistortPoints(points, candidate.camera_matrix, candidate.dist_coeffs).reshape(-1, 2)
    rays = np.column_stack((normalized, np.ones(len(points))))
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    distances = np.linalg.norm(translation-rays*(rays@translation)[:, None], axis=1)
    radius = float(np.linalg.norm(renderer.CUBE_CORNER_POINTS_MM, axis=1).max())
    scale = max(renderer.projected_cube_diameter(candidate)/220., .3)
    # Two landmark cutoffs plus three depth sigmas make the gate conservative
    # around motion blur and noisy middle-frame position estimates.
    margin = 16.*scale*depth/min(candidate.camera_matrix[0, 0], candidate.camera_matrix[1, 1])
    margin += radius*3.*max(6., .012*depth)/depth
    keep = distances <= radius+margin
    classes = candidate.assignment_observed_class_ids
    diagnostics = dict(depth_prior_mm=float(depth), depth_sigma_mm=max(6., .012*depth),
        landmark_input_count=len(points), spatial_outlier_count=int((~keep).sum()),
        spatial_outlier_points=points[~keep].tolist(), spatial_radius_mm=radius+margin)
    return replace(candidate, assignment_image_points=points[keep],
        assignment_observed_class_ids=None if classes is None else np.asarray(classes)[keep],
        joint_pose_diagnostics=diagnostics)


def _materialize(renderer, base, axis, parameters):
    x = np.asarray(parameters)
    angles = tuple(0. if abs(a) < .5 else float(a) for a in (x[6], 0., x[7]))
    pose = replace(base, rvec=x[:3].reshape(3, 1), tvec=x[3:6].reshape(3, 1),
                   layer_rotations=(), sticker_layer_rotations=None)
    limit = renderer.PNP_FINAL_ASSIGNMENT_ERROR_FRACTION * renderer.projected_cube_diameter(pose)
    options, nearest, valid = renderer.parallel_turn_frame_geometry(pose, limit, axis, angles=angles)
    if not options:
        return None
    option = options[0]
    score = float(renderer.balanced_geometry_cost(option['keypoint_error'], option['sticker_error']))
    if not math.isfinite(score):
        return None
    rotations = tuple(option['layer_rotations'])
    candidate = replace(pose, layer_rotations=rotations, sticker_layer_rotations=rotations,
        assignment_object_points=option['assignment_object_points'],
        assignment_projected_points=option['assignment_projected_points'],
        assignment_projected_class_ids=option['assignment_projected_class_ids'],
        assignment_fixed_mask=np.zeros(len(pose.assignment_image_points), bool),
        image_points=np.empty((0, 2)), object_points=np.empty((0, 3)),
        projected_sticker_points=np.empty((0, 2)), final_solve_image_points=None,
        turn_state_moves=(), turn_color_offset_degrees=0.,
        reprojection_error=option['keypoint_error'],
        overall_reprojection_error=option['keypoint_error'],
        turn_fit_search_rvec=pose.rvec.copy(), turn_fit_search_tvec=pose.tvec.copy())
    return PoseFit(candidate, axis, angles, option, nearest[0], valid[0], score)


def swept_geometries(renderer, base, camera, *, sweep_device='cpu'):
    """Evaluate the complete cached slab-angle grid at one camera pose.

    Move the middle angle into the rigid rotation and wrap the outer offsets
    modulo 90 degrees. This preserves the surface while expressing every
    proposal in the middle-slab frame used by refinement and move search.
    """
    reference = cv2.Rodrigues(base.rvec)[0]
    aligned = renderer._align_rigid_rotation(camera.rvec, reference)
    pose = replace(base, rvec=cv2.Rodrigues(aligned)[0], tvec=camera.tvec,
                   layer_rotations=(), sticker_layer_rotations=None)
    limit = renderer.PNP_FINAL_ASSIGNMENT_ERROR_FRACTION * renderer.projected_cube_diameter(pose)
    fits = []
    if sweep_device != 'cpu':
        from scripts.independent_keypoint_gpu_sweep import parallel_turn_sweeps
        gpu_sweeps = parallel_turn_sweeps(renderer, pose, limit, device=sweep_device, middle_frame=True)
    for axis in range(3):
        if sweep_device == 'cpu':
            options, _, _ = renderer.parallel_turn_frame_geometry(pose, limit, axis, middle_frame=True)
        else:
            options, _, _ = gpu_sweeps[axis]
        # The batched sweep omits the shared all-zero geometry.
        options += renderer.parallel_turn_frame_geometry(pose, limit, axis, angles=(0., 0., 0.))[0]
        middle_cameras = {}
        for option in options:
            score = float(renderer.balanced_geometry_cost(option['keypoint_error'], option['sticker_error']))
            if not math.isfinite(score):
                continue
            angles = np.asarray(renderer.parallel_layer_angles(option['layer_rotations']))
            middle_key = np.asarray(angles[1], dtype=np.float64).tobytes()
            if middle_key not in middle_cameras:
                middle = cv2.Rodrigues(np.eye(3)[axis]*np.deg2rad(angles[1]))[0]
                middle_cameras[middle_key] = cv2.Rodrigues(aligned @ middle)[0]
            relative = (angles-angles[1]+45.) % 90.-45.
            # Keep the two endpoint interpretations when their signed travel differs.
            relative[(relative == -45.) & (angles-angles[1] > 0)] = 45.
            rotations = renderer.parallel_layer_rotations(axis, relative)
            candidate = replace(pose, rvec=middle_cameras[middle_key].copy(),
                                layer_rotations=rotations)
            # These lightweight proposals need full assignments only if retained.
            fits.append(PoseFit(candidate, axis, tuple(relative), option, None, None, score))
    return fits


def diverse_shortlist(renderer, fits, size=5, tolerance=.01):
    """Keep five distinct camera/angle solutions and strict score-cutoff ties."""
    fits = [fit for fit in fits if fit is not None and math.isfinite(fit.score)]
    descriptions = {}
    symmetries = np.asarray(renderer.CUBE_SYMMETRY_MATRICES)

    def description(fit):
        key = id(fit)
        if key not in descriptions:
            descriptions[key] = (cv2.Rodrigues(fit.candidate.rvec)[0],
                                  renderer._rigid_core_coordinates(fit.candidate))
        return descriptions[key]

    def equivalent(first, second):
        r1, core1 = description(first)
        r2, core2 = description(second)
        if (np.linalg.norm(core1[:2]-core2[:2]) > 3.
                or abs(core1[2]-core2[2]) > max(3., .015*min(core1[2], core2[2]))):
            return False
        alternatives = r2 @ symmetries.transpose(0, 2, 1)
        index = np.argmax(np.einsum('nij,ij->n', alternatives, r1))
        distance = np.rad2deg(np.linalg.norm(cv2.Rodrigues(alternatives[index] @ r1.T)[0]))
        if distance > 4.:
            return False
        rotations = tuple(renderer.transform_layer_rotation_by_symmetry(r, symmetries[index])
                          for r in second.candidate.layer_rotations)
        angles = np.asarray(renderer.parallel_layer_angles(rotations))
        if rotations and first.candidate.layer_rotations and rotations[0].axis != first.axis:
            return False
        if not np.array_equal(np.asarray(first.angles) != 0., angles != 0.):
            return False
        return bool(np.max(np.abs((np.asarray(first.angles)-angles+45.) % 90.-45.)) <= 6.)

    ranked = []
    for fit in sorted(fits, key=lambda f: (f.score, f.axis, f.angles)):
        if len(ranked) >= size and fit.score >= ranked[size-1].score+tolerance:
            break
        if not any(equivalent(anchor, fit) for anchor in ranked):
            ranked.append(fit)
    if not ranked:
        return []
    cutoff = ranked[min(size, len(ranked))-1].score+tolerance
    return [fit for fit in ranked if fit.score < cutoff]


@lru_cache(maxsize=4)
def _sticker_symmetry_permutations(renderer):
    """Map source face IDs into each relabeled cube frame."""
    centers = np.asarray(renderer._turn_gpu_static_geometry()[2]).mean(axis=1)
    symmetries = np.asarray(renderer.CUBE_SYMMETRY_MATRICES)
    transformed = np.einsum('sij,nj->sni', symmetries, centers)
    distances = np.linalg.norm(transformed[:, :, None]-centers[None, None], axis=-1)
    return distances.argmin(axis=-1)


def sticker_assignment_signature(renderer, fit, reference):
    """One face ID (or -1 for unassigned) per observed sticker keypoint."""
    rotation = cv2.Rodrigues(fit.candidate.rvec)[0]
    symmetries = np.asarray(renderer.CUBE_SYMMETRY_MATRICES)
    alternatives = rotation @ symmetries.transpose(0, 2, 1)
    index = int(np.argmax(np.einsum('nij,ij->n', alternatives, reference)))
    nearest = np.asarray(fit.nearest, dtype=np.intp)
    valid = np.asarray(fit.valid, dtype=bool)
    if nearest.shape != valid.shape or nearest.ndim != 1:
        raise ValueError('Assignment selection requires materialized sticker assignments')
    return tuple(np.where(valid, _sticker_symmetry_permutations(renderer)[index, nearest], -1))


def assignment_shortlist(renderer, fits, size=5):
    """Best five assignment patterns, followed by a reserved zero-angle fit.

    The flat fallback is outside the uniqueness quota, even if its assignment
    pattern or pose is already represented. Never pad missing patterns with
    duplicates. Seed/sweep diversity still uses diverse_shortlist above.
    """
    ranked = sorted((replace(fit, flat_fallback=False) for fit in fits
                     if fit is not None and math.isfinite(fit.score)),
                    key=lambda fit: (fit.score, fit.axis, fit.angles))
    if not ranked:
        return []
    reference = cv2.Rodrigues(ranked[0].candidate.rvec)[0]
    selected, signatures = [], set()
    for fit in ranked:
        signature = sticker_assignment_signature(renderer, fit, reference)
        if signature not in signatures:
            signatures.add(signature)
            selected.append(fit)
            if len(selected) == size:
                break
    flat = next((fit for fit in ranked if not any(fit.angles)), None)
    if flat is not None:
        selected.append(replace(flat, flat_fallback=True))
    return selected


def final_assignment_bank(renderer, fits, size=5, *, diagnostics=None):
    """Try cheap weak-layer flattening only after temporal fitting is finished.

    Camera and supported angles are fixed. Existing proximity support <=1
    gates each nonzero outer angle. A variant must preserve valid sticker
    assignments or have non-worse balanced reprojection cost. No solver runs.
    The reserved whole-flat fallback never displaces regular candidates.
    """
    ordinary = [fit for fit in fits if not fit.flat_fallback][:size]
    fallback = next((fit for fit in fits if fit.flat_fallback), None)
    if not ordinary:
        if diagnostics is not None:
            diagnostics.update(attempted_materializations=0, qualified_variants=0,
                               selected_replacements=0)
        return list(fits)
    reference = cv2.Rodrigues(ordinary[0].candidate.rvec)[0]

    def cost(fit):
        return float(renderer.balanced_geometry_cost(
            fit.option['keypoint_error'], fit.option['sticker_error']))

    def order(fit):
        return (sum(angle != 0. for angle in fit.angles),
                sum(abs(angle) for angle in fit.angles), cost(fit), fit.axis, fit.angles)

    def same_assignments(first, second):
        return (np.array_equal(first.valid, second.valid)
                and np.array_equal(first.nearest[first.valid], second.nearest[second.valid]))

    preferred, attempts, qualified = [], 0, 0
    for original in ordinary:
        support = original.option.get('layer_support', (math.inf,)*3)
        weak = [layer for layer in (0, 2)
                if original.angles[layer] != 0. and support[layer] <= 1]
        selected = original
        for mask in range(1, 1 << len(weak)):
            angles = list(original.angles)
            for bit, layer in enumerate(weak):
                if mask & (1 << bit):
                    angles[layer] = 0.
            parameters = np.r_[original.candidate.rvec.ravel(), original.candidate.tvec.ravel(),
                               angles[0], angles[2]]
            attempts += 1
            variant = _materialize(renderer, original.candidate, original.axis, parameters)
            if variant is None or not (same_assignments(original, variant)
                                       or cost(variant) <= cost(original)+1e-9):
                continue
            qualified += 1
            variant.proposal_source = 'weak_layer_flat_output'
            if order(variant) < order(selected):
                selected = variant
        preferred.append(selected)

    selected, signatures = [], {}
    for fit in [*preferred, *ordinary]:
        signature = sticker_assignment_signature(renderer, fit, reference)
        if signature in signatures:
            index = signatures[signature]
            if order(fit) < order(selected[index]):
                selected[index] = fit
        elif len(selected) < size:
            signatures[signature] = len(selected)
            selected.append(fit)
    if fallback is not None:
        selected.append(fallback)
    if diagnostics is not None:
        def original_geometry(fit):
            return any(fit.axis == old.axis and fit.angles == old.angles
                       and np.array_equal(fit.candidate.rvec, old.candidate.rvec)
                       and np.array_equal(fit.candidate.tvec, old.candidate.tvec)
                       for old in ordinary)

        diagnostics.update(attempted_materializations=attempts, qualified_variants=qualified,
                           selected_replacements=sum(not fit.flat_fallback and not original_geometry(fit)
                                                     for fit in selected))
    return selected


def fit_frame(renderer, candidate, *, static=False, prior=None, neighbors=(), retained=(), sweep_device='cpu',
              refinement_device='cpu'):
    """Sweep/refine, retain five unique sticker assignments and a flat fallback.

    A temporal call supplies retained fits and sweeps only the new neighboring
    cameras. The original camera sweep and its refinement are not repeated.
    """
    if (candidate is None or candidate.assignment_image_points is None
            or len(candidate.assignment_image_points) < 3 or not candidate.sticker_mask_centers):
        return []
    base = replace(candidate, layer_rotations=(), sticker_layer_rotations=None)
    reference = cv2.Rodrigues(base.rvec)[0]
    prior_parameters = None
    if prior is not None:
        prior_parameters = np.r_[cv2.Rodrigues(renderer._align_rigid_rotation(
            prior.rvec, reference))[0].ravel(), prior.tvec.ravel()]
    if static:
        seed = np.r_[base.rvec.ravel(), base.tvec.ravel(), 0., 0.]
        fit = _materialize(renderer, base, 0, _refine(ArticulatedProjection(renderer, base, 0),
            seed, fixed_angles=(0., 0.), prior=prior_parameters))
        return [replace(fit, flat_fallback=True)] if fit is not None else []
    cameras = tuple(neighbors) if retained else (base, *neighbors)
    models = {axis: ArticulatedProjection(renderer, base, axis) for axis in range(3)}
    proposals, seeds = [], []
    for camera in cameras:
        swept = (swept_geometries(renderer, base, camera) if sweep_device == 'cpu' else
                 swept_geometries(renderer, base, camera, sweep_device=sweep_device))
        proposals.extend(swept)
        # Rank before expensive fitting, preserving each axis and active-slab
        # hypothesis. Pure single-slab fits keep the other outer angle fixed.
        for axis in range(3):
            for mask in ((False, False), (True, False), (False, True), (True, True)):
                group = [f for f in swept if f.axis == axis
                         and tuple(np.asarray(f.angles)[[0, 2]] != 0.) == mask]
                seeds.extend(diverse_shortlist(renderer, group, size=2)[:2])
    # Preserve the winning sweep measurements independently of optimizer endpoints.
    fits = list(retained)
    for fit in diverse_shortlist(renderer, proposals):
        seed = np.r_[fit.candidate.rvec.ravel(), fit.candidate.tvec.ravel(),
                     np.asarray(fit.angles)[[0, 2]]]
        measured = _materialize(renderer, base, fit.axis, seed)
        if measured is not None:
            fits.append(measured)
    batched_parameters = None
    if refinement_device != 'cpu' and seeds:
        from scripts.independent_keypoint_gpu_refine import refine_batch
        starts = [np.r_[fit.candidate.rvec.ravel(), fit.candidate.tvec.ravel(),
                        np.asarray(fit.angles)[[0, 2]]] for fit in seeds]
        batched_parameters = refine_batch(renderer, base, [fit.axis for fit in seeds], starts,
            device=refinement_device, prior=prior_parameters)
    for seed_index, fit in enumerate(seeds):
        seed = np.r_[fit.candidate.rvec.ravel(), fit.candidate.tvec.ravel(),
                     np.asarray(fit.angles)[[0, 2]]]
        active = np.flatnonzero(seed[6:] != 0.)+6
        constraints = (dict(fixed_angles=(0., 0.)) if len(active) == 0 else
                       dict(free_angle=int(active[0])) if len(active) == 1 else {})
        parameters = (_refine(models[fit.axis], seed, prior=prior_parameters, **constraints)
                      if batched_parameters is None else batched_parameters[seed_index])
        revised = _materialize(renderer, base, fit.axis, parameters)
        if revised is not None:
            fits.append(replace(revised, proposal_source='angle_sweep_refined'))
    return assignment_shortlist(renderer, fits)


def projection_option(renderer, fit):
    """Serializable, candidate-specific pose and projection cache for search."""
    result = renderer.turn_trial_log_payload(fit.option)
    # Keep numerical fields alongside the renderer's human-readable aliases so
    # final alternatives can also be consumed directly by the CPU scorer.
    for name in ('combined_mean_error', 'keypoint_error', 'sticker_error',
                 'robust_error', 'p90_error', 'sticker_match_count'):
        if name in fit.option:
            result[name] = fit.option[name]
    result.update(joint_pose_fit=True, pose_rvec=fit.candidate.rvec.tolist(),
        geometry_bank_policy='sticker_assignments_plus_flat', flat_fallback=fit.flat_fallback,
        pose_proposal_source=fit.proposal_source,
        pose_tvec=fit.candidate.tvec.tolist(),
        cached_sticker_indices=fit.nearest.tolist(), cached_sticker_valid=fit.valid.tolist(),
        assignment_projected_points=fit.option['assignment_projected_points'].tolist(),
        assignment_object_points=fit.option['assignment_object_points'].tolist(),
        assignment_projected_class_ids=fit.option['assignment_projected_class_ids'].tolist())
    return result


def orient_fits(renderer, fits, colors, tracker, frame, fps):
    result = []
    for fit in fits:
        oriented = renderer.match_orientation_to_colors(fit.candidate, colors, tracker,
            frame_index=frame, source_fps=fps)
        rotations = tuple(oriented.layer_rotations)
        axis = rotations[0].axis if rotations else 0
        parameters = np.r_[oriented.rvec.ravel(), oriented.tvec.ravel(),
                            np.asarray(renderer.parallel_layer_angles(rotations))[[0, 2]]]
        result.append(_materialize(renderer, oriented, axis, parameters))
        if result[-1] is not None:
            result[-1].proposal_source = fit.proposal_source
            result[-1].flat_fallback = fit.flat_fallback
            result[-1].candidate = replace(result[-1].candidate,
                pose_continuity_diagnostics=oriented.pose_continuity_diagnostics,
                orientation_axis_center_labels=oriented.orientation_axis_center_labels)
    return assignment_shortlist(renderer, result)


def depth_fallback_reason(candidate, target, spatial_outliers=0):
    """Recover sparse fits with implausible depth or confirmed background points."""
    if candidate is None or target is None:
        return None
    landmarks = len(candidate.assignment_image_points)
    stickers = len(candidate.sticker_mask_centers)
    sigma = max(6., .012*target)
    jump = abs(float(candidate.tvec[2, 0])-target)
    # Under roughly a face of sticker observations, articulation is weakly
    # constrained. Abundant observations retain the original fitter's freedom.
    if landmarks >= 12 or stickers >= 10 or (jump <= 3.*sigma and spatial_outliers < 2):
        return None
    return dict(trigger=('sparse_observations_and_depth_jump' if jump > 3.*sigma
                         else 'sparse_observations_and_spatial_outliers'),
        landmark_count=landmarks, sticker_count=stickers,
        baseline_depth_mm=float(candidate.tvec[2, 0]),
        depth_prior_mm=float(target), depth_sigma_mm=sigma, depth_jump_sigmas=jump/sigma)


def apply_depth_fallback(renderer, cache, originals, args, depth_targets):
    """Recover isolated weak fits without feeding changes into baseline fitting."""
    for record, original, target in zip(cache, originals, depth_targets, strict=True):
        baseline = record.pose_candidate
        if baseline is None or original is None or target is None or renderer._frame_forces_static_hold(
                record.frame_index, getattr(args, 'turning_start_frame', None)):
            continue
        source = replace(original, layer_rotations=(), sticker_layer_rotations=None,
            rvec=cv2.Rodrigues(renderer._align_rigid_rotation(
                original.rvec, cv2.Rodrigues(baseline.rvec)[0]))[0])
        obs = renderer._rigid_pose_observations(record, args)
        if obs is not None:
            source = replace(source, assignment_image_points=obs.landmarks,
                assignment_observed_class_ids=obs.classes, sticker_mask_centers=tuple(obs.stickers),
                sticker_color_names=obs.colors)
        source = reject_spatial_outliers(renderer, source, target)
        reason = depth_fallback_reason(baseline, target,
            source.joint_pose_diagnostics['spatial_outlier_count'])
        if reason is None:
            continue
        saved = baseline.joint_geometry_candidates
        if len(source.assignment_image_points) < 3:
            continue
        origin = np.r_[source.rvec.ravel(), source.tvec.ravel()]
        fits = []
        # Keep the source rigid proposal and every original articulated camera
        # unchanged. Additional proposals refit cameras at fixed slab angles.
        seeds = [(0, np.r_[origin, 0., 0.], 'source_rigid')]
        for option in saved:
            rotations = tuple(renderer.CubeLayerRotation(**r) for r in option['layer_rotations'])
            seeds.append((option['layer_axis'], np.r_[np.asarray(option['pose_rvec']).ravel(),
                np.asarray(option['pose_tvec']).ravel(),
                np.asarray(renderer.parallel_layer_angles(rotations))[[0, 2]]], 'baseline'))
        for axis, seed, provenance in seeds:
            model = ArticulatedProjection(renderer, source, axis)
            revised = _refine_depth_recovery(model, seed, target)
            for parameters, name in ((seed, provenance), (revised, provenance+'_depth_refit')):
                fit = _materialize(renderer, source, axis, parameters)
                if fit is None:
                    continue
                # Compare the whole union under one outlier-resistant cost.
                values, jacobian = model.evaluate(parameters)
                residual, _ = capped_projection_residuals(values, jacobian, len(model.observed))
                depth_error = (float(parameters[5])-target)/max(6., .012*target)
                uv = renderer._rigid_core_coordinates(fit.candidate)[:2]
                image_sigma = max(3., .025*renderer.projected_cube_diameter(source))
                center_error = (uv-renderer._rigid_core_coordinates(source)[:2])/image_sigma
                rotation_error = cv2.Rodrigues(cv2.Rodrigues(parameters[:3])[0]
                    @ cv2.Rodrigues(source.rvec)[0].T)[0].ravel()/.6
                # Sparse data must justify the extra slab-angle parameters.
                # This model-selection penalty only applies inside recovery.
                active_slabs = sum(abs(a) >= .5 for a in fit.angles)
                cost = .5*(residual@residual+depth_error**2+center_error@center_error
                            +rotation_error@rotation_error)+active_slabs
                fit.score = float(cost/max(len(residual), 1))
                fit.option.update(depth_fallback=True, recovery_source=name,
                    recovery_score=fit.score)
                fits.append(fit)
        retained = assignment_shortlist(renderer, fits)
        if not retained:
            continue
        options = tuple(projection_option(renderer, fit) for fit in retained)
        diagnostics = dict(source.joint_pose_diagnostics or {}, **reason,
            baseline_geometry_options=list(saved),
            recovery_scoring='capped_observations_plus_pose_priors_and_slab_parameter_cost')
        best = replace(retained[0].candidate, joint_pose_diagnostics=diagnostics,
            pose_continuity_diagnostics=baseline.pose_continuity_diagnostics,
            orientation_axis_center_labels=baseline.orientation_axis_center_labels,
            joint_geometry_candidates=options,
            turn_search_diagnostics=dict(status='joint_pose_geometry_depth_fallback', geometry=dict(
                joint_refined=True, scope='per_geometry_middle_pose', options=options)))
        record.pose_candidate = best
        record.pose_candidates = tuple(fit.candidate for fit in retained)


def process_pose_geometry(renderer, cache, args, colors, fps, progress=None):
    """Sweep/refine source cameras, add neighboring cameras, then orient."""
    banks = []
    previous = None
    originals = [record.pose_candidate for record in cache]
    depth_targets = depth_trend_targets(renderer, cache, args, fps)
    sweep_device = getattr(args, 'geometry_sweep_device', 'cpu')
    refinement_device = getattr(args, 'geometry_refinement_device', 'cpu')
    for index, record in enumerate(cache):
        if progress:
            progress('joint pose/geometry refinement', index, len(cache))
        candidate = originals[index]
        obs = renderer._rigid_pose_observations(record, args)
        if candidate is None:
            # A neighboring camera can seed current detections; its old
            # observations must never be treated as evidence in this frame.
            candidate = (previous or next((c for c in originals[index+1:] if c is not None), None)
                         if obs is not None else None)
        if candidate is not None:
            if obs is not None:
                candidate = replace(candidate, assignment_image_points=obs.landmarks,
                    assignment_observed_class_ids=obs.classes,
                    sticker_mask_centers=tuple(obs.stickers), sticker_color_names=obs.colors,
                    sticker_mask_center_uncertainties=tuple(np.ones(len(obs.stickers))))
        static = renderer._frame_forces_static_hold(record.frame_index, getattr(args, 'turning_start_frame', None))
        fits = fit_frame(renderer, candidate, static=static, prior=previous, sweep_device=sweep_device,
                         refinement_device=refinement_device)
        banks.append(fits)
        if fits:
            previous = fits[0].candidate
    if progress:
        progress('joint pose/geometry refinement', len(cache), len(cache))
    if getattr(args, 'pose_smoothing', True):
        # Sweep the two new neighboring cameras using an immutable snapshot.
        # Existing source-camera results remain eligible without refitting.
        selected = [fits[0].candidate if fits else None for fits in banks]
        for index, fits in enumerate(banks):
            if progress:
                progress('joint trajectory refit', index, len(cache))
            neighbors = [selected[i] for i in (index-1, index+1)
                         if 0 <= i < len(cache) and selected[i] is not None]
            if not fits or not neighbors:
                continue
            base = fits[0].candidate
            reference = cv2.Rodrigues(base.rvec)[0]
            rotations = [renderer._align_rigid_rotation(c.rvec, reference) for c in neighbors]
            mean_delta = np.mean([cv2.Rodrigues(r @ reference.T)[0].ravel() for r in rotations], axis=0)
            prior = np.r_[cv2.Rodrigues(cv2.Rodrigues(mean_delta)[0] @ reference)[0].ravel(),
                           np.mean([c.tvec.ravel() for c in neighbors], axis=0)]
            static = renderer._frame_forces_static_hold(cache[index].frame_index,
                getattr(args, 'turning_start_frame', None))
            prior_pose = replace(base, rvec=prior[:3].reshape(3, 1), tvec=prior[3:].reshape(3, 1))
            banks[index] = fit_frame(renderer, base, static=static, prior=prior_pose,
                                     neighbors=neighbors, retained=fits, sweep_device=sweep_device,
                                     refinement_device=refinement_device)
        if progress:
            progress('joint trajectory refit', len(cache), len(cache))
    tracker = renderer.CubeSymmetryTracker()
    for index, (record, fits) in enumerate(zip(cache, banks, strict=True)):
        if progress:
            progress('orientation matching', index, len(cache))
        fits = orient_fits(renderer, fits, colors, tracker, record.frame_index, fps)
        banks[index] = fits
        if not fits:
            record.pose_candidate = None
            record.pose_candidates = ()
            record.pose_ready = True
            continue
        options = tuple(projection_option(renderer, fit) for fit in fits)
        best = fits[0].candidate
        record.pose_candidate = replace(best, joint_geometry_candidates=options,
            turn_search_diagnostics=dict(status='joint_pose_geometry', geometry=dict(
                joint_refined=True, scope='per_geometry_middle_pose', options=options)),
            turn_state_moves=(), turn_candidate_diagnostics=(), turn_fit_evaluated=False)
        record.pose_candidates = tuple(fit.candidate for fit in fits)
        record.pose_ready = True
        tracker.observe(best.rvec, record.frame_index)
    if progress:
        progress('orientation matching', len(cache), len(cache))
    before_recovery = [record.pose_candidate for record in cache]
    if getattr(args, 'joint_depth_fallback', True):
        apply_depth_fallback(renderer, cache, originals, args, depth_targets)
    # Final alternatives never become initial, neighbor or orientation seeds.
    for index, record in enumerate(cache):
        if record.pose_candidate is None:
            continue
        fits = banks[index]
        if record.pose_candidate is not before_recovery[index]:
            # Recovery is uncommon. Materialize its returned candidates from
            # this frame's observations, without another fitting/solver pass.
            options = record.pose_candidate.joint_geometry_candidates
            if not any(any(angle != 0. and option.get('layer_support', (math.inf,)*3)[layer] <= 1
                           for layer, angle in enumerate(renderer.parallel_layer_angles(candidate.layer_rotations)))
                       for candidate, option in zip(record.pose_candidates, options, strict=True)):
                continue
            fits = []
            for candidate, option in zip(record.pose_candidates, options, strict=True):
                rotations = candidate.layer_rotations
                axis = rotations[0].axis if rotations else 0
                angles = renderer.parallel_layer_angles(rotations)
                fit = _materialize(renderer, candidate, axis, np.r_[
                    candidate.rvec.ravel(), candidate.tvec.ravel(), angles[0], angles[2]])
                if fit is not None:
                    fit.score = option.get('recovery_score', fit.score)
                    fit.flat_fallback = option.get('flat_fallback', False)
                    fit.option.update({name: option[name] for name in
                                       ('depth_fallback', 'recovery_source', 'recovery_score')
                                       if name in option})
                    fits.append(fit)
        final = final_assignment_bank(renderer, fits)
        if len(final) == len(fits) and all(a is b for a, b in zip(final, fits, strict=True)):
            continue
        options = tuple(projection_option(renderer, fit) for fit in final)
        record.pose_candidate = replace(final[0].candidate,
            joint_pose_diagnostics=record.pose_candidate.joint_pose_diagnostics,
            joint_geometry_candidates=options,
            turn_search_diagnostics=dict(status='joint_pose_geometry', geometry=dict(
                joint_refined=True, scope='per_geometry_middle_pose', options=options)),
            turn_state_moves=(), turn_candidate_diagnostics=(), turn_fit_evaluated=False)
        record.pose_candidates = tuple(fit.candidate for fit in final)

