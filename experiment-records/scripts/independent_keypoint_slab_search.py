"""Coarse-to-fine static slab poses, followed by separate articulation fitting.

A static pose is supported by one rigid slab, not necessarily the whole cube.
Previous layer angles never seed or restrict the new frame's angle search.
Only the complete articulated reprojection can mark an update as successful.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from time import perf_counter

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from scripts import independent_keypoint_pose_geometry as geometry
from scripts.independent_keypoint_physics_tracker import PhysicsPoseTracker, Branch, _left_jacobian
from scripts.independent_keypoint_motion_prediction import MotionState, rotation_vector


@dataclass
class StaticSlabPose:
    parameters: np.ndarray
    axis: int
    layer: int
    score: float
    landmark_support: int
    sticker_support: int
    parent: int
    spatial_support: bool = True


class SlabScorer:
    """Batch partial static scores, without residuals from other moving slabs.

    One-to-one (mutual nearest) class-aware landmark matches, sticker matches,
    and spatial support prevent one coincident point claiming a slab fit.
    Other slabs are not assumed to be static or used as occluders here.
    """
    def __init__(self, renderer, base):
        self.m, self.base = renderer, base
        self.points = renderer.CUBE_LATTICE_OBJECT_POINTS_MM
        self.classes = renderer.CUBE_LATTICE_CLASS_IDS
        self.normals = renderer.CUBE_LATTICE_BOUNDARY_NORMALS
        self.face_mask = renderer.CUBE_LATTICE_BOUNDARY_FACE_MASK
        faces = renderer._turn_gpu_static_geometry()
        self.centers, self.face_normals = faces[2].mean(axis=1), faces[5]
        self.observed = np.asarray(base.assignment_image_points).reshape(-1, 2)
        self.observed_classes = (np.full(len(self.observed), -1) if base.assignment_observed_class_ids is None
                                 else np.asarray(base.assignment_observed_class_ids))
        self.stickers = np.asarray(base.sticker_mask_centers).reshape(-1, 2)
        self.diameter = renderer.projected_cube_diameter(base)
        self.compatible = ((self.classes[:, None] == self.observed_classes[None]) |
                           (self.observed_classes[None] < 0))
        self.slabs = []
        for axis in range(3):
            layers, vertices, face_layers, _, _ = renderer._parallel_slab_geometry(axis)
            for layer in range(3):
                self.slabs.append((axis, layer, np.unique(vertices[layers == layer]),
                                   np.flatnonzero(face_layers == layer)))
        self.scored_poses = 0

    def _project(self, objects, rotations, translations):
        if not np.any(self.base.dist_coeffs):
            xyz = objects @ rotations.transpose(0, 2, 1)+translations[:, None]
            homogeneous = xyz @ self.base.camera_matrix.T
            return homogeneous[..., :2]/np.maximum(xyz[..., 2:], 1.)
        return np.asarray([cv2.projectPoints(objects, rotation_vector(r), t,
            self.base.camera_matrix, self.base.dist_coeffs)[0].reshape(-1, 2)
            for r, t in zip(rotations, translations, strict=True)])

    def score(self, parameters, parents, *, slab_indices=None):
        parameters = np.asarray(parameters).reshape(-1, 6)
        self.scored_poses += len(parameters)
        rotations = Rotation.from_rotvec(parameters[:, :3]).as_matrix()
        translations = parameters[:, 3:6]
        camera = -np.einsum('bji,bj->bi', rotations, translations)
        visible = (self.face_mask[None] & (np.sum(self.normals[None]*(
            camera[:, None, None]-self.points[None, :, None]), axis=-1) > 0)).any(axis=2)
        face_visible = np.sum(self.face_normals[None]*(camera[:, None]-self.centers[None]), axis=-1) > 0
        projected = self._project(self.points, rotations, translations)
        centers = self._project(self.centers, rotations, translations)
        distance = np.linalg.norm(projected[:, :, None]-self.observed[None, None], axis=-1)
        distance = np.where(visible[:, :, None] & self.compatible[None], distance, np.inf)
        sticker_distance = np.linalg.norm(centers[:, :, None]-self.stickers[None, None], axis=-1)
        sticker_distance = np.where(face_visible[:, :, None], sticker_distance, np.inf)
        all_scores, support_counts = [], []
        slab_indices = list(range(9)) if slab_indices is None else slab_indices
        for index in slab_indices:
            _, _, vertices, faces = self.slabs[index]
            d, s = distance[:, vertices], sticker_distance[:, faces]
            # Each detected object can support at most one model vertex/sticker.
            def mutual(table):
                columns = table.argmin(axis=2)
                rows = table.argmin(axis=1)
                errors = np.take_along_axis(table, columns[..., None], axis=2)[..., 0]
                is_mutual = np.take_along_axis(rows, columns, axis=1) == np.arange(table.shape[1])[None]
                return np.where(is_mutual, errors, np.inf), columns
            key_errors, key_columns = mutual(d)
            sticker_errors, _ = mutual(s)
            key_support = (key_errors <= .045*self.diameter).sum(axis=1)
            sticker_support = (sticker_errors <= .045*self.diameter).sum(axis=1)
            keys = np.sort(np.minimum(key_errors/self.diameter, .08), axis=1)[:, :6]
            paint = np.sort(np.minimum(sticker_errors/self.diameter, .08), axis=1)[:, :2]
            cost = .65*keys.mean(axis=1)+.35*paint.mean(axis=1)
            cost += .012*np.maximum(0, 4-key_support)+.012*np.maximum(0, 1-sticker_support)
            usable = key_errors <= .045*self.diameter
            support_points = self.observed[key_columns]
            mean = (support_points*usable[..., None]).sum(axis=1)/np.maximum(key_support[:, None], 1)
            centered = support_points-mean[:, None]
            covariance = np.einsum('bni,bnj,bn->bij', centered, centered, usable)/np.maximum(key_support[:, None, None], 1)
            spatial_support = np.linalg.eigvalsh(covariance)[:, 0] >= (.012*self.diameter)**2
            cost += .03*(~spatial_support)
            all_scores.append(cost)
            support_counts.append((key_support, sticker_support, spatial_support))
        scores = np.asarray(all_scores).T
        winners = scores.argmin(axis=1)
        return [StaticSlabPose(x.copy(), self.slabs[slab_indices[w]][0], self.slabs[slab_indices[w]][1],
                float(scores[i, w]), int(support_counts[w][0][i]), int(support_counts[w][1][i]), int(parent),
                bool(support_counts[w][2][i]))
                for i, (x, w, parent) in enumerate(zip(parameters, winners, parents, strict=True))]

    def refine(self, proposal, iterations=3):
        """Static-only ICP/PnP on the supported slab; never fit layer angles."""
        best = proposal
        _, _, vertices, faces = self.slabs[3*proposal.axis+proposal.layer]
        for _ in range(iterations):
            r, t = best.parameters[:3], best.parameters[3:6]
            projected = cv2.projectPoints(self.points[vertices], r, t, self.base.camera_matrix,
                                         self.base.dist_coeffs)[0].reshape(-1, 2)
            distance = np.linalg.norm(projected[:, None]-self.observed[None], axis=-1)
            distance[~self.compatible[vertices]] = np.inf
            rows, columns = [], []
            used = set()
            for row in np.argsort(distance.min(axis=1)):
                column = int(distance[row].argmin())
                if column not in used and distance[row, column] <= .06*self.diameter:
                    rows.append(row)
                    columns.append(column)
                    used.add(column)
            if len(rows) < 4:
                break
            objects, observed = self.points[vertices[rows]], self.observed[columns]
            # Degenerate collinear support cannot determine a static pose.
            if np.linalg.matrix_rank(objects-objects.mean(axis=0), tol=1e-4) < 2:
                break
            try:
                ok, revised_r, revised_t = cv2.solvePnP(np.asarray(objects, float), np.asarray(observed, float),
                    self.base.camera_matrix, self.base.dist_coeffs, r.reshape(3, 1).copy(),
                    t.reshape(3, 1).copy(), True, flags=cv2.SOLVEPNP_ITERATIVE)
            except cv2.error:
                break
            if not ok or np.linalg.norm(revised_t.ravel()-t) > 35. or revised_t[2, 0] <= self.m.CUBE_SIZE_MM+1.:
                break
            change = rotation_vector(cv2.Rodrigues(revised_r)[0] @ cv2.Rodrigues(r)[0].T)
            if np.linalg.norm(change) > np.deg2rad(30.):
                break
            revised = self.score([np.r_[revised_r.ravel(), revised_t.ravel()]], [proposal.parent],
                                 slab_indices=[3*best.axis+best.layer])[0]
            if revised.score >= best.score:
                break
            best = revised
        return best


def static_shortlist(renderer, proposals, count=8, separation_degrees=5.):
    selected = []
    symmetries = np.asarray(renderer.CUBE_SYMMETRY_MATRICES)
    for proposal in sorted(proposals, key=lambda p: p.score):
        r = cv2.Rodrigues(proposal.parameters[:3])[0]
        duplicate = False
        for anchor in selected:
            if np.linalg.norm(proposal.parameters[3:6]-anchor.parameters[3:6]) > 5.:
                continue
            reference = cv2.Rodrigues(anchor.parameters[:3])[0]
            alternatives = r @ symmetries
            trace = np.einsum('nij,ij->n', alternatives, reference).max()
            if np.arccos(np.clip((trace-1.)/2., -1., 1.)) < np.deg2rad(separation_degrees):
                duplicate = True
                break
        if not duplicate:
            selected.append(proposal)
            if len(selected) == count:
                break
    return selected


def static_hierarchy(renderer, scorer, origins, parents, *, width_degrees=45., beam=8):
    """Sparse wide orientation probes, finer local stencils, then static PnP."""
    parameters, ancestry = [], []
    distinct_origins = []
    for origin, parent in zip(origins, parents, strict=True):
        rotation = cv2.Rodrigues(origin[:3])[0]
        if any(np.linalg.norm(origin[3:6]-old[0][3:6]) < 3. and np.linalg.norm(rotation_vector(
            rotation @ cv2.Rodrigues(old[0][:3])[0].T)) < np.deg2rad(4.) for old in distinct_origins):
            continue
        distinct_origins.append((origin, parent))
    for origin, parent in distinct_origins:
        rotation = cv2.Rodrigues(origin[:3])[0]
        offsets = list(product((-width_degrees, 0., width_degrees), repeat=3))
        offsets += [tuple(sign*2*width_degrees*np.eye(3)[axis]) for axis in range(3) for sign in (-1, 1)]
        for offset in offsets:
            revised = cv2.Rodrigues(np.deg2rad(np.asarray(offset)))[0] @ rotation
            parameters.append(np.r_[rotation_vector(revised), origin[3:6]])
            ancestry.append(parent)
    candidates = scorer.score(parameters, ancestry)
    selected = static_shortlist(renderer, candidates, beam, separation_degrees=12.)
    for step, separation in ((12., 6.), (4., 3.)):
        parameters, ancestry = [], []
        for proposal in selected:
            rotation = cv2.Rodrigues(proposal.parameters[:3])[0]
            for offset in [np.zeros(3), *(sign*step*np.eye(3)[axis] for axis in range(3) for sign in (-1, 1))]:
                revised = cv2.Rodrigues(np.deg2rad(offset))[0] @ rotation
                parameters.append(np.r_[rotation_vector(revised), proposal.parameters[3:6]])
                ancestry.append(proposal.parent)
            for axis, delta in enumerate((3., 3., 10.)):
                for sign in (-1, 1):
                    revised = proposal.parameters.copy()
                    revised[3+axis] += sign*delta*step/12.
                    parameters.append(revised)
                    ancestry.append(proposal.parent)
        candidates += scorer.score(parameters, ancestry)
        selected = static_shortlist(renderer, candidates, beam, separation_degrees=separation)
    candidates = [scorer.refine(p) for p in selected]
    return static_shortlist(renderer, candidates, beam, separation_degrees=3.)


def articulation_candidates(renderer, base, proposal, *, all_axes=False):
    """Search other slabs independently of their previous-frame angles."""
    pose = replace(base, rvec=proposal.parameters[:3].reshape(3, 1),
                   tvec=proposal.parameters[3:6].reshape(3, 1))
    # The supported slab defines the static camera. Anchor it at zero and
    # search the other two freely, including a moving middle slab.
    if all_axes:
        # A static camera proposal does not determine the actual turn axis.
        # Any one of the three slabs may define that camera's zero angle.
        grid = [angles for angles in product(np.arange(-45., 46., 15.), repeat=3)
                if any(angle == 0. for angle in angles)]
    else:
        free = [layer for layer in range(3) if layer != proposal.layer]
        grid = []
        for angles in product(np.arange(-45., 46., 15.), repeat=2):
            row = np.zeros(3)
            row[free] = angles
            grid.append(row)
    limit = renderer.PNP_FINAL_ASSIGNMENT_ERROR_FRACTION*renderer.projected_cube_diameter(pose)
    options = []
    for axis in range(3) if all_axes else (proposal.axis,):
        choices, _, _ = renderer.parallel_turn_frame_geometry(pose, limit, axis, angle_grid=grid)
        choices += renderer.parallel_turn_frame_geometry(pose, limit, axis, angles=(0., 0., 0.))[0]
        options.extend(choices)
    return pose, sorted(options, key=lambda o: renderer.balanced_geometry_cost(o['keypoint_error'], o['sticker_error']))


def articulation_parameters(renderer, pose, axis, option):
    angles = np.asarray(renderer.parallel_layer_angles(option['layer_rotations']))
    rotation = cv2.Rodrigues(pose.rvec)[0] @ cv2.Rodrigues(np.eye(3)[axis]*np.deg2rad(angles[1]))[0]
    return np.r_[rotation_vector(rotation), pose.tvec.ravel(), angles[[0, 2]]-angles[1]]


def fit_articulation(renderer, base, proposal, *, iterations=3, prepared=None):
    pose, options = articulation_candidates(renderer, base, proposal) if prepared is None else prepared
    result = []
    for option in options[:2]:
        x = articulation_parameters(renderer, pose, proposal.axis, option)
        # Keep the supporting slab's rigid pose fixed. Middle angle is folded
        # into R, so moving the middle requires an equal opposite offset on
        # the anchored outer slab. Derivative accounts for this constraint.
        model = geometry.ArticulatedProjection(renderer, base, proposal.axis)
        for _ in range(iterations):
            residual, jacobian = model.evaluate(x)
            if proposal.layer == 1:
                transform = np.zeros((8, 2))
                transform[6, 0], transform[7, 1] = 1., 1.
            else:
                # Parameterize two outer offsets while fixing the anchor's
                # camera R @ axis_rotation(anchor_offset).
                transform = np.zeros((8, 2))
                transform[6, 0], transform[7, 1] = 1., 1.
                column = 0 if proposal.layer == 0 else 1
                transform[:3, column] = _left_jacobian(x[:3], inverse=True) @ (
                    -cv2.Rodrigues(x[:3])[0] @ np.eye(3)[proposal.axis]*np.pi/180.)
            jac = jacobian @ transform
            weight = 1./np.sqrt(1.+(residual/3.)**2)
            delta = np.linalg.solve(jac.T @ (jac*weight[:, None])+.02*np.eye(2), -jac.T @ (residual*weight))
            delta = np.clip(delta, -10., 10.)
            candidate = x+transform @ delta
            # Reconstruct the exact anchor constraint, including near pi.
            if proposal.layer != 1:
                anchor = candidate[6 if proposal.layer == 0 else 7]
                anchored_rotation = cv2.Rodrigues(pose.rvec)[0] @ cv2.Rodrigues(
                    -np.eye(3)[proposal.axis]*np.deg2rad(anchor))[0]
                candidate[:3] = rotation_vector(anchored_rotation)
            new_residual, _ = model.evaluate(candidate)
            if np.sum(np.sqrt(1.+(new_residual/3.)**2)) <= np.sum(np.sqrt(1.+(residual/3.)**2)):
                x = candidate
            else:
                break
        x[6:] = (x[6:]+45.) % 90.-45.
        fit = geometry._materialize(renderer, base, proposal.axis, x)
        if fit is not None:
            fit.proposal_source = 'static_slab_then_articulation'
            result.append(fit)
    return result


class SlabHierarchyTracker(PhysicsPoseTracker):
    def __init__(self, renderer, *, beam=8, iterations=3, damping=.25, all_axes=False):
        super().__init__(renderer, iterations=iterations)
        self.beam, self.damping = beam, damping
        self.all_axes = all_axes
        self.last_prediction_diagnostics = {}

    def initialize(self, fits, time, frame=None):
        super().initialize(fits, time, frame)
        for branch in self.branches:
            branch.motion = MotionState(branch.fit, time, damping=self.damping)

    def update(self, observations, time, frame=None):
        if not self.branches or time <= self.time:
            raise ValueError('Initialize first, then use increasing timestamps')
        started = perf_counter()
        origins = [b.motion.predict(time).center[:6] for b in self.branches]
        base = replace(observations, rvec=origins[0][:3].reshape(3, 1), tvec=origins[0][3:6].reshape(3, 1),
            layer_rotations=(), sticker_layer_rotations=None, assignment_object_points=None,
            assignment_projected_points=None, assignment_projected_class_ids=None, joint_geometry_candidates=())
        scorer = SlabScorer(self.m, base)
        # All cameras in the previous bank contribute; no layer angle is read.
        width = min(45., max(15., 1350.*(time-self.time)))
        static = static_hierarchy(self.m, scorer, origins, list(range(len(origins))),
                                  width_degrees=width, beam=self.beam)
        static_ms = 1000.*(perf_counter()-started)
        proposals, parents, rough, source, reservoir = [], {}, [], {}, []
        for slab in static:
            pose, options = articulation_candidates(self.m, base, slab, all_axes=self.all_axes)
            reservoir.extend((slab, pose, option) for option in options)
            selected_options = ([option for axis in range(3) for option in
                [o for o in options if o['layer_axis'] == axis][:2]] if self.all_axes else options[:4])
            for option in selected_options:
                axis = option['layer_axis']
                x = articulation_parameters(self.m, pose, axis, option)
                angles = (x[6], 0., x[7])
                candidate = replace(base, rvec=x[:3].reshape(3, 1), tvec=x[3:6].reshape(3, 1),
                    layer_rotations=self.m.parallel_layer_rotations(axis, angles))
                fit = geometry.PoseFit(candidate, axis, angles, option, None, None,
                    float(self.m.balanced_geometry_cost(option['keypoint_error'], option['sticker_error'])))
                rough.append(fit)
                source[id(fit)] = slab, pose, option
        # Rank complete articulated explanations before spending on continuous
        # refinement; a partial slab score alone never selects the final bank.
        chosen = geometry.diverse_shortlist(self.m, rough, size=self.beam)[:self.beam]
        if self.all_axes:
            for axis in range(3):
                winner = min((f for f in rough if f.axis == axis), key=lambda f:f.score)
                if not any(f is winner for f in chosen):
                    chosen.append(winner)
        for seed in chosen:
            slab, pose, option = source[id(seed)]
            anchors = (np.flatnonzero(np.isclose(self.m.parallel_layer_angles(option['layer_rotations']), 0.))
                       if self.all_axes else [slab.layer])
            for anchor in anchors:
                supported = replace(slab, axis=option['layer_axis'], layer=int(anchor))
                for fit in fit_articulation(self.m, base, supported, iterations=self.iterations, prepared=(pose, [option])):
                    proposals.append(fit)
                    parents[id(fit)] = slab.parent
        retained = geometry.diverse_shortlist(self.m, proposals)[:5]
        if len(retained) < 5:
            # Fixed-camera coarse geometries can preserve real alternative
            # configurations without claiming a good full reprojection.
            for slab, pose, option in sorted(reservoir, key=lambda row:
                    self.m.balanced_geometry_cost(row[2]['keypoint_error'], row[2]['sticker_error'])):
                axis = option['layer_axis']
                x = articulation_parameters(self.m, pose, axis, option)
                x[6:] = (x[6:]+45.) % 90.-45.
                fit = geometry._materialize(self.m, base, axis, x)
                if fit is not None:
                    proposals.append(fit)
                    parents[id(fit)] = slab.parent
                retained = geometry.diverse_shortlist(self.m, proposals)[:5]
                if len(retained) == 5:
                    break
        error = self.m.pose_reprojection_fit_metric(retained[0].candidate) if retained else float('inf')
        # Partial fit and whole-cube fit are separate requirements. A low
        # partial score alone never reports a successful articulated update.
        accepted = bool(retained and error <= .035 and retained[0].score <= .2)
        old, self.branches = self.branches, []
        for fit in retained:
            parent = old[parents[id(fit)]]
            motion = parent.motion.copy()
            reliable = self.m.pose_reprojection_fit_metric(fit.candidate) <= .035 and fit.score <= .2
            if reliable:
                motion.observe(fit, time, reliable=True)
            else:
                # Propagate an uncertain pose proposal so temporary tracking
                # failures do not freeze the camera at an old position. It
                # cannot train velocity or the trusted working-position anchor.
                trusted_history = motion.history
                motion.history = type(trusted_history)(trusted_history)
                motion.observe(fit, time, reliable=False)
                motion.history = trusted_history
            core = parent.core.copy()
            core.observe(time, fit.candidate.tvec)
            branch = Branch(fit, core)
            branch.motion = motion
            self.branches.append(branch)
        self.time = float(time)
        self.core = self.branches[0].core
        self.last_prediction_diagnostics = dict(static_ms=static_ms,
            articulation_ms=1000.*(perf_counter()-started)-static_ms,
            scored_static_poses=scorer.scored_poses, retained_static_poses=len(static),
            best_static_score=static[0].score if static else None, final_full_fit_accepted=accepted,
            full_reprojection_error=error, refinements=len(proposals),
            articulation_predicted=False, all_articulation_axes_searched=self.all_axes, static_width_degrees=width)
        return retained
