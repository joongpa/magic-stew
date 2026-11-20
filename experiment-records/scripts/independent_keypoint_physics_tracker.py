"""Experimental five-geometry tracker with weak, causal core mean reversion.

No detector confidence channels are inferred or changed. New-frame pose fields
are discarded; only its observations and camera calibration enter the tracker.
The final bank uses the existing geometry score and diversity policy.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from time import perf_counter

import cv2
import numpy as np

from scripts import independent_keypoint_pose_geometry as geometry
from scripts.independent_keypoint_color_tracks import (
    StickerColorTracker, TrackedStickerProjection, bind_color_tracks, color_pose_seed, color_similarity_seed,
    center_sticker_rows, center_orientation_seed, sticker_objects,
    center_profile_support,
)


def wrapped_angle_delta(current, previous):
    return (np.asarray(current)-np.asarray(previous)+45.) % 90.-45.


def _skew(vector):
    x, y, z = np.asarray(vector)
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def _left_jacobian(vector, inverse=False):
    """SO(3) exponential Jacobian, including its small-angle limit."""
    angle = np.linalg.norm(vector)
    cross = _skew(vector)
    if inverse:
        coefficient = (1./12.+angle**2/720. if angle < 1e-5 else
                       (1.-.5*angle/np.tan(.5*angle))/angle**2)
        return np.eye(3)-.5*cross+coefficient*(cross @ cross)
    a, b = ((.5-angle**2/24., 1./6.-angle**2/120.) if angle < 1e-5 else
            ((1.-np.cos(angle))/angle**2, (angle-np.sin(angle))/angle**3))
    return np.eye(3)+a*cross+b*(cross @ cross)


class QuarterTurnPrior:
    """Causal lattice of 24 preferred colored orientations, with free offsets.

    The nearest mode is a prior target, not a pose replacement. Real quarter
    turns retain different mode IDs; no candidate's color frame is relabeled.
    """

    def __init__(self, anchor, symmetries, time, *, sigma_degrees=25., weight=4.):
        if not np.isfinite(sigma_degrees) or sigma_degrees <= 0 or weight < 0:
            raise ValueError('Invalid quarter-turn prior parameters')
        self.anchor = np.asarray(anchor, float).reshape(3, 3).copy()
        self.symmetries = np.asarray(symmetries, float)
        self.time = float(time)
        self.start_time = float(time)
        self.scale = np.sqrt(weight)/np.deg2rad(sigma_degrees)
        self.count = 1
        self.history = deque([self.anchor.copy()], maxlen=30)

    @property
    def ready(self):
        return self.count >= 5 and self.time-self.start_time >= .12

    def copy(self):
        result = QuarterTurnPrior(self.anchor, self.symmetries, self.time)
        result.scale, result.count = self.scale, self.count
        result.start_time = self.start_time
        result.history = deque((rotation.copy() for rotation in self.history), maxlen=30)
        return result

    def nearest(self, rotation):
        wells = self.anchor @ self.symmetries
        index = int(np.argmax(np.einsum('nij,ij->n', wells, rotation)))
        return wells[index], index

    def residual_jacobian(self, rvec):
        if not self.ready:
            return np.zeros(3), np.zeros((3, 3))
        rotation = cv2.Rodrigues(np.asarray(rvec))[0]
        well, _ = self.nearest(rotation)
        error = cv2.Rodrigues(well.T @ rotation)[0].ravel()
        derivative = _left_jacobian(error, inverse=True) @ well.T @ _left_jacobian(rvec)
        return self.scale*error, self.scale*derivative

    def observe(self, rotation, time, *, reliable=True):
        if time <= self.time:
            raise ValueError('Orientation updates must have increasing timestamps')
        dt = time-self.time
        self.time = float(time)
        if not reliable:
            return
        self.history.append(np.asarray(rotation).copy())
        # Follow gradual changes of working posture. The lattice cannot be
        # pulled toward one poor frame or look ahead to a later solved pose.
        gain = min(.15, 1.-np.exp(-dt/.8))
        # Estimate from the accepted history, not just the latest pose.
        anchor = self.anchor
        for _ in range(2):
            alternatives = np.asarray(self.history)[:, None] @ self.symmetries.transpose(0, 2, 1)[None]
            indices = np.einsum('nsij,ij->ns', alternatives, anchor).argmax(axis=1)
            aligned_history = alternatives[np.arange(len(indices)), indices]
            u, _, vt = np.linalg.svd(aligned_history.sum(axis=0))
            anchor = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
        error = cv2.Rodrigues(anchor @ self.anchor.T)[0].ravel()
        self.anchor = cv2.Rodrigues(gain*error)[0] @ self.anchor
        self.count += 1


class CoreReturnModel:
    """Weak pull to a moving working position, estimated from past accepted poses.

    This is a bounded-position prior, not an assertion that hands obey a passive
    spring. Velocity extrapolation is off by default: it lost to stationary
    prediction on the held-out fitted trajectories.
    """

    def __init__(self, reversion=2., velocity_damping=0., window=.5):
        if reversion < 0 or not 0 <= velocity_damping <= 1 or window <= 0:
            raise ValueError('Invalid core dynamics parameters')
        self.reversion = reversion
        self.velocity_damping = velocity_damping
        self.window = window
        self.history = deque()

    def observe(self, time, translation):
        if self.history and time <= self.history[-1][0]:
            raise ValueError('Core observations must have increasing timestamps')
        position = np.asarray(translation, float).reshape(3)
        if not np.isfinite(position).all():
            raise ValueError('Core position must be finite')
        self.history.append((float(time), position.copy()))
        while len(self.history) > 2 and self.history[1][0] < time-self.window:
            self.history.popleft()

    def predict(self, time, translation=None):
        if not self.history or time <= self.history[-1][0]:
            raise ValueError('Prediction timestamp must follow the latest observation')
        last_time, last = self.history[-1]
        position = last if translation is None else np.asarray(translation, float).reshape(3)
        dt = time-last_time
        center = np.median([p for _, p in self.history], axis=0)
        velocity = np.zeros(3)
        if len(self.history) >= 2:
            t, p = self.history[-2]
            velocity = (last-p)/(last_time-t)
        return (position + self.velocity_damping*dt*velocity
                + (1.-np.exp(-self.reversion*dt))*(center-position))

    def copy(self):
        result = CoreReturnModel(self.reversion, self.velocity_damping, self.window)
        result.history = deque((time, position.copy()) for time, position in self.history)
        return result


@dataclass
class Branch:
    fit: object
    core: CoreReturnModel | None = None
    parent_rotation: np.ndarray | None = None
    parent_angles: np.ndarray | None = None
    parent_axis: int | None = None
    previous_dt: float | None = None
    quarter: QuarterTurnPrior | None = None
    center_tracks: dict | None = None


def fast_geometry_score(renderer, model, parameters):
    """Cheap proposal ranking; all admitted fits are rescored by the old scorer."""
    residual, _ = model.evaluate(parameters)
    distances = np.linalg.norm(residual.reshape(-1, 2), axis=1)*model.scale
    count = len(model.observed)
    diameter = renderer.projected_cube_diameter(model.base, rvec=parameters[:3], tvec=parameters[3:6])
    keypoint = distances[:count].mean()/diameter
    sticker = distances[count:]/diameter
    tolerance = min(.02+min(1./diameter, .01), .05-1e-9)
    fraction = np.clip((sticker-tolerance)/(.05-tolerance), 0., 1.)
    sticker_cost = renderer.PNP_TURN_CENTER_PENALTY_SCALE*np.mean(fraction**2*(3.-2.*fraction))
    return float(renderer.balanced_geometry_cost(keypoint, sticker_cost))


def local_solve(model, seed, predicted_core, *, iterations=3, quarter=None, envelope=None):
    """Small damped Gauss-Newton solve using the existing analytic Jacobian.

    No SciPy multi-start or trust-region setup runs per proposal. The old exact
    geometry scorer remains responsible for admission after this local solve.
    """
    x = np.asarray(seed, float).copy()
    scale = np.array([.2, .2, .2, 5., 5., 15., 20., 20.])
    prior = np.r_[seed[:3], predicted_core, seed[6:]]
    sigmas = np.array([.5, .5, .5, 5., 5., 18., 90., 90.])
    regularizer = scale/sigmas
    def prior_terms(parameters):
        if envelope is not None:
            return envelope.residual_jacobian(parameters)
        return (parameters-prior)/sigmas, np.diag(1./sigmas)
    for _ in range(iterations):
        residual, jacobian = model.evaluate(x)
        # soft_l1 influence, matching the existing local solver's robust scale.
        weight = 1./np.sqrt(1.+(residual/3.)**2)
        jac = jacobian*scale
        if envelope is None:
            normal = jac.T @ (jac*weight[:, None]) + np.diag(regularizer**2+.05)
            gradient = jac.T @ (residual*weight) + regularizer*(x-prior)/sigmas
        else:
            values, derivative = prior_terms(x)
            derivative = derivative*scale
            normal = jac.T @ (jac*weight[:, None]) + derivative.T @ derivative + .05*np.eye(8)
            gradient = jac.T @ (residual*weight) + derivative.T @ values
        if quarter is not None:
            values, derivative = quarter.residual_jacobian(x[:3])
            derivative = derivative*scale[:3]
            normal[:3, :3] += derivative.T @ derivative
            gradient[:3] += derivative.T @ values
        step = np.linalg.solve(normal, -gradient)
        step /= max(1., np.linalg.norm(step)/2.)
        proposal = x+step*scale
        proposal[5] = max(proposal[5], model.m.CUBE_SIZE_MM+1.)
        # Allow a seam crossing in the optimizer. Folding is deferred until
        # materialization so +44 -> +46 is not mistaken for an 88-degree jump.
        proposal[6:] = np.clip(proposal[6:], seed[6:]-90., seed[6:]+90.)
        if envelope is not None:
            proposal = envelope.constrain(proposal)
        next_residual, _ = model.evaluate(proposal)
        old_cost = np.sum(9.*(np.sqrt(1.+(residual/3.)**2)-1.)) + .5*np.sum(prior_terms(x)[0]**2)
        next_cost = np.sum(9.*(np.sqrt(1.+(next_residual/3.)**2)-1.)) + .5*np.sum(prior_terms(proposal)[0]**2)
        if quarter is not None:
            old_angular, _ = quarter.residual_jacobian(x[:3])
            new_angular, _ = quarter.residual_jacobian(proposal[:3])
            old_cost += .5*(old_angular @ old_angular)
            next_cost += .5*(new_angular @ new_angular)
        if next_cost <= old_cost:
            x = proposal
        else:
            x += .25*step*scale
            x[5] = max(x[5], model.m.CUBE_SIZE_MM+1.)
            if envelope is not None:
                x = envelope.constrain(x)
        if np.linalg.norm(step) < 1e-3:
            break
    return x


def pnp_seed(model, seed, predicted_core):
    """Recover rigid motion from predicted articulated correspondences cheaply."""
    model.evaluate(seed)
    objects, observed = model.landmark_correspondences
    if len(objects) < 6:
        return seed
    try:
        ok, rotation, translation = cv2.solvePnP(np.ascontiguousarray(objects),
            np.ascontiguousarray(observed), model.base.camera_matrix, model.base.dist_coeffs,
            seed[:3].reshape(3, 1).copy(), seed[3:6].reshape(3, 1).copy(),
            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    except cv2.error:
        return seed
    if (not ok or translation[2, 0] <= model.m.CUBE_SIZE_MM+1.
            or np.linalg.norm(translation.ravel()-predicted_core) > 40.):
        return seed
    return np.r_[rotation.ravel(), translation.ravel(), seed[6:]]


class PhysicsPoseTracker:
    """Retain five distinct geometries, with shared per-axis projection models."""

    def __init__(self, renderer, *, reversion=2., iterations=2, proposals_to_refine=3,
                 use_pnp=True, quarter_turn_prior=False, quarter_sigma_degrees=25.,
                 quarter_weight=4., color_tracking=False, color_fast_lane=True):
        self.m = renderer
        self.core = CoreReturnModel(reversion=reversion)
        self.iterations = iterations
        self.proposals_to_refine = proposals_to_refine
        self.use_pnp = use_pnp
        self.quarter_turn_prior = quarter_turn_prior
        self.quarter_sigma_degrees = quarter_sigma_degrees
        self.quarter_weight = quarter_weight
        self.color_tracking = color_tracking
        self.color_fast_lane = color_fast_lane
        self.color_tracker = StickerColorTracker() if color_tracking else None
        self.last_color_diagnostics = {}
        self.branches = []
        self.time = None

    def initialize(self, fits, time, frame=None):
        retained = geometry.diverse_shortlist(self.m, fits)[:5]
        if len(retained) < 5:
            raise ValueError('Bootstrap needs five distinct geometries')
        self.branches = []
        for fit in retained:
            core = CoreReturnModel(reversion=self.core.reversion)
            core.observe(time, fit.candidate.tvec)
            quarter = (QuarterTurnPrior(cv2.Rodrigues(fit.candidate.rvec)[0], self.m.CUBE_SYMMETRY_MATRICES,
                time, sigma_degrees=self.quarter_sigma_degrees, weight=self.quarter_weight)
                if self.quarter_turn_prior else None)
            self.branches.append(Branch(fit, core, quarter=quarter))
        self.time = float(time)
        self.core = self.branches[0].core
        if self.color_tracker is not None:
            if frame is None:
                raise ValueError('Color tracking needs the actual video frame')
            pose = retained[0].candidate
            self.color_tracker.initialize(frame, pose.sticker_mask_centers, pose.sticker_color_names)
            self.current_center_support = center_profile_support(pose, self.m.projected_cube_diameter(pose))
            for branch in self.branches:
                branch.center_tracks = self._update_center_tracks(branch.fit, {}, set())

    def _update_center_tracks(self, fit, previous, continued):
        retained = {identity: face for identity, face in previous.items() if identity in continued}
        if fit.score > .2 or self.m.pose_reprojection_fit_metric(fit.candidate) > .025:
            return retained
        for row, identity in enumerate(self.color_tracker.ids):
            if int(identity) in retained or not fit.valid[row] or not self.current_center_support[row]:
                continue
            face = int(fit.nearest[row])
            if len(center_sticker_rows(self.m, np.array([face]))):
                retained[int(identity)] = face
        return retained

    def update(self, observations, time, frame=None):
        if not self.branches or time <= self.time:
            raise ValueError('Initialize first, then use increasing timestamps')
        m = self.m
        dt = time-self.time
        best = self.branches[0].fit.candidate
        predicted_core = self.core.predict(time)
        # Calibration and new observations survive; the cached current pose does not.
        base = replace(observations, rvec=best.rvec.copy(), tvec=predicted_core.reshape(3, 1),
            layer_rotations=(), sticker_layer_rotations=None, assignment_object_points=None,
            assignment_projected_points=None, assignment_projected_class_ids=None,
            joint_geometry_candidates=())
        models = {axis: geometry.ArticulatedProjection(m, base, axis,
            cache_sticker_jacobians=self.color_tracking) for axis in range(3)}
        color_bindings, color_seeds, color_diagnostics = {}, {}, {}
        if self.color_tracker is not None:
            tracking_started = perf_counter()
            if frame is None:
                raise ValueError('Color tracking needs the actual video frame')
            matches = self.color_tracker.match(frame, observations.sticker_mask_centers,
                observations.sticker_color_names, m.projected_cube_diameter(best))
            self.current_center_support = center_profile_support(observations, m.projected_cube_diameter(best))
            self.last_color_matches = matches
            for parent, branch in enumerate(self.branches):
                faces, image_points, identities = bind_color_tracks(m, branch.fit, matches,
                    np.asarray(observations.sticker_mask_centers).reshape(-1, 2), branch.center_tracks,
                    return_track_ids=True)
                seed, diagnostics = color_pose_seed(m, branch.fit, faces, image_points,
                    branch.core.predict(time), 2700.*dt)
                center_rows = center_sticker_rows(m, faces)
                center_rows = np.asarray([row for row in center_rows
                    if int(identities[row]) in (branch.center_tracks or {})], int)
                if len(center_rows):
                    color_bindings[parent] = faces[center_rows], image_points[center_rows]
                diagnostics['center_tracks'] = len(center_rows)
                if seed is not None and diagnostics.get('strong'):
                    if len(center_rows) < 2:
                        diagnostics['strong'] = False
                    else:
                        projected = cv2.projectPoints(sticker_objects(m, branch.fit, faces[center_rows]),
                            seed[:3], seed[3:6], branch.fit.candidate.camera_matrix,
                            branch.fit.candidate.dist_coeffs)[0].reshape(-1, 2)
                        error = np.linalg.norm(projected-image_points[center_rows], axis=1)
                        diagnostics['strong'] = bool(np.max(error) <= max(2., .02*m.projected_cube_diameter(best)))
                centered = center_orientation_seed(m, branch.fit, faces[center_rows], image_points[center_rows],
                    branch.core.predict(time), 2700.*dt)
                if centered is not None and not diagnostics.get('strong'):
                    seed = centered
                color_diagnostics[parent] = diagnostics
                if seed is not None:
                    color_seeds[parent] = seed
                else:
                    current_pairs = np.asarray(observations.sticker_mask_centers).reshape(-1, 2)[matches.current]
                    weak = color_similarity_seed(m, branch.fit, faces, current_pairs, matches,
                                                 branch.core.predict(time), 2700.*dt)
                    if weak is not None:
                        color_seeds[parent] = weak
            self.last_color_diagnostics = dict(matches.diagnostics,
                pose_proposals=len(color_seeds),
                maximum_pnp_inliers=max((d['inliers'] for d in color_diagnostics.values()), default=0),
                strong_motion_proposals=sum(d.get('strong', False) for d in color_diagnostics.values()))
            self.last_color_diagnostics['maximum_center_tracks'] = max(
                (d['center_tracks'] for d in color_diagnostics.values()), default=0)
            self.last_color_diagnostics['tracking_and_pnp_ms'] = 1000.*(perf_counter()-tracking_started)
        else:
            self.last_color_diagnostics = {}
        fast_lane = bool(self.color_fast_lane and any(d.get('strong') for d in color_diagnostics.values()))
        seeds, parent_seeds = [], []
        for parent, branch in enumerate(self.branches):
            fit = branch.fit
            pose = fit.candidate
            rotation = cv2.Rodrigues(pose.rvec)[0]
            angles = np.asarray(fit.angles)[[0, 2]]
            core = branch.core.predict(time)
            stationary = fit.axis, np.r_[pose.rvec.ravel(), core, angles], parent
            seeds.append(stationary)
            if self.use_pnp:
                revised = fit.axis, pnp_seed(models[fit.axis], stationary[1], core), parent
                seeds.append(revised)
                parent_seeds.append(min((stationary, revised), key=lambda row:
                    fast_geometry_score(m, models[row[0]], row[1])))
            else:
                parent_seeds.append(stationary)
            if parent in color_seeds:
                colored = fit.axis, color_seeds[parent], parent
                seeds.append(colored)
                old_score = fast_geometry_score(m, models[fit.axis], parent_seeds[-1][1])
                color_score = fast_geometry_score(m, models[fit.axis], colored[1])
                if color_score <= old_score+.1:
                    parent_seeds[-1] = colored
            if branch.quarter is not None and branch.quarter.ready:
                well, _ = branch.quarter.nearest(rotation)
                toward_well = cv2.Rodrigues(well @ rotation.T)[0].ravel()
                # Both the continuous prediction and the preferred orientation
                # are proposals. No angle is snapped in the admitted output.
                for fraction in (.5, 1.):
                    revised = cv2.Rodrigues(fraction*toward_well)[0] @ rotation
                    seeds.append((fit.axis, np.r_[cv2.Rodrigues(revised)[0].ravel(), core, angles], parent))
            if branch.parent_rotation is not None:
                delta = cv2.Rodrigues(rotation @ branch.parent_rotation.T)[0].ravel()
                delta *= min(1., np.deg2rad(2700.*dt)/max(np.linalg.norm(delta), 1e-12))
                predicted_rotation = cv2.Rodrigues(.5*dt/branch.previous_dt*delta)[0] @ rotation
                predicted_angles = angles.copy()
                if branch.parent_axis == fit.axis:
                    travel = wrapped_angle_delta(angles, branch.parent_angles)
                    predicted_angles += .5*dt/branch.previous_dt*np.clip(travel, -2700.*dt, 2700.*dt)
                seeds.append((fit.axis, np.r_[cv2.Rodrigues(predicted_rotation)[0].ravel(),
                                               core, predicted_angles], parent))
        # Axis transitions and new turns remain possible even when every parent
        # happens to be flat. These probes share one predicted camera and core.
        origin = np.r_[best.rvec.ravel(), predicted_core]
        origin_parent = 0
        if fast_lane:
            parent = max((p for p in color_seeds if color_diagnostics[p].get('strong')),
                         key=lambda p: color_diagnostics[p]['inliers'])
            origin = color_seeds[parent][:6]
            origin_parent = parent
        for axis in range(3):
            angle_seeds = (((0., 0.), (-30., 0.), (30., 0.), (0., -30.), (0., 30.)) if fast_lane else
                           tuple((a, b) for a in (-30., 0., 30.) for b in (-30., 0., 30.)))
            for angles in angle_seeds:
                seeds.append((axis, np.r_[origin, angles], origin_parent))
            rotation = cv2.Rodrigues(best.rvec)[0]
            if self.quarter_turn_prior and self.branches[0].quarter.ready:
                well, _ = self.branches[0].quarter.nearest(rotation)
                for angles in ((0., 0.), (-30., 0.), (30., 0.), (0., -30.), (0., 30.)):
                    seeds.append((axis, np.r_[cv2.Rodrigues(well)[0].ravel(), predicted_core, angles], 0))
            for degrees in (() if fast_lane else (-30., -15., 15., 30.)):
                revised = rotation @ cv2.Rodrigues(np.eye(3)[axis]*np.deg2rad(degrees))[0]
                seeds.append((axis, np.r_[cv2.Rodrigues(revised)[0].ravel(), predicted_core,
                                             (0., 0.)], 0))
        ranked = sorted(seeds, key=lambda row: fast_geometry_score(m, models[row[0]], row[1]))
        # Always give each retained parent a chance to survive, independently of
        # the approximate probe ranking. Use at least five real distinct outputs.
        axis_winners = [next(row for row in ranked if row[0] == axis) for axis in range(3)]
        chosen = [*parent_seeds, *axis_winners[:self.proposals_to_refine],
                  *ranked[:max(0, self.proposals_to_refine-3)]]
        proposals, parents, seen = [], {}, set()

        def refine_proposal(axis, seed, parent, iterations, use_color=True):
            key = axis, tuple(np.round(seed, 5))
            if key in seen:
                return
            seen.add(key)
            model = models[axis]
            if use_color and parent in color_bindings:
                faces, tracked_points = color_bindings[parent]
                old_fit = self.branches[parent].fit
                if len(faces) >= 2 and (axis == old_fit.axis or np.max(np.abs(old_fit.angles)) < 5.):
                    model = TrackedStickerProjection(model, faces, tracked_points)
            parameters = local_solve(model, seed, self.branches[parent].core.predict(time),
                                      iterations=iterations,
                                      quarter=(self.branches[parent].quarter if self.quarter_weight > 0 else None))
            parameters[6:] = (parameters[6:]+45.) % 90.-45.
            fit = geometry._materialize(m, base, axis, parameters)
            if fit is not None:
                fit.proposal_source = 'quarter_turn_prior' if self.quarter_turn_prior else 'physics_prediction'
                proposals.append(fit)
                parents[id(fit)] = parent

        for axis, seed, parent in chosen:
            refine_proposal(axis, seed, parent, min(2, self.iterations) if fast_lane else self.iterations)
        retained = geometry.diverse_shortlist(m, proposals)[:5]
        poor_fast_fit = fast_lane and (not retained or retained[0].score > .2 or
            m.pose_reprojection_fit_metric(retained[0].candidate) > .025)
        self.last_color_diagnostics['geometry_fallback'] = bool(poor_fast_fit)
        if poor_fast_fit:
            # A color-supported shortcut is provisional. Restore ordinary
            # geometric proposals and fit without tracked-color constraints.
            fallback = []
            for axis in range(3):
                origin = np.r_[best.rvec.ravel(), predicted_core]
                for a in (-30., 0., 30.):
                    for b in (-30., 0., 30.):
                        fallback.append((axis, np.r_[origin, a, b], 0))
            fallback.sort(key=lambda row: fast_geometry_score(m, models[row[0]], row[1]))
            for axis in range(3):
                row = next(row for row in fallback if row[0] == axis)
                # Even a repeated seed receives the normal fitting budget.
                seen.discard((row[0], tuple(np.round(row[1], 5))))
                refine_proposal(*row, self.iterations, use_color=False)
            retained = geometry.diverse_shortlist(m, proposals)[:5]
        if len(retained) < 5:
            # Rescore diverse unrefined probes, never duplicate a fit to claim a
            # five-element bank. These evaluations are included in runtime.
            for axis, seed, parent in ranked:
                parameters = seed.copy()
                parameters[6:] = (parameters[6:]+45.) % 90.-45.
                fit = geometry._materialize(m, base, axis, parameters)
                if fit is None:
                    continue
                proposals.append(fit)
                parents[id(fit)] = parent
                retained = geometry.diverse_shortlist(m, proposals)[:5]
                if len(retained) >= 5:
                    break
        old = self.branches
        self.branches = []
        for fit in retained:
            parent_branch = old[parents[id(fit)]]
            parent = parent_branch.fit
            core = parent_branch.core.copy()
            core.observe(time, fit.candidate.tvec)
            quarter = parent_branch.quarter.copy() if parent_branch.quarter else None
            if quarter is not None:
                quarter.observe(cv2.Rodrigues(fit.candidate.rvec)[0], time,
                    reliable=m.pose_reprojection_fit_metric(fit.candidate) <= .025 and fit.score <= .2)
            center_tracks = None
            if self.color_tracker is not None:
                center_tracks = self._update_center_tracks(fit, parent_branch.center_tracks or {},
                    set(map(int, self.last_color_matches.track_ids)))
            self.branches.append(Branch(fit, core, cv2.Rodrigues(parent.candidate.rvec)[0],
                np.asarray(parent.angles)[[0, 2]], parent.axis, dt, quarter, center_tracks))
        self.time = float(time)
        if retained:
            self.core = self.branches[0].core
        self.last_color_diagnostics['fast_lane'] = fast_lane
        return retained
