"""Causal continuous-motion prediction and adaptive local five-pose search.

This experiment uses no discrete preferred orientations or sticker identities.
Motion history and uncertainty belong to each hypothesis's own ancestry.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import copy

import cv2
import numpy as np

from scripts import independent_keypoint_pose_geometry as geometry
from scripts.independent_keypoint_physics_tracker import (
    PhysicsPoseTracker, Branch, _left_jacobian, wrapped_angle_delta,
    fast_geometry_score, local_solve, pnp_seed,
)


def rotation_vector(matrix):
    return cv2.Rodrigues(matrix)[0].ravel()


@dataclass
class MotionEnvelope:
    center: np.ndarray
    sigma: np.ndarray
    radius: np.ndarray

    def residual_jacobian(self, parameters):
        error = np.asarray(parameters)-self.center
        rotation = cv2.Rodrigues(parameters[:3])[0]
        reference = cv2.Rodrigues(self.center[:3])[0]
        error[:3] = rotation_vector(rotation @ reference.T)
        error[6:] = wrapped_angle_delta(parameters[6:], self.center[6:])
        jacobian = np.eye(8)
        jacobian[:3, :3] = _left_jacobian(error[:3], inverse=True) @ _left_jacobian(parameters[:3])
        return error/self.sigma, jacobian/self.sigma[:, None]

    def constrain(self, parameters):
        result = np.asarray(parameters).copy()
        rotation = cv2.Rodrigues(result[:3])[0]
        reference = cv2.Rodrigues(self.center[:3])[0]
        delta = rotation_vector(rotation @ reference.T)
        delta *= min(1., self.radius[0]/max(np.linalg.norm(delta), 1e-12))
        result[:3] = rotation_vector(cv2.Rodrigues(delta)[0] @ reference)
        result[3:6] = np.clip(result[3:6], self.center[3:6]-self.radius[3:6],
                            self.center[3:6]+self.radius[3:6])
        result[6:] = self.center[6:]+np.clip(wrapped_angle_delta(result[6:], self.center[6:]),
                                            -self.radius[6:], self.radius[6:])
        return result


class MotionState:
    """Damped velocity on SO(3), near-stationary core, and innovation limits.

    Sigma is a search scale learned from fitted-pose innovations, not a
    calibrated probability or detector confidence. Poor fits widen it.
    """
    def __init__(self, fit, time, *, damping=.25):
        if not 0. <= damping <= 1.:
            raise ValueError('Motion damping must be between zero and one')
        self.damping = damping
        self.time = float(time)
        self.rotation = cv2.Rodrigues(fit.candidate.rvec)[0]
        self.position = fit.candidate.tvec.ravel().copy()
        self.angles = np.asarray(fit.angles)[[0, 2]].copy()
        self.axis = fit.axis
        self.angular_velocity = np.zeros(3)
        self.velocity = np.zeros(3)
        self.turn_velocity = np.zeros(2)
        self.history = deque([(self.time, self.position.copy())])
        self.sigma = np.r_[np.full(3, np.deg2rad(12.)), 3., 3., 10., 18., 18.]

    def copy(self):
        return copy.deepcopy(self)

    def predict(self, time):
        dt = time-self.time
        if dt <= 0:
            raise ValueError('Predict after the latest observation')
        # 90 degrees per 1/30 s is a speed limit, not an orientation mode.
        travel = self.damping*dt*self.angular_velocity
        travel *= min(1., np.deg2rad(2700.*dt)/max(np.linalg.norm(travel), 1e-12))
        rotation = cv2.Rodrigues(travel)[0] @ self.rotation
        anchor = np.median([p for _, p in self.history], axis=0)
        position = (self.position + .15*self.damping*dt*self.velocity
                    + (1.-np.exp(-2.*dt))*(anchor-self.position))
        angles = self.angles+self.damping*dt*self.turn_velocity
        center = np.r_[rotation_vector(rotation), position, angles]
        sigma = self.sigma.copy()*np.sqrt(max(dt/(1./30.), .5))
        sigma[:3] = np.clip(sigma[:3], np.deg2rad(4.), np.deg2rad(25.))
        sigma[3:6] = np.clip(sigma[3:6], [1.5, 1.5, 4.], [10., 10., 25.])
        sigma[6:] = np.clip(sigma[6:], 8., 35.)
        radius = 2.5*sigma
        radius[:3] = min(radius[0], np.deg2rad(2700.*dt+5.))
        radius[6:] = np.minimum(radius[6:], min(45., 2700.*dt+5.))
        return MotionEnvelope(center, sigma, radius)

    def observe(self, fit, time, *, reliable):
        envelope = self.predict(time)
        dt = time-self.time
        rotation = cv2.Rodrigues(fit.candidate.rvec)[0]
        position = fit.candidate.tvec.ravel()
        angles = np.asarray(fit.angles)[[0, 2]]
        innovation = envelope.residual_jacobian(np.r_[rotation_vector(rotation), position, angles])[0]*envelope.sigma
        innovation[:3] = np.linalg.norm(innovation[:3])/np.sqrt(3.)
        if fit.axis != self.axis:
            innovation[6:] = 25.
        # Do not let a single apparently perfect local fit collapse the region.
        floor = np.r_[np.full(3, np.deg2rad(4.)), 1.5, 1.5, 4., 8., 8.]
        self.sigma = np.sqrt(.8*self.sigma**2+.2*np.maximum(innovation**2, floor**2))
        if reliable:
            angular = rotation_vector(rotation @ self.rotation.T)/dt
            angular *= min(1., np.deg2rad(2700.)/max(np.linalg.norm(angular), 1e-12))
            self.angular_velocity = .5*self.angular_velocity+.5*angular
            self.velocity = .5*self.velocity+.5*np.clip((position-self.position)/dt, -300., 300.)
            self.turn_velocity = (.5*self.turn_velocity+.5*np.clip(
                wrapped_angle_delta(angles, self.angles)/dt, -2700., 2700.)
                if fit.axis == self.axis else np.zeros(2))
        else:
            self.sigma *= 1.3
            self.angular_velocity *= .5
            self.velocity *= .5
            self.turn_velocity *= .5
        self.time, self.rotation, self.position = float(time), rotation, position.copy()
        self.angles, self.axis = angles.copy(), fit.axis
        self.history.append((self.time, self.position.copy()))
        while len(self.history) > 2 and self.history[1][0] < time-.5:
            self.history.popleft()


class CountedProjection(geometry.ArticulatedProjection):
    def evaluate(self, parameters):
        if self._last is None or not np.array_equal(parameters, self._last[0]):
            self.evaluations = getattr(self, 'evaluations', 0)+1
        return super().evaluate(parameters)


class PredictivePoseTracker(PhysicsPoseTracker):
    def __init__(self, renderer, *, iterations=2, damping=.25):
        super().__init__(renderer, iterations=iterations)
        self.damping = damping
        self.last_prediction_diagnostics = {}

    def initialize(self, fits, time, frame=None):
        super().initialize(fits, time, frame)
        for branch in self.branches:
            branch.motion = MotionState(branch.fit, time, damping=self.damping)

    def update(self, observations, time, frame=None):
        if not self.branches or time <= self.time:
            raise ValueError('Initialize first, then use increasing timestamps')
        m = self.m
        envelopes = [b.motion.predict(time) for b in self.branches]
        best = self.branches[0].fit.candidate
        base = replace(observations, rvec=best.rvec.copy(), tvec=envelopes[0].center[3:6].reshape(3, 1),
            layer_rotations=(), sticker_layer_rotations=None, assignment_object_points=None,
            assignment_projected_points=None, assignment_projected_class_ids=None,
            joint_geometry_candidates=())
        models = {a: CountedProjection(m, base, a) for a in range(3)}
        proposals, parents, scored, seen = [], {}, [], set()
        refinements = 0

        def refine(axis, seed, parent, *, bounded=True, iterations=None):
            nonlocal refinements
            key = axis, tuple(np.round(seed, 5)), bounded
            if key in seen:
                return
            seen.add(key)
            envelope = envelopes[parent] if bounded else None
            if envelope is not None and axis != self.branches[parent].fit.axis:
                # A newly starting turn has no history on this axis.
                envelope = MotionEnvelope(envelope.center.copy(), envelope.sigma.copy(), envelope.radius.copy())
                envelope.center[6:] = seed[6:]
                envelope.sigma[6:], envelope.radius[6:] = 30., 45.
            parameters = local_solve(models[axis], seed, envelopes[parent].center[3:6],
                iterations=self.iterations if iterations is None else iterations, envelope=envelope)
            parameters[6:] = (parameters[6:]+45.) % 90.-45.
            fit = geometry._materialize(m, base, axis, parameters)
            refinements += 1
            if fit is not None:
                fit.proposal_source = 'bounded_motion_prediction' if bounded else 'motion_recovery'
                proposals.append(fit)
                parents[id(fit)] = parent

        # One small fit per parent, initialized by prediction or a bounded
        # current-landmark PnP correction, rather than blind rotation probes.
        for parent, branch in enumerate(self.branches):
            axis, envelope = branch.fit.axis, envelopes[parent]
            seed = envelope.center.copy()
            stationary = np.r_[branch.fit.candidate.rvec.ravel(), envelope.center[3:6],
                               np.asarray(branch.fit.angles)[[0, 2]]]
            revised = envelope.constrain(pnp_seed(models[axis], seed, envelope.center[3:6]))
            choices = [seed, stationary, revised]
            ranked = sorted(((fast_geometry_score(m, models[axis], x), x) for x in choices), key=lambda row: row[0])
            scored.extend((score, axis, x, parent) for score, x in ranked)
            refine(axis, envelope.constrain(ranked[0][1]), parent)

        retained = geometry.diverse_shortlist(m, proposals)[:5]
        local_best = retained[0] if retained else None
        recovery = (local_best is None or local_best.score > .2 or
                    m.pose_reprojection_fit_metric(local_best.candidate) > .035)
        # Cheap axis probes allow new turns even while the camera prediction is
        # good. Full orientation probes are reserved for failed predictions.
        origin = envelopes[0].center[:6]
        probes = []
        for axis in range(3):
            for angles in ((0., 0.), (-30., 0.), (30., 0.), (0., -30.), (0., 30.)):
                seed = np.r_[origin, angles]
                probes.append((fast_geometry_score(m, models[axis], seed), axis, seed, 0))
            if recovery:
                for a in (-30., 30.):
                    for b in (-30., 30.):
                        seed = np.r_[origin, a, b]
                        probes.append((fast_geometry_score(m, models[axis], seed), axis, seed, 0))
                rotation = cv2.Rodrigues(best.rvec)[0]
                for degrees in (-30., -15., 15., 30.):
                    revised = rotation @ cv2.Rodrigues(np.eye(3)[axis]*np.deg2rad(degrees))[0]
                    seed = np.r_[rotation_vector(revised), origin[3:6], 0., 0.]
                    probes.append((fast_geometry_score(m, models[axis], seed), axis, seed, 0))
        probes.sort(key=lambda row: row[0])
        threshold = min(row[0] for row in scored)-.01
        for axis in range(3):
            row = next(p for p in probes if p[1] == axis)
            if recovery or len(retained) < 5 or row[0] < threshold:
                refine(*row[1:], bounded=not recovery, iterations=max(3, self.iterations) if recovery else None)
                retained = geometry.diverse_shortlist(m, proposals)[:5]
        # Preserve genuine diversity, not five copies of one local optimum.
        if len(retained) < 5:
            for _, axis, seed, parent in sorted([*scored, *probes], key=lambda row: row[0]):
                parameters = seed.copy()
                parameters[6:] = (parameters[6:]+45.) % 90.-45.
                fit = geometry._materialize(m, base, axis, parameters)
                if fit is not None:
                    proposals.append(fit)
                    parents[id(fit)] = parent
                    retained = geometry.diverse_shortlist(m, proposals)[:5]
                    if len(retained) == 5:
                        break
        old, self.branches = self.branches, []
        for fit in retained:
            parent = old[parents[id(fit)]]
            motion = parent.motion.copy()
            motion.observe(fit, time, reliable=fit.score <= .2 and m.pose_reprojection_fit_metric(fit.candidate) <= .035)
            core = parent.core.copy()
            core.observe(time, fit.candidate.tvec)
            branch = Branch(fit, core)
            branch.motion = motion
            self.branches.append(branch)
        self.time = float(time)
        self.core = self.branches[0].core
        self.last_prediction_diagnostics = dict(recovery=recovery, refinements=refinements,
            projection_evaluations=sum(getattr(model, 'evaluations', 0) for model in models.values()),
            scored_seeds=len(scored)+len(probes),
            rotation_radius_degrees=float(np.rad2deg(envelopes[0].radius[0])),
            translation_radius_mm=envelopes[0].radius[3:6].tolist())
        return retained
