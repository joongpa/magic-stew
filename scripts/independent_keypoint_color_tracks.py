"""Short-lived sticker observations for causal orientation proposals.

Color is a hard compatibility label for matching, not a permanent identity.
Every match needs two detector observations and forward/backward image motion.
An occlusion breaks the track; a later observation receives a fresh ID.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class StickerMatches:
    previous: np.ndarray
    current: np.ndarray
    tracking_error_pixels: np.ndarray
    track_ids: np.ndarray
    diagnostics: dict


class StickerColorTracker:
    def __init__(self, *, resolution=320, forward_backward_pixels=1.5):
        self.resolution = resolution
        self.forward_backward_pixels = forward_backward_pixels
        self.gray = None
        self.points = np.empty((0, 2))
        self.colors = ()
        self.ids = np.empty(0, int)
        self.next_id = 0

    def initialize(self, frame, points, colors):
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        self.points = np.asarray(points, float).reshape(-1, 2).copy()
        self.colors = tuple(colors)
        if len(self.points) != len(self.colors):
            raise ValueError('Every sticker point needs a detected color')
        self.ids = np.arange(self.next_id, self.next_id+len(self.points))
        self.next_id += len(self.points)

    def match(self, frame, points, colors, diameter):
        current = np.asarray(points, float).reshape(-1, 2)
        colors = tuple(colors)
        if len(current) != len(colors):
            raise ValueError('Every sticker point needs a detected color')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        predictions = np.full_like(self.points, np.nan)
        tracking_errors = np.full(len(self.points), np.inf)
        if (self.gray is not None and self.gray.shape == gray.shape and len(self.points)
                and len(current) and np.isfinite(diameter) and diameter > 0):
            all_points = np.vstack((self.points, current))
            height, width = gray.shape
            pad = .3*diameter
            lower = np.maximum(0, np.floor(all_points.min(axis=0)-pad)).astype(int)
            upper = np.minimum([width, height], np.ceil(all_points.max(axis=0)+pad)).astype(int)
            x0, y0 = lower
            x1, y1 = upper
            if x1-x0 >= 16 and y1-y0 >= 16:
                scale = min(2., self.resolution/max(x1-x0, y1-y0))
                size = max(16, round((x1-x0)*scale)), max(16, round((y1-y0)*scale))
                before = cv2.resize(self.gray[y0:y1, x0:x1], size)
                after = cv2.resize(gray[y0:y1, x0:x1], size)
                # Satellite patches reach nearby sticker boundaries even when
                # the color keypoint itself lies in a textureless paint region.
                radius = .035*diameter
                offsets = radius*np.array([[0., 0.], [-1., -1.], [1., -1.], [-1., 1.], [1., 1.]])
                initial = ((self.points[:, None]+offsets[None]-lower)*scale).astype(np.float32).reshape(-1, 1, 2)
                options = dict(winSize=(21, 21), maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, .01),
                    minEigThreshold=1e-4)
                forward, good_forward, _ = cv2.calcOpticalFlowPyrLK(before, after, initial, None, **options)
                if forward is not None:
                    backward, good_backward, _ = cv2.calcOpticalFlowPyrLK(after, before, forward, None, **options)
                    if backward is not None:
                        error = np.linalg.norm((backward-initial).reshape(-1, 2), axis=1)
                        good = (good_forward.ravel().astype(bool) & good_backward.ravel().astype(bool)
                                & np.isfinite(error) & (error <= self.forward_backward_pixels))
                        displacement = ((forward-initial).reshape(-1, 5, 2))/scale
                        good = good.reshape(-1, 5)
                        error = error.reshape(-1, 5)/scale
                        for i in range(len(self.points)):
                            valid = good[i] & np.isfinite(displacement[i]).all(axis=1)
                            if np.count_nonzero(valid) < 2:
                                continue
                            motion = np.median(displacement[i, valid], axis=0)
                            spread = np.median(np.linalg.norm(displacement[i, valid]-motion, axis=1))
                            if spread > max(2./scale, .025*diameter):
                                continue
                            predictions[i] = self.points[i]+motion
                            tracking_errors[i] = np.median(error[i, valid])+spread
        old_rows, new_rows, errors = match_color_predictions(
            predictions, self.colors, current, colors, max_distance=max(3., .06*diameter))
        ids = np.arange(self.next_id, self.next_id+len(current))
        self.next_id += len(current)
        if len(old_rows):
            ids[new_rows] = self.ids[old_rows]
        diagnostics = dict(previous_observations=len(self.points), current_observations=len(current),
            flow_supported_observations=int(np.count_nonzero(np.isfinite(predictions).all(axis=1))),
            matched_observations=len(old_rows), new_tracks=int(len(current)-len(old_rows)),
            dropped_tracks=int(len(self.points)-len(old_rows)))
        result = StickerMatches(old_rows, new_rows, tracking_errors[old_rows]+errors,
                                ids[new_rows].copy(), diagnostics)
        self.gray, self.points, self.colors, self.ids = gray, current.copy(), colors, ids
        return result


def match_color_predictions(predictions, old_colors, current, new_colors, *, max_distance):
    """One-to-one color-gated matches, rejecting ambiguous same-color choices."""
    predictions, current = np.asarray(predictions), np.asarray(current)
    if not len(predictions) or not len(current):
        return np.empty(0, int), np.empty(0, int), np.empty(0)
    distance = np.linalg.norm(predictions[:, None]-current[None], axis=-1)
    compatible = np.asarray(old_colors)[:, None] == np.asarray(new_colors)[None]
    allowed = compatible & np.isfinite(distance) & (distance <= max_distance)
    cost = np.where(allowed, distance, 1e6)
    previous, following = linear_sum_assignment(cost)
    keep = allowed[previous, following]
    # A repeated color is not enough to resolve two equally plausible stickers.
    for k, (i, j) in enumerate(zip(previous, following, strict=True)):
        alternatives = distance[i, compatible[i] & np.isfinite(distance[i])]
        if len(alternatives) >= 2:
            nearest = np.sort(alternatives)[:2]
            if nearest[1]-nearest[0] < max(1., .15*max_distance):
                keep[k] = False
    previous, following = previous[keep], following[keep]
    return previous, following, distance[previous, following]


def bind_color_tracks(renderer, fit, matches, current, center_tracks=None, *, return_track_ids=False):
    """Associate tracks with each parent's own previous geometric sticker slots."""
    if fit.nearest is None or fit.valid is None:
        result = np.empty(0, int), np.empty((0, 2))
        return (*result, np.empty(0, int)) if return_track_ids else result
    valid_pairs = []
    center_tracks = center_tracks or {}
    for i, j, error, identity in zip(matches.previous, matches.current,
                                    matches.tracking_error_pixels, matches.track_ids, strict=True):
        if int(identity) in center_tracks:
            valid_pairs.append((error, center_tracks[int(identity)], int(j), int(identity)))
        elif i < len(fit.valid) and fit.valid[i]:
            valid_pairs.append((error, int(fit.nearest[i]), int(j), int(identity)))
    seen, faces, points, identities = set(), [], [], []
    for _, face, row, identity in sorted(valid_pairs):
        if face in seen:
            continue
        seen.add(face)
        faces.append(face)
        points.append(current[row])
        identities.append(identity)
    result = np.asarray(faces, int), np.asarray(points, float).reshape(-1, 2)
    return (*result, np.asarray(identities, int)) if return_track_ids else result


def center_profile_support(candidate, diameter):
    """Positive image evidence for a center sticker: four inner-grid landmarks.

    Merely landing in a fitted center polygon is insufficient; a wrong pose can
    place an ordinary colored sticker there. No confidence values are created.
    """
    stickers = np.asarray(candidate.sticker_mask_centers).reshape(-1, 2)
    supported = np.zeros(len(stickers), bool)
    if candidate.assignment_image_points is None or candidate.assignment_observed_class_ids is None:
        return supported
    observed = np.asarray(candidate.assignment_image_points).reshape(-1, 2)
    classes = np.asarray(candidate.assignment_observed_class_ids).reshape(-1)
    if len(classes) != len(observed) or not np.isfinite(diameter) or diameter <= 0:
        return supported
    inner = observed[classes == 2]
    if len(inner) < 4:
        return supported
    for row, point in enumerate(stickers):
        nearest = inner[np.argsort(np.linalg.norm(inner-point, axis=1))[:4]]
        hull = cv2.convexHull(nearest.astype(np.float32)).reshape(-1, 2)
        if len(hull) != 4:
            continue
        area = cv2.contourArea(hull)
        if not .0003*diameter**2 <= area <= .1*diameter**2:
            continue
        if cv2.pointPolygonTest(hull, tuple(map(float, point)), False) < 0:
            continue
        # A true projected square center is the diagonal intersection.
        first, second = hull[2]-hull[0], hull[3]-hull[1]
        system = np.column_stack((first, -second))
        if abs(np.linalg.det(system)) < 1e-6:
            continue
        fraction = np.linalg.solve(system, hull[1]-hull[0])[0]
        center = hull[0]+fraction*first
        supported[row] = np.linalg.norm(center-point) <= max(2., .03*diameter)
    return supported


def sticker_objects(renderer, fit, faces):
    face_polygons = renderer._turn_gpu_static_geometry()[2]
    face_layers = renderer._parallel_slab_geometry(fit.axis)[2]
    angles = np.asarray(fit.angles)
    matrices = np.asarray([cv2.Rodrigues(np.eye(3)[fit.axis]*np.deg2rad(a))[0] for a in angles])
    polygons = np.einsum('nij,nkj->nki', matrices[face_layers[faces]], face_polygons[faces])
    return polygons.mean(axis=1)


@lru_cache(maxsize=4)
def _center_sticker_mask(renderer):
    centers = renderer._turn_gpu_static_geometry()[2].mean(axis=1)
    return np.count_nonzero(np.abs(centers) > 1e-4, axis=1) == 1


def center_sticker_rows(renderer, faces):
    return np.flatnonzero(_center_sticker_mask(renderer)[faces])


def center_orientation_seed(renderer, fit, faces, observed, predicted_core, max_rotation_degrees):
    """Weak regularized orientation proposal from the tracked center cross.

    Unlike ordinary stickers, face-center positions do not move under an outer
    face turn. One or two visible centers are intentionally only a proposal:
    priors supply the unobserved degrees of freedom, never fake measurements.
    """
    if not len(faces):
        return None
    objects = sticker_objects(renderer, fit, faces)
    pose = fit.candidate
    original = np.r_[pose.rvec.ravel(), predicted_core]
    x = original.copy()
    sigma = np.array([.3, .3, .3, 4., 4., 15.])
    scale = np.array([.15, .15, .15, 3., 3., 8.])
    pixel_scale = max(renderer.projected_cube_diameter(pose)/220., .3)
    for _ in range(3):
        projected, derivative = cv2.projectPoints(objects, x[:3], x[3:6], pose.camera_matrix, pose.dist_coeffs)
        residual = (projected.reshape(-1, 2)-observed).ravel()/pixel_scale
        jac = derivative[:, :6]*scale/pixel_scale
        weight = 1./np.sqrt(1.+(residual/3.)**2)
        regularizer = scale/sigma
        normal = jac.T @ (jac*weight[:, None])+np.diag(regularizer**2+.1)
        gradient = jac.T @ (residual*weight)+regularizer*(x-original)/sigma
        step = np.linalg.solve(normal, -gradient)
        step /= max(1., np.linalg.norm(step))
        x += step*scale
    angle = np.rad2deg(np.linalg.norm(cv2.Rodrigues(
        cv2.Rodrigues(x[:3])[0] @ cv2.Rodrigues(pose.rvec)[0].T)[0]))
    if x[5] <= renderer.CUBE_SIZE_MM or angle > max_rotation_degrees+10.:
        return None
    return np.r_[x, np.asarray(fit.angles)[[0, 2]]]


def color_pose_seed(renderer, fit, faces, image_points, predicted_core, max_rotation_degrees):
    """Robust rigid proposal; moving-layer tracks may be outliers."""
    if len(faces) < 6:
        return None, dict(inliers=0, strong=False)
    objects = sticker_objects(renderer, fit, faces)
    pose = fit.candidate
    gate = max(2., .025*renderer.projected_cube_diameter(pose))
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(np.ascontiguousarray(objects),
            np.ascontiguousarray(image_points), pose.camera_matrix, pose.dist_coeffs,
            pose.rvec.copy(), pose.tvec.copy(), useExtrinsicGuess=True, iterationsCount=60,
            reprojectionError=gate, confidence=.99, flags=cv2.SOLVEPNP_ITERATIVE)
    except cv2.error:
        return None, dict(inliers=0, strong=False)
    if not ok or inliers is None or len(inliers) < 6 or tvec[2, 0] <= renderer.CUBE_SIZE_MM:
        return None, dict(inliers=0, strong=False)
    relative = cv2.Rodrigues(rvec)[0] @ cv2.Rodrigues(pose.rvec)[0].T
    angle = np.rad2deg(np.linalg.norm(cv2.Rodrigues(relative)[0]))
    if np.linalg.norm(tvec.ravel()-predicted_core) > 40. or angle > max_rotation_degrees+10.:
        return None, dict(inliers=len(inliers), strong=False)
    selected = objects[inliers.ravel()]
    singular = np.linalg.svd(selected-selected.mean(axis=0), compute_uv=False)
    nonplanar = singular[-1] > .05*max(singular[0], 1e-9)
    layers = renderer._parallel_slab_geometry(fit.axis)[2][faces[inliers.ravel()]]
    supports_core = len(np.unique(layers)) >= 2 and np.any(layers == 1)
    strong = len(inliers) >= 8 and len(inliers) >= .7*len(faces) and nonplanar and supports_core
    return np.r_[rvec.ravel(), tvec.ravel(), np.asarray(fit.angles)[[0, 2]]], dict(
        inliers=len(inliers), observations=len(faces), strong=bool(strong), planar=not nonplanar,
        inlier_rows=inliers.ravel().tolist())


def color_similarity_seed(renderer, fit, faces, image_points, matches, predicted_core,
                          max_rotation_degrees=90.):
    """Weaker in-plane motion cue when too few tracks support 3-D PnP."""
    if len(matches.previous) < 3:
        return None
    previous = np.asarray(fit.candidate.sticker_mask_centers).reshape(-1, 2)[matches.previous]
    # image_points here follows the observation pairing, not the deduplicated
    # face binding used by the rigid 3-D proposal.
    transform, inliers = cv2.estimateAffinePartial2D(previous, image_points,
        method=cv2.RANSAC, ransacReprojThreshold=max(2., .025*renderer.projected_cube_diameter(fit.candidate)),
        maxIters=80, confidence=.99)
    if transform is None or inliers is None or np.count_nonzero(inliers) < 3:
        return None
    scale = np.sqrt(abs(np.linalg.det(transform[:, :2])))
    if not .8 <= scale <= 1.25:
        return None
    angle = np.arctan2(transform[1, 0], transform[0, 0])
    if abs(np.rad2deg(angle)) > max_rotation_degrees+10.:
        return None
    pose = fit.candidate
    before = renderer._rigid_core_coordinates(pose)
    uv = transform[:, :2] @ before[:2]+transform[:, 2]
    translation = renderer._rigid_translation(np.r_[uv, before[2]/scale], pose).ravel()
    if np.linalg.norm(translation-predicted_core) > 40.:
        return None
    rotation = cv2.Rodrigues(np.array([0., 0., angle]))[0] @ cv2.Rodrigues(pose.rvec)[0]
    return np.r_[cv2.Rodrigues(rotation)[0].ravel(), translation, np.asarray(fit.angles)[[0, 2]]]


class TrackedStickerProjection:
    """Soft current-image constraints from short-lived physical associations."""
    def __init__(self, base, faces, observed, weight=.7):
        self.base, self.faces, self.observed = base, faces, observed
        self.weight, self.m = weight, base.m

    def evaluate(self, parameters):
        residual, jacobian = self.base.evaluate(parameters)
        centers, derivatives = self.base.sticker_projection_cache
        fixed = (centers[self.faces]-self.observed).ravel()/self.base.scale
        derivative = derivatives[self.faces].reshape(-1, 8)/self.base.scale
        return np.r_[residual, self.weight*fixed], np.vstack((jacobian, self.weight*derivative))


COLOR_CENTER_DIRECTIONS = {
    'red': np.array([1., 0., 0.]), 'orange': np.array([-1., 0., 0.]),
    'blue': np.array([0., 1., 0.]), 'green': np.array([0., -1., 0.]),
    'white': np.array([0., 0., 1.]), 'yellow': np.array([0., 0., -1.]),
}


def colored_center_orientation(directions, colors):
    """Recover a labeled center cross from at least two nonparallel colors.

    Returns None for duplicate, opposite-only, or inconsistent center labels.
    This is a model-supported orientation cue, not a detector confidence score.
    """
    if len(colors) < 2 or len(set(colors)) != len(colors):
        return None
    source = np.asarray([COLOR_CENTER_DIRECTIONS[c] for c in colors])
    target = np.asarray(directions, float).copy()
    target /= np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-12)
    if np.linalg.matrix_rank(source, tol=.1) < 2 or np.linalg.matrix_rank(target, tol=.1) < 2:
        return None
    if np.max(np.abs(source @ source.T-target @ target.T)) > .25:
        return None
    u, _, vt = np.linalg.svd(target.T @ source)
    return u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt


def tracked_center_orientation(renderer, branch, tracker):
    by_id = {int(identity):color for identity, color in zip(tracker.ids, tracker.colors, strict=True)}
    mapping = branch.center_tracks or {}
    entries = [(face, by_id[identity]) for identity, face in mapping.items() if identity in by_id]
    if len(entries) < 2:
        return None
    faces, labels = zip(*entries)
    objects = renderer._turn_gpu_static_geometry()[2][list(faces)].mean(axis=1)
    rotation = cv2.Rodrigues(branch.fit.candidate.rvec)[0]
    return colored_center_orientation(objects @ rotation.T, labels)
