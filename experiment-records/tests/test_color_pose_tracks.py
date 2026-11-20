"""Short color tracks must survive motion and fail safely at occlusions."""
from pathlib import Path
import json

import cv2
import numpy as np
import pytest

from scripts import independent_keypoint_color_tracks as colors
from scripts import independent_keypoint_pose_geometry as geometry
from scripts import render_yolo11n_independent_keypoints as renderer


def painted_stickers():
    image = np.full((240, 240, 3), 30, np.uint8)
    points = np.array([(50+70*x, 50+70*y) for y in range(3) for x in range(3)], float)
    labels = tuple(('red', 'green', 'blue')[i % 3] for i in range(9))
    palette = dict(red=(30, 30, 220), green=(30, 200, 30), blue=(220, 30, 30))
    rng = np.random.default_rng(8)
    for point, label in zip(points, labels, strict=True):
        x, y = point.astype(int)
        cv2.rectangle(image, (x-24, y-24), (x+24, y+24), palette[label], -1)
        for _ in range(24):
            dx, dy = rng.integers(-20, 21, 2)
            cv2.circle(image, (x+int(dx), y+int(dy)), 1, (220, 220, 220), -1)
    return image, points, labels


def test_color_and_ambiguity_gates_do_not_force_identity():
    predictions = np.array([[0., 0.], [30., 0.], [60., 0.]])
    current = np.array([[1., 0.], [29., 0.], [31., 0.], [61., 0.]])
    old, new, _ = colors.match_color_predictions(predictions, ('red', 'blue', 'red'),
        current, ('green', 'blue', 'blue', 'red'), max_distance=10.)
    assert old.tolist() == [2]
    assert new.tolist() == [3]


def test_optical_flow_matches_colored_stickers_after_image_motion():
    image, points, labels = painted_stickers()
    tracker = colors.StickerColorTracker()
    tracker.initialize(image, points, labels)
    moved = cv2.warpAffine(image, np.array([[1., 0., 6.], [0., 1., -4.]], np.float32), (240, 240))
    matches = tracker.match(moved, points+[6., -4.], labels, 200.)
    assert len(matches.previous) >= 8
    np.testing.assert_array_equal(matches.previous, matches.current)
    assert np.max(matches.tracking_error_pixels) < 2.


def test_lost_sticker_is_reborn_instead_of_reusing_occluded_identity():
    image, points, labels = painted_stickers()
    tracker = colors.StickerColorTracker()
    tracker.initialize(image, points, labels)
    old_id = tracker.ids[4]
    occluded = image.copy()
    x, y = points[4].astype(int)
    cv2.rectangle(occluded, (x-28, y-28), (x+28, y+28), (0, 0, 0), -1)
    keep = np.arange(len(points)) != 4
    tracker.match(occluded, points[keep], tuple(np.asarray(labels)[keep]), 200.)
    tracker.match(image, points, labels, 200.)
    assert tracker.ids[4] != old_id


@pytest.fixture
def fit():
    import torch
    from scripts import render_independent_keypoint_search as playback
    from scripts import search_independent_keypoint_full_solve as search
    torch.set_num_threads(2)
    record = json.loads((Path(__file__).parent/'fixtures/matty469-observations.json').read_text())
    palette = search.initial_sticker_colors(renderer,
        "U2 F2 U B2 U R2 F2 R2 D' U' L' B2 U2 F L' R U B' L' F'")
    candidate = playback.candidate_for_step(renderer, record, None,
        renderer.CubeGeometryState.identity(), palette)
    seed = np.r_[candidate.rvec.ravel(), candidate.tvec.ravel(), -23., 4.]
    return geometry._materialize(renderer, candidate, 1, seed)


def test_fixed_color_correspondence_jacobian_matches_finite_difference(fit):
    model = geometry.ArticulatedProjection(renderer, fit.candidate, fit.axis, cache_sticker_jacobians=True)
    seed = np.r_[fit.candidate.rvec.ravel(), fit.candidate.tvec.ravel(), -23., 4.]
    model.evaluate(seed)
    faces = np.array([1, 5, 13])
    tracked = colors.TrackedStickerProjection(model, faces,
        model.sticker_projection_cache[0][faces]+1.)
    _, analytic = tracked.evaluate(seed)
    step = 1e-6
    numeric = np.column_stack([(tracked.evaluate(seed+axis*step)[0]-
        tracked.evaluate(seed-axis*step)[0])/(2*step) for axis in np.eye(8)])
    np.testing.assert_allclose(analytic[-6:], numeric[-6:], atol=1e-5, rtol=1e-5)


def test_color_ransac_discards_bad_tracks_when_proposing_rigid_motion(fit):
    faces = np.unique(fit.nearest[fit.valid]).astype(int)
    assert len(faces) >= 9
    objects = colors.sticker_objects(renderer, fit, faces)
    rotation = cv2.Rodrigues(np.array([.02, -.01, .01]))[0] @ cv2.Rodrigues(fit.candidate.rvec)[0]
    target_rvec = cv2.Rodrigues(rotation)[0]
    target_tvec = fit.candidate.tvec+np.array([[1.], [-1.], [2.]])
    points = cv2.projectPoints(objects, target_rvec, target_tvec,
        fit.candidate.camera_matrix, fit.candidate.dist_coeffs)[0].reshape(-1, 2)
    points[:2] += [200., -250.]
    proposal, diagnostics = colors.color_pose_seed(renderer, fit, faces, points,
        fit.candidate.tvec.ravel(), 90.)
    assert proposal is not None
    assert diagnostics['inliers'] <= len(faces)-2
    relative = cv2.Rodrigues(proposal[:3])[0] @ rotation.T
    assert np.rad2deg(np.linalg.norm(cv2.Rodrigues(relative)[0])) < .5


def test_one_moving_slab_never_enables_fast_rigid_tracking(fit):
    layers = renderer._parallel_slab_geometry(fit.axis)[2]
    faces = np.flatnonzero(layers == 2)
    objects = colors.sticker_objects(renderer, fit, faces)
    points = cv2.projectPoints(objects, fit.candidate.rvec, fit.candidate.tvec,
        fit.candidate.camera_matrix, fit.candidate.dist_coeffs)[0].reshape(-1, 2)
    _, diagnostics = colors.color_pose_seed(renderer, fit, faces, points,
        fit.candidate.tvec.ravel(), 90.)
    assert not diagnostics['strong']


def test_center_correspondences_survive_geometric_rematching_but_not_track_loss(fit):
    faces = np.arange(54)
    center = int(colors.center_sticker_rows(renderer, faces)[0])
    nearest = fit.nearest.copy()
    nearest[0] = (center+1) % 54
    from dataclasses import replace
    damaged = replace(fit, nearest=nearest, valid=np.ones_like(fit.valid))
    matches = colors.StickerMatches(np.array([0]), np.array([0]), np.array([.1]),
        np.array([42]), {})
    actual, _ = colors.bind_color_tracks(renderer, damaged, matches, np.array([[10., 20.]]),
                                         center_tracks={42:center})
    assert actual.tolist() == [center]
    matches.track_ids[:] = 43  # A reacquired observation is a different track.
    actual, _ = colors.bind_color_tracks(renderer, damaged, matches, np.array([[10., 20.]]),
                                         center_tracks={42:center})
    assert actual.tolist() == [int(nearest[0])]


def test_center_positions_are_invariant_to_outer_slab_turns(fit):
    from dataclasses import replace
    faces = np.arange(54)
    centers = colors.center_sticker_rows(renderer, faces)
    first = colors.sticker_objects(renderer, fit, centers)
    changed = replace(fit, angles=(43., 0., -41.))
    second = colors.sticker_objects(renderer, changed, centers)
    np.testing.assert_allclose(first, second, atol=1e-6)


def test_center_motion_proposes_orientation_without_faking_unobserved_points(fit):
    all_faces = np.arange(54)
    faces = colors.center_sticker_rows(renderer, all_faces)
    objects = colors.sticker_objects(renderer, fit, faces)
    camera = -cv2.Rodrigues(fit.candidate.rvec)[0].T @ fit.candidate.tvec.ravel()
    visible = np.sum(objects*camera, axis=1) > 0
    faces, objects = faces[visible], objects[visible]
    target_rotation = cv2.Rodrigues(np.array([.02, -.01, .01]))[0] @ cv2.Rodrigues(fit.candidate.rvec)[0]
    target_rvec = cv2.Rodrigues(target_rotation)[0]
    observed = cv2.projectPoints(objects, target_rvec, fit.candidate.tvec,
        fit.candidate.camera_matrix, fit.candidate.dist_coeffs)[0].reshape(-1, 2)
    result = colors.center_orientation_seed(renderer, fit, faces, observed,
        fit.candidate.tvec.ravel(), 90.)
    assert result is not None
    before = cv2.projectPoints(objects, fit.candidate.rvec, fit.candidate.tvec,
        fit.candidate.camera_matrix, fit.candidate.dist_coeffs)[0].reshape(-1, 2)
    after = cv2.projectPoints(objects, result[:3], result[3:6],
        fit.candidate.camera_matrix, fit.candidate.dist_coeffs)[0].reshape(-1, 2)
    assert np.linalg.norm(after-observed) < np.linalg.norm(before-observed)


def test_two_nonparallel_center_colors_determine_labeled_orientation():
    rotation = cv2.Rodrigues(np.array([.2, -.3, .4]))[0]
    canonical = np.array([colors.COLOR_CENTER_DIRECTIONS['red'], colors.COLOR_CENTER_DIRECTIONS['white']])
    directions = canonical @ rotation.T
    before = directions.copy()
    estimated = colors.colored_center_orientation(directions, ('red', 'white'))
    np.testing.assert_allclose(estimated, rotation, atol=1e-12)
    np.testing.assert_array_equal(directions, before)


def test_missing_duplicate_and_opposite_only_centers_do_not_claim_orientation():
    assert colors.colored_center_orientation([[1., 0., 0.]], ('red',)) is None
    assert colors.colored_center_orientation([[1., 0., 0.], [0., 0., 1.]], ('red', 'red')) is None
    assert colors.colored_center_orientation([[1., 0., 0.], [-1., 0., 0.]], ('red', 'orange')) is None


def test_center_profile_does_not_mistake_ordinary_stickers_for_face_centers():
    from types import SimpleNamespace
    _, stickers, _ = painted_stickers()
    candidate = SimpleNamespace(assignment_image_points=np.array([[85., 85.], [155., 85.],
        [85., 155.], [155., 155.]]), assignment_observed_class_ids=np.array([2, 2, 2, 2]),
        sticker_mask_centers=stickers)
    support = colors.center_profile_support(candidate, 300.)
    assert np.flatnonzero(support).tolist() == [4]
    candidate.assignment_observed_class_ids[:] = 1
    assert not colors.center_profile_support(candidate, 300.).any()
