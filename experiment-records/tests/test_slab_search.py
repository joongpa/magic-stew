from dataclasses import replace

import cv2
import numpy as np
import pytest

from scripts import render_yolo11n_independent_keypoints as m
from scripts import independent_keypoint_pose_geometry as geometry
from scripts.independent_keypoint_slab_search import SlabScorer, StaticSlabPose, fit_articulation, SlabHierarchyTracker, articulation_candidates


@pytest.fixture
def turning_cube():
    import torch
    torch.set_num_threads(2)
    axis, angles = 0, np.array([-28., 0., 24.])
    rvec, tvec = np.array([.4, -.6, .2]), np.array([-10., 2., 550.])
    camera_matrix = np.array([[1000., 0., 320.], [0., 1000., 240.], [0., 0., 1.]])
    rotation = cv2.Rodrigues(rvec)[0]
    camera = -rotation.T @ tvec
    matrices = np.asarray([cv2.Rodrigues(np.eye(3)[axis]*np.deg2rad(a))[0] for a in angles])
    layers, vertices, face_layers, _, _ = m._parallel_slab_geometry(axis)
    objects = np.einsum('nij,nj->ni', matrices[layers], m.CUBE_LATTICE_OBJECT_POINTS_MM[vertices])
    normals = np.einsum('nij,nfj->nfi', matrices[layers], m.CUBE_LATTICE_BOUNDARY_NORMALS[vertices])
    visible = (m.CUBE_LATTICE_BOUNDARY_FACE_MASK[vertices] & (
        np.sum(normals*(camera-objects)[:, None], axis=-1) > 0)).any(axis=1)
    faces = m._turn_gpu_static_geometry()
    polygons = np.einsum('nij,nkj->nki', matrices[face_layers], faces[2])
    centers = polygons.mean(axis=1)
    face_normals = np.einsum('nij,nj->ni', matrices[face_layers], faces[5])
    face_visible = np.sum(face_normals*(camera-centers), axis=-1) > 0
    for layer in range(3):
        visible &= m._parallel_slab_entry(camera, (objects-camera)[None], matrices[layer:layer+1], axis, layer)[0] >= 1.-1e-5
        face_visible &= m._parallel_slab_entry(camera, (centers-camera)[None], matrices[layer:layer+1], axis, layer)[0] >= 1.-1e-5
    project = lambda points: cv2.projectPoints(points, rvec, tvec, camera_matrix, np.zeros(5))[0].reshape(-1, 2)
    image_points = project(objects[visible])
    classes = m.CUBE_LATTICE_CLASS_IDS[vertices[visible]]
    _, unique = np.unique(np.round(np.column_stack((image_points, classes)), 5), axis=0, return_index=True)
    stickers = project(polygons.reshape(-1, 3)).reshape(-1, 4, 2).mean(axis=1)[face_visible]
    base = m.StickerPnPCandidate(sticker_type='synthetic', sticker_average_confidence=float('nan'),
        image_points=np.empty((0, 2)), object_points=np.empty((0, 3)), projected_sticker_points=np.empty((0, 2)),
        reprojection_error=float('nan'), camera_matrix=camera_matrix, dist_coeffs=np.zeros(5),
        image_size=(640, 480), rvec=rvec.reshape(3, 1), tvec=tvec.reshape(3, 1),
        assignment_image_points=image_points[unique], assignment_observed_class_ids=classes[unique],
        sticker_mask_centers=tuple(stickers), sticker_color_names=tuple('red' for _ in stickers))
    return base, axis, angles


def test_one_correct_static_slab_does_not_require_other_slabs_to_be_static(turning_cube):
    base, axis, angles = turning_cube
    x = np.r_[base.rvec.ravel(), base.tvec.ravel()]
    partial = SlabScorer(m, base).score([x], [0])[0]
    assert partial.landmark_support >= 4
    assert partial.sticker_support >= 1
    assert partial.spatial_support
    assert partial.score < .02
    rigid = geometry._materialize(m, base, axis, np.r_[x, 0., 0.])
    full = geometry._materialize(m, base, axis, np.r_[x, angles[[0, 2]]])
    assert m.pose_reprojection_fit_metric(rigid.candidate) > .01
    assert m.pose_reprojection_fit_metric(full.candidate) < 1e-6


@pytest.mark.parametrize('anchor', [0, 1, 2])
def test_fresh_articulation_recovers_full_cube_while_supported_slab_stays_fixed(turning_cube, anchor):
    base, axis, angles = turning_cube
    r = cv2.Rodrigues(base.rvec)[0] @ cv2.Rodrigues(np.eye(3)[axis]*np.deg2rad(angles[anchor]))[0]
    x = np.r_[cv2.Rodrigues(r)[0].ravel(), base.tvec.ravel()]
    proposal = StaticSlabPose(x, axis, anchor, 0., 8, 3, 0)
    fits = fit_articulation(m, base, proposal, iterations=6)
    fit = min(fits, key=lambda f: f.score)
    assert m.pose_reprojection_fit_metric(fit.candidate) < .003
    np.testing.assert_array_equal(fit.candidate.tvec, base.tvec)
    anchor_rotation = cv2.Rodrigues(fit.candidate.rvec)[0] @ cv2.Rodrigues(
        np.eye(3)[axis]*np.deg2rad(fit.angles[anchor]))[0]
    # Quarter-turn aliases preserve the slab shape when output angles fold.
    distances = [np.linalg.norm(cv2.Rodrigues(anchor_rotation @ (r @ s).T)[0]) for s in m.CUBE_SYMMETRY_MATRICES]
    assert min(distances) < 1e-8


def test_many_duplicate_detections_cannot_manufacture_a_good_slab_fit(turning_cube):
    base, _, _ = turning_cube
    duplicate = replace(base, assignment_image_points=np.repeat(base.assignment_image_points[:1], 30, axis=0),
        assignment_observed_class_ids=np.repeat(base.assignment_observed_class_ids[:1], 30))
    pose = SlabScorer(m, duplicate).score([np.r_[base.rvec.ravel(), base.tvec.ravel()]], [0])[0]
    assert pose.landmark_support <= 1
    assert not pose.spatial_support


@pytest.mark.parametrize('full_fit_acceptable', [True, False])
@pytest.mark.parametrize('all_axes', [False, True])
def test_single_supported_camera_still_returns_five_distinct_articulations(turning_cube, monkeypatch, full_fit_acceptable, all_axes):
    from scripts import independent_keypoint_slab_search as slab
    base, axis, _ = turning_cube
    x = np.r_[base.rvec.ravel(), base.tvec.ravel()]
    supported = StaticSlabPose(x, (axis+1) % 3 if all_axes else axis, 1, 0., 8, 3, 0)
    monkeypatch.setattr(slab, 'static_hierarchy', lambda *args, **kwargs: [supported])
    fits = [geometry._materialize(m, base, a, np.r_[x, angles]) for a, angles in
            [(0, (-30., 0.)), (0, (30., 0.)), (1, (0., -30.)), (1, (0., 30.)), (2, (30., -30.))]]
    tracker = SlabHierarchyTracker(m, all_axes=all_axes)
    tracker.initialize(fits, 0.)
    if not full_fit_acceptable:
        monkeypatch.setattr(m, 'pose_reprojection_fit_metric', lambda candidate: .08)
    retained = tracker.update(base, 1./30.)
    assert len(retained) == len(geometry.diverse_shortlist(m, retained)) == 5
    assert tracker.last_prediction_diagnostics['final_full_fit_accepted'] == full_fit_acceptable
    if not full_fit_acceptable:
        assert all(len(branch.motion.history) == 1 for branch in tracker.branches)
        assert all(branch.motion.history[0][0] == 0. for branch in tracker.branches)
        assert all(not branch.motion.angular_velocity.any() for branch in tracker.branches)


def test_full_wireframe_check_detects_bad_articulation_with_exact_translation(turning_cube):
    from scripts.audit_slab_pose_failures import wire_error
    base, axis, angles = turning_cube
    x = np.r_[base.rvec.ravel(), base.tvec.ravel()]
    reference = geometry._materialize(m, base, axis, np.r_[x, angles[[0, 2]]]).candidate
    rigid = geometry._materialize(m, base, axis, np.r_[x, 0., 0.]).candidate
    assert wire_error(reference, reference) == 0.
    np.testing.assert_array_equal(rigid.tvec, reference.tvec)
    assert wire_error(rigid, reference) > .02


def test_articulation_search_does_not_lock_to_wrong_static_slab_axis(turning_cube):
    base, axis, angles = turning_cube
    x = np.r_[base.rvec.ravel(), base.tvec.ravel()]
    wrong_axis = (axis+1) % 3
    proposal = StaticSlabPose(x, wrong_axis, 0, 0., 8, 3, 0)
    pose, options = articulation_candidates(m, base, proposal, all_axes=True)
    assert {option['layer_axis'] for option in options} == {0, 1, 2}
    correct = next(option for option in options if option['layer_axis'] == axis)
    fit = min(fit_articulation(m, base, replace(proposal, axis=axis, layer=1),
        prepared=(pose, [correct]), iterations=6), key=lambda f:f.score)
    assert m.pose_reprojection_fit_metric(fit.candidate) < .003
