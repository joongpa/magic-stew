"""The learned renderer must preserve projection, articulation and score semantics."""
import argparse
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from live_recognition.direct_cube_pose import articulated_landmarks, project_points
from live_recognition.staged_pose.kinematics import rotation_exp
from scripts import render_yolo11n_independent_keypoints as renderer
from scripts import independent_keypoint_staged_pose as staged
from scripts import search_independent_keypoint_full_solve as search


def options(*extra):
    return renderer.build_parser().parse_args([
        '--source', 'example.png', '--initial-scramble', '', '--pose-method', 'staged',
        '--no-display', '--no-show-fps', '--no-show-sticker-identities', *extra])


def prediction(family=0):
    angles = torch.zeros(1, 6, dtype=torch.float64)
    angles[0, 2*family:2*family+2] = torch.tensor([23., -11.])
    pose = dict(rotation_matrix=rotation_exp(torch.tensor([[.3, -.7, .2]], dtype=torch.float64)),
        angles=angles, center=torch.tensor([[.53, .46]], dtype=torch.float64),
        principal=torch.tensor([[.47, .54]], dtype=torch.float64),
        scale=torch.tensor([.13], dtype=torch.float64), perspective=torch.tensor([.21], dtype=torch.float64))
    value = staged.StagedPrediction({k:v[0].numpy() for k,v in pose.items()},
        np.array([[80, 80], [120, 120], [150, 90], [100, 100]], dtype=np.float32),
        np.array([.99, .98, .97, .96]), np.array([0, 1, 2, 3]), np.array([.8, .7, .6, .95]))
    return value, pose


@pytest.mark.parametrize('family', [0, 1, 2])
def test_camera_and_all_face_turn_signs_match_training_projection(family):
    value, pose = prediction(family)
    roi = (43, 29, 431, 337)  # Off-center, non-square ROI exercises both focal lengths.
    candidate = staged.candidate_from_prediction(renderer, value, roi, (640, 480), options())
    # Compare every cubelet's eight vertices, preserving physical ownership.
    owners = torch.cartesian_prod(*[torch.tensor([-1., 0., 1.], dtype=torch.float64)]*3)
    offsets = torch.cartesian_prod(*[torch.tensor([-.5, .5], dtype=torch.float64)]*3)
    rest = (owners[:, None]+offsets).reshape(-1, 3)
    actual = articulated_landmarks(rest, owners.repeat_interleave(8, 0), pose['angles'])
    expected = project_points(actual, pose)[0].numpy() * (388, 308) + (43, 29)
    converted = []
    for center, vertices in zip(owners.numpy(), rest.reshape(27, 8, 3).numpy(), strict=True):
        _, points = renderer.apply_cubelet_layer_rotations(
            center*renderer.CUBE_LAYER_SIZE_MM, vertices*renderer.CUBE_LAYER_SIZE_MM,
            candidate.layer_rotations)
        converted.append(points)
    converted = np.concatenate(converted)
    pixels, _ = cv2.projectPoints(converted, candidate.rvec, candidate.tvec,
        candidate.camera_matrix, candidate.dist_coeffs)
    np.testing.assert_allclose(pixels.reshape(-1, 2), expected, atol=5e-5, rtol=0)


def test_predictor_uses_actual_object_scores_and_separate_presence_one_rgb_pass():
    args = options('--pose-conf', '0.1')
    value, pose = prediction()
    presence = torch.tensor([.82, .999, .65])
    object_scores = torch.zeros(1, 9, 3)
    object_scores[0, 0, 0] = .73
    object_scores[0, 3, 1] = .94
    # Third object is suppressed by NMS, even though presence is positive.
    raw_kpts = torch.zeros(1, 3, 3)
    raw_kpts[0, 2] = presence.logit()
    calls = []
    class Model:
        def encode(self, images):
            calls.append(images.clone())
            assert images.shape == (1, 3, 128, 128)
            assert images[0, :, 0, 0].tolist() == [1., 0., 0.]  # BGR -> RGB
            return dict(static=pose, encoded=dict(points=torch.tensor([[[.1, .2], [.3, .4], [.5, .6]]]),
                object_scores=object_scores, raw={'one2many': {'kpts': raw_kpts}}))
        def articulate(self, context):
            return None, None
    instance = staged.StagedPosePredictor.__new__(staged.StagedPosePredictor)
    instance.model = Model(); instance.device = torch.device('cpu'); instance.image_size = 128
    from unittest.mock import patch
    with patch('live_recognition.staged_pose.phase.decode', return_value=(pose['angles'], None)):
        result = instance.predict_crop(np.full((91, 103, 3), [0, 0, 255], dtype=np.uint8), args)
    assert len(calls) == 1
    np.testing.assert_allclose(result.object_scores, [.94, .73])
    np.testing.assert_allclose(result.presence_scores, [.999, .82])
    assert result.class_ids.tolist() == [3, 0]
    np.testing.assert_allclose(result.local_points, np.array([[.3, .4], [.1, .2]])*args.view_size)


def test_staged_fresh_and_cached_render_never_call_manual_fitters(monkeypatch):
    args = options('--no-show-pnp')
    value, _ = prediction()
    monkeypatch.setattr(renderer, 'shared_roi_for_frame', lambda *_: ((10, 10, 110, 110), None, 'detector'))
    def forbidden(*_a, **_kw):
        pytest.fail('The staged path called a manual pose fitter')
    for name in ['predict_pose_result', 'best_sticker_pnp_candidates', 'solve_pnp_from_low_error_assignments',
                 'recover_cached_pose_symmetries', 'smooth_cached_rigid_poses']:
        monkeypatch.setattr(renderer, name, forbidden)
    from scripts import independent_keypoint_pose_geometry as geometry
    monkeypatch.setattr(geometry, 'process_pose_geometry', forbidden)
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    cache = []
    class Model:
        count = 0
        def predict_crop(self, roi, config):
            self.count += 1
            return value
    model = Model()
    renderer.render_frame(frame, None, model, renderer.RoiTracker(), args, neural_cache=cache, neural_only=True)
    assert model.count == 1
    assert cache[0].pose_ready and cache[0].pose_candidate is not None
    original = cache[0].pose_candidate
    renderer.render_frame(frame, None, None, None, args, neural_data=cache[0], candidate_only=True)
    args.show_pnp = True
    monkeypatch.setattr(renderer, 'orient_cached_poses', lambda *_: None)
    renderer.process_cached_poses(cache, args, {})
    renderer.render_frame(frame, None, None, None, args, neural_data=cache[0], cached_final_only=True)
    np.testing.assert_array_equal(cache[0].pose_candidate.rvec, original.rvec)
    np.testing.assert_array_equal(cache[0].pose_candidate.tvec, original.tvec)
    assert cache[0].pose_candidate.layer_rotations == original.layer_rotations
    assert len(cache[0].pose_candidate.joint_geometry_candidates) == 1
    assert model.count == 1


def test_fixed_geometry_cache_preserves_predicted_camera_and_angles(monkeypatch):
    value, _ = prediction(2)
    args = options()
    candidate = staged.candidate_from_prediction(renderer, value, (0, 0, 256, 256), (256, 256), args)
    def forbidden(*_a, **_kw):
        pytest.fail('Fixed predicted geometry must not sweep or optimize')
    from scripts import independent_keypoint_pose_geometry as geometry
    monkeypatch.setattr(geometry, '_refine', forbidden)
    monkeypatch.setattr(geometry, 'swept_geometries', forbidden)
    cached = staged.cache_predicted_geometry(renderer, candidate)
    np.testing.assert_array_equal(cached.rvec, candidate.rvec)
    np.testing.assert_array_equal(cached.tvec, candidate.tvec)
    assert cached.layer_rotations == candidate.layer_rotations
    assert len(cached.joint_geometry_candidates) == 1
    payload = cached.joint_geometry_candidates[0]
    assert payload['pose_proposal_source'] == 'staged_network'
    # Fitter's projection cache agrees with direct OpenCV projection too.
    projected, _ = cv2.projectPoints(cached.assignment_object_points, cached.rvec, cached.tvec,
        cached.camera_matrix, cached.dist_coeffs)
    np.testing.assert_allclose(projected.reshape(-1, 2), cached.assignment_projected_points, atol=1e-5)


def test_source_search_exposes_same_selection_and_keeps_manual_default():
    parser = argparse.ArgumentParser();search.add_nine_class_input_arguments(parser)
    args = parser.parse_args(['--source', 'video.mp4'])
    assert args.pose_method == 'manual'
    assert renderer.build_parser().parse_args(['--source', '0', '--initial-scramble', '']).pose_method == 'manual'
    args = parser.parse_args(['--source', 'video.mp4', '--pose-method', 'staged',
                             '--staged-model', 'learned.pt', '--inference-device', '0'])
    assert staged.pose_checkpoint_path(args).name == 'learned.pt'
    assert args.inference_device == '0'


def test_sparse_staged_search_observation_does_not_trigger_manual_angle_sweep(monkeypatch):
    value, _ = prediction()
    candidate = staged.candidate_from_prediction(renderer, value, (0, 0, 256, 256), (256, 256), options())
    def forbidden(*_a, **_kw):
        pytest.fail('Sparse staged observations must not silently fall back to manual fitting')
    monkeypatch.setattr(renderer, '_build_turn_geometry_frame', forbidden)
    result = search.prepare_search_record(renderer,
        SimpleNamespace(pose_candidate=candidate, frame_index=0), {}, None, torch.device('cpu'))
    assert result['has_pose']
    assert result['joint_pose']['pose_method'] == 'staged'
    assert result['search']['status'] == 'unavailable_geometry'
