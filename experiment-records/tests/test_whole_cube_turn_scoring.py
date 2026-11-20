"""Whole-cube metrics must survive missing support on the proposed moving layer."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "whole_cube_renderer", Path(__file__).resolve().parents[1] /
    "scripts/render_yolo11n_independent_keypoints.py")
m = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = m
SPEC.loader.exec_module(m)


@pytest.fixture(scope="module")
def torch():
    module = pytest.importorskip("torch")
    module.set_num_threads(2)
    return module


def frame_231():
    data = json.loads((Path(__file__).parent /
                       "fixtures/xuanyi_frame_231_whole_cube_scoring.json").read_text())
    p = data['pose']
    mask = np.asarray(p['assignment_fixed_mask'], dtype=bool)
    observed = np.asarray(p['assignment_image_points'])
    projected = np.asarray(p['assignment_projected_points'])
    return m.StickerPnPCandidate(
        sticker_type=p['sticker_type'], sticker_average_confidence=1.,
        image_points=observed[mask], object_points=np.asarray(data['seed_object_points']),
        projected_sticker_points=projected[mask], reprojection_error=0.,
        camera_matrix=np.asarray(p['camera_matrix']), dist_coeffs=np.asarray(p['dist_coeffs']),
        rvec=np.asarray(p['rvec']), tvec=np.asarray(p['tvec']), image_size=tuple(p['image_size']),
        assignment_image_points=observed, assignment_projected_points=projected,
        assignment_object_points=np.asarray(data['assignment_object_points']),
        assignment_observed_class_ids=np.asarray(p['assignment_observed_class_ids']),
        assignment_projected_class_ids=np.asarray(p['assignment_observed_class_ids']),
        assignment_fixed_mask=mask,
        sticker_mask_centers=tuple(np.asarray(v) for v in p['sticker_mask_centers']),
        sticker_mask_center_uncertainties=tuple(p['sticker_mask_center_uncertainties']),
        sticker_color_names=tuple(p['sticker_color_names']))


def test_global_errors_do_not_depend_on_proposed_layer_and_allow_outliers(torch):
    distances = torch.tensor([[1., 2., 30., float('inf')]]).repeat(2, 1)
    objects = torch.zeros((2, 4, 3))
    objects[:, :, 0] = -m.CUBE_HALF_SIZE_MM
    mean, robust, p90, inliers, layer_inliers, valid = m._gpu_keypoint_fit_metrics(
        torch, distances, objects, torch.zeros((2, 4), dtype=torch.long),
        torch.zeros(2, dtype=torch.long), torch.tensor([1., -1.]), torch.ones(2),
        torch.zeros(4, dtype=torch.long), 100., 2.)
    assert valid.tolist() == [True, True]
    assert mean.tolist() == pytest.approx([.11, .11])
    np.testing.assert_allclose(robust[0], robust[1])
    np.testing.assert_allclose(p90[0], p90[1])
    assert inliers.tolist() == [2, 2]
    assert layer_inliers.tolist() == [0, 2]


def test_class_constraints_apply_to_stationary_part_of_cube_too(torch):
    objects = torch.tensor([[[-27.5, 0., 0.], [27.5, 0., 0.]]])
    mean, robust, p90, inliers, _layer, valid = m._gpu_keypoint_fit_metrics(
        torch, torch.tensor([[1., 1.]]), objects, torch.tensor([[1, 0]]),
        torch.tensor([0]), torch.tensor([1.]), torch.tensor([1.]), torch.tensor([0, 0]), 100., 2.)
    assert not valid.item()
    assert all(np.isinf(value.item()) for value in (mean, robust, p90))
    assert inliers.item() == 0


@pytest.mark.parametrize("gate", [False, True])
def test_sticker_support_cannot_zero_an_invalid_global_landmark_metric(torch, monkeypatch, gate):
    def invalid(_torch, distances, *_args):
        count = distances.shape[0]
        error = distances.new_full((count,), float('inf'))
        zero = torch.zeros(count, dtype=torch.long)
        return error, error, error, zero, zero, torch.zeros(count, dtype=torch.bool)
    monkeypatch.setattr(m, '_gpu_keypoint_fit_metrics', invalid)
    candidate = frame_231()
    options = m._score_turn_configurations_on_gpu(
        candidate, .05*m.projected_cube_diameter(candidate), 'cpu',
        angle_values=np.asarray([27.]), enforce_layer_observation_gate=False,
        enforce_validity_gate=gate)
    assert options and any(o['sticker_layer_match_count'] > 0 for o in options)
    assert all(np.isinf(o['keypoint_error']) and np.isinf(o['combined_mean_error']) for o in options)
    assert all(np.isinf(o['robust_error']) and np.isinf(o['p90_error']) for o in options)


def test_frame_231_preserves_all_cube_landmark_error_without_moving_layer_inliers(torch):
    candidate = frame_231()
    limit = .05*m.projected_cube_diameter(candidate)
    options = m._score_turn_configurations_on_gpu(
        candidate, limit, 'cpu', angle_values=np.asarray([27.]),
        enforce_layer_observation_gate=False, enforce_validity_gate=False)
    trial = next(o for o in options if o['layer_rotations'] and
                 (o['layer_axis'], o['layer_side'], o['layer_count']) == (0, -1, 2))
    distances = np.linalg.norm(candidate.assignment_image_points -
                               trial['assignment_projected_points'], axis=1)
    assert len(distances) == 17
    assert trial['keypoint_error'] == pytest.approx(float(distances.mean()/m.projected_cube_diameter(candidate)))
    assert trial['keypoint_error'] > 0.
    assert trial['keypoint_inlier_count'] == np.count_nonzero(distances <= limit) > 0
    assert trial['keypoint_layer_inlier_count'] == 0
    static = [o['keypoint_error'] for o in options if o['is_baseline']]
    assert len(static) == 12 and min(static) > 0.
    np.testing.assert_allclose(static, static[0])
