from dataclasses import replace

import numpy as np
import pytest
import torch

from test_independent_keypoint_pose_geometry import observation
from scripts import render_yolo11n_independent_keypoints as renderer
from scripts.independent_keypoint_gpu_sweep import parallel_turn_sweep, project


FLOAT_FIELDS = ('keypoint_error', 'sticker_error', 'combined_mean_error', 'robust_error', 'p90_error')
COUNT_FIELDS = ('inlier_count', 'layer_inlier_count', 'keypoint_inlier_count',
                'keypoint_layer_inlier_count', 'sticker_match_count')


@pytest.mark.parametrize('axis', range(3))
@pytest.mark.parametrize('middle_frame', [False, True])
def test_tensor_port_matches_original_exhaustive_cpu_sweep(observation, axis, middle_frame):
    candidate = observation[0]
    limit = renderer.PNP_FINAL_ASSIGNMENT_ERROR_FRACTION*renderer.projected_cube_diameter(candidate)
    expected, expected_nearest, expected_valid = renderer.parallel_turn_frame_geometry(candidate, limit, axis, middle_frame=middle_frame)
    actual, nearest, valid = parallel_turn_sweep(renderer, candidate, limit, axis, device='cpu',
                                                dtype=torch.float64, middle_frame=middle_frame)
    assert len(actual) == len(expected) == 342
    for a, e in zip(actual, expected, strict=True):
        assert a['layer_rotations'] == e['layer_rotations']
        for field in FLOAT_FIELDS:
            np.testing.assert_allclose(a[field], e[field], atol=1e-9, rtol=1e-10)
        for field in COUNT_FIELDS:
            assert a[field] == e[field], (axis, field, a['layer_rotations'])
        assert a['layer_support'] == e['layer_support']
    np.testing.assert_array_equal(valid, expected_valid)
    np.testing.assert_array_equal(nearest[valid], expected_nearest[valid])


def test_tensor_projection_matches_opencv_with_distortion(observation):
    import cv2
    candidate = observation[0]
    objects = renderer.CUBE_LATTICE_OBJECT_POINTS_MM
    coefficients = np.array([.03, -.02, .001, -.002, .003, .0001, .0002, -.0003, .0001, -.0002, .0003, -.0001])
    expected = cv2.projectPoints(objects, candidate.rvec, candidate.tvec,
        candidate.camera_matrix, coefficients)[0].reshape(-1, 2)
    actual = project(torch.tensor(objects, dtype=torch.float64),
        torch.tensor(cv2.Rodrigues(candidate.rvec)[0]), torch.tensor(candidate.tvec.ravel()),
        torch.tensor(candidate.camera_matrix), torch.tensor(coefficients)).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-9)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS device unavailable in this process')
def test_mps_evaluates_complete_grid_with_close_geometry_scores(observation):
    candidate = observation[0]
    limit = renderer.PNP_FINAL_ASSIGNMENT_ERROR_FRACTION*renderer.projected_cube_diameter(candidate)
    for axis in range(3):
        expected = renderer.parallel_turn_frame_geometry(candidate, limit, axis, middle_frame=True)[0]
        actual = parallel_turn_sweep(renderer, candidate, limit, axis, device='mps', middle_frame=True)[0]
        assert len(actual) == 342
        np.testing.assert_allclose([o['keypoint_error'] for o in actual],
            [o['keypoint_error'] for o in expected], atol=2e-5, rtol=2e-4)
        np.testing.assert_allclose([o['sticker_error'] for o in actual],
            [o['sticker_error'] for o in expected], atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_all_axes_batch_matches_single_axis(observation, device):
    from scripts.independent_keypoint_gpu_sweep import parallel_turn_sweeps
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS device unavailable in this process')
    candidate = observation[0]
    limit = renderer.PNP_FINAL_ASSIGNMENT_ERROR_FRACTION*renderer.projected_cube_diameter(candidate)
    actual = parallel_turn_sweeps(renderer, candidate, limit, device=device, middle_frame=True)
    for axis, (options, nearest, valid) in enumerate(actual):
        expected, old_nearest, old_valid = parallel_turn_sweep(renderer, candidate, limit, axis,
                                                             device=device, middle_frame=True)
        for a, e in zip(options, expected, strict=True):
            for field in FLOAT_FIELDS:
                np.testing.assert_allclose(a[field], e[field], atol=2e-5 if device == 'mps' else 1e-9,
                                           rtol=2e-4 if device == 'mps' else 1e-10)
            for field in COUNT_FIELDS:
                assert a[field] == e[field]
        np.testing.assert_array_equal(valid, old_valid)
        np.testing.assert_array_equal(nearest[valid], old_nearest[valid])


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_grazing_sticker_is_not_hidden_by_its_own_slab(device):
    from scripts.independent_keypoint_gpu_sweep import static_tensors, local_slab_entry
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS device unavailable in this process')
    # Nahm frame 716: camera nearly tangent to a sticker on the 15-degree slab.
    # Rotating the long world-space ray in float32 put its entry below 1-1e-5.
    camera = torch.tensor([-33.98718024398602, 129.05482721552994, -235.028310311343],
                          dtype=torch.float32, device=device)
    data = static_tensors(renderer, 1, device, torch.float32)
    origin = camera @ data['matrices'][4]
    endpoint = data['local_centers'][4, 4, 33]
    entry = local_slab_entry(origin, endpoint-origin, data['lower'][0], data['upper'][0])
    assert entry.item() >= 1.-1e-5
