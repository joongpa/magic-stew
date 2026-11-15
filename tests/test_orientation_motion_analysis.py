"""Physical color-frame forecasts must remain causal and retain quarter turns."""
import numpy as np
from scipy.spatial.transform import Rotation

from scripts.analyze_orientation_motion import (
    causal_features, prediction, rotation_error, rotation_mean, symmetry_align, selected_rotation,
)


def test_physical_quarter_turn_is_not_removed_by_analysis():
    quarter = Rotation.from_euler('z', 90., degrees=True).as_matrix()
    assert abs(float(rotation_error(np.eye(3), quarter))-90.) < 1e-8
    # Symmetry reduction exists only as a separately labeled diagnostic.
    aligned, _ = symmetry_align(quarter, np.eye(3))
    assert float(rotation_error(np.eye(3), aligned)) < 1e-8


def test_constant_velocity_forecast_is_exact_for_constant_rotation():
    matrices = Rotation.from_euler('z', (np.arange(12)*7.)[:, None], degrees=True).as_matrix()
    case = dict(times=np.arange(12)/30., rotations=matrices)
    features = causal_features(case)
    actual = prediction(features, damping=1., reversion=0.)
    np.testing.assert_allclose(rotation_error(actual, features['actual']), 0., atol=1e-10)


def test_orientation_forecast_features_do_not_read_future_rotations():
    matrices = Rotation.from_euler('xyz', np.arange(60).reshape(20, 3), degrees=True).as_matrix()
    case = dict(times=np.arange(20)/30., rotations=matrices)
    expected = causal_features(case)
    poisoned = matrices.copy()
    poisoned[11:] = Rotation.from_euler('y', 150., degrees=True).as_matrix()
    actual = causal_features(dict(case, rotations=poisoned))
    early = expected['index'] <= 10
    for key in ('rotation', 'velocity', 'smooth_velocity', 'local_return', 'initial_return', 'quarter_return'):
        np.testing.assert_allclose(actual[key][early], expected[key][early], atol=1e-12)


def test_rotation_mean_stays_proper_across_rotation_vector_wrap():
    matrices = Rotation.from_euler('z', [[179.], [-179.]], degrees=True).as_matrix()
    mean = rotation_mean(matrices)
    assert abs(np.linalg.det(mean)-1.) < 1e-10
    np.testing.assert_allclose(rotation_error(mean, matrices), [1., 1.], atol=1e-10)


def test_final_selected_pose_overrides_bank_pose_and_applies_its_axis_correction():
    option = dict(pose_rvec=[[0.], [0.], [0.]])
    step_rotation = Rotation.from_euler('z', 90., degrees=True)
    correction = Rotation.from_euler('y', 90., degrees=True).as_matrix()
    step = dict(pose_rvec=step_rotation.as_rotvec().reshape(3, 1).tolist(),
                orientation_symmetry=correction.tolist())
    actual = selected_rotation(option, step)
    np.testing.assert_allclose(actual, step_rotation.as_matrix() @ correction.T, atol=1e-12)
    assert float(rotation_error(np.eye(3), actual)) > 90.
