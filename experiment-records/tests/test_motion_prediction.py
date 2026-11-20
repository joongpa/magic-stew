from types import SimpleNamespace

import cv2
import numpy as np

from scripts.independent_keypoint_motion_prediction import MotionState, MotionEnvelope


def pose(degrees, position=(0., 0., 600.), angles=(0., 0., 0.), axis=0):
    return SimpleNamespace(candidate=SimpleNamespace(rvec=np.array([0., 0., np.deg2rad(degrees)]),
        tvec=np.asarray(position, float)), angles=angles, axis=axis)


def test_continuous_rotation_prediction_crosses_pi_without_snapping():
    model = MotionState(pose(175.), 0.)
    model.observe(pose(179.), 1./30., reliable=True)
    prediction = model.predict(2./30.)
    expected = cv2.Rodrigues(np.array([0., 0., np.deg2rad(179.5)]))[0]
    np.testing.assert_allclose(cv2.Rodrigues(prediction.center[:3])[0], expected, atol=1e-12)


def test_prediction_bounds_widen_after_unreliable_motion_and_copies_are_independent():
    model = MotionState(pose(0.), 0.)
    for i in range(1, 20):
        model.observe(pose(0.), i/30., reliable=True)
    radius = model.predict(20./30.).radius.copy()
    other = model.copy()
    other.observe(pose(50., position=(10., 10., 630.)), 20./30., reliable=False)
    assert np.all(other.predict(21./30.).radius[:6] > radius[:6])
    assert model.time == 19./30.
    other.history[0][1][0] = 100.
    assert model.history[0][1][0] == 0.


def test_motion_prior_jacobian_matches_finite_differences_near_pi():
    center = np.r_[0., 0., 3.1, 0., 0., 600., 44., -44.]
    envelope = MotionEnvelope(center, np.array([.2, .2, .2, 3., 3., 10., 15., 15.]), np.ones(8))
    parameters = center+np.array([.04, -.02, .08, 1., -1., 2., 2., -2.])
    _, analytic = envelope.residual_jacobian(parameters)
    delta = 1e-6
    numeric = np.column_stack([(envelope.residual_jacobian(parameters+v*delta)[0]-
        envelope.residual_jacobian(parameters-v*delta)[0])/(2*delta) for v in np.eye(8)])
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-7)


def test_search_bounds_use_geodesic_rotation_and_wrapped_layer_angles():
    center = np.r_[0., 0., 3.1, 0., 0., 600., 44., -44.]
    radius = np.array([.1, .1, .1, 3., 3., 10., 5., 5.])
    envelope = MotionEnvelope(center, radius, radius)
    x = np.r_[0., 0., -3., 50., -50., 650., -44., 44.]
    constrained = envelope.constrain(x)
    residual, _ = envelope.residual_jacobian(constrained)
    assert np.linalg.norm(residual[:3]) <= 1.+1e-10
    assert np.max(np.abs(residual)) <= 1.+1e-10
    np.testing.assert_allclose(constrained[6:], [46., -46.])


def test_elapsed_time_scales_prediction_instead_of_assuming_source_fps():
    model = MotionState(pose(0.), 0., damping=1.)
    model.observe(pose(10.), .02, reliable=True)
    short = cv2.Rodrigues(model.predict(.04).center[:3])[0]
    long = cv2.Rodrigues(model.predict(.06).center[:3])[0]
    assert np.linalg.norm(cv2.Rodrigues(long @ short.T)[0]) > .01
