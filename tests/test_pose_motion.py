import numpy as np

from live_recognition.pose_motion import PoseMotionFilter, PoseMotionFilterConfig


def test_filter_initializes_without_changing_pose():
    motion_filter = PoseMotionFilter(
        PoseMotionFilterConfig(dt_seconds=1.0)
    )
    rvec = np.asarray([[0.1], [-0.2], [0.3]], dtype=np.float64)
    tvec = np.asarray([[4.0], [-3.0], [180.0]], dtype=np.float64)

    filtered_rvec, filtered_tvec, diagnostics = motion_filter.update(rvec, tvec)

    np.testing.assert_allclose(filtered_rvec, rvec)
    np.testing.assert_allclose(filtered_tvec, tvec)
    assert diagnostics.initialized is False


def test_stationary_measurements_converge_without_persistent_motion():
    motion_filter = PoseMotionFilter(
        PoseMotionFilterConfig(
            dt_seconds=1.0,
            position_alpha=0.4,
            position_beta=0.1,
            position_gamma=0.02,
            max_acceleration_units_s2=10.0,
            max_jerk_units_s3=20.0,
        )
    )
    filtered_positions = []
    for _ in range(12):
        _rvec, filtered_tvec, diagnostics = motion_filter.update(
            np.zeros((3, 1), dtype=np.float64),
            np.asarray([[5.0], [-2.0], [200.0]], dtype=np.float64),
        )
        filtered_positions.append(filtered_tvec.reshape(3))

    np.testing.assert_allclose(filtered_positions[-1], [5.0, -2.0, 200.0], atol=0.5)
    assert diagnostics.position_acceleration_norm <= 10.0 + 1e-9


def test_acceleration_and_jerk_limits_are_respected():
    motion_filter = PoseMotionFilter(
        PoseMotionFilterConfig(
            dt_seconds=1.0,
            position_alpha=0.5,
            position_beta=0.5,
            position_gamma=0.5,
            max_acceleration_units_s2=2.0,
            max_jerk_units_s3=3.0,
        )
    )
    motion_filter.update(np.zeros((3, 1)), np.zeros((3, 1)))
    _rvec, _tvec, diagnostics = motion_filter.update(
        np.zeros((3, 1)),
        np.asarray([[100.0], [0.0], [0.0]]),
    )

    assert diagnostics.position_acceleration_norm <= 2.0 + 1e-9
    assert diagnostics.position_jerk_norm <= 3.0 + 1e-9


def test_orientation_filter_returns_valid_rotation_vector():
    motion_filter = PoseMotionFilter(
        PoseMotionFilterConfig(
            dt_seconds=1.0 / 30.0,
            max_angular_acceleration_rad_s2=2.0,
            max_angular_jerk_rad_s3=6.0,
        )
    )
    motion_filter.update(np.zeros((3, 1)), np.asarray([[0.0], [0.0], [180.0]]))
    rvec, _tvec, diagnostics = motion_filter.update(
        np.asarray([[0.8], [-0.4], [0.2]]),
        np.asarray([[0.0], [0.0], [180.0]]),
    )

    assert np.isfinite(rvec).all()
    assert diagnostics.angular_acceleration_norm <= 2.0 + 1e-9
    assert diagnostics.angular_jerk_norm <= 6.0 + 1e-9


def test_accept_measurement_returns_current_pose_without_blending():
    motion_filter = PoseMotionFilter(
        PoseMotionFilterConfig(dt_seconds=1.0, max_acceleration_units_s2=1.0)
    )
    first_rvec = np.zeros((3, 1), dtype=np.float64)
    first_tvec = np.asarray([[0.0], [0.0], [200.0]], dtype=np.float64)
    motion_filter.accept_measurement(first_rvec, first_tvec)

    measured_rvec = np.asarray([[0.0], [0.0], [0.8]], dtype=np.float64)
    measured_tvec = np.asarray([[40.0], [0.0], [200.0]], dtype=np.float64)
    accepted_rvec, accepted_tvec, _diagnostics = motion_filter.accept_measurement(
        measured_rvec,
        measured_tvec,
    )

    np.testing.assert_allclose(accepted_rvec, measured_rvec)
    np.testing.assert_allclose(accepted_tvec, measured_tvec)
    predicted = motion_filter.predict_pose()
    assert predicted is not None
    assert not np.allclose(predicted[0], measured_rvec)
