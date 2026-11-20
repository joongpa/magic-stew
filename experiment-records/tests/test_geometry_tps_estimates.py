"""Geometry TPS experiment units, threshold semantics and clock regressions."""
import pytest
from scripts import analyze_geometry_tps as tps
from scripts import evaluate_geometry_tps_limits as limits


def frame(angles, axis=0, flagged=False):
    return {"bank": [{"axis": axis, "angles": angles, "flat_fallback": False, "layer_inliers": 3}],
            "geometry_flagged": flagged}


@pytest.mark.parametrize("angles,expected", [
    ([0, 0, 0], 0), ([6, 0, -6], 0), ([6.001, 0, 0], 1),
    ([-6.001, 0, 0], 1), ([16, 10, 10], 0), ([16.001, 10, 10], 1),
    ([84, 0, 0], 0), ([83.999, 0, 0], 1), ([20, 0, 20], 2),
])
def test_strict_six_degree_threshold_uses_middle_gauge_and_nearest_alignment(angles, expected):
    assert tps.active_layers(frame(angles)["bank"][0], 6) == expected


def test_fractional_windows_use_actual_elapsed_time():
    integral = tps.Integral([0., .02, .06], [1., 1.])
    assert integral.rate(0., .06) == pytest.approx(2. / .06)
    assert integral.rate(.01, .04) == pytest.approx(1. / .03)
    assert integral.rate(.02, .06) == pytest.approx(25.)


def test_phase_changes_track_slowdown_while_per_frame_activity_does_not():
    frames = [frame([20, 0, 0]), frame([-20, 0, 0]), frame([0, 0, 0])]
    normal = tps.geometry_features(frames, [1 / 30] * 3)
    repeated = tps.geometry_features([f for f in frames for _ in range(2)], [1 / 30] * 6)
    assert sum(normal["primary_snapped6_travel"]) / .1 == pytest.approx(20.)
    assert sum(repeated["primary_snapped6_travel"]) / .2 == pytest.approx(10.)
    assert sum(normal["primary_gt6"]) / .1 == pytest.approx(sum(repeated["primary_gt6"]) / .2)


def test_missing_geometry_fallback_is_a_rate_not_fifty_four_turns_per_frame():
    values = tps.geometry_features([{"bank": [], "geometry_flagged": True}], [.04])
    assert values["primary_gt6_unknown54"][0] / .04 == pytest.approx(54.)


def test_correlations_do_not_reward_constant_overestimates():
    assert tps.pearson([30, 30, 30], [5, 10, 15]) is None
    assert tps.pearson([7, 12, 17], [5, 10, 15]) == pytest.approx(1.)


def test_high_window_coverage_can_still_cut_every_saved_path():
    rows = [{"case": "solve-a", "cut": i == 99} for i in range(100)]
    summary = limits.gate_summary(rows, "cut")
    assert summary["violating_windows"] == 1
    assert summary["saved_capture_paths_preserved"] == 0
    assert summary["captures_with_a_saved_path_cut"] == ["solve-a"]


def test_burst_allowance_is_in_turns_and_respects_the_interval_ceiling():
    assert limits.burst_cap(10., .25, 1.) == pytest.approx(18.5)
    assert limits.burst_cap(10., .5, 1.) == pytest.approx(16.5)
    assert limits.burst_cap(40., .25, 3.) == 54.
