"""Causal core dynamics and independent five-hypothesis tracking state."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import independent_keypoint_physics_tracker as physics
from scripts import independent_keypoint_pose_geometry as geometry
from scripts import render_yolo11n_independent_keypoints as renderer
from scripts.analyze_core_motion import causal_features


def warm_quarter_prior(prior):
    for i in range(1, 5):
        prior.observe(prior.anchor.copy(), .05*i)
    assert prior.ready
    return prior


def test_core_reverts_to_working_center_without_overshooting():
    model = physics.CoreReturnModel(reversion=8.)
    model.observe(0., [0., 0., 600.])
    model.observe(.02, [0., 0., 600.])
    model.observe(.04, [6., -6., 620.])
    prediction = model.predict(.07)
    assert 0. < prediction[0] < 6.
    assert -6. < prediction[1] < 0.
    assert 600. < prediction[2] < 620.
    assert len(model.history) == 3  # Predict does not accept its own result.


def test_core_anchor_can_relocate_after_pickup():
    model = physics.CoreReturnModel(reversion=8., window=.2)
    model.observe(0., [0., 0., 600.])
    for i in range(1, 21):
        model.observe(.05*i, [20., -60., 650.])
    np.testing.assert_allclose(model.predict(1.05), [20., -60., 650.])


def test_each_geometry_has_an_independent_core_history():
    first = physics.CoreReturnModel()
    first.observe(0., [0., 0., 600.])
    second = first.copy()
    second.observe(.03, [20., 0., 610.])
    assert len(first.history) == 1
    assert len(second.history) == 2
    second.history[0][1][0] = 90.
    assert first.history[0][1][0] == 0.


def test_quarter_turn_seam_is_small_relative_motion():
    np.testing.assert_allclose(physics.wrapped_angle_delta([-44., 44.], [44., -44.]), [2., -2.])


def test_analysis_features_do_not_use_future_core_positions():
    times = np.arange(20)/30.
    positions = np.column_stack((np.arange(20), np.zeros(20), np.full(20, 600.)))
    original = causal_features(times, positions)
    changed = positions.copy()
    changed[11:] = 10000.
    actual = causal_features(times, changed)
    for first, second in zip(original, actual, strict=True):
        if first['index'] > 10:
            continue
        for key in ('position', 'velocity', 'smooth_velocity', 'local_center', 'initial_center'):
            np.testing.assert_allclose(first[key], second[key])


def test_fast_solver_reduces_projection_residual():
    seed = np.r_[np.zeros(3), 0., 0., 600., 0., 0.]
    target = seed+np.array([.03, -.02, .01, 1., -1., 2., 2., -2.])
    model = SimpleNamespace(m=SimpleNamespace(CUBE_SIZE_MM=55.),
        evaluate=lambda x: (x-target, np.eye(8)))
    solved = physics.local_solve(model, seed, seed[3:6], iterations=4)
    assert np.linalg.norm(solved-target) < np.linalg.norm(seed-target)


@pytest.fixture
def bank_record():
    # Use a small existing fixture and materialize deterministic diverse seeds;
    # this fixture is not an accuracy target and uses no external run artifacts.
    from scripts import render_independent_keypoint_search as playback
    from scripts import search_independent_keypoint_full_solve as search
    data = json.loads((Path(__file__).parent/'fixtures/matty469-observations.json').read_text())
    colors = search.initial_sticker_colors(renderer,
        "U2 F2 U B2 U R2 F2 R2 D' U' L' B2 U2 F L' R U B' L' F'")
    candidate = playback.candidate_for_step(renderer, data, None,
        renderer.CubeGeometryState.identity(), colors)
    fits = []
    for axis, angles in [(0, (-30., 0.)), (0, (30., 0.)), (1, (0., -30.)),
                         (1, (0., 30.)), (2, (30., -30.))]:
        x = np.r_[candidate.rvec.ravel(), candidate.tvec.ravel(), angles]
        fits.append(geometry._materialize(renderer, candidate, axis, x))
    return candidate, fits


@pytest.mark.parametrize('quarter_turn_prior', [False, True, 'motion', 'slab'])
def test_tracker_ignores_current_saved_pose_and_keeps_five_distinct_fits(bank_record, quarter_turn_prior):
    import torch
    torch.set_num_threads(2)
    observations, fits = bank_record
    def tracker():
        if quarter_turn_prior == 'slab':
            from scripts.independent_keypoint_slab_search import SlabHierarchyTracker
            return SlabHierarchyTracker(renderer, iterations=1, beam=5)
        if quarter_turn_prior == 'motion':
            from scripts.independent_keypoint_motion_prediction import PredictivePoseTracker
            return PredictivePoseTracker(renderer, iterations=1)
        return physics.PhysicsPoseTracker(renderer, iterations=1, quarter_turn_prior=quarter_turn_prior)
    first, second = tracker(), tracker()
    first.initialize(fits, 0.)
    second.initialize(fits, 0.)
    assert len({id(b.core) for b in first.branches}) == 5
    if quarter_turn_prior == 'slab':
        # No old articulation angle or velocity may enter the static search
        # or restrict the new frame's separate articulation sweep.
        for branch in second.branches:
            branch.fit = replace(branch.fit, angles=(43., 0., -39.))
            branch.motion.angles[:] = [43., -39.]
            branch.motion.turn_velocity[:] = [2000., -2000.]
    poisoned = replace(observations, rvec=np.full((3, 1), np.nan),
        tvec=np.full((3, 1), np.nan), assignment_projected_points=np.full((22, 2), np.nan))
    expected = first.update(observations, 1./30.)
    actual = second.update(poisoned, 1./30.)
    assert len(expected) == len(actual) == 5
    assert len(geometry.diverse_shortlist(renderer, actual)) == 5
    for a, b in zip(expected, actual, strict=True):
        np.testing.assert_allclose(a.candidate.rvec, b.candidate.rvec)
        np.testing.assert_allclose(a.candidate.tvec, b.candidate.tvec)
        assert a.score == pytest.approx(b.score)
    assert len({id(b.core) for b in second.branches}) == 5
    assert all(len(b.core.history) == 2 for b in second.branches)
    if quarter_turn_prior is True:
        assert len({id(b.quarter) for b in second.branches}) == 5
    if quarter_turn_prior in ('motion', 'slab'):
        assert len({id(b.motion) for b in second.branches}) == 5


def test_quarter_prior_keeps_distinct_physical_modes():
    prior = warm_quarter_prior(physics.QuarterTurnPrior(np.eye(3), renderer.CUBE_SYMMETRY_MATRICES, 0.))
    first, first_mode = prior.nearest(np.eye(3))
    quarter = physics.cv2.Rodrigues(np.array([0., 0., np.pi/2.]))[0]
    second, second_mode = prior.nearest(quarter)
    assert first_mode != second_mode
    np.testing.assert_allclose(first, np.eye(3))
    np.testing.assert_allclose(second, quarter, atol=1e-12)
    for rotation in (np.eye(3), quarter):
        residual, _ = prior.residual_jacobian(physics.cv2.Rodrigues(rotation)[0].ravel())
        np.testing.assert_allclose(residual, 0., atol=1e-12)


def test_quarter_prior_analytic_derivative_matches_finite_difference():
    anchor = physics.cv2.Rodrigues(np.array([.5, -.3, .8]))[0]
    prior = warm_quarter_prior(physics.QuarterTurnPrior(anchor, renderer.CUBE_SYMMETRY_MATRICES, 0.))
    vector = np.array([.7, -.1, 1.])
    _, analytic = prior.residual_jacobian(vector)
    assert np.linalg.norm(analytic) > 1.
    step = 1e-6
    numeric = np.column_stack([(prior.residual_jacobian(vector+axis*step)[0]-
        prior.residual_jacobian(vector-axis*step)[0])/(2*step) for axis in np.eye(3)])
    np.testing.assert_allclose(analytic, numeric, atol=1e-7, rtol=1e-6)


def test_quarter_prior_is_soft_and_hypothesis_history_is_independent():
    prior = warm_quarter_prior(physics.QuarterTurnPrior(np.eye(3), renderer.CUBE_SYMMETRY_MATRICES, 0.))
    seed = np.r_[0., 0., np.deg2rad(20.), 0., 0., 600., 0., 0.]
    model = SimpleNamespace(m=SimpleNamespace(CUBE_SIZE_MM=55.),
        evaluate=lambda x: (x-seed, np.eye(8)))
    fitted = physics.local_solve(model, seed, seed[3:6], iterations=2, quarter=prior)
    assert 0. < fitted[2] < seed[2]
    clone = prior.copy()
    clone.observe(physics.cv2.Rodrigues(seed[:3])[0], .25)
    np.testing.assert_array_equal(prior.anchor, np.eye(3))
    assert prior.count == 5
    assert clone.count == 6


def test_quarter_prior_waits_for_a_reliable_history_before_constraining_fits():
    prior = physics.QuarterTurnPrior(np.eye(3), renderer.CUBE_SYMMETRY_MATRICES, 0.)
    for i in range(1, 8):
        prior.observe(np.eye(3), .03*i, reliable=False)
    assert not prior.ready
    residual, jacobian = prior.residual_jacobian(np.array([.2, .1, .3]))
    np.testing.assert_array_equal(residual, np.zeros(3))
    np.testing.assert_array_equal(jacobian, np.zeros((3, 3)))
