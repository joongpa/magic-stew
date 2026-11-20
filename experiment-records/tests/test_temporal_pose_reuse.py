"""Evidence isolation and optimizer bounds for the temporal-fitting experiment."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import benchmark_temporal_pose_reuse as temporal
from scripts import independent_keypoint_pose_geometry as geometry
from scripts import render_independent_keypoint_search as playback
from scripts import render_yolo11n_independent_keypoints as renderer
from scripts import search_independent_keypoint_full_solve as search


@pytest.fixture
def observation():
    record = json.loads((Path(__file__).parent/'fixtures/matty469-observations.json').read_text())
    colors = search.initial_sticker_colors(renderer,
        "U2 F2 U B2 U R2 F2 R2 D' U' L' B2 U2 F L' R U B' L' F'")
    return playback.candidate_for_step(renderer, record, None,
                                       renderer.CubeGeometryState.identity(), colors)


def test_local_refinement_cannot_read_current_reference_pose(observation):
    import torch

    torch.set_num_threads(2)
    previous = replace(observation,
        layer_rotations=renderer.parallel_layer_rotations(1, (-23., 0., 4.)))
    expected = temporal.reuse_previous(observation, previous, iterations=3)
    poisoned = replace(observation, rvec=np.full((3, 1), np.nan),
        tvec=np.full((3, 1), np.nan),
        layer_rotations=renderer.parallel_layer_rotations(2, (45., 0., -45.)),
        assignment_object_points=np.full((22, 3), np.nan),
        assignment_projected_points=np.full((22, 2), np.nan))
    actual = temporal.reuse_previous(poisoned, previous, iterations=3)
    assert len(actual) == len(expected) == 3
    for first, second in zip(expected, actual, strict=True):
        assert first.axis == second.axis
        assert first.score == pytest.approx(second.score)
        np.testing.assert_allclose(first.candidate.rvec, second.candidate.rvec)
        np.testing.assert_allclose(first.candidate.tvec, second.candidate.tvec)
        np.testing.assert_array_equal(second.candidate.assignment_image_points,
                                      observation.assignment_image_points)


def test_refinement_obeys_local_pose_and_angle_windows():
    seed = np.r_[np.zeros(5), 100., 0., 0.]
    target = seed + 20.
    model = SimpleNamespace(m=SimpleNamespace(CUBE_SIZE_MM=55.),
        evaluate=lambda x: (x-target, np.eye(8)))
    radius = np.array([.01, .02, .03, 1., 2., 3.])
    result = geometry._refine(model, seed, pose_radius=radius, angle_radius=2.)
    assert np.all(np.abs(result[:6]-seed[:6]) <= radius+1e-9)
    assert np.all(np.abs(result[6:]) <= 2.+1e-9)
    assert np.max(result[6:]) > 1.9


@pytest.mark.parametrize('radius', [[1.]*5, [0.]*6, [float('nan')]*6])
def test_invalid_pose_windows_are_rejected(radius):
    with pytest.raises(ValueError, match='pose_radius'):
        geometry._refine(None, np.r_[np.zeros(5), 100., 0., 0.], pose_radius=radius)


def test_bad_quality_expands_the_numerical_search_window():
    narrow_pose, narrow_angle = temporal.local_radii(.005)
    wide_pose, wide_angle = temporal.local_radii(.03)
    assert np.all(wide_pose > narrow_pose)
    assert wide_angle > narrow_angle


def test_dynamic_gate_checks_stickers_as_well_as_displayed_reprojection(monkeypatch):
    monkeypatch.setattr(renderer, 'pose_reprojection_fit_metric', lambda candidate: .005)
    fit = SimpleNamespace(candidate=object(), score=.5)
    assert not temporal.acceptable([fit], .025, .15)
    fit.score = .08
    assert temporal.acceptable([fit], .025, .15)
    monkeypatch.setattr(renderer, 'pose_reprojection_fit_metric', lambda candidate: .08)
    assert not temporal.acceptable([fit], .025, .15)
    assert temporal.acceptable([fit], .1, .15)


def test_exhaustive_recovery_also_discards_current_reference_pose(monkeypatch, observation):
    previous = replace(observation, rvec=np.array([[.1], [.2], [.3]]))
    current = replace(observation, rvec=np.full((3, 1), np.nan),
                      tvec=np.full((3, 1), np.nan))
    monkeypatch.setattr(temporal, 'reuse_previous', lambda *args, **kwargs: [])
    monkeypatch.setattr(temporal, 'restricted_sweep', lambda *args, **kwargs: [])

    def exhaustive(m, base, *, prior):
        assert prior is previous
        np.testing.assert_array_equal(base.rvec, previous.rvec)
        np.testing.assert_array_equal(base.tvec, previous.tvec)
        np.testing.assert_array_equal(base.assignment_image_points, current.assignment_image_points)
        return []

    monkeypatch.setattr(geometry, 'fit_frame', exhaustive)
    fits, stage = temporal.adaptive_reuse(current, previous)
    assert fits == []
    assert stage == 'exhaustive_previous_camera'


def test_low_error_stops_before_expanding_search(monkeypatch, observation):
    fit = SimpleNamespace(candidate=observation, score=.04)
    monkeypatch.setattr(renderer, 'pose_reprojection_fit_metric', lambda candidate: .005)
    calls = []

    def local(*args, **kwargs):
        calls.append(kwargs)
        return [fit]

    def forbidden(*args, **kwargs):
        raise AssertionError('A good narrow fit must not launch broader search')

    monkeypatch.setattr(temporal, 'reuse_previous', local)
    monkeypatch.setattr(temporal, 'restricted_sweep', forbidden)
    monkeypatch.setattr(geometry, 'fit_frame', forbidden)
    fits, stage = temporal.adaptive_reuse(observation, observation)
    assert fits == [fit]
    assert stage == 'narrow_local'
    assert len(calls) == 1
    assert calls[0]['angle_radius'] == 15.


def test_restricted_sweep_matches_the_same_rows_in_full_sweep(observation):
    import torch

    torch.set_num_threads(2)
    limit = renderer.PNP_FINAL_ASSIGNMENT_ERROR_FRACTION * renderer.projected_cube_diameter(observation)
    full, nearest, valid = renderer.parallel_turn_frame_geometry(observation, limit, 1)
    requested = [(-45., 0., 45.), (15., 0., -30.), (0., 0., 0.)]
    subset, sub_nearest, sub_valid = renderer.parallel_turn_frame_geometry(
        observation, limit, 1, angle_grid=requested)
    assert len(subset) == 2
    for index, option in enumerate(subset):
        angles = renderer.parallel_layer_angles(option['layer_rotations'])
        original_index = next(i for i, row in enumerate(full)
                              if renderer.parallel_layer_angles(row['layer_rotations']) == angles)
        original = full[original_index]
        for field in ('keypoint_error', 'sticker_error', 'robust_error', 'inlier_count', 'p90_error'):
            assert option[field] == pytest.approx(original[field])
        np.testing.assert_array_equal(sub_nearest[index], nearest[original_index])
        np.testing.assert_array_equal(sub_valid[index], valid[original_index])


def test_zero_only_grid_returns_no_duplicate_flat_geometry(observation):
    options, nearest, valid = renderer.parallel_turn_frame_geometry(
        observation, 10., 0, angle_grid=[[0., 0., 0.]])
    assert options == []
    assert nearest.shape == valid.shape == (0, len(observation.sticker_mask_centers))


@pytest.mark.parametrize('grid', [[], [[7.5, 0., 0.]], [[60., 0., 0.]], [[np.nan, 0., 0.]]])
def test_invalid_restricted_grids_are_rejected(observation, grid):
    with pytest.raises(ValueError, match='angle_grid'):
        renderer.parallel_turn_frame_geometry(observation, 10., 0, angle_grid=grid)
