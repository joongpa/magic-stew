"""Joint middle-frame pose fitting and candidate-specific projection caches."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import cv2

from scripts import independent_keypoint_pose_geometry as geometry
from scripts import render_yolo11n_independent_keypoints as m
from scripts import render_independent_keypoint_search as playback
from scripts import search_independent_keypoint_full_solve as search


@pytest.fixture(scope='module')
def observation():
    torch = pytest.importorskip('torch')
    torch.set_num_threads(2)
    record = json.loads((Path(__file__).parent/'fixtures/matty469-observations.json').read_text())
    colors = search.initial_sticker_colors(m,
        "U2 F2 U B2 U R2 F2 R2 D' U' L' B2 U2 F L' R U B' L' F'")
    return playback.candidate_for_step(m, record, None, m.CubeGeometryState.identity(), colors), colors


@pytest.fixture(scope='module')
def fits(observation):
    return geometry.fit_frame(m, observation[0])


def test_analytic_projection_jacobian_matches_finite_difference(observation):
    candidate, _ = observation
    model = geometry.ArticulatedProjection(m, candidate, 1)
    x = np.r_[candidate.rvec.ravel(), candidate.tvec.ravel(), -23., 4.]
    _, analytic = model.evaluate(x)
    step = 1e-6
    numeric = np.column_stack([(model.evaluate(x+np.eye(8)[i]*step)[0] -
        model.evaluate(x-np.eye(8)[i]*step)[0])/(2*step) for i in range(8)])
    np.testing.assert_allclose(analytic, numeric, atol=1e-5, rtol=1e-5)


def test_articulated_geometry_cache_does_not_cache_camera_or_stale_layer_angles(observation):
    candidate, _ = observation
    model = geometry.ArticulatedProjection(m, candidate, 1)
    x = np.r_[candidate.rvec.ravel(), candidate.tvec.ravel(), -23., 4.]
    model.evaluate(x)
    for change in (np.array([.03, -.02, .01, 2., -1., 5., 0., 0.]),
                   np.array([0., 0., 0., 0., 0., 0., 7., -9.]),
                   np.array([-.02, .01, .02, -1., 2., -3., 0., 0.])):
        x = x+change
        residual, jacobian = model.evaluate(x)
        fresh = geometry.ArticulatedProjection(m, candidate, 1)
        expected_residual, expected_jacobian = fresh.evaluate(x)
        np.testing.assert_array_equal(residual, expected_residual)
        np.testing.assert_array_equal(jacobian, expected_jacobian)
        for actual, expected in zip(model.landmark_correspondences, fresh.landmark_correspondences, strict=True):
            np.testing.assert_array_equal(actual, expected)


def test_capped_landmark_loss_has_no_background_pull_and_correct_jacobian():
    rng = np.random.default_rng(76)
    values = np.array([2., 1., 90., -60., 1., -2., -80., 70.])
    projection_jacobian = rng.normal(size=(8, 8))
    residual, analytic = geometry.capped_projection_residuals(values, projection_jacobian, 2)
    step = 1e-6
    numeric = np.column_stack([(geometry.capped_projection_residuals(
        values+projection_jacobian[:, i]*step, projection_jacobian, 2)[0] -
        geometry.capped_projection_residuals(
        values-projection_jacobian[:, i]*step, projection_jacobian, 2)[0])/(2*step)
        for i in range(8)])
    np.testing.assert_allclose(analytic, numeric, atol=1e-7)
    assert np.all(analytic[[1, 3]] == 0.)
    assert np.all(residual[[1, 3]] > 0.)  # Rejection is a capped cost, never a free match.


@pytest.fixture(scope='module')
def solved_window(observation):
    data = json.loads((Path(__file__).parent/'fixtures/matty-solved-depth-observations.json').read_text())
    colors = observation[1]
    records = [SimpleNamespace(frame_index=r['frame_index'], roi_xyxy=None,
        pose_candidate=playback.candidate_for_step(m, r, None, m.CubeGeometryState.identity(), colors))
        for r in data['records']]
    return records, data['fps']


def test_robust_depth_trend_does_not_follow_a_bad_current_depth(solved_window):
    records, fps = solved_window
    expected = geometry.depth_trend_targets(m, records, SimpleNamespace(), fps)
    damaged = [SimpleNamespace(**vars(r)) for r in records]
    index = next(i for i, r in enumerate(damaged) if r.frame_index == 574)
    pose = damaged[index].pose_candidate
    damaged[index].pose_candidate = replace(pose, tvec=pose.tvec*1.3)
    actual = geometry.depth_trend_targets(m, damaged, SimpleNamespace(), fps)
    assert abs(actual[index]-expected[index]) < 5.
    assert abs(actual[index]-pose.tvec[2, 0]) < 25.


def test_spatial_gate_rejects_mat_landmarks_and_preserves_real_d(observation, solved_window):
    real = geometry.reject_spatial_outliers(m, observation[0], observation[0].tvec[2, 0])
    assert len(real.assignment_image_points) == len(observation[0].assignment_image_points)
    record = next(r for r in solved_window[0] if r.frame_index == 573)
    pose = record.pose_candidate
    filtered = geometry.reject_spatial_outliers(m, pose, pose.tvec[2, 0])
    rejected = filtered.joint_pose_diagnostics['spatial_outlier_points']
    for index in (5, 8, 9):
        assert pose.assignment_image_points[index].tolist() in rejected


def test_recovery_requires_sparse_observations_and_depth_or_spatial_evidence(observation, solved_window):
    dense = observation[0]
    assert geometry.depth_fallback_reason(dense, dense.tvec[2, 0]+150.) is None
    sparse = next(r.pose_candidate for r in solved_window[0] if r.frame_index == 573)
    assert geometry.depth_fallback_reason(sparse, sparse.tvec[2, 0]+10.) is None
    assert geometry.depth_fallback_reason(sparse, sparse.tvec[2, 0]+10., spatial_outliers=1) is None
    reason = geometry.depth_fallback_reason(sparse, sparse.tvec[2, 0]+10., spatial_outliers=2)
    assert reason['trigger'] == 'sparse_observations_and_spatial_outliers'
    assert geometry.depth_fallback_reason(dense, dense.tvec[2, 0], spatial_outliers=2) is None
    reason = geometry.depth_fallback_reason(sparse, sparse.tvec[2, 0]+150.)
    assert reason['trigger'] == 'sparse_observations_and_depth_jump'
    assert reason['depth_jump_sigmas'] > 3.


def test_targeted_recovery_preserves_baseline_banks_and_neighbors(solved_window, observation, monkeypatch):
    records, fps = solved_window
    cache = [SimpleNamespace(**vars(r)) for r in records]
    originals = {}
    recovered = {}
    original_recovery = geometry.apply_depth_fallback

    def recover(renderer, records, *args):
        originals.update({r.frame_index: r.pose_candidate for r in records})
        original_recovery(renderer, records, *args)
        recovered.update({r.frame_index: r.pose_candidate for r in records})

    monkeypatch.setattr(geometry, 'apply_depth_fallback', recover)
    args = m.build_parser().parse_args(['--source', 'clip.mp4', '--initial-scramble', '',
                                      '--wireframe-only', '--no-display'])
    geometry.process_pose_geometry(m, cache, args, observation[1], fps)
    changed = []
    for record in cache:
        before = originals[record.frame_index]
        # Compare this stage before the independent final-output alternatives.
        after = recovered[record.frame_index]
        if after is before:
            continue
        changed.append(record.frame_index)
        assert after.layer_rotations == ()
        assert after.joint_pose_diagnostics['baseline_geometry_options'] == list(before.joint_geometry_candidates)
        assert abs(after.tvec[2, 0]-after.joint_pose_diagnostics['depth_prior_mm']) < 20.
        options = after.joint_geometry_candidates
        scorer = search.PreparedReplayScorer(m, {}, 'cpu', balance_geometry_color=True)
        ordinary = [o for o in options if not o['flat_fallback']]
        assert [scorer._geometry_score(o) for o in ordinary] == sorted(o['recovery_score'] for o in ordinary)
        assert options[-1]['flat_fallback'] and not options[-1]['layer_rotations']
    assert changed == [573, 574, 575]


def test_recovers_matty_d_assignments_without_axis_hint(observation, monkeypatch):
    pool = []
    select = geometry.assignment_shortlist

    def capture(renderer, candidates, **kwargs):
        pool.extend(candidates)
        return select(renderer, candidates, **kwargs)

    monkeypatch.setattr(geometry, 'assignment_shortlist', capture)
    fits = geometry.fit_frame(m, observation[0])
    assert len(fits) <= 6
    assert all(fit.angles[1] == 0 for fit in fits)
    # The saved successful diagnostic was .1532; the old omitted D was .8056.
    correct = [fit for fit in pool if fit.axis == 2 and -29 < fit.angles[0] < -18
               and abs(fit.angles[2]) < .5]
    assert correct and min(f.score for f in correct) < .18
    assert fits[0].score < .18
    reference = cv2.Rodrigues(fits[0].candidate.rvec)[0]
    expected = geometry.sticker_assignment_signature(m, min(correct, key=lambda f: f.score), reference)
    assert expected in {geometry.sticker_assignment_signature(m, f, reference) for f in fits[:-1]}
    assert fits[-1].flat_fallback and fits[-1].angles == (0., 0., 0.)


def test_assignment_bank_has_five_unique_rows_plus_unconditional_flat(observation):
    base = geometry._materialize(m, observation[0], 0, np.r_[
        observation[0].rvec.ravel(), observation[0].tvec.ravel(), -15., 0.])
    candidates = []
    for index in range(7):
        candidates.append(replace(base, nearest=np.array([index, 0]), valid=np.array([True, False]),
                                  score=index*.01))
    # A different invalid placeholder and different slab angles are not new assignments.
    candidates.append(replace(candidates[0], nearest=np.array([0, 53]), angles=(30., 0., 0.), score=.001))
    flat = replace(base, angles=(0., 0., 0.), score=10.,
                   nearest=np.array([0, 0]), valid=np.array([True, False]))
    bank = geometry.assignment_shortlist(m, [*candidates, flat])
    assert len(bank) == 6
    assert [f.score for f in bank[:5]] == [i*.01 for i in range(5)]
    assert len({geometry.sticker_assignment_signature(m, f, cv2.Rodrigues(base.candidate.rvec)[0])
                for f in bank[:5]}) == 5
    assert bank[-1].flat_fallback and bank[-1].angles == (0., 0., 0.)
    assert bank[-1].score == 10.  # The fallback ignores the score cutoff.


@pytest.mark.parametrize(('same_assignments', 'cost_delta', 'accepted'), [
    (True, .2, True), (False, -.02, True), (False, 0., True), (False, .2, False),
])
def test_final_flat_variants_freeze_camera_and_supported_layer_with_or_qualification(
        observation, monkeypatch, same_assignments, cost_delta, accepted):
    base = geometry._materialize(m, observation[0], 0, np.r_[
        observation[0].rvec.ravel(), observation[0].tvec.ravel(), -20., 15.])
    option = dict(base.option, layer_support=(1, 8, 20),
                  keypoint_error=.1*m.PNP_FINAL_ASSIGNMENT_ERROR_FRACTION,
                  sticker_error=.1*m.PNP_TURN_CENTER_PENALTY_SCALE)
    base = replace(base, nearest=np.array([0, 0]), valid=np.array([True, False]), option=option)
    fallback = replace(base, angles=(0., 0., 0.), flat_fallback=True)
    calls = []

    def project(renderer, candidate, axis, parameters):
        calls.append(parameters.copy())
        angles = (parameters[6], 0., parameters[7])
        variant_option = dict(option, keypoint_error=(.1+2*cost_delta)*m.PNP_FINAL_ASSIGNMENT_ERROR_FRACTION)
        return replace(base, angles=angles, option=variant_option,
                       nearest=np.array([0 if same_assignments else 1, 53]))

    monkeypatch.setattr(geometry, '_materialize', project)
    diagnostics = {}
    bank = geometry.final_assignment_bank(m, [base, fallback], diagnostics=diagnostics)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][:6], np.r_[base.candidate.rvec.ravel(), base.candidate.tvec.ravel()])
    np.testing.assert_array_equal(calls[0][6:], [0., 15.])
    assert bank[0].angles == ((0., 0., 15.) if accepted else base.angles)
    assert bank[-1] is fallback
    assert diagnostics['attempted_materializations'] == 1
    assert diagnostics['qualified_variants'] == int(accepted)


def test_final_flat_variants_do_no_projection_for_supported_layers(observation, monkeypatch):
    base = geometry._materialize(m, observation[0], 0, np.r_[
        observation[0].rvec.ravel(), observation[0].tvec.ravel(), -20., 15.])
    base = replace(base, option=dict(base.option, layer_support=(2, 8, 20)))

    def unexpected_projection(*args):
        pytest.fail('Supported layers must not incur another projection')

    monkeypatch.setattr(geometry, '_materialize', unexpected_projection)
    diagnostics = {}
    bank = geometry.final_assignment_bank(m, [base], diagnostics=diagnostics)
    assert bank[0] is base
    assert diagnostics['attempted_materializations'] == 0


def test_assignment_bank_collapses_symmetry_relabeling_but_keeps_validity_changes(observation):
    base = geometry._materialize(m, observation[0], 0, np.r_[
        observation[0].rvec.ravel(), observation[0].tvec.ravel(), -15., 0.])
    base = replace(base, nearest=np.array([0, 8]), valid=np.array([True, False]), score=0.)
    reference = cv2.Rodrigues(base.candidate.rvec)[0]
    variants = [base]
    for symmetry, permutation in zip(m.CUBE_SYMMETRY_MATRICES, geometry._sticker_symmetry_permutations(m)):
        renamed = replace(base.candidate, rvec=cv2.Rodrigues(reference @ symmetry)[0])
        variants.append(replace(base, candidate=renamed, nearest=np.argsort(permutation)[base.nearest], score=.001))
    changed = replace(base, valid=np.array([True, True]), score=.002)
    bank = geometry.assignment_shortlist(m, [*variants, changed])
    assert len(bank) == 2  # No padding, and no flat candidate was supplied to this selector.
    assert bank[0].score == 0. and bank[1].score == .002


def test_every_axis_evaluates_the_complete_angle_grid(observation, monkeypatch):
    evaluated = {}
    original = m.parallel_turn_frame_geometry

    def sweep(candidate, limit, axis, **kwargs):
        result = original(candidate, limit, axis, **kwargs)
        if 'angles' not in kwargs:
            evaluated[axis] = {m.parallel_layer_angles(o['layer_rotations']) for o in result[0]}
        return result

    monkeypatch.setattr(m, 'parallel_turn_frame_geometry', sweep)
    geometry.fit_frame(m, observation[0])
    from itertools import product
    expected = set(product(range(-45, 46, 15), repeat=3))-{(0, 0, 0)}
    assert evaluated == {axis: expected for axis in range(3)}


def test_middle_frame_conversion_preserves_geometry_scores(observation):
    candidate = observation[0]
    swept = geometry.swept_geometries(m, candidate, candidate)
    for axis in range(3):
        for angles in ((-45., 15., 30.), (30., -30., -30.), (0., 45., 0.)):
            fit = next(f for f in swept if f.axis == axis and
                       m.parallel_layer_angles(f.option['layer_rotations']) == angles)
            seed = np.r_[fit.candidate.rvec.ravel(), fit.candidate.tvec.ravel(),
                         np.asarray(fit.angles)[[0, 2]]]
            materialized = geometry._materialize(m, candidate, axis, seed)
            assert materialized.score == pytest.approx(fit.score, abs=1e-8)


def test_good_sweep_measurements_survive_bad_refinement(observation, monkeypatch):
    candidate = observation[0]
    swept = geometry.swept_geometries(m, candidate, candidate)
    expected = min(f.score for f in swept)

    def bad_refinement(model, seed, **kwargs):
        return np.r_[candidate.rvec.ravel(), candidate.tvec.ravel(), 0., 0.]

    monkeypatch.setattr(geometry, '_refine', bad_refinement)
    fits = geometry.fit_frame(m, candidate)
    assert fits[0].score <= expected+1e-10
    assert fits[0].proposal_source == 'angle_sweep'


def test_matty425_d_only_fit_is_generated_from_exhaustive_sweep(observation, monkeypatch):
    record = json.loads((Path(__file__).parent/'fixtures/matty425-observations.json').read_text())
    candidate = playback.candidate_for_step(m, record, None, m.CubeGeometryState.identity(), observation[1])
    pool = []
    select = geometry.assignment_shortlist
    def capture(renderer, candidates, **kwargs):
        pool.extend(candidates)
        return select(renderer, candidates, **kwargs)
    monkeypatch.setattr(geometry, 'assignment_shortlist', capture)
    fits = geometry.fit_frame(m, candidate)
    correct = [f for f in pool if f.axis == 2 and -32 < f.angles[0] < -27 and f.angles[2] == 0.]
    assert correct and min(f.score for f in correct) < .08
    reference = cv2.Rodrigues(fits[0].candidate.rvec)[0]
    key = geometry.sticker_assignment_signature(m, min(correct, key=lambda f: f.score), reference)
    assert key in {geometry.sticker_assignment_signature(m, f, reference) for f in fits[:-1]}


def test_nearby_angle_solutions_share_a_pose_slot(observation):
    base = observation[0]
    parameters = np.r_[base.rvec.ravel(), base.tvec.ravel(), -24., 4.]
    center = geometry._materialize(m, base, 2, parameters)
    variations = []
    for angle in (-3., 3.):
        shifted = parameters.copy()
        shifted[6] += angle
        fit = geometry._materialize(m, base, 2, shifted)
        variations.append(fit)
    opposite = parameters.copy()
    opposite[6] = 24.
    distinct = geometry._materialize(m, base, 2, opposite)
    retained = geometry.diverse_shortlist(m, [center, *variations, distinct])
    assert len(retained) == 2
    assert any(fit.angles[0] > 0 for fit in retained)
    assert any(fit.angles[0] < 0 for fit in retained)


def test_bidirectional_seeds_recover_matty420_middle_pose(observation, monkeypatch):
    data = json.loads((Path(__file__).parent/'fixtures/matty420-temporal-poses.json').read_text())
    records = data['records']
    colors = observation[1]
    sources = [playback.candidate_for_step(m, r, None, m.CubeGeometryState.identity(), colors)
               for r in records]
    fitted = []
    for source, record in zip(sources, records, strict=True):
        option = record['refined']
        rotations = tuple(m.CubeLayerRotation(**r) for r in option['layer_rotations'])
        fitted.append(replace(source, rvec=np.asarray(option['pose_rvec']),
            tvec=np.asarray(option['pose_tvec']), layer_rotations=rotations))
    option = records[1]['refined']
    parameters = np.r_[fitted[1].rvec.ravel(), fitted[1].tvec.ravel(),
                        np.asarray(m.parallel_layer_angles(fitted[1].layer_rotations))[[0, 2]]]
    previous_fit = geometry._materialize(m, sources[1], option['layer_axis'], parameters)
    seeded_axes = {direction: set() for direction in ('previous_frame', 'next_frame')}
    reference = cv2.Rodrigues(sources[1].rvec)[0]
    expected = {direction: np.r_[cv2.Rodrigues(m._align_rigid_rotation(pose.rvec, reference))[0].ravel(),
                                 pose.tvec.ravel()]
                for direction, pose in (('previous_frame', fitted[0]), ('next_frame', fitted[2]))}
    original_sweep = m.parallel_turn_frame_geometry

    def sweep(candidate, limit, axis, **kwargs):
        pose_parameters = np.r_[candidate.rvec.ravel(), candidate.tvec.ravel()]
        for direction, pose in expected.items():
            if 'angles' not in kwargs and np.allclose(pose_parameters, pose):
                seeded_axes[direction].add(axis)
        return original_sweep(candidate, limit, axis, **kwargs)

    monkeypatch.setattr(m, 'parallel_turn_frame_geometry', sweep)
    proposals = geometry.fit_frame(m, sources[1], retained=[previous_fit],
        neighbors=(fitted[0], fitted[2]), prior=fitted[0])
    assert seeded_axes == {'previous_frame': {0, 1, 2}, 'next_frame': {0, 1, 2}}
    best = proposals[0]
    errors = np.linalg.norm(best.candidate.assignment_image_points-
                            best.candidate.assignment_projected_points, axis=1)
    assert best.score < .18 < previous_fit.score
    assert errors.mean() < 4.
    assert all(fit.angles[1] == 0. for fit in proposals)


def test_cached_options_skip_all_projection_during_state_exploration(observation, fits, monkeypatch):
    candidate, colors = observation
    options = tuple(geometry.projection_option(m, fit) for fit in fits)
    selected = replace(fits[0].candidate, joint_geometry_candidates=options)
    record = search.prepare_search_record(m, SimpleNamespace(pose_candidate=selected,
        frame_index=469), colors, 380, 'cpu')
    assert record['search']['geometry']['joint_refined']
    for fit, option in zip(fits, options, strict=True):
        _, nearest, valid = m.parallel_turn_frame_geometry(fit.candidate,
            m.PNP_FINAL_ASSIGNMENT_ERROR_FRACTION*m.projected_cube_diameter(fit.candidate),
            fit.axis, angles=fit.angles)
        np.testing.assert_array_equal(nearest[0], option['cached_sticker_indices'])
        np.testing.assert_array_equal(valid[0], option['cached_sticker_valid'])

    def forbidden(*args, **kwargs):
        raise AssertionError('State scoring must reuse the per-geometry projection bank')

    monkeypatch.setattr(m, '_score_turn_color_offsets_for_states_on_gpu', forbidden)
    monkeypatch.setattr(m, 'parallel_turn_frame_geometry', forbidden)
    scorer = search.PreparedReplayScorer(m, {469: record}, 'cpu', parallel_layers=True,
        geometry_shortlist_size=5, balance_geometry_color=True, initial_colors=colors)
    state = scorer.key(m.CubeGeometryState.identity())
    compiled = scorer.prepare(469)
    assert compiled['projection_count'] == len(fits)
    assert len(fits)-1 in compiled['selected_geometry_indices']
    assert options[-1]['flat_fallback']
    choices = scorer(469, state)
    assert choices
    for choice in choices:
        payload = choice.payload
        option = options[payload['geometry_option_index']]
        np.testing.assert_allclose(payload['pose_rvec'], option['pose_rvec'])
        rendered = playback.candidate_for_step(m, record, payload,
            m.CubeGeometryState.identity(), colors)
        np.testing.assert_allclose(rendered.rvec, option['pose_rvec'])


def test_wireframe_cache_needs_no_roi_and_never_observes_moves(observation, fits, monkeypatch):
    args = m.build_parser().parse_args(['--source', 'clip.mp4', '--initial-scramble', '',
        '--wireframe-only', '--no-display'])
    frame = np.zeros((1920, 1080, 3), np.uint8)
    record = SimpleNamespace(pose_candidate=fits[0].candidate, pose_ready=True,
        roi_xyxy=None, roi_source='cached', local_points=np.empty((0, 2)))

    def forbidden(*args, **kwargs):
        raise AssertionError('A wireframe render must not invoke detection or move search')

    monkeypatch.setattr(m, 'shared_roi_for_frame', forbidden)
    output, _, _ = m.render_frame(frame, None, None, None, args,
        cached_final_only=True, neural_data=record,
        move_accumulator=SimpleNamespace(observe=forbidden))
    assert np.count_nonzero(output) > 100
    assert not frame.any()


def test_default_cached_pipeline_refits_before_orientation_without_search(observation, monkeypatch):
    from scripts import independent_keypoint_video_search as video_search

    candidate, colors = observation
    args = m.build_parser().parse_args(['--source', 'clip.mp4', '--initial-scramble', '',
                                      '--wireframe-only', '--no-display'])
    records = [SimpleNamespace(frame_index=frame, pose_candidate=candidate, roi_xyxy=None)
               for frame in (468, 469, 470)]
    stages = []
    original_refine = geometry._refine
    original_orient = m.match_orientation_to_colors

    def refine(model, seed, **kwargs):
        assert 'orientation' not in stages
        stages.append('refine')
        return original_refine(model, seed, **kwargs)

    def orient(*args, **kwargs):
        assert stages and stages[0] == 'refine'
        stages.append('orientation')
        return original_orient(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError('The joint pipeline must not run rigid-only refitting or move search')

    monkeypatch.setattr(geometry, '_refine', refine)
    monkeypatch.setattr(m, 'match_orientation_to_colors', orient)
    monkeypatch.setattr(m, 'recover_cached_pose_symmetries', forbidden)
    monkeypatch.setattr(m, 'smooth_cached_rigid_poses', forbidden)
    monkeypatch.setattr(video_search, 'run_cached_full_search', forbidden)
    m.process_cached_poses(records, args, colors, source_fps=59.95)
    assert stages[-1] == 'orientation'
    for record in records:
        pose = record.pose_candidate
        assert record.pose_ready and pose.turn_state_moves == ()
        assert len(pose.joint_geometry_candidates) >= 5
        assert pose.turn_search_diagnostics['geometry']['joint_refined']
