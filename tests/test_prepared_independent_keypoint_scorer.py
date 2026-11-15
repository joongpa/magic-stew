import importlib.util
from pathlib import Path
import sys
import random

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "prepared_replay_tests",
    Path(__file__).resolve().parents[1] / "scripts/search_independent_keypoint_full_solve.py",
)
search = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = search
SPEC.loader.exec_module(search)


@pytest.fixture(scope="module")
def renderer():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    return search.load_renderer()


def observation_record(m, kind="normal"):
    candidate = m.StickerPnPCandidate(
        sticker_type="corner", sticker_average_confidence=.9,
        image_points=np.zeros((4, 2)), object_points=m.canonical_sticker_object_points("corner"),
        projected_sticker_points=np.zeros((4, 2)),
        camera_matrix=m.default_camera_matrix((256, 256), 1.2),
        dist_coeffs=np.zeros((4, 1)), rvec=np.array([[.18], [-.25], [.12]]),
        tvec=np.array([[0.], [0.], [200.]]), reprojection_error=0.)
    centers = [polygon.mean(axis=0).tolist() for _, _, polygon in
               m.projected_sticker_faces(candidate, candidate.rvec, candidate.tvec)]
    if kind == "outside":
        centers = [[-10000., -10000.]] * len(centers)
    elif kind == "empty":
        centers = []
    likelihoods = np.random.default_rng(86).integers(0, 2, (54, len(centers))).astype(float)
    if kind == "zero":
        likelihoods[::2] = 0.
    options = []
    for axis in range(3):
        for side in (-1, 1):
            for width in (1, 2):
                for angle in (0., -33., 21., 45.):
                    options.append(dict(
                        layer_axis=axis, layer_side=side, layer_count=width,
                        label=f"{axis}:{side}:{width}:{angle}", is_baseline=angle == 0,
                        layer_rotations=[] if not angle else [dict(
                            axis=axis, side=side, layer_count=width, angle_degrees=angle)],
                        keypoint_mean_error=.01 + len(options) * .00013,
                        heatmap_mean_error=.02, keypoint_p90_error=.03,
                        robust_error=.07, inlier_count=10, layer_inlier_count=4))
    return dict(search_pose=dict(
        sticker_type="corner", camera_matrix=candidate.camera_matrix.tolist(),
        dist_coeffs=candidate.dist_coeffs.tolist(), rvec=candidate.rvec.tolist(),
        tvec=candidate.tvec.tolist(), sticker_mask_centers=centers,
        sticker_mask_center_uncertainties=[1. + i % 3 for i in range(len(centers))],
        image_size=[256, 256]),
        search=dict(geometry=dict(options=options)), turn_color_matches=likelihoods.tolist())


@pytest.mark.parametrize(('size', 'expected'), [(3, tuple(range(6))), (5, tuple(range(8)))])
def test_geometry_shortlist_counts_hold_once_and_includes_strict_cutoff_ties(renderer, size, expected):
    scorer = search.PreparedReplayScorer(renderer, {}, 'cpu', geometry_shortlist_size=size)
    flat = dict(layer_axis=0, layer_rotations=(), combined_mean_error=0.)
    geometry = [dict(flat, layer_axis=axis) for axis in range(3)]
    for axis, side, score in ((0, 1, .01), (1, -1, .02), (0, -1, .029), (2, 1, .03),
                             (1, 1, .039), (2, -1, .04)):
        geometry.append(dict(layer_axis=axis, combined_mean_error=score,
                             layer_rotations=(renderer.CubeLayerRotation(axis, side, 15.),)))
    assert scorer._geometry_shortlist(geometry) == expected


def test_shortlisted_hold_r_and_f_only_score_those_layer_updates(renderer):
    record = observation_record(renderer)
    flat = dict(record['search']['geometry']['options'][0], keypoint_mean_error=.025, heatmap_mean_error=0.)
    options = [flat]
    for axis, side, error in ((0, 1, .01), (1, -1, .02), (2, 1, .2)):
        options.append(dict(flat, layer_axis=axis, layer_side=side, is_baseline=False,
            keypoint_mean_error=error, layer_inlier_count=4,
            layer_rotations=[dict(axis=axis, side=side, angle_degrees=15., layer_count=1)]))
    record['search']['geometry']['options'] = options
    scorer = search.PreparedReplayScorer(renderer, {7: record}, 'cpu', geometry_shortlist_size=3)
    state = scorer.key(renderer.CubeGeometryState.identity())
    choices = scorer(7, state)
    assert scorer.frame_actions(7) == {('R',), ("R'",), ('F',), ("F'",)}
    assert {tuple(choice.payload['state_moves']) for choice in choices} == {
        (), ('R',), ("R'",), ('F',), ("F'",)}
    assert scorer.prepare(7)['configuration_count'] == 7
    assert scorer.score_action(7, state, ('U',)) == ()
    assert {choice.payload['geometry_option_index'] for choice in choices} <= {0, 1, 2}


def test_shortlisted_wide_fit_admits_every_subset_of_its_moving_slabs(renderer):
    record = observation_record(renderer, 'empty')
    option = dict(record['search']['geometry']['options'][0], is_baseline=False, layer_side=1,
        layer_count=2, layer_rotations=[dict(axis=0, side=1, layer_count=2, angle_degrees=15.)])
    record['search']['geometry']['options'] = [option]
    scorer = search.PreparedReplayScorer(renderer, {7: record}, 'cpu', geometry_shortlist_size=3)
    state = scorer.key(renderer.CubeGeometryState.identity())
    choices = scorer(7, state)
    expected = {action for action, offsets in renderer.parallel_turn_actions(0).items() if not offsets[0]}
    assert {tuple(choice.payload['state_moves']) for choice in choices} == expected
    assert len(expected) == 9
    assert all(choice.payload['parallel_layers'] for choice in choices)


def test_shortlisted_flat_geometry_checks_between_frame_turns_on_every_axis(renderer):
    record = observation_record(renderer)
    record['search']['geometry']['options'] = record['search']['geometry']['options'][:1]
    scorer = search.PreparedReplayScorer(renderer, {7: record}, 'cpu', geometry_shortlist_size=3)
    choices = scorer(7, scorer.key(renderer.CubeGeometryState.identity()))
    expected = {action for axis in range(3) for action in renderer.parallel_turn_actions(axis)}
    assert {choice.payload['state_moves'] for choice in choices} == expected
    assert scorer.frame_actions(7) == expected - {()}
    assert scorer.flat_turn_fallback(7)


@pytest.mark.parametrize('balance', [False, True])
@pytest.mark.parametrize('difference', [0., .00035, .009999, .01, .010001])
def test_flat_turn_fallback_includes_strict_geometry_ties(renderer, balance, difference):
    record = observation_record(renderer)
    error = difference * 2 * renderer.PNP_FINAL_ASSIGNMENT_ERROR_FRACTION if balance else difference
    flat = dict(record['search']['geometry']['options'][0], keypoint_mean_error=error, heatmap_mean_error=0.)
    partial = dict(flat, is_baseline=False, keypoint_mean_error=0.,
        layer_rotations=[dict(axis=0, side=1, angle_degrees=15., layer_count=1)])
    record['search']['geometry']['options'] = [flat, partial]
    scorer = search.PreparedReplayScorer(renderer, {7: record}, 'cpu', geometry_shortlist_size=3,
                                         balance_geometry_color=balance)
    expected = difference < search.GEOMETRY_TIE_TOLERANCE
    assert scorer.flat_turn_fallback(7) == expected
    assert (('U',) in scorer.frame_actions(7)) == expected
    state = scorer.key(renderer.CubeGeometryState.identity())
    assert bool(scorer.score_action(7, state, ('U',))) == expected


@pytest.mark.parametrize('handoff_available', [False, True])
def test_flat_best_fit_uses_broad_turns_only_without_a_handoff(renderer, monkeypatch, handoff_available):
    from scripts import search_independent_keypoint_frame as frame_search

    record = observation_record(renderer)
    flat = dict(record['search']['geometry']['options'][0], keypoint_mean_error=0., heatmap_mean_error=0.)
    record['search']['geometry']['options'] = [flat] + [dict(flat, layer_axis=axis, layer_side=side,
        is_baseline=False, keypoint_mean_error=error,
        layer_rotations=[dict(axis=axis, side=side, angle_degrees=15., layer_count=1)])
        for axis, side, error in ((0, 1, .01), (1, -1, .02))]
    scorer = search.PreparedReplayScorer(renderer, {7: record}, 'cpu', geometry_shortlist_size=3)
    state = scorer.key(renderer.CubeGeometryState.identity())
    if not handoff_available:
        monkeypatch.setattr(frame_search, 'score_handoffs', lambda *args: [])
    choices = frame_search.frame_choices(scorer, 7, state, 'U')
    moves = {choice.payload['state_moves'] for choice in choices}
    assert scorer.flat_turn_fallback(7)
    if handoff_available:
        base = {(), ('R',), ("R'",), ('F',), ("F'",)}
        assert moves == base | {('U',) + action for action in base}
        assert any(choice.payload.get('handoff_commit') for choice in choices)
    else:
        assert moves == {action for axis in range(3) for action in renderer.parallel_turn_actions(axis)}


@pytest.mark.parametrize('kind', ['normal', 'empty'])
def test_shortlist_without_finite_geometry_returns_no_moves(renderer, kind):
    record = observation_record(renderer, kind)
    for option in record['search']['geometry']['options']:
        option['keypoint_mean_error'] = None
    scorer = search.PreparedReplayScorer(renderer, {7: record}, 'cpu', geometry_shortlist_size=3)
    assert scorer(7, scorer.key(renderer.CubeGeometryState.identity())) == ()
    assert scorer.frame_actions(7) == frozenset()


@pytest.mark.parametrize(('fps', 'moving'), [(24., True), (60., False)])
def test_rotation_budget_flat_frames_compile_only_affordable_updates(renderer, fps, moving):
    record = observation_record(renderer)
    record['search']['geometry']['options'] = record['search']['geometry']['options'][:1]
    scorer = search.PreparedReplayScorer(renderer, {7: record}, 'cpu', geometry_shortlist_size=5)
    scorer.configure_rotation_budget((7,), fps, 32.)
    state = scorer.key(renderer.CubeGeometryState.identity())
    choices = scorer.rotation_choices(7, state)
    assert any(choice.payload['state_moves'] for choice in choices) == moving
    assert all(choice.payload['angular_travel_degrees'] <= 2880./fps for choice in choices)
    assert all(choice.payload['rotation_context'] == choices[0].payload['rotation_context'] for choice in choices)
    if not moving:
        assert scorer.child_states_built == 0
        assert scorer.prepare(7)['configuration_count'] == 2  # Legal hold and endpoint-only row.


@pytest.mark.parametrize('confusion', [False, True])
def test_rotation_budget_forward_and_reverse_agree_on_hidden_turn_between_wide_residuals(renderer, confusion):
    records = {}
    for frame, angle in ((7, 3.), (8, 14.)):
        record = observation_record(renderer)
        option = dict(record['search']['geometry']['options'][0], layer_axis=2, layer_side=1,
            layer_count=2, is_baseline=False, layer_rotations=[dict(
                axis=2, side=1, layer_count=2, angle_degrees=angle)])
        record['search']['geometry']['options'] = [option]
        if confusion:
            count = len(record['search_pose']['sticker_mask_centers'])
            record['search_pose']['sticker_color_names'] = ['red', 'green', 'white'] * (count//3) + ['red'] * (count%3)
        records[frame] = record
    scorer = search.PreparedReplayScorer(renderer, records, 'cpu', geometry_shortlist_size=5,
        color_confusion_scoring=confusion, initial_colors=search.initial_sticker_colors(renderer, 'R U') if confusion else None)
    scorer.configure_rotation_budget((7, 8), 24., 32.)
    state = scorer.key(renderer.CubeGeometryState.identity())
    before_context = scorer.prepare(7)['selected_geometry_indices'][0]
    choices = scorer.rotation_choices(8, state, before_context)
    forward = next(choice for choice in choices if choice.payload['state_moves'] == ("F'",))
    assert forward.payload['angular_travel_degrees'] == 107.
    reverse = next((before, context, choice, prior) for before, context, choice, prior in
        scorer.rotation_predecessors(8, forward.state, forward.payload['rotation_context'])
        if before == state and choice.payload['state_moves'] == ("F'",))
    before, context, choice, prior = reverse
    assert context == before_context
    assert choice.score == pytest.approx(forward.score)
    assert choice.payload['angular_travel_degrees'] == 107.
    expected_prior = scorer.score_geometry_hold(7, before, context).payload['color_error']
    assert prior == pytest.approx(expected_prior)


@pytest.mark.parametrize('angle', [30., 45.])
def test_rotation_budget_finishes_d_relative_to_aligned_middle_and_upper(renderer, angle):
    records = {frame: observation_record(renderer) for frame in (7, 8)}
    for frame, rotation in ((7, dict(axis=2, side=1, angle_degrees=-angle, layer_count=2)),
                            (8, dict(axis=1, side=-1, angle_degrees=-15., layer_count=1))):
        flat = records[frame]['search']['geometry']['options'][0]
        records[frame]['search']['geometry']['options'] = [dict(flat,
            layer_axis=rotation['axis'], layer_side=rotation['side'],
            layer_count=rotation['layer_count'], is_baseline=False, layer_rotations=[rotation])]
    scorer = search.PreparedReplayScorer(renderer, records, 'cpu', geometry_shortlist_size=5)
    scorer.configure_rotation_budget((7, 8), 24., 32.)
    context = scorer.prepare(7)['selected_geometry_indices'][0]
    state = scorer.key(renderer.CubeGeometryState.identity())
    forward = next(choice for choice in scorer.rotation_choices(8, state, context)
                   if choice.payload['state_moves'] == ('D',))
    assert forward.payload['angular_travel_degrees'] == 90. - angle + 15.
    backward = next(choice for before, previous, choice, _ in
        scorer.rotation_predecessors(8, forward.state, forward.payload['rotation_context'])
        if before == state and previous == context and choice.payload['state_moves'] == ('D',))
    assert backward.score == pytest.approx(forward.score)
    assert backward.payload['angular_travel_degrees'] == forward.payload['angular_travel_degrees']


def test_rotation_context_does_not_reuse_a_route_that_only_fits_another_start_angle(renderer):
    records = {7: observation_record(renderer), 8: observation_record(renderer)}
    flat = records[7]['search']['geometry']['options'][0]
    partial = dict(flat, layer_axis=0, layer_side=1, is_baseline=False,
        layer_rotations=[dict(axis=0, side=1, angle_degrees=40., layer_count=1)])
    records[7]['search']['geometry']['options'] = [flat, partial]
    records[8]['search']['geometry']['options'] = [dict(partial,
        layer_rotations=[dict(axis=0, side=1, angle_degrees=-40., layer_count=1)])]
    scorer = search.PreparedReplayScorer(renderer, records, 'cpu', geometry_shortlist_size=5)
    scorer.configure_rotation_budget((7, 8), 24., 32.)
    contexts = scorer.prepare(7)['rotation_geometries']
    flat_context = next(index for index, geometry in contexts.items() if geometry.axis is None)
    partial_context = next(index for index, geometry in contexts.items() if geometry.axis is not None)
    state = scorer.key(renderer.CubeGeometryState.identity())
    assert not any(choice.payload['state_moves'] == ('R2',) for choice in scorer.rotation_choices(8, state, flat_context))
    choice = next(choice for choice in scorer.rotation_choices(8, state, partial_context)
                  if choice.payload['state_moves'] == ('R2',))
    assert choice.payload['angular_travel_degrees'] == 100.


def test_rotation_budget_uses_source_time_and_spans_unavailable_observations(renderer):
    records = {index: observation_record(renderer) for index in (7, 8, 9)}
    for index, source in ((7, 100), (8, 102), (9, 105)):
        records[index]['original_source_frame'] = source
    records[8]['has_pose'] = False
    scorer = search.PreparedReplayScorer(renderer, records, 'cpu')
    scorer.configure_rotation_budget((7, 8, 9), 60., 32.)
    assert scorer.rotation_limits == {7: 48., 9: 240.}
    assert scorer.rotation_previous == {7: None, 9: 7}


@pytest.mark.parametrize("kind", ["normal", "zero", "outside", "empty"])
@pytest.mark.parametrize('track', [False, True])
def test_base_pose_projection_and_move_permutations_match_legacy(renderer, kind, track):
    import torch
    records = {7: observation_record(renderer, kind)}
    legacy = search.ReplayScorer(renderer, records, torch.device("cpu"), cache_size=0,
                                 track_sticker_observations=track)
    prepared = search.PreparedReplayScorer(renderer, records, torch.device("cpu"), cache_size=0,
                                          track_sticker_observations=track)
    initial = renderer.CubeGeometryState.identity()
    for state in (initial, initial.apply_moves(("R", "U", "F'")),
                  initial.apply_moves(("Rw'", "Uw", "Bw", "Lw", "Dw'", "Fw"))):
        old, new = legacy(7, state), prepared(7, state)
        def key(t):
            return legacy.key(t.state), t.update_key
        a, b = {key(t): t for t in old}, {key(t): t for t in new}
        assert a.keys() == b.keys()
        if kind == "normal":
            assert list(map(key, old)) == list(map(key, new))
        for k, expected in a.items():
            actual = b[k]
            assert actual.score == pytest.approx(expected.score, abs=1e-10)
            assert actual.rank == pytest.approx(expected.rank, abs=1e-10)
            assert actual.payload.keys() == expected.payload.keys()
            if kind == "normal":
                assert actual.payload["trial_id"] == expected.payload["trial_id"]
            for field in ("state_moves", "color_match_count", "keypoint_mean_error",
                          "heatmap_mean_error", "keypoint_p90_error", "inlier_count"):
                assert actual.payload[field] == expected.payload[field]
            if track:
                for field in ('sticker_observed_mask_hex','sticker_correct_mask_hex'):
                    assert actual.payload[field] == expected.payload[field]
                observed = int(actual.payload['sticker_observed_mask_hex'],16)
                correct = int(actual.payload['sticker_correct_mask_hex'],16)
                assert observed.bit_count() == actual.payload['color_match_count']
                assert correct.bit_count() == round((1-actual.payload['color_error'])*observed.bit_count())
    compiled = prepared.prepare(7)
    assert compiled["source_geometry_count"] == 48
    assert compiled["configuration_count"] == 144
    assert compiled["projection_count"] == 37  # 36 partial poses plus one static pose.
    assert not prepared.prepared  # Large projection tensors are released.


def test_prepared_frames_survive_revisiting_beyond_legacy_lru(renderer, monkeypatch):
    import torch
    record = observation_record(renderer)
    scorer = search.PreparedReplayScorer(renderer, dict.fromkeys(range(35), record), torch.device("cpu"))
    project = renderer._torch_project_turn_points
    calls = []

    def count_projection(*args, **kwargs):
        calls.append(args[1].shape[0])
        return project(*args, **kwargs)

    monkeypatch.setattr(renderer, "_torch_project_turn_points", count_projection)
    for frame in range(35):
        scorer.prepare(frame)
    initial = renderer.CubeGeometryState.identity()
    first = scorer(0, initial)
    assert scorer(0, initial) is first
    scorer(0, initial.apply_move("Rw"))
    assert calls == [37] * 35
    assert scorer.cache_hits == 1
    assert scorer.scoring_calls == 2
    assert len(scorer.compiled_frames) == 35


@pytest.mark.parametrize("cutoff", [None, 2./3.])
def test_balanced_scoring_and_color_cutoff_match_legacy(renderer, cutoff):
    record = observation_record(renderer)
    args = dict(balance_geometry_color=True, minimum_color_accuracy=cutoff)
    legacy = search.ReplayScorer(renderer, {7: record}, "cpu", **args)
    prepared = search.PreparedReplayScorer(renderer, {7: record}, "cpu", **args)
    for state in (renderer.CubeGeometryState.identity(), renderer.CubeGeometryState.identity().apply_move("Rw")):
        expected = {(legacy.key(t.state), t.update_key): t for t in legacy(7, state)}
        actual = {(t.state, t.update_key): t for t in prepared(7, prepared.key(state))}
        assert expected.keys() == actual.keys()
        for key, t in actual.items():
            assert t.score == pytest.approx(expected[key].score, abs=1e-10)
            c = t.payload["score_components"]
            assert 0. <= c["geometry"] <= 1.
            assert t.score == pytest.approx(.5*c["geometry"] + .5*c["color"])
            if cutoff is not None:
                assert t.payload["color_error"] <= 1. - cutoff


def test_geometry_normalization_assigns_equal_bounded_budgets(renderer):
    f = renderer.balanced_geometry_cost
    assert f(0., 0.) == 0.
    assert f(.05, 0.) == .5
    assert f(0., .05) == .5
    assert f(.05, .05) == 1.
    assert f(50., 0.) == .5
    assert np.isinf(f(float("inf"), 0.))


def test_prepared_scorer_requires_cpu(renderer):
    with pytest.raises(ValueError, match="requires CPU"):
        search.PreparedReplayScorer(renderer, {}, "mps")


def test_multi_state_scoring_matches_scalar_candidates_and_diagnostics(renderer):
    record = observation_record(renderer)
    count = len(record['search_pose']['sticker_mask_centers'])
    record['search_pose']['sticker_color_names'] = ['red', 'green', 'white'] * (count//3) + ['red'] * (count%3)
    options = dict(balance_geometry_color=True, color_confusion_scoring=True,
                   initial_colors=search.initial_sticker_colors(renderer, 'R U F2'))
    scorer = search.PreparedReplayScorer(renderer, {7: record}, 'cpu', **options)
    identity = renderer.CubeGeometryState.identity()
    states = [scorer.key(identity.apply_moves(moves)) for moves in ((), ('R',), ('U', "F'"), ('Rw', 'D2'))]
    expected = [scorer(7, state) for state in states]
    scorer.cache.clear()
    actual = scorer.evaluate_many(7, states)
    for old, new in zip(expected, actual, strict=True):
        assert [(t.state, t.rank, dict(t.payload)) for t in new] == [
            (t.state, t.rank, dict(t.payload)) for t in old]
    assert scorer.evaluate_many(7, []) == ()


def test_compact_transitions_match_geometry_for_turns_wide_turns_and_rotations(renderer):
    scorer = search.PreparedReplayScorer(renderer, {}, "cpu")
    state = renderer.CubeGeometryState.identity()
    compact = scorer.key(state)
    moves = [face + suffix for face in ("R", "L", "U", "D", "F", "B",
                                        "Rw", "Lw", "Uw", "Dw", "Fw", "Bw", "x", "y", "z")
             for suffix in ("", "'", "2")]
    rng = random.Random(102)
    for move in moves + rng.choices(moves, k=150):
        state = state.apply_moves((move,))
        compact = scorer.apply_moves(compact, (move,))
        assert isinstance(compact, bytes)
        assert len(compact) == 54
        assert compact == scorer.key(state)
        assert scorer.state_from_key(compact) is compact


def test_compact_goal_uses_physical_colors_and_accepts_whole_cube_rotations(renderer):
    scorer = search.PreparedReplayScorer(renderer, {}, "cpu")
    initial = renderer.CubeGeometryState.identity()
    scrambled = renderer.CubeState.from_scramble("R U F2")
    colors = {label: scrambled.colors[position] for position, label in initial.facelets.items()}
    goal = scorer.solved_predicate(colors)
    states = [initial, initial.apply_moves(("F2", "U'", "R'"))]
    states += [states[-1].apply_moves(rotation) for rotation in
               (("x",), ("y'",), ("z2",), ("Rw",), ("R",), ("x", "y", "z"))]
    assert not goal(scorer.key(initial))
    assert goal(scorer.key(states[1]))
    for state in states:
        assert goal(scorer.key(state)) == state.is_solved(colors)
    assert not scorer.solved_predicate({})(scorer.key(initial))
    bad_colors = dict(colors, **{next(iter(colors)): "unknown"})
    assert not scorer.solved_predicate(bad_colors)(scorer.key(initial))


def test_early_filter_skips_color_work_and_keeps_cache_histories_separate(renderer):
    scorer = search.PreparedReplayScorer(renderer, {7: observation_record(renderer)}, "cpu")
    initial = scorer.key(renderer.CubeGeometryState.identity())
    all_choices = scorer(7, initial)
    assert all(isinstance(t.state, bytes) for t in all_choices)
    r = next(t for t in all_choices if t.payload["state_moves"] == ("R",))
    before = scorer.color_configurations_scored, scorer.child_states_built
    only_stay, skipped = scorer.for_update_context(7, initial, (6, r.update_key, 2), 2)
    assert [t.payload["state_moves"] for t in only_stay] == [()]
    assert skipped == 24
    assert scorer.color_configurations_scored - before[0] == 48
    assert scorer.child_states_built == before[1]
    before = scorer.color_configurations_scored, scorer.child_states_built
    double, skipped = scorer.for_update_context(7, initial, (6, r.update_key, 1), 2)
    assert {t.payload["state_moves"] for t in double} == {(), ("R",)}
    assert skipped == 23
    assert scorer.color_configurations_scored - before[0] == 52
    assert scorer.child_states_built - before[1] == 1
    assert scorer.for_update_context(7, initial, (5, r.update_key, 1), 3)[0] == only_stay
    assert scorer.for_update_context(7, initial, (6, r.update_key, 1), 1)[0] is all_choices
    assert scorer.for_update_context(7, initial, (None, None, 0), 2)[0] is all_choices
    assert scorer.for_update_context(7, initial, (5, r.update_key, 2), 2)[0] is all_choices
    by_key = {(t.state, t.update_key): t for t in all_choices}
    for t in double:
        expected = by_key[(t.state, t.update_key)]
        assert t.rank == expected.rank
        assert t.payload == expected.payload


@pytest.mark.parametrize('cls', [search.ReplayScorer, search.PreparedReplayScorer])
def test_replay_rejects_cached_zero_keypoint_fallback_geometry(renderer, cls):
    record = observation_record(renderer)
    record['search_pose']['assignment_image_points'] = [[1., 1.], [2., 2.], [3., 3.]]
    record['search']['geometry']['options'][0].update(
        keypoint_mean_error=0., keypoint_inlier_count=0)
    scorer = cls(renderer, {7: record}, 'cpu')
    with pytest.raises(ValueError, match='zero-keypoint fallback.*regenerate'):
        scorer.prepare(7)


@pytest.mark.parametrize('cls', [search.ReplayScorer, search.PreparedReplayScorer])
def test_replay_preserves_nonfinite_geometry_serialized_as_null(renderer, cls):
    record = observation_record(renderer)
    for option in record['search']['geometry']['options']:
        option.update(keypoint_mean_error=None, keypoint_p90_error=None, global_robust_error=None)
    scorer = cls(renderer, {7: record}, 'cpu', balance_geometry_color=True, minimum_color_accuracy=2/3)
    assert scorer(7, renderer.CubeGeometryState.identity()) == ()


def test_split_tps_double_span_has_matching_forward_and_reverse_routes(renderer):
    records = {frame: observation_record(renderer) for frame in (7, 8, 9, 10)}
    for frame, angle in ((8, 20.), (9, -20.)):
        flat = records[frame]['search']['geometry']['options'][0]
        records[frame]['search']['geometry']['options'] = [dict(flat, layer_axis=0,
            is_baseline=False, parallel_layers=True, layer_rotations=[dict(
                axis=0, side=-1, angle_degrees=angle, layer_count=1)])]
    for frame in (7, 10):
        records[frame]['search']['geometry']['options'] = records[frame]['search']['geometry']['options'][:1]
    scorer = search.PreparedReplayScorer(renderer, records, 'cpu', geometry_shortlist_size=5)
    scorer.configure_rotation_budget((7, 8, 9, 10), 60., 40., 50.)
    state = scorer.key(renderer.CubeGeometryState.identity())
    before_context = scorer.prepare(7)['selected_geometry_indices'][0]
    choices = scorer.rotation_choices(8, state, before_context)
    fast = next(c for c in choices if c.payload['angular_travel_degrees'] == 70.)
    assert fast.payload['rotation_budget_degrees'] == pytest.approx(75.)
    reverse = next(c for prior, context, c, _ in scorer.rotation_predecessors(
        8, fast.state, fast.payload['rotation_context'])
        if prior == state and context == before_context and c.payload['state_moves'] == fast.payload['state_moves'])
    assert reverse.score == pytest.approx(fast.score)
    assert reverse.payload['angular_travel_degrees'] == fast.payload['angular_travel_degrees']
    assert reverse.payload['rotation_budget_degrees'] == fast.payload['rotation_budget_degrees']
    # Without the second part of the double, the exact same first 70 degrees
    # must be rejected; changing the TPS settings also invalidates caches.
    records[9]['search']['geometry']['options'] = records[10]['search']['geometry']['options']
    single = search.PreparedReplayScorer(renderer, records, 'cpu', geometry_shortlist_size=5)
    single.configure_rotation_budget((7, 8, 9), 60., 40., 50.)
    assert all(c.payload['angular_travel_degrees'] <= 60.
               for c in single.rotation_choices(8, state, before_context))
