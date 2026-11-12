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
