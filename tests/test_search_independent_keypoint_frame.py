import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "frame_search_tests", Path(__file__).resolve().parents[1] / "scripts/search_independent_keypoint_frame.py")
search = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = search
SPEC.loader.exec_module(search)


def edge(state, score, moves=(), direction=None):
    return search.replay.Transition(state, score, (score,), dict(state_moves=moves), direction)


@pytest.mark.parametrize("backend", ["sqlite", "buffered"])
def test_global_queue_uses_latest_frame_not_path_mean_forecast_or_depth_first(backend):
    graph = {(0, "root"): (edge("a", 0), edge("b", 4)),
             (1, "a"): (edge("a_goal", 5),), (1, "b"): (edge("b_goal", 4.5),)}
    visited = []
    def transitions(frame, state):
        visited.append((frame, state))
        return graph.get((frame, state), ())
    queue = (search.storage.MeanQueue() if backend == "sqlite" else
             search.storage.BufferedMeanQueue(frontier_cache_size=1))
    try:
        result = search.frame_full_solve("root", (0, 1), transitions,
            lambda state: state.endswith("goal"), str, queue=queue)
    finally:
        queue.close()
    # A history mean or two-frame forecast would choose [0, 5]; depth first
    # would also commit that first generated goal. Global latest-frame order
    # pops b (4) before a_goal (5), then b_goal (4.5) before a_goal (5).
    assert [step["state_score"] for step in result["steps"]] == [4, 4.5]
    assert all(step["queue_priority"] == step["state_score"] for step in result["steps"])
    assert visited == [(0, "root"), (1, "a"), (1, "b")]


def test_same_context_dominance_uses_latest_score_even_when_history_cost_is_higher():
    graph = {(0, "root"): (edge("a", 0), edge("b", 5)),
             (1, "a"): (edge("goal", 8),), (1, "b"): (edge("goal", 7),)}
    result = search.frame_full_solve("root", (0, 1),
        lambda frame, state: graph.get((frame, state), ()), lambda state: state == "goal", str)
    assert [step["state_score"] for step in result["steps"]] == [5, 7]
    assert result["improved_pending"] == 1


def test_queue_can_reweight_geometry_without_changing_candidate_fit():
    def fitted(state, geometry, color):
        fit = .5 * geometry + .5 * color
        return search.replay.Transition(state, fit, (fit,), dict(
            state_moves=(), color_error=color,
            score_components=dict(geometry=geometry, color=color)), None)

    graph = {(0, "root"): (fitted("a", .1, .26), fitted("b", .3, .08)),
             (1, "a"): (edge("a_goal", 0.),), (1, "b"): (edge("b_goal", 0.),)}

    def transitions(frame, state):
        return graph.get((frame, state), ())

    def solved(state):
        return state.endswith("goal")

    baseline = search.frame_full_solve("root", (0, 1), transitions, solved, str)
    reweighted = search.frame_full_solve(
        "root", (0, 1), transitions, solved, str, queue_geometry_weight=1/3)
    color_only = search.frame_full_solve(
        "root", (0, 1), transitions, solved, str, ignore_queue_geometry=True)
    assert baseline["steps"][0]["state_score"] == pytest.approx(.18)
    for result, priority in ((reweighted, .14), (color_only, .04)):
        assert result["steps"][0]["state_score"] == pytest.approx(.19)
        assert result["steps"][0]["queue_priority"] == pytest.approx(priority)


def test_hold_wins_score_tie_with_unsupported_update():
    move = search.replay.Transition("turned", .2, (.2,), dict(
        state_moves=("R",), layer_inlier_count=0), None)
    graph = {(0, "root"): (move, edge("held", .2)),
             (1, "held"): (edge("goal", .1),)}
    visited = []

    def transitions(frame, state):
        visited.append((frame, state))
        return graph.get((frame, state), ())

    result = search.frame_full_solve("root", (0, 1), transitions,
        lambda state: state == "goal", str)
    assert result["status"] == "solved"
    assert visited == [(0, "root"), (1, "held")]


def test_bottom_quartile_orders_queue_using_the_complete_observed_history():
    def fitted(state, fit, matches):
        return search.replay.Transition(state, fit, (fit,), dict(
            state_moves=(), color_match_count=4, color_error=1 - matches.bit_count() / 4,
            sticker_observed_mask_hex='f', sticker_correct_mask_hex=f'{matches:x}'))

    graph = {(1, 'root'): (fitted('a', .125, 0b1110), fitted('b', .17, 0b1111)),
             (3, 'b'): (fitted('goal', .125, 0b1110),)}
    visited = []

    def transitions(frame, state):
        visited.append((frame, state))
        return graph[frame, state]

    result = search.frame_full_solve('root', (1, 3), transitions,
        lambda state: state == 'goal', str, bottom_quartile_weight=.25)
    assert result['status'] == 'solved'
    assert visited == [(1, 'root'), (3, 'b')]
    step = result['steps'][-1]
    assert step['bottom_quartile_match_rate'] == .5
    assert step['state_score'] == .125
    assert step['queue_priority'] == .25


def test_mismatch_window_pools_counts_reorders_queue_and_expires_oldest_frame():
    def fitted(state, observed, missed=0, out_of_table=0):
        error = missed / observed
        fit = .5 * error
        return search.replay.Transition(state, fit, (fit,), dict(
            state_moves=(), color_match_count=observed, color_error=error,
            color_forbidden_count=out_of_table))

    def transitions(frame, state):
        if frame == 0:
            return (fitted('prefix', 4, 1, 1),)
        if frame == 7:
            return fitted('wrong', 16, 2, 2), fitted('right', 16, 3)
        if frame == 8:
            return (fitted(state + '_goal', 8),)
        return (fitted('prefix', 1),)

    def solve(weight):
        return search.frame_full_solve('root', range(9), transitions,
            lambda state: state.endswith('_goal'), str,
            ignore_queue_geometry=True, mismatch_window_weight=weight)

    assert solve(0.)['steps'][7]['color_forbidden_count'] == 2
    steps = solve(1.)['steps']
    assert all(step['window_mismatch_penalty'] == 0. for step in steps[:7])
    assert steps[7]['color_forbidden_count'] == 0
    assert steps[7]['state_score'] == .5 * 3 / 16
    assert steps[7]['window_mismatch_rate'] == 1 / 26
    assert steps[8]['window_color_observations'] == 30
    assert steps[8]['window_mismatch_rate'] == 0.


@pytest.mark.parametrize('stage_budget', [1, 5000])
def test_branch_widening_releases_saved_siblings_without_scoring_parent_again(stage_budget):
    def fitted(state, score, color):
        return search.replay.Transition(state, score, (score,), dict(
            state_moves=(), color_error=color), None)

    calls = []
    events = []

    def transitions(frame, state):
        calls.append((frame, state))
        if state == 'root':
            # The winning child has the best color but ranks fifth by local fit.
            return [fitted(i, i / 10, .2 if i != 5 else 0.) for i in reversed(range(1, 9))]
        return []

    result = search.frame_full_solve('root', range(2), transitions, lambda state: state == 5, str,
        branch_widths=(3, 5, 8), branch_stage_expansions=stage_budget, ignore_queue_geometry=True,
        progress=events.append)

    assert result['status'] == 'solved'
    assert result['steps'][0]['sibling_rank'] == 5
    assert result['steps'][0]['queue_priority'] == 0.
    assert result['steps'][0]['state_score'] == .5
    assert [e['branch_width'] for e in events if e['event'] == 'widened'] == [5]
    assert result['deferred_promoted'] == 2
    assert calls.count((0, 'root')) == 1


@pytest.mark.parametrize('widths', [(), (1, 2)])
def test_resume_restores_the_queue_order_and_path_history(tmp_path, widths):
    def transitions(frame, state):
        return tuple(search.replay.Transition(next_state, .1, (.1,), dict(
            state_moves=(), color_match_count=10, color_error=.2,
            color_forbidden_count=misses))
            for next_state, misses in ((state+1, frame % 3), (state, 2)))

    def solve(**options):
        return search.frame_full_solve(0, range(12), transitions, lambda state: state == 10, str,
            ignore_queue_geometry=True, mismatch_window_weight=1.,
            branch_widths=widths, branch_stage_expansions=8, **options)

    uninterrupted = solve()
    checkpoint = {}

    def stop(item):
        if item['event'] == 'progress' and item['evaluated_states'] == 8:
            checkpoint.update(item)
            raise search.SearchTimeLimit

    path = tmp_path / 'search.sqlite3'
    queue = search.storage.BufferedMeanQueue(path)
    try:
        with pytest.raises(search.SearchTimeLimit):
            solve(queue=queue, progress=stop, progress_seconds=1e-9)
    finally:
        queue.close()
    queue = search.storage.BufferedMeanQueue(path)
    try:
        resumed = solve(queue=queue, resume_from=checkpoint)
    finally:
        queue.close()
    assert resumed['status'] == 'solved'
    assert resumed['steps'] == uninterrupted['steps']
    assert resumed['evaluated_states'] == uninterrupted['evaluated_states']


def test_finite_high_costs_remain_eligible_without_rolling_gate_or_budget():
    result = search.frame_full_solve(0, tuple(range(8)),
        lambda frame, state: (edge(state + 1, 1000 + frame),), lambda state: state == 8, str)
    assert result["status"] == "solved"
    assert len(result["steps"]) == 8
    assert result["current_frame_score"] == 1007


@pytest.fixture(scope="module")
def renderer():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    return search.replay.load_renderer()


def observation_record(m):
    c = m.StickerPnPCandidate(
        sticker_type="corner", sticker_average_confidence=1., image_points=np.zeros((4, 2)),
        object_points=m.canonical_sticker_object_points("corner"), projected_sticker_points=np.zeros((4, 2)),
        camera_matrix=m.default_camera_matrix((256, 256), 1.2), dist_coeffs=np.zeros((4, 1)),
        rvec=np.array([[.18], [-.25], [.12]]), tvec=np.array([[0.], [0.], [200.]]),
        reprojection_error=0., image_size=(256, 256))
    faces = m.projected_sticker_faces(c, c.rvec, c.tvec)
    solved = search.replay.initial_sticker_colors(m, "")
    centers = [polygon.mean(axis=0).tolist() for _, _, polygon in faces]
    names = [solved[identity.label] for identity, _, _ in faces]
    return dict(frame_index=0, has_pose=True, color_scoring="direct_match_fraction",
        turn_color_observation_names=names,
        # Deliberately stale; supplying the scramble must rebuild matches from
        # the model's raw color classes, independently of a prior local search.
        turn_color_matches=np.zeros((54, len(centers))).tolist(),
        search_pose=dict(sticker_type="corner", rvec=c.rvec.tolist(), tvec=c.tvec.tolist(),
            camera_matrix=c.camera_matrix.tolist(), dist_coeffs=c.dist_coeffs.tolist(),
            sticker_mask_centers=centers, sticker_mask_center_uncertainties=[1.] * len(centers),
            sticker_color_names=names, image_size=[256, 256]),
        search=dict(geometry=dict(options=[dict(layer_axis=0, layer_side=1, layer_count=1,
            layer_rotations=[], label="static R", is_baseline=True,
            keypoint_mean_error=0., heatmap_mean_error=0., keypoint_p90_error=0., robust_error=0.)])))


def test_confusion_tracking_keeps_exact_color_fit_and_reports_out_of_table_misses(renderer):
    record = observation_record(renderer)
    names = record['search_pose']['sticker_color_names']
    actual = names[0]
    names[0] = next(color for color in renderer.COLOR_CLASS_CONFUSIONS
                    if color != actual and color not in renderer.COLOR_CLASS_CONFUSIONS[actual])
    colors = search.replay.initial_sticker_colors(renderer, '')
    scorers = [search.replay.PreparedReplayScorer(renderer, {0: record}, 'cpu',
               initial_colors=colors, balance_geometry_color=True, track_color_confusions=track)
               for track in (False, True)]
    state = renderer.CubeGeometryState.identity()
    plain, tracked = (scorer.score_hold(0, scorer.key(state)) for scorer in scorers)
    assert tracked.score == plain.score
    assert tracked.payload['color_error'] == plain.payload['color_error']
    assert tracked.payload['color_forbidden_count'] == 1
    assert 'color_forbidden_count' not in plain.payload


@pytest.mark.parametrize('expected,predicted,cost', [
    ('white', 'yellow', .5), ('yellow', 'white', .5), ('yellow', 'orange', .5),
    ('orange', 'red', .5), ('red', 'orange', .5),
    ('orange', 'yellow', 1.), ('white', 'orange', 1.), ('yellow', 'green', 1.),
    ('white', 'white', 0.),
])
def test_26m_confusion_scoring_uses_only_direct_pairs(renderer, expected, predicted, cost):
    record = observation_record(renderer)
    names = record['search_pose']['sticker_color_names']
    names[:] = [predicted] * len(names)
    colors = {identity.label: expected for identity in renderer.canonical_sticker_identities()}
    scorer = search.replay.PreparedReplayScorer(renderer, {0: record}, 'cpu',
        initial_colors=colors, track_color_confusions=True, color_confusion_scoring=True,
        balance_geometry_color=True, tracked_color_confusions=renderer.POSE_26M_COLOR_CONFUSIONS)
    hold = scorer.score_hold(0, scorer.key(renderer.CubeGeometryState.identity()))
    payload = hold.payload
    observed = payload['color_match_count']
    assert observed > 0
    assert payload['color_error'] == cost
    assert hold.score == .5 * cost
    assert payload['color_correct_count'] == (observed if cost == 0. else 0)
    assert payload['color_confusion_count'] == (observed if cost == .5 else 0)
    assert payload['color_forbidden_count'] == (observed if cost == 1. else 0)


def test_handoff_keeps_distinct_cues_and_bonuses_only_the_next_commit(renderer):
    initial = renderer.CubeGeometryState.identity()
    previous_rotation = dict(axis=1, side=-1, angle_degrees=39., layer_count=1)
    next_rotation = dict(axis=2, side=1, angle_degrees=-33., layer_count=2)
    move = search.supported_completion(renderer, dict(
        layer_rotations=[previous_rotation], layer_inlier_count=1))
    assert move == "F'"
    assert search.supported_completion(renderer, dict(
        layer_rotations=[dict(previous_rotation, angle_degrees=12.)], layer_inlier_count=1)) is None
    assert search.supported_completion(renderer, dict(
        layer_rotations=[previous_rotation], layer_inlier_count=0)) is None
    assert search.supported_completion(renderer, dict(
        layer_rotations=[previous_rotation], layer_inlier_count=1, state_moves=("F",))) is None
    after = initial.apply_moves((move,))

    class Scorer:
        m = renderer

        def __call__(self, frame, state):
            flat = edge(state, .1 if frame == 443 else .3)
            if frame == 445:
                return (flat, edge(after, .25, (move,)))
            partial = search.replay.Transition(state, .15, (.15,), dict(
                state_moves=(), layer_rotations=[previous_rotation], layer_inlier_count=1), None)
            return flat, partial

        def apply_moves(self, state, moves):
            return state.apply_moves(moves)

        def score_hold(self, frame, state):
            assert frame == 445 and state.signature() == after.signature()
            return search.replay.Transition(state, .18, (.18,), dict(
                state_moves=(), layer_rotations=[next_rotation], layer_inlier_count=1,
                label="partial next layer", geometry_option_index=0, color_offset=0.), None)

    scorer = Scorer()
    goal = lambda state: state.signature() == after.signature()
    key = lambda state: state.signature()
    baseline = search.frame_full_solve(initial, (443, 445), scorer, goal, key)
    result = search.frame_full_solve(initial, (443, 445), scorer, goal, key, turn_handoff_bonus=.02)
    assert baseline['steps'][-1]['state_score'] == .25
    assert result['steps'][0]['layer_rotations'] == [previous_rotation]
    handoff = result['steps'][-1]
    assert handoff['handoff_commit']
    assert handoff['state_moves'] == (move,)
    assert handoff['layer_rotations'] == [next_rotation]
    assert handoff['color_offset'] == 0.
    assert handoff['state_score'] == .18
    assert handoff['queue_priority'] == pytest.approx(.16)


@pytest.mark.parametrize("cls", [search.replay.ReplayScorer, search.replay.PreparedReplayScorer])
@pytest.mark.parametrize("layer_inliers", [0, 1])
def test_partial_fit_needs_layer_support_even_with_lower_geometry_error(renderer, cls, layer_inliers):
    record = observation_record(renderer)
    options = record["search"]["geometry"]["options"]
    flat = options[0]
    flat.update(inlier_count=12, layer_inlier_count=0, keypoint_mean_error=.02)
    options.append(dict(flat, is_baseline=False, keypoint_mean_error=.01,
                        layer_inlier_count=layer_inliers, layer_rotations=[
                            dict(axis=0, side=1, angle_degrees=9., layer_count=1)]))
    # Every observed sticker matches, so only geometry distinguishes the fits.
    record["turn_color_matches"] = np.ones_like(record["turn_color_matches"]).tolist()
    scorer = cls(renderer, {0: record}, "cpu", balance_geometry_color=True)
    choices = scorer(0, renderer.CubeGeometryState.identity())
    hold = next(choice for choice in choices if not choice.payload["state_moves"])
    assert bool(hold.payload["layer_rotations"]) == bool(layer_inliers)
    assert hold.payload["inlier_count"] == 12
    assert hold.payload["keypoint_mean_error"] == (.01 if layer_inliers else .02)


@pytest.mark.parametrize("cls", [search.replay.ReplayScorer, search.replay.PreparedReplayScorer])
def test_static_prefix_and_missing_pose_never_invoke_turn_scoring(renderer, monkeypatch, cls):
    records = {0: dict(frame_index=0, has_pose=True, search=dict(status="forced_static")),
               1: dict(frame_index=1, has_pose=False)}
    scorer = cls(renderer, records, "cpu", turning_start_frame=0)
    def forbidden(*args, **kwargs):
        pytest.fail("No turn geometry or color work is allowed for held/unobserved frames")
    monkeypatch.setattr(renderer, "_score_turn_color_offsets_for_states_on_gpu", forbidden)
    state = renderer.CubeGeometryState.identity()
    for frame in (0, 1):
        assert scorer.prepare(frame) is None
        choices = scorer(frame, state)
        assert len(choices) == 1 and choices[0].state is state
        assert choices[0].payload["state_moves"] == ()
        assert choices[0].score == 0
    assert scorer.scoring_calls == 0


def test_replay_rejects_old_probability_logs_and_nonbinary_matches(renderer):
    record = observation_record(renderer)
    record.pop("turn_color_matches")
    record["turn_color_likelihoods"] = [[.5]] * 54
    scorer = search.replay.ReplayScorer(renderer, {0: record}, "cpu")
    with pytest.raises(ValueError, match="regenerate.*probability log"):
        scorer.prepare(0)
    record["turn_color_matches"] = np.full((54, len(record["turn_color_observation_names"])), .5).tolist()
    with pytest.raises(ValueError, match="binary"):
        scorer.prepare(0)


def test_full_search_main_solves_both_parities_with_raw_colors(renderer, tmp_path, monkeypatch):
    source = tmp_path / "observations.jsonl"
    source.write_text("".join(json.dumps(dict(observation_record(renderer), frame_index=i)) + "\n"
                              for i in range(6)))
    output = tmp_path / "search"
    monkeypatch.setattr(sys, "argv", ["frame-search", "--log", str(source),
        "--initial-scramble", "R", "--fps", "60", "--turning-start-frame", "2",
        "--branch-widths", "3", "5", "8", "--output-dir", str(output)])
    search.main()
    summary = json.loads((output / "summary.json").read_text())
    assert set(summary["passes"]) == {"even", "odd"}
    for name, frame, stage_budget in (("even", 4, 100), ("odd", 3, 200)):
        result = json.loads((output / name / "result.json").read_text())
        assert result["status"] == "solved"
        assert result["solved_frame_index"] == frame
        assert result["steps"][-1]["state_moves"] == ["R'"]
        assert result["steps"][-1]["state_score"] == pytest.approx(0.)
        assert result["steps"][-1]["queue_priority"] == result["steps"][-1]["state_score"]
        assert result["config"]["score_weights"] == {"geometry": .5, "color": .5}
        assert result["config"]["color_scoring"] == "direct_match_fraction"
        assert result["config"]["fps"] == 30.
        assert result["config"]["branch_stage_expansions"] == stage_budget


def test_parity_sampling_preserves_source_indices_and_accepts_split_logs():
    records = {i: {} for i in range(6)}
    assert search.frame_passes(records, tuple(range(1, 6)), "both", 60) == [
        ("even", (2, 4), 30), ("odd", (1, 3, 5), 30),
    ]
    split = {i: {"original_source_frame": 2*i+1} for i in range(3)}
    assert search.frame_passes(split, (0, 1, 2), "both", 30, already_sampled=True) == [
        ("odd", (0, 1, 2), 30),
    ]


def test_video_input_defaults_to_medium_nine_class_checkpoint_and_static_gate():
    args = search.build_parser().parse_args(["--source", "solve.mp4", "--initial-scramble", "R U",
        "--turning-start-frame", "380", "--output-dir", "output"])
    assert args.model == search.ROOT / "runs/pose/runs/26m-pose/weights/best.pt"
    assert args.turning_start_frame == 380
    assert args.fps is None
    assert args.frame_parity == "both"
    for removed in ("queue_order", "soft_top_k", "compensatory_beam", "require_layer_activity",
                    "pooled_match_history_weight", "bridge_unreliable_frames", "search_strategy"):
        assert not hasattr(args, removed)
