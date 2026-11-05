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
            lambda state: state.endswith("goal"), str, min_update_gap=1, queue=queue)
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


def test_finite_high_costs_remain_eligible_without_rolling_gate_or_budget():
    result = search.frame_full_solve(0, tuple(range(8)),
        lambda frame, state: (edge(state + 1, 1000 + frame),), lambda state: state == 8, str)
    assert result["status"] == "solved"
    assert not result["error_gate_enabled"]
    assert len(result["steps"]) == 8
    assert result["current_frame_score"] == 1007


def test_color_cutoff_prunes_below_two_thirds_but_keeps_exact_boundary():
    bad, exact = edge("bad", 0.), edge("goal", .8)
    bad.payload["color_error"] = .5
    exact.payload["color_error"] = 1. - 2./3.
    result = search.frame_full_solve("root", (0,), lambda *_: (bad, exact), lambda s: s == "goal", str)
    assert result["status"] == "solved"
    assert result["color_pruned"] == 1
    assert result["steps"][0]["state_score"] == .8


def test_color_pruning_exhausts_without_retrying_pruned_branches():
    bad = edge("goal", 0.)
    bad.payload["color_error"] = .5
    result = search.frame_full_solve("root", (0,), lambda *_: (bad,), lambda s: s == "goal", str)
    assert result["status"] == "exhausted"
    assert result["color_pruned"] == 1
    assert result["queue_pending"] == 0
    assert result["steps"] == []


def test_gap_exception_allows_only_second_consecutive_same_direction_turn():
    direction = ("layer", 0, 1, 1, (1,))
    graph = {(0, 0): (edge(1, 1, ("R",), direction),),
             (1, 1): (edge(2, 1, ("R",), direction),),
             (2, 2): (edge(3, 0, ("R",), direction), edge(2, 1)),
             (3, 2): (edge(3, 1, ("R",), direction),)}
    result = search.frame_full_solve(0, (0, 1, 2, 3),
        lambda frame, state: graph.get((frame, state), ()), lambda state: state == 3, str,
        min_update_gap=2)
    assert result["status"] == "solved"
    assert result["spacing_pruned"] == 1
    assert [step["state_moves"] for step in result["steps"]] == [("R",), ("R",), (), ("R",)]


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


def test_full_search_main_solves_using_raw_nine_class_colors(renderer, tmp_path, monkeypatch):
    source = tmp_path / "observations.jsonl"
    source.write_text(json.dumps(observation_record(renderer)) + "\n")
    output = tmp_path / "search"
    monkeypatch.setattr(sys, "argv", ["frame-search", "--log", str(source),
        "--initial-scramble", "R", "--fps", "30", "--output-dir", str(output)])
    search.main()
    result = json.loads((output / "result.json").read_text())
    assert result["status"] == "solved"
    assert result["steps"][0]["state_moves"] == ["R'"]
    assert result["steps"][0]["state_score"] == pytest.approx(0.)
    assert result["config"]["priority_statistic"] == "latest_frame_geometric_plus_color_cost"
    assert result["config"]["forecast_frames"] == 0
    assert result["config"]["past_path_priority_weight"] == 0.


def test_video_input_defaults_to_medium_nine_class_checkpoint_and_static_gate():
    args = search.build_parser().parse_args(["--source", "solve.mp4", "--initial-scramble", "R U",
        "--turning-start-frame", "380", "--output-dir", "output"])
    assert "yoloPoseMedium" in str(args.model)
    assert args.turning_start_frame == 380
    assert args.fps is None
    assert not hasattr(args, "threshold") and not hasattr(args, "forecast_frames")
