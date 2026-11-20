import importlib.util
import json
from pathlib import Path
import pickle
import sys

import pytest

SPEC = importlib.util.spec_from_file_location("forecast_search_tests",
    Path(__file__).resolve().parents[1]/"scripts/search_independent_keypoint_forecast.py")
search = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = search
SPEC.loader.exec_module(search)


def edge(state, score, moves=(), key=None):
    return search.replay.Transition(state, score, (score,), dict(state_moves=moves), key)


def test_forecast_is_uniform_includes_candidate_and_shortens_at_end():
    graph = {(1, "a"): (edge("end", 10),)}
    f = search.GreedyForecaster((0, 1), lambda frame, state: graph.get((frame, state), ()), str)
    priority, forecast = f.candidate(0, edge("a", 0), (None, None, 0), 20)
    assert priority == 5
    assert forecast == search.Forecast(10, 2, True)
    assert f.candidate(1, edge("end", 7), (None, None, 0), 20)[0] == 7


def test_forecast_follows_one_greedy_continuation_without_backtracking():
    visited = []
    graph = {(1, "a"): (edge("trap", 0), edge("repair", 5)),
             (2, "repair"): (edge("end", 1),)}
    def transitions(frame, state):
        visited.append((frame, state))
        return graph.get((frame, state), ())
    f = search.GreedyForecaster((0, 1, 2), transitions, str)
    priority, forecast = f.candidate(0, edge("a", 1), (None, None, 0), 3)
    assert priority == float("inf")
    assert not forecast.complete
    assert visited == [(1, "a"), (2, "trap")]


def test_forecast_work_is_linear_in_horizon_and_cached():
    calls = []
    def transitions(frame, state):
        calls.append((frame, state))
        return tuple(edge(state+1, 1+i) for i in range(25))
    f = search.GreedyForecaster(tuple(range(20)), transitions, str)
    args = (0, edge(1, 1), (None, None, 0), 20)
    assert f.candidate(*args)[0] == 1
    assert len(calls) == 19
    assert f.candidate(*args)[0] == 1
    assert len(calls) == 19
    assert f.cache_hits == 1


@pytest.mark.parametrize("cache_size", [0, 1, 100])
def test_forecast_cache_eviction_changes_neither_score_nor_choices(cache_size):
    f = search.GreedyForecaster(tuple(range(8)),
        lambda frame, state: (edge(state+1, frame+.25), edge(state+2, 20)), str,
        cache_size=cache_size)
    expected = sum(frame+.25 for frame in range(1, 8))+2
    for _ in range(3):
        score, forecast = f.candidate(0, edge(1, 2), (None, None, 0), 20)
        assert forecast == search.Forecast(expected, 8, True)
        assert score == expected/8
    assert len(f.cache) <= cache_size


def test_forecast_honors_gap_and_allows_only_second_same_direction_quarter():
    key = ("layer", 0, 1, 1, (1,))
    other = ("layer", 1, 1, 1, (1,))
    graph = {
        (1, "a"): (edge("wrong", 0, ("B",), other), edge("b", 4, ("A",), key)),
        (2, "b"): (edge("wrong", 0, ("A",), key), edge("end", 5)),
    }
    f = search.GreedyForecaster((0, 1, 2), lambda frame, state: graph.get((frame, state), ()), str)
    priority, result = f.candidate(0, edge("a", 1, ("A",), key), (0, key, 1), 3)
    assert result.complete
    assert priority == 10/3


@pytest.mark.parametrize("backend", ["sqlite", "buffered"])
def test_search_orders_by_forecast_not_immediate_or_whole_path_mean(backend):
    graph = {(0, "root"): (edge("a", 0), edge("b", 4)),
             (1, "a"): (edge("a_goal", 100),), (1, "b"): (edge("b_goal", 0),)}
    queue = search.storage.MeanQueue() if backend == "sqlite" else search.storage.BufferedMeanQueue(frontier_cache_size=1)
    try:
        result = search.forecast_full_solve("root", (0, 1),
            lambda frame, state: graph.get((frame, state), ()), lambda state: state.endswith("goal"), str,
            horizon=2, queue=queue)
        assert result["status"] == "solved"
        assert [step["state_score"] for step in result["steps"]] == [4, 0]
        assert result["steps"][0]["forecast_mean"] == 2
        assert result["steps"][-1]["whole_path_mean"] == 2
    finally:
        queue.close()


def test_no_gate_accepts_high_errors_and_no_forecast_moves_are_committed():
    result = search.forecast_full_solve(0, tuple(range(5)),
        lambda frame, state: (edge(state+1, 100),), lambda state: state == 5, str)
    assert result["status"] == "solved"
    assert not result["error_gate_enabled"]
    assert result["evaluated_states"] == 5
    assert len(result["steps"]) == 5
    assert [step["frame_index"] for step in result["steps"]] == list(range(5))
    assert [step["forecast_frames"] for step in result["steps"]] == [5, 4, 3, 2, 1]


def test_failed_greedy_forecast_keeps_branch_with_non_greedy_solution():
    graph = {(0, "root"): (edge("a", 0),),
             (1, "a"): (edge("trap", 0), edge("repair", 5)),
             (2, "repair"): (edge("goal", 1),)}
    events = []
    result = search.forecast_full_solve("root", (0, 1, 2),
        lambda frame, state: graph.get((frame, state), ()), lambda state: state == "goal", str,
        progress=events.append)
    assert result["status"] == "solved"
    assert [step["state_score"] for step in result["steps"]] == [0, 5, 1]
    assert result["steps"][0]["forecast_mean"] is None
    assert not result["steps"][0]["forecast_complete"]
    json.dumps(events, allow_nan=False)
    json.dumps(result, allow_nan=False)


def test_unsolved_endpoints_exhaust_and_infinite_forecasts_are_not_pruned():
    graph = {(0, "root"): (edge("a", 1), edge("b", 2)),
             (1, "b"): (edge("end", 3),)}
    result = search.forecast_full_solve("root", (0, 1),
        lambda frame, state: graph.get((frame, state), ()), lambda state: False, str)
    assert result["status"] == "exhausted"
    assert result["evaluated_states"] == 3
    assert result["unsolved_endpoints"] == 1


def test_expired_move_history_is_equivalent_but_live_history_is_not():
    key = ("layer", 0, 1, 1, (1,))
    assert search.canonical_update((0, key, 1), 2, 2) == (None, None, 0)
    assert search.canonical_update((1, key, 1), 2, 2) == (1, key, 1)
    assert search.canonical_update((1, key, 2), 2, 2) == (1, key, 2)
    assert search.canonical_update((1, key, 2), None, 2) == (None, None, 0)


def test_identical_future_contexts_merge_without_numeric_score_history():
    graph = {(0, "root"): (edge("a", 1), edge("b", 2)),
             (1, "a"): (edge("same", 1),), (1, "b"): (edge("same", 10),)}
    result = search.forecast_full_solve("root", (0, 1, 2),
        lambda frame, state: graph.get((frame, state), ()), lambda state: False, str)
    assert result["status"] == "exhausted"
    assert result["dominated_pruned"] == 1


@pytest.mark.parametrize("backend", ["sqlite", "buffered"])
def test_forecast_priority_survives_spill_and_reopening_database(tmp_path, backend):
    path = tmp_path/"queue.sqlite3"
    q = search.storage.MeanQueue(path) if backend == "sqlite" else search.storage.BufferedMeanQueue(path, frontier_cache_size=1)
    for index, priority in enumerate((float("inf"), 2., 1.)):
        node = search.storage.MeanNode(1, index, 100-index, (None, None, 0), priority=priority)
        q.offer(str(index).encode(), node, pickle.dumps(index))
    q.close()
    q = search.storage.MeanQueue(path)
    try:
        nodes = [q.pop(pickle.loads) for _ in range(3)]
        assert [node.state for node in nodes] == [2, 1, 0]
        assert [node.priority for node in nodes] == [1., 2., float("inf")]
        assert [node.mean for node in nodes] == [98, 99, 100]
    finally:
        q.close()


@pytest.mark.parametrize("kwargs", [dict(horizon=0), dict(min_update_gap=0),
    dict(forecast_cache_size=-1), dict(progress_seconds=0)])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        search.forecast_full_solve(0, (0, 1), lambda frame, state: (), lambda state: False, str, **kwargs)
