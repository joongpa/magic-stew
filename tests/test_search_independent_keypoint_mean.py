import importlib.util
from pathlib import Path
import pickle
import sys

import pytest

SPEC = importlib.util.spec_from_file_location("mean_search_tests",
    Path(__file__).resolve().parents[1]/"scripts/search_independent_keypoint_mean.py")
search = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = search
SPEC.loader.exec_module(search)


@pytest.fixture(params=["sqlite", "buffered"], autouse=True)
def queue_backend(request, monkeypatch):
    if request.param == "buffered":
        monkeypatch.setattr(search, "MeanQueue", lambda path=":memory:": search.BufferedMeanQueue(
            path, frontier_cache_size=2, cost_cache_size=3, write_batch_size=2))


def edge(state,score,rank=0,move=(),key=None):
    return search.replay.Transition(state,score,(rank,),dict(state_moves=move),key)


def test_queue_orders_full_path_mean_instead_of_last_score_or_dfs():
    graph = {
        (0,"root"):(edge("a",1,0),edge("b",4,1)),
        (1,"a"):(edge("a_goal",10),),
        (1,"b"):(edge("b_goal",0),),
    }
    result = search.mean_full_solve("root",(0,1),lambda f,s:graph.get((f,s),()),
        lambda s:s.endswith("goal"),str,threshold=100)
    assert result["status"] == "solved"
    assert [s["state_score"] for s in result["steps"]] == [4,0]
    assert result["steps"][-1]["whole_path_mean"] == 2


def test_mean_keeps_the_highest_ten_percent_of_costs():
    def transitions(frame,state):
        if frame == 0:
            return (edge(("a",1),0,0),edge(("b",1),20,1))
        name,n = state
        return (edge((name,n+1),300 if name=="a" and frame==9 else 0 if name=="a" else 20),)
    result = search.mean_full_solve(("root",0),tuple(range(10)),transitions,
        lambda s:s[1]==10,str,threshold=500)
    assert result["status"] == "solved"
    assert [s["state_score"] for s in result["steps"]] == [20]*10
    assert result["steps"][-1]["whole_path_mean"] == 20


def test_gate_defers_and_retries_only_the_required_window_then_returns_to_two():
    scores = (1,4,4,4,1,1)
    events = []
    result = search.mean_full_solve(0,tuple(range(6)),
        lambda f,s:(edge(s+1,scores[f]),),lambda s:s==6,str,
        threshold=2,progress=events.append)
    assert result["status"] == "solved"
    assert result["deferred_resumed"] == 2
    assert [s["applied_threshold"] for s in result["steps"]] == [2,2,2,3,3,2]
    assert result["steps"][-1]["whole_path_mean"] == 2.5
    assert all(e["queue_pending"]==1 and e["return_threshold_next_frame"]==2
               for e in events if e["event"]=="temporary_allowance")


def test_all_normal_work_finishes_before_any_deferred_retry():
    graph = {
        (0,"root"):(edge("a",.1,0),edge("b",3,1)),
        (1,"a"):(edge("a2",3),),(2,"a2"):(edge("a3",3),),
        (3,"a3"):(edge("goal",3),),
        (1,"b"):(edge("b2",1),),(2,"b2"):(edge("b3",1),),
        (3,"b3"):(edge("goal",1),),
    }
    result = search.mean_full_solve("root",(0,1,2,3),lambda f,s:graph.get((f,s),()),
        lambda s:s=="goal",str,threshold=2)
    assert result["status"] == "solved"
    assert result["threshold_deferred"] == 1
    assert result["deferred_resumed"] == 0


def test_reopen_improved_context_and_propagate_lower_cost():
    expanded_same = []
    graph = {
        (0,"root"):(edge("a",0,0),edge("b",10,1)),
        (1,"a"):(edge("c",12),),(1,"b"):(edge("d",0),),
        (2,"c"):(edge("e",0),),(2,"d"):(edge("f",0),),
        (3,"e"):(edge("g",0),),(3,"f"):(edge("h",0),),
        (4,"g"):(edge("same",0),),(4,"h"):(edge("same",0),),
        (5,"same"):(edge("next",100),),
        (6,"next"):(edge("goal",0),),
    }
    def transitions(frame,state):
        if state=="same":expanded_same.append(state)
        return graph.get((frame,state),())
    result = search.mean_full_solve("root",tuple(range(7)),transitions,
        lambda s:s=="goal",str,threshold=1000)
    assert result["status"] == "solved"
    assert len(expanded_same) == 2
    assert result["reopened_states"] >= 1
    assert result["steps"][0]["state_score"] == 10
    assert result["steps"][-1]["path_total_cost"] == 110


def test_numeric_recent_history_preserves_different_gate_choices():
    graph = {
        (0,"root"):(edge("a",8,0),edge("b",3,1)),
        (1,"a"):(edge("a2",8),),(1,"b"):(edge("b2",3),),
        (2,"a2"):(edge("same",1),),(2,"b2"):(edge("same",1),),
        (3,"same"):(edge("goal",1),),
    }
    result = search.mean_full_solve("root",(0,1,2,3),lambda f,s:graph.get((f,s),()),
        lambda s:s=="goal",str,threshold=2)
    assert result["status"] == "solved"
    assert result["deferred_resumed"] == 0
    assert [s["state_score"] for s in result["steps"]] == [3,3,1,1]


@pytest.mark.parametrize("second",[("layer",1,1,1,(1,)),("layer",0,1,1,(-1,)),("layer",0,1,2,(1,))])
def test_gap_blocks_other_axis_reversal_or_width(second):
    first = ("layer",0,1,1,(1,))
    graph = {
        (0,0):(edge(1,1,move=("A",),key=first),),
        (1,1):(edge(2,1,0,("B",),second),edge(1,1,1)),
        (2,1):(edge(2,1,move=("B",),key=second),),
    }
    result = search.mean_full_solve(0,(0,1,2),lambda f,s:graph.get((f,s),()),
        lambda s:s==2,str,min_update_gap=2)
    assert result["spacing_pruned"] == 1
    assert [s["state_moves"] for s in result["steps"]] == [("A",),(),("B",)]


def test_double_turn_allows_two_adjacent_quarters_but_not_three():
    key = ("layer",0,1,1,(1,))
    graph = {
        (0,0):(edge(1,1,move=("A",),key=key),),
        (1,1):(edge(2,1,move=("A",),key=key),),
        (2,2):(edge(3,1,0,("A",),key),edge(2,1,1)),
        (3,2):(edge(3,1,move=("A",),key=key),),
    }
    result = search.mean_full_solve(0,(0,1,2,3),lambda f,s:graph.get((f,s),()),
        lambda s:s==3,str,min_update_gap=2)
    assert result["spacing_pruned"] == 1
    assert [s["state_moves"] for s in result["steps"]] == [("A",),("A",),(),("A",)]


def test_disk_queue_persists_pending_improvement_and_shared_path(tmp_path):
    path = tmp_path/"search.sqlite3"
    q = search.MeanQueue(path)
    node = search.MeanNode(1,"state",8,(None,None,0),recent=(8,))
    assert q.offer(b"key",node,pickle.dumps("state"),dict(state_moves=("A",))) == "new"
    node.total_cost = 6
    assert q.offer(b"key",node,pickle.dumps("state"),dict(state_moves=("B",))) == "improved_pending"
    node.total_cost = 9
    assert q.offer(b"key",node,pickle.dumps("state")) is None
    q.close()
    q = search.MeanQueue(path)
    try:
        assert q.pending_count == 1
        popped = q.pop(pickle.loads)
        assert popped.total_cost == 6
        assert q.steps(popped.path_id) == [dict(state_moves=("B",))]
        node.total_cost = 4
        assert q.offer(b"key",node,pickle.dumps("state")) == "reopened"
    finally:q.close()


def test_unsolved_endpoints_and_nonfinite_choices_do_not_stop_other_paths():
    choices = (edge("invalid",float("inf"),0),edge("invalid",float("nan"),1),
               edge("dead",1,2),edge("goal",2,3))
    result = search.mean_full_solve("root",(7,),lambda f,s:choices,
        lambda s:s=="goal",str)
    assert result["status"] == "solved"
    assert result["invalid_pruned"] == 2
    assert result["unsolved_endpoints"] == 1
    assert result["solved_frame_index"] == 7


def test_missing_frames_are_rejected():
    with pytest.raises(ValueError,match="consecutive"):
        search.mean_full_solve(0,(1,3),lambda f,s:(),bool,str)


def test_deferred_retry_uses_smallest_required_rolling_allowance():
    graph = {
        (0,"root"):(edge("a",1,0),edge("b",1,1)),
        (1,"a"):(edge("a2",9),),(2,"a2"):(edge("a3",9),),
        (1,"b"):(edge("b2",5),),(2,"b2"):(edge("b3",5),),
        (3,"a3"):(edge("goal",9),),(3,"b3"):(edge("goal",9),),
    }
    result = search.mean_full_solve("root",(0,1,2,3),lambda f,s:graph.get((f,s),()),
        lambda s:s=="goal",str)
    assert result["deferred_resumed"] == 1
    assert [s["state_score"] for s in result["steps"]] == [1,5,5,9]
    assert result["steps"][-1]["applied_threshold"] == pytest.approx(11/3)


def test_double_turn_streak_and_gap_survive_deferred_retries():
    a=("layer",0,1,1,(1,));b=("layer",1,1,1,(1,))
    graph = {
        (0,0):(edge(0,9),),
        (1,0):(edge(1,9,move=("A",),key=a),),
        (2,1):(edge(1,9),),
        (3,1):(edge(2,1,move=("B",),key=b),),
        (4,2):(edge(3,1,move=("B",),key=b),),
        (5,3):(edge(4,1,0,("B",),b),edge(3,1,1)),
        (6,3):(edge(4,1,move=("B",),key=b),),
    }
    result = search.mean_full_solve(0,tuple(range(7)),lambda f,s:graph.get((f,s),()),
        lambda s:s==4,str,min_update_gap=2)
    assert result["status"] == "solved"
    assert result["deferred_resumed"] == 2
    assert result["spacing_pruned"] == 1
    assert [s["state_moves"] for s in result["steps"]] == [(),("A",),(),("B",),("B",),(),("B",)]


@pytest.mark.parametrize("engine", [search.mean_full_solve, search.replay.adaptive_full_solve_dfs])
def test_early_filter_hook_preserves_deferred_choice_indices_and_double_turn_gap(engine):
    a=("layer",0,1,1,(1,));b=("layer",1,1,1,(1,))
    graph = {
        (0,0):(edge(0,9),),
        (1,0):(edge(1,9,move=("A",),key=a),),
        (2,1):(edge(1,9),),
        (3,1):(edge(2,1,move=("B",),key=b),),
        (4,2):(edge(99,1,0,("A",),a),edge(3,1,1,("B",),b)),
        (5,3):(edge(99,1,0,("B",),b),edge(3,1,1)),
        (6,3):(edge(4,1,move=("B",),key=b),),
    }
    class Early:
        def __call__(self, frame, state):
            raise AssertionError("Engine must supply update context before evaluation")

        def for_update_context(self, frame, state, update, gap):
            last, key, streak = update
            choices = graph.get((frame,state), ())
            kept = tuple(c for c in choices if not c.payload['state_moves'] or last is None
                         or frame-last >= gap
                         or (frame-last == 1 and streak < 2 and c.update_key == key))
            return kept, len(choices)-len(kept)

    result = engine(0,tuple(range(7)),Early(),lambda s:s==4,str,threshold=2,min_update_gap=2)
    assert result['status'] == 'solved'
    assert result['deferred_resumed'] >= 2
    assert result['spacing_pruned_before_scoring'] == 2
    assert result['spacing_pruned'] == 2
    assert [s['state_moves'] for s in result['steps']] == [(),('A',),(),('B',),('B',),(),('B',)]
