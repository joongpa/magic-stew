import importlib.util
from pathlib import Path
import sys
import random

import pytest


SPEC = importlib.util.spec_from_file_location("full_solve_search", Path(__file__).resolve().parents[1] / "scripts/search_independent_keypoint_full_solve.py")
search = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = search
SPEC.loader.exec_module(search)


def transition(state, score, rank=0):
    return search.Transition(state, score, (rank,), {"state_moves": ()})


def test_second_best_tolerates_two_outliers_but_is_not_always_below_trimmed_mean():
    assert search.second_best_four((1., 2., 80., 100.)) == 2
    assert search.second_best_four((1., 8., 9., 10.)) == 8
    assert search.second_best_four((100., 100., 100.)) is None


def test_best_three_mean_discards_one_outlier_and_uses_only_last_four_frames():
    assert search.best_three_mean_four((1., 2., 3., 100.)) == 2
    assert search.best_three_mean_four((1., 2., 80., 100.)) == pytest.approx(83 / 3)
    assert search.best_three_mean_four((999., 1., 2., 3., 100.)) == 2
    assert search.best_three_mean_four((1., 2., 3.)) is None


def test_ranked_search_backtracks_after_rolling_failure_and_reports_monotonic_furthest():
    events = []
    graph = {
        (10, "root"): (transition("bad", 1, 0), transition("good", 1, 1)),
        (11, "bad"): (transition("bad2", 20),),
        (12, "bad2"): (transition("bad3", 20),),
        (13, "bad3"): (transition("solved", 20),),
        (11, "good"): (transition("good2", 1),),
        (12, "good2"): (transition("good3", 1),),
        (13, "good3"): (transition("solved", 1),),
    }
    result = search.full_solve_dfs("root", (10, 11, 12, 13),
        lambda frame, state: graph.get((frame, state), ()),
        lambda state: state == "solved", lambda state: state,
        threshold=10, progress=events.append)
    assert result["status"] == "solved"
    assert result["threshold_pruned"] == 1
    assert result["backtracks"] > 0
    assert result["solved_frame_index"] == 13
    assert [e["furthest_frames"] for e in events] == sorted(e["furthest_frames"] for e in events)
    assert events[-1]["furthest_frames"] == 4


def test_reconverging_state_keeps_distinct_rolling_threshold_history():
    graph = {
        (0, "root"): (transition("bad", 8, 0), transition("good", 3, 1)),
        (1, "bad"): (transition("bad2", 8),),
        (1, "good"): (transition("good2", 3),),
        (2, "bad2"): (transition("same", 1),),
        (2, "good2"): (transition("same", 1),),
        (3, "same"): (transition("solved", 1),),
    }
    result = search.full_solve_dfs("root", (0, 1, 2, 3),
        lambda frame, state: graph.get((frame, state), ()),
        lambda state: state == "solved", lambda state: state, threshold=2)
    assert result["status"] == "solved"
    assert result["threshold_pruned"] == 1


def test_all_ranked_choices_remain_eligible_and_unsolved_end_of_video_backtracks():
    result = search.full_solve_dfs(0, (7,),
        lambda frame, state: tuple(transition(i, 1, i) for i in range(1, 14)),
        lambda state: state == 13, lambda state: state)
    assert result["status"] == "solved"
    assert result["attempted_transitions"] == 13
    assert result["unsolved_endpoints"] == 12


def test_solved_goal_can_stop_before_video_end():
    result = search.full_solve_dfs(0, (7, 8, 9),
        lambda frame, state: (transition(1, 1),),
        lambda state: state == 1, lambda state: state)
    assert result["solved_frame_index"] == 7
    assert len(result["steps"]) == 1


def test_exhausted_disk_memo_can_be_reused(tmp_path):
    memo = search.DiskMemo(tmp_path / "failed.sqlite3")
    try:
        run = lambda: search.full_solve_dfs(0, (7,), lambda frame, state: (),
            lambda state: False, lambda state: state, failed=memo)
        assert run()["status"] == "exhausted"
        assert run()["memo_pruned"] == 1
    finally:
        memo.close()


def test_rejects_missing_source_frames():
    with pytest.raises(ValueError, match="consecutive"):
        search.full_solve_dfs(0, (1, 3), lambda frame, state: (), bool, str)


def test_adaptive_retries_cheapest_blocked_continuation_at_exact_required_score():
    events = []
    graph = {
        (0,"root"): (transition("a",1,0), transition("b",1,1)),
        (1,"a"): (transition("a2",9),), (2,"a2"): (transition("same",9),),
        (1,"b"): (transition("b2",5),), (2,"b2"): (transition("same",5),),
        (3,"same"): (transition("solved",9),),
    }
    result = search.adaptive_full_solve_dfs("root",(0,1,2,3),
        lambda f,s:graph.get((f,s),()),lambda s:s=="solved",str,threshold=2,progress=events.append)
    assert result["status"]=="solved"
    assert result["steps"][1]["state_score"]==5
    assert result["steps"][-1]["applied_threshold"]==pytest.approx(11/3)
    assert result["deferred_resumed"]==1
    assert [e["temporary_threshold"] for e in events if e["event"]=="temporary_allowance"]==pytest.approx([11/3])


def test_adaptive_allowance_returns_to_normal_on_the_next_frame():
    scores = (1,4,4,4,1,1)
    events = []
    result = search.adaptive_full_solve_dfs(0,tuple(range(6)),
        lambda f,s:(transition(s+1,scores[f]),),lambda s:s==6,str,
        threshold=2,progress=events.append)
    assert result["status"]=="solved"
    assert result["deferred_resumed"]==2
    assert [s["applied_threshold"] for s in result["steps"]]==[2,2,2,3,3,2]
    assert result["largest_temporary_threshold"]==3
    assert all(e["return_threshold_next_frame"]==2 for e in events if e["event"]=="temporary_allowance")


def test_adaptive_finishes_normal_work_before_retrying_blocked_work():
    graph = {
        (0,"root"): (transition("a",1,0),transition("b",1,1)),
        (1,"a"): (transition("a2",8),), (2,"a2"): (transition("a3",8),),
        (3,"a3"): (transition("solved",8),),
        (1,"b"): (transition("b2",1),), (2,"b2"): (transition("b3",1),),
        (3,"b3"): (transition("solved",1),),
    }
    result = search.adaptive_full_solve_dfs("root",(0,1,2,3),
        lambda f,s:graph.get((f,s),()),lambda s:s=="solved",str,threshold=2)
    assert result["status"]=="solved"
    assert result["threshold_deferred"]==1
    assert result["deferred_resumed"]==0
    assert all(s["applied_threshold"]==2 for s in result["steps"])


def test_adaptive_keeps_rejected_siblings_after_first_retry_fails(tmp_path):
    queue = search.DeferredBranches(tmp_path/"deferred.sqlite3")
    graph = {
        (0,0): (transition(1,8),), (1,1): (transition(2,8),),
        (2,2): (transition(3,8),),
        (3,3): (transition(4,3,0), transition(5,4,1)),
    }
    try:
        result = search.adaptive_full_solve_dfs(0,(0,1,2,3),lambda f,s:graph.get((f,s),()),
            lambda s:s==5,str,threshold=2,deferred=queue)
        assert result["status"]=="solved"
        assert result["deferred_resumed"]==2
        assert result["unsolved_endpoints"]==1
        assert len(result["steps"])==4
        assert result["steps"][-1]["state_score"]==4
    finally:
        queue.close()


def test_adaptive_exhausts_unsolvable_graph_after_retrying_all_blocked_work():
    result = search.adaptive_full_solve_dfs(0,(0,1,2,3),
        lambda f,s:(transition(s+1,8),),lambda s:False,str,threshold=2)
    assert result["status"]=="exhausted"
    assert result["threshold_deferred"]==1
    assert result["deferred_resumed"]==1
    assert result["deferred_pending"]==0


def test_adaptive_retains_numeric_history_for_deferred_priority():
    graph = {
        (0,"root"): (transition("a",7,0),transition("b",3,1)),
        (1,"a"): (transition("a2",7),), (2,"a2"): (transition("same",7),),
        (1,"b"): (transition("b2",3),), (2,"b2"): (transition("same",3),),
        (3,"same"): (transition("solved",1),),
    }
    result = search.adaptive_full_solve_dfs("root",(0,1,2,3),
        lambda f,s:graph.get((f,s),()),lambda s:s=="solved",str,threshold=2)
    assert result["status"]=="solved"
    assert result["steps"][-1]["applied_threshold"]==pytest.approx(7/3)
    assert result["largest_temporary_threshold"]==pytest.approx(7/3)


def test_adaptive_matches_exhaustive_reachability_and_returns_a_legal_path():
    rng = random.Random(1907)
    for case in range(12):
        graph = {}
        for frame in range(5):
            for state_id in range(4):
                choices = []
                for index in range(rng.randrange(4)):
                    score = rng.choice((.2,1.,2.,2.0001,3.7,8.))
                    trial_id = f"{frame}:{state_id}:{index}"
                    choices.append(search.Transition((frame+1,rng.randrange(4)),score,(score,index),
                        dict(state_moves=(),trial_id=trial_id)))
                graph[frame,state_id] = tuple(choices)
        reachable = {0}
        for frame in range(5):
            reachable = {t.state[1] for state_id in reachable for t in graph[frame,state_id]}
        result = search.adaptive_full_solve_dfs((0,0),tuple(range(5)),
            lambda f,s:graph[f,s[1]],lambda s:s==(5,3),lambda s:s,threshold=2)
        assert (result["status"]=="solved") == (3 in reachable), case
        if result["status"]=="solved":
            state = (0,0)
            scores = []
            for frame,step in enumerate(result["steps"]):
                choice = next(t for t in graph[frame,state[1]] if t.payload["trial_id"]==step["trial_id"])
                assert choice.score==step["state_score"]
                state = choice.state
                scores.append(choice.score)
                statistic = search.best_three_mean_four(tuple(scores))
                assert step["applied_threshold"]==max(2,statistic if statistic is not None else 2)
            assert state==(5,3)


def moving(state, move, key, score=1., rank=0):
    return search.Transition(state,score,(rank,),dict(state_moves=(move,)),key)


@pytest.mark.parametrize("engine",[search.full_solve_dfs,search.adaptive_full_solve_dfs])
@pytest.mark.parametrize("second_key",[
    ("layer",1,1,1,(1,)),  # Different axis.
    ("layer",0,1,1,(-1,)),  # Reversal on the same layer.
    ("layer",0,1,2,(1,)),  # Different layer width.
])
def test_gap_blocks_consecutive_updates_unless_same_layer_and_direction(engine,second_key):
    first_key=("layer",0,1,1,(1,))
    graph={
        (0,0):(moving(1,"A",first_key),),
        (1,1):(moving(2,"B",second_key,rank=0),transition(1,1,1)),
        (2,1):(moving(2,"B",second_key),),
    }
    result=engine(0,(0,1,2),lambda f,s:graph.get((f,s),()),lambda s:s==2,str,
                  threshold=2,min_update_gap=2)
    assert result["status"]=="solved"
    assert result["spacing_pruned"]==1
    assert [s["state_moves"] for s in result["steps"]]==[("A",),(),("B",)]


@pytest.mark.parametrize("engine",[search.full_solve_dfs,search.adaptive_full_solve_dfs])
def test_double_turn_allows_two_consecutive_quarters_but_requires_gap_before_third(engine):
    key=("layer",0,1,1,(1,))
    graph={
        (0,0):(moving(1,"A",key),),
        (1,1):(moving(2,"A",key),),
        (2,2):(moving(3,"A",key,rank=0),transition(2,1,1)),
        (3,2):(moving(3,"A",key),),
    }
    result=engine(0,(0,1,2,3),lambda f,s:graph.get((f,s),()),lambda s:s==3,str,
                  threshold=2,min_update_gap=2)
    assert result["status"]=="solved"
    assert result["spacing_pruned"]==1
    assert [s["state_moves"] for s in result["steps"]]==[("A",),("A",),(),("A",)]


def test_deferred_retry_preserves_gap_and_double_turn_context():
    a=("layer",0,1,1,(1,));b=("layer",1,1,1,(1,));c=("layer",2,1,1,(1,))
    graph={
        (0,0):(transition(0,9),),
        (1,0):(moving(1,"A",a,score=9),),
        (2,1):(transition(1,9),),
        (3,1):(moving(2,"B",b),),
        (4,2):(moving(3,"C",c,rank=0),transition(2,1,1)),
        (5,2):(moving(3,"C",c),),
    }
    result=search.adaptive_full_solve_dfs(0,tuple(range(6)),
        lambda f,s:graph.get((f,s),()),lambda s:s==3,str,threshold=2,min_update_gap=2)
    assert result["status"]=="solved"
    assert result["deferred_resumed"]==2
    assert result["spacing_pruned"]==1
    assert [s["state_moves"] for s in result["steps"]]==[(),("A",),(),("B",),(),("C",)]


def test_parser_uses_fps_scaled_gap_by_default():
    args=search.build_parser().parse_args([
        "--log","source.jsonl","--initial-scramble","R",
        "--start-frame","0","--end-frame","10","--output-dir","output"])
    assert args.fps==60
    assert args.min_state_update_gap is None
    assert args.scorer=="auto"


def test_double_turn_streak_survives_deferred_retry():
    a=("layer",0,1,1,(1,));b=("layer",1,1,1,(1,))
    graph={
        (0,0):(transition(0,9),),
        (1,0):(moving(1,"A",a,score=9),),
        (2,1):(transition(1,9),),
        (3,1):(moving(2,"B",b),),
        (4,2):(moving(3,"B",b),),
        (5,3):(moving(4,"B",b,rank=0),transition(3,1,1)),
        (6,3):(moving(4,"B",b),),
    }
    result=search.adaptive_full_solve_dfs(0,tuple(range(7)),
        lambda f,s:graph.get((f,s),()),lambda s:s==4,str,threshold=2,min_update_gap=2)
    assert result["status"]=="solved"
    assert result["deferred_resumed"]==2
    assert result["spacing_pruned"]==1
    assert [s["state_moves"] for s in result["steps"]]==[(),("A",),(),("B",),("B",),(),("B",)]


@pytest.mark.parametrize("engine",[search.full_solve_dfs,search.adaptive_full_solve_dfs])
def test_two_outliers_block_a_path_that_second_best_would_admit(engine):
    scores=(1.,2.,80.,100.)
    result=engine(0,tuple(range(4)),lambda f,s:(transition(s+1,scores[f]),),
        lambda s:s==4,str,threshold=2)
    if engine is search.full_solve_dfs:
        assert result["status"]=="exhausted"
        assert result["threshold_pruned"]==1
    else:
        assert result["status"]=="solved"
        assert result["deferred_resumed"]==1
        assert result["steps"][-1]["best_three_mean_four"]==pytest.approx(83/3)
        assert result["steps"][-1]["applied_threshold"]==pytest.approx(83/3)
