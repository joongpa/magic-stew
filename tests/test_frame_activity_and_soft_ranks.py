"""Legal-transition evidence and soft preference, independently of fit code."""
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('activity_search_tests', ROOT/'scripts/search_independent_keypoint_frame.py')
f = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = f
SPEC.loader.exec_module(f)


def edge(state, score=0., move=None, partial=0., layer=(0, 1, 1)):
    axis, side, width = layer
    payload = dict(state_moves=() if move is None else (move,),
                   layer_axis=axis, layer_side=side, layer_count=width,
                   layer_rotations=([dict(axis=axis, side=side, layer_count=width, angle_degrees=partial)]
                                    if partial else []))
    direction = ('layer', axis, side, width, (-1 if move.endswith("'") else 1,)) if move else None
    return f.replay.Transition(state, score, (score,), payload, direction)


def solve(graph, frames, *, activity=True, gap=1, queue=None, **kwargs):
    return f.frame_full_solve('root', frames, lambda frame, state: graph.get((frame, state), ()),
        lambda state: state == 'goal', str, require_layer_activity=activity,
        min_update_gap=gap, minimum_color_accuracy=None, queue=queue, **kwargs)


@pytest.mark.parametrize('backend', ['sqlite', 'buffered'])
def test_same_cube_with_different_activity_is_not_merged(backend):
    graph = {(0, 'root'): (edge('mid', .1, partial=3), edge('mid', 0., partial=3, layer=(1, 1, 1))),
             (1, 'mid'): (edge('goal', .05, move='R'),)}
    queue = f.storage.MeanQueue() if backend == 'sqlite' else f.storage.BufferedMeanQueue(frontier_cache_size=1, write_batch_size=1)
    try:
        result = solve(graph, (0, 1), queue=queue)
    finally:
        queue.close()
    assert result['status'] == 'solved'
    assert result['steps'][0]['state_score'] == .1
    assert result['steps'][1]['previous_layer_activity'] == ((0, 1, 1),)
    assert result['activity_pruned'] == 1


def test_direction_is_free_but_layer_and_width_must_match():
    graph = {(0, 'root'): (edge('mid', partial=3),),
             (1, 'mid'): (edge('goal', 0., move='U', layer=(1, 1, 1)),
                          edge('goal', 0., move='Rw', layer=(0, 1, 2)),
                          edge('goal', .1, move="R'"))}
    result = solve(graph, (0, 1))
    assert result['steps'][-1]['state_moves'] == ("R'",)
    assert result['activity_pruned'] == 2


def test_state_change_with_zero_partial_counts_for_double_turn():
    graph = {(0, 'root'): (edge('mid', partial=-3),),
             (1, 'mid'): (edge('turned', move='R'),),
             (2, 'turned'): (edge('goal', move='R'),)}
    result = solve(graph, (0, 1, 2), gap=2)
    assert result['status'] == 'solved'
    assert result['steps'][-1]['previous_layer_activity'] == ((0, 1, 1),)


def test_activity_does_not_bypass_move_spacing_direction_exception():
    graph = {(0, 'root'): (edge('mid', partial=3),),
             (1, 'mid'): (edge('turned', move='R'),),
             (2, 'turned'): (edge('goal', move="R'"),)}
    result = solve(graph, (0, 1, 2), gap=2)
    assert result['status'] == 'exhausted'
    assert result['spacing_pruned'] == 1


def test_an_aligned_hold_expires_activity_immediately():
    graph = {(0, 'root'): (edge('mid', partial=3),),
             (1, 'mid'): (edge('aligned'),),
             (2, 'aligned'): (edge('goal', move='R'),)}
    result = solve(graph, (0, 1, 2))
    assert result['status'] == 'exhausted'
    assert result['activity_pruned'] == 1


def test_current_partial_cannot_supply_both_required_frames():
    result = solve({(0, 'root'): (edge('goal', move='R', partial=21),)}, (0,))
    assert result['status'] == 'exhausted' and result['activity_pruned'] == 1


def test_zero_partial_is_inactive_and_unobserved_frames_keep_bridge_policy():
    assert f.activity.previous(dict(frame_index=0, **edge('x').payload), 1) == ()
    assert f.activity.previous(dict(frame_index=0, **edge('x', partial=3).payload), 2) == ()
    evidence = f.activity.previous(dict(frame_index=0, bridge_inferred=True), 1)
    assert evidence == f.activity.UNOBSERVED
    assert f.activity.permits(evidence, edge('x', move='U', layer=(1, 1, 1)).payload)


def test_activity_uses_transformed_physical_axes():
    # These are the physical B axes after pose rescue, not the source R axes.
    prior = dict(frame_index=8, orientation_symmetry_index=7,
                 **edge('mid', partial=3, layer=(2, -1, 1)).payload)
    evidence = f.activity.previous(prior, 9)
    assert f.activity.permits(evidence, edge('goal', move='B', layer=(2, -1, 1)).payload)
    assert not f.activity.permits(evidence, edge('goal', move='R').payload)


def test_outside_top_eight_is_interleaved_and_can_win():
    graph = {(0, 'root'): tuple(edge(f'p{i}', i*.01) for i in range(9)),
             (1, 'p0'): (edge('goal', .4),), (1, 'p8'): (edge('goal', .09),)}
    result = solve(graph, (0, 1), activity=False, soft_top_k=8, outside_top_k_penalty=.25)
    assert result['status'] == 'solved'
    assert result['steps'][0]['local_frame_rank'] == 9
    assert result['steps'][0]['queue_priority'] == pytest.approx(.33)
    assert result['steps'][1]['queue_priority'] == pytest.approx(.09)
    assert result['outside_top_k_offered'] == 1


def test_equal_scores_at_top_eight_boundary_receive_same_penalty():
    graph = {(0, 'root'): tuple(edge(f'p{i}', 0. if i < 7 else .1) for i in range(10)),
             (1, 'p9'): (edge('goal'),)}
    result = solve(graph, (0, 1), activity=False, soft_top_k=8)
    assert result['steps'][0]['local_frame_rank'] == 8
    assert result['steps'][0]['outside_top_k_penalty'] == 0.


def test_dominance_uses_penalized_priority_for_equivalent_states():
    graph = {(0, 'root'): (edge('a'), edge('b', .1)),
             (1, 'a'): tuple(edge(f'dead{i}', i*.01) for i in range(8))+(edge('x', .08),),
             (1, 'b'): (edge('x', .1),), (2, 'x'): (edge('goal', .1),)}
    result = solve(graph, (0, 1, 2), activity=False, soft_top_k=8)
    assert result['steps'][0]['state_score'] == .1  # Prefer b -> x over penalized a -> x.
    assert result['steps'][1]['queue_priority'] == .1
    assert result['improved_pending'] >= 1


def test_rank_penalty_does_not_reenable_color_pruning():
    e = edge('goal', .8)
    e.payload.update(color_error=1., color_forbidden_count=10)
    result = solve({(0, 'root'): (e,)}, (0,), activity=False, soft_top_k=8)
    assert result['status'] == 'solved' and result['color_pruned'] == 0


def test_default_search_has_no_hidden_percentage_color_cutoff():
    e = edge('goal', .8)
    e.payload.update(color_error=1., color_forbidden_count=10)
    result = f.frame_full_solve('root', (0,), lambda *_: (e,), lambda state: state == 'goal', str)
    assert result['status'] == 'solved' and result['color_pruned'] == 0


def test_partial_variants_count_as_one_successor_in_local_ranking():
    variants = tuple(edge('same', i*.01, partial=3 if i else 0) for i in range(10))
    graph = {(0, 'root'): variants+tuple(edge(f'p{i}', .1+i*.01) for i in range(9)),
             (1, 'p6'): (edge('goal'),)}
    result = solve(graph, (0, 1), activity=False, soft_top_k=8)
    assert result['steps'][0]['local_frame_rank'] == 8
    assert result['steps'][0]['local_candidate_count'] == 10
    assert result['steps'][0]['outside_top_k_penalty'] == 0


def test_rank_is_assigned_before_activity_gate():
    wrong = tuple(edge(f'u{i}', i*.01, move='U', layer=(1, 1, 1)) for i in range(8))
    graph = {(0, 'root'): (edge('mid', partial=3),),
             (1, 'mid'): wrong+(edge('goal', .08, move='R'),)}
    result = solve(graph, (0, 1), soft_top_k=8)
    assert result['steps'][-1]['local_frame_rank'] == 9
    assert result['steps'][-1]['outside_top_k_penalty'] == .25
    assert result['activity_pruned'] == 8
