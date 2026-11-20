from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import independent_keypoint_search_fast as fast


@pytest.mark.parametrize('observations', [0, 1, 31])
def test_native_color_batch_matches_duplicate_sticker_reference(observations):
    if fast.native is None:
        pytest.skip('optional native kernels are not built')
    from render_yolo11n_independent_keypoints import sticker_color_confusion_metrics
    rng = np.random.default_rng(984)
    states = np.stack([rng.permutation(54) for _ in range(17)])
    # Multiple observations deliberately vote for the same physical sticker.
    nearest = rng.integers(0, 9, (23, observations), dtype=np.uint8)
    valid = rng.random(nearest.shape) > .2
    valid[0] = False
    costs = rng.choice([0., .5, 1., np.nan], (54, observations))
    class Renderer:
        pass
    renderer = Renderer()
    renderer.sticker_color_confusion_metrics = sticker_color_confusion_metrics
    expected = sticker_color_confusion_metrics(states, nearest, valid, costs, .37)
    actual = fast.color_metrics(renderer, states, nearest, valid, costs, .37)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_native_grouped_ranking_preserves_all_tie_breaks_and_nan_order():
    if fast.native is None:
        pytest.skip('optional native kernels are not built')
    rng = np.random.default_rng(90)
    for _ in range(50):
        count = 120
        selection = dict(group_ids=rng.integers(0, 15, count),
            inliers=rng.integers(0, 5, count).astype(float),
            p90s=rng.choice([0., .1, np.nan, np.inf], count),
            layer_inliers=rng.integers(0, 3, count).astype(float))
        selection.update(rank_groups=np.unique(selection['group_ids'], return_inverse=True)[1].astype(np.intp),
            rank_inliers=selection['inliers'], rank_layer_inliers=selection['layer_inliers'])
        costs = rng.choice([0., .1, .2, np.inf, np.nan], count)
        actual = fast.best_rows(costs, selection)
        eligible = np.flatnonzero(np.isfinite(costs))
        order = eligible[np.lexsort((-selection['layer_inliers'][eligible], selection['p90s'][eligible],
            -selection['inliers'][eligible], costs[eligible], selection['group_ids'][eligible]))]
        groups = selection['group_ids'][order]
        expected = np.sort(order[np.r_[True, groups[1:] != groups[:-1]]])
        np.testing.assert_array_equal(actual, expected)


def test_batched_permutations_are_exact_and_validate_buffers():
    state = bytes(range(54))
    sources = np.stack([np.arange(54), np.arange(53, -1, -1)]).astype(np.intp)
    assert fast.permute_many(state, sources) == [state, state[::-1]]
    assert fast.permute(state, sources[1]) == state[::-1]
    if fast.native is not None:
        invalid = sources.copy(); invalid[1, 2] = 54
        with pytest.raises(ValueError, match='out of range'):
            fast.native.permute_many(state, invalid)
        with pytest.raises(ValueError):
            fast.native.permute(state, sources[0].astype(np.float64))
        with pytest.raises(ValueError):
            fast.native.color_metrics(np.zeros((2, 53), np.uint8),
                np.zeros((1, 1), np.uint8), np.ones((1, 1), bool), np.zeros((54, 1)), .5)


def test_diagnostics_are_deferred_until_the_record_is_exported():
    calls = []
    class Renderer:
        @staticmethod
        def balanced_robust_cost(robust, color):
            calls.append((robust, color))
            return .75
    payload = fast.ScoredPayload(dict(state_moves=('R',), label='turn'), Renderer(), .2, True, .1,
                                color_error=.3, state_score=.2)
    step = fast.PayloadOverlay(payload, frame_index=7)
    assert step['state_moves'] == ('R',)
    assert step.get('handoff_commit') is None
    assert not calls
    exported = dict(step)
    assert calls == [(.2, .3)]
    assert exported['global_robust_error'] == .75
    assert exported['score_components']['geometry'] == .1
    assert dict(step) == exported and len(calls) == 1


@pytest.mark.parametrize('threshold,count', [(None, 3), (3, 5), (2, 3)])
def test_row_admission_keeps_existing_ties_alias_rules_and_sparse_branch(threshold, count):
    # Different moves can reach the same state/context; duplicate moves can
    # have different residuals. Neither should consume another sibling slot.
    records = [((score,), moves, state, 3) for score, moves, state in (
        (.1, ('R',), ('a', 0)), (.11, ('R',), ('a', 1)),
        (.12, ('alias',), ('a', 0)), (.2, ('U',), ('b', 0)),
        (.2, ('F',), ('c', 0)), (.3, ('L',), ('d', 0)), (.4, ('B',), ('e', 0)))]
    created = []
    def materialize(index):
        created.append(index)
        return index
    batch = fast.ExpansionBatch(records, materialize)
    selected, unrestricted = batch.select(2, threshold)
    assert selected == ([0, 3, 4, 5, 6] if count == 5 else [0, 3, 4])
    assert created == selected
    assert unrestricted == (count == 5)
