import importlib.util
import json
from pathlib import Path
import pickle
import sys
from dataclasses import replace

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    'bridge_tests', Path(__file__).resolve().parents[1]/'scripts/search_independent_keypoint_frame.py')
f = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = f
SPEC.loader.exec_module(f)


@pytest.fixture(scope='module')
def m():
    torch = pytest.importorskip('torch')
    torch.set_num_threads(2)
    return f.replay.load_renderer()


def record(weak=False):
    return dict(search_pose=dict(rvec=[[0.], [0.], [0.]],
        assignment_image_points=[[0., 0.]]*4), search=dict(geometry=dict(options=[dict(
        keypoint_mean_error=.08 if weak else .01, heatmap_mean_error=.04 if weak else .01,
        reprojection_fit_error=.12 if weak else .02, keypoint_inlier_count=4)])))


class Scorer:
    def __init__(self, m, weak, *, target=None, reject=False):
        self.m, self.target, self.reject = m, target, reject
        self.records = {i: record(i in weak) for i in range(5)}
        self.calls = []

    def hold_reason(self, frame):
        return None

    @staticmethod
    def key(state):
        return pickle.dumps(state.signature())

    @staticmethod
    def apply_moves(state, moves):
        return state.apply_moves(moves)

    def __call__(self, frame, state):
        self.calls.append(frame)
        correct = self.target is None or state.signature() == self.target.signature()
        return (f.replay.Transition(state, .2, (.2,), dict(state_moves=(),
                color_error=0. if correct and not self.reject or frame == 0 else .9)),)


def bridge(scorer, frames, **kwargs):
    return f.FrameObservationBridge(scorer, frames, fps=60., advance_update=f.advance_update,
        canonical_update=f.canonical_update, transition_type=f.replay.Transition, **kwargs)


def test_quality_uses_display_fit_and_requires_one_jointly_supported_geometry(m):
    s = Scorer(m, {1})
    s.records[2]['search']['geometry']['options'] = [
        dict(reprojection_fit_error=.001, keypoint_inlier_count=0),
        dict(reprojection_fit_error=.11, keypoint_inlier_count=4)]
    b = bridge(s, (0, 1, 2, 3))
    assert b.quality[0]['reliable'] and not b.quality[1]['reliable']
    assert not b.quality[2]['reliable']  # Cost and coverage cannot come from different fits.
    assert s.calls == []  # No state/color scoring participates in the quality decision.


def test_display_fit_cutoff_is_inclusive_and_independent_of_balanced_cost(m):
    s = Scorer(m, ())
    for frame, fit in enumerate((.1, .100001, None, float('inf'))):
        s.records[frame]['search']['geometry']['options'][0].update(
            reprojection_fit_error=fit, keypoint_mean_error=0., heatmap_mean_error=0.)
    b = bridge(s, tuple(range(4)))
    assert [b.quality[i]['reliable'] for i in range(4)] == [True, False, False, False]
    assert b.quality[0]['best_reprojection_fit'] == .1
    assert s.calls == []


def test_two_weak_frames_keep_exact_state_and_timing_contexts_and_respect_caps(m):
    s = Scorer(m, {1, 2})
    b = bridge(s, (1, 2, 3))
    initial = m.CubeGeometryState.identity()
    end, parents = b.expand(0, initial, (None, None, 0), 2, s.key, lambda: None)
    assert end == 2 and len(parents) == 73
    assert s.calls == []
    assert all(len(steps) == 2 and all(v['state_score'] is None for v in steps)
               for _state, _update, steps in parents)
    assert bridge(s, (1, 2, 3), max_frames=1).endpoint(0) == 1
    assert bridge(s, (1, 2, 3), max_seconds=.01).endpoint(0) == 1
    s.records[3] = record(True)
    assert bridge(s, (1, 2, 3, 4)).endpoint(0) == 2


@pytest.mark.parametrize('backend', ['buffered', 'sqlite'])
@pytest.mark.parametrize('reject', [False, True])
def test_bridge_records_moves_and_null_scores_with_optional_color_cutoff(m, backend, reject):
    initial = m.CubeGeometryState.identity()
    target = initial.apply_move('R')
    s = Scorer(m, {1}, target=target, reject=reject)
    q = (f.storage.BufferedMeanQueue(frontier_cache_size=1) if backend == 'buffered'
         else f.storage.MeanQueue())
    try:
        result = f.frame_full_solve(initial, (0, 1, 2), s,
            lambda state: state.signature() == target.signature(), s.key,
            queue=q, min_update_gap=2, bridge=bridge(s, (0, 1, 2)), minimum_color_accuracy=2./3.)
    finally:
        q.close()
    assert 1 not in s.calls
    if reject:
        assert result['status'] == 'exhausted'
        assert result['color_pruned'] > 0
    else:
        assert result['status'] == 'solved' and result['solved_frame_index'] == 2
        assert [v['frame_index'] for v in result['steps']] == [0, 1, 2]
        assert result['steps'][1]['state_moves'] == ('R',)
        assert result['steps'][1]['state_score'] is None
        assert result['steps'][-1]['queue_priority'] == .2


def test_real_incoming_move_restricts_a_single_bad_frame_to_two_choices(m):
    s = Scorer(m, {1})
    b = bridge(s, (1, 2))
    payload = dict(state_moves=("U'",), layer_axis=2, layer_side=1, layer_count=1)
    update = (0, m.state_update_layer_direction(payload), 1)
    _, parents = b.expand(0, m.CubeGeometryState.identity(), update, 2, s.key, lambda: None)
    assert {steps[0]['state_moves'] for _, _, steps in parents} == {(), ("U'",)}


def test_terminal_bad_gap_never_uses_observations_or_confirms_a_goal(m):
    initial = m.CubeGeometryState.identity()
    target = initial.apply_move('R')

    s = Scorer(m, {1})
    result = f.frame_full_solve(initial, (0, 1), s,
        lambda state: state.signature() == target.signature(), s.key,
        bridge=bridge(s, (0, 1)))
    assert result['status'] == 'exhausted'
    assert result['unsolved_endpoints'] == 25
    assert s.calls == [0]


@pytest.mark.parametrize('backend', ['buffered', 'sqlite'])
def test_long_bad_gap_is_transition_only_across_multiple_batches(m, backend):
    initial = m.CubeGeometryState.identity()
    target = initial.apply_move('R')
    s = Scorer(m, {1, 2, 3}, target=target)
    q = (f.storage.BufferedMeanQueue(frontier_cache_size=1) if backend == 'buffered'
         else f.storage.MeanQueue())
    try:
        result = f.frame_full_solve(initial, tuple(range(5)), s,
            lambda state: state.signature() == target.signature(), s.key, queue=q,
            bridge=bridge(s, tuple(range(5))))
    finally:
        q.close()
    assert result['status'] == 'solved' and result['solved_frame_index'] == 4
    assert set(s.calls) == {0, 4}
    assert [v['frame_index'] for v in result['steps']] == list(range(5))
    assert all(v['state_score'] is None and v['queue_priority'] is None
               for v in result['steps'][1:4])
    assert result['steps'][2]['inherited_queue_priority'] == .2
    assert result['steps'][4]['state_score'] == .2


def test_old_cache_recovers_the_same_display_metric_as_materialized_geometry(m):
    data = json.loads((Path(__file__).parent/'fixtures/xuanyi_frame_231_whole_cube_scoring.json').read_text())
    p = data['pose']
    fixed = np.asarray(p['assignment_fixed_mask'], dtype=bool)
    observed = np.asarray(p['assignment_image_points'])
    projected = np.asarray(p['assignment_projected_points'])
    c = m.StickerPnPCandidate(sticker_type=p['sticker_type'], sticker_average_confidence=1.,
        image_points=observed[fixed], object_points=np.asarray(data['seed_object_points']),
        projected_sticker_points=projected[fixed], reprojection_error=0.,
        camera_matrix=np.asarray(p['camera_matrix']), dist_coeffs=np.asarray(p['dist_coeffs']),
        rvec=np.asarray(p['rvec']), tvec=np.asarray(p['tvec']), image_size=tuple(p['image_size']),
        assignment_image_points=observed, assignment_projected_points=projected,
        assignment_object_points=np.asarray(data['assignment_object_points']),
        assignment_observed_class_ids=np.asarray(p['assignment_observed_class_ids']),
        assignment_projected_class_ids=np.asarray(p['assignment_observed_class_ids']),
        assignment_fixed_mask=fixed)
    geometry = m._build_turn_geometry_frame(c,
        m.PNP_FINAL_ASSIGNMENT_ERROR_FRACTION*m.projected_cube_diameter(c), 'cpu')
    assert geometry is not None
    options = [m.turn_trial_log_payload(o) for o in geometry['options']]
    expected = [m.pose_reprojection_fit_metric(replace(c,
        assignment_projected_points=o['assignment_projected_points'])) for o in geometry['options']]
    assert [o['reprojection_fit_error'] for o in options] == pytest.approx(expected)
    old_options = [{k: v for k, v in o.items() if k != 'reprojection_fit_error'} for o in options]
    recovered = f.bridge_module.reprojection_options(m,
        dict(search_pose=p, selected_pose=p, search=dict(geometry=dict(options=old_options))))
    assert [o['reprojection_fit_error'] for o in recovered] == pytest.approx(expected, abs=1e-7)
