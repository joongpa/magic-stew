from dataclasses import dataclass
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from search_independent_keypoint_bidirectional import Link, bidirectional_search, replay, widening_join
from independent_keypoint_window_forecast import WindowForecaster


@dataclass
class Rotation:
    axis: int
    side: int
    angle_degrees: float
    layer_count: int


class Renderer:
    CubeLayerRotation = Rotation

    @staticmethod
    def cube_rotation_to_notation(rotation):
        if (rotation.axis, rotation.side, rotation.layer_count) == (0, 1, 1):
            return "R" if rotation.angle_degrees > 0 else "R'"
        return None


class Scorer:
    m = Renderer()

    @staticmethod
    def apply_moves(state, moves):
        delta = sum(1 if move == "R" else -1 for move in moves)
        return bytes([(state[0]+delta) % 4])

    @staticmethod
    def hold_reason(frame):
        return None

    def __call__(self, frame, state):
        if frame == 0 and state == b"\x00":
            return (
                replay.Transition(b"\x03", 0., (0.,), dict(state_moves=("R'",), color_error=0.), None),
                replay.Transition(state, .1, (.1,), dict(state_moves=(), color_error=.2,
                    layer_inlier_count=4, layer_rotations=[
                        dict(axis=0, side=1, angle_degrees=27., layer_count=1)]), None),
            )
        if frame == 1 and state == b"\x01":
            return (replay.Transition(state, 0., (0.,), dict(state_moves=(),
                color_error=0., geometry_option_index=1, label="flat"), None),)
        return ()

    def score_hold(self, frame, state):
        return next((choice for choice in self(frame, state) if not choice.payload["state_moves"]), None)

    def _evaluate(self, frame, state, direction):
        return self(frame, state)


@pytest.mark.parametrize("retries,expected", [(0, "beam_exhausted"), (1, "solved")])
def test_deferred_prefix_can_join_a_suffix_that_requires_a_handoff(retries, expected):
    result = bidirectional_search(Scorer(), (0, 1), b"\x00", (b"\x01",), width=1,
        retries=retries, overlap=0, time_limit=5, state_limit=20, handoff_bonus=.02,
        progress=lambda row: None)
    assert result["status"] == expected
    if retries:
        assert result["retries"] == 1
        assert result["join_depth"] == 1
        assert [step["state_moves"] for step in result["steps"]] == [(), ("R",)]
        assert result["steps"][1]["handoff_commit"]
        assert result["steps"][1]["queue_priority"] == -.02


def test_independent_widening_releases_prefix_and_verifies_handoff():
    events = []
    result = bidirectional_search(Scorer(), (0, 1), b"\x00", (b"\x01",), width=1,
        retries=0, overlap=0, time_limit=5, state_limit=20, handoff_bonus=.02,
        branch_widths=(1, 2), branch_stage_expansions=1, progress=events.append)

    assert result["status"] == "solved"
    assert result["join_depth"] == 1
    assert [step["state_moves"] for step in result["steps"]] == [(), ("R",)]
    assert result["steps"][0]["sibling_rank"] == 2
    assert result["steps"][1]["handoff_commit"]
    widened = [row for row in events if row["event"] == "widened"]
    assert any(row["direction"] == "forward" and row["forward_expanded"] == 1 for row in widened)
    assert all(row[row["direction"] + "_expanded"] == 1 for row in widened)


def test_geometry_shortlist_keeps_a_handoff_missing_from_current_frame_actions():
    class GeometryRenderer(Renderer):
        @staticmethod
        def parallel_turn_actions(axis):
            return {('R',): (0., 0., 90.), ("R'",): (0., 0., -90.)} if axis == 0 else {}

    class GeometryScorer(Scorer):
        m = GeometryRenderer()
        geometry_shortlist_size = 3

        @staticmethod
        def frame_actions(frame):
            return frozenset({('R',)}) if frame == 0 else frozenset()

        def score_action(self, frame, state, action):
            assert action in self.frame_actions(frame)
            return tuple(choice for choice in self(frame, state) if choice.payload['state_moves'] == action)

    result = bidirectional_search(GeometryScorer(), (0, 1), b'\x00', (b'\x01',), width=1,
        retries=0, overlap=0, time_limit=5, state_limit=20, handoff_bonus=.02,
        branch_widths=(1, 2), branch_stage_expansions=1, progress=lambda row: None)
    assert result['status'] == 'solved'
    assert result['steps'][1]['handoff_commit']
    assert result['steps'][1]['state_moves'] == ('R',)


@pytest.mark.parametrize('widths', [(), (3,)])
def test_reverse_search_and_join_compose_a_handoff_with_the_current_turn(widths):
    m = replay.load_renderer()

    class ComposedScorer:
        parallel_layers = True
        geometry_shortlist_size = 5

        @staticmethod
        def hold_reason(frame):
            return None

        @staticmethod
        def frame_actions(frame):
            return frozenset({('F',), ("F'",)}) if frame == 0 else frozenset({('D',), ("D'",)})

        @staticmethod
        def apply_moves(state, moves):
            value = list(state)
            for move in moves:
                axis = 0 if move.startswith('F') else 1
                value[axis] = (value[axis] + (-1 if move.endswith("'") else 1)) % 4
            return bytes(value)

        def __call__(self, frame, state):
            if frame == 0 and state == b'\0\0':
                return (replay.Transition(state, 0., (0.,), dict(state_moves=(), color_error=0.,
                    layer_inlier_count=3, geometry_option_index=0, label='partial F',
                    layer_rotations=[dict(axis=1, side=-1, angle_degrees=-30., layer_count=1)])),)
            if frame == 1 and state == b'\1\0':
                return (replay.Transition(b'\1\1', .1, (.1,), dict(state_moves=('D',), color_error=0.,
                    layer_inlier_count=3, geometry_option_index=1, label='partial D', color_offset=-90.,
                    layer_rotations=[dict(axis=2, side=-1, angle_degrees=33., layer_count=1)])),)
            return ()

        def score_hold(self, frame, state):
            return next((choice for choice in self(frame, state) if not choice.payload['state_moves']), None)

        def score_action(self, frame, state, action):
            return tuple(choice for choice in self(frame, state) if choice.payload['state_moves'] == action)

    scorer = ComposedScorer()
    scorer.m = m
    result = bidirectional_search(scorer, (0, 1), b'\0\0', (b'\1\1',), width=3,
        retries=0, overlap=0, time_limit=5, state_limit=20, handoff_bonus=.02,
        branch_widths=widths, progress=lambda row: None)
    assert result['status'] == 'solved'
    step = result['steps'][1]
    assert step['state_moves'] == ('F', 'D')
    assert step['handoff_moves'] == ('F',)
    assert step['frame_state_moves'] == ('D',)
    assert step['queue_priority'] == pytest.approx(-.02)
    assert result['join_depth'] == 1


@pytest.mark.parametrize('direction', [0, 1])
@pytest.mark.parametrize('widths', [(3,), (2, 3)])
def test_widening_join_keeps_full_tie_at_branch_cutoff(direction, widths):
    def expand(node, depth):
        return [Link(state, step=dict(state_moves=(), state_score=score, queue_priority=score),
                     rank=((tiebreak,),))
                for state, score, tiebreak in (
                    (b'a', .1, 0), (b'b', .1, 0), (b'a', .1, 0), (b'c', .2, 0),
                    (b'd', .2, 0), (b'target', .2, 0), (b'worse', .2, 1))]

    def join(target):
        expansions = [expand, lambda *_: ()]
        seeds = [b'root', target]
        if direction == 1:
            expansions.reverse()
            seeds.reverse()
        return widening_join(*expansions, seeds[0], (seeds[1],), 1,
            widths=widths, stage_budget=1, mismatch_weight=0., expanded=[0, 0], reached=[0, 1],
            time_limit_check=lambda: None, state_limit=10, report=lambda *_, **__: None)

    match = join(b'target')
    assert match is not None
    assert match[direction + 1].step['sibling_rank'] == 5
    assert join(b'worse') is None


def test_lookahead_ranks_complete_windows_and_reuses_the_shifted_forecast():
    calls = []

    def edge(state, cost, cue=None):
        return Link(state, cue, step=dict(queue_priority=cost, state_moves=()), rank=(cost,))

    def expand(depth, state, cue):
        calls.append((depth, state, cue))
        cost = .8 if state == b"a" else .1
        return (edge(state, cost, cue),)

    forecast = WindowForecaster(expand, lambda count: (1/count,)*count, frame_count=13,
                                horizon=10, budget=20, check_budget=lambda _: None)
    initial = [edge(b"a", 0.), edge(b"b", .2, "R")]
    ranked = forecast.order(1, initial, width=1)
    assert ranked == initial[::-1]
    assert calls == [(depth, node.state, node.cue) for node in initial for depth in range(1, 11)]
    before = len(calls)
    successor = forecast.edges(1, b"b", "R")[0]
    assert len(calls) == before
    forecast.order(2, [successor], width=1)
    assert calls[before:] == [(11, b"b", "R")]
    assert forecast.warm_starts == 1
    assert forecast.cross_window_hits > 0
    forecast.edges(1, b"b", None)
    assert calls[-1] == (1, b"b", None)
