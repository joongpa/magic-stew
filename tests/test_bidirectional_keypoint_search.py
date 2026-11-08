from dataclasses import dataclass
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from search_independent_keypoint_bidirectional import Link, bidirectional_search, replay
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
