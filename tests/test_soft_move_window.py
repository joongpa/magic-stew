import importlib.util
from pathlib import Path
import pickle
import sys

import pytest

SPEC = importlib.util.spec_from_file_location(
    "soft_window_tests",
    Path(__file__).resolve().parents[1] / "scripts/search_independent_keypoint_frame.py",
)
f = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = f
SPEC.loader.exec_module(f)


def choice(state, score=0.1, correct=True, moves=(), **extra):
    return f.replay.Transition(
        state,
        score,
        (score,),
        dict(
            state_moves=moves,
            color_error=0.0 if correct else 1.0,
            color_match_count=1,
            score_components=dict(geometry=0.1),
            sticker_observed_mask_hex="1",
            sticker_correct_mask_hex="1" if correct else "0",
            **extra,
        ),
    )


def queue_for(backend, path=":memory:"):
    return (
        f.storage.MeanQueue(path)
        if backend == "sqlite"
        else f.storage.BufferedMeanQueue(path, frontier_cache_size=1, write_batch_size=1)
    )


@pytest.mark.parametrize("backend", ["sqlite", "buffered"])
def test_merge_reopens_same_physical_context_for_better_complete_history(backend):
    def choices(frame, state):
        if frame == 0:
            return (choice("a", 0.4, False, branch="a"), choice("b", 0.01, True, branch="b"))
        if frame < 3:
            return (choice(state, 0.4 if state == "a" else 0.01, frame == 2 and state == "a"),)
        if frame == 3:
            return (choice("merged", 0.2, state == "a"),)
        return (choice("end", 0.2, False),)

    q = queue_for(backend)
    try:
        result = f.frame_full_solve(
            "root",
            tuple(range(5)),
            choices,
            lambda _: False,
            str,
            queue=q,
            sticker_history=True,
            merge_histories=True,
            minimum_color_accuracy=None,
        )
        if hasattr(q, "flush"):
            q.flush()
        rows = [
            row
            for row in q.connection.execute("SELECT key,total,payload FROM contexts")
            if pickle.loads(row[0])[2] == "merged"
        ]
        assert len(rows) == 1
        _, total, payload = rows[0]
        _, _, path_id, recent = q.unpack(payload)
        assert total == pytest.approx(0.2 + 0.1 * 0.25)
        assert f.history.penalty(recent) == 0.25
        steps = q.steps(path_id)
        assert steps[0]["branch"] == "a"
        rebuilt = f.history.EMPTY
        for step in steps:
            rebuilt = f.history.observe(rebuilt, step)
        assert recent == rebuilt  # One entire real history, never per-sticker mixtures.
        assert result["reopened_states"] >= 1  # The worse history was already expanded.
    finally:
        q.close()


@pytest.mark.parametrize("backend", ["sqlite", "buffered"])
def test_merge_preserves_distinct_unexpired_move_spacing_contexts(backend):
    def choices(frame, state):
        if frame == 0:
            return (
                f.replay.Transition(
                    "a", 0.1, (0.1,), dict(choice("a").payload, state_moves=("R",)), ("R",)
                ),
                f.replay.Transition(
                    "b", 0.1, (0.1,), dict(choice("b").payload, state_moves=("L",)), ("L",)
                ),
            )
        return (choice("merged"),)

    q = queue_for(backend)
    try:
        f.frame_full_solve(
            "root",
            (0, 1, 2),
            choices,
            lambda _: False,
            str,
            queue=q,
            sticker_history=True,
            merge_histories=True,
            minimum_color_accuracy=None,
            min_update_gap=3,
        )
        if hasattr(q, "flush"):
            q.flush()
        contexts = [
            pickle.loads(row[0]) for row in q.connection.execute("SELECT key FROM contexts")
        ]
        at_frame_one = [c for c in contexts if c[1] == 2 and c[2] == "merged"]
        assert len(at_frame_one) == 2
        assert {c[3][1] for c in at_frame_one} == {("R",), ("L",)}
    finally:
        q.close()


def test_history_merging_does_not_cherry_pick_individual_sticker_outcomes():
    def choices(frame, state):
        if frame == 0:
            return tuple(
                f.replay.Transition(
                    name,
                    0.01,
                    (0.01,),
                    dict(
                        state_moves=(),
                        branch=name,
                        sticker_observed_mask_hex="3",
                        sticker_correct_mask_hex=correct,
                    ),
                )
                for name, correct in [("a", "1"), ("b", "2")]
            )
        return (f.replay.Transition("goal", 0.1, (0.1,), dict(state_moves=())),)

    result = f.frame_full_solve(
        "root",
        (0, 1),
        choices,
        lambda state: state == "goal",
        str,
        sticker_history=True,
        merge_histories=True,
        minimum_color_accuracy=None,
    )
    assert result["steps"][0]["branch"] == "a"
    assert result["steps"][-1]["sticker_history_penalty"] == 0.0625


def test_loop_erased_support_excludes_canceling_moves_and_whole_cube_rotations():
    window = f.SoftMoveWindow(lambda state: state.split("@")[0], moves=1)
    trail = window.initial(f.history.EMPTY, "root")[1]
    trail = window.trail_after(trail, "a", 10, ("R",), supported=True)
    trail = window.trail_after(trail, "a@rotated", 11, ("x",), supported=True)
    assert len(trail) == 2
    trail = window.trail_after(trail, "b", 12, ("U",), supported=True)
    trail = window.trail_after(trail, "a", 13, ("U'",), supported=True)
    assert sum(good for _, _, good in trail) == 1
    window.establish((f.history.EMPTY, trail, None))
    assert window.protected_frame is None


def test_only_supported_moves_advance_window_and_weak_frames_pause():
    window = f.SoftMoveWindow(str, moves=4)
    trail = window.initial(f.history.EMPTY, "root")[1]
    for frame in range(4):
        trail = window.trail_after(trail, str(frame), frame, ("R",), supported=True)
    window.establish((f.history.EMPTY, trail, None))
    assert window.protected_frame is None
    assert not window.is_supported(choice("x").payload, 1.0, 1.0, reliable=False)
    assert not window.is_supported(choice("x", moves=("R",)).payload, 1.0, 1.1)
    bad_geometry = dict(choice("x", moves=("R",)).payload, score_components=dict(geometry=0.6))
    assert not window.is_supported(bad_geometry, 1.0, 1.0)
    assert window.is_supported(choice("x", moves=("R",)).payload, 1.0, 1.0)
    trail = window.trail_after(trail, "next", 10, ("U",), supported=True)
    window.establish((f.history.EMPTY, trail, None))
    assert window.protected_frame == 0


def test_old_fork_penalty_is_soft_temporary_and_does_not_reprice_existing_branches():
    window = f.SoftMoveWindow(str)
    window.protected_frame = 100
    assert window.fork_cost(80, None, new_fork=False) == (0.0, None)
    assert window.fork_cost(80, None, new_fork=True) == (0.5, 100)
    assert window.fork_cost(90, 100) == (0.5, 100)
    assert window.fork_cost(101, 100) == (0.0, None)
    assert window.fork_cost(101, None, new_fork=True) == (0.0, None)


@pytest.mark.parametrize("backend", ["sqlite", "buffered"])
def test_new_old_forks_wait_but_can_catch_up_without_pruning(backend):
    def choices(frame, state):
        if frame == 0:
            return (choice("run1", 0.01, moves=("R",)), choice("old1", 0.2))
        if state.startswith("run"):
            return (choice("run" + str(frame + 1), 0.01, moves=("R",)),)
        if frame == 1:
            return (choice("best_old", 0.2, moves=("R",)), choice("fork_old", 0.21, moves=("U",)))
        return (choice("end_" + state, 0.8 if state == "best_old" else 0.01),)

    window = f.SoftMoveWindow(str, moves=1)
    q = queue_for(backend)
    try:
        result = f.frame_full_solve(
            "root",
            (0, 1, 2),
            choices,
            lambda _: False,
            str,
            queue=q,
            min_update_gap=1,
            sticker_history=True,
            merge_histories=True,
            soft_window=window,
            minimum_color_accuracy=None,
        )
        if hasattr(q, "flush"):
            q.flush()
        saved = {}
        for key, payload in q.connection.execute("SELECT key,payload FROM contexts"):
            _, _, path_id, _ = q.unpack(payload)
            saved[pickle.loads(key)[2]] = q.steps(path_id)
        fork = saved["fork_old"][-1]
        assert fork["soft_window_penalty"] == 0.5
        assert fork["queue_priority"] == pytest.approx(0.71)
        assert saved["best_old"][-1]["soft_window_penalty"] == 0.0
        assert saved["end_fork_old"][-1]["soft_window_penalty"] == 0.0
        assert result["unsolved_endpoints"] == 3
        assert result["late_forks_penalized"] == 1
        assert result["color_pruned"] == 0
    finally:
        q.close()


def test_rotation_invariant_progress_key_preserves_real_moves():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    m = f.replay.load_renderer()
    scorer = f.replay.PreparedReplayScorer(m, {}, "cpu")
    key = f.rotation_invariant_state_key(scorer)
    initial = scorer.key(m.CubeGeometryState.identity())
    turned = scorer.apply_moves(initial, ("R",))
    assert key(initial) == key(scorer.apply_moves(initial, ("x", "y'")))
    assert key(initial) != key(turned)
    assert key(turned) == key(scorer.apply_moves(turned, ("z",)))
