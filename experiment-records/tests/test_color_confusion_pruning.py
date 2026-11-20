import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "confusion_tests", ROOT / "scripts/search_independent_keypoint_frame.py"
)
f = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = f
SPEC.loader.exec_module(f)


@pytest.fixture(scope="module")
def m():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    return f.replay.load_renderer()


PAIRS = {
    ("red", "orange"),
    ("orange", "yellow"),
    ("orange", "red"),
    ("yellow", "orange"),
    ("yellow", "white"),
    ("yellow", "green"),
    ("white", "blue"),
    ("white", "yellow"),
    ("green", "yellow"),
}
COLORS = ("white", "yellow", "red", "orange", "blue", "green")


def costs(m, expected, predicted, penalty=0.5):
    candidate = SimpleNamespace(
        sticker_color_names=predicted, sticker_mask_centers=[(0.0, 0.0)] * len(predicted)
    )
    colors = {i.label: expected for i in m.canonical_sticker_identities()}
    return m.turn_color_confusion_costs(candidate, colors, penalty)


@pytest.mark.parametrize("expected", COLORS)
@pytest.mark.parametrize("predicted", COLORS)
def test_all_six_by_six_expected_to_predicted_pairs(m, expected, predicted):
    error = 0 if expected == predicted else 0.5 if (expected, predicted) in PAIRS else 1
    matrix = costs(m, expected, [predicted])
    assert np.all(matrix == error)
    metrics = m.sticker_color_confusion_metrics(
        np.arange(54), np.array([[0]]), np.array([[True]]), matrix
    )
    assert metrics["error"][0] == error
    assert metrics["forbidden"][0] == int(error == 1)


def test_no_chaining_and_no_implicit_reverse_edges(m):
    assert costs(m, "white", ["yellow", "orange"])[0].tolist() == [0.5, 1.0]
    assert costs(m, "red", ["orange", "yellow"])[0].tolist() == [0.5, 1.0]
    assert costs(m, "green", ["yellow", "white"])[0].tolist() == [0.5, 1.0]
    assert costs(m, "white", ["blue"])[0, 0] == 0.5
    assert costs(m, "blue", ["white"])[0, 0] == 1.0


def test_duplicate_points_choose_best_color_and_outside_points_do_not_vote(m):
    matrix = costs(m, "orange", ["blue", "orange", "red", "yellow", "blue"])
    metric = m.sticker_color_confusion_metrics(
        np.arange(54),
        np.array([[0, 0, 1, 2, 3]]),
        np.array([[True, True, True, True, False]]),
        matrix,
    )
    assert metric["total"][0] == 3 and metric["correct"][0] == 1
    assert metric["confused"][0] == 2 and metric["forbidden"][0] == 0
    assert metric["error"][0] == pytest.approx(1 / 3)
    # An allowed prediction can also rescue a forbidden duplicate, at a cost.
    metric = m.sticker_color_confusion_metrics(
        np.arange(54),
        np.array([[0, 0]]),
        np.ones((1, 2), dtype=bool),
        costs(m, "orange", ["blue", "red"]),
    )
    assert metric["total"][0] == 1 and metric["confused"][0] == 1 and metric["forbidden"][0] == 0
    assert metric["error"][0] == 0.5


def test_one_forbidden_sticker_is_not_diluted_by_many_exact_matches(m):
    matrix = costs(m, "orange", ["orange"] * 20 + ["blue"])
    metrics = m.sticker_color_confusion_metrics(
        np.arange(54), np.arange(21)[None, :], np.ones((1, 21), dtype=bool), matrix
    )
    assert metrics["correct"][0] == 20 and metrics["forbidden"][0] == 1


def test_batched_metrics_keep_pose_identities_separate(m):
    matrix = costs(m, "orange", ["blue"])
    matrix[1, 0] = 0
    identities = np.stack([np.arange(54), np.array([1, 0, *range(2, 54)])])
    metrics = m.sticker_color_confusion_metrics(
        identities, np.array([[0]]), np.array([[True]]), matrix
    )
    assert metrics["forbidden"].tolist() == [[1], [0]]
    assert metrics["correct"].tolist() == [[0], [1]]


@pytest.mark.parametrize("penalty", [0.0, 1.0, float("nan"), -0.5])
def test_confusions_have_a_positive_cost_less_than_forbidden(m, penalty):
    with pytest.raises(ValueError):
        costs(m, "orange", ["yellow"], penalty)


@pytest.mark.parametrize("cls", [f.replay.PreparedReplayScorer, f.replay.ReplayScorer])
@pytest.mark.parametrize(
    "predicted,survives", [("yellow", True), ("blue", False), ("white", False)]
)
def test_scorers_reject_forbidden_fits_before_state_construction(
    m, cls, predicted, survives
):
    data = json.loads((ROOT / "tests/fixtures/nahm_frame_587_joint_orientation.json").read_text())
    record = copy.deepcopy(data["record"])
    pose = record["search_pose"]
    pose["sticker_color_names"] = [predicted] * len(pose["sticker_mask_centers"])
    colors = {i.label: "orange" for i in m.canonical_sticker_identities()}
    scorer = cls(
        m,
        {587: record},
        "cpu",
        initial_colors=colors,
        balance_geometry_color=True,
        color_confusion_pruning=True,
        color_confusion_cost=0.5,
    )
    state = scorer.state_from_key(bytes(data["encoded_parent_state"]))
    choices = scorer(587, state)
    assert bool(choices) == survives
    if survives:
        for choice in choices:
            p = choice.payload
            assert p["color_error"] == 0.5
            assert p["color_correct_count"] == 0
            assert p["color_confusion_count"] == p["color_match_count"] > 0
            assert p["color_forbidden_count"] == 0
            assert choice.score == pytest.approx(0.5 * p["score_components"]["geometry"] + 0.25)
    else:
        assert scorer.confusion_pruned_fits > 0
        assert scorer.confusion_dead_end_states == 1
        if cls is f.replay.PreparedReplayScorer:
            assert scorer.child_states_built == 0


@pytest.mark.parametrize('cls', [f.replay.PreparedReplayScorer, f.replay.ReplayScorer])
@pytest.mark.parametrize('predicted,error', [('orange', 0.), ('yellow', .5), ('blue', 1.)])
def test_weighted_color_scoring_without_hard_rejection(m, cls, predicted, error):
    data = json.loads((ROOT/'tests/fixtures/nahm_frame_587_joint_orientation.json').read_text())
    record = copy.deepcopy(data['record'])
    record['search_pose']['sticker_color_names'] = [predicted]*len(record['search_pose']['sticker_mask_centers'])
    colors = {i.label: 'orange' for i in m.canonical_sticker_identities()}
    scorer = cls(m, {587: record}, 'cpu', initial_colors=colors, balance_geometry_color=True,
                 color_confusion_scoring=True)
    choices = scorer(587, scorer.state_from_key(bytes(data['encoded_parent_state'])))
    assert choices
    assert all(t.payload['color_error'] == error for t in choices)
    assert all(t.score == pytest.approx(.5*t.payload['score_components']['geometry']+.5*error) for t in choices)
    if predicted == 'blue':
        assert all(t.payload['color_forbidden_count'] == t.payload['color_match_count'] > 0 for t in choices)
    assert scorer.confusion_pruned_fits == scorer.confusion_dead_end_states == 0


@pytest.mark.parametrize('uniform', [False, True])
def test_hold_activity_survives_fit_collapse_and_backends_agree(m, uniform):
    data = json.loads((ROOT/'tests/fixtures/nahm_frame_587_joint_orientation.json').read_text())
    record = copy.deepcopy(data['record'])
    if uniform:
        record['search_pose']['sticker_color_names'] = ['yellow']*len(record['search_pose']['sticker_mask_centers'])
    colors = ({i.label: 'orange' for i in m.canonical_sticker_identities()} if uniform else
              f.replay.initial_sticker_colors(m, data['scramble']))
    results = []
    for cls in (f.replay.PreparedReplayScorer, f.replay.ReplayScorer):
        scorer = cls(m, {587: record}, 'cpu', initial_colors=colors, balance_geometry_color=True,
                     color_confusion_scoring=True, preserve_layer_activity=True)
        choices = scorer(587, scorer.state_from_key(bytes(data['encoded_parent_state'])))
        keys = [(tuple(t.payload['state_moves']), m.turn_trial_activity_layer(t.payload)) for t in choices]
        assert len(keys) == len(set(keys))
        holds = [key[1] for key in keys if not key[0]]
        assert None in holds and len(holds) >= 2
        assert len(choices) <= 37
        if cls is f.replay.PreparedReplayScorer:
            assert scorer.child_states_built <= 24
        results.append({key:t.score for key,t in zip(keys, choices)})
    assert results[0] == pytest.approx(results[1])
