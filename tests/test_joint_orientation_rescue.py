import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "joint_rescue_tests", ROOT / "scripts/search_independent_keypoint_full_solve.py"
)
search = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = search
SPEC.loader.exec_module(search)


@pytest.fixture(scope="module")
def renderer():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    return search.load_renderer()


@pytest.fixture
def data():
    return json.loads((ROOT / "tests/fixtures/nahm_frame_587_joint_orientation.json").read_text())


def make_scorer(m, data, cls=search.PreparedReplayScorer, **kwargs):
    settings = dict(
        initial_colors=search.initial_sticker_colors(m, data["scramble"]),
        balance_geometry_color=True,
        track_sticker_observations=True,
        joint_orientation_rescue=True,
    )
    settings.update(kwargs)
    return cls(m, {587: data["record"]}, "cpu", **settings)


@pytest.mark.parametrize("cls", [search.PreparedReplayScorer, search.ReplayScorer])
def test_nahm_587_selects_user_confirmed_orientation_with_reference_move(renderer, data, cls):
    scorer = make_scorer(renderer, data, cls)
    state = scorer.state_from_key(bytes(data["encoded_parent_state"]))
    choices = scorer(587, state)
    d = next(t for t in choices if t.payload["state_moves"] == ("D",))
    assert d.score == pytest.approx(data["expected"]["fit"])
    assert d.payload["orientation_symmetry_index"] == 6
    assert d.payload["color_match_count"] == 24
    assert d.payload["color_error"] == .25
    assert d.payload["score_components"]["geometry"] == pytest.approx(data["expected"]["geometry"])
    assert scorer.key(d.state) == scorer.key(state.apply_move("D") if not isinstance(state, bytes)
                                             else scorer.apply_moves(state, ("D",)))
    rescue = d.payload["orientation_rescue"]
    assert rescue["phase"] == "joint_orientation_and_turn_fit"
    assert rescue["tested_orientations"] == 24
    assert rescue["selected_color_error"] == .25
    assert len(choices) == len({(scorer.key(t.state), t.update_key) for t in choices}) == 25
    if cls is search.PreparedReplayScorer:
        assert scorer.child_states_built == 24  # Build only the final physical successors.


@pytest.mark.parametrize("balance", [False, True])
@pytest.mark.parametrize("cutoff", [None, 2/3])
def test_joint_prepared_matches_joint_legacy_scores_and_identity_associations(renderer, data, balance, cutoff):
    args = dict(balance_geometry_color=balance, minimum_color_accuracy=cutoff)
    old = make_scorer(renderer, data, search.ReplayScorer, **args)
    new = make_scorer(renderer, data, **args)
    state = bytes(data["encoded_parent_state"])
    expected = {(old.key(t.state), t.update_key): t for t in old(587, old.state_from_key(state))}
    actual = {(t.state, t.update_key): t for t in new(587, state)}
    assert actual.keys() == expected.keys()
    for key, t in actual.items():
        other = expected[key]
        assert t.score == pytest.approx(other.score, abs=1e-10)
        assert t.rank == pytest.approx(other.rank, abs=1e-10)
        for field in ("sticker_observed_mask_hex", "sticker_correct_mask_hex", "orientation_rescue"):
            assert t.payload[field] == other.payload[field]
        observed = int(t.payload["sticker_observed_mask_hex"], 16)
        correct = int(t.payload["sticker_correct_mask_hex"], 16)
        assert observed.bit_count() == t.payload["color_match_count"]
        assert correct.bit_count() == round((1-t.payload["color_error"]) * observed.bit_count())


def test_batched_joint_fit_matches_exhaustive_single_orientation_fits(renderer, data, monkeypatch):
    joint = make_scorer(renderer, data)
    single = make_scorer(renderer, data, joint_orientation_rescue=False, cache_size=0)
    state = bytes(data["encoded_parent_state"])
    expected = {}
    poses = [None] + [i for i, s in enumerate(renderer.CUBE_SYMMETRY_MATRICES)
                     if not np.array_equal(s, np.eye(3))]
    for pose in poses:
        def forced(prepared, encoded, pose=pose):
            oriented = np.frombuffer(encoded, dtype=np.uint8)
            if pose is not None:
                oriented = oriented[single.orientation_sources[pose]]
            return oriented.tobytes(), pose, None
        monkeypatch.setattr(single, "_orientation_for_state", forced)
        for t in single(587, state):
            key = t.state, t.update_key
            if key not in expected or t.rank < expected[key].rank:
                expected[key] = t
    actual = {(t.state, t.update_key): t for t in joint(587, state)}
    assert actual.keys() == expected.keys()
    for key, t in actual.items():
        assert t.rank == pytest.approx(expected[key].rank)
        for field in ("sticker_observed_mask_hex", "sticker_correct_mask_hex"):
            assert t.payload[field] == expected[key].payload[field]


def test_joint_rescue_reuses_projections_and_filters_physical_move_context_before_color_work(renderer, data, monkeypatch):
    scorer = make_scorer(renderer, data)
    state = bytes(data["encoded_parent_state"])
    scorer.prepare(587)
    def forbidden(*args, **kwargs):
        pytest.fail("Joint orientation rescue must reuse prepared projections")
    monkeypatch.setattr(renderer, "_torch_project_turn_points", forbidden)
    full = scorer(587, state)
    d = next(t for t in full if t.payload["state_moves"] == ("D",))
    before = scorer.color_configurations_scored
    restricted, skipped = scorer.for_update_context(587, state, (586, d.update_key, 1), 2)
    assert {t.payload["state_moves"] for t in restricted} == {(), ("D",)}
    assert skipped == 23
    assert scorer.color_configurations_scored - before < 24 * scorer.prepare(587)["configuration_count"]
    expected = {(t.state, t.update_key): t for t in full}
    for t in restricted:
        assert t.payload == expected[t.state, t.update_key].payload
    only_hold, skipped = scorer.for_update_context(587, state, (586, d.update_key, 2), 2)
    assert [t.payload["state_moves"] for t in only_hold] == [()]
    assert skipped == 24
    calls = scorer.scoring_calls
    assert scorer.for_update_context(587, state, (586, d.update_key, 1), 2)[0] is restricted
    assert scorer.scoring_calls == calls


@pytest.mark.parametrize("correct,rescue", [(1, True), (2, False), (3, False)])
def test_joint_rescue_preserves_strict_two_thirds_trigger(renderer, data, correct, rescue):
    prototype = make_scorer(renderer, data)
    prepared = prototype.prepare(587)
    row = prepared["static_row"]
    selected, seen = [], set()
    for point, (nearest, valid) in enumerate(zip(prepared["nearest"][row], prepared["weights"][row])):
        if valid and int(nearest) not in seen:
            selected.append(point)
            seen.add(int(nearest))
        if len(selected) == 3:
            break
    record = copy.deepcopy(data["record"])
    pose = record["search_pose"]
    pose["sticker_mask_centers"] = [pose["sticker_mask_centers"][i] for i in selected]
    pose["sticker_mask_center_uncertainties"] = [1.] * 3
    pose.pop("sticker_color_names", None)
    record["turn_color_matches"] = np.tile([1.] * correct + [0.] * (3-correct), (54, 1)).tolist()
    scorer = search.PreparedReplayScorer(renderer, {587: record}, "cpu", joint_orientation_rescue=True)
    scorer(587, bytes(data["encoded_parent_state"]))
    assert scorer.joint_rescue_calls == int(rescue)


def test_joint_masks_count_duplicate_color_points_once(renderer, data):
    record = copy.deepcopy(data["record"])
    pose = record["search_pose"]
    pose["sticker_mask_centers"].extend(copy.deepcopy(pose["sticker_mask_centers"]))
    pose["sticker_mask_center_uncertainties"] *= 2
    pose["sticker_color_names"] *= 2
    duplicate = make_scorer(renderer, dict(data, record=record))
    original = make_scorer(renderer, data)
    state = bytes(data["encoded_parent_state"])
    expected = {(t.state, t.update_key): t for t in original(587, state)}
    for t in duplicate(587, state):
        other = expected[t.state, t.update_key]
        assert t.score == pytest.approx(other.score)
        for field in ("color_match_count", "sticker_observed_mask_hex", "sticker_correct_mask_hex"):
            assert t.payload[field] == other.payload[field]
