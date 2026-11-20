import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fixed_pose_tests", ROOT / "scripts/search_independent_keypoint_full_solve.py"
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
    settings = {
        'initial_colors': search.initial_sticker_colors(m, data['scramble']),
        'balance_geometry_color': True,
        'track_sticker_observations': True,
    }
    settings.update(kwargs)
    return cls(m, {587: data["record"]}, "cpu", **settings)


@pytest.mark.parametrize('cls', [search.PreparedReplayScorer, search.ReplayScorer])
def test_color_mismatch_does_not_change_the_supplied_pose(renderer, data, cls, monkeypatch):
    scorer = make_scorer(renderer, data, cls)
    state = scorer.state_from_key(bytes(data['encoded_parent_state']))
    scorer.prepare(587)

    def forbidden(*args, **kwargs):
        pytest.fail('State scoring must use the supplied orientation')

    monkeypatch.setattr(renderer, 'transform_rvec_by_symmetry', forbidden)
    monkeypatch.setattr(renderer, 'transform_candidate_by_symmetry', forbidden)
    monkeypatch.setattr(renderer, '_geometry_state_under_symmetry', forbidden)
    choices = scorer(587, state)
    assert choices
    assert all('orientation_symmetry_index' not in t.payload and 'orientation_rescue' not in t.payload
               for t in choices)


@pytest.mark.parametrize('balance', [False, True])
@pytest.mark.parametrize('cutoff', [None, 2/3])
def test_fixed_pose_backends_match_scores_and_sticker_identities(renderer, data, balance, cutoff):
    settings = {'balance_geometry_color': balance, 'minimum_color_accuracy': cutoff}
    old = make_scorer(renderer, data, search.ReplayScorer, **settings)
    new = make_scorer(renderer, data, **settings)
    state = bytes(data['encoded_parent_state'])
    expected = {(old.key(t.state), t.update_key): t for t in old(587, old.state_from_key(state))}
    actual = {(t.state, t.update_key): t for t in new(587, state)}
    assert actual.keys() == expected.keys()
    for key, choice in actual.items():
        other = expected[key]
        assert choice.rank == pytest.approx(other.rank, abs=1e-10)
        for field in ('sticker_observed_mask_hex', 'sticker_correct_mask_hex'):
            assert choice.payload[field] == other.payload[field]


def test_move_restrictions_reuse_fixed_projections(renderer, data, monkeypatch):
    scorer = make_scorer(renderer, data)
    state = bytes(data['encoded_parent_state'])
    scorer.prepare(587)

    def forbidden(*args, **kwargs):
        pytest.fail('State scoring must reuse prepared geometry')

    monkeypatch.setattr(renderer, '_torch_project_turn_points', forbidden)
    full = scorer(587, state)
    d = next(t for t in full if t.payload['state_moves'] == ('D',))
    restricted, skipped = scorer.for_update_context(587, state, (586, d.update_key, 1), 2)
    assert {t.payload['state_moves'] for t in restricted} == {(), ('D',)}
    assert skipped == 23


