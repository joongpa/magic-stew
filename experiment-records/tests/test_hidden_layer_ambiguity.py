from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.verify_hidden_layer_ambiguity import diagnose, make_state  # noqa: E402


def test_diagnostic_exposes_hidden_state_supervision_without_claiming_support():
    report = diagnose()
    assert report["diagnostic_passed"]
    assert not report["ambiguity_handled_by_current_loss"]
    for case in report["cases"]:
        assert case["matched_front_landmarks"] == 16
        assert case["observed_front_rmse_px"] < 1e-3
    assert report["cases"][0]["geometry_loss_excluding_quality"] < 1e-8
    assert all(case["geometry_loss_excluding_quality"] > .01 for case in report["cases"][1:])


def test_wide_turn_uses_a_different_joint_state_with_compensating_front_rotation():
    back = make_state(20)
    wide = make_state(20, wide=True)
    assert not np.allclose(back["rotation_matrix"], wide["rotation_matrix"])
    assert back["angles"][0, 3] == 20
    assert wide["angles"][0, 2] == 20
    assert wide["angles"][0, 3] == 0
