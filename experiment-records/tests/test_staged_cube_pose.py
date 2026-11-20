"""Geometry/frame invariants and portable staged training contract."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live_recognition.staged_pose.kinematics import rotation_exp
from live_recognition.staged_pose.loss import StagedPoseLoss
from live_recognition.staged_pose.model import ImageContextPhaseHead, StaticPoseHead
from scripts.export_direct_cube_pose_dataset import symmetry_manifest
from scripts.train_yolo26_staged_cube_pose import (
    build_parser,
    split_for_initialization,
    validate_args,
)


@pytest.fixture
def criterion(monkeypatch):
    monkeypatch.setattr(
        "live_recognition.staged_pose.loss.independent_pose_criterion", lambda _: None
    )
    return StagedPoseLoss(SimpleNamespace(detector=None), symmetry_manifest())


def test_exact_context_architecture_sizes_and_face_permutation():
    torch.manual_seed(11)
    head = ImageContextPhaseHead().eval()
    assert sum(p.numel() for p in head.parameters()) == 78429
    assert sum(p.numel() for p in StaticPoseHead([256, 512, 512]).parameters()) == 2387087
    with torch.no_grad():
        head.phase.weight.normal_(std=0.02)
        head.axis[-1].weight.normal_(std=0.02)
    x = torch.randn(2, 6, 2, 16, 24)
    x[..., 6] = x[..., 6].sigmoid()
    valid = torch.rand(2, 6, 2, 16) > 0.25
    appearance = torch.randn(2, 16, 1280)
    scores = torch.rand(2, 16, 9)
    value = head(x, valid, appearance, scores)[0]
    perm = torch.tensor([4, 5, 0, 1, 2, 3])
    axis = torch.tensor([2, 0, 1])
    other = head(x[:, perm], valid[:, perm], appearance, scores)[0]
    torch.testing.assert_close(value[:, axis], other, atol=2e-6, rtol=1e-5)
    swap = torch.tensor([1, 0, 3, 2, 5, 4])
    other = head(x[:, swap], valid[:, swap], appearance, scores)[0]
    torch.testing.assert_close(value.transpose(-1, -2), other, atol=2e-6, rtol=1e-5)


def sample_pose():
    return {
        "rotation_matrix": rotation_exp(torch.tensor([[0.3, -0.7, 0.2]])),
        "angles": torch.tensor([[23.0, -11.0, 0.0, 0.0, 0.0, 0.0]]),
        "center": torch.tensor([[0.51, 0.49]]),
        "scale": torch.tensor([0.12]),
        "perspective": torch.tensor([0.2]),
        "principal": torch.tensor([[0.5, 0.5]]),
    }


def test_visible_geometry_accepts_all24_coherent_frames(criterion):
    pose = sample_pose()
    source = criterion.reference.project(pose, torch.tensor([0]))
    target = {
        **pose,
        "rotation": pose["rotation_matrix"],
        "eligible": torch.ones(1, 6, dtype=torch.bool),
        "visible_xy": source,
        "visible_mask": torch.ones(1, 110, dtype=torch.bool),
    }
    for index, Q in enumerate(criterion.rotations):
        candidate = dict(
            pose,
            rotation_matrix=pose["rotation_matrix"] @ Q,
            angles=pose["angles"][:, criterion.faces[index]],
        )
        truth, _, qi = criterion.gauge(candidate, target)
        xy, mask = criterion.reframe_targets(qi, target)
        loss = criterion.joint_visible_loss(candidate, truth, xy, mask, torch.tensor([True]), 128)
        assert float(loss) < 1e-8, (index, float(loss))


def test_visible_geometry_accepts_slab_quarters_and_has_turn_gradient(criterion):
    pose = sample_pose()
    xy = criterion.reference.project(pose, torch.tensor([0]))
    mask = torch.ones(1, 110, dtype=torch.bool)
    candidate = dict(
        pose, angles=torch.tensor([[113.0, -101.0, 0.0, 0.0, 0.0, 0.0]], requires_grad=True)
    )
    assert (
        float(
            criterion.joint_visible_loss(
                candidate, pose, xy, mask, torch.tensor([True]), 128
            ).detach()
        )
        < 1e-8
    )
    candidate = dict(
        pose, angles=torch.tensor([[25.0, -13.0, 0.0, 0.0, 0.0, 0.0]], requires_grad=True)
    )
    loss = criterion.joint_visible_loss(candidate, pose, xy, mask, torch.tensor([True]), 128)
    loss.backward()
    assert candidate["angles"].grad[:, :2].abs().min() > 0
    assert torch.isfinite(candidate["angles"].grad).all()


def test_warmstart_training_images_never_enter_validation():
    entries = [
        {"key": str(i), "source_dataset": "source", "mode": "mid_turn", "symmetry_count": 1}
        for i in range(40)
    ]
    train, val = split_for_initialization(
        entries, {"training_keys": [str(i) for i in range(20)]}, 0.5, 17
    )
    assert len(train) + len(val) == 40
    assert not {e["key"] for e in val} & {str(i) for i in range(20)}
    assert not {e["key"] for e in val} & {e["key"] for e in train}


def test_cli_defaults_target_prepared_dataset_and_full_joint_schedule():
    args = validate_args(build_parser().parse_args(["--device", "0", "--batch", "-1"]))
    assert args.dataset == Path("data/direct_cube_pose_dataset") and args.imgsz == 128
    assert args.device == "0" and args.batch == -1
    assert all(
        getattr(args, f"{p}_epochs") > 0 for p in ("static", "articulation", "handoff", "joint")
    )
    assert (
        validate_args(
            build_parser().parse_args(["--source-dir", "prepared", "--resolution", "144"])
        ).imgsz
        == 160
    )


@pytest.mark.parametrize(
    "flag,value", [("--batch", "0"), ("--joint-epochs", "-1"), ("--imgsz", "0")]
)
def test_cli_invalid_values_fail_before_training(flag, value):
    with pytest.raises(ValueError):
        validate_args(build_parser().parse_args([flag, value]))
