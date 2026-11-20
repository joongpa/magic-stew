"""Verify C4 routing through assignment, native loss, gradients and validation."""

import pytest
import torch
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import PoseModel

from live_recognition.sticker_quad_training import (
    StickerQuadModel,
    cyclic_keypoint_iou,
    select_cyclic_targets,
)


def test_routing_moves_coordinates_and_visibility_together_and_keeps_corner_fixed():
    gt = torch.tensor([[[1.0, 1.0, 0.0], [4.0, 1.0, 1.0], [4.0, 4.0, 2.0], [1.0, 4.0, 2.0]]] * 2)
    pred = gt.roll(-1, dims=1).clone()
    pred[..., 2] = torch.where(pred[..., 2] > 0, 10.0, -10.0)
    routed = select_cyclic_targets(
        pred,
        gt,
        torch.tensor([True, False]),
        torch.full((2, 1), 9.0),
        torch.full((4,), 0.25),
        12.0,
        1.0,
    )
    torch.testing.assert_close(routed[0], gt[0].roll(-1, dims=0))
    torch.testing.assert_close(routed[1], gt[1])


def test_validation_accepts_cyclic_center_but_not_reflections_or_corner_rotation():
    gt = torch.tensor([[[1.0, 1.0, 2.0], [4.0, 1.0, 2.0], [4.0, 4.0, 2.0], [1.0, 4.0, 2.0]]] * 2)
    predictions = torch.stack((gt[0].roll(-1, dims=0), gt[0].flip(0)))
    score = cyclic_keypoint_iou(
        gt, predictions, torch.tensor([2, 0]), torch.tensor([9.0, 9.0]), [0.25] * 4
    )
    assert score[0, 0] == 1
    assert score[1, 0] < 0.5
    assert score[0, 1] < 0.6


@pytest.mark.parametrize("family", ["yolo11n", "yolo26n"])
def test_actual_native_loss_is_cyclic_invariant_and_has_finite_gradients(family):
    torch.set_num_threads(2)
    torch.manual_seed(4)
    model = StickerQuadModel(f"{family}-pose.yaml", nc=18, data_kpt_shape=[4, 3], verbose=False)
    model.args = get_cfg()
    model.train()
    batch = {
        "img": torch.rand(2, 3, 128, 128),
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[2.0], [5.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]] * 2),
        "keypoints": torch.tensor(
            [[[0.25, 0.25, 0.0], [0.75, 0.25, 1.0], [0.75, 0.75, 2.0], [0.25, 0.75, 2.0]]] * 2
        ),
    }
    predictions = model(batch["img"])
    criterion = model.init_criterion()
    baseline, _ = criterion(predictions, batch)
    for shift in (1, 2, 3):
        shifted = {**batch, "keypoints": batch["keypoints"].roll(-shift, dims=1)}
        actual, _ = criterion(predictions, shifted)
        torch.testing.assert_close(actual, baseline, atol=2e-4, rtol=2e-5)
    baseline.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.model[-1].parameters())

    # Fixed corner/edge order retains the native objective, including RLE.
    fixed_batch = {**batch, "cls": torch.tensor([[0.0], [1.0]])}
    predictions = model(batch["img"])
    native_loss, native_items = PoseModel.init_criterion(model)(predictions, fixed_batch)
    quad_loss, quad_items = model.init_criterion()(predictions, fixed_batch)
    torch.testing.assert_close(quad_loss, native_loss)
    for key in native_items:
        torch.testing.assert_close(quad_items[key], native_items[key])


def test_model_checkpoint_keeps_importable_symmetry_criterion(tmp_path):
    model = StickerQuadModel("yolo11n-pose.yaml", nc=18, data_kpt_shape=[4, 3], verbose=False)
    model.args = get_cfg()
    path = tmp_path / "model.pt"
    torch.save({"model": model}, path)
    loaded = torch.load(path, weights_only=False)["model"]
    assert type(loaded.init_criterion()).__name__ == "StickerQuadLoss"
