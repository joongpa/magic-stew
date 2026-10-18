import torch
from torch import nn

from scripts.train_yolo_cube_pose_local import scheduled_loss_weights

from live_recognition.cube_pose import (
    ANGLE_LIMIT_DEGREES,
    CubePoseHead,
    PROTOTYPE_MASK_CHANNELS,
    POSE_MASK_CHANNELS,
    RawImageCubePoseModel,
    YOLOCubePoseModel,
    _nonoverflow_huber,
    cube_pose_geometry_loss,
    cube_pose_loss,
)


class FakeSegmentationHead(nn.Module):
    def forward(self, features):
        batch_size = features[0].shape[0]
        device = features[0].device
        return (
            features[0],
            {
                "proto": torch.rand(batch_size, 4, 8, 8, device=device),
                "mask_coefficient": torch.rand(batch_size, 4, 3, device=device),
                "scores": torch.rand(batch_size, 2, 3, device=device),
                "boxes": torch.full((batch_size, 4, 3), 32.0, device=device),
            },
        )


class FakeSegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleList(
            [
                nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                FakeSegmentationHead(),
            ]
        )

    def forward(self, images):
        high_resolution = self.model[0](images)
        low_resolution = self.model[1](high_resolution)
        return self.model[-1]([low_resolution, high_resolution])


def direct_pose(*, u=45.0, v=0.0, twist=0.0, center=(0.5, 0.5), layers=(0.0, 0.0)):
    return {
        "u": torch.tensor([u]),
        "v": torch.tensor([v]),
        "twist": torch.tensor([twist]),
        "image_center": torch.tensor([center]),
        "image_scale": torch.tensor([0.25]),
        "perspective_strength": torch.tensor([0.75]),
        "layer_angles": torch.tensor([layers]),
    }


def test_pose_head_decodes_bounded_outputs():
    head = CubePoseHead((32, 16))
    output = head(
        (
            torch.rand(2, 32, 16, 16),
            torch.rand(2, 16, 32, 32),
        ),
        torch.rand(2, POSE_MASK_CHANNELS, 16, 16),
    )

    assert output["u"].shape == (2,)
    assert output["v"].shape == (2,)
    assert output["twist"].shape == (2,)
    assert output["image_center"].shape == (2, 2)
    assert output["image_scale"].shape == (2,)
    assert output["perspective_strength"].shape == (2,)
    assert output["layer_angles"].shape == (2, 2)
    assert torch.all(output["layer_angles"].abs() <= ANGLE_LIMIT_DEGREES)
    assert torch.all((output["u"] >= 0.0) & (output["u"] < 90.0))
    assert set(output) == {
        "u",
        "v",
        "twist",
        "image_center",
        "image_scale",
        "perspective_strength",
        "layer_angles",
    }


def test_soft_segmentation_maps_return_configured_channels():
    model = YOLOCubePoseModel(
        FakeSegmentationModel(),
        (32, 16),
        freeze_base=False,
    )
    maps = model._soft_segmentation_maps(
        model.base_model(torch.rand(1, 3, 32, 32)),
        image_size=(32, 32),
    )

    assert maps.shape == (1, POSE_MASK_CHANNELS, 8, 8)
    assert torch.all((maps >= 0.0) & (maps <= 1.0))


def test_soft_segmentation_maps_can_return_prototype_unions_only():
    model = YOLOCubePoseModel(
        FakeSegmentationModel(),
        (32, 16),
        freeze_base=False,
        use_soft_masks=True,
        mask_channels=PROTOTYPE_MASK_CHANNELS,
    )
    maps = model._soft_segmentation_maps(
        model.base_model(torch.rand(1, 3, 32, 32)),
        image_size=(32, 32),
    )

    assert maps.shape == (1, PROTOTYPE_MASK_CHANNELS, 8, 8)
    assert torch.all((maps >= 0.0) & (maps <= 1.0))
    model.close()


def test_pose_loss_is_zero_for_matching_direct_predictions():
    pose = direct_pose(u=45.0, v=20.0, twist=12.0, layers=(12.0, 0.0))
    losses = cube_pose_loss(pose, pose)
    assert losses["total"] < 1e-6


def test_projected_geometry_loss_is_zero_for_matching_direct_pose():
    pose = direct_pose(u=35.0, v=20.0, twist=12.0, center=(0.47, 0.53), layers=(12.0, -8.0))
    assert cube_pose_geometry_loss(pose, pose) < 1e-6


def test_projected_geometry_loss_increases_with_visible_displacement():
    target = direct_pose()
    small = direct_pose(center=(0.505, 0.5))
    large = direct_pose(center=(0.55, 0.5))
    assert cube_pose_geometry_loss(small, target) < cube_pose_geometry_loss(large, target)


def test_projected_geometry_loss_has_finite_gradient_at_exact_match():
    pose = direct_pose(u=89.0, v=0.0, twist=12.0, layers=(12.0, -8.0))
    pose = {key: value.requires_grad_() for key, value in pose.items()}

    cube_pose_geometry_loss(pose, pose).backward()

    assert all(torch.isfinite(value.grad).all() for value in pose.values())


def test_geometry_huber_stays_finite_for_large_finite_distances():
    distances = torch.tensor(
        [1.0e8, 1.0e16, 1.0e30, torch.finfo(torch.float32).max / 2.0]
    )
    costs = _nonoverflow_huber(distances, beta=0.01)
    assert torch.isfinite(costs).all()
    assert torch.allclose(costs, distances, rtol=1e-5)


def test_wrapped_angle_errors_are_small_at_period_boundary():
    target = direct_pose(v=179.0, twist=44.0, layers=(44.0, -44.0))
    prediction = direct_pose(v=-179.0, twist=-44.0, layers=(-44.0, 44.0))
    losses = cube_pose_loss(prediction, target)
    assert losses["rotation"] < 0.01
    assert losses["layer_angles"] < 0.01


def test_yolo_pose_wrapper_captures_features():
    model = YOLOCubePoseModel(
        FakeSegmentationModel(),
        feature_channels=(32, 16),
        freeze_base=True,
    )
    output = model(torch.rand(2, 3, 64, 64))
    assert output["u"].shape == (2,)
    assert output["layer_angles"].shape == (2, 2)
    assert all(not parameter.requires_grad for parameter in model.base_model.parameters())
    model.close()


def test_yolo_pose_wrapper_supports_feature_only_inputs():
    model = YOLOCubePoseModel(
        FakeSegmentationModel(),
        feature_channels=(32, 16),
        freeze_base=True,
        use_soft_masks=False,
    )
    output = model(torch.rand(2, 3, 64, 64))
    assert output["u"].shape == (2,)
    assert output["layer_angles"].shape == (2, 2)
    assert model.pose_head.mask_projection is None
    model.close()


def test_raw_image_pose_model_uses_only_rgb_input():
    model = RawImageCubePoseModel()
    output = model(torch.rand(2, 3, 256, 256))
    assert output["u"].shape == (2,)
    assert output["image_center"].shape == (2, 2)
    assert output["layer_angles"].shape == (2, 2)
    model.close()


def test_loss_weight_schedule_defaults_to_head_only():
    first = scheduled_loss_weights(
        1,
        10,
        head_start=1.0,
        head_end=1.0,
        geometry_start=0.0,
        geometry_end=0.0,
    )
    last = scheduled_loss_weights(
        10,
        10,
        head_start=1.0,
        head_end=1.0,
        geometry_start=0.0,
        geometry_end=0.0,
    )
    assert first == (1.0, 0.0)
    assert last == (1.0, 0.0)
