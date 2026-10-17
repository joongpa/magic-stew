import torch
from torch import nn

from scripts.train_yolo_cube_pose_local import scheduled_loss_weights

from live_recognition.cube_pose import (
    ANGLE_LIMIT_DEGREES,
    CubePoseHead,
    YOLOCubePoseModel,
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


def test_pose_head_decodes_bounded_outputs():
    head = CubePoseHead((32, 16))
    output = head(
        (
            torch.rand(2, 32, 16, 16),
            torch.rand(2, 16, 32, 32),
        ),
        torch.rand(2, 3, 16, 16),
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


def test_soft_segmentation_maps_return_unions_and_sticker_boundary():
    model = YOLOCubePoseModel(
        FakeSegmentationModel(),
        (32, 16),
        freeze_base=False,
    )
    maps = model._soft_segmentation_maps(
        model.base_model(torch.rand(1, 3, 32, 32))
    )

    assert maps.shape == (1, 3, 8, 8)
    assert torch.all((maps >= 0.0) & (maps <= 1.0))

    mask_probabilities = torch.zeros(1, 1, 5, 5)
    mask_probabilities[0, 0, 1:4, 1:4] = 1.0
    border = model._one_pixel_inner_border(mask_probabilities)[0, 0]
    expected_border = torch.zeros(5, 5)
    expected_border[1:4, 1:4] = 1.0
    expected_border[2, 2] = 0.0
    assert torch.allclose(border, expected_border)


def test_pose_loss_is_zero_for_matching_predictions():
    quaternion = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    prediction = {
        "rotation_quaternion": quaternion,
        "image_center": torch.tensor([[0.5, 0.5]]),
        "image_scale": torch.tensor([0.25]),
        "perspective_strength": torch.tensor([0.75]),
        "layer_angles": torch.tensor([[12.0, 0.0]]),
    }
    losses = cube_pose_loss(prediction, prediction)
    assert losses["total"] < 1e-6


def test_projected_geometry_loss_is_zero_for_matching_pose():
    pose = {
        "rotation_quaternion": torch.tensor([[0.13302687, -0.16872216, 0.23081309, 0.94897945]]),
        "image_center": torch.tensor([[0.47, 0.53]]),
        "image_scale": torch.tensor([0.25]),
        "perspective_strength": torch.tensor([0.75]),
        "layer_angles": torch.tensor([[12.0, -8.0]]),
    }

    assert cube_pose_geometry_loss(pose, pose) < 1e-6


def test_projected_geometry_loss_increases_with_visible_displacement():
    target = {
        "rotation_quaternion": torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        "image_center": torch.tensor([[0.5, 0.5]]),
        "image_scale": torch.tensor([0.25]),
        "perspective_strength": torch.tensor([0.75]),
        "layer_angles": torch.tensor([[0.0, 0.0]]),
    }
    small = {**target, "image_center": torch.tensor([[0.505, 0.5]])}
    large = {**target, "image_center": torch.tensor([[0.55, 0.5]])}

    assert cube_pose_geometry_loss(small, target) < cube_pose_geometry_loss(large, target)


def test_projected_geometry_loss_has_finite_gradient_at_exact_match():
    pose = {
        "u": torch.tensor([90.0], requires_grad=True),
        "v": torch.tensor([0.0], requires_grad=True),
        "twist": torch.tensor([12.0], requires_grad=True),
        "image_center": torch.tensor([[0.5, 0.5]], requires_grad=True),
        "image_scale": torch.tensor([0.25], requires_grad=True),
        "perspective_strength": torch.tensor([0.75], requires_grad=True),
        "layer_angles": torch.tensor([[12.0, -8.0]], requires_grad=True),
    }

    cube_pose_geometry_loss(pose, pose).backward()

    assert all(torch.isfinite(value.grad).all() for value in pose.values())


def test_pose_loss_accepts_a_ninety_degree_cube_symmetry():
    target = {
        "rotation_quaternion": torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        "image_center": torch.tensor([[0.5, 0.5]]),
        "image_scale": torch.tensor([0.25]),
        "perspective_strength": torch.tensor([0.75]),
        "layer_angles": torch.tensor([[0.0, 0.0]]),
    }
    quarter_turn = 2.0**-0.5
    prediction = {**target, "rotation_quaternion": torch.tensor([[0.0, 0.0, quarter_turn, quarter_turn]])}

    losses = cube_pose_loss(prediction, target)

    assert losses["rotation"] < 1e-6
    assert losses["total"] < 1e-6


def test_pose_loss_accepts_ninety_degree_local_rotation_for_nontrivial_pose():
    target = {
        "rotation_quaternion": torch.tensor(
            [[0.13302687, -0.16872216, 0.23081309, 0.94897945]]
        ),
        "image_center": torch.tensor([[0.5, 0.5]]),
        "image_scale": torch.tensor([0.25]),
        "perspective_strength": torch.tensor([0.75]),
        "layer_angles": torch.tensor([[0.0, 0.0]]),
    }
    prediction = {
        **target,
        "rotation_quaternion": torch.tensor(
            [[-0.02524039, -0.21336878, 0.83423930, 0.50782031]]
        ),
    }

    losses = cube_pose_loss(prediction, target)

    assert losses["rotation"] < 1e-6
    assert losses["total"] < 1e-6


def test_noncanonical_stack_axis_is_penalized_even_when_silhouette_matches():
    target = {
        "rotation_quaternion": torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        "image_center": torch.tensor([[0.5, 0.5]]),
        "image_scale": torch.tensor([0.25]),
        "perspective_strength": torch.tensor([0.75]),
        "layer_angles": torch.tensor([[0.001, -0.001]]),
    }
    quarter_turn = 2.0**-0.5
    prediction = {
        **target,
        "rotation_quaternion": torch.tensor([[quarter_turn, 0.0, 0.0, quarter_turn]]),
    }

    losses = cube_pose_loss(prediction, target)

    assert losses["rotation"] > 1e-3
    assert losses["total"] > 1e-3


def test_stack_changing_and_direct_layer_errors_are_identical():
    target = {
        "rotation_quaternion": torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        "image_center": torch.tensor([[0.5, 0.5]]),
        "image_scale": torch.tensor([0.25]),
        "perspective_strength": torch.tensor([0.75]),
        "layer_angles": torch.tensor([[12.0, -8.0]]),
    }
    direct_prediction = {
        **target,
        "layer_angles": torch.tensor([[10.0, -6.0]]),
    }
    quarter_turn = 2.0**-0.5
    stack_changed_prediction = {
        **direct_prediction,
        "rotation_quaternion": torch.tensor([[quarter_turn, 0.0, 0.0, quarter_turn]]),
    }

    direct = cube_pose_loss(direct_prediction, target)
    stack_changed = cube_pose_loss(stack_changed_prediction, target)

    assert torch.allclose(direct["layer_angles"], stack_changed["layer_angles"])


def test_turned_pose_with_swapped_layers_is_not_a_canonical_match():
    target = {
        "rotation_quaternion": torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        "image_center": torch.tensor([[0.5, 0.5]]),
        "image_scale": torch.tensor([0.25]),
        "perspective_strength": torch.tensor([0.75]),
        "layer_angles": torch.tensor([[12.0, 0.0]]),
    }
    prediction = {
        **target,
        "rotation_quaternion": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "layer_angles": torch.tensor([[0.0, -12.0]]),
    }

    losses = cube_pose_loss(prediction, target)

    assert losses["rotation"] > 1e-3
    assert losses["layer_angles"] > 1e-3


def test_turned_pose_accepts_ninety_degree_rotation_about_stack_axis():
    target = {
        "rotation_quaternion": torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        "image_center": torch.tensor([[0.5, 0.5]]),
        "image_scale": torch.tensor([0.25]),
        "perspective_strength": torch.tensor([0.75]),
        "layer_angles": torch.tensor([[12.0, 0.0]]),
    }
    quarter_turn = 2.0**-0.5
    prediction = {
        **target,
        "rotation_quaternion": torch.tensor([[0.0, 0.0, quarter_turn, quarter_turn]]),
    }

    losses = cube_pose_loss(prediction, target)

    assert losses["rotation"] < 1e-6
    assert losses["layer_angles"] < 1e-6


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


def test_loss_weight_schedule_moves_from_head_to_geometry():
    first = scheduled_loss_weights(
        1,
        10,
        head_start=0.6,
        head_end=0.3,
        geometry_start=0.4,
        geometry_end=0.7,
    )
    last = scheduled_loss_weights(
        10,
        10,
        head_start=0.6,
        head_end=0.3,
        geometry_start=0.4,
        geometry_end=0.7,
    )
    assert first == (0.6, 0.4)
    assert last == (0.3, 0.7)

