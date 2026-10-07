import numpy as np
import torch
from torch import nn

from live_recognition.color_segmentation import (
    COLOR_CLASS_NAMES,
    IGNORE_INDEX,
    explicit_color_features,
    YOLOColorMapModel,
)
from live_recognition.color_targets import dense_color_target


class FakeSegmentationHead(nn.Module):
    def forward(self, features):
        return features[0]


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


def test_explicit_color_features_include_oklab_channels():
    images = torch.rand(2, 3, 16, 16)

    features = explicit_color_features(images)

    assert features.shape == (2, 11, 16, 16)
    assert torch.isfinite(features).all()


def test_color_head_returns_full_resolution_probability_map():
    model = YOLOColorMapModel(
        FakeSegmentationModel(),
        feature_channels=(32, 16),
        freeze_base=True,
    )

    output = model(torch.rand(2, 3, 64, 64))

    assert output["color_logits"].shape == (2, len(COLOR_CLASS_NAMES), 64, 64)
    assert output["color_probabilities"].shape == (2, len(COLOR_CLASS_NAMES), 64, 64)
    assert torch.allclose(
        output["color_probabilities"].sum(dim=1),
        torch.ones(2, 64, 64),
        atol=1e-5,
    )


def test_dense_color_target_maps_polygons_and_ignores_boundaries():
    target = dense_color_target(
        image_shape=(100, 100),
        polygons=[
            ("Red", [(0.10, 0.10), (0.40, 0.10), (0.40, 0.40), (0.10, 0.40)]),
        ],
        class_names=["White", "Yellow", "Red", "Orange", "Blue", "Green"],
        roi_xyxy=(0, 0, 100, 100),
        output_size=20,
        boundary_ignore_px=1,
    )

    red_index = COLOR_CLASS_NAMES.index("red")
    assert target[5, 5] == red_index
    assert target[0, 0] == COLOR_CLASS_NAMES.index("none")
    assert np.count_nonzero(target == IGNORE_INDEX) > 0
