from __future__ import annotations

import torch
from torch import Tensor, nn


class TinyCubePoseNet(nn.Module):
    """Small placeholder network for cube mask + keypoint heatmaps.

    This is intentionally minimal so the POC has a PyTorch extension point. It is not trained
    and should not be used for meaningful predictions until a dataset and training loop exist.
    """

    def __init__(self, keypoint_count: int = 8) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.mask_head = nn.Conv2d(64, 1, kernel_size=1)
        self.keypoint_head = nn.Conv2d(64, keypoint_count, kernel_size=1)

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        features = self.encoder(images)
        return {
            "mask_logits": self.mask_head(features),
            "keypoint_heatmaps": self.keypoint_head(features),
        }


def load_pose_model(checkpoint_path: str | None = None, *, device: str = "cpu") -> TinyCubePoseNet:
    model = TinyCubePoseNet().to(device)
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    return model

