"""Slow NMS must never fabricate missing observations for later batch images."""

from types import SimpleNamespace

import pytest
import torch

from live_recognition.direct_cube_pose import DirectCubePoseModel
from live_recognition.staged_pose.model import IndependentPointSource


@pytest.mark.parametrize("empty_middle", [False, True])
def test_every_image_processed_despite_simulated_nms_deadline(monkeypatch, empty_middle):
    from ultralytics.utils import nms

    # Two non-overlapping sticker objects per image. The third keypoint channel
    # is presence; the .95 detection/class score determines native ranking.
    decoded = torch.zeros(3, 16, 2)
    centers = torch.tensor([[32.0, 64.0], [32.0, 64.0]])
    decoded[:, :2] = centers
    decoded[:, 2:4] = 8.0
    decoded[:, 4 + 3] = 0.95
    decoded[:, 13:15] = centers
    decoded[:, 15] = 0.99
    if empty_middle:
        decoded[1, 4:13] = 0.0

    class Head(torch.nn.Module):
        def _inference(self, _branch):
            return decoded.clone()

    class Detector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.ModuleList([Head()])

        def forward(self, images):
            return {"feats": [images]}

    monkeypatch.setattr(
        DirectCubePoseModel,
        "_decode_observation_tokens",
        lambda *_args: (
            decoded[:, 13:15].transpose(1, 2) / 128,
            decoded[:, 4:13].clone().requires_grad_(),
        ),
    )
    # Advance the NMS clock100seconds per call without an actual slow test.
    # With the default timeout this exits after image0 and loses image2.
    ticks = iter(range(0, 10000, 100))
    monkeypatch.setattr(nms, "time", SimpleNamespace(time=lambda: next(ticks)))
    actual_nms = nms.non_max_suppression

    def complete_cpu_nms(prediction, **kwargs):
        assert prediction.device.type == "cpu" and prediction.dtype == torch.float32
        assert kwargs["max_time_img"] == float("inf")
        return actual_nms(prediction, **kwargs)

    monkeypatch.setattr(nms, "non_max_suppression", complete_cpu_nms)
    result = IndependentPointSource(Detector())(torch.zeros(3, 3, 128, 128))
    expected = [2, 0 if empty_middle else 2, 2]
    assert result["native"]["mask"].sum(-1).tolist() == expected
    assert result["object_scores"].requires_grad
    result["object_scores"].sum().backward()
