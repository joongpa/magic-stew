from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.train_yolo26_direct_cube_pose import (  # noqa: E402
    BatchSizeProbe, build_parser, normalize_training_arguments, resolve_batch_size,
)


def test_existing_trainer_arguments_round_144_and_accept_automatic_batch(capsys):
    args = normalize_training_arguments(build_parser().parse_args(["--imgsz", "144", "--batch", "-1"]))
    assert args.imgsz == 160 and args.batch == -1
    assert "144 to 160" in capsys.readouterr().out


def test_resolution_alias_and_aligned_size_remain_unchanged(capsys):
    args = normalize_training_arguments(build_parser().parse_args(["--resolution", "128", "--batch", "8"]))
    assert args.imgsz == 128 and args.batch == 8
    assert not capsys.readouterr().out


@pytest.mark.parametrize("flag,value", [("--batch", "0"), ("--batch", "-2"), ("--imgsz", "0"), ("--epochs", "0")])
def test_invalid_argument_names_are_reported_individually(flag, value):
    with pytest.raises(ValueError, match=flag):
        normalize_training_arguments(build_parser().parse_args([flag, value]))


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_auto_batch_has_explicit_cpu_and_mps_fallback(device, capsys):
    assert resolve_batch_size(-1, None, torch.device(device), 160, True, 50, 100) == 16
    assert "using batch 16" in capsys.readouterr().out
    assert resolve_batch_size(8, None, torch.device(device), 160, True, 50, 100) == 8


class TinyNestedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.detector = SimpleNamespace(yaml={"channels": 3}, stride=torch.tensor([8., 16., 32.]))

    def forward(self, images):
        value = images * self.weight
        return {"independent": ({"scores": value}, [value.mean()]), "pose": {"center": value.mean()}}


def test_auto_batch_probe_supports_backward_through_nested_model_outputs():
    model = TinyNestedModel()
    value = BatchSizeProbe(model)(torch.ones(2, 3, 64, 64))
    assert value.ndim == 0
    value.backward()
    assert model.weight.grad is not None and model.weight.grad.abs().sum() > 0


def test_cuda_auto_batch_profiles_the_custom_wrapper(monkeypatch):
    from ultralytics.utils import autobatch

    captured = {}

    def estimate(model, **kwargs):
        captured.update(kwargs)
        assert isinstance(model, BatchSizeProbe)
        return 7

    monkeypatch.setattr(autobatch, "check_train_batch_size", estimate)
    result = resolve_batch_size(-1, TinyNestedModel(), torch.device("cuda:0"), 160, True, 57, 120)
    assert result == 7
    assert captured == {"imgsz": 160, "amp": True, "batch": -1, "max_num_obj": 57, "dataset_size": 120}
