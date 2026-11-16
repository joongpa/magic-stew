from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.train_yolo26_direct_cube_pose import (  # noqa: E402
    BatchSizeProbe, build_parser, normalize_training_arguments, resolve_batch_size, resolve_training_amp,
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


def test_failed_old_profiler_identifies_the_staged_trainer_without_claiming_oom(monkeypatch):
    from ultralytics.utils import autobatch

    def failed(*_args, **_kwargs):
        raise RuntimeError("AutoBatch: no usable batch size found while profiling batch=[1, 2, 4]")

    monkeypatch.setattr(autobatch, "check_train_batch_size", failed)
    with pytest.raises(RuntimeError, match="train_yolo26_staged_cube_pose.py") as result:
        resolve_batch_size(-1, TinyNestedModel(), torch.device("cuda:0"), 128, False, 57, 120)
    assert "does not establish" in str(result.value)
    assert isinstance(result.value.__cause__, RuntimeError)


def test_other_cuda_profiler_errors_preserve_original_cause(monkeypatch):
    from ultralytics.utils import autobatch

    original = RuntimeError("CUDA device unavailable")

    def failed(*_args, **_kwargs):
        raise original

    monkeypatch.setattr(autobatch, "check_train_batch_size", failed)
    with pytest.raises(RuntimeError) as result:
        resolve_batch_size(-1, TinyNestedModel(), torch.device("cuda:0"), 128, False, 57, 120)
    assert result.value is original


def test_stock_amp_check_rejects_reported_gtx_1660_super_without_forward(monkeypatch):
    # Exercise the actual installed Ultralytics blacklist without CUDA hardware.
    detector = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device=torch.device('cuda:0'))]))
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda device: 'NVIDIA GeForce GTX 1660 SUPER')
    assert not resolve_training_amp(True, detector, torch.device('cuda:0'))


@pytest.mark.parametrize('requested,device', [(False, 'cuda:0'), (True, 'cpu'), (True, 'mps')])
def test_disabled_or_non_cuda_amp_skips_compatibility_probe(monkeypatch, requested, device):
    from ultralytics.utils import checks

    def forbidden(_model):
        raise AssertionError('No compatibility probe needed')

    monkeypatch.setattr(checks, 'check_amp', forbidden)
    assert not resolve_training_amp(requested, None, torch.device(device))


@pytest.mark.parametrize('supported', [True, False])
def test_amp_uses_stock_check_result_and_restores_callbacks(monkeypatch, supported):
    from ultralytics.utils import checks, callbacks

    detector = object()
    original = callbacks.default_callbacks.copy()

    def check(model):
        assert model is detector
        callbacks.default_callbacks = {}
        return supported

    monkeypatch.setattr(checks, 'check_amp', check)
    assert resolve_training_amp(True, detector, torch.device('cuda:0')) == supported
    assert callbacks.default_callbacks == original
