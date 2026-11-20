import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from live_recognition.direct_cube_pose import DirectPoseHead, SymmetryPoseLoss  # noqa: E402
from live_recognition.direct_cube_pose_data import landmark_template, target_tensors  # noqa: E402
from live_recognition.direct_pose_training_numerics import compute_loss_attempt  # noqa: E402
from scripts.train_yolo26_direct_cube_pose import run_epoch  # noqa: E402
from scripts import train_yolo26_direct_cube_pose as trainer  # noqa: E402
from scripts.verify_hidden_layer_ambiguity import front_only_target, make_state  # noqa: E402
from scripts.export_direct_cube_pose_dataset import symmetry_manifest  # noqa: E402
from scripts.check_direct_pose_batch_numerics import classification, evaluate_case, load_saved_batch, trace_forward  # noqa: E402
from types import SimpleNamespace  # noqa: E402


class NumericalProbeModel(torch.nn.Module):
    def __init__(self, failure=None):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 1)
        self.bn = torch.nn.BatchNorm2d(4)
        raw = torch.zeros(31)
        raw[:6] = torch.tensor((1, 0, 0, 0, 1, 0))
        raw[6:8] = .5
        raw[8] = -2.
        raw[9] = -1.4
        raw[10:12] = .5
        raw[18:30] = .5
        self.raw = torch.nn.Parameter(raw)
        self.failure = failure
        self.forward_modes = []

    def forward(self, images):
        amp = torch.is_autocast_enabled("cpu")
        self.forward_modes.append(amp)
        feature = self.bn(self.conv(images))
        scores = feature.mean((2, 3)) * 1000
        if self.failure == "always" or self.failure == "amp" and amp:
            scores = scores * float("inf")
        with torch.autocast("cpu", enabled=False):
            pose = DirectPoseHead.decode(self.raw[None].expand(len(images), -1))
        return {"independent": {"scores": scores, "feats": [feature]}, "pose": pose}


class PrecisionCheckingLoss:
    @staticmethod
    def raw_predictions(value):
        return value

    def __call__(self, raw, batch):
        assert not torch.is_autocast_enabled("cpu")
        assert raw["scores"].dtype == torch.float32
        assert raw["feats"][0].dtype == torch.float32
        # This reduction can overflow when evaluated on FP16 inputs.
        # Normalize after the FP32 reduction, keeping this synthetic probe's
        # unscaled FP16 backward in range as well as its loss computation.
        loss = raw["scores"].square().mean().reshape(1) / 100
        return loss, loss.detach()


class StandardPrecisionLoss(PrecisionCheckingLoss):
    def __call__(self, raw, batch):
        loss = raw["scores"].square().mean().reshape(1)
        return loss, loss.detach()


def loss_inputs():
    payload, _ = front_only_target(make_state())
    target = target_tensors(payload)
    target = {key: value[None].expand(2, *value.shape) for key, value in target.items()}
    batch = {"img": torch.rand(2, 3, 16, 16)}
    criterion = SymmetryPoseLoss(symmetry_manifest(), *landmark_template())
    return batch, target, criterion


def test_mixed_precision_forward_keeps_loss_math_in_float32_with_gradients():
    model = NumericalProbeModel()
    with torch.no_grad():
        # Large enough to overflow a FP16 squared loss, without depending on
        # random BatchNorm means or the RNG state left by preceding tests.
        model.conv.weight.copy_(torch.tensor([[.5, 0, 0], [0, .5, 0], [0, 0, .5], [.3, .3, .3]])[:, :, None, None])
        model.conv.bias.zero_()
        model.bn.bias.fill_(1)
    batch, target, direct = loss_inputs()
    attempt = compute_loss_attempt(model, batch, target, PrecisionCheckingLoss(), direct,
                                   torch.device("cpu"), amp_enabled=True, amp_dtype=torch.float16)
    assert not attempt["nonfinite"]
    assert attempt["total"].dtype == torch.float32
    attempt["total"].backward()
    assert model.conv.weight.grad is not None
    assert torch.isfinite(model.conv.weight.grad).all()
    assert model.raw.grad is not None and torch.isfinite(model.raw.grad).all()


def test_training_does_not_retry_or_scan_outputs_on_its_normal_path(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Normal training must not run expensive diagnostics")

    monkeypatch.setattr(trainer, "attempt_diagnostics", forbidden)
    monkeypatch.setattr(trainer, "nonfinite_fields", forbidden)
    model = NumericalProbeModel()
    batch, target, direct = loss_inputs()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    metrics = run_epoch(model, [(batch, target, ("a", "b"))], StandardPrecisionLoss(), direct,
                        torch.device("cpu"), optimizer, scaler)
    assert model.forward_modes == [False]
    assert model.bn.num_batches_tracked.item() == 1
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_training_stops_on_nonfinite_amp_loss_without_retrying_or_skipping(tmp_path):
    model = NumericalProbeModel(failure="amp")
    batch, target, direct = loss_inputs()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    with pytest.raises(FloatingPointError, match="Nonfinite loss"):
        run_epoch(model, [(batch, target, ("a", "b"))], StandardPrecisionLoss(), direct,
                  torch.device("cpu"), optimizer, scaler, diagnostic_dir=tmp_path)
    assert model.forward_modes == [True]
    assert model.bn.num_batches_tracked.item() == 1
    diagnostic = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert diagnostic["keys"] == ["a", "b"]
    assert "float32_retry" not in diagnostic


def test_persistent_nonfinite_outputs_are_reported_and_batch_saved(tmp_path):
    model = NumericalProbeModel(failure="always")
    batch, target, direct = loss_inputs()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    with pytest.raises(FloatingPointError, match=r"yolo"):
        run_epoch(model, [(batch, target, ("bad", "batch"))], StandardPrecisionLoss(), direct,
                  torch.device("cpu"), optimizer, scaler, diagnostic_dir=tmp_path)
    diagnostic = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert "yolo" in diagnostic["nonfinite_fields"]
    assert diagnostic["outputs"]["independent.scores"]["nonfinite_values"] > 0
    assert len(list(tmp_path.glob("*_batch.pt"))) == 1


def test_standalone_probe_isolates_loss_precision_on_identical_predictions():
    batch, target, direct = loss_inputs()
    criterion = StandardPrecisionLoss()
    criterion.flat = SimpleNamespace()
    criterion.end_to_end = SimpleNamespace(one2many=criterion.flat, one2one=criterion.flat)
    model = NumericalProbeModel().eval()
    with torch.no_grad():
        prediction = model(batch["img"])
    # Finite half-precision predictions whose squared loss exceeds FP16 range.
    prediction["independent"]["scores"] = torch.full((2, 4), 1000., dtype=torch.float16)
    original = evaluate_case(prediction, batch, target, criterion, direct, torch.device("cpu"),
                             amp_loss=True, fp32_inputs=False)
    loss32 = evaluate_case(prediction, batch, target, criterion, direct, torch.device("cpu"),
                           amp_loss=False, fp32_inputs=True)
    assert not original["passed"] and loss32["passed"]
    assert "AMP loss arithmetic" in classification({"original_amp": original,
                                                     "amp_forward_fp32_losses": loss32, "all_fp32": loss32})
    assert prediction["independent"]["scores"].dtype == torch.float16


def test_probe_traces_first_failing_layer_and_removes_hooks():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    with torch.no_grad():
        model[0].weight.fill_(float("nan"))
    report = {}
    with trace_forward(model, report):
        model(torch.ones(1, 2))
    first = report["first_nonfinite_module"]
    assert first["name"] == "0"
    assert first["inputs"]["nonfinite_fields"] == []
    assert first["outputs"]["nonfinite_fields"]
    assert first["state"]["nonfinite_fields"] == ["weight"]
    assert all(not module._forward_hooks for module in model.modules())


def test_probe_replays_exact_saved_batch(tmp_path):
    batch, target, _ = loss_inputs()
    path = tmp_path / "failed_batch.pt"
    torch.save({"batch": batch, "target": target, "keys": ("a", "b")}, path)
    restored, restored_target, keys = load_saved_batch(path)
    assert keys == ("a", "b")
    assert torch.equal(restored["img"], batch["img"])
    assert all(torch.equal(restored_target[key], value) for key, value in target.items())


def test_epoch_moving_angle_metric_weights_faces_and_excludes_masked_images():
    model = NumericalProbeModel()
    batch, target, direct = loss_inputs()
    target = {k: torch.cat((v, v[:1])) for k, v in target.items()}
    target["angles"][:] = torch.tensor([[20., 0, 0, 0, 0, 0], [5., -15., 0, 0, 0, 0], [45., 45., 45., 0, 0, 0]])
    target["enabled"][2] = False
    images = torch.rand(3, 3, 16, 16)

    class Loader:
        dataset = SimpleNamespace(entries=[{"key": k, "mode": "none", "symmetry_count": 1} for k in ("a", "b", "c")])

        def __iter__(self):
            for indexes in ([0, 1], [2]):
                yield ({"img": images[indexes]}, {k: v[indexes] for k, v in target.items()},
                       tuple(("a", "b", "c")[i] for i in indexes))

    metrics = run_epoch(model, Loader(), StandardPrecisionLoss(), direct, torch.device("cpu"))
    assert metrics["moving_angle_mae_deg"] == pytest.approx(40 / 3)
    assert metrics["stationary_angle_mae_deg"] == 0
    assert metrics["moving_face_count"] == 3 and metrics["stationary_face_count"] == 9
    assert metrics["by_mode/none/moving_angle_mae_deg"] == pytest.approx(40 / 3)
