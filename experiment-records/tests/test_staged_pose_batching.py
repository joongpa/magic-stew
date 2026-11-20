"""Stage memory sizing, fixed overrides, resume policy and full-step probing."""

import copy
from types import SimpleNamespace

import pytest
import torch

from live_recognition.staged_pose.batching import largest_fitting_batch, probe_training_step
from scripts.train_yolo26_staged_cube_pose import (
    batch_cli_overrides,
    build_parser,
    restore_phase_batches,
    stage_batch_request,
    validate_args,
)


def test_search_measures_largest_batch_within_budget_and_handles_oom():
    batch, trials = largest_fitting_batch(lambda b: None if b >= 24 else 100 + 10 * b, 64, 270)
    assert batch == 17 and {t["batch"] for t in trials} >= {1, 16, 17, 18}
    batch, _ = largest_fitting_batch(lambda b: None if b >= 24 else 100 + 10 * b, 64, 10000)
    assert batch == 23


def test_search_respects_cap_and_does_not_guess_when_batch1_fails():
    assert largest_fitting_batch(lambda b: b, 11, 100)[0] == 11
    with pytest.raises(RuntimeError, match="batch1"):
        largest_fitting_batch(lambda _b: None, 64, 100)


def test_independent_stage_batch_flags_and_global_auto():
    args = validate_args(
        build_parser().parse_args(
            [
                "--batch",
                "-1",
                "--articulation-batch",
                "128",
                "--handoff-batch",
                "256",
                "--joint-batch",
                "16",
            ]
        )
    )
    assert [
        stage_batch_request(args, p) for p in ("static", "articulation", "handoff", "joint")
    ] == [-1, 128, 256, 16]
    with pytest.raises(ValueError, match="--joint-batch"):
        validate_args(build_parser().parse_args(["--joint-batch", "0"]))


@pytest.mark.parametrize(
    "flags",
    [["--batch-memory-fraction", "0"], ["--batch-memory-fraction", "1"], ["--max-auto-batch", "0"]],
)
def test_invalid_memory_policy(flags):
    with pytest.raises(ValueError):
        validate_args(build_parser().parse_args(flags))


def test_resume_keeps_resolved_sizes_until_explicitly_overridden():
    saved = {
        "phase": "handoff",
        "train_args": {"batch": 16},
        "phase_batch_sizes": {"static": 32, "articulation": 128, "handoff": 128, "joint": 16},
    }
    args = build_parser().parse_args([])
    assert restore_phase_batches(saved, args, {}) == saved["phase_batch_sizes"]
    argv = ["--joint-batch", "8"]
    args = build_parser().parse_args(argv)
    assert restore_phase_batches(saved, args, batch_cli_overrides(args, argv)) == {
        "static": 32,
        "articulation": 128,
        "handoff": 128,
    }
    argv = ["--batch=-1"]
    args = build_parser().parse_args(argv)
    assert restore_phase_batches(saved, args, batch_cli_overrides(args, argv)) == {}
    old = {"phase": "articulation", "train_args": {"batch": 24}}
    assert restore_phase_batches(old, args, {}) == {"articulation": 24}


def test_probe_runs_backward_and_real_adam_states_on_disposable_copy():
    class Criterion:
        called = False

        def __call__(self, model, batch, _target, phase):
            assert phase == "articulation" and not batch["img"].requires_grad
            loss = (model(batch["img"]).mean() - 1).square()
            return loss, {}, None

        def metrics(self, *_args):
            self.called = True
            return {}

    model = torch.nn.Conv2d(3, 1, 1)
    probe = copy.deepcopy(model)
    before = model.weight.detach().clone()
    criterion = Criterion()
    optimizers = []

    def factory(m, _phase):
        optimizer = torch.optim.AdamW(m.parameters(), lr=0.01)
        optimizers.append(optimizer)
        return optimizer

    sample = (torch.zeros(3, 8, 8), {"enabled": torch.tensor(True)}, torch.empty(0, 8), "sample")
    probe_training_step(probe, criterion, [sample], 4, torch.device("cpu"), "articulation", factory)
    assert criterion.called and torch.equal(model.weight, before)
    assert optimizers[0].state and all("exp_avg_sq" in v for v in optimizers[0].state.values())
    assert all(p.grad is None for p in probe.parameters())
    assert not torch.equal(model.bias, probe.bias)


def test_cuda_calibration_accounts_for_copy_and_preserves_training_rng(monkeypatch):
    from contextlib import contextmanager

    from live_recognition.staged_pose import batching

    model = torch.nn.Linear(1, 1)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    sample = (torch.ones(3, 8, 8), {}, torch.empty(0, 8), "sample")

    class Dataset:
        labels = [torch.empty(1, 8)] * 64

        def __len__(self):
            return 64

        def __getitem__(self, _i):
            return sample

    class Criterion:
        def __init__(self, *_args):
            pass

        def to(self, _device):
            return self

    active = [0]
    allocated = iter([100, 200])
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _d: next(allocated))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _d: (1000, 4000))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _d: 200 + 10 * active[0])
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _d: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _d: None)
    real_fork = torch.random.fork_rng

    @contextmanager
    def fork(devices):
        assert devices == [0]
        with real_fork(devices=[]):
            yield

    monkeypatch.setattr(torch.random, "fork_rng", fork)

    def step(probe, _criterion, _samples, batch, *_args):
        active[0] = batch
        with torch.no_grad():
            probe.weight.add_(torch.rand_like(probe.weight))

    monkeypatch.setattr(batching, "probe_training_step", step)
    before = torch.get_rng_state().clone()
    result, report = batching.resolve_phase_batch_size(
        -1,
        model,
        SimpleNamespace(type="cuda", index=0),
        "articulation",
        Dataset(),
        {},
        None,
        memory_fraction=0.5,
        maximum=64,
        criterion_factory=Criterion,
    )
    assert result == 50 and report["budget_bytes"] == 600
    assert torch.equal(torch.get_rng_state(), before)
    assert all(torch.equal(v, original[k]) for k, v in model.state_dict().items())
