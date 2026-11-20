"""One-time CUDA batch calibration for each actual staged training objective."""

import copy
import gc
import time

import torch

from live_recognition.direct_cube_pose_data import collate_direct_pose

from .loss import StagedPoseLoss


def largest_fitting_batch(measure, maximum, budget):
    """Find the largest measured batch within budget; None denotes CUDA OOM."""
    trials = []

    def fits(batch):
        peak = measure(batch)
        trials.append({"batch": batch, "peak_bytes": peak})
        return peak is not None and peak <= budget

    safe = 0
    candidate = 1
    while fits(candidate):
        safe = candidate
        if candidate == maximum:
            return safe, trials
        candidate = min(maximum, candidate * 2)
    if not safe:
        raise RuntimeError(
            "Even batch1 exceeds the CUDA memory budget; reduce --imgsz or free GPU memory"
        )
    low, high = safe + 1, candidate - 1
    while low <= high:
        candidate = (low + high) // 2
        if fits(candidate):
            safe, low = candidate, candidate + 1
        else:
            high = candidate - 1
    return safe, trials


def probe_training_step(model, criterion, samples, batch_size, device, phase, optimizer_factory):
    """Exercise loss, metrics, backward, clipping and lazy AdamW/AMP allocations.

    Called only on a disposable model copy. Optimizer/scaler are local so an
    OOM during state initialization cannot poison a subsequent smaller trial.
    Input images do NOT require gradients when the source is frozen.
    """
    optimizer = optimizer_factory(model, phase)
    # A memory probe needs allocated optimizer states, not scale warm-up. A
    # unit scale avoids overflowing disposable AMP gradients at initialization.
    scaler = torch.amp.GradScaler("cuda", enabled=getattr(model, "use_amp", False), init_scale=1.0)
    model.train()
    try:
        batch, target, _keys = collate_direct_pose(
            [samples[i % len(samples)] for i in range(batch_size)]
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        target = {k: v.to(device) for k, v in target.items()}
        loss, _parts, context = criterion(model, batch, target, phase)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite {phase} loss during batch calibration")
        with torch.no_grad():
            criterion.metrics(model, context, target, batch["img"].shape[-1])
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        for group in optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group["params"], 10, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
    finally:
        model.zero_grad(set_to_none=True)


def resolve_phase_batch_size(
    requested,
    model,
    device,
    phase,
    dataset,
    symmetries,
    optimizer_factory,
    memory_fraction=0.85,
    maximum=1024,
    criterion_factory=StagedPoseLoss,
):
    if requested != -1:
        return requested, None
    maximum = min(maximum, len(dataset))
    if device.type != "cuda":
        batch = min(16, maximum)
        print(
            f"{phase}: CUDA auto-sizing is unavailable on {device.type}; using batch {batch}",
            flush=True,
        )
        return batch, None

    print(
        f"{phase}: calibrating CUDA batch (real loss/backward/AdamW; {memory_fraction:.0%} free-memory budget)",
        flush=True,
    )

    # Include densely annotated real training images in every trial. This
    # exercises the shipped detector loss and real geometric target shapes.
    order = sorted(range(len(dataset)), key=lambda i: len(dataset.labels[i]), reverse=True)
    samples = [dataset[i] for i in order[:8]]
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated(device)
    free, _total = torch.cuda.mem_get_info(device)
    budget = int(baseline + free * memory_fraction)
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    started = time.monotonic()
    report = None
    probe = probe_criterion = None
    # Real parameters and every random stream used by training stay intact.
    with torch.random.fork_rng(devices=[device_index]):
        try:
            probe = copy.deepcopy(model)
            probe_criterion = criterion_factory(probe, symmetries).to(device)
            duplicate_bytes = torch.cuda.memory_allocated(device) - baseline

            def measure(batch_size):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                try:
                    probe_training_step(
                        probe,
                        probe_criterion,
                        samples,
                        batch_size,
                        device,
                        phase,
                        optimizer_factory,
                    )
                    torch.cuda.synchronize(device)
                    # Remove only the extra fixed model/loss copy; retain the
                    # REAL optimizer moments, gradients and step temporaries.
                    return max(0, torch.cuda.max_memory_allocated(device) - duplicate_bytes)
                except torch.cuda.OutOfMemoryError:
                    return None

            batch, trials = largest_fitting_batch(measure, maximum, budget)
            report = {
                "phase": phase,
                "batch": batch,
                "memory_fraction": memory_fraction,
                "budget_bytes": budget,
                "trials": trials,
                "seconds": time.monotonic() - started,
                "method": "Real stage loss/metrics/backward/AdamW step on disposable copy; active freeze mask and detector AMP; original weights/RNG unchanged",
            }
        finally:
            probe = probe_criterion = None
            gc.collect()
            torch.cuda.empty_cache()
    print(
        f"{phase}: auto batch={batch} (measured full stage step; {memory_fraction:.0%} free-memory budget)",
        flush=True,
    )
    return batch, report
