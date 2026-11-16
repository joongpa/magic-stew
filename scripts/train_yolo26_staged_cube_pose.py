#!/usr/bin/env python3
"""Train the exact staged45 architecture on an exported dataset (Windows/CUDA).

python scripts/train_yolo26_staged_cube_pose.py --dataset data/direct_cube_pose_dataset \
    --init-checkpoint runs/staged_cube_pose_epoch45_init.pt --device 0 --batch 16

All four phases run automatically: static, known-static articulation, robust
handoff, full joint fine-tuning. The shared detector and both heads use ONE
current RGB encoder pass; original default independent YOLO loss is retained.
Checkpoints include detector architecture/weights and both custom heads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import random
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from live_recognition.direct_cube_pose import make_detector
from live_recognition.direct_cube_pose_data import collate_direct_pose, read_manifest, split_entries
from live_recognition.staged_pose.batching import resolve_phase_batch_size
from live_recognition.staged_pose.data import StagedPoseDataset
from live_recognition.staged_pose.loss import StagedPoseLoss
from live_recognition.staged_pose.model import (
    OBSERVATION_VERSION,
    StagedCubePoseModel,
    checkpoint_model,
    model_from_checkpoint,
)
from live_recognition.training_progress import ConsoleProgress
from scripts.train_yolo26_direct_cube_pose import (
    resolve_training_amp,
    seed_worker,
    select_device,
)

PHASES = ("static", "articulation", "handoff", "joint")
BATCH_FIELDS = ("batch", *(f"{phase}_batch" for phase in PHASES))
BATCH_POLICY_FIELDS = ("batch_memory_fraction", "max_auto_batch")


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dataset", "--source-dir", type=Path, default=Path("data/direct_cube_pose_dataset")
    )
    p.add_argument("--model", default="runs/pose/runs/26m-pose/weights/best.pt")
    p.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Portable staged checkpoint; warm-start all components on a NEW split/run",
    )
    p.add_argument(
        "--static-checkpoint",
        type=Path,
        help="Optional original experimental static/source checkpoint",
    )
    p.add_argument(
        "--articulation-checkpoint",
        type=Path,
        help="Optional original experimental ImageContextPhaseHead checkpoint",
    )
    p.add_argument(
        "--initial-split",
        type=Path,
        help="Original train/val split for experimental component checkpoints; prevents validation overlap",
    )
    p.add_argument(
        "--export-init",
        type=Path,
        help="Bundle the supplied starting components, then exit without training",
    )
    p.add_argument(
        "--resume",
        type=Path,
        help="Continue interrupted training, including optimizer, schedule, split and RNG",
    )
    p.add_argument("--output-dir", type=Path, default=Path("runs/staged_cube_pose"))
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Default stage batch; -1 calibrates separately at each CUDA stage",
    )
    for phase in PHASES:
        p.add_argument(
            f"--{phase}-batch",
            type=int,
            help="Override this stage's batch; -1 auto-sizes this stage",
        )
    p.add_argument(
        "--batch-memory-fraction",
        type=float,
        default=0.85,
        help="Fraction of free CUDA memory used by automatic stage sizing",
    )
    p.add_argument(
        "--max-auto-batch",
        type=int,
        default=1024,
        help="Upper bound for automatic stage batch sizes",
    )
    p.add_argument("--imgsz", "--resolution", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument(
        "--train-metrics-every",
        type=int,
        default=10,
        help="Compute training pose diagnostics every N batches; all loss updates and validation batches remain complete",
    )
    p.add_argument("--static-epochs", type=int, default=30)
    p.add_argument("--articulation-epochs", type=int, default=40)
    p.add_argument("--handoff-epochs", type=int, default=30)
    p.add_argument("--joint-epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--limit", type=int, help="Representative small subset for smoke tests")
    p.add_argument(
        "--max-batches", type=int, help="Bound both train and validation for smoke tests"
    )
    p.add_argument(
        "--progress-test",
        action="store_true",
        help="Test console replacement for two seconds without loading weights/data or training",
    )
    return p


def validate_args(args):
    if args.imgsz < 64:
        raise ValueError("--imgsz must be at least64")
    rounded = math.ceil(args.imgsz / 32) * 32
    if rounded != args.imgsz:
        print(f"Adjusting --imgsz {args.imgsz} to {rounded} (stride32)", flush=True)
    args.imgsz = rounded
    validate_batch_args(args)
    if args.workers < 0 or args.threads < 1:
        raise ValueError("Invalid --workers/--threads")
    if args.train_metrics_every < 1:
        raise ValueError("--train-metrics-every must be positive")
    if any(getattr(args, f"{phase}_epochs") < 0 for phase in PHASES):
        raise ValueError("Stage epoch counts must be nonnegative")
    if not args.export_init and not any(getattr(args, f"{phase}_epochs") for phase in PHASES):
        raise ValueError("At least one training phase is required")
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between0 and1")
    if args.limit is not None and args.limit < 2:
        raise ValueError("--limit must be at least2")
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError("--max-batches must be positive")
    if args.init_checkpoint and (args.static_checkpoint or args.articulation_checkpoint):
        raise ValueError("Use --init-checkpoint OR individual starting checkpoints")
    if args.resume and (
        args.init_checkpoint
        or args.static_checkpoint
        or args.articulation_checkpoint
        or args.export_init
    ):
        raise ValueError("--resume cannot be combined with initialization/export flags")
    if (args.static_checkpoint or args.articulation_checkpoint) and not args.initial_split:
        raise ValueError(
            "Individual component checkpoints require --initial-split with their original training keys"
        )
    return args


def validate_batch_args(args):
    for name in BATCH_FIELDS:
        value = getattr(args, name)
        if value is not None and value != -1 and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive or -1")
    if not 0 < args.batch_memory_fraction < 1 or args.max_auto_batch < 1:
        raise ValueError(
            "--batch-memory-fraction must be between0 and1; --max-auto-batch must be positive"
        )


def initialize_model(args):
    if args.init_checkpoint:
        model, _ = model_from_checkpoint(args.init_checkpoint)
    else:
        model = StagedCubePoseModel(make_detector(args.model))
        if args.static_checkpoint:
            saved = torch.load(args.static_checkpoint, weights_only=False, map_location="cpu")
            model.source.load_state_dict(saved["source_state"])
            model.static_head.load_state_dict(saved["static_state"])
        if args.articulation_checkpoint:
            saved = torch.load(args.articulation_checkpoint, weights_only=False, map_location="cpu")
            model.articulation_head.load_state_dict(saved["model_state"])
        model.initialization = {
            "model": args.model,
            "static_checkpoint": str(args.static_checkpoint),
            "articulation_checkpoint": str(args.articulation_checkpoint),
        }
        if args.initial_split:
            split = json.loads(args.initial_split.read_text(encoding="utf-8"))
            model.initialization.update(
                training_keys=split["train"], development_keys=split.get("val", [])
            )
    return model


def atomic_save(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def split_for_initialization(entries, initialization, val_fraction, seed, limit=None):
    train, val = split_entries(entries, val_fraction, seed, limit)
    previous = set(initialization.get("training_keys", []))
    train.extend(e for e in val if e["key"] in previous)
    val = [e for e in val if e["key"] not in previous]
    if not val:
        candidate = next((i for i, e in enumerate(train) if e["key"] not in previous), None)
        if candidate is None:
            raise ValueError(
                "No validation images remain outside the initialization training set; provide more data or a larger --limit"
            )
        val.append(train.pop(candidate))
    return train, val


def optimizer_for(model, phase):
    model.configure_phase(phase)
    if phase == "static":
        rates = {"static": 1e-4, "source": 2e-5, "turn": 0.0}
    elif phase == "joint":
        rates = {"static": 1e-5, "source": 2e-6, "turn": 5e-5}
    else:
        rates = {"static": 0.0, "source": 0.0, "turn": 5e-4}
    groups = []
    for name, module in [
        ("static", model.static_head),
        ("source", model.source),
        ("turn", model.articulation_head),
    ]:
        parameters = [p for p in module.parameters() if p.requires_grad]
        if parameters:
            groups.append({"params": parameters, "lr": rates[name], "label": name})
    return torch.optim.AdamW(groups, weight_decay=0.001)


def stage_batch_request(args, phase):
    override = getattr(args, f"{phase}_batch", None)
    return args.batch if override is None else override


def batch_cli_overrides(args, argv):
    flags = {token.split("=", 1)[0] for token in argv}
    return {
        name: getattr(args, name)
        for name in (*BATCH_FIELDS, *BATCH_POLICY_FIELDS)
        if "--" + name.replace("_", "-") in flags
    }


def restore_phase_batches(resume, args, overrides):
    resolved = dict(resume.get("phase_batch_sizes", {})) if resume else {}
    if resume and not resolved:
        # Older checkpoints used one resolved batch for every stage.
        resolved[resume["phase"]] = resume["train_args"]["batch"]
    for phase in PHASES:
        field = f"{phase}_batch"
        if field in overrides or ("batch" in overrides and getattr(args, field, None) is None):
            resolved.pop(phase, None)
        if (
            any(name in overrides for name in BATCH_POLICY_FIELDS)
            and stage_batch_request(args, phase) == -1
        ):
            resolved.pop(phase, None)
    return resolved


def stage_loaders(datasets, batch, args, device, sampler_rng, worker_rngs):
    return [
        DataLoader(
            d,
            batch_size=batch,
            sampler=RandomSampler(d, generator=sampler_rng) if i == 0 else None,
            num_workers=args.workers,
            collate_fn=collate_direct_pose,
            worker_init_fn=seed_worker,
            generator=worker_rngs[i],
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
            **({"multiprocessing_context": "spawn"} if args.workers else {}),
        )
        for i, d in enumerate(datasets)
    ]


def run_epoch(
    model, loader, criterion, device, phase, optimizer, scaler, max_batches, metrics_every=10
):
    training = optimizer is not None
    model.train(training)
    sums = {}
    counts = {}
    batches = min(len(loader), max_batches) if max_batches is not None else len(loader)
    metric_batches = metric_images = processed_images = 0
    started = time.monotonic()
    with ConsoleProgress() as progress:
        for index, (batch, target, keys) in enumerate(loader):
            if index >= batches:
                break
            batch = {k: v.to(device, non_blocking=device.type == "cuda") for k, v in batch.items()}
            target = {
                k: v.to(device, non_blocking=device.type == "cuda") for k, v in target.items()
            }
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                loss, parts, context = criterion(model, batch, target, phase)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Nonfinite {phase} loss for {keys}: "
                        + str({k: float(v.detach()) for k, v in parts.items()})
                    )
                batch_metrics = {}
                if not training or index % metrics_every == 0 or index + 1 == batches:
                    with torch.no_grad():
                        batch_metrics = criterion.metrics(
                            model, context, target, batch["img"].shape[-1]
                        )
                    metric_batches += 1
                    metric_images += len(keys)
                if training:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    for group in optimizer.param_groups:
                        torch.nn.utils.clip_grad_norm_(group["params"], 10, error_if_nonfinite=True)
                    scaler.step(optimizer)
                    scaler.update()
            for name, value in dict(total=loss, **parts).items():
                sums[name] = sums.get(name, 0.0) + float(value.detach()) * len(keys)
                counts[name] = counts.get(name, 0) + len(keys)
            processed_images += len(keys)
            # Metrics use the actual RGB pose/own predicted axis, even during
            # teacher-conditioned training phases. GT-only diagnostic is named.
            for name, (value, count) in batch_metrics.items():
                sums[name] = sums.get(name, 0.0) + value
                counts[name] = counts.get(name, 0) + count
            if index % 10 == 0 or index + 1 == batches:
                moving = sums.get("moving_turn_deg", 0) / max(counts.get("moving_turn_deg", 0), 1)
                rotation = sums.get("static_rotation_deg", 0) / max(
                    counts.get("static_rotation_deg", 0), 1
                )
                visible = sums.get("visible_keypoint_rmse_px", 0) / max(
                    counts.get("visible_keypoint_rmse_px", 0), 1
                )
                elapsed = time.monotonic() - started
                rate = processed_images / max(elapsed, 1e-6)
                remaining = int(elapsed / (index + 1) * (batches - index - 1))
                message = f"{'train' if training else 'val'} {index + 1}/{batches} {rate:.1f}img/s ETA={remaining // 60}:{remaining % 60:02d} loss={float(loss.detach()):.3f} R={rotation:.1f} turn={moving:.1f} vis={visible:.1f}px"
                progress.update(message)
    result = {k: sums[k] / counts[k] if counts[k] else None for k in sums}
    result.update(
        pose_metric_batches=metric_batches,
        pose_metric_images=metric_images,
        images_per_second=processed_images / max(time.monotonic() - started, 1e-6),
    )
    return result


def selection_score(metrics, phase):
    if phase == "static":
        return metrics["static_rotation_deg"]
    if phase == "articulation":
        return (
            metrics["teacher_observed_moving_turn_deg"]
            if metrics["teacher_observed_moving_turn_deg"] is not None
            else metrics["phase"]
        )
    return (
        metrics["moving_turn_deg"] if metrics["moving_turn_deg"] is not None else metrics["total"]
    )


def main():
    argv = sys.argv[1:]
    args = build_parser().parse_args(argv)
    if args.progress_test:
        with ConsoleProgress() as progress:
            print(f"Batch progress: {progress.backend}", flush=True)
            for batch in range(1, 21):
                progress.update(f"batch {batch}/20 loss=1.234 R=12.3 turn=17.0 vis=6.2px")
                time.sleep(0.1)
        return
    args = validate_args(args)
    print(
        "Trainer: staged static pose + image-context articulation (train_yolo26_staged_cube_pose.py)",
        flush=True,
    )
    batch_overrides = batch_cli_overrides(args, argv)
    explicit_metrics = any(token.split("=", 1)[0] == "--train-metrics-every" for token in argv)
    resume = None
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.resume:
        model, resume = model_from_checkpoint(args.resume)
        original = resume["train_args"]
        # Device/workers/root can change when moving the run from Windows to
        # another machine; its split, data fingerprint and training plan cannot.
        for key, value in original.items():
            if (
                key not in ("device", "workers", "threads", "dataset", "output_dir", "resume")
                and key not in batch_overrides
                and not (key == "train_metrics_every" and explicit_metrics)
            ):
                setattr(
                    args,
                    key,
                    Path(value)
                    if key
                    in (
                        "init_checkpoint",
                        "static_checkpoint",
                        "articulation_checkpoint",
                        "initial_split",
                        "export_init",
                    )
                    and value
                    else value,
                )
        args.output_dir = args.resume.resolve().parent.parent
        validate_batch_args(args)
    else:
        model = initialize_model(args)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.export_init:
        if args.export_init.exists():
            raise ValueError(f"Preserving existing file: {args.export_init}")
        atomic_save(checkpoint_model(model), args.export_init)
        print(f"Saved portable staged initialization: {args.export_init.resolve()}", flush=True)
        return
    device = select_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable in this Python environment; install the CUDA-enabled PyTorch build"
        )
    model.to(device)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and resume is None:
        raise ValueError(f"Output exists; use --resume or a new --output-dir: {output}")
    if not (args.dataset / "manifest.jsonl").is_file():
        raise ValueError(
            "--dataset/--source-dir must point to the exported direct_cube_pose_dataset (manifest.jsonl + targets/ + pose/ + images/)"
        )
    entries, symmetries = read_manifest(args.dataset)
    fingerprint = hashlib.sha256(
        (args.dataset / "manifest.jsonl").read_bytes()
        + (args.dataset / "symmetries.json").read_bytes()
    ).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    if resume:
        if resume["dataset_fingerprint"] != fingerprint:
            raise ValueError("Dataset manifest/symmetries changed since checkpoint")
        split = resume["split"]
        lookup = {e["key"]: e for e in entries}
        train = [lookup[k] for k in split["train"]]
        val = [lookup[k] for k in split["val"]]
    else:
        train, val = split_for_initialization(
            entries, getattr(model, "initialization", {}), args.val_fraction, args.seed, args.limit
        )
        split = {"train": [e["key"] for e in train], "val": [e["key"] for e in val]}
    if set(split["train"]) & set(split["val"]):
        raise ValueError("Train/validation overlap")
    (output / "split.json").write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    print(
        f"PyTorch {torch.__version__}; Ultralytics {version('ultralytics')}; device={device}; images={len(entries)} train={len(train)} val={len(val)}; input={args.imgsz}px",
        flush=True,
    )
    datasets = [StagedPoseDataset(args.dataset, e, args.imgsz) for e in (train, val)]
    args.amp = resolve_training_amp(args.amp, model.detector, device)
    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(device)}; CUDA runtime: {torch.version.cuda}",
            flush=True,
        )
    print(
        f"Detector precision: {'AMP' if args.amp else 'FP32'}; structured heads/geometry: FP32",
        flush=True,
    )
    model.use_amp = args.amp
    print(f"Batch progress: {ConsoleProgress().backend}", flush=True)
    loader_rng = torch.Generator().manual_seed(args.seed)
    worker_rngs = [torch.Generator().manual_seed(args.seed + 100 + i) for i in range(2)]
    phase_batches = restore_phase_batches(resume, args, batch_overrides)
    observations_changed = (
        resume is not None and resume.get("observation_version") != OBSERVATION_VERSION
    )
    if observations_changed:
        print(
            "Point selector updated to complete CPU NMS; resetting the current-stage validation best. Model/optimizer/schedule progress is retained.",
            flush=True,
        )
    criterion = StagedPoseLoss(model, symmetries).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    train_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (output / "args.json").write_text(json.dumps(train_args, indent=2) + "\n", encoding="utf-8")
    if resume:
        torch.set_rng_state(resume["torch_rng"])
        np.random.set_state(resume["numpy_rng"])
        random.setstate(resume["python_rng"])
        loader_rng.set_state(resume["loader_rng"])
        for generator, state in zip(worker_rngs, resume.get("worker_rngs", [])):
            generator.set_state(state)
        if device.type == "cuda" and resume.get("cuda_rng"):
            torch.cuda.set_rng_state_all(resume["cuda_rng"])
    started = time.monotonic()
    for phase_index, phase in enumerate(PHASES):
        epochs = getattr(args, f"{phase}_epochs")
        if not epochs or resume and phase_index < resume["phase_index"]:
            continue
        model.zero_grad(set_to_none=True)
        model.configure_phase(phase)
        if phase not in phase_batches:
            phase_batches[phase], profile = resolve_phase_batch_size(
                stage_batch_request(args, phase),
                model,
                device,
                phase,
                datasets[0],
                symmetries,
                optimizer_for,
                memory_fraction=args.batch_memory_fraction,
                maximum=args.max_auto_batch,
            )
            if profile is not None:
                with (output / "batch_profiles.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(profile) + "\n")
        stage_batch = phase_batches[phase]
        loaders = stage_loaders(datasets, stage_batch, args, device, loader_rng, worker_rngs)
        optimizer = optimizer_for(model, phase)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-6)
        first = 1
        best = float("inf")
        if resume and phase_index == resume["phase_index"]:
            optimizer.load_state_dict(resume["optimizer"])
            scheduler.load_state_dict(resume["scheduler"])
            if scaler.is_enabled():
                scaler.load_state_dict(resume["scaler"])
            first = resume["phase_epoch"] + 1
            best = float("inf") if observations_changed else resume["best_score"]
        print(
            f"Phase {phase}: {epochs} epochs; batch={stage_batch}; trainable "
            + ", ".join(g["label"] for g in optimizer.param_groups),
            flush=True,
        )
        for epoch in range(first, epochs + 1):
            print(f"{phase} epoch {epoch}/{epochs}", flush=True)
            training = run_epoch(
                model,
                loaders[0],
                criterion,
                device,
                phase,
                optimizer,
                scaler,
                args.max_batches,
                metrics_every=args.train_metrics_every,
            )
            validation = run_epoch(
                model, loaders[1], criterion, device, phase, None, scaler, args.max_batches
            )
            scheduler.step()
            score = selection_score(validation, phase)
            improved = score < best
            best = min(score, best)
            record = {
                "phase": phase,
                "batch": stage_batch,
                "epoch": epoch,
                "train": training,
                "val": validation,
                "score": score,
                "seconds": time.monotonic() - started,
            }
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, allow_nan=False) + "\n")
            saved = dict(
                checkpoint_model(model),
                train_args=train_args,
                phase_index=phase_index,
                phase=phase,
                phase_epoch=epoch,
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(),
                best_score=best,
                dataset_fingerprint=fingerprint,
                symmetries=symmetries,
                split=split,
                phase_batch_sizes=dict(phase_batches),
                metrics=record,
                torch_rng=torch.get_rng_state(),
                numpy_rng=np.random.get_state(),
                python_rng=random.getstate(),
                loader_rng=loader_rng.get_state(),
                worker_rngs=[g.get_state() for g in worker_rngs],
                cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            )
            atomic_save(saved, output / "weights/last.pt")
            if improved:
                atomic_save(saved, output / f"weights/{phase}_best.pt")
            if phase == "joint" and improved:
                atomic_save(saved, output / "weights/best.pt")
            del saved  # Do not retain GPU optimizer states across stage changes.
            print(
                f"  validation: static={validation['static_rotation_deg']}deg moving={validation['moving_turn_deg']}deg observed={validation['observed_moving_turn_deg']}deg; best score={best:.4f}",
                flush=True,
            )
        # Hand off the best component from THIS phase, not its final epoch.
        best_path = output / f"weights/{phase}_best.pt"
        if best_path.exists():
            stage_best = torch.load(best_path, map_location="cpu", weights_only=False)
            if stage_best.get("observation_version") == OBSERVATION_VERSION:
                model.load_state_dict(stage_best["model_state"])
            else:
                print(
                    f"{phase}: historical best used the previous point selector; keeping the loaded weights for handoff.",
                    flush=True,
                )
            del stage_best
        del loaders, optimizer, scheduler
    final = output / "weights/best.pt"
    if not final.exists():
        final = output / "weights/last.pt"
    print(f"Training complete: {final}", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
