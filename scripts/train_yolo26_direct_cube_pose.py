#!/usr/bin/env python3
"""Train the independent YOLO detector and its structured pose/articulation head.

    .venv/bin/python scripts/train_yolo26_direct_cube_pose.py \
        --dataset data/direct_cube_pose_dataset \
        --model runs/pose/runs/26m-pose/weights/best.pt \
        --device 0 --batch 16 --epochs 100

The original nine classes use the installed default YOLO pose loss. The new
branch predicts rotation, exact projection parameters, six face offsets,
auxiliary ordered centers and a projection-error-derived pose quality score.
It sees predicted landmark/sticker points and feature maps, never GT points.
One allowed symmetry is selected jointly across all direct-pose losses.

The first five epochs train the new branch with the detector frozen; later
epochs fine-tune both with a smaller detector learning rate. Validation uses a
saved, stratified image split. All articulation modes are retained; corner cuts
follow the dataset's dominant-angle approximation. Spatial augmentation is
disabled; brightness/contrast/blur augmentation preserves label geometry.

--limit 32 --epochs 1 --freeze-epochs 0 --model yolo26n-pose.yaml --imgsz 64
--batch 2 --workers 0 --device cpu is a small, randomly initialized smoke run.
Checkpoints are self-contained and load with load_direct_pose_checkpoint().
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from live_recognition.direct_cube_pose import (  # noqa: E402
    CHECKPOINT_FORMAT, CLASS_NAMES, FACE_ORDER, DirectCubePoseModel, SymmetryPoseLoss,
    independent_pose_criterion, load_direct_pose_checkpoint, make_detector,
)
from live_recognition.direct_cube_pose_data import (  # noqa: E402
    DirectPoseDataset, collate_direct_pose, landmark_template, read_manifest, split_entries,
)


def seed_worker(_worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)


def select_device(value):
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if value.isdigit():
        value = f"cuda:{value}"
    return torch.device(value)


def run_epoch(model, loader, independent_loss, direct_loss, device, optimizer=None, scaler=None,
              independent_weight=1., max_batches=None):
    training = optimizer is not None
    model.train(training)
    totals, denominators = {}, {}
    entries = {entry["key"]: entry for entry in loader.dataset.entries} if not training else {}
    for batch_index, (batch, target, keys) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = {key: value.to(device, non_blocking=device.type == "cuda") for key, value in batch.items()}
        target = {key: value.to(device, non_blocking=device.type == "cuda") for key, value in target.items()}
        size = len(keys)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=scaler is not None and scaler.is_enabled()):
                prediction = model(batch["img"])
                # Ultralytics returns component losses scaled by batch size.
                # Preserve its criterion and gains, then average per image.
                independent, _ = independent_loss(prediction["independent"], batch)
                independent = independent.sum() / size
            # Keep the geometry and symmetry selection in FP32 under CUDA AMP.
            pose = {key: value.float() for key, value in prediction["pose"].items()}
            losses, metrics, _ = direct_loss(pose, target, batch["img"].shape[-1])
            per_image = metrics.pop("per_image")
            total = losses["total"] + independent_weight * independent
            if not torch.isfinite(total):
                raise FloatingPointError(f"Nonfinite loss for {keys}")
            if training:
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
        values = {"loss": total.detach(), "independent_loss": independent.detach(),
                  **{f"direct_{name}": value.detach() for name, value in losses.items()}, **metrics}
        valid_count = int(target["enabled"].sum())
        for name, value in values.items():
            # Direct geometric losses/metrics average over supervised poses;
            # independent and quality losses also include empty-label images.
            weight = valid_count if (name.startswith("direct_") and name not in ("direct_quality", "direct_total")
                                     or name in ("rotation_deg", "center_rmse_px", "landmark_rmse_px", "angle_mae_deg")) else size
            if name == "observed_landmark_rmse_px":
                weight = int((target["enabled"] & target["observed_mask"].any(-1)).sum())
            totals[name] = totals.get(name, 0.) + float(value) * weight
            denominators[name] = denominators.get(name, 0) + weight
        if not training:
            # These are existing label-based ambiguity groups, not a claim that
            # missing historical labels establish actual image visibility.
            for index, key in enumerate(keys):
                if not bool(target["enabled"][index]):
                    continue
                entry = entries[key]
                for group in (f"by_mode/{entry['mode']}", f"by_allowed_symmetries/{entry['symmetry_count']}"):
                    for name, value in per_image.items():
                        if name == "observed_landmark_rmse_px" and not bool(target["observed_mask"][index].any()):
                            continue
                        field = f"{group}/{name}"
                        totals[field] = totals.get(field, 0.) + float(value[index])
                        denominators[field] = denominators.get(field, 0) + 1
        if training and (batch_index + 1) % 50 == 0:
            print(f"  batch {batch_index + 1}/{len(loader)} loss={float(total.detach()):.4f}", flush=True)
    if not totals:
        raise ValueError("No batches were processed")
    return {name: value / max(1, denominators[name]) for name, value in totals.items()}


def save_checkpoint(path, model, optimizer, scheduler, scaler, args, epoch, best, bad_epochs, manifest, generator):
    checkpoint = {"format": CHECKPOINT_FORMAT, "epoch": epoch, "best_val_loss": best,
                  "bad_epochs": bad_epochs, "detector_yaml": copy.deepcopy(model.detector.yaml),
                  "detector_end2end": model.detector.end2end,
                  "head_config": model.head_config, "class_names": list(CLASS_NAMES), "face_order": list(FACE_ORDER),
                  "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                  "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                  "scaler_state": scaler.state_dict(), "train_args": vars(args), "symmetries": manifest,
                  "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                  "python_rng": random.getstate(), "loader_rng": generator.get_state(),
                  "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                  "quality_definition": "exp(-0.5*(max(joint-selected articulated target landmark RMSE, saved observed landmark RMSE) / quality_sigma)^2); zero where pose supervision is masked; not a calibrated probability"}
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("data/direct_cube_pose_dataset"))
    parser.add_argument("--model", default="runs/pose/runs/26m-pose/weights/best.pt")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=160)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--val-fraction", type=float, default=.1)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--detector-lr-factor", type=float, default=.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, help="Use a representative subset for smoke/overfit runs")
    parser.add_argument("--max-batches", type=int, help="Bound train AND validation batches for smoke tests")
    parser.add_argument("--independent-weight", type=float, default=1.)
    for name, default in (("rotation", 1.), ("projection", 1.), ("angles", 1.), ("centers", 10.), ("geometry", 10.), ("quality", .25)):
        parser.add_argument(f"--{name}-weight", type=float, default=default)
    parser.add_argument("--quality-sigma", type=float, default=.02, help="Quality-target RMSE scale in normalized image coordinates")
    return parser


def main():
    args = build_parser().parse_args()
    if min(args.epochs, args.batch, args.threads, args.topk) < 1 or args.imgsz < 64 or args.imgsz % 32:
        raise ValueError("Epochs/batch/threads/topk must be positive; imgsz must be a multiple of 32 and at least 64")
    if args.workers < 0 or args.freeze_epochs < 0 or args.width < 8 or args.width % 8:
        raise ValueError("Workers/freeze epochs must be nonnegative; head width must be a positive multiple of 8")
    if args.lr <= 0 or args.detector_lr_factor <= 0 or args.quality_sigma <= 0:
        raise ValueError("Learning rates and quality sigma must be positive")
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError("max-batches must be positive")
    weights = {name: getattr(args, f"{name}_weight") for name in ("rotation", "projection", "angles", "centers", "geometry", "quality")}
    if min(*weights.values(), args.independent_weight) < 0:
        raise ValueError("Loss weights cannot be negative")
    if not any(weights[name] for name in ("rotation", "angles", "centers", "geometry")):
        raise ValueError("At least one symmetry-sensitive geometric loss must be enabled")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = select_device(args.device)
    entries, manifest = read_manifest(args.dataset)
    args.dataset = str(args.dataset.resolve())
    checkpoint = None
    if args.resume:
        model, checkpoint = load_direct_pose_checkpoint(args.resume, device)
        if checkpoint.get("diagnostic_only"):
            raise ValueError("This checkpoint is an overfit diagnostic; start a normal run with --model instead of --resume")
        previous = checkpoint["train_args"]
        args.model = previous["model"]
        # These settings define the model/data/objective. Resume is deliberately
        # strict; changing them silently would not continue the same experiment.
        immutable = ("dataset", "imgsz", "batch", "seed", "val_fraction", "freeze_epochs", "augment", "limit",
                     "width", "topk", "lr", "detector_lr_factor", "weight_decay", "independent_weight",
                     "quality_sigma", *(f"{name}_weight" for name in weights))
        for name in immutable:
            if getattr(args, name) != previous[name]:
                raise ValueError(f"Resume requires the original --{name.replace('_', '-')}={previous[name]!r}")
        model.detector.args.epochs = args.epochs
    else:
        model = DirectCubePoseModel(make_detector(args.model, args.epochs), args.width, args.topk).to(device)
    output = (args.output_dir or (args.resume.resolve().parent.parent if args.resume else Path("runs/direct_cube_pose"))).resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError(f"Output already exists; use --resume or a new --output-dir: {output}")
    (output / "weights").mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output)
    args.resume = str(args.resume.resolve()) if args.resume else None
    if checkpoint:
        split = json.loads((output / "split.json").read_text())
        by_key = {entry["key"]: entry for entry in entries}
        train_entries = [by_key[key] for key in split["train"]]
        val_entries = [by_key[key] for key in split["val"]]
        if set(split["train"]) & set(split["val"]):
            raise ValueError("Saved train/validation split overlaps")
        if checkpoint["symmetries"] != manifest:
            raise ValueError("Dataset symmetry manifest changed since checkpoint")
    else:
        train_entries, val_entries = split_entries(entries, args.val_fraction, args.seed, args.limit)
        (output / "split.json").write_text(json.dumps({"dataset": args.dataset, "seed": args.seed,
                                                      "train": [e["key"] for e in train_entries],
                                                      "val": [e["key"] for e in val_entries]}, indent=2) + "\n")
    print(f"Device: {device}; train={len(train_entries)} val={len(val_entries)}; caching labels", flush=True)
    datasets = [DirectPoseDataset(args.dataset, train_entries, args.imgsz, args.augment),
                DirectPoseDataset(args.dataset, val_entries, args.imgsz)]
    generator = torch.Generator().manual_seed(args.seed)
    loaders = [DataLoader(dataset, batch_size=args.batch, shuffle=index == 0, num_workers=args.workers,
                          collate_fn=collate_direct_pose, worker_init_fn=seed_worker, generator=generator if index == 0 else None,
                          pin_memory=device.type == "cuda") for index, dataset in enumerate(datasets)]
    independent_loss = independent_pose_criterion(model.detector)
    rest, owners = landmark_template()
    direct_loss = SymmetryPoseLoss(manifest, rest, owners, **weights, quality_sigma=args.quality_sigma).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.pose_head.parameters(), "lr": args.lr},
        {"params": model.detector.parameters(), "lr": args.lr * args.detector_lr_factor},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * .001)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    start_epoch, best, bad_epochs = 0, float("inf"), 0
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scheduler.T_max = args.epochs
        if scaler.is_enabled() and checkpoint["scaler_state"]:
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch, best = checkpoint["epoch"], checkpoint["best_val_loss"]
        # Recompute the current cosine point when extending a completed run;
        # otherwise an optimizer already at eta_min would remain there.
        for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
            group["lr"] = scheduler.eta_min + (base_lr - scheduler.eta_min) * (1 + math.cos(math.pi * start_epoch / args.epochs)) / 2
        bad_epochs = checkpoint["bad_epochs"]
        torch.set_rng_state(checkpoint["torch_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        generator.set_state(checkpoint["loader_rng"])
        if checkpoint["cuda_rng"] is not None and device.type == "cuda":
            for index, state in enumerate(checkpoint["cuda_rng"][:torch.cuda.device_count()]):
                torch.cuda.set_rng_state(state, index)
        for _ in range(start_epoch):
            if hasattr(independent_loss, "update"):
                independent_loss.update()
    if start_epoch >= args.epochs:
        raise ValueError("Checkpoint already reached --epochs; increase it to continue")
    (output / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    print(f"Training from epoch {start_epoch + 1}; checkpoints: {output / 'weights'}", flush=True)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        started = time.monotonic()
        frozen = epoch <= args.freeze_epochs
        model.freeze_detector(frozen)
        train = run_epoch(model, loaders[0], independent_loss, direct_loss, device, optimizer, scaler,
                          args.independent_weight, args.max_batches)
        with torch.no_grad():
            val = run_epoch(model, loaders[1], independent_loss, direct_loss, device,
                            independent_weight=args.independent_weight, max_batches=args.max_batches)
        scheduler.step()
        improved = val["loss"] < best
        best = min(best, val["loss"])
        bad_epochs = 0 if improved else bad_epochs + 1
        record = {"epoch": epoch, "detector_frozen": frozen, "seconds": time.monotonic() - started,
                  "train": train, "val": val, "learning_rates": [group["lr"] for group in optimizer.param_groups]}
        with (output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        save_checkpoint(output / "weights" / "last.pt", model, optimizer, scheduler, scaler,
                        args, epoch, best, bad_epochs, manifest, generator)
        if improved:
            save_checkpoint(output / "weights" / "best.pt", model, optimizer, scheduler, scaler,
                            args, epoch, best, bad_epochs, manifest, generator)
        print(f"Epoch {epoch}/{args.epochs} {'head-only' if frozen else 'joint'} "
              f"train={train['loss']:.4f} val={val['loss']:.4f} "
              f"centers={val['center_rmse_px']:.2f}px landmarks={val['landmark_rmse_px']:.2f}px "
              f"observed={val['observed_landmark_rmse_px']:.2f}px "
              f"rotation={val['rotation_deg']:.2f}deg turns={val['angle_mae_deg']:.2f}deg "
              f"({record['seconds']:.1f}s)", flush=True)
        if hasattr(independent_loss, "update"):
            independent_loss.update()
        if args.patience > 0 and bad_epochs >= args.patience:
            print(f"Stopped after {bad_epochs} epochs without validation improvement", flush=True)
            break
    print(f"Best checkpoint: {output / 'weights' / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
