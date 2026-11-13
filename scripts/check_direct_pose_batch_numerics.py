#!/usr/bin/env python3
"""Compare one reported batch on CUDA, without training or changing weights.

    .venv/bin/python scripts/check_direct_pose_batch_numerics.py \
        --run-dir runs/direct_cube_pose --device 0

Run with the same --model, --dataset, --imgsz, --workers, --seed and augmentation
options as training. Three cases compare the original AMP arithmetic, FP32 loss
arithmetic on the SAME AMP outputs, and an entirely FP32 forward/loss. Hooks
inspect assignment, box/keypoint/RLE terms only in this standalone diagnostic.
Nothing here runs during normal training. CPU AMP is explicitly an emulation;
only a CUDA run diagnoses CUDA behavior. No optimizer steps are performed.
"""

from __future__ import annotations

from importlib.metadata import version
import hashlib
import json
from pathlib import Path
import random
import sys
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from live_recognition.direct_cube_pose import (  # noqa: E402
    DirectCubePoseModel, SymmetryPoseLoss, independent_pose_criterion,
    load_direct_pose_checkpoint, make_detector,
)
from live_recognition.direct_cube_pose_data import DirectPoseDataset, collate_direct_pose, landmark_template  # noqa: E402
from live_recognition.direct_pose_training_numerics import (  # noqa: E402
    attempt_diagnostics, batchnorm_snapshot, floating_to_float32, nonfinite_fields, restore_batchnorm,
)
from scripts.train_yolo26_direct_cube_pose import build_parser, normalize_training_arguments, select_device  # noqa: E402

REPORTED_KEYS = ("keypoints_only_dataset_part3__003592", "keypoints_only_dataset__002263",
                 "keypoints_only_dataset__005163", "keypoints_only_dataset_part3__002030")


def inspect_tree(value):
    tree = value if isinstance(value, dict) else {"value": value}
    return attempt_diagnostics({"outputs": tree, "nonfinite": nonfinite_fields(tree)})


def stage_hooks(criterion, direct, stages):
    handles = []
    losses = (("flat", criterion.flat), ("one2many", criterion.end_to_end.one2many),
              ("one2one", criterion.end_to_end.one2one))
    for prefix, loss in losses:
        for field in ("assigner", "bbox_loss", "keypoint_loss", "rle_loss"):
            module = getattr(loss, field, None)
            if not isinstance(module, torch.nn.Module):
                continue
            name = f"{prefix}.{field}"

            def before(_module, inputs, name=name):
                stages[f"{name}.inputs"] = inspect_tree(inputs)

            def after(_module, _inputs, output, name=name):
                stages[f"{name}.outputs"] = inspect_tree(output)

            handles.extend((module.register_forward_pre_hook(before), module.register_forward_hook(after)))

    def pose_after(_module, _inputs, output):
        stages["custom_pose.outputs"] = inspect_tree(output)

    handles.append(direct.register_forward_hook(pose_after))
    return handles


def evaluate_case(prediction, batch, target, criterion, direct, device, *, amp_loss, fp32_inputs, independent_weight=1.):
    raw = criterion.raw_predictions(prediction["independent"])
    if fp32_inputs:
        raw = floating_to_float32(raw)
    pose = floating_to_float32(prediction["pose"])
    attempt = {"outputs": {"independent": raw, "pose": pose}, "nonfinite": []}
    stages = {}
    handles = stage_hooks(criterion, direct, stages)
    try:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_loss):
            components, _ = criterion(raw, batch)
            independent = components.sum() / len(batch["img"])
        with torch.autocast(device_type=device.type, enabled=False):
            losses, _, _ = direct(pose, target, batch["img"].shape[-1])
            total = independent_weight * independent + losses["total"]
        attempt.update(independent_components=components, pose_losses=losses,
                       nonfinite=nonfinite_fields({"yolo": components, "pose": losses, "total": total}))
        result = {"passed": not attempt["nonfinite"], **attempt_diagnostics(attempt)}
    except Exception as error:
        attempt["nonfinite"] = nonfinite_fields(attempt["outputs"])
        result = {"passed": False, "exception": f"{type(error).__name__}: {error}",
                  "traceback": traceback.format_exc(), **attempt_diagnostics(attempt)}
    finally:
        for handle in handles:
            handle.remove()
    result["stages"] = stages
    return result


def classification(cases):
    original, loss32, all32 = (cases[name]["passed"] for name in ("original_amp", "amp_forward_fp32_losses", "all_fp32"))
    if original and loss32 and all32:
        return "Not reproduced with these weights, inputs and environment."
    if not original and loss32:
        return "Failure isolated to AMP loss arithmetic on identical network predictions."
    if not loss32 and all32:
        return "Failure depends on the mixed-precision forward outputs, even with FP32 losses."
    if not all32:
        return "Failure persists in CUDA/CPU float32; inspect the failing component and stage."
    return "Precision-sensitive difference; inspect the individual component results."


def main():
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument("--keys", nargs="+", default=list(REPORTED_KEYS))
    parser.add_argument("--run-dir", type=Path, help="Read the failing run's exact settings from args.json")
    args = parser.parse_args()
    if args.run_dir:
        saved = json.loads((args.run_dir / "args.json").read_text())
        fields = ("dataset", "model", "resume", "imgsz", "seed", "workers", "amp", "augment", "freeze_epochs",
                  "width", "topk", "epochs", "quality_sigma", "independent_weight",
                  *(f"{name}_weight" for name in ("rotation", "projection", "angles", "centers", "geometry", "quality")))
        for name in fields:
            if name in saved:
                value = saved[name]
                setattr(args, name, Path(value) if name in ("dataset", "resume") and value is not None else value)
        if not any(item == "--device" or item.startswith("--device=") for item in sys.argv[1:]):
            args.device = saved.get("device", args.device)
    args = normalize_training_arguments(args)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = select_device(args.device)
    root = args.dataset.resolve()
    by_key = {entry["key"]: entry for line in (root / "manifest.jsonl").read_text().splitlines()
              if line.strip() and (entry := json.loads(line))["key"] in args.keys}
    missing = set(args.keys) - set(by_key)
    if missing:
        raise ValueError(f"Missing diagnostic sample keys: {sorted(missing)}")
    if args.resume:
        model, checkpoint = load_direct_pose_checkpoint(args.resume, device)
        epoch = checkpoint["epoch"] + 1
    else:
        model = DirectCubePoseModel(make_detector(args.model, args.epochs), args.width, args.topk).to(device)
        epoch = 1
    model.freeze_detector(epoch <= args.freeze_epochs)
    model.train()
    dataset = DirectPoseDataset(root, [by_key[key] for key in args.keys], args.imgsz, args.augment)
    # The reported samples are the first batch from seed 17. Reproduce worker
    # zero's photometric RNG without spawning workers or reading all labels.
    if args.workers:
        generator = torch.Generator().manual_seed(args.seed)
        seed = int(torch.empty((), dtype=torch.int64).random_(generator=generator))
        np.random.seed(seed % (2 ** 32))
    batch, target, keys = collate_direct_pose([dataset[index] for index in range(len(dataset))])
    batch = {key: value.to(device) for key, value in batch.items()}
    target = {key: value.to(device) for key, value in target.items()}
    criterion = independent_pose_criterion(model.detector)
    manifest = json.loads((root / "symmetries.json").read_text())
    weights = {name: getattr(args, f"{name}_weight") for name in ("rotation", "projection", "angles", "centers", "geometry", "quality")}
    direct = SymmetryPoseLoss(manifest, *landmark_template(), **weights, quality_sigma=args.quality_sigma).to(device)
    output = (args.output_dir or (args.run_dir / "numeric_probe" if args.run_dir else Path("runs/direct_pose_numeric_probe"))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"torch": torch.__version__, "ultralytics": version("ultralytics"), "device": str(device),
              "cuda_runtime": torch.version.cuda, "cpu_float16_emulation": device.type == "cpu",
              "training_amp_requested": args.amp,
              "run_config": str((args.run_dir / "args.json").resolve()) if args.run_dir else None,
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "cudnn_version": torch.backends.cudnn.version(),
              "model_sha256": hashlib.sha256(Path(args.resume or args.model).read_bytes()).hexdigest()
              if Path(args.resume or args.model).is_file() else None,
              "input_sha256": [hashlib.sha256(image.detach().cpu().numpy().tobytes()).hexdigest() for image in batch["img"]],
              "image_size": args.imgsz, "keys": list(keys), "model": args.model,
              "head_config": model.head_config, "loss_weights": weights,
              "detector_frozen": model.base_frozen, "detector_end2end": model.detector.end2end,
              "optimizer_steps": 0, "cases": {}}
    snapshot = batchnorm_snapshot(model)
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            amp_prediction = model(batch["img"])
        for name, loss_amp, fp32_inputs in (("original_amp", True, False), ("amp_forward_fp32_losses", False, True)):
            report["cases"][name] = evaluate_case(amp_prediction, batch, target, criterion, direct, device,
                                                   amp_loss=loss_amp, fp32_inputs=fp32_inputs,
                                                   independent_weight=args.independent_weight)
            print(name, "PASS" if report["cases"][name]["passed"] else "FAIL",
                  report["cases"][name].get("yolo_components"), flush=True)
            (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        del amp_prediction
        restore_batchnorm(snapshot)
        with torch.autocast(device_type=device.type, enabled=False):
            fp32_prediction = model(batch["img"])
        report["cases"]["all_fp32"] = evaluate_case(fp32_prediction, batch, target, criterion, direct, device,
                                                       amp_loss=False, fp32_inputs=True,
                                                       independent_weight=args.independent_weight)
    print("all_fp32", "PASS" if report["cases"]["all_fp32"]["passed"] else "FAIL",
          report["cases"]["all_fp32"].get("yolo_components"), flush=True)
    report["interpretation"] = classification(report["cases"])
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(report["interpretation"], flush=True)
    print(f"Diagnostic report: {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
