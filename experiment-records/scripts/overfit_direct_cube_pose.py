#!/usr/bin/env python3
"""Memorize a fixed, representative subset with the current symmetry-aware head.

    .venv/bin/python scripts/overfit_direct_cube_pose.py

This is a training-image memorization test, NOT held-out accuracy evaluation.
The pretrained independent detector is frozen. Its own predicted features,
keypoints and class scores are cached once; the unchanged custom head/loss train
on those inputs with augmentation and weight decay off. Final predictions are
checked again through the full image-to-pose path and a reloaded checkpoint.
Only the existing cube-frame symmetries are accepted; hidden-articulation
alternatives are intentionally outside this first experiment.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from live_recognition.direct_cube_pose import (  # noqa: E402
    CHECKPOINT_FORMAT, CLASS_NAMES, FACE_ORDER, DirectCubePoseModel, SymmetryPoseLoss,
    load_direct_pose_checkpoint, make_detector,
)
from live_recognition.direct_cube_pose_data import (  # noqa: E402
    DirectPoseDataset, collate_direct_pose, landmark_template, read_manifest,
)
from scripts import prepare_direct_cube_pose_targets as geometry  # noqa: E402
from scripts.train_yolo26_direct_cube_pose import select_device  # noqa: E402


def select_subset(entries, count, seed):
    if count < 5:
        raise ValueError("Use at least five samples to cover all articulation modes")
    rng = random.Random(seed)
    groups = defaultdict(list)
    for entry in entries:
        if entry["pose_supervision"]:
            groups[(entry["source_dataset"], entry["mode"])].append(entry)
    for values in groups.values():
        rng.shuffle(values)
    selected = []
    if count < 22:
        for mode in ("none", "mid_turn", "simultaneous_turn", "slice_turn", "corner_cut"):
            key = next((key for key in sorted(groups) if key[1] == mode), None)
            if key is not None:
                selected.append(groups[key][0])
    # Include the two previously visually checked 0-/1-center examples when
    # available; this does not declare all historical missing labels reliable.
    known = {"keypoints_only_dataset_part7__001978", "keypoints_only_dataset_part7__001192"}
    for entry in entries:
        if len(selected) < count and entry["key"] in known and entry["pose_supervision"] and entry not in selected:
            selected.append(entry)
    used = {entry["key"] for entry in selected}
    keys = sorted(groups, key=lambda key: (key[1], key[0]))
    while len(selected) < count and groups:
        for index, key in enumerate(keys):
            if key not in groups:
                continue
            values = [entry for entry in groups[key] if entry["key"] not in used]
            if not values:
                del groups[key]
                continue
            desired_symmetry = (1, 4, 24)[index % 3]
            chosen = next((entry for entry in values if entry["symmetry_count"] == desired_symmetry), values[0])
            selected.append(chosen)
            used.add(chosen["key"])
            if len(selected) == count:
                break
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} eligible samples available")
    return selected


def evaluate(head, encoded, target, criterion, image_size):
    head.eval()
    with torch.no_grad():
        prediction = head(*encoded)
        losses, metrics, selected = criterion(prediction, target, image_size)
    per_image = {name: value.cpu().tolist() for name, value in metrics.pop("per_image").items()}
    result = {name: float(value) for name, value in metrics.items()}
    result["losses"] = {name: float(value) for name, value in losses.items()}
    result["geometric_loss"] = sum(criterion.weights[name] * float(losses[name])
                                   for name in ("rotation", "projection", "angles", "centers", "geometry"))
    result["per_image"] = per_image
    result["selected_symmetries"] = selected.cpu().tolist()
    rows = torch.arange(len(selected), device=selected.device)
    matched_angles = target["angles"][:, criterion.center_inverse][rows, selected]
    angle_error = (prediction["angles"] - matched_angles).abs()
    result["max_face_angle_error_deg"] = float(angle_error.max())
    result["per_image"]["max_face_angle_error_deg"] = angle_error.amax(-1).cpu().tolist()
    return prediction, result


def target_for_render(prediction, index, image_size):
    def array(name):
        return prediction[name][index].detach().cpu().numpy()
    return {"image_size": [image_size, image_size], "rotation_matrix": array("rotation_matrix"),
            "projection": {"center_xy_normalized": array("center"), "principal_point_normalized": array("principal"),
                           "image_scale_normalized": float(array("scale")), "perspective_strength": float(array("perspective"))},
            "angles": array("angles")}


def draw_wire(image, target):
    canvas = cv2.resize(image, (320, 320), interpolation=cv2.INTER_LINEAR)
    quads, normals = geometry.surface_quads(target["angles"])
    projection = target["projection"]
    translation = np.r_[(np.asarray(projection["center_xy_normalized"]) - projection["principal_point_normalized"])
                         / projection["image_scale_normalized"], 3 / projection["perspective_strength"]]
    camera = -np.asarray(target["rotation_matrix"]).T @ translation
    scaled = {**target, "image_size": [320, 320]}
    for quad, normal in zip(quads, normals):
        if normal @ (camera - quad.mean(0)) <= 0:
            continue
        pixels = np.rint(geometry.project_target(quad, scaled)).astype(np.int32)
        cv2.polylines(canvas, [pixels], True, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.polylines(canvas, [pixels], True, (60, 240, 255), 1, cv2.LINE_AA)
    return canvas


def save_overlays(root, output, entries, initial, final, final_metrics, image_size):
    previews = []
    for index, entry in enumerate(entries):
        image = geometry.read_image(root / entry["image"])
        payload = json.loads((root / entry["target"]).read_text())
        target = {"image_size": [320, 320], "rotation_matrix": np.asarray(payload["rotation_matrix"]),
                  "projection": payload["projection"], "angles": payload["layer_angles_degrees"]}
        panels = [draw_wire(image, target), draw_wire(image, target_for_render(initial, index, image_size)),
                  draw_wire(image, target_for_render(final, index, image_size))]
        sheet = np.full((386, 960, 3), 24, np.uint8)
        for column, (panel, label) in enumerate(zip(panels, ("Target", "Before", "After"))):
            sheet[:320, column * 320:(column + 1) * 320] = panel
            geometry.text(sheet, label, (column * 320 + 8, 22), scale=.55)
        geometry.text(sheet, f"{entry['key']} | {entry['mode']} | {entry['symmetry_count']} symmetries", (8, 343), scale=.45)
        geometry.text(sheet, f"After: landmarks {final_metrics['per_image']['landmark_rmse_px'][index]:.3f}px; "
                      f"rotation {final_metrics['per_image']['rotation_deg'][index]:.3f}deg; "
                      f"turns {final_metrics['per_image']['angle_mae_deg'][index]:.3f}deg", (8, 369), scale=.45)
        geometry.save_image(output / "overlays" / f"{entry['key']}.jpg", sheet)
        previews.append(sheet)
    # One representative of each mode, plus the explicitly checked one-center
    # and zero-center examples, remains readable in the contact sheet.
    indexes, seen = [], set()
    for index, entry in enumerate(entries):
        if entry["mode"] not in seen:
            indexes.append(index)
            seen.add(entry["mode"])
    contact = np.concatenate([previews[index] for index in indexes], axis=0)
    geometry.save_image(output / "before_after.jpg", contact)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("data/direct_cube_pose_dataset"))
    parser.add_argument("--model", default="runs/pose/runs/26m-pose/weights/best.pt")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/direct_cube_pose_overfit"))
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--imgsz", type=int, default=160)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.steps < 1 or args.lr <= 0 or args.imgsz < 64 or args.imgsz % 32 or args.threads < 1:
        raise ValueError("Positive steps/lr/threads and image size >=64 divisible by 32 are required")
    output, root = args.output_dir.resolve(), args.dataset.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output must be empty: {output}")
    (output / "overlays").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = select_device(args.device)
    entries, manifest = read_manifest(root)
    selected = select_subset(entries, args.samples, args.seed)
    (output / "subset.json").write_text(json.dumps(selected, indent=2) + "\n")
    dataset = DirectPoseDataset(root, selected, args.imgsz, augment=False)
    independent, target, _ = collate_direct_pose([dataset[index] for index in range(len(dataset))])
    images = independent["img"].to(device)
    target = {key: value.to(device) for key, value in target.items()}
    model = DirectCubePoseModel(make_detector(args.model, args.steps)).to(device)
    model.freeze_detector(True)
    model.eval()
    with torch.no_grad():
        _, features, points, scores = model.encode_observations(images)
    encoded = ([feature.detach() for feature in features], points.detach(), scores.detach())
    criterion = SymmetryPoseLoss(manifest, *landmark_template()).to(device)
    optimizer = torch.optim.AdamW(model.pose_head.parameters(), lr=args.lr, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * .01)
    initial, initial_metrics = evaluate(model.pose_head, encoded, target, criterion, args.imgsz)
    best_loss, best_step = initial_metrics["geometric_loss"], 0
    best_state = copy.deepcopy(model.pose_head.state_dict())
    started = time.monotonic()
    print(f"Overfitting {len(selected)} fixed training images at {args.imgsz}px on {device}; frozen detector", flush=True)
    with (output / "metrics.jsonl").open("w") as log:
        log.write(json.dumps({"step": 0, **initial_metrics}) + "\n")
        for step in range(1, args.steps + 1):
            model.pose_head.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model.pose_head(*encoded)
            losses, _, _ = criterion(prediction, target, args.imgsz)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.pose_head.parameters(), 10., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            if step % 25 == 0 or step == args.steps:
                _, metrics = evaluate(model.pose_head, encoded, target, criterion, args.imgsz)
                log.write(json.dumps({"step": step, **metrics}, allow_nan=False) + "\n")
                log.flush()
                if metrics["geometric_loss"] < best_loss:
                    best_loss, best_step = metrics["geometric_loss"], step
                    best_state = copy.deepcopy(model.pose_head.state_dict())
                print(f"step {step}: loss={metrics['geometric_loss']:.5f} centers={metrics['center_rmse_px']:.3f}px "
                      f"landmarks={metrics['landmark_rmse_px']:.3f}px rotation={metrics['rotation_deg']:.3f}deg "
                      f"turns={metrics['angle_mae_deg']:.3f}deg ({time.monotonic()-started:.1f}s)", flush=True)
    model.pose_head.load_state_dict(best_state)
    final, final_metrics = evaluate(model.pose_head, encoded, target, criterion, args.imgsz)
    model.eval()
    with torch.no_grad():
        replay = model(images)["pose"]
    replay_max_error = max(float((replay[key] - final[key]).abs().max()) for key in final)
    checkpoint = {"format": CHECKPOINT_FORMAT, "diagnostic_only": True,
                  "detector_yaml": copy.deepcopy(model.detector.yaml), "head_config": model.head_config,
                  "detector_end2end": model.detector.end2end,
                  "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                  "train_args": {"epochs": args.steps, "imgsz": args.imgsz}, "class_names": list(CLASS_NAMES),
                  "face_order": list(FACE_ORDER), "symmetries": manifest, "overfit_step": best_step}
    torch.save(checkpoint, output / "overfit.pt")
    reloaded, _ = load_direct_pose_checkpoint(output / "overfit.pt", device)
    with torch.no_grad():
        reloaded_prediction = reloaded(images)["pose"]
    reload_max_error = max(float((reloaded_prediction[key] - replay[key]).abs().max()) for key in replay)
    limits = {"center_rmse_px": 1., "landmark_rmse_px": 1.5, "rotation_deg": 3., "angle_mae_deg": 1.5,
              "max_face_angle_error_deg": 1.}
    per_image_maxima = {name: max(final_metrics["per_image"][name]) for name in limits}
    passed = all(per_image_maxima[name] <= limit for name, limit in limits.items()) and replay_max_error < 1e-4 and reload_max_error < 1e-4
    report = {"passed": passed, "purpose": "memorization of the SAME training images, not held-out accuracy",
              "detector_frozen": True, "augmentation": False, "weight_decay": 0., "device": str(device),
              "images": len(selected), "modes": dict(Counter(entry["mode"] for entry in selected)),
              "sources": dict(Counter(entry["source_dataset"] for entry in selected)),
              "symmetry_counts": dict(Counter(entry["symmetry_count"] for entry in selected)),
              "steps": args.steps, "best_step": best_step, "seconds": time.monotonic() - started,
              "initial": initial_metrics, "final": final_metrics, "per_image_maxima": per_image_maxima,
              "per_image_limits": limits, "full_pipeline_replay_max_error": replay_max_error,
              "checkpoint_reload_max_error": reload_max_error, "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save_overlays(root, output, selected, initial, final, final_metrics, args.imgsz)
    print(json.dumps({key: report[key] for key in ("passed", "best_step", "per_image_maxima", "full_pipeline_replay_max_error", "checkpoint_reload_max_error")}, indent=2), flush=True)
    if not passed:
        raise SystemExit("Overfit thresholds were not reached; inspect report.json and overlays")


if __name__ == "__main__":
    main()
