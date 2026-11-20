"""Train the compact cube-pose head on generated images.

The base model is an Ultralytics segmentation model.  Its multiscale feature
pyramid is tapped immediately before the segmentation head and passed to the
 custom pose decoder.  The pose target is the compact ``cube_pose_v7`` schema:

* direct canonical ``u``, ``v``, and ``twist`` angles,
* normalized image center and cube scale,
* perspective strength, and
* two signed outer-layer angles in ``[-45, 45)`` degrees.

Training combines direct canonical-head terms with an optional
closest-projected-line geometry loss. The default schedule uses 100% direct
canonical-head loss for every epoch so the bounded pose coordinates can
converge without the geometry term competing with them. The geometry weights
remain configurable from the command line for later refinement experiments.

By default the head consumes the YOLO feature pyramid plus two differentiable
prototype-derived union maps: cube and sticker. The fixed-width postprocessed
default-head sticker masks can be enabled explicitly for comparison, or all
mask inputs can be disabled for a strict feature-only run. ``--raw-image-only``
selects a separate compact CNN baseline that receives only the resized RGB
image. Ground-truth masks are never fed to the pose head because that would
create a train/inference mismatch.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from live_recognition.cube_pose import (  # noqa: E402
    ANGLE_PERIOD_DEGREES,
    PROTOTYPE_MASK_CHANNELS,
    POSE_MASK_CHANNELS,
    RawImageCubePoseModel,
    YOLOCubePoseModel,
    cube_pose_loss,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
@dataclass(frozen=True)
class PoseSample:
    image_path: Path
    pose_path: Path


def collect_samples(source_dir: Path) -> list[PoseSample]:
    images_dir = source_dir / "images"
    pose_dir = source_dir / "pose"
    if not images_dir.is_dir():
        raise RuntimeError(f"Missing image directory: {images_dir}")
    samples = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        pose_path = pose_dir / f"{image_path.stem}.json"
        if pose_path.is_file():
            samples.append(PoseSample(image_path, pose_path))
    if not samples:
        raise RuntimeError(f"No image/pose pairs found below {source_dir}")
    return samples


def _as_float_tensor(values: Any, *, shape: tuple[int, ...], name: str) -> Tensor:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return torch.from_numpy(array.copy())


def pose_target_from_json(payload: dict[str, Any]) -> dict[str, Tensor]:
    if payload.get("format") != "cube_pose_v7":
        raise ValueError("Training requires cube_pose_v7 labels")
    projection = payload.get("image_projection")
    articulation = payload.get("articulation")
    if not isinstance(projection, dict):
        raise ValueError("Pose label is missing image_projection")
    if not isinstance(articulation, dict):
        raise ValueError("Pose label is missing articulation")
    cube_pose = payload.get("cube_pose")
    if not isinstance(cube_pose, dict):
        raise ValueError("v7 pose label is missing cube_pose")
    u_value = float(cube_pose.get("u_colatitude_degrees"))
    v_value = float(cube_pose.get("v_azimuth_degrees"))
    twist_value = float(cube_pose.get("twist_degrees"))
    if not 0.0 <= u_value < 90.0:
        raise ValueError("v7 colatitude must be in [0, 90)")
    if not -180.0 <= v_value < 180.0:
        raise ValueError("v7 azimuth must be in [-180, 180)")
    if not -45.0 <= twist_value < 45.0:
        raise ValueError("v7 twist must be in [-45, 45)")
    layer_angles = articulation.get("layer_angles_degrees")
    if not isinstance(layer_angles, list) or len(layer_angles) != 2:
        raise ValueError("v7 articulation must contain two layer angles")
    layer_angles = [
        ((float(angle) + ANGLE_PERIOD_DEGREES / 2.0) % ANGLE_PERIOD_DEGREES)
        - ANGLE_PERIOD_DEGREES / 2.0
        for angle in layer_angles
    ]

    perspective = float(projection.get("perspective_strength", 0.0))
    if perspective <= 0.0:
        raise ValueError("image_projection.perspective_strength must be positive")
    return {
        "u": torch.tensor(u_value, dtype=torch.float32),
        "v": torch.tensor(v_value, dtype=torch.float32),
        "twist": torch.tensor(twist_value, dtype=torch.float32),
        "image_center": _as_float_tensor(
            projection.get("cube_center_normalized"),
            shape=(2,),
            name="cube_center_normalized",
        ),
        "image_scale": torch.tensor(
            float(projection.get("image_scale_normalized")),
            dtype=torch.float32,
        ),
        "perspective_strength": torch.tensor(perspective, dtype=torch.float32),
        "layer_angles": _as_float_tensor(
            layer_angles,
            shape=(2,),
            name="layer_angles_degrees",
        ),
    }


class CubePoseDataset(Dataset[tuple[Tensor, dict[str, Tensor], str]]):
    def __init__(self, samples: list[PoseSample], image_size: int) -> None:
        self.samples = samples
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor], str]:
        sample = self.samples[index]
        image_bgr = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Could not read {sample.image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(
            image_rgb,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )
        image = torch.from_numpy(image_rgb.copy()).permute(2, 0, 1).float() / 255.0
        payload = json.loads(sample.pose_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Pose label must be a JSON object: {sample.pose_path}")
        return image, pose_target_from_json(payload), sample.image_path.stem


def move_target(target: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    non_blocking = device.type == "cuda"
    return {
        key: value.to(device=device, non_blocking=non_blocking)
        for key, value in target.items()
    }


def _require_finite(
    values: dict[str, Tensor],
    *,
    label: str,
    stems: tuple[str, ...] | list[str],
    device: torch.device | None = None,
) -> None:
    """Fail at the first non-finite training value with useful context."""
    bad = [
        name
        for name, value in values.items()
        if not bool(torch.isfinite(value).all())
    ]
    if bad:
        raise FloatingPointError(
            f"Non-finite {label} in batch starting at {stems[0]!r}; "
            f"bad fields: {', '.join(bad)}; batch stems: {', '.join(stems)}"
            + (f"; device: {device}" if device is not None else "")
        )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    use_perspective: bool,
    geometry_weight: float,
    head_weight: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    sample_count = 0
    for images, targets, _stems in loader:
        images = images.to(device=device, non_blocking=device.type == "cuda")
        targets = move_target(targets, device)
        _require_finite(targets, label="target", stems=_stems, device=device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        prediction = model(images)
        _require_finite(prediction, label="model output", stems=_stems, device=device)
        losses = cube_pose_loss(
            prediction,
            targets,
            use_perspective=use_perspective,
            geometry_weight=geometry_weight,
            head_weight=head_weight,
        )
        _require_finite(losses, label="loss", stems=_stems, device=device)
        if training:
            losses["total"].backward()
            gradient_values = {
                name: parameter.grad
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and parameter.grad is not None
            }
            _require_finite(gradient_values, label="gradient", stems=_stems, device=device)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=10.0,
                error_if_nonfinite=True,
            )
            optimizer.step()
            _require_finite(
                {
                    name: parameter
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                },
                label="parameter",
                stems=_stems,
                device=device,
            )
        batch_size = int(images.shape[0])
        sample_count += batch_size
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach()) * batch_size
    if sample_count == 0:
        raise RuntimeError("Cannot run an epoch on an empty loader")
    return {key: value / sample_count for key, value in totals.items()}


def scheduled_loss_weights(
    epoch: int,
    epochs: int,
    *,
    head_start: float,
    head_end: float,
    geometry_start: float,
    geometry_end: float,
) -> tuple[float, float]:
    """Linearly interpolate direct-head and geometry loss weights."""
    if epochs <= 1:
        fraction = 0.0
    else:
        fraction = min(max((epoch - 1) / float(epochs - 1), 0.0), 1.0)
    head = head_start + fraction * (head_end - head_start)
    geometry = geometry_start + fraction * (geometry_end - geometry_start)
    return head, geometry


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    base_model: str | None,
    model_type: str,
    image_size: int,
    epoch: int,
    metrics: dict[str, float],
    use_soft_masks: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "cube_pose_head_v1",
            "model_type": model_type,
            "base_model": base_model,
            "imgsz": image_size,
            "epoch": epoch,
            "metrics": metrics,
            "model": model.state_dict(),
            "feature_channels": model.pose_head.feature_channels,
            "mask_channels": getattr(model, "mask_channels", 0),
            "use_soft_masks": use_soft_masks,
        },
        path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data" / "mcp_dataset_v5")
    parser.add_argument("--model", type=str, default="yolo11n-seg.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "cube_pose_head")
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--unfreeze-base", action="store_true")
    mask_group = parser.add_mutually_exclusive_group()
    mask_group.add_argument(
        "--feature-only",
        action="store_true",
        help="Use only the YOLO feature pyramid; omit all mask inputs.",
    )
    mask_group.add_argument(
        "--use-mask-inputs",
        action="store_true",
        help="Use the full postprocessed sticker-mask stack plus prototype unions.",
    )
    mask_group.add_argument(
        "--raw-image-only",
        action="store_true",
        help="Use only a compact raw-RGB CNN baseline; do not load YOLO.",
    )
    parser.add_argument(
        "--no-perspective",
        action="store_true",
        help="Exclude perspective loss while retaining the target field in labels.",
    )
    parser.add_argument(
        "--head-weight-start",
        type=float,
        default=1.0,
        help="Direct canonical-head loss weight at the first epoch (default: 1.0).",
    )
    parser.add_argument(
        "--head-weight-end",
        type=float,
        default=1.0,
        help="Direct canonical-head loss weight at the final epoch (default: 1.0).",
    )
    parser.add_argument(
        "--geometry-weight-start",
        type=float,
        default=0.0,
        help="Closest-line geometry loss weight at the first epoch (default: 0.0).",
    )
    parser.add_argument(
        "--geometry-weight-end",
        type=float,
        default=0.0,
        help="Closest-line geometry loss weight at the final epoch (default: 0.0).",
    )
    args = parser.parse_args()
    if not 0.0 <= args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be in [0, 1)")
    if any(
        value < 0.0
        for value in (
            args.head_weight_start,
            args.head_weight_end,
            args.geometry_weight_start,
            args.geometry_weight_end,
        )
    ):
        parser.error("loss weights must be non-negative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    samples = collect_samples(args.source_dir)
    random.Random(args.seed).shuffle(samples)
    validation_count = round(len(samples) * args.validation_fraction)
    if validation_count and validation_count == len(samples):
        validation_count -= 1
    validation_samples = samples[:validation_count]
    training_samples = samples[validation_count:]
    if not training_samples:
        raise RuntimeError("The training split is empty")

    train_dataset = CubePoseDataset(training_samples, args.imgsz)
    validation_dataset = CubePoseDataset(validation_samples, args.imgsz) if validation_samples else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = (
        DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        if validation_dataset is not None
        else None
    )

    if args.raw_image_only:
        if args.unfreeze_base:
            parser.error("--unfreeze-base cannot be used with --raw-image-only")
        model = RawImageCubePoseModel().to(device)
        use_soft_masks = False
        model_type = "raw_image"
        base_model = None
    else:
        use_soft_masks = not args.feature_only
        mask_channels = (
            POSE_MASK_CHANNELS
            if args.use_mask_inputs
            else PROTOTYPE_MASK_CHANNELS
        )
        model = YOLOCubePoseModel.from_checkpoint(
            args.model,
            input_size=args.imgsz,
            freeze_base=not args.unfreeze_base,
            use_soft_masks=use_soft_masks,
            mask_channels=mask_channels,
        ).to(device)
        model_type = "yolo"
        base_model = args.model
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    args.output.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    try:
        for epoch in range(1, args.epochs + 1):
            head_weight, geometry_weight = scheduled_loss_weights(
                epoch,
                args.epochs,
                head_start=args.head_weight_start,
                head_end=args.head_weight_end,
                geometry_start=args.geometry_weight_start,
                geometry_end=args.geometry_weight_end,
            )
            train_metrics = run_epoch(
                model,
                train_loader,
                optimizer,
                device,
                use_perspective=not args.no_perspective,
                geometry_weight=geometry_weight,
                head_weight=head_weight,
            )
            if validation_loader is not None:
                with torch.no_grad():
                    validation_metrics = run_epoch(
                        model,
                        validation_loader,
                        None,
                        device,
                        use_perspective=not args.no_perspective,
                        geometry_weight=geometry_weight,
                        head_weight=head_weight,
                    )
            else:
                validation_metrics = {}
            scheduler.step()
            score = validation_metrics.get("total", train_metrics["total"])
            metrics = {
                "train_total": train_metrics["total"],
                "validation_total": score,
                "train_geometry": train_metrics["geometry"],
                "train_head": train_metrics["head"],
                "train_pose_auxiliary": train_metrics["pose_auxiliary"],
                "train_rotation": train_metrics["rotation"],
                "train_center": train_metrics["center"],
                "train_scale": train_metrics["scale"],
                "train_perspective": train_metrics["perspective"],
                "train_layer_angles": train_metrics["layer_angles"],
                "head_weight": head_weight,
                "geometry_weight": geometry_weight,
            }
            print(
                f"epoch {epoch:03d}/{args.epochs}: "
                f"train={train_metrics['total']:.5f} val={score:.5f} "
                f"geo={train_metrics['geometry']:.5f} "
                f"head={train_metrics['head']:.5f} "
                f"w=({head_weight:.2f}h/{geometry_weight:.2f}g) "
                f"rot={train_metrics['rotation']:.5f} "
                f"center={train_metrics['center']:.5f} "
                f"angles={train_metrics['layer_angles']:.5f}"
            )
            save_checkpoint(
                args.output / "last.pt",
                model,
                base_model=base_model,
                model_type=model_type,
                image_size=args.imgsz,
                epoch=epoch,
                metrics=metrics,
                use_soft_masks=use_soft_masks,
            )
            if score < best_loss:
                best_loss = score
                save_checkpoint(
                    args.output / "best.pt",
                    model,
                    base_model=base_model,
                    model_type=model_type,
                    image_size=args.imgsz,
                    epoch=epoch,
                    metrics=metrics,
                    use_soft_masks=use_soft_masks,
                )
    finally:
        model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

