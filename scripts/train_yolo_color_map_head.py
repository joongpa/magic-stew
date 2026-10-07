"""Train a dense seven-class color head on a YOLO cube-segmentation model.

Expected generated-data layout::

    synthetic_cube_dataset/
      images/*.png
      labels/
        stickers/
          classes.json
          *.txt
        whole_cube/
          *.txt

The YOLO cube model is frozen by default. Its feature pyramid is captured
before the existing segmentation head and fused with explicit RGB/color
features by :class:`YOLOColorMapModel`. The loss uses training-set class
balancing and focal weighting, and validation reports per-class metrics and a
target-by-prediction confusion matrix so common color confusions are visible.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from live_recognition.color_segmentation import (  # noqa: E402
    COLOR_CLASS_COUNT,
    COLOR_CLASS_NAMES,
    IGNORE_INDEX,
    NONE_CLASS_INDEX,
    YOLOColorMapModel,
)
from live_recognition.color_targets import dense_color_target  # noqa: E402


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def load_class_names(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [str(value) for value in raw]
    if isinstance(raw, dict):
        return [str(raw[str(index)]) for index in range(len(raw))]
    raise ValueError(f"Expected a list or contiguous-index object in {path}")


def nonempty_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_polygons(path: Path, class_names: list[str]) -> list[tuple[str, list[tuple[float, float]]]]:
    polygons = []
    for line_number, line in enumerate(nonempty_lines(path), start=1):
        values = line.split()
        if len(values) < 7 or (len(values) - 1) % 2:
            raise ValueError(f"{path}:{line_number} is not a valid YOLO polygon")
        class_id = int(float(values[0]))
        if not 0 <= class_id < len(class_names):
            raise ValueError(f"{path}:{line_number} has invalid class id {class_id}")
        coordinates = [float(value) for value in values[1:]]
        if any(value < 0.0 or value > 1.0 for value in coordinates):
            raise ValueError(f"{path}:{line_number} contains coordinates outside [0, 1]")
        polygons.append((class_names[class_id], list(zip(coordinates[::2], coordinates[1::2]))))
    return polygons


def square_roi_from_whole_polygons(
    polygons: list[tuple[str, list[tuple[float, float]]]],
    image_shape: tuple[int, int],
    padding: float,
) -> tuple[int, int, int, int]:
    """Derive the same clamped square ROI convention used by the runtime tools."""

    height, width = image_shape
    points = np.asarray([point for _name, polygon in polygons for point in polygon], dtype=np.float32)
    if points.size == 0:
        return 0, 0, width, height
    pixel_points = points * np.asarray((width, height), dtype=np.float32)
    min_x, min_y = pixel_points.min(axis=0)
    max_x, max_y = pixel_points.max(axis=0)
    side = max(max_x - min_x, max_y - min_y) * (1.0 + 2.0 * padding)
    side = max(1.0, min(side, width, height))
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    x1 = max(0, min(width - int(round(side)), int(round(center_x - side / 2.0))))
    y1 = max(0, min(height - int(round(side)), int(round(center_y - side / 2.0))))
    side_int = max(1, min(int(round(side)), width, height))
    return x1, y1, x1 + side_int, y1 + side_int


@dataclass(frozen=True)
class ColorMapSample:
    image_path: Path
    sticker_label_path: Path
    whole_cube_label_path: Path


class ColorMapDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        samples: list[ColorMapSample],
        *,
        sticker_class_names: list[str],
        image_size: int,
        roi_padding: float,
        boundary_ignore_px: int,
    ) -> None:
        self.samples = samples
        self.sticker_class_names = sticker_class_names
        self.image_size = image_size
        self.roi_padding = roi_padding
        self.boundary_ignore_px = boundary_ignore_px

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        sample = self.samples[index]
        image_bgr = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Could not read {sample.image_path}")
        image_height, image_width = image_bgr.shape[:2]
        sticker_polygons = parse_polygons(sample.sticker_label_path, self.sticker_class_names)
        whole_cube_polygons = parse_polygons(sample.whole_cube_label_path, ["cube"])
        roi_xyxy = square_roi_from_whole_polygons(
            whole_cube_polygons,
            (image_height, image_width),
            self.roi_padding,
        )
        x1, y1, x2, y2 = roi_xyxy
        roi_bgr = image_bgr[y1:y2, x1:x2]
        roi_bgr = cv2.resize(
            roi_bgr,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(roi_rgb.copy()).permute(2, 0, 1).float() / 255.0
        target = dense_color_target(
            (image_height, image_width),
            sticker_polygons,
            self.sticker_class_names,
            roi_xyxy,
            self.image_size,
            boundary_ignore_px=self.boundary_ignore_px,
        )
        return image, torch.from_numpy(target)


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    color_mean_iou: float
    color_macro_f1: float
    pixel_accuracy: float
    sticker_pixel_accuracy: float
    color_ious: tuple[float, ...]
    color_precisions: tuple[float, ...]
    color_recalls: tuple[float, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]
    none_iou: float


def collect_samples(source_dir: Path) -> tuple[list[ColorMapSample], list[str]]:
    images_dir = source_dir / "images"
    sticker_dir = source_dir / "labels" / "stickers"
    whole_cube_dir = source_dir / "labels" / "whole_cube"
    class_names = load_class_names(sticker_dir / "classes.json")
    samples = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        sticker_path = sticker_dir / f"{image_path.stem}.txt"
        whole_cube_path = whole_cube_dir / f"{image_path.stem}.txt"
        if sticker_path.exists() and whole_cube_path.exists():
            samples.append(ColorMapSample(image_path, sticker_path, whole_cube_path))
    if not samples:
        raise RuntimeError(f"No image/sticker/whole-cube triples found below {source_dir}")
    return samples, class_names


def dense_dice_loss(logits: Tensor, target: Tensor) -> Tensor:
    valid = target != IGNORE_INDEX
    safe_target = target.masked_fill(~valid, 0)
    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(safe_target, num_classes=COLOR_CLASS_COUNT).permute(0, 3, 1, 2).float()
    valid_float = valid.unsqueeze(1).float()
    # The downstream segmentation mask already removes most non-sticker
    # pixels. Optimize Dice for the six useful sticker colors; ``none`` is
    # retained as an output channel but is not a primary training objective.
    probabilities = probabilities[:, :NONE_CLASS_INDEX] * valid_float
    one_hot = one_hot[:, :NONE_CLASS_INDEX] * valid_float
    intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    present = one_hot.sum(dim=(0, 2, 3)) > 0
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return (1.0 - dice[present]).mean() if bool(present.any()) else logits.sum() * 0.0


def compute_class_weights(
    dataset: Dataset[tuple[Tensor, Tensor]],
    *,
    none_weight: float,
    balance_strength: float,
) -> Tensor:
    """Estimate stable per-class weights from valid training pixels."""

    counts = torch.zeros(COLOR_CLASS_COUNT, dtype=torch.float64)
    for _image, target in dataset:
        valid_target = target[target != IGNORE_INDEX].long()
        counts += torch.bincount(valid_target, minlength=COLOR_CLASS_COUNT).double()

    color_counts = counts[:NONE_CLASS_INDEX]
    present = color_counts > 0
    color_weights = torch.ones(NONE_CLASS_INDEX, dtype=torch.float64)
    if bool(present.any()):
        reference_count = color_counts[present].mean()
        color_weights[present] = (reference_count / color_counts[present]).pow(balance_strength)
        color_weights[present] /= color_weights[present].mean()

    weights = torch.ones(COLOR_CLASS_COUNT, dtype=torch.float64)
    weights[:NONE_CLASS_INDEX] = color_weights
    weights[NONE_CLASS_INDEX] = none_weight
    return weights.float()


def compute_loss(
    logits: Tensor,
    target: Tensor,
    class_weights: Tensor,
    focal_gamma: float,
) -> Tensor:
    """Use class-balanced focal CE plus class-balanced dense Dice loss."""

    safe_target = target.masked_fill(target == IGNORE_INDEX, 0)
    log_probabilities = F.log_softmax(logits, dim=1)
    target_log_probability = log_probabilities.gather(
        1, safe_target.unsqueeze(1)
    ).squeeze(1)
    probabilities = target_log_probability.exp()
    focal_factor = (1.0 - probabilities).clamp_min(0.0).pow(focal_gamma)
    pixel_weights = class_weights.to(device=logits.device, dtype=logits.dtype)[safe_target]
    valid = (target != IGNORE_INDEX).to(logits.dtype)
    cross_entropy = (
        -target_log_probability * focal_factor * pixel_weights * valid
    ).sum() / valid.sum().clamp_min(1.0)
    return cross_entropy + 0.5 * dense_dice_loss(logits, target)


def update_metrics(
    logits: Tensor,
    target: Tensor,
    intersections: Tensor,
    unions: Tensor,
    correct: list[int],
    total: list[int],
    sticker_correct: list[int],
    sticker_total: list[int],
    confusion_matrix: Tensor,
) -> None:
    predictions = logits.argmax(dim=1)
    valid = target != IGNORE_INDEX
    for class_id in range(COLOR_CLASS_COUNT):
        predicted_class = predictions == class_id
        target_class = target == class_id
        intersections[class_id] += (predicted_class & target_class & valid).sum().item()
        unions[class_id] += ((predicted_class | target_class) & valid).sum().item()
        correct[class_id] += (predicted_class & target_class & valid).sum().item()
        total[class_id] += (target_class & valid).sum().item()
    sticker_pixels = valid & (target != NONE_CLASS_INDEX)
    sticker_correct[0] += ((predictions == target) & sticker_pixels).sum().item()
    sticker_total[0] += sticker_pixels.sum().item()
    valid_target = target[valid].long().cpu()
    valid_predictions = predictions[valid].long().cpu()
    batch_indices = valid_target * COLOR_CLASS_COUNT + valid_predictions
    confusion_matrix += torch.bincount(
        batch_indices,
        minlength=COLOR_CLASS_COUNT * COLOR_CLASS_COUNT,
    ).reshape(COLOR_CLASS_COUNT, COLOR_CLASS_COUNT)


def run_epoch(
    model: YOLOColorMapModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    class_weights: Tensor,
    focal_gamma: float,
    *,
    phase: str,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    batch_total = len(loader)
    progress_interval = max(1, batch_total // 10)
    print(f"{phase}: starting {batch_total} batches", flush=True)
    total_loss = 0.0
    sample_count = 0
    intersections = torch.zeros(COLOR_CLASS_COUNT, dtype=torch.float64)
    unions = torch.zeros(COLOR_CLASS_COUNT, dtype=torch.float64)
    correct = [0] * COLOR_CLASS_COUNT
    total = [0] * COLOR_CLASS_COUNT
    sticker_correct = [0]
    sticker_total = [0]
    confusion_matrix = torch.zeros(
        (COLOR_CLASS_COUNT, COLOR_CLASS_COUNT), dtype=torch.int64
    )
    for batch_number, (images, targets) in enumerate(loader, start=1):
        images = images.to(device)
        targets = targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        logits = outputs["color_logits"]
        loss = compute_loss(logits, targets, class_weights, focal_gamma)
        if training:
            loss.backward()
            optimizer.step()
        batch_size = images.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        sample_count += batch_size
        update_metrics(
            logits.detach(),
            targets,
            intersections,
            unions,
            correct,
            total,
            sticker_correct,
            sticker_total,
            confusion_matrix,
        )
        if (
            batch_number == 1
            or batch_number == batch_total
            or batch_number % progress_interval == 0
        ):
            print(
                f"{phase}: batch {batch_number}/{batch_total} "
                f"loss={float(loss.detach().cpu()):.4f}",
                flush=True,
            )

    true_positive = confusion_matrix.diag().double()
    predicted_counts = confusion_matrix.sum(dim=0).double()
    target_counts = confusion_matrix.sum(dim=1).double()
    color_unions = predicted_counts + target_counts - true_positive
    valid_color_classes = color_unions[:NONE_CLASS_INDEX] > 0
    color_ious_tensor = torch.zeros(NONE_CLASS_INDEX, dtype=torch.float64)
    color_ious_tensor[valid_color_classes] = (
        true_positive[:NONE_CLASS_INDEX][valid_color_classes]
        / color_unions[:NONE_CLASS_INDEX][valid_color_classes]
    )
    color_precisions_tensor = torch.zeros(NONE_CLASS_INDEX, dtype=torch.float64)
    color_recalls_tensor = torch.zeros(NONE_CLASS_INDEX, dtype=torch.float64)
    valid_precision = predicted_counts[:NONE_CLASS_INDEX] > 0
    valid_recall = target_counts[:NONE_CLASS_INDEX] > 0
    color_precisions_tensor[valid_precision] = (
        true_positive[:NONE_CLASS_INDEX][valid_precision]
        / predicted_counts[:NONE_CLASS_INDEX][valid_precision]
    )
    color_recalls_tensor[valid_recall] = (
        true_positive[:NONE_CLASS_INDEX][valid_recall]
        / target_counts[:NONE_CLASS_INDEX][valid_recall]
    )
    color_mean_iou = (
        float(color_ious_tensor[valid_color_classes].mean())
        if bool(valid_color_classes.any())
        else 0.0
    )
    color_f1 = (
        2.0
        * color_precisions_tensor
        * color_recalls_tensor
        / (color_precisions_tensor + color_recalls_tensor).clamp_min(1e-8)
    )
    color_macro_f1 = (
        float(color_f1[valid_color_classes].mean())
        if bool(valid_color_classes.any())
        else 0.0
    )
    none_iou = (
        float(true_positive[NONE_CLASS_INDEX] / color_unions[NONE_CLASS_INDEX])
        if color_unions[NONE_CLASS_INDEX] > 0
        else 0.0
    )
    pixel_accuracy = sum(correct) / max(1, sum(total))
    sticker_pixel_accuracy = sticker_correct[0] / max(1, sticker_total[0])
    print(f"{phase}: complete", flush=True)
    return EpochMetrics(
        loss=total_loss / max(1, sample_count),
        color_mean_iou=color_mean_iou,
        color_macro_f1=color_macro_f1,
        pixel_accuracy=pixel_accuracy,
        sticker_pixel_accuracy=sticker_pixel_accuracy,
        color_ious=tuple(float(value) for value in color_ious_tensor),
        color_precisions=tuple(float(value) for value in color_precisions_tensor),
        color_recalls=tuple(float(value) for value in color_recalls_tensor),
        confusion_matrix=tuple(
            tuple(int(value) for value in row) for row in confusion_matrix.tolist()
        ),
        none_iou=none_iou,
    )


def resolve_device(name: str) -> torch.device:
    requested = torch.device(name)
    if requested.type == "mps" and not torch.backends.mps.is_available():
        print("MPS is unavailable; falling back to CPU")
        return torch.device("cpu")
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU")
        return torch.device("cpu")
    return requested


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/synthetic_cube_dataset"))
    parser.add_argument("--model", type=Path, required=True, help="Whole-cube YOLO segmentation weights")
    parser.add_argument("--workdir", type=Path, default=Path("runs/cube_color_map_head"))
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--roi-padding", type=float, default=0.08)
    parser.add_argument("--boundary-ignore-px", type=int, default=1)
    parser.add_argument(
        "--none-weight",
        type=float,
        default=0.0,
        help="Optional loss weight for background/none targets; 0 focuses training on sticker colors",
    )
    parser.add_argument(
        "--class-balance-strength",
        type=float,
        default=0.5,
        help="Strength of inverse-frequency weighting for the six color classes",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=1.0,
        help="Focal-loss gamma; use 0 to disable hard-pixel emphasis",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--unfreeze-backbone",
        action="store_true",
        help="Fine-tune YOLO weights jointly instead of training only the color head",
    )
    args = parser.parse_args()
    if not args.model.is_file():
        raise SystemExit(f"YOLO model does not exist: {args.model}")
    if args.class_balance_strength < 0.0:
        raise SystemExit("--class-balance-strength must be non-negative")
    if args.focal_gamma < 0.0:
        raise SystemExit("--focal-gamma must be non-negative")
    if args.none_weight < 0.0:
        raise SystemExit("--none-weight must be non-negative")

    print(f"Loading dataset metadata from {args.source_dir}...", flush=True)
    samples, sticker_class_names = collect_samples(args.source_dir)
    random.Random(args.seed).shuffle(samples)
    val_count = max(1, round(len(samples) * args.val_fraction)) if len(samples) > 1 else 0
    val_samples = samples[:val_count]
    train_samples = samples[val_count:]
    train_dataset = ColorMapDataset(
        train_samples,
        sticker_class_names=sticker_class_names,
        image_size=args.imgsz,
        roi_padding=args.roi_padding,
        boundary_ignore_px=args.boundary_ignore_px,
    )
    val_dataset = ColorMapDataset(
        val_samples,
        sticker_class_names=sticker_class_names,
        image_size=args.imgsz,
        roi_padding=args.roi_padding,
        boundary_ignore_px=args.boundary_ignore_px,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )

    class_weights = compute_class_weights(
        train_dataset,
        none_weight=args.none_weight,
        balance_strength=args.class_balance_strength,
    )

    device = resolve_device(args.device)
    print(f"Using device: {device}", flush=True)
    print(f"Loading YOLO segmentation model: {args.model}", flush=True)
    model = YOLOColorMapModel.from_checkpoint(
        args.model,
        input_size=args.imgsz,
        freeze_base=not args.unfreeze_backbone,
    ).to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    args.workdir.mkdir(parents=True, exist_ok=True)
    best_color_miou = -1.0
    print(f"Training samples: {len(train_dataset)}, validation samples: {len(val_dataset)}", flush=True)
    print(f"Color classes: {', '.join(COLOR_CLASS_NAMES)}", flush=True)
    print(
        "Class weights: "
        + " ".join(
            f"{name}={float(weight):.3f}"
            for name, weight in zip(COLOR_CLASS_NAMES, class_weights)
        ),
        flush=True,
    )
    print(f"Focal gamma: {args.focal_gamma:.3f}", flush=True)
    print(
        f"Trainable parameters: {sum(parameter.numel() for parameter in trainable_parameters):,} "
        f"({'YOLO unfrozen' if args.unfreeze_backbone else 'color head only'})",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            class_weights,
            args.focal_gamma,
            phase=f"epoch {epoch} train",
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                None,
                device,
                class_weights,
                args.focal_gamma,
                phase=f"epoch {epoch} val",
            )
        print(
            f"epoch {epoch:03d} "
            f"train loss={train_metrics.loss:.4f} "
            f"color_mIoU={train_metrics.color_mean_iou:.4f} "
            f"color_macroF1={train_metrics.color_macro_f1:.4f} "
            f"sticker_acc={train_metrics.sticker_pixel_accuracy:.4f} "
            f"val loss={val_metrics.loss:.4f} "
            f"color_mIoU={val_metrics.color_mean_iou:.4f} "
            f"color_macroF1={val_metrics.color_macro_f1:.4f} "
            f"sticker_acc={val_metrics.sticker_pixel_accuracy:.4f} "
            f"none_IoU={val_metrics.none_iou:.4f}",
            flush=True,
        )
        per_color = " ".join(
            f"{name}=IoU:{iou:.3f}/P:{precision:.3f}/R:{recall:.3f}"
            for name, iou, precision, recall in zip(
                COLOR_CLASS_NAMES[:-1],
                val_metrics.color_ious,
                val_metrics.color_precisions,
                val_metrics.color_recalls,
            )
        )
        print(f"validation per-color metrics: {per_color}", flush=True)
        print("validation confusion matrix (rows=target, cols=prediction):", flush=True)
        for name, row in zip(COLOR_CLASS_NAMES, val_metrics.confusion_matrix):
            print(f"  {name:>6}: {row}", flush=True)
        if val_metrics.color_mean_iou > best_color_miou:
            best_color_miou = val_metrics.color_mean_iou
            checkpoint = {
                "model": model.state_dict(),
                "base_model": str(args.model.resolve()),
                "imgsz": args.imgsz,
                "class_names": COLOR_CLASS_NAMES,
                "epoch": epoch,
                "val_miou": val_metrics.color_mean_iou,
                "class_weights": class_weights.tolist(),
                "focal_gamma": args.focal_gamma,
            }
            torch.save(checkpoint, args.workdir / "best_color_map_head.pt")

    torch.save(
        {
            "model": model.state_dict(),
            "base_model": str(args.model.resolve()),
            "imgsz": args.imgsz,
            "class_names": COLOR_CLASS_NAMES,
            "val_miou": best_color_miou,
            "class_weights": class_weights.tolist(),
            "focal_gamma": args.focal_gamma,
        },
        args.workdir / "last_color_map_head.pt",
    )
    print(f"Best validation color mIoU: {best_color_miou:.4f}", flush=True)


if __name__ == "__main__":
    main()
