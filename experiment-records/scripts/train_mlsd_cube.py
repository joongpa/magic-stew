#!/usr/bin/env python3
"""Fine-tune the PyTorch M-LSD model on the MCP cube wireframes.

This uses the existing normalized wireframe labels under ``data/mcp_dataset_v4``
and adds the empty-line ROI images under ``data/negative_examples``. The
pretrained M-LSD checkpoint is used as the initialization; the TFLite model in
``.models`` is inference-only and is not used for training.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
MLSD_ROOT = ROOT / ".mlsd_pytorch"
if not MLSD_ROOT.is_dir():
    raise RuntimeError("Missing .mlsd_pytorch; clone the PyTorch M-LSD repository first.")
if str(MLSD_ROOT) not in sys.path:
    sys.path.insert(0, str(MLSD_ROOT))

from mlsd_pytorch.loss import LineSegmentLoss  # noqa: E402
from mlsd_pytorch.models.build_model import build_model  # noqa: E402


_DATA_UTILS_SPEC = importlib.util.spec_from_file_location(
    "mlsd_cube_data_utils", MLSD_ROOT / "mlsd_pytorch/data/utils.py"
)
if _DATA_UTILS_SPEC is None or _DATA_UTILS_SPEC.loader is None:
    raise RuntimeError("Could not load the M-LSD target-generation utilities")
_DATA_UTILS = importlib.util.module_from_spec(_DATA_UTILS_SPEC)
_DATA_UTILS_SPEC.loader.exec_module(_DATA_UTILS)
gen_SOL_map = _DATA_UTILS.gen_SOL_map
gen_TP_mask2 = _DATA_UTILS.gen_TP_mask2
gen_junction_and_line_mask = _DATA_UTILS.gen_junction_and_line_mask
get_ext_lines = _DATA_UTILS.get_ext_lines


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_DATASET = ROOT / "data/mcp_dataset_v4"
DEFAULT_NEGATIVE = ROOT / "data/negative_examples/negative_ROI_backgrounds"
DEFAULT_CHECKPOINT = MLSD_ROOT / "models/mlsd_large_512_fp32.pth"
DEFAULT_OUTPUT = ROOT / "runs/mlsd_cube_finetune"


@dataclass(frozen=True)
class Sample:
    image_path: Path
    lines: np.ndarray
    negative: bool


def parse_wireframe_label(path: Path) -> np.ndarray:
    lines: list[list[float]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        parts = raw_line.split()
        if len(parts) != 5 or int(parts[0]) != 0:
            raise ValueError(f"Unsupported wireframe label: {path}: {raw_line}")
        values = [float(value) for value in parts[1:]]
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"Out-of-range normalized label in {path}: {raw_line}")
        lines.append(values)
    return np.asarray(lines, dtype=np.float32).reshape(-1, 4)


def collect_samples(dataset_dir: Path, negative_dir: Path) -> list[Sample]:
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels" / "wireframe"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Expected images/ and labels/wireframe/ under {dataset_dir}")

    labels = {path.stem: path for path in labels_dir.glob("*.txt")}
    samples = [
        Sample(path, parse_wireframe_label(labels[path.stem]), False)
        for path in sorted(images_dir.iterdir())
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.stem in labels
    ]
    if not negative_dir.is_dir():
        raise FileNotFoundError(f"Missing negative-image directory: {negative_dir}")
    samples.extend(
        Sample(path, np.empty((0, 4), dtype=np.float32), True)
        for path in sorted(negative_dir.iterdir())
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not samples:
        raise RuntimeError("No training samples found")
    return samples


def transform_lines(lines: np.ndarray, *, hflip: bool, vflip: bool, rotate_cw: bool) -> np.ndarray:
    transformed = lines.copy()
    if hflip:
        transformed[:, [0, 2]] = 1.0 - transformed[:, [0, 2]]
    if vflip:
        transformed[:, [1, 3]] = 1.0 - transformed[:, [1, 3]]
    if rotate_cw:
        x0, y0, x1, y1 = transformed.T
        transformed = np.stack((1.0 - y0, x0, 1.0 - y1, x1), axis=1)
    return transformed


def make_target(lines: np.ndarray, input_size: int) -> tuple[np.ndarray, list[list[float]], list[list[float]]]:
    map_size = input_size // 2
    norm_lines = lines.tolist()
    tp_map = gen_TP_mask2(norm_lines, map_size, map_size, with_ext=False)
    sol_map, _ = gen_SOL_map(norm_lines, map_size, map_size, with_ext=False)
    junction_map, line_map = gen_junction_and_line_mask(norm_lines, map_size, map_size)
    label = np.zeros((16, map_size, map_size), dtype=np.float32)
    label[0:7] = sol_map
    label[7:14] = tp_map
    label[14] = junction_map[0]
    label[15] = line_map[0]
    lines_512 = (lines * 512.0).tolist()
    sol_lines_512 = (np.asarray(get_ext_lines(norm_lines, map_size, map_size), dtype=np.float32) * 512.0).tolist()
    return label, lines_512, sol_lines_512


class CubeLineDataset(Dataset):
    def __init__(self, samples: list[Sample], input_size: int, train: bool, seed: int) -> None:
        self.samples = samples
        self.input_size = input_size
        self.train = train
        self.rng = random.Random(seed)
        self.mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
        self.std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Could not read {sample.image_path}")
        lines = sample.lines.copy()
        if self.train:
            hflip = self.rng.random() < 0.5
            vflip = self.rng.random() < 0.5
            rotate_cw = self.rng.random() < 0.25
            if hflip:
                image = cv2.flip(image, 1)
            if vflip:
                image = cv2.flip(image, 0)
            if rotate_cw:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            lines = transform_lines(lines, hflip=hflip, vflip=vflip, rotate_cw=rotate_cw)
            if self.rng.random() < 0.5:
                alpha = 1.0 + self.rng.uniform(-0.15, 0.15)
                beta = self.rng.uniform(-18.0, 18.0)
                image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        image = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image_rgb = (image_rgb - self.mean) / self.std
        label, lines_512, sol_lines_512 = make_target(lines, self.input_size)
        # The released M-LSD checkpoint has an RGBA input stem. Its alpha
        # weights are zero, but retain the channel so the checkpoint loads
        # exactly and the fine-tuned model stays compatible with TFLite.
        alpha_channel = np.ones((self.input_size, self.input_size, 1), dtype=np.float32)
        image_rgba = np.concatenate((image_rgb, alpha_channel), axis=2)
        image_tensor = torch.from_numpy(image_rgba.transpose(2, 0, 1).copy()).float()
        return image_tensor, torch.from_numpy(label), lines_512, sol_lines_512, str(sample.image_path)


def collate(batch):
    images, labels, gt_lines, sol_lines, paths = zip(*batch)
    return {
        "xs": torch.stack(images),
        "ys": torch.stack(labels).float(),
        "gt_lines": list(gt_lines),
        "sol_lines": list(sol_lines),
        "paths": list(paths),
    }


def safe_displacement_loss(pred_dis: torch.Tensor, gt_dis: torch.Tensor, _gt_center=None) -> torch.Tensor:
    pos_mask = (gt_dis.abs().sum(dim=1) != 0).float()
    count = pos_mask.sum().clamp_min(1.0)
    pred_dis = pred_dis * pos_mask.unsqueeze(1)
    gt_dis = gt_dis * pos_mask.unsqueeze(1)
    loss_a = F.smooth_l1_loss(pred_dis, gt_dis, reduction="none").sum(dim=1)
    pred_swap = torch.cat((pred_dis[:, 2:], pred_dis[:, :2]), dim=1)
    loss_b = F.smooth_l1_loss(pred_swap, gt_dis, reduction="none").sum(dim=1)
    return torch.minimum(loss_a, loss_b).sum() / count


def safe_len_angle_loss(pred_len, pred_angle, gt_len, gt_angle):
    mask = (gt_len != 0).float()
    count = mask.sum().clamp_min(1.0)
    pred_len = torch.sigmoid(pred_len)
    pred_angle = torch.sigmoid(pred_angle)
    length_loss = (F.smooth_l1_loss(pred_len, gt_len, reduction="none") * mask).sum() / count
    angle_loss = (F.smooth_l1_loss(pred_angle, gt_angle, reduction="none") * mask).sum() / count
    return length_loss, angle_loss


def safe_weighted_bce(logits, target, pos_w=1.0, neg_w=30.0):
    pos = (target != 0.0).float()
    neg = 1.0 - pos
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pos_loss = (loss * pos).sum() / pos.sum().clamp_min(1.0)
    neg_loss = (loss * neg).sum() / neg.sum().clamp_min(1.0)
    return pos_loss * pos_w + neg_loss * neg_w


def safe_focal_neg_loss_with_logits(logits, target, alpha=2, beta=4):
    """Numerically stable version of the upstream M-LSD focal loss."""

    probabilities = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
    positive = target.eq(1)
    negative = target.lt(1)
    negative_weights = torch.pow(1.0 - target[negative], beta)
    positive_probabilities = probabilities[positive]
    negative_probabilities = probabilities[negative]
    positive_loss = torch.log(positive_probabilities) * torch.pow(1.0 - positive_probabilities, alpha)
    negative_loss = torch.log(1.0 - negative_probabilities) * torch.pow(
        negative_probabilities, alpha
    ) * negative_weights
    positive_count = positive.float().sum()
    if positive_probabilities.numel() == 0:
        return -negative_loss.sum()
    return -(positive_loss.sum() + negative_loss.sum()) / positive_count.clamp_min(1.0)


def patch_negative_only_loss_handling() -> None:
    """Avoid NaNs when a batch contains only empty-line negative images."""

    import mlsd_pytorch.loss.mlsd_multi_loss as loss_module

    loss_module.safe_weighted_bce_with_logits = safe_weighted_bce
    loss_module.weighted_bce_with_logits = safe_weighted_bce
    loss_module.displacement_loss_func = safe_displacement_loss
    loss_module.len_and_angle_loss_func = safe_len_angle_loss
    loss_module.focal_neg_loss_with_logits = safe_focal_neg_loss_with_logits


def build_finetune_model(checkpoint: Path, device: torch.device, model_name: str):
    cfg = SimpleNamespace(model=SimpleNamespace(model_name=model_name, with_deconv=True))
    # The repository's constructor tries to download ImageNet weights before
    # loading the complete M-LSD checkpoint. The checkpoint already contains
    # the backbone, so suppress that optional network lookup.
    import torch.utils.model_zoo as model_zoo

    original_load_url = model_zoo.load_url
    model_zoo.load_url = lambda *_args, **_kwargs: {}
    try:
        model = build_model(cfg)
    finally:
        model_zoo.load_url = original_load_url
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if all(str(key).startswith("module.") for key in state):
        state = {str(key)[len("module.") :]: value for key, value in state.items()}
    stem = model.backbone.features[0][0]
    checkpoint_stem = state.get("backbone.features.0.0.weight")
    if checkpoint_stem is not None and checkpoint_stem.shape[1] != stem.in_channels:
        if checkpoint_stem.shape[1] != 4 or stem.in_channels != 3:
            raise RuntimeError(
                "Unsupported M-LSD input stem: "
                f"checkpoint has {checkpoint_stem.shape[1]} channels, model has {stem.in_channels}"
            )
        rgba_stem = torch.nn.Conv2d(
            4,
            stem.out_channels,
            kernel_size=stem.kernel_size,
            stride=stem.stride,
            padding=stem.padding,
            dilation=stem.dilation,
            groups=stem.groups,
            bias=stem.bias is not None,
            padding_mode=stem.padding_mode,
        )
        with torch.no_grad():
            rgba_stem.weight[:, :3].copy_(stem.weight)
            rgba_stem.weight[:, 3:].zero_()
            if stem.bias is not None and rgba_stem.bias is not None:
                rgba_stem.bias.copy_(stem.bias)
        model.backbone.features[0][0] = rgba_stem
    model.load_state_dict(state, strict=True)
    return model.to(device)


def make_cfg(input_size: int):
    return SimpleNamespace(
        datasets=SimpleNamespace(input_size=input_size, with_centermap_extend=False),
        model=SimpleNamespace(model_name="mobilev2_mlsd_large", with_deconv=True),
        loss=SimpleNamespace(
            with_sol_loss=True,
            with_match_loss=False,
            with_focal_loss=True,
            focal_loss_level=0,
            match_sap_thresh=5.0,
            loss_weight_dict_list=[{"tp_center_loss": 10.0, "sol_center_loss": 1.0}],
        ),
        decode=SimpleNamespace(score_thresh=0.05, len_thresh=5, top_k=500),
    )


def move_batch(batch, device: torch.device):
    return batch["xs"].to(device), batch["ys"].to(device), batch["gt_lines"], batch["sol_lines"]


def evaluate(model, loader, loss_fn, device, max_batches: int | None = None) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            images, labels, gt_lines, sol_lines = move_batch(batch, device)
            loss = loss_fn(model(images), labels, gt_lines, sol_lines)["loss"]
            total += float(loss.item())
            count += 1
            if max_batches is not None and batch_index + 1 >= max_batches:
                break
    return total / max(count, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--negative-dir", type=Path, default=DEFAULT_NEGATIVE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--input-size",
        type=int,
        default=256,
        help="Square training resolution; normalized labels support mixed source sizes (default: 256)",
    )
    parser.add_argument("--model", choices=("mobilev2_mlsd_large", "mobilev2_mlsd"), default="mobilev2_mlsd_large")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"Could not find pretrained M-LSD checkpoint: {args.checkpoint}")
    if not 0.0 < args.val_fraction < 1.0:
        raise SystemExit("--val-fraction must be between 0 and 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    samples = collect_samples(args.dataset_dir, args.negative_dir)
    random.Random(args.seed).shuffle(samples)
    val_count = max(1, round(len(samples) * args.val_fraction))
    valid_samples = samples[:val_count]
    train_samples = samples[val_count:]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "split_summary.json").write_text(
        json.dumps(
            {
                "train": {"images": len(train_samples), "negative": sum(s.negative for s in train_samples)},
                "valid": {"images": len(valid_samples), "negative": sum(s.negative for s in valid_samples)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    train_loader = DataLoader(
        CubeLineDataset(train_samples, args.input_size, True, args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    valid_loader = DataLoader(
        CubeLineDataset(valid_samples, args.input_size, False, args.seed),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    patch_negative_only_loss_handling()
    model = build_finetune_model(args.checkpoint, device, args.model)
    cfg = make_cfg(args.input_size)
    loss_fn = LineSegmentLoss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_val = float("inf")
    log_path = args.output_dir / "loss.csv"
    log_path.write_text("epoch,train_loss,val_loss,learning_rate\n", encoding="utf-8")

    print(f"Device: {device}")
    print(f"Train samples: {len(train_samples)} ({sum(s.negative for s in train_samples)} negative)")
    print(f"Valid samples: {len(valid_samples)} ({sum(s.negative for s in valid_samples)} negative)")
    print(f"Pretrained checkpoint: {args.checkpoint}")

    for epoch in range(args.epochs):
        freeze = epoch < args.freeze_backbone_epochs
        for parameter in model.backbone.parameters():
            parameter.requires_grad = not freeze
        model.train()
        if freeze:
            model.backbone.eval()
        train_total = 0.0
        train_count = 0
        for batch_index, batch in enumerate(train_loader):
            images, labels, gt_lines, sol_lines = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels, gt_lines, sol_lines)["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite M-LSD loss at epoch {epoch + 1}, batch {batch_index}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            train_total += float(loss.item())
            train_count += 1
            if args.max_train_batches is not None and batch_index + 1 >= args.max_train_batches:
                break

        val_loss = evaluate(model, valid_loader, loss_fn, device, args.max_val_batches)
        train_loss = train_total / max(train_count, 1)
        learning_rate = optimizer.param_groups[0]["lr"]
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"{epoch + 1},{train_loss:.8f},{val_loss:.8f},{learning_rate:.8g}\n")
        print(
            f"epoch {epoch + 1:03d}/{args.epochs:03d} train={train_loss:.5f} "
            f"val={val_loss:.5f} lr={learning_rate:.3g} frozen={freeze}"
        )

        state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss,
            "args": vars(args),
        }
        torch.save(state, args.output_dir / "checkpoint_latest.pth")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(state, args.output_dir / "checkpoint_best.pth")

    print(f"Best validation loss: {best_val:.6f}")
    print(f"Output directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
