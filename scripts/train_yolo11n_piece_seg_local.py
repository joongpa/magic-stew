"""Train YOLO11 nano pose locally on generated piece-wise cube keypoint labels.

Expected input layout:

    synthetic_cube_dataset/
      images/*.png
      labels/*.txt
      masks/*.png          # optional visualization masks, not used for training

The script shuffles image/label pairs with a fixed seed, writes a local
Ultralytics dataset layout, and trains yolo11n-pose.pt.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import shutil
from collections import Counter
from pathlib import Path

try:
    import torch
    from ultralytics.models.yolo.pose.train import PoseTrainer
    from ultralytics.nn.tasks import PoseModel
    from ultralytics.utils import RANK
    from ultralytics.utils.loss import KeypointLoss, v8PoseLoss
    from ultralytics.utils.ops import xyxy2xywh
except ImportError:
    torch = None
    PoseTrainer = object
    PoseModel = object
    RANK = -1
    KeypointLoss = object
    v8PoseLoss = object
    xyxy2xywh = None


CLASS_NAMES = ["corner", "edge"]
CORNER_CLASS_ID = 0
EDGE_CLASS_ID = 1
KPT_SHAPE = [7, 3]
POSE_LABEL_FIELD_COUNT = 1 + 4 + (KPT_SHAPE[0] * KPT_SHAPE[1])
BBOX_FIELD_COUNT = 4
KEYPOINT_VISIBILITY_VALUES = {0, 1, 2}
PIECE_KEYPOINT_SIGMA = 0.08


def label_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def corner_keypoint_permutations() -> list[list[int]]:
    permutations = [
        [0, *edge_indexes, *center_indexes]
        for edge_indexes in itertools.permutations((1, 2, 3))
        for center_indexes in itertools.permutations((4, 5, 6))
    ]
    if len(permutations) != 36 or len({tuple(permutation) for permutation in permutations}) != 36:
        raise ValueError("Expected 36 unique corner keypoint permutations")
    return permutations


def edge_keypoint_permutations() -> list[list[int]]:
    permutations = []
    for edge_swap, first_side_swap, second_side_swap, side_pair_swap in itertools.product((False, True), repeat=4):
        edge_line = [1, 0] if edge_swap else [0, 1]
        first_side = [3, 2] if first_side_swap else [2, 3]
        second_side = [5, 4] if second_side_swap else [4, 5]
        center_sides = [second_side, first_side] if side_pair_swap else [first_side, second_side]
        permutations.append([*edge_line, *center_sides[0], *center_sides[1], 6])

    if len(permutations) != 16 or len({tuple(permutation) for permutation in permutations}) != 16:
        raise ValueError("Expected 16 unique edge keypoint permutations")
    return permutations


def validate_label_file(path: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    lines = label_lines(path)
    for line_number, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != POSE_LABEL_FIELD_COUNT:
            raise ValueError(
                f"{path}:{line_number} is not valid YOLO pose format; "
                f"expected {POSE_LABEL_FIELD_COUNT} fields for class, bbox, and {KPT_SHAPE[0]} keypoints, "
                f"got {len(parts)} fields"
            )
        class_id = int(float(parts[0]))
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            raise ValueError(f"{path}:{line_number} has invalid class id {class_id}")
        bbox_values = [float(part) for part in parts[1 : 1 + BBOX_FIELD_COUNT]]
        keypoint_values = [float(part) for part in parts[1 + BBOX_FIELD_COUNT :]]
        for value_index, value in enumerate(bbox_values, start=1):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{path}:{line_number} bbox value {value_index} is outside [0, 1]: {value}")
        for keypoint_index in range(KPT_SHAPE[0]):
            offset = keypoint_index * KPT_SHAPE[1]
            x, y, visibility = keypoint_values[offset : offset + KPT_SHAPE[1]]
            if not 0.0 <= x <= 1.0:
                raise ValueError(f"{path}:{line_number} keypoint {keypoint_index} x is outside [0, 1]: {x}")
            if not 0.0 <= y <= 1.0:
                raise ValueError(f"{path}:{line_number} keypoint {keypoint_index} y is outside [0, 1]: {y}")
            if int(visibility) != visibility or int(visibility) not in KEYPOINT_VISIBILITY_VALUES:
                raise ValueError(
                    f"{path}:{line_number} keypoint {keypoint_index} visibility must be 0, 1, or 2: {visibility}"
                )
        counts[class_id] += 1
    return counts


def collect_pairs(source_dir: Path) -> list[tuple[Path, Path]]:
    images_dir = source_dir / "images"
    labels_dir = source_dir / "labels"
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Missing labels directory: {labels_dir}")

    image_paths = sorted(
        path for path in images_dir.glob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    label_by_stem = {path.stem: path for path in labels_dir.glob("*.txt")}
    pairs = [(image_path, label_by_stem[image_path.stem]) for image_path in image_paths if image_path.stem in label_by_stem]
    if not pairs:
        raise RuntimeError(f"No image/label pairs found under {source_dir}")
    return pairs


def write_dataset(
    source_dir: Path,
    dataset_dir: Path,
    val_fraction: float,
    seed: int,
) -> Path:
    pairs = collect_pairs(source_dir)
    random.Random(seed).shuffle(pairs)

    val_count = max(1, round(len(pairs) * val_fraction)) if len(pairs) > 1 else 0
    splits = {"val": pairs[:val_count], "train": pairs[val_count:]}

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    for split in ["train", "val"]:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest = []
    summary = {}
    for split, split_pairs in splits.items():
        class_counts: Counter[int] = Counter()
        label_count_distribution: Counter[int] = Counter()

        for index, (image_path, label_path) in enumerate(split_pairs):
            lines = label_lines(label_path)
            label_count_distribution[len(lines)] += 1
            class_counts.update(validate_label_file(label_path))

            target_stem = f"{split}_{index:06d}"
            target_image = dataset_dir / "images" / split / f"{target_stem}{image_path.suffix.lower()}"
            target_label = dataset_dir / "labels" / split / f"{target_stem}.txt"
            shutil.copy2(image_path, target_image)
            shutil.copy2(label_path, target_label)
            manifest.append({
                "split": split,
                "source_image": str(image_path),
                "source_label": str(label_path),
                "image": str(target_image),
                "label": str(target_label),
                "objects": len(lines),
            })

        summary[split] = {
            "images": len(split_pairs),
            "label_count_distribution": dict(sorted(label_count_distribution.items())),
            "class_objects": {
                CLASS_NAMES[class_id]: class_counts[class_id] for class_id in range(len(CLASS_NAMES))
            },
        }

    (dataset_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(
        "path: " + str(dataset_dir.resolve()) + "\n"
        "train: images/train\n"
        "val: images/val\n"
        f"kpt_shape: [{KPT_SHAPE[0]}, {KPT_SHAPE[1]}]\n"
        "names:\n"
        + "".join(f"  {index}: {name}\n" for index, name in enumerate(CLASS_NAMES)),
        encoding="utf-8",
    )

    print("Dataset summary:")
    print(json.dumps(summary, indent=2))
    print(data_yaml.read_text())
    return data_yaml


class PerAnchorKeypointLoss(KeypointLoss):
    """Ultralytics KeypointLoss with the final batch reduction delayed."""

    def forward_per_anchor(
        self,
        pred_kpts: torch.Tensor,
        gt_kpts: torch.Tensor,
        kpt_mask: torch.Tensor,
        area: torch.Tensor,
    ) -> torch.Tensor:
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean(dim=1)


class CubeSymmetricPoseLoss(v8PoseLoss):
    def __init__(self, model, tal_topk: int = 10, tal_topk2: int = 10):
        if torch is None:
            raise ImportError("PyTorch and Ultralytics are required for cube-symmetric pose training")
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        # Ultralytics defaults non-COCO keypoint sigmas to 1 / nkpt. With 7 piece
        # keypoints that can still saturate the OKS-style pose loss and
        # leave almost no useful gradient at startup.
        piece_sigmas = torch.full(
            (KPT_SHAPE[0],),
            PIECE_KEYPOINT_SIGMA,
            device=self.device,
            dtype=torch.float32,
        )
        self.keypoint_loss.sigmas = piece_sigmas
        self.per_anchor_keypoint_loss = PerAnchorKeypointLoss(piece_sigmas)
        self.corner_permutations = torch.tensor(
            corner_keypoint_permutations(),
            dtype=torch.long,
            device=self.device,
        )
        self.edge_permutations = torch.tensor(
            edge_keypoint_permutations(),
            dtype=torch.long,
            device=self.device,
        )

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate pose loss with class-specific keypoint permutation symmetry."""
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
        loss = torch.zeros(5, device=self.device)  # box, kpt_location, kpt_visibility, cls, dfl
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        batch_size = pred_kpts.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))

        if fg_mask.sum():
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]
            target_classes = self._select_target_classes(
                batch["cls"].to(self.device),
                batch["batch_idx"].view(-1, 1),
                target_gt_idx,
                fg_mask,
            )

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,
                target_bboxes,
                pred_kpts,
                target_classes,
            )

        loss[1] *= self.hyp.pose
        loss[2] *= self.hyp.kobj

        return loss * batch_size, loss.detach()

    @staticmethod
    def _select_target_classes(
        classes: torch.Tensor,
        batch_idx: torch.Tensor,
        target_gt_idx: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        batch_idx = batch_idx.flatten().to(classes.device)
        batch_size = len(masks)
        max_targets = torch.unique(batch_idx, return_counts=True)[1].max()
        batched_classes = torch.full(
            (batch_size, max_targets),
            -1,
            dtype=torch.long,
            device=classes.device,
        )

        batch_idx_long = batch_idx.long()
        offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=classes.device)
        offsets.scatter_add_(0, batch_idx_long + 1, torch.ones_like(batch_idx_long))
        offsets = offsets.cumsum(0)
        within_idx = torch.arange(len(batch_idx), device=classes.device) - offsets[batch_idx_long]
        batched_classes[batch_idx_long, within_idx] = classes.view(-1).long()

        selected_classes = batched_classes.gather(1, target_gt_idx.to(classes.device))
        return selected_classes[masks.to(classes.device)].view(-1)

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
        target_classes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = torch.zeros((), device=self.device)
        kpts_obj_loss = torch.zeros((), device=self.device)
        if not masks.any():
            return kpts_loss, kpts_obj_loss

        target_bboxes_scaled = target_bboxes / stride_tensor
        gt_kpt_all = selected_keypoints[masks]
        area_all = xyxy2xywh(target_bboxes_scaled[masks])[:, 2:].prod(1, keepdim=True)
        pred_kpt_all = pred_kpts[masks]
        target_classes = target_classes.to(self.device)

        best_anchor_losses = torch.zeros((gt_kpt_all.shape[0],), device=self.device)
        best_mask = torch.zeros(gt_kpt_all.shape[:2], dtype=torch.bool, device=self.device)
        class_permutations = (
            (CORNER_CLASS_ID, self.corner_permutations),
            (EDGE_CLASS_ID, self.edge_permutations),
        )
        for class_id, permutations in class_permutations:
            class_mask = target_classes == class_id
            if not class_mask.any():
                continue
            gt_kpt = gt_kpt_all[class_mask]
            area = area_all[class_mask]
            pred_kpt = pred_kpt_all[class_mask]

            permutation_losses = []
            permutation_masks = []
            for permutation in permutations:
                permuted_gt = gt_kpt.index_select(1, permutation)
                kpt_mask = (
                    permuted_gt[..., 2] != 0
                    if permuted_gt.shape[-1] == 3
                    else torch.full_like(permuted_gt[..., 0], True, dtype=torch.bool)
                )
                permutation_masks.append(kpt_mask)
                per_anchor_loss = self.per_anchor_keypoint_loss.forward_per_anchor(pred_kpt, permuted_gt, kpt_mask, area)
                permutation_losses.append(per_anchor_loss)

            stacked_losses = torch.stack(permutation_losses)
            class_best_losses, class_best_permutation = stacked_losses.min(dim=0)
            best_anchor_losses[class_mask] = class_best_losses

            if pred_kpt.shape[-1] == 3:
                stacked_masks = torch.stack(permutation_masks)
                anchor_indexes = torch.arange(class_best_permutation.shape[0], device=class_best_permutation.device)
                best_mask[class_mask] = stacked_masks[class_best_permutation, anchor_indexes]

        kpts_loss = best_anchor_losses.mean()

        if pred_kpt_all.shape[-1] == 3:
            kpts_obj_loss = self.bce_pose(pred_kpt_all[..., 2], best_mask.float())

        return kpts_loss, kpts_obj_loss


class CubeSymmetricPoseModel(PoseModel):
    def init_criterion(self):
        return CubeSymmetricPoseLoss(self)


class CubeSymmetricPoseTrainer(PoseTrainer):
    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = CubeSymmetricPoseModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            data_kpt_shape=self.data["kpt_shape"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model


def cube_symmetric_pose_trainer():
    if torch is None:
        raise ImportError("PyTorch and Ultralytics are required for cube-symmetric pose training")
    return CubeSymmetricPoseTrainer


def train(data_yaml: Path, runs_dir: Path, args: argparse.Namespace) -> Path:
    from ultralytics import YOLO

    model = YOLO(args.model)
    trainer = None if args.no_symmetric_piece_loss else cube_symmetric_pose_trainer()
    model.train(
        trainer=trainer,
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(runs_dir),
        name=args.name,
        patience=args.patience,
        cache=args.cache,
        plots=True,
    )

    best_model_path = runs_dir / args.name / "weights" / "best.pt"
    print(f"Best model: {best_model_path}")

    best_model = YOLO(str(best_model_path))
    metrics = best_model.val(data=str(data_yaml), imgsz=args.imgsz, device=args.device)
    print(metrics)
    return best_model_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("data/synthetic_cube_dataset"))
    parser.add_argument("--workdir", type=Path, default=Path("runs/cube_piece_keypoint_pose_local"))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--model", default="yolo11n-pose.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--name", default="cube_piece_keypoint_yolo11n_pose")
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--no-symmetric-piece-loss",
        action="store_true",
        help="Use the stock YOLO pose loss instead of minimizing over corner/edge keypoint symmetries.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    dataset_dir = args.workdir / "dataset"
    runs_dir = args.workdir / "runs"
    data_yaml = write_dataset(
        source_dir=args.source_dir,
        dataset_dir=dataset_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    if args.prepare_only:
        print(f"Prepared dataset: {dataset_dir}")
        return

    train(data_yaml, runs_dir, args)


if __name__ == "__main__":
    main()
