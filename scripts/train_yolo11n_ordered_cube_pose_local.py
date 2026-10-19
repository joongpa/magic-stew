"""Train YOLO pose on the ordered 26-point cube labels.

The Blender generator writes this source layout::

    synthetic_cube_dataset/
      images/*.png
      pose/*.txt
      ordered_keypoint_symmetries.json

Each pose row is standard YOLO pose format with one cube object and 26
keypoints (x, y, visibility).  The trainer prepares ordinary Ultralytics
``images/{train,val}`` and ``labels/{train,val}`` folders, then replaces the
stock point-by-point pose loss with a symmetry-aware loss.  For every matched
cube, the loss evaluates all 24 proper cube rotations and uses the lowest
weighted keypoint location/visibility loss.  The permutation convention is
the same one written by the Blender generator:

    permutation[output_index] = source_index

The prediction remains a normal 26-point YOLO pose output.  Symmetry is only
used while comparing it to the target, so downstream keypoint indices remain
stable for geometric post-processing.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ultralytics.nn.tasks import PoseModel  # noqa: E402
from ultralytics.models.yolo.pose.train import PoseTrainer  # noqa: E402
from ultralytics.utils.loss import v8PoseLoss  # noqa: E402
from ultralytics.utils.ops import xyxy2xywh  # noqa: E402


KEYPOINT_COUNT = 26
KEYPOINT_DIMS = 3
POSE_LABEL_FIELD_COUNT = 1 + 4 + KEYPOINT_COUNT * KEYPOINT_DIMS
KEYPOINT_VISIBILITY_VALUES = {0, 1, 2}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
CLASS_NAMES = ("cube",)
ORDERED_KEYPOINT_NAMES = (
    "UBL", "UBR", "UFR", "UFL",
    "DBL", "DBR", "DFR", "DFL",
    "UB", "UR", "UF", "UL",
    "DB", "DR", "DF", "DL",
    "BR", "BL", "FR", "FL",
    "U", "D", "F", "B", "R", "L",
)


def label_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_label_file(path: Path) -> None:
    lines = label_lines(path)
    if len(lines) != 1:
        raise ValueError(f"{path} must contain exactly one cube pose row, got {len(lines)}")
    parts = lines[0].split()
    if len(parts) != POSE_LABEL_FIELD_COUNT:
        raise ValueError(
            f"{path} is not a 26-point YOLO pose label; expected "
            f"{POSE_LABEL_FIELD_COUNT} fields, got {len(parts)}"
        )
    class_id = int(float(parts[0]))
    if class_id != 0:
        raise ValueError(f"{path} must use cube class 0, got {class_id}")
    bbox_values = [float(value) for value in parts[1:5]]
    if any(not 0.0 <= value <= 1.0 for value in bbox_values):
        raise ValueError(f"{path} has a bounding-box value outside [0, 1]")
    keypoint_values = [float(value) for value in parts[5:]]
    for keypoint_index in range(KEYPOINT_COUNT):
        offset = keypoint_index * KEYPOINT_DIMS
        x, y, visibility = keypoint_values[offset : offset + KEYPOINT_DIMS]
        if int(visibility) != visibility or int(visibility) not in KEYPOINT_VISIBILITY_VALUES:
            raise ValueError(
                f"{path} keypoint {keypoint_index} visibility must be 0, 1, or 2; got {visibility}"
            )
        if int(visibility) != 0 and (not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0):
            raise ValueError(
                f"{path} visible keypoint {keypoint_index} must have normalized coordinates in [0, 1]"
            )


def load_symmetry_permutations(path: Path) -> tuple[tuple[int, ...], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Symmetry manifest must be an object: {path}")
    if payload.get("permutation_convention") != "permutation[output_index] = source_index":
        raise ValueError(f"Unexpected symmetry permutation convention in {path}")
    if payload.get("keypoint_count") != KEYPOINT_COUNT:
        raise ValueError(f"Symmetry manifest must declare keypoint_count={KEYPOINT_COUNT}: {path}")
    names = payload.get("keypoint_names")
    if not isinstance(names, list) or tuple(names) != ORDERED_KEYPOINT_NAMES:
        raise ValueError(
            f"Symmetry manifest keypoint order does not match the ordered cube convention: {path}"
        )
    raw_permutations = payload.get("permutations")
    if not isinstance(raw_permutations, list) or len(raw_permutations) != 24:
        raise ValueError(f"Symmetry manifest must contain 24 permutations: {path}")
    permutations: list[tuple[int, ...]] = []
    for permutation_index, raw_permutation in enumerate(raw_permutations):
        if not isinstance(raw_permutation, list) or len(raw_permutation) != KEYPOINT_COUNT:
            raise ValueError(f"Symmetry {permutation_index} must contain 26 indices")
        permutation = tuple(int(value) for value in raw_permutation)
        if set(permutation) != set(range(KEYPOINT_COUNT)):
            raise ValueError(f"Symmetry {permutation_index} is not a bijection")
        permutations.append(permutation)
    if len(set(permutations)) != 24:
        raise ValueError("Symmetry manifest contains duplicate permutations")
    if tuple(range(KEYPOINT_COUNT)) not in permutations:
        raise ValueError("Symmetry manifest is missing the identity permutation")
    return tuple(permutations)


def collect_pairs(source_dir: Path) -> tuple[list[tuple[Path, Path]], tuple[tuple[int, ...], ...]]:
    images_dir = source_dir / "images"
    pose_dir = source_dir / "pose"
    symmetry_path = source_dir / "ordered_keypoint_symmetries.json"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    if not pose_dir.is_dir():
        raise FileNotFoundError(f"Missing ordered pose directory: {pose_dir}")
    permutations = load_symmetry_permutations(symmetry_path)
    pairs: list[tuple[Path, Path]] = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        pose_path = pose_dir / f"{image_path.stem}.txt"
        if not pose_path.is_file():
            raise FileNotFoundError(f"Missing pose label for {image_path.name}: {pose_path}")
        validate_label_file(pose_path)
        pairs.append((image_path, pose_path))
    if not pairs:
        raise RuntimeError(f"No image/pose pairs found below {source_dir}")
    return pairs, permutations


def write_prepared_dataset(
    source_dir: Path,
    dataset_dir: Path,
    val_fraction: float,
    seed: int,
    force: bool,
) -> tuple[Path, tuple[tuple[int, ...], ...]]:
    pairs, permutations = collect_pairs(source_dir)
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")
    if dataset_dir.exists():
        if not force:
            raise FileExistsError(
                f"Prepared dataset already exists: {dataset_dir}; pass --force to replace it"
            )
        shutil.rmtree(dataset_dir)

    random.Random(seed).shuffle(pairs)
    val_count = max(1, round(len(pairs) * val_fraction)) if len(pairs) > 1 else 0
    splits = {"val": pairs[:val_count], "train": pairs[val_count:]}
    for split in splits:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest = []
    for split, split_pairs in splits.items():
        for index, (image_path, pose_path) in enumerate(split_pairs):
            target_stem = f"{split}_{index:06d}"
            target_image = dataset_dir / "images" / split / f"{target_stem}{image_path.suffix.lower()}"
            target_label = dataset_dir / "labels" / split / f"{target_stem}.txt"
            shutil.copy2(image_path, target_image)
            shutil.copy2(pose_path, target_label)
            manifest.append({
                "split": split,
                "source_image": str(image_path),
                "source_pose_label": str(pose_path),
                "image": str(target_image),
                "label": str(target_label),
            })

    symmetry_path = dataset_dir / "ordered_keypoint_symmetries.json"
    symmetry_path.write_text(
        json.dumps(
            {
                "format": "ordered_cube_keypoint_symmetry_v1",
                "keypoint_count": KEYPOINT_COUNT,
                "keypoint_names": list(ORDERED_KEYPOINT_NAMES),
                "permutation_convention": "permutation[output_index] = source_index",
                "permutations": [list(permutation) for permutation in permutations],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(
        "path: " + str(dataset_dir.resolve()) + "\n"
        "train: images/train\n"
        "val: images/val\n"
        f"kpt_shape: [{KEYPOINT_COUNT}, {KEYPOINT_DIMS}]\n"
        "kpt_symmetry_file: ordered_keypoint_symmetries.json\n"
        "names:\n"
        "  0: cube\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(splits['train'])} training and {len(splits['val'])} validation images")
    print(f"Prepared dataset: {data_yaml}")
    return data_yaml, permutations


def _symmetry_location_loss(
    prediction: Tensor,
    target: Tensor,
    keypoint_mask: Tensor,
    area: Tensor,
    sigmas: Tensor,
) -> Tensor:
    """Return one location-loss value per matched cube for one permutation."""
    squared_distance = (prediction[..., :2] - target[..., :2]).pow(2).sum(dim=-1)
    exponent = squared_distance / ((2.0 * sigmas).pow(2) * (area + 1e-9) * 2.0)
    visible_count = keypoint_mask.sum(dim=-1)
    factor = keypoint_mask.shape[-1] / (visible_count + 1e-9)
    return factor * ((1.0 - torch.exp(-exponent)) * keypoint_mask).mean(dim=-1)


class SymmetryAwarePoseLoss(v8PoseLoss):
    """Ultralytics pose loss with per-object minimum over 24 cube symmetries."""

    def __init__(self, model: PoseModel) -> None:
        super().__init__(model)
        raw_permutations = getattr(model, "ordered_keypoint_symmetry_permutations", None)
        if raw_permutations is None:
            raise RuntimeError("The pose model has no ordered cube symmetry permutations")
        self.ordered_keypoint_symmetry_permutations = torch.as_tensor(
            raw_permutations,
            dtype=torch.long,
            device=self.device,
        )
        if tuple(self.kpt_shape) != (KEYPOINT_COUNT, KEYPOINT_DIMS):
            raise ValueError(
                "SymmetryAwarePoseLoss requires kpt_shape=[26, 3], "
                f"got {self.kpt_shape}"
            )

    def calculate_keypoints_loss(
        self,
        masks: Tensor,
        target_gt_idx: Tensor,
        keypoints: Tensor,
        batch_idx: Tensor,
        stride_tensor: Tensor,
        target_bboxes: Tensor,
        pred_kpts: Tensor,
    ) -> tuple[Tensor, Tensor]:
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)
        if not masks.any():
            return torch.zeros((), device=pred_kpts.device), torch.zeros((), device=pred_kpts.device)

        target_bboxes = target_bboxes / stride_tensor
        gt_kpt = selected_keypoints[masks]
        pred_kpt = pred_kpts[masks]
        area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
        permutations = self.ordered_keypoint_symmetry_permutations
        candidate_targets = gt_kpt[:, permutations, :]
        candidate_masks = candidate_targets[..., 2] != 0
        candidate_prediction = pred_kpt.unsqueeze(1)
        candidate_area = area.unsqueeze(1)
        candidate_sigmas = self.keypoint_loss.sigmas.to(pred_kpt.device).view(1, 1, -1)
        location_losses = _symmetry_location_loss(
            candidate_prediction.expand(-1, permutations.shape[0], -1, -1),
            candidate_targets,
            candidate_masks,
            candidate_area,
            candidate_sigmas,
        )
        visibility_losses = F.binary_cross_entropy_with_logits(
            pred_kpt[:, None, :, 2].expand(-1, permutations.shape[0], -1),
            candidate_masks.float(),
            reduction="none",
        ).mean(dim=2)
        candidate_total = location_losses * self.hyp.pose + visibility_losses * self.hyp.kobj
        best_index = candidate_total.argmin(dim=1)
        row_index = torch.arange(gt_kpt.shape[0], device=gt_kpt.device)
        best_gt_kpt = candidate_targets[row_index, best_index]
        best_mask = candidate_masks[row_index, best_index]

        kpts_loss = self.keypoint_loss(pred_kpt, best_gt_kpt, best_mask, area)
        kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], best_mask.float())
        return kpts_loss, kpts_obj_loss


class SymmetryAwarePoseModel(PoseModel):
    """PoseModel that installs the ordered-cube symmetry-aware criterion."""

    ordered_keypoint_symmetry_permutations: tuple[tuple[int, ...], ...] | None = None

    def init_criterion(self):
        return SymmetryAwarePoseLoss(self)


class SymmetryAwarePoseTrainer(PoseTrainer):
    """PoseTrainer that loads the generated symmetry manifest into the model."""

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = SymmetryAwarePoseModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            data_kpt_shape=self.data["kpt_shape"],
            verbose=verbose,
        )
        # Ultralytics' loss reads the scalar gains (pose/kobj/etc.) through
        # attribute access on model.args.  A model built directly from YAML
        # does not receive that trainer namespace automatically.
        model.args = self.args
        dataset_root = Path(self.data["path"])
        symmetry_file = self.data.get("kpt_symmetry_file", "ordered_keypoint_symmetries.json")
        model.ordered_keypoint_symmetry_permutations = load_symmetry_permutations(
            dataset_root / str(symmetry_file)
        )
        if weights:
            model.load(weights)
        return model


def train(data_yaml: Path, args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        trainer=SymmetryAwarePoseTrainer,
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.runs_dir),
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        cache=args.cache,
        plots=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("data/synthetic_cube_dataset"))
    parser.add_argument("--workdir", type=Path, default=Path("runs/ordered_cube_pose_local"))
    parser.add_argument("--model", default="yolo11n-pose.pt")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="ordered_cube_yolo11n_pose")
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true", help="Replace an existing prepared dataset directory")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    data_yaml, _permutations = write_prepared_dataset(
        source_dir=args.source_dir,
        dataset_dir=args.workdir / "dataset",
        val_fraction=args.val_fraction,
        seed=args.seed,
        force=args.force,
    )
    if not args.prepare_only:
        args.runs_dir = args.workdir / "runs"
        train(data_yaml, args)


if __name__ == "__main__":
    main()
