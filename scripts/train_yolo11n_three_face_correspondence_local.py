"""Train a separate 37-point YOLO pose model for three-face correspondence.

The input dataset is produced by
``prepare_three_face_correspondence_dataset.py``::

    independent_keypoints_dataset_v3_three_face/
      images/*.png
      pose/*.txt
      correspondence_manifest.json

The model is a normal YOLO11 pose network with a 37-point head, but it uses a
custom criterion.  The criterion chooses the proper cube rotations that
preserve the subset of faces with face-exclusive visible evidence: four
in-plane rotations for one face, two face-preserving swaps for two adjacent
faces, and three cyclic corner rotations for all three faces.  Shared-edge
points do not count as evidence for both faces.  If only shared points remain,
the criterion uses a conservative union of candidate relabelings.  Blank
keypoints participate in the visibility term, but never contribute coordinate
error.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import sys
from collections import Counter
from itertools import permutations, product
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ultralytics.nn.tasks import PoseModel  # noqa: E402
from ultralytics.models.yolo.pose.train import PoseTrainer  # noqa: E402
from ultralytics.models.yolo.pose.val import PoseValidator  # noqa: E402
from ultralytics.utils.loss import v8PoseLoss  # noqa: E402


KEYPOINT_COUNT = 37
KEYPOINT_DIMS = 3
POSE_LABEL_FIELD_COUNT = 1 + 4 + KEYPOINT_COUNT * KEYPOINT_DIMS
BBOX_FIELD_COUNT = 4
KEYPOINT_VISIBILITY_VALUES = {0, 1, 2}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
CLASS_NAMES = ("cube",)
FACE_NAMES = ("U", "F", "R")
FACE_BITS = {face: 1 << index for index, face in enumerate(FACE_NAMES)}
FACE_NORMALS = {
    "U": (0, 0, 1),
    "F": (0, -1, 0),
    "R": (1, 0, 0),
    "D": (0, 0, -1),
    "B": (0, 1, 0),
    "L": (-1, 0, 0),
}
DEFAULT_SOURCE_DIR = Path("data/independent_keypoints_dataset_v3_three_face")
DEFAULT_EXTRA_NEGATIVE_DIR = Path("data/youtube_solve_frames_hd/cropped-negatives")
DEFAULT_WORKDIR = Path("runs/three_face_correspondence_pose_local")

# Rotation around the corner whose canonical metric coordinate is (+,+,+).
# It cycles U -> F -> R -> U and preserves the union of the three emitted
# face grids.  The second non-identity element is its square.
CORNER_ROTATION_120 = (
    (0, -1, 0),
    (0, 0, -1),
    (1, 0, 0),
)
IDENTITY_ROTATION = (
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
)


def label_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_label_file(path: Path) -> bool:
    lines = label_lines(path)
    if not lines:
        return False
    if len(lines) != 1:
        raise ValueError(f"{path} must contain one pose row, got {len(lines)}")
    fields = lines[0].split()
    if len(fields) != POSE_LABEL_FIELD_COUNT:
        raise ValueError(
            f"{path} is not a {KEYPOINT_COUNT}-point YOLO pose label; expected "
            f"{POSE_LABEL_FIELD_COUNT} fields, got {len(fields)}"
        )
    if int(float(fields[0])) != 0:
        raise ValueError(f"{path} must use cube class 0")
    bbox = [float(value) for value in fields[1 : 1 + BBOX_FIELD_COUNT]]
    if any(not 0.0 <= value <= 1.0 for value in bbox):
        raise ValueError(f"{path} has a bounding-box value outside [0, 1]")
    keypoints = [float(value) for value in fields[1 + BBOX_FIELD_COUNT :]]
    for point_index in range(KEYPOINT_COUNT):
        offset = point_index * KEYPOINT_DIMS
        x, y, visibility = keypoints[offset : offset + KEYPOINT_DIMS]
        if int(visibility) != visibility or int(visibility) not in KEYPOINT_VISIBILITY_VALUES:
            raise ValueError(f"{path} keypoint {point_index} has invalid visibility {visibility}")
        if int(visibility) != 0 and not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"{path} visible keypoint {point_index} is outside [0, 1]")
    return True


def _matrix_product(
    left: tuple[tuple[int, ...], ...],
    right: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(sum(left[row][middle] * right[middle][column] for middle in range(3)) for column in range(3))
        for row in range(3)
    )


def _matrix_transpose(matrix: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(matrix[column][row] for column in range(3)) for row in range(3))


def _apply_rotation(rotation: tuple[tuple[int, ...], ...], coordinate: tuple[int, int, int]) -> tuple[int, int, int]:
    centered = tuple(value - 1.5 for value in coordinate)
    rotated = tuple(
        sum(rotation[row][column] * centered[column] for column in range(3)) + 1.5
        for row in range(3)
    )
    rounded = tuple(round(value) for value in rotated)
    if any(value not in range(4) for value in rounded):
        raise ValueError(f"Rotation produced an invalid lattice coordinate: {rounded}")
    return rounded


def _permutation_parity(permutation: tuple[int, ...]) -> int:
    inversions = sum(
        permutation[left] > permutation[right]
        for left in range(3)
        for right in range(left + 1, 3)
    )
    return -1 if inversions % 2 else 1


def _proper_cube_rotations() -> tuple[tuple[tuple[int, ...], ...], ...]:
    rotations = []
    for axis_order in permutations(range(3)):
        parity = _permutation_parity(axis_order)
        for signs in product((-1, 1), repeat=3):
            if parity * signs[0] * signs[1] * signs[2] != 1:
                continue
            rotations.append(
                tuple(
                    tuple(signs[row] if column == axis_order[row] else 0 for column in range(3))
                    for row in range(3)
                )
            )
    return tuple(rotations)


def _rotate_vector(
    rotation: tuple[tuple[int, ...], ...], vector: tuple[int, int, int]
) -> tuple[int, int, int]:
    return tuple(
        sum(rotation[row][column] * vector[column] for column in range(3))
        for row in range(3)
    )


def _face_for_normal(normal: tuple[int, int, int]) -> str:
    for face, face_normal in FACE_NORMALS.items():
        if face_normal == normal:
            return face
    raise ValueError(f"Not a cube face normal: {normal}")


def _identity_permutation() -> tuple[int, ...]:
    return tuple(range(KEYPOINT_COUNT))


def load_symmetry_definition(
    path: Path,
) -> tuple[
    dict[int, tuple[tuple[int, ...], ...]],
    tuple[tuple[bool, ...], ...],
    tuple[tuple[bool, ...], ...],
]:
    """Load face-aware symmetry candidates and the slot-to-face membership masks."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("keypoint_count") != KEYPOINT_COUNT:
        raise ValueError(f"Expected {KEYPOINT_COUNT} keypoints in {path}")
    keypoints = payload.get("keypoints")
    if not isinstance(keypoints, list) or len(keypoints) != KEYPOINT_COUNT:
        raise ValueError(f"Manifest has an invalid keypoint list: {path}")
    coordinates = []
    for index, keypoint in enumerate(keypoints):
        coordinate = keypoint.get("coordinate") if isinstance(keypoint, dict) else None
        if not isinstance(coordinate, list) or len(coordinate) != 3:
            raise ValueError(f"Manifest keypoint {index} has no 3-D lattice coordinate")
        coordinates.append(tuple(int(value) for value in coordinate))
    index_by_coordinate = {coordinate: index for index, coordinate in enumerate(coordinates)}
    if len(index_by_coordinate) != KEYPOINT_COUNT:
        raise ValueError(f"Manifest contains duplicate lattice coordinates: {path}")

    face_slot_masks = tuple(
        tuple(
            isinstance(keypoint, dict)
            and face in (keypoint.get("faces") or [])
            for keypoint in keypoints
        )
        for face in FACE_NAMES
    )
    exclusive_face_slot_masks = tuple(
        tuple(
            isinstance(keypoint, dict)
            and tuple(keypoint.get("faces") or ()) == (face,)
            for keypoint in keypoints
        )
        for face in FACE_NAMES
    )
    rotations = _proper_cube_rotations()
    permutations_by_face_mask: dict[int, tuple[tuple[int, ...], ...]] = {}
    for face_mask in range(1 << len(FACE_NAMES)):
        selected_faces = {face for face, bit in FACE_BITS.items() if face_mask & bit}
        if not selected_faces:
            permutations_by_face_mask[face_mask] = (_identity_permutation(),)
            continue
        stable_rotations = []
        for rotation in rotations:
            mapped_faces = {
                _face_for_normal(_rotate_vector(rotation, FACE_NORMALS[face]))
                for face in selected_faces
            }
            if mapped_faces == selected_faces:
                stable_rotations.append(rotation)
        active_indices = {
            index
            for index, mask in enumerate(zip(*face_slot_masks, strict=True))
            if any(mask[face_index] for face_index, face in enumerate(FACE_NAMES) if face_mask & FACE_BITS[face])
        }
        candidates = []
        for rotation in stable_rotations:
            inverse = _matrix_transpose(rotation)
            permutation = list(_identity_permutation())
            for output_index in active_indices:
                source_coordinate = _apply_rotation(inverse, coordinates[output_index])
                source_index = index_by_coordinate.get(source_coordinate)
                if source_index is None or source_index not in active_indices:
                    raise ValueError(
                        f"Face-mask symmetry {face_mask} leaves the 37-point set: {path}"
                    )
                permutation[output_index] = source_index
            candidates.append(tuple(permutation))
        unique_candidates = tuple(dict.fromkeys(candidates))
        if not unique_candidates or _identity_permutation() not in unique_candidates:
            raise ValueError(f"Face-mask symmetry {face_mask} is missing identity: {path}")
        identity = _identity_permutation()
        permutations_by_face_mask[face_mask] = (identity,) + tuple(
            permutation for permutation in unique_candidates if permutation != identity
        )
    ambiguous_candidates = tuple(
        dict.fromkeys(
            permutation
            for face_mask in range(1, 1 << len(FACE_NAMES))
            for permutation in permutations_by_face_mask[face_mask]
        )
    )
    permutations_by_face_mask[0] = ambiguous_candidates
    return permutations_by_face_mask, face_slot_masks, exclusive_face_slot_masks


def load_symmetry_permutations(path: Path) -> dict[int, tuple[tuple[int, ...], ...]]:
    """Compatibility helper returning the candidates keyed by visible-face mask."""
    permutations_by_face_mask, _face_slot_masks, _exclusive_face_slot_masks = load_symmetry_definition(path)
    return permutations_by_face_mask


def collect_pairs(
    source_dir: Path,
    extra_negative_dir: Path | None,
) -> tuple[
    list[tuple[Path, Path | None]],
    dict[int, tuple[tuple[int, ...], ...]],
    tuple[tuple[bool, ...], ...],
    tuple[tuple[bool, ...], ...],
]:
    images_dir = source_dir / "images"
    pose_dir = source_dir / "pose"
    manifest_path = source_dir / "correspondence_manifest.json"
    if not images_dir.is_dir() or not pose_dir.is_dir():
        raise FileNotFoundError(f"Expected images/ and pose/ below {source_dir}")
    permutations, face_slot_masks, exclusive_face_slot_masks = load_symmetry_definition(manifest_path)
    pairs: list[tuple[Path, Path | None]] = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = pose_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing pose label for {image_path.name}: {label_path}")
        validate_label_file(label_path)
        pairs.append((image_path, label_path))
    if extra_negative_dir is not None:
        if not extra_negative_dir.is_dir():
            raise FileNotFoundError(f"Missing extra-negative image directory: {extra_negative_dir}")
        pairs.extend(
            (path, None)
            for path in sorted(extra_negative_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    if not pairs:
        raise RuntimeError(f"No image/label pairs found below {source_dir}")
    return pairs, permutations, face_slot_masks, exclusive_face_slot_masks


def write_prepared_dataset(
    source_dir: Path,
    dataset_dir: Path,
    val_fraction: float,
    seed: int,
    extra_negative_dir: Path | None,
    force: bool,
) -> Path:
    pairs, permutations_by_face_mask, face_slot_masks, exclusive_face_slot_masks = collect_pairs(
        source_dir, extra_negative_dir
    )
    random.Random(seed).shuffle(pairs)
    val_count = max(1, round(len(pairs) * val_fraction)) if len(pairs) > 1 else 0
    splits = {"val": pairs[:val_count], "train": pairs[val_count:]}
    if dataset_dir.exists():
        if not force:
            raise FileExistsError(f"Output exists: {dataset_dir}; pass --force to replace it")
        shutil.rmtree(dataset_dir)
    for split in splits:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for split, split_pairs in splits.items():
        label_counts: Counter[int] = Counter()
        empty_count = 0
        for index, (image_path, label_path) in enumerate(split_pairs):
            lines = label_lines(label_path) if label_path is not None else []
            if label_path is not None and lines:
                validate_label_file(label_path)
            if not lines:
                empty_count += 1
            target_stem = f"{split}_{index:06d}"
            target_image = dataset_dir / "images" / split / f"{target_stem}{image_path.suffix.lower()}"
            target_label = dataset_dir / "labels" / split / f"{target_stem}.txt"
            shutil.copy2(image_path, target_image)
            target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            label_counts[len(lines)] += 1
            manifest.append({
                "split": split,
                "source_image": str(image_path),
                "source_label": str(label_path) if label_path is not None else None,
                "image": str(target_image),
                "label": str(target_label),
                "empty_label": not lines,
            })
        summary[split] = {
            "images": len(split_pairs),
            "empty_labels": empty_count,
            "nonempty_labels": len(split_pairs) - empty_count,
            "label_count_distribution": dict(sorted(label_counts.items())),
        }

    (dataset_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (dataset_dir / "three_face_symmetries.json").write_text(
        json.dumps({
            "format": "three_face_correspondence_symmetry_v2",
            "keypoint_count": KEYPOINT_COUNT,
            "permutation_convention": "permutation[output_index] = source_index",
            "symmetry_scope": "proper_rotations_preserving_the_observed_face_subset",
            "permutations_by_face_mask": {
                str(face_mask): [list(permutation) for permutation in permutations]
                for face_mask, permutations in permutations_by_face_mask.items()
            },
            "face_bit_order": list(FACE_NAMES),
            "face_slot_masks": {
                face: list(mask) for face, mask in zip(FACE_NAMES, face_slot_masks, strict=True)
            },
            "exclusive_face_slot_masks": {
                face: list(mask)
                for face, mask in zip(FACE_NAMES, exclusive_face_slot_masks, strict=True)
            },
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(
        "path: " + str(dataset_dir.resolve()) + "\n"
        "train: images/train\n"
        "val: images/val\n"
        f"kpt_shape: [{KEYPOINT_COUNT}, {KEYPOINT_DIMS}]\n"
        "kpt_symmetry_file: three_face_symmetries.json\n"
        "names:\n"
        "  0: cube\n",
        encoding="utf-8",
    )
    (dataset_dir / "preparation_summary.json").write_text(
        json.dumps({"source": str(source_dir), "summary": summary}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Prepared dataset: {data_yaml}")
    return data_yaml


def _symmetry_candidate_losses(
    prediction: Tensor,
    target: Tensor,
    permutations: Tensor,
    *,
    pose_weight: float,
    kobj_weight: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return location loss, visibility loss, and detached winner per sample."""
    candidate_targets = target[:, permutations, :]
    candidate_masks = candidate_targets[..., 2] != 0
    coordinate_error = F.smooth_l1_loss(
        prediction[:, None, :, :2].expand(-1, permutations.shape[0], -1, -1),
        candidate_targets[:, :, :, :2],
        reduction="none",
        beta=1.0,
    ).sum(dim=-1)
    visible_count = candidate_masks.sum(dim=-1)
    location_losses = (coordinate_error * candidate_masks).sum(dim=-1) / visible_count.clamp_min(1)
    visibility_losses = F.binary_cross_entropy_with_logits(
        prediction[:, None, :, 2].expand(-1, permutations.shape[0], -1),
        candidate_masks.float(),
        reduction="none",
    ).mean(dim=2)
    candidate_total = location_losses * pose_weight + visibility_losses * kobj_weight
    winner = candidate_total.detach().argmin(dim=1)
    selected_location = location_losses.gather(1, winner[:, None]).squeeze(1).mean()
    selected_visibility = visibility_losses.gather(1, winner[:, None]).squeeze(1).mean()
    return selected_location, selected_visibility, winner


def _visible_face_masks(target_keypoints: Tensor, exclusive_face_slot_masks: Tensor) -> Tensor:
    """Infer face support without treating a shared seam as two faces."""
    visible_slots = target_keypoints[..., 2] != 0
    observed_faces = (
        visible_slots[:, None, :] & exclusive_face_slot_masks[None, :, :]
    ).any(dim=2)
    face_bits = torch.as_tensor(
        [FACE_BITS[face] for face in FACE_NAMES],
        dtype=torch.long,
        device=target_keypoints.device,
    )
    return (observed_faces.to(torch.long) * face_bits[None, :]).sum(dim=1)


class ThreeFaceCorrespondencePoseLoss(v8PoseLoss):
    """YOLO pose loss with hard minimum-cost routing over face-aware symmetries."""

    def __init__(self, model: PoseModel) -> None:
        super().__init__(model)
        raw_permutations_by_face_mask = getattr(
            model, "three_face_symmetry_permutations_by_face_mask", None
        )
        raw_exclusive_face_slot_masks = getattr(model, "three_face_exclusive_face_slot_masks", None)
        if raw_permutations_by_face_mask is None or raw_exclusive_face_slot_masks is None:
            raise RuntimeError("The pose model has no face-aware symmetry definition")
        self.symmetry_permutations_by_face_mask = {
            int(face_mask): torch.as_tensor(
                permutations_for_mask, dtype=torch.long, device=self.device
            )
            for face_mask, permutations_for_mask in raw_permutations_by_face_mask.items()
        }
        self.exclusive_face_slot_masks = torch.as_tensor(
            raw_exclusive_face_slot_masks, dtype=torch.bool, device=self.device
        )
        if tuple(self.exclusive_face_slot_masks.shape) != (len(FACE_NAMES), KEYPOINT_COUNT):
            raise ValueError(
                f"Expected face slot masks with shape [{len(FACE_NAMES)}, {KEYPOINT_COUNT}], "
                f"got {tuple(self.exclusive_face_slot_masks.shape)}"
            )
        if tuple(self.kpt_shape) != (KEYPOINT_COUNT, KEYPOINT_DIMS):
            raise ValueError(
                f"ThreeFaceCorrespondencePoseLoss requires kpt_shape=[{KEYPOINT_COUNT}, {KEYPOINT_DIMS}], "
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
        del target_bboxes
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)
        if not masks.any():
            zero = torch.zeros((), device=pred_kpts.device)
            return zero, zero
        target_keypoints = selected_keypoints[masks]
        prediction_keypoints = pred_kpts[masks]
        face_masks = _visible_face_masks(target_keypoints, self.exclusive_face_slot_masks)

        location_total = torch.zeros((), device=pred_kpts.device)
        visibility_total = torch.zeros((), device=pred_kpts.device)
        for face_mask in torch.unique(face_masks).tolist():
            sample_indices = face_masks == int(face_mask)
            permutations = self.symmetry_permutations_by_face_mask[int(face_mask)]
            location_loss, visibility_loss, _winner = _symmetry_candidate_losses(
                prediction_keypoints[sample_indices],
                target_keypoints[sample_indices],
                permutations,
                pose_weight=float(self.hyp.pose),
                kobj_weight=float(self.hyp.kobj),
            )
            sample_weight = sample_indices.float().sum()
            location_total = location_total + location_loss * sample_weight
            visibility_total = visibility_total + visibility_loss * sample_weight
        sample_count = face_masks.shape[0]
        return location_total / sample_count, visibility_total / sample_count


class ThreeFaceCorrespondencePoseModel(PoseModel):
    """Standard YOLO pose architecture with the dedicated 37-point criterion."""

    three_face_symmetry_permutations_by_face_mask: dict[int, tuple[tuple[int, ...], ...]] | None = None
    three_face_exclusive_face_slot_masks: tuple[tuple[bool, ...], ...] | None = None

    def init_criterion(self):
        return ThreeFaceCorrespondencePoseLoss(self)


class ThreeFaceCorrespondencePoseTrainer(PoseTrainer):
    """Trainer that installs the 37-point model and loads its symmetry set."""

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = ThreeFaceCorrespondencePoseModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            data_kpt_shape=self.data["kpt_shape"],
            verbose=verbose,
        )
        model.args = self.args
        dataset_root = Path(self.data["path"])
        symmetry_file = self.data.get("kpt_symmetry_file", "three_face_symmetries.json")
        symmetry_path = dataset_root / str(symmetry_file)
        payload = json.loads(symmetry_path.read_text(encoding="utf-8"))
        raw_permutations = payload.get("permutations_by_face_mask")
        raw_exclusive_face_slot_masks = payload.get("exclusive_face_slot_masks")
        if not isinstance(raw_permutations, dict) or not isinstance(raw_exclusive_face_slot_masks, dict):
            raise ValueError(f"Missing face-aware symmetry definition: {symmetry_path}")
        model.three_face_symmetry_permutations_by_face_mask = {
            int(face_mask): tuple(
                tuple(int(value) for value in permutation) for permutation in permutations_for_mask
            )
            for face_mask, permutations_for_mask in raw_permutations.items()
        }
        model.three_face_exclusive_face_slot_masks = tuple(
            tuple(bool(value) for value in raw_exclusive_face_slot_masks[face])
            for face in FACE_NAMES
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"
        return PoseValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy.copy(self.args),
            _callbacks=self.callbacks,
        )


def train(data_yaml: Path, runs_dir: Path, args: argparse.Namespace) -> Path:
    from ultralytics import YOLO

    model_path = args.resume_checkpoint or args.model
    if args.resume_checkpoint is not None and not args.resume_checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {args.resume_checkpoint}")
    model = YOLO(str(model_path))
    model.train(
        trainer=ThreeFaceCorrespondencePoseTrainer,
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(runs_dir),
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        cache=args.cache,
        mosaic=args.mosaic,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        fliplr=args.fliplr,
        plots=True,
        resume=str(args.resume_checkpoint) if args.resume_checkpoint is not None else False,
    )
    trainer = getattr(model, "trainer", None)
    best_model_path = getattr(trainer, "best", None)
    if best_model_path is None:
        raise RuntimeError("Training completed without exposing a best-model path")
    best_model_path = Path(best_model_path).resolve()
    if not best_model_path.is_file():
        raise FileNotFoundError(f"Training completed but best model was not found: {best_model_path}")
    print(f"Best model: {best_model_path}")
    return best_model_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--model", default="yolo11n-pose.pt")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--extra-negative-dir", type=Path, default=DEFAULT_EXTRA_NEGATIVE_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--mosaic", type=float, default=0.0)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="three_face_correspondence_yolo11n_pose")
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fliplr", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    data_yaml = write_prepared_dataset(
        source_dir=args.source_dir,
        dataset_dir=args.workdir / "dataset",
        val_fraction=args.val_fraction,
        seed=args.seed,
        extra_negative_dir=args.extra_negative_dir,
        force=args.force,
    )
    if not args.prepare_only:
        train(data_yaml, args.workdir / "runs", args)


if __name__ == "__main__":
    main()
