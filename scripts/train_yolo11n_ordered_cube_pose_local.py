"""Train YOLO pose on the ordered 26-point cube labels.

The Blender generator writes this source layout::

    synthetic_cube_dataset/
      images/*.png
      pose/*.txt
      ordered_keypoint_symmetries.json

Each pose row is standard YOLO pose format with one cube object and 26
keypoints (x, y, visibility).  The trainer prepares ordinary Ultralytics
``images/{train,val}`` and ``labels/{train,val}`` folders, then replaces the
stock point-by-point pose loss with a symmetry-aware loss. Every sample uses
the same 24 proper cube rotations, so orientation remains irrelevant without
discarding the layer-aware coordinates.
The permutation convention is the same one written by Blender generator:

    permutation[output_index] = source_index

The prediction remains a normal 26-point YOLO pose output.  Symmetry is only
used while comparing it to the target, so downstream keypoint indices remain
stable for geometric post-processing.
"""

from __future__ import annotations

import argparse
import copy
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
from ultralytics.models.yolo.pose.val import PoseValidator  # noqa: E402
from ultralytics.utils.loss import v8PoseLoss  # noqa: E402
from ultralytics.utils.ops import xyxy2xywh  # noqa: E402


KEYPOINT_COUNT = 26
KEYPOINT_DIMS = 3
POSE_LABEL_FIELD_COUNT = 1 + 4 + KEYPOINT_COUNT * KEYPOINT_DIMS
KEYPOINT_VISIBILITY_VALUES = {0, 1, 2}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
CLASS_NAMES = ("cube",)
DEFAULT_EXTRA_NEGATIVE_DIR = Path("data/youtube_solve_frames_hd/cropped-negatives")
ORDERED_KEYPOINT_NAMES = (
    "UBL", "UBR", "UFR", "UFL",
    "DBL", "DBR", "DFR", "DFL",
    "UB", "UR", "UF", "UL",
    "DB", "DR", "DF", "DL",
    "BR", "BL", "FR", "FL",
    "U", "D", "F", "B", "R", "L",
)
# All samples use the same 24 proper cube rotations. Layer-state metadata is retained in
# the prepared dataset for inspection, but it must not change the loss rule.
TURN_LAYER_MODES = frozenset(("mid_turn", "corner_cut"))
ORDERED_CUBE_FACES = frozenset(("U", "D", "F", "B", "R", "L"))
TRAIN_LAYER_MODE_CHOICES = ("all", "stationary", "turning")
TRAINING_IDENTITY_ONLY = False
SYMMETRY_SCOPE = "full_cube_rotations"
SYMMETRY_COUNT = 24

# The nine cubies on a face are the points that move when that face layer is
# turned.  These are used only for diagnostics; the training loss remains
# agnostic to the layer-state metadata.
FACE_KEYPOINT_NAMES = {
    "U": ("UBL", "UBR", "UFR", "UFL", "UB", "UR", "UF", "UL", "U"),
    "D": ("DBL", "DBR", "DFR", "DFL", "DB", "DR", "DF", "DL", "D"),
    "F": ("UFR", "UFL", "DFR", "DFL", "UF", "DF", "FR", "FL", "F"),
    "B": ("UBL", "UBR", "DBL", "DBR", "UB", "DB", "BR", "BL", "B"),
    "R": ("UBR", "UFR", "DBR", "DFR", "UR", "DR", "BR", "FR", "R"),
    "L": ("UBL", "UFL", "DBL", "DFL", "UL", "DL", "BL", "FL", "L"),
}
KEYPOINT_NAME_TO_INDEX = {name: index for index, name in enumerate(ORDERED_KEYPOINT_NAMES)}


def label_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_label_file(path: Path) -> bool:
    lines = label_lines(path)
    # Keep empty-label support for older datasets or deliberately excluded
    # samples. Current generator versions emit layer-aware labels for turns.
    if not lines:
        return False
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
    return True


def load_symmetry_permutations(path: Path) -> tuple[tuple[int, ...], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Symmetry manifest must be an object: {path}")
    if payload.get("permutation_convention") != "permutation[output_index] = source_index":
        raise ValueError(f"Unexpected symmetry permutation convention in {path}")
    if payload.get("keypoint_count") != KEYPOINT_COUNT:
        raise ValueError(f"Symmetry manifest must declare keypoint_count={KEYPOINT_COUNT}: {path}")
    declared_scope = payload.get("symmetry_scope")
    if declared_scope is not None and declared_scope != SYMMETRY_SCOPE:
        raise ValueError(
            f"Symmetry manifest must use {SYMMETRY_SCOPE!r}, got {declared_scope!r}: {path}"
        )
    declared_count = payload.get("symmetry_count")
    if declared_count is not None and declared_count != SYMMETRY_COUNT:
        raise ValueError(
            f"Symmetry manifest must declare symmetry_count={SYMMETRY_COUNT}: {path}"
        )
    names = payload.get("keypoint_names")
    if not isinstance(names, list) or tuple(names) != ORDERED_KEYPOINT_NAMES:
        raise ValueError(
            f"Symmetry manifest keypoint order does not match the ordered cube convention: {path}"
        )
    raw_permutations = payload.get("permutations")
    if not isinstance(raw_permutations, list):
        raise ValueError(f"Symmetry manifest permutations must be a list: {path}")
    permutations: list[tuple[int, ...]] = []
    for permutation_index, raw_permutation in enumerate(raw_permutations):
        if not isinstance(raw_permutation, list) or len(raw_permutation) != KEYPOINT_COUNT:
            raise ValueError(f"Symmetry {permutation_index} must contain 26 indices")
        permutation = tuple(int(value) for value in raw_permutation)
        if set(permutation) != set(range(KEYPOINT_COUNT)):
            raise ValueError(f"Symmetry {permutation_index} is not a bijection")
        permutations.append(permutation)
    if len(permutations) != SYMMETRY_COUNT:
        raise ValueError(f"Symmetry manifest must contain {SYMMETRY_COUNT} permutations: {path}")
    if len(set(permutations)) != SYMMETRY_COUNT:
        raise ValueError("Symmetry manifest contains duplicate permutations")
    if tuple(range(KEYPOINT_COUNT)) not in permutations:
        raise ValueError("Symmetry manifest is missing the identity permutation")
    return tuple(permutations)


def collect_pairs(
    source_dir: Path,
    extra_negative_dir: Path | None = None,
) -> tuple[list[tuple[Path, Path | None]], tuple[tuple[int, ...], ...]]:
    images_dir = source_dir / "images"
    pose_dir = source_dir / "pose"
    symmetry_path = source_dir / "ordered_keypoint_symmetries.json"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    if not pose_dir.is_dir():
        raise FileNotFoundError(f"Missing ordered pose directory: {pose_dir}")
    permutations = load_symmetry_permutations(symmetry_path)
    pairs: list[tuple[Path, Path | None]] = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        pose_path = pose_dir / f"{image_path.stem}.txt"
        if not pose_path.is_file():
            raise FileNotFoundError(f"Missing pose label for {image_path.name}: {pose_path}")
        if validate_label_file(pose_path):
            pairs.append((image_path, pose_path))
    if not pairs:
        raise RuntimeError(f"No image/pose pairs found below {source_dir}")

    if extra_negative_dir is not None:
        if not extra_negative_dir.is_dir():
            raise FileNotFoundError(f"Missing extra-negative image directory: {extra_negative_dir}")
        negative_paths = sorted(
            path
            for path in extra_negative_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        pairs.extend((image_path, None) for image_path in negative_paths)
    return pairs, permutations


def load_source_layer_states(source_dir: Path) -> dict[str, dict[str, object]]:
    """Load Blender layer-state metadata keyed by source image stem."""
    metadata_path = source_dir / "metadata.json"
    if not metadata_path.is_file():
        return {}
    records = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a metadata list in {metadata_path}")
    states: dict[str, dict[str, object]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("image"), str):
            continue
        layer_state = record.get("layer_state")
        if not isinstance(layer_state, dict):
            layer_state = {"mode": "none"}
        mode = str(layer_state.get("mode", "none"))
        face = layer_state.get("face")
        if mode in TURN_LAYER_MODES and face not in ORDERED_CUBE_FACES:
            raise ValueError(
                f"Layer-turn metadata for {record['image']} has invalid face {face!r}"
            )
        adjacent_face = layer_state.get("adjacent_face")
        if mode == "corner_cut" and adjacent_face not in ORDERED_CUBE_FACES:
            raise ValueError(
                f"Corner-cut metadata for {record['image']} has invalid adjacent face "
                f"{adjacent_face!r}"
            )
        states[Path(record["image"]).stem] = {
            "mode": mode,
            "face": face,
            "angle_degrees": layer_state.get("angle_degrees", 0.0),
            "adjacent_face": adjacent_face,
            "adjacent_angle_degrees": layer_state.get("adjacent_angle_degrees", 0.0),
        }
    return states


def load_prepared_layer_states(data: dict[str, object]) -> dict[str, dict[str, object]]:
    """Load layer states keyed by prepared image stem for dataset batches."""
    dataset_root = Path(str(data["path"]))
    state_file = data.get("kpt_layer_state_file", "layer_state_manifest.json")
    state_path = dataset_root / str(state_file)
    if not state_path.is_file():
        return {}
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a prepared layer-state mapping in {state_path}")
    return payload


def misaligned_source_mask(layer_state: dict[str, object] | None) -> Tensor:
    """Return the source-slot mask for points on the affected layer(s)."""
    mask = torch.zeros(KEYPOINT_COUNT, dtype=torch.bool)
    if not layer_state or layer_state.get("mode") not in TURN_LAYER_MODES:
        return mask

    faces = [layer_state.get("face")]
    if layer_state.get("mode") == "corner_cut":
        faces.append(layer_state.get("adjacent_face"))
    for face in faces:
        for name in FACE_KEYPOINT_NAMES.get(str(face), ()):
            mask[KEYPOINT_NAME_TO_INDEX[name]] = True
    return mask


def write_prepared_dataset(
    source_dir: Path,
    dataset_dir: Path,
    val_fraction: float,
    seed: int,
    force: bool,
    extra_negative_dir: Path | None = None,
    overfit_images: int | None = None,
    train_layer_mode: str = "all",
) -> tuple[Path, tuple[tuple[int, ...], ...]]:
    if train_layer_mode not in TRAIN_LAYER_MODE_CHOICES:
        raise ValueError(
            f"train_layer_mode must be one of {TRAIN_LAYER_MODE_CHOICES}, got {train_layer_mode!r}"
        )
    pairs, permutations = collect_pairs(
        source_dir,
        extra_negative_dir=extra_negative_dir,
    )
    source_layer_states = load_source_layer_states(source_dir)
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")
    if dataset_dir.exists():
        if not force:
            raise FileExistsError(
                f"Prepared dataset already exists: {dataset_dir}; pass --force to replace it"
            )
        shutil.rmtree(dataset_dir)

    random.Random(seed).shuffle(pairs)
    if overfit_images is not None:
        if overfit_images < 1:
            raise ValueError(f"overfit_images must be positive, got {overfit_images}")
        if overfit_images > len(pairs):
            raise ValueError(
                f"overfit_images={overfit_images} exceeds the available pair count {len(pairs)}"
            )
        # Deliberately evaluate on the same tiny set.  This is a diagnostic:
        # if the loss/model wiring is healthy, training and validation should
        # be able to drive this memorization test down despite augmentation.
        selected_pairs = pairs[:overfit_images]
        splits = {"val": selected_pairs, "train": selected_pairs}
    else:
        val_count = max(1, round(len(pairs) * val_fraction)) if len(pairs) > 1 else 0
        splits = {"val": pairs[:val_count], "train": pairs[val_count:]}
    def include_in_filtered_training_set(pair: tuple[Path, Path | None]) -> bool:
        image_path, pose_path = pair
        # Keep cube-free negatives in every training mode. They have no layer
        # state and remain useful for the detector/objectness task.
        if pose_path is None or train_layer_mode == "all":
            return True
        state = source_layer_states.get(image_path.stem, {"mode": "none"})
        source_mode = str(state.get("mode", "none"))
        if train_layer_mode == "stationary":
            return source_mode == "none"
        # Keep mid-turn and corner-cut examples together in the turning set.
        return source_mode in TURN_LAYER_MODES

    if train_layer_mode != "all":
        filtered_train = [
            pair for pair in splits["train"] if include_in_filtered_training_set(pair)
        ]
        if not filtered_train:
            raise RuntimeError(
                f"Training layer mode {train_layer_mode!r} produced no training images"
            )
        if overfit_images is not None:
            # Overfit mode intentionally uses identical train/validation data;
            # apply the filter to both sides in that special case.
            splits["train"] = filtered_train
            splits["val"] = filtered_train
        else:
            splits["train"] = filtered_train

    prepared_layer_states: dict[str, dict[str, object]] = {}
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
            if pose_path is None:
                target_label.write_text("", encoding="utf-8")
            else:
                shutil.copy2(pose_path, target_label)
            source_state = source_layer_states.get(
                image_path.stem,
                {"mode": "none", "face": None},
            )
            prepared_layer_states[target_stem] = source_state
            manifest.append({
                "split": split,
                "source_image": str(image_path),
                "source_pose_label": str(pose_path) if pose_path is not None else None,
                "image": str(target_image),
                "label": str(target_label),
                "negative": pose_path is None,
                "layer_state": source_state,
            })

    symmetry_path = dataset_dir / "ordered_keypoint_symmetries.json"
    symmetry_path.write_text(
        json.dumps(
            {
                "format": "ordered_cube_keypoint_symmetry_v1",
                "keypoint_count": KEYPOINT_COUNT,
                "keypoint_names": list(ORDERED_KEYPOINT_NAMES),
                "permutation_convention": "permutation[output_index] = source_index",
                "symmetry_scope": SYMMETRY_SCOPE,
                "symmetry_count": SYMMETRY_COUNT,
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
    (dataset_dir / "layer_state_manifest.json").write_text(
        json.dumps(prepared_layer_states, indent=2) + "\n",
        encoding="utf-8",
    )
    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(
        "path: " + str(dataset_dir.resolve()) + "\n"
        "train: images/train\n"
        "val: images/val\n"
        f"ordered_train_layer_mode: {train_layer_mode}\n"
        f"kpt_shape: [{KEYPOINT_COUNT}, {KEYPOINT_DIMS}]\n"
        "kpt_symmetry_file: ordered_keypoint_symmetries.json\n"
        "kpt_layer_state_file: layer_state_manifest.json\n"
        "names:\n"
        "  0: cube\n",
        encoding="utf-8",
    )
    negative_count = sum(pose_path is None for _image_path, pose_path in pairs)
    print(
        f"Prepared {len(splits['train'])} training and {len(splits['val'])} validation images "
        f"({negative_count} extra negatives; train layer mode: {train_layer_mode})"
    )
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
    del area, sigmas
    # The stock OKS form, 1-exp(-e), is effectively flat when a newly
    # expanded 26-point head starts several stride units from the targets.
    # Smooth-L1 keeps a useful gradient during memorization and early training,
    # while remaining robust to an occasional mislabeled/occluded point.
    coordinate_error = F.smooth_l1_loss(
        prediction[..., :2],
        target[..., :2],
        reduction="none",
        beta=1.0,
    ).sum(dim=-1)
    visible_count = keypoint_mask.sum(dim=-1)
    return (coordinate_error * keypoint_mask).sum(dim=-1) / (visible_count + 1e-9)


class SymmetryAwarePoseLoss(v8PoseLoss):
    """Pose loss with smooth routing over the fixed cube symmetry set.

    The target coordinates are never geometrically transformed. Each symmetry
    only reassigns the rendered ``(x, y, visibility)`` triplets to output
    slots. This is important for partial turns: the moved-layer locations
    remain exactly where Blender rendered them, while the loss still permits
    any globally equivalent cube labeling.
    """

    def __init__(self, model: PoseModel) -> None:
        super().__init__(model)
        raw_permutations = getattr(model, "ordered_keypoint_symmetry_permutations", None)
        if raw_permutations is None:
            raise RuntimeError("The pose model has no ordered cube symmetry permutations")
        self.identity_only = bool(getattr(model, "identity_only", False))
        if self.identity_only:
            raw_permutations = (tuple(range(KEYPOINT_COUNT)),)
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

        # A soft average over 24 candidates is harmful at initialization:
        # similar candidate costs average the target into a symmetric blur and
        # produce almost-zero useful gradient for any individual keypoint.
        # Select the best symmetry using detached costs, then backpropagate
        # only through that candidate. All 24 cube rotations remain eligible
        # for every sample; this is a hard minimum, not an identity-only rule.
        winner = candidate_total.detach().argmin(dim=1)
        kpts_loss = location_losses.gather(1, winner[:, None]).squeeze(1).mean()
        kpts_obj_loss = visibility_losses.gather(1, winner[:, None]).squeeze(1).mean()
        return kpts_loss, kpts_obj_loss


class SymmetryAwarePoseModel(PoseModel):
    """PoseModel that installs the ordered-cube symmetry-aware criterion."""

    ordered_keypoint_symmetry_permutations: tuple[tuple[int, ...], ...] | None = None

    def init_criterion(self):
        return SymmetryAwarePoseLoss(self)


class MisalignmentTrackingPoseValidator(PoseValidator):
    """Pose validator that reports stationary and turned-layer point errors.

    The diagnostic uses the same hard symmetry selection principle as the
    training criterion: choose the candidate permutation with the smallest
    visible-point coordinate error, then split that winning candidate's
    errors according to the source keypoint's layer membership.  Errors are
    reported in pixels on the validation image, which makes a small turn
    failure visible even when the total 26-point average looks acceptable.
    """

    def init_metrics(self, model: torch.nn.Module) -> None:
        super().init_metrics(model)
        self._diagnostic_error_sums = {"stationary": 0.0, "misaligned": 0.0}
        self._diagnostic_point_counts = {"stationary": 0, "misaligned": 0}
        self._diagnostic_images_with_targets = 0
        self._diagnostic_images_with_predictions = 0
        self._diagnostic_images_without_predictions = 0

        dataset_root = Path(self.data["path"])
        state_file = self.data.get("kpt_layer_state_file", "layer_state_manifest.json")
        state_path = dataset_root / str(state_file)
        if state_path.is_file():
            self._diagnostic_layer_states = json.loads(state_path.read_text(encoding="utf-8"))
        else:
            self._diagnostic_layer_states = {}

        raw_permutations = getattr(model, "ordered_keypoint_symmetry_permutations", None)
        if raw_permutations is None:
            raw_permutations = (tuple(range(KEYPOINT_COUNT)),)
        self._diagnostic_permutations = torch.as_tensor(raw_permutations, dtype=torch.long)

    @staticmethod
    def _misaligned_source_mask(layer_state: dict[str, object] | None) -> Tensor:
        return misaligned_source_mask(layer_state)

    def _track_keypoint_errors(self, preds: list[dict[str, Tensor]], batch: dict[str, object]) -> None:
        for sample_index, pred in enumerate(preds):
            prepared_batch = self._prepare_batch(sample_index, batch)
            target_keypoints = prepared_batch["keypoints"]
            if target_keypoints.shape[0] == 0:
                continue
            self._diagnostic_images_with_targets += 1

            if pred["keypoints"].shape[0] == 0:
                self._diagnostic_images_without_predictions += 1
                continue
            self._diagnostic_images_with_predictions += 1

            # The dataset contains one cube per image.  Use the most
            # confident prediction, matching the normal single-object case.
            prediction_index = pred["conf"].argmax()
            prediction_xy = pred["keypoints"][prediction_index, ..., :2]
            target_xy = target_keypoints[0, ..., :2]
            target_visible = target_keypoints[0, ..., 2] != 0

            permutations = self._diagnostic_permutations.to(prediction_xy.device)
            candidate_targets = target_xy[permutations]
            candidate_visible = target_visible[permutations]
            point_errors = torch.linalg.vector_norm(
                prediction_xy.unsqueeze(0) - candidate_targets,
                dim=-1,
            )
            candidate_scores = (
                point_errors * candidate_visible
            ).sum(dim=1) / candidate_visible.sum(dim=1).clamp_min(1)
            winner = candidate_scores.argmin()

            winner_errors = point_errors[winner]
            winner_visible = candidate_visible[winner]
            source_indices = permutations[winner].detach().cpu()
            state = self._diagnostic_layer_states.get(
                Path(batch["im_file"][sample_index]).stem,
                {"mode": "none"},
            )
            source_misaligned = self._misaligned_source_mask(state)
            output_misaligned = source_misaligned[source_indices].to(winner_errors.device)

            for category, category_mask in (
                ("misaligned", winner_visible & output_misaligned),
                ("stationary", winner_visible & ~output_misaligned),
            ):
                count = int(category_mask.sum().item())
                if count:
                    self._diagnostic_error_sums[category] += float(
                        winner_errors[category_mask].sum().item()
                    )
                    self._diagnostic_point_counts[category] += count

    def update_metrics(self, preds: list[dict[str, Tensor]], batch: dict[str, object]) -> None:
        super().update_metrics(preds, batch)
        self._track_keypoint_errors(preds, batch)

    def get_stats(self) -> dict[str, float]:
        stats = super().get_stats()
        stationary_count = self._diagnostic_point_counts["stationary"]
        misaligned_count = self._diagnostic_point_counts["misaligned"]
        stationary_error = (
            self._diagnostic_error_sums["stationary"] / stationary_count
            if stationary_count
            else 0.0
        )
        misaligned_error = (
            self._diagnostic_error_sums["misaligned"] / misaligned_count
            if misaligned_count
            else 0.0
        )
        stats.update({
            "val_stationary_kpt_error_px": round(stationary_error, 5),
            "val_misaligned_kpt_error_px": round(misaligned_error, 5),
            "val_stationary_kpt_count": float(stationary_count),
            "val_misaligned_kpt_count": float(misaligned_count),
            "val_kpt_eval_images": float(self._diagnostic_images_with_predictions),
            "val_kpt_missing_prediction_images": float(self._diagnostic_images_without_predictions),
        })
        print(
            "Keypoint error (symmetry-matched, visible points): "
            f"stationary={stationary_error:.2f}px ({stationary_count}), "
            f"misaligned={misaligned_error:.2f}px ({misaligned_count}); "
            f"missing predictions={self._diagnostic_images_without_predictions}"
        )
        return stats


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
        model.identity_only = TRAINING_IDENTITY_ONLY
        dataset_root = Path(self.data["path"])
        symmetry_file = self.data.get("kpt_symmetry_file", "ordered_keypoint_symmetries.json")
        model.ordered_keypoint_symmetry_permutations = load_symmetry_permutations(
            dataset_root / str(symmetry_file)
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"
        return MisalignmentTrackingPoseValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy.copy(self.args),
            _callbacks=self.callbacks,
        )


def train(data_yaml: Path, args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    global TRAINING_IDENTITY_ONLY
    TRAINING_IDENTITY_ONLY = bool(args.identity_only)

    model_path = args.resume_checkpoint or args.model
    if args.resume_checkpoint is not None and not args.resume_checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {args.resume_checkpoint}")

    model = YOLO(str(model_path))
    overfit_mode = args.overfit_images is not None
    overfit_with_augmentations = overfit_mode and args.overfit_augmentations
    if overfit_with_augmentations:
        # Leave Ultralytics' augmentation defaults untouched for an explicit
        # augmentation stress test on the selected overfit image set.
        train_options = {}
        if args.overfit_disable_mosaic:
            train_options["mosaic"] = 0.0
    else:
        train_options = {
            "mosaic": 0.0,
            "mixup": 0.0,
            "cutmix": 0.0,
            "copy_paste": 0.0,
            "fliplr": args.fliplr,
        }
    if overfit_mode and not overfit_with_augmentations:
        # A memorization test must present the same pixels and geometry on
        # every pass.  Otherwise a persistent pose loss can be caused by the
        # augmentation stream rather than by the criterion or pose head.
        train_options.update({
            # The default detector schedule decays too far before a newly
            # expanded 26-point pose head can memorize a tiny dataset.  Use
            # a stable optimizer schedule for this diagnostic only; normal
            # training keeps Ultralytics' defaults below.
            "optimizer": "AdamW",
            "lr0": 1e-3,
            "lrf": 0.1,
            "warmup_epochs": 0.0,
            "weight_decay": 0.0,
            "degrees": 0.0,
            "translate": 0.0,
            "scale": 0.0,
            "shear": 0.0,
            "perspective": 0.0,
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "erasing": 0.0,
            "fliplr": 0.0,
            "flipud": 0.0,
        })
    if overfit_mode and args.overfit_fixed_lr:
        # Keep the update size constant when testing whether LR decay is the
        # reason a small overfit run stops improving.  This is diagnostic-only
        # and does not alter the normal training schedule.
        train_options.update({
            "optimizer": "AdamW",
            "lr0": 2e-4,
            "lrf": 1.0,
            "warmup_epochs": 0.0,
            "weight_decay": 0.0,
        })

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
        **train_options,
        plots=True,
        resume=str(args.resume_checkpoint) if args.resume_checkpoint is not None else False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("data/synthetic_cube_dataset"))
    parser.add_argument("--workdir", type=Path, default=Path("runs/ordered_cube_pose_local"))
    parser.add_argument("--model", default="yolo11n-pose.pt")
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Resume from an Ultralytics last.pt checkpoint, preserving optimizer and scheduler state",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--train-layer-mode",
        choices=TRAIN_LAYER_MODE_CHOICES,
        default="all",
        help=(
            "Filter the training split by layer state: all, stationary, or turning "
            "(mid_turn and corner_cut together)"
        ),
    )
    parser.add_argument(
        "--overfit-images",
        type=int,
        default=None,
        help="Use the same small image set for train and val as a memorization diagnostic",
    )
    parser.add_argument(
        "--overfit-augmentations",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep Ultralytics' default augmentations enabled during an overfit diagnostic",
    )
    parser.add_argument(
        "--overfit-disable-mosaic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable only mosaic when --overfit-augmentations is enabled",
    )
    parser.add_argument(
        "--overfit-fixed-lr",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep AdamW learning rate fixed during an overfit diagnostic",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--extra-negative-dir",
        type=Path,
        default=DEFAULT_EXTRA_NEGATIVE_DIR,
        help=(
            "Directory of cube-free images to include with empty pose labels "
            f"(default: {DEFAULT_EXTRA_NEGATIVE_DIR})"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Total target epochs; when resuming, set this above the completed epoch count",
    )
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument(
        "--identity-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Train against the fixed Blender keypoint order only; "
            "symmetry routing is enabled by default"
        ),
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--fliplr",
        type=float,
        default=0.0,
        help="Horizontal image-flip probability; keep at 0 for ordered cube geometry",
    )
    parser.add_argument("--name", default="ordered_cube_yolo11n_pose")
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true", help="Replace an existing prepared dataset directory")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    dataset_dir = args.workdir / "dataset"
    existing_data_yaml = dataset_dir / "data.yaml"
    if args.resume_checkpoint is not None and dataset_dir.is_dir() and not args.force:
        if not existing_data_yaml.is_file():
            raise FileNotFoundError(
                f"Existing resume dataset is missing its data.yaml: {existing_data_yaml}; "
                "pass --force to rebuild it"
            )
        data_yaml = existing_data_yaml
        print(f"Reusing prepared dataset: {data_yaml}")
    else:
        data_yaml, _permutations = write_prepared_dataset(
            source_dir=args.source_dir,
            dataset_dir=dataset_dir,
            val_fraction=args.val_fraction,
            seed=args.seed,
            force=args.force,
            extra_negative_dir=args.extra_negative_dir,
            overfit_images=args.overfit_images,
            train_layer_mode=args.train_layer_mode,
        )
    if not args.prepare_only:
        args.runs_dir = args.workdir / "runs"
        train(data_yaml, args)


if __name__ == "__main__":
    main()
