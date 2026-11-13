"""Manifest-based loading for the additional direct cube pose labels."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations, product
import json
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .direct_cube_pose import FACE_ORDER, FACE_COLORS


def landmark_template():
    points = []
    for outer_count in (3, 2, 1):
        for outer_axes in combinations(range(3), outer_count):
            for signs in product((-1, 1), repeat=3):
                points.append([signs[i] * (1.5 if i in outer_axes else .5) for i in range(3)])
    points = np.asarray(points, dtype=np.float32)
    return points, np.where(np.abs(points) > 1, np.sign(points), 0)


def read_manifest(root):
    root = Path(root).resolve()
    entries = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines() if line.strip()]
    if len({entry["key"] for entry in entries}) != len(entries):
        raise ValueError("Dataset manifest contains duplicate keys")
    if len(entries) < 2:
        raise ValueError("At least two images are required for training and validation")
    symmetry = json.loads((root / "symmetries.json").read_text())
    if symmetry.get("face_order") != list(FACE_ORDER) or symmetry.get("center_color_order") != list(FACE_COLORS):
        raise ValueError("Dataset symmetry face/color order is incompatible")
    if [candidate["id"] for candidate in symmetry["candidates"]] != list(range(24)):
        raise ValueError("Expected 24 contiguous cube symmetry IDs")
    for entry in entries:
        for field in ("image", "pose", "target"):
            if not (root / entry[field]).is_file():
                raise FileNotFoundError(root / entry[field])
    return entries, symmetry


def split_entries(entries, val_fraction=.1, seed=17, limit=None):
    """Stratify by source, articulation mode and allowed symmetry count."""
    if not 0 < val_fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    rng = random.Random(seed)
    groups = defaultdict(list)
    for entry in entries:
        groups[(entry["source_dataset"], entry["mode"], entry["symmetry_count"])].append(entry)
    for values in groups.values():
        rng.shuffle(values)
    if limit is not None:
        if limit < 2:
            raise ValueError("Training limit must be at least two")
        # Round-robin includes varied source/mode/symmetry strata in smoke runs.
        selected = []
        while groups and len(selected) < limit:
            for key in sorted(list(groups)):
                selected.append(groups[key].pop())
                if not groups[key]:
                    del groups[key]
                if len(selected) == limit:
                    break
        return split_entries(selected, val_fraction, seed)
    train, val = [], []
    for key in sorted(groups):
        values = groups[key]
        count = max(1, min(len(values) - 1, round(len(values) * val_fraction))) if len(values) > 1 else 0
        val.extend(values[:count])
        train.extend(values[count:])
    if not val:
        rng.shuffle(train)
        val.append(train.pop())
    if not train:
        raise ValueError("Training split is empty")
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def target_tensors(payload):
    if payload.get("format") != "direct_cube_centers_projection_v3":
        raise ValueError("Expected the six-center direct_cube_centers_projection_v3 dataset")
    if payload["pose_anchor_order"] != list(FACE_ORDER) or payload["pose_anchor_color_order"] != list(FACE_COLORS):
        raise ValueError("Pose center order differs from the model")
    width, height = payload["image_size"]
    if width != height:
        raise ValueError("Training currently expects square cube crops")
    enabled = payload["pose_supervision"]["enabled"]
    allowed = torch.zeros(24, dtype=torch.bool)
    ids = payload["symmetry"]["allowed_ids"]
    if not ids or any(not isinstance(value, int) or not 0 <= value < 24 for value in ids):
        raise ValueError("Invalid allowed symmetry IDs")
    allowed[ids] = True
    result = {"enabled": torch.tensor(enabled), "allowed": allowed,
              "rotation": torch.eye(3), "center": torch.full((2,), .5),
              "scale": torch.tensor(.12), "perspective": torch.tensor(.2), "principal": torch.full((2,), .5),
              "angles": torch.zeros(6), "centers": torch.full((6, 2), .5), "landmarks": torch.full((56, 2), .5),
              "observed_landmarks": torch.zeros(56, 2), "observed_mask": torch.zeros(56, dtype=torch.bool)}
    if enabled:
        centers = payload["virtual_core_anchors"]
        if [anchor["name"] for anchor in centers] != list(FACE_ORDER):
            raise ValueError("Expected exactly six ordered center anchors")
        if len(payload["landmark_anchors"]) != 56:
            raise ValueError("Expected 56 ordered articulated landmark anchors")
        names = {anchor["name"]: index for index, anchor in enumerate(payload["landmark_anchors"])}
        observed_counts = torch.zeros(56)
        for observation in payload["matched_observations"]:
            if observation["visibility"] not in (1, 2):
                continue
            index = names[observation["anchor"]]
            result["observed_landmarks"][index] += torch.tensor(observation["observed_pixel"]) / width
            observed_counts[index] += 1
        result["observed_landmarks"] /= observed_counts[:, None].clamp_min(1)
        result["observed_mask"] = observed_counts > 0
        projection = payload["projection"]
        result.update(rotation=torch.tensor(payload["rotation_matrix"], dtype=torch.float32),
                      center=torch.tensor(projection["center_xy_normalized"], dtype=torch.float32),
                      scale=torch.tensor(projection["image_scale_normalized"], dtype=torch.float32),
                      perspective=torch.tensor(projection["perspective_strength"], dtype=torch.float32),
                      principal=torch.tensor(projection["principal_point_normalized"], dtype=torch.float32),
                      angles=torch.tensor(payload["layer_angles_degrees"], dtype=torch.float32),
                      centers=torch.tensor([anchor["pixel_point"] for anchor in centers], dtype=torch.float32) / width,
                      landmarks=torch.tensor([anchor["pixel_point"] for anchor in payload["landmark_anchors"]], dtype=torch.float32) / width)
        if result["scale"] <= 0 or not 0 < result["perspective"] < 1:
            raise ValueError("Invalid projection scale/perspective")
        if result["angles"].abs().max() > 45.0001:
            raise ValueError("Face offsets must be in [-45,45] degrees")
    if any(not torch.isfinite(value).all() for value in result.values()):
        raise ValueError("Nonfinite direct-pose target")
    return result


def independent_labels(path):
    rows = [list(map(float, line.split())) for line in path.read_text().splitlines() if line.strip()]
    if any(len(row) != 8 for row in rows):
        raise ValueError(f"Expected nine-class, one-keypoint YOLO labels: {path}")
    values = torch.tensor(rows, dtype=torch.float32).reshape(-1, 8)
    if not torch.isfinite(values).all():
        raise ValueError(f"Nonfinite independent labels: {path}")
    if len(values) and ((values[:, 0] != values[:, 0].long()).any() or values[:, 0].min() < 0 or values[:, 0].max() > 8):
        raise ValueError(f"Invalid independent class ID: {path}")
    if len(values) and ((values[:, 1:7] < 0).any() or (values[:, 1:7] > 1).any()
                        or not torch.isin(values[:, 7], values.new_tensor((0, 1, 2))).all()):
        raise ValueError(f"Invalid independent coordinates/visibility: {path}")
    return values


class DirectPoseDataset(Dataset):
    """Cache compact labels once; resize square RGB crops without spatial augmentation."""

    def __init__(self, root, entries, image_size=160, augment=False):
        self.root = Path(root).resolve()
        self.entries = entries
        self.image_size = image_size
        self.augment = augment
        targets = [target_tensors(json.loads((self.root / entry["target"]).read_text())) for entry in entries]
        self.targets = {key: torch.stack([target[key] for target in targets]) for key in targets[0]}
        self.labels = [independent_labels(self.root / entry["pose"]) for entry in entries]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        image = cv2.imread(str(self.root / entry["image"]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read {entry['image']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        if self.augment:
            # Brightness/contrast and modest blur preserve color identities and
            # target geometry. No GT keypoints are supplied to the model.
            image = np.clip(image.astype(np.float32) * np.random.uniform(.8, 1.2)
                            + np.random.uniform(-8, 8), 0, 255)
            if np.random.random() < .15:
                image = cv2.GaussianBlur(image, (3, 3), np.random.uniform(.2, .8))
        image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255
        return image, {key: value[index] for key, value in self.targets.items()}, self.labels[index], entry["key"]


def collate_direct_pose(samples):
    images, targets, labels, keys = zip(*samples)
    independent = {"img": torch.stack(images),
                   "batch_idx": torch.cat([torch.full((len(label),), index, dtype=torch.float32) for index, label in enumerate(labels)]),
                   "cls": torch.cat([label[:, :1] for label in labels]),
                   "bboxes": torch.cat([label[:, 1:5] for label in labels]),
                   "keypoints": torch.cat([label[:, 5:].reshape(-1, 1, 3) for label in labels])}
    direct = {key: torch.stack([target[key] for target in targets]) for key in targets[0]}
    return independent, direct, keys
