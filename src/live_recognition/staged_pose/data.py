"""Portable loading of exported metadata-backed independent-keypoint images."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from live_recognition.direct_cube_pose_data import independent_labels, target_tensors

from .template import CorrespondenceTemplate


class PhysicalReference(torch.nn.Module):
    """Unique56 landmark +54 sticker slots with canonical cubie ownership."""

    def __init__(self):
        super().__init__()
        self.template = CorrespondenceTemplate()
        t = self.template
        sources = []
        for family in range(3):
            owned = t.owned_mask[family].nonzero().flatten()
            sources.append(owned[t.mapping[family, owned].argsort()])
        self.register_buffer("sources", torch.stack(sources))
        self.register_buffer("rest", t.rest[0, self.sources[0]])
        self.register_buffer("owner", t.owner[0, self.sources[0]])
        self.register_buffer("types", t.types[0, self.sources[0]])

    def project(self, pose, family):
        uv = self.template.geometry.project(pose, family)[0]
        return uv.gather(1, self.sources[family, ..., None].expand(-1, -1, 2))


@torch.no_grad()
def additional_targets(payload, target, reference):
    """Bind loss-only visible targets without changing any original YOLO row."""
    xy = torch.zeros(110, 2)
    mask = torch.zeros(110, dtype=torch.bool)
    if bool(target["enabled"]):
        pose = {
            k: target["rotation" if k == "rotation_matrix" else k][None]
            for k in ["rotation_matrix", "center", "scale", "perspective", "principal", "angles"]
        }
        family = pose["angles"].abs().reshape(1, 3, 2).sum(-1).argmax(-1)
        projected = reference.project(pose, family)[0]
        width = float(payload["image_size"][0])
        names = {v["name"]: i for i, v in enumerate(payload["landmark_anchors"])}
        count = torch.zeros(56)
        for observation in payload["matched_observations"]:
            if observation["visibility"] not in (1, 2):
                continue
            slot = names[observation["anchor"]]
            xy[slot] += torch.tensor(observation["observed_pixel"]) / width
            count[slot] += 1
        xy[:56] /= count[:, None].clamp_min(1)
        mask[:56] = count > 0
        # Known center labels anchor their physical slots. Remaining sticker
        # rows require a unique nearest model projection at the128px scale.
        centers = payload["virtual_core_anchors"]
        center_slots = torch.arange(6) * 9 + 4
        used = set()
        best_error = {}
        for observation in payload["sticker_observations"]:
            if observation["visibility"] not in (1, 2):
                continue
            point = torch.tensor(observation["normalized_point"], dtype=torch.float32)
            forced = None
            for face, anchor in enumerate(centers):
                observed = anchor.get("source_observed_pixel")
                if (
                    anchor.get("source_label_visibility") in (1, 2)
                    and observed is not None
                    and (point - torch.tensor(observed) / width).norm() * 128 < 0.05
                ):
                    forced = face * 9 + 4
                    break
            distance = (projected[56:] - point).norm(dim=-1) * 128
            if forced is None:
                distance[center_slots] = 1e6
                error, index = distance.topk(2, largest=False)
                if error[0] > 1.5 or error[1] - error[0] < 0.1:
                    continue
                slot = int(index[0])
                value = float(error[0])
            else:
                slot = forced
                value = float(distance[slot])
            if slot in used and value >= best_error[slot]:
                continue
            used.add(slot)
            best_error[slot] = value
            xy[56 + slot] = point
            mask[56 + slot] = True
    axes = reference.template.geometry.axes
    radius = (
        (reference.rest.square().sum(-1)[:, None] - (reference.rest @ axes.T).square())
        .clamp_min(0)
        .sqrt()
    )
    count = ((reference.owner @ axes.T > 0.5) & (radius > 0.5) & mask[:, None]).sum(0)
    return {"visible_xy": xy, "visible_mask": mask, "eligible": count >= 3}


class StagedPoseDataset(Dataset):
    def __init__(self, root, entries, image_size=128):
        self.root = Path(root).resolve()
        self.entries = entries
        self.image_size = image_size
        reference = PhysicalReference()
        self.labels = []
        rows = []
        for i, entry in enumerate(entries):
            payload = json.loads((self.root / entry["target"]).read_text(encoding="utf-8"))
            target = target_tensors(payload)
            target.update(additional_targets(payload, target, reference))
            rows.append(target)
            self.labels.append(independent_labels(self.root / entry["pose"]))
            if i % 500 == 0:
                print(f"\rPreparing structured targets {i + 1}/{len(entries)}", end="", flush=True)
        print(flush=True)
        self.targets = {k: torch.stack([row[k] for row in rows]) for k in rows[0]}

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        image = cv2.imread(str(self.root / entry["image"]))
        if image is None:
            raise ValueError(f"Cannot read image {self.root / entry['image']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
        )
        image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255
        return (
            image,
            {k: v[index] for k, v in self.targets.items()},
            self.labels[index],
            entry["key"],
        )
