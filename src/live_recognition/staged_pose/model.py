"""Exact staged45 base+image-context architecture, with one live RGB encoder."""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from live_recognition.direct_cube_pose import (
    DirectCubePoseModel,
    DirectPoseHead,
    make_detector,
)

from .canonical import CanonicalPointHead, canonical_features_with_grad
from .fields import CubePoseFields
from .phase import decode
from .template import CorrespondenceTemplate


class ImageContextPhaseHead(nn.Module):
    """Retain the pretrained head; add a zero-initialized global image residual."""

    def __init__(self):
        super().__init__()
        self.base = CanonicalPointHead()
        self.appearance = nn.Sequential(nn.Linear(1280, 24), nn.LayerNorm(24), nn.SiLU())
        self.point = nn.Sequential(
            nn.Linear(24 + 24 + 9, 48), nn.SiLU(), nn.Linear(48, 32), nn.SiLU()
        )
        self.face = nn.Sequential(nn.Linear(64, 48), nn.SiLU())
        self.global_context = nn.TransformerEncoderLayer(
            48, 4, dim_feedforward=96, dropout=0.0, batch_first=True, norm_first=True
        )
        self.phase = nn.Linear(48, 9)
        self.axis = nn.Sequential(nn.Linear(96, 48), nn.SiLU(), nn.Linear(48, 1))
        for layer in [self.phase, self.axis[-1]]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def context_features(self, x, valid, appearance, class_scores):
        base, membership = self.base(x, valid)
        _b, f, k, _n, _ = x.shape
        image = self.appearance(appearance)[:, None, None].expand(-1, f, k, -1, -1)
        color = class_scores[:, None, None].expand(-1, f, k, -1, -1)
        token = self.point(torch.cat((x, image, color), -1))
        weights = membership.sigmoid() * valid * x[..., 6].square()
        mean = (token * weights[..., None]).sum((-2, -3)) / weights.sum((-1, -2)).clamp_min(1e-6)[
            ..., None
        ]
        maximum = token.masked_fill(~valid[..., None], -10000.0).amax((-2, -3))
        maximum = torch.where(valid.any((-1, -2))[..., None], maximum, 0.0)
        context = self.global_context(self.face(torch.cat((mean, maximum), -1)))
        return base, membership, context

    def forward(self, x, valid, appearance, class_scores):
        base, membership, context = self.context_features(x, valid, appearance, class_scores)
        b = len(x)
        phase = self.phase(context).reshape(b, 3, 2, 9)
        pair = context.reshape(b, 3, 2, 48)
        axis = self.axis(
            torch.cat((pair.sum(2), (pair[:, :, 0] - pair[:, :, 1]).abs()), -1)
        ).squeeze(-1)
        return base + phase[:, :, 0, :, None] + phase[:, :, 1, None, :] + axis[
            :, :, None, None
        ], membership


class StaticPoseHead(DirectPoseHead):
    def __init__(self, channels):
        super().__init__(channels, width=64, topk=16)
        self.regressor[-1] = nn.Linear(256, 15)
        nn.init.normal_(self.regressor[-1].weight, std=0.001)
        nn.init.zeros_(self.regressor[-1].bias)
        with torch.no_grad():
            fields = CubePoseFields()
            self.regressor[-1].bias[:9] = fields.encode_rotation(torch.eye(3)[None])[0]
            self.regressor[-1].bias[9:] = torch.tensor([0.5, 0.5, -2.1202635, -1.3862944, 0.5, 0.5])

    @staticmethod
    def decode(raw):
        return raw


def static_pose(raw, fields):
    camera = raw[:, -6:]
    return {
        "rotation_matrix": fields.decode_rotation(raw[:, :9]),
        "center": camera[:, :2],
        "scale": camera[:, 2].clamp(-7, 2).exp(),
        "perspective": camera[:, 3].sigmoid(),
        "principal": camera[:, 4:6],
        "angles": raw.new_zeros(len(raw), 6),
    }


class IndependentPointSource(nn.Module):
    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        detector.model[-1].end2end = False

    def forward(self, images):
        from ultralytics.utils.nms import non_max_suppression

        # Preserve the verified branch and fixed BN statistics in every phase.
        self.detector.eval()
        head = self.detector.model[-1]
        head.training = True
        raw = self.detector(images)
        branch = raw.get("one2many", raw)
        points, scores = DirectCubePoseModel._decode_observation_tokens(self, images, branch)
        with torch.no_grad():
            decoded = head._inference(branch).detach().clone()
            objects, indices = non_max_suppression(
                decoded, conf_thres=0.001, iou_thres=0.7, nc=9, end2end=False, return_idxs=True
            )
            selection = torch.zeros_like(scores)
            for i, (rows, index) in enumerate(zip(objects, indices)):
                if not len(rows):
                    continue
                cls = rows[:, 5].long()
                floor = torch.where(cls >= 3, 0.25, 0.05)
                keep = (rows[:, 8] >= floor) & (
                    (rows[:, 6:8] >= 0) & (rows[:, 6:8] <= images.shape[-1])
                ).all(-1)
                selection[i, cls[keep], index.long().flatten()[keep]] = 1
        object_scores = scores * selection
        all_scores, classes = object_scores.max(1)
        count = min(64, all_scores.shape[-1])
        score, index = all_scores.topk(count, -1)
        xy = points.gather(1, index[..., None].expand(-1, -1, 2))
        classes = classes.gather(1, index)
        native = {
            "xy": xy.float(),
            "classes": classes,
            "score": score.float(),
            "class_scores": scores.transpose(1, 2)
            .gather(1, index[..., None].expand(-1, -1, 9))
            .float(),
            "mask": (score >= 0.05) & ((xy >= 0) & (xy <= 1)).all(-1),
        }
        return {
            "raw": raw,
            "features": branch["feats"],
            "points": points.float(),
            "object_scores": object_scores.float(),
            "native": native,
        }


class StagedCubePoseModel(nn.Module):
    def __init__(self, detector):
        super().__init__()
        self.source = IndependentPointSource(detector)
        channels = [v[0].conv.in_channels for v in detector.model[-1].cv2]
        if list(channels) != [256, 512, 512]:
            raise ValueError(
                f"This exact staged45 architecture requires YOLO26m FPN256/512/512, got {channels}"
            )
        self.static_head = StaticPoseHead(channels)
        self.articulation_head = ImageContextPhaseHead()
        self.fields = CubePoseFields()
        self.template = CorrespondenceTemplate()
        self.phase = "joint"

    @property
    def detector(self):
        return self.source.detector

    def configure_phase(self, phase):
        if phase not in ("static", "articulation", "handoff", "joint"):
            raise ValueError(phase)
        self.phase = phase
        for name, p in self.source.named_parameters():
            p.requires_grad_(
                phase in ("static", "joint") and "one2one" not in name and ".dfl" not in name
            )
        self.static_head.requires_grad_(phase in ("static", "joint"))
        self.articulation_head.requires_grad_(phase != "static")
        self.fields.requires_grad_(False)
        self.template.requires_grad_(False)

    def encode(self, images, amp=False):
        with torch.autocast(device_type=images.device.type, enabled=amp):
            encoded = self.source(images)
        # All structured geometry and head arithmetic stays FP32, also in AMP.
        features = [v.float() for v in encoded["features"]]
        raw = self.static_head(features, encoded["points"], encoded["object_scores"])
        base = static_pose(raw, self.fields)
        xy = encoded["native"]["xy"]
        appearance = (
            torch.cat(
                [
                    F.grid_sample(v, 2 * xy[:, None] - 1, align_corners=False)[:, :, 0].transpose(
                        1, 2
                    )
                    for v in features
                ],
                -1,
            )
            .half()
            .float()
        )
        return {"encoded": encoded, "raw_static": raw, "static": base, "appearance": appearance}

    def articulate(self, context, base=None):
        pose = context["static"] if base is None else base
        native = context["encoded"]["native"]
        x, valid = canonical_features_with_grad(pose, native, self.template)
        logits, membership = self.articulation_head(
            x, valid, context["appearance"], native["class_scores"]
        )
        return logits, membership

    def forward(self, images, amp=False):
        context = self.encode(images, amp)
        logits, _ = self.articulate(context)
        return {
            "pose": dict(context["static"], angles=decode(logits)[0]),
            "logits": logits,
            "independent": context["encoded"]["raw"],
        }


CHECKPOINT_FORMAT = "staged_cube_pose_context_v1"


def model_from_checkpoint(path, device="cpu"):
    saved = torch.load(Path(path), map_location="cpu", weights_only=False)
    if saved.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Expected staged cube pose context checkpoint")
    model = StagedCubePoseModel(make_detector(saved["detector_yaml"]))
    model.load_state_dict(saved["model_state"])
    model.to(device)
    model.initialization = copy.deepcopy(saved.get("initialization", {}))
    # A new run must keep every image previously used to train these custom
    # heads out of its validation split, also when warm-starting a later run.
    known = set(model.initialization.get("training_keys", []))
    known.update(saved.get("split", {}).get("train", []))
    model.initialization["training_keys"] = sorted(known)
    return model, saved


def checkpoint_model(model):
    return {
        "format": CHECKPOINT_FORMAT,
        "detector_yaml": copy.deepcopy(model.detector.yaml),
        "initialization": copy.deepcopy(getattr(model, "initialization", {})),
        "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    }
