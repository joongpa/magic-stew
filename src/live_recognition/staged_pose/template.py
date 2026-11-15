"""Pose-conditioned CNN descriptor matching against the articulated template.

Input geometry is generated ONLY from the predicted static body. Ground-truth
point indices and rotation frames belong to the training targets, not forward.
"""

import torch
from torch import nn
from torch.nn import functional as F

from live_recognition.direct_cube_pose_data import landmark_template

from .geometry import ClassNearestGeometry


class CorrespondenceTemplate(nn.Module):
    def __init__(self):
        super().__init__()
        geometry = ClassNearestGeometry()
        self.geometry = geometry
        landmark = torch.tensor(landmark_template()[0]).float()
        unique = torch.cat((landmark, geometry.quads.mean(1)), 0)
        source = (landmark[:, None] - geometry.rest[0, None]).square().sum(-1).argmin(-1)
        self.register_buffer("unique_source", torch.cat((source, torch.arange(80, 134))))
        rest = torch.cat((geometry.rest, geometry.quads.mean(1)[None].expand(3, -1, -1)), 1)
        axis = geometry.axes[::2]
        layers = ((geometry.sticker_owners[None] * axis[:, None]).sum(-1).round() + 1).long()
        layers = torch.cat((geometry.layers, layers), -1)
        owner = torch.where(geometry.rest.abs() > 1, geometry.rest.sign(), 0.0)
        owner = (
            owner
            + axis[:, None] * ((layers[:, :80] - 1) - (owner * axis[:, None]).sum(-1))[..., None]
        )
        owner = torch.cat((owner, geometry.sticker_owners[None].expand(3, -1, -1)), 1)
        mapping = (rest[:, :, None] - unique[None, None]).square().sum(-1).argmin(-1)
        assert torch.equal(unique[mapping], rest)
        self.register_buffer("rest", rest)
        self.register_buffer("owner", owner)
        self.register_buffer("mapping", mapping)
        self.register_buffer(
            "types", torch.cat((geometry.types, torch.full((3, 54), 3, dtype=torch.long)), 1)
        )
        self.register_buffer("axis", axis)
        default_owner = torch.where(rest[:, :80].abs() > 1, rest[:, :80].sign(), 0.0)
        owned = torch.cat(
            (
                (owner[:, :80] - default_owner).abs().sum(-1) < 1e-6,
                torch.ones(3, 54, dtype=torch.bool),
            ),
            1,
        )
        assert (owned.sum(-1) == 110).all()
        self.register_buffer("owned_mask", owned, persistent=False)

    def axis_features(self, rotation):
        normal = rotation @ self.axis.T
        normal = normal.transpose(-1, -2)
        matrix = normal[..., None] * normal[..., None, :]
        return matrix[
            ...,
            torch.tensor([0, 1, 2, 0, 0, 1], device=rotation.device),
            torch.tensor([0, 1, 2, 1, 2, 2], device=rotation.device),
        ]

    @staticmethod
    def camera_features(base):
        return torch.cat(
            (
                base["perspective"][:, None],
                base["scale"].log()[:, None],
                (base["principal"] - base["center"]) / base["scale"][:, None],
            ),
            -1,
        )

    def project_unique(self, base):
        family = torch.zeros(len(base["scale"]), device=base["scale"].device, dtype=torch.long)
        uv, visible, _, _ = self.geometry.project(base, family)
        return uv[:, self.unique_source], visible[:, self.unique_source]

    def describe(self, base, family, uv_unique, visible_unique, cnn_unique):
        mapping = self.mapping[family]
        uv = uv_unique.gather(1, mapping[..., None].expand(-1, -1, 2))
        visible = visible_unique.gather(1, mapping)
        cnn = cnn_unique.gather(1, mapping[..., None].expand(-1, -1, 128))
        xyz = self.rest[family] @ base["rotation_matrix"].transpose(-1, -2)
        owner = self.owner[family] @ base["rotation_matrix"].transpose(-1, -2)
        axis = self.axis_features(base["rotation_matrix"])[
            torch.arange(len(family), device=family.device), family
        ]
        camera = self.camera_features(base)
        features = torch.cat(
            (
                cnn,
                (uv - base["center"][:, None]) / base["scale"][:, None, None],
                xyz / 1.5,
                owner / 1.5,
                axis[:, None].expand(-1, 134, -1),
                visible[..., None].float(),
                F.one_hot(self.types[family], 4).float(),
                camera[:, None].expand(-1, 134, -1),
            ),
            -1,
        )
        return features, uv, visible, xyz, owner, cnn
