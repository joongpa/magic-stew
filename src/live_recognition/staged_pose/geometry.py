"""Identity-free class-nearest articulated geometry for supervised training.

Mirrors the fitter's core observation->visible-projection matching. Landmark
classes0/1/2 are hard constraints; sticker centers are geometric observations,
with color orientation handled separately as in ArticulatedProjection.
Includes duplicate seam vertices for independently moving slabs. The objective
keeps every retained observation with radial Huber influence, not hard trimming.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from live_recognition.direct_cube_pose import FACE_AXES, project_points
from live_recognition.direct_cube_pose_data import landmark_template

from .kinematics import rotation_exp
from .templates import sticker_template


class ClassNearestGeometry(nn.Module):
    def __init__(self, beta=0.02):
        super().__init__()
        self.beta = beta
        rest, _ = landmark_template()
        rest = torch.tensor(rest).float()
        axes = torch.tensor(FACE_AXES).float()
        classes = torch.tensor([0] * 8 + [1] * 24 + [2] * 24)
        points = []
        layers = []
        types = []
        bounds = []
        for family in range(3):
            n = axes[2 * family]
            coordinate = rest @ n
            ll, ix = torch.where(
                (coordinate[None] - (torch.arange(3) - 1)[:, None]).abs() <= 0.50001
            )
            points.append(rest[ix])
            layers.append(ll)
            types.append(classes[ix])
            low = torch.full((3, 3), -1.5)
            high = torch.full((3, 3), 1.5)
            j = int(n.abs().argmax())
            sign = int(n[j])
            a = torch.arange(3) - 1.5
            b = a + 1
            low[:, j] = a if sign > 0 else -b
            high[:, j] = b if sign > 0 else -a
            bounds.append(torch.stack((low, high), 1))
        r, o, n = sticker_template()
        r = torch.tensor(r).float()
        o = torch.tensor(o).float()
        n = torch.tensor(n).float()
        helper = F.one_hot(n.abs().argmax(-1).add(1).remainder(3), 3).float()
        second = torch.linalg.cross(n, helper)
        quad = torch.stack(
            (
                r - 0.5 * helper - 0.5 * second,
                r + 0.5 * helper - 0.5 * second,
                r + 0.5 * helper + 0.5 * second,
                r - 0.5 * helper + 0.5 * second,
            ),
            1,
        )
        self.register_buffer("axes", axes)
        self.register_buffer("rest", torch.stack(points))
        self.register_buffer("layers", torch.stack(layers))
        self.register_buffer("types", torch.stack(types))
        self.register_buffer("bounds", torch.stack(bounds))
        self.register_buffer("quads", quad)
        self.register_buffer("sticker_owners", o)
        self.register_buffer("sticker_normals", n)
        self.register_buffer("surface", (self.rest @ axes.T) > 1.49)

    def visibility(self, points, camera, matrices, bounds):
        origin = torch.einsum("bi,bsij->bsj", camera, matrices)[:, :, None]
        direction = torch.einsum("bni,bsij->bsnj", points - camera[:, None], matrices)
        lower = bounds[:, :, 0, None]
        upper = bounds[:, :, 1, None]
        parallel = direction.abs() < 1e-12
        inside = (origin >= lower - 1e-7) & (origin <= upper + 1e-7)
        safe = torch.where(parallel, 1.0, direction)
        first = (lower - origin) / safe
        last = (upper - origin) / safe
        entry = torch.where(parallel, -torch.inf, torch.minimum(first, last)).amax(-1).clamp_min(0)
        exit = torch.where(parallel, torch.inf, torch.maximum(first, last)).amin(-1)
        hit = (entry <= exit + 1e-7) & ((~parallel) | inside).all(-1)
        entry = torch.where(hit, entry, torch.inf)
        return (entry >= 1 - 1e-5).all(1)

    def project(self, p, family):
        b = len(family)
        row = torch.arange(b, device=family.device)
        n = self.axes[2 * family]
        a = p["angles"].gather(1, torch.stack((2 * family + 1, 2 * family), -1))
        theta = torch.stack((a[:, 0], torch.zeros_like(a[:, 0]), -a[:, 1]), -1) * math.pi / 180
        matrices = rotation_exp((n[:, None] * theta[..., None]).reshape(-1, 3)).reshape(b, 3, 3, 3)
        rest = self.rest[family]
        layer = self.layers[family]
        point_matrix = matrices[row[:, None], layer]
        xyz = torch.einsum("bnij,bnj->bni", point_matrix, rest)
        face_layers = ((self.sticker_owners[None] * n[:, None]).sum(-1).round() + 1).long()
        face_matrix = matrices[row[:, None], face_layers]
        polygons = torch.einsum("bnij,nkj->bnki", face_matrix, self.quads)
        centers = polygons.mean(2)
        uv = project_points(xyz, p)
        face_uv = project_points(polygons.flatten(1, 2), p).reshape(b, 54, 4, 2).mean(2)
        translation = torch.cat(
            ((p["center"] - p["principal"]) / p["scale"][:, None], 3 / p["perspective"][:, None]),
            -1,
        )
        camera = -(p["rotation_matrix"].transpose(-1, -2) @ translation[..., None]).squeeze(-1)
        with torch.no_grad():
            normals = torch.einsum("bnij,fj->bnfi", point_matrix, self.axes)
            front = (
                self.surface[family]
                & ((normals * (camera[:, None, None] - xyz[:, :, None])).sum(-1) > 0)
            ).any(-1)
            normals = torch.einsum("bnij,nj->bni", face_matrix, self.sticker_normals)
            face_front = (normals * (camera[:, None] - centers)).sum(-1) > 0
            all_xyz = torch.cat((xyz, centers), 1)
            clear = self.visibility(all_xyz, camera, matrices, self.bounds[family])
            visible = torch.cat((front, face_front), 1) & clear
        projected = torch.cat((uv, face_uv), 1)
        types = torch.cat((self.types[family], family.new_full((b, 54), 3)), 1)
        return (
            projected,
            visible,
            types,
            {
                "xyz": xyz,
                "centers": centers,
                "polygons": polygons,
                "camera": camera,
                "matrices": matrices,
                "landmark_visible": visible[:, :80],
                "sticker_visible": visible[:, 80:],
            },
        )

    def forward(self, p, observed, classes, mask, family):
        projected, visible, types, _ = self.project(p, family)
        obs_type = classes.clamp_max(3)
        compatible = (obs_type[:, :, None] == types[:, None]) & visible[:, None]
        squared = (observed[:, :, None] - projected[:, None]).square().sum(-1)
        nearest = squared.masked_fill(~compatible, 1000.0).amin(-1).clamp_min(1e-12).sqrt()
        valid = compatible.any(-1)
        nearest = torch.where(valid, nearest, torch.full_like(nearest, 0.25))
        nearest = torch.where(mask, nearest, 0.0)
        penalty = F.smooth_l1_loss(
            nearest, torch.zeros_like(nearest), reduction="none", beta=self.beta
        )
        groups = []
        rms = []
        for sticker in (False, True):
            active = mask & ((classes >= 3) if sticker else (classes < 3))
            count = active.sum(-1)
            groups.append(((penalty * active).sum(-1) / count.clamp_min(1), count > 0))
            rms.append(((nearest.detach().square() * active).sum(-1) / count.clamp_min(1)).sqrt())
        count = groups[0][1].float() + groups[1][1].float()
        loss = (groups[0][0] + groups[1][0]) / count.clamp_min(1)
        rmse = (nearest.detach().square().sum(-1) / mask.sum(-1).clamp_min(1)).sqrt()
        return (
            loss,
            rmse,
            {
                "landmark_rmse": rms[0],
                "sticker_rmse": rms[1],
                "unmatched_observations": (mask & ~valid).sum(-1),
            },
        )
