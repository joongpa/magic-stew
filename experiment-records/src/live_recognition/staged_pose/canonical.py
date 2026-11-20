"""Known-static canonical point features, without projecting candidate turns.

Intersect each detected point's camera ray with both possible outer-sticker
planes (landmarks only use1.5). Radius and C4-invariant angular moments preserve
the rotation evidence that XY mean pooling discards. No part/angle labels enter.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from live_recognition.direct_cube_pose import FACE_AXES


def backproject(static, xy, planes=(1.5, 1.0)):
    """Return Bx6x2xNx3 object-space ray/plane intersections and conditioning."""
    axes = xy.new_tensor(FACE_AXES)
    normals = static["rotation_matrix"] @ axes.T
    normals = normals.transpose(-1, -2)
    u = (xy - static["center"][:, None]) / static["scale"][:, None, None]
    v = (
        (static["perspective"][:, None, None] / 3)
        * (xy - static["principal"][:, None])
        / static["scale"][:, None, None]
    )
    ray = torch.cat((v, torch.ones_like(v[..., :1])), -1)
    origin = torch.cat((u, torch.zeros_like(u[..., :1])), -1)
    denominator = torch.einsum("bfc,bnc->bfn", normals, ray)
    numerator = (
        xy.new_tensor(planes)[None, None, :, None]
        - torch.einsum("bfc,bnc->bfn", normals, origin)[:, :, None]
    )
    safe = torch.where(
        denominator.abs() > 0.005, denominator, torch.where(denominator >= 0, 0.005, -0.005)
    )
    depth = numerator / safe[:, :, None]
    camera = origin[:, None, None] + depth[..., None] * ray[:, None, None]
    points = torch.einsum("bfknc,bcd->bfknd", camera, static["rotation_matrix"])
    forward = (1 + static["perspective"][:, None, None, None] * depth / 3) > 0.02
    valid = (denominator.abs() > 0.005)[:, :, None] & forward
    return points, denominator, valid


@torch.inference_mode()
def canonical_features(
    static, objects, template, middle=False, safe_grad=False, projective_camera=None
):
    planes = (0.5, 0.0) if middle else (1.5, 1.0)
    points, denominator, valid = (
        backproject(static, objects["xy"], planes)
        if projective_camera is None
        else projective_camera.backproject(objects["xy"], planes)
    )
    b, _, _, n, _ = points.shape
    axes = points.new_tensor(FACE_AXES)
    j = axes.abs().argmax(-1)
    tangent = F.one_hot((j + 1) % 3, 3).to(points.dtype)
    second = torch.linalg.cross(axes, tangent)
    x = (points * tangent[None, :, None, None]).sum(-1)
    y = (points * second[None, :, None, None]).sum(-1)
    radius = (x.square() + y.square()).clamp_min(1e-12).sqrt()
    if safe_grad:
        central = (x.square() + y.square()) < 1e-12
        beta = torch.atan2(
            torch.where(central, torch.zeros_like(y), y),
            torch.where(central, torch.ones_like(x), x),
        )
    else:
        beta = torch.atan2(y, x)
    # Membership is a hypothesis, never supplied by metadata in actual forward.
    classes = objects["classes"].clamp_max(3)
    valid = valid & (radius < 3) & objects["mask"][:, None, None]
    plane_index = torch.arange(2, device=points.device)[None, None, :, None]
    if middle:
        valid = valid & (
            ((plane_index == 0) & ((classes[:, None, None] == 1) | (classes[:, None, None] == 2)))
            | ((plane_index == 1) & (classes[:, None, None] >= 3))
        )
    else:
        valid = valid & ((plane_index == 0) | (classes[:, None, None] >= 3))
    harmonics = torch.arange(1, 4, device=points.device) * 4
    cos = (beta[..., None] * harmonics).cos()
    sin = (beta[..., None] * harmonics).sin()
    angular = torch.stack((cos, -sin), -1).flatten(-2)
    angular = angular * radius.clamp_max(0.1)[..., None] / 0.1
    # Expected rest-angle moments at each class's possible radial ring.
    rest_r = torch.zeros(4, 2, 3, device=points.device)
    rest_beta = torch.zeros_like(rest_r)
    ring_mask = torch.zeros_like(rest_r, dtype=torch.bool)
    for cls, r, angle in [
        (0, math.sqrt(4.5), math.pi / 4),
        (1, math.sqrt(2.5), math.atan(1 / 3)),
        (2, math.sqrt(0.5), math.pi / 4),
    ]:
        rest_r[cls, 0, 0] = r
        rest_beta[cls, 0, 0] = angle
        ring_mask[cls, 0, 0] = True
    rest_r[3, 0] = points.new_tensor([0.0, 1.0, math.sqrt(2)])
    rest_beta[3, 0] = points.new_tensor([0.0, 0.0, math.pi / 4])
    ring_mask[3, 0] = True
    rest_r[3, 1, :2] = points.new_tensor([1.5, math.sqrt(3.25)])
    rest_beta[3, 1, :2] = points.new_tensor([0.0, math.atan(2 / 3)])
    ring_mask[3, 1, :2] = True
    if middle:
        rest_r.zero_()
        rest_beta.zero_()
        ring_mask.zero_()
        rest_r[1, 0, 0] = math.sqrt(4.5)
        rest_beta[1, 0, 0] = math.pi / 4
        ring_mask[1, 0, 0] = True
        rest_r[2, 0, 0] = math.sqrt(2.5)
        rest_beta[2, 0, 0] = math.atan(1 / 3)
        ring_mask[2, 0, 0] = True
        rest_r[3, 1, :2] = points.new_tensor([1.5, math.sqrt(3.25)])
        rest_beta[3, 1, :2] = points.new_tensor([0.0, math.atan(2 / 3)])
        ring_mask[3, 1, :2] = True
    # Bx2xNx3 -> broadcast over six face hypotheses.
    rr = rest_r[classes].transpose(1, 2)[:, None]
    rb = rest_beta[classes].transpose(1, 2)[:, None]
    rm = ring_mask[classes].transpose(1, 2)[:, None]
    distance = (radius[..., None] - rr).abs().masked_fill(~rm, 10.0)
    ring_weight = (-distance.square() / (2 * 0.15**2)).softmax(-1)
    moment = (ring_weight[..., None] * (rb[..., None] * harmonics).cos()).sum(-2)
    noncentral = (ring_weight * (rr > 0.1)).sum(-1)
    moment = moment * noncentral[..., None]
    vote = (torch.stack((cos, -sin), -1) * moment[..., None]).flatten(-2)
    # A single static projection supplies residuals to stationary middle slabs;
    # there are NO candidate angle projections or iterative fitting here.
    base = dict(static, angles=points.new_zeros(b, 6))
    family = torch.zeros(b, device=points.device, dtype=torch.long)
    if projective_camera is None:
        uv, _, types, _ = template.geometry.project(base, family)
    else:
        uv, types = projective_camera.static_projection(template)
    owner = template.owner[0]
    compatible = classes[:, :, None] == types[:, None]
    squared = (objects["xy"][:, :, None] - uv[:, None]).square().sum(-1) * 128**2
    middle = []
    for axis in axes[::2]:
        stationary = (owner @ axis).abs() < 0.1
        minimum = squared.masked_fill(
            ~(compatible & (stationary & template.owned_mask[0])[None, None]), 1e6
        ).amin(-1)
        d = (minimum.clamp_min(1e-12) if safe_grad else minimum).sqrt()
        middle.append(d)
    middle = torch.stack(middle, 1).repeat_interleave(2, 1)[:, :, None].expand(-1, -1, 2, -1)
    minimum = squared.masked_fill(~(compatible & template.owned_mask[0][None, None]), 1e6).amin(-1)
    all_static = (
        (minimum.clamp_min(1e-12) if safe_grad else minimum)
        .sqrt()[:, None, None]
        .expand(-1, 6, 2, -1)
    )
    # Face viewing angle is invariant to quarter-turn choices of tangent basis.
    if projective_camera is None:
        translation = torch.cat(
            (
                (static["center"] - static["principal"]) / static["scale"][:, None],
                3 / static["perspective"][:, None],
            ),
            -1,
        )
        camera = -(static["rotation_matrix"].transpose(-1, -2) @ translation[..., None]).squeeze(-1)
    else:
        camera = projective_camera.camera_center()
    facing = ((camera[:, None] - 1.5 * axes[None]) * axes[None]).sum(-1) / (
        camera[:, None] - 1.5 * axes[None]
    ).norm(dim=-1)
    cls = F.one_hot(classes, 4).to(points.dtype)[:, None, None].expand(-1, 6, 2, -1, -1)
    scalar = torch.stack(
        (
            radius.clamp_max(3),
            distance.amin(-1).clamp_max(2),
            objects["score"][:, None, None].expand(-1, 6, 2, -1),
            denominator[:, :, None].expand(-1, -1, 2, -1).clamp(-3, 3),
            facing[:, :, None, None].expand(-1, -1, 2, n),
            (middle / 8).clamp_max(4),
            (all_static / 8).clamp_max(4),
            torch.arange(2, device=points.device, dtype=points.dtype)[None, None, :, None].expand(
                b, 6, -1, n
            ),
        ),
        -1,
    )
    result = torch.cat((cls, scalar, angular, vote), -1)
    assert result.shape[-1] == 24 and torch.isfinite(result).all()
    return result, valid


def canonical_features_with_grad(static, objects, template, middle=False, projective_camera=None):
    """Same analytic representation, with finite center/zero-distance gradients.

    The original inference-only API retains its original arithmetic by default.
    Only this explicit training entry point bypasses the inference decorator.
    """
    return canonical_features.__wrapped__(
        static,
        objects,
        template,
        middle=middle,
        safe_grad=True,
        projective_camera=projective_camera,
    )


class CanonicalPointHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.token = nn.Sequential(nn.Linear(24, 48), nn.SiLU(), nn.Linear(48, 32), nn.SiLU())
        self.membership = nn.Linear(32, 1)
        self.face = nn.Sequential(
            nn.Linear(32 * 2 + 12 + 1, 64), nn.SiLU(), nn.Linear(64, 48), nn.SiLU()
        )
        self.phase = nn.Linear(48, 9)
        self.axis = nn.Sequential(nn.Linear(96, 48), nn.SiLU(), nn.Linear(48, 1))
        nn.init.zeros_(self.phase.weight)
        nn.init.zeros_(self.phase.bias)
        with torch.no_grad():
            self.phase.bias[4] = 2.0
        nn.init.zeros_(self.axis[-1].weight)
        nn.init.zeros_(self.axis[-1].bias)

    def forward(self, x, valid, gt_membership=None):
        b = x.shape[0]
        encoded = self.token(x)
        assignment = self.membership(encoded).squeeze(-1)
        score = x[..., 6].square()  # onehot4 + radius + radialresidual + POSEOBJECTCLASSscore
        weights = assignment.sigmoid() * valid * score
        if gt_membership is not None:
            weights = gt_membership * valid * score
        mass = weights.sum((-1, -2)).clamp_min(1e-6)
        mean = (encoded * weights[..., None]).sum((-2, -3)) / mass[..., None]
        maximum = encoded.masked_fill(~valid[..., None], -1e4).amax((-2, -3))
        maximum = torch.where(valid.any((-1, -2))[..., None], maximum, 0.0)
        moments = (x[..., -12:] * weights[..., None]).sum((-2, -3)) / mass[..., None]
        context = self.face(torch.cat((mean, maximum, moments, mass.log1p()[..., None]), -1))
        phase = self.phase(context).log_softmax(-1).reshape(b, 3, 2, 9)
        pair = context.reshape(b, 3, 2, 48)
        axis = self.axis(
            torch.cat((pair.sum(2), (pair[:, :, 0] - pair[:, :, 1]).abs()), -1)
        ).squeeze(-1)
        joint = phase[:, :, 0, :, None] + phase[:, :, 1, None, :] + axis[:, :, None, None]
        return joint, assignment
