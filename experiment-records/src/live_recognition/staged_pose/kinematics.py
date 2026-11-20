"""Equivalent vectorized rotation and opposite-layer kinematics for decoding."""

import math

import torch


def rotation_exp(vector):
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(-1, 3, 3)
    theta = vector.norm(dim=-1)[:, None, None]
    return (
        torch.eye(3, device=vector.device, dtype=vector.dtype)[None]
        + torch.sinc(theta / math.pi) * skew
        + 0.5 * torch.sinc(theta / (2 * math.pi)).square() * (skew @ skew)
    )
