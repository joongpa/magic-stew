"""Continuous cube-frame-invariant orientation and circular articulation fields.

Targets are unchanged by proper cube-frame permutations; face colors can
resolve the frame separately. Angle fields are periodic in90degrees.
"""

import itertools
import math

import torch
from torch import nn
from torch.nn import functional as F

from live_recognition.direct_cube_pose import FACE_AXES


def make_bases():
    tuples = list(itertools.product(range(3), repeat=4))
    groups = sorted({tuple(sorted(v)) for v in tuples})
    symmetric = torch.zeros(81, 15, dtype=torch.float64)
    for column, group in enumerate(groups):
        ids = [i for i, value in enumerate(tuples) if tuple(sorted(value)) == group]
        symmetric[ids, column] = 1 / math.sqrt(len(ids))
    pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    trace = torch.zeros(6, 81, dtype=torch.float64)
    for row, (k, l) in enumerate(pairs):
        for i in range(3):
            trace[row, tuples.index((i, i, k, l))] = 1
    _, _, vh = torch.linalg.svd(trace @ symmetric, full_matrices=True)
    harmonic = symmetric @ vh[6:].T
    eye = torch.eye(3, dtype=torch.float64)
    isotropic = (
        torch.einsum("ij,kl->ijkl", eye, eye)
        + torch.einsum("ik,jl->ijkl", eye, eye)
        + torch.einsum("il,jk->ijkl", eye, eye)
    ) / 5
    matrix_basis = torch.zeros(6, 3, 3, dtype=torch.float64)
    for i, (a, b) in enumerate(pairs):
        matrix_basis[i, a, b] = 1 if a == b else 1 / math.sqrt(2)
        matrix_basis[i, b, a] = matrix_basis[i, a, b]
    assert (harmonic.T @ harmonic - torch.eye(9)).abs().max() < 1e-12
    assert (trace @ harmonic).abs().max() < 1e-12
    return harmonic.float(), isotropic.float(), matrix_basis.float()


class CubePoseFields(nn.Module):
    def __init__(self, seeds=48, iterations=10):
        super().__init__()
        harmonic, isotropic, matrix_basis = make_bases()
        self.register_buffer("harmonic", harmonic)
        self.register_buffer("isotropic", isotropic)
        self.register_buffer("matrix_basis", matrix_basis)
        index = torch.arange(seeds).float()
        z = 1 - 2 * (index + 0.5) / seeds
        azimuth = index * (math.pi * (3 - math.sqrt(5)))
        radius = (1 - z.square()).sqrt()
        sphere = torch.stack((radius * azimuth.cos(), radius * azimuth.sin(), z), -1)
        self.register_buffer("seeds", torch.cat((sphere, torch.eye(3), -torch.eye(3))))
        self.iterations = iterations
        self.register_buffer(
            "orientation_norm", self.encode_rotation(torch.eye(3)[None]).norm(dim=-1)[0]
        )

    def encode_rotation(self, rotation):
        tensor = torch.einsum("bia,bja,bka,bla->bijkl", rotation, rotation, rotation, rotation)
        return tensor.flatten(1) @ self.harmonic

    def decode_rotation(self, field):
        # The anisotropy's amplitude expresses regression uncertainty; its
        # direction determines the represented frame. Normalize only decoding.
        field = field / field.norm(dim=-1, keepdim=True).clamp_min(1e-6) * self.orientation_norm
        tensor = (field @ self.harmonic.T).reshape(-1, 3, 3, 3, 3) + self.isotropic[None]

        def contraction(v):
            return torch.einsum("bijkl,bnj,bnk,bnl->bni", tensor, v, v, v)

        seeds = self.seeds[None].expand(len(field), -1, -1)
        v = seeds
        for _ in range(self.iterations):
            v = F.normalize(contraction(v), dim=-1, eps=1e-8)
        score = (contraction(v) * v).sum(-1)
        row = torch.arange(len(field), device=field.device)
        first = v[row, score.argmax(-1)]
        v = F.normalize(
            seeds - (seeds * first[:, None]).sum(-1, keepdim=True) * first[:, None],
            dim=-1,
            eps=1e-8,
        )
        for _ in range(self.iterations):
            updated = contraction(v)
            v = F.normalize(
                updated - (updated * first[:, None]).sum(-1, keepdim=True) * first[:, None],
                dim=-1,
                eps=1e-8,
            )
        score = (contraction(v) * v).sum(-1)
        second = v[row, score.argmax(-1)]
        third = F.normalize(torch.linalg.cross(first, second, dim=-1), dim=-1)
        second = torch.linalg.cross(third, first, dim=-1)
        return torch.stack((first, second, third), -1)

    def encode_articulation(self, rotation, angles):
        normal = rotation @ rotation.new_tensor(FACE_AXES).T
        phase = angles * math.pi / 45
        values = []
        for scalar in [phase.sin(), phase.cos()]:
            matrix = 0.5 * torch.einsum("bif,bjf,bf->bij", normal, normal, scalar)
            vector = 0.5 * torch.einsum("bif,bf->bi", normal, scalar)
            values.extend((torch.einsum("bij,kij->bk", matrix, self.matrix_basis), vector))
        return torch.cat(values, -1)

    def decode_articulation(self, rotation, field):
        normal = rotation @ rotation.new_tensor(FACE_AXES).T
        values = []
        for offset in [0, 9]:
            matrix = torch.einsum("bk,kij->bij", field[:, offset : offset + 6], self.matrix_basis)
            vector = field[:, offset + 6 : offset + 9]
            values.append(
                torch.einsum("bif,bij,bjf->bf", normal, matrix, normal)
                + torch.einsum("bif,bi->bf", normal, vector)
            )
        return torch.atan2(values[0], values[1]) * 45 / math.pi
