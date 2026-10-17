"""A compact pose head for the simplified 1x1x3 cube representation.

The head predicts only quantities that are present in the compact pose labels:

* canonical ``u``, ``v``, and ``twist`` angles,
* normalized cube image center and scale,
* a scalar perspective-strength term, and
* two signed angles for the negative and positive outer layers.

The two layer angles are in degrees and are constrained to ``[-45, 45)`` by
the decoder.  A no-turn sample is simply ``[0, 0]``.  There is deliberately no
turn-present or turn-layer output: those would duplicate information already
encoded by the two angle channels.

``YOLOCubePoseModel`` follows the same feature-tap convention as the color
head.  It captures the multiscale feature pyramid immediately before an
Ultralytics segmentation head and trains a small pose decoder on those
 features plus the predicted cube union, sticker union, and sticker-boundary
 soft masks. It does not consume
rasterized ground-truth masks, so the training path has the same information
available at inference time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import permutations, product
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


POSE_OUTPUT_NAMES = (
    "u",
    "v",
    "twist",
    "image_center",
    "image_scale",
    "perspective_strength",
    "layer_angles",
)
# u/v/twist, center, scale, perspective, and two outer-layer angles.
POSE_OUTPUT_DIM = 1 + 1 + 1 + 2 + 1 + 1 + 2
POSE_MASK_CHANNEL_NAMES = (
    "cube_union",
    "sticker_union",
    "sticker_boundary",
)
POSE_MASK_CHANNELS = len(POSE_MASK_CHANNEL_NAMES)
SEGMENTATION_CLASS_CHANNELS = 2
ANGLE_LIMIT_DEGREES = 45.0
GEOMETRY_HUBER_BETA = 0.01
LINE_DIRECTION_WEIGHT = 0.25
ANGLE_PERIOD_DEGREES = 90.0
IMAGE_ZENITH = (0.0, -1.0, 0.0)


def _cube_symmetry_matrices() -> tuple[Tensor, ...]:
    matrices = []
    for permutation in permutations(range(3)):
        for signs in product((-1.0, 1.0), repeat=3):
            matrix = torch.zeros(3, 3, dtype=torch.float32)
            for row, column in enumerate(permutation):
                matrix[row, column] = signs[row]
            if torch.linalg.det(matrix) > 0.5:
                matrices.append(matrix)
    return tuple(matrices)


def _matrix_to_quaternion(matrix: Tensor) -> Tensor:
    """Convert one proper rotation matrix to an xyzw quaternion."""
    trace = float(torch.trace(matrix))
    if trace > 0.0:
        scale = (trace + 1.0) ** 0.5 * 2.0
        w = 0.25 * scale
        x = float(matrix[2, 1] - matrix[1, 2]) / scale
        y = float(matrix[0, 2] - matrix[2, 0]) / scale
        z = float(matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal_index = int(torch.argmax(torch.diagonal(matrix)))
        if diagonal_index == 0:
            scale = max(float(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]), 1e-12) ** 0.5 * 2.0
            x = 0.25 * scale
            y = float(matrix[0, 1] + matrix[1, 0]) / scale
            z = float(matrix[0, 2] + matrix[2, 0]) / scale
            w = float(matrix[2, 1] - matrix[1, 2]) / scale
        elif diagonal_index == 1:
            scale = max(float(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]), 1e-12) ** 0.5 * 2.0
            x = float(matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = float(matrix[1, 2] + matrix[2, 1]) / scale
            w = float(matrix[0, 2] - matrix[2, 0]) / scale
        else:
            scale = max(float(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]), 1e-12) ** 0.5 * 2.0
            x = float(matrix[0, 2] + matrix[2, 0]) / scale
            y = float(matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
            w = float(matrix[1, 0] - matrix[0, 1]) / scale
    quaternion = torch.tensor((x, y, z, w), dtype=torch.float32)
    return F.normalize(quaternion, dim=0)


_CUBE_SYMMETRY_MATRICES = torch.stack(_cube_symmetry_matrices())
_CUBE_SYMMETRY_QUATERNIONS = torch.stack(
    tuple(_matrix_to_quaternion(matrix) for matrix in _CUBE_SYMMETRY_MATRICES)
)
_CUBE_SYMMETRY_PRESERVES_STACK = _CUBE_SYMMETRY_MATRICES[:, 2, 2].abs() > 0.5
_CUBE_SYMMETRY_FLIPS_STACK = _CUBE_SYMMETRY_MATRICES[:, 2, 2] < -0.5


def _articulated_surface_geometry() -> tuple[Tensor, Tensor]:
    """Return a symmetric grid of points on the three slab surfaces."""
    points: list[tuple[float, float, float]] = []
    layers: list[int] = []
    grid = tuple(-1.5 + 0.5 * index for index in range(7))
    for layer_index, layer_coordinate in enumerate((-1.0, 0.0, 1.0)):
        z_grid = tuple(
            value for value in grid
            if layer_coordinate - 0.5 <= value <= layer_coordinate + 0.5
        )
        for x in (-1.5, 1.5):
            for y in grid:
                for z in z_grid:
                    points.append((x, y, z))
                    layers.append(layer_index)
        for y in (-1.5, 1.5):
            for x in grid:
                for z in z_grid:
                    points.append((x, y, z))
                    layers.append(layer_index)
        if layer_index == 0:
            for x in grid:
                for y in grid:
                    points.append((x, y, -1.5))
                    layers.append(layer_index)
        if layer_index == 2:
            for x in grid:
                for y in grid:
                    points.append((x, y, 1.5))
                    layers.append(layer_index)
    return torch.tensor(points, dtype=torch.float32), torch.tensor(layers, dtype=torch.long)


_ARTICULATED_SURFACE_POINTS, _ARTICULATED_SURFACE_LAYERS = _articulated_surface_geometry()


def _articulated_vertex_geometry() -> tuple[Tensor, Tensor]:
    """Return ordered vertices for the three articulated slabs."""
    points: list[tuple[float, float, float]] = []
    layers: list[int] = []
    for layer_index, layer_coordinate in enumerate((-1.0, 0.0, 1.0)):
        for z in (layer_coordinate - 0.5, layer_coordinate + 0.5):
            for y in (-1.5, 1.5):
                for x in (-1.5, 1.5):
                    points.append((x, y, z))
                    layers.append(layer_index)
    return torch.tensor(points, dtype=torch.float32), torch.tensor(layers, dtype=torch.long)


_ARTICULATED_VERTEX_POINTS, _ARTICULATED_VERTEX_LAYERS = _articulated_vertex_geometry()
_ARTICULATED_EDGE_INDEXES = torch.tensor(
    tuple(
        (start, end)
        for slab in range(3)
        for start, end in (
            (slab * 8 + 0, slab * 8 + 1),
            (slab * 8 + 1, slab * 8 + 3),
            (slab * 8 + 3, slab * 8 + 2),
            (slab * 8 + 2, slab * 8 + 0),
            (slab * 8 + 4, slab * 8 + 5),
            (slab * 8 + 5, slab * 8 + 7),
            (slab * 8 + 7, slab * 8 + 6),
            (slab * 8 + 6, slab * 8 + 4),
            (slab * 8 + 0, slab * 8 + 4),
            (slab * 8 + 1, slab * 8 + 5),
            (slab * 8 + 2, slab * 8 + 6),
            (slab * 8 + 3, slab * 8 + 7),
        )
    ),
    dtype=torch.long,
)

_LINE_SAMPLE_PARAMETERS = torch.linspace(0.0, 1.0, 9, dtype=torch.float32)
_ARTICULATED_LINE_POINTS = torch.stack(
    tuple(
        _ARTICULATED_VERTEX_POINTS[start]
        + parameter * (
            _ARTICULATED_VERTEX_POINTS[end] - _ARTICULATED_VERTEX_POINTS[start]
        )
        for start, end in _ARTICULATED_EDGE_INDEXES.tolist()
        for parameter in _LINE_SAMPLE_PARAMETERS
    )
).reshape(len(_ARTICULATED_EDGE_INDEXES), len(_LINE_SAMPLE_PARAMETERS), 3)
_ARTICULATED_LINE_LAYERS = _ARTICULATED_VERTEX_LAYERS[
    _ARTICULATED_EDGE_INDEXES[:, 0]
]


def _quaternion_multiply(first: Tensor, second: Tensor) -> Tensor:
    """Multiply xyzw quaternions with broadcasting over leading dimensions."""
    x1, y1, z1, w1 = first.unbind(dim=-1)
    x2, y2, z2, w2 = second.unbind(dim=-1)
    return torch.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        dim=-1,
    )


def _quaternion_conjugate(quaternion: Tensor) -> Tensor:
    """Return the inverse of a unit xyzw quaternion."""
    return torch.cat((-quaternion[..., :3], quaternion[..., 3:]), dim=-1)


def _quaternion_matrix(quaternion: Tensor) -> Tensor:
    """Convert normalized xyzw quaternions to row-batch rotation matrices."""
    x, y, z, w = quaternion.unbind(dim=-1)
    return torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def _wrap_degrees(values: Tensor, period: float) -> Tensor:
    """Wrap angles to the centered half-open interval ``[-period/2, period/2)``."""
    return torch.remainder(values + period / 2.0, period) - period / 2.0


def rotation_to_head_parameters(rotation: Tensor) -> dict[str, Tensor]:
    """Decode a canonical rotation into the loss-space head parameters.

    The local +Z axis is the articulated stack axis.  ``u`` is its
    colatitude from image zenith, ``v`` is azimuth around image zenith, and
    ``twist`` is the remaining in-plane rotation.  The construction matches
    the custom head convention, including the image coordinate convention
    where image-up is camera ``-Y``.
    """
    stack = rotation[..., :, 2]
    zenith = stack.new_tensor(IMAGE_ZENITH)
    image_right = stack.new_tensor((1.0, 0.0, 0.0))
    fallback_right = stack.new_tensor((0.0, 0.0, 1.0))
    cosine = (stack * zenith).sum(dim=-1).clamp(-1.0, 1.0)
    u = torch.acos(cosine) * (180.0 / torch.pi)
    v = torch.atan2(stack[..., 0], stack[..., 2]) * (180.0 / torch.pi)
    use_fallback = image_right.expand_as(stack).mul(stack).sum(dim=-1).abs() > 0.95
    reference = torch.where(use_fallback[..., None], fallback_right, image_right)
    projected_right = reference - (reference * stack).sum(dim=-1, keepdim=True) * stack
    local_x = F.normalize(projected_right, dim=-1, eps=1e-6)
    local_y = F.normalize(torch.cross(stack, local_x, dim=-1), dim=-1, eps=1e-6)
    rotation_x = rotation[..., :, 0]
    twist = torch.atan2(
        (rotation_x * local_y).sum(dim=-1),
        (rotation_x * local_x).sum(dim=-1),
    ) * (180.0 / torch.pi)
    return {
        "u": u,
        "v": _wrap_degrees(v, 360.0),
        "twist": _wrap_degrees(twist, ANGLE_PERIOD_DEGREES),
    }


def head_parameters_to_rotation(
    u: Tensor,
    v: Tensor,
    twist: Tensor,
) -> Tensor:
    """Construct camera-frame rotation matrices from direct angle outputs.

    This is the inverse of :func:`rotation_to_head_parameters` away from the
    unavoidable azimuth/twist coordinate singularities.  The shared basis is
    deliberately the same basis used by the label decoder so geometry loss
    does not need a predicted quaternion.
    """
    u_radians = u * (torch.pi / 180.0)
    v_radians = v * (torch.pi / 180.0)
    twist_radians = twist * (torch.pi / 180.0)

    stack = torch.stack(
        (
            u_radians.sin() * v_radians.sin(),
            -u_radians.cos(),
            u_radians.sin() * v_radians.cos(),
        ),
        dim=-1,
    )
    image_right = stack.new_tensor((1.0, 0.0, 0.0))
    fallback_right = stack.new_tensor((0.0, 0.0, 1.0))
    use_fallback = stack[..., 0].abs() > 0.95
    reference = torch.where(use_fallback[..., None], fallback_right, image_right)
    projected_right = reference - (reference * stack).sum(dim=-1, keepdim=True) * stack
    local_x = F.normalize(projected_right, dim=-1, eps=1e-6)
    local_y = F.normalize(torch.cross(stack, local_x, dim=-1), dim=-1, eps=1e-6)
    rotation_x = (
        twist_radians.cos()[..., None] * local_x
        + twist_radians.sin()[..., None] * local_y
    )
    rotation_y = torch.cross(stack, rotation_x, dim=-1)
    return torch.stack((rotation_x, rotation_y, stack), dim=-1)


def _pose_rotation_matrix(pose: Mapping[str, Tensor]) -> Tensor:
    """Return a pose rotation, preferring the direct angle representation."""
    if all(name in pose for name in ("u", "v", "twist")):
        return head_parameters_to_rotation(pose["u"], pose["v"], pose["twist"])
    # Backward compatibility for legacy target dictionaries and callers.  The
    # network itself no longer returns this field.
    quaternion = F.normalize(pose["rotation_quaternion"], dim=1, eps=1e-6)
    return _quaternion_matrix(quaternion)


def _canonical_target_rotation(target: Mapping[str, Tensor]) -> Tensor:
    """Resolve legacy label symmetries once, before direct or geometry loss.

    Native v6 labels already carry the canonical rotation.  Legacy v5 labels
    are resolved here without changing the dataset: moving samples use their
    recorded turning-face axis, while no-move samples choose the cube axis
    closest to image zenith.  The chosen discrete symmetry is a label
    interpretation step, not a min-over-symmetries training loss.
    """
    if "rotation_quaternion" not in target:
        if all(name in target for name in ("u", "v", "twist")):
            return rotation_matrix_to_quaternion(
                head_parameters_to_rotation(
                    target["u"], target["v"], target["twist"]
                )
            )
        raise KeyError("Pose target requires rotation_quaternion or u/v/twist")
    target_q = F.normalize(target["rotation_quaternion"], dim=1, eps=1e-6)
    canonical_flags = target.get("canonical_rotation")
    if canonical_flags is None:
        canonical_flags = torch.ones(
            target_q.shape[0], dtype=torch.bool, device=target_q.device
        )
    else:
        canonical_flags = canonical_flags.to(device=target_q.device, dtype=torch.bool)
    if bool(canonical_flags.all()):
        return target_q

    rotations = _quaternion_matrix(target_q)
    symmetries = _CUBE_SYMMETRY_MATRICES.to(
        device=target_q.device,
        dtype=target_q.dtype,
    )
    candidates = torch.einsum("bij,sjk->bsik", rotations, symmetries)
    candidate_stack = candidates[..., :, 2]
    zenith = target_q.new_tensor(IMAGE_ZENITH)
    colatitude = torch.acos(
        (candidate_stack * zenith).sum(dim=-1).clamp(-1.0, 1.0)
    )
    valid = torch.ones_like(colatitude, dtype=torch.bool)
    turn_present = target.get("turn_present")
    face_axis = target.get("turn_face_axis")
    turn_layer = target.get("turn_layer_index")
    if turn_present is not None and face_axis is not None and turn_layer is not None:
        turn_present = turn_present.to(device=target_q.device, dtype=torch.bool)
        face_axis = face_axis.to(device=target_q.device, dtype=target_q.dtype)
        turn_layer = turn_layer.to(device=target_q.device)
        transformed_face = torch.einsum("sji,bj->bsi", symmetries, face_axis)
        desired_sign = torch.where(turn_layer == 2, 1.0, -1.0).to(target_q.dtype)
        turn_valid = (
            transformed_face[..., 0].abs() < 1e-5
        ) & (
            transformed_face[..., 1].abs() < 1e-5
        ) & (
            (transformed_face[..., 2] * desired_sign[:, None] - 1.0).abs() < 1e-5
        )
        valid = torch.where(turn_present[:, None], turn_valid, valid)
    candidate_u = colatitude.masked_fill(~valid, float("inf"))
    selected = candidate_u.argmin(dim=1)
    selected_rotation = candidates[torch.arange(target_q.shape[0], device=target_q.device), selected]
    selected_quaternion = rotation_matrix_to_quaternion(selected_rotation)
    return torch.where(canonical_flags[:, None], target_q, selected_quaternion)


def rotation_matrix_to_quaternion(matrix: Tensor) -> Tensor:
    """Convert a batch of proper rotation matrices to xyzw quaternions."""
    # The branch-free formulation is stable for the small rotations used by
    # this dataset and keeps the operation differentiable for predictions.
    trace = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
    w = torch.sqrt((1.0 + trace).clamp_min(1e-8)) / 2.0
    denominator = (4.0 * w).clamp_min(1e-6)
    x = (matrix[..., 2, 1] - matrix[..., 1, 2]) / denominator
    y = (matrix[..., 0, 2] - matrix[..., 2, 0]) / denominator
    z = (matrix[..., 1, 0] - matrix[..., 0, 1]) / denominator
    return F.normalize(torch.stack((x, y, z, w), dim=-1), dim=-1, eps=1e-6)


def _project_articulated_points(
    pose: Mapping[str, Tensor],
    points: Tensor,
    layers: Tensor,
) -> Tensor:
    """Project ordered articulated-cube points to normalized image points."""
    rotation = _pose_rotation_matrix(pose)
    batch_size = rotation.shape[0]
    local_template = points.to(
        device=rotation.device,
        dtype=rotation.dtype,
    )
    layer_indexes = layers.to(device=rotation.device)
    local_points = local_template.unsqueeze(0).expand(batch_size, -1, -1)
    angle_indexes = torch.zeros_like(layer_indexes)
    angle_indexes = torch.where(layer_indexes == 2, torch.ones_like(angle_indexes), angle_indexes)
    angle_values = pose["layer_angles"][:, angle_indexes]
    angle_values = torch.where(
        layer_indexes[None, :] == 0,
        -angle_values,
        angle_values,
    )
    angle_values = torch.where(
        layer_indexes[None, :] == 1,
        torch.zeros_like(angle_values),
        angle_values,
    )
    angles = angle_values * (torch.pi / 180.0)
    cosines = angles.cos()
    sines = angles.sin()
    x = local_points[..., 0]
    y = local_points[..., 1]
    local_points = torch.stack(
        (
            cosines * x - sines * y,
            sines * x + cosines * y,
            local_points[..., 2],
        ),
        dim=-1,
    )
    camera_points = torch.einsum("bij,bnj->bni", rotation, local_points)
    denominator = 1.0 + (pose["perspective_strength"][:, None, None] / 3.0) * camera_points[..., 2:3]
    denominator = denominator.clamp_min(1e-4)
    return pose["image_center"][:, None, :] + pose["image_scale"][:, None, None] * camera_points[..., :2] / denominator


def _project_articulated_geometry(pose: Mapping[str, Tensor]) -> Tensor:
    return _project_articulated_points(
        pose,
        _ARTICULATED_SURFACE_POINTS,
        _ARTICULATED_SURFACE_LAYERS,
    )


def _project_articulated_vertices(pose: Mapping[str, Tensor]) -> Tensor:
    return _project_articulated_points(
        pose,
        _ARTICULATED_VERTEX_POINTS,
        _ARTICULATED_VERTEX_LAYERS,
    )


def _project_articulated_lines(pose: Mapping[str, Tensor]) -> Tensor:
    """Project the sampled 3x3x3 articulated-cube edge lines."""
    return _project_articulated_points(
        pose,
        _ARTICULATED_LINE_POINTS.reshape(-1, 3),
        _ARTICULATED_LINE_LAYERS.repeat_interleave(len(_LINE_SAMPLE_PARAMETERS)),
    ).reshape(
        _pose_rotation_matrix(pose).shape[0],
        len(_ARTICULATED_EDGE_INDEXES),
        len(_LINE_SAMPLE_PARAMETERS),
        2,
    )


def _projected_bbox_diagonal(points: Tensor) -> Tensor:
    bounds = points.amin(dim=(-3, -2)), points.amax(dim=(-3, -2))
    return (bounds[1] - bounds[0]).square().sum(dim=-1).sqrt().clamp_min(1e-4)


def cube_pose_geometry_loss(
    prediction: Mapping[str, Tensor],
    target: Mapping[str, Tensor],
    *,
    huber_beta: float = GEOMETRY_HUBER_BETA,
) -> Tensor:
    """Compare projected cube lines using bidirectional closest-line matching.

    The target and prediction are not paired by line index.  Each projected
    line is matched to its closest projected counterpart using sampled points
    and an orientation term, then the matching is made symmetric.  This keeps
    the geometry term sensitive to a wrong orientation while remaining
    tolerant of line ordering and the cube's 90-degree symmetries.
    """
    predicted_lines = _project_articulated_lines(prediction)
    target_lines = _project_articulated_lines(target)
    predicted_vectors = predicted_lines[:, :, -1] - predicted_lines[:, :, 0]
    target_vectors = target_lines[:, :, -1] - target_lines[:, :, 0]
    predicted_vectors = F.normalize(predicted_vectors, dim=-1, eps=1e-6)
    target_vectors = F.normalize(target_vectors, dim=-1, eps=1e-6)

    point_differences = (
        predicted_lines[:, :, None, :, :]
        - target_lines[:, None, :, :, :]
    )
    # Clamp before sqrt: without this, an exact line match has an undefined
    # derivative (0 / sqrt(0)) and poisons the whole training step with NaNs.
    point_distances = point_differences.square().sum(dim=-1).clamp_min(1e-12).sqrt()
    point_cost = F.smooth_l1_loss(
        point_distances,
        torch.zeros_like(point_distances),
        beta=huber_beta,
        reduction="none",
    ).mean(dim=3)
    direction_similarity = (
        predicted_vectors[:, :, None, :] * target_vectors[:, None, :, :]
    ).sum(dim=-1).abs().clamp(max=1.0)
    direction_cost = 1.0 - direction_similarity
    pairwise_cost = point_cost + LINE_DIRECTION_WEIGHT * direction_cost
    predicted_to_target = pairwise_cost.min(dim=2).values
    target_to_predicted = pairwise_cost.min(dim=1).values
    line_loss = 0.5 * (predicted_to_target.mean(dim=1) + target_to_predicted.mean(dim=1))
    normalization = _projected_bbox_diagonal(target_lines).detach()
    normalized_loss = line_loss / normalization
    return torch.where(
        normalized_loss.abs() < 1e-7,
        torch.zeros_like(normalized_loss),
        normalized_loss,
    ).mean()


def _pose_head_parameters(pose: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Return direct angle fields, decoding only legacy quaternion mappings."""
    if all(name in pose for name in ("u", "v", "twist")):
        return {
            "u": pose["u"],
            "v": pose["v"],
            "twist": pose["twist"],
        }
    # Backward compatibility for legacy target dictionaries and callers. The
    # network itself no longer returns this field.
    return rotation_to_head_parameters(_pose_rotation_matrix(pose))


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class FeatureProjection(nn.Sequential):
    """Project one YOLO feature map to a common width."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class CubePoseHead(nn.Module):
    """Decode YOLO features and cube/sticker soft masks into pose outputs.

    Three-channel soft masks are required inputs: cube union, sticker union,
    and sticker boundary. Spatial pyramid pooling is used
    instead of one global average pool. The
    1x1, 2x2, and 4x4 bins preserve enough coarse image layout for the center,
    projection, and orientation predictions while keeping the head independent
    of the backbone's exact feature-map sizes.
    """

    def __init__(
        self,
        feature_channels: Sequence[int],
        *,
        projection_channels: int = 96,
        hidden_channels: int = 384,
        dropout: float = 0.05,
        mask_channels: int = POSE_MASK_CHANNELS,
    ) -> None:
        super().__init__()
        if not feature_channels:
            raise ValueError("At least one feature map is required")
        self.feature_channels = tuple(int(value) for value in feature_channels)
        self.mask_channels = int(mask_channels)
        if self.mask_channels != POSE_MASK_CHANNELS:
            raise ValueError(
                "CubePoseHead requires three soft mask channels: "
                "cube union, sticker union, and sticker boundary"
            )
        self.projections = nn.ModuleList(
            FeatureProjection(channels, projection_channels)
            for channels in self.feature_channels
        )
        self.mask_projection = FeatureProjection(self.mask_channels, projection_channels)
        self.pool_bins = (1, 2, 4)
        pooled_width = projection_channels * sum(size * size for size in self.pool_bins)
        input_width = pooled_width * len(self.feature_channels)
        if self.mask_projection is not None:
            input_width += pooled_width
        self.mlp = nn.Sequential(
            nn.Linear(input_width, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.output = nn.Linear(hidden_channels, POSE_OUTPUT_DIM)

    def forward(
        self,
        feature_pyramid: Sequence[Tensor],
        soft_masks: Tensor,
    ) -> dict[str, Tensor]:
        if len(feature_pyramid) != len(self.projections):
            raise ValueError(
                "Feature pyramid length changed after initialization: "
                f"expected {len(self.projections)}, got {len(feature_pyramid)}"
            )

        pooled: list[Tensor] = []
        for projection, feature in zip(self.projections, feature_pyramid):
            projected = projection(feature)
            for bin_size in self.pool_bins:
                pooled.append(
                    F.adaptive_avg_pool2d(projected, output_size=(bin_size, bin_size))
                    .flatten(1)
                )
        if soft_masks is None:
            raise ValueError(
                "CubePoseHead requires three-channel soft segmentation maps: "
                "cube union, sticker union, and sticker boundary"
            )
        if soft_masks.ndim != 4 or soft_masks.shape[1] != self.mask_channels:
            raise ValueError(
                f"soft_masks must have shape [B, {self.mask_channels}, H, W]"
            )
        projected_masks = self.mask_projection(soft_masks)
        for bin_size in self.pool_bins:
            pooled.append(
                F.adaptive_avg_pool2d(
                    projected_masks,
                    output_size=(bin_size, bin_size),
                ).flatten(1)
            )
        raw = self.output(self.mlp(torch.cat(pooled, dim=1)))

        # These are direct angle outputs.  The tanh bounds match the wrapped
        # angle domains used by the labels while avoiding quaternion/acos
        # singularities in the training path.
        u = 90.0 * torch.sigmoid(raw[:, 0])
        v = 180.0 * torch.tanh(raw[:, 1])
        twist = ANGLE_LIMIT_DEGREES * torch.tanh(raw[:, 2])
        center = torch.sigmoid(raw[:, 3:5])
        scale = F.softplus(raw[:, 5:6]) + 1e-5
        perspective = F.softplus(raw[:, 6:7]) + 1e-5
        layer_angles = ANGLE_LIMIT_DEGREES * torch.tanh(raw[:, 7:9])
        return {
            "u": u,
            "v": v,
            "twist": twist,
            "image_center": center,
            "image_scale": scale.squeeze(1),
            "perspective_strength": perspective.squeeze(1),
            "layer_angles": layer_angles,
        }


def cube_pose_loss(
    prediction: Mapping[str, Tensor],
    target: Mapping[str, Tensor],
    *,
    rotation_weight: float = 1.0,
    center_weight: float = 2.0,
    scale_weight: float = 1.0,
    perspective_weight: float = 1.0,
    layer_angle_weight: float = 1.0,
    use_perspective: bool = True,
    geometry_weight: float = 0.4,
    head_weight: float = 0.6,
) -> dict[str, Tensor]:
    """Return direct canonical-head and closest-line geometry losses.

    The direct term is calculated in the same coordinates exposed by the
    custom head design: ``u`` (colatitude), wrapped ``v`` (azimuth), wrapped
    ``twist``, and the two wrapped layer angles. Legacy quaternion mappings
    are accepted for target/checkpoint compatibility, but the network itself
    predicts the angle fields directly.

    Legacy v5 labels are canonicalized once by ``_canonical_target_rotation``.
    There is no differentiable min over cube symmetries in the training loss.
    """
    if geometry_weight < 0.0 or head_weight < 0.0:
        raise ValueError("geometry_weight and head_weight must be non-negative")

    canonical_target = dict(target)
    canonical_target["rotation_quaternion"] = _canonical_target_rotation(target)
    target_q = F.normalize(canonical_target["rotation_quaternion"], dim=1, eps=1e-6)
    target_rotation = _quaternion_matrix(target_q)
    predicted_head = _pose_head_parameters(prediction)
    target_head = rotation_to_head_parameters(target_rotation)
    canonical_target.update(target_head)

    u = F.smooth_l1_loss(
        predicted_head["u"] / 90.0,
        target_head["u"] / 90.0,
    )
    v_residual = _wrap_degrees(predicted_head["v"] - target_head["v"], 360.0)
    v = F.smooth_l1_loss(v_residual / 180.0, torch.zeros_like(v_residual))
    twist_residual = _wrap_degrees(
        predicted_head["twist"] - target_head["twist"], ANGLE_PERIOD_DEGREES
    )
    twist = F.smooth_l1_loss(
        twist_residual / ANGLE_LIMIT_DEGREES,
        torch.zeros_like(twist_residual),
    )
    layer_residual = _wrap_degrees(
        prediction["layer_angles"] - canonical_target["layer_angles"],
        ANGLE_PERIOD_DEGREES,
    )
    layer_angles = F.smooth_l1_loss(
        layer_residual / ANGLE_LIMIT_DEGREES,
        torch.zeros_like(layer_residual),
    )
    rotation = (u + v + twist) / 3.0

    center = F.smooth_l1_loss(prediction["image_center"], canonical_target["image_center"])
    predicted_log_scale = torch.log(prediction["image_scale"].clamp_min(1e-6))
    target_log_scale = torch.log(canonical_target["image_scale"].clamp_min(1e-6))
    scale = F.smooth_l1_loss(predicted_log_scale, target_log_scale)

    predicted_log_perspective = torch.log(
        prediction["perspective_strength"].clamp_min(1e-6)
    )
    target_log_perspective = torch.log(
        canonical_target["perspective_strength"].clamp_min(1e-6)
    )
    perspective = F.smooth_l1_loss(predicted_log_perspective, target_log_perspective)

    zero = rotation.new_zeros(())
    weighted_perspective = perspective_weight * perspective if use_perspective else zero
    head = (
        rotation_weight * rotation
        + layer_angle_weight * layer_angles
        + center_weight * center
        + scale_weight * scale
        + weighted_perspective
    )
    geometry = cube_pose_geometry_loss(
        prediction,
        canonical_target,
    )
    total = geometry_weight * geometry + head_weight * head
    return {
        "total": total,
        "geometry": geometry,
        "head": head,
        "pose_auxiliary": head,
        "rotation": rotation,
        "u": u,
        "v": v,
        "twist": twist,
        "center": center,
        "scale": scale,
        "perspective": perspective,
        "layer_angles": layer_angles,
    }


class YOLOCubePoseModel(nn.Module):
    """Attach :class:`CubePoseHead` to an Ultralytics segmentation model."""

    def __init__(
        self,
        base_model: nn.Module,
        feature_channels: Sequence[int],
        *,
        freeze_base: bool = True,
        use_soft_masks: bool = True,
        mask_channels: int = POSE_MASK_CHANNELS,
        **head_kwargs: Any,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.freeze_base = bool(freeze_base)
        if not use_soft_masks:
            raise ValueError(
                "CubePoseModel requires cube, sticker, and sticker-boundary soft masks"
            )
        self.use_soft_masks = True
        self.mask_channels = int(mask_channels)
        if self.mask_channels != POSE_MASK_CHANNELS:
            raise ValueError(
                "CubePoseModel requires three soft mask channels: "
                "cube union, sticker union, and sticker boundary"
            )
        self.pose_head = CubePoseHead(
            feature_channels,
            mask_channels=self.mask_channels,
            **head_kwargs,
        )
        self._captured_features: tuple[Tensor, ...] | None = None

        layers = getattr(base_model, "model", None)
        if layers is None or len(layers) == 0:
            raise TypeError("base_model must expose a non-empty .model layer sequence")
        self.feature_tap = layers[-1]
        self._feature_hook = self.feature_tap.register_forward_pre_hook(self._capture_features)
        if self.freeze_base:
            self.freeze_backbone()

    def _capture_features(self, _module: nn.Module, inputs: tuple[Any, ...]) -> None:
        if len(inputs) != 1 or not isinstance(inputs[0], (list, tuple)):
            raise RuntimeError(
                "The final YOLO segmentation head did not receive a feature pyramid. "
                "Use a compatible Ultralytics segmentation model."
            )
        self._captured_features = tuple(inputs[0])

    @staticmethod
    def infer_feature_channels(
        base_model: nn.Module,
        input_size: int | tuple[int, int],
    ) -> tuple[int, ...]:
        if isinstance(input_size, int):
            height = width = input_size
        else:
            height, width = input_size
        parameters = list(base_model.parameters())
        device = parameters[0].device if parameters else torch.device("cpu")
        dtype = parameters[0].dtype if parameters else torch.float32
        layers = getattr(base_model, "model", None)
        if layers is None or len(layers) == 0:
            raise TypeError("base_model must expose a non-empty .model layer sequence")

        captured: list[Tensor] = []

        def capture(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
            if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
                captured[:] = list(inputs[0])

        handle = layers[-1].register_forward_pre_hook(capture)
        was_training = base_model.training
        base_model.eval()
        try:
            with torch.no_grad():
                base_model(torch.zeros(1, 3, height, width, device=device, dtype=dtype))
        finally:
            handle.remove()
            base_model.train(was_training)
        if not captured:
            raise RuntimeError("Could not capture feature maps from the YOLO segmentation head")
        return tuple(int(feature.shape[1]) for feature in captured)

    @classmethod
    def from_yolo(
        cls,
        yolo: Any,
        *,
        input_size: int | tuple[int, int] = 256,
        freeze_base: bool = True,
        **head_kwargs: Any,
    ) -> "YOLOCubePoseModel":
        base_model = getattr(yolo, "model", yolo)
        if not isinstance(base_model, nn.Module):
            raise TypeError("yolo must be an Ultralytics YOLO object or torch.nn.Module")
        feature_channels = cls.infer_feature_channels(base_model, input_size)
        return cls(
            base_model,
            feature_channels,
            freeze_base=freeze_base,
            **head_kwargs,
        )

    @classmethod
    def from_checkpoint(
        cls,
        model_path: str | Path,
        *,
        input_size: int | tuple[int, int] = 256,
        freeze_base: bool = True,
        **head_kwargs: Any,
    ) -> "YOLOCubePoseModel":
        from ultralytics import YOLO

        return cls.from_yolo(
            YOLO(str(model_path)),
            input_size=input_size,
            freeze_base=freeze_base,
            **head_kwargs,
        )

    @classmethod
    def from_pose_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        base_model_path: str | Path | None = None,
        map_location: str | torch.device = "cpu",
    ) -> "YOLOCubePoseModel":
        """Restore a trained pose head and its underlying YOLO checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        resolved_base_model = base_model_path or checkpoint.get("base_model")
        if resolved_base_model is None:
            raise ValueError("Pose checkpoint does not contain a base_model path")
        model = cls.from_checkpoint(
            resolved_base_model,
            input_size=int(checkpoint.get("imgsz", 256)),
            freeze_base=True,
            use_soft_masks=True,
            mask_channels=int(checkpoint.get("mask_channels", POSE_MASK_CHANNELS)),
        )
        model.load_state_dict(checkpoint["model"])
        return model

    def freeze_backbone(self) -> None:
        self.freeze_base = True
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)

    def unfreeze_backbone(self) -> None:
        self.freeze_base = False
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(True)

    def train(self, mode: bool = True) -> "YOLOCubePoseModel":
        super().train(mode)
        if self.freeze_base:
            self.base_model.eval()
        return self

    def forward(self, images_rgb: Tensor) -> dict[str, Tensor]:
        if images_rgb.ndim != 4 or images_rgb.shape[1] != 3:
            raise ValueError("images_rgb must have shape [B, 3, H, W]")
        self._captured_features = None
        if self.freeze_base:
            with torch.no_grad():
                base_output = self.base_model(images_rgb)
        else:
            base_output = self.base_model(images_rgb)
        if self._captured_features is None:
            raise RuntimeError("YOLO forward completed without producing a feature pyramid")
        soft_masks = self._soft_segmentation_maps(base_output)
        return self.pose_head(self._captured_features, soft_masks)

    @staticmethod
    def _soft_union(instance_masks: Tensor) -> Tensor:
        """Union soft instance masks without averaging away their coverage."""
        return 1.0 - (1.0 - instance_masks).prod(dim=1, keepdim=True)

    @staticmethod
    def _one_pixel_inner_border(instance_masks: Tensor) -> Tensor:
        """Return each instance's one-pixel inner border with no widening."""
        padded = F.pad(instance_masks, (1, 1, 1, 1), mode="replicate")
        eroded = -F.max_pool2d(-padded, kernel_size=3, stride=1)
        return (instance_masks - eroded).clamp_min(0.0)

    def _soft_segmentation_maps(self, base_output: Any) -> Tensor:
        """Build cube/sticker unions plus the raw one-pixel sticker border."""
        prediction = base_output
        if isinstance(prediction, (tuple, list)):
            prediction = next(
                (item for item in reversed(prediction) if isinstance(item, dict)),
                None,
            )
        if not isinstance(prediction, dict):
            raise RuntimeError("Could not find Ultralytics segmentation outputs for soft masks")
        proto = prediction.get("proto")
        coefficients = prediction.get("mask_coefficient")
        scores = prediction.get("scores")
        if not all(isinstance(value, Tensor) for value in (proto, coefficients, scores)):
            raise RuntimeError(
                "Ultralytics output does not contain proto, mask_coefficient, and scores"
            )
        if proto.ndim != 4 or coefficients.ndim != 3 or scores.ndim != 3:
            raise RuntimeError("Unexpected Ultralytics segmentation output shapes")
        if scores.shape[1] < SEGMENTATION_CLASS_CHANNELS:
            raise RuntimeError(
                f"Segmentation model exposes {scores.shape[1]} classes, "
                f"but at least {SEGMENTATION_CLASS_CHANNELS} cube/sticker classes are required"
            )
        mask_logits = torch.einsum("bkn,bkhw->bnhw", coefficients, proto)
        mask_probabilities = mask_logits.sigmoid()
        class_scores = scores[:, :SEGMENTATION_CLASS_CHANNELS].sigmoid()

        cube_instances = mask_probabilities * class_scores[:, 0, :, None, None]
        sticker_instances = mask_probabilities * class_scores[:, 1, :, None, None]
        cube_union = self._soft_union(cube_instances)
        sticker_union = self._soft_union(sticker_instances)

        # Compute the border from each sticker instance before unioning. This
        # preserves sticker-to-sticker boundaries even when their union is one
        # connected region. Replicate padding avoids inventing a border merely
        # because a sticker touches the image edge.
        sticker_borders = self._one_pixel_inner_border(mask_probabilities)
        sticker_borders = sticker_borders * class_scores[:, 1, :, None, None]
        sticker_boundary = self._soft_union(sticker_borders)
        return torch.cat((cube_union, sticker_union, sticker_boundary), dim=1)

    def close(self) -> None:
        self._feature_hook.remove()

    def __del__(self) -> None:
        hook = getattr(self, "_feature_hook", None)
        if hook is not None:
            hook.remove()

