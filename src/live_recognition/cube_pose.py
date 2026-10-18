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
head. It captures the multiscale feature pyramid immediately before an
Ultralytics segmentation head and trains a small pose decoder on those
features. An optional path adds the default head's selected sticker masks and
the prototype-derived cube and sticker unions. It does not consume rasterized
ground-truth masks, so either training path has the same information available
at inference time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
DEFAULT_STICKER_MASK_SLOTS = 64
PROTOTYPE_MASK_CHANNELS = 2
POSE_MASK_CHANNEL_NAMES = tuple(
    f"default_sticker_{index:02d}" for index in range(DEFAULT_STICKER_MASK_SLOTS)
) + (
    "cube_prototype_union",
    "sticker_prototype_union",
)
POSE_MASK_CHANNELS = len(POSE_MASK_CHANNEL_NAMES)
SEGMENTATION_CLASS_CHANNELS = 2
MASK_LOGIT_TEMPERATURE = 0.35
MASK_SCORE_GATE_THRESHOLD = 0.25
MASK_SCORE_GATE_TEMPERATURE = 0.10
MASK_BOX_CROP_TEMPERATURE = 0.50
ANGLE_LIMIT_DEGREES = 45.0
GEOMETRY_HUBER_BETA = 0.01
LINE_DIRECTION_WEIGHT = 0.25
ANGLE_PERIOD_DEGREES = 90.0
IMAGE_ZENITH = (0.0, -1.0, 0.0)
# The v7 training labels stay well inside these normalized projection ranges.
# Bounding them also keeps the perspective denominator away from zero during
# geometry-loss optimization.
IMAGE_SCALE_MAX = 0.5
PERSPECTIVE_STRENGTH_MAX = 1.0


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


def _wrap_degrees(values: Tensor, period: float) -> Tensor:
    """Wrap angles to the centered half-open interval ``[-period/2, period/2)``."""
    return torch.remainder(values + period / 2.0, period) - period / 2.0


def head_parameters_to_rotation(
    u: Tensor,
    v: Tensor,
    twist: Tensor,
) -> Tensor:
    """Construct camera-frame rotation matrices from direct angle outputs.

    This constructs the articulated cube orientation directly from the three
    angle outputs used by the pose head.
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
    # The projected reference is guaranteed to have a non-zero norm because
    # the fallback is selected before the reference becomes nearly parallel
    # to the stack. Use explicit bounded arithmetic here instead of the
    # device-specific normalize/cross kernels; this path is used for both
    # predictions and target labels during geometry loss.
    projected_norm = projected_right.square().sum(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
    local_x = projected_right / projected_norm
    local_y = torch.stack(
        (
            stack[..., 1] * local_x[..., 2] - stack[..., 2] * local_x[..., 1],
            stack[..., 2] * local_x[..., 0] - stack[..., 0] * local_x[..., 2],
            stack[..., 0] * local_x[..., 1] - stack[..., 1] * local_x[..., 0],
        ),
        dim=-1,
    )
    rotation_x = (
        twist_radians.cos()[..., None] * local_x
        + twist_radians.sin()[..., None] * local_y
    )
    rotation_y = torch.stack(
        (
            stack[..., 1] * rotation_x[..., 2] - stack[..., 2] * rotation_x[..., 1],
            stack[..., 2] * rotation_x[..., 0] - stack[..., 0] * rotation_x[..., 2],
            stack[..., 0] * rotation_x[..., 1] - stack[..., 1] * rotation_x[..., 0],
        ),
        dim=-1,
    )
    return torch.stack((rotation_x, rotation_y, stack), dim=-1)


def _pose_rotation_matrix(pose: Mapping[str, Tensor]) -> Tensor:
    """Return the rotation constructed from direct angle fields."""
    return head_parameters_to_rotation(pose["u"], pose["v"], pose["twist"])


def _project_articulated_points(
    pose: Mapping[str, Tensor],
    points: Tensor,
    layers: Tensor,
) -> Tensor:
    """Project ordered articulated-cube points to normalized image points."""
    for name in (
        "u",
        "v",
        "twist",
        "image_center",
        "image_scale",
        "perspective_strength",
        "layer_angles",
    ):
        _require_finite_geometry(f"pose_{name}", pose[name])
    rotation = _pose_rotation_matrix(pose)
    _require_finite_geometry("rotation", rotation)
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
    _require_finite_geometry("camera_points", camera_points)
    denominator = 1.0 + (pose["perspective_strength"][:, None, None] / 3.0) * camera_points[..., 2:3]
    denominator = denominator.clamp_min(1e-4)
    _require_finite_geometry("denominator", denominator)
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
    return _stable_l2_norm(bounds[1] - bounds[0], dim=-1).clamp_min(1e-4)


def _stable_l2_norm(value: Tensor, *, dim: int, eps: float = 1e-12) -> Tensor:
    """Compute an L2 norm without squaring large finite values directly."""
    scale = value.abs().amax(dim=dim, keepdim=True).clamp_min(eps)
    scaled = value / scale
    return scale.squeeze(dim) * (scaled.square().sum(dim=dim) + eps).sqrt()


def _stable_normalize(value: Tensor, *, dim: int, eps: float = 1e-12) -> Tensor:
    """Normalize vectors without overflowing their squared norm."""
    scale = value.abs().amax(dim=dim, keepdim=True).clamp_min(eps)
    scaled = value / scale
    norm = (scaled.square().sum(dim=dim, keepdim=True) + eps).sqrt()
    return scaled / norm


def _nonoverflow_huber(value: Tensor, *, beta: float) -> Tensor:
    """Compute Huber loss without evaluating an overflowing inactive branch."""
    positive_value = value.clamp_min(0.0)
    quadratic_ratio = (positive_value / beta).clamp_max(1.0)
    quadratic = 0.5 * beta * quadratic_ratio.square()
    linear = (positive_value - beta).clamp_min(0.0)
    return quadratic + linear


def _require_finite_geometry(name: str, value: Tensor) -> None:
    """Raise at the first non-finite geometry intermediate."""
    if not bool(torch.isfinite(value).all()):
        bad = (~torch.isfinite(value)).flatten()
        first_bad = int(bad.nonzero(as_tuple=False)[0].item())
        raise FloatingPointError(
            f"Non-finite geometry intermediate {name}; "
            f"shape={tuple(value.shape)}, first_bad_flat_index={first_bad}"
        )


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
    _require_finite_geometry("predicted_lines", predicted_lines)
    _require_finite_geometry("target_lines", target_lines)
    predicted_vector_differences = predicted_lines[:, :, -1] - predicted_lines[:, :, 0]
    target_vector_differences = target_lines[:, :, -1] - target_lines[:, :, 0]
    _require_finite_geometry(
        "predicted_vector_differences", predicted_vector_differences
    )
    _require_finite_geometry("target_vector_differences", target_vector_differences)
    predicted_vectors = _stable_normalize(predicted_vector_differences, dim=-1)
    target_vectors = _stable_normalize(target_vector_differences, dim=-1)
    _require_finite_geometry("predicted_vectors", predicted_vectors)
    _require_finite_geometry("target_vectors", target_vectors)

    point_differences = (
        predicted_lines[:, :, None, :, :]
        - target_lines[:, None, :, :, :]
    )
    _require_finite_geometry("point_differences", point_differences)
    # Rescale before squaring so a large, finite coordinate difference cannot
    # overflow in ``difference.square()``.  Epsilon inside sqrt keeps an exact
    # line match from evaluating the undefined derivative 0 / sqrt(0).
    point_distances = _stable_l2_norm(point_differences, dim=-1)
    _require_finite_geometry("point_distances", point_distances)
    point_cost = _nonoverflow_huber(
        point_distances,
        beta=huber_beta,
    ).mean(dim=3)
    _require_finite_geometry("point_cost", point_cost)
    direction_similarity = (
        predicted_vectors[:, :, None, :] * target_vectors[:, None, :, :]
    ).sum(dim=-1).abs().clamp(max=1.0)
    _require_finite_geometry("direction_similarity", direction_similarity)
    direction_cost = 1.0 - direction_similarity
    pairwise_cost = point_cost + LINE_DIRECTION_WEIGHT * direction_cost
    _require_finite_geometry("pairwise_cost", pairwise_cost)
    predicted_to_target = pairwise_cost.min(dim=2).values
    target_to_predicted = pairwise_cost.min(dim=1).values
    line_loss = 0.5 * (predicted_to_target.mean(dim=1) + target_to_predicted.mean(dim=1))
    normalization = _projected_bbox_diagonal(target_lines).detach()
    _require_finite_geometry("line_loss", line_loss)
    _require_finite_geometry("normalization", normalization)
    normalized_loss = line_loss / normalization
    _require_finite_geometry("normalized_loss", normalized_loss)
    return torch.where(
        normalized_loss.abs() < 1e-7,
        torch.zeros_like(normalized_loss),
        normalized_loss,
    ).mean()


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


class RawImageFeatureStage(nn.Sequential):
    """Small stride-2 convolutional stage for the raw-image pose baseline."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class CubePoseHead(nn.Module):
    """Decode YOLO features, optionally augmented with soft masks, into pose outputs.

    When enabled, the mask input contains a fixed stack of selected
    default-head sticker masks followed by prototype-derived cube and sticker
    unions. Spatial pyramid pooling is used instead of one global average pool.
    The 1x1, 2x2, and 4x4 bins preserve coarse image layout for the center,
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
        use_masks: bool = True,
    ) -> None:
        super().__init__()
        if not feature_channels:
            raise ValueError("At least one feature map is required")
        self.feature_channels = tuple(int(value) for value in feature_channels)
        self.use_masks = bool(use_masks)
        self.mask_channels = int(mask_channels)
        if self.use_masks and self.mask_channels not in {
            PROTOTYPE_MASK_CHANNELS,
            POSE_MASK_CHANNELS,
        }:
            raise ValueError(
                "CubePoseHead requires either the two prototype union channels "
                "or the full default-sticker mask stack"
            )
        self.projections = nn.ModuleList(
            FeatureProjection(channels, projection_channels)
            for channels in self.feature_channels
        )
        self.mask_projection = (
            FeatureProjection(self.mask_channels, projection_channels)
            if self.use_masks
            else None
        )
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
        soft_masks: Tensor | None = None,
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
        if self.mask_projection is not None:
            if soft_masks is None:
                raise ValueError(
                    "CubePoseHead configured with mask inputs but soft_masks was omitted"
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
        # Angle bounds match the direct v7 label domains.
        u = 90.0 * torch.sigmoid(raw[:, 0])
        v = 180.0 * torch.tanh(raw[:, 1])
        twist = ANGLE_LIMIT_DEGREES * torch.tanh(raw[:, 2])
        center = torch.sigmoid(raw[:, 3:5])
        scale = IMAGE_SCALE_MAX * torch.sigmoid(raw[:, 5:6]) + 1e-5
        perspective = PERSPECTIVE_STRENGTH_MAX * torch.sigmoid(raw[:, 6:7]) + 1e-5
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
    ``twist``, and the two wrapped layer angles.
    """
    if geometry_weight < 0.0 or head_weight < 0.0:
        raise ValueError("geometry_weight and head_weight must be non-negative")

    required_fields = ("u", "v", "twist", "layer_angles")
    missing_prediction = [name for name in required_fields if name not in prediction]
    missing_target = [name for name in required_fields if name not in target]
    if missing_prediction or missing_target:
        raise ValueError(
            "cube_pose_loss requires direct angle fields; "
            f"missing prediction={missing_prediction}, target={missing_target}"
        )
    canonical_target = dict(target)
    target_head = {name: target[name] for name in ("u", "v", "twist")}
    predicted_head = {name: prediction[name] for name in ("u", "v", "twist")}
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
        self.use_soft_masks = bool(use_soft_masks)
        self.mask_channels = int(mask_channels)
        if self.use_soft_masks and self.mask_channels not in {
            PROTOTYPE_MASK_CHANNELS,
            POSE_MASK_CHANNELS,
        }:
            raise ValueError(
                "CubePoseModel requires either the two prototype union channels "
                "or the full default-sticker mask stack"
            )
        self.pose_head = CubePoseHead(
            feature_channels,
            mask_channels=self.mask_channels,
            use_masks=self.use_soft_masks,
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
        soft_masks = None
        if self.use_soft_masks:
            soft_masks = self._soft_segmentation_maps(
                base_output,
                image_size=images_rgb.shape[-2:],
            )
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

    @staticmethod
    def _soft_box_crop(
        mask_values: Tensor,
        boxes: Tensor,
        image_size: tuple[int, int],
    ) -> Tensor:
        """Approximate Ultralytics' box crop while retaining box gradients."""
        if boxes.ndim != 3 or boxes.shape[1] != 4:
            raise RuntimeError("Expected decoded xywh boxes with shape [B, 4, N]")
        height, width = image_size
        _, _, mask_height, mask_width = mask_values.shape
        boxes = boxes.transpose(1, 2)
        centers = boxes[..., :2]
        half_sizes = boxes[..., 2:].clamp_min(0.0) / 2.0
        x1 = ((centers[..., 0] - half_sizes[..., 0]) * mask_width / width)[..., None, None]
        x2 = ((centers[..., 0] + half_sizes[..., 0]) * mask_width / width)[..., None, None]
        y1 = ((centers[..., 1] - half_sizes[..., 1]) * mask_height / height)[..., None, None]
        y2 = ((centers[..., 1] + half_sizes[..., 1]) * mask_height / height)[..., None, None]
        temperature = MASK_BOX_CROP_TEMPERATURE
        grid_x = torch.arange(mask_width, device=mask_values.device, dtype=mask_values.dtype)[None, None, None, :]
        grid_y = torch.arange(mask_height, device=mask_values.device, dtype=mask_values.dtype)[None, None, :, None]
        crop_x = torch.sigmoid((grid_x - x1) / temperature) * torch.sigmoid((x2 - grid_x) / temperature)
        crop_y = torch.sigmoid((grid_y - y1) / temperature) * torch.sigmoid((y2 - grid_y) / temperature)
        return mask_values * crop_x * crop_y

    def _default_sticker_masks(
        self,
        base_output: Any,
        prediction: Mapping[str, Tensor],
        sticker_instances: Tensor,
        image_size: tuple[int, int],
    ) -> Tensor:
        """Return fixed-width, default-head-style sticker mask outputs.

        The default segmentation head produces a variable number of masks
        after confidence filtering and NMS.  The pose head receives the top
        ``DEFAULT_STICKER_MASK_SLOTS`` sticker masks sorted by confidence and
        zero-padded when fewer are present.  These masks are intentionally
        postprocessed and detached: the default segmentation model is frozen
        for this training path.
        """
        batch_size, _, mask_height, mask_width = sticker_instances.shape
        selected_masks = sticker_instances.new_zeros(
            batch_size,
            DEFAULT_STICKER_MASK_SLOTS,
            mask_height,
            mask_width,
        )
        raw_predictions = None
        if isinstance(base_output, (tuple, list)):
            for item in base_output:
                if (
                    isinstance(item, (tuple, list))
                    and item
                    and isinstance(item[0], Tensor)
                    and item[0].ndim == 3
                    and item[0].shape[1] >= 6
                ):
                    raw_predictions = item[0]
                    break

        if raw_predictions is None:
            # Small fake/test models may expose only reconstructed candidates.
            # Preserve a deterministic fallback for those models.
            sticker_scores = sticker_instances.flatten(2).amax(dim=2)
            order = sticker_scores.argsort(dim=1, descending=True)
            count = min(DEFAULT_STICKER_MASK_SLOTS, sticker_instances.shape[1])
            selected_masks[:, :count] = sticker_instances.gather(
                1,
                order[:, :count, None, None].expand(
                    -1,
                    -1,
                    mask_height,
                    mask_width,
                ),
            )
            return selected_masks

        from ultralytics.utils.nms import non_max_suppression
        from ultralytics.utils.ops import process_mask

        proto = prediction["proto"]
        with torch.no_grad():
            detections = non_max_suppression(
                raw_predictions,
                conf_thres=0.25,
                iou_thres=0.70,
                nc=SEGMENTATION_CLASS_CHANNELS,
                max_det=300,
            )
            for batch_index, detection in enumerate(detections):
                if detection.numel() == 0:
                    continue
                stickers = detection[detection[:, 5].long() == 1]
                if stickers.numel() == 0:
                    continue
                stickers = stickers[stickers[:, 4].argsort(descending=True)]
                stickers = stickers[:DEFAULT_STICKER_MASK_SLOTS]
                masks = process_mask(
                    proto[batch_index],
                    stickers[:, 6:],
                    stickers[:, :4],
                    image_size,
                    upsample=True,
                ).float()
                masks = F.adaptive_avg_pool2d(
                    masks[:, None],
                    output_size=(mask_height, mask_width),
                ).squeeze(1)
                selected_masks[batch_index, : masks.shape[0]] = masks
        return selected_masks

    def _soft_segmentation_maps(
        self,
        base_output: Any,
        *,
        image_size: tuple[int, int] | None = None,
    ) -> Tensor:
        """Build default sticker outputs and prototype-derived unions."""
        prediction = base_output
        decoded_boxes = None
        if isinstance(base_output, (tuple, list)):
            for item in base_output:
                if (
                    isinstance(item, (tuple, list))
                    and item
                    and isinstance(item[0], Tensor)
                    and item[0].ndim == 3
                    and item[0].shape[1] >= 4
                ):
                    decoded_boxes = item[0][:, :4]
                    break
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
        boxes = decoded_boxes if decoded_boxes is not None else prediction.get("boxes")
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
        mask_probabilities = (mask_logits / MASK_LOGIT_TEMPERATURE).sigmoid()
        if boxes is not None:
            if image_size is None:
                raise RuntimeError("image_size is required when decoded boxes are present")
            mask_probabilities = self._soft_box_crop(mask_probabilities, boxes, image_size)
        class_scores = scores[:, :SEGMENTATION_CLASS_CHANNELS].sigmoid()
        confidence_gates = torch.sigmoid(
            (class_scores - MASK_SCORE_GATE_THRESHOLD) / MASK_SCORE_GATE_TEMPERATURE
        )

        cube_instances = mask_probabilities * confidence_gates[:, 0, :, None, None]
        sticker_instances = mask_probabilities * confidence_gates[:, 1, :, None, None]
        cube_union = self._soft_union(cube_instances)
        sticker_union = self._soft_union(sticker_instances)
        if self.mask_channels == PROTOTYPE_MASK_CHANNELS:
            return torch.cat((cube_union, sticker_union), dim=1)
        default_sticker_masks = self._default_sticker_masks(
            base_output,
            prediction,
            sticker_instances,
            image_size=(
                image_size if image_size is not None else (proto.shape[-2], proto.shape[-1])
            ),
        )
        return torch.cat((default_sticker_masks, cube_union, sticker_union), dim=1)

    def close(self) -> None:
        self._feature_hook.remove()

    def __del__(self) -> None:
        hook = getattr(self, "_feature_hook", None)
        if hook is not None:
            hook.remove()


class RawImageCubePoseModel(nn.Module):
    """Raw-RGB-only pose baseline using a compact spatial CNN."""

    def __init__(
        self,
        *,
        feature_channels: Sequence[int] = (64, 128, 256),
        projection_channels: int = 96,
        hidden_channels: int = 384,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if tuple(feature_channels) != (64, 128, 256):
            raise ValueError("RawImageCubePoseModel expects feature channels (64, 128, 256)")
        self.feature_channels = tuple(int(value) for value in feature_channels)
        self.image_stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(32), 32),
            nn.SiLU(inplace=True),
        )
        self.stage2 = RawImageFeatureStage(32, 64)
        self.stage3 = RawImageFeatureStage(64, 64)
        self.stage4 = RawImageFeatureStage(64, 128)
        self.stage5 = RawImageFeatureStage(128, 256)
        self.pose_head = CubePoseHead(
            self.feature_channels,
            projection_channels=projection_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
            use_masks=False,
        )

    def forward(self, images_rgb: Tensor) -> dict[str, Tensor]:
        if images_rgb.ndim != 4 or images_rgb.shape[1] != 3:
            raise ValueError("images_rgb must have shape [B, 3, H, W]")
        features = self.image_stem(images_rgb)
        features = self.stage2(features)
        p3 = self.stage3(features)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)
        return self.pose_head((p3, p4, p5))

    def close(self) -> None:
        """Match the YOLO pose wrapper lifecycle for shared training code."""

