"""Dense color prediction on top of an Ultralytics YOLO segmentation model.

The base YOLO model is kept intact.  A pre-hook captures the multiscale feature
pyramid immediately before the existing segmentation head, and a small PyTorch
decoder combines those features with explicit color features computed from the
ROI image, including RGB, chromaticity, luminance, saturation, and OkLab.

The model expects RGB tensors in ``[0, 1]`` with shape ``[B, 3, H, W]``.  It
returns seven-channel logits/probabilities in the same spatial resolution:

    white, yellow, red, orange, blue, green, none

``IGNORE_INDEX`` is intended for target pixels whose label is uncertain; it is
not an output class.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


COLOR_CLASS_NAMES = ("white", "yellow", "red", "orange", "blue", "green", "none")
COLOR_CLASS_COUNT = len(COLOR_CLASS_NAMES)
NONE_CLASS_INDEX = COLOR_CLASS_NAMES.index("none")
IGNORE_INDEX = 255


def _group_count(channels: int, maximum: int = 8) -> int:
    """Choose a stable GroupNorm count for a channel width."""

    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    """Small convolution block that remains well behaved for small batches."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


def _srgb_to_linear(images_rgb: Tensor) -> Tensor:
    """Convert normalized sRGB values to linear RGB for OkLab conversion."""

    return torch.where(
        images_rgb <= 0.04045,
        images_rgb / 12.92,
        ((images_rgb + 0.055) / 1.055).clamp_min(0.0).pow(2.4),
    )


def _oklab_features(images_rgb: Tensor) -> Tensor:
    """Return normalized sRGB converted to the three OkLab channels."""

    linear_rgb = _srgb_to_linear(images_rgb)
    rgb_to_lms = images_rgb.new_tensor(
        (
            (0.4122214708, 0.5363325363, 0.0514459929),
            (0.2119034982, 0.6806995451, 0.1073969566),
            (0.0883024619, 0.2817188376, 0.6299787005),
        )
    )
    lms_to_oklab = images_rgb.new_tensor(
        (
            (0.2104542553, 0.7936177850, -0.0040720468),
            (1.9779984951, -2.4285922050, 0.4505937099),
            (0.0259040371, 0.7827717662, -0.8086757660),
        )
    )
    lms = torch.einsum("ij,bjhw->bihw", rgb_to_lms, linear_rgb)
    lms_cbrt = torch.sign(lms) * lms.abs().clamp_min(1e-8).pow(1.0 / 3.0)
    return torch.einsum("ij,bjhw->bihw", lms_to_oklab, lms_cbrt)


def explicit_color_features(images_rgb: Tensor) -> Tensor:
    """Build differentiable color features from an RGB image tensor.

    In addition to RGB, the branch sees chromaticity, luminance, saturation, and
    OkLab. OkLab provides a more perceptually organized color representation,
    while the raw RGB and simpler features preserve information that may still
    be useful for the learned head.
    These are deliberately simple features: the learned head can decide which
    representation is useful, while the raw RGB skip preserves exact local color.
    """

    rgb = images_rgb.float().clamp(0.0, 1.0)
    channel_sum = rgb.sum(dim=1, keepdim=True)
    chromaticity = rgb / (channel_sum + 1e-4)
    luminance = (
        0.2126 * rgb[:, 0:1]
        + 0.7152 * rgb[:, 1:2]
        + 0.0722 * rgb[:, 2:3]
    )
    maximum = rgb.amax(dim=1, keepdim=True)
    minimum = rgb.amin(dim=1, keepdim=True)
    saturation = maximum - minimum
    oklab = _oklab_features(rgb)
    return torch.cat((rgb, chromaticity, luminance, saturation, oklab), dim=1)


class DenseColorDecoder(nn.Module):
    """Fuse YOLO's multiscale features with an image color branch."""

    color_feature_channels = 11

    def __init__(
        self,
        feature_channels: Sequence[int],
        *,
        class_count: int = COLOR_CLASS_COUNT,
        local_channels: int = 48,
        color_channels: int = 32,
        global_channels: int = 32,
        decoder_channels: int = 128,
    ) -> None:
        super().__init__()
        if len(feature_channels) < 1:
            raise ValueError("At least one YOLO feature map is required")

        self.feature_channels = tuple(int(value) for value in feature_channels)
        self.local_projections = nn.ModuleList(
            ConvNormAct(channels, local_channels, kernel_size=1)
            for channels in self.feature_channels
        )
        self.color_stem = nn.Sequential(
            ConvNormAct(self.color_feature_channels, color_channels),
            ConvNormAct(color_channels, color_channels),
        )
        self.global_projection = ConvNormAct(
            self.feature_channels[-1], global_channels, kernel_size=1
        )

        fused_channels = local_channels * len(self.feature_channels) + color_channels + global_channels
        self.fusion = nn.Sequential(
            ConvNormAct(fused_channels, decoder_channels),
            ConvNormAct(decoder_channels, decoder_channels),
        )
        self.decoder = nn.ModuleList(
            (
                ConvNormAct(decoder_channels, decoder_channels // 2),
                ConvNormAct(decoder_channels // 2, decoder_channels // 4),
                ConvNormAct(decoder_channels // 4, decoder_channels // 4),
            )
        )
        high_resolution_channels = decoder_channels // 4
        self.high_resolution_fusion = nn.Sequential(
            ConvNormAct(high_resolution_channels + color_channels, high_resolution_channels),
            ConvNormAct(high_resolution_channels, high_resolution_channels),
        )
        self.classifier = nn.Conv2d(high_resolution_channels, class_count, kernel_size=1)

    def forward(self, images_rgb: Tensor, feature_pyramid: Sequence[Tensor]) -> Tensor:
        if len(feature_pyramid) != len(self.local_projections):
            raise ValueError(
                "Feature pyramid length changed after initialization: "
                f"expected {len(self.local_projections)}, got {len(feature_pyramid)}"
            )

        reference_size = feature_pyramid[0].shape[-2:]
        local_features = []
        for projection, feature in zip(self.local_projections, feature_pyramid):
            projected = projection(feature)
            if projected.shape[-2:] != reference_size:
                projected = F.interpolate(
                    projected, size=reference_size, mode="bilinear", align_corners=False
                )
            local_features.append(projected)

        color_features = self.color_stem(explicit_color_features(images_rgb))
        color_at_reference = F.interpolate(
            color_features, size=reference_size, mode="bilinear", align_corners=False
        )

        global_feature = self.global_projection(feature_pyramid[-1])
        global_feature = F.adaptive_avg_pool2d(global_feature, output_size=1)
        global_feature = global_feature.expand(-1, -1, reference_size[0], reference_size[1])

        x = self.fusion(torch.cat((*local_features, color_at_reference, global_feature), dim=1))
        for decoder_layer in self.decoder:
            next_size = (max(1, x.shape[-2] * 2), max(1, x.shape[-1] * 2))
            x = F.interpolate(x, size=next_size, mode="bilinear", align_corners=False)
            x = decoder_layer(x)

        x = F.interpolate(x, size=images_rgb.shape[-2:], mode="bilinear", align_corners=False)
        color_high_resolution = F.interpolate(
            color_features, size=images_rgb.shape[-2:], mode="bilinear", align_corners=False
        )
        x = self.high_resolution_fusion(torch.cat((x, color_high_resolution), dim=1))
        return self.classifier(x)


class YOLOColorMapModel(nn.Module):
    """A dense color head attached to an existing YOLO segmentation model.

    ``base_model`` is the raw Ultralytics segmentation network, i.e.
    ``YOLO(...).model`` rather than the high-level ``YOLO`` wrapper.  The
    final segmentation head receives a list of multiscale feature maps; a
    forward pre-hook captures that list for the color decoder.
    """

    def __init__(
        self,
        base_model: nn.Module,
        feature_channels: Sequence[int],
        *,
        class_count: int = COLOR_CLASS_COUNT,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.freeze_base = bool(freeze_base)
        self.color_decoder = DenseColorDecoder(
            feature_channels,
            class_count=class_count,
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

    @classmethod
    def from_yolo(
        cls,
        yolo: Any,
        *,
        input_size: int | tuple[int, int] = 256,
        class_count: int = COLOR_CLASS_COUNT,
        freeze_base: bool = True,
    ) -> "YOLOColorMapModel":
        """Build the wrapper from an Ultralytics ``YOLO`` object."""

        base_model = getattr(yolo, "model", yolo)
        if not isinstance(base_model, nn.Module):
            raise TypeError("yolo must be an Ultralytics YOLO object or torch.nn.Module")
        feature_channels = cls.infer_feature_channels(base_model, input_size)
        return cls(
            base_model,
            feature_channels,
            class_count=class_count,
            freeze_base=freeze_base,
        )

    @classmethod
    def from_checkpoint(
        cls,
        model_path: str | Path,
        *,
        input_size: int | tuple[int, int] = 256,
        class_count: int = COLOR_CLASS_COUNT,
        freeze_base: bool = True,
    ) -> "YOLOColorMapModel":
        """Load a YOLO segmentation checkpoint and attach the color decoder."""

        from ultralytics import YOLO

        return cls.from_yolo(
            YOLO(str(model_path)),
            input_size=input_size,
            class_count=class_count,
            freeze_base=freeze_base,
        )

    @classmethod
    def from_color_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        base_model_path: str | Path | None = None,
        map_location: str | torch.device = "cpu",
    ) -> "YOLOColorMapModel":
        """Restore a trained color head and its underlying YOLO checkpoint."""

        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        resolved_base_model = base_model_path or checkpoint.get("base_model")
        if resolved_base_model is None:
            raise ValueError("Color checkpoint does not contain a base_model path")
        model = cls.from_checkpoint(
            resolved_base_model,
            input_size=int(checkpoint.get("imgsz", 256)),
            class_count=len(checkpoint.get("class_names", COLOR_CLASS_NAMES)),
            freeze_base=True,
        )
        model.load_state_dict(checkpoint["model"])
        return model

    @staticmethod
    def infer_feature_channels(
        base_model: nn.Module,
        input_size: int | tuple[int, int],
    ) -> tuple[int, ...]:
        """Infer the channel widths of the feature pyramid without assumptions about YOLO size."""

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

    def freeze_backbone(self) -> None:
        """Freeze YOLO weights while keeping the custom decoder trainable."""

        self.freeze_base = True
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)

    def unfreeze_backbone(self) -> None:
        """Allow joint fine-tuning of YOLO and the color decoder."""

        self.freeze_base = False
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(True)

    def train(self, mode: bool = True) -> "YOLOColorMapModel":
        """Keep a frozen YOLO feature extractor in eval mode during head training."""

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
                self.base_model(images_rgb)
        else:
            self.base_model(images_rgb)

        if self._captured_features is None:
            raise RuntimeError("YOLO forward completed without producing a feature pyramid")

        logits = self.color_decoder(images_rgb, self._captured_features)
        return {
            "color_logits": logits,
            "color_probabilities": torch.softmax(logits, dim=1),
        }

    def close(self) -> None:
        """Remove the internal feature hook when the wrapper is no longer needed."""

        self._feature_hook.remove()

    def __del__(self) -> None:
        hook = getattr(self, "_feature_hook", None)
        if hook is not None:
            hook.remove()
