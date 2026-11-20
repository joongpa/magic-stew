"""Adapters for the pretrained M-LSD, LINEA, and F-Clip line detectors.

The adapters deliberately expose only visible line segments. They do not try
to complete occluded cube edges or infer a pose.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from types import SimpleNamespace

import cv2
import numpy as np


@dataclass(frozen=True)
class LineDetections:
    """Line segments in the coordinate system of the input ROI."""

    lines_xyxy: np.ndarray
    scores: np.ndarray

    @property
    def count(self) -> int:
        return int(len(self.lines_xyxy))

    @property
    def bounds_xyxy(self) -> tuple[int, int, int, int] | None:
        if self.count == 0:
            return None
        points = self.lines_xyxy.reshape(-1, 2)
        x_min, y_min = np.floor(points.min(axis=0)).astype(int)
        x_max, y_max = np.ceil(points.max(axis=0)).astype(int)
        return int(x_min), int(y_min), int(x_max), int(y_max)


class LineDetector(Protocol):
    def detect(self, roi_bgr: np.ndarray) -> LineDetections:
        """Return visible line segments in ROI-local coordinates."""


def _empty_detections() -> LineDetections:
    return LineDetections(
        lines_xyxy=np.empty((0, 4), dtype=np.float32),
        scores=np.empty((0,), dtype=np.float32),
    )


def _filter_lines(
    lines: np.ndarray,
    scores: np.ndarray,
    *,
    min_length: float,
    max_lines: int,
) -> LineDetections:
    lines = np.asarray(lines, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(lines) == 0:
        return _empty_detections()
    lengths = np.linalg.norm(lines[:, 2:] - lines[:, :2], axis=1)
    keep = np.isfinite(lines).all(axis=1) & np.isfinite(scores) & (lengths >= min_length)
    lines = lines[keep]
    scores = scores[keep]
    if len(lines) > max_lines:
        order = np.argsort(scores)[::-1][:max_lines]
        lines = lines[order]
        scores = scores[order]
    return LineDetections(lines_xyxy=lines, scores=scores)


class MlsdTFLiteDetector:
    """Run the official M-LSD TFLite model.

    The official model emits a center-point heatmap and endpoint displacement
    map. The parser follows NAVER's released ``pred_lines`` representation but
    maps directly from the model output resolution to the input ROI size.
    """

    def __init__(
        self,
        model_path: Path,
        *,
        input_size: int = 512,
        score_threshold: float = 0.10,
        min_length: float = 5.0,
        max_lines: int = 500,
    ) -> None:
        try:
            import tensorflow as tf
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError as exc:
                try:
                    from ai_edge_litert.interpreter import Interpreter
                except ImportError:
                    raise RuntimeError(
                        "M-LSD requires tensorflow, tflite-runtime, or ai-edge-litert. "
                        "Install one of them before running the M-LSD baseline."
                    ) from exc
            self.interpreter = Interpreter(model_path=str(model_path))
        else:
            self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_size = int(input_size or 512)
        self.score_threshold = float(score_threshold)
        self.min_length = float(min_length)
        self.max_lines = int(max_lines)

    def detect(self, roi_bgr: np.ndarray) -> LineDetections:
        height, width = roi_bgr.shape[:2]
        if height <= 0 or width <= 0:
            return _empty_detections()
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            roi_rgb,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_AREA,
        )
        rgba = np.concatenate(
            [resized, np.ones((self.input_size, self.input_size, 1), dtype=np.uint8)],
            axis=-1,
        )
        input_info = self.input_details[0]
        input_data = rgba.astype(np.float32)[None, ...]
        if input_info["dtype"] != input_data.dtype:
            scale, zero_point = input_info.get("quantization", (0.0, 0))
            if scale:
                input_data = np.round(input_data / scale + zero_point).astype(input_info["dtype"])
            else:
                input_data = input_data.astype(input_info["dtype"])
        self.interpreter.set_tensor(input_info["index"], input_data)
        self.interpreter.invoke()

        outputs = [self.interpreter.get_tensor(detail["index"]) for detail in self.output_details]
        if len(outputs) < 3:
            raise RuntimeError("M-LSD model must expose point, score, and displacement outputs")
        points = np.asarray(outputs[0])[0]
        point_scores = np.asarray(outputs[1])[0]
        displacement = np.asarray(outputs[2])[0]
        if displacement.ndim != 3 or displacement.shape[-1] < 4:
            return _empty_detections()
        map_height, map_width = displacement.shape[:2]
        lines = []
        scores = []
        for center, score in zip(points, point_scores):
            y, x = np.rint(center).astype(int)
            if not (0 <= y < map_height and 0 <= x < map_width):
                continue
            score = float(np.asarray(score).reshape(-1)[0])
            if score < self.score_threshold:
                continue
            dx1, dy1, dx2, dy2 = displacement[y, x, :4]
            x1 = (x + float(dx1)) * width / map_width
            y1 = (y + float(dy1)) * height / map_height
            x2 = (x + float(dx2)) * width / map_width
            y2 = (y + float(dy2)) * height / map_height
            lines.append((x1, y1, x2, y2))
            scores.append(score)
        return _filter_lines(
            np.asarray(lines, dtype=np.float32),
            np.asarray(scores, dtype=np.float32),
            min_length=self.min_length,
            max_lines=self.max_lines,
        )


class MlsdTorchDetector:
    """Run a PyTorch M-LSD checkpoint, including fine-tuned checkpoints."""

    def __init__(
        self,
        model_path: Path,
        repo_path: Path,
        *,
        input_size: int | None = None,
        score_threshold: float = 0.10,
        min_length: float = 5.0,
        max_lines: int = 500,
        device: str = "auto",
    ) -> None:
        import torch

        repo_path = repo_path.resolve()
        if not repo_path.is_dir():
            raise FileNotFoundError(f"Could not find PyTorch M-LSD repository: {repo_path}")
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))
        try:
            from mlsd_pytorch.models.build_model import build_model
            from mlsd_pytorch.utils.decode import deccode_lines_TP
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch M-LSD requires the cloned mlsd_pytorch repository and its dependencies."
            ) from exc

        checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
        checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
        if not isinstance(checkpoint_args, dict):
            checkpoint_args = vars(checkpoint_args) if hasattr(checkpoint_args, "__dict__") else {}
        model_name = str(checkpoint_args.get("model", "mobilev2_mlsd_large"))
        state = checkpoint
        if isinstance(checkpoint, dict):
            if isinstance(checkpoint.get("model"), dict):
                state = checkpoint["model"]
            elif isinstance(checkpoint.get("state_dict"), dict):
                state = checkpoint["state_dict"]
        if not isinstance(state, dict):
            raise RuntimeError(f"Unsupported PyTorch M-LSD checkpoint format: {model_path}")
        if state and all(str(key).startswith("module.") for key in state):
            state = {str(key)[len("module.") :]: value for key, value in state.items()}

        cfg = SimpleNamespace(model=SimpleNamespace(model_name=model_name, with_deconv=True))
        # The repository's constructor attempts to download ImageNet weights;
        # the supplied M-LSD checkpoint already contains the complete model.
        import torch.utils.model_zoo as model_zoo

        original_load_url = model_zoo.load_url
        model_zoo.load_url = lambda *_args, **_kwargs: {}
        try:
            model = build_model(cfg)
        finally:
            model_zoo.load_url = original_load_url

        # The released M-LSD checkpoint uses an RGBA stem while the PyTorch
        # source defines an RGB stem. The alpha weights are zero, so preserve
        # the RGB initialization and add the checkpoint's fourth channel.
        stem = model.backbone.features[0][0]
        checkpoint_stem = state.get("backbone.features.0.0.weight")
        if checkpoint_stem is not None and checkpoint_stem.shape[1] != stem.in_channels:
            if checkpoint_stem.shape[1] != 4 or stem.in_channels != 3:
                raise RuntimeError(
                    "Unsupported M-LSD input stem: "
                    f"checkpoint has {checkpoint_stem.shape[1]} channels, model has {stem.in_channels}"
                )
            rgba_stem = torch.nn.Conv2d(
                4,
                stem.out_channels,
                kernel_size=stem.kernel_size,
                stride=stem.stride,
                padding=stem.padding,
                dilation=stem.dilation,
                groups=stem.groups,
                bias=stem.bias is not None,
                padding_mode=stem.padding_mode,
            )
            with torch.no_grad():
                rgba_stem.weight[:, :3].copy_(stem.weight)
                rgba_stem.weight[:, 3:].zero_()
                if stem.bias is not None and rgba_stem.bias is not None:
                    rgba_stem.bias.copy_(stem.bias)
            model.backbone.features[0][0] = rgba_stem
        model.load_state_dict(state, strict=True)

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.torch = torch
        self.model = model.to(device).eval()
        self.decode_lines = deccode_lines_TP
        self.device = torch.device(device)
        self.input_size = int(input_size or checkpoint_args.get("input_size", 512))
        self.score_threshold = float(score_threshold)
        self.min_length = float(min_length)
        self.max_lines = int(max_lines)

    def detect(self, roi_bgr: np.ndarray) -> LineDetections:
        height, width = roi_bgr.shape[:2]
        if height <= 0 or width <= 0:
            return _empty_detections()
        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        resized = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        normalized = (resized - np.asarray((0.485, 0.456, 0.406), dtype=np.float32)) / np.asarray(
            (0.229, 0.224, 0.225), dtype=np.float32
        )
        alpha = np.ones((self.input_size, self.input_size, 1), dtype=np.float32)
        rgba = np.concatenate((normalized, alpha), axis=2)
        tensor = self.torch.from_numpy(rgba.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            output = self.model(tensor)
            map_height, map_width = output.shape[-2:]
            _, lines, _, scores = self.decode_lines(
                output,
                score_thresh=self.score_threshold,
                len_thresh=0.0,
                topk_n=min(self.max_lines, map_height * map_width),
                ksize=3,
            )
        lines = lines.detach().cpu().numpy().astype(np.float32)
        scores = scores.detach().cpu().numpy().astype(np.float32)
        if len(lines) == 0:
            return _empty_detections()
        lines[:, [0, 2]] *= float(width) / float(map_width)
        lines[:, [1, 3]] *= float(height) / float(map_height)
        return _filter_lines(lines, scores, min_length=self.min_length, max_lines=self.max_lines)


class LineaDetector:
    """Run the official LINEA PyTorch checkpoint from a cloned repository."""

    def __init__(
        self,
        repo_path: Path,
        config_path: Path,
        checkpoint_path: Path,
        *,
        input_size: int = 640,
        score_threshold: float = 0.40,
        min_length: float = 5.0,
        max_lines: int = 300,
        device: str = "cpu",
    ) -> None:
        import torch

        repo_path = repo_path.resolve()
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))
        try:
            from models.registry import MODULE_BUILD_FUNCS
            from util.slconfig import SLConfig
        except ImportError as exc:
            raise RuntimeError(
                "LINEA requires the official LINEA repository in --linea-repo."
            ) from exc

        config = SLConfig.fromfile(str(config_path))
        if "HGNetv2" in config.backbone:
            config.pretrained = False
        config.multiscale = None
        build_func = MODULE_BUILD_FUNCS.get(getattr(config, "modelname"))
        model, postprocessor = build_func(config)
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        state = checkpoint.get("ema", {}).get("module") or checkpoint["model"]
        model.load_state_dict(state)

        class DeployedModel(torch.nn.Module):
            def __init__(self, model, postprocessor):
                super().__init__()
                self.model = model.deploy()
                self.postprocessor = postprocessor.deploy()

            def forward(self, images, original_sizes):
                return self.postprocessor(self.model(images), original_sizes)

        self.torch = torch
        self.model = DeployedModel(model, postprocessor).to(device).eval()
        self.device = torch.device(device)
        self.input_size = int(input_size)
        self.score_threshold = float(score_threshold)
        self.min_length = float(min_length)
        self.max_lines = int(max_lines)

    def detect(self, roi_bgr: np.ndarray) -> LineDetections:
        height, width = roi_bgr.shape[:2]
        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        image = self.torch.from_numpy(resized).permute(2, 0, 1).float().div(255.0)
        image = self.torchvision_normalize(image).unsqueeze(0).to(self.device)
        original_size = self.torch.tensor([[width, height]], device=self.device)
        with self.torch.no_grad():
            lines, scores = self.model(image, original_size)
        lines = lines[0].detach().cpu().numpy()
        scores = scores[0].detach().cpu().numpy()
        keep = scores >= self.score_threshold
        return _filter_lines(
            lines[keep],
            scores[keep],
            min_length=self.min_length,
            max_lines=self.max_lines,
        )

    @property
    def torchvision_normalize(self):
        from torchvision.transforms import Normalize

        return Normalize(mean=[0.538, 0.494, 0.453], std=[0.257, 0.263, 0.273])


class FClipDetector:
    """Run an official F-Clip checkpoint on a square ROI."""

    MODEL_CONFIGS = {
        "HG1_D2": "config/fclip_HG1_D2.yaml",
        "HG1_D3": "config/fclip_HG1_D3.yaml",
        "HG1": "config/fclip_HG1.yaml",
        "HG2": "config/fclip_HG2.yaml",
        "HG2_LB": "config/fclip_HG2_LB.yaml",
        "HR": "config/fclip_HR.yaml",
    }

    def __init__(
        self,
        repo_path: Path,
        checkpoint_path: Path,
        *,
        variant: str = "HR",
        config_path: Path | None = None,
        input_size: int | None = None,
        score_threshold: float = 0.40,
        min_length: float = 5.0,
        max_lines: int = 300,
        device: str = "auto",
    ) -> None:
        import torch

        repo_path = repo_path.resolve()
        if variant not in self.MODEL_CONFIGS:
            raise ValueError(f"Unsupported F-Clip variant: {variant}")
        if not repo_path.is_dir():
            raise FileNotFoundError(f"Could not find F-Clip repository: {repo_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Could not find F-Clip checkpoint: {checkpoint_path}")
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        try:
            from FClip.config import C, M
            from FClip.models import MultitaskHead, hg, hgl, hr
            from FClip.models.stage_1 import FClip as FClipStage
        except ImportError as exc:
            raise RuntimeError(
                "F-Clip requires the official F-Clip repository and its dependencies."
            ) from exc

        if config_path is None:
            config_path = repo_path / self.MODEL_CONFIGS[variant]
        else:
            config_path = config_path.resolve()
            if not config_path.is_file():
                raise FileNotFoundError(f"Could not find F-Clip config: {config_path}")
        base_config_path = repo_path / "config" / "base.yaml"
        C.update(C.from_yaml(filename=str(base_config_path)))
        C.update(C.from_yaml(filename=str(config_path)))
        M.update(C.model)
        C.io.model_initialize_file = str(checkpoint_path.resolve())

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("F-Clip was configured for CUDA, but CUDA is unavailable.")
        self.device = torch.device(device)
        self.torch = torch
        if M.backbone == "stacked_hourglass":
            backbone = hg(
                depth=M.depth,
                head=lambda channels_in, channels_out: MultitaskHead(channels_in, channels_out),
                num_stacks=M.num_stacks,
                num_blocks=M.num_blocks,
                num_classes=sum(sum(MultitaskHead._get_head_size(), [])),
            )
        elif M.backbone == "hourglass_lines":
            backbone = hgl(
                depth=M.depth,
                head=lambda channels_in, channels_out: MultitaskHead(channels_in, channels_out),
                num_stacks=M.num_stacks,
                num_blocks=M.num_blocks,
                num_classes=sum(sum(MultitaskHead._get_head_size(), [])),
            )
        elif M.backbone == "hrnet":
            backbone = hr(
                head=lambda channels_in, channels_out: MultitaskHead(channels_in, channels_out),
                num_classes=sum(sum(MultitaskHead._get_head_size(), [])),
            )
        else:
            raise ValueError(f"Unsupported F-Clip backbone: {M.backbone}")
        self.model = FClipStage(backbone).to(self.device)
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("model"))
        if state_dict is None:
            raise RuntimeError("F-Clip checkpoint does not contain model_state_dict or model.")
        if state_dict and all(str(key).startswith("module.") for key in state_dict):
            state_dict = {
                str(key)[len("module.") :]: value
                for key, value in state_dict.items()
            }
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()
        default_input_size = int(M.resolution) * 4
        self.input_size = int(input_size or default_input_size)
        self.image_mean = np.asarray(M.image.mean, dtype=np.float32)
        self.image_stddev = np.asarray(M.image.stddev, dtype=np.float32)
        self.score_threshold = float(score_threshold)
        self.min_length = float(min_length)
        self.max_lines = int(max_lines)
        self.variant = variant

    def detect(self, roi_bgr: np.ndarray) -> LineDetections:
        height, width = roi_bgr.shape[:2]
        if height <= 0 or width <= 0:
            return _empty_detections()
        resized = cv2.resize(
            roi_bgr,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_AREA,
        )
        rgb = resized[:, :, ::-1].astype(np.float32)
        normalized = (rgb - self.image_mean) / self.image_stddev
        tensor = self.torch.from_numpy(normalized.transpose(2, 0, 1)).float().unsqueeze(0)
        tensor = tensor.to(self.device)
        with self.torch.no_grad():
            outputs = self.model({"image": tensor}, isTest=True)
        lines = outputs["heatmaps"]["lines"][0]
        scores = outputs["heatmaps"]["score"][0]
        keep = scores > self.score_threshold
        lines = (lines[keep] * 4.0).detach().cpu().numpy()
        scores = scores[keep].detach().cpu().numpy()
        if len(lines) == 0:
            return _empty_detections()

        input_height = input_width = float(self.input_size)
        lines[:, :, 0] *= height / input_height
        lines[:, :, 1] *= width / input_width
        lines_xyxy = np.column_stack(
            (
                lines[:, 0, 1],
                lines[:, 0, 0],
                lines[:, 1, 1],
                lines[:, 1, 0],
            )
        )
        return _filter_lines(
            lines_xyxy,
            scores,
            min_length=self.min_length,
            max_lines=self.max_lines,
        )


def build_line_detector(kind: str, **kwargs) -> LineDetector:
    """Construct the selected second-stage detector."""

    if kind == "mlsd":
        model_path = kwargs.pop("model_path")
        if model_path.suffix.lower() in {".pt", ".pth"}:
            repo_path = kwargs.pop("repo_path", model_path.parent.parent)
            return MlsdTorchDetector(model_path, repo_path, **kwargs)
        kwargs.pop("repo_path", None)
        kwargs.pop("device", None)
        return MlsdTFLiteDetector(model_path, **kwargs)
    if kind == "linea":
        return LineaDetector(
            kwargs.pop("repo_path"),
            kwargs.pop("config_path"),
            kwargs.pop("checkpoint_path"),
            **kwargs,
        )
    if kind == "fclip":
        return FClipDetector(
            kwargs.pop("repo_path"),
            kwargs.pop("checkpoint_path"),
            **kwargs,
        )
    raise ValueError(f"Unsupported line detector: {kind}")
