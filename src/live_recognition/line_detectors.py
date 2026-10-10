"""Adapters for the pretrained M-LSD and LINEA line-segment detectors.

The adapters deliberately expose only visible line segments. They do not try
to complete occluded cube edges or infer a pose.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

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
        self.input_size = int(input_size)
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


def build_line_detector(kind: str, **kwargs) -> LineDetector:
    """Construct the selected second-stage detector."""

    if kind == "mlsd":
        return MlsdTFLiteDetector(kwargs.pop("model_path"), **kwargs)
    if kind == "linea":
        return LineaDetector(
            kwargs.pop("repo_path"),
            kwargs.pop("config_path"),
            kwargs.pop("checkpoint_path"),
            **kwargs,
        )
    raise ValueError(f"Unsupported line detector: {kind}")
