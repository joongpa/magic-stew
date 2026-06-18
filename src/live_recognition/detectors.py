from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from live_recognition.model_utils import (
    ensure_yolo_config_dir,
    resolve_model_path,
    resolve_sam3_model_path,
)
from live_recognition.types import CubeCandidate

_resolve_model_path = resolve_model_path
_resolve_sam3_model_path = resolve_sam3_model_path


class CubeDetector(Protocol):
    def detect(self, frame_bgr: NDArray[np.uint8]) -> CubeCandidate | None:
        """Return the best cube candidate for a frame."""


def _best_confident_box_index(confidences, threshold: float) -> int | None:
    above_threshold = confidences >= threshold
    if not bool(above_threshold.any()):
        return None

    candidate_indexes = above_threshold.nonzero().flatten()
    best_local_index = int(confidences[candidate_indexes].argmax().item())
    return int(candidate_indexes[best_local_index].item())


def _bbox_from_mask(mask: NDArray[np.uint8]) -> tuple[int, int, int, int] | None:
    points = cv2.findNonZero(mask)
    if points is None:
        return None

    x, y, w, h = cv2.boundingRect(points)
    return x, y, x + w, y + h


def _bbox_from_xyxy(box_xyxy) -> tuple[int, int, int, int]:
    values = box_xyxy.detach().cpu().numpy() if hasattr(box_xyxy, "detach") else np.asarray(box_xyxy)
    x1, y1, x2, y2 = values.astype(float).round().astype(int).tolist()
    return x1, y1, x2, y2


def _convex_hull_mask(mask: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], NDArray[np.float32] | None]:
    """Fill the convex hull around non-zero mask pixels."""

    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return binary, None

    points = np.concatenate(contours, axis=0)
    hull = cv2.convexHull(points)
    hull_mask = np.zeros_like(binary)
    cv2.fillConvexPoly(hull_mask, hull, 255)
    return hull_mask, hull.reshape(-1, 2).astype(np.float32)


@dataclass
class SAM3CubeDetector:
    """Local SAM 3 text-prompt cube detector."""

    model_name: str = "sam3.pt"
    confidence_threshold: float = 0.5
    prompt: str = "3x3x3 rubik's cube"
    half: bool = False
    return_mask: bool = False

    def __post_init__(self) -> None:
        ensure_yolo_config_dir()
        os.environ.setdefault("YOLO_CONFIG_DIR", ".ultralytics")

        from ultralytics.models.sam import SAM3SemanticPredictor

        class SAM3BoxOnlyPredictor(SAM3SemanticPredictor):
            def postprocess(self, preds, img, orig_imgs):
                import torch
                import torchvision
                from ultralytics.engine.results import Results
                from ultralytics.utils import ops

                pred_boxes = preds["pred_boxes"]
                pred_logits = preds["pred_logits"]
                pred_scores = pred_logits.sigmoid()
                presence_score = preds["presence_logit_dec"].sigmoid().unsqueeze(1)
                pred_scores = (pred_scores * presence_score).squeeze(-1)
                pred_cls = torch.tensor(
                    list(range(pred_scores.shape[0])),
                    dtype=pred_scores.dtype,
                    device=pred_scores.device,
                )[:, None].expand_as(pred_scores)
                pred_boxes = torch.cat([pred_boxes, pred_scores[..., None], pred_cls[..., None]], dim=-1)
                keep = pred_scores > self.args.conf
                pred_boxes = pred_boxes[keep]
                pred_boxes[:, :4] = ops.xywh2xyxy(pred_boxes[:, :4])

                c = pred_boxes[:, 5:6] * (0 if self.args.agnostic_nms else 7680)
                nms_boxes = pred_boxes[:, :4] + c
                keep = torchvision.ops.nms(nms_boxes, pred_boxes[:, 4], self.args.iou)
                pred_boxes = pred_boxes[keep]
                names = getattr(self.model, "names", [str(i) for i in range(pred_scores.shape[0])])

                if not isinstance(orig_imgs, list):
                    orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

                results = []
                for boxes, orig_img, img_path in zip([pred_boxes], orig_imgs, self.batch[0]):
                    if boxes.shape[0] == 0:
                        boxes = torch.zeros((0, 6), device=pred_boxes.device)
                    else:
                        boxes[..., [0, 2]] *= orig_img.shape[1]
                        boxes[..., [1, 3]] *= orig_img.shape[0]
                    results.append(Results(orig_img, path=img_path, names=names, masks=None, boxes=boxes))
                return results

        model_path = resolve_sam3_model_path(self.model_name)

        overrides = {
            "conf": self.confidence_threshold,
            "task": "segment",
            "mode": "predict",
            "model": model_path,
            "half": self.half,
            "verbose": False,
        }
        predictor_cls = SAM3SemanticPredictor if self.return_mask else SAM3BoxOnlyPredictor
        self.predictor = predictor_cls(overrides=overrides)

    def detect_all(self, frame_bgr: NDArray[np.uint8]) -> list[CubeCandidate]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(frame_rgb)
        results = self.predictor(text=[self.prompt])
        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        candidates = []
        indexes = (boxes.conf >= self.confidence_threshold).nonzero().flatten()
        for index_tensor in indexes:
            index = int(index_tensor.item())
            confidence = float(boxes.conf[index].item())
            bbox = _bbox_from_xyxy(boxes.xyxy[index])

            hull = None
            hull_mask = None
            masks = result.masks
            if self.return_mask and masks is not None:
                visible_mask = masks.data[index].detach().cpu().numpy().astype(np.uint8) * 255
                hull_mask, hull = _convex_hull_mask(visible_mask)

            candidates.append(
                CubeCandidate(
                    bbox_xyxy=bbox,
                    confidence=confidence,
                    contour=hull,
                    mask=hull_mask,
                )
            )

        return candidates

    def detect(self, frame_bgr: NDArray[np.uint8]) -> CubeCandidate | None:
        candidates = self.detect_all(frame_bgr)
        if not candidates:
            return None
        return max(candidates, key=lambda candidate: candidate.confidence)


@dataclass
class YOLOCubeDetector:
    """Cube detector backed by a trained Ultralytics YOLO model."""

    model_name: str = "runs/500ExamplesWeights.pt"
    confidence_threshold: float = 0.25
    image_size: int = 640

    def __post_init__(self) -> None:
        ensure_yolo_config_dir()
        os.environ.setdefault("YOLO_CONFIG_DIR", ".ultralytics")
        from ultralytics import YOLO

        self.model = YOLO(resolve_model_path(self.model_name, "YOLO"))

    def detect_all(self, frame_bgr: NDArray[np.uint8]) -> list[CubeCandidate]:
        results = self.model.predict(
            source=frame_bgr,
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            verbose=False,
        )
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        candidates = []
        for index in range(len(boxes)):
            confidence = float(boxes.conf[index].item())
            if confidence < self.confidence_threshold:
                continue
            candidates.append(
                CubeCandidate(
                    bbox_xyxy=_bbox_from_xyxy(boxes.xyxy[index]),
                    confidence=confidence,
                )
            )
        return candidates

    def detect(self, frame_bgr: NDArray[np.uint8]) -> CubeCandidate | None:
        candidates = self.detect_all(frame_bgr)
        if not candidates:
            return None
        return max(candidates, key=lambda candidate: candidate.confidence)


@dataclass
class HeuristicCubeDetector:
    """Find the strongest square-ish, colorful object in a frame.

    This is a bootstrap detector for demos and data collection. It is expected to fail under
    hard occlusion; the interface is what should survive when a learned model replaces it.
    """

    min_area_ratio: float = 0.01

    def detect(self, frame_bgr: NDArray[np.uint8]) -> CubeCandidate | None:
        height, width = frame_bgr.shape[:2]
        frame_area = float(height * width)

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        colorful = cv2.inRange(saturation, 45, 255)
        bright = cv2.inRange(value, 45, 255)
        mask = cv2.bitwise_and(colorful, bright)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best: CubeCandidate | None = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area_ratio * frame_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / max(h, 1)
            aspect_score = max(0.0, 1.0 - abs(np.log(aspect)))

            hull = cv2.convexHull(contour)
            hull_area = max(cv2.contourArea(hull), 1.0)
            solidity = min(area / hull_area, 1.0)

            extent = area / max(float(w * h), 1.0)
            area_score = min(area / (0.20 * frame_area), 1.0)
            confidence = float(0.35 * aspect_score + 0.30 * solidity + 0.20 * extent + 0.15 * area_score)

            epsilon = 0.03 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).astype(np.float32)

            hull_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillConvexPoly(hull_mask, hull, 255)

            candidate = CubeCandidate(
                bbox_xyxy=(x, y, x + w, y + h),
                confidence=confidence,
                contour=hull.reshape(-1, 2).astype(np.float32),
                keypoints_xy=approx if len(approx) <= 12 else None,
                mask=hull_mask,
            )
            if best is None or candidate.confidence > best.confidence:
                best = candidate

        return best
