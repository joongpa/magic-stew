from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from live_recognition.types import FrameResult


COLOR_DEBUG_MIN_SATURATION = 8
COLOR_DEBUG_MIN_VALUE = 18
COLOR_DEBUG_MAX_VALUE = 254
COLOR_DEBUG_MIN_CHANNEL_DELTA = 0
COLOR_DEBUG_LIGHTNESS = 220
COLOR_DEBUG_CHROMA_CONTRAST = 1.55


def draw_result(
    frame_bgr: NDArray[np.uint8],
    result: FrameResult,
    *,
    draw_mask_overlay: bool = False,
    fps: float | None = None,
    draw_color_debug: bool = False,
) -> NDArray[np.uint8]:
    output = build_luminance_debug_view(frame_bgr) if draw_color_debug else frame_bgr.copy()
    if fps is not None:
        cv2.putText(
            output,
            f"FPS {fps:.1f}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            3,
        )
        cv2.putText(
            output,
            f"FPS {fps:.1f}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (20, 20, 20),
            1,
        )

    if result.candidate is not None:
        if draw_mask_overlay and result.candidate.mask is not None:
            overlay = np.zeros_like(output)
            overlay[:, :] = (40, 220, 40)
            mask = result.candidate.mask > 0
            output[mask] = cv2.addWeighted(output, 0.45, overlay, 0.55, 0)[mask]

        x1, y1, x2, y2 = result.candidate.bbox_xyxy
        cv2.rectangle(output, (x1, y1), (x2, y2), (40, 220, 40), 2)
        label = f"cube {result.candidate.confidence:.2f}"
        cv2.putText(output, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 220, 40), 2)

        if draw_mask_overlay and result.candidate.keypoints_xy is not None:
            for point in result.candidate.keypoints_xy:
                cv2.circle(output, tuple(point.astype(int)), 4, (0, 220, 255), -1)

        if draw_mask_overlay and result.candidate.contour is not None:
            contour = result.candidate.contour.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(output, [contour], isClosed=True, color=(0, 255, 255), thickness=2)

    return output


def build_lab_chroma_debug_view(crop_bgr: NDArray[np.uint8]) -> NDArray[np.uint8] | None:
    if crop_bgr.size == 0:
        return None

    reliable_mask = _reliable_color_mask(crop_bgr)
    if int(reliable_mask.sum()) < 10:
        reliable_mask = np.ones(crop_bgr.shape[:2], dtype=bool)

    balanced = _gray_world_white_balance(crop_bgr, reliable_mask)
    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = COLOR_DEBUG_LIGHTNESS
    lab[:, :, 1:] = np.clip(
        128.0 + (lab[:, :, 1:].astype(np.float32) - 128.0) * COLOR_DEBUG_CHROMA_CONTRAST,
        0,
        255,
    ).astype(np.uint8)

    chroma_view = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    rejected_mask = ~reliable_mask
    chroma_view[rejected_mask] = (chroma_view[rejected_mask] * 0.25).astype(np.uint8)
    return chroma_view


def build_luminance_debug_view(frame_bgr: NDArray[np.uint8]) -> NDArray[np.uint8]:
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    luminance = lab[:, :, 0]
    return cv2.cvtColor(luminance, cv2.COLOR_GRAY2BGR)


def lab_ab_samples_from_bbox(
    frame_bgr: NDArray[np.uint8],
    bbox_xyxy: tuple[int, int, int, int],
    *,
    max_samples: int = 6000,
) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
    x1, y1, x2, y2 = _clip_bbox(bbox_xyxy, frame_bgr.shape[1], frame_bgr.shape[0])
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    crop = frame_bgr[y1:y2, x1:x2]
    reliable_mask = _reliable_color_mask(crop)
    if int(reliable_mask.sum()) < 10:
        reliable_mask = np.ones(crop.shape[:2], dtype=bool)

    balanced = _gray_world_white_balance(crop, reliable_mask)
    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)
    ab_samples = lab[:, :, 1:3][reliable_mask].astype(np.float32)
    bgr_samples = balanced[reliable_mask]
    if len(ab_samples) > max_samples:
        indexes = np.linspace(0, len(ab_samples) - 1, max_samples, dtype=np.int32)
        ab_samples = ab_samples[indexes]
        bgr_samples = bgr_samples[indexes]
    return ab_samples, bgr_samples


def save_lab_ab_samples_plot(
    ab_samples: NDArray[np.float32],
    bgr_samples: NDArray[np.uint8],
    output_path: str,
    *,
    title: str = "Cube bbox reliable pixels in LAB a/b",
) -> None:
    if len(ab_samples) == 0:
        raise ValueError("No LAB a/b samples available for plotting")

    cache_dir = Path(".matplotlib-cache")
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb_samples = cv2.cvtColor(bgr_samples.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3)
    colors = rgb_samples.astype(np.float32) / 255.0
    figure, axes = plt.subplots(figsize=(7, 7), dpi=140)
    axes.scatter(ab_samples[:, 0], ab_samples[:, 1], c=colors, s=4, alpha=0.35, linewidths=0)
    axes.axvline(128, color="0.75", linewidth=1)
    axes.axhline(128, color="0.75", linewidth=1)
    axes.set_xlim(0, 255)
    axes.set_ylim(0, 255)
    axes.set_xlabel("LAB a")
    axes.set_ylabel("LAB b")
    axes.set_title(title)
    axes.grid(True, color="0.9", linewidth=0.8)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def save_lab_ab_plot(
    frame_bgr: NDArray[np.uint8],
    bbox_xyxy: tuple[int, int, int, int],
    output_path: str,
) -> None:
    ab_samples, bgr_samples = lab_ab_samples_from_bbox(frame_bgr, bbox_xyxy)
    save_lab_ab_samples_plot(ab_samples, bgr_samples, output_path)


def _reliable_color_mask(crop_bgr: NDArray[np.uint8]) -> NDArray[np.bool_]:
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    hue = hsv[:, :, 0]
    value = hsv[:, :, 2]
    max_channel = crop_bgr.max(axis=2)
    min_channel = crop_bgr.min(axis=2)

    orange_candidate = (hue <= 30) & (saturation >= 20) & (value >= 35)
    bright_white_candidate = value >= 180
    colorful_enough = (
        (saturation >= COLOR_DEBUG_MIN_SATURATION) | bright_white_candidate | orange_candidate
    )
    not_too_dark = value >= COLOR_DEBUG_MIN_VALUE
    not_clipped = (max_channel <= COLOR_DEBUG_MAX_VALUE) | orange_candidate
    not_flat_gray = (max_channel - min_channel) >= COLOR_DEBUG_MIN_CHANNEL_DELTA
    return colorful_enough & not_too_dark & not_clipped & not_flat_gray


def _gray_world_white_balance(
    crop_bgr: NDArray[np.uint8],
    reliable_mask: NDArray[np.bool_],
) -> NDArray[np.uint8]:
    pixels = crop_bgr[reliable_mask].astype(np.float32)
    if pixels.size == 0:
        return crop_bgr.copy()

    channel_means = pixels.mean(axis=0)
    target = float(channel_means.mean())
    scales = target / np.maximum(channel_means, 1.0)
    balanced = crop_bgr.astype(np.float32) * scales.reshape(1, 1, 3)
    return np.clip(balanced, 0, 255).astype(np.uint8)


def _clip_bbox(
    bbox_xyxy: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    return (
        max(0, min(width, x1)),
        max(0, min(height, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    )
