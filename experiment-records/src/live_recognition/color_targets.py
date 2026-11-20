"""Convert the generated YOLO sticker polygons into dense color targets."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import cv2
import numpy as np

from live_recognition.color_segmentation import (
    COLOR_CLASS_NAMES,
    IGNORE_INDEX,
    NONE_CLASS_INDEX,
)


def normalize_class_names(class_names: Sequence[str]) -> dict[str, int]:
    """Return a case-insensitive mapping for the six sticker colors."""

    normalized = {str(name).strip().lower(): index for index, name in enumerate(class_names)}
    required = set(COLOR_CLASS_NAMES[:-1])
    missing = sorted(required - set(normalized))
    if missing:
        raise ValueError(f"Sticker labels are missing color classes: {', '.join(missing)}")
    # Map source-label ordering onto the model's stable canonical ordering.
    return {name: index for index, name in enumerate(COLOR_CLASS_NAMES[:-1])}


def _polygon_to_output_points(
    polygon: Sequence[tuple[float, float]],
    *,
    image_width: int,
    image_height: int,
    roi_xyxy: tuple[int, int, int, int],
    output_width: int,
    output_height: int,
) -> np.ndarray:
    """Map normalized full-image polygon coordinates into an output ROI."""

    x1, y1, x2, y2 = roi_xyxy
    roi_width = max(1, x2 - x1)
    roi_height = max(1, y2 - y1)
    points = np.asarray(polygon, dtype=np.float32)
    points[:, 0] = (points[:, 0] * image_width - x1) * output_width / roi_width
    points[:, 1] = (points[:, 1] * image_height - y1) * output_height / roi_height
    return np.round(points).astype(np.int32)


def dense_color_target(
    image_shape: tuple[int, int],
    polygons: Iterable[tuple[str, Sequence[tuple[float, float]]]],
    class_names: Sequence[str],
    roi_xyxy: tuple[int, int, int, int],
    output_size: int | tuple[int, int],
    *,
    boundary_ignore_px: int = 1,
) -> np.ndarray:
    """Rasterize sticker polygons into a ``[H, W]`` class-index target.

    Uncovered pixels are ``none``.  Sticker boundaries are ``IGNORE_INDEX`` by
    default because antialiasing, segmentation polygon precision, and sticker
    seams make those pixels poor color supervision.
    """

    image_height, image_width = image_shape
    if isinstance(output_size, int):
        output_height = output_width = output_size
    else:
        output_height, output_width = output_size

    target = np.full(
        (output_height, output_width), NONE_CLASS_INDEX, dtype=np.int64
    )
    sticker_union = np.zeros((output_height, output_width), dtype=np.uint8)
    sticker_boundaries = np.zeros((output_height, output_width), dtype=np.uint8)
    class_by_name = normalize_class_names(class_names)

    for class_name, polygon in polygons:
        color_name = str(class_name).strip().lower()
        if color_name not in class_by_name:
            raise ValueError(f"Unknown sticker color class: {class_name!r}")
        points = _polygon_to_output_points(
            polygon,
            image_width=image_width,
            image_height=image_height,
            roi_xyxy=roi_xyxy,
            output_width=output_width,
            output_height=output_height,
        )
        if len(points) < 3:
            continue
        polygon_mask = np.zeros_like(sticker_union)
        cv2.fillPoly(polygon_mask, [points], 1)
        target[polygon_mask.astype(bool)] = class_by_name[color_name]
        sticker_union |= polygon_mask
        if boundary_ignore_px > 0:
            kernel_size = boundary_ignore_px * 2 + 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            eroded_polygon = cv2.erode(polygon_mask, kernel, iterations=1)
            sticker_boundaries |= polygon_mask & (eroded_polygon == 0)

    if boundary_ignore_px > 0 and bool(sticker_union.any()):
        target[sticker_boundaries > 0] = IGNORE_INDEX

    return target
