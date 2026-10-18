"""Shared cube-bounding-box ROI acquisition and second-model tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def square_roi_from_center(
    center_x: float,
    center_y: float,
    side: float,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    """Return a clipped square ROI in full-frame coordinates."""

    side_px = max(1, min(int(round(side)), frame_width, frame_height))
    x1 = max(0, min(frame_width - side_px, int(round(center_x - side_px / 2.0))))
    y1 = max(0, min(frame_height - side_px, int(round(center_y - side_px / 2.0))))
    return x1, y1, x1 + side_px, y1 + side_px


def square_roi_from_bbox(
    bbox_xyxy: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    """Return a padded square ROI around a full-frame bounding box."""

    x1, y1, x2, y2 = bbox_xyxy
    side = max(x2 - x1, y2 - y1) * (1.0 + max(0.0, float(padding)) * 2.0)
    return square_roi_from_center(
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
        side,
        frame_width,
        frame_height,
    )


def bbox_xyxy_from_candidate(candidate: Any) -> tuple[int, int, int, int]:
    """Extract a bbox from either a CubeCandidate-like object or an xyxy tuple."""

    bbox = getattr(candidate, "bbox_xyxy", candidate)
    values = tuple(int(round(float(value))) for value in bbox)
    if len(values) != 4:
        raise ValueError(f"Expected four bbox coordinates, got {candidate!r}")
    return values  # type: ignore[return-value]


@dataclass
class RoiTracker:
    """Track an ROI from second-model detections between bbox fallbacks.

    ``tracking_bounds_xyxy`` passed to :meth:`observe_tracking_bounds` are local
    to the current ROI. The tracker moves the ROI from the detection's center
    and changes its side length from the observed detection extent. If the
    second model returns no usable bounds, the tracker resets so the bbox
    detector is run on the next frame.
    """

    roi_xyxy: tuple[int, int, int, int] | None = None
    cube_candidate: Any | None = None
    base_side: int | None = None
    roi_padding: float | None = None

    def reset(self) -> None:
        self.roi_xyxy = None
        self.cube_candidate = None
        self.base_side = None

    def initialize(
        self,
        candidate: Any,
        frame_width: int,
        frame_height: int,
        padding: float | None = None,
    ) -> tuple[int, int, int, int]:
        self.cube_candidate = candidate if hasattr(candidate, "bbox_xyxy") else None
        if padding is not None:
            self.roi_padding = max(0.0, float(padding))
        padding = self.roi_padding if self.roi_padding is not None else 0.0
        self.roi_xyxy = square_roi_from_bbox(
            bbox_xyxy_from_candidate(candidate),
            frame_width,
            frame_height,
            padding,
        )
        x1, y1, x2, y2 = self.roi_xyxy
        self.base_side = min(x2 - x1, y2 - y1)
        return self.roi_xyxy

    def observe_tracking_bounds(
        self,
        tracking_bounds_xyxy: tuple[int, int, int, int] | None,
        frame_width: int,
        frame_height: int,
        tracking_gain: float = 1.0,
        tracking_padding: float = 0.0,
    ) -> None:
        """Update the ROI from second-model bounds or reset on a miss."""

        if self.roi_xyxy is None:
            return
        if tracking_bounds_xyxy is None:
            self.reset()
            return

        x1, y1, x2, y2 = self.roi_xyxy
        side = min(x2 - x1, y2 - y1)
        base_side = self.base_side if self.base_side is not None else side
        bx1, by1, bx2, by2 = tracking_bounds_xyxy
        tracking_point_x = (bx1 + bx2) / 2.0
        tracking_point_y = (by1 + by2) / 2.0
        observed_side = max(bx2 - bx1, by2 - by1)
        observed_padded_side = observed_side * (
            1.0 + max(0.0, float(tracking_padding)) * 2.0
        )
        # ``base_side`` already includes the initial bbox padding. Apply the
        # same padding to a newly larger segmentation bound without padding
        # the already-padded base a second time.
        target_side = max(base_side, observed_padded_side)
        gain = max(0.0, float(tracking_gain))
        dx = (tracking_point_x - side / 2.0) * gain
        dy = (tracking_point_y - side / 2.0) * gain
        self.roi_xyxy = square_roi_from_center(
            (x1 + x2) / 2.0 + dx,
            (y1 + y2) / 2.0 + dy,
            target_side,
            frame_width,
            frame_height,
        )


def roi_for_frame(
    frame: np.ndarray,
    bbox_detector: Any,
    tracker: RoiTracker,
    padding: float | None = None,
) -> tuple[tuple[int, int, int, int] | None, Any | None, str]:
    """Get the current ROI, running the bbox detector only after a miss."""

    height, width = frame.shape[:2]
    if tracker.roi_padding is None and padding is not None:
        tracker.roi_padding = max(0.0, float(padding))
    if tracker.roi_xyxy is not None:
        return tracker.roi_xyxy, tracker.cube_candidate, "tracker"

    candidate = bbox_detector.detect(frame)
    if candidate is None:
        tracker.reset()
        return None, None, "none"

    return tracker.initialize(candidate, width, height), candidate, "detector"
