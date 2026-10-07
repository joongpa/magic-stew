"""Display sticker segmentation and dense color-map predictions.

Examples:

    .venv/bin/python scripts/render_yolo_sticker_segmentation.py \
        --source path/to/cube.png
    .venv/bin/python scripts/render_yolo_sticker_segmentation.py \
        --source path/to/video.mp4
    .venv/bin/python scripts/render_yolo_sticker_segmentation.py \
        --source path/to/video.mp4 \
        --color-map-model runs/cube_color_map_head/best_color_map_head.pt

Click a pixel in the top-1 or top-2 color panel to inspect its seven-class
probability distribution. Press `q`, `Esc`, or `Enter` to close the window.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from live_recognition.detectors import YOLOCubeDetector  # noqa: E402

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm"}
DEFAULT_BBOX_MODEL = ROOT / "runs/500ExamplesWeights.pt"
STICKER_COLORS_BGR = {
    "white": (255, 255, 255),
    "yellow": (0, 255, 255),
    "red": (0, 0, 255),
    "orange": (0, 165, 255),
    "blue": (255, 0, 0),
    "green": (0, 180, 0),
}
STICKER_COLORS_BY_ID = tuple(STICKER_COLORS_BGR[name] for name in STICKER_COLORS_BGR)
COLOR_MAP_CLASS_NAMES = (*STICKER_COLORS_BGR.keys(), "none")
COLOR_MAP_COLORS_BGR = np.asarray(
    (*STICKER_COLORS_BY_ID, (72, 72, 72)),
    dtype=np.float32,
)
WINDOW_NAME = "sticker color segmentation"


def default_model_path(segmentation_type: str) -> Path:
    if segmentation_type == "whole-cube":
        candidates = (
            ROOT / "runs/cube_whole_seg_local/runs/cube_whole_yolo11n_seg/weights/best.pt",
            ROOT / "runs/segment/runs/cube_whole_seg_local/runs/cube_whole_yolo11n_seg/weights/best.pt",
        )
    else:
        candidates = (
            ROOT / "runs/segment/runs/cube_sticker_seg_local/runs/cube_sticker_yolo11n_seg/weights/best.pt",
            ROOT / "runs/cube_sticker_seg_local/runs/cube_sticker_yolo11n_seg/weights/best.pt",
        )
    return next((path for path in candidates if path.exists()), candidates[0])


def parse_source(source: str) -> int | str:
    source = source.strip().strip("\"'").strip()
    return int(source) if source.isdigit() else source


def is_image_source(source: int | str) -> bool:
    return isinstance(source, str) and Path(source).suffix.lower() in IMAGE_EXTENSIONS


def square_roi_from_center(
    center_x: float,
    center_y: float,
    side: float,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
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
    x1, y1, x2, y2 = bbox_xyxy
    side = max(x2 - x1, y2 - y1) * (1.0 + padding * 2.0)
    return square_roi_from_center(
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
        side,
        frame_width,
        frame_height,
    )


class RoiTracker:
    """Track the sticker-derived ROI between cube-detector fallbacks."""

    def __init__(self) -> None:
        self.roi_xyxy: tuple[int, int, int, int] | None = None
        self.base_side: int | None = None

    def reset(self) -> None:
        self.roi_xyxy = None
        self.base_side = None

    def initialize(
        self,
        bbox_xyxy: tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
        padding: float,
    ) -> tuple[int, int, int, int]:
        self.roi_xyxy = square_roi_from_bbox(
            bbox_xyxy,
            frame_width,
            frame_height,
            padding,
        )
        x1, y1, x2, y2 = self.roi_xyxy
        self.base_side = min(x2 - x1, y2 - y1)
        return self.roi_xyxy

    def observe_tracking_bounds(
        self,
        bounds_xyxy: tuple[int, int, int, int] | None,
        frame_width: int,
        frame_height: int,
        tracking_gain: float = 1.4,
        tracking_padding: float = 0.18,
    ) -> None:
        if self.roi_xyxy is None:
            return
        if bounds_xyxy is None:
            self.reset()
            return

        x1, y1, x2, y2 = self.roi_xyxy
        side = min(x2 - x1, y2 - y1)
        base_side = self.base_side if self.base_side is not None else side
        bx1, by1, bx2, by2 = bounds_xyxy
        observed_side = max(bx2 - bx1, by2 - by1)
        target_side = max(base_side, observed_side) * (1.0 + tracking_padding * 2.0)
        dx = ((bx1 + bx2) / 2.0 - side / 2.0) * max(0.0, tracking_gain)
        dy = ((by1 + by2) / 2.0 - side / 2.0) * max(0.0, tracking_gain)
        self.roi_xyxy = square_roi_from_center(
            (x1 + x2) / 2.0 + dx,
            (y1 + y2) / 2.0 + dy,
            target_side,
            frame_width,
            frame_height,
        )


def class_name(names: object, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def class_color(names: object, class_id: int, segmentation_type: str) -> tuple[int, int, int]:
    if segmentation_type == "whole-cube":
        return (255, 0, 255)
    name = class_name(names, class_id).strip().lower()
    if name in STICKER_COLORS_BGR:
        return STICKER_COLORS_BGR[name]
    if 0 <= class_id < len(STICKER_COLORS_BY_ID):
        return STICKER_COLORS_BY_ID[class_id]
    return (180, 180, 180)


def tensor_to_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def render_segmentation_overlay(
    image: np.ndarray,
    result: object,
    opacity: float,
    segmentation_type: str,
) -> np.ndarray:
    overlay = image.copy()
    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if masks is None or boxes is None:
        return image.copy()

    mask_data = tensor_to_numpy(masks.data)
    class_ids = tensor_to_numpy(boxes.cls).reshape(-1)
    names = getattr(result, "names", {})
    height, width = image.shape[:2]
    rendered_masks = [
        cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST) >= 0.5
        for mask in mask_data[: len(class_ids)]
    ]
    rendered_indexes = range(len(rendered_masks))
    if segmentation_type == "whole-cube" and rendered_masks:
        rendered_indexes = (max(rendered_indexes, key=lambda index: int(rendered_masks[index].sum())),)

    for index in rendered_indexes:
        mask = rendered_masks[index]
        color = class_color(names, int(class_ids[index]), segmentation_type)
        overlay[mask] = color

    output = cv2.addWeighted(overlay, opacity, image, 1.0 - opacity, 0.0)
    for index in rendered_indexes:
        mask = rendered_masks[index]
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(
            output,
            contours,
            -1,
            class_color(names, int(class_ids[index]), segmentation_type),
            2,
        )
    return output


def predict_color_probabilities(
    color_model: object,
    roi_bgr: np.ndarray,
    output_size: int,
    device: object,
) -> np.ndarray:
    """Predict and resize the seven-channel color probabilities for one ROI."""

    import torch

    model_roi = resize_to_square_size(roi_bgr, output_size)
    roi_rgb = cv2.cvtColor(model_roi, cv2.COLOR_BGR2RGB)
    image = (
        torch.from_numpy(roi_rgb.copy())
        .permute(2, 0, 1)
        .float()
        .div(255.0)
        .unsqueeze(0)
        .to(device)
    )
    with torch.no_grad():
        probabilities = color_model(image)["color_probabilities"][0]
    probabilities = probabilities.detach().cpu().numpy()

    roi_height, roi_width = roi_bgr.shape[:2]
    resized_probabilities = np.stack(
        [
            cv2.resize(
                probability.astype(np.float32),
                (roi_width, roi_height),
                interpolation=cv2.INTER_LINEAR,
            )
            for probability in probabilities
        ],
        axis=0,
    )
    probability_sum = resized_probabilities.sum(axis=0, keepdims=True)
    resized_probabilities /= np.maximum(probability_sum, 1e-6)
    return resized_probabilities


def render_color_probability_map(
    probabilities: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Render a soft palette mixture of the seven color probabilities."""

    color_map = np.tensordot(
        probabilities.transpose(1, 2, 0),
        COLOR_MAP_COLORS_BGR,
        axes=([2], [0]),
    )
    color_map = np.clip(color_map, 0, 255).astype(np.uint8)
    if valid_mask is not None:
        if valid_mask.shape != color_map.shape[:2]:
            raise ValueError(
                "valid_mask must have the same height and width as roi_bgr: "
                f"expected {color_map.shape[:2]}, got {valid_mask.shape}"
            )
        color_map[~valid_mask.astype(bool)] = (28, 28, 28)
    return color_map


def render_color_rank_map(
    probabilities: np.ndarray,
    rank: int,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Render the class at a per-pixel probability rank, where rank 0 is top-1."""

    if not 0 <= rank < probabilities.shape[0]:
        raise ValueError(f"rank must be in [0, {probabilities.shape[0]}), got {rank}")
    class_order = np.argsort(probabilities, axis=0)
    class_indices = class_order[-1 - rank]
    color_map = COLOR_MAP_COLORS_BGR[class_indices].astype(np.uint8)
    if valid_mask is not None:
        if valid_mask.shape != color_map.shape[:2]:
            raise ValueError(
                "valid_mask must have the same height and width as probabilities: "
                f"expected {color_map.shape[:2]}, got {valid_mask.shape}"
            )
        color_map[~valid_mask.astype(bool)] = (28, 28, 28)
    return color_map


def empty_color_panel(image: np.ndarray) -> np.ndarray:
    """Return a blank right-hand panel when no color map is available."""

    panel = np.full_like(image, 28)
    cv2.putText(
        panel,
        "color map unavailable",
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (210, 210, 210),
        2,
        cv2.LINE_AA,
    )
    return panel


class ColorDistributionPicker:
    """Interactive exact probability readout for pixels in the color panels."""

    def __init__(self) -> None:
        self.probabilities: np.ndarray | None = None
        self.valid_mask: np.ndarray | None = None
        self.roi_xyxy: tuple[int, int, int, int] | None = None
        self.frame_shape: tuple[int, int] | None = None
        self.selected_roi_xy: tuple[int, int] | None = None
        self.message = "click a pixel in the top-1 or top-2 map"

    def clear(self, frame_shape: tuple[int, int] | None = None) -> None:
        self.probabilities = None
        self.valid_mask = None
        self.roi_xyxy = None
        self.frame_shape = frame_shape
        self.selected_roi_xy = None
        self.message = "no color map available for this frame"

    def update(
        self,
        probabilities: np.ndarray,
        valid_mask: np.ndarray | None,
        roi_xyxy: tuple[int, int, int, int],
        frame_shape: tuple[int, int],
    ) -> None:
        self.probabilities = probabilities
        self.valid_mask = valid_mask
        self.roi_xyxy = roi_xyxy
        self.frame_shape = frame_shape
        if self.selected_roi_xy is not None:
            x, y = self.selected_roi_xy
            height, width = probabilities.shape[-2:]
            if not (0 <= x < width and 0 <= y < height):
                self.selected_roi_xy = None

    def handle_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or self.probabilities is None:
            return
        if self.frame_shape is None or self.roi_xyxy is None:
            return
        frame_height, frame_width = self.frame_shape

        # The top-1 panel is the top-right grid cell. The top-2 panel is the
        # bottom-left grid cell. Both contain the same frame-sized ROI panel.
        if frame_width <= x < 2 * frame_width and 0 <= y < frame_height:
            panel_x, panel_y = x - frame_width, y
        elif 0 <= x < frame_width and frame_height <= y < 2 * frame_height:
            panel_x, panel_y = x, y - frame_height
        else:
            return

        x1, y1, x2, y2 = self.roi_xyxy
        roi_x = panel_x - x1
        roi_y = panel_y - y1
        height, width = self.probabilities.shape[-2:]
        if not (0 <= roi_x < width and 0 <= roi_y < height):
            self.selected_roi_xy = None
            self.message = "selected pixel is outside the output ROI"
            return
        if self.valid_mask is not None and not bool(self.valid_mask[roi_y, roi_x]):
            self.selected_roi_xy = (roi_x, roi_y)
            self.message = "selected pixel is outside the predicted segmentation mask"
            return
        self.selected_roi_xy = (roi_x, roi_y)
        self.message = ""

    def render(self, panel_shape: tuple[int, int, int]) -> np.ndarray:
        height, width = panel_shape[:2]
        panel = np.full((height, width, 3), 28, dtype=np.uint8)
        cv2.putText(
            panel,
            "pixel color distribution",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )
        if self.probabilities is None:
            cv2.putText(
                panel,
                self.message,
                (20, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (190, 190, 190),
                1,
                cv2.LINE_AA,
            )
            return panel

        if self.selected_roi_xy is None:
            cv2.putText(
                panel,
                self.message,
                (20, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (190, 190, 190),
                1,
                cv2.LINE_AA,
            )
            return panel

        roi_x, roi_y = self.selected_roi_xy
        if self.valid_mask is not None and not bool(self.valid_mask[roi_y, roi_x]):
            cv2.putText(
                panel,
                self.message,
                (20, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (190, 190, 190),
                1,
                cv2.LINE_AA,
            )
            return panel

        probabilities = self.probabilities[:, roi_y, roi_x]
        best_class = int(probabilities.argmax())
        entropy = float(
            -(probabilities * np.log(np.maximum(probabilities, 1e-8))).sum()
            / np.log(max(2, len(probabilities)))
        )
        cv2.putText(
            panel,
            f"ROI pixel ({roi_x}, {roi_y})  top={COLOR_MAP_CLASS_NAMES[best_class]}  H={entropy:.2f}",
            (20, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        row_height = max(24, min(42, (height - 105) // len(COLOR_MAP_CLASS_NAMES)))
        label_x = 20
        bar_x = max(120, width // 3)
        bar_width = max(80, width - bar_x - 24)
        for class_id, (name, probability) in enumerate(
            zip(COLOR_MAP_CLASS_NAMES, probabilities)
        ):
            y = 105 + class_id * row_height
            color = tuple(int(value) for value in COLOR_MAP_COLORS_BGR[class_id])
            cv2.putText(
                panel,
                name,
                (label_x, y + row_height - 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (225, 225, 225),
                1,
                cv2.LINE_AA,
            )
            cv2.rectangle(
                panel,
                (bar_x, y + 5),
                (bar_x + bar_width, y + row_height - 12),
                (65, 65, 65),
                thickness=-1,
            )
            filled_width = int(round(bar_width * float(probability)))
            if filled_width > 0:
                cv2.rectangle(
                    panel,
                    (bar_x, y + 5),
                    (bar_x + filled_width, y + row_height - 12),
                    color,
                    thickness=-1,
                )
            cv2.putText(
                panel,
                f"{float(probability):.3f}",
                (bar_x + bar_width - 52, y + row_height - 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
        return panel


def refresh_distribution_panel(
    rendered: np.ndarray,
    distribution_picker: ColorDistributionPicker | None,
) -> np.ndarray:
    """Refresh the picker cell after a mouse click without rerunning inference."""

    if distribution_picker is None or distribution_picker.frame_shape is None:
        return rendered
    frame_height, frame_width = distribution_picker.frame_shape
    if rendered.shape[0] < 2 * frame_height or rendered.shape[1] < 2 * frame_width:
        return rendered
    refreshed = rendered.copy()
    refreshed[
        frame_height : 2 * frame_height,
        frame_width : 2 * frame_width,
    ] = distribution_picker.render((frame_height, frame_width, 3))
    return refreshed


def color_panel_for_roi(
    image_shape: tuple[int, int, int],
    color_map: np.ndarray | None,
    roi_xyxy: tuple[int, int, int, int] | None,
    title: str = "color probability map",
) -> np.ndarray:
    """Place the color map in a frame-sized black panel without an image overlay."""

    height, width = image_shape[:2]
    panel = np.full((height, width, 3), 28, dtype=np.uint8)
    if color_map is None or roi_xyxy is None:
        return panel
    x1, y1, x2, y2 = roi_xyxy
    if x2 <= x1 or y2 <= y1:
        return panel
    panel[y1:y2, x1:x2] = cv2.resize(
        color_map,
        (x2 - x1, y2 - y1),
        interpolation=cv2.INTER_LINEAR,
    )
    cv2.putText(
        panel,
        title,
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )
    return panel


def split_screen(
    left: np.ndarray,
    right: np.ndarray | list[np.ndarray] | None,
) -> np.ndarray:
    """Return the existing view or a compact two-column comparison grid."""

    if right is None:
        return left
    panels = right if isinstance(right, list) else [right]
    all_panels = [left, *panels]
    if len(all_panels) % 2:
        all_panels.append(np.full_like(left, 28))
    rows = [
        np.hstack(all_panels[index : index + 2])
        for index in range(0, len(all_panels), 2)
    ]
    return np.vstack(rows)


def has_segmentation(result: object) -> bool:
    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if masks is None or boxes is None:
        return False
    mask_data = tensor_to_numpy(masks.data)
    class_ids = tensor_to_numpy(boxes.cls).reshape(-1)
    return mask_data.ndim >= 3 and len(mask_data) > 0 and len(class_ids) > 0


def segmentation_union_mask(
    result: object,
    output_shape: tuple[int, int],
) -> np.ndarray | None:
    """Return the union of predicted instance masks at the ROI resolution."""

    if not has_segmentation(result):
        return None
    masks = getattr(result, "masks")
    mask_data = tensor_to_numpy(masks.data)
    output_height, output_width = output_shape
    combined = np.zeros((output_height, output_width), dtype=bool)
    for mask in mask_data:
        resized = cv2.resize(
            mask.astype(np.float32),
            (output_width, output_height),
            interpolation=cv2.INTER_LINEAR,
        )
        combined |= resized > 0.5
    return combined if bool(combined.any()) else None


def segmentation_bounds(
    result: object,
    roi_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    if not has_segmentation(result):
        return None
    masks = result.masks
    mask_data = tensor_to_numpy(masks.data)
    roi_height, roi_width = roi_shape
    combined = np.zeros((roi_height, roi_width), dtype=bool)
    for mask in mask_data:
        resized = cv2.resize(
            mask.astype(np.float32),
            (roi_width, roi_height),
            interpolation=cv2.INTER_LINEAR,
        )
        combined |= resized > 0.5
    ys, xs = np.nonzero(combined)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def predict(model: object, image: np.ndarray, args: argparse.Namespace) -> object | None:
    predictions = model.predict(
        source=image,
        conf=args.conf,
        imgsz=args.imgsz,
        verbose=False,
    )
    return predictions[0] if predictions else None


def predict_and_render(
    model: object,
    color_model: object | None,
    color_device: object | None,
    distribution_picker: ColorDistributionPicker | None,
    bbox_detector: YOLOCubeDetector | None,
    roi_tracker: RoiTracker,
    image: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, YOLOCubeDetector | None]:
    height, width = image.shape[:2]
    roi_source = "tracker"
    if roi_tracker.roi_xyxy is None:
        if bbox_detector is None:
            bbox_detector = YOLOCubeDetector(
                model_name=str(args.bbox_model),
                confidence_threshold=args.bbox_conf,
                image_size=args.bbox_imgsz,
            )
        candidate = bbox_detector.detect(image)
        if candidate is None:
            if color_model is not None:
                if distribution_picker is not None:
                    distribution_picker.clear(image.shape[:2])
                right_panel = [
                    empty_color_panel(image),
                    empty_color_panel(image),
                    distribution_picker.render(image.shape) if distribution_picker else empty_color_panel(image),
                ]
            else:
                right_panel = None
            return split_screen(image.copy(), right_panel), bbox_detector
        roi_tracker.initialize(candidate.bbox_xyxy, width, height, args.roi_padding)
        roi_source = "detector"

    roi_xyxy = roi_tracker.roi_xyxy
    assert roi_xyxy is not None
    x1, y1, x2, y2 = roi_xyxy
    roi = image[y1:y2, x1:x2]
    model_roi = resize_to_square_size(roi, args.imgsz)
    roi_result = predict(model, model_roi, args)
    if roi_result is None or not has_segmentation(roi_result):
        output = image.copy()
        if args.show_bbox:
            cv2.rectangle(output, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), 2)
        color_map = None
        if distribution_picker is not None:
            distribution_picker.clear(image.shape[:2])
        roi_tracker.reset()
        right_panel = (
            [
                color_panel_for_roi(image.shape, color_map, roi_xyxy, "top-1 color mask"),
                color_panel_for_roi(image.shape, color_map, roi_xyxy, "top-2 color mask"),
                distribution_picker.render(image.shape) if distribution_picker else empty_color_panel(image),
            ]
            if color_model is not None
            else None
        )
        return split_screen(output, right_panel), bbox_detector

    roi_tracker.observe_tracking_bounds(
        segmentation_bounds(roi_result, roi.shape[:2]),
        width,
        height,
        tracking_gain=args.roi_tracking_gain,
        tracking_padding=args.roi_tracking_padding,
    )
    rendered_roi = render_segmentation_overlay(
        roi,
        roi_result,
        args.opacity,
        args.segmentation_type,
    )
    output = image.copy()
    output[y1:y2, x1:x2] = rendered_roi
    if args.show_bbox:
        roi_color = (0, 0, 255) if roi_source == "detector" else (40, 220, 40)
        cv2.rectangle(output, (x1, y1), (x2 - 1, y2 - 1), roi_color, 2)
    right_panel = None
    if color_model is not None and color_device is not None:
        valid_mask = segmentation_union_mask(roi_result, roi.shape[:2])
        probabilities = predict_color_probabilities(color_model, roi, args.imgsz, color_device)
        if distribution_picker is not None:
            distribution_picker.update(
                probabilities,
                valid_mask,
                roi_xyxy,
                image.shape[:2],
            )
        top_color_map = render_color_rank_map(probabilities, rank=0, valid_mask=valid_mask)
        second_color_map = render_color_rank_map(probabilities, rank=1, valid_mask=valid_mask)
        right_panel = [
            color_panel_for_roi(image.shape, top_color_map, roi_xyxy, "top-1 color mask"),
            color_panel_for_roi(image.shape, second_color_map, roi_xyxy, "top-2 color mask"),
            distribution_picker.render(image.shape)
            if distribution_picker is not None
            else empty_color_panel(image),
        ]
    return split_screen(output, right_panel), bbox_detector


def resize_to_square_size(image: np.ndarray, size: int) -> np.ndarray:
    if size <= 0:
        return image
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def create_video_writer(path: Path, frame: np.ndarray, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    fourcc = "mp4v" if suffix in {".mp4", ".m4v", ".mov"} else "MJPG"
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*fourcc),
        fps if fps > 0.0 else 30.0,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Display YOLO segmentation masks and an optional dense color map"
    )
    parser.add_argument("--source", required=True, help="Image, video path, or camera index")
    parser.add_argument(
        "--segmentation-type",
        choices=("sticker", "whole-cube"),
        default="sticker",
        help="Segmentation model and rendering mode",
    )
    parser.add_argument("--model", type=Path, help="Override the selected segmentation model weights path")
    parser.add_argument(
        "--color-map-model",
        type=Path,
        help="Trained dense color-map checkpoint; enables split-screen rendering",
    )
    parser.add_argument(
        "--color-map-device",
        default="mps",
        help="Torch device for the dense color-map model",
    )
    parser.add_argument("--bbox-model", type=Path, default=DEFAULT_BBOX_MODEL, help="Cube bbox YOLO weights path")
    parser.add_argument("--bbox-conf", type=float, default=0.25, help="Cube bbox confidence threshold")
    parser.add_argument("--bbox-imgsz", type=int, default=640, help="Cube bbox inference image size")
    parser.add_argument("--roi-padding", type=float, default=0.08, help="Padding ratio around the bbox ROI")
    parser.add_argument("--roi-tracking-gain", type=float, default=1.4, help="How strongly segmentation moves the ROI")
    parser.add_argument("--roi-tracking-padding", type=float, default=0.18, help="Padding around tracked segmentation")
    parser.add_argument("--show-bbox", action=argparse.BooleanOptionalAction, default=True, help="Draw the active ROI")
    parser.add_argument("--conf", type=float, default=0.35, help="Segmentation confidence threshold")
    parser.add_argument("--imgsz", type=int, default=256, help="YOLO inference image size")
    parser.add_argument("--opacity", type=float, default=0.45, help="Mask opacity in the range [0, 1]")
    parser.add_argument("--once", action="store_true", help="Process one frame and exit")
    parser.add_argument("--save", type=Path, help="Optional output image path for image or --once mode")
    parser.add_argument("--save-video", type=Path, help="Optional output path for rendered video frames")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0.0 <= args.opacity <= 1.0:
        raise SystemExit("--opacity must be in the range [0, 1]")
    model_path = args.model or default_model_path(args.segmentation_type)
    if not model_path.is_file():
        raise SystemExit(f"Could not find model weights: {model_path}")

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    color_model = None
    color_device = None
    if args.color_map_model is not None:
        if not args.color_map_model.is_file():
            raise SystemExit(f"Could not find color-map model weights: {args.color_map_model}")
        import torch

        color_device = torch.device(args.color_map_device)
        if color_device.type == "mps" and not torch.backends.mps.is_available():
            print("MPS is unavailable for the color map; falling back to CPU")
            color_device = torch.device("cpu")
        if color_device.type == "cuda" and not torch.cuda.is_available():
            print("CUDA is unavailable for the color map; falling back to CPU")
            color_device = torch.device("cpu")
        from live_recognition.color_segmentation import YOLOColorMapModel

        print(f"Loading color-map model: {args.color_map_model}")
        color_model = YOLOColorMapModel.from_color_checkpoint(
            args.color_map_model,
            map_location="cpu",
        ).to(color_device).eval()
    distribution_picker = ColorDistributionPicker() if color_model is not None else None
    cv2.namedWindow(WINDOW_NAME)
    if distribution_picker is not None:
        cv2.setMouseCallback(WINDOW_NAME, distribution_picker.handle_mouse)
    bbox_detector = None
    roi_tracker = RoiTracker()

    source = parse_source(args.source)
    if is_image_source(source):
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"Could not read image: {source}")
        rendered, bbox_detector = predict_and_render(
            model,
            color_model,
            color_device,
            distribution_picker,
            bbox_detector,
            roi_tracker,
            image,
            args,
        )
        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.save), rendered):
                raise SystemExit(f"Could not write rendered overlay: {args.save}")
            print(f"Saved render: {args.save}")
        if args.save_video:
            raise SystemExit("--save-video requires a video file or camera source")
        while True:
            cv2.imshow(WINDOW_NAME, rendered)
            key = cv2.waitKey(30) & 0xFF
            rendered = refresh_distribution_panel(rendered, distribution_picker)
            if key in (ord("q"), 27, 13):
                break
        cv2.destroyAllWindows()
        return 0

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {source}")

    writer = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rendered, bbox_detector = predict_and_render(
                model,
                color_model,
                color_device,
                distribution_picker,
                bbox_detector,
                roi_tracker,
                frame,
                args,
            )
            if args.save_video:
                if writer is None:
                    writer = create_video_writer(
                        args.save_video,
                        rendered,
                        float(capture.get(cv2.CAP_PROP_FPS)),
                    )
                writer.write(rendered)
            if args.save and args.once:
                args.save.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(args.save), rendered):
                    raise SystemExit(f"Could not write rendered overlay: {args.save}")
            rendered = refresh_distribution_panel(rendered, distribution_picker)
            if args.once:
                while True:
                    cv2.imshow(WINDOW_NAME, rendered)
                    key = cv2.waitKey(30) & 0xFF
                    rendered = refresh_distribution_panel(rendered, distribution_picker)
                    if key in (ord("q"), 27, 13):
                        break
                break
            cv2.imshow(WINDOW_NAME, rendered)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        if writer is not None:
            writer.release()
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
