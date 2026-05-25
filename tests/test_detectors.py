import torch
import pytest
import numpy as np

from live_recognition.detectors import (
    _bbox_from_mask,
    _bbox_from_xyxy,
    _best_confident_box_index,
    _convex_hull_mask,
    _resolve_model_path,
    _resolve_sam3_model_path,
)


def test_best_confident_box_index_excludes_all_boxes_below_threshold():
    confidences = torch.tensor([0.12, 0.72, 0.94])

    assert _best_confident_box_index(confidences, 0.95) is None


def test_best_confident_box_index_selects_best_box_above_threshold():
    confidences = torch.tensor([0.97, 0.72, 0.99])

    assert _best_confident_box_index(confidences, 0.95) == 2


def test_resolve_sam3_model_path_reports_missing_weights():
    with pytest.raises(FileNotFoundError, match="SAM 3 weights not found"):
        _resolve_sam3_model_path("missing-sam3.pt")


def test_resolve_model_path_reports_missing_weights():
    with pytest.raises(FileNotFoundError, match="YOLO weights not found"):
        _resolve_model_path("missing-yolo.pt", "YOLO")


def test_convex_hull_mask_fills_between_visible_regions():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:4, 2:4] = 255
    mask[2:4, 6:8] = 255
    mask[6:8, 2:4] = 255
    mask[6:8, 6:8] = 255

    hull_mask, hull = _convex_hull_mask(mask)

    assert hull is not None
    assert hull_mask[5, 5] == 255
    assert hull_mask.sum() > mask.sum()


def test_bbox_from_mask_uses_nonzero_pixels():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 3:7] = 255

    assert _bbox_from_mask(mask) == (3, 2, 7, 5)


def test_bbox_from_xyxy_rounds_model_box_coordinates():
    box = torch.tensor([2.2, 3.7, 8.4, 9.6])

    assert _bbox_from_xyxy(box) == (2, 4, 8, 10)
