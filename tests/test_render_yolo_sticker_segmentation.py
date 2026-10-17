import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "render_yolo_sticker_segmentation.py"
SPEC = importlib.util.spec_from_file_location("render_yolo_sticker_segmentation", SCRIPT_PATH)
render_script = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(render_script)

render_segmentation_overlay = render_script.render_segmentation_overlay
segmentation_union_mask = render_script.segmentation_union_mask


def test_whole_cube_segmentation_renders_only_largest_mask():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    masks = np.zeros((2, 10, 10), dtype=np.float32)
    masks[0, 1:3, 1:3] = 1.0
    masks[1, 5:9, 5:9] = 1.0
    result = SimpleNamespace(
        masks=SimpleNamespace(data=masks),
        boxes=SimpleNamespace(cls=np.array([0, 0])),
        names={0: "cube"},
    )

    output = render_segmentation_overlay(image, result, opacity=1.0, segmentation_type="whole-cube")

    assert np.all(output[5:9, 5:9] == (255, 0, 255))
    assert np.all(output[1:3, 1:3] == 0)


def test_whole_cube_segmentation_prefers_cube_class_over_larger_sticker_mask():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    masks = np.zeros((2, 10, 10), dtype=np.float32)
    masks[0, 1:4, 1:4] = 1.0
    masks[1, 4:9, 4:9] = 1.0
    result = SimpleNamespace(
        masks=SimpleNamespace(data=masks),
        boxes=SimpleNamespace(cls=np.array([0, 1])),
        names={0: "cube", 1: "sticker"},
    )

    output = render_segmentation_overlay(image, result, opacity=1.0, segmentation_type="whole-cube")

    assert np.all(output[1:4, 1:4] == (255, 0, 255))
    assert np.all(output[4:9, 4:9] == 0)


def test_whole_cube_segmentation_does_not_fallback_to_sticker_when_cube_is_missing():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    masks = np.zeros((1, 10, 10), dtype=np.float32)
    masks[0, 2:8, 2:8] = 1.0
    result = SimpleNamespace(
        masks=SimpleNamespace(data=masks),
        boxes=SimpleNamespace(cls=np.array([1])),
        names={0: "cube", 1: "sticker"},
    )

    output = render_segmentation_overlay(image, result, opacity=1.0, segmentation_type="whole-cube")

    assert np.all(output == 0)


def test_sticker_segmentation_renders_only_sticker_class_for_combined_model():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    masks = np.zeros((2, 10, 10), dtype=np.float32)
    masks[0, 1:5, 1:5] = 1.0  # cube
    masks[1, 5:9, 5:9] = 1.0  # sticker
    result = SimpleNamespace(
        masks=SimpleNamespace(data=masks),
        boxes=SimpleNamespace(cls=np.array([0, 1])),
        names={0: "cube", 1: "sticker"},
    )

    output = render_segmentation_overlay(image, result, opacity=1.0, segmentation_type="sticker")

    assert np.all(output[1:5, 1:5] == 0)
    assert np.all(output[5:9, 5:9] == (0, 255, 255))


def test_sticker_segmentation_still_renders_all_masks():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    masks = np.zeros((2, 10, 10), dtype=np.float32)
    masks[0, 1:3, 1:3] = 1.0
    masks[1, 5:9, 5:9] = 1.0
    result = SimpleNamespace(
        masks=SimpleNamespace(data=masks),
        boxes=SimpleNamespace(cls=np.array([0, 1])),
        names={0: "white", 1: "yellow"},
    )

    output = render_segmentation_overlay(image, result, opacity=1.0, segmentation_type="sticker")

    assert np.all(output[1:3, 1:3] == (255, 255, 255))
    assert np.all(output[5:9, 5:9] == (0, 255, 255))


def test_combined_segmentation_renders_only_cube_sticker_intersection():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    masks = np.zeros((2, 10, 10), dtype=np.float32)
    masks[0, 2:8, 2:8] = 1.0  # cube
    masks[1, 1:5, 1:5] = 1.0  # sticker, partly outside cube
    result = SimpleNamespace(
        masks=SimpleNamespace(data=masks),
        boxes=SimpleNamespace(cls=np.array([0, 1])),
        names={0: "cube", 1: "sticker"},
    )

    output = render_segmentation_overlay(image, result, opacity=1.0, segmentation_type="combined")

    assert np.all(output[2:5, 2:5] == (0, 255, 255))
    assert np.all(output[1:2, 1:5] == 0)
    assert np.all(output[5:8, 5:8] == 0)

    intersection = segmentation_union_mask(result, (10, 10), "combined")
    assert intersection is not None
    assert int(intersection.sum()) == 9


def test_combined_segmentation_draws_each_sticker_outline_separately():
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    masks = np.zeros((3, 12, 12), dtype=np.float32)
    masks[0, 1:11, 1:11] = 1.0  # cube
    masks[1, 2:5, 2:5] = 1.0  # sticker 1
    masks[2, 7:10, 7:10] = 1.0  # sticker 2
    result = SimpleNamespace(
        masks=SimpleNamespace(data=masks),
        boxes=SimpleNamespace(cls=np.array([0, 1, 1])),
        names={0: "cube", 1: "sticker"},
    )

    with patch.object(render_script.cv2, "drawContours", wraps=render_script.cv2.drawContours) as draw_contours:
        render_segmentation_overlay(image, result, opacity=1.0, segmentation_type="combined")

    assert draw_contours.call_count == 2
