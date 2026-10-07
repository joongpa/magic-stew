import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "render_yolo_sticker_segmentation.py"
SPEC = importlib.util.spec_from_file_location("render_yolo_sticker_segmentation", SCRIPT_PATH)
render_script = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(render_script)

render_segmentation_overlay = render_script.render_segmentation_overlay


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
