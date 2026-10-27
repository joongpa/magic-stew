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
build_parser = render_script.build_parser


def test_random_sticker_colors_is_opt_in_parser_flag():
    assert not build_parser().parse_args(["--source", "frame.png"]).random_sticker_colors
    assert build_parser().parse_args(
        ["--source", "frame.png", "--random-sticker-colors"]
    ).random_sticker_colors


def test_masks_only_is_opt_in_parser_flag():
    assert not build_parser().parse_args(["--source", "frame.png"]).masks_only
    assert build_parser().parse_args(["--source", "frame.png", "--masks-only"]).masks_only


def test_sticker_mask_erosion_parser_flag():
    assert build_parser().parse_args(["--source", "frame.png"]).sticker_mask_erosion == 0
    assert build_parser().parse_args(
        ["--source", "frame.png", "--sticker-mask-erosion", "2"]
    ).sticker_mask_erosion == 2


def test_sticker_mask_erosion_shrinks_sticker_and_combined_masks():
    masks = np.zeros((2, 9, 9), dtype=np.float32)
    masks[:, 2:7, 2:7] = 1.0
    result = SimpleNamespace(
        masks=SimpleNamespace(data=masks),
        boxes=SimpleNamespace(cls=np.array([0, 1])),
        names={0: "cube", 1: "sticker"},
    )

    sticker_mask = segmentation_union_mask(result, (9, 9), "sticker", sticker_mask_erosion=1)
    combined_mask = segmentation_union_mask(result, (9, 9), "combined", sticker_mask_erosion=1)

    assert sticker_mask is not None
    assert combined_mask is not None
    assert int(sticker_mask.sum()) == 9
    assert int(combined_mask.sum()) == 9


def test_predict_requests_high_resolution_masks():
    calls = {}

    class StubModel:
        def predict(self, **kwargs):
            calls.update(kwargs)
            return [object()]

    args = SimpleNamespace(conf=0.25, imgsz=256)
    assert render_script.predict(StubModel(), np.zeros((32, 32, 3), dtype=np.uint8), args) is not None
    assert calls["retina_masks"] is True


def test_masks_only_uses_a_black_background():
    image = np.full((10, 10, 3), 80, dtype=np.uint8)
    masks = np.zeros((1, 10, 10), dtype=np.float32)
    masks[0, 2:6, 3:7] = 1.0
    result = SimpleNamespace(
        masks=SimpleNamespace(data=masks),
        boxes=SimpleNamespace(cls=np.array([0])),
        names={0: "white"},
    )

    output = render_segmentation_overlay(
        image,
        result,
        opacity=0.45,
        segmentation_type="sticker",
        masks_only=True,
    )

    assert np.all(output[0, 0] == 0)
    assert np.all(output[9, 9] == 0)
    assert np.any(output[2:6, 3:7] != 0)


def test_predict_and_render_reapplies_roi_padding_after_bbox_reacquisition(monkeypatch):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    seen_roi_shapes = []

    class StubBBoxDetector:
        def __init__(self):
            self.calls = 0
            self.boxes = [(10, 10, 30, 30), (20, 20, 50, 50)]

        def detect(self, _frame):
            bbox = self.boxes[self.calls]
            self.calls += 1
            return SimpleNamespace(bbox_xyxy=bbox, confidence=0.9)

    def no_segmentation(_model, roi, _args):
        seen_roi_shapes.append(roi.shape)
        return None

    monkeypatch.setattr(render_script, "predict", no_segmentation)
    args = SimpleNamespace(
        bbox_model="bbox.pt",
        bbox_conf=0.25,
        bbox_imgsz=640,
        roi_padding=0.8,
        imgsz=256,
        conf=0.25,
        show_bbox=False,
        opacity=0.45,
        segmentation_type="sticker",
        roi_tracking_gain=1.4,
        roi_tracking_padding=0.18,
        random_sticker_colors=False,
    )
    tracker = render_script.RoiTracker()
    detector = StubBBoxDetector()

    render_script.predict_and_render(None, None, None, None, detector, tracker, frame, args)
    render_script.predict_and_render(None, None, None, None, detector, tracker, frame, args)

    assert seen_roi_shapes == [(52, 52, 3), (78, 78, 3)]


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

    assert np.all(output[6:8, 6:8] == (255, 0, 255))
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

    assert np.all(output[2:3, 2:3] == (255, 0, 255))
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

    output = render_segmentation_overlay(
        image,
        result,
        opacity=1.0,
        segmentation_type="sticker",
        random_sticker_colors=True,
    )

    assert np.all(output[1:5, 1:5] == 0)
    assert np.all(output[6:8, 6:8] == render_script.sticker_instance_color(1))


def test_sticker_segmentation_renders_each_detection_with_a_distinct_stable_color():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    masks = np.zeros((2, 10, 10), dtype=np.float32)
    masks[0, 1:5, 1:5] = 1.0
    masks[1, 5:9, 5:9] = 1.0
    result = SimpleNamespace(
        masks=SimpleNamespace(data=masks),
        boxes=SimpleNamespace(cls=np.array([0, 1])),
        names={0: "white", 1: "yellow"},
    )

    output = render_segmentation_overlay(
        image,
        result,
        opacity=1.0,
        segmentation_type="sticker",
        random_sticker_colors=True,
    )

    first_color = output[2, 2].tolist()
    second_color = output[6, 6].tolist()
    assert first_color != second_color
    assert np.all(output[2:4, 2:4] == first_color)
    assert np.all(output[6:8, 6:8] == second_color)

    repeated = render_segmentation_overlay(
        image,
        result,
        opacity=1.0,
        segmentation_type="sticker",
        random_sticker_colors=True,
    )
    assert np.array_equal(output, repeated)


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

    output = render_segmentation_overlay(
        image,
        result,
        opacity=1.0,
        segmentation_type="combined",
        random_sticker_colors=True,
    )

    assert np.all(output[3, 3] == render_script.sticker_instance_color(0))
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
    assert all(call.args[4] == 1 for call in draw_contours.call_args_list)
    assert all(call.args[3] == render_script.STICKER_BORDER_COLOR_BGR for call in draw_contours.call_args_list)


def test_combined_segmentation_keeps_sticker_outline_after_mask_cleanup():
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    masks = np.zeros((2, 12, 12), dtype=np.float32)
    masks[0, 3:9, 3:9] = 1.0  # cube
    masks[1, 3:9, 3:9] = 1.0  # sticker
    result = SimpleNamespace(
        masks=SimpleNamespace(data=masks),
        boxes=SimpleNamespace(cls=np.array([0, 1])),
        names={0: "cube", 1: "sticker"},
    )

    output = render_segmentation_overlay(
        image,
        result,
        opacity=1.0,
        segmentation_type="combined",
        random_sticker_colors=True,
    )

    sticker_color = render_script.sticker_instance_color(0)
    assert np.any(np.all(output == sticker_color, axis=2))
