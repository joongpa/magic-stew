"""Confidence provenance, four-point mapping, and shared ROI recovery."""

import argparse
from types import SimpleNamespace

import numpy as np
import pytest

from live_recognition.roi_tracking import RoiTracker
from live_recognition.sticker_quad_schema import CLASS_NAMES
from scripts import render_yolo_sticker_quads as renderer


def args(*options):
    return renderer.build_parser().parse_args(["--source", "0", "--no-show-fps", *options])


def prediction(scores=(0.9,), classes=(0,), presence=None):
    points = np.tile(
        np.asarray([[5, 5], [15, 5], [15, 15], [5, 15]], np.float32), (len(scores), 1, 1)
    )
    if presence is None:
        presence = np.full((len(scores), 4), 0.999, np.float32)
    return SimpleNamespace(
        boxes=SimpleNamespace(conf=np.asarray(scores), cls=np.asarray(classes)),
        keypoints=SimpleNamespace(xy=points, conf=np.asarray(presence)),
        names=dict(enumerate(CLASS_NAMES)),
    )


class Detector:
    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return SimpleNamespace(bbox_xyxy=(10, 10, 30, 30), confidence=0.82)


class Model:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [self.result]


def test_four_points_and_confidence_channels_are_kept_separate():
    detection = renderer.extract_quads(prediction(scores=(0.42,)))[0]
    assert detection["points"].shape == (4, 2)
    assert detection["object_confidence"] == pytest.approx(0.42)
    np.testing.assert_allclose(detection["keypoint_presence"], [0.999] * 4)


def test_class_overrides_accept_groups_and_individual_names():
    thresholds = renderer.parse_class_conf("edge=0.6,blue_corner=0.8,white=0.4,13=0.7")
    assert thresholds[CLASS_NAMES.index("blue corner")] == 0.8
    assert thresholds[CLASS_NAMES.index("blue edge")] == 0.7
    assert thresholds[CLASS_NAMES.index("white edge")] == 0.4
    for bad in ["pink=0.4", "corner=nan", "blue=2", "blue", "18=0.5"]:
        with pytest.raises(argparse.ArgumentTypeError):
            renderer.parse_class_conf(bad)


def test_class_score_filters_objects_even_when_presence_is_high():
    options = args("--pose-conf", ".8", "--roi-padding", "0", "--full-frame")
    model, tracker, detector = (
        Model(prediction(scores=(0.4, 0.9), classes=(0, 1))),
        RoiTracker(),
        Detector(),
    )
    _, report = renderer.render_frame(
        np.zeros((80, 80, 3), np.uint8), detector, model, tracker, options
    )
    assert len(report["detections"]) == 1
    assert report["detections"][0]["object_confidence"] == pytest.approx(0.9)
    assert report["cube_bbox_confidence"] == 0.82
    np.testing.assert_allclose(
        report["detections"][0]["keypoints_full_frame"], [[15, 15], [25, 15], [25, 25], [15, 25]]
    )
    assert model.calls[0]["imgsz"] == 128


def test_presence_threshold_filters_points_without_replacing_object_score():
    options = args("--keypoint-conf", ".5", "--roi-padding", "0")
    model = Model(prediction(scores=(0.91,), presence=[[0.1, 0.9, 0.8, 0.7]]))
    _, report = renderer.render_frame(
        np.zeros((80, 80, 3), np.uint8), Detector(), model, RoiTracker(), options
    )
    assert report["detections"][0]["displayed_points"] == [False, True, True, True]
    assert report["detections"][0]["object_confidence"] == pytest.approx(0.91)


def test_lower_class_override_reaches_model_and_tracker_reacquires_after_miss():
    options = args(
        "--pose-conf",
        ".8",
        "--pose-class-conf",
        "corner=0.02",
        "--track-conf",
        ".05",
        "--roi-padding",
        "0",
    )
    model, detector, tracker = Model(prediction(scores=(0.03,))), Detector(), RoiTracker()
    frame = np.zeros((80, 80, 3), np.uint8)
    _, first = renderer.render_frame(frame, detector, model, tracker, options)
    assert first["roi_source"] == "detector" and len(first["detections"]) == 1
    assert model.calls[0]["conf"] == 0.02
    # Below track-conf: draw it but require bbox acquisition on the next frame.
    assert tracker.roi_xyxy is None
    model.result = prediction(scores=(0.9,))
    renderer.render_frame(frame, detector, model, tracker, options)
    _, tracked = renderer.render_frame(frame, detector, model, tracker, options)
    assert detector.calls == 2 and tracked["roi_source"] == "tracker"
    model.result = SimpleNamespace(boxes=None, keypoints=None)
    renderer.render_frame(frame, detector, model, tracker, options)
    assert tracker.roi_xyxy is None


@pytest.mark.parametrize("class_id", [12, 13, 14])
def test_initial_is_sticker_color_and_outline_is_piece_type(class_id):
    options = args("--no-show-keypoints", "--no-show-confidences", "--no-show-labels")
    output = np.zeros((100, 100, 3), np.uint8)
    detection = renderer.extract_quads(prediction(classes=(class_id,)))[0]
    quad = np.asarray([[20, 20], [80, 20], [80, 80], [20, 80]], float)
    renderer.draw_quad(output, detection, quad, np.ones(4, bool), options)
    # The interior is unfilled away from the centered blue B.
    np.testing.assert_array_equal(output[30, 30], [0, 0, 0])
    interior = output[35:65, 35:65].reshape(-1, 3).astype(float)
    expected_letter = np.asarray(renderer.COLORS_BGR[4])
    assert (np.max(np.abs(interior - expected_letter), axis=1) < 2).any()
    actual = output[20, 50].astype(float)
    expected = np.asarray(renderer.TYPE_OUTLINES_BGR[class_id % 3], float)
    assert actual.max() > 220  # Antialiasing can reduce intensity, preserving hue.
    np.testing.assert_allclose(actual / actual.max(), expected / expected.max(), atol=0.01)
