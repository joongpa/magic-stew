from types import SimpleNamespace

import numpy as np

from live_recognition.roi_tracking import RoiTracker, roi_for_frame


class CountingDetector:
    def __init__(self):
        self.calls = 0

    def detect(self, _frame):
        self.calls += 1
        return SimpleNamespace(bbox_xyxy=(10, 10, 30, 30), confidence=0.9)


class IndependentBBoxDetector:
    def __init__(self):
        self.calls = 0
        self.boxes = [(10, 10, 30, 30), (20, 20, 50, 50)]

    def detect(self, _frame):
        bbox = self.boxes[self.calls]
        self.calls += 1
        return SimpleNamespace(bbox_xyxy=bbox, confidence=0.9)


def test_bbox_runs_once_until_second_stage_miss_then_reacquires():
    frame = np.zeros((60, 60, 3), dtype=np.uint8)
    detector = CountingDetector()
    tracker = RoiTracker()

    roi, candidate, source = roi_for_frame(frame, detector, tracker, 0.0)
    assert roi == (10, 10, 30, 30)
    assert candidate is not None
    assert source == "detector"

    tracker.observe_tracking_bounds((10, 5, 20, 15), 60, 60, tracking_padding=0.0)
    roi, candidate, source = roi_for_frame(frame, detector, tracker, 0.0)
    assert detector.calls == 1
    assert roi == (15, 10, 35, 30)
    assert candidate is not None
    assert source == "tracker"

    tracker.observe_tracking_bounds(None, 60, 60)
    roi, candidate, source = roi_for_frame(frame, detector, tracker, 0.0)
    assert detector.calls == 2
    assert roi == (10, 10, 30, 30)
    assert candidate is not None
    assert source == "detector"


def test_roi_padding_survives_tracker_reset_for_independent_bbox_detection():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detector = IndependentBBoxDetector()
    tracker = RoiTracker()

    roi, _candidate, source = roi_for_frame(frame, detector, tracker, 0.8)
    assert source == "detector"
    assert roi == (0, 0, 52, 52)
    assert tracker.roi_padding == 0.8

    tracker.observe_tracking_bounds(None, 100, 100)
    roi, _candidate, source = roi_for_frame(frame, detector, tracker, 0.08)

    assert detector.calls == 2
    assert source == "detector"
    assert roi == (0, 0, 78, 78)
    assert tracker.roi_padding == 0.8
