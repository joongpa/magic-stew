from types import SimpleNamespace

import numpy as np

from live_recognition.roi_tracking import RoiTracker, roi_for_frame


class CountingDetector:
    def __init__(self):
        self.calls = 0

    def detect(self, _frame):
        self.calls += 1
        return SimpleNamespace(bbox_xyxy=(10, 10, 30, 30), confidence=0.9)


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
