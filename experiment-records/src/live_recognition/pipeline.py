from __future__ import annotations

from live_recognition.detectors import CubeDetector, HeuristicCubeDetector
from live_recognition.types import FrameResult


class CubeOrientationPipeline:
    def __init__(self, detector: CubeDetector | None = None) -> None:
        self.detector = detector or HeuristicCubeDetector()

    def process_frame(self, frame_bgr):
        candidate = self.detector.detect(frame_bgr)
        return FrameResult(
            candidate=candidate,
            orientations=[],
        )
