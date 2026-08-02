import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "test_piece_seg_roi.py"
SPEC = importlib.util.spec_from_file_location("test_piece_seg_roi_script", SCRIPT_PATH)
piece_seg_script = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(piece_seg_script)


class StubBboxDetector:
    def detect(self, frame):
        return SimpleNamespace(bbox_xyxy=(5, 5, 25, 25), confidence=0.9)


class CountingBboxDetector:
    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return SimpleNamespace(bbox_xyxy=(10, 10, 30, 30), confidence=0.9)


class StubBoxes:
    cls = [SimpleNamespace(item=lambda: 0)]
    conf = [SimpleNamespace(item=lambda: 0.9)]

    def __len__(self):
        return 1


class StubMixedBoxes:
    cls = [SimpleNamespace(item=lambda: 0), SimpleNamespace(item=lambda: 1)]
    conf = [SimpleNamespace(item=lambda: 0.9), SimpleNamespace(item=lambda: 0.8)]

    def __len__(self):
        return 2


class StubTwoFaceBoxes:
    cls = [SimpleNamespace(item=lambda: 0), SimpleNamespace(item=lambda: 0)]
    conf = [SimpleNamespace(item=lambda: 0.9), SimpleNamespace(item=lambda: 0.8)]

    def __len__(self):
        return 2


class StubPoseBoxes:
    cls = [SimpleNamespace(item=lambda: 1)]
    conf = [SimpleNamespace(item=lambda: 0.9)]

    def __len__(self):
        return 1


class StubFacePoseBoxes:
    cls = [SimpleNamespace(item=lambda: 0)]
    conf = [SimpleNamespace(item=lambda: 0.9)]

    def __len__(self):
        return 1


class TensorLike:
    def __init__(self, data):
        self.data = data

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data


class StubCornerPoseBoxes:
    cls = TensorLike(np.array([0], dtype=np.float32))
    conf = TensorLike(np.array([0.9], dtype=np.float32))

    def __len__(self):
        return 1


class StubStickerModel:
    names = {0: "U_sticker", 1: "F_sticker", 2: "R_sticker"}

    def predict(self, source, conf, imgsz, verbose):
        mask = np.zeros((1, source.shape[0], source.shape[1]), dtype=np.float32)
        y1, y2 = source.shape[0] // 4, source.shape[0] * 3 // 4
        x1, x2 = source.shape[1] // 4, source.shape[1] * 3 // 4
        mask[:, y1:y2, x1:x2] = 1.0
        masks = SimpleNamespace(
            data=SimpleNamespace(
                detach=lambda: SimpleNamespace(
                    cpu=lambda: SimpleNamespace(numpy=lambda: mask)
                )
            )
        )
        return [SimpleNamespace(masks=masks, boxes=StubBoxes())]


class StubFaceSegmentationModel(StubStickerModel):
    names = {0: "face"}


class StubOverlappingFaceSegmentationModel:
    names = {0: "face"}

    def predict(self, source, conf, imgsz, verbose):
        mask = np.zeros((2, source.shape[0], source.shape[1]), dtype=np.float32)
        mask[0, 40:180, 40:180] = 1.0
        mask[1, 30:130, 30:130] = 1.0
        masks = SimpleNamespace(
            data=SimpleNamespace(
                detach=lambda: SimpleNamespace(
                    cpu=lambda: SimpleNamespace(numpy=lambda: mask)
                )
            )
        )
        return [SimpleNamespace(masks=masks, boxes=StubTwoFaceBoxes())]


class StubPoseStickerModel:
    names = {0: "U_sticker"}

    def predict(self, source, conf, imgsz, verbose):
        keypoints = np.array(
            [
                [
                    [5.0, 5.0],
                    [15.0, 5.0],
                    [15.0, 15.0],
                    [5.0, 15.0],
                ]
            ],
            dtype=np.float32,
        )
        return [
            SimpleNamespace(
                keypoints=SimpleNamespace(xy=TensorLike(keypoints)),
                boxes=StubPoseBoxes(),
            )
        ]


class StubVisibleFacePoseModel:
    names = {0: "U_sticker"}

    def predict(self, source, conf, imgsz, verbose):
        keypoints = np.array(
            [
                [
                    [5.0, 5.0],
                    [15.0, 5.0],
                    [15.0, 15.0],
                    [5.0, 15.0],
                ]
            ],
            dtype=np.float32,
        )
        return [
            SimpleNamespace(
                keypoints=SimpleNamespace(xy=TensorLike(keypoints)),
                boxes=StubFacePoseBoxes(),
            )
        ]


class StubCornerPoseModel:
    names = {0: "corner"}

    def __init__(self, point_xy=(192.0, 128.0)):
        self.point_xy = point_xy

    def predict(self, source, conf, imgsz, verbose):
        keypoints = np.asarray([[self.point_xy]], dtype=np.float32)
        keypoint_conf = np.asarray([[0.9]], dtype=np.float32)
        return [
            SimpleNamespace(
                keypoints=SimpleNamespace(
                    xy=TensorLike(keypoints),
                    conf=TensorLike(keypoint_conf),
                ),
                boxes=StubCornerPoseBoxes(),
            )
        ]


class StubNoPoseModel:
    names = {0: "corner"}

    def predict(self, source, conf, imgsz, verbose):
        return []


class StubPoseSolver:
    def solve_correspondences(self, correspondence, image_size, observed_graph=None):
        return None


class ResetRecordingPoseSolver:
    def __init__(self):
        self.reset_count = 0

    def solve_correspondences(self, correspondence, image_size, observed_graph=None):
        return None

    def reset_temporal_tracking(self):
        self.reset_count += 1


class RaisingPoseSolver:
    def solve_correspondences(self, correspondence, image_size, observed_graph=None):
        return None


class RecordingStickerModel:
    names = {0: "U_sticker", 1: "F_sticker", 2: "R_sticker"}

    def __init__(self):
        self.source_shape = None
        self.source = None

    def predict(self, source, conf, imgsz, verbose):
        self.source_shape = source.shape[:2]
        self.source = source.copy()
        mask = np.zeros((1, source.shape[0], source.shape[1]), dtype=np.float32)
        mask[:, 2:8, 5:15] = 1.0
        masks = SimpleNamespace(
            data=SimpleNamespace(
                detach=lambda: SimpleNamespace(
                    cpu=lambda: SimpleNamespace(numpy=lambda: mask)
                )
            )
        )
        return [SimpleNamespace(masks=masks, boxes=StubBoxes())]


class ShiftingStickerModel:
    names = {0: "U_sticker"}

    def __init__(self, offsets):
        self.offsets = list(offsets)
        self.calls = 0

    def predict(self, source, conf, imgsz, verbose):
        offset_x, offset_y = self.offsets[min(self.calls, len(self.offsets) - 1)]
        self.calls += 1
        mask = np.zeros((1, source.shape[0], source.shape[1]), dtype=np.float32)
        mask[:, 5 + offset_y : 15 + offset_y, 5 + offset_x : 15 + offset_x] = 1.0
        masks = SimpleNamespace(
            data=SimpleNamespace(
                detach=lambda: SimpleNamespace(
                    cpu=lambda: SimpleNamespace(numpy=lambda: mask)
                )
            )
        )
        return [SimpleNamespace(masks=masks, boxes=StubBoxes())]


class MasklessStickerModel:
    names = {0: "U_sticker"}

    def predict(self, source, conf, imgsz, verbose):
        mask = np.zeros((1, source.shape[0], source.shape[1]), dtype=np.float32)
        masks = SimpleNamespace(
            data=SimpleNamespace(
                detach=lambda: SimpleNamespace(
                    cpu=lambda: SimpleNamespace(numpy=lambda: mask)
                )
            )
        )
        return [SimpleNamespace(masks=masks, boxes=StubBoxes())]


class VariableSizeStickerModel:
    names = {0: "U_sticker"}

    def __init__(self, sizes):
        self.sizes = list(sizes)
        self.calls = 0

    def predict(self, source, conf, imgsz, verbose):
        size = self.sizes[min(self.calls, len(self.sizes) - 1)]
        self.calls += 1
        mask = np.zeros((1, source.shape[0], source.shape[1]), dtype=np.float32)
        size = min(size, source.shape[0], source.shape[1])
        y1 = (source.shape[0] - size) // 2
        x1 = (source.shape[1] - size) // 2
        mask[:, y1 : y1 + size, x1 : x1 + size] = 1.0
        masks = SimpleNamespace(
            data=SimpleNamespace(
                detach=lambda: SimpleNamespace(
                    cpu=lambda: SimpleNamespace(numpy=lambda: mask)
                )
            )
        )
        return [SimpleNamespace(masks=masks, boxes=StubBoxes())]


class StubMixedStickerModel:
    names = {0: "U_sticker", 1: "F_sticker", 2: "R_sticker"}

    def predict(self, source, conf, imgsz, verbose):
        mask = np.zeros((2, source.shape[0], source.shape[1]), dtype=np.float32)
        y1, y2 = source.shape[0] // 4, source.shape[0] * 3 // 4
        mask[0, y1:y2, source.shape[1] // 4 : source.shape[1] * 3 // 4] = 1.0
        mask[1, y1:y2, source.shape[1] * 3 // 5 : source.shape[1]] = 1.0
        masks = SimpleNamespace(
            data=SimpleNamespace(
                detach=lambda: SimpleNamespace(
                    cpu=lambda: SimpleNamespace(numpy=lambda: mask)
                )
            )
        )
        return [SimpleNamespace(masks=masks, boxes=StubMixedBoxes())]


def default_args(**overrides):
    values = {
        "sticker_conf": 0.25,
        "sticker_imgsz": 256,
        "enable_sticker_segmentation": True,
        "roi_padding": 0.0,
        "roi_tracking_gain": 1.4,
        "roi_tracking_padding": 0.18,
        "segmentation_padding_px": 0,
        "debug_overlay": False,
        "show_bbox": False,
        "show_confidence": False,
        "show_original_underlay": False,
        "mask_opacity": 1.0,
        "color_plot_samples_per_frame": 800,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_process_frame_uses_same_output_when_collecting_plot_samples():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    args = default_args(debug_overlay=True, show_bbox=True)

    output, bgr_samples = piece_seg_script.process_frame(
        frame,
        StubBboxDetector(),
        StubStickerModel(),
        args,
    )
    plot_output, plot_bgr_samples = piece_seg_script.process_frame(
        frame,
        StubBboxDetector(),
        StubStickerModel(),
        args,
    )

    assert np.array_equal(output, plot_output)
    assert np.array_equal(bgr_samples, plot_bgr_samples)
    assert len(bgr_samples) > 1


def test_process_frame_skips_sticker_segmentation_by_default():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    args = SimpleNamespace(
        roi_padding=0.0,
        roi_tracking_gain=1.4,
        roi_tracking_padding=0.18,
        debug_overlay=False,
        show_bbox=False,
    )

    output, bgr_samples = piece_seg_script.process_frame(
        frame,
        StubBboxDetector(),
        StubStickerModel(),
        args,
    )

    assert bgr_samples.shape == (0, 3)
    assert np.array_equal(output, frame)


def test_process_frame_requires_segmentation_masks():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :] = [40, 40, 40]
    frame[10:20, 10:20] = [0, 0, 220]

    output, bgr_samples = piece_seg_script.process_frame(
        frame,
        StubBboxDetector(),
        StubPoseStickerModel(),
        default_args(),
    )

    assert bgr_samples.shape == (0, 3)
    assert np.array_equal(output, frame)


def test_process_frame_ignores_pose_only_outputs():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)

    output, bgr_samples = piece_seg_script.process_frame(
        frame,
        StubBboxDetector(),
        StubVisibleFacePoseModel(),
        default_args(),
    )

    assert bgr_samples.shape == (0, 3)
    assert np.array_equal(output, frame)


def test_process_frame_can_show_original_image_under_overlay():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :] = [40, 40, 40]
    frame[10:20, 10:20] = [0, 0, 220]
    args = default_args(show_original_underlay=True, mask_opacity=0.5)

    output, bgr_samples = piece_seg_script.process_frame(
        frame,
        StubBboxDetector(),
        StubStickerModel(),
        args,
    )

    assert len(bgr_samples) > 1
    assert np.array_equal(output[0, 0], np.asarray([40, 40, 40], dtype=np.uint8))
    expected_blend = (
        np.asarray([0, 0, 220], dtype=np.float32) * 0.5
        + np.asarray(piece_seg_script.DEFAULT_STICKER_MASK_COLOR, dtype=np.float32) * 0.5
    ).astype(np.uint8)
    assert np.array_equal(output[15, 15], expected_blend)


def test_process_frame_accepts_face_segmentation_class():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :] = [40, 40, 40]
    args = default_args(mask_opacity=1.0)

    output, bgr_samples = piece_seg_script.process_frame(
        frame,
        StubBboxDetector(),
        StubFaceSegmentationModel(),
        args,
    )

    assert len(bgr_samples) > 0
    assert output.shape == frame.shape
    assert np.any(output != frame)


def test_roi_tracker_runs_bbox_once_and_moves_roi_from_sticker_bounds_center():
    frame = np.zeros((60, 60, 3), dtype=np.uint8)
    detector = CountingBboxDetector()
    tracker = piece_seg_script.RoiTracker()
    sticker_model = ShiftingStickerModel(offsets=[(0, 0), (3, 0)])
    args = default_args(show_bbox=True, sticker_imgsz=20, roi_tracking_padding=0.0)

    first_output, _ = piece_seg_script.process_frame(
        frame,
        detector,
        sticker_model,
        args,
        roi_tracker=tracker,
    )
    assert detector.calls == 1
    assert tracker.roi_xyxy == (10, 10, 30, 30)
    assert first_output[0, 0].tolist() != [40, 220, 40]
    assert first_output[10, 10].tolist() == [0, 0, 255]

    second_output, _ = piece_seg_script.process_frame(
        frame,
        detector,
        sticker_model,
        args,
        roi_tracker=tracker,
    )

    assert detector.calls == 1
    assert tracker.roi_xyxy == (14, 10, 34, 30)
    assert second_output[10, 10].tolist() != [40, 220, 40]
    assert second_output[10, 14].tolist() == [40, 220, 40]
    assert second_output[10, 14].tolist() != [0, 0, 255]


def test_roi_tracker_runs_bbox_once_and_moves_roi_from_pose_keypoint_bounds():
    frame = np.zeros((60, 60, 3), dtype=np.uint8)
    detector = CountingBboxDetector()
    tracker = piece_seg_script.RoiTracker()
    args = default_args(
        enable_sticker_segmentation=False,
        show_bbox=True,
        roi_tracking_padding=0.0,
    )

    first_output, _ = piece_seg_script.process_frame(
        frame,
        detector,
        None,
        args,
        roi_tracker=tracker,
        pose_model=StubCornerPoseModel(point_xy=(192.0, 128.0)),
        pose_solver=StubPoseSolver(),
    )

    assert detector.calls == 1
    assert tracker.roi_xyxy == (17, 10, 37, 30)
    assert first_output[10, 17].tolist() == [0, 0, 255]

    second_output, _ = piece_seg_script.process_frame(
        frame,
        detector,
        None,
        args,
        roi_tracker=tracker,
        pose_model=StubCornerPoseModel(point_xy=(128.0, 128.0)),
        pose_solver=StubPoseSolver(),
    )

    assert detector.calls == 1
    assert tracker.roi_xyxy == (17, 10, 37, 30)
    assert second_output[10, 17].tolist() == [40, 220, 40]


def test_process_frame_hides_pose_graph_coords_by_default(monkeypatch):
    calls = []
    frame = np.zeros((60, 60, 3), dtype=np.uint8)
    args = default_args(
        enable_sticker_segmentation=False,
        show_bbox=True,
        roi_tracking_padding=0.0,
    )

    monkeypatch.setattr(
        piece_seg_script,
        "draw_pose_graph_correspondence",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    piece_seg_script.process_frame(
        frame,
        CountingBboxDetector(),
        None,
        args,
        roi_tracker=piece_seg_script.RoiTracker(),
        pose_model=StubCornerPoseModel(point_xy=(192.0, 128.0)),
        pose_solver=StubPoseSolver(),
    )

    assert calls == []


def test_roi_tracker_reacquires_bbox_after_missing_sticker_mask():
    frame = np.zeros((60, 60, 3), dtype=np.uint8)
    detector = CountingBboxDetector()
    tracker = piece_seg_script.RoiTracker()
    args = default_args(sticker_imgsz=20, roi_tracking_padding=0.0)

    piece_seg_script.process_frame(
        frame,
        detector,
        ShiftingStickerModel(offsets=[(0, 0)]),
        args,
        roi_tracker=tracker,
    )
    assert detector.calls == 1
    assert tracker.roi_xyxy == (10, 10, 30, 30)

    piece_seg_script.process_frame(
        frame,
        detector,
        MasklessStickerModel(),
        args,
        roi_tracker=tracker,
    )
    assert detector.calls == 1
    assert tracker.roi_xyxy is None

    piece_seg_script.process_frame(
        frame,
        detector,
        ShiftingStickerModel(offsets=[(0, 0)]),
        args,
        roi_tracker=tracker,
    )
    assert detector.calls == 2
    assert tracker.roi_xyxy == (10, 10, 30, 30)


def test_roi_tracker_reacquires_bbox_after_missing_pose_keypoints():
    frame = np.zeros((60, 60, 3), dtype=np.uint8)
    detector = CountingBboxDetector()
    tracker = piece_seg_script.RoiTracker()
    args = default_args(enable_sticker_segmentation=False, roi_tracking_padding=0.0)

    piece_seg_script.process_frame(
        frame,
        detector,
        None,
        args,
        roi_tracker=tracker,
        pose_model=StubCornerPoseModel(),
        pose_solver=StubPoseSolver(),
    )
    assert detector.calls == 1
    assert tracker.roi_xyxy is not None

    piece_seg_script.process_frame(
        frame,
        detector,
        None,
        args,
        roi_tracker=tracker,
        pose_model=StubNoPoseModel(),
        pose_solver=StubPoseSolver(),
    )
    assert detector.calls == 1
    assert tracker.roi_xyxy is None

    piece_seg_script.process_frame(
        frame,
        detector,
        None,
        args,
        roi_tracker=tracker,
        pose_model=StubCornerPoseModel(),
        pose_solver=StubPoseSolver(),
    )
    assert detector.calls == 2
    assert tracker.roi_xyxy is not None


def test_roi_tracker_adds_padding_around_tracked_roi():
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    detector = CountingBboxDetector()
    tracker = piece_seg_script.RoiTracker()
    args = default_args(
        sticker_imgsz=20,
        roi_tracking_gain=1.0,
        roi_tracking_padding=0.25,
    )

    piece_seg_script.process_frame(
        frame,
        detector,
        ShiftingStickerModel(offsets=[(0, 0)]),
        args,
        roi_tracker=tracker,
    )

    assert tracker.roi_xyxy == (5, 5, 35, 35)
    first_side = tracker.roi_xyxy[2] - tracker.roi_xyxy[0]

    piece_seg_script.process_frame(
        frame,
        detector,
        ShiftingStickerModel(offsets=[(0, 0)]),
        args,
        roi_tracker=tracker,
    )
    second_side = tracker.roi_xyxy[2] - tracker.roi_xyxy[0]

    assert second_side == first_side


def test_roi_tracker_padded_roi_can_shrink_after_larger_observed_bounds():
    frame = np.zeros((90, 90, 3), dtype=np.uint8)
    detector = CountingBboxDetector()
    tracker = piece_seg_script.RoiTracker()
    sticker_model = VariableSizeStickerModel(sizes=[20, 20, 10])
    args = default_args(
        sticker_imgsz=20,
        roi_tracking_gain=1.0,
        roi_tracking_padding=0.25,
    )

    piece_seg_script.process_frame(frame, detector, sticker_model, args, roi_tracker=tracker)
    first_side = tracker.roi_xyxy[2] - tracker.roi_xyxy[0]
    piece_seg_script.process_frame(frame, detector, sticker_model, args, roi_tracker=tracker)
    grown_side = tracker.roi_xyxy[2] - tracker.roi_xyxy[0]
    piece_seg_script.process_frame(frame, detector, sticker_model, args, roi_tracker=tracker)
    shrunk_side = tracker.roi_xyxy[2] - tracker.roi_xyxy[0]

    assert first_side == 30
    assert grown_side > first_side
    assert shrunk_side < grown_side


def test_square_roi_from_bbox_keeps_initial_tracking_roi_square_at_frame_edge():
    roi = piece_seg_script.square_roi_from_bbox((0, 10, 12, 24), 40, 30, 0.0)

    assert roi == (0, 10, 14, 24)


def test_resize_to_square_size_scales_viewport_to_model_input_size():
    image = np.zeros((40, 40, 3), dtype=np.uint8)

    resized = piece_seg_script.resize_to_square_size(image, 20)

    assert resized.shape == (20, 20, 3)


def test_pose_keypoints_draw_even_when_pnp_pose_estimation_is_disabled(monkeypatch):
    monkeypatch.setattr(piece_seg_script, "ENABLE_POSE_ESTIMATION", False)
    monkeypatch.setattr(piece_seg_script, "DRAW_POSE_KEYPOINTS", True)
    frame = np.zeros((24, 24, 3), dtype=np.uint8)
    roi = frame[2:22, 2:22]

    point_count, pnp_result, tracking_bounds = piece_seg_script.draw_pose_estimate(
        frame,
        roi,
        (2, 2, 22, 22),
        StubCornerPoseModel(point_xy=(128.0, 128.0)),
        RaisingPoseSolver(),
        default_args(enable_sticker_segmentation=False),
    )

    assert point_count == 1
    assert pnp_result is None
    assert tracking_bounds is not None
    assert np.any(frame != 0)


def test_pose_estimate_resets_temporal_tracking_when_pose_model_misses():
    frame = np.zeros((24, 24, 3), dtype=np.uint8)
    roi = frame[2:22, 2:22]
    pose_solver = ResetRecordingPoseSolver()

    point_count, pnp_result, tracking_bounds = piece_seg_script.draw_pose_estimate(
        frame,
        roi,
        (2, 2, 22, 22),
        StubNoPoseModel(),
        pose_solver,
        default_args(enable_sticker_segmentation=False),
    )

    assert point_count == 0
    assert pnp_result is None
    assert tracking_bounds is None
    assert pose_solver.reset_count == 1


def test_pose_pnp_options_are_named_constants_not_parser_arguments():
    args = piece_seg_script.build_parser().parse_args([])

    assert not hasattr(args, "pose_model")
    assert not hasattr(args, "pose_imgsz")
    assert not hasattr(args, "pose_keypoint_conf")
    assert not hasattr(args, "pnp_bootstrap_frames")
    assert not hasattr(args, "pnp_focal_steps")
    assert not hasattr(args, "pnp_corner_limit")
    assert not hasattr(args, "pnp_inlier_threshold")
    assert not hasattr(args, "draw_pose_keypoints")
    assert not hasattr(args, "draw_pnp_cube")
    assert not hasattr(args, "draw_pose_graph_coords")

    assert piece_seg_script.POSE_IMAGE_SIZE == 256
    assert piece_seg_script.POSE_OBJECT_CONFIDENCE == 0.5
    assert piece_seg_script.POSE_KEYPOINT_CONFIDENCE == 0.8
    assert piece_seg_script.ENABLE_POSE_ESTIMATION is True
    assert piece_seg_script.ENABLE_POSE_GRAPH_CORRESPONDENCE is True
    assert piece_seg_script.DRAW_PNP_CUBE is True
    assert piece_seg_script.DRAW_POSE_GRAPH_COORDS is False
    assert piece_seg_script.PNP_FOCAL_SCALE_MIN == 0.7
    assert piece_seg_script.PNP_FOCAL_SCALE_MAX == 1.8
    assert piece_seg_script.PNP_CAMERA_BOOTSTRAP_FRAME_COUNT == 12
    assert piece_seg_script.PNP_MAX_REPROJECTION_ERROR_PX == 12.0
    assert piece_seg_script.PNP_INLIER_THRESHOLD_PX == 10.0
    assert piece_seg_script.PNP_PRUNE_REPROJECTION_ERROR_PX == 18.0
    assert not hasattr(piece_seg_script, "PNP_MAX_FRAME_TRANSLATION_MM")
    assert piece_seg_script.ADJACENCY_REFERENCE_ROI_SIZE_PX == 600.0
    assert piece_seg_script.ADJACENCY_MIN_EDGE_LENGTH_PX == 32.0
    assert piece_seg_script.ADJACENCY_MAX_AVERAGE_LENGTH_RATIO == 1.8
    assert not hasattr(piece_seg_script, "PNP_GRAPH_CORRESPONDENCE_POINT_COUNT")


def test_pose_graph_candidate_edges_follow_class_neighbor_limits():
    points = [
        piece_seg_script.PoseGraphPoint(0, "corner", (0.0, 0.0), 0.9),
        piece_seg_script.PoseGraphPoint(1, "corner", (1.0, 1.0), 0.9),
        piece_seg_script.PoseGraphPoint(2, "corner-edge", (3.0, 0.0), 0.9),
        piece_seg_script.PoseGraphPoint(3, "corner-edge", (6.0, 0.0), 0.9),
        piece_seg_script.PoseGraphPoint(4, "corner-edge", (9.0, 0.0), 0.9),
        piece_seg_script.PoseGraphPoint(5, "corner-edge", (30.0, 0.0), 0.9),
        piece_seg_script.PoseGraphPoint(6, "corner-edge-center", (4.0, 2.0), 0.9),
        piece_seg_script.PoseGraphPoint(7, "corner-edge-center", (5.0, 2.0), 0.9),
        piece_seg_script.PoseGraphPoint(8, "corner-edge-center", (40.0, 2.0), 0.9),
    ]

    edges = set(piece_seg_script.pose_graph_candidate_edges(points))

    assert (0, 1) not in edges
    assert {(0, 2), (0, 3), (0, 4)} <= edges
    assert (0, 5) not in edges
    assert (2, 3) in edges
    assert {(2, 6), (2, 7)} <= edges
    assert (2, 8) not in edges


def test_shared_lattice_boundary_faces_finds_the_square_face():
    assert piece_seg_script.shared_lattice_boundary_faces(
        ((0, 0, 0), (3, 0, 0), (3, 3, 0))
    ) == frozenset({(2, 0)})


def fake_pose_pnp_result(
    *,
    image_points=None,
    object_points=None,
    camera_matrix=None,
    display_points=None,
    display_edge_indexes=tuple(),
    matched_error=0.0,
):
    image_points = (
        np.zeros((3, 2), dtype=np.float32)
        if image_points is None
        else np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    )
    object_points = (
        np.zeros((len(image_points), 3), dtype=np.float32)
        if object_points is None
        else np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
    )
    display_points = (
        np.zeros((8, 3), dtype=np.float32)
        if display_points is None
        else np.asarray(display_points, dtype=np.float32).reshape(-1, 3)
    )
    camera_matrix = (
        piece_seg_script.default_camera_matrix((100, 100))
        if camera_matrix is None
        else np.asarray(camera_matrix, dtype=np.float64)
    )
    return piece_seg_script.PosePnPResult(
        error=matched_error,
        inlier_count=len(image_points),
        matched_error=matched_error,
        camera_matrix=camera_matrix,
        dist_coeffs=np.zeros((4, 1), dtype=np.float64),
        rvec=np.zeros((3, 1), dtype=np.float64),
        tvec=np.asarray([[0.0], [0.0], [180.0]], dtype=np.float64),
        image_points=image_points,
        object_points=object_points,
        display_points=display_points,
        projected_object_points=image_points.copy(),
        projected_display_points=np.zeros((len(display_points), 2), dtype=np.float32),
        point_names=tuple(),
        display_edge_indexes=display_edge_indexes,
        invisible_point_count=0,
    )


def test_observed_pose_score_uses_nearest_projected_same_class_point_only(monkeypatch):
    result = SimpleNamespace(rvec=np.zeros((3, 1), dtype=np.float64))
    graph = piece_seg_script.PoseGraph(
        points=(
            piece_seg_script.PoseGraphPoint(0, "corner", (10.0, 10.0), 0.9),
            piece_seg_script.PoseGraphPoint(1, "corner-edge", (50.0, 10.0), 0.9),
            piece_seg_script.PoseGraphPoint(2, "corner-edge-center", (50.0, 50.0), 0.9),
            piece_seg_script.PoseGraphPoint(3, "center", (200.0, 200.0), 0.9),
        ),
        edges=tuple(),
    )

    def fake_projected_lattice_points_by_class(_result):
        return {
            "corner": (
                np.asarray([[10.0, 10.0], [500.0, 500.0]], dtype=np.float32),
                np.zeros((2, 3), dtype=np.float32),
            ),
            "corner-edge": (
                np.asarray([[500.0, 10.0], [51.0, 10.0]], dtype=np.float32),
                np.zeros((2, 3), dtype=np.float32),
            ),
            "corner-edge-center": (
                np.asarray([[50.0, 52.0], [700.0, 700.0]], dtype=np.float32),
                np.zeros((2, 3), dtype=np.float32),
            ),
        }

    monkeypatch.setattr(
        piece_seg_script,
        "projected_lattice_points_by_class",
        fake_projected_lattice_points_by_class,
    )
    monkeypatch.setattr(piece_seg_script, "object_point_has_visible_surface", lambda *_args: True)

    error, inlier_count, point_count, invisible_count = (
        piece_seg_script.observed_pose_model_nearest_same_class_score(
            result,
            graph,
            inlier_threshold_px=3.0,
        )
    )

    assert point_count == 3
    assert inlier_count == 3
    assert invisible_count == 0
    assert error == 1.0


def test_projected_lattice_points_by_class_applies_result_lattice_transform(monkeypatch):
    transform = ((2, 1, 0), (-1, 1, 1))
    result = fake_pose_pnp_result()
    result.lattice_transform = transform
    projected_object_points = []

    def fake_project_points(object_points, *_args, **_kwargs):
        object_points = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
        projected_object_points.append(object_points.copy())
        return object_points[:, :2].reshape(-1, 1, 2), None

    monkeypatch.setattr(piece_seg_script.cv2, "projectPoints", fake_project_points)

    projected_by_class = piece_seg_script.projected_lattice_points_by_class(result)

    assert set(projected_by_class) == {"corner", "corner-edge", "corner-edge-center"}
    expected_corner_points = piece_seg_script.cube_lattice_points_for_coords(
        piece_seg_script.CUBE_LATTICE_COORDS_BY_CLASS["corner"],
        lattice_transform=transform,
    )
    np.testing.assert_allclose(projected_by_class["corner"][1], expected_corner_points)
    np.testing.assert_allclose(projected_object_points[0], expected_corner_points)


def test_observed_pose_score_prefers_visible_same_class_match_over_backside_match(monkeypatch):
    result = SimpleNamespace(rvec=np.zeros((3, 1), dtype=np.float64))
    graph = piece_seg_script.PoseGraph(
        points=(
            piece_seg_script.PoseGraphPoint(0, "corner", (10.0, 10.0), 0.9),
            piece_seg_script.PoseGraphPoint(1, "corner", (20.0, 10.0), 0.9),
            piece_seg_script.PoseGraphPoint(2, "corner", (20.0, 20.0), 0.9),
        ),
        edges=tuple(),
    )

    monkeypatch.setattr(
        piece_seg_script,
        "projected_lattice_points_by_class",
        lambda _result: {
            "corner": (
                np.asarray([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0]], dtype=np.float32),
                np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
            )
        },
    )
    def fake_visible(object_point, _rvec):
        point = np.asarray(object_point, dtype=np.float32).reshape(3)
        return not np.isclose(float(point[0]), 1.0)

    monkeypatch.setattr(piece_seg_script, "object_point_has_visible_surface", fake_visible)

    error, inlier_count, point_count, invisible_count = (
        piece_seg_script.observed_pose_model_nearest_same_class_score(
            result,
            graph,
            inlier_threshold_px=3.0,
        )
    )

    assert error == pytest.approx(10.0 / 3.0)
    assert inlier_count == 2
    assert point_count == 3
    assert invisible_count == 0


def test_observed_pose_score_matches_same_class_model_points_once(monkeypatch):
    result = SimpleNamespace(rvec=np.zeros((3, 1), dtype=np.float64))
    graph = piece_seg_script.PoseGraph(
        points=(
            piece_seg_script.PoseGraphPoint(0, "corner", (10.0, 10.0), 0.9),
            piece_seg_script.PoseGraphPoint(1, "corner", (10.2, 10.0), 0.9),
            piece_seg_script.PoseGraphPoint(2, "corner", (100.0, 100.0), 0.9),
        ),
        edges=tuple(),
    )

    monkeypatch.setattr(
        piece_seg_script,
        "projected_lattice_points_by_class",
        lambda _result: {
            "corner": (
                np.asarray([[10.0, 10.0], [100.0, 100.0], [101.0, 100.0]], dtype=np.float32),
                np.zeros((3, 3), dtype=np.float32),
            )
        },
    )
    monkeypatch.setattr(piece_seg_script, "object_point_has_visible_surface", lambda *_args: True)

    error, inlier_count, point_count, invisible_count = (
        piece_seg_script.observed_pose_model_nearest_same_class_score(
            result,
            graph,
            inlier_threshold_px=3.0,
        )
    )

    assert error == pytest.approx(np.linalg.norm(np.asarray([90.8, 90.0])) / 3.0)
    assert inlier_count == 2
    assert point_count == 3
    assert invisible_count == 0


def test_pose_graph_correspondence_flow_is_disabled_when_flag_is_off(monkeypatch):
    def fail_correspondence(_graph):
        raise AssertionError("pose graph traversal should stay disabled")

    monkeypatch.setattr(piece_seg_script, "ENABLE_POSE_ESTIMATION", True)
    monkeypatch.setattr(piece_seg_script, "ENABLE_POSE_GRAPH_CORRESPONDENCE", False)
    monkeypatch.setattr(piece_seg_script, "vector_guided_pose_graph_correspondence", fail_correspondence)
    frame = np.zeros((24, 24, 3), dtype=np.uint8)
    roi = frame[2:22, 2:22]

    point_count, pnp_result, tracking_bounds = piece_seg_script.draw_pose_estimate(
        frame,
        roi,
        (2, 2, 22, 22),
        StubCornerPoseModel(point_xy=(128.0, 128.0)),
        RaisingPoseSolver(),
        default_args(enable_sticker_segmentation=False),
    )

    assert point_count == 1
    assert pnp_result is None
    assert tracking_bounds is not None


def test_edge_evidence_requires_mostly_full_blurry_candidate_segment():
    edge_map = np.zeros((24, 100), dtype=np.uint8)
    piece_seg_script.cv2.line(edge_map, (4, 11), (84, 11), 255, 1)
    points = [
        piece_seg_script.PoseGraphPoint(0, "corner", (0.0, 12.0), 0.9),
        piece_seg_script.PoseGraphPoint(1, "corner-edge", (90.0, 12.0), 0.9),
    ]

    evidence = piece_seg_script.pose_graph_edge_evidence(
        points,
        (0, 0, 100, 24),
        edge_map,
        SimpleNamespace(
            adjacency_edge_dilation=3,
            adjacency_reference_roi_size=100.0,
            adjacency_min_coverage=0.95,
        ),
    )

    assert evidence[0].accepted
    assert evidence[0].coverage > 0.95
    assert evidence[0].max_uncovered_gap < (
        piece_seg_script.ADJACENCY_MAX_UNCOVERED_GAP_RATIO * 90.0
    )


def test_edge_evidence_ignores_candidate_edges_shorter_than_min_length():
    edge_map = np.full((24, 64), 255, dtype=np.uint8)
    points = [
        piece_seg_script.PoseGraphPoint(0, "corner", (0.0, 12.0), 0.9),
        piece_seg_script.PoseGraphPoint(1, "corner-edge", (31.0, 12.0), 0.9),
        piece_seg_script.PoseGraphPoint(2, "corner-edge", (32.0, 12.0), 0.9),
    ]

    evidence = piece_seg_script.pose_graph_edge_evidence(
        points,
        (0, 0, 64, 24),
        edge_map,
        SimpleNamespace(adjacency_edge_dilation=0, adjacency_reference_roi_size=64.0),
    )

    assert {(edge.start_index, edge.end_index) for edge in evidence} == {(0, 2)}
    assert evidence[0].accepted


def test_edge_evidence_rejects_candidate_that_skips_observed_keypoint():
    edge_map = np.zeros((24, 120), dtype=np.uint8)
    piece_seg_script.cv2.line(edge_map, (0, 12), (100, 12), 255, 1)
    points = [
        piece_seg_script.PoseGraphPoint(0, "corner", (0.0, 12.0), 0.9),
        piece_seg_script.PoseGraphPoint(1, "corner-edge", (50.0, 13.0), 0.9),
        piece_seg_script.PoseGraphPoint(2, "corner-edge", (100.0, 12.0), 0.9),
    ]

    evidence = piece_seg_script.pose_graph_edge_evidence(
        points,
        (0, 0, 120, 24),
        edge_map,
        SimpleNamespace(
            adjacency_edge_dilation=1,
            adjacency_keypoint_intersection_distance=4.0,
            adjacency_reference_roi_size=120.0,
        ),
    )

    skipped_edge = next(edge for edge in evidence if edge.start_index == 0 and edge.end_index == 2)

    assert skipped_edge.coverage == 0.0
    assert not skipped_edge.accepted


def test_draw_pose_graph_adjacencies_highlights_all_candidate_edge_strip_width(monkeypatch):
    line_calls = []
    graph = piece_seg_script.PoseGraph(
        points=(
            piece_seg_script.PoseGraphPoint(0, "corner", (10.0, 12.0), 0.9),
            piece_seg_script.PoseGraphPoint(1, "corner-edge", (60.0, 12.0), 0.9),
            piece_seg_script.PoseGraphPoint(2, "corner-edge", (60.0, 32.0), 0.9),
        ),
        edges=(
            piece_seg_script.PoseGraphEdge(0, 1, 1.0, 0.0, True),
            piece_seg_script.PoseGraphEdge(0, 2, 0.0, 20.0, False),
        ),
    )

    monkeypatch.setattr(
        piece_seg_script.cv2,
        "line",
        lambda *args, **kwargs: line_calls.append((args, kwargs)),
    )

    accepted_count = piece_seg_script.draw_pose_graph_adjacencies(
        np.zeros((48, 80, 3), dtype=np.uint8),
        graph,
        SimpleNamespace(adjacency_edge_dilation=10, debug_overlay=False),
    )

    assert accepted_count == 1
    assert line_calls[0][0][3] == piece_seg_script.POSE_GRAPH_EDGE_STRIP_COLOR
    assert line_calls[0][0][4] == 21
    assert line_calls[1][0][3] == piece_seg_script.POSE_GRAPH_EDGE_STRIP_COLOR
    assert line_calls[1][0][4] == 21
    assert line_calls[2][0][4] == 5
    assert line_calls[3][0][4] == 2


def test_adjacency_pixel_parameters_scale_from_reference_roi_size():
    args = SimpleNamespace(adjacency_edge_dilation=10)

    assert piece_seg_script.adjacency_edge_strip_width_px(args, roi_side=180.0) == 7
    assert piece_seg_script.adjacency_edge_strip_width_px(args, roi_side=600.0) == 21
    assert piece_seg_script.adjacency_edge_strip_width_px(args, roi_side=1200.0) == 41
    assert (
        piece_seg_script.scaled_adjacency_px(
            piece_seg_script.ADJACENCY_MIN_EDGE_LENGTH_PX,
            300.0,
            SimpleNamespace(),
        )
        == 16.0
    )


def test_pose_graph_prunes_edges_noticeably_longer_than_average():
    points = (
        piece_seg_script.PoseGraphPoint(0, "corner", (0.0, 0.0), 0.9),
        piece_seg_script.PoseGraphPoint(1, "corner-edge", (10.0, 0.0), 0.9),
        piece_seg_script.PoseGraphPoint(2, "corner-edge", (20.0, 0.0), 0.9),
        piece_seg_script.PoseGraphPoint(3, "corner-edge", (30.0, 0.0), 0.9),
        piece_seg_script.PoseGraphPoint(4, "corner-edge", (100.0, 0.0), 0.9),
    )
    graph = piece_seg_script.PoseGraph(
        points=points,
        edges=(
            piece_seg_script.PoseGraphEdge(0, 1, 1.0, 0.0, True),
            piece_seg_script.PoseGraphEdge(1, 2, 1.0, 0.0, True),
            piece_seg_script.PoseGraphEdge(2, 3, 1.0, 0.0, True),
            piece_seg_script.PoseGraphEdge(0, 4, 1.0, 0.0, True),
        ),
    )

    pruned = piece_seg_script.prune_long_pose_graph_edges(graph, SimpleNamespace())

    assert [edge.accepted for edge in pruned.edges] == [True, True, True, False]


def test_vector_guided_pose_graph_correspondence_assigns_corner_branches_by_image_direction():
    points = [
        piece_seg_script.PoseGraphPoint(0, "corner", (10.0, 10.0), 0.90),
        piece_seg_script.PoseGraphPoint(1, "corner-edge", (30.0, 10.0), 0.80),
        piece_seg_script.PoseGraphPoint(2, "corner-edge", (-10.0, 10.0), 0.70),
        piece_seg_script.PoseGraphPoint(3, "corner-edge", (10.0, 30.0), 0.60),
        piece_seg_script.PoseGraphPoint(4, "corner-edge-center", (30.0, 30.0), 0.50),
    ]
    graph = piece_seg_script.PoseGraph(
        points=tuple(points),
        edges=(
            piece_seg_script.PoseGraphEdge(0, 1, 1.0, 0.0, True),
            piece_seg_script.PoseGraphEdge(0, 2, 1.0, 0.0, True),
            piece_seg_script.PoseGraphEdge(0, 3, 1.0, 0.0, True),
            piece_seg_script.PoseGraphEdge(1, 4, 1.0, 0.0, True),
            piece_seg_script.PoseGraphEdge(3, 4, 1.0, 0.0, True),
        ),
    )

    correspondence = piece_seg_script.vector_guided_pose_graph_correspondence(graph)

    assert correspondence is not None
    coord_by_graph_index = dict(zip(correspondence.graph_indexes, correspondence.lattice_coords))
    assert coord_by_graph_index[0] == (0, 0, 0)
    assert coord_by_graph_index[1] == (1, 0, 0)
    assert coord_by_graph_index[2] == (0, 1, 0)
    assert coord_by_graph_index[3] == (0, 0, 1)
    assert coord_by_graph_index[4] == (1, 0, 1)


def test_vector_guided_pose_graph_correspondence_uses_first_neighbor_as_local_frame():
    points = [
        piece_seg_script.PoseGraphPoint(0, "corner", (10.0, 10.0), 0.95),
        piece_seg_script.PoseGraphPoint(1, "corner-edge", (30.0, 10.0), 0.60),
        piece_seg_script.PoseGraphPoint(2, "corner-edge", (-10.0, 10.0), 0.90),
        piece_seg_script.PoseGraphPoint(3, "corner-edge", (10.0, 30.0), 0.80),
        piece_seg_script.PoseGraphPoint(4, "corner-edge-center", (30.0, 30.0), 0.50),
    ]
    graph = piece_seg_script.PoseGraph(
        points=tuple(points),
        edges=(
            piece_seg_script.PoseGraphEdge(0, 1, 1.0, 0.0, True),
            piece_seg_script.PoseGraphEdge(0, 2, 1.0, 0.0, True),
            piece_seg_script.PoseGraphEdge(0, 3, 1.0, 0.0, True),
            piece_seg_script.PoseGraphEdge(1, 4, 1.0, 0.0, True),
            piece_seg_script.PoseGraphEdge(3, 4, 1.0, 0.0, True),
        ),
    )

    correspondence = piece_seg_script.vector_guided_pose_graph_correspondence(graph)

    assert correspondence is not None
    coord_by_graph_index = dict(zip(correspondence.graph_indexes, correspondence.lattice_coords))
    assert coord_by_graph_index[2] == (1, 0, 0)
    assert coord_by_graph_index[3] == (0, 1, 0)
    assert coord_by_graph_index[1] == (0, 0, 1)
    assert coord_by_graph_index[4] == (0, 1, 1)


def test_vector_guided_pose_graph_correspondence_assigns_corner_edge_start():
    points = [
        piece_seg_script.PoseGraphPoint(0, "corner-edge", (10.0, 10.0), 0.90),
        piece_seg_script.PoseGraphPoint(1, "corner", (30.0, 10.0), 0.80),
        piece_seg_script.PoseGraphPoint(2, "corner-edge", (-10.0, 10.0), 0.70),
        piece_seg_script.PoseGraphPoint(3, "corner-edge-center", (10.0, 30.0), 0.60),
    ]
    graph = piece_seg_script.PoseGraph(
        points=tuple(points),
        edges=(
            piece_seg_script.PoseGraphEdge(0, 1, 1.0, 0.0, True),
            piece_seg_script.PoseGraphEdge(0, 2, 1.0, 0.0, True),
            piece_seg_script.PoseGraphEdge(0, 3, 1.0, 0.0, True),
        ),
    )

    correspondence = piece_seg_script.vector_guided_pose_graph_correspondence(graph)

    assert correspondence is not None
    coord_by_graph_index = dict(zip(correspondence.graph_indexes, correspondence.lattice_coords))
    assert coord_by_graph_index[0] == (1, 0, 0)
    assert coord_by_graph_index[1] == (0, 0, 0)
    assert coord_by_graph_index[2] == (2, 0, 0)
    assert coord_by_graph_index[3] == (1, 1, 0)


def test_vector_guided_pose_graph_correspondence_uses_previous_edge_as_local_frame():
    points = [
        piece_seg_script.PoseGraphPoint(0, "corner", (0.0, 0.0), 0.90),
        piece_seg_script.PoseGraphPoint(1, "corner-edge", (20.0, 0.0), 0.80),
        piece_seg_script.PoseGraphPoint(2, "corner-edge-center", (20.0, 20.0), 0.70),
        piece_seg_script.PoseGraphPoint(3, "corner-edge-center", (40.0, 20.0), 0.60),
    ]
    graph = piece_seg_script.PoseGraph(
        points=tuple(points),
        edges=(
            piece_seg_script.PoseGraphEdge(0, 1, 1.0, 0.0, True),
            piece_seg_script.PoseGraphEdge(1, 2, 1.0, 0.0, True),
            piece_seg_script.PoseGraphEdge(2, 3, 1.0, 0.0, True),
        ),
    )

    correspondence = piece_seg_script.vector_guided_pose_graph_correspondence(graph)

    assert correspondence is not None
    coord_by_graph_index = dict(zip(correspondence.graph_indexes, correspondence.lattice_coords))
    assert coord_by_graph_index[0] == (0, 0, 0)
    assert coord_by_graph_index[1] == (1, 0, 0)
    assert coord_by_graph_index[2] == (1, 0, 1)
    assert coord_by_graph_index[3] == (2, 0, 1)


def test_vector_guided_pose_graph_correspondence_traverses_entire_component():
    points = [
        piece_seg_script.PoseGraphPoint(0, "corner", (0.0, 0.0), 0.90),
        piece_seg_script.PoseGraphPoint(1, "corner-edge", (20.0, 0.0), 0.80),
        piece_seg_script.PoseGraphPoint(2, "corner-edge", (40.0, 0.0), 0.70),
        piece_seg_script.PoseGraphPoint(3, "corner", (60.0, 0.0), 0.60),
        piece_seg_script.PoseGraphPoint(4, "corner-edge", (60.0, 20.0), 0.50),
        piece_seg_script.PoseGraphPoint(5, "corner-edge", (60.0, 40.0), 0.40),
        piece_seg_script.PoseGraphPoint(6, "corner", (60.0, 60.0), 0.30),
    ]
    graph = piece_seg_script.PoseGraph(
        points=tuple(points),
        edges=tuple(
            piece_seg_script.PoseGraphEdge(index, index + 1, 1.0, 0.0, True)
            for index in range(len(points) - 1)
        ),
    )

    correspondence = piece_seg_script.vector_guided_pose_graph_correspondence(graph)

    assert correspondence is not None
    assert correspondence.graph_indexes == tuple(range(len(points)))
    assert len(correspondence.image_points) == len(points)
    assert len(correspondence.object_points) == len(points)


def test_vector_guided_pose_graph_correspondence_revisits_branch_nodes_from_new_local_frames():
    points = [
        piece_seg_script.PoseGraphPoint(0, "corner", (0.0, 0.0), 0.90),
        piece_seg_script.PoseGraphPoint(1, "corner-edge", (20.0, 0.0), 0.80),
        piece_seg_script.PoseGraphPoint(2, "corner-edge", (0.0, 20.0), 0.70),
        piece_seg_script.PoseGraphPoint(3, "corner-edge-center", (20.0, 20.0), 0.60),
        piece_seg_script.PoseGraphPoint(4, "corner-edge-center", (40.0, 20.0), 0.50),
    ]
    graph = piece_seg_script.PoseGraph(
        points=tuple(points),
        edges=(
            piece_seg_script.PoseGraphEdge(0, 1, 1.0, 0.0, True),
            piece_seg_script.PoseGraphEdge(0, 2, 1.0, 0.0, True),
            piece_seg_script.PoseGraphEdge(1, 3, 1.0, 0.0, True),
            piece_seg_script.PoseGraphEdge(1, 4, 1.0, 0.0, True),
            piece_seg_script.PoseGraphEdge(2, 3, 1.0, 0.0, True),
            piece_seg_script.PoseGraphEdge(3, 4, 1.0, 0.0, True),
        ),
    )

    correspondence = piece_seg_script.vector_guided_pose_graph_correspondence(graph)

    assert correspondence is not None
    assert set(correspondence.graph_indexes) == set(range(len(points)))


def test_vector_guided_pose_graph_correspondence_uses_assigned_neighbor_constraints():
    points = [
        piece_seg_script.PoseGraphPoint(0, "corner", (0.0, 0.0), 0.90),
        piece_seg_script.PoseGraphPoint(1, "corner-edge", (20.0, 0.0), 0.80),
        piece_seg_script.PoseGraphPoint(2, "corner-edge", (40.0, 0.0), 0.70),
        piece_seg_script.PoseGraphPoint(3, "corner-edge-center", (20.0, -20.0), 0.60),
        piece_seg_script.PoseGraphPoint(4, "corner-edge-center", (40.0, 20.0), 0.50),
    ]
    graph = piece_seg_script.PoseGraph(
        points=tuple(points),
        edges=(
            piece_seg_script.PoseGraphEdge(0, 1, 1.0, 0.0, True),
            piece_seg_script.PoseGraphEdge(1, 2, 1.0, 0.0, True),
            piece_seg_script.PoseGraphEdge(1, 3, 1.0, 0.0, True),
            piece_seg_script.PoseGraphEdge(2, 4, 1.0, 0.0, True),
            piece_seg_script.PoseGraphEdge(3, 4, 1.0, 0.0, True),
        ),
    )

    correspondence = piece_seg_script.vector_guided_pose_graph_correspondence(graph)

    assert correspondence is not None
    coord_by_graph_index = dict(zip(correspondence.graph_indexes, correspondence.lattice_coords))
    assert coord_by_graph_index[2] == (2, 0, 0)
    assert coord_by_graph_index[3] == (1, 1, 0)
    assert coord_by_graph_index[4] == (2, 1, 0)


def test_vector_guided_pose_graph_correspondence_prefers_path_face_continuity():
    points = [
        piece_seg_script.PoseGraphPoint(0, "corner", (0.0, 60.0), 0.95),
        piece_seg_script.PoseGraphPoint(1, "corner-edge", (20.0, 60.0), 0.90),
        piece_seg_script.PoseGraphPoint(2, "corner-edge-center", (20.0, 40.0), 0.85),
        piece_seg_script.PoseGraphPoint(3, "corner-edge-center", (20.0, 20.0), 0.80),
        piece_seg_script.PoseGraphPoint(4, "corner-edge", (20.0, 0.0), 0.75),
        piece_seg_script.PoseGraphPoint(5, "corner-edge", (40.0, 0.0), 0.70),
        piece_seg_script.PoseGraphPoint(6, "corner-edge-center", (40.0, 20.0), 0.65),
    ]
    graph = piece_seg_script.PoseGraph(
        points=tuple(points),
        edges=(
            piece_seg_script.PoseGraphEdge(0, 1, 1.0, 0.0, True),
            piece_seg_script.PoseGraphEdge(1, 2, 1.0, 0.0, True),
            piece_seg_script.PoseGraphEdge(2, 3, 1.0, 0.0, True),
            piece_seg_script.PoseGraphEdge(3, 4, 1.0, 0.0, True),
            piece_seg_script.PoseGraphEdge(4, 5, 1.0, 0.0, True),
            piece_seg_script.PoseGraphEdge(5, 6, 1.0, 0.0, True),
        ),
    )

    correspondence = piece_seg_script.vector_guided_pose_graph_correspondence(graph)

    assert correspondence is not None
    coord_by_graph_index = dict(zip(correspondence.graph_indexes, correspondence.lattice_coords))
    assert coord_by_graph_index[5] == (2, 3, 0)
    assert coord_by_graph_index[6] == (2, 2, 0)


def test_vector_guided_pose_graph_correspondence_keeps_side_context_at_corner_edge_center():
    points = [
        piece_seg_script.PoseGraphPoint(0, "corner", (0.0, 0.0), 0.95),
        piece_seg_script.PoseGraphPoint(1, "corner-edge", (20.0, 0.0), 0.90),
        piece_seg_script.PoseGraphPoint(2, "corner-edge", (40.0, 0.0), 0.85),
        piece_seg_script.PoseGraphPoint(3, "corner", (60.0, 0.0), 0.80),
        piece_seg_script.PoseGraphPoint(4, "corner-edge", (60.0, -20.0), 0.75),
        piece_seg_script.PoseGraphPoint(5, "corner-edge", (80.0, 0.0), 0.70),
        piece_seg_script.PoseGraphPoint(6, "corner-edge-center", (80.0, -20.0), 0.65),
    ]
    graph = piece_seg_script.PoseGraph(
        points=tuple(points),
        edges=(
            piece_seg_script.PoseGraphEdge(0, 1, 1.0, 0.0, True),
            piece_seg_script.PoseGraphEdge(1, 2, 1.0, 0.0, True),
            piece_seg_script.PoseGraphEdge(2, 3, 1.0, 0.0, True),
            piece_seg_script.PoseGraphEdge(3, 4, 1.0, 0.0, True),
            piece_seg_script.PoseGraphEdge(3, 5, 1.0, 0.0, True),
            piece_seg_script.PoseGraphEdge(4, 6, 1.0, 0.0, True),
            piece_seg_script.PoseGraphEdge(5, 6, 1.0, 0.0, True),
        ),
    )

    correspondence = piece_seg_script.vector_guided_pose_graph_correspondence(graph)

    assert correspondence is not None
    coord_by_graph_index = dict(zip(correspondence.graph_indexes, correspondence.lattice_coords))
    assert {coord_by_graph_index[4], coord_by_graph_index[5]} == {
        (3, 0, 1),
        (3, 1, 0),
    }
    assert coord_by_graph_index[6] == (3, 1, 1)


def test_graph_pnp_solver_uses_current_graph_only(monkeypatch):
    calls = []
    correspondence = piece_seg_script.PoseGraphCorrespondence(
        image_points=np.asarray(
            [[20.0, 20.0], [40.0, 20.0], [20.0, 40.0], [20.0, 60.0], [40.0, 40.0]],
            dtype=np.float32,
        ),
        object_points=piece_seg_script.cube_lattice_points_for_coords(
            ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 2, 0), (1, 1, 0)),
        ),
        graph_indexes=(0, 1, 2, 3, 4),
        lattice_coords=((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 2, 0), (1, 1, 0)),
    )

    def fake_solve_lattice_graph_pnp(*args, **kwargs):
        calls.append((args, kwargs))
        return None

    monkeypatch.setattr(
        piece_seg_script,
        "solve_lattice_graph_pnp",
        fake_solve_lattice_graph_pnp,
    )

    result = piece_seg_script.PoseGraphPnPSolver(
        focal_min=1.0,
        focal_max=1.0,
        focal_steps=1,
    ).solve_correspondences(
        correspondence,
        (100, 100),
    )

    assert result is None
    assert len(calls) == len(piece_seg_script.CUBE_LATTICE_AXIS_TRANSFORMS)
    for args, kwargs in calls:
        np.testing.assert_allclose(args[0], correspondence.image_points)
        assert len(args[1]) == len(correspondence.lattice_coords)
        assert len(kwargs["display_points"]) == len(piece_seg_script.CUBE_LATTICE_DISPLAY_POINTS_MM)
        assert kwargs["display_edge_indexes"] == piece_seg_script.CUBE_LATTICE_DISPLAY_EDGE_INDEXES


def test_graph_pnp_solver_rejects_high_error_result(monkeypatch):
    camera_matrix = piece_seg_script.default_camera_matrix((100, 100))
    correspondence = piece_seg_script.PoseGraphCorrespondence(
        image_points=np.asarray([[20.0, 20.0], [40.0, 20.0], [20.0, 40.0]], dtype=np.float32),
        object_points=np.asarray(
            [[-27.5, -27.5, -27.5], [0.0, -27.5, -27.5], [-27.5, 0.0, -27.5]],
            dtype=np.float32,
        ),
        graph_indexes=(0, 1, 2),
        lattice_coords=((0, 0, 0), (1, 0, 0), (0, 1, 0)),
    )
    high_error_result = SimpleNamespace(
        inlier_count=0,
        error=9999.0,
        matched_error=9999.0,
        camera_matrix=camera_matrix,
    )

    monkeypatch.setattr(
        piece_seg_script,
        "solve_lattice_graph_pnp",
        lambda *args, **kwargs: high_error_result,
    )

    result = piece_seg_script.PoseGraphPnPSolver(
        focal_min=1.2,
        focal_max=1.2,
        focal_steps=1,
    ).solve_correspondences(
        correspondence,
        (100, 100),
    )

    assert result is None


def test_graph_pnp_solver_rejects_observed_point_outside_inlier_threshold(monkeypatch):
    correspondence = piece_seg_script.PoseGraphCorrespondence(
        image_points=np.asarray([[20.0, 20.0], [40.0, 20.0], [20.0, 40.0]], dtype=np.float32),
        object_points=piece_seg_script.cube_lattice_points_for_coords(
            ((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        ),
        graph_indexes=(0, 1, 2),
        lattice_coords=((0, 0, 0), (1, 0, 0), (0, 1, 0)),
    )
    observed_graph = piece_seg_script.PoseGraph(
        points=(
            piece_seg_script.PoseGraphPoint(0, "corner", (20.0, 20.0), 0.9),
            piece_seg_script.PoseGraphPoint(1, "corner-edge", (40.0, 20.0), 0.9),
            piece_seg_script.PoseGraphPoint(2, "corner-edge", (20.0, 40.0), 0.9),
        ),
        edges=tuple(),
    )
    pnp_result = SimpleNamespace(
        projected_object_points=correspondence.image_points,
        matched_error=0.0,
        rvec=np.zeros((3, 1), dtype=np.float64),
        tvec=np.zeros((3, 1), dtype=np.float64),
        camera_matrix=piece_seg_script.default_camera_matrix((100, 100)),
        dist_coeffs=np.zeros((4, 1), dtype=np.float64),
        lattice_transform=piece_seg_script.CUBE_LATTICE_IDENTITY_TRANSFORM,
    )

    monkeypatch.setattr(
        piece_seg_script,
        "solve_best_lattice_graph_pnp_current",
        lambda *args, **kwargs: pnp_result,
    )
    monkeypatch.setattr(
        piece_seg_script.cv2,
        "projectPoints",
        lambda *_args, **_kwargs: (
            np.asarray([[20.0, 20.0], [40.0, 20.0], [31.0, 40.0]], dtype=np.float32).reshape(-1, 1, 2),
            None,
        ),
    )

    result = piece_seg_script.PoseGraphPnPSolver(
        focal_min=1.0,
        focal_max=1.0,
        focal_steps=1,
    ).solve_correspondences(
        correspondence,
        (100, 100),
        observed_graph=observed_graph,
    )

    assert result is None


def test_graph_pnp_solver_allows_single_frame_translation_jump(monkeypatch):
    correspondence = piece_seg_script.PoseGraphCorrespondence(
        image_points=np.asarray([[20.0, 20.0], [40.0, 20.0], [20.0, 40.0]], dtype=np.float32),
        object_points=piece_seg_script.cube_lattice_points_for_coords(
            ((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        ),
        graph_indexes=(0, 1, 2),
        lattice_coords=((0, 0, 0), (1, 0, 0), (0, 1, 0)),
    )
    results = [
        SimpleNamespace(
            projected_object_points=correspondence.image_points,
            matched_error=0.0,
            tvec=np.asarray([[0.0], [0.0], [100.0]], dtype=np.float64),
        ),
        SimpleNamespace(
            projected_object_points=correspondence.image_points,
            matched_error=0.0,
            tvec=np.asarray([[0.0], [0.0], [149.0]], dtype=np.float64),
        ),
        SimpleNamespace(
            projected_object_points=correspondence.image_points,
            matched_error=0.0,
            tvec=np.asarray([[0.0], [0.0], [201.0]], dtype=np.float64),
        ),
    ]

    monkeypatch.setattr(
        piece_seg_script,
        "solve_best_lattice_graph_pnp_current",
        lambda *args, **kwargs: results.pop(0),
    )

    solver = piece_seg_script.PoseGraphPnPSolver(
        focal_min=1.0,
        focal_max=1.0,
        focal_steps=1,
    )

    first_result = solver.solve_correspondences(correspondence, (100, 100))
    second_result = solver.solve_correspondences(correspondence, (100, 100))
    third_result = solver.solve_correspondences(correspondence, (100, 100))

    assert first_result is not None
    assert second_result is not None
    assert third_result is not None
    assert not hasattr(solver, "last_tvec")


def test_graph_pnp_solver_missing_correspondence_does_not_affect_next_pose(monkeypatch):
    correspondence = piece_seg_script.PoseGraphCorrespondence(
        image_points=np.asarray([[20.0, 20.0], [40.0, 20.0], [20.0, 40.0]], dtype=np.float32),
        object_points=piece_seg_script.cube_lattice_points_for_coords(
            ((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        ),
        graph_indexes=(0, 1, 2),
        lattice_coords=((0, 0, 0), (1, 0, 0), (0, 1, 0)),
    )
    results = [
        SimpleNamespace(
            projected_object_points=correspondence.image_points,
            matched_error=0.0,
            tvec=np.asarray([[0.0], [0.0], [100.0]], dtype=np.float64),
        ),
        SimpleNamespace(
            projected_object_points=correspondence.image_points,
            matched_error=0.0,
            tvec=np.asarray([[0.0], [0.0], [201.0]], dtype=np.float64),
        ),
    ]

    monkeypatch.setattr(
        piece_seg_script,
        "solve_best_lattice_graph_pnp_current",
        lambda *args, **kwargs: results.pop(0),
    )

    solver = piece_seg_script.PoseGraphPnPSolver(
        focal_min=1.0,
        focal_max=1.0,
        focal_steps=1,
    )

    first_result = solver.solve_correspondences(correspondence, (100, 100))
    missed_result = solver.solve_correspondences(None, (100, 100))
    second_result = solver.solve_correspondences(correspondence, (100, 100))

    assert first_result is not None
    assert missed_result is None
    assert second_result is not None
    assert not hasattr(solver, "last_tvec")


def test_graph_pnp_solver_prunes_high_error_points_once(monkeypatch):
    calls = []
    correspondence = piece_seg_script.PoseGraphCorrespondence(
        image_points=np.asarray(
            [[20.0, 20.0], [40.0, 20.0], [20.0, 40.0], [80.0, 80.0]],
            dtype=np.float32,
        ),
        object_points=piece_seg_script.cube_lattice_points_for_coords(
            ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)),
        ),
        graph_indexes=(0, 1, 2, 3),
        lattice_coords=((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)),
    )
    first_result = SimpleNamespace(
        projected_object_points=np.asarray(
            [[21.0, 20.0], [39.0, 21.0], [20.0, 41.0], [140.0, 140.0]],
            dtype=np.float32,
        ),
        matched_error=30.0,
    )
    second_result = SimpleNamespace(
        projected_object_points=np.asarray(
            [[20.0, 20.0], [40.0, 20.0], [20.0, 40.0]],
            dtype=np.float32,
        ),
        matched_error=0.0,
    )

    def fake_solve_best(image_points, lattice_coords, *args, **kwargs):
        calls.append((np.asarray(image_points).copy(), tuple(lattice_coords)))
        return first_result if len(calls) == 1 else second_result

    monkeypatch.setattr(
        piece_seg_script,
        "solve_best_lattice_graph_pnp_current",
        fake_solve_best,
    )

    result = piece_seg_script.PoseGraphPnPSolver(
        focal_min=1.0,
        focal_max=1.0,
        focal_steps=1,
    ).solve_correspondences(
        correspondence,
        (100, 100),
    )

    assert result is second_result
    assert len(calls) == 2
    assert calls[0][1] == correspondence.lattice_coords
    assert calls[1][1] == correspondence.lattice_coords[:3]
    np.testing.assert_allclose(calls[1][0], correspondence.image_points[:3])
    assert result.point_names == tuple(str(coord) for coord in correspondence.lattice_coords[:3])
    assert result.pose_label == "graph"


def test_graph_pnp_solver_rejects_using_observed_points_after_pruning(monkeypatch):
    calls = []
    correspondence = piece_seg_script.PoseGraphCorrespondence(
        image_points=np.asarray(
            [[20.0, 20.0], [40.0, 20.0], [20.0, 40.0], [80.0, 80.0]],
            dtype=np.float32,
        ),
        object_points=piece_seg_script.cube_lattice_points_for_coords(
            ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)),
        ),
        graph_indexes=(0, 1, 2, 3),
        lattice_coords=((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)),
    )
    observed_graph = piece_seg_script.PoseGraph(
        points=(
            piece_seg_script.PoseGraphPoint(0, "corner", (20.0, 20.0), 0.9),
            piece_seg_script.PoseGraphPoint(1, "corner-edge", (40.0, 20.0), 0.9),
            piece_seg_script.PoseGraphPoint(2, "corner-edge", (20.0, 40.0), 0.9),
            piece_seg_script.PoseGraphPoint(3, "corner-edge-center", (80.0, 80.0), 0.9),
        ),
        edges=tuple(),
    )
    first_result = SimpleNamespace(
        projected_object_points=np.asarray(
            [[20.0, 20.0], [40.0, 20.0], [20.0, 40.0], [150.0, 150.0]],
            dtype=np.float32,
        ),
        matched_error=30.0,
    )
    second_result = SimpleNamespace(
        projected_object_points=np.asarray(
            [[20.0, 20.0], [40.0, 20.0], [20.0, 40.0]],
            dtype=np.float32,
        ),
        matched_error=0.0,
        rvec=np.zeros((3, 1), dtype=np.float64),
        tvec=np.zeros((3, 1), dtype=np.float64),
        camera_matrix=piece_seg_script.default_camera_matrix((100, 100)),
        dist_coeffs=np.zeros((4, 1), dtype=np.float64),
        lattice_transform=piece_seg_script.CUBE_LATTICE_IDENTITY_TRANSFORM,
    )

    def fake_solve_best(image_points, lattice_coords, *args, **kwargs):
        calls.append((np.asarray(image_points).copy(), tuple(lattice_coords)))
        return first_result if len(calls) == 1 else second_result

    def fake_project_points(points, *_args, **_kwargs):
        projected = np.asarray(
            [[20.0, 20.0], [40.0, 20.0], [200.0, 200.0]],
            dtype=np.float32,
        )[: len(points)]
        return projected.reshape(-1, 1, 2), None

    monkeypatch.setattr(
        piece_seg_script,
        "solve_best_lattice_graph_pnp_current",
        fake_solve_best,
    )
    monkeypatch.setattr(piece_seg_script.cv2, "projectPoints", fake_project_points)

    result = piece_seg_script.PoseGraphPnPSolver(
        focal_min=1.0,
        focal_max=1.0,
        focal_steps=1,
    ).solve_correspondences(
        correspondence,
        (100, 100),
        observed_graph=observed_graph,
    )

    assert result is None
    assert len(calls) == 2
    assert calls[1][1] == correspondence.lattice_coords[:3]


def test_graph_pnp_solver_rejects_when_observed_graph_error_cannot_be_scored(monkeypatch):
    correspondence = piece_seg_script.PoseGraphCorrespondence(
        image_points=np.asarray([[20.0, 20.0], [40.0, 20.0], [20.0, 40.0]], dtype=np.float32),
        object_points=piece_seg_script.cube_lattice_points_for_coords(
            ((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        ),
        graph_indexes=(0, 1, 2),
        lattice_coords=((0, 0, 0), (1, 0, 0), (0, 1, 0)),
    )
    observed_graph = piece_seg_script.PoseGraph(
        points=(
            piece_seg_script.PoseGraphPoint(0, "corner", (20.0, 20.0), 0.9),
            piece_seg_script.PoseGraphPoint(1, "corner-edge", (40.0, 20.0), 0.9),
            piece_seg_script.PoseGraphPoint(2, "corner-edge", (20.0, 40.0), 0.9),
        ),
        edges=tuple(),
    )

    monkeypatch.setattr(
        piece_seg_script,
        "solve_best_lattice_graph_pnp_current",
        lambda *args, **kwargs: SimpleNamespace(
            projected_object_points=correspondence.image_points,
            matched_error=0.0,
        ),
    )

    result = piece_seg_script.PoseGraphPnPSolver(
        focal_min=1.0,
        focal_max=1.0,
        focal_steps=1,
    ).solve_correspondences(
        correspondence,
        (100, 100),
        observed_graph=observed_graph,
    )

    assert result is None


def test_graph_pnp_solver_selects_current_frame_camera_matrix_from_graph_errors(monkeypatch):
    calls = []
    correspondence = piece_seg_script.PoseGraphCorrespondence(
        image_points=np.asarray([[20.0, 20.0], [40.0, 20.0], [20.0, 40.0]], dtype=np.float32),
        object_points=np.asarray(
            [[-27.5, -27.5, -27.5], [0.0, -27.5, -27.5], [-27.5, 0.0, -27.5]],
            dtype=np.float32,
        ),
        graph_indexes=(0, 1, 2),
        lattice_coords=((0, 0, 0), (1, 0, 0), (0, 1, 0)),
    )

    def fake_solve_lattice_graph_pnp(_image_points, _object_points, camera_matrix, **_kwargs):
        calls.append(float(camera_matrix[0, 0]))
        return SimpleNamespace(
            matched_error=abs(float(camera_matrix[0, 0]) - 100.0),
            camera_matrix=camera_matrix,
        )

    monkeypatch.setattr(
        piece_seg_script,
        "solve_lattice_graph_pnp",
        fake_solve_lattice_graph_pnp,
    )

    result = piece_seg_script.PoseGraphPnPSolver(
        focal_min=0.5,
        focal_max=1.5,
        focal_steps=3,
    ).solve_correspondences(
        correspondence,
        (100, 80),
    )

    assert result is not None
    assert set(calls) == {50.0, 100.0, 150.0}
    assert result.pose_label == "graph"
    np.testing.assert_allclose(result.camera_matrix, piece_seg_script.default_camera_matrix((100, 80), focal_scale=1.0))


def test_graph_pnp_solver_reuses_camera_matrix_after_bootstrap(monkeypatch):
    calls = []
    correspondence = piece_seg_script.PoseGraphCorrespondence(
        image_points=np.asarray([[20.0, 20.0], [40.0, 20.0], [20.0, 40.0]], dtype=np.float32),
        object_points=np.asarray(
            [[-27.5, -27.5, -27.5], [0.0, -27.5, -27.5], [-27.5, 0.0, -27.5]],
            dtype=np.float32,
        ),
        graph_indexes=(0, 1, 2),
        lattice_coords=((0, 0, 0), (1, 0, 0), (0, 1, 0)),
    )

    def fake_solve_lattice_graph_pnp(_image_points, _object_points, camera_matrix, **_kwargs):
        focal = float(camera_matrix[0, 0])
        calls.append(focal)
        return SimpleNamespace(
            projected_object_points=correspondence.image_points,
            matched_error=abs(focal - 100.0),
            camera_matrix=camera_matrix,
            tvec=np.asarray([[0.0], [0.0], [100.0]], dtype=np.float64),
        )

    monkeypatch.setattr(
        piece_seg_script,
        "solve_lattice_graph_pnp",
        fake_solve_lattice_graph_pnp,
    )

    solver = piece_seg_script.PoseGraphPnPSolver(
        focal_min=0.5,
        focal_max=1.5,
        focal_steps=3,
        camera_bootstrap_frames=1,
    )

    first_result = solver.solve_correspondences(correspondence, (100, 80))
    first_call_count = len(calls)
    second_result = solver.solve_correspondences(correspondence, (100, 80))
    second_calls = calls[first_call_count:]

    assert first_result is not None
    assert second_result is not None
    assert set(calls[:first_call_count]) == {50.0, 100.0, 150.0}
    assert second_calls == [100.0] * len(piece_seg_script.CUBE_LATTICE_AXIS_TRANSFORMS)
    np.testing.assert_allclose(solver.camera_matrix, piece_seg_script.default_camera_matrix((100, 80), focal_scale=1.0))


def test_graph_pnp_camera_fit_can_use_non_identity_lattice_embedding():
    coords = (
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
        (2, 0, 0),
    )
    lattice_transform = ((2, 1, 0), (-1, 1, 1))
    camera_matrix = piece_seg_script.default_camera_matrix((640, 480), focal_scale=1.2)
    object_points = piece_seg_script.cube_lattice_points_for_coords(
        coords,
        lattice_transform=lattice_transform,
    )
    image_points, _ = piece_seg_script.cv2.projectPoints(
        object_points,
        np.asarray([[0.35], [-0.2], [0.15]], dtype=np.float64),
        np.asarray([[5.0], [-3.0], [190.0]], dtype=np.float64),
        camera_matrix,
        np.zeros((4, 1), dtype=np.float64),
    )

    result = piece_seg_script.solve_best_lattice_graph_pnp_current(
        image_points.reshape(-1, 2).astype(np.float32),
        coords,
        (640, 480),
        focal_min=1.2,
        focal_max=1.2,
        focal_steps=1,
    )

    assert result is not None
    assert result.lattice_transform != piece_seg_script.CUBE_LATTICE_IDENTITY_TRANSFORM
    assert result.matched_error < 1e-3
    np.testing.assert_allclose(
        result.display_points,
        piece_seg_script.cube_lattice_display_points_for_transform(result.lattice_transform),
    )


def test_solve_lattice_graph_pnp_scores_paired_graph_points():
    camera_matrix = piece_seg_script.default_camera_matrix((120, 120))
    object_points = np.asarray(
        [
            piece_seg_script.CUBE_LATTICE_POINT_BY_COORD[(0, 0, 0)],
            piece_seg_script.CUBE_LATTICE_POINT_BY_COORD[(1, 0, 0)],
            piece_seg_script.CUBE_LATTICE_POINT_BY_COORD[(0, 1, 0)],
            piece_seg_script.CUBE_LATTICE_POINT_BY_COORD[(0, 0, 1)],
            piece_seg_script.CUBE_LATTICE_POINT_BY_COORD[(1, 1, 0)],
            piece_seg_script.CUBE_LATTICE_POINT_BY_COORD[(1, 0, 1)],
        ],
        dtype=np.float32,
    )
    rvec = np.asarray([[0.3], [-0.2], [0.1]], dtype=np.float64)
    tvec = np.asarray([[0.0], [0.0], [180.0]], dtype=np.float64)
    image_points, _ = piece_seg_script.cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        np.zeros((4, 1), dtype=np.float64),
    )

    result = piece_seg_script.solve_lattice_graph_pnp(
        image_points.reshape(-1, 2),
        object_points,
        camera_matrix,
        display_points=piece_seg_script.CUBE_LATTICE_DISPLAY_POINTS_MM,
        display_edge_indexes=piece_seg_script.CUBE_LATTICE_DISPLAY_EDGE_INDEXES,
    )

    assert result is not None
    assert result.inlier_count == len(object_points)
    assert result.point_names == tuple()
    assert result.matched_error < 1e-3


def test_solve_lattice_graph_pnp_refines_candidate_pose(monkeypatch):
    calls = []
    camera_matrix = piece_seg_script.default_camera_matrix((120, 120))
    object_points = piece_seg_script.cube_lattice_points_for_coords(
        ((0, 0, 0), (1, 0, 0), (0, 1, 0)),
    )
    image_points = np.asarray([[20.0, 20.0], [40.0, 20.0], [20.0, 40.0]], dtype=np.float32)

    def fake_solve_pnp_generic(*_args, **_kwargs):
        return (
            1,
            (np.zeros((3, 1), dtype=np.float64),),
            (np.asarray([[0.0], [0.0], [180.0]], dtype=np.float64),),
            np.zeros((1, 1), dtype=np.float64),
        )

    def fake_refine(_object_points, _image_points, _camera_matrix, _dist_coeffs, rvec, tvec):
        calls.append((rvec.copy(), tvec.copy()))
        return rvec + 1.0, tvec

    monkeypatch.setattr(piece_seg_script.cv2, "solvePnPGeneric", fake_solve_pnp_generic)
    monkeypatch.setattr(piece_seg_script.cv2, "solvePnPRefineLM", fake_refine)

    result = piece_seg_script.solve_lattice_graph_pnp(
        image_points,
        object_points,
        camera_matrix,
        display_points=piece_seg_script.CUBE_LATTICE_DISPLAY_POINTS_MM,
        display_edge_indexes=piece_seg_script.CUBE_LATTICE_DISPLAY_EDGE_INDEXES,
    )

    assert result is not None
    assert calls
    np.testing.assert_allclose(result.rvec, np.ones((3, 1), dtype=np.float64))


def test_solve_lattice_graph_pnp_prefers_visible_surface_candidate(monkeypatch):
    image_points = np.asarray([[20.0, 20.0], [40.0, 20.0], [20.0, 40.0]], dtype=np.float32)
    object_points = piece_seg_script.cube_lattice_points_for_coords(
        ((0, 0, 0), (1, 0, 0), (0, 1, 0)),
    )
    camera_matrix = piece_seg_script.default_camera_matrix((100, 100))
    hidden_rvec = np.asarray([[np.pi], [0.0], [0.0]], dtype=np.float64)
    visible_rvec = np.zeros((3, 1), dtype=np.float64)

    def fake_solve_pnp_generic(*_args, **_kwargs):
        return (
            1,
            (hidden_rvec, visible_rvec),
            (np.asarray([[0.0], [0.0], [180.0]], dtype=np.float64),) * 2,
            np.zeros((2, 1), dtype=np.float64),
        )

    def fake_project_points(points, rvec, *_args, **_kwargs):
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if len(points) == len(image_points):
            offset = 0.0 if np.isclose(float(rvec[0]), np.pi) else 20.0
            projected = image_points + np.asarray([offset, 0.0], dtype=np.float32)
        else:
            projected = np.zeros((len(points), 2), dtype=np.float32)
        return projected.reshape(-1, 1, 2), None

    monkeypatch.setattr(piece_seg_script.cv2, "solvePnPGeneric", fake_solve_pnp_generic)
    monkeypatch.setattr(piece_seg_script.cv2, "solvePnPRefineLM", lambda *_args: (_args[-2], _args[-1]))
    monkeypatch.setattr(piece_seg_script.cv2, "projectPoints", fake_project_points)

    result = piece_seg_script.solve_lattice_graph_pnp(
        image_points,
        object_points,
        camera_matrix,
        display_points=piece_seg_script.CUBE_LATTICE_DISPLAY_POINTS_MM,
        display_edge_indexes=piece_seg_script.CUBE_LATTICE_DISPLAY_EDGE_INDEXES,
    )

    assert result is not None
    assert result.invisible_point_count == 0
    np.testing.assert_allclose(result.rvec, visible_rvec)


def test_draw_pnp_cube_draws_only_projected_object_point_markers(monkeypatch):
    line_calls = []
    circle_calls = []
    text_calls = []
    result = SimpleNamespace(
        projected_display_points=np.asarray(
            [[10.0, 10.0], [20.0, 10.0], [20.0, 20.0]],
            dtype=np.float32,
        ),
        projected_object_points=np.asarray(
            [[12.0, 12.0], [18.0, 18.0]],
            dtype=np.float32,
        ),
        display_edge_indexes=((0, 1), (1, 2)),
        edge_indexes=tuple(),
        image_points=np.asarray(
            [[20.0, 20.0], [24.0, 20.0], [24.0, 24.0]],
            dtype=np.float32,
        ),
        camera_matrix=piece_seg_script.default_camera_matrix((32, 32)),
        lattice_transform=piece_seg_script.CUBE_LATTICE_IDENTITY_TRANSFORM,
    )

    monkeypatch.setattr(
        piece_seg_script.cv2,
        "line",
        lambda *args, **kwargs: line_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        piece_seg_script.cv2,
        "circle",
        lambda *args, **kwargs: circle_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        piece_seg_script.cv2,
        "putText",
        lambda *args, **kwargs: text_calls.append((args, kwargs)),
    )

    piece_seg_script.draw_pnp_cube(np.zeros((32, 32, 3), dtype=np.uint8), result)

    assert line_calls
    assert len(circle_calls) == 4
    centers = [call[0][1] for call in circle_calls]
    assert centers == [(12, 12), (12, 12), (18, 18), (18, 18)]
    assert not any("1:" in call[0][1] for call in text_calls)


def test_draw_pose_graph_correspondence_highlights_used_graph_and_labels_coords():
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    graph = piece_seg_script.PoseGraph(
        points=(
            piece_seg_script.PoseGraphPoint(0, "corner", (10.0, 10.0), 0.9),
            piece_seg_script.PoseGraphPoint(1, "corner-edge", (30.0, 10.0), 0.8),
            piece_seg_script.PoseGraphPoint(2, "corner-edge", (10.0, 30.0), 0.7),
        ),
        edges=(
            piece_seg_script.PoseGraphEdge(0, 1, 1.0, 0.0, True),
            piece_seg_script.PoseGraphEdge(0, 2, 1.0, 0.0, False),
        ),
    )
    correspondence = piece_seg_script.PoseGraphCorrespondence(
        image_points=np.asarray([[10.0, 10.0], [30.0, 10.0]], dtype=np.float32),
        object_points=np.asarray([[-27.5, -27.5, -27.5], [-9.17, -27.5, -27.5]], dtype=np.float32),
        graph_indexes=(0, 1),
        lattice_coords=((0, 0, 0), (1, 0, 0)),
    )

    piece_seg_script.draw_pose_graph_correspondence(frame, graph, correspondence)

    assert np.any(frame != 0)


def test_cube_corner_geometry_uses_55mm_cube_size():
    assert piece_seg_script.CUBE_SIZE_MM == 55.0
    assert piece_seg_script.CUBE_LAYER_SIZE_MM == 55.0 / 3.0
    np.testing.assert_allclose(
        piece_seg_script.CUBE_CORNER_POINTS_MM.max(axis=0),
        np.array([27.5, 27.5, 27.5]),
    )
    np.testing.assert_allclose(
        piece_seg_script.CUBE_CORNER_POINTS_MM.min(axis=0),
        np.array([-27.5, -27.5, -27.5]),
    )


def test_sticker_sampling_uses_same_mask_as_rendering():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :] = [40, 40, 40]
    frame[10:20, 10:20] = [0, 0, 220]
    frame[24:30, 24:30] = [220, 0, 0]
    args = default_args()

    _, bgr_samples = piece_seg_script.process_frame(
        frame,
        StubBboxDetector(),
        StubStickerModel(),
        args,
    )

    assert len(bgr_samples) > 1
    assert (bgr_samples[:, 2] > bgr_samples[:, 0]).all()
    assert not (bgr_samples == [220, 0, 0]).all(axis=1).any()


def test_segmentation_padding_changes_displayed_and_sampled_mask():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :] = [40, 40, 40]
    frame[12:18, 12:18] = [0, 0, 220]
    args = default_args(segmentation_padding_px=-2)

    output, bgr_samples = piece_seg_script.process_frame(
        frame,
        StubBboxDetector(),
        StubStickerModel(),
        args,
    )

    assert len(bgr_samples) > 1
    assert (bgr_samples[:, 2] > bgr_samples[:, 0]).all()
    assert (output[10, 10] == [0, 0, 0]).all()
    assert not (output[15, 15] == [0, 0, 0]).all()


def test_sticker_color_sampling_returns_ten_samples_per_sticker():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :] = [40, 40, 40]
    frame[10:20, 10:20] = [0, 0, 220]

    _, bgr_samples = piece_seg_script.process_frame(
        frame,
        StubBboxDetector(),
        StubStickerModel(),
        default_args(),
    )

    assert bgr_samples.shape == (10, 3)
    assert (bgr_samples[:, 2] > bgr_samples[:, 0]).all()


def test_sticker_color_sample_points_are_random_inside_mask():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 1:9] = True

    ys, xs = piece_seg_script.random_mask_sample_points(mask, 6)
    repeat_ys, repeat_xs = piece_seg_script.random_mask_sample_points(mask, 6)

    assert len(xs) == 6
    assert len(set(zip(xs, ys))) == 6
    assert mask[ys, xs].all()
    assert np.array_equal(xs, repeat_xs)
    assert np.array_equal(ys, repeat_ys)


def test_parser_exposes_original_underlay_option():
    args = piece_seg_script.build_parser().parse_args(["--show-original-underlay"])

    assert args.show_original_underlay is True


def test_parser_shows_original_underlay_by_default_and_can_hide_it():
    assert piece_seg_script.build_parser().parse_args([]).show_original_underlay is True
    assert piece_seg_script.build_parser().parse_args(["--hide-original-underlay"]).show_original_underlay is False


def test_parser_exposes_roi_tracking_gain_option():
    args = piece_seg_script.build_parser().parse_args(["--roi-tracking-gain", "1.8"])

    assert args.roi_tracking_gain == 1.8


def test_parser_exposes_roi_tracking_padding_option():
    args = piece_seg_script.build_parser().parse_args(["--roi-tracking-padding", "0.25"])

    assert args.roi_tracking_padding == 0.25


def test_parser_displays_by_default_and_can_disable_display():
    assert piece_seg_script.build_parser().parse_args([]).display is True
    assert piece_seg_script.build_parser().parse_args(["--no-display"]).display is False


def test_parser_exposes_density_color_classifier_options():
    args = piece_seg_script.build_parser().parse_args(
        [
            "--color-classifier-mode",
            "density-cluster",
            "--cluster-eps-radians",
            "0.02",
            "--density-cluster-min-neighbors",
            "70",
            "--sticker-color-min-brightness",
            "50",
        ]
    )

    assert args.color_classifier_mode == "density-cluster"
    assert args.density_cluster_eps_radians == 0.02
    assert args.density_cluster_min_neighbors == 70
    assert args.sticker_color_min_brightness == 50


def test_color_plot_frames_zero_disables_color_features():
    args = piece_seg_script.build_parser().parse_args(
        [
            "--color-plot-frames",
            "0",
            "--rgb-color-plot",
            "runs/colors.png",
        ]
    )

    assert piece_seg_script.color_features_enabled(args) is False
    assert piece_seg_script.color_plot_outputs_requested(args) is False


def test_classifier_cluster_indexes_match_display_classifier():
    classifier = piece_seg_script.ColorDensityClassifier(
        required_frames=1,
        cluster_count=2,
        min_brightness=1,
        density_cluster_eps_radians=0.2,
        density_cluster_min_neighbors=2,
    )
    classifier.centroids = np.array(
        [
            [220.0, 0.0, 0.0],
            [0.0, 220.0, 0.0],
        ],
        dtype=np.float32,
    )
    classifier.cluster_samples = [
        np.array([[220.0, 0.0, 0.0], [210.0, 5.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 220.0, 0.0], [5.0, 210.0, 0.0]], dtype=np.float32),
    ]
    bgr_samples = np.array(
        [
            [0, 0, 220],
            [0, 220, 0],
            [1, 1, 1],
        ],
        dtype=np.uint8,
    )

    indexes = piece_seg_script.classifier_cluster_indexes_for_bgr_samples(
        classifier,
        bgr_samples,
    )

    assert indexes.tolist() == [0, 1, -1]


def test_normalized_rgb_html_plot_path_uses_rgb_html_stem():
    path = piece_seg_script.normalized_rgb_html_plot_path(Path("runs/rgb.html"))

    assert path == Path("runs/rgb_normalized.html")


def test_normalized_rgb_html_plot_path_adds_html_suffix():
    path = piece_seg_script.normalized_rgb_html_plot_path(Path("runs/rgb"))

    assert path == Path("runs/rgb_normalized.html")
