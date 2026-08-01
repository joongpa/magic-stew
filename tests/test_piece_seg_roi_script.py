import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


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
    def solve(self, image_points, image_size, *, edge_points=None):
        return None


class RaisingPoseSolver:
    def solve(self, image_points, image_size, *, edge_points=None):
        raise AssertionError("PnP solver should not run")


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

    assert piece_seg_script.POSE_IMAGE_SIZE == 256
    assert piece_seg_script.POSE_OBJECT_CONFIDENCE == 0.5
    assert piece_seg_script.POSE_KEYPOINT_CONFIDENCE == 0.2
    assert piece_seg_script.PNP_BOOTSTRAP_FRAME_COUNT == 12
    assert piece_seg_script.PNP_FOCAL_SCALE_STEPS == 5
    assert piece_seg_script.PNP_HYPOTHESIS_POINT_COUNT == 3
    assert piece_seg_script.PNP_HYPOTHESIS_ATTEMPT_LIMIT == 3
    assert piece_seg_script.PNP_CORNER_DETECTION_LIMIT == 8
    assert piece_seg_script.PNP_EDGE_DETECTION_LIMIT == 12
    assert piece_seg_script.PNP_INLIER_THRESHOLD_PX == 10.0
    assert piece_seg_script.PNP_TEMPORAL_TRACKING_ENABLED is True
    assert piece_seg_script.PNP_TEMPORAL_MAX_CORNER_SHIFT_PX == 18.0
    assert piece_seg_script.MID_TURN_POSE_ESTIMATION_ENABLED is True
    assert piece_seg_script.MID_TURN_ANGLE_DEGREES == tuple(range(5, 50, 5))


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
        SimpleNamespace(adjacency_edge_dilation=3),
    )

    assert evidence[0].accepted
    assert evidence[0].coverage > piece_seg_script.ADJACENCY_MIN_COVERAGE
    assert evidence[0].max_uncovered_gap < (
        piece_seg_script.ADJACENCY_MAX_UNCOVERED_GAP_RATIO * 90.0
    )


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
        ),
    )

    skipped_edge = next(edge for edge in evidence if edge.start_index == 0 and edge.end_index == 2)

    assert skipped_edge.coverage == 0.0
    assert not skipped_edge.accepted


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


def test_cube_correspondence_triples_are_filtered_by_orientation_symmetry():
    assert len(piece_seg_script.CUBE_ORIENTATION_CORNER_PERMUTATIONS) == 24
    assert len(piece_seg_script.CUBE_CORNER_CORRESPONDENCE_TRIPLES) == 14
    assert all(len(set(triple)) == 3 for triple in piece_seg_script.CUBE_CORNER_CORRESPONDENCE_TRIPLES)


def test_mid_turn_model_candidates_cover_six_faces_and_nine_angles():
    candidates = piece_seg_script.mid_turn_corner_model_candidates()

    assert len(candidates) == 54
    assert {
        face_name
        for face_name, _angle, _corners, _edge_points, _edges, _mask, _display_points, _display_edges in candidates
    } == {"B", "F", "U", "D", "R", "L"}
    assert {
        angle
        for _face_name, angle, _corners, _edge_points, _edges, _mask, _display_points, _display_edges in candidates
    } == set(range(5, 50, 5))


def test_mid_turn_scoring_model_uses_drawn_edges_instead_of_direct_layer_connections():
    corner_points, edge_points, edge_indexes, allowed_corner_mask = piece_seg_script.mid_turn_scoring_model("U", 25.0)

    assert len(corner_points) == 8
    assert len(edge_points) == 12
    assert len(allowed_corner_mask) == 20
    assert allowed_corner_mask[:8].all()
    assert not allowed_corner_mask[8:].any()
    assert len(edge_indexes) == 16
    assert not any(
        start < 8
        and end < 8
        and np.sign(piece_seg_script.CUBE_CORNER_POINTS_MM[start, 1])
        != np.sign(piece_seg_script.CUBE_CORNER_POINTS_MM[end, 1])
        for start, end in edge_indexes
    )
    assert all(start >= 8 or end >= 8 for start, end in edge_indexes)


def test_mid_turn_display_model_draws_two_exact_cuboids():
    display_points, display_edges = piece_seg_script.mid_turn_display_model("U", 25.0)

    assert len(display_points) == 16
    assert len(display_edges) == len(piece_seg_script.CUBE_CORNER_EDGES) * 2
    np.testing.assert_allclose(
        display_points[:8, 1].max(),
        piece_seg_script.CUBE_HALF_SIZE_MM - piece_seg_script.CUBE_LAYER_SIZE_MM,
    )
    assert all(
        (start < 8 and end < 8) or (start >= 8 and end >= 8)
        for start, end in display_edges
    )


def test_mid_turn_solver_recovers_pose_when_rigid_solver_rejects_deformed_corners():
    camera_matrix = piece_seg_script.default_camera_matrix((640, 480), focal_scale=1.1)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    rvec = np.array([[0.25], [-0.15], [0.12]], dtype=np.float64)
    tvec = np.array([[5.0], [-3.0], [180.0]], dtype=np.float64)
    model_corner_points = piece_seg_script.mid_turn_corner_points("U", 25.0)
    projected, _ = piece_seg_script.cv2.projectPoints(
        model_corner_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    image_points = projected.reshape(-1, 2)[[2, 0, 3, 1, 6, 4, 7, 5]]

    rigid_result = piece_seg_script.solve_best_corner_pnp(
        image_points,
        [camera_matrix],
        inlier_threshold=2.0,
    )
    mid_turn_result = piece_seg_script.solve_best_mid_turn_corner_pnp(
        image_points,
        [camera_matrix],
        inlier_threshold=2.0,
    )

    assert rigid_result is None
    assert mid_turn_result is not None
    assert mid_turn_result.corner_inlier_count == 8
    assert mid_turn_result.error < 1e-3
    assert mid_turn_result.pose_label.endswith("+25")


def test_visible_cube_masks_allow_only_camera_facing_surface_at_identity_pose():
    visible_corners, visible_edges = piece_seg_script.visible_cube_corner_and_edge_masks(
        np.zeros((3, 1), dtype=np.float64),
        np.array([[0.0], [0.0], [180.0]], dtype=np.float64),
    )

    assert set(np.flatnonzero(visible_corners)) == {4, 5, 6, 7}
    assert set(np.flatnonzero(visible_edges)) == {4, 5, 6, 7}


def test_corner_score_rejects_back_side_corner_assignments():
    projected = np.asarray(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
            [20.0, 20.0],
            [30.0, 20.0],
            [30.0, 30.0],
            [20.0, 30.0],
        ],
        dtype=np.float32,
    )
    visible_corners = np.asarray([False, False, False, False, True, True, True, True])

    hidden_error, hidden_count = piece_seg_script.robust_projected_corner_score(
        np.asarray([[0.0, 0.0]], dtype=np.float32),
        projected,
        allowed_corner_mask=visible_corners,
        inlier_threshold=1.0,
    )
    visible_error, visible_count = piece_seg_script.robust_projected_corner_score(
        np.asarray([[20.0, 20.0]], dtype=np.float32),
        projected,
        allowed_corner_mask=visible_corners,
        inlier_threshold=1.0,
    )

    assert hidden_count == 0
    assert np.isinf(hidden_error)
    assert visible_count == 1
    assert visible_error == 0.0


def test_collect_corner_image_points_keeps_highest_confidence_corners_first():
    result = SimpleNamespace(
        keypoints=None,
        boxes=SimpleNamespace(cls=TensorLike(np.array([0, 0, 0, 0], dtype=np.float32))),
    )
    pose_model = SimpleNamespace(names={0: "corner"})
    xy = np.asarray(
        [
            [[1.0, 1.0]],
            [[2.0, 2.0]],
            [[3.0, 3.0]],
            [[4.0, 4.0]],
        ],
        dtype=np.float32,
    )
    conf = np.asarray([[0.2], [0.9], [0.4], [0.8]], dtype=np.float32)

    points = piece_seg_script.collect_corner_image_points(
        result,
        pose_model,
        xy,
        conf,
        confidence_threshold=0.1,
        offset_xy=(0, 0),
        scale_xy=(1.0, 1.0),
        point_limit=3,
    )

    np.testing.assert_allclose(
        points,
        np.asarray([[2.0, 2.0], [4.0, 4.0], [3.0, 3.0]], dtype=np.float32),
    )


def test_pnp_hypothesis_point_sets_are_capped_to_three_attempts():
    points = np.asarray(
        [[float(index), float(index)] for index in range(8)],
        dtype=np.float32,
    )

    point_sets = piece_seg_script.pnp_hypothesis_point_sets(points)

    assert len(point_sets) == 3
    np.testing.assert_allclose(point_sets[0], points[[0, 1, 2]])
    np.testing.assert_allclose(point_sets[1], points[[0, 1, 3]])
    np.testing.assert_allclose(point_sets[2], points[[0, 1, 4]])


def test_solve_best_corner_pnp_recovers_low_error_from_three_unlabeled_corners():
    camera_matrix = piece_seg_script.default_camera_matrix((640, 480), focal_scale=1.1)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    corner_indexes = [0, 2, 5]
    object_points = piece_seg_script.CUBE_CORNER_POINTS_MM[corner_indexes]
    rvec = np.array([[0.25], [-0.15], [0.12]], dtype=np.float64)
    tvec = np.array([[5.0], [-3.0], [180.0]], dtype=np.float64)
    projected, _ = piece_seg_script.cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    image_points = projected.reshape(-1, 2)[[2, 0, 1]]

    result = piece_seg_script.solve_best_corner_pnp(image_points, [camera_matrix])

    assert result is not None
    assert result.error < 1e-3
    assert len(result.corner_names) == 3
    assert len(set(result.corner_names)) == 3


def test_solve_best_corner_pnp_scores_and_refines_with_edge_midpoints():
    camera_matrix = piece_seg_script.default_camera_matrix((640, 480), focal_scale=1.1)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    rvec = np.array([[0.25], [-0.15], [0.12]], dtype=np.float64)
    tvec = np.array([[5.0], [-3.0], [180.0]], dtype=np.float64)
    projected, _ = piece_seg_script.cv2.projectPoints(
        piece_seg_script.CUBE_CORNER_POINTS_MM,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    projected = projected.reshape(-1, 2)
    image_points = projected[[0, 2, 5]]

    corner_only_result = piece_seg_script.solve_best_corner_pnp(image_points, [camera_matrix])
    assert corner_only_result is not None
    edge_points = np.asarray(
        [
            corner_only_result.projected_edge_points[0]
        ],
        dtype=np.float32,
    )

    result = piece_seg_script.solve_best_corner_pnp(
        image_points,
        [camera_matrix],
        scoring_edge_points=edge_points,
        inlier_threshold=2.0,
    )

    assert result is not None
    assert len(result.image_points) == 4
    assert len(result.object_points) == 4
    assert result.corner_inlier_count == 3
    assert result.edge_inlier_count == 1
    assert result.inlier_count == 4


def test_projected_edge_score_accepts_points_on_same_projected_edge_segment():
    projected = np.asarray(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
            [0.0, 20.0],
            [10.0, 20.0],
            [10.0, 30.0],
            [0.0, 30.0],
        ],
        dtype=np.float32,
    )

    midpoint_error, midpoint_count = piece_seg_script.robust_projected_edge_score(
        np.asarray([[5.0, 0.0]], dtype=np.float32),
        projected,
        inlier_threshold=1.0,
    )
    same_line_error, same_line_count = piece_seg_script.robust_projected_edge_score(
        np.asarray([[2.0, 0.0]], dtype=np.float32),
        projected,
        inlier_threshold=1.0,
    )

    assert midpoint_error == 0.0
    assert midpoint_count == 1
    assert same_line_error == 0.0
    assert same_line_count == 1


def test_solve_best_corner_pnp_rejects_when_an_observed_corner_does_not_fit():
    camera_matrix = piece_seg_script.default_camera_matrix((640, 480), focal_scale=1.1)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    corner_indexes = [0, 2, 5]
    object_points = piece_seg_script.CUBE_CORNER_POINTS_MM[corner_indexes]
    rvec = np.array([[0.25], [-0.15], [0.12]], dtype=np.float64)
    tvec = np.array([[5.0], [-3.0], [180.0]], dtype=np.float64)
    projected, _ = piece_seg_script.cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    good_points = projected.reshape(-1, 2)[[2, 0, 1]]
    image_points = np.vstack([good_points, np.array([[2000.0, -1500.0]], dtype=np.float32)])

    result = piece_seg_script.solve_best_corner_pnp(
        image_points,
        [camera_matrix],
        inlier_threshold=2.0,
    )

    assert result is None


def test_solve_best_corner_pnp_fits_three_corners_but_scores_all_observed_corners():
    camera_matrix = piece_seg_script.default_camera_matrix((640, 480), focal_scale=1.1)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    corner_indexes = [0, 2, 5, 7]
    object_points = piece_seg_script.CUBE_CORNER_POINTS_MM[corner_indexes]
    rvec = np.array([[0.25], [-0.15], [0.12]], dtype=np.float64)
    tvec = np.array([[5.0], [-3.0], [180.0]], dtype=np.float64)
    projected, _ = piece_seg_script.cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    image_points = projected.reshape(-1, 2)[[2, 0, 3, 1]]

    result = piece_seg_script.solve_best_corner_pnp(
        image_points,
        [camera_matrix],
        inlier_threshold=2.0,
    )

    assert result is not None
    assert len(result.image_points) == 4
    assert len(result.object_points) == 4
    assert len(result.scoring_points) == 4
    assert result.corner_inlier_count == 4


def test_temporal_corner_pnp_uses_previous_projection_before_full_search(monkeypatch):
    camera_matrix = piece_seg_script.default_camera_matrix((640, 480), focal_scale=1.1)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    corner_indexes = [0, 2, 5, 7]
    object_points = piece_seg_script.CUBE_CORNER_POINTS_MM[corner_indexes]
    rvec = np.array([[0.25], [-0.15], [0.12]], dtype=np.float64)
    previous_tvec = np.array([[5.0], [-3.0], [180.0]], dtype=np.float64)
    current_tvec = np.array([[5.4], [-3.2], [180.5]], dtype=np.float64)
    previous_projected, _ = piece_seg_script.cv2.projectPoints(
        object_points,
        rvec,
        previous_tvec,
        camera_matrix,
        dist_coeffs,
    )
    current_projected, _ = piece_seg_script.cv2.projectPoints(
        object_points,
        rvec,
        current_tvec,
        camera_matrix,
        dist_coeffs,
    )
    point_order = [2, 0, 3, 1]
    previous_points = previous_projected.reshape(-1, 2)[point_order]
    current_points = current_projected.reshape(-1, 2)[point_order]
    previous_result = piece_seg_script.solve_best_corner_pnp(
        previous_points,
        [camera_matrix],
        inlier_threshold=2.0,
    )
    assert previous_result is not None

    def fail_full_search(*args, **kwargs):
        raise AssertionError("temporal PnP should be tried before full correspondence search")

    monkeypatch.setattr(piece_seg_script, "solve_best_corner_pnp", fail_full_search)

    result = piece_seg_script.solve_temporal_or_best_corner_pnp(
        current_points,
        [camera_matrix],
        previous_result=previous_result,
        inlier_threshold=2.0,
    )

    assert result is not None
    assert result.corner_inlier_count == len(current_points)
    assert result.error < 1e-3


def test_temporal_corner_pnp_falls_back_when_previous_projection_is_not_close(monkeypatch):
    camera_matrix = piece_seg_script.default_camera_matrix((640, 480), focal_scale=1.1)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    corner_indexes = [0, 2, 5]
    object_points = piece_seg_script.CUBE_CORNER_POINTS_MM[corner_indexes]
    rvec = np.array([[0.25], [-0.15], [0.12]], dtype=np.float64)
    tvec = np.array([[5.0], [-3.0], [180.0]], dtype=np.float64)
    projected, _ = piece_seg_script.cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    previous_points = projected.reshape(-1, 2)
    previous_result = piece_seg_script.solve_best_corner_pnp(previous_points, [camera_matrix])
    assert previous_result is not None
    current_points = previous_points + np.asarray([200.0, -120.0], dtype=np.float32)
    fallback_result = object()

    def fake_full_search(*args, **kwargs):
        return fallback_result

    monkeypatch.setattr(piece_seg_script, "solve_best_corner_pnp", fake_full_search)

    result = piece_seg_script.solve_temporal_or_best_corner_pnp(
        current_points,
        [camera_matrix],
        previous_result=previous_result,
    )

    assert result is fallback_result


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
