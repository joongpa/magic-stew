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
        "draw_sticker_graph": False,
        "graph_adjacency_radius": 6,
        "graph_min_sticker_area": 12,
        "draw_cube_grid_fit": False,
        "cube_grid_fit_line_thickness": 2,
        "cube_grid_fit_min_coverage": 0.88,
        "empty_space_line_width": 5,
        "basis_angle_bucket_degrees": 5.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def projected_cube_sticker_masks(
    *,
    roi_shape=(140, 140),
    origin=(54.0, 61.0),
    axes=((48.0, -18.0), (12.0, 46.0), (-42.0, -22.0)),
    inset=0.18,
):
    masks = []
    origin = np.asarray(origin, dtype=np.float32)
    axes = np.asarray(axes, dtype=np.float32)
    for first_index, second_index in ((0, 1), (1, 2), (2, 0)):
        first_axis = axes[first_index]
        second_axis = axes[second_index]
        for row in range(3):
            for column in range(3):
                low_u = (column + inset) / 3.0
                high_u = (column + 1.0 - inset) / 3.0
                low_v = (row + inset) / 3.0
                high_v = (row + 1.0 - inset) / 3.0
                points = np.asarray(
                    [
                        origin + first_axis * low_u + second_axis * low_v,
                        origin + first_axis * high_u + second_axis * low_v,
                        origin + first_axis * high_u + second_axis * high_v,
                        origin + first_axis * low_u + second_axis * high_v,
                    ],
                    dtype=np.float32,
                )
                mask_u8 = np.zeros(roi_shape, dtype=np.uint8)
                cv2 = piece_seg_script.cv2
                cv2.fillConvexPoly(mask_u8, np.round(points).astype(np.int32), 1)
                masks.append(mask_u8 > 0)
    return masks


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
    assert output.shape[0] == frame.shape[0]
    assert output.shape[1] > frame.shape[1]
    assert np.any(output[:, : frame.shape[1]] != frame)


def test_draw_sticker_segments_prunes_face_mostly_covered_by_larger_face():
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    roi = np.zeros_like(frame)
    args = default_args(sticker_imgsz=256, mask_opacity=1.0)

    count, bgr_samples, tracking_bounds, projection_panel = piece_seg_script.draw_sticker_segments(
        frame,
        roi,
        StubOverlappingFaceSegmentationModel(),
        (0, 0, 256, 256),
        args,
    )

    assert count == 1
    assert len(bgr_samples) > 0
    assert tracking_bounds == (40, 40, 180, 180)
    assert projection_panel is not None
    assert np.array_equal(frame[35, 35], np.asarray([0, 0, 0], dtype=np.uint8))
    assert np.array_equal(
        frame[50, 50],
        np.asarray(piece_seg_script.FACE_INSTANCE_MASK_COLORS[0], dtype=np.uint8),
    )


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


def test_low_intersection_lines_draw_four_side_lines_per_sticker_quad():
    left_mask = np.zeros((40, 40), dtype=bool)
    right_mask = np.zeros((40, 40), dtype=bool)
    left_mask[10:30, 8:16] = True
    right_mask[10:30, 24:32] = True

    overlay = piece_seg_script.low_intersection_sticker_lines(
        [left_mask, right_mask],
        roi_shape=(40, 40),
        line_width_px=5,
    )
    lines = overlay.lines

    assert len(overlay.sampled_points_xy) == 8
    assert len(overlay.quad_edge_segments_xy) == 8
    assert len(lines) == 8
    point_indexes = np.round(overlay.sampled_points_xy).astype(np.int32)
    assert left_mask[
        point_indexes[:, 1],
        point_indexes[:, 0],
    ].any()
    assert right_mask[
        point_indexes[:, 1],
        point_indexes[:, 0],
    ].any()
    assert all(line.length_px > 30 for line in lines)


def test_low_intersection_lines_are_deterministic_for_same_mask():
    mask = np.zeros((60, 60), dtype=bool)
    mask[8:22, 8:52] = True
    mask[38:52, 8:52] = True
    mask[8:52, 8:16] = True

    first_overlay = piece_seg_script.low_intersection_sticker_lines(
        [mask],
        roi_shape=(60, 60),
        line_width_px=5,
    )
    second_overlay = piece_seg_script.low_intersection_sticker_lines(
        [mask],
        roi_shape=(60, 60),
        line_width_px=5,
    )

    assert np.array_equal(first_overlay.sampled_points_xy, second_overlay.sampled_points_xy)
    assert [(line.start_xy, line.end_xy) for line in first_overlay.lines] == [
        (line.start_xy, line.end_xy) for line in second_overlay.lines
    ]


def test_low_intersection_line_angles_come_from_quad_sides():
    mask = np.zeros((50, 50), dtype=bool)
    mask[20:28, 8:42] = True

    overlay = piece_seg_script.low_intersection_sticker_lines(
        [mask],
        roi_shape=(50, 50),
        line_width_px=3,
    )

    assert overlay.lines
    side_angles = []
    for start, end in overlay.quad_edge_segments_xy:
        delta = end - start
        side_angles.append(float(np.degrees(np.arctan2(delta[1], delta[0])) % 180.0))

    assert all(
        min(
            abs(((line.angle_degrees - side_angle + 90.0) % 180.0) - 90.0)
            for side_angle in side_angles
        ) <= 1e-3
        for line in overlay.lines
    )


def test_basis_vectors_average_angles_around_two_histogram_peaks():
    lines = [
        piece_seg_script.LowIntersectionLine(
            score=0.0,
            intersection_px=0,
            length_px=10,
            angle_degrees=angle,
            start_xy=(0.0, 0.0),
            end_xy=(1.0, 0.0),
        )
        for angle in [2.0, 7.0, 12.0, 48.0, 53.0, 58.0, 120.0]
    ]

    vectors = piece_seg_script._basis_vectors_from_line_angle_histogram(
        lines,
        bucket_degrees=5.0,
    )
    angles = sorted(float(np.degrees(np.arctan2(vector[1], vector[0])) % 180.0) for vector in vectors)

    assert len(angles) == 2
    assert abs(angles[0] - 7.0) < 0.5
    assert abs(angles[1] - 53.0) < 0.5


def test_overlay_points_come_from_sticker_quads():
    left_mask = np.zeros((50, 50), dtype=bool)
    right_mask = np.zeros((50, 50), dtype=bool)
    left_mask[12:38, 8:16] = True
    right_mask[12:38, 34:42] = True

    overlay = piece_seg_script.low_intersection_sticker_lines(
        [left_mask, right_mask],
        roi_shape=(50, 50),
        line_width_px=5,
    )

    points = overlay.sampled_points_xy
    assert len(points) == 8
    edge_mask = piece_seg_script._sticker_edge_mask([left_mask, right_mask], roi_shape=(50, 50))
    assert edge_mask is not None
    indexes = np.round(points).astype(np.int32)
    assert edge_mask[indexes[:, 1], indexes[:, 0]].all()


def test_sticker_quad_fit_preserves_skewed_quadrilateral_shape():
    mask = np.zeros((50, 50), dtype=np.uint8)
    expected = np.asarray(
        [
            [10, 12],
            [36, 8],
            [32, 34],
            [7, 28],
        ],
        dtype=np.int32,
    )
    piece_seg_script.cv2.fillConvexPoly(mask, expected, 1)

    quad = piece_seg_script._sticker_mask_quad_points(mask > 0, roi_shape=(50, 50))

    assert quad is not None
    assert len(quad) == 4
    for corner in expected.astype(np.float32):
        distances = np.linalg.norm(quad - corner, axis=1)
        assert float(distances.min()) <= 2.0


def test_face_parallelogram_reconciliation_shares_neighbor_side():
    left = np.asarray(
        [
            [10.0, 10.0],
            [40.0, 12.0],
            [42.0, 42.0],
            [12.0, 40.0],
        ],
        dtype=np.float32,
    )
    right = np.asarray(
        [
            [43.0, 12.0],
            [72.0, 10.0],
            [74.0, 40.0],
            [45.0, 42.0],
        ],
        dtype=np.float32,
    )

    reconciled = piece_seg_script._reconcile_shared_face_parallelogram_sides(
        [left, right],
        roi_shape=(100, 100),
    )

    left_edge = piece_seg_script._quad_edge_points(reconciled[0], 1)
    right_edge = piece_seg_script._quad_edge_points(reconciled[1], 3)
    assert np.allclose(left_edge[0], right_edge[1])
    assert np.allclose(left_edge[1], right_edge[0])
    for quad in reconciled:
        assert np.allclose(quad[1] - quad[0], quad[2] - quad[3])
        assert np.allclose(quad[3] - quad[0], quad[2] - quad[1])


def test_face_projection_uses_image_under_mask_and_blacks_outside_mask():
    roi = np.zeros((10, 10, 3), dtype=np.uint8)
    roi[:, :] = [20, 80, 200]
    mask = np.zeros((10, 10), dtype=bool)
    mask[:, :5] = True
    quad = np.asarray(
        [
            [0.0, 0.0],
            [9.0, 0.0],
            [9.0, 9.0],
            [0.0, 9.0],
        ],
        dtype=np.float32,
    )

    projection = piece_seg_script.project_masked_face_to_square(
        roi,
        mask,
        quad,
        square_size=10,
    )

    assert np.array_equal(projection[5, 2], np.asarray([20, 80, 200], dtype=np.uint8))
    assert np.array_equal(projection[5, 8], np.asarray([0, 0, 0], dtype=np.uint8))


def test_face_projection_panel_draws_three_by_three_grid():
    roi = np.zeros((30, 30, 3), dtype=np.uint8)
    roi[:, :] = [20, 80, 200]
    mask = np.ones((30, 30), dtype=bool)

    panel = piece_seg_script.face_projection_panel(
        roi,
        [mask],
        roi_shape=(30, 30),
        output_height=80,
    )

    assert panel is not None
    padding = 10
    square_size = 60
    first_line = padding + int(round((square_size - 1) / 3.0))
    second_line = padding + int(round((square_size - 1) * 2.0 / 3.0))
    assert np.array_equal(
        panel[padding + square_size // 2, first_line],
        np.asarray(piece_seg_script.PROJECTED_FACE_GRID_COLOR, dtype=np.uint8),
    )
    assert np.array_equal(
        panel[second_line, padding + square_size // 2],
        np.asarray(piece_seg_script.PROJECTED_FACE_GRID_COLOR, dtype=np.uint8),
    )


def test_low_intersection_lines_extend_to_roi_bounds():
    mask = np.zeros((50, 50), dtype=bool)
    mask[12:38, 12:18] = True
    mask[12:38, 32:38] = True

    overlay = piece_seg_script.low_intersection_sticker_lines(
        [mask],
        roi_shape=(50, 50),
        line_width_px=6,
    )
    lines = overlay.lines

    assert lines
    assert any(line.length_px > 45 for line in lines)
    assert all(0 <= line.start_xy[0] <= 49 and 0 <= line.end_xy[0] <= 49 for line in lines)
    assert all(0 <= line.start_xy[1] <= 49 and 0 <= line.end_xy[1] <= 49 for line in lines)


def test_draw_low_intersection_lines_uses_inclusion_width():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    overlay = piece_seg_script.EmptySpaceLineOverlay(
        lines=[
            piece_seg_script.LowIntersectionLine(
                score=0.1,
                intersection_px=0,
                length_px=20,
                angle_degrees=0.0,
                start_xy=(10.0, 35.0),
                end_xy=(40.0, 35.0),
            )
        ],
        sampled_points_xy=np.empty((0, 2), dtype=np.float32),
        line_width_px=7,
    )

    piece_seg_script.draw_low_intersection_lines(
        frame,
        overlay,
        offset_xy=(0, 0),
    )

    yellow_pixels = (
        (frame[:, :, 0] < 80)
        & (frame[:, :, 1] > 100)
        & (frame[:, :, 2] > 100)
    )
    ys = np.flatnonzero(np.any(yellow_pixels[:, 20:30], axis=1))
    assert ys[-1] - ys[0] + 1 >= 7


def test_draw_low_intersection_lines_draws_fitted_quad_edges():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    overlay = piece_seg_script.EmptySpaceLineOverlay(
        lines=[],
        sampled_points_xy=np.empty((0, 2), dtype=np.float32),
        quad_edge_segments_xy=np.asarray(
            [
                [[8.0, 12.0], [30.0, 12.0]],
                [[30.0, 12.0], [30.0, 28.0]],
            ],
            dtype=np.float32,
        ),
    )

    piece_seg_script.draw_low_intersection_lines(frame, overlay, offset_xy=(0, 0))

    magenta_pixels = (
        (frame[:, :, 0] > 100)
        & (frame[:, :, 1] < 80)
        & (frame[:, :, 2] > 100)
    )
    assert magenta_pixels.any()


def test_draw_low_intersection_lines_draws_basis_vectors_at_center():
    frame = np.zeros((60, 60, 3), dtype=np.uint8)
    overlay = piece_seg_script.EmptySpaceLineOverlay(
        lines=[],
        sampled_points_xy=np.empty((0, 2), dtype=np.float32),
        basis_vectors_xy=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        basis_center_xy=(30.0, 30.0),
    )

    piece_seg_script.draw_low_intersection_lines(frame, overlay, offset_xy=(0, 0))

    blue_pixels = (
        (frame[:, :, 0] > 100)
        & (frame[:, :, 1] < 80)
        & (frame[:, :, 2] < 80)
    )
    red_pixels = (
        (frame[:, :, 0] < 80)
        & (frame[:, :, 1] < 80)
        & (frame[:, :, 2] > 100)
    )
    assert blue_pixels.any()
    assert red_pixels.any()


def test_form_fit_area_does_not_include_convex_hull_corner():
    mask = np.zeros((80, 80), dtype=bool)
    mask[10:25, 10:65] = True
    mask[10:65, 10:25] = True

    form_mask = piece_seg_script._form_fit_sticker_area_mask(mask)

    assert form_mask is not None
    assert not form_mask[62, 62]
    assert form_mask[18, 56]
    assert form_mask[56, 18]


def test_projected_cube_grid_fit_covers_stickers_and_follows_edges():
    masks = projected_cube_sticker_masks()

    fit = piece_seg_script.fit_projected_cube_grid(
        masks,
        roi_shape=(140, 140),
        line_thickness_px=1,
        minimum_coverage=0.9,
    )

    assert fit is not None
    assert fit.coverage_ratio >= 0.9
    assert fit.line_edge_distance_score < 0.2


def test_projected_cube_grid_edge_score_prefers_lines_near_sticker_edges():
    masks = projected_cube_sticker_masks()
    union_mask = piece_seg_script._union_sticker_mask(masks, (140, 140))
    assert union_mask is not None
    coverage_points = piece_seg_script._sample_mask_xy_points(union_mask, max_points=160)
    edge_distance_image = piece_seg_script._sticker_edge_distance_image(
        masks,
        roi_shape=(140, 140),
    )
    assert edge_distance_image is not None
    axes = np.asarray(
        [
            [48.0, -18.0],
            [12.0, 46.0],
            [-42.0, -22.0],
        ],
        dtype=np.float32,
    )
    origin = np.asarray((54.0, 61.0), dtype=np.float32)
    hull, lines = piece_seg_script._projected_cube_grid_geometry(axes)
    aligned_hull = hull + origin
    aligned_lines = [
        (
            tuple((np.asarray(start, dtype=np.float32) + origin).tolist()),
            tuple((np.asarray(end, dtype=np.float32) + origin).tolist()),
        )
        for start, end in lines
    ]
    shifted_lines = [
        (
            tuple((np.asarray(start, dtype=np.float32) + origin + (20.0, -16.0)).tolist()),
            tuple((np.asarray(end, dtype=np.float32) + origin + (20.0, -16.0)).tolist()),
        )
        for start, end in lines
    ]

    aligned = piece_seg_script._score_projected_cube_grid_from_edge_distance(
        edge_distance_image,
        coverage_points_xy=coverage_points,
        mask_area=float(union_mask.sum()),
        hull_points_xy=aligned_hull,
        line_segments_xy=aligned_lines,
        line_thickness_px=1,
        minimum_coverage=0.9,
    )
    shifted = piece_seg_script._score_projected_cube_grid_from_edge_distance(
        edge_distance_image,
        coverage_points_xy=coverage_points,
        mask_area=float(union_mask.sum()),
        hull_points_xy=aligned_hull,
        line_segments_xy=shifted_lines,
        line_thickness_px=1,
        minimum_coverage=0.9,
    )

    assert aligned.line_edge_distance_score < shifted.line_edge_distance_score
    assert aligned.score < shifted.score


def test_projected_cube_grid_line_score_overlooks_unsupported_empty_lines():
    enough_good_lines = np.asarray(
        [0.01, 0.02, 0.01, 0.03, 0.02, 0.01, 0.04, 0.02, 0.95, 0.98, 1.0],
        dtype=np.float32,
    )
    too_few_good_lines = np.asarray(
        [0.01, 0.02, 0.95, 0.98, 1.0, 0.99, 0.97, 0.96, 0.94, 0.93, 0.92],
        dtype=np.float32,
    )

    enough_score = piece_seg_script._robust_line_edge_distance_score(enough_good_lines)
    sparse_score = piece_seg_script._robust_line_edge_distance_score(too_few_good_lines)

    assert enough_score < 0.05
    assert sparse_score > enough_score + 0.15


def test_projected_cube_candidates_are_limited_by_observed_edge_directions():
    masks = projected_cube_sticker_masks()
    edge_mask = piece_seg_script._sticker_edge_mask(masks, roi_shape=(140, 140))
    assert edge_mask is not None

    directions = piece_seg_script._dominant_edge_direction_angles(edge_mask)
    filtered = piece_seg_script._projected_cube_axis_candidates_for_edge_directions(directions)
    all_candidates = piece_seg_script._projected_cube_axis_candidates()

    assert len(directions) == 3
    assert 0 < len(filtered) < len(all_candidates)
    assert all(
        piece_seg_script._candidate_direction_alignment_cost(candidate, directions)
        <= piece_seg_script.CUBE_GRID_FIT_DIRECTION_TOLERANCE_RADIANS
        for candidate in filtered
    )


def test_projected_cube_axis_candidates_are_possible_cube_projections():
    candidates = piece_seg_script._projected_cube_axis_candidates()

    assert candidates
    assert all(
        piece_seg_script._cube_projected_axes_are_possible(candidate)
        for candidate in candidates
    )
    assert all(
        piece_seg_script._projected_cube_silhouette_is_hexagonal(candidate)
        for candidate in candidates
    )


def test_projected_cube_axis_validator_rejects_impossible_grid():
    impossible_axes = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [-0.2, 0.9],
        ],
        dtype=np.float32,
    )

    assert not piece_seg_script._cube_projected_axes_are_possible(impossible_axes)


def test_projected_cube_axis_validator_rejects_rectangle_silhouette():
    rectangle_axes = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 0.72],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )

    assert not piece_seg_script._projected_cube_silhouette_is_hexagonal(rectangle_axes)
    assert not piece_seg_script._cube_projected_axes_are_possible(rectangle_axes)


def test_projected_cube_grid_score_penalizes_missing_sticker_coverage():
    masks = projected_cube_sticker_masks()
    union_mask = piece_seg_script._union_sticker_mask(masks, (140, 140))
    assert union_mask is not None
    axes = np.asarray(
        [
            [48.0, -18.0],
            [12.0, 46.0],
            [-42.0, -22.0],
        ],
        dtype=np.float32,
    )
    hull, lines = piece_seg_script._projected_cube_grid_geometry(axes)
    displacement = np.asarray((104.0, 108.0), dtype=np.float32)
    misplaced_hull = hull + displacement
    misplaced_lines = [
        (
            tuple((np.asarray(start, dtype=np.float32) + displacement).tolist()),
            tuple((np.asarray(end, dtype=np.float32) + displacement).tolist()),
        )
        for start, end in lines
    ]

    score = piece_seg_script._score_projected_cube_grid(
        union_mask,
        mask_area=float(union_mask.sum()),
        hull_points_xy=misplaced_hull,
        line_segments_xy=misplaced_lines,
        line_thickness_px=1,
        minimum_coverage=0.9,
    )

    assert score.coverage_ratio < 0.25
    assert score.score > 2.5


def test_parser_exposes_projected_cube_grid_fit_options():
    args = piece_seg_script.build_parser().parse_args(
        [
            "--draw-cube-grid-fit",
            "--cube-grid-fit-line-thickness",
            "3",
            "--cube-grid-fit-min-coverage",
            "0.92",
        ]
    )

    assert args.draw_cube_grid_fit is True
    assert args.cube_grid_fit_line_thickness == 3
    assert args.cube_grid_fit_min_coverage == 0.92


def test_parser_exposes_low_intersection_line_options():
    args = piece_seg_script.build_parser().parse_args(
        [
            "--empty-space-line-width",
            "6",
            "--basis-angle-bucket-degrees",
            "10",
        ]
    )

    assert args.empty_space_line_width == 6
    assert args.basis_angle_bucket_degrees == 10


def test_parser_defaults_low_intersection_line_options():
    args = piece_seg_script.build_parser().parse_args([])

    assert args.empty_space_line_width == 5
    assert args.basis_angle_bucket_degrees == 5.0


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


def test_segmentation_padding_changes_displayed_and_sampled_mask_not_graph_mask():
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


def test_process_frame_can_draw_sticker_graph_overlay():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :] = [40, 40, 40]
    frame[10:20, 10:20] = [0, 0, 220]
    args = default_args(draw_sticker_graph=True, graph_min_sticker_area=10_000)

    output, _ = piece_seg_script.process_frame(
        frame,
        StubBboxDetector(),
        StubStickerModel(),
        args,
    )

    assert np.any(np.all(output == piece_seg_script.DEFAULT_STICKER_MASK_COLOR, axis=2))


def test_process_frame_graph_uses_unpadded_mask_when_sampling_is_padded():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :] = [40, 40, 40]
    frame[12:18, 12:18] = [0, 0, 220]
    args = default_args(
        draw_sticker_graph=True,
        graph_min_sticker_area=1,
        segmentation_padding_px=-2,
    )
    captured = {}
    original_draw_sticker_graph = piece_seg_script.draw_sticker_graph

    def capture_graph(frame, graph, **kwargs):
        captured["graph"] = graph
        return frame

    piece_seg_script.draw_sticker_graph = capture_graph
    try:
        piece_seg_script.process_frame(
            frame,
            StubBboxDetector(),
            StubStickerModel(),
            args,
        )
    finally:
        piece_seg_script.draw_sticker_graph = original_draw_sticker_graph

    graph = captured["graph"]
    assert len(graph.nodes) == 1
    assert graph.nodes[0].mask.sum() == 100 * 255


def test_process_frame_grid_fit_uses_unpadded_segmentation_mask():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :] = [40, 40, 40]
    args = default_args(
        draw_cube_grid_fit=True,
        segmentation_padding_px=-2,
    )
    captured = {}
    original_low_intersection_sticker_lines = piece_seg_script.low_intersection_sticker_lines
    original_draw_low_intersection_lines = piece_seg_script.draw_low_intersection_lines

    def capture_low_intersection_lines(sticker_masks, **kwargs):
        captured["mask_sum"] = int(sticker_masks[0].sum())
        return piece_seg_script.EmptySpaceLineOverlay(
            lines=[],
            sampled_points_xy=np.empty((0, 2), dtype=np.float32),
        )

    piece_seg_script.low_intersection_sticker_lines = capture_low_intersection_lines
    piece_seg_script.draw_low_intersection_lines = lambda *args, **kwargs: None
    try:
        piece_seg_script.process_frame(
            frame,
            StubBboxDetector(),
            StubStickerModel(),
            args,
        )
    finally:
        piece_seg_script.low_intersection_sticker_lines = original_low_intersection_sticker_lines
        piece_seg_script.draw_low_intersection_lines = original_draw_low_intersection_lines

    assert captured["mask_sum"] == 100


def test_process_frame_does_not_accumulate_cube_net_without_classifier():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :] = [40, 40, 40]
    frame[10:20, 10:20] = [0, 0, 220]
    args = default_args(graph_min_sticker_area=1)
    net_state = piece_seg_script.CubeNetState()

    piece_seg_script.process_frame(
        frame,
        StubBboxDetector(),
        StubStickerModel(),
        args,
        cube_net_state=net_state,
    )

    assert net_state.cells == {}


def test_parser_exposes_sticker_graph_options():
    args = piece_seg_script.build_parser().parse_args(
        [
            "--draw-sticker-graph",
            "--adjacency-radius",
            "9",
            "--graph-min-sticker-area",
            "5",
        ]
    )

    assert args.draw_sticker_graph is True
    assert args.graph_adjacency_radius == 9
    assert args.graph_min_sticker_area == 5


def test_parser_exposes_original_underlay_option():
    args = piece_seg_script.build_parser().parse_args(["--show-original-underlay"])

    assert args.show_original_underlay is True


def test_parser_exposes_roi_tracking_gain_option():
    args = piece_seg_script.build_parser().parse_args(["--roi-tracking-gain", "1.8"])

    assert args.roi_tracking_gain == 1.8


def test_parser_exposes_roi_tracking_padding_option():
    args = piece_seg_script.build_parser().parse_args(["--roi-tracking-padding", "0.25"])

    assert args.roi_tracking_padding == 0.25


def test_parser_exposes_cube_net_output_option():
    args = piece_seg_script.build_parser().parse_args(["--save-net", "runs/net.png"])

    assert str(args.save_net) == "runs/net.png"


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
