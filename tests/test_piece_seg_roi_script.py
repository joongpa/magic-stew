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


def projected_cube_face_masks(*, roi_shape=(120, 120)):
    cv2 = piece_seg_script.cv2
    origin = np.asarray([58.0, 58.0], dtype=np.float32)
    axes = [
        np.asarray([26.08, 15.99], dtype=np.float32),
        np.asarray([-27.17, 27.42], dtype=np.float32),
        np.asarray([-13.47, -24.35], dtype=np.float32),
    ]
    masks = []
    for first_axis, second_axis in ((axes[0], axes[1]), (axes[1], axes[2]), (axes[2], axes[0])):
        points = np.asarray(
            [
                origin,
                origin + first_axis,
                origin + first_axis + second_axis,
                origin + second_axis,
            ],
            dtype=np.float32,
        )
        mask_u8 = np.zeros(roi_shape, dtype=np.uint8)
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


def test_process_frame_can_show_masks_in_side_panel():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :] = [40, 40, 40]
    frame[10:20, 10:20] = [0, 0, 220]
    args = default_args(show_masks_side_panel=True)

    output, bgr_samples = piece_seg_script.process_frame(
        frame,
        StubBboxDetector(),
        StubStickerModel(),
        args,
    )

    assert len(bgr_samples) > 1
    assert output.shape == (40, 80, 3)
    assert np.array_equal(output[:, :40], frame)
    assert np.any(output[10:20, 50:60] != 0)
    assert np.all(output[:, 40:50] == 0)


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
    assert np.any(np.all(output == piece_seg_script.DEFAULT_STICKER_MASK_COLOR, axis=2))


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
    assert projection_panel is None
    assert np.array_equal(frame[35, 35], np.asarray([0, 0, 0], dtype=np.uint8))
    assert np.array_equal(
        frame[50, 50],
        np.asarray(piece_seg_script.DEFAULT_STICKER_MASK_COLOR, dtype=np.uint8),
    )


def test_constrained_face_masks_remove_overlap_from_all_faces():
    first = np.zeros((20, 20), dtype=bool)
    first[2:14, 2:14] = True
    second = np.zeros((20, 20), dtype=bool)
    second[8:18, 8:18] = True

    selected = piece_seg_script.constrained_face_masks_by_index(
        [(0, first), (1, second)],
        roi_shape=(20, 20),
    )

    assert set(selected) == {0, 1}
    assert not selected[0][9, 9]
    assert not selected[1][9, 9]
    assert selected[0][3, 3]
    assert selected[1][16, 16]


def test_constrained_face_masks_keep_valid_cube_corner_triplet():
    face_masks = projected_cube_face_masks()
    extra = np.zeros((120, 120), dtype=bool)
    extra[80:112, 8:40] = True

    selected = piece_seg_script.constrained_face_masks_by_index(
        [(0, face_masks[0]), (1, face_masks[1]), (2, face_masks[2]), (3, extra)],
        roi_shape=(120, 120),
    )

    assert set(selected) == {0, 1, 2}
    selected_stack = np.stack([selected[index] for index in sorted(selected)], axis=0)
    assert not np.any(selected_stack.sum(axis=0) > 1)


def test_face_quad_fits_make_triplet_share_exact_corner_and_sides():
    face_masks = projected_cube_face_masks()

    fitted = piece_seg_script.face_quad_fits_for_masks(
        face_masks,
        roi_shape=(120, 120),
    )

    assert len(fitted) == 3
    quads = [quad for _mask, quad in fitted]
    edge_indexes_by_face = [[] for _ in range(3)]
    for left_index, right_index in ((0, 1), (0, 2), (1, 2)):
        match = piece_seg_script._closest_compatible_quad_edges(
            quads[left_index],
            quads[right_index],
            max_distance=0.01,
        )
        assert match is not None
        left_edge_index, right_edge_index, _right_reversed = match
        left_start, left_end = piece_seg_script._quad_edge_points(quads[left_index], left_edge_index)
        right_start, right_end = piece_seg_script._quad_edge_points(quads[right_index], right_edge_index)
        same = np.linalg.norm(left_start - right_start) + np.linalg.norm(left_end - right_end)
        reversed_distance = np.linalg.norm(left_start - right_end) + np.linalg.norm(left_end - right_start)
        assert min(float(same), float(reversed_distance)) <= 0.01
        edge_indexes_by_face[left_index].append(left_edge_index)
        edge_indexes_by_face[right_index].append(right_edge_index)

    common_corners = []
    for face_index, edge_indexes in enumerate(edge_indexes_by_face):
        corner = piece_seg_script._quad_common_corner_for_adjacent_edges(
            quads[face_index],
            edge_indexes[0],
            edge_indexes[1],
        )
        assert corner is not None
        common_corners.append(corner)
    assert np.allclose(common_corners[0], common_corners[1])
    assert np.allclose(common_corners[0], common_corners[2])


def test_projected_cube_fit_uses_orthographic_cube_basis_lengths():
    face_masks = projected_cube_face_masks()
    parallelograms = [
        piece_seg_script._face_parallelogram_points(mask, roi_shape=(120, 120))
        for mask in face_masks
    ]
    assert all(quad is not None for quad in parallelograms)
    reconciled = piece_seg_script._reconcile_shared_face_parallelogram_sides(
        parallelograms,
        roi_shape=(120, 120),
    )

    fitted = piece_seg_script._adjust_face_triplet_quads_for_shared_corner_and_mask_fit(
        face_masks,
        reconciled,
        roi_shape=(120, 120),
    )

    edge_indexes_by_face = [[] for _ in range(3)]
    shared_axis_edges = []
    for left_index, right_index in ((0, 1), (0, 2), (1, 2)):
        match = piece_seg_script._closest_compatible_quad_edges(
            fitted[left_index],
            fitted[right_index],
            max_distance=0.01,
        )
        assert match is not None
        left_edge_index, right_edge_index, _right_reversed = match
        assert abs(
            piece_seg_script._quad_edge_length(fitted[left_index], left_edge_index)
            - piece_seg_script._quad_edge_length(fitted[right_index], right_edge_index)
        ) <= 1e-4
        edge_indexes_by_face[left_index].append(left_edge_index)
        edge_indexes_by_face[right_index].append(right_edge_index)
        shared_axis_edges.append((left_index, left_edge_index))

    common_corner_indexes = []
    for edge_indexes in edge_indexes_by_face:
        corner_index = piece_seg_script._quad_common_corner_index_for_adjacent_edges(
            edge_indexes[0],
            edge_indexes[1],
        )
        assert corner_index is not None
        common_corner_indexes.append(corner_index)

    axis_vectors = []
    for face_index, edge_index in shared_axis_edges:
        far_point = piece_seg_script._quad_edge_far_point_from_common_corner(
            fitted[face_index],
            edge_index,
            common_corner_indexes[face_index],
        )
        assert far_point is not None
        axis_vectors.append(far_point - fitted[face_index][common_corner_indexes[face_index]])

    projection_moment = sum(np.outer(vector, vector) for vector in axis_vectors)
    isotropic_moment = np.eye(2, dtype=np.float32) * (float(np.trace(projection_moment)) / 2.0)
    assert np.allclose(projection_moment, isotropic_moment, rtol=0.0, atol=1e-2)


def test_projected_cube_fit_ignores_masks_after_shared_sides_are_set():
    face_masks = projected_cube_face_masks()
    parallelograms = [
        piece_seg_script._face_parallelogram_points(mask, roi_shape=(120, 120))
        for mask in face_masks
    ]
    assert all(quad is not None for quad in parallelograms)
    reconciled = piece_seg_script._reconcile_shared_face_parallelogram_sides(
        parallelograms,
        roi_shape=(120, 120),
    )
    occluded_masks = []
    for mask in face_masks:
        occluded = mask.copy()
        occluded[:60, :60] = False
        occluded_masks.append(occluded)

    edge_indexes_by_face = [[] for _ in range(3)]
    matches_by_pair = {}
    for left_index, right_index in ((0, 1), (0, 2), (1, 2)):
        match = piece_seg_script._closest_compatible_quad_edges(
            reconciled[left_index],
            reconciled[right_index],
            max_distance=20.0,
        )
        assert match is not None
        left_edge_index, right_edge_index, _right_reversed = match
        edge_indexes_by_face[left_index].append(left_edge_index)
        edge_indexes_by_face[right_index].append(right_edge_index)
        matches_by_pair[(left_index, right_index)] = match

    common_corner_indexes = []
    for edge_indexes in edge_indexes_by_face:
        corner_index = piece_seg_script._quad_common_corner_index_for_adjacent_edges(
            edge_indexes[0],
            edge_indexes[1],
        )
        assert corner_index is not None
        common_corner_indexes.append(corner_index)

    centered = [quad.copy() for quad in reconciled]
    shared_center = np.mean(
        np.asarray(
            [
                centered[face_index][corner_index]
                for face_index, corner_index in enumerate(common_corner_indexes)
            ],
            dtype=np.float32,
        ),
        axis=0,
    )
    for face_index, corner_index in enumerate(common_corner_indexes):
        centered[face_index][corner_index] = shared_center

    fitted = piece_seg_script._fit_projected_cube_quads_from_shared_sides(
        centered,
        face_masks,
        edge_indexes_by_face,
        matches_by_pair,
        common_corner_indexes,
        shared_center,
        roi_shape=(120, 120),
    )
    fitted_with_occlusion = piece_seg_script._fit_projected_cube_quads_from_shared_sides(
        centered,
        occluded_masks,
        edge_indexes_by_face,
        matches_by_pair,
        common_corner_indexes,
        shared_center,
        roi_shape=(120, 120),
    )

    assert all(
        np.allclose(quad, occluded_quad)
        for quad, occluded_quad in zip(fitted, fitted_with_occlusion)
    )


def test_orthogonal_fit_score_prefers_one_fully_covered_face_over_compromise():
    masks = []
    for bounds in ((10, 10, 42, 42), (10, 10, 30, 42), (10, 10, 42, 28)):
        mask = np.zeros((64, 64), dtype=bool)
        x1, y1, x2, y2 = bounds
        mask[y1:y2, x1:x2] = True
        masks.append(mask)

    full_face_quad = np.asarray(
        [[10.0, 10.0], [42.0, 10.0], [42.0, 42.0], [10.0, 42.0]],
        dtype=np.float32,
    )
    compromise_quad = np.asarray(
        [[10.0, 10.0], [33.0, 10.0], [33.0, 33.0], [10.0, 33.0]],
        dtype=np.float32,
    )

    reliable_face_score = piece_seg_script._orthogonal_quad_mask_fit_score(
        [full_face_quad, full_face_quad, full_face_quad],
        masks,
        roi_shape=(64, 64),
    )
    compromise_score = piece_seg_script._orthogonal_quad_mask_fit_score(
        [compromise_quad, compromise_quad, compromise_quad],
        masks,
        roi_shape=(64, 64),
    )

    assert reliable_face_score > compromise_score


def test_outer_face_corner_slides_toward_center_to_reduce_spillover():
    cv2 = piece_seg_script.cv2
    quad = np.asarray(
        [
            [10.0, 10.0],
            [50.0, 10.0],
            [50.0, 50.0],
            [10.0, 50.0],
        ],
        dtype=np.float32,
    )
    mask_u8 = np.zeros((64, 64), dtype=np.uint8)
    clipped = np.asarray(
        [
            [10, 10],
            [50, 10],
            [42, 42],
            [10, 50],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(mask_u8, clipped, 1)

    adjusted = piece_seg_script._slide_outer_quad_corner_toward_center_for_mask_fit(
        quad,
        mask_u8 > 0,
        common_index=0,
        outer_index=2,
        roi_shape=(64, 64),
    )

    assert np.linalg.norm(adjusted[2] - adjusted[0]) < np.linalg.norm(quad[2] - quad[0])
    assert np.allclose(adjusted[0], quad[0])
    assert np.allclose(adjusted[1], quad[1])
    assert np.allclose(adjusted[3], quad[3])
    assert piece_seg_script._quad_fully_covers_mask(
        adjusted,
        mask_u8 > 0,
        roi_shape=(64, 64),
    )


def test_triplet_outer_face_corners_use_same_slide_percentage():
    cv2 = piece_seg_script.cv2
    quads = [
        np.asarray(
            [
                [30.0, 30.0],
                [74.0, 30.0],
                [74.0, 74.0],
                [30.0, 74.0],
            ],
            dtype=np.float32,
        ),
        np.asarray(
            [
                [30.0, 30.0],
                [30.0, 82.0],
                [-22.0, 82.0],
                [-22.0, 30.0],
            ],
            dtype=np.float32,
        ),
        np.asarray(
            [
                [30.0, 30.0],
                [70.0, -10.0],
                [110.0, 30.0],
                [70.0, 70.0],
            ],
            dtype=np.float32,
        ),
    ]
    masks = []
    target_scale = 0.82
    for quad in quads:
        clipped = piece_seg_script._quad_with_outer_corner_slide(
            quad,
            common_index=0,
            outer_index=2,
            scale=target_scale,
        )
        mask_u8 = np.zeros((120, 120), dtype=np.uint8)
        cv2.fillConvexPoly(mask_u8, np.round(clipped).astype(np.int32), 1)
        masks.append(mask_u8 > 0)

    adjusted = piece_seg_script._slide_triplet_outer_quad_corners_with_shared_scale(
        quads,
        masks,
        [0, 0, 0],
        roi_shape=(120, 120),
    )

    slide_ratios = [
        np.linalg.norm(quad[2] - quad[0]) / np.linalg.norm(original[2] - original[0])
        for quad, original in zip(adjusted, quads)
    ]
    assert np.allclose(slide_ratios, slide_ratios[0])
    assert abs(float(slide_ratios[0]) - target_scale) <= 0.02
    for quad, mask in zip(adjusted, masks):
        assert piece_seg_script._quad_fully_covers_mask(
            quad,
            mask,
            roi_shape=(120, 120),
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
    assert args.cluster_merge_ray_distance_divisor == 4.0


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


def test_process_frame_grid_fit_display_is_disabled():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :] = [40, 40, 40]
    args = default_args(
        draw_cube_grid_fit=True,
        segmentation_padding_px=-2,
    )
    original_low_intersection_sticker_lines = piece_seg_script.low_intersection_sticker_lines
    original_draw_low_intersection_lines = piece_seg_script.draw_low_intersection_lines

    def capture_low_intersection_lines(sticker_masks, **kwargs):
        raise AssertionError("cube grid fitting display should be disabled")

    def capture_draw_low_intersection_lines(*args, **kwargs):
        raise AssertionError("cube grid fitting display should be disabled")

    piece_seg_script.low_intersection_sticker_lines = capture_low_intersection_lines
    piece_seg_script.draw_low_intersection_lines = capture_draw_low_intersection_lines
    try:
        output, _ = piece_seg_script.process_frame(
            frame,
            StubBboxDetector(),
            StubStickerModel(),
            args,
        )
    finally:
        piece_seg_script.low_intersection_sticker_lines = original_low_intersection_sticker_lines
        piece_seg_script.draw_low_intersection_lines = original_draw_low_intersection_lines

    assert np.any(np.all(output == piece_seg_script.DEFAULT_STICKER_MASK_COLOR, axis=2))


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


def test_parser_exposes_masks_side_panel_option():
    args = piece_seg_script.build_parser().parse_args(["--show-masks-side-panel"])

    assert args.show_masks_side_panel is True


def test_parser_exposes_roi_tracking_gain_option():
    args = piece_seg_script.build_parser().parse_args(["--roi-tracking-gain", "1.8"])

    assert args.roi_tracking_gain == 1.8


def test_parser_exposes_roi_tracking_padding_option():
    args = piece_seg_script.build_parser().parse_args(["--roi-tracking-padding", "0.25"])

    assert args.roi_tracking_padding == 0.25


def test_parser_exposes_cube_net_output_option():
    args = piece_seg_script.build_parser().parse_args(["--save-net", "runs/net.png"])

    assert str(args.save_net) == "runs/net.png"


def test_parser_exposes_video_output_option():
    args = piece_seg_script.build_parser().parse_args(["--save-video", "runs/out.mp4"])

    assert str(args.save_video) == "runs/out.mp4"


def test_capture_fps_or_default_uses_capture_fps():
    capture = SimpleNamespace(get=lambda prop: 24.0)

    assert piece_seg_script.capture_fps_or_default(capture) == 24.0


def test_capture_fps_or_default_falls_back_for_invalid_fps():
    capture = SimpleNamespace(get=lambda prop: 0.0)

    assert piece_seg_script.capture_fps_or_default(capture) == 30.0


def test_parser_exposes_density_color_classifier_options():
    args = piece_seg_script.build_parser().parse_args(
        [
            "--color-classifier-mode",
            "density-cluster",
            "--cluster-eps",
            "12",
            "--density-cluster-eps-reference-rgb",
            "160",
            "--density-cluster-min-neighbors",
            "70",
            "--sticker-color-min-brightness",
            "50",
            "--sticker-color-max-brightness",
            "240",
            "--sticker-color-max-cluster-distance",
            "14",
            "--cluster-merge-ray-distance-divisor",
            "3",
        ]
    )

    assert args.color_classifier_mode == "density-cluster"
    assert args.density_cluster_eps == 12
    assert args.density_cluster_eps_reference_rgb == 160
    assert args.density_cluster_min_neighbors == 70
    assert args.sticker_color_min_brightness == 50
    assert args.sticker_color_max_brightness == 240
    assert args.sticker_color_max_cluster_distance == 14
    assert args.cluster_merge_ray_distance_divisor == 3


def test_parser_exposes_hsv_density_color_classifier_options():
    args = piece_seg_script.build_parser().parse_args(
        [
            "--color-classifier-mode",
            "hsv-density-cluster",
            "--hsv-density-cluster-eps",
            "18",
            "--hsv-density-cluster-min-neighbors",
            "5",
        ]
    )

    assert args.color_classifier_mode == "hsv-density-cluster"
    assert args.hsv_density_cluster_eps == 18
    assert args.hsv_density_cluster_min_neighbors == 5


def test_parser_exposes_balanced_gmm_color_classifier_options():
    args = piece_seg_script.build_parser().parse_args(
        [
            "--color-classifier-mode",
            "balanced-gmm",
            "--gmm-max-mahalanobis-distance",
            "12",
            "--gmm-regularization",
            "16",
            "--gmm-ray-variance-floor",
            "4096",
            "--gmm-constrain-covariance-to-ray",
            "--gmm-perpendicular-variance-floor",
            "25",
            "--gmm-perpendicular-variance-ceiling",
            "49",
            "--gmm-dense-core-k-neighbors",
            "5",
            "--gmm-dense-core-quantile",
            "0.25",
            "--gmm-color-priors",
            "red=255,0,0;green=0,255,0;blue=0,0,255",
            "--gmm-prior-strength",
            "0.5",
            "--gmm-min-assignment-confidence",
            "0.7",
            "--gmm-assignment-balance",
            "0.75",
            "--gmm-balance-iterations",
            "12",
            "--gmm-max-iterations",
            "20",
            "--gmm-tolerance",
            "0.01",
        ]
    )

    assert args.color_classifier_mode == "balanced-gmm"
    assert args.gmm_max_mahalanobis_distance == 12
    assert args.gmm_regularization == 16
    assert args.gmm_ray_variance_floor == 4096
    assert args.gmm_constrain_covariance_to_ray is True
    assert args.gmm_perpendicular_variance_floor == 25
    assert args.gmm_perpendicular_variance_ceiling == 49
    assert args.gmm_dense_core_k_neighbors == 5
    assert args.gmm_dense_core_quantile == 0.25
    assert np.allclose(
        args.gmm_color_priors,
        np.array(
            [
                [255.0, 0.0, 0.0],
                [0.0, 255.0, 0.0],
                [0.0, 0.0, 255.0],
            ],
            dtype=np.float32,
        ),
    )
    assert args.gmm_prior_strength == 0.5
    assert args.gmm_min_assignment_confidence == 0.7
    assert args.gmm_assignment_balance == 0.75
    assert args.gmm_balance_iterations == 12
    assert args.gmm_max_iterations == 20
    assert args.gmm_tolerance == 0.01


def test_parser_exposes_density_peak_color_classifier_options():
    args = piece_seg_script.build_parser().parse_args(
        [
            "--color-classifier-mode",
            "density-peak",
            "--density-peak-k-neighbors",
            "9",
            "--density-peak-max-fit-samples",
            "1200",
            "--density-peak-min-peak-distance",
            "22",
        ]
    )

    assert args.color_classifier_mode == "density-peak"
    assert args.density_peak_k_neighbors == 9
    assert args.density_peak_max_fit_samples == 1200
    assert args.density_peak_min_peak_distance == 22


def test_parser_exposes_dense_ray_color_classifier_options():
    args = piece_seg_script.build_parser().parse_args(
        [
            "--color-classifier-mode",
            "dense-ray",
            "--dense-ray-k-neighbors",
            "7",
            "--dense-ray-max-fit-samples",
            "900",
            "--dense-ray-min-anchor-distance",
            "24",
            "--dense-ray-max-distance",
            "11",
        ]
    )

    assert args.color_classifier_mode == "dense-ray"
    assert args.dense_ray_k_neighbors == 7
    assert args.dense_ray_max_fit_samples == 900
    assert args.dense_ray_min_anchor_distance == 24
    assert args.dense_ray_max_distance == 11


def test_parser_exposes_balanced_cone_color_classifier_options():
    args = piece_seg_script.build_parser().parse_args(
        [
            "--color-classifier-mode",
            "balanced-cone",
            "--cone-max-angle-radians",
            "0.2",
            "--cone-assignment-balance",
            "0.6",
            "--cone-assignment-sharpness",
            "55",
            "--cone-balance-iterations",
            "8",
            "--cone-max-iterations",
            "12",
        ]
    )

    assert args.color_classifier_mode == "balanced-cone"
    assert args.cone_max_angle_radians == 0.2
    assert args.cone_assignment_balance == 0.6
    assert args.cone_assignment_sharpness == 55
    assert args.cone_balance_iterations == 8
    assert args.cone_max_iterations == 12


def test_parser_rejects_removed_sticker_color_frames_option():
    with pytest.raises(SystemExit):
        piece_seg_script.build_parser().parse_args(["--sticker-color-frames", "12"])


def test_parser_does_not_expose_hsv_grid_plot_mode():
    with pytest.raises(SystemExit):
        piece_seg_script.build_parser().parse_args(["--hsv-plot-mode", "grid"])


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
        density_cluster_eps=20,
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


def test_balanced_gmm_classifier_learns_long_thin_rgb_rays():
    classifier = piece_seg_script.BalancedGMMColorClassifier(
        required_frames=1,
        cluster_count=3,
        min_brightness=1,
        max_mahalanobis_distance=16,
        regularization=9,
        max_iterations=40,
    )
    rgb_samples = []
    for brightness in range(30, 241, 30):
        rgb_samples.append([brightness, 4, 0])
        rgb_samples.append([4, brightness, 0])
        rgb_samples.append([0, 4, brightness])
    bgr_samples = np.asarray(rgb_samples, dtype=np.uint8)[:, ::-1]

    classifier.observe_frame(bgr_samples)

    assert classifier.ready
    assert len(classifier.cluster_samples) == 3
    red_cluster = classifier.classify_bgr(np.array([0, 0, 120], dtype=np.uint8))
    green_cluster = classifier.classify_bgr(np.array([0, 120, 0], dtype=np.uint8))
    blue_cluster = classifier.classify_bgr(np.array([120, 0, 0], dtype=np.uint8))
    assert red_cluster is not None
    assert green_cluster is not None
    assert blue_cluster is not None
    assert len({red_cluster, green_cluster, blue_cluster}) == 3


def test_balanced_gmm_classifier_rejects_low_confidence_assignments():
    classifier = piece_seg_script.BalancedGMMColorClassifier(
        required_frames=1,
        cluster_count=2,
        min_brightness=1,
        max_mahalanobis_distance=10000,
        regularization=1,
        min_assignment_confidence=0.7,
    )
    classifier.centroids = np.array(
        [
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
        ],
        dtype=np.float32,
    )
    classifier.covariances = np.array(
        [
            np.eye(3, dtype=np.float32) * 1000.0,
            np.eye(3, dtype=np.float32) * 1000.0,
        ],
        dtype=np.float32,
    )
    classifier.precisions = np.array(
        [
            np.linalg.inv(classifier.covariances[0]),
            np.linalg.inv(classifier.covariances[1]),
        ],
        dtype=np.float32,
    )
    classifier.cluster_samples = [
        np.array([[100.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 100.0, 0.0]], dtype=np.float32),
    ]

    assert classifier.classify_bgr(np.array([0, 0, 100], dtype=np.uint8)) == 0
    assert classifier.classify_bgr(np.array([0, 50, 50], dtype=np.uint8)) is None


def test_balanced_gmm_classifier_rejects_samples_above_max_brightness():
    classifier = piece_seg_script.BalancedGMMColorClassifier(
        required_frames=1,
        cluster_count=1,
        min_brightness=1,
        max_brightness=120,
        max_mahalanobis_distance=10000,
        regularization=1,
        min_assignment_confidence=0.0,
    )
    classifier.centroids = np.array([[100.0, 0.0, 0.0]], dtype=np.float32)
    classifier.covariances = np.array([np.eye(3, dtype=np.float32) * 1000.0], dtype=np.float32)
    classifier.precisions = np.array([np.linalg.inv(classifier.covariances[0])], dtype=np.float32)
    classifier.cluster_samples = [np.array([[100.0, 0.0, 0.0]], dtype=np.float32)]

    assert classifier.classify_bgr(np.array([0, 0, 100], dtype=np.uint8)) == 0
    assert classifier.classify_bgr(np.array([0, 0, 200], dtype=np.uint8)) is None


def test_balanced_gmm_ray_variance_floor_keeps_components_long():
    rgb_samples = []
    for brightness in range(30, 241, 30):
        rgb_samples.append([brightness, 3, 0])
        rgb_samples.append([3, brightness, 0])
        rgb_samples.append([0, 3, brightness])

    _labels, centroids, _clusters, covariances, _precisions = (
        piece_seg_script.balanced_gmm_rgb_cluster_memberships(
            np.asarray(rgb_samples, dtype=np.uint8),
            target_clusters=3,
            min_brightness=1,
            max_mahalanobis_distance=25,
            regularization=4,
            ray_variance_floor=4096,
            max_iterations=30,
        )
    )

    assert centroids.shape == (3, 3)
    for centroid, covariance in zip(centroids, covariances):
        ray = centroid / np.linalg.norm(centroid)
        assert float(ray @ covariance @ ray) >= 4096 - 1e-3


def test_balanced_gmm_memberships_mark_samples_above_max_brightness_unknown():
    rgb_samples = np.array(
        [
            [40, 0, 0],
            [80, 0, 0],
            [160, 0, 0],
        ],
        dtype=np.uint8,
    )

    labels, centroids, clusters, _covariances, _precisions = (
        piece_seg_script.balanced_gmm_rgb_cluster_memberships(
            rgb_samples,
            target_clusters=1,
            min_brightness=1,
            max_brightness=100,
            max_mahalanobis_distance=10000,
            regularization=4,
            min_assignment_confidence=0.0,
            max_iterations=10,
        )
    )

    assert centroids.shape == (1, 3)
    assert len(clusters) == 1
    assert labels.tolist() == [0, 0, -1]


def test_balanced_gmm_can_constrain_covariance_to_origin_ray():
    covariance = np.array(
        [
            [100.0, 60.0, 20.0],
            [60.0, 80.0, 10.0],
            [20.0, 10.0, 40.0],
        ],
        dtype=np.float32,
    )
    shaped = piece_seg_script._shape_gmm_covariance_for_ray(
        covariance,
        np.array([10.0, 0.0, 0.0], dtype=np.float32),
        ray_variance_floor=400.0,
        constrain_covariance_to_ray=True,
        perpendicular_variance_floor=25.0,
        perpendicular_variance_ceiling=30.0,
    )

    assert np.allclose(shaped, np.diag([400.0, 30.0, 30.0]))


def test_balanced_gmm_memberships_use_constrained_ray_covariances():
    rgb_samples = []
    for brightness in range(40, 221, 45):
        rgb_samples.append([brightness, 2, 0])
        rgb_samples.append([2, brightness, 0])

    _labels, centroids, _clusters, covariances, _precisions = (
        piece_seg_script.balanced_gmm_rgb_cluster_memberships(
            np.asarray(rgb_samples, dtype=np.uint8),
            target_clusters=2,
            min_brightness=1,
            max_mahalanobis_distance=25,
            regularization=4,
            ray_variance_floor=900,
            constrain_covariance_to_ray=True,
            perpendicular_variance_floor=16,
            perpendicular_variance_ceiling=25,
            max_iterations=20,
        )
    )

    assert centroids.shape == (2, 3)
    for centroid, covariance in zip(centroids, covariances):
        ray = centroid / np.linalg.norm(centroid)
        perpendicular = np.eye(3, dtype=np.float32) - np.outer(ray, ray)
        cross_component = perpendicular @ covariance @ ray
        assert np.linalg.norm(cross_component) < 1e-3
        assert float(ray @ covariance @ ray) >= 900 - 1e-3


def test_balanced_gmm_color_priors_lock_component_rays():
    rgb_samples = []
    for brightness in range(40, 221, 45):
        rgb_samples.append([brightness, 3, 0])
        rgb_samples.append([3, brightness, 0])
    rgb_samples.extend([[90, 75, 0], [120, 90, 0], [150, 105, 0]])
    priors = np.array(
        [
            [255.0, 0.0, 0.0],
            [0.0, 255.0, 0.0],
        ],
        dtype=np.float32,
    )

    labels, centroids, _clusters, covariances, _precisions = (
        piece_seg_script.balanced_gmm_rgb_cluster_memberships(
            np.asarray(rgb_samples, dtype=np.uint8),
            target_clusters=2,
            min_brightness=1,
            max_mahalanobis_distance=10000,
            regularization=4,
            ray_variance_floor=900,
            constrain_covariance_to_ray=True,
            perpendicular_variance_floor=16,
            perpendicular_variance_ceiling=25,
            prior_rgb_values=priors,
            prior_strength=1.0,
            max_iterations=30,
        )
    )

    assert centroids.shape == (2, 3)
    assert labels[0] != labels[1]
    assert centroids[0, 0] > 0
    assert np.allclose(centroids[0, 1:], [0.0, 0.0], atol=1e-4)
    assert centroids[1, 1] > 0
    assert np.allclose(centroids[1, [0, 2]], [0.0, 0.0], atol=1e-4)
    for prior, covariance in zip(priors, covariances):
        ray = prior / np.linalg.norm(prior)
        perpendicular = np.eye(3, dtype=np.float32) - np.outer(ray, ray)
        cross_component = perpendicular @ covariance @ ray
        assert np.linalg.norm(cross_component) < 1e-3


def test_gmm_dense_core_filter_drops_sparse_bridge_samples():
    samples = np.array(
        [
            [40.0, 0.0, 0.0],
            [42.0, 0.0, 0.0],
            [44.0, 0.0, 0.0],
            [120.0, 0.0, 0.0],
            [200.0, 0.0, 0.0],
            [202.0, 0.0, 0.0],
            [204.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    indexes = piece_seg_script._gmm_dense_core_local_indexes(
        samples,
        k_neighbors=2,
        drop_quantile=0.25,
    )

    assert 3 not in indexes.tolist()
    assert {0, 1, 2, 4, 5}.issubset(set(indexes.tolist()))


def test_gmm_dense_core_filter_honors_extreme_drop_quantile():
    samples = np.stack(
        [
            np.array([float(index), 0.0, 0.0], dtype=np.float32)
            for index in range(100)
        ],
        axis=0,
    )

    indexes = piece_seg_script._gmm_dense_core_local_indexes(
        samples,
        k_neighbors=3,
        drop_quantile=0.99,
    )

    assert len(indexes) == 1


def test_balanced_gmm_dense_core_prefilter_still_labels_all_usable_samples():
    rgb_samples = np.array(
        [
            [40, 0, 0],
            [42, 0, 0],
            [44, 0, 0],
            [120, 0, 0],
            [200, 0, 0],
            [202, 0, 0],
            [204, 0, 0],
        ],
        dtype=np.uint8,
    )

    labels, centroids, clusters, _covariances, _precisions = (
        piece_seg_script.balanced_gmm_rgb_cluster_memberships(
            rgb_samples,
            target_clusters=2,
            min_brightness=1,
            max_mahalanobis_distance=10000,
            regularization=4,
            dense_core_k_neighbors=2,
            dense_core_quantile=0.25,
            max_iterations=20,
        )
    )

    assert centroids.shape == (2, 3)
    assert len(clusters) == 2
    assert np.all(labels >= 0)


def test_balanced_gmm_dense_core_prefilter_can_mark_eliminated_samples_unknown():
    rgb_samples = np.array(
        [
            [40, 0, 0],
            [42, 0, 0],
            [44, 0, 0],
            [120, 0, 0],
            [200, 0, 0],
            [202, 0, 0],
            [204, 0, 0],
        ],
        dtype=np.uint8,
    )

    labels, centroids, clusters, _covariances, _precisions = (
        piece_seg_script.balanced_gmm_rgb_cluster_memberships(
            rgb_samples,
            target_clusters=2,
            min_brightness=1,
            max_mahalanobis_distance=10000,
            regularization=4,
            dense_core_k_neighbors=2,
            dense_core_quantile=0.25,
            dense_core_outliers_unknown=True,
            max_iterations=20,
        )
    )

    assert centroids.shape == (2, 3)
    assert len(clusters) == 2
    assert labels[3] == -1
    assert labels.tolist().count(-1) >= 1


def test_balanced_gmm_memberships_return_dense_core_mask():
    rgb_samples = np.array(
        [
            [40, 0, 0],
            [42, 0, 0],
            [44, 0, 0],
            [120, 0, 0],
            [200, 0, 0],
            [202, 0, 0],
            [204, 0, 0],
        ],
        dtype=np.uint8,
    )

    _labels, _centroids, _clusters, _covariances, _precisions, dense_core_mask = (
        piece_seg_script.balanced_gmm_rgb_cluster_memberships_with_core_mask(
            rgb_samples,
            target_clusters=2,
            min_brightness=1,
            max_mahalanobis_distance=10000,
            regularization=4,
            dense_core_k_neighbors=2,
            dense_core_quantile=0.25,
            max_iterations=20,
        )
    )

    assert dense_core_mask.dtype == bool
    assert dense_core_mask.shape == (len(rgb_samples),)
    assert dense_core_mask[3] == np.bool_(False)


def test_rgb_plot_clusters_mark_gmm_dense_core_rejections_unknown():
    classifier = piece_seg_script.BalancedGMMColorClassifier(
        required_frames=1,
        cluster_count=2,
        min_brightness=1,
        max_mahalanobis_distance=10000,
        regularization=4,
        dense_core_k_neighbors=2,
        dense_core_quantile=0.25,
        max_iterations=20,
    )
    bgr_samples = np.array(
        [
            [0, 0, 40],
            [0, 0, 42],
            [0, 0, 44],
            [0, 0, 120],
            [0, 0, 200],
            [0, 0, 202],
            [0, 0, 204],
        ],
        dtype=np.uint8,
    )

    cluster_indexes, _centroids = piece_seg_script.rgb_plot_clusters_for_samples(
        bgr_samples,
        classifier,
    )

    assert cluster_indexes is not None
    assert cluster_indexes[3] == -1


def test_balanced_gmm_responsibilities_prevent_mop_up_component():
    log_probabilities = np.array(
        [
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    vanilla = piece_seg_script._balanced_gmm_responsibilities(
        log_probabilities,
        assignment_balance=0.0,
        balance_iterations=20,
    )
    balanced = piece_seg_script._balanced_gmm_responsibilities(
        log_probabilities,
        assignment_balance=1.0,
        balance_iterations=30,
    )

    assert vanilla.sum(axis=0)[0] > 5.0
    assert np.allclose(balanced.sum(axis=0), [2.0, 2.0, 2.0], atol=1e-3)
    assert np.allclose(balanced.sum(axis=1), np.ones(6), atol=1e-6)


def test_density_peak_clustering_splits_dense_modes_joined_by_bridge():
    left = np.array(
        [
            [40, 0, 0],
            [42, 1, 0],
            [44, 0, 1],
            [46, 2, 0],
            [48, 0, 2],
            [50, 1, 1],
        ],
        dtype=np.uint8,
    )
    right = np.array(
        [
            [140, 0, 0],
            [142, 1, 0],
            [144, 0, 1],
            [146, 2, 0],
            [148, 0, 2],
            [150, 1, 1],
        ],
        dtype=np.uint8,
    )
    bridge = np.array([[70, 0, 0], [95, 0, 0], [120, 0, 0]], dtype=np.uint8)
    samples = np.concatenate([left, bridge, right], axis=0)

    labels, centroids, clusters = piece_seg_script.density_peak_rgb_cluster_memberships(
        samples,
        target_clusters=2,
        min_brightness=1,
        k_neighbors=3,
        max_fit_samples=100,
        min_peak_distance=40,
    )

    assert centroids.shape == (2, 3)
    assert len(clusters) == 2
    assert labels[0] != labels[-1]
    assert len(set(labels.tolist())) == 2


def test_density_peak_assignment_uses_basin_samples_not_compact_peaks():
    points = np.array(
        [
            [0, 0, 0],
            [100, 0, 0],
            [102, 0, 0],
        ],
        dtype=np.float32,
    )
    peak_points = np.array(
        [
            [0, 0, 0],
            [100, 0, 0],
        ],
        dtype=np.float32,
    )
    basin_samples = [
        np.array([[0, 0, 0], [102, 0, 0]], dtype=np.float32),
        np.array([[100, 0, 0]], dtype=np.float32),
    ]

    labels = piece_seg_script._assign_samples_to_density_peak_clusters(
        points,
        peak_points,
        basin_samples,
    )

    assert labels.tolist() == [0, 1, 0]


def test_density_peak_classifier_learns_rgb_modes():
    classifier = piece_seg_script.DensityPeakColorClassifier(
        required_frames=1,
        cluster_count=3,
        min_brightness=1,
        k_neighbors=3,
        max_fit_samples=100,
        min_peak_distance=40,
        max_cluster_distance=18,
    )
    rgb_samples = np.array(
        [
            [220, 0, 0],
            [218, 2, 0],
            [216, 0, 2],
            [0, 220, 0],
            [2, 218, 0],
            [0, 216, 2],
            [0, 0, 220],
            [0, 2, 218],
            [2, 0, 216],
        ],
        dtype=np.uint8,
    )

    classifier.observe_frame(rgb_samples[:, ::-1])

    assert classifier.ready
    assert len(classifier.cluster_samples) == 3
    red_cluster = classifier.classify_bgr(np.array([0, 0, 219], dtype=np.uint8))
    green_cluster = classifier.classify_bgr(np.array([0, 219, 0], dtype=np.uint8))
    blue_cluster = classifier.classify_bgr(np.array([219, 0, 0], dtype=np.uint8))
    assert len({red_cluster, green_cluster, blue_cluster}) == 3


def test_dense_ray_anchor_points_choose_densest_separated_points():
    samples = np.array(
        [
            [100, 0, 0],
            [101, 0, 0],
            [102, 0, 0],
            [0, 100, 0],
            [0, 101, 0],
            [0, 102, 0],
            [180, 180, 180],
        ],
        dtype=np.uint8,
    )

    anchors = piece_seg_script.dense_rgb_anchor_points(
        samples,
        target_clusters=2,
        k_neighbors=2,
        min_anchor_distance=40,
    )

    assert anchors.shape == (2, 3)
    assert np.linalg.norm(anchors[0] - anchors[1]) >= 40
    assert not np.any(np.all(anchors == [180, 180, 180], axis=1))


def test_dense_ray_keeps_brightness_tail_on_same_origin_ray():
    samples = np.array(
        [
            [40, 0, 0],
            [80, 0, 0],
            [160, 0, 0],
            [0, 40, 0],
            [0, 80, 0],
            [0, 160, 0],
        ],
        dtype=np.uint8,
    )

    labels, centroids, clusters, ray_directions = piece_seg_script.dense_ray_rgb_cluster_memberships(
        samples,
        target_clusters=2,
        min_brightness=1,
        k_neighbors=1,
        max_fit_samples=100,
        min_anchor_distance=40,
    )

    assert centroids.shape == (2, 3)
    assert len(clusters) == 2
    assert ray_directions.shape == (2, 3)
    assert len(set(labels[:3].tolist())) == 1
    assert len(set(labels[3:].tolist())) == 1
    assert labels[0] != labels[3]


def test_dense_ray_classifier_uses_perpendicular_distance_to_ray():
    classifier = piece_seg_script.DenseRayColorClassifier(
        required_frames=1,
        cluster_count=2,
        min_brightness=1,
        k_neighbors=1,
        max_fit_samples=100,
        min_anchor_distance=40,
        max_ray_distance=5,
    )
    rgb_samples = np.array(
        [
            [50, 0, 0],
            [150, 0, 0],
            [0, 50, 0],
            [0, 150, 0],
        ],
        dtype=np.uint8,
    )

    classifier.observe_frame(rgb_samples[:, ::-1])

    assert classifier.ready
    red_cluster = classifier.classify_bgr(np.array([0, 0, 220], dtype=np.uint8))
    green_cluster = classifier.classify_bgr(np.array([0, 220, 0], dtype=np.uint8))
    assert red_cluster is not None
    assert green_cluster is not None
    assert red_cluster != green_cluster
    assert classifier.classify_bgr(np.array([20, 20, 220], dtype=np.uint8)) is None


def test_balanced_cone_classifier_uses_angular_rgb_cones():
    classifier = piece_seg_script.BalancedConeColorClassifier(
        required_frames=1,
        cluster_count=2,
        min_brightness=1,
        max_angle_radians=0.12,
        assignment_balance=1.0,
        assignment_sharpness=80,
        max_iterations=20,
    )
    rgb_samples = np.array(
        [
            [40, 2, 0],
            [120, 4, 0],
            [220, 6, 0],
            [2, 40, 0],
            [4, 120, 0],
            [6, 220, 0],
        ],
        dtype=np.uint8,
    )

    classifier.observe_frame(rgb_samples[:, ::-1])

    assert classifier.ready
    red_cluster = classifier.classify_bgr(np.array([0, 0, 20], dtype=np.uint8))
    bright_red_cluster = classifier.classify_bgr(np.array([0, 0, 240], dtype=np.uint8))
    green_cluster = classifier.classify_bgr(np.array([0, 240, 0], dtype=np.uint8))
    assert red_cluster is not None
    assert red_cluster == bright_red_cluster
    assert green_cluster is not None
    assert green_cluster != red_cluster
    assert classifier.classify_bgr(np.array([0, 60, 60], dtype=np.uint8)) is None


def test_balanced_cone_plot_clusters_are_raw_rgb_cone_assignments():
    classifier = piece_seg_script.BalancedConeColorClassifier(
        required_frames=1,
        cluster_count=2,
        min_brightness=1,
        max_angle_radians=0.12,
        assignment_balance=1.0,
        assignment_sharpness=80,
        max_iterations=20,
    )
    bgr_samples = np.array(
        [
            [0, 2, 40],
            [0, 4, 120],
            [0, 6, 220],
            [0, 40, 2],
            [0, 120, 4],
            [0, 220, 6],
            [60, 60, 0],
        ],
        dtype=np.uint8,
    )
    classifier.observe_frame(bgr_samples[:6])

    cluster_indexes, cluster_centroids = piece_seg_script.rgb_plot_clusters_for_samples(
        bgr_samples,
        classifier,
    )

    assert cluster_indexes is not None
    assert cluster_centroids is not None
    assert cluster_centroids.shape == (2, 3)
    assert len(set(cluster_indexes[:3].tolist())) == 1
    assert len(set(cluster_indexes[3:6].tolist())) == 1
    assert cluster_indexes[0] != cluster_indexes[3]
    assert cluster_indexes[6] == -1


def test_rgb_html_plot_keeps_original_point_colors_for_cone_clusters(monkeypatch, tmp_path):
    calls = []

    def fake_save_rgb_samples_interactive_plot(*args, **kwargs):
        calls.append((args, kwargs))
        return str(args[1])

    monkeypatch.setattr(
        piece_seg_script,
        "save_rgb_samples_interactive_plot",
        fake_save_rgb_samples_interactive_plot,
    )
    classifier = piece_seg_script.BalancedConeColorClassifier(
        required_frames=1,
        cluster_count=2,
        min_brightness=1,
        max_angle_radians=0.12,
        assignment_balance=1.0,
        assignment_sharpness=80,
        max_iterations=20,
    )
    bgr_samples = np.array(
        [
            [0, 2, 40],
            [0, 4, 120],
            [0, 40, 2],
            [0, 120, 4],
        ],
        dtype=np.uint8,
    )
    classifier.observe_frame(bgr_samples)

    piece_seg_script.save_rgb_html_plots(
        bgr_samples,
        tmp_path / "rgb.html",
        title="RGB",
        normalize_rgb=False,
        args=SimpleNamespace(plotly_point_size=1.0),
        color_classifier=classifier,
    )

    assert len(calls) == 2
    for _args, kwargs in calls:
        assert "color_by_cluster" not in kwargs
        assert kwargs["cluster_indexes"] is not None
        assert kwargs["cluster_centroids"] is not None


def test_balanced_gmm_plot_clusters_fit_from_samples_when_classifier_is_not_ready():
    classifier = piece_seg_script.BalancedGMMColorClassifier(
        required_frames=90,
        cluster_count=2,
        min_brightness=1,
        max_mahalanobis_distance=16,
        regularization=9,
        max_iterations=20,
    )
    bgr_samples = np.array(
        [
            [0, 5, 40],
            [0, 8, 140],
            [0, 10, 220],
            [5, 40, 0],
            [8, 140, 0],
            [10, 220, 0],
        ],
        dtype=np.uint8,
    )

    cluster_indexes, cluster_centroids = piece_seg_script.rgb_plot_clusters_for_samples(
        bgr_samples,
        classifier,
    )

    assert cluster_indexes is not None
    assert cluster_centroids is not None
    assert len(set(cluster_indexes.tolist())) == 2
    assert cluster_centroids.shape == (2, 3)
    assert not classifier.ready


def test_rgb_plot_clusters_fit_from_samples_when_classifier_is_not_ready():
    classifier = piece_seg_script.ColorDensityClassifier(
        required_frames=90,
        cluster_count=2,
        min_brightness=1,
        density_cluster_eps=20,
        density_cluster_min_neighbors=2,
    )
    bgr_samples = np.array(
        [
            [20, 20, 220],
            [22, 18, 210],
            [18, 22, 200],
            [20, 220, 20],
            [18, 210, 22],
            [22, 200, 18],
        ],
        dtype=np.uint8,
    )

    cluster_indexes, cluster_centroids = piece_seg_script.rgb_plot_clusters_for_samples(
        bgr_samples,
        classifier,
    )

    assert cluster_indexes is not None
    assert cluster_centroids is not None
    assert len(set(cluster_indexes.tolist())) == 2
    assert cluster_centroids.shape == (2, 3)
    assert not classifier.ready


def test_rgb_plot_clusters_show_density_outliers_as_unknown():
    classifier = piece_seg_script.ColorDensityClassifier(
        required_frames=90,
        cluster_count=2,
        min_brightness=1,
        density_cluster_eps=1,
        density_cluster_min_neighbors=2,
        max_cluster_distance=50,
    )
    bgr_samples = np.array(
        [
            [0, 0, 10],
            [0, 0, 10],
            [0, 0, 30],
        ],
        dtype=np.uint8,
    )

    cluster_indexes, cluster_centroids = piece_seg_script.rgb_plot_clusters_for_samples(
        bgr_samples,
        classifier,
    )

    assert cluster_indexes is not None
    assert cluster_centroids is not None
    assert cluster_indexes.tolist() == [0, 0, -1]


def test_rgb_plot_clusters_show_all_unknown_when_no_density_clusters_survive():
    classifier = piece_seg_script.ColorDensityClassifier(
        required_frames=90,
        cluster_count=2,
        min_brightness=1,
        density_cluster_eps=1,
        density_cluster_min_neighbors=2,
    )
    bgr_samples = np.array(
        [
            [0, 0, 10],
            [0, 0, 30],
            [0, 0, 50],
        ],
        dtype=np.uint8,
    )

    cluster_indexes, cluster_centroids = piece_seg_script.rgb_plot_clusters_for_samples(
        bgr_samples,
        classifier,
    )

    assert cluster_indexes is not None
    assert cluster_centroids is not None
    assert cluster_indexes.tolist() == [-1, -1, -1]
    assert cluster_centroids.shape == (0, 3)


def test_cluster_merge_weights_origin_ray_component_less_than_other_components():
    off_ray_neighbor = np.array(
        [[100.0, 28.0, 0.0], [102.0, 30.0, 0.0]],
        dtype=np.float32,
    )
    source = np.array(
        [[98.0, 20.0, 0.0], [102.0, 20.0, 0.0]],
        dtype=np.float32,
    )
    ray_parallel_neighbor = np.array(
        [[112.0, 22.4, 0.0], [114.0, 22.8, 0.0]],
        dtype=np.float32,
    )

    nearest_index = piece_seg_script.nearest_cluster_index(
        [off_ray_neighbor, source, ray_parallel_neighbor],
        source_index=1,
        ray_distance_divisor=4.0,
    )

    assert nearest_index == 2


def test_cluster_merge_divisor_one_uses_unweighted_nearest_neighbor():
    off_ray_neighbor = np.array(
        [[100.0, 28.0, 0.0], [102.0, 30.0, 0.0]],
        dtype=np.float32,
    )
    source = np.array(
        [[98.0, 20.0, 0.0], [102.0, 20.0, 0.0]],
        dtype=np.float32,
    )
    ray_parallel_neighbor = np.array(
        [[112.0, 22.4, 0.0], [114.0, 22.8, 0.0]],
        dtype=np.float32,
    )

    nearest_index = piece_seg_script.nearest_cluster_index(
        [off_ray_neighbor, source, ray_parallel_neighbor],
        source_index=1,
        ray_distance_divisor=1.0,
    )

    assert nearest_index == 0


def test_cluster_merging_compares_against_original_accepted_clusters():
    accepted_a = np.array(
        [[100.0, 0.0, 0.0], [101.0, 0.0, 0.0], [102.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    accepted_b = np.array(
        [[200.0, 0.0, 0.0], [201.0, 0.0, 0.0], [202.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    subcluster_c = np.array([[130.0, 0.0, 0.0]], dtype=np.float32)
    subcluster_d = np.array([[160.0, 0.0, 0.0]], dtype=np.float32)

    clusters = piece_seg_script.merge_smallest_clusters_to_nearest_neighbor(
        [accepted_a, accepted_b, subcluster_c, subcluster_d],
        target_count=2,
        ray_distance_divisor=1.0,
    )

    cluster_with_a = next(cluster for cluster in clusters if 100.0 in cluster[:, 0])
    cluster_with_b = next(cluster for cluster in clusters if 200.0 in cluster[:, 0])
    assert 130.0 in cluster_with_a[:, 0]
    assert 160.0 not in cluster_with_a[:, 0]
    assert 160.0 in cluster_with_b[:, 0]


def test_density_cluster_eps_scales_with_rgb_distance_from_origin():
    rgb_samples = np.array(
        [
            [150, 30, 0],
            [170, 30, 0],
            [40, 8, 0],
            [60, 8, 0],
        ],
        dtype=np.uint8,
    )

    centroids, clusters = piece_seg_script.density_rgb_clusters(
        rgb_samples,
        target_clusters=6,
        min_brightness=1,
        eps=35.0,
        eps_reference_rgb=150.0,
        min_points=2,
        merge_ray_distance_divisor=4.0,
    )

    assert len(clusters) == 1
    assert centroids.shape == (1, 3)
    assert np.allclose(centroids[0], [160.0, 30.0, 0.0])


def test_density_cluster_eps_uses_reference_rgb_as_anchor():
    rgb_samples = np.array(
        [
            [150, 30, 0],
            [170, 30, 0],
        ],
        dtype=np.uint8,
    )

    centroids, clusters = piece_seg_script.density_rgb_clusters(
        rgb_samples,
        target_clusters=6,
        min_brightness=1,
        eps=15.0,
        eps_reference_rgb=150.0,
        min_points=2,
        merge_ray_distance_divisor=4.0,
    )

    assert len(clusters) == 0
    assert centroids.shape == (0, 3)

    centroids, clusters = piece_seg_script.density_rgb_clusters(
        rgb_samples,
        target_clusters=6,
        min_brightness=1,
        eps=15.0,
        eps_reference_rgb=60.0,
        min_points=2,
        merge_ray_distance_divisor=4.0,
    )

    assert len(clusters) == 1
    assert centroids.shape == (1, 3)
    assert np.allclose(centroids[0], [160.0, 30.0, 0.0])


def test_density_cluster_eps_has_nonzero_origin_intercept():
    labels = piece_seg_script.density_bin_labels(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.05, 0.0, 0.0],
                [0.2, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        eps=20.0,
        eps_reference_rgb=150.0,
        min_points=2,
    )

    assert labels.tolist() == [0, 0, -1]


def test_fixed_hsv_density_labels_wrap_hue_at_red_boundary():
    hsv_samples = np.array(
        [
            [1.0, 220.0, 220.0],
            [2.0, 218.0, 222.0],
            [179.0, 219.0, 221.0],
            [90.0, 220.0, 220.0],
        ],
        dtype=np.float32,
    )

    labels = piece_seg_script.density_fixed_hsv_labels(
        hsv_samples,
        eps=6,
        min_points=3,
    )

    assert labels[:3].tolist() == [0, 0, 0]
    assert labels[3] == -1


def test_hsv_density_classifier_learns_hsv_clusters_from_bgr_samples():
    classifier = piece_seg_script.HSVColorDensityClassifier(
        required_frames=1,
        cluster_count=2,
        min_brightness=1,
        density_cluster_eps=8,
        density_cluster_min_neighbors=3,
        max_cluster_distance=8,
    )
    hsv_samples = np.array(
        [
            [1, 220, 220],
            [2, 218, 222],
            [179, 219, 221],
            [60, 220, 220],
            [61, 218, 222],
            [59, 219, 221],
        ],
        dtype=np.uint8,
    )
    bgr_samples = piece_seg_script.cv2.cvtColor(
        hsv_samples.reshape(-1, 1, 3),
        piece_seg_script.cv2.COLOR_HSV2BGR,
    ).reshape(-1, 3)

    classifier.observe_frame(bgr_samples)

    assert classifier.ready
    red_cluster = classifier.classify_bgr(bgr_samples[0])
    green_cluster = classifier.classify_bgr(bgr_samples[-1])
    assert red_cluster is not None
    assert green_cluster is not None
    assert red_cluster != green_cluster


def test_cluster_merge_still_uses_nearest_neighbor_when_ray_distance_is_too_far():
    nearby_off_ray = np.array(
        [[100.0, 28.0, 0.0], [102.0, 30.0, 0.0]],
        dtype=np.float32,
    )
    source = np.array(
        [[98.0, 20.0, 0.0], [102.0, 20.0, 0.0]],
        dtype=np.float32,
    )
    far_ray_parallel = np.array(
        [[180.0, 36.0, 0.0], [190.0, 38.0, 0.0]],
        dtype=np.float32,
    )

    nearest_index = piece_seg_script.nearest_cluster_index(
        [nearby_off_ray, source, far_ray_parallel],
        source_index=1,
    )

    assert nearest_index == 0


def test_color_classifier_uses_raw_rgb_distance_not_brightness_normalization():
    classifier = piece_seg_script.ColorDensityClassifier(
        required_frames=1,
        cluster_count=1,
        min_brightness=1,
        density_cluster_eps=20,
        density_cluster_min_neighbors=1,
    )
    classifier.centroids = np.array([[220.0, 0.0, 0.0]], dtype=np.float32)
    classifier.cluster_samples = [
        np.array([[220.0, 0.0, 0.0]], dtype=np.float32),
    ]

    assert classifier.classify_bgr(np.array([0, 0, 220], dtype=np.uint8)) == 0
    assert classifier.classify_bgr(np.array([0, 0, 40], dtype=np.uint8)) is None


def test_color_classifier_uses_dedicated_max_cluster_distance_for_classification():
    classifier = piece_seg_script.ColorDensityClassifier(
        required_frames=1,
        cluster_count=1,
        min_brightness=1,
        density_cluster_eps=20,
        density_cluster_min_neighbors=1,
        max_cluster_distance=5.0,
    )
    classifier.centroids = np.array([[220.0, 0.0, 0.0]], dtype=np.float32)
    classifier.cluster_samples = [
        np.array([[220.0, 0.0, 0.0]], dtype=np.float32),
    ]

    assert classifier.classify_bgr(np.array([0, 0, 224], dtype=np.uint8)) == 0
    assert classifier.classify_bgr(np.array([0, 0, 226], dtype=np.uint8)) is None


def test_normalized_rgb_html_plot_path_uses_rgb_html_stem():
    path = piece_seg_script.normalized_rgb_html_plot_path(Path("runs/rgb.html"))

    assert path == Path("runs/rgb_normalized.html")


def test_normalized_rgb_html_plot_path_adds_html_suffix():
    path = piece_seg_script.normalized_rgb_html_plot_path(Path("runs/rgb"))

    assert path == Path("runs/rgb_normalized.html")
