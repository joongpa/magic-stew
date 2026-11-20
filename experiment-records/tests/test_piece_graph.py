import cv2
import numpy as np

from live_recognition.piece_graph import (
    PieceMask,
    StickerNode,
    _legal_mask_adjacencies,
    build_piece_graph,
    build_sticker_adjacency_graph,
    draw_piece_graph,
    sticker_shape_features,
    sticker_shape_index,
    sticker_shape_similarity,
)


def node(node_id, piece_id, piece_type, centroid_xy, mask_slice):
    mask = np.zeros((24, 24), dtype=np.uint8)
    y1, y2, x1, x2 = mask_slice
    mask[y1:y2, x1:x2] = 255
    return StickerNode(
        node_id=node_id,
        piece_id=piece_id,
        piece_type=piece_type,
        confidence=1.0,
        centroid_xy=centroid_xy,
        mask=mask,
    )


def test_graph_connects_all_adjacent_stickers():
    center = node(0, 0, "center", (10.0, 10.0), (10, 12, 10, 12))
    near_edge_sticker = node(1, 1, "edge", (12.0, 10.0), (10, 12, 12, 14))
    far_edge_sticker = node(2, 1, "edge", (18.0, 10.0), (10, 12, 18, 20))

    edges = _legal_mask_adjacencies(
        [center, near_edge_sticker, far_edge_sticker],
        adjacency_radius_px=8,
    )

    assert edges == [(0, 1), (0, 2), (1, 2)]


def test_graph_connects_adjacent_stickers_regardless_of_piece_type():
    center = node(0, 0, "center", (10.5, 10.5), (10, 12, 10, 12))
    near_edge = node(1, 1, "edge", (15.5, 10.5), (10, 12, 15, 17))
    far_edge = node(2, 1, "edge", (22.5, 10.5), (10, 12, 21, 24))
    corner = node(3, 3, "corner", (13.5, 10.5), (10, 12, 13, 15))

    edges = _legal_mask_adjacencies(
        [center, near_edge, far_edge, corner],
        adjacency_radius_px=4,
    )

    assert (0, 1) in edges
    assert (0, 2) not in edges
    assert (0, 3) in edges
    assert (1, 3) in edges


def test_graph_does_not_connect_legal_neighbors_beyond_distance_threshold():
    center = node(0, 0, "center", (2.5, 2.5), (2, 4, 2, 4))
    edge = node(1, 1, "edge", (20.5, 2.5), (2, 4, 20, 22))

    edges = _legal_mask_adjacencies([center, edge], adjacency_radius_px=4)

    assert edges == []


def test_sticker_graph_consumes_adjacency_radius():
    center = node(0, 0, "center", (5.0, 5.0), (4, 6, 4, 6))
    edge = node(1, 1, "edge", (9.0, 5.0), (4, 6, 9, 11))

    small_radius_graph = build_sticker_adjacency_graph(
        [center, edge],
        adjacency_radius_px=1,
    )
    large_radius_graph = build_sticker_adjacency_graph(
        [center, edge],
        adjacency_radius_px=4,
    )

    assert small_radius_graph.edges == []
    assert large_radius_graph.edges == [(0, 1)]


def test_graph_uses_unpadded_mask_geometry_and_padded_sampling_mask_for_color():
    frame = np.zeros((30, 30, 3), dtype=np.uint8)
    frame[10:20, 10:20] = [0, 0, 220]
    frame[10:20, 10:12] = [220, 0, 0]

    roi_mask = np.zeros((30, 30), dtype=bool)
    roi_mask[10:20, 10:20] = True
    sample_roi_mask = np.zeros((30, 30), dtype=bool)
    sample_roi_mask[10:20, 10:12] = True
    full_mask = roi_mask.astype(np.uint8) * 255

    graph = build_piece_graph(
        frame,
        [
            PieceMask(
                piece_id=0,
                class_id=0,
                piece_type="center",
                confidence=1.0,
                mask=full_mask,
                roi_mask=roi_mask,
                roi_xyxy=(0, 0, 30, 30),
                sample_roi_mask=sample_roi_mask,
            )
        ],
        min_component_area_px=1,
    )

    assert len(graph.nodes) == 1
    assert graph.nodes[0].centroid_xy == (14.5, 14.5)
    assert graph.nodes[0].mask.sum() == full_mask.sum()
    assert graph.nodes[0].mean_bgr == (220, 0, 0)


def test_sticker_graph_groups_adjacent_same_type_nodes_as_one_piece():
    first_edge_sticker = node(0, 10, "edge", (10.0, 10.0), (10, 12, 10, 12))
    second_edge_sticker = node(1, 11, "edge", (13.0, 10.0), (10, 12, 13, 15))
    center = node(2, 12, "center", (7.0, 10.0), (10, 12, 7, 9))

    graph = build_sticker_adjacency_graph(
        [first_edge_sticker, second_edge_sticker, center],
        adjacency_radius_px=2,
    )

    edge_piece_ids = {
        graph.nodes[0].piece_id,
        graph.nodes[1].piece_id,
    }
    assert len(edge_piece_ids) == 1
    assert graph.nodes[2].piece_id not in edge_piece_ids
    assert graph.edges == [(0, 1), (0, 2)]


def test_draw_piece_graph_can_overlay_sticker_node_masks_without_piece_masks():
    frame = np.zeros((24, 24, 3), dtype=np.uint8)
    sticker = node(0, 0, "center", (10.0, 10.0), (10, 12, 10, 12))

    output = draw_piece_graph(frame, build_sticker_adjacency_graph([sticker]))

    assert np.any(output[10:12, 10:12] != frame[10:12, 10:12])


def test_sticker_shape_similarity_scores_matching_shapes_highly():
    left = rotated_rect_mask((80, 80), (40, 40), (32, 18), 22)
    right = rotated_rect_mask((80, 80), (42, 38), (30, 17), 24)

    assert sticker_shape_similarity(left, right) > 0.90


def test_sticker_shape_similarity_penalizes_different_face_geometry():
    left = rotated_rect_mask((90, 90), (38, 42), (32, 18), 20)
    different_axes = rotated_rect_mask((90, 90), (48, 42), (32, 18), -35)
    different_aspect = rotated_rect_mask((90, 90), (48, 42), (38, 10), 20)

    assert sticker_shape_similarity(left, different_axes) < 0.80
    assert sticker_shape_similarity(left, different_aspect) < 0.80


def test_sticker_shape_features_returns_none_for_empty_mask():
    assert sticker_shape_features(np.zeros((20, 20), dtype=np.uint8)) is None


def test_sticker_shape_index_tracks_absolute_axis_orientation():
    left = rotated_rect_mask((80, 80), (40, 40), (34, 12), 30)
    right = rotated_rect_mask((80, 80), (42, 38), (30, 10), 32)

    left_index = sticker_shape_index(left)
    right_index = sticker_shape_index(right)

    assert left_index is not None
    assert right_index is not None
    axis_dot = np.dot(left_index.long_axis_xy, right_index.long_axis_xy)
    assert axis_dot > 0.99
    assert 25 <= left_index.angle_degrees <= 35


def test_sticker_shape_index_compactness_runs_from_line_like_to_square_like():
    line_like = rotated_rect_mask((80, 80), (40, 40), (42, 4), 15)
    rectangle = rotated_rect_mask((80, 80), (40, 40), (32, 18), 15)
    square_like = rotated_rect_mask((80, 80), (40, 40), (24, 24), 15)

    line_index = sticker_shape_index(line_like)
    rectangle_index = sticker_shape_index(rectangle)
    square_index = sticker_shape_index(square_like)

    assert line_index is not None
    assert rectangle_index is not None
    assert square_index is not None
    assert line_index.compactness < rectangle_index.compactness < square_index.compactness
    assert square_index.compactness > 0.95


def test_sticker_shape_index_returns_none_for_empty_mask():
    assert sticker_shape_index(np.zeros((20, 20), dtype=np.uint8)) is None


def rotated_rect_mask(shape, center, size, angle_degrees):
    mask = np.zeros(shape, dtype=np.uint8)
    box = cv2.boxPoints((center, size, angle_degrees))
    cv2.fillConvexPoly(mask, np.round(box).astype(np.int32), 255)
    return mask
