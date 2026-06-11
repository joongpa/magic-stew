import numpy as np

from live_recognition.piece_graph import (
    PieceMask,
    StickerNode,
    _legal_mask_adjacencies,
    build_piece_graph,
    build_sticker_adjacency_graph,
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


def test_graph_connects_node_to_closest_node_on_neighboring_piece():
    center = node(0, 0, "center", (10.0, 10.0), (10, 12, 10, 12))
    near_edge_sticker = node(1, 1, "edge", (12.0, 10.0), (10, 12, 12, 14))
    far_edge_sticker = node(2, 1, "edge", (18.0, 10.0), (10, 12, 18, 20))

    edges = _legal_mask_adjacencies(
        [center, near_edge_sticker, far_edge_sticker],
        adjacency_radius_px=8,
    )

    assert edges == [(0, 1)]


def test_graph_connects_closest_legal_neighbors_without_mask_contact():
    center = node(0, 0, "center", (10.5, 10.5), (10, 12, 10, 12))
    near_edge = node(1, 1, "edge", (15.5, 10.5), (10, 12, 15, 17))
    far_edge = node(2, 1, "edge", (22.5, 10.5), (10, 12, 21, 24))
    illegal_corner = node(3, 3, "corner", (13.5, 10.5), (10, 12, 13, 15))

    edges = _legal_mask_adjacencies(
        [center, near_edge, far_edge, illegal_corner],
        adjacency_radius_px=4,
    )

    assert (0, 1) in edges
    assert (0, 2) not in edges
    assert (0, 3) not in edges


def test_graph_does_not_connect_legal_neighbors_beyond_distance_threshold():
    center = node(0, 0, "center", (2.5, 2.5), (2, 4, 2, 4))
    edge = node(1, 1, "edge", (20.5, 2.5), (2, 4, 20, 22))

    edges = _legal_mask_adjacencies([center, edge], adjacency_radius_px=4)

    assert edges == []


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
    assert graph.edges == [(0, 2)]
