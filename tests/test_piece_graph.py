import numpy as np

from live_recognition.piece_graph import StickerNode, _legal_mask_adjacencies


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
