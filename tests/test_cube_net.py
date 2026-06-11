import numpy as np

from live_recognition.cube_net import (
    CubeColorScheme,
    CubeNetState,
    build_observed_cube_net_from_graph,
    build_cube_net_from_graph,
    draw_cube_net,
)
from live_recognition.piece_graph import PieceGraph, StickerNode


def node(node_id, piece_id, piece_type, xy, bgr):
    mask = np.zeros((80, 80), dtype=np.uint8)
    x = int(xy[0])
    y = int(xy[1])
    mask[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] = 255
    return StickerNode(
        node_id=node_id,
        piece_id=piece_id,
        piece_type=piece_type,
        confidence=0.9,
        centroid_xy=xy,
        mask=mask,
        mean_bgr=bgr,
    )


def test_net_places_observed_centers():
    scheme = CubeColorScheme()
    graph = PieceGraph(
        nodes=[node(0, 0, "center", (40, 40), scheme.palette_bgr["green"])],
        edges=[],
    )

    state = build_cube_net_from_graph(
        graph,
        scheme=scheme,
        classified_colors_by_node_id={0: "green"},
    )

    assert state.cells[("F", 1, 1)].color == "green"


def test_net_places_edge_only_when_piece_connects_to_two_centers():
    scheme = CubeColorScheme()
    graph = PieceGraph(
        nodes=[
            node(0, 0, "center", (40, 40), scheme.palette_bgr["green"]),
            node(1, 1, "center", (60, 40), scheme.palette_bgr["red"]),
            node(2, 2, "edge", (50, 40), scheme.palette_bgr["green"]),
            node(3, 2, "edge", (54, 40), scheme.palette_bgr["red"]),
        ],
        edges=[(0, 2), (1, 3)],
    )

    state = build_cube_net_from_graph(
        graph,
        scheme=scheme,
        classified_colors_by_node_id={0: "green", 1: "red", 2: "green", 3: "red"},
    )

    assert state.cells[("F", 1, 2)].color == "green"
    assert state.cells[("R", 1, 0)].color == "red"


def test_net_does_not_place_edge_with_one_center_connection():
    scheme = CubeColorScheme()
    graph = PieceGraph(
        nodes=[
            node(0, 0, "center", (40, 40), scheme.palette_bgr["green"]),
            node(1, 1, "edge", (50, 40), scheme.palette_bgr["green"]),
            node(2, 1, "edge", (54, 40), scheme.palette_bgr["red"]),
        ],
        edges=[(0, 1)],
    )

    state = build_cube_net_from_graph(
        graph,
        scheme=scheme,
        classified_colors_by_node_id={0: "green", 1: "green", 2: "red"},
    )

    assert ("F", 1, 2) not in state.cells
    assert state.cells[("F", 1, 1)].color == "green"


def test_net_can_use_explicit_node_colors_and_face_mapping():
    learned_bgr = (7, 8, 240)
    graph = PieceGraph(
        nodes=[node(0, 0, "center", (40, 40), (0, 0, 0))],
        edges=[],
    )

    state = build_cube_net_from_graph(
        graph,
        classified_colors_by_node_id={0: "cluster-1"},
        classified_bgr_by_color={"cluster-1": learned_bgr},
        face_by_color={"cluster-1": "R"},
    )

    assert state.cells[("R", 1, 1)].color == "cluster-1"
    assert state.cells[("R", 1, 1)].bgr == learned_bgr


def test_net_state_replace_with_uses_latest_values():
    first = CubeNetState()
    first.add_cell("F", 1, 1, "green", "observed", 0.5)
    second = CubeNetState()
    second.add_cell("R", 1, 1, "red", "observed", 0.9)

    first.replace_with(second)

    assert ("F", 1, 1) not in first.cells
    assert first.cells[("R", 1, 1)].color == "red"


def test_draw_cube_net_writes_nonblank_image():
    scheme = CubeColorScheme()
    graph = PieceGraph(
        nodes=[node(0, 0, "center", (40, 40), scheme.palette_bgr["green"])],
        edges=[],
    )
    state = build_cube_net_from_graph(graph, scheme=scheme)

    image = draw_cube_net(state, scheme=scheme)

    assert image.ndim == 3
    assert image.shape[2] == 3
    assert image.mean() < 245


def test_observed_net_uses_raw_bgr():
    raw_bgr = (12, 34, 56)
    graph = PieceGraph(
        nodes=[node(0, 0, "center", (40, 40), raw_bgr)],
        edges=[],
    )

    state = build_observed_cube_net_from_graph(graph)

    assert state.cells[("F", 1, 1)].color == "node-0"
    assert state.cells[("F", 1, 1)].bgr == raw_bgr


def test_observed_net_places_corner_from_adjacent_edge_face():
    graph = PieceGraph(
        nodes=[
            node(0, 0, "center", (40, 40), (0, 200, 0)),
            node(1, 1, "edge", (40, 25), (0, 180, 0)),
            node(2, 2, "corner", (25, 25), (0, 160, 0)),
        ],
        edges=[(0, 1), (1, 2)],
    )

    state = build_observed_cube_net_from_graph(graph)

    assert state.cells[("F", 0, 1)].bgr == (0, 180, 0)
    assert state.cells[("F", 0, 0)].bgr == (0, 160, 0)
