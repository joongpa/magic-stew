from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from live_recognition.piece_graph import PieceGraph, StickerNode
from live_recognition.types import Face

CellSource = Literal["observed"]
ColorName = str


NET_FACE_LAYOUT: dict[Face, tuple[int, int]] = {
    "U": (3, 0),
    "L": (0, 3),
    "F": (3, 3),
    "R": (6, 3),
    "B": (9, 3),
    "D": (3, 6),
}


@dataclass(frozen=True)
class CubeColorScheme:
    face_colors: dict[Face, ColorName] = field(
        default_factory=lambda: {
            "U": "white",
            "D": "yellow",
            "F": "green",
            "B": "blue",
            "R": "red",
            "L": "orange",
        }
    )
    palette_bgr: dict[ColorName, tuple[int, int, int]] = field(
        default_factory=lambda: {
            "white": (235, 235, 235),
            "yellow": (0, 225, 225),
            "green": (35, 170, 35),
            "blue": (210, 75, 20),
            "red": (35, 35, 210),
            "orange": (0, 125, 235),
        }
    )

    @property
    def color_to_face(self) -> dict[ColorName, Face]:
        return {color: face for face, color in self.face_colors.items()}

@dataclass(frozen=True)
class NetCell:
    color: ColorName
    source: CellSource
    confidence: float
    bgr: tuple[int, int, int] | None = None


@dataclass
class CubeNetState:
    cells: dict[tuple[Face, int, int], NetCell] = field(default_factory=dict)

    def replace_with(self, other: "CubeNetState") -> None:
        self.cells = dict(other.cells)

    def add_cell(
        self,
        face: Face,
        row: int,
        col: int,
        color: ColorName,
        source: CellSource,
        confidence: float,
        bgr: tuple[int, int, int] | None = None,
    ) -> None:
        if 0 <= row <= 2 and 0 <= col <= 2:
            self.cells[(face, row, col)] = NetCell(
                color=color,
                source=source,
                confidence=confidence,
                bgr=bgr,
            )


def build_cube_net_from_graph(
    graph: PieceGraph,
    *,
    scheme: CubeColorScheme | None = None,
    classified_colors_by_node_id: dict[int, ColorName] | None = None,
    classified_bgr_by_color: dict[ColorName, tuple[int, int, int]] | None = None,
    face_by_color: dict[ColorName, Face] | None = None,
) -> CubeNetState:
    scheme = scheme or CubeColorScheme()
    color_by_node_id = _node_colors(graph, scheme, classified_colors_by_node_id)
    color_to_face = _color_to_face(graph, color_by_node_id, scheme, face_by_color)
    nodes_by_id = {node.node_id: node for node in graph.nodes}
    centers_by_id = {
        node.node_id: node for node in graph.nodes if node.piece_type == "center"
    }

    state = CubeNetState()
    center_face_by_id: dict[int, Face] = {}
    for center_id, center in centers_by_id.items():
        color = color_by_node_id.get(center_id)
        if color is None:
            continue
        face = color_to_face.get(color)
        if face is None:
            continue
        center_face_by_id[center_id] = face
        state.add_cell(
            face,
            1,
            1,
            color,
            "observed",
            center.confidence,
            bgr=_bgr_for_color(color, classified_bgr_by_color, scheme),
        )

    center_edges = _center_edge_connections(graph, nodes_by_id)
    edge_nodes_by_piece = _edge_nodes_by_piece(graph.nodes)
    for edge_nodes in edge_nodes_by_piece.values():
        connected = []
        for edge_node in edge_nodes:
            center_ids = [
                center_id
                for center_id in center_edges.get(edge_node.node_id, [])
                if center_id in center_face_by_id
            ]
            if len(center_ids) == 1:
                connected.append((edge_node, center_ids[0]))
        center_ids = {center_id for _, center_id in connected}
        if len(connected) != 2 or len(center_ids) != 2:
            continue
        for edge_node, center_id in connected:
            color = color_by_node_id.get(edge_node.node_id)
            if color is None:
                continue
            center = centers_by_id[center_id]
            face = center_face_by_id[center_id]
            row, col = _edge_cell_for_center(center, edge_node)
            state.add_cell(
                face,
                row,
                col,
                color,
                "observed",
                edge_node.confidence,
                bgr=_bgr_for_color(color, classified_bgr_by_color, scheme),
            )
    return state


def build_observed_cube_net_from_graph(graph: PieceGraph) -> CubeNetState:
    """Construct a view-relative net from observed sticker BGR values only."""
    nodes_by_id = {node.node_id: node for node in graph.nodes}
    centers = [node for node in graph.nodes if node.piece_type == "center"]
    ordered_centers, face_order = _view_relative_center_order_and_faces(centers)
    center_face_by_id = {
        center.node_id: face for center, face in zip(ordered_centers, face_order)
    }
    centers_by_id = {center.node_id: center for center in centers}
    center_by_face = {
        center_face_by_id[center.node_id]: center
        for center in centers
        if center.node_id in center_face_by_id
    }

    state = CubeNetState()
    for center in centers:
        face = center_face_by_id.get(center.node_id)
        if face is None:
            continue
        _add_observed_node_cell(state, center, face, 1, 1)

    center_edges = _center_edge_connections(graph, nodes_by_id)
    edge_face_by_node_id: dict[int, Face] = {}
    for edge_id, center_ids in center_edges.items():
        known_center_ids = [
            center_id for center_id in center_ids if center_id in center_face_by_id
        ]
        if len(known_center_ids) != 1:
            continue
        edge = nodes_by_id[edge_id]
        center = centers_by_id[known_center_ids[0]]
        face = center_face_by_id[known_center_ids[0]]
        edge_face_by_node_id[edge_id] = face
        row, col = _edge_cell_for_center(center, edge)
        _add_observed_node_cell(state, edge, face, row, col)

    corner_edges = _corner_edge_connections(graph, nodes_by_id)
    for corner_id, edge_ids in corner_edges.items():
        corner = nodes_by_id[corner_id]
        face = _nearest_known_edge_face(corner, edge_ids, nodes_by_id, edge_face_by_node_id)
        if face is None or face not in center_by_face:
            continue
        row, col = _corner_cell_for_center(center_by_face[face], corner)
        _add_observed_node_cell(state, corner, face, row, col)

    return state


def draw_cube_net(
    state: CubeNetState,
    *,
    scheme: CubeColorScheme | None = None,
    cell_size: int = 42,
    margin: int = 18,
) -> NDArray[np.uint8]:
    scheme = scheme or CubeColorScheme()
    width = 12 * cell_size + 2 * margin
    height = 9 * cell_size + 2 * margin
    output = np.full((height, width, 3), 245, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for face, (face_col, face_row) in NET_FACE_LAYOUT.items():
        for row in range(3):
            for col in range(3):
                key = (face, row, col)
                cell = state.cells.get(key)
                color = _cell_bgr(cell, scheme) if cell is not None else (224, 224, 224)
                x1 = margin + (face_col + col) * cell_size
                y1 = margin + (face_row + row) * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                cv2.rectangle(output, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(output, (x1, y1), (x2, y2), (35, 35, 35), 1)
        label_x = margin + face_col * cell_size + 4
        label_y = margin + face_row * cell_size - 5
        cv2.putText(output, face, (label_x, label_y), font, 0.45, (45, 45, 45), 1, cv2.LINE_AA)
    return output


def _node_colors(
    graph: PieceGraph,
    scheme: CubeColorScheme,
    classified_colors_by_node_id: dict[int, ColorName] | None,
) -> dict[int, ColorName]:
    return dict(classified_colors_by_node_id or {})


def _color_to_face(
    graph: PieceGraph,
    color_by_node_id: dict[int, ColorName],
    scheme: CubeColorScheme,
    face_by_color: dict[ColorName, Face] | None,
) -> dict[ColorName, Face]:
    explicit = face_by_color or {}
    mapping = dict(scheme.color_to_face)
    mapping.update(explicit)
    used_faces = set(explicit.values())
    mapping.update(_view_relative_center_faces(graph, color_by_node_id, mapping, used_faces))
    return mapping


def _view_relative_center_faces(
    graph: PieceGraph,
    color_by_node_id: dict[int, ColorName],
    color_to_face: dict[ColorName, Face],
    used_faces: set[Face],
) -> dict[ColorName, Face]:
    centers = [
        node
        for node in graph.nodes
        if node.piece_type == "center"
        and node.node_id in color_by_node_id
        and color_by_node_id[node.node_id] not in color_to_face
    ]
    if not centers:
        return {}
    ordered_centers, face_order = _view_relative_center_order_and_faces(centers)
    available_faces = [face for face in face_order if face not in used_faces]
    return {
        color_by_node_id[center.node_id]: face
        for center, face in zip(ordered_centers, available_faces)
    }


def _view_relative_center_order_and_faces(
    centers: list[StickerNode],
) -> tuple[list[StickerNode], tuple[Face, ...]]:
    if len(centers) <= 1:
        return centers, ("F", "R", "U", "L", "D", "B")

    top = min(centers, key=lambda node: node.centroid_xy[1])
    bottom = max(centers, key=lambda node: node.centroid_xy[1])
    x_span = max(node.centroid_xy[0] for node in centers) - min(
        node.centroid_xy[0] for node in centers
    )
    y_span = bottom.centroid_xy[1] - top.centroid_xy[1]
    if len(centers) == 2 and y_span <= x_span:
        return sorted(centers, key=lambda node: node.centroid_xy[0]), (
            "F",
            "R",
            "U",
            "L",
            "D",
            "B",
        )

    remaining = [center for center in centers if center is not top]
    return [top] + sorted(remaining, key=lambda node: node.centroid_xy[0]), (
        "U",
        "F",
        "R",
        "L",
        "D",
        "B",
    )


def _center_edge_connections(
    graph: PieceGraph,
    nodes_by_id: dict[int, StickerNode],
) -> dict[int, list[int]]:
    center_ids_by_edge_id: dict[int, list[int]] = defaultdict(list)
    for left_id, right_id in graph.edges:
        left = nodes_by_id[left_id]
        right = nodes_by_id[right_id]
        if left.piece_type == "edge" and right.piece_type == "center":
            center_ids_by_edge_id[left_id].append(right_id)
        elif right.piece_type == "edge" and left.piece_type == "center":
            center_ids_by_edge_id[right_id].append(left_id)
    return center_ids_by_edge_id


def _edge_nodes_by_piece(nodes: list[StickerNode]) -> dict[int, list[StickerNode]]:
    edge_nodes: dict[int, list[StickerNode]] = defaultdict(list)
    for node in nodes:
        if node.piece_type == "edge":
            edge_nodes[node.piece_id].append(node)
    return edge_nodes


def _corner_edge_connections(
    graph: PieceGraph,
    nodes_by_id: dict[int, StickerNode],
) -> dict[int, list[int]]:
    edge_ids_by_corner_id: dict[int, list[int]] = defaultdict(list)
    for left_id, right_id in graph.edges:
        left = nodes_by_id[left_id]
        right = nodes_by_id[right_id]
        if left.piece_type == "corner" and right.piece_type == "edge":
            edge_ids_by_corner_id[left_id].append(right_id)
        elif right.piece_type == "corner" and left.piece_type == "edge":
            edge_ids_by_corner_id[right_id].append(left_id)
    return edge_ids_by_corner_id


def _edge_cell_for_center(center: StickerNode, edge: StickerNode) -> tuple[int, int]:
    dx = edge.centroid_xy[0] - center.centroid_xy[0]
    dy = edge.centroid_xy[1] - center.centroid_xy[1]
    if abs(dx) >= abs(dy):
        return 1, 2 if dx > 0 else 0
    return (2 if dy > 0 else 0), 1


def _corner_cell_for_center(center: StickerNode, corner: StickerNode) -> tuple[int, int]:
    dx = corner.centroid_xy[0] - center.centroid_xy[0]
    dy = corner.centroid_xy[1] - center.centroid_xy[1]
    return (2 if dy > 0 else 0), (2 if dx > 0 else 0)


def _nearest_known_edge_face(
    corner: StickerNode,
    edge_ids: list[int],
    nodes_by_id: dict[int, StickerNode],
    edge_face_by_node_id: dict[int, Face],
) -> Face | None:
    candidates = [
        (_node_distance(corner, nodes_by_id[edge_id]), edge_face_by_node_id[edge_id])
        for edge_id in edge_ids
        if edge_id in edge_face_by_node_id
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: candidate[0])[1]


def _node_distance(left: StickerNode, right: StickerNode) -> float:
    left_xy = np.asarray(left.centroid_xy, dtype=np.float32)
    right_xy = np.asarray(right.centroid_xy, dtype=np.float32)
    return float(np.linalg.norm(left_xy - right_xy))


def _add_observed_node_cell(
    state: CubeNetState,
    node: StickerNode,
    face: Face,
    row: int,
    col: int,
) -> None:
    state.add_cell(
        face,
        row,
        col,
        f"node-{node.node_id}",
        "observed",
        node.confidence,
        bgr=node.mean_bgr,
    )


def _bgr_for_color(
    color: ColorName,
    classified_bgr_by_color: dict[ColorName, tuple[int, int, int]] | None,
    scheme: CubeColorScheme,
) -> tuple[int, int, int] | None:
    if classified_bgr_by_color is not None and color in classified_bgr_by_color:
        return classified_bgr_by_color[color]
    return scheme.palette_bgr.get(color)


def _cell_bgr(cell: NetCell, scheme: CubeColorScheme) -> tuple[int, int, int]:
    if cell.bgr is not None:
        return cell.bgr
    return scheme.palette_bgr.get(cell.color, (224, 224, 224))
