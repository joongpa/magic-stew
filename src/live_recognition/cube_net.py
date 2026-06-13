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

FACE_SIDE_BY_NEIGHBOR: dict[Face, dict[Face, tuple[int, int]]] = {
    "F": {"U": (0, 1), "R": (1, 2), "D": (2, 1), "L": (1, 0)},
    "R": {"U": (0, 1), "B": (1, 2), "D": (2, 1), "F": (1, 0)},
    "B": {"U": (0, 1), "L": (1, 2), "D": (2, 1), "R": (1, 0)},
    "L": {"U": (0, 1), "F": (1, 2), "D": (2, 1), "B": (1, 0)},
    "U": {"B": (0, 1), "R": (1, 2), "F": (2, 1), "L": (1, 0)},
    "D": {"F": (0, 1), "R": (1, 2), "B": (2, 1), "L": (1, 0)},
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
        faces_by_node_id = {
            edge_node.node_id: center_face_by_id[center_id]
            for edge_node, center_id in connected
        }
        for edge_node, center_id in connected:
            color = color_by_node_id.get(edge_node.node_id)
            if color is None:
                continue
            center = centers_by_id[center_id]
            face = center_face_by_id[center_id]
            neighbor_face = _edge_piece_neighbor_face(edge_node, faces_by_node_id)
            row_col = _edge_cell_for_neighbor_face(face, neighbor_face)
            if row_col is None:
                row_col = _edge_cell_for_center(center, edge_node)
            state.add_cell(
                face,
                row_col[0],
                row_col[1],
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

    state = CubeNetState()
    neighbors_by_id = _graph_neighbors(graph.edges)
    edge_cell_overrides = _cross_face_edge_cell_overrides(
        graph,
        nodes_by_id,
        center_face_by_id,
    )
    face_cells_by_face: dict[Face, dict[tuple[int, int], StickerNode]] = {}
    for center in ordered_centers:
        face = center_face_by_id.get(center.node_id)
        if face is None:
            continue
        face_cells_by_face[face] = _face_cells_from_graph(
            center,
            nodes_by_id,
            neighbors_by_id,
            edge_cell_overrides,
        )

    _expand_cross_face_same_type_cells(face_cells_by_face, graph, nodes_by_id)
    for face, cells in face_cells_by_face.items():
        for (row, col), node in cells.items():
            _add_observed_node_cell(state, node, face, row, col)

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


def _cross_face_edge_cell_overrides(
    graph: PieceGraph,
    nodes_by_id: dict[int, StickerNode],
    center_face_by_id: dict[int, Face],
) -> dict[tuple[int, int], tuple[int, int]]:
    center_edges = _center_edge_connections(graph, nodes_by_id)
    overrides: dict[tuple[int, int], tuple[int, int]] = {}
    for left_id, right_id in graph.edges:
        left = nodes_by_id[left_id]
        right = nodes_by_id[right_id]
        if left.piece_type != "edge" or right.piece_type != "edge":
            continue

        left_center_id = _edge_center_id(left, center_edges, nodes_by_id, center_face_by_id)
        right_center_id = _edge_center_id(right, center_edges, nodes_by_id, center_face_by_id)
        if left_center_id is None or right_center_id is None or left_center_id == right_center_id:
            continue

        left_face = center_face_by_id[left_center_id]
        right_face = center_face_by_id[right_center_id]
        left_cell = _edge_cell_for_neighbor_face(left_face, right_face)
        right_cell = _edge_cell_for_neighbor_face(right_face, left_face)
        if left_cell is not None:
            overrides[(left_center_id, left.node_id)] = left_cell
        if right_cell is not None:
            overrides[(right_center_id, right.node_id)] = right_cell
    return overrides


def _edge_center_id(
    edge: StickerNode,
    center_edges: dict[int, list[int]],
    nodes_by_id: dict[int, StickerNode],
    center_face_by_id: dict[int, Face],
) -> int | None:
    center_ids = [
        center_id
        for center_id in center_edges.get(edge.node_id, [])
        if center_id in center_face_by_id
    ]
    if not center_ids:
        return None
    return min(center_ids, key=lambda center_id: _node_distance(edge, nodes_by_id[center_id]))


def _graph_neighbors(edges: list[tuple[int, int]]) -> dict[int, set[int]]:
    neighbors: dict[int, set[int]] = defaultdict(set)
    for left_id, right_id in edges:
        neighbors[left_id].add(right_id)
        neighbors[right_id].add(left_id)
    return neighbors


def _face_cells_from_graph(
    center: StickerNode,
    nodes_by_id: dict[int, StickerNode],
    neighbors_by_id: dict[int, set[int]],
    edge_cell_overrides: dict[tuple[int, int], tuple[int, int]],
) -> dict[tuple[int, int], StickerNode]:
    cells_by_node_id: dict[int, tuple[int, int]] = {center.node_id: (1, 1)}

    for neighbor_id in neighbors_by_id.get(center.node_id, set()):
        neighbor = nodes_by_id[neighbor_id]
        cell = edge_cell_overrides.get((center.node_id, neighbor_id))
        if cell is None:
            cell = _seed_cell_from_center(center, neighbor)
        if cell is not None:
            cells_by_node_id[neighbor_id] = cell

    changed = True
    while changed:
        changed = False
        for node_id, cell in list(cells_by_node_id.items()):
            node = nodes_by_id[node_id]
            for neighbor_id in neighbors_by_id.get(node_id, set()):
                if neighbor_id in cells_by_node_id:
                    continue
                neighbor = nodes_by_id[neighbor_id]
                inferred = _infer_neighbor_cell(center, node, cell, neighbor)
                if inferred is None:
                    continue
                cells_by_node_id[neighbor_id] = inferred
                changed = True

    _refine_corner_cells_from_edges(cells_by_node_id, nodes_by_id, neighbors_by_id)
    return _dedupe_face_cells(center, cells_by_node_id, nodes_by_id)


def _seed_cell_from_center(
    center: StickerNode,
    node: StickerNode,
) -> tuple[int, int] | None:
    if node.piece_type == "center":
        return (1, 1) if node.node_id == center.node_id else None
    if node.piece_type == "edge":
        return _edge_cell_for_center(center, node)
    if node.piece_type == "corner":
        return _corner_cell_for_center(center, node)
    return None


def _infer_neighbor_cell(
    center: StickerNode,
    node: StickerNode,
    cell: tuple[int, int],
    neighbor: StickerNode,
) -> tuple[int, int] | None:
    row, col = cell
    if neighbor.piece_type == "center":
        return None

    if node.piece_type == "center":
        return _seed_cell_from_center(center, neighbor)

    if node.piece_type == "edge" and neighbor.piece_type == "corner":
        fallback_row, fallback_col = _corner_cell_for_center(center, neighbor)
        return _corner_cell_from_single_edge(row, col, fallback_row, fallback_col)

    if node.piece_type == "corner" and neighbor.piece_type == "edge":
        return _edge_cell_for_corner(center, node, neighbor)

    return None


def _edge_cell_for_corner(
    center: StickerNode,
    corner: StickerNode,
    edge: StickerNode,
) -> tuple[int, int] | None:
    corner_row, corner_col = _corner_cell_for_center(center, corner)
    edge_row, edge_col = _edge_cell_for_center(center, edge)
    if edge_col == 1:
        return corner_row, 1
    if edge_row == 1:
        return 1, corner_col
    return edge_row, edge_col


def _refine_corner_cells_from_edges(
    cells_by_node_id: dict[int, tuple[int, int]],
    nodes_by_id: dict[int, StickerNode],
    neighbors_by_id: dict[int, set[int]],
) -> None:
    for node_id, node in nodes_by_id.items():
        if node.piece_type != "corner":
            continue
        edge_cells = [
            (cells_by_node_id[neighbor_id], nodes_by_id[neighbor_id])
            for neighbor_id in neighbors_by_id.get(node_id, set())
            if neighbor_id in cells_by_node_id and nodes_by_id[neighbor_id].piece_type == "edge"
        ]
        row_edge_rows = [
            (_node_distance(node, edge), row)
            for (row, col), edge in edge_cells
            if col == 1 and row != 1
        ]
        col_edge_cols = [
            (_node_distance(node, edge), col)
            for (row, col), edge in edge_cells
            if row == 1 and col != 1
        ]
        if row_edge_rows and col_edge_cols:
            cells_by_node_id[node_id] = (
                min(row_edge_rows, key=lambda candidate: candidate[0])[1],
                min(col_edge_cols, key=lambda candidate: candidate[0])[1],
            )


def _dedupe_face_cells(
    center: StickerNode,
    cells_by_node_id: dict[int, tuple[int, int]],
    nodes_by_id: dict[int, StickerNode],
) -> dict[tuple[int, int], StickerNode]:
    candidates: dict[tuple[int, int], tuple[float, StickerNode]] = {}
    for node_id, cell in cells_by_node_id.items():
        row, col = cell
        if not (0 <= row <= 2 and 0 <= col <= 2):
            continue
        node = nodes_by_id[node_id]
        score = _cell_assignment_score(center, node, row, col)
        current = candidates.get(cell)
        if current is None or score < current[0]:
            candidates[cell] = (score, node)
    return {cell: node for cell, (_, node) in candidates.items()}


def _cell_assignment_score(
    center: StickerNode,
    node: StickerNode,
    row: int,
    col: int,
) -> float:
    expected_type = "center" if (row, col) == (1, 1) else "edge" if row == 1 or col == 1 else "corner"
    type_penalty = 0.0 if node.piece_type == expected_type else 10_000.0
    if node.node_id == center.node_id:
        type_penalty -= 1_000.0
    return type_penalty + _node_distance(center, node)


def _expand_cross_face_same_type_cells(
    face_cells_by_face: dict[Face, dict[tuple[int, int], StickerNode]],
    graph: PieceGraph,
    nodes_by_id: dict[int, StickerNode],
) -> None:
    neighbors_by_id = _graph_neighbors(graph.edges)
    face_node_cells: dict[Face, dict[int, tuple[int, int]]] = {
        face: {node.node_id: cell for cell, node in cells.items()}
        for face, cells in face_cells_by_face.items()
    }
    assigned_node_ids = {
        node_id
        for node_cells in face_node_cells.values()
        for node_id in node_cells
    }

    changed = True
    while changed:
        changed = False
        candidates_by_node: dict[int, dict[tuple[Face, tuple[int, int]], tuple[int, float]]] = (
            defaultdict(dict)
        )
        for face, node_cells in face_node_cells.items():
            for node_id, cell in node_cells.items():
                source = nodes_by_id[node_id]
                for neighbor_id in neighbors_by_id.get(node_id, set()):
                    if neighbor_id in assigned_node_ids:
                        continue
                    neighbor = nodes_by_id[neighbor_id]
                    for target_face, target_cell in _cross_face_neighbor_cells(
                        face,
                        cell,
                        neighbor.piece_type,
                    ):
                        target = (target_face, target_cell)
                        current = candidates_by_node[neighbor_id].get(target)
                        support = 1 if current is None else current[0] + 1
                        distance = _node_distance(source, neighbor)
                        total_distance = distance if current is None else current[1] + distance
                        candidates_by_node[neighbor_id][target] = (
                            support,
                            total_distance,
                        )

        candidates_by_cell: dict[Face, dict[tuple[int, int], dict[int, tuple[int, float]]]] = (
            defaultdict(lambda: defaultdict(dict))
        )
        for node_id, node_candidates in candidates_by_node.items():
            target_face, target_cell = min(
                node_candidates,
                key=lambda target: (
                    -node_candidates[target][0],
                    node_candidates[target][1],
                    target[0],
                    target[1],
                ),
            )
            candidates_by_cell[target_face][target_cell][node_id] = node_candidates[
                (target_face, target_cell)
            ]

        for face, cells in candidates_by_cell.items():
            face_cells_by_face.setdefault(face, {})
            face_node_cells.setdefault(face, {})
            for cell, node_candidates in cells.items():
                if cell in face_cells_by_face[face]:
                    continue
                node_id = min(
                    node_candidates,
                    key=lambda candidate_id: (
                        -node_candidates[candidate_id][0],
                        node_candidates[candidate_id][1],
                        candidate_id,
                    ),
                )
                face_cells_by_face[face][cell] = nodes_by_id[node_id]
                face_node_cells[face][node_id] = cell
                assigned_node_ids.add(node_id)
                changed = True


def _cross_face_cells(face: Face, cell: tuple[int, int]) -> list[tuple[Face, tuple[int, int]]]:
    vector = _face_cell_vector(face, cell)
    cells = []
    for neighbor_face, side in FACE_SIDE_BY_NEIGHBOR.get(face, {}).items():
        if not _cell_on_side(cell, side):
            continue
        neighbor_cell = _vector_face_cell(neighbor_face, vector)
        if neighbor_cell is not None:
            cells.append((neighbor_face, neighbor_cell))
    return cells


def _cross_face_neighbor_cells(
    face: Face,
    cell: tuple[int, int],
    neighbor_piece_type: str,
) -> list[tuple[Face, tuple[int, int]]]:
    candidates = []
    for target_face, target_cell in _cross_face_cells(face, cell):
        target_piece_type = _cell_piece_type(target_cell)
        if target_piece_type == neighbor_piece_type:
            candidates.append((target_face, target_cell))
        elif target_piece_type == "edge" and neighbor_piece_type == "corner":
            candidates.extend(
                (target_face, corner_cell)
                for corner_cell in _corner_cells_for_edge_cell(target_cell)
            )
        elif target_piece_type == "corner" and neighbor_piece_type == "edge":
            candidates.extend(
                (target_face, edge_cell)
                for edge_cell in _edge_cells_for_corner_cell(target_cell)
            )
    return candidates


def _corner_cells_for_edge_cell(cell: tuple[int, int]) -> list[tuple[int, int]]:
    row, col = cell
    if row in (0, 2) and col == 1:
        return [(row, 0), (row, 2)]
    if col in (0, 2) and row == 1:
        return [(0, col), (2, col)]
    return []


def _edge_cells_for_corner_cell(cell: tuple[int, int]) -> list[tuple[int, int]]:
    row, col = cell
    if row not in (0, 2) or col not in (0, 2):
        return []
    return [(row, 1), (1, col)]


def _cell_on_side(cell: tuple[int, int], side: tuple[int, int]) -> bool:
    row, col = cell
    side_row, side_col = side
    if side_row in (0, 2):
        return row == side_row
    if side_col in (0, 2):
        return col == side_col
    return False


def _face_cell_vector(face: Face, cell: tuple[int, int]) -> tuple[int, int, int]:
    row, col = cell
    if face == "F":
        return col - 1, row - 1, 1
    if face == "B":
        return 1 - col, row - 1, -1
    if face == "R":
        return 1, row - 1, 1 - col
    if face == "L":
        return -1, row - 1, col - 1
    if face == "U":
        return col - 1, -1, row - 1
    return col - 1, 1, 1 - row


def _vector_face_cell(face: Face, vector: tuple[int, int, int]) -> tuple[int, int] | None:
    x, y, z = vector
    if face == "F" and z == 1:
        return y + 1, x + 1
    if face == "B" and z == -1:
        return y + 1, 1 - x
    if face == "R" and x == 1:
        return y + 1, 1 - z
    if face == "L" and x == -1:
        return y + 1, z + 1
    if face == "U" and y == -1:
        return z + 1, x + 1
    if face == "D" and y == 1:
        return 1 - z, x + 1
    return None


def _cell_piece_type(cell: tuple[int, int]) -> str:
    row, col = cell
    if (row, col) == (1, 1):
        return "center"
    if row == 1 or col == 1:
        return "edge"
    return "corner"


def _edge_nodes_by_piece(nodes: list[StickerNode]) -> dict[int, list[StickerNode]]:
    edge_nodes: dict[int, list[StickerNode]] = defaultdict(list)
    for node in nodes:
        if node.piece_type == "edge":
            edge_nodes[node.piece_id].append(node)
    return edge_nodes


def _edge_piece_neighbor_face(
    edge: StickerNode,
    faces_by_node_id: dict[int, Face],
) -> Face | None:
    face = faces_by_node_id.get(edge.node_id)
    if face is None:
        return None
    neighbor_faces = {
        other_face
        for node_id, other_face in faces_by_node_id.items()
        if node_id != edge.node_id and other_face != face
    }
    for neighbor_face in sorted(neighbor_faces):
        if neighbor_face in FACE_SIDE_BY_NEIGHBOR.get(face, {}):
            return neighbor_face
    return None


def _edge_cell_for_neighbor_face(face: Face, neighbor_face: Face | None) -> tuple[int, int] | None:
    if neighbor_face is None:
        return None
    return FACE_SIDE_BY_NEIGHBOR.get(face, {}).get(neighbor_face)


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


def _corner_cell_from_single_edge(
    edge_row: int,
    edge_col: int,
    fallback_row: int,
    fallback_col: int,
) -> tuple[int, int]:
    row = fallback_row
    col = fallback_col
    if edge_col == 1 and edge_row != 1:
        row = edge_row
    if edge_row == 1 and edge_col != 1:
        col = edge_col
    return row, col


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
