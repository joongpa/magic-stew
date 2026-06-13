from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import os

import cv2
import numpy as np
from numpy.typing import NDArray

from live_recognition.model_utils import ensure_yolo_config_dir, resolve_model_path
from live_recognition.types import CubeCandidate
from live_recognition.visualization import _reliable_color_mask


DEFAULT_PIECE_SEGMENTATION_MODEL = (
    "runs/segment/runs/cube_piece_seg_sticker/runs/cube_piece_yolo11n_seg/weights/best.pt"
)
PIECE_TYPES = {"center", "edge", "corner"}


@dataclass(frozen=True)
class PieceMask:
    piece_id: int
    class_id: int
    piece_type: str
    confidence: float
    mask: NDArray[np.uint8]
    roi_mask: NDArray[np.bool_]
    roi_xyxy: tuple[int, int, int, int]
    sample_roi_mask: NDArray[np.bool_] | None = None


@dataclass(frozen=True)
class StickerNode:
    node_id: int
    piece_id: int
    piece_type: str
    confidence: float
    centroid_xy: tuple[float, float]
    mask: NDArray[np.uint8]
    mean_bgr: tuple[int, int, int] = (0, 0, 0)


@dataclass(frozen=True)
class PieceGraph:
    nodes: list[StickerNode]
    edges: list[tuple[int, int]]


@dataclass(frozen=True)
class _MaskAdjacencyData:
    node: StickerNode
    mask_bbox: tuple[int, int, int, int]
    expanded_bbox: tuple[int, int, int, int]
    dilated_mask: NDArray[np.uint8]


class YOLOPieceSegmenter:
    """Piece segmentation model that returns center/edge/corner instance masks."""

    def __init__(
        self,
        model_name: str = DEFAULT_PIECE_SEGMENTATION_MODEL,
        *,
        confidence_threshold: float = 0.25,
        image_size: int = 640,
    ) -> None:
        ensure_yolo_config_dir()
        os.environ.setdefault("YOLO_CONFIG_DIR", ".ultralytics")
        from ultralytics import YOLO

        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.model = YOLO(resolve_model_path(model_name, "YOLO piece segmentation"))

    def segment_roi(
        self,
        roi_bgr: NDArray[np.uint8],
        *,
        frame_shape: tuple[int, int],
        roi_xyxy: tuple[int, int, int, int],
        segmentation_padding_px: int = -2,
    ) -> list[PieceMask]:
        x1, y1, x2, y2 = roi_xyxy
        roi_height = y2 - y1
        roi_width = x2 - x1
        results = self.model.predict(
            source=roi_bgr,
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        masks = result.masks
        boxes = result.boxes
        if masks is None or boxes is None or len(boxes) == 0:
            return []

        names = getattr(result, "names", None) or getattr(self.model, "names", {})
        frame_height, frame_width = frame_shape
        mask_data = masks.data.detach().cpu().numpy()
        piece_masks = []
        for index in range(len(boxes)):
            confidence = float(boxes.conf[index].item())
            if confidence < self.confidence_threshold:
                continue

            class_id = int(boxes.cls[index].item())
            piece_type = str(names.get(class_id, class_id)).lower()
            if piece_type not in PIECE_TYPES:
                continue

            roi_mask = _piece_mask_bool(mask_data, index, roi_width, roi_height)
            if not bool(roi_mask.any()):
                continue

            sample_roi_mask = apply_mask_padding(roi_mask, segmentation_padding_px)
            if not bool(sample_roi_mask.any()):
                sample_roi_mask = roi_mask

            full_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = roi_mask.astype(np.uint8) * 255
            piece_masks.append(
                PieceMask(
                    piece_id=len(piece_masks),
                    class_id=class_id,
                    piece_type=piece_type,
                    confidence=confidence,
                    mask=full_mask,
                    roi_mask=roi_mask,
                    roi_xyxy=roi_xyxy,
                    sample_roi_mask=sample_roi_mask,
                )
            )
        return piece_masks


def build_piece_graph(
    frame_bgr: NDArray[np.uint8],
    piece_masks: list[PieceMask],
    *,
    adjacency_radius_px: int = 6,
    min_component_area_px: int = 12,
    infer_piece_ids: bool = False,
) -> PieceGraph:
    nodes = _sticker_nodes_from_piece_masks(
        frame_bgr,
        piece_masks,
        min_component_area_px=min_component_area_px,
    )
    if infer_piece_ids:
        nodes = infer_piece_ids_from_same_type_adjacencies(
            nodes,
            adjacency_radius_px=adjacency_radius_px,
        )
    edges = _legal_mask_adjacencies(nodes, adjacency_radius_px=adjacency_radius_px)
    return PieceGraph(nodes=nodes, edges=edges)


def build_sticker_adjacency_graph(
    nodes: list[StickerNode],
    *,
    adjacency_radius_px: int = 6,
    infer_piece_ids: bool = True,
) -> PieceGraph:
    """Build a geometric sticker adjacency graph from already-segmented sticker nodes."""
    graph_nodes = (
        infer_piece_ids_from_same_type_adjacencies(
            nodes,
            adjacency_radius_px=adjacency_radius_px,
        )
        if infer_piece_ids
        else list(nodes)
    )
    return PieceGraph(
        nodes=graph_nodes,
        edges=_legal_mask_adjacencies(graph_nodes, adjacency_radius_px=adjacency_radius_px),
    )


def infer_piece_ids_from_same_type_adjacencies(
    nodes: list[StickerNode],
    *,
    adjacency_radius_px: int,
) -> list[StickerNode]:
    """Group adjacent stickers of the same piece type into the same piece id."""
    if not nodes:
        return []

    same_type_edges = _same_type_mask_adjacencies(nodes, adjacency_radius_px=adjacency_radius_px)
    neighbors: dict[int, set[int]] = {node.node_id: set() for node in nodes}
    for left_id, right_id in same_type_edges:
        neighbors[left_id].add(right_id)
        neighbors[right_id].add(left_id)

    next_piece_id = 0
    piece_id_by_node_id: dict[int, int] = {}
    for node in nodes:
        if node.node_id in piece_id_by_node_id:
            continue
        stack = [node.node_id]
        while stack:
            node_id = stack.pop()
            if node_id in piece_id_by_node_id:
                continue
            piece_id_by_node_id[node_id] = next_piece_id
            stack.extend(neighbors[node_id] - piece_id_by_node_id.keys())
        next_piece_id += 1

    return [
        replace(node, piece_id=piece_id_by_node_id.get(node.node_id, node.piece_id))
        for node in nodes
    ]


def clipped_square_roi(
    bbox_xyxy: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * (1.0 + padding * 2.0)

    rx1 = int(round(cx - side / 2.0))
    ry1 = int(round(cy - side / 2.0))
    rx2 = int(round(cx + side / 2.0))
    ry2 = int(round(cy + side / 2.0))

    rx1 = max(0, min(frame_width - 1, rx1))
    ry1 = max(0, min(frame_height - 1, ry1))
    rx2 = max(rx1 + 1, min(frame_width, rx2))
    ry2 = max(ry1 + 1, min(frame_height, ry2))
    return rx1, ry1, rx2, ry2


def apply_mask_padding(mask: NDArray[np.bool_], padding_px: int) -> NDArray[np.bool_]:
    if padding_px == 0 or not bool(mask.any()):
        return mask

    radius = abs(padding_px)
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    mask_u8 = mask.astype(np.uint8) * 255
    if padding_px < 0:
        padded = cv2.erode(mask_u8, kernel, iterations=1)
    else:
        padded = cv2.dilate(mask_u8, kernel, iterations=1)
    return padded > 0


def draw_piece_graph(
    frame_bgr: NDArray[np.uint8],
    graph: PieceGraph,
    *,
    cube_candidate: CubeCandidate | None = None,
    roi_xyxy: tuple[int, int, int, int] | None = None,
    piece_masks: list[PieceMask] | None = None,
    draw_masks: bool = True,
) -> NDArray[np.uint8]:
    output = frame_bgr.copy()

    if draw_masks and piece_masks:
        overlay = output.copy()
        for piece_mask in piece_masks:
            color = _piece_type_color(piece_mask.piece_type)
            mask = piece_mask.mask > 0
            overlay[mask] = color
        output = cv2.addWeighted(output, 0.70, overlay, 0.30, 0)
    elif draw_masks:
        overlay = output.copy()
        for node in graph.nodes:
            color = _piece_type_color(node.piece_type)
            mask = node.mask > 0
            overlay[mask] = color
        output = cv2.addWeighted(output, 0.70, overlay, 0.30, 0)

    if roi_xyxy is not None or cube_candidate is not None:
        x1, y1, x2, y2 = roi_xyxy if roi_xyxy is not None else cube_candidate.bbox_xyxy
        cv2.rectangle(output, (x1, y1), (x2, y2), (60, 230, 60), 2)

    nodes_by_id = {node.node_id: node for node in graph.nodes}
    for start_id, end_id in graph.edges:
        start = nodes_by_id[start_id]
        end = nodes_by_id[end_id]
        cv2.line(
            output,
            _int_point(start.centroid_xy),
            _int_point(end.centroid_xy),
            (255, 255, 255),
            5,
            lineType=cv2.LINE_AA,
        )
        cv2.line(
            output,
            _int_point(start.centroid_xy),
            _int_point(end.centroid_xy),
            (30, 30, 30),
            2,
            lineType=cv2.LINE_AA,
        )

    for node in graph.nodes:
        center = _int_point(node.centroid_xy)
        color = _piece_type_color(node.piece_type)
        cv2.circle(output, center, 7, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(output, center, 5, color, -1, lineType=cv2.LINE_AA)

    return output


def _sticker_nodes_from_piece_masks(
    frame_bgr: NDArray[np.uint8],
    piece_masks: list[PieceMask],
    *,
    min_component_area_px: int,
) -> list[StickerNode]:
    nodes = []
    for piece_mask in piece_masks:
        if int(piece_mask.roi_mask.sum()) < min_component_area_px:
            continue
        x1, y1, _, _ = piece_mask.roi_xyxy
        sticker_ys, sticker_xs = np.nonzero(piece_mask.roi_mask)
        if len(sticker_xs) == 0:
            continue

        mean_bgr = _mean_bgr_from_roi_mask(
            frame_bgr,
            piece_mask.roi_xyxy,
            _color_sample_mask(piece_mask.roi_mask, piece_mask.sample_roi_mask),
        )
        if mean_bgr is None:
            continue

        nodes.append(
            StickerNode(
                node_id=len(nodes),
                piece_id=piece_mask.piece_id,
                piece_type=piece_mask.piece_type,
                confidence=piece_mask.confidence,
                centroid_xy=(
                    float(sticker_xs.mean() + x1),
                    float(sticker_ys.mean() + y1),
                ),
                mask=piece_mask.mask,
                mean_bgr=mean_bgr,
            )
        )
    return nodes


def _color_sample_mask(
    sticker_mask: NDArray[np.bool_],
    sample_piece_mask: NDArray[np.bool_] | None,
) -> NDArray[np.bool_]:
    if sample_piece_mask is None:
        return sticker_mask

    sample_mask = sticker_mask & sample_piece_mask
    return sample_mask if bool(sample_mask.any()) else sticker_mask


def _mean_bgr_from_roi_mask(
    frame_bgr: NDArray[np.uint8],
    roi_xyxy: tuple[int, int, int, int],
    sample_mask: NDArray[np.bool_],
) -> tuple[int, int, int] | None:
    x1, y1, x2, y2 = roi_xyxy
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0 or not bool(sample_mask.any()):
        return None

    reliable_mask = _reliable_color_mask(crop) & sample_mask
    if int(reliable_mask.sum()) < 10:
        reliable_mask = sample_mask
    if not bool(reliable_mask.any()):
        return None

    mean_bgr = (
        crop[reliable_mask]
        .astype(np.float32)
        .mean(axis=0)
        .clip(0, 255)
        .astype(np.uint8)
    )
    return tuple(int(channel) for channel in mean_bgr)


def _piece_mask_bool(
    mask_data: NDArray[np.float32],
    index: int,
    roi_width: int,
    roi_height: int,
) -> NDArray[np.bool_]:
    mask = cv2.resize(
        mask_data[index].astype(np.float32),
        (roi_width, roi_height),
        interpolation=cv2.INTER_LINEAR,
    )
    return mask > 0.5


def _legal_mask_adjacencies(
    nodes: list[StickerNode],
    *,
    adjacency_radius_px: int,
) -> list[tuple[int, int]]:
    if not nodes:
        return []

    adjacency_data = _mask_adjacency_data(nodes, adjacency_radius_px)

    edges = []
    for left_index, left_data in enumerate(adjacency_data):
        left = left_data.node
        for right_data in adjacency_data[left_index + 1 :]:
            right = right_data.node
            if not _mask_data_within_radius(left_data, right_data):
                continue
            edges.append(tuple(sorted((left.node_id, right.node_id))))
    return sorted(edges)


def _same_type_mask_adjacencies(
    nodes: list[StickerNode],
    *,
    adjacency_radius_px: int,
) -> list[tuple[int, int]]:
    if not nodes:
        return []

    adjacency_data = _mask_adjacency_data(nodes, adjacency_radius_px)
    edges = []
    for left_index, left_data in enumerate(adjacency_data):
        left = left_data.node
        for right_data in adjacency_data[left_index + 1 :]:
            right = right_data.node
            if left.piece_type != right.piece_type:
                continue
            if _mask_data_within_radius(left_data, right_data):
                edges.append(tuple(sorted((left.node_id, right.node_id))))
    return sorted(edges)


def _mask_adjacency_data(
    nodes: list[StickerNode],
    adjacency_radius_px: int,
) -> list[_MaskAdjacencyData]:
    radius = max(0, int(adjacency_radius_px))
    kernel = None
    if radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (radius * 2 + 1, radius * 2 + 1),
        )

    data = []
    for node in nodes:
        bbox = _mask_bbox(node.mask)
        if bbox is None:
            continue

        height, width = node.mask.shape[:2]
        expanded_bbox = _expand_bbox(bbox, radius, width, height)
        x1, y1, x2, y2 = expanded_bbox
        mask_crop = node.mask[y1:y2, x1:x2]
        dilated_mask = cv2.dilate(mask_crop, kernel, iterations=1) if kernel is not None else mask_crop
        data.append(
            _MaskAdjacencyData(
                node=node,
                mask_bbox=bbox,
                expanded_bbox=expanded_bbox,
                dilated_mask=dilated_mask,
            )
        )
    return data


def _mask_data_within_radius(
    left: _MaskAdjacencyData,
    right: _MaskAdjacencyData,
) -> bool:
    return (
        _dilated_crop_intersects_mask(left, right)
        or _dilated_crop_intersects_mask(right, left)
    )


def _dilated_crop_intersects_mask(
    dilated: _MaskAdjacencyData,
    target: _MaskAdjacencyData,
) -> bool:
    intersection = _bbox_intersection(dilated.expanded_bbox, target.mask_bbox)
    if intersection is None:
        return False

    x1, y1, x2, y2 = intersection
    dx1, dy1, _, _ = dilated.expanded_bbox
    dilated_crop = dilated.dilated_mask[y1 - dy1 : y2 - dy1, x1 - dx1 : x2 - dx1]
    target_crop = target.node.mask[y1:y2, x1:x2]
    return cv2.countNonZero(cv2.bitwise_and(dilated_crop, target_crop)) > 0


def _mask_bbox(mask: NDArray[np.uint8]) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    radius: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, x1 - radius),
        max(0, y1 - radius),
        min(width, x2 + radius),
        min(height, y2 + radius),
    )


def _bbox_intersection(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    if x1 >= x2 or y1 >= y2:
        return None
    return x1, y1, x2, y2


def _masks_within_radius(
    left_mask: NDArray[np.uint8],
    right_mask: NDArray[np.uint8],
    left_dilated_mask: NDArray[np.uint8],
    right_dilated_mask: NDArray[np.uint8],
) -> bool:
    return (
        cv2.countNonZero(cv2.bitwise_and(left_dilated_mask, right_mask)) > 0
        or cv2.countNonZero(cv2.bitwise_and(right_dilated_mask, left_mask)) > 0
    )


def _record_closest_piece_candidate(
    candidates_by_node_piece: dict[tuple[int, int], tuple[float, int, int]],
    *,
    source_node_id: int,
    target_piece_id: int,
    target_node_id: int,
    distance: float,
) -> None:
    key = (source_node_id, target_piece_id)
    current = candidates_by_node_piece.get(key)
    candidate = (distance, target_piece_id, target_node_id)
    if current is None or candidate < current:
        candidates_by_node_piece[key] = candidate


def _centroid_distance(left: StickerNode, right: StickerNode) -> float:
    left_xy = np.asarray(left.centroid_xy, dtype=np.float32)
    right_xy = np.asarray(right.centroid_xy, dtype=np.float32)
    return float(np.linalg.norm(left_xy - right_xy))




def _piece_type_color(piece_type: str) -> tuple[int, int, int]:
    return {
        "center": (70, 220, 70),
        "edge": (70, 170, 255),
        "corner": (220, 100, 255),
    }.get(piece_type, (220, 220, 220))


def _int_point(point_xy: tuple[float, float]) -> tuple[int, int]:
    return int(round(point_xy[0])), int(round(point_xy[1]))


def _clip_bbox(
    bbox_xyxy: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    return (
        max(0, min(width, x1)),
        max(0, min(height, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    )
