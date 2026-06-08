from __future__ import annotations

from dataclasses import dataclass
import os

import cv2
import numpy as np
from numpy.typing import NDArray

from live_recognition.model_utils import ensure_yolo_config_dir, resolve_model_path
from live_recognition.types import CubeCandidate
from live_recognition.visualization import _gray_world_white_balance, _reliable_color_mask


DEFAULT_PIECE_SEGMENTATION_MODEL = (
    "runs/segment/runs/cube_piece_seg_local/runs/cube_piece_yolo11n_seg-3/weights/best.pt"
)
PIECE_TYPES = {"center", "edge", "corner"}
LEGAL_ADJACENCIES = {frozenset(("center", "edge")), frozenset(("edge", "corner"))}
STICKER_COUNTS_BY_CLASS = {
    0: 1,
    1: 2,
    2: 3,
}


@dataclass(frozen=True)
class PieceMask:
    piece_id: int
    class_id: int
    piece_type: str
    confidence: float
    mask: NDArray[np.uint8]
    roi_mask: NDArray[np.bool_]
    roi_xyxy: tuple[int, int, int, int]


@dataclass(frozen=True)
class StickerNode:
    node_id: int
    piece_id: int
    piece_type: str
    confidence: float
    centroid_xy: tuple[float, float]
    mask: NDArray[np.uint8]


@dataclass(frozen=True)
class PieceGraph:
    nodes: list[StickerNode]
    edges: list[tuple[int, int]]


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
        piece_masks = []
        for index in range(len(boxes)):
            confidence = float(boxes.conf[index].item())
            if confidence < self.confidence_threshold:
                continue

            class_id = int(boxes.cls[index].item())
            piece_type = str(names.get(class_id, class_id)).lower()
            if piece_type not in PIECE_TYPES:
                continue

            roi_mask = _piece_mask_bool(result, index, roi_width, roi_height)
            roi_mask = apply_mask_padding(roi_mask, segmentation_padding_px)
            if not bool(roi_mask.any()):
                continue

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
                )
            )
        return piece_masks


def build_piece_graph(
    frame_bgr: NDArray[np.uint8],
    piece_masks: list[PieceMask],
    *,
    adjacency_radius_px: int = 6,
    min_component_area_px: int = 12,
    min_sticker_segment_ratio: float = 0.25,
) -> PieceGraph:
    nodes = _sticker_nodes_from_piece_masks(
        frame_bgr,
        piece_masks,
        min_component_area_px=min_component_area_px,
        min_sticker_segment_ratio=min_sticker_segment_ratio,
    )
    edges = _legal_mask_adjacencies(nodes, adjacency_radius_px=adjacency_radius_px)
    return PieceGraph(nodes=nodes, edges=edges)


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


def sticker_segments_from_piece(
    frame_bgr: NDArray[np.uint8],
    roi_xyxy: tuple[int, int, int, int],
    piece_mask: NDArray[np.bool_],
    class_id: int,
    *,
    spatial_weight: float = 1.0,
    max_iterations: int = 30,
    min_segment_ratio: float = 0.25,
) -> list[tuple[NDArray[np.bool_], NDArray[np.uint8], NDArray[np.float32]]]:
    x1, y1, x2, y2 = roi_xyxy
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0 or not bool(piece_mask.any()):
        return []

    sticker_count = min(
        STICKER_COUNTS_BY_CLASS.get(class_id, 1),
        int(piece_mask.sum()),
    )
    if sticker_count <= 1:
        mean_ab, mean_bgr = _lab_ab_samples_from_piece(
            frame_bgr,
            roi_xyxy,
            piece_mask,
            max_samples=1,
            average_color=True,
        )
        return [(piece_mask, mean_bgr[0], mean_ab[0])] if len(mean_bgr) else []

    labels = _rgbxy_kmeans_labels(
        crop,
        piece_mask,
        cluster_count=sticker_count,
        spatial_weight=spatial_weight,
        max_iterations=max_iterations,
        min_segment_ratio=min_segment_ratio,
    )
    stickers = []
    for cluster_index in sorted(int(label) for label in np.unique(labels[labels >= 0])):
        sticker_mask = piece_mask & (labels == cluster_index)
        if not bool(sticker_mask.any()):
            continue
        mean_ab, mean_bgr = _lab_ab_samples_from_piece(
            frame_bgr,
            roi_xyxy,
            sticker_mask,
            max_samples=1,
            average_color=True,
        )
        if len(mean_bgr) == 0:
            continue
        stickers.append((sticker_mask, mean_bgr[0], mean_ab[0]))
    return stickers


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
    min_sticker_segment_ratio: float,
) -> list[StickerNode]:
    nodes = []
    for piece_mask in piece_masks:
        stickers = sticker_segments_from_piece(
            frame_bgr,
            piece_mask.roi_xyxy,
            piece_mask.roi_mask,
            piece_mask.class_id,
            min_segment_ratio=min_sticker_segment_ratio,
        )
        x1, y1, _, _ = piece_mask.roi_xyxy
        for sticker_mask, _, _ in stickers:
            if int(sticker_mask.sum()) < min_component_area_px:
                continue

            sticker_ys, sticker_xs = np.nonzero(sticker_mask)
            if len(sticker_xs) == 0:
                continue

            full_mask = np.zeros_like(piece_mask.mask)
            full_mask[
                y1 : y1 + sticker_mask.shape[0],
                x1 : x1 + sticker_mask.shape[1],
            ] = sticker_mask.astype(np.uint8) * 255
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
                    mask=full_mask,
                )
            )
    return nodes


def _piece_mask_bool(result, index: int, roi_width: int, roi_height: int) -> NDArray[np.bool_]:
    mask_data = result.masks.data.detach().cpu().numpy()
    mask = cv2.resize(
        mask_data[index].astype(np.float32),
        (roi_width, roi_height),
        interpolation=cv2.INTER_LINEAR,
    )
    return mask > 0.5


def _lab_ab_samples_from_piece(
    frame_bgr: NDArray[np.uint8],
    roi_xyxy: tuple[int, int, int, int],
    piece_mask: NDArray[np.bool_],
    max_samples: int,
    average_color: bool,
) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
    x1, y1, x2, y2 = roi_xyxy
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0 or not bool(piece_mask.any()):
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    reliable_mask = _reliable_color_mask(crop) & piece_mask
    if int(reliable_mask.sum()) < 10:
        reliable_mask = piece_mask.copy()
    if not bool(reliable_mask.any()):
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    balanced = _gray_world_white_balance(crop, reliable_mask)
    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)
    ab_samples = lab[:, :, 1:3][reliable_mask].astype(np.float32)
    bgr_samples = crop[reliable_mask]
    if average_color:
        mean_bgr_sample = (
            bgr_samples.astype(np.float32)
            .mean(axis=0, keepdims=True)
            .clip(0, 255)
            .astype(np.uint8)
        )
        return (
            ab_samples.mean(axis=0, keepdims=True).astype(np.float32),
            mean_bgr_sample,
        )
    if len(ab_samples) > max_samples:
        indexes = np.linspace(0, len(ab_samples) - 1, max_samples, dtype=np.int32)
        ab_samples = ab_samples[indexes]
        bgr_samples = bgr_samples[indexes]
    return ab_samples, bgr_samples


def _rgbxy_kmeans_labels(
    crop_bgr: NDArray[np.uint8],
    mask: NDArray[np.bool_],
    *,
    cluster_count: int,
    spatial_weight: float = 1.0,
    max_iterations: int = 30,
    min_segment_ratio: float = 0.25,
) -> NDArray[np.int32]:
    ys, xs = np.nonzero(mask)
    labels_image = np.full(mask.shape, -1, dtype=np.int32)
    if len(xs) == 0 or cluster_count <= 0:
        return labels_image

    k = min(cluster_count, len(xs))
    rgb = crop_bgr[ys, xs, ::-1].astype(np.float32)
    x_scale = max(mask.shape[1] - 1, 1)
    y_scale = max(mask.shape[0] - 1, 1)
    xy = np.column_stack(
        [
            xs.astype(np.float32) / x_scale * 255.0,
            ys.astype(np.float32) / y_scale * 255.0,
        ]
    )
    features = np.column_stack([rgb, xy * float(spatial_weight)]).astype(np.float32)
    centers = _initialize_kmeans_centers(features, k)
    labels = np.zeros(len(features), dtype=np.int32)
    for _ in range(max_iterations):
        distances = np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2)
        next_labels = distances.argmin(axis=1).astype(np.int32)
        next_centers = centers.copy()
        for cluster_index in range(k):
            members = features[next_labels == cluster_index]
            if len(members) > 0:
                next_centers[cluster_index] = members.mean(axis=0)
        if np.array_equal(labels, next_labels) and np.allclose(centers, next_centers):
            labels = next_labels
            break
        centers = next_centers
        labels = next_labels

    labels = _merge_small_sticker_labels(
        labels,
        features,
        k,
        min_segment_ratio=min_segment_ratio,
    )
    labels = _reorder_sticker_labels(labels, features, int(labels.max()) + 1)
    labels_image[ys, xs] = labels
    return labels_image


def _initialize_kmeans_centers(
    features: NDArray[np.float32],
    cluster_count: int,
) -> NDArray[np.float32]:
    centers = [features[0]]
    while len(centers) < cluster_count:
        distances = np.min(
            np.linalg.norm(features[:, None, :] - np.asarray(centers)[None, :, :], axis=2),
            axis=1,
        )
        centers.append(features[int(np.argmax(distances))])
    return np.asarray(centers, dtype=np.float32)


def _reorder_sticker_labels(
    labels: NDArray[np.int32],
    features: NDArray[np.float32],
    cluster_count: int,
) -> NDArray[np.int32]:
    centers = []
    for cluster_index in range(cluster_count):
        members = features[labels == cluster_index]
        if len(members) == 0:
            centers.append((np.inf, np.inf, cluster_index))
            continue
        center = members.mean(axis=0)
        centers.append((float(center[4]), float(center[3]), cluster_index))
    ordered_indexes = [cluster_index for _, _, cluster_index in sorted(centers)]
    remap = {old_index: new_index for new_index, old_index in enumerate(ordered_indexes)}
    return np.asarray([remap[int(label)] for label in labels], dtype=np.int32)


def _merge_small_sticker_labels(
    labels: NDArray[np.int32],
    features: NDArray[np.float32],
    cluster_count: int,
    *,
    min_segment_ratio: float,
) -> NDArray[np.int32]:
    if min_segment_ratio <= 0 or cluster_count <= 1:
        return labels

    sizes = np.bincount(labels, minlength=cluster_count)
    if len(sizes) == 0:
        return labels

    largest_size = int(sizes.max())
    if largest_size <= 0:
        return labels

    min_size = largest_size * float(min_segment_ratio)
    keep_indexes = np.flatnonzero(sizes >= min_size)
    if len(keep_indexes) == 0:
        keep_indexes = np.array([int(sizes.argmax())], dtype=np.int32)

    centers = []
    for cluster_index in range(cluster_count):
        members = features[labels == cluster_index]
        if len(members) == 0:
            centers.append(np.full(features.shape[1], np.inf, dtype=np.float32))
        else:
            centers.append(members.mean(axis=0))
    centers_array = np.asarray(centers, dtype=np.float32)
    kept_centers = centers_array[keep_indexes]
    kept_index_set = set(int(index) for index in keep_indexes)

    next_labels = labels.copy()
    for cluster_index in range(cluster_count):
        if cluster_index in kept_index_set:
            continue
        member_mask = labels == cluster_index
        if not bool(member_mask.any()):
            continue
        distances = np.linalg.norm(
            features[member_mask, None, :] - kept_centers[None, :, :],
            axis=2,
        )
        nearest_kept = keep_indexes[distances.argmin(axis=1)]
        next_labels[member_mask] = nearest_kept

    compact_indexes = {
        old_index: new_index for new_index, old_index in enumerate(sorted(set(next_labels)))
    }
    return np.asarray(
        [compact_indexes[int(label)] for label in next_labels],
        dtype=np.int32,
    )


def _legal_mask_adjacencies(
    nodes: list[StickerNode],
    *,
    adjacency_radius_px: int,
) -> list[tuple[int, int]]:
    if not nodes:
        return []

    radius = max(0, int(adjacency_radius_px))
    kernel = None
    if radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (radius * 2 + 1, radius * 2 + 1),
        )
    dilated_masks = [
        cv2.dilate(node.mask, kernel, iterations=1) if kernel is not None else node.mask
        for node in nodes
    ]

    candidates_by_node_piece: dict[tuple[int, int], tuple[float, int, int]] = {}
    for left_index, left in enumerate(nodes):
        for right_index in range(left_index + 1, len(nodes)):
            right = nodes[right_index]
            if left.piece_id == right.piece_id:
                continue
            if frozenset((left.piece_type, right.piece_type)) not in LEGAL_ADJACENCIES:
                continue
            if not _masks_within_radius(
                left.mask,
                right.mask,
                dilated_masks[left_index],
                dilated_masks[right_index],
            ):
                continue
            distance = _centroid_distance(left, right)
            _record_closest_piece_candidate(
                candidates_by_node_piece,
                source_node_id=left.node_id,
                target_piece_id=right.piece_id,
                target_node_id=right.node_id,
                distance=distance,
            )
            _record_closest_piece_candidate(
                candidates_by_node_piece,
                source_node_id=right.node_id,
                target_piece_id=left.piece_id,
                target_node_id=left.node_id,
                distance=distance,
            )

    edges = set()
    piece_ids_by_node_id = {node.node_id: node.piece_id for node in nodes}
    for (source_node_id, _), (_, _, target_node_id) in candidates_by_node_piece.items():
        reverse_key = (target_node_id, piece_ids_by_node_id[source_node_id])
        reverse_candidate = candidates_by_node_piece.get(reverse_key)
        if reverse_candidate is not None and reverse_candidate[2] == source_node_id:
            edges.add(tuple(sorted((source_node_id, target_node_id))))
    return sorted(edges)


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
