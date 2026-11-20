from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from live_recognition.cube_net import build_observed_cube_net_from_graph, draw_cube_net
from live_recognition.detectors import YOLOCubeDetector
from live_recognition.piece_graph import (
    PieceGraph as StickerGraph,
    StickerNode,
    build_sticker_adjacency_graph,
    clipped_square_roi,
    draw_piece_graph as draw_sticker_graph,
)


DEFAULT_STICKER_SEGMENTATION_MODEL = (
    "runs/segment/runs/cube_piece_seg_sticker/runs/cube_piece_yolo11n_seg/weights/best.pt"
)
STICKER_TYPE_COLORS = {
    "center": (80, 220, 255),
    "edge": (80, 255, 120),
    "corner": (255, 120, 80),
}
STICKER_CLASS_NAMES = {"center_sticker", "edge_sticker", "corner_sticker"}


@dataclass(frozen=True)
class StickerMask:
    sticker_id: int
    class_id: int
    sticker_type: str
    confidence: float
    mask: np.ndarray
    roi_mask: np.ndarray
    roi_xyxy: tuple[int, int, int, int]


@dataclass(frozen=True)
class StickerSegmentation:
    masks: list[StickerMask]
    nodes: list[StickerNode]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a raw observed-color Rubik's Cube net from one cube image"
    )
    parser.add_argument("--source", required=True, help="Image path")
    parser.add_argument(
        "--bbox-model",
        default="runs/500ExamplesWeights.pt",
        help="YOLO bounding-box model weights path",
    )
    parser.add_argument(
        "--sticker-model",
        default=DEFAULT_STICKER_SEGMENTATION_MODEL,
        help="YOLO sticker segmentation model weights path",
    )
    parser.add_argument("--bbox-imgsz", type=int, default=640, help="BBox YOLO inference image size")
    parser.add_argument("--sticker-imgsz", type=int, default=256, help="Sticker YOLO inference image size")
    parser.add_argument("--bbox-conf", type=float, default=0.25, help="Cube bbox confidence threshold")
    parser.add_argument(
        "--sticker-conf",
        type=float,
        default=0.25,
        help="Sticker segmentation confidence threshold",
    )
    parser.add_argument("--piece-conf", type=float, dest="sticker_conf", help=argparse.SUPPRESS)
    parser.add_argument(
        "--adjacency-radius",
        type=int,
        default=0,
        help="Pixel radius for sticker adjacency",
    )
    parser.add_argument(
        "--min-sticker-area",
        type=int,
        default=12,
        help="Minimum sticker mask area kept as a graph node",
    )
    parser.add_argument(
        "--roi-padding",
        type=float,
        default=0.0,
        help="Padding ratio applied to the detected cube bbox before sticker segmentation",
    )
    parser.add_argument(
        "--segmentation-padding-px",
        type=int,
        default=-2,
        help="Pixel padding applied to each sticker mask before color sampling",
    )
    parser.add_argument("--save-net", type=Path, help="Optional output net image path")
    parser.add_argument("--save-graph", type=Path, help="Optional output graph overlay path")
    parser.add_argument("--display", action="store_true", help="Show OpenCV preview windows")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    frame = cv2.imread(args.source)
    if frame is None:
        raise SystemExit(f"Could not read image: {args.source}")

    from ultralytics import YOLO

    cube_detector = YOLOCubeDetector(
        model_name=args.bbox_model,
        confidence_threshold=args.bbox_conf,
        image_size=args.bbox_imgsz,
    )
    sticker_model = YOLO(str(args.sticker_model))

    height, width = frame.shape[:2]
    cube_candidate = cube_detector.detect(frame)
    if cube_candidate is None:
        raise SystemExit("Could not build net because no cube was detected")
    roi_xyxy = clipped_square_roi(cube_candidate.bbox_xyxy, width, height, args.roi_padding)

    x1, y1, x2, y2 = roi_xyxy
    roi = frame[y1:y2, x1:x2]
    sticker_segmentation = segment_stickers(
        sticker_model,
        roi,
        frame_shape=(height, width),
        roi_xyxy=roi_xyxy,
        confidence_threshold=args.sticker_conf,
        image_size=args.sticker_imgsz,
        segmentation_padding_px=args.segmentation_padding_px,
        min_sticker_area_px=args.min_sticker_area,
    )
    sticker_masks = sticker_segmentation.masks
    net_graph = build_sticker_adjacency_graph(
        sticker_segmentation.nodes,
        adjacency_radius_px=args.adjacency_radius,
        infer_piece_ids=False,
    )
    net_state = build_observed_cube_net_from_graph(net_graph)
    net_image = draw_cube_net(net_state)

    if args.save_net:
        args.save_net.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_net), net_image)

    graph_image = None
    if args.save_graph or args.display:
        display_graph = build_sticker_adjacency_graph(
            sticker_segmentation.nodes,
            adjacency_radius_px=args.adjacency_radius,
            infer_piece_ids=False,
        )
        graph_image = draw_sticker_segmentation_graph(
            frame,
            sticker_masks,
            display_graph,
            cube_candidate=cube_candidate,
            roi_xyxy=roi_xyxy,
        )
    if args.save_graph and graph_image is not None:
        args.save_graph.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_graph), graph_image)

    print(
        f"Built net with {len(net_state.cells)} observed cells from "
        f"{len(net_graph.nodes)} sticker nodes, {len(net_graph.edges)} graph edges, "
        f"and {len(sticker_masks)} segmented stickers."
    )

    if args.display:
        if graph_image is not None:
            cv2.imshow("cube sticker graph", graph_image)
        cv2.imshow("cube net", net_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def draw_sticker_segmentation_graph(
    frame_bgr: np.ndarray,
    sticker_masks: list[StickerMask],
    graph: StickerGraph,
    *,
    cube_candidate=None,
    roi_xyxy: tuple[int, int, int, int],
    mask_opacity: float = 1.0,
    show_bbox: bool = True,
    show_confidence: bool = True,
) -> np.ndarray:
    output = frame_bgr.copy()
    overlay = output.copy()
    confidence_labels = []
    for sticker_mask in sticker_masks:
        x1, y1, _, _ = sticker_mask.roi_xyxy
        roi_overlay = overlay[
            y1 : y1 + sticker_mask.roi_mask.shape[0],
            x1 : x1 + sticker_mask.roi_mask.shape[1],
        ]
        roi_overlay[sticker_mask.roi_mask] = STICKER_TYPE_COLORS.get(
            sticker_mask.sticker_type,
            (255, 255, 255),
        )

        ys, xs = np.nonzero(sticker_mask.roi_mask)
        if len(xs) > 0:
            confidence_labels.append(
                (
                    (int(xs.mean()) + x1, int(ys.mean()) + y1),
                    sticker_mask.confidence,
                )
            )

    output = cv2.addWeighted(output, 1.0 - mask_opacity, overlay, mask_opacity, 0)
    output = draw_sticker_graph(
        output,
        graph,
        roi_xyxy=roi_xyxy,
        draw_masks=False,
    )
    if show_bbox:
        x1, y1, x2, y2 = roi_xyxy
        cv2.rectangle(output, (x1, y1), (x2, y2), (40, 220, 40), 2)
    if show_confidence:
        for label_at, confidence in confidence_labels:
            cv2.putText(
                output,
                f"{confidence:.2f}",
                label_at,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                output,
                f"{confidence:.2f}",
                label_at,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
    return output


def segment_stickers(
    sticker_model,
    roi_bgr: np.ndarray,
    *,
    frame_shape: tuple[int, int],
    roi_xyxy: tuple[int, int, int, int],
    confidence_threshold: float,
    image_size: int,
    segmentation_padding_px: int,
    min_sticker_area_px: int = 12,
) -> StickerSegmentation:
    _ = segmentation_padding_px
    x1, y1, x2, y2 = roi_xyxy
    roi_height = y2 - y1
    roi_width = x2 - x1
    results = sticker_model.predict(
        source=roi_bgr,
        conf=confidence_threshold,
        imgsz=image_size,
        verbose=False,
    )
    if not results:
        return StickerSegmentation(masks=[], nodes=[])

    result = results[0]
    masks = result.masks
    boxes = result.boxes
    if masks is None or boxes is None:
        return StickerSegmentation(masks=[], nodes=[])

    frame_height, frame_width = frame_shape
    mask_data = masks.data.detach().cpu().numpy()
    sticker_masks = []
    sticker_nodes = []
    for index in range(len(boxes)):
        class_id = int(boxes.cls[index].item())
        class_name = _class_name_for_id(result, sticker_model, class_id)
        if class_name not in STICKER_CLASS_NAMES:
            continue

        confidence = float(boxes.conf[index].item())
        if confidence < confidence_threshold:
            continue

        roi_mask = _sticker_mask_bool(mask_data, index, roi_width, roi_height)
        if not bool(roi_mask.any()):
            continue

        mask_area = int(roi_mask.sum())
        sticker_id = len(sticker_masks)
        full_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = roi_mask.astype(np.uint8) * 255
        sticker_masks.append(
            StickerMask(
                sticker_id=sticker_id,
                class_id=class_id,
                sticker_type=_sticker_type(class_name),
                confidence=confidence,
                mask=full_mask,
                roi_mask=roi_mask,
                roi_xyxy=roi_xyxy,
            )
        )
        if mask_area < min_sticker_area_px:
            continue

        ys, xs = np.nonzero(roi_mask)
        mean_bgr = (
            roi_bgr[roi_mask]
            .astype(np.float32)
            .mean(axis=0)
            .clip(0, 255)
            .astype(np.uint8)
        )
        sticker_nodes.append(
            StickerNode(
                node_id=len(sticker_nodes),
                piece_id=sticker_id,
                piece_type=_sticker_type(class_name),
                confidence=confidence,
                centroid_xy=(float(xs.mean() + x1), float(ys.mean() + y1)),
                mask=full_mask,
                mean_bgr=tuple(int(channel) for channel in mean_bgr),
            )
        )
    return StickerSegmentation(masks=sticker_masks, nodes=sticker_nodes)


def _class_name_for_id(result, sticker_model, class_id: int) -> str:
    names = getattr(result, "names", None) or getattr(sticker_model, "names", {}) or {}
    return str(names.get(class_id, class_id)).lower()


def _sticker_type(class_name: str) -> str:
    return class_name.removesuffix("_sticker")


def _sticker_mask_bool(
    mask_data: np.ndarray,
    index: int,
    roi_width: int,
    roi_height: int,
) -> np.ndarray:
    mask = cv2.resize(
        mask_data[index].astype(np.float32),
        (roi_width, roi_height),
        interpolation=cv2.INTER_LINEAR,
    )
    return mask > 0.5


if __name__ == "__main__":
    raise SystemExit(main())
