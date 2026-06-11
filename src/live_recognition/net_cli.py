from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from live_recognition.cube_net import build_observed_cube_net_from_graph, draw_cube_net
from live_recognition.detectors import YOLOCubeDetector
from live_recognition.piece_graph import (
    DEFAULT_PIECE_SEGMENTATION_MODEL,
    YOLOPieceSegmenter,
    build_piece_graph,
    build_sticker_adjacency_graph,
    clipped_square_roi,
    draw_piece_graph,
)


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
        "--piece-model",
        default=DEFAULT_PIECE_SEGMENTATION_MODEL,
        help="YOLO piece segmentation model weights path",
    )
    parser.add_argument("--bbox-imgsz", type=int, default=640, help="BBox YOLO inference image size")
    parser.add_argument("--piece-imgsz", type=int, default=256, help="Piece YOLO inference image size")
    parser.add_argument("--bbox-conf", type=float, default=0.25, help="Cube bbox confidence threshold")
    parser.add_argument(
        "--piece-conf",
        type=float,
        default=0.8,
        help="Piece segmentation confidence threshold",
    )
    parser.add_argument(
        "--adjacency-radius",
        type=int,
        default=20,
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
        default=0.08,
        help="Padding ratio applied to the detected cube bbox before piece segmentation",
    )
    parser.add_argument(
        "--segmentation-padding-px",
        type=int,
        default=0,
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

    cube_detector = YOLOCubeDetector(
        model_name=args.bbox_model,
        confidence_threshold=args.bbox_conf,
        image_size=args.bbox_imgsz,
    )
    piece_segmenter = YOLOPieceSegmenter(
        model_name=args.piece_model,
        confidence_threshold=args.piece_conf,
        image_size=args.piece_imgsz,
    )

    cube_candidate = cube_detector.detect(frame)
    if cube_candidate is None:
        raise SystemExit("Could not build net because no cube was detected")

    height, width = frame.shape[:2]
    roi_xyxy = clipped_square_roi(cube_candidate.bbox_xyxy, width, height, args.roi_padding)
    x1, y1, x2, y2 = roi_xyxy
    roi = frame[y1:y2, x1:x2]
    piece_masks = piece_segmenter.segment_roi(
        roi,
        frame_shape=(height, width),
        roi_xyxy=roi_xyxy,
        segmentation_padding_px=args.segmentation_padding_px,
    )
    segmented_graph = build_piece_graph(
        frame,
        piece_masks,
        adjacency_radius_px=args.adjacency_radius,
        min_component_area_px=args.min_sticker_area,
    )
    graph = build_sticker_adjacency_graph(
        segmented_graph.nodes,
        adjacency_radius_px=args.adjacency_radius,
        infer_piece_ids=True,
    )
    net_state = build_observed_cube_net_from_graph(graph)
    net_image = draw_cube_net(net_state)

    if args.save_net:
        args.save_net.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_net), net_image)

    graph_image = None
    if args.save_graph or args.display:
        graph_image = draw_piece_graph(
            frame,
            graph,
            cube_candidate=cube_candidate,
            roi_xyxy=roi_xyxy,
            piece_masks=piece_masks,
        )
    if args.save_graph and graph_image is not None:
        args.save_graph.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_graph), graph_image)

    print(
        f"Built net with {len(net_state.cells)} observed cells from "
        f"{len(graph.nodes)} sticker nodes and {len(graph.edges)} graph edges."
    )

    if args.display:
        if graph_image is not None:
            cv2.imshow("cube sticker graph", graph_image)
        cv2.imshow("cube net", net_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
