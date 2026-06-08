from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from live_recognition.detectors import YOLOCubeDetector
from live_recognition.piece_graph import (
    DEFAULT_PIECE_SEGMENTATION_MODEL,
    YOLOPieceSegmenter,
    build_piece_graph,
    clipped_square_roi,
    draw_piece_graph,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and visualize a Rubik's Cube piece adjacency graph"
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
        default=0.25,
        help="Piece segmentation confidence threshold",
    )
    parser.add_argument(
        "--adjacency-radius",
        type=int,
        default=20,
        help="Pixel radius for mask-contact adjacency",
    )
    parser.add_argument(
        "--min-sticker-area",
        type=int,
        default=12,
        help="Minimum connected-component area kept as a sticker node",
    )
    parser.add_argument(
        "--min-sticker-segment-ratio",
        type=float,
        default=0.25,
        help=(
            "Merge RGBXY sticker segments smaller than this fraction of the largest "
            "segment before graph node creation; 0 disables"
        ),
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
        default=-2,
        help="Pixel padding applied to each piece mask before sticker splitting",
    )
    parser.add_argument("--save", type=Path, help="Optional output graph overlay image path")
    parser.add_argument("--display", action="store_true", help="Show an OpenCV preview window")
    parser.add_argument(
        "--no-mask-overlay",
        action="store_true",
        help="Only draw graph nodes and edges, without translucent piece masks",
    )
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
        raise SystemExit("Could not build graph because no cube was detected")

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
    graph = build_piece_graph(
        frame,
        piece_masks,
        adjacency_radius_px=args.adjacency_radius,
        min_component_area_px=args.min_sticker_area,
        min_sticker_segment_ratio=args.min_sticker_segment_ratio,
    )
    output = draw_piece_graph(
        frame,
        graph,
        cube_candidate=cube_candidate,
        roi_xyxy=roi_xyxy,
        piece_masks=piece_masks,
        draw_masks=not args.no_mask_overlay,
    )

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save), output)
        print(
            f"Saved {args.save} with {len(graph.nodes)} sticker nodes, "
            f"{len(graph.edges)} graph edges, and {len(piece_masks)} segmented pieces."
        )
    else:
        print(
            f"Built graph with {len(graph.nodes)} sticker nodes, "
            f"{len(graph.edges)} graph edges, and {len(piece_masks)} segmented pieces."
        )

    if args.display:
        cv2.imshow("cube piece graph", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
