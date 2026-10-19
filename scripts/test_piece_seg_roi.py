#!/usr/bin/env python3
"""Test ordered model keypoints followed by direct solvePnP on the ROI.

The old pose-graph matching search has been removed. The ordered model now
supplies a direct index-to-3D mapping using the keypoint names from Blender's
``ordered_keypoint_symmetries.json``:

1. Acquire the cube ROI with the bbox detector.
2. Run the ordered 26-point pose model inside that ROI.
3. Keep model keypoints above ``--min-keypoint-conf``.
4. Pass those same ordered 2D/3D pairs directly to solvePnP.
5. Use the model's keypoint bounds to track the next ROI.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from live_recognition.detectors import YOLOCubeDetector  # noqa: E402
from live_recognition.roi_tracking import RoiTracker, roi_for_frame  # noqa: E402
from render_yolo11n_ordered_cube_pose import (  # noqa: E402
    DEFAULT_BBOX_MODEL,
    DEFAULT_POSE_MODEL,
    DEFAULT_SYMMETRY_MANIFEST,
    draw_ordered_pose,
    is_image_source,
    load_keypoint_layout,
    local_to_frame_points,
    parse_source,
    point_confidence,
    print_keypoint_confidences,
    register_training_checkpoint_classes,
    result_keypoints,
    resize_to_square,
    select_detection_index,
    tracking_bounds,
)


CUBE_SIZE_MM = 55.0
CUBE_HALF_SIZE_MM = CUBE_SIZE_MM / 2.0
FACE_AXES = {
    "U": np.asarray((0.0, 0.0, 1.0), dtype=np.float32),
    "D": np.asarray((0.0, 0.0, -1.0), dtype=np.float32),
    "F": np.asarray((0.0, -1.0, 0.0), dtype=np.float32),
    "B": np.asarray((0.0, 1.0, 0.0), dtype=np.float32),
    "R": np.asarray((1.0, 0.0, 0.0), dtype=np.float32),
    "L": np.asarray((-1.0, 0.0, 0.0), dtype=np.float32),
}
WINDOW_NAME = "ordered cube pose + solvePnP"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
CUBE_CORNER_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


@dataclass
class OrderedPnPResult:
    used_indices: tuple[int, ...]
    image_points: np.ndarray
    projected_points: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    mean_error: float
    max_error: float
    inlier_count: int


def ordered_object_points(keypoint_names: tuple[str, ...]) -> np.ndarray:
    """Build Blender-frame 3D points from the manifest's ordered names."""

    points = []
    for name in keypoint_names:
        try:
            point = sum((FACE_AXES[face] for face in name), start=np.zeros(3, dtype=np.float32))
        except KeyError as error:
            raise ValueError(f"Unknown face letter in Blender keypoint {name!r}") from error
        points.append(point * CUBE_HALF_SIZE_MM)
    return np.asarray(points, dtype=np.float32)


def camera_matrix_for_frame(frame_shape: tuple[int, ...], focal_scale: float) -> np.ndarray:
    height, width = frame_shape[:2]
    focal = max(width, height) * float(focal_scale)
    return np.asarray(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def solve_ordered_pnp(
    frame_points: np.ndarray,
    confidences: np.ndarray | None,
    keypoint_names: tuple[str, ...],
    frame_shape: tuple[int, ...],
    args: argparse.Namespace,
) -> OrderedPnPResult | None:
    object_points_all = ordered_object_points(keypoint_names)
    height, width = frame_shape[:2]
    used_indices = tuple(
        index
        for index, point in enumerate(frame_points)
        if point_confidence(confidences, index) >= args.keypoint_conf
        and np.isfinite(point).all()
        and 0 <= float(point[0]) < width
        and 0 <= float(point[1]) < height
    )
    if len(used_indices) < 4:
        return None

    image_points = np.asarray(frame_points[list(used_indices)], dtype=np.float32)
    object_points = object_points_all[list(used_indices)]
    camera_matrix = camera_matrix_for_frame(frame_shape, args.pnp_focal_scale)
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_SQPNP,
            iterationsCount=args.pnp_iterations,
            reprojectionError=args.pnp_reprojection_error,
            confidence=0.99,
        )
    except cv2.error:
        return None
    if not ok:
        return None

    inlier_indices = (
        np.asarray(inliers, dtype=np.int32).reshape(-1)
        if inliers is not None
        else np.arange(len(object_points), dtype=np.int32)
    )
    refinement_object_points = object_points[inlier_indices]
    refinement_image_points = image_points[inlier_indices]
    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            refinement_object_points,
            refinement_image_points,
            camera_matrix,
            dist_coeffs,
            np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        )
    except cv2.error:
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

    projected_all, _ = cv2.projectPoints(
        object_points_all,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    projected_all = projected_all.reshape(-1, 2)
    errors = np.linalg.norm(projected_all[list(used_indices)] - image_points, axis=1)
    return OrderedPnPResult(
        used_indices=used_indices,
        image_points=image_points,
        projected_points=projected_all,
        rvec=rvec,
        tvec=tvec,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        mean_error=float(np.mean(errors)),
        max_error=float(np.max(errors)),
        inlier_count=0 if inliers is None else int(len(inliers)),
    )


def draw_pnp_result(
    frame: np.ndarray,
    result: OrderedPnPResult | None,
    keypoint_names: tuple[str, ...],
) -> None:
    if result is None:
        cv2.putText(
            frame,
            "solvePnP: insufficient usable keypoints",
            (12, frame.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return

    for start_index, end_index in CUBE_CORNER_EDGES:
        start = tuple(np.round(result.projected_points[start_index]).astype(int))
        end = tuple(np.round(result.projected_points[end_index]).astype(int))
        cv2.line(frame, start, end, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.line(frame, start, end, (255, 0, 255), 3, cv2.LINE_AA)

    label = (
        f"solvePnP {len(result.used_indices)}/{len(keypoint_names)} pts "
        f"inliers {result.inlier_count} err {result.mean_error:.1f}px "
        f"max {result.max_error:.1f}px"
    )
    cv2.putText(
        frame,
        label,
        (12, frame.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        (12, frame.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )


def process_frame(
    frame: np.ndarray,
    bbox_detector: YOLOCubeDetector,
    pose_model,
    roi_tracker: RoiTracker,
    keypoint_names: tuple[str, ...],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray | None, str, int]:
    output = frame.copy()
    height, width = frame.shape[:2]
    roi_xyxy, candidate, roi_source = roi_for_frame(
        frame,
        bbox_detector,
        roi_tracker,
        args.roi_padding,
    )
    if roi_xyxy is None:
        cv2.putText(output, "no cube bbox", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        return output, None, "none", 0

    x1, y1, x2, y2 = roi_xyxy
    roi = frame[y1:y2, x1:x2]
    model_roi = resize_to_square(roi, args.view_size)
    predict_kwargs = {} if args.device is None else {"device": args.device}
    results = pose_model.predict(
        source=model_roi,
        conf=args.pose_conf,
        imgsz=args.imgsz,
        max_det=1,
        verbose=False,
        **predict_kwargs,
    )
    result = results[0] if results else None
    points_by_detection, confidences_by_detection = (
        result_keypoints(result) if result is not None else (None, None)
    )
    detection_index = (
        select_detection_index(result, len(points_by_detection))
        if points_by_detection is not None
        else None
    )
    if detection_index is None:
        roi_tracker.reset()
        cv2.rectangle(output, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), 2)
        cv2.putText(output, "no ordered pose", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        return output, None, roi_source, 0

    local_points = np.asarray(points_by_detection[detection_index], dtype=np.float32)
    if local_points.shape != (len(keypoint_names), 2):
        raise RuntimeError(
            f"Expected {len(keypoint_names)} ordered keypoints, got {local_points.shape}"
        )
    confidences = (
        None
        if confidences_by_detection is None
        else np.asarray(confidences_by_detection[detection_index], dtype=np.float32).reshape(-1)
    )
    frame_points, local_roi_points = local_to_frame_points(
        local_points,
        roi_xyxy,
        args.view_size,
    )
    output, point_count = draw_ordered_pose(
        output,
        frame_points,
        confidences,
        keypoint_names,
        args.keypoint_conf,
        args.show_labels,
        args.show_confidences,
    )
    pnp_result = solve_ordered_pnp(
        frame_points,
        confidences,
        keypoint_names,
        frame.shape,
        args,
    )
    draw_pnp_result(output, pnp_result, keypoint_names)

    roi_tracker.observe_tracking_bounds(
        tracking_bounds(local_roi_points, confidences, args.track_conf, roi.shape[:2]),
        width,
        height,
        tracking_gain=args.roi_tracking_gain,
        tracking_padding=args.roi_tracking_padding,
    )
    roi_color = (0, 0, 255) if roi_source == "detector" else (40, 220, 40)
    cv2.rectangle(output, (x1, y1), (x2 - 1, y2 - 1), roi_color, 2)
    label = f"{roi_source} ROI | {point_count}/{len(keypoint_names)} points"
    if candidate is not None:
        label += f" | bbox {candidate.confidence:.2f}"
    cv2.putText(output, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, roi_color, 2, cv2.LINE_AA)
    return output, confidences, roi_source, point_count


def create_video_writer(path: Path, frame: np.ndarray, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = "mp4v" if path.suffix.lower() in {".mp4", ".m4v", ".mov"} else "MJPG"
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*fourcc),
        fps if fps > 0.0 else 30.0,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ordered pose keypoints through direct solvePnP")
    parser.add_argument("--source", required=True, help="Image, video path, or camera index")
    parser.add_argument("--model", "--checkpoint", dest="model", type=Path, default=DEFAULT_POSE_MODEL)
    parser.add_argument("--bbox-model", type=Path, default=DEFAULT_BBOX_MODEL)
    parser.add_argument("--symmetry-manifest", type=Path, default=DEFAULT_SYMMETRY_MANIFEST)
    parser.add_argument("--device", default=None)
    parser.add_argument("--bbox-conf", type=float, default=0.25)
    parser.add_argument("--bbox-imgsz", type=int, default=640)
    parser.add_argument("--pose-conf", type=float, default=0.001)
    parser.add_argument(
        "--keypoint-conf",
        "--min-keypoint-conf",
        "--keypoint-confidence",
        "--min-keypoint-confidence",
        dest="keypoint_conf",
        type=float,
        default=0.05,
        help="Minimum confidence for drawing and solvePnP input keypoints",
    )
    parser.add_argument("--track-conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--view-size", type=int, default=256)
    parser.add_argument("--roi-padding", type=float, default=0.08)
    parser.add_argument("--roi-tracking-gain", type=float, default=1.4)
    parser.add_argument("--roi-tracking-padding", type=float, default=0.18)
    parser.add_argument("--pnp-focal-scale", type=float, default=1.2)
    parser.add_argument("--pnp-iterations", type=int, default=200)
    parser.add_argument("--pnp-reprojection-error", type=float, default=8.0)
    parser.add_argument("--show-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-confidences", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-keypoint-confidences", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--save-video", type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.set_defaults(display=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for path, label in (
        (args.model, "ordered pose model"),
        (args.bbox_model, "bbox model"),
        (args.symmetry_manifest, "Blender symmetry manifest"),
    ):
        if not path.is_file():
            raise SystemExit(f"Could not find {label}: {path}")

    keypoint_names = load_keypoint_layout(args.symmetry_manifest)
    register_training_checkpoint_classes()
    from ultralytics import YOLO

    bbox_detector = YOLOCubeDetector(
        model_name=str(args.bbox_model),
        confidence_threshold=args.bbox_conf,
        image_size=args.bbox_imgsz,
    )
    pose_model = YOLO(str(args.model))
    roi_tracker = RoiTracker()
    source = parse_source(args.source)
    print_confidences = args.print_keypoint_confidences

    if is_image_source(source):
        frame = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if frame is None:
            raise SystemExit(f"Could not read image: {source}")
        output, confidences, roi_source, point_count = process_frame(
            frame, bbox_detector, pose_model, roi_tracker, keypoint_names, args
        )
        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.save), output):
                raise SystemExit(f"Could not write render: {args.save}")
            print(f"Saved render: {args.save}")
        if args.display:
            cv2.imshow(WINDOW_NAME, output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if print_confidences is not False:
            print_keypoint_confidences(keypoint_names, confidences)
        print(f"ROI source: {roi_source}; rendered points: {point_count}/{len(keypoint_names)}")
        return 0

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {source}")
    writer = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            output, confidences, _roi_source, _point_count = process_frame(
                frame, bbox_detector, pose_model, roi_tracker, keypoint_names, args
            )
            if args.save_video:
                if writer is None:
                    writer = create_video_writer(args.save_video, output, float(capture.get(cv2.CAP_PROP_FPS)))
                writer.write(output)
            if args.save and args.once:
                args.save.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(args.save), output)
            if print_confidences is True or (print_confidences is None and args.once):
                print_keypoint_confidences(keypoint_names, confidences)
            if args.display:
                cv2.imshow(WINDOW_NAME, output)
                key = cv2.waitKey(30 if args.once else 1) & 0xFF
                if args.once or key in (ord("q"), 27, 13):
                    break
            elif args.once:
                break
    finally:
        if writer is not None:
            writer.release()
        capture.release()
        if args.display:
            cv2.destroyAllWindows()
    if writer is not None:
        print(f"Saved video: {args.save_video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
