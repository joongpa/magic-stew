#!/usr/bin/env python3
"""Render 18-class four-point stickers with the shared bbox -> square ROI flow.

Object/class scores are result.boxes.conf. The separate cube detector has its
own score; result.keypoints.conf contains presence scores, not localization
confidence. No pose fitting is applied to these raw neural predictions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from live_recognition.detectors import YOLOCubeDetector
from live_recognition.roi_tracking import RoiTracker, roi_for_frame
from live_recognition.sticker_quad_schema import CLASS_NAMES, COLORS, TYPES
from scripts.render_rfdetr_keypoints import as_numpy, parse_source, resolve_checkpoint

DEFAULT_MODEL = ROOT / "runs/pose/runs/sticker_quads/weights/best.pt"
DEFAULT_BBOX_MODEL = ROOT / "runs/500ExamplesWeights.pt"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
COLORS_BGR = (
    (255, 255, 255),
    (0, 238, 245),
    (0, 146, 242),
    (1, 32, 231),
    (227, 177, 101),
    (1, 227, 80),
)
TYPE_OUTLINES_BGR = ((255, 255, 255), (0, 255, 255), (255, 120, 255))
WINDOW = "sticker quads"


def draw_label(image, text, x, y, color):
    """Compact captions keep neighboring sticker scores legible."""
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.38
    width, height = cv2.getTextSize(text, font, scale, 1)[0]
    x = max(0, min(x + 3, image.shape[1] - width - 1))
    y = max(height + 2, min(y - 3, image.shape[0] - 2))
    cv2.putText(image, text, (x, y), font, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), font, scale, color, 1, cv2.LINE_AA)


def parse_class_conf(value):
    """Accept individual classes, color/type groups, or numeric class IDs."""
    overrides = {}
    for item in value.split(","):
        try:
            key, raw = item.rsplit("=", 1)
            if key.strip().startswith("-") and key.strip()[1:].isdigit():
                raise ValueError
            key = " ".join(key.lower().replace("_", " ").replace("-", " ").split())
            score = float(raw)
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError
            if key in CLASS_NAMES:
                ids = [CLASS_NAMES.index(key)]
            elif key in COLORS:
                ids = [i for i, name in enumerate(CLASS_NAMES) if name.startswith(key + " ")]
            elif key in TYPES:
                ids = [i for i, name in enumerate(CLASS_NAMES) if name.endswith(" " + key)]
            elif key.isdigit() and 0 <= int(key) < 18:
                ids = [int(key)]
            else:
                raise ValueError
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "Use class=value, color=value, or type=value, e.g. "
                "blue_corner=0.8,edge=0.5,white=0.4 (scores in [0,1])"
            ) from exc
        overrides.update(dict.fromkeys(ids, score))
    return overrides


def extract_quads(result):
    """Read all four points, keeping object scores and presence separate."""
    boxes, keypoints = getattr(result, "boxes", None), getattr(result, "keypoints", None)
    if boxes is None or keypoints is None:
        return []
    points = as_numpy(keypoints.xy)
    if points is None or not len(points):
        return []
    if points.ndim != 3 or points.shape[1:] != (4, 2):
        raise ValueError(f"Expected four-keypoint sticker outputs, got {points.shape}")
    presence = as_numpy(keypoints.conf)
    presence = np.ones(points.shape[:2], dtype=np.float32) if presence is None else presence
    scores, classes = as_numpy(boxes.conf), as_numpy(boxes.cls)
    if scores is None or classes is None or presence.shape != points.shape[:2]:
        raise ValueError("Malformed sticker confidence/presence tensors")
    if len(scores) != len(points) or len(classes) != len(points):
        raise ValueError("Sticker boxes and keypoints have different lengths")
    result_names = result.names
    detections = []
    for p, v, score, cls in zip(points, presence, scores, classes):
        name = result_names[int(cls)]
        if name not in CLASS_NAMES:
            raise ValueError(f"Unexpected sticker class: {name}")
        if not np.isfinite(score):
            continue
        class_id = CLASS_NAMES.index(name)
        detections.append(
            {
                "class_id": class_id,
                "class_name": name,
                "object_confidence": float(score),
                "points": p,
                "keypoint_presence": v,
            }
        )
    return detections


def inference_conf(args):
    return min(args.pose_conf, args.track_conf, *args.pose_class_conf.values())


def draw_quad(output, detection, points, valid, args):
    class_id = detection["class_id"]
    sticker_color = COLORS_BGR[class_id // 3]
    color = TYPE_OUTLINES_BGR[class_id % 3]
    rounded = np.rint(np.clip(np.where(np.isfinite(points), points, 0), -1e6, 1e6)).astype(np.int32)
    if args.show_outlines:
        for i in range(4):
            j = (i + 1) % 4
            if valid[i] and valid[j]:
                cv2.line(output, tuple(rounded[i]), tuple(rounded[j]), (0, 0, 0), 3, cv2.LINE_AA)
                cv2.line(output, tuple(rounded[i]), tuple(rounded[j]), color, 1, cv2.LINE_AA)
    for i, point in enumerate(rounded):
        if not valid[i]:
            continue
        x, y = map(int, point)
        if args.show_keypoints:
            if i == 0:
                for radius, fill in [(5, (0, 0, 0)), (3, color)]:
                    diamond = np.asarray(
                        [(x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)]
                    )
                    cv2.fillPoly(output, [diamond], fill, cv2.LINE_AA)
            else:
                cv2.circle(output, (x, y), 4, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(output, (x, y), 2, color, -1, cv2.LINE_AA)
        text = str(i) if args.show_point_indices else ""
        if args.show_keypoint_presence:
            text += f" p={detection['keypoint_presence'][i]:.3f}"
        if text:
            draw_label(output, text.strip(), x, y, color)
    parts = []
    if args.show_labels:
        kind = ("C", "E", "CTR")[class_id % 3]
        parts.append(f"{COLORS[class_id // 3][0].upper()}/{kind}")
    if args.show_confidences:
        parts.append(f"{detection['object_confidence']:.2f}")
    if valid.any():
        center = np.rint(points[valid].mean(0)).astype(int)
        label_y = int(center[1])
        if args.show_color_initials:
            letter = COLORS[class_id // 3][0].upper()
            extent = float(np.ptp(points[valid], axis=0).min())
            scale = float(np.clip(extent / 85, 0.3, 0.85))
            font = cv2.FONT_HERSHEY_SIMPLEX
            width, height = cv2.getTextSize(letter, font, scale, 2)[0]
            x = max(0, min(int(center[0] - width / 2), output.shape[1] - width - 1))
            y = max(height, min(int(center[1] + height / 2), output.shape[0] - 1))
            cv2.putText(output, letter, (x, y), font, scale, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(output, letter, (x, y), font, scale, sticker_color, 2, cv2.LINE_AA)
            label_y = y + 15
        if parts:
            draw_label(output, " ".join(parts), int(center[0]), label_y, color)


def render_frame(
    frame,
    bbox_detector,
    model,
    tracker,
    args,
    frame_index=0,
    *,
    extractor=extract_quads,
    drawer=draw_quad,
):
    start = perf_counter()
    if args.bbox_every and frame_index % args.bbox_every == 0:
        tracker.reset()
    roi, candidate, source = roi_for_frame(frame, bbox_detector, tracker, args.roi_padding)
    if args.full_frame:
        output = frame.copy()
    else:
        output = np.zeros((args.view_size, args.view_size, 3), dtype=np.uint8)
    report = {
        "frame": frame_index,
        "roi_xyxy": roi,
        "roi_source": source,
        "cube_bbox_confidence": float(candidate.confidence) if candidate is not None else None,
        "detections": [],
    }
    if roi is None:
        if args.show_status:
            draw_label(output, "cube not detected", 8, 20, (255, 255, 255))
        return output, report
    x1, y1, x2, y2 = roi
    crop = frame[y1:y2, x1:x2]
    if not args.full_frame:
        output = cv2.resize(crop, (args.view_size, args.view_size), interpolation=cv2.INTER_LINEAR)
    predictions = model.predict(
        source=crop,
        imgsz=args.imgsz,
        conf=inference_conf(args),
        iou=args.iou,
        max_det=args.max_det,
        verbose=False,
        device=args.device,
    )
    detections = extractor(predictions[0]) if predictions else []
    tracking, weak_tracking = [], []
    for d in detections:
        p, presence, score = d["points"], d["keypoint_presence"], d["object_confidence"]
        valid = (
            np.isfinite(p).all(-1)
            & np.isfinite(presence)
            & (presence >= args.keypoint_conf)
            & (p[:, 0] >= 0)
            & (p[:, 0] < crop.shape[1])
            & (p[:, 1] >= 0)
            & (p[:, 1] < crop.shape[0])
        )
        if score >= args.track_conf and valid.any():
            weak_tracking.extend(p[valid])
        threshold = args.pose_class_conf.get(d["class_id"], args.pose_conf)
        if score < threshold or not valid.any():
            continue
        if score >= args.track_conf:
            tracking.extend(p[valid])
        projected = (
            p + [x1, y1]
            if args.full_frame
            else p * [args.view_size / crop.shape[1], args.view_size / crop.shape[0]]
        )
        drawer(output, d, projected, valid, args)
        report["detections"].append(
            {
                "class_id": d["class_id"],
                "class_name": d["class_name"],
                "object_confidence": score,
                "keypoints_full_frame": [
                    [float(v) if np.isfinite(v) else None for v in point] for point in p + [x1, y1]
                ],
                "keypoint_presence": [float(v) if np.isfinite(v) else None for v in presence],
                "displayed_points": valid.tolist(),
            }
        )
    bounds = np.asarray(tracking or weak_tracking)
    if len(bounds):
        tracker.observe_tracking_bounds(
            (
                int(np.floor(bounds[:, 0].min())),
                int(np.floor(bounds[:, 1].min())),
                int(np.ceil(bounds[:, 0].max())),
                int(np.ceil(bounds[:, 1].max())),
            ),
            frame.shape[1],
            frame.shape[0],
            args.roi_tracking_gain,
            args.roi_tracking_padding,
        )
    else:
        tracker.reset()
    if args.full_frame and args.show_roi:
        cv2.rectangle(output, (x1, y1), (x2 - 1, y2 - 1), (255, 0, 255), 1, cv2.LINE_AA)
    report["elapsed_ms"] = (perf_counter() - start) * 1000
    if args.show_status:
        text = f"ROI:{source} | objects:{len(report['detections'])}"
        if candidate is not None:
            text += f" | cube bbox(last)={candidate.confidence:.3f}"
        draw_label(output, text, 4, 18, (255, 255, 255))
    if args.show_fps:
        draw_label(
            output,
            f"{1000 / max(report['elapsed_ms'], 1e-6):.1f} fps",
            4,
            output.shape[0] - 12,
            (255, 255, 255),
        )
    return output, report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Image/video path or camera index")
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL, help="Checkpoint .pt or training directory"
    )
    parser.add_argument("--bbox-model", type=Path, default=DEFAULT_BBOX_MODEL)
    parser.add_argument(
        "--device", "--inference-device", default="cpu", help="cpu, mps, or CUDA index (0)"
    )
    parser.add_argument(
        "--bbox-conf", type=float, default=0.25, help="Separate cube detector score threshold"
    )
    parser.add_argument("--bbox-imgsz", type=int, default=640)
    parser.add_argument(
        "--pose-conf",
        "--conf",
        "--confidence",
        dest="pose_conf",
        type=float,
        default=0.25,
        help="Sticker object/class score threshold (boxes.conf), default 0.25",
    )
    parser.add_argument(
        "--pose-class-conf",
        "--class-conf",
        type=parse_class_conf,
        default={},
        help='Override object scores: "blue_corner=0.8,edge=0.5,white=0.4"',
    )
    parser.add_argument(
        "--keypoint-conf",
        "--min-keypoint-conf",
        "--keypoint-presence",
        dest="keypoint_conf",
        type=float,
        default=0.05,
        help="Per-corner PRESENCE threshold; this is not localization confidence",
    )
    parser.add_argument(
        "--track-conf", type=float, default=0.05, help="Sticker object score used for ROI tracking"
    )
    parser.add_argument("--imgsz", "--resolution", type=int, default=128)
    parser.add_argument(
        "--iou", type=float, default=0.7, help="Postprocessing IoU threshold where NMS applies"
    )
    parser.add_argument("--max-det", type=int, default=128)
    parser.add_argument("--roi-padding", type=float, default=0.4)
    parser.add_argument("--roi-tracking-gain", type=float, default=1.4)
    parser.add_argument("--roi-tracking-padding", type=float, default=0.18)
    parser.add_argument(
        "--bbox-every",
        type=int,
        default=0,
        help="Reacquire bbox every N frames; 0 reacquires only after a tracking miss",
    )
    parser.add_argument(
        "--view-size",
        type=int,
        default=512,
        help="Crop output size with --no-full-frame; independent of neural imgsz",
    )
    for name, default, help_text in [
        (
            "full-frame",
            True,
            "Overlay on the original frame (default); --no-full-frame shows the ROI crop",
        ),
        ("show-outlines", True, "Type outlines: corner=white, edge=yellow, center=purple"),
        ("show-keypoints", True, "Draw all qualifying points; point 0 is a diamond"),
        ("show-point-indices", False, "Label corner indices 0..3"),
        ("show-labels", False, "Display compact color/type labels: W/C, B/E, G/CTR"),
        ("show-confidences", True, "Display one sticker object/class score per quad"),
        ("show-keypoint-presence", False, "Display per-corner PRESENCE scores with p= prefix"),
        ("show-color-initials", True, "Draw W/Y/O/R/B/G in the predicted sticker color"),
        ("show-roi", True, "Draw the ROI boundary in full-frame output"),
        ("show-status", False, "Show counts, ROI source, and separate cube bbox score"),
        ("show-fps", True, "Display processing FPS"),
        ("display", True, "Show an OpenCV preview window"),
    ]:
        parser.add_argument(
            "--" + name, action=argparse.BooleanOptionalAction, default=default, help=help_text
        )
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=30)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 processes the whole stream")
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--save",
        "--save-video",
        dest="save",
        type=Path,
        help="Output image or video (.mp4 or .avi)",
    )
    parser.add_argument(
        "--save-json", type=Path, help="Save per-frame JSONL scores and full-frame points"
    )
    return parser


def validate_args(parser, args):
    for name in ("pose_conf", "bbox_conf", "keypoint_conf", "track_conf", "iou"):
        if not math.isfinite(getattr(args, name)) or not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    for name in ("imgsz", "bbox_imgsz", "view_size", "max_det"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "roi_padding",
        "roi_tracking_gain",
        "roi_tracking_padding",
        "camera_fps",
        "bbox_every",
        "start_frame",
        "max_frames",
        "camera_width",
        "camera_height",
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")


def load_model(path):
    from ultralytics import YOLO

    model = YOLO(str(resolve_checkpoint(path)))
    if model.task != "pose" or tuple(model.model.model[-1].kpt_shape) != (4, 3):
        raise ValueError("Expected an 18-class, four-keypoint sticker pose checkpoint")
    if set(model.names.values()) != set(CLASS_NAMES):
        raise ValueError(f"Unexpected checkpoint classes: {model.names}")
    return model


def run(args, model, frame_renderer=render_frame):
    """Shared full-frame image/video/camera flow for independent pose heads."""
    if not args.bbox_model.is_file():
        raise ValueError(f"Cube bbox model not found: {args.bbox_model}")
    detector = YOLOCubeDetector(str(args.bbox_model), args.bbox_conf, args.bbox_imgsz, args.device)
    source = parse_source(args.source)
    tracker = RoiTracker()
    print(f"Pose model: {resolve_checkpoint(args.model)} | imgsz={args.imgsz}")
    print("obj=pose object/class score; p=keypoint presence; cube bbox score is separate")
    json_file = None
    capture = None
    writer = None
    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        json_file = args.save_json.open("w", encoding="utf-8")
    try:
        image_source = isinstance(source, Path) and source.suffix.lower() in IMAGE_EXTENSIONS
        if image_source:
            frame = cv2.imread(str(source))
            if frame is None:
                raise ValueError(f"Could not read image: {source}")
            output, report = frame_renderer(frame, detector, model, tracker, args)
            if args.save:
                args.save.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(args.save), output):
                    raise RuntimeError(f"Could not save image: {args.save}")
            if json_file:
                json_file.write(json.dumps(report) + "\n")
            print(
                f"Detected {len(report['detections'])} objects | ROI from {report['roi_source']}"
            )
            if args.display:
                cv2.imshow(WINDOW, output)
                cv2.waitKey(0)
        else:
            capture = cv2.VideoCapture(source if isinstance(source, int) else str(source))
            if not capture.isOpened():
                raise ValueError(f"Could not open source: {source}")
            if isinstance(source, int):
                for prop, value in [
                    (cv2.CAP_PROP_FRAME_WIDTH, args.camera_width),
                    (cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height),
                    (cv2.CAP_PROP_FPS, args.camera_fps),
                ]:
                    if value:
                        capture.set(prop, value)
            elif args.start_frame:
                capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
            fps = capture.get(cv2.CAP_PROP_FPS)
            fps = fps if math.isfinite(fps) and fps > 0 else 30
            frames = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                output, report = frame_renderer(
                    frame, detector, model, tracker, args, args.start_frame + frames
                )
                if args.save:
                    if args.once and args.save.suffix.lower() in IMAGE_EXTENSIONS:
                        args.save.parent.mkdir(parents=True, exist_ok=True)
                        if not cv2.imwrite(str(args.save), output):
                            raise RuntimeError(f"Could not save image: {args.save}")
                    else:
                        if writer is None:
                            args.save.parent.mkdir(parents=True, exist_ok=True)
                            codec = "MJPG" if args.save.suffix.lower() == ".avi" else "mp4v"
                            writer = cv2.VideoWriter(
                                str(args.save),
                                cv2.VideoWriter_fourcc(*codec),
                                fps,
                                (output.shape[1], output.shape[0]),
                            )
                            if not writer.isOpened():
                                raise RuntimeError(f"Could not open output video: {args.save}")
                        writer.write(output)
                if json_file:
                    json_file.write(json.dumps(report) + "\n")
                frames += 1
                if args.display:
                    cv2.imshow(WINDOW, output)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
                if args.once or (args.max_frames and frames >= args.max_frames):
                    break
            if not frames:
                raise RuntimeError(
                    f"No decodable frames at --start-frame {args.start_frame}: {source}"
                )
            print(f"Processed {frames} frames")
        if args.save:
            print(f"Saved: {args.save}")
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        if json_file:
            json_file.close()
        if args.display:
            cv2.destroyAllWindows()


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    run(args, load_model(args.model))


if __name__ == "__main__":
    main()
