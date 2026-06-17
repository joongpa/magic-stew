"""Display YOLO cube-pose keypoints on a centered 256x256 viewport."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CUBE_POSE_MODEL = ROOT / "runs/pose/runs/cube_pose/runs/cube_yolo11n_pose/weights/best.pt"
DEFAULT_VIEW_SIZE = 256
CORNER_KEYPOINT_INDEXES = set(range(7))


def parse_source(source: str) -> int | str:
    source = source.strip().strip("\"'").strip()
    return int(source) if source.isdigit() else source


def centered_square_viewport(width: int, height: int) -> tuple[int, int, int, int]:
    viewport_size = min(width, height)
    center_x = width // 2
    center_y = height // 2
    x1 = max(0, min(width - viewport_size, center_x - viewport_size // 2))
    y1 = max(0, min(height - viewport_size, center_y - viewport_size // 2))
    return x1, y1, x1 + viewport_size, y1 + viewport_size


def centered_view(frame: np.ndarray, size: int) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = centered_square_viewport(width, height)
    roi = frame[y1:y2, x1:x2]
    if roi.shape[0] == size and roi.shape[1] == size:
        return roi.copy()
    return cv2.resize(roi, (size, size), interpolation=cv2.INTER_AREA)


def tensor_to_numpy(value) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    return array if array.size else None


def result_keypoints(result, size: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None:
        return None, None

    xy = tensor_to_numpy(getattr(keypoints, "xy", None))
    conf = tensor_to_numpy(getattr(keypoints, "conf", None))
    if xy is not None:
        return np.asarray(xy, dtype=np.float32), None if conf is None else np.asarray(conf, dtype=np.float32)

    xyn = tensor_to_numpy(getattr(keypoints, "xyn", None))
    if xyn is not None:
        xy = np.asarray(xyn, dtype=np.float32).copy()
        xy[..., 0] *= size
        xy[..., 1] *= size
        return xy, None if conf is None else np.asarray(conf, dtype=np.float32)

    data = tensor_to_numpy(getattr(keypoints, "data", None))
    if data is None:
        return None, None
    data = np.asarray(data, dtype=np.float32)
    if data.shape[-1] >= 2:
        xy = data[..., :2]
        conf = data[..., 2] if data.shape[-1] >= 3 else None
        return xy, conf
    return None, None


def draw_keypoints(
    frame: np.ndarray,
    xy: np.ndarray | None,
    conf: np.ndarray | None,
    *,
    corners_only: bool,
) -> np.ndarray:
    output = frame.copy()
    if xy is None or len(xy) == 0:
        cv2.putText(output, "no cube pose", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        return output

    points = np.asarray(xy[0], dtype=np.float32)
    scores = None if conf is None or len(conf) == 0 else np.asarray(conf[0], dtype=np.float32)
    for index, point in enumerate(points):
        if corners_only and index not in CORNER_KEYPOINT_INDEXES:
            continue

        x, y = (int(round(value)) for value in point)
        if not (0 <= x < output.shape[1] and 0 <= y < output.shape[0]):
            continue

        score = None if scores is None or index >= len(scores) else float(scores[index])
        color = (255, 0, 0) if index == 0 else (255, 255, 255)
        if score is not None and score < 0.35:
            color = (120, 120, 120)

        cv2.circle(output, (x, y), 4, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(output, (x, y), 5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        cv2.putText(output, str(index), (x + 5, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)
    return output


def process_frame(frame: np.ndarray, model, args: argparse.Namespace) -> np.ndarray:
    view = centered_view(frame, args.view_size)
    results = model.predict(source=view, conf=args.conf, imgsz=args.imgsz, verbose=False)
    result = results[0] if results else None
    xy, conf = result_keypoints(result, args.view_size) if result is not None else (None, None)
    return draw_keypoints(view, xy, conf, corners_only=args.corners_only)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Camera index, video path, or image path")
    parser.add_argument("--model", type=Path, default=DEFAULT_CUBE_POSE_MODEL)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_VIEW_SIZE)
    parser.add_argument("--view-size", type=int, default=DEFAULT_VIEW_SIZE)
    parser.add_argument("--save", type=Path, help="Save one annotated frame/image")
    parser.add_argument("--corners-only", action="store_true", help="Draw only cube corner keypoints")
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.set_defaults(display=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    from ultralytics import YOLO

    model = YOLO(str(args.model))
    source = parse_source(args.source)

    if isinstance(source, str) and Path(source).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
        frame = cv2.imread(source)
        if frame is None:
            raise SystemExit(f"Could not read image: {source}")
        output = process_frame(frame, model, args)
        if args.save:
            cv2.imwrite(str(args.save), output)
        if args.display:
            cv2.imshow("cube pose points", output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return 0

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    previous = perf_counter()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        output = process_frame(frame, model, args)
        now = perf_counter()
        fps = 1.0 / max(now - previous, 1e-6)
        previous = now
        cv2.putText(output, f"FPS {fps:.1f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 220, 40), 1, cv2.LINE_AA)
        if args.save:
            cv2.imwrite(str(args.save), output)
            break
        if args.display:
            cv2.imshow("cube pose points", output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    capture.release()
    if args.display:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
