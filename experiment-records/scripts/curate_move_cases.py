from __future__ import annotations

import argparse
import json
import queue
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


WINDOW_NAME = "Cube move case curator"
STATUS_WINDOW_NAME = "Curation controls and status"
CAMERA_WINDOW_NAME = "Camera position relative to solver"
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}


@dataclass(frozen=True)
class VideoSource:
    value: str
    path: Path
    url: str | None


class CameraPositionPicker:
    """Mouse-driven top and side views for manually annotating camera position."""

    WIDTH = 520
    HEIGHT = 280
    TOP_CENTER = (130, 155)
    SIDE_CENTER = (390, 155)
    RADIUS = 82

    def __init__(self) -> None:
        self.horizontal_degrees = 0.0
        self.vertical_degrees = 0.0
        self.horizontal_selected = False
        self.vertical_selected = False
        self.drag_panel: str | None = None

    @property
    def ready(self) -> bool:
        return self.horizontal_selected and self.vertical_selected

    def as_dict(self) -> dict[str, object]:
        return {
            "horizontal_angle_degrees": round(self.horizontal_degrees, 1),
            "vertical_angle_degrees": round(self.vertical_degrees, 1),
            "tilt_angle_degrees": 0.0,
            "reference": (
                "Relative to solver facing the cube: horizontal 0 is straight ahead, "
                "positive is solver-right; vertical positive is above level."
            ),
            "manually_set": self.ready,
        }

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_panel = "horizontal" if x < self.WIDTH // 2 else "vertical"
            self._set_from_point(self.drag_panel, x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_panel is not None:
            self._set_from_point(self.drag_panel, x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drag_panel is not None:
                self._set_from_point(self.drag_panel, x, y)
            self.drag_panel = None

    def _set_from_point(self, panel: str, x: int, y: int) -> None:
        if panel == "horizontal":
            cx, cy = self.TOP_CENTER
            self.horizontal_degrees = float(np.degrees(np.arctan2(x - cx, cy - y)))
            self.horizontal_selected = True
        else:
            cx, cy = self.SIDE_CENTER
            self.vertical_degrees = float(
                np.clip(np.degrees(np.arctan2(cy - y, x - cx)), -90.0, 90.0)
            )
            self.vertical_selected = True

    def render(self) -> np.ndarray:
        image = np.full((self.HEIGHT, self.WIDTH, 3), (36, 38, 42), dtype=np.uint8)
        cv2.line(image, (260, 10), (260, self.HEIGHT - 10), (90, 90, 90), 1)
        cv2.putText(
            image, "TOP VIEW", (78, 24), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (235, 235, 235), 1, cv2.LINE_AA
        )
        cv2.putText(
            image, "SIDE VIEW", (333, 24), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (235, 235, 235), 1, cv2.LINE_AA
        )

        top = self.TOP_CENTER
        cv2.circle(image, top, self.RADIUS, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.arrowedLine(
            image,
            top,
            (top[0], top[1] - self.RADIUS - 17),
            (180, 220, 180),
            2,
            cv2.LINE_AA,
            tipLength=0.18,
        )
        cv2.putText(
            image, "solver forward", (53, 48), cv2.FONT_HERSHEY_SIMPLEX,
            0.38, (180, 220, 180), 1, cv2.LINE_AA
        )
        cv2.rectangle(image, (top[0] - 15, top[1] - 12), (top[0] + 15, top[1] + 12), (190, 190, 190), 2)
        top_radians = np.radians(self.horizontal_degrees)
        top_camera = (
            int(round(top[0] + self.RADIUS * np.sin(top_radians))),
            int(round(top[1] - self.RADIUS * np.cos(top_radians))),
        )
        cv2.line(image, top, top_camera, (80, 180, 255), 2, cv2.LINE_AA)
        cv2.circle(image, top_camera, 9, (80, 180, 255), -1, cv2.LINE_AA)
        cv2.putText(
            image, "camera", (top_camera[0] - 24, top_camera[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA
        )
        h_status = "SET" if self.horizontal_selected else "CLICK TO SET"
        cv2.putText(
            image,
            f"Horizontal: {self.horizontal_degrees:+.1f} deg  [{h_status}]",
            (18, 265),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

        side = self.SIDE_CENTER
        cv2.ellipse(image, side, (self.RADIUS, self.RADIUS), 0, -90, 90, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.line(image, (side[0] - 20, side[1]), (side[0] + self.RADIUS + 18, side[1]), (110, 110, 110), 1, cv2.LINE_AA)
        cv2.rectangle(image, (side[0] - 14, side[1] - 12), (side[0] + 14, side[1] + 12), (190, 190, 190), 2)
        vertical_radians = np.radians(self.vertical_degrees)
        side_camera = (
            int(round(side[0] + self.RADIUS * np.cos(vertical_radians))),
            int(round(side[1] - self.RADIUS * np.sin(vertical_radians))),
        )
        cv2.line(image, side, side_camera, (80, 180, 255), 2, cv2.LINE_AA)
        cv2.circle(image, side_camera, 9, (80, 180, 255), -1, cv2.LINE_AA)
        cv2.putText(
            image, "camera", (side_camera[0] - 24, side_camera[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA
        )
        v_status = "SET" if self.vertical_selected else "CLICK TO SET"
        cv2.putText(
            image,
            f"Vertical: {self.vertical_degrees:+.1f} deg  [{v_status}]",
            (277, 265),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        return image


class VideoRoiPicker:
    """Play a video while choosing a fixed crop rectangle in source coordinates."""

    MAX_PREVIEW_WIDTH = 1280
    MAX_PREVIEW_HEIGHT = 800

    def __init__(
        self,
        capture: cv2.VideoCapture,
        first_frame: np.ndarray,
        fps: float,
        frame_count: int,
    ) -> None:
        self.capture = capture
        self.frame = first_frame
        self.frame_index = 0
        self.fps = fps
        self.frame_count = frame_count
        self.playing = False
        self.drag_start: tuple[int, int] | None = None
        self.roi_xyxy: tuple[int, int, int, int] | None = None
        self.preview_width = first_frame.shape[1]
        self.preview_height = first_frame.shape[0]

    def _source_point(self, x: int, y: int) -> tuple[int, int]:
        source_height, source_width = self.frame.shape[:2]
        source_x = round(x * source_width / max(1, self.preview_width))
        source_y = round(y * source_height / max(1, self.preview_height))
        return (
            int(np.clip(source_x, 0, source_width - 1)),
            int(np.clip(source_y, 0, source_height - 1)),
        )

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        point = self._source_point(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = point
            self.roi_xyxy = (point[0], point[1], point[0], point[1])
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.roi_xyxy = (
                self.drag_start[0],
                self.drag_start[1],
                point[0],
                point[1],
            )
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drag_start is not None:
                self.roi_xyxy = (
                    self.drag_start[0],
                    self.drag_start[1],
                    point[0],
                    point[1],
                )
            self.drag_start = None

    def render_video(self) -> np.ndarray:
        source_height, source_width = self.frame.shape[:2]
        scale = min(
            1.0,
            self.MAX_PREVIEW_WIDTH / source_width,
            self.MAX_PREVIEW_HEIGHT / source_height,
        )
        self.preview_width = max(1, int(round(source_width * scale)))
        self.preview_height = max(1, int(round(source_height * scale)))
        preview = cv2.resize(self.frame, (self.preview_width, self.preview_height))
        if self.roi_xyxy is not None:
            x1, y1, x2, y2 = self.roi_xyxy
            left, right = sorted((x1, x2))
            top, bottom = sorted((y1, y2))
            start = (
                round(left * self.preview_width / source_width),
                round(top * self.preview_height / source_height),
            )
            end = (
                round(right * self.preview_width / source_width),
                round(bottom * self.preview_height / source_height),
            )
            cv2.rectangle(preview, start, end, (40, 230, 90), 2, cv2.LINE_AA)
        return preview

    def render_status(self) -> np.ndarray:
        roi_text = "not selected"
        if self.roi_xyxy is not None:
            x1, y1, x2, y2 = self.roi_xyxy
            roi_text = f"x={min(x1, x2)}, y={min(y1, y2)}, w={abs(x2 - x1)}, h={abs(y2 - y1)}"
        return render_status_panel(
            [
                "ROI selection",
                f"{'PLAY' if self.playing else 'PAUSE'} | frame {self.frame_index} | "
                f"{self.frame_index / self.fps:.3f}s",
                f"Crop: {roi_text}",
                "Drag a rectangle over the cube; press Enter to confirm.",
                "Space play/pause | ,/. step frame | j/l jump 1s | c clear ROI | q quit",
            ]
        )

    def select(self) -> tuple[tuple[int, int, int, int], int, np.ndarray] | None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(STATUS_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)
        try:
            while True:
                cv2.imshow(WINDOW_NAME, self.render_video())
                cv2.imshow(STATUS_WINDOW_NAME, self.render_status())
                raw_key = cv2.waitKey(max(1, int(1000 / self.fps)) if self.playing else 30)
                key = raw_key & 0xFF if raw_key >= 0 else -1

                if key in (10, 13):
                    if self.roi_xyxy is not None:
                        x1, y1, x2, y2 = self.roi_xyxy
                        left, right = sorted((x1, x2))
                        top, bottom = sorted((y1, y2))
                        if right > left and bottom > top:
                            return (
                                (left, top, right - left, bottom - top),
                                self.frame_index,
                                self.frame,
                            )
                elif key == ord("q"):
                    return None
                elif key == ord("c"):
                    self.roi_xyxy = None
                    self.drag_start = None
                elif key == ord(" "):
                    self.playing = not self.playing
                elif key in (ord(","), ord(".")):
                    self.playing = False
                    next_index = self.frame_index + (-1 if key == ord(",") else 1)
                    next_index = max(0, min(next_index, self.frame_count - 1))
                    next_frame = frame_at(self.capture, next_index)
                    if next_frame is not None:
                        self.frame_index, self.frame = next_index, next_frame
                elif key in (ord("j"), ord("l")):
                    self.playing = False
                    offset = int(round(self.fps)) * (-1 if key == ord("j") else 1)
                    next_index = max(0, self.frame_index + offset)
                    if self.frame_count:
                        next_index = min(next_index, self.frame_count - 1)
                    next_frame = frame_at(self.capture, next_index)
                    if next_frame is not None:
                        self.frame_index, self.frame = next_index, next_frame

                if self.playing and key == -1:
                    success, next_frame = self.capture.read()
                    if not success:
                        self.playing = False
                    else:
                        self.frame_index += 1
                        self.frame = next_frame
        finally:
            cv2.destroyWindow(WINDOW_NAME)
            cv2.destroyWindow(STATUS_WINDOW_NAME)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Browse cubing videos, mark two frames around a move interval, and save "
            "labeled image-pair cases."
        )
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--source",
        nargs="+",
        help="One or more local video paths or YouTube URLs.",
    )
    source_group.add_argument(
        "--source-list",
        type=Path,
        help="Text file with one local video path or YouTube URL per line (# comments allowed).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "move_cases",
        help="Directory for case folders (default: data/move_cases).",
    )
    parser.add_argument(
        "--video-cache",
        type=Path,
        default=Path("data") / "move_case_video_cache",
        help="Where YouTube source videos are cached (default: data/move_case_video_cache).",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=720,
        help="Maximum YouTube download height; use 0 for best available (default: 720).",
    )
    return parser


def read_sources(args: argparse.Namespace) -> list[str]:
    if args.source is not None:
        return [str(value) for value in args.source]
    if not args.source_list.is_file():
        raise SystemExit(f"Source list does not exist: {args.source_list}")
    sources = []
    for raw_line in args.source_list.read_text(encoding="utf-8").splitlines():
        value = raw_line.strip()
        if value and not value.startswith("#"):
            if not is_url(value):
                path = Path(value).expanduser()
                if not path.is_absolute():
                    value = str((args.source_list.parent / path).resolve())
            sources.append(value)
    if not sources:
        raise SystemExit(f"No sources found in: {args.source_list}")
    return sources


def is_url(value: str) -> bool:
    return value.startswith("https://") or value.startswith("http://")


def resolve_source(value: str, cache_dir: Path, max_height: int) -> VideoSource:
    if not is_url(value):
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Video does not exist: {path}")
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video extension: {path.suffix}")
        return VideoSource(value=value, path=path, url=None)

    downloader = shutil.which("yt-dlp")
    if downloader is None:
        raise RuntimeError("YouTube URLs require yt-dlp. Install it or use a local video file.")
    cache_dir.mkdir(parents=True, exist_ok=True)
    format_selector = f"best[height<={max_height}]/best" if max_height > 0 else "best"
    output_template = str(cache_dir / "%(id)s.%(ext)s")
    command = [
        downloader,
        "--no-playlist",
        "--no-overwrites",
        "--format",
        format_selector,
        "--output",
        output_template,
        "--print",
        "after_move:filepath",
        value,
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"yt-dlp could not download {value}: {detail}")
    paths = [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    if not paths or not paths[-1].is_file():
        raise RuntimeError(f"yt-dlp finished but did not report a video file for {value}")
    return VideoSource(value=value, path=paths[-1].resolve(), url=value)


def next_case_id(output_dir: Path) -> int:
    highest = 0
    for path in output_dir.glob("case_*"):
        match = re.fullmatch(r"case_(\d+)", path.name)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def frame_at(capture: cv2.VideoCapture, index: int) -> np.ndarray | None:
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    success, frame = capture.read()
    return frame if success else None


def render_status_panel(lines: Sequence[str], width: int = 1000) -> np.ndarray:
    line_height = 28
    height = max(150, 28 + len(lines) * line_height)
    panel = np.full((height, width, 3), (32, 34, 38), dtype=np.uint8)
    for index, line in enumerate(lines):
        cv2.putText(
            panel,
            line,
            (14, 28 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    return panel


def parse_move_label(raw: str) -> list[str]:
    return [token for token in raw.strip().split() if token]


def save_case(
    output_dir: Path,
    case_id: int,
    source: VideoSource,
    first_frame: np.ndarray,
    first_index: int,
    second_frame: np.ndarray,
    second_index: int,
    fps: float,
    moves: Sequence[str],
    camera_position: dict[str, object],
    roi_xywh: tuple[int, int, int, int],
    original_size_wh: tuple[int, int],
) -> Path:
    case_dir = output_dir / f"case_{case_id:06d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    first_path = case_dir / "frame_a.png"
    second_path = case_dir / "frame_b.png"
    if not cv2.imwrite(str(first_path), first_frame):
        raise OSError(f"Could not write {first_path}")
    if not cv2.imwrite(str(second_path), second_frame):
        raise OSError(f"Could not write {second_path}")
    metadata = {
        "case_id": case_dir.name,
        "frame_a": first_path.name,
        "frame_b": second_path.name,
        "source_input": source.value,
        "source_video": source.path.name,
        "source_url": source.url,
        "source_frame_a": first_index,
        "source_frame_b": second_index,
        "fps": fps,
        "timestamp_a_seconds": first_index / fps,
        "timestamp_b_seconds": second_index / fps,
        "delta_seconds": (second_index - first_index) / fps,
        "moves": list(moves),
        "camera_position_relative_to_solver": camera_position,
        "roi_xywh": list(roi_xywh),
        "original_frame_size_wh": list(original_size_wh),
        "start_state": None,
        "end_state": None,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (case_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return case_dir


def curate_video(source: VideoSource, output_dir: Path, first_case_id: int) -> tuple[int, bool]:
    capture = cv2.VideoCapture(str(source.path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source.path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    first_frame = frame_at(capture, 0)
    if first_frame is None:
        capture.release()
        raise RuntimeError(f"Video contains no readable frames: {source.path}")

    roi_selection = VideoRoiPicker(capture, first_frame, fps, frame_count).select()
    if roi_selection is None:
        capture.release()
        return 0, True
    roi_xywh, frame_index, original_frame = roi_selection
    roi_x, roi_y, roi_width, roi_height = roi_xywh
    original_height, original_width = first_frame.shape[:2]

    def crop(frame: np.ndarray) -> np.ndarray:
        return frame[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]

    frame = crop(original_frame)

    playing = False
    saved_count = 0
    leave_queue = False
    case_ids_by_frame: dict[int, int] = {}
    labels_by_frame: dict[int, list[str]] = {}
    camera_position = CameraPositionPicker()
    label_results: queue.Queue[tuple[int, str]] = queue.Queue()
    pending_frame_index: int | None = None
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(STATUS_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(CAMERA_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(CAMERA_WINDOW_NAME, camera_position.on_mouse)

    def record_transition(transition_index: int, moves: list[str]) -> None:
        nonlocal saved_count
        previous_original = frame_at(capture, transition_index - 1)
        current_original = frame_at(capture, transition_index)
        if previous_original is None or current_original is None:
            print("Could not read both frames for this transition.")
            return

        case_id = case_ids_by_frame.get(transition_index, first_case_id + saved_count)
        case_dir = save_case(
            output_dir,
            case_id,
            source,
            crop(previous_original),
            transition_index - 1,
            crop(current_original),
            transition_index,
            fps,
            moves,
            camera_position.as_dict(),
            roi_xywh,
            (original_width, original_height),
        )
        if transition_index not in case_ids_by_frame:
            case_ids_by_frame[transition_index] = case_id
            saved_count += 1
        labels_by_frame[transition_index] = moves
        print(
            f"Saved {case_dir}: transition {transition_index - 1}->{transition_index}, "
            f"moves={moves or 'none'}"
        )

    try:
        while True:
            try:
                transition_index, raw_moves = label_results.get_nowait()
            except queue.Empty:
                pass
            else:
                pending_frame_index = None
                record_transition(transition_index, parse_move_label(raw_moves))

            current_moves = labels_by_frame.get(frame_index)
            label_text = (
                "unlabeled (no move)"
                if current_moves is None
                else "no move (recorded)"
                if not current_moves
                else " ".join(current_moves)
            )
            status_lines = [
                f"Video: {source.path.name}",
                f"{'PLAY' if playing else 'PAUSE'} | frame {frame_index} | {frame_index / fps:.3f}s",
                (
                    f"Transition {frame_index - 1}->{frame_index}: {label_text}"
                    if frame_index > 0
                    else "Frame 0 has no preceding transition"
                ),
                "Space play/pause | ,/. step one frame | j/l jump one second",
                "m opens the terminal move label for the current n-1 -> n transition",
                f"Fixed crop ROI: x={roi_x}, y={roi_y}, width={roi_width}, height={roi_height}",
                (
                    f"Terminal input pending for transition "
                    f"{pending_frame_index - 1}->{pending_frame_index}"
                    if pending_frame_index is not None
                    else ""
                ),
                "n next video | q quit",
            ]
            cv2.imshow(WINDOW_NAME, frame)
            cv2.imshow(STATUS_WINDOW_NAME, render_status_panel(status_lines))
            cv2.imshow(CAMERA_WINDOW_NAME, camera_position.render())
            raw_key = cv2.waitKey(max(1, int(1000 / fps)) if playing else 30)
            key = raw_key & 0xFF if raw_key >= 0 else -1

            if key == ord("q"):
                if pending_frame_index is None:
                    leave_queue = True
                    break
                print("Finish the pending terminal label before quitting.")
            if key == ord("n"):
                if pending_frame_index is None:
                    break
                print("Finish the pending terminal label before moving to the next video.")
            if key == ord(" "):
                playing = not playing
            elif key == ord("m"):
                playing = False
                if frame_index <= 0:
                    print("Frame 0 has no preceding frame to label.")
                elif pending_frame_index is not None:
                    print(
                        f"A terminal label is already pending for "
                        f"{pending_frame_index - 1}->{pending_frame_index}."
                    )
                elif not camera_position.ready:
                    print(
                        "Set the camera position in both top and side views before saving a case."
                    )
                else:
                    pending_frame_index = frame_index

                    def read_move_label(transition_index: int = frame_index) -> None:
                        try:
                            raw_moves = input(
                                f"Move applied from frame {transition_index - 1} to "
                                f"{transition_index} (space-separated notation; blank = no move): "
                            )
                        except EOFError:
                            raw_moves = ""
                        label_results.put((transition_index, raw_moves))

                    threading.Thread(target=read_move_label, daemon=True).start()
            elif key in (ord(","), ord(".")):
                playing = False
                next_index = frame_index + (-1 if key == ord(",") else 1)
                next_index = (
                    max(0, min(next_index, frame_count - 1))
                    if frame_count
                    else max(0, next_index)
                )
                next_original = frame_at(capture, next_index)
                if next_original is not None:
                    frame_index, frame = next_index, crop(next_original)
            elif key in (ord("j"), ord("l")):
                playing = False
                offset = int(round(fps)) * (-1 if key == ord("j") else 1)
                next_index = max(0, frame_index + offset)
                if frame_count:
                    next_index = min(next_index, frame_count - 1)
                next_original = frame_at(capture, next_index)
                if next_original is not None:
                    frame_index, frame = next_index, crop(next_original)

            if playing and key == -1:
                success, next_original = capture.read()
                if not success:
                    playing = False
                else:
                    frame_index += 1
                    frame = crop(next_original)
    finally:
        capture.release()
        cv2.destroyWindow(WINDOW_NAME)
        cv2.destroyWindow(STATUS_WINDOW_NAME)
        cv2.destroyWindow(CAMERA_WINDOW_NAME)
    return saved_count, leave_queue


def main() -> int:
    args = build_parser().parse_args()
    if args.max_height < 0:
        raise SystemExit("--max-height must be 0 or greater")
    sources = read_sources(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    next_id = next_case_id(args.output_dir)
    total_saved = 0

    for value in sources:
        try:
            source = resolve_source(value, args.video_cache, args.max_height)
            print(f"Opening {source.path} (case IDs start at {next_id:06d})")
            saved, quit_requested = curate_video(source, args.output_dir, next_id)
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            print(f"Skipping {value}: {error}")
            continue
        total_saved += saved
        next_id += saved
        if quit_requested:
            break

    print(f"Done. Saved {total_saved} case(s) under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
