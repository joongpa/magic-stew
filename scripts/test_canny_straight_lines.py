"""Visual test for Canny-backed straight, continuous line evidence.

This script is intentionally separate from the recognition pipeline. It opens an
image, video, or webcam source, runs Canny, asks probabilistic Hough for straight
line candidates, then keeps only candidates with enough edge pixels along the
segment and no large unsupported gap.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


WINDOW_NAME = "canny straight-line evidence"
TRACKBAR_WINDOW = "line filter controls"


@dataclass(frozen=True)
class LineFilterSettings:
    blur_kernel: int
    canny_low: int
    canny_high: int
    hough_threshold: int
    min_line_length: int
    max_line_gap: int
    min_edge_support: float
    max_missing_gap: int
    sample_step: float


@dataclass(frozen=True)
class ScoredLine:
    endpoints: tuple[int, int, int, int]
    support: float
    longest_gap: int
    accepted: bool


def parse_source(source: str) -> int | str:
    source = source.strip().strip("\"'").strip()
    return int(source) if source.isdigit() else source


def odd_kernel(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def clamp_percent(value: int) -> float:
    return max(0.0, min(100.0, float(value))) / 100.0


def create_trackbars(defaults: LineFilterSettings) -> None:
    cv2.namedWindow(TRACKBAR_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TRACKBAR_WINDOW, 420, 360)
    cv2.createTrackbar("blur", TRACKBAR_WINDOW, defaults.blur_kernel, 31, lambda _value: None)
    cv2.createTrackbar("canny low", TRACKBAR_WINDOW, defaults.canny_low, 255, lambda _value: None)
    cv2.createTrackbar("canny high", TRACKBAR_WINDOW, defaults.canny_high, 255, lambda _value: None)
    cv2.createTrackbar(
        "hough votes",
        TRACKBAR_WINDOW,
        defaults.hough_threshold,
        200,
        lambda _value: None,
    )
    cv2.createTrackbar(
        "min length",
        TRACKBAR_WINDOW,
        defaults.min_line_length,
        400,
        lambda _value: None,
    )
    cv2.createTrackbar(
        "hough gap",
        TRACKBAR_WINDOW,
        defaults.max_line_gap,
        120,
        lambda _value: None,
    )
    cv2.createTrackbar(
        "support %",
        TRACKBAR_WINDOW,
        int(round(defaults.min_edge_support * 100)),
        100,
        lambda _value: None,
    )
    cv2.createTrackbar(
        "max gap px",
        TRACKBAR_WINDOW,
        defaults.max_missing_gap,
        120,
        lambda _value: None,
    )


def read_trackbar_settings(defaults: LineFilterSettings) -> LineFilterSettings:
    canny_low = cv2.getTrackbarPos("canny low", TRACKBAR_WINDOW)
    canny_high = cv2.getTrackbarPos("canny high", TRACKBAR_WINDOW)
    if canny_low > canny_high:
        canny_low, canny_high = canny_high, canny_low

    return LineFilterSettings(
        blur_kernel=odd_kernel(cv2.getTrackbarPos("blur", TRACKBAR_WINDOW)),
        canny_low=canny_low,
        canny_high=max(canny_high, canny_low + 1),
        hough_threshold=max(1, cv2.getTrackbarPos("hough votes", TRACKBAR_WINDOW)),
        min_line_length=max(1, cv2.getTrackbarPos("min length", TRACKBAR_WINDOW)),
        max_line_gap=max(0, cv2.getTrackbarPos("hough gap", TRACKBAR_WINDOW)),
        min_edge_support=clamp_percent(cv2.getTrackbarPos("support %", TRACKBAR_WINDOW)),
        max_missing_gap=max(0, cv2.getTrackbarPos("max gap px", TRACKBAR_WINDOW)),
        sample_step=defaults.sample_step,
    )


def sample_line_pixels(
    edge_image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    sample_step: float,
) -> np.ndarray:
    length = float(np.hypot(x2 - x1, y2 - y1))
    sample_count = max(2, int(round(length / max(0.25, sample_step))) + 1)
    xs = np.rint(np.linspace(x1, x2, sample_count)).astype(np.int32)
    ys = np.rint(np.linspace(y1, y2, sample_count)).astype(np.int32)
    xs = np.clip(xs, 0, edge_image.shape[1] - 1)
    ys = np.clip(ys, 0, edge_image.shape[0] - 1)
    return edge_image[ys, xs] > 0


def longest_false_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in values:
        if value:
            current = 0
            continue
        current += 1
        longest = max(longest, current)
    return longest


def score_lines(
    edge_image: np.ndarray,
    lines: np.ndarray | None,
    settings: LineFilterSettings,
) -> list[ScoredLine]:
    if lines is None:
        return []

    scored_lines: list[ScoredLine] = []
    max_gap_samples = int(round(settings.max_missing_gap / max(0.25, settings.sample_step)))
    for raw_line in lines.reshape(-1, 4):
        x1, y1, x2, y2 = (int(value) for value in raw_line)
        edge_samples = sample_line_pixels(edge_image, x1, y1, x2, y2, settings.sample_step)
        support = float(np.mean(edge_samples)) if edge_samples.size else 0.0
        longest_gap = longest_false_run(edge_samples)
        accepted = support >= settings.min_edge_support and longest_gap <= max_gap_samples
        scored_lines.append(
            ScoredLine(
                endpoints=(x1, y1, x2, y2),
                support=support,
                longest_gap=int(round(longest_gap * settings.sample_step)),
                accepted=accepted,
            )
        )
    return scored_lines


def build_edges(frame: np.ndarray, settings: LineFilterSettings) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if settings.blur_kernel > 1:
        gray = cv2.GaussianBlur(gray, (settings.blur_kernel, settings.blur_kernel), 0)
    return cv2.Canny(gray, settings.canny_low, settings.canny_high)


def detect_scored_lines(frame: np.ndarray, settings: LineFilterSettings) -> tuple[np.ndarray, list[ScoredLine]]:
    edges = build_edges(frame, settings)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=settings.hough_threshold,
        minLineLength=settings.min_line_length,
        maxLineGap=settings.max_line_gap,
    )
    return edges, score_lines(edges, lines, settings)


def draw_overlay(frame: np.ndarray, edges: np.ndarray, lines: list[ScoredLine], show_rejected: bool) -> np.ndarray:
    edge_overlay = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edge_overlay[:, :, 0] = np.maximum(edge_overlay[:, :, 0], edges)
    combined = cv2.addWeighted(frame, 0.72, edge_overlay, 0.35, 0)

    accepted_count = 0
    for line in sorted(lines, key=lambda item: item.support):
        x1, y1, x2, y2 = line.endpoints
        if not line.accepted and not show_rejected:
            continue
        color = (40, 220, 60) if line.accepted else (70, 70, 220)
        thickness = 2 if line.accepted else 1
        cv2.line(combined, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        if line.accepted:
            accepted_count += 1
            cv2.circle(combined, (x1, y1), 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(combined, (x2, y2), 3, (255, 255, 255), -1, cv2.LINE_AA)

    label = f"accepted {accepted_count}/{len(lines)}  q/esc quit  space pause  s save"
    cv2.rectangle(combined, (8, 8), (8 + 520, 42), (0, 0, 0), -1)
    cv2.putText(
        combined,
        label,
        (18, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return combined


def resize_for_display(frame: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0 or frame.shape[1] <= max_width:
        return frame
    scale = max_width / frame.shape[1]
    height = int(round(frame.shape[0] * scale))
    return cv2.resize(frame, (max_width, height), interpolation=cv2.INTER_AREA)


def save_debug_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write debug image to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Canny edges filtered to straight-ish, continuous-ish line segments."
    )
    parser.add_argument(
        "source",
        nargs="?",
        default="0",
        help="Camera index, image path, or video path. Defaults to webcam 0.",
    )
    parser.add_argument("--blur", type=int, default=5, help="Gaussian blur kernel size.")
    parser.add_argument("--canny-low", type=int, default=60, help="Lower Canny threshold.")
    parser.add_argument("--canny-high", type=int, default=160, help="Upper Canny threshold.")
    parser.add_argument("--hough-threshold", type=int, default=35, help="Minimum Hough votes.")
    parser.add_argument("--min-line-length", type=int, default=35, help="Minimum line length in pixels.")
    parser.add_argument("--max-line-gap", type=int, default=8, help="Maximum Hough gap in pixels.")
    parser.add_argument(
        "--min-edge-support",
        type=float,
        default=0.62,
        help="Minimum fraction of sampled segment pixels that must be Canny edge pixels.",
    )
    parser.add_argument(
        "--max-missing-gap",
        type=int,
        default=12,
        help="Maximum continuous missing edge run along an accepted segment, in pixels.",
    )
    parser.add_argument("--sample-step", type=float, default=1.0, help="Line sampling step in pixels.")
    parser.add_argument("--max-display-width", type=int, default=1280, help="Scale display down to this width.")
    parser.add_argument(
        "--show-rejected",
        action="store_true",
        help="Also draw candidate Hough lines rejected by support/gap filtering.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("runs/canny_straight_lines_debug.png"),
        help="Path used when pressing s.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    defaults = LineFilterSettings(
        blur_kernel=odd_kernel(args.blur),
        canny_low=max(0, args.canny_low),
        canny_high=max(1, args.canny_high),
        hough_threshold=max(1, args.hough_threshold),
        min_line_length=max(1, args.min_line_length),
        max_line_gap=max(0, args.max_line_gap),
        min_edge_support=max(0.0, min(1.0, args.min_edge_support)),
        max_missing_gap=max(0, args.max_missing_gap),
        sample_step=max(0.25, args.sample_step),
    )

    source = parse_source(args.source)
    still_image = None
    capture = None
    if isinstance(source, str) and Path(source).is_file():
        still_image = cv2.imread(source)
        if still_image is None:
            raise RuntimeError(f"Could not read image source: {source}")
    else:
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    create_trackbars(defaults)

    paused = False
    last_frame = still_image
    last_overlay = None

    while True:
        if still_image is not None:
            frame = still_image.copy()
        elif paused and last_frame is not None:
            frame = last_frame.copy()
        else:
            ok, frame = capture.read()
            if not ok:
                break
            last_frame = frame.copy()

        settings = read_trackbar_settings(defaults)
        edges, lines = detect_scored_lines(frame, settings)
        overlay = draw_overlay(frame, edges, lines, args.show_rejected)
        last_overlay = overlay
        display = resize_for_display(overlay, args.max_display_width)
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(0 if still_image is not None or paused else 1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
        if key == ord("s") and last_overlay is not None:
            save_debug_image(args.save_path, last_overlay)
            print(f"Saved {args.save_path}")

        if still_image is not None and key not in (255,):
            continue

    if capture is not None:
        capture.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
