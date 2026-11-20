"""Report YOLO segmentation labels that look like rectangle fallback polygons."""

from __future__ import annotations

import argparse
from pathlib import Path


def points_from_line(line: str) -> list[tuple[float, float]]:
    parts = line.split()
    values = [float(value) for value in parts[1:]]
    return list(zip(values[0::2], values[1::2]))


def is_axis_aligned_rectangle(points: list[tuple[float, float]], tolerance: float = 1e-6) -> bool:
    if len(points) != 4:
        return False
    xs = {round(x / tolerance) for x, _y in points}
    ys = {round(y / tolerance) for _x, y in points}
    return len(xs) == 2 and len(ys) == 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", type=Path, default=Path("synthetic_cube"))
    parser.add_argument("--show", type=int, default=50)
    args = parser.parse_args()

    labels_dir = args.dataset / "labels"
    if not labels_dir.exists():
        raise SystemExit(f"Missing labels directory: {labels_dir}")

    total_segments = 0
    rectangle_segments = 0
    files_with_rectangles = []

    for label_path in sorted(labels_dir.glob("*.txt")):
        file_rectangles = 0
        lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for line in lines:
            total_segments += 1
            if is_axis_aligned_rectangle(points_from_line(line)):
                rectangle_segments += 1
                file_rectangles += 1
        if file_rectangles:
            files_with_rectangles.append((label_path, file_rectangles, len(lines)))

    print(f"Total segments: {total_segments}")
    print(f"Rectangle-like segments: {rectangle_segments}")
    print(f"Files with rectangle-like segments: {len(files_with_rectangles)}")
    for label_path, rectangles, total in files_with_rectangles[: args.show]:
        print(f"  {label_path}: {rectangles}/{total}")
    if len(files_with_rectangles) > args.show:
        print(f"  ... {len(files_with_rectangles) - args.show} more")

    if rectangle_segments:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
