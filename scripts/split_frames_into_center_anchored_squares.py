#!/usr/bin/env python3
"""Split frames into four square crops, retaining each crop's center-facing edge."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


def image_size(path: Path) -> tuple[int, int]:
    result = subprocess.run(
        ["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    width = re.search(r"pixelWidth:\s*(\d+)", result.stdout)
    height = re.search(r"pixelHeight:\s*(\d+)", result.stdout)
    if not width or not height:
        raise RuntimeError(f"Could not read dimensions for {path}")
    return int(width.group(1)), int(height.group(1))


def crop_box(width: int, height: int, quadrant: str) -> tuple[int, int, int]:
    """Return (side, left, top), preserving the edge nearest the source center."""
    mid_x = width // 2
    mid_y = height // 2
    boxes = {
        "tl": (0, 0, mid_x, mid_y),
        "tr": (mid_x, 0, width, mid_y),
        "bl": (0, mid_y, mid_x, height),
        "br": (mid_x, mid_y, width, height),
    }
    x1, y1, x2, y2 = boxes[quadrant]
    quadrant_width = x2 - x1
    quadrant_height = y2 - y1
    side = min(quadrant_width, quadrant_height)

    if quadrant_width > quadrant_height:
        # Remove excess width only from the outer edge.
        left = x2 - side if quadrant in {"tl", "bl"} else x1
        top = y1
    else:
        # Remove excess height only from the outer edge.
        left = x1
        top = y2 - side if quadrant in {"tl", "tr"} else y1
    return side, left, top


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    sources = sorted(args.input_dir.glob("*.png"))
    if not sources:
        raise SystemExit(f"No PNG frames found in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    quadrants = ("tl", "tr", "bl", "br")
    created = 0
    for source in sources:
        width, height = image_size(source)
        for quadrant in quadrants:
            side, left, top = crop_box(width, height, quadrant)
            destination = args.output_dir / f"{source.stem}_{quadrant}.png"
            subprocess.run(
                [
                    "sips",
                    "-c",
                    str(side),
                    str(side),
                    "--cropOffset",
                    str(top),
                    str(left),
                    str(source),
                    "--out",
                    str(destination),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            created += 1

    print(f"Created {created} square crops from {len(sources)} frames in {args.output_dir}")


if __name__ == "__main__":
    main()
