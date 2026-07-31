"""Visualize patch-level scalar blurriness over a whole image.

This script deliberately avoids cube detectors, bounding boxes, and pose models.
It splits the full image into fixed-size patches, computes local sharpness, and
draws a scalar blurriness map where warmer colors mean blurrier patches.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


DEFAULT_OUTPUT_PATH = Path("runs/patch_blur_debug.png")
DEFAULT_CSV_PATH = Path("runs/patch_blur_debug.csv")


@dataclass(frozen=True)
class BlurPatch:
    row: int
    col: int
    x: int
    y: int
    width: int
    height: int
    laplacian_var: float
    tenengrad: float
    blur_magnitude: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize whole-image patch-level blur estimates."
    )
    parser.add_argument("image", type=Path, help="Input image path.")
    parser.add_argument(
        "--patch-size",
        type=int,
        default=16,
        help="Patch size in pixels. Defaults to 16.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output overlay path. Defaults to {DEFAULT_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help=f"Optional patch metrics CSV path. Example: {DEFAULT_CSV_PATH}.",
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Do not draw patch grid lines.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the overlay in an OpenCV window after saving.",
    )
    return parser.parse_args()


def robust_blur_magnitudes(sharpness_values: np.ndarray) -> np.ndarray:
    """Convert sharpness to relative blur, where 1.0 is blurriest."""
    if sharpness_values.size == 0:
        return sharpness_values

    low = float(np.percentile(sharpness_values, 10))
    high = float(np.percentile(sharpness_values, 90))
    if high <= low + 1e-9:
        return np.zeros_like(sharpness_values, dtype=np.float64)

    sharpness = np.clip((sharpness_values - low) / (high - low), 0.0, 1.0)
    return 1.0 - sharpness


def estimate_blur_patches(gray: np.ndarray, patch_size: int) -> list[BlurPatch]:
    height, width = gray.shape[:2]
    raw_patches: list[dict[str, float | int | None]] = []
    tenengrad_values: list[float] = []

    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch = gray[y : min(y + patch_size, height), x : min(x + patch_size, width)]
            laplacian_var = float(cv2.Laplacian(patch, cv2.CV_64F).var())
            gx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
            tenengrad = float(np.mean(gx * gx + gy * gy))

            raw_patches.append(
                {
                    "row": y // patch_size,
                    "col": x // patch_size,
                    "x": x,
                    "y": y,
                    "width": int(patch.shape[1]),
                    "height": int(patch.shape[0]),
                    "laplacian_var": laplacian_var,
                    "tenengrad": tenengrad,
                }
            )
            tenengrad_values.append(tenengrad)

    blur_magnitudes = robust_blur_magnitudes(np.asarray(tenengrad_values, dtype=np.float64))
    patches: list[BlurPatch] = []
    for raw_patch, blur_magnitude in zip(raw_patches, blur_magnitudes, strict=True):
        patches.append(
            BlurPatch(
                row=int(raw_patch["row"]),
                col=int(raw_patch["col"]),
                x=int(raw_patch["x"]),
                y=int(raw_patch["y"]),
                width=int(raw_patch["width"]),
                height=int(raw_patch["height"]),
                laplacian_var=float(raw_patch["laplacian_var"]),
                tenengrad=float(raw_patch["tenengrad"]),
                blur_magnitude=float(blur_magnitude),
            )
        )
    return patches


def build_blur_map(shape: tuple[int, int], patches: list[BlurPatch]) -> np.ndarray:
    blur_map = np.zeros(shape, dtype=np.float32)
    for patch in patches:
        blur_map[patch.y : patch.y + patch.height, patch.x : patch.x + patch.width] = patch.blur_magnitude
    return blur_map


def draw_overlay(
    image: np.ndarray,
    patches: list[BlurPatch],
    patch_size: int,
    draw_grid: bool,
) -> np.ndarray:
    blur_map = build_blur_map(image.shape[:2], patches)
    overlay = cv2.applyColorMap(np.uint8(np.round(blur_map * 255.0)), cv2.COLORMAP_TURBO)

    if draw_grid:
        for x in range(0, image.shape[1], patch_size):
            cv2.line(overlay, (x, 0), (x, image.shape[0] - 1), (30, 30, 30), 1)
        for y in range(0, image.shape[0], patch_size):
            cv2.line(overlay, (0, y), (image.shape[1] - 1, y), (30, 30, 30), 1)

    label = "scalar patch blurriness: blue/green sharp, yellow/red blurry"
    cv2.rectangle(overlay, (8, 8), (min(image.shape[1] - 1, 760), 42), (0, 0, 0), -1)
    cv2.putText(
        overlay,
        label,
        (18, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return overlay


def write_csv(path: Path, patches: list[BlurPatch]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "row",
                "col",
                "x",
                "y",
                "width",
                "height",
                "laplacian_var",
                "tenengrad",
                "blur_magnitude",
            ],
        )
        writer.writeheader()
        for patch in patches:
            writer.writerow(
                {
                    "row": patch.row,
                    "col": patch.col,
                    "x": patch.x,
                    "y": patch.y,
                    "width": patch.width,
                    "height": patch.height,
                    "laplacian_var": patch.laplacian_var,
                    "tenengrad": patch.tenengrad,
                    "blur_magnitude": patch.blur_magnitude,
                }
            )


def main() -> int:
    args = parse_args()
    if args.patch_size < 2:
        raise ValueError("--patch-size must be at least 2")

    image = cv2.imread(str(args.image))
    if image is None:
        raise RuntimeError(f"Could not read image: {args.image}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    patches = estimate_blur_patches(gray, args.patch_size)
    overlay = draw_overlay(
        image=image,
        patches=patches,
        patch_size=args.patch_size,
        draw_grid=not args.no_grid,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), overlay):
        raise RuntimeError(f"Could not write output image: {args.output}")
    print(f"Saved overlay to {args.output}")

    if args.csv is not None:
        write_csv(args.csv, patches)
        print(f"Saved patch metrics to {args.csv}")

    if args.show:
        cv2.imshow("patch blur", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
