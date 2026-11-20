"""Prepare the MCP cube wireframes for the official F-Clip trainer.

The generated layout follows F-Clip's processed dataset convention::

    output/
      train/
        sample_000000.png
        sample_000000_label.npz
        sample_000000_line.npz
      valid/
        ...

Positive samples use the normalized endpoint pairs in
``labels/wireframe``. Negative ROI images are emitted with empty line and
heatmap targets so F-Clip learns to suppress lines on background-only crops.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
# F-Clip stores canonical line coordinates at 128x128. The trainer rescales
# those coordinates when model.resolution is lower than 128. The corrected
# cube experiment uses resolution=64, which means 256x256 network inputs.
FCLIP_IMAGE_SIZE = 256
FCLIP_HEATMAP_SIZE = 128


@dataclass(frozen=True)
class Sample:
    image_path: Path
    wireframe_label_path: Path | None

    @property
    def negative(self) -> bool:
        return self.wireframe_label_path is None


def parse_wireframe_label(path: Path) -> np.ndarray:
    lines: list[list[float]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        parts = raw_line.split()
        if len(parts) != 5:
            raise ValueError(
                f"{path}:{line_number} must contain class_id and two normalized endpoints"
            )
        try:
            class_id = int(parts[0])
            coordinates = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number} contains non-numeric values") from exc
        if class_id != 0:
            raise ValueError(f"{path}:{line_number} has unsupported line class {class_id}")
        if not all(0.0 <= value <= 1.0 for value in coordinates):
            raise ValueError(f"{path}:{line_number} contains coordinates outside [0, 1]")
        lines.append(coordinates)
    return np.asarray(lines, dtype=np.float32).reshape(-1, 2, 2)


def collect_samples(source_dir: Path, negative_dir: Path | None) -> list[Sample]:
    images_dir = source_dir / "images"
    labels_dir = source_dir / "labels" / "wireframe"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing MCP images directory: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing wireframe labels directory: {labels_dir}")

    image_paths = sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    label_by_stem = {path.stem: path for path in labels_dir.glob("*.txt")}
    samples = [
        Sample(image_path, label_by_stem[image_path.stem])
        for image_path in image_paths
        if image_path.stem in label_by_stem
    ]
    if not samples:
        raise RuntimeError(f"No image/wireframe-label pairs found under {source_dir}")

    if negative_dir is not None:
        if not negative_dir.is_dir():
            raise FileNotFoundError(f"Missing negative-image directory: {negative_dir}")
        samples.extend(
            Sample(path, None)
            for path in sorted(negative_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return samples


def _fclip_targets(lines_xy: np.ndarray, width: int, height: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Create F-Clip's 128x128 line targets from pixel-space x/y endpoints."""
    fy = FCLIP_HEATMAP_SIZE / height
    fx = FCLIP_HEATMAP_SIZE / width
    lines_yx = lines_xy.copy()
    lines_yx[:, :, 0] *= fy
    lines_yx[:, :, 1] *= fx
    lines_yx = lines_yx[:, :, ::-1]

    lcmap = np.zeros((FCLIP_HEATMAP_SIZE, FCLIP_HEATMAP_SIZE), dtype=np.float32)
    lcoff = np.zeros((2, FCLIP_HEATMAP_SIZE, FCLIP_HEATMAP_SIZE), dtype=np.float32)
    lleng = np.zeros((FCLIP_HEATMAP_SIZE, FCLIP_HEATMAP_SIZE), dtype=np.float32)
    angle = np.zeros((FCLIP_HEATMAP_SIZE, FCLIP_HEATMAP_SIZE), dtype=np.float32)

    for v0, v1 in lines_yx:
        v = (v0 + v1) / 2.0
        vint = tuple(map(int, v))
        if not (0 <= vint[0] < FCLIP_HEATMAP_SIZE and 0 <= vint[1] < FCLIP_HEATMAP_SIZE):
            continue
        lcmap[vint] = 1.0
        lcoff[:, vint[0], vint[1]] = v - vint - 0.5
        lleng[vint] = np.sqrt(np.sum((v0 - v1) ** 2)) / 2.0

        vv = v0 if v0[0] <= v[0] else v1
        direction = vv - v
        direction_norm = np.sqrt(np.sum(direction ** 2))
        if direction_norm > 1e-4:
            angle[vint] = np.sum(direction * np.array([0.0, 1.0])) / direction_norm

    targets = {
        "lcmap": lcmap,
        "lcoff": lcoff,
        "lleng": lleng,
        "angle": angle,
    }
    lpos = np.concatenate(
        [lines_yx, np.zeros((len(lines_yx), 2, 1), dtype=np.float32)],
        axis=2,
    )
    return targets, lpos


def _write_sample(sample: Sample, output_dir: Path, output_stem: str) -> dict[str, object]:
    image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {sample.image_path}")
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError(f"Image has invalid dimensions: {sample.image_path}")

    resized = cv2.resize(image, (FCLIP_IMAGE_SIZE, FCLIP_IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    image_path = output_dir / f"{output_stem}.png"
    if not cv2.imwrite(str(image_path), resized):
        raise OSError(f"Could not write prepared image: {image_path}")

    if sample.negative:
        lines_xy = np.empty((0, 2, 2), dtype=np.float32)
    else:
        assert sample.wireframe_label_path is not None
        normalized_lines = parse_wireframe_label(sample.wireframe_label_path)
        lines_xy = normalized_lines * np.array([width, height], dtype=np.float32)

    targets, lpos = _fclip_targets(lines_xy, width, height)
    line_path = output_dir / f"{output_stem}_line.npz"
    label_path = output_dir / f"{output_stem}_label.npz"
    np.savez_compressed(line_path, **targets)
    np.savez_compressed(
        label_path,
        aspect_ratio=1.0,
        lpos=lpos,
    )
    return {
        "source_image": str(sample.image_path),
        "source_wireframe_label": str(sample.wireframe_label_path)
        if sample.wireframe_label_path is not None
        else None,
        "image": str(image_path),
        "label": str(label_path),
        "line": str(line_path),
        "negative": sample.negative,
        "line_count": int(len(lines_xy)),
    }


def prepare_dataset(
    source_dir: Path,
    negative_dir: Path | None,
    output_dir: Path,
    val_fraction: float,
    seed: int,
) -> Path:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in the range [0, 1)")

    samples = collect_samples(source_dir, negative_dir)
    random.Random(seed).shuffle(samples)
    val_count = max(1, round(len(samples) * val_fraction)) if len(samples) > 1 and val_fraction else 0
    splits = {"valid": samples[:val_count], "train": samples[val_count:]}

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    manifest: list[dict[str, object]] = []
    summary: dict[str, dict[str, int]] = {}
    for split, split_samples in splits.items():
        split_dir = output_dir / split
        split_dir.mkdir()
        negative_count = 0
        line_count = 0
        for index, sample in enumerate(split_samples):
            record = _write_sample(sample, split_dir, f"cube_{index:06d}")
            record["split"] = split
            manifest.append(record)
            negative_count += int(sample.negative)
            line_count += int(record["line_count"])
        summary[split] = {
            "images": len(split_samples),
            "negative_images": negative_count,
            "positive_images": len(split_samples) - negative_count,
            "wireframe_lines": line_count,
        }

    (output_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (output_dir / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/mcp_dataset_v4"))
    parser.add_argument(
        "--negative-dir",
        type=Path,
        default=Path("data/negative_examples/negative_ROI_backgrounds"),
        help="Background-only ROI images; emitted with empty line targets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("F-Clip/data/cube_wireframe_mcp_v4_256"),
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--no-negative-examples", action="store_true")
    args = parser.parse_args()
    prepare_dataset(
        source_dir=args.source_dir,
        negative_dir=None if args.no_negative_examples else args.negative_dir,
        output_dir=args.output_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
