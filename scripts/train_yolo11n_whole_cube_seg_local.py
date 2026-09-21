"""Train YOLO11 nano on whole-cube polygon segmentation labels locally.

Expected input layout:

    synthetic_cube_dataset/
      images/*.png
      labels/
        whole_cube/
          classes.json
          *.txt
        stickers/              # ignored by this script
      masks/*.png               # optional visualization masks, not used

The script shuffles image/label pairs with a fixed seed, consolidates any
disconnected source contours into one whole-cube instance per image, writes a
local Ultralytics segmentation dataset layout, and trains yolo11n-seg.pt.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MIN_POLYGON_FIELDS = 1 + 3 * 2


def label_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_class_names(labels_dir: Path) -> list[str]:
    classes_path = labels_dir / "classes.json"
    if not classes_path.exists():
        raise FileNotFoundError(f"Missing whole-cube class mapping: {classes_path}")

    try:
        raw_classes = json.loads(classes_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read class mapping: {classes_path}") from exc

    if isinstance(raw_classes, list):
        class_names = [str(name) for name in raw_classes]
    elif isinstance(raw_classes, dict):
        try:
            class_names = [str(raw_classes[str(index)]) for index in range(len(raw_classes))]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"Class mapping must have contiguous string keys starting at 0: {classes_path}"
            ) from exc
    else:
        raise ValueError(f"Class mapping must be a JSON list or object: {classes_path}")

    if not class_names or any(not name.strip() for name in class_names):
        raise ValueError(f"Class mapping must contain at least one non-empty name: {classes_path}")
    return class_names


def parse_label_file(
    path: Path,
    class_names: list[str],
) -> list[tuple[int, list[tuple[float, float]]]]:
    polygons = []
    for line_number, line in enumerate(label_lines(path), start=1):
        parts = line.split()
        if len(parts) < MIN_POLYGON_FIELDS or (len(parts) - 1) % 2:
            raise ValueError(
                f"{path}:{line_number} is not valid YOLO segmentation format; "
                f"expected a class and at least 3 (x, y) points"
            )

        try:
            class_id = int(parts[0])
            coordinates = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number} contains non-numeric values") from exc

        if class_id < 0 or class_id >= len(class_names):
            raise ValueError(f"{path}:{line_number} has invalid class id {class_id}")
        for value_index, value in enumerate(coordinates, start=1):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{path}:{line_number} polygon value {value_index} is outside [0, 1]: {value}"
                )
        polygons.append((class_id, list(zip(coordinates[::2], coordinates[1::2]))))
    return polygons


def validate_label_file(path: Path, class_names: list[str]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for class_id, _points in parse_label_file(path, class_names):
        counts[class_id] += 1
    return counts


def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return one counter-clockwise polygon around all component points."""
    unique_points = sorted(set(points))
    if len(unique_points) <= 1:
        return unique_points

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (
            (first[0] - origin[0]) * (second[1] - origin[1])
            - (first[1] - origin[1]) * (second[0] - origin[0])
        )

    lower: list[tuple[float, float]] = []
    for point in unique_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def consolidated_label_lines(path: Path, class_names: list[str]) -> list[str]:
    """Make one YOLO instance row from all whole-cube source contours."""
    polygons = parse_label_file(path, class_names)
    if not polygons:
        return []

    class_ids = {class_id for class_id, _points in polygons}
    if len(class_ids) != 1:
        raise ValueError(
            f"{path} contains multiple classes; whole-cube labels must contain one cube class"
        )
    class_id = next(iter(class_ids))
    if len(polygons) == 1:
        points = polygons[0][1]
    else:
        points = convex_hull(
            [point for _polygon_class_id, polygon in polygons for point in polygon]
        )
    if len(points) < 3:
        raise ValueError(f"{path} does not contain enough points for a consolidated polygon")

    return [
        str(class_id)
        + " "
        + " ".join(f"{coordinate:.6f}" for point in points for coordinate in point)
    ]


def collect_pairs(source_dir: Path) -> tuple[list[tuple[Path, Path]], list[str]]:
    images_dir = source_dir / "images"
    labels_dir = source_dir / "labels" / "whole_cube"
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Missing whole-cube labels directory: {labels_dir}")

    image_paths = sorted(
        path for path in images_dir.glob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    label_by_stem = {path.stem: path for path in labels_dir.glob("*.txt")}
    pairs = [
        (image_path, label_by_stem[image_path.stem])
        for image_path in image_paths
        if image_path.stem in label_by_stem
    ]
    if not pairs:
        raise RuntimeError(f"No image/whole-cube-label pairs found under {source_dir}")
    return pairs, load_class_names(labels_dir)


def write_dataset(
    source_dir: Path,
    dataset_dir: Path,
    val_fraction: float,
    seed: int,
) -> Path:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in the range [0, 1)")

    pairs, class_names = collect_pairs(source_dir)
    random.Random(seed).shuffle(pairs)

    val_count = max(1, round(len(pairs) * val_fraction)) if len(pairs) > 1 and val_fraction else 0
    splits = {"val": pairs[:val_count], "train": pairs[val_count:]}

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    for split in ["train", "val"]:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest = []
    summary = {}
    for split, split_pairs in splits.items():
        class_counts: Counter[int] = Counter()
        source_contour_count_distribution: Counter[int] = Counter()
        training_instance_count_distribution: Counter[int] = Counter()
        merged_image_count = 0

        for index, (image_path, label_path) in enumerate(split_pairs):
            source_lines = label_lines(label_path)
            source_contour_count_distribution[len(source_lines)] += 1
            validate_label_file(label_path, class_names)
            lines = consolidated_label_lines(label_path, class_names)
            training_instance_count_distribution[len(lines)] += 1
            if len(source_lines) > 1:
                merged_image_count += 1
            class_counts.update(int(line.split()[0]) for line in lines)

            target_stem = f"{split}_{index:06d}"
            target_image = dataset_dir / "images" / split / f"{target_stem}{image_path.suffix.lower()}"
            target_label = dataset_dir / "labels" / split / f"{target_stem}.txt"
            shutil.copy2(image_path, target_image)
            target_label.write_text(
                "\n".join(lines) + ("\n" if lines else ""),
                encoding="utf-8",
            )
            manifest.append(
                {
                    "split": split,
                    "source_image": str(image_path),
                    "source_label": str(label_path),
                    "image": str(target_image),
                    "label": str(target_label),
                    "source_contours": len(source_lines),
                    "training_instances": len(lines),
                    "consolidated_contours": len(source_lines) > 1,
                }
            )

        summary[split] = {
            "images": len(split_pairs),
            "merged_images": merged_image_count,
            "source_contour_count_distribution": dict(sorted(source_contour_count_distribution.items())),
            "training_instance_count_distribution": dict(sorted(training_instance_count_distribution.items())),
            "training_class_instances": {
                class_names[class_id]: class_counts[class_id] for class_id in range(len(class_names))
            },
        }

    (dataset_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(
        "path: " + str(dataset_dir.resolve()) + "\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        + "".join(f"  {index}: {name}\n" for index, name in enumerate(class_names)),
        encoding="utf-8",
    )

    print("Dataset summary:")
    print(json.dumps(summary, indent=2))
    print(
        "Training labels contain one consolidated cube instance per non-empty image; "
        "source contour counts above are pre-consolidation diagnostics."
    )
    print(data_yaml.read_text(encoding="utf-8"))
    return data_yaml


def train(data_yaml: Path, runs_dir: Path, args: argparse.Namespace) -> Path:
    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(runs_dir),
        name=args.name,
        patience=args.patience,
        cache=args.cache,
        plots=True,
    )

    # Ultralytics increments the run directory when args.name already exists
    # (for example, cube_whole_yolo11n_seg-2). Use the trainer's resolved path
    # instead of reconstructing the unsuffixed path.
    trainer = getattr(model, "trainer", None)
    best_model_path = getattr(trainer, "best", None)
    if best_model_path is None:
        raise RuntimeError("Training completed without exposing a best-model path.")
    best_model_path = Path(best_model_path).resolve()
    if not best_model_path.is_file():
        raise FileNotFoundError(f"Training completed but best model was not found: {best_model_path}")
    print(f"Best model: {best_model_path}")

    best_model = YOLO(str(best_model_path))
    metrics = best_model.val(data=str(data_yaml), imgsz=args.imgsz, device=args.device)
    print(metrics)
    return best_model_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("data/synthetic_cube_dataset"))
    parser.add_argument("--workdir", type=Path, default=Path("runs/cube_whole_seg_local"))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--model", default="yolo11n-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--name", default="cube_whole_yolo11n_seg")
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    dataset_dir = args.workdir / "dataset"
    runs_dir = args.workdir / "runs"
    data_yaml = write_dataset(
        source_dir=args.source_dir,
        dataset_dir=dataset_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    if args.prepare_only:
        print(f"Prepared dataset: {dataset_dir}")
        return

    train(data_yaml, runs_dir, args)


if __name__ == "__main__":
    main()
