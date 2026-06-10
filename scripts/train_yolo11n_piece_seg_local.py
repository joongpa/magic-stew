"""Train YOLO11 nano segmentation locally on generated cube sticker labels.

Expected input layout:

    synthetic_cube/
      images/*.png
      labels/*.txt
      masks/*.png          # optional, not used for training

The script shuffles image/label pairs with a fixed seed, writes a local
Ultralytics dataset layout, and trains yolo11n-seg.pt.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path


CLASS_NAMES = ["center_sticker", "edge_sticker", "corner_sticker"]


def label_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_label_file(path: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    for line_number, line in enumerate(label_lines(path), start=1):
        parts = line.split()
        if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
            raise ValueError(f"{path}:{line_number} is not valid YOLO segmentation format")
        class_id = int(float(parts[0]))
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            raise ValueError(f"{path}:{line_number} has invalid class id {class_id}")
        counts[class_id] += 1
    return counts


def collect_pairs(source_dir: Path) -> list[tuple[Path, Path]]:
    images_dir = source_dir / "images"
    labels_dir = source_dir / "labels"
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Missing labels directory: {labels_dir}")

    image_paths = sorted(
        path for path in images_dir.glob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    label_by_stem = {path.stem: path for path in labels_dir.glob("*.txt")}
    pairs = [(image_path, label_by_stem[image_path.stem]) for image_path in image_paths if image_path.stem in label_by_stem]
    if not pairs:
        raise RuntimeError(f"No image/label pairs found under {source_dir}")
    return pairs


def write_dataset(
    source_dir: Path,
    dataset_dir: Path,
    val_fraction: float,
    seed: int,
) -> Path:
    pairs = collect_pairs(source_dir)
    random.Random(seed).shuffle(pairs)

    val_count = max(1, round(len(pairs) * val_fraction)) if len(pairs) > 1 else 0
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
        label_count_distribution: Counter[int] = Counter()

        for index, (image_path, label_path) in enumerate(split_pairs):
            lines = label_lines(label_path)
            label_count_distribution[len(lines)] += 1
            class_counts.update(validate_label_file(label_path))

            target_stem = f"{split}_{index:06d}"
            target_image = dataset_dir / "images" / split / f"{target_stem}{image_path.suffix.lower()}"
            target_label = dataset_dir / "labels" / split / f"{target_stem}.txt"
            shutil.copy2(image_path, target_image)
            shutil.copy2(label_path, target_label)
            manifest.append({
                "split": split,
                "source_image": str(image_path),
                "source_label": str(label_path),
                "image": str(target_image),
                "label": str(target_label),
                "segments": len(lines),
            })

        summary[split] = {
            "images": len(split_pairs),
            "label_count_distribution": dict(sorted(label_count_distribution.items())),
            "class_segments": {
                CLASS_NAMES[class_id]: class_counts[class_id] for class_id in range(len(CLASS_NAMES))
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
        + "".join(f"  {index}: {name}\n" for index, name in enumerate(CLASS_NAMES)),
        encoding="utf-8",
    )

    print("Dataset summary:")
    print(json.dumps(summary, indent=2))
    print(data_yaml.read_text())
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

    best_model_path = runs_dir / args.name / "weights" / "best.pt"
    print(f"Best model: {best_model_path}")

    best_model = YOLO(str(best_model_path))
    metrics = best_model.val(data=str(data_yaml), imgsz=args.imgsz, device=args.device)
    print(metrics)
    return best_model_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("synthetic_cube"))
    parser.add_argument("--workdir", type=Path, default=Path("runs/cube_piece_seg_local"))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--model", default="yolo11n-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--device", default=0)
    parser.add_argument("--name", default="cube_piece_yolo11n_seg")
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
