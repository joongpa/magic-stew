"""Train YOLO11 nano segmentation on cube and sticker polygons.

Expected input layout:

    mcp_dataset_v5.3/
      images/*.png
      labels/*.txt

    extra-negative-dir/    # optional cube-free images; written with empty labels

Each label file uses standard YOLO segmentation rows:

    <class_id> <x1> <y1> <x2> <y2> ...

Class 0 is ``cube`` and class 1 is ``sticker`` by default. The script makes a
deterministic train/validation split, copies the images and labels into a
local Ultralytics layout, and trains ``yolo11n-seg.pt``.
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
DEFAULT_EXTRA_NEGATIVE_DIR = Path("data/youtube_solve_frames_hd/cropped-negatives")
Sample = tuple[Path, Path | None]


def label_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_class_names(raw: str) -> list[str]:
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if names != ["cube", "sticker"]:
        raise ValueError("This trainer requires --class-names cube,sticker")
    return names


def validate_label_file(path: Path, class_names: list[str]) -> Counter[int]:
    counts: Counter[int] = Counter()
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
        counts[class_id] += 1
    return counts


def collect_pairs(
    source_dir: Path,
    class_names: list[str],
) -> tuple[list[Sample], Counter[int]]:
    images_dir = source_dir / "images"
    labels_dir = source_dir / "labels"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing labels directory: {labels_dir}")

    image_paths = sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    label_by_stem = {path.stem: path for path in labels_dir.glob("*.txt")}
    missing_labels = [path for path in image_paths if path.stem not in label_by_stem]
    if missing_labels:
        raise RuntimeError(
            f"{len(missing_labels)} images have no matching label file; first missing label is "
            f"{missing_labels[0].stem}.txt"
        )

    pairs: list[Sample] = [(image_path, label_by_stem[image_path.stem]) for image_path in image_paths]
    if not pairs:
        raise RuntimeError(f"No image/label pairs found under {source_dir}")

    class_counts: Counter[int] = Counter()
    for _image_path, label_path in pairs:
        class_counts.update(validate_label_file(label_path, class_names))

    return pairs, class_counts


def write_dataset(
    source_dirs: list[Path],
    dataset_dir: Path,
    val_fraction: float,
    seed: int,
    class_names: list[str],
    extra_negative_dir: Path | None = None,
) -> Path:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in the range [0, 1)")

    pairs: list[Sample] = []
    source_class_counts: Counter[int] = Counter()
    for source_dir in source_dirs:
        source_pairs, class_counts = collect_pairs(source_dir, class_names)
        pairs.extend(source_pairs)
        source_class_counts.update(class_counts)
    if extra_negative_dir is not None:
        if not extra_negative_dir.is_dir():
            raise FileNotFoundError(f"Missing extra-negative image directory: {extra_negative_dir}")
        negative_paths = sorted(
            path
            for path in extra_negative_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        pairs.extend((image_path, None) for image_path in negative_paths)
    if not pairs:
        raise RuntimeError("No image/label pairs found in the supplied dataset directories")
    missing_classes = [
        class_names[class_id]
        for class_id in range(len(class_names))
        if not source_class_counts[class_id]
    ]
    if missing_classes:
        raise RuntimeError(f"Combined datasets contain no instances of: {', '.join(missing_classes)}")
    random.Random(seed).shuffle(pairs)

    val_count = max(1, round(len(pairs) * val_fraction)) if len(pairs) > 1 and val_fraction else 0
    splits = {"val": pairs[:val_count], "train": pairs[val_count:]}

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    for split in splits:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest = []
    summary = {}
    for split, split_pairs in splits.items():
        class_counts: Counter[int] = Counter()
        object_count_distribution: Counter[int] = Counter()

        negative_image_count = 0
        for index, (image_path, label_path) in enumerate(split_pairs):
            is_negative = label_path is None
            lines = label_lines(label_path) if label_path is not None else []
            if label_path is not None:
                counts = validate_label_file(label_path, class_names)
                class_counts.update(counts)
            else:
                negative_image_count += 1
            object_count_distribution[len(lines)] += 1

            target_stem = f"{split}_{index:06d}"
            target_image = dataset_dir / "images" / split / f"{target_stem}{image_path.suffix.lower()}"
            target_label = dataset_dir / "labels" / split / f"{target_stem}.txt"
            shutil.copy2(image_path, target_image)
            if label_path is None:
                target_label.write_text("", encoding="utf-8")
            else:
                shutil.copy2(label_path, target_label)
            manifest.append(
                {
                    "split": split,
                    "source_image": str(image_path),
                    "source_label": str(label_path) if label_path is not None else None,
                    "image": str(target_image),
                    "label": str(target_label),
                    "objects": len(lines),
                    "negative": is_negative,
                }
            )

        summary[split] = {
            "images": len(split_pairs),
            "negative_images": negative_image_count,
            "object_count_distribution": dict(sorted(object_count_distribution.items())),
            "class_objects": {
                class_names[class_id]: class_counts[class_id]
                for class_id in range(len(class_names))
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

    print("Source class totals:")
    print(json.dumps({class_names[index]: source_class_counts[index] for index in range(len(class_names))}, indent=2))
    print("Prepared dataset summary:")
    print(json.dumps(summary, indent=2))
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
        mask_ratio=args.mask_ratio,
        overlap_mask=args.overlap_mask,
        mosaic=args.mosaic,
        plots=True,
    )

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
    parser.add_argument(
        "--source-dir",
        "--source-dirs",
        dest="source_dir_groups",
        type=Path,
        nargs="+",
        action="append",
        default=None,
        help=(
            "Dataset root(s) containing images and YOLO segmentation labels; provide multiple "
            "paths after this option or repeat it to combine datasets."
        ),
    )
    parser.add_argument(
        "--extra-negative-dir",
        type=Path,
        default=DEFAULT_EXTRA_NEGATIVE_DIR,
        help=(
            "Directory of cube-free images to include with empty labels "
            f"(default: {DEFAULT_EXTRA_NEGATIVE_DIR})"
        ),
    )
    parser.add_argument(
        "--no-extra-negatives",
        action="store_true",
        help="Train from positive cube/sticker samples only",
    )
    parser.add_argument("--workdir", type=Path, default=Path("runs/cube_sticker_seg_local"))
    parser.add_argument("--class-names", default="cube,sticker")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--model", default="yolo11n-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument(
        "--device",
        default=None,
        help="Training device override, e.g. cpu, mps, cuda, 0, or 0,1; default lets Ultralytics choose.",
    )
    parser.add_argument("--name", default="cube_sticker_yolo11n_seg")
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument(
        "--mosaic",
        type=float,
        default=0.0,
        help="Mosaic augmentation probability; 0.0 disables mosaic augmentation.",
    )
    parser.add_argument(
        "--mask-ratio",
        type=int,
        default=2,
        help="Segmentation-mask downsample ratio; lower values preserve more mask detail",
    )
    parser.add_argument(
        "--overlap-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow Ultralytics to collapse overlapping instance masks (usually off for cube+sticker)",
    )
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    if args.mask_ratio < 1:
        raise SystemExit("--mask-ratio must be at least 1")
    class_names = parse_class_names(args.class_names)
    source_dirs = [
        source_dir
        for source_dir_group in (args.source_dir_groups or [[Path("data/mcp_dataset_v5.3")]])
        for source_dir in source_dir_group
    ]
    dataset_dir = args.workdir / "dataset"
    runs_dir = args.workdir / "runs"
    data_yaml = write_dataset(
        source_dirs=source_dirs,
        dataset_dir=dataset_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
        class_names=class_names,
        extra_negative_dir=None if args.no_extra_negatives else args.extra_negative_dir,
    )

    if args.prepare_only:
        print(f"Prepared dataset: {dataset_dir}")
        return

    train(data_yaml, runs_dir, args)


if __name__ == "__main__":
    main()
