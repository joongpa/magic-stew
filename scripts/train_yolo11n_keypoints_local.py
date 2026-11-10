#!/usr/bin/env python3
"""Train a YOLO pose model on cube landmarks, stickers, and cube cores.

Expected source layout:

    dataset/
      images/*.png
      pose/*.txt
      sticker_keypoint_manifest.json

Landmark and sticker rows have one keypoint:

    class cx cy width height keypoint_x keypoint_y visibility

New Blender exports declare a keypoint shape of [6, 3] and include the
`cube-core` class alongside the three landmark and six sticker classes.
Each core row has six (x, y, visibility) triples in white, yellow, orange,
red, blue, green order. Use visibility 2 for visible centers, 1 for hidden
centers inside the crop, and 0 0 0 for centers outside it. Core targets do
not use the sticker and landmark face-angle cutoff.

The preparer pads one-keypoint rows to six slots for the shared YOLO head
when the dataset includes cube cores. It also accepts already padded rows
and legacy one-keypoint datasets.

The training comparison intentionally disables mosaic, hue, and saturation
augmentation while retaining value/brightness and ordinary geometric
augmentation.

Example:

    .venv/bin/python scripts/train_yolo11n_keypoints_local.py \
        --model-family yolo26 \
        --model-size small \
        --source-dir data/keypoints_only_dataset \
        --device 0 \
        --batch 12 \
        --epochs 100
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
CUBE_CORE_CLASS = "cube-core"
CUBE_CORE_COLORS = ("white", "yellow", "orange", "red", "blue", "green")
VISIBILITY_VALUES = {0, 1, 2}
DEFAULT_CLASS_NAMES = ["corner", "corner-edge", "corner-edge-center"]
YOLO_POSE_MODEL_FAMILIES = {
    "yolo11": {
        "nano": "yolo11n-pose.pt",
        "small": "yolo11s-pose.pt",
        "medium": "yolo11m-pose.pt",
        "large": "yolo11l-pose.pt",
        "xlarge": "yolo11x-pose.pt",
    },
    "yolo26": {
        "nano": "yolo26n-pose.pt",
        "small": "yolo26s-pose.pt",
        "medium": "yolo26m-pose.pt",
        "large": "yolo26l-pose.pt",
        "xlarge": "yolo26x-pose.pt",
    },
}


def parse_class_names(value: str) -> list[str]:
    names = [name.strip() for name in value.split(",")]
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("Class names must be nonempty and unique.")
    return names


def keypoint_shape(class_names: list[str]) -> list[int]:
    return [len(CUBE_CORE_COLORS) if CUBE_CORE_CLASS in class_names else 1, 3]


def source_class_names(source_dir: Path) -> list[str]:
    """Read the generator manifest before falling back to data.yaml."""

    for filename in ("sticker_keypoint_manifest.json", "independent_keypoint_manifest.json"):
        path = source_dir / filename
        if path.is_file():
            metadata = json.loads(path.read_text(encoding="utf-8"))
            names = parse_class_names(",".join(metadata["class_names"]))
            shape = keypoint_shape(names)
            if metadata.get("keypoint_shape") != shape:
                raise ValueError(f"{path}: expected keypoint_shape {shape}")
            return names

    data_yaml = source_dir / "data.yaml"
    if data_yaml.is_file():
        import yaml

        metadata = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        names = metadata.get("names")
        if isinstance(names, dict):
            indexed = {int(index): str(name) for index, name in names.items()}
            names = [indexed[index] for index in range(len(indexed))]
        names = parse_class_names(",".join(names)) if isinstance(names, list) else DEFAULT_CLASS_NAMES.copy()
        shape = keypoint_shape(names)
        if metadata.get("kpt_shape") != shape:
            raise ValueError(f"{data_yaml}: expected kpt_shape {shape}")
        return names
    return DEFAULT_CLASS_NAMES.copy()


def resolve_class_names(source_dirs: list[Path], override: list[str] | None) -> list[str]:
    if override is not None:
        return override
    names = source_class_names(source_dirs[0])
    for source_dir in source_dirs[1:]:
        other_names = source_class_names(source_dir)
        if other_names != names:
            raise ValueError(f"Class order differs between {source_dirs[0]} and {source_dir}")
    return names


def read_label_file(path: Path, class_names: list[str]) -> tuple[Counter[int], str]:
    """Validate source rows and pad single points for a shared six-slot head."""

    counts: Counter[int] = Counter()
    source = path.read_text(encoding="utf-8")
    rows = []
    keypoint_count, dimensions = keypoint_shape(class_names)
    single_field_count = 5 + dimensions
    full_field_count = 5 + keypoint_count * dimensions
    for line_number, raw_line in enumerate(source.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        prefix = f"{path}:{line_number}"
        if len(fields) not in (single_field_count, full_field_count):
            expected = (
                str(single_field_count) if keypoint_count == 1
                else f"{single_field_count} or {full_field_count}"
            )
            raise ValueError(f"{prefix}: expected {expected} fields, got {len(fields)}")
        try:
            values = [float(field) for field in fields]
        except ValueError as error:
            raise ValueError(f"{prefix}: all fields must be numeric") from error

        if not values[0].is_integer() or not 0 <= values[0] < len(class_names):
            raise ValueError(f"{prefix}: invalid class ID {values[0]}")
        class_id = int(values[0])
        if not all(0.0 <= value <= 1.0 for value in values[1:5]):
            raise ValueError(f"{prefix}: bounding-box values must be in [0, 1]")
        is_core = class_names[class_id] == CUBE_CORE_CLASS
        if is_core and len(fields) != full_field_count:
            raise ValueError(f"{prefix}: {CUBE_CORE_CLASS} requires six keypoint triples")
        if not is_core and any(values[single_field_count:]):
            raise ValueError(f"{prefix}: unused keypoint slots must be 0 0 0")

        point_fields = values[5:] if is_core else values[5:single_field_count]
        for index, offset in enumerate(range(0, len(point_fields), dimensions), 1):
            x, y, visibility = point_fields[offset:offset + dimensions]
            if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
                raise ValueError(f"{prefix}: keypoint {index} coordinates must be in [0, 1]")
            if visibility not in VISIBILITY_VALUES:
                raise ValueError(f"{prefix}: keypoint {index} visibility must be 0, 1, or 2")
            if is_core and visibility == 0 and (x != 0.0 or y != 0.0):
                raise ValueError(f"{prefix}: unlabeled core keypoint {index} must be 0 0 0")
        counts[class_id] += 1
        if keypoint_count > 1:
            fields.extend(["0"] * (full_field_count - len(fields)))
            rows.append(" ".join(fields))
    if keypoint_count > 1:
        source = "".join(f"{row}\n" for row in rows)
    return counts, source


def collect_pairs(source_dir: Path, pose_subdir: str) -> list[tuple[Path, Path]]:
    images_dir = source_dir / "images"
    labels_dir = source_dir / pose_subdir
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing pose directory: {labels_dir}")

    labels = {path.stem: path for path in labels_dir.glob("*.txt")}
    pairs = [
        (image_path, labels[image_path.stem])
        for image_path in sorted(images_dir.iterdir())
        if image_path.suffix.lower() in IMAGE_EXTENSIONS and image_path.stem in labels
    ]
    if not pairs:
        raise ValueError(f"No image/pose pairs found under {source_dir}")
    return pairs


def write_dataset(
    source_dirs: list[Path],
    dataset_dir: Path,
    val_fraction: float,
    seed: int,
    pose_subdir: str,
    class_names: list[str],
) -> Path:
    pairs = [pair for source_dir in source_dirs for pair in collect_pairs(source_dir, pose_subdir)]
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")

    source_counts = Counter()
    for _, label_path in pairs:
        label_counts, _ = read_label_file(label_path, class_names)
        source_counts.update(label_counts)

    random.Random(seed).shuffle(pairs)
    val_count = max(1, round(len(pairs) * val_fraction))
    splits = {"val": pairs[:val_count], "train": pairs[val_count:]}
    if not splits["train"]:
        raise ValueError("The dataset needs at least two labeled images")

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    manifest: list[dict[str, object]] = []
    split_counts: dict[str, dict[str, int]] = {}
    for split, split_pairs in splits.items():
        images_dir = dataset_dir / "images" / split
        labels_dir = dataset_dir / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        counts = Counter()
        for index, (image_path, label_path) in enumerate(split_pairs):
            label_counts, label_text = read_label_file(label_path, class_names)
            counts.update(label_counts)
            stem = f"{split}_{index:06d}"
            target_image = images_dir / f"{stem}{image_path.suffix.lower()}"
            target_label = labels_dir / f"{stem}.txt"
            shutil.copy2(image_path, target_image)
            target_label.write_text(label_text, encoding="utf-8")
            manifest.append(
                {
                    "split": split,
                    "source_image": str(image_path.resolve()),
                    "source_label": str(label_path.resolve()),
                    "image": str(target_image),
                    "label": str(target_label),
                    "objects": sum(label_counts.values()),
                }
            )
        split_counts[split] = {name: counts[index] for index, name in enumerate(class_names)}

    (dataset_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (dataset_dir / "dataset_summary.json").write_text(
        json.dumps(
            {
                "images": len(pairs),
                "class_objects": {name: source_counts[index] for index, name in enumerate(class_names)},
                "splits": split_counts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(
        "path: "
        + str(dataset_dir.resolve())
        + "\ntrain: images/train\nval: images/val\n"
        + f"kpt_shape: {keypoint_shape(class_names)}\n"
        + "names:\n"
        + "".join(f"  {index}: {name}\n" for index, name in enumerate(class_names)),
        encoding="utf-8",
    )
    print(json.dumps(json.loads((dataset_dir / "dataset_summary.json").read_text()), indent=2))
    print(f"Prepared YOLO pose dataset: {data_yaml}")
    return data_yaml


def train(data_yaml: Path, runs_dir: Path, args: argparse.Namespace) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("YOLO training requires: python -m pip install ultralytics") from error

    model_name = args.model or YOLO_POSE_MODEL_FAMILIES[args.model_family][args.model_size]
    print(f"YOLO pose model: {model_name}")
    model = YOLO(model_name)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(runs_dir),
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        cache=args.cache,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        plots=True,
    )
    trainer = getattr(model, "trainer", None)
    best_model_path = getattr(trainer, "best", None)
    if best_model_path is None:
        raise RuntimeError("Training completed without exposing a best-model path")
    best_model_path = Path(best_model_path).resolve()
    if not best_model_path.is_file():
        raise FileNotFoundError(f"Best YOLO pose checkpoint not found: {best_model_path}")
    print(f"Best model: {best_model_path}")
    return best_model_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        "--source-dirs",
        dest="source_dir_groups",
        type=Path,
        nargs="+",
        action="append",
        help="Dataset root(s) containing images/ and pose/; repeat to combine sources.",
    )
    parser.add_argument("--pose-subdir", default="pose")
    parser.add_argument(
        "--class-names", type=parse_class_names, default=None,
        help="Override the source class order. Include cube-core for six-center labels.",
    )
    parser.add_argument("--workdir", type=Path, default=Path("runs/yolo11n_keypoints_local"))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--model-family",
        "--family",
        choices=tuple(YOLO_POSE_MODEL_FAMILIES),
        default="yolo11",
        help="YOLO pose model family; ignored when --model is supplied.",
    )
    parser.add_argument(
        "--model-size",
        choices=tuple(next(iter(YOLO_POSE_MODEL_FAMILIES.values()))),
        default="nano",
        help="YOLO pose model size; ignored when --model is supplied.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Explicit YOLO pose checkpoint override, such as a custom .pt file.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", "--resolution", type=int, default=288)
    parser.add_argument("--batch", type=int, default=-1, help="Batch size; -1 lets Ultralytics auto-select.")
    parser.add_argument("--device", default=None, help="Training device, e.g. 0, cpu, or mps.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--name", default="cube_keypoints_yolo11n_pose")
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.imgsz <= 0:
        parser.error("--imgsz must be positive")
    if args.epochs <= 0 or args.workers < 0 or args.patience < 0:
        parser.error("--epochs must be positive; --workers and --patience must be nonnegative")
    if args.batch == 0 or args.batch < -1:
        parser.error("--batch must be -1 or a positive integer")

    source_dirs = [
        path
        for group in (args.source_dir_groups or [[Path("data/keypoints_only_dataset")]])
        for path in group
    ]
    class_names = resolve_class_names(source_dirs, args.class_names)
    data_yaml = write_dataset(
        source_dirs,
        args.workdir / "dataset",
        args.val_fraction,
        args.seed,
        args.pose_subdir,
        class_names,
    )
    if args.prepare_only:
        return
    train(data_yaml, args.workdir / "runs", args)


if __name__ == "__main__":
    main()
