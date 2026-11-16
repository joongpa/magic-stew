#!/usr/bin/env python3
"""Train the 18-class, four-corner sticker detector with cyclic center loss.

First relabel existing direct-pose data using scripts/relabel_sticker_quads.py.
Corner and edge order is fixed clockwise; centers accept four cyclic starts.
An existing pose checkpoint can initialize compatible backbone/neck weights.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from live_recognition.sticker_quad_schema import CLASS_NAMES

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
KEYPOINT_SHAPE = [4, 3]
LABEL_FIELD_COUNT = 1 + 4 + KEYPOINT_SHAPE[0] * KEYPOINT_SHAPE[1]
VISIBILITY_VALUES = {0, 1, 2}
DEFAULT_CLASS_NAMES = list(CLASS_NAMES)
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


def source_class_names(source_dir: Path) -> list[str]:
    """Read the generator manifest before falling back to data.yaml."""

    for filename in ("sticker_keypoint_manifest.json", "independent_keypoint_manifest.json"):
        path = source_dir / filename
        if path.is_file():
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if metadata.get("keypoint_shape") != KEYPOINT_SHAPE:
                raise ValueError(f"{path}: expected keypoint_shape {KEYPOINT_SHAPE}")
            return parse_class_names(",".join(metadata["class_names"]))

    data_yaml = source_dir / "data.yaml"
    if data_yaml.is_file():
        import yaml

        metadata = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        if metadata.get("kpt_shape") != KEYPOINT_SHAPE:
            raise ValueError(f"{data_yaml}: expected kpt_shape {KEYPOINT_SHAPE}")
        names = metadata.get("names")
        if isinstance(names, dict):
            indexed = {int(index): str(name) for index, name in names.items()}
            names = [indexed[index] for index in range(len(indexed))]
        if isinstance(names, list):
            return parse_class_names(",".join(names))
    return DEFAULT_CLASS_NAMES.copy()


def resolve_class_names(source_dirs: list[Path], override: list[str] | None) -> list[str]:
    if override is not None and tuple(override) != CLASS_NAMES:
        raise ValueError("This trainer requires the fixed 18 sticker color/type classes")
    names = source_class_names(source_dirs[0])
    for source_dir in source_dirs[1:]:
        other_names = source_class_names(source_dir)
        if other_names != names:
            raise ValueError(f"Class order differs between {source_dirs[0]} and {source_dir}")
    if tuple(names) != CLASS_NAMES:
        raise ValueError("Run scripts/relabel_sticker_quads.py first; this trainer requires the 18 four-corner classes")
    return names


def validate_label_file(path: Path, class_names: list[str]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        prefix = f"{path}:{line_number}"
        if len(fields) != LABEL_FIELD_COUNT:
            raise ValueError(f"{prefix}: expected {LABEL_FIELD_COUNT} fields, got {len(fields)}")
        try:
            values = [float(field) for field in fields]
        except ValueError as error:
            raise ValueError(f"{prefix}: all fields must be numeric") from error

        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"{prefix}: fields must be finite")
        class_id = int(values[0])
        if values[0] != class_id or not 0 <= class_id < len(class_names):
            raise ValueError(f"{prefix}: invalid class ID {values[0]}")
        if not all(0.0 <= value <= 1.0 for value in values[1:5]):
            raise ValueError(f"{prefix}: bounding-box values must be in [0, 1]")
        for index in range(4):
            keypoint_x, keypoint_y, visibility = values[5+index*3:8+index*3]
            if not 0.0 <= keypoint_x <= 1.0 or not 0.0 <= keypoint_y <= 1.0:
                raise ValueError(f"{prefix}: keypoint coordinates must be in [0, 1]")
            if visibility != int(visibility) or int(visibility) not in VISIBILITY_VALUES:
                raise ValueError(f"{prefix}: visibility must be 0, 1, or 2")
        counts[class_id] += 1
    return counts


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
        source_counts.update(validate_label_file(label_path, class_names))

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
            label_counts = validate_label_file(label_path, class_names)
            counts.update(label_counts)
            stem = f"{split}_{index:06d}"
            target_image = images_dir / f"{stem}{image_path.suffix.lower()}"
            target_label = labels_dir / f"{stem}.txt"
            shutil.copy2(image_path, target_image)
            shutil.copy2(label_path, target_label)
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
        + f"kpt_shape: [{KEYPOINT_SHAPE[0]}, {KEYPOINT_SHAPE[1]}]\n"
        + "names:\n"
        + "".join(f"  {index}: {name}\n" for index, name in enumerate(class_names)),
        encoding="utf-8",
    )
    print(json.dumps(json.loads((dataset_dir / "dataset_summary.json").read_text()), indent=2))
    print(f"Prepared YOLO pose dataset: {data_yaml}")
    return data_yaml


def train(data_yaml: Path, runs_dir: Path, args: argparse.Namespace) -> Path:
    try:
        from ultralytics import YOLO, __version__
        from ultralytics.utils.checks import check_version
    except ImportError as error:
        raise RuntimeError("YOLO training requires: python -m pip install ultralytics") from error

    if not check_version(__version__, ">=8.4.185"):
        raise RuntimeError("Sticker quad training requires ultralytics>=8.4.185; run python -m pip install -U ultralytics")
    model_name = args.model or YOLO_POSE_MODEL_FAMILIES[args.model_family][args.model_size]
    print(f"YOLO pose model: {model_name}")
    model = YOLO(model_name)
    from live_recognition.sticker_quad_training import StickerQuadTrainer
    model.train(
        trainer=StickerQuadTrainer,
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
        fliplr=0.0,
        flipud=0.0,  # Corner and edge reflections require different index permutations.
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
    parser.add_argument("--class-names", type=parse_class_names, default=None)
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
    parser.add_argument("--imgsz", "--resolution", type=int, default=128)
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
        for group in (args.source_dir_groups or [[Path("data/sticker_quad_dataset")]])
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
