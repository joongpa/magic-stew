"""Fine-tune RF-DETR Keypoint Preview on independent cube/sticker points.

Sources contain images/ and pose/ with YOLO rows:
    class cx cy width height keypoint_x keypoint_y visibility

Class order comes from the generator's sticker_keypoint_manifest.json or
independent_keypoint_manifest.json, then data.yaml, or --class-names.
--extra-negative-dir retains the YOLO trainer's background-image interface.
Dataset preparation works without RF-DETR installed; training needs .[rfdetr].
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path


DEFAULT_CLASS_NAMES = ["corner", "corner-edge", "corner-edge-center"]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
KEYPOINT_SHAPE = [1, 3]


def parse_class_names(value: str) -> list[str]:
    names = [name.strip() for name in value.split(",")]
    if not all(names) or len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("Class names must be nonempty and unique.")
    return names


def source_class_names(source_dir: Path) -> list[str]:
    """Prefer pose manifests: legacy data.yaml names describe segmentation."""
    import yaml

    for filename in ("sticker_keypoint_manifest.json", "independent_keypoint_manifest.json"):
        path = source_dir / filename
        if path.is_file():
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if metadata.get("keypoint_shape") != KEYPOINT_SHAPE:
                raise ValueError(f"{path}: expected keypoint_shape {KEYPOINT_SHAPE}")
            return parse_class_names(",".join(metadata["class_names"]))
    path = source_dir / "data.yaml"
    if path.is_file():
        metadata = yaml.safe_load(path.read_text(encoding="utf-8"))
        if metadata.get("kpt_shape") != KEYPOINT_SHAPE:
            raise ValueError(f"{path}: expected kpt_shape {KEYPOINT_SHAPE}")
        names = metadata["names"]
        if isinstance(names, dict):
            indexed_names = {int(key): value for key, value in names.items()}
            if sorted(indexed_names) != list(range(len(indexed_names))):
                raise ValueError(f"{path}: class IDs must be contiguous, starting at zero")
            names = [indexed_names[index] for index in range(len(indexed_names))]
        return parse_class_names(",".join(names))
    return DEFAULT_CLASS_NAMES.copy()


def resolve_class_names(source_dirs: list[Path], override: list[str] | None) -> list[str]:
    if override is not None:
        return override
    names = source_class_names(source_dirs[0])
    for source_dir in source_dirs[1:]:
        if source_class_names(source_dir) != names:
            raise ValueError(
                f"Class order differs between sources: {source_dirs[0]} and {source_dir}. "
                "Remap labels before combining datasets."
            )
    return names


def validate_label_file(path: Path, class_names: list[str]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        prefix = f"{path}:{line_number}"
        try:
            values = [float(part) for part in line.split()]
        except ValueError as exc:
            raise ValueError(f"{prefix}: nonnumeric pose label") from exc
        if len(values) != 8 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"{prefix}: expected 8 finite fields for a one-keypoint pose")
        class_id, cx, cy, width, height, x, y, visibility = values
        if not class_id.is_integer() or not 0 <= class_id < len(class_names):
            raise ValueError(f"{prefix}: invalid class ID {class_id}")
        if not all(0 <= value <= 1 for value in (cx, cy, width, height, x, y)):
            raise ValueError(f"{prefix}: box/keypoint coordinates must be normalized to [0, 1]")
        if width <= 0 or height <= 0:
            raise ValueError(f"{prefix}: bounding-box width and height must be positive")
        if visibility not in (0, 1, 2):
            raise ValueError(f"{prefix}: visibility must be 0, 1, or 2")
        counts[int(class_id)] += 1
    return counts


def write_dataset(
    source_dirs: list[Path],
    dataset_dir: Path,
    val_fraction: float,
    seed: int,
    pose_subdir: str = "pose",
    extra_negative_dir: Path | None = None,
    class_names: list[str] | None = None,
) -> Path:
    """Validate all labels, then prepare reproducible train/val splits."""
    import yaml

    if not source_dirs or not 0 <= val_fraction < 1:
        raise ValueError("Supply sources and a validation fraction in [0, 1).")
    source_dirs = list(dict.fromkeys(path.resolve() for path in source_dirs))
    class_names = resolve_class_names(source_dirs, class_names)
    dataset_dir = dataset_dir.resolve()
    inputs = source_dirs + ([extra_negative_dir.resolve()] if extra_negative_dir else [])
    for input_dir in inputs:
        if (
            dataset_dir == input_dir
            or dataset_dir in input_dir.parents
            or input_dir in dataset_dir.parents
        ):
            raise ValueError(f"Prepared dataset must be separate from input directory: {input_dir}")
    if dataset_dir.exists() and not (
        (dataset_dir / "conversion_manifest.json").is_file()
        and (dataset_dir / "data.yaml").is_file()
    ):
        raise FileExistsError(
            f"Refusing to replace an unrecognized dataset directory: {dataset_dir}"
        )

    samples = []
    seen_images: set[Path] = set()
    for source_dir in source_dirs:
        images_dir, labels_dir = source_dir / "images", source_dir / pose_subdir
        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise FileNotFoundError(f"Expected {images_dir} and {labels_dir}")
        images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise ValueError(f"No images found: {images_dir}")
        for image_path in images:
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing pose label: {label_path}")
            counts = validate_label_file(label_path, class_names)
            if image_path.resolve() not in seen_images:
                samples.append((image_path, label_path, counts))
                seen_images.add(image_path.resolve())
    if extra_negative_dir is not None:
        if not extra_negative_dir.is_dir():
            raise FileNotFoundError(f"Missing extra-negative image directory: {extra_negative_dir}")
        for image_path in sorted(extra_negative_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                if image_path.resolve() in seen_images:
                    raise ValueError(
                        f"Image supplied as both labeled and extra-negative: {image_path}"
                    )
                samples.append((image_path.resolve(), None, Counter()))

    # Split foreground and backgrounds separately so negatives cannot displace
    # all labeled validation images in a small dataset.
    rng = random.Random(seed)
    splits: dict[str, list] = {"train": [], "val": []}
    for group in (
        [sample for sample in samples if sample[2]],
        [sample for sample in samples if not sample[2]],
    ):
        rng.shuffle(group)
        val_count = (
            min(len(group) - 1, max(1, round(len(group) * val_fraction)))
            if len(group) > 1 and val_fraction > 0
            else 0
        )
        splits["val"].extend(group[:val_count])
        splits["train"].extend(group[val_count:])

    # No output is replaced until every source label and directory is valid.
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    manifest, summary = [], {}
    for split, split_samples in splits.items():
        rng.shuffle(split_samples)
        images_dir, labels_dir = dataset_dir / split / "images", dataset_dir / split / "labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        class_counts: Counter[int] = Counter()
        for index, (image_path, label_path, counts) in enumerate(split_samples):
            stem = f"{split}_{index:06d}"
            target_image = images_dir / f"{stem}{image_path.suffix.lower()}"
            target_label = labels_dir / f"{stem}.txt"
            shutil.copy2(image_path, target_image)
            if label_path is None:
                target_label.write_text("", encoding="utf-8")
            else:
                shutil.copy2(label_path, target_label)
            class_counts.update(counts)
            manifest.append(
                {
                    "split": split,
                    "source_image": str(image_path),
                    "source_label": str(label_path) if label_path else None,
                    "image": str(target_image),
                    "label": str(target_label),
                    "objects": sum(counts.values()),
                    "negative": not bool(counts),
                }
            )
        summary[split] = {
            "images": len(split_samples),
            "negative_images": sum(not bool(sample[2]) for sample in split_samples),
            "class_objects": {name: class_counts[index] for index, name in enumerate(class_names)},
        }
    (dataset_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (dataset_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_dir),
                "train": "train/images",
                "val": "val/images",
                "names": dict(enumerate(class_names)),
                "kpt_shape": KEYPOINT_SHAPE,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print("Dataset summary:")
    print(json.dumps(summary, indent=2))
    return data_yaml


def train(data_yaml: Path, runs_dir: Path, args: argparse.Namespace) -> Path:
    import yaml

    summary = json.loads((data_yaml.parent / "dataset_summary.json").read_text())
    if any(not sum(summary[split]["class_objects"].values()) for split in ("train", "val")):
        raise ValueError(
            "Training needs labeled images in both splits; use at least two positives."
        )
    try:
        from rfdetr import RFDETRKeypointPreview
    except ImportError as exc:
        raise RuntimeError(
            'RF-DETR Keypoint training requires: python -m pip install -e ".[rfdetr]"'
        ) from exc

    class_names = list(yaml.safe_load(data_yaml.read_text())["names"].values())
    model_kwargs = {
        "num_classes": len(class_names),
        "num_keypoints_per_class": [1] * len(class_names),
        "resolution": args.imgsz,
    }
    if args.model is not None:
        model_kwargs["pretrain_weights"] = str(args.model)
    if args.device is not None:
        model_kwargs["device"] = args.device
    model = RFDETRKeypointPreview(**model_kwargs)

    output_dir = runs_dir / args.name
    suffix = 2
    while output_dir.exists():
        output_dir = runs_dir / f"{args.name}-{suffix}"
        suffix += 1
    train_kwargs = {
        "dataset_dir": str(data_yaml.parent),
        "dataset_file": "yolo",
        "class_names": class_names,
        "output_dir": str(output_dir.resolve()),
        "epochs": args.epochs,
        "batch_size": args.batch,
        "grad_accum_steps": args.grad_accum_steps,
        "lr": args.lr,
        "lr_encoder": args.lr_encoder,
        "num_workers": args.workers,
        "seed": args.seed,
        "early_stopping": args.patience > 0,
        "early_stopping_patience": args.patience,
        "skip_best_epochs": min(10, args.epochs - 1),
        "run_test": False,
        "use_ema": args.ema,
        "tensorboard": args.tensorboard,
        # Synthetic data already supplies camera/lighting diversity. Keep the
        # fixed input size and physical handedness; don't shift hue/color labels.
        "multi_scale": False,
        "expanded_scales": False,
        "scale_jitter": False,
        "aug_config": {},
        "keypoint_flip_pairs": [],
    }
    if args.device is not None:
        train_kwargs["device"] = args.device
    model.train(**train_kwargs)
    best_model_path = output_dir / "checkpoint_best_total.pth"
    if not best_model_path.is_file():
        raise FileNotFoundError(f"Training completed without a best checkpoint: {best_model_path}")
    print(f"Best model: {best_model_path.resolve()}")
    return best_model_path.resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        "--source-dirs",
        dest="source_dir_groups",
        type=Path,
        nargs="+",
        action="append",
        help="Dataset root(s); repeat to combine sources.",
    )
    parser.add_argument("--pose-subdir", default="pose")
    parser.add_argument(
        "--extra-negative-dir",
        type=Path,
        default=None,
        help="Cube-free images to include with empty pose labels (omitted by default).",
    )
    parser.add_argument(
        "--class-names",
        type=parse_class_names,
        help="Comma-separated class names in label ID order; overrides source metadata.",
    )
    parser.add_argument("--workdir", type=Path, default=Path("runs/rfdetr_keypoints_local"))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--model", type=Path, help="Optional pretrained checkpoint override.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", "--resolution", type=int, default=288)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lr-encoder", type=float, default=2e-5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", help="cpu, mps, cuda, cuda:N, or a GPU index such as 0.")
    parser.add_argument("--name", default="cube_keypoints_rfdetr")
    parser.add_argument("--patience", type=int, default=25, help="Early-stop patience; 0 disables.")
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    if args.imgsz <= 0 or args.imgsz % 24:
        parser.error("--imgsz must be a positive multiple of 24 (e.g. 144, 264, 288).")
    if not 0 <= args.val_fraction < 1:
        parser.error("--val-fraction must be in [0, 1).")
    if min(args.epochs, args.batch, args.grad_accum_steps) <= 0:
        parser.error("--epochs, --batch, and --grad-accum-steps must be positive.")
    if args.workers < 0 or args.patience < 0:
        parser.error("--workers and --patience must be nonnegative.")
    if not all(math.isfinite(rate) and rate > 0 for rate in (args.lr, args.lr_encoder)):
        parser.error("Learning rates must be positive and finite.")
    if not args.name or Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("--name must be a single directory name.")
    if args.device is not None:
        args.device = f"cuda:{args.device}" if args.device.isdigit() else args.device
        if args.device not in ("cpu", "mps", "cuda") and not (
            args.device.startswith("cuda:") and args.device[5:].isdigit()
        ):
            parser.error("--device must be cpu, mps, cuda, cuda:N, or a GPU index.")
    return args


def main() -> None:
    args = parse_args()
    source_dirs = [
        path
        for group in (args.source_dir_groups or [[Path("data/synthetic_cube_dataset")]])
        for path in group
    ]
    data_yaml = write_dataset(
        source_dirs,
        args.workdir / "dataset",
        args.val_fraction,
        args.seed,
        args.pose_subdir,
        args.extra_negative_dir,
        args.class_names,
    )
    if args.prepare_only:
        print(f"Prepared dataset: {data_yaml.parent}")
        return
    train(data_yaml, args.workdir / "runs", args)


if __name__ == "__main__":
    main()
