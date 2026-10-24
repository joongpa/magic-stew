#!/usr/bin/env python3
"""Run a fixed-validation learning-curve experiment for independent keypoints.

The experiment trains the same YOLO pose model independently on nested subsets
of the training split, while keeping one validation split fixed across every
run. This makes it possible to tell whether adding data is still improving the
model or whether the current architecture/labels have reached a plateau.

Example with automatic device selection::

    .venv/bin/python scripts/run_independent_keypoint_learning_curve.py \
        --source-dir data/independent_keypoints_dataset_v2 \
        --batch -1 \
        --workers 8 \
        --epochs 100

Outputs are written below ``runs/independent_keypoint_learning_curve`` by
default:

    split_manifest.json
    datasets/fraction_010/{train,val}/...
    runs/fraction_010/...
    learning_curve.csv
    learning_curve.json

The fraction subsets are nested and are selected from the same deterministic
training order. Each run starts from the same pretrained checkpoint; runs are
never warm-started from another fraction.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_yolo11n_piece_seg_local import (  # noqa: E402
    CLASS_NAMES,
    DEFAULT_EXTRA_NEGATIVE_DIR,
    IMAGE_EXTENSIONS,
    collect_pairs,
    label_lines,
    validate_label_file,
)  # noqa: E402


DEFAULT_SOURCE_DIR = Path("data/independent_keypoints_dataset_v2")
DEFAULT_WORKDIR = Path("runs/independent_keypoint_learning_curve")
DEFAULT_FRACTIONS = (0.10, 0.25, 0.50, 1.0)
KPT_SHAPE = (1, 3)


@dataclass(frozen=True)
class Pair:
    image: Path
    label: Path | None


def parse_fractions(value: str) -> tuple[float, ...]:
    fractions = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not fractions:
        raise argparse.ArgumentTypeError("at least one fraction is required")
    if any(not 0.0 < fraction <= 1.0 for fraction in fractions):
        raise argparse.ArgumentTypeError("fractions must be in the interval (0, 1]")
    if len(set(fractions)) != len(fractions):
        raise argparse.ArgumentTypeError("fractions must be unique")
    return tuple(sorted(fractions))


def fraction_name(fraction: float) -> str:
    return f"fraction_{round(fraction * 100):03d}"


def collect_all_pairs(
    source_dirs: list[Path],
    pose_subdir: str,
    extra_negative_dir: Path | None,
) -> list[Pair]:
    pairs: list[Pair] = []
    for source_dir in source_dirs:
        pairs.extend(Pair(image, label) for image, label in collect_pairs(source_dir, pose_subdir))

    if extra_negative_dir is not None:
        if not extra_negative_dir.is_dir():
            raise FileNotFoundError(f"Missing extra-negative image directory: {extra_negative_dir}")
        pairs.extend(
            Pair(path, None)
            for path in sorted(extra_negative_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    if not pairs:
        raise RuntimeError("No image/label pairs found")
    return pairs


def validate_pairs(pairs: list[Pair]) -> dict[str, Any]:
    class_counts: Counter[int] = Counter()
    negative_images = 0
    empty_label_images = 0
    for pair in pairs:
        if pair.label is None:
            negative_images += 1
            continue
        lines = label_lines(pair.label)
        if not lines:
            empty_label_images += 1
        class_counts.update(validate_label_file(pair.label))
    return {
        "images": len(pairs),
        "negative_images": negative_images,
        "empty_label_images": empty_label_images,
        "objects_by_class": {
            CLASS_NAMES[class_id]: class_counts[class_id]
            for class_id in range(len(CLASS_NAMES))
        },
    }


def make_fixed_split(
    pairs: list[Pair],
    val_fraction: float,
    seed: int,
) -> tuple[list[Pair], list[Pair]]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    ordered = list(pairs)
    random.Random(seed).shuffle(ordered)
    val_count = max(1, round(len(ordered) * val_fraction)) if len(ordered) > 1 and val_fraction else 0
    if val_count >= len(ordered):
        raise ValueError("validation split must leave at least one training image")
    return ordered[val_count:], ordered[:val_count]


def write_split_dataset(
    dataset_dir: Path,
    train_pairs: list[Pair],
    val_pairs: list[Pair],
    *,
    overwrite: bool,
) -> Path:
    if dataset_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Dataset already exists: {dataset_dir}; use --overwrite-datasets to rebuild it"
            )
        shutil.rmtree(dataset_dir)

    manifest: list[dict[str, Any]] = []
    for split, pairs in (("train", train_pairs), ("val", val_pairs)):
        images_dir = dataset_dir / "images" / split
        labels_dir = dataset_dir / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        for index, pair in enumerate(pairs):
            stem = f"{split}_{index:06d}"
            target_image = images_dir / f"{stem}{pair.image.suffix.lower()}"
            target_label = labels_dir / f"{stem}.txt"
            shutil.copy2(pair.image, target_image)
            if pair.label is None:
                target_label.write_text("", encoding="utf-8")
            else:
                shutil.copy2(pair.label, target_label)
            manifest.append(
                {
                    "split": split,
                    "source_image": str(pair.image.resolve()),
                    "source_label": str(pair.label.resolve()) if pair.label is not None else None,
                    "image": str(target_image.resolve()),
                    "label": str(target_label.resolve()),
                    "negative": pair.label is None,
                    "objects": len(label_lines(pair.label)) if pair.label is not None else 0,
                }
            )

    (dataset_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {dataset_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"kpt_shape: [{KPT_SHAPE[0]}, {KPT_SHAPE[1]}]\n"
        "names:\n"
        + "".join(f"  {index}: {name}\n" for index, name in enumerate(CLASS_NAMES)),
        encoding="utf-8",
    )
    return data_yaml


def write_split_manifest(
    path: Path,
    train_pairs: list[Pair],
    val_pairs: list[Pair],
    fractions: tuple[float, ...],
    config: dict[str, Any],
) -> None:
    train_order = [
        {
            "image": str(pair.image.resolve()),
            "label": str(pair.label.resolve()) if pair.label is not None else None,
        }
        for pair in train_pairs
    ]
    val_order = [
        {
            "image": str(pair.image.resolve()),
            "label": str(pair.label.resolve()) if pair.label is not None else None,
        }
        for pair in val_pairs
    ]
    path.write_text(
        json.dumps(
            {
                "config": config,
                "fractions": list(fractions),
                "train_order": train_order,
                "val_order": val_order,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def jsonable_metrics(metrics: Any) -> dict[str, float]:
    result = getattr(metrics, "results_dict", {})
    output: dict[str, float] = {}
    for key, value in result.items():
        try:
            output[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return output


def best_epoch(results_csv: Path) -> tuple[int | None, dict[str, float]]:
    if not results_csv.is_file():
        return None, {}
    with results_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None, {}
    metric_name = "metrics/mAP50-95(P)"
    best = max(rows, key=lambda row: float(row.get(metric_name, "-inf")))
    metrics = {}
    for key, value in best.items():
        if key == "epoch":
            continue
        try:
            metrics[key] = float(value)
        except (TypeError, ValueError):
            continue
    return int(float(best["epoch"])), metrics


def train_fraction(
    data_yaml: Path,
    run_dir: Path,
    run_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from ultralytics import YOLO

    best_path = run_dir / "weights" / "best.pt"
    if args.skip_existing and best_path.is_file():
        print(f"Skipping existing run: {run_name}")
    else:
        model = YOLO(args.model)
        model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=str(args.runs_dir),
            name=run_name,
            exist_ok=True,
            patience=args.patience,
            cache=args.cache,
            mosaic=args.mosaic,
            plots=True,
            seed=args.seed,
        )
        trainer = getattr(model, "trainer", None)
        trainer_best = getattr(trainer, "best", None)
        if trainer_best is not None:
            best_path = Path(trainer_best).resolve()
        if not best_path.is_file():
            raise FileNotFoundError(f"Training did not produce {best_path}")

    best_model = YOLO(str(best_path))
    metrics = best_model.val(
        data=str(data_yaml),
        split="val",
        imgsz=args.imgsz,
        batch=args.val_batch,
        device=args.device,
        workers=args.workers,
        plots=True,
    )
    epoch, csv_metrics = best_epoch(run_dir / "results.csv")
    val_metrics = jsonable_metrics(metrics)
    return {
        "best_model": str(best_path),
        "best_epoch": epoch,
        "results_csv": str((run_dir / "results.csv").resolve()),
        "metrics": csv_metrics or val_metrics,
        "validation_metrics": val_metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        "--source-dirs",
        dest="source_dir_groups",
        type=Path,
        nargs="+",
        action="append",
        default=None,
        help="Dataset root(s) containing images/ and pose/.",
    )
    parser.add_argument("--pose-subdir", default="pose")
    parser.add_argument(
        "--extra-negative-dir",
        type=Path,
        default=DEFAULT_EXTRA_NEGATIVE_DIR,
        help=f"Cube-free image directory; use --no-extra-negatives to disable (default: {DEFAULT_EXTRA_NEGATIVE_DIR}).",
    )
    parser.add_argument("--no-extra-negatives", action="store_true")
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--fractions", type=parse_fractions, default=DEFAULT_FRACTIONS)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--model", default="yolo11n-pose.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument(
        "--val-batch",
        type=int,
        default=16,
        help="Positive validation batch size; training --batch may use -1 for auto-sizing.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Training device override, e.g. cpu, mps, 0, or 0,1; default lets Ultralytics choose.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--mosaic", type=float, default=0.0)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--overwrite-datasets",
        action="store_true",
        help="Rebuild generated fraction datasets if they already exist.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.val_batch <= 0:
        raise SystemExit("--val-batch must be a positive integer")
    source_dirs = [
        source_dir
        for group in (args.source_dir_groups or [[DEFAULT_SOURCE_DIR]])
        for source_dir in group
    ]
    extra_negative_dir = None if args.no_extra_negatives else args.extra_negative_dir
    all_pairs = collect_all_pairs(source_dirs, args.pose_subdir, extra_negative_dir)
    input_summary = validate_pairs(all_pairs)
    train_pairs, val_pairs = make_fixed_split(all_pairs, args.val_fraction, args.seed)
    if not train_pairs:
        raise RuntimeError("The fixed split produced no training images")

    args.workdir.mkdir(parents=True, exist_ok=True)
    args.runs_dir = args.workdir / "runs"
    config = {
        "source_dirs": [str(path.resolve()) for path in source_dirs],
        "extra_negative_dir": str(extra_negative_dir.resolve()) if extra_negative_dir else None,
        "pose_subdir": args.pose_subdir,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "model": args.model,
        "imgsz": args.imgsz,
        "device": args.device,
        "epochs": args.epochs,
        "batch": args.batch,
        "val_batch": args.val_batch,
        "workers": args.workers,
        "input_summary": input_summary,
        "train_images": len(train_pairs),
        "val_images": len(val_pairs),
    }
    write_split_manifest(args.workdir / "split_manifest.json", train_pairs, val_pairs, args.fractions, config)

    summaries: list[dict[str, Any]] = []
    for fraction in args.fractions:
        train_count = max(1, round(len(train_pairs) * fraction))
        fraction_train = train_pairs[:train_count]
        name = fraction_name(fraction)
        dataset_dir = args.workdir / "datasets" / name
        existing_data_yaml = dataset_dir / "data.yaml"
        if dataset_dir.exists() and not args.overwrite_datasets:
            if not existing_data_yaml.is_file():
                raise FileExistsError(
                    f"Dataset directory exists without data.yaml: {dataset_dir}; "
                    "use --overwrite-datasets to rebuild it"
                )
            data_yaml = existing_data_yaml
        else:
            data_yaml = write_split_dataset(
                dataset_dir,
                fraction_train,
                val_pairs,
                overwrite=args.overwrite_datasets,
            )
        print(f"\n=== {name}: {len(fraction_train)} train / {len(val_pairs)} val images ===")
        summary = {
            "fraction": fraction,
            "train_images": len(fraction_train),
            "val_images": len(val_pairs),
            "dataset": str(data_yaml.resolve()),
        }
        if args.prepare_only:
            summaries.append(summary)
            continue
        run_dir = args.runs_dir / name
        summary.update(train_fraction(data_yaml, run_dir, name, args))
        summaries.append(summary)

    (args.workdir / "learning_curve.json").write_text(
        json.dumps({"config": config, "runs": summaries}, indent=2),
        encoding="utf-8",
    )
    if summaries:
        fieldnames = ["fraction", "train_images", "val_images", "dataset", "best_model", "best_epoch"]
        metric_names = sorted(
            {
                metric_name
                for summary in summaries
                for metric_name in summary.get("metrics", {})
            }
        )
        fieldnames.extend(metric_names)
        with (args.workdir / "learning_curve.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for summary in summaries:
                row = {key: summary.get(key) for key in fieldnames}
                row.update(summary.get("metrics", {}))
                writer.writerow(row)

    print(f"\nExperiment output: {args.workdir.resolve()}")
    if args.prepare_only:
        print("Datasets prepared; no training was run.")
    else:
        print(f"Summary CSV: {(args.workdir / 'learning_curve.csv').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
