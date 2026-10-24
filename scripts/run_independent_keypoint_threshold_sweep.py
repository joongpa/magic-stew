#!/usr/bin/env python3
"""Evaluate an independent YOLO pose model at several box-confidence thresholds.

The standard Ultralytics confusion matrix is generated once for each supplied
``boxes.conf`` threshold.  The keypoint confidence is intentionally not used
here: it is a point visibility/objectness score and is not the detector score
that controls background false positives.

Example::

    .venv/bin/python scripts/run_independent_keypoint_threshold_sweep.py \
        --model /path/to/fraction_100/weights/best.pt \
        --data /path/to/fraction_100/data.yaml \
        --thresholds 0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50

Outputs are written below ``runs/independent_keypoint_threshold_sweep`` by
default.  Each threshold directory contains Ultralytics' plots and these
additional files:

    confusion_matrix_raw.csv
    confusion_matrix_column_normalized.csv

The matrix convention is the same as the Ultralytics plot: rows are predicted
classes, columns are true classes, and the final row/column is background.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)


def parse_thresholds(value: str) -> tuple[float, ...]:
    """Parse a comma-separated list of distinct confidence thresholds."""

    try:
        thresholds = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("thresholds must be comma-separated numbers") from exc
    if not thresholds:
        raise argparse.ArgumentTypeError("at least one threshold is required")
    if any(not 0.0 <= threshold <= 1.0 for threshold in thresholds):
        raise argparse.ArgumentTypeError("thresholds must be in the interval [0, 1]")
    if len(set(thresholds)) != len(thresholds):
        raise argparse.ArgumentTypeError("thresholds must be unique")
    return tuple(sorted(thresholds))


def threshold_name(threshold: float) -> str:
    """Return a filesystem-safe, stable name for a confidence threshold."""

    return f"conf_{threshold:.3f}".replace(".", "p")


def class_names(model: Any) -> list[str]:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    raise RuntimeError("The loaded model does not expose class names")


def matrix_from_metrics(metrics: Any) -> list[list[float]]:
    confusion = getattr(metrics, "confusion_matrix", None)
    matrix = getattr(confusion, "matrix", None)
    if matrix is None:
        raise RuntimeError("Ultralytics validation did not return a confusion matrix")
    if hasattr(matrix, "tolist"):
        matrix = matrix.tolist()
    return [[float(value) for value in row] for row in matrix]


def normalize_columns(matrix: list[list[float]]) -> list[list[float]]:
    column_count = len(matrix[0]) if matrix else 0
    normalized = [[0.0 for _ in range(column_count)] for _ in matrix]
    for column in range(column_count):
        total = sum(row[column] for row in matrix)
        if total:
            for row in range(len(matrix)):
                normalized[row][column] = matrix[row][column] / total
    return normalized


def write_matrix(path: Path, labels: list[str], matrix: list[list[float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["predicted\\true", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])


def metric_value(metrics: Any, key: str) -> float | None:
    value = getattr(metrics, "results_dict", {}).get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def prepare_local_data_yaml(
    data_yaml: Path,
    output_dir: Path,
    dataset_root: Path | None,
) -> tuple[Path, Path]:
    """Create a local data YAML when the original YAML contains foreign paths.

    Learning-curve datasets are sometimes copied from Windows with a YAML such
    as ``path: C:\\Users\\...``.  On another operating system that path is not
    usable, even though the dataset itself is next to the copied YAML.  Prefer
    an explicit ``--dataset-root``; otherwise fall back to the YAML's parent
    directory when it contains the declared train and validation directories.
    """

    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise RuntimeError(f"Dataset YAML must contain a mapping: {data_yaml}")

    declared_root = config.get("path")
    candidate_roots: list[Path] = []
    if dataset_root is not None:
        candidate_roots.append(dataset_root.expanduser().resolve())
    if declared_root:
        declared_path = Path(str(declared_root).replace("\\", "/")).expanduser()
        if declared_path.is_absolute():
            candidate_roots.append(declared_path)
    candidate_roots.append(data_yaml.parent.resolve())

    root: Path | None = None
    for candidate in candidate_roots:
        if all((candidate / str(config[split])).exists() for split in ("train", "val")):
            root = candidate.resolve()
            break
    if root is None:
        raise FileNotFoundError(
            "Could not resolve dataset paths from "
            f"{data_yaml}. Tried the declared path, the YAML directory, and "
            f"--dataset-root={dataset_root!s}. Pass --dataset-root pointing to "
            "the directory containing images/ and labels/."
        )

    normalized = dict(config)
    normalized["path"] = str(root)
    normalized_path = output_dir / "data_resolved.yaml"
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_path.write_text(yaml.safe_dump(normalized, sort_keys=False), encoding="utf-8")
    return normalized_path, root


def summarize_matrix(
    matrix: list[list[float]],
    names: list[str],
    threshold: float,
    metrics: Any,
    output_dir: Path,
) -> list[dict[str, Any]]:
    expected_size = len(names) + 1
    if len(matrix) != expected_size or any(len(row) != expected_size for row in matrix):
        raise RuntimeError(
            f"Expected a {expected_size}x{expected_size} confusion matrix, "
            f"got {len(matrix)}x{len(matrix[0]) if matrix else 0}"
        )

    labels = [*names, "background"]
    normalized = normalize_columns(matrix)
    write_matrix(output_dir / "confusion_matrix_raw.csv", labels, matrix)
    write_matrix(output_dir / "confusion_matrix_column_normalized.csv", labels, normalized)

    background_index = len(names)
    background_false_positives = sum(matrix[class_id][background_index] for class_id in range(len(names)))
    background_total = sum(row[background_index] for row in matrix)
    summary_rows: list[dict[str, Any]] = []
    for class_id, name in enumerate(names):
        true_count = sum(matrix[row][class_id] for row in range(expected_size))
        predicted_count = sum(matrix[class_id])
        true_positive = matrix[class_id][class_id]
        summary_rows.append(
            {
                "threshold": threshold,
                "class": name,
                "true_positive": true_positive,
                "true_count": true_count,
                "predicted_count": predicted_count,
                "recall": true_positive / true_count if true_count else 0.0,
                "precision": true_positive / predicted_count if predicted_count else 0.0,
                "background_false_positives": matrix[class_id][background_index],
                "background_false_positive_share": (
                    matrix[class_id][background_index] / background_total if background_total else 0.0
                ),
                "missed_objects": matrix[background_index][class_id],
                "map50_pose": metric_value(metrics, "metrics/mAP50(P)"),
                "map50_95_pose": metric_value(metrics, "metrics/mAP50-95(P)"),
                "precision_pose": metric_value(metrics, "metrics/precision(P)"),
                "recall_pose": metric_value(metrics, "metrics/recall(P)"),
                "background_false_positives_total": background_false_positives,
                "background_count": background_total,
            }
        )
    return summary_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Trained YOLO pose checkpoint, normally weights/best.pt")
    parser.add_argument("--data", required=True, type=Path, help="Ultralytics data.yaml used for validation")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Local directory containing the dataset's images/ and labels/; useful after copying a YAML across machines.",
    )
    parser.add_argument("--thresholds", type=parse_thresholds, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--project", type=Path, default=Path("runs/independent_keypoint_threshold_sweep"))
    parser.add_argument("--split", default="val", help="Dataset split to evaluate, normally val")
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16, help="Positive validation batch size")
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device override, e.g. cpu, mps, 0, or 0,1; default lets Ultralytics choose.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold used by validation matching")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.batch <= 0:
        raise SystemExit("--batch must be a positive integer")
    if not 0.0 < args.iou <= 1.0:
        raise SystemExit("--iou must be in the interval (0, 1]")
    if not args.model.is_file():
        raise SystemExit(f"Model checkpoint not found: {args.model}")
    if not args.data.is_file():
        raise SystemExit(f"Dataset YAML not found: {args.data}")

    from ultralytics import YOLO

    args.project.mkdir(parents=True, exist_ok=True)
    resolved_data_yaml, resolved_dataset_root = prepare_local_data_yaml(
        args.data,
        args.project,
        args.dataset_root,
    )
    print(f"Using dataset root: {resolved_dataset_root}")
    print(f"Using normalized data YAML: {resolved_data_yaml}")
    model = YOLO(str(args.model))
    names = class_names(model)
    all_rows: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []

    for threshold in args.thresholds:
        name = threshold_name(threshold)
        output_dir = args.project / name
        normalized_path = output_dir / "confusion_matrix_column_normalized.csv"
        if args.skip_existing and normalized_path.is_file():
            print(f"Skipping existing threshold: {threshold:.3f}")
            continue

        print(f"\n=== boxes.conf >= {threshold:.3f} ===")
        metrics = model.val(
            data=str(resolved_data_yaml),
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            conf=threshold,
            iou=args.iou,
            plots=True,
            project=str(args.project),
            name=name,
            exist_ok=True,
        )
        output_dir = Path(getattr(metrics, "save_dir", output_dir))
        matrix = matrix_from_metrics(metrics)
        rows = summarize_matrix(matrix, names, threshold, metrics, output_dir)
        all_rows.extend(rows)
        run_summaries.append(
            {
                "threshold": threshold,
                "output_dir": str(output_dir.resolve()),
                "background_false_positives": sum(matrix[class_id][-1] for class_id in range(len(names))),
                "background_count": sum(row[-1] for row in matrix),
                "metrics": {
                    key: value
                    for key in (
                        "metrics/precision(P)",
                        "metrics/recall(P)",
                        "metrics/mAP50(P)",
                        "metrics/mAP50-95(P)",
                    )
                    if (value := metric_value(metrics, key)) is not None
                },
            }
        )

    if all_rows:
        fieldnames = list(all_rows[0])
        with (args.project / "threshold_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
    (args.project / "threshold_summary.json").write_text(
        json.dumps(
            {
                "model": str(args.model.resolve()),
                "data": str(args.data.resolve()),
                "resolved_data": str(resolved_data_yaml.resolve()),
                "dataset_root": str(resolved_dataset_root),
                "thresholds": list(args.thresholds),
                "split": args.split,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "device": args.device,
                "iou": args.iou,
                "runs": run_summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSweep output: {args.project.resolve()}")
    print(f"Summary CSV: {(args.project / 'threshold_summary.csv').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
