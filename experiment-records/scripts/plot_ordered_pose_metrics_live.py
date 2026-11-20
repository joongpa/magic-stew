#!/usr/bin/env python3
"""Live-plot ordered cube pose losses and layer-specific keypoint errors."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


METRIC_GROUPS = (
    (
        "Box loss",
        ("train/box_loss", "val/box_loss"),
        "loss",
    ),
    (
        "Pose loss",
        ("train/pose_loss", "val/pose_loss"),
        "loss",
    ),
    (
        "Keypoint objectness loss",
        ("train/kobj_loss", "val/kobj_loss"),
        "loss",
    ),
    (
        "Classification and DFL loss",
        ("train/cls_loss", "val/cls_loss", "train/dfl_loss", "val/dfl_loss"),
        "loss",
    ),
    (
        "Visible keypoint error",
        ("val_stationary_kpt_error_px", "val_misaligned_kpt_error_px"),
        "pixels",
    ),
)

DISPLAY_NAMES = {
    "train/box_loss": "train box",
    "val/box_loss": "val box",
    "train/pose_loss": "train pose",
    "val/pose_loss": "val pose",
    "train/kobj_loss": "train kobj",
    "val/kobj_loss": "val kobj",
    "train/cls_loss": "train cls",
    "val/cls_loss": "val cls",
    "train/dfl_loss": "train dfl",
    "val/dfl_loss": "val dfl",
    "val_stationary_kpt_error_px": "stationary points",
    "val_misaligned_kpt_error_px": "misaligned points",
}


def read_metrics(
    path: Path,
    metrics: tuple[str, ...],
) -> tuple[list[float], dict[str, list[float]], str | None]:
    """Read complete numeric rows, tolerating a partially written final row."""
    if not path.is_file():
        return [], {metric: [] for metric in metrics}, f"waiting for {path}"

    epochs: list[float] = []
    values = {metric: [] for metric in metrics}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return [], values, "CSV has no header"
            missing = [metric for metric in metrics if metric not in reader.fieldnames]
            for row in reader:
                try:
                    epoch = float(row["epoch"])
                    parsed = {
                        metric: float(row[metric])
                        for metric in metrics
                        if metric in reader.fieldnames and row.get(metric, "") not in {"", None}
                    }
                except (KeyError, TypeError, ValueError):
                    continue
                if not math.isfinite(epoch):
                    continue
                epochs.append(epoch)
                for metric in metrics:
                    value = parsed.get(metric)
                    values[metric].append(value if value is not None and math.isfinite(value) else math.nan)
    except OSError as exc:
        return [], values, str(exc)

    status = f"missing columns: {', '.join(missing)}" if missing else None
    return epochs, values, status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Ultralytics results.csv to monitor")
    parser.add_argument("--interval", type=float, default=2.0, help="Refresh interval in seconds")
    parser.add_argument("--save", type=Path, help="Continuously save the latest dashboard to this PNG")
    parser.add_argument("--once", action="store_true", help="Render one snapshot and exit")
    args = parser.parse_args()

    if args.interval <= 0:
        parser.error("--interval must be positive")

    all_metrics = tuple(metric for _title, metrics, _ylabel in METRIC_GROUPS for metric in metrics)
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), squeeze=False)
    axes_flat = tuple(axis for row in axes for axis in row)
    fig.canvas.manager.set_window_title("Ordered cube pose training metrics")
    title_text = fig.suptitle("Ordered cube pose metrics", fontsize=14)
    status_text = fig.text(0.01, 0.01, "", ha="left", va="bottom", fontsize=9, color="dimgray")

    def update(_frame: object = None) -> tuple[object, ...]:
        epochs, values, status = read_metrics(args.csv_path, all_metrics)
        all_lines: list[object] = []
        last_epoch = epochs[-1] if epochs else None

        for axis, (title, metrics, ylabel) in zip(axes_flat, METRIC_GROUPS):
            axis.clear()
            plotted = False
            for metric in metrics:
                metric_values = values[metric]
                if not metric_values or not any(math.isfinite(value) for value in metric_values):
                    continue
                axis.plot(
                    epochs,
                    metric_values,
                    linewidth=1.8,
                    label=DISPLAY_NAMES.get(metric, metric),
                )
                plotted = True

            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.3)
            if plotted:
                axis.legend()
            all_lines.extend(axis.lines)

        status_parts = []
        if status:
            status_parts.append(status)
        if last_epoch is not None:
            status_parts.append(f"latest epoch: {last_epoch:g}")
        if values["val_stationary_kpt_error_px"] and values["val_misaligned_kpt_error_px"]:
            stationary = values["val_stationary_kpt_error_px"][-1]
            misaligned = values["val_misaligned_kpt_error_px"][-1]
            if math.isfinite(stationary) and math.isfinite(misaligned):
                status_parts.append(
                    f"latest visible error: stationary={stationary:.2f}px, "
                    f"misaligned={misaligned:.2f}px"
                )
        title_text.set_text(f"{args.csv_path.parent.name} — ordered pose metrics")
        status_text.set_text(" | ".join(status_parts))
        fig.tight_layout(rect=(0, 0.03, 1, 0.96))
        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.save, dpi=150)
        return tuple(all_lines)

    update()
    if args.once:
        return

    animation = FuncAnimation(
        fig,
        update,
        interval=round(args.interval * 1000),
        cache_frame_data=False,
    )
    # Keep a strong reference for Matplotlib's animation timer.
    fig._ordered_pose_metrics_animation = animation  # type: ignore[attr-defined]
    plt.show()


if __name__ == "__main__":
    main()
