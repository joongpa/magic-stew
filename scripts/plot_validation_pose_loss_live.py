#!/usr/bin/env python3
"""Live-plot validation pose loss from two Ultralytics results.csv files."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


DEFAULT_METRIC = "val/pose_loss"


def read_metric(path: Path, metric: str) -> tuple[list[float], list[float], str | None]:
    """Read complete numeric rows, tolerating a partially written final row."""
    if not path.is_file():
        return [], [], f"waiting for {path}"

    epochs: list[float] = []
    values: list[float] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return [], [], "CSV has no header"
            if metric not in reader.fieldnames:
                return [], [], f"missing column {metric!r}"
            for row in reader:
                try:
                    epoch = float(row["epoch"])
                    value = float(row[metric])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(epoch) and math.isfinite(value):
                    epochs.append(epoch)
                    values.append(value)
    except OSError as exc:
        return [], [], str(exc)

    return epochs, values, None


def display_label(path: Path, label: str | None) -> str:
    return label or path.parent.name or path.stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_a", type=Path, help="First Ultralytics results.csv")
    parser.add_argument("csv_b", type=Path, help="Second Ultralytics results.csv")
    parser.add_argument("--label-a", help="Legend label for the first run")
    parser.add_argument("--label-b", help="Legend label for the second run")
    parser.add_argument("--metric", default=DEFAULT_METRIC, help="CSV column to plot")
    parser.add_argument("--interval", type=float, default=2.0, help="Refresh interval in seconds")
    parser.add_argument("--save", type=Path, help="Continuously save the latest graph to this PNG")
    parser.add_argument("--once", action="store_true", help="Render one snapshot and exit")
    args = parser.parse_args()

    if args.interval <= 0:
        parser.error("--interval must be positive")

    label_a = display_label(args.csv_a, args.label_a)
    label_b = display_label(args.csv_b, args.label_b)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.canvas.manager.set_window_title("Validation pose loss comparison")

    def update(_frame: object = None) -> tuple[object, ...]:
        epochs_a, values_a, status_a = read_metric(args.csv_a, args.metric)
        epochs_b, values_b, status_b = read_metric(args.csv_b, args.metric)

        ax.clear()
        if values_a:
            ax.plot(epochs_a, values_a, marker="o", markersize=3, linewidth=2, label=label_a)
        if values_b:
            ax.plot(epochs_b, values_b, marker="o", markersize=3, linewidth=2, label=label_b)

        ax.set_title(f"{args.metric} — live comparison")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(args.metric)
        ax.grid(True, alpha=0.3)
        if values_a or values_b:
            ax.legend()

        status_parts = []
        for label, epochs, values, status in (
            (label_a, epochs_a, values_a, status_a),
            (label_b, epochs_b, values_b, status_b),
        ):
            if status:
                status_parts.append(f"{label}: {status}")
            elif values:
                best_index = min(range(len(values)), key=values.__getitem__)
                status_parts.append(
                    f"{label}: {len(values)} epochs, best {values[best_index]:.5g} "
                    f"(epoch {epochs[best_index]:g})"
                )
        ax.text(
            0.01,
            0.01,
            "\n".join(status_parts),
            transform=ax.transAxes,
            va="bottom",
            fontsize=9,
            color="dimgray",
        )
        fig.tight_layout()
        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.save, dpi=150)
        return tuple(ax.lines)

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
    fig._validation_pose_loss_animation = animation  # type: ignore[attr-defined]
    plt.show()


if __name__ == "__main__":
    main()
