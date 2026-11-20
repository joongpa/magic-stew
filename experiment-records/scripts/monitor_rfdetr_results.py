#!/usr/bin/env python3
"""Monitor RF-DETR's sparse results.csv while training is running.

RF-DETR writes multiple rows per epoch: training-step rows, learning-rate
rows, and a final row containing validation metrics. This monitor groups rows
by epoch and chooses the most populated row as the epoch summary, so the CSV
is readable without pandas or a notebook.

Examples:

    .venv/bin/python scripts/monitor_rfdetr_results.py \
        --csv runs/pose/runs/rfdetr/results.csv

    .venv/bin/python scripts/monitor_rfdetr_results.py \
        --csv runs/pose/runs/rfdetr/results.csv --once --history 12
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class ResultRow:
    epoch: int
    step: int
    values: dict[str, float]
    source_row: int


def parse_number(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def unique_headers(raw_headers: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    headers: list[str] = []
    for index, raw in enumerate(raw_headers):
        base = raw.strip() or f"column_{index}"
        count = counts.get(base, 0)
        counts[base] = count + 1
        headers.append(base if count == 0 else f"{base}_{count + 1}")
    return headers


def read_results(path: Path) -> tuple[list[str], list[ResultRow]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            headers = unique_headers(next(reader))
        except StopIteration:
            return [], []

        rows: list[ResultRow] = []
        for source_row, raw in enumerate(reader, 2):
            if len(raw) < 2:
                continue
            try:
                epoch = int(float(raw[0]))
                step = int(float(raw[1]))
            except ValueError:
                # Tolerate a partially written or non-data row.
                continue

            values: dict[str, float] = {}
            for index, value in enumerate(raw[2:], 2):
                number = parse_number(value)
                if number is not None and index < len(headers):
                    values[headers[index]] = number
            rows.append(ResultRow(epoch, step, values, source_row))
    return headers, rows


def epoch_summaries(rows: list[ResultRow]) -> list[ResultRow]:
    """Return the most populated row for each epoch, preserving epoch order."""

    def summary_rank(row: ResultRow) -> tuple[bool, int, int, int]:
        # RF-DETR writes a dense training row and a slightly sparser
        # validation-summary row for the same epoch. Prefer validation when it
        # exists; otherwise fall back to the most populated training row.
        validation_fields = sum(name.startswith("val/") for name in row.values)
        return (
            validation_fields > 0,
            validation_fields,
            len(row.values),
            row.source_row,
        )

    summaries: dict[int, ResultRow] = {}
    for row in rows:
        previous = summaries.get(row.epoch)
        if previous is None or summary_rank(row) >= summary_rank(previous):
            summaries[row.epoch] = row
    return [summaries[epoch] for epoch in sorted(summaries)]


def normalized_name(name: str) -> str:
    return "".join(character.lower() for character in name if character.isalnum())


def metric_priority(name: str) -> tuple[int, str]:
    normalized = normalized_name(name)
    score = 0
    if "val" in normalized:
        score += 30
    if any(token in normalized for token in ("map", "mar", "precision", "recall", "f1")):
        score += 25
    if "loss" in normalized:
        score += 15
    if normalized.startswith("lr") or "learningrate" in normalized:
        score += 10
    if "train" in normalized:
        score -= 3
    return score, name


def useful_columns(headers: list[str], summaries: list[ResultRow], limit: int) -> list[str]:
    populated = {
        header
        for summary in summaries
        for header in summary.values
        if header not in {"epoch", "step"}
    }
    ranked = sorted(populated, key=lambda name: metric_priority(name), reverse=True)
    return ranked[:limit]


def format_value(value: float | None) -> str:
    if value is None:
        return ""
    if abs(value) >= 1000 or (0 < abs(value) < 0.001):
        return f"{value:.4g}"
    return f"{value:.6g}"


def print_table(summaries: list[ResultRow], columns: list[str]) -> None:
    if not summaries:
        print("No complete result rows yet.")
        return
    # The caller prints history separately; this table is intentionally kept
    # to the latest summary for a compact live view.
    latest = summaries[-1]
    values = [str(latest.epoch), str(latest.step)] + [
        format_value(latest.values.get(column)) for column in columns
    ]
    headings = ["epoch", "step", *columns]
    widths = [max(len(heading), len(value)) for heading, value in zip(headings, values)]
    print(" | ".join(value.ljust(width) for value, width in zip(headings, widths)))
    print("-+-".join("-" * width for width in widths))
    print(" | ".join(value.rjust(width) for value, width in zip(values, widths)))


def print_history(summaries: list[ResultRow], columns: list[str], history: int) -> None:
    if not summaries:
        return
    selected = summaries[-history:]
    headings = ["epoch", "step", *columns]
    rendered = [
        [str(row.epoch), str(row.step), *[format_value(row.values.get(column)) for column in columns]]
        for row in selected
    ]
    widths = [
        max(len(headings[index]), *(len(row[index]) for row in rendered))
        for index in range(len(headings))
    ]
    print("\nRecent epoch summaries:")
    print(" | ".join(value.ljust(width) for value, width in zip(headings, widths)))
    print("-+-".join("-" * width for width in widths))
    for row in rendered:
        print(" | ".join(value.rjust(width) for value, width in zip(row, widths)))


def print_latest(path: Path, history: int, max_columns: int, clear: bool) -> tuple[int, int] | None:
    if not path.is_file():
        print(f"Waiting for results file: {path}")
        return None
    try:
        headers, rows = read_results(path)
    except (OSError, csv.Error) as error:
        print(f"Could not read {path}: {error}")
        return None
    summaries = epoch_summaries(rows)
    if clear:
        sys.stdout.write("\033[2J\033[H")
    print(f"RF-DETR results monitor: {path}")
    print(f"Updated: {datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec='seconds')}")
    if not summaries:
        print("No numeric training rows yet.")
        return None

    validation_summaries = [
        summary
        for summary in summaries
        if any(name.startswith("val/") for name in summary.values)
    ]
    display_summaries = validation_summaries or summaries
    columns = useful_columns(headers, display_summaries, max_columns)
    latest = display_summaries[-1]
    print(f"\nLatest epoch: {latest.epoch} | step: {latest.step} | populated fields: {len(latest.values)}")
    print_table(display_summaries, columns)
    print_history(display_summaries, columns, history)
    if summaries[-1].epoch != latest.epoch:
        print(f"\nCurrent training epoch in progress: {summaries[-1].epoch}")
    return latest.epoch, latest.step


def resolve_csv_path(path: Path) -> Path:
    return path / "results.csv" if path.is_dir() else path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="results.csv path or its containing directory")
    parser.add_argument("--interval", type=float, default=10.0, help="Refresh interval in seconds")
    parser.add_argument("--history", type=int, default=8, help="Number of epoch summaries to show")
    parser.add_argument("--max-columns", type=int, default=10, help="Metric columns shown")
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit")
    parser.add_argument("--no-clear", action="store_true", help="Keep previous snapshots in the terminal")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.interval <= 0 or args.history <= 0 or args.max_columns <= 0:
        raise SystemExit("--interval, --history, and --max-columns must be positive")
    path = resolve_csv_path(args.csv)

    while True:
        print_latest(path, args.history, args.max_columns, clear=not args.no_clear)
        if args.once:
            return 0
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
