"""Sanity-check generated YOLO segmentation labels.

Each rendered cube view should expose 9, 15, or 19 visible piece segments.
This script reports any label file whose non-empty line count is outside that set.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


EXPECTED_COUNTS = {9, 15, 19}


def label_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def class_distribution(lines: list[str]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for line in lines:
        parts = line.split()
        if parts:
            counts[int(float(parts[0]))] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", type=Path, default=Path("synthetic_cube"))
    parser.add_argument(
        "--expected",
        type=int,
        nargs="+",
        default=sorted(EXPECTED_COUNTS),
        help="Allowed label counts per image.",
    )
    parser.add_argument("--show", type=int, default=25, help="Maximum bad files to print.")
    parser.add_argument(
        "--high-threshold",
        type=int,
        default=20,
        help="Print every label file with a count higher than this value.",
    )
    args = parser.parse_args()

    labels_dir = args.dataset / "labels"
    if not labels_dir.exists():
        raise SystemExit(f"Missing labels directory: {labels_dir}")

    expected = set(args.expected)
    label_paths = sorted(labels_dir.glob("*.txt"))
    if not label_paths:
        raise SystemExit(f"No label files found in {labels_dir}")

    count_distribution: Counter[int] = Counter()
    aggregate_class_distribution: Counter[int] = Counter()
    bad = []

    for label_path in label_paths:
        lines = label_lines(label_path)
        count = len(lines)
        count_distribution[count] += 1
        aggregate_class_distribution.update(class_distribution(lines))
        if count not in expected:
            bad.append((label_path, count, class_distribution(lines)))

    print(f"Checked {len(label_paths)} label files in {labels_dir}")
    print("Label count distribution:")
    for count, file_count in sorted(count_distribution.items()):
        marker = "OK" if count in expected else "BAD"
        print(f"  {count:3d}: {file_count:5d} {marker}")

    print("Aggregate class distribution:")
    for class_id, count in sorted(aggregate_class_distribution.items()):
        print(f"  class {class_id}: {count}")

    if bad:
        high_count = [
            (label_path, count, classes)
            for label_path, count, classes in bad
            if count > args.high_threshold
        ]
        if high_count:
            print(f"Files with more than {args.high_threshold} labels:")
            for label_path, count, classes in high_count:
                print(f"  {label_path}: {count} labels, classes={dict(sorted(classes.items()))}")

        print(f"Bad label files: {len(bad)}")
        for label_path, count, classes in bad[: args.show]:
            print(f"  {label_path}: {count} labels, classes={dict(sorted(classes.items()))}")
        if len(bad) > args.show:
            print(f"  ... {len(bad) - args.show} more")
        raise SystemExit(1)

    print(f"All label files have one of the expected counts: {sorted(expected)}")


if __name__ == "__main__":
    main()
