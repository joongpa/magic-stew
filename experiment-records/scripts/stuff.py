"""Debug Ultralytics mask-to-YOLO conversion on the synthetic cube masks.

This intentionally does not run the converter on the original visual grayscale
masks. It first remaps only the configured sparse piece IDs into contiguous
values 1..N, because Ultralytics' converter expects that layout.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics.data.converter import convert_segment_masks_to_yolo_seg


DATASET_DIR = Path("synthetic_cube")
MASKS_DIR = DATASET_DIR / "masks"
DEBUG_DIR = DATASET_DIR / "debug_yolo_conversion"
CONVERTER_MASKS_DIR = DEBUG_DIR / "converter_masks"
CLASS_MASKS_DIR = DEBUG_DIR / "class_masks_for_visual_check"
LABELS_DIR = DEBUG_DIR / "labels"

CLASS_NAMES = ["center", "edge", "corner"]
INSTANCE_VALUES = {
    "center": [20, 40, 60, 80, 100, 120],
    "edge": [130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240],
    "corner": [248, 249, 250, 251, 252, 253, 254, 255],
}
MASK_VALUE_TO_CLASS = {
    value: class_id
    for class_id, class_name in enumerate(CLASS_NAMES)
    for value in INSTANCE_VALUES[class_name]
}


def configured_values() -> list[int]:
    return sorted(MASK_VALUE_TO_CLASS)


def remap_mask_for_ultralytics(mask: np.ndarray) -> np.ndarray:
    remapped = np.zeros_like(mask, dtype=np.uint8)
    for converter_value, mask_value in enumerate(configured_values(), start=1):
        remapped[mask == mask_value] = converter_value
    return remapped


def remap_mask_for_visual_class_check(mask: np.ndarray) -> np.ndarray:
    remapped = np.zeros_like(mask, dtype=np.uint8)
    for mask_value, class_id in MASK_VALUE_TO_CLASS.items():
        remapped[mask == mask_value] = class_id + 1
    return remapped


def print_mappings() -> None:
    print("Configured source mask value -> converter value -> target class:")
    for converter_value, mask_value in enumerate(configured_values(), start=1):
        class_id = MASK_VALUE_TO_CLASS[mask_value]
        print(
            f"  {mask_value:3d} -> {converter_value:2d} -> "
            f"{class_id} ({CLASS_NAMES[class_id]})"
        )


def summarize_original_masks(mask_paths: list[Path]) -> None:
    all_values = set()
    for mask_path in mask_paths:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not read mask: {mask_path}")
        all_values.update(int(value) for value in np.unique(mask))

    expected = {0, *configured_values()}
    unexpected = sorted(all_values - expected)
    missing = sorted(expected - all_values)

    print("Original mask unique values across sample:")
    print(sorted(all_values))
    print("Unexpected original values:", unexpected)
    print("Configured values missing from sample:", missing)


def write_converter_masks(mask_paths: list[Path]) -> None:
    for mask_path in mask_paths:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not read mask: {mask_path}")

        remapped = remap_mask_for_ultralytics(mask)
        class_check = remap_mask_for_visual_class_check(mask)
        print(f"{mask_path.name}: original={np.unique(mask).tolist()}")
        print(f"{mask_path.name}: converter_mask={np.unique(remapped).tolist()}")
        print(f"{mask_path.name}: visual_class_mask={np.unique(class_check).tolist()}")
        cv2.imwrite(str(CONVERTER_MASKS_DIR / mask_path.name), remapped)
        cv2.imwrite(str(CLASS_MASKS_DIR / mask_path.name), class_check)


def remap_label_classes() -> None:
    values = configured_values()
    converter_class_to_target_class = {
        converter_value - 1: MASK_VALUE_TO_CLASS[mask_value]
        for converter_value, mask_value in enumerate(values, start=1)
    }

    for label_path in sorted(LABELS_DIR.glob("*.txt")):
        remapped_lines = []
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            converter_class = int(float(parts[0]))
            target_class = converter_class_to_target_class[converter_class]
            remapped_lines.append(" ".join([str(target_class), *parts[1:]]))
        label_path.write_text("\n".join(remapped_lines) + ("\n" if remapped_lines else ""))


def summarize_labels() -> None:
    total = 0
    per_file = {}
    per_class = {name: 0 for name in CLASS_NAMES}

    for label_path in sorted(LABELS_DIR.glob("*.txt")):
        lines = [line for line in label_path.read_text().splitlines() if line.strip()]
        per_file[label_path.name] = len(lines)
        total += len(lines)
        for line in lines:
            class_id = int(line.split()[0])
            per_class[CLASS_NAMES[class_id]] += 1

    print("YOLO label line counts per file:")
    print(per_file)
    print("Total YOLO segments:", total)
    print("YOLO segments by class:", per_class)


def main() -> None:
    if not MASKS_DIR.exists():
        raise RuntimeError(f"Missing masks directory: {MASKS_DIR}")

    mask_paths = sorted(MASKS_DIR.glob("*.png"))[:10]
    if not mask_paths:
        raise RuntimeError(f"No PNG masks found in {MASKS_DIR}")

    if DEBUG_DIR.exists():
        shutil.rmtree(DEBUG_DIR)
    CONVERTER_MASKS_DIR.mkdir(parents=True)
    CLASS_MASKS_DIR.mkdir(parents=True)
    LABELS_DIR.mkdir(parents=True)

    print_mappings()
    summarize_original_masks(mask_paths)
    write_converter_masks(mask_paths)

    convert_segment_masks_to_yolo_seg(
        masks_dir=str(CONVERTER_MASKS_DIR),
        output_dir=str(LABELS_DIR),
        classes=len(configured_values()),
    )
    remap_label_classes()
    summarize_labels()

    print(f"Debug converter masks: {CONVERTER_MASKS_DIR}")
    print(f"Debug visual class masks: {CLASS_MASKS_DIR}")
    print(f"Debug YOLO labels: {LABELS_DIR}")


if __name__ == "__main__":
    main()
