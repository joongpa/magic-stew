from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path

import cv2

from live_recognition.detectors import SAM3CubeDetector
from live_recognition.types import CubeCandidate


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CLASS_ID = 0
CLASS_NAME = "cube"
SUBSET_NAME = "default"


@dataclass(frozen=True)
class SampledImage:
    source_path: Path
    dataset_path: Path
    width: int
    height: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample frames, run SAM cube detection, and export editable bbox labels."
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=Path("runs") / "cubing_frames",
        help="Directory containing extracted frame images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "cube_bbox_bootstrap",
        help="Dataset output directory.",
    )
    parser.add_argument("--sample-size", type=int, default=500, help="Number of frames to sample.")
    parser.add_argument("--seed", type=int, default=17, help="Random seed for repeatable sampling.")
    parser.add_argument("--model", default="sam3.pt", help="SAM 3 model weights.")
    parser.add_argument("--prompt", default="3x3x3 rubik's cube", help="SAM 3 text prompt.")
    parser.add_argument("--conf", type=float, default=0.4, help="SAM confidence threshold.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the existing output directory.",
    )
    return parser


def iter_images(frames_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in frames_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def sample_images(image_paths: list[Path], sample_size: int, seed: int) -> list[Path]:
    if sample_size <= 0:
        raise SystemExit("--sample-size must be greater than 0")
    if sample_size >= len(image_paths):
        return image_paths
    rng = random.Random(seed)
    return sorted(rng.sample(image_paths, sample_size))


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise SystemExit(f"Output directory already exists: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    (output_dir / "images" / SUBSET_NAME).mkdir(parents=True)
    (output_dir / "annotations").mkdir(parents=True)
    (output_dir / "labels" / SUBSET_NAME).mkdir(parents=True)


def copy_sampled_image(source_path: Path, frames_dir: Path, output_dir: Path) -> SampledImage | None:
    frame = cv2.imread(str(source_path))
    if frame is None:
        print(f"Skipping unreadable image: {source_path}")
        return None

    height, width = frame.shape[:2]
    relative_path = source_path.relative_to(frames_dir)
    digest = sha1(str(relative_path).encode("utf-8")).hexdigest()[:10]
    dataset_name = f"{source_path.stem}_{digest}{source_path.suffix.lower()}"
    dataset_path = output_dir / "images" / SUBSET_NAME / dataset_name
    shutil.copy2(source_path, dataset_path)
    return SampledImage(
        source_path=source_path,
        dataset_path=dataset_path,
        width=width,
        height=height,
    )


def clip_bbox(candidate: CubeCandidate, width: int, height: int) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = candidate.bbox_xyxy
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 0 or box_height <= 0:
        return None
    return float(x1), float(y1), float(box_width), float(box_height)


def write_yolo_label(label_path: Path, boxes_xywh: list[tuple[float, float, float, float]], width: int, height: int) -> None:
    lines = []
    for x, y, box_width, box_height in boxes_xywh:
        x_center = (x + box_width / 2.0) / width
        y_center = (y + box_height / 2.0) / height
        normalized_width = box_width / width
        normalized_height = box_height / height
        lines.append(
            f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {normalized_width:.6f} {normalized_height:.6f}"
        )
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_dataset_metadata(output_dir: Path) -> None:
    (output_dir / "classes.txt").write_text(f"{CLASS_NAME}\n", encoding="utf-8")
    (output_dir / "data.yaml").write_text(
        "path: .\n"
        f"train: images/{SUBSET_NAME}\n"
        f"val: images/{SUBSET_NAME}\n"
        "names:\n"
        f"  {CLASS_ID}: {CLASS_NAME}\n",
        encoding="utf-8",
    )


def write_coco(output_dir: Path, images: list[dict], annotations: list[dict]) -> Path:
    coco_path = output_dir / "annotations" / "instances_default.json"
    coco = {
        "info": {
            "description": "SAM-bootstrapped Rubik's cube bounding boxes",
            "version": "0.1.0",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": CLASS_NAME, "supercategory": "object"}],
    }
    coco_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    return coco_path


def write_flat_cvat_annotations(output_dir: Path, images: list[dict], annotations: list[dict]) -> Path:
    coco_path = output_dir / "annotations" / "instances_cvat_task.json"
    flat_images = []
    for image in images:
        flat_image = dict(image)
        flat_image["file_name"] = Path(flat_image["file_name"]).name
        flat_images.append(flat_image)

    coco = {
        "info": {
            "description": "SAM-bootstrapped Rubik's cube bounding boxes",
            "version": "0.1.0",
        },
        "licenses": [],
        "images": flat_images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": CLASS_NAME, "supercategory": "object"}],
    }
    coco_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    return coco_path


def create_cvat_zip(output_dir: Path) -> Path:
    zip_base = output_dir.parent / f"{output_dir.name}_cvat_coco"
    zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=output_dir, base_dir=".")
    return Path(zip_path)


def create_flat_image_zip(output_dir: Path) -> Path:
    zip_base = output_dir.parent / f"{output_dir.name}_cvat_images"
    zip_path = shutil.make_archive(
        str(zip_base),
        "zip",
        root_dir=output_dir / "images" / SUBSET_NAME,
        base_dir=".",
    )
    return Path(zip_path)


def main() -> int:
    args = build_parser().parse_args()
    if not args.frames_dir.is_dir():
        raise SystemExit(f"Frames directory does not exist: {args.frames_dir}")

    image_paths = iter_images(args.frames_dir)
    if not image_paths:
        raise SystemExit(f"No frame images found in: {args.frames_dir}")

    selected_paths = sample_images(image_paths, args.sample_size, args.seed)
    prepare_output_dir(args.output_dir, args.overwrite)
    write_dataset_metadata(args.output_dir)

    detector = SAM3CubeDetector(
        model_name=args.model,
        confidence_threshold=args.conf,
        prompt=args.prompt,
        return_mask=False,
    )

    coco_images = []
    coco_annotations = []
    annotation_id = 1
    copied_count = 0
    box_count = 0

    for image_id, source_path in enumerate(selected_paths, start=1):
        sampled_image = copy_sampled_image(source_path, args.frames_dir, args.output_dir)
        if sampled_image is None:
            continue

        frame = cv2.imread(str(sampled_image.dataset_path))
        candidates = detector.detect_all(frame)
        boxes = []
        for candidate in candidates:
            box = clip_bbox(candidate, sampled_image.width, sampled_image.height)
            if box is None:
                continue
            boxes.append(box)
            x, y, box_width, box_height = box
            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x, y, box_width, box_height],
                    "segmentation": [],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                    "score": candidate.confidence,
                }
            )
            annotation_id += 1

        label_path = args.output_dir / "labels" / SUBSET_NAME / f"{sampled_image.dataset_path.stem}.txt"
        write_yolo_label(label_path, boxes, sampled_image.width, sampled_image.height)
        coco_images.append(
            {
                "id": image_id,
                "width": sampled_image.width,
                "height": sampled_image.height,
                "file_name": f"{SUBSET_NAME}/{sampled_image.dataset_path.name}",
                "source_path": str(sampled_image.source_path),
            }
        )
        copied_count += 1
        box_count += len(boxes)
        print(f"{copied_count}/{len(selected_paths)} {sampled_image.dataset_path.name}: {len(boxes)} boxes")

    coco_path = write_coco(args.output_dir, coco_images, coco_annotations)
    flat_cvat_path = write_flat_cvat_annotations(args.output_dir, coco_images, coco_annotations)
    zip_path = create_cvat_zip(args.output_dir)
    flat_image_zip_path = create_flat_image_zip(args.output_dir)
    print(f"Done. Sampled {copied_count} images and exported {box_count} boxes.")
    print(f"COCO annotations: {coco_path}")
    print(f"CVAT COCO zip: {zip_path}")
    print(f"CVAT task images zip: {flat_image_zip_path}")
    print(f"CVAT task annotations: {flat_cvat_path}")
    print(f"YOLO labels: {args.output_dir / 'labels'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
