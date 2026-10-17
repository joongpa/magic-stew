"""Mine realistic whole-cube segmentation failures with model-in-the-loop search.

The Blender generator already randomizes physically plausible cube poses, hands,
lighting, camera framing, and solve-station layouts.  This script samples that
existing distribution, renders ground-truth whole-cube labels, evaluates the
current whole-cube model, and keeps the hardest realistic examples.

Example:

    .venv/bin/python scripts/mine_whole_cube_failures.py \
        --candidate-count 256 \
        --keep 24 \
        --output-dir runs/whole_cube_failure_mining

The search does not invent arbitrary adversarial pixels or modify the model.
It only selects failures from the generator's existing scene distribution.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts/apply_blender_cube_moves.py"
DEFAULT_BLEND_FILE = ROOT / "syntheticAssets/cube.blend"
DEFAULT_MODEL_CANDIDATES = (
    ROOT / "runs/cube_whole_seg_local/runs/cube_whole_yolo11n_seg/weights/best.pt",
    ROOT / "runs/segment/runs/cube_whole_seg_local/runs/cube_whole_yolo11n_seg-4/weights/best.pt",
)
DEFAULT_IMAGE_SIZE = 256
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass(frozen=True)
class FailureScore:
    stem: str
    image_path: str
    label_path: str
    iou: float
    precision: float
    recall: float
    lower_region_recall: float
    ground_truth_area: int
    predicted_area: int
    false_positive_area: int
    hand_false_positive_area: int
    non_hand_false_positive_area: int
    hand_false_positive_fraction: float
    non_hand_false_positive_fraction: float
    predicted: bool
    confidence: float
    failure_score: float


def default_model_path() -> Path:
    return next(
        (path for path in DEFAULT_MODEL_CANDIDATES if path.is_file()),
        DEFAULT_MODEL_CANDIDATES[0],
    )


def resolve_blender(executable: Path | None) -> Path:
    if executable is not None:
        if executable.is_file():
            return executable
        raise FileNotFoundError(f"Blender executable does not exist: {executable}")

    candidates = (
        Path("/Applications/Blender.app/Contents/MacOS/Blender"),
        Path("blender"),
    )
    for candidate in candidates:
        if candidate.is_file() or shutil.which(str(candidate)):
            return candidate
    raise FileNotFoundError(
        "Could not find Blender. Pass --blender with the Blender executable path."
    )


def parse_ground_truth_label(
    label_path: Path,
    width: int,
    height: int,
) -> np.ndarray:
    """Rasterize the generator's one-class YOLO polygon into a binary mask."""

    mask = np.zeros((height, width), dtype=np.uint8)
    if not label_path.is_file():
        return mask.astype(bool)

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7 or (len(parts) - 1) % 2:
            continue
        try:
            coordinates = np.asarray([float(value) for value in parts[1:]], dtype=np.float32)
        except ValueError:
            continue
        points = np.round(
            coordinates.reshape(-1, 2) * np.asarray((width, height), dtype=np.float32)
        ).astype(np.int32)
        if len(points) >= 3:
            cv2.fillPoly(mask, [points.reshape(-1, 1, 2)], 1)
    return mask.astype(bool)


def tensor_to_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def largest_predicted_mask(
    result: object | None,
    output_shape: tuple[int, int],
) -> tuple[np.ndarray | None, float]:
    """Return the largest predicted class-0 mask at original image resolution."""

    if result is None:
        return None, 0.0
    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if masks is None or boxes is None:
        return None, 0.0

    mask_data = tensor_to_numpy(masks.data)
    class_ids = tensor_to_numpy(boxes.cls).reshape(-1)
    confidences = tensor_to_numpy(boxes.conf).reshape(-1)
    if mask_data.ndim < 3 or len(mask_data) == 0:
        return None, 0.0

    output_height, output_width = output_shape
    candidates: list[tuple[int, np.ndarray, float]] = []
    for index, mask in enumerate(mask_data[: len(class_ids)]):
        if int(class_ids[index]) != 0:
            continue
        resized = cv2.resize(
            mask.astype(np.float32),
            (output_width, output_height),
            interpolation=cv2.INTER_NEAREST,
        )
        binary = resized > 0.5
        area = int(binary.sum())
        if area:
            confidence = float(confidences[index]) if index < len(confidences) else 0.0
            candidates.append((area, binary, confidence))
    if not candidates:
        return None, 0.0
    _area, selected, confidence = max(candidates, key=lambda item: item[0])
    return selected, confidence


def lower_region(mask: np.ndarray, fraction: float = 0.65) -> np.ndarray:
    ys, _xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.zeros_like(mask, dtype=bool)
    start = int(np.floor(float(ys.min()) + (float(ys.max()) - float(ys.min())) * fraction))
    result = mask.copy()
    result[:start] = False
    return result


def score_masks(
    ground_truth: np.ndarray,
    prediction: np.ndarray | None,
    confidence: float,
    stem: str,
    image_path: Path,
    label_path: Path,
    visibility: np.ndarray | None = None,
) -> FailureScore:
    predicted = prediction is not None and bool(prediction.any())
    prediction = prediction if predicted else np.zeros_like(ground_truth, dtype=bool)
    intersection = int(np.logical_and(ground_truth, prediction).sum())
    union = int(np.logical_or(ground_truth, prediction).sum())
    ground_truth_area = int(ground_truth.sum())
    predicted_area = int(prediction.sum())
    false_positive = prediction & ~ground_truth
    false_positive_area = int(false_positive.sum())
    hand_false_positive_area = 0
    non_hand_false_positive_area = false_positive_area
    if visibility is not None and visibility.shape == ground_truth.shape:
        hand_pixels = visibility == 2
        hand_false_positive_area = int((false_positive & hand_pixels).sum())
        non_hand_false_positive_area = int(
            (false_positive & ~hand_pixels).sum()
        )
    hand_false_positive_fraction = (
        hand_false_positive_area / false_positive_area
        if false_positive_area
        else 0.0
    )
    non_hand_false_positive_fraction = (
        non_hand_false_positive_area / false_positive_area
        if false_positive_area
        else 0.0
    )
    precision = intersection / predicted_area if predicted_area else 0.0
    recall = intersection / ground_truth_area if ground_truth_area else 0.0
    iou = intersection / union if union else 0.0

    lower_ground_truth = lower_region(ground_truth)
    lower_intersection = int(np.logical_and(lower_ground_truth, prediction).sum())
    lower_area = int(lower_ground_truth.sum())
    lower_recall = lower_intersection / lower_area if lower_area else recall

    # Missing detections dominate, followed by whole-object IoU and the
    # specific lower-region failure seen in the collected examples.
    failure_score = (
        (1.0 if not predicted else 0.0)
        + (1.0 - iou)
        + 0.45 * (1.0 - lower_recall)
    )
    return FailureScore(
        stem=stem,
        image_path=str(image_path),
        label_path=str(label_path),
        iou=iou,
        precision=precision,
        recall=recall,
        lower_region_recall=lower_recall,
        ground_truth_area=ground_truth_area,
        predicted_area=predicted_area,
        false_positive_area=false_positive_area,
        hand_false_positive_area=hand_false_positive_area,
        non_hand_false_positive_area=non_hand_false_positive_area,
        hand_false_positive_fraction=hand_false_positive_fraction,
        non_hand_false_positive_fraction=non_hand_false_positive_fraction,
        predicted=predicted,
        confidence=confidence,
        failure_score=failure_score,
    )


def draw_failure_overlay(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray | None,
    score: FailureScore,
) -> np.ndarray:
    output = image.copy()
    if prediction is None:
        prediction = np.zeros_like(ground_truth, dtype=bool)

    overlay = output.copy()
    overlay[ground_truth] = (0, 210, 0)       # green: ground truth
    overlay[prediction] = (255, 0, 255)       # magenta: prediction
    both = ground_truth & prediction
    overlay[both] = (0, 210, 210)             # yellow: overlap
    output = cv2.addWeighted(overlay, 0.42, output, 0.58, 0.0)

    for mask, color in ((ground_truth, (0, 210, 0)), (prediction, (255, 0, 255))):
        contours, _hierarchy = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(output, contours, -1, color, 2, cv2.LINE_AA)
    cv2.putText(
        output,
        f"IoU {score.iou:.3f}  lower recall {score.lower_region_recall:.3f}  conf {score.confidence:.3f}",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def run_blender_candidates(
    blender: Path,
    blend_file: Path,
    output_dir: Path,
    count: int,
    seed: int,
) -> None:
    command = [
        str(blender),
        "--background",
        "--factory-startup",
        str(blend_file),
        "--python",
        str(GENERATOR),
        "--",
        "--count",
        str(count),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(seed),
        "--no-masks",
    ]
    print("Running Blender candidate generation...")
    subprocess.run(command, cwd=ROOT, check=True)


def metadata_by_stem(candidate_dir: Path) -> dict[str, dict[str, object]]:
    metadata_path = candidate_dir / "metadata.json"
    if not metadata_path.is_file():
        return {}
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        Path(str(entry.get("image", ""))).stem: entry
        for entry in raw
        if isinstance(entry, dict) and entry.get("image")
    }


def evaluate_candidates(
    candidate_dir: Path,
    model_path: Path,
    confidence: float,
    image_size: int,
) -> tuple[list[FailureScore], dict[str, tuple[np.ndarray, np.ndarray | None]]]:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    images_dir = candidate_dir / "images"
    labels_dir = candidate_dir / "labels" / "whole_cube"
    visibility_dir = candidate_dir / "visibility_masks"
    scores: list[FailureScore] = []
    masks: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}

    image_paths = sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Skipping unreadable image: {image_path}", file=sys.stderr)
            continue
        height, width = image.shape[:2]
        label_path = labels_dir / f"{image_path.stem}.txt"
        ground_truth = parse_ground_truth_label(label_path, width, height)
        if not bool(ground_truth.any()):
            print(f"Skipping candidate without a valid cube label: {image_path.stem}")
            continue
        visibility_path = visibility_dir / f"{image_path.stem}.png"
        visibility = cv2.imread(str(visibility_path), cv2.IMREAD_GRAYSCALE)
        if visibility is not None and visibility.shape != ground_truth.shape:
            visibility = cv2.resize(
                visibility,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        results = model.predict(
            source=image,
            conf=float(confidence),
            imgsz=int(image_size),
            retina_masks=True,
            verbose=False,
        )
        result = results[0] if results else None
        prediction, prediction_confidence = largest_predicted_mask(
            result,
            (height, width),
        )
        score = score_masks(
            ground_truth,
            prediction,
            prediction_confidence,
            image_path.stem,
            image_path,
            label_path,
            visibility,
        )
        scores.append(score)
        masks[image_path.stem] = (ground_truth, prediction)
        print(
            f"Evaluated {index}/{len(image_paths)} {image_path.stem}: "
            f"IoU={score.iou:.3f}, lower_recall={score.lower_region_recall:.3f}, "
            f"predicted={score.predicted}, non_hand_fp={score.non_hand_false_positive_area}"
        )
    scores.sort(key=lambda score: score.failure_score, reverse=True)
    return scores, masks


def summarize_conditions(
    scores: list[FailureScore],
    metadata: dict[str, dict[str, object]],
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Aggregate failures by controllable scene factors for follow-up tuning."""

    grouped: dict[str, dict[str, list[FailureScore]]] = {}
    for score in scores:
        entry = metadata.get(score.stem, {})
        camera = entry.get("camera") if isinstance(entry.get("camera"), dict) else {}
        light = entry.get("light") if isinstance(entry.get("light"), dict) else {}
        hands = entry.get("hands") if isinstance(entry.get("hands"), dict) else {}
        layer_state = entry.get("layer_state") if isinstance(entry.get("layer_state"), dict) else {}
        background = entry.get("background") if isinstance(entry.get("background"), dict) else {}
        background_objects = background.get("objects", [])
        background_families = sorted(
            {
                str(obj.get("family"))
                for obj in background_objects
                if isinstance(obj, dict) and obj.get("family")
            }
        ) if isinstance(background_objects, list) else []

        conditions = {
            "camera_viewpoint": str(camera.get("viewpoint", "unknown")),
            "light_setup": str(light.get("setup", "unknown")),
            "hand_visibility": str(hands.get("visibility", "unknown")),
            "layer_state": str(layer_state.get("mode", "unknown")),
            "background_families": "+".join(background_families) or "none",
            "desk_no_hands": str(bool(entry.get("desk_no_hands", False))),
        }
        for factor, value in conditions.items():
            grouped.setdefault(factor, {}).setdefault(value, []).append(score)

    summary: dict[str, dict[str, dict[str, float | int]]] = {}
    for factor, values in grouped.items():
        summary[factor] = {}
        for value, value_scores in values.items():
            summary[factor][value] = {
                "count": len(value_scores),
                "mean_iou": float(np.mean([score.iou for score in value_scores])),
                "mean_lower_region_recall": float(
                    np.mean([score.lower_region_recall for score in value_scores])
                ),
                "mean_failure_score": float(
                    np.mean([score.failure_score for score in value_scores])
                ),
            }
    return summary


def save_failures(
    output_dir: Path,
    candidate_dir: Path,
    scores: list[FailureScore],
    masks: dict[str, tuple[np.ndarray, np.ndarray | None]],
    keep: int,
) -> list[dict[str, object]]:
    failures_dir = output_dir / "failures"
    images_dir = failures_dir / "images"
    overlays_dir = failures_dir / "overlays"
    labels_dir = failures_dir / "labels" / "whole_cube"
    for directory in (images_dir, overlays_dir, labels_dir):
        directory.mkdir(parents=True, exist_ok=True)

    classes_path = candidate_dir / "labels" / "whole_cube" / "classes.json"
    if classes_path.is_file():
        shutil.copy2(classes_path, labels_dir / classes_path.name)

    candidate_metadata = metadata_by_stem(candidate_dir)
    kept_metadata: list[dict[str, object]] = []
    for rank, score in enumerate(scores[:keep], start=1):
        source_image = Path(score.image_path)
        source_label = Path(score.label_path)
        output_stem = f"failure_{rank:04d}_{score.stem}"
        target_image = images_dir / f"{output_stem}{source_image.suffix.lower()}"
        target_label = labels_dir / f"{output_stem}.txt"
        shutil.copy2(source_image, target_image)
        shutil.copy2(source_label, target_label)

        image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
        if image is None:
            continue
        ground_truth, prediction = masks[score.stem]
        overlay = draw_failure_overlay(image, ground_truth, prediction, score)
        cv2.imwrite(str(overlays_dir / f"{output_stem}.png"), overlay)

        entry = {
            "rank": rank,
            "score": asdict(score),
            "source_metadata": candidate_metadata.get(score.stem),
            "image": str(target_image),
            "label": str(target_label),
            "overlay": str(overlays_dir / f"{output_stem}.png"),
        }
        kept_metadata.append(entry)

    (failures_dir / "metadata.json").write_text(
        json.dumps(kept_metadata, indent=2),
        encoding="utf-8",
    )
    return kept_metadata


def cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-count", type=int, default=128)
    parser.add_argument("--keep", type=int, default=24)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/whole_cube_failure_mining")
    parser.add_argument("--blend-file", type=Path, default=DEFAULT_BLEND_FILE)
    parser.add_argument("--blender", type=Path)
    parser.add_argument("--model", type=Path, default=default_model_path())
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMAGE_SIZE)
    return parser.parse_args()


def main() -> int:
    args = cli_args()
    if args.candidate_count <= 0:
        raise SystemExit("--candidate-count must be positive")
    if args.keep <= 0:
        raise SystemExit("--keep must be positive")
    if not args.blend_file.is_file():
        raise SystemExit(f"Could not find Blender scene: {args.blend_file}")
    if not args.model.is_file():
        raise SystemExit(f"Could not find whole-cube model: {args.model}")

    blender = resolve_blender(args.blender)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_dir / f"search_{timestamp}_seed{args.seed}"
    candidate_dir = run_dir / "candidates"
    run_dir.mkdir(parents=True, exist_ok=False)

    run_blender_candidates(
        blender,
        args.blend_file,
        candidate_dir,
        args.candidate_count,
        args.seed,
    )
    scores, masks = evaluate_candidates(
        candidate_dir,
        args.model,
        args.conf,
        args.imgsz,
    )
    kept = save_failures(run_dir, candidate_dir, scores, masks, args.keep)
    all_metadata = metadata_by_stem(candidate_dir)
    summary = {
        "run_dir": str(run_dir),
        "candidate_count": len(scores),
        "kept_count": len(kept),
        "seed": args.seed,
        "model": str(args.model),
        "blend_file": str(args.blend_file),
        "condition_summary": summarize_conditions(scores, all_metadata),
        "scores": [asdict(score) for score in scores],
    }
    (run_dir / "search_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(kept)} failure cases to {run_dir / 'failures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
