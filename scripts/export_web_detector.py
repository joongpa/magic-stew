#!/usr/bin/env python3
"""Export both stages of the browser detector and verify PyTorch/ONNX parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))


def source_metadata(source: Path) -> dict:
    return {
        "source": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else source.name,
        "sourceSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=ROOT / "runs/pose/runs/26m-pose/weights/best.pt"
    )
    parser.add_argument("--bbox-model", type=Path, default=ROOT / "runs/500ExamplesWeights.pt")
    parser.add_argument("--bbox-size", type=int, default=640)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "web/public/models")
    parser.add_argument("--sizes", type=int, nargs="+", default=[160, 256, 288])
    args = parser.parse_args()

    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO

    expected_names = [
        "corner",
        "corner-edge",
        "corner-edge-center",
        "white",
        "yellow",
        "orange",
        "red",
        "blue",
        "green",
    ]
    source, bbox_source = args.model.resolve(), args.bbox_model.resolve()
    model, bbox_model = YOLO(str(source)), YOLO(str(bbox_source))
    names = [model.names[i] for i in range(len(model.names))]
    if names != expected_names or list(model.model.model[-1].kpt_shape) != [1, 3]:
        raise ValueError("Expected nine independent pose classes with one 3-channel keypoint")
    if bbox_model.task != "detect" or bbox_model.names != {0: "cube"}:
        raise ValueError("Expected a single-class cube bounding-box detector")
    for candidate, sizes in [(model, args.sizes), (bbox_model, [args.bbox_size])]:
        stride = int(candidate.model.stride.max())
        if any(size < stride or size % stride for size in sizes):
            raise ValueError(f"Input sizes must be positive multiples of model stride {stride}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    verification = []

    # Stage all exports; replace the manifest only after every parity check passes.
    with tempfile.TemporaryDirectory(prefix="web-export-", dir=args.output_dir) as temp:
        staging = Path(temp)

        def export_checked(checkpoint: Path, size: int, channels: int, kind: str) -> dict:
            copied = staging / f"{kind}.pt"
            shutil.copyfile(checkpoint, copied)
            exported = Path(
                YOLO(str(copied)).export(
                    format="onnx",
                    imgsz=size,
                    batch=1,
                    device="cpu",
                    half=False,
                    dynamic=kind == "bbox",
                    simplify=True,
                    opset=17,
                    nms=False,
                    end2end=False,
                )
            )
            onnx.checker.check_model(onnx.load(exported))
            session = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
            image = np.random.default_rng(17).random((1, 3, size, size), dtype=np.float32)
            reference = YOLO(str(copied)).model.float().eval().fuse(verbose=False)
            reference.end2end = False
            with torch.inference_mode():
                raw = reference(torch.from_numpy(image))
                expected = (raw[0] if isinstance(raw, tuple) else raw).numpy()
            actual = session.run(None, {session.get_inputs()[0].name: image})[0]
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=3e-3)
            shape = (1, channels, sum((size // s) ** 2 for s in (8, 16, 32)))
            if actual.shape != shape:
                raise ValueError(f"Unexpected {kind} output: {actual.shape}, expected {shape}")
            if kind == "bbox":
                for height, width in [(384, size), (size, 384)]:
                    rectangle = np.random.default_rng(19).random(
                        (1, 3, height, width), dtype=np.float32
                    )
                    with torch.inference_mode():
                        expected_rect = reference(torch.from_numpy(rectangle))[0].numpy()
                    actual_rect = session.run(None, {session.get_inputs()[0].name: rectangle})[0]
                    np.testing.assert_allclose(actual_rect, expected_rect, rtol=2e-4, atol=3e-3)
                    verification.append(
                        {
                            "stage": kind,
                            "inputShape": [1, 3, height, width],
                            "maxAbsoluteError": float(np.max(np.abs(actual_rect - expected_rect))),
                            "pytorchOnnxParity": "passed",
                        }
                    )
            filename = f"cube-{kind}-{size}.onnx"
            shutil.move(exported, staging / filename)
            verification.append(
                {
                    "stage": kind,
                    "size": size,
                    "maxAbsoluteError": float(np.max(np.abs(actual - expected))),
                    "pytorchOnnxParity": "passed",
                }
            )
            return {
                "size": size,
                "file": filename,
                "bytes": (staging / filename).stat().st_size,
                "inputName": session.get_inputs()[0].name,
                "outputName": session.get_outputs()[0].name,
                "outputShape": list(actual.shape),
            }

        bbox = export_checked(bbox_source, args.bbox_size, 5, "bbox")
        variants = [export_checked(source, size, 16, "pose") for size in dict.fromkeys(args.sizes)]
        manifest = {
            "version": 3,
            "viewSize": 256,
            "name": "Cube acquisition + independent pose tracking",
            "defaultSize": 256 if 256 in args.sizes else args.sizes[0],
            **source_metadata(source),
            "classes": names,
            "classThresholds": [0.25, 0.40, 0.25, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90],
            "keypointShape": [1, 3],
            "presenceThreshold": 0.05,
            "stickerPresenceThreshold": 0.25,
            "iouThreshold": 0.7,
            "maxDetections": 128,
            "layout": "NCHW-xywh-classes-keypoint",
            "variants": variants,
            "bbox": {
                **bbox,
                **source_metadata(bbox_source),
                "classes": ["cube"],
                "threshold": 0.25,
                "dynamic": True,
                "rect": True,
                "stride": 32,
                "layout": "NCHW-xywh-classes",
            },
            "tracking": {
                "roiPadding": 0.4,
                "gain": 1.4,
                "padding": 0.18,
                "presenceThreshold": 0.05,
            },
        }
        for variant in [bbox, *variants]:
            shutil.move(staging / variant["file"], args.output_dir / variant["file"])
        (args.output_dir / "detector.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (args.output_dir / "verification.json").write_text(
            json.dumps(verification, indent=2) + "\n"
        )
    print(json.dumps({"models": [bbox, *variants], "verification": verification}, indent=2))


if __name__ == "__main__":
    main()
