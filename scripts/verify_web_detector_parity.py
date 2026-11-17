#!/usr/bin/env python3
"""Freeze native independent-search inputs/outputs for cross-language web parity tests.

Uses the renderer's actual preprocessing, predict call, presence/object filtering
and ROI tracker. Regenerate after intentional native pipeline/checkpoint changes.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def main():
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    from scripts import render_yolo11n_independent_keypoints as flow
    from live_recognition.roi_tracking import RoiTracker

    output = ROOT / "web/src/lib/fixtures"
    output.mkdir(parents=True, exist_ok=True)
    args = flow.build_parser().parse_args(
        ["--source", "parity-fixture", "--initial-scramble", "", "--roi-padding", ".4"]
    )
    manifest = json.loads((ROOT / "web/public/models/detector.json").read_text())
    report = {
        "ultralyticsVersion": __import__("ultralytics").__version__,
        "reference": "scripts/search_independent_keypoint_full_solve.py -> render_yolo11n_independent_keypoints.py",
        "scriptDefaults": {
            key: getattr(args, key)
            for key in [
                "imgsz",
                "view_size",
                "bbox_imgsz",
                "max_det",
                "roi_padding",
                "roi_tracking_gain",
                "roi_tracking_padding",
                "keypoint_conf",
                "track_conf",
            ]
        },
        "sourceSha256": manifest["sourceSha256"],
        "resize": [],
        "pose": [],
    }
    rng = np.random.default_rng(41)
    for width, height, tw, th, kind in [
        (31, 23, 17, 13, "area"),
        (17, 13, 9, 7, "linear"),
        (7, 5, 13, 11, "area"),
        (16, 16, 8, 8, "area"),
        (15, 15, 8, 8, "area"),
        (13, 9, 17, 15, "linear"),
    ]:
        rgb = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        resized = cv2.resize(
            rgb, (tw, th), interpolation=cv2.INTER_AREA if kind == "area" else cv2.INTER_LINEAR
        )
        report["resize"].append(
            dict(
                width=width,
                height=height,
                targetWidth=tw,
                targetHeight=th,
                kind=kind,
                source=rgb.reshape(-1).tolist(),
                expected=resized.reshape(-1).tolist(),
            )
        )

    image = cv2.imread(str(ROOT / "IMG_9046_cube_crop.jpg"))
    height, width = image.shape[:2]
    roi = (140, 180, 1500, 1540)
    x1, y1, x2, y2 = roi
    view = flow.resize_to_square(image[y1:y2, x1:x2], args.view_size)
    model = flow.load_pose_model(args)
    for size in (160, 256):
        args.imgsz = size
        result = flow.predict_pose_result(model, view, args)
        tensor = model.predictor.preprocess([view])
        with torch.inference_mode():
            raw = model.model(tensor)[0].cpu().numpy().copy()
        raw_name = f"pose-{size}.f32"
        raw.astype("<f4").tofile(output / raw_name)
        points, presence, classes, scores = flow.pose_predictions(result)
        full, local = flow.local_to_frame_points(points, roi, args.view_size)
        mask = flow.usable_pose_points(
            points, presence, classes, scores, args.keypoint_conf, args.pose_class_conf
        )
        expected = [
            dict(
                classId=int(classes[i]),
                score=float(scores[i]),
                presence=float(presence[i]),
                x=float(full[i, 0]),
                y=float(full[i, 1]),
                localX=float(local[i, 0]),
                localY=float(local[i, 1]),
            )
            for i in np.flatnonzero(mask)
        ]
        bounds = flow.tracking_bounds(
            local,
            presence,
            classes,
            scores,
            args.pose_class_conf,
            args.track_conf,
            (y2 - y1, x2 - x1),
        )
        tracker = RoiTracker(roi_xyxy=roi, base_side=x2 - x1, roi_padding=args.roi_padding)
        tracker.observe_tracking_bounds(
            bounds, width, height, args.roi_tracking_gain, args.roi_tracking_padding
        )
        report["pose"].append(
            dict(
                size=size,
                raw=raw_name,
                shape=list(raw.shape),
                sourceWidth=width,
                sourceHeight=height,
                roi=list(roi),
                expected=expected,
                nextRoi=tracker.roi_xyxy,
            )
        )
        print(f"Native pose {size}: {len(expected)} displayed points; next ROI {tracker.roi_xyxy}")

    # A non-square frame exercises native rect=True acquisition (384x640, not 640x640).
    frame = np.full((720, 1280, 3), 40, dtype=np.uint8)
    frame[180:540, 160:520] = cv2.resize(image, (360, 360))
    detector = flow.YOLOCubeDetector(
        str(args.bbox_model), args.bbox_conf, args.bbox_imgsz, args.device
    )
    cube = detector.detect(frame)
    predictor = detector.model.predictor
    tensor = predictor.preprocess([frame])
    with torch.inference_mode():
        raw = detector.model.model(tensor)[0].cpu().numpy().copy()
    raw.astype("<f4").tofile(output / "bbox-rect.f32")
    report["bbox"] = dict(
        raw="bbox-rect.f32",
        shape=list(raw.shape),
        inputShape=list(tensor.shape),
        sourceWidth=1280,
        sourceHeight=720,
        expected=None if cube is None else dict(box=list(cube.bbox_xyxy), score=cube.confidence),
    )
    (output / "native-parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Native bbox input: {list(tensor.shape)}; cube {report['bbox']['expected']}")


if __name__ == "__main__":
    main()
