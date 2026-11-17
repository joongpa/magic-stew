#!/usr/bin/env python3
"""Render the 21 independent landmark/color-piece-type points in full frames."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from live_recognition.typed_sticker_keypoints import CLASS_NAMES, COLORS
from scripts import render_yolo_sticker_quads as flow

DEFAULT_CLASS_CONF = {0: 0.25, 1: 0.40, 2: 0.25, **dict.fromkeys(range(3, 21), 0.90)}


def parse_class_conf(value):
    thresholds = {}
    for item in value.split(","):
        try:
            raw_name, number = item.rsplit("=", 1)
            raw_name = raw_name.strip().lower()
            score = float(number)
            name = " ".join(raw_name.replace("_", " ").replace("-", " ").split())
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError
            if raw_name in CLASS_NAMES:
                ids = [CLASS_NAMES.index(raw_name)]
            elif name in CLASS_NAMES:
                ids = [CLASS_NAMES.index(name)]
            elif name == "landmarks":
                ids = list(range(3))
            elif name == "stickers":
                ids = list(range(3, 21))
            elif name in COLORS:
                ids = [i for i in range(3, 21) if CLASS_NAMES[i].startswith(name + " ")]
            elif name in ("corner stickers", "edge", "center", "center stickers"):
                kind = {"corner stickers": "corner", "center stickers": "center"}.get(name, name)
                ids = [i for i in range(3, 21) if CLASS_NAMES[i].endswith(" " + kind)]
            elif raw_name.isdigit() and 0 <= int(raw_name) < 21:
                ids = [int(raw_name)]
            else:
                raise ValueError
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "Use class/group=value, e.g. corner-edge=0.4,stickers=0.9,blue_corner=0.8"
            ) from exc
        thresholds.update(dict.fromkeys(ids, score))
    return thresholds


def extract_points(result):
    boxes, keypoints = getattr(result, "boxes", None), getattr(result, "keypoints", None)
    if boxes is None or keypoints is None:
        return []
    points = flow.as_numpy(keypoints.xy)
    if points is None or not len(points):
        return []
    if points.shape[1:] != (1, 2):
        raise ValueError(f"Expected one point per object, got {points.shape}")
    scores, classes = flow.as_numpy(boxes.conf), flow.as_numpy(boxes.cls)
    presence = flow.as_numpy(keypoints.conf)
    if presence is None:
        presence = np.ones(points.shape[:2])
    if len(scores) != len(points) or presence.shape != points.shape[:2]:
        raise ValueError("Inconsistent prediction lengths")
    detections = []
    for p, v, score, cls in zip(points, presence, scores, classes):
        name = result.names[int(cls)]
        if name not in CLASS_NAMES:
            raise ValueError(f"Unexpected class {name}")
        if np.isfinite(score):
            detections.append(
                {
                    "class_id": CLASS_NAMES.index(name),
                    "class_name": name,
                    "object_confidence": float(score),
                    "points": p,
                    "keypoint_presence": v,
                }
            )
    return detections


def draw_point(output, detection, points, valid, args):
    if not valid[0]:
        return
    cls = detection["class_id"]
    x, y = map(int, np.rint(points[0]))
    if cls < 3:
        color = flow.TYPE_OUTLINES_BGR[cls]
        if args.show_keypoints:
            cv2.circle(output, (x, y), 4, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(output, (x, y), 2, color, -1, cv2.LINE_AA)
        tag = ("C", "CE", "CEC")[cls]
        label_y = y
    else:
        color_id, kind = divmod(cls - 3, 3)
        color = flow.TYPE_OUTLINES_BGR[kind]
        if args.show_keypoints:
            cv2.circle(output, (x, y), 10, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.circle(output, (x, y), 10, color, 1, cv2.LINE_AA)
        if args.show_color_initials:
            letter = COLORS[color_id][0].upper()
            font = cv2.FONT_HERSHEY_SIMPLEX
            width, height = cv2.getTextSize(letter, font, 0.5, 2)[0]
            anchor = (x - width // 2, y + height // 2)
            cv2.putText(output, letter, anchor, font, 0.5, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(
                output, letter, anchor, font, 0.5, flow.COLORS_BGR[color_id], 2, cv2.LINE_AA
            )
        tag = COLORS[color_id][0].upper() + "/" + ("C", "E", "CTR")[kind]
        label_y = y + 24
    text = []
    if args.show_labels:
        text.append(tag)
    if args.show_confidences:
        text.append(f"{detection['object_confidence']:.2f}")
    if args.show_keypoint_presence:
        text.append(f"p={detection['keypoint_presence'][0]:.3f}")
    if text:
        flow.draw_label(output, " ".join(text), x, label_y, color)


def render_frame(frame, detector, model, tracker, args, frame_index=0):
    return flow.render_frame(
        frame,
        detector,
        model,
        tracker,
        args,
        frame_index,
        extractor=extract_points,
        drawer=draw_point,
    )


def build_parser():
    parser = flow.build_parser()
    parser.description = __doc__
    parser.set_defaults(
        model=ROOT / "runs/pose/runs/typed_sticker_keypoints/weights/best.pt",
        pose_conf=None,
        show_labels=True,
        pose_class_conf={},
    )
    for action in parser._actions:
        if action.dest == "pose_class_conf":
            action.type = parse_class_conf
            action.help = "Per-object thresholds: corner-edge=0.4,stickers=0.9,blue_corner=0.8"
        if action.dest == "pose_conf":
            action.help = "Override all object/class thresholds; omitted uses landmark .25/.40/.25 and sticker .90"
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    thresholds = (
        dict(DEFAULT_CLASS_CONF)
        if args.pose_conf is None
        else dict.fromkeys(range(21), args.pose_conf)
    )
    thresholds.update(args.pose_class_conf)
    args.pose_class_conf = thresholds
    args.pose_conf = min(thresholds.values())
    flow.validate_args(parser, args)
    from ultralytics import YOLO

    model = YOLO(str(flow.resolve_checkpoint(args.model)))
    if (
        model.task != "pose"
        or tuple(model.model.model[-1].kpt_shape) != (1, 3)
        or set(model.names.values()) != set(CLASS_NAMES)
    ):
        raise ValueError("Expected 21-class single-point landmark and typed-sticker checkpoint")
    flow.run(args, model, render_frame)


if __name__ == "__main__":
    main()
