"""Three original landmarks plus eighteen one-point color/piece-type classes."""

import re

LANDMARK_NAMES = ("corner", "corner-edge", "corner-edge-center")
COLORS = ("white", "yellow", "orange", "red", "blue", "green")
PIECE_TYPES = ("corner", "edge", "center")
STICKER_NAMES = tuple(f"{color} {kind}" for color in COLORS for kind in PIECE_TYPES)
CLASS_NAMES = LANDMARK_NAMES + STICKER_NAMES
FORMAT = "typed_sticker_keypoints_v1"


def sticker_class_id(color, kind):
    return 3 + COLORS.index(color.lower()) * 3 + PIECE_TYPES.index(kind)


def sticker_owner(sticker):
    if sticker.get("source_cubie"):
        return sticker["source_cubie"]
    match = re.search(r"cubie(?:\.\d+)?", sticker["source_object"])
    return match.group() if match else sticker["source_object"].split(":")[0]


def manifest():
    return {
        "format": FORMAT,
        "class_names": list(CLASS_NAMES),
        "landmark_class_names": list(LANDMARK_NAMES),
        "sticker_class_names": list(STICKER_NAMES),
        "keypoint_shape": [1, 3],
        "keypoints_per_sticker": 1,
        "visibility_policy": "preserve existing visibility and angle/coverage selection",
        "sticker_location": "original sticker center",
    }
