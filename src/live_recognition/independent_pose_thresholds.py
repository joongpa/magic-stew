"""Shared per-class confidence thresholds for the independent pose model."""

from __future__ import annotations

from collections.abc import Mapping

INDEPENDENT_POSE_CLASS_NAMES = ("corner", "corner-edge", "corner-edge-center")

# Chosen from the 100% learning-curve threshold sweep. The edge class needs a
# stricter threshold because it contributes most of the background false
# positives.
DEFAULT_INDEPENDENT_POSE_CLASS_CONFIDENCES = {
    "corner": 0.25,
    "corner-edge": 0.4,
    "corner-edge-center": 0.25,
}


def parse_independent_pose_class_confidences(value: str) -> dict[str, float]:
    """Parse ``class=value`` pairs, filling unspecified classes from defaults."""

    thresholds = dict(DEFAULT_INDEPENDENT_POSE_CLASS_CONFIDENCES)
    assignments = [part.strip() for part in value.split(",") if part.strip()]
    if not assignments:
        raise ValueError("at least one class threshold is required")
    for assignment in assignments:
        if "=" not in assignment:
            raise ValueError(f"expected class=threshold, got {assignment!r}")
        class_name, raw_threshold = (part.strip() for part in assignment.split("=", 1))
        if class_name not in INDEPENDENT_POSE_CLASS_NAMES:
            valid = ", ".join(INDEPENDENT_POSE_CLASS_NAMES)
            raise ValueError(f"unknown pose class {class_name!r}; expected one of: {valid}")
        try:
            threshold = float(raw_threshold)
        except ValueError as exc:
            raise ValueError(f"invalid threshold for {class_name!r}: {raw_threshold!r}") from exc
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold for {class_name!r} must be in [0, 1]")
        thresholds[class_name] = threshold
    return thresholds


def independent_pose_class_threshold(
    class_name: str,
    thresholds: Mapping[str, float],
    fallback: float = 1.0,
) -> float:
    """Return a class threshold, rejecting unknown classes by default."""

    return float(thresholds.get(class_name, fallback))
