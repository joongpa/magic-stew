from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
else:
    NDArray = Any

Face = Literal["U", "D", "F", "B", "R", "L"]
Axis = Literal["+x", "-x", "+y", "-y", "+z", "-z"]


@dataclass(frozen=True)
class CubeCandidate:
    """A detected cube candidate in image coordinates."""

    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    contour: NDArray[np.float32] | None = None
    keypoints_xy: NDArray[np.float32] | None = None
    mask: NDArray[np.uint8] | None = None


@dataclass(frozen=True)
class OrientationHypothesis:
    """One possible cube orientation with probability mass."""

    face_to_axis: dict[Face, Axis]
    probability: float


@dataclass(frozen=True)
class FrameResult:
    """Full probabilistic output for one frame."""

    candidate: CubeCandidate | None
    orientations: list[OrientationHypothesis]
