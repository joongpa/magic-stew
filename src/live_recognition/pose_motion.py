"""Small rigid-pose helpers for temporal displacement gating.

The filter in this module is intentionally independent of cube
correspondence. It tracks the last pose of whichever part of the cube is
currently being treated as the rigid anchor, while correspondence changes
remain outside the motion model.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def _as_vector(value: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(3).copy()


@dataclass(frozen=True)
class PoseMotionFilterConfig:
    """Configuration for the core-displacement temporal gate.

    Translation units are the same as the PnP translation vector (millimetres
    in the current cube model). The legacy acceleration fields remain for the
    unused pose-graph scorer; the live filter uses only displacement.
    """

    dt_seconds: float = 1.0 / 30.0
    max_displacement_mm: float = 40.0
    position_alpha: float = 0.35
    position_beta: float = 0.08
    position_gamma: float = 0.02
    orientation_alpha: float = 0.35
    orientation_beta: float = 0.08
    orientation_gamma: float = 0.02
    # Translation is expressed in millimetres, so 20 m/s^2 is 20,000 here.
    max_acceleration_units_s2: float = 20000.0
    max_jerk_units_s3: float = float("inf")
    max_angular_acceleration_rad_s2: float = 12.0
    max_angular_jerk_rad_s3: float = float("inf")
    velocity_decay: float = 0.90
    acceleration_decay: float = 0.75

    def __post_init__(self) -> None:
        if float(self.dt_seconds) <= 0.0:
            raise ValueError("dt_seconds must be positive")
        if float(self.max_displacement_mm) < 0.0:
            raise ValueError("max_displacement_mm must be non-negative")
        for name in (
            "position_alpha",
            "position_beta",
            "position_gamma",
            "orientation_alpha",
            "orientation_beta",
            "orientation_gamma",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in (
            "max_acceleration_units_s2",
            "max_jerk_units_s3",
            "max_angular_acceleration_rad_s2",
            "max_angular_jerk_rad_s3",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 <= float(self.velocity_decay) <= 1.0:
            raise ValueError("velocity_decay must be in [0, 1]")
        if not 0.0 <= float(self.acceleration_decay) <= 1.0:
            raise ValueError("acceleration_decay must be in [0, 1]")


@dataclass(frozen=True)
class PoseMotionDiagnostics:
    initialized: bool
    position_residual_norm: float
    orientation_residual_rad: float
    position_acceleration_norm: float
    position_jerk_norm: float
    angular_acceleration_norm: float
    angular_jerk_norm: float


class PoseMotionFilter:
    """Track the last accepted pose for a core-displacement gate.

    The current measurement is never smoothed or rotated toward a prediction.
    The caller compares its translation to ``predict_pose()`` using the
    configured displacement tolerance.
    """

    def __init__(self, config: PoseMotionFilterConfig | None = None) -> None:
        self.config = config or PoseMotionFilterConfig()
        self.reset()

    def reset(self) -> None:
        self._rotation: np.ndarray | None = None
        self._translation: np.ndarray | None = None

    @property
    def initialized(self) -> bool:
        return self._rotation is not None and self._translation is not None

    def current_pose(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return the last filtered pose without changing filter state."""

        if not self.initialized:
            return None
        assert self._rotation is not None
        assert self._translation is not None
        return (
            cv2.Rodrigues(self._rotation)[0].reshape(3, 1),
            self._translation.reshape(3, 1).copy(),
        )

    def predict_pose(
        self,
        *,
        dt_seconds: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Predict the next pose without changing filter state.

        Prediction is deliberately separate from measurement acceptance.  A
        good solvePnP hypothesis can therefore be displayed at its measured
        pose while this state remains available for ranking the next frame's
        hypotheses.
        """

        if not self.initialized:
            return None
        # Keep the temporal prior stationary between accepted measurements.
        # The displacement gate, rather than a velocity/acceleration model,
        # decides whether the next core position is plausible.
        predicted_translation = self._translation
        predicted_rotation = self._rotation
        assert predicted_translation is not None
        assert predicted_rotation is not None
        return (
            cv2.Rodrigues(predicted_rotation)[0].reshape(3, 1),
            predicted_translation.reshape(3, 1).copy(),
        )

    def accept_measurement(
        self,
        rvec: np.ndarray,
        tvec: np.ndarray,
        *,
        dt_seconds: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, PoseMotionDiagnostics]:
        """Store a pose and report only its core translation displacement."""

        measured_rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        measured_translation = _as_vector(tvec)
        measured_rotation, _ = cv2.Rodrigues(measured_rvec)
        if not np.isfinite(measured_rotation).all() or not np.isfinite(measured_translation).all():
            raise ValueError("pose measurements must be finite")
        if dt_seconds is not None and float(dt_seconds) <= 0.0:
            raise ValueError("dt_seconds must be positive")

        initialized = self.initialized
        displacement = (
            float(np.linalg.norm(measured_translation - self._translation))
            if initialized and self._translation is not None
            else 0.0
        )
        self._rotation = measured_rotation.copy()
        self._translation = measured_translation.copy()
        diagnostics = PoseMotionDiagnostics(
            initialized=initialized,
            position_residual_norm=displacement,
            orientation_residual_rad=0.0,
            position_acceleration_norm=0.0,
            position_jerk_norm=0.0,
            angular_acceleration_norm=0.0,
            angular_jerk_norm=0.0,
        )
        return measured_rvec.copy(), measured_translation.reshape(3, 1).copy(), diagnostics

    def update(
        self,
        rvec: np.ndarray,
        tvec: np.ndarray,
        *,
        dt_seconds: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, PoseMotionDiagnostics]:
        """Compatibility alias for :meth:`accept_measurement`."""

        return self.accept_measurement(rvec, tvec, dt_seconds=dt_seconds)
