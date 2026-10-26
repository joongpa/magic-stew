"""Small rigid-pose motion models for temporal pose filtering.

The filter in this module is intentionally independent of cube
correspondence.  It can smooth the pose of whichever part of the cube is
currently being treated as the rigid anchor, while correspondence changes
remain outside the motion model.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def _as_vector(value: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(3).copy()


def _clip_norm(value: np.ndarray, limit: float) -> np.ndarray:
    value = _as_vector(value)
    limit = float(limit)
    if not np.isfinite(limit) or limit <= 0.0:
        return np.zeros(3, dtype=np.float64) if limit == 0.0 else value
    norm = float(np.linalg.norm(value))
    if norm <= limit or norm == 0.0:
        return value
    return value * (limit / norm)


@dataclass(frozen=True)
class PoseMotionFilterConfig:
    """Configuration for the bounded alpha-beta-gamma pose filter.

    Translation units are the same as the PnP translation vector (millimetres
    in the current cube model).  Rotation units are radians.
    """

    dt_seconds: float = 1.0 / 30.0
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
    """A damped, bounded constant-acceleration pose filter.

    This is an alpha-beta-gamma filter with acceleration and jerk bounds.  A
    zero-velocity state naturally behaves like a stationary prior.  The
    damping terms prevent a short-lived detection error from creating a
    persistent velocity or acceleration estimate.
    """

    def __init__(self, config: PoseMotionFilterConfig | None = None) -> None:
        self.config = config or PoseMotionFilterConfig()
        self.reset()

    def reset(self) -> None:
        self._rotation: np.ndarray | None = None
        self._translation: np.ndarray | None = None
        self._velocity = np.zeros(3, dtype=np.float64)
        self._acceleration = np.zeros(3, dtype=np.float64)
        self._angular_velocity = np.zeros(3, dtype=np.float64)
        self._angular_acceleration = np.zeros(3, dtype=np.float64)

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
        dt = float(self.config.dt_seconds if dt_seconds is None else dt_seconds)
        if dt <= 0.0:
            raise ValueError("dt_seconds must be positive")
        predicted_translation, predicted_rotation = self._predict_state(dt)
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
        """Commit a pose measurement while returning that exact measurement.

        This method updates the hidden motion state used for future
        prediction, but it does not blend the current pose with the
        prediction.  That distinction matters for cube turns: a visually
        valid one-frame turn must be allowed to appear immediately.
        """

        measured_rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
        measured_translation = _as_vector(tvec)
        if not np.isfinite(measured_rotation).all() or not np.isfinite(measured_translation).all():
            raise ValueError("pose measurements must be finite")
        dt = float(self.config.dt_seconds if dt_seconds is None else dt_seconds)
        if dt <= 0.0:
            raise ValueError("dt_seconds must be positive")

        if not self.initialized:
            self._rotation = measured_rotation.copy()
            self._translation = measured_translation.copy()
            zero = PoseMotionDiagnostics(
                initialized=False,
                position_residual_norm=0.0,
                orientation_residual_rad=0.0,
                position_acceleration_norm=0.0,
                position_jerk_norm=0.0,
                angular_acceleration_norm=0.0,
                angular_jerk_norm=0.0,
            )
            return (
                np.asarray(rvec, dtype=np.float64).reshape(3, 1).copy(),
                measured_translation.reshape(3, 1).copy(),
                zero,
            )

        assert self._rotation is not None
        assert self._translation is not None
        predicted_translation, predicted_rotation = self._predict_state(dt)
        position_residual = measured_translation - predicted_translation
        relative_rotation = predicted_rotation.T @ measured_rotation
        orientation_residual = cv2.Rodrigues(relative_rotation)[0].reshape(3)

        # Keep the bounded acceleration state useful for future prediction,
        # while incorporating the accepted measurement into velocity. The
        # decay prevents one bad frame from becoming a full-size prediction
        # jump on the next frame.
        target_acceleration = self._bounded_position_acceleration(
            position_residual,
            dt,
        )
        position_jerk = target_acceleration - self._acceleration
        position_jerk = _clip_norm(
            position_jerk,
            float(self.config.max_jerk_units_s3) * dt,
        )
        acceleration = _clip_norm(
            self._acceleration + position_jerk,
            float(self.config.max_acceleration_units_s2),
        )
        observed_velocity = (measured_translation - self._translation) / dt
        velocity_decay = float(self.config.velocity_decay)
        velocity = (
            velocity_decay * self._velocity
            + (1.0 - velocity_decay) * observed_velocity
        )

        target_angular_acceleration = self._bounded_angular_acceleration(
            orientation_residual,
            dt,
        )
        angular_jerk = target_angular_acceleration - self._angular_acceleration
        angular_jerk = _clip_norm(
            angular_jerk,
            float(self.config.max_angular_jerk_rad_s3) * dt,
        )
        angular_acceleration = _clip_norm(
            self._angular_acceleration + angular_jerk,
            float(self.config.max_angular_acceleration_rad_s2),
        )
        observed_angular_velocity = cv2.Rodrigues(
            self._rotation.T @ measured_rotation
        )[0].reshape(3) / dt
        angular_velocity = (
            velocity_decay * self._angular_velocity
            + (1.0 - velocity_decay) * observed_angular_velocity
        )

        self._translation = measured_translation.copy()
        self._rotation = measured_rotation.copy()
        self._velocity = velocity
        self._acceleration = acceleration
        self._angular_velocity = angular_velocity
        self._angular_acceleration = angular_acceleration
        diagnostics = PoseMotionDiagnostics(
            initialized=True,
            position_residual_norm=float(np.linalg.norm(position_residual)),
            orientation_residual_rad=float(np.linalg.norm(orientation_residual)),
            position_acceleration_norm=float(np.linalg.norm(acceleration)),
            position_jerk_norm=float(np.linalg.norm(position_jerk)) / dt,
            angular_acceleration_norm=float(np.linalg.norm(angular_acceleration)),
            angular_jerk_norm=float(np.linalg.norm(angular_jerk)) / dt,
        )
        return (
            np.asarray(rvec, dtype=np.float64).reshape(3, 1).copy(),
            measured_translation.reshape(3, 1).copy(),
            diagnostics,
        )

    def _predict_state(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        assert self._rotation is not None
        assert self._translation is not None
        predicted_translation = (
            self._translation
            + self._velocity * dt
            + 0.5 * self._acceleration * dt * dt
        )
        rotation_delta = (
            self._angular_velocity * dt
            + 0.5 * self._angular_acceleration * dt * dt
        ).reshape(3, 1)
        predicted_rotation = self._rotation @ cv2.Rodrigues(rotation_delta)[0]
        return predicted_translation, predicted_rotation

    def _bounded_position_acceleration(
        self,
        residual: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        predicted_acceleration = self._acceleration * float(self.config.acceleration_decay)
        return predicted_acceleration + (
            float(self.config.position_gamma) * residual / (dt * dt)
        )

    def _bounded_angular_acceleration(
        self,
        residual: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        predicted_acceleration = self._angular_acceleration * float(self.config.acceleration_decay)
        return predicted_acceleration + (
            float(self.config.orientation_gamma) * residual / (dt * dt)
        )

    def update(
        self,
        rvec: np.ndarray,
        tvec: np.ndarray,
        *,
        dt_seconds: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, PoseMotionDiagnostics]:
        """Filter one object-to-camera pose and return ``(rvec, tvec, stats)``."""

        measured_rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
        measured_translation = _as_vector(tvec)
        if not np.isfinite(measured_rotation).all() or not np.isfinite(measured_translation).all():
            raise ValueError("pose measurements must be finite")

        dt = float(self.config.dt_seconds if dt_seconds is None else dt_seconds)
        if dt <= 0.0:
            raise ValueError("dt_seconds must be positive")

        if not self.initialized:
            self._rotation = measured_rotation.copy()
            self._translation = measured_translation.copy()
            zero = PoseMotionDiagnostics(
                initialized=False,
                position_residual_norm=0.0,
                orientation_residual_rad=0.0,
                position_acceleration_norm=0.0,
                position_jerk_norm=0.0,
                angular_acceleration_norm=0.0,
                angular_jerk_norm=0.0,
            )
            return (
                cv2.Rodrigues(self._rotation)[0].reshape(3, 1),
                self._translation.reshape(3, 1).copy(),
                zero,
            )

        assert self._rotation is not None
        assert self._translation is not None

        velocity_decay = float(self.config.velocity_decay)
        acceleration_decay = float(self.config.acceleration_decay)
        predicted_translation = (
            self._translation
            + self._velocity * dt
            + 0.5 * self._acceleration * dt * dt
        )
        predicted_velocity = (
            self._velocity * velocity_decay
            + self._acceleration * dt
        )
        predicted_acceleration = self._acceleration * acceleration_decay
        position_residual = measured_translation - predicted_translation

        target_acceleration = predicted_acceleration + (
            float(self.config.position_gamma) * position_residual / (dt * dt)
        )
        position_jerk = target_acceleration - self._acceleration
        position_jerk = _clip_norm(
            position_jerk,
            float(self.config.max_jerk_units_s3) * dt,
        )
        acceleration = _clip_norm(
            self._acceleration + position_jerk,
            float(self.config.max_acceleration_units_s2),
        )
        translation = predicted_translation + float(self.config.position_alpha) * position_residual
        velocity = predicted_velocity + (
            float(self.config.position_beta) * position_residual / dt
        )

        predicted_rotation = self._rotation @ cv2.Rodrigues(
            (self._angular_velocity * dt + 0.5 * self._angular_acceleration * dt * dt).reshape(3, 1)
        )[0]
        predicted_angular_velocity = (
            self._angular_velocity * velocity_decay
            + self._angular_acceleration * dt
        )
        predicted_angular_acceleration = self._angular_acceleration * acceleration_decay
        relative_rotation = predicted_rotation.T @ measured_rotation
        orientation_residual = cv2.Rodrigues(relative_rotation)[0].reshape(3)
        target_angular_acceleration = predicted_angular_acceleration + (
            float(self.config.orientation_gamma) * orientation_residual / (dt * dt)
        )
        angular_jerk = target_angular_acceleration - self._angular_acceleration
        angular_jerk = _clip_norm(
            angular_jerk,
            float(self.config.max_angular_jerk_rad_s3) * dt,
        )
        angular_acceleration = _clip_norm(
            self._angular_acceleration + angular_jerk,
            float(self.config.max_angular_acceleration_rad_s2),
        )
        rotation_correction = cv2.Rodrigues(
            (float(self.config.orientation_alpha) * orientation_residual).reshape(3, 1)
        )[0]
        rotation = predicted_rotation @ rotation_correction
        angular_velocity = predicted_angular_velocity + (
            float(self.config.orientation_beta) * orientation_residual / dt
        )

        position_jerk_norm = float(np.linalg.norm(position_jerk)) / dt
        angular_jerk_norm = float(np.linalg.norm(angular_jerk)) / dt
        self._translation = translation
        self._velocity = velocity
        self._acceleration = acceleration
        self._rotation = rotation
        self._angular_velocity = angular_velocity
        self._angular_acceleration = angular_acceleration
        diagnostics = PoseMotionDiagnostics(
            initialized=True,
            position_residual_norm=float(np.linalg.norm(position_residual)),
            orientation_residual_rad=float(np.linalg.norm(orientation_residual)),
            position_acceleration_norm=float(np.linalg.norm(acceleration)),
            position_jerk_norm=position_jerk_norm,
            angular_acceleration_norm=float(np.linalg.norm(angular_acceleration)),
            angular_jerk_norm=angular_jerk_norm,
        )
        return (
            cv2.Rodrigues(rotation)[0].reshape(3, 1),
            translation.reshape(3, 1).copy(),
            diagnostics,
        )
