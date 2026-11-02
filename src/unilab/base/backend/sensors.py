"""Backend-neutral stereo/IMU packets; no estimator or simulator imports."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StereoSensorConfig:
    """Opt-in single-environment camera rig, attached rigidly to one body.

    Position and camera axes use the body frame. Camera +X points right,
    +Y points up, and -Z is the viewing direction (MuJoCo convention).
    Sampling periods are integer physics steps, never wall-clock periods.
    """

    body_name: str = "torso_link"
    pelvis_body: str = "pelvis"
    position: tuple[float, float, float] = (0.08, 0.0, 0.30)
    xyaxes: tuple[float, float, float, float, float, float] = (0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
    baseline_m: float = 0.06
    width: int = 320
    height: int = 240
    fovy_deg: float = 70.0
    camera_period_steps: int = 5
    gyro_sensor: str = "torso_gyro"
    accelerometer_sensor: str = "torso_acceleration"
    left_camera: str = "sttt_left"
    right_camera: str = "sttt_right"

    def __post_init__(self) -> None:
        for name in ("width", "height", "camera_period_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.baseline_m) or self.baseline_m <= 0:
            raise ValueError("baseline_m must be positive and finite")
        if not math.isfinite(self.fovy_deg) or not 0 < self.fovy_deg < 180:
            raise ValueError("fovy_deg must be between 0 and 180")
        if len(self.position) != 3 or not np.isfinite(self.position).all():
            raise ValueError("position must contain three finite coordinates")
        axes = np.asarray(self.xyaxes, dtype=np.float64)
        if axes.shape != (6,) or not np.isfinite(axes).all():
            raise ValueError("xyaxes must contain six finite coordinates")
        if np.linalg.norm(axes[:3]) < 1e-12 or np.linalg.norm(np.cross(axes[:3], axes[3:])) < 1e-12:
            raise ValueError("camera axes must be nonzero and nonparallel")
        if not all((self.body_name, self.gyro_sensor, self.accelerometer_sensor,
                    self.pelvis_body,
                    self.left_camera, self.right_camera)):
            raise ValueError("body, sensor and camera names must be nonempty")
        if self.left_camera == self.right_camera:
            raise ValueError("left and right camera names must differ")


@dataclass(frozen=True)
class StereoCalibration:
    """Transforms map the named sensor's coordinates into the left optical rig.

    The optical convention is X-right, Y-down, Z-forward. Matrices are 4x4
    SE(3); focal/principal are pixel coordinates. Images are ideal rectified
    pinhole images. Producers own these arrays for the stream lifetime.
    """

    width: int
    height: int
    focal: tuple[float, float]
    principal: tuple[float, float]
    rig_from_left: np.ndarray
    rig_from_right: np.ndarray
    rig_from_imu: np.ndarray
    imu_frequency_hz: float


@dataclass(frozen=True)
class ImuSample:
    timestamp_ns: int
    acceleration: np.ndarray  # specific force, m/s^2, local IMU axes
    angular_velocity: np.ndarray  # rad/s, local IMU axes


@dataclass(frozen=True)
class StereoFrame:
    """Owned RGB uint8 images; IMU covers the interval since the previous frame.

    Both images share one physics snapshot. sequence_id changes at reset;
    timestamps need only increase inside a sequence. No ground-truth pose is
    available in this estimator input.
    """

    sequence_id: int
    timestamp_ns: int
    left_rgb: np.ndarray
    right_rgb: np.ndarray
    imu: tuple[ImuSample, ...]


@dataclass(frozen=True)
class SensorClock:
    sequence_id: int
    timestamp_ns: int


@dataclass(frozen=True)
class RigKinematicsSample:
    """Image-time relative FK, not a world-position measurement.

    ``pelvis_from_rig`` maps left optical coordinates into pelvis coordinates.
    ``up_in_pelvis`` is the ideal simulated gravity/up direction in pelvis
    axes. It supplies tilt only for one-time Z-up initialization, not heading
    or position, and is never sent to the visual tracker.
    """

    sequence_id: int
    timestamp_ns: int
    pelvis_from_rig: np.ndarray
    up_in_pelvis: np.ndarray


@dataclass(frozen=True)
class RigReferencePose:
    """Evaluation-only ground truth, deliberately separate from StereoFrame."""

    sequence_id: int
    timestamp_ns: int
    world_from_rig: np.ndarray


class BackendSensorStream(ABC):
    """Backend-owned resources sampled by normal stepping; drains never step."""

    @property
    @abstractmethod
    def calibration(self) -> StereoCalibration:
        """Return the fixed rig calibration."""

    @property
    @abstractmethod
    def clock(self) -> SensorClock:
        """Return the current sequence and physical time, including before the first frame."""

    @abstractmethod
    def drain_kinematics(self) -> tuple[RigKinematicsSample, ...]:
        """Consume relative FK snapshots paired with captured images."""

    @abstractmethod
    def drain(self) -> tuple[StereoFrame, ...]:
        """Consume frames produced since the last drain; do not repeat frames."""

    @abstractmethod
    def drain_reference_poses(self) -> tuple[RigReferencePose, ...]:
        """Consume evaluation-only references matching produced frame times."""

    @abstractmethod
    def close(self) -> None:
        """Release resources; safe to call more than once."""
