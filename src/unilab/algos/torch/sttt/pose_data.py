"""Shared packets for measured pelvis poses and historical action intervals."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from unilab.base.backend.sensors import SensorClock, StereoSensorConfig


@dataclass(frozen=True)
class PoseRuntimeConfig:
    """Measurement-source and alignment settings, independent of adaptation."""

    max_pose_age_s: float = 0.1
    odometry_source: str = "simulation"
    sensor: StereoSensorConfig = field(default_factory=StereoSensorConfig)
    visual_mode: str = "stereo"
    imu_noise: dict[str, float] | None = None
    max_pose_gap_s: float = 0.1
    max_alignment_age_s: float = 0.2
    max_pose_samples: int = 512
    max_pending_intervals: int = 512

    def __post_init__(self) -> None:
        for name in ("max_pose_age_s", "max_pose_gap_s", "max_alignment_age_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.odometry_source not in {"simulation", "cuvslam"}:
            raise ValueError("odometry_source must be simulation or cuvslam")
        if isinstance(self.sensor, Mapping):
            object.__setattr__(self, "sensor", StereoSensorConfig(**self.sensor))
        if not isinstance(self.sensor, StereoSensorConfig):
            raise TypeError("sensor must be a StereoSensorConfig or mapping")
        if self.visual_mode not in {"stereo", "inertial"}:
            raise ValueError("visual_mode must be stereo or inertial")
        for name, minimum in (("max_pose_samples", 2), ("max_pending_intervals", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.odometry_source == "cuvslam" and self.visual_mode == "inertial":
            keys = {
                "gyroscope_noise_density", "gyroscope_random_walk",
                "accelerometer_noise_density", "accelerometer_random_walk",
            }
            if self.imu_noise is None or set(self.imu_noise) != keys:
                raise ValueError("inertial cuvslam requires all four explicit imu_noise parameters")
            if any(not math.isfinite(float(value)) or float(value) <= 0 for value in self.imu_noise.values()):
                raise ValueError("imu_noise parameters must be positive and finite")


@dataclass(frozen=True)
class TimedPelvisPose:
    """A metric Z-up pelvis pose, with interpolation provenance preserved.

    navigation_from_pelvis maps pelvis into a local Z-up frame whose origin
    and heading are fixed once per localization epoch. Invalid samples carry
    None. source_timestamps_ns is (t,t) for a measurement and (lo,hi) for an
    interpolated query; timestamp_ns is always the represented physical time.
    """

    sequence_id: int
    tracking_epoch: int
    timestamp_ns: int
    navigation_from_pelvis: np.ndarray | None
    valid: bool = True
    source_timestamps_ns: tuple[int, int] | None = None
    frame_id: str = "local_odom_z_up"


@dataclass(frozen=True)
class ControlInterval:
    """Executed env-input action and command held over [start_ns,end_ns]."""

    sequence_id: int
    transition_id: int
    start_ns: int
    end_ns: int
    action: np.ndarray  # (29,), actual env_input, not PD targets or proposed actions
    command: np.ndarray  # (3,), vx/vy/yaw_rate, recorded before executing the action


@dataclass(frozen=True)
class AlignedPoseInterval:
    control: ControlInterval
    start_pose: TimedPelvisPose
    end_pose: TimedPelvisPose


@dataclass(frozen=True)
class PoseRuntimeUpdate:
    aligned: tuple[AlignedPoseInterval, ...]
    status: Literal["waiting", "tracking", "lost"]
    sequence_id: int
    tracking_epoch: int
    dropped_intervals: int = 0
    invalidated: bool = False


class OnlinePoseRuntime(Protocol):
    """One controller-owned source; never starts a second simulation loop."""

    @property
    def clock(self) -> SensorClock:
        """Read physical time after stepping without opening a new interval."""
        ...

    def begin_step(self) -> SensorClock:
        """Prepare the source after environment reset and return physical time."""
        ...

    def complete_step(self, control: ControlInterval) -> PoseRuntimeUpdate:
        """Consume already-produced measurements and resolve historical intervals."""
        ...

    def reset(self) -> None:
        """Drop local evidence; environment reset may happen after this call."""
        ...

    def close(self) -> None:
        """Release owned tracker and stream; idempotent."""
        ...
