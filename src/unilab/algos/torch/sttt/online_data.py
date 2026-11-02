"""Detached evidence and configuration for local spatial-feedback adaptation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral

import torch

from unilab.algos.torch.distill.fada.online_data import FADAInferenceSnapshot
from unilab.base.backend.sensors import StereoSensorConfig


@dataclass(frozen=True)
class OdometrySample:
    """A base pose in one metric frame and localization epoch (seconds, wxyz)."""

    timestamp_s: float
    position: torch.Tensor
    quaternion: torch.Tensor
    frame_id: str = "world"
    tracking_epoch: int = 0
    valid: bool = True

    def __post_init__(self) -> None:
        for name in ("position", "quaternion"):
            object.__setattr__(self, name, getattr(self, name).detach().clone())
        if self.position.shape != (3,) or self.quaternion.shape != (4,):
            raise ValueError("Odometry requires position (3,) and wxyz quaternion (4,)")
        if self.valid and (
            not math.isfinite(self.timestamp_s)
            or not self.frame_id
            or not torch.isfinite(self.position).all()
            or not torch.isfinite(self.quaternion).all()
            or not torch.isclose(self.quaternion.norm(), self.quaternion.new_tensor(1.0), atol=1e-4)
        ):
            raise ValueError("Valid odometry must contain a finite timestamp and unit quaternion")


@dataclass(frozen=True)
class SpatialOnlineSample:
    """A mature, same-episode window; every policy input is an actual saved query.

    Endpoints are relative to the measured heading at the window origin.
    ``actual_actions`` are target-order env_input actions, not PD targets.
    """

    proprio_history: torch.Tensor
    snapshots: tuple[FADAInferenceSnapshot, ...]
    actual_actions: torch.Tensor
    measured_endpoint: torch.Tensor
    reference_endpoint: torch.Tensor
    episode_id: int
    origin_step: int

    def __post_init__(self) -> None:
        for name in ("proprio_history", "actual_actions", "measured_endpoint", "reference_endpoint"):
            value = getattr(self, name)
            if not torch.isfinite(value).all():
                raise ValueError(f"Spatial sample {name} must be finite")
            object.__setattr__(self, name, value.detach().clone())
        if self.proprio_history.ndim != 3 or self.proprio_history.shape[0] != 1 or self.proprio_history.shape[-1] != 64:
            raise ValueError("Spatial history must have shape (1, history_length, 64)")
        if self.actual_actions.shape != (1, len(self.snapshots), 29) or not self.snapshots:
            raise ValueError("Spatial actions must match H saved policy queries with shape (1, H, 29)")
        if self.measured_endpoint.shape != (1, 3) or self.reference_endpoint.shape != (1, 3):
            raise ValueError("Spatial endpoints must have shape (1, 3)")
        if self.episode_id < 0 or self.origin_step < 0:
            raise ValueError("Spatial sample identity must be non-negative")


@dataclass(frozen=True)
class SpatialAdaptationConfig:
    enabled: bool = False
    world_model_checkpoint: str | None = None
    space_weight: float = 1.0
    proprio_weight: float = 1.0
    coordinate_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    max_pose_age_s: float = 0.1
    odometry_source: str = "simulation"
    sensor: StereoSensorConfig = field(default_factory=StereoSensorConfig)
    visual_mode: str = "stereo"
    imu_noise: dict[str, float] | None = None
    max_pose_gap_s: float = 0.1
    max_alignment_age_s: float = 0.2
    max_pose_samples: int = 512
    max_pending_intervals: int = 512
    adapter_checkpoint_path: str | None = None
    save_adapter_path: str | None = None

    def __post_init__(self) -> None:
        if self.enabled and not self.world_model_checkpoint:
            raise ValueError("Spatial adaptation requires world_model_checkpoint")
        for name in ("space_weight", "proprio_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"spatial_adaptation.{name} must be positive and finite")
        if len(self.coordinate_weights) != 3 or any(
            not math.isfinite(value) or value < 0 for value in self.coordinate_weights
        ) or sum(self.coordinate_weights) <= 0:
            raise ValueError("Spatial coordinate weights require three non-negative finite values with positive sum")
        if not math.isfinite(self.max_pose_age_s) or self.max_pose_age_s <= 0:
            raise ValueError("max_pose_age_s must be positive and finite")
        if self.odometry_source not in {"simulation", "cuvslam"}:
            raise ValueError("odometry_source must be simulation or cuvslam")
        if isinstance(self.sensor, Mapping):
            object.__setattr__(self, "sensor", StereoSensorConfig(**self.sensor))
        if not isinstance(self.sensor, StereoSensorConfig):
            raise TypeError("sensor must be a StereoSensorConfig or mapping")
        if self.visual_mode not in {"stereo", "inertial"}:
            raise ValueError("visual_mode must be stereo or inertial")
        for name in ("max_pose_gap_s", "max_alignment_age_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"spatial_adaptation.{name} must be positive and finite")
        for name, minimum in (("max_pose_samples", 2), ("max_pending_intervals", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"spatial_adaptation.{name} must be an integer >= {minimum}")
        if self.odometry_source == "cuvslam" and self.visual_mode == "inertial":
            keys = {"gyroscope_noise_density", "gyroscope_random_walk",
                    "accelerometer_noise_density", "accelerometer_random_walk"}
            if self.imu_noise is None or set(self.imu_noise) != keys:
                raise ValueError("inertial cuvslam requires all four explicit imu_noise parameters")
            if any(not math.isfinite(float(v)) or float(v) <= 0 for v in self.imu_noise.values()):
                raise ValueError("imu_noise parameters must be positive and finite")
