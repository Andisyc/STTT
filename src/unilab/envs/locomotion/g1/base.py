from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from unilab.envs.locomotion.common.base import (
    BaseNoiseConfig,
    ControlConfigBase,
    LocomotionBaseCfg,
    LocomotionBaseEnv,
)
from unilab.envs.locomotion.common.base import (
    Sensor as LocomotionSensor,
)


@dataclass
class NoiseConfig(BaseNoiseConfig):
    scale_joint_angle: float = 0.02
    scale_joint_vel: float = 0.3
    scale_gyro: float = 0.1


@dataclass
class ControlConfig(ControlConfigBase):
    action_scale: float | np.ndarray = 0.25  # type: ignore[assignment]


@dataclass
class Sensor(LocomotionSensor):
    local_linvel: str = "pelvis_local_linvel"
    gyro: str = "torso_gyro"
    upvector: str = "torso_upvector"


@dataclass
class Asset:
    base_name = "pelvis"
    foot_name = "ankle_roll_link"
    ground = "floor"


@dataclass
class G1BaseCfg(LocomotionBaseCfg):
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)  # type: ignore[assignment]
    control_config: ControlConfig = field(default_factory=ControlConfig)  # type: ignore[assignment]
    sensor: Sensor = field(default_factory=Sensor)
    asset: Asset = field(default_factory=Asset)
    sim_dt: float = 0.02 / 3.0
    ctrl_dt: float = 0.02


class G1BaseEnv(LocomotionBaseEnv):
    _cfg: G1BaseCfg
    _keyframe_name = "stand"
    _use_global_dtype = False

    def _obs_noise(self, data: np.ndarray, scale: float) -> np.ndarray:
        """Same as base, but coerces back to ``data.dtype`` (G1 runs in float32)."""
        return np.asarray(super()._obs_noise(data, scale), dtype=data.dtype)

    def get_base_pos(self) -> np.ndarray:
        """Return batched world-frame base positions through the task boundary."""
        return np.asarray(self._backend.get_base_pos())

    def get_base_quat(self) -> np.ndarray:
        """Return batched world-frame base quaternions in wxyz order."""
        return np.asarray(self._backend.get_base_quat())

    def get_base_lin_vel(self) -> np.ndarray:
        """Return batched world-frame base linear velocity."""
        return np.asarray(self._backend.get_base_lin_vel())

    def get_base_ang_vel(self) -> np.ndarray:
        """Return world-frame angular velocity for source-aligned policy inputs."""
        return np.asarray(self._backend.get_base_ang_vel())

    def get_actuator_gains(self) -> tuple[np.ndarray, np.ndarray]:
        """Expose source/target PD contract checks without backend traversal."""
        return self._backend.get_actuator_gains()

    def get_actuated_joint_names(self) -> tuple[str, ...]:
        """Return the target action ordering through the backend contract."""
        return self._backend.get_actuated_joint_names()

    def get_upvector(self) -> np.ndarray:
        """Return the configured torso upvector used by proprioception."""
        return np.asarray(self._backend.get_sensor_data(self._cfg.sensor.upvector))

    def get_foot_pos(self) -> np.ndarray:
        """Return left/right foot world positions with shape ``(N, 2, 3)``."""
        left = np.asarray(self._backend.get_sensor_data("left_foot_pos"))
        right = np.asarray(self._backend.get_sensor_data("right_foot_pos"))
        if left.ndim != 2 or left.shape[-1] != 3 or right.shape != left.shape:
            raise ValueError(
                "G1 foot position sensors must both have shape (num_envs, 3), "
                f"got left={left.shape} right={right.shape}"
            )
        return np.stack((left, right), axis=1)
