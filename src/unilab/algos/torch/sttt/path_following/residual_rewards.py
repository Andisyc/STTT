"""Task tracking and measured-body costs for window-held physical residuals."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResidualRewardConfig:
    joint_limit_weight: float = 1.
    joint_acceleration_weight: float = 7.25e-6  # 29-joint mean, equivalent to 2.5e-7 * sum(ddq**2).
    velocity_weight: float = 1.
    yaw_rate_weight: float = .5
    posture_weight: float = 1.
    path_weight: float = 1.
    residual_rate_weight: float = .01
    termination_cost: float = 200.
    joint_limit_fraction: float = .9
    joint_limit_tolerance_rad: float = .1
    residual_rate_tolerance_rad_s: float = .025
    velocity_tolerance_m_s: float = .5
    yaw_rate_tolerance_rad_s: float = 1.
    lateral_tolerance_m: float = .3
    tilt_tolerance_rad: float = .5

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0 or ("tolerance" in name and value == 0):
                raise ValueError(f"invalid residual reward {name}")
        if not 0 < self.joint_limit_fraction <= 1:
            raise ValueError("joint_limit_fraction must be in (0,1]")


def measured_residual_reward(*, joint_positions, joint_velocities, previous_joint_velocities,
                             joint_limits, residual, previous_residual, residual_dt,
                             command, motion, errors, tilt, terminated, ctrl_dt,
                             config: ResidualRewardConfig):
    """Use physical radians in target order; residuals retain source joint order."""
    excess = np.maximum(joint_limits[:, 0] - joint_positions, 0.)
    excess += np.maximum(joint_positions - joint_limits[:, 1], 0.)
    acceleration = (joint_velocities - previous_joint_velocities) / ctrl_dt
    residual_rate = (residual - previous_residual) / residual_dt
    rates = motion / ctrl_dt
    terms = {
        "joint_limit": -config.joint_limit_weight * np.square(excess / config.joint_limit_tolerance_rad).mean(-1),
        "joint_acceleration": -config.joint_acceleration_weight * np.square(acceleration).mean(-1),
        "velocity": config.velocity_weight * np.exp(-np.square((rates[:, :2] - command[:, :2]) / config.velocity_tolerance_m_s).mean(-1)),
        "yaw_rate": config.yaw_rate_weight * np.exp(-np.square((rates[:, 2] - command[:, 2]) / config.yaw_rate_tolerance_rad_s)),
        "posture": config.posture_weight * np.exp(-np.square(tilt / config.tilt_tolerance_rad)),
        "path": config.path_weight * np.exp(-np.square(errors[:, 0] / config.lateral_tolerance_m)),
        "residual_rate": -config.residual_rate_weight * np.square(residual_rate / config.residual_rate_tolerance_rad_s).mean(-1),
    }
    terms = {name: value * ctrl_dt for name, value in terms.items()}
    terms["termination"] = -config.termination_cost * terminated.astype(np.float32)
    diagnostics = {
        "joint_limit_excess_rad": excess.max(-1),
        "joint_acceleration_rms_rad_s2": np.sqrt(np.square(acceleration).mean(-1)),
        "residual_rate_rms_rad_s": np.sqrt(np.square(residual_rate).mean(-1)),
        "lateral_abs_m": np.abs(errors[:, 0]),
        "forward_velocity_m_s": rates[:, 0],
        "yaw_rate_error_rad_s": rates[:, 2] - command[:, 2],
    }
    return np.asarray(sum(terms.values()), dtype=np.float32), terms, diagnostics
