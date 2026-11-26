"""Dense rewards over measured source-order motion and unmodified Planner references."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResidualRewardConfig:
    joint_position_weight: float = .75
    joint_velocity_weight: float = .5
    velocity_weight: float = 1.
    yaw_rate_weight: float = .5
    posture_weight: float = 1.
    path_weight: float = 1.
    action_rate_weight: float = .01
    termination_cost: float = 200.
    joint_position_tolerance_rad: float = .25
    joint_velocity_tolerance_rad_s: float = 2.
    velocity_tolerance_m_s: float = .5
    yaw_rate_tolerance_rad_s: float = 1.
    lateral_tolerance_m: float = .3
    tilt_tolerance_rad: float = .5

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0 or ("tolerance" in name and value == 0):
                raise ValueError(f"invalid residual reward {name}")


def measured_residual_reward(*, following, nominal, command, motion, errors, tilt, actions,
                             previous_actions, terminated, ctrl_dt, preprocessing,
                             observation_contract, config: ResidualRewardConfig):
    """Reward a real transition; joint references are the untouched Planner's next frame."""
    if observation_contract == "official_compact_64_v1":
        position, velocity = slice(3, 32), slice(32, 61)
    elif observation_contract == "g1_fada_state_v2":
        position, velocity = slice(6, 35), slice(35, 64)
    else:
        raise ValueError("unknown residual reward observation contract")
    scales = preprocessing["term_scale"]
    q_error = (following[:, position] - nominal[:, position]) / scales["dof_pos"]
    dq_error = (following[:, velocity] - nominal[:, velocity]) / scales["dof_vel"]
    rates = motion / ctrl_dt
    terms = {
        "joint_position": config.joint_position_weight * np.exp(-np.square(q_error / config.joint_position_tolerance_rad).mean(-1)),
        "joint_velocity": config.joint_velocity_weight * np.exp(-np.square(dq_error / config.joint_velocity_tolerance_rad_s).mean(-1)),
        "velocity": config.velocity_weight * np.exp(-np.square((rates[:, :2] - command[:, :2]) / config.velocity_tolerance_m_s).mean(-1)),
        "yaw_rate": config.yaw_rate_weight * np.exp(-np.square((rates[:, 2] - command[:, 2]) / config.yaw_rate_tolerance_rad_s)),
        "posture": config.posture_weight * np.exp(-np.square(tilt / config.tilt_tolerance_rad)),
        "path": config.path_weight * np.exp(-np.square(errors[:, 0] / config.lateral_tolerance_m)),
        "action_rate": -config.action_rate_weight * np.square(actions - previous_actions).mean(-1),
    }
    terms = {name: value * ctrl_dt for name, value in terms.items()}
    terms["termination"] = -config.termination_cost * terminated.astype(np.float32)
    return np.asarray(sum(terms.values()), dtype=np.float32), terms
