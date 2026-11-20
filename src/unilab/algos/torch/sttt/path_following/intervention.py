"""Compose physical joint residuals without altering Planner base predictions."""

from __future__ import annotations

import torch

from unilab.algos.torch.sttt.path_following.contract import _validate_scales


def compose_joint_position_residual(
    nominal_future: torch.Tensor,
    residual_rad: torch.Tensor,
    *,
    ctrl_dt: float,
    position_scale: float,
    velocity_scale: float,
    observation_contract: str = "official_compact_64_v1",
) -> torch.Tensor:
    """Return an owned, differentiable Planner future in its declared layout.

    Inputs are [B,K,D] and [B,29], in source joint order. Positions are
    checkpoint-scaled relative angles; residuals are physical radians. For
    r[-1]=0, r[0]=residual, r[k>=1]=0, add s_q*r[k] to position and
    s_v*(r[k]-r[k-1])/ctrl_dt to velocity. Original Planner velocities are
    retained underneath this increment, so zero residual is an exact identity.
    Gyro and gravity are never derived from joint positions. No clamp is applied.
    """
    _validate_scales(ctrl_dt, position_scale, velocity_scale)
    if observation_contract == "official_compact_64_v1":
        width, position, velocity = 64, slice(3, 32), slice(32, 61)
    elif observation_contract == "g1_fada_state_v2":
        width, position, velocity = 66, slice(6, 35), slice(35, 64)
    else:
        raise ValueError("unsupported joint intervention observation contract")
    if (
        nominal_future.ndim != 3 or nominal_future.shape[0] < 1
        or nominal_future.shape[1] < 1 or nominal_future.shape[2] != width
        or not nominal_future.is_floating_point()
    ):
        raise ValueError("joint intervention future differs from its declared observation layout")
    if (
        residual_rad.shape != (nominal_future.shape[0], 29)
        or residual_rad.dtype != nominal_future.dtype
        or residual_rad.device != nominal_future.device
    ):
        raise ValueError("residual_rad must be [batch, 29] with the future's dtype and device")
    if not bool(torch.isfinite(nominal_future).all()) or not bool(torch.isfinite(residual_rad).all()):
        raise ValueError("joint intervention requires finite future and residual")
    adapted = nominal_future.clone()
    adapted[:, 0, position] += position_scale * residual_rad
    velocity_increment = (velocity_scale / ctrl_dt) * residual_rad
    adapted[:, 0, velocity] += velocity_increment
    if adapted.shape[1] > 1:
        adapted[:, 1, velocity] -= velocity_increment
    if not bool(torch.isfinite(adapted).all()):
        raise ValueError("joint intervention produced a non-finite future")
    return adapted
