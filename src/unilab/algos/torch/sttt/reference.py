"""World-anchored SE(2) reference trajectory for spatial online adaptation.

The reference integrates the commanded constant body twist with the exact
SE(2) exponential per control step. It is anchored to the measured pose only
at episode reset (or re-localization) and is never corrected by the actual
pose per step, so accumulated heading drift stays visible in the endpoint
comparison. Every pose returned is detached; nothing here is learned.
"""

from __future__ import annotations

import math

import torch

from unilab.algos.torch.sttt.geometry import wrap_to_pi, yaw_from_quat
from unilab.algos.torch.sttt.online_data import OdometrySample

__all__ = ["ReferenceTrajectory"]

_EPS = 1e-12


class ReferenceTrajectory:
    """Desired world-frame SE(2) path integrated from velocity commands."""

    def __init__(self, ctrl_dt: float) -> None:
        if not math.isfinite(ctrl_dt) or ctrl_dt <= 0:
            raise ValueError("ctrl_dt must be positive and finite")
        self._ctrl_dt = float(ctrl_dt)
        # World-frame desired pose (x, y, yaw); None until reset.
        self._pose: torch.Tensor | None = None

    @torch.no_grad()
    def reset(self, pose: OdometrySample) -> None:
        """Anchor the reference world x/y/yaw at an episode start or relocalization."""
        if not pose.valid:
            raise ValueError("cannot anchor the reference to an invalid odometry sample")
        position = pose.position.detach().to(dtype=torch.float64).cpu()
        yaw = yaw_from_quat(pose.quaternion.detach().reshape(1, 4).to(dtype=torch.float64).cpu())[0]
        self._pose = torch.stack((position[0], position[1], yaw)).clone()

    @torch.no_grad()
    def unanchor(self) -> None:
        """Drop the anchor; the next valid pose must re-anchor the reference."""
        self._pose = None

    @property
    def ctrl_dt(self) -> float:
        return self._ctrl_dt

    @torch.no_grad()
    def advance(self, command: torch.Tensor) -> torch.Tensor:
        """Integrate one control step of the commanded body twist.

        ``command`` is ``[vx, vy, yaw_rate]`` in the desired body heading. The
        update is the exact constant-twist SE(2) exponential and reduces to
        pure rotation (no translation) as the yaw rate approaches zero.
        Returns the desired world SE(2) pose ``[x, y, yaw]`` (detached).
        """
        if self._pose is None:
            raise RuntimeError("reference trajectory requires reset before advance")
        command = command.detach().reshape(-1).to(dtype=torch.float64).cpu()
        if command.shape != (3,) or not torch.isfinite(command).all():
            raise ValueError("command must be a finite [vx, vy, yaw_rate] tensor")
        dt = self._ctrl_dt
        vx, vy, omega = command.unbind()
        theta = omega * dt
        if abs(float(theta)) < 1e-8:
            # Zero-yaw-rate limit: V -> identity, pure straight-line motion.
            dx_body = vx * dt
            dy_body = vy * dt
        else:
            sin_t = torch.sin(theta)
            cos_t = torch.cos(theta)
            a = sin_t / theta
            b = (1.0 - cos_t) / theta
            dx_body = a * vx * dt - b * vy * dt
            dy_body = b * vx * dt + a * vy * dt
        yaw = self._pose[2]
        cos_y = torch.cos(yaw)
        sin_y = torch.sin(yaw)
        next_pose = torch.stack(
            (
                self._pose[0] + cos_y * dx_body - sin_y * dy_body,
                self._pose[1] + sin_y * dx_body + cos_y * dy_body,
                wrap_to_pi(yaw + theta),
            )
        )
        if not torch.isfinite(next_pose).all():
            raise ValueError("reference integration produced a non-finite pose")
        self._pose = next_pose.clone()
        return next_pose.to(dtype=torch.float32)

    @property
    def anchored(self) -> bool:
        return self._pose is not None

    @property
    def current_pose(self) -> torch.Tensor:
        """Latest desired world SE(2) pose ``[x, y, yaw]`` (detached, float32)."""
        if self._pose is None:
            raise RuntimeError("reference trajectory has no pose before reset")
        return self._pose.clone().to(dtype=torch.float32)
