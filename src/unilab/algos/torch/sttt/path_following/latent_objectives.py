"""Five positive path costs for held-window latent residual control."""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral

import torch

from unilab.algos.torch.sttt.path_following.latent_rollout import RolloutResult
from unilab.algos.torch.sttt.world_model.types import RolloutStart


@dataclass(frozen=True)
class LatentCostConfig:
    lateral_weight: float = 1.
    lateral_velocity_weight: float = 1.
    forward_weight: float = 1.
    smoothness_weight: float = 1.
    risk_weight: float = 1.
    huber_delta: float | None = None

    def __post_init__(self) -> None:
        for name in ("lateral_weight", "lateral_velocity_weight", "forward_weight", "smoothness_weight", "risk_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.huber_delta is not None and (
            isinstance(self.huber_delta, bool) or not math.isfinite(self.huber_delta) or self.huber_delta <= 0
        ):
            raise ValueError("huber_delta must be positive and finite, or None")


def _penalty(value: torch.Tensor, delta: float | None) -> torch.Tensor:
    if delta is None:
        return .5 * value.square()
    magnitude = value.abs()
    quadratic = magnitude.clamp(max=delta)
    return .5 * quadratic.square() + delta * (magnitude - quadratic)


def latent_actor_loss(
    result: RolloutResult,
    start: RolloutStart,
    *,
    ctrl_dt: float,
    window_steps: int,
    config: LatentCostConfig,
) -> dict[str, torch.Tensor]:
    """Return total and unweighted ``*_loss`` scalar tensors.

    Lateral offset uses all future ticks; actual lateral and forward velocities
    use summed fixed-path displacements over each complete actor window. The
    nominal forward velocity is the initial body command projected into the
    path frame with the initial heading, then detached and held fixed. Residual
    smoothness penalizes changes per window duration, including the transition
    from ``start.previous_residual``. Risk is probability integrated in seconds.
    There is no heading-to-zero, residual-magnitude or survival-discount term:
    nonzero heading/joint trim is free when it produces the desired motion.
    """
    if isinstance(ctrl_dt, bool) or not math.isfinite(ctrl_dt) or ctrl_dt <= 0:
        raise ValueError("ctrl_dt must be positive and finite")
    if isinstance(window_steps, bool) or not isinstance(window_steps, Integral) or window_steps < 1:
        raise ValueError("window_steps must be a positive integer")
    delta = result.path_delta
    if delta.ndim != 3 or delta.shape[0] < 1 or delta.shape[-1] != 3 or delta.shape[1] < 1:
        raise ValueError("latent actor costs require nonempty path_delta [B,S,3]")
    batch, steps = delta.shape[:2]
    if steps % window_steps:
        raise ValueError("latent actor costs require complete residual windows")
    windows = steps // window_steps
    expected = (
        ("errors", result.errors, (batch, steps + 1, 2)),
        ("actions", result.actions, (batch, steps, 29)),
        ("source_actions", result.source_actions, (batch, steps, 29)),
        ("residuals", result.residuals, (batch, windows, 29)),
        ("risk", result.risk, (batch, steps, 1)),
        ("commands", result.commands, (batch, steps, 7)),
        ("start.errors", start.errors, (batch, 2)),
        ("start.command", start.command, (batch, 7)),
        ("start.previous_residual", start.previous_residual, (batch, 29)),
    )
    if not delta.is_floating_point() or not bool(torch.isfinite(delta).all()):
        raise ValueError("latent actor costs require finite floating-point path increments")
    for name, value, shape in expected:
        if value.shape != shape or value.dtype != delta.dtype or value.device != delta.device or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} has incompatible shape, dtype, device or non-finite values")
    if bool(((result.risk < 0) | (result.risk > 1)).any()):
        raise ValueError("rollout risk must contain probabilities, not logits")
    duration = float(window_steps) * ctrl_dt
    window_delta = delta.reshape(batch, windows, window_steps, 3).sum(dim=2)
    lateral_velocity = window_delta[..., 1] / duration
    forward_velocity = window_delta[..., 0] / duration
    nominal_forward = (
        start.command[:, 0] * torch.cos(start.errors[:, 1])
        - start.command[:, 1] * torch.sin(start.errors[:, 1])
    ).detach().unsqueeze(1)
    previous = torch.cat((start.previous_residual.detach().unsqueeze(1), result.residuals[:, :-1]), dim=1)
    residual_velocity = (result.residuals - previous) / duration
    losses = {
        "lateral_loss": _penalty(result.errors[:, 1:, 0], config.huber_delta).mean(),
        "lateral_velocity_loss": _penalty(lateral_velocity, config.huber_delta).mean(),
        "forward_loss": _penalty(forward_velocity - nominal_forward, config.huber_delta).mean(),
        "smoothness_loss": _penalty(residual_velocity, config.huber_delta).mean(),
        "risk_loss": (result.risk.sum(dim=1) * ctrl_dt).mean(),
    }
    total = (
        config.lateral_weight * losses["lateral_loss"]
        + config.lateral_velocity_weight * losses["lateral_velocity_loss"]
        + config.forward_weight * losses["forward_loss"]
        + config.smoothness_weight * losses["smoothness_loss"]
        + config.risk_weight * losses["risk_loss"]
    )
    return {"loss": total, **losses}
