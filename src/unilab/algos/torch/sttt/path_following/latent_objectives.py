"""Five positive path costs for held-window latent residual control."""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral

import torch

from unilab.algos.torch.sttt.path_following.latent_rollout import RolloutResult
from unilab.algos.torch.sttt.world_model.types import RolloutStart

@dataclass(frozen=True)
class ObservedCostWindow:
    """Measured physical outcomes; no imagined state is used as a real label."""
    errors: torch.Tensor
    path_delta: torch.Tensor
    actions: torch.Tensor
    source_actions: torch.Tensor
    residuals: torch.Tensor
    risk: torch.Tensor
    commands: torch.Tensor
    target_speed: torch.Tensor


@dataclass(frozen=True)
class LatentCostConfig:
    lateral_weight: float = .5
    lateral_velocity_weight: float = 1.
    forward_weight: float = 1.
    smoothness_weight: float = .01
    risk_weight: float = 1.
    lateral_tolerance_m: float = .3
    lateral_velocity_tolerance_m_s: float = 1.
    forward_speed_floor_m_s: float = .1
    discount: float = .99
    failure_cost_rate: float = 4.

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.discount >= 1:
            raise ValueError("native-step discount must be in (0,1)")
        maximum = sum(getattr(self, name) for name in
                      ("lateral_weight", "lateral_velocity_weight", "forward_weight", "smoothness_weight", "risk_weight"))
        if self.failure_cost_rate <= maximum:
            raise ValueError("absorbing failure cost must exceed the complete nonterminal cost bound")


def _penalty(value: torch.Tensor) -> torch.Tensor:
    return -torch.expm1(-value.square())


def latent_window_costs(
    result: RolloutResult | ObservedCostWindow, start: RolloutStart, *, ctrl_dt: float, window_steps: int,
    config: LatentCostConfig, residual_limits: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Native-time discounted integrals, one value per held residual window."""
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
    command_dim = start.command.shape[-1]
    if command_dim not in (3, 7):
        raise ValueError("actor costs require an admitted velocity-command contract")
    expected = (
        ("errors", result.errors, (batch, steps + 1, 2)),
        ("actions", result.actions, (batch, steps, 29)),
        ("source_actions", result.source_actions, (batch, steps, 29)),
        ("residuals", result.residuals, (batch, windows, 29)),
        ("risk", result.risk, (batch, steps, 1)),
        ("commands", result.commands, (batch, steps, command_dim)),
        ("target_speed", result.target_speed, (batch, steps)),
        ("start.errors", start.errors, (batch, 2)),
        ("start.command", start.command, (batch, command_dim)),
        ("start.previous_residual", start.previous_residual, (batch, 29)),
        ("residual_limits", residual_limits, (29,)),
    )
    if not delta.is_floating_point() or not bool(torch.isfinite(delta).all()):
        raise ValueError("latent actor costs require finite floating-point path increments")
    for name, value, shape in expected:
        if value.shape != shape or value.dtype != delta.dtype or value.device != delta.device or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} has incompatible shape, dtype, device or non-finite values")
    if bool((residual_limits <= 0).any()):
        raise ValueError("residual limits must be positive")
    if bool((result.residuals.abs() > residual_limits + 1e-7).any()) or bool((start.previous_residual.abs() > residual_limits + 1e-7).any()):
        raise ValueError("residuals exceed the execution envelope")
    if bool(((result.risk < 0) | (result.risk > 1)).any()):
        raise ValueError("rollout risk must contain probabilities, not logits")
    duration = float(window_steps) * ctrl_dt
    window_delta = delta.reshape(batch, windows, window_steps, 3).sum(dim=2)
    lateral_velocity = window_delta[..., 1] / duration
    forward_velocity = window_delta[..., 0] / duration
    nominal_forward = result.target_speed.detach().reshape(batch, windows, window_steps).mean(-1)
    forward_scale = nominal_forward.abs().clamp_min(config.forward_speed_floor_m_s)
    previous = torch.cat((start.previous_residual.detach().unsqueeze(1), result.residuals[:, :-1]), dim=1)
    change = (result.residuals - previous) / (2 * residual_limits)
    step_discount = config.discount ** torch.arange(window_steps, device=delta.device, dtype=delta.dtype)
    window_measure = step_discount.sum() * ctrl_dt
    lateral = _penalty(result.errors[:, 1:, 0] / config.lateral_tolerance_m)
    lateral = (lateral.reshape(batch, windows, window_steps) * step_discount).sum(-1) * ctrl_dt
    risk = (result.risk[..., 0].reshape(batch, windows, window_steps) * step_discount).sum(-1) * ctrl_dt
    return {
        "lateral_loss": lateral,
        "lateral_velocity_loss": _penalty(lateral_velocity / config.lateral_velocity_tolerance_m_s) * window_measure,
        "forward_loss": _penalty((forward_velocity - nominal_forward) / forward_scale) * window_measure,
        "smoothness_loss": change.square().mean(-1) * window_measure,
        "risk_loss": risk,
    }


def weighted_window_costs(costs: dict[str, torch.Tensor], config: LatentCostConfig) -> torch.Tensor:
    return (config.lateral_weight * costs["lateral_loss"]
            + config.lateral_velocity_weight * costs["lateral_velocity_loss"]
            + config.forward_weight * costs["forward_loss"]
            + config.smoothness_weight * costs["smoothness_loss"]
            + config.risk_weight * costs["risk_loss"])


def latent_actor_loss(
    result: RolloutResult | ObservedCostWindow, start: RolloutStart, *, ctrl_dt: float, window_steps: int,
    config: LatentCostConfig, residual_limits: torch.Tensor,
    terminal_cost: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    costs = latent_window_costs(result, start, ctrl_dt=ctrl_dt, window_steps=window_steps,
                                config=config, residual_limits=residual_limits)
    combined = weighted_window_costs(costs, config)
    discount = config.discount ** window_steps
    horizon_discount = discount ** torch.arange(combined.shape[1], device=combined.device, dtype=combined.dtype)
    weights = {"lateral_loss": config.lateral_weight, "lateral_velocity_loss": config.lateral_velocity_weight,
               "forward_loss": config.forward_weight, "smoothness_loss": config.smoothness_weight, "risk_loss": config.risk_weight}
    losses = {name: (value * horizon_discount).sum(-1).mean() for name, value in costs.items()}
    weighted = {f"weighted_{name}": value * weights[name] for name, value in losses.items()}
    bootstrap = combined.new_zeros(())
    if terminal_cost is not None:
        if terminal_cost.shape != (combined.shape[0],) or not bool(torch.isfinite(terminal_cost).all()):
            raise ValueError("terminal cost must contain one finite value per rollout")
        bootstrap = terminal_cost.mean() * discount ** combined.shape[1]
    return {"loss": (combined * horizon_discount).sum(-1).mean() + bootstrap,
            **losses, **weighted, "terminal_cost": bootstrap}
