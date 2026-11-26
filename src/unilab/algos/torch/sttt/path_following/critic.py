"""Window-rate cost values with TD-lambda targets and a slow target network."""
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CriticConfig:
    hidden_dims: tuple[int, ...] = (256, 128)
    learning_rate: float = .0005
    epochs: int = 8
    minibatches: int = 4
    lambda_return: float = .95
    target_tau: float = .01
    max_grad_norm: float = 100.
    optimizer_betas: tuple[float, float] = (.7, .95)

    def __post_init__(self) -> None:
        if not self.hidden_dims or any(type(width) is not int or width < 1 for width in self.hidden_dims):
            raise ValueError("critic hidden dimensions must be positive integers")
        for name in ("epochs", "minibatches"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"critic {name} must be a positive integer")
        for name in ("learning_rate", "lambda_return", "target_tau", "max_grad_norm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"critic {name} must be positive and finite")
        if self.lambda_return > 1 or self.target_tau > 1:
            raise ValueError("critic lambda and target_tau must be in (0,1]")
        if len(self.optimizer_betas) != 2 or any(isinstance(value, bool) or not math.isfinite(value) or not 0 <= value < 1 for value in self.optimizer_betas):
            raise ValueError("critic optimizer betas must contain two values in [0,1)")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CriticConfig:
        raw = dict(value)
        if "hidden_dims" in raw:
            raw["hidden_dims"] = tuple(raw["hidden_dims"])
        if "optimizer_betas" in raw:
            raw["optimizer_betas"] = tuple(raw["optimizer_betas"])
        return cls(**raw)


class CostCritic(nn.Module):
    def __init__(self, context_dim: int, config: CriticConfig, maximum_value: float):
        super().__init__()
        self.maximum_value = maximum_value
        layers: list[nn.Module] = []
        previous = context_dim
        for width in config.hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.ELU(), nn.LayerNorm(width)))
            previous = width
        head = nn.Linear(previous, 1)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        layers.append(head)
        self.net = nn.Sequential(*layers)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(context).squeeze(-1)).clamp_max(self.maximum_value)


def lambda_cost_targets(costs: torch.Tensor, values: torch.Tensor, *, discount: float,
                        lambda_return: float, terminal: torch.Tensor,
                        failure_value: float) -> torch.Tensor:
    """True failures retain their absorbing cost; time limits keep bootstrapping."""
    if costs.ndim != 2 or values.shape != (costs.shape[0], costs.shape[1] + 1):
        raise ValueError("cost targets require [B,H] costs and [B,H+1] values")
    if terminal.shape != costs.shape or terminal.dtype != torch.bool:
        raise ValueError("terminal flags must be boolean [B,H]")
    with torch.no_grad():
        targets = torch.empty_like(costs)
        following = values[:, -1]
        for step in reversed(range(costs.shape[1])):
            bootstrap = (1 - lambda_return) * values[:, step + 1] + lambda_return * following
            bootstrap = torch.where(terminal[:, step], failure_value, bootstrap)
            following = costs[:, step] + discount * bootstrap
            targets[:, step] = following
    return targets


class CriticLearner:
    def __init__(self, context_dim: int, config: CriticConfig, *, parameter: torch.Tensor,
                 ctrl_dt: float, window_steps: int, discount: float, failure_cost_rate: float):
        self.config = config
        self.discount = discount ** window_steps
        self.failure_value = failure_cost_rate * ctrl_dt / (1 - discount)
        self.critic = CostCritic(context_dim, config, self.failure_value).to(parameter)
        self.target = deepcopy(self.critic).eval().requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.learning_rate, betas=config.optimizer_betas)
        self.updates = 0
        self.signature = {"config": asdict(config), "context_dim": context_dim,
                          "discount": self.discount, "failure_value": self.failure_value}

    def update(self, contexts: torch.Tensor, costs: torch.Tensor,
               terminal: torch.Tensor, *, observed_contexts: torch.Tensor | None = None,
               observed_targets: torch.Tensor | None = None) -> dict[str, float]:
        contexts, costs = contexts.detach(), costs.detach()
        with torch.no_grad():
            values = self.target(contexts)
            targets = lambda_cost_targets(costs, values, discount=self.discount,
                lambda_return=self.config.lambda_return, terminal=terminal,
                failure_value=self.failure_value).flatten()
        inputs = contexts[:, :-1].flatten(0, 1)
        total = costs.new_zeros(())
        count = 0
        self.critic.train()
        if observed_contexts is not None:
            if observed_targets is None or observed_targets.shape != (observed_contexts.shape[0],):
                raise ValueError("measured critic contexts require matched return targets")
            inputs = torch.cat((inputs, observed_contexts.detach()))
            targets = torch.cat((targets, observed_targets.detach()))
        for _ in range(self.config.epochs):
            order = torch.randperm(len(targets), device=inputs.device)
            for index in order.tensor_split(min(self.config.minibatches, len(targets))):
                self.optimizer.zero_grad(set_to_none=True)
                loss = F.mse_loss(self.critic(inputs[index]), targets[index])
                if not bool(torch.isfinite(loss)):
                    raise ValueError("non-finite cost critic loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm,
                                               error_if_nonfinite=True)
                self.optimizer.step()
                with torch.no_grad():
                    for target, current in zip(self.target.parameters(), self.critic.parameters()):
                        target.lerp_(current, self.config.target_tau)
                self.updates += 1
                count += 1
                total += loss.detach()
        self.optimizer.zero_grad(set_to_none=True)
        return {"critic_loss": float(total / count), "critic_updates": self.updates,
                "value_target_mean": float(targets.mean())}

    def state_dict(self) -> dict[str, Any]:
        return {"version": 2, "signature": self.signature, "critic": self.critic.state_dict(),
                "target": self.target.state_dict(), "optimizer": self.optimizer.state_dict(),
                "optimizer_steps": self.updates}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != 2 or state.get("signature") != self.signature:
            raise ValueError("cost critic configuration or temporal meaning changed")
        self.critic.load_state_dict(state["critic"], strict=True)
        self.target.load_state_dict(state["target"], strict=True)
        if any(not bool(torch.isfinite(value).all()) for model in (self.critic, self.target)
               for value in model.state_dict().values()):
            raise ValueError("cost critic checkpoint contains non-finite values")
        if type(state["optimizer_steps"]) is not int or state["optimizer_steps"] < 0:
            raise ValueError("invalid critic optimizer counter")
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = state["optimizer_steps"]
