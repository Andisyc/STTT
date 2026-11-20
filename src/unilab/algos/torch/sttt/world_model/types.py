"""Tensor state shared by the physical model and imagined control loop."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class LatentState:
    state: torch.Tensor
    belief: torch.Tensor

    def feature(self) -> torch.Tensor:
        return torch.cat((self.state.flatten(-2), self.belief), dim=-1)

    def detach(self) -> LatentState:
        return LatentState(self.state.detach(), self.belief.detach())

    def clone(self) -> LatentState:
        return LatentState(self.state.clone(), self.belief.clone())

    def to(self, *args: Any, **kwargs: Any) -> LatentState:
        return LatentState(self.state.to(*args, **kwargs), self.belief.to(*args, **kwargs))


@dataclass(frozen=True)
class RolloutStart:
    latent: LatentState
    observation_history: torch.Tensor
    action_history: torch.Tensor
    history_valid_mask: torch.Tensor
    errors: torch.Tensor
    command: torch.Tensor
    previous_residual: torch.Tensor

    def __post_init__(self) -> None:
        obs = self.observation_history
        if obs.ndim != 3 or obs.shape[-1] not in (64, 66) or obs.shape[0] < 1 or obs.shape[1] < 1:
            raise ValueError("rollout history requires [batch, history, observation_dim] observations")
        batch, history = obs.shape[:2]
        fields = (
            (self.action_history, (batch, history, 29)),
            (self.errors, (batch, 2)),
            (self.command, (batch, 3 if obs.shape[-1] == 66 else 7)),
            (self.previous_residual, (batch, 29)),
        )
        for value, shape in fields:
            if value.shape != shape or value.device != obs.device or value.dtype != obs.dtype:
                raise ValueError("rollout inputs must have matching batch, device and floating dtype")
        if self.history_valid_mask.shape != (batch, history) or self.history_valid_mask.dtype != torch.bool:
            raise ValueError("rollout history requires a boolean validity mask")
        if self.history_valid_mask.device != obs.device or not bool(self.history_valid_mask[:, -1].all()):
            raise ValueError("rollout requires a current valid observation on the history device")
        if self.latent.belief.shape[0] != batch or self.latent.state.shape[0] != batch:
            raise ValueError("rollout latent and history batches differ")
