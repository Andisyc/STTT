"""Owned, graph-free data passed from FADA inference to online adaptation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from unilab.algos.torch.distill.fada.observation import (
    FADA_G1_STATE_OBSERVATION_CONTRACT,
    fada_g1_joint_position_offsets,
)


@dataclass(frozen=True)
class FADAInferenceSnapshot:
    """One query's inputs and outputs; consumers must not mutate its tensors."""

    observation_history: torch.Tensor
    action_history: torch.Tensor
    command: torch.Tensor
    nominal_future: torch.Tensor
    action_snapshot: torch.Tensor

    def __post_init__(self) -> None:
        for name in (
            "observation_history", "action_history", "command",
            "nominal_future", "action_snapshot",
        ):
            object.__setattr__(self, name, getattr(self, name).detach().clone())

    @property
    def nominal_joint_positions(self) -> torch.Tensor:
        """Next-step joint offsets in radians, not absolute position commands."""
        return fada_g1_joint_position_offsets(
            self.nominal_future[:, 0],
            observation_contract=FADA_G1_STATE_OBSERVATION_CONTRACT,
        )


@dataclass(frozen=True)
class FADAOnlineSample:
    """A snapshot closed by its next observation; labels are never recomputed."""

    snapshot: FADAInferenceSnapshot
    joint_position_error: torch.Tensor
    target_action: torch.Tensor
    episode_id: int
    timestep: int

    def __post_init__(self) -> None:
        for name in ("joint_position_error", "target_action"):
            value = getattr(self, name)
            if value.shape != self.snapshot.action_snapshot.shape:
                raise ValueError(f"{name} must match the snapshot action shape")
            object.__setattr__(self, name, value.detach().clone())
        if self.episode_id < 0 or self.timestep < 0:
            raise ValueError("online sample episode_id and timestep must be non-negative")
