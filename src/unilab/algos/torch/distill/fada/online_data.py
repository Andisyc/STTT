"""Owned, graph-free data passed from FADA inference to online adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import torch

from unilab.algos.torch.distill.fada.observation import FADA_G1_STATE_DIM


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



@dataclass(frozen=True)
class FADATrajectoryPair:
    """Own a complete realized future aligned to an inference snapshot's origin.

    Fields are frozen; consumers must also avoid in-place tensor mutation.
    The collector owns causal alignment across the prediction horizon.
    """

    snapshot: FADAInferenceSnapshot
    realized_future: torch.Tensor
    episode_id: int
    timestep: int

    def __post_init__(self) -> None:
        for name in ("episode_id", "timestep"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"trajectory {name} must be a non-negative integer")
        nominal = self.snapshot.nominal_future
        if (
            nominal.ndim != 3
            or nominal.shape[0] != 1
            or nominal.shape[1] < 1
            or nominal.shape[2] != FADA_G1_STATE_DIM
            or self.realized_future.shape != nominal.shape
        ):
            raise ValueError("trajectory futures must share shape (1, K, 66) with K > 0")
        if (
            self.realized_future.dtype != nominal.dtype
            or self.realized_future.device != nominal.device
        ):
            raise ValueError("trajectory futures must share dtype and device")
        if not bool(torch.isfinite(nominal).all()) or not bool(torch.isfinite(self.realized_future).all()):
            raise ValueError("trajectory futures must be finite")
        object.__setattr__(self, "realized_future", self.realized_future.detach().clone())


@dataclass(frozen=True)
class FADAOnlineSample:
    """A trajectory pair with an owned, fixed action-space regression target."""

    trajectory: FADATrajectoryPair
    target_action: torch.Tensor

    def __post_init__(self) -> None:
        reference = self.trajectory.snapshot.action_snapshot
        if self.target_action.shape != reference.shape:
            raise ValueError("expert target must match the snapshot action shape")
        if self.target_action.dtype != reference.dtype or self.target_action.device != reference.device:
            raise ValueError("expert target must share the snapshot action dtype and device")
        if not bool(torch.isfinite(self.target_action).all()):
            raise ValueError("expert target must be finite")
        object.__setattr__(self, "target_action", self.target_action.detach().clone())
