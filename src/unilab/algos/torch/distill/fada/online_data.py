"""Owned, graph-free data passed from FADA inference to online adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import torch

from unilab.algos.torch.distill.fada.observation import (
    FADA_G1_STATE_DIM,
    FADA_G1_STATE_OBSERVATION_CONTRACT,
    FADA_OFFICIAL_COMPACT_OBSERVATION_CONTRACT,
)


@dataclass(frozen=True)
class FADAInferenceSnapshot:
    """One query's inputs and outputs; consumers must not mutate its tensors."""

    observation_history: torch.Tensor
    action_history: torch.Tensor
    command: torch.Tensor
    nominal_future: torch.Tensor
    action_snapshot: torch.Tensor
    observation_contract: str = FADA_G1_STATE_OBSERVATION_CONTRACT
    history_valid_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        for name in (
            "observation_history", "action_history", "command",
            "nominal_future", "action_snapshot",
        ):
            object.__setattr__(self, name, getattr(self, name).detach().clone())
        if self.history_valid_mask is not None:
            object.__setattr__(self, "history_valid_mask", self.history_valid_mask.detach().clone())
        if self.observation_contract == FADA_OFFICIAL_COMPACT_OBSERVATION_CONTRACT:
            obs, act, mask = self.observation_history, self.action_history, self.history_valid_mask
            if obs.ndim != 3 or obs.shape[-1] != 64 or act.shape != (*obs.shape[:2], 29):
                raise ValueError("official snapshot requires aligned 64/29 observation/action histories")
            if self.command.shape != (obs.shape[0], 7) or self.action_snapshot.shape != (obs.shape[0], 29):
                raise ValueError("official snapshot requires 7D command and 29D first action")
            if self.nominal_future.ndim != 3 or self.nominal_future.shape[0] != obs.shape[0] or self.nominal_future.shape[-1] != 64:
                raise ValueError("official snapshot requires a 64D nominal future")
            if mask is None or mask.shape != obs.shape[:2] or mask.dtype != torch.bool or mask.device != obs.device:
                raise ValueError("official snapshot requires a boolean history_valid_mask")
            if not bool(mask[:, -1].all()) or bool((mask[:, :-1] & ~mask[:, 1:]).any()):
                raise ValueError("official snapshot history mask must be a valid suffix")
            if bool((obs[~mask] != 0).any()) or bool((act[~mask] != 0).any()):
                raise ValueError("official snapshot history padding must be zero")
        elif self.observation_contract != FADA_G1_STATE_OBSERVATION_CONTRACT:
            raise ValueError("unsupported online snapshot observation contract")



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
        expected_dim = (
            64 if self.snapshot.observation_contract == FADA_OFFICIAL_COMPACT_OBSERVATION_CONTRACT
            else FADA_G1_STATE_DIM
        )
        if (
            nominal.ndim != 3
            or nominal.shape[0] != 1
            or nominal.shape[1] < 1
            or nominal.shape[2] != expected_dim
            or self.realized_future.shape != nominal.shape
        ):
            raise ValueError(f"trajectory futures must share shape (1, K, {expected_dim}) with K > 0")
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
