"""Owned real-history starts shared by collection and effective-batch learners."""
from __future__ import annotations

from dataclasses import fields
from typing import Any, Sequence

import torch

from ..world_model.types import LatentState, RolloutStart
from .trajectory import PathState, ReferenceTrajectory




def select_path(path: PathState, index) -> PathState:
    return PathState(ReferenceTrajectory(path.reference.xy[index], path.reference.speed[index], path.reference.config,
                                         path.reference.command_bounds),
                     path.pose[index], path.progress[index])


def select_rollout_start(start: RolloutStart, index) -> RolloutStart:
    values = {item.name: getattr(start, item.name)[index] for item in fields(start)
              if item.name not in {"latent", "path_state"}}
    values["latent"] = LatentState(start.latent.state[index], start.latent.belief[index])
    values["path_state"] = None if start.path_state is None else select_path(start.path_state, index)
    return RolloutStart(**values)


def join_rollout_starts(starts: Sequence[RolloutStart]) -> RolloutStart:
    if not starts or any(start.path_state is None for start in starts):
        raise ValueError("real-history actor starts require explicit reference states")
    paths = [start.path_state for start in starts if start.path_state is not None]
    config = paths[0].reference.config
    if any(path.reference.config != config or path.reference.command_bounds != paths[0].reference.command_bounds for path in paths):
        raise ValueError("joint actor starts must share the reference preview contract")
    reference = ReferenceTrajectory(torch.cat([path.reference.xy for path in paths]),
                                    torch.cat([path.reference.speed for path in paths]), config,
                                    paths[0].reference.command_bounds)
    values: dict[str, Any] = {item.name: torch.cat([getattr(start, item.name) for start in starts])
                             for item in fields(starts[0]) if item.name not in {"latent", "path_state"}}
    values["latent"] = LatentState(torch.cat([start.latent.state for start in starts]),
                                   torch.cat([start.latent.belief for start in starts]))
    values["path_state"] = PathState(reference, torch.cat([path.pose for path in paths]),
                                     torch.cat([path.progress for path in paths]))
    return RolloutStart(**values)
