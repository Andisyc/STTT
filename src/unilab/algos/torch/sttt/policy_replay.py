"""Gradient-preserving replay of recorded FADA policy queries for spatial TTT.

Each :class:`~unilab.algos.torch.distill.fada.online_data.FADAInferenceSnapshot`
holds one actual past inference query (histories, command, nominal future,
mask). Replaying concatenates H saved queries and re-runs the IDM with the
current (LoRA-adapted) weights, so the first-action predictions carry an
intact autograd graph into the spatial endpoint loss. Snapshots are treated
as immutable evidence: they are never mutated, and no live playback history,
guessed future proprioception, or padded horizon is consulted.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from unilab.algos.torch.distill.fada.model import FADAPlannerIDMPolicy
from unilab.algos.torch.distill.fada.official_model import PlannerIDMPolicy as OfficialPlannerIDMPolicy
from unilab.algos.torch.distill.fada.online_data import FADAInferenceSnapshot
from unilab.algos.torch.distill.fada.runtime_policy import (
    FADARuntimeSpec,
    get_fada_runtime_spec,
    invoke_fada_idm,
)

__all__ = ["replay_tracker_actions"]


@torch.enable_grad()
def replay_tracker_actions(
    policy: FADAPlannerIDMPolicy | OfficialPlannerIDMPolicy,
    snapshots: Sequence[FADAInferenceSnapshot],
    *,
    runtime_spec: FADARuntimeSpec | None = None,
    action_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Re-run the IDM on H saved queries and return first actions ``(1, H, 29)``.

    The predictions keep their autograd graph (no ``no_grad``/``detach``), so
    downstream losses differentiate through the current shared LoRA weights.
    ``action_transform`` (e.g. the official affine env-input reorder, plus any
    existing clip wrapper supplied by the caller) is applied to the flattened
    ``(H, 29)`` predictions before reshaping, preserving the Torch graph.

    The IDM is set to eval mode for deterministic replay and its original
    module modes are restored afterwards. The frozen Planner is never invoked.
    """
    if not snapshots:
        raise ValueError("policy replay requires at least one snapshot")
    if runtime_spec is not None:
        spec = runtime_spec
    elif isinstance(policy, (FADAPlannerIDMPolicy, OfficialPlannerIDMPolicy)):
        spec = get_fada_runtime_spec(policy)
    else:
        raise TypeError("policy replay requires a legacy or official FADA policy")
    idm = policy.idm

    observations = torch.cat([snapshot.observation_history for snapshot in snapshots], dim=0)
    actions = torch.cat([snapshot.action_history for snapshot in snapshots], dim=0)
    futures = torch.cat([snapshot.nominal_future for snapshot in snapshots], dim=0)
    if spec.family == "official":
        commands = torch.cat([snapshot.command for snapshot in snapshots], dim=0)
        masks = [snapshot.history_valid_mask for snapshot in snapshots]
        if any(mask is None for mask in masks):
            raise ValueError("official policy replay requires snapshot history_valid_mask")
        mask = torch.cat(masks, dim=0)
        if mask.dtype != torch.bool:
            raise ValueError("official policy replay requires a boolean history_valid_mask")
    else:
        commands = None
        mask = None

    modes = tuple((module, module.training) for module in idm.modules())
    idm.eval()
    try:
        predicted = invoke_fada_idm(
            idm, spec, observations, actions, commands, futures, history_valid_mask=mask
        )[:, 0]
    finally:
        for module, training in modes:
            module.training = training

    if predicted.ndim != 2 or predicted.shape[0] != len(snapshots):
        raise ValueError("replayed first actions must have shape (H, action_dim)")
    if action_transform is not None:
        predicted = action_transform(predicted)
        if predicted.ndim != 2 or predicted.shape[0] != len(snapshots):
            raise ValueError("action_transform must preserve the (H, action_dim) shape")
    return predicted.unsqueeze(0)
