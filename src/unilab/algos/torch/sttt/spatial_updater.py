"""Local spatial-gradient online updater for STTT Step 3.

Reuses the existing FADA ``OnlineSnapshotUpdater`` as the sole LoRA owner
(injection, optimizer, regression + anchor losses) and adds a measurable
real-endpoint objective: recent actual tracker/IDM actions are replayed with
gradients, pushed through the frozen differentiable world model, corrected
against the measured endpoint by exact SE(2) composition, and compared with
the episode-anchored reference endpoint using a periodic yaw residual.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch

from unilab.algos.torch.distill.fada.adaptation import (
    FADALoRAConfig,
    fada_adapter_named_parameters,
)
from unilab.algos.torch.distill.fada.model import FADAPlannerIDMPolicy
from unilab.algos.torch.distill.fada.official_model import PlannerIDMPolicy as OfficialPlannerIDMPolicy
from unilab.algos.torch.distill.fada.online_data import FADAOnlineSample
from unilab.algos.torch.distill.fada.online_updater import OnlineSnapshotUpdater
from unilab.algos.torch.distill.fada.runtime_policy import FADARuntimeSpec

from .geometry import wrap_to_pi
from .model import SpatialWorldModel
from .online_data import SpatialOnlineSample
from .policy_replay import replay_tracker_actions

__all__ = ["SpatialOnlineUpdater"]


class SpatialOnlineUpdater:
    """One optimizer over IDM LoRA adapters; frozen world model and base weights."""

    def __init__(
        self,
        policy: FADAPlannerIDMPolicy | OfficialPlannerIDMPolicy,
        lora_config: FADALoRAConfig,
        world_model: SpatialWorldModel,
        *,
        sigma: float,
        lr: float = 1e-4,
        max_grad_norm: float | None = 1.0,
        space_weight: float = 1.0,
        proprio_weight: float = 1.0,
        coordinate_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
        runtime_spec: FADARuntimeSpec | None = None,
        action_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        for name, value in (("space_weight", space_weight), ("proprio_weight", proprio_weight)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            len(coordinate_weights) != 3
            or any(not math.isfinite(w) or w < 0 for w in coordinate_weights)
            or sum(coordinate_weights) <= 0
        ):
            raise ValueError(
                "coordinate_weights require three non-negative finite values with positive sum"
            )

        self.snapshot_updater = OnlineSnapshotUpdater(
            policy, lora_config, sigma=sigma, lr=lr, max_grad_norm=max_grad_norm,
            runtime_spec=runtime_spec,
        )
        self.policy = self.snapshot_updater.policy
        self.optimizer = self.snapshot_updater.optimizer
        self.space_weight = float(space_weight)
        self.proprio_weight = float(proprio_weight)
        self.action_transform = action_transform
        self._adapters = dict(fada_adapter_named_parameters(self.policy))

        reference = next(self.policy.parameters())
        self.world_model = world_model.to(device=reference.device, dtype=reference.dtype)
        self.world_model.requires_grad_(False)
        self.world_model.eval()
        self._coordinate_weights = reference.new_tensor(tuple(coordinate_weights))

    @staticmethod
    def _corrected_endpoint(measured: torch.Tensor, predicted: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """T_measured * inverse(T_predicted_baseline) * T_predicted_candidate in SE(2)."""
        mx, my, myaw = measured.unbind(-1)
        bx, by, byaw = predicted[0].unbind(-1)
        cx, cy, cyaw = predicted[1].unbind(-1)
        # inverse(T_baseline) * T_candidate, both local to the window origin.
        x_rel = torch.cos(byaw) * (cx - bx) + torch.sin(byaw) * (cy - by)
        y_rel = -torch.sin(byaw) * (cx - bx) + torch.cos(byaw) * (cy - by)
        yaw_rel = wrap_to_pi(cyaw - byaw)
        # T_measured * relative transform.
        x_out = mx + torch.cos(myaw) * x_rel - torch.sin(myaw) * y_rel
        y_out = my + torch.sin(myaw) * x_rel + torch.cos(myaw) * y_rel
        yaw_out = wrap_to_pi(myaw + yaw_rel)
        return torch.stack((x_out, y_out, yaw_out), dim=-1)

    @torch.enable_grad()
    def compute_loss(
        self,
        spatial_samples: Sequence[SpatialOnlineSample],
        proprio_samples: Sequence[FADAOnlineSample],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (total, spatial, proprio_regression, anchor) scalar losses.

        ``spatial_samples`` may be empty when odometry is unavailable; the
        objective then reduces to the existing proprioceptive regression +
        anchor. The anchor enters the total exactly once.
        """
        _, regression, anchor = self.snapshot_updater.compute_loss(proprio_samples)
        total = self.proprio_weight * regression + anchor
        reference = self._coordinate_weights

        spatial = reference.new_zeros(())
        mature = 0
        for sample in spatial_samples:
            if sample.proprio_history.shape[1] != self.world_model.hist_len:
                raise ValueError("spatial history length must match the world model hist_len")
            if len(sample.snapshots) != self.world_model.horizon:
                raise ValueError("spatial snapshot count must match the world model horizon")
            if sample.actual_actions.shape != (1, self.world_model.horizon, self.world_model.act_dim):
                raise ValueError("actual actions must have shape (1, horizon, act_dim)")

            history = sample.proprio_history.to(reference)
            actual = sample.actual_actions.to(reference)
            candidate = replay_tracker_actions(
                self.policy,
                sample.snapshots,
                runtime_spec=self.snapshot_updater.runtime_spec,
                action_transform=self.action_transform,
            ).to(history.dtype)

            with torch.no_grad():
                baseline = self.world_model(history, actual)[:, -1]
            corrected = self._corrected_endpoint(
                sample.measured_endpoint.to(reference),
                (baseline, self.world_model(history, candidate)[:, -1]),
            )
            target = sample.reference_endpoint.to(reference)
            residual = corrected - target
            residual = torch.cat(
                (residual[..., :2], wrap_to_pi(residual[..., 2]).unsqueeze(-1)), dim=-1
            )
            spatial = spatial + (
                (self._coordinate_weights * residual.square()).sum(-1).mean()
                / self._coordinate_weights.sum()
            )
            mature += 1
        if mature:
            spatial = spatial / mature
            total = total + self.space_weight * spatial
        return total, spatial, regression, anchor

    @torch.enable_grad()
    def update(
        self,
        spatial_samples: Sequence[SpatialOnlineSample],
        proprio_samples: Sequence[FADAOnlineSample],
    ) -> dict[str, float | int]:
        """Take one Adam step on the LoRA adapters, rejecting non-finite state."""
        self.optimizer.zero_grad(set_to_none=True)
        total, spatial, regression, anchor = self.compute_loss(spatial_samples, proprio_samples)
        if not bool(torch.isfinite(total)):
            raise ValueError("spatial adaptation loss must be finite")
        total.backward()
        for name, parameter in self._adapters.items():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise ValueError(f"spatial adaptation gradient must be finite: {name}")
        if self.snapshot_updater.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                tuple(self._adapters.values()),
                self.snapshot_updater.max_grad_norm,
                error_if_nonfinite=True,
            )
        self.optimizer.step()
        return {
            "loss": float(total.detach()),
            "spatial_loss": float(spatial.detach()),
            "regression_loss": float(regression.detach()),
            "anchor_loss": float(anchor.detach()),
            "num_spatial_samples": len(spatial_samples),
            "num_proprio_samples": len(proprio_samples),
        }
