"""Online streaming adaptation updater for FADA LoRA-IDM."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from unilab.algos.torch.distill.fada.adaptation import (
    FADALoRAConfig,
    fada_adapter_named_parameters,
    inject_fada_idm_lora,
)
from unilab.algos.torch.distill.fada.model import FADAInverseDynamicsModel, FADAPlannerIDMPolicy
from unilab.algos.torch.distill.fada.online_data import FADAOnlineSample, FADATrajectoryPair


@torch.no_grad()
def build_expert_samples(
    base_idm: FADAInverseDynamicsModel,
    trajectories: Sequence[FADATrajectoryPair],
    *,
    gamma: float,
) -> tuple[FADAOnlineSample, ...]:
    """Translate matched futures through the frozen base, preserving executed snapshots."""
    if not trajectories:
        raise ValueError("expert labels require a non-empty trajectory window")
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError("gamma must be finite and non-negative")
    if any(module.training for module in base_idm.modules()) or any(
        parameter.requires_grad for parameter in base_idm.parameters()
    ):
        raise ValueError("base IDM must be frozen and in eval mode")
    reference = next(base_idm.parameters())
    snapshots = tuple(pair.snapshot for pair in trajectories)
    observations = torch.cat([snapshot.observation_history for snapshot in snapshots]).to(reference)
    actions = torch.cat([snapshot.action_history for snapshot in snapshots]).to(reference)
    nominal = torch.cat([snapshot.nominal_future for snapshot in snapshots]).to(reference)
    realized = torch.cat([pair.realized_future for pair in trajectories]).to(reference)
    executed = torch.cat([snapshot.action_snapshot for snapshot in snapshots]).to(reference)
    base_nominal = base_idm(observations, actions, nominal)[:, 0]
    base_realized = base_idm(observations, actions, realized)[:, 0]
    if base_nominal.shape != executed.shape or base_realized.shape != executed.shape:
        raise ValueError("expert predictions must match first-action snapshots")
    targets = executed + gamma * (base_nominal - base_realized)
    return tuple(
        FADAOnlineSample(pair, target[None].to(pair.snapshot.action_snapshot))
        for pair, target in zip(trajectories, targets)
    )


class OnlineSnapshotUpdater:
    """Regress fixed snapshot labels with an explicit deployment-parameter anchor.

    The collector owns the window. Each call consumes it without retaining,
    draining, or rebuilding samples; only IDM LoRA parameters are optimized.
    """

    def __init__(
        self,
        policy: FADAPlannerIDMPolicy,
        lora_config: FADALoRAConfig,
        *,
        sigma: float,
        lr: float = 1e-4,
        max_grad_norm: float | None = 1.0,
    ) -> None:
        if not math.isfinite(sigma) or sigma < 0.0:
            raise ValueError("sigma must be finite and non-negative")
        if not math.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be finite and positive")
        if max_grad_norm is not None and (
            not math.isfinite(max_grad_norm) or max_grad_norm <= 0.0
        ):
            raise ValueError("max_grad_norm must be finite and positive or None")

        adapted = inject_fada_idm_lora(policy, lora_config)
        self.policy = adapted.policy
        self.lora_config = adapted.lora_config
        self.sigma = float(sigma)
        self.max_grad_norm = max_grad_norm
        self._adapters = dict(fada_adapter_named_parameters(self.policy))
        self._parameters = tuple(self._adapters.values())
        self.theta_base = {
            name: parameter.detach().clone()
            for name, parameter in self._adapters.items()
        }
        self.optimizer = torch.optim.Adam(self._parameters, lr=lr, weight_decay=0.0)

    @torch.enable_grad()
    def compute_loss(
        self, samples: Sequence[FADAOnlineSample]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return total, per-sample squared-norm regression, and L2 anchor losses."""
        if not samples:
            raise ValueError("snapshot regression requires a non-empty sample window")
        reference = self._parameters[0]
        observations = torch.cat(
            [sample.trajectory.snapshot.observation_history for sample in samples], dim=0
        ).to(reference)
        actions = torch.cat(
            [sample.trajectory.snapshot.action_history for sample in samples], dim=0
        ).to(reference)
        futures = torch.cat(
            [sample.trajectory.snapshot.nominal_future for sample in samples], dim=0
        ).to(reference)
        targets = torch.cat(
            [sample.target_action for sample in samples], dim=0
        ).to(reference)

        modes = tuple((module, module.training) for module in self.policy.idm.modules())
        self.policy.idm.eval()
        try:
            predicted = self.policy.idm(observations, actions, futures)[:, 0]
            if predicted.shape != targets.shape:
                raise ValueError("snapshot targets must match first-action predictions")
            regression = 0.5 * (predicted - targets).square().sum(dim=-1).mean()
            anchor = reference.new_zeros(())
            for name, parameter in self._adapters.items():
                anchor = anchor + (parameter - self.theta_base[name]).square().sum()
            anchor = (0.5 * self.sigma) * anchor
            return regression + anchor, regression, anchor
        finally:
            for module, training in modes:
                module.training = training

    @torch.enable_grad()
    def update(self, samples: Sequence[FADAOnlineSample]) -> dict[str, float | int]:
        """Take one Adam step, including when called from a no-grad playback loop."""
        self.optimizer.zero_grad(set_to_none=True)
        total, regression, anchor = self.compute_loss(samples)
        if not bool(torch.isfinite(total)):
            raise ValueError("snapshot adaptation loss must be finite")
        total.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._parameters, self.max_grad_norm)
        self.optimizer.step()
        return {
            "loss": float(total.detach()),
            "regression_loss": float(regression.detach()),
            "anchor_loss": float(anchor.detach()),
            "num_samples": len(samples),
        }
