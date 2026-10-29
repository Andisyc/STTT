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
from unilab.algos.torch.distill.fada.observation import (
    FADA_OFFICIAL_COMPACT_OBSERVATION_CONTRACT,
)
from unilab.algos.torch.distill.fada.official_model import IDM as OfficialIDM
from unilab.algos.torch.distill.fada.official_model import PlannerIDMPolicy as OfficialPlannerIDMPolicy
from unilab.algos.torch.distill.fada.online_data import FADAOnlineSample, FADATrajectoryPair
from unilab.algos.torch.distill.fada.runtime_policy import (
    FADARuntimeSpec,
    get_fada_runtime_spec,
    invoke_fada_idm,
)


def _resolve_runtime_spec(
    subject: object, runtime_spec: FADARuntimeSpec | None
) -> FADARuntimeSpec:
    """Bind the explicit deployment spec, deriving legacy/official identity when omitted."""
    if runtime_spec is not None:
        return runtime_spec
    if isinstance(subject, (FADAPlannerIDMPolicy, OfficialPlannerIDMPolicy)):
        return get_fada_runtime_spec(subject)
    if isinstance(subject, OfficialIDM):
        raise ValueError(
            "official expert labels require an explicit runtime_spec for the base official IDM"
        )
    if isinstance(subject, FADAInverseDynamicsModel):
        config = subject.config
        return FADARuntimeSpec(
            family="legacy",
            obs_dim=config.obs_dim,
            action_dim=config.action_dim,
            command_dim=config.command_dim,
            observation_contract=config.observation_contract,
            history_length=config.history_length,
            prediction_horizon=config.prediction_horizon,
        )
    raise TypeError(
        "runtime_spec is required unless the subject is a recognizable legacy or official FADA module"
    )


def _trajectory_masks(
    spec: FADARuntimeSpec, snapshots: tuple, reference: torch.Tensor
) -> torch.Tensor | None:
    """Concatenate official boolean masks; legacy passes no mask without allocating."""
    if spec.family != "official":
        return None
    masks = [snapshot.history_valid_mask for snapshot in snapshots]
    if any(mask is None for mask in masks):
        raise ValueError("official expert labels require snapshot history_valid_mask")
    combined = torch.cat(masks).to(device=reference.device)
    if combined.dtype != torch.bool:
        raise ValueError("official expert labels require a boolean history_valid_mask")
    return combined


@torch.no_grad()
def build_expert_samples(
    base_idm,
    trajectories: Sequence[FADATrajectoryPair],
    *,
    gamma: float,
    runtime_spec: FADARuntimeSpec | None = None,
) -> tuple[FADAOnlineSample, ...]:
    """Translate matched futures through the frozen base, preserving executed snapshots."""
    if not trajectories:
        raise ValueError("expert labels require a non-empty trajectory window")
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError("gamma must be finite and non-negative")
    spec = _resolve_runtime_spec(base_idm, runtime_spec)
    if spec.family == "official" and any(
        pair.snapshot.observation_contract != FADA_OFFICIAL_COMPACT_OBSERVATION_CONTRACT
        for pair in trajectories
    ):
        raise ValueError("official expert labels require official_compact_64_v1 snapshots")
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
    if spec.family == "official":
        commands = torch.cat([snapshot.command for snapshot in snapshots]).to(reference)
        masks = _trajectory_masks(spec, snapshots, reference)
        base_nominal = invoke_fada_idm(
            base_idm, spec, observations, actions, commands, nominal, history_valid_mask=masks
        )[:, 0]
        base_realized = invoke_fada_idm(
            base_idm, spec, observations, actions, commands, realized, history_valid_mask=masks
        )[:, 0]
    else:
        base_nominal = invoke_fada_idm(
            base_idm, spec, observations, actions, None, nominal
        )[:, 0]
        base_realized = invoke_fada_idm(
            base_idm, spec, observations, actions, None, realized
        )[:, 0]
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
        policy: FADAPlannerIDMPolicy | OfficialPlannerIDMPolicy,
        lora_config: FADALoRAConfig,
        *,
        sigma: float,
        lr: float = 1e-4,
        max_grad_norm: float | None = 1.0,
        runtime_spec: FADARuntimeSpec | None = None,
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
        self.runtime_spec = (
            runtime_spec if runtime_spec is not None else get_fada_runtime_spec(self.policy)
        )
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
        spec = self.runtime_spec
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
            if spec.family == "official":
                snapshots = tuple(sample.trajectory.snapshot for sample in samples)
                commands = torch.cat(
                    [snapshot.command for snapshot in snapshots], dim=0
                ).to(reference)
                masks = _trajectory_masks(spec, snapshots, reference)
                predicted = invoke_fada_idm(
                    self.policy.idm,
                    spec,
                    observations,
                    actions,
                    commands,
                    futures,
                    history_valid_mask=masks,
                )[:, 0]
            else:
                predicted = invoke_fada_idm(
                    self.policy.idm, spec, observations, actions, None, futures
                )[:, 0]
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
