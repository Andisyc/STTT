"""Pure-tensor official FADA Planner/IDM learning update.

Ports the tensor-only training semantics of the pinned upstream
``LeCAR-Lab/FADA`` checkout (commit
``OFFICIAL_FADA_SOURCE_REVISION`` = 290b3a9c5c0ebe2e9851b61350617544be05e9bc,
Apache License 2.0) so an official-shaped ``PlannerIDMPolicy`` student can be
trained from collected ``OfficialFADASourceBatch`` windows without importing the
holosoma trainer, replay buffers, or any native simulator stack.

Ported from upstream sources (same commit, Apache-2.0):

- ``src/holosoma/holosoma/fada/planner_idm/trainer.py``
  (``FADATrainer._train_step_separate_pass``: IDM pass first with forced
  teacher-forcing ratio 1.0, then a planner pass that runs with IDM parameters
  frozen while gradients still flow through planner-produced future
  observations; ``requires_grad`` flags are restored even when backward raises).
- ``src/holosoma/holosoma/fada/planner_idm/trainer_batching.py``
  (``_BatchingMixin`` augmentation helpers, per-source IDM loss routing, and
  source weighting by original per-source batch row counts).
- ``src/holosoma/holosoma/fada/planner_idm/loss_utils.py``
  (``weighted_horizon_mse``).
- ``src/holosoma/holosoma/fada/planner_idm/train.py`` (AdamW construction:
  separate optimizers over ``planner_parameters()`` / ``idm_parameters()``
  with ``lr=cfg.lr`` and ``weight_decay=cfg.weight_decay``).

Semantics preserved exactly:

- ``FADATrainer._use_io_normalization()`` is constant-folded to ``False``
  upstream, so no input/output normalization branch exists here either.
- Per-source teacher forcing: real-trajectory sources (``suboptimal``,
  ``online_trajectory``) always run teacher forcing 1.0; oracle-shadow sources
  (``optimal``, ``suboptimal_expert``, ``online``) follow
  ``idm_use_teacher_forcing`` / ``idm_teacher_forcing_ratio``. The IDM pass
  forces ratio 1.0 for every source; the planner pass forces ratio 0.0 with
  ``detach_override=False`` so gradients flow through the planner future obs.
- ``trajectory_target_valid`` filtering applies only inside real-trajectory
  source losses; a source whose rows are all invalid contributes exact zeros
  while its original row count still participates in the source weighting.
- The planner obs loss coefficient defaults to 0.0, but the planner forward
  and every augmentation/RNG-consuming pass are still executed in the official
  order so random-state consumption matches upstream.
- ``idm_action_loss_horizon_gamma`` defaults to 0.0 (first-action-only
  supervision); the weighted-horizon MSE is evaluated exactly as upstream for
  any explicit gamma in ``[0, 1]``.

This module never constructs optimizers for online-LoRA adaptation, never
samples/stores replay data, and never touches checkpoints; it owns only the
single-update learning rule.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from unilab.algos.torch.distill.fada.collection_contract import (
    OFFICIAL_FADA_SOURCE_REVISION,
    OfficialFADASourceBatch,
    validate_official_fada_preprocessing,
)
from unilab.algos.torch.distill.fada.timing import FADAPerformanceRecorder, measure

__all__ = [
    "OfficialFADALearner",
    "canonical_official_training_config",
]

_IDM_SOURCE_KEYS: tuple[str, ...] = (
    "optimal",
    "suboptimal",
    "suboptimal_expert",
    "online",
    "online_trajectory",
)
_TRAJECTORY_TARGET_SOURCES = frozenset({"suboptimal", "online_trajectory"})

# Official fixed model/window dimensions (pinned upstream config defaults for
# the G1 locomotion student: 64-dim compact obs, 29-dim actions, 7-dim
# locomotion command, H=30 history, K=6 prediction horizon).
_OFFICIAL_OBS_DIM = 64
_OFFICIAL_ACT_DIM = 29
_OFFICIAL_CMD_DIM = 7
_OFFICIAL_HISTORY_LEN = 30
_OFFICIAL_PRED_HORIZON = 6

# Canonical compact-observation term order shared with
# ``collection_contract.validate_official_fada_preprocessing``.
_TERM_ORDER: tuple[str, ...] = ("base_ang_vel", "dof_pos", "dof_vel", "projected_gravity")


def _default_augment_obs_noise_overrides() -> dict[str, float]:
    """Upstream ``FADAConfig.augment_obs_noise_overrides`` default (verbatim)."""

    return {
        "base_ang_vel": 0.15,
        "projected_gravity": 0.02,
        "dof_vel": 0.4,
        "dof_pos": 0.02,
    }


def canonical_official_training_config(
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve the training-relevant slice of upstream ``FADAConfig`` defaults.

    Returns a fresh dict on every call. Keys match the pinned upstream
    ``FADAConfig`` field names; defaults are the pinned dataclass defaults
    (revision ``OFFICIAL_FADA_SOURCE_REVISION``), not values inferred from any
    historical checkpoint. Preprocessing magnitudes (``compact_obs_term_scale``
    / ``compact_obs_term_noise``) are intentionally absent: they must come from
    the actual run and are bound separately through ``preprocessing``.

    Source-quota fields are retained verbatim so the caller (sampling owner)
    can derive batch counts with the upstream formulas; this learner itself
    only consumes already-sampled batches. Source enable flags are derived by
    the sampling owner, not serialized as independently configurable fields.

    Unknown keys raise ``ValueError``; no defaults are fabricated for keys the
    caller did not supply and upstream did not define.
    """

    defaults: dict[str, Any] = {
        # Core sequence/window semantics consumed by the loss path.
        "history_len": _OFFICIAL_HISTORY_LEN,
        "pred_horizon": _OFFICIAL_PRED_HORIZON,
        "predict_future_obs": True,
        "planner_predict_delta": True,
        "planner_use_action_history": False,
        "idm_use_teacher_forcing": True,
        "idm_teacher_forcing_ratio": 1.0,
        "idm_detach_planner_future_obs": True,
        "idm_use_current_command_for_history": False,
        # Loss coefficients / horizon weighting.
        "planner_obs_loss_coef": 0.0,
        "idm_action_loss_coef": 1.0,
        "idm_action_loss_horizon_gamma": 0.0,
        "idm_planner_pass_action_loss_coef": 1.0,
        # Optimizer construction (train.py: AdamW over each parameter group).
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "grad_clip": 1.0,
        # Augmentation (always-on training-batch noise/masking upstream).
        "augment_obs_noise": True,
        "augment_obs_noise_overrides": _default_augment_obs_noise_overrides(),
        "augment_action_noise_level": 0.01,
        "augment_history_mask_ratio": 0.2,
        "augment_future_mask_ratio": 0.2,
        # Source-quota fields retained for the sampling owner (main agent);
        # this learner does not sample, but the canonical config carries the
        # exact upstream defaults so quotas are derived from one source of
        # truth.
        "batch_size": 4096,
        "mixed_batch_offline_ratio": 0.2,
        "idm_suboptimal_batch_ratio": 0.375,
        "planner_suboptimal_batch_ratio": 0.375,
        "online_trajectory_batch_ratio": 0.5,
        "suboptimal_expert_batch_ratio": 0.5,
        "warmup_idm_suboptimal_batch_ratio": 0.75,
        "warmup_planner_suboptimal_batch_ratio": 0.75,
        "suboptimal_data_ratio": 1.0,
        "use_generalized_idm": True,
    }
    resolved = dict(defaults)
    if config is not None:
        if not isinstance(config, Mapping):
            raise TypeError("official training config must be a mapping")
        unknown = sorted(set(config) - set(defaults))
        if unknown:
            raise ValueError(f"unknown official training config keys: {unknown}")
        resolved.update(config)

    # Validate the fields this learner consumes, mirroring upstream
    # ``FADAConfig.validate`` ranges for the same names.
    if int(resolved["history_len"]) <= 0:
        raise ValueError("history_len must be positive")
    if int(resolved["pred_horizon"]) <= 0:
        raise ValueError("pred_horizon must be positive")
    if not bool(resolved["predict_future_obs"]):
        raise ValueError("official learner requires predict_future_obs=True")
    if not (0.0 <= float(resolved["idm_teacher_forcing_ratio"]) <= 1.0):
        raise ValueError("idm_teacher_forcing_ratio must be within [0, 1]")
    if float(resolved["planner_obs_loss_coef"]) < 0.0:
        raise ValueError("planner_obs_loss_coef must be non-negative")
    if float(resolved["idm_action_loss_coef"]) < 0.0:
        raise ValueError("idm_action_loss_coef must be non-negative")
    if not (0.0 <= float(resolved["idm_action_loss_horizon_gamma"]) <= 1.0):
        raise ValueError(
            "idm_action_loss_horizon_gamma must be in [0, 1], "
            f"got {resolved['idm_action_loss_horizon_gamma']}"
        )
    if float(resolved["idm_planner_pass_action_loss_coef"]) < 0.0:
        raise ValueError("idm_planner_pass_action_loss_coef must be non-negative")
    if not (0.0 <= float(resolved["augment_history_mask_ratio"]) < 1.0):
        raise ValueError("augment_history_mask_ratio must be in [0, 1)")
    if not (0.0 <= float(resolved["augment_future_mask_ratio"]) < 1.0):
        raise ValueError("augment_future_mask_ratio must be in [0, 1)")
    if float(resolved["augment_action_noise_level"]) < 0.0:
        raise ValueError("augment_action_noise_level must be non-negative")
    if float(resolved["lr"]) <= 0.0:
        raise ValueError("lr must be positive")
    if float(resolved["weight_decay"]) < 0.0:
        raise ValueError("weight_decay must be non-negative")
    if float(resolved["grad_clip"]) < 0.0:
        raise ValueError("grad_clip must be non-negative")
    if int(resolved["batch_size"]) <= 0:
        raise ValueError("batch_size must be positive")
    if not (0.0 < float(resolved["mixed_batch_offline_ratio"]) < 1.0):
        raise ValueError("mixed_batch_offline_ratio must be in (0, 1)")
    for name in (
        "idm_suboptimal_batch_ratio",
        "planner_suboptimal_batch_ratio",
        "warmup_idm_suboptimal_batch_ratio",
        "warmup_planner_suboptimal_batch_ratio",
    ):
        if not (0.0 <= float(resolved[name]) < 1.0):
            raise ValueError(f"{name} must be in [0, 1)")
    for name in ("online_trajectory_batch_ratio", "suboptimal_expert_batch_ratio"):
        if not (0.0 <= float(resolved[name]) <= 1.0):
            raise ValueError(f"{name} must be in [0, 1]")
    if float(resolved["suboptimal_data_ratio"]) < 0.0:
        raise ValueError("suboptimal_data_ratio must be non-negative")
    if resolved["augment_obs_noise_overrides"] is not None:
        if not isinstance(resolved["augment_obs_noise_overrides"], Mapping):
            raise ValueError("augment_obs_noise_overrides must be a mapping or None")
        unknown_terms = sorted(
            set(resolved["augment_obs_noise_overrides"]) - set(_TERM_ORDER)
        )
        if unknown_terms:
            raise ValueError(
                f"augment_obs_noise_overrides has unknown terms: {unknown_terms}"
            )
        resolved["augment_obs_noise_overrides"] = {
            str(term): float(value)
            for term, value in resolved["augment_obs_noise_overrides"].items()
        }
    for name in (
        "predict_future_obs",
        "planner_predict_delta",
        "planner_use_action_history",
        "idm_use_teacher_forcing",
        "idm_detach_planner_future_obs",
        "idm_use_current_command_for_history",
        "augment_obs_noise",
        "use_generalized_idm",
    ):
        resolved[name] = bool(resolved[name])
    for name in ("history_len", "pred_horizon", "batch_size"):
        resolved[name] = int(resolved[name])
    for name in (
        "idm_teacher_forcing_ratio",
        "planner_obs_loss_coef",
        "idm_action_loss_coef",
        "idm_action_loss_horizon_gamma",
        "idm_planner_pass_action_loss_coef",
        "lr",
        "weight_decay",
        "grad_clip",
        "augment_action_noise_level",
        "augment_history_mask_ratio",
        "augment_future_mask_ratio",
        "mixed_batch_offline_ratio",
        "idm_suboptimal_batch_ratio",
        "planner_suboptimal_batch_ratio",
        "online_trajectory_batch_ratio",
        "suboptimal_expert_batch_ratio",
        "warmup_idm_suboptimal_batch_ratio",
        "warmup_planner_suboptimal_batch_ratio",
        "suboptimal_data_ratio",
    ):
        resolved[name] = float(resolved[name])

    return resolved


def _weighted_horizon_mse(pred: torch.Tensor, target: torch.Tensor, gamma: float) -> torch.Tensor:
    """Upstream ``weighted_horizon_mse`` (loss_utils.py, verbatim semantics).

    w_t = gamma^t for t = 0..K-1; loss = sum_t(w_t * mse_t) / sum_t(w_t) with
    mse_t averaged over batch and feature dims. ``gamma == 1.0`` degenerates to
    ``F.mse_loss`` exactly; ``gamma == 0.0`` supervises the first step only.
    """

    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    if gamma == 1.0:
        return F.mse_loss(pred, target)
    horizon = pred.shape[1]
    steps = torch.arange(horizon, device=pred.device, dtype=pred.dtype)
    weights = gamma**steps
    mse_per_step = ((pred - target) ** 2).mean(dim=(0, 2))
    return (weights * mse_per_step).sum() / weights.sum()


def _batch_to_mapping(batch: OfficialFADASourceBatch, source_name: str) -> None:
    if not isinstance(batch, OfficialFADASourceBatch):
        raise TypeError(
            f"official learner expected OfficialFADASourceBatch for {source_name!r}, "
            f"got {type(batch)}"
        )


class OfficialFADALearner:
    """Single-step official Planner/IDM update over pre-sampled tensor batches.

    Mirrors ``FADATrainer._train_step_separate_pass``: first the IDM pass over
    ``idm_batches_by_source`` with forced teacher forcing 1.0, then the planner
    pass over ``planner_batch`` with the IDM parameters frozen (restored even
    on error) while the IDM action loss backpropagates through the planner's
    future-observation outputs.

    The supplied ``policy`` must follow the upstream ``PlannerIDMPolicy``
    protocol: a ``planner`` submodule callable as
    ``planner(history_obs, current_command, history_action=...,
    history_valid_mask=...)``, ``predict_idm_actions(history_obs, history_act,
    current_command_or_none, future_obs, history_valid_mask=...)``,
    ``planner_parameters()`` / ``idm_parameters()``, and the boolean attributes
    ``planner_use_action_history`` / ``idm_use_current_command_for_history``.
    """

    def __init__(
        self,
        policy: nn.Module,
        *,
        training_config: Mapping[str, Any] | None = None,
        preprocessing: Mapping[str, Any],
        performance: FADAPerformanceRecorder | None = None,
    ) -> None:
        if not isinstance(policy, nn.Module):
            raise TypeError(f"policy must be an nn.Module, got {type(policy)}")
        self._policy = policy
        # Opt-in performance instrumentation; never serialized into the
        # canonical training config or checkpoints.
        self._performance = performance
        overrides = dict(training_config or {})
        model_fields = {
            "history_len": policy.planner.history_len,
            "pred_horizon": policy.planner.pred_horizon,
            "predict_future_obs": policy.predict_future_obs,
            "planner_predict_delta": policy.planner_predict_delta,
            "planner_use_action_history": policy.planner_use_action_history,
            "idm_use_current_command_for_history": policy.idm_use_current_command_for_history,
        }
        for name, value in model_fields.items():
            if name in overrides and overrides[name] != value:
                raise ValueError(f"training {name} contradicts the official student model")
            overrides[name] = value
        self._training_config = canonical_official_training_config(overrides)
        # Binds the actual run preprocessing (term scales + augment noise
        # magnitudes) exactly once; the validator rejects fabricated defaults
        # for missing or non-canonical payloads.
        self._preprocessing = validate_official_fada_preprocessing(preprocessing)
        self._update_count = 0

        idm_params = list(policy.idm_parameters())
        planner_params = list(policy.planner_parameters())
        if not idm_params:
            raise ValueError("policy exposes no IDM parameters")
        if not planner_params:
            raise ValueError("policy exposes no planner parameters")
        lr = float(self._training_config["lr"])
        weight_decay = float(self._training_config["weight_decay"])
        # Upstream train.py: two independent AdamW optimizers, one per group;
        # no online-LoRA optimizer is created here.
        self._idm_optimizer = torch.optim.AdamW(idm_params, lr=lr, weight_decay=weight_decay)
        self._planner_optimizer = torch.optim.AdamW(
            planner_params, lr=lr, weight_decay=weight_decay
        )

    @property
    def policy(self) -> nn.Module:
        return self._policy

    @property
    def training_config(self) -> dict[str, Any]:
        return dict(self._training_config)

    @property
    def preprocessing(self) -> dict[str, Any]:
        return dict(self._preprocessing)

    @property
    def idm_optimizer(self) -> torch.optim.AdamW:
        return self._idm_optimizer

    @property
    def planner_optimizer(self) -> torch.optim.AdamW:
        return self._planner_optimizer

    @property
    def update_count(self) -> int:
        return self._update_count

    @update_count.setter
    def update_count(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("official update_count must be a non-negative integer")
        self._update_count = value

    # ------------------------------------------------------------------
    # Tensor coercion / validation
    # ------------------------------------------------------------------
    def _policy_device(self) -> torch.device:
        return next(self._policy.planner_parameters()).device

    def _coerce_batch(
        self, batch: OfficialFADASourceBatch, source_name: str
    ) -> dict[str, torch.Tensor]:
        _batch_to_mapping(batch, source_name)
        device = self._policy_device()
        out: dict[str, torch.Tensor] = {}
        for name in (
            "history_obs",
            "history_act",
            "current_command",
            "target_actions",
            "target_next_observations",
            "trajectory_target_actions",
            "trajectory_target_next_observations",
        ):
            tensor = getattr(batch, name)
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{source_name}.{name} must be a torch.Tensor")
            out[name] = tensor.to(device=device, dtype=torch.float32)
        mask = batch.history_valid_mask
        valid = batch.trajectory_target_valid
        if not isinstance(mask, torch.Tensor) or not isinstance(valid, torch.Tensor):
            raise TypeError(f"{source_name} masks must be torch.Tensor")
        out["history_valid_mask"] = mask.to(device=device, dtype=torch.bool)
        out["trajectory_target_valid"] = valid.to(device=device, dtype=torch.bool)
        if int(out["history_obs"].shape[0]) <= 0:
            raise ValueError(f"{source_name} batch must contain at least one row")
        return out

    # ------------------------------------------------------------------
    # Augmentation helpers (upstream _BatchingMixin, same RNG consumption order)
    # ------------------------------------------------------------------
    def _obs_term_slices(self) -> list[tuple[str, int, int]]:
        obs_dim = _OFFICIAL_OBS_DIM
        ndof = (obs_dim - 6) // 2  # base_ang_vel=3, projected_gravity=3
        slices: list[tuple[str, int, int]] = []
        idx = 0
        for term_name in _TERM_ORDER:
            if term_name == "base_ang_vel":
                dim = 3
            elif term_name == "dof_pos":
                dim = ndof
            elif term_name == "dof_vel":
                dim = ndof
            elif term_name == "projected_gravity":
                dim = 3
            else:  # pragma: no cover - canonical order is fixed
                raise ValueError(f"Unknown compact obs term: {term_name}")
            slices.append((term_name, idx, idx + dim))
            idx += dim
        return slices

    def _build_augment_noise_scales(self, device: torch.device) -> torch.Tensor | None:
        if not bool(self._training_config["augment_obs_noise"]):
            return None
        # Training-batch augmentation noise magnitudes come from the actual run
        # preprocessing (bound through validate_official_fada_preprocessing),
        # with upstream per-term augment overrides applied on top.
        term_noise = dict(self._preprocessing["term_noise"])
        overrides = self._training_config["augment_obs_noise_overrides"]
        if overrides:
            for term, value in overrides.items():
                term_noise[term] = float(value)
        term_scale = self._preprocessing["term_scale"]
        noise_vec = torch.zeros(_OFFICIAL_OBS_DIM, dtype=torch.float32, device=device)
        for term_name, start, end in self._obs_term_slices():
            noise_vec[start:end] = float(term_noise[term_name]) * float(term_scale[term_name])
        return noise_vec

    def _apply_obs_noise(self, obs: torch.Tensor) -> torch.Tensor:
        noise_scales = self._build_augment_noise_scales(obs.device)
        if noise_scales is None:
            return obs
        shape = obs.shape
        if obs.ndim == 3:
            noise_scales = noise_scales.view(1, 1, -1)
        elif obs.ndim == 2:
            noise_scales = noise_scales.view(1, -1)
        noise = (torch.rand(shape, device=obs.device, dtype=obs.dtype) * 2.0 - 1.0) * noise_scales
        return obs + noise

    def _apply_history_mask(
        self,
        history_obs: torch.Tensor,
        history_act: torch.Tensor,
        history_valid_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        ratio = float(self._training_config["augment_history_mask_ratio"])
        if ratio <= 0.0:
            return history_obs, history_act, history_valid_mask
        batch, steps, _ = history_obs.shape
        mask_candidates = steps - 1  # never mask the last (current) step
        if mask_candidates <= 0:
            return history_obs, history_act, history_valid_mask
        drop_mask = torch.rand((batch, mask_candidates), device=history_obs.device) < ratio
        full_drop = torch.cat(
            [
                drop_mask,
                torch.zeros((batch, 1), device=history_obs.device, dtype=torch.bool),
            ],
            dim=1,
        )
        history_obs = history_obs.clone()
        history_act = history_act.clone()
        history_obs[full_drop] = 0.0
        history_act[full_drop] = 0.0
        if history_valid_mask is not None:
            history_valid_mask = history_valid_mask.clone()
            history_valid_mask[full_drop] = False
        return history_obs, history_act, history_valid_mask

    def _apply_future_mask(self, future: torch.Tensor) -> torch.Tensor:
        ratio = float(self._training_config["augment_future_mask_ratio"])
        if ratio <= 0.0:
            return future
        batch, steps, _ = future.shape
        drop_mask = torch.rand((batch, steps), device=future.device) < ratio
        future = future.clone()
        future[drop_mask] = 0.0
        return future

    def _apply_action_noise(self, actions: torch.Tensor) -> torch.Tensor:
        level = float(self._training_config["augment_action_noise_level"])
        if level <= 0.0:
            return actions
        noise = (torch.rand_like(actions) * 2.0 - 1.0) * level
        return actions + noise

    def _to_delta_obs(self, target_obs: torch.Tensor, ref_obs: torch.Tensor) -> torch.Tensor:
        return target_obs - ref_obs.unsqueeze(1)

    def _from_delta_obs(self, delta_obs: torch.Tensor, ref_obs: torch.Tensor) -> torch.Tensor:
        return delta_obs + ref_obs.unsqueeze(1)

    def _compute_per_term_mse(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> dict[str, float]:
        results: dict[str, float] = {}
        for term_name, start, end in self._obs_term_slices():
            term_loss = F.mse_loss(pred[..., start:end], target[..., start:end])
            results[term_name] = float(term_loss.detach().cpu().item())
        return results

    # ------------------------------------------------------------------
    # Planner loss (upstream _compute_planner_batch_loss_from_tensors)
    # ------------------------------------------------------------------
    def _compute_planner_loss(
        self,
        *,
        history_obs: torch.Tensor,
        history_act: torch.Tensor,
        current_command: torch.Tensor,
        target_next_obs: torch.Tensor,
        history_valid_mask: torch.Tensor | None,
        augment: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        # Save the clean reference obs for delta conversion before any noise.
        ref_obs = history_obs[:, -1, :]
        if augment:
            history_obs = self._apply_obs_noise(history_obs)
            history_act = self._apply_action_noise(history_act)
            history_obs, history_act, history_valid_mask = self._apply_history_mask(
                history_obs, history_act, history_valid_mask
            )
        target_obs_loss_input = target_next_obs
        if bool(self._training_config["planner_predict_delta"]):
            # No IO normalization upstream (constant-folded False), so the
            # reference obs is used as-is.
            target_obs_loss_input = self._to_delta_obs(target_obs_loss_input, ref_obs)
        pred_future_obs = self._policy.planner(
            history_obs,
            current_command,
            history_action=(
                history_act if self._policy.planner_use_action_history else None
            ),
            history_valid_mask=history_valid_mask,
        )
        planner_loss = F.mse_loss(pred_future_obs, target_obs_loss_input)
        per_term = self._compute_per_term_mse(pred_future_obs, target_obs_loss_input)
        weighted_total = float(self._training_config["planner_obs_loss_coef"]) * planner_loss
        return weighted_total, planner_loss, per_term

    # ------------------------------------------------------------------
    # IDM loss (upstream _resolve_idm_future_obs + _compute_idm_batch_loss_from_tensors)
    # ------------------------------------------------------------------
    def _resolve_idm_future_obs(
        self,
        *,
        planner_future_obs: torch.Tensor,
        teacher_future_obs: torch.Tensor,
        ref_obs: torch.Tensor | None = None,
        use_teacher_forcing: bool | None = None,
        teacher_forcing_ratio: float | None = None,
        detach_override: bool | None = None,
    ) -> torch.Tensor:
        use_tf = (
            bool(self._training_config["idm_use_teacher_forcing"])
            if use_teacher_forcing is None
            else bool(use_teacher_forcing)
        )
        teacher_ratio = (
            float(self._training_config["idm_teacher_forcing_ratio"])
            if teacher_forcing_ratio is None
            else float(teacher_forcing_ratio)
        )
        if not use_tf:
            teacher_ratio = 0.0
        should_detach = (
            bool(self._training_config["idm_detach_planner_future_obs"])
            if detach_override is None
            else bool(detach_override)
        )
        planner_source = planner_future_obs.detach() if should_detach else planner_future_obs
        if bool(self._training_config["planner_predict_delta"]) and ref_obs is not None:
            planner_source = self._from_delta_obs(planner_source, ref_obs)
        if teacher_ratio <= 0.0:
            return planner_source
        if teacher_ratio >= 1.0:
            return teacher_future_obs
        batch_size = int(planner_future_obs.shape[0])
        mix_mask = (
            torch.rand(
                (batch_size, 1, 1),
                device=planner_future_obs.device,
                dtype=planner_future_obs.dtype,
            )
            < teacher_ratio
        ).to(dtype=planner_future_obs.dtype)
        return mix_mask * teacher_future_obs + (1.0 - mix_mask) * planner_source

    def _compute_idm_loss_from_tensors(
        self,
        *,
        history_obs: torch.Tensor,
        history_act: torch.Tensor,
        current_command: torch.Tensor,
        target_actions: torch.Tensor,
        target_next_obs: torch.Tensor,
        history_valid_mask: torch.Tensor | None,
        use_teacher_forcing: bool | None = None,
        teacher_forcing_ratio: float | None = None,
        detach_override: bool | None = None,
        augment: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Save the clean reference obs for delta reconstruction before noise.
        ref_obs = history_obs[:, -1, :]
        if augment:
            history_obs = self._apply_obs_noise(history_obs)
            history_act = self._apply_action_noise(history_act)
            history_obs, history_act, history_valid_mask = self._apply_history_mask(
                history_obs, history_act, history_valid_mask
            )

        use_tf = (
            bool(self._training_config["idm_use_teacher_forcing"])
            if use_teacher_forcing is None
            else bool(use_teacher_forcing)
        )
        teacher_ratio = (
            float(self._training_config["idm_teacher_forcing_ratio"])
            if teacher_forcing_ratio is None
            else float(teacher_forcing_ratio)
        )
        if not use_tf:
            teacher_ratio = 0.0

        if teacher_ratio >= 1.0:
            # Pure teacher forcing: teacher obs used directly, planner forward
            # skipped entirely (upstream early-exit; no RNG consumed there).
            idm_future_obs = target_next_obs
        else:
            ref_obs_for_delta: torch.Tensor | None = None
            if bool(self._training_config["planner_predict_delta"]):
                ref_obs_for_delta = ref_obs
            planner_future_obs = self._policy.planner(
                history_obs,
                current_command,
                history_action=(
                    history_act if self._policy.planner_use_action_history else None
                ),
                history_valid_mask=history_valid_mask,
            )
            idm_future_obs = self._resolve_idm_future_obs(
                planner_future_obs=planner_future_obs,
                teacher_future_obs=target_next_obs,
                ref_obs=ref_obs_for_delta,
                use_teacher_forcing=use_teacher_forcing,
                teacher_forcing_ratio=teacher_forcing_ratio,
                detach_override=detach_override,
            )
        if augment:
            idm_future_obs = self._apply_obs_noise(idm_future_obs)
            idm_future_obs = self._apply_future_mask(idm_future_obs)
        idm_current_command = self._policy._resolve_idm_current_command(
            current_command,
            idm_use_current_command_for_history=self._policy.idm_use_current_command_for_history,
        )
        pred_actions = self._policy.predict_idm_actions(
            history_obs,
            history_act,
            idm_current_command,
            idm_future_obs,
            history_valid_mask=history_valid_mask,
        )
        target_actions_loss_input = target_actions
        if pred_actions.shape[1] != target_actions_loss_input.shape[1]:
            if pred_actions.shape[1] != 1:
                raise RuntimeError(
                    f"IDM pred_actions shape {tuple(pred_actions.shape)} incompatible with "
                    f"target shape {tuple(target_actions_loss_input.shape)}"
                )
            target_actions_loss_input = target_actions_loss_input[:, :1, :]
        idm_loss_unweighted = F.mse_loss(pred_actions, target_actions_loss_input)
        gamma = float(self._training_config["idm_action_loss_horizon_gamma"])
        idm_loss = _weighted_horizon_mse(pred_actions, target_actions_loss_input, gamma)
        weighted_total = float(self._training_config["idm_action_loss_coef"]) * idm_loss
        return weighted_total, idm_loss, idm_loss_unweighted

    def _compute_idm_loss_from_batch(
        self,
        batch: dict[str, torch.Tensor],
        *,
        use_teacher_forcing: bool | None = None,
        teacher_forcing_ratio: float | None = None,
        use_trajectory_targets: bool = False,
        detach_override: bool | None = None,
        augment: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history_obs = batch["history_obs"]
        history_act = batch["history_act"]
        current_command = batch["current_command"]
        target_actions = batch["target_actions"]
        target_next_obs = batch["target_next_observations"]
        history_valid_mask = batch["history_valid_mask"]
        if use_trajectory_targets:
            # Real-trajectory sources route executed-action / real-next-obs
            # fields and drop rows whose trajectory tail was zero-filled; this
            # filtering never touches the oracle-shadow expert targets.
            target_actions = batch["trajectory_target_actions"]
            target_next_obs = batch["trajectory_target_next_observations"]
            valid_mask = batch["trajectory_target_valid"]
            if not bool(valid_mask.all()):
                valid_idx = valid_mask.nonzero(as_tuple=True)[0]
                if valid_idx.numel() == 0:
                    zero = torch.zeros((), device=history_obs.device, dtype=torch.float32)
                    return zero, zero, zero
                history_obs = history_obs[valid_idx]
                history_act = history_act[valid_idx]
                current_command = current_command[valid_idx]
                target_actions = target_actions[valid_idx]
                target_next_obs = target_next_obs[valid_idx]
                history_valid_mask = history_valid_mask[valid_idx]
        return self._compute_idm_loss_from_tensors(
            history_obs=history_obs,
            history_act=history_act,
            current_command=current_command,
            target_actions=target_actions,
            target_next_obs=target_next_obs,
            history_valid_mask=history_valid_mask,
            use_teacher_forcing=use_teacher_forcing,
            teacher_forcing_ratio=teacher_forcing_ratio,
            detach_override=detach_override,
            augment=augment,
        )

    def _idm_source_uses_teacher_forcing(self, source_name: str) -> bool:
        if source_name in _TRAJECTORY_TARGET_SOURCES:
            return True
        return bool(self._training_config["idm_use_teacher_forcing"])

    def _compute_idm_loss_from_source_batches(
        self,
        batches_by_source: Mapping[str, dict[str, torch.Tensor]],
        *,
        force_teacher_forcing_ratio: float | None = None,
        detach_override: bool | None = None,
        augment: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weighted_losses: list[torch.Tensor] = []
        idm_losses: list[torch.Tensor] = []
        idm_losses_unweighted: list[torch.Tensor] = []
        batch_sizes: list[int] = []
        for source_name in _IDM_SOURCE_KEYS:
            batch = batches_by_source.get(source_name)
            if batch is None:
                continue
            if force_teacher_forcing_ratio is not None:
                tf_ratio = float(force_teacher_forcing_ratio)
                use_tf = tf_ratio > 0.0
            else:
                tf_ratio = 1.0 if source_name in _TRAJECTORY_TARGET_SOURCES else None
                use_tf = self._idm_source_uses_teacher_forcing(source_name)
            loss_total, idm_loss, idm_loss_unweighted = self._compute_idm_loss_from_batch(
                batch,
                use_teacher_forcing=use_tf,
                use_trajectory_targets=(source_name in _TRAJECTORY_TARGET_SOURCES),
                teacher_forcing_ratio=tf_ratio,
                detach_override=detach_override,
                augment=augment,
            )
            weighted_losses.append(loss_total)
            idm_losses.append(idm_loss)
            idm_losses_unweighted.append(idm_loss_unweighted)
            # Upstream weights each source by its ORIGINAL batch row count,
            # regardless of how many rows survive trajectory-validity filtering.
            batch_sizes.append(int(batch["history_obs"].shape[0]))
        if not weighted_losses:
            raise RuntimeError("No IDM source batches were provided for loss computation")
        total_batch = int(sum(batch_sizes))
        device = weighted_losses[0].device
        dtype = weighted_losses[0].dtype
        weights = [
            torch.as_tensor(size / total_batch, device=device, dtype=dtype)
            for size in batch_sizes
        ]
        loss_total = sum(
            weight * loss for weight, loss in zip(weights, weighted_losses, strict=False)
        )
        idm_loss = sum(
            weight * loss for weight, loss in zip(weights, idm_losses, strict=False)
        )
        idm_loss_unweighted = sum(
            weight * loss
            for weight, loss in zip(weights, idm_losses_unweighted, strict=False)
        )
        return loss_total, idm_loss, idm_loss_unweighted

    # ------------------------------------------------------------------
    # Public update (upstream FADATrainer._train_step_separate_pass)
    # ------------------------------------------------------------------
    def update(
        self,
        idm_batches_by_source: Mapping[str, OfficialFADASourceBatch],
        planner_batch: OfficialFADASourceBatch,
    ) -> dict[str, float]:
        if not isinstance(idm_batches_by_source, Mapping) or not idm_batches_by_source:
            raise ValueError("idm_batches_by_source must be a non-empty mapping")
        unknown_sources = sorted(set(idm_batches_by_source) - set(_IDM_SOURCE_KEYS))
        if unknown_sources:
            raise ValueError(f"unknown IDM source keys: {unknown_sources}")

        performance = self._performance

        with measure(
            performance, "learner_batch_staging", group="learner_coerce"
        ):
            idm_batches: dict[str, dict[str, torch.Tensor]] = {
                source: self._coerce_batch(batch, source)
                for source, batch in idm_batches_by_source.items()
            }
            planner_tensors = self._coerce_batch(planner_batch, "planner")

        self._policy.train()
        idm_params = list(self._policy.idm_parameters())
        grad_clip = float(self._training_config["grad_clip"])

        # ── IDM pass (first): all sources, forced TF=1.0 ────────────────────
        with measure(performance, "learner_forward", group="idm", cuda=True):
            idm_loss_total, idm_loss, idm_loss_unweighted = (
                self._compute_idm_loss_from_source_batches(
                    idm_batches, force_teacher_forcing_ratio=1.0
                )
            )
        with measure(performance, "learner_backward", group="idm", cuda=True):
            self._idm_optimizer.zero_grad(set_to_none=True)
            idm_loss_total.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(idm_params, grad_clip)
        with measure(performance, "optimizer_step", group="idm", cuda=True):
            self._idm_optimizer.step()

        # ── Planner pass (second): obs loss + IDM action loss through inputs ──
        # Freeze IDM parameters so autograd skips their gradient computation;
        # gradients still reach the planner through the future-obs inputs.
        prev_requires_grad = [param.requires_grad for param in idm_params]
        for param in idm_params:
            param.requires_grad_(False)
        try:
            with measure(performance, "learner_forward", group="planner", cuda=True):
                planner_loss_total, planner_loss, planner_per_term = self._compute_planner_loss(
                    history_obs=planner_tensors["history_obs"],
                    history_act=planner_tensors["history_act"],
                    current_command=planner_tensors["current_command"],
                    target_next_obs=planner_tensors["target_next_observations"],
                    history_valid_mask=planner_tensors["history_valid_mask"],
                )
                _, idm_planner_loss, idm_planner_loss_uw = self._compute_idm_loss_from_batch(
                    planner_tensors,
                    use_teacher_forcing=False,
                    teacher_forcing_ratio=0.0,
                    use_trajectory_targets=False,
                    detach_override=False,
                )
                coef = float(self._training_config["idm_planner_pass_action_loss_coef"])
                idm_planner_loss_total_weighted = (
                    coef * float(self._training_config["idm_action_loss_coef"]) * idm_planner_loss
                )
                total_planner_loss = planner_loss_total + idm_planner_loss_total_weighted
            with measure(performance, "learner_backward", group="planner", cuda=True):
                self._planner_optimizer.zero_grad(set_to_none=True)
                try:
                    total_planner_loss.backward()
                finally:
                    for param, prev in zip(idm_params, prev_requires_grad):
                        param.requires_grad_(prev)
        finally:
            # Guarantee restoration even if the loss construction itself raises.
            for param, prev in zip(idm_params, prev_requires_grad):
                if param.requires_grad != prev:
                    param.requires_grad_(prev)
        with measure(performance, "optimizer_step", group="planner", cuda=True):
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(list(self._policy.planner_parameters()), grad_clip)
            self._planner_optimizer.step()

        self._update_count += 1
        if performance is not None:
            performance.increment("learner_updates")

        loss_total_combined = total_planner_loss + idm_loss_total
        metrics: dict[str, float] = {
            "loss_total": float(loss_total_combined.detach().cpu().item()),
            "planner_loss_total": float(planner_loss_total.detach().cpu().item()),
            "planner_future_obs_loss": float(planner_loss.detach().cpu().item()),
            "planner_future_obs_loss_weighted": float(
                float(self._training_config["planner_obs_loss_coef"])
                * planner_loss.detach().cpu().item()
            ),
            "idm_loss_total": float(idm_loss_total.detach().cpu().item()),
            "idm_action_loss": float(idm_loss.detach().cpu().item()),
            "idm_action_loss_unweighted": float(idm_loss_unweighted.detach().cpu().item()),
            "idm_action_loss_weighted": float(
                float(self._training_config["idm_action_loss_coef"])
                * idm_loss.detach().cpu().item()
            ),
            "idm_planner_pass_action_loss": float(idm_planner_loss.detach().cpu().item()),
            "idm_planner_pass_action_loss_unweighted": float(
                idm_planner_loss_uw.detach().cpu().item()
            ),
            "idm_planner_pass_action_loss_weighted": float(
                idm_planner_loss_total_weighted.detach().cpu().item()
            ),
        }
        for term_name, term_loss in planner_per_term.items():
            metrics[f"planner_term/{term_name}"] = float(term_loss)
        return metrics
