"""Teacher-filtered reconstruction and real-action, open-prior trajectory losses."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F

from .model import LatentWorldModel
from .motion import window_motion
from .reference import categorical_probs
from .types import LatentState

if TYPE_CHECKING:
    from .pretraining_data import PretrainingBatch


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(value)
    return torch.where(mask, value, 0.).sum() / mask.sum().clamp_min(1)


def _categorical_kl(target: torch.Tensor, prediction: torch.Tensor, unimix: float) -> torch.Tensor:
    target_prob = categorical_probs(target, unimix)
    prediction_prob = categorical_probs(prediction, unimix)
    tiny = torch.finfo(target_prob.dtype).tiny
    return (target_prob * (target_prob.clamp_min(tiny).log() - prediction_prob.clamp_min(tiny).log())).sum((-1, -2))


def _motion_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # These are motion increments, not interchangeable absolute orientations.
    return (prediction - target.detach()).square().mean(-1)


def world_model_loss(
    model: LatentWorldModel, batch: Mapping[str, torch.Tensor], *, free_nats: float = 1.,
    kl_scale: float = 1., reconstruction_scale: float = 1., motion_scale: float = 1.,
    risk_scale: float = 1., overshoot_scale: float = 1., kl_balance: float = .8,
    prior_reconstruction_scale: float = 1., window_scale: float = 1.,
    initial_state: LatentState | None = None,
    initial_previous_action: torch.Tensor | None = None,
    observation_mask: torch.Tensor | None = None,
    transition_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Fit logged ENV actions, with step-local motion and composed window targets."""
    weights = (free_nats, kl_scale, reconstruction_scale, motion_scale, risk_scale, overshoot_scale, prior_reconstruction_scale, window_scale)
    if any(not math.isfinite(value) or value < 0 for value in weights) or not 0 <= kl_balance <= 1:
        raise ValueError("loss weights must be finite nonnegative and kl_balance in [0,1]")
    observations, actions = batch["observations"], batch["actions"]
    if observations.ndim != 3 or actions.ndim != 3:
        raise ValueError("world-model loss requires batched sequences")
    size, steps = actions.shape[:2]
    if steps < 1 or observations.shape != (size, steps + 1, model.config.obs_dim) or actions.shape[-1] != model.config.action_dim:
        raise ValueError("actions must align with observation-to-next-observation transitions")
    first, terminal = batch["is_first"].bool(), batch["terminal"].bool()
    if first.shape != (size, steps + 1) or terminal.shape != (size, steps):
        raise ValueError("invalid reset/terminal masks")
    observation_mask = torch.ones_like(first) if observation_mask is None else observation_mask
    transition_mask = torch.ones_like(terminal) if transition_mask is None else transition_mask
    for name, value, shape in (
        ("observation_mask", observation_mask, (size, steps + 1)),
        ("transition_mask", transition_mask, (size, steps)),
    ):
        if value.shape != shape or value.dtype != torch.bool or value.device != actions.device:
            raise ValueError(f"{name} must be boolean {shape} on the sequence device")
    transition_valid = transition_mask & ~first[:, 1:] & ~terminal
    observation_valid = observation_mask & torch.cat((torch.ones_like(first[:, :1]), ~terminal | first[:, 1:]), 1)
    posterior_states: list[LatentState] = []
    priors, posteriors, reconstructions = [], [], []
    latent = model.initial(size, device=observations.device, dtype=observations.dtype) if initial_state is None else initial_state.detach()
    if initial_state is not None and initial_previous_action is None:
        raise ValueError("a warmed history requires its actual action preceding the origin")
    previous_action = actions.new_zeros(size, model.config.action_dim) if initial_previous_action is None else initial_previous_action
    if previous_action.shape != (size, model.config.action_dim):
        raise ValueError("origin previous action must be [batch,29]")
    for index in range(steps + 1):
        kwargs = {key: batch[key][:, index] for key in ("ego_motion", "ego_valid", "ego_age_s") if key in batch}
        reset = first[:, index] if index or initial_state is not None else torch.ones_like(first[:, 0])
        if index:
            reset = reset | terminal[:, index - 1]
        observation = torch.where(observation_mask[:, index, None], observations[:, index], 0.)
        latent, prior_logits, posterior_logits = model.posterior(
            latent, previous_action if index == 0 else actions[:, index - 1],
            observation, is_first=reset, **kwargs,
        )
        posterior_states.append(latent)
        priors.append(prior_logits)
        posteriors.append(posterior_logits)
        reconstructions.append(model.decode(latent))
    prior_logits = torch.stack(priors, 1)
    posterior_logits = torch.stack(posteriors, 1)
    reconstruction = _masked_mean((torch.stack(reconstructions, 1) - observations.detach()).square().mean(-1), observation_valid)
    dynamics_kl = _categorical_kl(posterior_logits.detach(), prior_logits, model.config.unimix)
    representation_kl = _categorical_kl(posterior_logits, prior_logits.detach(), model.config.unimix)
    kl = _masked_mean(kl_balance * dynamics_kl.clamp_min(free_nats) + (1 - kl_balance) * representation_kl.clamp_min(free_nats), observation_valid)

    latent = posterior_states[0]
    predicted_motion, predicted_risk, predicted_observations, open_logits = [], [], [], []
    for index in range(steps):
        # A reset starts a new independent segment, never a bridge through it.
        if index:
            reset = first[:, index] | terminal[:, index - 1]
            origin = posterior_states[index]
            latent = LatentState(torch.where(reset[:, None, None], origin.state, latent.state), torch.where(reset[:, None], origin.belief, latent.belief))
        motion = model.motion(latent, actions[:, index])
        predicted_motion.append(motion)
        predicted_risk.append(model.risk(latent, actions[:, index]))
        latent, logits = model.prior(latent, actions[:, index])
        open_logits.append(logits)
        predicted_observations.append(model.decode(latent))
    motion_prediction = torch.stack(predicted_motion, 1)
    prior_reconstruction = _masked_mean((torch.stack(predicted_observations, 1) - observations[:, 1:].detach()).square().mean(-1), transition_valid)
    overshoot = _masked_mean(_categorical_kl(posterior_logits[:, 1:].detach(), torch.stack(open_logits, 1), model.config.unimix).clamp_min(free_nats), transition_valid)
    motion_mask = batch["motion_valid"].bool() & transition_valid
    motion_target = torch.where(motion_mask[..., None], batch["path_delta"], 0.)
    motion_loss = _masked_mean(_motion_error(motion_prediction, motion_target), motion_mask)

    start, end = batch["window_start"].long(), batch["window_end"].long()
    if start.shape != end.shape or start.ndim != 2 or start.shape[0] != size:
        raise ValueError("window endpoints must be matched [batch,windows] indices")
    label_valid = batch["window_valid"].bool()
    bounds = (start >= 0) & (end > start) & (end <= steps)
    if bool((label_valid & ~bounds).any()):
        raise ValueError("valid window endpoints must lie inside the observation sequence")
    safe_start, safe_end = start.clamp(0, steps), end.clamp(0, steps)
    window_prediction = window_motion(motion_prediction, safe_start, safe_end)
    invalid_prefix = torch.cat((torch.zeros(size, 1, device=actions.device, dtype=torch.long), (~transition_valid).long().cumsum(1)), 1)
    crosses_reset = invalid_prefix.gather(1, safe_end) != invalid_prefix.gather(1, safe_start)
    window_mask = label_valid & bounds & ~crosses_reset
    window_target = torch.where(window_mask[..., None], batch["window_delta"], 0.)
    # Do not let a full predicted revolution disappear at the window boundary.
    predicted_yaw = F.pad(motion_prediction[..., 2].cumsum(1), (1, 0))
    predicted_turn = predicted_yaw.gather(1, safe_end) - predicted_yaw.gather(1, safe_start)
    target_yaw = F.pad(motion_target[..., 2].cumsum(1), (1, 0))
    target_turn = target_yaw.gather(1, safe_end) - target_yaw.gather(1, safe_start)
    missing_motion = F.pad((~motion_mask).long().cumsum(1), (1, 0))
    dense_window = missing_motion.gather(1, safe_end) == missing_motion.gather(1, safe_start)
    # Dense labels identify genuine turns beyond pi. Sparse short-window labels
    # retain their measured principal-angle target, never the prediction's winding.
    target_turn = torch.where(dense_window, target_turn, window_target[..., 2])
    continuous_prediction = torch.cat((window_prediction[..., :2], predicted_turn[..., None]), -1)
    continuous_target = torch.cat((window_target[..., :2], target_turn[..., None]), -1)
    window_loss = _masked_mean(_motion_error(continuous_prediction, continuous_target), window_mask)
    window_xy_error = (window_prediction[..., :2] - window_target[..., :2]).square().sum(-1)
    window_xy_sum = torch.where(window_mask, window_xy_error, 0.).sum()
    risk_target = batch["risk"].detach().to(observations)
    risk_mask = batch["risk_valid"].bool() & transition_mask
    # Terminal transitions remain valid risk labels even without usable next proprio.
    risk_loss = _masked_mean(F.binary_cross_entropy_with_logits(torch.stack(predicted_risk, 1), torch.where(risk_mask[..., None], risk_target, 0.), reduction="none"), risk_mask)
    total = reconstruction_scale * reconstruction + kl_scale * kl + prior_reconstruction_scale * prior_reconstruction + overshoot_scale * overshoot + motion_scale * (motion_loss + window_scale * window_loss) + risk_scale * risk_loss
    return {
        "loss": total, "reconstruction_loss": reconstruction, "prior_reconstruction_loss": prior_reconstruction,
        "kl_loss": kl, "dynamics_kl": _masked_mean(dynamics_kl, observation_valid),
        "representation_kl": _masked_mean(representation_kl, observation_valid), "overshoot_loss": overshoot,
        "motion_loss": motion_loss, "window_loss": window_loss, "risk_loss": risk_loss,
        "window_xy_squared_error_sum": window_xy_sum, "window_xy_count": window_mask.sum(),
    }


def pretraining_loss(model: LatentWorldModel, batch: PretrainingBatch, **loss_config: float) -> dict[str, torch.Tensor]:
    """Filter the past without gradients, then learn current/future observations."""
    observations = batch.sequence["observations"]
    with torch.no_grad():
        if batch.history_observations.shape[1] and bool(batch.history_valid_mask.any()):
            state = model.encode_history(
                batch.history_observations, batch.history_previous_actions, batch.history_valid_mask,
                ego_motion=batch.history_ego_motion, ego_valid=batch.history_ego_valid,
                ego_age_s=batch.history_ego_age_s,
            )
        else:
            state = model.initial(observations.shape[0], device=observations.device, dtype=observations.dtype)
    return world_model_loss(
        model, batch.sequence, initial_state=state.detach(), initial_previous_action=batch.previous_action,
        observation_mask=batch.observation_mask, transition_mask=batch.transition_mask,
        **loss_config,
    )
