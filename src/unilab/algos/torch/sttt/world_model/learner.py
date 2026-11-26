"""One effective WM optimizer update, independent of replay and microbatch size."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from .losses import PREDICTION_STATISTICS, pretraining_loss
from .pretraining_data import PretrainingBatch, _positive_integer, select_pretraining_batch


def _target_counts(batch: PretrainingBatch) -> dict[str, torch.Tensor]:
    sequence = batch.sequence
    first, terminal = sequence["is_first"], sequence["terminal"]
    transition = batch.transition_mask & ~first[:, 1:] & ~terminal
    observation = batch.observation_mask & torch.cat((torch.ones_like(first[:, :1]), ~terminal | first[:, 1:]), 1)
    start, end = sequence["window_start"], sequence["window_end"]
    prefix = torch.cat((torch.zeros_like(transition[:, :1], dtype=torch.long), (~transition).long().cumsum(1)), 1)
    window = sequence["window_valid"] & (end > start) & (prefix.gather(1, start) == prefix.gather(1, end))
    return {"observation": observation.sum(), "transition": transition.sum(),
            "motion": (sequence["motion_valid"] & transition).sum(),
            "window": window.sum(), "risk": (sequence["risk_valid"] & batch.transition_mask).sum()}


class WorldModelLearner:
    def __init__(self, model, training: Mapping[str, Any], loss: Mapping[str, Any], *, updates: int = 0):
        self.model, self.loss_config = model, dict(loss)
        microbatch = training.get("microbatch_size")
        self.microbatch_size = _positive_integer(training["batch_size"] if microbatch is None else microbatch, "microbatch_size")
        self.clip = float(training["gradient_clip_norm"])
        learning_rate, decay = training["learning_rate"], training.get("weight_decay", 0.)
        if any(isinstance(value, bool) or not math.isfinite(value) for value in (learning_rate, decay, self.clip)) or learning_rate <= 0 or decay < 0 or self.clip <= 0:
            raise ValueError("WM optimizer requires positive learning rate/clip and nonnegative weight decay")
        if type(updates) is not int or updates < 0:
            raise ValueError("WM update count must be nonnegative")
        self.updates = updates
        self.optimizer_started_at = updates
        motion_rate = training.get("motion_learning_rate", learning_rate * .1)
        self.motion_clip = training.get("motion_gradient_clip_norm", 1.)
        self.warmup_updates = training.get("warmup_updates", 1000)
        if any(isinstance(value, bool) or not math.isfinite(value) or value <= 0 for value in (motion_rate, self.motion_clip)):
            raise ValueError("motion-head learning rate and gradient clip must be positive")
        if type(self.warmup_updates) is not int or self.warmup_updates < 0:
            raise ValueError("WM warmup_updates must be nonnegative")
        model.requires_grad_(True)
        motion_parameters = list(model.motion_head.parameters())
        motion_ids = {id(parameter) for parameter in motion_parameters}
        self.optimizer = torch.optim.AdamW([
            {"params": [parameter for parameter in model.parameters() if id(parameter) not in motion_ids],
             "lr": learning_rate, "base_lr": learning_rate},
            {"params": motion_parameters, "lr": motion_rate, "base_lr": motion_rate},
        ], weight_decay=decay)
        self.signature = {"learning_rate": learning_rate, "motion_learning_rate": motion_rate,
                          "warmup_updates": self.warmup_updates, "weight_decay": decay,
                          "gradient_clip_norm": self.clip, "motion_gradient_clip_norm": self.motion_clip,
                          "microbatch_size": self.microbatch_size, "loss": self.loss_config}

    def restart_optimizer(self) -> None:
        """Start a fresh optimizer after selecting the warmup model; keep total update counts."""
        self.optimizer.state.clear()
        self.optimizer_started_at = self.updates

    def update(self, batch: PretrainingBatch) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        denominators = _target_counts(batch)
        normalized_by = {"reconstruction_loss": "observation", "kl_loss": "observation", "dynamics_kl": "observation",
                         "representation_kl": "observation", "prior_reconstruction_loss": "transition",
                         "overshoot_loss": "transition", "motion_loss": "motion", "window_loss": "window", "risk_loss": "risk"}
        config = self.loss_config
        weights = {"reconstruction_loss": config.get("reconstruction_scale", 1.), "kl_loss": config.get("kl_scale", 1.),
                   "prior_reconstruction_loss": config.get("prior_reconstruction_scale", 1.),
                   "overshoot_loss": config.get("overshoot_scale", 1.), "motion_loss": config.get("motion_scale", 1.),
                   "window_loss": config.get("motion_scale", 1.) * config.get("window_scale", 1.), "risk_loss": config.get("risk_scale", 1.)}
        totals: dict[str, torch.Tensor] = {}
        for offset in range(0, batch.previous_action.shape[0], self.microbatch_size):
            part = select_pretraining_batch(batch, slice(offset, offset + self.microbatch_size))
            losses = pretraining_loss(self.model, part, **config)
            if not bool(torch.isfinite(torch.stack(list(losses.values()))).all()):
                raise ValueError("non-finite world-model training loss")
            counts = _target_counts(part)
            scaled = {key: losses[key] * counts[group] / denominators[group].clamp_min(1)
                      for key, group in normalized_by.items()}
            total = losses["loss"] if self.microbatch_size >= batch.previous_action.shape[0] else torch.stack([weights[key] * scaled[key] for key in weights]).sum()
            total.backward()
            values = {**scaled, "loss": total,
                      **{key: losses[key] for key in PREDICTION_STATISTICS}}
            for key, value in values.items():
                totals[key] = totals.get(key, torch.zeros_like(value)) + value.detach()
        motion_norm = torch.nn.utils.clip_grad_norm_(self.model.motion_head.parameters(), self.motion_clip, error_if_nonfinite=True)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip, error_if_nonfinite=True)
        factor = min(1., (self.updates - self.optimizer_started_at + 1) / max(1, self.warmup_updates))
        for group in self.optimizer.param_groups:
            group["lr"] = group["base_lr"] * factor
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.updates += 1
        totals["gradient_norm"] = norm.detach()
        totals["motion_gradient_norm"] = motion_norm.detach()
        totals["learning_rate"] = norm.new_tensor(self.optimizer.param_groups[0]["lr"])
        totals["motion_learning_rate"] = norm.new_tensor(self.optimizer.param_groups[1]["lr"])
        values = torch.stack([value.float() for value in totals.values()]).cpu().tolist()
        return dict(zip(totals, values))

    def state_dict(self) -> dict[str, Any]:
        return {"version": 2, "signature": self.signature, "optimizer_steps": self.updates,
                "optimizer_started_at": self.optimizer_started_at, "optimizer_state": self.optimizer.state_dict()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if set(state) != {"version", "signature", "optimizer_steps", "optimizer_started_at", "optimizer_state"} or state["version"] != 2 or state["signature"] != self.signature:
            raise ValueError("WM learner checkpoint optimizer or loss configuration changed")
        if type(state["optimizer_steps"]) is not int or state["optimizer_steps"] < 0:
            raise ValueError("invalid saved WM optimizer-step count")
        if type(state["optimizer_started_at"]) is not int or not 0 <= state["optimizer_started_at"] <= state["optimizer_steps"]:
            raise ValueError("invalid saved WM optimizer warmup origin")
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.updates = state["optimizer_steps"]
        self.optimizer_started_at = state["optimizer_started_at"]
