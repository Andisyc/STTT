"""One effective WM optimizer update, independent of replay and microbatch size."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from .losses import pretraining_loss
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
        model.requires_grad_(True)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=decay)
        self.signature = {"learning_rate": learning_rate, "weight_decay": decay, "gradient_clip_norm": self.clip,
                          "microbatch_size": self.microbatch_size, "loss": self.loss_config}

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
            values = {**scaled, "loss": total, "window_xy_squared_error_sum": losses["window_xy_squared_error_sum"],
                      "window_xy_count": losses["window_xy_count"]}
            for key, value in values.items():
                totals[key] = totals.get(key, torch.zeros_like(value)) + value.detach()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip, error_if_nonfinite=True)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.updates += 1
        totals["gradient_norm"] = norm.detach()
        values = torch.stack([value.float() for value in totals.values()]).cpu().tolist()
        return dict(zip(totals, values))

    def state_dict(self) -> dict[str, Any]:
        return {"version": 1, "signature": self.signature, "optimizer_steps": self.updates,
                "optimizer_state": self.optimizer.state_dict()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if set(state) != {"version", "signature", "optimizer_steps", "optimizer_state"} or state["version"] != 1 or state["signature"] != self.signature:
            raise ValueError("WM learner checkpoint optimizer or loss configuration changed")
        if type(state["optimizer_steps"]) is not int or state["optimizer_steps"] < 0:
            raise ValueError("invalid saved WM optimizer-step count")
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.updates = state["optimizer_steps"]
