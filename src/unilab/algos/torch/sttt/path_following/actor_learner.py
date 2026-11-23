"""Persistent residual-policy optimizer through frozen native-rate imagination."""
from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Mapping

import torch

from .joint_data import select_rollout_start
from .latent_objectives import LatentCostConfig, latent_actor_loss
from .latent_rollout import _frozen_model
from ..world_model.types import RolloutStart


class ActorLearner:
    def __init__(self, actor, rollout, training: Mapping[str, Any], costs: Mapping[str, Any] | LatentCostConfig,
                 *, updates: int = 0):
        self.actor, self.rollout = actor, rollout
        self.costs = costs if isinstance(costs, LatentCostConfig) else LatentCostConfig(**dict(costs))
        self.window_steps, self.horizon_windows = training["window_steps"], training["horizon_windows"]
        self.microbatch_size = training.get("microbatch_size", training["batch_size"])
        if self.microbatch_size is None:
            self.microbatch_size = training["batch_size"]
        for name in ("window_steps", "horizon_windows", "microbatch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"actor {name} must be a positive integer")
        self.clip, rate = training["max_grad_norm"], training["learning_rate"]
        if any(isinstance(x, bool) or not math.isfinite(x) or x <= 0 for x in (self.clip, rate)):
            raise ValueError("actor optimizer requires positive finite learning rate and gradient clip")
        if type(updates) is not int or updates < 0:
            raise ValueError("actor update count must be nonnegative")
        self.updates = updates
        self.optimizer = torch.optim.AdamW(actor.parameters(), lr=rate, weight_decay=0.)
        self.signature = {"learning_rate": rate, "max_grad_norm": self.clip, "microbatch_size": self.microbatch_size,
                          "window_steps": self.window_steps, "horizon_windows": self.horizon_windows,
                          "costs": asdict(self.costs)}

    def update(self, start: RolloutStart) -> dict[str, float]:
        self.actor.train()
        self.optimizer.zero_grad(set_to_none=True)
        batch_size = start.command.shape[0]
        totals: dict[str, torch.Tensor] = {}
        with _frozen_model(self.rollout.model), _frozen_model(self.rollout.policy):
            for offset in range(0, batch_size, self.microbatch_size):
                part = select_rollout_start(start, slice(offset, offset + self.microbatch_size))
                result = self.rollout.run(part, self.actor, window_steps=self.window_steps, horizon_windows=self.horizon_windows)
                losses = latent_actor_loss(result, part, ctrl_dt=self.rollout.profile.ctrl_dt,
                                           window_steps=self.window_steps, config=self.costs)
                if not bool(torch.isfinite(torch.stack(list(losses.values()))).all()):
                    raise ValueError("non-finite imagined actor cost")
                scale = part.command.shape[0] / batch_size
                (losses["loss"] * scale).backward()
                for key, value in losses.items():
                    totals[key] = totals.get(key, torch.zeros_like(value)) + value.detach() * scale
        norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip, error_if_nonfinite=True)
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
            raise ValueError("actor learner checkpoint optimizer, horizon or costs changed")
        if type(state["optimizer_steps"]) is not int or state["optimizer_steps"] < 0:
            raise ValueError("invalid saved actor optimizer-step count")
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.updates = state["optimizer_steps"]
