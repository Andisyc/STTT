"""Persistent control context adaptation through a frozen physical rollout."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch

from .latent_rollout import _frozen_model


@dataclass(frozen=True)
class ContextAdaptationConfig:
    dimension: int = 16
    learning_rate: float = .001
    updates_per_window: int = 1
    horizon_windows: int = 2
    max_grad_norm: float = 1.

    def __post_init__(self) -> None:
        for name in ("dimension", "updates_per_window", "horizon_windows"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"context {name} must be a positive integer")
        for name in ("learning_rate", "max_grad_norm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"context {name} must be positive and finite")


class ContextAdapter:
    def __init__(self, actor, rollout, *, batch_size: int, window_steps: int,
                 config: ContextAdaptationConfig):
        if actor.adaptation_dim != config.dimension or type(batch_size) is not int or batch_size < 1:
            raise ValueError("FiLM and context dimensions differ")
        if type(window_steps) is not int or window_steps < 1:
            raise ValueError("context adaptation requires a physical residual window")
        self.actor, self.rollout, self.config = actor, rollout, config
        self.window_steps = window_steps
        self.context = torch.nn.Parameter(actor.default_context.expand(batch_size, -1).clone())
        self.optimizer = torch.optim.Adam([self.context], lr=config.learning_rate)
        self.updates = 0

    @torch.enable_grad()
    def step(self, start) -> dict[str, float]:
        if torch.is_inference_mode_enabled():
            raise ValueError("context adaptation must run outside inference_mode")
        if start.command.shape[0] != self.context.shape[0]:
            raise ValueError("context and physical start batches differ")
        loss = self.context.new_zeros(())
        norm = self.context.new_zeros(())
        with _frozen_model(self.actor), _frozen_model(self.rollout.model), _frozen_model(self.rollout.policy):
            for _ in range(self.config.updates_per_window):
                self.optimizer.zero_grad(set_to_none=True)
                result = self.rollout.run(start, self.actor, window_steps=self.window_steps,
                    horizon_windows=self.config.horizon_windows, adaptation_context=self.context)
                # Predicted errors already equal measured initial error plus predicted changes.
                per_stream = result.errors[:, self.window_steps::self.window_steps].abs().mean(dim=(1, 2))
                loss = per_stream.sum()
                if not bool(torch.isfinite(loss)):
                    raise ValueError("non-finite trajectory context loss")
                gradient, = torch.autograd.grad(loss, self.context)
                if not bool(torch.isfinite(gradient).all()):
                    raise ValueError("non-finite context gradient")
                stream_norms = gradient.norm(dim=-1, keepdim=True)
                norm = stream_norms.mean()
                self.context.grad = gradient * (self.config.max_grad_norm / stream_norms.clamp_min(1e-12)).clamp_max(1.)
                self.optimizer.step()
                self.updates += 1
        self.optimizer.zero_grad(set_to_none=True)
        return {"context_loss": float(loss.detach()) / self.context.shape[0],
                "context_gradient_norm": float(norm), "context_norm": float(self.context.detach().norm(dim=-1).mean())}

    @torch.no_grad()
    def reset(self, rows: torch.Tensor | None = None) -> None:
        if rows is None:
            self.context.copy_(self.actor.default_context)
            self.optimizer.state.clear()
            return
        else:
            self.context[rows] = self.actor.default_context
        for value in self.optimizer.state.get(self.context, {}).values():
            if isinstance(value, torch.Tensor) and value.shape == self.context.shape:
                if rows is None:
                    value.zero_()
                else:
                    value[rows] = 0.

    def state_dict(self) -> dict[str, Any]:
        return {"version": 1, "config": asdict(self.config), "window_steps": self.window_steps,
                "context": self.context.detach().cpu().clone(), "optimizer": self.optimizer.state_dict(),
                "updates": self.updates}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state["version"] != 1 or state["config"] != asdict(self.config) or state["window_steps"] != self.window_steps:
            raise ValueError("context adaptation checkpoint configuration changed")
        value = state["context"].to(self.context)
        if value.shape != self.context.shape or not bool(torch.isfinite(value).all()):
            raise ValueError("invalid saved adaptation context")
        with torch.no_grad():
            self.context.copy_(value)
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = state["updates"]
