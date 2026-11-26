"""Persistent residual-policy optimizer through frozen native-rate imagination."""
from __future__ import annotations

import math
from dataclasses import asdict, replace
from typing import Any, Mapping

import torch

from .joint_data import select_rollout_start
from .critic import CriticConfig, CriticLearner
from .latent_objectives import LatentCostConfig, ObservedCostWindow, latent_actor_loss, latent_window_costs, weighted_window_costs
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
        self.betas = tuple(training.get("optimizer_betas", (.7, .95)))
        self.final_rate = training.get("final_learning_rate", 1e-5)
        if len(self.betas) != 2 or any(isinstance(value, bool) or not math.isfinite(value) or not 0 <= value < 1 for value in self.betas):
            raise ValueError("actor optimizer betas must contain two values in [0,1)")
        if isinstance(self.final_rate, bool) or not math.isfinite(self.final_rate) or self.final_rate <= 0:
            raise ValueError("final learning rate must be positive and finite")
        self.optimizer = torch.optim.Adam(actor.parameters(), lr=rate, betas=self.betas)
        self.signature = {"learning_rate": rate, "max_grad_norm": self.clip, "microbatch_size": self.microbatch_size,
                          "window_steps": self.window_steps, "horizon_windows": self.horizon_windows,
                          "costs": asdict(self.costs), "optimizer_betas": self.betas,
                          "final_learning_rate": self.final_rate}
        self.critic_learner = CriticLearner(actor.input_dim, CriticConfig.from_mapping(training.get("critic", {})),
            parameter=next(actor.parameters()), ctrl_dt=rollout.profile.ctrl_dt,
            window_steps=self.window_steps, discount=self.costs.discount, failure_cost_rate=self.costs.failure_cost_rate)
        self.schedule_updates = training.get("max_updates")
        if self.schedule_updates is not None and (type(self.schedule_updates) is not int or self.schedule_updates < 1):
            raise ValueError("actor learning-rate schedule requires a positive update budget")
        parameter = next(actor.parameters())
        # Parallel running moments: count, mean and sum of squared deviations.
        self.return_moments = parameter.new_tensor([1e-4, 0., 1e-4])

    @torch.no_grad()
    def refresh_start(self, start: RolloutStart) -> RolloutStart:
        """WM updates invalidate cached encodings, not the owned real measurements."""
        valid = start.history_valid_mask
        previous = torch.cat((torch.zeros_like(valid[:, :1]), valid[:, :-1]), 1)
        actions = self.rollout.source_to_env(start.action_history).masked_fill(~(valid & previous)[..., None], 0.)
        with _frozen_model(self.rollout.model):
            latent = self.rollout.model.encode_history(start.observation_history, actions, valid,
                ego_motion=start.ego_motion, ego_valid=start.ego_valid, ego_age_s=start.ego_age_s)
        return replace(start, latent=latent.detach())

    def fit_critic(self, start: RolloutStart,
                   failures: tuple[tuple[RolloutStart, ObservedCostWindow], ...] = ()) -> dict[str, float]:
        contexts, costs = [], []
        with torch.no_grad(), _frozen_model(self.rollout.model), _frozen_model(self.rollout.policy):
            for offset in range(0, start.command.shape[0], self.microbatch_size):
                part = select_rollout_start(start, slice(offset, offset + self.microbatch_size))
                result = self.rollout.run(part, self.actor, window_steps=self.window_steps,
                                          horizon_windows=self.horizon_windows, stochastic_policy=True)
                windows = latent_window_costs(result, part, ctrl_dt=self.rollout.profile.ctrl_dt,
                    window_steps=self.window_steps, config=self.costs, residual_limits=self.actor.residual_limit_rad29)
                contexts.append(result.decision_contexts)
                costs.append(weighted_window_costs(windows, self.costs))
        return self._fit_critic(torch.cat(contexts), torch.cat(costs), failures)

    def _fit_critic(self, contexts, values, failures) -> dict[str, float]:
        observed_contexts, observed_targets = [], []
        with torch.no_grad():
            for previous, measured in failures:
                previous = self.refresh_start(previous)
                steps = measured.actions.shape[1]
                actual = latent_window_costs(measured, previous, ctrl_dt=self.rollout.profile.ctrl_dt,
                    window_steps=steps, config=self.costs, residual_limits=self.actor.residual_limit_rad29)
                context = self.actor.context(previous.latent.feature(), previous.errors, previous.command,
                                             previous.path_state.preview(), previous.previous_residual)
                observed_contexts.append(context)
                observed_targets.append(weighted_window_costs(actual, self.costs)[:, 0]
                    + self.costs.discount ** steps * self.critic_learner.failure_value)
        # Imagined risk never deletes future costs; real failures retain an absorbing tail.
        return self.critic_learner.update(contexts, values, torch.zeros_like(values, dtype=torch.bool),
            observed_contexts=torch.cat(observed_contexts) if observed_contexts else None,
            observed_targets=torch.cat(observed_targets) if observed_targets else None)

    def update(self, start: RolloutStart,
               failures: tuple[tuple[RolloutStart, ObservedCostWindow], ...] = ()) -> dict[str, float]:
        start = self.refresh_start(start)
        imagined = []
        metrics = self.actor_step(start, critic_rollouts=imagined)
        contexts, costs = zip(*imagined)
        return {**metrics, **self._fit_critic(torch.cat(contexts), torch.cat(costs), failures)}

    def actor_step(self, start: RolloutStart, *, critic_rollouts=None) -> dict[str, float]:
        """Apply one actor update; optionally retain its detached rollout for Critic fitting."""
        self.actor.train()
        if self.schedule_updates is not None:
            fraction = min(self.updates / self.schedule_updates, 1.)
            for optimizer, rate in ((self.optimizer, self.signature["learning_rate"]),
                                    (self.critic_learner.optimizer, self.critic_learner.config.learning_rate)):
                for group in optimizer.param_groups:
                    group["lr"] = rate + (min(rate, self.final_rate) - rate) * fraction
        self.optimizer.zero_grad(set_to_none=True)
        batch_size = start.command.shape[0]
        totals: dict[str, torch.Tensor] = {}
        returns = []
        return_scale = (self.return_moments[2] / self.return_moments[0]).sqrt().clamp_min(1e-4).detach()
        with _frozen_model(self.rollout.model), _frozen_model(self.rollout.policy), _frozen_model(self.critic_learner.target):
            for offset in range(0, batch_size, self.microbatch_size):
                part = select_rollout_start(start, slice(offset, offset + self.microbatch_size))
                result = self.rollout.run(part, self.actor, window_steps=self.window_steps,
                                          horizon_windows=self.horizon_windows, stochastic_policy=True)
                terminal = self.critic_learner.target(result.decision_contexts[:, -1])
                losses = latent_actor_loss(result, part, ctrl_dt=self.rollout.profile.ctrl_dt,
                                           window_steps=self.window_steps, config=self.costs,
                                           residual_limits=self.actor.residual_limit_rad29,
                                           terminal_cost=terminal)
                with torch.no_grad():
                    windows = latent_window_costs(result, part, ctrl_dt=self.rollout.profile.ctrl_dt,
                        window_steps=self.window_steps, config=self.costs, residual_limits=self.actor.residual_limit_rad29)
                    costs = weighted_window_costs(windows, self.costs)
                    discount = self.costs.discount ** self.window_steps
                    weights = discount ** torch.arange(costs.shape[1], device=costs.device, dtype=costs.dtype)
                    returns.append((costs * weights).sum(-1) + discount ** costs.shape[1] * terminal.detach())
                    if critic_rollouts is not None:
                        critic_rollouts.append((result.decision_contexts.detach(), costs))
                if not bool(torch.isfinite(torch.stack(list(losses.values()))).all()):
                    raise ValueError("non-finite imagined actor cost")
                scale = part.command.shape[0] / batch_size
                (losses["loss"] * scale / return_scale).backward()
                for key, value in losses.items():
                    totals[key] = totals.get(key, torch.zeros_like(value)) + value.detach() * scale
        norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip, error_if_nonfinite=True)
        self.optimizer.step()
        with torch.no_grad():
            self.actor.log_std.clamp_(min=self.actor.min_log_std)
            observed = torch.cat(returns)
            variance, mean = torch.var_mean(observed, unbiased=False)
            count, old_mean, m2 = self.return_moments.unbind()
            total = count + observed.numel()
            delta = mean - old_mean
            self.return_moments.copy_(torch.stack((
                total, old_mean + delta * observed.numel() / total,
                m2 + variance * observed.numel() + delta.square() * count * observed.numel() / total)))
        self.optimizer.zero_grad(set_to_none=True)
        self.updates += 1
        totals["gradient_norm"] = norm.detach()
        values = torch.stack([value.float() for value in totals.values()]).cpu().tolist()
        return dict(zip(totals, values))

    def state_dict(self) -> dict[str, Any]:
        return {"version": 3, "signature": self.signature, "optimizer_steps": self.updates,
                "optimizer_state": self.optimizer.state_dict(), "critic_learner": self.critic_learner.state_dict(),
                "return_moments": self.return_moments.detach().clone(), "schedule_updates": self.schedule_updates}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if set(state) != {"version", "signature", "optimizer_steps", "optimizer_state", "critic_learner",
                          "return_moments", "schedule_updates"} or state["version"] != 3 or state["signature"] != self.signature:
            raise ValueError("expected matching version-3 stochastic Actor/Critic optimizer, horizon and costs")
        if type(state["optimizer_steps"]) is not int or state["optimizer_steps"] < 0:
            raise ValueError("invalid saved actor optimizer-step count")
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.updates = state["optimizer_steps"]
        self.critic_learner.load_state_dict(state["critic_learner"])
        moments = torch.as_tensor(state["return_moments"]).to(self.return_moments)
        if moments.shape != (3,) or not bool(torch.isfinite(moments).all()) or moments[0] <= 0 or moments[2] < 0:
            raise ValueError("invalid saved return normalization")
        budget = state["schedule_updates"]
        if budget is not None and (type(budget) is not int or budget < 1):
            raise ValueError("invalid saved learning-rate schedule")
        self.return_moments.copy_(moments)
        self.schedule_updates = budget
