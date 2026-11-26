"""Window-rate PPO seam over the existing native Planner–Tracker collector."""
from __future__ import annotations

import torch
from tensordict import TensorDict

from .joint_data import join_rollout_starts
from .residual_rewards import ResidualRewardConfig


class ResidualControlEnv:
    """PPO sees held residuals; the inner environment still executes native-rate actions."""

    def __init__(self, collectors, replay, actor, *, rewards: ResidualRewardConfig, context_adapter=None):
        self.collectors, self.replay, self.actor = collectors, replay, actor
        self.rewards, self.context_adapter = rewards, context_adapter
        self.num_envs = sum(item.num_envs for item in collectors.items)
        self.num_actions = 29
        self.rollout = collectors.items[0].rollout
        self.device = next(actor.parameters()).device
        self.physical_control_steps = 0
        self.completed_directories = []
        self.last_context_metrics = {}
        self.refresh(adapt=False)

    def policy_observation(self, start, context: torch.Tensor) -> TensorDict:
        if start.path_state is None:
            raise ValueError("PPO requires an observed reference preview")
        with torch.no_grad():
            value = self.actor.context(start.latent.feature(), start.errors, start.command,
                start.path_state.preview(), start.previous_residual)
        return TensorDict({"policy": value.detach().clone(), "adaptation_context": context.detach().clone()},
                          batch_size=[value.shape[0]], device=self.device)

    def refresh(self, *, adapt: bool = True) -> TensorDict:
        self.start = join_rollout_starts([item.current_start() for item in self.collectors.items])
        if self.context_adapter is not None:
            if adapt:
                self.last_context_metrics = self.context_adapter.step(self.start)
            context = self.context_adapter.context
        else:
            context = self.actor.default_context.expand(self.num_envs, -1)
        self.observation = self.policy_observation(self.start, context)
        return self.observation

    def get_observations(self) -> TensorDict:
        return self.observation

    def step(self, raw_actions: torch.Tensor, *, versions):
        residuals = self.actor.physical_residual(raw_actions)
        previous_context = self.observation["adaptation_context"]
        outcomes, offset = [], 0
        for collector in self.collectors.items:
            end = offset + collector.num_envs
            outcome = collector.step_residual_window(residuals[offset:end], self.replay,
                versions=versions, rewards=self.rewards)
            outcomes.append(outcome)
            self.physical_control_steps += outcome.block.physical_control_steps
            self.completed_directories.extend(outcome.block.completed_directories)
            offset = end
        reward = torch.cat([outcome.rewards for outcome in outcomes])
        done = torch.cat([outcome.done for outcome in outcomes])
        timeouts = torch.cat([outcome.timeouts for outcome in outcomes])
        terminal = join_rollout_starts([outcome.terminal_start for outcome in outcomes])
        terminal_obs = self.policy_observation(terminal, previous_context)
        if self.context_adapter is not None and bool(done.any()):
            self.context_adapter.reset(done)
        observation = self.refresh()
        terms, diagnostics = {}, {}
        for outcome in outcomes:
            for name, value in outcome.reward_terms.items():
                terms[name] = terms.get(name, 0.) + value / self.num_envs
            for name, value in outcome.diagnostics.items():
                diagnostics[name] = diagnostics.get(name, 0.) + value / self.num_envs
        return observation, reward, done, {"time_outs": timeouts, "terminal_observation": terminal_obs,
            "reward_terms": terms, "body_metrics": diagnostics, **self.last_context_metrics}
