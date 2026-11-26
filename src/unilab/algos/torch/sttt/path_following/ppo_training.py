"""Native RSL-RL PPO over real residual execution, with separate WM supervision."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict
from torch import nn


@dataclass(frozen=True)
class ResidualPPOConfig:
    rollout_windows: int = 16
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    clip_param: float = .2
    gamma: float = .99
    lam: float = .95
    value_loss_coef: float = 1.
    entropy_coef: float = .01
    learning_rate: float = .0005
    max_grad_norm: float = 1.
    desired_kl: float = .01

    def __post_init__(self) -> None:
        for name in ("rollout_windows", "num_learning_epochs", "num_mini_batches"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"PPO {name} must be positive")
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid PPO {name}")
        if not 0 < self.gamma <= 1 or not 0 <= self.lam <= 1 or not 0 < self.clip_param < 1:
            raise ValueError("invalid PPO discount or clipping range")
        if self.learning_rate == 0 or self.max_grad_norm == 0:
            raise ValueError("PPO learning rate and gradient clip must be positive")


class ResidualPolicyModel(MLPModel):
    """RSL model interface backed by the exact deployment actor and Gaussian measure."""

    distribution: GaussianDistribution

    def __init__(self, actor):
        nn.Module.__init__(self)
        self.actor = actor
        self.obs_normalization = False
        self.obs_normalizer = nn.Identity()
        self.obs_groups = ["policy", "adaptation_context"]
        self.obs_dim = actor.input_dim + actor.adaptation_dim
        self.distribution = GaussianDistribution(29, init_std=math.exp(-1.), std_type="log")
        self.distribution.log_std_param = actor.log_std

    def forward(self, obs: TensorDict, masks=None, hidden_state=None, stochastic_output: bool = False):
        mean = self.actor.raw_action(obs["policy"], obs["adaptation_context"])
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return mean


class ResidualPPOTrainer:
    def __init__(self, actor, env, config: ResidualPPOConfig, *, stage: str):
        samples = env.num_envs * config.rollout_windows
        if config.num_mini_batches > samples or samples % config.num_mini_batches:
            raise ValueError("PPO minibatches must partition all collected residual decisions exactly")
        actor.train_stage(stage)
        self.actor, self.env, self.rollout, self.config, self.stage = actor, env, env.rollout, config, stage
        observation = env.get_observations()
        policy = ResidualPolicyModel(actor)
        critic = MLPModel(observation, {"critic": ["policy", "adaptation_context"]}, "critic", 1,
                          hidden_dims=actor.hidden_dims, activation="elu").to(next(actor.parameters()))
        storage = RolloutStorage("rl", env.num_envs, config.rollout_windows, observation, [29], str(env.device))
        arguments = asdict(config)
        arguments.pop("rollout_windows")
        self.algorithm = PPO(policy, critic, storage, device=str(env.device), **arguments)
        self.updates = 0

    def update(self, *, versions):
        self.actor.train_stage(self.stage)
        self.algorithm.train_mode()
        observation = self.env.get_observations()
        totals = {}
        reward_sum = 0.
        for _ in range(self.config.rollout_windows):
            # Context adaptation explicitly enables autograd; inference_mode would disable it.
            with torch.no_grad():
                actions = self.algorithm.act(observation)
                following, reward, done, extras = self.env.step(actions, versions=versions)
                # Bootstrap the actual timeout state, not the pre-action value or reset state.
                terminal_value = self.algorithm.critic(extras["terminal_observation"]).squeeze(-1)
                adjusted_reward = reward + self.config.gamma * terminal_value * extras["time_outs"]
                self.algorithm.process_env_step(following, adjusted_reward, done, {})
            reward_sum += float(reward.mean())
            for name, value in extras["reward_terms"].items():
                totals[name] = totals.get(name, 0.) + value
            observation = following
        with torch.no_grad():
            self.algorithm.compute_returns(observation)
        losses = self.algorithm.update()
        self.algorithm.optimizer.zero_grad(set_to_none=True)
        self.updates += 1
        return {"loss": losses["surrogate"], "critic_loss": losses["value"], "entropy": losses["entropy"],
                "mean_reward": reward_sum / self.config.rollout_windows,
                "reward_terms": {name: value / self.config.rollout_windows for name, value in totals.items()},
                "lateral_error_m": float(self.env.start.errors[:, 0].abs().mean()),
                **self.env.last_context_metrics}

    def state_dict(self):
        return {"version": 4, "algorithm": "rsl_rl_ppo", "stage": self.stage,
                "config": asdict(self.config), "updates": self.updates,
                "ppo": self.algorithm.save(), "learning_rate": self.algorithm.learning_rate,
                "context": None if self.env.context_adapter is None else self.env.context_adapter.state_dict()}

    def load_state_dict(self, state):
        if state["version"] != 4 or state["algorithm"] != "rsl_rl_ppo" or state["stage"] != self.stage or state["config"] != asdict(self.config):
            raise ValueError("PPO resume configuration or stage changed")
        self.algorithm.load(state["ppo"], load_cfg=None, strict=True)
        self.algorithm.learning_rate = float(state["learning_rate"])
        self.updates = state["updates"]
        if self.env.context_adapter is not None:
            self.env.context_adapter.load_state_dict(state["context"])
        self.env.refresh(adapt=False)
