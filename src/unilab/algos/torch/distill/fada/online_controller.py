"""Close the deployed FADA loop with fixed snapshot labels and LoRA regression."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import torch

from unilab.algos.torch.distill.fada.adaptation import FADALoRAConfig
from unilab.algos.torch.distill.fada.model import FADAPlannerIDMPolicy
from unilab.algos.torch.distill.fada.online_collector import OnlineTrajectoryCollector
from unilab.algos.torch.distill.fada.online_updater import OnlineSnapshotUpdater, build_expert_samples
from unilab.algos.torch.distill.fada.playback import FADAPlaybackController


@dataclass(frozen=True)
class FADAOnlineAdaptationConfig:
    enabled: bool = False
    window_size: int = 10
    warmup_steps: int = 40
    gamma: float = 0.05
    sigma: float = 1e-4
    update_interval: int = 10
    learning_rate: float = 1e-4
    max_grad_norm: float | None = 1.0
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05

    def __post_init__(self) -> None:
        for name in ("window_size", "update_interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"online_adaptation.{name} must be a positive integer")
        if (
            isinstance(self.warmup_steps, bool)
            or not isinstance(self.warmup_steps, Integral)
            or self.warmup_steps < 0
        ):
            raise ValueError("online_adaptation.warmup_steps must be a non-negative integer")
        for name in ("gamma", "sigma"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"online_adaptation.{name} must be finite and non-negative")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("online_adaptation.learning_rate must be finite and positive")
        if self.max_grad_norm is not None and (
            not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0.0
        ):
            raise ValueError("online_adaptation.max_grad_norm must be finite and positive or None")
        FADALoRAConfig(rank=self.lora_rank, alpha=self.lora_alpha, dropout=self.lora_dropout)


class OnlineAdaptationController:
    """Share one policy between action generation and snapshot-label regression."""

    def __init__(
        self,
        policy: FADAPlannerIDMPolicy,
        lora_config: FADALoRAConfig,
        *,
        window_size: int = 10,
        warmup_steps: int = 40,
        gamma: float = 0.05,
        sigma: float = 1e-4,
        update_interval: int = 10,
        lr: float = 1e-4,
        max_grad_norm: float | None = 1.0,
        device: str | torch.device = "cpu",
    ) -> None:
        FADAOnlineAdaptationConfig(
            enabled=True, window_size=window_size, gamma=gamma, sigma=sigma,
            warmup_steps=warmup_steps, update_interval=update_interval, learning_rate=lr,
            max_grad_norm=max_grad_norm, lora_rank=lora_config.rank,
            lora_alpha=lora_config.alpha, lora_dropout=lora_config.dropout,
        )
        self.device = torch.device(device)
        self.collector = OnlineTrajectoryCollector(policy.config, window_size=window_size)
        self.base_idm = copy.deepcopy(policy.idm).eval().requires_grad_(False)
        self.updater = OnlineSnapshotUpdater(
            policy, lora_config, sigma=sigma, lr=lr, max_grad_norm=max_grad_norm
        )
        self.policy = self.updater.policy
        self.policy.eval()
        self.playback = FADAPlaybackController(self.policy, device=self.device)
        self.update_interval = int(update_interval)
        self.warmup_steps = int(warmup_steps)
        self.gamma = float(gamma)
        self._step_count = 0
        self._collection_suspended = False

    def project_observation(self, observation: Any) -> torch.Tensor:
        return self.playback.project_observation(observation)

    def act(self, observation: Any, command: Any) -> torch.Tensor:
        projected = self.project_observation(observation)
        if projected.shape[0] != 1:
            raise ValueError("Online adaptation supports exactly one environment")
        if self._collection_suspended:
            return self.playback.act_projected(projected, command)
        action, snapshot = self.playback.act_projected_with_snapshot(projected, command)
        self.collector.begin(snapshot)
        return action

    def on_env_step(
        self,
        *,
        executed_action: Any,
        next_observation: Any,
        done: Any | None = None,
    ) -> dict[str, float | int] | None:
        if self._collection_suspended:
            return None
        if done is not None and bool(torch.as_tensor(done).any().item()):
            # complete ignores the terminal observation, which may already be reset.
            self.collector.complete(next_observation, done=True)
            self._collection_suspended = True
            return None
        self.collector.complete(self.project_observation(next_observation))
        self._step_count += 1
        if (
            self._step_count >= self.warmup_steps
            and self._step_count % self.update_interval == 0
            and self.collector.ready
        ):
            samples = build_expert_samples(
                self.base_idm, self.collector.get_window(), gamma=self.gamma
            )
            stats = self.updater.update(samples)
            self.collector.clear_window()
            return stats
        return None

    def reset(self, done: Any | None = None) -> None:
        """Reset collection/history, retaining learned LoRA, anchor and Adam state."""
        if done is not None and not bool(torch.as_tensor(done).any().item()):
            return
        self.playback.reset(done)
        self.collector.reset()
        self._step_count = 0
        self._collection_suspended = False
