"""Online adaptation controller integrating playback, collection, and update."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
import torch

from unilab.algos.torch.distill.fada.adaptation import (
    FADALoRAConfig,
    FADALoRALinear,
    FADALoRAQVMultiheadAttention,
)
from unilab.algos.torch.distill.fada.model import FADAPlannerIDMPolicy
from unilab.algos.torch.distill.fada.online_collector import OnlineTrajectoryCollector
from unilab.algos.torch.distill.fada.online_updater import OnlineAdaptationUpdater
from unilab.algos.torch.distill.fada.playback import FADAPlaybackController


@dataclass(frozen=True)
class FADAOnlineAdaptationConfig:
    enabled: bool = False
    lambda_forget: float = 0.99
    update_interval: int = 10
    learning_rate: float = 1e-4
    max_grad_norm: float | None = 1.0
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 < self.lambda_forget <= 1.0:
            raise ValueError("online_adaptation.lambda_forget must be in (0, 1]")
        if isinstance(self.update_interval, bool) or not isinstance(self.update_interval, Integral) or self.update_interval <= 0:
            raise ValueError("online_adaptation.update_interval must be a positive integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("online_adaptation.learning_rate must be finite and positive")
        if self.max_grad_norm is not None and (not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0.0):
            raise ValueError("online_adaptation.max_grad_norm must be None or finite and positive")
        FADALoRAConfig(rank=self.lora_rank, alpha=self.lora_alpha, dropout=self.lora_dropout)


class OnlineAdaptationController:
    """Closed-loop controller: inference → collection → update.

    Owns the playback policy by injecting IDM LoRA adapters into it, so every
    playback action and every online update observe the exact same module.
    Online collection is single-env only.
    """

    def __init__(
        self,
        policy: FADAPlannerIDMPolicy,
        lora_config: FADALoRAConfig,
        *,
        lambda_forget: float = 0.99,
        update_interval: int = 10,
        lr: float = 1e-4,
        max_grad_norm: float | None = 1.0,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        FADAOnlineAdaptationConfig(
            enabled=True, lambda_forget=lambda_forget, update_interval=update_interval,
            learning_rate=lr, max_grad_norm=max_grad_norm,
            lora_rank=lora_config.rank, lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
        )
        if any(
            isinstance(module, (FADALoRALinear, FADALoRAQVMultiheadAttention))
            for module in policy.idm.modules()
        ):
            raise ValueError(
                "OnlineAdaptationController requires a policy without existing LoRA "
                "adapters; reinjecting into an already-adapted policy is unsupported."
            )

        self.collector = OnlineTrajectoryCollector(policy.config)
        self.updater = OnlineAdaptationUpdater(
            policy,
            lora_config,
            lambda_forget=lambda_forget,
            lr=lr,
            max_grad_norm=max_grad_norm,
        )
        # The updater injected adapters into and returns the same playback policy.
        self.policy = self.updater.policy
        self.playback = FADAPlaybackController(self.policy, device=self.device)

        self.update_interval = int(update_interval)
        self._step_count = 0
        self._last_obs: np.ndarray | None = None
        self._last_command: np.ndarray | None = None
        # A_real_(t-1); zero at episode start / after reset.
        self._prev_real_action = np.zeros(
            self.policy.config.action_dim, dtype=np.float32
        )

    def project_observation(self, observation: Any) -> torch.Tensor:
        """Project a raw wrapped actor observation to the FADA policy contract."""

        return self.playback.project_observation(observation)

    def _projected_row(self, observation: Any) -> np.ndarray:
        obs = self.project_observation(observation)
        if obs.ndim != 2 or obs.shape[0] != 1:
            raise ValueError(
                "Online adaptation supports exactly one environment: expected a "
                f"single projected observation row, got shape {tuple(obs.shape)}"
            )
        return obs[0].detach().cpu().numpy().astype(np.float32, copy=True)

    @staticmethod
    def _row(value: Any, *, name: str) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        arr = np.array(value, dtype=np.float32, copy=True)
        if arr.ndim > 1:
            if arr.shape[0] != 1:
                raise ValueError(
                    f"Online adaptation supports exactly one environment: {name} "
                    f"must be a single row, got shape {arr.shape}"
                )
            arr = arr.reshape(-1)
        if arr.ndim != 1 or not np.isfinite(arr).all():
            raise ValueError(f"Online adaptation {name} must be a finite rank-1 row")
        return arr

    def act(self, observation: Any, command: Any) -> torch.Tensor:
        """Execute one receding-horizon action from a raw wrapped observation."""

        projected = self.project_observation(observation)
        if projected.shape[0] != 1:
            raise ValueError("Online adaptation supports exactly one environment")
        action = self.playback.act_projected(projected, command)

        # B1: previous_action (A_real_(t-1)) 由 on_env_step 从上一周期 executed_action 更新;
        # 这里仅缓存当前 observation/command 供 post-step 组装 transition.
        self._last_obs = projected[0].detach().cpu().numpy().astype(np.float32, copy=True)
        self._last_command = self._row(command, name="command")
        return action

    def on_env_step(
        self,
        *,
        executed_action: Any,
        next_observation: Any,
        done: Any | None = None,
    ) -> dict[str, float] | None:
        """Collect one post-step transition and run the periodic LoRA update.

        ``executed_action`` is the environment's ``info["current_actions"]`` row and
        ``next_observation`` the raw wrapped observation after ``env.step``. When
        ``done`` is provided (autoreset enabled), terminal transitions are rejected
        before collection so no window crosses an episode boundary.
        """

        if done is not None and bool(torch.as_tensor(done).reshape(-1)[0].item()):
            # B2: autoreset 已把环境切到新 episode, 当前 transition 的 next_observation
            # 属于下一 episode, 在进入 collector 前整体丢弃.
            self._last_obs = None
            self._last_command = None
            self._prev_real_action.fill(0.0)
            return None
        if self._last_obs is None or self._last_command is None:
            return None

        executed = self._row(executed_action, name="executed_action")
        window = self.collector.append(
            observation=self._last_obs,
            previous_action=self._prev_real_action.copy(),
            command=self._last_command,
            executed_action=executed,
            next_observation=self._projected_row(next_observation),
        )
        self._last_obs = None
        self._last_command = None
        # B2b: 当前 executed_action (A_real_t) 成为下一 transition 的 previous_action.
        self._prev_real_action = executed

        if window is not None:
            self.updater.append_window(window)

        self._step_count += 1
        if self._step_count % self.update_interval == 0:
            return self.updater.update()
        return None

    def reset(self, done: Any | None = None) -> None:
        """Reset playback state; ``reset(done)`` keeps learned LoRA and buffered windows."""

        self.playback.reset(done)
        if done is not None and not bool(torch.as_tensor(done).any().item()):
            return
        # B3: episode 边界清空流式轨迹与 pending transition, 并将 previous_action 归零
        # (episode 起点无真实上一动作), 但保留已学习 LoRA 与 updater 内有效窗口.
        self.collector.reset()
        self._last_obs = None
        self._last_command = None
        self._prev_real_action.fill(0.0)
        if done is None:
            self._step_count = 0
