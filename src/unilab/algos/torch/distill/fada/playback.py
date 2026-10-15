from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import torch

from unilab.algos.torch.distill.fada.model import FADAArchitectureConfig
from unilab.algos.torch.distill.fada.observation import project_fada_observation_tensor


class FADAPlaybackActionOutput(Protocol):
    """Action-bearing output consumed by receding-horizon playback."""

    @property
    def action(self) -> torch.Tensor: ...


class FADAPlaybackPolicy(Protocol):
    """Three-input healthy/Context policy boundary used by playback."""

    @property
    def config(self) -> FADAArchitectureConfig: ...

    def __call__(
        self,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
        command: torch.Tensor,
    ) -> FADAPlaybackActionOutput: ...
class FADAPlaybackController:
    """Own FADA execution histories and emit one receding-horizon action per query."""

    def __init__(self, policy: FADAPlaybackPolicy, *, device: str | torch.device) -> None:
        self.policy = policy
        self.config = policy.config
        self.device = torch.device(device)
        self._observation_history: torch.Tensor | None = None
        self._action_history: torch.Tensor | None = None
        self._pending_reset: torch.Tensor | None = None

    def reset(self, done: Any | None = None) -> None:
        """Reset every history, or mark selected rows for reset at their next observation."""

        if done is None:
            self._observation_history = None
            self._action_history = None
            self._pending_reset = None
            return
        mask = torch.as_tensor(done, dtype=torch.bool, device=self.device).reshape(-1)
        if (
            self._observation_history is not None
            and mask.numel() != self._observation_history.shape[0]
        ):
            raise ValueError(
                "FADA playback reset mask batch mismatch: "
                f"expected={self._observation_history.shape[0]} observed={mask.numel()}"
            )
        self._pending_reset = mask

    def project_observation(self, observation: Any) -> torch.Tensor:
        """Project a raw actor observation batch to the FADA policy contract."""

        return self._observation_tensor(observation)

    def on_env_step(
        self,
        *,
        executed_action: Any,
        next_observation: Any,
        done: Any | None = None,
    ) -> dict[str, float] | None:
        """Uniform post-step hook; offline playback performs no collection or update."""

        return None

    @torch.no_grad()
    def act(self, observation: Any, command: Any) -> torch.Tensor:
        """Project a raw actor observation and execute one receding-horizon action."""

        return self._act(self.project_observation(observation), command)

    @torch.no_grad()
    def act_projected(self, observation: Any, command: Any) -> torch.Tensor:
        """Execute from an observation already projected to the policy contract."""

        return self._act(self._projected_observation_tensor(observation), command)

    def _act(self, obs: torch.Tensor, command: Any) -> torch.Tensor:
        cmd = torch.as_tensor(command, dtype=torch.float32, device=self.device)
        if cmd.ndim == 1:
            cmd = cmd.unsqueeze(0)
        expected_command_shape = (obs.shape[0], self.config.command_dim)
        if tuple(cmd.shape) != expected_command_shape:
            raise ValueError(
                "FADA playback command shape mismatch: "
                f"expected={expected_command_shape} observed={tuple(cmd.shape)}"
            )
        if not bool(torch.isfinite(cmd).all()):
            raise ValueError("FADA playback command must contain only finite values")

        # B2: 冷启动及 episode reset 使用当前 observation 重复与零 action 初始化对应 history.
        self._advance_observation_history(obs)
        assert self._observation_history is not None
        assert self._action_history is not None

        # B3: 查询 Planner-IDM 并仅记录/返回第一动作, 为下一 control step 形成执行 history.
        action = self.policy(self._observation_history, self._action_history, cmd).action.detach()
        expected_action_shape = (obs.shape[0], self.config.action_dim)
        if tuple(action.shape) != expected_action_shape:
            raise ValueError(
                "FADA playback action shape mismatch: "
                f"expected={expected_action_shape} observed={tuple(action.shape)}"
            )
        if not bool(torch.isfinite(action).all()):
            raise ValueError("FADA playback policy produced non-finite actions")
        self._action_history = torch.cat((self._action_history[:, 1:], action.unsqueeze(1)), dim=1)
        return action

    def _observation_tensor(self, observation: Any) -> torch.Tensor:
        if isinstance(observation, Mapping):
            if "actor" in observation:
                observation = observation["actor"]
            elif "obs" in observation:
                observation = observation["obs"]
            elif "policy" in observation:
                observation = observation["policy"]
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        return self._validate_projected_observation(
            project_fada_observation_tensor(
                obs,
                observation_contract=self.config.observation_contract,
            )
        )

    def _projected_observation_tensor(self, observation: Any) -> torch.Tensor:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        return self._validate_projected_observation(obs, projected=True)

    def _validate_projected_observation(
        self,
        obs: torch.Tensor,
        *,
        projected: bool = False,
    ) -> torch.Tensor:
        expected_feature_dim = self.config.obs_dim
        if obs.ndim != 2 or obs.shape[1] != expected_feature_dim:
            boundary = "projected observation" if projected else "observation"
            raise ValueError(
                f"FADA playback {boundary} shape mismatch: expected "
                f"[batch, {expected_feature_dim}] observed={tuple(obs.shape)}"
            )
        if not bool(torch.isfinite(obs).all()):
            raise ValueError("FADA playback observation must contain only finite values")
        return obs

    def _advance_observation_history(self, obs: torch.Tensor) -> None:
        batch_size = int(obs.shape[0])
        history_length = self.config.history_length
        if self._observation_history is None:
            self._observation_history = obs.unsqueeze(1).repeat(1, history_length, 1)
            self._action_history = torch.zeros(
                batch_size,
                history_length,
                self.config.action_dim,
                device=self.device,
                dtype=obs.dtype,
            )
            self._pending_reset = None
            return
        if self._observation_history.shape[0] != batch_size:
            raise ValueError(
                "FADA playback observation batch changed without reset: "
                f"expected={self._observation_history.shape[0]} observed={batch_size}"
            )
        self._observation_history = torch.cat(
            (self._observation_history[:, 1:], obs.unsqueeze(1)), dim=1
        )
        if self._pending_reset is not None:
            mask = self._pending_reset
            if bool(mask.any()):
                self._observation_history[mask] = (
                    obs[mask].unsqueeze(1).repeat(1, history_length, 1)
                )
                assert self._action_history is not None
                self._action_history[mask] = 0.0
            self._pending_reset = None
