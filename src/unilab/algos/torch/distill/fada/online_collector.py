"""Online streaming collector for FADA target windows during deployment."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from numbers import Integral

import numpy as np
import torch

from unilab.algos.torch.distill.fada.model import FADAArchitectureConfig
from unilab.algos.torch.distill.fada.observation import (
    FADA_G1_STATE_OBSERVATION_CONTRACT,
    fada_g1_joint_position_offsets,
)
from unilab.algos.torch.distill.fada.online_data import FADAInferenceSnapshot, FADAOnlineSample


class OnlineSnapshotCollector:
    """Close each inference snapshot one step later and retain fixed-label samples.

    ``begin`` precedes physics, ``complete`` consumes the next projected state.
    Termination drops the pending transition and requires an explicit reset.
    Reading the window never drains it; this collector does not run an optimizer.
    """

    def __init__(
        self,
        config: FADAArchitectureConfig,
        *,
        window_size: int,
        gamma: float,
        action_scale: float | Sequence[float] | np.ndarray | torch.Tensor,
    ) -> None:
        if config.observation_contract != FADA_G1_STATE_OBSERVATION_CONTRACT:
            raise ValueError("snapshot collection requires g1_fada_state_v2")
        if isinstance(window_size, bool) or not isinstance(window_size, Integral) or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        if not math.isfinite(gamma) or gamma < 0.0:
            raise ValueError("gamma must be finite and non-negative")
        scale = torch.as_tensor(action_scale, dtype=torch.float64).detach()
        if scale.ndim == 0:
            scale = scale.expand(config.action_dim)
        elif tuple(scale.shape) != (config.action_dim,):
            raise ValueError("action_scale must be a scalar or a 29-element vector")
        if not bool(torch.isfinite(scale).all()) or bool((scale == 0).any()):
            raise ValueError("action_scale must be finite and nonzero")

        self._config = config
        self.window_size = int(window_size)
        self.gamma = float(gamma)
        self._action_scale = scale.reshape(1, config.action_dim).clone()
        self._samples: deque[FADAOnlineSample] = deque(maxlen=self.window_size)
        self._pending: FADAInferenceSnapshot | None = None
        self._episode_id = 0
        self._timestep = 0
        self._needs_reset = False

    def begin(self, snapshot: FADAInferenceSnapshot) -> None:
        """Retain the actual query, never a recomputed action or future."""
        if self._needs_reset:
            raise RuntimeError("snapshot collection requires reset after termination")
        if self._pending is not None:
            raise RuntimeError("complete the pending snapshot before beginning another")
        expected_shapes = {
            "observation_history": (1, self._config.history_length, self._config.obs_dim),
            "action_history": (1, self._config.history_length, self._config.action_dim),
            "command": (1, self._config.command_dim),
            "nominal_future": (1, self._config.prediction_horizon, self._config.obs_dim),
            "action_snapshot": (1, self._config.action_dim),
        }
        reference = snapshot.action_snapshot
        for name, shape in expected_shapes.items():
            value = getattr(snapshot, name)
            if tuple(value.shape) != shape:
                raise ValueError(f"snapshot {name} must have shape {shape}")
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError("snapshot tensors must share the action device and dtype")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"snapshot {name} must be finite")
        self._action_scale = self._action_scale.to(
            device=reference.device, dtype=reference.dtype
        )
        if not bool(torch.isfinite(self._action_scale).all()) or bool((self._action_scale == 0).any()):
            raise ValueError("action_scale is not representable in the snapshot dtype")
        self._pending = snapshot

    @torch.no_grad()
    def complete(
        self, next_projected_observation: torch.Tensor, *, done: bool = False
    ) -> FADAOnlineSample | None:
        """Close t with the observation at t+1; terminal observations are ignored."""
        snapshot = self._pending
        if snapshot is None:
            raise RuntimeError("begin a snapshot before completing a transition")
        if done:
            self._pending = None
            self._samples.clear()
            self._needs_reset = True
            return None
        if tuple(next_projected_observation.shape) != (1, self._config.obs_dim):
            raise ValueError("next_projected_observation must have shape (1, 66)")
        observation = next_projected_observation.to(
            device=snapshot.action_snapshot.device,
            dtype=snapshot.action_snapshot.dtype,
        )
        if not bool(torch.isfinite(observation).all()):
            raise ValueError("next_projected_observation must be finite")
        real_offsets = fada_g1_joint_position_offsets(
            observation, observation_contract=self._config.observation_contract
        )
        error = snapshot.nominal_joint_positions - real_offsets
        target = snapshot.action_snapshot + self.gamma * error / self._action_scale
        if not bool(torch.isfinite(target).all()):
            raise ValueError("snapshot target action must be finite")
        sample = FADAOnlineSample(
            snapshot=snapshot,
            joint_position_error=error,
            target_action=target,
            episode_id=self._episode_id,
            timestep=self._timestep,
        )
        self._samples.append(sample)
        self._pending = None
        self._timestep += 1
        return sample

    @property
    def ready(self) -> bool:
        """Only a full T-sample window is eligible for later optimization."""
        return len(self._samples) == self.window_size

    def get_window(self) -> tuple[FADAOnlineSample, ...]:
        """Return samples oldest first without clearing or rebuilding labels."""
        return tuple(self._samples)

    def clear_window(self) -> None:
        """Consume completed samples without resetting inference or episode state."""
        self._samples.clear()

    def reset(self) -> None:
        """Start a new episode without carrying pending or completed samples."""
        self._pending = None
        self._samples.clear()
        self._episode_id += 1
        self._timestep = 0
        self._needs_reset = False
