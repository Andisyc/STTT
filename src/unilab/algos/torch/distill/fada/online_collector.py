"""Online streaming collector for FADA target windows during deployment."""

from __future__ import annotations

from collections import deque
from numbers import Integral

import torch

from unilab.algos.torch.distill.fada.model import FADAArchitectureConfig
from unilab.algos.torch.distill.fada.observation import (
    FADA_G1_STATE_OBSERVATION_CONTRACT,
    FADA_OFFICIAL_COMPACT_OBSERVATION_CONTRACT,
)
from unilab.algos.torch.distill.fada.online_data import (
    FADAInferenceSnapshot,
    FADATrajectoryPair,
)
from unilab.algos.torch.distill.fada.runtime_policy import FADARuntimeSpec


def _validate_snapshot(
    spec: FADARuntimeSpec, snapshot: FADAInferenceSnapshot
) -> None:
    if snapshot.observation_contract != spec.observation_contract:
        raise ValueError(
            "snapshot observation contract does not match the bound runtime spec: "
            f"expected={spec.observation_contract!r} observed={snapshot.observation_contract!r}"
        )
    expected_shapes = {
        "observation_history": (1, spec.history_length, spec.obs_dim),
        "action_history": (1, spec.history_length, spec.action_dim),
        "command": (1, spec.command_dim),
        "nominal_future": (1, spec.prediction_horizon, spec.obs_dim),
        "action_snapshot": (1, spec.action_dim),
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
    if spec.family == "official":
        mask = snapshot.history_valid_mask
        if mask is None:
            raise ValueError("official snapshot collection requires history_valid_mask")
        if mask.shape != (1, spec.history_length):
            raise ValueError(
                f"official history_valid_mask must have shape (1, {spec.history_length})"
            )
        if mask.dtype != torch.bool:
            raise ValueError("official history_valid_mask must be boolean")
        if mask.device != reference.device:
            raise ValueError("official history_valid_mask must share the snapshot device")
    elif snapshot.history_valid_mask is not None:
        raise ValueError("legacy snapshot collection does not use history_valid_mask")


def _spec_from_architecture(config: FADAArchitectureConfig) -> FADARuntimeSpec:
    return FADARuntimeSpec(
        family="legacy",
        obs_dim=config.obs_dim,
        action_dim=config.action_dim,
        command_dim=config.command_dim,
        observation_contract=config.observation_contract,
        history_length=config.history_length,
        prediction_horizon=config.prediction_horizon,
    )


class OnlineTrajectoryCollector:
    """Close saved queries with their K subsequent states, without making labels."""

    def __init__(
        self, config: FADAArchitectureConfig | FADARuntimeSpec, *, window_size: int
    ) -> None:
        if isinstance(config, FADARuntimeSpec):
            spec = config
            supported = (
                FADA_G1_STATE_OBSERVATION_CONTRACT,
                FADA_OFFICIAL_COMPACT_OBSERVATION_CONTRACT,
            )
            if spec.observation_contract not in supported:
                raise ValueError(
                    "trajectory collection requires g1_fada_state_v2 or official_compact_64_v1"
                )
        elif isinstance(config, FADAArchitectureConfig):
            if config.observation_contract != FADA_G1_STATE_OBSERVATION_CONTRACT:
                raise ValueError("trajectory collection requires g1_fada_state_v2")
            spec = _spec_from_architecture(config)
        else:
            raise TypeError(
                "trajectory collection requires FADAArchitectureConfig or FADARuntimeSpec"
            )
        if isinstance(window_size, bool) or not isinstance(window_size, Integral) or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        self._spec = spec
        self.window_size = int(window_size)
        self._samples: deque[FADATrajectoryPair] = deque(maxlen=self.window_size)
        self._pending: FADAInferenceSnapshot | None = None
        self._inflight: deque[tuple[int, FADAInferenceSnapshot]] = deque()
        self._future: deque[torch.Tensor] = deque(maxlen=spec.prediction_horizon)
        self._command: torch.Tensor | None = None
        self._episode_id = 0
        self._timestep = 0
        self._needs_reset = False

    def begin(self, snapshot: FADAInferenceSnapshot) -> None:
        """Save the actual query; command changes invalidate only in-flight futures."""
        if self._needs_reset:
            raise RuntimeError("trajectory collection requires reset after termination")
        if self._pending is not None:
            raise RuntimeError("complete the pending snapshot before beginning another")
        _validate_snapshot(self._spec, snapshot)
        if self._spec.family == "official":
            # Only the velocity command invalidates in-flight futures; continuous
            # sin/cos gait phase evolution inside the 7D command is not a change.
            changed = (
                self._command is not None
                and not torch.equal(snapshot.command[:, :3], self._command[:, :3])
            )
        else:
            changed = self._command is not None and not torch.equal(
                snapshot.command, self._command
            )
        if changed:
            self._inflight.clear()
            self._future.clear()
        self._command = snapshot.command
        self._pending = snapshot

    @torch.no_grad()
    def complete(
        self, next_projected_observation: torch.Tensor, *, done: bool = False
    ) -> FADATrajectoryPair | None:
        """Advance one control step; only complete K-step futures become samples."""
        snapshot = self._pending
        if snapshot is None:
            raise RuntimeError("begin a snapshot before completing a transition")
        if done:
            self._pending = None
            self._inflight.clear()
            self._future.clear()
            self._samples.clear()
            self._command = None
            self._needs_reset = True
            return None
        if tuple(next_projected_observation.shape) != (1, self._spec.obs_dim):
            raise ValueError(
                f"next_projected_observation must have shape (1, {self._spec.obs_dim})"
            )
        observation = next_projected_observation.to(snapshot.action_snapshot)
        if not bool(torch.isfinite(observation).all()):
            raise ValueError("next_projected_observation must be finite")
        self._future.append(observation.detach().clone())
        self._inflight.append((self._timestep, snapshot))
        self._pending = None
        self._timestep += 1
        if len(self._future) < self._spec.prediction_horizon:
            return None
        origin, first = self._inflight.popleft()
        pair = FADATrajectoryPair(
            snapshot=first,
            realized_future=torch.stack(tuple(self._future), dim=1),
            episode_id=self._episode_id,
            timestep=origin,
        )
        self._samples.append(pair)
        return pair

    @property
    def ready(self) -> bool:
        return len(self._samples) == self.window_size

    def get_window(self) -> tuple[FADATrajectoryPair, ...]:
        """Read mature trajectory pairs oldest first, without consuming them."""
        return tuple(self._samples)

    def clear_window(self) -> None:
        """Consume mature pairs without disturbing pending or in-flight queries."""
        self._samples.clear()

    def reset(self) -> None:
        """Start a new episode without bridging any collection state across it."""
        self._pending = None
        self._inflight.clear()
        self._future.clear()
        self._samples.clear()
        self._command = None
        self._episode_id += 1
        self._timestep = 0
        self._needs_reset = False
