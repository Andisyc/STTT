"""One playback history and one optimizer for proprioceptive + spatial TTT."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from unilab.algos.torch.distill.fada.adaptation import FADALoRAConfig
from unilab.algos.torch.distill.fada.online_collector import OnlineTrajectoryCollector
from unilab.algos.torch.distill.fada.online_controller import FADAOnlineAdaptationConfig
from unilab.algos.torch.distill.fada.online_updater import build_expert_samples
from unilab.algos.torch.distill.fada.playback import FADAPlaybackController
from unilab.algos.torch.distill.fada.runtime_policy import get_fada_runtime_spec
from unilab.base.backend.sensors import SensorClock

from .checkpoint import data_contract_from_metadata, load_world_model_checkpoint
from .contract import PROPRIO_LAYOUT
from .odometry import capture_spatial_proprio
from .online_collector import SpatialOnlineCollector
from .online_data import SpatialAdaptationConfig
from .odometry_runtime import create_pose_runtime
from .pose_data import ControlInterval, OnlinePoseRuntime
from .spatial_updater import SpatialOnlineUpdater


@dataclass(frozen=True)
class _PendingQuery:
    clock: SensorClock
    snapshot: Any
    proprio: torch.Tensor
    command: np.ndarray


class SpatialOnlineController:
    """Replay real recent queries, never an imagined future observation rollout."""

    adapter_identity: dict[str, Any]

    def __init__(
        self, policy: Any, lora_config: FADALoRAConfig, *, env: Any,
        online: FADAOnlineAdaptationConfig, spatial: SpatialAdaptationConfig,
        root_dir: str | Path, device: str | torch.device = "cpu",
        playback: Any | None = None,
        action_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        pose_runtime: OnlinePoseRuntime | None = None,
    ) -> None:
        self.env = env
        self.online_config = online
        self.spatial_config = spatial
        self.runtime_spec = get_fada_runtime_spec(policy)
        if playback is None and self.runtime_spec.family == "official":
            raise ValueError("Official spatial adaptation requires source-bound playback")
        if playback is not None and playback.policy is not policy:
            raise ValueError("Playback and spatial adaptation must share one policy")
        path = Path(str(spatial.world_model_checkpoint)).expanduser()
        if not path.is_absolute():
            path = Path(root_dir) / path
        self.world_model_checkpoint_path = path.resolve()
        world_model, payload = load_world_model_checkpoint(path, device=device)
        contract = payload["data_contract"]
        actual_contract = data_contract_from_metadata({
            "obs_dim": 64, "act_dim": int(env.action_space.shape[0]),
            "history_length": world_model.hist_len, "horizon": world_model.horizon,
            "ctrl_dt": float(env.cfg.ctrl_dt),
            "proprio_layout": {key: list(value) for key, value in PROPRIO_LAYOUT.items()},
            "joint_names": list(env.get_actuated_joint_names()),
            "action_scale": np.asarray(env.cfg.control_config.action_scale, dtype=float).reshape(-1).tolist(),
            "default_joint_angles": np.asarray(env.default_angles, dtype=float).reshape(-1).tolist(),
            "action_representation": "env_input", "quaternion_order": "wxyz",
            "coordinate_frame": "reference_heading",
        })
        if contract != actual_contract:
            raise ValueError("Spatial world-model data contract does not match the playback environment")
        self.world_model_data_contract = actual_contract
        self.base_idm = copy.deepcopy(policy.idm).eval().requires_grad_(False)
        self.updater = SpatialOnlineUpdater(
            policy, lora_config, world_model, sigma=online.sigma,
            lr=online.learning_rate, max_grad_norm=online.max_grad_norm,
            space_weight=spatial.space_weight, proprio_weight=spatial.proprio_weight,
            coordinate_weights=spatial.coordinate_weights, runtime_spec=self.runtime_spec,
            action_transform=action_transform,
        )
        self.policy = self.updater.policy
        self.policy.eval()
        self.playback = playback if playback is not None else FADAPlaybackController(self.policy, device=device)
        self.collector = OnlineTrajectoryCollector(self.runtime_spec, window_size=online.window_size)
        self.spatial_collector = SpatialOnlineCollector(
            history_length=world_model.hist_len, horizon=world_model.horizon,
            ctrl_dt=float(env.cfg.ctrl_dt), window_size=online.window_size,
            max_alignment_age_s=spatial.max_alignment_age_s,
        )
        self.pose_runtime = pose_runtime if pose_runtime is not None else create_pose_runtime(env, spatial)
        self._pending_query: _PendingQuery | None = None
        self._transition_id = 0
        self._closed = False
        self._step_count = 0
        self._collection_suspended = False
        self.completed_updates = 0

    def project_observation(self, observation: Any) -> torch.Tensor:
        return self.playback.project_observation(observation)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("SpatialOnlineController is closed")

    def act(self, observation: Any, command: Any) -> torch.Tensor:
        self._require_open()
        projected = self.project_observation(observation)
        if projected.shape[0] != 1:
            raise ValueError("Spatial online adaptation supports exactly one environment")
        if self._collection_suspended:
            return self.playback.act_projected(projected, command)
        if self._pending_query is not None:
            raise RuntimeError("Complete the preceding environment step before another policy query")
        clock = self.pose_runtime.begin_step()
        action, snapshot = self.playback.act_projected_with_snapshot(projected, command)
        self.collector.begin(snapshot)
        commands = np.asarray(self.env.state.info["commands"][0, :3], dtype=np.float64).copy()
        self._pending_query = _PendingQuery(clock, snapshot, capture_spatial_proprio(self.env), commands)
        return action

    def on_env_step(
        self, *, executed_action: Any, next_observation: Any, done: Any | None = None,
    ) -> dict[str, float | int] | None:
        self._require_open()
        if self._collection_suspended:
            return None
        terminal = done is not None and bool(torch.as_tensor(done).any())
        pending, self._pending_query = self._pending_query, None
        if terminal:
            # Autoreset may already have changed the state: never use that pose.
            self.spatial_collector.invalidate_tracking()
            self.pose_runtime.reset()
            self.collector.complete(next_observation, done=True)
            self._collection_suspended = True
            return None
        if pending is None:
            raise RuntimeError("A completed control step requires its original policy query")
        clock = self.pose_runtime.clock
        if clock.sequence_id != pending.clock.sequence_id:
            raise RuntimeError("A control interval crossed a physical reset")
        action = torch.as_tensor(executed_action).detach().reshape(-1).to(device="cpu", dtype=torch.float64)
        if action.shape != (29,) or not torch.isfinite(action).all():
            raise ValueError("executed_action must contain 29 finite env_input values")
        control = ControlInterval(
            clock.sequence_id, self._transition_id, pending.clock.timestamp_ns,
            clock.timestamp_ns, action.numpy().copy(), pending.command,
        )
        self._transition_id += 1
        self.spatial_collector.record_step(pending.snapshot, pending.proprio, control)
        update = self.pose_runtime.complete_step(control)
        if update.invalidated or update.status == "lost":
            self.spatial_collector.invalidate_tracking()
        if not update.invalidated and update.status == "tracking" and update.aligned:
            self.spatial_collector.accept_aligned(update.aligned)
        self.collector.complete(self.project_observation(next_observation))
        self._step_count += 1
        if (self._step_count < self.online_config.warmup_steps
                or self._step_count % self.online_config.update_interval
                or not self.collector.ready):
            return None
        proprio_samples = build_expert_samples(
            self.base_idm, self.collector.get_window(), gamma=self.online_config.gamma,
            runtime_spec=self.runtime_spec,
        )
        spatial_samples = self.spatial_collector.get_window() if self.spatial_collector.ready else ()
        stats = self.updater.update(spatial_samples, proprio_samples)
        stats.update({
            "localization_waiting": int(update.status == "waiting"),
            "localization_lost": int(update.status == "lost"),
            "aligned_control_intervals": len(update.aligned),
            "dropped_control_intervals": update.dropped_intervals,
        })
        self.collector.clear_window()
        if spatial_samples:
            self.spatial_collector.clear_window()
        self.completed_updates += 1
        return stats

    def reset(self, done: Any | None = None) -> None:
        """Reset evidence, retaining LoRA, optimizer and deployment anchor."""
        self._require_open()
        if done is not None and not bool(torch.as_tensor(done).any()):
            return
        self._pending_query = None
        self.pose_runtime.reset()
        self.playback.reset(done)
        self.collector.reset()
        self.spatial_collector.reset()
        self._transition_id = 0
        self._step_count = 0
        self._collection_suspended = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pending_query = None
        self.pose_runtime.close()
