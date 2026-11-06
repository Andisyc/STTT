"""One playback controller owns measured windows, delayed decisions and A/B cadence."""
from __future__ import annotations

import hashlib
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.distill.fada.official_playback import OfficialFADAPlaybackController
from unilab.algos.torch.distill.fada.runtime_policy import FADAFutureQuery
from unilab.algos.torch.sttt.odometry_runtime import create_pose_runtime
from unilab.algos.torch.sttt.path_following.checkpoint import load_response_checkpoint
from unilab.algos.torch.sttt.path_following.collection import WindowCollector
from unilab.algos.torch.sttt.path_following.contract import PathFollowingWindowSample, _integer
from unilab.algos.torch.sttt.path_following.features import response_contract_from_metadata
from unilab.algos.torch.sttt.path_following.online_learning import ActorUpdateContext, PathFollowingUpdater
from unilab.algos.torch.sttt.pose_data import ControlInterval, OnlinePoseRuntime, PoseRuntimeConfig
from unilab.base.backend.sensors import SensorClock


@dataclass(frozen=True)
class PathFollowingRuntimeConfig:
    update_interval_steps: int
    max_cached_queries: int
    max_pending_windows: int
    max_trace_records: int
    pose: PoseRuntimeConfig


def prepare_path_following(
    node: Any, *, cfg: Any, playback_cfg: Any, root_dir: str | Path,
    checkpoint_path: str | Path, policy: Any, profile: Any, spec: Any, device: str,
) -> tuple[PathFollowingUpdater, PathFollowingRuntimeConfig]:
    """Cold validation before environment/renderer/tracker allocation."""
    values = OmegaConf.to_container(node, resolve=True) if OmegaConf.is_config(node) else dict(node)
    required = {"enabled", "checkpoint_path", "learning_config_path", "learning", "update_interval_steps", "max_cached_queries", "max_pending_windows", "max_trace_records", "pose"}
    if not isinstance(values, dict) or set(values) != required or values["enabled"] is not True:
        raise ValueError("use path_following=online for the explicit runtime configuration")
    if playback_cfg.action_mode != "policy" or int(playback_cfg.num_envs) != 1 or playback_cfg.keyboard:
        raise ValueError("path following requires one official policy environment without keyboard control")
    if OmegaConf.select(cfg, "training.task_name") != "G1WalkFlat" or OmegaConf.select(cfg, "training.sim_backend") != "mujoco":
        raise ValueError("path following requires task=g1_walk_flat/mujoco")
    for path in ("interactive.keyboard", "env.commands.heading_command", "demo.figure_eight.enabled"):
        if OmegaConf.select(cfg, path, default=False):
            raise ValueError(f"disable {path} for fixed straight-path following")
    if OmegaConf.select(cfg, "env.commands.resampling_time", default=None) != 0:
        raise ValueError("configure env.commands.resampling_time=0; runtime never rewrites velocity commands")
    counts = {}
    for name in ("update_interval_steps", "max_cached_queries", "max_pending_windows", "max_trace_records"):
        counts[name] = _integer(values[name], name)
        if counts[name] < 1:
            raise ValueError(f"{name} must be positive")
    pose = PoseRuntimeConfig(**values["pose"])
    if pose.odometry_source == "cuvslam" and not pose.sensor.capture_initial_frame:
        raise ValueError("cuVSLAM path following requires pose.sensor.capture_initial_frame=true")
    root = Path(root_dir)
    def resolve(value: Any) -> Path:
        if not value:
            raise ValueError("path following requires explicit B and learning configuration paths")
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else root / path
    loaded = load_response_checkpoint(resolve(values["checkpoint_path"]), device=device)
    with Path(checkpoint_path).open("rb") as handle:
        hasher = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
        digest = hasher.hexdigest()
    data = {
        "window_steps": loaded.contract["data"]["window_steps"], "ctrl_dt": profile.ctrl_dt,
        "history_length": spec.history_length, "prediction_horizon": spec.prediction_horizon,
        "joint_names": list(profile.joint_names), "position_scale": spec.preprocessing["term_scale"]["dof_pos"],
        "velocity_scale": spec.preprocessing["term_scale"]["dof_vel"],
        "tensor_dtype": str(next(policy.parameters()).detach().cpu().numpy().dtype),
    }
    expected = response_contract_from_metadata({"contract": data, "source": {
        "preprocessing": spec.preprocessing, "source_profile": asdict(profile), "checkpoint_sha256": digest,
        "delta_frame": "window_start_horizontal_heading", "delta_units": ["m", "m", "rad"],
    }})
    if loaded.contract != expected:
        raise ValueError("B artifact is incompatible with the actual policy, profile, preprocessing or window interpretation")
    learning = OmegaConf.merge(OmegaConf.load(resolve(values["learning_config_path"])), OmegaConf.create(values["learning"]))
    if not isinstance(learning, DictConfig):
        raise TypeError("A/B learning configuration must be a mapping")
    updater = PathFollowingUpdater(loaded, learning)
    return updater, PathFollowingRuntimeConfig(pose=pose, **counts)


class PathFollowingController:
    """Outermost controller; only the session steps physics. Failures halt until reset.

    Cache queries through alignment, not just through the prediction horizon.
    Candidate evaluation uses the latest *paired historical* decision point with
    its real timestamp; it never attaches an old pose to a current query.
    """

    def __init__(
        self, inner: Any, playback: OfficialFADAPlaybackController, env: Any,
        updater: PathFollowingUpdater, config: PathFollowingRuntimeConfig, *,
        pose_runtime: OnlinePoseRuntime | None = None,
    ) -> None:
        if playback.future_transform is not None:
            raise ValueError("path following must be the sole future transform")
        self.inner, self.playback, self.env = inner, playback, env
        self.policy = playback.policy
        self.updater, self.config = updater, config
        self.poses = pose_runtime if pose_runtime is not None else create_pose_runtime(env, config.pose)
        self.trace: deque[dict[str, Any]] = deque(maxlen=config.max_trace_records)
        self._closed = False
        self._collector: WindowCollector | None = None
        self._queries: OrderedDict[int, tuple[SensorClock, FADAFutureQuery]] = OrderedDict()
        self._pending: list[PathFollowingWindowSample] = []
        self._before: SensorClock | None = None
        self._start: SensorClock | None = None
        self._decision: ActorUpdateContext | None = None
        self._command: np.ndarray | None = None
        self._bound = False
        self._failed = False
        self._terminal = False
        self._steps = 0
        self._last_decision_update = -1
        self.expired_queries = 0
        self.last_update: Any = None
        self._zero = torch.zeros((1, 29), dtype=torch.float32, device=playback.device)
        self._candidate = self._zero
        self._locked = self._zero
        playback.future_transform = self._transform

    @property
    def collector(self) -> WindowCollector:
        if self._collector is None:
            raise RuntimeError("path-following controller must be reset before use")
        return self._collector

    def reset(self, done: Any | None = None) -> None:
        if done is not None and not bool(torch.as_tensor(done).any()):
            return
        if self._closed:
            raise RuntimeError("path-following controller is closed")
        if self._collector is not None:
            self._collector.close()
        self.poses.reset()
        self.inner.reset(done)
        self._collector = WindowCollector(
            self.updater.contract["data"]["window_steps"],
            max_alignment_age_ns=round(self.config.pose.max_alignment_age_s * 1e9),
        )
        self._queries.clear()
        self._pending.clear()
        self._before = self._start = None
        self._decision = None
        self._command = None
        self._bound = self._failed = self._terminal = False
        self._steps = 0
        self._last_decision_update = -1
        self._candidate = self._locked = self._zero
        self.trace.clear()
        self.last_update = None

    def _fail(self, reason: str) -> None:
        self._failed = True
        self.collector.close()
        self._queries.clear()
        self._pending.clear()
        self.trace.append({"event": "halt", "reason": reason, "step": self._steps})

    def _fresh_decision(self, now_ns: int) -> ActorUpdateContext | None:
        if self._decision is None:
            return None
        age = now_ns - self._decision.timestamp_ns
        return self._decision if 0 <= age <= round(self.config.pose.max_pose_age_s * 1e9) else None

    def _select_candidate(self, now_ns: int) -> torch.Tensor:
        decision = self._fresh_decision(now_ns)
        if not self.updater.calibration_complete or decision is None:
            return self._zero
        reference = next(self.updater.actor.parameters())
        with torch.no_grad():
            candidate = self.updater.actor(decision.errors.to(reference), decision.context.to(reference)).to(self._zero)
        if not bool(torch.isfinite(candidate).all()):
            raise ValueError("non-finite A candidate; no fallback or clipping is applied")
        return candidate

    def act(self, observation: Any, command: Any) -> torch.Tensor:
        if self._closed or self._failed or self._terminal:
            raise RuntimeError("path following stopped; an explicit reset is required")
        try:
            if self._before is not None:
                raise RuntimeError("previous query has not been committed by on_env_step")
            before = self.poses.begin_step()
            if self._start is None:
                self._start = before
            if before.sequence_id != self._start.sequence_id:
                raise RuntimeError("physical reset crossed path-following episode")
            velocity = np.asarray(torch.as_tensor(command).detach().cpu(), dtype=np.float64)
            if velocity.shape != (1, 7):
                raise ValueError("path following requires the official [1, 7] command")
            velocity = velocity[0, :3].copy()
            if not np.isfinite(velocity).all() or velocity[0] <= 0 or not np.allclose(velocity[1:], 0., atol=1e-8, rtol=0):
                raise ValueError("path following requires fixed [vx > 0, 0, 0] velocity")
            if self._command is None:
                self._command = velocity
            elif not np.array_equal(velocity, self._command):
                raise ValueError("velocity command changed; runtime never rewrites the command")
            if self._decision is not None and self._fresh_decision(before.timestamp_ns) is None:
                raise RuntimeError("paired localization observation expired")
            if self._decision is None and before.timestamp_ns - self._start.timestamp_ns > round(self.config.pose.max_alignment_age_s * 1e9):
                raise RuntimeError("initial task-time pose is unavailable; reference cannot be re-anchored")
            while self._queries and before.timestamp_ns - next(iter(self._queries.values()))[0].timestamp_ns > round(self.config.pose.max_alignment_age_s * 1e9):
                self._queries.popitem(last=False)
                self.expired_queries += 1
            if len(self._queries) >= self.config.max_cached_queries:
                raise RuntimeError("pending query cache capacity exceeded")
            self._candidate = self._select_candidate(before.timestamp_ns)
            if self.collector.window.at_boundary:
                self.collector.submit_candidate(self._candidate)
            self._before = before
            self.collector.begin_step(before)
            self._locked = self.collector.window.lock()
            return self.inner.act(observation, command)
        except Exception as error:
            self._fail(str(error))
            raise

    def _transform(self, query: FADAFutureQuery) -> torch.Tensor:
        if self._before is None:
            raise RuntimeError("future transform requires a prepared physical control step")
        # Histories roll on the next query; own causal fields until their pose arrives.
        owned = FADAFutureQuery(
            query.observation_history.detach().clone(), query.action_history.detach().clone(),
            query.command.detach().clone(), query.history_valid_mask.detach().clone(),
            query.planner_future.detach().clone(), query.preprocessing, query.profile,
        )
        if self._steps in self._queries:
            raise RuntimeError("one control step attempted multiple Planner queries")
        self._queries[self._steps] = self._before, owned
        return self.collector.transform(query)

    def on_env_step(self, *, executed_action: Any, next_observation: Any, done: Any | None = None) -> dict[str, float | int]:
        if self._closed or self._failed or self._terminal or self._before is None:
            raise RuntimeError("no live path-following query to complete")
        try:
            terminal = done is not None and bool(torch.as_tensor(done).any())
            if terminal:
                self.inner.on_env_step(executed_action=executed_action, next_observation=next_observation, done=done)
                self._terminal = True
                self.collector.close()
                self._queries.clear()
                self._pending.clear()
                self._before = None
                self.trace.append({"event": "terminal", "step": self._steps})
                return {"path_terminal": 1}
            after = self.poses.clock
            before = self._before
            if after.sequence_id != before.sequence_id or after.timestamp_ns <= before.timestamp_ns:
                raise RuntimeError("physical reset or nonadvancing clock crossed control interval")
            action = np.asarray(torch.as_tensor(executed_action).detach().cpu(), dtype=np.float64)
            if action.shape != (1, 29) or self._command is None:
                raise ValueError("path evidence requires actual env-input actions [1,29]")
            control = ControlInterval(before.sequence_id, self._steps, before.timestamp_ns, after.timestamp_ns, action[0].copy(), self._command)
            update = self.poses.complete_step(control)
            samples = self.collector.complete_step(control, update)
            if self.collector.halted:
                raise RuntimeError("localization identity or task-initial evidence invalidated; reset explicitly")
            reference = self.collector.reference
            if reference is not None:
                if not self._bound:
                    self.updater.begin_episode(reference)
                    self._bound = True
                for interval in update.aligned:
                    cached = self._queries.pop(interval.control.transition_id, None)
                    if cached is not None:
                        clock, query = cached
                        point = ActorUpdateContext.from_aligned_query(query, clock, interval.start_pose, reference)
                        if self._decision is None or point.timestamp_ns > self._decision.timestamp_ns:
                            self._decision = point
                if update.aligned:
                    watermark = update.aligned[-1].control.transition_id
                    while self._queries and next(iter(self._queries)) <= watermark:
                        self._queries.popitem(last=False)
                        self.expired_queries += 1
            self._pending.extend(samples)
            if len(self._pending) > self.config.max_pending_windows:
                raise RuntimeError("completed window queue capacity exceeded")
            inner_stats = self.inner.on_env_step(executed_action=executed_action, next_observation=next_observation, done=done)
            self._steps += 1
            self._before = None
            if self._steps % self.config.update_interval_steps == 0:
                decision = self._fresh_decision(after.timestamp_ns)
                if decision is not None and decision.timestamp_ns <= self._last_decision_update:
                    decision = None
                if self._pending or (decision is not None and self.updater.calibration_complete):
                    # Any learner exception halts this controller: never replay a B
                    # batch after its commit followed by an A-stage failure.
                    self.last_update = self.updater.update(tuple(self._pending), decision)
                    self._pending.clear()
                    if decision is not None and self.last_update.a_loss is not None:
                        self._last_decision_update = decision.timestamp_ns
            self._record(after)
            stats = dict(inner_stats or {})
            stats.update({"path_steps": self._steps, "path_windows": self.collector.completed_windows,
                          "path_a_updates": self.updater.actor_updates, "path_b_updates": self.updater.response_updates,
                          "path_pending_windows": len(self._pending), "path_cached_queries": len(self._queries)})
            return stats
        except Exception as error:
            self._fail(str(error))
            raise

    def _record(self, clock: SensorClock) -> None:
        decision = self._decision
        self.trace.append({
            "event": "step", "sequence_id": clock.sequence_id, "timestamp_ns": clock.timestamp_ns,
            "step": self._steps, "window_id": self.collector.window.window_id,
            "measurement_timestamp_ns": None if decision is None else decision.timestamp_ns,
            "observation_age_ns": None if decision is None else clock.timestamp_ns - decision.timestamp_ns,
            "errors_m_rad": None if decision is None else decision.errors[0].cpu().tolist(),
            "candidate_rad": self._candidate[0].cpu().tolist(), "locked_rad": self._locked[0].cpu().tolist(),
            "a_updates": self.updater.actor_updates, "b_updates": self.updater.response_updates,
            "dropped_windows": dict(self.collector.dropped_windows), "expired_queries": self.expired_queries,
        })

    def drain_trace(self) -> tuple[dict[str, Any], ...]:
        """Transfer pending evidence to a per-step consumer before ring overwrite."""
        records = tuple(self.trace)
        self.trace.clear()
        return records

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._collector is not None:
            self._collector.close()
        self._queries.clear()
        self._pending.clear()
        self.playback.future_transform = None
        try:
            self.poses.close()
        finally:
            close = getattr(self.inner, "close", None)
            if callable(close):
                close()
