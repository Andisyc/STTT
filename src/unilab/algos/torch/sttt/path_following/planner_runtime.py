"""Dual LoRA on the existing playback/pose/window lifecycle, without A or FiLM."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from unilab.algos.torch.sttt.odometry_runtime import create_pose_runtime
from unilab.algos.torch.sttt.path_following.latent_rollout import PlannerTrackerRollout
from unilab.algos.torch.sttt.path_following.latent_runtime import _ReceivingPoses, _canonical, _digest
from unilab.algos.torch.sttt.path_following.planner_losses import timed_reference
from unilab.algos.torch.sttt.path_following.runtime import PathFollowingController, PathFollowingRuntimeConfig
from unilab.algos.torch.sttt.path_following.trajectory import MeasuredTrajectory, ReferenceTrajectory
from unilab.algos.torch.sttt.pose_data import PoseRuntimeConfig
from unilab.algos.torch.sttt.world_model.checkpoint import LoadedWorldModel, load_world_model
from unilab.algos.torch.sttt.world_model.motion import MOTION_FRAME
from unilab.algos.torch.sttt.world_model.planner_online import PlannerLoRAUpdater, PlannerOnlineConfig
from unilab.algos.torch.sttt.world_model.sequence_collection import CausalEgoHistory


@dataclass(frozen=True)
class PlannerRuntimeSetup:
    loaded: LoadedWorldModel
    learning: PlannerOnlineConfig
    runtime: PathFollowingRuntimeConfig
    resume_checkpoint_path: Path | None
    save_checkpoint_path: Path | None

    def bind(self, playback: Any, env: Any) -> PlannerLoRAUpdater:
        source = self.loaded.contract["source"]
        actual = {
            "target_joint_names": list(env.get_actuated_joint_names()),
            "target_default_joint_angles": np.asarray(env.default_angles).reshape(-1).tolist(),
            "target_action_scale": np.broadcast_to(np.asarray(env.cfg.control_config.action_scale), (29,)).tolist(),
        }
        for name, value in actual.items():
            if _canonical(source[name]) != _canonical(value):
                raise ValueError(f"WM deployment differs from collected {name}")
        if float(env.cfg.ctrl_dt) != self.loaded.contract["data"]["ctrl_dt"]:
            raise ValueError("WM control period differs from deployment")

        def rollout_factory(model):
            return PlannerTrackerRollout(model, playback.policy, playback.profile, source["preprocessing"],
                                         target_joint_names=actual["target_joint_names"],
                                         target_default_joint_angles=actual["target_default_joint_angles"],
                                         target_action_scale=actual["target_action_scale"])

        if self.resume_checkpoint_path is None:
            return PlannerLoRAUpdater(self.loaded, rollout_factory(self.loaded.model), self.learning)
        updater = PlannerLoRAUpdater.restore(self.resume_checkpoint_path, rollout_factory,
            device=str(next(self.loaded.model.parameters()).device), expected_contract=self.loaded.contract)
        if updater.config != self.learning:
            raise ValueError("online resume changed the dual LoRA learning configuration")
        return updater


def prepare_planner_path_following(node: Any, *, cfg: Any, playback_cfg: Any, root_dir: str | Path,
                                   checkpoint_path: str | Path, policy: Any, profile: Any,
                                   spec: Any, device: str) -> PlannerRuntimeSetup:
    values = OmegaConf.to_container(node, resolve=True) if OmegaConf.is_config(node) else dict(node)
    required = {"enabled", "mode", "checkpoint_path", "learning_config_path", "learning", "update_interval_steps",
                "max_cached_queries", "max_pending_windows", "max_trace_records", "pose", "resume_checkpoint_path", "save_checkpoint_path"}
    if not isinstance(values, dict) or set(values) != required or values["enabled"] is not True or values["mode"] != "planner_wm_lora":
        raise ValueError("use path_following=planner_wm_lora; A/FiLM configuration is not accepted")
    if playback_cfg.action_mode != "policy" or int(playback_cfg.num_envs) != 1 or playback_cfg.keyboard:
        raise ValueError("dual LoRA requires one policy environment without keyboard control")
    if OmegaConf.select(cfg, "training.task_name") != "G1WalkFlat" or OmegaConf.select(cfg, "training.sim_backend") != "mujoco":
        raise ValueError("dual LoRA requires task=g1_walk_flat/mujoco")
    for path in ("interactive.keyboard", "env.commands.heading_command", "demo.figure_eight.enabled", "online_adaptation.enabled"):
        if OmegaConf.select(cfg, path, default=False):
            raise ValueError(f"disable {path}; Planner LoRA is the only policy adaptation owner")
    if OmegaConf.select(cfg, "env.commands.resampling_time") != 0:
        raise ValueError("the task-time reference owns nominal commands; disable command resampling")
    root = Path(root_dir)

    def resolve(value):
        if value is None:
            return None
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else root / path

    model_path, learning_path = resolve(values["checkpoint_path"]), resolve(values["learning_config_path"])
    if model_path is None or learning_path is None:
        raise ValueError("provide the independent WM checkpoint and dual LoRA learning config")
    loaded = load_world_model(model_path, device=device)
    with Path(checkpoint_path).open("rb") as handle:
        digest = _digest(handle)
    source, data = loaded.contract["source"], loaded.contract["data"]
    expected_source = {"checkpoint_sha256": digest, "source_profile": _canonical(asdict(profile)),
                       "preprocessing": _canonical(dict(spec.preprocessing)), "delta_frame": MOTION_FRAME,
                       "delta_units": ["m", "m", "rad"], "ego_frame": "previous_measurement_horizontal_heading"}
    for name, expected in expected_source.items():
        if _canonical(source.get(name)) != expected:
            raise ValueError(f"WM artifact is incompatible with actual {name}")
    for name, expected in {"ctrl_dt": profile.ctrl_dt, "history_length": spec.history_length,
                           "prediction_horizon": spec.prediction_horizon, "observation_dim": spec.obs_dim,
                           "action_dim": spec.action_dim}.items():
        if data.get(name) != expected:
            raise ValueError(f"WM artifact is incompatible with actual {name}")
    if data.get("command_dim", 7) != spec.command_dim or data.get("observation_contract", "official_compact_64_v1") != spec.observation_contract:
        raise ValueError("WM observation/command identity differs from the policy")
    counts = {name: values[name] for name in
              ("update_interval_steps", "max_cached_queries", "max_pending_windows", "max_trace_records")}
    if any(type(value) is not int or value < 1 for value in counts.values()):
        raise ValueError("runtime capacities and update cadence must be positive integers")
    learning = OmegaConf.to_container(OmegaConf.merge(OmegaConf.load(learning_path), OmegaConf.create(values["learning"])), resolve=True)
    if not isinstance(learning, dict) or set(learning) != {"online"} or not isinstance(learning["online"], dict):
        raise ValueError("dual LoRA learning config requires an online mapping")
    pose = PoseRuntimeConfig(**values["pose"])
    if pose.odometry_source == "cuvslam" and not pose.sensor.capture_initial_frame:
        raise ValueError("cuVSLAM requires initial-frame capture")
    resume, save = resolve(values["resume_checkpoint_path"]), resolve(values["save_checkpoint_path"])
    if save is not None and save.exists() and save != resume:
        raise FileExistsError("select a new online output or explicitly resume that checkpoint")
    return PlannerRuntimeSetup(loaded, PlannerOnlineConfig.from_mapping(learning["online"]),
                               PathFollowingRuntimeConfig(pose=pose, **counts), resume, save)


class PlannerPathFollowingController(PathFollowingController):
    """The session still owns every physical step; learning never queries the simulator."""

    def __init__(self, inner: Any, playback: Any, env: Any, updater: PlannerLoRAUpdater,
                 config: PathFollowingRuntimeConfig, *, pose_runtime: Any = None,
                 save_checkpoint_path: Path | None = None) -> None:
        self.ego_history = CausalEgoHistory(max_records=config.max_cached_queries)
        self.save_checkpoint_path = save_checkpoint_path
        parameter = next(updater.model.parameters())
        self.reference_geometry = ReferenceTrajectory.generate(updater.config.trajectory,
                                                               device=parameter.device, dtype=parameter.dtype)
        source = pose_runtime if pose_runtime is not None else create_pose_runtime(env, config.pose)
        super().__init__(inner, playback, env, updater, config,
            pose_runtime=_ReceivingPoses(source, self._receive_poses), allow_command_changes=True,
            reference_factory=lambda initial: MeasuredTrajectory(initial, updater.config.trajectory,
                reference=self.reference_geometry, device=parameter.device, dtype=parameter.dtype))
        updater.ego_history = self.ego_history
        self._learning_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="planner-wm-learning")
        self._learning_future: Future | None = None
        self._submitted_timestamp = -1

    def _finish_learning(self, *, wait: bool = False) -> None:
        future = self._learning_future
        if future is None or (not wait and not future.done()):
            return
        self.last_update = future.result()
        self._learning_future = None
        self.updater.publish()
        if self.last_update.planner_loss is not None:
            self._last_decision_update = self._submitted_timestamp
        self.trace.append({"event": "learning_complete", "step": self._steps,
            **self.updater.update_counters(), "wm_loss": self.last_update.wm_loss,
            "planner_loss": self.last_update.planner_loss, "body_loss": self.last_update.body_loss,
            "motion_loss": self.last_update.motion_loss, "elapsed_s": self.last_update.elapsed_s})

    def _learning_ready(self) -> bool:
        return self._learning_future is None

    def _run_learning(self, samples, decision):
        prepared = self.updater.prepare_update(samples, decision)
        self._submitted_timestamp = -1 if decision is None else decision.timestamp_ns
        self._learning_future = self._learning_executor.submit(self.updater.update_prepared, prepared)
        return None

    def _select_candidate(self, now_ns: int) -> torch.Tensor:
        if self.collector.window.at_boundary:
            self._finish_learning()
        return self._zero

    def _command_for_step(self, command: Any) -> torch.Tensor:
        if self._start is None:
            raise RuntimeError("nominal command requires the physical task clock")
        elapsed = (self.poses.clock.timestamp_ns - self._start.timestamp_ns) / 1e9
        with torch.no_grad():
            _, velocity = timed_reference(self.reference_geometry, elapsed, 1, self.updater.ctrl_dt)
            template = torch.as_tensor(command, device=velocity.device, dtype=velocity.dtype)
            return torch.cat((velocity[:, 0], template[:, 3:]), -1)

    def _receive_poses(self, update: Any, now_ns: int) -> None:
        self.ego_history.add(update.received_poses)
        record = self.updater.records.get(self._steps)
        if record is not None:
            ego, valid, age = self.ego_history.at(now_ns)
            record.update(next_ego_motion=ego, next_ego_valid=valid, next_ego_age_s=age)

    def on_env_step(self, *, executed_action: Any, next_observation: Any, done: Any = None) -> dict[str, float | int]:
        if self._closed or self._failed or self._terminal or self._before is None:
            raise RuntimeError("no live Planner query to complete")
        try:
            if done is None or not bool(torch.as_tensor(done).any()):
                before, query = self._queries[self._steps]
                compact = self.playback.project_observation(next_observation).detach()
                source_action = self.playback.last_source_action
                action = torch.as_tensor(executed_action, device=query.command.device, dtype=query.command.dtype).detach()
                if source_action is None or action.shape != (1, 29) or compact.shape != (1, self.updater.model.config.obs_dim):
                    raise ValueError("WM records require actual ENV/SOURCE commands and aligned next proprioception")
                if not bool(torch.isfinite(action).all()) or not bool(torch.isfinite(compact).all()):
                    raise ValueError("WM records require finite measured transitions")
                ego, valid, age = self.ego_history.at(before.timestamp_ns)
                self.updater.record_transition(self._steps, {
                    "observation": query.observation_history[0, -1].cpu().numpy().copy(),
                    "next_observation": compact[0].cpu().numpy().copy(), "action": action[0].cpu().numpy().copy(),
                    "source_action": source_action[0].detach().cpu().numpy().copy(),
                    "residual": self._locked[0].cpu().numpy().copy(), "command": query.command[0].cpu().numpy().copy(),
                    "ego_motion": ego, "ego_valid": valid, "ego_age_s": age,
                }, capacity=self.config.max_cached_queries)
            return super().on_env_step(executed_action=executed_action, next_observation=next_observation, done=done)
        except Exception as error:
            if not self._failed:
                self._fail(str(error))
            raise

    def reset(self, done: Any = None) -> None:
        if done is not None and not bool(torch.as_tensor(done).any()):
            return
        self._finish_learning(wait=True)
        self.ego_history.clear()
        self.updater.discard_pending()
        super().reset(done)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._finish_learning(wait=True)
            if self.save_checkpoint_path is not None and not self._failed:
                self.updater.save(self.save_checkpoint_path)
        finally:
            self._learning_executor.shutdown(wait=True, cancel_futures=True)
            super().close()
