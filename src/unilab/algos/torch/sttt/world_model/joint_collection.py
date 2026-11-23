"""Persistent simulation streams with real actions, independent resets and live replay."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from omegaconf import OmegaConf

from unilab.algos.torch.distill.fada.official_playback import OfficialFADAPlaybackController
from unilab.algos.torch.distill.fada.playback import FADAPlaybackController
from unilab.algos.torch.distill.fada.runtime_policy import FADAFutureQuery
from unilab.algos.torch.distill.fada.legacy_deployment import LegacyFADAPlaybackProfile
from unilab.algos.torch.sttt.odometry_runtime import BatchSimulationPoseRuntime
from unilab.algos.torch.sttt.path_following.command_reference import command_ranges, sample_command_reference
from unilab.algos.torch.sttt.path_following.intervention import compose_joint_position_residual
from unilab.algos.torch.sttt.path_following.joint_config import CaptureSchedule, TrainingVersions
from unilab.algos.torch.sttt.path_following.joint_data import join_rollout_starts, select_path, select_rollout_start
from unilab.algos.torch.sttt.path_following.latent_rollout import PlannerTrackerRollout
from unilab.algos.torch.sttt.path_following.trajectory import PathState, ReferenceTrajectory
from unilab.visualization.playback_sessions import RslRlPlaybackConfig

from .checkpoint import canonical_contract
from .motion import MOTION_FRAME
from .sequence_collection import _path_delta, proprio_tilt_rad
from .sequence_data import SequenceDatasetWriter, _BOOL, _INT, _WINDOW, _sequence_specs
from .types import RolloutStart


@dataclass(frozen=True)
class CollectionBlock:
    physical_control_steps: int
    starts: RolloutStart | None
    completed_directories: tuple[str, ...]


class _Episode:
    """Preallocated episode storage; snapshots retain reset and terminal evidence."""

    def __init__(self, contract, limit, initial_observation, initial_pose, identity, window_steps, initial_errors):
        self.identity, self.steps, self.windows = identity, 0, 0
        self.window_steps, self.contract = window_steps, contract
        self.poses = [initial_pose]
        self.versions: list[list[int]] = []
        self.errors = np.zeros((limit + 1, 2), dtype=np.float32)
        self.errors[0] = initial_errors
        state, transition = _sequence_specs(contract)
        self.arrays = {}
        for spec, count in ((state, limit + 1), (transition, limit), (_WINDOW, math.ceil(limit / window_steps))):
            for name, tail in spec.items():
                dtype = np.bool_ if name in _BOOL else np.int64 if name in _INT else np.float32
                self.arrays[name] = np.zeros((count, *tail), dtype=dtype)
        self.arrays["is_first"][0] = True
        self.arrays["observations"][0] = initial_observation
        self.arrays["timestamps_ns"][0] = initial_pose.timestamp_ns

    def append(self, *, observation, command, action, source_action, residual, next_observation,
               next_pose, terminal, risk, versions, next_errors):
        index = self.steps
        if not np.array_equal(self.arrays["observations"][index], observation):
            raise ValueError("collector query changed an already-recorded physical observation")
        period_ns = round(self.contract["data"]["ctrl_dt"] * 1e9)
        if abs(next_pose.timestamp_ns - self.poses[-1].timestamp_ns - period_ns) > 1:
            raise ValueError("actual physical control period differs from the source execution profile")
        values = {"commands": command, "actions": action, "source_actions": source_action,
                  "residuals": residual, "terminal": terminal, "risk": [risk], "risk_valid": True,
                  "path_delta": _path_delta(self.poses[-1], next_pose), "motion_valid": not terminal}
        for key, value in values.items():
            self.arrays[key][index] = value
        self.arrays["observations"][index + 1] = next_observation
        self.arrays["timestamps_ns"][index + 1] = next_pose.timestamp_ns
        self.arrays["ego_motion"][index + 1] = values["path_delta"]
        self.arrays["ego_valid"][index + 1] = not terminal
        self.poses.append(next_pose)
        self.versions.append([versions.actor, versions.world_model])
        self.errors[index + 1] = next_errors
        self.steps += 1
        if self.steps % self.window_steps == 0 and not terminal:
            start = self.steps - self.window_steps
            self.arrays["window_start"][self.windows] = start
            self.arrays["window_end"][self.windows] = self.steps
            self.arrays["window_delta"][self.windows] = _path_delta(self.poses[start], next_pose)
            self.arrays["window_valid"][self.windows] = True
            self.windows += 1

    def cost_statistics(self):
        raw = self.sequence()
        active = ~raw["terminal"]
        statistics = {"lateral_m": {"sum_squares": float(np.square(self.errors[1:self.steps + 1, 0][active]).sum()), "count": int(active.sum())}}
        values = {"lateral_velocity_m_s": [], "forward_velocity_m_s": [], "residual_rate_rad_s": []}
        for start, end in zip(raw["window_start"], raw["window_end"]):
            duration = (end - start) * self.contract["data"]["ctrl_dt"]
            values["lateral_velocity_m_s"].append((self.errors[end, 0] - self.errors[start, 0]) / duration)
            heading = self.errors[start:end, 1]
            motion = raw["path_delta"][start:end]
            forward = (np.cos(heading) * motion[:, 0] - np.sin(heading) * motion[:, 1]).sum() / duration
            target = np.linalg.norm(raw["commands"][start:end, :2], axis=1).mean()
            values["forward_velocity_m_s"].append(forward - target)
            previous = raw["residuals"][start - 1] if start else np.zeros(29, np.float32)
            values["residual_rate_rad_s"].extend(((raw["residuals"][start] - previous) / duration).tolist())
        for key, samples in values.items():
            array = np.asarray(samples, dtype=np.float64)
            statistics[key] = {"sum_squares": float(np.square(array).sum()), "count": len(samples)}
        return statistics

    def sequence(self):
        state, transition = _sequence_specs(self.contract)
        return {name: array[:self.steps + 1 if name in state else self.steps if name in transition else self.windows]
                for name, array in self.arrays.items()}


class BatchedSequenceCollector:
    """A persistent session per physical condition, sharing no episode-local state."""

    def __init__(self, cfg, *, root_dir, output_dir, capture: CaptureSchedule, trajectory,
                 num_envs: int, stream_offset: int, seed: int, condition: str,
                 device: str, playback_deps=None, resume_state=None):
        from unilab.visualization.playback_distill_sessions import create_fada_playback_session

        if str(cfg.training.task_name) != "G1WalkFlat" or str(cfg.training.sim_backend) != "mujoco":
            raise ValueError("joint physical collection requires G1WalkFlat/mujoco")
        if OmegaConf.select(cfg, "world_model_collection.pose.odometry_source", default="simulation") != "simulation":
            raise ValueError("joint pretraining collection uses simulation truth")
        if OmegaConf.select(cfg, "online_adaptation.enabled", default=False) or OmegaConf.select(cfg, "path_following.enabled", default=False):
            raise ValueError("joint collection must not start a deployment learner")
        self.capture, self.distribution = capture, trajectory
        self.num_envs, self.stream_offset, self.condition = num_envs, stream_offset, condition
        self.rng = np.random.default_rng(seed)
        self.output = Path(output_dir).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.capture_id = uuid4().hex
        self.seed, self.episode_counter, self.total_steps = seed, 0, 0
        if resume_state is not None:
            expected = {"version", "condition", "num_envs", "stream_offset", "episode_counter", "total_steps", "rng_state"}
            if set(resume_state) != expected or resume_state["version"] != 1 or (resume_state["condition"], resume_state["num_envs"], resume_state["stream_offset"]) != (condition, num_envs, stream_offset):
                raise ValueError("collector resume stream/condition partition changed")
            self.rng.bit_generator.state = resume_state["rng_state"]
            self.episode_counter, self.total_steps = resume_state["episode_counter"], resume_state["total_steps"]
        self.model = self.actor = self.rollout = None
        self.versions = TrainingVersions()
        self._roots: list[RolloutStart | None] = [None] * num_envs
        self._query_observation = self._query_command = None
        policy_path = Path(str(cfg.training.play_checkpoint_path)).expanduser()
        policy_path = policy_path if policy_path.is_absolute() else Path(root_dir) / policy_path
        playback = RslRlPlaybackConfig(task="G1WalkFlat", load_run="-1", checkpoint=None,
            action_mode="policy", policy_obs_mode="actor", algo_log_name="distill", log_root=None,
            num_envs=num_envs, checkpoint_path=str(policy_path), keyboard=False)
        self.session, _, resolved = create_fada_playback_session(playback_cfg=playback, cfg=cfg,
            root_dir=Path(root_dir), device=device, future_transform=self._transform, deps=playback_deps)
        self.poses = None
        try:
            controller = self.session.controller
            if not isinstance(controller, (OfficialFADAPlaybackController, FADAPlaybackController)) or controller.profile is None or controller.spec is None or resolved is None:
                raise TypeError("joint capture requires a checkpoint-bound native Planner–Tracker controller")
            self.controller, self.profile, self.spec = controller, controller.profile, controller.spec
            if self.spec.preprocessing is None or not isinstance(controller.policy, torch.nn.Module):
                raise ValueError("checkpoint preprocessing and a native torch policy are required")
            self.policy, self.preprocessing = controller.policy, dict(self.spec.preprocessing)
            parameter = next(controller.policy.parameters())
            self.device, self.dtype = parameter.device, parameter.dtype
            controller.policy.eval().requires_grad_(False)
            digest_state = hashlib.sha256()
            with Path(resolved).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest_state.update(chunk)
            digest = digest_state.hexdigest()
            self.contract = canonical_contract({
                "data": {"window_steps": capture.window_steps, "ctrl_dt": self.profile.ctrl_dt,
                    "history_length": self.spec.history_length, "prediction_horizon": self.spec.prediction_horizon,
                    "observation_dim": self.spec.obs_dim, "action_dim": self.spec.action_dim,
                    "command_dim": self.spec.command_dim, "observation_contract": self.spec.observation_contract},
                "source": {"checkpoint_sha256": digest, "source_profile": asdict(self.profile),
                    "preprocessing": dict(self.spec.preprocessing),
                    "target_joint_names": list(self.session.env.get_actuated_joint_names()),
                    "target_default_joint_angles": np.asarray(self.session.env.default_angles).reshape(-1).tolist(),
                    "target_action_scale": np.broadcast_to(np.asarray(self.session.env.cfg.control_config.action_scale), (29,)).tolist(),
                    "delta_frame": MOTION_FRAME, "delta_units": ["m", "m", "rad"],
                    "ego_frame": "previous_measurement_horizontal_heading", "risk_tilt_limit_rad": capture.risk_tilt_limit_rad}})
            bounds = capture.command_ranges
            if bounds is None:
                if not isinstance(self.profile, LegacyFADAPlaybackProfile):
                    raise ValueError("set capture.command_ranges explicitly for a source profile without command ranges")
                source_bounds = self.profile.source_profile["environment_config"].get("commands", {}).get("vel_limit")
                if source_bounds is None:
                    raise ValueError("source profile omits command ranges; set supported capture.command_ranges")
                bounds = np.asarray(source_bounds).T
            self.bounds = command_ranges(bounds)
            if not self.bounds[2, 0] < 0 < self.bounds[2, 1]:
                raise ValueError("generic path pretraining requires left/right yaw-rate coverage; set supported capture.command_ranges")
            if self.session.env.cfg.noise_config.level != 0:
                raise ValueError("clean joint capture requires zero source-environment observation noise")
            self.source_config = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
            self._steps = np.zeros(num_envs, dtype=np.int64)
            self._residual = torch.zeros(num_envs, 29, device=self.device, dtype=self.dtype)
            self._residual_versions = np.zeros((num_envs, 2), dtype=np.int64)
            self._command_template = torch.zeros(num_envs, 3, device=self.device, dtype=self.dtype)
            self.amplitude = np.broadcast_to(np.asarray(capture.residual_amplitude_rad, dtype=np.float64), (29,))
            self.session.set_autoreset(False)
            self.session.reset()
            self.poses = BatchSimulationPoseRuntime(self.session.env, num_envs, sequence_offset=stream_offset,
                                                    sequence_stride=capture.num_envs)
            initial = self.poses.sample()
            self._origin = np.empty((num_envs, 3), dtype=np.float64)
            self._path = None
            self._programs: list[dict[str, Any]] = [{} for _ in range(num_envs)]
            self._records: list[_Episode] = []
            self._new_rows(np.arange(num_envs), initial)
        except BaseException:
            self.session.close()
            raise

    def bind(self, model, actor) -> None:
        self.model, self.actor = model, actor
        source = self.contract["source"]
        self.rollout = PlannerTrackerRollout(model, self.policy, self.profile, self.preprocessing,
            target_joint_names=source["target_joint_names"], target_default_joint_angles=source["target_default_joint_angles"],
            target_action_scale=source["target_action_scale"])

    def _physical_observation(self):
        observation = self.session.official_adapter.observation() if self.session.official_adapter is not None else self.session.obs
        return self.controller.project_observation(observation).detach().cpu().numpy().copy()

    def _new_rows(self, indices, poses) -> None:
        reference, program = sample_command_reference(self.distribution.base, self.bounds,
            episode_steps=self.capture.max_episode_steps, horizon_steps=self.capture.steps_per_iteration,
            ctrl_dt=self.profile.ctrl_dt, resampling_s=self.capture.command_resampling_s,
            batch_size=len(indices), rng=self.rng, lateral_range=self.distribution.lateral_range,
            heading_range=self.distribution.heading_range, device=self.device, dtype=self.dtype)
        if self._path is None:
            pose = reference.xy.new_zeros(self.num_envs, 3)
            progress = reference.xy.new_zeros(self.num_envs)
        else:
            xy, speed = self._path.reference.xy.clone(), self._path.reference.speed.clone()
            xy[indices], speed[indices] = reference.xy, reference.speed
            reference = ReferenceTrajectory(xy, speed, self.distribution.base)
            pose, progress = self._path.pose.clone(), self._path.progress.clone()
            pose[indices], progress[indices] = 0., 0.
        projected, _ = reference.project(pose, progress)
        self._path = PathState(reference, pose, projected)
        observations = self._physical_observation()
        initial_errors = self._path.errors().detach().cpu().numpy()
        for position, row in enumerate(indices):
            row = int(row)
            matrix = np.asarray(poses[row].navigation_from_pelvis)
            self._origin[row] = (matrix[0, 3], matrix[1, 3], math.atan2(matrix[1, 0], matrix[0, 0]))
            self._programs[row] = {key: value[position] if key in {"desired_commands", "initial_lateral_m", "initial_heading_rad"} else value
                                    for key, value in program.items()}
            identity = f"{self.capture_id}:{self.stream_offset + row}:{self.episode_counter}"
            self.episode_counter += 1
            record = _Episode(self.contract, self.capture.max_episode_steps, observations[row], poses[row], identity, self.capture.window_steps, initial_errors[row])
            if len(self._records) < self.num_envs:
                self._records.append(record)
            else:
                self._records[row] = record
            self._steps[row] = 0
            self._residual[row] = 0.
            self._residual_versions[row] = 0

    def _update_path(self, poses) -> None:
        matrices = np.stack([pose.navigation_from_pelvis for pose in poses])
        dx, dy = matrices[:, 0, 3] - self._origin[:, 0], matrices[:, 1, 3] - self._origin[:, 1]
        c, s = np.cos(self._origin[:, 2]), np.sin(self._origin[:, 2])
        heading = np.arctan2(matrices[:, 1, 0], matrices[:, 0, 0]) - self._origin[:, 2]
        values = np.stack((c * dx + s * dy, -s * dx + c * dy, (heading + np.pi) % (2 * np.pi) - np.pi), axis=-1)
        pose = torch.as_tensor(values, device=self.device, dtype=self.dtype)
        progress, _ = self._path.reference.project(pose, self._path.progress)
        self._path = PathState(self._path.reference, pose, progress)

    @torch.no_grad()
    def _transform(self, query):
        self._query_observation = query.observation_history[:, -1].detach().cpu().numpy().copy()
        self._query_command = query.command.detach().cpu().numpy().copy()
        indices = np.flatnonzero(self._steps % self.capture.window_steps == 0)
        if len(indices):
            self._residual_versions[indices] = (self.versions.actor, self.versions.world_model)
            index = torch.as_tensor(indices, device=self.device, dtype=torch.long)
            noise = torch.as_tensor(self.rng.uniform(-1., 1., (len(indices), 29)) * self.amplitude,
                                    device=self.device, dtype=self.dtype)
            if self.rollout is not None and self.actor is not None:
                history = self.spec.history_length
                ego = np.zeros((len(indices), history, 3), np.float32)
                valid = np.zeros((len(indices), history, 1), bool)
                age = np.zeros((len(indices), history, 1), np.float32)
                for output, row in enumerate(indices):
                    record = self._records[int(row)]
                    count = min(history, record.steps + 1)
                    source = slice(record.steps + 1 - count, record.steps + 1)
                    ego[output, -count:] = record.arrays["ego_motion"][source]
                    valid[output, -count:] = record.arrays["ego_valid"][source]
                    age[output, -count:] = record.arrays["ego_age_s"][source]
                subset = FADAFutureQuery(query.observation_history[index], query.action_history[index], query.command[index],
                    query.history_valid_mask[index], query.planner_future[index], query.preprocessing, query.profile)
                path = select_path(self._path, index)
                start = self.rollout.encode_start(subset, errors=path.errors(), previous_residual=self._residual[index],
                    ego_motion=torch.as_tensor(ego, device=self.device), ego_valid=torch.as_tensor(valid, device=self.device),
                    ego_age_s=torch.as_tensor(age, device=self.device), path_state=path)
                for output, row in enumerate(indices):
                    if self._roots[int(row)] is None:
                        self._roots[int(row)] = select_rollout_start(start, slice(output, output + 1))
                proposed = self.actor(start.latent.feature(), start.errors, start.command, path.preview()) + noise
                self._residual[index] = proposed.clamp(-self.actor.residual_limit_rad29, self.actor.residual_limit_rad29)
            else:
                self._residual[index] = noise
        scales = query.preprocessing["term_scale"]
        return compose_joint_position_residual(query.planner_future, self._residual, ctrl_dt=self.profile.ctrl_dt,
            position_scale=scales["dof_pos"], velocity_scale=scales["dof_vel"], observation_contract=self.spec.observation_contract)

    def _persist(self, row: int, *, prefix: bool) -> str:
        record = self._records[row]
        suffix = f"_prefix_{record.steps}" if prefix else ""
        directory = self.output / f"episode_{record.identity.replace(':', '_')}{suffix}"
        with SequenceDatasetWriter(directory, self.contract) as writer:
            writer.append(record.sequence())
        reference = select_path(self._path, slice(row, row + 1)).reference
        np.savez_compressed(directory / "reference.npz", xy=reference.xy[0].detach().cpu().numpy(),
                            speed=reference.speed[0].detach().cpu().numpy())
        metadata = {"capture_id": record.identity, "episode_ids": [record.identity], "stream_id": self.stream_offset + row,
            "condition": self.condition, "label_source": "simulation", "closed_as_prefix": prefix,
            "source_config": self.source_config, "reference_program": self._programs[row],
            "trajectory": asdict(self.distribution.base), "versions_actor_wm": record.versions,
            "physical_control_steps": record.steps, "reference_geometry": "reference.npz",
            "cost_statistics": record.cost_statistics(),
            "risk_positive_count": int(np.count_nonzero(record.arrays["risk"][:record.steps] > .5)),
            "pose_receipts": [{"sequence_id": pose.sequence_id, "tracking_epoch": pose.tracking_epoch,
                "timestamp_ns": pose.timestamp_ns, "available_at_ns": pose.timestamp_ns,
                "source_timestamps_ns": pose.source_timestamps_ns, "frame_id": pose.frame_id,
                "valid": pose.valid, "navigation_from_pelvis": np.asarray(pose.navigation_from_pelvis).tolist()} for pose in record.poses]}
        (directory / "capture.json").write_text(json.dumps(metadata, allow_nan=False) + "\n")
        return str(directory)

    def _store(self, replay, row):
        record = self._records[row]
        replay.update_episode(record.identity, record.sequence(), metadata={"stream_id": self.stream_offset + row,
            "condition": self.condition, "versions_actor_wm": record.versions})

    def advance(self, steps: int, replay, *, versions: TrainingVersions) -> CollectionBlock:
        if type(steps) is not int or steps < self.capture.window_steps or steps % self.capture.window_steps:
            raise ValueError("persistent collection advances complete residual windows")
        self.versions, self._roots = versions, [None] * self.num_envs
        completed = []
        for _ in range(steps):
            before = self.poses.sample()
            self._update_path(before)
            command = self._path.command(self._command_template)
            self.session.set_external_command(command[:, :3].cpu().numpy())
            self._query_observation = self._query_command = None
            self.session.step_once()
            after = self.poses.sample()
            self._update_path(after)
            if self._query_observation is None or self._query_command is None or self.controller.last_source_action is None:
                raise RuntimeError("native Planner–Tracker did not emit its physical query/action")
            actions = np.asarray(self.session.env.state.info["current_actions"], dtype=np.float32)
            source_actions = self.controller.last_source_action.detach().cpu().numpy()
            if actions.shape != (self.num_envs, 29) or source_actions.shape != actions.shape:
                raise ValueError("actual target and source actions must retain their [num_envs,29] contract")
            following = self._physical_observation()
            terminated = np.asarray(self.session.env.state.terminated, dtype=bool).reshape(self.num_envs)
            done = terminated | np.asarray(self.session.env.state.truncated, dtype=bool).reshape(self.num_envs)
            done |= self._steps + 1 >= self.capture.max_episode_steps
            residual = self._residual.detach().cpu().numpy()
            errors = self._path.errors().detach().cpu().numpy()
            for row, record in enumerate(self._records):
                risk = bool(terminated[row]) or proprio_tilt_rad(following[row], observation_contract=self.spec.observation_contract) > self.capture.risk_tilt_limit_rad
                record.append(observation=self._query_observation[row], command=self._query_command[row], action=actions[row],
                    source_action=source_actions[row], residual=residual[row], next_observation=following[row], next_pose=after[row],
                    terminal=bool(done[row]), risk=float(risk),
                    versions=TrainingVersions(actor=int(self._residual_versions[row, 0]), world_model=int(self._residual_versions[row, 1])),
                    next_errors=errors[row])
            self._steps += 1
            self.total_steps += self.num_envs
            if done.any():
                indices = np.flatnonzero(done)
                for row in indices:
                    record = self._records[int(row)]
                    self._store(replay, int(row))
                    completed.append(self._persist(int(row), prefix=False))
                self.poses.reset(indices)
                self.session.reset_rows(indices)
                self._new_rows(indices, self.poses.sample())
        for row, record in enumerate(self._records):
            if record.steps:
                self._store(replay, row)
        if self.actor is not None and any(start is None for start in self._roots):
            raise RuntimeError("collection block did not visit an actor decision in every stream")
        starts = join_rollout_starts([start for start in self._roots if start is not None]) if self.actor is not None else None
        return CollectionBlock(steps * self.num_envs, starts, tuple(completed))

    def flush(self, replay) -> tuple[str, ...]:
        directories = []
        for row, record in enumerate(self._records):
            if record.steps:
                self._store(replay, row)
                directories.append(self._persist(row, prefix=True))
        return tuple(directories)

    def state_dict(self):
        return {"version": 1, "condition": self.condition, "num_envs": self.num_envs,
                "stream_offset": self.stream_offset, "episode_counter": self.episode_counter,
                "total_steps": self.total_steps, "rng_state": self.rng.bit_generator.state}

    def close(self) -> None:
        try:
            if self.poses is not None:
                self.poses.close()
        finally:
            self.session.close()
