"""Every-step official playback capture with causal inputs and delayed pose labels."""
from __future__ import annotations

import hashlib
import json
from collections import deque
import math
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.distill.fada.official_playback import OfficialFADAPlaybackController
from unilab.algos.torch.sttt.odometry_runtime import create_pose_runtime
from unilab.algos.torch.sttt.path_following.intervention import compose_joint_position_residual
from unilab.algos.torch.sttt.path_following.reference import FixedStraightPath, pose_identity, pose_matrix
from unilab.algos.torch.sttt.pose_data import ControlInterval, PoseRuntimeConfig
from unilab.algos.torch.sttt.world_model.observation_augmentation import ObservationAugmentationConfig, PoseInputAugmenter
from unilab.algos.torch.sttt.world_model.sequence_data import SequenceDatasetWriter, _arrays


def _wrap(value):
    return (value + np.pi) % (2 * np.pi) - np.pi


def _path_delta(start, end, reference) -> np.ndarray:
    if pose_identity(start) != pose_identity(end) or pose_identity(start) != (reference.sequence_id, reference.tracking_epoch, reference.frame_id):
        raise ValueError("window endpoints must share the fixed path localization identity")
    if end.timestamp_ns <= start.timestamp_ns:
        raise ValueError("window endpoints must increase in physical time")
    a, b = pose_matrix(start), pose_matrix(end)
    dx, dy = b[:2, 3] - a[:2, 3]
    c, s = math.cos(reference.heading_rad), math.sin(reference.heading_rad)
    yaw = _wrap(math.atan2(b[1, 0], b[0, 0]) - math.atan2(a[1, 0], a[0, 0]))
    return np.asarray([c*dx + s*dy, -s*dx + c*dy, yaw], dtype=np.float32)


def _record_sequence(controls, records: Mapping[int, dict]) -> dict[str, np.ndarray]:
    if not controls:
        raise ValueError("sequence requires executed controls")
    rows = [records[c.transition_id] for c in controls]
    for i, control in enumerate(controls):
        if i and (control.sequence_id != controls[0].sequence_id or control.start_ns != controls[i-1].end_ns or control.transition_id != controls[i-1].transition_id + 1):
            raise ValueError("records must form one physically contiguous episode")
        if not np.allclose(rows[i]["action"], control.action, atol=1e-7, rtol=0) or not np.allclose(rows[i]["command"][:3], control.command, atol=1e-7, rtol=0):
            raise ValueError("record does not match its actual executed interval")
        if i and not np.array_equal(rows[i-1]["next_observation"], rows[i]["observation"]):
            raise ValueError("adjacent records disagree on the same physical observation")
    n = len(rows)
    last = rows[-1]
    result = {
        "observations": np.asarray([r["observation"] for r in rows] + [last["next_observation"]], dtype=np.float32),
        "actions": np.asarray([r["action"] for r in rows], dtype=np.float32),
        "source_actions": np.asarray([r["source_action"] for r in rows], dtype=np.float32),
        "commands": np.asarray([r["command"] for r in rows], dtype=np.float32),
        "is_first": np.asarray([True] + [False]*n, dtype=bool),
        "ego_motion": np.asarray([r["ego_motion"] for r in rows] + [last.get("next_ego_motion", np.zeros(3))], dtype=np.float32),
        "ego_valid": np.asarray([r["ego_valid"] for r in rows] + [last.get("next_ego_valid", False)], dtype=bool).reshape(n+1, 1),
        "ego_age_s": np.asarray([r["ego_age_s"] for r in rows] + [last.get("next_ego_age_s", 0.)], dtype=np.float32).reshape(n+1, 1),
        "errors": np.zeros((n+1, 2), dtype=np.float32),
        "path_delta": np.zeros((n, 3), dtype=np.float32), "motion_valid": np.zeros(n, dtype=bool),
        "risk": np.asarray([r.get("risk", 0.) for r in rows], dtype=np.float32).reshape(n, 1),
        "risk_valid": np.asarray([r.get("risk_valid", False) for r in rows], dtype=bool),
        "terminal": np.asarray([r.get("terminal", False) for r in rows], dtype=bool),
        "timestamps_ns": np.asarray([c.start_ns for c in controls] + [controls[-1].end_ns], dtype=np.int64),
        "window_start": np.empty(0, dtype=np.int64), "window_end": np.empty(0, dtype=np.int64),
        "window_delta": np.empty((0, 3), dtype=np.float32), "window_valid": np.empty(0, dtype=bool),
    }
    return result


def window_sequence(sample, records: Mapping[int, dict]) -> dict[str, np.ndarray]:
    """Assemble online evidence; only measured endpoints supervise sparse motion.

    Records contain causal ego inputs captured at query time, never reconstructed
    from delayed labels. Interior errors remain zero and are NOT dense labels.
    """
    result = _record_sequence(sample.controls, records)
    n = len(sample.controls)
    if sample.start_pose.timestamp_ns != sample.controls[0].start_ns or sample.end_pose.timestamp_ns != sample.controls[-1].end_ns:
        raise ValueError("window pose labels do not match executed physical endpoints")
    result["errors"][0] = sample.reference.errors(sample.start_pose)
    result["errors"][-1] = sample.reference.errors(sample.end_pose)
    result["window_start"] = np.asarray([0], dtype=np.int64)
    result["window_end"] = np.asarray([n], dtype=np.int64)
    result["window_delta"] = _path_delta(sample.start_pose, sample.end_pose, sample.reference)[None]
    result["window_valid"] = np.ones(1, dtype=bool)
    if result["terminal"].any():
        raise ValueError("completed online windows must not contain terminal transitions")
    return _arrays(result)


class CausalEgoHistory:
    """Bounded raw receive history, never interpolated or future-aligned inputs."""

    def __init__(self, max_records: int = 512):
        if isinstance(max_records, bool) or not isinstance(max_records, int) or max_records < 2:
            raise ValueError("max_records must be an integer >= 2")
        self.receipts = deque(maxlen=max_records)

    def clear(self) -> None:
        self.receipts.clear()

    def add(self, received_poses) -> None:
        for received in received_poses:
            pose = received.pose
            if received.available_at_ns < pose.timestamp_ns or pose.timestamp_ns < 0:
                raise ValueError("raw receive time must follow nonnegative capture time")
            if self.receipts:
                previous = self.receipts[-1]
                if pose_identity(previous.pose) != pose_identity(pose):
                    self.clear()
                elif received.available_at_ns < previous.available_at_ns:
                    raise ValueError("raw receive times must not move backwards")
            self.receipts.append(received)

    def at(self, timestamp_ns: int) -> tuple[np.ndarray, bool, float]:
        if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, (int, np.integer)) or timestamp_ns < 0:
            raise ValueError("decision time must be nonnegative integer nanoseconds")
        available = [r.pose for r in self.receipts if r.available_at_ns <= timestamp_ns and r.pose.timestamp_ns <= timestamp_ns]
        if not available or not available[-1].valid:
            return np.zeros(3, dtype=np.float32), False, 0.
        end = max(available, key=lambda p: p.timestamp_ns)
        age = (timestamp_ns-end.timestamp_ns)/1e9
        prior = next((p for p in sorted(available, key=lambda p: p.timestamp_ns, reverse=True)
                      if p.valid and pose_identity(p) == pose_identity(end) and p.timestamp_ns < end.timestamp_ns), None)
        if prior is None:
            return np.zeros(3, dtype=np.float32), False, age
        return _path_delta(prior, end, FixedStraightPath.from_initial_pose(prior)), True, age


class _CaptureController:
    """Observe the real source action return, before the deployment mapping."""

    def __init__(self, playback):
        self.playback = playback
        self.source_action = None

    def act(self, mapping, command):
        action = self.playback.act(mapping, command)
        self.source_action = action[0].detach().cpu().numpy().copy()
        return action

    def reset(self, done=None):
        self.source_action = None
        self.playback.reset(done)

    def on_env_step(self, **kwargs):
        return self.playback.on_env_step(**kwargs)


def proprio_tilt_rad(observation: np.ndarray) -> float:
    gravity = np.asarray(observation, dtype=np.float64)[-3:]
    norm = float(np.linalg.norm(gravity))
    if gravity.shape != (3,) or not np.isfinite(gravity).all() or norm <= 1e-8:
        raise ValueError("risk labels require a valid projected gravity vector")
    return float(np.arccos(np.clip(-gravity[2] / norm, -1., 1.)))


def collect_world_model_sequences(cfg, *, root_dir, output_dir, num_steps, window_steps,
                                  residual_amplitude_rad, seed, risk_tilt_limit_rad,
                                  device=None, playback_deps=None, observation_augmentation=None) -> dict[str, Any]:
    """Capture one real episode without changing existing B collection or training."""
    from unilab.visualization.playback_distill_sessions import create_fada_playback_session
    from unilab.visualization.playback_sessions import RslRlPlaybackConfig

    for name, value in (("num_steps", num_steps), ("window_steps", window_steps)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(risk_tilt_limit_rad, bool) or not isinstance(risk_tilt_limit_rad, (int, float)) or not math.isfinite(risk_tilt_limit_rad) or not 0 < risk_tilt_limit_rad <= math.pi:
        raise ValueError("risk_tilt_limit_rad must be an explicit angle in (0, pi]")
    amplitude = np.asarray(residual_amplitude_rad, dtype=np.float64)
    if isinstance(residual_amplitude_rad, (bool, np.bool_)) or amplitude.shape not in ((), (29,)) or not np.isfinite(amplitude).all() or np.any(amplitude < 0):
        raise ValueError("residual_amplitude_rad must be an explicit finite nonnegative scalar or per-joint 29-vector")
    if not isinstance(cfg, DictConfig):
        raise TypeError("collection requires a composed Hydra DictConfig")
    if str(cfg.training.task_name) != "G1WalkFlat" or str(cfg.training.sim_backend) != "mujoco":
        raise ValueError("world-model capture requires G1WalkFlat/mujoco")
    if OmegaConf.select(cfg, "path_following.enabled", default=False) or OmegaConf.select(cfg, "online_adaptation.enabled", default=False):
        raise ValueError("offline capture cannot run simultaneous path following or Tracker adaptation")
    if OmegaConf.select(cfg, "interactive.keyboard", default=False) or OmegaConf.select(cfg, "demo.figure_eight.enabled", default=False):
        raise ValueError("capture requires an explicit noninteractive straight-path command")
    raw_pose = OmegaConf.select(cfg, "world_model_collection.pose", default={})
    pose_kwargs = OmegaConf.to_container(raw_pose, resolve=True) if OmegaConf.is_config(raw_pose) else dict(raw_pose)
    pose_config = PoseRuntimeConfig(**pose_kwargs)
    if observation_augmentation is None:
        observation_augmentation = OmegaConf.select(cfg, "world_model_collection.observation_augmentation", default={})
    if OmegaConf.is_config(observation_augmentation):
        observation_augmentation = OmegaConf.to_container(observation_augmentation, resolve=True)
    augmentation_config = (observation_augmentation if isinstance(observation_augmentation, ObservationAugmentationConfig)
                           else ObservationAugmentationConfig.from_mapping(observation_augmentation))
    if pose_config.odometry_source == "cuvslam" and not augmentation_config.is_identity:
        raise ValueError("nonzero observation_augmentation is simulation-only; cuvslam inputs must not be augmented twice")
    augmenter = PoseInputAugmenter(augmentation_config)
    raw_checkpoint = OmegaConf.select(cfg, "training.play_checkpoint_path", default=None)
    if not raw_checkpoint:
        raise FileNotFoundError("set training.play_checkpoint_path explicitly")
    root = Path(root_dir).expanduser().resolve()
    checkpoint = Path(str(raw_checkpoint)).expanduser()
    checkpoint = (root / checkpoint if not checkpoint.is_absolute() else checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    rng = np.random.default_rng(seed)
    capture = {"query": None, "residual": None}

    def transform(query):
        if query.observation_history.shape[0] != 1:
            raise ValueError("sequence capture requires exactly one environment")
        capture["query"] = {
            "observation": query.observation_history[0, -1].detach().cpu().numpy().copy(),
            "command": query.command[0].detach().cpu().numpy().copy(),
        }
        return compose_joint_position_residual(
            query.planner_future, capture["residual"], ctrl_dt=query.profile.ctrl_dt,
            position_scale=query.preprocessing["term_scale"]["dof_pos"],
            velocity_scale=query.preprocessing["term_scale"]["dof_vel"],
        )

    capture_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, enum_to_str=True))
    OmegaConf.update(capture_cfg, "training.play_checkpoint_path", str(checkpoint))
    playback = RslRlPlaybackConfig(task="G1WalkFlat", load_run="-1", checkpoint=None,
        action_mode="policy", policy_obs_mode="actor", algo_log_name="distill", log_root=None,
        num_envs=1, checkpoint_path=str(checkpoint), keyboard=False)
    session, _, resolved = create_fada_playback_session(playback_cfg=playback, cfg=capture_cfg,
        root_dir=root, device=device, future_transform=transform, deps=playback_deps)
    poses = None
    writer = None
    try:
        controller = session.controller
        if not isinstance(controller, OfficialFADAPlaybackController) or session.official_adapter is None:
            raise TypeError("sequence capture requires actual official compact64 playback")
        profile, spec = controller.profile, controller.spec
        if spec.preprocessing is None or resolved is None:
            raise ValueError("capture requires checkpoint-bound preprocessing and checkpoint identity")
        digest = hashlib.sha256()
        with Path(resolved).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        contract = {
            "data": {"window_steps": int(window_steps), "ctrl_dt": profile.ctrl_dt,
                     "history_length": spec.history_length, "prediction_horizon": spec.prediction_horizon,
                     "observation_dim": 64, "action_dim": 29},
            "source": {"checkpoint_sha256": digest.hexdigest(), "source_profile": asdict(profile),
                "preprocessing": dict(spec.preprocessing), "target_joint_names": list(session.env.get_actuated_joint_names()),
                "target_default_joint_angles": np.asarray(session.env.default_angles).reshape(-1).tolist(),
                "target_action_scale": np.broadcast_to(np.asarray(session.env.cfg.control_config.action_scale), (29,)).tolist(),
                "delta_frame": "fixed_straight_path", "delta_units": ["m", "m", "rad"],
                "ego_frame": "previous_measurement_horizontal_heading",
                "risk_tilt_limit_rad": float(risk_tilt_limit_rad)},
        }
        provenance = {
            "pose_config": asdict(pose_config), "label_source": pose_config.odometry_source,
            "capture_id": uuid4().hex,
            "residual_seed": int(seed), "residual_amplitude_rad": amplitude.tolist(),
            "residual_distribution": "independent_uniform_per_joint_held_per_window",
            "reference": "initial_physical_pose_never_reanchored", "checkpoint_path": str(resolved),
            "source_config": OmegaConf.to_container(capture_cfg, resolve=True, enum_to_str=True),
        }
        writer = SequenceDatasetWriter(output, contract)
        controller.policy.eval().requires_grad_(False)
        proxy = _CaptureController(controller)
        session.controller = proxy
        session.set_autoreset(False)
        session.reset()
        poses = create_pose_runtime(session.env, pose_config)
        poses.reset()
        history = CausalEgoHistory()
        all_receipts = []
        records, controls, label_poses = {}, [], {}
        reference = None
        epoch = None
        terminated = invalidated = reset_crossed = False
        ref_parameter = next(controller.policy.parameters())
        executed = 0
        for step in range(int(num_steps)):
            before = poses.begin_step()
            if controls and (before.sequence_id != controls[-1].sequence_id or before.timestamp_ns != controls[-1].end_ns):
                reset_crossed = True
                history.clear()
                augmenter.clear()
                break
            if step % window_steps == 0:
                residual = rng.uniform(-1., 1., (1, 29)) * amplitude
                capture["residual"] = torch.as_tensor(residual, dtype=ref_parameter.dtype, device=ref_parameter.device)
            if pose_config.odometry_source == "simulation":
                history.add(augmenter.release(before.timestamp_ns))
            ego, ego_valid, ego_age = history.at(before.timestamp_ns)
            capture["query"] = None
            proxy.source_action = None
            session.step_once()
            executed += 1
            after = poses.clock
            if after.sequence_id != before.sequence_id or after.timestamp_ns <= before.timestamp_ns:
                reset_crossed = True
                history.clear()
                augmenter.clear()
                break  # The reset observation is never a next-state training target.
            if abs(after.timestamp_ns-before.timestamp_ns-round(profile.ctrl_dt*1e9)) > 1:
                raise ValueError("physical control period disagrees with the source profile")
            query = capture["query"]
            if query is None or proxy.source_action is None:
                raise RuntimeError("real playback did not produce its query/source action")
            command = query["command"]
            if command.shape != (7,) or command[0] <= 0 or not np.allclose(command[1:3], 0., atol=1e-7, rtol=0):
                raise ValueError("capture requires a forward straight-path velocity command")
            action = np.asarray(session.env.state.info["current_actions"], dtype=np.float32)
            if action.shape != (1, 29):
                raise ValueError("current_actions must be actual target-order env-input [1,29]")
            control = ControlInterval(before.sequence_id, step, before.timestamp_ns, after.timestamp_ns, action[0].copy(), command[:3].copy())
            update = poses.complete_step(control)
            if epoch is None:
                epoch = update.tracking_epoch
            if update.invalidated or update.status == "lost" or update.tracking_epoch != epoch or update.sequence_id != before.sequence_id:
                invalidated = True
            if any(r.available_at_ns > after.timestamp_ns for r in update.received_poses):
                raise ValueError("pose runtime reported future availability")
            all_receipts.extend(update.received_poses)
            if invalidated:
                history.clear()
                augmenter.clear()
            else:
                if pose_config.odometry_source == "simulation":
                    augmenter.add(update.received_poses)
                    history.add(augmenter.release(after.timestamp_ns))
                else:
                    history.add(update.received_poses)
            next_ego, next_valid, next_age = history.at(after.timestamp_ns)
            next_obs = controller.project_observation(session.official_adapter.observation())[0].detach().cpu().numpy().copy()
            terminated = bool(np.any(session.env.state.terminated | session.env.state.truncated))
            records[step] = {**query, "next_observation": next_obs, "action": action[0].copy(),
                "source_action": proxy.source_action, "ego_motion": ego, "ego_valid": ego_valid, "ego_age_s": ego_age,
                "next_ego_motion": next_ego, "next_ego_valid": next_valid, "next_ego_age_s": next_age,
                "risk": float(bool(np.any(session.env.state.terminated)) or proprio_tilt_rad(next_obs) > risk_tilt_limit_rad),
                "risk_valid": True, "terminal": terminated}
            controls.append(control)
            if not invalidated:
                for received in update.received_poses:
                    if received.pose.valid:
                        label_poses[received.pose.timestamp_ns] = received.pose
                for aligned in update.aligned:
                    expected = next((c for c in controls if c.transition_id == aligned.control.transition_id), None)
                    if expected is None or (expected.sequence_id, expected.start_ns, expected.end_ns) != (aligned.control.sequence_id, aligned.control.start_ns, aligned.control.end_ns) or not np.array_equal(expected.action, aligned.control.action):
                        raise ValueError("delayed alignment does not match a historical executed action")
                    label_poses[aligned.start_pose.timestamp_ns] = aligned.start_pose
                    label_poses[aligned.end_pose.timestamp_ns] = aligned.end_pose
                initial = label_poses.get(controls[0].start_ns)
                if reference is None and initial is not None:
                    reference = FixedStraightPath.from_initial_pose(initial)
            if terminated or invalidated:
                break
        if controls:
            sequence = _record_sequence(controls, records)
            times = sequence["timestamps_ns"]
            if reference is not None:
                for i, timestamp in enumerate(times):
                    pose = label_poses.get(int(timestamp))
                    if pose is not None and pose_identity(pose) == (reference.sequence_id, reference.tracking_epoch, reference.frame_id):
                        sequence["errors"][i] = reference.errors(pose)
                starts, ends, deltas = [], [], []
                for s in range(0, len(controls), int(window_steps)):
                    e = s + int(window_steps)
                    if e > len(controls):
                        break
                    a, b = label_poses.get(int(times[s])), label_poses.get(int(times[e]))
                    if a is not None and b is not None and not sequence["terminal"][s:e].any():
                        starts.append(s); ends.append(e); deltas.append(_path_delta(a, b, reference))
                # Dense labels are allowed only for exact simulation truth.
                if pose_config.odometry_source == "simulation":
                    for i in range(len(controls)):
                        a, b = label_poses.get(int(times[i])), label_poses.get(int(times[i+1]))
                        if a is not None and b is not None and not sequence["terminal"][i]:
                            sequence["path_delta"][i] = _path_delta(a, b, reference)
                            sequence["motion_valid"][i] = True
                sequence["window_start"] = np.asarray(starts, dtype=np.int64)
                sequence["window_end"] = np.asarray(ends, dtype=np.int64)
                sequence["window_delta"] = np.asarray(deltas, dtype=np.float32).reshape(-1, 3)
                sequence["window_valid"] = np.ones(len(starts), dtype=bool)
            # Keep the terminal transition for positive risk supervision.
            # The model loss masks its next observation and motion targets.
            if controls:
                writer.append(sequence)
        provenance["pose_receipts"] = [
            {"sequence_id": r.pose.sequence_id, "tracking_epoch": r.pose.tracking_epoch,
             "timestamp_ns": r.pose.timestamp_ns, "available_at_ns": r.available_at_ns,
             "source_timestamps_ns": r.pose.source_timestamps_ns, "frame_id": r.pose.frame_id,
             "valid": r.pose.valid, "navigation_from_pelvis": None if r.pose.navigation_from_pelvis is None else np.asarray(r.pose.navigation_from_pelvis).tolist()}
            for r in all_receipts]
        provenance["observation_augmentation"] = {
            "applied": pose_config.odometry_source == "simulation",
            **augmenter.provenance(),
        }
        provenance["episode_ids"] = [f"{provenance['capture_id']}:{controls[0].sequence_id}"] if controls else []
        provenance["terminal_events"] = [
            {"transition_id": c.transition_id, "start_ns": c.start_ns, "end_ns": c.end_ns,
             "risk": records[c.transition_id]["risk"], "risk_valid": True}
            for c in ([control] if terminated else [])]
        provenance["label_pose_provenance"] = [
            {"timestamp_ns": p.timestamp_ns, "source_timestamps_ns": p.source_timestamps_ns,
             "sequence_id": p.sequence_id, "tracking_epoch": p.tracking_epoch, "frame_id": p.frame_id}
            for p in label_poses.values()]
        (output / "capture.json").write_text(json.dumps(provenance, allow_nan=False, indent=2) + "\n", encoding="utf-8")
        writer.close()
        return {"sequences": writer.sequence_count, "transitions": writer.transition_count,
                "executed_steps": executed, "episode_terminated": terminated,
                "localization_invalidated": invalidated, "reset_crossed": reset_crossed,
                "output_dir": str(output)}
    finally:
        try:
            if poses is not None:
                poses.close()
        finally:
            session.close()
