"""Owned physical branches for residual authority and learned-control diagnostics."""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from omegaconf import OmegaConf

from unilab.algos.torch.distill.fada.official_playback import OfficialFADAPlaybackController
from unilab.algos.torch.distill.fada.playback import FADAPlaybackController
from unilab.algos.torch.distill.fada.runtime_policy import FADAFutureQuery
from unilab.algos.torch.sttt.pose_data import ReceivedPose, TimedPelvisPose
from unilab.algos.torch.sttt.world_model.sequence_collection import CausalEgoHistory, _path_delta, proprio_tilt_rad
from unilab.visualization.playback_distill_sessions import create_fada_playback_session
from unilab.visualization.playback_sessions import RslRlPlaybackConfig
from .intervention import compose_joint_position_residual
from .response_probe import _identity, _pose
from .trajectory import MeasuredTrajectory, PathState, TrajectoryConfig

FORMAT = "sttt_physical_control_probe"
VERSION = 1


@dataclass(frozen=True)
class ProbeDecision:
    query: FADAFutureQuery
    pose: TimedPelvisPose
    path: PathState
    ego_motion: torch.Tensor
    ego_valid: torch.Tensor
    ego_age_s: torch.Tensor
    previous_residual: torch.Tensor
    step: int
    window_steps: int

    def encode(self, rollout):
        return rollout.encode_start(self.query, errors=self.path.errors(), previous_residual=self.previous_residual,
            ego_motion=self.ego_motion, ego_valid=self.ego_valid, ego_age_s=self.ego_age_s, path_state=self.path)

    def arrays(self) -> dict[str, np.ndarray]:
        values = {"observations": self.query.observation_history, "source_actions": self.query.action_history,
                  "valid": self.query.history_valid_mask, "command": self.query.command,
                  "future": self.query.planner_future, "ego_motion": self.ego_motion,
                  "ego_valid": self.ego_valid, "ego_age_s": self.ego_age_s,
                  "previous_residual": self.previous_residual, "path_pose": self.path.pose,
                  "path_progress": self.path.progress, "reference_xy": self.path.reference.xy,
                  "reference_speed": self.path.reference.speed}
        return {key: value.detach().cpu().numpy().copy() for key, value in values.items()}


@dataclass(frozen=True)
class PhysicalTrace:
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(directory / "trace.npz", **self.arrays)
        (directory / "metadata.json").write_text(json.dumps(self.metadata, allow_nan=False, indent=2) + "\n")

    @classmethod
    def load(cls, directory: str | Path) -> PhysicalTrace:
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text())
        if metadata.get("format") != FORMAT or metadata.get("version") != VERSION:
            raise ValueError("unsupported physical-control trace")
        with np.load(directory / "trace.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        if len(arrays["timestamps_ns"]) != len(arrays["actions"]) + 1:
            raise ValueError("physical trace states and actions are not paired")
        if np.any(np.diff(arrays["timestamps_ns"]) != metadata["ctrl_dt_ns"]):
            raise ValueError("physical trace contains a gap or clock change")
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("physical trace contains non-finite values")
        return cls(metadata, arrays)


def same_origin(reference: PhysicalTrace, candidate: PhysicalTrace) -> None:
    for key in ("policy_identity", "profile", "preprocessing", "trajectory", "window_steps", "reference_commands"):
        if reference.metadata[key] != candidate.metadata[key]:
            raise ValueError(f"physical branch changed {key}")
    keys = {key for key in reference.arrays if key.startswith("origin_")}
    if not keys or keys != {key for key in candidate.arrays if key.startswith("origin_")}:
        raise ValueError("physical branch is missing its origin")
    for key in keys:
        if reference.arrays[key].dtype != candidate.arrays[key].dtype or not np.array_equal(reference.arrays[key], candidate.arrays[key]):
            raise ValueError(f"physical branch changed {key}")


def playback_configuration(root: Path, policy: Path, *, profile_path: str | None, seed: int,
                           speed: float, source_overrides: list[str]):
    from .pretraining import capture_configuration
    cfg = capture_configuration(root, policy, profile_path=profile_path, seed=seed, source_overrides=source_overrides)
    for key, value in {"env.commands.vel_limit": [[speed, 0., 0.], [speed, 0., 0.]],
                       "video_recording.enabled": False, "demo.figure_eight.enabled": False,
                       "demo.footprints.enabled": False}.items():
        OmegaConf.update(cfg, key, value)
    dr = dict(OmegaConf.to_container(cfg.env.domain_rand, resolve=True))
    if any(bool(value) for key, value in dr.items() if str(key).startswith("random") or key == "push_robots") or dr.get("torque_rfi_fraction", 0) != 0 or OmegaConf.select(cfg, "env.domain_rand.actuator_strength.enabled", default=False):
        raise ValueError("physical control probes require fixed, non-randomized dynamics")
    if cfg.env.noise_config.level != 0 or OmegaConf.select(cfg, "env.runtime_payload.enabled", default=False):
        raise ValueError("baseline control probes require clean observations and no runtime payload")
    return cfg


def run_physical_branch(cfg, *, root: Path, policy: Path, program: Callable[[ProbeDecision], torch.Tensor],
                        name: str, seed: int, warmup_steps: int, window_steps: int,
                        residual_limits: np.ndarray, trajectory: TrajectoryConfig,
                        observation_steps: int | None = None, observation_cycles: float = 2.,
                        phase_fraction: float = 0., reference_commands: bool = False,
                        capture_steps: tuple[int, ...] = (0,), canonical: PhysicalTrace | None = None,
                        device: str = "cpu", playback_deps=None, on_session=None, observer=None,
                        risk_tilt_limit_rad: float = .8, stop_on_risk: bool = True) -> PhysicalTrace:
    """Execute real native playback; programs may only modify joint references."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("probe seed must fit uint32")
    if type(warmup_steps) is not int or warmup_steps < 1 or type(window_steps) is not int or window_steps < 1:
        raise ValueError("probe warmup and residual window must be positive integers")
    limits = np.asarray(residual_limits, dtype=np.float64)
    if limits.shape != (29,) or not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError("probe limits require 29 positive physical-radian bounds")
    if not 0 <= phase_fraction < 1 or not math.isfinite(observation_cycles) or observation_cycles <= 0:
        raise ValueError("probe phase and cycle count are invalid")
    if observation_steps is not None and (type(observation_steps) is not int or observation_steps < 1):
        raise ValueError("observation_steps must be a positive integer")
    if not math.isfinite(risk_tilt_limit_rad) or not 0 < risk_tilt_limit_rad <= math.pi:
        raise ValueError("invalid physical tilt limit")
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.update(cfg, "algo.seed", seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    session = controller = measured = None
    ego = CausalEgoHistory()
    current_pose = None
    step = -1
    warmup = warmup_steps
    previous = None
    final_observation = None
    latest: dict[str, Any] = {}
    origins: dict[str, np.ndarray] = {}
    rows: dict[str, list] = {key: [] for key in ("observations", "commands", "residuals", "actions", "source_actions",
        "poses", "timestamps_ns", "path_pose", "path_progress", "errors", "body_delta", "risk", "terminal",
        "ego_motion", "ego_valid", "ego_age_s")}
    warm_times, warm_poses, warm_actions = [], [], []
    metadata: dict[str, Any] = {"format": FORMAT, "version": VERSION, "name": name, "seed": seed,
        "complete": False, "valid": False, "error": None, "end_reason": None, "reference_commands": reference_commands,
        "trajectory": asdict(trajectory), "window_steps": window_steps, "residual_limits_rad": limits.tolist(),
        "risk_tilt_limit_rad": risk_tilt_limit_rad, "policy_identity": _identity(policy)}

    def state_row(pose, observation):
        assert measured is not None
        state = measured.state_at(pose)
        motion, valid, age = ego.at(pose.timestamp_ns)
        for key, value in {"observations": observation, "poses": np.asarray(pose.navigation_from_pelvis),
            "timestamps_ns": pose.timestamp_ns, "path_pose": state.pose[0].detach().cpu().numpy(),
            "path_progress": float(state.progress[0]), "errors": state.errors()[0].detach().cpu().numpy(),
            "ego_motion": motion, "ego_valid": [valid], "ego_age_s": [age]}.items():
            rows[key].append(np.asarray(value).copy())

    def transform(query):
        nonlocal previous
        assert current_pose is not None and measured is not None and session is not None
        if latest.get("query_step") == step:
            raise ValueError("multiple physical queries in one control step")
        latest["query_step"] = step
        history = query.observation_history.shape[1]
        motion, valid, age = [], [], []
        dt_ns = round(query.profile.ctrl_dt * 1e9)
        for index in range(history):
            when = current_pose.timestamp_ns - (history - 1 - index) * dt_ns
            item = ego.at(when) if when >= 0 and bool(query.history_valid_mask[0, index]) else (np.zeros(3, np.float32), False, 0.)
            motion.append(item[0]); valid.append([item[1]]); age.append([item[2]])
        if previous is None:
            previous = query.command.new_zeros(1, 29)
        decision = ProbeDecision(query, current_pose, measured.state_at(current_pose),
            query.command.new_tensor(np.asarray(motion)[None]), torch.tensor([valid], device=query.command.device, dtype=torch.bool),
            query.command.new_tensor([age]), previous, step - warmup, window_steps)
        if decision.step in capture_steps:
            prefix = "origin_" if decision.step == 0 else f"at_{decision.step}_"
            origins.update({prefix + key: value for key, value in decision.arrays().items()})
            if decision.step == 0:
                native = np.asarray(session.env.capture_rollout_snapshot().backend_state["physics_state"])
                if native.dtype != np.float64 or native.shape[0] != 1 or not np.isfinite(native).all():
                    raise ValueError("probe requires an owned native float64 origin")
                origins.update(origin_native_state=native.copy(), origin_warmup_times_ns=np.asarray(warm_times, dtype=np.int64),
                    origin_warmup_poses=np.asarray(warm_poses), origin_warmup_actions=np.asarray(warm_actions),
                    origin_gait_phase=np.asarray(session.info["gait_phase"]).copy())
                if canonical is not None:
                    same_origin(canonical, PhysicalTrace(metadata, origins))
        requested = program(decision)
        if requested.shape != (1, 29) or requested.device != query.command.device or requested.dtype != query.command.dtype or not bool(torch.isfinite(requested).all()):
            raise ValueError("residual program must return finite source-order [1,29] physical radians")
        if bool((requested.abs() > requested.new_tensor(limits) + 1e-7).any()):
            raise ValueError("residual program exceeded the declared physical bounds")
        latest.update(observation=query.observation_history[0, -1].detach().cpu().numpy().copy(),
                      command=query.command[0].detach().cpu().numpy().copy(), residual=requested[0].detach().cpu().numpy().copy())
        previous = requested.detach().clone()
        scales = query.preprocessing["term_scale"]
        return compose_joint_position_residual(query.planner_future, requested, ctrl_dt=query.profile.ctrl_dt,
            position_scale=scales["dof_pos"], velocity_scale=scales["dof_vel"],
            observation_contract=query.preprocessing.get("observation_contract", "official_compact_64_v1"))

    try:
        playback = RslRlPlaybackConfig(task="G1WalkFlat", load_run="-1", checkpoint=None, action_mode="policy",
            policy_obs_mode="actor", algo_log_name="distill", log_root=None, num_envs=1, checkpoint_path=str(policy), keyboard=False)
        session, _, resolved = create_fada_playback_session(playback_cfg=playback, cfg=cfg, root_dir=root,
            device=device, future_transform=transform, deps=playback_deps, log=lambda _: None)
        if resolved is None or _identity(Path(resolved)) != metadata["policy_identity"]:
            raise ValueError("physical session loaded a different policy")
        controller = session.controller
        if not isinstance(controller, (OfficialFADAPlaybackController, FADAPlaybackController)) or controller.profile is None or controller.spec is None:
            raise ValueError("physical probe requires ordinary checkpoint-bound Planner–Tracker playback")
        profile, spec = controller.profile, controller.spec
        if not isinstance(controller.policy, torch.nn.Module) or controller.spec.preprocessing is None:
            raise ValueError("physical probe requires a frozen torch policy with bound preprocessing")
        controller.policy.eval().requires_grad_(False)
        warmup = warmup_steps + round(phase_fraction / (profile.gait_frequency * profile.ctrl_dt))
        count = observation_steps if observation_steps is not None else max(window_steps * 2,
            math.ceil(observation_cycles / (profile.gait_frequency * profile.ctrl_dt) / window_steps) * window_steps)
        if float(session.env.cfg.ctrl_dt) != profile.ctrl_dt:
            raise ValueError("physical and source control periods differ")
        metadata.update(profile=json.loads(json.dumps(asdict(profile))), preprocessing=dict(spec.preprocessing),
            observation_contract=spec.observation_contract, ctrl_dt=profile.ctrl_dt, ctrl_dt_ns=round(profile.ctrl_dt * 1e9),
            warmup_steps=warmup, observation_steps=count, phase_fraction=phase_fraction,
            target_joint_names=list(session.env.get_actuated_joint_names()),
            target_default_joint_angles=np.asarray(session.env.default_angles).reshape(-1).tolist(),
            target_action_scale=np.broadcast_to(np.asarray(session.env.cfg.control_config.action_scale), (29,)).tolist())
        if on_session is not None:
            on_session(controller, session.env)
        session.set_autoreset(False); session.reset()
        current_pose = _pose(session.env)
        parameter = next(controller.policy.parameters())
        measured = MeasuredTrajectory(current_pose, trajectory, device=parameter.device, dtype=parameter.dtype)
        ego.add((ReceivedPose(current_pose, current_pose.timestamp_ns),))
        if observer is not None: observer(current_pose)
        warm_times.append(current_pose.timestamp_ns); warm_poses.append(np.asarray(current_pose.navigation_from_pelvis).copy())
        for step in range(warmup + count):
            command = measured.state_at(current_pose).command(np.zeros((1, spec.command_dim), np.float32)) if reference_commands else parameter.new_tensor([[trajectory.speed_m_s, 0., 0.]])
            session.set_external_command(command[:, :3].detach().cpu().numpy())
            latest.clear()
            session.step_once()
            if "observation" not in latest or controller.last_source_action is None:
                raise ValueError("physical control step omitted its real query/source action")
            after = _pose(session.env)
            if after.timestamp_ns - current_pose.timestamp_ns != metadata["ctrl_dt_ns"]:
                raise ValueError("physical branch changed localization identity or control time")
            action = np.asarray(session.info["current_actions"], dtype=np.float32)
            motion = _path_delta(current_pose, after)
            source_action = controller.last_source_action.detach().cpu().numpy()
            if action.shape != (1, 29) or source_action.shape != action.shape or not np.isfinite(action).all():
                raise ValueError("physical actions must retain actual target/source [1,29] semantics")
            if step >= warmup:
                state_row(current_pose, latest["observation"])
                for key, value in {"commands": latest["command"], "residuals": latest["residual"],
                    "actions": action[0], "source_actions": source_action[0], "body_delta": motion}.items():
                    rows[key].append(np.asarray(value).copy())
            else:
                warm_actions.append(action[0].copy()); warm_times.append(after.timestamp_ns)
                warm_poses.append(np.asarray(after.navigation_from_pelvis).copy())
            ego.add((ReceivedPose(after, after.timestamp_ns),))
            current_pose = after
            if observer is not None: observer(after)
            next_input = session.official_adapter.observation() if session.official_adapter is not None else session.obs
            final_observation = controller.project_observation(next_input)[0].detach().cpu().numpy().copy()
            terminated = bool(np.any(session.env.state.terminated))
            truncated = bool(np.any(session.env.state.truncated))
            risk = terminated or proprio_tilt_rad(final_observation, observation_contract=spec.observation_contract) > risk_tilt_limit_rad
            if step >= warmup:
                rows["risk"].append([float(risk)]); rows["terminal"].append(terminated)
            if terminated or truncated or (risk and stop_on_risk):
                metadata["end_reason"] = "terminated" if terminated else ("truncated" if truncated else "tilt_limit")
                if step < warmup: raise ValueError("physical warmup ended before the common origin")
                break
        if rows["actions"]:
            assert final_observation is not None
            state_row(current_pose, final_observation)
        metadata["complete"] = len(rows["actions"]) == count
        metadata["valid"] = bool(origins) and (metadata["complete"] or metadata["end_reason"] is not None)
        metadata["end_reason"] = metadata["end_reason"] or "requested_horizon"
    except Exception as error:
        metadata["error"] = {"type": type(error).__name__, "message": str(error)}
        metadata["valid"] = False
        if controller is not None and rows["actions"] and len(rows["observations"]) == len(rows["actions"]):
            try:
                next_input = session.official_adapter.observation() if session.official_adapter is not None else session.obs
                state_row(current_pose, controller.project_observation(next_input)[0].detach().cpu().numpy())
            except Exception:
                pass  # The incomplete physical branch remains explicitly invalid.
    finally:
        if session is not None:
            try: session.close()
            except Exception as error:
                metadata.update(valid=False, error={"type": type(error).__name__, "message": str(error)})
    arrays = {key: np.asarray(values, dtype=(np.int64 if key == "timestamps_ns" else bool if key in {"ego_valid", "terminal"} else None)) for key, values in rows.items()}
    arrays.update(origins)
    metadata["executed_steps"] = len(rows["actions"])
    return PhysicalTrace(metadata, arrays)
