"""Verified fresh-session finite contrasts through the frozen latent control loop."""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra._internal.config_loader_impl import ConfigLoaderImpl
from hydra._internal.utils import create_config_search_path
from hydra.types import RunMode
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import nn

from unilab.algos.torch.distill.fada.official_playback import OfficialFADAPlaybackController
from unilab.algos.torch.distill.fada.playback import FADAPlaybackController
from unilab.algos.torch.sttt.path_following.contract import _integer
from unilab.algos.torch.sttt.path_following.features import pack_response_context
from unilab.algos.torch.sttt.path_following.intervention import compose_joint_position_residual
from unilab.algos.torch.sttt.path_following.latent_rollout import PlannerTrackerRollout, RolloutResult
from unilab.algos.torch.sttt.path_following.reference import FixedStraightPath
from unilab.algos.torch.sttt.path_following.response_probe import (
    _identity, _pose, _positive, _publish, _same_start, compare_probe_responses,
)
from unilab.algos.torch.sttt.pose_data import ReceivedPose, TimedPelvisPose
from unilab.algos.torch.sttt.world_model.checkpoint import load_world_model
from unilab.algos.torch.sttt.world_model.sequence_collection import CausalEgoHistory, _path_delta
from unilab.visualization.playback_distill_sessions import create_fada_playback_session
from unilab.visualization.playback_sessions import RslRlPlaybackConfig


def _canonical(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


class _HeldResidual(nn.Module):
    residual: torch.Tensor

    def __init__(self, residual: torch.Tensor):
        super().__init__()
        self.register_buffer("residual", residual.detach().clone())

    def forward(self, feature, _errors, _command):
        return self.residual.expand(feature.shape[0], -1)


class _Intervention:
    def __init__(self, warmup, residual, loaded, horizon, seed, canonical):
        self.warmup, self.residual, self.loaded = warmup, residual, loaded
        self.horizon, self.seed, self.canonical = horizon, seed, canonical
        self.step, self.last_query_step = 0, -1
        self.context: torch.Tensor | None = None
        self.prediction: RolloutResult | None = None
        self.encoded: dict[str, np.ndarray] = {}
        self.ego = CausalEgoHistory()
        self.rollout: PlannerTrackerRollout | None = None
        self.reference: FixedStraightPath | None = None
        self.pose: TimedPelvisPose | None = None

    def observe(self, pose):
        self.pose = pose
        self.ego.add((ReceivedPose(pose, pose.timestamp_ns),))
        if self.reference is None:
            self.reference = FixedStraightPath.from_initial_pose(pose)

    def __call__(self, query):
        if self.step == self.last_query_step:
            raise ValueError("multiple inference queries in one probe control step")
        self.last_query_step = self.step
        if self.step == self.warmup:
            self.context = pack_response_context(query).detach().cpu()
            if self.canonical is not None and not np.array_equal(self.context.numpy(), self.canonical["context"]):
                raise ValueError("first probe query differs despite repeated warmup")
            rollout, pose, reference = self.rollout, self.pose, self.reference
            if rollout is None or pose is None or reference is None:
                raise RuntimeError("probe origin must be bound before prediction")
            history = query.observation_history.shape[1]
            ego = np.zeros((history, 3), dtype=np.float32)
            valid = np.zeros((history, 1), dtype=bool)
            age = np.zeros((history, 1), dtype=np.float32)
            dt_ns = round(query.profile.ctrl_dt * 1e9)
            for index in range(history):
                timestamp = pose.timestamp_ns - (history - 1 - index) * dt_ns
                if timestamp >= 0 and bool(query.history_valid_mask[0, index]):
                    ego[index], valid[index, 0], age[index, 0] = self.ego.at(timestamp)
            parameter = next(self.loaded.model.parameters())
            devices = [parameter.device.index] if parameter.is_cuda else []
            # Common random numbers isolate the residual contrast from RSSM sampling.
            # Restore RNG afterwards so imagination cannot change real execution.
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                torch.manual_seed(self.seed)
                start = rollout.encode_start(
                    query, errors=query.command.new_tensor(reference.errors(pose)[None]),
                    previous_residual=torch.zeros_like(self.residual),
                    ego_motion=query.command.new_tensor(ego[None]),
                    ego_valid=torch.as_tensor(valid[None], device=parameter.device),
                    ego_age_s=query.command.new_tensor(age[None]),
                )
                self.encoded = {
                    "ego_motion": ego, "ego_valid": valid, "ego_age_s": age,
                    "latent_state": start.latent.state.cpu().numpy().copy(),
                    "latent_belief": start.latent.belief.cpu().numpy().copy(),
                }
                if self.canonical is not None:
                    for key, value in self.encoded.items():
                        if not np.array_equal(value, self.canonical[key]):
                            raise ValueError(f"probe start {key} differs from the canonical branch")
                self.prediction = rollout.run(start, _HeldResidual(self.residual),
                    window_steps=self.loaded.contract["data"]["window_steps"], horizon_windows=self.horizon)
        residual = self.residual.to(query.planner_future) if self.step >= self.warmup else torch.zeros_like(self.residual).to(query.planner_future)
        scales = query.preprocessing["term_scale"]
        return compose_joint_position_residual(query.planner_future, residual, ctrl_dt=query.profile.ctrl_dt,
            position_scale=scales["dof_pos"], velocity_scale=scales["dof_vel"],
            observation_contract=query.preprocessing.get("observation_contract", "official_compact_64_v1"))


def _bind(loaded, controller, env, policy_identity) -> PlannerTrackerRollout:
    if not isinstance(controller, (OfficialFADAPlaybackController, FADAPlaybackController)) or controller.profile is None or controller.spec is None:
        raise ValueError("probe requires checkpoint-bound ordinary Planner–Tracker playback")
    profile, spec = controller.profile, controller.spec
    policy = controller.policy
    if not isinstance(policy, nn.Module) or spec.preprocessing is None:
        raise ValueError("probe requires a torch policy and checkpoint-bound preprocessing")
    expected = {
        "checkpoint_sha256": policy_identity["sha256"], "source_profile": asdict(profile),
        "preprocessing": spec.preprocessing, "delta_frame": "fixed_straight_path",
        "delta_units": ["m", "m", "rad"], "ego_frame": "previous_measurement_horizontal_heading",
        "target_joint_names": list(env.get_actuated_joint_names()),
        "target_default_joint_angles": np.asarray(env.default_angles).reshape(-1).tolist(),
        "target_action_scale": np.broadcast_to(np.asarray(env.cfg.control_config.action_scale), (29,)).tolist(),
    }
    for name, value in expected.items():
        if _canonical(loaded.contract["source"].get(name)) != _canonical(value):
            raise ValueError(f"WM differs from the probe execution {name}")
    data = loaded.contract["data"]
    for name, value in {"ctrl_dt": profile.ctrl_dt, "history_length": spec.history_length,
                        "prediction_horizon": spec.prediction_horizon, "observation_dim": spec.obs_dim,
                        "action_dim": spec.action_dim}.items():
        if data.get(name) != value:
            raise ValueError(f"WM differs from the probe {name}")
    if data.get("command_dim", 7) != spec.command_dim or float(env.cfg.ctrl_dt) != profile.ctrl_dt:
        raise ValueError("WM command width or environment control period differs")
    if next(loaded.model.parameters()).dtype != next(policy.parameters()).dtype:
        raise ValueError("WM and policy must use the same floating dtype")
    return PlannerTrackerRollout(loaded.model, policy, profile, spec.preprocessing,
        target_joint_names=expected["target_joint_names"],
        target_default_joint_angles=expected["target_default_joint_angles"], target_action_scale=expected["target_action_scale"])


def _run_branch(name, residual, cfg, *, root, output, seed, warmup, horizon, loaded, policy_identity, canonical, playback_deps):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    intervention = _Intervention(warmup, residual, loaded, horizon, seed ^ 0x5DEECE66D, canonical)
    session = None
    poses, times, actions = [], [], []
    start: dict[str, np.ndarray] = {}
    result = {"branch": name, "complete": False, "error": None,
              "residual_rad": residual[0].cpu().tolist(), "completed_steps": 0}
    try:
        playback = RslRlPlaybackConfig(task="G1WalkFlat", load_run="-1", checkpoint=None, action_mode="policy",
            policy_obs_mode="actor", algo_log_name="distill", log_root=None, num_envs=1,
            checkpoint_path=str(cfg.training.play_checkpoint_path), keyboard=False)
        session, _, resolved = create_fada_playback_session(playback_cfg=playback, cfg=cfg, root_dir=root,
            device=str(residual.device), future_transform=intervention, deps=playback_deps, log=lambda _: None)
        if resolved is None or _identity(Path(resolved))["sha256"] != policy_identity["sha256"]:
            raise ValueError("actual policy checkpoint differs from the requested source")
        rollout = _bind(loaded, session.controller, session.env, policy_identity)
        intervention.rollout = rollout
        session.set_autoreset(False)
        session.reset()
        pose = _pose(session.env)
        intervention.observe(pose)
        poses.append(pose.navigation_from_pelvis)
        times.append(pose.timestamp_ns)
        first_probe_pose = None
        window_steps = loaded.contract["data"]["window_steps"]
        for step in range(warmup + window_steps * horizon):
            if step == warmup:
                native = np.asarray(session.env.capture_rollout_snapshot().backend_state["physics_state"])
                if native.dtype != np.float64 or native.ndim != 2 or native.shape[0] != 1 or not np.isfinite(native).all():
                    raise ValueError("probe requires finite native float64 physics state")
                start = {"warmup_times_ns": np.asarray(times, dtype=np.int64), "warmup_actions": np.asarray(actions),
                         "warmup_poses": np.asarray(poses), "native_state": native.copy(),
                         "gait_phase": np.asarray(session.info["gait_phase"]).copy(),
                         "command": np.asarray(session.info["commands"]).copy()}
                if canonical is not None:
                    _same_start(canonical, start)
                first_probe_pose = pose
            intervention.step = step
            session.step_once()
            next_pose = _pose(session.env)
            if next_pose.timestamp_ns - pose.timestamp_ns != round(rollout.profile.ctrl_dt * 1e9):
                raise ValueError("probe control period differs from the WM contract")
            action = np.asarray(session.info["current_actions"], dtype=np.float32)
            if action.shape != (1, 29) or not np.isfinite(action).all():
                raise ValueError("probe requires finite actual env-input actions [1,29]")
            actions.append(action[0].copy())
            poses.append(next_pose.navigation_from_pelvis)
            times.append(next_pose.timestamp_ns)
            pose = next_pose
            intervention.observe(pose)
            result["completed_steps"] += 1
            if bool(np.any(session.env.state.terminated | session.env.state.truncated)):
                raise ValueError("probe branch terminated/truncated; no retry or replacement")
        if first_probe_pose is None or intervention.prediction is None or intervention.context is None:
            raise RuntimeError("probe did not reach its prediction origin")
        predicted_actions = intervention.prediction.actions[0].cpu().numpy()
        if not np.allclose(predicted_actions[0], actions[warmup], rtol=1e-6, atol=1e-6):
            raise ValueError("first imagined action differs from actual Planner–Tracker execution")
        delta = intervention.prediction.path_delta[0].sum(0).cpu().numpy().copy()
        delta[2] = math.atan2(math.sin(delta[2]), math.cos(delta[2]))
        if not np.isfinite(delta).all():
            raise ValueError("WM produced non-finite probe motion")
        result.update(complete=True, delta_m_m_rad=_path_delta(first_probe_pose, pose, intervention.reference).tolist(),
            predicted_delta_m_m_rad=delta.tolist(), context_sha256=hashlib.sha256(intervention.context.numpy().tobytes()).hexdigest())
        start.update(context=intervention.context.numpy().copy(), **intervention.encoded)
    except Exception as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        result["failed_step_physics_may_have_advanced"] = True
    finally:
        if session is not None:
            try:
                session.close()
            except Exception as error:
                result.update(complete=False, close_error={"type": type(error).__name__, "message": str(error)})
        arrays = {"timestamps_ns": np.asarray(times, dtype=np.int64), "poses": np.asarray(poses),
                  "executed_actions": np.asarray(actions), "residual_rad": residual.cpu().numpy(),
                  **{f"start_{key}": value for key, value in start.items()}}
        if intervention.prediction is not None:
            arrays.update(predicted_actions=intervention.prediction.actions[0].cpu().numpy(),
                          predicted_source_actions=intervention.prediction.source_actions[0].cpu().numpy(),
                          predicted_path_delta=intervention.prediction.path_delta[0].cpu().numpy())
        np.savez_compressed(output / f"{name}.npz", **arrays)
    return result, start, np.asarray(poses[warmup:])


def run_latent_response_direction_probe(cfg: DictConfig, *, root_dir=None, playback_deps=None):
    """Run exactly four finite simulation branches; never update any network."""
    fields = {"policy_checkpoint", "profile_path", "world_model_checkpoint", "output_dir", "device", "seed",
              "warmup_steps", "horizon_windows", "residual_amplitude_rad", "replay_position_tolerance_m",
              "replay_heading_tolerance_rad", "minimum_effect_m", "minimum_effect_rad", "source_overrides", "branch_order"}
    if not isinstance(cfg, DictConfig) or set(cfg) != fields:
        raise ValueError("use conf/sttt/latent_response_direction_probe.yaml")
    seed, warmup, horizon = (_integer(cfg[key], key) for key in ("seed", "warmup_steps", "horizon_windows"))
    if seed >= 2**32 or warmup < 1 or horizon < 1:
        raise ValueError("seed must fit uint32; warmup and horizon must be positive")
    amplitude = _positive(cfg.residual_amplitude_rad, "residual_amplitude_rad")
    tolerances = {key: _positive(cfg[key], key) for key in
                  ("replay_position_tolerance_m", "replay_heading_tolerance_rad", "minimum_effect_m", "minimum_effect_rad")}
    order = list(cfg.branch_order)
    if len(order) != 4 or order[0] != "zero_a" or order[-1] != "zero_b" or set(order[1:3]) != {"plus", "minus"}:
        raise ValueError("require zero_a, plus/minus in either order, zero_b")
    root = Path(root_dir).resolve() if root_dir is not None else Path(__file__).resolve().parents[6]
    def resolve(value):
        if not value:
            raise ValueError("policy, WM and output paths must be explicit")
        path = Path(str(value)).expanduser()
        return (path if path.is_absolute() else root / path).resolve()
    policy_path, model_path, output = map(resolve, (cfg.policy_checkpoint, cfg.world_model_checkpoint, cfg.output_dir))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"probe output must be empty: {output}")
    identities = {"policy": _identity(policy_path), "world_model": _identity(model_path)}
    loaded = load_world_model(model_path, device=str(cfg.device))
    if loaded.contract["source"]["checkpoint_sha256"] != identities["policy"]["sha256"]:
        raise ValueError("WM was trained for another policy checkpoint")
    loaded.model.eval().requires_grad_(False)
    direction = np.random.default_rng(seed).uniform(-1., 1., (1, 29))
    direction *= amplitude / np.max(np.abs(direction))
    parameter = next(loaded.model.parameters())
    residual = torch.as_tensor(direction, dtype=parameter.dtype, device=parameter.device)
    loader = ConfigLoaderImpl(config_search_path=create_config_search_path(str(root / "conf/distill")))
    playback_cfg = loader.load_configuration(config_name="config", overrides=[
        "task=g1_walk_flat/mujoco", "recording_profile=straight_payload", *list(cfg.source_overrides), "~path_following"],
        run_mode=RunMode.RUN, from_shell=False)
    with open_dict(playback_cfg):
        del playback_cfg["hydra"]
    for key, value in (("training.play_checkpoint_path", str(policy_path)),
                       ("official_playback.profile_path", str(resolve(cfg.profile_path)) if cfg.profile_path else None),
                       ("online_adaptation.enabled", False), ("env.runtime_payload.enabled", False),
                       ("video_recording.enabled", False), ("demo.figure_eight.enabled", False),
                       ("demo.footprints.enabled", False), ("algo.seed", seed)):
        OmegaConf.update(playback_cfg, key, value)
    if playback_cfg.training.task_name != "G1WalkFlat" or playback_cfg.training.sim_backend != "mujoco":
        raise ValueError("probe requires g1_walk_flat/mujoco")
    if playback_cfg.interactive.keyboard or playback_cfg.env.commands.heading_command or playback_cfg.env.commands.resampling_time != 0:
        raise ValueError("probe requires fixed commands without keyboard/heading control")
    limits = np.asarray(playback_cfg.env.commands.vel_limit, dtype=float)
    if limits.shape != (2, 3) or not np.isfinite(limits).all() or not np.array_equal(limits[0], limits[1]) or limits[0, 0] <= 0 or np.any(limits[0, 1:] != 0):
        raise ValueError("probe requires deterministic [vx>0,0,0] commands")
    dr = OmegaConf.to_container(playback_cfg.env.domain_rand, resolve=True)
    if not isinstance(dr, dict) or any(bool(value) for key, value in dr.items() if str(key).startswith("random") or key == "push_robots") or dr.get("torque_rfi_fraction", 0) != 0 or OmegaConf.select(playback_cfg, "env.domain_rand.actuator_strength.enabled", default=False) or playback_cfg.env.noise_config.level != 0:
        raise ValueError("probe requires clean conditions without randomization/noise")
    output.mkdir(parents=True, exist_ok=True)
    _publish(output / "source.json", {"format": "sttt_latent_response_direction_probe", "schema_version": 1,
        "models": identities, "probe_config": OmegaConf.to_container(cfg, resolve=True),
        "playback_config": OmegaConf.to_container(playback_cfg, resolve=True, enum_to_str=True),
        "world_model_contract": loaded.contract, "residual_rad": residual[0].cpu().tolist(),
        "method": "fresh_sessions_with_verified_zero_warmup_and_common_rssm_random_numbers",
        "delta_frame": "fixed_straight_path", "scope": "one_start_one_direction_no_training"})
    summary = {"complete": False, "valid": False, "branches": [], "comparison": None,
               "channels": ["dx", "dy", "dyaw"], "units": ["m", "m", "rad"],
               "horizon_s": horizon * loaded.contract["data"]["window_steps"] * loaded.contract["data"]["ctrl_dt"],
               "scope": "local finite-amplitude contrast through frozen Planner–Tracker and WM"}
    _publish(output / "summary.json", summary)
    canonical = None
    trajectories, deltas, predictions = {}, {}, {}
    for name in order:
        candidate = residual if name == "plus" else (-residual if name == "minus" else torch.zeros_like(residual))
        result, start, trajectory = _run_branch(name, candidate, playback_cfg, root=root, output=output,
            seed=seed, warmup=warmup, horizon=horizon, loaded=loaded, policy_identity=identities["policy"],
            canonical=canonical, playback_deps=playback_deps)
        summary["branches"].append(result)
        _publish(output / "summary.json", summary)
        if not result["complete"]:
            return summary
        if canonical is None:
            canonical = start
        trajectories[name], deltas[name], predictions[name] = trajectory, result["delta_m_m_rad"], result["predicted_delta_m_m_rad"]
    summary.update(complete=True, predictions_m_m_rad=predictions)
    comparison = compare_probe_responses(deltas, predictions, trajectories["zero_a"], trajectories["zero_b"], **tolerances)
    summary.update(valid=comparison["valid"], comparison=comparison)
    _publish(output / "summary.json", summary)
    return summary
