"""One-state, one-direction probe using fresh sessions and verified zero warmup.

No snapshot restore, native-library change, optimizer, A, OAT, or SLAM is used.
Identical seeds alone are not accepted as evidence of an identical start.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.distill.fada.official_playback import OfficialFADAPlaybackController
from unilab.algos.torch.sttt.geometry import quat_to_mat, wrap_to_pi
from unilab.algos.torch.sttt.path_following.checkpoint import load_response_checkpoint
from unilab.algos.torch.sttt.path_following.contract import _integer
from unilab.algos.torch.sttt.path_following.features import pack_response_context, response_contract_from_metadata
from unilab.algos.torch.sttt.path_following.intervention import compose_joint_position_residual
from unilab.algos.torch.sttt.path_following.reference import measured_window_delta
from unilab.algos.torch.sttt.pose_data import TimedPelvisPose
from unilab.visualization.playback_distill_sessions import create_fada_playback_session
from unilab.visualization.playback_sessions import RslRlPlaybackConfig


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be an explicit positive finite value")
    return float(value)


def _identity(path: Path) -> dict[str, str]:
    with path.open("rb") as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def _publish(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _pose(env: Any) -> TimedPelvisPose:
    timestamp = _integer(env.get_sim_time_ns(), "probe physical timestamp")
    position, quaternion = np.asarray(env.get_base_pos(), dtype=np.float64), np.asarray(env.get_base_quat(), dtype=np.float64)
    if position.shape != (1, 3) or quaternion.shape != (1, 4) or not np.isfinite(position).all() or not np.isfinite(quaternion).all() or not np.allclose(np.linalg.norm(quaternion, axis=1), 1., atol=1e-5, rtol=0):
        raise ValueError("probe requires finite single-env base poses with unit wxyz quaternions")
    matrix = np.eye(4)
    matrix[:3, :3] = quat_to_mat(torch.from_numpy(quaternion))[0].numpy()
    matrix[:3, 3] = position[0]
    return TimedPelvisPose(0, 0, timestamp, matrix, True, (timestamp, timestamp), "probe_world")


def _angle_difference(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    difference = difference.copy()
    difference[..., 2] = wrap_to_pi(torch.from_numpy(np.asarray(difference[..., 2]).copy())).numpy()
    return difference


def compare_probe_responses(
    deltas: Mapping[str, Any], predictions: Mapping[str, Any], zero_poses_a: np.ndarray, zero_poses_b: np.ndarray,
    *, replay_position_tolerance_m: float, replay_heading_tolerance_rad: float,
    minimum_effect_m: float, minimum_effect_rad: float,
) -> dict[str, Any]:
    """Finite-amplitude local contrast, never an infinitesimal/global Jacobian claim."""
    position_tol = _positive(replay_position_tolerance_m, "replay_position_tolerance_m")
    heading_tol = _positive(replay_heading_tolerance_rad, "replay_heading_tolerance_rad")
    minimum = np.array([_positive(minimum_effect_m, "minimum_effect_m")] * 2 + [_positive(minimum_effect_rad, "minimum_effect_rad")])
    a, b = np.asarray(zero_poses_a), np.asarray(zero_poses_b)
    if a.shape != b.shape or a.ndim != 3 or a.shape[0] < 2 or a.shape[1:] != (4, 4) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("zero-repeat comparison requires aligned finite pose trajectories")
    position_error = float(np.linalg.norm(a[:, :3, 3] - b[:, :3, 3], axis=1).max())
    headings_a, headings_b = np.arctan2(a[:, 1, 0], a[:, 0, 0]), np.arctan2(b[:, 1, 0], b[:, 0, 0])
    heading_error = float(torch.abs(wrap_to_pi(torch.from_numpy(headings_a - headings_b))).max())
    repeatable = position_error <= position_tol and heading_error <= heading_tol
    actual = _angle_difference(np.asarray(deltas["plus"]), np.asarray(deltas["minus"]))
    predicted = _angle_difference(np.asarray(predictions["plus"]), np.asarray(predictions["minus"]))
    repeat_error = np.abs(_angle_difference(np.asarray(deltas["zero_b"]), np.asarray(deltas["zero_a"])))
    if actual.shape != (3,) or predicted.shape != (3,) or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("probe comparisons require finite three-channel motion deltas")
    labels = []
    for measured, estimate, threshold in zip(actual, predicted, np.maximum(minimum, repeat_error)):
        if not repeatable:
            labels.append("invalid_replay")
        elif abs(measured) <= threshold:
            labels.append("unresolved_effect")
        elif abs(estimate) <= threshold:
            labels.append("prediction_negligible")
        else:
            labels.append("same_sign" if np.sign(measured) == np.sign(estimate) else "opposite_sign")
    return {
        "valid": repeatable, "zero_repeat_max_position_error_m": position_error,
        "zero_repeat_max_heading_error_rad": heading_error,
        "zero_repeat_delta_error_m_m_rad": repeat_error.tolist(),
        "actual_plus_minus_m_m_rad": actual.tolist(), "predicted_plus_minus_m_m_rad": predicted.tolist(),
        "effect_error_m_m_rad": _angle_difference(predicted, actual).tolist(), "channel_status": labels,
    }


class _Intervention:
    def __init__(self, warmup_steps: int, residual: torch.Tensor, canonical_context: torch.Tensor | None):
        self.warmup_steps, self.residual, self.canonical_context = warmup_steps, residual, canonical_context
        self.step = 0
        self.last_query_step = -1
        self.context: torch.Tensor | None = None

    def __call__(self, query: Any) -> torch.Tensor:
        if self.last_query_step == self.step:
            raise ValueError("a probe control step attempted more than one inference query")
        self.last_query_step = self.step
        if self.step == self.warmup_steps:
            self.context = pack_response_context(query).detach().cpu()
            if self.canonical_context is not None and not torch.equal(self.context, self.canonical_context):
                raise ValueError("first probe query context differs despite repeated warmup")
        residual = self.residual.to(query.planner_future) if self.step >= self.warmup_steps else torch.zeros_like(self.residual).to(query.planner_future)
        scales = query.preprocessing["term_scale"]
        return compose_joint_position_residual(query.planner_future, residual, ctrl_dt=query.profile.ctrl_dt,
                                              position_scale=scales["dof_pos"], velocity_scale=scales["dof_vel"])


def _same_start(canonical: Mapping[str, np.ndarray], candidate: Mapping[str, np.ndarray]) -> None:
    for field in ("warmup_times_ns", "warmup_actions", "warmup_poses", "native_state", "gait_phase", "command"):
        if canonical[field].dtype != candidate[field].dtype or not np.array_equal(canonical[field], candidate[field]):
            raise ValueError(f"fresh-session warmup did not reproduce {field}; no automatic tolerance relaxation")


def _run_branch(
    name: str, residual: torch.Tensor, cfg: DictConfig, *, root: Path, output: Path, seed: int, warmup_steps: int,
    loaded: Any, policy_identity: Mapping[str, str], canonical: Mapping[str, np.ndarray] | None,
    canonical_context: torch.Tensor | None, playback_deps: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray], torch.Tensor | None, np.ndarray]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transform = _Intervention(warmup_steps, residual, canonical_context)
    session = None
    poses, times, actions = [], [], []
    start: dict[str, np.ndarray] = {}
    result: dict[str, Any] = {"branch": name, "complete": False, "error": None, "residual_rad": residual[0].tolist(), "completed_steps": 0}
    window_steps = loaded.contract["data"]["window_steps"]
    try:
        playback = RslRlPlaybackConfig(task="G1WalkFlat", load_run="-1", checkpoint=None, action_mode="policy",
                                     policy_obs_mode="actor", algo_log_name="distill", log_root=None, num_envs=1,
                                     checkpoint_path=str(cfg.training.play_checkpoint_path), keyboard=False)
        session, _, resolved = create_fada_playback_session(playback_cfg=playback, cfg=cfg, root_dir=root,
            device=str(next(loaded.model.parameters()).device), future_transform=transform, deps=playback_deps, log=lambda _: None)
        controller = session.controller
        if not isinstance(controller, OfficialFADAPlaybackController):
            raise ValueError("probe requires ordinary official playback, without online adapters")
        if resolved is None or _identity(Path(resolved))["sha256"] != policy_identity["sha256"]:
            raise ValueError("actual policy checkpoint changed or differs from the requested probe source")
        spec, profile = controller.spec, controller.profile
        if spec.preprocessing is None:
            raise ValueError("probe requires checkpoint-bound preprocessing")
        expected = response_contract_from_metadata({"contract": {
            "window_steps": window_steps, "ctrl_dt": profile.ctrl_dt, "history_length": spec.history_length,
            "prediction_horizon": spec.prediction_horizon, "joint_names": list(profile.joint_names),
            "position_scale": spec.preprocessing["term_scale"]["dof_pos"], "velocity_scale": spec.preprocessing["term_scale"]["dof_vel"],
            "tensor_dtype": "float32",
        }, "source": {"preprocessing": spec.preprocessing, "source_profile": asdict(profile),
                      "checkpoint_sha256": policy_identity["sha256"], "delta_frame": "window_start_horizontal_heading", "delta_units": ["m", "m", "rad"]}})
        if expected != loaded.contract:
            raise ValueError("B does not match the actual policy/profile/window contract")
        controller.policy.eval().requires_grad_(False)
        session.set_autoreset(False)
        session.reset()
        pose = _pose(session.env)
        poses.append(pose.navigation_from_pelvis)
        times.append(pose.timestamp_ns)
        first_probe_pose = None
        for step in range(warmup_steps + window_steps):
            if step == warmup_steps:
                # Existing public snapshot is read once for native precision; no
                # snapshot restore or private backend access is introduced.
                snapshot = session.env.capture_rollout_snapshot()
                native = np.asarray(snapshot.backend_state["physics_state"])
                if native.dtype != np.float64 or native.ndim != 2 or native.shape[0] != 1 or not np.isfinite(native).all():
                    raise ValueError("same-start checks require finite native float64 physics state")
                start = {"warmup_times_ns": np.asarray(times, dtype=np.int64), "warmup_actions": np.asarray(actions),
                         "warmup_poses": np.asarray(poses), "native_state": native,
                         "gait_phase": np.asarray(session.info["gait_phase"]).copy(), "command": np.asarray(session.info["commands"]).copy()}
                if canonical is not None:
                    _same_start(canonical, start)
                first_probe_pose = pose
            transform.step = step
            session.step_once()
            next_pose = _pose(session.env)
            if abs(next_pose.timestamp_ns - pose.timestamp_ns - round(profile.ctrl_dt * 1e9)) > 1:
                raise ValueError("probe physical control interval differs from the B window contract")
            action = np.asarray(session.info["current_actions"], dtype=np.float64)
            if action.shape != (1, 29) or not np.isfinite(action).all():
                raise ValueError("probe requires actual env-input actions [1,29]")
            actions.append(action[0].copy())
            poses.append(next_pose.navigation_from_pelvis)
            times.append(next_pose.timestamp_ns)
            pose = next_pose
            result["completed_steps"] += 1
            state = session.env.state
            if bool(np.any(state.terminated | state.truncated)):
                raise ValueError("probe branch terminated/truncated; keep this outcome, do not retry or replace the start")
        assert first_probe_pose is not None and transform.context is not None
        result["delta_m_m_rad"] = measured_window_delta(first_probe_pose, pose).tolist()
        result["context_sha256"] = hashlib.sha256(transform.context.numpy().tobytes()).hexdigest()
        result["complete"] = True
    except Exception as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        result["failed_step_physics_may_have_advanced"] = True
    finally:
        if session is not None:
            try:
                session.close()
            except Exception as error:
                result["complete"] = False
                result["close_error"] = {"type": type(error).__name__, "message": str(error)}
        arrays: dict[str, Any] = {
            "timestamps_ns": np.asarray(times, dtype=np.int64), "poses": np.asarray(poses),
            "executed_actions": np.asarray(actions), "residual_rad": residual.cpu().numpy(),
            "context": np.empty((0,)) if transform.context is None else transform.context.numpy(),
            **{f"start_{key}": value for key, value in start.items()},
        }
        np.savez_compressed(output / f"{name}.npz", **arrays)
    probe_poses = np.asarray(poses[warmup_steps:])
    return result, start, transform.context, probe_poses


def run_response_direction_probe(
    cfg: DictConfig, *, root_dir: str | Path | None = None, playback_deps: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Explicit finite experiment entry. Construction verification injects physical I/O."""
    fields = {"policy_checkpoint", "profile_path", "response_checkpoint", "output_dir", "device", "seed", "warmup_steps",
              "residual_amplitude_rad", "replay_position_tolerance_m", "replay_heading_tolerance_rad", "minimum_effect_m",
              "minimum_effect_rad", "source_overrides", "branch_order"}
    if not isinstance(cfg, DictConfig) or set(cfg) != fields:
        raise ValueError("use conf/sttt/response_direction_probe.yaml")
    seed, warmup = _integer(cfg.seed, "seed"), _integer(cfg.warmup_steps, "warmup_steps")
    if seed >= 2**32 or warmup < 1:
        raise ValueError("seed must be <2**32 and warmup_steps positive")
    amplitude = _positive(cfg.residual_amplitude_rad, "residual_amplitude_rad")
    tolerances = {name: _positive(cfg[name], name) for name in ("replay_position_tolerance_m", "replay_heading_tolerance_rad", "minimum_effect_m", "minimum_effect_rad")}
    order = list(cfg.branch_order)
    if len(order) != 4 or order[0] != "zero_a" or order[-1] != "zero_b" or set(order[1:3]) != {"plus", "minus"}:
        raise ValueError("exactly four branches are required: zero_a, plus/minus in either order, zero_b")
    root = Path(root_dir) if root_dir is not None else Path(__file__).resolve().parents[6]
    def resolve(value: Any) -> Path:
        if not value:
            raise ValueError("all model/profile/output paths must be explicit")
        path = Path(str(value)).expanduser()
        return (path if path.is_absolute() else root / path).resolve()
    policy_path, profile_path, response_path, output = map(resolve, (cfg.policy_checkpoint, cfg.profile_path, cfg.response_checkpoint, cfg.output_dir))
    identities = {"policy": _identity(policy_path), "profile": _identity(profile_path), "response": _identity(response_path)}
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"probe output must be empty: {output}")
    loaded = load_response_checkpoint(response_path, device=str(cfg.device))
    loaded.model.eval().requires_grad_(False)
    direction = np.random.default_rng(seed).uniform(-1., 1., (1, 29))
    direction *= amplitude / np.max(np.abs(direction))
    residual = torch.as_tensor(direction, dtype=torch.float32)
    reference = next(loaded.model.parameters())
    candidates = torch.cat((torch.zeros_like(residual), residual, -residual)).to(reference)
    if not bool(((candidates >= loaded.model.residual_min) & (candidates <= loaded.model.residual_max)).all()):
        raise ValueError("zero/positive/negative residual exceeds B training coverage; no clipping or amplitude expansion is allowed")
    with initialize_config_dir(version_base="1.3", config_dir=str(root / "conf/distill")):
        playback_cfg = compose(config_name="config", overrides=["task=g1_walk_flat/mujoco", "recording_profile=straight_payload", *list(cfg.source_overrides), "~path_following"])
    for key, value in (("training.play_checkpoint_path", str(policy_path)), ("official_playback.profile_path", str(profile_path)),
                       ("online_adaptation.enabled", False), ("env.runtime_payload.enabled", False),
                       ("video_recording.enabled", False), ("demo.figure_eight.enabled", False), ("demo.footprints.enabled", False), ("algo.seed", seed)):
        OmegaConf.update(playback_cfg, key, value)
    if playback_cfg.training.task_name != "G1WalkFlat" or playback_cfg.training.sim_backend != "mujoco":
        raise ValueError("probe requires g1_walk_flat/mujoco")
    if OmegaConf.select(playback_cfg, "interactive.keyboard") or OmegaConf.select(playback_cfg, "env.commands.heading_command") or playback_cfg.env.commands.resampling_time != 0:
        raise ValueError("probe requires fixed straight commands without keyboard/heading controller")
    limits = np.asarray(playback_cfg.env.commands.vel_limit, dtype=float)
    if limits.shape != (2, 3) or not np.isfinite(limits).all() or not np.array_equal(limits[0], limits[1]) or limits[0, 0] <= 0 or not np.array_equal(limits[0, 1:], [0., 0.]):
        raise ValueError("probe requires deterministic [vx>0,0,0] velocity limits")
    dr = OmegaConf.to_container(playback_cfg.env.domain_rand, resolve=True)
    if not isinstance(dr, dict) or any(bool(value) for key, value in dr.items() if str(key).startswith("random") or key == "push_robots") or dr.get("torque_rfi_fraction", 0) != 0 or OmegaConf.select(playback_cfg, "env.domain_rand.actuator_strength.enabled", default=False) or OmegaConf.select(playback_cfg, "env.noise_config.level") != 0:
        raise ValueError("probe requires clean deterministic pretraining conditions without randomization/noise")
    output.mkdir(parents=True, exist_ok=True)
    _publish(output / "source.json", {"format": "sttt_response_direction_probe", "schema_version": 1,
        "models": identities, "probe_config": OmegaConf.to_container(cfg, resolve=True),
        "playback_config": OmegaConf.to_container(playback_cfg, resolve=True, enum_to_str=True),
        "response_contract": loaded.contract, "residual_rad": residual[0].tolist(),
        "method": "fresh_sessions_with_verified_zero_warmup", "scope": "one_start_one_direction_no_training"})
    summary: dict[str, Any] = {"complete": False, "valid": False, "branches": [], "comparison": None,
        "channels": ["dx", "dy", "dyaw"], "units": ["m", "m", "rad"],
        "scope": "local finite-amplitude simulator contrast only, not global Jacobian or closed-loop verification"}
    _publish(output / "summary.json", summary)
    canonical = canonical_context = None
    trajectories, deltas = {}, {}
    for name in order:
        branch_residual = residual if name == "plus" else (-residual if name == "minus" else torch.zeros_like(residual))
        result, start, context, trajectory = _run_branch(name, branch_residual, playback_cfg, root=root, output=output,
            seed=seed, warmup_steps=warmup, loaded=loaded, policy_identity=identities["policy"], canonical=canonical,
            canonical_context=canonical_context, playback_deps=playback_deps)
        summary["branches"].append(result)
        _publish(output / "summary.json", summary)
        if not result["complete"]:
            return summary
        if canonical is None:
            canonical, canonical_context = start, context
        trajectories[name] = trajectory
        deltas[name] = result["delta_m_m_rad"]
    assert canonical_context is not None
    try:
        with torch.no_grad():
            predicted = loaded.model(canonical_context.to(reference).expand(3, -1), candidates).cpu().numpy()
        if predicted.shape != (3, 3) or not np.isfinite(predicted).all():
            raise ValueError("B produced invalid/non-finite probe predictions")
        predictions = {"zero": predicted[0].tolist(), "plus": predicted[1].tolist(), "minus": predicted[2].tolist()}
        comparison = compare_probe_responses(deltas, predictions, trajectories["zero_a"], trajectories["zero_b"], **tolerances)
    except Exception as error:
        summary["error"] = {"type": type(error).__name__, "message": str(error)}
        _publish(output / "summary.json", summary)
        return summary
    summary.update(complete=True, valid=comparison["valid"], comparison=comparison, predictions_m_m_rad=predictions)
    _publish(output / "summary.json", summary)
    return summary
