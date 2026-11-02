"""STTT spatial collection runtime: session factory and transition loop.

Builds a real FADA Planner-IDM playback session from an explicit checkpoint
and streams :class:`SpatialTransition` records with a fully manual reset
lifecycle. No training defaults are changed globally: every collection
override is applied to a private clone of the composed Hydra config.
"""

from __future__ import annotations

import hashlib
import random
import time
from collections.abc import Iterator, Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from unilab.algos.torch.sttt.contract import (
    DEFAULT_OBS_DIM,
    PROPRIO_LAYOUT,
    SPATIAL_ACTION_REPRESENTATION,
    SPATIAL_DATA_FORMAT_VERSION,
    SpatialTransition,
)
from unilab.visualization.playback_distill_sessions import create_fada_playback_session
from unilab.visualization.playback_sessions import FADAPlaybackSession, RslRlPlaybackConfig

__all__ = ["create_collection_session", "collect_transitions", "render_transitions"]

_COLLECTION_TASK_NAME = "G1WalkFlat"
_COLLECTION_BACKEND = "mujoco"
_SHA256_CHUNK_BYTES = 1 << 20
_COMMAND_HOLD_SECONDS = (1.0, 2.0)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_SHA256_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_primitive(value: Any) -> Any:
    """Convert config values to JSON primitives with the stdlib conversions only.

    ``dataclass`` -> ``asdict``, ``np.ndarray``/``torch.Tensor`` -> nested
    lists, NumPy scalars -> Python scalars. No registry and no silent string
    fallback: an unconvertible leaf is a programming error, not data.
    """

    if OmegaConf.is_config(value):
        return _json_primitive(OmegaConf.to_container(value, resolve=True))
    if is_dataclass(value) and not isinstance(value, type):
        return _json_primitive(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_primitive(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_primitive(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return _json_primitive(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("collection context must not contain non-finite numbers")
        return value
    raise TypeError(f"collection context value is not JSON-safe: {type(value).__name__}")


def _collection_context(env: Any) -> dict[str, Any]:
    """Record the effective collection condition block from the live env config.

    These are the configured conditions of this session (command ranges and
    resampling, runtime payload, domain randomization, observation noise), not
    evidence that a payload was physically attached inside every window.
    Whole config blocks are recorded with the stdlib dataclass conversion so
    no field is silently dropped.
    """

    env_cfg = env.cfg
    context: dict[str, Any] = {}
    for key in ("commands", "runtime_payload", "domain_rand", "noise_config"):
        block = getattr(env_cfg, key, None)
        if block is None:
            raise RuntimeError(
                f"STTT collection requires env.cfg.{key} to record the collection context"
            )
        context[key] = _json_primitive(block)
    return context


def _set_collection_override(cfg: Any, dotted_key: str, value: Any) -> None:
    """Set a leaf on the private collection config clone only."""

    if OmegaConf.is_config(cfg):
        OmegaConf.update(cfg, dotted_key, value, merge=False)
        return
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        node = getattr(node, part)
    setattr(node, parts[-1], value)


def _clone_collection_cfg(cfg: Any) -> Any:
    """Return a private resolved copy; the caller's config is never mutated."""

    if OmegaConf.is_config(cfg):
        return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    import copy

    return copy.deepcopy(cfg)


def create_collection_session(
    cfg: Any,
    *,
    root_dir: str | Path,
    device: str | None = None,
    seed: int = 0,
    prepare_rollout_branch: bool = False,
) -> tuple[FADAPlaybackSession, dict[str, Any]]:
    """Create a single-env FADA policy session for spatial data collection.

    ``cfg.training.play_checkpoint_path`` must name an existing Planner-IDM
    checkpoint; there is no recent-run or zero-policy fallback. The returned
    metadata dict is the export-envelope metadata minus ``windows`` (writer
    supplies ``history_length``/``horizon``); ``seed`` is set by the caller.

    ``metadata["collection_context"]`` records the effective
    ``env.cfg.commands`` / ``runtime_payload`` / ``domain_rand`` /
    ``noise_config`` blocks as JSON primitives, read from the live env config
    after the collection overrides are applied. It describes the configured
    collection condition for this session only; it is not evidence that a
    payload was actually attached in any collected window.
    """

    task_name = str(getattr(getattr(cfg, "training", None), "task_name", ""))
    if task_name != _COLLECTION_TASK_NAME:
        raise ValueError(
            f"STTT collection is restricted to task g1_walk_flat/mujoco "
            f"({_COLLECTION_TASK_NAME}), got {task_name!r}"
        )
    sim_backend = str(getattr(getattr(cfg, "training", None), "sim_backend", _COLLECTION_BACKEND))
    if sim_backend != _COLLECTION_BACKEND:
        raise ValueError(f"STTT collection requires sim_backend=mujoco, got {sim_backend!r}")

    raw_path = getattr(cfg.training, "play_checkpoint_path", None)
    if raw_path in (None, ""):
        raise FileNotFoundError(
            "STTT collection requires an explicit training.play_checkpoint_path; "
            "no recent-run or zero-policy fallback exists."
        )
    checkpoint = Path(str(raw_path)).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = Path(root_dir) / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"training.play_checkpoint_path does not exist: {checkpoint}")
    checkpoint_sha256 = _sha256_file(checkpoint)
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError("collection seed must be an integer in [0, 2**32)")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    collection_cfg = _clone_collection_cfg(cfg)
    # Deterministic collection contract on the private clone only.
    _set_collection_override(collection_cfg, "online_adaptation.enabled", False)
    _set_collection_override(collection_cfg, "spatial_adaptation.enabled", False)
    _set_collection_override(collection_cfg, "interactive.keyboard", False)
    _set_collection_override(collection_cfg, "demo.figure_eight.enabled", False)
    _set_collection_override(collection_cfg, "demo.footprints.enabled", False)
    _set_collection_override(collection_cfg, "env.commands.resampling_time", 0.0)
    _set_collection_override(collection_cfg, "env.commands.heading_command", False)

    playback_cfg = RslRlPlaybackConfig(
        task=_COLLECTION_TASK_NAME,
        load_run="-1",
        checkpoint=None,
        action_mode="policy",
        policy_obs_mode="actor",
        algo_log_name=str(getattr(getattr(cfg, "algo", None), "algo_log_name", "distill")),
        log_root=None,
        num_envs=1,
        checkpoint_path=str(checkpoint),
        keyboard=False,
    )
    session, _policy_obs_mode, resolved_checkpoint = create_fada_playback_session(
        playback_cfg=playback_cfg,
        cfg=collection_cfg,
        root_dir=root_dir,
        device=device,
        **({"prepare_rollout_branch": True} if prepare_rollout_branch else {}),
    )
    try:
        session.set_autoreset(False)
        env = session.env
        ctrl_dt = float(env.cfg.ctrl_dt)
        action_scale = np.asarray(env.cfg.control_config.action_scale, dtype=np.float64).reshape(-1)
        default_angles = np.asarray(env.default_angles, dtype=np.float64).reshape(-1)
        act_dim = int(env.action_space.shape[0])
        if act_dim != 29 or default_angles.shape != (act_dim,):
            raise ValueError("STTT collection requires 29 target joints and default angles")
        joint_names = list(env.get_actuated_joint_names())
        if len(joint_names) != act_dim:
            raise ValueError("Target joint names must match the action dimension")
        metadata: dict[str, Any] = {
            "seed": int(seed),
            "label_source": "mujoco_ground_truth",
            "collection_context": _collection_context(env),
            "checkpoint_path": str(resolved_checkpoint or checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "task": "g1_walk_flat/mujoco",
            "backend": _COLLECTION_BACKEND,
            "ctrl_dt": ctrl_dt,
            "obs_dim": DEFAULT_OBS_DIM,
            "act_dim": act_dim,
            "proprio_layout": {name: list(spec) for name, spec in PROPRIO_LAYOUT.items()},
            "joint_names": joint_names,
            "action_scale": action_scale.tolist(),
            "default_joint_angles": default_angles.tolist(),
            "action_representation": SPATIAL_ACTION_REPRESENTATION,
            "quaternion_order": "wxyz",
            "coordinate_frame": "reference_heading",
            "format_version": SPATIAL_DATA_FORMAT_VERSION,
        }
        return session, metadata
    except BaseException:
        session.close()
        raise


def _info_array(info: Mapping[str, Any], key: str, *, shape_tail: tuple[int, ...]) -> np.ndarray:
    value = info.get(key)
    if value is None:
        raise RuntimeError(f"env.state.info[{key!r}] is required for STTT collection")
    arr = np.asarray(value)
    if arr.ndim != 1 + len(shape_tail) or tuple(arr.shape[1:]) != shape_tail:
        raise ValueError(
            f"env.state.info[{key!r}] must have shape (num_envs, {shape_tail}), got {arr.shape}"
        )
    return arr


#: Per-row runtime-payload truth keys published by the env owner.
_PAYLOAD_ACTIVE_KEY = "runtime_payload_active"
_PAYLOAD_ATTACHED_KEY = "runtime_payload_attached_this_step"
_PAYLOAD_MASS_KEY = "runtime_payload_added_mass_kg"


def _payload_condition(info: Mapping[str, Any]) -> tuple[bool, float, bool] | None:
    """Read the actual payload truth for env row 0, or ``None`` when unpublished.

    The configured ``env.cfg.runtime_payload.enabled`` flag is never used as a
    substitute: absent truth arrays mean "unknown", not "nominal".
    """

    active = info.get(_PAYLOAD_ACTIVE_KEY)
    attached = info.get(_PAYLOAD_ATTACHED_KEY)
    added_mass = info.get(_PAYLOAD_MASS_KEY)
    if active is None or attached is None or added_mass is None:
        return None
    active_row = np.asarray(active).reshape(-1)
    attached_row = np.asarray(attached).reshape(-1)
    mass_row = np.asarray(added_mass, dtype=np.float64).reshape(-1)
    if active_row.size < 1 or attached_row.size < 1 or mass_row.size < 1:
        return None
    return bool(active_row[0]), float(mass_row[0]), bool(attached_row[0])


def _step_condition(
    pre: tuple[bool, float, bool] | None,
    post: tuple[bool, float, bool] | None,
) -> dict[str, object]:
    """Label the payload phase actually active during one executed step.

    ``stable`` requires identical pre/post active flag and added mass and no
    attach event during the step; any change is ``transitioning``. Missing
    truth arrays yield ``unknown``; the configured enabled flag is never used.
    """

    if pre is None or post is None:
        return {"phase": "unknown"}
    pre_active, pre_mass, _pre_attached = pre
    post_active, post_mass, attached_this_step = post
    if attached_this_step or pre_active != post_active or pre_mass != post_mass:
        return {"phase": "transitioning"}
    return {"phase": "stable", "payload_active": bool(pre_active), "added_mass_kg": float(pre_mass)}


def _env_vector_field(env: Any, getter_name: str, *, dim: int, field: str) -> np.ndarray:
    """Read a batched physical field through the public env owner, never the backend."""

    getter = getattr(env, getter_name, None)
    if not callable(getter):
        raise RuntimeError(
            f"STTT collection requires env.{getter_name}() for the {field}; "
            "no task-boundary owner exists on this environment."
        )
    arr = np.asarray(getter(), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise ValueError(f"env.{getter_name}() must have shape (num_envs, {dim}), got {arr.shape}")
    return arr


def _capture_proprio(session: FADAPlaybackSession) -> np.ndarray:
    """Assemble the pinned 64-dim proprio vector through public env APIs."""

    env = session.env
    dtype = np.dtype(np.float32)
    gyro = _env_vector_field(env, "get_gyro", dim=3, field="base angular velocity")
    gravity = _env_vector_field(env, "get_upvector", dim=3, field="torso upvector")
    act_dim = int(env.action_space.shape[0])
    diff = _env_vector_field(env, "get_dof_pos", dim=act_dim, field="dof positions") - np.asarray(
        env.default_angles, dtype=dtype
    ).reshape(1, -1)
    dof_vel = _env_vector_field(env, "get_dof_vel", dim=act_dim, field="dof velocities")
    parts = (
        gyro * PROPRIO_LAYOUT["base_ang_vel"][2],
        gravity * PROPRIO_LAYOUT["projected_gravity_neg"][2],
        diff * PROPRIO_LAYOUT["dof_pos_minus_default"][2],
        dof_vel * PROPRIO_LAYOUT["dof_vel"][2],
    )
    proprio = np.concatenate(parts, axis=1, dtype=dtype)[0]
    if proprio.shape != (DEFAULT_OBS_DIM,):
        raise ValueError(f"proprio layout must total {DEFAULT_OBS_DIM} dims, got {proprio.shape}")
    return np.asarray(proprio, dtype=np.float32)


def _capture_base_pose(session: FADAPlaybackSession) -> tuple[np.ndarray, np.ndarray]:
    env = session.env
    position = np.asarray(env.get_base_pos(), dtype=np.float64).reshape(-1, 3)[0].copy()
    quaternion = np.asarray(env.get_base_quat(), dtype=np.float64).reshape(-1, 4)[0].copy()
    return position, quaternion


def _to_cpu_tensor(array: np.ndarray, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(array), dtype=dtype).clone().cpu()


def _command_hold_bounds(ctrl_dt: float) -> tuple[int, int]:
    min_steps = max(int(round(_COMMAND_HOLD_SECONDS[0] / ctrl_dt)), 1)
    max_steps = max(int(round(_COMMAND_HOLD_SECONDS[1] / ctrl_dt)), min_steps)
    return min_steps, max_steps


def collect_transitions(
    session: FADAPlaybackSession,
    *,
    num_steps: int,
    seed: int = 0,
    rollout_id: str,
) -> Iterator[SpatialTransition]:
    """Yield detached ``SpatialTransition`` records with manual episode resets.

    Constant velocity commands are sampled uniformly from the effective
    ``env.cfg.commands.vel_limit`` and held 1-2 s (actual ``ctrl_dt``).
    Terminal transitions are yielded as boundary markers; the session is
    reset before the next episode starts. No static-action filtering.

    Each record also carries ``condition``, built from the env's actual
    ``runtime_payload_active`` / ``runtime_payload_added_mass_kg`` /
    ``runtime_payload_attached_this_step`` truth read before and after the
    step. The configured ``env.cfg.runtime_payload.enabled`` flag is never
    used as a stand-in: unpublished truth yields ``{"phase": "unknown"}``.
    """

    if int(num_steps) < 1:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if not str(rollout_id):
        raise ValueError("rollout_id must be a non-empty string")
    rng = np.random.default_rng(seed)

    env = session.env
    ctrl_dt = float(env.cfg.ctrl_dt)
    vel_limit = np.asarray(env.cfg.commands.vel_limit, dtype=np.float64)
    if vel_limit.shape != (2, 3):
        raise ValueError(f"commands.vel_limit must have shape (2, 3), got {vel_limit.shape}")
    command_low, command_high = vel_limit[0], vel_limit[1]
    hold_min, hold_max = _command_hold_bounds(ctrl_dt)

    session.reset()
    episode_id = -1
    episode_step = 0
    steps_left = int(num_steps)

    while steps_left > 0:
        # New episode: advance the local clock and hold a fresh command.
        episode_id += 1
        episode_step = 0
        command = rng.uniform(low=command_low, high=command_high)
        hold_steps = int(rng.integers(hold_min, hold_max + 1))

        while steps_left > 0:
            if hold_steps <= 0:
                command = rng.uniform(low=command_low, high=command_high)
                hold_steps = int(rng.integers(hold_min, hold_max + 1))

            session.set_external_command(np.asarray(command, dtype=np.float64))
            info = session.env.state.info
            command_row = _info_array(info, "commands", shape_tail=(3,))[0].copy()
            pre_payload = _payload_condition(info)
            proprio = _capture_proprio(session)
            position, quaternion = _capture_base_pose(session)

            session.step_once()

            post_state = session.env.state
            if post_state is None:
                raise RuntimeError("env state is unavailable after step_once()")
            terminated = bool(np.asarray(post_state.terminated).reshape(-1)[0])
            truncated = bool(np.asarray(post_state.truncated).reshape(-1)[0])
            action_row = _info_array(
                post_state.info, "current_actions", shape_tail=(int(env.action_space.shape[0]),)
            )[0]
            next_position, next_quaternion = _capture_base_pose(session)

            yield SpatialTransition(
                rollout_id=str(rollout_id),
                episode_id=episode_id,
                episode_step=episode_step,
                proprio=_to_cpu_tensor(proprio),
                action=_to_cpu_tensor(action_row),
                commands=_to_cpu_tensor(command_row),
                position=_to_cpu_tensor(position),
                quaternion=_to_cpu_tensor(quaternion),
                next_position=_to_cpu_tensor(next_position),
                next_quaternion=_to_cpu_tensor(next_quaternion),
                terminated=terminated,
                truncated=truncated,
                condition=_step_condition(pre_payload, _payload_condition(post_state.info)),
            )
            steps_left -= 1
            hold_steps -= 1
            episode_step += 1

            if terminated or truncated:
                # Terminal transition is a boundary marker, never a label.
                # Reset before any further stepping; never step a done env.
                session.reset()
                break


def render_transitions(
    session: FADAPlaybackSession,
    transitions: Iterator[SpatialTransition],
) -> Iterator[SpatialTransition]:
    """Pass-through wrapper that renders each collected step in a passive viewer.

    Uses the existing playback viewer model loading and the env's public
    physics-state snapshot; returns early when the viewer window is closed.
    """

    import mujoco
    import mujoco.viewer

    from unilab.visualization.playback_viewer import _load_viewer_model

    env = session.env
    model = _load_viewer_model(env, use_env_visual_model=True)
    data = mujoco.MjData(model)
    state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    ctrl_dt = float(env.cfg.ctrl_dt)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for transition in transitions:
            if not viewer.is_running():
                return
            started = time.perf_counter()
            phys = np.asarray(session.physics_state(), dtype=np.float64)[0]
            mujoco.mj_setState(model, data, phys, state_spec)
            mujoco.mj_forward(model, data)
            viewer.sync()
            yield transition
            remaining = ctrl_dt - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)
