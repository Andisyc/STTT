"""Offline B-data assembly over the existing public single-env playback loop."""
from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.distill.fada.official_playback import OfficialFADAPlaybackController
from unilab.algos.torch.sttt.odometry_runtime import create_pose_runtime
from unilab.algos.torch.sttt.path_following.collection import WindowCollector
from unilab.algos.torch.sttt.path_following.contract import _integer
from unilab.algos.torch.sttt.path_following.dataset import PathFollowingDatasetWriter
from unilab.algos.torch.sttt.pose_data import ControlInterval, OnlinePoseRuntime, PoseRuntimeConfig
from unilab.visualization.playback_distill_sessions import create_fada_playback_session
from unilab.visualization.playback_sessions import RslRlPlaybackConfig


def _velocity_command(env: Any) -> np.ndarray:
    command = np.asarray(env.state.info["commands"], dtype=np.float64)
    if command.ndim != 2 or command.shape[0] != 1 or command.shape[1] < 3:
        raise ValueError("response collection requires one velocity command row")
    result = command[0, :3].copy()
    if not np.isfinite(result).all() or result[0] <= 0 or not np.allclose(result[1:], 0., atol=1e-8, rtol=0):
        raise ValueError("configure a fixed forward command [vx > 0, 0, 0]; collection never rewrites it")
    return result


def collect_path_following_data(
    cfg: DictConfig,
    *,
    root_dir: str | Path,
    output_dir: str | Path,
    num_steps: int,
    window_steps: int,
    residual_amplitude_rad: float,
    seed: int,
    shard_size: int = 256,
    device: str | None = None,
    pose_config: PoseRuntimeConfig | None = None,
    playback_deps: Mapping[str, Any] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Collect uniform per-joint random interventions; never train either policy.

    N, amplitude (physical radians) and the local residual RNG seed are explicit.
    The caller's task config and velocity command are not mutated. This API is
    simulation-ground-truth pretraining only; it does not claim live SLAM support.
    ``playback_deps`` forwards the existing factory seam for offline I/O checks.
    """
    steps = _integer(num_steps, "num_steps")
    n = _integer(window_steps, "window_steps")
    seed = _integer(seed, "seed")
    size = _integer(shard_size, "shard_size")
    if n < 1 or steps < n or size < 1:
        raise ValueError("num_steps must cover a positive window; shard_size must be positive")
    if isinstance(residual_amplitude_rad, bool) or not math.isfinite(residual_amplitude_rad) or residual_amplitude_rad <= 0:
        raise ValueError("residual_amplitude_rad must be explicit, positive and finite; no safety bound is guessed")
    if not isinstance(cfg, DictConfig):
        raise TypeError("response collection requires a composed Hydra DictConfig")
    if cfg.training.task_name != "G1WalkFlat" or cfg.training.sim_backend != "mujoco":
        raise ValueError("response collection requires task=g1_walk_flat/mujoco")
    for path in ("interactive.keyboard", "demo.figure_eight.enabled", "demo.footprints.enabled", "env.commands.heading_command"):
        if bool(OmegaConf.select(cfg, path, default=False)):
            raise ValueError(f"disable {path} explicitly for fixed-command response collection")
    if OmegaConf.select(cfg, "env.commands.resampling_time", default=None) != 0:
        raise ValueError("configure env.commands.resampling_time=0; collection never changes command rules")
    source_pose = pose_config if pose_config is not None else PoseRuntimeConfig(odometry_source="simulation")
    if source_pose.odometry_source != "simulation":
        raise ValueError("this offline collection API requires simulation poses; live SLAM belongs to online integration")
    raw_path = OmegaConf.select(cfg, "training.play_checkpoint_path", default=None)
    if not raw_path:
        raise FileNotFoundError("set training.play_checkpoint_path to an official Planner-IDM checkpoint")
    root = Path(root_dir).expanduser().resolve()
    checkpoint = Path(str(raw_path)).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"response output directory must be empty: {output}")
    capture_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, enum_to_str=True))
    OmegaConf.update(capture_cfg, "online_adaptation.enabled", False, force_add=True)
    OmegaConf.update(capture_cfg, "training.play_checkpoint_path", str(checkpoint))
    collector = WindowCollector(n, max_alignment_age_ns=round(source_pose.max_alignment_age_s * 1e9))
    playback = RslRlPlaybackConfig(
        task="G1WalkFlat", load_run="-1", checkpoint=None, action_mode="policy", policy_obs_mode="actor",
        algo_log_name="distill", log_root=None, num_envs=1, checkpoint_path=str(checkpoint), keyboard=False,
    )
    session, _, resolved_checkpoint = create_fada_playback_session(
        playback_cfg=playback, cfg=capture_cfg, root_dir=root, device=device,
        future_transform=collector.transform, deps=playback_deps, log=log,
    )
    poses: OnlinePoseRuntime | None = None
    writer: PathFollowingDatasetWriter | None = None
    successful = False
    executed_steps = 0
    terminated = False
    try:
        if not isinstance(session.controller, OfficialFADAPlaybackController):
            raise TypeError("response collection requires the official playback controller")
        controller = session.controller
        preprocessing = controller.spec.preprocessing
        if preprocessing is None or resolved_checkpoint is None:
            raise ValueError("capture requires the actual checkpoint path and bound preprocessing")
        with Path(resolved_checkpoint).open("rb") as handle:
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        controller.policy.eval().requires_grad_(False)
        session.set_autoreset(False)
        session.reset()
        initial_command = _velocity_command(session.env)
        poses = create_pose_runtime(session.env, source_pose)
        poses.reset()
        profile = controller.profile
        source = {
            "task": "g1_walk_flat/mujoco", "backend": "mujoco", "label_source": "simulation_ground_truth",
            "checkpoint_path": str(resolved_checkpoint), "checkpoint_sha256": digest.hexdigest(),
            "source_profile": asdict(profile), "preprocessing": dict(preprocessing),
            "target_joint_names": list(session.env.get_actuated_joint_names()),
            "target_default_joint_angles": np.asarray(session.env.default_angles).reshape(-1).tolist(),
            "target_action_scale": np.broadcast_to(np.asarray(session.env.cfg.control_config.action_scale), (29,)).tolist(),
            "residual_seed": seed, "residual_distribution": "independent_uniform_per_joint",
            "residual_amplitude_rad": float(residual_amplitude_rad), "velocity_command": initial_command.tolist(),
            "pose_config": asdict(source_pose),
            "source_config": OmegaConf.to_container(capture_cfg, resolve=True, enum_to_str=True),
            "delta_frame": "window_start_horizontal_heading", "delta_units": ["m", "m", "rad"],
            "reference": "task_initial_position_and_heading_never_reanchored",
        }
        writer = PathFollowingDatasetWriter(output, source=source, shard_size=size)
        rng = np.random.default_rng(seed)
        model_reference = next(controller.policy.parameters())
        for transition in range(steps):
            before = poses.begin_step()
            command = _velocity_command(session.env)
            if not np.array_equal(command, initial_command):
                raise ValueError("velocity command changed during fixed-command response collection")
            if collector.window.at_boundary:
                residual = rng.uniform(-residual_amplitude_rad, residual_amplitude_rad, (1, 29))
                collector.submit_candidate(torch.as_tensor(
                    residual, dtype=model_reference.dtype, device=model_reference.device,
                ))
            collector.begin_step(before)
            session.step_once()
            after = poses.clock
            if after.sequence_id != before.sequence_id or after.timestamp_ns <= before.timestamp_ns:
                raise RuntimeError("physical reset or non-advancing clock inside execution")
            action = np.asarray(session.env.state.info["current_actions"], dtype=np.float64)
            if action.shape != (1, 29):
                raise ValueError("executed env-input action must be [1, 29] in target order")
            control = ControlInterval(
                before.sequence_id, transition, before.timestamp_ns, after.timestamp_ns,
                action[0].copy(), command,
            )
            update = poses.complete_step(control)
            terminated = bool(np.any(session.env.state.terminated | session.env.state.truncated))
            for sample in collector.complete_step(control, update, terminal=terminated):
                writer.append(sample)
            executed_steps += 1
            if terminated or collector.halted:
                break
        collector.close()
        summary = {
            "samples": writer.sample_count, "executed_steps": executed_steps,
            "episode_terminated": terminated, "localization_invalidated": collector.halted and not terminated,
            "dropped_windows": dict(collector.dropped_windows), "output_dir": str(output),
        }
        writer.close(capture_summary=summary)
        successful = True
        return summary
    finally:
        collector.close()
        if writer is not None and not successful:
            writer.abort()
        try:
            if poses is not None:
                poses.close()
        finally:
            session.close()
