"""Finite, offscreen video recording of the official online FADA session."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import mediapy as media
import mujoco
import numpy as np

from unilab.visualization.playback_viewer import (
    _default_viewer_camera_distance,
    _load_viewer_model,
    _resolve_focus_body_id,
)


@dataclass(frozen=True)
class FADAVideoRecordingConfig:
    enabled: bool = True
    output_path: str = "logs/fada_online/online_adaptation.mp4"
    duration_s: float = 30.0
    fps: int = 30
    width: int = 1280
    height: int = 720

    def __post_init__(self) -> None:
        if not math.isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError("video_recording.duration_s must be finite and positive")
        for name in ("fps", "width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"video_recording.{name} must be a positive integer")
        if self.width % 2 or self.height % 2:
            raise ValueError("MP4 recording width and height must be even")
        if not self.output_path or Path(self.output_path).suffix.lower() != ".mp4":
            raise ValueError("video_recording.output_path must name an MP4 file")


def record_fada_playback(
    session: Any,
    args: Any,
    config: FADAVideoRecordingConfig,
    *,
    log: Callable[[str], None] = print,
) -> str:
    """Advance simulation without wall-clock pacing and stream frames to FFmpeg.

    Frame timestamps use simulation time, not the time spent adapting the policy.
    Only the current RGB frame is retained, regardless of recording duration.
    """
    ctrl_dt = float(session.env.cfg.ctrl_dt)
    if not math.isfinite(ctrl_dt) or ctrl_dt <= 0.0:
        raise ValueError("FADA recording requires finite positive env.ctrl_dt")
    output = Path(config.output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    num_steps = math.ceil(config.duration_s / ctrl_dt)
    num_frames = math.ceil(config.duration_s * config.fps)

    model = _load_viewer_model(session.env, use_env_visual_model=args.use_env_visual_model)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, config.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, config.height)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    focus_body_id = _resolve_focus_body_id(model, session.env, args.camera_focus_body_name)
    camera.distance = (
        args.camera_distance if args.camera_distance is not None
        else _default_viewer_camera_distance(model, session.env, follow_body=args.camera_follow_body)
    )
    if args.camera_elevation is not None:
        camera.elevation = args.camera_elevation
    if args.camera_azimuth is not None:
        camera.azimuth = args.camera_azimuth

    session.reset()
    log(f"Recording {config.duration_s:g}s simulation to {output} ({config.width}x{config.height}, {config.fps} fps); no viewer or realtime pacing")
    frame_index = 0
    with mujoco.Renderer(model, height=config.height, width=config.width) as renderer:
        with media.VideoWriter(str(output), shape=(config.height, config.width), fps=config.fps) as writer:
            for step_index in range(num_steps + 1):
                if step_index:
                    session.step_once()
                    if session.last_update_stats is not None:
                        log(f"sim_time={step_index * ctrl_dt:.3f}s Update: {session.last_update_stats}")
                    if np.any(session.info.get("runtime_payload_attached_this_step", False)):
                        log(f"sim_time={step_index * ctrl_dt:.3f}s Payload attached: {session.info['runtime_payload_added_mass_kg']} kg")

                elapsed = step_index * ctrl_dt
                if frame_index >= num_frames or frame_index / config.fps > elapsed + 1e-9:
                    continue
                physics = np.asarray(session.physics_state()[0], dtype=np.float64)
                mujoco.mj_setState(model, data, physics, mujoco.mjtState.mjSTATE_FULLPHYSICS)
                mujoco.mj_forward(model, data)
                if args.camera_follow_body:
                    camera.lookat[:] = data.xpos[focus_body_id]
                    camera.lookat[2] += args.camera_height_offset
                renderer.update_scene(data, camera=camera)
                image = renderer.render()
                while frame_index < num_frames and frame_index / config.fps <= elapsed + 1e-9:
                    writer.add_image(image)
                    frame_index += 1

    log(f"Video saved: {output} ({frame_index} frames, {num_steps} control steps)")
    return str(output)
