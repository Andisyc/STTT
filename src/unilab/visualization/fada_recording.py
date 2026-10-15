"""Finite, offscreen video recording of the official online FADA session."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

import mediapy as media
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from unilab.envs.common.rotation import np_wrap_to_pi, np_yaw_from_quat

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
    progress_interval_s: float = 1.0
    show_status_overlay: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError("video_recording.duration_s must be finite and positive")
        if not math.isfinite(self.progress_interval_s) or self.progress_interval_s <= 0.0:
            raise ValueError("video_recording.progress_interval_s must be finite and positive")
        for name in ("fps", "width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"video_recording.{name} must be a positive integer")
        if self.width % 2 or self.height % 2:
            raise ValueError("MP4 recording width and height must be even")
        if not self.output_path or Path(self.output_path).suffix.lower() != ".mp4":
            raise ValueError("video_recording.output_path must name an MP4 file")


def _annotate_frame(
    image: np.ndarray,
    *,
    title: str,
    details: tuple[str, ...],
    alert: str | None,
    loaded: bool,
    font: Any,
) -> np.ndarray:
    """Draw a screen-space status panel; never add physical payload geometry."""
    frame = Image.fromarray(image)
    draw = ImageDraw.Draw(frame)
    line_height = font.getbbox("Ag")[3] + 8
    lines = (title, *details, *((alert,) if alert else ()))
    panel_height = 12 + len(lines) * line_height
    draw.rectangle((6, 6, frame.width - 6, panel_height), fill=(20, 23, 28))
    draw.rectangle((6, 6, 12, panel_height), fill=(255, 165, 40) if loaded else (70, 210, 120))
    for index, text in enumerate(lines):
        color = (255, 200, 80) if index == 0 else (240, 240, 240)
        if alert and index == len(lines) - 1:
            color = (255, 105, 90)
        draw.text((20, 10 + index * line_height), text, fill=color, font=font)
    return np.asarray(frame)


def record_fada_playback(
    session: Any,
    args: Any,
    config: FADAVideoRecordingConfig,
    *,
    log: Callable[[str], None] = partial(print, flush=True),
) -> str:
    """Advance simulation without wall-clock pacing and stream frames to FFmpeg.

    Frame timestamps use simulation time, not the time spent adapting the policy.
    Only the current RGB frame is retained, regardless of recording duration.
    Progress is flushed at completed-step boundaries on a wall-clock interval;
    a single long-running adaptation step cannot report completion until it returns.
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
    payload_config = getattr(session.env.cfg, "runtime_payload", None)
    payload_body_id = -1
    payload_position = payload_half_size = None
    payload_rgba = np.asarray([1.0, 0.35, 0.05, 1.0], dtype=np.float32)
    if payload_config is not None and payload_config.enabled:
        payload_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, payload_config.body_name)
        if payload_body_id < 0:
            raise ValueError(f"Payload body is absent from recording model: {payload_config.body_name}")
        payload_position = np.asarray(payload_config.attachment_pos_m, dtype=np.float64)
        payload_half_size = np.asarray(payload_config.half_size_m, dtype=np.float64)
    camera.distance = (
        args.camera_distance if args.camera_distance is not None
        else _default_viewer_camera_distance(model, session.env, follow_body=args.camera_follow_body)
    )
    if args.camera_elevation is not None:
        camera.elevation = args.camera_elevation
    if args.camera_azimuth is not None:
        camera.azimuth = args.camera_azimuth

    session.reset()
    reset_count = 0
    last_reset_time: float | None = None
    last_attach_time: float | None = None
    update_count = 0
    last_loss: float | None = None
    origin_xy: np.ndarray | None = None
    origin_yaw = 0.0
    font = (
        ImageFont.load_default(size=max(12, config.height // 40))
        if config.show_status_overlay else None
    )
    log(f"Recording {config.duration_s:g}s simulation to {output} ({config.width}x{config.height}, {config.fps} fps); no viewer or realtime pacing")
    frame_index = 0
    started = time.monotonic()
    last_progress = started

    def report_progress(step: int, now: float) -> None:
        log(
            f"Progress: step={step}/{num_steps} ({100.0 * step / num_steps:.1f}%) "
            f"sim_time={step * ctrl_dt:.2f}/{config.duration_s:g}s "
            f"frames={frame_index}/{num_frames} wall_time={now - started:.1f}s"
        )

    report_progress(0, started)
    with mujoco.Renderer(model, height=config.height, width=config.width) as renderer:
        with media.VideoWriter(str(output), shape=(config.height, config.width), fps=config.fps) as writer:
            for step_index in range(num_steps + 1):
                if step_index:
                    session.step_once()
                    if session.last_update_stats is not None:
                        log(f"sim_time={step_index * ctrl_dt:.3f}s Update: {session.last_update_stats}")
                        update_count += 1
                        last_loss = float(session.last_update_stats["loss"])
                    if np.any(session.info.get("runtime_payload_attached_this_step", False)):
                        log(f"sim_time={step_index * ctrl_dt:.3f}s Payload attached: {session.info['runtime_payload_added_mass_kg']} kg")
                        last_attach_time = step_index * ctrl_dt

                    state = session.env.state
                    assert state is not None
                    if np.any(state.terminated | state.truncated):
                        reset_count += 1
                        last_reset_time = step_index * ctrl_dt
                        reason = "terminated" if np.any(state.terminated) else "timeout"
                        log(
                            f"sim_time={last_reset_time:.3f}s EPISODE RESET ({reason}); "
                            f"reset_count={reset_count}. Do not treat the trajectory as uninterrupted walking."
                        )
                elapsed = step_index * ctrl_dt
                if frame_index < num_frames and frame_index / config.fps <= elapsed + 1e-9:
                    physics = np.asarray(session.physics_state()[0], dtype=np.float64)
                    mujoco.mj_setState(model, data, physics, mujoco.mjtState.mjSTATE_FULLPHYSICS)
                    mujoco.mj_forward(model, data)
                    if args.camera_follow_body:
                        camera.lookat[:] = data.xpos[focus_body_id]
                        camera.lookat[2] += args.camera_height_offset
                    renderer.update_scene(data, camera=camera)
                    active = bool(np.asarray(session.info.get("runtime_payload_active", [False])).reshape(-1)[0])
                    if active and payload_body_id >= 0:
                        scene = renderer.scene
                        if scene.ngeom >= scene.maxgeom:
                            raise RuntimeError("Recording scene has no room for the payload marker")
                        rotation = data.xmat[payload_body_id].reshape(3, 3)
                        center = data.xpos[payload_body_id] + rotation @ payload_position
                        # Visualize the same rigid box used in inertia composition; no extra physics mass.
                        mujoco.mjv_initGeom(
                            scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_BOX,
                            payload_half_size, center, rotation.reshape(9), payload_rgba,
                        )
                        scene.ngeom += 1
                    image = renderer.render()
                    if config.show_status_overlay:
                        position = data.xpos[focus_body_id]
                        yaw = float(np_yaw_from_quat(data.xquat[focus_body_id:focus_body_id + 1])[0])
                        if origin_xy is None:
                            origin_xy = position[:2].copy()
                            origin_yaw = yaw
                        dx, dy = position[:2] - origin_xy
                        forward = math.cos(origin_yaw) * dx + math.sin(origin_yaw) * dy
                        lateral = -math.sin(origin_yaw) * dx + math.cos(origin_yaw) * dy
                        yaw_error = math.degrees(float(np_wrap_to_pi(yaw - origin_yaw)))
                        added_mass = float(np.asarray(session.info.get("runtime_payload_added_mass_kg", [0.0])).reshape(-1)[0])
                        status = f"PAYLOAD +{added_mass:g} kg" if active else "NO PAYLOAD"
                        if active and payload_config is not None:
                            status += f" @ {payload_config.body_name}"
                        alert = None
                        if last_attach_time is not None and elapsed - last_attach_time < 2.0:
                            alert = f"PAYLOAD ATTACHED +{added_mass:g} kg"
                        if last_reset_time is not None and elapsed - last_reset_time < 2.0:
                            alert = "EPISODE RESET - trajectory interrupted"
                        loss_text = f" loss={last_loss:.4g}" if last_loss is not None else ""
                        image = _annotate_frame(
                            image,
                            title=f"t={elapsed:.2f}s | {status}",
                            details=(
                                f"forward={forward:+.2f}m lateral={lateral:+.2f}m",
                                f"yaw error={yaw_error:+.1f}deg resets={reset_count}",
                                f"LoRA updates={update_count}{loss_text}",
                            ),
                            alert=alert, loaded=active, font=font,
                        )
                    while frame_index < num_frames and frame_index / config.fps <= elapsed + 1e-9:
                        writer.add_image(image)
                        frame_index += 1

                now = time.monotonic()
                if now - last_progress >= config.progress_interval_s or step_index == num_steps:
                    report_progress(step_index, now)
                    last_progress = now

            log("Finalizing MP4 encoding...")

    log(f"Video saved: {output} ({frame_index} frames, {num_steps} control steps)")
    return str(output)
