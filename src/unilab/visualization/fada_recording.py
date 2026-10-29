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

from unilab.visualization.fada_rendering import (
    FADADemoRenderConfig,
    FADAPlaybackRendering,
)
from unilab.visualization.playback_viewer import _load_viewer_model


@dataclass(frozen=True)
class FADAVideoRecordingConfig:
    enabled: bool = True
    output_path: str = "logs/fada_online/online_adaptation.mp4"
    duration_s: float | None = 30.0
    fps: int = 30
    width: int = 1280
    height: int = 720
    progress_interval_s: float = 1.0
    show_status_overlay: bool = True

    def __post_init__(self) -> None:
        if self.duration_s is not None and (
            not math.isfinite(self.duration_s) or self.duration_s <= 0.0
        ):
            raise ValueError("video_recording.duration_s must be finite and positive, or null for one figure-eight cycle")
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


@dataclass(frozen=True)
class FADARecordingTiming:
    duration_s: float
    control_steps: int
    frame_count: int
    include_terminal_frame: bool = False


def resolve_fada_recording_timing(session: Any, config: FADAVideoRecordingConfig) -> FADARecordingTiming:
    """Resolve one full command cycle without advancing the policy or environment."""
    ctrl_dt = float(session.env.cfg.ctrl_dt)
    if not math.isfinite(ctrl_dt) or ctrl_dt <= 0.0:
        raise ValueError("FADA recording requires finite positive env.ctrl_dt")
    if config.duration_s is not None:
        return FADARecordingTiming(
            config.duration_s, math.ceil(config.duration_s / ctrl_dt),
            math.ceil(config.duration_s * config.fps),
        )
    program = getattr(session, "command_program", None)
    if program is None:
        raise ValueError("video_recording.duration_s=null requires demo.figure_eight.enabled=true")
    if not math.isclose(program.ctrl_dt, ctrl_dt, rel_tol=1e-9):
        raise ValueError("figure-eight program and recording must use the same ctrl_dt")
    steps = int(program.total_control_steps)
    duration = steps * ctrl_dt
    episode_seconds = float(session.env.cfg.max_episode_seconds)
    episode_steps = int(getattr(session.env.cfg, "max_episode_steps", math.floor(episode_seconds / ctrl_dt)))
    if episode_steps <= steps:
        raise ValueError(
            f"Full figure-eight recording needs env.max_episode_seconds > {duration:g}s; "
            f"got {episode_seconds:g}s ({episode_steps} steps)"
        )
    # The extra encoded frame holds the completed cycle's final state. For a
    # nonintegral FPS boundary it is flushed at the final control step, never
    # by stepping into the next command cycle.
    return FADARecordingTiming(duration, steps, math.ceil(duration * config.fps) + 1, True)


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
    render_config: FADADemoRenderConfig | None = None,
    log: Callable[[str], None] = partial(print, flush=True),
) -> str:
    """Advance simulation without wall-clock pacing and stream frames to FFmpeg.

    Frame timestamps use simulation time, not the time spent adapting the policy.
    Only the current RGB frame is retained, regardless of recording duration.
    Progress is flushed at completed-step boundaries on a wall-clock interval;
    a single long-running adaptation step cannot report completion until it returns.
    """
    timing = resolve_fada_recording_timing(session, config)
    ctrl_dt = float(session.env.cfg.ctrl_dt)
    duration_s = timing.duration_s
    output = Path(config.output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    num_steps, num_frames = timing.control_steps, timing.frame_count

    model = _load_viewer_model(session.env, use_env_visual_model=args.use_env_visual_model)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, config.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, config.height)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    session.reset()
    rendering = FADAPlaybackRendering(session, model, args, render_config)
    rendering.initialize_camera(camera)
    font = (
        ImageFont.load_default(size=max(12, config.height // 40))
        if config.show_status_overlay else None
    )
    log(f"Recording {duration_s:g}s simulation to {output} ({config.width}x{config.height}, {config.fps} fps); no viewer or realtime pacing")
    frame_index = 0
    started = time.monotonic()
    last_progress = started

    def report_progress(step: int, now: float) -> None:
        log(
            f"Progress: step={step}/{num_steps} ({100.0 * step / num_steps:.1f}%) "
            f"sim_time={step * ctrl_dt:.2f}/{duration_s:g}s "
            f"frames={frame_index}/{num_frames} wall_time={now - started:.1f}s"
        )

    report_progress(0, started)
    with mujoco.Renderer(model, height=config.height, width=config.width) as renderer:
        with media.VideoWriter(str(output), shape=(config.height, config.width), fps=config.fps) as writer:
            for step_index in range(num_steps + 1):
                if step_index:
                    session.step_once()
                    rendering.observe_step()
                    if session.last_update_stats is not None:
                        log(f"sim_time={step_index * ctrl_dt:.3f}s Update: {session.last_update_stats}")
                    if np.any(session.info.get("runtime_payload_attached_this_step", False)):
                        log(f"sim_time={step_index * ctrl_dt:.3f}s Payload attached: {session.info['runtime_payload_added_mass_kg']} kg")

                    state = session.env.state
                    assert state is not None
                    if np.any(state.terminated | state.truncated):
                        reason = "terminated" if np.any(state.terminated) else "timeout"
                        log(
                            f"sim_time={step_index * ctrl_dt:.3f}s EPISODE {'RESET' if session.autoreset_enabled else 'TERMINATED'} ({reason}); "
                            f"reset_count={rendering.reset_count}. Do not treat the trajectory as uninterrupted walking."
                        )
                elapsed = step_index * ctrl_dt
                final_frame = timing.include_terminal_frame and step_index == num_steps
                if frame_index < num_frames and (frame_index / config.fps <= elapsed + 1e-9 or final_frame):
                    rendering.sync_physics(data)
                    rendering.update_camera(camera, data)
                    renderer.update_scene(data, camera=camera)
                    rendering.append_scene(renderer.scene, data)
                    image = renderer.render()
                    if config.show_status_overlay:
                        status = rendering.status(data)
                        image = _annotate_frame(
                            image, title=status.title, details=status.details,
                            alert=status.alert, loaded=status.loaded, font=font,
                        )
                    while frame_index < num_frames and (frame_index / config.fps <= elapsed + 1e-9 or final_frame):
                        writer.add_image(image)
                        frame_index += 1

                now = time.monotonic()
                if now - last_progress >= config.progress_interval_s or step_index == num_steps:
                    report_progress(step_index, now)
                    last_progress = now

            log("Finalizing MP4 encoding...")

    log(f"Video saved: {output} ({frame_index} frames, {num_steps} control steps)")
    return str(output)
