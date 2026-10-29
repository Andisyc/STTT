"""Offline, backend-independent piecewise-constant demo velocity commands."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from numbers import Integral
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FigureEightCommandConfig:
    width_m: float = 6.0
    height_m: float = 3.0
    speed_m_s: float = 0.3
    segment_duration_s: float = 0.8


class FigureEightCommandProgram:
    """Repeat a closed Gerono eight using constant body-frame circular arcs.

    Geometry is sampled once at construction. The first quarter is resampled
    by arc length; reflection supplies the other quarters. Circular arcs fit
    each chord exactly, so integrating the actual velocity table closes both
    position and heading. The entire route is rotated to start along +x.
    Episode-local progress belongs to the playback session, not this table.
    """

    @classmethod
    def from_config(
        cls, config: FigureEightCommandConfig, *, ctrl_dt: float,
        prediction_horizon: int = 1, window_size: int = 0, update_interval: int = 0,
    ) -> FigureEightCommandProgram:
        for name in ("width_m", "height_m", "speed_m_s", "segment_duration_s"):
            value = float(getattr(config, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"demo.figure_eight.{name} must be finite and positive")
        if not np.isfinite(ctrl_dt) or ctrl_dt <= 0.0:
            raise ValueError("figure-eight ctrl_dt must be finite and positive")
        segment_steps = max(1, int(ceil(config.segment_duration_s / ctrl_dt - 1e-10)))
        required_steps = max(prediction_horizon, prediction_horizon + window_size - 1,
                             prediction_horizon + update_interval)
        if segment_steps < required_steps:
            raise ValueError(
                f"demo.figure_eight segment holds {segment_steps} control steps; "
                f"requires at least {required_steps} for prediction_horizon="
                f"{prediction_horizon}, window_size={window_size}, update_interval={update_interval}"
            )
        segment_dt = segment_steps * float(ctrl_dt)
        parameter = np.linspace(0.0, np.pi / 2.0, 4097)
        a, b = config.width_m / 2.0, config.height_m
        dense = np.column_stack((a * np.sin(parameter), b * np.sin(parameter) * np.cos(parameter)))
        dense[[0, -1], 1] = 0.0
        arc_length = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))))
        if not np.isfinite(arc_length).all() or arc_length[-1] <= 0.0:
            raise ValueError("figure-eight geometry must have finite positive arc length")
        # At least eight arcs per quarter keep even a tiny route an eight.
        count = max(8, int(ceil(arc_length[-1] / (config.speed_m_s * segment_dt))))
        distances = np.linspace(0.0, arc_length[-1], count + 1)
        points = np.column_stack([np.interp(distances, arc_length, dense[:, i]) for i in range(2)])
        chords = np.diff(points, axis=0)
        chord_length = np.linalg.norm(chords, axis=1)
        chord_heading = np.unwrap(np.arctan2(chords[:, 1], chords[:, 0]))
        heading = np.empty(count + 1)
        heading[-1] = -np.pi / 2.0
        # A constant-twist arc's chord points along its mean boundary heading.
        # Fit backwards from the vertical tangent at the quarter endpoint.
        for index in range(count - 1, -1, -1):
            heading[index] = 2.0 * chord_heading[index] - heading[index + 1]
        turn = np.diff(heading)
        if np.any(chord_length <= 0.0) or np.any(np.abs(turn) >= np.pi):
            raise ValueError("figure-eight arcs are under-resolved; reduce segment_duration_s")
        velocity = chord_length / (segment_dt * np.sinc(turn / (2.0 * np.pi)))
        yaw = turn / segment_dt
        commands = np.column_stack((
            np.concatenate((velocity, velocity[::-1], velocity, velocity[::-1])),
            np.zeros(4 * count),
            np.concatenate((yaw, yaw[::-1], -yaw, -yaw[::-1])),
        ))
        if not np.isfinite(commands).all() or np.any(commands[:, 0] <= 0.0):
            raise ValueError("figure-eight commands must be finite with positive forward speed")
        route = np.concatenate((points, points[-2::-1] * [1, -1],
                                points[1:] * [-1, 1], points[-2::-1] * [-1, -1]))
        angle = heading[0]
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        program = cls()
        program.commands = commands
        program.reference_points = route @ rotation
        program.ctrl_dt = float(ctrl_dt)
        program.segment_steps = segment_steps
        program.segment_duration_s = segment_dt
        program.duration_s = len(commands) * segment_dt
        program.total_control_steps = len(commands) * segment_steps
        program.commands.setflags(write=False)
        program.reference_points.setflags(write=False)
        return program

    def segment_index_at_control_step(self, control_step: int) -> int:
        if isinstance(control_step, bool) or not isinstance(control_step, Integral) or control_step < 0:
            raise ValueError("figure-eight control_step must be a non-negative integer")
        return (int(control_step) // self.segment_steps) % len(self.commands)

    def command_at_control_step(self, control_step: int) -> np.ndarray:
        return self.commands[self.segment_index_at_control_step(control_step)].copy()

    def time_at_control_step(self, control_step: int) -> float:
        self.segment_index_at_control_step(control_step)
        return int(control_step) * self.ctrl_dt

    def validate_environment(self, env_cfg: Any, *, keyboard: bool) -> None:
        """Reject competing command owners and canonicalization that changes the route."""
        if keyboard:
            raise ValueError("demo.figure_eight requires interactive.keyboard=false")
        command_cfg = env_cfg.commands
        if float(command_cfg.resampling_time) != 0.0:
            raise ValueError("demo.figure_eight requires env.commands.resampling_time=0.0")
        if bool(command_cfg.heading_command):
            raise ValueError("demo.figure_eight requires env.commands.heading_command=false")
        limits = np.asarray(command_cfg.vel_limit, dtype=np.float64)
        if (limits.shape != (2, 3) or not np.isfinite(limits).all()
                or np.any(limits[0] > limits[1])):
            raise ValueError("env.commands.vel_limit must contain ordered finite 3D bounds")
        if np.any(self.commands < limits[0]) or np.any(self.commands > limits[1]):
            raise ValueError(
                "demo.figure_eight commands exceed env.commands.vel_limit: "
                f"required min={self.commands.min(axis=0)}, max={self.commands.max(axis=0)}"
            )
        # Use the command owner's pure functions, including its joint XY/yaw
        # dead zone (which does not zero yaw independently).
        from unilab.envs.locomotion.g1.walk_commands import (
            apply_g1_command_dead_zone,
            canonicalize_g1_commands,
        )
        from unilab.envs.locomotion.common.commands import zero_small_xy_commands

        phase_cfg = getattr(env_cfg, "command_gated_phase_contact", None)
        if phase_cfg is not None and bool(phase_cfg.enabled):
            canonical = canonicalize_g1_commands(
                self.commands, xy_dead_zone=phase_cfg.command_xy_dead_zone,
                yaw_dead_zone=phase_cfg.command_yaw_dead_zone,
            )
        else:
            canonical = apply_g1_command_dead_zone(self.commands.copy(), command_cfg)
            threshold = float(getattr(command_cfg, "small_xy_threshold", 0.0))
            if threshold > 0.0:
                canonical = canonical.copy()
                zero_small_xy_commands(canonical, threshold=threshold)
        if not np.allclose(canonical, self.commands, rtol=1e-6, atol=1e-9):
            raise ValueError(
                "demo.figure_eight commands are altered by env command dead zones / "
                "small_xy_threshold; reduce the applicable thresholds"
            )
