"""Periodic desired commands integrated into a fixed, drift-independent reference."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from .trajectory import ReferenceTrajectory, TrajectoryConfig


def command_ranges(value: Any) -> np.ndarray:
    bounds = np.asarray(value, dtype=np.float64)
    if bounds.shape != (3, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError("command_ranges requires ordered finite [vx,vy,yaw_rate] lower/upper pairs")
    if bounds[0, 1] <= 0:
        raise ValueError("path pretraining requires a positive forward-command range")
    bounds = bounds.copy()
    bounds[0, 0] = max(0., bounds[0, 0])
    return bounds


def sample_command_reference(config: TrajectoryConfig, bounds: Any, *, episode_steps: int,
                             horizon_steps: int, ctrl_dt: float, resampling_s: float,
                             batch_size: int, rng: np.random.Generator,
                             lateral_range: tuple[float, float] = (0., 0.),
                             heading_range: tuple[float, float] = (0., 0.),
                             device: Any = "cpu", dtype: torch.dtype = torch.float32,
                             ) -> tuple[ReferenceTrajectory, dict[str, Any]]:
    """Sample commands, not named path families; anchor this geometry once per reset."""
    bounds = command_ranges(bounds)
    for name, value in (("episode_steps", episode_steps), ("horizon_steps", horizon_steps), ("batch_size", batch_size)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not math.isfinite(ctrl_dt) or ctrl_dt <= 0 or not math.isfinite(resampling_s) or resampling_s < ctrl_dt:
        raise ValueError("command resampling period must cover at least one positive control step")
    for values in (lateral_range, heading_range):
        if len(values) != 2 or not all(math.isfinite(v) for v in values) or values[0] > values[1]:
            raise ValueError("initial reference offsets require ordered finite bounds")
    period = round(resampling_s / ctrl_dt)
    count = episode_steps + horizon_steps
    sampled = rng.uniform(bounds[:, 0], bounds[:, 1], (batch_size, math.ceil(count / period) + 1, 3))
    instant = np.arange(count)
    segment = instant // period
    fraction = (instant % period) / period
    blend = (fraction * fraction * (3. - 2. * fraction))[None, :, None]
    velocities = sampled[:, segment] * (1. - blend) + sampled[:, segment + 1] * blend
    theta = np.concatenate((np.zeros((batch_size, 1)), np.cumsum(velocities[..., 2] * ctrl_dt, axis=1)), axis=1)
    c, s = np.cos(theta[:, :-1]), np.sin(theta[:, :-1])
    delta = ctrl_dt * np.stack((c * velocities[..., 0] - s * velocities[..., 1],
                               s * velocities[..., 0] + c * velocities[..., 1]), axis=-1)
    xy = np.concatenate((np.zeros((batch_size, 1, 2)), np.cumsum(delta, axis=1)), axis=1)
    speeds = np.linalg.norm(velocities[..., :2], axis=-1)
    if np.any(speeds * ctrl_dt <= 1e-8):
        raise ValueError("sampled stationary command cannot define an advancing path; use positive locomotion ranges")
    # Extend the fixed reference spatially for the actor's complete preview,
    # independent of speed; never re-anchor to the measured robot position.
    extra = config.preview_points + 1
    distance = config.preview_spacing_m
    extension = np.empty((batch_size, extra, 2), dtype=np.float64)
    position, angle = xy[:, -1].copy(), theta[:, -1].copy()
    step_time = distance / speeds[:, -1]
    for index in range(extra):
        c, s = np.cos(angle), np.sin(angle)
        position += step_time[:, None] * np.stack((c * velocities[:, -1, 0] - s * velocities[:, -1, 1],
                                                   s * velocities[:, -1, 0] + c * velocities[:, -1, 1]), axis=-1)
        extension[:, index] = position
        angle += step_time * velocities[:, -1, 2]
    xy = np.concatenate((xy, extension), axis=1)
    speeds = np.concatenate((speeds, np.repeat(speeds[:, -1:], extra + 1, axis=1)), axis=1)
    lateral = rng.uniform(*lateral_range, size=batch_size)
    heading = rng.uniform(*heading_range, size=batch_size)
    first = xy[:, 1] - xy[:, 0]
    rotation = np.arctan2(first[:, 1], first[:, 0]) - heading
    c, s = np.cos(rotation)[:, None], np.sin(rotation)[:, None]
    xy = np.stack((c * xy[..., 0] + s * xy[..., 1], -s * xy[..., 0] + c * xy[..., 1] - lateral[:, None]), axis=-1)
    reference = ReferenceTrajectory(torch.as_tensor(xy, device=device, dtype=dtype),
                                    torch.as_tensor(speeds, device=device, dtype=dtype), config,
                                    tuple(tuple(float(value) for value in pair) for pair in bounds))
    return reference, {"resampling_control_steps": period, "desired_commands": sampled.tolist(),
                       "interpolation": "smoothstep", "initial_lateral_m": lateral.tolist(),
                       "initial_heading_rad": heading.tolist(), "command_ranges": bounds.tolist(),
                       "reference": "fixed_from_desired_commands_at_episode_reset"}
