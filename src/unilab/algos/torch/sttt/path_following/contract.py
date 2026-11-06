"""Owned query and executed-window records for path-following response data."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np
import torch

from unilab.algos.torch.distill.fada.runtime_policy import FADAFutureQuery
from unilab.algos.torch.sttt.path_following.reference import (
    FixedStraightPath, measured_window_delta, pose_identity, pose_matrix,
)
from unilab.algos.torch.sttt.pose_data import ControlInterval, TimedPelvisPose


def _validate_scales(ctrl_dt: float, position_scale: float, velocity_scale: float) -> None:
    for name, value in (("ctrl_dt", ctrl_dt), ("position_scale", position_scale), ("velocity_scale", velocity_scale)):
        if isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if ctrl_dt <= 0:
        raise ValueError("ctrl_dt must be positive")


@dataclass(frozen=True)
class InterventionQueryRecord:
    """Detached copies of a query and its intentional source-order intervention.

    ``planner_future`` and ``adapted_future`` use checkpoint-scaled compact64.
    ``residual_rad`` uses physical radians in ``joint_names`` order. This record
    proves a query only; execution time and measured displacement belong to the
    later window collector. Consumers must not mutate these owned tensors.
    """

    observation_history: torch.Tensor
    action_history: torch.Tensor
    command: torch.Tensor
    history_valid_mask: torch.Tensor
    planner_future: torch.Tensor
    adapted_future: torch.Tensor
    residual_rad: torch.Tensor
    joint_names: tuple[str, ...]
    ctrl_dt: float
    position_scale: float
    velocity_scale: float

    def __post_init__(self) -> None:
        _validate_scales(self.ctrl_dt, self.position_scale, self.velocity_scale)
        names = tuple(self.joint_names)
        if len(names) != 29 or len(set(names)) != 29 or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("intervention requires 29 distinct source joint names")
        object.__setattr__(self, "joint_names", names)
        obs, future = self.observation_history, self.planner_future
        if obs.ndim != 3 or obs.shape[0] < 1 or obs.shape[1] < 1 or obs.shape[2] != 64:
            raise ValueError("intervention requires nonempty [batch, history, 64] observations")
        batch, history = obs.shape[:2]
        if future.ndim != 3 or future.shape[0] != batch or future.shape[1] < 1 or future.shape[2] != 64:
            raise ValueError("intervention requires a nonempty compact64 Planner future")
        expected = {
            "observation_history": obs.shape,
            "action_history": (batch, history, 29),
            "command": (batch, 7),
            "planner_future": future.shape,
            "adapted_future": future.shape,
            "residual_rad": (batch, 29),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape or value.dtype != future.dtype or value.device != future.device:
                raise ValueError(f"intervention {name} has incompatible shape, dtype or device")
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"intervention {name} must contain finite floating-point values")
        mask = self.history_valid_mask
        if mask.shape != (batch, history) or mask.dtype != torch.bool or mask.device != future.device:
            raise ValueError("intervention requires an aligned boolean history_valid_mask")
        if not bool(mask[:, -1].all()) or bool((mask[:, :-1] & ~mask[:, 1:]).any()):
            raise ValueError("intervention history mask must be a valid suffix")
        if bool((obs[~mask] != 0).any()) or bool((self.action_history[~mask] != 0).any()):
            raise ValueError("intervention history padding must be zero")
        for name in (*expected, "history_valid_mask"):
            object.__setattr__(self, name, getattr(self, name).detach().clone())

    @classmethod
    def from_query(
        cls, query: FADAFutureQuery, adapted_future: torch.Tensor, residual_rad: torch.Tensor,
    ) -> InterventionQueryRecord:
        scales = query.preprocessing["term_scale"]
        return cls(
            observation_history=query.observation_history,
            action_history=query.action_history,
            command=query.command,
            history_valid_mask=query.history_valid_mask,
            planner_future=query.planner_future,
            adapted_future=adapted_future,
            residual_rad=residual_rad,
            joint_names=query.profile.joint_names,
            ctrl_dt=query.profile.ctrl_dt,
            position_scale=float(scales["dof_pos"]),
            velocity_scale=float(scales["dof_vel"]),
        )


def _integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def owned_control(control: ControlInterval) -> ControlInterval:
    for name in ("sequence_id", "transition_id", "start_ns", "end_ns"):
        _integer(getattr(control, name), name)
    if control.end_ns <= control.start_ns:
        raise ValueError("control interval must advance physical time")
    action = np.array(control.action, dtype=np.float64, copy=True)
    command = np.array(control.command, dtype=np.float64, copy=True)
    if action.shape != (29,) or command.shape != (3,) or not np.isfinite(action).all() or not np.isfinite(command).all():
        raise ValueError("control requires finite env-input action (29,) and velocity command (3,)")
    return ControlInterval(
        control.sequence_id, control.transition_id, control.start_ns, control.end_ns, action, command,
    )


def _owned_pose(pose: TimedPelvisPose) -> TimedPelvisPose:
    for name in ("sequence_id", "tracking_epoch", "timestamp_ns"):
        _integer(getattr(pose, name), name)
    source = pose.source_timestamps_ns or (pose.timestamp_ns, pose.timestamp_ns)
    if len(source) != 2:
        raise ValueError("pose requires two physical source timestamps")
    for value in source:
        _integer(value, "source timestamp")
    if not source[0] <= pose.timestamp_ns <= source[1]:
        raise ValueError("pose source timestamps must bracket the represented time")
    return TimedPelvisPose(
        pose.sequence_id, pose.tracking_epoch, pose.timestamp_ns, pose_matrix(pose).copy(),
        True, (source[0], source[1]), pose.frame_id,
    )


@dataclass(frozen=True)
class PathFollowingWindowSample:
    """One complete execution window; delta is displacement, never velocity.

    Context is the first *executed* query. Endpoint acquisition provenance is
    preserved; all N control intervals were checked by the collector before
    construction. Consumers must not mutate the owned tensors or arrays.
    """

    window_id: int
    start_query: InterventionQueryRecord
    controls: tuple[ControlInterval, ...]
    start_pose: TimedPelvisPose
    end_pose: TimedPelvisPose
    reference: FixedStraightPath
    delta: np.ndarray = field(init=False)
    path_errors: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        _integer(self.window_id, "window_id")
        if self.start_query.observation_history.shape[0] != 1 or not self.controls:
            raise ValueError("a response sample requires one environment and a nonempty execution window")
        controls = tuple(owned_control(control) for control in self.controls)
        first, last = controls[0], controls[-1]
        dt_ns = round(self.start_query.ctrl_dt * 1e9)
        if dt_ns < 1:
            raise ValueError("ctrl_dt must represent at least one physical nanosecond")
        command = self.start_query.command[0, :3].cpu().numpy()
        for index, control in enumerate(controls):
            if (
                control.sequence_id != first.sequence_id
                or control.transition_id != first.transition_id + index
                or abs(control.end_ns - control.start_ns - dt_ns) > 1
                or (index and control.start_ns != controls[index - 1].end_ns)
                or not np.allclose(control.command, command, rtol=0, atol=1e-7)
            ):
                raise ValueError("response window must have contiguous fixed-period controls and one velocity command")
        start, end = _owned_pose(self.start_pose), _owned_pose(self.end_pose)
        if (
            start.sequence_id != first.sequence_id
            or start.timestamp_ns != first.start_ns or end.timestamp_ns != last.end_ns
            or pose_identity(start) != pose_identity(end)
            or self.reference.timestamp_ns > start.timestamp_ns
        ):
            raise ValueError("response poses must match the executed window and its task reference")
        errors = np.stack((self.reference.errors(start), self.reference.errors(end)))
        object.__setattr__(self, "controls", controls)
        object.__setattr__(self, "start_pose", start)
        object.__setattr__(self, "end_pose", end)
        object.__setattr__(self, "delta", measured_window_delta(start, end))
        object.__setattr__(self, "path_errors", errors)
