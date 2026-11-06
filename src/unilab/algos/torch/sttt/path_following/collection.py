"""Match committed constant-intervention windows to delayed measured poses."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import torch

from unilab.algos.torch.distill.fada.runtime_policy import FADAFutureQuery
from unilab.algos.torch.sttt.path_following.contract import (
    InterventionQueryRecord, PathFollowingWindowSample, _integer, owned_control,
)
from unilab.algos.torch.sttt.path_following.intervention import compose_joint_position_residual
from unilab.algos.torch.sttt.path_following.reference import FixedStraightPath, pose_identity
from unilab.algos.torch.sttt.path_following.window import ConstantResidualWindow
from unilab.algos.torch.sttt.pose_data import AlignedPoseInterval, ControlInterval, PoseRuntimeUpdate
from unilab.base.backend.sensors import SensorClock


@dataclass
class _Evidence:
    window_id: int
    query: InterventionQueryRecord
    controls: list[ControlInterval] = field(default_factory=list)
    aligned: dict[int, AlignedPoseInterval] = field(default_factory=dict)
    sealed: bool = False
    invalid_reason: str | None = None


class WindowCollector:
    """One task/episode, one env; no simulation loop or optimizer is owned here.

    Call begin_step -> transform (via playback) -> complete_step. Only the last
    call commits execution. Delayed labels never control window boundaries.
    A new task requires a new collector, not a path re-anchor after localization
    loss. Returned samples and dropped-window counts survive collector.close().
    """

    def __init__(self, window_steps: int, *, max_alignment_age_ns: int) -> None:
        self.window = ConstantResidualWindow(window_steps)
        self.max_alignment_age_ns = _integer(max_alignment_age_ns, "max_alignment_age_ns")
        if self.max_alignment_age_ns < 1:
            raise ValueError("max_alignment_age_ns must be positive")
        self.reference: FixedStraightPath | None = None
        self.completed_windows = 0
        self.dropped_windows: Counter[str] = Counter()
        self.halted = False
        self._closed = False
        self._task_clock: SensorClock | None = None
        self._before: SensorClock | None = None
        self._last_end_ns: int | None = None
        self._next_transition = 0
        self._last_aligned = -1
        self._queried = False
        self._query_command: np.ndarray | None = None
        self._binding: tuple | None = None
        self._active: _Evidence | None = None
        self._pending: dict[int, _Evidence] = {}
        self._owners: dict[int, _Evidence] = {}

    def submit_candidate(self, residual_rad: torch.Tensor) -> None:
        self._require_open()
        self.window.submit_candidate(residual_rad)

    def _require_open(self) -> None:
        if self._closed or self.halted:
            raise RuntimeError("collector is closed or invalidated; start a new task explicitly")

    def begin_step(self, clock: SensorClock) -> None:
        self._require_open()
        if self._before is not None:
            raise RuntimeError("previous prepared step was not committed")
        _integer(clock.sequence_id, "sequence_id")
        _integer(clock.timestamp_ns, "timestamp_ns")
        if self._task_clock is None:
            self._task_clock = clock
        elif clock.sequence_id != self._task_clock.sequence_id or clock.timestamp_ns != self._last_end_ns:
            self._stop("reset_or_time_gap")
            raise ValueError("execution crossed a reset or physical-time gap")
        self.window.lock()
        self._before = clock
        self._queried = False

    def transform(self, query: FADAFutureQuery) -> torch.Tensor:
        self._require_open()
        if self._before is None or self._queried:
            raise RuntimeError("each prepared control step requires exactly one future query")
        scales = query.preprocessing["term_scale"]
        binding = (
            query.profile.joint_names, query.profile.ctrl_dt, scales["dof_pos"], scales["dof_vel"],
            tuple(query.observation_history.shape), tuple(query.planner_future.shape),
        )
        if self._binding is None:
            self._binding = binding
        elif binding != self._binding:
            raise ValueError("source interpretation or context shape changed during collection")
        residual = self.window.lock()
        future = compose_joint_position_residual(
            query.planner_future, residual, ctrl_dt=query.profile.ctrl_dt,
            position_scale=scales["dof_pos"], velocity_scale=scales["dof_vel"],
        )
        if self.window.step_index == 0:
            if self._active is not None:
                raise RuntimeError("previous window was not sealed")
            record = InterventionQueryRecord.from_query(query, future, residual)
            if record.command.shape[0] != 1:
                raise ValueError("window capture requires exactly one environment")
            self._active = _Evidence(self.window.window_id, record)
            self._pending[self.window.window_id] = self._active
        self._query_command = query.command[0, :3].detach().cpu().numpy().copy()
        self._queried = True
        return future

    def complete_step(
        self, control: ControlInterval, update: PoseRuntimeUpdate, *, terminal: bool = False,
    ) -> tuple[PathFollowingWindowSample, ...]:
        self._require_open()
        if self._before is None or not self._queried or self._active is None or self._query_command is None:
            raise RuntimeError("execution commit requires its prepared query")
        control = owned_control(control)
        if (
            control.sequence_id != self._before.sequence_id
            or control.start_ns != self._before.timestamp_ns
            or control.transition_id != self._next_transition
            or not np.allclose(control.command, self._query_command, rtol=0, atol=1e-7)
            or abs(control.end_ns - control.start_ns - round(self._active.query.ctrl_dt * 1e9)) > 1
        ):
            self._stop("invalid_execution")
            raise ValueError("committed control does not match the prepared query and physical period")
        active = self._active
        active.controls.append(control)
        self._owners[control.transition_id] = active
        if not np.array_equal(control.command, active.controls[0].command):
            active.invalid_reason = "command_changed"
        self._last_end_ns = control.end_ns
        self._next_transition += 1
        self._before = None
        self._queried = False
        if self.window.commit_step():
            active.sealed = True
            self._active = None
        if terminal:
            active.invalid_reason = "terminal"
        if (
            update.invalidated or update.status == "lost" or update.sequence_id != control.sequence_id
            or (self.reference is not None and update.tracking_epoch != self.reference.tracking_epoch)
        ):
            self._stop("localization_changed")
            return ()
        for aligned in update.aligned:
            self._accept_alignment(aligned, update.tracking_epoch)
            if self.halted:
                return ()
        result = self._drain(control.end_ns)
        if terminal:
            self.halted = True
        return result

    def _accept_alignment(self, aligned: AlignedPoseInterval, epoch: int) -> None:
        control = aligned.control
        transition = control.transition_id
        if transition <= self._last_aligned or transition >= self._next_transition:
            raise ValueError("aligned intervals must be historical, unique and increasing")
        for missing in range(self._last_aligned + 1, transition):
            owner = self._owners.get(missing)
            if owner is not None:
                owner.invalid_reason = "missing_pose_interval"
        self._last_aligned = transition
        owner = self._owners.get(transition)
        if owner is None:
            return  # This window already expired; a late pose cannot revive it.
        expected = owner.controls[transition - owner.controls[0].transition_id]
        if (
            (control.sequence_id, control.start_ns, control.end_ns)
            != (expected.sequence_id, expected.start_ns, expected.end_ns)
            or not np.array_equal(control.action, expected.action)
            or not np.array_equal(control.command, expected.command)
        ):
            raise ValueError("pose alignment does not belong to the recorded executed control")
        start, end = aligned.start_pose, aligned.end_pose
        if (
            not start.valid or not end.valid or start.timestamp_ns != control.start_ns
            or end.timestamp_ns != control.end_ns or pose_identity(start) != pose_identity(end)
            or start.sequence_id != control.sequence_id or start.tracking_epoch != epoch
        ):
            self._stop("invalid_pose_interval")
            return
        if self.reference is None:
            if transition != 0 or self._task_clock is None or start.timestamp_ns != self._task_clock.timestamp_ns:
                self._stop("initial_pose_missing")
                return
            self.reference = FixedStraightPath.from_initial_pose(start)
        if pose_identity(start) != (self.reference.sequence_id, self.reference.tracking_epoch, self.reference.frame_id):
            self._stop("localization_changed")
            return
        owner.aligned[transition] = aligned

    def _remove(self, evidence: _Evidence, reason: str | None = None) -> None:
        if reason is not None:
            self.dropped_windows[reason] += 1
        self._pending.pop(evidence.window_id)
        for control in evidence.controls:
            self._owners.pop(control.transition_id, None)

    def _drain(self, now_ns: int) -> tuple[PathFollowingWindowSample, ...]:
        ready = []
        for evidence in tuple(self._pending.values()):
            if not evidence.sealed:
                continue
            if evidence.invalid_reason is not None:
                self._remove(evidence, evidence.invalid_reason)
                continue
            if len(evidence.aligned) == self.window.window_steps:
                assert self.reference is not None
                first, last = evidence.controls[0], evidence.controls[-1]
                ready.append(PathFollowingWindowSample(
                    evidence.window_id, evidence.query, tuple(evidence.controls),
                    evidence.aligned[first.transition_id].start_pose,
                    evidence.aligned[last.transition_id].end_pose, self.reference,
                ))
                self.completed_windows += 1
                self._remove(evidence)
            elif now_ns - evidence.controls[-1].end_ns > self.max_alignment_age_ns:
                self._remove(evidence, "alignment_timeout")
        return tuple(ready)

    def _stop(self, reason: str) -> None:
        for evidence in tuple(self._pending.values()):
            self._remove(evidence, reason)
        self._active = None
        self.halted = True

    def close(self) -> None:
        if self._closed:
            return
        for evidence in tuple(self._pending.values()):
            reason = evidence.invalid_reason or ("unresolved_pose" if evidence.sealed else "partial_window")
            self._remove(evidence, reason)
        self._active = None
        self._before = None
        self._closed = True
