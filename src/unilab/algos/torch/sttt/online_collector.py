"""Rolling collector for spatial online adaptation windows under delayed poses.

Every actual IDM query is saved with the executed env-input control interval
under ``(sequence_id, transition_id)``: the query snapshot, the pinned 64-dim
proprio vector, the realized action, the command and the full physical interval
``[start_ns, end_ns]``. Measured pelvis poses arrive later through
:meth:`SpatialOnlineCollector.accept_aligned` and are attached only to the
record carrying the same key, so a delayed measurement is never paired with a
newer action.

A window matures once ``history_length - 1`` earlier contiguous transitions and
its ``horizon`` query transitions are all retained, share one command, and every
measured transition carries both interval endpoints from one navigation frame
and localization epoch. The measured endpoint comes from the real start/end
poses; the reference endpoint is the world-anchored reference pose of the
window's own last step, expressed in the measured heading frame of the window
origin. Mature windows are emitted in original step order, each candidate end at
most once, and never stitched across a dropped transition or a command change.

The reference anchors once, at the first usable aligned control start (or after
relocalization), and advances exactly once per recorded command in original step
order — never in measurement-arrival order. A late anchor replays the already
saved commands. A command change never re-anchors the reference and never
discards the records already saved: windows that would span the change fail the
per-window command check, while windows entirely inside one command stay usable
when their measurements arrive late.

Retained evidence is bounded by
``history_length + horizon - 1 + ceil(max_alignment_age_s / ctrl_dt)``
transitions plus a small margin, with the reference command log covering the same
window. Every discard is explicit: an alignment whose query was already evicted
is expired evidence, a structurally broken window is skipped rather than
stitched, and a capacity eviction removes the oldest transition so no window can
survive a silently dropped transition.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from numbers import Integral

import numpy as np
import torch

from unilab.algos.torch.distill.fada.online_data import FADAInferenceSnapshot
from unilab.algos.torch.sttt.geometry import mat_to_quat, wrap_to_pi, yaw_from_quat
from unilab.algos.torch.sttt.online_data import OdometrySample, SpatialOnlineSample
from unilab.algos.torch.sttt.pose_data import (
    AlignedPoseInterval,
    ControlInterval,
    TimedPelvisPose,
)
from unilab.algos.torch.sttt.reference import ReferenceTrajectory

__all__ = ["SpatialOnlineCollector"]

PROPRIO_DIM = 64
ACTION_DIM = 29
COMMAND_DIM = 3

_CAPACITY_MARGIN = 2


class _Window(Enum):
    """Maturity state of one candidate window ending at a given control step."""

    READY = "ready"
    WAIT = "wait"  # missing evidence may still arrive
    DEAD = "dead"  # structurally broken; never matures


@dataclass(frozen=True)
class _Kinematics:
    """Fixed-frame pose fields of one measured or interpolated pelvis sample."""

    nav_key: tuple[int, int, str]
    timestamp_ns: int
    position: torch.Tensor
    quaternion: torch.Tensor
    yaw: float

    @property
    def frame_id(self) -> str:
        return self.nav_key[2]

    @property
    def tracking_epoch(self) -> int:
        return self.nav_key[1]


@dataclass
class _StepRecord:
    """One saved IDM query with its executed control interval and its evidence."""

    step: int
    key: tuple[int, int]
    control: ControlInterval
    snapshot: FADAInferenceSnapshot
    proprio: torch.Tensor
    start: _Kinematics | None = None
    end: _Kinematics | None = None
    reference_pose: torch.Tensor | None = None

    @property
    def served(self) -> bool:
        return self.start is not None and self.end is not None


def _kinematics(pose: TimedPelvisPose) -> _Kinematics | None:
    """Metric pose fields, or None when the sample cannot serve as evidence."""
    if not pose.valid:
        return None
    matrix = pose.navigation_from_pelvis
    if matrix is None:
        return None
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (4, 4) or not np.isfinite(values).all():
        return None
    rotation = torch.as_tensor(np.ascontiguousarray(values[:3, :3]), dtype=torch.float64)
    quaternion = mat_to_quat(rotation)
    if not torch.isfinite(quaternion).all():
        return None
    position = torch.as_tensor(np.ascontiguousarray(values[:3, 3]), dtype=torch.float64)
    return _Kinematics(
        nav_key=(int(pose.sequence_id), int(pose.tracking_epoch), str(pose.frame_id)),
        timestamp_ns=int(pose.timestamp_ns),
        position=position.clone(),
        quaternion=quaternion.clone(),
        yaw=float(yaw_from_quat(quaternion)),
    )


def _odometry_sample(pose: _Kinematics) -> OdometrySample:
    """Detached odometry packet used only to anchor the reference trajectory."""
    return OdometrySample(
        timestamp_s=pose.timestamp_ns / 1e9,
        position=pose.position,
        quaternion=pose.quaternion,
        frame_id=pose.frame_id,
        tracking_epoch=pose.tracking_epoch,
    )


class SpatialOnlineCollector:
    """Mature same-episode spatial windows from real, key-matched evidence."""

    def __init__(
        self,
        history_length: int,
        horizon: int,
        ctrl_dt: float,
        window_size: int,
        max_alignment_age_s: float,
    ) -> None:
        for name, value in (
            ("history_length", history_length),
            ("horizon", horizon),
            ("window_size", window_size),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(ctrl_dt) or ctrl_dt <= 0:
            raise ValueError("ctrl_dt must be positive and finite")
        if not math.isfinite(max_alignment_age_s) or max_alignment_age_s <= 0:
            raise ValueError("max_alignment_age_s must be positive and finite")
        self._history_length = int(history_length)
        self._horizon = int(horizon)
        self.window_size = int(window_size)
        self._ctrl_dt = float(ctrl_dt)
        self._max_alignment_age_s = float(max_alignment_age_s)
        self._capacity = (
            self._history_length
            + self._horizon
            + int(math.ceil(self._max_alignment_age_s / self._ctrl_dt))
            + _CAPACITY_MARGIN
        )
        self._reference = ReferenceTrajectory(self._ctrl_dt)
        self._records: dict[int, _StepRecord] = {}
        self._keys: dict[tuple[int, int], int] = {}
        self._steps: deque[int] = deque()
        self._commands: dict[int, np.ndarray] = {}
        self._samples: deque[SpatialOnlineSample] = deque(maxlen=self.window_size)
        self._episode_id = 0
        self._step = 0
        self._last_window_end: int | None = None
        self._reference_step: int | None = None
        self._anchor_step: int | None = None
        self._sequence_id: int | None = None
        self._frame_id: str | None = None
        self._tracking_epoch: int | None = None

    # ------------------------------------------------------------------ input

    @torch.no_grad()
    def record_step(
        self,
        snapshot: FADAInferenceSnapshot,
        proprio: torch.Tensor,
        control: ControlInterval,
    ) -> None:
        """Save one actual IDM query together with its executed control interval."""
        if not isinstance(snapshot, FADAInferenceSnapshot):
            raise TypeError("snapshot must be a FADAInferenceSnapshot")
        if not isinstance(control, ControlInterval):
            raise TypeError("control must be a ControlInterval")
        key = (int(control.sequence_id), int(control.transition_id))
        if key in self._keys:
            raise ValueError(f"control interval {key} was already recorded")
        if int(control.end_ns) <= int(control.start_ns):
            raise ValueError("control interval end_ns must be greater than start_ns")
        proprio_t = torch.as_tensor(proprio).detach().reshape(-1).to(dtype=torch.float32).cpu().clone()
        if proprio_t.shape != (PROPRIO_DIM,) or not torch.isfinite(proprio_t).all():
            raise ValueError(f"proprio must be a finite ({PROPRIO_DIM},) tensor")
        action = np.asarray(control.action)
        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise ValueError(f"control action must be {ACTION_DIM} finite env_input values")
        command = np.asarray(control.command)
        if command.shape != (COMMAND_DIM,) or not np.isfinite(command).all():
            raise ValueError(f"control command must be {COMMAND_DIM} finite values")

        if self._sequence_id is not None and key[0] != self._sequence_id:
            # A new sequence is a new time origin for the whole collector.
            self._clear_evidence()
            self._reference.unanchor()
            self._reference_step = None
            self._anchor_step = None
            self._frame_id = None
            self._tracking_epoch = None
        self._sequence_id = key[0]
        # A command change never re-anchors the reference and never discards the
        # records already saved: windows that would span the change are rejected
        # by the per-window command check, while windows entirely inside one
        # command stay usable when their measurements arrive late.

        record = _StepRecord(
            step=self._step,
            key=key,
            control=ControlInterval(
                sequence_id=key[0],
                transition_id=key[1],
                start_ns=int(control.start_ns),
                end_ns=int(control.end_ns),
                action=action.copy(),
                command=command.copy(),
            ),
            snapshot=snapshot,
            proprio=proprio_t,
        )
        self._records[self._step] = record
        self._keys[key] = self._step
        self._steps.append(self._step)
        self._commands[self._step] = record.control.command
        self._step += 1
        self._advance_reference()
        self._enforce_capacity()
        self._mature()

    @torch.no_grad()
    def accept_aligned(self, intervals: Sequence[AlignedPoseInterval]) -> None:
        """Attach measured interval endpoints to the exact records they belong to."""
        for interval in intervals:
            if not isinstance(interval, AlignedPoseInterval):
                raise TypeError("accept_aligned requires AlignedPoseInterval records")
            control = interval.control
            step = self._keys.get((int(control.sequence_id), int(control.transition_id)))
            if step is None:
                # The query was evicted before its measurement resolved, so this
                # alignment is expired evidence and can never train.
                continue
            # Alignments release in control order, so an older still-unserved step
            # can no longer be bracketed once a later interval exists.
            self._evict_unservable(step)
            record = self._records[step]
            start = _kinematics(interval.start_pose)
            end = _kinematics(interval.end_pose)
            if (
                not self._matches(record.control, control)
                or start is None
                or end is None
                or start.timestamp_ns != record.control.start_ns
                or end.timestamp_ns != record.control.end_ns
                or start.nav_key != end.nav_key
                or start.nav_key[0] != record.control.sequence_id
            ):
                # A different executed control, an unmeasured endpoint, or two
                # navigation frames: no evidence, and never a bridged sample.
                self._forget_step(step)
                continue
            if self._frame_id is not None and (
                start.frame_id != self._frame_id
                or start.tracking_epoch != self._tracking_epoch
            ):
                # Relocalization: earlier evidence shares no metric frame, so the retained
                # history and its mature samples are dropped and this measured start
                # re-anchors the reference. Retained queries at and after this step
                # stay usable and their stale reference poses are replayed in order.
                for stale in [older for older in self._steps if older < step]:
                    self._forget_step(stale)
                self._samples.clear()
                self._last_window_end = None
                for stale in [older for older in self._commands if older < step]:
                    del self._commands[stale]
                for record_ in self._records.values():
                    record_.reference_pose = None
                self._reference.unanchor()
                self._reference_step = None
                self._anchor_step = None
            self._frame_id = start.frame_id
            self._tracking_epoch = start.tracking_epoch
            record.start = start
            record.end = end
            if not self._reference.anchored:
                # First usable aligned control start of this epoch: anchor here.
                self._reference.reset(_odometry_sample(start))
                self._reference_step = step - 1
                self._anchor_step = step
            self._advance_reference()
            self._mature()

    # --------------------------------------------------------------- evidence

    @staticmethod
    def _matches(stored: ControlInterval, incoming: ControlInterval) -> bool:
        """True when the aligned interval is the very control record we saved."""
        return (
            stored.sequence_id == int(incoming.sequence_id)
            and stored.transition_id == int(incoming.transition_id)
            and stored.start_ns == int(incoming.start_ns)
            and stored.end_ns == int(incoming.end_ns)
            and np.array_equal(stored.action, incoming.action)
            and np.array_equal(stored.command, incoming.command)
        )

    def _forget_step(self, step: int) -> None:
        """Remove one retained transition and its key, explicitly."""
        record = self._records.pop(step, None)
        if record is not None:
            self._keys.pop(record.key, None)
        self._steps.remove(step)

    def _evict_unservable(self, step: int) -> None:
        """Drop older transitions a later released interval proves unmeasurable.

        Alignments release in control order, so once an interval at ``step`` has
        been released, every earlier interval with no evidence was dropped by the
        source and can never be bracketed.
        """
        for stale in [older for older in self._steps if older < step]:
            if not self._records[stale].served:
                self._forget_step(stale)

    def _enforce_capacity(self) -> None:
        """Keep retained training evidence bounded; the oldest transition goes first."""
        while len(self._steps) > self._capacity:
            self._forget_step(self._steps[0])
        self._prune_commands()

    def _prune_commands(self) -> None:
        """Keep the command log exactly over the retained record window.

        Integration starts at the anchor and must replay every recorded command
        forward, and any still-retained transition may become that anchor after a
        relocalization, so the log covers the same bounded window as the training
        evidence and nothing outside it survives.
        """
        keep_from = self._step - self._capacity
        for stale in [step for step in self._commands if step < keep_from]:
            del self._commands[stale]

    def _clear_evidence(self) -> None:
        """Drop retained transitions and mature samples; keep the episode counter."""
        self._records.clear()
        self._keys.clear()
        self._steps.clear()
        self._commands.clear()
        self._samples.clear()
        self._last_window_end = None

    def invalidate_tracking(self) -> None:
        """Drop spatial evidence and the reference anchor; the episode counter stays."""
        self._clear_evidence()
        self._reference.unanchor()
        self._reference_step = None
        self._anchor_step = None
        self._frame_id = None
        self._tracking_epoch = None

    # -------------------------------------------------------------- reference

    def _advance_reference(self) -> None:
        """Integrate each recorded command exactly once, in original step order."""
        if not self._reference.anchored or self._reference_step is None:
            return
        newest = self._step - 1
        while self._reference_step < newest:
            following = self._reference_step + 1
            command = self._commands.get(following)
            if command is None:
                # The command log must cover every step after the anchor; a gap
                # would require inventing motion, so it is a contract violation.
                raise RuntimeError(
                    f"reference command for recorded step {following} is missing"
                )
            pose = self._reference.advance(torch.as_tensor(command, dtype=torch.float64))
            self._reference_step = following
            record = self._records.get(following)
            if record is not None:
                record.reference_pose = pose

    # --------------------------------------------------------------- maturity

    @torch.no_grad()
    def _mature(self) -> None:
        """Emit every mature window in original step order, each end at most once."""
        while self._steps:
            earliest = self._steps[0] + self._history_length + self._horizon - 2
            if self._last_window_end is None:
                end = earliest
            else:
                end = max(self._last_window_end + 1, earliest)
            status = self._window_status(end)
            if status is _Window.WAIT:
                return
            self._last_window_end = end
            if status is _Window.DEAD:
                continue
            self._emit(end)

    def _window_status(self, end: int) -> _Window:
        """Classify the candidate window that ends at control step ``end``."""
        if not self._steps or end > self._steps[-1]:
            return _Window.WAIT
        first = end - self._horizon + 1
        prefix = first - self._history_length + 1
        chain: list[_StepRecord] = []
        for step in range(prefix, end + 1):
            record = self._records.get(step)
            if record is None:
                return _Window.DEAD
            chain.append(record)
        command = chain[0].control.command
        for earlier, record in zip(chain, chain[1:]):
            if not np.array_equal(record.control.command, command):
                # A window may never span a command change, history included.
                return _Window.DEAD
            if (
                record.control.sequence_id != earlier.control.sequence_id
                or record.control.transition_id != earlier.control.transition_id + 1
                or record.control.start_ns != earlier.control.end_ns
            ):
                return _Window.DEAD
        measured = chain[-self._horizon:]
        for record in measured:
            if record.start is None or record.end is None:
                return _Window.WAIT
            if record.start.nav_key != record.end.nav_key:
                return _Window.DEAD
        for earlier, record in zip(measured, measured[1:]):
            assert earlier.end is not None and record.start is not None
            if (
                earlier.end.nav_key != record.start.nav_key
                or earlier.end.timestamp_ns != record.start.timestamp_ns
            ):
                return _Window.DEAD
        if self._anchor_step is not None and end < self._anchor_step:
            # The reference anchored only at a later measured start, so this
            # window's endpoint predates the anchor; it can never be expressed.
            return _Window.DEAD
        if measured[-1].reference_pose is None:
            return _Window.WAIT
        if self._anchor_step is None or self._anchor_step > first:
            # The reference starts after this window's measured origin, so its
            # endpoint could only be produced by inventing history.
            return _Window.DEAD
        return _Window.READY

    def _emit(self, end: int) -> None:
        """Build one mature sample from the retained real evidence."""
        first = end - self._horizon + 1
        prefix = first - self._history_length + 1
        history = torch.stack(
            [self._records[step].proprio for step in range(prefix, first + 1)]
        ).unsqueeze(0)
        window = [self._records[step] for step in range(first, end + 1)]
        origin = window[0].start
        endpoint = window[-1].end
        reference_pose = window[-1].reference_pose
        assert origin is not None and endpoint is not None and reference_pose is not None
        origin_x = float(origin.position[0])
        origin_y = float(origin.position[1])
        origin_yaw = origin.yaw
        cos_o = math.cos(origin_yaw)
        sin_o = math.sin(origin_yaw)

        def _into_origin(dx: float, dy: float, dyaw: float) -> torch.Tensor:
            return torch.tensor(
                [
                    cos_o * dx + sin_o * dy,
                    -sin_o * dx + cos_o * dy,
                    float(wrap_to_pi(torch.tensor(dyaw, dtype=torch.float64))),
                ],
                dtype=torch.float32,
            )

        measured = _into_origin(
            float(endpoint.position[0]) - origin_x,
            float(endpoint.position[1]) - origin_y,
            endpoint.yaw - origin_yaw,
        )
        reference = _into_origin(
            float(reference_pose[0]) - origin_x,
            float(reference_pose[1]) - origin_y,
            float(reference_pose[2]) - origin_yaw,
        )
        actions = torch.stack(
            [
                torch.as_tensor(record.control.action, dtype=torch.float32).reshape(-1)
                for record in window
            ]
        ).unsqueeze(0)
        self._samples.append(
            SpatialOnlineSample(
                proprio_history=history,
                snapshots=tuple(record.snapshot for record in window),
                actual_actions=actions,
                measured_endpoint=measured.reshape(1, 3),
                reference_endpoint=reference.reshape(1, 3),
                episode_id=self._episode_id,
                origin_step=first,
            )
        )

    # ------------------------------------------------------------------- read

    @property
    def ready(self) -> bool:
        return len(self._samples) == self.window_size

    def get_window(self) -> tuple[SpatialOnlineSample, ...]:
        """Read mature samples oldest first, without consuming them."""
        return tuple(self._samples)

    def clear_window(self) -> None:
        """Consume mature samples.

        Windows are deleted here, never re-emitted: each candidate end matures at
        most once, so clearing cannot resurrect a window from retained evidence.
        """
        self._samples.clear()

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        """Drop all episode-local data and unanchor the reference; start a new episode."""
        self._clear_evidence()
        self._reference.unanchor()
        self._reference_step = None
        self._anchor_step = None
        self._sequence_id = None
        self._frame_id = None
        self._tracking_epoch = None
        self._step = 0
        self._episode_id += 1