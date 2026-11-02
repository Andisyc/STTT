"""Measured-pose / executed-action time alignment for STTT Step 2.

Keeps one episode's worth of valid pelvis poses on their true physical
timestamps and answers "what was the pelvis pose at ``t``" only from real
measurements that bracket ``t``. Executed control intervals are queued and
released once both of their endpoints can be bracketed, so an interval is
never filled with a nearest-neighbour pose and never extrapolated.

Invariants
----------
- Cached poses are strictly increasing in ``timestamp_ns`` inside one
  ``sequence_id``. A new sequence invalidates every timestamp and clears the
  whole timeline.
- A new ``tracking_epoch`` or ``frame_id`` is a fresh pose origin: it can never
  be joined to an earlier sample, so the first pose of the new identity is
  recorded as non-joinable. Earlier samples stay available for intervals that
  were measured entirely inside the old identity, and no interpolation bridges
  the change.
- An invalid pose marks a measurement hole: the next valid pose is stitched to
  its predecessor as non-joinable, so no interpolation spans the hole.
- Linear xyz interpolation and shortest-arc wxyz SLERP never look further than
  ``max_gap_ns`` between the two bracketing measurements.
- Queries carry the queried physical time in ``timestamp_ns`` and keep the
  bracketing measurement times in ``source_timestamps_ns``.
- Every discard is counted in :attr:`PoseTimeline.dropped_intervals`; nothing
  is dropped silently and no fake data is produced.

Nothing here mutates the shared packets or the caller's arrays: control times
and values are stored exactly as enqueued, and every returned pose owns its
matrix. Pose matrices are held in float64 so measured and interpolated
endpoints have one dtype.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from unilab.algos.torch.sttt.pose_data import (
    AlignedPoseInterval,
    ControlInterval,
    TimedPelvisPose,
)

__all__ = ["PoseTimeline"]

ACTION_DIM = 29
COMMAND_DIM = 3

_ORTHONORMAL_ATOL = 1e-4
_SLERP_COINCIDENT_DOT = 1.0 - 1e-9
_COMPACT_HEAD_THRESHOLD = 64


class _Lookup(Enum):
    """Outcome of a pose query at one physical time."""

    OK = "ok"
    LOST = "lost"  # no future measurement can bracket this time
    WAIT = "wait"  # a later measurement may still bracket this time


@dataclass(frozen=True)
class _PoseSample:
    """One cached measurement plus the rotation-form copies used to interpolate."""

    pose: TimedPelvisPose
    position: np.ndarray
    quaternion: np.ndarray
    joins_previous: bool


def _ns(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer physical nanosecond")
    return int(value)


def _sample_time(sample: _PoseSample) -> int:
    return sample.pose.timestamp_ns


def _quat_from_mat(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a unit w-first quaternion."""
    trace = float(rotation[0, 0] + rotation[1, 1] + rotation[2, 2])
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = (
            0.25 * scale,
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
        )
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        quat = (
            (rotation[2, 1] - rotation[1, 2]) / scale,
            0.25 * scale,
            (rotation[0, 1] + rotation[1, 0]) / scale,
            (rotation[0, 2] + rotation[2, 0]) / scale,
        )
    elif rotation[1, 1] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        quat = (
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[0, 1] + rotation[1, 0]) / scale,
            0.25 * scale,
            (rotation[1, 2] + rotation[2, 1]) / scale,
        )
    else:
        scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        quat = (
            (rotation[1, 0] - rotation[0, 1]) / scale,
            (rotation[0, 2] + rotation[2, 0]) / scale,
            (rotation[1, 2] + rotation[2, 1]) / scale,
            0.25 * scale,
        )
    quaternion = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("valid pelvis pose has a degenerate rotation matrix")
    return quaternion / norm


def _mat_from_quat(quaternion: np.ndarray) -> np.ndarray:
    """Convert a unit w-first quaternion to a 3x3 rotation matrix."""
    w, x, y, z = (float(value) for value in quaternion)
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)),
            (2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)),
            (2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _matrix(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """Assemble a 4x4 SE(3) matrix from a position and a unit w-first quaternion."""
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _mat_from_quat(quaternion)
    matrix[:3, 3] = position
    return matrix


def _acquisition_span(sample: _PoseSample) -> tuple[int, int]:
    """Earliest and latest measurement time represented by one cached sample."""
    source = sample.pose.source_timestamps_ns
    if source is None:
        return sample.pose.timestamp_ns, sample.pose.timestamp_ns
    return min(source), max(source)


def _same_identity(start: TimedPelvisPose, end: TimedPelvisPose) -> bool:
    """True when both endpoint poses live in one navigation frame and epoch."""
    return (
        start.sequence_id == end.sequence_id
        and start.tracking_epoch == end.tracking_epoch
        and start.frame_id == end.frame_id
    )


def _slerp(before: np.ndarray, after: np.ndarray, fraction: float) -> np.ndarray:
    """Shortest-arc SLERP on w-first unit quaternions, degenerate inputs included."""
    dot = float(np.dot(before, after))
    target = after
    if dot < 0.0:  # q and -q are the same rotation; walk the short arc
        target = -after
        dot = -dot
    if dot >= _SLERP_COINCIDENT_DOT:
        blended = before + (target - before) * fraction
        norm = float(np.linalg.norm(blended))
        return before.copy() if norm <= 1e-12 else blended / norm
    angle = math.acos(min(dot, 1.0))
    sin_angle = math.sin(angle)
    weight_before = math.sin((1.0 - fraction) * angle) / sin_angle
    weight_after = math.sin(fraction * angle) / sin_angle
    blended = before * weight_before + target * weight_after
    norm = float(np.linalg.norm(blended))
    return before.copy() if norm <= 1e-12 else blended / norm


class PoseTimeline:
    """Single-episode timeline aligning measured pelvis poses with control intervals."""

    def __init__(
        self,
        *,
        max_gap_ns: int,
        max_age_ns: int,
        max_samples: int = 512,
        max_pending: int = 512,
    ) -> None:
        self.max_gap_ns = _ns(max_gap_ns, "max_gap_ns")
        if self.max_gap_ns <= 0:
            raise ValueError("max_gap_ns must be positive")
        self.max_age_ns = _ns(max_age_ns, "max_age_ns")
        if self.max_age_ns <= 0:
            raise ValueError("max_age_ns must be positive")
        self.max_samples = _ns(max_samples, "max_samples")
        if self.max_samples < 2:
            raise ValueError("max_samples must be at least 2 to bracket a query")
        self.max_pending = _ns(max_pending, "max_pending")
        if self.max_pending < 1:
            raise ValueError("max_pending must be positive")

        self._poses: list[_PoseSample] = []
        self._pose_head = 0
        self._pending: deque[ControlInterval] = deque()
        self._sequence_id: int | None = None
        self._tracking_epoch: int | None = None
        self._frame_id: str | None = None
        self._last_pose_ts: int | None = None
        self._last_enqueued_start: int | None = None
        self._last_enqueued_end: int | None = None
        self._barrier_pending = False
        self._dropped_intervals = 0

    # ------------------------------------------------------------------ state

    @property
    def dropped_intervals(self) -> int:
        """Cumulative intervals discarded for any reason since construction."""
        return self._dropped_intervals

    @property
    def pending_intervals(self) -> int:
        """Control intervals still queued, in enqueue order."""
        return len(self._pending)

    @property
    def cached_samples(self) -> int:
        """Measured poses currently usable for bracketing."""
        return len(self._poses) - self._pose_head

    def reset(self) -> None:
        """Forget the episode; queued intervals are counted as dropped."""
        self._clear_sequence()

    def _clear_sequence(self) -> None:
        self._dropped_intervals += len(self._pending)
        self._pending.clear()
        self._poses.clear()
        self._pose_head = 0
        self._last_pose_ts = None
        self._barrier_pending = False
        self._sequence_id = None
        self._tracking_epoch = None
        self._frame_id = None
        self._last_enqueued_start = None
        self._last_enqueued_end = None

    # ------------------------------------------------------------------ input

    def add_pose(self, pose: TimedPelvisPose) -> None:
        """Cache one measured pelvis pose; invalid poses mark a measurement hole."""
        timestamp_ns = _ns(pose.timestamp_ns, "pose timestamp_ns")
        matrix: np.ndarray | None = None
        if pose.valid:
            if not pose.frame_id:
                raise ValueError("valid pelvis pose requires a nonempty frame_id")
            if pose.navigation_from_pelvis is None:
                raise ValueError("valid pelvis pose requires navigation_from_pelvis")
            matrix = np.asarray(pose.navigation_from_pelvis, dtype=np.float64)
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise ValueError("valid pelvis pose requires a finite 4x4 navigation_from_pelvis")
            rotation = matrix[:3, :3]
            if (
                not np.allclose(rotation.T @ rotation, np.eye(3), atol=_ORTHONORMAL_ATOL)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=_ORTHONORMAL_ATOL)
                or not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9)
            ):
                raise ValueError("valid pelvis pose requires a proper SE(3) navigation_from_pelvis")

        if self._sequence_id is None:
            self._sequence_id = pose.sequence_id
        elif pose.sequence_id != self._sequence_id:
            # Timestamps are only comparable inside one episode: drop it all.
            self._clear_sequence()
            self._sequence_id = pose.sequence_id

        if self._last_pose_ts is not None and timestamp_ns <= self._last_pose_ts:
            raise ValueError(
                "pelvis pose timestamps must strictly increase inside "
                f"sequence {self._sequence_id}"
            )
        self._last_pose_ts = timestamp_ns

        if matrix is None:
            # A hole, not a pose: it carries no frame or epoch, so identity is
            # untouched. The next valid measurement cannot be joined to its
            # predecessor, so no interpolation spans the gap. Intervals that
            # still have real measurements at both endpoints stay resolvable.
            self._barrier_pending = True
            return

        if self._tracking_epoch is None:
            self._tracking_epoch = pose.tracking_epoch
            self._frame_id = pose.frame_id
        elif pose.tracking_epoch != self._tracking_epoch or pose.frame_id != self._frame_id:
            # A new localization epoch or navigation frame is a fresh origin. It
            # cannot be interpolated with earlier samples, and any queued interval
            # whose interior spans the switch has no single metric frame.
            self._tracking_epoch = pose.tracking_epoch
            self._frame_id = pose.frame_id
            self._barrier_pending = True

        sample = _PoseSample(
            pose=TimedPelvisPose(
                sequence_id=pose.sequence_id,
                tracking_epoch=pose.tracking_epoch,
                timestamp_ns=timestamp_ns,
                navigation_from_pelvis=matrix.copy(),
                valid=True,
                # A measurement stands at its own instant; an upstream lo/hi (an
                # already-interpolated sample) is preserved as supplied.
                source_timestamps_ns=pose.source_timestamps_ns
                if pose.source_timestamps_ns is not None
                else (timestamp_ns, timestamp_ns),
                frame_id=pose.frame_id,
            ),
            position=matrix[:3, 3].copy(),
            quaternion=_quat_from_mat(matrix[:3, :3]),
            joins_previous=not self._barrier_pending,
        )
        self._barrier_pending = False
        self._poses.append(sample)
        while len(self._poses) - self._pose_head > self.max_samples:
            self._pose_head += 1
        self._drop_unservable_front()
        self._compact()

    def enqueue(self, interval: ControlInterval) -> None:
        """Queue one executed env-input interval without rewriting its times."""
        start_ns = _ns(interval.start_ns, "interval start_ns")
        end_ns = _ns(interval.end_ns, "interval end_ns")
        if end_ns <= start_ns:
            raise ValueError("control interval end_ns must be greater than start_ns")
        action = np.asarray(interval.action)
        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise ValueError(f"control interval action must be {ACTION_DIM} finite values")
        command = np.asarray(interval.command)
        if command.shape != (COMMAND_DIM,) or not np.isfinite(command).all():
            raise ValueError(f"control interval command must be {COMMAND_DIM} finite values")

        if self._sequence_id is None:
            self._sequence_id = interval.sequence_id
        elif interval.sequence_id != self._sequence_id:
            self._clear_sequence()
            self._sequence_id = interval.sequence_id

        if self._last_enqueued_start is not None and start_ns < self._last_enqueued_start:
            raise ValueError("control intervals must be enqueued in non-decreasing start order")
        if self._last_enqueued_end is not None and end_ns < self._last_enqueued_end:
            raise ValueError("control intervals must be enqueued in non-decreasing end order")
        if len(self._pending) >= self.max_pending:
            raise RuntimeError(f"pending control intervals exceed max_pending={self.max_pending}")

        self._last_enqueued_start = start_ns
        self._last_enqueued_end = end_ns
        # A defensive copy keeps the queue immune to caller buffer reuse; the
        # recorded times and values are unchanged.
        self._pending.append(
            ControlInterval(
                sequence_id=interval.sequence_id,
                transition_id=interval.transition_id,
                start_ns=start_ns,
                end_ns=end_ns,
                action=action.copy(),
                command=command.copy(),
            )
        )

    # ------------------------------------------------------------------ query

    def sample_at(self, timestamp_ns: int) -> TimedPelvisPose | None:
        """Pose at ``timestamp_ns`` from bracketing measurements, else None."""
        timestamp_ns = _ns(timestamp_ns, "timestamp_ns")
        status, pose = self._lookup(timestamp_ns)
        return pose if status is _Lookup.OK and pose is not None else None

    def drain_ready(self, *, now_ns: int) -> tuple[AlignedPoseInterval, ...]:
        """Release every interval whose endpoints are bracketed by measurements."""
        now_ns = _ns(now_ns, "now_ns")
        ready: list[AlignedPoseInterval] = []
        while self._pending:
            control = self._pending[0]
            if control.end_ns > now_ns:
                break  # not yet historical; a later pose cannot exist for it
            if now_ns - control.end_ns > self.max_age_ns:
                self._drop_front()
                continue

            start_status, start_pose = self._lookup(control.start_ns)
            if start_status is _Lookup.WAIT:
                break
            if start_status is _Lookup.LOST or start_pose is None:
                self._drop_front()
                continue
            end_status, end_pose = self._lookup(control.end_ns)
            if end_status is _Lookup.WAIT:
                break
            if end_status is _Lookup.LOST or end_pose is None:
                self._drop_front()
                continue
            if not _same_identity(start_pose, end_pose):
                # The interval straddles a re-localization or frame change, so its
                # two endpoints are not in one metric frame.
                self._drop_front()
                continue
            if self._spans_barrier(control.start_ns, control.end_ns):
                # A measurement hole or identity switch sits inside the interval:
                # its interior is unknown even though both ends are measured.
                self._drop_front()
                continue

            ready.append(
                AlignedPoseInterval(control=control, start_pose=start_pose, end_pose=end_pose)
            )
            self._pending.popleft()

        self._prune_served()
        self._compact()
        return tuple(ready)

    # ------------------------------------------------------------------ helpers

    def _drop_front(self) -> None:
        self._pending.popleft()
        self._dropped_intervals += 1

    def _lookup(self, timestamp_ns: int) -> tuple[_Lookup, TimedPelvisPose | None]:
        poses = self._poses
        head = self._pose_head
        if head >= len(poses):
            return _Lookup.WAIT, None
        index = bisect_left(poses, timestamp_ns, head, len(poses), key=_sample_time)
        if index < len(poses) and poses[index].pose.timestamp_ns == timestamp_ns:
            return _Lookup.OK, self._measured(poses[index])
        if index <= head:
            return _Lookup.LOST, None  # oldest cached measurement is already later
        if index >= len(poses):
            return _Lookup.WAIT, None  # the right bracket has not been measured yet
        before = poses[index - 1]
        after = poses[index]
        if not after.joins_previous:
            return _Lookup.LOST, None  # a measurement hole sits inside the pair
        span_ns = after.pose.timestamp_ns - before.pose.timestamp_ns
        if span_ns > self.max_gap_ns:
            return _Lookup.LOST, None  # bridging this pair would fabricate motion
        fraction = (timestamp_ns - before.pose.timestamp_ns) / span_ns
        return _Lookup.OK, self._interpolate(before, after, timestamp_ns, fraction)

    def _measured(self, sample: _PoseSample) -> TimedPelvisPose:
        """Expose a cached measurement with its own provenance, isolated from the cache."""
        pose = sample.pose
        return TimedPelvisPose(
            sequence_id=pose.sequence_id,
            tracking_epoch=pose.tracking_epoch,
            timestamp_ns=pose.timestamp_ns,
            navigation_from_pelvis=pose.navigation_from_pelvis.copy(),
            valid=True,
            source_timestamps_ns=pose.source_timestamps_ns,
            frame_id=pose.frame_id,
        )

    def _interpolate(
        self,
        before: _PoseSample,
        after: _PoseSample,
        timestamp_ns: int,
        fraction: float,
    ) -> TimedPelvisPose:
        position = before.position + (after.position - before.position) * fraction
        quaternion = _slerp(before.quaternion, after.quaternion, fraction)
        origin = before.pose
        return TimedPelvisPose(
            sequence_id=origin.sequence_id,
            tracking_epoch=origin.tracking_epoch,
            timestamp_ns=timestamp_ns,
            navigation_from_pelvis=_matrix(position, quaternion),
            valid=True,
            # Report the outermost real acquisition times that produced both
            # brackets, so an interpolated bracket never hides its origin.
            source_timestamps_ns=(
                _acquisition_span(before)[0],
                _acquisition_span(after)[1],
            ),
            frame_id=origin.frame_id,
        )

    def _spans_barrier(self, start_ns: int, end_ns: int) -> bool:
        """True if an unbridgeable boundary lies strictly inside the interval.

        A sample whose ``joins_previous`` is false cannot be interpolated from
        its predecessor, so any interval whose interior covers that transition
        has an unknown interior even when both endpoints are exact measurements.
        """
        poses = self._poses
        head = self._pose_head
        # Barrier i is the unbridgeable stretch (t[i-1], t[i]). It touches the
        # interval iff t[i] > start_ns and t[i-1] < end_ns.
        lower = max(head + 1, bisect_right(poses, start_ns, head, len(poses), key=_sample_time))
        upper = min(bisect_left(poses, end_ns, head, len(poses), key=_sample_time) + 1, len(poses))
        for index in range(lower, upper):
            if not poses[index].joins_previous:
                return True
        return False

    def _drop_unservable_front(self) -> None:
        """Discard queued intervals whose front endpoint is already evicted."""
        if self._pose_head >= len(self._poses) or not self._pending:
            return
        oldest_ns = self._poses[self._pose_head].pose.timestamp_ns
        while self._pending and self._pending[0].start_ns < oldest_ns:
            self._drop_front()

    def _prune_served(self) -> None:
        """Keep the left bracket of the oldest queued interval; drop older poses."""
        if not self._pending or self._pose_head >= len(self._poses):
            return
        limit_ns = self._pending[0].start_ns
        head = self._pose_head
        index = bisect_right(self._poses, limit_ns, head, len(self._poses), key=_sample_time)
        keep_from = max(head, index - 1)
        if keep_from > head:
            self._pose_head = keep_from

    def _compact(self) -> None:
        head = self._pose_head
        if head >= _COMPACT_HEAD_THRESHOLD and head * 2 >= len(self._poses):
            del self._poses[:head]
            self._pose_head = 0