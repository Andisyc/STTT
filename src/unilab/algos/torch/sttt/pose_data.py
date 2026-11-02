"""Shared packets for measured pelvis poses and historical action intervals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from unilab.base.backend.sensors import SensorClock


@dataclass(frozen=True)
class TimedPelvisPose:
    """A metric Z-up pelvis pose, with interpolation provenance preserved.

    navigation_from_pelvis maps pelvis into a local Z-up frame whose origin
    and heading are fixed once per localization epoch. Invalid samples carry
    None. source_timestamps_ns is (t,t) for a measurement and (lo,hi) for an
    interpolated query; timestamp_ns is always the represented physical time.
    """

    sequence_id: int
    tracking_epoch: int
    timestamp_ns: int
    navigation_from_pelvis: np.ndarray | None
    valid: bool = True
    source_timestamps_ns: tuple[int, int] | None = None
    frame_id: str = "local_odom_z_up"


@dataclass(frozen=True)
class ControlInterval:
    """Executed env-input action and command held over [start_ns,end_ns]."""

    sequence_id: int
    transition_id: int
    start_ns: int
    end_ns: int
    action: np.ndarray  # (29,), actual env_input, not PD targets or proposed actions
    command: np.ndarray  # (3,), vx/vy/yaw_rate, recorded before executing the action


@dataclass(frozen=True)
class AlignedPoseInterval:
    control: ControlInterval
    start_pose: TimedPelvisPose
    end_pose: TimedPelvisPose


@dataclass(frozen=True)
class PoseRuntimeUpdate:
    aligned: tuple[AlignedPoseInterval, ...]
    status: Literal["waiting", "tracking", "lost"]
    sequence_id: int
    tracking_epoch: int
    dropped_intervals: int = 0
    invalidated: bool = False


class OnlinePoseRuntime(Protocol):
    """One controller-owned source; never starts a second simulation loop."""

    @property
    def clock(self) -> SensorClock:
        """Read physical time after stepping without opening a new interval."""
        ...

    def begin_step(self) -> SensorClock:
        """Prepare the source after environment reset and return physical time."""
        ...

    def complete_step(self, control: ControlInterval) -> PoseRuntimeUpdate:
        """Consume already-produced measurements and resolve historical intervals."""
        ...

    def reset(self) -> None:
        """Drop local evidence; environment reset may happen after this call."""
        ...

    def close(self) -> None:
        """Release owned tracker and stream; idempotent."""
        ...
