"""Persistence and orchestration for pelvis conversion and action-time alignment."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

from unilab.algos.torch.sttt.pose_conversion import PelvisPoseConverter
from unilab.algos.torch.sttt.pose_data import ControlInterval
from unilab.base.backend.sensors import RigKinematicsSample


CONTROL_HEADER = [
    "sequence_id", "transition_id", "start_ns", "end_ns",
    *[f"action_{i}" for i in range(29)], *[f"command_{i}" for i in range(3)],
]


def control_row(control: ControlInterval) -> list[Any]:
    return [control.sequence_id, control.transition_id, control.start_ns, control.end_ns,
            *control.action, *control.command]


def load_controls(path: Path) -> Iterator[ControlInterval]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}: recapture with action-time recording enabled")
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            yield ControlInterval(
                sequence_id=int(row["sequence_id"]), transition_id=int(row["transition_id"]),
                start_ns=int(row["start_ns"]), end_ns=int(row["end_ns"]),
                action=np.array([float(row[f"action_{i}"]) for i in range(29)]),
                command=np.array([float(row[f"command_{i}"]) for i in range(3)]),
            )


class PoseAlignmentOutput:
    """The converter and aligner never receive a ground-truth world pose."""

    def __init__(self, directory: Path, cfg: Any) -> None:
        from unilab.algos.torch.sttt.pose_timeline import PoseTimeline

        self.converter = PelvisPoseConverter()
        self.timeline = PoseTimeline(
            max_gap_ns=int(float(cfg.max_gap_s) * 1e9),
            max_age_ns=int(float(cfg.max_age_s) * 1e9),
            max_samples=int(cfg.max_samples), max_pending=int(cfg.max_pending),
        )
        self.enqueued = self.aligned = self.valid_poses = 0
        self._files = ExitStack()
        try:
            pose_file = self._files.enter_context((directory / "pelvis_odometry.csv").open("w", newline=""))
            aligned_file = self._files.enter_context((directory / "aligned_intervals.csv").open("w", newline=""))
            self._pose_writer = csv.writer(pose_file)
            self._aligned_writer = csv.writer(aligned_file)
            self._pose_writer.writerow([
                "sequence_id", "tracking_epoch", "timestamp_ns", "valid", "source_lo_ns", "source_hi_ns",
                *[f"T{i}{j}" for i in range(4) for j in range(4)],
            ])
            self._aligned_writer.writerow([
                *CONTROL_HEADER, "tracking_epoch", "start_source_lo_ns", "start_source_hi_ns",
                "end_source_lo_ns", "end_source_hi_ns",
                *[f"start_T{i}{j}" for i in range(4) for j in range(4)],
                *[f"end_T{i}{j}" for i in range(4) for j in range(4)],
            ])
        except BaseException:
            self._files.close()
            raise

    def enqueue(self, control: ControlInterval) -> None:
        self.timeline.enqueue(control)
        self.enqueued += 1

    def observe(self, estimate: Any, kinematics: RigKinematicsSample) -> None:
        pose = self.converter.convert(estimate, kinematics)
        self.timeline.add_pose(pose)
        self.valid_poses += int(pose.valid)
        matrix = pose.navigation_from_pelvis
        values = matrix.reshape(-1).tolist() if matrix is not None else [""] * 16
        source_times = pose.source_timestamps_ns
        if source_times is None:
            raise RuntimeError("Converted pose has no measurement-time provenance")
        self._pose_writer.writerow([
            pose.sequence_id, pose.tracking_epoch, pose.timestamp_ns, pose.valid,
            *source_times, *values,
        ])

    def drain(self, *, now_ns: int) -> None:
        for interval in self.timeline.drain_ready(now_ns=now_ns):
            start, end = interval.start_pose, interval.end_pose
            if (start.navigation_from_pelvis is None or end.navigation_from_pelvis is None
                    or start.source_timestamps_ns is None or end.source_timestamps_ns is None):
                raise RuntimeError("An aligned interval must carry two poses and their measurement brackets")
            self._aligned_writer.writerow([
                *control_row(interval.control), start.tracking_epoch,
                *start.source_timestamps_ns, *end.source_timestamps_ns,
                *start.navigation_from_pelvis.reshape(-1), *end.navigation_from_pelvis.reshape(-1),
            ])
            self.aligned += 1

    def summary(self) -> dict[str, int]:
        dropped = int(self.timeline.dropped_intervals)
        return {
            "valid_pelvis_poses": self.valid_poses,
            "control_intervals": self.enqueued,
            "aligned_intervals": self.aligned,
            "dropped_intervals": dropped,
            "unresolved_intervals": self.enqueued - self.aligned - dropped,
        }

    def close(self) -> None:
        self._files.close()
