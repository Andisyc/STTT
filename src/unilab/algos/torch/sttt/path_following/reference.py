"""A task-fixed straight line and horizontal heading-frame motion labels."""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import torch

from unilab.algos.torch.sttt.geometry import compute_se2_deltas, mat_to_quat, wrap_to_pi
from unilab.algos.torch.sttt.pose_data import TimedPelvisPose


def pose_identity(pose: TimedPelvisPose) -> tuple[int, int, str]:
    return pose.sequence_id, pose.tracking_epoch, pose.frame_id


def pose_matrix(pose: TimedPelvisPose) -> np.ndarray:
    if not pose.valid or pose.navigation_from_pelvis is None or not pose.frame_id:
        raise ValueError("path geometry requires a valid measured pelvis pose")
    matrix = np.asarray(pose.navigation_from_pelvis, dtype=np.float64)
    if (
        matrix.shape != (4, 4) or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], (0., 0., 0., 1.), atol=1e-9, rtol=0)
        or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-4, rtol=0)
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-4, rtol=0)
        or np.linalg.norm(matrix[:2, 0]) < 1e-8
    ):
        raise ValueError("path geometry requires a proper SE(3) pose with defined horizontal heading")
    return matrix


def _wrap(angle: float) -> float:
    return float(wrap_to_pi(torch.tensor(angle, dtype=torch.float64)))


@dataclass(frozen=True)
class FixedStraightPath:
    sequence_id: int
    tracking_epoch: int
    frame_id: str
    timestamp_ns: int
    origin_xy: tuple[float, float]
    heading_rad: float

    def __post_init__(self) -> None:
        for name in ("sequence_id", "tracking_epoch", "timestamp_ns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"fixed path {name} must be a non-negative integer")
        origin = tuple(float(value) for value in self.origin_xy)
        if len(origin) != 2 or not all(math.isfinite(value) for value in (*origin, self.heading_rad)):
            raise ValueError("fixed path requires a finite xy origin and heading")
        if not self.frame_id:
            raise ValueError("fixed path requires an explicit frame identity")
        object.__setattr__(self, "origin_xy", origin)

    @classmethod
    def from_initial_pose(cls, pose: TimedPelvisPose) -> FixedStraightPath:
        matrix = pose_matrix(pose)
        return cls(
            pose.sequence_id, pose.tracking_epoch, pose.frame_id, pose.timestamp_ns,
            (float(matrix[0, 3]), float(matrix[1, 3])),
            float(np.arctan2(matrix[1, 0], matrix[0, 0])),
        )

    def errors(self, pose: TimedPelvisPose) -> np.ndarray:
        if pose_identity(pose) != (self.sequence_id, self.tracking_epoch, self.frame_id):
            raise ValueError("pose cannot be compared with a path in another localization identity")
        matrix = pose_matrix(pose)
        dx, dy = matrix[:2, 3] - np.asarray(self.origin_xy)
        lateral = -math.sin(self.heading_rad) * dx + math.cos(self.heading_rad) * dy
        yaw = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
        return np.array((lateral, _wrap(yaw - self.heading_rad)), dtype=np.float64)


def measured_window_delta(start: TimedPelvisPose, end: TimedPelvisPose) -> np.ndarray:
    if pose_identity(start) != pose_identity(end) or end.timestamp_ns <= start.timestamp_ns:
        raise ValueError("motion endpoints must share one identity and increase in physical time")
    matrices = np.stack((pose_matrix(start), pose_matrix(end)))
    positions = torch.from_numpy(matrices[:, :3, 3].copy()).unsqueeze(0)
    quaternions = mat_to_quat(torch.from_numpy(matrices[:, :3, :3].copy())).unsqueeze(0)
    return compute_se2_deltas(positions, quaternions)[0, 0].numpy().copy()
