"""Spatial Test-Time Training (STTT) via Differentiable World Model."""

from __future__ import annotations

from .contract import (
    DEFAULT_OBS_DIM,
    PROPRIO_LAYOUT,
    SPATIAL_ACTION_REPRESENTATION,
    SPATIAL_DATA_FORMAT_VERSION,
    SpatialBatch,
    SpatialTrajectoryWindow,
    SpatialTransition,
    validate_trajectory_window,
)
from .geometry import (
    compute_se2_deltas,
    mat_to_quat,
    matmul,
    matvec,
    quat_to_mat,
    so3_exp,
    so3_log,
    wrap_to_pi,
    yaw_from_quat,
    yaw_mat,
)
from .model import RunningNorm, SpatialWorldModel, mlp

__all__ = [
    "DEFAULT_OBS_DIM",
    "PROPRIO_LAYOUT",
    "SPATIAL_ACTION_REPRESENTATION",
    "SPATIAL_DATA_FORMAT_VERSION",
    "RunningNorm",
    "SpatialBatch",
    "SpatialTrajectoryWindow",
    "SpatialTransition",
    "SpatialWorldModel",
    "compute_se2_deltas",
    "mat_to_quat",
    "matmul",
    "matvec",
    "mlp",
    "quat_to_mat",
    "so3_exp",
    "so3_log",
    "validate_trajectory_window",
    "wrap_to_pi",
    "yaw_from_quat",
    "yaw_mat",
]
