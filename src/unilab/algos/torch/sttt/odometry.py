"""Public-seam proprio capture for online spatial adaptation.

All reads go through public environment getters only; no backend or XML
traversal. Measured pose sources live in :mod:`unilab.algos.torch.sttt.odometry_runtime`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from unilab.algos.torch.sttt.contract import DEFAULT_OBS_DIM, PROPRIO_LAYOUT

__all__ = ["capture_spatial_proprio"]


def _env_vector_field(env: Any, getter_name: str, *, dim: int, field: str) -> np.ndarray:
    """Read a batched physical field through the public env owner, never the backend."""
    getter = getattr(env, getter_name, None)
    if not callable(getter):
        raise RuntimeError(
            f"spatial proprio capture requires env.{getter_name}() for the {field}; "
            "no task-boundary owner exists on this environment."
        )
    arr = np.asarray(getter(), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise ValueError(f"env.{getter_name}() must have shape (num_envs, {dim}), got {arr.shape}")
    return arr


def capture_spatial_proprio(env: Any) -> torch.Tensor:
    """Assemble the pinned 64-dim proprio vector (CPU) via public env APIs.

    Uses the same layout and scales as ``collection_runtime``: gyro, negated
    projected gravity (via the upvector), dof positions minus default angles,
    and dof velocities, all for the first environment row.
    """
    dtype = np.dtype(np.float32)
    gyro = _env_vector_field(env, "get_gyro", dim=3, field="base angular velocity")
    gravity = _env_vector_field(env, "get_upvector", dim=3, field="torso upvector")
    act_dim = int(env.action_space.shape[0])
    diff = _env_vector_field(env, "get_dof_pos", dim=act_dim, field="dof positions") - np.asarray(
        env.default_angles, dtype=dtype
    ).reshape(1, -1)
    dof_vel = _env_vector_field(env, "get_dof_vel", dim=act_dim, field="dof velocities")
    parts = (
        gyro * PROPRIO_LAYOUT["base_ang_vel"][2],
        gravity * PROPRIO_LAYOUT["projected_gravity_neg"][2],
        diff * PROPRIO_LAYOUT["dof_pos_minus_default"][2],
        dof_vel * PROPRIO_LAYOUT["dof_vel"][2],
    )
    proprio = np.concatenate(parts, axis=1, dtype=dtype)[0]
    if proprio.shape != (DEFAULT_OBS_DIM,):
        raise ValueError(f"proprio layout must total {DEFAULT_OBS_DIM} dims, got {proprio.shape}")
    return torch.as_tensor(np.ascontiguousarray(proprio), dtype=torch.float32).clone().cpu()