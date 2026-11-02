"""Data contracts for Spatial Test-Time Training (STTT).

Defines the typed data structures exchanged between the trajectory
collector, the differentiable :class:`SpatialWorldModel`, and the
test-time training loop, together with strict validation helpers that
fail fast on shape mismatches, NaN/Inf corruption, or out-of-range
yaw deltas.

Conventions
-----------
- SE(2) pose deltas are expressed in the robot's local horizontal frame
  at ``t = 0`` as ``[Δx, Δy, Δyaw]`` where
  ``[Δx, Δy] = R(yaw_0)^T @ (pos_t - pos_0)`` and
  ``Δyaw = wrap_to_pi(yaw_t - yaw_0) ∈ [-π, π]``.
- ``commands`` follow the UniLab locomotion convention
  ``[vx, vy, yaw_rate]``.
- Default robot: Unitree G1 with 29 actuated DoF.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

__all__ = [
    "PROPRIO_LAYOUT",
    "DEFAULT_OBS_DIM",
    "SPATIAL_DATA_FORMAT_VERSION",
    "SPATIAL_ACTION_REPRESENTATION",
    "SpatialTransition",
    "SpatialTrajectoryWindow",
    "SpatialBatch",
    "validate_trajectory_window",
]

#: Pinned compact proprioceptive layout (order + scales), totalling 64 dims.
#: Matches the G1 walk-profile scaling convention (gyro ×0.25, dof_vel ×0.05,
#: negated projected gravity) minus actor-coupled terms (last_actions,
#: commands, gait phase) that must never enter the world model's state input.
#: Order and scales match ``walk_observation.assemble_walk_observation``.
PROPRIO_LAYOUT: dict[str, tuple[int, int, float]] = {
    # name: (start, stop, scale)
    "base_ang_vel": (0, 3, 0.25),
    "projected_gravity_neg": (3, 6, -1.0),
    "dof_pos_minus_default": (6, 35, 1.0),
    "dof_vel": (35, 64, 0.05),
}

#: Default proprioceptive observation dimension (sum of PROPRIO_LAYOUT).
DEFAULT_OBS_DIM: int = 64

SPATIAL_DATA_FORMAT_VERSION = 1
SPATIAL_ACTION_REPRESENTATION = "env_input"


@dataclass(frozen=True)
class SpatialTransition:
    """One detached control transition, before any episode reset.

    ``action`` is ``env.state.info["current_actions"]`` in target joint
    order, not an executed PD target. ``proprio`` and the reference pose
    belong to ``episode_step``; the next pose is after that action.
    Terminal transitions are boundary markers, never training labels.
    """

    rollout_id: str
    episode_id: int
    episode_step: int
    proprio: torch.Tensor
    action: torch.Tensor
    commands: torch.Tensor
    position: torch.Tensor
    quaternion: torch.Tensor
    next_position: torch.Tensor
    next_quaternion: torch.Tensor
    terminated: bool = False
    truncated: bool = False
    condition: dict[str, object] | None = None

#: Yaw deltas are wrapped into the principal range ``[-π, π]``.
_YAW_LIMIT = math.pi


@dataclass(frozen=True)
class SpatialTrajectoryWindow:
    """A single spatial training window for the world model.

    Tensors may be unbatched (leading dims as documented) or batched with
    a leading batch dimension. All tensors in a window must consistently
    be either batched or unbatched.

    Attributes:
        proprio_history: Proprioceptive observation history with shape
            ``(H_hist, obs_dim)`` or ``(batch, H_hist, obs_dim)``.
        action_sequence: Executed (or planned) action sequence with shape
            ``(horizon, act_dim)`` or ``(batch, horizon, act_dim)``.
        delta_trajectory: SE(2) pose deltas ``[Δx, Δy, Δyaw]`` in the
            robot's local horizontal frame at ``t = 0``, with shape
            ``(horizon, 3)`` or ``(batch, horizon, 3)``.
        commands: Locomotion commands ``[vx, vy, yaw_rate]`` with shape
            ``(3,)`` or ``(batch, 3)``.
        step_indices: Environment step indices covering the window; a
            ``torch.Tensor`` or a ``tuple[int, ...]``. When batched as a
            tensor, shape is ``(batch, num_indices)``.
    """

    proprio_history: torch.Tensor
    action_sequence: torch.Tensor
    delta_trajectory: torch.Tensor
    commands: torch.Tensor
    step_indices: torch.Tensor | tuple[int, ...]


def _check_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all().item():
        raise ValueError(
            f"{name} contains NaN or Inf values; refusing to process "
            f"corrupted trajectory window."
        )


def _check_shape(
    name: str,
    tensor: torch.Tensor,
    unbatched_shape: tuple[int, ...],
    rank: int,
) -> bool:
    """Check shape against unbatched/batched variants.

    Returns True if the tensor is batched, False if unbatched.
    """
    if tensor.dim() == rank:
        expected = unbatched_shape
        batched = False
    elif tensor.dim() == rank + 1:
        expected = (tensor.shape[0],) + unbatched_shape
        batched = True
    else:
        raise ValueError(
            f"{name} must have rank {rank} (unbatched) or {rank + 1} "
            f"(batched), got rank {tensor.dim()} with shape "
            f"{tuple(tensor.shape)}."
        )
    if tuple(tensor.shape) != expected:
        raise ValueError(
            f"{name} has shape {tuple(tensor.shape)}, expected "
            f"{expected} ({'batched' if batched else 'unbatched'})."
        )
    return batched


def validate_trajectory_window(
    window: SpatialTrajectoryWindow,
    expected_obs_dim: int = DEFAULT_OBS_DIM,
    expected_act_dim: int = 29,
    expected_horizon: int = 15,
    expected_hist: int = 15,
) -> None:
    """Validate a :class:`SpatialTrajectoryWindow` in place.

    Args:
        window: The window to validate.
        expected_obs_dim: Expected proprioceptive observation dimension
            (64 for the pinned compact G1 proprio layout, see
            :data:`PROPRIO_LAYOUT`).
        expected_act_dim: Expected action dimension (29 for Unitree G1).
        expected_horizon: Expected future prediction horizon ``H``.
        expected_hist: Expected history length ``H_hist``.

    Raises:
        TypeError: If a field has an unexpected type.
        ValueError: If shapes mismatch, tensors contain NaN/Inf, or
            ``delta_trajectory[..., 2]`` lies outside ``[-π, π]``.
    """
    if not isinstance(window, SpatialTrajectoryWindow):
        raise TypeError(
            f"window must be a SpatialTrajectoryWindow, got {type(window).__name__}."
        )

    tensor_fields = (
        "proprio_history",
        "action_sequence",
        "delta_trajectory",
        "commands",
    )
    for name in tensor_fields:
        if not isinstance(getattr(window, name), torch.Tensor):
            raise TypeError(
                f"{name} must be a torch.Tensor, got "
                f"{type(getattr(window, name)).__name__}."
            )

    batched_flags = {
        "proprio_history": _check_shape(
            "proprio_history",
            window.proprio_history,
            (expected_hist, expected_obs_dim),
            rank=2,
        ),
        "action_sequence": _check_shape(
            "action_sequence",
            window.action_sequence,
            (expected_horizon, expected_act_dim),
            rank=2,
        ),
        "delta_trajectory": _check_shape(
            "delta_trajectory",
            window.delta_trajectory,
            (expected_horizon, 3),
            rank=2,
        ),
        "commands": _check_shape(
            "commands",
            window.commands,
            (3,),
            rank=1,
        ),
    }

    if len(set(batched_flags.values())) != 1:
        raise ValueError(
            f"Inconsistent batching across window tensors: {batched_flags}. "
            f"All tensors must be either batched or unbatched."
        )

    for name in tensor_fields:
        _check_finite(name, getattr(window, name))

    yaw = window.delta_trajectory[..., 2]
    yaw_max = yaw.abs().max().item()
    if yaw_max > _YAW_LIMIT + 1e-6:
        raise ValueError(
            f"delta_trajectory[..., 2] (Δyaw) must lie within [-π, π]; "
            f"found max |Δyaw| = {yaw_max:.6f}. Wrap yaw deltas with "
            f"wrap_to_pi before constructing the window."
        )

    if isinstance(window.step_indices, torch.Tensor):
        if window.step_indices.dim() not in (1, 2):
            raise ValueError(
                f"step_indices tensor must have rank 1 (unbatched) or 2 "
                f"(batched), got rank {window.step_indices.dim()} with "
                f"shape {tuple(window.step_indices.shape)}."
            )
    elif not isinstance(window.step_indices, tuple):
        raise TypeError(
            f"step_indices must be a torch.Tensor or tuple[int, ...], got "
            f"{type(window.step_indices).__name__}."
        )


@dataclass
class SpatialBatch:
    """Batched container of :class:`SpatialTrajectoryWindow` tensors.

    All tensors carry a leading batch dimension:
    ``proprio_history`` ``(B, H_hist, obs_dim)``,
    ``action_sequence`` ``(B, horizon, act_dim)``,
    ``delta_trajectory`` ``(B, horizon, 3)``,
    ``commands`` ``(B, 3)``, and ``step_indices`` ``(B, num_indices)``
    when present as a tensor.
    """

    proprio_history: torch.Tensor
    action_sequence: torch.Tensor
    delta_trajectory: torch.Tensor
    commands: torch.Tensor
    step_indices: torch.Tensor | tuple[int, ...] | None = field(default=None)

    @classmethod
    def collate(
        cls,
        windows: list[SpatialTrajectoryWindow],
        device: torch.device | str | None = None,
    ) -> "SpatialBatch":
        """Stack a list of (unbatched or singly-batched) windows.

        Args:
            windows: Non-empty list of windows with identical trailing
                shapes. Unbatched windows are stacked along a new batch
                dimension; batched windows are concatenated along dim 0.
            device: Target device for the stacked tensors. ``None`` keeps
                the device of the first window's tensors.

        Returns:
            A :class:`SpatialBatch` with all tensors on ``device``.

        Raises:
            ValueError: If ``windows`` is empty or trailing shapes differ.
        """
        if not windows:
            raise ValueError("collate requires a non-empty list of windows.")

        def _stack(name: str, unbatched_rank: int) -> torch.Tensor:
            tensors = [getattr(w, name) for w in windows]
            ref = tensors[0]
            for i, t in enumerate(tensors):
                if t.dim() != ref.dim() or tuple(t.shape[1:]) != tuple(ref.shape[1:]):
                    raise ValueError(
                        f"collate: {name} shape mismatch at window {i}: "
                        f"{tuple(t.shape)} vs {tuple(ref.shape)}."
                    )
            batched = [t.unsqueeze(0) if t.dim() == unbatched_rank else t for t in tensors]
            stacked = torch.cat(batched, dim=0)
            if device is not None:
                stacked = stacked.to(device)
            return stacked

        proprio_history = _stack("proprio_history", unbatched_rank=2)
        action_sequence = _stack("action_sequence", unbatched_rank=2)
        delta_trajectory = _stack("delta_trajectory", unbatched_rank=2)
        commands = _stack("commands", unbatched_rank=1)

        step_indices: torch.Tensor | tuple[int, ...] | None
        first_indices = windows[0].step_indices
        if isinstance(first_indices, torch.Tensor):
            idx_tensors = []
            for i, w in enumerate(windows):
                if not isinstance(w.step_indices, torch.Tensor):
                    raise ValueError(
                        f"collate: step_indices type mismatch at window {i}; "
                        f"expected torch.Tensor to match window 0."
                    )
                idx = w.step_indices
                if idx.dim() == 1:
                    idx = idx.unsqueeze(0)
                idx_tensors.append(idx)
            step_indices = torch.cat(idx_tensors, dim=0)
            if device is not None:
                step_indices = step_indices.to(device)
        else:
            step_indices = tuple(
                idx for w in windows for idx in (
                    w.step_indices if isinstance(w.step_indices, tuple) else ()
                )
            )

        return cls(
            proprio_history=proprio_history,
            action_sequence=action_sequence,
            delta_trajectory=delta_trajectory,
            commands=commands,
            step_indices=step_indices,
        )

    @property
    def batch_size(self) -> int:
        return self.proprio_history.shape[0]
