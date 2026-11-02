"""Autograd-vs-finite-difference Jacobian diagnostics for the spatial world model.

This diagnostic probes the *learned function* only. Agreement between
autograd and central finite differences proves numerical consistency of
the implemented computation graph; it is NOT evidence of a physical
Jacobian, strict positive realness (SPR), or closed-loop stability. A
zero-initialized or zero-valued head yields exactly zero gradients and is
a valid numerical result, not physical evidence of any kind.

The dense MLP flattens the whole action sequence, so a selected output
step may be sensitive to actions strictly AFTER that step. The
``max_abs_future_action_sensitivity`` field exposes this future-action
leakage of the current architecture; it does not declare the method
causal.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from typing import Any

import torch

from .contract import SpatialTrajectoryWindow
from .model import SpatialWorldModel

__all__ = ["diagnose_spatial_jacobian"]

_COORDINATE_NAMES = ("dx", "dy", "dyaw")


def _resolve_output_step(output_step: int, horizon: int) -> int:
    if not -horizon <= output_step < horizon:
        raise ValueError(
            f"output_step {output_step} is out of range for horizon {horizon}; "
            f"expected an index in [{-horizon}, {horizon - 1}]."
        )
    return output_step % horizon


def _resolve_indices(
    name: str,
    indices: Sequence[int] | None,
    default: Sequence[int],
    upper: int,
) -> list[int]:
    resolved = list(default) if indices is None else [int(i) for i in indices]
    if not resolved:
        raise ValueError(f"{name} must contain at least one index.")
    bad = [i for i in resolved if not 0 <= i < upper]
    if bad:
        raise ValueError(
            f"{name} {bad} out of range; expected indices in [0, {upper - 1}]."
        )
    return resolved


def diagnose_spatial_jacobian(
    model: SpatialWorldModel,
    window: SpatialTrajectoryWindow,
    *,
    action_steps: Sequence[int] | None = None,
    joint_indices: Sequence[int] | None = None,
    output_step: int = -1,
    output_coordinate: int = 2,
    epsilon: float = 1e-5,
) -> dict[str, Any]:
    """Compare autograd and central finite-difference sensitivities.

    Computes d(output[output_step, output_coordinate]) / d(actions) on a
    detached float64 CPU deepcopy of ``model`` and a copied unbatched
    ``window``. Neither the caller's model (parameters, buffers, training
    mode, normalization statistics, gradients) nor the caller's window
    tensors are mutated; no normalization statistics are fitted.

    Args:
        model: Spatial world model in any mode/device/dtype.
        window: Unbatched spatial trajectory window.
        action_steps: Action time steps to probe (default ``[0]``).
        joint_indices: Action joints to probe (default ``[0, 1, 2]``).
        output_step: Trajectory output step; negative wraps (default last).
        output_coordinate: 0=dx, 1=dy, 2=dyaw.
        epsilon: Central-difference half-width in action units.

    Returns:
        JSON-serializable dict with per-entry sensitivities and errors,
        ``max_abs_error``, ``all_zero_gradient``, and
        ``max_abs_future_action_sensitivity`` (None if no future actions).
    """
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError(f"epsilon must be a positive finite float, got {epsilon!r}.")
    if not 0 <= output_coordinate <= 2:
        raise ValueError(
            f"output_coordinate must be 0 (dx), 1 (dy), or 2 (dyaw), "
            f"got {output_coordinate}."
        )

    horizon = model.horizon
    act_dim = model.act_dim
    out_step = _resolve_output_step(int(output_step), horizon)
    steps = _resolve_indices("action_steps", action_steps, (0,), horizon)
    joints = _resolve_indices(
        "joint_indices", joint_indices, tuple(range(min(3, act_dim))), act_dim
    )

    if window.proprio_history.dim() != 2 or window.action_sequence.dim() != 2:
        raise ValueError(
            "window must be unbatched: expected proprio_history of rank 2 and "
            f"action_sequence of rank 2, got ranks {window.proprio_history.dim()} "
            f"and {window.action_sequence.dim()}."
        )
    if window.action_sequence.shape[0] != horizon or window.action_sequence.shape[1] != act_dim:
        raise ValueError(
            f"window action_sequence shape {tuple(window.action_sequence.shape)} "
            f"does not match model (horizon={horizon}, act_dim={act_dim})."
        )
    if window.proprio_history.shape[0] != model.hist_len or window.proprio_history.shape[1] != model.obs_dim:
        raise ValueError(
            f"window proprio_history shape {tuple(window.proprio_history.shape)} "
            f"does not match model (hist_len={model.hist_len}, obs_dim={model.obs_dim})."
        )

    # Detached float64 CPU copies; caller state is never touched.
    probe = copy.deepcopy(model).to(device="cpu", dtype=torch.float64)
    probe.eval()
    probe.requires_grad_(False)

    obs = window.proprio_history.detach().to(device="cpu", dtype=torch.float64).clone().unsqueeze(0)
    base_actions = window.action_sequence.detach().to(device="cpu", dtype=torch.float64).clone()

    def evaluate(actions: torch.Tensor) -> torch.Tensor:
        return probe(obs, actions.unsqueeze(0))[0, out_step, output_coordinate]

    # Autograd sensitivity on the copied input.
    grad_actions = base_actions.clone().requires_grad_(True)
    scalar = evaluate(grad_actions)
    autograd_full = torch.autograd.grad(scalar, grad_actions)[0]
    if not torch.isfinite(autograd_full).all():
        raise ValueError("Autograd produced non-finite sensitivities; aborting diagnostic.")
    scalar_value = float(scalar.detach())

    entries: list[dict[str, Any]] = []
    for step in steps:
        for joint in joints:
            plus = base_actions.clone()
            minus = base_actions.clone()
            plus[step, joint] += epsilon
            minus[step, joint] -= epsilon
            fd = float((evaluate(plus) - evaluate(minus)) / (2.0 * epsilon))
            ag = float(autograd_full[step, joint])
            if not math.isfinite(fd):
                raise ValueError(
                    f"Finite difference at action step {step}, joint index {joint} "
                    "produced a non-finite value; aborting diagnostic."
                )
            entries.append(
                {
                    "step": step,
                    "joint": joint,
                    "autograd": ag,
                    "finite_difference": fd,
                    "abs_error": abs(ag - fd),
                }
            )

    max_abs_error = max(entry["abs_error"] for entry in entries)
    all_zero_gradient = bool((autograd_full == 0).all().item())

    future_steps = [s for s in range(horizon) if s > out_step]
    if future_steps:
        future = autograd_full[out_step + 1 :, :]
        max_future: float | None = float(future.abs().max())
    else:
        max_future = None

    return {
        "selected": {
            "output_step": out_step,
            "output_coordinate": output_coordinate,
            "output_coordinate_name": _COORDINATE_NAMES[output_coordinate],
            "scalar_value": scalar_value,
        },
        "epsilon": float(epsilon),
        "entries": entries,
        "max_abs_error": max_abs_error,
        "all_zero_gradient": all_zero_gradient,
        "max_abs_future_action_sensitivity": max_future,
        "scope": {
            "numerical": (
                "Agreement between autograd and central finite differences "
                "proves numerical consistency of the learned function's "
                "computation graph only. It is NOT evidence of a physical "
                "Jacobian, strict positive realness (SPR), or closed-loop "
                "stability."
            ),
            "zero_gradient": (
                "A zero-initialized or zero-valued head yields exactly zero "
                "gradients; that is a valid numerical result, not physical "
                "evidence."
            ),
            "causality": (
                "The dense MLP consumes the full flattened action sequence; "
                "max_abs_future_action_sensitivity exposes sensitivity of the "
                "selected output step to actions strictly after it "
                "(future-action leakage). This does not declare the method "
                "causal."
            ),
        },
    }

