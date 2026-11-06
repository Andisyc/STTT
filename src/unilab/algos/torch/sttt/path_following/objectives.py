"""Measured errors plus predicted heading-frame displacement; no steering heuristic."""

from __future__ import annotations

import torch

from unilab.algos.torch.sttt.geometry import wrap_to_pi


def predict_path_errors(errors: torch.Tensor, predicted_delta: torch.Tensor) -> torch.Tensor:
    if errors.ndim != 2 or errors.shape[0] < 1 or errors.shape[1] != 2 or predicted_delta.shape != (errors.shape[0], 3):
        raise ValueError("path-error composition requires errors [batch, 2] and increments [batch, 3]")
    if not errors.is_floating_point() or errors.dtype != predicted_delta.dtype or errors.device != predicted_delta.device:
        raise ValueError("path-error inputs must share a floating-point dtype and device")
    lateral, heading = errors.unbind(dim=1)
    dx, dy, dyaw = predicted_delta.unbind(dim=1)
    return torch.stack((
        lateral + torch.sin(heading) * dx + torch.cos(heading) * dy,
        wrap_to_pi(heading + dyaw),
    ), dim=1)


def actor_loss(errors: torch.Tensor, predicted_delta: torch.Tensor) -> torch.Tensor:
    return 0.5 * predict_path_errors(errors, predicted_delta).square().sum(dim=1).mean()
