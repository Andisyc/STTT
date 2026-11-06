"""Predictive error and model derivative consistency, not causal physics proof."""
from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch

from unilab.algos.torch.sttt.path_following.model import ResponseBatch, ResponseModel


@torch.no_grad()
def evaluate_response_model(model: ResponseModel, batches: Iterable[ResponseBatch]) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    reference = next(model.parameters())
    squared = torch.zeros(3, dtype=torch.float64)
    absolute = torch.zeros(3, dtype=torch.float64)
    count = 0
    try:
        for context, residual, target in batches:
            prediction = model(context.to(reference), residual.to(reference))
            if target.shape != prediction.shape or not bool(torch.isfinite(target).all()):
                raise ValueError("validation requires finite [batch, 3] measured displacements")
            error = (prediction - target.to(reference)).cpu().double()
            if not bool(torch.isfinite(error).all()):
                raise ValueError("B produced non-finite validation predictions")
            squared += error.square().sum(0)
            absolute += error.abs().sum(0)
            count += error.shape[0]
    finally:
        model.train(was_training)
    if count == 0:
        raise ValueError("independent B validation requires nonempty held-out data")
    return {
        "samples": count, "loss": float(0.5 * squared.sum() / count),
        "rmse_m_m_rad": (squared / count).sqrt().tolist(),
        "mae_m_m_rad": (absolute / count).tolist(),
    }


def diagnose_response_jacobian(
    model: ResponseModel, context: torch.Tensor, residual_rad: torch.Tensor, *, epsilon_rad: float,
) -> dict[str, Any]:
    """Derivative is d[dx,dy,dyaw]/d(physical joint radians), including normalization."""
    if isinstance(epsilon_rad, bool) or not math.isfinite(epsilon_rad) or epsilon_rad <= 0:
        raise ValueError("finite-difference epsilon must be positive physical radians")
    if context.shape != (1, model.context_dim) or residual_rad.shape != (1, 29):
        raise ValueError("Jacobian diagnostic requires exactly one context and residual")
    if torch.is_inference_mode_enabled():
        raise ValueError("Jacobian diagnostics must run outside torch.inference_mode")
    was_training = model.training
    parameters = list(model.parameters())
    flags = [parameter.requires_grad for parameter in parameters]
    reference = parameters[0]
    model.eval()
    for parameter in parameters:
        parameter.requires_grad_(False)
    try:
        x = context.detach().to(reference)
        with torch.enable_grad():
            residual = residual_rad.detach().to(reference).clone().requires_grad_(True)
            prediction = model(x, residual)
            jacobian = torch.stack([
                torch.autograd.grad(prediction[0, axis], residual, retain_graph=axis < 2)[0][0]
                for axis in range(3)
            ])
        with torch.no_grad():
            base = residual.detach()
            perturbations = torch.eye(29, dtype=reference.dtype, device=reference.device) * epsilon_rad
            plus = model(x.expand(29, -1), base + perturbations)
            minus = model(x.expand(29, -1), base - perturbations)
            numeric = ((plus - minus) / (2 * epsilon_rad)).T
            if not bool(torch.isfinite(jacobian).all()) or not bool(torch.isfinite(numeric).all()):
                raise ValueError("non-finite model response derivative")
            inside = bool(((base >= model.residual_min) & (base <= model.residual_max)).all())
            probes_inside = inside & ((base[0] - epsilon_rad >= model.residual_min) & (base[0] + epsilon_rad <= model.residual_max))
            return {
                "evidence": "model_autograd_vs_finite_difference_only",
                "physical_direction_verified": False, "epsilon_rad": float(epsilon_rad),
                "jacobian": jacobian.cpu().tolist(), "finite_difference": numeric.cpu().tolist(),
                "max_abs_difference": float((jacobian - numeric).abs().max()),
                "base_inside_training_bounds": inside,
                "probe_inside_training_bounds": probes_inside.cpu().tolist(),
            }
    finally:
        for parameter, flag in zip(parameters, flags):
            parameter.requires_grad_(flag)
        model.train(was_training)
