"""Window-held physical joint residuals conditioned on the latent state."""
from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral

import torch
from torch import nn


class LatentResidualActor(nn.Module):
    """Emit bounded radians in source joint order, initially exactly zero."""

    residual_limit_rad29: torch.Tensor

    def __init__(
        self,
        feature_dim: int,
        hidden_dims: Sequence[int],
        residual_limit_rad29: Sequence[float] | torch.Tensor,
        *,
        hidden_init_std: float = .01,
        command_dim: int = 7,
        preview_points: int = 8,
    ) -> None:
        super().__init__()
        widths = tuple(hidden_dims)
        if (
            isinstance(feature_dim, bool) or not isinstance(feature_dim, Integral) or feature_dim < 1
            or any(isinstance(width, bool) or not isinstance(width, Integral) or width < 1 for width in widths)
        ):
            raise ValueError("latent actor requires positive integer feature and hidden dimensions")
        if type(command_dim) is not int or command_dim not in (3, 7):
            raise ValueError("latent actor requires an admitted 3D or 7D command")
        self.command_dim = command_dim
        if type(preview_points) is not int or preview_points < 2:
            raise ValueError("latent actor requires at least two reference preview points")
        self.preview_points = preview_points
        if isinstance(hidden_init_std, bool) or not math.isfinite(hidden_init_std) or hidden_init_std <= 0:
            raise ValueError("hidden_init_std must be positive and finite")
        limits = torch.as_tensor(residual_limit_rad29, dtype=torch.float32).detach().clone()
        if limits.shape != (29,) or not bool(torch.isfinite(limits).all()) or bool((limits <= 0).any()):
            raise ValueError("latent actor requires 29 positive finite physical residual limits")
        self.feature_dim = int(feature_dim)
        self.hidden_dims = tuple(int(width) for width in widths)
        self.register_buffer("residual_limit_rad29", limits)
        layers: list[nn.Module] = []
        self.input_dim = self.feature_dim + 2 + self.command_dim + 5 * self.preview_points + 29
        previous = self.input_dim
        for width in self.hidden_dims:
            layer = nn.Linear(previous, width)
            nn.init.normal_(layer.weight, std=hidden_init_std)
            nn.init.zeros_(layer.bias)
            layers.extend((layer, nn.SiLU()))
            previous = width
        output = nn.Linear(previous, 29)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.net = nn.Sequential(*layers)

    def context(self, features: torch.Tensor, errors: torch.Tensor, command: torch.Tensor,
                preview: torch.Tensor, previous_residual: torch.Tensor) -> torch.Tensor:
        if (
            features.ndim != 2 or features.shape[0] < 1 or features.shape[1] != self.feature_dim
            or errors.shape != (features.shape[0], 2) or command.shape != (features.shape[0], self.command_dim)
            or preview.shape != (features.shape[0], self.preview_points, 5)
            or previous_residual.shape != (features.shape[0], 29)
        ):
            raise ValueError("latent actor inputs differ from its execution contract")
        for value in (features, errors, command, preview, previous_residual):
            if value.dtype != self.residual_limit_rad29.dtype or value.device != self.residual_limit_rad29.device:
                raise ValueError("latent actor inputs must match its floating-point dtype and device")
            if not bool(torch.isfinite(value).all()):
                raise ValueError("latent actor inputs must be finite")
        return torch.cat((features, errors, command, preview.flatten(1),
                          previous_residual / self.residual_limit_rad29), dim=-1)

    def forward(self, features: torch.Tensor, errors: torch.Tensor, command: torch.Tensor,
                preview: torch.Tensor, previous_residual: torch.Tensor) -> torch.Tensor:
        context = self.context(features, errors, command, preview, previous_residual)
        return torch.tanh(self.net(context)) * self.residual_limit_rad29
