"""Window-held physical joint residuals conditioned on the latent state."""
from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral

import torch
from torch import nn


class LatentResidualActor(nn.Module):
    """Bounded source-order radians: zero initial mean, reparameterized training samples."""

    residual_limit_rad29: torch.Tensor
    min_log_std = -1.427

    def __init__(
        self,
        feature_dim: int,
        hidden_dims: Sequence[int],
        residual_limit_rad29: Sequence[float] | torch.Tensor,
        *,
        hidden_init_std: float = .01,
        command_dim: int = 7,
        preview_points: int = 8,
        adaptation_dim: int = 0,
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
        if type(adaptation_dim) is not int or adaptation_dim < 0:
            raise ValueError("adaptation_dim must be a nonnegative integer")
        if isinstance(hidden_init_std, bool) or not math.isfinite(hidden_init_std) or hidden_init_std <= 0:
            raise ValueError("hidden_init_std must be positive and finite")
        self.log_std = nn.Parameter(torch.full((29,), -1.))
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
            layers.extend((layer, nn.ELU(), nn.LayerNorm(width)))
            previous = width
        output = nn.Linear(previous, 29)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.net = nn.Sequential(*layers)
        self.adaptation_dim = 0
        self.film: nn.Linear | None = None
        self.register_buffer("default_context", limits.new_zeros(0))
        if adaptation_dim:
            self.enable_film(adaptation_dim)

    def enable_film(self, adaptation_dim: int = 16) -> None:
        if type(adaptation_dim) is not int or adaptation_dim < 1 or self.film is not None:
            raise ValueError("enable FiLM once with a positive context dimension")
        width = self.hidden_dims[-1] if self.hidden_dims else self.input_dim
        self.film = nn.Linear(adaptation_dim, 2 * width, bias=False).to(self.residual_limit_rad29)
        nn.init.normal_(self.film.weight, std=.01)
        self.adaptation_dim = adaptation_dim
        self.default_context = self.residual_limit_rad29.new_zeros(adaptation_dim)

    def train_stage(self, stage: str) -> None:
        if stage not in {"base", "film"} or (stage == "film") != bool(self.adaptation_dim):
            raise ValueError("actor architecture differs from the PPO training stage")
        self.requires_grad_(stage == "base")
        if self.film is not None:
            self.film.requires_grad_(stage == "film")

    def raw_action(self, context: torch.Tensor, adaptation_context: torch.Tensor | None = None) -> torch.Tensor:
        hidden = context
        for index in range(len(self.net) - 1):
            hidden = self.net[index](hidden)
        if self.film is not None:
            value = self.default_context.expand(context.shape[0], -1) if adaptation_context is None else adaptation_context
            if value.shape != (context.shape[0], self.adaptation_dim):
                raise ValueError("adaptation context must have shape [batch, adaptation_dim]")
            scale, shift = self.film(value).chunk(2, dim=-1)
            hidden = hidden * (1. + scale) + shift
        elif adaptation_context is not None and adaptation_context.shape[-1]:
            raise ValueError("base policy has no adaptation interface")
        return self.net[-1](hidden)

    def physical_residual(self, raw_action: torch.Tensor) -> torch.Tensor:
        return raw_action.tanh() * self.residual_limit_rad29

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
                preview: torch.Tensor, previous_residual: torch.Tensor, *,
                adaptation_context: torch.Tensor | None = None) -> torch.Tensor:
        context = self.context(features, errors, command, preview, previous_residual)
        return self.physical_residual(self.raw_action(context, adaptation_context))

    def sample(self, features: torch.Tensor, errors: torch.Tensor, command: torch.Tensor,
               preview: torch.Tensor, previous_residual: torch.Tensor, *,
               adaptation_context: torch.Tensor | None = None) -> torch.Tensor:
        context = self.context(features, errors, command, preview, previous_residual)
        mean = self.raw_action(context, adaptation_context)
        std = self.log_std.clamp_min(self.min_log_std).exp()
        return self.physical_residual(mean + torch.randn_like(mean) * std)
