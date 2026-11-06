"""A starts at exactly zero and emits source-order physical joint radians."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from unilab.algos.torch.sttt.path_following.contract import _integer
from unilab.algos.torch.sttt.path_following.features import normalized_context_parts
from unilab.algos.torch.sttt.path_following.model import ResponseModel


class ResidualModel(nn.Module):
    context_mean: torch.Tensor
    context_inv_scale: torch.Tensor

    def __init__(self, response: ResponseModel, hidden_dims: Sequence[int], *, hidden_init_std: float) -> None:
        super().__init__()
        widths = tuple(_integer(width, "A hidden width") for width in hidden_dims)
        if len(widths) != 2 or min(widths) < 1:
            raise ValueError("A requires two positive hidden widths")
        if isinstance(hidden_init_std, bool) or not math.isfinite(hidden_init_std) or hidden_init_std <= 0:
            raise ValueError("A hidden_init_std must be positive and finite")
        if int(response.normalization_samples) < 1:
            raise ValueError("A requires B's fitted context interpretation")
        self.history_length = response.history_length
        self.prediction_horizon = response.prediction_horizon
        self.context_dim = response.context_dim
        self.register_buffer("context_mean", response.context_mean.detach().clone())
        self.register_buffer("context_inv_scale", response.context_inv_scale.detach().clone())
        # Do not store response as a child module: A.parameters() must never own B.
        dtype = response.context_mean.dtype
        self.hidden = nn.Sequential(
            nn.Linear(self.context_dim + 2, widths[0], device="cpu", dtype=dtype), nn.SiLU(),
            nn.Linear(widths[0], widths[1], device="cpu", dtype=dtype), nn.SiLU(),
        )
        self.output = nn.Linear(widths[1], 29, device="cpu", dtype=dtype)
        for layer in self.hidden:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0., std=hidden_init_std)
                assert layer.bias is not None
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.output.weight)
        assert self.output.bias is not None
        nn.init.zeros_(self.output.bias)
        self.to(device=response.context_mean.device)

    def forward(self, errors: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 2 or context.shape[0] < 1 or context.shape[1] != self.context_dim or errors.shape != (context.shape[0], 2):
            raise ValueError("A requires context [batch, context_dim] and measured errors [batch, 2]")
        if context.dtype != self.context_mean.dtype or errors.dtype != context.dtype or context.device != self.context_mean.device or errors.device != context.device:
            raise ValueError("A inputs must match its dtype and device")
        parts = normalized_context_parts(context, self.context_mean, self.context_inv_scale, self.history_length)
        hidden = self.hidden(torch.cat((*parts, errors), dim=1))
        return self.output(hidden)
