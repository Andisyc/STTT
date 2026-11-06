"""B predicts physical displacement; fixed input statistics remain differentiable."""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import torch
from torch import nn

from unilab.algos.torch.common.normalization import EmpiricalNormalization
from unilab.algos.torch.sttt.path_following.contract import _integer
from unilab.algos.torch.sttt.path_following.features import context_dimension, normalized_context_parts

ResponseBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class ResponseModel(nn.Module):
    context_mean: torch.Tensor
    context_inv_scale: torch.Tensor
    residual_mean: torch.Tensor
    residual_inv_scale: torch.Tensor
    residual_min: torch.Tensor
    residual_max: torch.Tensor
    normalization_samples: torch.Tensor

    def __init__(self, history_length: int, prediction_horizon: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        self.history_length = _integer(history_length, "history_length")
        self.prediction_horizon = _integer(prediction_horizon, "prediction_horizon")
        widths = tuple(_integer(width, "hidden width") for width in hidden_dims)
        if self.history_length < 1 or self.prediction_horizon < 1 or len(widths) != 2 or min(widths) < 1:
            raise ValueError("response model requires positive H/K and two positive hidden widths")
        self.hidden_dims = widths
        self.context_dim = context_dimension(self.history_length, self.prediction_horizon)
        self._obs_end = 64 * self.history_length
        self._act_end = self._obs_end + 29 * self.history_length
        self._mask_end = self._act_end + self.history_length
        self.net = nn.Sequential(
            nn.Linear(self.context_dim + 29, widths[0]), nn.SiLU(),
            nn.Linear(widths[0], widths[1]), nn.SiLU(), nn.Linear(widths[1], 3),
        )
        for name, width, initial in (
            ("context_mean", self.context_dim, 0.), ("context_inv_scale", self.context_dim, 1.),
            ("residual_mean", 29, 0.), ("residual_inv_scale", 29, 1.),
            ("residual_min", 29, 0.), ("residual_max", 29, 0.),
        ):
            self.register_buffer(name, torch.full((width,), initial))
        self.register_buffer("normalization_samples", torch.tensor(0, dtype=torch.long))

    @property
    def model_config(self) -> dict:
        return {"history_length": self.history_length, "prediction_horizon": self.prediction_horizon, "hidden_dims": list(self.hidden_dims)}

    def _check_shapes(self, context: torch.Tensor, residual_rad: torch.Tensor) -> None:
        if context.ndim != 2 or context.shape[0] < 1 or context.shape[1] != self.context_dim or residual_rad.shape != (context.shape[0], 29):
            raise ValueError("response inputs require [batch, context_dim] and physical residual [batch, 29]")

    def forward(self, context: torch.Tensor, residual_rad: torch.Tensor) -> torch.Tensor:
        self._check_shapes(context, residual_rad)
        if context.dtype != self.context_mean.dtype or residual_rad.dtype != context.dtype or context.device != self.context_mean.device or residual_rad.device != context.device:
            raise ValueError("response inputs must match the model dtype and device")
        parts = normalized_context_parts(context, self.context_mean, self.context_inv_scale, self.history_length)
        residual = (residual_rad - self.residual_mean) * self.residual_inv_scale
        inputs = torch.cat((*parts, residual), dim=1)
        return self.net(inputs)

    @torch.no_grad()
    def fit_normalization(self, batches: Iterable[ResponseBatch], *, epsilon: float) -> None:
        """Cold train-only fit; never call the shared no-grad normalizer.forward."""
        if isinstance(epsilon, bool) or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("normalization epsilon must be positive and finite")
        statistics = [EmpiricalNormalization(width, "cpu").double() for width in (64, 29, 7, 64, 29)]
        lower, upper = torch.full((29,), float("inf"), dtype=torch.float64), torch.full((29,), -float("inf"), dtype=torch.float64)
        for context, residual, _target in batches:
            self._check_shapes(context, residual)
            x = context.detach().to(device="cpu", dtype=torch.float64)
            r = residual.detach().to(device="cpu", dtype=torch.float64)
            mask = x[:, self._act_end:self._mask_end]
            if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(r).all()) or not bool(((mask == 0) | (mask == 1)).all()):
                raise ValueError("normalizer requires finite inputs and boolean-valued history masks")
            valid = mask.bool()
            groups = (
                x[:, :self._obs_end].reshape(-1, self.history_length, 64)[valid],
                x[:, self._obs_end:self._act_end].reshape(-1, self.history_length, 29)[valid],
                x[:, self._mask_end:self._mask_end + 7],
                x[:, self._mask_end + 7:].reshape(-1, 64), r,
            )
            for statistic, values in zip(statistics, groups):
                if values.shape[0]:
                    statistic.update(values)
            lower = torch.minimum(lower, r.amin(0))
            upper = torch.maximum(upper, r.amax(0))
        if any(int(statistic.count) == 0 for statistic in statistics) or not bool((upper > lower).all()):
            raise ValueError("B pretraining requires valid history and varying residuals in all 29 channels")
        means = [statistic.mean for statistic in statistics]
        scales = [statistic.std + epsilon for statistic in statistics]
        h, k = self.history_length, self.prediction_horizon
        self.context_mean.copy_(torch.cat((means[0].repeat(h), means[1].repeat(h), torch.zeros(h), means[2], means[3].repeat(k))))
        self.context_inv_scale.copy_(torch.cat((scales[0].repeat(h), scales[1].repeat(h), torch.ones(h), scales[2], scales[3].repeat(k))).reciprocal())
        self.residual_mean.copy_(means[4])
        self.residual_inv_scale.copy_(scales[4].reciprocal())
        self.residual_min.copy_(lower)
        self.residual_max.copy_(upper)
        self.normalization_samples.copy_(statistics[4].count)


def response_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.ndim != 2 or prediction.shape[0] < 1 or prediction.shape[1] != 3 or target.shape != prediction.shape:
        raise ValueError("B loss requires matching nonempty [batch, 3] displacement tensors")
    return 0.5 * (prediction - target).square().sum(dim=1).mean()
