"""Differentiable spatial world model backbone for STTT.

Pure PyTorch. Predicts relative SE(2) trajectory deltas ``[dx, dy, dyaw]``
in the robot's local horizontal frame at ``t = 0`` from an observation
history and a candidate action sequence.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class RunningNorm(nn.Module):
    """Running mean/variance normalization via batched Welford updates.

    Statistics are buffers (no autograd). Fit them explicitly on training
    data only; subsequent forward/evaluation calls never update them.
    Updates freeze after a complete batch reaches ``max_count``.
    """

    mean: torch.Tensor
    var: torch.Tensor
    count: torch.Tensor

    def __init__(
        self,
        dim: int,
        max_count: float = 2e8,
        eps: float = 1e-4,
        no_sync: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_count = float(max_count)
        self.eps = float(eps)
        # Preserve the existing constructor API; updates run on the buffer device.
        self.no_sync = no_sync
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Merge a batch of samples into the running statistics."""
        x = x.reshape(-1, self.dim).to(self.mean.dtype)
        batch_count = x.shape[0]
        if batch_count == 0 or self.count >= self.max_count:
            return

        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)

        count = self.count
        new_count = count + batch_count

        # Parallel Welford merge: correction uses pre-merge mean difference.
        delta = batch_mean - self.mean
        self.mean.copy_(self.mean + delta * (batch_count / new_count))
        m_a = self.var * count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * (count * batch_count / new_count)
        self.var.copy_(m2 / new_count)
        self.count.copy_(new_count)

    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var + self.eps**2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std()


def mlp(
    in_dim: int,
    hidden: Sequence[int],
    out_dim: int,
    act: type[nn.Module] = nn.SiLU,
    layer_norm: bool = True,
) -> nn.Sequential:
    """Multi-layer perceptron with optional LayerNorm after each hidden Linear."""
    layers: list[nn.Module] = []
    prev_dim = in_dim
    for h_dim in hidden:
        layers.append(nn.Linear(prev_dim, h_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(h_dim))
        layers.append(act())
        prev_dim = h_dim
    layers.append(nn.Linear(prev_dim, out_dim))
    return nn.Sequential(*layers)


class SpatialWorldModel(nn.Module):
    """Predicts relative SE(2) trajectory deltas from obs history + actions.

    Input:
        - ``obs_history``: ``(batch, hist_len, obs_dim)``
        - ``action_sequence``: ``(batch, horizon, act_dim)``
    Output:
        - ``(batch, horizon, 3)`` per-step ``[dx, dy, dyaw]`` deltas in the
          robot's local horizontal frame at ``t = 0``.

    The final layer is zero-initialized and the output is unnormalized via
    ``out_norm`` (initial mean 0, std ~1), so initial predictions are
    exactly zero.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        horizon: int = 15,
        hist_len: int = 15,
        hidden: Sequence[int] = (512, 512, 512),
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.horizon = horizon
        self.hist_len = hist_len
        self.hidden_dims = tuple(hidden)

        in_dim = obs_dim * hist_len + act_dim * horizon
        out_dim = horizon * 3

        self.in_norm = RunningNorm(in_dim)
        self.out_norm = RunningNorm(out_dim, eps=1e-4)
        self.net = mlp(in_dim, hidden, out_dim, act=nn.SiLU, layer_norm=True)

        # Zero-init output head: initial predicted increments are exactly zero.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs_hist: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        batch = obs_hist.shape[0]
        x = torch.cat(
            [obs_hist.reshape(batch, -1), action_seq.reshape(batch, -1)],
            dim=-1,
        )
        x = self.in_norm(x)
        pred = self.net(x)
        pred = pred * self.out_norm.std() + self.out_norm.mean
        return pred.reshape(batch, self.horizon, 3)

    def compute_loss(
        self,
        pred_deltas: torch.Tensor,
        target_deltas: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Weighted position and periodic yaw error in physical units.

        ``weights`` accepts coordinate weights ``(3,)``, sample/horizon
        weights ``(batch, horizon)``, or a tensor broadcastable to the output.
        """
        error = pred_deltas - target_deltas
        yaw_error = torch.atan2(torch.sin(error[..., 2]), torch.cos(error[..., 2]))
        sq_err = torch.cat((error[..., :2], yaw_error.unsqueeze(-1)), dim=-1).square()
        if weights is None:
            return sq_err.mean()
        w = weights.to(sq_err)
        if w.ndim == 2:
            w = w.unsqueeze(-1)
        w = torch.broadcast_to(w, sq_err.shape)
        if not torch.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
            raise ValueError("Loss weights must be finite, non-negative, and have positive sum")
        return (sq_err * w).sum() / w.sum()
