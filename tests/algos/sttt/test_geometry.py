"""Geometry behavior retained independently of retired spatial learning code."""

from __future__ import annotations

import math

import torch

from unilab.algos.torch.sttt import compute_se2_deltas, mat_to_quat, quat_to_mat


def test_geometry_se2_deltas_and_rotations():
    batch_size = 2
    horizon = 5
    quat0 = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    pos0 = torch.zeros(batch_size, 3)
    future_pos = torch.zeros(batch_size, horizon, 3)
    for h in range(horizon):
        future_pos[:, h, 0] = (h + 1) * 0.1
    future_quat = quat0.unsqueeze(1).expand(-1, horizon, -1)
    deltas = compute_se2_deltas(future_pos, future_quat, pos0=pos0, quat0=quat0)
    assert torch.allclose(deltas[..., 0], future_pos[..., 0], atol=1e-5)
    assert torch.allclose(deltas[..., 1], torch.zeros_like(deltas[..., 1]), atol=1e-5)
    assert torch.allclose(deltas[..., 2], torch.zeros_like(deltas[..., 2]), atol=1e-5)

    val = math.sqrt(2) / 2
    quat_90 = torch.tensor([val, 0.0, 0.0, val]).expand(batch_size, horizon, 4)
    deltas_rot = compute_se2_deltas(future_pos, quat_90, pos0=pos0, quat0=quat0)
    assert torch.allclose(deltas_rot[..., 2], torch.full_like(deltas_rot[..., 2], math.pi / 2), atol=1e-5)
    recon_q = mat_to_quat(quat_to_mat(quat0))
    assert torch.allclose(recon_q, quat0, atol=1e-6)
