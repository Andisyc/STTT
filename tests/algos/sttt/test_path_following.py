"""Consumer-visible compact64 intervention invariants."""

from __future__ import annotations

import pytest
import torch

from unilab.algos.torch.sttt.path_following import compose_joint_position_residual


@pytest.mark.parametrize("horizon", [1, 4])
def test_physical_residual_uses_first_frame_and_scaled_velocity_difference(horizon):
    future = torch.arange(2 * horizon * 64, dtype=torch.float64).reshape(2, horizon, 64) / 100
    original = future.clone()
    residual = torch.linspace(-0.03, 0.04, 58, dtype=torch.float64).reshape(2, 29).requires_grad_()
    dt, sq, sv = 0.02, 2.5, 0.15
    adapted = compose_joint_position_residual(
        future, residual, ctrl_dt=dt, position_scale=sq, velocity_scale=sv,
    )
    offsets = torch.cat((residual[:, None], torch.zeros(2, horizon - 1, 29, dtype=torch.float64)), dim=1)
    expected = original.clone()
    expected[:, :, 3:32] += sq * offsets
    expected[:, :, 32:61] += sv * torch.diff(offsets, dim=1, prepend=torch.zeros(2, 1, 29)) / dt
    torch.testing.assert_close(adapted, expected, rtol=1e-14, atol=1e-14)
    assert torch.equal(future, original)
    assert torch.equal(adapted[:, :, :3], original[:, :, :3])
    assert torch.equal(adapted[:, :, 61:], original[:, :, 61:])
    objective = 2 * adapted[:, 0, 3].sum() + 3 * adapted[:, 0, 32].sum()
    if horizon > 1:
        objective = objective + 5 * adapted[:, 1, 32].sum()
    gradient, = torch.autograd.grad(objective, residual)
    expected_gradient = torch.zeros_like(residual)
    expected_gradient[:, 0] = 2 * sq + (3 - (5 if horizon > 1 else 0)) * sv / dt
    torch.testing.assert_close(gradient, expected_gradient, rtol=1e-14, atol=0)


def test_zero_residual_preserves_arbitrary_planner_velocity_exactly():
    future = torch.arange(3 * 64, dtype=torch.float32).reshape(1, 3, 64) / 13
    adapted = compose_joint_position_residual(
        future, torch.zeros(1, 29), ctrl_dt=0.017, position_scale=1.7, velocity_scale=0.23,
    )
    assert torch.equal(adapted, future)


@pytest.mark.parametrize("failure", ["broadcast", "dtype", "nonfinite", "time"])
def test_intervention_rejects_uninterpretable_residuals(failure):
    future, residual = torch.zeros(2, 3, 64), torch.zeros(2, 29)
    dt = 0.02
    if failure == "broadcast":
        residual = residual[:1]
    elif failure == "dtype":
        residual = residual.double()
    elif failure == "nonfinite":
        residual[0, 0] = float("nan")
    else:
        dt = 0.0
    with pytest.raises(ValueError):
        compose_joint_position_residual(
            future, residual, ctrl_dt=dt, position_scale=1.0, velocity_scale=0.05,
        )
