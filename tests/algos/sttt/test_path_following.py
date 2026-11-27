"""Consumer-visible persistent joint-position intervention invariants."""

from __future__ import annotations

import pytest
import torch

from unilab.algos.torch.sttt.path_following import compose_joint_position_residual


@pytest.mark.parametrize("horizon", [1, 4])
@pytest.mark.parametrize("contract,width,position,velocity", [
    ("official_compact_64_v1", 64, slice(3, 32), slice(32, 61)),
    ("g1_fada_state_v2", 66, slice(6, 35), slice(35, 64)),
])
def test_physical_residual_offsets_entire_future_without_velocity_pulses(horizon, contract, width, position, velocity):
    future = torch.arange(2 * horizon * width, dtype=torch.float64).reshape(2, horizon, width) / 100
    original = future.clone()
    residual = torch.linspace(-0.03, 0.04, 58, dtype=torch.float64).reshape(2, 29).requires_grad_()
    sq = 2.5
    adapted = compose_joint_position_residual(
        future, residual, ctrl_dt=.02, position_scale=sq, velocity_scale=.15,
        observation_contract=contract,
    )
    expected = original.clone()
    expected[:, :, position] += sq * residual[:, None, :]
    torch.testing.assert_close(adapted, expected, rtol=1e-14, atol=1e-14)
    assert torch.equal(future, original)
    assert torch.equal(adapted[:, :, velocity], original[:, :, velocity])
    weights = torch.arange(1, horizon + 1, dtype=torch.float64)[None, :, None]
    objective = (adapted[:, :, position] * weights).sum() + adapted[:, :, velocity].sum()
    gradient, = torch.autograd.grad(objective, residual)
    torch.testing.assert_close(gradient, torch.full_like(residual, sq * horizon * (horizon + 1) / 2), rtol=1e-14, atol=0)


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
