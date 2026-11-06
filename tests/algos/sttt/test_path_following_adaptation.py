"""Zero-start A, rotated objectives, independent learning and episode boundaries."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from test_path_following_response import fitted_model, write_analytic_capture
from unilab.algos.torch.distill.fada.runtime_policy import FADAFutureQuery, OfficialFADAPlaybackProfile
from unilab.algos.torch.sttt.path_following import compose_joint_position_residual
from unilab.algos.torch.sttt.path_following.checkpoint import LoadedResponseModel
from unilab.algos.torch.sttt.path_following.dataset import iter_path_following_samples, read_path_following_metadata
from unilab.algos.torch.sttt.path_following.features import response_contract_from_metadata
from unilab.algos.torch.sttt.path_following.objectives import actor_loss, predict_path_errors
from unilab.algos.torch.sttt.path_following.online_learning import ActorUpdateContext, PathFollowingUpdater
from unilab.base.backend.sensors import SensorClock

ROOT = Path(__file__).resolve().parents[3]


def make_updater(directory):
    response, _ = fitted_model(directory / "b_data")
    contract = response_contract_from_metadata(read_path_following_metadata(directory / "b_data"))
    cfg = OmegaConf.merge(OmegaConf.load(ROOT / "conf/sttt/path_following_online.yaml"), {"actor": {"hidden_dims": [16, 8]}})
    assert isinstance(cfg, DictConfig)
    updater = PathFollowingUpdater(LoadedResponseModel(response, contract, {}), cfg)
    write_analytic_capture(directory / "zero", seed=1, varying=False, samples=6)
    windows = list(iter_path_following_samples(directory / "zero"))
    return updater, windows


def weights(module):
    return {name: value.detach().clone() for name, value in module.named_parameters()}


def assert_weights(module, expected):
    for name, value in module.named_parameters():
        torch.testing.assert_close(value, expected[name], atol=0, rtol=0)


def reidentify(sample, reference):
    return replace(
        sample, reference=reference,
        start_pose=replace(sample.start_pose, sequence_id=reference.sequence_id, tracking_epoch=reference.tracking_epoch, frame_id=reference.frame_id),
        end_pose=replace(sample.end_pose, sequence_id=reference.sequence_id, tracking_epoch=reference.tracking_epoch, frame_id=reference.frame_id),
        controls=tuple(replace(control, sequence_id=reference.sequence_id) for control in sample.controls),
    )


@pytest.mark.parametrize("heading,turn", [(0.3, -0.05), (3., 0.4)])
def test_actor_objective_keeps_forward_rotation_and_wrapped_heading_gradient(heading, turn):
    errors = torch.tensor([[0.2, heading]], dtype=torch.float64)
    delta = torch.tensor([[0.1, 0.02, turn]], dtype=torch.float64, requires_grad=True)
    lateral = 0.2 + math.sin(heading) * 0.1 + math.cos(heading) * 0.02
    yaw = (heading + turn + math.pi) % (2 * math.pi) - math.pi
    expected = torch.tensor([[lateral, yaw]], dtype=torch.float64)
    torch.testing.assert_close(predict_path_errors(errors, delta), expected, atol=1e-12, rtol=0)
    loss = actor_loss(errors, delta)
    assert float(loss.detach()) == pytest.approx(0.5 * (lateral**2 + yaw**2))
    gradient, = torch.autograd.grad(loss, delta)
    torch.testing.assert_close(gradient, torch.tensor([[lateral * math.sin(heading), lateral * math.cos(heading), yaw]], dtype=torch.float64), atol=1e-12, rtol=0)


def test_zero_start_calibration_and_first_two_actor_updates_are_isolated(tmp_path):
    updater, windows = make_updater(tmp_path)
    decision = ActorUpdateContext.from_window_start(windows[0])
    initial_actor = weights(updater.actor)
    hidden = weights(updater.actor.hidden)
    fixed_buffers = {name: value.clone() for name, value in updater.response.named_buffers()}
    residual = updater.actor(decision.errors, decision.context)
    assert torch.equal(residual, torch.zeros_like(residual))
    query = windows[0].start_query
    adapted = compose_joint_position_residual(query.planner_future, residual, ctrl_dt=query.ctrl_dt, position_scale=query.position_scale, velocity_scale=query.velocity_scale)
    assert torch.equal(adapted, query.planner_future)
    first = updater.update([windows[0]], decision)
    assert first.a_loss is None and not first.calibration_complete
    assert_weights(updater.actor, initial_actor)
    with torch.no_grad():
        second = updater.update([windows[1]], ActorUpdateContext.from_window_start(windows[1]))
    assert second.a_loss is not None and second.calibration_complete
    assert_weights(updater.actor.hidden, hidden)
    assert any(bool(torch.count_nonzero(value)) for value in updater.actor.output.parameters())
    before_b = weights(updater.response)
    updater.update([], ActorUpdateContext.from_window_start(windows[2]))
    assert_weights(updater.response, before_b)
    assert any(not torch.equal(value, hidden[name]) for name, value in updater.actor.hidden.named_parameters())
    before_a = weights(updater.actor)
    updater.update([windows[2]])
    assert_weights(updater.actor, before_a)
    for name, value in updater.response.named_buffers():
        torch.testing.assert_close(value, fixed_buffers[name], atol=0, rtol=0)
    assert all(parameter.grad is None for parameter in (*updater.actor.parameters(), *updater.response.parameters()))
    before = weights(updater.response)
    with pytest.raises(ValueError):
        updater.update([windows[0]])
    assert_weights(updater.response, before)
    assert updater.zero_calibration_windows == 2


def test_decision_factory_rejects_timestamp_and_localization_mismatch(tmp_path):
    updater, windows = make_updater(tmp_path)
    sample = windows[1]
    record = sample.start_query
    source = updater.contract["source"]
    profile = OfficialFADAPlaybackProfile.from_mapping(source["source_profile"], source_identity=source["source_profile"]["source_identity"])
    query = FADAFutureQuery(record.observation_history, record.action_history, record.command, record.history_valid_mask, record.planner_future, source["preprocessing"], profile)
    clock = SensorClock(sample.start_pose.sequence_id, sample.start_pose.timestamp_ns)
    decision = ActorUpdateContext.from_aligned_query(query, clock, sample.start_pose, sample.reference)
    torch.testing.assert_close(decision.errors, torch.as_tensor(sample.path_errors[:1], dtype=decision.errors.dtype))
    with pytest.raises(ValueError):
        ActorUpdateContext.from_aligned_query(query, SensorClock(clock.sequence_id, clock.timestamp_ns + 1), sample.start_pose, sample.reference)
    with pytest.raises(ValueError):
        ActorUpdateContext.from_aligned_query(query, clock, replace(sample.start_pose, tracking_epoch=2), sample.reference)


def test_reset_preserves_learning_but_restarts_incomplete_calibration(tmp_path):
    updater, windows = make_updater(tmp_path)
    updater.update([windows[0]])
    before = weights(updater.response)
    old_steps = [int(state["step"]) for state in updater.response_optimizer.state.values()]
    new_reference = replace(windows[0].reference, sequence_id=2)
    updater.begin_episode(new_reference)
    assert not updater.calibration_complete and updater.zero_calibration_windows == 0
    assert_weights(updater.response, before)
    assert [int(state["step"]) for state in updater.response_optimizer.state.values()] == old_steps
    with pytest.raises(ValueError):
        updater.update([windows[1]])
    newer = [reidentify(window, new_reference) for window in windows]
    updater.update(newer[:2], ActorUpdateContext.from_window_start(newer[1]))
    assert updater.calibration_complete
    actor = weights(updater.actor)
    updater.begin_episode(replace(new_reference, sequence_id=3))
    assert updater.calibration_complete
    assert_weights(updater.actor, actor)


def test_actor_failure_restores_b_mode_flags_and_allows_next_b_update(tmp_path):
    updater, windows = make_updater(tmp_path)
    updater.update(windows[:2])
    assert updater.calibration_complete
    updater.response.eval()
    before = weights(updater.response)
    flags = [parameter.requires_grad for parameter in updater.response.parameters()]
    with torch.no_grad():
        assert updater.actor.output.bias is not None
        updater.actor.output.bias.fill_(float("nan"))
    with pytest.raises(ValueError):
        updater.update([], ActorUpdateContext.from_window_start(windows[2]))
    assert not updater.response.training
    assert [parameter.requires_grad for parameter in updater.response.parameters()] == flags
    assert_weights(updater.response, before)
    result = updater.update([windows[2]])
    assert result.b_loss is not None and result.a_loss is None
    assert updater.response_updates == 2
    assert all(parameter.grad is None for parameter in updater.response.parameters())
