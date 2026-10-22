"""Behavior contracts for fixed-label streaming adaptation."""

from __future__ import annotations

import pytest
import torch

from unilab.algos.torch.distill.fada.adaptation import FADALoRAConfig, fada_adapter_named_parameters
from unilab.algos.torch.distill.fada.model import FADAArchitectureConfig, FADAPlannerIDMPolicy
from unilab.algos.torch.distill.fada.online_collector import OnlineSnapshotCollector
from unilab.algos.torch.distill.fada.online_data import FADAInferenceSnapshot
from unilab.algos.torch.distill.fada.online_updater import OnlineSnapshotUpdater
from unilab.algos.torch.distill.fada.playback import FADAPlaybackController


@pytest.fixture
def config():
    return FADAArchitectureConfig(
        obs_dim=66, action_dim=29, command_dim=3,
        observation_contract="g1_fada_state_v2", history_length=3,
        prediction_horizon=2, hidden_dim=8, num_heads=2,
        planner_layers=1, idm_encoder_layers=1, idm_decoder_layers=1,
        feedforward_dim=16, dropout=0.0,
    )


def _snapshot():
    future = torch.zeros(1, 2, 66)
    future[:, 0, 6:35] = 0.10
    future[:, 1, 6:35] = 10.0
    return FADAInferenceSnapshot(
        torch.full((1, 3, 66), -1.0), torch.zeros(1, 3, 29),
        torch.zeros(1, 3), future, torch.full((1, 29), 0.20),
    )


def test_label_uses_next_state_and_signed_joint_scales(config):
    scale = torch.full((29,), 0.5)
    scale[1] = -0.5
    scale[2] = 1.0
    collector = OnlineSnapshotCollector(config, window_size=1, gamma=0.05, action_scale=scale)
    collector.begin(_snapshot())
    following = torch.zeros(1, 66)
    following[:, 6:35] = 0.08
    sample = collector.complete(following)
    assert sample is not None
    following.fill_(999)
    scale.fill_(999)
    torch.testing.assert_close(sample.joint_position_error, torch.full((1, 29), 0.02))
    expected = torch.full((1, 29), 0.202)
    expected[0, 1] = 0.198
    expected[0, 2] = 0.201
    torch.testing.assert_close(collector.get_window()[0].target_action, expected)


def test_window_keeps_latest_samples_and_requires_reset_after_terminal(config):
    collector = OnlineSnapshotCollector(config, window_size=2, gamma=0.05, action_scale=0.5)
    for index in range(3):
        collector.begin(_snapshot())
        collector.complete(torch.zeros(1, 66))
        assert collector.ready == (index >= 1)
    assert [sample.timestep for sample in collector.get_window()] == [1, 2]
    old_episode = collector.get_window()[0].episode_id
    collector.begin(_snapshot())
    assert collector.complete(torch.full((1, 98), float("nan")), done=True) is None
    assert collector.get_window() == ()
    with pytest.raises(RuntimeError):
        collector.begin(_snapshot())
    collector.reset()
    collector.begin(_snapshot())
    sample = collector.complete(torch.zeros(1, 66))
    assert sample is not None and (sample.episode_id, sample.timestep) == (old_episode + 1, 0)


def test_clear_window_preserves_pending_sample_and_episode_timestep(config):
    collector = OnlineSnapshotCollector(config, window_size=1, gamma=0.05, action_scale=0.5)
    collector.begin(_snapshot())
    first = collector.complete(torch.zeros(1, 66))
    assert first is not None and collector.ready
    collector.begin(_snapshot())
    collector.clear_window()
    assert collector.get_window() == () and not collector.ready
    following = collector.complete(torch.zeros(1, 66))
    assert following is not None
    assert (following.episode_id, following.timestep) == (first.episode_id, first.timestep + 1)
    collector.begin(_snapshot())
    collector.complete(torch.zeros(1, 66), done=True)
    collector.clear_window()
    with pytest.raises(RuntimeError):
        collector.begin(_snapshot())


def test_regression_uses_squared_action_norm_and_only_first_action(config):
    with torch.random.fork_rng():
        torch.manual_seed(0)
        policy = FADAPlannerIDMPolicy(config).eval()
        updater = OnlineSnapshotUpdater(policy, FADALoRAConfig(rank=2), sigma=0.0)
    playback = FADAPlaybackController(policy, device="cpu")
    _, snapshot = playback.act_projected_with_snapshot(torch.zeros(1, 66), torch.zeros(1, 3))
    collector = OnlineSnapshotCollector(config, window_size=1, gamma=0.05, action_scale=0.5)
    collector.begin(snapshot)
    following = snapshot.nominal_future[:, 0].clone()
    following[:, 6:35] -= 0.1
    collector.complete(following)
    outputs = []
    def capture(module, inputs, result):
        result.retain_grad()
        outputs.append(result)
    handle = policy.idm.register_forward_hook(capture)
    try:
        total, regression, anchor = updater.compute_loss(collector.get_window())
        torch.testing.assert_close(regression, torch.tensor(0.00145), atol=1e-7, rtol=1e-5)
        total.backward()
        torch.testing.assert_close(outputs[0].grad[:, 0], torch.full((1, 29), -0.01), atol=1e-6, rtol=1e-5)
        assert torch.count_nonzero(outputs[0].grad[:, 1:]).item() == 0
        assert anchor.item() == 0.0
    finally:
        handle.remove()


def test_anchor_retains_nonzero_initial_lora_weights(config):
    with torch.random.fork_rng():
        torch.manual_seed(0)
        policy = FADAPlannerIDMPolicy(config).eval()
        updater = OnlineSnapshotUpdater(policy, FADALoRAConfig(rank=2), sigma=0.1)
    name, parameter = next(iter(fada_adapter_named_parameters(policy)))
    initial = parameter.detach().clone()
    assert torch.count_nonzero(initial).item() > 0
    with torch.no_grad():
        parameter.add_(0.01)
    playback = FADAPlaybackController(policy, device="cpu")
    _, snapshot = playback.act_projected_with_snapshot(torch.zeros(1, 66), torch.zeros(1, 3))
    collector = OnlineSnapshotCollector(config, window_size=1, gamma=0.0, action_scale=0.5)
    collector.begin(snapshot)
    collector.complete(snapshot.nominal_future[:, 0])
    _, _, anchor = updater.compute_loss(collector.get_window())
    torch.testing.assert_close(anchor, torch.tensor(0.5 * 0.1 * parameter.numel() * 0.01**2), atol=1e-9, rtol=1e-5)
    gradient, = torch.autograd.grad(anchor, parameter)
    torch.testing.assert_close(gradient, 0.1 * (parameter.detach() - initial))
    torch.testing.assert_close(updater.theta_base[name], initial, atol=0, rtol=0)
