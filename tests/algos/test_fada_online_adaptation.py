"""Behavior contracts for fixed-label streaming adaptation."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from unilab.algos.torch.distill.fada.adaptation import FADALoRAConfig, fada_adapter_named_parameters
from unilab.algos.torch.distill.fada.model import FADAArchitectureConfig, FADAPlannerIDMPolicy
from unilab.algos.torch.distill.fada.online_collector import (
    OnlineTrajectoryCollector,
)
from unilab.algos.torch.distill.fada.online_data import (
    FADAInferenceSnapshot,
    FADAOnlineSample,
    FADATrajectoryPair,
)
from unilab.algos.torch.distill.fada.online_updater import OnlineSnapshotUpdater, build_expert_samples
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


def test_expert_difference_preserves_snapshot_and_uses_nominal_minus_realized():
    class AffineExpert(torch.nn.Module):
        def __init__(self):
            super().__init__()
            gains = torch.arange(1, 30, dtype=torch.float32) / 10
            gains[1] = -2.0
            self.gains = torch.nn.Parameter(gains, requires_grad=False)

        def forward(self, observations, actions, future):
            bias = observations[:, -1, 6:35] + actions[:, -1]
            return future[:, :, 6:35] * self.gains + bias[:, None]

    expert = AffineExpert().eval()
    snapshot = _snapshot()
    realized = snapshot.nominal_future.clone()
    realized[:, 0, 6:35] -= 0.1
    realized[:, 1, 6:35] += 0.7
    changed = FADATrajectoryPair(snapshot, realized, 0, 4)
    matched = FADATrajectoryPair(snapshot, snapshot.nominal_future, 0, 5)
    samples = build_expert_samples(expert, (changed, matched), gamma=0.05)
    expected = torch.full((1, 29), 0.2) + 0.005 * expert.gains
    torch.testing.assert_close(samples[0].target_action, expected)
    torch.testing.assert_close(samples[1].target_action, snapshot.action_snapshot, atol=0, rtol=0)
    assert samples[0].trajectory is changed and samples[1].trajectory is matched
    assert all(not sample.target_action.requires_grad for sample in samples)
    assert expert.gains.grad is None


def test_trajectory_pair_owns_future_without_recopying_snapshot():
    snapshot = _snapshot()
    realized = torch.arange(132, dtype=torch.float32).reshape(1, 2, 66).requires_grad_()
    expected = realized.detach().clone()
    pair = FADATrajectoryPair(snapshot, realized, episode_id=2, timestep=7)
    with torch.no_grad():
        realized.fill_(999)
    torch.testing.assert_close(pair.realized_future, expected, atol=0, rtol=0)
    assert pair.snapshot is snapshot
    assert not pair.realized_future.requires_grad and pair.realized_future.grad_fn is None
    assert (pair.episode_id, pair.timestep) == (2, 7)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"realized_future": torch.zeros(1, 1, 66)}, "shape"),
        ({"realized_future": torch.zeros(1, 2, 65)}, "shape"),
        ({"realized_future": torch.zeros(1, 2, 66, dtype=torch.float64)}, "dtype and device"),
        ({"realized_future": torch.empty(1, 2, 66, device="meta")}, "dtype and device"),
        ({"realized_future": torch.full((1, 2, 66), float("nan"))}, "finite"),
        ({"episode_id": -1}, "episode_id"),
        ({"episode_id": True}, "episode_id"),
        ({"timestep": 1.5}, "timestep"),
        ({"timestep": -1}, "timestep"),
    ],
)
def test_trajectory_pair_rejects_incompatible_data(updates, message):
    fields = dict(snapshot=_snapshot(), realized_future=torch.zeros(1, 2, 66), episode_id=0, timestep=0)
    fields.update(updates)
    with pytest.raises(ValueError, match=message):
        FADATrajectoryPair(**fields)



def _trajectory_playback(config):
    with torch.random.fork_rng():
        torch.manual_seed(1)
        return FADAPlaybackController(FADAPlannerIDMPolicy(config).eval(), device="cpu")


def _trajectory_query(playback, index, command=0.4):
    _, snapshot = playback.act_projected_with_snapshot(
        torch.full((1, 66), float(index)), torch.tensor([[command, 0.0, 0.0]])
    )
    return snapshot


@pytest.mark.parametrize("horizon", [1, 6])
def test_trajectory_pairs_align_full_future_and_survive_clear(config, horizon):
    config = replace(config, prediction_horizon=horizon)
    collector = OnlineTrajectoryCollector(config, window_size=2)
    playback = _trajectory_playback(config)
    source = torch.zeros(1, 66)
    snapshots, completed = [], []
    for index in range(horizon + 3):
        snapshot = _trajectory_query(playback, index)
        snapshots.append(snapshot)
        collector.begin(snapshot)
        if index == horizon:
            collector.clear_window()  # Also retain the current act->step pending query.
            assert collector.get_window() == ()
        source.fill_(index + 1)
        pair = collector.complete(source)
        source.fill_(999)  # The environment may reuse this observation storage.
        if index + 1 < horizon:
            assert pair is None and collector.get_window() == ()
            continue
        origin = index + 1 - horizon
        assert pair is not None and pair.snapshot is snapshots[origin]
        assert (pair.episode_id, pair.timestep) == (0, origin)
        expected = torch.arange(origin + 1, origin + horizon + 1, dtype=torch.float32)
        torch.testing.assert_close(
            pair.realized_future, expected.reshape(1, horizon, 1).expand(1, horizon, 66)
        )
        completed.append(pair)
        expected_origins = list(range(max(1, origin - 1), origin + 1)) if origin else [0]
        assert [item.timestep for item in collector.get_window()] == expected_origins
        assert collector.ready == (len(expected_origins) == 2)
    for pair in completed:
        expected = torch.arange(pair.timestep + 1, pair.timestep + horizon + 1, dtype=torch.float32)
        torch.testing.assert_close(
            pair.realized_future, expected.reshape(1, horizon, 1).expand(1, horizon, 66)
        )


def test_trajectory_command_change_keeps_only_valid_completed_pairs(config):
    collector = OnlineTrajectoryCollector(config, window_size=3)
    playback = _trajectory_playback(config)
    for index in range(4):
        command = 0.4 if index < 2 else 0.8
        collector.begin(_trajectory_query(playback, index, command))
        pair = collector.complete(torch.full((1, 66), float(index + 1)))
        if index in (0, 2):
            assert pair is None
    pairs = collector.get_window()
    assert [(pair.episode_id, pair.timestep) for pair in pairs] == [(0, 0), (0, 2)]
    torch.testing.assert_close(pairs[0].realized_future[:, :, 0], torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(pairs[1].realized_future[:, :, 0], torch.tensor([[3.0, 4.0]]))
    torch.testing.assert_close(pairs[1].snapshot.command, torch.tensor([[0.8, 0.0, 0.0]]))


def test_trajectory_terminal_and_manual_reset_discard_all_inflight_data(config):
    collector = OnlineTrajectoryCollector(config, window_size=1)
    playback = _trajectory_playback(config)
    for index in range(2):
        collector.begin(_trajectory_query(playback, index))
        collector.complete(torch.full((1, 66), float(index + 1)))
    assert collector.get_window()[0].timestep == 0
    collector.begin(_trajectory_query(playback, 2))
    assert collector.complete(torch.full((1, 98), float("nan")), done=True) is None
    assert collector.get_window() == ()
    collector.clear_window()
    with pytest.raises(RuntimeError, match="requires reset"):
        collector.begin(_trajectory_query(playback, 3))
    collector.reset()
    for index in (10, 11):
        collector.begin(_trajectory_query(playback, index))
        pair = collector.complete(torch.full((1, 66), float(index + 1)))
    assert pair is not None and (pair.episode_id, pair.timestep) == (1, 0)
    torch.testing.assert_close(pair.realized_future[:, :, 0], torch.tensor([[11.0, 12.0]]))
    collector.begin(_trajectory_query(playback, 12))
    collector.reset()  # Explicit reset also drops the current unexecuted query.
    for index in (20, 21):
        collector.begin(_trajectory_query(playback, index))
        pair = collector.complete(torch.full((1, 66), float(index + 1)))
    assert pair is not None and (pair.episode_id, pair.timestep) == (2, 0)
    torch.testing.assert_close(pair.realized_future[:, :, 0], torch.tensor([[21.0, 22.0]]))


def test_trajectory_invalid_calls_do_not_consume_valid_steps(config):
    collector = OnlineTrajectoryCollector(config, window_size=1)
    playback = _trajectory_playback(config)
    with pytest.raises(RuntimeError, match="begin a snapshot"):
        collector.complete(torch.zeros(1, 66))
    first = _trajectory_query(playback, 0)
    collector.begin(first)
    with pytest.raises(RuntimeError, match="complete the pending"):
        collector.begin(first)
    with pytest.raises(ValueError, match="shape"):
        collector.complete(torch.zeros(1, 65))
    with pytest.raises(ValueError, match="finite"):
        collector.complete(torch.full((1, 66), float("nan")))
    assert collector.complete(torch.ones(1, 66)) is None
    with pytest.raises(RuntimeError, match="begin a snapshot"):
        collector.complete(torch.zeros(1, 66))
    invalid = replace(
        first, command=torch.tensor([[0.8, 0.0, 0.0]]), nominal_future=torch.zeros(1, 1, 66)
    )
    with pytest.raises(ValueError, match="nominal_future"):
        collector.begin(invalid)  # Must not flush origin 0 for this invalid command change.
    collector.begin(_trajectory_query(playback, 1))
    pair = collector.complete(torch.full((1, 66), 2.0))
    assert pair is not None and (pair.episode_id, pair.timestep) == (0, 0)
    torch.testing.assert_close(pair.realized_future[:, :, 0], torch.tensor([[1.0, 2.0]]))


def test_regression_uses_squared_action_norm_and_only_first_action(config):
    with torch.random.fork_rng():
        torch.manual_seed(0)
        policy = FADAPlannerIDMPolicy(config).eval()
        updater = OnlineSnapshotUpdater(policy, FADALoRAConfig(rank=2), sigma=0.0)
    playback = FADAPlaybackController(policy, device="cpu")
    _, snapshot = playback.act_projected_with_snapshot(torch.zeros(1, 66), torch.zeros(1, 3))
    pair = FADATrajectoryPair(snapshot, snapshot.nominal_future + 10.0, 0, 0)
    sample = FADAOnlineSample(pair, snapshot.action_snapshot + 0.01)
    outputs = []
    def capture(module, inputs, result):
        result.retain_grad()
        outputs.append(result)
    handle = policy.idm.register_forward_hook(capture)
    try:
        total, regression, anchor = updater.compute_loss((sample,))
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
    pair = FADATrajectoryPair(snapshot, snapshot.nominal_future, 0, 0)
    sample = FADAOnlineSample(pair, snapshot.action_snapshot)
    _, _, anchor = updater.compute_loss((sample,))
    torch.testing.assert_close(anchor, torch.tensor(0.5 * 0.1 * parameter.numel() * 0.01**2), atol=1e-9, rtol=1e-5)
    gradient, = torch.autograd.grad(anchor, parameter)
    torch.testing.assert_close(gradient, 0.1 * (parameter.detach() - initial))
    torch.testing.assert_close(updater.theta_base[name], initial, atol=0, rtol=0)
