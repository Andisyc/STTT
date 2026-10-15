"""Tests for FADA online adaptation chain."""

import numpy as np
import pytest
import torch

from unilab.algos.torch.distill.fada.adaptation import FADALoRAConfig, fada_adapter_parameters
from unilab.algos.torch.distill.fada.model import FADAArchitectureConfig, FADAPlannerIDMPolicy
from unilab.algos.torch.distill.fada.online_collector import OnlineTrajectoryCollector
from unilab.algos.torch.distill.fada.online_controller import OnlineAdaptationController
from unilab.algos.torch.distill.fada.online_updater import OnlineAdaptationUpdater


@pytest.fixture
def config():
    return FADAArchitectureConfig(
        obs_dim=10,
        action_dim=5,
        command_dim=3,
        history_length=4,
        prediction_horizon=3,
        hidden_dim=32,
        num_heads=2,
        planner_layers=1,
        idm_encoder_layers=1,
        idm_decoder_layers=1,
        feedforward_dim=64,
        dropout=0.0,
    )


@pytest.fixture
def policy(config):
    return FADAPlannerIDMPolicy(config)


@pytest.fixture
def lora_config():
    return FADALoRAConfig(rank=2, alpha=4.0, dropout=0.0)


def test_online_controller_creation(policy, lora_config):
    controller = OnlineAdaptationController(policy, lora_config, device="cpu")
    assert controller.playback is not None
    assert controller.collector is not None
    assert controller.updater is not None


def test_inference_shape(policy, lora_config, config):
    controller = OnlineAdaptationController(policy, lora_config, device="cpu")
    obs = np.random.randn(config.obs_dim).astype(np.float32)
    cmd = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    action = controller.act(obs, cmd)
    assert action.shape == (1, config.action_dim)


def test_window_collection(policy, lora_config, config):
    controller = OnlineAdaptationController(policy, lora_config, device="cpu")
    obs = np.random.randn(config.obs_dim).astype(np.float32)
    cmd = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    
    # Fill buffer
    for _ in range(config.history_length + config.prediction_horizon):
        action = controller.act(obs, cmd)
        exec_action = np.asarray(action).squeeze() + np.random.randn(config.action_dim).astype(np.float32) * 0.1
        next_obs = np.random.randn(config.obs_dim).astype(np.float32)
        controller.on_env_step(executed_action=exec_action, next_observation=next_obs)
        obs = next_obs
    
    # Buffer should have windows
    assert len(controller.collector._records) == config.history_length + config.prediction_horizon - 1


def test_exponential_weighting(policy, lora_config):
    updater = OnlineAdaptationUpdater(policy, lora_config, lambda_forget=0.9, lr=1e-3)
    
    # Add windows with different losses
    from unilab.algos.torch.distill.fada.windows import FADACausalWindow
    for i in range(3):
        window = FADACausalWindow(
            observation_history=np.random.randn(4, 10).astype(np.float32),
            action_history=np.random.randn(4, 5).astype(np.float32),
            command=np.array([0.5, 0.0, 0.0], dtype=np.float32),
            realized_future=np.random.randn(3, 10).astype(np.float32),
            executed_action_chunk=np.random.randn(3, 5).astype(np.float32),
            episode_id=0,
            start_timestep=i * 10,
        )
        updater.append_window(window)
    
    # Compute weighted loss
    loss = updater._exponential_weighted_loss()
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar


def test_lora_updates(policy, lora_config):
    updater = OnlineAdaptationUpdater(policy, lora_config, lambda_forget=0.9, lr=1e-3)
    
    # Get initial LoRA params
    initial_params = [p.clone() for p in fada_adapter_parameters(updater.policy)]
    
    # Add window and update
    from unilab.algos.torch.distill.fada.windows import FADACausalWindow
    window = FADACausalWindow(
        observation_history=np.random.randn(4, 10).astype(np.float32),
        action_history=np.random.randn(4, 5).astype(np.float32),
        command=np.array([0.5, 0.0, 0.0], dtype=np.float32),
        realized_future=np.random.randn(3, 10).astype(np.float32),
        executed_action_chunk=np.random.randn(3, 5).astype(np.float32),
        episode_id=0,
        start_timestep=0,
    )
    updater.append_window(window)
    stats = updater.update()
    
    assert stats is not None
    assert "loss" in stats
    
    # Check at least one LoRA param changed (B is zero-init, may not change on first update)
    updated_params = list(fada_adapter_parameters(updater.policy))
    any_changed = any(
        not torch.allclose(initial, updated, atol=1e-8, rtol=1e-8)
        for initial, updated in zip(initial_params, updated_params)
    )
    assert any_changed, "No LoRA parameters updated"


def test_planner_frozen(policy, lora_config):
    updater = OnlineAdaptationUpdater(policy, lora_config, lambda_forget=0.9, lr=1e-3)
    
    # Get initial planner params
    initial_planner = [p.clone() for p in updater.policy.planner.parameters()]
    
    # Add window and update
    from unilab.algos.torch.distill.fada.windows import FADACausalWindow
    window = FADACausalWindow(
        observation_history=np.random.randn(4, 10).astype(np.float32),
        action_history=np.random.randn(4, 5).astype(np.float32),
        command=np.array([0.5, 0.0, 0.0], dtype=np.float32),
        realized_future=np.random.randn(3, 10).astype(np.float32),
        executed_action_chunk=np.random.randn(3, 5).astype(np.float32),
        episode_id=0,
        start_timestep=0,
    )
    updater.append_window(window)
    updater.update()
    
    # Check planner params unchanged
    updated_planner = list(updater.policy.planner.parameters())
    for initial, updated in zip(initial_planner, updated_planner):
        assert torch.allclose(initial, updated)
