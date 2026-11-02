"""Local-gradient and recorded-evidence smoke. No optimizer steps or physics runs."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from unilab.algos.torch.distill.fada.adaptation import FADALoRAConfig, fada_adapter_named_parameters
from unilab.algos.torch.distill.fada.model import FADAArchitectureConfig, FADAPlannerIDMPolicy
from unilab.algos.torch.distill.fada.observation import FADA_G1_STATE_OBSERVATION_CONTRACT
from unilab.algos.torch.distill.fada.online_controller import FADAOnlineAdaptationConfig
from unilab.algos.torch.distill.fada.online_data import FADAInferenceSnapshot, FADAOnlineSample, FADATrajectoryPair
from unilab.algos.torch.sttt.adapter_checkpoint import (
    bind_spatial_adapter_identity, load_spatial_adapter, save_spatial_adapter,
)
from unilab.algos.torch.sttt.checkpoint import save_world_model_checkpoint
from unilab.algos.torch.sttt.contract import PROPRIO_LAYOUT
from unilab.algos.torch.sttt.model import SpatialWorldModel
from unilab.algos.torch.sttt.online_collector import SpatialOnlineCollector
from unilab.algos.torch.sttt.online_controller import SpatialOnlineController
from unilab.algos.torch.sttt.online_data import OdometrySample, SpatialAdaptationConfig, SpatialOnlineSample
from unilab.algos.torch.sttt.pose_data import AlignedPoseInterval, ControlInterval, TimedPelvisPose
from unilab.algos.torch.sttt.reference import ReferenceTrajectory
from unilab.algos.torch.sttt.spatial_updater import SpatialOnlineUpdater


def _policy_and_snapshots(dtype=torch.float64):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        cfg = FADAArchitectureConfig(
            obs_dim=66, action_dim=29, command_dim=3,
            observation_contract=FADA_G1_STATE_OBSERVATION_CONTRACT,
            history_length=2, prediction_horizon=2, hidden_dim=8, num_heads=2,
            planner_layers=1, idm_encoder_layers=1, idm_decoder_layers=1,
            feedforward_dim=16, dropout=0.0,
        )
        policy = FADAPlannerIDMPolicy(cfg).to(dtype=dtype).eval()
        snapshots = []
        with torch.no_grad():
            for step in range(2):
                obs = torch.linspace(-0.2, 0.3, 132, dtype=dtype).reshape(1, 2, 66) + step * 0.01
                actions = torch.zeros(1, 2, 29, dtype=dtype)
                commands = torch.tensor([[0.4, 0.0, 0.0]], dtype=dtype)
                output = policy(obs, actions, commands)
                snapshots.append(FADAInferenceSnapshot(
                    observation_history=obs, action_history=actions, command=commands,
                    nominal_future=output.predicted_future, action_snapshot=output.action,
                ))
    return policy, tuple(snapshots)


def _world_model(dtype=torch.float64):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(29)
        model = SpatialWorldModel(64, 29, hist_len=2, horizon=2, hidden=(8,)).to(dtype=dtype)
        with torch.no_grad():
            model.net[-1].weight.normal_(0.0, 0.1)
    return model


def _pose(time, *, x=0.0, y=0.0, yaw=0.0, valid=True, frame_id="world", epoch=0):
    return OdometrySample(
        time, torch.tensor([x, y, 0.8]),
        torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]),
        frame_id=frame_id, tracking_epoch=epoch, valid=valid,
    )


def test_measured_endpoint_drives_lora_gradient_with_frozen_world_model():
    policy, snapshots = _policy_and_snapshots()
    base = copy.deepcopy(policy.idm).requires_grad_(False)
    updater = SpatialOnlineUpdater(
        policy, FADALoRAConfig(rank=2, alpha=2, dropout=0), _world_model(), sigma=1e-4,
    )
    actual = torch.stack([snap.action_snapshot[0] for snap in snapshots]).unsqueeze(0)
    sample = SpatialOnlineSample(
        proprio_history=torch.linspace(-0.1, 0.1, 128).double().reshape(1, 2, 64),
        snapshots=snapshots, actual_actions=actual,
        measured_endpoint=torch.tensor([[0.0, 0.1, 0.2]], dtype=torch.float64),
        reference_endpoint=torch.zeros(1, 3, dtype=torch.float64), episode_id=0, origin_step=0,
    )
    pair = FADATrajectoryPair(snapshots[0], snapshots[0].nominal_future, 0, 0)
    proprio = (FADAOnlineSample(pair, snapshots[0].action_snapshot),)
    frozen = {name: value.clone() for name, value in policy.state_dict().items()}
    phi = {name: value.clone() for name, value in updater.world_model.state_dict().items()}
    adapters = dict(fada_adapter_named_parameters(policy))
    params = tuple(adapters.values())
    total, spatial, regression, anchor = updater.compute_loss((sample,), proprio)
    assert spatial.item() == pytest.approx((0.1**2 + 0.2**2) / 3, abs=1e-10)
    assert updater.optimizer is updater.snapshot_updater.optimizer
    gradients = torch.autograd.grad(spatial, params, allow_unused=True)
    magnitudes = [0.0 if grad is None else float(grad.abs().max()) for grad in gradients]
    chosen = max(range(len(params)), key=lambda index: magnitudes[index])
    assert magnitudes[chosen] > 1e-9
    param = params[chosen]
    index = int(gradients[chosen].abs().argmax())
    analytic = float(gradients[chosen].flatten()[index])
    original = float(param.flatten()[index])
    epsilon = 1e-5
    try:
        with torch.no_grad():
            param.flatten()[index] = original + epsilon
        plus = float(updater.compute_loss((sample,), proprio)[1])
        with torch.no_grad():
            param.flatten()[index] = original - epsilon
        minus = float(updater.compute_loss((sample,), proprio)[1])
    finally:
        with torch.no_grad():
            param.flatten()[index] = original
    assert (plus - minus) / (2 * epsilon) == pytest.approx(analytic, rel=1e-3, abs=1e-8)
    changed = replace(sample, measured_endpoint=torch.tensor([[0.0, 0.1, 0.4]], dtype=torch.float64))
    other = torch.autograd.grad(updater.compute_loss((changed,), proprio)[1], params, allow_unused=True)
    assert not torch.allclose(gradients[chosen], other[chosen], atol=1e-10, rtol=1e-4)
    inner = updater.snapshot_updater.compute_loss(proprio)
    combined = updater.compute_loss((), proprio)
    torch.testing.assert_close(combined[0], inner[0])
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, frozen[name])
    for name, value in updater.world_model.state_dict().items():
        torch.testing.assert_close(value, phi[name])
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in updater.world_model.parameters())
    assert all(parameter.grad is None for parameter in policy.parameters() if not parameter.requires_grad)
    assert all(parameter.grad is None for parameter in base.parameters())
    with pytest.raises(ValueError):
        FADAOnlineSample(pair, torch.full_like(snapshots[0].action_snapshot, float("nan")))
    assert not updater.optimizer.state
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, frozen[name])
    print(f"Spatial LoRA derivative: autograd={analytic:.8g}, finite_difference={(plus-minus)/(2*epsilon):.8g}; no optimizer step.")


def test_reference_and_collector_preserve_drift_and_drop_broken_windows():
    _, snapshots = _policy_and_snapshots(torch.float32)
    reference = ReferenceTrajectory(0.1)
    reference.reset(_pose(0.0))
    for _ in range(10):
        reference.advance(torch.tensor([1.0, 0.0, 1.0]))
    torch.testing.assert_close(
        reference.current_pose,
        torch.tensor([math.sin(1.0), 1 - math.cos(1.0), 1.0], dtype=reference.current_pose.dtype),
        atol=1e-6, rtol=1e-6,
    )
    collector = SpatialOnlineCollector(history_length=2, horizon=2, ctrl_dt=0.1, window_size=1,
                                       max_alignment_age_s=1.0)
    command = np.array([0.4, 0.0, 0.0])

    def record(step, *, x_offset=0.0, epoch=0):
        start_ns, end_ns = step * 100_000_000, (step + 1) * 100_000_000
        control = ControlInterval(0, step, start_ns, end_ns,
                                  np.full(29, float(step)), command.copy())
        collector.record_step(snapshots[step % 2], torch.full((64,), float(step)), control)
        poses = []
        for time_ns in (start_ns, end_ns):
            transform = np.eye(4)
            transform[:3, 3] = [x_offset + time_ns / 1e9 * 0.4, time_ns / 1e9 * 0.2, 0.8]
            poses.append(TimedPelvisPose(0, epoch, time_ns, transform, True, (time_ns, time_ns), "world"))
        return AlignedPoseInterval(control, poses[0], poses[1])

    delayed = tuple(record(step) for step in range(5))
    assert not collector.ready
    collector.accept_aligned(delayed)
    sample = collector.get_window()[0]
    assert sample.origin_step == 3
    torch.testing.assert_close(sample.proprio_history[0, :, 0], torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(sample.actual_actions[0, :, 0], torch.tensor([3.0, 4.0]))
    assert sample.measured_endpoint[0, 1] > 0
    assert sample.reference_endpoint[0, 1] < 0
    collector.invalidate_tracking()
    assert collector.get_window() == ()
    collector.accept_aligned(tuple(record(step, x_offset=10.0, epoch=1) for step in range(6, 9)))
    assert collector.ready
    collector.reset()
    assert collector.get_window() == ()


class _RecordedSignals:
    """Known recorded signals, not a physics simulator or learned policy substitute."""
    def __init__(self):
        self.cfg = SimpleNamespace(ctrl_dt=0.1, control_config=SimpleNamespace(action_scale=0.25))
        self.action_space = SimpleNamespace(shape=(29,))
        self.default_angles = np.zeros(29, np.float32)
        self.state = SimpleNamespace(info={"commands": np.array([[0.4, 0.0, 0.0]], np.float32)})
        self.step_counter = 0

    def get_actuated_joint_names(self):
        return tuple(f"joint_{index}" for index in range(29))

    def get_sim_time_ns(self):
        return round(self.step_counter * self.cfg.ctrl_dt * 1e9)

    def get_gyro(self):
        return np.zeros((1, 3), np.float32)

    def get_upvector(self):
        return np.array([[0.0, 0.0, 1.0]], np.float32)

    def get_dof_pos(self):
        return np.zeros((1, 29), np.float32)

    def get_dof_vel(self):
        return np.zeros((1, 29), np.float32)

    def get_base_pos(self):
        time = self.step_counter * self.cfg.ctrl_dt
        return np.array([[time * 0.4, time * 0.2, 0.8]], np.float32)

    def get_base_quat(self):
        return np.array([[1.0, 0.0, 0.0, 0.0]], np.float32)


def test_controller_recorded_lifecycle_and_adapter_roundtrip(tmp_path):
    policy, _ = _policy_and_snapshots(torch.float32)
    signals = _RecordedSignals()
    world = _world_model(torch.float32)
    world_path = tmp_path / "world.pt"
    contract = {
        "obs_dim": 64, "act_dim": 29, "history_length": 2, "horizon": 2, "ctrl_dt": 0.1,
        "proprio_layout": PROPRIO_LAYOUT, "joint_names": list(signals.get_actuated_joint_names()),
        "action_scale": [0.25], "default_joint_angles": [0.0] * 29,
        "action_representation": "env_input", "quaternion_order": "wxyz", "coordinate_frame": "reference_heading",
    }
    save_world_model_checkpoint(world_path, world, data_contract=contract, data_identity={}, split={}, resolved_config={})
    source = tmp_path / "policy.pt"
    torch.save({"architecture": asdict(policy.config), "state_dict": policy.state_dict()}, source)
    online = FADAOnlineAdaptationConfig(window_size=1, warmup_steps=100, update_interval=1, lora_rank=2, lora_alpha=2, lora_dropout=0)
    spatial = SpatialAdaptationConfig(enabled=True, world_model_checkpoint=str(world_path))
    controller = SpatialOnlineController(
        policy, FADALoRAConfig(rank=2, alpha=2, dropout=0), env=signals,
        online=online, spatial=spatial, root_dir=tmp_path,
    )
    bind_spatial_adapter_identity(controller, source)
    observation = torch.zeros(1, 98)
    for _ in range(4):
        action = controller.act(observation, signals.state.info["commands"])
        signals.step_counter += 1
        assert controller.on_env_step(executed_action=action.detach().numpy(), next_observation=observation) is None
    assert controller.spatial_collector.ready and controller.collector.ready
    optimizer = controller.updater.optimizer
    parameters = {name: value.clone() for name, value in fada_adapter_named_parameters(controller.policy)}
    artifact = save_spatial_adapter(controller, tmp_path / "adapter.pt")
    with torch.no_grad():
        next(iter(dict(fada_adapter_named_parameters(policy)).values())).add_(0.01)
    load_spatial_adapter(controller, artifact)
    for name, value in fada_adapter_named_parameters(policy):
        torch.testing.assert_close(value, parameters[name])
    assert controller.updater.optimizer is optimizer
    assert not controller.spatial_collector.ready and not controller.collector.ready
    original_identity = controller.adapter_identity
    controller.adapter_identity = {**original_identity, "world_model_sha256": "wrong"}
    with pytest.raises(ValueError):
        load_spatial_adapter(controller, artifact)
    controller.adapter_identity = original_identity
    action = controller.act(observation, signals.state.info["commands"])
    signals.step_counter += 1
    controller.on_env_step(executed_action=action.detach().numpy(), next_observation=observation, done=np.array([True]))
    assert not controller.spatial_collector.ready and not controller.collector.ready
    controller.reset()
    assert controller.updater.optimizer is optimizer
    assert all(parameter.grad is None for parameter in controller.base_idm.parameters())
    print("Recorded controller lifecycle and adapter roundtrip passed; no optimizer steps, training, or physics simulation.")
