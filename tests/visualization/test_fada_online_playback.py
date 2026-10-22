"""Behavior checks for the official fixed-label FADA streaming session."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from unilab.algos.torch.distill.fada.adaptation import (
    FADALoRAConfig,
    fada_adapter_parameters,
)
from unilab.algos.torch.distill.fada.checkpoint import LoadedFADAPlannerIDMPolicy
from unilab.algos.torch.distill.fada.model import (
    FADAArchitectureConfig,
    FADAPlannerIDMPolicy,
)
from unilab.algos.torch.distill.fada.online_data import FADATrajectoryPair
from unilab.algos.torch.distill.fada.online_updater import build_expert_samples
from unilab.visualization.playback_distill_sessions import create_fada_playback_session
from unilab.visualization.playback_sessions import RslRlPlaybackConfig


OBS_DIM = 66
ACTION_DIM = 29
COMMAND_DIM = 3
HISTORY = 4
HORIZON = 6


@pytest.fixture
def arch_config() -> FADAArchitectureConfig:
    return FADAArchitectureConfig(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        command_dim=COMMAND_DIM,
        observation_contract="g1_fada_state_v2",
        history_length=HISTORY,
        prediction_horizon=HORIZON,
        hidden_dim=32,
        num_heads=2,
        planner_layers=1,
        idm_encoder_layers=1,
        idm_decoder_layers=1,
        feedforward_dim=64,
        dropout=0.0,
    )


@pytest.fixture
def policy(arch_config: FADAArchitectureConfig) -> FADAPlannerIDMPolicy:
    with torch.random.fork_rng():
        torch.manual_seed(0)
        return FADAPlannerIDMPolicy(arch_config).eval()


def _raw_obs(rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((1, 98), dtype=np.float32)


def _command() -> np.ndarray:
    return np.array([[0.4, 0.0, 0.0]], dtype=np.float32)


class _FakeActionSpace:
    shape = (ACTION_DIM,)
    low = -np.ones(ACTION_DIM, dtype=np.float32)
    high = np.ones(ACTION_DIM, dtype=np.float32)


class _FakeEnv:
    """Single-env fake honoring the FADA playback env contract."""

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self.num_envs = 1
        self.action_space = _FakeActionSpace()
        self.cfg = SimpleNamespace(
            ctrl_dt=0.02, control_config=SimpleNamespace(action_scale=0.5)
        )
        self._done = np.zeros(1, dtype=bool)
        self.closed = False
        self.action_offset = 0.0
        self.state = SimpleNamespace(
            info={
                "commands": _command(),
                "current_actions": np.zeros((1, ACTION_DIM), dtype=np.float32),
            }
        )

    def set_autoreset(self, enabled: bool) -> None:
        self.autoreset = bool(enabled)

    def set_next_done(self, done: bool) -> None:
        self._done = np.array([done], dtype=bool)

    def close(self):
        self.closed = True


class _FakeWrappedEnv:
    """Raw wrapped-env contract consumed by FADAPlaybackSession."""

    def __init__(self, env: _FakeEnv, rng: np.random.Generator) -> None:
        self.env = env
        self._rng = rng
        self._obs = _raw_obs(rng)
        self.num_obs = 98

    def reset(self):
        self._obs = _raw_obs(self._rng)
        self.env.set_next_done(False)
        self.env.state.info["commands"] = _command()
        self.env.state.info["current_actions"] = np.zeros((1, ACTION_DIM), np.float32)
        return {"actor": torch.from_numpy(self._obs)}, {}

    def step(self, actions):
        np.copyto(
            self.env.state.info["current_actions"],
            actions.detach().cpu().numpy() + self.env.action_offset,
        )
        np.copyto(self._obs, _raw_obs(self._rng))
        reward = torch.zeros(1)
        return {"actor": torch.from_numpy(self._obs)}, reward, torch.from_numpy(self.env._done.copy()), {}

    def get_observations(self):
        return {"actor": torch.from_numpy(self._obs)}

    def close(self):
        self.env.close()


@pytest.fixture
def checkpoint_file(tmp_path: Path) -> Path:
    path = tmp_path / "fada.pt"
    path.write_bytes(b"checkpoint")
    return path


def _deps(policy: FADAPlannerIDMPolicy, env: _FakeEnv, wrapped: _FakeWrappedEnv, ckpt: Path):
    return {
        "resolve_checkpoint": lambda playback_cfg, cfg, root_dir: ckpt,
        "load_fada_policy": lambda checkpoint, device: LoadedFADAPlannerIDMPolicy(
            policy=policy, checkpoint={"completed_steps": 1}
        ),
        "create_env": lambda *a, **k: env,
        "wrapper_cls": lambda env, device, policy_obs_mode: wrapped,
        "build_env_cfg_override": lambda cfg: {},
    }


def _playback_cfg(ckpt: Path) -> RslRlPlaybackConfig:
    return RslRlPlaybackConfig(
        task="t",
        load_run="-1",
        checkpoint=None,
        checkpoint_path=str(ckpt),
        action_mode="policy",
        policy_obs_mode="actor",
        algo_log_name="distill",
        log_root=None,
        num_envs=1,
    )


def _cfg(online: dict | None):
    from omegaconf import OmegaConf

    node = {"training": {"task_name": "T"}}
    if online is not None:
        node["online_adaptation"] = online
    return OmegaConf.create(node)


def _make_session(policy, env, wrapped, online, ckpt: Path):
    return create_fada_playback_session(
        playback_cfg=_playback_cfg(ckpt),
        cfg=_cfg(online),
        root_dir="/tmp",
        device="cpu",
        deps=_deps(policy, env, wrapped, ckpt),
        log=lambda msg: None,
    )


_ONLINE = {
    "enabled": True,
    "window_size": 3,
    "warmup_steps": 0,
    "gamma": 0.05,
    "sigma": 1e-4,
    "update_interval": 2,
    "learning_rate": 1e-3,
    "max_grad_norm": 1.0,
    "lora_rank": 2,
    "lora_alpha": 4.0,
    "lora_dropout": 0.0,
}


def test_disabled_preserves_checkpoint_actions_and_parameters(policy, checkpoint_file):
    env = _FakeEnv(np.random.default_rng(0))
    wrapped = _FakeWrappedEnv(env, env._rng)
    session, _, _ = _make_session(policy, env, wrapped, None, checkpoint_file)
    session.reset()
    projected = session.controller.project_observation(session.obs)
    with torch.no_grad():
        expected = policy(
            projected[:, None].repeat(1, HISTORY, 1),
            torch.zeros(1, HISTORY, ACTION_DIM),
            torch.from_numpy(_command()),
        ).action
    before = {name: p.detach().clone() for name, p in policy.named_parameters()}
    session.step_once()
    torch.testing.assert_close(session.actions, expected)
    for _ in range(3):
        session.step_once()
        assert session.last_update_stats is None
    for name, parameter in policy.named_parameters():
        torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)
    assert list(fada_adapter_parameters(policy)) == []


def test_frozen_base_survives_real_online_update_and_reset(policy, checkpoint_file):
    env = _FakeEnv(np.random.default_rng(6))
    wrapped = _FakeWrappedEnv(env, env._rng)
    session, _, _ = _make_session(policy, env, wrapped, dict(_ONLINE), checkpoint_file)
    session.reset()
    controller = session.controller
    base = controller.base_idm
    assert all(not module.training for module in base.modules())
    assert all(not parameter.requires_grad for parameter in base.parameters())
    base_tensors = tuple(base.parameters()) + tuple(base.buffers())
    online_tensors = tuple(policy.idm.parameters()) + tuple(policy.idm.buffers())
    assert {
        tensor.untyped_storage().data_ptr() for tensor in base_tensors if tensor.numel()
    }.isdisjoint({
        tensor.untyped_storage().data_ptr() for tensor in online_tensors if tensor.numel()
    })
    optimizer_ids = {
        id(parameter)
        for group in controller.updater.optimizer.param_groups
        for parameter in group["params"]
    }
    assert all(id(parameter) not in optimizer_ids for parameter in base.parameters())
    base_state = {name: value.detach().clone() for name, value in base.state_dict().items()}
    adapters = tuple(fada_adapter_parameters(policy))
    before = tuple(parameter.detach().clone() for parameter in adapters)
    realized = []
    for _ in range(HORIZON):
        session.step_once()
        realized.append(controller.project_observation(session.obs).clone())
        assert session.last_update_stats is None
    pair = controller.collector.get_window()[0]
    snapshot = pair.snapshot
    inputs = (snapshot.observation_history, snapshot.action_history, snapshot.nominal_future)
    with torch.no_grad():
        expected = base(*inputs).clone()
        torch.testing.assert_close(policy.idm(*inputs), expected)
    torch.testing.assert_close(pair.realized_future, torch.stack(realized, dim=1))
    for _ in range(2):  # K=6, T=3, interval=2: first full-window tick is step 8.
        session.step_once()
    assert session.last_update_stats["num_samples"] == 3
    assert any(not torch.equal(parameter, old) for parameter, old in zip(adapters, before))
    with torch.no_grad():
        torch.testing.assert_close(base(*inputs), expected, atol=0, rtol=0)
    for name, value in base.state_dict().items():
        torch.testing.assert_close(value, base_state[name], atol=0, rtol=0)
    assert all(parameter.grad is None for parameter in base.parameters())
    optimizer = controller.updater.optimizer
    anchors = {name: value.clone() for name, value in controller.updater.theta_base.items()}
    learned = tuple(parameter.detach().clone() for parameter in adapters)
    adam_steps = [state["step"].clone() for state in optimizer.state.values()]
    session.reset()
    assert controller.base_idm is base and controller.updater.optimizer is optimizer
    for parameter, old in zip(adapters, learned):
        torch.testing.assert_close(parameter, old, atol=0, rtol=0)
    for state, old in zip(optimizer.state.values(), adam_steps):
        torch.testing.assert_close(state["step"], old, atol=0, rtol=0)
    for name, value in controller.updater.theta_base.items():
        torch.testing.assert_close(value, anchors[name], atol=0, rtol=0)
    with torch.no_grad():
        torch.testing.assert_close(base(*inputs), expected, atol=0, rtol=0)


def test_online_uses_full_realized_future_and_keeps_expert_labels_static(policy, checkpoint_file):
    env = _FakeEnv(np.random.default_rng(2))
    env.action_offset = 7.0  # Execution-side action echoes are not the regression target.
    wrapped = _FakeWrappedEnv(env, env._rng)
    session, _, _ = _make_session(policy, env, wrapped, dict(_ONLINE), checkpoint_file)
    session.reset()
    controller = session.controller
    before = {name: p.detach().clone() for name, p in policy.named_parameters()}
    states, actions = [], []
    for _ in range(HORIZON):
        session.step_once()
        assert session.last_update_stats is None
        states.append(controller.project_observation(session.obs).clone())
        actions.append(session.actions.clone())
    first = controller.collector.get_window()[0]
    torch.testing.assert_close(first.realized_future, torch.stack(states, dim=1))
    torch.testing.assert_close(first.snapshot.action_snapshot, actions[0])
    samples = build_expert_samples(controller.base_idm, (first,), gamma=controller.gamma)
    label = samples[0].target_action.clone()
    matched = FADATrajectoryPair(
        first.snapshot, first.snapshot.nominal_future, first.episode_id, first.timestep
    )
    matched_sample, = build_expert_samples(controller.base_idm, (matched,), gamma=controller.gamma)
    torch.testing.assert_close(matched_sample.target_action, actions[0], atol=0, rtol=0)
    assert not torch.allclose(label, actions[0] + env.action_offset)
    session.step_once()
    assert [pair.timestep for pair in controller.collector.get_window()] == [0, 1]
    assert not controller.collector.ready and session.last_update_stats is None
    session.step_once()
    stats = session.last_update_stats
    assert stats["num_samples"] == 3
    assert stats["loss"] == pytest.approx(stats["regression_loss"] + stats["anchor_loss"])
    assert controller.collector.get_window() == ()
    controller.updater.compute_loss(samples)
    torch.testing.assert_close(samples[0].target_action, label, atol=0, rtol=0)
    rebuilt, = build_expert_samples(controller.base_idm, (first,), gamma=controller.gamma)
    torch.testing.assert_close(rebuilt.target_action, label, atol=0, rtol=0)
    assert not samples[0].target_action.requires_grad
    changed = {name for name, p in policy.named_parameters() if not torch.equal(p, before[name])}
    trainable = {name for name, p in policy.named_parameters() if p.requires_grad}
    assert changed and changed <= trainable
    assert all(name.startswith("idm.") for name in trainable)
    assert not policy.idm.training


@pytest.mark.parametrize("warmup", [40, 17])
def test_fresh_batches_wait_for_warmup_and_preserve_learning_across_consumption(
    policy, checkpoint_file, warmup
):
    env = _FakeEnv(np.random.default_rng(4))
    wrapped = _FakeWrappedEnv(env, env._rng)
    online = dict(_ONLINE, window_size=10, warmup_steps=warmup, update_interval=10)
    session, _, _ = _make_session(policy, env, wrapped, online, checkpoint_file)
    session.reset()
    controller = session.controller
    optimizer = controller.updater.optimizer
    anchors = {name: value.clone() for name, value in controller.updater.theta_base.items()}
    first_update = ((warmup + 9) // 10) * 10
    expected_ticks = [first_update + offset for offset in (0, 10, 20)]
    ticks = []
    episode = None
    actions = []
    for step in range(1, expected_ticks[-1] + 1):
        session.step_once()
        actions.append(session.actions.detach().clone())
        samples = controller.collector.get_window()
        if step in expected_ticks:
            ticks.append(step)
            assert session.last_update_stats["num_samples"] == 10
            assert samples == ()
            assert controller.updater.optimizer is optimizer
            assert {int(state["step"].item()) for state in optimizer.state.values()} == {len(ticks)}
        else:
            assert session.last_update_stats is None
            if step == first_update - 1:
                last_origin = step - HORIZON
                assert [sample.timestep for sample in samples] == list(range(last_origin - 9, last_origin + 1))
                episode = samples[-1].episode_id
            if step - 1 in expected_ticks:
                assert len(samples) == 1
                origin = step - HORIZON
                assert (samples[0].episode_id, samples[0].timestep) == (episode, origin)
                torch.testing.assert_close(samples[0].snapshot.action_history[:, -1], actions[origin - 1])
        for name, value in controller.updater.theta_base.items():
            torch.testing.assert_close(value, anchors[name], atol=0, rtol=0)
    assert ticks == expected_ticks
    learned = {name: p.detach().clone() for name, p in policy.named_parameters()}
    session.reset()
    for step in range(1, first_update):
        session.step_once()
        assert session.last_update_stats is None
        assert {int(state["step"].item()) for state in optimizer.state.values()} == {3}
        if step == HORIZON:
            sample = controller.collector.get_window()[0]
            assert (sample.episode_id, sample.timestep) == (episode + 1, 0)
            torch.testing.assert_close(
                sample.snapshot.action_history, torch.zeros_like(sample.snapshot.action_history)
            )
    for name, parameter in policy.named_parameters():
        torch.testing.assert_close(parameter, learned[name], atol=0, rtol=0)
    session.step_once()
    assert session.last_update_stats["num_samples"] == 10
    assert controller.collector.get_window() == ()
    assert {int(state["step"].item()) for state in optimizer.state.values()} == {4}


def test_failed_real_update_does_not_consume_batch(policy, checkpoint_file):
    env = _FakeEnv(np.random.default_rng(5))
    wrapped = _FakeWrappedEnv(env, env._rng)
    session, _, _ = _make_session(policy, env, wrapped, dict(_ONLINE), checkpoint_file)
    session.reset()
    for _ in range(7):
        session.step_once()
    controller = session.controller
    # Corrupt stored evidence; the real expert forward must fail without consuming the batch.
    controller.collector.get_window()[-1].realized_future.fill_(float("nan"))
    before = {name: p.detach().clone() for name, p in policy.named_parameters()}
    with pytest.raises(ValueError, match="finite"):
        session.step_once()
    assert [sample.timestep for sample in controller.collector.get_window()] == [0, 1, 2]
    assert controller.updater.optimizer.state == {}
    for name, parameter in policy.named_parameters():
        torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)


@pytest.mark.parametrize("autoreset", [True, False])
def test_terminal_and_manual_reset_keep_learning_state_but_isolate_samples(
    policy, checkpoint_file, autoreset
):
    import copy

    env = _FakeEnv(np.random.default_rng(3))
    wrapped = _FakeWrappedEnv(env, env._rng)
    online = dict(_ONLINE, window_size=2, update_interval=1)
    session, _, _ = _make_session(policy, env, wrapped, online, checkpoint_file)
    session.reset()
    session.set_autoreset(autoreset)
    for _ in range(HORIZON):
        session.step_once()
    old_episode = session.controller.collector.get_window()[-1].episode_id
    session.step_once()
    controller = session.controller
    params = {name: p.detach().clone() for name, p in policy.named_parameters()}
    anchors = {name: value.clone() for name, value in controller.updater.theta_base.items()}
    optimizer = controller.updater.optimizer
    adam_state = copy.deepcopy(optimizer.state_dict()["state"])
    assert adam_state
    env.set_next_done(True)
    session.step_once()
    assert session.last_update_stats is None
    assert controller.collector.get_window() == ()
    if autoreset:
        env.set_next_done(False)
    else:
        session.step_once()  # Physics still advances; collection must remain suspended.
        assert controller.collector.get_window() == ()
        assert session.last_update_stats is None
        session.reset()
    for _ in range(HORIZON):
        session.step_once()
    sample = controller.collector.get_window()[0]
    assert (sample.episode_id, sample.timestep) == (old_episode + 1, 0)
    torch.testing.assert_close(
        sample.snapshot.action_history, torch.zeros_like(sample.snapshot.action_history)
    )
    torch.testing.assert_close(
        sample.snapshot.observation_history,
        sample.snapshot.observation_history[:, -1:].repeat(1, HISTORY, 1),
    )
    assert controller.updater.optimizer is optimizer
    for key, state in optimizer.state_dict()["state"].items():
        for field, value in state.items():
            torch.testing.assert_close(value, adam_state[key][field], atol=0, rtol=0)
    for name, value in controller.updater.theta_base.items():
        torch.testing.assert_close(value, anchors[name], atol=0, rtol=0)
    for name, parameter in policy.named_parameters():
        torch.testing.assert_close(parameter, params[name], atol=0, rtol=0)


@pytest.mark.parametrize("failure", ["wrapper", "io", "adapted"])
def test_failed_online_construction_releases_environment(policy, checkpoint_file, failure):
    env = _FakeEnv(np.random.default_rng(4))
    wrapped = _FakeWrappedEnv(env, env._rng)
    deps = _deps(policy, env, wrapped, checkpoint_file)
    if failure == "wrapper":
        def fail_wrapper(*args, **kwargs):
            raise ValueError("wrapper construction failed")
        deps["wrapper_cls"] = fail_wrapper
    elif failure == "io":
        wrapped.num_obs = 97
    else:
        from unilab.algos.torch.distill.fada.adaptation import inject_fada_idm_lora
        inject_fada_idm_lora(policy, FADALoRAConfig(rank=2, alpha=4.0))
    with pytest.raises(ValueError):
        create_fada_playback_session(
            playback_cfg=_playback_cfg(checkpoint_file), cfg=_cfg(dict(_ONLINE)),
            root_dir="/tmp", device="cpu", deps=deps, log=lambda msg: None,
        )
    assert env.closed


@pytest.mark.parametrize("failure", ["multiple_envs", "zero_actions", "old_config"])
def test_invalid_online_request_fails_before_environment_creation(policy, checkpoint_file, failure):
    env = _FakeEnv(np.random.default_rng(5))
    wrapped = _FakeWrappedEnv(env, env._rng)
    playback = _playback_cfg(checkpoint_file)
    online = dict(_ONLINE)
    if failure == "multiple_envs":
        object.__setattr__(playback, "num_envs", 2)
    elif failure == "zero_actions":
        object.__setattr__(playback, "action_mode", "zero")
    else:
        online["lambda_forget"] = 0.9
    deps = _deps(policy, env, wrapped, checkpoint_file)
    def forbidden_env(*args, **kwargs):
        pytest.fail("invalid online request must not create an environment")
    deps["create_env"] = forbidden_env
    with pytest.raises((ValueError, TypeError)):
        create_fada_playback_session(
            playback_cfg=playback, cfg=_cfg(online), root_dir="/tmp",
            device="cpu", deps=deps, log=lambda msg: None,
        )
