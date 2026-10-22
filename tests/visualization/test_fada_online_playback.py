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
from unilab.visualization.playback_distill_sessions import create_fada_playback_session
from unilab.visualization.playback_sessions import RslRlPlaybackConfig


OBS_DIM = 66
ACTION_DIM = 29
COMMAND_DIM = 3
HISTORY = 4
HORIZON = 3


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


def test_online_closes_next_step_labels_and_updates_only_on_full_window_tick(policy, checkpoint_file):
    env = _FakeEnv(np.random.default_rng(2))
    env.action_offset = 7.0  # Execution-side values must not replace the model snapshot.
    env.cfg.control_config.action_scale = [0.25] + [0.5] * (ACTION_DIM - 1)
    wrapped = _FakeWrappedEnv(env, env._rng)
    session, _, _ = _make_session(policy, env, wrapped, dict(_ONLINE), checkpoint_file)
    session.reset()
    controller = session.controller
    before = {name: p.detach().clone() for name, p in policy.named_parameters()}
    anchors = {name: value.clone() for name, value in controller.updater.theta_base.items()}
    session.step_once()
    first = controller.collector.get_window()[0]
    error = first.snapshot.nominal_joint_positions - torch.from_numpy(wrapped._obs[:, 6:35].copy())
    torch.testing.assert_close(first.joint_position_error, error)
    expected_target = first.snapshot.action_snapshot + 0.05 * error / torch.tensor(
        env.cfg.control_config.action_scale
    )
    torch.testing.assert_close(first.target_action, expected_target)
    torch.testing.assert_close(first.snapshot.action_snapshot, session.actions)
    assert not torch.allclose(first.target_action, torch.from_numpy(env.state.info["current_actions"]))
    label = first.target_action.clone()
    for _ in range(2):
        assert session.last_update_stats is None
        session.step_once()
    assert controller.collector.ready and session.last_update_stats is None
    session.step_once()  # Full T=3, first update at valid step 4 (interval 2).
    stats = session.last_update_stats
    assert stats is not None and stats["num_samples"] == 3
    assert stats["loss"] == pytest.approx(stats["regression_loss"] + stats["anchor_loss"])
    assert [sample.timestep for sample in controller.collector.get_window()] == [1, 2, 3]
    torch.testing.assert_close(first.target_action, label, atol=0, rtol=0)
    changed = {name for name, p in policy.named_parameters() if not torch.equal(p, before[name])}
    trainable = {name for name, p in policy.named_parameters() if p.requires_grad}
    assert changed and changed <= trainable
    assert all(name.startswith("idm.") for name in trainable)
    for name, value in controller.updater.theta_base.items():
        torch.testing.assert_close(value, anchors[name], atol=0, rtol=0)
    assert not policy.idm.training


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
    session.step_once()
    session.step_once()
    controller = session.controller
    old_episode = controller.collector.get_window()[-1].episode_id
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


@pytest.mark.parametrize("failure", ["scale", "wrapper", "io", "adapted"])
def test_failed_online_construction_releases_environment(policy, checkpoint_file, failure):
    env = _FakeEnv(np.random.default_rng(4))
    wrapped = _FakeWrappedEnv(env, env._rng)
    deps = _deps(policy, env, wrapped, checkpoint_file)
    if failure == "scale":
        env.cfg.control_config.action_scale = 0.0
    elif failure == "wrapper":
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
