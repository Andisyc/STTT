"""Focused integration tests for FADA online playback controller selection."""

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
from unilab.algos.torch.distill.fada.online_controller import OnlineAdaptationController
from unilab.algos.torch.distill.fada.playback import FADAPlaybackController
from unilab.visualization.playback_distill_sessions import create_fada_playback_session
from unilab.visualization.playback_sessions import FADAPlaybackSession, RslRlPlaybackConfig


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
        self.cfg = SimpleNamespace(ctrl_dt=0.02)
        self._done = np.zeros(1, dtype=bool)
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


class _FakeWrappedEnv:
    """Raw wrapped-env contract consumed by FADAPlaybackSession."""

    def __init__(self, env: _FakeEnv, rng: np.random.Generator) -> None:
        self.env = env
        self._rng = rng
        self._obs = _raw_obs(rng)
        self.num_obs = 98

    def reset(self):
        self._obs = _raw_obs(self._rng)
        self.env.state.info["commands"] = _command()
        self.env.state.info["current_actions"] = np.zeros((1, ACTION_DIM), np.float32)
        return {"actor": torch.from_numpy(self._obs)}, {}

    def step(self, actions):
        np.copyto(self.env.state.info["current_actions"], actions.detach().cpu().numpy())
        np.copyto(self._obs, _raw_obs(self._rng))
        reward = torch.zeros(1)
        return {"actor": torch.from_numpy(self._obs)}, reward, torch.from_numpy(self.env._done.copy()), {}

    def get_observations(self):
        return {"actor": torch.from_numpy(self._obs)}


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
    "lambda_forget": 0.9,
    "update_interval": 2,
    "learning_rate": 1e-3,
    "max_grad_norm": 1.0,
    "lora_rank": 2,
    "lora_alpha": 4.0,
    "lora_dropout": 0.0,
}


def test_disabled_selects_offline_controller(policy, checkpoint_file):
    rng = np.random.default_rng(0)
    env, wrapped = _FakeEnv(rng), None
    wrapped = _FakeWrappedEnv(env, rng)
    session, _, _ = _make_session(policy, env, wrapped, None, checkpoint_file)
    assert isinstance(session.controller, FADAPlaybackController)
    assert not isinstance(session.controller, OnlineAdaptationController)
    session.reset()
    session.step_once()
    np.testing.assert_allclose(env.state.info["current_actions"], session.actions.numpy())
    assert session.last_update_stats is None
    assert list(fada_adapter_parameters(policy)) == []


def test_online_step_semantics_and_update_stats(policy, checkpoint_file):
    rng = np.random.default_rng(2)
    env = _FakeEnv(rng)
    wrapped = _FakeWrappedEnv(env, rng)
    session, _, _ = _make_session(policy, env, wrapped, dict(_ONLINE), checkpoint_file)
    controller: OnlineAdaptationController = session.controller
    assert controller.policy is controller.updater.policy is policy
    session.reset()

    transitions: list[dict[str, np.ndarray]] = []
    original_append = controller.collector.append

    def _spy(**kwargs):
        transitions.append(kwargs)
        return original_append(**kwargs)

    controller.collector.append = _spy

    planner_before = [p.detach().clone() for p in policy.planner.parameters()]
    lora_before = [p.detach().clone() for p in fada_adapter_parameters(policy)]

    num_steps = HISTORY + HORIZON + 2  # cross one full window plus two update ticks
    actions_seen: list[np.ndarray] = []
    observations_seen: list[np.ndarray] = []
    published_stats: list[dict] = []
    for _ in range(num_steps):
        observations_seen.append(np.concatenate((wrapped._obs[0, :64], wrapped._obs[0, 96:98])).copy())
        session.step_once()
        actions_seen.append(env.state.info["current_actions"].copy())
        np.testing.assert_array_equal(transitions[-1]["next_observation"], np.concatenate((wrapped._obs[0, :64], wrapped._obs[0, 96:98])))
        if session.last_update_stats is not None:
            published_stats.append(session.last_update_stats)

    # Session executed the online controller's actions (== playback policy output).
    assert session.actions is not None
    assert np.allclose(actions_seen[-1], session.actions.detach().cpu().numpy())

    # Exact previous/current action semantics on the projected contract.
    assert len(transitions) == num_steps
    for index, tr in enumerate(transitions):
        assert tr["observation"].shape == (OBS_DIM,)
        assert tr["next_observation"].shape == (OBS_DIM,)
        assert tr["executed_action"].shape == (ACTION_DIM,)
        assert np.allclose(tr["executed_action"], actions_seen[index][0])
        np.testing.assert_array_equal(tr["observation"], observations_seen[index])
        expected_prev = (
            np.zeros(ACTION_DIM, dtype=np.float32)
            if index == 0
            else actions_seen[index - 1][0]
        )
        assert np.allclose(tr["previous_action"], expected_prev)

    # At least one window collected and update stats published.
    assert published_stats, "expected at least one periodic LoRA update"
    assert "loss" in published_stats[-1]

    # LoRA mutated; Planner frozen.
    lora_after = list(fada_adapter_parameters(policy))
    assert any(
        not torch.allclose(b, a) for b, a in zip(lora_before, lora_after)
    ), "LoRA adapters did not update"
    planner_after = list(policy.planner.parameters())
    for before, after in zip(planner_before, planner_after):
        assert torch.allclose(before, after)
    assert not policy.idm.training


def test_terminal_transition_rejected_and_autoreset_resets_history(policy, checkpoint_file):
    rng = np.random.default_rng(3)
    env = _FakeEnv(rng)
    wrapped = _FakeWrappedEnv(env, rng)
    session, _, _ = _make_session(policy, env, wrapped, dict(_ONLINE), checkpoint_file)
    controller: OnlineAdaptationController = session.controller
    session.reset()
    session.set_autoreset(True)

    session.step_once()
    env.set_next_done(True)
    records_before = len(controller.collector._records)
    session.step_once()  # terminal step

    # Terminal transition rejected (no append), then reset(done) cleared episode
    # trajectory without discarding learned LoRA or valid updater windows.
    assert records_before == 1
    assert len(controller.collector._records) == 0
    assert controller._last_obs is None

    # Next step starts a fresh episode: previous_action is zero.
    captured: list[dict[str, np.ndarray]] = []
    original_append = controller.collector.append
    controller.collector.append = lambda **kw: (captured.append(kw), original_append(**kw))[1]
    env.set_next_done(False)
    session.step_once()
    assert np.allclose(captured[-1]["previous_action"], np.zeros(ACTION_DIM, np.float32))


def test_autoreset_disabled_keeps_history_on_done(policy, checkpoint_file):
    rng = np.random.default_rng(4)
    env = _FakeEnv(rng)
    wrapped = _FakeWrappedEnv(env, rng)
    session, _, _ = _make_session(policy, env, wrapped, dict(_ONLINE), checkpoint_file)
    controller: OnlineAdaptationController = session.controller
    session.reset()
    session.set_autoreset(False)

    session.step_once()
    env.set_next_done(True)
    records_before = len(controller.collector._records)
    session.step_once()

    # No done is passed downstream: transition kept, history/collector untouched.
    assert len(controller.collector._records) == records_before + 1


def test_enabled_requires_single_env(policy, checkpoint_file):
    rng = np.random.default_rng(5)
    env = _FakeEnv(rng)
    wrapped = _FakeWrappedEnv(env, rng)
    bad = _playback_cfg(checkpoint_file)
    object.__setattr__(bad, "num_envs", 2)
    with pytest.raises(ValueError, match="exactly one env"):
        create_fada_playback_session(
            playback_cfg=bad,
            cfg=_cfg(dict(_ONLINE)),
            root_dir="/tmp",
            device="cpu",
            deps=_deps(policy, env, wrapped, checkpoint_file),
            log=lambda msg: None,
        )


def test_enabled_rejects_already_adapted_policy(policy, checkpoint_file):
    rng = np.random.default_rng(6)
    env = _FakeEnv(rng)
    wrapped = _FakeWrappedEnv(env, rng)
    session, _, _ = _make_session(policy, env, wrapped, dict(_ONLINE), checkpoint_file)
    with pytest.raises(ValueError, match="without existing LoRA"):
        OnlineAdaptationController(policy, FADALoRAConfig(rank=2, alpha=4.0), device="cpu")


@pytest.mark.parametrize("values", [
    {"update_interval": 0}, {"update_interval": 1.5},
    {"learning_rate": float("nan")}, {"learning_rate": 0.0},
    {"max_grad_norm": -1.0}, {"max_grad_norm": float("inf")},
])
def test_online_config_rejects_invalid_update_settings(values):
    from unilab.algos.torch.distill.fada.online_controller import FADAOnlineAdaptationConfig

    with pytest.raises(ValueError):
        FADAOnlineAdaptationConfig(**values)


def test_online_requires_policy_mode_before_environment_creation(policy, checkpoint_file):
    rng = np.random.default_rng(8)
    env = _FakeEnv(rng)
    playback = _playback_cfg(checkpoint_file)
    object.__setattr__(playback, "action_mode", "zero")
    with pytest.raises(ValueError, match="policy action mode"):
        create_fada_playback_session(
            playback_cfg=playback, cfg=_cfg(dict(_ONLINE)), root_dir="/tmp",
            device="cpu", deps={},
        )
