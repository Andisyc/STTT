"""Real latent/Planner/Tracker/CLI consumers with deterministic analytical physical I/O."""
from __future__ import annotations

import hashlib
import json
import runpy
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra.core.global_hydra import GlobalHydra

from test_latent_world_model import ROOT, legacy_case, model_config
from unilab.algos.torch.sttt.path_following import latent_response_probe as probe_module
from unilab.algos.torch.sttt.world_model.checkpoint import save_world_model
from unilab.algos.torch.sttt.world_model.model import LatentWorldModel


def probe_case(directory, failure):
    case, profile = legacy_case(directory / "inputs")
    policy_path = case.deps["resolve_checkpoint"](None)
    contract = {
        "data": {"window_steps": 2, "ctrl_dt": .02, "history_length": 4, "prediction_horizon": 3,
                 "observation_dim": 66, "action_dim": 29, "command_dim": 3,
                 "observation_contract": "g1_fada_state_v2"},
        "source": {"checkpoint_sha256": hashlib.sha256(Path(policy_path).read_bytes()).hexdigest(),
                   "source_profile": asdict(profile), "preprocessing": profile.preprocessing,
                   "target_joint_names": list(profile.joint_names), "target_default_joint_angles": [0.] * 29,
                   "target_action_scale": [.5] * 29, "delta_frame": "previous_control_horizontal_heading",
                   "delta_units": ["m", "m", "rad"], "ego_frame": "previous_measurement_horizontal_heading",
                   "risk_tilt_limit_rad": .8},
    }
    checkpoint = directory / "wm.pt"
    model = LatentWorldModel(replace(model_config(), obs_dim=66))
    save_world_model(checkpoint, model, contract)
    create_env = case.deps["create_env"]
    load_policy = case.deps["load_fada_policy"]
    original_wrapper = case.deps["wrapper_cls"]
    policies = []

    def loader(*args, **kwargs):
        loaded = load_policy(*args, **kwargs)
        role = len(policies)
        if failure == "query" and role == 1:
            loaded.policy.planner.register_forward_hook(
                lambda _, args, output: output + .001 if case.environments[-1].tick == 4 else output)
        policies.append(loaded.policy)
        return loaded

    def environment(*args, **kwargs):
        role = len(case.environments)
        env = create_env(*args, **kwargs)
        get_position = env.get_base_pos
        # Nonzero path heading distinguishes fixed-path labels from window-body labels.
        def quaternion():
            yaw = .4 + env.tick * .01
            return np.array([[np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)]])
        env.get_base_quat = quaternion
        if failure == "start" and role == 1:
            env.get_base_pos = lambda: get_position() + np.array([[.001, 0., 0.]])
        elif failure == "zero_repeat" and role == 3:
            env.get_base_pos = lambda: get_position() + (np.array([[.01, 0., 0.]]) if env.tick > 4 else 0.)
        def capture():
            native = np.concatenate(([env.get_sim_time_ns() / 1e9], env.get_base_pos()[0],
                                     env.get_base_quat()[0], env.get_dof_pos()[0], env.get_dof_vel()[0]))
            return SimpleNamespace(backend_state={"physics_state": native[None].astype(np.float64)})
        env.capture_rollout_snapshot = capture
        return env

    def wrapper(env, **kwargs):
        wrapped = original_wrapper(env, **kwargs)
        if failure == "terminal" and len(case.environments) == 2:
            original_step = wrapped.step
            def step(action):
                result = original_step(action)
                if env.tick == 5:
                    env.state.terminated[:] = True
                    return result[0], result[1], torch.tensor([True]), result[3]
                return result
            wrapped.step = step
        return wrapped

    case.deps.update(create_env=environment, load_fada_policy=loader, wrapper_cls=wrapper)
    return case, checkpoint, policies


@pytest.mark.parametrize("failure", [None, "start", "query", "terminal", "zero_repeat"])
def test_real_cli_legacy_world_model_verified_contrast(tmp_path, monkeypatch, failure):
    case, checkpoint, policies = probe_case(tmp_path, failure)
    output = tmp_path / "probe"
    original = probe_module.run_latent_response_direction_probe
    reports = []
    def run(cfg, *, root_dir):
        result = original(cfg, root_dir=root_dir, playback_deps=case.deps)
        reports.append(result)
        return result
    monkeypatch.setattr(probe_module, "run_latent_response_direction_probe", run)
    monkeypatch.setattr(sys, "argv", [str(ROOT / "scripts/probe_latent_response_directions.py"),
        f"policy_checkpoint={case.cfg.training.play_checkpoint_path}", f"world_model_checkpoint={checkpoint}",
        f"output_dir={output}", "warmup_steps=4", "horizon_windows=2", "residual_amplitude_rad=0.001",
        "replay_position_tolerance_m=0.00000001", "replay_heading_tolerance_rad=0.00000001",
        "minimum_effect_m=0.00001", "minimum_effect_rad=0.00001", f"hydra.run.dir={tmp_path / 'hydra'}"])
    if failure is None:
        runpy.run_path(str(ROOT / "scripts/probe_latent_response_directions.py"), run_name="__main__")
    else:
        with pytest.raises(SystemExit) as error:
            runpy.run_path(str(ROOT / "scripts/probe_latent_response_directions.py"), run_name="__main__")
        assert error.value.code == 1
    assert not GlobalHydra.instance().is_initialized()
    report = reports[0]
    assert all(env.closed and not env.autoreset for env in case.environments)
    assert all(not parameter.requires_grad and parameter.grad is None for p in policies for parameter in p.parameters())
    if failure in {"start", "query", "terminal"}:
        assert not report["complete"] and not report["valid"] and report["comparison"] is None
        assert len(report["branches"]) == 2
        assert report["branches"][-1]["error"] is not None
        assert (output / "plus.npz").is_file()
        return
    assert report["complete"] and len(report["branches"]) == 4
    assert report["valid"] == (failure is None)
    assert len({b["context_sha256"] for b in report["branches"]}) == 1
    assert all(b["completed_steps"] == 8 for b in report["branches"])
    assert report["comparison"]["channel_status"] == (["unresolved_effect"] * 3 if failure is None else ["invalid_replay"] * 3)
    with np.load(output / "zero_a.npz", allow_pickle=False) as zero, np.load(output / "plus.npz", allow_pickle=False) as plus:
        np.testing.assert_array_equal(zero["start_native_state"], plus["start_native_state"])
        np.testing.assert_array_equal(zero["start_latent_state"], plus["start_latent_state"])
        np.testing.assert_array_equal(zero["start_latent_belief"], plus["start_latent_belief"])
        assert bool(zero["start_ego_valid"].all()) and np.linalg.norm(zero["start_ego_motion"]) > 0
        np.testing.assert_allclose(plus["predicted_actions"][0], plus["executed_actions"][4], rtol=1e-6, atol=1e-6)
        assert plus["predicted_actions"].shape == (4, 29) and plus["predicted_path_delta"].shape == (4, 3)
        if failure is None:
            expected = [np.cos(.4) * .04 + np.sin(.4) * .012,
                        -np.sin(.4) * .04 + np.cos(.4) * .012, .04]
            np.testing.assert_allclose(report["branches"][0]["delta_m_m_rad"], expected, rtol=1e-6, atol=1e-7)
    assert json.loads((output / "summary.json").read_text())["valid"] == report["valid"]
