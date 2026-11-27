"""Real network/session regressions on numerical I/O; no robot dynamics or trained-policy claims."""
from __future__ import annotations

import copy
import json
import subprocess
from dataclasses import replace
from threading import Event

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from test_latent_pretraining import physical_inputs
from test_latent_world_model import captured_query, legacy_case, parts
from test_path_following_runtime import ROOT, create_session
from unilab.algos.torch.distill.fada.runtime_policy import FADAFutureQuery, invoke_fada_idm, invoke_fada_planner
from unilab.algos.torch.sttt.path_following.latent_rollout import PlannerTrackerRollout
from unilab.algos.torch.sttt.path_following.planner_lora import adapter_parameters, adapter_state
from unilab.algos.torch.sttt.path_following.planner_losses import planner_trajectory_loss, timed_reference
from unilab.algos.torch.sttt.path_following.trajectory import PathState, ReferenceTrajectory, TrajectoryConfig
from unilab.algos.torch.sttt.world_model.checkpoint import LoadedWorldModel, save_world_model
from unilab.algos.torch.sttt.world_model.planner_online import PlannerLoRAUpdater, PlannerOnlineConfig


@pytest.fixture(autouse=True)
def one_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)


def legacy_parts(directory):
    case, profile = legacy_case(directory / "inputs")
    _, contract, model = physical_inputs(directory / "captures", source=(case, profile))

    def factory(physical):
        return PlannerTrackerRollout(physical, case.policy, profile, profile.preprocessing,
            target_joint_names=contract["source"]["target_joint_names"],
            target_default_joint_angles=contract["source"]["target_default_joint_angles"],
            target_action_scale=contract["source"]["target_action_scale"])

    return case, profile, contract, model, factory


@pytest.mark.parametrize("legacy", [False, True])
def test_zero_delta_retains_tracker_execution_and_bptt_reaches_only_planner(tmp_path, legacy):
    case, profile, contract, model, factory = legacy_parts(tmp_path) if legacy else parts(tmp_path / "inputs")
    rollout = factory(model)
    if legacy:
        obs = torch.linspace(-.1, .1, 66).reshape(1, 1, 66).repeat(1, 4, 1)
        obs[:, :, 3:6] = torch.tensor([0., 0., -1.])
        actions, valid, command = torch.zeros(1, 4, 29), torch.ones(1, 4, dtype=torch.bool), torch.tensor([[.4, 0., 0.]])
        nominal = invoke_fada_planner(case.policy, rollout.spec, obs, actions, command, history_valid_mask=valid)
        query = FADAFutureQuery(obs, actions, command, valid, nominal, rollout.preprocessing, profile)
        source = invoke_fada_idm(case.policy.idm, rollout.spec, obs, actions, command, nominal, history_valid_mask=valid)[:, 0]
        if profile.clip_actions is not None:
            source = source.clamp(-profile.clip_actions, profile.clip_actions)
    else:
        query, source = captured_query(case, profile)
    raw_history = query.observation_history.clone()
    tracker = copy.deepcopy(case.policy.idm.state_dict())
    updater = PlannerLoRAUpdater(LoadedWorldModel(model, contract, {}), rollout, PlannerOnlineConfig(lora_rank=2))
    geometry = ReferenceTrajectory.generate(TrajectoryConfig(), dtype=query.command.dtype)
    state = PathState(geometry, torch.zeros(1, 3), torch.zeros(1))
    start = rollout.encode_start(query, errors=torch.zeros(1, 2), previous_residual=torch.zeros(1, 29), path_state=state)
    target, commands = timed_reference(geometry, 0., 4, .02)
    prediction = rollout.run(start, None, window_steps=2, horizon_windows=2,
                             update_reference_commands=False, velocity_commands=commands)
    torch.testing.assert_close(prediction.source_actions[:, 0], source, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(prediction.actions[:, 0], rollout.source_to_env(source), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(query.observation_history, raw_history, atol=0, rtol=0)
    loss = planner_trajectory_loss(state.pose, prediction.body_delta, target,
                                  position_tolerance_m=.3, heading_tolerance_rad=.3)
    loss.backward()
    assert sum(float(parameter.grad.square().sum()) for parameter in updater.planner_parameters if parameter.grad is not None) > 0
    assert all(parameter.grad is None for parameter in updater.model.parameters())
    assert all(parameter.grad is None for parameter in updater.policy.get_submodule("idm").parameters())
    assert all(not parameter.requires_grad for parameter in updater.live_policy.parameters())
    for name, value in tracker.items():
        torch.testing.assert_close(updater.policy.get_submodule("idm").state_dict()[name], value, atol=0, rtol=0)


def test_task_time_penalizes_standing_and_pose_composition_preserves_heading():
    geometry = ReferenceTrajectory.generate(TrajectoryConfig(speed_m_s=.4), dtype=torch.float64)
    target, commands = timed_reference(geometry, 1., 10, .02)
    torch.testing.assert_close(target[0, [0, -1], 0], torch.tensor([.4, .48], dtype=torch.float64))
    torch.testing.assert_close(commands[..., 0], torch.full((1, 10), .4, dtype=torch.float64))
    origin = torch.tensor([[.4, 0., 0.]], dtype=torch.float64)
    standing = torch.zeros(1, 10, 3, dtype=torch.float64, requires_grad=True)
    walking = standing.detach().clone()
    walking[..., 0] = .008
    def cost(motion):
        return planner_trajectory_loss(origin, motion, target, position_tolerance_m=.3, heading_tolerance_rad=.3)
    assert float(cost(walking)) < 1e-20
    loss = cost(standing)
    assert float(loss.detach()) > 0
    assert float(torch.autograd.grad(loss, standing)[0][..., 0].sum()) < 0
    rotated = replace(geometry.config, initial_heading_rad=1.0)
    reference = ReferenceTrajectory.generate(rotated, dtype=torch.float64)
    poses, _ = timed_reference(reference, 0., 10, .02)
    heading = poses[:, 0, 2]
    aligned = torch.stack((torch.zeros_like(heading), torch.zeros_like(heading), heading), -1)
    assert float(planner_trajectory_loss(aligned, walking, poses, position_tolerance_m=.3, heading_tolerance_rad=.3)) < 1e-20


def configure_session(case, checkpoint):
    node = OmegaConf.load(ROOT / "conf/distill/path_following/planner_wm_lora.yaml")
    node.checkpoint_path = str(checkpoint)
    node.pose.odometry_source = "simulation"
    node.update_interval_steps = 2
    node.learning = {"online": {"lora_rank": 2, "calibration_windows": 1, "replay_sequence_windows": 1}}
    case.cfg.path_following = node
    case.cfg.online_adaptation.enabled = False
    return create_session(case)


@pytest.mark.parametrize("legacy", [False, True])
def test_session_reencodes_after_wm_update_and_publishes_only_completed_pairs(tmp_path, monkeypatch, legacy):
    case, _, contract, model, factory = legacy_parts(tmp_path) if legacy else parts(tmp_path / "inputs")
    raw_policy = copy.deepcopy(case.policy)
    wm = save_world_model(tmp_path / "wm.pt", model, contract, training_state={})
    session = configure_session(case, wm)
    controller, updater = session.controller, session.controller.updater
    live_before = adapter_state(updater.live_policy)
    wm_before = adapter_state(updater.inference_model)
    frozen = {name: parameter.detach().clone() for name, parameter in updater.model.named_parameters()
              if name not in adapter_parameters(updater.model)}
    tracker = copy.deepcopy(updater.policy.idm.state_dict())
    encoded_at = []
    original_encode, original_step = updater._encode, updater._step
    entered, release = Event(), Event()

    def encode(inputs):
        encoded_at.append(updater.wm_updates)
        return original_encode(inputs)

    def step(loss, parameters, optimizer):
        if parameters is updater.planner_parameters:
            entered.set()
            if not release.wait(10):
                raise TimeoutError("test failed to release the learner")
        return original_step(loss, parameters, optimizer)

    monkeypatch.setattr(updater, "_encode", encode)
    monkeypatch.setattr(updater, "_step", step)
    try:
        session.step_once()
        session.step_once()
        assert entered.wait(10)
        assert updater.wm_updates == 1 and updater.planner_updates == 0
        assert encoded_at == [0, 1]
        for name, value in live_before.items():
            torch.testing.assert_close(adapter_state(updater.live_policy)[name], value, atol=0, rtol=0)
        for name, value in wm_before.items():
            torch.testing.assert_close(adapter_state(updater.inference_model)[name], value, atol=0, rtol=0)
        session.step_once()  # Physical clock keeps advancing while learning is blocked.
        assert session.env.tick == 3
        release.set()
        controller._finish_learning(wait=True)
        assert updater.planner_updates == 1
        assert any(not torch.equal(adapter_state(updater.live_policy)[name], value) for name, value in live_before.items())
        assert any(not torch.equal(adapter_state(updater.inference_model)[name], value) for name, value in wm_before.items())
        for name, value in frozen.items():
            torch.testing.assert_close(dict(updater.model.named_parameters())[name], value, atol=0, rtol=0)
        for name, value in tracker.items():
            torch.testing.assert_close(updater.policy.idm.state_dict()[name], value, atol=0, rtol=0)
        path = tmp_path / "online.pt"
        updater.save(path)

        def resume_factory(physical):
            result = factory(physical)
            result.policy = copy.deepcopy(raw_policy)
            return result

        restored = PlannerLoRAUpdater.restore(path, resume_factory, expected_contract=contract)
        assert restored.update_counters() == updater.update_counters()
        for target, original in ((restored.model, updater.model), (restored.policy, updater.policy)):
            for name, value in adapter_state(original).items():
                torch.testing.assert_close(adapter_state(target)[name], value, atol=0, rtol=0)
        for left, right in ((restored.model_optimizer, updater.model_optimizer),
                            (restored.planner_optimizer, updater.planner_optimizer)):
            assert left.state_dict()["param_groups"] == right.state_dict()["param_groups"]
            for key, values in right.state_dict()["state"].items():
                for name, value in values.items():
                    torch.testing.assert_close(left.state_dict()["state"][key][name], value, atol=0, rtol=0)
        controller.reset()
        assert updater.update_counters() == restored.update_counters()
        assert not updater.records and not updater.replay
    finally:
        release.set()
        session.close()
    assert all(env.closed for env in case.environments)


def test_wm_only_launcher_compose_and_real_help(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("planner_test_launcher", ROOT / "scripts/launch_latent_pretraining.py")
    import sys
    assert spec is not None and spec.loader is not None
    launcher = importlib.util.module_from_spec(spec)
    monkey_name = spec.name
    sys.modules[monkey_name] = launcher
    try:
        spec.loader.exec_module(launcher)
        policy = tmp_path / "policy.pt"
        policy.touch()
        for split, parent in (("train", "capture/nominal/train"), ("validation", "validation/heldout")):
            directory = tmp_path / parent
            directory.mkdir(parents=True)
            (directory / "metadata.json").write_text(json.dumps({"transition_count": 10}))
            (directory / "capture.json").write_text(json.dumps({"split": split, "episode_ids": [split]}))
        command = launcher.training_command(tmp_path, policy, tmp_path / "output", "uv", stage="wm")
        assert command[3] == "scripts/train_latent_world_model.py"
        with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "conf/sttt")):
            cfg = compose(config_name="planner_wm_pretraining", overrides=command[5:])
        assert cfg.device == "cuda" and cfg.training.microbatch_size == 256
        assert list(cfg.data.train_dirs) == [str(tmp_path / "capture/nominal/train")]
        assert list(cfg.data.validation_dirs) == [str(tmp_path / "validation/heldout")]
        assert not (tmp_path / "output").exists()
        result = subprocess.run(["uv", "run", "--no-sync", "scripts/launch_latent_pretraining.py", "--help"],
                                cwd=ROOT, capture_output=True, text=True, check=True, timeout=10)
        assert "--stage {all,base,wm}" in result.stdout
    finally:
        sys.modules.pop(monkey_name, None)


def test_matched_evaluation_records_frozen_identification_and_dual_update_modes(tmp_path):
    from test_path_following_evaluation import evaluation_case
    from unilab.algos.torch.sttt.path_following.evaluation import run_path_following_evaluation

    modes = ("planner_wm_frozen", "planner_wm_only", "planner_wm_online")
    case, original, _ = evaluation_case(tmp_path / "evaluation", modes=modes)
    _, _, contract, model, _ = parts(tmp_path / "evaluation" / "inputs", case=case)
    artifact = save_world_model(tmp_path / "evaluation_wm.pt", model, contract)
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "conf/sttt")):
        preset = compose(config_name="planner_wm_evaluate")
    cfg = OmegaConf.merge(preset, original, {
        "world_model_checkpoint": str(artifact),
        "planner_learning": {"online": {"lora_rank": 2, "calibration_windows": 1, "replay_sequence_windows": 1}},
    })
    result = run_path_following_evaluation(cfg, root_dir=ROOT, playback_deps=case.deps)
    assert result["complete"], result
    frozen, identified, adapted = result["runs"]
    assert frozen["runtime"]["wm_updates"] == frozen["runtime"]["planner_updates"] == 0
    assert identified["runtime"]["wm_updates"] > 0 and identified["runtime"]["planner_updates"] == 0
    assert adapted["runtime"]["wm_updates"] > 0 and adapted["runtime"]["planner_updates"] > 0
    assert all(run["metrics"]["time_coverage"] == 1. for run in result["runs"])
    assert all(env.closed and not env.autoreset for env in case.environments)
