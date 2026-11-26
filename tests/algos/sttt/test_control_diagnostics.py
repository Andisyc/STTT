"""Consumer-visible physical contracts and isolated learned-control diagnostics."""
from __future__ import annotations

import copy
import json
import shutil
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from test_latent_response_direction_probe import probe_case
from test_latent_world_model import ROOT
from unilab.algos.torch.sttt.path_following.actor_checkpoint import save_actor_checkpoint
from test_residual_film_ppo import make_ppo_learner
from unilab.algos.torch.sttt.path_following.control_diagnostics import run_control_diagnostics
from unilab.algos.torch.sttt.path_following.control_model_diagnostics import (
    DiagnosticModels, isolated_randomness, shadow_context_update, trace_start,
)
from unilab.algos.torch.sttt.path_following.control_probe import PhysicalTrace
from unilab.algos.torch.sttt.path_following.control_response import (
    MeasuredFeedback, bounded_least_squares, fit_response_map,
)
from unilab.algos.torch.sttt.path_following.latent_actor import LatentResidualActor
from unilab.algos.torch.sttt.path_following.pretraining import load_rollout
from unilab.algos.torch.sttt.path_following.trajectory import PathState, ReferenceTrajectory, TrajectoryConfig
from unilab.algos.torch.sttt.world_model.checkpoint import load_world_model


def paired_run(tmp_path):
    case, checkpoint, policies = probe_case(tmp_path, None)
    original_wrapper = case.deps["wrapper_cls"]
    def command_aware_observations(env, **kwargs):
        wrapper = original_wrapper(env, **kwargs)
        def refresh_state():
            env.state.obs = wrapper.observation()
            return env.state
        env.refresh_state = refresh_state
        wrapper.get_observations = wrapper.observation
        return wrapper
    case.deps["wrapper_cls"] = command_aware_observations
    run = tmp_path / "trained"
    run.mkdir()
    shutil.copy2(checkpoint, run / "world_model.pt")
    policy = case.deps["resolve_checkpoint"](None)
    loaded = load_world_model(checkpoint)
    rollout = load_rollout(loaded, policy)
    trajectory = TrajectoryConfig(preview_points=2, speed_m_s=.35)
    actor = LatentResidualActor(loaded.model.feature_dim, (16,), [.001] * 29, command_dim=3, preview_points=2,
                                adaptation_dim=16)
    with torch.no_grad():
        actor.net[-1].weight.normal_(std=.01)
    learner = make_ppo_learner(actor, rollout, stage="film", batch_size=1)
    save_actor_checkpoint(run / "actor.pt", actor, loaded.model, loaded.contract, trajectory,
        training_state={"optimizer_steps": 0}, learner_state=learner.state_dict())
    config = {"policy_checkpoint": str(policy), "profile_path": None, "source_overrides": [],
        "capture": {"window_steps": 2}, "actor": {"training": {"residual_limit_rad": [.001] * 29},
        "trajectories": {"base": asdict(trajectory)}}}
    (run / "summary.json").write_text(json.dumps({"config": config,
        "world_model_checkpoint": str(run / "world_model.pt"), "actor_checkpoint": str(run / "actor.pt")}))
    cfg = OmegaConf.load(ROOT / "conf/sttt/latent_control_diagnostics.yaml")
    cfg.run_dir = str(run)
    cfg.output_dir = str(tmp_path / "diagnostics")
    cfg.warmup_steps = 4
    cfg.phase_fractions = [0.]
    cfg.amplitude_fractions = [1.]
    cfg.observation_cycles = .01
    cfg.feedback_duration_s = .08
    cfg.continuation_duration_s = .08
    cfg.model_top_directions = 1
    cfg.model_seeds = [0]
    cfg.context_trajectory_kinds = ["straight"]
    return case, cfg, trajectory, policies


def test_complete_diagnostics_and_offline_reanalysis_preserve_artifacts(tmp_path):
    case, cfg, trajectory, policies = paired_run(tmp_path)
    run = tmp_path / "trained"
    before = {name: (run / name).read_bytes() for name in ("actor.pt", "world_model.pt")}
    report = run_control_diagnostics(cfg, root_dir=ROOT, playback_deps=case.deps)
    assert report["complete"], report["error"]
    assert set(report["stages"]) == {"physical_authority", "measured_feedback", "world_model", "context_shadow"}
    assert report["stages"]["physical_authority"]["status"] == "unresolved_effect"
    assert all(row["first_action_matches"] for row in report["stages"]["world_model"]["branches"])
    shadow = report["stages"]["context_shadow"]["cases"]["straight"]
    assert shadow["status"] == "context_updated" and shadow["optimizer_steps"] == 1
    assert shadow["optimizer_metrics"]["context_gradient_norm"] > 0
    assert all(env.closed and not env.autoreset for env in case.environments)
    assert all(not parameter.requires_grad and parameter.grad is None for policy in policies for parameter in policy.parameters())
    for name, value in before.items():
        assert (run / name).read_bytes() == value
    original_env_count = len(case.environments)
    analyzed = copy.deepcopy(cfg)
    analyzed.mode = "analyze"
    analyzed.data_dir = cfg.output_dir
    analyzed.output_dir = str(tmp_path / "reanalyzed")
    offline = run_control_diagnostics(analyzed, root_dir=ROOT, playback_deps=case.deps)
    assert offline["complete"], offline["error"]
    assert len(case.environments) == original_env_count
    assert offline["stages"]["context_shadow"] == report["stages"]["context_shadow"]
    models = DiagnosticModels.load(run / "world_model.pt", run / "actor.pt", case.deps["resolve_checkpoint"](None), trajectory=trajectory, device="cpu")
    trace = PhysicalTrace.load(tmp_path / "diagnostics/context/straight_before")
    actor_before = {key: value.clone() for key, value in models.actor.state_dict().items()}
    wm_before = {key: value.clone() for key, value in models.loaded.model.state_dict().items()}
    from unilab.algos.torch.sttt.path_following.context_adaptation import ContextAdapter
    _, changed = shadow_context_update(trace, models, seed=19)
    direct = ContextAdapter(models.actor, models.rollout, batch_size=1, window_steps=models.window_steps,
                            config=models.context_config)
    start = trace_start(trace, models, seed=19)
    with isolated_randomness(models.loaded.model, 19):
        direct.step(start)
    torch.testing.assert_close(changed, direct.context, rtol=0, atol=0)
    for key, value in actor_before.items():
        torch.testing.assert_close(models.actor.state_dict()[key], value, rtol=0., atol=0.)
    for key, value in wm_before.items():
        torch.testing.assert_close(models.loaded.model.state_dict()[key], value, rtol=0., atol=0.)


def pulse_dataset():
    limits = [.005] * 29
    n, k = 20, 2
    baseline = np.zeros((n + 1, 3))
    baseline[:, 0] = np.arange(n + 1) * .02 * .35
    kernel = np.zeros((n + 1, 3))
    kernel[:, 1] = np.minimum(np.maximum(np.arange(n + 1) - 2, 0), 4) * .01
    metadata = {"complete": True, "valid": True, "profile": {}, "preprocessing": {}, "policy_identity": {},
        "trajectory": {}, "window_steps": k, "reference_commands": False, "ctrl_dt": .02, "residual_limits_rad": limits}
    branches = {}
    def trace(pose, residual):
        matrices = np.broadcast_to(np.eye(4), (n + 1, 4, 4)).copy()
        matrices[:, :2, 3] = pose[:, :2]
        return PhysicalTrace(metadata.copy(), {"poses": matrices, "path_pose": pose,
            "origin_warmup_poses": np.eye(4)[None], "origin_native_state": np.zeros((1, 4)), "origin_gait_phase": np.zeros((1, 2)),
            "residuals": residual, "actions": np.zeros((n, 29)), "timestamps_ns": np.arange(n + 1, dtype=np.int64) * 20_000_000})
    branches["p0_zero_a"] = trace(baseline.copy(), np.zeros((n, 29)))
    branches["p0_zero_b"] = trace(baseline.copy(), np.zeros((n, 29)))
    for joint in range(29):
        for index, fraction in enumerate((.5, 1.)):
            for name, sign in (("plus", 1), ("minus", -1)):
                residual = np.zeros((n, 29)); residual[:k, joint] = sign * fraction * limits[joint]
                pose = baseline + (sign * fraction * kernel if joint == 0 else 0.)
                branches[f"p0_j{joint}_a{index}_{name}"] = trace(pose, residual)
    return branches, kernel


def test_measured_delayed_authority_drives_bounded_feedback_both_directions():
    branches, kernel = pulse_dataset()
    response, report = fit_response_map(branches, [0.], [.5, 1.], minimum_effect_m=1e-5, minimum_effect_rad=1e-5,
        replay_position_tolerance_m=1e-7, replay_heading_tolerance_rad=1e-7)
    np.testing.assert_allclose(response.response[0, :, :, 0], kernel, atol=1e-12)
    assert report["phases"][0]["joints"][0]["delay_s"] == .06
    assert report["phases"][0]["joints"][0]["detectable_lateral_or_heading"]
    for offset in (-.15, .15):
        controller = MeasuredFeedback(response)
        previous = torch.zeros(1, 29)
        actions = []
        reference = ReferenceTrajectory.generate(TrajectoryConfig(speed_m_s=.35, initial_lateral_m=offset))
        for step in range(0, 100, 2):
            y = offset + sum(kernel[min(step - when, len(kernel) - 1), 1] * value for when, value in actions)
            state = PathState(reference, torch.tensor([[step * .02 * .35, y - offset, 0.]]), torch.tensor([step * .02 * .35]))
            observation = torch.zeros(1, 1, 66)
            query = SimpleNamespace(command=torch.tensor([[.35, 0., 0.]]), observation_history=observation)
            decision = SimpleNamespace(query=query, step=step, window_steps=2, previous_residual=previous, path=state)
            previous = controller(decision)
            assert float(previous.abs().max()) <= .00500001
            actions.append((step, float(previous[0, 0] / .005)))
        assert abs(y) < abs(offset) * .25
    x = bounded_least_squares(np.array([[2., 0.], [0., .5]]), np.array([8., -.25]))
    np.testing.assert_allclose(x, [1., -.5], atol=1e-8)
    corrupted = branches["p0_j0_a0_plus"]
    corrupted.arrays["residuals"][0, 0] *= -1
    with pytest.raises(ValueError, match="applied residual"):
        fit_response_map(branches, [0.], [.5, 1.], minimum_effect_m=1e-5, minimum_effect_rad=1e-5,
            replay_position_tolerance_m=1e-7, replay_heading_tolerance_rad=1e-7)


def test_feedback_rerun_reuses_measurements_without_learned_checkpoints(tmp_path, monkeypatch):
    from unilab.algos.torch.sttt.path_following import control_diagnostics

    case, cfg, _, _ = paired_run(tmp_path)
    cfg.mode = "physical"
    original = run_control_diagnostics(cfg, root_dir=ROOT, playback_deps=case.deps)
    assert original["complete"], original["error"]
    source = tmp_path / "diagnostics"
    preserved = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    for name in ("world_model.pt", "actor.pt"):
        (tmp_path / "trained" / name).unlink()

    executed = []
    native_branch = control_diagnostics.run_physical_branch

    def record_branch(*args, **kwargs):
        executed.append(kwargs["name"])
        return native_branch(*args, **kwargs)

    monkeypatch.setattr(control_diagnostics, "run_physical_branch", record_branch)
    cfg.mode = "feedback"
    cfg.data_dir = str(source)
    cfg.output_dir = str(tmp_path / "feedback_rerun")
    rerun = run_control_diagnostics(cfg, root_dir=ROOT, playback_deps=case.deps)

    assert rerun["complete"], rerun["error"]
    assert executed == ["feedback/o0_corrected", "feedback/o1_corrected"]
    assert set(rerun["stages"]) == {"physical_authority", "measured_feedback"}
    assert rerun["stages"]["measured_feedback"] == original["stages"]["measured_feedback"]
    assert (tmp_path / "feedback_rerun/feedback/o0_corrected/trace.npz").is_file()
    assert not (tmp_path / "feedback_rerun/pulses").exists()
    for path, content in preserved.items():
        assert path.read_bytes() == content
