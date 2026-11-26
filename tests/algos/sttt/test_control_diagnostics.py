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
from unilab.algos.torch.sttt.path_following.actor_learner import ActorLearner
from unilab.algos.torch.sttt.path_following.control_diagnostics import run_control_diagnostics
from unilab.algos.torch.sttt.path_following.control_model_diagnostics import (
    DiagnosticModels, interval_order, measured_return_interval, shadow_actor_update, trace_start,
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
    actor = LatentResidualActor(loaded.model.feature_dim, (16,), [.001] * 29, command_dim=3, preview_points=2)
    training = {"learning_rate": .001, "max_grad_norm": 10., "window_steps": 2, "horizon_windows": 2,
                "microbatch_size": 1, "batch_size": 1, "critic": {"hidden_dims": [16], "epochs": 1, "minibatches": 1}}
    learner = ActorLearner(actor, rollout, training, {})
    save_actor_checkpoint(run / "actor.pt", actor, loaded.model, loaded.contract, trajectory,
        training_state={"optimizer_steps": 0}, learner_state=learner.state_dict())
    config = {"policy_checkpoint": str(policy), "profile_path": None, "source_overrides": [],
        "capture": {"window_steps": 2}, "actor": {"training": {"residual_limit_rad": [.001] * 29},
        "trajectories": {"base": asdict(trajectory)}}}
    (run / "summary.json").write_text(json.dumps({"config": config}))
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
    cfg.critic_trajectory_kinds = ["straight"]
    return case, cfg, trajectory, policies


def test_complete_diagnostics_and_offline_reanalysis_preserve_artifacts(tmp_path):
    case, cfg, trajectory, policies = paired_run(tmp_path)
    run = tmp_path / "trained"
    before = {name: (run / name).read_bytes() for name in ("actor.pt", "world_model.pt")}
    report = run_control_diagnostics(cfg, root_dir=ROOT, playback_deps=case.deps)
    assert report["complete"], report["error"]
    assert set(report["stages"]) == {"physical_authority", "measured_feedback", "world_model", "critic", "actor_shadow"}
    assert report["stages"]["physical_authority"]["status"] == "unresolved_effect"
    assert all(row["first_action_matches"] for row in report["stages"]["world_model"]["branches"])
    shadow = report["stages"]["actor_shadow"]["cases"]["straight"]
    assert not shadow["critic_refit"] and shadow["optimizer_steps"] == 1
    assert "terminal_without_direct_memory" in shadow["term_gradients"]
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
    assert offline["stages"]["actor_shadow"] == report["stages"]["actor_shadow"]
    models = DiagnosticModels.load(run / "world_model.pt", run / "actor.pt", case.deps["resolve_checkpoint"](None), trajectory=trajectory, device="cpu")
    trace = PhysicalTrace.load(tmp_path / "diagnostics/critic/straight_actor")
    actor_before = {key: value.clone() for key, value in models.actor.state_dict().items()}
    critic_before = {key: value.clone() for key, value in models.learner.critic_learner.target.state_dict().items()}
    shadow_actor_update(trace, models)
    for key, value in actor_before.items():
        torch.testing.assert_close(models.actor.state_dict()[key], value, rtol=0., atol=0.)
    for key, value in critic_before.items():
        torch.testing.assert_close(models.learner.critic_learner.target.state_dict()[key], value, rtol=0., atol=0.)
    # A real terminal tail cannot disappear merely because recording stopped.
    failed = PhysicalTrace({**trace.metadata, "end_reason": "terminated"}, trace.arrays)
    interval = measured_return_interval(failed, models, step=4)
    assert interval["lower"] >= models.learner.costs.discount**4 * models.learner.critic_learner.failure_value
    assert interval_order({"status": "bounded", "lower": 1., "upper": 3.}, {"status": "bounded", "lower": 2., "upper": 4.}) is None
    # The extracted actor-only operation retains the exact production update.
    original = models.shadow_learner()
    split = models.shadow_learner()
    start = trace_start(trace, models)
    torch.manual_seed(19)
    original.update(start)
    torch.manual_seed(19)
    refreshed = split.refresh_start(start)
    split.fit_critic(refreshed)
    split.actor_step(refreshed)
    for key, value in original.actor.state_dict().items():
        torch.testing.assert_close(split.actor.state_dict()[key], value, rtol=0., atol=0.)


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
