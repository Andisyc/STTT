"""Behavioral ordering, temporal credit and asynchronous publication contracts."""
from __future__ import annotations

import math
from threading import Event

import pytest
import torch
from omegaconf import OmegaConf

from test_latent_pretraining import physical_inputs
from test_latent_world_model import ROOT, actor_artifact, create_session, model_config, parts, sequence
from unilab.algos.torch.sttt.path_following.actor_learner import ActorLearner
from unilab.algos.torch.sttt.path_following.critic import CriticConfig, CriticLearner, lambda_cost_targets
from unilab.algos.torch.sttt.path_following.latent_actor import LatentResidualActor
from unilab.algos.torch.sttt.path_following.latent_objectives import LatentCostConfig, ObservedCostWindow, latent_actor_loss
from unilab.algos.torch.sttt.path_following.pretraining import actor_training_start, load_rollout
from unilab.algos.torch.sttt.path_following.trajectory import TrajectoryConfig
from unilab.algos.torch.sttt.world_model.checkpoint import LoadedWorldModel, save_world_model
from unilab.algos.torch.sttt.world_model.losses import world_model_loss
from unilab.algos.torch.sttt.world_model.model import LatentWorldModel
from unilab.algos.torch.sttt.world_model.pretraining_data import PretrainingSequenceSampler
from unilab.algos.torch.sttt.world_model.sequence_data import batch_sequences
from unilab.algos.torch.sttt.world_model.types import LatentState, RolloutStart


def measured_cost(*, velocity=0., forward=.35, window_steps=10, trim=0., initial_trim=0., initial_error=.15):
    steps = 20
    windows = steps // window_steps
    time = torch.arange(steps + 1, dtype=torch.float64)[None] * .02
    errors = torch.stack((initial_error + velocity * time, torch.zeros_like(time)), -1)
    motion = torch.zeros(1, steps, 3, dtype=torch.float64)
    motion[..., 0], motion[..., 1] = forward * .02, velocity * .02
    command = torch.tensor([[.35, 0., 0.]], dtype=torch.float64)
    observation = torch.zeros(1, 1, 66, dtype=torch.float64)
    observation[..., 5] = -1.
    start = RolloutStart(
        latent=LatentState(torch.tensor([[[1., 0.]]], dtype=torch.float64), torch.zeros(1, 2, dtype=torch.float64)),
        observation_history=observation, action_history=torch.zeros(1, 1, 29, dtype=torch.float64),
        history_valid_mask=torch.ones(1, 1, dtype=torch.bool), errors=errors[:, 0], command=command,
        previous_residual=torch.full((1, 29), initial_trim, dtype=torch.float64),
        ego_motion=torch.zeros(1, 1, 3, dtype=torch.float64), ego_valid=torch.zeros(1, 1, 1, dtype=torch.bool),
        ego_age_s=torch.zeros(1, 1, 1, dtype=torch.float64))
    measured = ObservedCostWindow(errors, motion, torch.zeros(1, steps, 29, dtype=torch.float64),
        torch.zeros(1, steps, 29, dtype=torch.float64), torch.full((1, windows, 29), trim, dtype=torch.float64),
        torch.zeros(1, steps, 1, dtype=torch.float64), command[:, None].expand(-1, steps, -1),
        torch.full((1, steps), .35, dtype=torch.float64))
    return latent_actor_loss(measured, start, ctrl_dt=.02, window_steps=window_steps,
                            config=LatentCostConfig(), residual_limits=torch.full((29,), .005, dtype=torch.float64))


def test_return_is_preferred_to_offset_and_stopping_and_constant_trim_is_free():
    stationary = measured_cost()
    returning = measured_cost(velocity=-.025, trim=.004)
    stopped = measured_cost(forward=0., initial_error=0.)
    assert returning["loss"] < stationary["loss"] < stopped["loss"]
    assert measured_cost(trim=.004, initial_trim=.004)["smoothness_loss"] == 0.
    assert measured_cost(trim=.004, initial_trim=0.)["smoothness_loss"] > 0.


def test_native_time_discount_is_unchanged_by_residual_window_partition():
    five = measured_cost(window_steps=5)["loss"]
    ten = measured_cost(window_steps=10)["loss"]
    expected = .5 * (1 - math.exp(-(.15 / .3)**2)) * .02 * sum(.99**i for i in range(20))
    assert float(five) == pytest.approx(expected, rel=1e-12)
    torch.testing.assert_close(five, ten, rtol=1e-12, atol=1e-12)


def test_motion_tolerances_follow_physical_time_and_keep_window_precision():
    raw = sequence(steps=10)
    raw["path_delta"][:] = (.001, 0., 0.)
    raw["window_delta"][:] = (.01, 0., 0.)
    model = LatentWorldModel(model_config()).eval().requires_grad_(False)
    with torch.no_grad():
        head = model.motion_head.model[-1]
        assert isinstance(head, torch.nn.Linear)
        head.weight.zero_()
        head.bias.zero_()
    fast = world_model_loss(model, batch_sequences([raw]))
    raw["timestamps_ns"] *= 2
    slow = world_model_loss(model, batch_sequences([raw]))
    assert float(fast["motion_loss"]) == pytest.approx(1 / 3, rel=1e-6)
    assert float(fast["window_loss"]) == pytest.approx(1 / 3, rel=1e-6)
    torch.testing.assert_close(slow["motion_loss"], fast["motion_loss"] / 4)
    torch.testing.assert_close(slow["window_loss"], fast["window_loss"])


def test_failure_retains_absorbing_tail_while_time_limit_bootstraps():
    discount = .99**10
    costs = torch.full((2, 2), .1, dtype=torch.float64)
    values = torch.tensor([[1., 2., 3.], [1., 2., 3.]], dtype=torch.float64)
    terminal = torch.tensor([[False, True], [False, False]])
    target = lambda_cost_targets(costs, values, discount=discount, lambda_return=.95,
                                terminal=terminal, failure_value=8.)
    failure_end, continuing_end = .1 + discount * 8., .1 + discount * 3.
    assert float(target[0, 1]) == pytest.approx(failure_end)
    assert float(target[1, 1]) == pytest.approx(continuing_end)
    assert float(target[0, 0]) == pytest.approx(.1 + discount * (.05 * 2 + .95 * failure_end))
    assert target[0, 0] > target[1, 0]
    assert not target.requires_grad


def test_measured_failure_returns_raise_the_fitted_cost_value():
    cfg = CriticConfig(hidden_dims=(8,), epochs=1, minibatches=1, learning_rate=.001)
    torch.manual_seed(3)
    clean = CriticLearner(3, cfg, parameter=torch.zeros(1), ctrl_dt=.02, window_steps=10,
                          discount=.99, failure_cost_rate=4.)
    torch.manual_seed(3)
    failed = CriticLearner(3, cfg, parameter=torch.zeros(1), ctrl_dt=.02, window_steps=10,
                           discount=.99, failure_cost_rate=4.)
    contexts, costs = torch.zeros(1, 2, 3), torch.zeros(1, 1)
    terminal = torch.zeros_like(costs, dtype=torch.bool)
    before = float(clean.critic(contexts[:, 0]).detach())
    clean.update(contexts, costs, terminal)
    failed.update(contexts, costs, terminal, observed_contexts=contexts[:, 0],
                  observed_targets=torch.tensor([failed.failure_value]))
    assert float(clean.critic(contexts[:, 0]).detach()) < before
    assert float(failed.critic(contexts[:, 0]).detach()) > before


def test_terminal_critic_gradient_crosses_real_planner_tracker_and_world_model(tmp_path):
    policy, contract, model = physical_inputs(tmp_path)
    rollout = load_rollout(LoadedWorldModel(model, contract, {}), policy)
    actor = LatentResidualActor(model.feature_dim, (8,), (.02,) * 29, command_dim=3)
    learner = ActorLearner(actor, rollout, {"batch_size": 1, "microbatch_size": 1, "window_steps": 2,
        "horizon_windows": 2, "learning_rate": .0001, "max_grad_norm": 10.,
        "critic": {"hidden_dims": (8,), "epochs": 1, "minibatches": 1}}, LatentCostConfig())
    sampler = PretrainingSequenceSampler(tmp_path / "train", rollout_windows=1)
    start = actor_training_start(sampler.sample_indices([0]), rollout,
                                 TrajectoryConfig(initial_lateral_m=.2), seed=1)
    start = learner.refresh_start(start)
    learner.fit_critic(start)
    result = rollout.run(start, actor, window_steps=2, horizon_windows=2)
    terminal = learner.critic_learner.target(result.decision_contexts[:, -1]).mean()
    terminal.backward()
    assert sum(float(p.grad.abs().sum()) for p in actor.parameters() if p.grad is not None) > 0.
    assert all(p.grad is None for p in (*model.parameters(), *rollout.policy.parameters(),
                                      *learner.critic_learner.target.parameters()))


def test_control_keeps_advancing_while_learning_worker_is_paused(tmp_path, monkeypatch):
    case, _, contract, model, factory = parts(tmp_path / "case")
    wm_path = save_world_model(tmp_path / "wm.pt", model, contract)
    node = OmegaConf.load(ROOT / "conf/distill/path_following/latent_wm.yaml")
    node.checkpoint_path = str(wm_path)
    node.pretrained_actor_checkpoint = str(actor_artifact(tmp_path / "actor.pt", model, contract, factory(model)))
    node.pose.odometry_source = "simulation"
    node.update_interval_steps = 2
    node.learning = {"online": {"residual_limit_rad": [.02] * 29, "actor_hidden_dims": [8],
        "calibration_windows": 1, "horizon_windows": 1, "replay_sequence_windows": 1,
        "lora_rank": 2, "critic": {"hidden_dims": [8]}}}
    case.cfg.path_following = node
    session = create_session(case)
    runtime = session.controller
    entered, release = Event(), Event()
    original = runtime.updater.update_prepared

    def paused(prepared):
        entered.set()
        if not release.wait(5):
            raise TimeoutError("control waited on the learning worker")
        return original(prepared)

    monkeypatch.setattr(runtime.updater, "update_prepared", paused)
    try:
        session.step_once()
        session.step_once()
        assert entered.wait(2)
        session.step_once()
        session.step_once()
        assert runtime._steps == 4
        assert not runtime._learning_future.done()
        assert runtime.collector.completed_windows == 2 and len(runtime._pending) == 1
        release.set()
        runtime._finish_learning(wait=True)
        assert runtime.updater.actor_updates == runtime.updater.response_updates == 1
    finally:
        release.set()
        session.close()


def test_stochastic_actor_keeps_deterministic_deployment_and_bounded_exploration():
    with torch.random.fork_rng():
        torch.manual_seed(19)
        actor = LatentResidualActor(4, (8,), (.005,) * 29, command_dim=3, preview_points=2)
        inputs = (torch.zeros(128, 4), torch.zeros(128, 2), torch.zeros(128, 3),
                  torch.zeros(128, 2, 5), torch.zeros(128, 29))
        assert torch.equal(actor(*inputs), torch.zeros(128, 29))
        samples = actor.sample(*inputs)
        assert bool((samples.abs() <= .005).all())
        assert bool((samples > 0).any(0).all()) and bool((samples < 0).any(0).all())
        samples.square().sum().backward()
        assert actor.log_std.grad is not None and bool((actor.log_std.grad > 0).all())
        actor.eval()
        assert torch.equal(actor(*inputs), actor(*inputs))


def test_actor_and_critic_learn_from_the_same_sampled_trajectory(tmp_path, monkeypatch):
    import unilab.algos.torch.sttt.path_following.actor_learner as owner
    from unilab.algos.torch.sttt.path_following.latent_objectives import latent_window_costs, weighted_window_costs

    with torch.random.fork_rng():
        torch.manual_seed(23)
        policy, contract, model = physical_inputs(tmp_path)
        rollout = load_rollout(LoadedWorldModel(model, contract, {}), policy)
        actor = LatentResidualActor(model.feature_dim, (8,), (.02,) * 29, command_dim=3)
        learner = ActorLearner(actor, rollout, {"batch_size": 1, "microbatch_size": 1,
            "window_steps": 2, "horizon_windows": 2, "learning_rate": .0005, "max_grad_norm": 1.,
            "max_updates": 4, "critic": {"hidden_dims": (8,), "epochs": 1, "minibatches": 1}},
            LatentCostConfig())
        sampler = PretrainingSequenceSampler(tmp_path / "train", rollout_windows=1)
        start = actor_training_start(sampler.sample_indices([0]), rollout,
                                     TrajectoryConfig(initial_lateral_m=.2), seed=1)
        captured = {}
        loss_function = owner.latent_actor_loss
        fit = learner.critic_learner.update
        sample = actor.sample
        initial_weights = actor.net[-1].weight.detach().clone()

        def actor_loss(result, part, **kwargs):
            captured["contexts"] = result.decision_contexts.detach().clone()
            with torch.no_grad():
                windows = latent_window_costs(result, part, ctrl_dt=rollout.profile.ctrl_dt,
                    window_steps=2, config=learner.costs, residual_limits=actor.residual_limit_rad29)
                captured["costs"] = weighted_window_costs(windows, learner.costs)
                captured["targets"] = lambda_cost_targets(captured["costs"],
                    learner.critic_learner.target(captured["contexts"]),
                    discount=learner.critic_learner.discount, lambda_return=.95,
                    terminal=torch.zeros_like(captured["costs"], dtype=torch.bool),
                    failure_value=learner.critic_learner.failure_value)
            return loss_function(result, part, **kwargs)

        def fit_critic(contexts, costs, terminal, **kwargs):
            torch.testing.assert_close(contexts, captured["contexts"], rtol=0., atol=0.)
            torch.testing.assert_close(costs, captured["costs"], rtol=0., atol=0.)
            assert not torch.equal(actor.net[-1].weight, initial_weights)
            return fit(contexts, costs, terminal, **kwargs)

        def sample_policy(*inputs):
            assert all(not value.requires_grad for value in inputs)
            return sample(*inputs)

        monkeypatch.setattr(owner, "latent_actor_loss", actor_loss)
        monkeypatch.setattr(learner.critic_learner, "update", fit_critic)
        monkeypatch.setattr(actor, "sample", sample_policy)
        result = learner.update(start)
        assert result["value_target_mean"] == pytest.approx(float(captured["targets"].mean()), rel=1e-6)
        assert result["gradient_norm"] > 0.
        assert all(parameter.grad is None for parameter in (*model.parameters(), *rollout.policy.parameters()))
        saved = learner.state_dict()
        restored = ActorLearner(actor, rollout, {"batch_size": 1, "microbatch_size": 1,
            "window_steps": 2, "horizon_windows": 2, "learning_rate": .0005, "max_grad_norm": 1.,
            "critic": {"hidden_dims": (8,), "epochs": 1, "minibatches": 1}}, learner.costs)
        restored.load_state_dict(saved)
        torch.testing.assert_close(restored.return_moments, learner.return_moments, rtol=0., atol=0.)
        assert restored.schedule_updates == 4


def test_independent_joint_exploration_survives_a_saturated_policy(tmp_path):
    import numpy as np
    from types import SimpleNamespace
    from test_control_diagnostics import paired_run
    from unilab.algos.torch.sttt.path_following.joint_config import CaptureSchedule, TrainingVersions
    from unilab.algos.torch.sttt.path_following.pretraining import TrajectoryDistribution
    from unilab.algos.torch.sttt.world_model.checkpoint import load_world_model
    from unilab.algos.torch.sttt.world_model.joint_collection import BatchedSequenceCollector
    from unilab.algos.torch.sttt.world_model.replay import EpisodeReplay

    with torch.random.fork_rng():
        case, cfg, trajectory, _ = paired_run(tmp_path)
        create_env = case.deps["create_env"]

        def clean_environment(*args, **kwargs):
            env = create_env(*args, **kwargs)
            env.cfg.noise_config = SimpleNamespace(level=0.)
            env.get_sim_times_ns = lambda: np.array([env.get_sim_time_ns()], dtype=np.int64)
            return env

        case.deps["create_env"] = clean_environment
        capture = CaptureSchedule(initial=True, num_envs=1, initial_min_transitions=32,
            validation_episodes=1, steps_per_iteration=32, max_episode_steps=128, window_steps=2,
            residual_amplitude_rad=.001, risk_tilt_limit_rad=.8, command_resampling_s=1.,
            command_ranges=[[.2, .4], [-.1, .1], [-.2, .2]],
            conditions=({"name": "nominal", "source_overrides": []},), random_residual_probability=1.)
        collector = BatchedSequenceCollector(case.cfg, root_dir=ROOT, output_dir=tmp_path / "capture",
            capture=capture, trajectory=TrajectoryDistribution(base=trajectory), num_envs=1,
            stream_offset=0, seed=7, condition="nominal", device="cpu", playback_deps=case.deps)
        try:
            model = load_world_model(tmp_path / "trained/world_model.pt").model
            actor = LatentResidualActor(model.feature_dim, (8,), (.001,) * 29, command_dim=3,
                                        preview_points=trajectory.preview_points)
            with torch.no_grad():
                actor.net[-1].bias.fill_(20.)
            collector.bind(model, actor)
            replay = EpisodeReplay(collector.contract, capacity_transitions=128, rollout_windows=1, seed=3)
            collector.advance(32, replay, versions=TrainingVersions())
            residuals = collector._records[0].sequence()["residuals"]
            assert np.all(np.abs(residuals) <= .00100001)
            assert np.any(residuals < 0) and np.any(residuals > 0)
            np.testing.assert_array_equal(residuals[::2], residuals[1::2])
        finally:
            collector.close()
