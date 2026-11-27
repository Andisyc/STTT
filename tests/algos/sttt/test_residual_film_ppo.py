"""Native PPO and context adaptation on isolated numerical tasks and real networks."""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from unilab.algos.torch.sttt.path_following.context_adaptation import ContextAdaptationConfig, ContextAdapter
from unilab.algos.torch.sttt.path_following.joint_config import TrainingVersions
from unilab.algos.torch.sttt.path_following.latent_actor import LatentResidualActor
from unilab.algos.torch.sttt.path_following.ppo_training import ResidualPPOConfig, ResidualPPOTrainer
from unilab.algos.torch.sttt.path_following.residual_rewards import ResidualRewardConfig, measured_residual_reward


class DenseResidualTask:
    """A bounded 29-dimensional bandit; it is not a robot-physics substitute."""

    def __init__(self, actor, rollout, *, start=None, batch_size=2):
        self.actor, self.rollout = actor, rollout
        self.device = next(actor.parameters()).device
        self.num_envs = batch_size if start is None else start.command.shape[0]
        self.num_actions = 29
        self.start = start if start is not None else SimpleNamespace(errors=torch.full((self.num_envs, 2), .2))
        self.context_adapter = (ContextAdapter(actor, rollout, batch_size=self.num_envs, window_steps=2,
            config=ContextAdaptationConfig(dimension=actor.adaptation_dim, horizon_windows=1)) if actor.adaptation_dim else None)
        self.physical_control_steps = 0
        self.last_context_metrics = {}
        self.executed = []
        self.refresh(adapt=False)

    def refresh(self, *, adapt=False):
        context = (self.actor.default_context.expand(self.num_envs, -1) if self.context_adapter is None
                   else self.context_adapter.context.detach())
        if hasattr(self.start, "latent"):
            value = self.actor.context(self.start.latent.feature(), self.start.errors, self.start.command,
                                      self.start.path_state.preview(), self.start.previous_residual).detach()
            self.observation = TensorDict({"policy": value, "adaptation_context": context.clone()}, batch_size=[self.num_envs])
        else:
            features = torch.linspace(-.3, .3, self.actor.input_dim, device=self.device).expand(self.num_envs, -1).clone()
            self.observation = TensorDict({"policy": features, "adaptation_context": context.clone()}, batch_size=[self.num_envs])
        return self.observation

    def get_observations(self):
        return self.observation

    def step(self, actions, *, versions):
        residual = self.actor.physical_residual(actions)
        self.executed.append(residual.detach().clone())
        reward = -((residual / self.actor.residual_limit_rad29 - .4).square()).mean(-1)
        self.physical_control_steps += self.num_envs * 2
        following = self.refresh(adapt=False)
        done = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        return following, reward, done, {"terminal_observation": following,
            "time_outs": torch.zeros_like(done), "reward_terms": {"tracking": float(reward.sum())}}


def make_ppo_learner(actor, rollout, *, stage, start=None, config=None, batch_size=2):
    env = DenseResidualTask(actor, rollout, start=start, batch_size=batch_size)
    options = config or ResidualPPOConfig(rollout_windows=2, num_learning_epochs=1, num_mini_batches=1)
    return ResidualPPOTrainer(actor, env, options, stage=stage)


def test_zero_context_preserves_prior_without_destroying_context_gradient():
    with torch.random.fork_rng():
        torch.manual_seed(7)
        actor = LatentResidualActor(8, (16,), [.02] * 29, command_dim=3, preview_points=2)
        head = actor.net[-1]
        assert isinstance(head, torch.nn.Linear)
        with torch.no_grad():
            head.weight.normal_(std=.1)
        inputs = torch.randn(2, actor.input_dim)
        before = actor.raw_action(inputs).detach()
        actor.enable_film(16)
        context = torch.zeros(2, 16, requires_grad=True)
        after = actor.raw_action(inputs, context)
        torch.testing.assert_close(after, before, atol=0, rtol=0)
        gradient, = torch.autograd.grad(after.square().sum(), context)
        assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0
        assert (actor.physical_residual(after).abs() <= .02).all()


def test_native_ppo_preserves_raw_density_and_freezes_prior_during_film_training(tmp_path):
    from test_latent_world_model import parts
    with torch.random.fork_rng():
        torch.manual_seed(11)
        _, _, _, model, factory = parts(tmp_path / "case")
        rollout = factory(model)
        actor = LatentResidualActor(model.feature_dim, (8,), [.02] * 29, command_dim=rollout.spec.command_dim)
        with torch.no_grad():
            actor.log_std.fill_(.7)
        learner = make_ppo_learner(actor, rollout, stage="base")
        head = actor.net[-1]
        assert isinstance(head, torch.nn.Linear)
        before = head.weight.detach().clone()
        learner.update(versions=TrainingVersions())
        storage = learner.algorithm.storage
        assert storage.distribution_params is not None
        mean, std = storage.distribution_params
        expected = torch.distributions.Normal(mean, std).log_prob(storage.actions).sum(-1, keepdim=True)
        torch.testing.assert_close(storage.actions_log_prob, expected)
        torch.testing.assert_close(torch.stack(learner.env.executed), actor.residual_limit_rad29 * storage.actions.tanh())
        assert not torch.equal(before, head.weight)
        actor.enable_film(16)
        film = make_ppo_learner(actor, rollout, stage="film")
        with torch.no_grad():
            film.env.context_adapter.context.fill_(.2)
        sampled = film.env.refresh(adapt=False)
        frozen = {name: value.detach().clone() for name, value in actor.named_parameters() if not name.startswith("film.")}
        phi = actor.film
        assert isinstance(phi, torch.nn.Linear)
        phi_before = phi.weight.detach().clone()
        film.update(versions=TrainingVersions(actor=1))
        assert not torch.equal(phi_before, phi.weight)
        for name, value in frozen.items():
            torch.testing.assert_close(dict(actor.named_parameters())[name], value, atol=0, rtol=0)
        with torch.no_grad():
            film.env.context_adapter.context.fill_(.6)
        film.env.refresh(adapt=False)
        torch.testing.assert_close(sampled["adaptation_context"], torch.full((2, 16), .2))
        state = deepcopy(film.state_dict())
        clone = LatentResidualActor(model.feature_dim, (8,), [.02] * 29, command_dim=rollout.spec.command_dim, adaptation_dim=16)
        restored = make_ppo_learner(clone, rollout, stage="film")
        restored.load_state_dict(state)
        assert restored.algorithm.learning_rate == film.algorithm.learning_rate
        for name, value in actor.state_dict().items():
            torch.testing.assert_close(clone.state_dict()[name], value, atol=0, rtol=0)
        torch.testing.assert_close(restored.env.context_adapter.context, film.env.context_adapter.context)
        assert all(parameter.grad is None for parameter in (*model.parameters(), *rollout.policy.parameters()))


def test_context_update_crosses_real_tracker_and_wm_without_updating_weights(tmp_path):
    from test_latent_pretraining import physical_inputs
    from unilab.algos.torch.sttt.path_following.pretraining import actor_training_start, load_rollout
    from unilab.algos.torch.sttt.path_following.trajectory import TrajectoryConfig
    from unilab.algos.torch.sttt.world_model.checkpoint import LoadedWorldModel
    from unilab.algos.torch.sttt.world_model.pretraining_data import PretrainingSequenceSampler
    with torch.random.fork_rng():
        torch.manual_seed(29)
        policy, contract, model = physical_inputs(tmp_path)
        rollout = load_rollout(LoadedWorldModel(model, contract, {}), policy)
        actor = LatentResidualActor(model.feature_dim, (8,), [.02] * 29, command_dim=3, adaptation_dim=16)
        head = actor.net[-1]
        assert isinstance(head, torch.nn.Linear)
        with torch.no_grad():
            head.weight.normal_(std=.1)
        sampler = PretrainingSequenceSampler(tmp_path / "train", rollout_windows=1)
        start = actor_training_start(sampler.sample_indices([0]), rollout, TrajectoryConfig(initial_lateral_m=.2), seed=1)
        adapter = ContextAdapter(actor, rollout, batch_size=1, window_steps=2,
                                 config=ContextAdaptationConfig(horizon_windows=2))
        owners = (actor, model, rollout.policy)
        before = [{name: value.clone() for name, value in owner.state_dict().items()} for owner in owners]
        metrics = adapter.step(start)
        assert metrics["context_gradient_norm"] > 0 and torch.count_nonzero(adapter.context) > 0
        moment = adapter.optimizer.state[adapter.context]["exp_avg"]
        assert float((adapter.context.detach() * moment).sum()) < 0
        for owner, snapshot in zip(owners, before, strict=True):
            for name, value in snapshot.items():
                torch.testing.assert_close(owner.state_dict()[name], value, atol=0, rtol=0)
            assert all(parameter.grad is None for parameter in owner.parameters())
        saved = deepcopy(adapter.state_dict())
        resumed = ContextAdapter(actor, rollout, batch_size=1, window_steps=2, config=adapter.config)
        resumed.load_state_dict(saved)
        torch.testing.assert_close(resumed.context, adapter.context)
        torch.testing.assert_close(resumed.optimizer.state[resumed.context]["exp_avg"], moment)
        resumed.reset()
        assert torch.count_nonzero(resumed.context) == 0 and not resumed.optimizer.state


def test_measured_dense_reward_uses_physical_limits_acceleration_and_failure():
    position = np.zeros((3, 29))
    position[1, 17] = .5
    velocity = np.zeros_like(position)
    velocity[1, 17] = 2.
    reward, terms, diagnostics = measured_residual_reward(
        joint_positions=position, joint_velocities=velocity, previous_joint_velocities=np.zeros_like(velocity),
        joint_limits=np.tile([-.2, .2], (29, 1)), residual=np.zeros((3, 29)),
        previous_residual=np.zeros((3, 29)), residual_dt=.2,
        command=np.zeros((3, 3)), motion=np.zeros((3, 3)), errors=np.zeros((3, 2)), tilt=np.zeros(3),
        terminated=np.array([False, False, True]), ctrl_dt=.02, config=ResidualRewardConfig())
    assert reward[0] > reward[1] > reward[2]
    assert terms["joint_limit"][1] < terms["joint_limit"][0]
    assert terms["joint_acceleration"][1] < terms["joint_acceleration"][0]
    assert reward[2] - reward[0] == pytest.approx(-200.)
    assert diagnostics["joint_limit_excess_rad"][1] == pytest.approx(.3)


def test_constant_nonzero_trim_is_not_penalized_but_changing_trim_is():
    def evaluate(previous):
        return measured_residual_reward(
            joint_positions=np.zeros((1, 29)), joint_velocities=np.zeros((1, 29)),
            previous_joint_velocities=np.zeros((1, 29)), joint_limits=np.tile([-1., 1.], (29, 1)),
            residual=np.full((1, 29), .005), previous_residual=previous, residual_dt=.2,
            command=np.zeros((1, 3)), motion=np.zeros((1, 3)), errors=np.zeros((1, 2)),
            tilt=np.zeros(1), terminated=np.zeros(1, bool), ctrl_dt=.02, config=ResidualRewardConfig())
    steady, steady_terms, _ = evaluate(np.full((1, 29), .005))
    changing, changing_terms, _ = evaluate(np.zeros((1, 29)))
    assert steady_terms["residual_rate"][0] == 0.
    assert changing_terms["residual_rate"][0] < 0.
    assert steady[0] > changing[0]


def test_timeout_bootstraps_physical_terminal_not_preaction_or_reset_state(tmp_path):
    from test_latent_world_model import parts
    _, _, _, model, factory = parts(tmp_path / "case")
    actor = LatentResidualActor(model.feature_dim, (8,), [.02] * 29, command_dim=7)
    learner = make_ppo_learner(actor, factory(model), stage="base",
        config=ResidualPPOConfig(rollout_windows=1, num_learning_epochs=1, num_mini_batches=1))
    critic = learner.algorithm.critic
    with torch.no_grad():
        for layer in critic.modules():
            if isinstance(layer, torch.nn.Linear):
                layer.weight.zero_()
                layer.bias.zero_()
                layer.weight[0, 0] = 1.
    def observation(value):
        td = learner.env.get_observations().clone()
        td["policy"][:, 0] = value
        return td
    learner.env.observation = observation(1.)
    def terminal_step(actions, *, versions):
        return observation(3.), torch.zeros(2), torch.ones(2, dtype=torch.bool), {
            "terminal_observation": observation(7.), "time_outs": torch.tensor([True, False]), "reward_terms": {}}
    learner.env.step = terminal_step
    learner.update(versions=TrainingVersions())
    torch.testing.assert_close(learner.algorithm.storage.returns[0, :, 0], torch.tensor([.99 * 7., 0.]))


def test_two_stage_entry_reaches_native_ppo_context_and_frozen_wm(tmp_path):
    from dataclasses import asdict
    import json
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from test_latent_pretraining import physical_inputs
    from test_latent_world_model import ROOT, legacy_case
    from unilab.algos.torch.sttt.path_following.actor_checkpoint import load_actor_checkpoint
    from unilab.algos.torch.sttt.path_following.joint_training import _validation_capture, run_joint_pretraining
    from unilab.algos.torch.sttt.path_following.joint_config import JointTrainingConfig
    from unilab.algos.torch.sttt.path_following.pretraining import TrajectoryDistribution
    from unilab.algos.torch.sttt.path_following.trajectory import TrajectoryConfig
    from unilab.algos.torch.sttt.world_model.checkpoint import canonical_contract, load_world_model
    case, profile = legacy_case(tmp_path / "source")
    policy, contract, model = physical_inputs(tmp_path, source=(case, profile))
    wrapped = case.deps["wrapper_cls"]
    def refreshable(env, **kwargs):
        wrapper = wrapped(env, **kwargs)
        env.refresh_state = lambda: env.state
        env.cfg.noise_config = SimpleNamespace(level=0.)
        env.get_sim_times_ns = lambda: np.asarray([env.get_sim_time_ns()], dtype=np.int64)
        def reset_rows(indices):
            np.testing.assert_array_equal(indices, np.array([0]))
            wrapper.reset()
            env.state.truncated[:] = False
        env.reset_rows = reset_rows
        wrapper.get_observations = wrapper.observation
        return wrapper
    case.deps["wrapper_cls"] = refreshable
    with initialize_config_dir(config_dir=str(ROOT / "conf/sttt"), version_base="1.3"):
        cfg = compose(config_name="latent_pretraining")
    cfg.policy_checkpoint, cfg.output_dir = str(policy), str(tmp_path / "joint")
    cfg.capture.initial = False
    cfg.capture.num_envs = 1
    cfg.capture.conditions = [{"name": "nominal", "source_overrides": []}]
    cfg.capture.initial_min_transitions = 4
    cfg.capture.max_episode_steps = 4
    cfg.capture.validation_episodes = 1
    cfg.capture.steps_per_iteration = cfg.capture.window_steps = 2
    cfg.capture.residual_amplitude_rad, cfg.capture.risk_tilt_limit_rad = .005, .8
    cfg.capture.command_ranges = [[-.4, .6], [-.2, .2], [-.4, .4]]
    cfg.wm.model = asdict(model.config)
    cfg.wm.data.train_dirs, cfg.wm.data.validation_dirs = [str(tmp_path / "train")], [str(tmp_path / "validation")]
    cfg.wm.training.batch_size, cfg.wm.training.microbatch_size, cfg.wm.training.rollout_windows = 2, 1, 1
    cfg.wm.training.validation_batches = cfg.wm.training.validation_every = cfg.wm.training.checkpoint_every = 1
    cfg.actor.training.hidden_dims = [8]
    cfg.actor.training.residual_limit_rad = [.02] * 29
    cfg.actor.training.batch_size = cfg.actor.training.horizon_windows = 1
    cfg.ppo.rollout_windows, cfg.ppo.num_learning_epochs, cfg.ppo.num_mini_batches = 2, 1, 1
    for name in cfg.schedule:
        cfg.schedule[name] = 1
    cfg.replay.capacity_transitions = 32
    cfg.evaluation.enabled = True
    cfg.evaluation.num_steps = 4
    cfg.evaluation.seeds = [1]
    cfg.evaluation.trajectories = [{"kind": "straight", "speed_m_s": .35}]
    cfg.evaluation.periodic.num_steps = 2
    cfg.evaluation.periodic.forecast_enabled = False
    options = JointTrainingConfig.from_mapping(cfg)
    heldout = _validation_capture(options.capture, TrajectoryDistribution.from_mapping(cfg.actor.trajectories),
        root=ROOT, policy=policy, profile=None, output=tmp_path / "heldout-capture", seed=99,
        device="cpu", overrides=[], expected_contract=canonical_contract(contract), playback_deps=case.deps)
    assert len(heldout) == 1
    assert json.loads((Path(heldout[0]) / "capture.json").read_text())["split"] == "validation"
    result = run_joint_pretraining(cfg, root_dir=ROOT, playback_deps=case.deps)
    assert result["training_complete"] and result["base_updates"] == result["film_updates"] == 1
    assert result["wm_updates"] == 2 and result["physical_control_steps"] == 8
    assert result["physical_timing"]["wm_horizon_s"] == pytest.approx(.04)
    assert result["latest_pair"]["stage"] == "film"
    assert set(result["best_pairs"]) == {"base", "film"}
    for stage, iteration, mode in (("base", 1, "pretrained_actor"), ("film", 2, "adaptive_context")):
        brief = json.loads((tmp_path / f"joint/evaluation/{stage}_{iteration:07d}/summary.json").read_text())
        full = json.loads((tmp_path / f"joint/evaluation/{stage}_{iteration:07d}_final/summary.json").read_text())
        assert brief["runs"][0][mode]["executed_steps"] == 2
        assert brief["runs"][0][mode]["forecast_diagnostics"] == []
        assert full["runs"][0][mode]["executed_steps"] == 4
        assert full["runs"][0][mode]["forecast_diagnostics"]
    base = load_world_model(tmp_path / "joint/base_world_model.pt")
    final = load_world_model(result["world_model_checkpoint"])
    for name, value in base.model.state_dict().items():
        torch.testing.assert_close(value, final.model.state_dict()[name], atol=0, rtol=0)
    base_actor, _ = load_actor_checkpoint(tmp_path / "joint/base_actor.pt", base.model, base.contract, TrajectoryConfig())
    film_actor, metadata = load_actor_checkpoint(result["actor_checkpoint"], final.model, final.contract, TrajectoryConfig())
    assert metadata["stage"] == "film" and metadata["learner_state"]["context"]["updates"] > 0
    for name, parameter in base_actor.named_parameters():
        torch.testing.assert_close(parameter, dict(film_actor.named_parameters())[name], atol=0, rtol=0)
