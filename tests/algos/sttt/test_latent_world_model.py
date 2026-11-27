"""Small real-module checks over fixed numerical records, not robot dynamics tests."""
from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from test_path_following_runtime import ROOT, create_session, runtime_case
from unilab.algos.torch.distill.fada.official_playback import OfficialFADAPlaybackController
from unilab.algos.torch.distill.fada.runtime_policy import FADAFutureQuery, OfficialFADAPlaybackProfile, get_fada_runtime_spec
from unilab.algos.torch.sttt.path_following.contract import InterventionQueryRecord, PathFollowingWindowSample
from unilab.algos.torch.sttt.path_following.latent_actor import LatentResidualActor
from unilab.algos.torch.sttt.path_following.latent_objectives import LatentCostConfig, latent_actor_loss
from unilab.algos.torch.sttt.path_following.latent_rollout import PlannerTrackerRollout
from unilab.algos.torch.sttt.path_following.online_learning import ActorUpdateContext
from unilab.algos.torch.sttt.path_following.reference import FixedStraightPath
from unilab.algos.torch.sttt.pose_data import ControlInterval, ReceivedPose, TimedPelvisPose
from unilab.algos.torch.sttt.world_model.checkpoint import LoadedWorldModel, load_world_model, save_world_model
from unilab.algos.torch.sttt.world_model.losses import pretraining_loss, world_model_loss
from unilab.algos.torch.sttt.world_model.model import LatentWorldModel, WorldModelConfig
from unilab.algos.torch.sttt.world_model.motion import motion_prefix
from unilab.algos.torch.sttt.world_model.online import LatentOnlineConfig, LatentPathFollowingUpdater
from unilab.algos.torch.sttt.world_model.observation_augmentation import ObservationAugmentationConfig, PoseInputAugmenter
from unilab.algos.torch.sttt.world_model.pretraining_data import PretrainingSequenceSampler
from unilab.algos.torch.sttt.world_model.sequence_collection import CausalEgoHistory
from unilab.algos.torch.sttt.world_model.sequence_data import SequenceDatasetWriter, batch_sequences, iter_sequences
from unilab.algos.torch.sttt.world_model.training import train_world_model


def model_config():
    return WorldModelConfig(embed_dim=16, hidden_dim=16, belief_dim=16, stoch_groups=2, stoch_classes=4,
                            num_blocks=4, encoder_layers=1, decoder_layers=1, prior_layers=1, head_layers=1)


def sequence(seed=0, steps=3):
    rng = np.random.default_rng(seed)
    obs = rng.normal(0., .1, (steps + 1, 64)).astype(np.float32)
    obs[:, -3:] = [0., 0., -1.]
    delta = np.tile(np.array([.01, .002, 0.], np.float32), (steps, 1))
    return {
        "observations": obs, "actions": rng.normal(0., .2, (steps, 29)).astype(np.float32),
        "source_actions": rng.normal(0., .1, (steps, 29)).astype(np.float32),
        "commands": np.tile(np.array([.4, 0., 0., 0., 0., 1., -1.], np.float32), (steps, 1)),
        "residuals": np.zeros((steps, 29), np.float32),
        "is_first": np.array([True] + [False] * steps), "ego_motion": np.zeros((steps + 1, 3), np.float32),
        "ego_valid": np.zeros((steps + 1, 1), bool), "ego_age_s": np.zeros((steps + 1, 1), np.float32),
        "path_delta": delta, "motion_valid": np.ones(steps, bool), "risk": np.zeros((steps, 1), np.float32),
        "risk_valid": np.ones(steps, bool), "terminal": np.zeros(steps, bool),
        "timestamps_ns": np.arange(steps + 1, dtype=np.int64) * 20_000_000,
        "window_start": np.array([0], np.int64), "window_end": np.array([steps], np.int64),
        "window_delta": delta.sum(0, keepdims=True), "window_valid": np.ones(1, bool),
    }


def windowed_sequence(seed=0, steps=7):
    raw = sequence(seed, steps)
    raw["path_delta"][:, 2] = .001
    ends = np.arange(2, steps + 1, 2, dtype=np.int64)
    raw["window_start"], raw["window_end"] = ends - 2, ends
    raw["window_delta"] = np.asarray([motion_prefix(torch.from_numpy(raw["path_delta"][end - 2:end]))[-1].numpy() for end in ends], np.float32).reshape(-1, 3)
    raw["window_valid"] = np.ones(len(ends), bool)
    return raw


def parts(directory, case=None):
    case = runtime_case(directory) if case is None else case
    raw = OmegaConf.to_container(OmegaConf.load(case.cfg.official_playback.profile_path), resolve=True)
    assert isinstance(raw, dict)
    profile = OfficialFADAPlaybackProfile.from_mapping(raw, source_identity=raw["source_identity"])
    profile = replace(profile, gait_clock_mode="continuous")
    OmegaConf.save(OmegaConf.create(asdict(profile)), case.cfg.official_playback.profile_path)
    spec = get_fada_runtime_spec(case.policy)
    assert spec.preprocessing is not None
    contract = {
        "data": {"window_steps": 2, "ctrl_dt": .02, "history_length": 4, "prediction_horizon": 3,
                 "observation_dim": 64, "action_dim": 29},
        "source": {"checkpoint_sha256": hashlib.sha256((Path(directory) / "policy.pt").read_bytes()).hexdigest(),
                   "source_profile": asdict(profile), "preprocessing": dict(spec.preprocessing),
                   "target_joint_names": list(reversed(profile.joint_names)), "target_default_joint_angles": [0.] * 29,
                   "target_action_scale": [.5] * 29, "delta_frame": "previous_control_horizontal_heading",
                   "delta_units": ["m", "m", "rad"], "ego_frame": "previous_measurement_horizontal_heading",
                   "risk_tilt_limit_rad": .8},
    }
    model = LatentWorldModel(model_config()).eval()

    def rollout_factory(physical):
        return PlannerTrackerRollout(physical, case.policy, profile, spec.preprocessing,
                                    target_joint_names=contract["source"]["target_joint_names"],
                                    target_default_joint_angles=[0.] * 29, target_action_scale=[.5] * 29)

    return case, profile, contract, model, rollout_factory


def captured_query(case, profile):
    captured = []

    def remember(query):
        captured.append(FADAFutureQuery(query.observation_history.clone(), query.action_history.clone(),
                                      query.command.clone(), query.history_valid_mask.clone(), query.planner_future.clone(),
                                      query.preprocessing, query.profile))
        return query.planner_future

    controller = OfficialFADAPlaybackController(case.policy, profile=profile,
        target_joint_names=list(reversed(profile.joint_names)), device="cpu", future_transform=remember)
    obs = torch.linspace(-.1, .1, 64).unsqueeze(0)
    obs[:, -3:] = torch.tensor([0., 0., -1.])
    command = torch.tensor([[.4, 0., 0., 0., 0., 1., -1.]])
    source = controller.act_projected(obs, command)
    return captured[0], source


def actor_artifact(path, model, contract, rollout):
    from unilab.algos.torch.sttt.path_following.actor_checkpoint import save_actor_checkpoint
    from test_residual_film_ppo import make_ppo_learner
    from unilab.algos.torch.sttt.path_following.trajectory import TrajectoryConfig
    actor = LatentResidualActor(model.feature_dim, (8,), (.02,) * 29, command_dim=rollout.spec.command_dim,
                                adaptation_dim=16)
    with torch.no_grad():
        actor.net[-1].weight.normal_(std=.01)
    learner = make_ppo_learner(actor, rollout, stage="film", batch_size=1)
    return save_actor_checkpoint(path, actor, model, contract, TrajectoryConfig(),
                                  training_state={}, learner_state=learner.state_dict())


def pose(index, *, y=0., arrival_epoch=1):
    matrix = np.eye(4)
    matrix[:3, 3] = [index * .01, y, .8]
    return TimedPelvisPose(1, arrival_epoch, index * 20_000_000, matrix, True,
                         (index * 20_000_000,) * 2, "fixed_test_frame")


def test_real_action_magnitude_and_prior_belief_derivative():
    torch.manual_seed(31)
    model = LatentWorldModel(model_config()).double()
    state = model.initial(1, dtype=torch.float64)
    action = torch.linspace(-2., 2., 29, dtype=torch.float64).unsqueeze(0).requires_grad_()
    belief = model.prior(state, action)[0].belief
    gradient, = torch.autograd.grad(belief.square().sum(), action)
    direction = torch.linspace(.3, -.2, 29, dtype=torch.float64).unsqueeze(0)
    epsilon = 1e-3
    with torch.no_grad():
        plus = model.prior(state, action + epsilon * direction)[0].belief.square().sum()
        minus = model.prior(state, action - epsilon * direction)[0].belief.square().sum()
    torch.testing.assert_close((gradient * direction).sum(), (plus - minus) / (2 * epsilon), rtol=.02, atol=2e-4)
    assert float(gradient.norm()) > 1e-5
    assert not torch.equal(model.normalize_action(action), model.normalize_action(action * 2))


def test_delayed_raw_motion_is_not_an_early_encoder_input():
    history = CausalEgoHistory()
    history.add((ReceivedPose(pose(0), 20_000_000), ReceivedPose(pose(1, y=.004), 100_000_000)))
    assert not history.at(80_000_000)[1]
    motion, valid, age = history.at(100_000_000)
    assert valid and age == pytest.approx(.08)
    np.testing.assert_allclose(motion, [.01, .004, 0.], atol=1e-7)
    history.add((ReceivedPose(pose(2, arrival_epoch=2), 120_000_000),))
    assert not history.at(120_000_000)[1]


def test_sequence_loss_terminal_mask_and_artifact_identity(tmp_path):
    _, _, contract, model, _ = parts(tmp_path / "case")
    numerical = sequence()
    with SequenceDatasetWriter(tmp_path / "data", contract) as writer:
        writer.append(numerical)
    recovered = list(iter_sequences(tmp_path / "data"))
    torch.testing.assert_close(recovered[0]["actions"], torch.from_numpy(numerical["actions"]))
    batch = batch_sequences(recovered)
    batch["terminal"][:, -1] = True
    batch["risk"][:, -1] = 1.
    torch.manual_seed(9)
    original = world_model_loss(model, batch)
    poisoned = {key: value.clone() for key, value in batch.items()}
    poisoned["observations"][:, -1] = 1000.
    torch.manual_seed(9)
    changed = world_model_loss(model, poisoned)
    for key in ("reconstruction_loss", "prior_reconstruction_loss", "overshoot_loss", "risk_loss"):
        torch.testing.assert_close(original[key], changed[key])
    original["loss"].backward()
    assert model.risk_head.model[-1].weight.grad is not None
    artifact = save_world_model(tmp_path / "wm.pt", model, contract)
    loaded = load_world_model(artifact, expected_contract=contract)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded.model.state_dict()[key], atol=0, rtol=0)
    wrong = deepcopy(contract)
    wrong["source"]["target_action_scale"] = [.3] * 29
    with pytest.raises(ValueError, match="incompatible"):
        load_world_model(artifact, expected_contract=wrong)


def _angular_motion_loss(monkeypatch, predicted_turn, target_turn=0., *, dense=True):
    steps = 10
    raw = sequence(steps=steps)
    raw["path_delta"][:] = (0., 0., target_turn)
    raw["window_delta"][:] = (0., 0., math.atan2(math.sin(steps * target_turn), math.cos(steps * target_turn)))
    raw["motion_valid"][:] = dense
    batch = {key: value.double() if value.is_floating_point() else value
             for key, value in batch_sequences([raw]).items()}
    batch["path_delta"][..., 2] = target_turn
    batch["window_delta"][..., 2] = math.atan2(math.sin(steps * target_turn), math.cos(steps * target_turn))
    model = LatentWorldModel(model_config()).double().eval().requires_grad_(False)
    theta = torch.tensor(predicted_turn, dtype=torch.float64, requires_grad=True)
    monkeypatch.setattr(model, "motion", lambda latent, action: torch.cat(
        (action[:, :2] * 0., theta.expand(action.shape[0], 1)), -1))
    losses = world_model_loss(model, batch, kl_scale=0., reconstruction_scale=0.,
        prior_reconstruction_scale=0., overshoot_scale=0., risk_scale=0.)
    return losses, theta


@pytest.mark.parametrize("dense", [True, False])
@pytest.mark.parametrize("sign", [-1., 1.])
def test_angular_motion_rejects_full_turn_alias_and_corrects_gradient(monkeypatch, dense, sign):
    old_minimum = sign * 2 * math.pi * 10 / 101
    losses, theta = _angular_motion_loss(monkeypatch, old_minimum, dense=dense)
    gradient = torch.autograd.grad(losses["loss"], theta)[0]
    expected_weight = (100 / .02**2 + int(dense) / (.1 * .02)**2) / 3
    assert float(losses["loss"].detach()) == pytest.approx(expected_weight * old_minimum**2)
    assert float(gradient) == pytest.approx(2 * expected_weight * old_minimum)
    closer, _ = _angular_motion_loss(monkeypatch, old_minimum * .99, dense=dense)
    correct, _ = _angular_motion_loss(monkeypatch, 0., dense=dense)
    assert float(closer["loss"].detach()) < float(losses["loss"].detach())
    assert float(correct["loss"].detach()) == pytest.approx(0., abs=1e-12)


@pytest.mark.parametrize("sign", [-1., 1.])
def test_angular_motion_preserves_genuine_dense_turns_beyond_pi(monkeypatch, sign):
    target = sign * .7  # Ten valid increments record a genuine turn beyond 2*pi.
    matching, _ = _angular_motion_loss(monkeypatch, target, target)
    aliased, theta = _angular_motion_loss(monkeypatch, target - sign * 2 * math.pi / 10, target)
    assert float(matching["motion_loss"].detach()) == pytest.approx(0., abs=1e-12)
    assert float(matching["window_loss"].detach()) == pytest.approx(0., abs=1e-12)
    assert float(aliased["window_loss"].detach()) == pytest.approx((2 * math.pi / .02)**2 / 3)
    assert float(torch.autograd.grad(aliased["loss"], theta)[0]) * sign < 0


def test_actual_tracker_rollout_and_actor_credit_are_preserved(tmp_path):
    case, profile, _, model, factory = parts(tmp_path / "case")
    query, source = captured_query(case, profile)
    model.requires_grad_(False)
    rollout = factory(model)
    start = rollout.encode_start(query, errors=torch.tensor([[.2, .1]]), previous_residual=torch.zeros(1, 29))
    history_before = query.observation_history.clone()
    actor = LatentResidualActor(model.feature_dim, [8], [.02] * 29)
    result = rollout.run(start, actor, window_steps=1, horizon_windows=2)
    torch.testing.assert_close(result.source_actions[:, 0], source, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(result.actions[:, 0], rollout.source_to_env(source), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(query.observation_history, history_before, atol=0, rtol=0)
    cost = latent_actor_loss(result, start, ctrl_dt=.02, window_steps=1, config=LatentCostConfig(),
                            residual_limits=actor.residual_limit_rad29)["loss"]
    cost.backward()
    assert sum(float(p.grad.square().sum()) for p in actor.parameters() if p.grad is not None) > 0
    assert all(parameter.grad is None for parameter in (*model.parameters(), *case.policy.parameters()))
    expected_phase = math.sin(2 * math.pi * profile.gait_frequency * profile.ctrl_dt)
    assert result.commands[0, 1, 3].item() == pytest.approx(expected_phase, abs=1e-6)


def test_online_lora_and_context_ownership_and_resume(tmp_path):
    case, profile, contract, model, factory = parts(tmp_path / "case")
    query, source = captured_query(case, profile)
    config = LatentOnlineConfig(tuple([.02] * 29), actor_hidden_dims=(8,), calibration_windows=1,
                               horizon_windows=1, replay_sequence_windows=1, lora_rank=2)
    updater = LatentPathFollowingUpdater(LoadedWorldModel(model, contract, {}), factory(model), config)
    from unilab.algos.torch.sttt.path_following.actor_checkpoint import load_actor_checkpoint
    pretrained, metadata = load_actor_checkpoint(
        actor_artifact(tmp_path / "prior.pt", model, contract, factory(model)), model, contract, config.trajectory)
    updater.actor.load_state_dict(pretrained.state_dict())
    updater.initialize_pretrained(metadata["learner_state"])
    base = {name: p.detach().clone() for name, p in model.named_parameters() if "lora_" not in name}
    actor_before = {name: p.detach().clone() for name, p in updater.actor.named_parameters()}
    actions = factory(model).source_to_env(source).detach().numpy()[0]
    observations = [query.observation_history[0, -1].numpy().copy()]
    for _ in range(2):
        following = observations[-1].copy()
        following[:3] += .01
        observations.append(following)
    controls = tuple(ControlInterval(1, i, i*20_000_000, (i+1)*20_000_000, actions.copy(), query.command[0, :3].numpy()) for i in range(2))
    record = InterventionQueryRecord.from_query(query, query.planner_future, torch.zeros(1, 29))
    sample = PathFollowingWindowSample(0, record, controls, pose(0), pose(2, y=.006), FixedStraightPath.from_initial_pose(pose(0)))
    updater.begin_episode(sample.reference)
    for i in range(2):
        updater.record_transition(i, {"observation": observations[i], "next_observation": observations[i+1],
            "action": actions.copy(), "source_action": source[0].numpy().copy(), "command": query.command[0].numpy().copy(),
            "residual": np.zeros(29, np.float32),
            "ego_motion": np.zeros(3, np.float32), "ego_valid": False, "ego_age_s": 0., "risk": 0., "risk_valid": False}, capacity=16)
    from concurrent.futures import ThreadPoolExecutor
    before_publication = {name: value.clone() for name, value in updater.inference_actor.state_dict().items()}
    prepared = updater.prepare_update([sample], ActorUpdateContext.from_window_start(sample))
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(updater.update_prepared, prepared).result(timeout=30)
    for name, value in before_publication.items():
        torch.testing.assert_close(updater.inference_actor.state_dict()[name], value, rtol=0, atol=0)
    assert updater.publish()
    for name, value in updater.actor.state_dict().items():
        torch.testing.assert_close(updater.inference_actor.state_dict()[name], value, rtol=0, atol=0)
    assert result.b_loss is not None and result.a_loss is not None
    assert updater.response_updates == updater.context_updates == 1
    for name, parameter in model.named_parameters():
        if name in base:
            torch.testing.assert_close(parameter, base[name], atol=0, rtol=0)
    assert all(torch.equal(p, actor_before[name]) for name, p in updater.actor.named_parameters())
    assert torch.count_nonzero(updater.context_adapter.context) > 0
    assert any(bool(p.abs().sum()) for name, p in model.named_parameters() if name.endswith("lora_B.weight"))
    checkpoint = tmp_path / "online.pt"
    updater.save(checkpoint)
    restored = LatentPathFollowingUpdater.restore(checkpoint, factory, expected_contract=contract)
    for name, value in updater.actor.state_dict().items():
        torch.testing.assert_close(value, restored.actor.state_dict()[name], atol=0, rtol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.model.state_dict()[name], atol=0, rtol=0)
    assert restored.response_updates == restored.context_updates == 1
    torch.testing.assert_close(updater.context_adapter.context, restored.context_adapter.context, rtol=0, atol=0)
    torch.testing.assert_close(updater.context_adapter.optimizer.state[updater.context_adapter.context]["exp_avg"],
                               restored.context_adapter.optimizer.state[restored.context_adapter.context]["exp_avg"], rtol=0, atol=0)
    updater.discard_pending()
    assert not updater.replay and not updater.records
    assert updater.calibration_complete
    torch.testing.assert_close(updater.context_adapter.context, restored.context_adapter.context, rtol=0, atol=0)
    old_payload = torch.load(checkpoint, weights_only=True)
    old_payload["schema_version"] = 6
    torch.save(old_payload, tmp_path / "old-online.pt")
    with pytest.raises(ValueError, match="persistent-position-offset"):
        LatentPathFollowingUpdater.restore(tmp_path / "old-online.pt", factory, expected_contract=contract)


def test_public_training_resume_and_whole_capture_split(tmp_path):
    _, _, contract, _, _ = parts(tmp_path / "case")
    for name, seed in (("train", 1), ("validation", 2), ("copied", 1)):
        with SequenceDatasetWriter(tmp_path / name, contract) as writer:
            writer.append(windowed_sequence(seed, steps=6))
    cfg = {"data": {"train_dirs": [str(tmp_path / "train")], "validation_dirs": [str(tmp_path / "validation")]},
           "model": asdict(model_config()),
           "training": {"max_updates": 3, "batch_size": 2, "rollout_windows": 2,
                        "validation_every": 1, "validation_batches": 1, "checkpoint_every": 2},
           "seed": 1, "device": "cpu", "output_dir": str(tmp_path / "continuous")}
    full = train_world_model(cfg)
    split = deepcopy(cfg)
    split["training"]["max_updates"] = 1
    split["output_dir"] = str(tmp_path / "split")
    first = train_world_model(split)
    split["training"].update(max_updates=3, resume_checkpoint=first["checkpoint_path"])
    split["output_dir"] = str(tmp_path / "resumed")
    second = train_world_model(split)
    continuous, resumed = load_world_model(full["checkpoint_path"]), load_world_model(second["checkpoint_path"])

    def exact(actual, expected):
        if isinstance(actual, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(actual, dict):
            assert actual.keys() == expected.keys()
            for key in actual:
                exact(actual[key], expected[key])
        elif isinstance(actual, (tuple, list)):
            assert len(actual) == len(expected)
            for left, right in zip(actual, expected, strict=True):
                exact(left, right)
        else:
            assert actual == expected

    exact(resumed.model.state_dict(), continuous.model.state_dict())
    for key in ("optimizer_state", "optimizer_steps", "rng_state", "sampler_state", "history",
                "validation_indices", "best_metric", "best_step", "best_state_dict"):
        exact(resumed.training_state[key], continuous.training_state[key])
    assert resumed.training_state["optimizer_steps"] == 3
    assert resumed.training_state["best_metric"] == min(row["validation"]["selection_score"] for row in full["history"])
    best = load_world_model(second["best_checkpoint_path"])
    exact(best.model.state_dict(), resumed.training_state["best_state_dict"])
    assert best.training_state["selection_metric"] == "physical_prediction_score"
    for key, source in (("validation_dirs", "train"), ("validation_dirs", "copied")):
        bad = deepcopy(cfg)
        bad["data"][key] = [str(tmp_path / source)]
        bad["output_dir"] = str(tmp_path / f"bad_{source}")
        with pytest.raises(ValueError, match="disjoint" if source == "train" else "copied"):
            train_world_model(bad)
    changed = deepcopy(split)
    changed["training"]["rollout_windows"] = 3
    with pytest.raises(ValueError, match="configuration or data identity"):
        train_world_model(changed)


def test_factory_latent_mode_uses_real_history_and_window_activation(tmp_path):
    case, _, contract, model, factory = parts(tmp_path / "case")
    artifact = save_world_model(tmp_path / "wm.pt", model, contract)
    node = OmegaConf.load(ROOT / "conf/distill/path_following/latent_wm.yaml")
    node.checkpoint_path = str(artifact)
    node.pretrained_actor_checkpoint = str(actor_artifact(tmp_path / "actor.pt", model, contract, factory(model)))
    node.pose.odometry_source = "simulation"
    node.update_interval_steps = 2
    node.learning = {"online": {"residual_limit_rad": [.02] * 29, "actor_hidden_dims": [8], "calibration_windows": 1,
                                "horizon_windows": 1, "replay_sequence_windows": 1, "lora_rank": 2}}
    case.cfg.path_following = node
    session = create_session(case)
    try:
        for index in range(4):
            session.step_once()
            if (index + 1) % 2 == 0:
                session.controller._finish_learning(wait=True)
        runtime = session.controller
        assert runtime.updater.context_updates == runtime.updater.response_updates == 2
        assert runtime.collector.completed_windows == 2
        _, replay = runtime.updater.replay[-1]
        torch.testing.assert_close(torch.from_numpy(replay["actions"][-1]), torch.as_tensor(session.info["current_actions"][0]).float())
        torch.testing.assert_close(torch.from_numpy(replay["source_actions"][-1]), runtime.playback.last_source_action[0])
        assert runtime.risk_diagnostics()["labelled_windows"] == 2
        learned = {name: value.clone() for name, value in runtime.updater.actor.state_dict().items()}
        session.reset()
        assert not runtime.updater.records and not runtime.updater.replay
        for name, value in learned.items():
            torch.testing.assert_close(value, runtime.updater.actor.state_dict()[name], atol=0, rtol=0)
    finally:
        session.close()
    assert all(env.closed for env in case.environments)


def test_latent_evaluation_compares_adaptation_with_frozen_world_model(tmp_path):
    from test_path_following_evaluation import evaluation_case
    from unilab.algos.torch.sttt.path_following.evaluation import run_path_following_evaluation

    case, original, _ = evaluation_case(tmp_path / "evaluation", modes=("latent_wm", "latent_wm_frozen"))
    _, _, contract, model, factory = parts(tmp_path / "evaluation" / "inputs", case=case)
    artifact = save_world_model(tmp_path / "evaluation_wm.pt", model, contract)
    actor_path = actor_artifact(tmp_path / "evaluation_actor.pt", model, contract, factory(model))
    cfg = OmegaConf.merge(
        OmegaConf.load(ROOT / "conf/sttt/latent_path_following_evaluate.yaml"), original,
        {"world_model_checkpoint": str(artifact), "pretrained_actor_checkpoint": str(actor_path),
         "latent_learning": {"online": {"residual_limit_rad": [.02] * 29, "actor_hidden_dims": [8],
                                         "calibration_windows": 1, "horizon_windows": 1,
                                         "replay_sequence_windows": 1, "lora_rank": 2}},
         "thresholds": {"heading_tolerance_rad": math.pi}},
    )
    result = run_path_following_evaluation(cfg, root_dir=ROOT, playback_deps=case.deps)
    assert result["complete"]
    adapted, frozen = result["runs"]
    assert adapted["runtime"]["b_updates"] == 1
    assert frozen["runtime"]["b_updates"] == 0
    assert adapted["runtime"]["context_updates"] == frozen["runtime"]["context_updates"] == 1
    assert adapted["risk_diagnostics"]["labelled_windows"] == frozen["risk_diagnostics"]["labelled_windows"] == 5
    assert all(env.closed and not env.autoreset for env in case.environments)


def test_sequence_collector_keeps_terminal_risk_and_actual_action(tmp_path):
    from test_path_following_runtime import SyntheticWrapper
    from unilab.algos.torch.sttt.world_model.sequence_collection import collect_world_model_sequences

    case, _, _, _, _ = parts(tmp_path / "capture_inputs")
    del case.cfg.path_following
    case.cfg.training.play_checkpoint_path = str(tmp_path / "capture_inputs" / "policy.pt")

    class TerminalWrapper(SyntheticWrapper):
        def step(self, action):
            self.terminal_next = self.env.tick == 3
            return super().step(action)

    def wrapper(env, **kwargs):
        result = TerminalWrapper(env, **kwargs)
        case.wrappers.append(result)
        return result

    case.deps["wrapper_cls"] = wrapper
    result = collect_world_model_sequences(
        case.cfg, root_dir=ROOT, output_dir=tmp_path / "capture", num_steps=4, window_steps=2,
        residual_amplitude_rad=.01, risk_tilt_limit_rad=.8, seed=7, device="cpu", playback_deps=case.deps,
    )
    assert result["episode_terminated"] and result["transitions"] == 4
    captured = list(iter_sequences(tmp_path / "capture"))[0]
    assert captured["terminal"][-1] and captured["risk"][-1, 0] == 1 and captured["risk_valid"][-1]
    torch.testing.assert_close(captured["actions"][-1], torch.as_tensor(case.environments[0].state.info["current_actions"][0]).float())
    assert captured["window_end"].tolist() == [2]
    assert case.environments[0].closed
    augmented = collect_world_model_sequences(
        case.cfg, root_dir=ROOT, output_dir=tmp_path / "augmented", num_steps=4, window_steps=2,
        residual_amplitude_rad=.01, risk_tilt_limit_rad=.8, seed=7, device="cpu", playback_deps=case.deps,
        observation_augmentation={"position_noise_std_m": .02, "yaw_noise_std_rad": .03,
                                  "delay_s": .04, "jitter_s": .005, "dropout_probability": .2, "seed": 11},
    )
    altered = list(iter_sequences(tmp_path / "augmented"))[0]
    for key in captured.keys() - {"ego_motion", "ego_valid", "ego_age_s"}:
        torch.testing.assert_close(altered[key], captured[key], rtol=0, atol=0)
    assert not torch.equal(altered["ego_valid"], captured["ego_valid"])
    provenance = json.loads((tmp_path / "augmented" / "capture.json").read_text())["observation_augmentation"]
    assert augmented["transitions"] == 4 and provenance["applied"]
    assert provenance["recipe"]["delay_s"] == .04
    assert any(record["status"] in {"queued", "released", "dropped"} for record in provenance["receipts"])


def test_pretraining_crops_warm_history_and_mask_terminal_padding(tmp_path):
    _, _, contract, model, _ = parts(tmp_path / "inputs")
    raw = windowed_sequence(11)
    raw["terminal"][-1], raw["risk"][-1, 0], raw["motion_valid"][-1] = True, 1., False
    with SequenceDatasetWriter(tmp_path / "data", contract) as writer:
        writer.append(raw)
    sampler = PretrainingSequenceSampler(tmp_path / "data", rollout_windows=2, seed=4, cache_size=1)
    initial_rng = sampler.state_dict()
    batch = sampler.sample_indices([1, 3])
    assert sampler.state_dict() == initial_rng
    assert [origin for _, origin in batch.origins] == [2, 6]
    torch.testing.assert_close(batch.history_observations[0, 1:], torch.as_tensor(raw["observations"][:2]))
    torch.testing.assert_close(batch.history_previous_actions[0, -1], torch.as_tensor(raw["actions"][0]))
    torch.testing.assert_close(batch.history_previous_actions[1], torch.as_tensor(raw["actions"][2:5]))
    torch.testing.assert_close(batch.previous_action, torch.as_tensor(raw["actions"][[1, 5]]))
    torch.testing.assert_close(batch.sequence["window_delta"][0], torch.as_tensor(raw["window_delta"][1:3]))
    assert batch.sequence["window_start"][0].tolist() == [0, 2]
    assert not bool(batch.sequence["is_first"][:, 0].any())
    assert batch.transition_mask[1].tolist() == [True, False, False, False]
    assert batch.sequence["risk"][1, 0, 0] == 1 and batch.sequence["risk_valid"][1, 0]
    expected_origins = sampler.sample(7).origins
    restored = PretrainingSequenceSampler(tmp_path / "data", rollout_windows=2, seed=99)
    restored.load_state_dict(initial_rng)
    assert restored.sample(7).origins == expected_origins

    batch.history_observations.requires_grad_()
    batch.history_previous_actions.requires_grad_()
    batch.sequence["observations"].requires_grad_()
    torch.manual_seed(9)
    original = pretraining_loss(model, batch)
    original["loss"].backward()
    assert batch.history_observations.grad is None and batch.history_previous_actions.grad is None
    assert float(batch.sequence["observations"].grad[0, 0].abs().sum()) > 0
    assert not bool(batch.sequence["observations"].grad[1, 1:].any())
    assert float(original["risk_loss"]) > 0 and original["window_xy_count"] == 2

    poisoned = {key: value.detach().clone() for key, value in batch.sequence.items()}
    poisoned["observations"][~batch.observation_mask] = 1000.
    poisoned["actions"][~batch.transition_mask] = 20.
    poisoned["path_delta"][~batch.transition_mask] = 1000.
    poisoned["motion_valid"][~batch.transition_mask] = True
    poisoned["risk"][~batch.transition_mask] = 1.
    poisoned["risk_valid"][~batch.transition_mask] = True
    torch.manual_seed(9)
    padded = pretraining_loss(model, replace(batch, sequence=poisoned))
    for key in original:
        torch.testing.assert_close(padded[key], original[key], rtol=0, atol=0)
    future = {key: value.detach().clone() for key, value in batch.sequence.items()}
    future["observations"][:, 1:] += 10.
    torch.manual_seed(9)
    changed_future = pretraining_loss(model, replace(batch, sequence=future))
    for key in ("motion_loss", "window_loss"):
        torch.testing.assert_close(changed_future[key], original[key], rtol=0, atol=0)


def test_pose_augmentation_is_causal_reproducible_and_reset_scoped():
    receipts = tuple(ReceivedPose(pose(index), index * 20_000_000) for index in range(8))
    truths = [received.pose.navigation_from_pelvis.copy() for received in receipts]
    delayed = PoseInputAugmenter(ObservationAugmentationConfig(delay_s=.04))
    delayed.add(receipts[:1])
    assert not delayed.release(39_999_999)
    released = delayed.release(40_000_000)
    assert released[0].pose.timestamp_ns == 0 and released[0].available_at_ns == 40_000_000

    config = ObservationAugmentationConfig(position_noise_std_m=.02, yaw_noise_std_rad=.03,
                                          delay_s=.04, jitter_s=.005, seed=4)
    left, right = PoseInputAugmenter(config), PoseInputAugmenter(config)
    left.add(receipts)
    right.add(receipts)
    all_released = []
    for now in range(0, 300_000_001, 20_000_000):
        actual, repeated = left.release(now), right.release(now)
        assert len(actual) == len(repeated)
        for first, second in zip(actual, repeated, strict=True):
            assert first.available_at_ns == second.available_at_ns == now
            assert first.pose.timestamp_ns <= now
            np.testing.assert_array_equal(first.pose.navigation_from_pelvis, second.pose.navigation_from_pelvis)
        all_released.extend(actual)
    assert len(all_released) == len(receipts)
    assert left.provenance() == right.provenance()
    assert not np.array_equal(all_released[0].pose.navigation_from_pelvis, truths[0])
    for received, truth in zip(receipts, truths, strict=True):
        np.testing.assert_array_equal(received.pose.navigation_from_pelvis, truth)
    dropped = PoseInputAugmenter(ObservationAugmentationConfig(dropout_probability=1.))
    dropped.add(receipts)
    assert not dropped.release(1_000_000_000)
    delayed.add(receipts[1:2])
    delayed.clear()
    delayed.add((ReceivedPose(pose(0, arrival_epoch=2), 0),))
    new_epoch = delayed.release(40_000_000)
    assert len(new_epoch) == 1 and new_epoch[0].pose.tracking_epoch == 2


def legacy_case(directory, case=None):
    """Native schema-5 policy with explicit analytical physical I/O."""
    from unilab.algos.torch.distill.fada.model import FADAArchitectureConfig, FADAPlannerIDMPolicy
    from unilab.algos.torch.distill.fada.checkpoint import load_fada_policy_checkpoint, _canonical_state_dict_sha256
    from unilab.algos.torch.distill.fada.legacy_deployment import LegacyFADAPlaybackProfile, RAW_LAYOUT

    case = runtime_case(directory, enabled=False) if case is None else case
    if "path_following" in case.cfg:
        del case.cfg.path_following
    case.cfg.online_adaptation.enabled = False
    case.cfg.official_playback.profile_path = None
    environment = {
        "ctrl_dt": .02, "sim_dt": .005, "gait_phase_enabled": True, "gait_clock_mode": "continuous",
        "control_config": {"action_scale": .5, "simulate_action_latency": False},
        "reward_config": {"gait_frequency": 1., "scales": {"alive": 1.}},
        "commands": {"resampling_time": 0., "heading_command": False},
        "noise_config": {"level": 0.}, "domain_rand": {}, "runtime_payload": {"enabled": False},
    }
    source = {"kind": "unilab_oracle", "version": 1, "student_observation_contract": "g1_fada_state_v2",
              "environment_config": environment, "control_contract": {
                  "observation_layout": [list(x) for x in RAW_LAYOUT],
                  "actuated_joint_names": [f"j{i}" for i in reversed(range(29))],
                  "ctrl_dt": .02, "sim_dt": .005, "action_scale": .5,
                  "default_joint_pos": [0.] * 29, "kp": [50.] * 29, "kd": [2.] * 29}}
    profile = LegacyFADAPlaybackProfile.from_source(source)
    architecture = FADAArchitectureConfig(66, 29, 3, "g1_fada_state_v2", history_length=4,
        prediction_horizon=3, hidden_dim=16, num_heads=4, planner_layers=1,
        idm_encoder_layers=1, idm_decoder_layers=1, feedforward_dim=32)
    with torch.random.fork_rng():
        torch.manual_seed(7)
        policy = FADAPlannerIDMPolicy(architecture).eval()
    policy.legacy_preprocessing = profile.preprocessing
    checkpoint = case.deps["resolve_checkpoint"](None)
    torch.save({"schema_version": 5, "architecture": asdict(architecture),
        "planner_state_dict": policy.planner.state_dict(), "idm_state_dict": policy.idm.state_dict(),
        "idm_sha256": _canonical_state_dict_sha256(policy.idm.state_dict()),
        "training_schedule": "alternating_idm_then_planner", "completed_iterations": 31, "samples_seen": 16,
        "planner_optimizer_state_dict": torch.optim.AdamW(policy.planner.parameters()).state_dict(),
        "idm_optimizer_state_dict": torch.optim.AdamW(policy.idm.parameters()).state_dict(),
        "runtime_config": {"source_profile": source}, "quality_metrics": {}}, checkpoint)
    case.cfg.training.play_checkpoint_path = str(checkpoint)
    case.deps["load_fada_policy"] = load_fada_policy_checkpoint
    original_wrapper = case.deps["wrapper_cls"]

    class LegacyObservations:
        num_obs = 98
        def __init__(self, env, **kwargs):
            self.env, self.inner = env, original_wrapper(env, **kwargs)
        def observation(self):
            obs = torch.zeros(1, 98)
            obs[:, 3:6] = torch.tensor([0., 0., -1.])
            obs[:, 6:35] = torch.as_tensor(self.env.get_dof_pos()).float()
            obs[:, 64:93] = torch.as_tensor(self.env.state.info["current_actions"]).float()
            obs[:, 93:96] = torch.as_tensor(self.env.state.info["commands"]).float()
            obs[:, 96:98] = torch.as_tensor(self.env.state.info["gait_phase"]).float()
            return {"actor": obs}
        def reset(self):
            _, info = self.inner.reset()
            self.env.state.info["gait_phase"][:] = [0., math.pi]
            self.env.state.info["current_actions"][:] = 0.
            return self.observation(), info
        def step(self, action):
            _, reward, done, info = self.inner.step(action)
            self.env.state.info["current_actions"][:] = action.detach().numpy()
            self.env.state.info["gait_phase"][:] = (np.array([0., math.pi]) + self.env.tick * 2 * math.pi * .02) % (2 * math.pi)
            return self.observation(), reward, done, info
        def close(self):
            self.inner.close()

    case.deps["wrapper_cls"] = LegacyObservations
    case.policy = policy
    return case, profile


def test_legacy_native_zero_residual_history_and_multistep_credit(tmp_path):
    from unilab.algos.torch.distill.fada.playback import FADAPlaybackController
    from unilab.algos.torch.sttt.path_following.intervention import compose_joint_position_residual
    from unilab.algos.torch.sttt.world_model.sequence_collection import proprio_tilt_rad

    case, profile = legacy_case(tmp_path / "inputs")
    queries = []
    def transform(query):
        queries.append(query)
        return compose_joint_position_residual(query.planner_future, torch.zeros(1, 29), ctrl_dt=.02,
            position_scale=1., velocity_scale=.05, observation_contract="g1_fada_state_v2")
    bound = FADAPlaybackController(case.policy, device="cpu", profile=profile, future_transform=transform)
    plain = FADAPlaybackController(case.policy, device="cpu")
    observation = torch.linspace(-.1, .1, 66).reshape(1, 66)
    observation[:, 3:6] = torch.tensor([0., 0., -1.])
    observation[:, 64:66] = torch.tensor([0., math.pi])
    command = torch.tensor([[.4, 0., 0.]])
    for index in range(4):
        if index == 2:
            bound.reset(torch.tensor([True]))
            plain.reset(torch.tensor([True]))
        actual = bound.act_projected(observation, command)
        torch.testing.assert_close(actual, plain.act_projected(observation, command), rtol=0, atol=0)
        observation[:, 6:64] += .001
        observation[:, 64:66] = (observation[:, 64:66] + 2 * math.pi * .02).remainder(2 * math.pi)
    query = queries[-1]
    assert query.history_valid_mask[0].tolist() == [False, False, True, True]
    model = LatentWorldModel(replace(model_config(), obs_dim=66)).eval().requires_grad_(False)
    rollout = PlannerTrackerRollout(model, case.policy, profile, profile.preprocessing,
        target_joint_names=profile.joint_names, target_default_joint_angles=profile.default_joint_pos,
        target_action_scale=profile.action_scale)
    start = rollout.encode_start(query, errors=torch.tensor([[.2, .1]]), previous_residual=torch.zeros(1, 29))
    actor = LatentResidualActor(model.feature_dim, [8], [.02] * 29, command_dim=3)
    phases = []
    handle = case.policy.planner.register_forward_pre_hook(lambda _, args: phases.append(args[0][:, -1, 64:66].detach().clone()))
    try:
        result = rollout.run(start, actor, window_steps=2, horizon_windows=2)
    finally:
        handle.remove()
    torch.testing.assert_close(result.source_actions[:, 0], actual, rtol=1e-6, atol=1e-6)
    for index, phase in enumerate(phases):
        torch.testing.assert_close(phase, (start.observation_history[:, -1, 64:66] + index * 2 * math.pi * .02).remainder(2 * math.pi))
    latent_actor_loss(result, start, ctrl_dt=.02, window_steps=2, config=LatentCostConfig(),
                      residual_limits=actor.residual_limit_rad29)["loss"].backward()
    assert sum(float(p.grad.square().sum()) for p in actor.parameters() if p.grad is not None) > 0
    assert all(p.grad is None for p in (*case.policy.parameters(), *model.parameters()))
    residual = torch.full((1, 29), .02)
    adapted = compose_joint_position_residual(torch.zeros(1, 3, 66), residual, ctrl_dt=.02,
        position_scale=1., velocity_scale=.05, observation_contract="g1_fada_state_v2")
    torch.testing.assert_close(adapted[:, :, 6:35], residual[:, None, :].expand(-1, 3, -1))
    assert torch.count_nonzero(adapted[:, :, 35:64]) == 0
    assert not bool(adapted[:, :, 3:6].any()) and not bool(adapted[:, :, 64:66].any())
    assert proprio_tilt_rad(query.observation_history[0, -1].numpy(), observation_contract="g1_fada_state_v2") < .01


def test_legacy_capture_pretraining_resume_and_runtime_without_yaml(tmp_path):
    from test_path_following_evaluation import evaluation_case
    from unilab.algos.torch.sttt.path_following.evaluation import run_path_following_evaluation
    from unilab.algos.torch.sttt.path_following.pretraining import capture_configuration
    from unilab.algos.torch.sttt.world_model.sequence_collection import collect_world_model_sequences
    from unilab.algos.torch.sttt.world_model.sequence_data import read_sequence_metadata

    base, original, _ = evaluation_case(tmp_path / "case", modes=("nominal", "latent_wm_frozen"))
    case, profile = legacy_case(tmp_path / "case" / "inputs", case=base)
    for name, seed in (("train", 1), ("validation", 2)):
        capture_cfg = capture_configuration(ROOT, Path(case.cfg.training.play_checkpoint_path),
                                            source_overrides=["video_recording.enabled=false"], seed=seed)
        collect_world_model_sequences(capture_cfg, root_dir=ROOT, output_dir=tmp_path / name,
            num_steps=6, window_steps=2, residual_amplitude_rad=.01, risk_tilt_limit_rad=.8,
            seed=seed, device="cpu", playback_deps=case.deps)
    contract = read_sequence_metadata(tmp_path / "train")
    assert contract["data"]["observation_dim"] == 66 and contract["data"]["command_dim"] == 3
    captured = list(iter_sequences(tmp_path / "train"))[0]
    assert captured["observations"].shape == (7, 66) and captured["commands"].shape == (6, 3)
    assert not bool(captured["risk"].any())
    torch.testing.assert_close(captured["actions"], torch.stack(case.wrappers[0].actions)[:, 0])
    options = asdict(model_config())
    options["obs_dim"] = None
    cfg = {"data": {"train_dirs": [str(tmp_path / "train")], "validation_dirs": [str(tmp_path / "validation")]},
           "model": options, "training": {"max_updates": 2, "batch_size": 1, "validation_every": 1,
                                         "validation_batches": 1, "checkpoint_every": 1},
           "seed": 7, "device": "cpu", "output_dir": str(tmp_path / "continuous")}
    full = train_world_model(cfg)
    split = deepcopy(cfg)
    split["output_dir"] = str(tmp_path / "split")
    split["training"]["max_updates"] = 1
    first = train_world_model(split)
    split["training"].update(max_updates=2, resume_checkpoint=first["checkpoint_path"])
    second = train_world_model(split)
    expected, restored = load_world_model(full["checkpoint_path"]), load_world_model(second["checkpoint_path"])
    for key, value in expected.model.state_dict().items():
        torch.testing.assert_close(value, restored.model.state_dict()[key], rtol=0, atol=0)
    assert restored.model.config.obs_dim == 66 and restored.training_state["optimizer_steps"] == 2
    node = OmegaConf.load(ROOT / "conf/distill/path_following/latent_wm.yaml")
    node.checkpoint_path = second["best_checkpoint_path"]
    from unilab.algos.torch.sttt.path_following.pretraining import load_rollout
    best = load_world_model(node.checkpoint_path)
    node.pretrained_actor_checkpoint = str(actor_artifact(tmp_path / "legacy_actor.pt", best.model, best.contract,
        load_rollout(best, Path(case.cfg.training.play_checkpoint_path))))
    node.pose.odometry_source = "simulation"
    node.update_interval_steps = 2
    node.learning = {"online": {"residual_limit_rad": [.02] * 29, "actor_hidden_dims": [8],
                               "calibration_windows": 1, "horizon_windows": 1,
                               "replay_sequence_windows": 1, "lora_rank": 2}}
    case.cfg.path_following = node
    session = create_session(case)
    before = {name: value.clone() for name, value in session.controller.playback.policy.state_dict().items()}
    try:
        for index in range(6):
            session.step_once()
            if (index + 1) % 2 == 0:
                session.controller._finish_learning(wait=True)
        updater = session.controller.updater
        assert updater.actor.command_dim == 3 and updater.context_updates > 0 and updater.response_updates > 0
        for name, value in session.controller.playback.policy.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        updater.save(tmp_path / "online.pt")
        def factory(physical):
            return PlannerTrackerRollout(physical, case.policy, profile, profile.preprocessing,
                target_joint_names=profile.joint_names, target_default_joint_angles=profile.default_joint_pos,
                target_action_scale=profile.action_scale)
        resumed = LatentPathFollowingUpdater.restore(tmp_path / "online.pt", factory, expected_contract=contract)
        for name, value in updater.actor.state_dict().items():
            torch.testing.assert_close(value, resumed.actor.state_dict()[name], rtol=0, atol=0)
    finally:
        session.close()
    evaluation = OmegaConf.merge(OmegaConf.load(ROOT / "conf/sttt/latent_path_following_evaluate.yaml"), original,
        {"profile_path": None, "world_model_checkpoint": second["best_checkpoint_path"],
         "pretrained_actor_checkpoint": node.pretrained_actor_checkpoint,
         "latent_learning": node.learning, "source_overrides": ["path_following.update_interval_steps=2"]})
    # The optional path_following group exists only in the latent mode.
    evaluation.source_overrides = ["++path_following.update_interval_steps=2"]
    result = run_path_following_evaluation(evaluation, root_dir=ROOT, playback_deps=case.deps)
    assert result["complete"] and len(result["runs"]) == 2
    assert all(env.closed for env in case.environments)


def test_legacy_empty_scene_poles_migrate_through_strict_registry(tmp_path):
    from unilab.algos.torch.distill.fada.legacy_deployment import build_legacy_playback_env_override
    from unilab.base.registry import apply_cfg_overrides
    from unilab.base.scene import SceneCfg

    _, original_profile = legacy_case(tmp_path / "inputs")
    source = deepcopy(original_profile.source_profile)
    scene = {"model_file": "/ssd1/cyx/TTT/src/unilab/assets/robots/g1/scene_flat.xml",
             "fragment_files": [], "terrain": None, "task_poles": []}
    source["environment_config"]["scene"] = scene
    profile = replace(original_profile, source_profile=source)
    original = deepcopy(source)
    restored = build_legacy_playback_env_override(profile, {})
    target = SceneCfg(model_file="scene_previous.xml")
    apply_cfg_overrides(target, restored["scene"])
    expected = {key: value for key, value in scene.items() if key != "task_poles"}
    assert asdict(target) == expected
    assert source == original

    scene.pop("task_poles")
    assert build_legacy_playback_env_override(profile, {})["scene"] == expected
    for unsupported in (None, [{"position": [1., 0., 0.]}]):
        scene["task_poles"] = unsupported
        with pytest.raises(ValueError, match="scene.task_poles must be an empty list"):
            build_legacy_playback_env_override(profile, {})
    scene["task_poles"] = []
    scene["unrecognized_scene_field"] = None
    restored = build_legacy_playback_env_override(profile, {})
    with pytest.raises(ValueError, match="has no attribute 'unrecognized_scene_field'"):
        apply_cfg_overrides(SceneCfg(model_file="scene_previous.xml"), restored["scene"])


def test_legacy_full_environment_migration_preserves_execution_contract(tmp_path):
    from unilab.algos.torch.distill.fada.legacy_deployment import build_legacy_playback_env_override
    from unilab.base.registry import apply_cfg_overrides
    from unilab.envs.locomotion.g1.walk_config import G1WalkFlatCfg, G1WalkRewardConfig
    from hydra import compose, initialize_config_dir
    from unilab.envs.locomotion.g1.walk_reward_bindings import G1WalkRewardBindings
    from unilab.envs.locomotion.g1.walk_observation_bindings import G1WalkObservationBindings
    from unilab.envs.locomotion.g1.walk_observation import assemble_walk_observation
    from unilab.training.backend_adapter import BackendAdapter

    _, profile = legacy_case(tmp_path / "inputs")
    source = deepcopy(profile.source_profile)
    environment = asdict(G1WalkFlatCfg())
    environment.update(source["environment_config"])
    environment["reward_config"] = asdict(G1WalkRewardConfig(
        scales={"alive": .15, "base_height": -10., "energy": -2e-5, "feet_clearance": 1.,
                "feet_slide": -.2, "gait": .5, "penalty_action_rate": -.2,
                "penalty_ang_vel_xy": -.5, "penalty_close_feet_xy": -5.,
                "penalty_dof_acc": -2.5e-7, "penalty_dof_pos_limits": -5.,
                "penalty_dof_vel": -.001, "penalty_feet_ori": -2.,
                "penalty_lin_vel_z": -2., "penalty_orientation": -10., "pose": -.5,
                "tracking_ang_vel": 2.5, "tracking_lin_vel": 4.5, "undesired_contacts": -1.},
        tracking_sigma=.25, gait_frequency=profile.gait_frequency,
        feet_phase_swing_height=.09, feet_phase_tracking_sigma=.04,
        base_height_target=.754, min_base_height=.3, max_tilt_deg=65.))
    environment["reward_config"].update({
        "tracking_ang_sigma": .25, "contact_force_threshold": 1., "contact_history_steps": 3,
        "feet_clearance_std": .05, "feet_clearance_tanh_mult": 2., "feet_clearance_target": .05,
        "gait_duty_factor": .55, "gait_period_randomization_width": .2,
        "min_command_threshold": .1, "soft_dof_pos_limit": .95})
    environment["scene"]["task_poles"] = []
    environment["slope_evaluation"] = None
    environment["control_config"]["position_actuator_gains"] = None
    environment["runtime_payload"].update(additional_mass_events=[], requested_mass_levels_kg=[])
    source["environment_config"] = environment
    profile = replace(profile, source_profile=source)
    original = deepcopy(source)
    with initialize_config_dir(config_dir=str(ROOT / "conf/distill"), version_base=None):
        task_cfg = compose(config_name="config", overrides=[
            "task=g1_walk_flat/mujoco", "recording_profile=straight_payload",
            "env.runtime_payload.enabled=false", "online_adaptation.enabled=false"])
    task_overrides = BackendAdapter(task_cfg, root_dir=ROOT, algo_name="distill").build_play_env_cfg_override()
    task_overrides["reward_config"].update(
        gait_frequency=2., base_height_target=.9, min_base_height=.6, max_tilt_deg=45.)
    task_overrides["runtime_payload"]["added_mass_kg"] = 3.
    restored = build_legacy_playback_env_override(profile, task_overrides)
    target = G1WalkFlatCfg()
    apply_cfg_overrides(target, restored)
    target.validate()
    binding = G1WalkRewardBindings()
    binding._reward_cfg = target.reward_config
    binding._init_reward_functions()
    observations = G1WalkObservationBindings()
    observations._cfg = target
    observations._reward_cfg = target.reward_config
    gyro = np.array([[1., 2., 3.]], dtype=np.float32)
    dof_vel = np.arange(1, 30, dtype=np.float32).reshape(1, 29)
    zeros = np.zeros((1, 3), dtype=np.float32)
    joints = np.zeros((1, 29), dtype=np.float32)
    actual = assemble_walk_observation(
        noisy_gyro=gyro, noisy_gravity=zeros, noisy_diff=joints, noisy_dof_vel=dof_vel,
        gyro=gyro, gravity=zeros, diff=joints, dof_vel=dof_vel, last_actions=joints,
        command_obs=zeros, gait_phase=np.zeros((1, 2), dtype=np.float32),
        mode_obs=np.zeros((1, 1), dtype=np.float32), linvel=zeros, mode_observation=False,
        walk_profile=observations._uses_walk_observation_profile(), fada_privileged=False,
        privileged_strength=None, fada_privileged_obs=None, dtype=np.float32)["obs"]
    np.testing.assert_allclose(actual[:, :3], gyro * profile.preprocessing["term_scale"]["base_ang_vel"])
    np.testing.assert_allclose(actual[:, 35:64], dof_vel * profile.preprocessing["term_scale"]["dof_vel"])
    assert (target.ctrl_dt, target.sim_dt, target.control_config.action_scale) == (.02, .005, .5)
    assert target.gait_clock_mode == profile.gait_clock_mode
    assert target.reward_config.gait_frequency == profile.gait_frequency
    assert target.reward_config.scales == task_overrides["reward_config"]["scales"]
    assert target.reward_config.base_height_target == .754
    assert (target.reward_config.min_base_height, target.reward_config.max_tilt_deg) == (.3, 65.)
    assert target.runtime_payload.added_mass_kg == 3.
    assert target.position_actuator_gains == {
        "joint_names": list(profile.joint_names), "kp": list(profile.stiffness), "kd": list(profile.damping)}
    assert "position_actuator_gains" not in restored["control_config"]
    assert source == original

    for section, key, value in (
        (None, "slope_evaluation", {"enabled": True}),
        ("runtime_payload", "additional_mass_events", [{"time_s": 1., "mass_kg": 3.}]),
        ("runtime_payload", "requested_mass_levels_kg", [3.]),
        ("control_config", "position_actuator_gains", {"kp": [1.] * 29}),
    ):
        incompatible = deepcopy(source)
        values = incompatible["environment_config"]
        if section is not None:
            values = values[section]
        values[key] = value
        with pytest.raises(ValueError, match=key):
            build_legacy_playback_env_override(replace(profile, source_profile=incompatible), {})
    incompatible = deepcopy(source)
    incompatible["environment_config"]["reward_config"]["unknown_reward_parameter"] = 1.
    with pytest.raises((TypeError, ValueError), match="unknown_reward_parameter"):
        apply_cfg_overrides(G1WalkFlatCfg(), build_legacy_playback_env_override(
            replace(profile, source_profile=incompatible), {}))
    incompatible_task = deepcopy(task_overrides)
    incompatible_task["reward_config"]["scales"] = {"orientation": -1.}
    with pytest.raises(ValueError, match="observation scaling"):
        build_legacy_playback_env_override(profile, incompatible_task)
    incompatible_task = deepcopy(task_overrides)
    incompatible_task["reward_config"]["scales"]["unknown_reward_term"] = 1.
    target = G1WalkFlatCfg()
    apply_cfg_overrides(target, build_legacy_playback_env_override(profile, incompatible_task))
    binding._reward_cfg = target.reward_config
    with pytest.raises(ValueError, match="unknown_reward_term"):
        binding._init_reward_functions()
