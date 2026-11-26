"""Reference geometry and real-network offline learning; no physics substitutes."""
from __future__ import annotations

import hashlib
import math
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from test_latent_world_model import legacy_case, model_config, sequence
from unilab.algos.torch.sttt.path_following.actor_checkpoint import load_actor_checkpoint
from unilab.algos.torch.sttt.path_following.latent_actor import LatentResidualActor
from unilab.algos.torch.sttt.path_following.latent_objectives import LatentCostConfig, latent_actor_loss
from unilab.algos.torch.sttt.path_following.pretraining import actor_training_start, load_rollout, train_latent_actor
from unilab.algos.torch.sttt.path_following.trajectory import PathState, ReferenceTrajectory, TrajectoryConfig
from unilab.algos.torch.sttt.world_model.checkpoint import load_world_model, save_world_model
from unilab.algos.torch.sttt.world_model.model import LatentWorldModel
from unilab.algos.torch.sttt.world_model.motion import MOTION_FRAME, motion_prefix, window_motion
from unilab.algos.torch.sttt.world_model.pretraining_data import PretrainingSequenceSampler
from unilab.algos.torch.sttt.world_model.sequence_data import SequenceDatasetWriter
from unilab.algos.torch.sttt.world_model.training import train_world_model


def physical_inputs(tmp_path, *, source=None):
    case, profile = legacy_case(tmp_path / "source") if source is None else source
    checkpoint = Path(case.cfg.training.play_checkpoint_path)
    contract = {
        "data": {"window_steps": 2, "ctrl_dt": .02, "history_length": 4, "prediction_horizon": 3,
                 "observation_dim": 66, "action_dim": 29, "command_dim": 3, "observation_contract": "g1_fada_state_v2"},
        "source": {"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   "source_profile": asdict(profile), "preprocessing": profile.preprocessing,
                   "target_joint_names": list(profile.joint_names), "target_default_joint_angles": list(profile.default_joint_pos),
                   "target_action_scale": list(profile.action_scale), "delta_frame": MOTION_FRAME,
                   "delta_units": ["m", "m", "rad"], "ego_frame": "previous_measurement_horizontal_heading",
                   "risk_tilt_limit_rad": .8},
    }
    model = LatentWorldModel(replace(model_config(), obs_dim=66))
    for name, seed in (("train", 1), ("validation", 2)):
        raw = sequence(seed, steps=6)
        rng = np.random.default_rng(seed)
        obs = rng.normal(0., .05, (7, 66)).astype(np.float32)
        obs[:, 3:6] = (0., 0., -1.)
        obs[:, 64:66] = (np.arange(7)[:, None] * 2 * math.pi * .02 + np.array([0., math.pi])).astype(np.float32)
        raw["observations"] = obs
        raw["commands"] = raw["commands"][:, :3]
        raw["source_actions"] = raw["actions"].copy()
        ends = np.arange(2, 7, 2, dtype=np.int64)
        raw["window_start"], raw["window_end"] = ends - 2, ends
        raw["window_delta"] = np.stack([motion_prefix(torch.from_numpy(raw["path_delta"][end-2:end]))[-1].numpy() for end in ends])
        raw["window_valid"] = np.ones(3, bool)
        with SequenceDatasetWriter(tmp_path / name, contract) as writer:
            writer.append(raw)
    return checkpoint, contract, model


def test_legacy_archive_import_validates_json_without_deepcopy(tmp_path, monkeypatch):
    import copy
    import json
    from unilab.algos.torch.sttt.path_following.joint_training import _import_datasets
    from unilab.algos.torch.sttt.world_model.checkpoint import canonical_contract
    from unilab.algos.torch.sttt.world_model.replay import EpisodeReplay

    _, contract, _ = physical_inputs(tmp_path)
    expected = json.loads(json.dumps(contract))
    wrong_profile = json.loads(json.dumps(contract))
    wrong_profile["source"]["source_profile"]["ctrl_dt"] *= 2
    wrong_preprocessing = json.loads(json.dumps(contract))
    wrong_preprocessing["source"]["preprocessing"]["term_scale"]["dof_vel"] *= 2
    replay = EpisodeReplay(contract, capacity_transitions=20, rollout_windows=2)

    def unavailable_deepcopy(*args, **kwargs):
        raise TypeError("issubclass() arg 1 must be a class")

    with monkeypatch.context() as patch:
        patch.setattr(copy, "deepcopy", unavailable_deepcopy)
        _import_datasets([tmp_path / "train"], replay)
        assert canonical_contract(contract) == expected
        for invalid in (wrong_profile, wrong_preprocessing):
            with pytest.raises(ValueError, match="source profile/preprocessing metadata disagree"):
                canonical_contract(invalid)
    assert replay.total_collected == 6
    assert replay.sample(1).sequence["actions"].shape == (1, 4, 29)


def test_rotating_frame_windows_are_composed_not_summed():
    motion = torch.tensor([[[1., 0., math.pi / 2], [1., 0., 0.], [0., 1., -.2]]], dtype=torch.float64, requires_grad=True)
    poses = motion_prefix(motion)
    torch.testing.assert_close(poses[0, 2], torch.tensor([1., 1., math.pi / 2], dtype=torch.float64))
    windows = window_motion(motion, torch.tensor([[0, 1]]), torch.tensor([[2, 3]]))
    torch.testing.assert_close(windows[0, 1], torch.tensor([1., 1., -.2], dtype=torch.float64))
    windows.square().sum().backward()
    assert motion.grad is not None
    assert bool(torch.isfinite(motion.grad).all()) and float(motion.grad[0, 0].abs().sum()) > 0


def test_reference_crossing_keeps_the_current_eight_branch():
    xy = torch.tensor([[[-1., -1.], [0., 0.], [1., 1.], [1., -1.], [0., 0.], [-1., 1.]]], dtype=torch.float64)
    cfg = TrajectoryConfig(projection_window_m=.4)
    path = ReferenceTrajectory(xy, torch.ones(1, 6, dtype=torch.float64), cfg)
    first, second = path.arc[0, 1], path.arc[0, 4]
    pose = torch.zeros(1, 3, dtype=torch.float64)
    a, _ = path.project(pose, first.reshape(1))
    b, _ = path.project(pose, second.reshape(1))
    torch.testing.assert_close(a, first.reshape(1))
    torch.testing.assert_close(b, second.reshape(1))
    assert float(b - a) > 4.


def test_body_motion_and_curve_geometry_keep_on_path_motion_on_path():
    cfg = TrajectoryConfig(kind="circle", radius_m=2., samples=4096, length_m=10.)
    reference = ReferenceTrajectory.generate(cfg, dtype=torch.float64)
    delta_angle = cfg.length_m / (cfg.samples - 1) / cfg.radius_m
    theta = .15
    raw = torch.tensor([cfg.radius_m * math.sin(theta), cfg.radius_m * (1 - math.cos(theta))], dtype=torch.float64)
    c, s = math.cos(delta_angle / 2), math.sin(delta_angle / 2)
    end = torch.tensor([[c * raw[0] + s * raw[1], -s * raw[0] + c * raw[1], theta - delta_angle / 2]], dtype=torch.float64)
    state = PathState(reference, torch.zeros(1, 3, dtype=torch.float64), torch.zeros(1, dtype=torch.float64))
    following = state.advance(end)
    assert abs(float(following.errors()[0, 0])) < 1e-5
    assert abs(float(following.errors()[0, 1])) < .002
    assert float(following.progress) == pytest.approx(cfg.radius_m * theta, abs=1e-5)
    preview = following.preview()
    assert preview.shape == (1, cfg.preview_points, 5)
    assert float(preview[0, -1, 1]) > .1


def test_causal_actor_start_and_multistep_tracker_gradients(tmp_path):
    checkpoint, contract, model = physical_inputs(tmp_path)
    artifact = save_world_model(tmp_path / "wm.pt", model, contract)
    loaded = load_world_model(artifact)
    rollout = load_rollout(loaded, checkpoint)
    loaded.model.requires_grad_(False)
    sampler = PretrainingSequenceSampler(tmp_path / "train", rollout_windows=1)
    batch = sampler.sample_indices([0])
    cfg = TrajectoryConfig(kind="circle", initial_lateral_m=.2, radius_m=2.)
    torch.manual_seed(12)
    start = actor_training_start(batch, rollout, cfg, seed=1)
    poisoned = {name: value.clone() for name, value in batch.sequence.items()}
    poisoned["observations"][:, 1:] = 1000.
    poisoned["path_delta"][:] = 1000.
    torch.manual_seed(12)
    other = actor_training_start(replace(batch, sequence=poisoned), rollout, cfg, seed=1)
    torch.testing.assert_close(start.latent.feature(), other.latent.feature(), rtol=0, atol=0)
    torch.testing.assert_close(start.observation_history[:, :3], start.observation_history[:, -1:].expand(-1, 3, -1))
    actor = LatentResidualActor(model.feature_dim, (8,), (.02,) * 29, command_dim=3)
    result = rollout.run(start, actor, window_steps=2, horizon_windows=2)
    cost = latent_actor_loss(result, start, ctrl_dt=.02, window_steps=2, config=LatentCostConfig(),
                            residual_limits=actor.residual_limit_rad29)["loss"]
    cost.backward()
    assert sum(float(parameter.grad.abs().sum()) for parameter in actor.parameters() if parameter.grad is not None) > 0
    assert all(parameter.grad is None for parameter in (*loaded.model.parameters(), *rollout.policy.parameters()))
    assert bool(torch.isfinite(result.body_delta).all())
    assert not torch.equal(result.commands[:, 0], result.commands[:, -1])


def test_native_ppo_actor_artifact_preserves_weights_and_model_binding(tmp_path):
    from test_residual_film_ppo import make_ppo_learner
    from unilab.algos.torch.sttt.path_following.actor_checkpoint import save_actor_checkpoint
    from unilab.algos.torch.sttt.path_following.joint_config import TrainingVersions
    from unilab.algos.torch.sttt.world_model.checkpoint import LoadedWorldModel
    policy, contract, model = physical_inputs(tmp_path)
    rollout = load_rollout(LoadedWorldModel(model, contract, {}), policy)
    actor = LatentResidualActor(model.feature_dim, (8,), (.02,) * 29, command_dim=3)
    learner = make_ppo_learner(actor, rollout, stage="base")
    learner.update(versions=TrainingVersions())
    checkpoint = save_actor_checkpoint(tmp_path / "actor.pt", actor, model, contract, TrajectoryConfig(),
                                      training_state={}, learner_state=learner.state_dict())
    restored, metadata = load_actor_checkpoint(checkpoint, model, contract, TrajectoryConfig())
    resumed = make_ppo_learner(restored, rollout, stage="base")
    resumed.load_state_dict(metadata["learner_state"])
    for name, value in actor.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    assert torch.count_nonzero(restored.net[-1].weight) > 0
    with torch.no_grad():
        next(model.parameters()).add_(.01)
    with pytest.raises(ValueError, match="another WM base"):
        load_actor_checkpoint(checkpoint, model, contract, TrajectoryConfig())




def _raw_capture(directory):
    from unilab.algos.torch.sttt.world_model.sequence_data import iter_sequences
    return {key: value.numpy() for key, value in next(iter_sequences(directory)).items()}


def _physical_prefix(raw, steps, contract):
    from unilab.algos.torch.sttt.world_model.sequence_data import _WINDOW, _sequence_specs
    state, transition = _sequence_specs(contract)
    complete = raw["window_end"] <= steps
    return {key: value[:steps + 1].copy() if key in state else value[:steps].copy() if key in transition else value[complete].copy()
            for key, value in raw.items() if key in state or key in transition or key in _WINDOW}


def test_batch_physical_clocks_and_pose_resets_are_independent():
    from unilab.algos.torch.sttt.odometry_runtime import BatchSimulationPoseRuntime
    from unilab.base.backend.mujoco.backend import MuJoCoBackend

    backend = object.__new__(MuJoCoBackend)
    backend._num_envs = 2
    backend._physics_state = np.zeros((2, 4), dtype=np.float64)
    backend._physics_state[:, 0] = (.02, .04)

    class Rows:
        def get_sim_times_ns(self):
            return backend.get_sim_times_ns()

        def get_base_pos(self):
            return np.array([[1., 0., .8], [2., 0., .8]])

        def get_base_quat(self):
            return np.array([[1., 0., 0., 0.], [1., 0., 0., 0.]])

    runtime = BatchSimulationPoseRuntime(Rows(), 2)
    first = runtime.sample()
    assert [pose.timestamp_ns for pose in first] == [20_000_000, 40_000_000]
    with pytest.raises(ValueError, match="exactly one"):
        backend.get_sim_time_ns()
    runtime.reset(np.array([0], dtype=np.int64))
    backend._physics_state[0, 0] = 0.
    second = runtime.sample()
    assert (second[0].sequence_id, second[0].tracking_epoch, second[0].timestamp_ns) == (2, 1, 0)
    assert second[1] is first[1]
    backend._physics_state[1, 0] = .03
    with pytest.raises(ValueError, match="backwards"):
        runtime.sample()


def test_session_partial_reset_preserves_other_tracker_history(tmp_path):
    from unilab.algos.torch.distill.fada.playback import FADAPlaybackController
    from unilab.algos.torch.sttt.world_model.checkpoint import LoadedWorldModel
    from unilab.visualization.playback_sessions import FADAPlaybackSession

    policy, contract, model = physical_inputs(tmp_path)
    rollout = load_rollout(LoadedWorldModel(model, contract, {}), policy)
    controller = FADAPlaybackController(rollout.policy, device="cpu", profile=rollout.profile)
    reference = FADAPlaybackController(rollout.policy, device="cpu", profile=rollout.profile)
    observations = torch.zeros(2, 66)
    observations[:, 5] = -1.
    command = torch.tensor([[.4, 0., 0.], [.4, 0., 0.]])
    controller.act_projected(observations, command)
    reference.act_projected(observations, command)

    class Rows:
        def reset_rows(self, indices):
            observations[indices, 6] = .2

        def get_observations(self):
            return {"actor": observations}

    rows = Rows()
    session = FADAPlaybackSession(controller=controller, env=rows, wrapped_env=rows,
                                  device="cpu", action_mode="policy", num_envs=2)
    session.command_control_steps[:] = [3, 9]
    session.step_count = 12
    session.reset_rows(np.array([0], dtype=np.int64))
    actual, snapshot = controller.act_projected_with_snapshot(observations, command)
    expected, expected_snapshot = reference.act_projected_with_snapshot(observations, command)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    torch.testing.assert_close(snapshot.observation_history[1], expected_snapshot.observation_history[1], rtol=0, atol=0)
    torch.testing.assert_close(snapshot.observation_history[0], observations[0].expand_as(snapshot.observation_history[0]))
    assert session.command_control_steps.tolist() == [0, 9]
    assert session.step_count == 12


def test_collector_records_command_refreshed_tracker_observations(tmp_path):
    from types import SimpleNamespace

    from unilab.algos.torch.distill.fada.playback import FADAPlaybackController
    from unilab.algos.torch.sttt.path_following.joint_config import TrainingVersions
    from unilab.algos.torch.sttt.pose_data import TimedPelvisPose
    from unilab.algos.torch.sttt.world_model.checkpoint import LoadedWorldModel
    from unilab.algos.torch.sttt.world_model.joint_collection import BatchedSequenceCollector, _Episode
    from unilab.envs.locomotion.g1.walk_config import G1WalkFlatCfg
    from unilab.envs.locomotion.g1.walk_control_bindings import G1WalkControlBindings
    from unilab.envs.locomotion.g1.walk_observation import assemble_walk_observation
    from unilab.visualization.playback_sessions import FADAPlaybackSession

    policy, contract, model = physical_inputs(tmp_path)
    rollout = load_rollout(LoadedWorldModel(model, contract, {}), policy)
    queries = []

    def capture(query):
        queries.append(query.observation_history[:, -1].detach().cpu().numpy().copy())
        return query.planner_future

    controller = FADAPlaybackController(rollout.policy, device="cpu", profile=rollout.profile,
                                      future_transform=capture)

    class RecordedSensors(G1WalkControlBindings):
        """Fixed sensor samples; exercise command publication without advancing physics."""

        def __init__(self):
            self._cfg = G1WalkFlatCfg()
            self._cfg.gait_clock_mode = "command_fixed"
            self._cfg.command_gated_phase_contact.enabled = False
            self._cfg.command_gated_phase_contact.stand_phase = (0., 0.)
            self._cfg.command_gated_phase_contact.startup_phase = (0., math.pi)
            self.state = SimpleNamespace(info={
                "commands": np.zeros((2, 3), np.float32),
                "gait_phase": np.zeros((2, 2), np.float32),
                "command_is_null": np.ones(2, bool),
            })
            self.refresh_state()

        def refresh_state(self):
            self._synchronize_external_command_for_observation(self.state.info)
            vector = np.zeros((2, 3), np.float32)
            gravity = np.tile(np.array([0., 0., 1.], np.float32), (2, 1))
            joints = np.zeros((2, 29), np.float32)
            self.state.obs = assemble_walk_observation(
                noisy_gyro=vector, noisy_gravity=gravity, noisy_diff=joints, noisy_dof_vel=joints,
                gyro=vector, gravity=gravity, diff=joints, dof_vel=joints, last_actions=joints,
                command_obs=self.state.info["commands"], gait_phase=self.state.info["gait_phase"],
                mode_obs=np.zeros((2, 1), np.float32), linvel=vector, mode_observation=False,
                walk_profile=True, fada_privileged=False, privileged_strength=None,
                fada_privileged_obs=None, dtype=np.dtype(np.float32))

        def get_observations(self):
            return {"actor": torch.from_numpy(self.state.obs["obs"])}

    sensors = RecordedSensors()
    session = FADAPlaybackSession(controller=controller, env=sensors, wrapped_env=sensors,
                                  device="cpu", action_mode="policy", num_envs=2)
    session.refresh_observation()
    collector = object.__new__(BatchedSequenceCollector)
    collector.controller, collector.session = controller, session
    collector._command_template = torch.zeros(2, 3)
    reference = ReferenceTrajectory.generate(TrajectoryConfig(speed_m_s=.4), batch_size=2)
    collector._path = PathState(reference, torch.zeros(2, 3), torch.zeros(2))
    initial = collector._decision_observation()
    np.testing.assert_array_equal(initial[:, 64:66], np.tile(np.array([0., math.pi], np.float32), (2, 1)))
    poses = lambda step: [TimedPelvisPose(row, 0, step * 20_000_000, np.eye(4)) for row in range(2)]
    records = [_Episode(contract, 4, initial[row], poses(0)[row], str(row), 2, np.zeros(2))
               for row in range(2)]

    for step in range(2):
        actions = controller.act(session.obs, sensors.state.info["commands"]).detach().cpu().numpy()
        command = sensors.state.info["commands"].copy()
        # A task's post-step command resampling can reset phase before the collector
        # restores its external command for the next decision at the same timestamp.
        session.set_external_command(np.zeros((2, 3), np.float32))
        following = collector._decision_observation()
        for row, record in enumerate(records):
            record.append(observation=queries[-1][row], command=command[row], action=actions[row],
                          source_action=actions[row], residual=np.zeros(29), next_observation=following[row],
                          next_pose=poses(step + 1)[row], terminal=False, risk=0.,
                          versions=TrainingVersions(), next_errors=np.zeros(2))
    for row, record in enumerate(records):
        np.testing.assert_array_equal(record.sequence()["observations"][:-1],
                                      np.stack([query[row] for query in queries]))
        assert record.steps == 2


@pytest.mark.parametrize("invalid", ["count", "pair", "value"])
def test_replay_failure_snapshot_preserves_exact_rejected_input(tmp_path, invalid):
    import pickle
    from types import SimpleNamespace

    from unilab.algos.torch.sttt.world_model.joint_collection import BatchedSequenceCollector
    from unilab.algos.torch.sttt.world_model.replay import EpisodeReplay, ReplayVersionError

    _, contract, _ = physical_inputs(tmp_path)
    raw = _raw_capture(tmp_path / "train")
    versions = [[0, 0] for _ in range(len(raw["actions"]))]
    replay = EpisodeReplay(contract, capacity_transitions=100, rollout_windows=2)
    metadata = {"stream_id": 0, "condition": "nominal", "versions_actor_wm": deepcopy(versions[:3])}
    replay.update_episode("failed", _physical_prefix(raw, 3, contract), metadata=metadata)
    if invalid == "count":
        versions.pop()
    elif invalid == "pair":
        versions[3] = [0]
    else:
        versions[3] = [0.0, 0]
    collector = object.__new__(BatchedSequenceCollector)
    collector.output, collector.contract = tmp_path / "capture", contract
    collector.stream_offset, collector.condition = 0, "nominal"
    collector._records = [SimpleNamespace(identity="failed", validation=False, versions=versions, sequence=lambda: raw)]

    with pytest.raises(ReplayVersionError) as failure:
        collector._store(replay, 0)

    assert failure.value.snapshot_path is not None
    with Path(failure.value.snapshot_path).open("rb") as handle:
        snapshot = pickle.load(handle)
    assert snapshot["identity"] == "failed"
    assert snapshot["metadata"]["versions_actor_wm"] == versions
    assert snapshot["input_metadata"]["versions_actor_wm"] == versions
    if invalid == "value":
        assert type(snapshot["metadata"]["versions_actor_wm"][3][0]) is float
    for key, values in raw.items():
        np.testing.assert_array_equal(snapshot["sequence"][key], values)
    assert replay.transitions == replay.total_collected == 3
    assert replay.episode_metadata("failed") == metadata

    restored = EpisodeReplay(snapshot["contract"], capacity_transitions=100, rollout_windows=2)
    with pytest.raises(ReplayVersionError) as reproduced:
        restored.update_episode(snapshot["identity"], snapshot["sequence"], metadata=snapshot["metadata"])
    assert reproduced.value.action_count == len(raw["actions"])
    assert str(reproduced.value) == snapshot["error"]
    assert reproduced.value.metadata == failure.value.metadata
    repaired = {**metadata, "versions_actor_wm": [[0, 0] for _ in range(len(raw["actions"]))]}
    replay.update_episode("failed", raw, metadata=repaired)
    assert replay.transitions == len(raw["actions"])


def test_live_replay_growing_prefixes_terminal_tails_and_resume(tmp_path):
    from unilab.algos.torch.sttt.world_model.replay import EpisodeReplay

    _, contract, _ = physical_inputs(tmp_path)
    raw = _raw_capture(tmp_path / "train")
    replay = EpisodeReplay(contract, capacity_transitions=9, rollout_windows=2, seed=9)
    metadata = {"stream_id": 0, "condition": "nominal", "versions_actor_wm": [[0, 0]] * 3}
    replay.update_episode("first", _physical_prefix(raw, 3, contract), metadata=metadata)
    metadata["versions_actor_wm"] += [[1, 2]] * 3
    replay.update_episode("first", raw, metadata=metadata)
    assert replay.total_collected == replay.transitions == 6
    assert replay.episode_metadata("first")["versions_actor_wm"] == [[0, 0]] * 3 + [[1, 2]] * 3
    archive = PretrainingSequenceSampler(tmp_path / "train", rollout_windows=2)
    a, b = archive.sample_indices([0, 1, 2]), replay.sample_indices([0, 1, 2])
    for key, value in a.sequence.items():
        torch.testing.assert_close(value, b.sequence[key])
    torch.testing.assert_close(a.history_previous_actions, b.history_previous_actions)
    torch.testing.assert_close(a.previous_action, b.previous_action)
    short = _physical_prefix(raw, 2, contract)
    short["terminal"][-1] = True
    short["risk"][-1] = 1.
    short["motion_valid"][-1] = False
    short["window_valid"][:] = False
    replay.update_episode("terminal", short)
    terminal = replay.sample_indices([3])
    assert terminal.sequence["terminal"][0, 1] and terminal.sequence["risk"][0, 1, 0] == 1
    assert int(terminal.transition_mask.sum()) == 2
    assert not bool(terminal.history_valid_mask.any())
    assert torch.count_nonzero(terminal.previous_action) == 0
    replay.update_episode("third", _physical_prefix(raw, 2, contract))
    assert replay.transitions == 4 and replay.total_collected == 10
    with pytest.raises(ValueError, match="identity"):
        replay.update_episode("first", raw)
    path = tmp_path / "replay.pt"
    torch.save(replay.state_dict(), path)
    restored = EpisodeReplay(contract, capacity_transitions=9, rollout_windows=2, seed=0)
    restored.load_state_dict(torch.load(path, weights_only=True))
    original_batch, restored_batch = replay.sample(7), restored.sample(7)
    assert original_batch.origins == restored_batch.origins
    torch.testing.assert_close(original_batch.sequence["observations"], restored_batch.sequence["observations"])


def test_command_references_keep_their_anchor_and_cover_preview():
    from unilab.algos.torch.sttt.path_following.command_reference import sample_command_reference

    cfg = TrajectoryConfig(preview_points=4, preview_spacing_m=.1)
    reference, program = sample_command_reference(cfg, [[.4, .4], [0., 0.], [.2, .2]],
        episode_steps=10, horizon_steps=4, ctrl_dt=.02, resampling_s=.1, batch_size=2,
        rng=np.random.default_rng(11), lateral_range=(.2, .2), heading_range=(0., 0.))
    original = reference.xy.clone()
    start = PathState(reference, torch.zeros(2, 3), torch.zeros(2))
    moved = start.advance(torch.tensor([[.05, .01, .03], [.04, -.01, .02]]))
    assert moved.preview().shape == (2, 4, 5)
    torch.testing.assert_close(reference.xy, original, rtol=0, atol=0)
    assert float(reference.arc[:, -1].min()) > 4 * .1 + .4 * .02 * 14
    assert program["resampling_control_steps"] == 5
    assert float(start.errors()[0, 0]) == pytest.approx(.2, abs=1e-5)
    assert float(reference.xy[0, 10, 1]) > -.2


def test_bounded_commands_survive_imagined_motion_and_path_batching():
    from unilab.algos.torch.sttt.path_following.command_reference import sample_command_reference
    from unilab.algos.torch.sttt.path_following.joint_data import select_path
    bounds = [[0., 1.], [-.4, .4], [-.8, .8]]
    cfg = TrajectoryConfig(preview_points=4, preview_spacing_m=.15)
    reference, _ = sample_command_reference(cfg, bounds, episode_steps=600, horizon_steps=20,
        ctrl_dt=.02, resampling_s=.4, batch_size=2, rng=np.random.default_rng(91))
    progress = reference.arc[:, :-1].clone()
    _, tangent, speed = reference.sample(progress)
    ahead = reference.sample(progress + cfg.preview_spacing_m)[1]
    turn = torch.atan2(ahead[..., 1], ahead[..., 0]) - torch.atan2(tangent[..., 1], tangent[..., 0])
    turn = torch.atan2(turn.sin(), turn.cos())
    assert bool((speed * turn.abs() / cfg.preview_spacing_m <= .80001).all())
    for index in range(0, progress.shape[1], 7):
        pose = torch.cat((reference.xy[:, index], torch.full((2, 1), .3)), -1)
        path = PathState(reference, pose, progress[:, index])
        for candidate in (path, select_path(path, slice(0, 1)), path.advance(torch.zeros(2, 3))):
            commands = candidate.command(torch.zeros(candidate.pose.shape[0], 3))
            assert bool((commands >= torch.tensor(bounds)[:, 0] - 1e-6).all())
            assert bool((commands <= torch.tensor(bounds)[:, 1] + 1e-6).all())


def test_launch_reuses_all_conditions_without_training_on_heldout_prefixes(tmp_path):
    import json
    import runpy
    root = Path(__file__).resolve().parents[3]
    checkpoint = tmp_path / "policy.pt"
    checkpoint.touch()
    rows = [
        ("capture/nominal/short", "train", "same-episode", 2),
        ("capture/nominal/full", "train", "same-episode", 8),
        ("capture/payload/full", "train", "payload-episode", 8),
        ("capture/friction/heldout", "validation", "heldout-episode", 8),
        ("validation/full", "validation", "initial-heldout", 8),
    ]
    for name, split, episode, count in rows:
        directory = tmp_path / name
        directory.mkdir(parents=True)
        (directory / "metadata.json").write_text(json.dumps({"transition_count": count}))
        (directory / "capture.json").write_text(json.dumps({"split": split, "episode_ids": [episode]}))
    command = runpy.run_path(str(root / "scripts/launch_latent_pretraining.py"))["training_command"](
        tmp_path, checkpoint, tmp_path / "output", "uv")
    arguments = dict(value.split("=", 1) for value in command if "=" in value)
    assert set(json.loads(arguments["wm.data.train_dirs"])) == {
        str(tmp_path / "capture/nominal/full"), str(tmp_path / "capture/payload/full")}
    assert set(json.loads(arguments["wm.data.validation_dirs"])) == {
        str(tmp_path / "capture/friction/heldout"), str(tmp_path / "validation/full")}


def test_validation_episode_never_enters_training_replay(tmp_path):
    from types import SimpleNamespace
    from unilab.algos.torch.sttt.world_model.joint_collection import BatchedSequenceCollector
    from unilab.algos.torch.sttt.world_model.replay import EpisodeReplay
    _, contract, _ = physical_inputs(tmp_path)
    collector = object.__new__(BatchedSequenceCollector)
    collector.stream_offset, collector.condition = 0, "payload"
    collector.validation_replay = EpisodeReplay(contract, capacity_transitions=32, rollout_windows=1)
    training = EpisodeReplay(contract, capacity_transitions=32, rollout_windows=1)
    raw = _raw_capture(tmp_path / "train")
    collector._records = [SimpleNamespace(identity="held-out-from-reset", validation=True,
        sequence=lambda: raw, versions=[[0, 0]] * len(raw["actions"]))]
    collector._store(training, 0)
    assert training.transitions == 0
    assert collector.validation_replay.transitions == len(raw["actions"])
    longer = _raw_capture(tmp_path / "train")
    collector._records[0].sequence = lambda: longer
    collector._store(training, 0)
    assert training.transitions == 0


def test_velocity_readout_integrates_the_saved_physical_period(tmp_path):
    from unilab.algos.torch.sttt.world_model.model import LatentWorldModel
    _, contract, _ = physical_inputs(tmp_path)
    model = LatentWorldModel(replace(model_config(), obs_dim=66, ctrl_dt=.01)).eval()
    with torch.no_grad():
        for parameter in model.motion_head.parameters():
            parameter.zero_()
        model.motion_head.model[-1].bias.copy_(torch.tensor([.5, -.2, .3]))
    actual = model.motion(model.initial(1), torch.zeros(1, 29))
    torch.testing.assert_close(actual, torch.tensor([[.005, -.002, .003]]))
    with pytest.raises(ValueError, match="integration period"):
        save_world_model(tmp_path / "wrong-clock.pt", model, contract)


def test_terminal_only_validation_reports_missing_motion_and_risk_positives(tmp_path):
    from unilab.algos.torch.sttt.world_model.replay import EpisodeReplay
    from unilab.algos.torch.sttt.world_model.training import _validate
    _, contract, model = physical_inputs(tmp_path)
    raw = _physical_prefix(_raw_capture(tmp_path / "train"), 1, contract)
    raw["terminal"][:] = True
    raw["motion_valid"][:] = False
    raw["risk"][:] = 1.
    replay = EpisodeReplay(contract, capacity_transitions=32, rollout_windows=1)
    replay.update_episode("early-terminal", raw)
    report = _validate(model, replay, [0], batch_size=1, device=torch.device("cpu"),
        seed=7, loss_config={}, require_windows=False)
    assert report["window_xy_rmse_m"] is None and report["selection_score"] is None
    assert report["risk_positive_count"] == 1 and report["risk_validation_available"]


def test_joint_effective_updates_and_paired_resume(tmp_path):
    from test_residual_film_ppo import make_ppo_learner
    from unilab.algos.torch.sttt.path_following.joint_checkpoint import load_joint_checkpoint, save_joint_checkpoint
    from unilab.algos.torch.sttt.path_following.joint_config import JointSchedule
    from unilab.algos.torch.sttt.path_following.joint_training import JointTrainer, _random_state, _restore_random_state
    from unilab.algos.torch.sttt.world_model.checkpoint import LoadedWorldModel
    from unilab.algos.torch.sttt.world_model.learner import WorldModelLearner
    from unilab.algos.torch.sttt.world_model.replay import EpisodeReplay
    initial_rng, threads = _random_state(torch.device("cpu")), torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(17)
        policy, contract, initial_model = physical_inputs(tmp_path)
        schedule = JointSchedule(2, 2, 1, 8, 1)
        wm_options = {"batch_size": 2, "microbatch_size": 1, "learning_rate": .0003,
                      "weight_decay": .000001, "gradient_clip_norm": 100.}
        def build(model):
            rollout = load_rollout(LoadedWorldModel(model, contract, {}), policy)
            actor = LatentResidualActor(model.feature_dim, (8,), (.02,) * 29, command_dim=3)
            wm = WorldModelLearner(model, wm_options, {})
            learner = make_ppo_learner(actor, rollout, stage="base")
            replay = EpisodeReplay(contract, capacity_transitions=32, rollout_windows=1, seed=3)
            return JointTrainer(wm, learner, replay, schedule, wm_batch_size=2), rollout
        engine, rollout = build(initial_model)
        engine.replay.update_episode("physical", _raw_capture(tmp_path / "train"))
        engine.warmup_update()
        engine.warmup_update()
        source_before = {key: value.clone() for key, value in rollout.policy.state_dict().items()}
        engine.update_iteration()
        assert (engine.world_model.updates, engine.actor.updates, engine.physical_control_steps) == (10, 1, 8)
        assert int(next(iter(engine.world_model.optimizer.state.values()))["step"]) == 10
        assert int(next(iter(engine.actor.algorithm.optimizer.state.values()))["step"]) == 1
        for key, value in rollout.policy.state_dict().items():
            torch.testing.assert_close(value, source_before[key], rtol=0, atol=0)
        saved_state = {"phase": "base", "iteration": engine.iteration, "base_updates": engine.base_updates,
            "film_updates": engine.film_updates, "physical_control_steps": engine.physical_control_steps,
            "random_state": _random_state(torch.device("cpu"))}
        path = save_joint_checkpoint(tmp_path / "joint.pt", engine.world_model, engine.actor, engine.replay,
            run_id="paired-test", signature={"schedule": asdict(schedule)}, state=saved_state)
        payload = load_joint_checkpoint(path)
        engine.update_iteration()
        restored, _ = build(LatentWorldModel(initial_model.config))
        restored.world_model.model.load_state_dict(payload["world_model"])
        restored.actor.actor.load_state_dict(payload["actor"])
        restored.world_model.load_state_dict(payload["world_model_learner"])
        restored.actor.load_state_dict(payload["actor_learner"])
        restored.replay.load_state_dict(payload["replay"])
        restored.restore_counters(payload["state"])
        _restore_random_state(payload["state"]["random_state"], torch.device("cpu"))
        restored.update_iteration()
        assert (restored.world_model.updates, restored.actor.updates, restored.physical_control_steps) == (18, 2, 16)
        for key, value in engine.world_model.model.state_dict().items():
            torch.testing.assert_close(value, restored.world_model.model.state_dict()[key], rtol=0, atol=0)
        for key, value in engine.actor.actor.state_dict().items():
            torch.testing.assert_close(value, restored.actor.actor.state_dict()[key], rtol=0, atol=0)
        for key, value in engine.actor.algorithm.critic.state_dict().items():
            torch.testing.assert_close(value, restored.actor.algorithm.critic.state_dict()[key], rtol=0, atol=0)
    finally:
        torch.set_num_threads(threads)
        _restore_random_state(initial_rng, torch.device("cpu"))


def test_forecast_risk_scoring_excludes_censored_safe_tails(tmp_path):
    import json
    from unilab.algos.torch.sttt.path_following.pretrained_evaluation import score_forecasts

    _, contract, _ = physical_inputs(tmp_path)
    raw = _physical_prefix(_raw_capture(tmp_path / "train"), 3, contract)
    raw["risk"][:] = 0.
    raw["risk"][-1] = 1.
    raw["terminal"][-1] = True
    raw["motion_valid"][-1] = False
    with SequenceDatasetWriter(tmp_path / "risk", contract) as writer:
        writer.append(raw)
    forecasts = [{"origin_step": 0, "risk": [.1, .1, .1, .1], "body_delta": [[0., 0., 0.]] * 4,
                  "actions": [[0.] * 29] * 4},
                 {"origin_step": 2, "risk": [.2, .2], "body_delta": [[0., 0., 0.]] * 2,
                  "actions": [[0.] * 29] * 2}]
    path = tmp_path / "risk" / "capture.json"
    path.write_text(json.dumps({"rollout_forecasts": forecasts}))
    metrics = score_forecasts(path.parent, risk_threshold=.5)
    assert metrics[0]["windows"] == 2 and metrics[0]["low_risk_event_fraction"] == .5
    assert metrics[1]["windows"] == 1 and metrics[1]["low_risk_event_fraction"] == 1.
    raw["risk"][-1] = 0.
    with SequenceDatasetWriter(tmp_path / "safe_tail", contract) as writer:
        writer.append(raw)
    (tmp_path / "safe_tail" / "capture.json").write_text(path.read_text())
    safe_metrics = score_forecasts(tmp_path / "safe_tail", risk_threshold=.5)
    assert len(safe_metrics) == 1 and safe_metrics[0]["windows"] == 1


def test_joint_cli_budget_contract_and_physical_cost_scales(tmp_path):
    import subprocess
    from omegaconf import OmegaConf
    from unilab.algos.torch.sttt.path_following.joint_config import JointTrainingConfig
    from unilab.algos.torch.sttt.path_following.pretraining_calibration import prepare_calibration

    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(["uv", "run", "--no-sync", "scripts/pretrain_latent_path_following.py", "--cfg", "job",
                             "schedule.pretrain_wm_updates=7", "schedule.base_iterations=3"],
                            cwd=root, capture_output=True, text=True, check=True)
    cfg = OmegaConf.create(result.stdout)
    cfg.capture.residual_amplitude_rad = .005
    cfg.capture.risk_tilt_limit_rad = .8
    cfg.actor.training.residual_limit_rad = [.02] * 29
    options = JointTrainingConfig.from_mapping(cfg)
    assert options.raw["wm"]["training"]["max_updates"] == 7
    assert options.raw["actor"]["training"]["max_updates"] == 3
    cfg.capture.steps_per_iteration = 10
    with pytest.raises(ValueError, match="same complete"):
        JointTrainingConfig.from_mapping(cfg)
    cfg.capture.steps_per_iteration = 20
    cfg.refinement_cycles = 2
    with pytest.raises(ValueError):
        JointTrainingConfig.from_mapping(cfg)
    costs = asdict(LatentCostConfig())
    calibration = {"response_probe_result": None, "require_response_probe": False}
    report = prepare_calibration(calibration, costs, directories=[], contract={},
                                 residual_limits=(.02,) * 29, horizon_steps=20, root=tmp_path)
    assert report["costs"] == costs
    assert report["scale_method"] == "declared_physical_task_tolerances"
    with pytest.raises(ValueError, match="calibration requires"):
        prepare_calibration({**calibration, "cost_scales": {}}, costs, directories=[], contract={},
                            residual_limits=(.02,) * 29, horizon_steps=20, root=tmp_path)
