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


def physical_inputs(tmp_path):
    case, profile = legacy_case(tmp_path / "source")
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
    cost = latent_actor_loss(result, start, ctrl_dt=.02, window_steps=2, config=LatentCostConfig())["loss"]
    cost.backward()
    assert sum(float(parameter.grad.abs().sum()) for parameter in actor.parameters() if parameter.grad is not None) > 0
    assert all(parameter.grad is None for parameter in (*loaded.model.parameters(), *rollout.policy.parameters()))
    assert bool(torch.isfinite(result.body_delta).all())
    assert not torch.equal(result.commands[:, 0], result.commands[:, -1])


def test_real_wm_actor_pretraining_and_exact_actor_resume(tmp_path):
    policy, _, model = physical_inputs(tmp_path)
    wm = train_world_model({"data": {"train_dirs": [str(tmp_path / "train")], "validation_dirs": [str(tmp_path / "validation")]},
                            "model": asdict(model.config), "seed": 7, "device": "cpu", "output_dir": str(tmp_path / "wm_training"),
                            "training": {"max_updates": 1, "batch_size": 1, "validation_every": 1, "validation_batches": 1, "checkpoint_every": 1}})
    cfg = {"world_model_checkpoint": wm["best_checkpoint_path"], "policy_checkpoint": str(policy),
           "data_dirs": [str(tmp_path / "train")], "output_dir": str(tmp_path / "actor_full"), "device": "cpu",
           "training": {"residual_limit_rad": [.02] * 29, "hidden_dims": [8], "max_updates": 2,
                        "batch_size": 1, "horizon_windows": 1, "checkpoint_every": 1, "seed": 5}}
    full = train_latent_actor(cfg)
    split = deepcopy(cfg)
    split["output_dir"] = str(tmp_path / "actor_first")
    split["training"]["max_updates"] = 1
    first = train_latent_actor(split)
    split["output_dir"] = str(tmp_path / "actor_resumed")
    split["training"]["max_updates"] = 2
    split["resume_checkpoint"] = first["actor_checkpoint"]
    resumed = train_latent_actor(split)
    loaded = load_world_model(wm["best_checkpoint_path"])
    a, _ = load_actor_checkpoint(full["actor_checkpoint"], loaded.model, loaded.contract, TrajectoryConfig())
    b, _ = load_actor_checkpoint(resumed["actor_checkpoint"], loaded.model, loaded.contract, TrajectoryConfig())
    for name, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0)
    assert int(torch.count_nonzero(a.net[-1].state_dict()["weight"])) > 0
    with torch.no_grad():
        next(loaded.model.parameters()).add_(.01)
    with pytest.raises(ValueError, match="another WM base"):
        load_actor_checkpoint(full["actor_checkpoint"], loaded.model, loaded.contract, TrajectoryConfig())


