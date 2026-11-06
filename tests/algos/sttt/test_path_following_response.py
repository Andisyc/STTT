"""B objective, causal inputs, frozen-weight derivatives and data/artifact boundaries."""

from __future__ import annotations

import hashlib
import math
import shutil
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.distill.fada.runtime_policy import FADAFutureQuery, OfficialFADAPlaybackProfile
from unilab.algos.torch.sttt.path_following import FixedStraightPath, InterventionQueryRecord, PathFollowingWindowSample, compose_joint_position_residual
from unilab.algos.torch.sttt.path_following.checkpoint import load_response_checkpoint, save_response_checkpoint
from unilab.algos.torch.sttt.path_following.dataset import PathFollowingDatasetWriter, iter_path_following_samples, read_path_following_metadata
from unilab.algos.torch.sttt.path_following.diagnostics import diagnose_response_jacobian
from unilab.algos.torch.sttt.path_following.features import pack_response_context, response_contract_from_metadata
from unilab.algos.torch.sttt.path_following.model import ResponseModel, response_loss
from unilab.algos.torch.sttt.path_following.training import train_response_model
from unilab.algos.torch.sttt.pose_data import ControlInterval, TimedPelvisPose

ROOT = Path(__file__).resolve().parents[3]


def write_analytic_capture(directory, *, seed, samples=8, context_offset=0., varying=True):
    """Analytical response fixture, not a simulation or a learned physical policy."""
    profile = OfficialFADAPlaybackProfile(
        source_identity={"run": "analytical-fixture"}, joint_names=tuple(f"j{i}" for i in range(29)),
        default_joint_pos=(0.2,) * 29, action_scale=(0.35,) * 29, stiffness=(50.,) * 29,
        damping=(2.,) * 29, clip_actions=0.35, ctrl_dt=0.02, gait_frequency=1.5,
        gait_clock_mode="source_clock", phase_init=(0., math.pi),
    )
    terms = ("base_ang_vel", "dof_pos", "dof_vel", "projected_gravity")
    preprocessing = {
        "term_order": list(terms), "noise_enabled": False,
        "term_scale": dict(zip(terms, (0.25, 2.5, 0.15, 1.))), "term_noise": dict.fromkeys(terms, 0.),
    }
    source = {
        "fixture": "analytical_response_not_physics", "label_source": "simulation_ground_truth",
        "source_profile": asdict(profile), "preprocessing": preprocessing,
        "checkpoint_sha256": hashlib.sha256(b"analytical-policy-fixture").hexdigest(),
        "delta_frame": "window_start_horizontal_heading", "delta_units": ["m", "m", "rad"], "residual_seed": seed,
    }
    writer = PathFollowingDatasetWriter(directory, source=source, shard_size=4)
    rng = np.random.default_rng(seed)
    position, yaw = np.array([0., 0., 0.8]), 0.3

    def pose(index):
        matrix = np.eye(4)
        matrix[:2, :2] = ((math.cos(yaw), -math.sin(yaw)), (math.sin(yaw), math.cos(yaw)))
        matrix[:3, 3] = position
        timestamp = index * 20_000_000
        return TimedPelvisPose(1, 1, timestamp, matrix, True, (timestamp, timestamp), "fixture_world")

    reference = FixedStraightPath.from_initial_pose(pose(0))
    for index in range(samples):
        residual = torch.tensor(rng.uniform(-0.02, 0.02, (1, 29)), dtype=torch.float32) if varying else torch.zeros(1, 29)
        obs = torch.full((1, 2, 64), context_offset + 0.1 * index)
        mask = torch.ones(1, 2, dtype=torch.bool)
        if index == 0:
            mask[0, 0] = False
            obs[0, 0] = 0.
        phase = 0.1 * index
        command = torch.tensor([[0.4, 0., 0., math.sin(phase), -math.sin(phase), math.cos(phase), -math.cos(phase)]])
        query = FADAFutureQuery(
            obs, torch.zeros(1, 2, 29), command, mask,
            torch.full((1, 3, 64), context_offset + 0.01 * index), preprocessing, profile,
        )
        adapted = compose_joint_position_residual(query.planner_future, residual, ctrl_dt=0.02, position_scale=2.5, velocity_scale=0.15)
        record = InterventionQueryRecord.from_query(query, adapted, residual)
        start = pose(index)
        local = np.array([0.03 + 0.4 * float(residual[0, 0]), 0.5 * float(residual[0, 1])])
        rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        position[:2] += rotation @ local
        yaw += 0.6 * float(residual[0, 2])
        control = ControlInterval(1, index, index * 20_000_000, (index + 1) * 20_000_000, np.zeros(29), command[0, :3].numpy())
        writer.append(PathFollowingWindowSample(index, record, (control,), start, pose(index + 1), reference))
    writer.close()


def capture_batch(directory):
    samples = list(iter_path_following_samples(directory))
    return (
        torch.cat([pack_response_context(sample.start_query) for sample in samples]),
        torch.cat([sample.start_query.residual_rad for sample in samples]),
        torch.from_numpy(np.stack([sample.delta for sample in samples])).float(),
    )


def small_config(train, validation, output):
    cfg = OmegaConf.load(ROOT / "conf/sttt/response_model.yaml")
    merged = OmegaConf.merge(cfg, {
        "data": {"train_dirs": [str(train)], "validation_dirs": [str(validation)]},
        "model": {"hidden_dims": [16, 8]}, "training": {"batch_size": 4, "shuffle_buffer_size": 8, "epochs": 1},
        "output_checkpoint": str(output), "diagnostics": {"jacobian_samples": 1},
    })
    assert isinstance(merged, DictConfig)
    return merged


def fitted_model(directory):
    write_analytic_capture(directory, seed=3)
    batch = capture_batch(directory)
    with torch.random.fork_rng():
        torch.manual_seed(3)
        model = ResponseModel(2, 3, [16, 8])
    model.fit_normalization([batch], epsilon=1e-6)
    return model, batch


def test_b_loss_is_half_squared_physical_norm_averaged_over_samples():
    prediction = torch.tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
    target = torch.tensor([[0.5, 1., 1.5], [0., 0., 0.]])
    loss = response_loss(prediction, target)
    assert loss.item() == 20.125
    gradient, = torch.autograd.grad(loss, prediction)
    torch.testing.assert_close(gradient, (prediction.detach() - target) / 2, atol=0, rtol=0)


def test_fitted_statistics_ignore_history_padding_and_never_update_in_forward(tmp_path):
    model, (context, residual, _) = fitted_model(tmp_path)
    expected = 2 * sum(0.1 * index for index in range(8)) / 15
    assert float(model.context_mean[0]) == pytest.approx(expected, abs=1e-7)
    before = model.context_mean.clone()
    model.train()
    original = model(context[:1], residual[:1])
    poisoned = context[:1].clone()
    poisoned[0, :64] = 10000.
    torch.testing.assert_close(model(poisoned, residual[:1]), original, atol=0, rtol=0)
    shifted = context.clone()
    shifted[:, :128] += 100.
    model(shifted, residual)
    torch.testing.assert_close(model.context_mean, before, atol=0, rtol=0)


def test_physical_residual_derivative_survives_frozen_b_and_normalization(tmp_path):
    model, (context, residual, _) = fitted_model(tmp_path)
    model.double().requires_grad_(False).train()
    report = diagnose_response_jacobian(model, context[:1].double(), residual[:1].double(), epsilon_rad=1e-6)
    derivative, numeric = np.asarray(report["jacobian"]), np.asarray(report["finite_difference"])
    np.testing.assert_allclose(derivative, numeric, rtol=1e-5, atol=1e-7)
    assert np.linalg.norm(derivative) > 1e-3
    assert model.training and all(not parameter.requires_grad for parameter in model.parameters())


def test_artifact_preserves_normalized_predictions_and_rejects_other_contracts(tmp_path):
    model, (context, residual, _) = fitted_model(tmp_path / "data")
    model.double().eval()
    contract = response_contract_from_metadata(read_path_following_metadata(tmp_path / "data"))
    path = save_response_checkpoint(tmp_path / "b.pt", model, contract=contract, training_config={}, data_sources=[], metrics={})
    loaded = load_response_checkpoint(path, expected_contract=contract)
    torch.testing.assert_close(loaded.model(context.double(), residual.double()), model(context.double(), residual.double()), atol=0, rtol=0)
    different = deepcopy(contract)
    different["data"]["window_steps"] += 1
    with pytest.raises(ValueError, match="incompatible"):
        load_response_checkpoint(path, expected_contract=different)
    different = deepcopy(contract)
    different["data"]["joint_names"].reverse()
    different["source"]["source_profile"]["joint_names"].reverse()
    with pytest.raises(ValueError, match="incompatible"):
        load_response_checkpoint(path, expected_contract=different)
    torch.save({"format": "retired_world_model", "state_dict": model.state_dict()}, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="unsupported B"):
        load_response_checkpoint(tmp_path / "old.pt")


def test_training_holds_out_whole_captures_and_fits_only_train_statistics(tmp_path):
    train, validation = tmp_path / "train", tmp_path / "validation"
    write_analytic_capture(train, seed=11, context_offset=2.)
    write_analytic_capture(validation, seed=12, context_offset=100.)
    result = train_response_model(small_config(train, validation, tmp_path / "model.pt"), log=lambda _: None)
    expected = 2. + 2 * sum(0.1 * index for index in range(8)) / 15
    assert float(result.model.context_mean[0]) == pytest.approx(expected, abs=1e-6)
    assert int(result.model.normalization_samples) == 8
    copied = tmp_path / "copied_train"
    shutil.copytree(train, copied)
    with pytest.raises(ValueError, match="disjoint captures"):
        train_response_model(small_config(train, copied, tmp_path / "invalid.pt"), log=lambda _: None)


def test_constant_interventions_are_not_accepted_as_direction_pretraining(tmp_path):
    train, validation = tmp_path / "train", tmp_path / "validation"
    write_analytic_capture(train, seed=11, varying=False)
    write_analytic_capture(validation, seed=12)
    with pytest.raises(ValueError, match="varying residuals"):
        train_response_model(small_config(train, validation, tmp_path / "invalid.pt"), log=lambda _: None)
