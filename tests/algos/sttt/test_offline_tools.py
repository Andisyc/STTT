"""Offline numerical/data contracts. No optimizer steps, epochs, or simulation."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from unilab.algos.torch.sttt.checkpoint import (
    load_world_model_checkpoint,
    restore_training_state,
    save_world_model_checkpoint,
)
from unilab.algos.torch.sttt.collection import build_spatial_windows, save_spatial_collection
from unilab.algos.torch.sttt.contract import PROPRIO_LAYOUT, SpatialBatch, SpatialTransition
from unilab.algos.torch.sttt.model import RunningNorm, SpatialWorldModel

ROOT = Path(__file__).resolve().parents[3]


def _make_collection(path: Path, *, episodes: int = 4, action_scale: float = 0.25) -> None:
    records = []
    for episode in range(episodes):
        for step in range(6):
            speed = float(episode + 1)
            records.append(SpatialTransition(
                rollout_id="offline-smoke", episode_id=episode, episode_step=step,
                proprio=torch.full((64,), float(episode * 100 + step)),
                action=torch.full((29,), float(step) / 10 + episode),
                commands=torch.tensor([0.4, 0.0, 0.0]),
                position=torch.tensor([step * speed, 0.0, 0.8]),
                quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                next_position=torch.tensor([(step + 1) * speed, 0.0, 0.8]),
                next_quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                truncated=step == 5,
            ))
    metadata = {
        "checkpoint_path": "/source-policy.pt", "checkpoint_sha256": "0" * 64,
        "task": "g1_walk_flat/mujoco", "backend": "mujoco", "ctrl_dt": 0.02,
        "obs_dim": 64, "act_dim": 29, "proprio_layout": PROPRIO_LAYOUT,
        "joint_names": [f"joint_{i}" for i in range(29)], "action_scale": [action_scale],
        "default_joint_angles": [0.0] * 29, "action_representation": "env_input",
        "quaternion_order": "wxyz", "coordinate_frame": "reference_heading", "seed": 0,
    }
    windows = build_spatial_windows(records, history_length=2, horizon=2)
    save_spatial_collection(path, windows, metadata, history_length=2, horizon=2)


def test_running_norm_keeps_correct_statistics_at_freeze_boundary():
    norm = RunningNorm(1, max_count=3)
    norm.update(torch.tensor([[0.0], [2.0]]))
    norm.update(torch.tensor([[4.0], [6.0]]))
    torch.testing.assert_close(norm.mean, torch.tensor([3.0]))
    torch.testing.assert_close(norm.var, torch.tensor([5.0]))
    norm.update(torch.tensor([[100.0]]))
    torch.testing.assert_close(norm.mean, torch.tensor([3.0]))
    torch.testing.assert_close(norm.var, torch.tensor([5.0]))


def test_yaw_loss_takes_short_arc_and_respects_coordinate_weights():
    model = SpatialWorldModel(64, 29, horizon=1, hist_len=1, hidden=(4,))
    predicted = torch.tensor([[[2.0, 0.0, -math.pi + 0.01]]], dtype=torch.float64)
    target = torch.tensor([[[0.0, 0.0, math.pi - 0.01]]], dtype=torch.float64)
    assert model.compute_loss(predicted, target).item() == pytest.approx((4 + 0.02**2) / 3)
    assert model.compute_loss(predicted, target, torch.tensor([0., 0., 1.])).item() == pytest.approx(0.02**2)
    with pytest.raises(ValueError):
        model.compute_loss(predicted, target, torch.zeros(3))


def test_offline_data_checkpoint_and_diagnostic_roundtrip(tmp_path):
    from unilab.algos.torch.sttt.dataset import SpatialTrajectoryDataset, make_group_split
    from unilab.algos.torch.sttt.diagnostics import diagnose_spatial_jacobian
    from unilab.algos.torch.sttt.training import evaluate_world_model, fit_normalizers

    data_path = tmp_path / "collection.pt"
    _make_collection(data_path)
    dataset = SpatialTrajectoryDataset([data_path])
    train, validation, split = make_group_split(dataset, validation_fraction=0.25, seed=7)
    train_groups = {tuple(group) for group in split["train_groups"]}
    validation_groups = {tuple(group) for group in split["validation_groups"]}
    assert train_groups.isdisjoint(validation_groups)
    assert train_groups | validation_groups == {("offline-smoke", i) for i in range(4)}
    for index in train.indices:
        row = dataset.windows[index]
        assert (row["rollout_id"], row["episode_id"]) in train_groups
    for index in validation.indices:
        row = dataset.windows[index]
        assert (row["rollout_id"], row["episode_id"]) in validation_groups

    train_loader = DataLoader(train, batch_size=4, collate_fn=SpatialBatch.collate)
    validation_loader = DataLoader(validation, batch_size=2, collate_fn=SpatialBatch.collate)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        model = SpatialWorldModel(64, 29, horizon=2, hist_len=2, hidden=(8, 8))
        fit_normalizers(model, train_loader)
        average_episode = sum(group[1] for group in train_groups) / len(train_groups)
        assert model.in_norm.mean[0].item() == pytest.approx(100 * average_episode + 1.0)
        assert model.out_norm.mean[0].item() == pytest.approx(average_episode + 1.0)
        with torch.no_grad():
            head = model.net[-1].weight
            head.copy_(torch.linspace(-0.05, 0.05, head.numel()).reshape_as(head))
        before = {name: value.clone() for name, value in model.state_dict().items()}
        batch = next(iter(train_loader))
        prediction = model(batch.proprio_history, batch.action_sequence)
        model.compute_loss(prediction, batch.delta_trajectory).backward()
        assert model.net[-1].weight.grad is not None
        assert model.net[-1].weight.grad.abs().sum().item() > 0
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        generator = torch.Generator().manual_seed(19)
        checkpoint = tmp_path / "world_model.pt"
        save_world_model_checkpoint(
            checkpoint, model, data_contract=dataset.data_contract,
            data_identity=dataset.data_identity, split=split, resolved_config={},
            optimizer=optimizer, shuffle_generator=generator,
        )
        rng_before = torch.get_rng_state().clone()
        restored, payload = load_world_model_checkpoint(
            checkpoint, expected_data_contract=dataset.data_contract,
            expected_data_identity=dataset.data_identity, expected_split=split,
        )
        torch.testing.assert_close(torch.get_rng_state(), rng_before)
        torch.testing.assert_close(restored(batch.proprio_history, batch.action_sequence), prediction)
        for name, value in restored.state_dict().items():
            torch.testing.assert_close(value, before[name])
        recovered_optimizer = torch.optim.Adam(restored.parameters(), lr=0.9)
        restore_training_state(payload, recovered_optimizer, shuffle_generator=generator)
        assert recovered_optimizer.param_groups[0]["lr"] == 3e-4
        with pytest.raises(ValueError):
            load_world_model_checkpoint(checkpoint, expected_data_identity={"files": []})
        with pytest.raises(ValueError):
            load_world_model_checkpoint(
                checkpoint, expected_data_contract={**dataset.data_contract, "ctrl_dt": 0.01},
            )

        metrics = evaluate_world_model(restored, validation_loader)
        rows = SpatialBatch.collate([validation[i] for i in range(len(validation))])
        with torch.no_grad():
            error = restored(rows.proprio_history, rows.action_sequence) - rows.delta_trajectory
        assert metrics["endpoint_xy_m"] == pytest.approx(error[:, -1, :2].norm(dim=-1).mean().item())
        expected_yaw = torch.atan2(torch.sin(error[:, -1, 2]), torch.cos(error[:, -1, 2])).abs().mean().item()
        assert metrics["endpoint_yaw_rad"] == pytest.approx(expected_yaw)
        report = diagnose_spatial_jacobian(
            restored, dataset[0], action_steps=[0, 1], joint_indices=[0, 28], output_step=0,
        )
        assert report["max_abs_error"] < 1e-6
        for name, value in restored.state_dict().items():
            torch.testing.assert_close(value, before[name])

    evaluation = subprocess.run(
        [sys.executable, str(ROOT / "scripts/evaluate_spatial_world_model.py"),
         f"data.paths=[{data_path}]", f"evaluation.checkpoint={checkpoint}", "training.batch_size=2"],
        cwd=ROOT, text=True, capture_output=True, timeout=30,
    )
    assert evaluation.returncode == 0, evaluation.stdout + evaluation.stderr
    evaluated = json.loads(evaluation.stdout)
    assert evaluated["endpoint_xy_m"] == pytest.approx(metrics["endpoint_xy_m"])
    diagnostic = subprocess.run(
        [sys.executable, str(ROOT / "scripts/diagnose_spatial_jacobian.py"),
         "--checkpoint", str(checkpoint), "--data", str(data_path),
         "--output-step", "0", "--action-steps", "0", "1", "--joint-indices", "0", "28"],
        cwd=ROOT, text=True, capture_output=True, timeout=30,
    )
    assert diagnostic.returncode == 0, diagnostic.stdout + diagnostic.stderr
    print("Offline CPU smoke: grouped split, train-only stats, backward, checkpoint, eval and Jacobian CLIs; no optimizer steps.")


def test_dataset_rejects_single_episode_holdout_and_mixed_action_contracts(tmp_path):
    from unilab.algos.torch.sttt.dataset import SpatialTrajectoryDataset, make_group_split

    single = tmp_path / "single.pt"
    _make_collection(single, episodes=1)
    dataset = SpatialTrajectoryDataset([single])
    with pytest.raises(ValueError):
        make_group_split(dataset)
    incompatible = tmp_path / "incompatible.pt"
    _make_collection(incompatible, action_scale=0.5)
    with pytest.raises(ValueError):
        SpatialTrajectoryDataset([single, incompatible])
