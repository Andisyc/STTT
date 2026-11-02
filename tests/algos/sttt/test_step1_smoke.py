"""Smoke test suite for Step 1 of Spatial Test-Time Training (STTT)."""

from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import sys
import torch
import pytest

from unilab.algos.torch.sttt import (
    DEFAULT_OBS_DIM,
    PROPRIO_LAYOUT,
    RunningNorm,
    SpatialBatch,
    SpatialTrajectoryWindow,
    SpatialWorldModel,
    compute_se2_deltas,
    mat_to_quat,
    quat_to_mat,
    validate_trajectory_window,
    wrap_to_pi,
    yaw_mat,
)


def test_geometry_se2_deltas_and_rotations():
    """Verify SE(2) projection physics and rotation consistency."""
    batch_size = 2
    horizon = 5

    # Identity quat (w=1, x=0, y=0, z=0)
    quat0 = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    pos0 = torch.zeros(batch_size, 3)

    # 1. Pure forward motion along X: x increases by 0.1 each step
    future_pos = torch.zeros(batch_size, horizon, 3)
    for h in range(horizon):
        future_pos[:, h, 0] = (h + 1) * 0.1
    future_quat = quat0.unsqueeze(1).expand(-1, horizon, -1)

    deltas = compute_se2_deltas(future_pos, future_quat, pos0=pos0, quat0=quat0)
    assert deltas.shape == (batch_size, horizon, 3)
    # dx should be positive and matching
    assert torch.allclose(deltas[..., 0], future_pos[..., 0], atol=1e-5)
    # dy and dyaw should be 0
    assert torch.allclose(deltas[..., 1], torch.zeros_like(deltas[..., 1]), atol=1e-5)
    assert torch.allclose(deltas[..., 2], torch.zeros_like(deltas[..., 2]), atol=1e-5)

    # 2. Pure 90 deg rotation around Z
    # 90 deg quat: cos(45 deg) + sin(45 deg) * k = [sqrt(2)/2, 0, 0, sqrt(2)/2]
    val = math.sqrt(2) / 2
    quat_90 = torch.tensor([val, 0.0, 0.0, val]).expand(batch_size, horizon, 4)
    deltas_rot = compute_se2_deltas(future_pos, quat_90, pos0=pos0, quat0=quat0)
    expected_yaw = math.pi / 2
    assert torch.allclose(deltas_rot[..., 2], torch.full_like(deltas_rot[..., 2], expected_yaw), atol=1e-5)

    # 3. Round-trip quat -> mat -> quat
    mat = quat_to_mat(quat0)
    recon_q = mat_to_quat(mat)
    assert torch.allclose(recon_q, quat0, atol=1e-6)


def test_spatial_world_model_autograd():
    """Verify zero-init property, gradient flow to actions, and RunningNorm."""
    batch = 4
    h_hist = 15
    horizon = 15
    obs_dim = DEFAULT_OBS_DIM  # 64
    act_dim = 29

    model = SpatialWorldModel(
        obs_dim=obs_dim,
        act_dim=act_dim,
        horizon=horizon,
        hist_len=h_hist,
        hidden=(128, 128),
    )

    obs_hist = torch.randn(batch, h_hist, obs_dim)
    action_seq = torch.randn(batch, horizon, act_dim, requires_grad=True)

    # 1. Output head is zero-initialized: initial predictions must be identically 0
    pred = model(obs_hist, action_seq)
    assert pred.shape == (batch, horizon, 3)
    assert torch.allclose(pred, torch.zeros_like(pred), atol=1e-7)

    # 2. Perturb final linear weight to verify backprop gradient flow to action_seq
    with torch.no_grad():
        model.net[-1].weight.normal_(0.0, 0.1)

    pred = model(obs_hist, action_seq)
    target = torch.randn(batch, horizon, 3)
    loss = model.compute_loss(pred, target)
    loss.backward()

    assert action_seq.grad is not None
    assert torch.isfinite(action_seq.grad).all()
    assert (action_seq.grad.abs().sum().item()) > 0.0


def test_contract_and_batch_collator():
    """Verify contract validations, fail-fast on NaN/out-of-range, and collation."""
    h_hist = 15
    horizon = 15
    obs_dim = DEFAULT_OBS_DIM
    act_dim = 29

    valid_window = SpatialTrajectoryWindow(
        proprio_history=torch.randn(h_hist, obs_dim),
        action_sequence=torch.randn(horizon, act_dim),
        delta_trajectory=torch.zeros(horizon, 3),
        commands=torch.tensor([0.4, 0.0, 0.0]),
        step_indices=tuple(range(h_hist + horizon)),
    )

    # Should pass without exception
    validate_trajectory_window(valid_window)

    # 1. Test NaN detection
    nan_window = SpatialTrajectoryWindow(
        proprio_history=torch.full((h_hist, obs_dim), float("nan")),
        action_sequence=torch.randn(horizon, act_dim),
        delta_trajectory=torch.zeros(horizon, 3),
        commands=torch.tensor([0.4, 0.0, 0.0]),
        step_indices=tuple(range(h_hist + horizon)),
    )
    with pytest.raises(ValueError, match="NaN"):
        validate_trajectory_window(nan_window)

    # 2. Test yaw range enforcement
    invalid_yaw_window = SpatialTrajectoryWindow(
        proprio_history=torch.randn(h_hist, obs_dim),
        action_sequence=torch.randn(horizon, act_dim),
        delta_trajectory=torch.full((horizon, 3), 4.0),  # > pi
        commands=torch.tensor([0.4, 0.0, 0.0]),
        step_indices=tuple(range(h_hist + horizon)),
    )
    with pytest.raises(ValueError):
        validate_trajectory_window(invalid_yaw_window)

    # 3. Test Collation into SpatialBatch
    batch = SpatialBatch.collate([valid_window, valid_window])
    assert batch.batch_size == 2
    assert batch.proprio_history.shape == (2, h_hist, obs_dim)
    assert batch.action_sequence.shape == (2, horizon, act_dim)
    assert batch.delta_trajectory.shape == (2, horizon, 3)
    assert batch.commands.shape == (2, 3)


def _transitions(episode_id=0, steps=6, terminal_at=5):
    from unilab.algos.torch.sttt.contract import SpatialTransition

    for step in range(steps):
        yield SpatialTransition(
            rollout_id="regression",
            episode_id=episode_id,
            episode_step=step,
            proprio=torch.full((64,), float(100 * episode_id + step)),
            action=torch.full((29,), float(10 * episode_id + step) + 0.25),
            commands=torch.tensor([0.4, 0.0, 0.0]),
            position=torch.tensor([float(100 * episode_id + step), 0.0, 0.8]),
            quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            next_position=torch.tensor([float(100 * episode_id + step + 1), 0.0, 0.8]),
            next_quaternion=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            terminated=step == terminal_at,
        )


def _metadata():
    return {
        "checkpoint_path": "/policy.pt",
        "checkpoint_sha256": "0" * 64,
        "task": "G1WalkFlat",
        "backend": "mujoco",
        "ctrl_dt": 0.02,
        "obs_dim": 64,
        "act_dim": 29,
        "proprio_layout": PROPRIO_LAYOUT,
        "joint_names": [f"joint_{i}" for i in range(29)],
        "action_scale": 0.25,
        "default_joint_angles": [0.0] * 29,
        "action_representation": "env_input",
        "quaternion_order": "wxyz",
        "coordinate_frame": "reference_heading",
        "seed": 0,
    }


def test_collection_windows_align_actions_and_exclude_terminal_states():
    from unilab.algos.torch.sttt.collection import build_spatial_windows

    records = [*_transitions(), *_transitions(episode_id=1)]
    windows = build_spatial_windows(records, history_length=3, horizon=2)
    assert [(w["episode_id"], w["step_indices"].tolist()) for w in windows] == [
        (0, [0, 1, 2, 3, 4]),
        (0, [1, 2, 3, 4, 5]),
        (1, [0, 1, 2, 3, 4]),
        (1, [1, 2, 3, 4, 5]),
    ]
    for window, episode, origin in zip(windows, [0, 0, 1, 1], [2, 3, 2, 3], strict=True):
        torch.testing.assert_close(
            window["proprio_history"][:, 0],
            torch.arange(origin - 2, origin + 1).float() + 100 * episode,
        )
        torch.testing.assert_close(
            window["action_sequence"][:, 0],
            torch.tensor([origin, origin + 1]).float() + 10 * episode + 0.25,
        )
        torch.testing.assert_close(
            window["delta_trajectory"],
            torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        )


def test_collection_discards_history_across_missing_steps():
    from unilab.algos.torch.sttt.collection import build_spatial_windows

    records = list(_transitions(steps=8, terminal_at=-1))
    windows = build_spatial_windows(
        [records[0], records[1], *records[3:]], history_length=3, horizon=2,
    )
    assert [w["step_indices"].tolist() for w in windows] == [
        [3, 4, 5, 6, 7], [4, 5, 6, 7, 8],
    ]


def test_collection_roundtrip_preserves_contract_and_rejects_ambiguous_actions(tmp_path):
    from unilab.algos.torch.sttt.collection import (
        build_spatial_windows,
        load_spatial_collection,
        save_spatial_collection,
    )

    windows = build_spatial_windows(_transitions(), history_length=3, horizon=2)
    path = tmp_path / "collection.pt"
    save_spatial_collection(path, windows, _metadata(), history_length=3, horizon=2)
    restored = load_spatial_collection(path)
    keys = ("proprio_history", "action_sequence", "delta_trajectory", "commands", "step_indices")
    batch = SpatialBatch.collate([
        SpatialTrajectoryWindow(**{key: window[key] for key in keys})
        for window in restored["windows"]
    ])
    torch.testing.assert_close(batch.action_sequence[0], windows[0]["action_sequence"])
    assert restored["windows"][0]["episode_id"] == 0
    assert restored["metadata"]["action_representation"] == "env_input"
    with pytest.raises(ValueError):
        save_spatial_collection(
            tmp_path / "wrong.pt", windows,
            {**_metadata(), "action_representation": "pd_target"},
            history_length=3, horizon=2,
        )
    with pytest.raises(ValueError):
        save_spatial_collection(tmp_path / "empty.pt", [], _metadata())
    assert not (tmp_path / "empty.pt").exists()


@pytest.mark.skipif(not os.environ.get("STTT_SMOKE_CHECKPOINT"), reason="Requires a real local FADA checkpoint")
def test_real_checkpoint_collection_roundtrip(tmp_path):
    from unilab.algos.torch.sttt.collection import load_spatial_collection

    root = Path(__file__).resolve().parents[3]
    path = tmp_path / "real.pt"
    result = subprocess.run(
        [
            sys.executable, str(root / "scripts/collect_spatial_data.py"),
            "--checkpoint", os.environ["STTT_SMOKE_CHECKPOINT"],
            "--device", "cpu", "--num-steps", "100", "--out", str(path),
            "+env.max_episode_seconds=0.8",
            "++env.commands.vel_limit=[[0.4,0.0,0.0],[0.4,0.0,0.0]]",
        ],
        cwd=root, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    artifact = load_spatial_collection(path)
    windows = artifact["windows"]
    assert len({window["episode_id"] for window in windows}) >= 2
    assert any(torch.count_nonzero(window["action_sequence"]).item() for window in windows)
    for window in windows:
        validate_trajectory_window(SpatialTrajectoryWindow(**{
            key: window[key] for key in
            ("proprio_history", "action_sequence", "delta_trajectory", "commands", "step_indices")
        }))
    print(result.stdout)
