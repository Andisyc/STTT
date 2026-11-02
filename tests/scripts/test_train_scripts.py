"""Tests for script entry-point utilities (pure functions and Hydra config defaults).

Coverage targets:
  - train_offpolicy.py: Hydra defaults, default_device(), resolve_checkpoint_path()
  - train_mlx_ppo.py:   get_latest_run(), get_latest_checkpoint()  (skipped if mlx absent)
  - play_interactive.py: resolve_checkpoint()                       (skipped if mujoco absent)
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from unilab.algos.torch.distill import LEGACY_REQUEST_STAGE_NAMES
from unilab.algos.torch.distill.workflows import entry_collection as distill_entry_collection
from unilab.algos.torch.distill.workflows import entry_plan as distill_entry_plan
from unilab.algos.torch.distill.workflows import entry_training as distill_entry_training
from unilab.algos.torch.distill.workflows import entry_workflow as distill_entry_workflow
from unilab.base.backend.base import BackendSceneArtifacts
from unilab.base.backend.motrix.playback import run_motrix_playback
from unilab.visualization import playback_checkpoint_contract, playback_viewer
from unilab.visualization import playback_cli as playback_cli_owner

_SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
_CONF_DIR = Path(__file__).parent.parent.parent / "conf"
_SRC_DIR = Path(__file__).parent.parent.parent / "src"


def _normalize_overrides(overrides: list[str] | None, *, offpolicy: bool = False) -> list[str]:
    normalized: list[str] = []
    algo = "sac"
    task_selected = False

    for override in overrides or []:
        if override.startswith("algo="):
            algo = override.split("=", 1)[1]
            normalized.append(override)
            continue
        if override.startswith("task="):
            task_selected = True
            normalized.append(override)
            continue
        normalized.append(override)

    if not task_selected:
        if offpolicy:
            normalized.append(f"task={algo}/g1_walk_flat/mujoco")
        else:
            normalized.append("task=go1_joystick_flat/mujoco")
    return normalized


def _load_script(name: str) -> Any:
    """Load a scripts/<name>.py as a fresh module (no __init__ required)."""
    path = _SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _load_deploy_script(name: str) -> Any:
    path = _SCRIPTS_DIR / "deploy" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_distill_script_exports_owner_level_route_functions() -> None:
    mod = _load_script("train_distill")

    assert mod.build_teacher_spec.__module__.endswith("distill.workflows.entry_training")
    assert mod.run_collect_dataset.__module__.endswith("distill.workflows.entry_collection")
    assert mod.run_single_entry_workflow.__module__.endswith("distill.workflows.entry_workflow")


@pytest.mark.parametrize(
    ("task", "task_name"),
    [
        ("g1_walk_height", "G1WalkHeight"),
        ("g1_stand_height", "G1StandHeight"),
    ],
)
def test_g1_height_tracking_live_path_owner_contracts(task, task_name):
    mod = _load_deploy_script("check_unilab_g1_height_tracking_live_path")

    cfg = mod._compose_cfg(task)
    contract = mod._task_contract(task)

    assert cfg.training.task_name == task_name
    assert contract.registry_task_name == task_name
    assert contract.actor_obs_dim == 99
    assert contract.critic_obs_dim == 102


def test_g1_height_tracking_live_path_stand_height_contract():
    mod = _load_deploy_script("check_unilab_g1_height_tracking_live_path")

    class FakeBackend:
        def get_sensor_data(self, name):
            if name == "upvector":
                return np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32)
            if name.startswith(("left_foot_contact_", "right_foot_contact_")):
                return np.asarray([1.0], dtype=np.float32)
            raise KeyError(name)

    obs = np.zeros((1, 99), dtype=np.float32)
    obs[:, 96] = 0.7
    state = types.SimpleNamespace(
        obs={
            "obs": obs,
            "critic": np.zeros((1, 102), dtype=np.float32),
        },
        info={
            "commands": np.zeros((1, 3), dtype=np.float32),
            "height_commands": np.asarray([[0.7]], dtype=np.float32),
            "log": {"reward/track_base_height_exp_smooth": 1.0},
        },
        reward=np.asarray([1.0], dtype=np.float32),
        terminated=np.asarray([False]),
    )

    class FakeEnv:
        num_envs = 1
        action_space = types.SimpleNamespace(shape=(29,))
        obs_groups_spec = {"obs": 99, "critic": 102}
        cfg = types.SimpleNamespace(sensor=types.SimpleNamespace(upvector="upvector"))
        _backend = FakeBackend()
        closed = False

        def init_state(self):
            return state

        def step(self, actions):
            assert actions.shape == (1, 29)
            return state

        def _terrain_relative_base_height(self):
            return np.asarray([0.71], dtype=np.float32)

        def close(self):
            self.closed = True

    fake_env = FakeEnv()
    captured: dict[str, Any] = {}

    def create_env_fn(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return fake_env

    checks, details = mod.run_check(
        task="g1_stand_height",
        num_envs=1,
        steps=1,
        seed=7,
        create_env_fn=create_env_fn,
    )

    assert not any(check.level == "FAIL" for check in checks)
    assert details["height_tracking/config_task"] == "G1StandHeight"
    assert details["height_tracking/obs_dim"] == 99
    assert details["height_tracking/critic_dim"] == 102
    assert details["height_tracking/commands_max_abs"] == 0.0
    assert details["height_tracking/double_support_fraction"] == 1.0
    assert details["height_tracking/terminated_total"] == 0
    assert captured["args"][0].training.task_name == "G1StandHeight"
    assert captured["kwargs"]["sim_backend"] == "mujoco"
    assert fake_env.closed is True


def test_analyze_offpolicy_trace_reports_training_e2e(tmp_path, capsys):
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"name": "learner/wait_for_data", "ph": "X", "ts": 0.0, "dur": 10.0},
                    {"name": "learner/wait_for_data", "ph": "X", "ts": 1000.0, "dur": 10.0},
                    {"name": "learner/training_e2e", "ph": "X", "ts": 0.0, "dur": 2500.0},
                    {"name": "learner/weight_sync_write", "ph": "X", "ts": 800.0, "dur": 100.0},
                    {
                        "name": "learner/update_critic",
                        "ph": "X",
                        "ts": 1200.0,
                        "dur": 50.0,
                        "args": {"update_idx": 0},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    mod = _load_script("analyze_offpolicy_trace")

    mod.analyze_training_e2e(trace_path)
    mod.analyze_iteration_resume_gap(trace_path)

    out = capsys.readouterr().out
    assert "training_e2e: n=1 mean=2.500ms" in out
    assert "weight_sync_end_to_next_update0_start_gap: n=1 mean=0.300ms" in out


def test_g1_distill_playback_live_sentinel_contract(capsys):
    mod = _load_deploy_script("check_unilab_g1_distill_playback_live_sentinel")

    class FakeEnv:
        action_space = types.SimpleNamespace(shape=(29,))
        closed = False

        def close(self):
            self.closed = True

    class FakeSession:
        def __init__(self):
            self.env = FakeEnv()
            self.actions = None
            self.info = {"commands": np.zeros((1, 3), dtype=np.float32)}
            self.reset_count = 0
            self.step_count = 0

        def reset(self):
            self.reset_count += 1
            return np.zeros((1, 99), dtype=np.float32)

        def step_once(self):
            self.step_count += 1
            self.actions = np.zeros((1, 29), dtype=np.float32)
            return np.zeros((1, 99), dtype=np.float32)

        def physics_state(self):
            return np.zeros((1, 75), dtype=np.float32)

    captured: dict[str, Any] = {}

    def fake_create_session(**kwargs):
        captured.update(kwargs)
        kwargs["log"]("Policy obs mode: actor")
        kwargs["log"]("Action mode: zero")
        return FakeSession(), "actor", None

    checks, details = mod.run_check(
        steps=2,
        action_mode="zero",
        load_run="-1",
        checkpoint=None,
        create_session_fn=fake_create_session,
    )
    mod.print_report(checks, details)
    out = capsys.readouterr().out

    assert all(check.level == "PASS" for check in checks)
    assert captured["playback_cfg"].algo_log_name == "distill"
    assert captured["playback_cfg"].task == "G1WalkHeight"
    assert details["distill_playback/policy_obs_mode"] == "actor"
    assert details["distill_playback/action_dim"] == 29
    assert details["distill_playback/physics_shape"] == [1, 75]
    assert details["distill_playback/actions_shape"] == [1, 29]
    assert "UniLab G1 generic distill playback live sentinel" in out
    assert "[PASS] distill_playback/actions: (1, 29)" in out


def test_g1_distill_repeated_reset_probe_reports_standing_contract() -> None:
    import torch

    mod = _load_deploy_script("check_unilab_g1_distill_playback_live_sentinel")

    class FakeBackend:
        def get_base_lin_vel(self):
            return np.zeros((1, 3), dtype=np.float32)

        def get_base_ang_vel(self):
            return np.zeros((1, 3), dtype=np.float32)

    class FakeEnv:
        def __init__(self) -> None:
            self._backend = FakeBackend()
            self.state = types.SimpleNamespace(
                info={
                    "commands": np.zeros((1, 3), dtype=np.float32),
                    "gait_enabled": np.zeros((1,), dtype=np.float32),
                }
            )

        def _command_observation(self, info, num_obs):
            return np.asarray(info["commands"], dtype=np.float32)

    class FakeSession:
        def __init__(self, env: FakeEnv) -> None:
            self.env = env
            self.obs = torch.zeros((1, 98), dtype=torch.float32)
            self.actions = None
            self.policy = None

        @property
        def info(self):
            return self.env.state.info

        def reset(self):
            self.obs.zero_()
            self.actions = None

        def step_once(self):
            self.actions = torch.zeros((1, 29), dtype=torch.float32)

    env = FakeEnv()
    session = FakeSession(env)
    checks, details = mod._run_repeated_reset_probe(session, env, repetitions=2, action_mode="zero")

    assert all(check.level == "PASS" for check in checks)
    assert details["distill_playback/reset_repetitions"] == 2
    assert details["distill_playback/reset_base_qvel_norm_max"] == 0.0
    assert len(details["distill_playback/reset_probe_records"]) == 2


def test_g1_distill_viewer_path_preflight_reaches_viewer_model(tmp_path: Path, monkeypatch, capsys):
    mod = _load_deploy_script("check_unilab_g1_distill_viewer_path")
    checkpoint = tmp_path / "model_2.pt"
    checkpoint.write_bytes(b"checkpoint")

    class FakeEnv:
        closed = False

        def close(self):
            self.closed = True

    class FakeSession:
        def __init__(self):
            self.env = FakeEnv()
            self.actions = np.full((1, 29), 0.1, dtype=np.float32)

        def reset(self):
            return np.zeros((1, 98), dtype=np.float32)

        def step_once(self):
            return np.zeros((1, 98), dtype=np.float32)

        def physics_state(self):
            return np.zeros((1, 72), dtype=np.float32)

    class FakeViewerModel:
        nq = 35
        nv = 34
        nu = 29

    captured: dict[str, Any] = {}

    def fake_create_session(**kwargs):
        captured.update(kwargs)
        kwargs["log"]("Policy obs mode: actor")
        kwargs["log"]("Action mode: policy")
        return FakeSession(), "actor", str(checkpoint)

    monkeypatch.setattr(mod.shutil, "which", lambda name: "/usr/local/bin/mjpython")
    checks, details = mod.run_check(
        task="g1_stand_still/mujoco",
        action_mode="policy",
        load_run=str(checkpoint),
        checkpoint=None,
        device="cpu",
        create_session_fn=fake_create_session,
        load_viewer_model_fn=lambda env, *, use_env_visual_model: FakeViewerModel(),
        state_transfer_fn=lambda model, physics: {"qpos_shape": [35], "qvel_shape": [34]},
    )
    mod.print_report(checks, details)
    out = capsys.readouterr().out

    assert all(check.level == "PASS" for check in checks)
    assert captured["playback_cfg"].task == "G1StandStill"
    assert captured["playback_cfg"].action_mode == "policy"
    assert details["distill_viewer/task_owner"] == "g1_stand_still/mujoco"
    assert details["distill_viewer/cfg_student_obs_dim"] == 98
    assert details["distill_viewer/cfg_teacher_obs_dim"] == 98
    assert details["distill_viewer/checkpoint_path"] == str(checkpoint)
    assert (
        "mjpython scripts/play_interactive.py --algo distill"
        in details["distill_viewer/viewer_command"]
    )
    assert "UniLab G1 generic distill viewer path preflight" in out
    assert "[PASS] distill_viewer/state_transfer" in out


def test_g1_distill_moe_expert_semantics_checker_reads_role_labels(tmp_path: Path, capsys):
    import torch

    from unilab.algos.torch.distill import (
        MoEStudentPolicy,
        build_distillation_dataset,
        save_distillation_checkpoint,
        save_distillation_dataset,
    )

    mod = _load_deploy_script("check_unilab_g1_distill_moe_expert_semantics")
    student = MoEStudentPolicy(
        obs_dim=4,
        action_dim=2,
        num_experts=3,
        expert_hidden_dims=(),
        router_hidden_dims=(),
        routing_mode="hard",
        squash_action=False,
    )
    with torch.no_grad():
        student.router[-1].weight.zero_()
        student.router[-1].bias.zero_()
        student.router[-1].weight[0, 0] = 4.0
        student.router[-1].weight[1, 1] = 4.0
        student.router[-1].weight[2, 2] = 4.0
    checkpoint_path = tmp_path / "moe_student.pt"
    save_distillation_checkpoint(
        checkpoint_path,
        student=student,
        agent_steps=6,
        distill_runtime_cfg={
            "student_model_type": "moe",
            "student_obs_dim": 4,
            "student_action_dim": 2,
            "student_num_experts": 3,
            "student_expert_hidden_dims": [],
            "student_router_hidden_dims": [],
            "student_routing_mode": "hard",
            "student_router_temperature": 1.0,
            "student_activation": "elu",
            "student_squash_action": False,
            "teacher_obs_dim": 4,
        },
    )
    dataset_path = tmp_path / "dataset.pt"
    dataset = build_distillation_dataset(
        torch.tensor(
            [
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        torch.zeros(3, 4),
        expected_student_obs_dim=4,
        expected_teacher_obs_dim=4,
        metadata={"role_labels": ["stand", "walk", "recovery"]},
    )
    save_distillation_dataset(dataset_path, dataset)

    checks, details = mod.run_check(
        task="g1_stand_still/mujoco",
        dataset_path=dataset_path,
        student_checkpoint=checkpoint_path,
        device="cpu",
        hard_routing=True,
        collapse_fraction=0.95,
    )
    mod.print_report(checks, details)
    out = capsys.readouterr().out

    assert not any(check.level == "FAIL" for check in checks)
    assert details["moe_expert/student_model_type"] == "moe"
    assert details["moe_expert/role_labels_present"] is True
    by_role = {item["role"]: item for item in details["moe_expert/diagnostics"]["by_role"]}
    assert by_role["stand"]["dominant_expert"] == 0
    assert by_role["walk"]["dominant_expert"] == 1
    assert by_role["recovery"]["dominant_expert"] == 2
    assert "[PASS] moe_expert/role_labels" in out
    assert "[PASS] moe_expert/collapse_guard" in out


def test_g1_distill_teacher_obs_contract_reports_live_identity(capsys):
    mod = _load_deploy_script("check_unilab_g1_distill_teacher_obs_contract")
    cfg = _distill_cfg()

    class FakeEnv:
        obs_groups_spec = {"obs": 99, "critic": 102}
        closed = False

        def reset(self, env_indices):
            assert env_indices.shape == (1,)
            return {
                "obs": np.zeros((1, 99), dtype=np.float32),
                "critic": np.zeros((1, 102), dtype=np.float32),
            }, {}

        def close(self):
            self.closed = True

    fake_env = FakeEnv()
    captured: dict[str, Any] = {}

    def create_env_fn(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return fake_env

    checks, details = mod.run_check(
        cfg=cfg,
        create_env_fn=create_env_fn,
        env_cfg_override_fn=lambda cfg: {"owner": "teacher-obs-test"},
    )
    mod.print_report(checks, details)
    out = capsys.readouterr().out

    assert not any(check.level == "FAIL" for check in checks)
    assert any(
        check.level == "PASS"
        and check.name == "distill_teacher_obs/teacher_live_dim"
        and check.detail == "99"
        for check in checks
    )
    assert any(
        check.level == "WARN" and check.name == "distill_teacher_obs/checkpoint_input_dim"
        for check in checks
    )
    assert captured["kwargs"]["env_cfg_override"] == {"owner": "teacher-obs-test"}
    assert captured["kwargs"]["task_name"] == "G1WalkHeight"
    assert details["distill_teacher_obs/live_obs_shape"] == (1, 99)
    assert details["distill_teacher_obs/live_critic_shape"] == (1, 102)
    assert fake_env.closed is True
    assert "UniLab G1 generic distill teacher obs contract audit" in out


def test_g1_distill_teacher_obs_contract_checks_checkpoint_input_dim(tmp_path: Path):
    import torch

    mod = _load_deploy_script("check_unilab_g1_distill_teacher_obs_contract")
    cfg = _distill_cfg()
    checkpoint_path = tmp_path / "teacher.pt"
    torch.save({"actor": {"net.0.weight": torch.zeros((512, 99))}}, checkpoint_path)

    class FakeEnv:
        obs_groups_spec = {"obs": 99, "critic": 102}

        def reset(self, env_indices):
            return {
                "obs": np.zeros((1, 99), dtype=np.float32),
                "critic": np.zeros((1, 102), dtype=np.float32),
            }, {}

    checks, details = mod.run_check(
        checkpoint_path=checkpoint_path,
        cfg=cfg,
        create_env_fn=lambda *args, **kwargs: FakeEnv(),
        env_cfg_override_fn=lambda cfg: {},
    )

    assert not any(check.level == "FAIL" for check in checks)
    assert any(
        check.level == "PASS" and check.name == "distill_teacher_obs/checkpoint_input_dim"
        for check in checks
    )
    assert details["distill_teacher_obs/checkpoint_first_weight"] == "net.0.weight"
    assert details["distill_teacher_obs/checkpoint_input_dim"] == 99


def test_g1_distill_teacher_obs_contract_reports_legacy_projection_bridge(capsys):
    mod = _load_deploy_script("check_unilab_g1_distill_teacher_obs_contract")
    cfg = _distill_cfg(["teacher.obs_dim=100", "training.collect_teacher_projection=pad_zeros"])

    class FakeEnv:
        obs_groups_spec = {"obs": 99, "critic": 102}

        def reset(self, env_indices):
            return {
                "obs": np.zeros((1, 99), dtype=np.float32),
                "critic": np.zeros((1, 102), dtype=np.float32),
            }, {}

    checks, details = mod.run_check(
        cfg=cfg,
        create_env_fn=lambda *args, **kwargs: FakeEnv(),
        env_cfg_override_fn=lambda cfg: {},
    )
    mod.print_report(checks, details)
    out = capsys.readouterr().out

    assert not any(check.level == "FAIL" for check in checks)
    assert any(
        check.level == "WARN"
        and check.name == "distill_teacher_obs/projection_bridge"
        and "live_obs=99 -> teacher=100" in check.detail
        for check in checks
    )
    assert details["distill_teacher_obs/teacher_obs_dim"] == 100
    assert details["distill_teacher_obs/teacher_projection"] == "pad_zeros"
    assert "synthetic_tail=1" in out


def test_g1_distill_teacher_obs_contract_reports_stand_still_identity(
    tmp_path: Path,
    capsys,
):
    import torch

    mod = _load_deploy_script("check_unilab_g1_distill_teacher_obs_contract")
    cfg = _distill_cfg(["task=g1_stand_still/mujoco"])
    checkpoint_path = tmp_path / "stand_teacher.pt"
    torch.save({"actor": {"net.0.weight": torch.zeros((512, 98))}}, checkpoint_path)

    class FakeEnv:
        obs_groups_spec = {"obs": 98, "critic": 101}

        def reset(self, env_indices):
            return {
                "obs": np.zeros((1, 98), dtype=np.float32),
                "critic": np.zeros((1, 101), dtype=np.float32),
            }, {}

    checks, details = mod.run_check(
        task="g1_stand_still/mujoco",
        checkpoint_path=checkpoint_path,
        cfg=cfg,
        create_env_fn=lambda *args, **kwargs: FakeEnv(),
        env_cfg_override_fn=lambda cfg: {},
    )
    mod.print_report(checks, details)
    out = capsys.readouterr().out

    assert not any(check.level == "FAIL" for check in checks)
    assert any(
        check.level == "PASS"
        and check.name == "distill_teacher_obs/teacher_live_dim"
        and check.detail == "98"
        for check in checks
    )
    assert any(
        check.level == "PASS" and check.name == "distill_teacher_obs/checkpoint_input_dim"
        for check in checks
    )
    assert details["distill_teacher_obs/task_owner"] == "g1_stand_still/mujoco"
    assert details["distill_teacher_obs/task"] == "G1StandStill"
    assert details["distill_teacher_obs/teacher_obs_dim"] == 98
    assert details["distill_teacher_obs/student_obs_dim"] == 98
    assert details["distill_teacher_obs/teacher_projection"] == "identity"
    assert details["distill_teacher_obs/live_obs_shape"] == (1, 98)
    assert details["distill_teacher_obs/live_critic_shape"] == (1, 101)
    assert details["distill_teacher_obs/checkpoint_input_dim"] == 98
    assert "[PASS] distill_teacher_obs/teacher_live_dim: 98" in out


def test_g1_distill_playback_live_sentinel_policy_checkpoint_contract(capsys):
    from unilab.algos.torch.distill import load_distillation_student_policy

    mod = _load_deploy_script("check_unilab_g1_distill_playback_live_sentinel")

    class FakeEnv:
        action_space = types.SimpleNamespace(shape=(29,))

    class FakeSession:
        env = FakeEnv()
        info = {"commands": np.zeros((1, 3), dtype=np.float32)}
        actions = None

        def reset(self):
            return np.zeros((1, 99), dtype=np.float32)

        def step_once(self):
            self.actions = np.full((1, 29), 0.05, dtype=np.float32)
            return np.zeros((1, 99), dtype=np.float32)

        def physics_state(self):
            return np.zeros((1, 75), dtype=np.float32)

    captured: dict[str, Any] = {}

    def fake_create_session(**kwargs):
        captured.update(kwargs)
        checkpoint = Path(kwargs["playback_cfg"].load_run) / "model_1.pt"
        loaded = load_distillation_student_policy(checkpoint, device="cpu")
        captured["loaded"] = loaded
        kwargs["log"](f"Loading distillation student checkpoint: {checkpoint}")
        return FakeSession(), "actor", str(checkpoint)

    checks, details = mod.run_check(
        steps=1,
        action_mode="policy",
        load_run="-1",
        checkpoint=None,
        make_temp_policy_checkpoint=True,
        create_session_fn=fake_create_session,
    )
    mod.print_report(checks, details)
    out = capsys.readouterr().out

    assert all(check.level == "PASS" for check in checks)
    assert captured["playback_cfg"].action_mode == "policy"
    assert details["distill_playback/checkpoint_path"].endswith("model_1.pt")
    assert details["distill_playback/actions_abs_max"] == pytest.approx(0.05)
    assert "[PASS] distill_playback/policy_checkpoint:" in out
    assert "[PASS] distill_playback/policy_action_nonzero: 0.050000" in out
    assert captured["loaded"].obs_dim == 99
    assert captured["loaded"].action_dim == 29


def test_g1_distill_playback_live_sentinel_stand_still_policy_checkpoint_contract(
    capsys,
):
    from unilab.algos.torch.distill import load_distillation_student_policy

    mod = _load_deploy_script("check_unilab_g1_distill_playback_live_sentinel")

    class FakeEnv:
        action_space = types.SimpleNamespace(shape=(29,))

    class FakeSession:
        env = FakeEnv()
        info = {"commands": np.zeros((1, 3), dtype=np.float32)}
        actions = None

        def reset(self):
            return np.zeros((1, 98), dtype=np.float32)

        def step_once(self):
            self.actions = np.full((1, 29), 0.05, dtype=np.float32)
            return np.zeros((1, 98), dtype=np.float32)

        def physics_state(self):
            return np.zeros((1, 75), dtype=np.float32)

    captured: dict[str, Any] = {}

    def fake_create_session(**kwargs):
        captured.update(kwargs)
        checkpoint = Path(kwargs["playback_cfg"].load_run) / "model_1.pt"
        loaded = load_distillation_student_policy(checkpoint, device="cpu")
        captured["loaded"] = loaded
        kwargs["log"](f"Loading distillation student checkpoint: {checkpoint}")
        return FakeSession(), "actor", str(checkpoint)

    checks, details = mod.run_check(
        steps=1,
        task="g1_stand_still/mujoco",
        action_mode="policy",
        load_run="-1",
        checkpoint=None,
        make_temp_policy_checkpoint=True,
        create_session_fn=fake_create_session,
    )
    mod.print_report(checks, details)
    out = capsys.readouterr().out

    assert all(check.level == "PASS" for check in checks)
    assert captured["playback_cfg"].task == "G1StandStill"
    assert details["distill_playback/task"] == "G1StandStill"
    assert details["distill_playback/task_owner"] == "g1_stand_still/mujoco"
    assert details["distill_playback/cfg_student_obs_dim"] == 98
    assert details["distill_playback/cfg_teacher_obs_dim"] == 98
    assert details["distill_playback/actions_abs_max"] == pytest.approx(0.05)
    assert captured["loaded"].obs_dim == 98
    assert captured["loaded"].action_dim == 29
    assert "[PASS] distill_playback/policy_action_nonzero: 0.050000" in out


# ---------------------------------------------------------------------------
def test_g1_distill_playback_live_sentinel_moe_policy_checkpoint_contract(capsys):
    from unilab.algos.torch.distill import MoEStudentPolicy, load_distillation_student_policy

    mod = _load_deploy_script("check_unilab_g1_distill_playback_live_sentinel")

    class FakeEnv:
        action_space = types.SimpleNamespace(shape=(29,))

    class FakeSession:
        env = FakeEnv()
        info = {"commands": np.zeros((1, 3), dtype=np.float32)}
        actions = None
        policy = types.SimpleNamespace(
            _unilab_distill_command_routing_mode="hard",
            _unilab_distill_command_routing_applied=True,
            _unilab_distill_last_command_intents=("inactive",),
            _unilab_distill_last_expected_experts=(1,),
            _unilab_distill_last_selected_experts=(1,),
        )

        def reset(self):
            return np.zeros((1, 99), dtype=np.float32)

        def step_once(self):
            self.actions = np.full((1, 29), 0.05, dtype=np.float32)
            return np.zeros((1, 99), dtype=np.float32)

        def physics_state(self):
            return np.zeros((1, 75), dtype=np.float32)

    captured: dict[str, Any] = {}

    def fake_create_session(**kwargs):
        captured.update(kwargs)
        checkpoint = Path(kwargs["playback_cfg"].load_run) / "model_1.pt"
        loaded = load_distillation_student_policy(checkpoint, device="cpu")
        captured["loaded"] = loaded
        kwargs["log"](f"Loading distillation student checkpoint: {checkpoint}")
        return FakeSession(), "actor", str(checkpoint)

    checks, details = mod.run_check(
        steps=1,
        action_mode="policy",
        load_run="-1",
        checkpoint=None,
        make_temp_policy_checkpoint=True,
        temp_student_model_type="moe",
        create_session_fn=fake_create_session,
    )
    mod.print_report(checks, details)
    out = capsys.readouterr().out

    assert all(check.level == "PASS" for check in checks)
    assert captured["playback_cfg"].action_mode == "policy"
    assert details["distill_playback/temp_student_model_type"] == "moe"
    assert details["distill_playback/checkpoint_path"].endswith("model_1.pt")
    assert details["distill_playback/actions_abs_max"] == pytest.approx(0.05)
    assert details["distill_playback/command_routing_mode"] == "hard"
    assert details["distill_playback/command_intents"] == ["inactive"]
    assert details["distill_playback/command_expected_experts"] == [1]
    assert details["distill_playback/command_selected_experts"] == [1]
    assert "Loading distillation student checkpoint:" in out
    assert "[PASS] distill_playback/command_routing_contract: inactive->1" in out
    assert isinstance(captured["loaded"].policy, MoEStudentPolicy)
    assert captured["loaded"].obs_dim == 99
    assert captured["loaded"].action_dim == 29
    assert captured["loaded"].distill_runtime_cfg["student_model_type"] == "moe"


# Helpers
# ---------------------------------------------------------------------------


def _mlx_runtime_usable() -> bool:
    """Probe whether importing mlx.core is safe in a subprocess on this host."""
    if sys.platform != "darwin":
        return False
    if importlib.util.find_spec("mlx.core") is None:
        return False
    result = subprocess.run(
        [sys.executable, "-c", "import mlx.core"], capture_output=True, text=True, timeout=10
    )
    return result.returncode == 0


_HAS_MLX = _mlx_runtime_usable()

try:
    import mujoco  # noqa: F401

    _HAS_MUJOCO = True
except ImportError:
    _HAS_MUJOCO = False


# ---------------------------------------------------------------------------
# train_offpolicy.py — Hydra config defaults
# ---------------------------------------------------------------------------


def _offpolicy_cfg(overrides=None):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONF_DIR / "offpolicy"), version_base="1.3"):
        return compose(
            "config",
            overrides=_normalize_overrides(overrides, offpolicy=True),
            return_hydra_config=True,
        )


def _ppo_cfg(overrides=None):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONF_DIR / "ppo"), version_base="1.3"):
        return compose("config", overrides=_normalize_overrides(overrides))


def _appo_cfg(overrides=None):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONF_DIR / "appo"), version_base="1.3"):
        return compose("config", overrides=_normalize_overrides(overrides))


def _hora_distill_cfg(overrides=None):
    """Compose the HORA distillation Hydra config.

    Args:
        overrides: Optional Hydra override strings to apply during composition.

    Returns:
        The composed HORA distillation config.
    """
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONF_DIR / "hora_distill"), version_base="1.3"):
        return compose("config", overrides=overrides or [])


def _distill_cfg(overrides=None):
    """Compose the generic behavior distillation Hydra config."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONF_DIR / "distill"), version_base="1.3"):
        return compose("config", overrides=overrides or [])


def _train_rsl_rl(monkeypatch: pytest.MonkeyPatch):
    import types

    for module_name in list(sys.modules):
        if module_name == "unilab" or module_name.startswith("unilab."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    runners_mod = cast(Any, types.ModuleType("rsl_rl.runners"))
    runners_mod.OnPolicyRunner = object
    rsl_pkg = cast(Any, types.ModuleType("rsl_rl"))
    rsl_pkg.runners = runners_mod
    monkeypatch.setitem(sys.modules, "rsl_rl", rsl_pkg)
    monkeypatch.setitem(sys.modules, "rsl_rl.runners", runners_mod)
    return _load_script("train_rsl_rl")


def _train_appo():
    return _load_script("train_appo")


def _train_hora_distill():
    """Load the HORA distillation entrypoint module.

    Args:
        None.

    Returns:
        The loaded ``scripts/train_hora_distill.py`` module.
    """
    return _load_script("train_hora_distill")


def _train_distill():
    """Load the generic behavior distillation entrypoint module."""
    return _load_script("train_distill")


def test_distill_torch_serialization_runtime_sentinel_reports_callable_identity(capsys):
    mod = _train_distill()

    snapshot = mod._probe_torch_serialization_runtime("workflow/after_bootstrap")

    output = capsys.readouterr().out.strip()
    assert output.startswith("[distill-runtime-sentinel] ")
    payload = json.loads(output.removeprefix("[distill-runtime-sentinel] "))
    assert payload == snapshot
    assert payload["stage"] == "workflow/after_bootstrap"
    assert payload["pid"] == mod.os.getpid()
    assert payload["is_storage_type"] == "function"
    assert payload["is_storage_callable"] is True
    assert payload["is_storage_module"] == "torch"


def test_distill_torch_serialization_runtime_sentinel_fails_fast_on_cell(
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    mod = _train_distill()
    marker = object()
    cell = (lambda: marker).__closure__[0]
    monkeypatch.setattr(mod.torch, "is_storage", cell)

    with pytest.raises(
        RuntimeError,
        match=(
            "torch serialization runtime identity corrupted: "
            "stage=workflow/iteration_1/before_aggregate .*type=cell callable=False"
        ),
    ):
        mod._probe_torch_serialization_runtime("workflow/iteration_1/before_aggregate")

    payload = json.loads(
        capsys.readouterr().out.strip().removeprefix("[distill-runtime-sentinel] ")
    )
    assert payload["stage"] == "workflow/iteration_1/before_aggregate"
    assert payload["is_storage_type"] == "cell"
    assert payload["is_storage_callable"] is False
    assert payload["is_storage_module"] is None


def test_distill_script_cli_result_compacts_large_metadata():
    mod = _train_distill()

    result = {
        "distill_source": "offline_dataset",
        "dataset_metadata": {
            "role_labels": ["stand"] * 20 + ["walk_flat"] * 20,
            "command_intents": ["inactive"] * 20 + ["active"] * 20,
            "source_metadata": [
                {"command_intents": ["inactive"] * 32},
                {"role_labels": ["walk_flat"] * 32},
            ],
        },
        "offline_batch_label_counts": [{"stand": 1, "walk_flat": 1}] * 200,
    }

    compact = mod._compact_cli_result(result)
    rendered = mod._format_cli_result(result)

    assert compact["dataset_metadata"]["role_labels"] == {
        "count": 40,
        "head": ["stand", "stand", "stand", "stand"],
        "tail": ["walk_flat", "walk_flat", "walk_flat", "walk_flat"],
        "counts": {"stand": 20, "walk_flat": 20},
    }
    assert compact["dataset_metadata"]["command_intents"]["counts"] == {
        "inactive": 20,
        "active": 20,
    }
    assert compact["dataset_metadata"]["source_metadata"][0]["command_intents"]["count"] == 32
    assert compact["offline_batch_label_counts"]["count"] == 200
    assert len(rendered) < 1500


def test_distill_main_routes_enabled_single_entry_workflow(monkeypatch, capsys):
    mod = _train_distill()
    cfg = _distill_cfg(["training.workflow.enabled=true"])
    captured = {}

    def fake_workflow(actual_cfg):
        captured["cfg"] = actual_cfg
        return {"distill_source": "single_entry_workflow", "stage": "BOOTSTRAP_COMPLETE"}

    monkeypatch.setattr(mod, "run_single_entry_workflow", fake_workflow)

    mod.main.__wrapped__(cfg)

    assert captured["cfg"] is cfg
    assert '"distill_source": "single_entry_workflow"' in capsys.readouterr().out


def test_distill_native_fail_stop_aborts_on_any_unhandled_diagnostic_error(
    monkeypatch,
) -> None:
    mod = _train_distill()

    class NativeAbortRequestedError(RuntimeError):
        pass

    def fail_main() -> None:
        raise ValueError("moving semantic failure")

    def request_abort() -> None:
        raise NativeAbortRequestedError

    monkeypatch.setenv("UNILAB_NATIVE_ABORT_ON_CORRUPTION", "1")
    monkeypatch.setattr(mod, "main", fail_main)
    monkeypatch.setattr(mod.os, "abort", request_abort)

    with pytest.raises(NativeAbortRequestedError):
        mod._run_main_with_native_fail_stop()


def test_distill_native_fail_stop_preserves_original_error_when_disabled(
    monkeypatch,
) -> None:
    mod = _train_distill()

    def fail_main() -> None:
        raise ValueError("ordinary failure")

    monkeypatch.setenv("UNILAB_NATIVE_ABORT_ON_CORRUPTION", "0")
    monkeypatch.setattr(mod, "main", fail_main)
    monkeypatch.setattr(
        mod.os,
        "abort",
        lambda: pytest.fail("disabled native fail-stop must not abort"),
    )

    with pytest.raises(ValueError, match="ordinary failure"):
        mod._run_main_with_native_fail_stop()


def test_distill_walk_stand_workflow_profile_composes_teacher_roles(monkeypatch):
    monkeypatch.setenv("UNILAB_G1_WALK_TEACHER", "/models/walk.pt")
    monkeypatch.setenv("UNILAB_G1_STAND_TEACHER", "/models/stand.pt")
    monkeypatch.setenv("UNILAB_G1_WALK_DATASET", "/data/walk.pt")
    monkeypatch.setenv("UNILAB_G1_STAND_DATASET", "/data/stand.pt")

    cfg = _distill_cfg(["workflow=g1_walk_stand"])

    roles = OmegaConf.to_container(cfg.training.workflow.roles, resolve=True)
    assert cfg.teacher.obs_dim == 98
    assert cfg.student.obs_dim == 98
    assert cfg.algo.expert_behavior_loss_source == "auto"
    assert cfg.training.workflow.transition_min_post_switch_steps == 20
    assert cfg.training.workflow.transition_nominal_settle_steps == 0
    assert cfg.training.workflow.dagger_min_transition_replay_passes == 8
    assert cfg.training.workflow.dagger_min_transition_replay_labels == ["walk_to_stop"]
    assert roles == [
        {
            "role": "walk_flat",
            "task": "g1_walk_flat/mujoco",
            "teacher_checkpoint_path": "/models/walk.pt",
            "dataset_path": "/data/walk.pt",
        },
        {
            "role": "stand",
            "task": "g1_stand_still/mujoco",
            "teacher_checkpoint_path": "/models/stand.pt",
            "dataset_path": "/data/stand.pt",
        },
    ]


def test_distill_single_entry_uses_task_owner_and_generated_artifact_paths(
    tmp_path: Path, monkeypatch
):
    from types import SimpleNamespace

    mod = _train_distill()
    cfg = _distill_cfg(["training.workflow.enabled=true"])
    teacher_path = tmp_path / "teacher.pt"
    teacher_path.write_bytes(b"teacher")
    cfg.training.workflow.run_dir = str(tmp_path / "run")
    cfg.training.workflow.artifact_dir = str(tmp_path / "artifacts")
    cfg.training.workflow.roles = [
        {
            "role": "stand",
            "task": "g1_stand_still/mujoco",
            "teacher_checkpoint_path": str(teacher_path),
        }
    ]
    captured = {}

    def fake_bootstrap(**kwargs):
        captured.update(kwargs)
        run_dir = Path(kwargs["run_dir"])
        return SimpleNamespace(
            run_dir=run_dir,
            manifest_path=run_dir / "run_manifest.json",
            role_decisions={"stand": "COLLECT"},
            bootstrap_dataset_path=run_dir / "datasets" / "bootstrap_merged.pt",
            bootstrap_num_samples=8,
            checkpoint_path=run_dir / "checkpoints" / "bootstrap_student.pt",
            bootstrap_updates=2,
        )

    monkeypatch.setattr(distill_entry_workflow, "run_bootstrap_workflow", fake_bootstrap)

    def fake_dagger(**kwargs):
        captured["dagger_kwargs"] = kwargs
        run_dir = Path(kwargs["run_dir"])
        return SimpleNamespace(
            run_dir=run_dir,
            manifest_path=run_dir / "run_manifest.json",
            completed_iterations=8,
            checkpoint_path=run_dir / "checkpoints" / "dagger_iteration_8.pt",
            cumulative_num_samples=24,
        )

    monkeypatch.setattr(distill_entry_workflow, "run_multirole_dagger_workflow", fake_dagger)
    monkeypatch.setattr(
        distill_entry_workflow,
        "finalize_workflow_performance",
        lambda **kwargs: captured.setdefault("finalize_kwargs", kwargs),
    )

    result = mod.run_single_entry_workflow(cfg)

    spec = captured["role_specs"][0]
    assert spec.role == "stand"
    assert spec.task == "g1_stand_still/mujoco"
    assert spec.dataset_path == tmp_path / "artifacts" / "stand.pt"
    assert spec.command_sample_filter == "inactive"
    assert captured["dagger_kwargs"]["execution_mode"] == "legacy"
    assert callable(captured["dagger_kwargs"]["collect_scenario"]) is False
    assert captured["dagger_kwargs"]["scenario_collector"] is None
    performance_context = captured["dagger_kwargs"]["performance_context"]
    assert performance_context.execution_mode == "legacy"
    assert performance_context.teacher_checkpoint_sha256 == (mod.file_sha256(teacher_path),)
    assert result["execution_mode"] == "legacy"
    assert result["checkpoint_path"].endswith("checkpoints/dagger_iteration_8.pt")


def test_distill_workflow_dagger_update_does_not_resume_or_save_optimizer_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    mod = _train_distill()
    cfg = _distill_cfg(["training.workflow.enabled=true"])
    teacher_path = tmp_path / "teacher.pt"
    teacher_path.write_bytes(b"teacher")
    cfg.training.workflow.run_dir = str(tmp_path / "run")
    cfg.training.workflow.artifact_dir = str(tmp_path / "artifacts")
    cfg.training.workflow.dagger_updates_per_iteration = 3
    cfg.training.workflow.roles = [
        {
            "role": "stand",
            "task": "g1_stand_still/mujoco",
            "teacher_checkpoint_path": str(teacher_path),
        }
    ]

    monkeypatch.setattr(
        distill_entry_workflow,
        "run_bootstrap_workflow",
        lambda **kwargs: SimpleNamespace(
            run_dir=Path(kwargs["run_dir"]),
            manifest_path=Path(kwargs["run_dir"]) / "run_manifest.json",
            role_decisions={"stand": "COLLECT"},
            bootstrap_dataset_path=Path(kwargs["run_dir"]) / "datasets" / "bootstrap_merged.pt",
            bootstrap_num_samples=8,
            checkpoint_path=Path(kwargs["run_dir"]) / "checkpoints" / "bootstrap_student.pt",
            bootstrap_updates=2,
        ),
    )

    captured_update_cfgs: list[Any] = []

    def fake_offline_update(update_cfg, **_kwargs):
        captured_update_cfgs.append(update_cfg)
        return {
            "update_count": 3,
            "performance_stage_observations": [
                {
                    "stage": "learner_batch_staging",
                    "duration_seconds": 0.0,
                    "row_count": 1,
                    "env_step_count": 0,
                    "success": True,
                    "error": None,
                    "cleanup_state": "not_applicable",
                },
                {
                    "stage": "learner_forward",
                    "duration_seconds": 0.0,
                    "row_count": 1,
                    "env_step_count": 0,
                    "success": True,
                    "error": None,
                    "cleanup_state": "not_applicable",
                },
                {
                    "stage": "learner_backward",
                    "duration_seconds": 0.0,
                    "row_count": 1,
                    "env_step_count": 0,
                    "success": True,
                    "error": None,
                    "cleanup_state": "not_applicable",
                },
                {
                    "stage": "checkpoint_save",
                    "duration_seconds": 0.0,
                    "row_count": 1,
                    "env_step_count": 0,
                    "success": True,
                    "error": None,
                    "cleanup_state": "not_applicable",
                },
            ],
        }

    def fake_dagger(**kwargs):
        update_result = kwargs["update_student"](
            tmp_path / "aggregate.pt",
            tmp_path / "input_student.pt",
            tmp_path / "output_student.pt",
        )
        assert update_result.updates == 3
        run_dir = Path(kwargs["run_dir"])
        return SimpleNamespace(
            run_dir=run_dir,
            manifest_path=run_dir / "run_manifest.json",
            completed_iterations=1,
            checkpoint_path=run_dir / "checkpoints" / "dagger_iteration_1.pt",
            cumulative_num_samples=24,
        )

    monkeypatch.setattr(distill_entry_workflow, "run_offline_dataset_update", fake_offline_update)
    monkeypatch.setattr(distill_entry_workflow, "run_multirole_dagger_workflow", fake_dagger)
    monkeypatch.setattr(
        distill_entry_workflow, "finalize_workflow_performance", lambda **_kwargs: None
    )

    mod.run_single_entry_workflow(cfg)

    assert len(captured_update_cfgs) == 1
    update_cfg = captured_update_cfgs[0]
    assert update_cfg.training.offline_resume_optimizer is False
    assert update_cfg.training.offline_save_optimizer is False
    assert update_cfg.training.offline_init_checkpoint == str(tmp_path / "input_student.pt")


def test_distill_single_entry_persistent_execution_routes_factory_and_closes_service(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    mod = _train_distill()
    cfg = _distill_cfg(["training.workflow.enabled=true"])
    teacher_path = tmp_path / "stand_teacher.pt"
    teacher_path.write_bytes(b"stand-teacher")
    walk_teacher_path = tmp_path / "walk_teacher.pt"
    walk_teacher_path.write_bytes(b"walk-teacher")
    cfg.training.workflow.run_dir = str(tmp_path / "persistent_run")
    cfg.training.workflow.artifact_dir = str(tmp_path / "artifacts")
    cfg.training.workflow.execution_mode = "persistent_async"
    cfg.training.workflow.roles = [
        {
            "role": "stand",
            "task": "g1_stand_still/mujoco",
            "teacher_checkpoint_path": str(teacher_path),
        },
        {
            "role": "walk_flat",
            "task": "g1_walk_flat/mujoco",
            "teacher_checkpoint_path": str(walk_teacher_path),
        },
    ]
    scenarios = (
        mod.WorkflowScenarioSpec("stand", "role", ("stand",), 0.5),
        mod.WorkflowScenarioSpec("walk_flat", "role", ("walk_flat",), 0.5),
    )
    monkeypatch.setattr(distill_entry_plan, "_workflow_scenario_specs", lambda *_args: scenarios)
    monkeypatch.setattr(
        distill_entry_workflow,
        "run_bootstrap_workflow",
        lambda **kwargs: SimpleNamespace(
            run_dir=Path(kwargs["run_dir"]),
            manifest_path=Path(kwargs["run_dir"]) / "run_manifest.json",
            role_decisions={"stand": "COLLECT", "walk_flat": "COLLECT"},
            bootstrap_dataset_path=Path(kwargs["run_dir"]) / "datasets" / "bootstrap_merged.pt",
            bootstrap_num_samples=8,
            checkpoint_path=Path(kwargs["run_dir"]) / "checkpoints" / "bootstrap_student.pt",
            bootstrap_updates=2,
        ),
    )
    service = MagicMock()
    service.close_report = {
        "worker_pid": 1234,
        "resource_counters": {"env_builds": 1},
    }
    factory_inputs: dict[str, Any] = {}

    def fake_factory(**kwargs):
        factory_inputs.update(kwargs)
        return service

    dagger_inputs: dict[str, Any] = {}

    def fake_dagger(**kwargs):
        dagger_inputs.update(kwargs)
        run_dir = Path(kwargs["run_dir"])
        return SimpleNamespace(
            run_dir=run_dir,
            manifest_path=run_dir / "run_manifest.json",
            completed_iterations=1,
            checkpoint_path=run_dir / "checkpoints" / "dagger_iteration_1.pt",
            cumulative_num_samples=24,
        )

    monkeypatch.setattr(distill_entry_workflow, "run_multirole_dagger_workflow", fake_dagger)
    monkeypatch.setattr(
        distill_entry_workflow,
        "finalize_workflow_performance",
        lambda **kwargs: dagger_inputs.setdefault("finalize_kwargs", kwargs),
    )

    result = mod.run_single_entry_workflow(
        cfg,
        persistent_scenario_collector_factory=fake_factory,
    )

    assert factory_inputs["cfg"] is cfg
    assert set(factory_inputs["role_cfgs"]) == {"stand", "walk_flat"}
    assert [spec.role for spec in factory_inputs["role_specs"]] == [
        "stand",
        "walk_flat",
    ]
    assert factory_inputs["scenario_specs"] == scenarios
    assert dagger_inputs["execution_mode"] == "persistent_async"
    assert dagger_inputs["collect_scenario"] is None
    assert dagger_inputs["scenario_collector"] is service
    assert dagger_inputs["runtime_sentinel"] is mod._probe_torch_serialization_runtime
    performance_context = dagger_inputs["performance_context"]
    assert performance_context.execution_mode == "persistent_async"
    assert performance_context.teacher_checkpoint_sha256 == tuple(
        sorted(
            (
                mod.file_sha256(teacher_path),
                mod.file_sha256(walk_teacher_path),
            )
        )
    )
    assert performance_context.config_sha256 == mod.config_fingerprint(
        mod.OmegaConf.to_container(cfg, resolve=True)
    )
    assert performance_context.seed == int(cfg.algo.seed)
    assert performance_context.device == mod._distill_device(cfg)
    assert performance_context.num_envs == int(cfg.training.workflow.collect_num_envs)
    service.close.assert_called_once_with()
    assert result["execution_mode"] == "persistent_async"


def test_distill_single_entry_persistent_execution_uses_production_factory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mod = _train_distill()
    cfg = _distill_cfg(["training.workflow.enabled=true"])
    teacher_path = tmp_path / "teacher.pt"
    teacher_path.write_bytes(b"teacher")
    cfg.training.workflow.run_dir = str(tmp_path / "persistent_run")
    cfg.training.workflow.execution_mode = "persistent_async"
    cfg.training.workflow.roles = [
        {
            "role": "stand",
            "task": "g1_stand_still/mujoco",
            "teacher_checkpoint_path": str(teacher_path),
        }
    ]
    monkeypatch.setattr(
        distill_entry_plan,
        "_workflow_scenario_specs",
        lambda *_args: (mod.WorkflowScenarioSpec("stand", "role", ("stand",), 1.0),),
    )
    monkeypatch.setattr(distill_entry_workflow, "run_bootstrap_workflow", lambda **_kwargs: None)
    service = MagicMock()
    service.close_report = {
        "worker_pid": 1234,
        "resource_counters": {"env_builds": 1},
    }
    factory_inputs: dict[str, Any] = {}

    def fake_production_factory(**kwargs):
        factory_inputs.update(kwargs)
        return service

    monkeypatch.setattr(
        distill_entry_workflow,
        "build_persistent_g1_distillation_runtime",
        fake_production_factory,
    )
    monkeypatch.setattr(
        distill_entry_workflow,
        "run_multirole_dagger_workflow",
        lambda **kwargs: types.SimpleNamespace(
            run_dir=Path(kwargs["run_dir"]),
            manifest_path=Path(kwargs["run_dir"]) / "run_manifest.json",
            completed_iterations=1,
            checkpoint_path=Path(kwargs["run_dir"]) / "checkpoints" / "dagger_iteration_1.pt",
            cumulative_num_samples=1,
        ),
    )
    monkeypatch.setattr(
        distill_entry_workflow, "finalize_workflow_performance", lambda **_kwargs: None
    )

    result = mod.run_single_entry_workflow(cfg)

    assert factory_inputs["cfg"] is cfg
    assert [spec.role for spec in factory_inputs["role_specs"]] == ["stand"]
    assert result["execution_mode"] == "persistent_async"
    service.close.assert_called_once_with()


class _FakeDistillCollectEnv:
    def __init__(
        self,
        *,
        num_envs: int = 2,
        action_dim: int = 29,
        obs_dim: int = 99,
        critic_dim: int = 102,
        command_batches: list[np.ndarray] | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.critic_dim = critic_dim
        self.command_batches = command_batches
        self.action_space = types.SimpleNamespace(shape=(action_dim,))
        self.state = None
        self.reset_calls = 0
        self.step_calls = 0
        self.closed = False
        self.last_actions = None

    def init_state(self) -> None:
        self.state = object()

    def reset(self, env_indices):
        self.reset_calls += 1
        info = {"reset_indices": np.asarray(env_indices)}
        info.update(self._command_info(0))
        return self._obs(0), info

    def step(self, actions):
        self.step_calls += 1
        assert actions.shape == (self.num_envs, self.action_space.shape[0])
        self.last_actions = np.asarray(actions, dtype=np.float32)
        return types.SimpleNamespace(
            obs=self._obs(self.step_calls),
            info=self._command_info(self.step_calls),
        )

    def _obs(self, offset: int) -> dict[str, np.ndarray]:
        obs = np.arange(self.num_envs * self.obs_dim, dtype=np.float32).reshape(
            self.num_envs, self.obs_dim
        )
        critic = np.arange(self.num_envs * self.critic_dim, dtype=np.float32).reshape(
            self.num_envs, self.critic_dim
        )
        return {"obs": obs + float(offset), "critic": critic + 500.0}

    def _command_info(self, batch_index: int) -> dict[str, np.ndarray]:
        if self.command_batches is None:
            return {}
        index = min(int(batch_index), len(self.command_batches) - 1)
        return {"commands": self.command_batches[index]}

    def close(self) -> None:
        self.closed = True


class _IncrementingClock:
    def __init__(self, step: float = 0.1) -> None:
        self.value = 0.0
        self.step = float(step)

    def __call__(self) -> float:
        current = self.value
        self.value += self.step
        return current


def test_distill_collect_wrapper_emits_legacy_request_observations_only_when_opted_in(
    tmp_path: Path,
) -> None:
    mod = _train_distill()
    cfg = _distill_cfg(
        [
            "training.collect_num_samples=2",
            "training.collect_num_envs=2",
            "training.collect_max_env_steps=1",
        ]
    )
    fake_env = _FakeDistillCollectEnv(num_envs=2, obs_dim=99, critic_dim=102)

    result = mod.run_collect_dataset(
        cfg,
        dataset_path=tmp_path / "timed.pt",
        create_env_fn=lambda *args, **kwargs: fake_env,
        env_cfg_override_fn=lambda cfg: {"owner": "legacy-performance-test"},
        performance_clock=_IncrementingClock(),
    )

    observations = result["performance_stage_observations"]
    assert [item["stage"] for item in observations] == list(LEGACY_REQUEST_STAGE_NAMES)
    assert observations[0]["duration_seconds"] == pytest.approx(0.1)
    assert observations[-2]["row_count"] == 2
    assert observations[-1]["row_count"] == 2
    assert observations[-1]["cleanup_state"] == "pending"


def test_offpolicy_hydra_default_algo():
    cfg = _offpolicy_cfg()
    assert cfg.algo.algo == "sac"


def test_appo_runner_kwargs_forward_algorithm_seed():
    mod = _train_appo()
    cfg = _appo_cfg(["algo.seed=37"])
    rl_cfg = OmegaConf.to_container(cfg.algo, resolve=True)

    kwargs = mod.build_appo_runner_kwargs(
        cfg,
        env_cfg_override={"reward_config": {}},
        collector_device="cpu",
        rl_cfg=cast(dict[str, Any], rl_cfg),
    )

    assert kwargs["seed"] == 37


def test_appo_runner_kwargs_default_load_run_does_not_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _train_appo()
    cfg = _appo_cfg(["task=allegro_inhand/mujoco", "algo.load_run=-1"])

    def fail_resolve(*args, **kwargs):
        del args, kwargs
        raise AssertionError("training default load_run=-1 must not request resume")

    monkeypatch.setattr(mod, "resolve_appo_checkpoint_path", fail_resolve)

    kwargs = mod.build_appo_runner_kwargs(
        cfg,
        env_cfg_override={"reward_config": {}},
        collector_device="cpu",
    )

    assert "resume_path" not in kwargs


def test_appo_runner_kwargs_explicit_load_run_sets_resume_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mod = _train_appo()
    cfg = _appo_cfg(["task=allegro_inhand/mujoco", "algo.load_run=run1"])
    log_root = tmp_path / "logs" / "appo"
    run_dir = log_root / cfg.training.task_name / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "model_3.pt").write_bytes(b"")
    (run_dir / "model_9.pt").write_bytes(b"")
    monkeypatch.setattr(mod, "_get_log_root", lambda _cfg: str(log_root))

    kwargs = mod.build_appo_runner_kwargs(
        cfg,
        env_cfg_override={"reward_config": {}},
        collector_device="cpu",
    )

    assert kwargs["resume_path"] == str(run_dir / "model_9.pt")


def test_appo_runner_kwargs_missing_explicit_load_run_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mod = _train_appo()
    cfg = _appo_cfg(["task=allegro_inhand/mujoco", "algo.load_run=missing_run"])
    monkeypatch.setattr(mod, "_get_log_root", lambda _cfg: str(tmp_path / "logs" / "appo"))

    with pytest.raises(FileNotFoundError, match="missing_run"):
        mod.build_appo_runner_kwargs(
            cfg,
            env_cfg_override={"reward_config": {}},
            collector_device="cpu",
        )


def test_offpolicy_hydra_default_task():
    cfg = _offpolicy_cfg()
    assert cfg.training.task_name == "G1WalkFlat"


def test_offpolicy_hydra_default_logger():
    cfg = _offpolicy_cfg()
    assert cfg.training.logger == "tensorboard"


def test_offpolicy_hydra_default_wandb_fields():
    cfg = _offpolicy_cfg()
    assert cfg.training.wandb_project == "unilab"
    assert cfg.training.wandb_entity is None
    assert cfg.training.wandb_group is None
    assert cfg.training.wandb_job_type is None
    assert cfg.training.wandb_name is None
    assert cfg.training.wandb_tags == []
    assert cfg.training.wandb_notes is None
    assert cfg.training.wandb_mode is None


def test_offpolicy_hydra_default_sim_backend():
    cfg = _offpolicy_cfg()
    assert cfg.training.sim_backend == "mujoco"


def test_ppo_hydra_default_wandb_fields():
    cfg = _ppo_cfg()
    assert cfg.training.wandb_project == "unilab"
    assert cfg.training.wandb_entity is None
    assert cfg.training.wandb_group is None
    assert cfg.training.wandb_job_type is None
    assert cfg.training.wandb_name is None
    assert cfg.training.wandb_tags == []
    assert cfg.training.wandb_notes is None
    assert cfg.training.wandb_mode is None


def test_offpolicy_hydra_default_play_flags():
    cfg = _offpolicy_cfg()
    assert cfg.training.play_only is False
    assert cfg.training.no_play is False
    assert cfg.training.export_onnx is True
    assert cfg.algo.load_run == "-1"


def test_offpolicy_hydra_default_trace_flags():
    cfg = _offpolicy_cfg()
    assert cfg.training.trace_enabled is False
    assert cfg.training.trace_output_dir is None
    assert cfg.training.trace_thread_time is False
    assert cfg.training.trace_cuda_events is True
    assert cfg.training.verbose_metrics is False
    assert "replay_h2d_submitter" not in cfg.training


def test_offpolicy_hydra_algo_td3():
    cfg = _offpolicy_cfg(["algo=td3"])
    assert cfg.algo.algo == "td3"


def test_hora_distill_run_config_records_hardware(tmp_path, monkeypatch):
    mod = _train_hora_distill()
    hardware = {
        "platform": "test-platform",
        "chip": "test-cpu",
        "cpu_total_cores": "8",
        "gpu_name": "test-gpu",
        "memory": "32 GB",
    }
    monkeypatch.setattr(mod, "get_device_info_dict", lambda: hardware)
    cfg = OmegaConf.create({"training": {"task_name": "Task", "sim_backend": "mujoco"}})

    mod._write_distill_run_config(
        tmp_path,
        cfg=cfg,
        teacher_metadata={"checkpoint_path": "teacher.pt"},
    )

    payload = json.loads((tmp_path / "distill_run_config.json").read_text(encoding="utf-8"))
    assert payload["run"]["hardware"] == hardware


def test_hora_distill_task_owner_overrides_root_config_defaults():
    mod = _train_hora_distill()
    root_cfg = OmegaConf.load(_CONF_DIR / "hora_distill" / "config.yaml")
    cfg = mod._apply_teacher_defaults(_hora_distill_cfg(["task=sharpa_inhand/mujoco"]))

    assert root_cfg.algo.num_envs == 4096
    assert root_cfg.algo.save_interval_steps == 100000000
    assert cfg.algo.num_envs == 16384
    assert cfg.algo.save_interval_steps == 10000000


def test_hora_distill_sharpa_appo_student_owner_selects_nodr_demo_profile():
    mod = _train_hora_distill()
    cfg = mod._apply_teacher_defaults(_hora_distill_cfg(["task=sharpa_inhand/mujoco_nodr"]))

    assert cfg.teacher.algo_family == "appo"
    assert cfg.teacher.task == "sharpa_inhand/mujoco_hora"
    assert cfg.training.task_name == "SharpaInhandRotation"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.interactive.action_mode == "policy"
    assert cfg.interactive.policy_obs_mode == "actor"
    assert cfg.env.post_step_forward_sensor is True
    assert cfg.env.domain_rand.scale_list == [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    assert cfg.env.domain_rand.randomize_gravity is False
    assert cfg.env.domain_rand.randomize_gravity_direction is False
    assert cfg.env.domain_rand.randomize_pd_gains is False
    assert cfg.env.domain_rand.randomize_friction is False
    assert cfg.env.domain_rand.randomize_com is False
    assert cfg.env.domain_rand.randomize_mass is False
    assert cfg.env.domain_rand.force_scale == pytest.approx(0.0)
    assert cfg.env.domain_rand.random_force_prob_scalar == pytest.approx(0.0)
    assert cfg.env.domain_rand.joint_noise_scale == pytest.approx(0.0)
    assert cfg.env.domain_rand.contact_latency == pytest.approx(0.0)
    assert cfg.env.domain_rand.contact_sensor_noise == pytest.approx(0.0)
    assert cfg.algo.model.priv_info_embed_dim == 9
    assert cfg.algo.model.priv_mlp_hidden_dims == [256, 128, 9]


def test_distill_script_builds_teacher_and_student_from_owner_config():
    from unilab.algos.torch.distill import MLPStudentPolicy

    mod = _train_distill()
    cfg = _distill_cfg(
        [
            "teacher.actor_hidden_dim=16",
            "teacher.use_layer_norm=false",
            "teacher.obs_normalization=false",
            "student.hidden_dims=[32,16]",
        ]
    )

    teacher_spec = mod.build_teacher_spec(cfg)
    student = mod.build_student_policy(cfg, device="cpu")

    assert teacher_spec.obs_dim == 99
    assert teacher_spec.action_dim == 29
    assert teacher_spec.actor_hidden_dim == 16
    assert teacher_spec.use_layer_norm is False
    assert teacher_spec.obs_normalization is False
    assert cfg.student.model_type == "mlp"
    assert isinstance(student, MLPStudentPolicy)
    assert student.obs_dim == 99
    assert student.action_dim == 29
    assert [layer.out_features for layer in student.net if hasattr(layer, "out_features")] == [
        32,
        16,
        29,
    ]


def test_distill_script_builds_stand_still_teacher_and_student_from_owner_config():
    mod = _train_distill()
    cfg = _distill_cfg(["task=g1_stand_still/mujoco"])

    teacher_spec = mod.build_teacher_spec(cfg)
    student = mod.build_student_policy(cfg, device="cpu")

    assert cfg.training.task_name == "G1StandStill"
    assert cfg.teacher.task_name == "G1StandStill"
    assert cfg.teacher.load_run == "2026-07-09_22-55-05_mujoco"
    assert cfg.teacher.checkpoint == 5000
    assert teacher_spec.obs_dim == 98
    assert teacher_spec.action_dim == 29
    assert student.obs_dim == 98
    assert student.action_dim == 29


def test_distill_script_builds_walk_flat_teacher_and_student_from_owner_config():
    mod = _train_distill()
    cfg = _distill_cfg(["task=g1_walk_flat/mujoco"])

    teacher_spec = mod.build_teacher_spec(cfg)
    student = mod.build_student_policy(cfg, device="cpu")

    assert cfg.training.task_name == "G1WalkFlat"
    assert cfg.teacher.task_name == "G1WalkFlat"
    assert cfg.teacher.load_run == "2026-07-09_02-48-58_mujoco"
    assert cfg.teacher.checkpoint == 5000
    assert teacher_spec.obs_dim == 98
    assert teacher_spec.action_dim == 29
    assert student.obs_dim == 98
    assert student.action_dim == 29


def test_distill_script_builds_moe_student_from_owner_config():
    from unilab.algos.torch.distill import MoEStudentPolicy

    mod = _train_distill()
    cfg = _distill_cfg(
        [
            "student.model_type=moe",
            "student.num_experts=3",
            "student.expert_hidden_dims=[32]",
            "student.router_hidden_dims=[16]",
            "student.routing_mode=soft",
            "student.router_temperature=0.75",
            "algo.aux_loss_coef=0.25",
        ]
    )

    student = mod.build_student_policy(cfg, device="cpu")

    assert isinstance(student, MoEStudentPolicy)
    assert student.obs_dim == 99
    assert student.action_dim == 29
    assert student.num_experts == 3
    assert student.routing_mode == "soft"
    assert student.router_temperature == pytest.approx(0.75)
    assert len(student.experts) == 3
    assert [layer.out_features for layer in student.router if hasattr(layer, "out_features")] == [
        16,
        3,
    ]
    assert [
        layer.out_features for layer in student.experts[0].net if hasattr(layer, "out_features")
    ] == [32, 29]


def test_distill_script_wires_iterative_dagger_owner_loop(tmp_path: Path, monkeypatch):
    from types import SimpleNamespace

    mod = _train_distill()
    init_checkpoint = tmp_path / "init.pt"
    output_checkpoint = tmp_path / "dagger.pt"
    init_checkpoint.touch()
    cfg = _distill_cfg(
        [
            "task=g1_walk_flat/mujoco",
            "student.model_type=moe",
            "training.online_dagger=true",
            "training.dagger_iterations=3",
            "training.dagger_samples_per_iteration=64",
            "training.dagger_batch_size=16",
            "training.dagger_updates_per_iteration=4",
            "training.dagger_role_label=walk_flat",
            "training.collect_command_sample_filter=active",
            f"training.offline_init_checkpoint={init_checkpoint}",
            f"training.dagger_checkpoint={output_checkpoint}",
        ]
    )
    trainer = SimpleNamespace(student=object(), teacher=object())
    captured = {}

    monkeypatch.setattr(
        distill_entry_collection, "build_distillation_trainer", lambda *args, **kwargs: trainer
    )
    monkeypatch.setattr(distill_entry_collection, "_teacher_metadata", lambda *args, **kwargs: {})

    def fake_dagger(env, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            iteration_count=3,
            update_count=12,
            samples_collected=192,
            samples_seen=192,
            checkpoint_path=output_checkpoint,
            iteration_results=(SimpleNamespace(last_loss=0.1),),
            collection_metadata=({},),
        )

    monkeypatch.setattr(distill_entry_collection, "run_iterative_dagger_updates", fake_dagger)
    env = SimpleNamespace(close=lambda: None)

    probe = mod.run_online_dagger_update(
        cfg,
        teacher_checkpoint=tmp_path / "teacher.pt",
        create_env_fn=lambda *args, **kwargs: env,
        env_cfg_override_fn=lambda cfg: {"owner": "distill-test"},
    )

    assert captured["num_iterations"] == 3
    assert captured["samples_per_iteration"] == 64
    assert captured["updates_per_iteration"] == 4
    assert captured["role_label"] == "walk_flat"
    assert captured["command_sample_filter"] == "active"
    assert captured["checkpoint_path"] == output_checkpoint
    assert probe["distill_source"] == "iterative_dagger"
    assert probe["checkpoint_path"] == str(output_checkpoint)


def test_distill_script_builds_role_conditioned_moe_trainer(tmp_path: Path):
    import torch

    from unilab.algos.torch.fast_sac.learner import SACActor

    mod = _train_distill()
    cfg = _distill_cfg(
        [
            "teacher.actor_hidden_dim=16",
            "teacher.use_layer_norm=false",
            "teacher.obs_normalization=false",
            "student.model_type=moe",
            "student.num_experts=3",
            "student.expert_hidden_dims=[32]",
            "student.router_hidden_dims=[16]",
            "algo.role_loss_coef=0.2",
            "+algo.role_expert_targets={stand:0,walk_height:1,height:2}",
            "algo.command_intent_loss_coef=0.3",
            "algo.command_intent_expert_targets={inactive:0,active:1}",
        ]
    )
    teacher = SACActor(99, 29, hidden_dim=16, use_layer_norm=False, device="cpu")
    teacher_checkpoint = tmp_path / "teacher.pt"
    torch.save({"actor": teacher.state_dict()}, teacher_checkpoint)

    trainer = mod.build_distillation_trainer(
        cfg,
        teacher_checkpoint=teacher_checkpoint,
        device="cpu",
    )
    runtime_cfg = mod._distill_runtime_cfg(cfg, distill_source="unit")

    assert trainer.role_loss_coef == pytest.approx(0.2)
    assert trainer.role_expert_targets == {"stand": 0, "walk_height": 1, "height": 2}
    assert trainer.command_intent_loss_coef == pytest.approx(0.3)
    assert trainer.command_intent_expert_targets == {"inactive": 0, "active": 1}
    assert runtime_cfg["role_loss_coef"] == pytest.approx(0.2)
    assert runtime_cfg["role_expert_targets"] == {
        "stand": 0,
        "walk_height": 1,
        "height": 2,
    }
    assert runtime_cfg["command_intent_loss_coef"] == pytest.approx(0.3)
    assert runtime_cfg["command_intent_expert_targets"] == {"inactive": 0, "active": 1}


def test_distill_script_resolves_teacher_checkpoint_with_training_semantics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    mod = _train_distill()
    cfg = _distill_cfg(
        [
            "teacher.load_run=teacher_run",
            "teacher.checkpoint=42",
            "training.log_root=teacher_logs",
        ]
    )
    captured: dict[str, Any] = {}

    def fake_resolve_task_checkpoint_path(root_dir, **kwargs):
        captured["root_dir"] = root_dir
        captured.update(kwargs)
        return tmp_path / "model_42.pt", tmp_path / "teacher_run"

    monkeypatch.setattr(
        distill_entry_training,
        "resolve_task_checkpoint_path",
        fake_resolve_task_checkpoint_path,
    )

    checkpoint_path, run_dir = mod.resolve_teacher_checkpoint(cfg, root_dir=tmp_path)

    assert checkpoint_path == tmp_path / "model_42.pt"
    assert run_dir == tmp_path / "teacher_run"
    assert captured == {
        "root_dir": tmp_path,
        "task_name": "G1WalkHeight",
        "load_run": "teacher_run",
        "algo_log_name": "fast_sac",
        "checkpoint": "42",
        "suffix": ".pt",
        "log_root": "teacher_logs",
    }


def test_distill_script_resolves_explicit_teacher_checkpoint_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    mod = _train_distill()
    checkpoint_path = tmp_path / "explicit_teacher.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    cfg = _distill_cfg(
        [
            f"teacher.checkpoint_path={checkpoint_path}",
            "teacher.load_run=should_not_be_used",
            "teacher.checkpoint=42",
        ]
    )

    def fail_resolve_task_checkpoint_path(*args, **kwargs):
        raise AssertionError("teacher.checkpoint_path must bypass run/checkpoint resolution")

    monkeypatch.setattr(mod, "resolve_task_checkpoint_path", fail_resolve_task_checkpoint_path)

    resolved_checkpoint_path, run_dir = mod.resolve_teacher_checkpoint(cfg, root_dir=tmp_path)

    assert resolved_checkpoint_path == checkpoint_path
    assert run_dir == tmp_path


def test_distill_script_fake_batch_probe_loads_teacher_and_updates_student(tmp_path: Path):
    import torch

    from unilab.algos.torch.distill import (
        MLPStudentPolicy,
        load_distillation_checkpoint,
        load_distillation_student_policy,
    )
    from unilab.algos.torch.fast_sac.learner import SACActor

    mod = _train_distill()
    cfg = _distill_cfg(
        [
            "teacher.actor_hidden_dim=16",
            "teacher.use_layer_norm=false",
            "teacher.obs_normalization=false",
            "student.hidden_dims=[32]",
            "algo.learning_rate=0.01",
            "algo.max_grad_norm=10.0",
        ]
    )
    teacher = SACActor(99, 29, hidden_dim=16, use_layer_norm=False, device="cpu")
    checkpoint_path = tmp_path / "teacher.pt"
    torch.save({"actor": teacher.state_dict()}, checkpoint_path)
    student_checkpoint_path = tmp_path / "student.pt"

    probe = mod.run_fake_batch_update(
        cfg,
        teacher_checkpoint=checkpoint_path,
        batch_size=2,
        max_updates=2,
        checkpoint_path=student_checkpoint_path,
        device="cpu",
    )

    assert probe["teacher_obs_shape"] == (2, 99)
    assert probe["student_obs_shape"] == (2, 99)
    assert probe["dataset_num_samples"] == 4
    assert probe["dataset_student_obs_dim"] == 99
    assert probe["dataset_teacher_obs_dim"] == 99
    assert probe["teacher_action_shape"] == (2, 29)
    assert probe["student_action_shape"] == (2, 29)
    assert probe["teacher_action_requires_grad"] is False
    assert probe["update_count"] == 2
    assert probe["samples_seen"] == 4
    assert probe["checkpoint_path"] == str(student_checkpoint_path)
    assert probe["student_grad_norm"] > 0.0
    assert probe["loss"] >= 0.0
    assert student_checkpoint_path.exists()

    restored = MLPStudentPolicy(obs_dim=99, action_dim=29, hidden_dims=(32,))
    checkpoint = load_distillation_checkpoint(restored, student_checkpoint_path)
    assert checkpoint["agent_steps"] == 4
    assert checkpoint["teacher_metadata"]["task_name"] == "G1WalkHeight"
    assert checkpoint["teacher_metadata"]["checkpoint_actor_input_dim"] == 99
    assert checkpoint["teacher_metadata"]["checkpoint_first_weight_key"] == "net.0.weight"
    assert checkpoint["distill_runtime_cfg"]["student_obs_dim"] == 99
    assert checkpoint["distill_runtime_cfg"]["student_action_dim"] == 29
    assert checkpoint["distill_runtime_cfg"]["student_hidden_dims"] == [32]
    assert checkpoint["distill_runtime_cfg"]["student_activation"] == "elu"

    loaded_student = load_distillation_student_policy(student_checkpoint_path, device="cpu")
    assert loaded_student.policy(torch.randn(1, 99)).shape == (1, 29)


def test_distill_script_rejects_legacy_teacher_checkpoint_without_override(tmp_path: Path):
    import torch

    from unilab.algos.torch.fast_sac.learner import SACActor

    mod = _train_distill()
    cfg = _distill_cfg(
        [
            "teacher.actor_hidden_dim=16",
            "teacher.use_layer_norm=false",
            "teacher.obs_normalization=false",
        ]
    )
    legacy_teacher = SACActor(100, 29, hidden_dim=16, use_layer_norm=False, device="cpu")
    checkpoint_path = tmp_path / "legacy_teacher.pt"
    torch.save({"actor": legacy_teacher.state_dict()}, checkpoint_path)

    with pytest.raises(ValueError, match="teacher.obs_dim=100"):
        mod.run_fake_batch_update(
            cfg,
            teacher_checkpoint=checkpoint_path,
            batch_size=1,
            max_updates=1,
            device="cpu",
        )


def test_distill_script_fake_batch_probe_records_moe_aux_diagnostics(tmp_path: Path):
    import torch

    from unilab.algos.torch.distill import (
        MoEStudentPolicy,
        load_distillation_checkpoint,
    )
    from unilab.algos.torch.fast_sac.learner import SACActor

    mod = _train_distill()
    cfg = _distill_cfg(
        [
            "teacher.actor_hidden_dim=16",
            "teacher.use_layer_norm=false",
            "teacher.obs_normalization=false",
            "student.model_type=moe",
            "student.num_experts=3",
            "student.expert_hidden_dims=[32]",
            "student.router_hidden_dims=[16]",
            "algo.learning_rate=0.01",
            "algo.max_grad_norm=10.0",
            "algo.aux_loss_coef=0.25",
        ]
    )
    teacher = SACActor(99, 29, hidden_dim=16, use_layer_norm=False, device="cpu")
    checkpoint_path = tmp_path / "teacher.pt"
    torch.save({"actor": teacher.state_dict()}, checkpoint_path)
    student_checkpoint_path = tmp_path / "moe_student.pt"

    probe = mod.run_fake_batch_update(
        cfg,
        teacher_checkpoint=checkpoint_path,
        batch_size=2,
        max_updates=2,
        checkpoint_path=student_checkpoint_path,
        device="cpu",
    )

    assert probe["student_model_type"] == "moe"
    assert probe["teacher_obs_shape"] == (2, 99)
    assert probe["student_obs_shape"] == (2, 99)
    assert probe["student_action_shape"] == (2, 29)
    assert probe["update_count"] == 2
    assert probe["samples_seen"] == 4
    assert probe["behavior_loss"] > 0.0
    assert probe["aux_loss"] >= 0.0
    assert probe["loss"] == pytest.approx(probe["behavior_loss"] + 0.25 * probe["aux_loss"])
    assert probe["expert_usage"] is not None
    assert len(probe["expert_usage"]) == 3
    assert sum(probe["expert_usage"]) == pytest.approx(2.0)
    assert probe["route_entropy"] is not None
    assert probe["route_entropy"] >= 0.0

    restored = MoEStudentPolicy(
        obs_dim=99,
        action_dim=29,
        num_experts=3,
        expert_hidden_dims=(32,),
        router_hidden_dims=(16,),
    )
    checkpoint = load_distillation_checkpoint(restored, student_checkpoint_path)
    runtime_cfg = checkpoint["distill_runtime_cfg"]
    assert runtime_cfg["student_model_type"] == "moe"
    assert runtime_cfg["student_num_experts"] == 3
    assert runtime_cfg["student_expert_hidden_dims"] == [32]
    assert runtime_cfg["student_router_hidden_dims"] == [16]
    assert runtime_cfg["aux_loss_coef"] == pytest.approx(0.25)


def test_distill_script_dataset_update_loads_saved_dataset_and_saves_moe_student(
    tmp_path: Path,
):
    import torch

    from unilab.algos.torch.distill import (
        MoEStudentPolicy,
        build_distillation_dataset,
        load_distillation_checkpoint,
        load_distillation_student_policy,
        save_distillation_dataset,
    )
    from unilab.algos.torch.fast_sac.learner import SACActor

    mod = _train_distill()
    cfg = _distill_cfg(
        [
            "teacher.actor_hidden_dim=16",
            "teacher.use_layer_norm=false",
            "teacher.obs_normalization=false",
            "student.model_type=moe",
            "student.num_experts=3",
            "student.expert_hidden_dims=[32]",
            "student.router_hidden_dims=[16]",
            "algo.learning_rate=0.01",
            "algo.max_grad_norm=10.0",
            "algo.aux_loss_coef=0.25",
        ]
    )
    teacher = SACActor(99, 29, hidden_dim=16, use_layer_norm=False, device="cpu")
    teacher_checkpoint = tmp_path / "teacher.pt"
    torch.save({"actor": teacher.state_dict()}, teacher_checkpoint)
    dataset_path = tmp_path / "dataset.pt"
    dataset = build_distillation_dataset(
        torch.randn(4, 99),
        torch.randn(4, 99),
        expected_student_obs_dim=99,
        expected_teacher_obs_dim=99,
        metadata={"source": "saved_fixture", "role": "offline_dataset"},
    )
    save_distillation_dataset(dataset_path, dataset)
    student_checkpoint = tmp_path / "offline_moe_student.pt"

    probe = mod.run_offline_dataset_update(
        cfg,
        teacher_checkpoint=teacher_checkpoint,
        dataset_path=dataset_path,
        batch_size=2,
        max_updates=2,
        checkpoint_path=student_checkpoint,
        device="cpu",
    )

    assert probe["distill_source"] == "offline_dataset"
    assert probe["dataset_path"] == str(dataset_path)
    assert probe["student_model_type"] == "moe"
    assert probe["dataset_num_samples"] == 4
    assert probe["dataset_student_obs_dim"] == 99
    assert probe["dataset_teacher_obs_dim"] == 99
    assert probe["dataset_metadata"] == {
        "source": "saved_fixture",
        "role": "offline_dataset",
    }
    assert probe["teacher_obs_shape"] == (2, 99)
    assert probe["student_obs_shape"] == (2, 99)
    assert probe["teacher_action_shape"] == (2, 29)
    assert probe["student_action_shape"] == (2, 29)
    assert probe["update_count"] == 2
    assert probe["samples_seen"] == 4
    assert probe["behavior_loss"] > 0.0
    assert probe["aux_loss"] >= 0.0
    assert probe["loss"] == pytest.approx(probe["behavior_loss"] + 0.25 * probe["aux_loss"])
    assert probe["checkpoint_path"] == str(student_checkpoint)

    restored = MoEStudentPolicy(
        obs_dim=99,
        action_dim=29,
        num_experts=3,
        expert_hidden_dims=(32,),
        router_hidden_dims=(16,),
    )
    checkpoint = load_distillation_checkpoint(restored, student_checkpoint)
    runtime_cfg = checkpoint["distill_runtime_cfg"]
    assert runtime_cfg["distill_source"] == "offline_dataset"
    assert runtime_cfg["dataset_path"] == str(dataset_path)
    assert runtime_cfg["student_model_type"] == "moe"
    assert runtime_cfg["student_num_experts"] == 3

    loaded_student = load_distillation_student_policy(student_checkpoint, device="cpu")
    assert loaded_student.policy(torch.randn(1, 99)).shape == (1, 29)


def test_distill_script_offline_update_uses_balanced_role_sampler(tmp_path: Path):
    import torch

    from unilab.algos.torch.distill import (
        MLPStudentPolicy,
        build_distillation_dataset,
        load_distillation_checkpoint,
        save_distillation_dataset,
    )
    from unilab.algos.torch.fast_sac.learner import SACActor

    mod = _train_distill()
    cfg = _distill_cfg(
        [
            "teacher.actor_hidden_dim=16",
            "teacher.use_layer_norm=false",
            "teacher.obs_normalization=false",
            "student.hidden_dims=[32]",
            "algo.learning_rate=0.01",
            "training.offline_balance_key=role",
            "training.offline_balanced_labels=[stand,walk]",
        ]
    )
    teacher = SACActor(99, 29, hidden_dim=16, use_layer_norm=False, device="cpu")
    teacher_checkpoint = tmp_path / "teacher.pt"
    torch.save({"actor": teacher.state_dict()}, teacher_checkpoint)
    dataset_path = tmp_path / "imbalanced_dataset.pt"
    dataset = build_distillation_dataset(
        torch.randn(6, 99),
        torch.randn(6, 99),
        expected_student_obs_dim=99,
        expected_teacher_obs_dim=99,
        expected_teacher_action_dim=29,
        teacher_actions=torch.randn(6, 29),
        role_labels=("stand", "walk", "walk", "walk", "walk", "walk"),
    )
    save_distillation_dataset(dataset_path, dataset)
    student_checkpoint = tmp_path / "balanced_student.pt"

    probe = mod.run_offline_dataset_update(
        cfg,
        teacher_checkpoint=teacher_checkpoint,
        dataset_path=dataset_path,
        batch_size=4,
        max_updates=2,
        checkpoint_path=student_checkpoint,
        device="cpu",
    )

    assert probe["offline_balance_key"] == "role"
    assert probe["offline_batch_label_counts"] == (
        {"stand": 2, "walk": 2},
        {"stand": 2, "walk": 2},
    )
    assert probe["offline_last_balance_label_counts"] == {"stand": 2, "walk": 2}
    assert probe["samples_seen"] == 8
    restored = MLPStudentPolicy(obs_dim=99, action_dim=29, hidden_dims=(32,))
    checkpoint = load_distillation_checkpoint(restored, student_checkpoint)
    runtime_cfg = checkpoint["distill_runtime_cfg"]
    assert runtime_cfg["offline_balance_key"] == "role"
    assert runtime_cfg["offline_balanced_labels"] == ["stand", "walk"]


def test_distill_script_offline_update_initializes_from_student_checkpoint(tmp_path: Path):
    import torch

    from unilab.algos.torch.distill import (
        MLPStudentPolicy,
        build_distillation_dataset,
        load_distillation_checkpoint,
        save_distillation_checkpoint,
        save_distillation_dataset,
    )
    from unilab.algos.torch.fast_sac.learner import SACActor

    mod = _train_distill()
    cfg = _distill_cfg(
        [
            "teacher.actor_hidden_dim=16",
            "teacher.use_layer_norm=false",
            "teacher.obs_normalization=false",
            "student.hidden_dims=[32]",
            "algo.learning_rate=0.0",
        ]
    )
    teacher = SACActor(99, 29, hidden_dim=16, use_layer_norm=False, device="cpu")
    teacher_checkpoint = tmp_path / "teacher.pt"
    torch.save({"actor": teacher.state_dict()}, teacher_checkpoint)
    dataset_path = tmp_path / "dataset.pt"
    save_distillation_dataset(
        dataset_path,
        build_distillation_dataset(
            torch.randn(4, 99),
            torch.randn(4, 99),
            expected_student_obs_dim=99,
            expected_teacher_obs_dim=99,
            expected_teacher_action_dim=29,
            teacher_actions=torch.randn(4, 29),
        ),
    )
    init_student = MLPStudentPolicy(obs_dim=99, action_dim=29, hidden_dims=(32,))
    for param in init_student.parameters():
        torch.nn.init.constant_(param, 0.123)
    init_optimizer = torch.optim.Adam(init_student.parameters(), lr=0.0)
    init_optimizer.zero_grad(set_to_none=True)
    init_student(torch.randn(2, 99)).sum().backward()
    init_optimizer.step()
    init_checkpoint = tmp_path / "init_student.pt"
    save_distillation_checkpoint(
        init_checkpoint,
        student=init_student,
        optimizer=init_optimizer,
        agent_steps=128,
        distill_runtime_cfg={
            "student_model_type": "mlp",
            "student_obs_dim": 99,
            "student_action_dim": 29,
            "student_activation": "elu",
            "student_squash_action": True,
            "student_hidden_dims": [32],
        },
    )
    output_checkpoint = tmp_path / "continued_student.pt"
    cfg.training.offline_init_checkpoint = str(init_checkpoint)

    probe = mod.run_offline_dataset_update(
        cfg,
        teacher_checkpoint=teacher_checkpoint,
        dataset_path=dataset_path,
        batch_size=2,
        max_updates=1,
        checkpoint_path=output_checkpoint,
        device="cpu",
    )

    assert probe["student_init_checkpoint_path"] == str(init_checkpoint)
    assert probe["student_init_agent_steps"] == 128
    assert probe["student_init_optimizer_requested"] is True
    assert probe["student_init_optimizer_loaded"] is True
    restored = MLPStudentPolicy(obs_dim=99, action_dim=29, hidden_dims=(32,))
    checkpoint = load_distillation_checkpoint(restored, output_checkpoint)
    runtime_cfg = checkpoint["distill_runtime_cfg"]
    assert runtime_cfg["student_init_checkpoint_path"] == str(init_checkpoint)
    assert runtime_cfg["student_init_agent_steps"] == 128
    assert runtime_cfg["student_init_optimizer_requested"] is True
    assert runtime_cfg["student_init_optimizer_loaded"] is True
    for init_param, restored_param in zip(init_student.parameters(), restored.parameters()):
        assert torch.allclose(init_param, restored_param)


def test_distill_script_offline_init_checkpoint_rejects_student_contract_mismatch(
    tmp_path: Path,
):
    import torch

    from unilab.algos.torch.distill import (
        MLPStudentPolicy,
        build_distillation_dataset,
        save_distillation_checkpoint,
        save_distillation_dataset,
    )
    from unilab.algos.torch.fast_sac.learner import SACActor

    mod = _train_distill()
    cfg = _distill_cfg(
        [
            "teacher.actor_hidden_dim=16",
            "teacher.use_layer_norm=false",
            "teacher.obs_normalization=false",
            "student.hidden_dims=[32]",
        ]
    )
    teacher = SACActor(99, 29, hidden_dim=16, use_layer_norm=False, device="cpu")
    teacher_checkpoint = tmp_path / "teacher.pt"
    torch.save({"actor": teacher.state_dict()}, teacher_checkpoint)
    dataset_path = tmp_path / "dataset.pt"
    save_distillation_dataset(
        dataset_path,
        build_distillation_dataset(
            torch.randn(2, 99),
            torch.randn(2, 99),
            expected_student_obs_dim=99,
            expected_teacher_obs_dim=99,
            expected_teacher_action_dim=29,
            teacher_actions=torch.randn(2, 29),
        ),
    )
    init_checkpoint = tmp_path / "bad_init_student.pt"
    save_distillation_checkpoint(
        init_checkpoint,
        student=MLPStudentPolicy(obs_dim=99, action_dim=29, hidden_dims=(64,)),
        agent_steps=1,
        distill_runtime_cfg={
            "student_model_type": "mlp",
            "student_obs_dim": 99,
            "student_action_dim": 29,
            "student_activation": "elu",
            "student_squash_action": True,
            "student_hidden_dims": [64],
        },
    )
    cfg.training.offline_init_checkpoint = str(init_checkpoint)

    with pytest.raises(ValueError, match="offline_init_checkpoint student runtime config mismatch"):
        mod.run_offline_dataset_update(
            cfg,
            teacher_checkpoint=teacher_checkpoint,
            dataset_path=dataset_path,
            batch_size=2,
            max_updates=1,
            checkpoint_path=tmp_path / "unused.pt",
            device="cpu",
        )


def test_distill_script_builds_multitask_dataset_from_saved_sources(tmp_path: Path):
    import torch

    from unilab.algos.torch.distill import (
        build_distillation_dataset,
        load_distillation_dataset,
        save_distillation_dataset,
    )

    mod = _train_distill()
    stand_path = tmp_path / "stand.pt"
    walk_path = tmp_path / "walk.pt"
    merged_path = tmp_path / "merged.pt"
    save_distillation_dataset(
        stand_path,
        build_distillation_dataset(
            torch.full((2, 99), 1.0),
            torch.full((2, 99), 2.0),
            expected_student_obs_dim=99,
            expected_teacher_obs_dim=99,
            expected_teacher_action_dim=29,
            teacher_actions=torch.full((2, 29), 0.1),
        ),
    )
    save_distillation_dataset(
        walk_path,
        build_distillation_dataset(
            torch.full((3, 99), 3.0),
            torch.full((3, 99), 4.0),
            expected_student_obs_dim=99,
            expected_teacher_obs_dim=99,
            expected_teacher_action_dim=29,
            teacher_actions=torch.full((3, 29), -0.2),
        ),
    )
    cfg = _distill_cfg(
        [
            "training.device=cuda:0",
            f"training.multitask_dataset_path={merged_path}",
            f"+training.multitask_sources=[{{path:{stand_path},role:stand}},{{path:{walk_path},role:walk_height}}]",
        ]
    )

    probe = mod.run_multitask_dataset_assembly(cfg, dataset_path=merged_path)

    assert probe["distill_source"] == "multitask_adapter"
    assert probe["dataset_path"] == str(merged_path)
    assert probe["aggregate_assembly_device"] == "cpu"
    assert probe["dataset_num_samples"] == 5
    assert probe["dataset_student_obs_dim"] == 99
    assert probe["dataset_teacher_obs_dim"] == 99
    assert probe["dataset_teacher_action_dim"] == 29
    assert probe["source_roles"] == ["stand", "walk_height"]
    assert probe["source_sample_counts"] == [2, 3]

    restored = load_distillation_dataset(
        merged_path,
        expected_student_obs_dim=99,
        expected_teacher_obs_dim=99,
        expected_teacher_action_dim=29,
    )
    assert restored.role_labels == (
        "stand",
        "stand",
        "walk_height",
        "walk_height",
        "walk_height",
    )
    assert restored.teacher_actions is not None
    assert torch.allclose(restored.teacher_actions[:2], torch.full((2, 29), 0.1))
    assert torch.allclose(restored.teacher_actions[2:], torch.full((3, 29), -0.2))


def test_distill_script_multitask_dataset_infers_source_dims(tmp_path: Path):
    import pytest
    import torch

    from unilab.algos.torch.distill import (
        build_distillation_dataset,
        load_distillation_dataset,
        save_distillation_dataset,
    )

    mod = _train_distill()
    stand_path = tmp_path / "stand_98.pt"
    walk_path = tmp_path / "walk_98.pt"
    merged_path = tmp_path / "merged_98.pt"
    save_distillation_dataset(
        stand_path,
        build_distillation_dataset(
            torch.full((2, 98), 1.0),
            torch.full((2, 98), 2.0),
            expected_student_obs_dim=98,
            expected_teacher_obs_dim=98,
            expected_teacher_action_dim=29,
            teacher_actions=torch.full((2, 29), 0.1),
        ),
    )
    save_distillation_dataset(
        walk_path,
        build_distillation_dataset(
            torch.full((3, 98), 3.0),
            torch.full((3, 98), 4.0),
            expected_student_obs_dim=98,
            expected_teacher_obs_dim=98,
            expected_teacher_action_dim=29,
            teacher_actions=torch.full((3, 29), -0.2),
        ),
    )
    cfg = _distill_cfg(
        [
            f"training.multitask_dataset_path={merged_path}",
            f"+training.multitask_sources=[{{path:{stand_path},role:stand}},{{path:{walk_path},role:walk_flat}}]",
        ]
    )

    probe = mod.run_multitask_dataset_assembly(cfg, dataset_path=merged_path)

    assert probe["dataset_student_obs_dim"] == 98
    assert probe["dataset_teacher_obs_dim"] == 98
    assert probe["dataset_teacher_action_dim"] == 29
    restored = load_distillation_dataset(
        merged_path,
        expected_student_obs_dim=98,
        expected_teacher_obs_dim=98,
        expected_teacher_action_dim=29,
    )
    assert restored.role_labels == ("stand", "stand", "walk_flat", "walk_flat", "walk_flat")

    strict_cfg = _distill_cfg(
        [
            f"training.multitask_dataset_path={tmp_path / 'strict_merged.pt'}",
            f"+training.multitask_sources=[{{path:{stand_path},role:stand}},{{path:{walk_path},role:walk_flat}}]",
            "training.multitask_expected_student_obs_dim=99",
            "training.multitask_expected_teacher_obs_dim=99",
            "training.multitask_expected_teacher_action_dim=29",
        ]
    )
    with pytest.raises(ValueError, match="student_obs dim mismatch"):
        mod.run_multitask_dataset_assembly(strict_cfg)


def test_g1_distill_multitask_runtime_probe_runs_cached_moe_update(tmp_path: Path, capsys):
    mod = _load_deploy_script("check_unilab_g1_distill_multitask_runtime_probe")

    payload = mod.run_check(work_dir=tmp_path, device="cpu")
    mod.print_report(payload)
    out = capsys.readouterr().out

    assert payload["status"] == "ok"
    assert payload["probe"] == "g1_distill_multitask_runtime"
    assert payload["merged_num_samples"] == 6
    assert payload["role_counts"] == {"height": 1, "stand": 2, "walk_height": 3}
    assert payload["teacher_action_shape"] == [6, 29]
    assert payload["offline_update"]["update_count"] == 2
    assert payload["offline_update"]["samples_seen"] == 6
    assert payload["offline_update"]["teacher_action_source"] == "cached"
    assert payload["offline_update"]["role_loss"] > 0.0
    assert payload["offline_update"]["role_target_count"] == 3
    assert payload["offline_update"]["student_grad_norm"] > 0.0
    assert "[PASS] g1_distill_multitask_runtime" in out


def test_g1_distill_dual_teacher_probe_requires_owner_intent_filters(
    tmp_path: Path,
    monkeypatch,
):
    import torch

    from unilab.algos.torch.distill import build_distillation_dataset, save_distillation_dataset

    mod = _load_deploy_script("check_unilab_g1_distill_dual_teacher_moe_probe")
    captured_filters: dict[str, str] = {}
    captured_checkpoint_paths: dict[str, str] = {}

    def fake_checkpoint_info(path):
        return {
            "checkpoint_path": str(path),
            "actor_input_dim": 98,
            "first_weight_key": "net.0.weight",
        }

    def fake_run_collect_dataset(cfg, *, dataset_path):
        task_name = str(cfg.training.task_name)
        role = "walk_flat" if task_name == "G1WalkFlat" else "stand"
        command_filter = str(cfg.training.collect_command_sample_filter)
        command_intent = "active" if command_filter == "active" else "inactive"
        command_value = 0.1 if command_intent == "active" else 0.0
        captured_filters[role] = command_filter
        captured_checkpoint_paths[role] = str(cfg.teacher.checkpoint_path)
        save_distillation_dataset(
            dataset_path,
            build_distillation_dataset(
                torch.full((2, 98), 1.0 if role == "walk_flat" else 2.0),
                torch.full((2, 98), 3.0 if role == "walk_flat" else 4.0),
                expected_student_obs_dim=98,
                expected_teacher_obs_dim=98,
                expected_teacher_action_dim=29,
                teacher_actions=torch.full((2, 29), 0.1 if role == "walk_flat" else -0.1),
                commands=torch.full((2, 3), command_value),
                command_intents=(command_intent, command_intent),
                metadata={
                    "source": "fake-filtered-collection",
                    "command_sample_filter": command_filter,
                    "command_seen_samples": 3,
                    "command_selected_samples": 2,
                    "action_abs_max": 0.1,
                    "env_steps": 1,
                },
            ),
        )
        return {
            "collect_command_sample_filter": command_filter,
            "collect_command_seen_samples": 3,
            "collect_command_selected_samples": 2,
        }

    monkeypatch.setattr(mod, "_checkpoint_info", fake_checkpoint_info)
    monkeypatch.setattr(mod.train_distill, "run_collect_dataset", fake_run_collect_dataset)

    payload = mod.run_check(
        walking_checkpoint=tmp_path / "walk.pt",
        standing_checkpoint=tmp_path / "stand.pt",
        work_dir=tmp_path,
        num_samples=2,
        num_envs=1,
        batch_size=2,
        max_updates=1,
        device="cpu",
    )

    assert captured_filters == {"stand": "inactive", "walk_flat": "active"}
    assert captured_checkpoint_paths == {
        "stand": str(tmp_path / "stand.pt"),
        "walk_flat": str(tmp_path / "walk.pt"),
    }
    assert payload["command_filter_contracts"]["walk_flat"]["expected_filter"] == "active"
    assert payload["command_filter_contracts"]["stand"]["expected_filter"] == "inactive"
    assert payload["command_filter_contracts"]["walk_flat"]["command_selected_samples"] == 2
    assert payload["command_filter_contracts"]["stand"]["command_selected_samples"] == 2
    assert payload["offline_update"]["command_intent_loss"] > 0.0
    assert payload["offline_update"]["command_intent_target_count"] == 2
    assert payload["offline_update"]["balance_key"] == "role"
    assert payload["offline_update"]["batch_label_counts"] == [{"walk_flat": 1, "stand": 1}]
    assert payload["offline_update"]["last_balance_label_counts"] == {
        "walk_flat": 1,
        "stand": 1,
    }


def test_distill_script_collects_live_env_dataset_with_owner_projection(tmp_path: Path):
    from unilab.algos.torch.distill import load_distillation_dataset

    mod = _train_distill()
    cfg = _distill_cfg(
        [
            "training.collect_num_samples=3",
            "training.collect_num_envs=2",
            "+training.collect_workflow_scenario=walk_flat",
        ]
    )
    dataset_path = tmp_path / "collected_dataset.pt"
    calls: dict[str, Any] = {}
    fake_env = _FakeDistillCollectEnv(num_envs=2, action_dim=29)

    def create_env_fn(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return fake_env

    probe = mod.run_collect_dataset(
        cfg,
        dataset_path=dataset_path,
        create_env_fn=create_env_fn,
        env_cfg_override_fn=lambda cfg: {"owner": "distill-test"},
    )

    assert probe["distill_source"] == "live_env_rollout"
    assert probe["dataset_path"] == str(dataset_path)
    assert probe["dataset_num_samples"] == 3
    assert probe["dataset_student_obs_dim"] == 99
    assert probe["dataset_teacher_obs_dim"] == 99
    assert probe["student_obs_shape"] == (3, 99)
    assert probe["teacher_obs_shape"] == (3, 99)
    assert probe["collect_num_envs"] == 2
    assert probe["collect_action_mode"] == "zero"
    assert probe["collect_action_seed"] is None
    assert probe["collect_action_abs_max"] == 0.0
    assert probe["teacher_projection"] == "identity"
    assert probe["student_projection"] == "identity"
    assert probe["student_drop_index"] is None
    assert calls["kwargs"]["num_envs"] == 2
    assert calls["kwargs"]["env_cfg_override"] == {"owner": "distill-test"}
    assert calls["kwargs"]["sim_backend"] == "mujoco"
    assert calls["kwargs"]["task_name"] == "G1WalkHeight"
    assert fake_env.reset_calls == 1
    assert fake_env.step_calls == 1
    assert fake_env.closed is True

    restored = load_distillation_dataset(
        dataset_path,
        expected_student_obs_dim=99,
        expected_teacher_obs_dim=99,
    )
    assert restored.metadata["source"] == "live_env_rollout"
    assert restored.metadata["teacher_projection"] == "identity"
    assert restored.metadata["student_projection"] == "identity"
    assert restored.metadata["student_drop_index"] is None
    assert restored.metadata["teacher_obs_key"] == "obs"
    assert restored.metadata["synthetic_teacher_tail"] is False
    assert restored.metadata["workflow_scenario"] == "walk_flat"


def test_distill_script_collects_stand_still_dataset_with_owner_config(tmp_path: Path):
    from unilab.algos.torch.distill import load_distillation_dataset

    mod = _train_distill()
    cfg = _distill_cfg(
        [
            "task=g1_stand_still/mujoco",
            "training.collect_num_samples=3",
            "training.collect_num_envs=2",
        ]
    )
    dataset_path = tmp_path / "stand_still_collected_dataset.pt"
    fake_env = _FakeDistillCollectEnv(
        num_envs=2,
        action_dim=29,
        obs_dim=98,
        critic_dim=101,
        command_batches=[
            np.zeros((2, 3), dtype=np.float32),
            np.zeros((2, 3), dtype=np.float32),
        ],
    )
    calls: dict[str, Any] = {}

    def create_env_fn(*args, **kwargs):
        calls["kwargs"] = kwargs
        return fake_env

    probe = mod.run_collect_dataset(
        cfg,
        dataset_path=dataset_path,
        create_env_fn=create_env_fn,
        env_cfg_override_fn=lambda cfg: {"owner": "stand-still-distill-test"},
    )

    assert probe["dataset_num_samples"] == 3
    assert probe["dataset_student_obs_dim"] == 98
    assert probe["dataset_teacher_obs_dim"] == 98
    assert probe["student_obs_shape"] == (3, 98)
    assert probe["teacher_obs_shape"] == (3, 98)
    assert probe["teacher_projection"] == "identity"
    assert probe["student_projection"] == "identity"
    assert probe["collect_command_sample_filter"] == "inactive"
    assert probe["collect_command_seen_samples"] == 4
    assert probe["collect_command_selected_samples"] == 4
    assert probe["collect_command_intent_counts"] == {"inactive": 3}
    assert calls["kwargs"]["task_name"] == "G1StandStill"
    assert calls["kwargs"]["sim_backend"] == "mujoco"
    assert calls["kwargs"]["env_cfg_override"] == {"owner": "stand-still-distill-test"}
    assert fake_env.step_calls == 1
    assert fake_env.closed is True

    restored = load_distillation_dataset(
        dataset_path,
        expected_student_obs_dim=98,
        expected_teacher_obs_dim=98,
    )
    assert restored.metadata["task_name"] == "G1StandStill"
    assert restored.metadata["teacher_projection"] == "identity"
    assert restored.metadata["student_projection"] == "identity"
    assert restored.metadata["synthetic_teacher_tail"] is False
    assert restored.metadata["command_sample_filter"] == "inactive"
    assert restored.metadata["command_selected_samples"] == 4
    assert restored.metadata["command_intent_counts"] == {"inactive": 3}


def test_distill_script_collects_owner_filtered_walk_dataset(tmp_path: Path):
    import torch

    from unilab.algos.torch.distill import load_distillation_dataset

    mod = _train_distill()
    cfg = _distill_cfg(
        [
            "task=g1_walk_flat/mujoco",
            "training.collect_num_samples=2",
            "training.collect_num_envs=2",
            "training.collect_max_env_steps=1",
        ]
    )
    assert cfg.training.collect_command_sample_filter == "active"
    dataset_path = tmp_path / "walk_filtered_dataset.pt"
    fake_env = _FakeDistillCollectEnv(
        num_envs=2,
        action_dim=29,
        obs_dim=98,
        critic_dim=101,
        command_batches=[
            np.asarray([[0.0, 0.0, 0.0], [0.10, 0.0, 0.0]], dtype=np.float32),
            np.asarray([[0.0, 0.0, 0.10], [0.0, 0.0, 0.0]], dtype=np.float32),
        ],
    )

    probe = mod.run_collect_dataset(
        cfg,
        dataset_path=dataset_path,
        create_env_fn=lambda *args, **kwargs: fake_env,
        env_cfg_override_fn=lambda cfg: {"owner": "walk-filter-distill-test"},
    )

    assert probe["dataset_num_samples"] == 2
    assert probe["collect_command_sample_filter"] == "active"
    assert probe["collect_command_seen_samples"] == 4
    assert probe["collect_command_selected_samples"] == 2
    assert probe["collect_command_intent_counts"] == {"active": 2}

    restored = load_distillation_dataset(
        dataset_path,
        expected_student_obs_dim=98,
        expected_teacher_obs_dim=98,
    )
    assert restored.metadata["command_sample_filter"] == "active"
    assert restored.metadata["command_seen_samples"] == 4
    assert restored.metadata["command_selected_samples"] == 2
    assert restored.metadata["command_intent_counts"] == {"active": 2}
    assert torch.equal(restored.teacher_obs[0], torch.arange(98, 196, dtype=torch.float32))
    assert torch.equal(restored.teacher_obs[1], torch.arange(98, dtype=torch.float32) + 1.0)


@pytest.mark.parametrize(
    ("task_override", "bad_filter", "expected_filter"),
    [
        ("task=g1_walk_flat/mujoco", "inactive", "active"),
        ("task=g1_stand_still/mujoco", "active", "inactive"),
    ],
)
def test_distill_script_rejects_owner_command_filter_override(
    tmp_path: Path,
    task_override: str,
    bad_filter: str,
    expected_filter: str,
) -> None:
    mod = _train_distill()
    cfg = _distill_cfg(
        [
            task_override,
            f"training.collect_command_sample_filter={bad_filter}",
            "training.collect_num_samples=2",
            "training.collect_num_envs=2",
        ]
    )

    with pytest.raises(
        ValueError, match=f"requires training.collect_command_sample_filter={expected_filter}"
    ):
        mod.run_collect_dataset(
            cfg,
            dataset_path=tmp_path / "bad_owner_filter.pt",
            create_env_fn=lambda *args, **kwargs: _FakeDistillCollectEnv(),
            env_cfg_override_fn=lambda cfg: {"owner": "must-not-create-env"},
        )


def test_distill_script_rejects_incomplete_teacher_policy_height_route(
    tmp_path: Path,
) -> None:
    mod = _train_distill()
    cfg = _distill_cfg(["training.collect_action_mode=teacher_policy"])

    with pytest.raises(ValueError, match="observed height_commands"):
        mod.run_collect_dataset(
            cfg,
            dataset_path=tmp_path / "height_teacher_policy_dataset.pt",
            create_env_fn=lambda *args, **kwargs: _FakeDistillCollectEnv(),
            env_cfg_override_fn=lambda cfg: {"owner": "must-not-create-env"},
        )


def test_distill_script_collects_stand_still_teacher_policy_dataset_and_updates(
    tmp_path: Path,
) -> None:
    import torch

    from unilab.algos.torch.distill import (
        load_distillation_dataset,
        load_distillation_student_policy,
    )
    from unilab.algos.torch.fast_sac.learner import SACActor

    mod = _train_distill()
    teacher = SACActor(98, 29, hidden_dim=16, use_layer_norm=False, device="cpu")
    teacher_checkpoint = tmp_path / "G1StandStill" / "model_5000.pt"
    teacher_checkpoint.parent.mkdir(parents=True)
    teacher_state = teacher.state_dict()
    for key, value in teacher_state.items():
        if torch.is_floating_point(value):
            teacher_state[key] = torch.full_like(value, 0.01)
    torch.save({"actor": teacher_state}, teacher_checkpoint)
    cfg = _distill_cfg(
        [
            "task=g1_stand_still/mujoco",
            f"teacher.load_run={teacher_checkpoint}",
            "teacher.checkpoint=-1",
            "teacher.actor_hidden_dim=16",
            "teacher.use_layer_norm=false",
            "teacher.obs_normalization=false",
            "student.hidden_dims=[32]",
            "training.collect_num_samples=3",
            "training.collect_num_envs=2",
            "training.collect_action_mode=teacher_policy",
        ]
    )
    dataset_path = tmp_path / "stand_teacher_policy_dataset.pt"
    student_checkpoint = tmp_path / "stand_student.pt"
    fake_env = _FakeDistillCollectEnv(
        num_envs=2,
        action_dim=29,
        obs_dim=98,
        critic_dim=101,
        command_batches=[
            np.zeros((2, 3), dtype=np.float32),
            np.zeros((2, 3), dtype=np.float32),
        ],
    )

    collect_probe = mod.run_collect_dataset(
        cfg,
        dataset_path=dataset_path,
        create_env_fn=lambda *args, **kwargs: fake_env,
        env_cfg_override_fn=lambda cfg: {"owner": "stand-teacher-policy-test"},
    )

    assert collect_probe["dataset_student_obs_dim"] == 98
    assert collect_probe["dataset_teacher_obs_dim"] == 98
    assert collect_probe["collect_action_mode"] == "teacher_policy"
    assert collect_probe["collect_action_seed"] is None
    assert collect_probe["collect_action_abs_max"] > 0.0
    assert collect_probe["collect_command_sample_filter"] == "inactive"
    assert collect_probe["collect_command_intent_counts"] == {"inactive": 3}
    assert collect_probe["teacher_policy_checkpoint_path"] == str(teacher_checkpoint)
    assert fake_env.last_actions is not None
    assert np.isfinite(fake_env.last_actions).all()
    assert np.max(np.abs(fake_env.last_actions)) > 0.0

    restored = load_distillation_dataset(
        dataset_path,
        expected_student_obs_dim=98,
        expected_teacher_obs_dim=98,
        expected_teacher_action_dim=29,
    )
    assert restored.metadata["task_name"] == "G1StandStill"
    assert restored.metadata["action_mode"] == "teacher_policy"
    assert restored.metadata["teacher_policy_checkpoint_path"] == str(teacher_checkpoint)
    assert restored.metadata["command_intent_counts"] == {"inactive": 3}
    assert restored.teacher_actions is not None
    assert restored.teacher_actions.shape == (3, 29)
    assert torch.isfinite(restored.teacher_actions).all()

    update_probe = mod.run_offline_dataset_update(
        cfg,
        teacher_checkpoint=teacher_checkpoint,
        dataset_path=dataset_path,
        batch_size=2,
        max_updates=1,
        checkpoint_path=student_checkpoint,
        device="cpu",
    )

    assert update_probe["distill_source"] == "offline_dataset"
    assert update_probe["dataset_student_obs_dim"] == 98
    assert update_probe["dataset_teacher_obs_dim"] == 98
    assert update_probe["teacher_obs_shape"] == (2, 98)
    assert update_probe["student_obs_shape"] == (2, 98)
    assert update_probe["teacher_action_shape"] == (2, 29)
    assert update_probe["student_action_shape"] == (2, 29)
    assert update_probe["teacher_action_requires_grad"] is False
    assert update_probe["update_count"] == 1
    assert update_probe["checkpoint_path"] == str(student_checkpoint)

    loaded_student = load_distillation_student_policy(student_checkpoint, device="cpu")
    assert loaded_student.obs_dim == 98
    assert loaded_student.action_dim == 29


def test_distill_script_collects_walk_flat_teacher_policy_cached_dataset(
    tmp_path: Path,
) -> None:
    import torch

    from unilab.algos.torch.distill import load_distillation_dataset
    from unilab.algos.torch.fast_sac.learner import SACActor

    mod = _train_distill()
    teacher = SACActor(98, 29, hidden_dim=16, use_layer_norm=False, device="cpu")
    teacher_checkpoint = tmp_path / "walk_teacher.pt"
    teacher_state = teacher.state_dict()
    for key, value in teacher_state.items():
        if torch.is_floating_point(value):
            teacher_state[key] = torch.full_like(value, 0.02)
    torch.save({"actor": teacher_state}, teacher_checkpoint)
    cfg = _distill_cfg(
        [
            "task=g1_walk_flat/mujoco",
            f"teacher.load_run={teacher_checkpoint}",
            "teacher.checkpoint=-1",
            "teacher.actor_hidden_dim=16",
            "teacher.use_layer_norm=false",
            "teacher.obs_normalization=false",
            "training.collect_num_samples=3",
            "training.collect_num_envs=2",
            "training.collect_action_mode=teacher_policy",
        ]
    )
    dataset_path = tmp_path / "walk_teacher_policy_dataset.pt"
    fake_env = _FakeDistillCollectEnv(
        num_envs=2,
        action_dim=29,
        obs_dim=98,
        critic_dim=101,
        command_batches=[
            np.asarray([[0.10, 0.0, 0.0], [0.20, 0.0, 0.0]], dtype=np.float32),
            np.asarray([[0.0, 0.0, 0.10], [0.30, 0.0, 0.0]], dtype=np.float32),
        ],
    )

    collect_probe = mod.run_collect_dataset(
        cfg,
        dataset_path=dataset_path,
        create_env_fn=lambda *args, **kwargs: fake_env,
        env_cfg_override_fn=lambda cfg: {"owner": "walk-flat-teacher-policy-test"},
    )

    assert collect_probe["dataset_student_obs_dim"] == 98
    assert collect_probe["dataset_teacher_obs_dim"] == 98
    assert collect_probe["collect_action_mode"] == "teacher_policy"
    assert collect_probe["collect_action_abs_max"] > 0.0
    assert collect_probe["collect_command_sample_filter"] == "active"
    assert collect_probe["collect_command_intent_counts"] == {"active": 3}
    assert collect_probe["teacher_policy_checkpoint_path"] == str(teacher_checkpoint)
    assert fake_env.last_actions is not None
    assert np.isfinite(fake_env.last_actions).all()

    restored = load_distillation_dataset(
        dataset_path,
        expected_student_obs_dim=98,
        expected_teacher_obs_dim=98,
        expected_teacher_action_dim=29,
    )
    assert restored.metadata["task_name"] == "G1WalkFlat"
    assert restored.metadata["action_mode"] == "teacher_policy"
    assert restored.metadata["teacher_policy_checkpoint_path"] == str(teacher_checkpoint)
    assert restored.metadata["command_intent_counts"] == {"active": 3}
    assert restored.teacher_actions is not None
    assert restored.teacher_actions.shape == (3, 29)
    assert torch.isfinite(restored.teacher_actions).all()


def test_distill_script_collects_walk_flat_inactive_student_rollout_with_stand_teacher(
    tmp_path: Path,
) -> None:
    import torch

    from unilab.algos.torch.distill import (
        MLPStudentPolicy,
        load_distillation_dataset,
        save_distillation_checkpoint,
    )
    from unilab.algos.torch.fast_sac.learner import SACActor

    mod = _train_distill()
    teacher = SACActor(98, 29, hidden_dim=16, use_layer_norm=False, device="cpu")
    teacher_checkpoint = tmp_path / "G1StandStill" / "model_5000.pt"
    teacher_checkpoint.parent.mkdir(parents=True)
    teacher_state = teacher.state_dict()
    for key, value in teacher_state.items():
        if torch.is_floating_point(value):
            teacher_state[key] = torch.full_like(value, 0.015)
    torch.save({"actor": teacher_state}, teacher_checkpoint)

    rollout_student = MLPStudentPolicy(obs_dim=98, action_dim=29, hidden_dims=(32,))
    rollout_checkpoint = tmp_path / "rollout_student.pt"
    save_distillation_checkpoint(
        rollout_checkpoint,
        student=rollout_student,
        agent_steps=128,
        distill_runtime_cfg={
            "student_model_type": "mlp",
            "student_obs_dim": 98,
            "student_action_dim": 29,
            "student_hidden_dims": [32],
            "student_activation": "elu",
            "student_squash_action": True,
        },
    )

    cfg = _distill_cfg(
        [
            "task=g1_walk_flat/mujoco",
            f"teacher.checkpoint_path={teacher_checkpoint}",
            "teacher.actor_hidden_dim=16",
            "teacher.use_layer_norm=false",
            "teacher.obs_normalization=false",
            "student.hidden_dims=[32]",
            "training.collect_num_samples=3",
            "training.collect_num_envs=2",
            "training.collect_command_sample_filter=inactive",
            "training.collect_action_mode=student_policy",
            f"training.collect_rollout_checkpoint_path={rollout_checkpoint}",
        ]
    )
    dataset_path = tmp_path / "walk_owner_stand_teacher_student_rollout.pt"
    fake_env = _FakeDistillCollectEnv(
        num_envs=2,
        action_dim=29,
        obs_dim=98,
        critic_dim=101,
        command_batches=[
            np.zeros((2, 3), dtype=np.float32),
            np.zeros((2, 3), dtype=np.float32),
        ],
    )
    calls: dict[str, Any] = {}

    def create_env_fn(*args, **kwargs):
        calls["cfg"] = args[0]
        calls["kwargs"] = kwargs
        return fake_env

    collect_probe = mod.run_collect_dataset(
        cfg,
        dataset_path=dataset_path,
        create_env_fn=create_env_fn,
        env_cfg_override_fn=lambda cfg: {
            "owner": "walk-flat-stand-teacher-dagger-test",
            "rel_standing_envs": OmegaConf.select(cfg, "env.commands.rel_standing_envs"),
            "rel_transition_envs": OmegaConf.select(cfg, "env.commands.rel_transition_envs"),
            "vel_limit": OmegaConf.to_container(
                OmegaConf.select(cfg, "env.commands.vel_limit"),
                resolve=True,
            ),
            "transition_vel_limit": OmegaConf.to_container(
                OmegaConf.select(cfg, "env.commands.transition_vel_limit"),
                resolve=True,
            ),
        },
    )

    assert collect_probe["collect_command_sample_filter"] == "inactive"
    assert collect_probe["collect_action_mode"] == "student_policy"
    assert collect_probe["collect_command_intent_counts"] == {"inactive": 3}
    assert collect_probe["collect_command_distribution_overrides"] == {
        "env.commands.vel_limit": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "env.commands.transition_vel_limit": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "env.commands.rel_standing_envs": 1.0,
        "env.commands.rel_transition_envs": 0.0,
        "env.commands.small_xy_threshold": 0.0,
    }
    assert collect_probe["teacher_policy_checkpoint_path"] == str(teacher_checkpoint)
    assert collect_probe["rollout_policy_checkpoint_path"] == str(rollout_checkpoint)
    assert calls["kwargs"]["env_cfg_override"] == {
        "owner": "walk-flat-stand-teacher-dagger-test",
        "rel_standing_envs": 1.0,
        "rel_transition_envs": 0.0,
        "vel_limit": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "transition_vel_limit": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    }
    assert fake_env.last_actions is not None
    assert np.isfinite(fake_env.last_actions).all()

    restored = load_distillation_dataset(
        dataset_path,
        expected_student_obs_dim=98,
        expected_teacher_obs_dim=98,
        expected_teacher_action_dim=29,
    )
    assert restored.metadata["task_name"] == "G1WalkFlat"
    assert restored.metadata["action_mode"] == "student_policy"
    assert restored.metadata["teacher_policy_checkpoint_path"] == str(teacher_checkpoint)
    assert restored.metadata["rollout_policy_checkpoint_path"] == str(rollout_checkpoint)
    assert restored.metadata["command_intent_counts"] == {"inactive": 3}
    assert restored.metadata["command_distribution_overrides"] == {
        "env.commands.vel_limit": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "env.commands.transition_vel_limit": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "env.commands.rel_standing_envs": 1.0,
        "env.commands.rel_transition_envs": 0.0,
        "env.commands.small_xy_threshold": 0.0,
    }
    assert restored.teacher_actions is not None
    assert restored.teacher_actions.shape == (3, 29)


def test_distill_script_formal_stand_still_run_writes_metadata_and_checkpoint(
    tmp_path: Path,
) -> None:
    import torch

    from unilab.algos.torch.distill import (
        build_distillation_dataset,
        load_distillation_student_policy,
        save_distillation_dataset,
    )
    from unilab.algos.torch.fast_sac.learner import SACActor

    mod = _train_distill()
    teacher = SACActor(98, 29, hidden_dim=16, use_layer_norm=False, device="cpu")
    teacher_checkpoint = tmp_path / "stand_teacher.pt"
    torch.save({"actor": teacher.state_dict()}, teacher_checkpoint)
    dataset_path = tmp_path / "stand_dataset.pt"
    dataset = build_distillation_dataset(
        torch.randn(4, 98),
        torch.randn(4, 98),
        expected_student_obs_dim=98,
        expected_teacher_obs_dim=98,
        metadata={"source": "stand_fixture", "action_mode": "teacher_policy"},
    )
    save_distillation_dataset(dataset_path, dataset)
    run_dir = tmp_path / "formal_run"
    cfg = _distill_cfg(
        [
            "task=g1_stand_still/mujoco",
            f"teacher.load_run={teacher_checkpoint}",
            "teacher.checkpoint=-1",
            "teacher.actor_hidden_dim=16",
            "teacher.use_layer_norm=false",
            "teacher.obs_normalization=false",
            "student.hidden_dims=[32]",
            "training.formal_run=true",
            f"training.formal_run_dir={run_dir}",
            f"training.offline_dataset_path={dataset_path}",
            "training.offline_batch_size=2",
            "training.offline_max_updates=2",
        ]
    )

    probe = mod.run_formal_offline_dataset_update(
        cfg,
        teacher_checkpoint=teacher_checkpoint,
        device="cpu",
    )

    checkpoint_path = run_dir / "model_4.pt"
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    run_summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))

    assert probe["distill_source"] == "formal_offline_dataset"
    assert probe["run_dir"] == str(run_dir)
    assert probe["checkpoint_path"] == str(checkpoint_path)
    assert probe["update_count"] == 2
    assert probe["samples_seen"] == 4
    assert checkpoint_path.exists()
    assert run_config["run"]["algo"] == "distill"
    assert run_config["run"]["task"] == "G1StandStill"
    assert run_config["config"]["training"]["formal_run"] is True
    assert run_config["config"]["training"]["formal_run_dir"] == str(run_dir)
    assert run_config["config"]["training"]["offline_dataset_path"] == str(dataset_path)
    assert run_summary["status"] == "completed"
    assert run_summary["distill_source"] == "formal_offline_dataset"
    assert run_summary["checkpoint_path"] == str(checkpoint_path)
    assert run_summary["samples_seen"] == 4

    loaded_student = load_distillation_student_policy(checkpoint_path, device="cpu")
    assert loaded_student.obs_dim == 98
    assert loaded_student.action_dim == 29
    assert loaded_student.agent_steps == 4
    assert loaded_student.distill_runtime_cfg["distill_source"] == "offline_dataset"


def test_distill_script_collects_random_action_dataset_with_seed(tmp_path: Path):
    from unilab.algos.torch.distill import load_distillation_dataset

    mod = _train_distill()
    cfg = _distill_cfg(
        [
            "training.collect_num_samples=3",
            "training.collect_num_envs=2",
            "training.collect_action_mode=random",
            "training.collect_action_seed=11",
        ]
    )
    dataset_path = tmp_path / "random_collected_dataset.pt"
    fake_env = _FakeDistillCollectEnv(num_envs=2, action_dim=29)

    probe = mod.run_collect_dataset(
        cfg,
        dataset_path=dataset_path,
        create_env_fn=lambda *args, **kwargs: fake_env,
        env_cfg_override_fn=lambda cfg: {"owner": "distill-test"},
    )

    assert probe["collect_action_mode"] == "random"
    assert probe["collect_action_seed"] == 11
    assert probe["collect_action_abs_max"] > 0.0
    assert fake_env.last_actions is not None
    assert np.isfinite(fake_env.last_actions).all()
    assert np.max(np.abs(fake_env.last_actions)) > 0.0

    restored = load_distillation_dataset(
        dataset_path,
        expected_student_obs_dim=99,
        expected_teacher_obs_dim=99,
    )
    assert restored.metadata["action_mode"] == "random"
    assert restored.metadata["action_seed"] == 11
    assert restored.metadata["action_abs_max"] > 0.0


def test_hora_distill_runtime_checkpoint_records_model_only():
    mod = _train_hora_distill()
    cfg = OmegaConf.create(
        {
            "training": {
                "task_name": "OwnerTask",
                "sim_backend": "mujoco",
                "cam_distance": 1.5,
            },
            "env": {
                "post_step_forward_sensor": True,
                "domain_rand": {"force_scale": 1.2},
            },
            "reward": {"scales": {"rotate": 2.5}},
            "algo": {"model": {"hidden_dims": [512, 256, 128]}},
        }
    )

    runtime = OmegaConf.to_container(mod._resolved_distill_runtime_cfg(cfg), resolve=True)

    assert runtime == {"algo": {"model": {"hidden_dims": [512, 256, 128]}}}


def test_hora_distill_checkpoint_runtime_only_restores_model_structure():
    mod = _train_hora_distill()
    cfg = _hora_distill_cfg(["task=sharpa_inhand/mujoco_nodr"])
    checkpoint = {
        "distill_runtime_cfg": {
            "training": {
                "task_name": "CheckpointTask",
                "sim_backend": "motrix",
                "render_spacing": 99.0,
            },
            "reward": {"scales": {"rotate": 999.0}},
            "env": {
                "post_step_forward_sensor": False,
                "domain_rand": {
                    "scale_list": [9.9],
                    "randomize_mass": True,
                    "force_scale": 99.0,
                },
            },
            "algo": {
                "model": {
                    "hidden_dims": [32, 16],
                    "priv_info_embed_dim": 7,
                    "priv_mlp_hidden_dims": [11, 7],
                }
            },
        }
    }

    restored = mod._cfg_with_checkpoint_runtime(cfg, checkpoint)

    assert restored.training.task_name == "SharpaInhandRotation"
    assert restored.training.sim_backend == "mujoco"
    assert restored.training.render_spacing == pytest.approx(0.5)
    assert restored.reward.scales.rotate != pytest.approx(999.0)
    assert restored.env.post_step_forward_sensor is True
    assert restored.env.domain_rand.scale_list == [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    assert restored.env.domain_rand.randomize_mass is False
    assert restored.env.domain_rand.force_scale == pytest.approx(0.0)
    assert restored.algo.model.hidden_dims == [32, 16]
    assert restored.algo.model.priv_info_embed_dim == 7
    assert restored.algo.model.priv_mlp_hidden_dims == [11, 7]


@pytest.mark.parametrize(
    ("teacher_algo_family", "checkpoint_model"),
    [
        ("ppo", {"hidden_dims": [512, 256, 128], "activation": "elu"}),
        ("appo", {"hidden_dims": [512, 256, 128], "activation": "elu"}),
        (
            "sac",
            {
                "teacher_arch": "hora_sac",
                "actor_hidden_dim": 512,
                "use_layer_norm": True,
            },
        ),
    ],
)
def test_hora_distill_checkpoint_runtime_only_overrides_model_side(
    monkeypatch: pytest.MonkeyPatch,
    teacher_algo_family: str,
    checkpoint_model: dict[str, Any],
):
    mod = _train_hora_distill()
    owner_cfg = OmegaConf.create(
        {
            "teacher": {"algo_family": teacher_algo_family},
            "training": {
                "task_name": "OwnerTask",
                "sim_backend": "mujoco",
                "cam_distance": 1.5,
            },
            "env": {
                "post_step_forward_sensor": False,
                "domain_rand": {"force_scale": 1.2, "randomize_mass": False},
            },
            "reward": {"scales": {"rotate": 2.5}},
            "algo": {"model": {"owner_model": True}},
        }
    )
    checkpoint = {
        "teacher_algo_family": teacher_algo_family,
        "distill_runtime_cfg": {
            "training": {
                "task_name": "CheckpointTask",
                "sim_backend": "mujoco",
                "cam_distance": 9.0,
            },
            "env": {
                "post_step_forward_sensor": True,
                "domain_rand": {"force_scale": 9.0, "randomize_mass": True},
            },
            "reward": {"scales": {"rotate": 99.0}},
            "algo": {"model": checkpoint_model},
        },
    }

    monkeypatch.setattr(mod, "_apply_teacher_defaults", lambda cfg: owner_cfg)

    effective_cfg = mod._cfg_with_checkpoint_runtime(OmegaConf.create({}), checkpoint)

    assert effective_cfg.training.task_name == "OwnerTask"
    assert effective_cfg.training.cam_distance == pytest.approx(1.5)
    assert effective_cfg.env.post_step_forward_sensor is False
    assert effective_cfg.env.domain_rand.force_scale == pytest.approx(1.2)
    assert effective_cfg.env.domain_rand.randomize_mass is False
    assert effective_cfg.reward.scales.rotate == pytest.approx(2.5)
    assert OmegaConf.to_container(effective_cfg.algo.model, resolve=True) == checkpoint_model


def test_hora_distill_script_delegates_teacher_owner_resolution():
    source = (_SCRIPTS_DIR / "train_hora_distill.py").read_text(encoding="utf-8")

    assert "OmegaConf.load" not in source
    assert "HoraActorModel" not in source
    assert 'conf" / str(algo_family)' not in source


@pytest.mark.parametrize("teacher_algo_family", ["ppo", "appo", "sac"])
def test_hora_distill_teacher_owner_defaults_support_ppo_appo_and_sac(
    teacher_algo_family: str,
):
    mod = _train_hora_distill()
    teacher_task = (
        "sac/sharpa_inhand/mujoco_hora"
        if teacher_algo_family == "sac"
        else "sharpa_inhand/mujoco_hora"
    )
    cfg = mod._apply_teacher_defaults(
        _hora_distill_cfg(
            [
                "task=sharpa_inhand/mujoco",
                f"teacher.algo_family={teacher_algo_family}",
                f"teacher.task={teacher_task}",
            ]
        )
    )

    assert cfg.training.task_name == "SharpaInhandRotation"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.algo.model.priv_info_embed_dim == 9
    assert cfg.algo.model.priv_mlp_hidden_dims == [256, 128, 9]
    if teacher_algo_family == "sac":
        assert cfg.algo.model.teacher_arch
        assert cfg.algo.model.actor_hidden_dim is not None


def test_hora_distill_sac_teacher_requires_hora_sac_runtime():
    mod = _train_hora_distill()

    with pytest.raises(ValueError, match="runtime_impl='hora_sac'"):
        mod._apply_teacher_defaults(
            _hora_distill_cfg(
                [
                    "task=sharpa_inhand/mujoco",
                    "teacher.algo_family=sac",
                    "teacher.task=sac/g1_walk_flat/mujoco",
                ]
            )
        )


@pytest.mark.parametrize("teacher_algo_family", ["ppo", "appo"])
def test_hora_distill_teacher_run_slug_omits_teacher_run_name(teacher_algo_family: str):
    mod = _train_hora_distill()
    cfg = OmegaConf.create({"teacher": {"task": "sharpa_inhand/mujoco"}})
    teacher_checkpoint = Path("/tmp") / "2026-04-22_13-26-45_mujoco" / "model_10000.pt"

    metadata = mod._teacher_run_metadata(
        cfg,
        teacher_algo_family=teacher_algo_family,
        teacher_checkpoint=teacher_checkpoint,
    )

    assert metadata["run_name"] == "2026-04-22_13-26-45_mujoco"
    assert metadata["run_slug"] == f"teacher-{teacher_algo_family}"


def test_offpolicy_go1_motrix_task_is_not_configured():
    """SAC has no Go1 Motrix owner config; use PPO for Go1 joystick tasks."""
    from hydra.errors import MissingConfigException

    with pytest.raises(MissingConfigException, match="task/sac/go1_joystick_flat/motrix"):
        _offpolicy_cfg(["task=sac/go1_joystick_flat/motrix"])


def test_offpolicy_g1_walk_flat_motrix_resolved_algo_matches_task_owner():
    """Motrix SAC G1 walk flat composes backend-owned algo hyperparameters."""
    cfg = _offpolicy_cfg(["task=sac/g1_walk_flat/motrix"])

    assert cfg.algo.num_envs == 2048
    assert cfg.algo.max_iterations == 5000
    assert cfg.algo.use_symmetry is False


def test_offpolicy_g1_walk_flat_env_cfg_override_has_reward_and_domain_rand():
    cfg = _offpolicy_cfg(["task=sac/g1_walk_flat/motrix"])

    env_cfg_override = _offpolicy().build_offpolicy_env_cfg_override("sac", cfg)

    assert env_cfg_override["reward_config"]["scales"]["tracking_lin_vel"] == pytest.approx(2.2)
    assert env_cfg_override["domain_rand"]["randomize_kp"] is False
    assert env_cfg_override["domain_rand"]["randomize_kd"] is False


def test_offpolicy_g1_walk_flat_backend_scoped_use_symmetry():
    mujoco_cfg = _offpolicy_cfg(["task=sac/g1_walk_flat/mujoco"])
    motrix_cfg = _offpolicy_cfg(["task=sac/g1_walk_flat/motrix"])

    assert mujoco_cfg.algo.use_symmetry is True
    assert motrix_cfg.algo.use_symmetry is False


def test_ppo_go1_resolved_algo_matches_old_motrix_behavior():
    """Equivalence: PPO Go1 algo hyperparams match pre-refactor motrix values."""
    cfg = _ppo_cfg(["task=go1_joystick_flat/motrix"])

    assert cfg.algo.max_iterations == 151
    assert cfg.algo.empirical_normalization is True
    assert cfg.algo.policy.init_noise_std == pytest.approx(0.5)
    assert cfg.algo.algorithm.learning_rate == pytest.approx(3.0e-4)
    assert cfg.algo.algorithm.entropy_coef == pytest.approx(1.0e-3)


def test_ppo_g1_resolved_algo_matches_motrix_owner():
    """Equivalence: PPO G1 algo hyperparams match the Motrix owner values.

    For this migration we align with the final UniLab1 Motrix runtime.
    """
    cfg = _ppo_cfg(["task=g1_walk_flat/motrix"])

    assert cfg.algo.max_iterations == 2200
    assert cfg.algo.empirical_normalization is True
    assert cfg.algo.obs_groups.actor == ["policy"]
    assert cfg.algo.policy.init_noise_std == pytest.approx(0.5)
    assert cfg.algo.algorithm.learning_rate == pytest.approx(3.0e-4)
    assert cfg.algo.algorithm.entropy_coef == pytest.approx(5.0e-3)


def test_ppo_g1_mujoco_base_hyperparams_remain_separate():
    cfg = _ppo_cfg(["task=g1_walk_flat/mujoco"])

    assert cfg.algo.max_iterations == 2200
    assert cfg.algo.empirical_normalization is False
    assert cfg.algo.obs_groups.actor == ["actor"]


def test_ppo_g1_env_preset_has_env_overrides():
    cfg = _ppo_cfg(["task=g1_walk_flat/motrix"])

    assert OmegaConf.select(cfg, "env.motrix_max_iterations") is None
    assert cfg.env.control_config.action_scale == pytest.approx(0.5)
    assert cfg.env.commands.vel_limit == [[0.4, 0.0, 0.0], [0.7, 0.0, 0.0]]
    assert cfg.env.gait_phase_init_mode == "offset_phase"
    assert cfg.env.reset_base_qvel_limit == pytest.approx(0.05)
    assert cfg.reward.scales.feet_phase_contrast == pytest.approx(1.5)
    assert cfg.reward.scales.feet_phase_contact == pytest.approx(1.0)
    assert cfg.reward.scales.feet_double_stance == pytest.approx(-1.0)
    assert cfg.reward.min_forward_speed_for_gait_reward == pytest.approx(0.05)


def test_ppo_task_go2_aligns_mujoco_with_motrix_defaults():
    cfg = _ppo_cfg(["task=go2_joystick_flat/mujoco"])

    assert cfg.algo.num_envs == 1024
    assert cfg.reward.scales.tracking_lin_vel == pytest.approx(1.0)
    assert cfg.reward.scales.tracking_ang_vel == pytest.approx(0.2)
    assert cfg.reward.scales.lin_vel_z == pytest.approx(-5.0)
    assert cfg.reward.scales.ang_vel_xy == pytest.approx(-0.1)
    assert cfg.algo.empirical_normalization is True
    assert cfg.algo.policy.init_noise_std == pytest.approx(0.5)
    assert cfg.algo.algorithm.learning_rate == pytest.approx(3.0e-4)
    assert cfg.algo.algorithm.entropy_coef == pytest.approx(1.0e-3)


def test_build_ppo_env_cfg_override_go1_motrix(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=go1_joystick_flat/motrix"])

    env_cfg_override = mod.build_ppo_env_cfg_override(cfg)

    # env_cfg_override has reward + env preset commands
    assert env_cfg_override["reward_config"]["scales"]["tracking_lin_vel"] == pytest.approx(1.0)
    assert env_cfg_override["commands"]["vel_limit"] == [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]


def test_build_ppo_env_cfg_override_g1_motrix(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=g1_walk_flat/motrix"])

    env_cfg_override = mod.build_ppo_env_cfg_override(cfg)

    # env_cfg_override has reward + env preset fields (flat, matching env cfg structure)
    assert env_cfg_override["reward_config"]["scales"]["upper_body_pose"] == pytest.approx(-0.05)
    assert env_cfg_override["reward_config"]["scales"]["penalty_feet_ori"] == pytest.approx(0.0)
    assert env_cfg_override["reward_config"]["scales"]["feet_phase_contrast"] == pytest.approx(1.5)
    assert env_cfg_override["reward_config"]["scales"]["feet_phase_contact"] == pytest.approx(1.0)
    assert env_cfg_override["reward_config"]["scales"]["feet_double_stance"] == pytest.approx(-1.0)
    assert env_cfg_override["reward_config"]["min_forward_speed_for_gait_reward"] == pytest.approx(
        0.05
    )
    assert "motrix_max_iterations" not in env_cfg_override
    assert env_cfg_override["control_config"]["action_scale"] == pytest.approx(0.5)
    assert env_cfg_override["commands"]["vel_limit"] == [[0.4, 0.0, 0.0], [0.7, 0.0, 0.0]]
    assert env_cfg_override["gait_phase_init_mode"] == "offset_phase"
    assert env_cfg_override["reset_base_qvel_limit"] == pytest.approx(0.05)


def test_build_ppo_env_cfg_override_carries_motrix_max_iterations_override(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=g1_walk_flat/motrix", "+env.motrix_max_iterations=9"])

    env_cfg_override = mod.build_ppo_env_cfg_override(cfg)

    assert env_cfg_override["motrix_max_iterations"] == 9


def test_build_ppo_env_cfg_override_carries_post_step_forward_sensor_override(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    for value in (True, False):
        cfg = _ppo_cfg(["task=g1_walk_flat/mujoco", f"env.post_step_forward_sensor={value}"])

        env_cfg_override = mod.build_ppo_env_cfg_override(cfg)

        assert env_cfg_override["post_step_forward_sensor"] is value


def test_offpolicy_g1_walk_flat_motrix_env_cfg_override_has_domain_rand():
    cfg = _offpolicy_cfg(["algo=sac", "task=sac/g1_walk_flat/motrix"])

    env_cfg_override = _offpolicy().build_offpolicy_env_cfg_override("sac", cfg)

    assert env_cfg_override["domain_rand"]["randomize_kp"] is False
    assert env_cfg_override["domain_rand"]["randomize_kd"] is False
    assert env_cfg_override["reward_config"]["scales"]["tracking_lin_vel"] == pytest.approx(2.2)


def test_build_ppo_env_cfg_override_applies_go2_motrix_reward(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=go2_joystick_flat/motrix"])

    env_cfg_override = mod.build_ppo_env_cfg_override(cfg)

    assert cfg.reward.scales.tracking_lin_vel == pytest.approx(1.0)
    assert cfg.algo.num_envs == 1024
    assert env_cfg_override["domain_rand"]["randomize_kp"] is False
    assert env_cfg_override["domain_rand"]["randomize_kd"] is False
    assert env_cfg_override["reward_config"]["scales"]["tracking_lin_vel"] == pytest.approx(1.0)
    assert env_cfg_override["reward_config"]["scales"]["tracking_ang_vel"] == pytest.approx(0.2)


def test_build_ppo_env_cfg_override_allegro_mujoco(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=allegro_inhand/mujoco"])
    ppo_motrix_cfg = _ppo_cfg(["task=allegro_inhand/motrix"])
    appo_cfg = _appo_cfg(["task=allegro_inhand/mujoco"])
    appo_motrix_cfg = _appo_cfg(["task=allegro_inhand/motrix"])

    env_cfg_override = mod.build_ppo_env_cfg_override(cfg)

    assert cfg.training.task_name == "AllegroInhandRotation"
    assert cfg.algo.empirical_normalization is False
    assert cfg.algo.actor.obs_normalization is True
    assert cfg.algo.critic.obs_normalization is True
    assert env_cfg_override["reward_config"]["scales"]["rotate"] == pytest.approx(1.25)
    assert env_cfg_override["reward_config"]["reset_z_threshold"] == pytest.approx(0.125)
    assert env_cfg_override["gen_grasp"] is False
    assert env_cfg_override["max_episode_seconds"] == pytest.approx(20.0)
    assert env_cfg_override["grasp_cache_path"] == "caches/allegro_grasp_50k.npy"
    assert env_cfg_override["domain_rand"]["randomize_base_mass"] is False
    assert env_cfg_override["domain_rand"]["random_com"] is False
    assert env_cfg_override["domain_rand"]["randomize_gravity"] is False
    assert env_cfg_override["domain_rand"]["push_robots"] is False
    assert env_cfg_override["domain_rand"]["joint_noise"] == pytest.approx(0.0)
    assert env_cfg_override["domain_rand"]["ball_vel_noise"] == pytest.approx(0.0)
    assert env_cfg_override["domain_rand"]["ball_z_offset"] == pytest.approx(0.0)
    assert appo_cfg.algo.steps_per_env == cfg.algo.num_steps_per_env
    assert list(appo_cfg.algo.actor.hidden_dims) == list(cfg.algo.actor.hidden_dims)
    assert appo_cfg.algo.actor.activation == cfg.algo.actor.activation
    assert appo_cfg.algo.actor.obs_normalization is True
    assert list(appo_cfg.algo.critic.hidden_dims) == list(cfg.algo.critic.hidden_dims)
    assert appo_cfg.algo.critic.activation == cfg.algo.critic.activation
    assert appo_cfg.algo.critic.obs_normalization is True
    assert appo_cfg.algo.algorithm.value_loss_coef == pytest.approx(
        cfg.algo.algorithm.value_loss_coef
    )
    assert appo_cfg.algo.algorithm.entropy_coef == pytest.approx(cfg.algo.algorithm.entropy_coef)
    assert appo_cfg.algo.algorithm.num_learning_epochs == cfg.algo.algorithm.num_learning_epochs
    assert appo_cfg.algo.algorithm.num_mini_batches == cfg.algo.algorithm.num_mini_batches
    assert appo_cfg.algo.algorithm.clip_param == pytest.approx(cfg.algo.algorithm.clip_param)
    assert appo_cfg.algo.algorithm.gamma == pytest.approx(cfg.algo.algorithm.gamma)
    assert appo_cfg.algo.algorithm.lam == pytest.approx(cfg.algo.algorithm.lam)
    assert appo_cfg.algo.algorithm.max_grad_norm == pytest.approx(cfg.algo.algorithm.max_grad_norm)
    assert (
        appo_cfg.algo.algorithm.use_clipped_value_loss is cfg.algo.algorithm.use_clipped_value_loss
    )
    assert appo_cfg.algo.algorithm.schedule == cfg.algo.algorithm.schedule
    assert appo_motrix_cfg.training.task_name == appo_cfg.training.task_name
    assert appo_motrix_cfg.training.sim_backend == ppo_motrix_cfg.training.sim_backend
    assert appo_motrix_cfg.algo.actor.obs_normalization is True
    assert appo_motrix_cfg.algo.critic.obs_normalization is True
    assert appo_motrix_cfg.reward.scales.rotate == pytest.approx(
        ppo_motrix_cfg.reward.scales.rotate
    )
    assert appo_motrix_cfg.env.gen_grasp is ppo_motrix_cfg.env.gen_grasp
    assert appo_motrix_cfg.env.domain_rand.randomize_base_mass is False
    assert appo_motrix_cfg.env.domain_rand.random_com is False
    assert appo_motrix_cfg.env.domain_rand.randomize_gravity is False
    assert appo_motrix_cfg.env.domain_rand.push_robots is False


def test_build_ppo_env_cfg_override_allegro_grasp_mujoco(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=allegro_inhand_grasp/mujoco"])

    env_cfg_override = mod.build_ppo_env_cfg_override(cfg)

    assert cfg.training.task_name == "AllegroInhandRotationGrasp"
    assert cfg.algo.empirical_normalization is False
    assert cfg.algo.actor.obs_normalization is True
    assert cfg.algo.critic.obs_normalization is True
    assert env_cfg_override["reward_config"]["scales"]["rotate"] == pytest.approx(0.0)
    assert env_cfg_override["gen_grasp"] is True
    assert env_cfg_override["grasp_collection_target"] == 50000
    assert env_cfg_override["grasp_quality_check"] is True
    assert env_cfg_override["domain_rand"]["randomize_base_mass"] is False
    assert env_cfg_override["domain_rand"]["random_com"] is False
    assert env_cfg_override["domain_rand"]["randomize_gravity"] is False
    assert env_cfg_override["domain_rand"]["push_robots"] is False
    assert env_cfg_override["domain_rand"]["ball_vel_noise"] == pytest.approx(0.0)
    assert env_cfg_override["domain_rand"]["joint_noise"] == pytest.approx(0.25)


def test_build_ppo_env_cfg_override_allegro_grasp_cli_override_wins(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(
        [
            "task=allegro_inhand_grasp/mujoco",
            "algo.max_iterations=1",
            "env.grasp_collection_target=128",
            "reward.scales.rotate=0.3",
        ]
    )

    env_cfg_override = mod.build_ppo_env_cfg_override(cfg)

    assert cfg.algo.max_iterations == 1
    assert env_cfg_override["grasp_collection_target"] == 128
    assert env_cfg_override["reward_config"]["scales"]["rotate"] == pytest.approx(0.3)
    assert env_cfg_override["gen_grasp"] is True


def test_build_ppo_env_cfg_override_sharpa_grasp_cli_override_wins(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(
        [
            "task=sharpa_inhand_grasp/mujoco",
            "algo.max_iterations=1",
            "env.grasp_collection_target=128",
            "reward.scales.rotate=0.3",
        ]
    )

    env_cfg_override = mod.build_ppo_env_cfg_override(cfg)

    assert cfg.algo.max_iterations == 1
    assert env_cfg_override["grasp_collection_target"] == 128
    assert env_cfg_override["reward_config"]["scales"]["rotate"] == pytest.approx(0.3)


def test_build_ppo_env_cfg_override_sharpa_grasp_motrix_owner(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(
        [
            "task=sharpa_inhand_grasp/motrix",
            "algo.max_iterations=1",
            "env.grasp_collection_target=128",
        ]
    )

    env_cfg_override = mod.build_ppo_env_cfg_override(cfg)

    assert cfg.training.task_name == "SharpaInhandRotationGrasp"
    assert cfg.training.sim_backend == "motrix"
    assert env_cfg_override["grasp_collection_target"] == 128
    assert env_cfg_override["domain_rand"]["scale_list"] == [0.8]


def test_ppo_cli_algo_override_wins_over_base(
    monkeypatch: pytest.MonkeyPatch,
):
    """CLI override takes precedence over base task algo values via Hydra compose."""
    cfg = _ppo_cfg(["task=g1_walk_flat/motrix", "algo.max_iterations=1"])

    assert cfg.algo.max_iterations == 1
    # Other base values remain intact
    assert cfg.algo.empirical_normalization is True


def test_g1_motion_tracking_ppo_motrix_prefers_backend_specific_reward(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=g1_motion_tracking/motrix"])

    assert cfg.reward.scales.motion_body_pos == pytest.approx(1.0)
    cfg.reward.scales.motion_body_pos = 1.25

    env_cfg_override = mod.build_ppo_env_cfg_override(cfg)

    assert env_cfg_override["reward_config"]["scales"]["motion_body_pos"] == pytest.approx(1.25)


def test_build_ppo_play_env_cfg_override_applies_g1_motion_tracking_play_profile(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=g1_motion_tracking/motrix", "training.play_only=true"])
    assert cfg.training.play_env_num == 16

    monkeypatch.setattr(
        mod,
        "materialize_scene_visual_override",
        lambda source_model_file, **kwargs: "/tmp/g1_motion_tracking_play_scene.xml",
    )

    env_cfg_override = mod.build_ppo_play_env_cfg_override(cfg)

    assert cfg.training.play_env_num == 16
    assert env_cfg_override["render_spacing"] == pytest.approx(2.5)
    assert env_cfg_override["scene"].model_file == "/tmp/g1_motion_tracking_play_scene.xml"
    assert env_cfg_override["reward_config"]["scales"]["motion_body_pos"] == pytest.approx(1.0)


def test_build_ppo_play_env_cfg_override_respects_cli_play_env_override(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(
        [
            "task=g1_motion_tracking/motrix",
            "training.play_only=true",
            "training.play_env_num=32",
        ]
    )
    assert cfg.training.play_env_num == 32
    monkeypatch.setattr(
        mod,
        "materialize_scene_visual_override",
        lambda source_model_file, **kwargs: "/tmp/g1_motion_tracking_play_scene.xml",
    )

    env_cfg_override = mod.build_ppo_play_env_cfg_override(cfg)

    assert cfg.training.play_env_num == 32
    assert env_cfg_override["render_spacing"] == pytest.approx(2.5)


def test_build_ppo_play_env_cfg_override_resolves_relative_ground_texture(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=g1_motion_tracking/motrix", "training.play_only=true"])
    cfg.play_profile.scene.ground_texture_file = "src/unilab/assets/robots/g1/textures/floor.png"

    captured = {}

    def _fake_materialize(source_model_file, **kwargs):
        captured["source_model_file"] = source_model_file
        captured.update(kwargs)
        return "/tmp/g1_motion_tracking_play_scene.xml"

    monkeypatch.setattr(mod, "materialize_scene_visual_override", _fake_materialize)

    mod.build_ppo_play_env_cfg_override(cfg)

    assert captured["ground_texture_file"] == str(
        mod.ROOT_DIR / "src/unilab/assets/robots/g1/textures/floor.png"
    )


def test_go2_arm_manip_loco_motrix_eval_uses_visual_floor(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=go2_arm_manip_loco/motrix", "training.play_only=true"])

    captured = {}

    def _fake_materialize(source_model_file, **kwargs):
        captured["source_model_file"] = source_model_file
        captured.update(kwargs)
        return "/tmp/go2_arm_manip_loco_play_scene.xml"

    monkeypatch.setattr(mod, "materialize_scene_visual_override", _fake_materialize)

    env_cfg_override = mod.build_ppo_play_env_cfg_override(cfg)

    assert captured["source_model_file"] == str(
        mod.ROOT_DIR / "src/unilab/assets/robots/go2_arm/scene_flat.xml"
    )
    assert captured["ground_texture_file"] == str(
        mod.ROOT_DIR / "src/unilab/assets/robots/g1/textures/floor.png"
    )
    assert captured["skybox_rgb1"] == [0.90, 0.90, 0.91]
    assert captured["skybox_rgb2"] == [0.68, 0.68, 0.70]
    assert captured["ground_texrepeat"] == [0.25, 0.25]
    assert env_cfg_override["scene"].model_file == "/tmp/go2_arm_manip_loco_play_scene.xml"


def test_run_motrix_rsl_play_loop_uses_render_spacing_and_offset_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    import numpy as np
    import torch
    from tensordict import TensorDict

    mod = _train_rsl_rl(monkeypatch)

    class FakePolicy:
        def __call__(self, obs):
            batch = obs.batch_size[0]
            return torch.zeros((batch, 3), dtype=torch.float32)

    class FakeBackend:
        def __init__(self):
            self.init_renderer_calls = []
            self.render_calls = 0

        def init_renderer(self, spacing=1.0, offset_mode="grid"):
            self.init_renderer_calls.append((spacing, offset_mode))

        def render(self):
            self.render_calls += 1

    class FakeEnv:
        def __init__(self):
            self._renderer = FakeBackend()
            self.cfg = type("Cfg", (), {"render_spacing": 2.5, "render_offset_mode": "zero"})()

        def init_play_renderer(self, render_spacing=None, render_offset_mode=None):
            offset_mode = "grid" if render_offset_mode is None else render_offset_mode
            if render_spacing is None:
                self._renderer.init_renderer(offset_mode=offset_mode)
            else:
                self._renderer.init_renderer(render_spacing, offset_mode=offset_mode)

        def render_play_frame(self):
            self._renderer.render()

        def run_playback(self, **kwargs):
            kwargs.pop("frame_state_getter", None)
            kwargs.setdefault("output_video", None)
            kwargs.setdefault("camera_kwargs", None)
            return run_motrix_playback(
                backend=self._renderer,
                env=self,
                headless=False if kwargs.get("headless") is None else bool(kwargs["headless"]),
                record_video=(
                    bool(kwargs["record_video"])
                    if kwargs.get("record_video") is not None
                    else kwargs.get("output_video") is not None
                ),
                **{k: v for k, v in kwargs.items() if k not in {"headless", "record_video"}},
            )

    class FakeWrapper:
        def __init__(self):
            self.env = FakeEnv()
            self.reset_calls = 0
            self.step_calls = 0

        def reset(self):
            self.reset_calls += 1
            return TensorDict({"policy": torch.ones((2, 5), dtype=torch.float32)}, batch_size=2), {}

        def step(self, actions):
            self.step_calls += 1
            return (
                TensorDict({"policy": torch.ones((2, 5), dtype=torch.float32)}, batch_size=2),
                torch.zeros((2,), dtype=torch.float32),
                torch.zeros((2,), dtype=torch.bool),
                {},
            )

    wrapped_env = FakeWrapper()

    mod.run_motrix_rsl_play_loop(
        wrapped_env=wrapped_env,
        policy=FakePolicy(),
        render_spacing=2.5,
        render_offset_mode="zero",
        num_steps=3,
    )

    assert wrapped_env.reset_calls == 1
    assert wrapped_env.step_calls == 3
    assert wrapped_env.env._renderer.init_renderer_calls == [(2.5, "zero")]
    assert wrapped_env.env._renderer.render_calls == 3


def test_g1_motion_tracking_appo_reward_extraction_prefers_backend_specific_reward():
    from unilab.training.reward import extract_reward_config

    cfg = _appo_cfg(["task=g1_motion_tracking/motrix"])

    assert cfg.reward.scales.motion_body_pos == pytest.approx(1.0)
    cfg.reward.scales.motion_body_pos = 1.5

    env_cfg_override = extract_reward_config(cfg)

    assert env_cfg_override["reward_config"]["scales"]["motion_body_pos"] == pytest.approx(1.5)


def test_g1_motion_tracking_ppo_task_exposes_final_reward():
    cfg = _ppo_cfg(["task=g1_motion_tracking/motrix"])

    assert cfg.reward.scales.motion_body_pos == pytest.approx(1.0)


def test_g1_motion_tracking_appo_task_exposes_final_reward():
    cfg = _appo_cfg(["task=g1_motion_tracking/motrix"])

    assert cfg.reward.scales.motion_body_pos == pytest.approx(1.0)


def test_sharpa_appo_motrix_owner_uses_backend_specific_overrides():
    cfg = _appo_cfg(["task=sharpa_inhand/motrix"])

    assert cfg.training.task_name == "SharpaInhandRotation"
    assert cfg.training.sim_backend == "motrix"
    assert cfg.algo.num_envs == 2048
    assert cfg.env.sim_dt == pytest.approx(0.01)
    assert cfg.env.domain_rand.randomize_gravity is True
    assert cfg.env.domain_rand.randomize_gravity_direction is False
    assert cfg.env.domain_rand.randomize_pd_gains is True


# ---------------------------------------------------------------------------
# train_appo.py — motrix runner / play helpers
# ---------------------------------------------------------------------------


def test_build_appo_runner_kwargs_forwards_sim_backend():
    mod = _train_appo()
    cfg = _appo_cfg(["task=g1_motion_tracking/motrix"])

    runner_kwargs = mod.build_appo_runner_kwargs(
        cfg,
        env_cfg_override={"reward_config": {"scales": {}}},
        collector_device="cpu",
    )

    assert runner_kwargs["env_name"] == "G1MotionTracking"
    assert runner_kwargs["sim_backend"] == "motrix"
    assert runner_kwargs["collector_device"] == "cpu"
    assert runner_kwargs["num_envs"] == cfg.algo.num_envs
    assert runner_kwargs["steps_per_env"] == cfg.algo.steps_per_env
    assert runner_kwargs["env_cfg_overrides"]["reward_config"]["scales"] == {}


def test_run_motrix_play_loop_runs_without_physics_state():
    import numpy as np
    import torch

    mod = _train_appo()

    class FakeActor:
        def __call__(self, td):
            batch = td.batch_size[0]
            return torch.zeros((batch, 3), dtype=torch.float32)

    class FakeBackend:
        def __init__(self):
            self.init_renderer_calls = 0
            self.render_calls = 0

        def init_renderer(self, spacing=1.0, offset_mode="grid", **kwargs):
            del spacing, offset_mode, kwargs
            self.init_renderer_calls += 1

        def render(self):
            self.render_calls += 1

    class FakeState:
        def __init__(self):
            self.obs = {"obs": np.ones((2, 5), dtype=np.float32)}

    class FakeEnv:
        def __init__(self):
            self.state = None
            self._renderer = FakeBackend()
            self.init_state_calls = 0
            self.reset_calls = 0
            self.step_calls = 0

        def init_state(self):
            self.init_state_calls += 1
            self.state = object()

        def reset(self, env_indices):
            self.reset_calls += 1
            assert env_indices.shape == (2,)
            return {"obs": np.ones((2, 5), dtype=np.float32)}, {}

        def step(self, actions):
            self.step_calls += 1
            assert actions.shape == (2, 3)
            return FakeState()

        def init_play_renderer(self, render_spacing=None, render_offset_mode=None):
            del render_spacing, render_offset_mode
            self._renderer.init_renderer()

        def render_play_frame(self):
            self._renderer.render()

        def run_playback(self, **kwargs):
            kwargs.pop("frame_state_getter", None)
            kwargs.setdefault("output_video", None)
            kwargs.setdefault("render_spacing", None)
            kwargs.setdefault("render_offset_mode", None)
            kwargs.setdefault("camera_kwargs", None)
            return run_motrix_playback(
                backend=self._renderer,
                env=self,
                headless=False if kwargs.get("headless") is None else bool(kwargs["headless"]),
                record_video=(
                    bool(kwargs["record_video"])
                    if kwargs.get("record_video") is not None
                    else kwargs.get("output_video") is not None
                ),
                **{k: v for k, v in kwargs.items() if k not in {"headless", "record_video"}},
            )

    env = FakeEnv()

    mod.run_motrix_play_loop(
        env=env,
        actor=FakeActor(),
        device="cpu",
        play_env_num=2,
        num_steps=3,
    )

    assert env.init_state_calls == 1
    assert env.reset_calls == 1
    assert env.step_calls == 3
    assert env._renderer.init_renderer_calls == 1
    assert env._renderer.render_calls == 3


def test_resolve_appo_checkpoint_path_prefers_latest_model_in_explicit_dir(tmp_path):
    mod = _train_appo()
    run_dir = tmp_path / "logs" / "appo" / "MyTask" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "model_1.pt").write_bytes(b"")
    (run_dir / "model_7.pt").write_bytes(b"")

    checkpoint_path, checkpoint_dir = mod.resolve_appo_checkpoint_path(
        base_log_dir=tmp_path / "logs" / "appo" / "MyTask",
        load_run=str(run_dir),
    )

    assert checkpoint_path is not None
    assert checkpoint_path.endswith("model_7.pt")
    assert checkpoint_dir == str(run_dir)


# ---------------------------------------------------------------------------
# train_offpolicy.py — default_device()
# ---------------------------------------------------------------------------


def _offpolicy():
    return _load_script("train_offpolicy")


def test_offpolicy_default_device_preferred_cpu():
    mock_torch = MagicMock()
    assert _offpolicy().default_device(mock_torch, preferred="cpu") == "cpu"


def test_offpolicy_default_device_preferred_cuda():
    mock_torch = MagicMock()
    assert _offpolicy().default_device(mock_torch, preferred="cuda") == "cuda"


def test_offpolicy_default_device_cuda_available():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    assert _offpolicy().default_device(mock_torch) == "cuda"


def test_offpolicy_default_device_mps_fallback():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.xpu.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = True
    assert _offpolicy().default_device(mock_torch) == "mps"


def test_offpolicy_default_device_xpu_before_mps():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.xpu.is_available.return_value = True
    mock_torch.backends.mps.is_available.return_value = True
    assert _offpolicy().default_device(mock_torch) == "xpu"


def test_offpolicy_default_device_cpu_fallback():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.xpu.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = False
    assert _offpolicy().default_device(mock_torch) == "cpu"


def test_offpolicy_enable_faulthandler_respects_disable_env(monkeypatch: pytest.MonkeyPatch):
    mod = _offpolicy()
    fake_faulthandler = types.SimpleNamespace(
        is_enabled=lambda: False,
        enable=MagicMock(),
    )
    monkeypatch.setitem(sys.modules, "faulthandler", fake_faulthandler)
    monkeypatch.setenv("UNILAB_FAULTHANDLER", "0")

    mod.enable_faulthandler()

    fake_faulthandler.enable.assert_not_called()


def test_offpolicy_enable_faulthandler_default_enables(monkeypatch: pytest.MonkeyPatch):
    mod = _offpolicy()
    fake_faulthandler = types.SimpleNamespace(
        is_enabled=lambda: False,
        enable=MagicMock(),
    )
    monkeypatch.setitem(sys.modules, "faulthandler", fake_faulthandler)
    monkeypatch.delenv("UNILAB_FAULTHANDLER", raising=False)

    mod.enable_faulthandler()

    fake_faulthandler.enable.assert_called_once_with(all_threads=True)


def test_offpolicy_build_failure_summary_preserves_failed_status():
    mod = _offpolicy()
    exc = RuntimeError("collector died")

    summary = mod.build_failure_summary(exc, {"status": "collector_died", "total_env_steps": 12})

    assert summary["status"] == "collector_died"
    assert summary["total_env_steps"] == 12
    assert summary["error_type"] == "RuntimeError"
    assert summary["error"] == "collector died"


def test_offpolicy_configured_actor_warm_start_calls_owner_before_training(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    mod = _offpolicy()
    checkpoint = tmp_path / "legacy.pt"
    checkpoint.write_bytes(b"fixture")
    cfg = _offpolicy_cfg(
        [
            "task=sac/g1_stand_height/mujoco",
            f"algo.actor_warm_start_checkpoint={checkpoint}",
        ]
    )
    learner = object()
    captured: dict[str, Any] = {}

    import unilab.algos.torch.offpolicy.checkpoint_adapter as adapter_module

    def fake_load(target_learner, source_path):
        captured["learner"] = target_learner
        captured["source_path"] = source_path
        return {
            "adapter_id": adapter_module.G1_HEIGHT_ACTOR_ADAPTER_ID,
            "parent_checkpoint_sha256": "a" * 64,
        }

    monkeypatch.setattr(adapter_module, "load_g1_height_actor_warm_start", fake_load)

    metadata = mod.apply_configured_actor_warm_start(
        "sac",
        cfg,
        types.SimpleNamespace(learner=learner),
    )

    assert captured == {"learner": learner, "source_path": str(checkpoint)}
    assert metadata is not None
    assert metadata["adapter_id"] == adapter_module.G1_HEIGHT_ACTOR_ADAPTER_ID


def test_offpolicy_actor_warm_start_is_noop_without_checkpoint():
    mod = _offpolicy()
    cfg = _offpolicy_cfg(["task=sac/g1_stand_height/mujoco"])

    assert mod.apply_configured_actor_warm_start("sac", cfg, object()) is None


def test_offpolicy_configured_actor_continuation_dispatches_strict_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    mod = _offpolicy()
    checkpoint = tmp_path / "stage1.pt"
    checkpoint.write_bytes(b"fixture")
    import unilab.algos.torch.offpolicy.checkpoint_adapter as adapter_module

    cfg = _offpolicy_cfg(
        [
            "task=sac/g1_stand_height/mujoco",
            f"algo.actor_warm_start_checkpoint={checkpoint}",
            "algo.actor_warm_start_adapter="
            f"{adapter_module.G1_HEIGHT_ACTOR_CONTINUATION_ADAPTER_ID}",
        ]
    )
    learner = object()
    captured = {}

    def fake_load(target_learner, source_path):
        captured.update(learner=target_learner, source_path=source_path)
        return {
            "adapter_id": adapter_module.G1_HEIGHT_ACTOR_CONTINUATION_ADAPTER_ID,
            "parent_checkpoint_sha256": "b" * 64,
        }

    monkeypatch.setattr(
        adapter_module,
        "load_g1_height_actor_continuation_warm_start",
        fake_load,
    )
    metadata = mod.apply_configured_actor_warm_start(
        "sac", cfg, types.SimpleNamespace(learner=learner)
    )

    assert captured == {"learner": learner, "source_path": str(checkpoint)}
    assert metadata["adapter_id"] == adapter_module.G1_HEIGHT_ACTOR_CONTINUATION_ADAPTER_ID


def test_offpolicy_main_failure_summary_and_skips_playback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import unilab.training.offpolicy.application as mod

    cfg = _offpolicy_cfg(
        [
            f"training.log_dir={tmp_path}",
            "training.no_play=false",
            "training.play_render_mode=record",
        ]
    )
    captured: dict[str, Any] = {"summaries": []}

    class FakeTracker:
        def __init__(self, **kwargs):
            captured["tracker_kwargs"] = kwargs

        def start(self):
            captured["tracker_started"] = True

        def update_summary(self, summary):
            captured["summaries"].append(summary)

        def log_video(self, path):
            captured["video"] = path

        def finish(self):
            captured["tracker_finished"] = True

    class FakeRunner:
        last_run_summary = {"status": "collector_died", "total_env_steps": 12}

        def learn(self, **kwargs):
            del kwargs
            raise RuntimeError("collector died")

        def close(self):
            captured["runner_closed"] = True

    monkeypatch.setattr(mod, "enable_faulthandler", lambda: None)
    monkeypatch.setattr(mod, "ensure_registries", lambda: None)
    monkeypatch.setattr(mod, "apply_configured_training_seed", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        mod, "assert_offpolicy_task_choice_matches_algo", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(mod, "ExperimentTracker", FakeTracker)
    monkeypatch.setattr(mod, "build_runner", lambda algo_name, cfg: FakeRunner())
    monkeypatch.setattr(
        mod,
        "play_offpolicy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("playback must not run after training failure")
        ),
    )

    with pytest.raises(RuntimeError, match="collector died"):
        mod.run_offpolicy(cfg)

    assert captured["tracker_started"] is True
    assert captured["tracker_finished"] is True
    assert captured["runner_closed"] is True
    assert len(captured["summaries"]) == 1
    failure_summary = captured["summaries"][0]
    assert failure_summary["status"] == "collector_died"
    assert failure_summary["total_env_steps"] == 12
    assert failure_summary["error_type"] == "RuntimeError"
    assert failure_summary["error"] == "collector died"
    assert "video" not in captured


# ---------------------------------------------------------------------------
# train_offpolicy.py — resolve_checkpoint_path()
# ---------------------------------------------------------------------------


def test_resolve_checkpoint_no_base_dir(tmp_path):
    """load_run='-1' with no log directory → (None, None)."""
    path, path_dir = _offpolicy().resolve_checkpoint_path(tmp_path, "sac", "MyTask", "-1")
    assert path is None
    assert path_dir is None


def test_resolve_checkpoint_explicit_existing_file(tmp_path):
    """load_run = absolute path to existing .pt → returns that path."""
    model_file = tmp_path / "model_100.pt"
    model_file.write_bytes(b"")
    path, path_dir = _offpolicy().resolve_checkpoint_path(
        tmp_path, "sac", "MyTask", str(model_file)
    )
    assert path == str(model_file)
    assert path_dir == str(tmp_path)


def test_resolve_checkpoint_latest_picks_highest_iter(tmp_path):
    """load_run='-1' picks model with numerically highest iteration."""
    task_dir = tmp_path / "logs" / "sac" / "MyTask" / "run1"
    task_dir.mkdir(parents=True)
    (task_dir / "model_10.pt").write_bytes(b"")
    (task_dir / "model_50.pt").write_bytes(b"")
    (task_dir / "model_100.pt").write_bytes(b"")

    path, path_dir = _offpolicy().resolve_checkpoint_path(tmp_path, "sac", "MyTask", "-1")
    assert path is not None
    assert "model_100.pt" in path


def test_resolve_checkpoint_accepts_integer_latest_run(tmp_path):
    """load_run=-1 from Hydra CLI picks the latest model."""
    task_dir = tmp_path / "logs" / "sac" / "MyTask" / "run1"
    task_dir.mkdir(parents=True)
    (task_dir / "model_10.pt").write_bytes(b"")
    (task_dir / "model_50.pt").write_bytes(b"")

    path, path_dir = _offpolicy().resolve_checkpoint_path(tmp_path, "sac", "MyTask", -1)

    assert path is not None
    assert "model_50.pt" in path
    assert path_dir == str(task_dir)


def test_resolve_checkpoint_explicit_run_name(tmp_path):
    """load_run = run-directory name under the log root."""
    task_dir = tmp_path / "logs" / "sac" / "MyTask" / "myrun"
    task_dir.mkdir(parents=True)
    (task_dir / "model_5.pt").write_bytes(b"")

    path, path_dir = _offpolicy().resolve_checkpoint_path(tmp_path, "sac", "MyTask", "myrun")
    assert path is not None
    assert "model_5.pt" in path
    assert path_dir == str(task_dir)


def test_resolve_checkpoint_nonexistent_explicit_path(tmp_path):
    """load_run points to a path that doesn't exist → (None, None)."""
    path, path_dir = _offpolicy().resolve_checkpoint_path(
        tmp_path, "sac", "MyTask", "/nonexistent/model.pt"
    )
    assert path is None
    assert path_dir is None


def test_resolve_checkpoint_empty_run_dir(tmp_path):
    """Run directory exists but has no model_*.pt → (None, None)."""
    task_dir = tmp_path / "logs" / "sac" / "MyTask" / "run1"
    task_dir.mkdir(parents=True)

    path, path_dir = _offpolicy().resolve_checkpoint_path(tmp_path, "sac", "MyTask", "-1")
    assert path is None


def test_offpolicy_extract_reset_obs_handles_two_tuple():
    obs = {"obs": "value"}

    result = _offpolicy().extract_reset_obs((obs, {"info": 1}))

    assert result is obs


def test_offpolicy_extract_reset_obs_rejects_three_tuple():
    obs = {"obs": "value"}

    with pytest.raises(ValueError, match="Unexpected env.reset return format"):
        _offpolicy().extract_reset_obs(("ignored", obs, {"info": 1}))


def test_offpolicy_resolve_play_obs_dim_ignores_critic():
    obs_dim = _offpolicy().resolve_play_obs_dim({"obs": 98, "critic": 101})

    assert obs_dim == 98


def test_offpolicy_extract_play_obs_uses_obs_group_only():
    import numpy as np

    obs = {
        "obs": np.ones((2, 98), dtype=np.float32),
        "critic": np.full((2, 101), 2.0, dtype=np.float32),
    }

    play_obs = _offpolicy().extract_play_obs(obs)

    assert play_obs.shape == (2, 98)
    assert np.allclose(play_obs, 1.0)


def test_offpolicy_play_actor_spec_uses_hora_sac_runtime():
    cfg = _offpolicy_cfg(
        [
            "algo=sac",
            "task=sac/sharpa_inhand/mujoco_hora",
        ]
    )

    actor_algo_type, actor_kwargs = _offpolicy().resolve_play_actor_spec(
        "sac",
        cfg,
        obs_dim=4,
        critic_obs_dim=6,
    )

    assert actor_algo_type == "hora_sac"
    assert actor_kwargs["priv_info_dim"] == 2


def test_offpolicy_play_actor_spec_keeps_standard_sac_and_flashsac():
    mod = _offpolicy()
    sac_cfg = _offpolicy_cfg(["algo=sac", "task=sac/g1_walk_flat/mujoco"])
    flashsac_cfg = _offpolicy_cfg(["algo=flashsac", "task=flashsac/g1_walk_flat/mujoco"])

    sac_algo_type, sac_kwargs = mod.resolve_play_actor_spec(
        "sac",
        sac_cfg,
        obs_dim=98,
        critic_obs_dim=101,
    )
    flash_algo_type, flash_kwargs = mod.resolve_play_actor_spec(
        "flashsac",
        flashsac_cfg,
        obs_dim=98,
        critic_obs_dim=101,
    )

    assert (sac_algo_type, sac_kwargs) == ("sac", {})
    assert (flash_algo_type, flash_kwargs) == ("flashsac", {})


def test_play_offpolicy_can_skip_onnx_export_and_still_record_video(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    import torch

    import unilab.training.offpolicy.playback as mod

    cfg = _offpolicy_cfg(
        [
            "algo=sac",
            "task=sac/g1_walk_flat/mujoco",
            "training.play_only=true",
            "training.play_render_mode=record",
            "training.export_onnx=false",
        ]
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "model_5000.pt"
    torch.save({"actor": {}}, checkpoint)

    captured: dict[str, Any] = {}

    class FakeActor:
        def eval(self):
            return self

        def load_state_dict(self, state_dict):
            captured["loaded_state_dict"] = state_dict

        def as_export_module(self):
            raise AssertionError("ONNX export should be skipped when training.export_onnx=false")

        def explore(self, obs, deterministic=True):
            captured["deterministic"] = deterministic
            return torch.zeros((obs.shape[0], 2), dtype=obs.dtype, device=obs.device)

    class FakeEnv:
        def __init__(self):
            self.obs_groups_spec = {"obs": 4}
            self.action_space = type("ActionSpace", (), {"shape": (2,)})()
            self.state = None

        def init_state(self):
            self.state = type(
                "State",
                (),
                {"obs": {"obs": np.zeros((cfg.training.play_env_num, 4), dtype=np.float32)}},
            )()

        def reset(self, env_ids):
            batch = len(env_ids)
            return ({"obs": np.zeros((batch, 4), dtype=np.float32)}, {})

        def step(self, actions):
            batch = actions.shape[0]
            self.state = type(
                "State",
                (),
                {"obs": {"obs": np.ones((batch, 4), dtype=np.float32)}},
            )()
            captured["actions_shape"] = actions.shape
            return self.state

        def run_playback_mode(self, **kwargs):
            captured["play_render_mode"] = kwargs["play_render_mode"]
            captured["output_video"] = kwargs["output_video"]
            init_obs = kwargs["initialize"]()
            captured["init_obs_shape"] = init_obs.shape
            next_obs = kwargs["step"](init_obs)
            captured["next_obs_shape"] = next_obs.shape
            return str(kwargs["output_video"])

    monkeypatch.setattr(mod, "build_offpolicy_env_cfg_override", lambda algo_name, cfg: {})
    monkeypatch.setattr(mod, "default_device", lambda torch_module, preferred=None: "cpu")
    monkeypatch.setattr(mod, "create_env", lambda *args, **kwargs: FakeEnv())
    monkeypatch.setattr(mod, "resolve_play_obs_dim", lambda obs_groups_spec: 4)
    monkeypatch.setattr(mod, "extract_play_obs", lambda obs_dict: obs_dict["obs"])
    monkeypatch.setattr(
        mod,
        "resolve_checkpoint_path",
        lambda *args, **kwargs: (str(checkpoint), str(run_dir)),
    )
    monkeypatch.setattr(
        torch.onnx,
        "export",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("torch.onnx.export should not be called when training.export_onnx=false")
        ),
    )

    import unilab.algos.torch.common.actor_factory as actor_factory

    monkeypatch.setattr(actor_factory, "build_actor", lambda *args, **kwargs: FakeActor())

    result = mod.play_offpolicy("sac", cfg)
    out = capsys.readouterr().out

    assert result == str(run_dir / "play_video.mp4")
    assert captured["loaded_state_dict"] == {}
    assert captured["play_render_mode"] == "record"
    assert captured["actions_shape"] == (cfg.training.play_env_num, 2)
    assert captured["init_obs_shape"] == (cfg.training.play_env_num, 4)
    assert captured["next_obs_shape"] == (cfg.training.play_env_num, 4)
    assert captured["deterministic"] is True
    assert "Skipping ONNX export because training.export_onnx=false." in out
    assert not (run_dir / "policy.onnx").exists()


def test_play_offpolicy_uses_hora_sac_actor_and_priv_info(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import torch

    import unilab.training.offpolicy.playback as mod

    cfg = _offpolicy_cfg(
        [
            "algo=sac",
            "task=sac/sharpa_inhand/mujoco_hora",
            "training.play_only=true",
            "training.play_render_mode=record",
            "training.export_onnx=false",
            "training.play_env_num=2",
        ]
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "model_5000.pt"
    torch.save({"actor": {}}, checkpoint)

    captured: dict[str, Any] = {}
    reset_priv = np.array([[4.0, 5.0], [6.0, 7.0]], dtype=np.float32)
    step_priv = np.array([[8.0, 9.0], [10.0, 11.0]], dtype=np.float32)

    class FakeHoraActor:
        def eval(self):
            return self

        def load_state_dict(self, state_dict):
            captured["loaded_state_dict"] = state_dict

        def explore(self, obs, priv_info, deterministic=True):
            captured["obs_shape"] = tuple(obs.shape)
            captured["priv_info"] = priv_info.detach().cpu().numpy()
            captured["deterministic"] = deterministic
            return torch.zeros((obs.shape[0], 2), dtype=obs.dtype, device=obs.device)

    class FakeEnv:
        def __init__(self):
            self.obs_groups_spec = {"obs": 3, "critic": 5}
            self.action_space = type("ActionSpace", (), {"shape": (2,)})()
            self.state = None

        def init_state(self):
            self.state = type(
                "State",
                (),
                {
                    "obs": {
                        "obs": np.zeros((cfg.training.play_env_num, 3), dtype=np.float32),
                        "critic": np.zeros((cfg.training.play_env_num, 5), dtype=np.float32),
                    },
                    "info": {"critic_info": reset_priv},
                },
            )()

        def reset(self, env_ids):
            batch = len(env_ids)
            return (
                {
                    "obs": np.zeros((batch, 3), dtype=np.float32),
                    "critic": np.concatenate(
                        [np.zeros((batch, 3), dtype=np.float32), reset_priv],
                        axis=1,
                    ),
                },
                {"critic_info": reset_priv},
            )

        def step(self, actions):
            batch = actions.shape[0]
            captured["actions_shape"] = actions.shape
            self.state = type(
                "State",
                (),
                {
                    "obs": {
                        "obs": np.ones((batch, 3), dtype=np.float32),
                        "critic": np.concatenate(
                            [np.ones((batch, 3), dtype=np.float32), step_priv],
                            axis=1,
                        ),
                    },
                    "info": {"critic_info": step_priv},
                },
            )()
            return self.state

        def run_playback_mode(self, **kwargs):
            init_obs = kwargs["initialize"]()
            captured["init_obs_shape"] = init_obs.shape
            next_obs = kwargs["step"](init_obs)
            captured["next_obs_shape"] = next_obs.shape
            return str(kwargs["output_video"])

    monkeypatch.setattr(mod, "build_offpolicy_env_cfg_override", lambda algo_name, cfg: {})
    monkeypatch.setattr(mod, "default_device", lambda torch_module, preferred=None: "cpu")
    monkeypatch.setattr(mod, "create_env", lambda *args, **kwargs: FakeEnv())
    monkeypatch.setattr(
        mod,
        "resolve_checkpoint_path",
        lambda *args, **kwargs: (str(checkpoint), str(run_dir)),
    )

    import unilab.algos.torch.common.actor_factory as actor_factory

    def fake_build_actor(algo_type, obs_dim, action_dim, hidden_dim, use_layer_norm, device, **kw):
        captured["build_actor"] = (algo_type, obs_dim, action_dim, kw)
        return FakeHoraActor()

    monkeypatch.setattr(actor_factory, "build_actor", fake_build_actor)

    result = mod.play_offpolicy("sac", cfg)

    assert result == str(run_dir / "play_video.mp4")
    assert captured["build_actor"][0] == "hora_sac"
    assert captured["build_actor"][1:3] == (3, 2)
    assert captured["build_actor"][3]["priv_info_dim"] == 2
    assert captured["loaded_state_dict"] == {}
    assert captured["actions_shape"] == (cfg.training.play_env_num, 2)
    assert captured["init_obs_shape"] == (cfg.training.play_env_num, 3)
    assert captured["next_obs_shape"] == (cfg.training.play_env_num, 3)
    assert captured["obs_shape"] == (cfg.training.play_env_num, 3)
    np.testing.assert_allclose(captured["priv_info"], reset_priv)
    assert captured["deterministic"] is True
    assert not (run_dir / "policy.onnx").exists()


# ---------------------------------------------------------------------------
# train_mlx_ppo.py — get_latest_run() / get_latest_checkpoint()
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not installed")
def test_mlx_get_latest_run_nonexistent_dir(tmp_path):
    mod = _load_script("train_mlx_ppo")
    assert mod.get_latest_run(tmp_path / "nonexistent") is None


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not installed")
def test_mlx_get_latest_run_empty_dir(tmp_path):
    mod = _load_script("train_mlx_ppo")
    assert mod.get_latest_run(tmp_path) is None


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not installed")
def test_mlx_get_latest_run_returns_last_sorted(tmp_path):
    mod = _load_script("train_mlx_ppo")
    (tmp_path / "2024-01-01_mujoco").mkdir()
    (tmp_path / "2024-03-15_mujoco").mkdir()
    (tmp_path / "2024-02-10_mujoco").mkdir()
    result = mod.get_latest_run(tmp_path)
    assert result is not None
    assert result.name == "2024-03-15_mujoco"


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not installed")
def test_mlx_get_latest_checkpoint_nonexistent_dir(tmp_path):
    mod = _load_script("train_mlx_ppo")
    assert mod.get_latest_checkpoint(tmp_path / "no_such_dir") is None


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not installed")
def test_mlx_get_latest_checkpoint_empty_dir(tmp_path):
    mod = _load_script("train_mlx_ppo")
    assert mod.get_latest_checkpoint(tmp_path) is None


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not installed")
def test_mlx_get_latest_checkpoint_picks_highest_iter(tmp_path):
    mod = _load_script("train_mlx_ppo")
    (tmp_path / "model_0.safetensors").write_bytes(b"")
    (tmp_path / "model_50.safetensors").write_bytes(b"")
    (tmp_path / "model_200.safetensors").write_bytes(b"")
    result = mod.get_latest_checkpoint(tmp_path)
    assert result is not None
    assert result.name == "model_200.safetensors"


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not installed")
def test_mlx_get_latest_checkpoint_ignores_non_safetensors(tmp_path):
    """Only .safetensors files count; .pt files must be ignored."""
    mod = _load_script("train_mlx_ppo")
    (tmp_path / "model_999.pt").write_bytes(b"")  # should be ignored
    assert mod.get_latest_checkpoint(tmp_path) is None


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not installed")
def test_mlx_time_limit_bootstrap_values_use_final_observation():
    mod = _load_script("train_mlx_ppo")

    class FakeModel:
        def __init__(self):
            self.last_obs = None

        def value(self, obs):
            self.last_obs = obs
            return mod.mx.sum(obs, axis=1)

    state = type(
        "State",
        (),
        {
            "truncated": np.array([True, False]),
            "final_observation": {
                "obs": np.array([[3.0, 4.0], [9.0, 9.0]], dtype=np.float32),
            },
            "info": {
                "final_observation": {
                    "obs": np.array([[3.0, 4.0], [9.0, 9.0]], dtype=np.float32),
                }
            },
        },
    )()
    model = FakeModel()

    values = mod.get_time_limit_bootstrap_values(state, model, mod.mx.float32)

    if values is None:
        raise AssertionError("expected bootstrap values")
    if model.last_obs is None:
        raise AssertionError("expected model to receive observations")
    np.testing.assert_allclose(np.array(values.tolist()), np.array([7.0, 18.0], dtype=np.float32))
    np.testing.assert_allclose(
        np.array(model.last_obs.tolist()),
        np.array([[3.0, 4.0], [9.0, 9.0]], dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# play_interactive.py — resolve_checkpoint()
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")
def test_play_resolve_checkpoint_nonexistent_run(tmp_path):
    """Passing a non-existent explicit path returns None."""
    mod = _load_script("play_interactive")
    result = mod.resolve_checkpoint("MyTask", str(tmp_path / "no_run"))
    assert result is None


@pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")
def test_play_resolve_checkpoint_dir_with_model(tmp_path):
    """Directory path containing model_*.pt files resolves to the latest."""
    mod = _load_script("play_interactive")
    run_dir = tmp_path / "2024-01-01_mujoco"
    run_dir.mkdir()
    (run_dir / "model_10.pt").write_bytes(b"")
    (run_dir / "model_50.pt").write_bytes(b"")

    result = mod.resolve_checkpoint("MyTask", str(run_dir))
    assert result is not None
    assert "model_50.pt" in result


@pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")
def test_play_resolve_checkpoint_explicit_file(tmp_path):
    """Absolute path to existing .pt file returns that path unchanged."""
    mod = _load_script("play_interactive")
    model_file = tmp_path / "model_99.pt"
    model_file.write_bytes(b"")
    result = mod.resolve_checkpoint("MyTask", str(model_file))
    assert result == str(model_file)


@pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")
def test_play_resolve_checkpoint_empty_dir(tmp_path):
    """Directory with no model_*.pt files returns None."""
    mod = _load_script("play_interactive")
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    result = mod.resolve_checkpoint("MyTask", str(run_dir))
    assert result is None


@pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")
def test_play_resolve_checkpoint_delegates_to_shared_helper(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    mod = _load_script("play_interactive")
    model_path = tmp_path / "resolved" / "model_12.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"")
    captured: dict[str, object] = {}

    def _fake_resolver(root_dir, **kwargs):
        captured["root_dir"] = root_dir
        captured.update(kwargs)
        return model_path, model_path.parent

    monkeypatch.setattr(mod, "resolve_task_checkpoint_path", _fake_resolver)

    result = mod.resolve_checkpoint("MyTask", "-1", checkpoint="12", algo_log_name="custom_ppo")

    assert result == str(model_path)
    assert captured["root_dir"] == mod.ROOT_DIR
    assert captured["task_name"] == "MyTask"
    assert captured["load_run"] == "-1"
    assert captured["algo_log_name"] == "custom_ppo"
    assert captured["checkpoint"] == "12"


# ---------------------------------------------------------------------------
# play_interactive.py — RslRlVecEnvWrapper contract behavior
# ---------------------------------------------------------------------------


def _play_interactive():
    """Load play_interactive.py as a module."""
    return _load_script("play_interactive")


def test_play_wrapper_imports_shared_implementation():
    """Verify play_interactive.py uses shared RslRlVecEnvWrapper."""
    from unilab.training.rsl_rl import RslRlVecEnvWrapper as SharedWrapper

    mod = _play_interactive()
    # The wrapper class in play_interactive should be the shared one
    assert mod.RslRlVecEnvWrapper is SharedWrapper


def test_play_wrapper_uses_current_reset_contract():
    """Verify wrapper reset() uses current (obs, info) contract, not old (_, obs, _)."""
    import numpy as np
    from tensordict import TensorDict

    from unilab.training.rsl_rl import RslRlVecEnvWrapper

    # Create a fake environment that returns (obs, info) tuple
    class FakeEnv:
        def __init__(self):
            self.num_envs = 2
            self.state = type("State", (), {"obs": {"obs": np.ones((2, 5), dtype=np.float32)}})()
            self.cfg = type("Cfg", (), {"max_episode_seconds": 10.0, "ctrl_dt": 0.02})()
            self.observation_space = type("Space", (), {"shape": (5,)})()
            self.action_space = type("Space", (), {"shape": (3,)})()
            self.obs_groups_spec = {"obs": 5}

        def init_state(self):
            pass

        def reset(self, env_indices):
            # Returns current contract: (obs, info)
            return {"obs": np.ones((2, 5), dtype=np.float32)}, {}

    env = FakeEnv()
    wrapper = RslRlVecEnvWrapper(env, device="cpu", policy_obs_mode="flat")

    # Reset should work with current contract
    obs_td, info = wrapper.reset()

    assert isinstance(obs_td, TensorDict)
    assert "policy" in obs_td
    assert "actor" in obs_td
    assert obs_td.batch_size == (2,)


def test_play_wrapper_policy_obs_mode_actor():
    """Verify wrapper supports policy_obs_mode='actor'."""
    import numpy as np

    from unilab.training.rsl_rl import RslRlVecEnvWrapper

    class FakeEnv:
        def __init__(self):
            self.num_envs = 1
            self.state = type("State", (), {"obs": {"obs": np.ones((1, 3), dtype=np.float32)}})()
            self.cfg = type("Cfg", (), {"max_episode_seconds": 10.0, "ctrl_dt": 0.02})()
            self.observation_space = type("Space", (), {"shape": (3,)})()
            self.action_space = type("Space", (), {"shape": (2,)})()
            self.obs_groups_spec = {"obs": 3, "critic": 5}

        def init_state(self):
            pass

        def reset(self, env_indices):
            return {
                "obs": np.ones((1, 3), dtype=np.float32),
                "critic": np.zeros((1, 5), dtype=np.float32),
            }, {}

    env = FakeEnv()

    # Test actor mode - num_obs should match actor obs dim only
    wrapper_actor = RslRlVecEnvWrapper(env, device="cpu", policy_obs_mode="actor")
    assert wrapper_actor.num_obs == 3  # Only "obs" group
    assert wrapper_actor._actor_obs_dim == 3
    assert wrapper_actor._flat_obs_dim == 3

    obs_td, _ = wrapper_actor.reset()
    # In actor mode, policy obs should equal actor obs
    assert obs_td["policy"].shape == (1, 3)
    assert obs_td["actor"].shape == (1, 3)
    assert obs_td["critic"].shape == (1, 5)


def test_play_wrapper_flat_policy_excludes_critic_only_group():
    import numpy as np

    from unilab.training.rsl_rl import RslRlVecEnvWrapper

    class FakeEnv:
        def __init__(self):
            self.num_envs = 1
            self.state = type(
                "State",
                (),
                {
                    "obs": {
                        "obs": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
                        "critic": np.array([[9.0, 9.0, 9.0, 9.0]], dtype=np.float32),
                    }
                },
            )()
            self.cfg = type("Cfg", (), {"max_episode_seconds": 10.0, "ctrl_dt": 0.02})()
            self.observation_space = type("Space", (), {"shape": (7,)})()
            self.action_space = type("Space", (), {"shape": (2,)})()
            self.obs_groups_spec = {"obs": 3, "critic": 4}

        def init_state(self):
            pass

        def reset(self, env_indices):
            return cast(dict[str, np.ndarray], getattr(self.state, "obs")), {}

    wrapper = RslRlVecEnvWrapper(FakeEnv(), device="cpu", policy_obs_mode="flat")
    obs_td, _ = wrapper.reset()

    np.testing.assert_allclose(
        obs_td["policy"].cpu().numpy(),
        np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        obs_td["critic"].cpu().numpy(),
        np.array([[9.0, 9.0, 9.0, 9.0]], dtype=np.float32),
    )
    assert wrapper.num_obs == 3
    assert wrapper.num_privileged_obs == 4


def test_play_wrapper_preserves_hora_priv_info_and_proprio_history():
    import numpy as np

    from unilab.algos.torch.hora.rsl_rl import HoraRslRlVecEnvWrapper

    class FakeEnv:
        def __init__(self):
            self.num_envs = 1
            self.state = type(
                "State",
                (),
                {
                    "obs": {
                        "obs": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
                        "critic": np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32),
                    },
                    "info": {
                        "critic_info": np.array([[4.0, 5.0]], dtype=np.float32),
                        "proprio_hist": np.array(
                            [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]],
                            dtype=np.float32,
                        ),
                    },
                },
            )()
            self.cfg = type("Cfg", (), {"max_episode_seconds": 10.0, "ctrl_dt": 0.02})()
            self.observation_space = type("Space", (), {"shape": (5,)})()
            self.action_space = type("Space", (), {"shape": (2,)})()
            self.obs_groups_spec = {"obs": 3, "critic": 5}

        def init_state(self):
            pass

        def reset_all(self):
            return self.state

    wrapper = HoraRslRlVecEnvWrapper(FakeEnv(), device="cpu", policy_obs_mode="flat")
    obs_td, _ = wrapper.reset()

    np.testing.assert_allclose(
        obs_td["priv_info"].cpu().numpy(),
        np.array([[4.0, 5.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        obs_td["proprio_hist"].cpu().numpy(),
        np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]], dtype=np.float32),
    )


def test_play_wrapper_step_exports_timeout_bootstrap_obs():
    import torch

    from unilab.training.rsl_rl import RslRlVecEnvWrapper

    class FakeEnv:
        def __init__(self):
            self.num_envs = 1
            self.cfg = type("Cfg", (), {"max_episode_seconds": 10.0, "ctrl_dt": 0.02})()
            self.observation_space = type("Space", (), {"shape": (3,)})()
            self.action_space = type("Space", (), {"shape": (2,)})()
            self.obs_groups_spec = {"obs": 3, "critic": 2}
            self.state = type("State", (), {"obs": {"obs": np.zeros((1, 3), dtype=np.float32)}})()

        def init_state(self):
            pass

        def reset(self, env_indices):
            return {"obs": np.zeros((1, 3), dtype=np.float32)}, {}

        def step(self, actions):
            return type(
                "StepState",
                (),
                {
                    "obs": {"obs": np.array([[1.0, 2.0, 3.0]], dtype=np.float32)},
                    "reward": np.array([1.0], dtype=np.float32),
                    "terminated": np.array([False]),
                    "truncated": np.array([True]),
                    "final_observation": {
                        "obs": np.array([[7.0, 8.0, 9.0]], dtype=np.float32),
                        "critic": np.array([[4.0, 5.0]], dtype=np.float32),
                    },
                    "info": {
                        "final_observation": {
                            "obs": np.array([[7.0, 8.0, 9.0]], dtype=np.float32),
                            "critic": np.array([[4.0, 5.0]], dtype=np.float32),
                        }
                    },
                },
            )()

    wrapper = RslRlVecEnvWrapper(FakeEnv(), device="cpu", policy_obs_mode="flat")

    _, _, _, infos = wrapper.step(torch.zeros((1, 2)))

    assert torch.equal(infos["time_outs"], torch.tensor([True]))
    np.testing.assert_allclose(
        infos["time_out_bootstrap_obs"]["policy"].cpu().numpy(),
        np.array([[7.0, 8.0, 9.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        infos["time_out_bootstrap_obs"]["critic"].cpu().numpy(),
        np.array([[4.0, 5.0]], dtype=np.float32),
    )


def test_play_wrapper_timeout_bootstrap_preserves_hora_priv_info():
    import torch

    from unilab.algos.torch.hora.rsl_rl import HoraRslRlVecEnvWrapper

    class FakeEnv:
        def __init__(self):
            self.num_envs = 1
            self.cfg = type("Cfg", (), {"max_episode_seconds": 10.0, "ctrl_dt": 0.02})()
            self.observation_space = type("Space", (), {"shape": (5,)})()
            self.action_space = type("Space", (), {"shape": (2,)})()
            self.obs_groups_spec = {"obs": 3, "critic": 5}
            self.state = type(
                "State",
                (),
                {
                    "obs": {
                        "obs": np.zeros((1, 3), dtype=np.float32),
                        "critic": np.zeros((1, 5), dtype=np.float32),
                    },
                    "info": {
                        "critic_info": np.zeros((1, 2), dtype=np.float32),
                        "proprio_hist": np.zeros((1, 2, 3), dtype=np.float32),
                    },
                },
            )()

        def init_state(self):
            pass

        def reset(self, env_indices):
            del env_indices
            return cast(dict[str, np.ndarray], getattr(self.state, "obs")), cast(
                dict[str, np.ndarray], getattr(self.state, "info")
            )

        def step(self, actions):
            del actions
            return type(
                "StepState",
                (),
                {
                    "obs": {"obs": np.array([[1.0, 2.0, 3.0]], dtype=np.float32)},
                    "reward": np.array([1.0], dtype=np.float32),
                    "terminated": np.array([True]),
                    "truncated": np.array([True]),
                    "final_observation": {
                        "obs": np.array([[7.0, 8.0, 9.0]], dtype=np.float32),
                        "critic": np.array([[7.0, 8.0, 9.0, 4.0, 5.0]], dtype=np.float32),
                    },
                    "info": {
                        "final_observation": {
                            "obs": np.array([[7.0, 8.0, 9.0]], dtype=np.float32),
                            "critic": np.array([[7.0, 8.0, 9.0, 4.0, 5.0]], dtype=np.float32),
                        },
                        "critic_info": np.array([[0.0, 0.0]], dtype=np.float32),
                        "proprio_hist": np.zeros((1, 2, 3), dtype=np.float32),
                    },
                },
            )()

    wrapper = HoraRslRlVecEnvWrapper(FakeEnv(), device="cpu", policy_obs_mode="flat")

    _, _, _, infos = wrapper.step(torch.zeros((1, 2)))

    np.testing.assert_allclose(
        infos["time_out_bootstrap_obs"]["priv_info"].cpu().numpy(),
        np.array([[4.0, 5.0]], dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Issue #168: Unified log directory and load_run resolution
# ---------------------------------------------------------------------------


def test_ppo_hydra_default_algo_log_name():
    """Verify PPO config has algo_log_name in algo section."""
    cfg = _ppo_cfg()
    assert cfg.algo.algo_log_name == "rsl_rl_ppo"


def test_ppo_hydra_load_run_in_algo_not_training():
    """Verify load_run is in algo section, not training section (issue #168)."""
    from omegaconf import OmegaConf

    cfg = _ppo_cfg()
    assert cfg.algo.load_run == "-1"
    # training section should NOT have load_run anymore
    assert "load_run" not in cfg.training or OmegaConf.is_missing(cfg.training, "load_run")


def test_appo_hydra_default_algo_log_name():
    """Verify APPO config has algo_log_name in algo section."""
    cfg = _appo_cfg()
    assert cfg.algo.algo_log_name == "appo"
    assert cfg.algo.load_run == "-1"


def test_offpolicy_sac_hydra_default_algo_log_name():
    """Verify SAC config has algo_log_name in algo section."""
    cfg = _offpolicy_cfg(["algo=sac"])
    assert cfg.algo.algo_log_name == "fast_sac"
    assert cfg.algo.load_run == "-1"


def test_offpolicy_td3_hydra_default_algo_log_name():
    """Verify TD3 config has algo_log_name in algo section."""
    cfg = _offpolicy_cfg(["algo=td3"])
    assert cfg.algo.algo_log_name == "fast_td3"
    assert cfg.algo.load_run == "-1"


def test_offpolicy_flashsac_hydra_algo_log_name():
    cfg = _offpolicy_cfg(["algo=flashsac", "task=flashsac/g1_walk_flat/mujoco"])
    assert cfg.algo.algo_log_name == "flash_sac"
    assert cfg.algo.load_run == "-1"


def test_offpolicy_flashsac_g1_walk_flat_task_composes() -> None:
    cfg = _offpolicy_cfg(["algo=flashsac", "task=flashsac/g1_walk_flat/mujoco"])
    assert cfg.training.task_name == "G1WalkFlat"
    assert cfg.training.sim_backend == "mujoco"


def test_offpolicy_g1_rough_terrain_task_composes() -> None:
    cfg = _offpolicy_cfg(["algo=sac", "task=sac/g1_walk_rough/mujoco"])

    assert cfg.training.task_name == "G1WalkRough"
    assert cfg.training.sim_backend == "mujoco"


def test_offpolicy_flashsac_rejects_multi_gpu():
    cfg = _offpolicy_cfg(
        [
            "algo=flashsac",
            "task=flashsac/g1_walk_flat/mujoco",
            "training.num_gpus=2",
        ]
    )

    with pytest.raises(ValueError, match="FlashSAC does not support training.num_gpus > 1"):
        _offpolicy().build_runner("flashsac", cfg)


def test_offpolicy_sac_multi_gpu_rejected_by_double_buffer():
    cfg = _offpolicy_cfg(
        [
            "algo=sac",
            "task=sac/g1_walk_flat/mujoco",
            "training.num_gpus=2",
            "training.device=cpu",
        ]
    )

    with pytest.raises(ValueError, match="currently single-GPU only"):
        _offpolicy().build_runner("sac", cfg)


def test_offpolicy_sac_multi_gpu_rejects_even_with_explicit_symmetry_disable():
    cfg = _offpolicy_cfg(
        [
            "algo=sac",
            "task=sac/g1_walk_flat/mujoco",
            "training.num_gpus=2",
            "training.device=cpu",
            "algo.use_symmetry=false",
        ]
    )

    with pytest.raises(ValueError, match="currently single-GPU only"):
        _offpolicy().build_runner("sac", cfg)


@pytest.mark.parametrize(
    ("algo", "task"),
    [
        ("flashsac", "sac/g1_walk_flat/mujoco"),
        ("sac", "flashsac/g1_walk_flat/mujoco"),
    ],
)
def test_offpolicy_rejects_algo_task_owner_mismatch(algo: str, task: str):
    cfg = _offpolicy_cfg([f"algo={algo}", f"task={task}"])

    with pytest.raises(ValueError, match="Off-policy algo/task mismatch"):
        _offpolicy().build_runner(algo, cfg)


def test_train_rsl_rl_get_log_root_uses_algo_log_name(monkeypatch: pytest.MonkeyPatch):
    """Verify _get_log_root uses algo.algo_log_name (issue #168)."""
    monkeypatch.delenv("UNILAB_TEST_LOG_ROOT", raising=False)
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg()

    # Override algo_log_name to test
    cfg.algo.algo_log_name = "test_rsl_rl_ppo"

    log_root = mod._get_log_root(cfg)
    assert "logs/test_rsl_rl_ppo" in log_root


def test_train_rsl_rl_play_missing_checkpoint_skips_env_creation_and_prints_context(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    monkeypatch.delenv("UNILAB_TEST_LOG_ROOT", raising=False)
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=go1_joystick_flat/mujoco", "training.play_only=true"])
    cfg.algo.algo_log_name = "custom_ppo"

    original_root = mod.ROOT_DIR
    mod.ROOT_DIR = tmp_path
    try:
        monkeypatch.setattr(
            mod,
            "create_env",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("play_rsl_rl should not create an env before checkpoint resolution")
            ),
        )

        result = mod.play_rsl_rl(cfg, device="cpu")
    finally:
        mod.ROOT_DIR = original_root

    captured = capsys.readouterr().out
    expected_task_log_root = tmp_path / "logs" / "custom_ppo" / cfg.training.task_name

    assert result is None
    assert "Could not resolve a checkpoint for play mode." in captured
    assert "Task log root does not exist." in captured
    assert f"task_log_root={expected_task_log_root}" in captured
    assert "algo.load_run='-1'" in captured


def test_train_rsl_rl_play_reports_missing_requested_checkpoint_in_resolved_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    monkeypatch.delenv("UNILAB_TEST_LOG_ROOT", raising=False)
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=go1_joystick_flat/mujoco", "training.play_only=true"])
    cfg.algo.algo_log_name = "custom_ppo"
    cfg.algo.checkpoint = 12

    run_dir = (
        tmp_path / "logs" / "custom_ppo" / cfg.training.task_name / "2024-01-01_00-00-00_mujoco"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "model_9.pt").write_bytes(b"")

    original_root = mod.ROOT_DIR
    mod.ROOT_DIR = tmp_path
    try:
        monkeypatch.setattr(
            mod,
            "create_env",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("play_rsl_rl should not create an env before checkpoint resolution")
            ),
        )

        result = mod.play_rsl_rl(cfg, device="cpu")
    finally:
        mod.ROOT_DIR = original_root

    captured = capsys.readouterr().out

    assert result is None
    assert "Could not resolve a checkpoint for play mode." in captured
    assert f"resolved_run={run_dir}" in captured
    assert "algo.checkpoint=12" in captured


def test_train_rsl_rl_motrix_auto_play_is_interactive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(
        [
            "task=go2_joystick_rough/motrix",
            "training.play_only=true",
            "training.play_steps=37",
            "training.render_spacing=2.5",
        ]
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "model_37.pt"
    mod.torch.save({"actor_state_dict": {}}, checkpoint)

    class FakeEnv:
        def __init__(self):
            self.cfg = type("Cfg", (), {"render_spacing": 2.5, "render_offset_mode": "zero"})()

        def run_playback_mode(self, **kwargs):
            assert kwargs["play_render_mode"] == "auto"
            assert kwargs["play_steps"] == 37
            plan = type(
                "Plan",
                (),
                {
                    "mode": "interactive",
                    "headless": False,
                    "record_video": False,
                    "num_steps": None,
                    "output_video": None,
                },
            )()
            kwargs["on_plan"](plan)
            captured["env"] = self
            captured.update({key: value for key, value in kwargs.items() if key != "on_plan"})
            captured["headless"] = plan.headless
            captured["record_video"] = plan.record_video
            captured["num_steps"] = plan.num_steps
            captured["output_video"] = plan.output_video
            return None

    class FakeWrapper:
        def __init__(self, env, device):
            self.env = env
            self.device = device

        def reset(self):
            return 0, {}

        def step(self, actions):
            return 0, 0, False, {}

    class FakeRunner:
        def __init__(self, wrapped_env, train_cfg, log_dir, device):
            self.wrapped_env = wrapped_env
            self.train_cfg = train_cfg
            self.log_dir = log_dir
            self.device = device

        def load(self, path, **kwargs):
            self.loaded_path = path
            self.load_kwargs = kwargs

        def get_inference_policy(self, device):
            return lambda obs: obs

    captured: dict[str, Any] = {}

    monkeypatch.setattr(mod, "EXPORT_POLICY", False, raising=False)
    monkeypatch.setattr(mod, "parse_checkpoint_path", lambda *args, **kwargs: (checkpoint, run_dir))
    monkeypatch.setattr(mod, "build_ppo_play_env_cfg_override", lambda cfg: {})
    monkeypatch.setattr(mod, "create_env", lambda *args, **kwargs: FakeEnv())
    monkeypatch.setattr(mod, "_resolve_ppo_wrapper_cls", lambda rl_cfg: FakeWrapper)
    monkeypatch.setattr(mod, "normalize_ppo_train_cfg", lambda rl_cfg: {})
    monkeypatch.setattr(mod, "OnPolicyRunner", FakeRunner)

    result = mod.play_rsl_rl(cfg, device="cpu")

    assert result is None
    assert captured["headless"] is False
    assert captured["record_video"] is False
    assert captured["num_steps"] is None
    assert captured["output_video"] is None
    assert captured["render_spacing"] == pytest.approx(2.5)
    assert captured["render_offset_mode"] == "zero"


def test_train_rsl_rl_record_play_uses_backend_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(
        [
            "task=go2_joystick_rough/motrix",
            "training.play_only=true",
            "training.play_render_mode=record",
            "training.play_steps=37",
        ]
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "model_37.pt"
    mod.torch.save({"actor_state_dict": {}}, checkpoint)

    class FakeEnv:
        def __init__(self):
            self.cfg = type("Cfg", (), {"render_spacing": 1.0, "render_offset_mode": "grid"})()

        def run_playback_mode(self, **kwargs):
            assert kwargs["play_render_mode"] == "record"
            assert kwargs["play_steps"] == 37
            plan = type(
                "Plan",
                (),
                {
                    "mode": "record",
                    "headless": True,
                    "record_video": True,
                    "num_steps": 37,
                    "output_video": kwargs["output_video"],
                },
            )()
            kwargs["on_plan"](plan)
            captured["env"] = self
            captured.update({key: value for key, value in kwargs.items() if key != "on_plan"})
            captured["headless"] = plan.headless
            captured["record_video"] = plan.record_video
            captured["num_steps"] = plan.num_steps
            captured["output_video"] = plan.output_video
            return str(plan.output_video)

    class FakeWrapper:
        def __init__(self, env, device):
            self.env = env
            self.device = device

        def reset(self):
            return 0, {}

        def step(self, actions):
            return 0, 0, False, {}

    class FakeRunner:
        def __init__(self, wrapped_env, train_cfg, log_dir, device):
            self.wrapped_env = wrapped_env
            self.train_cfg = train_cfg
            self.log_dir = log_dir
            self.device = device

        def load(self, path, map_location=None):
            self.loaded_path = path
            self.map_location = map_location

        def get_inference_policy(self, device):
            return lambda obs: obs

    captured: dict[str, Any] = {}

    monkeypatch.setattr(mod, "EXPORT_POLICY", False, raising=False)
    monkeypatch.setattr(mod, "parse_checkpoint_path", lambda *args, **kwargs: (checkpoint, run_dir))
    monkeypatch.setattr(mod, "build_ppo_play_env_cfg_override", lambda cfg: {})
    monkeypatch.setattr(mod, "create_env", lambda *args, **kwargs: FakeEnv())
    monkeypatch.setattr(mod, "_resolve_ppo_wrapper_cls", lambda rl_cfg: FakeWrapper)
    monkeypatch.setattr(mod, "normalize_ppo_train_cfg", lambda rl_cfg: {})
    monkeypatch.setattr(mod, "OnPolicyRunner", FakeRunner)

    result = mod.play_rsl_rl(cfg, device="cpu")

    assert result == str(run_dir / "play_video.mp4")
    assert captured["headless"] is True
    assert captured["record_video"] is True
    assert captured["num_steps"] == 37
    assert captured["output_video"] == run_dir / "play_video.mp4"


def test_train_appo_get_log_root_uses_algo_log_name(monkeypatch: pytest.MonkeyPatch):
    """Verify APPO _get_log_root uses algo.algo_log_name (issue #168)."""
    monkeypatch.delenv("UNILAB_TEST_LOG_ROOT", raising=False)
    mod = _train_appo()
    cfg = _appo_cfg()

    cfg.algo.algo_log_name = "test_appo"

    log_root = mod._get_log_root(cfg)
    assert "logs/test_appo" in log_root


def test_play_resolve_checkpoint_uses_algo_log_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Verify play_interactive.resolve_checkpoint uses algo_log_name (issue #168)."""
    monkeypatch.delenv("UNILAB_TEST_LOG_ROOT", raising=False)
    mod = _play_interactive()

    # Create test directory structure with custom algo_log_name
    run_dir = tmp_path / "logs" / "custom_ppo" / "MyTask" / "2024-01-01_mujoco"
    run_dir.mkdir(parents=True)
    (run_dir / "model_50.pt").write_bytes(b"")

    # Temporarily override ROOT_DIR to use tmp_path
    original_root = mod.ROOT_DIR
    try:
        mod.ROOT_DIR = tmp_path
        result = mod.resolve_checkpoint("MyTask", "-1", algo_log_name="custom_ppo")
        assert result is not None
        assert "model_50.pt" in result
    finally:
        mod.ROOT_DIR = original_root


def test_ppo_interactive_config_includes_playback_controls():
    cfg = _ppo_cfg()

    assert cfg.interactive.speed == pytest.approx(1.0)
    assert cfg.interactive.start_paused is False


def test_play_interactive_respects_training_device_override():
    mod = _play_interactive()
    cfg = OmegaConf.create({"training": {"device": "cpu"}})

    assert mod._select_playback_device(cfg) == "cpu"


def test_play_interactive_parses_explicit_cli():
    mod = _play_interactive()

    parsed = mod._parse_interactive_cli(
        ["--algo", "hora_distill", "--task", "sharpa_inhand", "--sim", "mujoco_nodr"]
    )

    assert parsed.algo == "hora_distill"
    assert parsed.task == "sharpa_inhand"
    assert parsed.sim == "mujoco_nodr"
    assert parsed.overrides == ["task=sharpa_inhand/mujoco_nodr"]


@pytest.mark.parametrize("algo", ["appo", "sac", "hora_distill"])
def test_play_interactive_parses_feature_algo_flags(algo: str):
    mod = _play_interactive()

    parsed = mod._parse_interactive_cli(
        [f"--algo={algo}", "--task", "sharpa_inhand", "--sim", "mujoco_hora"]
    )

    assert parsed.algo == algo
    assert parsed.overrides == ["task=sharpa_inhand/mujoco_hora"]


def test_play_interactive_cli_respects_owner_action_mode_and_user_override():
    mod = _play_interactive()

    default_parsed = mod._parse_interactive_cli(
        ["--algo", "ppo", "--task", "go2_joystick_rough", "--sim", "mujoco"]
    )
    default_cfg = mod._compose_interactive_config(default_parsed.algo, default_parsed.overrides)

    assert default_cfg.interactive.action_mode == "policy"
    assert default_cfg.training.play_only is True

    parsed = mod._parse_interactive_cli(
        [
            "--algo",
            "ppo",
            "--task",
            "go2_joystick_rough",
            "--sim",
            "mujoco",
            "interactive.action_mode=random",
        ]
    )
    cfg = mod._compose_interactive_config(parsed.algo, parsed.overrides)

    assert parsed.overrides == [
        "task=go2_joystick_rough/mujoco",
        "interactive.action_mode=random",
    ]
    assert cfg.interactive.action_mode == "random"


def test_play_interactive_rejects_unknown_algo_flag():
    mod = _play_interactive()

    with pytest.raises(SystemExit):
        mod._parse_interactive_cli(
            ["--algo=unknown", "--task", "go1_joystick_flat", "--sim", "mujoco"]
        )


def test_play_interactive_dynamic_compose_supports_algo_roots():
    mod = _play_interactive()

    ppo_cfg = mod._compose_interactive_config("ppo", ["task=go1_joystick_flat/mujoco"])
    appo_cfg = mod._compose_interactive_config("appo", ["task=sharpa_inhand/mujoco_hora"])
    sac_cfg = mod._compose_interactive_config("sac", ["task=sharpa_inhand/mujoco_hora"])
    distill_cfg = mod._compose_interactive_config("hora_distill", ["task=sharpa_inhand/mujoco"])

    assert ppo_cfg.algo.algo == "ppo"
    assert appo_cfg.algo.runtime_impl == "hora_appo"
    assert appo_cfg.interactive.action_mode == "policy"
    assert sac_cfg.algo.algo == "sac"
    assert sac_cfg.algo.runtime_impl == "hora_sac"
    assert sac_cfg.interactive.policy_obs_mode == "actor"
    assert distill_cfg.algo.algo_log_name == "hora_distill"
    assert distill_cfg.interactive.action_mode == "policy"


def test_play_interactive_sac_task_shorthand_rewrites_to_owner_group():
    mod = _play_interactive()

    overrides = mod._normalize_interactive_overrides(
        "sac",
        ["task=sharpa_inhand/mujoco_hora", "algo.load_run=my_run"],
    )

    assert overrides == [
        "algo=sac",
        "task=sac/sharpa_inhand/mujoco_hora",
        "algo.load_run=my_run",
        "training.play_only=true",
    ]


def test_play_interactive_replays_checkpoint_env_contract_for_sac_g1(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mod = _play_interactive()
    checkpoint = tmp_path / "model_10.pt"
    checkpoint.write_bytes(b"checkpoint")
    (tmp_path / "run_config.json").write_text(
        json.dumps(
            {
                "config": {
                    "env": {
                        "mode_observation": True,
                        "commands": {"rel_standing_envs": 1.0},
                    },
                    "reward": {
                        "scales": {"alive": 2.0},
                        "mode": {"enabled": True},
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        playback_checkpoint_contract,
        "resolve_task_checkpoint_path",
        lambda *args, **kwargs: (checkpoint, tmp_path),
    )
    args = types.SimpleNamespace(
        task="G1WalkFlat",
        load_run="run",
        checkpoint=None,
        algo_log_name="fast_sac",
        log_root=None,
    )

    merged = mod._apply_checkpoint_env_contract(
        {
            "mode_observation": False,
            "commands": {"rel_standing_envs": 0.0},
            "reward_config": {"scales": {}},
        },
        args,
    )

    assert merged["mode_observation"] is True
    assert merged["commands"]["rel_standing_envs"] == 1.0
    assert merged["reward_config"]["mode"]["enabled"] is True


def test_play_interactive_treats_missing_g1_mode_observation_as_legacy_false(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mod = _play_interactive()
    checkpoint = tmp_path / "model_10.pt"
    checkpoint.write_bytes(b"checkpoint")
    (tmp_path / "run_config.json").write_text(
        json.dumps(
            {
                "config": {
                    "env": {
                        "commands": {"rel_standing_envs": 0.0},
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        playback_checkpoint_contract,
        "resolve_task_checkpoint_path",
        lambda *args, **kwargs: (checkpoint, tmp_path),
    )
    args = types.SimpleNamespace(
        task="G1WalkFlat",
        load_run="run",
        checkpoint=None,
        algo_log_name="fast_sac",
        log_root=None,
    )

    merged = mod._apply_checkpoint_env_contract(
        {
            "mode_observation": True,
            "commands": {"rel_standing_envs": 1.0},
        },
        args,
    )

    assert merged["mode_observation"] is False
    assert merged["commands"]["rel_standing_envs"] == 0.0


def test_play_interactive_infers_missing_g1_mode_observation_from_checkpoint_dim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mod = _play_interactive()
    checkpoint = tmp_path / "model_10.pt"
    checkpoint.write_bytes(b"checkpoint")
    (tmp_path / "run_config.json").write_text(
        json.dumps(
            {
                "config": {
                    "env": {
                        "commands": {"rel_standing_envs": 0.0},
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        playback_checkpoint_contract,
        "resolve_task_checkpoint_path",
        lambda *args, **kwargs: (checkpoint, tmp_path),
    )
    monkeypatch.setattr(
        playback_checkpoint_contract, "_checkpoint_actor_input_dim", lambda args: 99
    )
    args = types.SimpleNamespace(
        algo="sac",
        task="G1WalkFlat",
        load_run="run",
        checkpoint=None,
        algo_log_name="fast_sac",
        log_root=None,
    )

    merged = mod._apply_checkpoint_env_contract(
        {"commands": {"rel_standing_envs": 1.0}},
        args,
    )

    assert merged["mode_observation"] is True
    assert merged["commands"]["rel_standing_envs"] == 0.0


def test_play_interactive_infers_missing_g1_height_command_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mod = _play_interactive()
    checkpoint = tmp_path / "model_5000.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        playback_checkpoint_contract,
        "resolve_task_checkpoint_path",
        lambda *args, **kwargs: (checkpoint, tmp_path),
    )
    monkeypatch.setattr(
        playback_checkpoint_contract, "_checkpoint_actor_input_dim", lambda args: 100
    )
    args = types.SimpleNamespace(
        algo="sac",
        task="G1WalkFlat",
        load_run="run",
        checkpoint=None,
        algo_log_name="fast_sac",
        log_root=None,
    )

    merged = mod._apply_checkpoint_env_contract(
        {
            "mode_observation": True,
            "commands": {"rel_standing_envs": 0.3},
        },
        args,
    )

    assert merged["mode_observation"] is True
    assert merged["commands"]["rel_standing_envs"] == 0.3
    assert merged["commands"]["observe_height_command"] is True


def test_play_interactive_does_not_infer_g1_height_for_legacy_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mod = _play_interactive()
    checkpoint = tmp_path / "model_5000.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        playback_checkpoint_contract,
        "resolve_task_checkpoint_path",
        lambda *args, **kwargs: (checkpoint, tmp_path),
    )
    monkeypatch.setattr(
        playback_checkpoint_contract, "_checkpoint_actor_input_dim", lambda args: 99
    )
    args = types.SimpleNamespace(
        algo="sac",
        task="G1WalkFlat",
        load_run="run",
        checkpoint=None,
        algo_log_name="fast_sac",
        log_root=None,
    )

    merged = mod._apply_checkpoint_env_contract(
        {
            "mode_observation": True,
            "commands": {"rel_standing_envs": 0.3},
        },
        args,
    )

    assert "observe_height_command" not in merged["commands"]


def test_play_interactive_warns_about_hard_gated_standing_checkpoint() -> None:
    mod = _play_interactive()

    issues = mod._g1_standing_contract_issues(
        {
            "config": {
                "env": {
                    "commands": {"rel_standing_envs": 0.4},
                    "stand_action_authority": True,
                },
                "reward": {
                    "scales": {"feet_phase": 0.0},
                    "mode": {
                        "enabled": True,
                        "stand_terms": [
                            "stand_still",
                            "stand_action_l2",
                            "stand_dof_vel_l2",
                            "stand_lin_vel_xy_l2",
                            "stand_yaw_vel_l2",
                        ],
                    },
                    "gait_constraint": {"freeze_phase_in_stand_mode": True},
                },
            }
        }
    )

    assert any("stand_action_authority=true" in issue for issue in issues)


def test_play_interactive_distill_playback_forces_standing_reset() -> None:
    mod = _play_interactive()
    env_cfg_override = {
        "commands": {
            "rel_standing_envs": 0.0,
            "rel_transition_envs": 1.0,
            "vel_limit": [[0.4, 0.0, 0.0], [0.7, 0.0, 0.0]],
        },
        "standing_reset_base_qvel_limit": 0.5,
    }

    for task_name in (
        "g1_walk_flat",
        "g1_walk_flat/mujoco",
        "G1WalkFlat",
        "g1-walk-flat",
    ):
        resolved = mod._apply_distill_playback_reset_contract(env_cfg_override, task_name)

        assert resolved is not env_cfg_override
        assert resolved["commands"]["rel_standing_envs"] == 1.0
        assert resolved["commands"]["rel_transition_envs"] == 0.0
        assert resolved["standing_reset_base_qvel_limit"] == 0.0

    untouched = mod._apply_distill_playback_reset_contract(env_cfg_override, "g1_stand_still")
    assert untouched == env_cfg_override
    assert env_cfg_override["commands"]["rel_standing_envs"] == 0.0


def test_play_interactive_command_obs_probe_disables_standing_sampling() -> None:
    mod = _play_interactive()
    probe = np.asarray(mod._COMMAND_OBS_VERIFY_COMMAND, dtype=np.float32)

    class Cfg:
        commands = types.SimpleNamespace(
            vel_limit=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            rel_standing_envs=0.4,
        )

    class State:
        info: dict[str, Any]
        obs: dict[str, np.ndarray]

    class Env:
        cfg = Cfg()
        state = State()

    env = Env()
    reset_rel_standing_values: list[float] = []

    def reset_fn() -> None:
        rel_standing = float(env.cfg.commands.rel_standing_envs)
        reset_rel_standing_values.append(rel_standing)
        if rel_standing > 0.0:
            command = np.zeros(3, dtype=np.float32)
        else:
            command = probe.copy()
        env.state.info = {"commands": command[None, :]}
        env.state.obs = {"obs": np.concatenate([np.zeros(5, dtype=np.float32), command])[None, :]}

    assert mod._policy_obs_contains_command(env, reset_fn=reset_fn) is True
    assert reset_rel_standing_values[0] == 0.0
    assert env.cfg.commands.rel_standing_envs == 0.4
    assert env.cfg.commands.vel_limit == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_play_interactive_command_obs_probe_refreshes_stale_reset_command() -> None:
    mod = _play_interactive()

    class Cfg:
        commands = types.SimpleNamespace(
            vel_limit=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            rel_standing_envs=0.4,
        )

    class State:
        info: dict[str, Any]
        obs: dict[str, np.ndarray]

    class Env:
        cfg = Cfg()
        state = State()

        def update_state(self, state: State) -> State:
            command = state.info["commands"][0, :3]
            state.obs = {"obs": np.concatenate([np.zeros(5, dtype=np.float32), command])[None, :]}
            return state

    env = Env()
    reset_count = 0

    def reset_fn() -> None:
        nonlocal reset_count
        reset_count += 1
        command = np.zeros(3, dtype=np.float32)
        env.state.info = {"commands": command[None, :]}
        env.state.obs = {"obs": np.concatenate([np.zeros(5, dtype=np.float32), command])[None, :]}

    assert mod._policy_obs_contains_command(env, reset_fn=reset_fn) is True
    assert reset_count == 2
    assert env.cfg.commands.rel_standing_envs == 0.4
    assert env.cfg.commands.vel_limit == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_play_interactive_command_obs_probe_restores_live_state_when_reset_is_external() -> None:
    mod = _play_interactive()
    probe = np.asarray(mod._COMMAND_OBS_VERIFY_COMMAND, dtype=np.float32)
    original_command = np.zeros(3, dtype=np.float32)
    original_obs = np.concatenate([np.zeros(5, dtype=np.float32), original_command])[None, :]

    class Cfg:
        commands = types.SimpleNamespace(
            vel_limit=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            rel_standing_envs=1.0,
        )

    class State:
        info: dict[str, Any]
        obs: dict[str, np.ndarray]

    class Env:
        cfg = Cfg()
        state = State()

        def update_state(self, state: State) -> State:
            command = state.info["commands"][0, :3]
            state.info["gait_enabled"] = np.asarray(
                [float(np.linalg.norm(command) > 1.0e-9)], dtype=np.float32
            )
            state.obs = {"obs": np.concatenate([np.zeros(5, dtype=np.float32), command])[None, :]}
            return state

    env = Env()
    env.state.info = {"commands": original_command[None, :].copy(), "gait_enabled": np.zeros(1)}
    env.state.obs = {"obs": original_obs.copy()}
    reset_count = 0

    def reset_fn() -> None:
        nonlocal reset_count
        reset_count += 1

    assert mod._policy_obs_contains_command(env, reset_fn=reset_fn) is True
    assert reset_count == 2
    assert np.allclose(env.state.info["commands"], original_command[None, :])
    assert np.allclose(env.state.info["gait_enabled"], np.zeros(1))
    assert np.allclose(env.state.obs["obs"], original_obs)
    assert not np.allclose(env.state.info["commands"][0, :3], probe)
    assert env.cfg.commands.rel_standing_envs == 1.0
    assert env.cfg.commands.vel_limit == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_play_interactive_runner_log_dir_uses_algo_log_name(monkeypatch: pytest.MonkeyPatch):
    import types

    mod = _play_interactive()
    captured: dict[str, object] = {}

    class FakeWrapper:
        def __init__(self, env, device, policy_obs_mode):
            self.env = env
            captured["policy_obs_mode"] = policy_obs_mode

        def reset(self):
            return None, {}

    class FakeRunner:
        def __init__(self, wrapped_env, train_cfg, log_dir, device):
            del wrapped_env, train_cfg, device
            captured["log_dir"] = log_dir

        def load(self, ckpt, load_cfg):
            captured["ckpt"] = ckpt
            captured["load_cfg"] = load_cfg

        def get_inference_policy(self, device):
            del device
            return object()

    class FakeViewer:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def is_running(self):
            return False

        def sync(self):
            pass

        user_scn = type("Scene", (), {"ngeom": 0})()

    fake_env = types.SimpleNamespace(
        obs_groups_spec={"obs": 5},
        action_space=types.SimpleNamespace(shape=(3,), low=np.full((3,), -1.0), high=np.ones((3,))),
        cfg=types.SimpleNamespace(ctrl_dt=0.02),
        get_scene_artifacts=lambda: BackendSceneArtifacts(),
        get_playback_model=lambda: object(),
        get_physics_state_snapshot=lambda: np.zeros((1, 8), dtype=np.float32),
    )

    monkeypatch.setattr(playback_viewer.registry, "make", lambda *args, **kwargs: fake_env)
    monkeypatch.setattr(
        playback_viewer, "resolve_checkpoint", lambda *args, **kwargs: "/tmp/model_10.pt"
    )
    monkeypatch.setattr(
        playback_viewer,
        "get_entrypoint_log_root",
        lambda root_dir, *, algo_log_name, log_root=None: Path("/tmp") / algo_log_name,
    )
    monkeypatch.setattr(playback_viewer, "RslRlVecEnvWrapper", FakeWrapper)
    monkeypatch.setattr(playback_viewer, "OnPolicyRunner", FakeRunner)
    monkeypatch.setattr(
        playback_cli_owner, "PPOConfig", lambda: types.SimpleNamespace(to_dict=lambda: {})
    )
    monkeypatch.setattr(playback_viewer.mujoco, "MjData", lambda model: object())
    monkeypatch.setattr(playback_viewer.mujoco, "mj_setState", lambda *args, **kwargs: None)
    monkeypatch.setattr(playback_viewer.mujoco, "mj_forward", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        playback_viewer.mujoco,
        "mjtState",
        types.SimpleNamespace(mjSTATE_FULLPHYSICS=0),
    )
    monkeypatch.setattr(
        playback_viewer.mujoco.viewer,
        "launch_passive",
        lambda *args, **kwargs: FakeViewer(),
    )

    args = types.SimpleNamespace(
        task="MyTask",
        load_run="-1",
        checkpoint=None,
        action_mode="policy",
        policy_obs_mode="flat",
        algo_log_name="custom_ppo",
        show_target_bodies=False,
        show_reward_debug=False,
        target_body_names="",
        target_max_bodies=0,
        target_marker_radius=0.02,
        target_axis_length=0.08,
        target_marker_alpha=0.75,
        target_show_axes=False,
        reward_debug_show_velocity=False,
        reward_debug_lin_vel_scale=0.08,
        reward_debug_ang_vel_scale=0.05,
        reward_debug_show_connectors=False,
        reward_debug_show_global_anchor=False,
        speed=1.0,
        start_paused=False,
    )

    mod.play_interactive(args)

    assert captured["ckpt"] == "/tmp/model_10.pt"
    assert captured["log_dir"] == "/tmp/custom_ppo/MyTask/play_temp"


def test_play_interactive_import_does_not_swallow_registry_bootstrap_errors(
    monkeypatch: pytest.MonkeyPatch,
):
    import types

    play_interactive_path = _SCRIPTS_DIR / "play_interactive.py"
    training_mod = cast(Any, types.ModuleType("unilab.training"))

    def _fail_bootstrap() -> None:
        raise RuntimeError("bootstrap failed")

    training_mod.ensure_registries = _fail_bootstrap
    training_mod.get_entrypoint_log_root = lambda *args, **kwargs: Path("/tmp")
    training_mod.resolve_task_checkpoint_path = lambda *args, **kwargs: (None, None)
    monkeypatch.setitem(sys.modules, "unilab.training", training_mod)

    mujoco_mod = cast(Any, types.ModuleType("mujoco"))
    mujoco_mod.viewer = cast(Any, types.ModuleType("mujoco.viewer"))
    monkeypatch.setitem(sys.modules, "mujoco", mujoco_mod)
    monkeypatch.setitem(sys.modules, "mujoco.viewer", mujoco_mod.viewer)

    spec = importlib.util.spec_from_file_location(
        "play_interactive_test_module", play_interactive_path
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="bootstrap failed"):
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
