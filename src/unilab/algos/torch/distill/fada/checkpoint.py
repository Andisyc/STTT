"""Paired FADA policy and optimizer checkpoint owner."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from unilab.algos.torch.distill.fada.async_config import validate_fada_training_schedule
from unilab.algos.torch.distill.fada.model import FADAPlannerIDMPolicy
from unilab.algos.torch.distill.fada.source_artifact import load_architecture_config
from unilab.algos.torch.distill.fada.trainer import FADATrainer
from unilab.algos.torch.distill.fada.collection_contract import (
    OFFICIAL_FADA_SOURCE_REVISION,
    validate_official_fada_preprocessing,
)
from unilab.algos.torch.distill.fada.official_learning import (
    OfficialFADALearner,
    canonical_official_training_config,
)
from unilab.algos.torch.distill.fada.official_model import (
    PlannerIDMPolicy as OfficialPlannerIDMPolicy,
    build_official_fada_student,
    canonical_official_model_config,
)

FADA_CHECKPOINT_SCHEMA_VERSION = 5
FADA_TRAINING_SCHEDULE = "alternating_idm_then_planner"


FADA_V005_REQUIRED_QUALITY_METRICS = (
    "scenario/walk/planner_idm_oracle_action_mse",
    "scenario/walk/cold_start_fraction",
    "scenario/walk/cold_start_planner_mse",
    "scenario/walk/steady_state_planner_mse",
    "scenario/static_stand/planner_idm_oracle_action_mse",
    "scenario/walk_to_stand/planner_idm_oracle_action_mse",
    "scenario/static_stand/cold_start_fraction",
    "scenario/static_stand/cold_start_planner_mse",
    "scenario/static_stand/steady_state_planner_mse",
)


@dataclass(frozen=True)
class LoadedFADAPlannerIDMPolicy:
    """Inference-ready FADA policy reconstructed from one strict checkpoint."""

    policy: FADAPlannerIDMPolicy
    checkpoint: Mapping[str, Any]


@dataclass(frozen=True)
class LoadedOfficialFADAStudentPolicy:
    """Explicit official model identity, separate from the legacy architecture."""

    policy: OfficialPlannerIDMPolicy
    checkpoint: Mapping[str, Any]


def _canonical_state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().to(device="cpu").contiguous()
        identity = f"{name}\0{tensor.dtype}\0{tuple(tensor.shape)}\0".encode("ascii")
        digest.update(len(identity).to_bytes(8, "big"))
        digest.update(identity)
        raw = tensor.view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _validated_idm_state(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    state = payload.get("idm_state_dict")
    if not isinstance(state, dict) or payload.get("idm_sha256") != _canonical_state_dict_sha256(
        state
    ):
        raise ValueError("FADA checkpoint IDM identity mismatch")
    return state


def _validate_schema5_training_state(payload: Mapping[str, Any]) -> None:
    validate_fada_training_schedule(payload.get("training_schedule"))
    required = ("idm_optimizer_state_dict", "planner_optimizer_state_dict")
    if any(not isinstance(payload.get(name), dict) for name in required):
        raise ValueError("schema-5 FADA checkpoint requires both optimizer states")


def save_fada_checkpoint(
    path: str | Path,
    policy: FADAPlannerIDMPolicy,
    trainer: FADATrainer,
    *,
    completed_iterations: int,
    samples_seen: int,
    runtime_config: Mapping[str, Any],
    quality_metrics: Mapping[str, float] | None = None,
) -> Path:
    """Atomically persist the paired FADA module and optimizer identity."""

    metrics = dict(quality_metrics or {})
    v005_enabled = bool(dict(runtime_config.get("v005_replay") or {}).get("enabled", False))
    if v005_enabled and int(completed_iterations) > 0:
        missing = [name for name in FADA_V005_REQUIRED_QUALITY_METRICS if name not in metrics]
        if missing:
            raise ValueError(f"v005 FADA checkpoint quality metrics are missing: {missing}")
        invalid = [
            name
            for name, value in metrics.items()
            if not isinstance(value, (int, float)) or not math.isfinite(float(value))
        ]
        if invalid:
            raise ValueError(f"v005 FADA checkpoint quality metrics are non-finite: {invalid}")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    idm_sha256 = _canonical_state_dict_sha256(policy.idm.state_dict())
    training_schedule = validate_fada_training_schedule(
        runtime_config.get("training_schedule", FADA_TRAINING_SCHEDULE)
    )
    payload = {
        "schema_version": FADA_CHECKPOINT_SCHEMA_VERSION,
        "architecture": asdict(policy.config),
        "planner_state_dict": policy.planner.state_dict(),
        "idm_state_dict": policy.idm.state_dict(),
        "training_schedule": training_schedule,
        "planner_optimizer_state_dict": trainer.planner_optimizer.state_dict(),
        "idm_optimizer_state_dict": trainer.idm_optimizer.state_dict(),
        "idm_sha256": idm_sha256,
        "completed_iterations": int(completed_iterations),
        "samples_seen": int(samples_seen),
        "runtime_config": dict(runtime_config),
        "quality_metrics": metrics,
    }
    torch.save(payload, temporary)
    temporary.replace(target)
    return target


def load_fada_checkpoint(
    path: str | Path,
    policy: FADAPlannerIDMPolicy,
    trainer: FADATrainer | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Restore one exact FADA architecture and optionally its paired optimizers."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") not in {1, 2, 3, 4, 5}:
        raise ValueError("unsupported or malformed FADA checkpoint schema")
    if trainer is not None:
        raise ValueError(
            "v011 training resume is disabled; load without a trainer for state inspection"
        )
    observed = load_architecture_config(
        payload.get("architecture"),
        schema_version=payload.get("schema_version"),
        contract_schema_version=FADA_CHECKPOINT_SCHEMA_VERSION,
        context="FADA checkpoint",
    )
    if observed != policy.config:
        raise ValueError(
            f"FADA checkpoint architecture mismatch: expected={policy.config} observed={observed}"
        )
    if payload.get("schema_version") in {4, 5}:
        _validated_idm_state(payload)
    if payload.get("schema_version") == 5:
        _validate_schema5_training_state(payload)
    policy.planner.load_state_dict(payload["planner_state_dict"], strict=True)
    policy.idm.load_state_dict(payload["idm_state_dict"], strict=True)
    return payload


def load_fada_policy_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> LoadedFADAPlannerIDMPolicy:
    """Construct an inference-only Planner-IDM policy from checkpoint-owned architecture."""

    # B1: 只读取 tensor 与基础类型, 并在模型构造前关闭 schema/architecture 不匹配.
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") not in {1, 2, 3, 4, 5}:
        raise ValueError("unsupported or malformed FADA checkpoint schema")
    config = load_architecture_config(
        payload.get("architecture"),
        schema_version=payload.get("schema_version"),
        contract_schema_version=FADA_CHECKPOINT_SCHEMA_VERSION,
        context="FADA checkpoint",
    )
    if payload.get("schema_version") in {4, 5}:
        _validated_idm_state(payload)
    if payload.get("schema_version") == 5:
        _validate_schema5_training_state(payload)

    # B2: 严格恢复 Planner 与 IDM, 产出 eval 模式的唯一 playback policy owner.
    policy = FADAPlannerIDMPolicy(config).to(device)
    policy.planner.load_state_dict(payload["planner_state_dict"], strict=True)
    policy.idm.load_state_dict(payload["idm_state_dict"], strict=True)
    policy.eval()
    return LoadedFADAPlannerIDMPolicy(policy=policy, checkpoint=payload)


_OFFICIAL_STUDENT_FORMAT = "unilab_official_fada_student"


def _official_source_identity(identity: Mapping[str, Any]) -> dict[str, str]:
    names = (
        "upstream_revision", "expert_checkpoint", "expert_checkpoint_sha256",
        "experiment_config_sha256",
    )
    if not isinstance(identity, Mapping) or any(
        not isinstance(identity.get(name), str) or not identity[name] for name in names
    ):
        raise ValueError("official student requires complete expert/experiment provenance")
    result = {name: identity[name] for name in names}
    if result["upstream_revision"] != OFFICIAL_FADA_SOURCE_REVISION:
        raise ValueError("official student upstream revision mismatch")
    if not Path(result["expert_checkpoint"]).is_absolute():
        raise ValueError("official expert checkpoint identity must be an absolute path")
    for name in ("expert_checkpoint_sha256", "experiment_config_sha256"):
        value = result[name]
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"official student has malformed {name}")
    return result


def save_official_fada_checkpoint(
    path: str | Path,
    learner: OfficialFADALearner,
    *,
    completed_iterations: int,
    samples_seen: int,
    source_identity: Mapping[str, Any],
    rollout_step_count: int = 0,
    online_buffer_path: str | Path | None = None,
) -> Path:
    """Save native-key student state and both optimizers, separate from legacy/PPO."""

    counters = (completed_iterations, samples_seen, learner.update_count, rollout_step_count)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counters):
        raise ValueError("official student counters must be non-negative integers")
    model_config = canonical_official_model_config(learner.policy.official_model_config)
    payload = {
        "format": _OFFICIAL_STUDENT_FORMAT,
        "schema_version": 2,
        "model_config": model_config,
        "training_config": canonical_official_training_config(learner.training_config),
        "preprocessing": validate_official_fada_preprocessing(learner.preprocessing),
        "source_identity": _official_source_identity(source_identity),
        "student_state_dict": learner.policy.state_dict(),
        "idm_optimizer_state_dict": learner.idm_optimizer.state_dict(),
        "planner_optimizer_state_dict": learner.planner_optimizer.state_dict(),
        "update_count": learner.update_count,
        "completed_iterations": completed_iterations,
        "samples_seen": samples_seen,
        "rollout_step_count": rollout_step_count,
        "online_buffer_path": None if online_buffer_path is None else str(online_buffer_path),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)
    return target


def _read_official_student_checkpoint(
    path: str | Path, *, map_location: str | torch.device
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("format") != _OFFICIAL_STUDENT_FORMAT
        or payload.get("schema_version") != 2
    ):
        raise ValueError("expected an official64/29/7 student checkpoint, not legacy or Oracle PPO/APPO")
    for name, canonicalize in (
        ("model_config", canonical_official_model_config),
        ("training_config", canonical_official_training_config),
        ("preprocessing", validate_official_fada_preprocessing),
        ("source_identity", _official_source_identity),
    ):
        value = payload.get(name)
        if not isinstance(value, dict) or canonicalize(value) != value:
            raise ValueError(f"official student checkpoint requires complete canonical {name}")
    for name in ("student_state_dict", "idm_optimizer_state_dict", "planner_optimizer_state_dict"):
        if not isinstance(payload.get(name), dict):
            raise ValueError(f"official student checkpoint missing {name}")
    for name in ("update_count", "completed_iterations", "samples_seen", "rollout_step_count"):
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"official student checkpoint has invalid {name}")
    replay_path = payload.get("online_buffer_path")
    if "online_buffer_path" not in payload or (
        replay_path is not None and (not isinstance(replay_path, str) or not replay_path)
    ):
        raise ValueError("official checkpoint has an invalid online Replay reference")
    return payload


def _restore_official_student(payload: Mapping[str, Any], *, device: str | torch.device):
    policy = build_official_fada_student(payload["model_config"], device=device)
    policy.load_state_dict(payload["student_state_dict"], strict=True)
    policy.official_preprocessing = payload["preprocessing"]
    policy.official_source_identity = payload["source_identity"]
    return policy.eval()


def load_official_fada_student_checkpoint(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> torch.nn.Module:
    """Inference-only loader usable directly by build_official_fada_runtime."""
    payload = _read_official_student_checkpoint(path, map_location=device)
    return _restore_official_student(payload, device=device)


def load_official_fada_playback_checkpoint(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> LoadedOfficialFADAStudentPolicy:
    payload = _read_official_student_checkpoint(path, map_location=device)
    return LoadedOfficialFADAStudentPolicy(
        policy=_restore_official_student(payload, device=device), checkpoint=payload
    )


def load_official_fada_checkpoint(
    path: str | Path,
    learner: OfficialFADALearner,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Resume only the same resolved model, optimization, and preprocessing contract."""

    payload = _read_official_student_checkpoint(path, map_location=map_location)
    if (
        payload["model_config"] != learner.policy.official_model_config
        or payload["training_config"] != learner.training_config
        or payload["preprocessing"] != learner.preprocessing
    ):
        raise ValueError("official student resume configuration/preprocessing mismatch")
    learner.policy.load_state_dict(payload["student_state_dict"], strict=True)
    learner.idm_optimizer.load_state_dict(payload["idm_optimizer_state_dict"])
    learner.planner_optimizer.load_state_dict(payload["planner_optimizer_state_dict"])
    learner.update_count = payload["update_count"]
    return payload
