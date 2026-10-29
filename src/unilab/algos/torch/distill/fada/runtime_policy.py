"""Explicit legacy/official deployment interfaces; no simulator imports."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from unilab.algos.torch.distill.fada.collection_contract import validate_official_fada_preprocessing
from unilab.algos.torch.distill.fada.model import FADAPlannerIDMPolicy, PlannerIDMOutput
from unilab.algos.torch.distill.fada.observation import FADA_OFFICIAL_COMPACT_OBSERVATION_CONTRACT
from unilab.algos.torch.distill.fada.official_model import PlannerIDMPolicy


@dataclass(frozen=True)
class FADARuntimeSpec:
    family: Literal["legacy", "official"]
    obs_dim: int
    action_dim: int
    command_dim: int
    observation_contract: str
    history_length: int
    prediction_horizon: int
    idm_use_current_command_for_history: bool = False
    preprocessing: Mapping[str, Any] | None = None


def get_fada_runtime_spec(policy: FADAPlannerIDMPolicy | PlannerIDMPolicy) -> FADARuntimeSpec:
    if isinstance(policy, PlannerIDMPolicy):
        cfg = cast(Mapping[str, Any], policy.official_model_config)
        preprocessing = getattr(policy, "official_preprocessing", None)
        return FADARuntimeSpec(
            family="official", obs_dim=cfg["obs_dim"], action_dim=cfg["act_dim"],
            command_dim=cfg["cmd_dim"], observation_contract=FADA_OFFICIAL_COMPACT_OBSERVATION_CONTRACT,
            history_length=cfg["history_len"], prediction_horizon=cfg["pred_horizon"],
            idm_use_current_command_for_history=cfg["idm_use_current_command_for_history"],
            preprocessing=(validate_official_fada_preprocessing(preprocessing) if preprocessing is not None else None),
        )
    if not isinstance(policy, FADAPlannerIDMPolicy):
        raise TypeError("deployment requires an explicitly identified legacy or official FADA model")
    legacy = policy.config
    return FADARuntimeSpec(
        "legacy", legacy.obs_dim, legacy.action_dim, legacy.command_dim, legacy.observation_contract,
        legacy.history_length, legacy.prediction_horizon,
    )


def invoke_fada_policy(policy, spec: FADARuntimeSpec, observations, actions, command, history_valid_mask=None) -> PlannerIDMOutput:
    if spec.family == "legacy":
        return policy(observations, actions, command)
    if history_valid_mask is None:
        raise ValueError("official inference requires history_valid_mask")
    chunk, future = policy(
        observations, actions, command, history_valid_mask=history_valid_mask, return_obs=True
    )
    return PlannerIDMOutput(predicted_future=future, action_chunk=chunk, action=chunk[:, 0])


def invoke_fada_idm(idm, spec: FADARuntimeSpec, observations, actions, command, future, history_valid_mask=None):
    if spec.family == "legacy":
        return idm(observations, actions, future)
    if history_valid_mask is None:
        raise ValueError("official IDM inference requires history_valid_mask")
    return idm(
        observations, actions,
        command if spec.idm_use_current_command_for_history else None,
        future, history_valid_mask=history_valid_mask,
    )


def _vector(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 29:
        raise ValueError(f"official playback {name} must contain 29 per-joint values")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"official playback {name} must contain numeric values")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) or (positive and item <= 0) or (nonnegative and item < 0) for item in result):
        raise ValueError(f"official playback {name} has invalid values")
    return result


@dataclass(frozen=True)
class OfficialFADAPlaybackProfile:
    """Required source-run values, not guessed deployment defaults.

    Arrays use source joint_names order. The existing source_identity binds this
    explicit profile to its checkpoint without loading the source PPO on playback.
    """
    source_identity: Mapping[str, str]
    joint_names: tuple[str, ...]
    default_joint_pos: tuple[float, ...]
    action_scale: tuple[float, ...]
    stiffness: tuple[float, ...]
    damping: tuple[float, ...]
    clip_actions: float
    ctrl_dt: float
    gait_frequency: float
    gait_clock_mode: str
    phase_init: tuple[float, float]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, source_identity: Mapping[str, str]) -> OfficialFADAPlaybackProfile:
        required = set(cls.__dataclass_fields__)
        if not isinstance(payload, Mapping):
            raise ValueError("official playback requires a source-bound control profile mapping")
        missing = sorted(required - set(payload))
        extra = sorted(set(payload) - required)
        if missing or extra:
            raise ValueError(f"official playback control profile fields: missing={missing}, extra={extra}")
        if dict(payload["source_identity"]) != dict(source_identity):
            raise ValueError("official playback control profile belongs to another source run")
        names = payload["joint_names"]
        if not isinstance(names, Sequence) or isinstance(names, (str, bytes)) or len(names) != 29 or len(set(names)) != 29 or not all(isinstance(name, str) and name for name in names):
            raise ValueError("official playback joint_names must contain 29 unique names")
        scalars = {}
        for name in ("clip_actions", "ctrl_dt", "gait_frequency"):
            value = payload[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"official playback {name} must be positive and finite")
            scalars[name] = float(value)
        phase = payload["phase_init"]
        if not isinstance(phase, Sequence) or isinstance(phase, (str, bytes)) or len(phase) != 2 or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in phase):
            raise ValueError("official playback phase_init must contain two finite radians")
        clock = payload["gait_clock_mode"]
        if not isinstance(clock, str) or not clock:
            raise ValueError("official playback gait_clock_mode must name the source clock")
        return cls(
            source_identity=dict(source_identity), joint_names=tuple(names),
            default_joint_pos=_vector(payload["default_joint_pos"], "default_joint_pos"),
            action_scale=_vector(payload["action_scale"], "action_scale", positive=True),
            stiffness=_vector(payload["stiffness"], "stiffness", positive=True),
            damping=_vector(payload["damping"], "damping", nonnegative=True),
            gait_clock_mode=clock, phase_init=(float(phase[0]), float(phase[1])), **scalars,
        )

    def source_indices(self, target_joint_names: Sequence[str]) -> tuple[int, ...]:
        names = tuple(target_joint_names)
        if len(names) != 29 or len(set(names)) != 29 or set(names) != set(self.joint_names):
            raise ValueError("official playback target joint names do not match source joints")
        indices = {name: index for index, name in enumerate(names)}
        return tuple(indices[name] for name in self.joint_names)
