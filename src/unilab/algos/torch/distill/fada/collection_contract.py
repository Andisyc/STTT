from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Literal, Mapping

import numpy as np
import torch

from unilab.algos.torch.distill.fada.model import FADASourceBatch
from unilab.algos.torch.distill.fada.windows import FADACausalTransition
from unilab.algos.torch.distill.runtime.async_runtime import DaggerCollectRequest


@dataclass(frozen=True)
class FADACollectionResult:
    batch: FADASourceBatch
    env_steps: int
    rejected_done_transitions: int
    rejected_command_windows: int
    rollout_mode: str
    command_scenario: str = "walk"
    oracle_role: str = "walking"
    rejected_scenario_windows: int = 0
    window_profile: str = "steady_state"


@dataclass(frozen=True)
class FADACollectionSpec:
    """Validated collection semantics that travel together across FADA callers."""

    observation_key: str = "obs"
    teacher_projection: str = "identity"
    student_projection: str = "identity"
    student_drop_index: int | None = None
    command_info_keys: tuple[str, ...] = ("commands",)
    max_env_steps: int | None = None
    collect_oracle_shadow: bool = False
    command_scenario: Literal["walk", "static_stand", "walk_to_stand"] = "walk"
    transition_walk_command: tuple[float, ...] = (0.4, 0.0, 0.0)
    transition_pre_switch_steps: int | None = None
    transition_post_switch_steps: int | None = None
    planner_eligible: bool = True
    cold_start_windows: bool = False


@dataclass(frozen=True)
class FADACollectionTransition(FADACausalTransition):
    oracle_action: np.ndarray
    oracle_future: np.ndarray
    oracle_action_chunk: np.ndarray
    oracle_shadow_valid: bool


OFFICIAL_FADA_SOURCE_REVISION = "290b3a9c5c0ebe2e9851b61350617544be05e9bc"
OFFICIAL_FADA_SCENARIO = "official_fada_iteration"


@dataclass(frozen=True)
class OfficialFADACollectionSpec:
    """Official source identity; never reinterpret the historical 66/3 route."""

    upstream_root: str
    expert_checkpoint: str
    config: dict[str, Any]
    experiment_override_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.upstream_root or not self.expert_checkpoint:
            raise ValueError("official FADA requires an upstream checkout and PPO checkpoint")
        if not isinstance(self.config, dict):
            raise TypeError("official FADA config must be a mapping of upstream FADAConfig fields")
        object.__setattr__(self, "config", dict(self.config))
        for name, value in (
            ("history_len", self.history_length),
            ("pred_horizon", self.prediction_horizon),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"official FADA {name} must be a positive integer")
        if self.config.get("predict_future_obs", True) is not True:
            raise ValueError("official collection requires future-observation labels")
        if self.config.get("strict_chunk_labels", True) is not True:
            raise ValueError("official collection requires strict expert chunk labels")
        if self.config.get("command_profile", "locomotion") != "locomotion":
            raise ValueError("official source collection supports the locomotion command profile")

    @property
    def history_length(self) -> int:
        return self.config.get("history_len", 30)

    @property
    def prediction_horizon(self) -> int:
        return self.config.get("pred_horizon", 6)


@dataclass(frozen=True)
class OfficialFADASourceBatch:
    """Preserve official shadow labels and independently valid real trajectories."""

    history_obs: torch.Tensor
    history_act: torch.Tensor
    history_valid_mask: torch.Tensor
    current_command: torch.Tensor
    target_actions: torch.Tensor
    target_next_observations: torch.Tensor
    trajectory_target_actions: torch.Tensor
    trajectory_target_next_observations: torch.Tensor
    trajectory_target_valid: torch.Tensor

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> OfficialFADASourceBatch:
        names = {item.name for item in fields(cls)}
        if set(mapping) != names:
            raise ValueError(
                "official FADA tensor fields mismatch: "
                f"missing={sorted(names - set(mapping))}, extra={sorted(set(mapping) - names)}"
            )
        return cls(**{name: torch.as_tensor(mapping[name]) for name in names})

    @property
    def num_samples(self) -> int:
        return int(self.current_command.shape[0])

    def validate(self, spec: OfficialFADACollectionSpec) -> OfficialFADASourceBatch:
        if self.current_command.ndim != 2:
            raise ValueError("official current_command must be a matrix")
        count = self.num_samples
        if count <= 0:
            raise ValueError("official source batches must contain samples")
        history, horizon = spec.history_length, spec.prediction_horizon
        shapes = {
            "history_obs": (count, history, 64),
            "history_act": (count, history, 29),
            "history_valid_mask": (count, history),
            "current_command": (count, 7),
            "target_actions": (count, horizon, 29),
            "target_next_observations": (count, horizon, 64),
            "trajectory_target_actions": (count, horizon, 29),
            "trajectory_target_next_observations": (count, horizon, 64),
            "trajectory_target_valid": (count,),
        }
        mask_names = {"history_valid_mask", "trajectory_target_valid"}
        device = self.current_command.device
        for name, shape in shapes.items():
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"official {name} must have shape {shape}")
            expected_dtype = torch.bool if name in mask_names else torch.float32
            if value.dtype != expected_dtype or value.device != device:
                raise ValueError(f"official {name} has inconsistent dtype/device")
            if name not in mask_names and not bool(torch.isfinite(value).all()):
                raise ValueError(f"official {name} must be finite")
        mask = self.history_valid_mask
        if not bool(mask[:, -1].all()) or bool((mask[:, :-1] & ~mask[:, 1:]).any()):
            raise ValueError("official history mask must be a left-padded valid suffix")
        for name in ("history_obs", "history_act"):
            if bool((getattr(self, name)[~mask] != 0).any()):
                raise ValueError(f"official {name} padding must be zero")
        return self


def validate_official_fada_preprocessing(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Bind actual run preprocessing once, without guessing missing magnitudes."""

    terms = ("base_ang_vel", "dof_pos", "dof_vel", "projected_gravity")
    if not isinstance(payload, Mapping):
        raise ValueError("official preprocessing must be a mapping")
    if tuple(payload.get("term_order", ())) != terms or payload.get("noise_enabled") is not False:
        raise ValueError("official preprocessing requires canonical terms and noise-free extraction")
    resolved: dict[str, Any] = {"term_order": list(terms), "noise_enabled": False}
    for field_name in ("term_scale", "term_noise"):
        values = payload.get(field_name)
        if not isinstance(values, Mapping) or set(values) != set(terms):
            raise ValueError(f"official preprocessing {field_name} must declare all compact terms")
        resolved_values: dict[str, float] = {}
        for term in terms:
            value = values[term]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or (field_name == "term_noise" and value < 0)
            ):
                raise ValueError(f"invalid official preprocessing {field_name}.{term}")
            resolved_values[term] = float(value)
        resolved[field_name] = resolved_values
    if payload.get("term_noise_effective", {term: 0.0 for term in terms}) != {
        term: 0.0 for term in terms
    }:
        raise ValueError("official extraction noise must remain disabled")
    resolved["term_noise_effective"] = {term: 0.0 for term in terms}
    return resolved


@dataclass(frozen=True)
class OfficialFADACollectRequest(DaggerCollectRequest):
    source: Literal["optimal", "suboptimal", "online"]
    weak_checkpoint: str | None = None
    collect_steps: int | None = None

    def __post_init__(self) -> None:
        if self.scenario != OFFICIAL_FADA_SCENARIO:
            raise ValueError("official source request requires its explicit scenario")
        if self.source not in {"optimal", "suboptimal", "online"}:
            raise ValueError("unknown official physical source")
        if (self.source == "suboptimal") != bool(self.weak_checkpoint):
            raise ValueError("only suboptimal requests require a weak checkpoint")
        if self.collect_steps is not None and (
            isinstance(self.collect_steps, bool)
            or not isinstance(self.collect_steps, int) or self.collect_steps <= 0
        ):
            raise ValueError("collect_steps must be a positive integer or None")


@dataclass(frozen=True)
class OfficialFADARawTransitionBatch:
    """Chronological, unsampled upstream transitions; no history-window truncation."""

    obs: torch.Tensor
    current_command: torch.Tensor
    executed_act: torch.Tensor
    expert_act: torch.Tensor
    expert_chunk: torch.Tensor
    expert_future_obs_chunk: torch.Tensor
    has_expert_chunk: torch.Tensor
    has_expert_future_obs_chunk: torch.Tensor
    strict_label_valid: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    env_id: torch.Tensor
    episode_id: torch.Tensor

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> OfficialFADARawTransitionBatch:
        names = {item.name for item in fields(cls)}
        if set(mapping) != names:
            raise ValueError("official raw transition fields mismatch")
        return cls(**{name: torch.as_tensor(mapping[name]) for name in names})

    @property
    def num_samples(self) -> int:
        return int(self.obs.shape[0])

    def validate(self, spec: OfficialFADACollectionSpec) -> OfficialFADARawTransitionBatch:
        if self.obs.ndim != 2 or self.num_samples <= 0:
            raise ValueError("official raw transitions must be a nonempty matrix")
        count, horizon = self.num_samples, spec.prediction_horizon
        shapes = {
            "obs": (count, 64), "current_command": (count, 7),
            "executed_act": (count, 29), "expert_act": (count, 29),
            "expert_chunk": (count, horizon, 29),
            "expert_future_obs_chunk": (count, horizon, 64),
        }
        boolean = {"has_expert_chunk", "has_expert_future_obs_chunk", "strict_label_valid", "done"}
        for item in fields(self):
            name, value = item.name, getattr(self, item.name)
            expected_shape = shapes.get(name, (count,))
            dtype = torch.bool if name in boolean else (
                torch.int32 if name == "env_id" else torch.int64 if name == "episode_id" else torch.float32
            )
            if tuple(value.shape) != expected_shape or value.dtype != dtype or value.device.type != "cpu":
                raise ValueError(f"official raw {name} has invalid shape, dtype or device")
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"official raw {name} must be finite")
        if bool((self.env_id < 0).any()) or bool((self.episode_id < 0).any()):
            raise ValueError("official trajectory IDs must be non-negative")
        return self
