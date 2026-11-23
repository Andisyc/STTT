"""Explicit budgets and identities for continuous physical-model/policy training."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

from omegaconf import OmegaConf


def positive_count(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def positive_scale(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


@dataclass(frozen=True)
class JointSchedule:
    pretrain_wm_updates: int
    online_iterations: int
    actor_updates_per_iteration: int
    wm_updates_per_iteration: int
    checkpoint_every: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            positive_count(getattr(self, name), f"schedule.{name}")


@dataclass(frozen=True)
class CaptureSchedule:
    initial: bool
    num_envs: int
    initial_min_transitions: int
    validation_episodes: int
    steps_per_iteration: int
    max_episode_steps: int
    window_steps: int
    residual_amplitude_rad: Any
    risk_tilt_limit_rad: float
    command_resampling_s: float
    command_ranges: Any
    conditions: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        if type(self.initial) is not bool:
            raise ValueError("capture.initial must be boolean")
        for name in ("num_envs", "initial_min_transitions", "validation_episodes", "steps_per_iteration", "max_episode_steps", "window_steps"):
            positive_count(getattr(self, name), f"capture.{name}")
        if self.steps_per_iteration % self.window_steps or self.max_episode_steps % self.window_steps:
            raise ValueError("collection iterations and episode limits must contain whole residual windows")
        positive_scale(self.risk_tilt_limit_rad, "capture.risk_tilt_limit_rad")
        if self.risk_tilt_limit_rad > math.pi:
            raise ValueError("risk tilt limit must not exceed pi")
        positive_scale(self.command_resampling_s, "capture.command_resampling_s")
        value = self.residual_amplitude_rad
        values = value if isinstance(value, (list, tuple)) else [value]
        if len(values) not in (1, 29):
            raise ValueError("capture residual amplitude requires a scalar or 29 values")
        for entry in values:
            if isinstance(entry, bool) or not isinstance(entry, (int, float)) or not math.isfinite(entry) or entry < 0:
                raise ValueError("set an explicit finite nonnegative capture residual amplitude")
        if not self.conditions or len(self.conditions) > self.num_envs:
            raise ValueError("provide at least one physical condition and at least one stream per condition")
        names = set()
        for condition in self.conditions:
            if set(condition) != {"name", "source_overrides"} or not isinstance(condition["name"], str) or not condition["name"]:
                raise ValueError("each capture condition requires name and source_overrides")
            if condition["name"] in names or not isinstance(condition["source_overrides"], list) or any(not isinstance(x, str) for x in condition["source_overrides"]):
                raise ValueError("condition names must be unique and overrides must be strings")
            if Path(condition["name"]).name != condition["name"] or condition["name"] in {".", ".."}:
                raise ValueError("physical condition names must be safe single path components")
            names.add(condition["name"])


@dataclass(frozen=True)
class JointTrainingConfig:
    raw: dict[str, Any]
    schedule: JointSchedule
    capture: CaptureSchedule
    replay_capacity: int

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> JointTrainingConfig:
        raw = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else dict(cfg)
        expected = {"output_dir", "policy_checkpoint", "profile_path", "device", "seed", "source_overrides", "wm", "actor", "evaluation", "capture", "schedule", "replay", "resume_checkpoint", "calibration"}
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError("use latent_pretraining.yaml; refinement_cycles/refine_* were replaced by schedule and replay")
        if type(raw["seed"]) is not int or raw["seed"] < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(raw["source_overrides"], list) or any(not isinstance(x, str) for x in raw["source_overrides"]):
            raise ValueError("source_overrides must be strings")
        schedule = JointSchedule(**raw["schedule"])
        capture_raw = dict(raw["capture"])
        capture_raw["conditions"] = tuple(capture_raw["conditions"])
        capture = CaptureSchedule(**capture_raw)
        if set(raw["replay"]) != {"capacity_transitions"}:
            raise ValueError("replay requires capacity_transitions")
        capacity = positive_count(raw["replay"]["capacity_transitions"], "replay.capacity_transitions")
        if capacity < capture.num_envs * capture.max_episode_steps:
            raise ValueError("replay capacity must retain every active stream's maximum-length episode")
        horizon = positive_count(raw["wm"]["training"]["rollout_windows"], "wm.training.rollout_windows") * capture.window_steps
        if horizon != capture.steps_per_iteration:
            raise ValueError("WM forecast and collection block must cover the same complete residual windows")
        if positive_count(raw["actor"]["training"]["horizon_windows"], "actor.training.horizon_windows") * capture.window_steps != horizon:
            raise ValueError("actor and WM forecast horizons must agree")
        if raw["wm"]["training"]["max_updates"] != schedule.pretrain_wm_updates or raw["actor"]["training"]["max_updates"] != schedule.online_iterations:
            raise ValueError("joint execution budgets must be owned by schedule (use YAML interpolation)")
        return cls(cast(dict[str, Any], raw), schedule, capture, capacity)


@dataclass(frozen=True)
class TrainingVersions:
    """Counters are optimizer steps, never hashes computed on the control path."""
    actor: int = 0
    world_model: int = 0

    def __post_init__(self) -> None:
        if any(type(x) is not int or x < 0 for x in (self.actor, self.world_model)):
            raise ValueError("training versions must be nonnegative optimizer-step counts")
