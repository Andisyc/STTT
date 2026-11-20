"""Cold-bound execution metadata for the existing g1_fada_state_v2 policy."""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

LEGACY_CONTRACT = "g1_fada_state_v2"
LEGACY_LAYOUT = (("gyro", 3), ("gravity", 3), ("dof_pos", 29), ("dof_vel", 29), ("gait_phase", 2))
RAW_LAYOUT = (("gyro", 3), ("gravity", 3), ("dof_pos", 29), ("dof_vel", 29), ("actions", 29), ("command", 3), ("gait_phase", 2))


def _vector(value: Any, name: str, *, positive: bool = False) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 29:
        raise ValueError(f"legacy source {name} requires 29 values")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or positive and x <= 0 for x in value):
        raise ValueError(f"invalid legacy source {name}")
    return tuple(float(x) for x in value)


@dataclass(frozen=True)
class LegacyFADAPlaybackProfile:
    source_profile: Mapping[str, Any]
    joint_names: tuple[str, ...]
    default_joint_pos: tuple[float, ...]
    action_scale: tuple[float, ...]
    stiffness: tuple[float, ...]
    damping: tuple[float, ...]
    ctrl_dt: float
    gait_frequency: float
    gait_clock_mode: str
    # The legacy playback/env path does not clip normalized joint commands.
    # Backend actuator force limits remain in the physical simulator.
    clip_actions: None = None

    def source_indices(self, target_joint_names: Sequence[str]) -> tuple[int, ...]:
        names = tuple(target_joint_names)
        if len(names) != 29 or len(set(names)) != 29 or set(names) != set(self.joint_names):
            raise ValueError("legacy source and target joint identities differ")
        return tuple(names.index(name) for name in self.joint_names)

    @classmethod
    def from_source(cls, raw: Mapping[str, Any]) -> LegacyFADAPlaybackProfile:
        if not isinstance(raw, Mapping):
            raise ValueError("legacy WM execution requires checkpoint runtime_config.source_profile")
        source = json.loads(json.dumps(dict(raw), allow_nan=False))
        if source.get("kind") != "unilab_oracle" or source.get("version") != 1 or source.get("student_observation_contract") != LEGACY_CONTRACT:
            raise ValueError("unsupported embedded legacy source profile identity")
        control, env = source.get("control_contract"), source.get("environment_config")
        if not isinstance(control, dict) or not isinstance(env, dict):
            raise ValueError("embedded legacy profile requires control_contract and environment_config")
        if control.get("observation_layout") != [list(field) for field in RAW_LAYOUT]:
            raise ValueError("embedded actor layout does not match g1_fada_state_v2 projection")
        names = control.get("actuated_joint_names")
        if not isinstance(names, list) or len(names) != 29 or len(set(names)) != 29 or any(not isinstance(x, str) or not x for x in names):
            raise ValueError("embedded legacy profile requires 29 distinct actuator names")
        dt = control.get("ctrl_dt")
        frequency = env.get("reward_config", {}).get("gait_frequency")
        for name, value in (("ctrl_dt", dt), ("gait_frequency", frequency)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"embedded legacy {name} must be positive and finite")
        if env.get("ctrl_dt") != dt or env.get("sim_dt") != control.get("sim_dt"):
            raise ValueError("embedded control and environment physical periods disagree")
        clock = env.get("gait_clock_mode")
        if clock not in {"continuous", "command_fixed"} or env.get("gait_phase_enabled") is not True:
            raise ValueError("legacy WM rollout requires enabled continuous/command_fixed gait phase")
        if env.get("command_gated_phase_contact", {}).get("enabled", False):
            raise ValueError("command-gated dynamic-frequency phase is not the admitted fixed-clock source")
        scale = control.get("action_scale")
        if isinstance(scale, (int, float)) and not isinstance(scale, bool):
            scale = [scale] * 29
        if env.get("control_config", {}).get("action_scale") != control.get("action_scale"):
            raise ValueError("embedded source action scales disagree")
        if env.get("control_config", {}).get("simulate_action_latency", False):
            raise ValueError("legacy WM source requires the recorded immediate-action execution path")
        return cls(source, tuple(names), _vector(control.get("default_joint_pos"), "default_joint_pos"),
                   _vector(scale, "action_scale", positive=True), _vector(control.get("kp"), "kp", positive=True),
                   _vector(control.get("kd"), "kd"), float(cast(float, dt)), float(cast(float, frequency)), clock)

    @property
    def preprocessing(self) -> dict[str, Any]:
        scales = self.source_profile["environment_config"]["reward_config"]["scales"]
        walk = any(name in scales for name in ("penalty_orientation", "penalty_ang_vel_xy", "penalty_action_rate", "alive"))
        return {"observation_contract": LEGACY_CONTRACT, "observation_layout": [list(x) for x in LEGACY_LAYOUT],
                "term_scale": {"base_ang_vel": .25 if walk else 1., "projected_gravity": 1.,
                               "dof_pos": 1., "dof_vel": .05 if walk else 1.}}


def legacy_profile_from_checkpoint(payload: Mapping[str, Any]) -> LegacyFADAPlaybackProfile:
    architecture = payload.get("architecture", {})
    if (architecture.get("obs_dim"), architecture.get("action_dim"), architecture.get("command_dim"), architecture.get("observation_contract")) != (66, 29, 3, LEGACY_CONTRACT):
        raise ValueError("embedded legacy execution requires the exact 66/29/3 g1_fada_state_v2 model")
    return LegacyFADAPlaybackProfile.from_source(payload.get("runtime_config", {}).get("source_profile"))


def validate_execution_preprocessing(value: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping) and value.get("observation_contract") == LEGACY_CONTRACT:
        canonical = json.loads(json.dumps(dict(value), allow_nan=False))
        if set(canonical) != {"observation_contract", "observation_layout", "term_scale"} or canonical["observation_layout"] != [list(x) for x in LEGACY_LAYOUT]:
            raise ValueError("invalid legacy execution preprocessing layout")
        terms = canonical["term_scale"]
        if set(terms) != {"base_ang_vel", "projected_gravity", "dof_pos", "dof_vel"} or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in terms.values()):
            raise ValueError("invalid legacy observation scales")
        return canonical
    from .collection_contract import validate_official_fada_preprocessing
    return validate_official_fada_preprocessing(value)


def build_legacy_playback_env_override(profile: LegacyFADAPlaybackProfile, overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Restore the source env, allowing only explicit task-condition overrides."""
    result = deepcopy(dict(profile.source_profile["environment_config"]))
    scene = result.get("scene")
    if isinstance(scene, dict) and "task_poles" in scene:
        if scene["task_poles"] != []:
            raise ValueError("legacy scene.task_poles must be an empty list; active task poles are unsupported")
        del scene["task_poles"]
    # Policy I/O and actuator calibration belong to the checkpoint. Task
    # conditions may change without replacing those physical definitions.
    for key in ("commands", "runtime_payload", "domain_rand", "noise_config", "curriculum"):
        if key in overrides:
            result[key] = {**dict(result.get(key) or {}), **deepcopy(dict(overrides[key]))}
    for key in ("max_episode_seconds", "reset_base_qvel_limit", "standing_reset_base_qvel_limit"):
        if key in overrides:
            result[key] = deepcopy(overrides[key])
    result["position_actuator_gains"] = {"joint_names": list(profile.joint_names), "kp": list(profile.stiffness), "kd": list(profile.damping)}
    result["control_config"] = {**result["control_config"], "position_actuator_gains": result["position_actuator_gains"]}
    return result


def validate_legacy_deployment(profile: LegacyFADAPlaybackProfile, env: Any) -> None:
    if tuple(env.get_actuated_joint_names()) != profile.joint_names:
        raise ValueError("legacy playback source joint order differs from the environment")
    if not np.allclose(np.asarray(env.default_angles).reshape(-1), profile.default_joint_pos, rtol=0., atol=1e-6):
        raise ValueError("legacy playback source default joint positions differ from the environment")
    if not np.allclose(np.broadcast_to(np.asarray(env.cfg.control_config.action_scale), (29,)), profile.action_scale, rtol=0., atol=1e-7):
        raise ValueError("legacy playback source action scale differs from the environment")
    if not math.isclose(float(env.cfg.ctrl_dt), profile.ctrl_dt, rel_tol=0., abs_tol=1e-12):
        raise ValueError("legacy playback source control period differs from the environment")
