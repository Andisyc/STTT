"""G1 joystick locomotion environments."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.envs.locomotion.common.commands import (
    Commands,
)
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.g1.base import G1BaseCfg
from unilab.envs.locomotion.g1.calibration_fault import (
    G1ActionExecutionFaultConfig,
)
from unilab.envs.locomotion.g1.fada_privileged import (
    DOF_POSITION_BIAS_LIMIT_RAD,
    TORQUE_RFI_FRACTION,
    G1FADAPrivilegedObservationConfig,
)


@dataclass
class G1ActuatorStrengthConfig:
    """Optional fixed per-actuator position-servo gain multipliers.

    This is a gain-based approximation of actuator effectiveness for controlled
    simulation experiments. It does not model a measured torque-limit curve.
    """

    enabled: bool = False
    multipliers: list[float] = field(default_factory=list)
    sampling_mode: str = "fixed"
    # Read-only legacy metadata: random sampling and its own curriculum were
    # removed. Keeping these fields lets disabled old checkpoint configs load.
    candidate_actuator_indices: list[int] = field(default_factory=list)
    multiplier_range: list[float] = field(default_factory=lambda: [1.0, 1.0])
    nominal_probability: float = 0.0
    include_in_critic_obs: bool = False
    curriculum_enabled: bool = False
    curriculum_multiplier_lows: list[float] = field(default_factory=list)
    curriculum_nominal_probabilities: list[float] = field(default_factory=list)
    curriculum_promote_threshold: float = 800.0
    curriculum_demote_threshold: float = 500.0
    curriculum_update_episodes: int = 1024
    # Physical DR schedule: independent of enabled/curriculum_enabled above.
    group_curriculum_enabled: bool = False
    group_curriculum_scales: list[float] = field(default_factory=list)
    curriculum_progress_mode: str = "episode_quality"
    curriculum_iteration_boundaries: list[int] = field(default_factory=list)
    curriculum_max_termination_rate: float = 0.1
    curriculum_brake_cooldown_steps: int = 100
    curriculum_recovery_hold_steps: int = 500


@dataclass
class G1DomainRandConfig(DomainRandConfig):
    randomize_kp: bool = True
    kp_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])

    randomize_kd: bool = True
    kd_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])

    actuator_strength: G1ActuatorStrengthConfig = field(default_factory=G1ActuatorStrengthConfig)
    randomize_dof_position_bias: bool = False
    dof_position_bias_range: list[float] = field(
        default_factory=lambda: [-DOF_POSITION_BIAS_LIMIT_RAD, DOF_POSITION_BIAS_LIMIT_RAD]
    )
    torque_rfi_fraction: float = 0.0
    randomize_control_delay: bool = False
    com_offset_y: list[float] = field(default_factory=lambda: [-0.05, 0.05])
    com_offset_z: list[float] = field(default_factory=lambda: [-0.05, 0.05])
    fada_push_interval_seconds: float = 7.5
    fada_max_push_velocity: float = 0.8

    def validate(self) -> None:
        ranges = (
            (self.randomize_kp, "kp_multiplier_range", True),
            (self.randomize_kd, "kd_multiplier_range", True),
            (self.randomize_ground_friction, "ground_friction_multiplier_range", True),
            (self.randomize_body_mass, "body_mass_multiplier_range", True),
            (self.randomize_base_mass, "added_mass_range", False),
            (self.randomize_dof_armature, "dof_armature_multiplier_range", True),
            (self.randomize_dof_position_bias, "dof_position_bias_range", False),
            *((self.random_com, f"com_offset_{axis}", False) for axis in "xyz"),
        )
        for enabled, name, positive in ranges:
            if not enabled:
                continue
            bounds = np.asarray(getattr(self, name), dtype=np.float64)
            if (bounds.shape != (2,) or not np.isfinite(bounds).all()
                    or bounds[0] > bounds[1] or (positive and bounds[0] <= 0.0)):
                raise ValueError(f"env.domain_rand.{name} requires finite ordered bounds")
        if self.randomize_gravity:
            bounds = np.asarray(self.gravity_range, dtype=np.float64)
            if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[0] > bounds[1]):
                raise ValueError("env.domain_rand.gravity_range requires ordered finite 3D bounds")


@dataclass
class InitState:
    pos = [0.0, 0.0, 0.754]


@dataclass
class GaitConstraintConfig:
    enabled: bool = False
    command_xy_threshold: float = 0.05
    command_yaw_threshold: float = 0.05
    height_weight: float = 1.0
    contrast_weight: float = 1.0
    contact_weight: float = 1.0
    epsilon: float = 0.02
    penalty_scale: float = 1.0
    apply_in_stand_mode: bool = False
    apply_when_tracking: bool = False
    tracking_threshold: float = 0.3
    freeze_phase_in_stand_mode: bool = False
    stand_phase: list[float] = field(default_factory=lambda: [math.pi, math.pi])


@dataclass
class G1CommandGatedPhaseContactConfig:
    enabled: bool = False
    command_xy_dead_zone: float = 0.1
    command_yaw_dead_zone: float = 0.1
    linear_intensity_span: float = 0.9
    yaw_intensity_span: float = 0.7
    min_frequency: float = 0.7
    max_frequency: float = 1.5
    duty_factor: float = 0.55
    contact_force_threshold: float = 1.0
    force_balance_epsilon: float = 1.0e-6
    startup_phase: list[float] = field(default_factory=lambda: [0.0, math.pi])
    stand_phase: list[float] = field(default_factory=lambda: [math.pi, math.pi])

    def validate(self) -> None:
        non_negative = ("command_xy_dead_zone", "command_yaw_dead_zone")
        positive = (
            "linear_intensity_span",
            "yaw_intensity_span",
            "min_frequency",
            "max_frequency",
            "contact_force_threshold",
            "force_balance_epsilon",
        )
        for name in non_negative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_frequency < self.min_frequency:
            raise ValueError("max_frequency must be at least min_frequency")
        if not 0.5 < float(self.duty_factor) < 1.0:
            raise ValueError("duty_factor must be in (0.5, 1.0)")
        for name in ("startup_phase", "stand_phase"):
            phase = np.asarray(getattr(self, name), dtype=np.float64)
            if phase.shape != (2,) or not np.isfinite(phase).all():
                raise ValueError(f"{name} must contain two finite radian phases")


@dataclass
class RewardModeConfig:
    enabled: bool = False
    standing_enabled: bool = False
    balance_common_terms: list[str] = field(default_factory=list)
    stand_terms: list[str] = field(default_factory=list)
    stand_recovery_terms: list[str] = field(default_factory=list)
    walk_terms: list[str] = field(default_factory=list)
    stand_scale_overrides: dict[str, float] = field(default_factory=dict)
    stand_recovery_scale_overrides: dict[str, float] = field(default_factory=dict)
    walk_scale_overrides: dict[str, float] = field(default_factory=dict)


def normalize_g1_gait_reward(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve historical reward names and signs at the configuration boundary.

    Work on copies: the raw Hydra configuration remains the checkpoint identity.
    Explicit curriculum membership preserves old positive-weight, signed costs.
    """
    result = dict(config)
    scales = dict(result.get("scales", {}))
    if result.get("penalty_curriculum_terms") is None:
        result["penalty_curriculum_terms"] = [name for name, scale in scales.items() if scale < 0]
    aliases = {
        "original_command_height_v1": ("relative_command_height", None, 1.0),
        "command_height_v1": ("filtered_command_height", "feet_phase", -1.0),
        "command_height_v2": ("command_height", "feet_phase", -0.5),
        "command_height_v3": ("phase_height", None, 1.0),
        "phase_contact_v1": ("command_gated_contact", "phase_contact", -1.0),
        "fixed_phase_contact_v1": ("fixed_contact", "phase_contact", -1.0),
        "fixed_phase_contact_v2": ("fixed_contact", None, 1.0),
    }
    mode = result.get("feet_phase_mode", "legacy")
    if mode == "fixed_phase_contact_v2":
        result["feet_orientation_stance_only"] = True
    canonical, term, multiplier = aliases.get(mode, (mode, None, 1.0))
    result["feet_phase_mode"] = canonical
    if term is not None:
        if term in scales:
            scales[term] *= multiplier
        # Mode dispatch has its own weights; normalize those at the same boundary.
        mode_cfg = result.get("mode")
        if isinstance(mode_cfg, dict):
            mode_cfg = dict(mode_cfg)
            for key in (
                "stand_scale_overrides", "stand_recovery_scale_overrides", "walk_scale_overrides"
            ):
                if key in mode_cfg:
                    overrides = dict(mode_cfg[key])
                    if term in overrides:
                        overrides[term] *= multiplier
                    mode_cfg[key] = overrides
            result["mode"] = mode_cfg
    result["scales"] = scales
    return result


@dataclass
class G1RewardConfig:
    scales: dict[str, float]
    tracking_sigma: float
    gait_frequency: float
    feet_phase_swing_height: float
    feet_phase_tracking_sigma: float
    base_height_target: float
    min_base_height: float
    max_tilt_deg: float
    min_forward_speed_for_gait_reward: float = 0.0
    min_planar_command_speed_for_gait_reward: float = 0.0
    feet_phase_mode: str = "legacy"
    feet_orientation_stance_only: bool = False
    penalty_curriculum_terms: list[str] | None = None
    tracking_lin_error_scale: float | None = None
    tracking_lin_relative_blend: bool = False
    tracking_lin_min_command: float = 0.05
    tracking_lin_relative_until: float = 0.25
    tracking_lin_absolute_from: float = 0.5
    under_speed_mode: str = "forward"
    under_speed_min_command: float = 0.05
    feet_phase_command_speed_scale: float = 0.3
    feet_phase_turn_length: float = 0.3
    feet_phase_settling_tau: float = 0.15
    feet_phase_height_scale: float = 0.09
    stand_recovery_lin_vel_xy_threshold: float = 0.2
    stand_recovery_tilt_deg_threshold: float = 8.0
    close_feet_threshold: float = 0.15
    straight_line_lateral_tolerance_m: float = 0.10
    straight_line_yaw_tolerance_rad: float = 0.10
    straight_line_corridor_max_violation: float | None = None
    stand_feet_x_target: float = 0.0
    stand_feet_y_width_target: float = 0.21
    stand_base_feet_center_x_target: float = 0.0
    stand_base_feet_center_y_target: float = 0.0
    stand_foot_contact_balance_epsilon: float = 1.0e-6
    stand_support_height_margin: float = 0.02
    gait_constraint: GaitConstraintConfig | dict[str, Any] = field(
        default_factory=GaitConstraintConfig
    )
    mode: RewardModeConfig | dict[str, Any] = field(default_factory=RewardModeConfig)
    pose_weights: list[float] = field(
        default_factory=lambda: [
            0.01,
            1.0,
            5.0,
            0.01,
            5.0,
            5.0,
            0.01,
            1.0,
            5.0,
            0.01,
            5.0,
            5.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
        ]
    )

    def __post_init__(self) -> None:
        # Normalize once, before reward dispatch or curriculum sees the weights.
        normalized = normalize_g1_gait_reward({
            "feet_phase_mode": self.feet_phase_mode,
            "feet_orientation_stance_only": self.feet_orientation_stance_only,
            "penalty_curriculum_terms": self.penalty_curriculum_terms,
            "scales": self.scales,
            "mode": asdict(self.mode) if isinstance(self.mode, RewardModeConfig) else self.mode,
        })
        for name, value in normalized.items():
            setattr(self, name, value)
        if isinstance(self.gait_constraint, dict):
            self.gait_constraint = GaitConstraintConfig(**self.gait_constraint)
        if isinstance(self.mode, dict):
            self.mode = RewardModeConfig(**self.mode)
        if self.feet_phase_mode not in {
            "legacy",
            "relative_command_height",
            "filtered_command_height",
            "command_height",
            "phase_height",
            "absolute_phase_height",
            "command_gated_contact",
            "fixed_contact",
        }:
            raise ValueError("unsupported feet_phase_mode")
        if self.feet_phase_mode == "relative_command_height":
            for name in (
                "feet_phase_swing_height",
                "feet_phase_command_speed_scale",
                "feet_phase_turn_length",
                "feet_phase_tracking_sigma",
            ):
                if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                    raise ValueError(f"{name} must be finite and positive")
        if self.feet_phase_mode in {"filtered_command_height", "command_height", "phase_height"}:
            for name in (
                "feet_phase_command_speed_scale",
                "feet_phase_turn_length",
                "feet_phase_settling_tau",
                "feet_phase_height_scale",
                "feet_phase_swing_height",
            ):
                value = getattr(self, name)
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{name} must be finite and positive")
        for name in ("tracking_sigma", "gait_frequency", "feet_phase_tracking_sigma"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.min_planar_command_speed_for_gait_reward)
            or self.min_planar_command_speed_for_gait_reward < 0
        ):
            raise ValueError(
                "min_planar_command_speed_for_gait_reward must be finite and non-negative"
            )
        if self.tracking_lin_error_scale is not None and (
            not math.isfinite(self.tracking_lin_error_scale) or self.tracking_lin_error_scale <= 0
        ):
            raise ValueError("tracking_lin_error_scale must be finite and positive")
        if self.tracking_lin_relative_blend:
            bounds = (self.tracking_lin_min_command, self.tracking_lin_relative_until, self.tracking_lin_absolute_from)
            if not all(math.isfinite(value) for value in bounds) or not 0 < bounds[0] <= bounds[1] < bounds[2]:
                raise ValueError("tracking blend requires 0 < min_command <= relative_until < absolute_from")
            if self.tracking_lin_error_scale is not None:
                raise ValueError("tracking blend requires tracking_lin_error_scale=null")
        if self.under_speed_mode not in {"forward", "command_direction"}:
            raise ValueError("unsupported under_speed_mode")
        if not math.isfinite(self.under_speed_min_command) or self.under_speed_min_command <= 0:
            raise ValueError("under_speed_min_command must be finite and positive")
        if self.straight_line_corridor_max_violation is not None and (
            not math.isfinite(self.straight_line_corridor_max_violation)
            or self.straight_line_corridor_max_violation <= 0
        ):
            raise ValueError("straight_line_corridor_max_violation must be finite and positive")
        if any(not math.isfinite(float(value)) for value in self.scales.values()):
            raise ValueError("reward scales must be finite")
        cost_terms = []
        if self.feet_phase_mode in {"filtered_command_height", "command_height"}:
            cost_terms.append("feet_phase")
        if self.feet_phase_mode in {"fixed_contact", "command_gated_contact"}:
            cost_terms.append("phase_contact")
        if any(self.scales.get(name, 0.0) > 0.0 for name in cost_terms):
            raise ValueError("gait costs require non-positive weights")
        if set(self.penalty_curriculum_terms or ()) - self.scales.keys():
            raise ValueError("penalty_curriculum_terms must name configured reward terms")


@dataclass
class G1WalkLegacyRewardConfig(G1RewardConfig):
    pass


@dataclass
class CurriculumConfig:
    enabled: bool = False
    initial_scale: float = 0.5
    min_scale: float = 0.5
    max_scale: float = 1.0
    level_down_threshold: float = 150.0
    level_up_threshold: float = 750.0
    degree: float = 0.001


@dataclass
class ForwardProgressTerminationConfig:
    enabled: bool = False
    grace_steps: int = 50
    min_command_forward_speed: float = 0.1
    min_average_forward_speed: float = 0.2


@dataclass
class G1RuntimePayloadConfig:
    """Rigid box payload; position and half sizes are in the target body's local axes."""

    enabled: bool = False
    body_name: str = "torso_link"
    added_mass_kg: float = 0.0
    attachment_pos_m: list[float] = field(default_factory=lambda: [0.004, 0.22, 0.15])
    half_size_m: list[float] = field(default_factory=lambda: [0.08, 0.06, 0.12])
    attach_time_s: float = 10.0

    def validate(self) -> None:
        if not self.enabled:
            return
        if not str(self.body_name).strip():
            raise ValueError("env.runtime_payload.body_name must be non-empty")
        if not math.isfinite(float(self.added_mass_kg)) or float(self.added_mass_kg) <= 0.0:
            raise ValueError("env.runtime_payload.added_mass_kg must be finite and positive")
        position = np.asarray(self.attachment_pos_m, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("env.runtime_payload.attachment_pos_m must be a finite 3D vector")
        half_size = np.asarray(self.half_size_m, dtype=np.float64)
        if half_size.shape != (3,) or not np.isfinite(half_size).all() or np.any(half_size <= 0.0):
            raise ValueError("env.runtime_payload.half_size_m must be a finite positive 3D vector")
        if not math.isfinite(float(self.attach_time_s)) or float(self.attach_time_s) < 0.0:
            raise ValueError("env.runtime_payload.attach_time_s must be finite and non-negative")


@dataclass
class G1WalkCommandsConfig(Commands):
    """Optional command-only dead zone, independent of gait scheduling."""

    dead_zone_enabled: bool = False
    dead_zone_xy: float = 0.1
    dead_zone_yaw: float = 0.1
    rel_straight_envs: float = 0.0

    def validate(self) -> None:
        ratios = np.asarray(
            [self.rel_standing_envs, self.rel_transition_envs, self.rel_straight_envs]
        )
        if not np.isfinite(ratios).all() or np.any(ratios < 0.0) or ratios.sum() > 1.0:
            raise ValueError(
                "standing, transition, and straight command ratios must be non-negative "
                "and sum to <= 1"
            )
        for name in ("resampling_time", "dead_zone_xy", "dead_zone_yaw"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"commands.{name} must be finite and non-negative")
        for name in ("vel_limit", "transition_vel_limit"):
            limits = np.asarray(getattr(self, name), dtype=np.float64)
            if (
                limits.shape != (2, 3)
                or not np.isfinite(limits).all()
                or np.any(limits[0] > limits[1])
            ):
                raise ValueError(f"commands.{name} must contain ordered finite 3D bounds")


@dataclass
class G1WalkEnvCfg(G1BaseCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")
        )
    )
    max_episode_seconds: float = 20.0
    init_state: InitState = field(default_factory=InitState)
    commands: G1WalkCommandsConfig = field(default_factory=G1WalkCommandsConfig)
    reward_config: G1RewardConfig | None = None
    domain_rand: G1DomainRandConfig = field(default_factory=G1DomainRandConfig)
    gait_phase_enabled: bool = True
    gait_phase_init_mode: str = "offset_phase"
    gait_clock_mode: str = "continuous"
    command_gated_phase_contact: G1CommandGatedPhaseContactConfig = field(
        default_factory=G1CommandGatedPhaseContactConfig
    )
    mode_observation: bool = False
    reset_base_qvel_limit: float = 0.5
    standing_reset_base_qvel_limit: float = 0.0
    stand_action_authority: bool = False
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    forward_progress_termination: ForwardProgressTerminationConfig = field(
        default_factory=ForwardProgressTerminationConfig
    )
    action_execution_fault: G1ActionExecutionFaultConfig | None = None
    fada_privileged_observation: G1FADAPrivilegedObservationConfig = field(
        default_factory=G1FADAPrivilegedObservationConfig
    )
    runtime_payload: G1RuntimePayloadConfig = field(default_factory=G1RuntimePayloadConfig)

    def validate(self) -> None:
        super().validate()
        self.commands.validate()
        self.domain_rand.validate()
        self.command_gated_phase_contact.validate()
        self.runtime_payload.validate()
        if self.runtime_payload.enabled and (
            self.domain_rand.randomize_base_mass
            or self.domain_rand.randomize_body_mass
            or self.domain_rand.random_com
        ):
            raise ValueError(
                "env.runtime_payload is deterministic and forbids "
                "domain_rand.randomize_base_mass, randomize_body_mass, and random_com"
            )
        if self.gait_clock_mode not in {"continuous", "command_fixed"}:
            raise ValueError("unsupported gait_clock_mode")
        if self.gait_clock_mode == "command_fixed" and (
            not self.gait_phase_enabled
            or self.gait_phase_init_mode != "offset_phase"
            or self.command_gated_phase_contact.enabled
        ):
            raise ValueError("command_fixed requires enabled offset phase and the v024 clock disabled")
        scales = self.reward_config.scales if self.reward_config is not None else {}
        phase_terms = ("phase_contact", "feet_phase", "feet_phase_contrast", "feet_phase_contact")
        if not self.gait_phase_enabled and any(scales.get(name, 0.0) != 0.0 for name in phase_terms):
            raise ValueError("nonzero phase Reward conflicts with env.gait_phase_enabled=false")
        if scales.get("phase_contact", 0.0) != 0.0 and self.reward_config.feet_phase_mode not in {
            "fixed_contact", "command_gated_contact"
        }:
            raise ValueError("reward.scales.phase_contact requires fixed_contact or command_gated_contact mode")
        if scales.get("null_torque_relaxation", 0.0) != 0.0 and not self.command_gated_phase_contact.enabled:
            raise ValueError("null_torque_relaxation requires command_gated_phase_contact.enabled for torque provenance")
        if (
            self.reward_config is not None
            and self.reward_config.feet_orientation_stance_only
            and scales.get("penalty_feet_ori", 0.0) != 0.0
            and not self.gait_phase_enabled
        ):
            raise ValueError("stance-only foot orientation requires an enabled gait phase")
        if (
            self.reward_config is not None
            and self.reward_config.feet_phase_mode == "fixed_contact"
            and scales.get("phase_contact", 0.0) != 0.0
        ):
            if self.command_gated_phase_contact.enabled:
                raise ValueError("fixed phase contact forbids the command-gated state machine")
            if not self.gait_phase_enabled or self.gait_phase_init_mode != "offset_phase":
                raise ValueError("fixed phase contact requires enabled offset_phase")
        if self.command_gated_phase_contact.enabled:
            if not self.gait_phase_enabled or self.gait_phase_init_mode != "command_gated":
                raise ValueError(
                    "command-gated phase contact requires enabled command_gated gait phase"
                )
            if (
                self.reward_config is None
                or self.reward_config.feet_phase_mode != "command_gated_contact"
            ):
                raise ValueError(
                    "command-gated phase contact requires reward feet_phase_mode=command_gated_contact"
                )
        if self.action_execution_fault is not None:
            self.action_execution_fault.validate()
        if self.fada_privileged_observation.enabled:
            if self.fada_privileged_observation.schema != "g1_fada_privileged_v1":
                raise ValueError("unsupported FADA privileged observation schema")
            bias_range = np.asarray(self.domain_rand.dof_position_bias_range, dtype=np.float64)
            if (
                bias_range.shape != (2,)
                or bias_range[0] < -DOF_POSITION_BIAS_LIMIT_RAD
                or bias_range[1] > DOF_POSITION_BIAS_LIMIT_RAD
            ):
                raise ValueError(
                    "FADA DoF position bias range exceeds the confirmed moderate limit"
                )
            if not 0.0 <= float(self.domain_rand.torque_rfi_fraction) <= TORQUE_RFI_FRACTION:
                raise ValueError("FADA torque RFI fraction exceeds the confirmed moderate limit")
            if self.domain_rand.push_robots:
                if not 5.0 <= float(self.domain_rand.fada_push_interval_seconds) <= 10.0:
                    raise ValueError("FADA push interval must remain in [5, 10] seconds")
                if not 0.1 <= float(self.domain_rand.fada_max_push_velocity) <= 1.5:
                    raise ValueError("FADA max push velocity must remain in [0.1, 1.5] m/s")
                if not self.domain_rand.push_body_name:
                    raise ValueError("FADA velocity push requires push_body_name")


def _walk_curriculum() -> CurriculumConfig:
    return CurriculumConfig(
        enabled=True,
        initial_scale=0.5,
        min_scale=0.5,
        max_scale=1.0,
        level_down_threshold=150.0,
        level_up_threshold=750.0,
        degree=0.001,
    )


@dataclass
class G1WalkControlConfig:
    action_scale: float = 1.0
    simulate_action_latency: bool = False


@dataclass
class G1WalkRewardConfig(G1RewardConfig):
    """Align reward weights with holosoma G1 walking."""


@registry.envcfg("G1WalkFlat")
@dataclass
class G1WalkFlatCfg(G1WalkEnvCfg):
    reward_config: G1WalkRewardConfig | None = None
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")
        )
    )
    control_config: G1WalkControlConfig = field(default_factory=G1WalkControlConfig)  # type: ignore[assignment]
    curriculum: CurriculumConfig = field(default_factory=_walk_curriculum)


@registry.envcfg("G1WalkHeight")
@dataclass
class G1WalkHeightCfg(G1WalkFlatCfg):
    pass


@registry.envcfg("G1StandStill")
@dataclass
class G1StandStillCfg(G1WalkFlatCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml"),
            fragment_files=[str(ASSETS_ROOT_PATH / "robots" / "g1" / "stand_support_task.xml")],
        )
    )


@registry.envcfg("G1StandHeight")
@dataclass
class G1StandHeightCfg(G1StandStillCfg):
    pass


@registry.envcfg("G1WalkRough")
@dataclass
class G1WalkRoughCfg(G1WalkFlatCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_rough.xml")
        )
    )
