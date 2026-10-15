"""G1 joystick locomotion environments."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from etils import epath

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.augmentation import SymmetryObsLayout
from unilab.base.backend import create_backend
from unilab.base.curriculum import EpisodeLengthTracker, PenaltyCurriculum
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dr import IntervalRandomizationPlan, ResetPlan, ResetRandomizationPayload
from unilab.dr.types import RESET_TERM_KD, RESET_TERM_KP
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import np_wrap_to_pi, np_yaw_from_quat
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.commands import (
    Commands,
    apply_heading_yaw_feedback,
    sample_heading_commands,
    sample_height_commands,
    sample_velocity_commands,
    zero_small_xy_commands,
)
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.g1.action_trace import (
    G1ActionTraceSnapshot,
    action_trace_enabled,
    action_trace_interval,
    emit_g1_action_trace,
)
from unilab.envs.locomotion.g1.base import G1BaseCfg, G1BaseEnv
from unilab.envs.locomotion.g1.calibration_fault import (
    G1ActionExecutionFaultConfig,
    apply_action_execution_fault,
)
from unilab.envs.locomotion.g1.fada_privileged import (
    DOF_POSITION_BIAS_LIMIT_RAD,
    TORQUE_RFI_FRACTION,
    G1FADAPrivilegedCheckpointLayoutIdentity,
    G1FADAPrivilegedObservationConfig,
    apply_fada_pd_target_perturbation,
    build_fada_reset_info,
    build_g1_fada_checkpoint_layout_identity,
    pack_fada_runtime_observation,
)
from unilab.envs.locomotion.g1.walk_commands import (
    command_resample_mask,
    freeze_inactive_gait_phase,
    sample_g1_walk_commands,
)
from unilab.envs.locomotion.g1.walk_config import (  # noqa: F401
    CurriculumConfig,
    ForwardProgressTerminationConfig,
    G1ActuatorStrengthConfig,
    G1CommandGatedPhaseContactConfig,
    G1DomainRandConfig,
    G1RewardConfig,
    G1StandHeightCfg,
    G1StandStillCfg,
    G1WalkControlConfig,
    G1WalkEnvCfg,
    G1WalkFlatCfg,
    G1WalkHeightCfg,
    G1WalkLegacyRewardConfig,
    G1WalkRewardConfig,
    G1WalkRoughCfg,
    GaitConstraintConfig,
    InitState,
    RewardModeConfig,
)
from unilab.envs.locomotion.g1.walk_control import (
    advance_gait_phase,
    select_authority_actions,
)
from unilab.envs.locomotion.g1.walk_control_bindings import G1WalkControlBindings
from unilab.envs.locomotion.g1.walk_domain_randomization import (  # noqa: F401
    G1WalkDomainRandomizationProvider,
)
from unilab.envs.locomotion.g1.walk_math import (  # noqa: F401
    build_upper_body_pose_weights,
    compute_aggregated_foot_contact,
    compute_aggregated_foot_contact_count,
    compute_command_active_mask,
    compute_external_command_mask,
    compute_feet_phase_contact_targets,
    compute_feet_phase_height_targets,
    compute_forward_command_mask,
    compute_forward_progress_failure,
    compute_forward_speed_gate,
    compute_gait_phase_contact_violation,
    compute_gait_phase_contrast_violation,
    compute_gait_phase_height_violation,
    compute_tracking_gate,
    sample_gait_phase_pairs,
    sample_reset_base_qvel,
)
from unilab.envs.locomotion.g1.walk_observation import (
    assemble_walk_observation,
    build_obs_groups_spec,
)
from unilab.envs.locomotion.g1.walk_observation_bindings import G1WalkObservationBindings
from unilab.envs.locomotion.g1.walk_reward import (
    normalized_corridor_violation,
    resolve_stand_height_target,
    stand_action_l2,
    stand_contact_balance_l1,
    stand_dof_vel_l2,
    stand_fall_l2,
    stand_height_deficit_l1,
    stand_height_margin_l2,
    stand_lin_vel_xy_l2,
    stand_still_l1,
    stand_tilt_l2,
    stand_tilt_margin_l2,
    stand_yaw_vel_l2,
)
from unilab.envs.locomotion.g1.walk_reward_bindings import G1WalkRewardBindings
from unilab.envs.locomotion.g1.walk_runtime_bindings import G1WalkRuntimeBindings
from unilab.envs.locomotion.g1.walk_runtime_payload import G1RuntimePayload

LEFT_FOOT_CONTACT_SENSORS = [f"left_foot_contact_{i}" for i in range(4)]
RIGHT_FOOT_CONTACT_SENSORS = [f"right_foot_contact_{i}" for i in range(4)]


class G1WalkEnv(
    G1WalkObservationBindings,
    G1WalkControlBindings,
    G1WalkRuntimeBindings,
    G1WalkRewardBindings,
    G1BaseEnv,
):
    _cfg: G1WalkEnvCfg
    _reward_cfg: Any

    def __init__(self, cfg: G1WalkEnvCfg, num_envs=1, backend_type="mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        if cfg.reward_config.scales.get("feet_phase", 0.0) != 0.0 and cfg.reward_config.feet_phase_mode in {
            "filtered_command_height",
            "command_height",
            "phase_height",
            "absolute_phase_height",
        }:
            canonical_scene = ASSETS_ROOT_PATH / "robots/g1/scene_flat.xml"
            if cfg.scene.fragment_files or cfg.scene.terrain is not None:
                raise ValueError("absolute foot-height targets forbid terrain or scene fragments")
            if (
                backend_type != "mujoco"
                or epath.Path(cfg.scene.model_file).resolve() != canonical_scene.resolve()
            ):
                raise ValueError("absolute foot-height targets require the canonical MuJoCo G1 flat scene")
            if not cfg.gait_phase_enabled or cfg.gait_phase_init_mode != "offset_phase":
                raise ValueError("absolute foot-height targets require enabled offset gait phase")
        if cfg.reward_config.scales.get("phase_contact", 0.0) != 0.0 and cfg.reward_config.feet_phase_mode in {
            "command_gated_contact", "fixed_contact"
        }:
            canonical_scene = ASSETS_ROOT_PATH / "robots/g1/scene_flat.xml"
            phase_contact_cfg = cfg.command_gated_phase_contact
            fixed_contact = cfg.reward_config.feet_phase_mode == "fixed_contact"
            if not fixed_contact and not phase_contact_cfg.enabled:
                raise ValueError("command-gated contact requires its phase clock")
            if cfg.scene.fragment_files or cfg.scene.terrain is not None:
                raise ValueError("phase contact forbids terrain or scene fragments")
            if (
                backend_type != "mujoco"
                or epath.Path(cfg.scene.model_file).resolve() != canonical_scene.resolve()
            ):
                raise ValueError("phase contact requires the canonical MuJoCo G1 flat scene")
            expected_phase_mode = "offset_phase" if fixed_contact else "command_gated"
            if not cfg.gait_phase_enabled or cfg.gait_phase_init_mode != expected_phase_mode:
                raise ValueError(f"phase contact requires enabled {expected_phase_mode} gait phase")
        progress_cfg = cfg.forward_progress_termination
        if progress_cfg.grace_steps <= 0:
            raise ValueError("forward_progress_termination.grace_steps must be positive")
        if progress_cfg.min_command_forward_speed < 0.0:
            raise ValueError(
                "forward_progress_termination.min_command_forward_speed must be non-negative"
            )
        if progress_cfg.min_average_forward_speed < 0.0:
            raise ValueError(
                "forward_progress_termination.min_average_forward_speed must be non-negative"
            )
        if cfg.reward_config.straight_line_lateral_tolerance_m <= 0.0:
            raise ValueError("straight_line_lateral_tolerance_m must be positive")
        if cfg.reward_config.straight_line_yaw_tolerance_rad <= 0.0:
            raise ValueError("straight_line_yaw_tolerance_rad must be positive")
        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            push_body_name=cfg.domain_rand.push_body_name,
            mujoco_num_threads=cfg.mujoco_num_threads,
            motrix_max_iterations=cfg.motrix_max_iterations,
            post_step_forward_sensor=cfg.post_step_forward_sensor,
        )
        super().__init__(cfg, backend, num_envs)
        self._enable_reward_log = True
        self._reward_cfg = cfg.reward_config
        self._runtime_payload = G1RuntimePayload(
            backend,
            num_envs,
            cfg.runtime_payload,
            ctrl_dt=cfg.ctrl_dt,
        )

        self._gait_phase_delta = float(
            2.0 * math.pi * self._reward_cfg.gait_frequency * cfg.ctrl_dt
        )
        self._pose_weights = np.array(self._reward_cfg.pose_weights, dtype=get_global_dtype())
        if self._pose_weights.shape[0] != self._num_action:
            raise ValueError("pose_weights length mismatch")
        self._upper_body_pose_weights = build_upper_body_pose_weights(self._reward_cfg.pose_weights)
        self._episode_tracker: EpisodeLengthTracker | None = None
        self._penalty_episode_tracker: EpisodeLengthTracker | None = None
        self._penalty_curriculum: PenaltyCurriculum | None = None
        strength = cfg.domain_rand.actuator_strength
        tracks_dr_quality = (
            strength.group_curriculum_enabled
            and strength.curriculum_progress_mode == "episode_quality"
        )
        if cfg.curriculum.enabled or tracks_dr_quality:
            self._episode_tracker = EpisodeLengthTracker(num_envs)
        if cfg.curriculum.enabled:
            self._penalty_episode_tracker = EpisodeLengthTracker(num_envs)
            self._penalty_curriculum = PenaltyCurriculum(
                self,
                enabled=True,
                initial_scale=cfg.curriculum.initial_scale,
                min_scale=cfg.curriculum.min_scale,
                max_scale=cfg.curriculum.max_scale,
                level_down_threshold=cfg.curriculum.level_down_threshold,
                level_up_threshold=cfg.curriculum.level_up_threshold,
                degree=cfg.curriculum.degree,
                penalty_names=self._reward_cfg.penalty_curriculum_terms,
            )

        self._init_reward_functions()
        self._fada_base_kp: np.ndarray | None = None
        self._fada_base_kd: np.ndarray | None = None
        self._fada_base_body_mass: np.ndarray | None = None
        self._fada_ground_geom_id: int | None = None
        self._fada_base_geom_friction: np.ndarray | None = None
        self._fada_tau_max: np.ndarray | None = None
        self._fada_body_names: tuple[str, ...] = ()
        self._fada_checkpoint_layout_identity: G1FADAPrivilegedCheckpointLayoutIdentity | None = (
            None
        )
        strength_enabled = bool(
            getattr(getattr(cfg.domain_rand, "actuator_strength", None), "enabled", False)
        )
        privileged_enabled = bool(cfg.fada_privileged_observation.enabled)
        if (
            cfg.domain_rand.randomize_kp
            or cfg.domain_rand.randomize_kd
            or strength_enabled
            or privileged_enabled
        ):
            base_kp, base_kd = backend.get_actuator_gains()
            self._fada_base_kp = np.asarray(base_kp, dtype=np.float64)
            self._fada_base_kd = np.asarray(base_kd, dtype=np.float64)
        else:
            base_kp = base_kd = None
        if privileged_enabled:
            if backend.backend_type != "mujoco":
                raise ValueError("g1_fada_privileged_v1 currently requires the MuJoCo backend")
            self._fada_base_body_mass = backend.get_body_mass()
            self._fada_base_geom_friction = backend.get_geom_friction()
            self._fada_ground_geom_id = backend.get_geom_id("floor")
            self._fada_tau_max = np.max(np.abs(backend.get_actuator_force_range()), axis=1)
            self._fada_body_names = backend.get_body_names()
            actuated_joint_names = backend.get_actuated_joint_names()
            if len(actuated_joint_names) != self._num_action:
                raise ValueError(
                    "FADA actuated joint order must contain exactly one name per action"
                )
            self._fada_checkpoint_layout_identity = build_g1_fada_checkpoint_layout_identity(
                body_names=self._fada_body_names,
                actuated_joint_names=actuated_joint_names,
                model_file=cfg.scene.model_file,
            )
        dr_provider = G1WalkDomainRandomizationProvider(
            base_kp=base_kp,
            base_kd=base_kd,
            base_body_mass=self._fada_base_body_mass,
            base_geom_friction=self._fada_base_geom_friction,
            ground_geom_id=self._fada_ground_geom_id,
        )
        self._fada_dr_provider = dr_provider
        self._init_domain_randomization(dr_provider)

    def init_state(self) -> NpEnvState:
        state = super().init_state()
        self._runtime_payload.init_info(state.info)
        return state

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        self._runtime_payload.reconcile_info(state.info)
        self._runtime_payload.before_physics(state.info)
        return super().apply_action(actions, state)

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        if self._state is not None:
            self._runtime_payload.restore_before_reset(self._state.info, env_indices)
        obs, reset_info = super().reset(env_indices)
        reset_info.update(self._runtime_payload.reset_info_updates(env_indices))
        return obs, reset_info


registry.register_env("G1WalkFlat", G1WalkEnv, sim_backend="mujoco")
registry.register_env("G1WalkFlat", G1WalkEnv, sim_backend="motrix")
registry.register_env("G1WalkHeight", G1WalkEnv, sim_backend="mujoco")
registry.register_env("G1StandStill", G1WalkEnv, sim_backend="mujoco")
registry.register_env("G1StandHeight", G1WalkEnv, sim_backend="mujoco")
registry.register_env("G1WalkRough", G1WalkEnv, sim_backend="mujoco")
registry.register_env("G1WalkRough", G1WalkEnv, sim_backend="motrix")
