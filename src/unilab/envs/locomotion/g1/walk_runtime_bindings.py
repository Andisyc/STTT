"""Framework bindings extracted from :mod:`joystick` by responsibility.

The concrete environment remains the sole owner of mutable state.
"""

from __future__ import annotations

import copy
from collections.abc import MutableMapping
from typing import Any, cast

import numpy as np

from unilab.base.np_env import NpEnvState
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.g1.action_trace import (
    G1ActionTraceSnapshot,
    action_trace_enabled,
    action_trace_interval,
    emit_g1_action_trace,
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
    sample_g1_walk_commands,
    sample_gait_phase_pairs,
    sample_reset_base_qvel,
)

LEFT_FOOT_CONTACT_SENSORS = [f"left_foot_contact_{i}" for i in range(4)]
RIGHT_FOOT_CONTACT_SENSORS = [f"right_foot_contact_{i}" for i in range(4)]


def publish_walk_termination_provenance(
    info: MutableMapping[str, Any],
    *,
    fall_terminated: np.ndarray,
    forward_progress_terminated: np.ndarray,
) -> None:
    """Publish task termination causes without sharing mutable owner buffers."""

    fall = np.asarray(fall_terminated, dtype=np.bool_)
    forward = np.asarray(forward_progress_terminated, dtype=np.bool_)
    if fall.ndim != 1 or forward.shape != fall.shape:
        raise ValueError("G1 walk termination provenance masks must be matching rank-1 arrays")
    info["fall_terminated"] = fall.copy()
    info["forward_progress_terminated"] = forward.copy()


class G1WalkRuntimeBindings:
    def get_foot_contact(self) -> np.ndarray:
        """Return real left/right foot contacts with shape ``(N, 2)``.

        Reuse the task's existing any-corner contact rule; gait targets and
        predicted contact schedules are not sensor observations.
        """
        left = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)
        right = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)
        if left.shape != (self._num_envs,) or right.shape != left.shape:
            raise ValueError("G1 foot contact sensors must match the environment rows")
        return np.stack((left, right), axis=1)

    def refresh_state(self) -> NpEnvState:
        """Refresh playback observations without a reward or curriculum transaction."""
        if self._state is None:
            self.init_state()
        assert self._state is not None
        self._state = self._observe_state(self._state)
        return self._state

    def _observe_state(self, state: NpEnvState) -> NpEnvState:
        self._synchronize_external_command_for_observation(state.info)
        obs = self._compute_obs(
            state.info,
            self.get_local_linvel(),
            self.get_gyro(),
            self._backend.get_sensor_data(self._cfg.sensor.upvector),
            self.get_dof_pos(),
            self.get_dof_vel(),
        )
        return state.replace(obs=obs)

    def _command_gated_physics_step_pending(self, info: dict) -> bool:
        if not self._command_gated_phase_contact_enabled():
            return True
        valid = np.asarray(
            info.get("command_gated_actuator_target_valid", np.zeros(self._num_envs)),
            dtype=np.bool_,
        )
        if valid.shape != (self._num_envs,):
            raise ValueError("v024 actuator target validity must match env rows")
        if np.any(valid) and not np.all(valid):
            raise ValueError("v024 actuator target validity cannot be partial")
        return bool(np.all(valid))

    def _debug_action_trace_enabled(self) -> bool:
        return action_trace_enabled()

    def _debug_action_trace_step(self, info: dict) -> int:
        steps = info.get("steps")
        if steps is not None:
            steps_arr = np.asarray(steps)
            if steps_arr.size > 0:
                return int(steps_arr.reshape(-1)[0])
        step = int(info.get("_g1_action_trace_step", 0))
        info["_g1_action_trace_step"] = step + 1
        return step

    def _debug_virtual_pd_torque(
        self, ctrl: np.ndarray | None, dof_pos: np.ndarray, dof_vel: np.ndarray
    ) -> np.ndarray | None:
        if ctrl is None:
            return None
        try:
            kp, kd = self._backend.get_actuator_gains()
        except (AttributeError, NotImplementedError, RuntimeError):
            return None
        kp_arr = np.asarray(kp, dtype=get_global_dtype())
        kd_arr = np.asarray(kd, dtype=get_global_dtype())
        if kp_arr.shape[0] != ctrl.shape[1] or kd_arr.shape[0] != ctrl.shape[1]:
            return None
        return np.asarray(kp_arr[None, :] * (ctrl - dof_pos) - kd_arr[None, :] * dof_vel)

    def _debug_action_trace(
        self,
        info: dict,
        *,
        reward: np.ndarray,
        terminated: np.ndarray,
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> None:
        if not self._debug_action_trace_enabled():
            return
        step = self._debug_action_trace_step(info)
        interval = action_trace_interval()
        if step % interval != 0:
            return

        current_actions = np.asarray(
            info.get("current_actions", np.zeros((self._num_envs, self._num_action))),
            dtype=get_global_dtype(),
        )
        executed_actions = np.asarray(
            info.get("executed_actions", np.zeros_like(current_actions)),
            dtype=get_global_dtype(),
        )
        ctrl = info.get("_g1_action_trace_ctrl")
        ctrl_arr = None if ctrl is None else np.asarray(ctrl, dtype=get_global_dtype())
        default = np.broadcast_to(self.default_angles, current_actions.shape).astype(
            get_global_dtype(),
            copy=False,
        )
        virtual_pd_tau = self._debug_virtual_pd_torque(ctrl_arr, dof_pos, dof_vel)
        torques = info.get("torques")

        commands = np.asarray(info.get("commands", np.zeros((self._num_envs, 3))))
        gait_enabled = self._gait_enabled_mask(info)
        dynamic_mode = self._dynamic_mode_mask(info)
        base_height = self._terrain_relative_base_height()
        base_height_target = float(self._reward_cfg.base_height_target)
        tilt_deg = np.rad2deg(np.arccos(np.clip(gravity[:, 2], -1.0, 1.0)))
        left_contact = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)
        right_contact = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)
        left_count, right_count = self._foot_contact_counts()
        base_feet_delta = self._base_delta_from_feet_center_in_base_yaw_frame()

        reward_log = info.get("log", {})
        emit_g1_action_trace(
            G1ActionTraceSnapshot(
                step=step,
                task_name=type(self._cfg).__name__,
                action_scale=float(self._cfg.control_config.action_scale),
                stand_action_authority=bool(self._cfg.stand_action_authority),
                mode_observation=bool(self._cfg.mode_observation),
                reward=reward,
                terminated=terminated,
                commands=commands,
                gait_enabled=gait_enabled,
                dynamic_mode=dynamic_mode,
                current_actions=current_actions,
                executed_actions=executed_actions,
                ctrl=ctrl_arr,
                default_angles=default,
                dof_pos=dof_pos,
                dof_vel=dof_vel,
                virtual_pd_torque=virtual_pd_tau,
                torques=None if torques is None else np.asarray(torques),
                linvel=linvel,
                gyro=gyro,
                base_height_target=base_height_target,
                base_height=base_height,
                tilt_deg=tilt_deg,
                left_contact=left_contact,
                right_contact=right_contact,
                left_contact_count=left_count,
                right_contact_count=right_count,
                base_minus_feet_center_xy=base_feet_delta,
                reward_log=reward_log if isinstance(reward_log, dict) else {},
            )
        )

    def update_state(self, state: NpEnvState) -> NpEnvState:
        physics_step_pending = self._command_gated_physics_step_pending(state.info)
        if not physics_step_pending:
            return self._observe_state(state)
        self._advance_command_state_for_reward(state.info)
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._cfg.sensor.upvector)
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        self._refresh_command_gated_torque(state.info, dof_pos, dof_vel)

        max_tilt_rad = np.deg2rad(self._reward_cfg.max_tilt_deg)
        tilt = np.arccos(np.clip(gravity[:, 2], -1, 1))
        fall_terminated = np.logical_or(
            tilt > max_tilt_rad,
            self._terrain_relative_base_height() < self._reward_cfg.min_base_height,
        )
        forward_progress_terminated = self._forward_progress_failure(state.info)
        terminated = np.logical_or(fall_terminated, forward_progress_terminated)
        publish_walk_termination_provenance(
            state.info,
            fall_terminated=fall_terminated,
            forward_progress_terminated=forward_progress_terminated,
        )

        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        self._debug_action_trace(
            state.info,
            reward=reward,
            terminated=terminated,
            linvel=linvel,
            gyro=gyro,
            gravity=gravity,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
        )
        self._commit_command_state_for_next_observation(state.info)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        state = state.replace(obs=obs, reward=reward, terminated=terminated)

        strength_cfg = getattr(self._cfg.domain_rand, "actuator_strength", None)
        iteration_mode = (
            str(getattr(strength_cfg, "curriculum_progress_mode", "episode_quality"))
            == "iterations"
        )
        if iteration_mode:
            self._fada_dr_provider.update_iteration_curriculum(
                self,
                self.step_counter + 1,
                float(np.mean(terminated.astype(get_global_dtype()))),
            )
        return state

    def _on_step_completed(self, state: NpEnvState) -> None:
        strength_cfg = self._cfg.domain_rand.actuator_strength
        iteration_mode = strength_cfg.curriculum_progress_mode == "iterations"
        done = state.terminated | state.truncated
        if self._episode_tracker is not None and np.any(done):
            done_indices = np.where(done)[0]
            episode_lengths = state.info["steps"][done_indices]
            self._episode_tracker.update(episode_lengths)
            if not iteration_mode:
                self._fada_dr_provider.update_actuator_strength_curriculum(
                    self, self._episode_tracker.average_length, len(done_indices)
                )
        # Preserve the historical penalty curriculum population explicitly:
        # pure timeouts remain in general/DR statistics, not penalty updates.
        if self._penalty_curriculum is not None and np.any(state.terminated):
            assert self._penalty_episode_tracker is not None
            self._penalty_episode_tracker.update(state.info["steps"][state.terminated])
            self._penalty_curriculum.update(self._penalty_episode_tracker.average_length)
        self._write_curriculum_log(state.info)

    def _write_curriculum_log(self, info: dict[str, Any]) -> None:
        log = info.setdefault("log", {})
        if self._episode_tracker is not None:
            log["curriculum/average_episode_length"] = float(self._episode_tracker.average_length)
        if self._penalty_curriculum is not None:
            log["curriculum/penalty_scale"] = float(self._penalty_curriculum.current_scale)
        strength_cfg = getattr(self._cfg.domain_rand, "actuator_strength", None)
        grouped_curriculum = bool(getattr(strength_cfg, "group_curriculum_enabled", False))
        if grouped_curriculum:
            _, scale = self._fada_dr_provider.grouped_domain_rand_curriculum_profile(self)
            log["curriculum/domain_randomization_scale"] = scale
        if grouped_curriculum:
            if (
                str(getattr(strength_cfg, "curriculum_progress_mode", "episode_quality"))
                == "iterations"
            ):
                log["curriculum/training_iteration"] = float(self.step_counter)

    def _capture_task_rollout_state(self) -> dict[str, Any]:
        """Capture G1 curriculum state that may change on a shadow termination."""

        return {
            "episode_average_length": (
                None
                if self._episode_tracker is None
                else float(self._episode_tracker.average_length)
            ),
            "penalty_episode_average_length": (
                None
                if self._penalty_episode_tracker is None
                else float(self._penalty_episode_tracker.average_length)
            ),
            "penalty_scale": (
                None
                if self._penalty_curriculum is None
                else float(self._penalty_curriculum.current_scale)
            ),
            "reward_scales": copy.deepcopy(self._reward_cfg.scales),
            "actuator_strength_curriculum": (
                None
                if not hasattr(self, "_fada_dr_provider")
                else self._fada_dr_provider.capture_actuator_strength_curriculum_state()
            ),
        }

    def _restore_task_rollout_state(self, snapshot: Any) -> None:
        """Restore G1 curriculum and its derived reward-scale mutation."""

        if not isinstance(snapshot, dict) or set(snapshot) != {
            "episode_average_length",
            "penalty_episode_average_length",
            "penalty_scale",
            "reward_scales",
            "actuator_strength_curriculum",
        }:
            raise ValueError("invalid G1 task rollout snapshot")
        if self._episode_tracker is not None:
            average = snapshot["episode_average_length"]
            if average is None:
                raise ValueError("G1 rollout snapshot is missing episode tracker state")
            self._episode_tracker.average_length = float(average)
        if self._penalty_episode_tracker is not None:
            average = snapshot["penalty_episode_average_length"]
            if average is None:
                raise ValueError("G1 rollout snapshot is missing penalty episode tracker state")
            self._penalty_episode_tracker.average_length = float(average)
        if self._penalty_curriculum is not None:
            scale = snapshot["penalty_scale"]
            if scale is None:
                raise ValueError("G1 rollout snapshot is missing penalty curriculum state")
            self._penalty_curriculum.current_scale = float(scale)
        self._reward_cfg.scales.clear()
        self._reward_cfg.scales.update(copy.deepcopy(snapshot["reward_scales"]))
        strength_state = snapshot["actuator_strength_curriculum"]
        if hasattr(self, "_fada_dr_provider") and strength_state is not None:
            self._fada_dr_provider.restore_actuator_strength_curriculum_state(strength_state)
