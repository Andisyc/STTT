"""Cold-bound official control mapping over public environment/backend APIs."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from unilab.algos.torch.distill.fada.runtime_policy import OfficialFADAPlaybackProfile
from unilab.envs.common.rotation import np_quat_apply_inverse


def load_official_playback_profile(cfg: Any, *, root_dir: str | Path, source_identity) -> OfficialFADAPlaybackProfile:
    path = OmegaConf.select(cfg, "official_playback.profile_path", default=None)
    if not path:
        fields = ", ".join(OfficialFADAPlaybackProfile.__dataclass_fields__)
        raise ValueError(
            "official FADA playback requires official_playback.profile_path bound to the "
            f"actual Oracle run; required profile fields: {fields}. No legacy control defaults are assumed."
        )
    profile_path = Path(str(path)).expanduser()
    if not profile_path.is_absolute():
        profile_path = Path(root_dir) / profile_path
    payload = OmegaConf.to_container(OmegaConf.load(profile_path), resolve=True)
    if not isinstance(payload, Mapping):
        raise ValueError("official playback control profile must be a mapping")
    return OfficialFADAPlaybackProfile.from_mapping(payload, source_identity=source_identity)


class OfficialFADADeploymentAdapter:
    """Keep source-normalized inference separate from target actuator order.

    Physical signals are read through public methods declared by SimBackend,
    forwarded by NpEnv, never through private MuJoCo data or XML on a hot path.
    """

    def __init__(self, env, profile: OfficialFADAPlaybackProfile, *, device: str):
        self.env = env
        self.profile = profile
        self.device = torch.device(device)
        self.target_joint_names = tuple(env.get_actuated_joint_names())
        source_from_target = profile.source_indices(self.target_joint_names)
        target_from_source = np.argsort(source_from_target)
        self._target_from_source = torch.tensor(target_from_source, dtype=torch.long, device=self.device)
        target_default = np.asarray(env.default_angles, dtype=np.float64).reshape(-1)
        target_scale = np.broadcast_to(np.asarray(env.cfg.control_config.action_scale, dtype=np.float64), (29,))
        if target_default.shape != (29,) or not np.isfinite(target_default).all() or not np.isfinite(target_scale).all() or (target_scale <= 0).any():
            raise ValueError("official playback requires finite target defaults and positive per-joint action scale")
        source_default = np.asarray(profile.default_joint_pos)[target_from_source]
        source_scale = np.asarray(profile.action_scale)[target_from_source]
        self._action_gain = torch.tensor(source_scale / target_scale, dtype=torch.float32, device=self.device)
        self._action_bias = torch.tensor((source_default - target_default) / target_scale, dtype=torch.float32, device=self.device)
        kp, kd = env.get_actuator_gains()
        for name, actual, expected in (
            ("stiffness", kp, profile.stiffness), ("damping", kd, profile.damping)
        ):
            actual = np.asarray(actual).reshape(-1)
            expected = np.asarray(expected)[target_from_source]
            if actual.shape != (29,) or not np.allclose(actual, expected, rtol=1e-6, atol=1e-7):
                raise ValueError(f"official source/target {name} mismatch; select source-aligned PD configuration")
        if not np.isclose(float(env.cfg.ctrl_dt), profile.ctrl_dt, rtol=1e-6, atol=1e-9):
            raise ValueError("official source/target ctrl_dt mismatch")
        if not np.isclose(float(env.cfg.reward_config.gait_frequency), profile.gait_frequency, rtol=1e-6, atol=1e-9):
            raise ValueError("official source/target gait_frequency mismatch")
        if str(env.cfg.gait_clock_mode) != profile.gait_clock_mode or not env.cfg.gait_phase_enabled:
            raise ValueError("official source/target gait clock mismatch")
        self._world_gravity = np.broadcast_to(np.array([0., 0., -1.]), (int(env.num_envs), 3))

    def observation(self) -> dict[str, torch.Tensor]:
        quat = np.asarray(self.env.get_base_quat())
        omega = np.asarray(self.env.get_base_ang_vel())
        if quat.shape != (int(self.env.num_envs), 4) or not np.isfinite(quat).all() or not np.allclose(np.linalg.norm(quat, axis=1), 1., rtol=1e-5, atol=1e-6):
            raise ValueError("official playback requires finite unit base quaternions in wxyz order")
        fields = {
            "base_ang_vel": np_quat_apply_inverse(quat, omega),
            "projected_gravity": np_quat_apply_inverse(quat, self._world_gravity),
            "dof_pos": self.env.get_dof_pos(),
            "dof_vel": self.env.get_dof_vel(),
        }
        return {name: torch.as_tensor(value, dtype=torch.float32, device=self.device) for name, value in fields.items()}

    def command(self, velocity, phase) -> torch.Tensor:
        from unilab.algos.torch.distill.fada.observation import project_official_7d_command
        if phase is None:
            raise ValueError("official FADA playback requires same-step gait_phase; phase is not zero-padded")
        return project_official_7d_command(
            torch.as_tensor(velocity, dtype=torch.float32, device=self.device),
            torch.as_tensor(phase, dtype=torch.float32, device=self.device),
        )

    def env_action(self, source_action: torch.Tensor) -> torch.Tensor:
        return source_action.index_select(1, self._target_from_source) * self._action_gain + self._action_bias

    def reset(self, done=None) -> None:
        state = self.env.state
        if state is None or "gait_phase" not in state.info:
            raise ValueError("official playback requires an initialized public gait_phase state")
        mask = np.ones(int(self.env.num_envs), dtype=bool) if done is None else np.asarray(
            torch.as_tensor(done, dtype=torch.bool).cpu(), dtype=bool
        ).reshape(-1)
        if mask.shape != (int(self.env.num_envs),):
            raise ValueError("official phase reset mask batch mismatch")
        state.info["gait_phase"][mask] = self.profile.phase_init
