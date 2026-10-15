"""Deterministic per-episode runtime payload lifecycle for G1 MuJoCo environments."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from unilab.base.backend.base import SimBackend
    from unilab.envs.locomotion.g1.walk_config import G1RuntimePayloadConfig

TRUTH_KEY_ACTIVE = "runtime_payload_active"
TRUTH_KEY_ATTACHED = "runtime_payload_attached_this_step"
TRUTH_KEY_ADDED_MASS = "runtime_payload_added_mass_kg"


class G1RuntimePayload:
    """Own the deterministic mid-episode payload attach/restore lifecycle.

    Per-row active/event/added-mass truth lives in the ``state.info`` arrays so
    rollout snapshot/restore stays consistent; this owner keeps only
    preallocated scratch masks plus the immutable baseline mass/COM and the
    derived attach schedule. Only public ``SimBackend`` methods are used.
    """

    def __init__(
        self,
        backend: "SimBackend",
        num_envs: int,
        config: "G1RuntimePayloadConfig",
        *,
        ctrl_dt: float,
    ) -> None:
        self._backend = backend
        self._num_envs = int(num_envs)
        self._enabled = bool(config.enabled)
        self._added_mass_kg = float(config.added_mass_kg)
        self._attach_steps = 0
        self._body_ids: np.ndarray | None = None
        self._baseline_mass = 0.0
        self._baseline_com = np.zeros(3, dtype=np.float64)
        self._target_com = np.zeros(3, dtype=np.float64)
        self._baseline_inertia = np.zeros((3, 3), dtype=np.float64)
        self._target_inertia = np.zeros((3, 3), dtype=np.float64)
        self._scratch = np.zeros((self._num_envs,), dtype=bool)
        if not self._enabled:
            return
        config.validate()
        if backend.backend_type != "mujoco":
            raise ValueError("runtime payload currently requires the MuJoCo backend")
        if not math.isfinite(ctrl_dt) or ctrl_dt <= 0.0:
            raise ValueError("runtime payload requires a finite positive ctrl_dt")
        body_name = str(config.body_name)
        if not body_name:
            raise ValueError("runtime payload requires a non-empty body_name")
        self._attach_steps = int(math.ceil(float(config.attach_time_s) / float(ctrl_dt)))
        body_id = int(backend.get_body_id(body_name))
        self._body_ids = np.asarray([body_id], dtype=np.int64)
        self._baseline_mass = float(np.asarray(backend.get_body_mass())[body_id])
        self._baseline_com = (
            np.asarray(backend.get_body_ipos(), dtype=np.float64)[body_id].copy()
        )
        self._baseline_inertia = np.asarray(
            backend.get_body_inertia_tensor(), dtype=np.float64
        )[body_id].copy()
        if not math.isfinite(self._baseline_mass) or self._baseline_mass <= 0.0:
            raise ValueError("runtime payload baseline body mass must be finite and positive")
        target_mass = self._baseline_mass + self._added_mass_kg
        if target_mass <= 0.0:
            raise ValueError("runtime payload total body mass must be positive")
        # Rigidly combine the original body and an axis-aligned finite-size box.
        position = np.asarray(config.attachment_pos_m, dtype=np.float64)
        half_size = np.asarray(config.half_size_m, dtype=np.float64)
        displacement = position - self._baseline_com
        self._target_com = self._baseline_com + (self._added_mass_kg / target_mass) * displacement
        hx2, hy2, hz2 = half_size * half_size
        box_inertia = np.diag(
            (self._added_mass_kg / 3.0) * np.asarray([hy2 + hz2, hx2 + hz2, hx2 + hy2])
        )
        reduced_mass = self._baseline_mass * self._added_mass_kg / target_mass
        self._target_inertia = self._baseline_inertia + box_inertia + reduced_mass * (
            np.dot(displacement, displacement) * np.eye(3) - np.outer(displacement, displacement)
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def init_info(self, info: dict) -> None:
        """Publish the per-row truth arrays on a fresh state carrier."""

        info[TRUTH_KEY_ACTIVE] = np.zeros((self._num_envs,), dtype=bool)
        info[TRUTH_KEY_ATTACHED] = np.zeros((self._num_envs,), dtype=bool)
        info[TRUTH_KEY_ADDED_MASS] = np.zeros((self._num_envs,), dtype=np.float64)

    def reconcile_info(self, info: dict) -> None:
        """Rebind truth arrays when an info carrier was replaced wholesale."""

        active = info.get(TRUTH_KEY_ACTIVE)
        event = info.get(TRUTH_KEY_ATTACHED)
        added = info.get(TRUTH_KEY_ADDED_MASS)
        if (
            not isinstance(active, np.ndarray)
            or active.shape != (self._num_envs,)
            or active.dtype != bool
            or not isinstance(event, np.ndarray)
            or event.shape != (self._num_envs,)
            or event.dtype != bool
            or not isinstance(added, np.ndarray)
            or added.shape != (self._num_envs,)
        ):
            self.init_info(info)
            return
        event.fill(False)
        added.fill(0.0)
        added[active] = self._added_mass_kg

    def before_physics(self, info: dict) -> None:
        """Attach the payload at the scheduled episode step, before the physics step."""

        if not self._enabled:
            return
        active = info[TRUTH_KEY_ACTIVE]
        event = info[TRUTH_KEY_ATTACHED]
        event.fill(False)
        steps = np.asarray(info["steps"])
        np.greater_equal(steps, self._attach_steps, out=self._scratch)
        self._scratch[active] = False
        if not np.any(self._scratch):
            return
        env_ids = np.flatnonzero(self._scratch).astype(np.int64)
        assert self._body_ids is not None
        num_attach = int(env_ids.size)
        self._backend.set_body_mass_and_com(
            env_ids,
            self._body_ids,
            mass=np.full((num_attach, 1), self._baseline_mass + self._added_mass_kg),
            center_of_mass=np.tile(self._target_com, (num_attach, 1, 1)),
            inertia_tensor=np.broadcast_to(self._target_inertia, (num_attach, 1, 3, 3)),
        )
        active[env_ids] = True
        event[env_ids] = True
        info[TRUTH_KEY_ADDED_MASS][env_ids] = self._added_mass_kg

    def reset_info_updates(self, env_indices: np.ndarray) -> dict[str, np.ndarray]:
        """Zeroed per-row truth values for the reset info carrier."""

        reset_rows = np.asarray(env_indices, dtype=np.int64).reshape(-1)
        return {
            TRUTH_KEY_ACTIVE: np.zeros((reset_rows.size,), dtype=bool),
            TRUTH_KEY_ATTACHED: np.zeros((reset_rows.size,), dtype=bool),
            TRUTH_KEY_ADDED_MASS: np.zeros((reset_rows.size,), dtype=np.float64),
        }

    def restore_before_reset(self, info: dict, env_indices: np.ndarray) -> None:
        """Restore baseline mass/COM for active rows and rearm them for the next episode."""

        if not self._enabled:
            return
        reset_rows = np.asarray(env_indices, dtype=np.int64).reshape(-1)
        if reset_rows.size == 0:
            return
        # Cold init: truth arrays may not exist yet (init_info runs after super().init_state()).
        if TRUTH_KEY_ACTIVE not in info:
            return
        active = info[TRUTH_KEY_ACTIVE]
        env_ids = reset_rows[active[reset_rows]]
        if env_ids.size:
            assert self._body_ids is not None
            num_restore = int(env_ids.size)
            self._backend.set_body_mass_and_com(
                env_ids,
                self._body_ids,
                mass=np.full((num_restore, 1), self._baseline_mass),
                center_of_mass=np.tile(self._baseline_com, (num_restore, 1, 1)),
                inertia_tensor=np.broadcast_to(self._baseline_inertia, (num_restore, 1, 3, 3)),
            )
            active[env_ids] = False
        info[TRUTH_KEY_ATTACHED][reset_rows] = False
        info[TRUTH_KEY_ADDED_MASS][reset_rows] = 0.0
