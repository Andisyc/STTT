"""Focused fake-backend proof of the G1 runtime payload lifecycle; no simulator."""

from __future__ import annotations

import numpy as np

from unilab.envs.locomotion.g1.walk_config import G1RuntimePayloadConfig
from unilab.envs.locomotion.g1.walk_runtime_payload import (
    G1RuntimePayload,
    TRUTH_KEY_ACTIVE,
    TRUTH_KEY_ADDED_MASS,
    TRUTH_KEY_ATTACHED,
)


class _FakeBackend:
    backend_type = "mujoco"

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    def get_body_id(self, name: str) -> int:
        assert name == "pelvis"
        return 3

    def get_body_mass(self) -> np.ndarray:
        masses = np.full(6, 2.0)
        masses[3] = 5.0
        return masses

    def get_body_ipos(self) -> np.ndarray:
        ipos = np.zeros((6, 3))
        ipos[3] = [0.01, 0.02, 0.03]
        return ipos

    def set_body_mass_and_com(
        self,
        env_indices: np.ndarray,
        body_ids: np.ndarray,
        *,
        mass: np.ndarray,
        center_of_mass: np.ndarray | None = None,
    ) -> None:
        assert center_of_mass is not None
        self.calls.append(
            (
                np.asarray(env_indices).copy(),
                np.asarray(body_ids).copy(),
                np.asarray(mass).copy(),
                np.asarray(center_of_mass).copy(),
            )
        )


def _owner(num_envs: int = 3) -> tuple[_FakeBackend, G1RuntimePayload, dict]:
    backend = _FakeBackend()
    owner = G1RuntimePayload(
        backend,
        num_envs,
        G1RuntimePayloadConfig(
            enabled=True,
            body_name="pelvis",
            added_mass_kg=1.5,
            com_offset_m=[0.1, 0.0, -0.1],
            # ceil(0.03 / 0.02) = 2 episode steps.
            attach_time_s=0.03,
        ),
        ctrl_dt=0.02,
    )
    info: dict = {"steps": np.zeros((num_envs,), dtype=np.int64)}
    owner.init_info(info)
    return backend, owner, info


def test_runtime_payload_zero_attach_time_attaches_on_first_tick() -> None:
    backend = _FakeBackend()
    owner = G1RuntimePayload(
        backend,
        2,
        G1RuntimePayloadConfig(
            enabled=True,
            body_name="pelvis",
            added_mass_kg=1.5,
            com_offset_m=[0.1, 0.0, -0.1],
            attach_time_s=0.0,
        ),
        ctrl_dt=0.02,
    )
    info: dict = {"steps": np.zeros((2,), dtype=np.int64)}
    owner.init_info(info)

    # steps == 0 on the first tick is exactly at the schedule and attaches.
    owner.before_physics(info)
    assert len(backend.calls) == 1
    env_ids, _, mass, _ = backend.calls[0]
    np.testing.assert_array_equal(env_ids, [0, 1])
    np.testing.assert_allclose(mass, [[6.5], [6.5]])
    np.testing.assert_array_equal(info[TRUTH_KEY_ACTIVE], [True, True])
    np.testing.assert_array_equal(info[TRUTH_KEY_ATTACHED], [True, True])

    # Reset restores and rearms; the next episode's first tick re-attaches.
    info["steps"][:] = [7, 7]
    owner.restore_before_reset(info, np.array([1]))
    assert len(backend.calls) == 2
    np.testing.assert_array_equal(backend.calls[1][0], [1])
    np.testing.assert_allclose(backend.calls[1][2], [[5.0]])
    info["steps"][:] = [8, 0]
    owner.before_physics(info)
    assert len(backend.calls) == 3
    np.testing.assert_array_equal(backend.calls[2][0], [1])


def test_runtime_payload_reset_info_updates_publish_zeroed_truth() -> None:
    _, owner, _ = _owner()
    updates = owner.reset_info_updates(np.array([2, 0]))
    assert set(updates) == {
        TRUTH_KEY_ACTIVE,
        TRUTH_KEY_ATTACHED,
        TRUTH_KEY_ADDED_MASS,
    }
    np.testing.assert_array_equal(updates[TRUTH_KEY_ACTIVE], [False, False])
    np.testing.assert_array_equal(updates[TRUTH_KEY_ATTACHED], [False, False])
    np.testing.assert_array_equal(updates[TRUTH_KEY_ADDED_MASS], [0.0, 0.0])


def test_runtime_payload_disabled_path_makes_no_backend_calls() -> None:
    backend = _FakeBackend()
    owner = G1RuntimePayload(
        backend,
        2,
        G1RuntimePayloadConfig(enabled=False),
        ctrl_dt=0.02,
    )
    info: dict = {"steps": np.full(2, 100, dtype=np.int64)}
    owner.init_info(info)
    owner.before_physics(info)
    owner.restore_before_reset(info, np.array([0, 1]))
    assert backend.calls == []
    np.testing.assert_array_equal(info[TRUTH_KEY_ACTIVE], [False, False])
    np.testing.assert_array_equal(info[TRUTH_KEY_ADDED_MASS], [0.0, 0.0])


def test_runtime_payload_attach_restore_and_rearm_lifecycle() -> None:
    backend, owner, info = _owner()

    # No early call before the scheduled episode step.
    owner.before_physics(info)
    assert backend.calls == []
    np.testing.assert_array_equal(info[TRUTH_KEY_ACTIVE], [False, False, False])
    np.testing.assert_array_equal(info[TRUTH_KEY_ATTACHED], [False, False, False])

    # Non-target rows stay untouched; the eligible row attaches exactly once.
    info["steps"][:] = [0, 2, 5]
    owner.before_physics(info)
    assert len(backend.calls) == 1
    env_ids, body_ids, mass, com = backend.calls[0]
    np.testing.assert_array_equal(env_ids, [1, 2])
    np.testing.assert_array_equal(body_ids, [3])
    np.testing.assert_allclose(mass, [[6.5], [6.5]])
    np.testing.assert_allclose(com, [[[0.11, 0.02, -0.07]], [[0.11, 0.02, -0.07]]])
    np.testing.assert_array_equal(info[TRUTH_KEY_ACTIVE], [False, True, True])
    np.testing.assert_array_equal(info[TRUTH_KEY_ATTACHED], [False, True, True])
    np.testing.assert_array_equal(info[TRUTH_KEY_ADDED_MASS], [0.0, 1.5, 1.5])

    # One-time trigger: the next step re-attaches nothing and clears the event row.
    info["steps"][:] = [1, 3, 6]
    owner.before_physics(info)
    assert len(backend.calls) == 1
    np.testing.assert_array_equal(info[TRUTH_KEY_ATTACHED], [False, False, False])
    np.testing.assert_array_equal(info[TRUTH_KEY_ACTIVE], [False, True, True])

    # Reset restores only active rows to absolute baseline values and rearms them.
    owner.restore_before_reset(info, np.array([0, 2]))
    assert len(backend.calls) == 2
    env_ids, body_ids, mass, com = backend.calls[1]
    np.testing.assert_array_equal(env_ids, [2])
    np.testing.assert_array_equal(body_ids, [3])
    np.testing.assert_allclose(mass, [[5.0]])
    np.testing.assert_allclose(com, [[[0.01, 0.02, 0.03]]])
    np.testing.assert_array_equal(info[TRUTH_KEY_ACTIVE], [False, True, False])
    np.testing.assert_array_equal(info[TRUTH_KEY_ADDED_MASS], [0.0, 1.5, 0.0])

    # The rearmed row re-attaches on schedule in its next episode.
    info["steps"][:] = [0, 4, 2]
    owner.before_physics(info)
    assert len(backend.calls) == 3
    env_ids, _, mass, com = backend.calls[2]
    np.testing.assert_array_equal(env_ids, [2])
    np.testing.assert_allclose(mass, [[6.5]])
    np.testing.assert_allclose(com, [[[0.11, 0.02, -0.07]]])
    np.testing.assert_array_equal(info[TRUTH_KEY_ACTIVE], [False, True, True])
    np.testing.assert_array_equal(info[TRUTH_KEY_ATTACHED], [False, False, True])

    # Reconcile is allocation-free on valid carriers: arrays are rebound in place.
    active_ref = info[TRUTH_KEY_ACTIVE]
    event_ref = info[TRUTH_KEY_ATTACHED]
    added_ref = info[TRUTH_KEY_ADDED_MASS]
    owner.reconcile_info(info)
    assert info[TRUTH_KEY_ACTIVE] is active_ref
    assert info[TRUTH_KEY_ATTACHED] is event_ref
    assert info[TRUTH_KEY_ADDED_MASS] is added_ref
    np.testing.assert_array_equal(info[TRUTH_KEY_ACTIVE], [False, True, True])
    np.testing.assert_array_equal(info[TRUTH_KEY_ATTACHED], [False, False, False])
    np.testing.assert_array_equal(info[TRUTH_KEY_ADDED_MASS], [0.0, 1.5, 1.5])
