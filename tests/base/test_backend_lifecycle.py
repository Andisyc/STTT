from __future__ import annotations

import numpy as np
import pytest

from unilab.base.np_env import NpEnv


class _CloseRecordingBackend:
    def __init__(self) -> None:
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


class _ClosableNpEnv(NpEnv):
    @property
    def action_space(self):
        return None

    def apply_action(self, actions):
        return None

    def update_state(self, state):
        return state


def test_np_env_close_delegates_to_backend_owner() -> None:
    env = object.__new__(_ClosableNpEnv)
    backend = _CloseRecordingBackend()
    env._backend = backend

    env.close()

    assert backend.close_count == 1


def test_mujoco_backend_close_releases_pool_once_and_is_idempotent() -> None:
    pytest.importorskip("mujoco", reason="mujoco not installed")
    from unilab.base.backend.mujoco.backend import MuJoCoBackend

    class _Pool:
        def __init__(self) -> None:
            self.close_count = 0

        def close(self) -> None:
            self.close_count += 1

    class _CleanupHandle:
        def __init__(self) -> None:
            self.cleanup_count = 0

        def cleanup(self) -> None:
            self.cleanup_count += 1

    backend = object.__new__(MuJoCoBackend)
    pool = _Pool()
    shadow_pool = _Pool()
    cleanup_handle = _CleanupHandle()
    backend._pool = pool
    backend._isolated_rollout_pool = shadow_pool
    backend._isolated_rollout_active = False
    backend._scene_cleanup_handle = cleanup_handle

    backend.close()
    backend.close()

    assert pool.close_count == 1
    assert shadow_pool.close_count == 1
    assert cleanup_handle.cleanup_count == 1
    assert backend._pool is None
    assert backend._scene_cleanup_handle is None


def test_mujoco_backend_close_cleans_scene_assets_when_pool_close_fails() -> None:
    pytest.importorskip("mujoco", reason="mujoco not installed")
    from unilab.base.backend.mujoco.backend import MuJoCoBackend

    class _FailingPool:
        def close(self) -> None:
            raise RuntimeError("pool close failed")

    class _CleanupHandle:
        def __init__(self) -> None:
            self.cleanup_count = 0

        def cleanup(self) -> None:
            self.cleanup_count += 1

    backend = object.__new__(MuJoCoBackend)
    cleanup_handle = _CleanupHandle()
    backend._pool = _FailingPool()
    shadow_pool = _CloseRecordingBackend()
    backend._isolated_rollout_pool = shadow_pool
    backend._isolated_rollout_active = False
    backend._scene_cleanup_handle = cleanup_handle

    with pytest.raises(RuntimeError, match="pool close failed"):
        backend.close()

    assert cleanup_handle.cleanup_count == 1
    assert shadow_pool.close_count == 1
    assert backend._pool is None
    assert backend._scene_cleanup_handle is None


def test_mujoco_backend_isolated_branch_switches_only_active_pool() -> None:
    pytest.importorskip("mujoco", reason="mujoco not installed")
    from unilab.base.backend.mujoco.backend import MuJoCoBackend

    primary = object()
    sibling = object()
    backend = object.__new__(MuJoCoBackend)
    backend._pool = primary
    backend._isolated_rollout_pool = sibling
    backend._isolated_rollout_active = False

    with backend.isolated_rollout_branch():
        assert backend._pool is sibling
        with pytest.raises(RuntimeError, match="cannot be nested"):
            with backend.isolated_rollout_branch():
                pass

    assert backend._pool is primary
    assert backend._isolated_rollout_active is False


def test_mujoco_backend_prepares_sibling_from_cold_model_identity() -> None:
    pytest.importorskip("mujoco", reason="mujoco not installed")
    import unilab.base.backend.mujoco.backend as mujoco_backend

    models = [object(), object()]

    class _SiblingPool:
        def __init__(self, model, *, nbatch, nthread) -> None:
            self.models = model
            self.nbatch = nbatch
            self.nthread = nthread
            self.forward_calls: list[np.ndarray] = []
            self.close_count = 0

        def forward(self, state):
            self.forward_calls.append(np.asarray(state).copy())
            return np.zeros((2, 0), dtype=np.float64)

        def close(self) -> None:
            self.close_count += 1

    backend = object.__new__(mujoco_backend.MuJoCoBackend)
    backend._pool = object()
    backend._isolated_rollout_pool = None
    backend._isolated_rollout_active = False
    backend._reset_transaction_count = 0
    backend._num_envs = 2
    backend._n_threads = 2
    backend._model_variants = tuple(models)
    backend._model_assignments = np.asarray([0, 1], dtype=np.int32)
    backend._physics_state = np.arange(8, dtype=np.float32).reshape(2, 4)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(mujoco_backend, "BatchEnvPool", _SiblingPool)
    try:
        backend.prepare_isolated_rollout_branch()
    finally:
        monkeypatch.undo()

    sibling = backend._isolated_rollout_pool
    assert sibling.models == models
    assert sibling.nbatch == 2
    assert sibling.nthread == 2
    assert len(sibling.forward_calls) == 1
    np.testing.assert_array_equal(sibling.forward_calls[0], backend._physics_state)


def test_mujoco_backend_closes_partial_sibling_when_prepare_forward_fails() -> None:
    pytest.importorskip("mujoco", reason="mujoco not installed")
    import unilab.base.backend.mujoco.backend as mujoco_backend

    class _FailingSiblingPool:
        instances: list[_FailingSiblingPool] = []

        def __init__(self, *_args, **_kwargs) -> None:
            self.close_count = 0
            self.instances.append(self)

        def forward(self, _state):
            raise RuntimeError("sibling forward failed")

        def close(self) -> None:
            self.close_count += 1

    backend = object.__new__(mujoco_backend.MuJoCoBackend)
    backend._pool = object()
    backend._isolated_rollout_pool = None
    backend._isolated_rollout_active = False
    backend._reset_transaction_count = 0
    backend._num_envs = 1
    backend._n_threads = 1
    backend._model_variants = (object(),)
    backend._model_assignments = np.zeros((1,), dtype=np.int32)
    backend._physics_state = np.zeros((1, 3), dtype=np.float32)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(mujoco_backend, "BatchEnvPool", _FailingSiblingPool)
    try:
        with pytest.raises(RuntimeError, match="sibling forward failed"):
            backend.prepare_isolated_rollout_branch()
    finally:
        monkeypatch.undo()

    assert backend._isolated_rollout_pool is None
    assert _FailingSiblingPool.instances[0].close_count == 1


def test_mujoco_backend_mirrors_one_reset_transaction_to_both_pools() -> None:
    pytest.importorskip("mujoco", reason="mujoco not installed")
    from unilab.base.backend.mujoco.backend import MuJoCoBackend

    fields = {
        name: np.full((1, index + 1), index + 0.25, dtype=np.float64)
        for index, name in enumerate(
            (
                "body_mass",
                "body_ipos",
                "body_iquat",
                "body_inertia",
                "dof_armature",
                "gravity",
                "geom_friction",
                "kp",
                "kd",
            )
        )
    }

    class _Pool:
        def __init__(self, tag: str) -> None:
            self.tag = tag
            self.calls: list[dict[str, object]] = []

        def reset(self, *, env_ids, initial_state, randomization):
            self.calls.append(
                {
                    "env_ids": np.asarray(env_ids).copy(),
                    "initial_state": np.asarray(initial_state).copy(),
                    "randomization": {
                        key: np.asarray(value).copy() for key, value in randomization.items()
                    },
                }
            )
            return np.asarray(initial_state).copy(), np.zeros((1, 2), dtype=np.float64)

    backend = object.__new__(MuJoCoBackend)
    primary = _Pool("primary")
    sibling = _Pool("sibling")
    backend._pool = primary
    backend._isolated_rollout_pool = sibling
    backend._isolated_rollout_active = False
    backend._reset_transaction_count = 0
    backend._physics_state = np.zeros((2, 4), dtype=np.float32)
    backend._sensor_data = np.zeros((2, 2), dtype=np.float32)
    backend._np_dtype = np.dtype(np.float32)
    backend.nq = 2
    backend.nv = 1
    backend._idx_qpos = 1
    backend._idx_qvel = 3
    backend._translate_reset_randomization = lambda *_args: fields  # type: ignore[method-assign]

    backend.set_state(
        np.asarray([1], dtype=np.int32),
        np.asarray([[0.5, -0.5]], dtype=np.float64),
        np.asarray([[0.25]], dtype=np.float64),
        randomization=object(),  # type: ignore[arg-type]
    )

    assert len(primary.calls) == len(sibling.calls) == 1
    for key in ("env_ids", "initial_state"):
        np.testing.assert_array_equal(primary.calls[0][key], sibling.calls[0][key])
    assert set(primary.calls[0]["randomization"]) == set(fields)  # type: ignore[arg-type]
    for name in fields:
        np.testing.assert_array_equal(
            primary.calls[0]["randomization"][name],  # type: ignore[index]
            sibling.calls[0]["randomization"][name],  # type: ignore[index]
        )


@pytest.mark.parametrize("sibling_close_fails", [False, True])
def test_mujoco_backend_reset_sync_failure_poisoned_until_single_owner_close(
    sibling_close_fails: bool,
) -> None:
    pytest.importorskip("mujoco", reason="mujoco not installed")
    from unilab.base.backend.mujoco.backend import MuJoCoBackend

    class _PrimaryPool:
        def __init__(self) -> None:
            self.close_count = 0

        def reset(self, *, env_ids, initial_state, randomization):
            del env_ids, randomization
            return np.asarray(initial_state).copy(), np.zeros((1, 0), dtype=np.float64)

        def close(self) -> None:
            self.close_count += 1

    class _FailingSiblingPool:
        def __init__(self) -> None:
            self.close_count = 0

        def reset(self, *, env_ids, initial_state, randomization):
            del env_ids, initial_state, randomization
            raise RuntimeError("sibling reset failed")

        def close(self) -> None:
            self.close_count += 1
            if sibling_close_fails:
                raise RuntimeError("sibling close failed")

    class _CleanupHandle:
        def __init__(self) -> None:
            self.cleanup_count = 0

        def cleanup(self) -> None:
            self.cleanup_count += 1

    backend = object.__new__(MuJoCoBackend)
    primary = _PrimaryPool()
    sibling = _FailingSiblingPool()
    cleanup_handle = _CleanupHandle()
    backend._pool = primary
    backend._isolated_rollout_pool = sibling
    backend._isolated_rollout_active = False
    backend._isolated_rollout_failure = None
    backend._reset_transaction_count = 0
    backend._physics_state = np.zeros((1, 3), dtype=np.float32)
    backend._sensor_data = np.zeros((1, 0), dtype=np.float32)
    backend._np_dtype = np.dtype(np.float32)
    backend._scene_cleanup_handle = cleanup_handle
    backend.nq = 1
    backend.nv = 1
    backend._idx_qpos = 1
    backend._idx_qvel = 2
    backend._translate_reset_randomization = lambda *_args: None  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="sibling reset failed"):
        backend.set_state(
            np.asarray([0], dtype=np.int32),
            np.asarray([[0.1]], dtype=np.float64),
            np.asarray([[0.0]], dtype=np.float64),
        )

    assert sibling.close_count == 0
    with pytest.raises(RuntimeError, match="isolated rollout pool is poisoned"):
        backend.step(np.zeros((1, 1), dtype=np.float32))
    with pytest.raises(RuntimeError, match="isolated rollout pool is poisoned"):
        backend.set_state(
            np.asarray([], dtype=np.int32),
            np.empty((0, 1), dtype=np.float64),
            np.empty((0, 1), dtype=np.float64),
        )
    with pytest.raises(RuntimeError, match="isolated rollout pool is poisoned"):
        with backend.isolated_rollout_branch():
            pass

    if sibling_close_fails:
        with pytest.raises(RuntimeError, match="sibling close failed"):
            backend.close()
    else:
        backend.close()
    backend.close()

    assert sibling.close_count == 1
    assert primary.close_count == 1
    assert cleanup_handle.cleanup_count == 1


def test_mujoco_backend_rejects_sibling_prepare_after_any_reset_transaction() -> None:
    pytest.importorskip("mujoco", reason="mujoco not installed")
    from unilab.base.backend.mujoco.backend import MuJoCoBackend

    class _Pool:
        def reset(self, *, env_ids, initial_state, randomization):
            del env_ids, randomization
            return np.asarray(initial_state).copy(), np.zeros((1, 0), dtype=np.float64)

    backend = object.__new__(MuJoCoBackend)
    backend._pool = _Pool()
    backend._isolated_rollout_pool = None
    backend._isolated_rollout_active = False
    backend._reset_transaction_count = 0
    backend._physics_state = np.zeros((1, 3), dtype=np.float32)
    backend._sensor_data = np.zeros((1, 0), dtype=np.float32)
    backend._np_dtype = np.dtype(np.float32)
    backend.nq = 1
    backend.nv = 1
    backend._idx_qpos = 1
    backend._idx_qvel = 2
    backend._translate_reset_randomization = lambda *_args: None  # type: ignore[method-assign]

    backend.set_state(
        np.asarray([0], dtype=np.int32),
        np.asarray([[0.1]], dtype=np.float64),
        np.asarray([[0.0]], dtype=np.float64),
    )

    assert backend._reset_transaction_count == 1
    with pytest.raises(RuntimeError, match="before any MuJoCo reset transaction"):
        backend.prepare_isolated_rollout_branch()


def test_mujoco_isolated_pool_matches_primary_dr_fields_and_next_transition() -> None:
    mujoco = pytest.importorskip("mujoco", reason="mujoco not installed")
    from mujoco.batch_env import BatchEnvPool

    from unilab.base.backend.mujoco.backend import MuJoCoBackend
    from unilab.dr import ResetRandomizationPayload

    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option timestep="0.01"/>
          <worldbody>
            <geom name="floor" type="plane" size="1 1 0.1"/>
            <body name="link" pos="0 0 1">
              <joint name="hinge" type="hinge"/>
              <geom name="link_geom" type="capsule" size="0.05 0.2" mass="1"/>
            </body>
          </worldbody>
          <actuator><position name="motor" joint="hinge" kp="2"/></actuator>
          <sensor><jointpos name="hinge_pos" joint="hinge"/></sensor>
        </mujoco>
        """
    )
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    backend = object.__new__(MuJoCoBackend)
    backend._model = model
    backend._model_variants = (model,)
    backend._model_assignments = np.zeros((1,), dtype=np.int32)
    backend._pool = BatchEnvPool(model, nbatch=1, nthread=1)
    backend._isolated_rollout_pool = None
    backend._isolated_rollout_active = False
    backend._reset_transaction_count = 0
    backend._num_envs = 1
    backend._n_threads = 1
    backend._np_dtype = np.dtype(np.float32)
    backend._physics_state = np.zeros((1, nstate), dtype=np.float32)
    backend._sensor_data = np.zeros((1, model.nsensordata), dtype=np.float32)
    backend._pending_xfrc_applied = np.zeros((1, 6 * model.nbody), dtype=np.float64)
    backend._pre_step_control_fn = None
    backend._post_step_forward_sensor = False
    backend._idx_qpos = 1
    backend._idx_qvel = 1 + model.nq
    backend.nq = model.nq
    backend.nv = model.nv
    backend._base_name = "link"
    backend._base_body_id = 1
    backend._base_body_mass = np.asarray(model.body_mass).copy()
    backend._base_body_ipos = np.asarray(model.body_ipos).copy()
    backend._scene_cleanup_handle = None
    backend._pool.forward(backend._physics_state)
    backend.prepare_isolated_rollout_branch()

    primary = backend._pool
    sibling = backend._isolated_rollout_pool
    assert sibling is not None
    fields = {
        name: primary.get_field(0, name).copy()
        for name in (
            "body_mass",
            "body_ipos",
            "body_iquat",
            "body_inertia",
            "dof_armature",
            "gravity",
            "geom_friction",
            "kp",
            "kd",
        )
    }
    fields["body_mass"][1] *= 1.1
    fields["dof_armature"][:] = 0.02
    fields["gravity"][2] = -9.7
    fields["geom_friction"][:] *= 0.95
    fields["kp"][:] = 2.2
    fields["kd"][:] = 0.1
    payload = ResetRandomizationPayload(
        base_com_offset=np.asarray([[0.01, -0.02, 0.03]], dtype=np.float64),
        body_mass=fields["body_mass"][None, :],
        body_iquat=fields["body_iquat"].reshape(1, model.nbody, 4),
        body_inertia=fields["body_inertia"].reshape(1, model.nbody, 3),
        dof_armature=fields["dof_armature"][None, :],
        gravity=fields["gravity"][None, :],
        geom_friction=fields["geom_friction"].reshape(1, model.ngeom, 3),
        kp=fields["kp"][None, :],
        kd=fields["kd"][None, :],
    )
    backend.set_state(
        np.asarray([0], dtype=np.int32),
        np.asarray([[0.15]], dtype=np.float64),
        np.asarray([[0.0]], dtype=np.float64),
        randomization=payload,
    )

    for name in fields:
        np.testing.assert_array_equal(primary.get_field(0, name), sibling.get_field(0, name))

    initial_state = backend._physics_state.copy()
    ctrl = np.asarray([[0.2]], dtype=np.float32)
    backend.step(ctrl)
    primary_state = backend._physics_state.copy()
    primary_sensor = backend._sensor_data.copy()
    backend._physics_state[:] = initial_state
    with backend.isolated_rollout_branch():
        backend.step(ctrl)
        sibling_state = backend._physics_state.copy()
        sibling_sensor = backend._sensor_data.copy()

    np.testing.assert_array_equal(primary_state, sibling_state)
    np.testing.assert_array_equal(primary_sensor, sibling_sensor)
    backend.close()


def test_mujoco_runtime_body_mass_and_com_preserves_state() -> None:
    mujoco = pytest.importorskip("mujoco", reason="mujoco not installed")
    from mujoco.batch_env import BatchEnvPool

    from unilab.base.backend.mujoco.backend import MuJoCoBackend

    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="link">
              <joint name="hinge" type="hinge"/>
              <geom type="capsule" size="0.05 0.2" mass="1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    num_envs = 2
    nstate = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    backend = object.__new__(MuJoCoBackend)
    backend._model = model
    backend._model_variants = (model,)
    backend._model_assignments = np.zeros((num_envs,), dtype=np.int32)
    backend._pool = BatchEnvPool(model, nbatch=num_envs, nthread=1)
    backend._isolated_rollout_pool = None
    backend._isolated_rollout_active = False
    backend._isolated_rollout_failure = None
    backend._reset_transaction_count = 0
    backend._num_envs = num_envs
    backend._n_threads = 1
    backend._np_dtype = np.dtype(np.float32)
    backend._physics_state = np.zeros((num_envs, nstate), dtype=np.float32)
    backend._sensor_data = np.zeros((num_envs, model.nsensordata), dtype=np.float32)
    backend._pending_xfrc_applied = np.zeros(
        (num_envs, 6 * model.nbody), dtype=np.float64
    )
    backend._pre_step_control_fn = None
    backend._post_step_forward_sensor = False
    backend._idx_qpos = 1
    backend._idx_qvel = 1 + model.nq
    backend.nq = model.nq
    backend.nv = model.nv
    backend._base_name = "link"
    backend._base_body_id = 1
    backend._base_body_mass = np.asarray(model.body_mass).copy()
    backend._base_body_ipos = np.asarray(model.body_ipos).copy()
    backend._scene_cleanup_handle = None
    backend._pool.forward(backend._physics_state)
    backend.prepare_isolated_rollout_branch()

    primary = backend._pool
    sibling = backend._isolated_rollout_pool
    assert sibling is not None
    body_id = backend._base_body_id
    original_mass = [primary.get_field(i, "body_mass").copy() for i in range(num_envs)]
    original_com = [
        primary.get_field(i, "body_ipos").reshape(model.nbody, 3).copy()
        for i in range(num_envs)
    ]
    physics_state = backend._physics_state.copy()
    reset_count = backend._reset_transaction_count
    updated_mass = np.asarray([[original_mass[0][body_id] + 2.0]], dtype=np.float64)
    updated_com = np.asarray(
        [[original_com[0][body_id] + np.array([0.01, -0.02, 0.03])]],
        dtype=np.float64,
    )

    try:
        backend.set_body_mass_and_com(
            np.asarray([0], dtype=np.int32),
            np.asarray([body_id], dtype=np.int32),
            mass=updated_mass,
            center_of_mass=updated_com,
        )

        np.testing.assert_allclose(primary.get_field(0, "body_mass")[body_id], updated_mass[0, 0])
        np.testing.assert_allclose(
            primary.get_field(0, "body_ipos").reshape(model.nbody, 3)[body_id],
            updated_com[0, 0],
        )
        np.testing.assert_array_equal(primary.get_field(1, "body_mass"), original_mass[1])
        np.testing.assert_array_equal(
            primary.get_field(1, "body_ipos").reshape(model.nbody, 3), original_com[1]
        )
        np.testing.assert_array_equal(sibling.get_field(0, "body_mass"), primary.get_field(0, "body_mass"))
        np.testing.assert_array_equal(sibling.get_field(0, "body_ipos"), primary.get_field(0, "body_ipos"))
        np.testing.assert_array_equal(backend._physics_state, physics_state)
        assert backend._reset_transaction_count == reset_count

        backend.set_body_mass_and_com(
            np.asarray([0], dtype=np.int32),
            np.asarray([body_id], dtype=np.int32),
            mass=np.asarray([[original_mass[0][body_id]]], dtype=np.float64),
            center_of_mass=np.asarray([[original_com[0][body_id]]], dtype=np.float64),
        )
        np.testing.assert_array_equal(primary.get_field(0, "body_mass"), original_mass[0])
        np.testing.assert_array_equal(
            primary.get_field(0, "body_ipos").reshape(model.nbody, 3), original_com[0]
        )
    finally:
        backend.close()


def test_mujoco_backend_rejects_invalid_thread_count_before_loading_scene() -> None:
    pytest.importorskip("mujoco", reason="mujoco not installed")
    from unilab.base.backend.mujoco.backend import MuJoCoBackend
    from unilab.base.scene import SceneCfg

    with pytest.raises(ValueError, match="num_threads must be between 1 and num_envs"):
        MuJoCoBackend(
            SceneCfg(model_file="intentionally-missing-model.xml"),
            num_envs=64,
            sim_dt=0.01,
            num_threads=0,
        )
