"""Native float64 state -> projected observations -> timed window, with no physics rollout.

The pool supplies controlled double-precision state increments. Real backend
step/reset/snapshot methods and real pose/window/storage consumers are exercised.
"""
from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import torch

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.backend.mujoco.backend import MuJoCoBackend
from unilab.base.scene import SceneCfg
from unilab.algos.torch.distill.fada.runtime_policy import FADAFutureQuery, OfficialFADAPlaybackProfile
from unilab.algos.torch.sttt.odometry_runtime import create_pose_runtime
from unilab.algos.torch.sttt.pose_data import ControlInterval, PoseRuntimeConfig
from unilab.algos.torch.sttt.path_following.collection import WindowCollector
from unilab.algos.torch.sttt.path_following.dataset import PathFollowingDatasetWriter, iter_path_following_samples

POSITION_INCREMENT = 0.001234567890123
VELOCITY_INCREMENT = 0.0001234567890123


class ControlledPool:
    def __init__(self, backend):
        self.backend = backend
        self.sim_dt = 0.005

    def step(self, state, *, nstep, **_):
        result = np.asarray(state, dtype=np.float64).copy()
        for _ in range(nstep):
            result[:, 0] += self.sim_dt
            result[:, self.backend._idx_qpos] += POSITION_INCREMENT
            result[:, self.backend._idx_qvel] += VELOCITY_INCREMENT
        sensors = np.zeros(self.backend._sensor_data.shape, dtype=np.float64)
        return result, sensors

    def reset(self, *, initial_state, **_):
        return initial_state.copy(), np.zeros((len(initial_state), self.backend._sensor_data.shape[1]), dtype=np.float64)

    def close(self):
        pass


class TimestampStream:
    def __init__(self):
        self.timestamp_ns = None
        self.state_time = None

    def ensure_capacity(self, **_):
        pass

    def observe_substep(self, *, state_row, timestamp_ns, **_):
        self.timestamp_ns = timestamp_ns
        self.state_time = float(state_row[0])

    def observe_reset(self, *, state_row, timestamp_ns):
        self.timestamp_ns = timestamp_ns
        self.state_time = float(state_row[0])

    def close(self):
        pass


def make_backend(*, path='plain', dtype=np.float32, num_envs=1):
    backend = MuJoCoBackend(
        SceneCfg(model_file=str(ASSETS_ROOT_PATH / 'robots/g1/scene_flat.xml')),
        num_envs=num_envs, sim_dt=0.005, base_name='pelvis', np_dtype=dtype,
    )
    object.__setattr__(backend, '_pool', ControlledPool(backend))
    if path == 'callback':
        backend.set_pre_step_control(lambda backend, control: control)
    elif path == 'sensor':
        object.__setattr__(backend, '_sensor_stream', TimestampStream())
    return backend


def query():
    profile = OfficialFADAPlaybackProfile(
        source_identity={'run': 'controlled-clock-not-physics'}, joint_names=tuple(f'joint_{i}' for i in range(29)),
        default_joint_pos=(0.,) * 29, action_scale=(0.25,) * 29, stiffness=(40.,) * 29,
        damping=(2.,) * 29, clip_actions=1., ctrl_dt=0.02,
        gait_frequency=1., gait_clock_mode='continuous', phase_init=(0., 3.14),
    )
    return FADAFutureQuery(
        torch.zeros(1, 2, 64), torch.zeros(1, 2, 29), torch.tensor([[0.4, 0., 0., 0., 0., 1., 1.]]),
        torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 3, 64),
        {'term_scale': {'dof_pos': 2.5, 'dof_vel': 0.15}}, profile,
    )


@pytest.mark.parametrize('path', ['plain', 'callback', 'sensor'])
@pytest.mark.parametrize('start_s', [0., 120.])
def test_native_clock_reaches_complete_stored_window_without_float32_rounding(tmp_path, path, start_s):
    backend = make_backend(path=path)
    backend.get_physics_state()[0, 0] = start_s
    poses = create_pose_runtime(backend, PoseRuntimeConfig())
    collector = WindowCollector(10, max_alignment_age_ns=200_000_000)
    collector.submit_candidate(torch.full((1, 29), 0.001))
    inputs = query()
    samples = []
    held_position = backend.get_base_pos()
    try:
        for step in range(10):
            before = poses.begin_step()
            collector.begin_step(before)
            collector.transform(inputs)
            backend.step(np.zeros((1, 29)), nsteps=4)
            after = poses.clock
            assert abs(after.timestamp_ns - before.timestamp_ns - 20_000_000) <= 1
            control = ControlInterval(before.sequence_id, step, before.timestamp_ns, after.timestamp_ns,
                                      np.zeros(29), inputs.command[0, :3].numpy())
            samples.extend(collector.complete_step(control, poses.complete_step(control)))
            assert held_position.dtype == np.float32
            np.testing.assert_array_equal(held_position, backend.get_physics_state()[:, 1:4].astype(np.float32))
            if path == 'sensor':
                assert cast(TimestampStream, backend._sensor_stream).timestamp_ns == after.timestamp_ns
        sample, = samples
        assert sample.controls[0].start_ns == round(start_s * 1e9)
        assert abs(sample.controls[-1].end_ns - round((start_s + 0.2) * 1e9)) <= 1
        np.testing.assert_allclose(sample.delta, [40 * POSITION_INCREMENT, 0., 0.], rtol=0, atol=1e-7)
        writer = PathFollowingDatasetWriter(tmp_path, source={'evidence': 'controlled-native-return-not-physics'})
        writer.append(sample)
        writer.close()
        restored, = iter_path_following_samples(tmp_path)
        assert restored.controls[-1].end_ns == sample.controls[-1].end_ns
        np.testing.assert_array_equal(restored.delta, sample.delta)
    finally:
        collector.close()
        poses.close()
        backend.close()


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_snapshot_restore_preserves_native_time_and_preexisting_observation_views(dtype):
    backend = make_backend(dtype=dtype)
    try:
        backend.get_physics_state()[0, 0] = 120.123456789
        position, velocity = backend.get_base_pos(), backend.get_base_lin_vel()
        backend.step(np.zeros((1, 29)), nsteps=4)
        snapshot = backend.capture_rollout_state()
        saved_position, saved_velocity = position.copy(), velocity.copy()
        saved_time = backend.get_sim_time_ns()
        backend.step(np.zeros((1, 29)), nsteps=4)
        expected_next = backend.get_physics_state().copy()
        backend.restore_rollout_state(snapshot)
        assert backend.get_sim_time_ns() == saved_time
        np.testing.assert_array_equal(position, saved_position)
        np.testing.assert_array_equal(velocity, saved_velocity)
        assert position.dtype == velocity.dtype == np.dtype(dtype)
        backend.step(np.zeros((1, 29)), nsteps=4)
        np.testing.assert_array_equal(backend.get_physics_state(), expected_next)
        assert backend.get_physics_state().dtype == np.float64
    finally:
        backend.close()


def test_partial_reset_refreshes_only_reset_rows_and_keeps_native_joint_precision():
    backend = make_backend(num_envs=2)
    try:
        backend.step(np.zeros((2, 29)), nsteps=4)
        native_before = backend.get_physics_state().copy()
        held_position = backend.get_base_pos()
        qpos = backend.model.qpos0[None].copy()
        qpos[0, 0] = 0.123456789012345
        backend.set_state(np.array([1]), qpos, np.zeros((1, backend.nv)))
        np.testing.assert_array_equal(backend.get_physics_state()[0], native_before[0])
        assert backend.get_physics_state()[1, 0] == 0.
        assert backend.get_physics_state()[1, 1] == qpos[0, 0]
        assert held_position[1, 0] == np.float32(qpos[0, 0])
        assert held_position[0, 0] == np.float32(native_before[0, 1])
    finally:
        backend.close()


def test_interval_velocity_mutates_next_native_step_not_just_observation_projection():
    backend = make_backend(path='callback')
    try:
        held_velocity = backend.get_base_lin_vel()
        delta = np.array([[[0.25, -0.5, 0.125]]], dtype=np.float64)
        backend.apply_body_linear_velocity_delta(np.array([backend.get_body_id('pelvis')]), delta)
        backend.step(np.zeros((1, 29)), nsteps=4)
        expected = delta[:, 0].copy()
        expected[:, 0] += 4 * VELOCITY_INCREMENT
        np.testing.assert_allclose(backend.get_physics_state()[:, backend._idx_qvel:backend._idx_qvel + 3], expected, atol=1e-15, rtol=0)
        np.testing.assert_array_equal(held_velocity, expected.astype(np.float32))
    finally:
        backend.close()


def test_incorrect_native_period_is_still_rejected_by_the_collector():
    backend = make_backend()
    cast(ControlledPool, backend._pool).sim_dt = 0.005000025  # A real period mismatch must still fail.
    poses = create_pose_runtime(backend, PoseRuntimeConfig())
    collector = WindowCollector(10, max_alignment_age_ns=200_000_000)
    inputs = query()
    collector.submit_candidate(torch.zeros(1, 29))
    try:
        before = poses.begin_step()
        collector.begin_step(before)
        collector.transform(inputs)
        backend.step(np.zeros((1, 29)), nsteps=4)
        after = poses.clock
        control = ControlInterval(before.sequence_id, 0, before.timestamp_ns, after.timestamp_ns, np.zeros(29), inputs.command[0, :3].numpy())
        with pytest.raises(ValueError, match='physical period'):
            collector.complete_step(control, poses.complete_step(control))
    finally:
        collector.close()
        poses.close()
        backend.close()
