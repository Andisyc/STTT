"""Geometry, timing and lifecycle tests for the MuJoCo stereo/IMU sensor stream.

These tests check physical properties (rigid-body transforms from real
kinematics, substep-derived IMU counts and timestamps, reset sequencing,
resource lifecycle, and the no-stream physics baseline) instead of forwarding
field values.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco", reason="mujoco not installed")

try:
    from mujoco.batch_env import BatchEnvPool  # noqa: F401
except Exception:  # pragma: no cover - platform/libstdc++ issue
    pytest.skip("mujoco.batch_env not available", allow_module_level=True)

import mujoco  # noqa: E402

from unilab.assets import ASSETS_ROOT_PATH  # noqa: E402
from unilab.base.backend.mujoco.backend import MuJoCoBackend  # noqa: E402
from unilab.base.backend.sensors import StereoSensorConfig  # noqa: E402
from unilab.base.scene import SceneCfg  # noqa: E402

MODEL_FILE = str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")
SIM_DT = 0.005
CAMERA_PERIOD_STEPS = 5
BASELINE_M = 0.06
WIDTH, HEIGHT = 64, 48
CAMERA_POSITION = (0.08, 0.0, 0.30)


def _open_backend(*, num_envs: int = 1) -> MuJoCoBackend:
    backend = MuJoCoBackend(
        SceneCfg(model_file=MODEL_FILE),
        num_envs=num_envs,
        sim_dt=SIM_DT,
        base_name="pelvis",
    )
    backend.materialize()
    return backend


def _config(**overrides) -> StereoSensorConfig:
    defaults = {
        "body_name": "torso_link",
        "position": CAMERA_POSITION,
        "camera_period_steps": CAMERA_PERIOD_STEPS,
        "baseline_m": BASELINE_M,
        "width": WIDTH,
        "height": HEIGHT,
    }
    return StereoSensorConfig(**{**defaults, **overrides})


def _hold_ctrl(backend: MuJoCoBackend) -> np.ndarray:
    return np.asarray(backend.model.qpos0[-backend.num_actuators :], dtype=np.float64)[None, :]


def _forward_data(backend: MuJoCoBackend, ctrl: np.ndarray) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Re-evaluate the current physics state with the control that produced it."""
    model = backend.model
    data = mujoco.MjData(model)
    state = np.ascontiguousarray(backend.get_physics_state()[0], dtype=np.float64)
    mujoco.mj_setState(model, data, state, int(mujoco.mjtState.mjSTATE_FULLPHYSICS))
    data.ctrl[:] = ctrl[0]
    mujoco.mj_forward(model, data)
    return model, data


def test_imu_runs_at_every_physics_substep_and_frames_group_by_period() -> None:
    backend = _open_backend()
    stream = backend.open_sensor_stream(_config())
    try:
        backend.reset_physics_rows_to_default(np.asarray([0]))
        stream.drain()
        stream.drain_reference_poses()

        backend.step(_hold_ctrl(backend), nsteps=CAMERA_PERIOD_STEPS)

        frames = stream.drain()
        assert len(frames) == 1
        frame = frames[0]
        assert frame.left_rgb.shape == (HEIGHT, WIDTH, 3)
        assert frame.right_rgb.shape == (HEIGHT, WIDTH, 3)
        assert frame.left_rgb.dtype == np.uint8
        # One sample per real substep; never a repeated sample standing in for a
        # higher rate.
        assert len(frame.imu) == CAMERA_PERIOD_STEPS
        assert stream.calibration.imu_frequency_hz == pytest.approx(1.0 / SIM_DT)
        assert stream.calibration.focal[0] == pytest.approx(0.5 * HEIGHT / np.tan(np.radians(35.0)))

        stamps = np.asarray([sample.timestamp_ns for sample in frame.imu])
        period_ns = int(round(SIM_DT * 1e9))
        np.testing.assert_array_equal(np.diff(stamps), np.full(CAMERA_PERIOD_STEPS - 1, period_ns))
        assert frame.timestamp_ns == int(stamps[-1])

        references = stream.drain_reference_poses()
        assert len(references) == 1
        assert references[0].timestamp_ns == frame.timestamp_ns
        assert references[0].sequence_id == frame.sequence_id
        np.testing.assert_allclose(references[0].world_from_rig[3], [0.0, 0.0, 0.0, 1.0])
    finally:
        stream.close()
        backend.close()


def test_imu_sample_matches_site_kinematics_at_the_frame_timestamp() -> None:
    backend = _open_backend()
    stream = backend.open_sensor_stream(_config())
    try:
        backend.reset_physics_rows_to_default(np.asarray([0]))
        stream.drain()
        stream.drain_reference_poses()
        ctrl = _hold_ctrl(backend)
        backend.step(ctrl, nsteps=CAMERA_PERIOD_STEPS)
        sample = stream.drain()[0].imu[-1]

        model, data = _forward_data(backend, ctrl)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu_in_torso")
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site_id, velocity, 1)
        acceleration = np.zeros(6)
        mujoco.mj_objectAcceleration(
            model, data, mujoco.mjtObj.mjOBJ_SITE, site_id, acceleration, 1
        )
        np.testing.assert_allclose(sample.angular_velocity, velocity[:3], atol=1e-6)
        np.testing.assert_allclose(sample.acceleration, acceleration[3:], atol=1e-4)
    finally:
        stream.close()
        backend.close()


def test_stereo_rig_geometry_matches_forward_kinematics() -> None:
    backend = _open_backend()
    stream = backend.open_sensor_stream(_config())
    try:
        backend.reset_physics_rows_to_default(np.asarray([0]))
        stream.drain()
        stream.drain_reference_poses()
        ctrl = _hold_ctrl(backend)
        backend.step(ctrl, nsteps=CAMERA_PERIOD_STEPS)
        reference = stream.drain_reference_poses()[0].world_from_rig
        calibration = stream.calibration

        np.testing.assert_allclose(calibration.rig_from_left, np.eye(4), atol=0.0)
        np.testing.assert_allclose(calibration.rig_from_right[:3, 3], [BASELINE_M, 0.0, 0.0])
        np.testing.assert_allclose(calibration.rig_from_right[:3, :3], np.eye(3), atol=1e-12)

        model, data = _forward_data(backend, ctrl)
        torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        torso_axes = data.xmat[torso_id].reshape(3, 3)
        torso_position = np.asarray(data.xpos[torso_id], dtype=np.float64)

        # The chest camera is rigidly bolted at the configured body-frame offset,
        # rotated by the configured xyaxes (rig x = -body y, rig y = -body z,
        # rig z = +body x, i.e. looking along torso +x).
        np.testing.assert_allclose(
            reference[:3, 3],
            torso_position + torso_axes @ np.asarray(CAMERA_POSITION),
            atol=1e-9,
        )
        body_from_rig = torso_axes.T @ reference[:3, :3]
        np.testing.assert_allclose(
            body_from_rig,
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
            atol=1e-9,
        )

        # rig_from_imu is the real static torso IMU <-> camera transform.
        imu_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu_in_torso")
        world_from_imu = np.eye(4)
        world_from_imu[:3, :3] = data.site_xmat[imu_site].reshape(3, 3)
        world_from_imu[:3, 3] = data.site_xpos[imu_site]
        np.testing.assert_allclose(reference @ calibration.rig_from_imu, world_from_imu, atol=1e-8)
        # ...and that IMU really belongs to the chest body, not the pelvis.
        assert int(model.site_bodyid[imu_site]) == torso_id
    finally:
        stream.close()
        backend.close()


def test_reset_bumps_sequence_and_drops_in_flight_samples() -> None:
    backend = _open_backend()
    stream = backend.open_sensor_stream(_config())
    try:
        backend.reset_physics_rows_to_default(np.asarray([0]))
        stream.drain()
        stream.drain_reference_poses()
        ctrl = _hold_ctrl(backend)

        backend.step(ctrl, nsteps=CAMERA_PERIOD_STEPS)
        first = stream.drain()[0]
        stream.drain_reference_poses()

        # Two substeps of in-flight IMU data, then reset, then a full period.
        backend.step(ctrl, nsteps=2)
        backend.reset_physics_rows_to_default(np.asarray([0]))
        backend.step(ctrl, nsteps=CAMERA_PERIOD_STEPS)

        frames = stream.drain()
        assert len(frames) == 1
        frame = frames[0]
        assert frame.sequence_id > first.sequence_id
        assert len(frame.imu) == CAMERA_PERIOD_STEPS
        reference = stream.drain_reference_poses()
        assert len(reference) == 1
        assert reference[0].sequence_id == frame.sequence_id
        assert reference[0].timestamp_ns == frame.timestamp_ns
    finally:
        stream.close()
        backend.close()


def test_close_is_idempotent_and_restores_the_plain_step_path() -> None:
    backend = _open_backend()
    stream = backend.open_sensor_stream(_config())
    try:
        backend.reset_physics_rows_to_default(np.asarray([0]))
        ctrl = _hold_ctrl(backend)

        with pytest.raises(RuntimeError, match="already open"):
            backend.open_sensor_stream(_config())

        stream.close()
        stream.close()
        assert backend._sensor_stream is None

        # The plain (no-stream) path still steps normally.
        backend.step(ctrl, nsteps=CAMERA_PERIOD_STEPS)
        # Draining a closed stream is safe but never invents data, and sampling
        # into it is refused rather than silently ignored.
        assert stream.drain() == ()
        assert stream.drain_reference_poses() == ()
        with pytest.raises(RuntimeError, match="closed"):
            stream.observe_substep(state_row=np.zeros(8), control_row=np.zeros(1), timestamp_ns=0)
    finally:
        backend.close()
        backend.close()


def test_single_environment_requirement_is_enforced() -> None:
    backend = _open_backend(num_envs=2)
    try:
        with pytest.raises(ValueError, match="exactly one environment"):
            backend.open_sensor_stream(_config())
    finally:
        backend.close()


def test_misplaced_imu_site_is_rejected_instead_of_relabelled() -> None:
    backend = _open_backend()
    try:
        with pytest.raises(ValueError, match="not rigidly attached"):
            backend.open_sensor_stream(
                _config(gyro_sensor="pelvis_gyro", accelerometer_sensor="pelvis_acceleration")
            )
    finally:
        backend.close()


def test_enabling_the_stream_does_not_change_the_physics_trajectory() -> None:
    """Sampling must ride the same substeps, not perturb the dynamics.

    The baseline is the backend's own single-substep loop (the pre-step-control
    shape), so this compares the *stream-enabled* path against the plain
    single-substep path and requires bitwise equality.
    """
    backend = _open_backend()
    try:
        ctrl = _hold_ctrl(backend)
        backend.reset_physics_rows_to_default(np.asarray([0]))
        baseline = []
        for _ in range(4):
            for _ in range(CAMERA_PERIOD_STEPS):
                backend.step(ctrl, nsteps=1)
            baseline.append(backend.get_physics_state()[0].copy())
    finally:
        backend.close()

    backend = _open_backend()
    stream = backend.open_sensor_stream(_config())
    try:
        backend.reset_physics_rows_to_default(np.asarray([0]))
        for expected in baseline:
            backend.step(ctrl, nsteps=CAMERA_PERIOD_STEPS)
            np.testing.assert_array_equal(backend.get_physics_state()[0], expected)
        assert len(stream.drain()) == len(baseline)
    finally:
        stream.close()
        backend.close()


def test_undrained_backlog_fails_loudly_instead_of_dropping_frames() -> None:
    backend = _open_backend()
    stream = backend.open_sensor_stream(_config())
    try:
        backend.reset_physics_rows_to_default(np.asarray([0]))
        ctrl = _hold_ctrl(backend)
        stream._max_pending_frames = 1
        backend.step(ctrl, nsteps=CAMERA_PERIOD_STEPS)
        state_before = backend.get_physics_state()[0].copy()
        with pytest.raises(RuntimeError, match="backlog"):
            backend.step(ctrl, nsteps=CAMERA_PERIOD_STEPS)
        # The refused step did not advance physics and did not drop the frame.
        np.testing.assert_array_equal(backend.get_physics_state()[0], state_before)
        assert len(stream.drain()) == 1
    finally:
        stream.close()
        backend.close()
