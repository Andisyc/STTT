"""Opt-in MuJoCo stereo camera and IMU sampling for one environment.

The physics pool keeps stepping the ``discardvisual`` model while this module
owns an independent, complete visual model plus its ``MjData``. Cameras are
attached on the cold path; sampling rides the normal stepping path, so nothing
here is wall-clock paced:

* one IMU sample per real physics substep (frequency ``1 / sim_dt``),
* one stereo frame every ``camera_period_steps`` substeps, both images rendered
  from the same physics snapshot,
* one relative pelvis-from-rig FK snapshot per captured frame (no world pose),
* ground-truth rig poses on a separate evaluation channel.

``clock`` reports the current sequence id and the physical simulation time of the
last integrated substep (or the freshly reset state), so it is meaningful before
the first frame is captured. No wall-clock or ``ctrl_step * dt`` value is used.

The adapter never fakes a higher rate than the physics substep: a caller asking
for ``nstep > 1`` is stepped one substep at a time and every substep is sampled.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from ..sensors import (
    BackendSensorStream,
    ImuSample,
    RigKinematicsSample,
    RigReferencePose,
    SensorClock,
    StereoCalibration,
    StereoFrame,
    StereoSensorConfig,
)

_DEFAULT_MAX_PENDING_FRAMES = 512
"""Frames retained without draining before sampling fails loudly.

The same bound applies to the kinematics and reference channels, so none of the
three can silently fall behind the frames.
"""

# MuJoCo camera frame (x right, y up, -z view) to optical frame (x right, y down, z forward).
_OPTICAL_ROTATION = np.diag(np.array([1.0, -1.0, -1.0]))


def _se3(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def _invert_se3(matrix: np.ndarray) -> np.ndarray:
    rotation = matrix[:3, :3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ matrix[:3, 3]
    return inverse


def _mat_from_quat(quat: np.ndarray) -> np.ndarray:
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, np.ascontiguousarray(quat, dtype=np.float64))
    return mat.reshape(3, 3)


def _quat_from_xyaxes(xyaxes) -> np.ndarray:
    """Rebuild MuJoCo's own orthogonalization of a camera ``xyaxes`` attribute."""
    x_axis = np.asarray(xyaxes[:3], dtype=np.float64)
    y_axis = np.asarray(xyaxes[3:], dtype=np.float64)
    x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    quat = np.zeros(4, dtype=np.float64)
    matrix = np.column_stack((x_axis, y_axis, z_axis)).reshape(9)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(matrix))
    return quat


def _named_id(model: mujoco.MjModel, objtype, name: str, *, role: str) -> int:
    obj_id = mujoco.mj_name2id(model, objtype, name)
    if obj_id < 0:
        raise ValueError(f"{role} '{name}' not found in the MuJoCo model")
    return int(obj_id)


def _resolve_imu_sensor(
    model: mujoco.MjModel,
    *,
    name: str,
    sensor_type: int,
    role: str,
) -> tuple[int, int]:
    """Return ``(site_id, sensordata_address)`` for one 3-axis site sensor."""
    sensor_id = _named_id(model, mujoco.mjtObj.mjOBJ_SENSOR, name, role=f"{role} sensor")
    if int(model.sensor_type[sensor_id]) != int(sensor_type):
        raise ValueError(f"{role} sensor '{name}' has the wrong MuJoCo sensor type")
    if int(model.sensor_dim[sensor_id]) != 3:
        raise ValueError(f"{role} sensor '{name}' must report three axes")
    if int(model.sensor_objtype[sensor_id]) != int(mujoco.mjtObj.mjOBJ_SITE):
        raise ValueError(f"{role} sensor '{name}' must be attached to a site")
    return int(model.sensor_objid[sensor_id]), int(model.sensor_adr[sensor_id])


@dataclass(frozen=True)
class _RigGeometry:
    left_camera_id: int
    right_camera_id: int
    pelvis_body_id: int
    gyro_address: int
    accelerometer_address: int
    up_in_world: np.ndarray
    calibration: StereoCalibration


def _build_visual_model(
    source: str,
    config: StereoSensorConfig,
    sim_dt: float,
) -> mujoco.MjModel:
    """Compile a full visual model with the chest stereo pair attached."""
    spec = mujoco.MjSpec.from_file(source)
    spec.compiler.discardvisual = False
    spec.option.timestep = float(sim_dt)
    spec.visual.global_.offwidth = max(int(spec.visual.global_.offwidth), int(config.width))
    spec.visual.global_.offheight = max(int(spec.visual.global_.offheight), int(config.height))
    body = spec.body(config.body_name)
    if body is None:
        raise ValueError(f"Body '{config.body_name}' not found in visual scene '{source}'")
    for name in (config.left_camera, config.right_camera):
        if spec.camera(name) is not None:
            raise ValueError(f"Camera '{name}' already exists in visual scene '{source}'")

    camera_quat = _quat_from_xyaxes(config.xyaxes)
    body_from_camera = _mat_from_quat(camera_quat)
    body_from_optical = body_from_camera @ _OPTICAL_ROTATION
    left_position = np.asarray(config.position, dtype=np.float64)
    right_position = left_position + body_from_optical @ np.array(
        [config.baseline_m, 0.0, 0.0], dtype=np.float64
    )
    for name, position in (
        (config.left_camera, left_position),
        (config.right_camera, right_position),
    ):
        camera = body.add_camera()
        camera.name = name
        camera.pos = position.tolist()
        camera.quat = camera_quat.tolist()
        camera.fovy = float(config.fovy_deg)
        camera.resolution = [int(config.width), int(config.height)]
    return spec.compile()


def _intrinsics(model: mujoco.MjModel, camera_id: int, config: StereoSensorConfig) -> float:
    """Pixels-per-unit-depth derived from the compiled camera projection.

    MuJoCo renders square pixels and derives the horizontal field of view from
    ``fovy`` and the camera aspect ratio, so one focal length covers both axes.
    """
    resolution = np.asarray(model.cam_resolution[camera_id], dtype=np.float64)
    if (
        resolution[0] <= 0.0
        or resolution[1] <= 0.0
        or float(resolution[0]) != float(config.width)
        or float(resolution[1]) != float(config.height)
    ):
        raise ValueError(
            f"camera {camera_id} resolution {resolution.tolist()} does not match "
            f"the requested {config.width}x{config.height}"
        )
    fovy = math.radians(float(model.cam_fovy[camera_id]))
    if not 0.0 < fovy < math.pi:
        raise ValueError(f"camera {camera_id} has an invalid vertical field of view")
    return 0.5 * float(config.height) / math.tan(0.5 * fovy)


def _resolve_rig(
    physics_model: mujoco.MjModel,
    visual_model: mujoco.MjModel,
    config: StereoSensorConfig,
    sim_dt: float,
) -> _RigGeometry:
    """Validate the rigid IMU attachment and derive the fixed calibration."""
    body_id = _named_id(physics_model, mujoco.mjtObj.mjOBJ_BODY, config.body_name, role="body")
    gyro_site_id, gyro_address = _resolve_imu_sensor(
        physics_model,
        name=config.gyro_sensor,
        sensor_type=int(mujoco.mjtSensor.mjSENS_GYRO),
        role="gyro",
    )
    accelerometer_site_id, accelerometer_address = _resolve_imu_sensor(
        physics_model,
        name=config.accelerometer_sensor,
        sensor_type=int(mujoco.mjtSensor.mjSENS_ACCELEROMETER),
        role="accelerometer",
    )
    if gyro_site_id != accelerometer_site_id:
        gyro_name = mujoco.mj_id2name(physics_model, mujoco.mjtObj.mjOBJ_SITE, gyro_site_id)
        accelerometer_name = mujoco.mj_id2name(
            physics_model, mujoco.mjtObj.mjOBJ_SITE, accelerometer_site_id
        )
        raise ValueError(
            f"gyro site '{gyro_name}' and accelerometer site '{accelerometer_name}' must be "
            "the same rigid site"
        )
    if int(physics_model.site_bodyid[gyro_site_id]) != body_id:
        site_name = mujoco.mj_id2name(physics_model, mujoco.mjtObj.mjOBJ_SITE, gyro_site_id)
        raise ValueError(
            f"IMU site '{site_name}' is not rigidly attached to body '{config.body_name}'; "
            "refusing to report a pelvis-mounted IMU as the chest IMU"
        )

    site_name = str(mujoco.mj_id2name(physics_model, mujoco.mjtObj.mjOBJ_SITE, gyro_site_id))
    visual_site_id = _named_id(visual_model, mujoco.mjtObj.mjOBJ_SITE, site_name, role="IMU site")
    if not (
        np.allclose(physics_model.site_pos[gyro_site_id], visual_model.site_pos[visual_site_id])
        and np.allclose(
            physics_model.site_quat[gyro_site_id], visual_model.site_quat[visual_site_id]
        )
    ):
        raise ValueError("visual and physics models disagree on the IMU site transform")

    left_camera_id = _named_id(
        visual_model, mujoco.mjtObj.mjOBJ_CAMERA, config.left_camera, role="left camera"
    )
    right_camera_id = _named_id(
        visual_model, mujoco.mjtObj.mjOBJ_CAMERA, config.right_camera, role="right camera"
    )
    visual_body_id = _named_id(
        visual_model, mujoco.mjtObj.mjOBJ_BODY, config.body_name, role="body"
    )
    pelvis_body_id = _named_id(
        visual_model, mujoco.mjtObj.mjOBJ_BODY, config.pelvis_body, role="pelvis body"
    )
    for camera_id in (left_camera_id, right_camera_id):
        if int(visual_model.cam_bodyid[camera_id]) != visual_body_id:
            raise ValueError("stereo cameras must be attached to the configured body")

    body_from_camera = _se3(
        _mat_from_quat(visual_model.cam_quat[left_camera_id]),
        np.asarray(visual_model.cam_pos[left_camera_id], dtype=np.float64),
    )
    body_from_optical = body_from_camera @ _se3(_OPTICAL_ROTATION, np.zeros(3))
    body_from_imu = _se3(
        _mat_from_quat(physics_model.site_quat[gyro_site_id]),
        np.asarray(physics_model.site_pos[gyro_site_id], dtype=np.float64),
    )
    rig_from_body = _invert_se3(body_from_optical)

    gravity = np.asarray(physics_model.opt.gravity, dtype=np.float64)
    gravity_norm = float(np.linalg.norm(gravity))
    if not (math.isfinite(gravity_norm) and gravity_norm > 1e-9):
        raise ValueError(
            "MuJoCo scene has zero gravity; a tilt-only pelvis up direction cannot be "
            "defined, refusing to invent a default"
        )
    up_in_world = -gravity / gravity_norm

    focal = _intrinsics(visual_model, left_camera_id, config)
    calibration = StereoCalibration(
        width=int(config.width),
        height=int(config.height),
        focal=(focal, focal),
        principal=(0.5 * float(config.width), 0.5 * float(config.height)),
        rig_from_left=np.eye(4, dtype=np.float64),
        rig_from_right=_se3(
            np.eye(3, dtype=np.float64),
            np.array([config.baseline_m, 0.0, 0.0], dtype=np.float64),
        ),
        rig_from_imu=rig_from_body @ body_from_imu,
        imu_frequency_hz=1.0 / float(sim_dt),
    )
    return _RigGeometry(
        left_camera_id=left_camera_id,
        right_camera_id=right_camera_id,
        pelvis_body_id=pelvis_body_id,
        gyro_address=gyro_address,
        accelerometer_address=accelerometer_address,
        up_in_world=up_in_world,
        calibration=calibration,
    )


class MujocoSensorStream(BackendSensorStream):
    """Backend-owned stereo/IMU stream fed by the MuJoCo adapter's stepping loop."""

    def __init__(
        self,
        *,
        physics_model: mujoco.MjModel,
        visual_model: mujoco.MjModel,
        geometry: _RigGeometry,
        config: StereoSensorConfig,
        max_pending_frames: int,
        initial_timestamp_ns: int,
        on_close: Callable[[MujocoSensorStream], None] | None,
    ) -> None:
        self._physics_model = physics_model
        self._imu_data = mujoco.MjData(physics_model)
        self._visual_model = visual_model
        self._data = mujoco.MjData(visual_model)
        self._geometry = geometry
        self._config = config
        self._calibration = geometry.calibration
        self._max_pending_frames = int(max_pending_frames)
        self._on_close = on_close
        self._sequence_id = 0
        self._timestamp_ns = int(initial_timestamp_ns)
        self._substep_count = 0
        self._pending_frames: list[StereoFrame] = []
        self._pending_imu: list[ImuSample] = []
        self._pending_kinematics: list[RigKinematicsSample] = []
        self._pending_references: list[RigReferencePose] = []
        self._closed = False
        try:
            self._renderer = mujoco.Renderer(
                visual_model,
                height=int(config.height),
                width=int(config.width),
            )
        except BaseException:
            self._data = None
            self._visual_model = None
            self._imu_data = None
            self._physics_model = None
            raise

    # -- fixed contract ---------------------------------------------------
    @property
    def calibration(self) -> StereoCalibration:
        return self._calibration

    @property
    def clock(self) -> SensorClock:
        return SensorClock(sequence_id=self._sequence_id, timestamp_ns=self._timestamp_ns)

    def drain_kinematics(self) -> tuple[RigKinematicsSample, ...]:
        samples = tuple(self._pending_kinematics)
        self._pending_kinematics.clear()
        return samples

    def drain(self) -> tuple[StereoFrame, ...]:
        frames = tuple(self._pending_frames)
        self._pending_frames.clear()
        return frames

    def drain_reference_poses(self) -> tuple[RigReferencePose, ...]:
        references = tuple(self._pending_references)
        self._pending_references.clear()
        return references

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pending_frames.clear()
        self._pending_imu.clear()
        self._pending_kinematics.clear()
        self._pending_references.clear()
        renderer, self._renderer = self._renderer, None
        self._data = None
        self._visual_model = None
        self._imu_data = None
        self._physics_model = None
        on_close, self._on_close = self._on_close, None
        if renderer is not None:
            renderer.close()
        if on_close is not None:
            on_close(self)

    # -- adapter callbacks ------------------------------------------------
    def capture_state(self, *, state_row: np.ndarray, timestamp_ns: int) -> None:
        """Capture an existing state without stepping or fabricating IMU samples."""
        if self._closed:
            raise RuntimeError("MuJoCo sensor stream is closed")
        if max(len(self._pending_frames), len(self._pending_kinematics), len(self._pending_references)) >= self._max_pending_frames:
            raise RuntimeError("MuJoCo sensor stream backlog exceeded at initial capture")
        state = np.ascontiguousarray(state_row, dtype=np.float64)
        if state.shape != (mujoco.mj_stateSize(self._visual_model, mujoco.mjtState.mjSTATE_FULLPHYSICS),):
            raise ValueError("initial capture requires a FULLPHYSICS state row")
        mujoco.mj_setState(self._visual_model, self._data, state, int(mujoco.mjtState.mjSTATE_FULLPHYSICS))
        mujoco.mj_forward(self._visual_model, self._data)
        self._timestamp_ns = int(timestamp_ns)
        self._capture(self._timestamp_ns)

    def observe_substep(
        self,
        *,
        state_row: np.ndarray,
        control_row: np.ndarray,
        timestamp_ns: int,
        xfrc_row: np.ndarray | None = None,
    ) -> None:
        """Record exactly one already-integrated physics substep.

        ``state_row`` is the FULLPHYSICS row the pool returned, ``control_row``
        the native control and ``xfrc_row`` the external force (when a push is
        active) that produced it. The physical model evaluates the IMU without
        advancing physics; the complete visual model is synchronized only when
        rendering a frame. Runtime actuator gains therefore remain authoritative.
        """
        if self._closed:
            raise RuntimeError("MuJoCo sensor stream is closed")
        produces_frame = (self._substep_count + 1) % self._config.camera_period_steps == 0
        if produces_frame and (
            len(self._pending_frames) >= self._max_pending_frames
            or len(self._pending_kinematics) >= self._max_pending_frames
            or len(self._pending_references) >= self._max_pending_frames
        ):
            raise RuntimeError(
                "MuJoCo sensor stream backlog exceeded "
                f"{self._max_pending_frames} undrained frames; drain frames, kinematics "
                "and reference poses while stepping instead of dropping them silently"
            )
        # Physical simulation time of this substep, never wall clock.
        self._timestamp_ns = int(timestamp_ns)
        state = np.ascontiguousarray(state_row, dtype=np.float64)
        mujoco.mj_setState(
            self._physics_model, self._imu_data, state,
            int(mujoco.mjtState.mjSTATE_FULLPHYSICS),
        )
        self._imu_data.ctrl[:] = control_row
        if xfrc_row is None:
            self._imu_data.xfrc_applied[:] = 0.0
        else:
            self._imu_data.xfrc_applied[:] = np.asarray(xfrc_row, dtype=np.float64).reshape(
                self._imu_data.xfrc_applied.shape
            )
        mujoco.mj_forward(self._physics_model, self._imu_data)

        sensor = self._imu_data.sensordata
        acceleration_address = self._geometry.accelerometer_address
        gyro_address = self._geometry.gyro_address
        self._pending_imu.append(
            ImuSample(
                timestamp_ns=int(timestamp_ns),
                acceleration=np.array(
                    sensor[acceleration_address : acceleration_address + 3], dtype=np.float64
                ),
                angular_velocity=np.array(
                    sensor[gyro_address : gyro_address + 3], dtype=np.float64
                ),
            )
        )
        self._substep_count += 1
        if self._substep_count % self._config.camera_period_steps == 0:
            mujoco.mj_setState(
                self._visual_model, self._data, state,
                int(mujoco.mjtState.mjSTATE_FULLPHYSICS),
            )
            mujoco.mj_forward(self._visual_model, self._data)
            self._capture(int(timestamp_ns))

    def ensure_capacity(self, *, upcoming_substeps: int) -> None:
        """Refuse a step that would overflow the backlog before physics advances.

        This keeps physics and sampling consistent: a rejected step leaves the
        stream and the simulation untouched instead of dropping a frame.
        """
        if self._closed:
            raise RuntimeError("MuJoCo sensor stream is closed")
        period = self._config.camera_period_steps
        produced = (self._substep_count + int(upcoming_substeps)) // period - (
            self._substep_count // period
        )
        pending = max(
            len(self._pending_frames),
            len(self._pending_kinematics),
            len(self._pending_references),
        )
        if pending + produced > self._max_pending_frames:
            raise RuntimeError(
                "MuJoCo sensor stream backlog would exceed "
                f"{self._max_pending_frames} undrained frames ({produced} more this step); "
                "drain frames, kinematics and reference poses while stepping instead of "
                "dropping them"
            )

    def observe_reset(self, *, timestamp_ns: int, state_row: np.ndarray) -> None:
        """Start a new sequence: drop in-flight IMU samples and bump the sequence id.

        Frames, kinematics and references that were already produced stay available
        and keep the sequence id and physical stamp they were captured with, so a
        consumer never mixes two episodes. The clock adopts the post-reset physics
        time supplied by the adapter.
        """
        if self._closed:
            return
        self._sequence_id += 1
        self._timestamp_ns = int(timestamp_ns)
        self._substep_count = 0
        self._pending_imu.clear()
        if self._config.capture_initial_frame:
            self.capture_state(state_row=state_row, timestamp_ns=timestamp_ns)

    # -- sampling ---------------------------------------------------------
    def _capture(self, timestamp_ns: int) -> None:
        left_rgb = self._render(self._geometry.left_camera_id)
        right_rgb = self._render(self._geometry.right_camera_id)
        world_from_rig = self._world_from_rig()
        imu = tuple(self._pending_imu)
        self._pending_imu.clear()
        self._pending_frames.append(
            StereoFrame(
                sequence_id=self._sequence_id,
                timestamp_ns=timestamp_ns,
                left_rgb=left_rgb,
                right_rgb=right_rgb,
                imu=imu,
            )
        )
        self._pending_references.append(
            RigReferencePose(
                sequence_id=self._sequence_id,
                timestamp_ns=timestamp_ns,
                world_from_rig=world_from_rig,
            )
        )
        self._pending_kinematics.append(
            self._kinematics_sample(timestamp_ns, world_from_rig=world_from_rig)
        )

    def _render(self, camera_id: int) -> np.ndarray:
        self._renderer.update_scene(self._data, camera=camera_id)
        return np.array(self._renderer.render(), dtype=np.uint8, copy=True)

    def _world_from_rig(self) -> np.ndarray:
        rotation = self._data.cam_xmat[self._geometry.left_camera_id].reshape(3, 3)
        position = np.asarray(self._data.cam_xpos[self._geometry.left_camera_id], dtype=np.float64)
        return _se3(rotation @ _OPTICAL_ROTATION, position.copy())

    def _kinematics_sample(
        self, timestamp_ns: int, *, world_from_rig: np.ndarray
    ) -> RigKinematicsSample:
        """Relative pelvis-from-rig FK plus the ideal simulated up axis.

        ``inverse(T_world_pelvis) @ T_world_rig`` is algebraically free of the
        floating-base world pose, so no world position or heading leaves this
        module. ``up_in_pelvis`` carries tilt only.

        The shared reference array stays owned by the reference channel; only the
        relative transform is copied here.
        """
        pelvis_body_id = self._geometry.pelvis_body_id
        world_from_pelvis = _se3(
            self._data.xmat[pelvis_body_id].reshape(3, 3),
            np.asarray(self._data.xpos[pelvis_body_id], dtype=np.float64),
        )
        pelvis_from_rig = _invert_se3(world_from_pelvis) @ world_from_rig
        up_in_pelvis = world_from_pelvis[:3, :3].T @ self._geometry.up_in_world
        return RigKinematicsSample(
            sequence_id=self._sequence_id,
            timestamp_ns=int(timestamp_ns),
            pelvis_from_rig=pelvis_from_rig,
            up_in_pelvis=up_in_pelvis,
        )


def create_mujoco_sensor_stream(
    *,
    visual_model_source: str,
    physics_model: mujoco.MjModel,
    config: StereoSensorConfig,
    sim_dt: float,
    initial_timestamp_ns: int,
    max_pending_frames: int = _DEFAULT_MAX_PENDING_FRAMES,
    on_close: Callable[[MujocoSensorStream], None] | None = None,
) -> MujocoSensorStream:
    """Build the visual model, validate the rig and return the stream."""
    if not math.isfinite(sim_dt) or sim_dt <= 0.0:
        raise ValueError("sim_dt must be positive and finite")
    if isinstance(initial_timestamp_ns, bool) or not isinstance(initial_timestamp_ns, int):
        raise ValueError("initial_timestamp_ns must be an integer physics time in nanoseconds")
    if initial_timestamp_ns < 0:
        raise ValueError("initial_timestamp_ns must be non-negative")
    if isinstance(max_pending_frames, bool) or not isinstance(max_pending_frames, int):
        raise ValueError("max_pending_frames must be a positive integer")
    if max_pending_frames <= 0:
        raise ValueError("max_pending_frames must be a positive integer")

    visual_model = _build_visual_model(visual_model_source, config, float(sim_dt))
    if physics_model.nq != visual_model.nq or physics_model.nv != visual_model.nv:
        raise ValueError(
            "visual model DoF layout does not match the physics model "
            f"(nq {visual_model.nq} vs {physics_model.nq}, "
            f"nv {visual_model.nv} vs {physics_model.nv})"
        )
    physics_state_size = mujoco.mj_stateSize(physics_model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    visual_state_size = mujoco.mj_stateSize(visual_model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    if physics_state_size != visual_state_size:
        raise ValueError(
            "visual model FULLPHYSICS state does not match the physics model "
            f"({visual_state_size} != {physics_state_size})"
        )
    if int(physics_model.nu) != int(visual_model.nu):
        raise ValueError(
            "visual model actuator count does not match the physics model "
            f"({visual_model.nu} != {physics_model.nu})"
        )
    geometry = _resolve_rig(physics_model, visual_model, config, float(sim_dt))
    return MujocoSensorStream(
        physics_model=physics_model,
        visual_model=visual_model,
        geometry=geometry,
        config=config,
        max_pending_frames=max_pending_frames,
        initial_timestamp_ns=int(initial_timestamp_ns),
        on_close=on_close,
    )
