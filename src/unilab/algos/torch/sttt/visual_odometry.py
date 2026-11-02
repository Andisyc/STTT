"""cuVSLAM stereo / stereo-inertial odometry over the backend stereo stream.

The adapter owns no clock: every frame is submitted with the backend's physics
``timestamp_ns`` unchanged, so the estimate timeline stays on simulation time.
It never reads ground truth; ``RigReferencePose`` is only compared afterwards by
the evaluation, never injected here.

The returned estimate is ``world_from_rig`` in the library's odometry world,
not the MuJoCo world. The rig is the left optical frame (X-right, Y-down,
Z-forward). Quaternions are converted from ``(x, y, z, w)`` to ``wxyz``.

PyCuVSLAM is imported lazily and, when missing, fails with an install hint: no
fallback estimator exists, so a missing wheel must surface as a hard error.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from unilab.base.backend.sensors import ImuSample, StereoCalibration, StereoFrame

__all__ = ["CuvslamOdometry", "VisualOdometryEstimate"]

_IMU_NOISE_KEYS = (
    "gyroscope_noise_density",
    "gyroscope_random_walk",
    "accelerometer_noise_density",
    "accelerometer_random_walk",
)

# Rectification defect tolerated before the rig contradicts `rectified_stereo_camera`.
_ROTATION_TOLERANCE = 1e-3
_LATERAL_TOLERANCE_M = 1e-3

_INSTALL_HINT = (
    "PyCuVSLAM is unavailable. Install a pre-built wheel on Linux only "
    "(Ubuntu 22.04 / py3.10 x86_64, Ubuntu 24.04 / py3.12+ x86_64, or Jetson "
    "aarch64 with JetPack) plus a CUDA toolkit, e.g. "
    "`pip install ./cuvslam_cu12-<version>-<tags>.whl[cuda]`; macOS is unsupported."
)


@dataclass(frozen=True)
class VisualOdometryEstimate:
    """One odometry result on the backend's physics clock.

    ``position`` is ``world_from_rig`` translation in metres and ``quaternion``
    is its ``wxyz`` rotation, both owned by the caller. Both are ``None`` exactly
    when ``valid`` is ``False`` (the library reported no pose for this frame).
    """

    timestamp_ns: int
    sequence_id: int
    valid: bool
    position: np.ndarray | None
    quaternion: np.ndarray | None  # wxyz

    def __post_init__(self) -> None:
        if self.valid:
            if self.position is None or self.quaternion is None:
                raise ValueError("A valid estimate must carry both position and quaternion")
            if self.position.shape != (3,) or self.quaternion.shape != (4,):
                raise ValueError("position must be (3,) and quaternion (4,)")
            if not np.isfinite(self.position).all() or not np.isfinite(self.quaternion).all():
                raise ValueError("A valid estimate must be finite")
        elif self.position is not None or self.quaternion is not None:
            raise ValueError("An invalid estimate must not carry a pose")


class CuvslamOdometry:
    """cuVSLAM visual odometry driven by ``StereoFrame`` packets.

    ``mode="stereo"`` uses ``OdometryMode.Multicamera`` and ignores packet IMU.
    ``mode="inertial"`` uses ``OdometryMode.Inertial`` and requires explicit
    ``imu_noise``: the simulator has no calibrated IMU noise model, so passing
    any density here is a modelling choice, not a measured value.
    """

    def __init__(
        self,
        calibration: StereoCalibration,
        *,
        mode: str = "stereo",
        imu_noise: Mapping[str, float] | None = None,
    ) -> None:
        if mode not in {"stereo", "inertial"}:
            raise ValueError(f"mode must be 'stereo' or 'inertial', got {mode!r}")
        self._mode = mode
        self._inertial = mode == "inertial"
        self._calibration = _validated_calibration(calibration, inertial=self._inertial)
        self._imu_noise = _validated_imu_noise(imu_noise, inertial=self._inertial)
        self._closed = False
        self._sequence_id: int | None = None
        self._last_timestamp_ns: int | None = None
        self._last_imu_timestamp_ns: int | None = None

        self._cuvslam = _import_cuvslam()
        self._rig = _build_rig(self._cuvslam, self._calibration, imu_noise=self._imu_noise)
        self._odometry_config = self._cuvslam.Odometry.Config(
            odometry_mode=(
                self._cuvslam.Odometry.OdometryMode.Inertial
                if self._inertial
                else self._cuvslam.Odometry.OdometryMode.Multicamera
            ),
            async_sba=False,
            rectified_stereo_camera=True,
        )
        # Building a Tracker initialises CUDA state, so it happens on the first
        # frame rather than in the constructor of a possibly unused object.
        self._tracker: Any = None

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def sequence_id(self) -> int | None:
        """Sequence currently owned by the live tracker, ``None`` before the first frame."""
        return self._sequence_id

    def track(self, frame: StereoFrame) -> VisualOdometryEstimate:
        """Feed one stereo frame (plus its IMU interval) and return the rig pose."""
        if self._closed:
            raise RuntimeError("CuvslamOdometry is closed")
        timestamp_ns = int(frame.timestamp_ns)
        if self._sequence_id != frame.sequence_id:
            # cuVSLAM has no reset API: a fresh Tracker restarts the world frame.
            self._restart()
            self._sequence_id = frame.sequence_id
        elif self._last_timestamp_ns is not None and timestamp_ns <= self._last_timestamp_ns:
            raise ValueError(
                "cuVSLAM requires strictly increasing frame timestamps within a sequence; "
                f"got {timestamp_ns} after {self._last_timestamp_ns}"
            )
        if self._inertial:
            self._register_imu(frame.imu, timestamp_ns=timestamp_ns)

        images = [
            _as_uint8_image(frame.left_rgb, name="left", calibration=self._calibration),
            _as_uint8_image(frame.right_rgb, name="right", calibration=self._calibration),
        ]
        estimate = self._tracker.track(timestamp_ns, images)[0]
        self._last_timestamp_ns = timestamp_ns

        world_from_rig = estimate.world_from_rig
        if world_from_rig is None:
            return VisualOdometryEstimate(timestamp_ns, frame.sequence_id, False, None, None)
        pose = world_from_rig.pose
        return VisualOdometryEstimate(
            timestamp_ns,
            frame.sequence_id,
            True,
            np.array(pose.translation, dtype=np.float64),
            _wxyz(np.asarray(pose.rotation, dtype=np.float64)),
        )

    def reset(self) -> None:
        """Drop the tracker so the next frame starts a new sequence from scratch."""
        if self._closed:
            return
        self._release_tracker()

    def close(self) -> None:
        """Release the tracker; safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        self._release_tracker()
        self._rig = None
        self._odometry_config = None

    def _restart(self) -> None:
        self._release_tracker()
        self._tracker = self._start_tracker()

    def _start_tracker(self) -> Any:
        if self._rig is None or self._odometry_config is None:
            raise RuntimeError("CuvslamOdometry is closed")
        # OdometryOnlyOffline keeps bundle adjustment in this thread, so each
        # returned pose is final and replay stays reproducible.
        return self._cuvslam.Tracker(
            self._rig,
            self._cuvslam.Tracker.Mode.OdometryOnlyOffline,
            self._odometry_config,
        )

    def _release_tracker(self) -> None:
        self._tracker = None
        self._sequence_id = None
        self._last_timestamp_ns = None
        self._last_imu_timestamp_ns = None

    def _register_imu(self, samples: tuple[ImuSample, ...], *, timestamp_ns: int) -> None:
        """Register the interval's IMU before the frame; never re-inject a timestamp."""
        for sample in samples:
            imu_timestamp_ns = int(sample.timestamp_ns)
            if imu_timestamp_ns > timestamp_ns:
                raise ValueError(
                    f"IMU sample {imu_timestamp_ns} is newer than its frame {timestamp_ns}"
                )
            if self._last_imu_timestamp_ns is not None:
                if imu_timestamp_ns == self._last_imu_timestamp_ns:
                    # Same instant already accepted; re-registering would re-inject it.
                    continue
                if imu_timestamp_ns < self._last_imu_timestamp_ns:
                    raise ValueError(
                        f"IMU timestamps must be ordered; got {imu_timestamp_ns} after "
                        f"{self._last_imu_timestamp_ns}"
                    )
            measurement = self._cuvslam.ImuMeasurement()
            measurement.timestamp_ns = imu_timestamp_ns
            measurement.linear_accelerations = _vector3(sample.acceleration, name="IMU acceleration")
            measurement.angular_velocities = _vector3(sample.angular_velocity, name="IMU angular velocity")
            self._tracker.register_imu_measurement(0, measurement)
            self._last_imu_timestamp_ns = imu_timestamp_ns


def _import_cuvslam() -> Any:
    try:
        import cuvslam
    except ImportError as error:  # pragma: no cover - depends on the host
        raise ImportError(_INSTALL_HINT) from error
    missing = [
        name
        for name in ("Rig", "Camera", "Pose", "ImuCalibration", "ImuMeasurement", "Odometry", "Tracker")
        if not hasattr(cuvslam, name)
    ]
    if missing:
        raise ImportError(f"Installed PyCuVSLAM lacks {missing}; cuVSLAM >= 17.0.0 is required")
    return cuvslam


def _validated_imu_noise(imu_noise: Mapping[str, float] | None, *, inertial: bool) -> dict[str, float] | None:
    if imu_noise is None:
        if inertial:
            raise ValueError(
                "mode='inertial' requires explicit positive imu_noise entries "
                f"({', '.join(_IMU_NOISE_KEYS)}); the simulator has no calibrated noise model"
            )
        return None
    if not inertial:
        raise ValueError("imu_noise is only used with mode='inertial'")
    if not isinstance(imu_noise, Mapping):
        raise TypeError(f"imu_noise must be a mapping, got {type(imu_noise).__name__}")
    unknown = sorted(set(imu_noise) - set(_IMU_NOISE_KEYS))
    if unknown:
        raise ValueError(f"Unknown imu_noise entries {unknown}; allowed: {list(_IMU_NOISE_KEYS)}")
    missing = [key for key in _IMU_NOISE_KEYS if key not in imu_noise]
    if missing:
        raise ValueError(f"imu_noise is missing {missing}")
    values: dict[str, float] = {}
    for key in _IMU_NOISE_KEYS:
        value = float(imu_noise[key])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"imu_noise.{key} must be positive and finite, got {imu_noise[key]!r}")
        values[key] = value
    return values


def _validated_calibration(calibration: StereoCalibration, *, inertial: bool) -> StereoCalibration:
    if calibration.width <= 0 or calibration.height <= 0:
        raise ValueError("calibration width and height must be positive")
    focal = np.asarray(calibration.focal, dtype=np.float64)
    principal = np.asarray(calibration.principal, dtype=np.float64)
    if focal.shape != (2,) or not np.isfinite(focal).all() or (focal <= 0.0).any():
        raise ValueError("calibration focal must be two positive finite values")
    if principal.shape != (2,) or not np.isfinite(principal).all():
        raise ValueError("calibration principal must be two finite values")
    rig_from_left = _se3(calibration.rig_from_left, name="rig_from_left")
    rig_from_right = _se3(calibration.rig_from_right, name="rig_from_right")
    if inertial:
        if not math.isfinite(calibration.imu_frequency_hz) or calibration.imu_frequency_hz <= 0.0:
            raise ValueError(
                "calibration imu_frequency_hz must be positive and finite for mode='inertial'"
            )
        _se3(calibration.rig_from_imu, name="rig_from_imu")
    _check_rectified(rig_from_left, rig_from_right)
    return calibration


def _se3(value: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) <= 0.0:
        raise ValueError(f"{name} rotation must be a proper rotation matrix")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise ValueError(f"{name} must be a homogeneous SE(3) matrix")
    return matrix


def _check_rectified(rig_from_left: np.ndarray, rig_from_right: np.ndarray) -> None:
    """cuVSLAM is configured for a rectified pair: identity relative rotation, pure +X baseline."""
    relative = np.linalg.inv(rig_from_left) @ rig_from_right
    rotation_error = float(np.abs(relative[:3, :3] - np.eye(3)).max())
    lateral = float(np.abs(relative[1:3, 3]).max())
    if rotation_error > _ROTATION_TOLERANCE or lateral > _LATERAL_TOLERANCE_M or relative[0, 3] <= 0.0:
        raise ValueError(
            "Right camera must be a rectified +X baseline of the left camera; got "
            f"rotation error {rotation_error:.3e}, lateral offset {lateral:.3e} m, "
            f"baseline {relative[0, 3]:.6f} m"
        )


def _build_rig(cuvslam: Any, calibration: StereoCalibration, *, imu_noise: Mapping[str, float] | None) -> Any:
    # `rig_from_*` already maps each sensor into the left optical rig, so these
    # DTO matrices are the extrinsics verbatim -- never a hand-written baseline
    # constant and never rebased a second time.
    cameras = [
        _camera(cuvslam, calibration, rig_from_camera=calibration.rig_from_left),
        _camera(cuvslam, calibration, rig_from_camera=calibration.rig_from_right),
    ]
    rig = cuvslam.Rig(cameras)
    if imu_noise is None:
        rig.imus = []
    else:
        imu = cuvslam.ImuCalibration()
        imu.rig_from_imu = _pose(cuvslam, np.asarray(calibration.rig_from_imu, dtype=np.float64))
        imu.frequency = float(calibration.imu_frequency_hz)
        for name, value in imu_noise.items():
            setattr(imu, name, value)
        rig.imus = [imu]
    return rig


def _camera(cuvslam: Any, calibration: StereoCalibration, *, rig_from_camera: np.ndarray) -> Any:
    camera = cuvslam.Camera()
    camera.size = (int(calibration.width), int(calibration.height))
    camera.principal = (float(calibration.principal[0]), float(calibration.principal[1]))
    camera.focal = (float(calibration.focal[0]), float(calibration.focal[1]))
    # The auto-generated rig produces ideal rectified pinhole images, so the
    # extrinsic is the calibration matrix verbatim (never a baseline constant).
    camera.rig_from_camera = _pose(cuvslam, rig_from_camera)
    return camera


def _pose(cuvslam: Any, transform: np.ndarray) -> Any:
    # The bindings copy a Python sequence into Array<4>/Array<3>; a float64
    # ndarray would only match their float32 fast path, so pass lists.
    return cuvslam.Pose(
        rotation=_quaternion_xyzw(transform[:3, :3]).tolist(),
        translation=transform[:3, 3].astype(np.float64).tolist(),
    )


def _quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Rotation matrix to cuVSLAM's ``(x, y, z, w)`` quaternion order."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(matrix[0, 0] + matrix[1, 1] + matrix[2, 2])
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = (
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
            0.25 * scale,
        )
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        quaternion = (
            0.25 * scale,
            (matrix[0, 1] + matrix[1, 0]) / scale,
            (matrix[0, 2] + matrix[2, 0]) / scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
        )
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        quaternion = (
            (matrix[0, 1] + matrix[1, 0]) / scale,
            0.25 * scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
        )
    else:
        scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        quaternion = (
            (matrix[0, 2] + matrix[2, 0]) / scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
            0.25 * scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        )
    result = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(result))
    if not math.isfinite(norm) or norm < 1e-8:
        raise ValueError("Cannot build a quaternion from a degenerate rotation")
    return result / norm


def _wxyz(quaternion_xyzw: np.ndarray) -> np.ndarray:
    """cuVSLAM ``(x, y, z, w)`` to the repository's ``wxyz`` convention."""
    if quaternion_xyzw.shape != (4,) or not np.isfinite(quaternion_xyzw).all():
        raise ValueError("cuVSLAM returned a malformed rotation quaternion")
    result = np.asarray(quaternion_xyzw[[3, 0, 1, 2]], dtype=np.float64)
    norm = float(np.linalg.norm(result))
    if norm < 1e-8:
        raise ValueError("cuVSLAM returned a degenerate rotation quaternion")
    return result / norm


def _vector3(value: Any, *, name: str) -> list[float]:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be three finite values")
    return vector.tolist()


def _as_uint8_image(image: Any, *, name: str, calibration: StereoCalibration) -> np.ndarray:
    """Continuous ``uint8`` image in one of the layouts cuVSLAM accepts.

    ``(H, W)`` is submitted as mono and ``(H, W, 3)`` as RGB without a channel
    swap or a luminance reduction, so no information is discarded on the way in.
    """
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise ValueError(f"{name} image must be uint8, got {array.dtype}")
    expected = (int(calibration.height), int(calibration.width))
    if array.shape == expected:
        return np.ascontiguousarray(array)
    if array.shape == (*expected, 1):
        return np.ascontiguousarray(array[..., 0])
    if array.shape == (*expected, 3):
        return np.ascontiguousarray(array)
    raise ValueError(
        f"{name} image must be {expected}, {(*expected, 1)}, or {(*expected, 3)}; got {array.shape}"
    )