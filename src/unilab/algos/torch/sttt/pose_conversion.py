"""Measured SLAM rig odometry to a metric Z-up pelvis pose.

The tracker reports ``world_from_rig`` (``T_OR``) in its own odometry world
``O``; the rig is the left optical frame. FK gives ``pelvis_from_rig``
(``T_PR``) at exactly the capture time, so ``T_OP = T_OR @ inv(T_PR)`` is the
pelvis pose in ``O`` with no world position and no ground-truth reference
involved at any point.

One localization epoch owns exactly one fixed navigation-from-odometry
transform ``T_NO``. It is built on the first valid pose: the FK up direction
supplies tilt, the pelvis forward axis supplies heading, and the pelvis
position becomes the origin with heading zero. Later poses of the same epoch
are ``T_NP = T_NO @ T_OP`` and never re-anchor, so real tilt and drift stay
visible in the output.

A tracking loss, a sequence change, and an explicit reset each open a new
epoch: the anchor is dropped, ``tracking_epoch`` advances once, and the next
valid pose anchors again. Poses are metric SE(3) in a Z-up frame.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from unilab.algos.torch.sttt.pose_data import TimedPelvisPose
from unilab.algos.torch.sttt.visual_odometry import VisualOdometryEstimate
from unilab.base.backend.sensors import RigKinematicsSample
from unilab.envs.common.rotation import np_matrix_from_quat

__all__ = ["PelvisPoseConverter"]

_EPS = 1e-9
# Below this the forward axis leaves no usable heading left after removing up.
_MIN_PROJECTION = 1e-6


class PelvisPoseConverter:
    """Convert odometry/first-kinematics pairs into navigation-frame pelvis poses."""

    def __init__(self) -> None:
        self._sequence_id: int | None = None
        self._last_timestamp_ns: int | None = None
        self._tracking_epoch = 0
        self._navigation_from_odom: np.ndarray | None = None

    @property
    def sequence_id(self) -> int | None:
        """Sequence the converter currently serves, ``None`` before the first sample."""
        return self._sequence_id

    @property
    def tracking_epoch(self) -> int:
        """Epoch index of the pose that :meth:`convert` would return next."""
        return self._tracking_epoch

    @property
    def anchored(self) -> bool:
        """Whether the current epoch already owns a fixed ``T_NO``."""
        return self._navigation_from_odom is not None

    def reset(self) -> None:
        """Isolate a new epoch; the next valid sample anchors it once."""
        self._sequence_id = None
        self._last_timestamp_ns = None
        self._tracking_epoch += 1
        self._navigation_from_odom = None

    def convert(
        self,
        estimate: VisualOdometryEstimate,
        kinematics: RigKinematicsSample,
    ) -> TimedPelvisPose:
        """Return the metric Z-up pelvis pose for one matched measurement pair."""
        sequence_id = _int_field(estimate.sequence_id, name="estimate.sequence_id")
        timestamp_ns = _int_field(estimate.timestamp_ns, name="estimate.timestamp_ns")
        _require_paired(kinematics, sequence_id=sequence_id, timestamp_ns=timestamp_ns)
        self._start_sequence(sequence_id)
        self._require_increasing(timestamp_ns)

        if not estimate.valid:
            # The epoch drops its anchor on the first loss only: a run of invalid
            # samples keeps the epoch it already opened.
            if self._navigation_from_odom is not None:
                self._open_epoch()
            self._last_timestamp_ns = timestamp_ns
            return TimedPelvisPose(
                sequence_id=sequence_id,
                tracking_epoch=self._tracking_epoch,
                timestamp_ns=timestamp_ns,
                navigation_from_pelvis=None,
                valid=False,
                source_timestamps_ns=(timestamp_ns, timestamp_ns),
            )

        odom_from_pelvis = _odom_from_pelvis(estimate, kinematics)
        if self._navigation_from_odom is None:
            self._navigation_from_odom = _navigation_from_odom(
                odom_from_pelvis, up_in_pelvis=_up_in_pelvis(kinematics)
            )
        navigation_from_pelvis = _se3(
            self._navigation_from_odom @ odom_from_pelvis,
            name="navigation_from_pelvis",
        )
        self._last_timestamp_ns = timestamp_ns
        return TimedPelvisPose(
            sequence_id=sequence_id,
            tracking_epoch=self._tracking_epoch,
            timestamp_ns=timestamp_ns,
            navigation_from_pelvis=navigation_from_pelvis,
            valid=True,
            source_timestamps_ns=(timestamp_ns, timestamp_ns),
        )

    def _start_sequence(self, sequence_id: int) -> None:
        if self._sequence_id == sequence_id:
            return
        if self._sequence_id is not None:
            self._open_epoch()
        self._sequence_id = sequence_id
        self._last_timestamp_ns = None

    def _open_epoch(self) -> None:
        self._tracking_epoch += 1
        self._navigation_from_odom = None

    def _require_increasing(self, timestamp_ns: int) -> None:
        if self._last_timestamp_ns is not None and timestamp_ns <= self._last_timestamp_ns:
            raise ValueError(
                "converted measurements must have strictly increasing timestamps "
                f"within a sequence; got {timestamp_ns} after {self._last_timestamp_ns}"
            )


def _int_field(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _require_paired(
    kinematics: RigKinematicsSample,
    *,
    sequence_id: int,
    timestamp_ns: int,
) -> None:
    kinematics_sequence = _int_field(kinematics.sequence_id, name="kinematics.sequence_id")
    kinematics_timestamp = _int_field(kinematics.timestamp_ns, name="kinematics.timestamp_ns")
    if kinematics_sequence != sequence_id or kinematics_timestamp != timestamp_ns:
        raise ValueError(
            "kinematics must describe the capture of this estimate exactly; got "
            f"sequence {kinematics_sequence} at {kinematics_timestamp} for estimate "
            f"sequence {sequence_id} at {timestamp_ns}"
        )


def _odom_from_pelvis(
    estimate: VisualOdometryEstimate,
    kinematics: RigKinematicsSample,
) -> np.ndarray:
    """``T_OP = T_OR @ inv(T_PR)``: pelvis in the odometry world."""
    position = np.asarray(estimate.position, dtype=np.float64)
    quaternion = np.asarray(estimate.quaternion, dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("estimate.position must be a finite (3,) vector")
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("estimate.quaternion must be a finite (4,) wxyz quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm < _EPS:
        raise ValueError("estimate.quaternion must be nonzero")

    odom_from_rig = np.eye(4, dtype=np.float64)
    odom_from_rig[:3, :3] = np_matrix_from_quat(quaternion / norm)
    odom_from_rig[:3, 3] = position
    odom_from_rig = _se3(odom_from_rig, name="world_from_rig")

    pelvis_from_rig = _se3(kinematics.pelvis_from_rig, name="kinematics.pelvis_from_rig")
    return odom_from_rig @ np.linalg.inv(pelvis_from_rig)


def _up_in_pelvis(kinematics: RigKinematicsSample) -> np.ndarray:
    up = np.asarray(kinematics.up_in_pelvis, dtype=np.float64)
    if up.shape != (3,) or not np.isfinite(up).all():
        raise ValueError("kinematics.up_in_pelvis must be a finite (3,) direction")
    norm = float(np.linalg.norm(up))
    if norm < _EPS:
        raise ValueError("kinematics.up_in_pelvis must be nonzero")
    return up / norm


def _navigation_from_odom(odom_from_pelvis: np.ndarray, *, up_in_pelvis: np.ndarray) -> np.ndarray:
    """Anchor ``T_NO`` from tilt and the pelvis forward axis of the first pose."""
    rotation = odom_from_pelvis[:3, :3]
    up_in_odom = rotation @ up_in_pelvis
    up_in_odom = up_in_odom / np.linalg.norm(up_in_odom)

    forward = rotation[:, 0]
    forward = forward - float(forward @ up_in_odom) * up_in_odom
    projection = float(np.linalg.norm(forward))
    if projection < _MIN_PROJECTION:
        raise ValueError(
            "cannot initialize Z-up heading: the pelvis forward axis is parallel to up, "
            "so no heading direction remains"
        )
    x_axis = forward / projection
    y_axis = np.cross(up_in_odom, x_axis)

    navigation_from_odom = np.eye(4, dtype=np.float64)
    rotation_no = np.stack((x_axis, y_axis, up_in_odom), axis=1)
    navigation_from_odom[:3, :3] = rotation_no.T
    navigation_from_odom[:3, 3] = -navigation_from_odom[:3, :3] @ odom_from_pelvis[:3, 3]
    return _se3(navigation_from_odom, name="navigation_from_odom")


def _se3(value: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    rotation = matrix[:3, :3]
    orthonormal = np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
    if not orthonormal or np.linalg.det(rotation) <= 0.0:
        raise ValueError(f"{name} rotation must be a proper rotation matrix")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise ValueError(f"{name} must be a homogeneous SE(3) matrix")
    return matrix