"""Controller-owned pose runtimes: measured poses on the physical clock.

One runtime serves one controller. ``begin_step`` prepares the source before the
policy query, ``complete_step`` consumes what the normal environment step has
already produced, and ``reset``/``close`` drop episode-local evidence or release
the owned stream and tracker. Nothing here runs a second simulation loop, steps
physics, or invents a pose when a source fails.

Two sources exist:

- ``simulation``: the public base pose is truth on the backend's physical clock
  (``env.get_sim_time_ns``). No camera, renderer, CUDA or cuVSLAM object is ever
  created on this path, and a measurement is cached only when physical time
  actually advanced, so the previous post and the next pre at one instant stay a
  single sample.
- ``cuvslam``: the backend stereo stream is opened on the first ``begin_step``,
  every frame is tracked once and converted with its exactly paired image-time
  kinematics, and evaluation-only reference poses are drained and discarded —
  never handed to the tracker, the converter, or the returned measurements.

Status describes the measurements, not the control loop: no measurement yet is
``waiting``, a last valid capture that is still fresh is ``tracking`` even on a
step without a new camera frame, and an invalid track or a capture older than
``max_pose_age_s`` is ``lost``. The first entry into ``lost`` clears the
timeline, drops the localization anchor, and reports ``invalidated`` exactly
once; a new epoch anchors again exactly once. A source sequence change isolates
all earlier evidence, so a window that predates a loss or a reset can never be
revived.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np

from unilab.algos.torch.sttt.online_data import SpatialAdaptationConfig
from unilab.algos.torch.sttt.pose_conversion import PelvisPoseConverter
from unilab.algos.torch.sttt.pose_data import (
    ControlInterval,
    OnlinePoseRuntime,
    PoseRuntimeUpdate,
    TimedPelvisPose,
)
from unilab.algos.torch.sttt.pose_timeline import PoseTimeline
from unilab.base.backend.sensors import BackendSensorStream, SensorClock
from unilab.envs.common.rotation import np_matrix_from_quat

__all__ = ["create_pose_runtime"]

_NS_PER_SECOND = 1e9
_QUATERNION_NORM_TOLERANCE = 1e-3
_PoseStatus = Literal["waiting", "tracking", "lost"]


def _nanoseconds(seconds: float, name: str) -> int:
    """Convert a positive finite duration in seconds to integer nanoseconds."""
    value = float(seconds)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return int(round(value * _NS_PER_SECOND))


def _env_pose_row(env: Any, getter_name: str, *, dim: int, field: str) -> np.ndarray:
    """Read one environment's base field through the public task boundary."""
    getter = getattr(env, getter_name, None)
    if not callable(getter):
        raise RuntimeError(
            f"simulation pose runtime requires env.{getter_name}() for the {field}; "
            "the environment exposes no public base-state accessor."
        )
    values = np.asarray(getter(), dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != dim:
        raise ValueError(f"env.{getter_name}() must have shape (num_envs, {dim}), got {values.shape}")
    if values.shape[0] != 1:
        raise ValueError("a scalar pose runtime requires exactly one environment")
    return np.ascontiguousarray(values[0])


def _env_sim_time_ns(env: Any) -> int:
    """Read the backend physical clock as an integer nanosecond."""
    getter = getattr(env, "get_sim_time_ns", None)
    if not callable(getter):
        raise RuntimeError(
            "simulation pose runtime requires env.get_sim_time_ns() for the physical clock."
        )
    value = getter()
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"env.get_sim_time_ns() must return an integer nanosecond, got {value!r}")
    return int(value)


class _TimelineEvidence:
    """One measurement timeline plus the freshness bookkeeping both sources share."""

    def __init__(self, spatial: SpatialAdaptationConfig) -> None:
        self._timeline = PoseTimeline(
            max_gap_ns=_nanoseconds(spatial.max_pose_gap_s, "max_pose_gap_s"),
            max_age_ns=_nanoseconds(spatial.max_alignment_age_s, "max_alignment_age_s"),
            max_samples=int(spatial.max_pose_samples),
            max_pending=int(spatial.max_pending_intervals),
        )
        self._max_pose_age_ns = _nanoseconds(spatial.max_pose_age_s, "max_pose_age_s")
        self._sequence_id = 0
        self._last_valid_ns: int | None = None
        self._lost = False
        self._closed = False

    # ----------------------------------------------------------------- state

    @property
    def _status(self) -> _PoseStatus:
        if self._lost:
            return "lost"
        return "waiting" if self._last_valid_ns is None else "tracking"

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")

    def _clear_evidence(self) -> None:
        """Drop every cached pose and queued interval; nothing is kept silently."""
        self._timeline.reset()
        self._last_valid_ns = None
        self._lost = False

    def _mark_lost(self) -> None:
        """Drop all evidence of the epoch that just stopped being current."""
        self._timeline.reset()
        self._last_valid_ns = None
        self._lost = True

    def _record(self, pose: TimedPelvisPose) -> None:
        """Cache one measurement and refresh the last valid capture time."""
        self._timeline.add_pose(pose)
        if pose.valid:
            self._last_valid_ns = int(pose.timestamp_ns)
            self._lost = False

    def _judge_age(self, now_ns: int) -> bool:
        """Enter lost when the last valid capture is older than ``max_pose_age_s``.

        Returns True only for the transition that invalidated the evidence. A
        source without a new capture but with a fresh last pose stays tracking.
        """
        last_valid_ns = self._last_valid_ns
        if last_valid_ns is None or self._lost:
            return False
        if now_ns - last_valid_ns <= self._max_pose_age_ns:
            return False
        self._mark_lost()
        self._open_epoch()
        return True

    def _update(self, *, now_ns: int, tracking_epoch: int, invalidated: bool) -> PoseRuntimeUpdate:
        aligned = self._timeline.drain_ready(now_ns=now_ns)
        return PoseRuntimeUpdate(
            aligned=aligned,
            status=self._status,
            sequence_id=self._sequence_id,
            tracking_epoch=tracking_epoch,
            # Cumulative since construction, exactly like ``PoseTimeline``: no
            # discard is hidden and no counter silently resets on a new epoch.
            dropped_intervals=self._timeline.dropped_intervals,
            invalidated=invalidated,
        )

    def _require_control_sequence(self, control: ControlInterval) -> None:
        if control.sequence_id != self._sequence_id:
            raise ValueError(
                f"control interval belongs to source sequence {control.sequence_id}, "
                f"current source sequence is {self._sequence_id}"
            )

    # -------------------------------------------------------------- subclass

    def _open_epoch(self) -> None:
        """Start a fresh localization epoch; the next valid pose anchors once."""
        raise NotImplementedError


class _SimulationPoseRuntime(_TimelineEvidence):
    """Ground-truth base pose on the backend physical clock; no camera exists."""

    _FRAME_ID = "world"

    def __init__(self, env: Any, spatial: SpatialAdaptationConfig) -> None:
        super().__init__(spatial)
        self._env = env
        self._tracking_epoch = 0
        self._last_measurement_ns: int | None = None

    # ----------------------------------------------------------------- input

    @property
    def clock(self) -> SensorClock:
        """Physical time of the single environment on this runtime's sequence."""
        self._require_open()
        return SensorClock(
            sequence_id=self._sequence_id,
            timestamp_ns=_env_sim_time_ns(self._env),
        )

    def begin_step(self) -> SensorClock:
        self._require_open()
        clock = self.clock
        self._measure(clock.timestamp_ns)
        return clock

    def complete_step(self, control: ControlInterval) -> PoseRuntimeUpdate:
        self._require_open()
        self._require_control_sequence(control)
        self._timeline.enqueue(control)
        # Truth is measured at the post-step instant, so the newest valid pose is
        # always the current clock; only a source that can miss frames goes stale.
        now_ns = _env_sim_time_ns(self._env)
        self._measure(now_ns)
        return self._update(
            now_ns=now_ns,
            tracking_epoch=self._tracking_epoch,
            invalidated=False,
        )

    # --------------------------------------------------------------- control

    def reset(self) -> None:
        """Drop the episode before ``env.reset``; the next ``begin_step`` resyncs.

        The old environment state is never read here: the clock, the poses and
        the post-reset epoch all start from the step that follows.
        """
        self._require_open()
        self._sequence_id += 1
        self._clear_evidence()
        self._open_epoch()
        self._last_measurement_ns = None

    def close(self) -> None:
        """Nothing is owned beyond the environment reference; idempotent."""
        self._closed = True

    # --------------------------------------------------------------- helpers

    def _open_epoch(self) -> None:
        self._tracking_epoch += 1

    def _measure(self, timestamp_ns: int) -> None:
        """Cache the truth pose once per physical instant, never twice."""
        last_measurement_ns = self._last_measurement_ns
        if last_measurement_ns == timestamp_ns:
            # The previous post and this pre are the same physical instant.
            return
        if last_measurement_ns is not None and timestamp_ns < last_measurement_ns:
            raise ValueError(
                "simulation physical time moved backwards: "
                f"{timestamp_ns} ns after {last_measurement_ns} ns"
            )
        pose = self._truth_pose(timestamp_ns)
        self._last_measurement_ns = timestamp_ns
        self._record(pose)

    def _truth_pose(self, timestamp_ns: int) -> TimedPelvisPose:
        position = _env_pose_row(self._env, "get_base_pos", dim=3, field="base position")
        quaternion = _env_pose_row(self._env, "get_base_quat", dim=4, field="base quaternion")
        norm = float(np.linalg.norm(quaternion))
        if not math.isfinite(norm) or abs(norm - 1.0) > _QUATERNION_NORM_TOLERANCE:
            raise ValueError(
                f"env.get_base_quat() must return a unit wxyz quaternion, got norm {norm}"
            )
        if not np.isfinite(position).all():
            raise ValueError("env.get_base_pos() must return finite world coordinates")
        navigation_from_pelvis = np.eye(4, dtype=np.float64)
        navigation_from_pelvis[:3, :3] = np_matrix_from_quat(quaternion / norm)
        navigation_from_pelvis[:3, 3] = position
        return TimedPelvisPose(
            sequence_id=self._sequence_id,
            tracking_epoch=self._tracking_epoch,
            timestamp_ns=timestamp_ns,
            navigation_from_pelvis=navigation_from_pelvis,
            valid=True,
            source_timestamps_ns=(timestamp_ns, timestamp_ns),
            frame_id=self._FRAME_ID,
        )


class _CuvslamPoseRuntime(_TimelineEvidence):
    """cuVSLAM stereo odometry over the backend stream; truth is never injected."""

    def __init__(self, env: Any, spatial: SpatialAdaptationConfig) -> None:
        super().__init__(spatial)
        self._env = env
        self._sensor_config = spatial.sensor
        self._visual_mode = spatial.visual_mode
        self._imu_noise = None if spatial.imu_noise is None else dict(spatial.imu_noise)
        self._converter = PelvisPoseConverter()
        self._stream: BackendSensorStream | None = None
        self._odometry: Any = None

    # ----------------------------------------------------------------- input

    @property
    def clock(self) -> SensorClock:
        """Physical time straight from the stream; never a second clock source.

        The sequence id is the stream's own, so a physical reset inside the step
        shows up here instead of being hidden by a cached copy.
        """
        self._require_open()
        stream = self._stream
        if stream is None:
            raise RuntimeError("CuvslamPoseRuntime requires begin_step() to open the stream first")
        return stream.clock

    def begin_step(self) -> SensorClock:
        self._require_open()
        stream = self._open()
        clock = stream.clock
        self._sync_sequence(clock.sequence_id)
        return clock

    def complete_step(self, control: ControlInterval) -> PoseRuntimeUpdate:
        self._require_open()
        stream = self._stream
        if stream is None:
            raise RuntimeError("begin_step() must open the cuVSLAM stream before complete_step()")
        odometry = self._odometry
        if odometry is None:
            raise RuntimeError("the cuVSLAM tracker is not open")
        # Verify the interval belongs to the sequence the caller started in.
        self._require_control_sequence(control)
        clock = stream.clock
        if int(clock.sequence_id) != self._sequence_id:
            raise RuntimeError(
                f"the stream moved to sequence {int(clock.sequence_id)} while sequence "
                f"{self._sequence_id} owned control interval {control.transition_id}: "
                "a physical reset crossed the interval"
            )
        self._timeline.enqueue(control)

        frames = stream.drain()
        kinematics_rows = stream.drain_kinematics()
        # Evaluation-only truth: drained so the stream never backlogs, never used.
        stream.drain_reference_poses()
        if len(frames) != len(kinematics_rows):
            raise RuntimeError(
                "stereo frames and image-time kinematics are not paired: "
                f"{len(frames)} frames vs {len(kinematics_rows)} kinematics"
            )
        kinematics_by_key = {
            (int(item.sequence_id), int(item.timestamp_ns)): item for item in kinematics_rows
        }
        invalidated = False
        for frame in frames:
            key = (int(frame.sequence_id), int(frame.timestamp_ns))
            if key[0] != self._sequence_id:
                # A packet from another episode cannot be paired with this
                # interval, and mixing the two would resurrect pre-reset
                # evidence. Fail loudly instead of silently dropping frames.
                raise RuntimeError(
                    f"stereo frame of sequence {key[0]} at {key[1]} ns arrived while "
                    f"sequence {self._sequence_id} owns the control interval"
                )
            kinematics = kinematics_by_key.pop(key, None)
            if kinematics is None:
                raise RuntimeError(
                    f"no image-time kinematics for frame sequence {key[0]} at {key[1]} ns"
                )
            pose = self._converter.convert(odometry.track(frame), kinematics)
            if not pose.valid:
                if not self._lost:
                    # One lost track invalidates once; the timeline, the queued
                    # intervals and the localization anchor all stop here. The
                    # converter already opened the new epoch, and the invalid
                    # sample itself is never cached, so the next valid pose
                    # anchors that epoch exactly once.
                    self._mark_lost()
                    invalidated = True
                continue
            self._record(pose)
        if kinematics_by_key:
            raise RuntimeError(
                "image-time kinematics without a matching frame: "
                f"{sorted(kinematics_by_key)[:4]}"
            )

        now_ns = int(stream.clock.timestamp_ns)
        if self._judge_age(now_ns):
            invalidated = True
        return self._update(
            now_ns=now_ns,
            tracking_epoch=self._converter.tracking_epoch,
            invalidated=invalidated,
        )

    # --------------------------------------------------------------- control

    def reset(self) -> None:
        """Drop the episode before ``env.reset``; the next ``begin_step`` resyncs.

        Nothing is re-opened or captured here: packets left from the old episode
        are discarded, the timeline is cleared, and the new stream sequence
        rebuilds the localization anchor lazily from its first valid pose.
        """
        self._require_open()
        stream = self._stream
        if stream is not None:
            stream.drain()
            stream.drain_kinematics()
            stream.drain_reference_poses()
        self._clear_evidence()
        self._open_epoch()

    def close(self) -> None:
        """Release the tracker then the stream; idempotent and leak-free on error."""
        if self._closed:
            return
        self._closed = True
        odometry, self._odometry = self._odometry, None
        stream, self._stream = self._stream, None
        failure: Exception | None = None
        if odometry is not None:
            try:
                odometry.close()
            except Exception as error:  # still release the remaining resource
                failure = error
        if stream is not None:
            try:
                stream.close()
            except Exception as error:
                failure = failure or error
        if failure is not None:
            raise failure

    # --------------------------------------------------------------- helpers

    def _open_epoch(self) -> None:
        """One epoch owns one fixed navigation anchor; the next pose re-anchors."""
        self._converter.reset()

    def _sync_sequence(self, sequence_id: int) -> None:
        """Adopt a new stream sequence and isolate every earlier measurement."""
        value = int(sequence_id)
        if value == self._sequence_id:
            return
        if value < self._sequence_id:
            raise RuntimeError(
                f"stereo packets of sequence {value} arrived after sequence "
                f"{self._sequence_id}: a physical reset crossed the control interval"
            )
        self._sequence_id = value
        self._clear_evidence()
        if self._converter.anchored:
            # The old navigation anchor belongs to the previous episode, so the
            # new sequence must build its own origin. A converter that already
            # dropped its anchor is left alone, so one sequence change opens one
            # epoch and the first valid pose anchors it exactly once.
            self._open_epoch()

    def _open(self) -> BackendSensorStream:
        """Open the stream and tracker on the first step; failures propagate."""
        stream = self._stream
        if stream is not None:
            return stream
        # Imported here so the simulation source never touches cuVSLAM at all.
        from unilab.algos.torch.sttt.visual_odometry import CuvslamOdometry

        stream = self._env.open_sensor_stream(self._sensor_config)
        try:
            odometry = CuvslamOdometry(
                stream.calibration,
                mode=self._visual_mode,
                imu_noise=self._imu_noise,
            )
        except BaseException:
            # Free the half-opened stream, but keep the original failure visible.
            try:
                stream.close()
            finally:
                raise
        self._stream = stream
        self._odometry = odometry
        return stream


def create_pose_runtime(env: Any, spatial: SpatialAdaptationConfig) -> OnlinePoseRuntime:
    """Build the single controller-owned pose source selected by the config."""
    if spatial.odometry_source == "simulation":
        return _SimulationPoseRuntime(env, spatial)
    if spatial.odometry_source == "cuvslam":
        return _CuvslamPoseRuntime(env, spatial)
    raise ValueError(f"unsupported odometry_source {spatial.odometry_source!r}")