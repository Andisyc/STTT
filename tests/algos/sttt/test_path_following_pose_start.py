"""Real pose conversion/alignment with a deterministic visual estimator, no CUDA."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from unilab.algos.torch.sttt.odometry_runtime import create_pose_runtime
from unilab.algos.torch.sttt.pose_data import ControlInterval, PoseRuntimeConfig
from unilab.algos.torch.sttt.visual_odometry import VisualOdometryEstimate
from unilab.base.backend.sensors import RigKinematicsSample, SensorClock, StereoFrame, StereoSensorConfig


def test_initial_and_autoreset_frame_reach_alignment_without_truth_or_duplicate_tracking(monkeypatch):
    class Stream:
        def __init__(self):
            self.clock = SensorClock(0, 0)
            self.calibration = object()
            self.frames, self.kinematics = [], []
            self.emit()
            self.closed = False
        def emit(self):
            sequence, timestamp = self.clock.sequence_id, self.clock.timestamp_ns
            image = np.zeros((2, 2, 3), dtype=np.uint8)
            self.frames.append(StereoFrame(sequence, timestamp, image, image, ()))
            self.kinematics.append(RigKinematicsSample(sequence, timestamp, np.eye(4), np.array([0., 0., 1.])))
        def drain(self):
            result, self.frames = tuple(self.frames), []
            return result
        def drain_kinematics(self):
            result, self.kinematics = tuple(self.kinematics), []
            return result
        def drain_reference_poses(self): return ()
        def close(self): self.closed = True

    class Tracker:
        def __init__(self, *_, **__): self.seen = []
        def track(self, frame):
            self.seen.append((frame.sequence_id, frame.timestamp_ns))
            return VisualOdometryEstimate(frame.timestamp_ns, frame.sequence_id, True, np.array([frame.timestamp_ns / 1e9, 0., 0.]), np.array([1., 0., 0., 0.]))
        def close(self): pass

    stream = Stream()
    class Env:
        def open_sensor_stream(self, config): return stream
        def get_base_pos(self): raise AssertionError('cuVSLAM cannot read ground truth')
        def get_base_quat(self): raise AssertionError('cuVSLAM cannot read ground truth')

    monkeypatch.setattr('unilab.algos.torch.sttt.visual_odometry.CuvslamOdometry', Tracker)
    source = create_pose_runtime(Env(), PoseRuntimeConfig(odometry_source='cuvslam', sensor=StereoSensorConfig(capture_initial_frame=True)))
    try:
        before = source.begin_step()
        stream.clock = SensorClock(0, 40_000_000)
        stream.emit()
        update = source.complete_step(ControlInterval(0, 0, before.timestamp_ns, stream.clock.timestamp_ns, np.zeros(29), np.array([0.4, 0., 0.])))
        first, = update.aligned
        assert first.end_pose.navigation_from_pelvis is not None
        np.testing.assert_allclose(first.end_pose.navigation_from_pelvis[:3, 3], [0.04, 0., 0.], atol=1e-12)
        stream.clock = SensorClock(1, 0)
        stream.emit()  # The backend already reset before the controller reset callback.
        source.reset()
        before = source.begin_step()
        stream.clock = SensorClock(1, 40_000_000)
        stream.emit()
        update = source.complete_step(ControlInterval(1, 0, before.timestamp_ns, stream.clock.timestamp_ns, np.zeros(29), np.array([0.4, 0., 0.])))
        reset, = update.aligned
        assert reset.start_pose.timestamp_ns == 0 and reset.start_pose.source_timestamps_ns == (0, 0)
        assert reset.start_pose.sequence_id == 1 and reset.start_pose.tracking_epoch > first.start_pose.tracking_epoch
        assert reset.end_pose.navigation_from_pelvis is not None
        np.testing.assert_allclose(reset.end_pose.navigation_from_pelvis[:3, 3], [0.04, 0., 0.], atol=1e-12)
    finally:
        source.close()
    assert stream.closed


@pytest.mark.parametrize("slam_enabled,missing_slam", [(False, False), (True, False), (True, True)])
def test_sdk_navigation_pose_does_not_substitute_odometry_for_missing_slam(monkeypatch, slam_enabled, missing_slam):
    from unilab.algos.torch.sttt.visual_odometry import CuvslamOdometry
    from unilab.base.backend.sensors import StereoCalibration

    odometry_pose = SimpleNamespace(translation=[9., 8., 7.], rotation=[0., 0., 0., 1.])
    slam_pose = SimpleNamespace(translation=[1., 2., 3.], rotation=[0., 0., 0., 1.])

    class Tracker:
        OdometryConfig = SlamConfig = SimpleNamespace

        def __init__(self, rig, *, odom_config, slam_config):
            self.slam_config = slam_config

        def track(self, timestamp_ns, images):
            return SimpleNamespace(world_from_rig=SimpleNamespace(pose=odometry_pose)), None if missing_slam else slam_pose

        def get_loop_closure_poses(self):
            return ()

    sdk = SimpleNamespace(
        Tracker=Tracker, Rig=lambda cameras: SimpleNamespace(cameras=cameras),
        Camera=SimpleNamespace, Pose=SimpleNamespace, ImuCalibration=SimpleNamespace,
        ImuMeasurement=SimpleNamespace,
        pycuvslam=SimpleNamespace(Odometry=SimpleNamespace(OdometryMode=SimpleNamespace(Multicamera=0, Inertial=1))),
    )
    monkeypatch.setitem(sys.modules, "cuvslam", sdk)
    right = np.eye(4)
    right[0, 3] = .06
    calibration = StereoCalibration(8, 8, (10., 10.), (4., 4.), np.eye(4), right, np.eye(4), 500.)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    tracker = CuvslamOdometry(calibration, slam_enabled=slam_enabled)
    try:
        result = tracker.track(StereoFrame(0, 123_456, image, image, ()))
        assert result.timestamp_ns == 123_456
        if slam_enabled and missing_slam:
            assert not result.valid and result.position is None and result.quaternion is None
        else:
            np.testing.assert_array_equal(result.position, [1., 2., 3.] if slam_enabled else [9., 8., 7.])
            np.testing.assert_array_equal(result.quaternion, [1., 0., 0., 0.])
        if slam_enabled:
            assert tracker._tracker.slam_config.sync_mode and not tracker._tracker.slam_config.gt_align_mode
            tracker._tracker.get_loop_closure_poses = lambda: (SimpleNamespace(timestamp_ns=123_456),)
            with pytest.raises(RuntimeError, match="loop closure"):
                tracker.track(StereoFrame(0, 123_457, image, image, ()))
    finally:
        tracker.close()
