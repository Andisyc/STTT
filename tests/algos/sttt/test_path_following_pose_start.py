"""Real pose conversion/alignment with a deterministic visual estimator, no CUDA."""
from __future__ import annotations

import numpy as np

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
