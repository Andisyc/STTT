"""Causal intervention timing, measured labels and persisted sample boundaries."""
from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from unilab.algos.torch.distill.fada.runtime_policy import FADAFutureQuery, OfficialFADAPlaybackProfile
from unilab.algos.torch.sttt.path_following import WindowCollector
from unilab.algos.torch.sttt.path_following.dataset import (
    PathFollowingDatasetWriter, iter_path_following_samples, read_path_following_metadata,
)
from unilab.algos.torch.sttt.pose_data import AlignedPoseInterval, ControlInterval, PoseRuntimeUpdate, TimedPelvisPose
from unilab.base.backend.sensors import SensorClock

DT_NS = 20_000_000


def make_profile():
    return OfficialFADAPlaybackProfile(
        source_identity={"run": "synthetic-cpu-check"}, joint_names=tuple(f"j{i}" for i in range(29)),
        default_joint_pos=(0.2,) * 29, action_scale=(0.35,) * 29, stiffness=(50.,) * 29,
        damping=(2.,) * 29, clip_actions=0.35, ctrl_dt=0.02,
        gait_frequency=1.5, gait_clock_mode="source_clock", phase_init=(0., math.pi),
    )


def make_query(step):
    phase = step * 0.1
    return FADAFutureQuery(
        observation_history=torch.full((1, 2, 64), float(step)), action_history=torch.zeros(1, 2, 29),
        command=torch.tensor([[0.4, 0., 0., math.sin(phase), -math.sin(phase), math.cos(phase), -math.cos(phase)]]),
        history_valid_mask=torch.ones(1, 2, dtype=torch.bool), planner_future=torch.full((1, 3, 64), float(step)),
        preprocessing={"term_scale": {"dof_pos": 2.5, "dof_vel": 0.15}}, profile=make_profile(),
    )


def measured_pose(step, *, epoch=1, frame="world"):
    yaw = math.pi / 2 + step * 0.1
    matrix = np.eye(4)
    matrix[:2, :2] = ((math.cos(yaw), -math.sin(yaw)), (math.sin(yaw), math.cos(yaw)))
    matrix[:3, 3] = (10 - step * 0.2, 20 + step * 0.5, 0.8)
    timestamp = step * DT_NS
    return TimedPelvisPose(
        1, epoch, timestamp, matrix, True,
        (max(0, timestamp - DT_NS // 2), timestamp + DT_NS // 2), frame,
    )


def aligned_step(step, *, epoch=1, frame="world"):
    control = ControlInterval(
        1, step, step * DT_NS, (step + 1) * DT_NS,
        np.arange(29, dtype=float) / 100 + step, np.array([np.float32(0.4), 0., 0.]),
    )
    return AlignedPoseInterval(control, measured_pose(step, epoch=epoch, frame=frame), measured_pose(step + 1, epoch=epoch, frame=frame))


def execute_step(collector, step, *, arrivals=None, epoch=1, frame="world", terminal=False):
    collector.begin_step(SensorClock(1, step * DT_NS))
    query = make_query(step)
    future = collector.transform(query)
    current = aligned_step(step, epoch=epoch, frame=frame)
    if arrivals is None:
        arrivals = (current,)
    update = PoseRuntimeUpdate(tuple(arrivals), "tracking" if arrivals else "waiting", 1, epoch)
    samples = collector.complete_step(current.control, update, terminal=terminal)
    return future - query.planner_future, samples


def complete_samples():
    collector = WindowCollector(2, max_alignment_age_ns=10 * DT_NS)
    collector.submit_candidate(torch.full((1, 29), 0.01))
    execute_step(collector, 0)
    _, first = execute_step(collector, 1)
    execute_step(collector, 2)
    _, second = execute_step(collector, 3)
    collector.close()
    return (*first, *second)


def test_boundary_lock_and_delayed_labels_keep_initial_context_and_path(tmp_path):
    collector = WindowCollector(2, max_alignment_age_ns=10 * DT_NS)
    collector.submit_candidate(torch.full((1, 29), 0.01))
    first, samples = execute_step(collector, 0, arrivals=())
    assert not samples
    collector.submit_candidate(torch.full((1, 29), -0.02))
    second, samples = execute_step(collector, 1, arrivals=())
    assert not samples
    third, first_samples = execute_step(collector, 2, arrivals=tuple(aligned_step(i) for i in range(3)))
    _, second_samples = execute_step(collector, 3)
    for correction in (first, second):
        torch.testing.assert_close(correction[0, 0, 3:32], torch.full((29,), 0.025), atol=1e-7, rtol=0)
    torch.testing.assert_close(third[0, 0, 3:32], torch.full((29,), -0.05), atol=1e-7, rtol=0)
    a, b = first_samples[0], second_samples[0]
    assert a.start_query.observation_history[0, -1, 0] == 0
    assert b.start_query.observation_history[0, -1, 0] == 2
    np.testing.assert_allclose(a.delta, [1., 0.4, 0.2], atol=2e-7, rtol=0)
    expected_second = [math.cos(0.2) + 0.4 * math.sin(0.2), -math.sin(0.2) + 0.4 * math.cos(0.2), 0.2]
    np.testing.assert_allclose(b.delta, expected_second, atol=2e-7, rtol=0)
    np.testing.assert_allclose(b.path_errors, [[0.4, 0.2], [0.8, 0.4]], atol=1e-12, rtol=0)
    ey, epsi = b.path_errors[0]
    np.testing.assert_allclose(ey + math.sin(epsi) * b.delta[0] + math.cos(epsi) * b.delta[1], 0.8, atol=2e-7, rtol=0)
    assert a.reference == b.reference and a.reference.timestamp_ns == 0
    writer = PathFollowingDatasetWriter(tmp_path, source={"label_source": "recorded_fixture"}, shard_size=1)
    writer.append(a)
    writer.append(b)
    writer.close()
    loaded = list(iter_path_following_samples(tmp_path))
    np.testing.assert_allclose(loaded[1].delta, expected_second, atol=2e-7, rtol=0)
    np.testing.assert_allclose(loaded[1].path_errors, b.path_errors, atol=0, rtol=0)
    assert loaded[0].end_pose.source_timestamps_ns == a.end_pose.source_timestamps_ns
    assert loaded[1].controls[0].start_ns == 2 * DT_NS
    assert len(read_path_following_metadata(tmp_path)["shards"]) == 2
    collector.close()


@pytest.mark.parametrize("failure", ["terminal", "epoch", "frame", "initial_pose_missing"])
def test_invalid_windows_never_become_samples_or_new_reference(failure):
    collector = WindowCollector(2, max_alignment_age_ns=10 * DT_NS)
    collector.submit_candidate(torch.zeros(1, 29))
    execute_step(collector, 0, arrivals=() if failure == "initial_pose_missing" else None)
    original_reference = collector.reference
    _, samples = execute_step(
        collector, 1, epoch=2 if failure == "epoch" else 1,
        frame="relocalized" if failure == "frame" else "world", terminal=failure == "terminal",
    )
    assert not samples and collector.halted
    assert collector.reference == original_reference
    collector.close()
    assert sum(collector.dropped_windows.values()) == 1


def test_missing_control_pose_invalidates_whole_window_and_partial_tail():
    collector = WindowCollector(2, max_alignment_age_ns=10 * DT_NS)
    collector.submit_candidate(torch.zeros(1, 29))
    execute_step(collector, 0)
    execute_step(collector, 1, arrivals=())
    _, samples = execute_step(collector, 2)
    assert not samples
    collector.close()
    assert collector.dropped_windows == {"missing_pose_interval": 1, "partial_window": 1}


def test_expired_window_cannot_be_revived_by_late_pose():
    collector = WindowCollector(2, max_alignment_age_ns=DT_NS // 4)
    collector.submit_candidate(torch.zeros(1, 29))
    execute_step(collector, 0)
    execute_step(collector, 1, arrivals=())
    execute_step(collector, 2, arrivals=())
    _, samples = execute_step(collector, 3, arrivals=tuple(aligned_step(i) for i in (1, 2, 3)))
    assert [sample.window_id for sample in samples] == [1]
    assert collector.completed_windows == 1
    assert collector.dropped_windows == {"alignment_timeout": 1}
    collector.close()


def test_reset_and_wrong_period_reject_execution_without_crossing_task():
    collector = WindowCollector(2, max_alignment_age_ns=10 * DT_NS)
    collector.submit_candidate(torch.zeros(1, 29))
    execute_step(collector, 0)
    with pytest.raises(ValueError, match="reset"):
        collector.begin_step(SensorClock(2, 0))
    assert collector.halted and collector.completed_windows == 0
    collector.close()
    collector = WindowCollector(2, max_alignment_age_ns=10 * DT_NS)
    collector.submit_candidate(torch.zeros(1, 29))
    collector.begin_step(SensorClock(1, 0))
    collector.transform(make_query(0))
    interval = aligned_step(0)
    with pytest.raises(ValueError, match="physical period"):
        collector.complete_step(
            replace(interval.control, end_ns=DT_NS + 100), PoseRuntimeUpdate((interval,), "tracking", 1, 1),
        )
    assert collector.halted and collector.completed_windows == 0
    collector.close()


def test_storage_rejects_unfinished_old_and_inconsistent_labels(tmp_path):
    unfinished = tmp_path / "unfinished"
    writer = PathFollowingDatasetWriter(unfinished, source={})
    with pytest.raises(ValueError, match="no complete"):
        writer.close()
    with pytest.raises(ValueError, match="incomplete"):
        read_path_following_metadata(unfinished)
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "source.json").write_text(json.dumps({"format": "retired_spatial", "complete": True}))
    with pytest.raises(ValueError, match="unsupported"):
        list(iter_path_following_samples(legacy))
    output = tmp_path / "corrupt"
    writer = PathFollowingDatasetWriter(output, source={})
    writer.append(complete_samples()[0])
    writer.close()
    shard = output / "windows_000000.npz"
    with np.load(shard, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    arrays["delta"][0, 1] += 1.
    np.savez_compressed(shard, **arrays)
    with pytest.raises(ValueError, match="stored response labels"):
        list(iter_path_following_samples(output))
