"""Physical-time scoring and real factory evaluation over analytical I/O only."""
from __future__ import annotations

import csv
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from test_path_following_runtime import ROOT, SyntheticEnv, SyntheticWrapper, runtime_case
from unilab.algos.torch.distill.fada.checkpoint import LoadedOfficialFADAStudentPolicy
from unilab.algos.torch.sttt.path_following.evaluation import run_path_following_evaluation
from unilab.algos.torch.sttt.path_following.evaluation_metrics import PathFollowingMetrics, PathFollowingThresholds
from unilab.algos.torch.sttt.pose_data import AlignedPoseInterval, PoseRuntimeUpdate, TimedPelvisPose
from unilab.base.backend.sensors import SensorClock


def thresholds():
    return PathFollowingThresholds(0.1, 0.1, 1., 0.02, 0.2)


def feed(points):
    metrics = PathFollowingMetrics(thresholds())
    for timestamp, errors, progress in points:
        metrics.observe(int(timestamp * 1e9), None if errors is None else np.array(errors), progress)
    return metrics


def test_time_weighted_rmse_and_joint_recovery_are_not_sample_averages():
    summary = feed([(0, [0., 0.], 0.), (1, [2., 0.2], 0.4), (3, [0., 0.], 1.)]).summary()
    np.testing.assert_allclose(summary['rmse_m_rad'], [math.sqrt(2), math.sqrt(0.02)], atol=1e-12)
    assert summary['tracking_state'] == 'not_recovered'
    assert summary['recovery_entry_time_s'] is None
    yaw_only = feed([(0, [0., 0.2], 0.), (2, [0., 0.2], 1.)]).summary()
    assert yaw_only['tracking_state'] == 'not_recovered'


def test_maintaining_a_path_is_not_recovery_or_progress_while_standing():
    maintained = feed([(0, [0., 0.], 0.), (2, [0., 0.], 1.)]).summary()
    stationary = feed([(0, [0., 0.], 0.), (2, [0., 0.], 0.)]).summary()
    assert maintained['tracking_state'] == 'maintained' and maintained['progress_pass']
    assert maintained['recovery_latency_s'] is None
    assert stationary['tracking_state'] == 'maintained' and not stationary['progress_pass']


def test_dwell_deadband_and_recovery_loss_are_separate_evidence():
    metric = feed([(0, [0.3, 0.], 0.), (1, [0.05, 0.05], 0.3), (2, [0.01, 0.], 0.6),
                   (3, [-0.05, 0.], 0.9), (4, [0.04, 0.], 1.2)])
    summary = metric.summary()
    assert summary['tracking_state'] == 'recovered'
    assert summary['recovery_entry_time_s'] == 1. and summary['recovery_confirmed_time_s'] == 2.
    assert summary['recovery_latency_s'] == 1. and summary['lateral_crossings'] == 2
    metric.observe(5_000_000_000, np.array([0.3, 0.]), 1.5)
    lost = metric.summary()
    assert lost['tracking_state'] == 'recovered_then_lost'
    assert lost['post_recovery_peak_lateral_m'] == 0.3


def test_missing_measurements_break_dwell_and_cannot_fill_unobserved_time():
    metric = feed([(0, [0., 0.], 0.), (1, None, None), (2, [0., 0.], 0.4), (3, [0., 0.], 0.6)])
    summary = metric.summary()
    assert summary['tracking_state'] == 'insufficient_coverage'
    assert summary['observed_duration_s'] == 1. and summary['time_coverage'] == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match='strictly increase'):
        metric.observe(3_000_000_000, np.zeros(2), 0.6)


def evaluation_case(directory, *, modes=('oat_path',), condition='clean', failure=None, delay=0):
    case = runtime_case(directory / 'inputs', online=True, delay=delay)
    config = OmegaConf.merge(OmegaConf.load(ROOT / 'conf/sttt/path_following_evaluate.yaml'), {
        'output_dir': str(directory / 'result'),
        'policy_checkpoint': str(case.deps['resolve_checkpoint'](None)),
        'profile_path': case.cfg.official_playback.profile_path,
        'response_checkpoint': case.cfg.path_following.checkpoint_path,
        'modes': list(modes), 'conditions': [condition], 'seeds': [17], 'duration_s': 0.2,
        'thresholds': {'lateral_tolerance_m': 0.1, 'heading_tolerance_rad': 0.2, 'dwell_s': 0.04,
                       'crossing_deadband_m': 0.01, 'minimum_forward_progress_m': 0.01},
        'source_overrides': [
            'env.runtime_payload.attach_time_s=0.04',
            'online_adaptation.window_size=2', 'online_adaptation.warmup_steps=6', 'online_adaptation.update_interval=2',
        ],
    })
    assert isinstance(config, DictConfig)
    if modes == ('oat_path',):
        config.source_overrides.extend([
            'path_following.update_interval_steps=2', 'path_following.max_trace_records=1',
            '++path_following.learning.actor.hidden_dims=[16,8]',
        ])
    loaded_policies = []
    original_loader = case.deps['load_fada_policy']
    def load_policy(*args, **kwargs):
        loaded = original_loader(*args, **kwargs)
        policy = deepcopy(loaded.policy)
        loaded_policies.append(policy)
        return LoadedOfficialFADAStudentPolicy(policy, loaded.checkpoint)
    case.deps['load_fada_policy'] = load_policy
    def create_env(*_, env_cfg_override, **__):
        env = SyntheticEnv(env_cfg_override)
        for name in ('commands', 'runtime_payload', 'domain_rand', 'noise_config'):
            setattr(env.cfg, name, OmegaConf.create(env_cfg_override[name]))
        env.state.info.update({
            'runtime_payload_active': np.zeros(1, dtype=bool),
            'runtime_payload_attached_this_step': np.zeros(1, dtype=bool),
            'runtime_payload_added_mass_kg': np.zeros(1),
        })
        case.environments.append(env)
        return env
    class Wrapper(SyntheticWrapper):
        def step(self, action):
            info = self.env.state.info
            info['runtime_payload_attached_this_step'][:] = False
            if self.env.cfg.runtime_payload.enabled and self.env.tick == 2 and failure != 'payload_missing':
                info['runtime_payload_active'][:] = True
                info['runtime_payload_attached_this_step'][:] = True
                info['runtime_payload_added_mass_kg'][:] = 3.
            result = super().step(action)
            if self.env.tick == 3:
                if failure == 'terminal':
                    self.env.state.terminated[:] = True
                    result = (result[0], result[1], torch.tensor([True]), result[3])
                elif failure == 'lost':
                    case.sources[-1].lost = True
                elif failure == 'post_step_error':
                    raise RuntimeError('analytical post-physics I/O failure')
                elif failure == 'missing_truth':
                    self.env.get_base_pos = lambda: np.array([[float('nan'), 0., 0.]])
            return result
    def wrapper(env, **kwargs):
        obj = Wrapper(env, **kwargs)
        case.wrappers.append(obj)
        return obj
    def overrides(cfg):
        return OmegaConf.to_container(cfg.env, resolve=True)
    case.deps.update(create_env=create_env, wrapper_cls=wrapper, build_env_cfg_override=overrides)
    return case, config, loaded_policies


def load_rows(directory, name):
    with (directory / name).open(newline='') as handle:
        return list(csv.DictReader(handle))


def test_full_evidence_survives_tiny_runtime_ring_and_delayed_pose_reuse(tmp_path):
    case, cfg, _ = evaluation_case(tmp_path, delay=2)
    result = run_path_following_evaluation(cfg, root_dir=ROOT, playback_deps=case.deps)
    assert result['complete']
    run = result['runs'][0]
    directory = Path(cfg.output_dir) / run['directory']
    trajectory = load_rows(directory, 'trajectory.csv')
    records = [json.loads(line) for line in (directory / 'runtime.jsonl').read_text().splitlines()]
    assert [int(row['timestamp_ns']) for row in trajectory] == list(range(0, 200_000_001, 20_000_000))
    assert len(records) == 10 and records[0]['measurement_timestamp_ns'] is None
    assert records[2]['measurement_timestamp_ns'] == 0 and records[2]['arrival_timestamp_ns'] == 60_000_000
    assert run['runtime']['unique_measurements'] == 8
    assert run['runtime']['unique_mean_age_s'] == pytest.approx(0.06)
    assert run['runtime']['a_updates'] == 3 and run['runtime']['b_updates'] == 4
    assert case.environments[0].closed and case.sources[0].closed
    assert not case.environments[0].autoreset
    assert json.loads((directory / 'source.json').read_text())['evaluation_pose_source'] == 'simulation_ground_truth_score_only'


def test_modes_have_fresh_policies_and_actual_payload_events_not_schedule_only(tmp_path):
    case, cfg, policies = evaluation_case(tmp_path, modes=('nominal', 'oat', 'oat_path'), condition='delayed_payload')
    result = run_path_following_evaluation(cfg, root_dir=ROOT, playback_deps=case.deps)
    assert result['complete'] and len(result['runs']) == 3
    assert len({next(policy.parameters()).data_ptr() for policy in policies}) == 3
    for run in result['runs']:
        assert run['condition_observed'] and run['actual_payload_event_start_ns'] == 40_000_000
        assert run['post_payload_metrics']['duration_s'] == pytest.approx(0.16)
        directory = Path(cfg.output_dir) / run['directory']
        source = json.loads((directory / 'source.json').read_text())
        assert source['playback_config']['algo']['seed'] == 17
        assert source['effective_conditions']['runtime_payload']['enabled'] is True
        rows = load_rows(directory, 'trajectory.csv')
        assert sum(row['payload_attached_this_step'] == 'True' for row in rows) == 1
    assert all(env.closed and not env.autoreset for env in case.environments)


@pytest.mark.parametrize('failure,reason', [('terminal', 'terminated'), ('lost', 'error'), ('post_step_error', 'error'), ('missing_truth', 'error')])
def test_early_failure_preserves_final_evidence_and_never_claims_completion(tmp_path, failure, reason):
    case, cfg, _ = evaluation_case(tmp_path, failure=failure)
    result = run_path_following_evaluation(cfg, root_dir=ROOT, playback_deps=case.deps)
    run = result['runs'][0]
    assert not result['complete'] and not run['complete'] and not run['tracking_success']
    assert run['stop_reason'] == reason and run['metrics']['duration_s'] == pytest.approx(0.06)
    directory = Path(cfg.output_dir) / run['directory']
    rows = load_rows(directory, 'trajectory.csv')
    assert int(rows[-1]['timestamp_ns']) == 60_000_000
    if failure == 'missing_truth':
        assert rows[-1]['valid'] == 'False' and run['metrics']['tracking_state'] == 'insufficient_coverage'
    if failure == 'lost':
        assert json.loads((directory / 'runtime.jsonl').read_text().splitlines()[-1])['event'] == 'halt'
    assert case.environments[0].closed and case.sources[0].closed


def test_absent_payload_is_not_a_completed_payload_experiment(tmp_path):
    case, cfg, _ = evaluation_case(tmp_path, condition='delayed_payload', failure='payload_missing')
    result = run_path_following_evaluation(cfg, root_dir=ROOT, playback_deps=case.deps)
    run = result['runs'][0]
    assert run['horizon_reached'] and not run['condition_observed']
    assert not run['complete'] and not run['tracking_success']
    assert run['post_payload_metrics'] is None


def test_preflight_requires_explicit_thresholds_and_never_overwrites_artifacts(tmp_path):
    case, cfg, _ = evaluation_case(tmp_path)
    cfg.thresholds.dwell_s = None
    with pytest.raises(ValueError, match='explicit positive'):
        run_path_following_evaluation(cfg, root_dir=ROOT, playback_deps=case.deps)
    assert not case.environments and not Path(cfg.output_dir).exists()
    cfg.thresholds.dwell_s = 0.04
    output = Path(cfg.output_dir)
    output.mkdir()
    (output / 'keep.txt').write_text('user evidence')
    with pytest.raises(FileExistsError):
        run_path_following_evaluation(cfg, root_dir=ROOT, playback_deps=case.deps)
    assert (output / 'keep.txt').read_text() == 'user evidence'


def test_evaluation_truth_does_not_replace_estimator_errors_and_repeated_pose_is_not_recounted(tmp_path):
    case, cfg, _ = evaluation_case(tmp_path)
    cfg.pose_source = 'cuvslam'

    class RecordedPoses:
        """A controlled estimator channel that reports no lateral motion, not a truth fallback."""
        def __init__(self, env):
            self.env, self.sequence, self.pending = env, 0, []
            self.closed = False
        @property
        def clock(self): return SensorClock(self.sequence, self.env.get_sim_time_ns())
        def begin_step(self): return self.clock
        def reset(self):
            self.sequence += 1
            self.pending.clear()
        def close(self): self.closed = True
        def complete_step(self, control):
            matrix = np.eye(4)
            start = TimedPelvisPose(self.sequence, 1, control.start_ns, matrix, True, (control.start_ns, control.start_ns), 'recorded_estimator')
            end = TimedPelvisPose(self.sequence, 1, control.end_ns, matrix, True, (control.end_ns, control.end_ns), 'recorded_estimator')
            self.pending.append(AlignedPoseInterval(control, start, end))
            ready = tuple(self.pending) if len(self.pending) == 2 else ()
            if ready:
                self.pending.clear()
            return PoseRuntimeUpdate(ready, 'tracking', self.sequence, 1)

    def pose_source(env, config):
        assert config.odometry_source == 'cuvslam'
        source = RecordedPoses(env)
        case.sources.append(source)
        return source
    case.deps['create_pose_runtime'] = pose_source
    result = run_path_following_evaluation(cfg, root_dir=ROOT, playback_deps=case.deps)
    run = result['runs'][0]
    assert result['complete']
    directory = Path(cfg.output_dir) / run['directory']
    records = [json.loads(line) for line in (directory / 'runtime.jsonl').read_text().splitlines()]
    assert records[-1]['errors_m_rad'] == [0., 0.]
    assert float(load_rows(directory, 'trajectory.csv')[-1]['ey_m']) == pytest.approx(0.03)
    assert run['runtime']['unique_measurements'] == 5
    assert run['runtime']['paired_control_fraction'] == pytest.approx(0.9)
    assert run['runtime']['unique_max_age_s'] == pytest.approx(0.02)
    assert sum(record['new_measurement'] for record in records) == 5
