"""Fresh public playback sessions with controlled I/O; no native dynamics rollout."""
from __future__ import annotations

import json
import runpy
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from hydra.errors import ConfigCompositionException
from omegaconf import DictConfig, OmegaConf

from test_path_following_runtime import ROOT, SyntheticEnv, SyntheticWrapper, runtime_case
from unilab.algos.torch.distill.fada.checkpoint import LoadedOfficialFADAStudentPolicy
from unilab.algos.torch.sttt.path_following import response_probe as probe_module
from unilab.algos.torch.sttt.path_following.response_probe import compare_probe_responses, run_response_direction_probe


def probe_case(directory, *, failure=None):
    case = runtime_case(directory / 'inputs', enabled=False)
    cfg = OmegaConf.merge(OmegaConf.load(ROOT / 'conf/sttt/response_direction_probe.yaml'), {
        'policy_checkpoint': str(case.deps['resolve_checkpoint'](None)),
        'profile_path': case.cfg.official_playback.profile_path,
        'response_checkpoint': str(directory / 'inputs/b.pt'), 'output_dir': str(directory / 'probe'),
        'warmup_steps': 4, 'residual_amplitude_rad': 0.001,
        'replay_position_tolerance_m': 1e-8, 'replay_heading_tolerance_rad': 1e-8,
        'minimum_effect_m': 1e-5, 'minimum_effect_rad': 1e-5,
    })
    assert isinstance(cfg, DictConfig)
    loaded_policies = []
    loader = case.deps['load_fada_policy']
    def load_policy(*args, **kwargs):
        loaded = loader(*args, **kwargs)
        policy = deepcopy(loaded.policy)
        role = len(loaded_policies)
        if failure == 'query' and role == 1:
            def alter_query(_, args, output):
                return output + 0.001 if case.environments[-1].tick == 4 else output
            policy.planner.register_forward_hook(alter_query)
        loaded_policies.append(policy)
        return LoadedOfficialFADAStudentPolicy(policy, loaded.checkpoint)
    def create_env(*_, env_cfg_override, **__):
        role = len(case.environments)
        env = SyntheticEnv(env_cfg_override)
        original = env.get_base_pos
        if failure == 'start' and role == 1:
            env.get_base_pos = lambda: original() + np.array([[0.001, 0., 0.]])
        elif failure == 'zero_repeat' and role == 3:
            env.get_base_pos = lambda: original() + (np.array([[0.01, 0., 0.]]) if env.tick > 4 else 0.)
        def capture():
            native = np.concatenate((np.array([env.get_sim_time_ns() / 1e9]), env.get_base_pos()[0],
                                     env.get_base_quat()[0], env.get_dof_pos()[0], env.get_dof_vel()[0]))[None]
            return SimpleNamespace(backend_state={'physics_state': native.astype(np.float64)})
        object.__setattr__(env, 'capture_rollout_snapshot', capture)
        case.environments.append(env)
        return env
    class Wrapper(SyntheticWrapper):
        def step(self, action):
            result = super().step(action)
            if failure == 'terminal' and self.env.tick == 5:
                self.env.state.terminated[:] = True
                result = (result[0], result[1], torch.tensor([True]), result[3])
            return result
    def wrapper(env, **kwargs):
        obj = Wrapper(env, **kwargs)
        case.wrappers.append(obj)
        return obj
    case.deps.update(load_fada_policy=load_policy, create_env=create_env, wrapper_cls=wrapper)
    return case, cfg, loaded_policies


def analytic_inputs():
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    deltas = {'zero_a': [0.08, 0., 0.], 'zero_b': [0.08, 0., 0.],
              'plus': [0.09, 0.01, 0.01], 'minus': [0.07, -0.01, -0.01]}
    predictions = {'zero': [0.08, 0., 0.], 'plus': deltas['plus'], 'minus': deltas['minus']}
    return poses, deltas, predictions


def compare(deltas, predictions, a, b):
    return compare_probe_responses(deltas, predictions, a, b, replay_position_tolerance_m=1e-6,
        replay_heading_tolerance_rad=1e-6, minimum_effect_m=1e-4, minimum_effect_rad=1e-4)


@pytest.mark.parametrize('reversed_prediction', [False, True])
def test_known_response_direction_is_reported_per_channel(reversed_prediction):
    poses, deltas, predictions = analytic_inputs()
    if reversed_prediction:
        predictions['plus'], predictions['minus'] = predictions['minus'], predictions['plus']
    result = compare(deltas, predictions, poses, poses.copy())
    assert result['valid']
    assert result['channel_status'] == ['opposite_sign' if reversed_prediction else 'same_sign'] * 3
    np.testing.assert_allclose(result['actual_plus_minus_m_m_rad'], [0.02, 0.02, 0.02], atol=1e-12, rtol=0)


def test_weak_effect_and_failed_zero_replay_do_not_become_direction_verdicts():
    poses, deltas, predictions = analytic_inputs()
    deltas['plus'] = [0.080001, 1e-6, 1e-6]
    deltas['minus'] = [0.08, 0., 0.]
    result = compare(deltas, predictions, poses, poses.copy())
    assert result['channel_status'] == ['unresolved_effect'] * 3
    drift = poses.copy()
    drift[-1, 0, 3] = 0.01
    result = compare(deltas, predictions, poses, drift)
    assert not result['valid'] and result['channel_status'] == ['invalid_replay'] * 3


def test_four_fresh_sessions_preserve_prefix_queries_and_bounded_budget(tmp_path):
    case, cfg, policies = probe_case(tmp_path)
    result = run_response_direction_probe(cfg, root_dir=ROOT, playback_deps=case.deps)
    assert result['complete'] and result['valid'], result
    assert [branch['branch'] for branch in result['branches']] == ['zero_a', 'plus', 'minus', 'zero_b']
    assert all(branch['completed_steps'] == 6 for branch in result['branches'])
    assert len({branch['context_sha256'] for branch in result['branches']}) == 1
    assert len({next(policy.parameters()).data_ptr() for policy in policies}) == 4
    output = Path(cfg.output_dir)
    with np.load(output / 'zero_a.npz', allow_pickle=False) as zero, np.load(output / 'plus.npz', allow_pickle=False) as plus:
        assert zero['start_native_state'].dtype == np.float64
        np.testing.assert_array_equal(zero['start_native_state'], plus['start_native_state'])
        np.testing.assert_array_equal(zero['start_warmup_actions'], plus['start_warmup_actions'])
        np.testing.assert_array_equal(zero['context'], plus['context'])
        assert plus['timestamps_ns'][-1] - plus['timestamps_ns'][4] == 40_000_000
    assert result['comparison']['channel_status'] == ['unresolved_effect'] * 3  # Controlled poses intentionally ignore interventions.
    assert all(env.closed and not env.autoreset for env in case.environments)
    assert all(not parameter.requires_grad for policy in policies for parameter in policy.parameters())


def test_swapping_positive_negative_order_keeps_results_attached_to_the_branch(tmp_path):
    case, cfg, _ = probe_case(tmp_path)
    first = run_response_direction_probe(cfg, root_dir=ROOT, playback_deps=case.deps)
    cfg.branch_order = ['zero_a', 'minus', 'plus', 'zero_b']
    cfg.output_dir = str(tmp_path / 'reordered')
    second = run_response_direction_probe(cfg, root_dir=ROOT, playback_deps=case.deps)
    assert first['valid'] and second['valid']
    by_name = {item['branch']: item for item in first['branches']}
    for branch in second['branches']:
        assert branch['residual_rad'] == by_name[branch['branch']]['residual_rad']
        np.testing.assert_array_equal(branch['delta_m_m_rad'], by_name[branch['branch']]['delta_m_m_rad'])
    assert first['predictions_m_m_rad'] == second['predictions_m_m_rad']


@pytest.mark.parametrize('failure', ['start', 'query', 'terminal'])
def test_invalid_prefix_query_or_terminal_stops_without_replacing_the_start(tmp_path, failure):
    case, cfg, _ = probe_case(tmp_path, failure=failure)
    result = run_response_direction_probe(cfg, root_dir=ROOT, playback_deps=case.deps)
    assert not result['complete'] and not result['valid']
    assert result['comparison'] is None
    assert result['branches'][-1]['error'] is not None
    assert len(result['branches']) == (1 if failure == 'terminal' else 2)
    assert all(env.closed for env in case.environments)
    output = Path(cfg.output_dir)
    last = result['branches'][-1]['branch']
    assert (output / f'{last}.npz').is_file()
    assert not json.loads((output / 'summary.json').read_text())['valid']


def test_nonrepeating_zero_branch_invalidates_the_finished_group(tmp_path):
    case, cfg, _ = probe_case(tmp_path, failure='zero_repeat')
    result = run_response_direction_probe(cfg, root_dir=ROOT, playback_deps=case.deps)
    assert result['complete'] and not result['valid']
    assert result['comparison']['channel_status'] == ['invalid_replay'] * 3
    assert result['comparison']['zero_repeat_max_position_error_m'] == pytest.approx(0.01)
    assert all(env.closed for env in case.environments)


def test_residual_coverage_and_existing_evidence_fail_before_new_rollouts(tmp_path):
    case, cfg, _ = probe_case(tmp_path)
    cfg.residual_amplitude_rad = 0.1
    with pytest.raises(ValueError):
        run_response_direction_probe(cfg, root_dir=ROOT, playback_deps=case.deps)
    assert not case.environments
    cfg.residual_amplitude_rad = 0.001
    output = Path(cfg.output_dir)
    output.mkdir()
    (output / 'keep.json').write_text('user evidence')
    with pytest.raises(FileExistsError):
        run_response_direction_probe(cfg, root_dir=ROOT, playback_deps=case.deps)
    assert (output / 'keep.json').read_text() == 'user evidence' and not case.environments


@pytest.mark.parametrize('invalid_task_override', [False, True])
def test_real_hydra_main_preserves_outer_context_during_probe_composition(tmp_path, monkeypatch, invalid_task_override):
    """Enter the decorated CLI, not __wrapped__ or --cfg, with only physical I/O substituted."""
    case, cfg, _ = probe_case(tmp_path)
    script = ROOT / 'scripts/probe_response_directions.py'
    argv = [str(script)]
    for key in ('policy_checkpoint', 'profile_path', 'response_checkpoint', 'output_dir', 'device', 'seed', 'warmup_steps',
                'residual_amplitude_rad', 'replay_position_tolerance_m', 'replay_heading_tolerance_rad', 'minimum_effect_m', 'minimum_effect_rad'):
        argv.append(f'{key}={cfg[key]}')
    argv.extend([f'hydra.run.dir={tmp_path / "hydra_job"}', 'hydra.output_subdir=null',
                 'hydra/job_logging=disabled', 'hydra/hydra_logging=disabled'])
    if invalid_task_override:
        argv.append("source_overrides=['env.missing_probe_field=1']")
    observed = []
    def checked_probe(config, *, root_dir):
        assert GlobalHydra.instance().is_initialized()
        outer = GlobalHydra.instance().hydra
        outer_config = HydraConfig.get()
        observed.append('entered_real_main')
        try:
            result = run_response_direction_probe(config, root_dir=root_dir, playback_deps=case.deps)
            assert result['complete'] and result['valid']
            observed.append('completed_four_branches')
            return result
        finally:
            assert GlobalHydra.instance().hydra is outer
            assert HydraConfig.get() is outer_config
    monkeypatch.setattr(probe_module, 'run_response_direction_probe', checked_probe)
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setenv('HYDRA_FULL_ERROR', '1')
    if invalid_task_override:
        with pytest.raises(ConfigCompositionException):
            runpy.run_path(str(script), run_name='__main__')
        assert observed == ['entered_real_main'] and not case.environments
    else:
        runpy.run_path(str(script), run_name='__main__')
        assert observed == ['entered_real_main', 'completed_four_branches']
        assert len(case.environments) == 4 and all(env.closed for env in case.environments)
    assert not GlobalHydra.instance().is_initialized()
