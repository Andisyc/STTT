"""Input-ablation evidence, group-preserving permutations and read-only provenance."""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from test_path_following_response import ROOT, capture_batch, write_analytic_capture
from unilab.algos.torch.sttt.path_following import residual_diagnostics as diagnostics
from unilab.algos.torch.sttt.path_following.checkpoint import LoadedResponseModel, save_response_checkpoint
from unilab.algos.torch.sttt.path_following.dataset import PathFollowingDatasetWriter, iter_path_following_samples, read_path_following_metadata
from unilab.algos.torch.sttt.path_following.features import response_contract_from_metadata
from unilab.algos.torch.sttt.path_following.model import ResponseModel


class AnalyticalResponse(ResponseModel):
    """The fixture's known physical-unit mapping, with no optimizer steps."""
    def __init__(self, sensitive=True):
        super().__init__(2, 3, [8, 8])
        self.sensitive = sensitive

    def forward(self, context, residual_rad):
        r = residual_rad if self.sensitive else torch.zeros_like(residual_rad)
        return torch.stack((0.03 + 0.4 * r[:, 0], 0.5 * r[:, 1], 0.6 * r[:, 2]), dim=1)


def fingerprint(directory):
    metadata = read_path_following_metadata(directory)
    digest = hashlib.sha256()
    for shard in metadata['shards']:
        digest.update((directory / shard['file']).read_bytes())
    return digest.hexdigest()


def source_entry(path, split):
    metadata = read_path_following_metadata(path)
    return {'path': str(path), 'split': split, 'capture_sha256': fingerprint(path),
            'samples': metadata['sample_count'], 'source': metadata['source']}


def diagnostic_fixture(directory, *, sensitive=True):
    train, validation = directory / 'train', directory / 'validation'
    write_analytic_capture(train, seed=1, samples=8)
    write_analytic_capture(validation, seed=5, samples=12)
    with torch.random.fork_rng():
        torch.manual_seed(3)
        model = AnalyticalResponse(sensitive).double()
    model.fit_normalization([capture_batch(train)], epsilon=1e-6)
    contract = response_contract_from_metadata(read_path_following_metadata(train))
    metadata = {'data_sources': [source_entry(train, 'train'), source_entry(validation, 'validation')],
                'training_config': {}, 'metrics': {}}
    return LoadedResponseModel(model, contract, metadata), train, validation


def test_correct_residual_beats_zero_and_shuffle_without_mutating_model_or_data(tmp_path):
    loaded, _, validation = diagnostic_fixture(tmp_path)
    before = {name: value.clone() for name, value in loaded.model.state_dict().items()}
    data_before = fingerprint(validation)
    report = diagnostics.compare_response_residuals(loaded, batch_size=5)
    assert report['correct']['samples'] == 12
    assert max(report['correct']['rmse_m_m_rad']) < 1e-7
    assert min(report['zero']['delta_rmse_m_m_rad']) > 0.001
    assert min(report['shuffled']['delta_rmse_m_m_rad']['min']) > 0.001
    assert report['physical_direction_verified'] is False
    assert all(trial['unchanged_vector_fraction'] == 0 for trial in report['shuffled']['trials'])
    for name, value in loaded.model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert fingerprint(validation) == data_before


def test_residual_independent_predictor_has_equal_errors_under_all_three_inputs(tmp_path):
    loaded, _, _ = diagnostic_fixture(tmp_path, sensitive=False)
    report = diagnostics.compare_response_residuals(loaded, batch_size=5)
    np.testing.assert_array_equal(report['correct']['rmse_m_m_rad'], report['zero']['rmse_m_m_rad'])
    for trial in report['shuffled']['trials']:
        np.testing.assert_array_equal(trial['metrics']['rmse_m_m_rad'], report['correct']['rmse_m_m_rad'])


def test_global_permutation_and_scores_do_not_depend_on_batch_size(tmp_path):
    loaded, _, _ = diagnostic_fixture(tmp_path)
    small = diagnostics.compare_response_residuals(loaded, batch_size=1, shuffle_seed=19)
    large = diagnostics.compare_response_residuals(loaded, batch_size=7, shuffle_seed=19)
    assert [trial['permutation_sha256'] for trial in small['shuffled']['trials']] == [trial['permutation_sha256'] for trial in large['shuffled']['trials']]
    for left, right in zip(small['shuffled']['trials'], large['shuffled']['trials']):
        np.testing.assert_allclose(left['metrics']['rmse_m_m_rad'], right['metrics']['rmse_m_m_rad'], rtol=0, atol=1e-12)


def test_derangement_does_not_cross_localization_epochs(tmp_path):
    loaded, _, validation = diagnostic_fixture(tmp_path)
    original = read_path_following_metadata(validation)
    grouped = tmp_path / 'two_epochs'
    writer = PathFollowingDatasetWriter(grouped, source=original['source'])
    for index, sample in enumerate(iter_path_following_samples(validation)):
        if index >= 6:
            sample = replace(sample, start_pose=replace(sample.start_pose, tracking_epoch=2),
                             end_pose=replace(sample.end_pose, tracking_epoch=2),
                             reference=replace(sample.reference, tracking_epoch=2))
        writer.append(sample)
    writer.close()
    capture, = diagnostics._captures(loaded, [grouped])
    permutation = diagnostics._derangement(capture, np.random.default_rng(13))
    assert not np.any(permutation == np.arange(12))
    np.testing.assert_array_equal(np.sort(permutation), np.arange(12))
    assert np.all(permutation[:6] < 6) and np.all(permutation[6:] >= 6)


@pytest.mark.parametrize('failure', ['training', 'copied_training', 'tampered_validation', 'incompatible_contract', 'single_window'])
def test_invalid_or_nonheldout_inputs_fail_closed(tmp_path, failure):
    loaded, train, validation = diagnostic_fixture(tmp_path)
    selected = None
    if failure == 'training':
        selected = [train]
    elif failure == 'copied_training':
        copied = tmp_path / 'copy'
        shutil.copytree(train, copied)
        selected = [copied]
    elif failure == 'tampered_validation':
        shard = validation / 'windows_000000.npz'
        with np.load(shard, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        arrays['delta'][0, 0] += 0.01
        np.savez_compressed(shard, **arrays)
    elif failure == 'incompatible_contract':
        path = validation / 'source.json'
        payload = json.loads(path.read_text())
        payload['contract']['window_steps'] += 1
        path.write_text(json.dumps(payload))
    else:
        single = tmp_path / 'single'
        write_analytic_capture(single, seed=21, samples=1)
        selected = [single]
    with pytest.raises(ValueError):
        diagnostics.compare_response_residuals(loaded, selected)


def test_cli_runner_uses_recorded_validation_and_never_overwrites_output_or_checkpoint(tmp_path):
    loaded, _, _ = diagnostic_fixture(tmp_path)
    # Save/load the actual existing MLP artifact schema. No subclass identity or
    # analytical forward is serialized, and no training occurs in this check.
    path = save_response_checkpoint(tmp_path / 'b.pt', loaded.model, contract=loaded.contract,
                                    training_config={}, data_sources=loaded.metadata['data_sources'], metrics={})
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    cfg = OmegaConf.load(ROOT / 'conf/sttt/response_residual_diagnostics.yaml')
    assert isinstance(cfg, DictConfig)
    cfg.checkpoint_path = str(path)
    cfg.batch_size = 5
    report = diagnostics.run_residual_input_diagnostics(cfg)
    assert report['captures'][0]['recorded_validation'] is True
    assert report['correct']['samples'] == 12
    assert report['checkpoint']['sha256'] == before
    output = Path(report['output_path'])
    saved = output.read_bytes()
    assert json.loads(saved)['comparison'] == 'ablation_rmse_minus_correct_rmse'
    with pytest.raises(FileExistsError):
        diagnostics.run_residual_input_diagnostics(cfg)
    assert output.read_bytes() == saved
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
