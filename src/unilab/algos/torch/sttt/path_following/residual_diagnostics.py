"""Held-out predictive input ablation, not a physical direction or causal proof."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from unilab.algos.torch.sttt.path_following.checkpoint import LoadedResponseModel, load_response_checkpoint
from unilab.algos.torch.sttt.path_following.contract import _integer
from unilab.algos.torch.sttt.path_following.dataset import iter_path_following_samples, read_path_following_metadata
from unilab.algos.torch.sttt.path_following.diagnostics import evaluate_response_model
from unilab.algos.torch.sttt.path_following.features import pack_response_context, response_contract_from_metadata
from unilab.algos.torch.sttt.path_following.model import ResponseBatch

_GROUP = tuple[int, int, str]
_IDENTITY = tuple[int, int, int, str]


def _digest(path: Path, metadata: Mapping[str, Any]) -> str:
    # Same byte-order definition as the training artifact's capture_sha256.
    digest = hashlib.sha256()
    for shard in metadata["shards"]:
        with (path / shard["file"]).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _identity(sample: Any) -> _IDENTITY:
    pose = sample.start_pose
    return sample.window_id, pose.sequence_id, pose.tracking_epoch, pose.frame_id


@dataclass(frozen=True)
class _Capture:
    path: Path
    fingerprint: str
    metadata: dict[str, Any]
    identities: tuple[_IDENTITY, ...]
    residuals: torch.Tensor
    groups: dict[_GROUP, list[int]]

    def batches(self, batch_size: int, *, mode: str, permutation: np.ndarray | None = None) -> Iterator[ResponseBatch]:
        contexts, residuals, targets = [], [], []
        count = 0
        for index, sample in enumerate(iter_path_following_samples(self.path)):
            if index >= len(self.identities) or _identity(sample) != self.identities[index] or not torch.equal(sample.start_query.residual_rad[0], self.residuals[index]):
                raise ValueError("validation capture changed during diagnosis")
            contexts.append(pack_response_context(sample.start_query))
            if mode == "correct":
                residual = self.residuals[index]
            elif mode == "zero":
                residual = torch.zeros_like(self.residuals[index])
            elif mode == "shuffled" and permutation is not None:
                residual = self.residuals[int(permutation[index])]
            else:
                raise ValueError("invalid residual diagnostic input mode")
            residuals.append(residual.unsqueeze(0))
            targets.append(torch.from_numpy(sample.delta.copy()).unsqueeze(0))
            count += 1
            if len(contexts) == batch_size:
                yield torch.cat(contexts), torch.cat(residuals), torch.cat(targets)
                contexts.clear()
                residuals.clear()
                targets.clear()
        if count != len(self.identities):
            raise ValueError("validation capture sample count changed during diagnosis")
        if contexts:
            yield torch.cat(contexts), torch.cat(residuals), torch.cat(targets)


def _captures(loaded: LoadedResponseModel, paths: Sequence[str | Path] | None) -> list[_Capture]:
    sources = loaded.metadata.get("data_sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("B artifact lacks training/validation provenance; cannot establish held-out input diagnosis")
    training_paths, training_hashes, validation = set(), set(), {}
    for source in sources:
        if not isinstance(source, dict) or source.get("split") not in ("train", "validation"):
            raise ValueError("B artifact has invalid data-source provenance")
        path = Path(source["path"]).expanduser().resolve()
        fingerprint = source.get("capture_sha256")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(char not in "0123456789abcdef" for char in fingerprint):
            raise ValueError("B artifact has invalid capture fingerprints")
        if source["split"] == "train":
            training_paths.add(path)
            training_hashes.add(fingerprint)
        else:
            validation[path] = fingerprint
    if not training_paths:
        raise ValueError("B artifact does not identify its training captures")
    if paths is None or len(paths) == 0:
        paths = list(validation)
    if isinstance(paths, (str, bytes)) or not paths:
        raise ValueError("provide validation_dirs or a B artifact with recorded validation captures")
    result = []
    seen_paths, seen_hashes = set(), set()
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if path in training_paths:
            raise ValueError("training capture cannot be used as a held-out diagnostic set")
        if path in seen_paths:
            raise ValueError("duplicate diagnostic capture")
        metadata = read_path_following_metadata(path)
        if response_contract_from_metadata(metadata) != loaded.contract:
            raise ValueError("validation capture is incompatible with the B response contract")
        fingerprint = _digest(path, metadata)
        if fingerprint in training_hashes:
            raise ValueError("copied training data cannot be used as held-out diagnostic data")
        if fingerprint in seen_hashes:
            raise ValueError("duplicate diagnostic capture contents")
        if path in validation and fingerprint != validation[path]:
            raise ValueError("recorded validation capture contents no longer match the B artifact")
        identities, residuals, groups = [], [], {}
        for index, sample in enumerate(iter_path_following_samples(path)):
            identity = _identity(sample)
            identities.append(identity)
            residuals.append(sample.start_query.residual_rad[0])
            groups.setdefault(identity[1:], []).append(index)
        if len(identities) != metadata["sample_count"]:
            raise ValueError("diagnostic sample count does not match capture metadata")
        if any(len(indices) < 2 for indices in groups.values()):
            raise ValueError("each capture/sequence/epoch/frame group requires at least two windows for derangement")
        result.append(_Capture(path, fingerprint, metadata, tuple(identities), torch.stack(residuals), groups))
        seen_paths.add(path)
        seen_hashes.add(fingerprint)
    return result


def _derangement(capture: _Capture, rng: np.random.Generator) -> np.ndarray:
    """One Sattolo cycle per group: whole-vector, no fixed points, batch-independent."""
    permutation = np.arange(len(capture.identities), dtype=np.int64)
    for indices in capture.groups.values():
        shuffled = list(indices)
        for index in range(len(shuffled) - 1, 0, -1):
            other = int(rng.integers(index))
            shuffled[index], shuffled[other] = shuffled[other], shuffled[index]
        permutation[indices] = shuffled
    return permutation


def _aggregate(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    count = sum(item["samples"] for item in metrics)
    if count < 1:
        raise ValueError("no held-out diagnostic samples")
    weights = np.asarray([item["samples"] for item in metrics], dtype=np.float64)[:, None]
    squared = (np.square(np.stack([item["rmse_m_m_rad"] for item in metrics])) * weights).sum(axis=0)
    absolute = (np.stack([item["mae_m_m_rad"] for item in metrics]) * weights).sum(axis=0)
    return {
        "samples": count, "loss": sum(item["loss"] * item["samples"] for item in metrics) / count,
        "rmse_m_m_rad": np.sqrt(squared / count).tolist(), "mae_m_m_rad": (absolute / count).tolist(),
    }


def _difference(ablation: Mapping[str, Any], correct: Mapping[str, Any]) -> list[float]:
    return (np.asarray(ablation["rmse_m_m_rad"]) - np.asarray(correct["rmse_m_m_rad"])).tolist()


def _shuffle_summary(trials: list[dict[str, Any]], correct: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"trials": trials}
    for name in ("loss", "rmse_m_m_rad", "mae_m_m_rad"):
        values = np.asarray([trial["metrics"][name] for trial in trials])
        result[name] = {"mean": values.mean(axis=0).tolist(), "min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist()}
    differences = np.asarray([_difference(trial["metrics"], correct) for trial in trials])
    result["delta_rmse_m_m_rad"] = {"mean": differences.mean(axis=0).tolist(), "min": differences.min(axis=0).tolist(), "max": differences.max(axis=0).tolist()}
    return result


def compare_response_residuals(
    loaded: LoadedResponseModel, validation_dirs: Sequence[str | Path] | None = None, *,
    batch_size: int = 128, shuffle_seed: int = 0, shuffle_repeats: int = 5,
) -> dict[str, Any]:
    """Evaluate all original labels under correct/zero/within-group shuffled residuals.

    Only residual vectors and window identities are cached; contexts and targets
    are re-read in bounded batches. No optimizer or normalizer fit is created.
    """
    batch_size = _integer(batch_size, "batch_size")
    seed = _integer(shuffle_seed, "shuffle_seed")
    repeats = _integer(shuffle_repeats, "shuffle_repeats")
    if batch_size < 1 or repeats < 1:
        raise ValueError("batch_size and shuffle_repeats must be positive")
    captures = _captures(loaded, validation_dirs)
    model = loaded.model
    correct = [evaluate_response_model(model, capture.batches(batch_size, mode="correct")) for capture in captures]
    zero = [evaluate_response_model(model, capture.batches(batch_size, mode="zero")) for capture in captures]
    rng = np.random.default_rng(seed)
    trial_results: list[list[dict[str, Any]]] = [[] for _ in captures]
    aggregate_trials = []
    for trial in range(repeats):
        aggregate_hash = hashlib.sha256()
        unchanged_count = 0
        metrics = []
        for index, capture in enumerate(captures):
            permutation = _derangement(capture, rng)
            permutation_hash = hashlib.sha256(permutation.astype("<i8", copy=False).tobytes()).hexdigest()
            aggregate_hash.update(permutation.astype("<i8", copy=False).tobytes())
            unchanged = int((capture.residuals[torch.from_numpy(permutation)] == capture.residuals).all(dim=1).sum())
            unchanged_count += unchanged
            result = evaluate_response_model(model, capture.batches(batch_size, mode="shuffled", permutation=permutation))
            metrics.append(result)
            trial_results[index].append({
                "trial": trial, "permutation_sha256": permutation_hash,
                "unchanged_vector_fraction": unchanged / len(capture.identities), "metrics": result,
            })
        aggregate_trials.append({
            "trial": trial, "permutation_sha256": aggregate_hash.hexdigest(),
            "unchanged_vector_fraction": unchanged_count / sum(len(capture.identities) for capture in captures),
            "metrics": _aggregate(metrics),
        })
    capture_results = []
    for index, capture in enumerate(captures):
        if read_path_following_metadata(capture.path) != capture.metadata or _digest(capture.path, capture.metadata) != capture.fingerprint:
            raise ValueError("validation capture changed during diagnosis; no report is produced")
        capture_results.append({
            "path": str(capture.path), "capture_sha256": capture.fingerprint, "samples": len(capture.identities),
            "recorded_validation": any(source.get("split") == "validation" and source.get("capture_sha256") == capture.fingerprint for source in loaded.metadata["data_sources"]),
            "source": capture.metadata["source"],
            "groups": [{"sequence_id": identity[0], "tracking_epoch": identity[1], "frame_id": identity[2], "windows": len(indices)} for identity, indices in capture.groups.items()],
            "correct": correct[index], "zero": {**zero[index], "delta_rmse_m_m_rad": _difference(zero[index], correct[index])},
            "shuffled": _shuffle_summary(trial_results[index], correct[index]),
        })
    aggregate_correct, aggregate_zero = _aggregate(correct), _aggregate(zero)
    return {
        "format": "sttt_response_residual_diagnostics", "schema_version": 1,
        "evidence": "held_out_predictive_input_ablation_only", "physical_direction_verified": False,
        "interpretation": "Positive delta RMSE means predictive benefit on this dataset; no causal, physical-sign or cross-condition verdict. Shuffle repetitions are not independent experiments.",
        "comparison": "ablation_rmse_minus_correct_rmse", "channels": ["dx", "dy", "dyaw"], "units": ["m", "m", "rad"],
        "shuffle_algorithm": "sattolo_cycle_per_capture_sequence_epoch_frame_v1", "shuffle_seed": seed, "shuffle_repeats": repeats,
        "batch_size": batch_size, "response_contract": loaded.contract,
        "zero_inside_training_residual_bounds": bool(((model.residual_min <= 0) & (model.residual_max >= 0)).all()),
        "correct": aggregate_correct, "zero": {**aggregate_zero, "delta_rmse_m_m_rad": _difference(aggregate_zero, aggregate_correct)},
        "shuffled": _shuffle_summary(aggregate_trials, aggregate_correct), "captures": capture_results,
    }


def run_residual_input_diagnostics(cfg: DictConfig) -> dict[str, Any]:
    if not isinstance(cfg, DictConfig) or set(cfg) != {"checkpoint_path", "validation_dirs", "batch_size", "shuffle_seed", "shuffle_repeats", "device", "output_path"}:
        raise ValueError("use conf/sttt/response_residual_diagnostics.yaml")
    if not cfg.checkpoint_path:
        raise ValueError("checkpoint_path must name the existing independent B artifact")
    checkpoint = Path(str(cfg.checkpoint_path)).expanduser().resolve()
    output = Path(str(cfg.output_path)).expanduser().resolve() if cfg.output_path else checkpoint.with_name("residual_diagnostics.json")
    if output.exists():
        raise FileExistsError(f"diagnostic output already exists: {output}")
    with torch.random.fork_rng(devices=[]):
        loaded = load_response_checkpoint(checkpoint, device=str(cfg.device))
    report = compare_response_residuals(
        loaded, list(cfg.validation_dirs), batch_size=cfg.batch_size,
        shuffle_seed=cfg.shuffle_seed, shuffle_repeats=cfg.shuffle_repeats,
    )
    with open(checkpoint, "rb") as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    report["checkpoint"] = {"path": str(checkpoint), "sha256": digest.hexdigest()}
    report["output_path"] = str(output)
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: never replace an earlier diagnostic or trained model.
    with output.open("x") as handle:
        handle.write(serialized)
    return report
