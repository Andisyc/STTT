"""Offline B-only training over whole held-out capture groups and bounded buffers."""
from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.sttt.path_following.checkpoint import save_response_checkpoint
from unilab.algos.torch.sttt.path_following.contract import _integer
from unilab.algos.torch.sttt.path_following.dataset import iter_path_following_samples, read_path_following_metadata
from unilab.algos.torch.sttt.path_following.diagnostics import diagnose_response_jacobian, evaluate_response_model
from unilab.algos.torch.sttt.path_following.features import pack_response_context, response_contract_from_metadata
from unilab.algos.torch.sttt.path_following.model import ResponseBatch, ResponseModel, response_loss


@dataclass(frozen=True)
class ResponseTrainingResult:
    model: ResponseModel
    contract: dict[str, Any]
    checkpoint_path: Path
    metrics: dict[str, Any]


@dataclass(frozen=True)
class _CaptureGroup:
    path: Path
    metadata: dict[str, Any]
    contract: dict[str, Any]
    fingerprint: str

    def provenance(self) -> dict[str, Any]:
        return {
            "path": str(self.path), "capture_sha256": self.fingerprint,
            "samples": self.metadata["sample_count"], "source": self.metadata["source"],
        }


def _capture_groups(paths: Sequence[str]) -> list[_CaptureGroup]:
    if isinstance(paths, (str, bytes)) or not isinstance(paths, Sequence) or not paths:
        raise ValueError("train_dirs and validation_dirs must each contain complete capture directories")
    groups = []
    seen = set()
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if path in seen:
            raise ValueError("duplicate capture directory")
        seen.add(path)
        metadata = read_path_following_metadata(path)
        if metadata["source"].get("label_source") != "simulation_ground_truth":
            raise ValueError("offline B pretraining requires simulation-ground-truth response windows")
        contract = response_contract_from_metadata(metadata)
        digest = hashlib.sha256()
        for shard in metadata["shards"]:
            with (path / shard["file"]).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
        groups.append(_CaptureGroup(path, metadata, contract, digest.hexdigest()))
    return groups


def _batches(
    groups: Sequence[_CaptureGroup], *, batch_size: int, buffer_size: int,
    generator: torch.Generator | None = None,
) -> Iterator[ResponseBatch]:
    """Validate and pack while loading, outside the tensor-only model path."""
    order = list(range(len(groups))) if generator is None else torch.randperm(len(groups), generator=generator).tolist()
    rows: list[ResponseBatch] = []

    def packed() -> Iterator[ResponseBatch]:
        context = torch.cat([row[0] for row in rows])
        residual = torch.cat([row[1] for row in rows])
        target = torch.cat([row[2] for row in rows])
        if generator is not None:
            indices = torch.randperm(len(rows), generator=generator)
            context, residual, target = context[indices], residual[indices], target[indices]
        for start in range(0, len(rows), batch_size):
            yield context[start:start + batch_size], residual[start:start + batch_size], target[start:start + batch_size]

    for index in order:
        for sample in iter_path_following_samples(groups[index].path):
            rows.append((
                pack_response_context(sample.start_query).float(), sample.start_query.residual_rad.float(),
                torch.from_numpy(sample.delta).float().unsqueeze(0),
            ))
            if len(rows) >= buffer_size:
                yield from packed()
                rows.clear()
    if rows:
        yield from packed()


def _finite(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return float(value)


def train_response_model(
    cfg: DictConfig | Mapping[str, Any], *, log: Callable[[str], None] = print,
) -> ResponseTrainingResult:
    """No env/policy is created. Existing B artifacts are never overwritten."""
    resolved = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True) if isinstance(cfg, DictConfig) else dict(cfg)
    if not isinstance(resolved, dict):
        raise TypeError("response training config must be a mapping")
    config = cast(dict[str, Any], resolved)
    required = {"data", "model", "training", "seed", "device", "output_checkpoint", "diagnostics"}
    if set(config) != required:
        raise ValueError("use the independent conf/sttt/response_model.yaml configuration")
    training, model_config, diagnostic = config["training"], config["model"], config["diagnostics"]
    batch_size = _integer(training["batch_size"], "batch_size")
    buffer_size = _integer(training["shuffle_buffer_size"], "shuffle_buffer_size")
    epochs = _integer(training["epochs"], "epochs")
    seed = _integer(config["seed"], "seed")
    probe_count = _integer(diagnostic["jacobian_samples"], "jacobian_samples")
    if batch_size < 1 or buffer_size < batch_size or epochs < 1 or probe_count < 1:
        raise ValueError("positive batch/epoch/probe counts and buffer_size >= batch_size are required")
    lr = _finite(training["learning_rate"], "learning_rate")
    decay = _finite(training["weight_decay"], "weight_decay", allow_zero=True)
    epsilon = _finite(model_config["normalization_epsilon"], "normalization_epsilon")
    fd_epsilon = _finite(diagnostic["finite_difference_epsilon_rad"], "finite_difference_epsilon_rad")
    if not config["output_checkpoint"]:
        raise ValueError("an explicit output_checkpoint path is required")
    output = Path(config["output_checkpoint"]).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    train = _capture_groups(config["data"]["train_dirs"])
    validation = _capture_groups(config["data"]["validation_dirs"])
    if {group.path for group in train} & {group.path for group in validation} or {group.fingerprint for group in train} & {group.fingerprint for group in validation}:
        raise ValueError("training and validation must use disjoint captures, not copies of the same data")
    contract = train[0].contract
    if any(group.contract != contract for group in (*train, *validation)):
        raise ValueError("capture groups have incompatible window, source-policy or context contracts")
    data = contract["data"]
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        model = ResponseModel(data["history_length"], data["prediction_horizon"], model_config["hidden_dims"])
    model.fit_normalization(_batches(train, batch_size=batch_size, buffer_size=batch_size), epsilon=epsilon)
    model.to(torch.device(config["device"]))
    reference = next(model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
    generator = torch.Generator().manual_seed(seed)
    history = []
    optimizer_steps = 0
    for epoch in range(epochs):
        model.train()
        total, samples = 0., 0
        for context, residual, target in _batches(train, batch_size=batch_size, buffer_size=buffer_size, generator=generator):
            optimizer.zero_grad(set_to_none=True)
            loss = response_loss(model(context.to(reference), residual.to(reference)), target.to(reference))
            value = float(loss.detach())
            if not math.isfinite(value):
                raise ValueError("non-finite B loss; no model artifact is produced")
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            total += value * context.shape[0]
            samples += context.shape[0]
        evaluation = evaluate_response_model(model, _batches(validation, batch_size=batch_size, buffer_size=batch_size))
        history.append({"epoch": epoch + 1, "train_loss": total / samples, "validation": evaluation})
        log(f"B epoch={epoch + 1} train_loss={total / samples:.6g} validation_loss={evaluation['loss']:.6g}")
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    jacobians = []
    for context, residual, _target in _batches(validation, batch_size=probe_count, buffer_size=probe_count):
        for index in range(context.shape[0]):
            jacobians.append(diagnose_response_jacobian(
                model, context[index:index + 1], residual[index:index + 1], epsilon_rad=fd_epsilon,
            ))
        break
    metrics = {
        "history": history, "validation": history[-1]["validation"], "optimizer_steps": optimizer_steps,
        "normalization_samples": int(model.normalization_samples),
        "residual_min_rad": model.residual_min.cpu().tolist(), "residual_max_rad": model.residual_max.cpu().tolist(),
        "jacobians": jacobians, "physical_direction_verified": False,
    }
    provenance = [{"split": split, **group.provenance()} for split, groups in (("train", train), ("validation", validation)) for group in groups]
    saved = save_response_checkpoint(
        output, model, contract=contract, training_config=config, data_sources=provenance, metrics=metrics,
    )
    return ResponseTrainingResult(model, contract, saved, metrics)
