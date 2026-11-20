"""Step-budget pretraining on causal history and bounded episode-local forecasts."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from omegaconf import DictConfig, OmegaConf

from .checkpoint import canonical_contract, load_world_model, save_training_checkpoint
from .losses import pretraining_loss
from .model import LatentWorldModel, WorldModelConfig
from .pretraining_data import PretrainingSequenceSampler
from .sequence_data import read_sequence_metadata

_TRAINING_DEFAULTS = {
    "max_updates": 200000, "batch_size": 16, "rollout_windows": 2, "cache_size": 2,
    "validation_every": 1000, "validation_batches": 16, "checkpoint_every": 1000,
    "learning_rate": 3e-4, "weight_decay": 1e-6, "gradient_clip_norm": 100.,
    "resume_checkpoint": None, "loss": {},
}
_LOSS_KEYS = {"free_nats", "kl_scale", "reconstruction_scale", "motion_scale", "risk_scale", "overshoot_scale", "kl_balance", "prior_reconstruction_scale", "window_scale"}


def _positive_integer(value: Any, name: str, *, zero: bool = False) -> int:
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if zero else 'positive'} integer")
    return value


def _finite(value: Any, name: str, *, zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or not zero and value == 0:
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return float(value)


def _resolve(path: str | Path, root: Path) -> Path:
    expanded = Path(path).expanduser()
    return (expanded if expanded.is_absolute() else root / expanded).resolve()


def _directories(raw: Any, root: Path) -> list[Path]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or not raw:
        raise ValueError("train_dirs and validation_dirs must each list complete capture directories")
    paths = [_resolve(path, root) for path in raw]
    if len(paths) != len(set(paths)) or any(not path.is_dir() for path in paths):
        raise ValueError("dataset directories must exist and may not repeat")
    return paths


def _data_sources(paths: list[Path], split: str, shard_index: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], set[str]]:
    provenance, fingerprints = [], set()
    shard_hashes = {entry["path"]: entry["sha256"] for entry in shard_index}
    for path in paths:
        manifest = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        names = manifest.get("sequences", [])
        if not names:
            raise ValueError(f"empty world-model dataset: {path}")
        capture_digest = hashlib.sha256()
        for name in names:
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError("sequence manifest contains an unsafe shard path")
            fingerprint = shard_hashes[str((path / name).resolve())]
            fingerprints.add(fingerprint)
            capture_digest.update(fingerprint.encode("ascii"))
        identity_path = path / "capture.json"
        identity_digest = None
        if identity_path.exists():
            identity_bytes = identity_path.read_bytes()
            identity_digest = hashlib.sha256(identity_bytes).hexdigest()
            identity = json.loads(identity_bytes)
            if not isinstance(identity, dict):
                raise ValueError("capture identity sidecar must be a mapping")
            capture_id, episode_ids = identity.get("capture_id"), identity.get("episode_ids", [])
            if capture_id is not None:
                if not isinstance(capture_id, str) or not capture_id:
                    raise ValueError("capture_id must be a nonempty string")
                fingerprints.add("capture:" + capture_id)
            if not isinstance(episode_ids, list) or any(not isinstance(item, str) or not item for item in episode_ids):
                raise ValueError("episode_ids must be a list of nonempty strings")
            fingerprints.update("episode:" + item for item in episode_ids)
        provenance.append({"split": split, "path": str(path), "capture_sha256": capture_digest.hexdigest(),
                           "capture_identity_sha256": identity_digest, "sequences": len(names)})
    return provenance, fingerprints


def _rng_state(device: torch.device) -> dict[str, Any]:
    state = {"rng_state": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    elif device.type == "mps":
        state["mps_rng_state"] = torch.mps.get_rng_state()
    return state


def _restore_rng(state: Mapping[str, Any], device: torch.device) -> None:
    torch.set_rng_state(state["rng_state"])
    if device.type == "cuda":
        if len(state["cuda_rng_state"]) != torch.cuda.device_count():
            raise ValueError("exact resume requires the same CUDA generator count")
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    elif device.type == "mps":
        torch.mps.set_rng_state(state["mps_rng_state"])


def _validate(model: LatentWorldModel, sampler: PretrainingSequenceSampler, indices: list[int], *,
              batch_size: int, device: torch.device, seed: int, loss_config: dict[str, float]) -> dict[str, float]:
    random_state, was_training = _rng_state(device), model.training
    totals: dict[str, float] = {}
    xy_sum, windows, rows = 0., 0, 0
    try:
        torch.manual_seed(seed)
        model.eval()
        with torch.no_grad():
            for offset in range(0, len(indices), batch_size):
                selected = indices[offset:offset + batch_size]
                batch = sampler.sample_indices(selected, device=device)
                losses = pretraining_loss(model, batch, **loss_config)
                if any(not bool(torch.isfinite(value)) for value in losses.values()):
                    raise ValueError("non-finite validation world-model loss")
                xy_sum += float(losses["window_xy_squared_error_sum"])
                windows += int(losses["window_xy_count"])
                for key, value in losses.items():
                    if key not in {"window_xy_squared_error_sum", "window_xy_count"}:
                        totals[key] = totals.get(key, 0.) + float(value) * len(selected)
                rows += len(selected)
    finally:
        _restore_rng(random_state, device)
        model.train(was_training)
    if not windows:
        raise ValueError("validation crops require valid physical window motion labels and known headings")
    return {**{key: value / rows for key, value in totals.items()},
            "window_xy_rmse_m": math.sqrt(xy_sum / windows), "window_count": windows}


def train_world_model(cfg: DictConfig | Mapping[str, Any], *, root_dir: str | Path | None = None) -> dict[str, Any]:
    """Fit logged actions to their physical outcomes; resume exact update state."""
    raw = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True) if isinstance(cfg, DictConfig) else dict(cfg)
    if not isinstance(raw, dict) or set(raw) - {"data", "model", "training", "device", "seed", "output_dir"}:
        raise ValueError("unknown world-model training config fields")
    if "data" not in raw or not raw.get("output_dir"):
        raise ValueError("explicit data split and output_dir are required")
    data = raw["data"]
    if not isinstance(data, Mapping) or set(data) != {"train_dirs", "validation_dirs"}:
        raise ValueError("data requires explicit train_dirs and validation_dirs only")
    training_input = raw.get("training", {})
    if not isinstance(training_input, Mapping) or set(training_input) - set(_TRAINING_DEFAULTS):
        raise ValueError("unknown world-model training options; use max_updates, not epochs")
    training = {**_TRAINING_DEFAULTS, **training_input}
    integers = {key: _positive_integer(training[key], key) for key in (
        "max_updates", "batch_size", "rollout_windows", "cache_size",
        "validation_every", "validation_batches", "checkpoint_every",
    )}
    max_updates, batch_size = integers["max_updates"], integers["batch_size"]
    seed = _positive_integer(raw.get("seed", 0), "seed", zero=True)
    learning_rate = _finite(training["learning_rate"], "learning_rate")
    weight_decay = _finite(training["weight_decay"], "weight_decay", zero=True)
    gradient_clip = _finite(training["gradient_clip_norm"], "gradient_clip_norm")
    loss_config = training["loss"]
    if not isinstance(loss_config, Mapping) or set(loss_config) - _LOSS_KEYS:
        raise ValueError("unknown world-model loss configuration")
    loss_config = {key: _finite(value, key, zero=True) for key, value in loss_config.items()}
    if loss_config.get("kl_balance", .8) > 1:
        raise ValueError("kl_balance must lie in [0,1]")
    training["loss"] = loss_config
    root = Path.cwd() if root_dir is None else Path(root_dir).expanduser().resolve()
    output = _resolve(raw["output_dir"], root)
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(output)
    train_dirs, validation_dirs = _directories(data["train_dirs"], root), _directories(data["validation_dirs"], root)
    if set(train_dirs) & set(validation_dirs) or any(a in b.parents or b in a.parents for a in train_dirs for b in validation_dirs):
        raise ValueError("training and validation must be disjoint whole-capture directories")
    contracts = [canonical_contract(read_sequence_metadata(path)) for path in (*train_dirs, *validation_dirs)]
    contract = contracts[0]
    if any(candidate != contract for candidate in contracts[1:]):
        raise ValueError("training/validation capture source, normalization or physical contracts differ")
    sampler_options = {"rollout_windows": integers["rollout_windows"], "cache_size": integers["cache_size"]}
    sampler = PretrainingSequenceSampler(train_dirs, seed=seed, **sampler_options)
    validation_sampler = PretrainingSequenceSampler(validation_dirs, seed=seed, **sampler_options)
    train_sources, train_hashes = _data_sources(train_dirs, "train", sampler.state_dict()["index"])
    validation_sources, validation_hashes = _data_sources(validation_dirs, "validation", validation_sampler.state_dict()["index"])
    if train_hashes & validation_hashes:
        raise ValueError("training/validation contain copied sequence shards from the same capture")
    sources = train_sources + validation_sources
    validation_seed = seed ^ 0x5DEECE66D
    validation_indices = torch.randint(
        validation_sampler.num_starts, (batch_size * integers["validation_batches"],),
        generator=torch.Generator().manual_seed(validation_seed),
    ).tolist()
    config = WorldModelConfig.from_mapping(raw.get("model", {}))
    device = torch.device(raw.get("device", "cpu"))
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("exact pretraining resume supports CPU, CUDA or MPS")
    signature = {"seed": seed, "device": str(device), **{
        key: value for key, value in training.items() if key not in {"max_updates", "resume_checkpoint"}
    }}
    torch.manual_seed(seed)
    resumed_state: dict[str, Any] = {}
    if training["resume_checkpoint"] is None:
        model = LatentWorldModel(config).to(device)
    else:
        loaded = load_world_model(_resolve(training["resume_checkpoint"], root), device=device, expected_contract=contract)
        if loaded.model.config != config:
            raise ValueError("resume model configuration disagrees with requested architecture/scaling")
        model, resumed_state = loaded.model, loaded.training_state
        required = {"run_id", "optimizer_state", "optimizer_steps", "rng_state", "history",
                    "sampler_state", "validation_sampler_state", "validation_indices",
                    "data_sources", "training_signature", "best_state_dict", "best_metric", "best_step"}
        if resumed_state.get("trainer_version") != 2 or not required <= resumed_state.keys():
            raise ValueError("exact resume requires version-2 update-budget training state")
        if resumed_state["training_signature"] != signature or resumed_state["data_sources"] != sources:
            raise ValueError("resume training configuration or data identity changed")
        if resumed_state["validation_indices"] != validation_indices:
            raise ValueError("resume validation crop identity changed")
        sampler.load_state_dict(resumed_state["sampler_state"])
        validation_sampler.load_state_dict(resumed_state["validation_sampler_state"])
    run_id = resumed_state.get("run_id", uuid4().hex)
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("training run_id must be a nonempty string")
    last_path, best_path = output / "last.pt", output / "best.pt"
    existing_best_step = None
    for path in (last_path, best_path):
        if path.exists():
            if not resumed_state:
                raise FileExistsError(f"new training output already contains a checkpoint: {path}")
            existing = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(existing, dict) or existing.get("format") != "sttt_latent_world_model" or existing.get("training_state", {}).get("run_id") != run_id:
                raise FileExistsError(f"refusing to reuse another run's output: {path}")
            if path == best_path:
                existing_best_step = existing["training_state"].get("optimizer_steps")
            del existing
    optimizer_steps = _positive_integer(resumed_state.get("optimizer_steps", 0), "optimizer_steps", zero=True)
    if optimizer_steps >= max_updates:
        raise ValueError("training.max_updates must exceed the completed resume update count")
    history = list(resumed_state.get("history", []))
    best_metric, best_step = resumed_state.get("best_metric"), resumed_state.get("best_step")
    best_state = resumed_state.get("best_state_dict")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if resumed_state:
        optimizer.load_state_dict(resumed_state["optimizer_state"])
        _restore_rng(resumed_state, device)

    def export_best(physical_model: LatentWorldModel) -> None:
        save_training_checkpoint(best_path, physical_model, contract, run_id=run_id, training_state={
            "trainer_version": 2, "run_id": run_id, "optimizer_steps": best_step,
            "selection_metric": "window_xy_rmse_m", "best_metric": best_metric,
        })

    if best_state is not None and existing_best_step != best_step:
        random_state = _rng_state(device)
        try:
            saved_best = LatentWorldModel(config).to(dtype=next(model.parameters()).dtype)
            saved_best.load_state_dict(best_state, strict=True)
            export_best(saved_best)
        finally:
            _restore_rng(random_state, device)
        del saved_best

    while optimizer_steps < max_updates:
        model.train()
        batch = sampler.sample(batch_size, device=device)
        optimizer.zero_grad(set_to_none=True)
        losses = pretraining_loss(model, batch, **loss_config)
        if any(not bool(torch.isfinite(value)) for value in losses.values()):
            raise ValueError("non-finite world-model training loss")
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        optimizer_steps += 1
        final = optimizer_steps == max_updates
        improved = False
        if optimizer_steps % integers["validation_every"] == 0 or final:
            validation = _validate(model, validation_sampler, validation_indices, batch_size=batch_size,
                                   device=device, seed=validation_seed, loss_config=loss_config)
            train_metrics = {key: float(value.detach()) for key, value in losses.items()
                             if key not in {"window_xy_squared_error_sum", "window_xy_count"}}
            history.append({"optimizer_steps": optimizer_steps, "train": train_metrics, "validation": validation})
            metric = validation["window_xy_rmse_m"]
            if best_metric is None or metric < best_metric:
                best_metric, best_step = metric, optimizer_steps
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                improved = True
            print(json.dumps({"optimizer_steps": optimizer_steps, "validation": validation, "best_window_xy_rmse_m": best_metric}), flush=True)
        if optimizer_steps % integers["checkpoint_every"] == 0 or final:
            optimizer.zero_grad(set_to_none=True)
            state = {
                "trainer_version": 2, "run_id": run_id, "optimizer_steps": optimizer_steps,
                "optimizer_state": optimizer.state_dict(), "history": history,
                "sampler_state": sampler.state_dict(), "validation_sampler_state": validation_sampler.state_dict(),
                "validation_indices": validation_indices, "data_sources": sources,
                "training_signature": signature, "best_metric": best_metric, "best_step": best_step,
                "best_state_dict": best_state, **_rng_state(device),
            }
            save_training_checkpoint(last_path, model, contract, run_id=run_id, training_state=state)
        if improved:
            export_best(model)
    model.eval()
    return {"checkpoint_path": str(last_path), "best_checkpoint_path": str(best_path),
            "history": history, "optimizer_steps": optimizer_steps, "best_step": best_step,
            "best_window_xy_rmse_m": best_metric, "validation": history[-1]["validation"], "contract": contract}
