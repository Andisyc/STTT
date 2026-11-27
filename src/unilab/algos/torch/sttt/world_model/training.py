"""Step-budget pretraining on causal history and bounded episode-local forecasts."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import torch
from omegaconf import DictConfig, OmegaConf

from .checkpoint import canonical_contract, load_world_model, save_training_checkpoint
from .learner import WorldModelLearner
from .losses import PREDICTION_STATISTICS, pretraining_loss
from .model import LatentWorldModel, WorldModelConfig
from .pretraining_data import PretrainingSequenceSampler
from .sequence_data import read_sequence_metadata

_TRAINING_DEFAULTS = {
    "max_updates": 200000, "batch_size": 16, "rollout_windows": 2, "cache_size": 2,
    "validation_every": 1000, "validation_batches": 16, "checkpoint_every": 1000,
    "learning_rate": 3e-4, "weight_decay": 1e-6, "gradient_clip_norm": 10.,
    "motion_learning_rate": 3e-5, "motion_gradient_clip_norm": 1., "warmup_updates": 1000,
    "resume_checkpoint": None, "initial_checkpoint": None, "loss": {},
    "microbatch_size": None,
}
_LOSS_KEYS = {"free_nats", "kl_scale", "reconstruction_scale", "motion_scale", "risk_scale", "overshoot_scale", "kl_balance", "prior_reconstruction_scale", "window_scale",
              "motion_velocity_tolerance", "motion_yaw_rate_tolerance", "window_xy_tolerance", "window_yaw_tolerance"}


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


def _validate(model: LatentWorldModel, sampler, indices: list[int], *,
              batch_size: int, device: torch.device, seed: int, loss_config: dict[str, float],
              require_windows: bool = True) -> dict[str, Any]:
    random_state, was_training = _rng_state(device), model.training
    totals: dict[str, float] = {}
    statistics = {key: 0. for key in PREDICTION_STATISTICS}
    rows = 0
    try:
        torch.manual_seed(seed)
        model.eval()
        with torch.no_grad():
            for offset in range(0, len(indices), batch_size):
                selected = indices[offset:offset + batch_size]
                losses = pretraining_loss(model, sampler.sample_indices(selected, device=device), **loss_config)
                if any(not bool(torch.isfinite(value)) for value in losses.values()):
                    raise ValueError("non-finite validation world-model loss")
                for key, value in losses.items():
                    if key in statistics:
                        statistics[key] += float(value)
                    else:
                        totals[key] = totals.get(key, 0.) + float(value) * len(selected)
                rows += len(selected)
    finally:
        _restore_rng(random_state, device)
        model.train(was_training)
    windows = int(statistics["window_xy_count"])
    if not windows and require_windows:
        raise ValueError("validation crops require valid physical window motion labels")
    result = {**{key: value / rows for key, value in totals.items()},
              "window_xy_rmse_m": math.sqrt(statistics["window_xy_squared_error_sum"] / windows) if windows else None,
              "window_yaw_rmse_rad": math.sqrt(statistics["window_yaw_squared_error_sum"] / windows) if windows else None,
              "window_yaw_bias_rad": statistics["window_yaw_error_sum"] / windows if windows else None,
              "window_count": windows}
    motion_count = statistics["motion_count"]
    result["yaw_rate_bias_rad_s"] = statistics["motion_yaw_rate_error_sum"] / motion_count if motion_count else None
    result["yaw_rate_rmse_rad_s"] = math.sqrt(statistics["motion_yaw_rate_squared_error_sum"] / motion_count) if motion_count else None
    for name, unit in (("joint_position", "rad"), ("joint_velocity", "rad_s"), ("gyro", "rad_s"), ("gravity", "")):
        result[name + "_rmse" + ("_" + unit if unit else "")] = math.sqrt(
            statistics[name + "_squared_error_sum"] / max(1, statistics["proprio_count"]))
    positives, low, missed = (int(statistics[key]) for key in ("risk_positive_count", "risk_low_count", "risk_missed_count"))
    result.update(risk_positive_count=positives, risk_label_count=int(statistics["risk_label_count"]),
                  risk_false_negative_rate=missed / positives if positives else None,
                  low_risk_event_fraction=missed / low if low else None,
                  risk_validation_available=positives > 0)
    result["selection_score"] = max(
        result["window_xy_rmse_m"] / loss_config.get("window_xy_tolerance", .01),
        result["window_yaw_rmse_rad"] / loss_config.get("window_yaw_tolerance", .02),
        result["joint_position_rmse_rad"] / .01, result["joint_velocity_rmse_rad_s"],
        result["gyro_rmse_rad_s"] / .1, result["gravity_rmse"] / .05) if windows else None
    return result


def train_world_model(cfg: DictConfig | Mapping[str, Any], *, root_dir: str | Path | None = None) -> dict[str, Any]:
    """Fit logged actions to their physical outcomes; resume exact update state."""
    started_at = perf_counter()
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
    print(json.dumps({"phase": "wm_index_start", "train_captures": len(train_dirs),
                      "validation_captures": len(validation_dirs), "batch_size": batch_size,
                      "max_updates": max_updates, "device": raw.get("device", "cpu")}), flush=True)
    index_started_at = perf_counter()
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
    print(json.dumps({"phase": "wm_index_ready", "seconds": perf_counter() - index_started_at,
                      "train_starts": sampler.num_starts,
                      "validation_starts": validation_sampler.num_starts}), flush=True)
    validation_seed = seed ^ 0x5DEECE66D
    validation_indices = torch.randint(
        validation_sampler.num_starts, (batch_size * integers["validation_batches"],),
        generator=torch.Generator().manual_seed(validation_seed),
    ).tolist()
    model_options = dict(raw.get("model", {}))
    if model_options.get("obs_dim") is None:
        model_options["obs_dim"] = contract["data"]["observation_dim"]
    if model_options.get("ctrl_dt") is None:
        model_options["ctrl_dt"] = contract["data"]["ctrl_dt"]
    config = WorldModelConfig.from_mapping(model_options)
    if config.obs_dim != contract["data"]["observation_dim"] or config.action_dim != contract["data"]["action_dim"]:
        raise ValueError("training model dimensions disagree with the captured policy contract")
    if not math.isclose(config.ctrl_dt, contract["data"]["ctrl_dt"], rel_tol=0., abs_tol=1e-9):
        raise ValueError("training model control period differs from the data")
    device = torch.device(raw.get("device", "cpu"))
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("exact pretraining resume supports CPU, CUDA or MPS")
    signature = {"seed": seed, "device": str(device), **{
        key: value for key, value in training.items()
        if key not in {"max_updates", "resume_checkpoint"} and not (key == "microbatch_size" and value is None)
    }}
    torch.manual_seed(seed)
    resumed_state: dict[str, Any] = {}
    if training["resume_checkpoint"] is not None and training["initial_checkpoint"] is not None:
        raise ValueError("select exact resume or WM warm start, not both")
    if training["resume_checkpoint"] is None:
        if training["initial_checkpoint"] is None:
            model = LatentWorldModel(config).to(device)
        else:
            loaded = load_world_model(_resolve(training["initial_checkpoint"], root), device=device, expected_contract=contract)
            if loaded.model.config != config:
                raise ValueError("WM warm-start architecture/scaling differs")
            model = loaded.model
    else:
        loaded = load_world_model(_resolve(training["resume_checkpoint"], root), device=device, expected_contract=contract)
        if loaded.model.config != config:
            raise ValueError("resume model configuration disagrees with requested architecture/scaling")
        model, resumed_state = loaded.model, loaded.training_state
        required = {"run_id", "optimizer_state", "optimizer_steps", "rng_state", "history",
                    "sampler_state", "validation_sampler_state", "validation_indices",
                    "data_sources", "training_signature", "best_state_dict", "best_metric", "best_step",
                    "optimizer_started_at"}
        if resumed_state.get("trainer_version") != 4 or not required <= resumed_state.keys():
            raise ValueError("exact resume requires version-4 velocity-readout training state")
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
    model.configure_observation_loss(contract["source"]["preprocessing"])
    learner = WorldModelLearner(model, {**training, "learning_rate": learning_rate, "weight_decay": weight_decay,
                                       "gradient_clip_norm": gradient_clip}, loss_config, updates=optimizer_steps)
    optimizer = learner.optimizer
    if resumed_state:
        optimizer.load_state_dict(resumed_state["optimizer_state"])
        origin = resumed_state["optimizer_started_at"]
        if type(origin) is not int or not 0 <= origin <= optimizer_steps:
            raise ValueError("invalid restored WM optimizer warmup origin")
        learner.optimizer_started_at = origin
        _restore_rng(resumed_state, device)

    def export_best(physical_model: LatentWorldModel) -> None:
        save_training_checkpoint(best_path, physical_model, contract, run_id=run_id, training_state={
            "trainer_version": 4, "run_id": run_id, "optimizer_steps": best_step,
            "selection_metric": "physical_prediction_score", "best_metric": best_metric, "loss_config": loss_config,
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

    first_update = optimizer_steps + 1
    print(json.dumps({"phase": "wm_train_start", "optimizer_steps": optimizer_steps,
                      "max_updates": max_updates, "microbatch_size": learner.microbatch_size,
                      "startup_seconds": perf_counter() - started_at}), flush=True)
    while optimizer_steps < max_updates:
        step_started_at = perf_counter()
        if optimizer_steps + 1 == first_update:
            print(json.dumps({"phase": "wm_sample_start", "optimizer_steps": optimizer_steps}), flush=True)
        batch = sampler.sample(batch_size, device=device)
        sampled_at = perf_counter()
        if optimizer_steps + 1 == first_update:
            print(json.dumps({"phase": "wm_batch_ready", "sample_seconds": sampled_at - step_started_at}), flush=True)
        losses = learner.update(batch)
        updated_at = perf_counter()
        optimizer_steps = learner.updates
        final = optimizer_steps == max_updates
        improved = False
        if optimizer_steps == first_update or optimizer_steps % 100 == 0 or final:
            print(json.dumps({"phase": "wm_train", "optimizer_steps": optimizer_steps,
                              "max_updates": max_updates, "train": losses,
                              "sample_seconds": sampled_at - step_started_at,
                              "update_seconds": updated_at - sampled_at,
                              "elapsed_seconds": updated_at - started_at}), flush=True)
        if optimizer_steps % integers["validation_every"] == 0 or final:
            print(json.dumps({"phase": "wm_validation_start", "optimizer_steps": optimizer_steps,
                              "samples": len(validation_indices)}), flush=True)
            validation_started_at = perf_counter()
            validation = _validate(model, validation_sampler, validation_indices, batch_size=batch_size,
                                   device=device, seed=validation_seed, loss_config=loss_config)
            train_metrics = {key: value for key, value in losses.items()
                             if key not in {"window_xy_squared_error_sum", "window_xy_count"}}
            history.append({"optimizer_steps": optimizer_steps, "train": train_metrics, "validation": validation})
            metric = validation["selection_score"]
            if best_metric is None or metric < best_metric:
                best_metric, best_step = metric, optimizer_steps
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                improved = True
            print(json.dumps({"phase": "wm_validation", "optimizer_steps": optimizer_steps,
                              "validation": validation, "best_prediction_score": best_metric,
                              "seconds": perf_counter() - validation_started_at}), flush=True)
        if optimizer_steps % integers["checkpoint_every"] == 0 or final:
            print(json.dumps({"phase": "wm_checkpoint_start", "optimizer_steps": optimizer_steps}), flush=True)
            checkpoint_started_at = perf_counter()
            optimizer.zero_grad(set_to_none=True)
            state = {
                "trainer_version": 4, "run_id": run_id, "optimizer_steps": optimizer_steps, "loss_config": loss_config,
                "optimizer_started_at": learner.optimizer_started_at,
                "optimizer_state": optimizer.state_dict(), "history": history,
                "sampler_state": sampler.state_dict(), "validation_sampler_state": validation_sampler.state_dict(),
                "validation_indices": validation_indices, "data_sources": sources,
                "training_signature": signature, "best_metric": best_metric, "best_step": best_step,
                "best_state_dict": best_state, **_rng_state(device),
            }
            save_training_checkpoint(last_path, model, contract, run_id=run_id, training_state=state)
            print(json.dumps({"phase": "wm_checkpoint", "optimizer_steps": optimizer_steps,
                              "path": str(last_path), "seconds": perf_counter() - checkpoint_started_at}), flush=True)
        if improved:
            export_best(model)
    model.eval()
    return {"checkpoint_path": str(last_path), "best_checkpoint_path": str(best_path),
            "history": history, "optimizer_steps": optimizer_steps, "best_step": best_step,
            "best_prediction_score": best_metric, "validation": history[-1]["validation"], "contract": contract}
