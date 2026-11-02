"""Offline spatial world-model training and evaluation.

Plain-PyTorch, dependency-free beyond OmegaConf for the config: Adam +
gradient clipping, group-disjoint train/validation split, epoch-boundary
checkpointing/resume, and a JSON-lines epoch metrics log. No schedulers,
AMP/DDP, streaming, or sample exploration.

The dense MLP does not enforce temporal causality; per-step metrics
surface this rather than hiding it.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import DataLoader, SequentialSampler, Subset

from .checkpoint import (
    load_world_model_checkpoint,
    restore_training_state,
    save_world_model_checkpoint,
)
from .contract import SpatialBatch
from .dataset import SpatialTrajectoryDataset, make_group_split
from .model import SpatialWorldModel

__all__ = [
    "fit_normalizers",
    "evaluate_world_model",
    "run_world_model_training",
    "run_world_model_evaluation",
]

_EVAL_SPLITS = ("validation", "train", "all")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _collate_on_device(device: torch.device):
    def _collate(windows):
        return SpatialBatch.collate(windows, device=device)

    return _collate


def _make_loader(
    data: Any,
    *,
    batch_size: int,
    device: torch.device,
    shuffle_generator: torch.Generator | None,
) -> DataLoader:
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle_generator is not None,
        generator=shuffle_generator,
        num_workers=0,
        collate_fn=_collate_on_device(device),
    )


def _loss_weights_tensor(weights: Any, device: torch.device) -> torch.Tensor:
    values = [float(v) for v in weights]
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError(f"loss_weights must be 3 finite floats, got {weights!r}.")
    return torch.tensor(values, dtype=torch.float32, device=device)


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


#: Accumulator fields: sample-weighted sums, finalized into means.
_METRIC_FIELDS = (
    "loss",
    "trajectory_xy_m",
    "trajectory_yaw_rad",
    "endpoint_xy_m",
    "endpoint_yaw_rad",
    "zero_motion_trajectory_xy_m",
    "zero_motion_trajectory_yaw_rad",
    "zero_motion_endpoint_xy_m",
    "zero_motion_endpoint_yaw_rad",
)


def _empty_accumulator() -> dict[str, float]:
    accumulator = {key: 0.0 for key in _METRIC_FIELDS}
    accumulator["count"] = 0.0
    return accumulator


def _accumulate(
    accumulator: dict[str, float],
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_value: float,
) -> None:
    """Add one batch slice to ``accumulator`` as raw sums.

    The zero-motion baseline is the target trajectory itself, because a
    stationary prediction is exactly zero in the local SE(2) frame at
    ``t = 0``; no extra model pass is needed.
    """
    count = int(pred.shape[0])
    error = pred - target
    accumulator["loss"] += loss_value * count
    accumulator["trajectory_xy_m"] += (
        torch.linalg.vector_norm(error[..., :2], dim=-1).sum().item()
    )
    accumulator["trajectory_yaw_rad"] += _wrap_angle(error[..., 2]).abs().sum().item()
    accumulator["endpoint_xy_m"] += (
        torch.linalg.vector_norm(error[:, -1, :2], dim=-1).sum().item()
    )
    accumulator["endpoint_yaw_rad"] += _wrap_angle(error[:, -1, 2]).abs().sum().item()
    accumulator["zero_motion_trajectory_xy_m"] += (
        torch.linalg.vector_norm(target[..., :2], dim=-1).sum().item()
    )
    accumulator["zero_motion_trajectory_yaw_rad"] += (
        _wrap_angle(target[..., 2]).abs().sum().item()
    )
    accumulator["zero_motion_endpoint_xy_m"] += (
        torch.linalg.vector_norm(target[:, -1, :2], dim=-1).sum().item()
    )
    accumulator["zero_motion_endpoint_yaw_rad"] += (
        _wrap_angle(target[:, -1, 2]).abs().sum().item()
    )
    accumulator["count"] += count


def _finalize(accumulator: dict[str, float], horizon: int) -> dict[str, float]:
    total = accumulator["count"]
    if total <= 0:
        raise ValueError("Cannot finalize metrics for zero samples.")
    return {
        "loss": accumulator["loss"] / total,
        "trajectory_xy_m": accumulator["trajectory_xy_m"] / (total * horizon),
        "trajectory_yaw_rad": accumulator["trajectory_yaw_rad"] / (total * horizon),
        "endpoint_xy_m": accumulator["endpoint_xy_m"] / total,
        "endpoint_yaw_rad": accumulator["endpoint_yaw_rad"] / total,
        "zero_motion_trajectory_xy_m": (
            accumulator["zero_motion_trajectory_xy_m"] / (total * horizon)
        ),
        "zero_motion_trajectory_yaw_rad": (
            accumulator["zero_motion_trajectory_yaw_rad"] / (total * horizon)
        ),
        "zero_motion_endpoint_xy_m": accumulator["zero_motion_endpoint_xy_m"] / total,
        "zero_motion_endpoint_yaw_rad": (
            accumulator["zero_motion_endpoint_yaw_rad"] / total
        ),
    }


def _window_indices(data: Any) -> list[int]:
    """Map a loader dataset (dataset or ``Subset`` view) to window indices."""
    if isinstance(data, SpatialTrajectoryDataset):
        return list(range(len(data.windows)))
    if isinstance(data, Subset):
        parent = _window_indices(data.dataset)
        return [parent[int(index)] for index in data.indices]
    raise TypeError(
        f"Cannot map {type(data).__name__} to spatial window indices; expected "
        f"SpatialTrajectoryDataset or a Subset view of one."
    )


# ---------------------------------------------------------------------------
# Normalizer fitting
# ---------------------------------------------------------------------------


@torch.no_grad()
def fit_normalizers(model: SpatialWorldModel, train_loader: DataLoader) -> None:
    """Fit input/output normalizers on training data only, exactly once.

    Flattens ``cat(proprio_history, action_sequence)`` for ``in_norm`` and
    the flattened target trajectories for ``out_norm`` (true physical
    values, never normalized copies). Intended to run before training;
    training and evaluation never update the buffers again.
    """
    for batch in train_loader:
        batch_size = batch.proprio_history.shape[0]
        inputs = torch.cat(
            [
                batch.proprio_history.reshape(batch_size, -1),
                batch.action_sequence.reshape(batch_size, -1),
            ],
            dim=-1,
        )
        model.in_norm.update(inputs)
        model.out_norm.update(batch.delta_trajectory.reshape(batch_size, -1))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_world_model(
    model: SpatialWorldModel,
    loader: DataLoader,
    *,
    device: str | torch.device = "cpu",
    loss_weights: tuple[float, float, float] | list[float] = (1.0, 1.0, 1.0),
    by_condition: bool = False,
) -> dict[str, Any]:
    """Evaluate without gradients or statistics mutation.

    Runs exactly one forward pass per batch. Returns a flat dict of
    sample-weighted means: the weighted periodic SE(2) ``loss``,
    full-trajectory XY distance MAE (``trajectory_xy_m``), wrapped
    full-trajectory yaw MAE (``trajectory_yaw_rad``), endpoint errors
    (``endpoint_xy_m``, ``endpoint_yaw_rad``), ``num_samples``, and the
    ``zero_motion_*`` counterparts of those four errors, measured against
    the target trajectory itself (a stationary prediction is zero in the
    local frame at ``t = 0``, so no second model pass is needed).

    With ``by_condition=True`` the same single pass also fills
    ``condition_metrics``: one record per
    ``(label_source, collection_context)`` provenance group present in
    ``loader.dataset``, each holding its own prediction and zero-motion
    metrics plus source files and sample counts.

    The model's training/eval mode is preserved.
    """
    device = torch.device(device)
    weights = _loss_weights_tensor(loss_weights, device)
    groups = _condition_groups_for_loader(loader) if by_condition else None
    overall = _empty_accumulator()
    per_group: dict[int, dict[str, float]] = {}
    was_training = model.training
    model.eval()
    try:
        offset = 0
        for batch in loader:
            batch_size = batch.proprio_history.shape[0]
            pred = model(batch.proprio_history, batch.action_sequence)
            loss = model.compute_loss(pred, batch.delta_trajectory, weights)
            _accumulate(overall, pred, batch.delta_trajectory, float(loss))
            if groups is not None:
                batch_groups = torch.as_tensor(
                    groups[offset : offset + batch_size], device=pred.device
                )
                for group in sorted(set(groups[offset : offset + batch_size])):
                    selector = batch_groups == group
                    group_pred = pred[selector]
                    group_target = batch.delta_trajectory[selector]
                    _accumulate(
                        per_group.setdefault(group, _empty_accumulator()),
                        group_pred,
                        group_target,
                        float(model.compute_loss(group_pred, group_target, weights)),
                    )
            offset += batch_size
        if int(overall["count"]) == 0:
            raise ValueError("evaluate_world_model received an empty loader.")
        if groups is not None and offset != len(groups):
            raise ValueError(
                f"Loader yielded {offset} samples but the dataset maps "
                f"{len(groups)} window indices."
            )
        horizon = model.horizon
        metrics: dict[str, Any] = _finalize(overall, horizon)
        metrics["num_samples"] = float(int(overall["count"]))
        if groups is not None:
            metrics["condition_metrics"] = _condition_records(loader, per_group, horizon)
        return metrics
    finally:
        if was_training:
            model.train()


def _condition_groups_for_loader(loader: DataLoader) -> list[int]:
    dataset = getattr(loader, "dataset", None)
    if not isinstance(dataset, (SpatialTrajectoryDataset, Subset)):
        raise TypeError(
            "by_condition evaluation requires a loader over a "
            f"SpatialTrajectoryDataset or Subset, got {type(dataset).__name__}."
        )
    sampler = getattr(loader, "sampler", None)
    if not isinstance(sampler, SequentialSampler):
        raise ValueError(
            "by_condition evaluation requires an unshuffled loader so batch "
            "samples map onto dataset provenance in order."
        )
    parent = dataset.dataset if isinstance(dataset, Subset) else dataset
    if not isinstance(parent, SpatialTrajectoryDataset):
        raise TypeError(
            "by_condition evaluation requires a loader over a "
            f"SpatialTrajectoryDataset, got {type(parent).__name__}."
        )
    if len(parent.condition_groups) != len(parent.windows):
        raise ValueError(
            "Dataset provenance indices are inconsistent with its windows."
        )
    return [parent.condition_groups[index] for index in _window_indices(dataset)]


def _condition_records(
    loader: DataLoader,
    per_group: dict[int, dict[str, float]],
    horizon: int,
) -> list[dict[str, Any]]:
    """JSON-safe per-condition records in dataset condition order."""
    dataset = loader.dataset
    parent = dataset.dataset if isinstance(dataset, Subset) else dataset
    indices_by_group: dict[int, list[int]] = {}
    for index in _window_indices(dataset):
        indices_by_group.setdefault(parent.condition_groups[index], []).append(index)
    records: list[dict[str, Any]] = []
    for group in sorted(per_group):
        condition = parent.conditions[group]
        indices = indices_by_group.get(group, [])
        files = sorted(
            {
                parent.data_identity["files"][parent.window_sources[index]]["path"]
                for index in indices
            }
        )
        record = {
            "condition": condition["condition"],
            "label_source": condition["label_source"],
            "collection_context": condition["collection_context"],
            "files": files,
            "num_samples": int(per_group[group]["count"]),
        }
        record.update(_finalize(per_group[group], horizon))
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _resolved_config(cfg: DictConfig | dict) -> dict[str, Any]:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return json.loads(json.dumps(cfg, allow_nan=False))


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_model(contract: dict[str, Any], hidden_dims: Any) -> SpatialWorldModel:
    hidden = (
        list(hidden_dims)
        if isinstance(hidden_dims, (list, tuple, ListConfig))
        else [512, 512, 512]
    )
    return SpatialWorldModel(
        obs_dim=int(contract["obs_dim"]),
        act_dim=int(contract["act_dim"]),
        horizon=int(contract["horizon"]),
        hist_len=int(contract["history_length"]),
        hidden=[int(dim) for dim in hidden],
    )


def _indices_for_groups(
    dataset: SpatialTrajectoryDataset, groups: list[list[Any]]
) -> list[int]:
    wanted = {(str(group[0]), int(group[1])) for group in groups}
    indices = [
        index
        for index, window in enumerate(dataset.windows)
        if (str(window["rollout_id"]), int(window["episode_id"])) in wanted
    ]
    if not indices:
        raise ValueError("No dataset windows match the recorded split groups.")
    return indices


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def run_world_model_training(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    """Train a spatial world model from ``conf/sttt/world_model.yaml`` config.

    Fits normalizers once on the train split (skipped on resume), runs
    Adam with gradient clipping, logs JSON-lines epoch metrics to
    ``<output_dir>/metrics.jsonl``, saves ``last.pt`` every completed
    epoch and ``best.pt`` on lower validation loss, and supports
    epoch-boundary resume from ``cfg.training.resume``.
    """
    resolved = _resolved_config(cfg)
    data_cfg, model_cfg, train_cfg = (
        resolved["data"],
        resolved["model"],
        resolved["training"],
    )
    device = torch.device(train_cfg["device"])
    seed = int(train_cfg["seed"])
    epochs = int(train_cfg["epochs"])
    batch_size = int(train_cfg["batch_size"])
    max_grad_norm = float(train_cfg["max_grad_norm"])
    weights = _loss_weights_tensor(
        train_cfg.get("loss_weights", (1.0, 1.0, 1.0)), device
    )
    resume_path = train_cfg.get("resume")
    output_dir = Path(train_cfg["output_dir"]).expanduser()
    if epochs < 1:
        raise ValueError(f"training.epochs must be >= 1, got {epochs}.")
    if resume_path is None and (output_dir / "last.pt").exists():
        raise FileExistsError(
            f"{output_dir / 'last.pt'} already exists; pass "
            f"training.resume={output_dir / 'last.pt'} to continue it or "
            f"choose a fresh output_dir (refusing to overwrite silently)."
        )

    dataset = SpatialTrajectoryDataset(list(data_cfg["paths"]))
    train_subset, val_subset, split = make_group_split(
        dataset,
        validation_fraction=float(data_cfg["validation_fraction"]),
        seed=int(data_cfg["split_seed"]),
    )

    _set_seeds(seed)
    shuffle_generator = torch.Generator().manual_seed(seed)

    if resume_path is not None:
        model, payload = load_world_model_checkpoint(
            resume_path,
            device=device,
            expected_data_contract=dataset.data_contract,
            expected_data_identity=dataset.data_identity,
            expected_split=split,
        )
        if model.hidden_dims != tuple(int(dim) for dim in model_cfg["hidden_dims"]):
            raise ValueError(
                f"cfg model.hidden_dims {list(model_cfg['hidden_dims'])} does "
                f"not match resumed checkpoint {list(model.hidden_dims)}."
            )
        saved_training = payload["resolved_config"].get("training", {})
        for key in ("learning_rate", "max_grad_norm", "loss_weights", "batch_size", "seed"):
            if saved_training.get(key) != train_cfg.get(key):
                raise ValueError(f"Resume training.{key} differs from the saved training configuration")
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(train_cfg["learning_rate"])
        )
        restore_training_state(payload, optimizer, shuffle_generator=shuffle_generator)
        start_epoch = int(payload["completed_epochs"])
        best_validation_loss = payload["best_validation_loss"]
        if best_validation_loss is None:
            best_validation_loss = math.inf
        resumed = True
    else:
        model = _make_model(dataset.data_contract, model_cfg["hidden_dims"]).to(device)
        model.train()
        # One train-only fit pass over true data values; buffers are never
        # updated again during training or evaluation.
        fit_normalizers(
            model,
            _make_loader(
                train_subset,
                batch_size=batch_size,
                device=device,
                shuffle_generator=None,
            ),
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(train_cfg["learning_rate"])
        )
        start_epoch = 0
        best_validation_loss = math.inf
        resumed = False

    if epochs <= start_epoch:
        raise ValueError("training.epochs is the target total and must exceed completed_epochs")

    train_loader = _make_loader(
        train_subset,
        batch_size=batch_size,
        device=device,
        shuffle_generator=shuffle_generator,
    )
    val_loader = _make_loader(
        val_subset,
        batch_size=batch_size,
        device=device,
        shuffle_generator=None,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    total_epochs = epochs
    completed_epochs = start_epoch
    for epoch in range(start_epoch + 1, total_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch in train_loader:
            pred = model(batch.proprio_history, batch.action_sequence)
            loss = model.compute_loss(pred, batch.delta_trajectory, weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            batch_size_now = batch.proprio_history.shape[0]
            train_loss_sum += float(loss) * batch_size_now
            train_count += batch_size_now
        train_loss = train_loss_sum / train_count
        val_metrics = evaluate_world_model(
            model, val_loader, device=device, loss_weights=tuple(weights.tolist())
        )
        completed_epochs = epoch
        is_best = val_metrics["loss"] < best_validation_loss
        if is_best:
            best_validation_loss = val_metrics["loss"]

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_trajectory_xy_m": val_metrics["trajectory_xy_m"],
            "val_trajectory_yaw_rad": val_metrics["trajectory_yaw_rad"],
            "val_endpoint_xy_m": val_metrics["endpoint_xy_m"],
            "val_endpoint_yaw_rad": val_metrics["endpoint_yaw_rad"],
            "best_validation_loss": best_validation_loss,
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False))

        checkpoint_kwargs = {
            "data_contract": dataset.data_contract,
            "data_identity": dataset.data_identity,
            "split": split,
            "resolved_config": resolved,
            "optimizer": optimizer,
            "completed_epochs": completed_epochs,
            "best_validation_loss": best_validation_loss,
            "shuffle_generator": shuffle_generator,
        }
        save_world_model_checkpoint(output_dir / "last.pt", model, **checkpoint_kwargs)
        if is_best:
            save_world_model_checkpoint(output_dir / "best.pt", model, **checkpoint_kwargs)

    return {
        "output_dir": str(output_dir),
        "epochs_completed": completed_epochs - start_epoch,
        "completed_epochs": completed_epochs,
        "best_validation_loss": best_validation_loss,
        "final_validation": evaluate_world_model(
            model, val_loader, device=device, loss_weights=tuple(weights.tolist())
        ),
        "train_samples": len(train_subset),
        "validation_samples": len(val_subset),
        "resumed": resumed,
        "checkpoint_last": str(output_dir / "last.pt"),
        "checkpoint_best": str(output_dir / "best.pt"),
        "metrics_path": str(metrics_path),
    }


# ---------------------------------------------------------------------------
# Evaluation entry point
# ---------------------------------------------------------------------------


def run_world_model_evaluation(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    """Evaluate a saved world model from ``conf/sttt/world_model.yaml`` config.

    ``evaluation.split = "validation"`` (or ``"train"``) reproduces the
    checkpoint's recorded group split and requires the exact original
    dataset identity; ``"all"`` evaluates every window of any compatible
    dataset and is never labeled as the original holdout.

    ``evaluation.by_condition`` (default ``true``) groups those exact
    evaluated samples by each collection's ``label_source`` and
    ``collection_context`` and reports one ``condition_metrics`` record per
    group in the same single forward pass that produces the overall
    metrics; ``false`` returns only the overall metrics plus the
    zero-motion baseline. Returns one JSON-safe dict.
    """
    resolved = _resolved_config(cfg)
    data_cfg, eval_cfg = resolved["data"], resolved["evaluation"]
    checkpoint_path = eval_cfg.get("checkpoint")
    if not checkpoint_path:
        raise ValueError("evaluation.checkpoint must name a checkpoint file.")
    split_name = str(eval_cfg.get("split", "validation"))
    if split_name not in _EVAL_SPLITS:
        raise ValueError(
            f"evaluation.split must be one of {_EVAL_SPLITS}, got {split_name!r}."
        )
    device = torch.device(resolved["training"]["device"])
    loss_weights = resolved["training"].get("loss_weights", (1.0, 1.0, 1.0))
    batch_size = int(resolved["training"]["batch_size"])

    dataset = SpatialTrajectoryDataset(list(data_cfg["paths"]))
    model, payload = load_world_model_checkpoint(
        checkpoint_path,
        device=device,
        expected_data_contract=dataset.data_contract,
    )
    loss_weights = payload["resolved_config"].get("training", {}).get("loss_weights", loss_weights)

    original_holdout = False
    if split_name == "all":
        eval_data: Any = dataset
    else:
        if dataset.data_identity != payload["data_identity"]:
            raise ValueError(
                f"evaluation.split={split_name!r} requires the original "
                f"training dataset (path + sha256 identity mismatch). Use "
                f"evaluation.split=all to evaluate a new compatible dataset."
            )
        for key in ("seed", "validation_fraction", "train_groups", "validation_groups"):
            if key not in payload["split"]:
                raise ValueError(f"Recorded checkpoint split is missing {key!r}.")
        groups = payload["split"][
            "train_groups" if split_name == "train" else "validation_groups"
        ]
        eval_data = Subset(dataset, _indices_for_groups(dataset, groups))
        original_holdout = split_name == "validation"

    by_condition = bool(eval_cfg.get("by_condition", True))
    loader = _make_loader(
        eval_data,
        batch_size=batch_size,
        device=device,
        shuffle_generator=None,
    )
    metrics = evaluate_world_model(
        model,
        loader,
        device=device,
        loss_weights=loss_weights,
        by_condition=by_condition,
    )
    num_samples = int(metrics.pop("num_samples"))
    condition_metrics = metrics.pop("condition_metrics", None)
    result: dict[str, Any] = {
        "checkpoint": str(Path(checkpoint_path).expanduser()),
        "split": split_name,
        "num_samples": num_samples,
        "original_holdout": original_holdout,
        "by_condition": by_condition,
        **metrics,
    }
    if condition_metrics is not None:
        result["condition_metrics"] = condition_metrics
    return result
