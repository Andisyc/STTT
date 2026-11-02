"""Versioned spatial world-model checkpoints and epoch-boundary recovery."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import SpatialWorldModel

WORLD_MODEL_CHECKPOINT_VERSION = 1
DATA_CONTRACT_KEYS = (
    "obs_dim", "act_dim", "history_length", "horizon", "ctrl_dt",
    "proprio_layout", "joint_names", "action_scale", "default_joint_angles",
    "action_representation", "quaternion_order", "coordinate_frame",
)


def _canonical(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def data_contract_from_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Select policy-independent I/O semantics; retain source identities separately."""
    missing = set(DATA_CONTRACT_KEYS) - metadata.keys()
    if missing:
        raise ValueError(f"Missing spatial data contract fields: {sorted(missing)}")
    return _canonical({key: metadata[key] for key in DATA_CONTRACT_KEYS})


def world_model_spec(model: SpatialWorldModel) -> dict[str, Any]:
    return {
        "architecture": "spatial_mlp_v1", "obs_dim": model.obs_dim,
        "act_dim": model.act_dim, "horizon": model.horizon,
        "hist_len": model.hist_len, "hidden": list(model.hidden_dims),
    }


def _rng_state(device: torch.device) -> dict[str, Any]:
    numpy_state = np.random.get_state(legacy=True)
    state = {
        "python": random.getstate(),
        "numpy": [numpy_state[0], torch.tensor(numpy_state[1].astype(np.int64)),
                  numpy_state[2], numpy_state[3], numpy_state[4]],
        "torch": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    elif device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], numpy_state[1].numpy().astype(np.uint32),
                         numpy_state[2], numpy_state[3], numpy_state[4]))
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        if not torch.cuda.is_available():
            raise ValueError("CUDA RNG recovery requires a CUDA runtime")
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state:
        if not torch.backends.mps.is_available():
            raise ValueError("MPS RNG recovery requires an MPS runtime")
        torch.mps.set_rng_state(state["mps"])


def save_world_model_checkpoint(
    path: str | Path,
    model: SpatialWorldModel,
    *,
    data_contract: Mapping[str, Any],
    data_identity: Mapping[str, Any],
    split: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    completed_epochs: int = 0,
    best_validation_loss: float | None = None,
    shuffle_generator: torch.Generator | None = None,
) -> Path:
    """Save only at an epoch boundary; epoch zero also supports a smoke snapshot."""
    contract = data_contract_from_metadata(data_contract)
    spec = world_model_spec(model)
    if any(spec[key] != contract[field] for key, field in
           (("obs_dim", "obs_dim"), ("act_dim", "act_dim"),
            ("hist_len", "history_length"), ("horizon", "horizon"))):
        raise ValueError("Model structure does not match the spatial data contract")
    if completed_epochs < 0:
        raise ValueError("completed_epochs cannot be negative")
    if best_validation_loss is not None and not math.isfinite(best_validation_loss):
        raise ValueError("best_validation_loss must be finite or None")
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Cannot save non-finite world-model state")
    payload = {
        "format_version": WORLD_MODEL_CHECKPOINT_VERSION,
        "kind": "spatial_world_model", "loss_contract": "periodic_se2_v1",
        "model_spec": spec, "model_state_dict": model.state_dict(),
        "model_dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
        "normalization": {
            name: {"eps": norm.eps, "max_count": norm.max_count}
            for name, norm in (("in_norm", model.in_norm), ("out_norm", model.out_norm))
        },
        "data_contract": contract, "data_identity": _canonical(data_identity),
        "split": _canonical(split), "resolved_config": _canonical(resolved_config),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "completed_epochs": int(completed_epochs),
        "best_validation_loss": best_validation_loss,
        "rng_state": _rng_state(next(model.parameters()).device),
        "shuffle_rng_state": None if shuffle_generator is None else shuffle_generator.get_state(),
    }
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_world_model_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    expected_data_contract: Mapping[str, Any] | None = None,
    expected_data_identity: Mapping[str, Any] | None = None,
    expected_split: Mapping[str, Any] | None = None,
    expected_model_spec: Mapping[str, Any] | None = None,
) -> tuple[SpatialWorldModel, dict[str, Any]]:
    """Rebuild an inference model without changing global RNG or fitting statistics."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (not isinstance(payload, dict) or payload.get("format_version") != WORLD_MODEL_CHECKPOINT_VERSION
            or payload.get("kind") != "spatial_world_model"
            or payload.get("loss_contract") != "periodic_se2_v1"):
        raise ValueError("Unsupported spatial world-model checkpoint")
    contract = data_contract_from_metadata(payload["data_contract"])
    comparisons = (
        ("data_contract", expected_data_contract), ("data_identity", expected_data_identity),
        ("split", expected_split), ("model_spec", expected_model_spec),
    )
    for key, expected in comparisons:
        if expected is not None and _canonical(payload[key]) != _canonical(expected):
            raise ValueError(f"Checkpoint {key} does not match the requested runtime")
    spec = dict(payload["model_spec"])
    if spec.pop("architecture", None) != "spatial_mlp_v1":
        raise ValueError("Unsupported world-model architecture")
    if any(spec[key] != contract[field] for key, field in
           (("obs_dim", "obs_dim"), ("act_dim", "act_dim"),
            ("hist_len", "history_length"), ("horizon", "horizon"))):
        raise ValueError("Checkpoint model structure conflicts with its data contract")
    dtypes = {"float32": torch.float32, "float64": torch.float64,
              "float16": torch.float16, "bfloat16": torch.bfloat16}
    if payload["model_dtype"] not in dtypes:
        raise ValueError("Unsupported checkpoint model dtype")
    with torch.random.fork_rng(devices=[]):
        model = SpatialWorldModel(**spec).to(dtype=dtypes[payload["model_dtype"]])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    for name in ("in_norm", "out_norm"):
        norm = getattr(model, name)
        norm.eps = float(payload["normalization"][name]["eps"])
        norm.max_count = float(payload["normalization"][name]["max_count"])
        if norm.eps <= 0 or norm.max_count <= 0:
            raise ValueError("Checkpoint normalization parameters must be positive")
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Checkpoint contains non-finite model state")
    return model.to(device).eval(), payload


def restore_training_state(
    payload: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    *,
    shuffle_generator: torch.Generator,
) -> None:
    """Restore optimizer and random state after dataset/model contract checks."""
    if payload.get("optimizer_state_dict") is None or payload.get("shuffle_rng_state") is None:
        raise ValueError("Checkpoint lacks optimizer or shuffle state for epoch-boundary resume")
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    _restore_rng(payload["rng_state"])
    shuffle_generator.set_state(payload["shuffle_rng_state"])
