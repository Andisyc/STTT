"""Independent B artifacts; no policy checkpoint or retired world-model loader."""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from unilab.algos.torch.sttt.path_following.features import validate_response_contract
from unilab.algos.torch.sttt.path_following.model import ResponseModel

FORMAT = "sttt_path_following_response_model"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LoadedResponseModel:
    model: ResponseModel
    contract: dict[str, Any]
    metadata: dict[str, Any]


def _validate_state(model: ResponseModel) -> None:
    if model.context_mean.dtype not in (torch.float32, torch.float64):
        raise ValueError("B artifacts support float32 or float64 parameters and statistics")
    if int(model.normalization_samples) < 1:
        raise ValueError("B artifact requires fitted training-only input statistics")
    if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        raise ValueError("B artifact contains non-finite weights or statistics")
    if not bool((model.context_inv_scale > 0).all()) or not bool((model.residual_inv_scale > 0).all()) or not bool((model.residual_max > model.residual_min).all()):
        raise ValueError("B artifact contains invalid input scales or residual coverage")


def _check_dimensions(model: ResponseModel, contract: Mapping[str, Any]) -> None:
    data = contract["data"]
    if model.history_length != data["history_length"] or model.prediction_horizon != data["prediction_horizon"]:
        raise ValueError("B architecture contradicts the source context dimensions")


def save_response_checkpoint(
    path: str | Path, model: ResponseModel, *, contract: Mapping[str, Any],
    training_config: Mapping[str, Any], data_sources: list[dict[str, Any]], metrics: Mapping[str, Any],
) -> Path:
    target = Path(path).expanduser()
    if target.exists():
        raise FileExistsError(f"B checkpoint already exists: {target}")
    canonical = validate_response_contract(contract)
    _check_dimensions(model, canonical)
    _validate_state(model)
    metadata = json.loads(json.dumps({
        "training_config": dict(training_config), "data_sources": data_sources, "metrics": dict(metrics),
    }, allow_nan=False))
    payload = {
        "format": FORMAT, "schema_version": SCHEMA_VERSION, "model_config": model.model_config,
        "response_contract": canonical, "state_dict": model.state_dict(), **metadata,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)
    return target


def load_response_checkpoint(
    path: str | Path, *, expected_contract: Mapping[str, Any] | None = None,
    device: str | torch.device = "cpu",
) -> LoadedResponseModel:
    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    fields = {
        "format", "schema_version", "model_config", "response_contract", "state_dict",
        "training_config", "data_sources", "metrics",
    }
    if not isinstance(payload, dict) or set(payload) != fields or payload["format"] != FORMAT or type(payload["schema_version"]) is not int or payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported B artifact; policy and retired world-model checkpoints are not accepted")
    contract = validate_response_contract(payload["response_contract"])
    if expected_contract is not None and contract != validate_response_contract(expected_contract):
        raise ValueError("B checkpoint is incompatible with the requested response contract")
    config = payload["model_config"]
    if not isinstance(config, dict) or set(config) != {"history_length", "prediction_horizon", "hidden_dims"}:
        raise ValueError("invalid B architecture config")
    state = payload["state_dict"]
    mean = state.get("context_mean") if isinstance(state, Mapping) else None
    if not isinstance(mean, torch.Tensor) or mean.dtype not in (torch.float32, torch.float64):
        raise ValueError("invalid B input-statistics dtype")
    if any(not isinstance(value, torch.Tensor) or (value.is_floating_point() and value.dtype != mean.dtype) for value in state.values()):
        raise ValueError("B state must contain consistently typed tensors")
    model = ResponseModel(**config).to(dtype=mean.dtype)
    _check_dimensions(model, contract)
    model.load_state_dict(state, strict=True)
    _validate_state(model)
    metadata = {key: payload[key] for key in ("training_config", "data_sources", "metrics")}
    json.dumps(metadata, allow_nan=False)
    return LoadedResponseModel(model.to(device).eval(), contract, metadata)
