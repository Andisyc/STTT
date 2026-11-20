"""Independent, weights-only loadable categorical world-model artifacts."""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import torch

from .model import LatentWorldModel, WorldModelConfig
from .reference import SOURCE

FORMAT = "sttt_latent_world_model"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LoadedWorldModel:
    model: LatentWorldModel
    contract: dict[str, Any]
    training_state: dict[str, Any]


def canonical_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("world-model artifact requires a nonempty source contract")
    try:
        canonical = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("world-model contract must be finite JSON metadata") from error
    data, source = canonical.get("data"), canonical.get("source")
    required_source = {"checkpoint_sha256", "source_profile", "preprocessing", "target_joint_names", "target_default_joint_angles", "target_action_scale", "delta_frame", "delta_units", "ego_frame"}
    if not isinstance(data, dict) or not isinstance(source, dict) or not required_source <= source.keys():
        raise ValueError("world-model contract requires complete data/source policy identities")
    if data.get("observation_dim") != 64 or data.get("action_dim") != 29:
        raise ValueError("world-model contract requires compact64 observations and 29D ENV actions")
    if source["delta_frame"] != "fixed_straight_path" or source["delta_units"] != ["m", "m", "rad"]:
        raise ValueError("world-model motion labels must use fixed straight path meters/radians")
    if not isinstance(source["checkpoint_sha256"], str) or not source["checkpoint_sha256"] or len(source["target_joint_names"]) != 29 or len(source["target_default_joint_angles"]) != 29:
        raise ValueError("world-model source checkpoint/joint mapping is incomplete")
    return canonical


def _safe_training_state(value: Any) -> Any:
    """Copy only primitive containers/tensors accepted by weights_only loading."""
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError("training state contains non-finite tensors")
        return value.detach().cpu().clone()
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not torch.isfinite(torch.tensor(value, dtype=torch.float64)):
            raise ValueError("training state contains a non-finite scalar")
        return value
    if isinstance(value, Mapping):
        if any(type(key) not in (int, str) for key in value):
            raise ValueError("training-state dictionary keys must be primitive")
        return {key: _safe_training_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_safe_training_state(item) for item in value)
    raise ValueError(f"unsupported training state value: {type(value).__name__}")


def _validate_model(model: LatentWorldModel) -> None:
    dtype = next(model.parameters()).dtype
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("world-model artifacts require float32 or float64 weights")
    for value in model.state_dict().values():
        if not bool(torch.isfinite(value).all()) or value.is_floating_point() and value.dtype != dtype:
            raise ValueError("world-model state must be finite and consistently typed")
    expected = torch.tensor(model.config.action_scale or (1.,) * model.config.action_dim, device=model.action_scale.device, dtype=dtype)
    if not torch.allclose(model.action_scale, expected, rtol=1e-7, atol=0.):
        raise ValueError("model action normalization disagrees with its architecture config")


def _save_world_model(path: str | Path, model: LatentWorldModel, contract: Mapping[str, Any], *,
                      training_state: Mapping[str, Any] | None = None, replace_run_id: str | None = None) -> Path:
    target = Path(path).expanduser()
    if target.exists():
        if replace_run_id is None:
            raise FileExistsError(f"world-model checkpoint already exists: {target}")
        previous = torch.load(target, map_location="cpu", weights_only=True)
        if not isinstance(previous, dict) or previous.get("format") != FORMAT or previous.get("training_state", {}).get("run_id") != replace_run_id:
            raise FileExistsError(f"refusing to replace another run's checkpoint: {target}")
    canonical = canonical_contract(contract)
    _validate_model(model)
    payload = {
        "format": FORMAT, "schema_version": SCHEMA_VERSION, "model_source": dict(SOURCE),
        "model_config": model.model_config, "contract": canonical,
        "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "training_state": _safe_training_state(dict(training_state or {})),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="wb", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def save_world_model(path: str | Path, model: LatentWorldModel, contract: Mapping[str, Any], *,
                     training_state: Mapping[str, Any] | None = None) -> Path:
    return _save_world_model(path, model, contract, training_state=training_state)


def save_training_checkpoint(path: str | Path, model: LatentWorldModel, contract: Mapping[str, Any], *,
                             training_state: Mapping[str, Any], run_id: str) -> Path:
    if Path(path).name not in {"last.pt", "best.pt"} or not run_id or training_state.get("run_id") != run_id:
        raise ValueError("training replacement requires a run-owned last.pt or best.pt")
    return _save_world_model(path, model, contract, training_state=training_state, replace_run_id=run_id)


def load_world_model(path: str | Path, *, device: str | torch.device = "cpu", expected_contract: Mapping[str, Any] | None = None) -> LoadedWorldModel:
    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    required = {"format", "schema_version", "model_source", "model_config", "contract", "state_dict", "training_state"}
    if not isinstance(payload, dict) or set(payload) != required or payload["format"] != FORMAT or type(payload["schema_version"]) is not int or payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported latent world-model artifact (policy/B checkpoints are not accepted)")
    if payload["model_source"] != SOURCE:
        raise ValueError("world-model reference revision or normalization semantics mismatch")
    contract = canonical_contract(payload["contract"])
    if expected_contract is not None and contract != canonical_contract(expected_contract):
        raise ValueError("world-model artifact has an incompatible source/data contract")
    config = WorldModelConfig.from_mapping(payload["model_config"])
    state = payload["state_dict"]
    if not isinstance(state, Mapping) or not state or any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("world-model state_dict must contain tensors")
    scale = state.get("action_scale")
    if not isinstance(scale, torch.Tensor) or scale.dtype not in (torch.float32, torch.float64):
        raise ValueError("invalid world-model normalization dtype")
    if any(value.is_floating_point() and value.dtype != scale.dtype for value in state.values()):
        raise ValueError("world-model checkpoint tensors have inconsistent floating dtypes")
    model = LatentWorldModel(config).to(dtype=scale.dtype)
    model.load_state_dict(state, strict=True)
    _validate_model(model)
    training_state = payload["training_state"]
    if not isinstance(training_state, dict):
        raise ValueError("world-model training state must be a dictionary")
    training_state = _safe_training_state(training_state)
    return LoadedWorldModel(model.to(device).eval(), contract, training_state)
