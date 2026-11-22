"""Pretrained residual policy bound to a physical model and reference preview schema."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from unilab.algos.torch.sttt.path_following.latent_actor import LatentResidualActor
from unilab.algos.torch.sttt.path_following.trajectory import TrajectoryConfig
from unilab.algos.torch.sttt.world_model.checkpoint import _safe_training_state, canonical_contract
from unilab.algos.torch.sttt.world_model.model import LatentWorldModel

FORMAT = "sttt_pretrained_latent_actor"
PREVIEW_CHANNELS = ["body_x_m", "body_y_m", "tangent_cos", "tangent_sin", "speed_m_s"]


def physical_model_identity(model: LatentWorldModel, contract: Mapping[str, Any]) -> str:
    """Hash base weights only; online LoRA updates retain the pretrained base identity."""
    digest = hashlib.sha256(json.dumps({"model": model.model_config, "contract": canonical_contract(contract)}, sort_keys=True).encode())
    state = {}
    for name, value in model.state_dict().items():
        if name.endswith(("lora_A.weight", "lora_B.weight")):
            continue
        name = name.replace(".base.", ".")
        state[name] = value
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def save_actor_checkpoint(path: str | Path, actor: LatentResidualActor, model: LatentWorldModel,
                          contract: Mapping[str, Any], trajectory: TrajectoryConfig, *,
                          training_state: Mapping[str, Any]) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if actor.preview_points != trajectory.preview_points:
        raise ValueError("actor and reference preview sizes differ")
    state = {name: value.detach().cpu().clone() for name, value in actor.state_dict().items()}
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError("pretrained actor contains non-finite parameters")
    payload = {"format": FORMAT, "schema_version": 1,
               "physical_model_identity": physical_model_identity(model, contract),
               "contract": canonical_contract(contract),
               "architecture": {"feature_dim": actor.feature_dim, "hidden_dims": actor.hidden_dims,
                                "command_dim": actor.command_dim, "preview_points": actor.preview_points,
                                "residual_limit_rad29": actor.residual_limit_rad29.detach().cpu().tolist()},
               "preview": {"channels": PREVIEW_CHANNELS, "spacing_m": trajectory.preview_spacing_m},
               "actor": state, "training_state": _safe_training_state(dict(training_state))}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        torch.save(payload, handle)
    temporary.replace(path)
    return path


def load_actor_checkpoint(path: str | Path, model: LatentWorldModel, contract: Mapping[str, Any],
                          trajectory: TrajectoryConfig, *, require_model_identity: bool = True,
                          residual_limits: tuple[float, ...] | None = None) -> tuple[LatentResidualActor, dict[str, Any]]:
    parameter = next(model.parameters())
    payload = torch.load(path, map_location=parameter.device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT or payload.get("schema_version") != 1:
        raise ValueError("expected a pretrained latent actor artifact")
    if payload["contract"] != canonical_contract(contract):
        raise ValueError("pretrained actor belongs to another physical control contract")
    if require_model_identity and payload["physical_model_identity"] != physical_model_identity(model, contract):
        raise ValueError("pretrained actor belongs to another WM base; retrain A after updating WM")
    if payload["preview"] != {"channels": PREVIEW_CHANNELS, "spacing_m": trajectory.preview_spacing_m}:
        raise ValueError("pretrained actor reference preview semantics differ")
    architecture = payload["architecture"]
    if architecture["feature_dim"] != model.feature_dim or architecture["preview_points"] != trajectory.preview_points:
        raise ValueError("pretrained actor feature/preview dimensions differ")
    if residual_limits is not None and not torch.equal(parameter.new_tensor(architecture["residual_limit_rad29"]), parameter.new_tensor(residual_limits)):
        raise ValueError("pretrained actor residual bounds differ from deployment")
    actor = LatentResidualActor(**architecture).to(parameter)
    actor.load_state_dict(payload["actor"], strict=True)
    for value in actor.state_dict().values():
        if not bool(torch.isfinite(value).all()):
            raise ValueError("pretrained actor contains non-finite tensors")
    return actor.eval(), payload["training_state"]
