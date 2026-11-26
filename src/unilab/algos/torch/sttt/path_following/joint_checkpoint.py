"""Atomic WM/PPO/FiLM/context state; physical streams resume in new episodes."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import torch

from ..world_model.checkpoint import _safe_training_state, canonical_contract
from ..world_model.reference import SOURCE

FORMAT = "sttt_joint_latent_pretraining"


def save_joint_checkpoint(path, world_model_learner, actor_learner, replay, *, run_id: str,
                          signature: Mapping[str, Any], state: Mapping[str, Any]) -> Path:
    path = Path(path)
    if not run_id:
        raise ValueError("joint checkpoint requires a run identity")
    if path.exists():
        previous = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(previous, dict) or previous.get("format") != FORMAT or previous.get("run_id") != run_id:
            raise FileExistsError(f"refusing to replace another run's checkpoint: {path}")
        del previous
    model, actor = world_model_learner.model, actor_learner.actor
    if actor_learner.rollout.model is not model:
        raise ValueError("paired learners must share the same physical model")
    model_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    actor_state = {key: value.detach().cpu() for key, value in actor.state_dict().items()}
    if any(value.is_floating_point() and not bool(torch.isfinite(value).all()) for value in (*model_state.values(), *actor_state.values())):
        raise ValueError("cannot checkpoint non-finite joint weights")
    payload = {"format": FORMAT, "schema_version": 5, "model_source": dict(SOURCE), "run_id": run_id,
        "signature": _safe_training_state(signature), "state": _safe_training_state(state),
        "contract": canonical_contract(replay.contract), "model_config": model.model_config,
        "world_model": model_state, "actor": actor_state,
        "actor_architecture": {"feature_dim": actor.feature_dim, "hidden_dims": actor.hidden_dims,
            "command_dim": actor.command_dim, "preview_points": actor.preview_points,
            "residual_limit_rad29": actor.residual_limit_rad29.detach().cpu().tolist(),
            "adaptation_dim": actor.adaptation_dim},
        "world_model_learner": _safe_training_state(world_model_learner.state_dict()),
        "actor_learner": _safe_training_state(actor_learner.state_dict()), "replay": replay.state_dict()}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def load_joint_checkpoint(path) -> dict[str, Any]:
    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    fields = {"format", "schema_version", "model_source", "run_id", "signature", "state", "contract", "model_config",
              "world_model", "actor", "actor_architecture", "world_model_learner", "actor_learner", "replay"}
    if not isinstance(payload, dict) or set(payload) != fields or payload["format"] != FORMAT or payload["schema_version"] != 5 or payload["model_source"] != SOURCE:
        raise ValueError("expected a version-5 velocity-readout/PPO/FiLM joint checkpoint")
    canonical_contract(payload["contract"])
    if not isinstance(payload["run_id"], str) or not payload["run_id"]:
        raise ValueError("paired checkpoint has no run identity")
    if payload["replay"]["contract"] != payload["contract"]:
        raise ValueError("paired replay and model contracts differ")
    return payload
