"""Causal context packing and cold-bound response-model interpretation."""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import torch

from unilab.algos.torch.distill.fada.collection_contract import validate_official_fada_preprocessing
from unilab.algos.torch.distill.fada.runtime_policy import OfficialFADAPlaybackProfile
from unilab.algos.torch.sttt.path_following.contract import _integer

if TYPE_CHECKING:
    from unilab.algos.torch.distill.fada.runtime_policy import FADAFutureQuery
    from unilab.algos.torch.sttt.path_following.contract import InterventionQueryRecord

CONTEXT_SCHEMA = "query_history_nominal_v1"


def context_dimension(history_length: int, prediction_horizon: int) -> int:
    return 94 * history_length + 64 * prediction_horizon + 7


def normalized_context_parts(
    context: torch.Tensor, mean: torch.Tensor, inv_scale: torch.Tensor, history_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared fixed-statistic arithmetic; preserve gradients and zero history padding."""
    obs_end = 64 * history_length
    act_end = obs_end + 29 * history_length
    normalized = (context - mean) * inv_scale
    mask = context[:, act_end:act_end + history_length, None]
    obs = normalized[:, :obs_end].reshape(-1, history_length, 64) * mask
    act = normalized[:, obs_end:act_end].reshape(-1, history_length, 29) * mask
    return obs.flatten(1), act.flatten(1), normalized[:, act_end:]


def pack_response_context(query: FADAFutureQuery | InterventionQueryRecord) -> torch.Tensor:
    """Pack only pre-intervention fields; neither outcomes nor adapted future enter z."""
    obs, act, mask = query.observation_history, query.action_history, query.history_valid_mask
    future, command = query.planner_future, query.command
    if (
        obs.ndim != 3 or obs.shape[0] < 1 or obs.shape[1] < 1 or obs.shape[2] != 64
        or act.shape != (*obs.shape[:2], 29) or mask.shape != obs.shape[:2]
        or mask.dtype != torch.bool or command.shape != (obs.shape[0], 7)
        or future.ndim != 3 or future.shape[0] != obs.shape[0] or future.shape[1] < 1 or future.shape[2] != 64
    ):
        raise ValueError("response context requires aligned Hx64/Hx29 histories, mask, command7 and Kx64 nominal future")
    for value in (act, command, future):
        if value.dtype != obs.dtype or value.device != obs.device:
            raise ValueError("context tensors must share dtype and device")
    if mask.device != obs.device or not obs.is_floating_point():
        raise ValueError("context mask must share the floating-point observation device")
    return torch.cat((obs.flatten(1), act.flatten(1), mask.to(obs.dtype), command, future.flatten(1)), dim=1)


def response_contract_from_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Bind dimensions and source control semantics, not payload/friction or path origin."""
    raw, source = metadata["contract"], metadata["source"]
    data = dict(raw)
    required = {
        "window_steps", "ctrl_dt", "history_length", "prediction_horizon", "joint_names",
        "position_scale", "velocity_scale", "tensor_dtype",
    }
    if set(data) != required:
        raise ValueError("response data contract has missing or unknown fields")
    for name in ("window_steps", "history_length", "prediction_horizon"):
        if _integer(data[name], name) < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("ctrl_dt", "position_scale", "velocity_scale"):
        value = data[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"invalid response {name}")
    if data["ctrl_dt"] <= 0 or data["tensor_dtype"] not in ("float32", "float64"):
        raise ValueError("response data has unsupported timing or dtype")
    preprocessing = validate_official_fada_preprocessing(source["preprocessing"])
    payload = source["source_profile"]
    profile = OfficialFADAPlaybackProfile.from_mapping(payload, source_identity=payload["source_identity"])
    if (
        tuple(data["joint_names"]) != profile.joint_names or data["ctrl_dt"] != profile.ctrl_dt
        or data["position_scale"] != preprocessing["term_scale"]["dof_pos"]
        or data["velocity_scale"] != preprocessing["term_scale"]["dof_vel"]
    ):
        raise ValueError("response data contradicts its bound source profile or preprocessing")
    digest = source["checkpoint_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("response source requires an actual checkpoint SHA256")
    if source["delta_frame"] != "window_start_horizontal_heading" or source["delta_units"] != ["m", "m", "rad"]:
        raise ValueError("response labels must be horizontal heading-frame displacement in m/m/rad")
    canonical_source = {
        "preprocessing": preprocessing, "source_profile": asdict(profile),
        "checkpoint_sha256": digest, "delta_frame": source["delta_frame"],
        "delta_units": source["delta_units"],
    }
    return json.loads(json.dumps({"context_schema": CONTEXT_SCHEMA, "data": data, "source": canonical_source}, allow_nan=False))


def validate_response_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    if set(contract) != {"context_schema", "data", "source"} or contract["context_schema"] != CONTEXT_SCHEMA:
        raise ValueError("unsupported response context schema")
    canonical = response_contract_from_metadata({"contract": contract["data"], "source": contract["source"]})
    if canonical != contract:
        raise ValueError("noncanonical response contract")
    return canonical
