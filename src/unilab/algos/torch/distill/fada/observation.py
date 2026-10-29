from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

FADA_LEGACY_OBSERVATION_CONTRACT = "legacy_actor_obs_v1"
FADA_G1_STATE_OBSERVATION_CONTRACT = "g1_fada_state_v2"
FADA_OFFICIAL_COMPACT_OBSERVATION_CONTRACT = "official_compact_64_v1"
FADA_G1_ACTOR_OBS_DIM = 98
FADA_G1_STATE_DIM = 66
FADA_G1_ACTION_DIM = 29
FADA_G1_COMMAND_DIM = 3


_OFFICIAL_COMPACT_TERMS = ("base_ang_vel", "dof_pos", "dof_vel", "projected_gravity")


def project_official_7d_command(command: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """Project a 3D velocity command and 2D gait phase into the official 7D command.

    The layout is ``[vx, vy, wz, sin(left), sin(right), cos(left), cos(right)]``.
    Phase components are never zeroed at zero velocity: the four sin/cos entries
    always encode the gait phase of the same control instant.
    """

    cmd = torch.as_tensor(command, dtype=torch.float32)
    if cmd.ndim == 1:
        cmd = cmd.unsqueeze(0)
    if cmd.ndim != 2 or cmd.shape[1] != FADA_G1_COMMAND_DIM:
        raise ValueError(
            "official 7D command projection requires rank-2 commands with width "
            f"{FADA_G1_COMMAND_DIM}, got {tuple(cmd.shape)}"
        )
    ph = torch.as_tensor(phase, dtype=torch.float32, device=cmd.device)
    if ph.ndim == 1:
        ph = ph.unsqueeze(0)
    if ph.ndim != 2 or ph.shape != (cmd.shape[0], 2):
        raise ValueError(
            "official 7D command projection requires phase shaped "
            f"[batch, 2] matching command batch {cmd.shape[0]}, got {tuple(ph.shape)}"
        )
    if not bool(torch.isfinite(cmd).all()) or not bool(torch.isfinite(ph).all()):
        raise ValueError("official 7D command projection requires finite command and phase")
    return torch.cat((cmd, torch.sin(ph), torch.cos(ph)), dim=1)


def project_official_compact_observation(
    mapping: Mapping[str, Any],
    *,
    preprocessing: Mapping[str, Any],
    default_joint_pos: torch.Tensor,
    source_indices: torch.Tensor,
) -> torch.Tensor:
    """Project a physical-state mapping into the official compact 64D observation.

    The mapping carries base-frame signals assembled by the session from public
    env APIs: ``base_ang_vel`` [B,3] in the BASE frame, ``dof_pos``/``dof_vel``
    [B,29] absolute joint angles/velocities in TARGET joint order, and
    ``projected_gravity`` [B,3] in the BASE frame. ``default_joint_pos`` and
    ``source_indices`` are the source-order default pose and source-order gather
    indices into target arrays, prepared once by the controller. Per-term scales
    come from the checkpoint-bound preprocessing, applied exactly once here.
    """

    if not isinstance(mapping, Mapping):
        raise ValueError("official compact projection requires a physical-state mapping")
    missing = [term for term in _OFFICIAL_COMPACT_TERMS if term not in mapping]
    if missing:
        raise ValueError(f"official compact projection requires physical terms, missing={missing}")
    if (
        not isinstance(preprocessing, Mapping)
        or tuple(preprocessing.get("term_order", ())) != _OFFICIAL_COMPACT_TERMS
        or preprocessing.get("noise_enabled") is not False
    ):
        raise ValueError("official compact projection requires checkpoint-bound canonical preprocessing")
    scales = preprocessing.get("term_scale")
    if not isinstance(scales, Mapping) or set(scales) != set(_OFFICIAL_COMPACT_TERMS):
        raise ValueError("official compact projection requires per-term scales for all compact terms")

    defaults = torch.as_tensor(default_joint_pos, dtype=torch.float32)
    indices = torch.as_tensor(source_indices, dtype=torch.long)
    if defaults.ndim != 1 or defaults.shape[0] != FADA_G1_ACTION_DIM:
        raise ValueError(
            f"official compact projection requires {FADA_G1_ACTION_DIM} source default joint positions"
        )
    if indices.ndim != 1 or indices.shape[0] != FADA_G1_ACTION_DIM:
        raise ValueError(
            f"official compact projection requires {FADA_G1_ACTION_DIM} source gather indices"
        )
    device = defaults.device
    indices = indices.to(device=device)

    def _term(name: str, width: int) -> torch.Tensor:
        tensor = torch.as_tensor(mapping[name], dtype=torch.float32, device=device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[1] != width:
            raise ValueError(
                f"official compact projection term {name!r} requires [batch, {width}], "
                f"got {tuple(tensor.shape)}"
            )
        return tensor

    base_ang_vel = _term("base_ang_vel", 3)
    dof_pos = _term("dof_pos", FADA_G1_ACTION_DIM)
    dof_vel = _term("dof_vel", FADA_G1_ACTION_DIM)
    projected_gravity = _term("projected_gravity", 3)
    batch = base_ang_vel.shape[0]
    if dof_pos.shape[0] != batch or dof_vel.shape[0] != batch or projected_gravity.shape[0] != batch:
        raise ValueError("official compact projection terms must share one batch")

    # Gather target-order joints into source order, then take relative angles
    # against the source default pose before applying checkpoint scales.
    source_pos = torch.index_select(dof_pos, 1, indices)
    source_vel = torch.index_select(dof_vel, 1, indices)
    terms = {
        "base_ang_vel": base_ang_vel,
        "dof_pos": source_pos - defaults,
        "dof_vel": source_vel,
        "projected_gravity": projected_gravity,
    }
    compact = torch.cat(
        [terms[name] * float(scales[name]) for name in _OFFICIAL_COMPACT_TERMS],
        dim=1,
    )
    if not bool(torch.isfinite(compact).all()):
        raise ValueError("official compact projection requires finite physical terms")
    return compact


def project_fada_g1_state(source_obs: np.ndarray) -> np.ndarray:
    """Project one G1 actor-observation batch into the non-leaking FADA state."""

    source = np.asarray(source_obs, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != FADA_G1_ACTOR_OBS_DIM:
        raise ValueError(
            "g1_fada_state_v2 requires finite rank-2 actor observations with width "
            f"{FADA_G1_ACTOR_OBS_DIM}, got {source.shape}"
        )
    if not bool(np.all(np.isfinite(source))):
        raise ValueError("g1_fada_state_v2 requires finite actor observations")
    return np.concatenate([source[:, :64], source[:, 96:98]], axis=1, dtype=np.float32)


def projection_for_fada_observation_contract(observation_contract: str) -> str:
    """Return the sole admitted raw-observation projection for one architecture identity."""

    if observation_contract == FADA_LEGACY_OBSERVATION_CONTRACT:
        return "identity"
    if observation_contract == FADA_G1_STATE_OBSERVATION_CONTRACT:
        return FADA_G1_STATE_OBSERVATION_CONTRACT
    raise ValueError(f"unsupported FADA observation_contract: {observation_contract!r}")


def raw_observation_dim_for_fada_contract(
    observation_contract: str,
    *,
    policy_observation_dim: int,
) -> int:
    """Return the public environment width consumed before policy projection."""

    projection = projection_for_fada_observation_contract(observation_contract)
    return int(policy_observation_dim) if projection == "identity" else FADA_G1_ACTOR_OBS_DIM


def project_fada_observation_tensor(
    observation: torch.Tensor,
    *,
    observation_contract: str,
) -> torch.Tensor:
    """Project a hot-path playback tensor without device or host transfer."""

    if observation_contract == FADA_LEGACY_OBSERVATION_CONTRACT:
        return observation
    if observation_contract != FADA_G1_STATE_OBSERVATION_CONTRACT:
        raise ValueError(f"unsupported FADA observation_contract: {observation_contract!r}")
    if observation.ndim != 2 or observation.shape[1] != FADA_G1_ACTOR_OBS_DIM:
        raise ValueError(
            "g1_fada_state_v2 playback requires rank-2 actor observations with width "
            f"{FADA_G1_ACTOR_OBS_DIM}, got {tuple(observation.shape)}"
        )
    return torch.cat((observation[:, :64], observation[:, 96:98]), dim=1)


def assert_fada_projection_matches_contract(
    *,
    observation_contract: str,
    projection: str,
) -> None:
    """Fail before lifecycle mutation when a route mixes observation semantics."""

    expected = projection_for_fada_observation_contract(observation_contract)
    if projection != expected:
        raise ValueError(
            "FADA observation projection does not match architecture contract: "
            f"contract={observation_contract!r} expected={expected!r} observed={projection!r}"
        )


def assert_fada_active_route_contract(
    *,
    observation_contract: str,
    projection: str,
) -> None:
    """Admit only the active non-leaking G1 contract at Stage C/D boundaries."""

    assert_fada_projection_matches_contract(
        observation_contract=observation_contract,
        projection=projection,
    )
    if observation_contract != FADA_G1_STATE_OBSERVATION_CONTRACT:
        raise ValueError(
            "active FADA route requires observation_contract="
            f"{FADA_G1_STATE_OBSERVATION_CONTRACT!r}, got {observation_contract!r}"
        )
