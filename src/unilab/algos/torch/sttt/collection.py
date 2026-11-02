"""Spatial trajectory collection: windowing, serialization, and loading.

Builds :class:`SpatialTrajectoryWindow` training windows from a stream of
:class:`SpatialTransition` records and persists them in the canonical
STTT spatial data format (``format_version == 1``).

Window semantics
----------------
For a window whose origin (reference step) is transition ``t``:

- ``proprio_history`` stacks ``proprio`` of transitions
  ``t - history_length + 1 .. t`` (inclusive), shape ``(H_hist, 64)``.
- ``action_sequence`` stacks ``action`` of transitions
  ``t .. t + horizon - 1``, shape ``(horizon, 29)``.
- ``delta_trajectory`` holds SE(2) deltas of the ``next_position`` /
  ``next_quaternion`` poses of transitions ``t .. t + horizon - 1``
  relative to the reference pose (``position``, ``quaternion``) at ``t``.
- ``commands`` are the origin transition's commands.
- ``step_indices`` covers every state index touched by the window: the
  first history state ``t - history_length + 1`` through the last future
  state ``t + horizon`` inclusive, length ``history_length + horizon``.

Boundary rules
--------------
Windows never cross rollout or episode boundaries and never span
``episode_step`` gaps. A ``terminated``/``truncated`` transition is a
pure boundary marker: it is dropped on arrival and clears all pending
context, so it is never used as history, as an origin, or as a future
label. The state *after* the last non-terminal transition still enters
windows through that transition's ``next_position``/``next_quaternion``.

Payload condition labels
------------------------
Each window carries ``condition``, aggregated from the per-transition
runtime-payload truth the collector actually observed: ``stable`` with
``payload_active``/``added_mass_kg`` when every step of the window agrees,
``transitioning`` when the payload attached or restored inside it, and
``unknown`` when the env published no truth. The configured
``env.cfg.runtime_payload`` block is a collection condition, not proof that
a payload was attached in any particular window, so it never fills a label.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .contract import (
    DEFAULT_OBS_DIM,
    SPATIAL_ACTION_REPRESENTATION,
    SPATIAL_DATA_FORMAT_VERSION,
    SpatialTrajectoryWindow,
    SpatialTransition,
    validate_trajectory_window,
)
from .geometry import compute_se2_deltas

__all__ = [
    "SPATIAL_METADATA_REQUIRED_KEYS",
    "build_spatial_windows",
    "save_spatial_collection",
    "load_spatial_collection",
    "summarize_spatial_collection",
    "trajectory_window_from_dict",
]

#: Canonical serialized field names for one window (SpatialTrajectoryWindow
#: field names plus identity). No legacy aliases are accepted on load.
_WINDOW_FIELDS: tuple[str, ...] = (
    "rollout_id",
    "episode_id",
    "proprio_history",
    "action_sequence",
    "delta_trajectory",
    "commands",
    "step_indices",
)

#: Forbidden legacy aliases that must never appear in serialized windows.
_LEGACY_FIELD_ALIASES: tuple[str, ...] = ("obs_history", "actions", "command")

#: Valid condition phases recorded per window. ``condition`` itself is
#: optional: windows collected before the label existed stay ``None`` and are
#: never promoted to a nominal label.
_WINDOW_CONDITION_PHASES: frozenset[str] = frozenset({"stable", "transitioning", "unknown"})

#: Essential metadata keys every spatial collection must carry. The writer
#: additionally injects ``history_length`` and ``horizon``.
SPATIAL_METADATA_REQUIRED_KEYS: tuple[str, ...] = (
    "checkpoint_path",
    "checkpoint_sha256",
    "task",
    "backend",
    "ctrl_dt",
    "obs_dim",
    "act_dim",
    "proprio_layout",
    "joint_names",
    "action_scale",
    "default_joint_angles",
    "action_representation",
    "quaternion_order",
    "coordinate_frame",
    "seed",
)


# ---------------------------------------------------------------------------
# Window construction
# ---------------------------------------------------------------------------


def _is_boundary(transition: SpatialTransition) -> bool:
    return bool(transition.terminated) or bool(transition.truncated)


def _validate_transition(name: str, transition: SpatialTransition) -> None:
    """Fail fast on corrupted transitions before they poison a window."""
    if not isinstance(transition, SpatialTransition):
        raise TypeError(
            f"transitions must yield SpatialTransition records, got "
            f"{type(transition).__name__} ({name})."
        )
    tensor_shapes = (
        ("proprio", (DEFAULT_OBS_DIM,)),
        ("action", (29,)),
        ("commands", (3,)),
        ("position", (3,)),
        ("quaternion", (4,)),
        ("next_position", (3,)),
        ("next_quaternion", (4,)),
    )
    for field_name, expected in tensor_shapes:
        tensor = getattr(transition, field_name)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"{name}.{field_name} must be a torch.Tensor, got "
                f"{type(tensor).__name__}."
            )
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"{name}.{field_name} has shape {tuple(tensor.shape)}, "
                f"expected {expected}."
            )
        if not torch.isfinite(tensor).all().item():
            raise ValueError(
                f"{name}.{field_name} contains NaN or Inf values; refusing "
                f"to build a window from corrupted input."
            )
    _validate_condition(name, transition.condition)


def _window_condition(pending: Sequence[SpatialTransition]) -> dict[str, object]:
    """Aggregate the actual per-step payload labels spanning a whole window.

    ``pending`` covers every state the window touches (history through the
    last future pose). A window is ``stable`` only when every step carries a
    concrete stable label and all of them agree on active flag and added mass.
    Any attach/detach inside the window is ``transitioning``. Missing truth
    stays ``unknown`` and is never filled from the configured payload block.
    """
    labels = [transition.condition for transition in pending]
    phases = [None if label is None else label.get("phase") for label in labels]

    if "transitioning" in phases:
        return {"phase": "transitioning"}
    if any(phase != "stable" for phase in phases):
        return {"phase": "unknown"}

    states = {
        (bool(label["payload_active"]), float(label["added_mass_kg"]))
        for label in labels
        if label is not None
    }
    if len(states) != 1:
        return {"phase": "transitioning"}
    payload_active, added_mass = states.pop()
    return {"phase": "stable", "payload_active": payload_active, "added_mass_kg": added_mass}


def _make_window(
    pending: list[SpatialTransition],
    history_length: int,
    horizon: int,
) -> dict[str, Any]:
    """Materialize one canonical window dict from a full pending buffer.

    ``pending`` holds exactly ``history_length + horizon - 1`` consecutive
    transitions of one episode; the origin is ``pending[history_length-1]``.
    """
    origin = pending[history_length - 1]
    history = pending[:history_length]
    future = pending[history_length - 1 : history_length - 1 + horizon]

    proprio_history = torch.stack([t.proprio for t in history])
    action_sequence = torch.stack([t.action for t in future])
    next_positions = torch.stack([t.next_position for t in future])
    next_quats = torch.stack([t.next_quaternion for t in future])

    delta_trajectory = compute_se2_deltas(
        next_positions.unsqueeze(0),
        next_quats.unsqueeze(0),
        pos0=origin.position.unsqueeze(0),
        quat0=origin.quaternion.unsqueeze(0),
    ).squeeze(0)

    first_step = pending[0].episode_step
    step_indices = torch.arange(
        first_step,
        first_step + history_length + horizon,
        dtype=torch.long,
    )

    window = SpatialTrajectoryWindow(
        proprio_history=proprio_history,
        action_sequence=action_sequence,
        delta_trajectory=delta_trajectory,
        commands=origin.commands,
        step_indices=step_indices,
    )
    validate_trajectory_window(
        window,
        expected_obs_dim=DEFAULT_OBS_DIM,
        expected_act_dim=29,
        expected_horizon=horizon,
        expected_hist=history_length,
    )

    return {
        "rollout_id": origin.rollout_id,
        "episode_id": int(origin.episode_id),
        "proprio_history": window.proprio_history,
        "action_sequence": window.action_sequence,
        "delta_trajectory": window.delta_trajectory,
        "commands": window.commands,
        "step_indices": window.step_indices,
        "condition": _window_condition(pending),
    }


def build_spatial_windows(
    transitions: Iterable[SpatialTransition],
    *,
    history_length: int = 15,
    horizon: int = 15,
) -> list[dict]:
    """Build canonical spatial trajectory windows from a transition stream.

    Args:
        transitions: Iterable of :class:`SpatialTransition` records.
            Records of one episode must arrive in consecutive
            ``episode_step`` order; different episodes/rollouts may be
            interleaved freely (only consecutive records within the same
            rollout/episode chain together).
        history_length: Number of history observations ``H_hist`` per
            window; must be >= 1.
        horizon: Number of future actions/poses ``H`` per window; must be
            >= 1.

    Returns:
        List of canonical window dicts. Each dict contains the
        :class:`SpatialTrajectoryWindow` field names (``proprio_history``,
        ``action_sequence``, ``delta_trajectory``, ``commands``,
        ``step_indices``) plus ``rollout_id`` (str), ``episode_id``
        (int), and ``condition``, the payload phase actually recorded
        across the window's steps (``stable`` with ``payload_active``/
        ``added_mass_kg``, ``transitioning``, or ``unknown`` when the env
        published no truth). ``step_indices`` is a ``torch.long`` tensor of
        length ``history_length + horizon`` covering the first history state
        through the last future state inclusive.

    Raises:
        ValueError: If ``history_length``/``horizon`` are invalid or a
            transition is corrupted (bad shape, NaN/Inf).
        TypeError: If a yielded record is not a :class:`SpatialTransition`.
    """
    if history_length < 1:
        raise ValueError(f"history_length must be >= 1, got {history_length}.")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}.")

    capacity = history_length + horizon - 1
    pending: deque[SpatialTransition] = deque(maxlen=capacity)
    windows: list[dict] = []

    for transition in transitions:
        _validate_transition("transition", transition)

        # Terminal/truncated records are pure boundaries: never history,
        # never origin, never a future label. Drop and reset the chain.
        if _is_boundary(transition):
            pending.clear()
            continue

        if pending:
            tail = pending[-1]
            same_episode = (
                transition.rollout_id == tail.rollout_id
                and transition.episode_id == tail.episode_id
            )
            consecutive = transition.episode_step == tail.episode_step + 1
            # Discard incomplete pending context on any chain break rather
            # than connecting unrelated segments.
            if not same_episode or not consecutive:
                pending.clear()

        pending.append(transition)

        if len(pending) == capacity:
            windows.append(_make_window(list(pending), history_length, horizon))

    return windows


# ---------------------------------------------------------------------------
# Collection summary (read-only diagnostics)
# ---------------------------------------------------------------------------


def _window_rows(sequence: torch.Tensor, *, width: int, name: str) -> torch.Tensor:
    """Normalize a per-window tensor to ``(rows, width)``; a single vector is one row."""
    tensor = torch.as_tensor(sequence, dtype=torch.float64)
    if tensor.dim() == 1:
        tensor = tensor.reshape(1, -1)
    if tensor.dim() != 2 or tensor.shape[1] != width:
        raise ValueError(f"{name} must have shape (rows, {width}), got {tuple(torch.as_tensor(sequence).shape)}.")
    return tensor


def _joint_action_stats(action_sequences: Sequence[torch.Tensor]) -> dict[str, list[float]]:
    """Per-joint min/max/std over the action rows present in the windows.

    Overlapping windows repeat the same executed steps, so these statistics
    describe window contents, not independent action samples.
    """
    stacked = torch.cat(
        [_window_rows(seq, width=29, name="window.action_sequence") for seq in action_sequences]
    )
    return {
        "min": stacked.min(dim=0).values.tolist(),
        "max": stacked.max(dim=0).values.tolist(),
        "std": stacked.std(dim=0, unbiased=False).tolist(),
    }


def _command_range(commands: Sequence[torch.Tensor]) -> dict[str, list[float]] | None:
    if not commands:
        return None
    stacked = torch.cat([_window_rows(cmd, width=3, name="window.commands") for cmd in commands])
    return {
        "min": stacked.min(dim=0).values.tolist(),
        "max": stacked.max(dim=0).values.tolist(),
    }


def _context_summary(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Report provenance without inferring missing facts.

    ``collection_context`` is the configured condition block of the session;
    when it is absent (older artifacts) it is reported as missing rather than
    assumed nominal, and a missing ``label_source`` stays ``"unspecified"``.
    """
    context = metadata.get("collection_context")
    commands: dict[str, Any] = {}
    if isinstance(context, Mapping):
        configured = context.get("commands")
        if isinstance(configured, Mapping):
            commands = dict(configured)

    label_source = metadata.get("label_source")
    seed = metadata.get("seed")
    rollout_seeds = metadata.get("rollout_seeds")
    rollout_ids = metadata.get("rollout_ids")
    return {
        "label_source": label_source if isinstance(label_source, str) and label_source else "unspecified",
        "base_seed": int(seed) if isinstance(seed, (int, np.integer)) and not isinstance(seed, bool) else None,
        "rollout_seeds": [int(value) for value in rollout_seeds] if isinstance(rollout_seeds, (list, tuple)) else None,
        "rollout_ids": [str(value) for value in rollout_ids] if isinstance(rollout_ids, (list, tuple)) else None,
        "collection_context_present": isinstance(context, Mapping),
        "configured_vel_limit": commands.get("vel_limit"),
        "configured_heading_command": commands.get("heading_command"),
        "configured_heading_range": commands.get("heading_range"),
        "configured_resampling_time": commands.get("resampling_time"),
        "configured_rel_standing_envs": commands.get("rel_standing_envs"),
        "runtime_payload": context.get("runtime_payload") if isinstance(context, Mapping) else None,
        "domain_rand": context.get("domain_rand") if isinstance(context, Mapping) else None,
        "noise_config": context.get("noise_config") if isinstance(context, Mapping) else None,
        "context_note": (
            "Describes the configured conditions used to collect these windows; "
            "it does not assert that a runtime payload was attached inside each window."
        ),
    }


def _condition_counts(windows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count windows by their actual recorded payload phase.

    Windows without a label are ``unlabeled``; they are never folded into a
    nominal or payload-free bucket. ``stable_payload``/``stable_nominal`` only
    count windows whose executed steps actually recorded that state.
    """
    phases = {"stable_payload": 0, "stable_nominal": 0, "transitioning": 0, "unknown": 0, "unlabeled": 0}
    for window in windows:
        condition = window.get("condition")
        if not isinstance(condition, Mapping):
            phases["unlabeled"] += 1
            continue
        phase = condition.get("phase")
        if phase == "stable":
            phases["stable_payload" if condition.get("payload_active") else "stable_nominal"] += 1
        elif phase == "transitioning":
            phases["transitioning"] += 1
        else:
            phases["unknown"] += 1
    return phases


def summarize_spatial_collection(
    windows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize collection coverage without touching windows, labels, or actions.

    Pure and read-only: it never reorders, drops, or rewrites a window, and
    it never converts overlapping windows into independent samples.

    Args:
        windows: Canonical window dicts as produced by
            :func:`build_spatial_windows`.
        metadata: Collection metadata (the exported envelope metadata).

    Returns:
        Primitive JSON-safe dict with the independent
        ``(rollout_id, episode_id)`` group count, rollout count, window
        count, per-joint action ``min``/``max``/``std`` over window action
        rows, the observed command range, and ``label_source``/
        ``collection_context`` provenance. Missing provenance is reported as
        ``"unspecified"``/``None``, never inferred as nominal or SLAM truth.
        ``window_condition_counts`` tallies the actual per-window payload
        phase labels (``unlabeled`` when an older artifact has none).
        ``train_val_sufficient`` is ``False`` below two groups and
        ``warning`` then states that no random per-window fallback exists.
    """
    if not isinstance(metadata, Mapping):
        raise TypeError(f"metadata must be a mapping, got {type(metadata).__name__}.")

    groups: set[tuple[str, int]] = set()
    rollout_ids: set[str] = set()
    action_sequences: list[torch.Tensor] = []
    commands: list[torch.Tensor] = []
    for index, window in enumerate(windows):
        if not isinstance(window, Mapping):
            raise TypeError(f"windows[{index}] must be a mapping, got {type(window).__name__}.")
        for key in ("rollout_id", "episode_id", "action_sequence", "commands"):
            if key not in window:
                raise ValueError(f"windows[{index}] is missing required field {key!r}.")
        rollout_id = str(window["rollout_id"])
        groups.add((rollout_id, int(window["episode_id"])))
        rollout_ids.add(rollout_id)
        action_sequences.append(window["action_sequence"])
        commands.append(window["commands"])

    group_count = len(groups)
    summary: dict[str, Any] = {
        "window_count": len(windows),
        "group_count": group_count,
        "rollout_count": len(rollout_ids),
        "groups": [{"rollout_id": rid, "episode_id": eid} for rid, eid in sorted(groups)],
        "joint_action": _joint_action_stats(action_sequences) if action_sequences else None,
        "observed_command_range": _command_range(commands),
        "action_rows_are_independent": False,
        "action_rows_note": (
            "Per-joint statistics are computed over window action rows; "
            "overlapping windows repeat executed steps, so rows are not "
            "independent action samples."
        ),
        "window_condition_counts": _condition_counts(windows),
        "condition_note": (
            "Window conditions come from the runtime_payload truth recorded "
            "during collection; a stable label is not evidence that the "
            "configured payload was attached for any other window."
        ),
        "train_val_sufficient": group_count >= 2,
    }
    if group_count < 2:
        summary["warning"] = (
            f"Only {group_count} independent (rollout_id, episode_id) group(s); "
            "a train/validation group split needs at least 2. There is no "
            "random per-window fallback, so collect more rollouts."
        )
    else:
        summary["warning"] = None
    summary["provenance"] = _context_summary(metadata)
    return summary


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _validate_metadata(metadata: Mapping[str, Any]) -> None:
    if not isinstance(metadata, Mapping):
        raise TypeError(
            f"metadata must be a mapping, got {type(metadata).__name__}."
        )
    missing = [key for key in SPATIAL_METADATA_REQUIRED_KEYS if key not in metadata]
    if missing:
        raise ValueError(
            f"metadata is missing required keys: {sorted(missing)}."
        )
    if metadata["action_representation"] != SPATIAL_ACTION_REPRESENTATION:
        raise ValueError(
            f"metadata['action_representation'] must be "
            f"{SPATIAL_ACTION_REPRESENTATION!r}, got "
            f"{metadata['action_representation']!r}."
        )
    if "history_length" not in metadata or "horizon" not in metadata:
        raise ValueError(
            "metadata must include 'history_length' and 'horizon'."
        )
    if int(metadata["history_length"]) < 1 or int(metadata["horizon"]) < 1:
        raise ValueError(
            f"metadata history_length/horizon must be >= 1, got "
            f"{metadata['history_length']}/{metadata['horizon']}."
        )
    if int(metadata["obs_dim"]) != DEFAULT_OBS_DIM:
        raise ValueError(
            f"metadata['obs_dim'] must be {DEFAULT_OBS_DIM}, got "
            f"{metadata['obs_dim']}."
        )
    if int(metadata["act_dim"]) != 29:
        raise ValueError(
            f"metadata['act_dim'] must be 29, got {metadata['act_dim']}."
        )
    joint_names = metadata["joint_names"]
    if len(joint_names) != int(metadata["act_dim"]):
        raise ValueError(
            f"metadata['joint_names'] must list {metadata['act_dim']} joints "
            f"in target order, got {len(joint_names)}."
        )
    default_angles = metadata["default_joint_angles"]
    if isinstance(default_angles, torch.Tensor):
        if tuple(default_angles.shape) != (int(metadata["act_dim"]),):
            raise ValueError(
                f"metadata['default_joint_angles'] tensor must have shape "
                f"({metadata['act_dim']},), got {tuple(default_angles.shape)}."
            )
    elif len(default_angles) != int(metadata["act_dim"]):
        raise ValueError(
            f"metadata['default_joint_angles'] must have "
            f"{metadata['act_dim']} entries, got {len(default_angles)}."
        )

    # Optional provenance: when present it must already be JSON primitives so
    # the artifact stays loadable with ``weights_only=True``. Absent context is
    # legal and stays absent; it is never filled with a nominal default.
    for key in ("collection_context", "rollout_ids", "rollout_seeds"):
        if key not in metadata:
            continue
        _validate_primitive_payload(f"metadata[{key!r}]", metadata[key])
    if "collection_context" in metadata and not isinstance(metadata["collection_context"], Mapping):
        raise TypeError(
            "metadata['collection_context'] must be a mapping when present, got "
            f"{type(metadata['collection_context']).__name__}."
        )


def _validate_primitive_payload(name: str, value: Any) -> None:
    """Reject non-primitive provenance so ``torch.load(weights_only=True)`` stays valid."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{name} keys must be strings, got {type(key).__name__}.")
            _validate_primitive_payload(f"{name}[{key!r}]", item)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_primitive_payload(f"{name}[{index}]", item)
        return
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{name} must not contain non-finite numbers.")
        return
    raise TypeError(
        f"{name} must contain JSON primitives, got {type(value).__name__}."
    )


def _validate_condition(name: str, condition: Any) -> dict[str, object] | None:
    """Validate an optional per-window payload label without inventing one."""
    if condition is None:
        return None
    if not isinstance(condition, Mapping):
        raise TypeError(
            f"{name}.condition must be a mapping or None, got {type(condition).__name__}."
        )
    phase = condition.get("phase")
    if phase not in _WINDOW_CONDITION_PHASES:
        raise ValueError(
            f"{name}.condition['phase'] must be one of "
            f"{sorted(_WINDOW_CONDITION_PHASES)}, got {phase!r}."
        )
    if phase != "stable":
        return {"phase": phase}
    payload_active = condition.get("payload_active")
    added_mass = condition.get("added_mass_kg")
    if not isinstance(payload_active, bool):
        raise TypeError(
            f"{name}.condition['payload_active'] must be a bool for a stable phase."
        )
    if isinstance(added_mass, bool) or not isinstance(added_mass, (int, float)):
        raise TypeError(
            f"{name}.condition['added_mass_kg'] must be a number for a stable phase."
        )
    if not np.isfinite(float(added_mass)):
        raise ValueError(f"{name}.condition['added_mass_kg'] must be finite.")
    return {"phase": "stable", "payload_active": payload_active, "added_mass_kg": float(added_mass)}


def _validate_window_dict(
    name: str,
    window: Mapping[str, Any],
    history_length: int,
    horizon: int,
) -> dict[str, object] | None:
    """Validate one canonical window dict against format and geometry.

    Returns the normalized optional ``condition`` label (``None`` when absent)
    so callers persist exactly what was validated.
    """
    if not isinstance(window, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(window).__name__}.")

    for alias in _LEGACY_FIELD_ALIASES:
        if alias in window:
            raise ValueError(
                f"{name} uses legacy field {alias!r}; canonical fields are "
                f"{_WINDOW_FIELDS}. Rebuild the artifact with the current "
                f"collector."
            )
    missing = [key for key in _WINDOW_FIELDS if key not in window]
    if missing:
        raise ValueError(f"{name} is missing required fields: {sorted(missing)}.")

    if not isinstance(window["rollout_id"], str):
        raise TypeError(
            f"{name}.rollout_id must be a str, got "
            f"{type(window['rollout_id']).__name__}."
        )
    if not isinstance(window["episode_id"], int):
        raise TypeError(
            f"{name}.episode_id must be an int, got "
            f"{type(window['episode_id']).__name__}."
        )

    step_indices = window["step_indices"]
    if isinstance(step_indices, torch.Tensor):
        if step_indices.dim() != 1:
            raise ValueError(
                f"{name}.step_indices must be a rank-1 tensor, got rank "
                f"{step_indices.dim()}."
            )
        length = step_indices.shape[0]
    elif isinstance(step_indices, tuple):
        length = len(step_indices)
    else:
        raise TypeError(
            f"{name}.step_indices must be a torch.Tensor or tuple[int, ...], "
            f"got {type(step_indices).__name__}."
        )
    if length != history_length + horizon:
        raise ValueError(
            f"{name}.step_indices must have length "
            f"{history_length + horizon} (first history state through last "
            f"future state inclusive), got {length}."
        )

    # Reuse the shared validator for shapes, finiteness, and yaw range.
    validate_trajectory_window(
        trajectory_window_from_dict(window),
        expected_obs_dim=DEFAULT_OBS_DIM,
        expected_act_dim=29,
        expected_horizon=horizon,
        expected_hist=history_length,
    )
    return _validate_condition(name, window.get("condition"))


def trajectory_window_from_dict(window: Mapping[str, Any]) -> SpatialTrajectoryWindow:
    """Convert a canonical window dict to a :class:`SpatialTrajectoryWindow`.

    Identity fields (``rollout_id``, ``episode_id``) are not part of the
    dataclass; they are retained in the dict for grouped splitting.

    Args:
        window: Mapping with the canonical field names
            ``proprio_history``, ``action_sequence``, ``delta_trajectory``,
            ``commands``, and ``step_indices``.

    Returns:
        The corresponding :class:`SpatialTrajectoryWindow` (unvalidated;
        call :func:`validate_trajectory_window` for checks).
    """
    return SpatialTrajectoryWindow(
        proprio_history=window["proprio_history"],
        action_sequence=window["action_sequence"],
        delta_trajectory=window["delta_trajectory"],
        commands=window["commands"],
        step_indices=window["step_indices"],
    )


def save_spatial_collection(
    path: str | Path,
    windows: list[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    *,
    history_length: int = 15,
    horizon: int = 15,
) -> None:
    """Save spatial trajectory windows to the canonical on-disk format.

    The envelope is ``{"format_version": 1, "metadata": ..., "windows":
    [...]}`` containing only tensors and primitives, loadable with
    ``torch.load(..., weights_only=True)``.

    Args:
        path: Destination file path.
        windows: Non-empty list of canonical window dicts as produced by
            :func:`build_spatial_windows`. An optional ``condition`` key is
            validated and preserved; windows without one are saved as
            ``None`` instead of a nominal label.
        metadata: Collection metadata. Must contain every key in
            :data:`SPATIAL_METADATA_REQUIRED_KEYS` with
            ``action_representation == "env_input"``; this function injects
            ``history_length`` and ``horizon``.
        history_length: History length ``H_hist`` the windows were built
            with.
        horizon: Future horizon ``H`` the windows were built with.

    Raises:
        ValueError: If ``windows`` is empty, a window is malformed, or the
            metadata violates the shared contract. Raised before the output
            artifact is created.
        TypeError: If a field has an unexpected type.
    """
    if history_length < 1:
        raise ValueError(f"history_length must be >= 1, got {history_length}.")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}.")
    if not windows:
        raise ValueError(
            "save_spatial_collection requires a non-empty window list; "
            "refusing to create an empty artifact."
        )

    full_metadata = dict(metadata)
    full_metadata["history_length"] = int(history_length)
    full_metadata["horizon"] = int(horizon)
    _validate_metadata(full_metadata)

    serialized_windows: list[dict[str, Any]] = []
    for index, window in enumerate(windows):
        condition = _validate_window_dict(
            f"windows[{index}]", window, history_length, horizon
        )
        serialized_windows.append(
            {
                "rollout_id": window["rollout_id"],
                "episode_id": int(window["episode_id"]),
                "proprio_history": torch.as_tensor(window["proprio_history"]).cpu(),
                "action_sequence": torch.as_tensor(window["action_sequence"]).cpu(),
                "delta_trajectory": torch.as_tensor(window["delta_trajectory"]).cpu(),
                "commands": torch.as_tensor(window["commands"]).cpu(),
                "step_indices": torch.as_tensor(window["step_indices"]).cpu(),
                # Absent labels stay absent; never promote them to nominal.
                "condition": condition,
            }
        )

    envelope = {
        "format_version": SPATIAL_DATA_FORMAT_VERSION,
        "metadata": full_metadata,
        "windows": serialized_windows,
    }

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(envelope, target)


def load_spatial_collection(path: str | Path) -> dict:
    """Load a canonical spatial collection saved by :func:`save_spatial_collection`.

    Args:
        path: Artifact path.

    Returns:
        Dict with keys ``format_version`` (int), ``metadata`` (dict,
        including ``history_length``/``horizon``), and ``windows`` (list of
        canonical window dicts retaining ``rollout_id``/``episode_id`` for
        grouped splitting).

    Raises:
        TypeError: If the payload or a field has an unexpected type.
        ValueError: If the format version, action representation, metadata,
            or a window violates the current contract. Legacy field names
            are never silently accepted.
    """
    # Map CPU storages instead of eagerly allocating each archive storage.
    payload = torch.load(Path(path), map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict):
        raise TypeError(
            f"spatial collection must be a dict, got {type(payload).__name__}."
        )
    if payload.get("format_version") != SPATIAL_DATA_FORMAT_VERSION:
        raise ValueError(
            f"unsupported spatial data format_version "
            f"{payload.get('format_version')!r}; expected "
            f"{SPATIAL_DATA_FORMAT_VERSION}."
        )

    metadata = payload.get("metadata")
    _validate_metadata(metadata)

    history_length = int(metadata["history_length"])
    horizon = int(metadata["horizon"])

    windows = payload.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError(
            "spatial collection 'windows' must be a non-empty list."
        )
    for index, window in enumerate(windows):
        _validate_window_dict(
            f"windows[{index}]", window, history_length, horizon
        )

    return {
        "format_version": SPATIAL_DATA_FORMAT_VERSION,
        "metadata": dict(metadata),
        "windows": list(windows),
    }
