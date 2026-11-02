"""Offline spatial trajectory dataset and episode-grouped splitting.

Loads canonical spatial collections (see :mod:`.collection`) exactly once,
pins their data contract, and splits windows by ``(rollout_id, episode_id)``
groups so windows from one episode never straddle train/validation.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset, Subset

from .checkpoint import data_contract_from_metadata
from .collection import load_spatial_collection, trajectory_window_from_dict
from .contract import SpatialTrajectoryWindow

__all__ = [
    "SpatialTrajectoryDataset",
    "make_group_split",
    "UNKNOWN_LABEL_SOURCE",
    "UNSPECIFIED_COLLECTION_CONTEXT",
]

#: Placeholder for collections that never recorded how their labels were made.
UNKNOWN_LABEL_SOURCE = "unknown"

#: Placeholder for collections that carry no ``collection_context`` metadata.
UNSPECIFIED_COLLECTION_CONTEXT = "unspecified"


def _canonical_json(value: Any) -> Any:
    """JSON round-trip normalization matching checkpoint canonicalization."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _json_primitive(value: Any) -> Any:
    """Normalize supported metadata without inventing a condition for malformed data."""
    if isinstance(value, Mapping):
        return {str(key): _json_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_primitive(item) for item in value]
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Collection provenance must not contain non-finite numbers")
        return value
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_primitive(tolist())
    raise TypeError(f"Collection provenance must contain JSON primitives, got {type(value).__name__}")


def _condition_key(label_source: str, context: Any) -> str:
    """Stable group key from label source plus JSON-normalized context."""
    if context == UNSPECIFIED_COLLECTION_CONTEXT:
        return f"{label_source}|{UNSPECIFIED_COLLECTION_CONTEXT}"
    return f"{label_source}|{json.dumps(context, sort_keys=True, allow_nan=False, separators=(',', ':'))}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SpatialTrajectoryDataset(Dataset):
    """In-memory dataset over canonical spatial collection files.

    All files are loaded once at construction, must share an identical
    :func:`data_contract_from_metadata` contract, and are pinned by
    ``(resolved path, sha256)`` so resumed runs verify they see the exact
    same bytes. Policy checkpoint provenance stays in each file's metadata
    and is *not* part of the model I/O contract.

    Attributes:
        data_contract: Canonical policy-independent I/O semantics dict.
        data_identity: ``{"files": [{"path", "sha256"}, ...]}`` in the
            canonical resolved-path order.
        windows: Canonical window dicts (with ``rollout_id`` /
            ``episode_id``) concatenated in canonical resolved-path order.
        file_metadata: Per-file provenance metadata, parallel to
            ``data_identity["files"]``; the descriptive extras only, never
            part of the model I/O contract.
        window_sources: Per-window index into ``file_metadata`` naming the
            collection file each window came from, parallel to ``windows``.
        condition_groups: Per-window condition key index into
            ``conditions``; windows with identical
            ``(label_source, collection_context)`` share a group.
        conditions: Ordered condition records with ``condition`` (the
            stable key), ``label_source`` and ``collection_context``.
            Missing provenance is reported as ``"unknown"`` /
            ``"unspecified"``, never guessed as nominal or SLAM.
    """

    def __init__(self, paths: Sequence[str | Path]) -> None:
        if isinstance(paths, (str, Path)) or not paths:
            raise ValueError(
                "SpatialTrajectoryDataset requires a non-empty sequence of "
                "collection paths."
            )

        resolved = sorted(Path(p).expanduser().resolve() for p in paths)
        counts = Counter(resolved)
        duplicates = sorted(str(p) for p, n in counts.items() if n > 1)
        if duplicates:
            raise ValueError(
                f"Duplicate collection paths are ambiguous: {duplicates}."
            )
        for path in resolved:
            if not path.is_file():
                raise FileNotFoundError(f"Spatial collection not found: {path}")

        files: list[dict[str, str]] = []
        windows: list[dict[str, Any]] = []
        file_metadata: list[dict[str, Any]] = []
        window_sources: list[int] = []
        condition_groups: list[int] = []
        conditions: list[dict[str, Any]] = []
        condition_index: dict[str, int] = {}
        contract: dict[str, Any] | None = None
        for file_index, path in enumerate(resolved):
            collection = load_spatial_collection(path)
            file_contract = _canonical_json(
                data_contract_from_metadata(collection["metadata"])
            )
            if contract is None:
                contract = file_contract
            elif file_contract != contract:
                raise ValueError(
                    f"Data contract mismatch in {path}: all collections must "
                    f"share an identical data contract."
                )
            metadata = collection["metadata"]
            label_source = _json_primitive(metadata.get("label_source"))
            if not isinstance(label_source, str) or not label_source:
                label_source = UNKNOWN_LABEL_SOURCE
            context = (
                _json_primitive(metadata["collection_context"])
                if metadata.get("collection_context") is not None
                else UNSPECIFIED_COLLECTION_CONTEXT
            )
            files.append({"path": str(path), "sha256": _sha256_file(path)})
            file_metadata.append(_json_primitive(metadata))
            for window in collection["windows"]:
                observed = window.get("condition")
                grouped_context = {
                    "configuration": context,
                    "observed_window": _json_primitive(observed) if observed is not None else "unspecified",
                }
                key = _condition_key(label_source, grouped_context)
                if key not in condition_index:
                    condition_index[key] = len(conditions)
                    conditions.append({"condition": key, "label_source": label_source,
                                       "collection_context": grouped_context})
                windows.append(window)
                window_sources.append(file_index)
                condition_groups.append(condition_index[key])

        self.data_contract: dict[str, Any] = contract if contract is not None else {}
        self.data_identity: dict[str, Any] = {"files": files}
        self.windows: list[dict[str, Any]] = windows
        self.file_metadata: list[dict[str, Any]] = file_metadata
        self.window_sources: list[int] = window_sources
        self.condition_groups: list[int] = condition_groups
        self.conditions: list[dict[str, Any]] = conditions

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> SpatialTrajectoryWindow:
        return trajectory_window_from_dict(self.windows[index])


def _group_key(window: Mapping[str, Any]) -> tuple[str, int]:
    return (str(window["rollout_id"]), int(window["episode_id"]))


def _group_list(key: tuple[str, int]) -> list[Any]:
    return [key[0], key[1]]


def make_group_split(
    dataset: SpatialTrajectoryDataset,
    *,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[Subset, Subset, dict[str, Any]]:
    """Split the dataset into train/validation subsets by episode group.

    Groups are the set of ``(rollout_id, episode_id)`` pairs present in the
    dataset (never individual windows or files), sorted into a canonical
    order and shuffled with a seeded :class:`random.Random`, so the split
    is deterministic and independent of input path ordering.

    Returns:
        ``(train_subset, val_subset, split)`` of
        :class:`torch.utils.data.Subset` views (use ``.indices`` for the
        underlying window indices) plus ``split``, a primitive JSON-safe
        dict with ``seed``, ``validation_fraction``, ``train_groups`` and
        ``validation_groups`` as ``[rollout_id, episode_id]`` lists.

    Raises:
        TypeError: If ``dataset`` is not a :class:`SpatialTrajectoryDataset`.
        ValueError: If the fraction is not in ``(0, 1)``, fewer than two
            groups exist, or a side would be empty. There is no
            random-window fallback.
    """
    if not isinstance(dataset, SpatialTrajectoryDataset):
        raise TypeError(
            f"dataset must be a SpatialTrajectoryDataset, got "
            f"{type(dataset).__name__}."
        )
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError(
            f"validation_fraction must lie in (0, 1), got {validation_fraction}."
        )

    groups = sorted({_group_key(window) for window in dataset.windows})
    if len(groups) < 2:
        raise ValueError(
            f"Group split requires at least 2 (rollout_id, episode_id) "
            f"groups, got {len(groups)}."
        )

    shuffled = list(groups)
    random.Random(int(seed)).shuffle(shuffled)
    n_val = int(round(len(shuffled) * float(validation_fraction)))
    n_val = min(max(n_val, 1), len(shuffled) - 1)
    validation_groups = shuffled[:n_val]
    train_groups = shuffled[n_val:]

    val_set = set(validation_groups)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for index, window in enumerate(dataset.windows):
        (val_indices if _group_key(window) in val_set else train_indices).append(index)
    if not train_indices or not val_indices:
        raise ValueError(
            "Group split produced an empty side; adjust validation_fraction."
        )

    split: dict[str, Any] = {
        "seed": int(seed),
        "validation_fraction": float(validation_fraction),
        "train_groups": [_group_list(key) for key in train_groups],
        "validation_groups": [_group_list(key) for key in validation_groups],
    }
    _canonical_json(split)  # fail fast if anything is not primitive JSON
    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        split,
    )
