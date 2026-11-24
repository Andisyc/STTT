"""Bounded, growing episode replay using the archive's causal crop implementation."""
from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from .pretraining_data import PretrainingBatch, _json_copy, _positive_integer, collate_pretraining_sequences
from .sequence_data import _arrays, _contract


class ReplayVersionError(ValueError):
    """Retain the exact normalized metadata rejected by the replay boundary."""

    def __init__(self, identity: str, action_count: int, metadata: dict[str, Any], detail: str):
        super().__init__(f"replay versions invalid: episode={identity!r}, actions={action_count}, {detail}")
        self.metadata = metadata
        self.action_count = action_count
        self.snapshot_path: str | None = None


class EpisodeReplay:
    def __init__(self, contract: Mapping[str, Any], *, capacity_transitions: int,
                 rollout_windows: int, seed: int = 0, device: Any = "cpu"):
        self.contract = _contract(contract)
        self.capacity = _positive_integer(capacity_transitions, "capacity_transitions")
        self.rollout_windows = _positive_integer(rollout_windows, "rollout_windows")
        self.window_steps = _positive_integer(self.contract["data"]["window_steps"], "window_steps")
        if type(seed) is not int or seed < 0:
            raise ValueError("replay seed must be a nonnegative integer")
        self.device = device
        self._episodes: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self._metadata: dict[str, dict[str, Any]] = {}
        self._retired: set[str] = set()
        self._transitions = self.total_collected = 0
        self._rng = np.random.default_rng(seed)
        self._keys: list[str] = []
        self._ends: list[int] = []
        self._dirty = True

    @property
    def transitions(self) -> int:
        return self._transitions

    @property
    def num_starts(self) -> int:
        self._index()
        return self._ends[-1] if self._ends else 0

    def update_episode(self, identity: str, sequence: Mapping[str, Any], *, metadata: Mapping[str, Any] | None = None) -> None:
        if not isinstance(identity, str) or not identity or identity in self._retired:
            raise ValueError("replay requires a new or active nonempty episode identity")
        raw = _arrays(sequence, self.contract)
        count = len(raw["actions"])
        if count > self.capacity or not raw["is_first"][0]:
            raise ValueError("replay requires a reset-rooted episode prefix that fits its capacity")
        period = round(self.contract["data"]["ctrl_dt"] * 1e9)
        if np.any(np.abs(np.diff(raw["timestamps_ns"]) - period) > 1):
            raise ValueError("replay physical periods differ from the captured execution contract")
        previous = self._episodes.get(identity)
        old_count = 0 if previous is None else len(previous["actions"])
        if previous is not None:
            if count < old_count or count > old_count and previous["terminal"].any():
                raise ValueError("replay episode prefixes cannot shrink or grow across a terminal transition")
            for key, values in previous.items():
                if not np.array_equal(raw[key][:len(values)], values):
                    raise ValueError(f"replay update changed previously observed evidence: {key}")
        provenance = _json_copy(dict(metadata)) if metadata is not None else self._metadata.get(identity, {})
        if "versions_actor_wm" in provenance:
            versions = provenance["versions_actor_wm"]
            if not isinstance(versions, list):
                raise ReplayVersionError(identity, count, provenance,
                                         f"expected version rows, got {type(versions).__name__}")
            if len(versions) != count:
                raise ReplayVersionError(identity, count, provenance, f"version_rows={len(versions)}")
            for step, pair in enumerate(versions):
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ReplayVersionError(identity, count, provenance,
                                             f"step={step}, expected actor/WM pair, got {pair!r}")
                for learner, version in zip(("actor", "world_model"), pair):
                    if type(version) is not int or version < 0:
                        raise ReplayVersionError(identity, count, provenance,
                            f"step={step}, learner={learner}, value={version!r}, type={type(version).__name__}")
        if previous is not None and identity in self._metadata:
            old_metadata = self._metadata[identity]
            if {key: value for key, value in old_metadata.items() if key != "versions_actor_wm"} != {key: value for key, value in provenance.items() if key != "versions_actor_wm"}:
                raise ValueError("replay physical episode provenance changed")
            if "versions_actor_wm" in old_metadata and provenance.get("versions_actor_wm", [])[:old_count] != old_metadata["versions_actor_wm"]:
                raise ValueError("replay changed the learner versions of past physical actions")
        self._metadata[identity] = provenance
        self._episodes[identity] = {key: value.copy() for key, value in raw.items()}
        self._transitions += count - old_count
        self.total_collected += count - old_count
        while self._transitions > self.capacity:
            expired, arrays = self._episodes.popitem(last=False)
            self._transitions -= len(arrays["actions"])
            self._retired.add(expired)
            self._metadata.pop(expired)
        self._dirty = True

    def _index(self) -> None:
        if not self._dirty:
            return
        self._keys, self._ends = [], []
        total = 0
        for identity, arrays in self._episodes.items():
            total += (len(arrays["actions"]) + self.window_steps - 1) // self.window_steps
            self._keys.append(identity)
            self._ends.append(total)
        self._dirty = False

    def sample(self, batch_size: int, *, device: Any = None) -> PretrainingBatch:
        batch_size = _positive_integer(batch_size, "batch_size")
        if not self.num_starts:
            raise ValueError("cannot train from an empty physical replay")
        return self.sample_indices(self._rng.integers(self.num_starts, size=batch_size).tolist(), device=device)

    def sample_indices(self, indices: Sequence[int], *, device: Any = None) -> PretrainingBatch:
        self._index()
        samples = []
        for selected in indices:
            if type(selected) is not int or not 0 <= selected < self.num_starts:
                raise ValueError("replay crop index is outside its current physical episodes")
            slot = bisect_right(self._ends, selected)
            first = self._ends[slot - 1] if slot else 0
            identity = self._keys[slot]
            samples.append((identity, self._episodes[identity], (selected - first) * self.window_steps))
        return collate_pretraining_sequences(samples, self.contract, rollout_windows=self.rollout_windows,
                                             device=self.device if device is None else device)

    def episodes(self):
        """Read-only snapshots for cold-path export and held-out identity checks."""
        return tuple((identity, {key: value.copy() for key, value in raw.items()}) for identity, raw in self._episodes.items())

    def episode_metadata(self, identity: str) -> dict[str, Any]:
        return _json_copy(self._metadata[identity])

    def state_dict(self) -> dict[str, Any]:
        return {"version": 1, "contract": _json_copy(self.contract), "capacity_transitions": self.capacity,
                "rollout_windows": self.rollout_windows, "total_collected": self.total_collected,
                "rng_state": _json_copy(self._rng.bit_generator.state), "retired": sorted(self._retired),
                "episodes": [{"identity": identity, "sequence": {key: torch.from_numpy(value.copy()) for key, value in raw.items()},
                              "metadata": _json_copy(self._metadata[identity])} for identity, raw in self._episodes.items()]}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        keys = {"version", "contract", "capacity_transitions", "rollout_windows", "total_collected", "rng_state", "retired", "episodes"}
        if set(state) != keys or state["version"] != 1 or state["contract"] != self.contract or state["capacity_transitions"] != self.capacity or state["rollout_windows"] != self.rollout_windows:
            raise ValueError("replay checkpoint contract, capacity or horizon changed")
        candidate = EpisodeReplay(self.contract, capacity_transitions=self.capacity, rollout_windows=self.rollout_windows, device=self.device)
        for entry in state["episodes"]:
            if set(entry) != {"identity", "sequence", "metadata"} or entry["identity"] in candidate._episodes:
                raise ValueError("replay checkpoint contains invalid or duplicate episodes")
            candidate.update_episode(entry["identity"], entry["sequence"], metadata=entry["metadata"])
        if candidate._retired or type(state["total_collected"]) is not int or state["total_collected"] < candidate.transitions:
            raise ValueError("replay checkpoint exceeds capacity or has inconsistent collection counts")
        retired = state["retired"]
        if not isinstance(retired, list) or any(not isinstance(x, str) or not x for x in retired) or len(set(retired)) != len(retired) or set(retired) & candidate._episodes.keys():
            raise ValueError("replay checkpoint episode identities overlap retired entries")
        candidate._retired = set(retired)
        candidate._rng.bit_generator.state = state["rng_state"]
        candidate.total_collected = state["total_collected"]
        self.__dict__.update(candidate.__dict__)
