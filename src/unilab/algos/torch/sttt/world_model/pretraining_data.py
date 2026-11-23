"""Bounded, episode-local pretraining crops over the unchanged raw archive."""
from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from numbers import Integral
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import torch

from .sequence_data import _BOOL, _INT, _WINDOW, _arrays, _contract, _manifest, _sequence_specs


@dataclass(frozen=True)
class PretrainingBatch:
    history_observations: torch.Tensor
    history_previous_actions: torch.Tensor
    history_valid_mask: torch.Tensor
    history_ego_motion: torch.Tensor
    history_ego_valid: torch.Tensor
    history_ego_age_s: torch.Tensor
    sequence: dict[str, torch.Tensor]
    previous_action: torch.Tensor
    previous_residual: torch.Tensor
    observation_mask: torch.Tensor
    transition_mask: torch.Tensor
    origins: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class _Shard:
    path: Path
    steps: int
    starts: int
    size: int
    mtime_ns: int
    digest: str

    def identity(self) -> dict[str, Any]:
        return {"path": str(self.path), "steps": self.steps, "starts": self.starts,
                "size": self.size, "sha256": self.digest}


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _shard_index(path: Path, window_steps: int) -> _Shard:
    """Read one NPY header, not the decompressed episode, to count starts."""
    before = path.stat()
    with ZipFile(path) as archive, archive.open("actions.npy") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"unsupported actions NPY version in {path}: {version}")
    if len(shape) != 2 or shape[1] != 29 or shape[0] < 1 or dtype.kind not in "fiu":
        raise ValueError(f"invalid action shape or dtype in {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"sequence shard changed while being indexed: {path}")
    steps = int(shape[0])
    return _Shard(path, steps, (steps + window_steps - 1) // window_steps,
                  after.st_size, after.st_mtime_ns, digest.hexdigest())


class PretrainingSequenceSampler:
    """Sample uniformly over window-grid starts, including padded final tails.

    The index stores only shard metadata; decompressed shards live in a bounded
    LRU. Validation crops never consume the training generator. Dataset identity
    includes the ordered absolute shard paths and their compressed-file hashes.
    """

    def __init__(
        self,
        directories: Iterable[str | Path] | str | Path,
        *,
        rollout_windows: int = 2,
        seed: int = 0,
        cache_size: int = 2,
    ):
        self.rollout_windows = _positive_integer(rollout_windows, "rollout_windows")
        self.cache_size = _positive_integer(cache_size, "cache_size")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        paths = [directories] if isinstance(directories, (str, Path)) else list(directories)
        if not paths:
            raise ValueError("pretraining requires at least one dataset")
        self._shards: list[_Shard] = []
        self._ends: list[int] = []
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self._contract_json: str | None = None
        seen: set[Path] = set()
        total = 0
        for directory in paths:
            root, manifest = _manifest(directory)
            contract = _contract(manifest["contract"])
            encoded = json.dumps(contract, sort_keys=True, allow_nan=False)
            if self._contract_json is None:
                self._contract_json = encoded
                data = contract.get("data", {})
                history_length = _positive_integer(data.get("history_length"), "contract.data.history_length")
                self.history_steps = history_length - 1
                self.window_steps = _positive_integer(data.get("window_steps"), "contract.data.window_steps")
                self.prediction_steps = self.rollout_windows * self.window_steps
            elif encoded != self._contract_json:
                raise ValueError("cannot mix datasets bound to different source contracts")
            dataset_steps = 0
            for name in manifest["sequences"]:
                path = (root / name).resolve()
                if path in seen:
                    raise ValueError(f"duplicate pretraining shard: {path}")
                seen.add(path)
                shard = _shard_index(path, self.window_steps)
                self._shards.append(shard)
                dataset_steps += shard.steps
                total += shard.starts
                self._ends.append(total)
            if dataset_steps != manifest["transition_count"]:
                raise ValueError("sequence manifest transition count mismatch")
        if total == 0:
            raise ValueError("pretraining datasets contain no physical transitions")
        self.num_starts = total
        self._rng = np.random.default_rng(int(seed))

    @property
    def contract(self) -> dict[str, Any]:
        # Consumers cannot mutate the identity or the sampler's effective contract.
        return json.loads(self._contract_json)

    def _load_shard(self, index: int) -> dict[str, np.ndarray]:
        shard = self._shards[index]
        stat = shard.path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (shard.size, shard.mtime_ns):
            raise ValueError(f"indexed pretraining shard changed: {shard.path}")
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        with np.load(shard.path, allow_pickle=False) as archive:
            arrays = _arrays(dict(archive), self.contract)
        if len(arrays["actions"]) != shard.steps:
            raise ValueError(f"sequence length changed after indexing: {shard.path}")
        if not arrays["is_first"][0]:
            raise ValueError(
                f"pretraining shard must begin at a real episode reset; preceding action is unavailable: {shard.path}"
            )
        dt = self.contract.get("data", {}).get("ctrl_dt")
        if dt is not None and np.any(np.abs(np.diff(arrays["timestamps_ns"]) - round(dt * 1e9)) > 1):
            raise ValueError("stored physical periods disagree with source contract")
        # Evict before inserting so the retained decompressed cache is bounded.
        if len(self._cache) >= self.cache_size:
            self._cache.popitem(last=False)
        self._cache[index] = arrays
        return arrays

    def sample(self, batch_size: int, *, device: str | torch.device = "cpu") -> PretrainingBatch:
        batch_size = _positive_integer(batch_size, "batch_size")
        indices = self._rng.integers(self.num_starts, size=batch_size)
        return self.sample_indices(indices, device=device)

    def sample_indices(
        self, indices: Sequence[int], *, device: str | torch.device = "cpu"
    ) -> PretrainingBatch:
        selected = list(indices)
        if not selected:
            raise ValueError("cannot sample an empty pretraining batch")
        if any(isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < self.num_starts
               for i in selected):
            raise ValueError("pretraining start indices must be integers in [0, num_starts)")
        samples = []
        for selected_index in selected:
            shard_index = bisect_right(self._ends, int(selected_index))
            first_index = self._ends[shard_index - 1] if shard_index else 0
            origin = (int(selected_index) - first_index) * self.window_steps
            samples.append((str(self._shards[shard_index].path), self._load_shard(shard_index), origin))
        return collate_pretraining_sequences(samples, self.contract, rollout_windows=self.rollout_windows, device=device)

    def _sampling_identity(self) -> dict[str, Any]:
        return {"contract": self.contract, "rollout_windows": self.rollout_windows,
                "history_steps": self.history_steps, "prediction_steps": self.prediction_steps,
                "num_starts": self.num_starts, "index": [shard.identity() for shard in self._shards]}

    def state_dict(self) -> dict[str, Any]:
        """Return detached, JSON-safe identity and exact PCG64 training state."""
        return _json_copy({"version": 1, **self._sampling_identity(), "rng_state": self._rng.bit_generator.state})

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("pretraining sampler state must be a mapping")
        try:
            restored = _json_copy(dict(state))
        except (TypeError, ValueError) as error:
            raise ValueError("pretraining sampler state must contain finite JSON primitives") from error
        identity = self._sampling_identity()
        if set(restored) != {"version", "rng_state", *identity} or type(restored["version"]) is not int or restored["version"] != 1:
            raise ValueError("unsupported pretraining sampler state format")
        if any(restored[key] != expected for key, expected in identity.items()):
            raise ValueError("pretraining sampler state contract/index identity mismatch")
        generator = np.random.Generator(np.random.PCG64(0))
        try:
            generator.bit_generator.state = restored["rng_state"]
        except (TypeError, ValueError, KeyError, OverflowError) as error:
            raise ValueError("invalid pretraining sampler RNG state") from error
        self._rng = generator


def collate_pretraining_sequences(samples, contract: Mapping[str, Any], *, rollout_windows: int,
                                  device: str | torch.device = "cpu") -> PretrainingBatch:
    """Crop real episode prefixes identically for immutable archives and live replay."""
    if not samples:
        raise ValueError("cannot sample an empty pretraining batch")
    windows = _positive_integer(rollout_windows, "rollout_windows")
    window_steps = _positive_integer(contract["data"]["window_steps"], "window_steps")
    history = _positive_integer(contract["data"]["history_length"], "history_length") - 1
    horizon, batch_size = windows * window_steps, len(samples)
    sequence: dict[str, np.ndarray] = {}
    state_spec, transition_spec = _sequence_specs(contract)
    for spec, length in ((state_spec, horizon + 1), (transition_spec, horizon), (_WINDOW, windows)):
        for key, tail in spec.items():
            dtype = np.bool_ if key in _BOOL else np.int64 if key in _INT else np.float32
            sequence[key] = np.zeros((batch_size, length, *tail), dtype=dtype)
    history_observations = np.zeros((batch_size, history, contract["data"]["observation_dim"]), dtype=np.float32)
    history_previous_actions = np.zeros((batch_size, history, 29), dtype=np.float32)
    history_valid_mask = np.zeros((batch_size, history), dtype=np.bool_)
    history_ego_motion = np.zeros((batch_size, history, 3), dtype=np.float32)
    history_ego_valid = np.zeros((batch_size, history, 1), dtype=np.bool_)
    history_ego_age_s = np.zeros((batch_size, history, 1), dtype=np.float32)
    previous_action = np.zeros((batch_size, 29), dtype=np.float32)
    previous_residual = np.zeros((batch_size, 29), dtype=np.float32)
    observation_mask = np.zeros((batch_size, horizon + 1), dtype=np.bool_)
    transition_mask = np.zeros((batch_size, horizon), dtype=np.bool_)
    origins: list[tuple[str, int]] = []
    for row, (identity, raw, origin) in enumerate(samples):
        steps = len(raw["actions"])
        if type(origin) is not int or not 0 <= origin < steps or origin % window_steps:
            raise ValueError("forecast origin must be an existing episode-local window-grid start")
        stop = min(origin + horizon, steps)
        transitions = stop - origin
        for key in state_spec:
            sequence[key][row, :transitions + 1] = raw[key][origin:stop + 1]
        for key in transition_spec:
            sequence[key][row, :transitions] = raw[key][origin:stop]
        observation_mask[row, :transitions + 1] = True
        transition_mask[row, :transitions] = True
        if origin:
            previous_action[row] = raw["actions"][origin - 1]
            previous_residual[row] = raw["residuals"][origin - 1]
        history_start = max(0, origin - history)
        history_count = origin - history_start
        if history_count:
            destination = slice(history - history_count, history)
            source = slice(history_start, origin)
            history_observations[row, destination] = raw["observations"][source]
            history_valid_mask[row, destination] = True
            history_ego_motion[row, destination] = raw["ego_motion"][source]
            history_ego_valid[row, destination] = raw["ego_valid"][source]
            history_ego_age_s[row, destination] = raw["ego_age_s"][source]
            action_start = max(1, history_start)
            if action_start < origin:
                offset = history - history_count + action_start - history_start
                history_previous_actions[row, offset:] = raw["actions"][action_start - 1:origin - 1]
        complete = (raw["window_start"] >= origin) & (raw["window_end"] <= stop)
        occupied: set[int] = set()
        for window in np.flatnonzero(complete):
            relative_start = int(raw["window_start"][window]) - origin
            slot = relative_start // window_steps
            if slot in occupied or not 0 <= slot < windows:
                raise ValueError(f"multiple or invalid window labels occupy one forecast slot: {identity}")
            occupied.add(slot)
            sequence["window_start"][row, slot] = relative_start
            sequence["window_end"][row, slot] = int(raw["window_end"][window]) - origin
            sequence["window_delta"][row, slot] = raw["window_delta"][window]
            sequence["window_valid"][row, slot] = raw["window_valid"][window]
        origins.append((str(identity), origin))
    as_tensor = lambda array: torch.as_tensor(array, device=device)
    return PretrainingBatch(
        history_observations=as_tensor(history_observations),
        history_previous_actions=as_tensor(history_previous_actions),
        history_valid_mask=as_tensor(history_valid_mask),
        history_ego_motion=as_tensor(history_ego_motion),
        history_ego_valid=as_tensor(history_ego_valid),
        history_ego_age_s=as_tensor(history_ego_age_s),
        sequence={key: as_tensor(value) for key, value in sequence.items()},
        previous_action=as_tensor(previous_action), previous_residual=as_tensor(previous_residual),
        observation_mask=as_tensor(observation_mask), transition_mask=as_tensor(transition_mask), origins=tuple(origins),
    )
def select_pretraining_batch(batch: PretrainingBatch, index: slice) -> PretrainingBatch:
    values = {}
    for item in fields(batch):
        value = getattr(batch, item.name)
        if item.name == "sequence":
            values[item.name] = {key: tensor[index] for key, tensor in value.items()}
        else:
            values[item.name] = value[index]
    return PretrainingBatch(**values)
