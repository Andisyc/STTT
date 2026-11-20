"""Strict, pickle-free storage for physically contiguous world-model sequences."""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

FORMAT = "sttt_world_model_sequences_v1"
_STATE = {"observations": (64,), "is_first": (), "ego_motion": (3,), "ego_valid": (1,),
          "ego_age_s": (1,), "errors": (2,), "timestamps_ns": ()}
_TRANSITION = {"actions": (29,), "source_actions": (29,), "commands": (7,),
               "path_delta": (3,), "motion_valid": (), "risk": (1,),
               "risk_valid": (), "terminal": ()}
_WINDOW = {"window_start": (), "window_end": (), "window_delta": (3,), "window_valid": ()}
_BOOL = {"is_first", "ego_valid", "motion_valid", "risk_valid", "terminal", "window_valid"}
_INT = {"timestamps_ns", "window_start", "window_end"}
_KEYS = set(_STATE) | set(_TRANSITION) | set(_WINDOW)


def _sequence_specs(contract: Mapping[str, Any] | None = None) -> tuple[dict, dict]:
    data = {} if contract is None else contract.get("data", {})
    return ({**_STATE, "observations": (data.get("observation_dim", 64),)},
            {**_TRANSITION, "commands": (data.get("command_dim", 7),)})


def _arrays(sequence: Mapping[str, Any], contract: Mapping[str, Any] | None = None) -> dict[str, np.ndarray]:
    if set(sequence) != _KEYS:
        raise ValueError(f"sequence keys mismatch: missing={sorted(_KEYS-set(sequence))}, extra={sorted(set(sequence)-_KEYS)}")
    result = {}
    for key, value in sequence.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if key in _BOOL:
            if array.dtype != np.bool_:
                raise ValueError(f"{key} must be boolean")
        elif key in _INT:
            if array.dtype.kind not in "iu" or (array.size and np.any(array > np.iinfo(np.int64).max)):
                raise ValueError(f"{key} must contain int64-compatible integers")
            array = array.astype(np.int64, copy=False)
        else:
            if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
                raise ValueError(f"{key} must contain finite real values")
            if np.any(np.abs(array) > np.finfo(np.float32).max):
                raise ValueError(f"{key} exceeds float32 storage range")
            array = array.astype(np.float32, copy=False)
        result[key] = array
    t = result["actions"].shape[0] if result["actions"].ndim else -1
    w = result["window_start"].shape[0] if result["window_start"].ndim else -1
    if t < 1 or w < 0:
        raise ValueError("a sequence requires at least one transition")
    state_spec, transition_spec = _sequence_specs(contract)
    for spec, length in ((state_spec, t + 1), (transition_spec, t), (_WINDOW, w)):
        for key, tail in spec.items():
            if result[key].shape != (length, *tail):
                raise ValueError(f"{key} must have shape {(length, *tail)}, got {result[key].shape}")
    times = result["timestamps_ns"]
    if np.any(times < 0) or np.any(np.diff(times) <= 0):
        raise ValueError("sequence timestamps must be nonnegative and strictly physically increasing")
    if np.any(result["is_first"][1:]) or np.any(result["terminal"][:-1]):
        raise ValueError("sequence crosses an episode/reset boundary")
    if np.any(result["ego_age_s"] < 0) or np.any((result["risk"] < 0) | (result["risk"] > 1)):
        raise ValueError("ego ages must be nonnegative and risk targets must lie in [0,1]")
    start, end = result["window_start"], result["window_end"]
    if np.any(start < 0) or np.any(end > t) or np.any(end <= start):
        raise ValueError("window labels require ordered physical endpoints inside the sequence")
    for key in ("path_delta", "window_delta"):
        if np.any(np.abs(result[key][..., 2]) > np.pi + 1e-5):
            raise ValueError(f"{key} headings must be wrapped to [-pi,pi]")
    for s, e, delta, valid in zip(start, end, result["window_delta"], result["window_valid"]):
        if valid and result["motion_valid"][s:e].all():
            dense = result["path_delta"][s:e].sum(axis=0, dtype=np.float64)
            dense[2] = (dense[2] + np.pi) % (2*np.pi) - np.pi
            if not np.allclose(dense, delta, atol=2e-5, rtol=1e-4):
                raise ValueError("window label disagrees with its valid dense physical labels")
    return result


def _contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(contract, Mapping) or not contract:
        raise ValueError("sequence metadata requires a nonempty source contract")
    try:
        result = json.loads(json.dumps(dict(contract), allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("source contract must be finite JSON metadata") from error
    data = result.get("data", {})
    if not isinstance(data, dict):
        raise ValueError("contract.data must be a mapping")
    dt = data.get("ctrl_dt")
    if dt is not None and (isinstance(dt, bool) or not isinstance(dt, (int, float)) or not np.isfinite(dt) or dt <= 0 or round(dt*1e9) < 1):
        raise ValueError("contract.data.ctrl_dt must be a finite positive physical period")
    obs_dim = data.get("observation_dim", 64)
    if obs_dim == 66:
        if data.get("observation_contract") != "g1_fada_state_v2" or data.get("command_dim") != 3:
            raise ValueError("66D data requires the explicit g1_fada_state_v2 / command3 contract")
    elif obs_dim != 64 or data.get("command_dim", 7) != 7 or data.get("observation_contract", "official_compact_64_v1") != "official_compact_64_v1":
        raise ValueError("unsupported observation/command contract")
    if data.get("action_dim", 29) != 29:
        raise ValueError("contract.data.action_dim must be 29")
    return result


class SequenceDatasetWriter:
    """One sequence per atomic NPZ shard; the final manifest commits the dataset."""

    def __init__(self, output_dir: str | Path, contract: Mapping[str, Any]):
        self.contract = _contract(contract)
        self.output_dir = Path(output_dir).expanduser().resolve()
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f"sequence output must be empty: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._files: list[str] = []
        self._closed = False
        self.transition_count = 0

    @property
    def sequence_count(self) -> int:
        return len(self._files)

    def append(self, sequence: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("sequence writer is closed")
        arrays = _arrays(sequence, self.contract)
        dt = self.contract.get("data", {}).get("ctrl_dt")
        if dt is not None and (not np.isfinite(dt) or dt <= 0 or np.any(np.abs(np.diff(arrays["timestamps_ns"]) - round(dt * 1e9)) > 1)):
            raise ValueError("physical sample periods disagree with contract.ctrl_dt")
        name = f"sequence_{len(self._files):08d}.npz"
        temporary = self.output_dir / (name + ".tmp")
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(self.output_dir / name)
        self._files.append(name)
        self.transition_count += len(arrays["actions"])

    def close(self) -> None:
        if self._closed:
            return
        manifest = {"format": FORMAT, "contract": self.contract, "sequences": self._files,
                    "transition_count": self.transition_count}
        temporary = self.output_dir / "metadata.json.tmp"
        temporary.write_text(json.dumps(manifest, allow_nan=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.output_dir / "metadata.json")
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.close()


def _manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    root = Path(path).expanduser().resolve()
    file = root / "metadata.json" if root.is_dir() else root
    payload = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError("unknown world-model sequence dataset format")
    _contract(payload.get("contract"))
    names = payload.get("sequences")
    if not isinstance(names, list) or any(
        not isinstance(n, str) or Path(n).name != n or not n.endswith(".npz") for n in names
    ) or len(names) != len(set(names)):
        raise ValueError("invalid sequence manifest shard names")
    count = payload.get("transition_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("invalid manifest transition count")
    return file.parent, payload


def read_sequence_metadata(path: str | Path) -> dict[str, Any]:
    return _manifest(path)[1]["contract"]


def iter_sequences(path: str | Path) -> Iterator[dict[str, torch.Tensor]]:
    root, manifest = _manifest(path)
    count = 0
    for name in manifest["sequences"]:
        with np.load(root / name, allow_pickle=False) as shard:
            arrays = _arrays(dict(shard), manifest["contract"])
        dt = manifest["contract"].get("data", {}).get("ctrl_dt")
        if dt is not None and (not np.isfinite(dt) or dt <= 0 or np.any(np.abs(np.diff(arrays["timestamps_ns"]) - round(dt * 1e9)) > 1)):
            raise ValueError("stored physical periods disagree with source contract")
        count += len(arrays["actions"])
        yield {k: torch.from_numpy(np.array(v, copy=True)) for k, v in arrays.items()}
    if count != manifest.get("transition_count"):
        raise ValueError("sequence manifest transition count mismatch")


def batch_sequences(sequences: Sequence[Mapping[str, Any]], *, device: str | torch.device = "cpu",
                    contract: Mapping[str, Any] | None = None) -> dict[str, torch.Tensor]:
    if not sequences:
        raise ValueError("cannot batch an empty sequence list")
    if contract is not None:
        _contract(contract)
    arrays = [_arrays(s, contract) for s in sequences]
    shapes = {k: v.shape for k, v in arrays[0].items()}
    if any(any(s[k].shape != shape for k, shape in shapes.items()) for s in arrays[1:]):
        raise ValueError("batch requires equal sequence and window shapes; padding is not implicit")
    return {k: torch.as_tensor(np.stack([s[k] for s in arrays]), device=device) for k in shapes}


