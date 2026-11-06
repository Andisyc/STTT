"""Bounded NPZ shards for v5 response windows; no retired format or pickle path."""
from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from unilab.algos.torch.sttt.path_following.contract import (
    InterventionQueryRecord, PathFollowingWindowSample, _integer,
)
from unilab.algos.torch.sttt.path_following.intervention import compose_joint_position_residual
from unilab.algos.torch.sttt.path_following.reference import FixedStraightPath, pose_matrix
from unilab.algos.torch.sttt.pose_data import ControlInterval, TimedPelvisPose

FORMAT = "sttt_path_following_windows"
SCHEMA_VERSION = 1
_TENSORS = (
    "observation_history", "action_history", "command", "history_valid_mask",
    "planner_future", "adapted_future", "residual_rad",
)


def _sample_contract(sample: PathFollowingWindowSample) -> dict[str, Any]:
    query = sample.start_query
    if query.planner_future.dtype not in (torch.float32, torch.float64):
        raise ValueError("response storage supports float32/float64 query tensors")
    return {
        "window_steps": len(sample.controls), "ctrl_dt": query.ctrl_dt,
        "history_length": query.observation_history.shape[1],
        "prediction_horizon": query.planner_future.shape[1],
        "joint_names": list(query.joint_names), "position_scale": query.position_scale,
        "velocity_scale": query.velocity_scale,
        "tensor_dtype": str(query.planner_future.cpu().numpy().dtype),
    }


def _encode(samples: list[PathFollowingWindowSample]) -> dict[str, Any]:
    arrays = {
        name: np.stack([getattr(sample.start_query, name).cpu().numpy()[0] for sample in samples])
        for name in _TENSORS
    }
    arrays.update({
        "format": np.asarray(FORMAT), "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
        "window_id": np.asarray([s.window_id for s in samples], dtype=np.int64),
        "sequence_id": np.asarray([s.start_pose.sequence_id for s in samples], dtype=np.int64),
        "tracking_epoch": np.asarray([s.start_pose.tracking_epoch for s in samples], dtype=np.int64),
        "frame_id": np.asarray([s.start_pose.frame_id for s in samples]),
        "control_transition_id": np.asarray([[c.transition_id for c in s.controls] for s in samples], dtype=np.int64),
        "control_times_ns": np.asarray([[(c.start_ns, c.end_ns) for c in s.controls] for s in samples], dtype=np.int64),
        "env_input_action": np.stack([np.stack([c.action for c in s.controls]) for s in samples]),
        "velocity_command": np.stack([np.stack([c.command for c in s.controls]) for s in samples]),
        "poses": np.stack([np.stack((pose_matrix(s.start_pose), pose_matrix(s.end_pose))) for s in samples]),
        "pose_source_timestamps_ns": np.asarray([[s.start_pose.source_timestamps_ns, s.end_pose.source_timestamps_ns] for s in samples], dtype=np.int64),
        "reference_timestamp_ns": np.asarray([s.reference.timestamp_ns for s in samples], dtype=np.int64),
        "reference_origin_xy": np.asarray([s.reference.origin_xy for s in samples], dtype=np.float64),
        "reference_heading_rad": np.asarray([s.reference.heading_rad for s in samples], dtype=np.float64),
        "delta": np.stack([s.delta for s in samples]),
        "path_errors": np.stack([s.path_errors for s in samples]),
    })
    return arrays


class PathFollowingDatasetWriter:
    def __init__(self, directory: str | Path, *, source: Mapping[str, Any], shard_size: int = 256) -> None:
        self.shard_size = _integer(shard_size, "shard_size")
        if self.shard_size < 1:
            raise ValueError("shard_size must be positive")
        self.directory = Path(directory)
        self._metadata: dict[str, Any] = {
            "format": FORMAT, "schema_version": SCHEMA_VERSION,
            "complete": False, "source": dict(source), "contract": None,
            "sample_count": 0, "shards": [],
        }
        json.dumps(self._metadata, allow_nan=False)
        if self.directory.exists() and any(self.directory.iterdir()):
            raise FileExistsError(f"response output directory must be empty: {self.directory}")
        self.directory.mkdir(parents=True, exist_ok=True)
        self._buffer: list[PathFollowingWindowSample] = []
        self._closed = False
        self.sample_count = 0
        self._last_window: dict[int, int] = {}
        self._publish()

    def _publish(self) -> None:
        path = self.directory / "source.json"
        temporary = self.directory / "source.json.tmp"
        temporary.write_text(json.dumps(self._metadata, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    def append(self, sample: PathFollowingWindowSample) -> None:
        if self._closed:
            raise RuntimeError("response writer is closed")
        contract = _sample_contract(sample)
        if self._metadata["contract"] is None:
            self._metadata["contract"] = contract
        elif contract != self._metadata["contract"]:
            raise ValueError("cannot mix response windows with different horizons, timing or source interpretation")
        sequence = sample.start_pose.sequence_id
        if sample.window_id <= self._last_window.get(sequence, -1):
            raise ValueError("window ids must increase within each sequence")
        self._last_window[sequence] = sample.window_id
        self._buffer.append(sample)
        self.sample_count += 1
        if len(self._buffer) >= self.shard_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        name = f"windows_{len(self._metadata['shards']):06d}.npz"
        np.savez_compressed(self.directory / name, **_encode(self._buffer))
        self._metadata["shards"].append({"file": name, "samples": len(self._buffer)})
        self._metadata["sample_count"] = self.sample_count
        self._buffer.clear()
        self._publish()

    def close(self, *, capture_summary: Mapping[str, Any] | None = None) -> None:
        if self._closed:
            return
        try:
            self._flush()
            if self.sample_count == 0:
                raise ValueError("capture produced no complete valid response windows")
            if capture_summary is not None:
                self._metadata["capture_summary"] = dict(capture_summary)
            self._metadata["complete"] = True
            self._publish()
        finally:
            self._closed = True

    def abort(self) -> None:
        """Leave an explicitly incomplete dataset, never a successful capture."""
        self._buffer.clear()
        self._closed = True


def read_path_following_metadata(directory: str | Path) -> dict[str, Any]:
    path = Path(directory) / "source.json"
    if not path.is_file():
        raise ValueError("not a versioned path-following response dataset; retired data is unsupported")
    metadata = json.loads(path.read_text())
    if (
        not isinstance(metadata, dict) or metadata.get("format") != FORMAT
        or metadata.get("schema_version") != SCHEMA_VERSION or metadata.get("complete") is not True
    ):
        raise ValueError("unsupported or incomplete path-following response dataset")
    contract = metadata.get("contract")
    if not isinstance(contract, dict) or not isinstance(metadata.get("source"), dict):
        raise ValueError("response dataset lacks its source interpretation")
    for field in ("window_steps", "history_length", "prediction_horizon"):
        if _integer(contract[field], field) < 1:
            raise ValueError(f"response contract {field} must be positive")
    if contract["tensor_dtype"] not in ("float32", "float64"):
        raise ValueError("unsupported response tensor dtype")
    shards = metadata.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("response dataset must contain nonempty shards")
    names = set()
    count = 0
    for shard in shards:
        name = shard["file"]
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".npz") or name in names:
            raise ValueError("invalid or duplicate response shard filename")
        names.add(name)
        size = _integer(shard["samples"], "shard samples")
        if size < 1:
            raise ValueError("response shards must be nonempty")
        count += size
    if count != metadata.get("sample_count"):
        raise ValueError("response shard counts do not match metadata")
    return metadata


def _validate_arrays(arrays: dict[str, np.ndarray], contract: dict[str, Any], count: int) -> None:
    h, k, n = contract["history_length"], contract["prediction_horizon"], contract["window_steps"]
    shapes = {
        "observation_history": (count, h, 64), "action_history": (count, h, 29),
        "command": (count, 7), "history_valid_mask": (count, h),
        "planner_future": (count, k, 64), "adapted_future": (count, k, 64),
        "residual_rad": (count, 29), "window_id": (count,), "sequence_id": (count,),
        "tracking_epoch": (count,), "frame_id": (count,),
        "control_transition_id": (count, n), "control_times_ns": (count, n, 2),
        "env_input_action": (count, n, 29), "velocity_command": (count, n, 3),
        "poses": (count, 2, 4, 4), "pose_source_timestamps_ns": (count, 2, 2),
        "reference_timestamp_ns": (count,), "reference_origin_xy": (count, 2),
        "reference_heading_rad": (count,), "delta": (count, 3), "path_errors": (count, 2, 2),
    }
    if set(arrays) != {*shapes, "format", "schema_version"}:
        raise ValueError("response shard fields do not match the window schema")
    if arrays["format"].shape != () or arrays["format"].item() != FORMAT or arrays["schema_version"].shape != () or arrays["schema_version"].item() != SCHEMA_VERSION:
        raise ValueError("unsupported response shard identity")
    integers = {"window_id", "sequence_id", "tracking_epoch", "control_transition_id", "control_times_ns", "pose_source_timestamps_ns", "reference_timestamp_ns"}
    for name, shape in shapes.items():
        array = arrays[name]
        if array.shape != shape:
            raise ValueError(f"invalid response shard shape for {name}: {array.shape}")
        if name in integers:
            if array.dtype.kind not in "iu" or (array < 0).any():
                raise ValueError(f"{name} requires nonnegative integer values")
        elif name == "history_valid_mask":
            if array.dtype != np.bool_:
                raise ValueError("history mask must be boolean")
        elif name == "frame_id":
            if array.dtype.kind != "U":
                raise ValueError("frame ids must be strings")
        elif array.dtype.kind != "f" or not np.isfinite(array).all():
            raise ValueError(f"{name} requires finite floating-point values")
        if name in _TENSORS and name != "history_valid_mask" and array.dtype != np.dtype(contract["tensor_dtype"]):
            raise ValueError("query tensor dtype differs from the response contract")


def iter_path_following_samples(directory: str | Path) -> Iterator[PathFollowingWindowSample]:
    directory = Path(directory)
    metadata = read_path_following_metadata(directory)
    contract = metadata["contract"]
    last_window: dict[int, int] = {}
    for shard in metadata["shards"]:
        with np.load(directory / shard["file"], allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        _validate_arrays(arrays, contract, shard["samples"])
        for index in range(shard["samples"]):
            sequence = int(arrays["sequence_id"][index])
            window_id = int(arrays["window_id"][index])
            if window_id <= last_window.get(sequence, -1):
                raise ValueError("duplicate or out-of-order response window")
            last_window[sequence] = window_id
            query = InterventionQueryRecord(
                **{name: torch.from_numpy(arrays[name][index:index + 1]) for name in _TENSORS},
                joint_names=tuple(contract["joint_names"]), ctrl_dt=contract["ctrl_dt"],
                position_scale=contract["position_scale"], velocity_scale=contract["velocity_scale"],
            )
            expected_future = compose_joint_position_residual(
                query.planner_future, query.residual_rad, ctrl_dt=query.ctrl_dt,
                position_scale=query.position_scale, velocity_scale=query.velocity_scale,
            )
            if not torch.equal(expected_future, query.adapted_future):
                raise ValueError("stored residual does not match the composed Tracker target")
            controls = tuple(ControlInterval(
                sequence, int(arrays["control_transition_id"][index, step]),
                int(arrays["control_times_ns"][index, step, 0]), int(arrays["control_times_ns"][index, step, 1]),
                arrays["env_input_action"][index, step], arrays["velocity_command"][index, step],
            ) for step in range(contract["window_steps"]))
            epoch, frame = int(arrays["tracking_epoch"][index]), str(arrays["frame_id"][index])
            poses = tuple(TimedPelvisPose(
                sequence, epoch, timestamp, arrays["poses"][index, endpoint], True,
                (int(arrays["pose_source_timestamps_ns"][index, endpoint, 0]),
                 int(arrays["pose_source_timestamps_ns"][index, endpoint, 1])), frame,
            ) for endpoint, timestamp in enumerate((controls[0].start_ns, controls[-1].end_ns)))
            reference = FixedStraightPath(
                sequence, epoch, frame, int(arrays["reference_timestamp_ns"][index]),
                (float(arrays["reference_origin_xy"][index, 0]), float(arrays["reference_origin_xy"][index, 1])),
                float(arrays["reference_heading_rad"][index]),
            )
            sample = PathFollowingWindowSample(window_id, query, controls, poses[0], poses[1], reference)
            if not np.allclose(sample.delta, arrays["delta"][index], rtol=0, atol=1e-9) or not np.allclose(sample.path_errors, arrays["path_errors"][index], rtol=0, atol=1e-9):
                raise ValueError("stored response labels do not match their measured poses and fixed reference")
            yield sample
