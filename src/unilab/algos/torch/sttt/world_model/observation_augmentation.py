"""Simulator-only receive-stream corruption, independent of truth and action RNG."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
import heapq
import math
from numbers import Real

import numpy as np

from unilab.algos.torch.sttt.pose_data import ReceivedPose


@dataclass(frozen=True)
class ObservationAugmentationConfig:
    position_noise_std_m: float = 0.
    yaw_noise_std_rad: float = 0.
    delay_s: float = 0.
    jitter_s: float = 0.
    dropout_probability: float = 0.
    seed: int = 0

    def __post_init__(self) -> None:
        for field in fields(self):
            name, value = field.name, getattr(self, field.name)
            if name == "seed":
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
                    raise ValueError("seed must be a nonnegative integer")
                object.__setattr__(self, name, int(value))
                continue
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            if name == "dropout_probability" and value > 1:
                raise ValueError("dropout_probability must be in [0, 1]")
            object.__setattr__(self, name, float(value))

    @classmethod
    def from_mapping(cls, mapping: Mapping) -> ObservationAugmentationConfig:
        if not isinstance(mapping, Mapping):
            raise TypeError("observation_augmentation must be a mapping")
        unknown = set(mapping) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown observation_augmentation fields: {sorted(unknown, key=str)}")
        return cls(**dict(mapping))

    @property
    def is_identity(self) -> bool:
        return not any((self.position_noise_std_m, self.yaw_noise_std_rad,
                        self.delay_s, self.jitter_s, self.dropout_probability))


class PoseInputAugmenter:
    """Queue noisy poses until a caller supplies an actual physical decision clock.

    XY noise is independent Gaussian navigation-frame translation; yaw noise is
    a Gaussian Z-up rotation about the pelvis origin (roll/pitch and Z survive).
    Extra latency is max(0, delay_s + Normal(0, jitter_s)); dropout is Bernoulli.
    Corruptions are sampled in that order using one private NumPy generator.
    clear() discards queued packets and clock/identity state, but retains audit
    records and advances, rather than reseeds, the private random stream.
    """

    def __init__(self, config: ObservationAugmentationConfig):
        if not isinstance(config, ObservationAugmentationConfig):
            raise TypeError("config must be ObservationAugmentationConfig")
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        self._queue: list[tuple[int, int, ReceivedPose]] = []
        self._records: list[dict] = []
        self._identity: tuple[int, int, str] | None = None
        self._last_release_ns: int | None = None
        self._generation = 0

    def add(self, received_poses: Sequence[ReceivedPose]) -> None:
        for received in received_poses:
            pose = received.pose
            for name, value in (("timestamp_ns", pose.timestamp_ns), ("available_at_ns", received.available_at_ns)):
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
                    raise ValueError(f"{name} must be nonnegative integer nanoseconds")
            if received.available_at_ns < pose.timestamp_ns:
                raise ValueError("pose availability must not precede represented time")
            identity = (pose.sequence_id, pose.tracking_epoch, pose.frame_id)
            if self._identity is not None and identity != self._identity:
                self.clear()
            self._identity = identity
            xy = np.zeros(2, dtype=np.float64)
            yaw = 0.
            augmented_pose = pose
            if pose.valid and pose.navigation_from_pelvis is not None:
                if self.config.position_noise_std_m:
                    xy = self._rng.normal(0., self.config.position_noise_std_m, size=2)
                if self.config.yaw_noise_std_rad:
                    yaw = float(self._rng.normal(0., self.config.yaw_noise_std_rad))
                if self.config.position_noise_std_m or self.config.yaw_noise_std_rad:
                    matrix = np.asarray(pose.navigation_from_pelvis, dtype=np.float64).copy()
                    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                        raise ValueError("pose noise requires a finite 4x4 SE(3) transform")
                    c, s = math.cos(yaw), math.sin(yaw)
                    rotation = np.asarray([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
                    matrix[:3, :3] = rotation @ matrix[:3, :3]
                    matrix[:2, 3] += xy
                    augmented_pose = replace(pose, navigation_from_pelvis=matrix)
            jitter = float(self._rng.normal(0., self.config.jitter_s)) if self.config.jitter_s else 0.
            latency_s = max(0., self.config.delay_s + jitter)
            scheduled = int(received.available_at_ns) + int(round(latency_s * 1e9))
            dropped = bool(self._rng.random() < self.config.dropout_probability) if self.config.dropout_probability else False
            record_id = len(self._records)
            self._records.append({
                "receipt_id": record_id, "generation": self._generation,
                "sequence_id": int(pose.sequence_id), "tracking_epoch": int(pose.tracking_epoch),
                "frame_id": pose.frame_id, "timestamp_ns": int(pose.timestamp_ns),
                "source_timestamps_ns": None if pose.source_timestamps_ns is None else [int(t) for t in pose.source_timestamps_ns],
                "raw_available_at_ns": int(received.available_at_ns),
                "scheduled_available_at_ns": scheduled, "available_at_ns": None,
                "position_noise_xy_m": xy.tolist(), "yaw_noise_rad": yaw,
                "extra_latency_s": latency_s, "dropped": dropped,
                "status": "dropped" if dropped else "queued", "valid": bool(pose.valid),
                "navigation_from_pelvis": None if augmented_pose.navigation_from_pelvis is None else np.asarray(augmented_pose.navigation_from_pelvis).tolist(),
            })
            if not dropped:
                packet = received if augmented_pose is pose else replace(received, pose=augmented_pose)
                heapq.heappush(self._queue, (scheduled, record_id, packet))

    def release(self, now_ns: int) -> tuple[ReceivedPose, ...]:
        if isinstance(now_ns, (bool, np.bool_)) or not isinstance(now_ns, (int, np.integer)) or now_ns < 0:
            raise ValueError("decision time must be nonnegative integer nanoseconds")
        now_ns = int(now_ns)
        if self._last_release_ns is not None and now_ns < self._last_release_ns:
            raise ValueError("decision clock must not move backwards without clear()")
        self._last_release_ns = now_ns
        released = []
        while self._queue and self._queue[0][0] <= now_ns:
            _, record_id, received = heapq.heappop(self._queue)
            self._records[record_id]["available_at_ns"] = now_ns
            self._records[record_id]["status"] = "released"
            released.append(received if received.available_at_ns == now_ns else replace(received, available_at_ns=now_ns))
        return tuple(released)

    def clear(self) -> None:
        for _, record_id, _ in self._queue:
            self._records[record_id]["status"] = "cleared"
        self._queue.clear()
        self._identity = None
        self._last_release_ns = None
        self._generation += 1

    def provenance(self) -> dict:
        # Copy nested arrays so callers cannot mutate the live receipt audit.
        return {
            "recipe": asdict(self.config),
            "semantics": {
                "position_noise": "independent_gaussian_navigation_xy_m",
                "yaw_noise": "gaussian_navigation_z_rotation_about_pelvis_rad",
                "extra_latency": "max(0, delay_s + gaussian_std_jitter_s)",
                "dropout": "independent_bernoulli_per_receipt",
                "order": ["pose_noise", "latency_jitter", "dropout"],
                "availability": "first_release_physical_clock_ns",
                "clear_rng": "continue_private_generator_without_reseeding",
            },
            "receipts": [
                {**record,
                 "position_noise_xy_m": list(record["position_noise_xy_m"]),
                 "source_timestamps_ns": None if record["source_timestamps_ns"] is None else list(record["source_timestamps_ns"]),
                 "navigation_from_pelvis": None if record["navigation_from_pelvis"] is None else [list(row) for row in record["navigation_from_pelvis"]]}
                for record in self._records
            ],
        }
