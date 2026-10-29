"""Sharded FADA replay storage, retention, and plan lifecycle owner."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from unilab.algos.torch.distill.fada.model import FADAArchitectureConfig, FADASourceBatch
from unilab.algos.torch.distill.fada.collection_contract import (
    OFFICIAL_FADA_SOURCE_REVISION,
    OfficialFADACollectionSpec,
    OfficialFADARawTransitionBatch,
    OfficialFADASourceBatch,
    validate_official_fada_preprocessing,
)
from unilab.algos.torch.distill.fada.official_learning import canonical_official_training_config
from unilab.algos.torch.distill.fada.replay_sampling import (
    FADAReplayCoverage,
    FADAReplaySamplingPlan,
    FADAReplaySamplingSpec,
    build_replay_sampling_plan,
    official_replay_source_counts,
    planner_sample_indices,
)
from unilab.algos.torch.distill.fada.source_artifact import (
    LoadedFADASourceArtifact,
    LoadedOfficialFADASourceArtifact,
    batch_to_device,
)
from unilab.algos.torch.distill.fada.timing import FADAPerformanceRecorder, measure


@dataclass(frozen=True)
class FADAReplayRoleCounts:
    """Read-only retained-row evidence for the admitted paper source roles."""

    planner_eligible: int
    planner_ineligible: int


def _slice_batch(batch: FADASourceBatch, start: int, end: int) -> FADASourceBatch:
    return FADASourceBatch(
        **{
            field: getattr(batch, field)[start:end]
            for field in FADASourceBatch.__dataclass_fields__
        }
    )


class FADAReplayBuffer:
    """Bounded replay that owns independent CPU shards instead of one monolith."""

    def __init__(
        self,
        config: FADAArchitectureConfig,
        *,
        capacity: int,
        suboptimal_retention_ratio: int | None = None,
    ) -> None:
        if int(capacity) <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if suboptimal_retention_ratio is not None:
            ratio = int(suboptimal_retention_ratio)
            if ratio <= 0 or float(ratio) != float(suboptimal_retention_ratio):
                raise ValueError(
                    "suboptimal_retention_ratio must be a positive integer, "
                    f"got {suboptimal_retention_ratio}"
                )
            if int(capacity) < ratio + 1:
                raise ValueError(
                    "capacity must fit one complete replay role-ratio block: "
                    f"capacity={capacity} ratio=1:{ratio}"
                )
        self.config = config
        self.capacity = int(capacity)
        self.suboptimal_retention_ratio = (
            None if suboptimal_retention_ratio is None else int(suboptimal_retention_ratio)
        )
        self._chunks: tuple[FADASourceBatch, ...] = ()
        self._size = 0
        self._generation = 0
        self._sampling_plan_cache: (
            tuple[int, int, FADAReplaySamplingSpec, FADAReplaySamplingPlan] | None
        ) = None

    def __len__(self) -> int:
        return self._size

    @property
    def _batch(self) -> FADASourceBatch | None:
        """Compatibility view for diagnostics/tests; runtime sampling stays sharded."""

        if not self._chunks:
            return None
        if len(self._chunks) == 1:
            return self._chunks[0]
        return FADASourceBatch(
            **{
                field: torch.cat([getattr(chunk, field) for chunk in self._chunks], dim=0)
                for field in FADASourceBatch.__dataclass_fields__
            }
        ).validate(self.config)

    @property
    def effective_capacity(self) -> int:
        ratio = self.suboptimal_retention_ratio
        if ratio is None:
            return self.capacity
        return (self.capacity // (ratio + 1)) * (ratio + 1)

    def source_role_counts(self) -> FADAReplayRoleCounts:
        planner_eligible = sum(int(chunk.planner_eligible.sum()) for chunk in self._chunks)
        return FADAReplayRoleCounts(
            planner_eligible=planner_eligible,
            planner_ineligible=len(self) - planner_eligible,
        )

    def _retained_indices(self, roles: Sequence[torch.Tensor]) -> torch.Tensor:
        size = sum(int(role.shape[0]) for role in roles)
        ratio = self.suboptimal_retention_ratio
        if ratio is None or size <= self.capacity:
            start = max(size - self.capacity, 0)
            return torch.arange(start, size, dtype=torch.int64)

        planner_eligible_capacity = self.capacity // (ratio + 1)
        planner_ineligible_capacity = planner_eligible_capacity * ratio
        planner_eligible = torch.cat(tuple(roles))
        planner_eligible_indices = torch.nonzero(planner_eligible, as_tuple=False).flatten()
        planner_ineligible_indices = torch.nonzero(~planner_eligible, as_tuple=False).flatten()
        if planner_eligible_indices.numel() < planner_eligible_capacity:
            raise ValueError(
                "paper FADA replay overflow lacks Planner-eligible main rows: "
                f"required={planner_eligible_capacity} "
                f"observed={planner_eligible_indices.numel()}"
            )
        if planner_ineligible_indices.numel() < planner_ineligible_capacity:
            raise ValueError(
                "paper FADA replay overflow lacks Planner-ineligible intermediate rows: "
                f"required={planner_ineligible_capacity} "
                f"observed={planner_ineligible_indices.numel()}"
            )
        selected = torch.cat(
            (
                planner_eligible_indices[-planner_eligible_capacity:],
                planner_ineligible_indices[-planner_ineligible_capacity:],
            )
        )
        return selected.sort().values

    def _retain_chunks(
        self,
        chunks: Sequence[FADASourceBatch],
        indices: torch.Tensor,
    ) -> tuple[FADASourceBatch, ...]:
        retained: list[FADASourceBatch] = []
        offset = 0
        for chunk in chunks:
            rows = int(chunk.command.shape[0])
            left = int(torch.searchsorted(indices, torch.tensor(offset)).item())
            right = int(torch.searchsorted(indices, torch.tensor(offset + rows)).item())
            local = indices[left:right] - offset
            offset += rows
            if local.numel() == 0:
                continue
            first = int(local[0])
            last = int(local[-1])
            if first == 0 and last + 1 == rows and local.numel() == rows:
                candidate = chunk
            elif last - first + 1 == local.numel():
                candidate = FADASourceBatch(
                    **{
                        field: getattr(chunk, field)[first : last + 1].clone()
                        for field in FADASourceBatch.__dataclass_fields__
                    }
                )
            else:
                candidate = FADASourceBatch(
                    **{
                        field: getattr(chunk, field).index_select(0, local)
                        for field in FADASourceBatch.__dataclass_fields__
                    }
                )
            retained.append(candidate.validate(self.config))
        return tuple(retained)

    def add(self, batch: FADASourceBatch) -> None:
        self.add_many((batch,))

    def add_many(self, batches: Iterable[FADASourceBatch]) -> None:
        """Copy caller-owned batches into one atomic replay transaction."""

        self._add_many(batches, copy_incoming=True)

    def _add_many(
        self,
        batches: Iterable[FADASourceBatch],
        *,
        copy_incoming: bool,
    ) -> None:
        """Atomically admit one collection transaction without global tensor copies."""

        incoming = tuple(
            FADASourceBatch(
                **{
                    field: getattr(batch, field).detach().to("cpu").clone()
                    for field in FADASourceBatch.__dataclass_fields__
                }
            ).validate(self.config)
            if copy_incoming
            else batch_to_device(batch.validate(self.config), torch.device("cpu"))
            for batch in batches
        )
        if not incoming:
            raise ValueError("cannot add an empty FADA replay transaction")
        candidates = (*self._chunks, *incoming)
        retained_indices = self._retained_indices(
            tuple(batch.planner_eligible for batch in candidates)
        )
        retained = self._retain_chunks(candidates, retained_indices)
        self._chunks = retained
        self._size = int(retained_indices.numel())
        self._generation += 1
        self._sampling_plan_cache = None

    def add_artifact(self, artifact: LoadedFADASourceArtifact) -> None:
        """Atomically retain only selected artifact shards, loading them one at a time."""

        incoming_roles = artifact.planner_role_vectors()
        role_chunks = (
            *(chunk.planner_eligible for chunk in self._chunks),
            *incoming_roles,
        )
        retained_indices = self._retained_indices(role_chunks)
        old_size = len(self)
        old_end = int(torch.searchsorted(retained_indices, torch.tensor(old_size)).item())
        retained: list[FADASourceBatch] = list(
            self._retain_chunks(self._chunks, retained_indices[:old_end])
        )
        incoming_indices = retained_indices[old_end:] - old_size
        offset = 0
        for index, roles in enumerate(incoming_roles):
            rows = int(roles.shape[0])
            left = int(torch.searchsorted(incoming_indices, torch.tensor(offset)).item())
            right = int(torch.searchsorted(incoming_indices, torch.tensor(offset + rows)).item())
            local = incoming_indices[left:right] - offset
            offset += rows
            if local.numel() == 0:
                continue
            batch = artifact.load_batch(index)
            retained.extend(self._retain_chunks((batch,), local))
        self._chunks = tuple(retained)
        self._size = int(retained_indices.numel())
        self._generation += 1
        self._sampling_plan_cache = None

    def _select_global_indices(self, indices: torch.Tensor) -> FADASourceBatch:
        outputs: dict[str, torch.Tensor] = {}
        first = self._chunks[0]
        for field in FADASourceBatch.__dataclass_fields__:
            tensor = getattr(first, field)
            outputs[field] = torch.empty(
                (indices.numel(), *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device
            )
        offset = 0
        for chunk in self._chunks:
            rows = int(chunk.command.shape[0])
            positions = torch.nonzero(
                (indices >= offset) & (indices < offset + rows), as_tuple=False
            ).flatten()
            if positions.numel() > 0:
                local = indices.index_select(0, positions) - offset
                for field in FADASourceBatch.__dataclass_fields__:
                    outputs[field].index_copy_(
                        0,
                        positions,
                        getattr(chunk, field).index_select(0, local),
                    )
            offset += rows
        return FADASourceBatch(**outputs).validate(self.config)

    def sample(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
        device: str | torch.device = "cpu",
    ) -> FADASourceBatch:
        if len(self) == 0:
            raise ValueError("cannot sample an empty FADA replay buffer")
        if int(batch_size) <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        indices = torch.randint(len(self), (int(batch_size),), generator=generator)
        sampled = self._select_global_indices(indices)
        return batch_to_device(sampled, torch.device(device)).validate(self.config)

    def _metadata_field(self, field: str) -> torch.Tensor:
        return torch.cat([getattr(chunk, field) for chunk in self._chunks])

    def _sampling_plan(
        self,
        sampling_spec: FADAReplaySamplingSpec,
        *,
        batch_size: int,
    ) -> FADAReplaySamplingPlan:
        ratio = self.suboptimal_retention_ratio
        cached = self._sampling_plan_cache
        if (
            cached is not None
            and cached[0] == self._generation
            and cached[1] == int(batch_size)
            and cached[2] == sampling_spec
        ):
            return cached[3]
        plan = build_replay_sampling_plan(
            int(batch_size),
            planner_eligible=self._metadata_field("planner_eligible"),
            command_scenario=self._metadata_field("command_scenario"),
            cold_start=self._metadata_field("cold_start"),
            command=self._metadata_field("command"),
            spec=sampling_spec,
            suboptimal_retention_ratio=ratio,
        )
        self._sampling_plan_cache = (self._generation, int(batch_size), sampling_spec, plan)
        return plan

    def sample_planner(
        self,
        batch_size: int,
        *,
        sampling_spec: FADAReplaySamplingSpec,
        generator: torch.Generator | None = None,
        device: str | torch.device = "cpu",
    ) -> FADASourceBatch:
        """Sample one exact scenario-balanced Planner batch from eligible replay rows."""

        if len(self) == 0:
            raise ValueError("cannot sample an empty FADA replay buffer")
        plan = self._sampling_plan(sampling_spec, batch_size=int(batch_size))
        indices = plan.sample_planner(generator)
        sampled = self._select_global_indices(indices)
        return batch_to_device(sampled, torch.device(device)).validate(self.config)

    def sample_idm(
        self,
        batch_size: int,
        *,
        sampling_spec: FADAReplaySamplingSpec,
        generator: torch.Generator | None = None,
        device: str | torch.device = "cpu",
    ) -> FADASourceBatch:
        """Sample one deterministic-role, speed-stratified IDM batch."""

        if len(self) == 0:
            raise ValueError("cannot sample an empty FADA replay buffer")
        plan = self._sampling_plan(sampling_spec, batch_size=int(batch_size))
        if not plan.idm:
            raise ValueError("IDM stratified sampling requires paper source role retention")
        indices = plan.sample_idm(generator)
        sampled = self._select_global_indices(indices)
        return batch_to_device(sampled, torch.device(device)).validate(self.config)

    def validate_sampling_spec(
        self,
        sampling_spec: FADAReplaySamplingSpec,
        *,
        batch_size: int,
    ) -> FADAReplayCoverage:
        """Validate every positive stratum before mutation and compute update floors."""

        if len(self) == 0:
            raise ValueError("cannot validate an empty FADA replay buffer")
        plan = self._sampling_plan(sampling_spec, batch_size=int(batch_size))
        if not plan.idm:
            raise ValueError("stratified replay validation requires paper source role retention")
        return plan.coverage


_OFFICIAL_DATASET_RELATIVE_PATH = Path(
    "src/holosoma/holosoma/fada/common/dataset.py"
)
_OFFICIAL_RAW_FIELDS: tuple[str, ...] = (
    "obs",
    "current_command",
    "executed_act",
    "expert_act",
    "expert_chunk",
    "expert_future_obs_chunk",
    "has_expert_chunk",
    "has_expert_future_obs_chunk",
    "strict_label_valid",
    "reward",
    "done",
    "env_id",
    "episode_id",
)


def _load_official_replay_classes(upstream_root: str) -> tuple[type, type, type]:
    """Load the pinned upstream pure-NumPy ``dataset.py`` by explicit file import.

    Only ``dataset.py`` is loaded; it depends on stdlib/NumPy alone, so the
    holosoma package, ``safe_torch_import``, and the native stack are never
    imported here, and no local reimplementation of the upstream sampling math
    is substituted. A missing/incomplete pinned checkout raises explicitly.
    """

    root = Path(upstream_root).expanduser().resolve()
    dataset_path = root / _OFFICIAL_DATASET_RELATIVE_PATH
    if not dataset_path.is_file():
        raise FileNotFoundError(
            "official FADA upstream dataset module not found: "
            f"{dataset_path} (upstream_root={root}, pinned revision "
            f"{OFFICIAL_FADA_SOURCE_REVISION})"
        )
    module_name = "unilab_official_fada_upstream_dataset"
    spec = importlib.util.spec_from_file_location(module_name, dataset_path)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"cannot load the official FADA dataset module from {dataset_path}"
        )
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    for name in (
        "ReplayBuffer",
        "ReplayCacheMetadataMismatchError",
        "ReplayCacheLegacyMetadataMissingError",
    ):
        if not hasattr(module, name):
            raise ImportError(
                f"official FADA dataset module {dataset_path} is missing {name}"
            )
    return (
        module.ReplayBuffer,
        module.ReplayCacheMetadataMismatchError,
        module.ReplayCacheLegacyMetadataMissingError,
    )


def _raw_field_array(batch: OfficialFADARawTransitionBatch, name: str) -> np.ndarray:
    return getattr(batch, name).detach().cpu().numpy()


def _append_official_raw(buffer: Any, batch: OfficialFADARawTransitionBatch) -> None:
    """Adapt stored validity flags to the upstream optional-chunk add_batch API."""

    arrays = {name: _raw_field_array(batch, name) for name in _OFFICIAL_RAW_FIELDS}
    action_valid = arrays.pop("has_expert_chunk")
    future_valid = arrays.pop("has_expert_future_obs_chunk")
    changes = np.flatnonzero(
        (action_valid[1:] != action_valid[:-1]) | (future_valid[1:] != future_valid[:-1])
    ) + 1
    start = 0
    for end in (*changes.tolist(), batch.num_samples):
        values = {name: value[start:end] for name, value in arrays.items()}
        if not action_valid[start]:
            values["expert_chunk"] = None
        if not future_valid[start]:
            values["expert_future_obs_chunk"] = None
        buffer.add_batch(**values)
        start = end


class OfficialFADAReplayBuffer:
    """Own the original upstream raw-transition replay rules in the parent.

    Wraps the pinned upstream pure-NumPy ``ReplayBuffer`` (loaded by file from
    ``spec.upstream_root``) for the three physical sources: the offline
    ``optimal`` and ``suboptimal`` buffers are growable and frozen once the
    warmup phase is finalized, while the ``online`` buffer is a fixed
    ``cfg.replay_capacity`` transition buffer with the original random/fifo
    episode-unit eviction. Physical collection never subsamples, splits, or
    samples windows; this owner applies trajectory subsampling and the
    train/val split exactly once with the original seeds and functions, and
    every update re-runs the upstream ``sample_chunk`` and converts the
    original nine-field result into ``OfficialFADASourceBatch``.
    """

    def __init__(
        self,
        spec: OfficialFADACollectionSpec,
        *,
        preprocessing: Mapping[str, Any],
        training_config: Mapping[str, Any],
        resolved_config: Mapping[str, Any],
        source_identity: Mapping[str, str] | None = None,
        performance: FADAPerformanceRecorder | None = None,
    ) -> None:
        if not isinstance(spec, OfficialFADACollectionSpec):
            raise TypeError(
                "official replay requires an OfficialFADACollectionSpec, "
                f"got {type(spec).__name__}"
            )
        self.spec = spec
        self._performance = performance
        self.preprocessing = validate_official_fada_preprocessing(preprocessing)
        self.training_config = canonical_official_training_config(training_config)
        if not isinstance(resolved_config, Mapping):
            raise TypeError("official replay requires the prepared resolved_config mapping")
        self.resolved_config = dict(resolved_config)
        for name in ("replay_capacity", "seed", "val_trajectory_ratio"):
            value = self.resolved_config.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"official replay resolved_config requires a numeric {name!r}"
                )
        if int(self.resolved_config["replay_capacity"]) <= 0:
            raise ValueError("official replay capacity must be a positive transition count")
        eviction = str(self.resolved_config.get("online_buffer_eviction_policy", "random"))
        if eviction not in {"fifo", "random"}:
            raise ValueError("online_buffer_eviction_policy must be 'fifo' or 'random'")

        replay_cls, mismatch_error, legacy_error = _load_official_replay_classes(
            spec.upstream_root
        )
        self._replay_buffer_cls = replay_cls
        # Upstream cache-provenance errors, re-raised by load_npz on mismatch;
        # the parent workflow catches these to order a recollection.
        self._cache_mismatch_errors: tuple[type, type] = (mismatch_error, legacy_error)
        buffer_kwargs = {
            "capacity": int(self.resolved_config["replay_capacity"]),
            "obs_dim": 64,
            "act_dim": 29,
            "cmd_dim": 7,
            "history_len": spec.history_length,
            "pred_horizon": spec.prediction_horizon,
            "require_future_obs_targets": True,
        }
        self._buffers: dict[str, Any] = {
            "optimal": replay_cls(growable=True, **buffer_kwargs),
            "suboptimal": replay_cls(growable=True, **buffer_kwargs),
            "online": replay_cls(eviction_policy=eviction, **buffer_kwargs),
        }
        self._val_buffers: dict[str, Any] = dict.fromkeys(self._buffers)
        self._offline_frozen = False
        self._expert_identity: tuple[str, str] | None = None
        if source_identity is not None:
            if (
                source_identity["upstream_revision"] != OFFICIAL_FADA_SOURCE_REVISION
                or Path(source_identity["expert_checkpoint"]).expanduser().resolve()
                != Path(spec.expert_checkpoint).expanduser().resolve()
            ):
                raise ValueError("official prepared source identity contradicts Replay")
            self._expert_identity = (
                source_identity["expert_checkpoint_sha256"],
                source_identity["experiment_config_sha256"],
            )
        # Collision-safe (env_id, episode_id) remapping: a fresh native worker
        # after resume restarts its local episode counters, so incoming ids are
        # offset past every id this buffer has already admitted instead of
        # silently merging with cached episodes.
        self._episode_offset = 0

    # ------------------------------------------------------------------
    # Admission
    # ------------------------------------------------------------------
    def _rebase_episode_offset(self, batch: OfficialFADARawTransitionBatch) -> None:
        """Lift the remap offset past already-admitted (env_id, episode_id) space."""

        if batch.num_samples <= 0:
            return
        episode_max = int(batch.episode_id.max().item())
        self._episode_offset = max(self._episode_offset, episode_max + 1)

    def _remap_episode_ids(
        self, batch: OfficialFADARawTransitionBatch
    ) -> OfficialFADARawTransitionBatch:
        """Collision-safely rebase incoming episode ids across worker restarts.

        A fresh native worker after resume restarts its local episode counters,
        so its ids would silently merge with cached episodes; shifting by the
        admitted (env, episode) upper bound keeps every trajectory distinct.
        """

        if batch.num_samples <= 0:
            return batch
        if not self._episode_offset:
            self._rebase_episode_offset(batch)
            return batch
        shifted = OfficialFADARawTransitionBatch(
            **{
                **{name: getattr(batch, name) for name in _OFFICIAL_RAW_FIELDS},
                "episode_id": batch.episode_id + self._episode_offset,
            }
        )
        self._rebase_episode_offset(shifted)
        return shifted

    def add(self, source: str, batch: OfficialFADARawTransitionBatch) -> int:
        """Admit one chronological raw batch into the addressed source buffer."""

        if source not in self._buffers:
            raise ValueError(f"unknown official physical source: {source!r}")
        if self._offline_frozen and source != "online":
            raise RuntimeError(
                f"official {source} replay is frozen after the warmup split; "
                "offline sources are collected exactly once"
            )
        if not isinstance(batch, OfficialFADARawTransitionBatch):
            raise TypeError(
                f"official replay admits OfficialFADARawTransitionBatch, got {type(batch)}"
            )
        batch.validate(self.spec)
        remapped = self._remap_episode_ids(batch)
        _append_official_raw(self._buffers[source], remapped)
        return batch.num_samples

    def _complete_episode_mask(
        self, env_id: torch.Tensor, episode_id: torch.Tensor, done_flags: torch.Tensor,
        required_length: int
    ) -> torch.Tensor:
        """Pinned ``_append_complete_episodes_from_buffer`` filter, verbatim.

        Per (env_id, episode_id) keep the episode iff its row count equals
        ``required_length``, its last row is done, and exactly one row is done;
        fallen or partial episodes are dropped entirely.
        """

        if required_length <= 0:
            raise ValueError("required_length must be positive")
        env_ids = env_id.tolist()
        episode_ids = episode_id.tolist()
        done = done_flags.tolist()
        positions: dict[tuple[int, int], list[int]] = {}
        for pos, key in enumerate(zip(env_ids, episode_ids)):
            positions.setdefault((int(key[0]), int(key[1])), []).append(pos)
        keep = torch.zeros(env_id.shape[0], dtype=torch.bool)
        for pos_list in positions.values():
            done_flags = [bool(done[pos]) for pos in pos_list]
            if (
                len(pos_list) == required_length
                and done_flags[-1]
                and sum(done_flags) == 1
            ):
                keep[torch.tensor(pos_list, dtype=torch.int64)] = True
        return keep

    def _admit_complete_episodes(
        self, batch: OfficialFADARawTransitionBatch, required_length: int
    ) -> int:
        keep = self._complete_episode_mask(batch.env_id, batch.episode_id, batch.done, required_length)
        kept = int(keep.sum().item())
        if kept <= 0:
            return 0
        filtered = OfficialFADARawTransitionBatch(
            **{name: getattr(batch, name)[keep] for name in _OFFICIAL_RAW_FIELDS}
        )
        return self._admit_shifted("optimal", filtered)

    def add_artifact(
        self, artifact: LoadedOfficialFADASourceArtifact, *, required_length: int | None = None
    ) -> int:
        """Admit one collected raw artifact, preserving shard order and flags.

        With ``required_length`` set (optimal offline rollouts only), each shard
        passes the pinned complete-episode filter before admission; suboptimal
        and online artifacts are admitted whole under their own original rules.
        """

        if not isinstance(artifact, LoadedOfficialFADASourceArtifact):
            raise TypeError(
                "official replay admits LoadedOfficialFADASourceArtifact, "
                f"got {type(artifact).__name__}"
            )
        if artifact.spec.history_length != self.spec.history_length or (
            artifact.spec.prediction_horizon != self.spec.prediction_horizon
        ):
            raise ValueError("official replay H/K identity mismatch")
        metadata = artifact.metadata
        if (
            validate_official_fada_preprocessing(metadata["preprocessing"])
            != self.preprocessing
        ):
            raise ValueError("official replay preprocessing identity mismatch")
        identity = (
            str(metadata["expert_checkpoint_sha256"]),
            str(metadata["experiment_config_sha256"]),
        )
        if self._expert_identity is not None and identity != self._expert_identity:
            raise ValueError("official replay cannot mix expert or experiment identities")
        source = str(metadata["source"])
        shards = tuple(artifact.iter_batches())
        if not shards:
            raise ValueError("official source artifact must contain at least one shard")
        # One remap offset per artifact: shards of one collection share the
        # worker's episode-id space, so each shard gets the same shift instead
        # of inflating the offset shard by shard.
        keep_all = None if required_length is None else self._complete_episode_mask(
            torch.cat([batch.env_id for batch in shards]),
            torch.cat([batch.episode_id for batch in shards]),
            torch.cat([batch.done for batch in shards]),
            required_length,
        )
        row = 0
        offset = self._episode_offset
        admitted = 0
        maximum_episode = -1
        for batch in shards:
            shifted = (
                batch
                if not offset
                else OfficialFADARawTransitionBatch(
                    **{
                        **{name: getattr(batch, name) for name in _OFFICIAL_RAW_FIELDS},
                        "episode_id": batch.episode_id + offset,
                    }
                )
            )
            if keep_all is None:
                admitted += self._admit_shifted(source, shifted)
            else:
                keep = keep_all[row : row + batch.num_samples]
                if bool(keep.any()):
                    admitted += self._admit_shifted(
                        source, OfficialFADARawTransitionBatch(
                            **{name: getattr(shifted, name)[keep] for name in _OFFICIAL_RAW_FIELDS}
                        )
                    )
            row += batch.num_samples
            maximum_episode = max(maximum_episode, int(shifted.episode_id.max()))
        self._episode_offset = max(self._episode_offset, maximum_episode + 1)
        self._expert_identity = identity
        return admitted

    def _admit_shifted(
        self, source: str, batch: OfficialFADARawTransitionBatch
    ) -> int:
        """Admit an already-remapped batch without re-shifting it."""

        if source not in self._buffers:
            raise ValueError(f"unknown official physical source: {source!r}")
        if self._offline_frozen and source != "online":
            raise RuntimeError(
                f"official {source} replay is frozen after the warmup split; "
                "offline sources are collected exactly once"
            )
        batch.validate(self.spec)
        _append_official_raw(self._buffers[source], batch)
        return batch.num_samples

    # ------------------------------------------------------------------
    # Offline caches (original NPZ schema + compact-obs provenance)
    # ------------------------------------------------------------------
    def _cache_provenance(self, *, require_identity: bool) -> dict[str, object]:
        """Original compact-obs cache provenance; actual expert identity, H/K bound by spec."""

        if require_identity and self._expert_identity is None:
            raise RuntimeError(
                "official replay cache provenance requires collected data first; "
                "refusing to fabricate a source checkpoint identity"
            )
        return {
            "term_scale": dict(self.preprocessing["term_scale"]),
            "term_noise": dict(self.preprocessing["term_noise"]),
            "add_noise": False,
            "noise_seed": None,
            "source_checkpoint_abs": str(
                Path(self.spec.expert_checkpoint).expanduser().resolve()
            ),
        }

    def save_offline_caches(
        self,
        *,
        optimal_path: str | Path,
        suboptimal_path: str | Path,
    ) -> None:
        """Persist the growable offline buffers with the original NPZ schema."""

        provenance = self._cache_provenance(require_identity=True)
        self._buffers["optimal"].save_npz(
            optimal_path, compact_obs_metadata=provenance
        )
        self._buffers["suboptimal"].save_npz(
            suboptimal_path, compact_obs_metadata=provenance
        )

    def load_offline_caches(
        self,
        *,
        optimal_path: str | Path,
        suboptimal_path: str | Path,
    ) -> dict[str, Any]:
        """Load original NPZ caches; only preprocessing/H/K/dims identity gates reuse.

        The full collection spec intentionally does not gate reuse: run-only and
        resume fields inside the spec config change across resumed runs while
        the cached data stays valid. A preprocessing metadata mismatch propagates
        as the original cache error so the caller can recollect.
        """

        stats = {
            "optimal_offline_cache_hit": False,
            "suboptimal_offline_cache_hit": False,
        }
        optimal = Path(optimal_path)
        suboptimal = Path(suboptimal_path)
        provenance = self._cache_provenance(require_identity=False)
        if optimal.is_file():
            loaded = self._buffers["optimal"].load_npz(
                optimal, expected_compact_obs_metadata=provenance
            )
            if loaded > 0:
                self._rebase_episode_offset(self._chronological_batch("optimal"))
            stats["optimal_offline_cache_hit"] = True
            stats["optimal_offline_loaded_transitions"] = int(loaded)
        if self.suboptimal_enabled and suboptimal.is_file():
            loaded = self._buffers["suboptimal"].load_npz(
                suboptimal, expected_compact_obs_metadata=provenance
            )
            stats["suboptimal_offline_cache_hit"] = True
            stats["suboptimal_offline_loaded_transitions"] = int(loaded)
        stats["offline_cache_hit"] = bool(stats["optimal_offline_cache_hit"]) and (
            not self.suboptimal_enabled or bool(stats["suboptimal_offline_cache_hit"])
        )
        return stats

    def save_online_buffer(self, path: str | Path) -> Path:
        """Persist the online buffer with the original NPZ schema."""

        target = self._buffers["online"].save_npz(
            path, compact_obs_metadata=self._cache_provenance(require_identity=True)
        )
        validation = self._val_buffers["online"]
        if validation is not None:
            validation.save_npz(
                target.with_name(f"{target.stem}_val.npz"),
                compact_obs_metadata=self._cache_provenance(require_identity=True),
            )
        return target

    def load_online_buffer(self, path: str | Path) -> int:
        """Restore the online buffer; restored episode ids rebase the remap offset."""

        loaded = self._buffers["online"].load_npz(
            path,
            expected_compact_obs_metadata=self._cache_provenance(require_identity=False),
        )
        if loaded > 0:
            self._rebase_episode_offset(self._chronological_batch("online"))
        validation_path = Path(path).with_name(f"{Path(path).stem}_val.npz")
        if validation_path.is_file():
            validation = self._new_online_validation_buffer()
            validation.load_npz(
                validation_path, expected_compact_obs_metadata=self._cache_provenance(require_identity=False)
            )
            self._val_buffers["online"] = validation
        return int(loaded)

    # ------------------------------------------------------------------
    # Warmup split / freeze (applied once, with the original seeds)
    # ------------------------------------------------------------------
    @property
    def suboptimal_enabled(self) -> bool:
        cfg = self.training_config
        planner_suboptimal = (
            float(cfg["planner_suboptimal_batch_ratio"]) > 0.0
            and float(cfg["suboptimal_data_ratio"]) > 0.0
        )
        idm_suboptimal = (
            bool(cfg["use_generalized_idm"])
            and float(cfg["suboptimal_data_ratio"]) > 0.0
        )
        return planner_suboptimal or idm_suboptimal

    def finalize_offline(self, *, val_ratio: float, seed: int) -> dict[str, int]:
        """Apply the original trajectory-level train split and freeze offline data.

        Mirrors upstream warmup: the optimal buffer splits with ``seed`` and the
        suboptimal buffer with ``seed + 1`` via the upstream
        ``split_by_trajectory``; with ``val_ratio <= 0`` the buffers are frozen
        in place. Offline buffers are never mutated afterwards.
        """

        if self._offline_frozen:
            raise RuntimeError("official offline replay split/freeze already applied")
        stats: dict[str, int] = {}
        for source, split_seed in (("optimal", int(seed)), ("suboptimal", int(seed) + 1)):
            buffer = self._buffers[source]
            if float(val_ratio) > 0.0 and len(buffer) > 0:
                # split_by_trajectory returns frozen train/val buffers.
                train_buf, _val_buf = buffer.split_by_trajectory(
                    val_ratio=float(val_ratio), seed=split_seed
                )
                self._buffers[source] = train_buf
                self._val_buffers[source] = _val_buf
                stats[f"{source}_train_size"] = len(train_buf)
                stats[f"{source}_val_size"] = len(_val_buf)
            else:
                buffer.freeze()
                stats[f"{source}_train_size"] = len(buffer)
                stats[f"{source}_val_size"] = 0
        self._offline_frozen = True
        return stats

    # ------------------------------------------------------------------
    # Online trajectory subsample / split (parent-only, once per round)
    # ------------------------------------------------------------------
    def _new_online_validation_buffer(self) -> Any:
        return self._replay_buffer_cls(
            capacity=max(int(self.resolved_config["replay_capacity"] * self.resolved_config["val_trajectory_ratio"]) + 1024, 4096),
            obs_dim=64, act_dim=29, cmd_dim=7, history_len=self.spec.history_length,
            pred_horizon=self.spec.prediction_horizon, require_future_obs_targets=True,
            eviction_policy=self.resolved_config["online_buffer_eviction_policy"],
        )

    def process_online_batch(
        self, batch: OfficialFADARawTransitionBatch, *, sample_ratio: float,
        val_ratio: float, seed: int, rollout_step_count: int,
    ) -> dict[str, int]:
        return self._process_online_batches(
            (batch,), count=batch.num_samples, sample_ratio=sample_ratio,
            val_ratio=val_ratio, seed=seed, rollout_step_count=rollout_step_count,
        )

    def process_online_artifact(
        self, artifact: LoadedOfficialFADASourceArtifact, *, sample_ratio: float,
        val_ratio: float, seed: int, rollout_step_count: int,
    ) -> dict[str, int]:
        """Apply trajectory selection once across all shards, not separately per shard."""

        identity = (
            artifact.metadata["expert_checkpoint_sha256"],
            artifact.metadata["experiment_config_sha256"],
        )
        if identity != self._expert_identity:
            raise ValueError("online artifact differs from the prepared source identity")
        return self._process_online_batches(
            artifact.iter_batches(), count=artifact.num_samples, sample_ratio=sample_ratio,
            val_ratio=val_ratio, seed=seed, rollout_step_count=rollout_step_count,
        )

    def _process_online_batches(
        self, batches: Iterable[OfficialFADARawTransitionBatch], *, count: int,
        sample_ratio: float, val_ratio: float, seed: int, rollout_step_count: int,
    ) -> dict[str, int]:
        temp = self._replay_buffer_cls(
            capacity=count, obs_dim=64, act_dim=29, cmd_dim=7,
            history_len=self.spec.history_length, pred_horizon=self.spec.prediction_horizon,
            require_future_obs_targets=True,
        )
        offset, maximum = self._episode_offset, -1
        for batch in batches:
            batch.validate(self.spec)
            shifted = OfficialFADARawTransitionBatch(
                **{**{name: getattr(batch, name) for name in _OFFICIAL_RAW_FIELDS},
                   "episode_id": batch.episode_id + offset}
            )
            _append_official_raw(temp, shifted)
            maximum = max(maximum, int(shifted.episode_id.max()))
        if len(temp) != count:
            raise ValueError("online artifact raw transition count mismatch")
        stats = {"collected_steps": count}
        if sample_ratio < 1:
            temp, _ = temp.split_by_trajectory(
                val_ratio=1.0 - sample_ratio, seed=seed + 2000 + rollout_step_count
            )
        stats["online_kept_steps"] = len(temp)
        if val_ratio > 0 and len(temp) > 0:
            temp, validation = temp.split_by_trajectory(
                val_ratio=val_ratio, seed=seed + 1000 + rollout_step_count
            )
            if self._val_buffers["online"] is None:
                self._val_buffers["online"] = self._new_online_validation_buffer()
            self._val_buffers["online"].add_from_buffer(validation)
            stats["online_val_size"] = len(validation)
        else:
            stats["online_val_size"] = 0
        if len(temp):
            self._buffers["online"].add_from_buffer(temp)
        self._episode_offset = max(self._episode_offset, maximum + 1)
        stats["online_buffer_size"] = len(self._buffers["online"])
        return stats

    # ------------------------------------------------------------------
    # Sampling (fresh upstream sample_chunk per update; Planner first)
    # ------------------------------------------------------------------
    def _chronological_batch(self, source: str) -> OfficialFADARawTransitionBatch:
        buffer = self._buffers[source]
        if len(buffer) <= 0:
            raise RuntimeError(f"official {source} replay is empty")
        return OfficialFADARawTransitionBatch.from_mapping(
            buffer._as_chronological_dict()
        ).validate(self.spec)

    def _sample_source(
        self, source: str, count: int, *, device: torch.device
    ) -> OfficialFADASourceBatch:
        buffer = self._buffers[source]
        if len(buffer) <= 0:
            raise RuntimeError(
                f"official {source} replay has no transitions for requested batch {count}"
            )
        with measure(self._performance, "learner_batch_staging", group="replay_sample"):
            result = buffer.sample_chunk(int(count))
            missing = sorted(set(OfficialFADASourceBatch.__dataclass_fields__) - set(result))
            extra = sorted(set(result) - set(OfficialFADASourceBatch.__dataclass_fields__))
            if missing or extra:
                raise RuntimeError(
                    "official upstream sample_chunk fields mismatch: "
                    f"missing={missing}, extra={extra}"
                )
            batch = OfficialFADASourceBatch.from_mapping(result)
        with measure(self._performance, "learner_batch_staging", group="h2d", cuda=True):
            staged = OfficialFADASourceBatch(
                **{
                    name: getattr(batch, name).to(device)
                    for name in OfficialFADASourceBatch.__dataclass_fields__
                }
            )
        if self._performance is not None:
            self._performance.increment(f"sampled_windows.{source}", int(count))
        return staged

    def sample_update(
        self,
        *,
        mode: str = "mixed",
        device: str | torch.device = "cpu",
        rng: np.random.Generator | None = None,
    ) -> tuple[dict[str, OfficialFADASourceBatch], OfficialFADASourceBatch]:
        """Draw Planner first, then the five IDM roles, in the original np order."""

        if mode == "offline" and not self._offline_frozen:
            raise RuntimeError(
                "official offline sampling requires finalize_offline (split/freeze) first"
            )
        if rng is not None:
            raise ValueError(
                "official replay sampling consumes the global np.random state exactly "
                "like upstream train.py; an explicit generator is not supported"
            )
        target_device = torch.device(device)
        planner_counts = official_replay_source_counts(
            self.training_config, mode=mode, planner=True
        )
        planner_parts = [
            self._sample_source(source, planner_counts[source], device=target_device)
            for source in ("optimal", "suboptimal", "online")
            if planner_counts[source]
        ]
        if not planner_parts:
            raise RuntimeError(f"official replay planner quota is empty in mode {mode!r}")
        planner_batch = (
            planner_parts[0]
            if len(planner_parts) == 1
            else OfficialFADASourceBatch(
                **{
                    name: torch.cat([getattr(part, name) for part in planner_parts], dim=0)
                    for name in OfficialFADASourceBatch.__dataclass_fields__
                }
            )
        )
        idm_counts = official_replay_source_counts(
            self.training_config, mode=mode, planner=False
        )
        roles = {"suboptimal_expert": "suboptimal", "online_trajectory": "online"}
        idm_batches = {
            role: self._sample_source(roles.get(role, role), size, device=target_device)
            for role, size in idm_counts.items()
            if size
        }
        return idm_batches, planner_batch

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return sum(len(buffer) for buffer in self._buffers.values())

    def size(self, source: str) -> int:
        if source not in self._buffers:
            raise ValueError(f"unknown official physical source: {source!r}")
        return len(self._buffers[source])

    def num_valid_chunks(self, source: str) -> int:
        if source not in self._buffers:
            raise ValueError(f"unknown official physical source: {source!r}")
        return int(self._buffers[source].num_valid_chunks())

    def iter_raw(
        self, source: str
    ) -> Iterator[OfficialFADARawTransitionBatch]:
        if source not in self._buffers:
            raise ValueError(f"unknown official physical source: {source!r}")
        if len(self._buffers[source]) > 0:
            yield self._chronological_batch(source)
