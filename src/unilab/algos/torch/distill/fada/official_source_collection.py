"""Official FADA raw-transition collection transaction.

Wraps the pinned upstream ``LeCAR-Lab/FADA`` trainer rollout helpers (commit
``OFFICIAL_FADA_SOURCE_REVISION``) to collect expert (optimal), weak-checkpoint
(suboptimal), and student (online) raw transitions inside the existing UniLab
collector worker, without creating optimizers, warmup, training loops, apps, or
processes here.

The collector runs exactly one upstream ``_collect_with_policy`` rollout per
request and exports every chronological raw transition through
``ReplayBuffer._as_chronological_dict``; trajectory subsampling, train/val
splitting, and window sampling are the parent Replay's single responsibility
and never happen here.

Upstream symbols are imported lazily inside the transaction so importing this
module never starts the native simulator stack; a missing checkout or missing
checkpoint fails with an explicit error instead of a fabricated batch.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import torch

from unilab.algos.torch.distill.fada.collection_contract import (
    OFFICIAL_FADA_SOURCE_REVISION,
    OfficialFADACollectionSpec,
    OfficialFADARawTransitionBatch,
)
from unilab.algos.torch.distill.fada.timing import (
    FADAPerformanceRecorder,
    measure,
)

__all__ = ["collect_official_fada_raw_transitions"]


def _load_upstream_replay_buffer(upstream_root: str) -> type:
    """Import the official ``ReplayBuffer`` from the pinned checkout lazily."""
    root = Path(upstream_root).expanduser().resolve()
    package_root = root / "src" / "holosoma"
    init_file = package_root / "holosoma" / "__init__.py"
    dataset_file = package_root / "holosoma" / "fada" / "common" / "dataset.py"
    if not init_file.is_file() or not dataset_file.is_file():
        raise FileNotFoundError(
            "official FADA checkout not found or incomplete: "
            f"expected holosoma package under {package_root} "
            f"(upstream_root={root}, pinned revision {OFFICIAL_FADA_SOURCE_REVISION})"
        )
    path_str = str(package_root)
    inserted = False
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
        inserted = True
    try:
        from holosoma.fada.common.dataset import ReplayBuffer
    except ImportError as exc:
        raise ImportError(
            "failed to import official holosoma.fada.common.dataset.ReplayBuffer from "
            f"{package_root}: {exc}"
        ) from exc
    finally:
        if inserted:
            try:
                sys.path.remove(path_str)
            except ValueError:
                pass
    return ReplayBuffer


def _num_envs(env: Any) -> int:
    num_envs = int(getattr(env, "num_envs", 0) or 0)
    if num_envs <= 0:
        raise RuntimeError(
            "official FADA env does not expose a positive num_envs; "
            "cannot size the collection transaction buffer"
        )
    return num_envs


def resolve_official_fada_rollout_steps(
    cfg: Any,
    *,
    source: str,
    num_envs: int,
    collect_steps: int | None,
) -> int:
    """Resolve the physical per-env control-step count for one upstream rollout.

    ``collect_steps`` (when given) is the parent's total raw transition budget
    across all envs; the upstream rollout only counts whole per-env control
    steps, so it is converted with ``ceil(collect_steps / num_envs)`` and the
    physical transition count stays ``steps * num_envs``. Without an explicit
    budget each optimal/suboptimal request uses ``warmup_max_steps_per_episode``;
    the parent owns repetition/retained-step targets. Online uses
    ``rollout_steps_per_iter``.
    """

    if num_envs <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs}")
    if collect_steps is not None:
        if isinstance(collect_steps, bool) or not isinstance(collect_steps, int) or collect_steps <= 0:
            raise ValueError(
                f"official FADA collect_steps must be a positive total transition budget, "
                f"got {collect_steps!r}"
            )
        return max(1, math.ceil(collect_steps / num_envs))
    if source in {"optimal", "suboptimal"}:
        horizon = int(cfg.warmup_max_steps_per_episode)
        if horizon <= 0:
            raise ValueError("official offline rollout horizon must be positive")
        return horizon
    if source == "online":
        rollout_steps = int(getattr(cfg, "rollout_steps_per_iter", 0) or 0)
        if rollout_steps <= 0:
            raise RuntimeError(
                "official FADA cfg.rollout_steps_per_iter must be positive for online "
                f"collection (got {rollout_steps!r}); refusing to guess a rollout length"
            )
        return rollout_steps
    raise ValueError(f"unsupported official FADA source: {source!r}")


def _collect_raw_transaction_buffer(
    resources: Any,
    *,
    source: str,
    weak_checkpoint: str | None,
    collect_steps: int | None,
    recorder: FADAPerformanceRecorder | None = None,
) -> tuple[Any, int]:
    """Run one upstream rollout transaction into a dedicated buffer.

    The transaction buffer is fresh per call, so no stale transitions from
    earlier transactions or from the trainer's training buffers can leak into
    the returned batch; the trainer's ``online_buffer``/``offline_buffer`` are
    never written here.
    """
    trainer = resources.trainer
    env = resources.env
    cfg = trainer.cfg
    spec: OfficialFADACollectionSpec = resources.spec
    replay_buffer_cls = _load_upstream_replay_buffer(spec.upstream_root)
    num_envs = _num_envs(env)
    rollout_steps = resolve_official_fada_rollout_steps(
        cfg,
        source=source,
        num_envs=num_envs,
        collect_steps=collect_steps,
    )

    buffer = trainer._make_collection_buffer(capacity=rollout_steps * num_envs + num_envs)
    if type(buffer) is not replay_buffer_cls:
        raise TypeError(
            "official FADA trainer collection buffer is not the pinned upstream "
            f"ReplayBuffer (got {type(buffer).__module__}.{type(buffer).__name__}, "
            f"expected {replay_buffer_cls.__module__}.{replay_buffer_cls.__name__})"
        )

    if source == "online":
        # Student policy drives the env with the configured DAgger exploration
        # noise; strict expert shadow labels come from the trainer's worker/env.
        kwargs: dict[str, Any] = {
            "num_steps": rollout_steps,
            "sigma": float(cfg.dagger_sigma),
            "execute_expert": False,
        }
    elif source == "optimal":
        # Final expert oracle drives the env; shadow chunks recorded per step.
        kwargs = {
            "num_steps": rollout_steps,
            "sigma": 0.0,
            "execute_expert": True,
        }
    elif source == "suboptimal":
        # The request-selected weak checkpoint drives the env; the strict-label
        # worker still labels with the final expert, so shadow targets stay
        # teacher-aligned.
        if not weak_checkpoint:
            raise ValueError(
                "official FADA suboptimal collection requires request.weak_checkpoint"
            )
        selected = resources.selected_weak_checkpoints
        if not selected:
            raise RuntimeError(
                "official FADA suboptimal collection requested before prepare resolved "
                "the official weak-checkpoint plan"
            )
        weak_path = Path(str(weak_checkpoint)).expanduser().resolve()
        if weak_path not in selected:
            raise ValueError(
                "official FADA suboptimal request weak checkpoint is not in the "
                f"prepare-selected official weak checkpoint list: {weak_path} not in "
                f"{[str(path) for path in selected]}"
            )
        if not weak_path.is_file():
            raise FileNotFoundError(
                f"official FADA weak checkpoint not found: {weak_path}"
            )
        trainer._maybe_load_expert_checkpoint(weak_path)
        kwargs = {
            "num_steps": rollout_steps,
            "sigma": 0.0,
            "execute_expert": False,
            "use_loaded_policy": True,
        }
    else:
        raise ValueError(f"unsupported official FADA source: {source!r}")

    # The completed upstream rollout transaction (the combined Oracle/shadow
    # rollout for optimal/suboptimal, the student DAgger rollout for online) is
    # timed as wall time; it is never a pure-physics or CUDA-kernel measurement.
    # group distinguishes it; the stage stays a known enum name.
    with measure(recorder, "total_elapsed", group="collect.rollout"):
        stats = trainer._collect_with_policy(
            env,
            target_buffer=buffer,
            progress_desc=f"Official {source.capitalize()} Collect",
            **kwargs,
        )
    return buffer, int(stats.get("collected_steps", 0))


def _export_raw_transition_batch(
    buffer: Any,
    *,
    spec: OfficialFADACollectionSpec,
    source: str,
    recorder: FADAPerformanceRecorder | None = None,
) -> OfficialFADARawTransitionBatch:
    if len(buffer) <= 0:
        raise RuntimeError(
            "official FADA source collection produced no raw transitions: "
            f"source={source!r}; refusing to fabricate samples"
        )
    # tensor_pack times the raw CPU tensor packing/validation of the completed
    # transaction buffer; it is wall time over CPU-resident data, not CUDA kernels.
    with measure(recorder, "tensor_pack", group="collect"):
        mapping = buffer._as_chronological_dict()
        if not isinstance(mapping, dict) or not mapping:
            raise RuntimeError(
                "official FADA ReplayBuffer._as_chronological_dict returned no data for "
                f"source={source!r}"
            )
        batch = OfficialFADARawTransitionBatch.from_mapping(mapping)
        return batch.validate(spec)


def collect_official_fada_raw_transitions(
    resources: Any,
    *,
    source: str,
    weak_checkpoint: str | None = None,
    collect_steps: int | None = None,
    recorder: FADAPerformanceRecorder | None = None,
) -> tuple[OfficialFADARawTransitionBatch, int]:
    """Collect one official FADA raw-transition batch from the worker-held resources.

    ``resources`` must expose ``spec`` (``OfficialFADACollectionSpec``), ``cfg``
    (upstream ``FADAConfig``), ``env`` (official training env), ``trainer``
    (upstream ``FADATrainer`` constructed with ``optimizer=None`` purely for its
    collection helpers), ``student_policy``, and ``selected_weak_checkpoints``
    (the prepare-time official weak-checkpoint plan). No training, warmup,
    optimizer step, or run loop is triggered; the official trainer's rollout
    methods are the only collection mechanism, so counter/episode semantics,
    strict expert shadow labels, and executed-trajectory targets match upstream
    exactly.

    Returns the chronological raw batch and the physical transition count
    (``steps * num_envs``) reported by the upstream rollout.
    """
    spec: OfficialFADACollectionSpec = resources.spec
    if not isinstance(spec, OfficialFADACollectionSpec):
        raise TypeError(
            "official FADA collection requires resources.spec to be an "
            f"OfficialFADACollectionSpec, got {type(spec).__name__}"
        )

    trainer = resources.trainer
    missing = [
        name
        for name in ("cfg", "env", "trainer", "student_policy")
        if getattr(resources, name, None) is None
    ]
    if missing:
        raise RuntimeError(
            "official FADA collection resources are incomplete: "
            f"missing {', '.join(missing)}"
        )
    if not bool(trainer._use_teacher_aligned_labels):
        raise RuntimeError(
            "official FADA collection requires teacher-aligned strict labels "
            "(strict_chunk_labels or predict_future_obs); the trainer was built "
            "without them"
        )
    student = resources.student_policy
    if student is not trainer.student_policy:
        raise RuntimeError(
            "official FADA online collection requires resources.student_policy to be "
            "the trainer's student model; a detached substitute would break the "
            "upstream rollout semantics"
        )

    final_expert = Path(trainer.configured_expert_checkpoint).expanduser().resolve()
    try:
        # The total envelope covers the actual rollout transaction plus raw CPU
        # tensor packing; the expert-checkpoint restore in ``finally`` is not
        # part of the measured transaction. Stage stays a known enum name.
        with measure(recorder, "total_elapsed", group="collect"):
            buffer, collected_steps = _collect_raw_transaction_buffer(
                resources,
                source=source,
                weak_checkpoint=weak_checkpoint,
                collect_steps=collect_steps,
                recorder=recorder,
            )
            batch = _export_raw_transition_batch(
                buffer, spec=spec, source=source, recorder=recorder
            )
        return batch, collected_steps
    finally:
        # The strict-label shadow must always label with the final expert; restore
        # it after any weak-checkpoint suboptimal transaction (and after failures).
        trainer._maybe_load_expert_checkpoint(final_expert)
