"""Shared-weight owner for persistent distillation collection.

状态: HP-3b2 active, workflow-compatible, production resource worker wired.
上游: persistent DAgger workflow checkpoint activation.
下游: ``SharedWeightSync`` and ``PersistentDaggerCollectorRunner``.
证据: S1/S2 contract-confirmed with a resident spawned toy model.
边界: real G1 teachers, envs, and scenario collection belong to the worker owner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal, Mapping

import torch

from unilab.algos.torch.distill.runtime.async_runtime import (
    DaggerCollectorService,
    DaggerCollectRequest,
    DaggerCollectResult,
    DaggerPrepareRequest,
    DaggerPrepareResult,
    PersistentDaggerCollectorRunner,
)
from unilab.ipc import SharedWeightSync

_WEIGHT_SYNC_KEYS = frozenset({"weight_sync_name", "weight_sync_lock", "weight_param_shapes"})


class PersistentDistillationRuntime:
    """Publish checkpoint weights to a resident or request-scoped collector."""

    def __init__(
        self,
        *,
        student_loader: Callable[[Path], torch.nn.Module],
        worker_factory: Callable[..., DaggerCollectorService],
        worker_kwargs: Mapping[str, Any] | None = None,
        worker_kwargs_factory: Callable[[Path], Mapping[str, Any]] | None = None,
        lifecycle_report_queue: Any | None = None,
        request_timeout_seconds: float = 300.0,
        worker_lifecycle: Literal["persistent", "request"] = "persistent",
        collector_daemon: bool = True,
        initial_student: torch.nn.Module | None = None,
    ) -> None:
        resolved_worker_kwargs = dict(worker_kwargs or {})
        collisions = sorted(_WEIGHT_SYNC_KEYS & resolved_worker_kwargs.keys())
        if collisions:
            raise ValueError(
                f"worker_kwargs must not override runtime-owned weight sync fields: {collisions}"
            )
        self._student_loader = student_loader
        self._worker_factory = worker_factory
        self._base_worker_kwargs = resolved_worker_kwargs
        self._worker_kwargs_factory = worker_kwargs_factory
        self._lifecycle_report_queue = lifecycle_report_queue
        self.close_report: dict[str, Any] | None = None
        self._request_timeout_seconds = float(request_timeout_seconds)
        self._worker_lifecycle: Literal["persistent", "request"] = worker_lifecycle
        self._collector_daemon = collector_daemon
        self._weight_sync: SharedWeightSync | None = None
        self._runner: PersistentDaggerCollectorRunner | None = None
        self._param_shapes: dict[str, torch.Size] | None = None
        self._pending_path: str | None = None
        self._pending_state_dict: dict[str, torch.Tensor] | None = None
        self._closed = False
        self._initial_student = initial_student

    @property
    def weight_version(self) -> int:
        return 0 if self._weight_sync is None else self._weight_sync.version

    def _load_state_dict(self, checkpoint_path: Path) -> dict[str, torch.Tensor]:
        student = self._student_loader(checkpoint_path)
        state_dict = {name: value.detach().cpu() for name, value in student.state_dict().items()}
        if not state_dict:
            raise ValueError(f"student checkpoint has an empty state_dict: {checkpoint_path}")
        return state_dict

    def _validate_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        assert self._param_shapes is not None
        expected_names = set(self._param_shapes)
        actual_names = set(state_dict)
        if actual_names != expected_names:
            raise ValueError(
                "checkpoint state key mismatch: "
                f"missing={sorted(expected_names - actual_names)}, "
                f"unexpected={sorted(actual_names - expected_names)}"
            )
        mismatches = {
            name: (tuple(self._param_shapes[name]), tuple(state_dict[name].shape))
            for name in sorted(expected_names)
            if tuple(state_dict[name].shape) != tuple(self._param_shapes[name])
        }
        if mismatches:
            raise ValueError(f"checkpoint state shape mismatch: {mismatches}")

    def _ensure_runtime(
        self,
        checkpoint_path: Path | None,
        state_dict: dict[str, torch.Tensor],
    ) -> None:
        if self._runner is not None:
            return
        if checkpoint_path is None and self._worker_kwargs_factory is not None:
            raise ValueError("prepare cannot use checkpoint-derived worker kwargs")
        self._param_shapes = {name: value.shape for name, value in state_dict.items()}
        self._weight_sync = SharedWeightSync(self._param_shapes)
        dynamic_worker_kwargs = (
            {}
            if self._worker_kwargs_factory is None
            else dict(self._worker_kwargs_factory(checkpoint_path))
        )
        dynamic_collisions = sorted(
            (_WEIGHT_SYNC_KEYS | self._base_worker_kwargs.keys()) & dynamic_worker_kwargs.keys()
        )
        if dynamic_collisions:
            raise ValueError(
                "dynamic worker kwargs collide with runtime-owned/base fields: "
                f"{dynamic_collisions}"
            )
        worker_kwargs = {
            **self._base_worker_kwargs,
            **dynamic_worker_kwargs,
            "weight_sync_name": self._weight_sync.name,
            "weight_sync_lock": self._weight_sync._lock,
            "weight_param_shapes": self._param_shapes,
        }
        self._pending_path = None if checkpoint_path is None else str(checkpoint_path.resolve())
        self._pending_state_dict = None if checkpoint_path is None else state_dict
        self._runner = PersistentDaggerCollectorRunner(
            worker_factory=self._worker_factory,
            worker_kwargs=worker_kwargs,
            checkpoint_activator=self._publish_checkpoint,
            request_timeout_seconds=self._request_timeout_seconds,
            worker_lifecycle=self._worker_lifecycle,
            collector_daemon=self._collector_daemon,
        )

    def _publish_checkpoint(self, checkpoint_path: Path) -> int:
        resolved_path = str(checkpoint_path.resolve())
        if self._pending_path == resolved_path and self._pending_state_dict is not None:
            state_dict = self._pending_state_dict
        else:
            state_dict = self._load_state_dict(checkpoint_path)
        self._validate_state_dict(state_dict)
        assert self._weight_sync is not None
        self._weight_sync.write_weights(state_dict)
        self._pending_path = None
        self._pending_state_dict = None
        return self._weight_sync.version

    def activate_checkpoint(self, checkpoint_path: Path) -> int:
        if self._closed:
            raise RuntimeError("cannot activate a closed persistent distillation runtime")
        resolved_path = Path(checkpoint_path).resolve()
        state_dict = self._load_state_dict(resolved_path)
        self._ensure_runtime(resolved_path, state_dict)
        assert self._param_shapes is not None
        self._validate_state_dict(state_dict)
        self._pending_path = str(resolved_path)
        self._pending_state_dict = state_dict
        assert self._runner is not None
        return self._runner.activate_checkpoint(resolved_path)

    def prepare(self, request: DaggerPrepareRequest) -> DaggerPrepareResult:
        if self._closed:
            raise RuntimeError("cannot prepare a closed persistent distillation runtime")
        if self._runner is None:
            if self._initial_student is None:
                raise RuntimeError("prepare requires an explicit initial student")
            state = {
                name: value.detach().cpu()
                for name, value in self._initial_student.state_dict().items()
            }
            if not state:
                raise ValueError("initial student has an empty state_dict")
            self._ensure_runtime(None, state)
            self._initial_student = None
        assert self._runner is not None
        return self._runner.prepare(request)

    def collect(self, request: DaggerCollectRequest) -> DaggerCollectResult:
        if self._runner is None:
            raise RuntimeError("activate_checkpoint must run before collect")
        return self._runner.collect(request)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._runner is not None:
                self._runner.close()
        finally:
            if self._weight_sync is not None:
                self._weight_sync.cleanup()
                self._weight_sync = None
            if self._lifecycle_report_queue is not None:
                try:
                    self.close_report = self._lifecycle_report_queue.get(timeout=2.0)
                except Exception:
                    self.close_report = None
                close_queue = getattr(self._lifecycle_report_queue, "close", None)
                if callable(close_queue):
                    close_queue()
