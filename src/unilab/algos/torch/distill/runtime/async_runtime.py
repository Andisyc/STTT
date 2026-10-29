"""Persistent collector runtime protocol for DAgger workflows.

DAgger owns request and result semantics. Process lifecycle and collector
failure propagation remain owned by :class:`unilab.ipc.async_runner.AsyncRunner`.

状态: active OFF-default persistent collector runtime, legacy 默认路径不变.
上游: ``run_multirole_dagger_workflow(execution_mode="persistent_async")``.
下游: persistent worker request/result 与 parent-side checkpoint activator.
证据: S1/S2 contract-confirmed; E61/E65/E67 S4 bounded G1 lifecycle, timing,
and legacy/persistent A/B runtime-confirmed.
缺口: end-to-end stable speedup 未证明(``NO_STABLE_SPEEDUP``); HP-6 production
gate 与 physical policy acceptance 仍未完成.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Protocol, cast

from unilab.ipc.async_runner import AsyncRunner

_SPAWN_CTX = mp.get_context("spawn")


@dataclass(frozen=True)
class DaggerCollectRequest:
    """One scenario collection request at a fixed outer-iteration barrier."""

    request_id: str
    scenario: str
    iteration: int
    checkpoint_path: str
    output_path: str
    expected_weight_version: int


@dataclass(frozen=True)
class DaggerCollectResult:
    """Identity and structured metrics returned by a persistent collector."""

    request_id: str
    scenario: str
    iteration: int
    checkpoint_path: str
    output_path: str
    expected_weight_version: int
    observed_weight_version: int
    num_samples: int
    worker_pid: int
    metrics: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DaggerPrepareRequest:
    request_id: str


@dataclass(frozen=True)
class DaggerPrepareResult:
    request_id: str
    worker_pid: int
    metadata: Mapping[str, Any]


class DaggerCollectorService(Protocol):
    """Worker-local service initialized once and reused across requests."""

    def collect(self, request: DaggerCollectRequest) -> DaggerCollectResult: ...

    def close(self) -> None: ...


class PreparedDaggerCollectorService(DaggerCollectorService, Protocol):
    def prepare(self, request: DaggerPrepareRequest) -> DaggerPrepareResult: ...


def validate_dagger_collect_result(
    request: DaggerCollectRequest,
    result: DaggerCollectResult,
) -> None:
    """Fail closed when a worker result crosses its request barrier."""

    identity_fields = (
        "request_id",
        "scenario",
        "iteration",
        "checkpoint_path",
        "output_path",
        "expected_weight_version",
    )
    mismatches = [
        name for name in identity_fields if getattr(request, name) != getattr(result, name)
    ]
    if mismatches:
        raise ValueError(f"persistent DAgger collector result identity mismatch: {mismatches}")
    if result.observed_weight_version != request.expected_weight_version:
        raise ValueError(
            "persistent DAgger collector weight version mismatch: "
            f"expected {request.expected_weight_version}, "
            f"observed {result.observed_weight_version}"
        )
    if result.num_samples <= 0:
        raise ValueError(
            "persistent DAgger collector must return a positive sample count, "
            f"got {result.num_samples}"
        )


def _persistent_dagger_collector_entry(
    *,
    stop_event: Any,
    request_queue: Any,
    result_queue: Any,
    worker_factory: Callable[..., DaggerCollectorService],
    worker_kwargs: Mapping[str, Any],
) -> None:
    """Spawn-safe service loop; exceptions escape to AsyncRunner's guard."""

    service = worker_factory(**dict(worker_kwargs))
    collect_error: BaseException | None = None
    try:
        while not stop_event.is_set():
            try:
                request = request_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if request is None:
                break
            if isinstance(request, DaggerPrepareRequest):
                result_queue.put(cast(PreparedDaggerCollectorService, service).prepare(request))
            else:
                result_queue.put(service.collect(request))
    except BaseException as exc:
        collect_error = exc
        raise
    finally:
        try:
            service.close()
        except BaseException as cleanup_error:
            if collect_error is None:
                raise
            print(
                "[persistent DAgger collector] cleanup failed while preserving "
                "the primary collection failure:",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exception(cleanup_error, file=sys.stderr)


class PersistentDaggerCollectorRunner(AsyncRunner):
    """Synchronous request facade backed by an explicitly scoped spawned worker.

    Collection remains sequential here to preserve the DAgger outer barrier.
    The default keeps one resident worker for compatible workflows. Request scope
    retires native/CUDA state after every completed collection transaction.
    """

    def __init__(
        self,
        *,
        worker_factory: Callable[..., DaggerCollectorService],
        worker_kwargs: Mapping[str, Any] | None = None,
        checkpoint_activator: Callable[[Path], int] | None = None,
        request_timeout_seconds: float = 300.0,
        worker_lifecycle: Literal["persistent", "request"] = "persistent",
        collector_daemon: bool = True,
    ) -> None:
        super().__init__(
            env_name="DAggerPersistentCollector",
            env_cfg_overrides={},
            rl_cfg={},
            device="cpu",
            collector_device="cpu",
            num_envs=1,
            collector_daemon=collector_daemon,
        )
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        if worker_lifecycle not in {"persistent", "request"}:
            raise ValueError(
                f"worker_lifecycle must be 'persistent' or 'request', got {worker_lifecycle!r}"
            )
        self._worker_factory = worker_factory
        self._worker_kwargs = dict(worker_kwargs or {})
        self._checkpoint_activator = checkpoint_activator
        self._request_timeout_seconds = float(request_timeout_seconds)
        self._worker_lifecycle = worker_lifecycle
        self._request_queue = _SPAWN_CTX.Queue(maxsize=1)
        self._result_queue = _SPAWN_CTX.Queue(maxsize=1)
        self._shared_resources.extend([self._request_queue, self._result_queue])
        self._closed = False
        self._active_checkpoint_path: str | None = None
        self._active_weight_version: int | None = None

    def _get_default_device(self) -> str:
        return "cpu"

    def _build_learner(self) -> None:
        return None

    def _collector_fn(self, stop_event: Any, **kwargs: Any) -> None:
        _persistent_dagger_collector_entry(stop_event=stop_event, **kwargs)

    def learn(
        self,
        max_iterations: int,
        save_interval: int = 50,
        log_dir: str = "logs",
    ) -> None:
        raise NotImplementedError("DAgger workflow owns the outer learning loop")

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("cannot restart a closed persistent DAgger collector")
        if self._collector_process is not None:
            if self._collector_process.is_alive():
                return
            raise RuntimeError(self._read_collector_error())
        self._start_collector(
            target_fn=_persistent_dagger_collector_entry,
            kwargs={
                "stop_event": self._stop_event,
                "request_queue": self._request_queue,
                "result_queue": self._result_queue,
                "worker_factory": self._worker_factory,
                "worker_kwargs": self._worker_kwargs,
                "_error_label": "persistent DAgger collector",
            },
        )

    def activate_checkpoint(self, checkpoint_path: Path) -> int:
        """Publish one checkpoint through the configured parent-side owner."""

        if self._checkpoint_activator is None:
            raise RuntimeError("persistent DAgger collector has no checkpoint_activator")
        resolved_path = str(Path(checkpoint_path).resolve())
        version = int(self._checkpoint_activator(Path(resolved_path)))
        if version < 0:
            raise ValueError(f"checkpoint_activator returned negative version {version}")
        self._active_checkpoint_path = resolved_path
        self._active_weight_version = version
        return version

    def collect(self, request: DaggerCollectRequest) -> DaggerCollectResult:
        if self._checkpoint_activator is not None:
            if self._active_checkpoint_path is None or self._active_weight_version is None:
                raise RuntimeError("activate_checkpoint must run before collect")
            if request.checkpoint_path != self._active_checkpoint_path:
                raise ValueError(
                    "persistent DAgger request checkpoint does not match active checkpoint: "
                    f"request={request.checkpoint_path!r}, "
                    f"active={self._active_checkpoint_path!r}"
                )
            if request.expected_weight_version != self._active_weight_version:
                raise ValueError(
                    "persistent DAgger request version does not match active version: "
                    f"request={request.expected_weight_version}, "
                    f"active={self._active_weight_version}"
                )
        return cast(DaggerCollectResult, self._exchange(request))

    def prepare(self, request: DaggerPrepareRequest) -> DaggerPrepareResult:
        if self._worker_lifecycle != "persistent":
            raise ValueError("prepare requires a persistent worker lifecycle")
        return cast(DaggerPrepareResult, self._exchange(request))

    def _exchange(
        self, request: DaggerCollectRequest | DaggerPrepareRequest
    ) -> DaggerCollectResult | DaggerPrepareResult:
        self.start()
        primary_error: BaseException | None = None
        try:
            self._request_queue.put(request)
            deadline = time.monotonic() + self._request_timeout_seconds
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"persistent DAgger collector timed out for request {request.request_id!r}"
                    )
                try:
                    result = self._result_queue.get(timeout=min(0.1, remaining))
                except queue.Empty:
                    if (
                        self._collector_process is not None
                        and not self._collector_process.is_alive()
                    ):
                        raise RuntimeError(self._read_collector_error())
                    continue
                if isinstance(request, DaggerPrepareRequest):
                    if not isinstance(result, DaggerPrepareResult) or result.request_id != request.request_id:
                        raise ValueError("persistent DAgger prepare result identity mismatch")
                else:
                    if not isinstance(result, DaggerCollectResult):
                        raise ValueError("expected a persistent DAgger collection result")
                    validate_dagger_collect_result(request, result)
                return result
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            if self._worker_lifecycle == "request":
                retirement_error: BaseException | None = None
                try:
                    self._retire_request_worker()
                except BaseException as exc:
                    retirement_error = exc
                if primary_error is not None or retirement_error is not None:
                    # A failed request can leave an unread queue item or uncertain native
                    # teardown. Retire the whole facade instead of allowing stale IPC state
                    # to cross into a later request.
                    self.close()
                if primary_error is None and retirement_error is not None:
                    raise retirement_error

    def _retire_request_worker(self) -> None:
        """Retire, join, and close the worker that owned one request transaction."""

        process = self._collector_process
        if process is None:
            return
        if process.is_alive():
            try:
                self._request_queue.put(None, timeout=0.1)
            except queue.Full:
                pass
            process.join(timeout=10.0)
        retirement_error: RuntimeError | None = None
        if process.is_alive():
            if not self._signal_collector_group(signal.SIGTERM):
                process.terminate()
            process.join(timeout=5.0)
            if process.is_alive():
                self._signal_collector_group(signal.SIGKILL)
                process.kill()
                process.join(timeout=5.0)
            retirement_error = RuntimeError(
                "request-scoped DAgger collector did not retire after its transaction"
            )
        elif process.exitcode != 0:
            retirement_error = RuntimeError(self._read_collector_error())
        self._signal_collector_group(signal.SIGKILL)
        process.close()
        self._collector_process = None
        if self._error_recv is not None:
            self._error_recv.close()
            self._error_recv = None
        if self._error_send is not None:
            self._error_send.close()
            self._error_send = None
        if retirement_error is not None:
            raise retirement_error

    @staticmethod
    def validate_result(
        request: DaggerCollectRequest,
        result: DaggerCollectResult,
    ) -> None:
        validate_dagger_collect_result(request, result)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._collector_process is not None and self._collector_process.is_alive():
                try:
                    self._request_queue.put_nowait(None)
                except queue.Full:
                    pass
        finally:
            super().close()
