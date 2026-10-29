"""Opt-in owner-local timing, separate from FADA learning and checkpoint identity."""

from __future__ import annotations

import math
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from typing import Any

import torch

from unilab.algos.torch.distill.observability.performance import (
    DistillationStageObservationAccumulator,
)


_NO_MEASUREMENT = nullcontext()


class FADAPerformanceRecorder:
    """Accumulate wall time and optional current-stream CUDA events.

    Only construct this object when profiling is enabled. CUDA events are
    resolved at reporting boundaries, never by synchronizing each operator.
    Parent/RPC and worker durations are envelopes and children, not additive.
    """

    def __init__(self, *, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self._started = time.perf_counter()
        self._groups: dict[str, DistillationStageObservationAccumulator] = {}
        self._stages: dict[str, set[str]] = {}
        self._calls: dict[str, int] = {}
        self._counters: dict[str, int] = {}
        self._worker_wall: dict[str, float] = {}
        self._pending_cuda: list[tuple[str, Any, Any]] = []
        self._cuda_seconds: dict[str, float] = {}

    @contextmanager
    def measure(self, stage: str, *, group: str = "", cuda: bool = False) -> Iterator[None]:
        accumulator = self._groups.get(group)
        if accumulator is None:
            accumulator = DistillationStageObservationAccumulator(clock=time.perf_counter)
            self._groups[group] = accumulator
            self._stages[group] = set()
        key = f"{group}.{stage}" if group else stage
        self._stages[group].add(stage)
        start = end = None
        with accumulator.measure(stage):
            if cuda and self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record(torch.cuda.current_stream(self.device))
            try:
                yield
            finally:
                if end is not None:
                    with torch.cuda.device(self.device):
                        end.record(torch.cuda.current_stream(self.device))
                    self._pending_cuda.append((key, start, end))
                self._calls[key] = self._calls.get(key, 0) + 1

    def increment(self, name: str, value: int = 1) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("performance counters must be non-negative integers")
        self._counters[name] = self._counters.get(name, 0) + value

    def add_worker_metrics(self, metrics: Mapping[str, Any], *, source: str) -> None:
        """Preserve only measured worker durations, without inventing absent stages."""
        for name, value in metrics.items():
            if not name.endswith("_seconds"):
                continue
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise ValueError(f"invalid worker duration: {name}")
            duration = float(value)
            if not math.isfinite(duration) or duration < 0:
                raise ValueError(f"invalid worker duration: {name}")
            key = f"{source}.{name}"
            self._worker_wall[key] = self._worker_wall.get(key, 0.0) + duration

    def summary(self, *, finished: bool = False) -> dict[str, Any]:
        if self._pending_cuda:
            # Resolve a batch of events once at a reporting boundary. These
            # durations cover this process's streams, not shadow subprocesses.
            torch.cuda.synchronize(self.device)
            for key, start, end in self._pending_cuda:
                self._cuda_seconds[key] = (
                    self._cuda_seconds.get(key, 0.0) + start.elapsed_time(end) / 1000.0
                )
            self._pending_cuda.clear()
        wall = {
            f"{group}.{stage}" if group else stage: accumulator.observation(
                stage=stage, row_count=0, env_step_count=0
            ).duration_seconds
            for group, accumulator in self._groups.items()
            for stage in sorted(self._stages[group])
        }
        if finished:
            wall["total_elapsed"] = time.perf_counter() - self._started
        return {
            "wall_seconds": wall,
            "cuda_stream_seconds": dict(self._cuda_seconds),
            "worker_wall_seconds": dict(self._worker_wall),
            "calls": dict(self._calls),
            "counters": dict(self._counters),
        }


def measure(
    recorder: FADAPerformanceRecorder | None,
    stage: str,
    *,
    group: str = "",
    cuda: bool = False,
):
    """Disabled paths allocate no timer, event, tensor, or generator context."""
    if recorder is None:
        return _NO_MEASUREMENT
    return recorder.measure(stage, group=group, cuda=cuda)
