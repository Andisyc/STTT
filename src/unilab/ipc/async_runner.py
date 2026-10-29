"""Base async runner for multi-process RL training."""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import sys
from abc import ABC, abstractmethod
from typing import Any, Callable

from unilab.ipc.collector_error import (
    ExceptionWrapper,
    collector_error_guard,
    create_error_pipe,
    format_collector_death,
)

_SPAWN_CTX = mp.get_context("spawn")


def _enable_collector_faulthandler() -> None:
    """Dump a Python-level stack on fatal native signals in collector subprocesses.

    Collectors can die inside native code (MuJoCo/CUDA) as a bare signal
    (e.g. SIGSEGV) with no Python traceback, leaving the parent with only
    "killed by signal 11". faulthandler prints the faulting thread's Python
    stack to stderr and then re-raises the default handler, so process
    termination semantics are unchanged. Best-effort diagnostics only.
    """
    try:
        import faulthandler

        if not faulthandler.is_enabled():
            faulthandler.enable(all_threads=True)
    except Exception:  # pragma: no cover - diagnostics must never break collection
        pass


def _collector_entry_wrapper(
    target_fn: Callable,
    error_conn: Any,
    kwargs: dict,
) -> None:
    """Top-level wrapper for collector subprocess entry point.

    Ensures ordinary Python exceptions (including import errors and env creation
    failures) are captured and sent to the parent via the error pipe.

    Status: active. 诊断开关启用时, collector 异常先写入错误通道, 随后在
    collector 本进程 abort, 以便 core dump 保留真实 collector native 状态.
    """
    label = kwargs.pop("_error_label", "collector")
    if kwargs.pop("_collector_session", False) and os.name == "posix":
        os.setsid()
    _enable_collector_faulthandler()
    try:
        with collector_error_guard(
            error_conn=error_conn,
            metrics_queue=kwargs.get("metrics_queue"),
            stop_event=kwargs.get("stop_event"),
            label=label,
        ):
            target_fn(**kwargs)
    except BaseException:
        if os.environ.get("UNILAB_NATIVE_ABORT_ON_CORRUPTION", "0") == "1":
            # 对普通 Exception, guard 已先发送 traceback. 此处只制造 collector-owned core.
            sys.stdout.flush()
            sys.stderr.flush()
            os.abort()
        raise


class AsyncRunner(ABC):
    """Base class for async RL algorithms.

    Manages:
    - Shared memory allocation/cleanup
    - Collector process lifecycle
    - Error propagation from collector subprocess
    - Training loop skeleton
    """

    def __init__(
        self,
        env_name: str,
        env_cfg_overrides: dict,
        rl_cfg: dict,
        *,
        device: str | None = None,
        collector_device: str | None = None,
        sim_backend: str = "mujoco",
        num_envs: int = 4096,
        collector_daemon: bool = True,
    ):
        self.env_name = env_name
        self.env_cfg_overrides = env_cfg_overrides
        self.rl_cfg = rl_cfg
        self.device = device or self._get_default_device()
        self.collector_device = collector_device or self.device
        self.sim_backend = sim_backend
        self.num_envs = num_envs
        self._collector_daemon = collector_daemon
        self._collector_group_closed = False

        self._collector_process: Any = None
        self._stop_event = _SPAWN_CTX.Event()
        self._shared_resources: list = []
        self._error_recv: Any = None
        self._error_send: Any = None

    @abstractmethod
    def _get_default_device(self) -> str:
        """Get default device (backend-specific)."""
        ...

    @abstractmethod
    def _build_learner(self) -> Any: ...

    @abstractmethod
    def _collector_fn(self, stop_event: Any, **kwargs) -> None: ...

    @abstractmethod
    def learn(
        self, max_iterations: int, save_interval: int = 50, log_dir: str = "logs"
    ) -> None: ...

    def _start_collector(self, target_fn: Callable, kwargs: dict) -> None:
        self._error_recv, self._error_send = create_error_pipe()
        kwargs = dict(kwargs)
        kwargs["_collector_session"] = not self._collector_daemon
        self._collector_group_closed = False

        self._collector_process = _SPAWN_CTX.Process(
            target=_collector_entry_wrapper,
            args=(target_fn, self._error_send, kwargs),
            daemon=self._collector_daemon,
        )
        self._collector_process.start()
        self._error_send.close()
        self._error_send = None

    def _check_collector_alive(self) -> bool:
        """Check if collector is alive. Prints full diagnostic if dead."""
        if self._collector_process is None:
            return True
        if self._collector_process.is_alive():
            return True

        death_info = self._read_collector_error()
        print(f"\n{death_info}\n", file=sys.stderr, flush=True)
        return False

    def _read_collector_error(self) -> str:
        """Read error info from dead collector — pipe first, then exit code."""
        traceback_text = None
        if self._error_recv is not None:
            try:
                if self._error_recv.poll(timeout=0.1):
                    obj = self._error_recv.recv()
                    if isinstance(obj, ExceptionWrapper):
                        traceback_text = obj.exc_msg
            except (EOFError, OSError):
                pass

        exitcode = getattr(self._collector_process, "exitcode", None)
        return format_collector_death(exitcode, traceback_text)

    def _signal_collector_group(self, sig: int) -> bool:
        process = self._collector_process
        if (
            self._collector_daemon
            or self._collector_group_closed
            or os.name != "posix"
            or process is None
        ):
            return False
        if sig == signal.SIGKILL:
            self._collector_group_closed = True
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return False
        return True

    def close(self) -> None:
        self._stop_event.set()
        if self._collector_process is not None and self._collector_process.is_alive():
            self._collector_process.join(timeout=10)
            if self._collector_process.is_alive():
                if not self._signal_collector_group(signal.SIGTERM):
                    self._collector_process.terminate()
                self._collector_process.join(timeout=5)
                if self._collector_process.is_alive():
                    self._signal_collector_group(signal.SIGKILL)
                    self._collector_process.kill()
                    self._collector_process.join(timeout=5)
        # Native crashes can leave a shadow child after the collector has exited.
        self._signal_collector_group(signal.SIGKILL)

        if self._collector_process is not None:
            exitcode = getattr(self._collector_process, "exitcode", None)
            # -15 (SIGTERM) is expected during normal close()
            if exitcode is not None and exitcode != 0 and exitcode != -15:
                death_info = self._read_collector_error()
                print(
                    f"\n[AsyncRunner] Collector exited with code {exitcode}:\n{death_info}\n",
                    file=sys.stderr,
                    flush=True,
                )

        for resource in self._shared_resources:
            if hasattr(resource, "cleanup"):
                resource.cleanup()
            elif hasattr(resource, "close"):
                resource.close()

        if self._error_recv is not None:
            try:
                self._error_recv.close()
            except Exception:
                pass
            self._error_recv = None
        if self._error_send is not None:
            try:
                self._error_send.close()
            except Exception:
                pass
            self._error_send = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
