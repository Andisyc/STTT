"""Execution windows advance on committed control steps, never on pose arrival."""
from __future__ import annotations

from numbers import Integral

import torch


class ConstantResidualWindow:
    def __init__(self, window_steps: int) -> None:
        if isinstance(window_steps, bool) or not isinstance(window_steps, Integral) or window_steps < 1:
            raise ValueError("window_steps must be a positive integer")
        self.window_steps = int(window_steps)
        self.window_id = 0
        self.step_index = 0
        self._candidate: torch.Tensor | None = None
        self._locked: torch.Tensor | None = None

    @property
    def at_boundary(self) -> bool:
        return self.step_index == 0 and self._locked is None

    def submit_candidate(self, residual_rad: torch.Tensor) -> None:
        if (
            residual_rad.shape != (1, 29) or not residual_rad.is_floating_point()
            or not bool(torch.isfinite(residual_rad).all())
        ):
            raise ValueError("candidate must be finite floating-point [1, 29] source-order radians")
        self._candidate = residual_rad.detach().clone()

    def lock(self) -> torch.Tensor:
        """Borrow the read-only residual used for every query in this window."""
        if self._locked is None:
            if self._candidate is None:
                raise RuntimeError("a residual candidate is required before the first query")
            self._locked = self._candidate
        return self._locked

    def commit_step(self) -> bool:
        if self._locked is None:
            raise RuntimeError("cannot commit an execution without a locked residual")
        self.step_index += 1
        if self.step_index < self.window_steps:
            return False
        self.window_id += 1
        self.step_index = 0
        self._locked = None
        return True
