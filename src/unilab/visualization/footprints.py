"""Bounded, episode-local touchdown records from real foot sensor samples."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class FootprintRecord:
    """Foot-site world position at a sampled air-to-contact transition.

    The position is an owned immutable value, not a collision contact point.
    Control steps and simulation time are relative to this row's episode.
    """

    env_index: int
    episode_id: int
    foot: Literal["left", "right"]
    control_step: int
    sim_time_s: float
    position_world: tuple[float, float, float]


class FootprintRecorder:
    """Record each foot's rising contact edge, with independent row lifecycles."""

    def __init__(self, *, num_envs: int, ctrl_dt: float, max_records: int = 2048) -> None:
        for name, value in (("num_envs", num_envs), ("max_records", max_records)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"footprints.{name} must be a positive integer")
        if not np.isfinite(ctrl_dt) or ctrl_dt <= 0.0:
            raise ValueError("footprints.ctrl_dt must be finite and positive")
        self.num_envs = int(num_envs)
        self.ctrl_dt = float(ctrl_dt)
        self.max_records = int(max_records)
        self._records: list[deque[FootprintRecord]] = [
            deque(maxlen=self.max_records) for _ in range(self.num_envs)
        ]
        self._contacts = np.zeros((self.num_envs, 2), dtype=bool)
        self._control_steps = np.zeros(self.num_envs, dtype=np.int64)
        self._episode_ids = np.full(self.num_envs, -1, dtype=np.int64)
        self._terminated = np.zeros(self.num_envs, dtype=bool)
        self._initialized = False

    def _contact_rows(self, contacts: np.ndarray) -> np.ndarray:
        values = np.asarray(contacts)
        if values.shape != (self.num_envs, 2) or values.dtype != np.bool_:
            raise ValueError(f"footprint contacts must be boolean with shape ({self.num_envs}, 2)")
        return values

    def _reset_rows(self, contacts: np.ndarray, rows: np.ndarray) -> None:
        for index in np.flatnonzero(rows):
            self._records[index].clear()
        self._contacts[rows] = contacts[rows]
        self._control_steps[rows] = 0
        self._episode_ids[rows] += 1
        self._terminated[rows] = False

    def reset(self, contacts: np.ndarray) -> None:
        """Start new episodes, using initial support only as a contact baseline."""
        contacts = self._contact_rows(contacts)
        self._reset_rows(contacts, np.ones(self.num_envs, dtype=bool))
        self._initialized = True

    def observe(
        self, *, foot_positions: np.ndarray, contacts: np.ndarray,
        done: np.ndarray, autoreset: bool,
    ) -> None:
        """Consume one completed control step, excluding terminal/reset rows.

        With autoreset, done rows already contain reset physics: clear their
        history and use that contact state only as the new episode's baseline.
        Without autoreset, recording stays stopped until an explicit reset.
        """
        if not self._initialized:
            raise RuntimeError("footprint recorder must be reset before observing")
        contacts = self._contact_rows(contacts)
        positions = np.asarray(foot_positions)
        done = np.asarray(done)
        if positions.shape != (self.num_envs, 2, 3):
            raise ValueError(f"footprint positions must have shape ({self.num_envs}, 2, 3)")
        if done.shape != (self.num_envs,) or done.dtype != np.bool_:
            raise ValueError(f"footprint done must be boolean with shape ({self.num_envs},)")
        active = ~(done | self._terminated)
        if not np.isfinite(positions[active]).all():
            raise ValueError("footprint positions must be finite on active rows")
        self._control_steps[active] += 1
        touchdown = contacts & ~self._contacts & active[:, None]
        for env_index, foot_index in np.argwhere(touchdown):
            step = int(self._control_steps[env_index])
            self._records[env_index].append(FootprintRecord(
                env_index=int(env_index),
                episode_id=int(self._episode_ids[env_index]),
                foot="left" if foot_index == 0 else "right",
                control_step=step,
                sim_time_s=step * self.ctrl_dt,
                position_world=tuple(float(value) for value in positions[env_index, foot_index]),
            ))
        self._contacts[active] = contacts[active]
        if autoreset:
            self._reset_rows(contacts, done)
        else:
            self._terminated |= done

    def snapshot(self, env_index: int | None = None) -> tuple[FootprintRecord, ...]:
        """Return immutable records, grouped by environment then sampling order."""
        if env_index is None:
            return tuple(record for history in self._records for record in history)
        if (isinstance(env_index, bool) or not isinstance(env_index, Integral)
                or not 0 <= env_index < self.num_envs):
            raise ValueError(f"footprint env_index must be in [0, {self.num_envs})")
        return tuple(self._records[env_index])
