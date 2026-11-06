"""Streaming physical-time metrics, independent of control or optimizer state."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PathFollowingThresholds:
    lateral_tolerance_m: float
    heading_tolerance_rad: float
    dwell_s: float
    crossing_deadband_m: float
    minimum_forward_progress_m: float

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be an explicit positive finite number")
        if self.crossing_deadband_m > self.lateral_tolerance_m:
            raise ValueError("crossing deadband must not exceed the lateral tracking tolerance")
        if self.heading_tolerance_rad > math.pi:
            raise ValueError("heading tolerance must be at most pi radians")


class PathFollowingMetrics:
    """Integrate squared errors by trapezoids only between adjacent valid samples.

    Recovery entry time is confirmed only after dwell_s of uninterrupted valid
    in-band evidence. Missing samples break dwell and integration; missing time
    cannot count as successful maintenance or recovery. Crossings require a
    change between non-deadband signs, not repeated noisy zeros.
    """

    def __init__(self, thresholds: PathFollowingThresholds) -> None:
        self.thresholds = thresholds
        self._first_ns: int | None = None
        self._last_ns: int | None = None
        self._previous: np.ndarray | None = None
        self._initial_progress: float | None = None
        self._last_progress: float | None = None
        self._last_error: np.ndarray | None = None
        self._squared_integral = np.zeros(2, dtype=np.float64)
        self._observed_ns = 0
        self._max_abs = np.zeros(2, dtype=np.float64)
        self._inside_since: int | None = None
        self._first_excursion: int | None = None
        self._recovery_entry: int | None = None
        self._recovery_confirmed: int | None = None
        self._post_recovery_peak = 0.
        self._inside_peak = 0.
        self._last_sign = 0
        self._crossings = 0
        self.samples = 0
        self.missing_samples = 0

    def observe(self, timestamp_ns: int, errors: np.ndarray | None, forward_position_m: float | None) -> None:
        if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, (int, np.integer)) or timestamp_ns < 0:
            raise ValueError("metrics require nonnegative physical nanoseconds")
        timestamp_ns = int(timestamp_ns)
        if self._last_ns is not None and timestamp_ns <= self._last_ns:
            raise ValueError("metric timestamps must strictly increase; duplicate observations must be deduplicated")
        error = None if errors is None else np.asarray(errors, dtype=np.float64)
        if error is not None and (error.shape != (2,) or not np.isfinite(error).all() or forward_position_m is None or not math.isfinite(forward_position_m)):
            raise ValueError("valid metric samples require finite errors [ey, epsi] and forward position")
        if self._first_ns is None:
            self._first_ns = timestamp_ns
        if error is None:
            self.missing_samples += 1
            self._previous = None
            self._inside_since = None
            self._last_sign = 0
            self._last_error = None
            self._last_progress = None
            self._last_ns = timestamp_ns
            return
        if self._previous is not None and self._last_ns is not None:
            dt_ns = timestamp_ns - self._last_ns
            self._squared_integral += 0.5 * self._previous**2 * dt_ns
            self._squared_integral += 0.5 * error**2 * dt_ns
            self._observed_ns += dt_ns
        assert forward_position_m is not None
        self.samples += 1
        self._max_abs = np.maximum(self._max_abs, np.abs(error))
        if self._initial_progress is None:
            self._initial_progress = float(forward_position_m)
        self._last_progress = float(forward_position_m)
        inside = abs(error[0]) <= self.thresholds.lateral_tolerance_m and abs(error[1]) <= self.thresholds.heading_tolerance_rad
        if not inside:
            if self._first_excursion is None:
                self._first_excursion = timestamp_ns
            self._inside_since = None
            self._inside_peak = 0.
        else:
            if self._inside_since is None:
                self._inside_since = timestamp_ns
                self._inside_peak = 0.
            self._inside_peak = max(self._inside_peak, abs(float(error[0])))
            if self._first_excursion is not None and self._recovery_entry is None and timestamp_ns - self._inside_since >= round(self.thresholds.dwell_s * 1e9):
                self._recovery_entry = self._inside_since
                self._recovery_confirmed = timestamp_ns
                self._post_recovery_peak = self._inside_peak
        if self._recovery_entry is not None:
            self._post_recovery_peak = max(self._post_recovery_peak, abs(float(error[0])))
        sign = 1 if error[0] > self.thresholds.crossing_deadband_m else (-1 if error[0] < -self.thresholds.crossing_deadband_m else 0)
        if sign:
            if self._last_sign and sign != self._last_sign:
                self._crossings += 1
            self._last_sign = sign
        self._last_error = error.copy()
        self._previous = error.copy()
        self._last_ns = timestamp_ns

    def summary(self) -> dict[str, Any]:
        duration_ns = 0 if self._first_ns is None or self._last_ns is None else self._last_ns - self._first_ns
        settled = self._inside_since is not None and self._last_ns is not None and self._last_ns - self._inside_since >= round(self.thresholds.dwell_s * 1e9)
        if self.missing_samples or not settled:
            state = "insufficient_coverage" if self.missing_samples else ("recovered_then_lost" if self._recovery_entry is not None else ("not_recovered" if self._first_excursion is not None else "insufficient_dwell"))
        else:
            state = "recovered" if self._first_excursion is not None else "maintained"
        progress = None if self._last_progress is None or self._initial_progress is None else self._last_progress - self._initial_progress
        def elapsed(timestamp: int | None) -> float | None:
            return None if timestamp is None or self._first_ns is None else (timestamp - self._first_ns) / 1e9
        return {
            "samples": self.samples, "missing_samples": self.missing_samples,
            "duration_s": duration_ns / 1e9, "observed_duration_s": self._observed_ns / 1e9,
            "time_coverage": self._observed_ns / duration_ns if duration_ns else 0.,
            "rmse_m_rad": np.sqrt(self._squared_integral / self._observed_ns).tolist() if self._observed_ns else None,
            "max_abs_m_rad": self._max_abs.tolist() if self.samples else None,
            "end_error_m_rad": self._last_error.tolist() if self._last_error is not None else None,
            "tracking_state": state, "first_excursion_time_s": elapsed(self._first_excursion),
            "recovery_entry_time_s": elapsed(self._recovery_entry), "recovery_confirmed_time_s": elapsed(self._recovery_confirmed),
            "recovery_latency_s": None if self._recovery_entry is None or self._first_excursion is None else (self._recovery_entry - self._first_excursion) / 1e9,
            "forward_progress_m": progress,
            "progress_pass": progress is not None and progress >= self.thresholds.minimum_forward_progress_m,
            "lateral_crossings": self._crossings,
            "post_recovery_peak_lateral_m": self._post_recovery_peak if self._recovery_entry is not None else None,
        }
