"""Measured pulse response fitting and a bounded, model-independent controller."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .control_probe import PhysicalTrace, same_origin


def relative_motion(trace: PhysicalTrace) -> np.ndarray:
    matrices = np.asarray(trace.arrays["poses"], dtype=np.float64)
    heading = np.arctan2(trace.arrays["origin_warmup_poses"][0, 1, 0], trace.arrays["origin_warmup_poses"][0, 0, 0])
    delta = matrices[:, :2, 3] - matrices[:1, :2, 3]
    yaw = np.unwrap(np.arctan2(matrices[:, 1, 0], matrices[:, 0, 0]))
    return np.column_stack((np.cos(heading) * delta[:, 0] + np.sin(heading) * delta[:, 1],
                            -np.sin(heading) * delta[:, 0] + np.cos(heading) * delta[:, 1], yaw - yaw[0]))


def bounded_least_squares(matrix: np.ndarray, target: np.ndarray, *, initial=None, iterations: int = 96) -> np.ndarray:
    """Projected coordinate descent for min ||A x-b||², -1 <= x <= 1."""
    matrix, target = np.asarray(matrix, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if matrix.ndim != 2 or target.shape != (matrix.shape[0],) or not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("bounded solve requires a finite matrix and paired target")
    if type(iterations) is not int or iterations < 1:
        raise ValueError("bounded solve requires a positive iteration count")
    gram, rhs = matrix.T @ matrix, matrix.T @ target
    diagonal = np.diag(gram)
    value = np.zeros(matrix.shape[1]) if initial is None else np.clip(np.asarray(initial, dtype=np.float64), -1., 1.)
    if value.shape != (matrix.shape[1],):
        raise ValueError("bounded solve warm start has the wrong width")
    coordinates = np.flatnonzero(diagonal > 0)
    for _ in range(iterations):
        maximum_change = 0.
        for index in coordinates:
            updated = float(np.clip(value[index] + (rhs[index] - gram[index] @ value) / diagonal[index], -1., 1.))
            maximum_change = max(maximum_change, abs(updated - value[index]))
            value[index] = updated
        if maximum_change <= 1e-10:
            break
    return value


@dataclass(frozen=True)
class ResponseMap:
    response: np.ndarray  # [phase, native time including origin, xyz, joint], per normalized pulse
    phases: np.ndarray
    phase_cycles: np.ndarray
    noise: np.ndarray
    usable: np.ndarray
    limits: np.ndarray
    dt: float
    window_steps: int

    def save(self, path: Path) -> None:
        np.savez_compressed(path, response=self.response, phases=self.phases, phase_cycles=self.phase_cycles,
            noise=self.noise, usable=self.usable, limits=self.limits, dt=np.asarray(self.dt), window_steps=np.asarray(self.window_steps))

    @classmethod
    def load(cls, path: Path) -> ResponseMap:
        with np.load(path, allow_pickle=False) as archive:
            return cls(**{key: archive[key].copy() for key in ("response", "phases", "phase_cycles", "noise", "usable", "limits")},
                       dt=float(archive["dt"]), window_steps=int(archive["window_steps"]))


def fit_response_map(branches: dict[str, PhysicalTrace], phases: list[float], fractions: list[float], *,
                     minimum_effect_m: float, minimum_effect_rad: float,
                     replay_position_tolerance_m: float, replay_heading_tolerance_rad: float,
                     channel_scales=(.3, .3, 1.)) -> tuple[ResponseMap, dict]:
    if not phases or not fractions or any(not 0 < fraction <= 1 for fraction in fractions):
        raise ValueError("response fitting requires positive bounded pulse fractions")
    if any(not np.isfinite(value) or value <= 0 for value in (*channel_scales, minimum_effect_m, minimum_effect_rad,
            replay_position_tolerance_m, replay_heading_tolerance_rad)):
        raise ValueError("response detectability and scaling must be explicit positive quantities")
    outputs, noises, usable, cycles, reports = [], [], [], [], []
    first = branches["p0_zero_a"]
    shape = relative_motion(first).shape
    for phase_index, phase in enumerate(phases):
        zero = branches[f"p{phase_index}_zero_a"]
        repeat = branches[f"p{phase_index}_zero_b"]
        same_origin(zero, repeat)
        if not zero.metadata["complete"] or not repeat.metadata["complete"] or not zero.metadata["valid"] or not repeat.metadata["valid"]:
            raise ValueError("zero-repeat branch did not complete: no response map can be identified")
        baseline, repeated = relative_motion(zero), relative_motion(repeat)
        if baseline.shape != shape or repeated.shape != shape:
            raise ValueError("pulse phases changed the observation duration")
        repeat_error = np.max(np.abs(repeated - baseline), axis=0)
        repeat_ok = bool(np.linalg.norm(repeat_error[:2]) <= replay_position_tolerance_m and repeat_error[2] <= replay_heading_tolerance_rad)
        resolution = 64 * np.finfo(baseline.dtype).eps * np.maximum(1., np.abs(baseline).max(0))
        noise = np.maximum(repeat_error, np.maximum(resolution, [minimum_effect_m, minimum_effect_m, minimum_effect_rad]))
        matrix = np.zeros((*shape, 29))
        valid = np.zeros(29, dtype=bool)
        details = []
        for joint in range(29):
            pairs = []
            for index, fraction in enumerate(fractions):
                plus, minus = (branches[f"p{phase_index}_j{joint}_a{index}_{sign}"] for sign in ("plus", "minus"))
                same_origin(zero, plus); same_origin(zero, minus)
                for trace, sign in ((plus, 1), (minus, -1)):
                    expected = np.zeros_like(trace.arrays["residuals"])
                    expected[:trace.metadata["window_steps"], joint] = sign * fraction * trace.metadata["residual_limits_rad"][joint]
                    if not np.allclose(trace.arrays["residuals"], expected, rtol=0., atol=1e-7):
                        raise ValueError("pulse labels disagree with the physically applied residual sequence")
                complete = all(item.metadata["valid"] and item.metadata["complete"] for item in (plus, minus))
                if complete:
                    p, m = relative_motion(plus), relative_motion(minus)
                    if p.shape != shape or m.shape != shape:
                        raise ValueError("pulse branch changed physical time coverage")
                    pairs.append((fraction, (p - m) / 2, (p + m) / 2 - baseline))
            if pairs:
                matrix[..., joint] = sum(fraction * effect for fraction, effect, _ in pairs) / sum(fraction**2 for fraction, _, _ in pairs)
            valid[joint] = repeat_ok and len(pairs) == len(fractions)
            signal = np.abs(matrix[..., joint]) > noise
            detectable = bool(valid[joint] and signal[:, 1:].any())
            times = np.flatnonzero(signal[:, 1:].any(-1))
            details.append({"joint": joint, "valid": bool(valid[joint]), "detectable_lateral_or_heading": detectable,
                "delay_s": float(times[0] * first.metadata["ctrl_dt"]) if detectable else None,
                "peak_response_m_m_rad": np.max(np.abs(matrix[..., joint]), axis=0).tolist(),
                "odd_fit_error_m_m_rad": np.max([np.max(np.abs(effect - fraction * matrix[..., joint]), axis=0) for fraction, effect, _ in pairs], axis=0).tolist() if pairs else None,
                "asymmetry_m_m_rad": np.max([np.max(np.abs(asymmetry), axis=0) for _, _, asymmetry in pairs], axis=0).tolist() if pairs else None})
        scaled = matrix[1:] / np.asarray(channel_scales)[None, :, None]
        singular = np.linalg.svd(scaled.reshape(-1, 29), compute_uv=False)
        cycles.append(float(np.asarray(zero.arrays["origin_gait_phase"]).reshape(-1)[0] / (2 * np.pi) % 1))
        outputs.append(matrix); noises.append(noise); usable.append(valid)
        reports.append({"phase_fraction": phase, "repeatable": repeat_ok, "repeat_error_m_m_rad": repeat_error.tolist(),
            "detection_threshold_m_m_rad": noise.tolist(), "normalized_singular_values": singular.tolist(), "joints": details})
    response = ResponseMap(np.stack(outputs), np.asarray(phases), np.asarray(cycles), np.stack(noises), np.stack(usable),
        np.asarray(first.metadata["residual_limits_rad"]), first.metadata["ctrl_dt"], first.metadata["window_steps"])
    detectable = any(joint["detectable_lateral_or_heading"] for phase in reports for joint in phase["joints"])
    report = {"status": "invalid_replay" if not all(phase["repeatable"] for phase in reports) else ("measured_response" if detectable else "unresolved_effect"), "phases": reports,
        "channel_scales_m_m_rad": list(channel_scales), "response_units": ["m", "m", "rad"],
        "interpretation": "Local, phase-conditioned pulse authority; closed-loop recovery is evaluated separately."}
    return response, report


def phase_cycle(decision) -> float:
    query = decision.query
    if query.command.shape[-1] == 7:
        return float(torch.atan2(query.command[0, 3], query.command[0, 5]).detach().cpu()) / (2 * np.pi) % 1
    return float(query.observation_history[0, -1, 64].detach().cpu()) / (2 * np.pi) % 1


class PulseProgram:
    def __init__(self, residual, *, pulse_windows=1):
        self.residual = np.asarray(residual)
        self.pulse_windows = pulse_windows

    def __call__(self, decision):
        active = 0 <= decision.step < decision.window_steps * self.pulse_windows
        return decision.query.command.new_tensor(self.residual[None]) if active else torch.zeros_like(decision.previous_residual)


class MeasuredFeedback:
    """Receding FIR control; past pulses and measured lateral drift enter explicitly."""
    def __init__(self, response: ResponseMap, *, lateral_scale=.3, velocity_scale=1., forward_scale=.35,
                 smoothness_weight=.01, iterations=96):
        self.model = response
        self.scales = np.asarray([forward_scale, lateral_scale, velocity_scale])
        if np.any(self.scales <= 0) or not np.isfinite(self.scales).all() or smoothness_weight <= 0:
            raise ValueError("feedback scaling and smoothness must be positive")
        self.smoothness = smoothness_weight
        self.iterations = iterations
        self.past: list[tuple[int, int, np.ndarray]] = []
        self.last_pose = None
        self.previous_plan = None

    def __call__(self, decision):
        if decision.step < 0:
            return torch.zeros_like(decision.previous_residual)
        if decision.step % decision.window_steps:
            return decision.previous_residual
        model, step, k = self.model, decision.step, decision.window_steps
        if k != model.window_steps:
            raise ValueError("measured feedback and execution windows differ")
        cycle = phase_cycle(decision)
        phase = int(np.argmin(np.abs((model.phase_cycles - cycle + .5) % 1 - .5)))
        count = (model.response.shape[1] - 1) // k
        times = np.arange(1, count + 1) * k
        pose = decision.path.pose[0].detach().cpu().numpy().astype(np.float64)
        velocity = np.asarray([self.scales[0], 0.]) if self.last_pose is None else (pose[:2] - self.last_pose[:2]) / (k * model.dt)
        self.last_pose = pose.copy()
        free = np.stack(((velocity[0] - self.scales[0]) * times * model.dt, pose[1] + velocity[1] * times * model.dt), -1)
        matrix = np.zeros((count, 2, count, 29))
        for future in range(count):
            for row, time in enumerate(times):
                lag = time - future * k
                if lag > 0:
                    matrix[row, :, future] = model.response[phase, min(lag, model.response.shape[1] - 1), :2]
        for old_step, old_phase, value in self.past:
            age = step - old_step
            last = model.response.shape[1] - 1
            if age >= last:
                continue
            response = model.response[old_phase]
            previous = response[max(0, age - k), :2] @ value
            current = response[age, :2] @ value
            current_velocity = (current - previous) / (k * model.dt)
            for row, time in enumerate(times):
                free[row] += response[min(age + time, last), :2] @ value - current - current_velocity * time * model.dt
        # Penalize side error, side velocity and unintended forward displacement.
        lateral = matrix[:, 1].reshape(count, -1) / self.scales[1]
        forward = matrix[:, 0].reshape(count, -1) / self.scales[0]
        lateral_rate = np.diff(np.concatenate((np.zeros_like(lateral[:1]), lateral)), axis=0) / (k * model.dt) * self.scales[1] / self.scales[2]
        free_rate = np.diff(np.r_[pose[1], free[:, 1]]) / (k * model.dt)
        change = np.eye(count) - np.eye(count, k=-1)
        smooth = np.kron(change, np.eye(29)) * np.sqrt(self.smoothness)
        targets = np.r_[-free[:, 1] / self.scales[1], -free_rate / self.scales[2], -free[:, 0] / self.scales[0],
                        decision.previous_residual[0].detach().cpu().numpy() / model.limits * np.sqrt(self.smoothness), np.zeros((count - 1) * 29)]
        design = np.vstack((lateral, lateral_rate, forward, smooth))
        allowed = np.tile(model.usable[phase], count)
        design[:, ~allowed] = 0.
        initial = None if self.previous_plan is None else np.r_[self.previous_plan[29:], np.zeros(29)]
        plan = bounded_least_squares(design, targets, initial=initial, iterations=self.iterations)
        plan[~allowed] = 0.
        self.previous_plan = plan
        residual = plan[:29]
        self.past = [(when, old_phase, value) for when, old_phase, value in self.past if step - when < model.response.shape[1]]
        self.past.append((step, phase, residual.copy()))
        return decision.query.command.new_tensor((residual * model.limits)[None])


def tracking_metrics(trace: PhysicalTrace) -> dict:
    if not trace.metadata["valid"]:
        return {"status": "invalid", "error": trace.metadata["error"]}
    errors = trace.arrays["errors"][:, 0]
    duration = len(trace.arrays["actions"]) * trace.metadata["ctrl_dt"]
    progress = float(trace.arrays["path_progress"][-1] - trace.arrays["path_progress"][0])
    tilt = trace.arrays["risk"]
    return {"status": "complete" if trace.metadata["complete"] else trace.metadata["end_reason"],
        "duration_s": duration, "initial_lateral_m": float(errors[0]), "final_lateral_m": float(errors[-1]),
        "lateral_rmse_m": float(np.sqrt(np.mean(errors**2))), "lateral_integral_m_s": float(np.abs(errors[1:]).sum() * trace.metadata["ctrl_dt"]),
        "forward_progress_m": progress, "forward_speed_m_s": progress / duration,
        "risk_event": bool(np.any(tilt)), "terminal": bool(np.any(trace.arrays["terminal"]))}
