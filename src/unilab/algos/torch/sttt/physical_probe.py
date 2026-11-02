"""Physical action-sensitivity probe packets and offline comparison.

Probe collection is opt-in and uses MuJoCo ground truth, not SLAM. Numerical
agreement inside a learned model is separate from agreement with physical
branches; neither establishes stability or successful closed-loop adaptation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .contract import SpatialTrajectoryWindow, validate_trajectory_window

PHYSICAL_PROBE_VERSION = 1


@dataclass(frozen=True)
class PhysicalProbeConfig:
    num_samples: int = 4
    warmup_steps: int = 30
    stride_steps: int = 30
    history_length: int = 15
    horizon: int = 15
    action_steps: tuple[int, ...] = (0,)
    joint_indices: tuple[int, ...] = (0, 1, 2)
    epsilon: float = 0.01
    replay_position_tolerance_m: float = 1e-4
    replay_yaw_tolerance_rad: float = 1e-4

    def __post_init__(self) -> None:
        for name in ("num_samples", "stride_steps", "history_length", "horizon"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.warmup_steps, bool) or not isinstance(self.warmup_steps, int) or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be a nonnegative integer")
        for name, maximum in (("action_steps", self.horizon), ("joint_indices", 29)):
            values = tuple(getattr(self, name))
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{name} must contain distinct indices")
            if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < maximum for v in values):
                raise ValueError(f"{name} indices must be in [0, {maximum})")
            object.__setattr__(self, name, values)
        for name in ("epsilon", "replay_position_tolerance_m", "replay_yaw_tolerance_rad"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True)
class PhysicalProbeCase:
    action_step: int
    joint_index: int
    epsilon: float
    plus_endpoint: torch.Tensor | None
    minus_endpoint: torch.Tensor | None
    plus_status: str = "ok"
    minus_status: str = "ok"


@dataclass(frozen=True)
class PhysicalProbeSample:
    """One origin, its committed nominal window, and controlled branch outcomes.

    Endpoints are [dx,dy,dyaw] in the same origin-heading frame as window labels.
    Statuses are explicit (ok, terminated, nonfinite, condition_changed,
    action_transformed, out_of_bounds, not_run). Failed branches never use zero
    as a replacement endpoint. No simulator snapshot is persisted in this packet.
    """

    rollout_id: str
    episode_id: int
    origin_step: int
    window: SpatialTrajectoryWindow
    baseline_endpoint: torch.Tensor | None
    baseline_status: str
    cases: tuple[PhysicalProbeCase, ...]
    condition: dict[str, Any] | None = None


def _endpoint(value: torch.Tensor | None, *, name: str) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = torch.as_tensor(value).detach().to(device="cpu").clone()
    if tensor.shape != (3,) or not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be a finite (3,) endpoint or None")
    return tensor


def save_physical_probes(path: str | Path, samples: Sequence[PhysicalProbeSample], metadata: Mapping[str, Any]) -> None:
    """Persist tensor/primitive records, loadable with torch weights_only=True."""
    import json
    from dataclasses import asdict

    from .checkpoint import data_contract_from_metadata

    contract = data_contract_from_metadata(metadata)
    records = []
    for sample in samples:
        validate_trajectory_window(sample.window, expected_obs_dim=64, expected_act_dim=29,
                                   expected_hist=int(contract["history_length"]),
                                   expected_horizon=int(contract["horizon"]))
        window = sample.window
        if sample.baseline_status == "ok" and sample.baseline_endpoint is None:
            raise ValueError("An ok baseline must carry a measured endpoint")
        cases = []
        for case in sample.cases:
            if ((case.plus_status == "ok" and case.plus_endpoint is None)
                    or (case.minus_status == "ok" and case.minus_endpoint is None)):
                raise ValueError("An ok branch must carry a measured endpoint")
            record = asdict(case)
            record["plus_endpoint"] = _endpoint(case.plus_endpoint, name="plus_endpoint")
            record["minus_endpoint"] = _endpoint(case.minus_endpoint, name="minus_endpoint")
            cases.append(record)
        records.append({
            "rollout_id": sample.rollout_id, "episode_id": sample.episode_id, "origin_step": sample.origin_step,
            "window": {"proprio_history": window.proprio_history.detach().cpu().clone(),
                       "action_sequence": window.action_sequence.detach().cpu().clone(),
                       "delta_trajectory": window.delta_trajectory.detach().cpu().clone(),
                       "commands": window.commands.detach().cpu().clone(),
                       "step_indices": window.step_indices},
            "baseline_endpoint": _endpoint(sample.baseline_endpoint, name="baseline_endpoint"),
            "baseline_status": sample.baseline_status, "cases": cases,
            "condition": sample.condition,
        })
    target = Path(path).expanduser()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite physical probes: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "sttt_physical_probes", "format_version": PHYSICAL_PROBE_VERSION,
                "data_contract": contract, "metadata": json.loads(json.dumps(dict(metadata), allow_nan=False)),
                "samples": records}, target)


def load_physical_probes(path: str | Path) -> tuple[tuple[PhysicalProbeSample, ...], dict[str, Any]]:
    from .collection import trajectory_window_from_dict
    from .checkpoint import data_contract_from_metadata

    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("kind") != "sttt_physical_probes" or payload.get("format_version") != PHYSICAL_PROBE_VERSION:
        raise ValueError("Unsupported physical probe artifact")
    contract = data_contract_from_metadata(payload["metadata"])
    if contract != payload["data_contract"]:
        raise ValueError("Probe metadata and stored data contract differ")
    samples = []
    for row in payload["samples"]:
        validate_trajectory_window(trajectory_window_from_dict(row["window"]),
                                   expected_obs_dim=int(contract["obs_dim"]),
                                   expected_act_dim=int(contract["act_dim"]),
                                   expected_hist=int(contract["history_length"]),
                                   expected_horizon=int(contract["horizon"]))
        cases = tuple(PhysicalProbeCase(**case) for case in row["cases"])
        samples.append(PhysicalProbeSample(row["rollout_id"], int(row["episode_id"]), int(row["origin_step"]),
                                          trajectory_window_from_dict(row["window"]),
                                          row["baseline_endpoint"], str(row["baseline_status"]), cases,
                                          row.get("condition")))
    return tuple(samples), payload["metadata"]


def compare_physical_probes(
    model: Any,
    samples: Sequence[PhysicalProbeSample],
    *,
    replay_position_tolerance_m: float = 1e-4,
    replay_yaw_tolerance_rad: float = 1e-4,
    gradient_zero_tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Compare recorded branches with a frozen learned endpoint Jacobian.

    This function does not execute physics. Replay mismatch and failed branches
    remain in the report and never become zero-valued reference gradients.
    """
    import copy

    for name, value in (("replay_position_tolerance_m", replay_position_tolerance_m),
                        ("replay_yaw_tolerance_rad", replay_yaw_tolerance_rad),
                        ("gradient_zero_tolerance", gradient_zero_tolerance)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    probe = copy.deepcopy(model).to(device="cpu", dtype=torch.float64).eval().requires_grad_(False)
    reports: list[dict[str, Any]] = []
    errors: list[torch.Tensor] = []
    usable_samples = usable_cases = total_cases = 0
    all_zero_model_gradient = True

    def wrapped(value: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(value), torch.cos(value))

    for sample in samples:
        window = sample.window
        validate_trajectory_window(window, expected_obs_dim=probe.obs_dim, expected_act_dim=probe.act_dim,
                                   expected_hist=probe.hist_len, expected_horizon=probe.horizon)
        target = window.delta_trajectory[-1].detach().to(device="cpu", dtype=torch.float64)
        baseline = _endpoint(sample.baseline_endpoint, name="baseline_endpoint")
        xy_error = yaw_error = None
        baseline_usable = False
        if sample.baseline_status == "ok" and baseline is not None:
            delta = baseline.double() - target
            xy_error = float(torch.linalg.vector_norm(delta[:2]))
            yaw_error = abs(float(wrapped(delta[2])))
            baseline_usable = (xy_error <= replay_position_tolerance_m
                               and yaw_error <= replay_yaw_tolerance_rad)
        row: dict[str, Any] = {
            "rollout_id": sample.rollout_id, "episode_id": sample.episode_id,
            "origin_step": sample.origin_step, "baseline_status": sample.baseline_status,
            "baseline_reproduced": baseline_usable, "baseline_xy_error_m": xy_error,
            "baseline_yaw_error_rad": yaw_error, "cases": [],
            "condition": sample.condition,
        }
        jacobian = None
        if baseline_usable and any(c.plus_status == "ok" and c.minus_status == "ok" for c in sample.cases):
            history = window.proprio_history.detach().to(device="cpu", dtype=torch.float64).unsqueeze(0)
            actions = window.action_sequence.detach().to(device="cpu", dtype=torch.float64).clone().requires_grad_(True)
            with torch.enable_grad():
                prediction = probe(history, actions.unsqueeze(0))[0, -1]
                jacobian = torch.stack([
                    torch.autograd.grad(prediction[i], actions, retain_graph=i < 2)[0]
                    for i in range(3)
                ])
            if not torch.isfinite(jacobian).all():
                raise ValueError("Model endpoint Jacobian is non-finite")
            residual = prediction.detach() - target
            row["prediction_endpoint_xy_m"] = float(torch.linalg.vector_norm(residual[:2]))
            row["prediction_endpoint_yaw_rad"] = abs(float(wrapped(residual[2])))
        sample_usable = False
        for case in sample.cases:
            total_cases += 1
            if not 0 <= case.action_step < probe.horizon or not 0 <= case.joint_index < probe.act_dim:
                raise ValueError("Physical probe action/joint index is outside the model contract")
            if not math.isfinite(case.epsilon) or case.epsilon <= 0:
                raise ValueError("Physical probe epsilon must be positive and finite")
            entry: dict[str, Any] = {
                "action_step": case.action_step, "joint_index": case.joint_index,
                "epsilon": case.epsilon, "plus_status": case.plus_status,
                "minus_status": case.minus_status, "usable": False,
            }
            if not baseline_usable:
                entry["reason"] = "baseline_not_reproduced"
            elif case.plus_status != "ok" or case.minus_status != "ok":
                entry["reason"] = "branch_not_usable"
            else:
                plus = _endpoint(case.plus_endpoint, name="plus_endpoint")
                minus = _endpoint(case.minus_endpoint, name="minus_endpoint")
                if plus is None or minus is None or jacobian is None:
                    raise ValueError("An ok physical branch must carry a real endpoint")
                difference = plus.double() - minus.double()
                physical = torch.cat((difference[:2], wrapped(difference[2]).reshape(1))) / (2 * case.epsilon)
                learned = jacobian[:, case.action_step, case.joint_index].detach()
                error = (learned - physical).abs()
                pnorm = float(physical.norm())
                mnorm = float(learned.norm())
                cosine = (float(torch.dot(learned, physical)) / (pnorm * mnorm)
                          if min(pnorm, mnorm) > gradient_zero_tolerance else None)
                agreement = [
                    None if abs(float(p)) <= gradient_zero_tolerance else
                    bool(abs(float(m)) > gradient_zero_tolerance and float(m * p) > 0)
                    for m, p in zip(learned, physical, strict=True)
                ]
                entry.update({"usable": True, "model_jacobian": learned.tolist(),
                              "physics_finite_difference": physical.tolist(), "abs_error": error.tolist(),
                              "sign_agreement": agreement,
                              "cosine_similarity": max(-1.0, min(1.0, cosine)) if cosine is not None else None})
                errors.append(error)
                usable_cases += 1
                sample_usable = True
                all_zero_model_gradient &= mnorm <= gradient_zero_tolerance
            row["cases"].append(entry)
        usable_samples += int(sample_usable)
        reports.append(row)
    mean_error = torch.stack(errors).mean(0).tolist() if errors else [None, None, None]
    return {
        "num_samples": len(samples), "usable_samples": usable_samples,
        "total_cases": total_cases, "usable_cases": usable_cases,
        "unusable_cases": total_cases - usable_cases,
        "jacobian_mae": dict(zip(("dx_m_per_action", "dy_m_per_action", "dyaw_rad_per_action"), mean_error, strict=True)),
        "all_zero_model_gradient": all_zero_model_gradient if usable_cases else None,
        "samples": reports,
        "scope": "Recorded finite-difference evidence only; no simulator was run by comparison. Not a stability or adaptation guarantee.",
    }


def run_physical_probe(cfg: Any, *, root_dir: Path) -> dict[str, Any]:
    """Only the explicit capture mode runs the existing simulation lifecycle."""
    from collections import Counter
    from .checkpoint import data_contract_from_metadata, load_world_model_checkpoint

    if str(cfg.mode) == "capture":
        from .physical_probe_runtime import collect_physical_probes

        samples, metadata = collect_physical_probes(cfg, root_dir=root_dir)
        save_physical_probes(cfg.output_path, samples, metadata)
        if not samples:
            raise RuntimeError(f"No complete probe windows; failure metadata saved to {cfg.output_path}")
        return {"output_path": str(cfg.output_path), "saved_samples": len(samples),
                "baseline_status_counts": dict(Counter(s.baseline_status for s in samples)),
                "probe_summary": metadata.get("probe_summary", {})}
    if str(cfg.mode) != "compare":
        raise ValueError("physical probe mode must be capture or compare")
    if not cfg.evaluation.checkpoint:
        raise ValueError("evaluation.checkpoint must name a trained world-model checkpoint")
    samples, metadata = load_physical_probes(cfg.input_path)
    model, _ = load_world_model_checkpoint(
        cfg.evaluation.checkpoint, device="cpu", expected_data_contract=data_contract_from_metadata(metadata),
    )
    stored = PhysicalProbeConfig(**metadata["probe_config"])
    report = compare_physical_probes(
        model, samples, replay_position_tolerance_m=stored.replay_position_tolerance_m,
        replay_yaw_tolerance_rad=stored.replay_yaw_tolerance_rad,
        gradient_zero_tolerance=float(cfg.evaluation.gradient_zero_tolerance),
    )
    report["label_source"] = metadata.get("label_source", "unspecified")
    report["probe_path"] = str(cfg.input_path)
    report["checkpoint"] = str(cfg.evaluation.checkpoint)
    return report
