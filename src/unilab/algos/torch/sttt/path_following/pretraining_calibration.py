"""Task-tolerance configuration and measured finite-response probe provenance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .latent_objectives import LatentCostConfig



def validate_calibration_input(cfg: Mapping[str, Any], *, root: Path) -> None:
    if set(cfg) != {"response_probe_result", "require_response_probe"} or type(cfg["require_response_probe"]) is not bool:
        raise ValueError("calibration requires response_probe_result and require_response_probe")
    path = cfg["response_probe_result"]
    if cfg["require_response_probe"] and not path:
        raise ValueError("set calibration.response_probe_result to a completed latent_response_direction_probe summary.json; select require_response_probe=false explicitly for an uncalibrated baseline")
    if path:
        path = Path(path).expanduser()
        path = path if path.is_absolute() else root / path
        if not path.is_file() or not (path.parent / "source.json").is_file():
            raise FileNotFoundError(f"response calibration requires summary.json and its source.json: {path}")


def response_probe_evidence(cfg, *, contract, residual_limits, horizon_steps, root: Path):
    validate_calibration_input(cfg, root=root)
    probe = None
    if cfg["response_probe_result"]:
        path = Path(cfg["response_probe_result"]).expanduser()
        path = path if path.is_absolute() else root / path
        summary, source = json.loads(path.read_text()), json.loads((path.parent / "source.json").read_text())
        if summary.get("complete") is not True or summary.get("valid") is not True:
            raise ValueError("physical response calibration did not complete with a valid contrast")
        if source["models"]["policy"]["sha256"] != contract["source"]["checkpoint_sha256"]:
            raise ValueError("response calibration belongs to another Planner–Tracker checkpoint")
        if max(residual_limits) > float(source["probe_config"]["residual_amplitude_rad"]) + 1e-12:
            raise ValueError("configured residual bound exceeds the measured response amplitude")
        horizon_s = horizon_steps * contract["data"]["ctrl_dt"]
        if not math.isclose(summary["horizon_s"], horizon_s, rel_tol=0., abs_tol=1e-9):
            raise ValueError("response probe and pretraining forecast must use the same physical horizon")
        probe = {"summary_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                 "source_sha256": hashlib.sha256((path.parent / "source.json").read_bytes()).hexdigest(),
                 "scope": source["scope"], "horizon_s": horizon_s,
                 "amplitude_rad": source["probe_config"]["residual_amplitude_rad"]}
    return probe


def prepare_calibration(cfg, costs, *, directories, contract, residual_limits, horizon_steps, root: Path,
                        previous: Mapping[str, Any] | None = None) -> dict[str, Any]:
    probe = response_probe_evidence(cfg, contract=contract, residual_limits=residual_limits,
                                    horizon_steps=horizon_steps, root=root)
    normalized = dict(costs)
    objective = LatentCostConfig(**normalized)
    report = {"scale_method": "declared_physical_task_tolerances", "costs": normalized,
              "prediction_selection": "window_xy_rmse_m",
              "failure_cost_rate": objective.failure_cost_rate,
              "response_probe": probe,
              "response_scope": "measured finite direction, not a full state-dependent safety envelope"}
    if previous is not None and report != dict(previous):
        raise ValueError("joint resume physical scales, costs or calibration evidence changed")
    return report
