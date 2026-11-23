"""Physical-unit cost scaling and existing finite-response probe provenance."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .joint_config import positive_scale
from .latent_objectives import LatentCostConfig

_SCALE_WEIGHTS = {"lateral_m": "lateral_weight", "lateral_velocity_m_s": "lateral_velocity_weight",
                  "forward_velocity_m_s": "forward_weight", "residual_rate_rad_s": "smoothness_weight"}


def validate_calibration_input(cfg: Mapping[str, Any], *, root: Path) -> None:
    if set(cfg) != {"response_probe_result", "require_response_probe", "cost_scales"} or type(cfg["require_response_probe"]) is not bool:
        raise ValueError("calibration requires response_probe_result, require_response_probe and cost_scales")
    path = cfg["response_probe_result"]
    if cfg["require_response_probe"] and not path:
        raise ValueError("set calibration.response_probe_result to a completed latent_response_direction_probe summary.json; select require_response_probe=false explicitly for an uncalibrated baseline")
    if path:
        path = Path(path).expanduser()
        path = path if path.is_absolute() else root / path
        if not path.is_file() or not (path.parent / "source.json").is_file():
            raise FileNotFoundError(f"response calibration requires summary.json and its source.json: {path}")
    if cfg["cost_scales"] is not None:
        if not isinstance(cfg["cost_scales"], Mapping) or set(cfg["cost_scales"]) != set(_SCALE_WEIGHTS):
            raise ValueError("cost_scales requires lateral_m, lateral_velocity_m_s, forward_velocity_m_s and residual_rate_rad_s")
        for key, value in cfg["cost_scales"].items():
            positive_scale(value, key)


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
    scales = cfg["cost_scales"]
    method = "explicit_physical_scales"
    if scales is None and previous is not None:
        scales, method = previous["cost_scales"], previous["scale_method"]
    if scales is None:
        totals = {key: {"sum_squares": 0., "count": 0} for key in _SCALE_WEIGHTS}
        for directory in directories:
            path = Path(directory) / "capture.json"
            metadata = json.loads(path.read_text()) if path.exists() else {}
            statistics = metadata.get("cost_statistics")
            if statistics is None:
                raise ValueError("supplied captures lack physical cost statistics; provide explicit calibration.cost_scales")
            for key, values in totals.items():
                values["sum_squares"] += statistics[key]["sum_squares"]
                values["count"] += statistics[key]["count"]
        scales = {}
        for key, values in totals.items():
            if not values["count"] or values["sum_squares"] <= 0:
                raise ValueError(f"bootstrap data cannot identify a nonzero {key} scale; provide explicit cost_scales")
            scales[key] = math.sqrt(values["sum_squares"] / values["count"])
        method = "initial_real_capture_rms_frozen_before_actor_updates"
    scales = {key: positive_scale(scales[key], key) for key in _SCALE_WEIGHTS}
    normalized = dict(costs)
    LatentCostConfig(**normalized)
    for key, weight in _SCALE_WEIGHTS.items():
        normalized[weight] /= scales[key] ** 2
    LatentCostConfig(**normalized)
    report = {"cost_scales": scales, "scale_method": method, "costs": normalized,
              "response_probe": probe, "response_scope": "measured finite direction, not a full state-dependent safety envelope"}
    if previous is not None and report != dict(previous):
        raise ValueError("joint resume physical scales, costs or calibration evidence changed")
    return report
