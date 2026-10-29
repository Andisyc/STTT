"""Immutable sampling contract and cached candidate plan for FADA replay."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch

from unilab.algos.torch.distill.fada.model import (
    FADA_COMMAND_SCENARIOS,
    FADA_SCENARIO_IDS,
)

FADA_WALK_SPEED_BINS = ("slow", "medium", "high")


def _allocate_ratio_counts(
    total: int,
    ratios: Mapping[str, float],
    *,
    ordered_names: Sequence[str],
    label: str,
) -> tuple[tuple[str, int], ...]:
    if int(total) <= 0:
        raise ValueError(f"{label} total must be positive, got {total}")
    unknown = set(ratios) - set(ordered_names)
    if unknown:
        raise ValueError(f"{label} ratios contain unknown labels: {sorted(unknown)}")
    values = [float(ratios.get(name, 0.0)) for name in ordered_names]
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError(f"{label} ratios must be finite and non-negative")
    if abs(sum(values) - 1.0) > 1.0e-6:
        raise ValueError(f"{label} ratios must sum to 1, got {sum(values)}")
    positive = sum(value > 0.0 for value in values)
    if int(total) < positive:
        raise ValueError(f"{label} total={total} cannot cover {positive} positive strata")
    raw = [int(total) * value for value in values]
    counts = [int(value) for value in raw]
    for index, value in enumerate(values):
        if value > 0.0 and counts[index] == 0:
            counts[index] = 1
    while sum(counts) > int(total):
        candidates = [index for index, count in enumerate(counts) if count > 1]
        if not candidates:
            raise ValueError(f"{label} allocation cannot preserve positive strata")
        counts[min(candidates, key=lambda item: (raw[item] - counts[item], -item))] -= 1
    while sum(counts) < int(total):
        counts[max(range(len(counts)), key=lambda item: (raw[item] - counts[item], -item))] += 1
    return tuple(
        (name, count) for name, count in zip(ordered_names, counts, strict=True) if count > 0
    )


def _validate_ratios(
    ratios: Mapping[str, float], *, ordered_names: Sequence[str], label: str
) -> None:
    missing = set(ordered_names) - set(ratios)
    unknown = set(ratios) - set(ordered_names)
    if missing or unknown:
        raise ValueError(
            f"{label} ratios must match {list(ordered_names)}: "
            f"unknown={sorted(unknown)} missing={sorted(missing)}"
        )
    _allocate_ratio_counts(1000, ratios, ordered_names=ordered_names, label=label)


@dataclass(frozen=True)
class FADAReplaySamplingSpec:
    """Validated batch-distribution identity for active FADA replay consumers."""

    scenario_ratios: Mapping[str, float]
    walk_cold_start_ratio: float
    static_cold_start_ratio: float
    walk_steady_speed_thresholds: tuple[float, float] | None = None
    walk_steady_speed_ratios: Mapping[str, float] | None = None
    min_high_speed_replay_passes: int = 0

    def __post_init__(self) -> None:
        scenario_ratios = MappingProxyType(
            {str(name): float(value) for name, value in self.scenario_ratios.items()}
        )
        object.__setattr__(self, "scenario_ratios", scenario_ratios)
        _validate_ratios(
            scenario_ratios, ordered_names=FADA_COMMAND_SCENARIOS, label="Planner scenario"
        )
        for name, value in (
            ("walk_cold_start_ratio", self.walk_cold_start_ratio),
            ("static_cold_start_ratio", self.static_cold_start_ratio),
        ):
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        thresholds = self.walk_steady_speed_thresholds
        ratios = self.walk_steady_speed_ratios
        if (thresholds is None) != (ratios is None):
            raise ValueError("walk speed thresholds and ratios must be configured together")
        if thresholds is None:
            if int(self.min_high_speed_replay_passes) != 0:
                raise ValueError("speed-unstratified replay requires zero high-speed passes")
            return
        assert ratios is not None
        normalized_thresholds = tuple(float(value) for value in thresholds)
        if (
            len(normalized_thresholds) != 2
            or not all(math.isfinite(value) for value in normalized_thresholds)
            or normalized_thresholds[0] < 0.0
            or normalized_thresholds[0] >= normalized_thresholds[1]
        ):
            raise ValueError("walk speed thresholds must be finite 0 <= slow < high")
        normalized_ratios = MappingProxyType(
            {str(name): float(value) for name, value in ratios.items()}
        )
        _validate_ratios(
            normalized_ratios,
            ordered_names=FADA_WALK_SPEED_BINS,
            label="walk steady speed",
        )
        if int(self.min_high_speed_replay_passes) <= 0:
            raise ValueError("speed-stratified replay requires positive high-speed passes")
        object.__setattr__(self, "walk_steady_speed_thresholds", normalized_thresholds)
        object.__setattr__(self, "walk_steady_speed_ratios", normalized_ratios)


@dataclass(frozen=True)
class FADAReplayCoverage:
    planner_high_rows: int
    planner_high_batch_quota: int
    required_planner_updates: int
    idm_main_high_rows: int
    idm_main_high_batch_quota: int
    idm_intermediate_high_rows: int
    idm_intermediate_high_batch_quota: int
    required_idm_updates: int


@dataclass(frozen=True)
class _Quota:
    label: str
    count: int
    candidates: torch.Tensor


@dataclass(frozen=True)
class FADAReplaySamplingPlan:
    """Candidate indices built once per replay generation and batch identity."""

    planner: tuple[_Quota, ...]
    idm: tuple[_Quota, ...]
    coverage: FADAReplayCoverage

    def _sample(
        self, quotas: Sequence[_Quota], generator: torch.Generator | None
    ) -> torch.Tensor:
        selected = [
            quota.candidates.index_select(
                0,
                torch.randint(quota.candidates.numel(), (quota.count,), generator=generator),
            )
            for quota in quotas
            if quota.count > 0
        ]
        indices = torch.cat(selected)
        return indices.index_select(0, torch.randperm(indices.numel(), generator=generator))

    def sample_planner(self, generator: torch.Generator | None = None) -> torch.Tensor:
        return self._sample(self.planner, generator)

    def sample_idm(self, generator: torch.Generator | None = None) -> torch.Tensor:
        if not self.idm:
            raise ValueError("IDM stratified sampling requires paper source role retention")
        return self._sample(self.idm, generator)


def _planner_counts(
    total: int, spec: FADAReplaySamplingSpec
) -> tuple[tuple[str, str | None, str | None, int], ...]:
    strata: list[tuple[str, str | None, str | None, int]] = []
    for scenario, scenario_count in _allocate_ratio_counts(
        total, spec.scenario_ratios, ordered_names=FADA_COMMAND_SCENARIOS, label="Planner scenario"
    ):
        if scenario not in {"walk", "static_stand"}:
            strata.append((scenario, None, None, scenario_count))
            continue
        cold_ratio = (
            spec.walk_cold_start_ratio
            if scenario == "walk"
            else spec.static_cold_start_ratio
        )
        for profile, profile_count in _allocate_ratio_counts(
            scenario_count,
            {"cold_start": cold_ratio, "steady_state": 1.0 - cold_ratio},
            ordered_names=("cold_start", "steady_state"),
            label=f"{scenario} Planner profile",
        ):
            if scenario != "walk" or profile != "steady_state" or spec.walk_steady_speed_ratios is None:
                strata.append((scenario, profile, None, profile_count))
                continue
            for speed_bin, speed_count in _allocate_ratio_counts(
                profile_count,
                spec.walk_steady_speed_ratios,
                ordered_names=FADA_WALK_SPEED_BINS,
                label="walk steady speed",
            ):
                strata.append((scenario, profile, speed_bin, speed_count))
    return tuple(strata)


def _role_counts(total: int, ratio: int) -> tuple[tuple[str, int], ...]:
    if int(ratio) <= 0:
        raise ValueError("IDM stratified sampling requires a positive suboptimal ratio")
    return _allocate_ratio_counts(
        total,
        {"main": 1.0 / (ratio + 1), "intermediate": ratio / (ratio + 1)},
        ordered_names=("main", "intermediate"),
        label="IDM replay role",
    )


def _speed_masks(command: torch.Tensor, spec: FADAReplaySamplingSpec) -> dict[str, torch.Tensor]:
    if command.ndim != 2 or int(command.shape[1]) < 2:
        raise ValueError("walk speed stratification requires rank-2 command with at least two columns")
    if not bool(torch.isfinite(command).all()):
        raise ValueError("walk speed stratification requires finite commands")
    if spec.walk_steady_speed_thresholds is None:
        all_rows = torch.ones(command.shape[0], dtype=torch.bool, device=command.device)
        return {name: all_rows for name in FADA_WALK_SPEED_BINS}
    speed = torch.linalg.vector_norm(command[:, :2], dim=1)
    slow, high = spec.walk_steady_speed_thresholds
    return {"slow": speed < slow, "medium": (speed >= slow) & (speed < high), "high": speed >= high}


def _quota(mask: torch.Tensor, count: int, label: str) -> _Quota:
    candidates = torch.nonzero(mask, as_tuple=False).flatten()
    if int(count) > 0 and candidates.numel() == 0:
        raise ValueError(f"FADA replay stratum {label!r} is empty for requested count={count}")
    return _Quota(label, int(count), candidates)


def _planner_quotas(
    batch_size: int,
    *,
    planner_eligible: torch.Tensor,
    command_scenario: torch.Tensor,
    cold_start: torch.Tensor,
    command: torch.Tensor,
    spec: FADAReplaySamplingSpec,
    prefix: str = "",
) -> tuple[_Quota, ...]:
    speed_masks = _speed_masks(command, spec)
    quotas: list[_Quota] = []
    for scenario, profile, speed_bin, count in _planner_counts(batch_size, spec):
        mask = planner_eligible & (command_scenario == FADA_SCENARIO_IDS[scenario])
        label = scenario
        if profile is not None:
            mask &= cold_start if profile == "cold_start" else ~cold_start
            label += f"/{profile}"
        if speed_bin is not None:
            mask &= speed_masks[speed_bin]
            label += f"/{speed_bin}"
        quotas.append(_quota(mask, count, prefix + label))
    return tuple(quotas)


def _idm_quotas(
    batch_size: int,
    *,
    planner_eligible: torch.Tensor,
    command_scenario: torch.Tensor,
    cold_start: torch.Tensor,
    command: torch.Tensor,
    spec: FADAReplaySamplingSpec,
    ratio: int,
) -> tuple[_Quota, ...]:
    quotas: list[_Quota] = []
    for role, count in _role_counts(batch_size, ratio):
        if role == "main":
            quotas.extend(
                _planner_quotas(
                    count,
                    planner_eligible=planner_eligible,
                    command_scenario=command_scenario,
                    cold_start=cold_start,
                    command=command,
                    spec=spec,
                    prefix="main/",
                )
            )
            continue
        base = ~planner_eligible & (command_scenario == FADA_SCENARIO_IDS["walk"]) & ~cold_start
        if spec.walk_steady_speed_ratios is None:
            quotas.append(_quota(base, count, "intermediate/walk/steady_state"))
            continue
        speed_masks = _speed_masks(command, spec)
        for speed_bin, speed_count in _allocate_ratio_counts(
            count,
            spec.walk_steady_speed_ratios,
            ordered_names=FADA_WALK_SPEED_BINS,
            label="intermediate walk steady speed",
        ):
            quotas.append(_quota(base & speed_masks[speed_bin], speed_count, f"intermediate/walk/steady_state/{speed_bin}"))
    return tuple(quotas)


def build_replay_sampling_plan(
    batch_size: int,
    *,
    planner_eligible: torch.Tensor,
    command_scenario: torch.Tensor,
    cold_start: torch.Tensor,
    command: torch.Tensor,
    spec: FADAReplaySamplingSpec,
    suboptimal_retention_ratio: int | None,
) -> FADAReplaySamplingPlan:
    planner = _planner_quotas(
        batch_size,
        planner_eligible=planner_eligible,
        command_scenario=command_scenario,
        cold_start=cold_start,
        command=command,
        spec=spec,
    )
    idm = () if suboptimal_retention_ratio is None else _idm_quotas(
        batch_size,
        planner_eligible=planner_eligible,
        command_scenario=command_scenario,
        cold_start=cold_start,
        command=command,
        spec=spec,
        ratio=suboptimal_retention_ratio,
    )
    if spec.min_high_speed_replay_passes <= 0:
        return FADAReplaySamplingPlan(planner, idm, FADAReplayCoverage(0, 0, 0, 0, 0, 0, 0, 0))

    def find(quotas: Sequence[_Quota], label: str) -> _Quota:
        return next(quota for quota in quotas if quota.label == label)

    def required(quota: _Quota) -> int:
        return int(
            math.ceil(
                int(quota.candidates.numel())
                * int(spec.min_high_speed_replay_passes)
                / int(quota.count)
            )
        )

    planner_high = find(planner, "walk/steady_state/high")
    if not idm:
        coverage = FADAReplayCoverage(
            planner_high.candidates.numel(), planner_high.count, required(planner_high), 0, 0, 0, 0, 0
        )
    else:
        main_high = find(idm, "main/walk/steady_state/high")
        intermediate_high = find(idm, "intermediate/walk/steady_state/high")
        coverage = FADAReplayCoverage(
            planner_high.candidates.numel(),
            planner_high.count,
            required(planner_high),
            main_high.candidates.numel(),
            main_high.count,
            intermediate_high.candidates.numel(),
            intermediate_high.count,
            max(required(main_high), required(intermediate_high)),
        )
    return FADAReplaySamplingPlan(planner, idm, coverage)


def planner_sample_indices(
    batch_size: int,
    *,
    planner_eligible: torch.Tensor,
    command_scenario: torch.Tensor,
    cold_start: torch.Tensor,
    command: torch.Tensor,
    sampling_spec: FADAReplaySamplingSpec,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    return build_replay_sampling_plan(
        batch_size,
        planner_eligible=planner_eligible,
        command_scenario=command_scenario,
        cold_start=cold_start,
        command=command,
        spec=sampling_spec,
        suboptimal_retention_ratio=None,
    ).sample_planner(generator)


def validate_sampling_batch_size(
    spec: FADAReplaySamplingSpec, *, batch_size: int, suboptimal_retention_ratio: int
) -> None:
    _planner_counts(batch_size, spec)
    for role, count in _role_counts(batch_size, suboptimal_retention_ratio):
        if role == "main":
            _planner_counts(count, spec)
        elif spec.walk_steady_speed_ratios is not None:
            _allocate_ratio_counts(
                count,
                spec.walk_steady_speed_ratios,
                ordered_names=FADA_WALK_SPEED_BINS,
                label="intermediate walk steady speed",
            )


def official_replay_source_counts(
    config: Mapping[str, Any], *, mode: str, planner: bool
) -> dict[str, int]:
    """Pinned FADA source budgets, including Python's original round semantics."""

    if mode not in {"offline", "mixed"}:
        raise ValueError(f"unknown official training mode: {mode}")
    size = int(config["batch_size"])
    if size <= 0:
        raise ValueError("official batch_size must be positive")
    planner_suboptimal = (
        float(config["planner_suboptimal_batch_ratio"]) > 0
        and float(config["suboptimal_data_ratio"]) > 0
    )
    suboptimal_enabled = planner_suboptimal or (
        bool(config["use_generalized_idm"]) and float(config["suboptimal_data_ratio"]) > 0
    )
    name = "planner" if planner else "idm"
    ratio_name = f"warmup_{name}_suboptimal_batch_ratio" if mode == "offline" else f"{name}_suboptimal_batch_ratio"
    enabled = planner_suboptimal if planner else suboptimal_enabled
    total_suboptimal = max(0, min(int(round(size * float(config[ratio_name]))) if enabled else 0, size))
    expert_count = (
        max(0, min(int(round(total_suboptimal * float(config["suboptimal_expert_batch_ratio"]))), total_suboptimal))
        if not planner else 0
    )
    remaining = size - total_suboptimal
    optimal = remaining if mode == "offline" else max(
        0, min(int(round(remaining * float(config["mixed_batch_offline_ratio"]))), remaining)
    )
    online_total = remaining - optimal
    trajectory_count = 0
    if not planner:
        trajectory_count = online_total if not config["use_generalized_idm"] else max(
            0, min(int(round(online_total * float(config["online_trajectory_batch_ratio"]))), online_total)
        )
    return {
        "optimal": optimal,
        "suboptimal": total_suboptimal - expert_count,
        "suboptimal_expert": expert_count,
        "online": online_total - trajectory_count,
        "online_trajectory": trajectory_count,
    }
