"""A/B update core only: no control clock, pose polling, replay or window activation."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.distill.fada.runtime_policy import FADAFutureQuery
from unilab.algos.torch.sttt.path_following.checkpoint import LoadedResponseModel
from unilab.algos.torch.sttt.path_following.contract import InterventionQueryRecord, PathFollowingWindowSample, _integer
from unilab.algos.torch.sttt.path_following.features import pack_response_context, validate_response_contract
from unilab.algos.torch.sttt.path_following.model import ResponseModel, response_loss
from unilab.algos.torch.sttt.path_following.objectives import actor_loss
from unilab.algos.torch.sttt.path_following.reference import FixedStraightPath
from unilab.algos.torch.sttt.path_following.residual_model import ResidualModel
from unilab.algos.torch.sttt.pose_data import TimedPelvisPose
from unilab.base.backend.sensors import SensorClock

QuerySignature = tuple[tuple[str, ...], float, float, float, int, int]


def _query_signature(query: FADAFutureQuery | InterventionQueryRecord) -> QuerySignature:
    if isinstance(query, InterventionQueryRecord):
        names, dt, sq, sv = query.joint_names, query.ctrl_dt, query.position_scale, query.velocity_scale
    else:
        scales = query.preprocessing["term_scale"]
        names, dt, sq, sv = query.profile.joint_names, query.profile.ctrl_dt, scales["dof_pos"], scales["dof_vel"]
    return tuple(names), float(dt), float(sq), float(sv), query.observation_history.shape[1], query.planner_future.shape[1]


@dataclass(frozen=True)
class ActorUpdateContext:
    """Owned measurements for one explicitly paired decision point, not a B label.

    Timestamp is represented physical time, not pose arrival time. Selection of
    the latest sufficiently fresh pair belongs to the runtime, not this learner.
    """

    reference: FixedStraightPath
    timestamp_ns: int
    context: torch.Tensor
    errors: torch.Tensor
    query_signature: QuerySignature

    def __post_init__(self) -> None:
        _integer(self.timestamp_ns, "actor timestamp_ns")
        if self.timestamp_ns < self.reference.timestamp_ns:
            raise ValueError("actor decision predates its task reference")
        if self.context.ndim != 2 or self.context.shape[0] != 1 or self.context.shape[1] < 1 or self.errors.shape != (1, 2):
            raise ValueError("actor decision requires one paired context and errors [1, 2]")
        if (
            not self.context.is_floating_point() or self.context.dtype != self.errors.dtype
            or self.context.device != self.errors.device
            or not bool(torch.isfinite(self.context).all()) or not bool(torch.isfinite(self.errors).all())
        ):
            raise ValueError("actor decision requires finite, consistently typed measured inputs")
        object.__setattr__(self, "context", self.context.detach().clone())
        object.__setattr__(self, "errors", self.errors.detach().clone())

    @classmethod
    def from_aligned_query(
        cls, query: FADAFutureQuery, clock: SensorClock, pose: TimedPelvisPose, reference: FixedStraightPath,
    ) -> ActorUpdateContext:
        if clock.sequence_id != pose.sequence_id or clock.timestamp_ns != pose.timestamp_ns:
            raise ValueError("actor pose and query must represent the same sequence and physical time")
        errors = torch.as_tensor(reference.errors(pose), dtype=query.command.dtype, device=query.command.device).unsqueeze(0)
        return cls(reference, clock.timestamp_ns, pack_response_context(query), errors, _query_signature(query))

    @classmethod
    def from_window_start(cls, sample: PathFollowingWindowSample) -> ActorUpdateContext:
        """Explicit historical point; never relabel it as the current decision."""
        query = sample.start_query
        errors = torch.as_tensor(sample.path_errors[:1], dtype=query.command.dtype, device=query.command.device)
        return cls(sample.reference, sample.start_pose.timestamp_ns, pack_response_context(query), errors, _query_signature(query))


@dataclass(frozen=True)
class PathFollowingUpdateResult:
    b_loss: float | None
    a_loss: float | None
    b_windows: int
    calibration_zero_windows: int
    calibration_complete: bool
    actor_skip_reason: str | None


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


@contextmanager
def _frozen_response(response: ResponseModel) -> Iterator[None]:
    parameters = list(response.parameters())
    flags = [parameter.requires_grad for parameter in parameters]
    was_training = response.training
    response.eval()
    for parameter in parameters:
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, flag in zip(parameters, flags):
            parameter.requires_grad_(flag)
        response.train(was_training)


class PathFollowingUpdater:
    """Own A/B and fresh optimizers, but never execute a candidate.

    Caller schedules update() at T and supplies only new completed windows.
    B commits before A: if A subsequently fails, B's successful consumption is
    retained. Exceptions propagate; there is no rollback or automatic retry.
    """

    def __init__(self, loaded: LoadedResponseModel, cfg: DictConfig | Mapping[str, Any]) -> None:
        if not isinstance(loaded, LoadedResponseModel):
            raise TypeError("online A/B updates require the independent loaded B artifact")
        raw = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)
        if not isinstance(raw, dict) or set(raw) != {"seed", "calibration_windows", "actor", "response"}:
            raise ValueError("use the independent path_following_online.yaml update-core configuration")
        config = cast(dict[str, Any], raw)
        if set(config["actor"]) != {"hidden_dims", "hidden_init_std", "learning_rate"} or set(config["response"]) != {"learning_rate"}:
            raise ValueError("unknown or missing A/B update hyperparameters")
        seed = _integer(config["seed"], "seed")
        self.calibration_windows = _integer(config["calibration_windows"], "calibration_windows")
        if self.calibration_windows < 1:
            raise ValueError("startup calibration requires at least one valid zero-residual window")
        actor_lr = _positive(config["actor"]["learning_rate"], "A learning_rate")
        response_lr = _positive(config["response"]["learning_rate"], "B learning_rate")
        std = _positive(config["actor"]["hidden_init_std"], "A hidden_init_std")
        self.contract = validate_response_contract(loaded.contract)
        self.response = loaded.model
        data = self.contract["data"]
        if self.response.history_length != data["history_length"] or self.response.prediction_horizon != data["prediction_horizon"]:
            raise ValueError("B model does not match the loaded context contract")
        self._signature: QuerySignature = (
            tuple(data["joint_names"]), data["ctrl_dt"], data["position_scale"], data["velocity_scale"],
            data["history_length"], data["prediction_horizon"],
        )
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            self.actor = ResidualModel(self.response, config["actor"]["hidden_dims"], hidden_init_std=std)
        self.response.requires_grad_(True)
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=actor_lr, weight_decay=0.)
        self.response_optimizer = torch.optim.AdamW(self.response.parameters(), lr=response_lr, weight_decay=0.)
        self.zero_calibration_windows = 0
        self.actor_updates = 0
        self.response_updates = 0
        self._reference: FixedStraightPath | None = None
        self._last_window_id = -1
        self._last_window_end_ns = -1
        self._last_actor_timestamp_ns = -1

    @property
    def calibration_complete(self) -> bool:
        return self.zero_calibration_windows >= self.calibration_windows

    def begin_episode(self, reference: FixedStraightPath) -> None:
        """Explicit identity advance; keep learned weights and both optimizer states."""
        if self._reference is not None:
            old = self._reference.sequence_id, self._reference.tracking_epoch
            new = reference.sequence_id, reference.tracking_epoch
            if new <= old:
                raise ValueError("a new episode must advance measurement sequence or tracking epoch")
        if not self.calibration_complete:
            self.zero_calibration_windows = 0
        self._reference = reference
        self._last_window_id = -1
        self._last_window_end_ns = -1
        self._last_actor_timestamp_ns = -1
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.response_optimizer.zero_grad(set_to_none=True)

    def _validate_inputs(
        self, windows: tuple[PathFollowingWindowSample, ...], decision: ActorUpdateContext | None,
    ) -> FixedStraightPath | None:
        reference = self._reference
        if reference is None:
            reference = windows[0].reference if windows else (decision.reference if decision is not None else None)
        last_id, last_end = self._last_window_id, self._last_window_end_ns
        for sample in windows:
            if sample.reference != reference or _query_signature(sample.start_query) != self._signature or len(sample.controls) != self.contract["data"]["window_steps"]:
                raise ValueError("completed window belongs to another episode, source interpretation or response duration")
            if sample.window_id <= last_id or sample.controls[0].start_ns < last_end:
                raise ValueError("completed windows must be new, unique and chronologically ordered")
            last_id, last_end = sample.window_id, sample.controls[-1].end_ns
        if decision is not None:
            if decision.reference != reference or decision.query_signature != self._signature or decision.context.shape[1] != self.response.context_dim:
                raise ValueError("actor decision belongs to another episode or context interpretation")
            if decision.timestamp_ns < self._last_actor_timestamp_ns:
                raise ValueError("actor decision time moved backwards; do not use an arriving old label as current state")
        return reference

    @torch.enable_grad()
    def update(
        self, completed_windows: Sequence[PathFollowingWindowSample], actor_context: ActorUpdateContext | None = None,
    ) -> PathFollowingUpdateResult:
        if torch.is_inference_mode_enabled():
            raise ValueError("A/B updates must run outside torch.inference_mode")
        windows = tuple(completed_windows)
        reference = self._validate_inputs(windows, actor_context)
        if self._reference is None and reference is not None:
            self.begin_episode(reference)
        parameter = next(self.response.parameters())
        b_value = a_value = None
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.response_optimizer.zero_grad(set_to_none=True)
        if windows:
            context = torch.cat([pack_response_context(sample.start_query) for sample in windows]).detach().to(parameter)
            executed = torch.cat([sample.start_query.residual_rad for sample in windows]).detach().to(parameter)
            target = torch.stack([torch.as_tensor(sample.delta, dtype=parameter.dtype, device=parameter.device) for sample in windows])
            was_training = self.response.training
            self.response.train()
            try:
                loss = response_loss(self.response(context, executed), target)
                b_value = float(loss.detach())
                if not math.isfinite(b_value):
                    raise ValueError("non-finite B adaptation loss")
                loss.backward()
                self.response_optimizer.step()
            finally:
                self.response_optimizer.zero_grad(set_to_none=True)
                self.response.train(was_training)
            self.response_updates += 1
            self._last_window_id = windows[-1].window_id
            self._last_window_end_ns = windows[-1].controls[-1].end_ns
            zeros = sum(not bool(torch.count_nonzero(sample.start_query.residual_rad)) for sample in windows)
            self.zero_calibration_windows = min(self.calibration_windows, self.zero_calibration_windows + zeros)
        reason = "calibration" if not self.calibration_complete else ("missing_context" if actor_context is None else None)
        if reason is None:
            assert actor_context is not None
            context = actor_context.context.detach().to(parameter)
            errors = actor_context.errors.detach().to(parameter)
            try:
                with _frozen_response(self.response):
                    candidate = self.actor(errors, context)
                    loss = actor_loss(errors, self.response(context, candidate))
                    a_value = float(loss.detach())
                    if not math.isfinite(a_value):
                        raise ValueError("non-finite A adaptation loss")
                    loss.backward()
                    self.actor_optimizer.step()
            finally:
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.response_optimizer.zero_grad(set_to_none=True)
            self.actor_updates += 1
            self._last_actor_timestamp_ns = actor_context.timestamp_ns
        return PathFollowingUpdateResult(
            b_value, a_value, len(windows), self.zero_calibration_windows, self.calibration_complete, reason,
        )
