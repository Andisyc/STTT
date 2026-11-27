"""Real-action WM LoRA identification, followed by Planner-only BPTT adaptation."""
from __future__ import annotations

import copy
import math
import time
from collections import OrderedDict, deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from unilab.algos.torch.sttt.path_following.features import context_dimension
from unilab.algos.torch.sttt.path_following.latent_rollout import PlannerTrackerRollout
from unilab.algos.torch.sttt.path_following.online_learning import ActorUpdateContext
from unilab.algos.torch.sttt.path_following.planner_lora import (
    adapter_parameters, adapter_state, inject_planner_lora, inject_wm_lora, load_adapter_state,
)
from unilab.algos.torch.sttt.path_following.planner_losses import (
    identification_loss, planner_trajectory_loss, timed_reference,
)
from unilab.algos.torch.sttt.path_following.trajectory import MeasuredTrajectory, TrajectoryConfig
from unilab.algos.torch.sttt.world_model.checkpoint import LoadedWorldModel, _safe_training_state, canonical_contract
from unilab.algos.torch.sttt.world_model.model import LatentWorldModel, WorldModelConfig
from unilab.algos.torch.sttt.world_model.online import _join_windows
from unilab.algos.torch.sttt.world_model.sequence_collection import window_sequence
from unilab.algos.torch.sttt.world_model.sequence_data import batch_sequences
from unilab.algos.torch.sttt.world_model.types import RolloutStart


@dataclass(frozen=True)
class PlannerOnlineConfig:
    calibration_windows: int = 2
    horizon_windows: int = 2
    replay_capacity_windows: int = 32
    replay_sequence_windows: int = 2
    wm_updates_per_cycle: int = 1
    planner_updates_per_cycle: int = 1
    wm_learning_rate: float = .0001
    planner_learning_rate: float = .0001
    lora_rank: int = 8
    lora_alpha: float = 16.
    max_grad_norm: float = 1.
    parameter_change_scale: float = .001
    position_tolerance_m: float = .3
    heading_tolerance_rad: float = .3
    motion_xy_tolerance_m: float = .01
    motion_yaw_tolerance_rad: float = .02
    adapt_world_model: bool = True
    adapt_planner: bool = True
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)

    def __post_init__(self) -> None:
        for name in ("calibration_windows", "horizon_windows", "replay_capacity_windows", "replay_sequence_windows",
                     "wm_updates_per_cycle", "planner_updates_per_cycle", "lora_rank"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("wm_learning_rate", "planner_learning_rate", "lora_alpha", "max_grad_norm",
                     "position_tolerance_m", "heading_tolerance_rad", "motion_xy_tolerance_m", "motion_yaw_tolerance_rad"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if isinstance(self.parameter_change_scale, bool) or not math.isfinite(self.parameter_change_scale) or self.parameter_change_scale < 0:
            raise ValueError("parameter_change_scale must be finite and nonnegative")
        if self.replay_capacity_windows < self.replay_sequence_windows:
            raise ValueError("replay must cover the identification sequence")
        if type(self.adapt_world_model) is not bool or type(self.adapt_planner) is not bool or not isinstance(self.trajectory, TrajectoryConfig):
            raise ValueError("adaptation flags/trajectory have invalid types")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> PlannerOnlineConfig:
        raw = dict(values)
        if "trajectory" in raw:
            raw["trajectory"] = TrajectoryConfig.from_mapping(raw["trajectory"])
        return cls(**raw)


@dataclass(frozen=True)
class ObservedPlannerInput:
    decision: ActorUpdateContext
    ego_motion: torch.Tensor
    ego_valid: torch.Tensor
    ego_age_s: torch.Tensor


@dataclass(frozen=True)
class PreparedPlannerUpdate:
    entries: tuple[tuple[Any, dict[str, Any], ObservedPlannerInput], ...]
    decision: ObservedPlannerInput | None


@dataclass(frozen=True)
class PlannerUpdateResult:
    wm_loss: float | None
    planner_loss: float | None
    body_loss: float | None
    motion_loss: float | None
    elapsed_s: float


class PlannerLoRAUpdater:
    """The learner owns separate weights; only the control thread publishes a finished pair."""

    def __init__(self, loaded: LoadedWorldModel, rollout: PlannerTrackerRollout,
                 config: PlannerOnlineConfig) -> None:
        if loaded.model is not rollout.model:
            raise ValueError("WM learner and differentiable rollout must share one physical model")
        self.model, self.rollout, self.config = loaded.model, rollout, config
        self.contract = canonical_contract(loaded.contract)
        self.window_steps = int(self.contract["data"]["window_steps"])
        self.ctrl_dt = float(self.contract["data"]["ctrl_dt"])
        self.live_policy = rollout.policy
        rollout.policy = copy.deepcopy(self.live_policy).eval().requires_grad_(False)
        self.policy = rollout.policy
        self.model.configure_observation_loss(self.contract["source"]["preprocessing"])
        self.wm_targets = inject_wm_lora(self.model, rank=config.lora_rank, alpha=config.lora_alpha)
        self.planner_targets = inject_planner_lora(self.policy, rank=config.lora_rank, alpha=config.lora_alpha)
        if inject_planner_lora(self.live_policy, rank=config.lora_rank, alpha=config.lora_alpha) != self.planner_targets:
            raise ValueError("live and learner Planner manifests differ")
        self.live_policy.requires_grad_(False)
        self.wm_parameters = list(adapter_parameters(self.model).values())
        self.planner_parameters = list(adapter_parameters(self.policy).values())
        motion_ids = {id(p) for name, p in adapter_parameters(self.model).items() if name.startswith("motion_head.")}
        self.model_optimizer = torch.optim.AdamW([
            {"params": [p for p in self.wm_parameters if id(p) not in motion_ids], "lr": config.wm_learning_rate},
            {"params": [p for p in self.wm_parameters if id(p) in motion_ids], "lr": config.wm_learning_rate * .1},
        ], betas=(.9, .99), weight_decay=0.)
        self.planner_optimizer = torch.optim.AdamW(self.planner_parameters, lr=config.planner_learning_rate,
                                                    betas=(.9, .99), weight_decay=0.)
        self.inference_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.records: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self.replay: deque = deque(maxlen=config.replay_capacity_windows)
        self.ego_history: Any = None
        self.wm_updates = self.planner_updates = self.observed_windows = 0
        self._reference: MeasuredTrajectory | None = None
        self._last_window_id = -1
        self._publication: Any = None

    @property
    def calibration_complete(self) -> bool:
        return self.observed_windows >= self.config.calibration_windows

    def update_counters(self) -> dict[str, int]:
        return {"wm_updates": self.wm_updates, "planner_updates": self.planner_updates}

    def begin_episode(self, reference: MeasuredTrajectory) -> None:
        if not isinstance(reference, MeasuredTrajectory):
            raise TypeError("Planner adaptation requires a time-anchored measured trajectory")
        if self._reference is not None and (reference.sequence_id, reference.tracking_epoch) <= (self._reference.sequence_id, self._reference.tracking_epoch):
            raise ValueError("a new episode must advance localization identity")
        self.replay.clear()
        self._last_window_id = -1
        self._reference = reference

    def discard_pending(self) -> None:
        self.records.clear()
        self.replay.clear()
        self._last_window_id = -1

    def record_transition(self, transition_id: int, record: Mapping[str, Any], *, capacity: int) -> None:
        if transition_id in self.records or len(self.records) >= capacity:
            raise RuntimeError("duplicate or full pending WM transition buffer")
        self.records[transition_id] = dict(record)

    def _own_input(self, decision: ActorUpdateContext) -> ObservedPlannerInput:
        history = self.contract["data"]["history_length"]
        parameter = next(self.model.parameters())
        ego = np.zeros((history, 3), dtype=np.float32)
        valid = np.zeros((history, 1), dtype=bool)
        age = np.zeros((history, 1), dtype=np.float32)
        if self.ego_history is not None:
            dt_ns = round(self.ctrl_dt * 1e9)
            for index in range(history):
                timestamp = decision.timestamp_ns - (history - 1 - index) * dt_ns
                if timestamp >= 0:
                    ego[index], valid[index, 0], age[index, 0] = self.ego_history.at(timestamp)
        return ObservedPlannerInput(decision,
            torch.as_tensor(ego, device=parameter.device, dtype=parameter.dtype).unsqueeze(0),
            torch.as_tensor(valid, device=parameter.device).unsqueeze(0),
            torch.as_tensor(age, device=parameter.device, dtype=parameter.dtype).unsqueeze(0))

    def _encode(self, inputs: ObservedPlannerInput) -> RolloutStart:
        decision, data = inputs.decision, self.contract["data"]
        history, horizon, width = data["history_length"], data["prediction_horizon"], self.model.config.obs_dim
        command_dim = self.rollout.spec.command_dim
        context = decision.context
        if context.shape != (1, context_dimension(history, horizon, observation_dim=width, command_dim=command_dim)):
            raise ValueError("stored decision differs from the deployed observation/command contract")
        obs_end, action_end = history * width, history * (width + 29)
        mask_end = action_end + history
        return self.rollout.encode_observed_start(
            context[:, :obs_end].reshape(1, history, width),
            context[:, obs_end:action_end].reshape(1, history, 29), context[:, action_end:mask_end].bool(),
            context[:, mask_end:mask_end + command_dim], errors=decision.errors,
            previous_residual=context.new_zeros(1, 29), path_state=decision.path_state,
            ego_motion=inputs.ego_motion, ego_valid=inputs.ego_valid, ego_age_s=inputs.ego_age_s,
        )

    def prepare_update(self, samples: Sequence[Any], decision: ActorUpdateContext | None = None) -> PreparedPlannerUpdate:
        entries = tuple((sample, window_sequence(sample, self.records),
                         self._own_input(ActorUpdateContext.from_window_start(sample))) for sample in samples)
        current = None if decision is None else self._own_input(decision)
        for sample in samples:
            for control in sample.controls:
                self.records.pop(control.transition_id)
        return PreparedPlannerUpdate(entries, current)

    @torch.enable_grad()
    def update_prepared(self, prepared: PreparedPlannerUpdate) -> PlannerUpdateResult:
        if torch.is_inference_mode_enabled():
            raise ValueError("dual LoRA learning must run outside inference_mode")
        started = time.perf_counter()
        wm_value = planner_value = body_value = motion_value = None
        for sample, sequence, seed in prepared.entries:
            if sample.reference is not self._reference or sample.window_id <= self._last_window_id:
                raise ValueError("WM windows must be new and from the bound localization episode")
            self.replay.append((sample, sequence, seed))
            self._last_window_id = sample.window_id
            self.observed_windows += 1
        count = self.config.replay_sequence_windows
        recent = list(self.replay)[-count:]
        contiguous = len(recent) == count and all(
            left[0].reference is right[0].reference and left[0].controls[-1].end_ns == right[0].controls[0].start_ns
            for left, right in zip(recent, recent[1:]))
        if contiguous and self.config.adapt_world_model:
            batch = batch_sequences([_join_windows([row[1] for row in recent])],
                                    device=next(self.model.parameters()).device, contract=self.contract)
            for _ in range(self.config.wm_updates_per_cycle):
                self.model_optimizer.zero_grad(set_to_none=True)
                initial = self._encode(recent[0][2]).latent
                losses = identification_loss(self.model, initial, batch,
                    xy_tolerance_m=self.config.motion_xy_tolerance_m, yaw_tolerance_rad=self.config.motion_yaw_tolerance_rad)
                self._step(losses["loss"], self.wm_parameters, self.model_optimizer)
                wm_value, body_value, motion_value = (float(losses[key].detach()) for key in ("loss", "body_loss", "motion_loss"))
                self.wm_updates += 1
        if self.calibration_complete and prepared.decision is not None and self.config.adapt_planner:
            decision = prepared.decision.decision
            if decision.reference is not self._reference or decision.path_state is None:
                raise ValueError("Planner decision must belong to the current measured reference")
            horizon = self.window_steps * self.config.horizon_windows
            target, commands = timed_reference(decision.path_state.reference,
                (decision.timestamp_ns - decision.reference.timestamp_ns) / 1e9, horizon, self.ctrl_dt)
            before = [parameter.detach().clone() for parameter in self.planner_parameters]
            for _ in range(self.config.planner_updates_per_cycle):
                self.planner_optimizer.zero_grad(set_to_none=True)
                # Re-encode raw history AFTER WM identification; never reuse the old latent.
                start = self._encode(prepared.decision)
                prediction = self.rollout.run(start, None, window_steps=self.window_steps,
                    horizon_windows=self.config.horizon_windows, update_reference_commands=False,
                    velocity_commands=commands)
                loss = planner_trajectory_loss(decision.path_state.pose, prediction.body_delta, target,
                    position_tolerance_m=self.config.position_tolerance_m, heading_tolerance_rad=self.config.heading_tolerance_rad)
                change = sum((parameter - previous).square().sum() for parameter, previous in zip(self.planner_parameters, before))
                change = change / sum(parameter.numel() for parameter in self.planner_parameters)
                loss = loss + self.config.parameter_change_scale * change
                self._step(loss, self.planner_parameters, self.planner_optimizer)
                planner_value = float(loss.detach())
                self.planner_updates += 1
        if wm_value is not None or planner_value is not None:
            self._publication = adapter_state(self.model), adapter_state(self.policy)
        return PlannerUpdateResult(wm_value, planner_value, body_value, motion_value, time.perf_counter() - started)

    def _step(self, loss: torch.Tensor, parameters: list[torch.nn.Parameter], optimizer: torch.optim.Optimizer) -> None:
        try:
            if not bool(torch.isfinite(loss)):
                raise ValueError("non-finite online adaptation loss")
            loss.backward()
            if any(parameter.grad is None for parameter in parameters):
                raise RuntimeError("an owned LoRA parameter received no gradient")
            torch.nn.utils.clip_grad_norm_(parameters, self.config.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
        finally:
            optimizer.zero_grad(set_to_none=True)

    def publish(self) -> bool:
        if self._publication is None:
            return False
        model, planner = self._publication
        load_adapter_state(self.inference_model, model)
        load_adapter_state(self.live_policy, planner)
        self._publication = None
        return True

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        base_state = {}
        adapter_names = set(adapter_parameters(self.model))
        for name, value in self.model.state_dict().items():
            if name in adapter_names:
                continue
            for target in self.wm_targets:
                if name.startswith(target + ".base."):
                    name = target + "." + name[len(target + ".base."):]
                    break
            base_state[name] = value.detach().cpu().clone()
        payload = {"format": "sttt_planner_wm_lora", "schema_version": 1, "contract": self.contract,
            "model_config": asdict(self.model.config), "base_state": base_state,
            "wm_targets": list(self.wm_targets), "planner_targets": list(self.planner_targets),
            "wm_adapters": adapter_state(self.model), "planner_adapters": adapter_state(self.policy),
            "config": asdict(self.config), "wm_optimizer": _safe_training_state(self.model_optimizer.state_dict()),
            "planner_optimizer": _safe_training_state(self.planner_optimizer.state_dict()),
            "counters": {**self.update_counters(), "observed_windows": self.observed_windows}}
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        try:
            torch.save(payload, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def restore(cls, path: str | Path, rollout_factory: Any, *, device: str = "cpu",
                expected_contract: Mapping[str, Any] | None = None) -> PlannerLoRAUpdater:
        payload = torch.load(path, map_location=device, weights_only=True)
        if not isinstance(payload, dict) or payload.get("format") != "sttt_planner_wm_lora" or payload.get("schema_version") != 1:
            raise ValueError("expected a dual Planner/WM LoRA checkpoint; A/FiLM artifacts are incompatible")
        contract = canonical_contract(payload["contract"])
        if expected_contract is not None and contract != canonical_contract(expected_contract):
            raise ValueError("saved dual LoRA control contract differs from deployment")
        model = LatentWorldModel(WorldModelConfig.from_mapping(payload["model_config"])).to(device)
        model.load_state_dict(payload["base_state"], strict=True)
        updater = cls(LoadedWorldModel(model, contract, {}), rollout_factory(model), PlannerOnlineConfig.from_mapping(payload["config"]))
        if list(updater.wm_targets) != payload["wm_targets"] or list(updater.planner_targets) != payload["planner_targets"]:
            raise ValueError("dual LoRA manifests differ from their architecture")
        load_adapter_state(updater.model, payload["wm_adapters"])
        load_adapter_state(updater.policy, payload["planner_adapters"])
        updater.model_optimizer.load_state_dict(payload["wm_optimizer"])
        updater.planner_optimizer.load_state_dict(payload["planner_optimizer"])
        for name in ("wm_updates", "planner_updates", "observed_windows"):
            value = payload["counters"][name]
            if type(value) is not int or value < 0:
                raise ValueError("invalid saved optimizer/calibration counter")
            setattr(updater, name, value)
        updater._publication = adapter_state(updater.model), adapter_state(updater.policy)
        updater.publish()
        return updater
