"""Real-action WM identification and persistent frozen-prior context adaptation."""
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
from torch import nn

from unilab.algos.torch.distill.fada.adaptation import FADALoRALinear
from unilab.algos.torch.distill.fada.runtime_policy import FADAFutureQuery
from unilab.algos.torch.sttt.path_following.features import context_dimension
from unilab.algos.torch.sttt.path_following.context_adaptation import ContextAdaptationConfig, ContextAdapter
from unilab.algos.torch.sttt.path_following.latent_actor import LatentResidualActor
from unilab.algos.torch.sttt.path_following.latent_rollout import PlannerTrackerRollout
from unilab.algos.torch.sttt.path_following.online_learning import ActorUpdateContext
from unilab.algos.torch.sttt.path_following.reference import FixedStraightPath
from unilab.algos.torch.sttt.path_following.trajectory import MeasuredTrajectory, TrajectoryConfig
from unilab.algos.torch.sttt.world_model.checkpoint import LoadedWorldModel
from unilab.algos.torch.sttt.world_model.losses import world_model_loss
from unilab.algos.torch.sttt.world_model.model import LatentWorldModel, WorldModelConfig
from unilab.algos.torch.sttt.world_model.sequence_collection import window_sequence
from unilab.algos.torch.sttt.world_model.sequence_data import batch_sequences
from unilab.algos.torch.sttt.world_model.types import RolloutStart


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class LatentOnlineConfig:
    residual_limit_rad: tuple[float, ...]
    actor_hidden_dims: tuple[int, ...] = (256, 128)
    seed: int = 0
    calibration_windows: int = 2
    horizon_windows: int = 2
    replay_capacity_windows: int = 32
    replay_sequence_windows: int = 2
    wm_updates_per_cycle: int = 1
    wm_learning_rate: float = 1e-4
    context_learning_rate: float = .001
    context_dim: int = 16
    context_updates_per_cycle: int = 1
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    adapt_world_model: bool = True
    risk_threshold: float = 0.5
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    adapt_context: bool = True

    def __post_init__(self) -> None:
        if len(self.residual_limit_rad) != 29:
            raise ValueError("provide 29 explicit source-order residual limits in radians")
        for limit in self.residual_limit_rad:
            _positive(limit, "residual limit")
        if not self.actor_hidden_dims:
            raise ValueError("actor hidden dimensions must be nonempty")
        for width in self.actor_hidden_dims:
            _count(width, "actor width")
        for name in ("calibration_windows", "horizon_windows", "replay_capacity_windows", "replay_sequence_windows", "wm_updates_per_cycle", "lora_rank", "context_dim", "context_updates_per_cycle"):
            _count(getattr(self, name), name)
        if self.replay_capacity_windows < self.replay_sequence_windows:
            raise ValueError("replay capacity must cover the imagined identification sequence")
        for name in ("wm_learning_rate", "context_learning_rate", "lora_alpha", "max_grad_norm"):
            _positive(getattr(self, name), name)
        if not math.isfinite(self.lora_weight_decay) or self.lora_weight_decay < 0:
            raise ValueError("LoRA weight decay must be finite and nonnegative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(self.adapt_world_model, bool):
            raise ValueError("adapt_world_model must be boolean")
        if not isinstance(self.adapt_context, bool) or not isinstance(self.trajectory, TrajectoryConfig):
            raise ValueError("adapt_context/trajectory have invalid types")
        if isinstance(self.risk_threshold, bool) or not math.isfinite(self.risk_threshold) or not 0 < self.risk_threshold < 1:
            raise ValueError("risk_threshold must be a probability in (0,1)")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> LatentOnlineConfig:
        raw = dict(values)
        if "residual_limit_rad" not in raw or raw["residual_limit_rad"] is None:
            raise ValueError("set explicit residual_limit_rad from the deployed controller's usable range")
        raw["residual_limit_rad"] = tuple(raw["residual_limit_rad"])
        if "actor_hidden_dims" in raw:
            raw["actor_hidden_dims"] = tuple(raw["actor_hidden_dims"])
        if "trajectory" in raw:
            raw["trajectory"] = TrajectoryConfig.from_mapping(raw["trajectory"])
        return cls(**raw)


@dataclass(frozen=True)
class LatentUpdateResult:
    b_loss: float | None
    a_loss: float | None
    b_windows: int
    calibration_zero_windows: int
    calibration_complete: bool
    actor_skip_reason: str | None
    elapsed_s: float


@dataclass(frozen=True)
class PreparedLatentUpdate:
    samples: tuple[Any, ...]
    sequences: tuple[dict[str, Any], ...]
    start: RolloutStart | None




def _join_windows(sequences: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Join contiguous windows without inserting artificial recurrent resets."""
    if not sequences:
        raise ValueError("identification requires at least one complete window")
    observation_keys = {"observations", "ego_motion", "ego_valid", "ego_age_s", "is_first", "timestamps_ns"}
    window_keys = {"window_start", "window_end", "window_delta", "window_valid"}
    result: dict[str, Any] = {}
    for key in sequences[0]:
        values = [torch.as_tensor(sequence[key]) for sequence in sequences]
        if key in {"window_start", "window_end"}:
            offset, pieces = 0, []
            for sequence, value in zip(sequences, values):
                pieces.append(value + offset)
                offset += len(sequence["actions"])
            result[key] = torch.cat(pieces)
        elif key in observation_keys:
            result[key] = torch.cat([values[0], *(value[1:] for value in values[1:])])
        elif key in window_keys:
            result[key] = torch.cat(values)
        else:
            result[key] = torch.cat(values)
    result["is_first"] = torch.zeros_like(result["is_first"], dtype=torch.bool)
    result["is_first"][0] = True
    return result


class LatentPathFollowingUpdater:
    """Own WM LoRA and c; A/FiLM weights and the playback clock remain fixed."""

    def __init__(self, loaded: LoadedWorldModel, rollout: PlannerTrackerRollout,
                 config: LatentOnlineConfig) -> None:
        if rollout.model is not loaded.model:
            raise ValueError("rollout and learner must share the loaded physical model")
        self.model = loaded.model
        self.contract = copy.deepcopy(loaded.contract)
        self.rollout, self.config = rollout, config
        self.window_steps = _count(self.contract["data"]["window_steps"], "window_steps")
        self.ctrl_dt = _positive(self.contract["data"]["ctrl_dt"], "ctrl_dt")
        self.model.eval().requires_grad_(False)
        self.model.configure_observation_loss(self.contract["source"]["preprocessing"])
        self.loss_config = {**loaded.training_state.get("loss_config", {}), "risk_scale": 0.}
        targets = tuple(name for name, module in self.model.named_modules()
                        if name and isinstance(module, nn.Linear)
                        and name.split(".", 1)[0] in {"prior_net", "motion_head"})
        if not targets:
            raise ValueError("the physical model exposes no supported Linear adaptation targets")
        self.adapter_targets = targets
        for name in targets:
            base = self.model.get_submodule(name)
            if not isinstance(base, nn.Linear):
                raise TypeError(f"WM adaptation target is not Linear: {name}")
            parent_name, _, child = name.rpartition(".")
            parent = self.model.get_submodule(parent_name) if parent_name else self.model
            setattr(parent, child, FADALoRALinear(base, rank=config.lora_rank, alpha=config.lora_alpha, dropout=0.0))
        self.adapters = [parameter for name, parameter in self.model.named_parameters()
                         if name.endswith(("lora_A.weight", "lora_B.weight"))]
        if not self.adapters or {id(p) for p in self.adapters} != {id(p) for p in self.model.parameters() if p.requires_grad}:
            raise ValueError("WM optimizer ownership differs from the adapter manifest")
        parameter = next(self.model.parameters())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.actor = LatentResidualActor(self.model.feature_dim, config.actor_hidden_dims, config.residual_limit_rad,
                                              command_dim=rollout.spec.command_dim,
                                              preview_points=config.trajectory.preview_points,
                                              adaptation_dim=config.context_dim).to(parameter).eval().requires_grad_(False)
        self.model_optimizer = torch.optim.AdamW(self.adapters, lr=config.wm_learning_rate, weight_decay=config.lora_weight_decay)
        self.context_adapter = ContextAdapter(self.actor, rollout, batch_size=1, window_steps=self.window_steps,
            config=ContextAdaptationConfig(dimension=config.context_dim, learning_rate=config.context_learning_rate,
                updates_per_window=config.context_updates_per_cycle, horizon_windows=config.horizon_windows,
                max_grad_norm=config.max_grad_norm))
        self.inference_context = self.context_adapter.context.detach().clone()
        self.inference_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.inference_actor = copy.deepcopy(self.actor).eval().requires_grad_(False)
        source = self.contract["source"]
        self.inference_rollout = PlannerTrackerRollout(self.inference_model, rollout.policy,
            rollout.profile, rollout.preprocessing, target_joint_names=source["target_joint_names"],
            target_default_joint_angles=source["target_default_joint_angles"],
            target_action_scale=source["target_action_scale"])
        self._publication = None
        self.replay: deque[tuple[Any, dict[str, Any]]] = deque(maxlen=config.replay_capacity_windows)
        self.records: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self.context_updates = self.response_updates = self.zero_calibration_windows = 0
        self._reference: FixedStraightPath | MeasuredTrajectory | None = None
        self._last_window_id = -1
        self.previous_residual = torch.zeros(1, 29, device=parameter.device, dtype=parameter.dtype)
        self.ego_history: Any = None

    def initialize_pretrained(self, state: Mapping[str, Any]) -> None:
        if state.get("algorithm") != "rsl_rl_ppo" or state.get("stage") != "film":
            raise ValueError("online c adaptation requires a stage-2 FiLM PPO artifact")
        self.actor.eval().requires_grad_(False)
        self.inference_actor.load_state_dict(self.actor.state_dict(), strict=True)
        self.context_adapter.reset()
        self.stage_publication()
        self.publish()

    def update_counters(self) -> dict[str, int]:
        return {"context_updates": self.context_updates, "b_updates": self.response_updates}

    def stage_publication(self) -> None:
        self._publication = (
            {name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()},
            self.context_adapter.context.detach().cpu().clone(),
        )

    def publish(self) -> bool:
        """Only the control thread activates a complete, finished learner snapshot."""
        if self._publication is None:
            return False
        model, context = self._publication
        self.inference_model.load_state_dict(model, strict=True)
        self.inference_context.copy_(context.to(self.inference_context))
        self._publication = None
        return True

    @property
    def calibration_complete(self) -> bool:
        return self.zero_calibration_windows >= self.config.calibration_windows

    def discard_pending(self) -> None:
        self.records.clear()
        self.replay.clear()
        self._last_window_id = -1
        self.previous_residual.zero_()
        if not self.calibration_complete:
            self.zero_calibration_windows = 0

    def begin_episode(self, reference: FixedStraightPath) -> None:
        if self._reference is not None and (reference.sequence_id, reference.tracking_epoch) <= (self._reference.sequence_id, self._reference.tracking_epoch):
            raise ValueError("a new localization episode must advance identity")
        self._reference = reference
        self._last_window_id = -1

    def record_transition(self, transition_id: int, record: Mapping[str, Any], *, capacity: int) -> None:
        if transition_id in self.records or len(self.records) >= capacity:
            raise RuntimeError("duplicate or full pending WM observation sequence")
        self.records[transition_id] = dict(record)

    def start_from_decision(self, decision: ActorUpdateContext) -> RolloutStart:
        data = self.contract["data"]
        history, horizon = data["history_length"], data["prediction_horizon"]
        context = decision.context
        width, command_dim = self.model.config.obs_dim, self.rollout.spec.command_dim
        if context.shape != (1, context_dimension(history, horizon, observation_dim=width, command_dim=command_dim)):
            raise ValueError("decision history is incompatible with the loaded control contract")
        obs_end = width * history
        act_end = obs_end + 29 * history
        mask_end = act_end + history
        query = FADAFutureQuery(context[:, :obs_end].reshape(1, history, width),
                               context[:, obs_end:act_end].reshape(1, history, 29),
                               context[:, mask_end:mask_end + command_dim], context[:, act_end:mask_end].bool(),
                               context[:, mask_end + command_dim:].reshape(1, horizon, width),
                               self.contract["source"]["preprocessing"], self.rollout.profile)
        ego = np.zeros((history, 3), dtype=np.float32)
        valid = np.zeros((history, 1), dtype=bool)
        age = np.zeros((history, 1), dtype=np.float32)
        if self.ego_history is not None:
            dt_ns = round(self.ctrl_dt * 1e9)
            for index in range(history):
                timestamp = decision.timestamp_ns - (history - 1 - index) * dt_ns
                if timestamp >= 0:
                    ego[index], valid[index, 0], age[index, 0] = self.ego_history.at(timestamp)
        parameter = next(self.model.parameters())
        with torch.no_grad():
            return self.inference_rollout.encode_start(
                query, errors=decision.errors.to(parameter), previous_residual=self.previous_residual,
                ego_motion=torch.as_tensor(ego, device=parameter.device, dtype=parameter.dtype).unsqueeze(0),
                ego_valid=torch.as_tensor(valid, device=parameter.device).unsqueeze(0),
                ego_age_s=torch.as_tensor(age, device=parameter.device, dtype=parameter.dtype).unsqueeze(0),
                path_state=decision.path_state,
            )

    def candidate(self, decision: ActorUpdateContext) -> torch.Tensor:
        start = self.start_from_decision(decision)
        with torch.no_grad():
            if start.path_state is None:
                raise ValueError("A requires the current reference preview")
            return self.inference_actor(start.latent.feature(), start.errors, start.command, start.path_state.preview(),
                                        start.previous_residual, adaptation_context=self.inference_context)

    def _recent_batch(self) -> dict[str, torch.Tensor] | None:
        count = self.config.replay_sequence_windows
        if len(self.replay) < count:
            return None
        entries = list(self.replay)[-count:]
        for (left, _), (right, _) in zip(entries, entries[1:]):
            if left.reference != right.reference or left.controls[-1].end_ns != right.controls[0].start_ns:
                return None
        sequence = _join_windows([entry for _, entry in entries])
        return batch_sequences([sequence], device=next(self.model.parameters()).device, contract=self.contract)

    def prepare_update(self, completed_windows: Sequence[Any],
                       actor_context: ActorUpdateContext | None = None) -> PreparedLatentUpdate:
        """Detach causal inputs on the control thread before handing them to learning."""
        samples = tuple(completed_windows)
        sequences = tuple(window_sequence(sample, self.records) for sample in samples)
        start = self.start_from_decision(actor_context) if actor_context is not None else None
        for sample in samples:
            for control in sample.controls:
                self.records.pop(control.transition_id)
        return PreparedLatentUpdate(samples, sequences, start)

    def update(self, completed_windows: Sequence[Any],
               actor_context: ActorUpdateContext | None = None) -> LatentUpdateResult:
        result = self.update_prepared(self.prepare_update(completed_windows, actor_context))
        self.publish()
        return result

    @torch.enable_grad()
    def update_prepared(self, prepared: PreparedLatentUpdate) -> LatentUpdateResult:
        if torch.is_inference_mode_enabled():
            raise ValueError("online learning must run outside inference_mode")
        started = time.perf_counter()
        wm_value = actor_value = None
        for sample, sequence in zip(prepared.samples, prepared.sequences):
            if self._reference is None:
                self.begin_episode(sample.reference)
            if sample.reference != self._reference or sample.window_id <= self._last_window_id:
                raise ValueError("WM windows must be new and from the current localization episode")
            self.replay.append((sample, sequence))
            self._last_window_id = sample.window_id
            if not bool(torch.count_nonzero(sample.start_query.residual_rad)):
                self.zero_calibration_windows = min(self.config.calibration_windows, self.zero_calibration_windows + 1)
        batch = self._recent_batch()
        if batch is not None and self.config.adapt_world_model:
            self.model.train()
            try:
                for _ in range(self.config.wm_updates_per_cycle):
                    self.model_optimizer.zero_grad(set_to_none=True)
                    losses = world_model_loss(self.model, batch, **self.loss_config)
                    loss = losses["loss"]
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("non-finite WM adaptation cost")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.adapters, self.config.max_grad_norm, error_if_nonfinite=True)
                    self.model_optimizer.step()
                    wm_value = float(loss.detach())
                    self.response_updates += 1
            finally:
                self.model_optimizer.zero_grad(set_to_none=True)
                self.model.eval()
        reason = "calibration" if not self.calibration_complete else ("missing_context" if prepared.start is None else None)
        if not self.config.adapt_context:
            reason = "frozen_context"
        if reason is None:
            metrics = self.context_adapter.step(prepared.start)
            actor_value = metrics["context_loss"]
            self.context_updates = self.context_adapter.updates
        if wm_value is not None or actor_value is not None:
            self.stage_publication()
        return LatentUpdateResult(wm_value, actor_value, len(prepared.samples), self.zero_calibration_windows,
                                  self.calibration_complete, reason, time.perf_counter() - started)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        base_state: dict[str, torch.Tensor] = {}
        adapters: dict[str, torch.Tensor] = {}
        for name, value in self.model.state_dict().items():
            if name.endswith(("lora_A.weight", "lora_B.weight")):
                adapters[name] = value.detach().cpu()
                continue
            for target in self.adapter_targets:
                if name.startswith(target + ".base."):
                    name = target + "." + name[len(target + ".base."):]
                    break
            base_state[name] = value.detach().cpu()
        payload = {"format": "sttt_latent_path_following_online", "schema_version": 5,
                   "model_config": asdict(self.model.config), "contract": self.contract,
                   "base_state": base_state, "adapters": adapters, "adapter_targets": list(self.adapter_targets),
                   "actor": self.actor.state_dict(), "config": asdict(self.config),
                   "model_optimizer": self.model_optimizer.state_dict(),
                   "context_adapter": self.context_adapter.state_dict(), "loss_config": self.loss_config,
                   "context_updates": self.context_updates, "response_updates": self.response_updates,
                   "zero_calibration_windows": self.zero_calibration_windows}
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(destination)

    @classmethod
    def restore(cls, path: str | Path, rollout_factory: Any, *, device: str = "cpu", expected_contract: Mapping[str, Any] | None = None) -> LatentPathFollowingUpdater:
        payload = torch.load(path, map_location=device, weights_only=True)
        if not isinstance(payload, dict) or payload.get("format") != "sttt_latent_path_following_online" or payload.get("schema_version") != 5:
            raise ValueError("expected a version-5 frozen-prior/FiLM/context online checkpoint")
        if expected_contract is not None and payload["contract"] != dict(expected_contract):
            raise ValueError("online checkpoint control contract differs from deployment")
        model = LatentWorldModel(WorldModelConfig.from_mapping(payload["model_config"])).to(device)
        model.load_state_dict(payload["base_state"], strict=True)
        loaded = LoadedWorldModel(model, payload["contract"], {"loss_config": payload["loss_config"]})
        updater = cls(loaded, rollout_factory(model), LatentOnlineConfig.from_mapping(payload["config"]))
        if tuple(payload["adapter_targets"]) != updater.adapter_targets:
            raise ValueError("online LoRA target manifest differs from this physical model")
        parameters = dict(updater.model.named_parameters())
        if set(payload["adapters"]) != {name for name in parameters if name.endswith(("lora_A.weight", "lora_B.weight"))}:
            raise ValueError("online checkpoint adapter weights do not match their manifest")
        with torch.no_grad():
            for name, value in payload["adapters"].items():
                parameters[name].copy_(value)
        updater.actor.load_state_dict(payload["actor"], strict=True)
        updater.model_optimizer.load_state_dict(payload["model_optimizer"])
        updater.inference_actor.load_state_dict(updater.actor.state_dict(), strict=True)
        updater.context_adapter.load_state_dict(payload["context_adapter"])
        for name in ("context_updates", "response_updates", "zero_calibration_windows"):
            setattr(updater, name, int(payload[name]))
        updater.stage_publication()
        updater.publish()
        return updater
