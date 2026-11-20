"""Opt-in latent WM binding on the existing measured-window controller lifecycle."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from unilab.algos.torch.sttt.odometry_runtime import create_pose_runtime
from unilab.algos.torch.sttt.path_following.latent_objectives import LatentCostConfig
from unilab.algos.torch.sttt.path_following.latent_rollout import PlannerTrackerRollout
from unilab.algos.torch.sttt.path_following.runtime import PathFollowingController, PathFollowingRuntimeConfig
from unilab.algos.torch.sttt.world_model.checkpoint import LoadedWorldModel, load_world_model
from unilab.algos.torch.sttt.world_model.online import LatentOnlineConfig, LatentPathFollowingUpdater
from unilab.algos.torch.sttt.world_model.sequence_collection import CausalEgoHistory, proprio_tilt_rad
from unilab.algos.torch.sttt.world_model.types import LatentState
from unilab.algos.torch.sttt.pose_data import PoseRuntimeConfig


def _canonical(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def _positive_count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class LatentRuntimeSetup:
    loaded: LoadedWorldModel
    learning: LatentOnlineConfig
    costs: LatentCostConfig
    runtime: PathFollowingRuntimeConfig
    resume_checkpoint_path: Path | None
    save_checkpoint_path: Path | None

    def bind(self, playback: Any, env: Any) -> LatentPathFollowingUpdater:
        source = self.loaded.contract["source"]
        actual = {
            "target_joint_names": list(env.get_actuated_joint_names()),
            "target_default_joint_angles": np.asarray(env.default_angles).reshape(-1).tolist(),
            "target_action_scale": np.broadcast_to(np.asarray(env.cfg.control_config.action_scale), (29,)).tolist(),
        }
        for name, value in actual.items():
            if _canonical(source[name]) != _canonical(value):
                raise ValueError(f"WM deployment differs from collected {name}")
        if float(env.cfg.ctrl_dt) != self.loaded.contract["data"]["ctrl_dt"]:
            raise ValueError("WM control period differs from the actual environment")

        def rollout_factory(model: Any) -> PlannerTrackerRollout:
            return PlannerTrackerRollout(model, playback.policy, playback.profile, source["preprocessing"],
                                         target_joint_names=actual["target_joint_names"],
                                         target_default_joint_angles=actual["target_default_joint_angles"],
                                         target_action_scale=actual["target_action_scale"])

        if self.resume_checkpoint_path is not None:
            parameter = next(self.loaded.model.parameters())
            updater = LatentPathFollowingUpdater.restore(self.resume_checkpoint_path, rollout_factory,
                                                         device=str(parameter.device), expected_contract=self.loaded.contract)
            if updater.config != self.learning or updater.costs != self.costs:
                raise ValueError("resume configuration differs from the saved online learning contract")
            return updater
        return LatentPathFollowingUpdater(self.loaded, rollout_factory(self.loaded.model), self.learning, self.costs)


def prepare_latent_path_following(node: Any, *, cfg: Any, playback_cfg: Any, root_dir: str | Path,
                                   checkpoint_path: str | Path, policy: Any, profile: Any,
                                   spec: Any, device: str) -> LatentRuntimeSetup:
    values = OmegaConf.to_container(node, resolve=True) if OmegaConf.is_config(node) else dict(node)
    required = {"enabled", "mode", "checkpoint_path", "learning_config_path", "learning", "update_interval_steps",
                "max_cached_queries", "max_pending_windows", "max_trace_records", "pose", "resume_checkpoint_path", "save_checkpoint_path"}
    if not isinstance(values, dict) or set(values) != required or values["enabled"] is not True or values["mode"] != "latent_wm":
        raise ValueError("use path_following=latent_wm for the explicit latent runtime")
    if playback_cfg.action_mode != "policy" or int(playback_cfg.num_envs) != 1 or playback_cfg.keyboard:
        raise ValueError("latent path following requires one official policy environment without keyboard control")
    if OmegaConf.select(cfg, "training.task_name") != "G1WalkFlat" or OmegaConf.select(cfg, "training.sim_backend") != "mujoco":
        raise ValueError("latent path following requires task=g1_walk_flat/mujoco")
    for path in ("interactive.keyboard", "env.commands.heading_command", "demo.figure_eight.enabled", "online_adaptation.enabled"):
        if OmegaConf.select(cfg, path, default=False):
            raise ValueError(f"disable {path}; latent adaptation owns WM/A and keeps Tracker frozen")
    if OmegaConf.select(cfg, "env.commands.resampling_time", default=None) != 0:
        raise ValueError("latent path following requires an explicit fixed velocity command")
    root = Path(root_dir)

    def resolve(value: Any, *, optional: bool = False) -> Path | None:
        if value is None and optional:
            return None
        if not value:
            raise ValueError("provide the independent WM checkpoint and learning configuration")
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else root / path

    model_path = resolve(values["checkpoint_path"])
    if model_path is None:
        raise ValueError("provide a world-model checkpoint")
    loaded = load_world_model(model_path, device=device)
    with Path(checkpoint_path).open("rb") as handle:
        digest = _digest(handle)
    source, data = loaded.contract["source"], loaded.contract["data"]
    expected_source = {"checkpoint_sha256": digest, "source_profile": _canonical(asdict(profile)),
                       "preprocessing": _canonical(dict(spec.preprocessing)), "delta_frame": "fixed_straight_path",
                       "delta_units": ["m", "m", "rad"], "ego_frame": "previous_measurement_horizontal_heading"}
    for name, expected in expected_source.items():
        if _canonical(source.get(name)) != expected:
            raise ValueError(f"WM artifact is incompatible with actual {name}")
    expected_data = {"ctrl_dt": profile.ctrl_dt, "history_length": spec.history_length,
                     "prediction_horizon": spec.prediction_horizon, "observation_dim": spec.obs_dim, "action_dim": spec.action_dim}
    for name, expected in expected_data.items():
        if data.get(name) != expected:
            raise ValueError(f"WM artifact is incompatible with actual {name}")
    if data.get("command_dim", 7) != spec.command_dim or data.get("observation_contract", "official_compact_64_v1") != spec.observation_contract:
        raise ValueError("WM artifact has an incompatible command or observation identity")
    _positive_count(data["window_steps"], "window_steps")
    counts = {name: _positive_count(values[name], name) for name in
              ("update_interval_steps", "max_cached_queries", "max_pending_windows", "max_trace_records")}
    pose = PoseRuntimeConfig(**values["pose"])
    if pose.odometry_source == "cuvslam" and not pose.sensor.capture_initial_frame:
        raise ValueError("cuVSLAM requires initial-frame capture")
    learning_path = resolve(values["learning_config_path"])
    if learning_path is None:
        raise ValueError("provide a latent learning configuration")
    learning = OmegaConf.merge(OmegaConf.load(learning_path), OmegaConf.create(values["learning"]))
    raw = OmegaConf.to_container(learning, resolve=True)
    if not isinstance(raw, dict) or set(raw) != {"online", "costs"}:
        raise ValueError("latent learning configuration requires online and costs sections")
    save_path = resolve(values["save_checkpoint_path"], optional=True)
    resume_path = resolve(values["resume_checkpoint_path"], optional=True)
    if save_path is not None and save_path.exists() and save_path != resume_path:
        raise FileExistsError("online output already exists; select a new checkpoint path")
    return LatentRuntimeSetup(loaded, LatentOnlineConfig.from_mapping(raw["online"]), LatentCostConfig(**raw["costs"]),
                              PathFollowingRuntimeConfig(pose=pose, **counts), resume_path, save_path)


def _digest(handle: Any) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1 << 20), b""):
        digest.update(chunk)
    return digest.hexdigest()


class _ReceivingPoses:
    """Observe raw receive envelopes without changing source alignment ownership."""

    def __init__(self, source: Any, receive: Any) -> None:
        self.source, self.receive = source, receive

    @property
    def clock(self) -> Any:
        return self.source.clock

    def begin_step(self) -> Any:
        return self.source.begin_step()

    def complete_step(self, control: Any) -> Any:
        update = self.source.complete_step(control)
        self.receive(update, control.end_ns)
        return update

    def reset(self) -> None:
        self.source.reset()

    def close(self) -> None:
        self.source.close()


class LatentPathFollowingController(PathFollowingController[LatentPathFollowingUpdater]):
    """Reuse measured windows, with frozen risk inference on real ENV actions."""

    def __init__(self, inner: Any, playback: Any, env: Any, updater: LatentPathFollowingUpdater,
                 config: PathFollowingRuntimeConfig, *, pose_runtime: Any = None,
                 save_checkpoint_path: Path | None = None) -> None:
        self.ego_history = CausalEgoHistory(max_records=config.max_cached_queries)
        self.save_checkpoint_path = save_checkpoint_path
        self._risk_tilt_limit = float(updater.contract["source"]["risk_tilt_limit_rad"])
        if not math.isfinite(self._risk_tilt_limit) or not 0 < self._risk_tilt_limit <= math.pi:
            raise ValueError("WM risk tilt limit must be in (0, pi]")
        source = pose_runtime if pose_runtime is not None else create_pose_runtime(env, config.pose)
        super().__init__(inner, playback, env, updater, config, pose_runtime=_ReceivingPoses(source, self._receive_poses))
        self._residual_limits = self._zero.new_tensor(updater.config.residual_limit_rad)
        updater.ego_history = self.ego_history
        self._reset_diagnostics()

    def _reset_diagnostics(self) -> None:
        self._real_latent: LatentState | None = None
        self._risk_window: dict[str, Any] | None = None
        self._labelled_windows = self._unknown_windows = 0
        self._low_risk_windows = self._low_risk_events = 0
        self._correlation_count = 0
        self._risk_mean = self._tilt_mean = 0.0
        self._risk_m2 = self._tilt_m2 = self._risk_tilt_covariance = 0.0
        self._last_diagnostic_update: Any = None
        self._budget_updates = self._budget_overruns = 0
        self._max_update_elapsed_s = 0.0
        self._diagnostic_steps = 0
        self._saturation_sum = self._peak_residual_fraction = 0.0

    def reset(self, done: Any = None) -> None:
        if done is not None and not bool(torch.as_tensor(done).any()):
            return
        self.ego_history.clear()
        self.updater.discard_pending()
        super().reset(done)
        self._reset_diagnostics()

    def _receive_poses(self, update: Any, now_ns: int) -> None:
        self.ego_history.add(update.received_poses)
        record = self.updater.records.get(self._steps)
        if record is not None:
            ego, valid, age = self.ego_history.at(now_ns)
            record.update(next_ego_motion=ego, next_ego_valid=valid, next_ego_age_s=age)

    def _select_candidate(self, now_ns: int) -> torch.Tensor:
        if not self.collector.window.at_boundary:
            return self._candidate
        decision = self._fresh_decision(now_ns)
        if not self.updater.calibration_complete or decision is None:
            return self._zero
        candidate = self.updater.candidate(decision).to(self._zero)
        if not bool(torch.isfinite(candidate).all()):
            raise ValueError("non-finite latent A candidate")
        return candidate

    @torch.no_grad()
    def _initialize_real_latent(self, query: Any, timestamp_ns: int) -> None:
        history = query.observation_history.shape[1]
        ego = np.zeros((history, 3), dtype=np.float32)
        valid = np.zeros((history, 1), dtype=bool)
        age = np.zeros((history, 1), dtype=np.float32)
        dt_ns = round(self.updater.ctrl_dt * 1e9)
        for index in range(history):
            timestamp = timestamp_ns - (history - 1 - index) * dt_ns
            if timestamp >= 0 and bool(query.history_valid_mask[0, index]):
                ego[index], valid[index, 0], age[index, 0] = self.ego_history.at(timestamp)
        reference = query.command
        start = self.updater.rollout.encode_start(
            query, errors=reference.new_zeros((1, 2)), previous_residual=self._locked,
            ego_motion=torch.as_tensor(ego, device=reference.device, dtype=reference.dtype).unsqueeze(0),
            ego_valid=torch.as_tensor(valid, device=reference.device).unsqueeze(0),
            ego_age_s=torch.as_tensor(age, device=reference.device, dtype=reference.dtype).unsqueeze(0),
        )
        self._real_latent = start.latent.detach()

    @torch.no_grad()
    def _advance_real_latent(self, action: torch.Tensor, observation: torch.Tensor, timestamp_ns: int) -> None:
        if self._real_latent is None:
            raise RuntimeError("real WM posterior requires its current observed state")
        ego, valid, age = self.ego_history.at(timestamp_ns)
        self._real_latent = self.updater.model.posterior(
            self._real_latent, action, observation,
            ego_motion=torch.as_tensor(ego, device=action.device, dtype=action.dtype).reshape(1, 3),
            ego_valid=torch.tensor([[valid]], device=action.device, dtype=torch.bool),
            ego_age_s=action.new_tensor([[age]]),
        )[0].detach()

    def _summarize_risk_window(self, *, terminal: bool) -> dict[str, Any]:
        window = self._risk_window
        if window is None:
            raise RuntimeError("risk window summary requires a real transition")
        # An observed positive event is decisive even if another label is absent.
        labelled = window["event"] or not window["unknown"]
        low_risk = window["max_probability"] < self.updater.config.risk_threshold
        if labelled:
            self._labelled_windows += 1
            if low_risk:
                self._low_risk_windows += 1
                self._low_risk_events += int(window["event"])
            if window["max_tilt_rad"] is not None and not window["missing_tilt"]:
                probability, tilt = window["max_probability"], window["max_tilt_rad"]
                self._correlation_count += 1
                dx, dy = probability - self._risk_mean, tilt - self._tilt_mean
                self._risk_mean += dx / self._correlation_count
                self._tilt_mean += dy / self._correlation_count
                self._risk_m2 += dx * (probability - self._risk_mean)
                self._tilt_m2 += dy * (tilt - self._tilt_mean)
                self._risk_tilt_covariance += dx * (tilt - self._tilt_mean)
        else:
            self._unknown_windows += 1
        summary = {
            "event": "risk_window", "step": self._steps, "window_id": window["window_id"],
            "executed_steps": window["steps"], "terminal": terminal,
            "max_risk_probability": window["max_probability"], "max_tilt_rad": window["max_tilt_rad"],
            "risk_label_valid": bool(labelled), "risk_event": bool(window["event"]) if labelled else None,
            "low_risk": bool(low_risk), "risk_threshold": self.updater.config.risk_threshold,
        }
        self._risk_window = None
        return summary

    def risk_diagnostics(self) -> dict[str, Any]:
        """Executed-window calibration; partial live windows are not labels.

        Correlation pairs window maxima only when every executed tilt is known.
        No next-state observation is inferred from an autoreset observation.
        """
        correlation = None
        if self._correlation_count >= 2 and self._risk_m2 > 0 and self._tilt_m2 > 0:
            correlation = float(np.clip(
                self._risk_tilt_covariance / math.sqrt(self._risk_m2 * self._tilt_m2), -1., 1.,
            ))
        return {
            "labelled_windows": self._labelled_windows, "unknown_windows": self._unknown_windows,
            "low_risk_windows": self._low_risk_windows, "low_risk_events": self._low_risk_events,
            "low_risk_event_rate": None if not self._low_risk_windows else self._low_risk_events / self._low_risk_windows,
            "risk_tilt_correlation": correlation,
            "update_count": self._budget_updates, "update_budget_overruns": self._budget_overruns,
            "max_update_elapsed_s": self._max_update_elapsed_s if self._budget_updates else None,
            "mean_residual_saturation_fraction": self._saturation_sum / self._diagnostic_steps if self._diagnostic_steps else None,
            "peak_residual_limit_fraction": self._peak_residual_fraction if self._diagnostic_steps else None,
        }

    def _record_risk_step(self, *, probability: float, tilt: float | None, event: bool,
                          label_valid: bool, terminal: bool, window_id: int) -> dict[str, Any]:
        fraction = self._locked.abs() / self._residual_limits
        saturation = float((fraction >= .99).float().mean())
        peak = float(fraction.max())
        self._diagnostic_steps += 1
        self._saturation_sum += saturation
        self._peak_residual_fraction = max(self._peak_residual_fraction, peak)
        budget = self.updater.window_steps * self.updater.ctrl_dt
        elapsed = None
        if self.last_update is not None and self.last_update is not self._last_diagnostic_update:
            self._last_diagnostic_update = self.last_update
            elapsed = self.last_update.elapsed_s
            self._budget_updates += 1
            self._budget_overruns += int(elapsed > budget)
            self._max_update_elapsed_s = max(self._max_update_elapsed_s, elapsed)
        fields = {
            "risk_probability": probability, "risk_tilt_rad": tilt,
            "risk_label_valid": label_valid, "risk_event": event if label_valid else None,
            "risk_tilt_limit_rad": self._risk_tilt_limit, "risk_threshold": self.updater.config.risk_threshold,
            "online_update_elapsed_s": elapsed, "update_budget_s": budget,
            "update_within_budget": None if elapsed is None else elapsed <= budget,
            "residual_saturation_fraction": saturation, "residual_peak_limit_fraction": peak,
        }
        self.trace[-1].update(fields)
        if self._risk_window is None:
            self._risk_window = {
                "window_id": window_id, "steps": 0, "max_probability": 0.,
                "max_tilt_rad": None, "event": False, "unknown": False, "missing_tilt": False,
            }
        window = self._risk_window
        if window["window_id"] != window_id:
            raise RuntimeError("risk diagnostics crossed an uncommitted residual window")
        window["steps"] += 1
        window["max_probability"] = max(window["max_probability"], probability)
        window["event"] |= event
        window["unknown"] |= not label_valid
        window["missing_tilt"] |= tilt is None
        if tilt is not None:
            window["max_tilt_rad"] = tilt if window["max_tilt_rad"] is None else max(window["max_tilt_rad"], tilt)
        if terminal or self.collector.window.at_boundary:
            self.trace.append(self._summarize_risk_window(terminal=terminal))
        return fields

    def on_env_step(self, *, executed_action: Any, next_observation: Any, done: Any = None) -> dict[str, float | int]:
        if self._closed or self._failed or self._terminal or self._before is None:
            raise RuntimeError("no live path-following query to complete")
        try:
            terminal = done is not None and bool(torch.as_tensor(done).any())
            before, query = self._queries[self._steps]
            after = self.poses.clock
            window_id = self.collector.window.window_id
            action = torch.as_tensor(executed_action, device=query.command.device, dtype=query.command.dtype).detach()
            if action.shape != (1, 29) or not bool(torch.isfinite(action).all()):
                raise ValueError("WM records require finite actual env-input commands [1,29]")
            if self._real_latent is None:
                self._initialize_real_latent(query, before.timestamp_ns)
            with torch.no_grad():
                probability = float(self.updater.model.risk(self._real_latent, action).sigmoid().item())
            if not math.isfinite(probability):
                raise ValueError("non-finite real-transition risk prediction")
            state = self.env.state
            terminated = bool(torch.as_tensor(state.terminated).any())
            truncated = bool(torch.as_tensor(state.truncated).any())
            same_episode = after.sequence_id == before.sequence_id and after.timestamp_ns > before.timestamp_ns
            # Cleared terminal flags identify an autoreset observation, not truth.
            next_is_real = same_episode and (not terminal or terminated or truncated)
            compact = None
            tilt = None
            if next_is_real and next_observation is not None:
                compact = self.playback.project_observation(next_observation).detach()
                if compact.shape != (1, self.updater.model.config.obs_dim):
                    raise ValueError("real WM posterior observation differs from its execution contract")
                if not bool(torch.isfinite(compact).all()):
                    if not terminal:
                        raise ValueError("real WM posterior requires finite proprioception")
                    compact = None
                if compact is not None:
                    try:
                        tilt = proprio_tilt_rad(compact[0].cpu().numpy(), observation_contract=self.playback.spec.observation_contract)
                    except ValueError:
                        if not terminal:
                            raise
                        compact = None
            event = terminated or (tilt is not None and tilt > self._risk_tilt_limit)
            label_valid = terminated or tilt is not None
            if terminal and not (terminated or truncated):
                event, label_valid = False, False
            if not terminal:
                if compact is None:
                    raise ValueError("nonterminal WM transition requires its real next proprioception")
                source_action = self.playback.last_source_action
                if source_action is None:
                    raise RuntimeError("WM records require the actual submitted SOURCE command")
                source_action = torch.as_tensor(source_action).detach()
                if source_action.shape != (1, 29) or not bool(torch.isfinite(source_action).all()):
                    raise ValueError("WM records require finite actual SOURCE commands [1,29]")
                ego, valid, age = self.ego_history.at(before.timestamp_ns)
                record = {
                    "observation": query.observation_history[0, -1].detach().cpu().numpy().copy(),
                    "next_observation": compact[0].cpu().numpy().copy(),
                    "action": action[0].cpu().numpy().copy(),
                    "source_action": source_action[0].cpu().numpy().copy(),
                    "command": query.command[0].detach().cpu().numpy().copy(),
                    "ego_motion": ego, "ego_valid": valid, "ego_age_s": age,
                    "next_ego_motion": ego.copy(), "next_ego_valid": valid,
                    "next_ego_age_s": age + (after.timestamp_ns - before.timestamp_ns) / 1e9,
                    "risk": float(event), "risk_valid": label_valid,
                }
                self.updater.record_transition(self._steps, record, capacity=self.config.max_cached_queries)
                self.updater.previous_residual = self._locked.detach().clone()
            stats = super().on_env_step(executed_action=executed_action, next_observation=next_observation, done=done)
            if compact is not None:
                self._advance_real_latent(action, compact, after.timestamp_ns)
            fields = self._record_risk_step(probability=probability, tilt=tilt, event=event,
                                            label_valid=label_valid, terminal=terminal, window_id=window_id)
            stats["path_risk_probability"] = probability
            stats["path_risk_label_valid"] = int(label_valid)
            stats["path_residual_saturation_fraction"] = fields["residual_saturation_fraction"]
            if tilt is not None:
                stats["path_risk_tilt_rad"] = tilt
            if label_valid:
                stats["path_risk_event"] = int(event)
            if fields["online_update_elapsed_s"] is not None:
                stats["path_update_elapsed_s"] = fields["online_update_elapsed_s"]
                stats["path_update_budget_s"] = fields["update_budget_s"]
                stats["path_update_within_budget"] = int(fields["update_within_budget"])
            if terminal:
                self.updater.discard_pending()
                self._real_latent = None
            return stats
        except Exception as error:
            if not self._failed:
                self._fail(str(error))
            raise

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self.save_checkpoint_path is not None and not self._failed:
                self.updater.save(self.save_checkpoint_path)
        finally:
            self._real_latent = None
            super().close()
