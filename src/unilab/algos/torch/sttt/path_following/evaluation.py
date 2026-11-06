"""Finite same-condition evaluation; truth is a score channel, never an estimator input."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.sttt.geometry import quat_to_mat
from unilab.algos.torch.sttt.path_following.contract import _integer
from unilab.algos.torch.sttt.path_following.evaluation_metrics import PathFollowingMetrics, PathFollowingThresholds
from unilab.algos.torch.sttt.path_following.reference import FixedStraightPath
from unilab.algos.torch.sttt.path_following.runtime import PathFollowingController
from unilab.algos.torch.sttt.pose_data import TimedPelvisPose
from unilab.visualization.playback_distill_sessions import create_fada_playback_session
from unilab.visualization.playback_sessions import RslRlPlaybackConfig

_MODES = {"nominal", "oat", "oat_path"}
_CONDITIONS = {"clean", "delayed_payload"}
_HEADER = (
    "timestamp_ns", "elapsed_s", "completed_step", "valid", "x_m", "y_m", "z_m",
    "ey_m", "epsi_rad", "forward_position_m", "payload_active", "payload_attached_this_step",
    "payload_event_start_ns", "payload_added_mass_kg",
)


def _plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True, enum_to_str=True)
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _publish(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_plain(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _file_identity(path: str | Path) -> dict[str, str]:
    path = Path(path).expanduser().resolve()
    with path.open("rb") as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def _truth(env: Any) -> TimedPelvisPose:
    timestamp = _integer(env.get_sim_time_ns(), "evaluation physical timestamp")
    position = np.asarray(env.get_base_pos(), dtype=np.float64)
    quaternion = np.asarray(env.get_base_quat(), dtype=np.float64)
    if position.shape != (1, 3) or quaternion.shape != (1, 4) or not np.isfinite(position).all() or not np.isfinite(quaternion).all() or not np.allclose(np.linalg.norm(quaternion, axis=1), 1., atol=1e-5, rtol=0):
        raise ValueError("evaluation requires finite single-env truth and unit wxyz quaternions")
    matrix = np.eye(4)
    matrix[:3, :3] = quat_to_mat(torch.from_numpy(quaternion))[0].numpy()
    matrix[:3, 3] = position[0]
    # This identity belongs only to the non-reset evaluation episode, not SLAM.
    return TimedPelvisPose(0, 0, timestamp, matrix, True, (timestamp, timestamp), "evaluation_world")


def _payload(info: Mapping[str, Any]) -> tuple[bool, bool, float]:
    names = ("runtime_payload_active", "runtime_payload_attached_this_step", "runtime_payload_added_mass_kg")
    if any(name not in info for name in names):
        raise ValueError("evaluation requires actual runtime payload truth fields, even when disabled")
    active, event, mass = (np.asarray(info[name]) for name in names)
    if active.shape != (1,) or event.shape != (1,) or mass.shape != (1,) or active.dtype != np.bool_ or event.dtype != np.bool_ or not np.isfinite(mass).all() or mass[0] < 0:
        raise ValueError("invalid per-environment payload truth")
    if bool(event[0]) and (not bool(active[0]) or mass[0] <= 0):
        raise ValueError("payload attachment event lacks active positive added mass")
    return bool(active[0]), bool(event[0]), float(mass[0])


class _RuntimeEvidence:
    """Keep arrivals, but count each represented measurement only once."""
    def __init__(self, handle: Any) -> None:
        self.handle = handle
        self.last_timestamp: dict[int, int] = {}
        self.records = self.steps = self.covered_steps = self.unique = 0
        self.age_sum = self.age_max = 0
        self.latest: dict[str, Any] = {}

    def drain(self, controller: Any, arrival_ns: int | None) -> None:
        if not isinstance(controller, PathFollowingController):
            return
        for record in controller.drain_trace():
            row = dict(record)
            row["arrival_timestamp_ns"] = arrival_ns
            new_measurement = False
            if record["event"] == "step":
                self.steps += 1
                timestamp, sequence = record["measurement_timestamp_ns"], record["sequence_id"]
                if timestamp is not None:
                    self.covered_steps += 1
                    if timestamp > self.last_timestamp.get(sequence, -1):
                        self.last_timestamp[sequence] = timestamp
                        self.unique += 1
                        age = int(record["observation_age_ns"])
                        self.age_sum += age
                        self.age_max = max(self.age_max, age)
                        new_measurement = True
                self.latest = record
            row["new_measurement"] = new_measurement
            self.handle.write(json.dumps(_plain(row), allow_nan=False) + "\n")
            self.records += 1

    def summary(self, enabled: bool) -> dict[str, Any]:
        return {
            "enabled": enabled, "records": self.records, "control_records": self.steps,
            "paired_control_fraction": self.covered_steps / self.steps if self.steps else None,
            "unique_measurements": self.unique,
            "unique_mean_age_s": self.age_sum / self.unique / 1e9 if self.unique else None,
            "unique_max_age_s": self.age_max / 1e9 if self.unique else None,
            "a_updates": self.latest.get("a_updates"), "b_updates": self.latest.get("b_updates"),
            "dropped_windows": self.latest.get("dropped_windows", {}),
            "expired_queries": self.latest.get("expired_queries"),
        }


def _run_episode(
    playback_cfg: DictConfig, *, mode: str, condition: str, seed: int, duration_s: float,
    thresholds: PathFollowingThresholds, output: Path, root: Path, device: str,
    playback_deps: Mapping[str, Any] | None,
) -> dict[str, Any]:
    output.mkdir()
    source: dict[str, Any] = {
        "format": "sttt_path_following_evaluation", "schema_version": 1,
        "mode": mode, "condition": condition, "seed": seed, "requested_duration_s": duration_s,
        "thresholds": asdict(thresholds), "playback_config": _plain(playback_cfg),
        "evaluation_pose_source": "simulation_ground_truth_score_only",
        "controller_pose_source": OmegaConf.select(playback_cfg, "path_following.pose.odometry_source", default=None),
        "policy": _file_identity(playback_cfg.training.play_checkpoint_path),
        "profile": _file_identity(playback_cfg.official_playback.profile_path),
        "response": _file_identity(playback_cfg.path_following.checkpoint_path) if mode == "oat_path" else None,
    }
    _publish(output / "source.json", source)
    metrics = PathFollowingMetrics(thresholds)
    post_payload: PathFollowingMetrics | None = None
    session = None
    reference = None
    initial_timestamp: int | None = None
    previous_timestamp: int | None = None
    previous_errors: np.ndarray | None = None
    previous_progress: float | None = None
    attach_time_ns: int | None = None
    completed_steps = 0
    reason, failure, close_failure = "error", None, None
    evidence = None
    with (output / "trajectory.csv").open("w", newline="", buffering=1) as trajectory, (output / "runtime.jsonl").open("w", buffering=1) as runtime_file:
        writer = csv.DictWriter(trajectory, fieldnames=_HEADER)
        writer.writeheader()
        evidence = _RuntimeEvidence(runtime_file)

        def observe_truth(*, completed: bool, interval_start_ns: int | None = None) -> None:
            nonlocal previous_timestamp, previous_errors, previous_progress, attach_time_ns, post_payload
            assert session is not None and reference is not None and initial_timestamp is not None
            try:
                pose = _truth(session.env)
            except Exception:
                timestamp = _integer(session.env.get_sim_time_ns(), "missing-truth timestamp")
                if previous_timestamp is not None and timestamp > previous_timestamp:
                    metrics.observe(timestamp, None, None)
                    if post_payload is not None:
                        post_payload.observe(timestamp, None, None)
                    writer.writerow({"timestamp_ns": timestamp, "elapsed_s": (timestamp - initial_timestamp) / 1e9,
                                     "completed_step": completed, "valid": False})
                    previous_timestamp, previous_errors, previous_progress = timestamp, None, None
                raise
            if previous_timestamp is not None and pose.timestamp_ns <= previous_timestamp:
                raise ValueError("evaluation physical time failed to advance; resetting cannot extend an episode")
            errors = reference.errors(pose)
            assert pose.navigation_from_pelvis is not None
            dx, dy = pose.navigation_from_pelvis[:2, 3] - np.asarray(reference.origin_xy)
            progress = float(math.cos(reference.heading_rad) * dx + math.sin(reference.heading_rad) * dy)
            active, attached, mass = _payload(session.info)
            event_time = None
            if attached:
                if interval_start_ns is None or attach_time_ns is not None:
                    raise ValueError("payload event must belong to exactly one executed interval")
                event_time = interval_start_ns
                attach_time_ns = event_time
                post_payload = PathFollowingMetrics(thresholds)
                assert previous_errors is not None and previous_progress is not None
                post_payload.observe(event_time, previous_errors, previous_progress)
            metrics.observe(pose.timestamp_ns, errors, progress)
            if post_payload is not None:
                post_payload.observe(pose.timestamp_ns, errors, progress)
            writer.writerow({
                "timestamp_ns": pose.timestamp_ns, "elapsed_s": (pose.timestamp_ns - initial_timestamp) / 1e9,
                "completed_step": completed, "valid": True,
                "x_m": float(pose.navigation_from_pelvis[0, 3]), "y_m": float(pose.navigation_from_pelvis[1, 3]), "z_m": float(pose.navigation_from_pelvis[2, 3]),
                "ey_m": float(errors[0]), "epsi_rad": float(errors[1]), "forward_position_m": progress,
                "payload_active": active, "payload_attached_this_step": attached,
                "payload_event_start_ns": event_time, "payload_added_mass_kg": mass,
            })
            previous_timestamp, previous_errors, previous_progress = pose.timestamp_ns, errors, progress
            evidence.drain(session.controller, pose.timestamp_ns)

        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            playback = RslRlPlaybackConfig(
                task="G1WalkFlat", load_run="-1", checkpoint=None, action_mode="policy", policy_obs_mode="actor",
                algo_log_name="distill", log_root=None, num_envs=1,
                checkpoint_path=str(playback_cfg.training.play_checkpoint_path), keyboard=False,
            )
            session, _, resolved = create_fada_playback_session(
                playback_cfg=playback, cfg=playback_cfg, root_dir=root, device=device,
                deps=playback_deps, log=lambda message: None,
            )
            session.set_autoreset(False)
            session.reset()
            initial = _truth(session.env)
            reference = FixedStraightPath.from_initial_pose(initial)
            initial_timestamp = initial.timestamp_ns
            dt = float(session.env.cfg.ctrl_dt)
            steps = round(duration_s / dt)
            if not math.isfinite(dt) or dt <= 0 or steps < 1 or not math.isclose(steps * dt, duration_s, rel_tol=0, abs_tol=1e-9):
                raise ValueError("evaluation duration must be a positive integer number of actual control periods")
            if resolved is None:
                raise ValueError("evaluation requires the actual resolved policy checkpoint")
            source["actual_policy"] = _file_identity(resolved)
            if source["actual_policy"]["sha256"] != source["policy"]["sha256"]:
                raise ValueError("resolved policy differs from the requested evaluation checkpoint")
            source["reference"] = asdict(reference)
            source["ctrl_dt"] = dt
            source["effective_conditions"] = {
                name: _plain(getattr(session.env.cfg, name))
                for name in ("commands", "runtime_payload", "domain_rand", "noise_config")
            }
            _publish(output / "source.json", source)
            observe_truth(completed=False)
            for _ in range(steps):
                start_ns = previous_timestamp
                session.step_once()
                completed_steps += 1
                observe_truth(completed=True, interval_start_ns=start_ns)
                state = session.env.state
                if bool(np.any(state.terminated)):
                    reason = "terminated"
                    break
                if bool(np.any(state.truncated)):
                    reason = "truncated"
                    break
            else:
                reason = "horizon"
        except (Exception, KeyboardInterrupt) as error:
            failure = {"type": type(error).__name__, "message": str(error)}
            reason = "error"
            if session is not None:
                # A post-step adaptation failure may still have advanced physics.
                # Preserve that final truth separately from a completed control step.
                try:
                    now = session.env.get_sim_time_ns()
                    if previous_timestamp is not None and now > previous_timestamp:
                        observe_truth(completed=False, interval_start_ns=previous_timestamp)
                    evidence.drain(session.controller, int(now))
                except Exception as evidence_error:
                    failure["final_evidence_error"] = str(evidence_error)
        finally:
            if session is not None:
                try:
                    session.close()
                except Exception as error:
                    close_failure = {"type": type(error).__name__, "message": str(error)}
                    reason = "error"
    metric_summary = metrics.summary()
    post_summary = post_payload.summary() if post_payload is not None else None
    condition_observed = attach_time_ns is not None if condition == "delayed_payload" else attach_time_ns is None
    complete = reason == "horizon" and failure is None and close_failure is None and condition_observed
    tracking_ok = metric_summary["tracking_state"] in ("maintained", "recovered")
    if condition == "delayed_payload":
        tracking_ok = tracking_ok and post_summary is not None and post_summary["tracking_state"] in ("maintained", "recovered")
    summary = {
        "mode": mode, "condition": condition, "seed": seed, "complete": complete,
        "horizon_reached": reason == "horizon",
        "stop_reason": reason, "error": failure, "close_error": close_failure,
        "completed_steps": completed_steps, "metrics": metric_summary, "post_payload_metrics": post_summary,
        "actual_payload_event_start_ns": attach_time_ns,
        "condition_observed": condition_observed,
        "tracking_success": complete and condition_observed and tracking_ok and metric_summary["progress_pass"],
        "runtime": evidence.summary(mode == "oat_path"),
    }
    _publish(output / "summary.json", summary)
    return summary


def run_path_following_evaluation(
    cfg: DictConfig, *, root_dir: str | Path | None = None, playback_deps: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run only the explicitly listed modes/conditions/seeds. Never called by smoke unless I/O is injected."""
    if not isinstance(cfg, DictConfig):
        raise TypeError("evaluation requires the independent Hydra configuration")
    expected = {"output_dir", "policy_checkpoint", "profile_path", "response_checkpoint", "pose_source", "device", "modes", "conditions", "seeds", "duration_s", "source_overrides", "thresholds"}
    if set(cfg) != expected:
        raise ValueError("use conf/sttt/path_following_evaluate.yaml")
    raw_thresholds = OmegaConf.to_container(cfg.thresholds, resolve=True)
    if not isinstance(raw_thresholds, dict) or set(raw_thresholds) != set(PathFollowingThresholds.__dataclass_fields__):
        raise ValueError("provide all five explicit physical evaluation thresholds")
    thresholds = PathFollowingThresholds(
        lateral_tolerance_m=raw_thresholds["lateral_tolerance_m"], heading_tolerance_rad=raw_thresholds["heading_tolerance_rad"],
        dwell_s=raw_thresholds["dwell_s"], crossing_deadband_m=raw_thresholds["crossing_deadband_m"],
        minimum_forward_progress_m=raw_thresholds["minimum_forward_progress_m"],
    )
    duration = cfg.duration_s
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration_s must be an explicit positive finite physical horizon")
    modes, conditions, seeds = list(cfg.modes), list(cfg.conditions), list(cfg.seeds)
    if not modes or not conditions or not seeds or len(set(modes)) != len(modes) or len(set(conditions)) != len(conditions) or len(set(seeds)) != len(seeds) or not set(modes) <= _MODES or not set(conditions) <= _CONDITIONS:
        raise ValueError("provide explicit unique modes, conditions and seeds; no automatic sweep is selected")
    for seed in seeds:
        if _integer(seed, "seed") >= 2**32:
            raise ValueError("evaluation seed must be less than 2**32")
    if cfg.pose_source not in ("simulation", "cuvslam"):
        raise ValueError("pose_source must explicitly select simulation or cuvslam")
    root = Path(root_dir) if root_dir is not None else Path(__file__).resolve().parents[6]
    def resolve(value: Any) -> Path:
        if not value:
            raise ValueError("output, policy and source-profile paths must be explicit; oat_path also requires B")
        path = Path(str(value)).expanduser()
        return (path if path.is_absolute() else root / path).resolve()
    policy, profile = resolve(cfg.policy_checkpoint), resolve(cfg.profile_path)
    _file_identity(policy)
    _file_identity(profile)
    response = resolve(cfg.response_checkpoint) if "oat_path" in modes else None
    if response is not None:
        _file_identity(response)
    output = resolve(cfg.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"evaluation output must be empty: {output}")
    cases = []
    with initialize_config_dir(version_base="1.3", config_dir=str(root / "conf/distill")):
        for condition in conditions:
            for mode in modes:
                overrides = ["task=g1_walk_flat/mujoco", "recording_profile=straight_payload", *list(cfg.source_overrides)]
                overrides.append("path_following=online" if mode == "oat_path" else "~path_following")
                playback_cfg = compose(config_name="config", overrides=overrides)
                OmegaConf.update(playback_cfg, "training.play_checkpoint_path", str(policy))
                OmegaConf.update(playback_cfg, "official_playback.profile_path", str(profile))
                OmegaConf.update(playback_cfg, "online_adaptation.enabled", mode != "nominal")
                OmegaConf.update(playback_cfg, "env.runtime_payload.enabled", condition == "delayed_payload")
                if mode == "oat_path":
                    OmegaConf.update(playback_cfg, "path_following.checkpoint_path", str(response))
                    OmegaConf.update(playback_cfg, "path_following.pose.odometry_source", cfg.pose_source)
                if playback_cfg.training.task_name != "G1WalkFlat" or playback_cfg.training.sim_backend != "mujoco":
                    raise ValueError("evaluation is limited to g1_walk_flat/mujoco")
                if OmegaConf.select(playback_cfg, "interactive.keyboard") or OmegaConf.select(playback_cfg, "demo.figure_eight.enabled") or OmegaConf.select(playback_cfg, "env.commands.heading_command") or OmegaConf.select(playback_cfg, "env.commands.resampling_time") != 0:
                    raise ValueError("all evaluation modes require fixed-command playback without heading/keyboard/demo control")
                limit = np.asarray(playback_cfg.env.commands.vel_limit, dtype=float)
                if limit.shape != (2, 3) or not np.isfinite(limit).all() or not np.array_equal(limit[0], limit[1]) or limit[0, 0] <= 0 or not np.array_equal(limit[0, 1:], [0., 0.]):
                    raise ValueError("all evaluation modes require the same deterministic forward velocity limits")
                for seed in seeds:
                    seeded_cfg = OmegaConf.create(OmegaConf.to_container(playback_cfg, resolve=True))
                    OmegaConf.update(seeded_cfg, "algo.seed", seed)
                    cases.append((mode, condition, seed, seeded_cfg))
    output.mkdir(parents=True, exist_ok=True)
    aggregate: dict[str, Any] = {"format": "sttt_path_following_evaluation", "schema_version": 1,
                                 "complete": False, "requested_cases": len(cases), "runs": [], "evaluation_config": _plain(cfg)}
    _publish(output / "summary.json", aggregate)
    for mode, condition, seed, playback_cfg in cases:
        name = f"{mode}__{condition}__seed_{seed}"
        result = _run_episode(playback_cfg, mode=mode, condition=condition, seed=seed, duration_s=float(duration),
                              thresholds=thresholds, output=output / name, root=root, device=str(cfg.device), playback_deps=playback_deps)
        aggregate["runs"].append({"directory": name, **result})
        _publish(output / "summary.json", aggregate)
        if result["stop_reason"] == "error":
            break  # No automatic retry or parameter search after a failed case.
    aggregate["complete"] = len(aggregate["runs"]) == len(cases) and all(run["complete"] for run in aggregate["runs"])
    _publish(output / "summary.json", aggregate)
    return aggregate
