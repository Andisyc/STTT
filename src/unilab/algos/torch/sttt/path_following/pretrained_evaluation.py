"""Same-condition, learning-free evaluation of one actor on multiple references."""
from __future__ import annotations

import json
import random
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from unilab.algos.torch.sttt.path_following.actor_checkpoint import load_actor_checkpoint
from unilab.algos.torch.sttt.path_following.trajectory import MeasuredTrajectory, ReferenceTrajectory, TrajectoryConfig
from unilab.algos.torch.sttt.pose_data import TimedPelvisPose
from unilab.algos.torch.sttt.world_model.checkpoint import load_world_model
from unilab.algos.torch.sttt.world_model.sequence_collection import collect_world_model_sequences
from unilab.algos.torch.sttt.world_model.motion import motion_prefix
from unilab.algos.torch.sttt.world_model.sequence_collection import proprio_tilt_rad
from unilab.algos.torch.sttt.world_model.sequence_data import iter_sequences


def score_capture(directory: Path, trajectory: TrajectoryConfig, requested_steps: int) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    metadata = json.loads((directory / "capture.json").read_text())
    receipts = {item["timestamp_ns"]: item for item in metadata["pose_receipts"] if item["valid"]}
    if len(receipts) < 2:
        raise ValueError("evaluation requires physical start/end poses")
    reference = None
    errors, progresses, times = [], [], []
    initial_matrix = None
    speeds = []
    geometry = None
    if metadata.get("reference_geometry"):
        filename = metadata["reference_geometry"]
        if Path(filename).name != filename:
            raise ValueError("reference geometry must remain inside its capture directory")
        with np.load(directory / filename, allow_pickle=False) as archive:
            geometry = ReferenceTrajectory(torch.from_numpy(archive["xy"].copy()).double()[None],
                torch.from_numpy(archive["speed"].copy()).double()[None], trajectory,
                tuple(map(tuple, archive["command_bounds"].tolist())) if "command_bounds" in archive else None)
    for timestamp, item in sorted(receipts.items()):
        pose = TimedPelvisPose(item["sequence_id"], item["tracking_epoch"], timestamp,
                              np.asarray(item["navigation_from_pelvis"]), True,
                              tuple(item["source_timestamps_ns"]), item["frame_id"])
        if reference is None:
            reference = MeasuredTrajectory(pose, trajectory, dtype=torch.float64, reference=geometry)
            initial_matrix = np.asarray(item["navigation_from_pelvis"])
        state = reference.state_at(pose)
        errors.append(state.errors()[0].numpy())
        progresses.append(float(state.progress[0]))
        times.append(timestamp)
        speeds.append(float(state.reference.sample(state.progress[:, None])[2][0, 0]))
    sequences = list(iter_sequences(directory))
    observations = sequences[0]["observations"][0].numpy()
    executed = sum(len(sequence["actions"]) for sequence in sequences)
    risk_events = sum(int((sequence["risk"][:, 0] > .5).sum()) for sequence in sequences)
    duration = (times[-1] - times[0]) / 1e9
    if duration <= 0 or initial_matrix is None:
        raise ValueError("evaluation physical time did not advance")
    errors = np.asarray(errors)
    progress = progresses[-1] - progresses[0]
    expected = float(np.sum(np.asarray(speeds[:-1]) * np.diff(times) / 1e9))
    if expected <= 0:
        raise ValueError("evaluation requires positive nominal forward progress")
    return {"lateral_rmse_m": float(np.sqrt(np.mean(errors[:, 0] ** 2))),
            "lateral_max_abs_m": float(np.max(np.abs(errors[:, 0]))),
            "final_lateral_m": float(errors[-1, 0]), "forward_progress_m": progress,
            "forward_speed_retention": progress / expected,
            "duration_s": duration, "executed_steps": executed,
            "horizon_reached": executed == requested_steps, "risk_event_fraction": risk_events / executed,
            "failure_event": bool(risk_events), "end_reason": metadata.get("end_reason")}, observations, initial_matrix


def score_forecasts(directory: Path, *, risk_threshold: float) -> list[dict[str, Any]]:
    """Compare each forecast lead against real windows; censored safe tails are excluded."""
    if not 0 < risk_threshold < 1:
        raise ValueError("risk threshold must lie strictly between zero and one")
    metadata = json.loads((directory / "capture.json").read_text())
    forecasts = metadata.get("rollout_forecasts", [])
    if not forecasts:
        return []
    sequences = list(iter_sequences(directory))
    if len(sequences) != 1:
        raise ValueError("forecast scoring requires one physical episode")
    sequence = sequences[0]
    manifest = json.loads((directory / "metadata.json").read_text())
    contract = manifest["contract"]["data"]
    window = contract["window_steps"]
    size = len(sequence["actions"])
    tilt = np.asarray([proprio_tilt_rad(value.numpy(), observation_contract=contract["observation_contract"]) for value in sequence["observations"]])
    groups: dict[int, dict[str, list]] = {}
    for forecast in forecasts:
        risk = np.asarray(forecast["risk"], dtype=np.float64)
        motion = torch.tensor(forecast["body_delta"], dtype=torch.float64)
        actions = np.asarray(forecast["actions"], dtype=np.float64)
        for offset in range(0, len(risk), window):
            start = forecast["origin_step"] + offset
            end = min(start + window, size)
            if start >= size:
                continue
            event = bool((sequence["risk"][start:end, 0] > .5).any())
            if end - start < window and not event:
                continue
            lead = offset // window + 1
            group = groups.setdefault(lead, {"scores": [], "events": [], "tilts": [], "motion": [], "actions": []})
            group["scores"].append(float(risk[offset:offset + window].max()))
            group["events"].append(event)
            group["tilts"].append(float(tilt[start + 1:end + 1].max()))
            if end - start == window and bool(sequence["motion_valid"][start:end].all()):
                predicted = motion_prefix(motion[offset:offset + window])[-1]
                measured = motion_prefix(sequence["path_delta"][start:end].double())[-1]
                group["motion"].append(float((predicted[:2] - measured[:2]).square().sum()))
                group["actions"].append(float(np.square(actions[offset:offset + window] - sequence["actions"][start:end].numpy()).mean()))
    result = []
    for lead, group in sorted(groups.items()):
        scores, events, peaks = np.asarray(group["scores"]), np.asarray(group["events"], dtype=bool), np.asarray(group["tilts"])
        low = scores < risk_threshold
        missed, low_count, event_count = int((low & events).sum()), int(low.sum()), int(events.sum())
        result.append({"lead_window": lead, "windows": len(scores), "actual_risk_windows": event_count,
            "low_risk_windows": low_count, "low_risk_actual_events": missed,
            "low_risk_event_fraction": missed / low_count if low_count else None,
            "risk_false_negative_rate": missed / event_count if event_count else None,
            "risk_peak_tilt_correlation": float(np.corrcoef(scores, peaks)[0, 1]) if len(scores) > 1 and np.ptp(scores) > 0 and np.ptp(peaks) > 0 else None,
            "closed_loop_window_xy_rmse_m": float(np.sqrt(np.mean(group["motion"]))) if group["motion"] else None,
            "closed_loop_action_rmse": float(np.sqrt(np.mean(group["actions"]))) if group["actions"] else None})
    return result


def evaluate_pretrained_actor(cfg: Mapping[str, Any], *, root_dir: str | Path, output_dir: str | Path,
                              policy_checkpoint: str | Path, profile_path: str | None,
                              world_model_checkpoint: str | Path, actor_checkpoint: str | Path,
                              source_overrides: list[str], device: str = "cpu", conditions=None, playback_deps=None) -> dict[str, Any]:
    from unilab.algos.torch.sttt.path_following.pretraining import capture_configuration
    from unilab.training import ensure_registries

    if type(cfg["num_steps"]) is not int or cfg["num_steps"] < 1 or not cfg["seeds"] or not cfg["trajectories"]:
        raise ValueError("provide explicit evaluation steps, seeds and reference trajectories")
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in cfg["seeds"]):
        raise ValueError("evaluation seeds must be uint32 integers")
    horizon, threshold = cfg.get("horizon_windows", 2), cfg.get("risk_threshold", .5)
    if type(horizon) is not int or horizon < 1 or not 0 < threshold < 1:
        raise ValueError("evaluation requires a positive forecast horizon and probability threshold")
    conditions = tuple(cfg.get("conditions", ({"name": "nominal", "source_overrides": []},))) if conditions is None else tuple(conditions)
    if not conditions or len({condition["name"] for condition in conditions}) != len(conditions):
        raise ValueError("evaluation requires uniquely named physical conditions")
    if any(Path(condition["name"]).name != condition["name"] or condition["name"] in {".", ".."} for condition in conditions):
        raise ValueError("evaluation condition names must be single path components")
    root, output = Path(root_dir).resolve(), Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    loaded = load_world_model(world_model_checkpoint, device=device)
    payload = torch.load(actor_checkpoint, map_location="cpu", weights_only=True)
    base = TrajectoryConfig(preview_points=payload["architecture"]["preview_points"], preview_spacing_m=payload["preview"]["spacing_m"])
    validated_actor, _ = load_actor_checkpoint(actor_checkpoint, loaded.model, loaded.contract, base)
    modes = ("nominal", "fixed_context", "adaptive_context") if validated_actor.adaptation_dim else ("nominal", "pretrained_actor")
    contract = loaded.contract
    del loaded, validated_actor
    ensure_registries()
    output.mkdir(parents=True, exist_ok=True)
    summary = {"format": "sttt_pretrained_actor_evaluation", "schema_version": 3, "complete": False,
        "modes": list(modes),
        "world_model_checkpoint": str(world_model_checkpoint), "actor_checkpoint": str(actor_checkpoint), "runs": []}
    for condition in conditions:
        for index, values in enumerate(cfg["trajectories"]):
            trajectory = replace(base, **dict(values))
            if trajectory.preview_points != base.preview_points or trajectory.preview_spacing_m != base.preview_spacing_m:
                raise ValueError("evaluation must preserve the pretrained reference preview schema")
            for seed in cfg["seeds"]:
                initial = None
                row = {"condition": condition["name"], "trajectory": asdict(trajectory), "seed": seed}
                for mode in modes:
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    source = capture_configuration(root, Path(policy_checkpoint),
                        source_overrides=[*source_overrides, *condition["source_overrides"]], seed=seed, profile_path=profile_path)
                    directory = output / condition["name"] / f"{index:03d}_{trajectory.kind}_seed_{seed}_{mode}"
                    result = collect_world_model_sequences(source, root_dir=root, output_dir=directory, num_steps=cfg["num_steps"],
                        window_steps=contract["data"]["window_steps"], residual_amplitude_rad=0., seed=seed,
                        risk_tilt_limit_rad=contract["source"]["risk_tilt_limit_rad"], device=device, trajectory=trajectory,
                        actor_checkpoint=actor_checkpoint if mode != "nominal" else None,
                        world_model_checkpoint=world_model_checkpoint if mode != "nominal" else None,
                        forecast_windows=horizon if mode != "nominal" else None,
                        adapt_context=mode == "adaptive_context", playback_deps=playback_deps)
                    if result["localization_invalidated"] or result["reset_crossed"]:
                        raise RuntimeError("evaluation crossed a physical/localization boundary")
                    metrics, observation, matrix = score_capture(directory, trajectory, cfg["num_steps"])
                    if initial is None:
                        initial = observation, matrix
                    elif not np.allclose(initial[0], observation, atol=1e-7, rtol=0) or not np.allclose(initial[1], matrix, atol=1e-7, rtol=0):
                        raise ValueError("evaluation modes did not start from the same physical state")
                    row[mode] = {"directory": str(directory), **metrics,
                                 "forecast_diagnostics": score_forecasts(directory, risk_threshold=threshold)}
                summary["runs"].append(row)
                temporary = output / "summary.json.tmp"
                temporary.write_text(json.dumps(summary, allow_nan=False, indent=2) + "\n")
                temporary.replace(output / "summary.json")
    summary["evaluation_finished"] = True
    summary["complete"] = all(row[mode]["horizon_reached"] for row in summary["runs"] for mode in modes)
    (output / "summary.json").write_text(json.dumps(summary, allow_nan=False, indent=2) + "\n")
    return summary
