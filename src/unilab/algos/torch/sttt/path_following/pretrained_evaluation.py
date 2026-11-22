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
from unilab.algos.torch.sttt.path_following.trajectory import MeasuredTrajectory, TrajectoryConfig
from unilab.algos.torch.sttt.pose_data import TimedPelvisPose
from unilab.algos.torch.sttt.world_model.checkpoint import load_world_model
from unilab.algos.torch.sttt.world_model.sequence_collection import collect_world_model_sequences
from unilab.algos.torch.sttt.world_model.sequence_data import iter_sequences


def score_capture(directory: Path, trajectory: TrajectoryConfig, requested_steps: int) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    metadata = json.loads((directory / "capture.json").read_text())
    receipts = {item["timestamp_ns"]: item for item in metadata["pose_receipts"] if item["valid"]}
    if len(receipts) < 2:
        raise ValueError("evaluation requires physical start/end poses")
    reference = None
    errors, progresses, times = [], [], []
    initial_matrix = None
    for timestamp, item in sorted(receipts.items()):
        pose = TimedPelvisPose(item["sequence_id"], item["tracking_epoch"], timestamp,
                              np.asarray(item["navigation_from_pelvis"]), True,
                              tuple(item["source_timestamps_ns"]), item["frame_id"])
        if reference is None:
            reference = MeasuredTrajectory(pose, trajectory, dtype=torch.float64)
            initial_matrix = np.asarray(item["navigation_from_pelvis"])
        state = reference.state_at(pose)
        errors.append(state.errors()[0].numpy())
        progresses.append(float(state.progress[0]))
        times.append(timestamp)
    sequences = list(iter_sequences(directory))
    observations = sequences[0]["observations"][0].numpy()
    executed = sum(len(sequence["actions"]) for sequence in sequences)
    risk_events = sum(int((sequence["risk"][:, 0] > .5).sum()) for sequence in sequences)
    duration = (times[-1] - times[0]) / 1e9
    if duration <= 0 or initial_matrix is None:
        raise ValueError("evaluation physical time did not advance")
    errors = np.asarray(errors)
    progress = progresses[-1] - progresses[0]
    return {"lateral_rmse_m": float(np.sqrt(np.mean(errors[:, 0] ** 2))),
            "lateral_max_abs_m": float(np.max(np.abs(errors[:, 0]))),
            "final_lateral_m": float(errors[-1, 0]), "forward_progress_m": progress,
            "forward_speed_retention": progress / (trajectory.speed_m_s * duration),
            "duration_s": duration, "executed_steps": executed,
            "horizon_reached": executed == requested_steps, "risk_event_fraction": risk_events / executed}, observations, initial_matrix


def evaluate_pretrained_actor(cfg: Mapping[str, Any], *, root_dir: str | Path, output_dir: str | Path,
                              policy_checkpoint: str | Path, profile_path: str | None,
                              world_model_checkpoint: str | Path, actor_checkpoint: str | Path,
                              source_overrides: list[str], device: str = "cpu") -> dict[str, Any]:
    from unilab.algos.torch.sttt.path_following.pretraining import capture_configuration
    from unilab.training import ensure_registries

    if type(cfg["num_steps"]) is not int or cfg["num_steps"] < 1 or not cfg["seeds"] or not cfg["trajectories"]:
        raise ValueError("provide explicit evaluation steps, seeds and reference trajectories")
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in cfg["seeds"]):
        raise ValueError("evaluation seeds must be uint32 integers")
    root, output = Path(root_dir).resolve(), Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    loaded = load_world_model(world_model_checkpoint, device=device)
    payload = torch.load(actor_checkpoint, map_location="cpu", weights_only=True)
    base = TrajectoryConfig(preview_points=payload["architecture"]["preview_points"],
                            preview_spacing_m=payload["preview"]["spacing_m"])
    load_actor_checkpoint(actor_checkpoint, loaded.model, loaded.contract, base)
    contract = loaded.contract
    del loaded
    ensure_registries()
    output.mkdir(parents=True, exist_ok=True)
    summary = {"format": "sttt_pretrained_actor_evaluation", "schema_version": 1, "complete": False,
               "world_model_checkpoint": str(world_model_checkpoint), "actor_checkpoint": str(actor_checkpoint), "runs": []}
    for index, values in enumerate(cfg["trajectories"]):
        trajectory = replace(base, **dict(values))
        if trajectory.preview_points != base.preview_points or trajectory.preview_spacing_m != base.preview_spacing_m:
            raise ValueError("evaluation must preserve the pretrained reference preview schema")
        for seed in cfg["seeds"]:
            initial = None
            row = {"trajectory": asdict(trajectory), "seed": seed}
            for mode in ("nominal", "pretrained_actor"):
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                source = capture_configuration(root, Path(policy_checkpoint), source_overrides=source_overrides, seed=seed, profile_path=profile_path)
                directory = output / f"{index:03d}_{trajectory.kind}_seed_{seed}_{mode}"
                result = collect_world_model_sequences(
                    source, root_dir=root, output_dir=directory, num_steps=cfg["num_steps"],
                    window_steps=contract["data"]["window_steps"], residual_amplitude_rad=0., seed=seed,
                    risk_tilt_limit_rad=contract["source"]["risk_tilt_limit_rad"], device=device,
                    trajectory=trajectory, actor_checkpoint=actor_checkpoint if mode == "pretrained_actor" else None,
                    world_model_checkpoint=world_model_checkpoint if mode == "pretrained_actor" else None,
                )
                if result["localization_invalidated"] or result["reset_crossed"]:
                    raise RuntimeError("evaluation crossed a physical/localization boundary")
                metrics, observation, matrix = score_capture(directory, trajectory, cfg["num_steps"])
                if initial is None:
                    initial = observation, matrix
                elif not np.allclose(initial[0], observation, atol=1e-7, rtol=0) or not np.allclose(initial[1], matrix, atol=1e-7, rtol=0):
                    raise ValueError("nominal and A evaluation did not start from the same physical state")
                row[mode] = {"directory": str(directory), **metrics}
            summary["runs"].append(row)
            temporary = output / "summary.json.tmp"
            temporary.write_text(json.dumps(summary, allow_nan=False, indent=2) + "\n")
            temporary.replace(output / "summary.json")
    summary["complete"] = all(row[mode]["horizon_reached"] for row in summary["runs"] for mode in ("nominal", "pretrained_actor"))
    (output / "summary.json").write_text(json.dumps(summary, allow_nan=False, indent=2) + "\n")
    return summary
