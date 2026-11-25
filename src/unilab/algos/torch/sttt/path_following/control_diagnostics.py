"""Five-stage residual authority diagnostics, without changing training artifacts."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from .control_probe import PhysicalTrace, playback_configuration, run_physical_branch
from .control_response import MeasuredFeedback, PulseProgram, fit_response_map, tracking_metrics
from .response_probe import _identity, _publish
from .trajectory import TrajectoryConfig


def run_control_diagnostics(cfg, *, root_dir: str | Path = ".", playback_deps=None) -> dict:
    from unilab.training import ensure_registries
    ensure_registries()
    raw = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else dict(cfg)
    if not isinstance(raw, dict):
        raise ValueError("diagnostic configuration must be a mapping")
    root = Path(root_dir).resolve()
    def resolve(value):
        if not value:
            raise ValueError("diagnostics require an explicit path")
        path = Path(value).expanduser()
        return (root / path).resolve() if not path.is_absolute() else path.resolve()
    mode = raw["mode"]
    if mode not in {"all", "physical", "analyze"}:
        raise ValueError("mode must be all, physical or analyze")
    run = resolve(raw["run_dir"]) if raw["run_dir"] else None
    training = json.loads((run / "summary.json").read_text())["config"] if run is not None else None
    if mode in {"all", "analyze"} and training is None:
        raise ValueError("model/value/actor diagnostics require run_dir with paired exported checkpoints")
    policy = resolve(raw["policy_checkpoint"] or (training or {}).get("policy_checkpoint"))
    if not policy.is_file():
        raise FileNotFoundError(policy)
    profile = raw["profile_path"] or (training or {}).get("profile_path")
    profile = str(resolve(profile)) if profile else None
    limits = raw["residual_limits_rad"] or (training or {}).get("actor", {}).get("training", {}).get("residual_limit_rad")
    limits = np.asarray(limits, dtype=np.float64)
    if limits.shape != (29,) or not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError("set the 29 actual source-order residual bounds, or supply their training run")
    k = raw["window_steps"] if raw["window_steps"] is not None else (training or {}).get("capture", {}).get("window_steps")
    if type(k) is not int or k < 1:
        raise ValueError("window_steps must match the production residual hold interval")
    phases, fractions = raw["phase_fractions"], raw["amplitude_fractions"]
    if len(set(phases)) != len(phases) or not phases or any(not math.isfinite(x) or not 0 <= x < 1 for x in phases):
        raise ValueError("phase_fractions require distinct fractions in [0,1)")
    if len(set(fractions)) != len(fractions) or not fractions or any(not math.isfinite(x) or not 0 < x <= 1 for x in fractions):
        raise ValueError("amplitude_fractions require distinct fractions in (0,1]")
    for name in ("warmup_steps", "feedback_solver_iterations", "model_top_directions"):
        if type(raw[name]) is not int or raw[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not raw["model_seeds"] or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in raw["model_seeds"]):
        raise ValueError("model_seeds must be nonempty uint32 common-random-number seeds")
    for name in ("speed_m_s", "observation_cycles", "feedback_duration_s", "continuation_duration_s", "risk_tilt_limit_rad"):
        if not math.isfinite(raw[name]) or raw[name] <= 0:
            raise ValueError(f"{name} must be positive and finite")
    base_values = (training or {}).get("actor", {}).get("trajectories", {}).get("base", {})
    base = replace(TrajectoryConfig.from_mapping(base_values), kind="straight", speed_m_s=raw["speed_m_s"],
                   initial_lateral_m=0., initial_heading_rad=0.)
    output = resolve(raw["output_dir"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    data_root = resolve(raw["data_dir"]) if mode == "analyze" else output
    report = {"format": "sttt_control_diagnostics", "version": 1, "complete": False, "mode": mode,
        "policy_identity": _identity(policy), "configuration": raw, "stages": {}, "error": None}
    _publish(output / "summary.json", report)
    _publish(output / "source.json", {"policy_identity": report["policy_identity"], "training_run": str(run) if run else None,
        "window_steps": k, "residual_limits_rad": limits.tolist(), "phase_fractions": phases,
        "amplitude_fractions": fractions, "trajectory": asdict(base), "physics": "fresh_native_sessions_without_snapshot_restore",
        "expected_pulse_branches": len(phases) * (2 + 29 * 2 * len(fractions)), "input_data": str(data_root)})

    def progress(stage, **values):
        print(json.dumps({"stage": stage, **values}, allow_nan=False), flush=True)

    def stage(name, value):
        report["stages"][name] = value
        _publish(output / "summary.json", report)
        progress(name, status=value["status"])

    source_overrides = list(raw["source_overrides"] if raw["source_overrides"] else (training or {}).get("source_overrides", []))
    playback_cfg = None
    def branch(name, program, *, trajectory=base, phase=0., seed=raw["seed"], warmup=None,
               steps=None, canonical=None, reference_commands=False, capture_steps=(0,)):
        nonlocal playback_cfg
        directory = data_root / name
        if mode == "analyze":
            trace = PhysicalTrace.load(directory)
            if trace.metadata["policy_identity"] != report["policy_identity"]:
                raise ValueError("analysis data belong to another policy")
            if canonical is not None:
                from .control_probe import same_origin
                same_origin(canonical, trace)
            return trace
        if playback_cfg is None:
            playback_cfg = playback_configuration(root, policy, profile_path=profile, seed=seed,
                speed=raw["speed_m_s"], source_overrides=source_overrides)
        progress("physical_branch", name=name, state="running")
        trace = run_physical_branch(playback_cfg, root=root, policy=policy, program=program, name=name, seed=seed,
            warmup_steps=raw["warmup_steps"] if warmup is None else warmup, window_steps=k, residual_limits=limits,
            trajectory=trajectory, observation_steps=steps, observation_cycles=raw["observation_cycles"], phase_fraction=phase,
            reference_commands=reference_commands, capture_steps=capture_steps, canonical=canonical,
            device=raw["device"], playback_deps=playback_deps, risk_tilt_limit_rad=raw["risk_tilt_limit_rad"])
        trace.save(directory)
        progress("physical_branch", name=name, state=trace.metadata["end_reason"], native_steps=trace.metadata["executed_steps"])
        if not trace.metadata["valid"]:
            raise ValueError(f"invalid physical branch {name}: {trace.metadata['error']}")
        return trace

    try:
        pulses = {}
        for p, phase in enumerate(phases):
            name = f"p{p}_zero_a"
            pulses[name] = branch("pulses/" + name, PulseProgram(np.zeros(29)), phase=phase)
            canonical = pulses[name]
            for joint in range(29):
                for amplitude, fraction in enumerate(fractions):
                    for sign, direction in (("plus", 1), ("minus", -1)):
                        residual = np.zeros(29); residual[joint] = direction * fraction * limits[joint]
                        name = f"p{p}_j{joint}_a{amplitude}_{sign}"
                        pulses[name] = branch("pulses/" + name, PulseProgram(residual), phase=phase, canonical=canonical)
            name = f"p{p}_zero_b"
            pulses[name] = branch("pulses/" + name, PulseProgram(np.zeros(29)), phase=phase, canonical=canonical)
        response, response_report = fit_response_map(pulses, phases, fractions,
            **{key: raw[key] for key in ("minimum_effect_m", "minimum_effect_rad", "replay_position_tolerance_m", "replay_heading_tolerance_rad")})
        response.save(output / "response_map.npz")
        stage("physical_authority", response_report)
        duration_steps = math.ceil(raw["feedback_duration_s"] / response.dt / k) * k
        feedback = {}
        for index, offset in enumerate(raw["feedback_offsets_m"]):
            case = replace(base, initial_lateral_m=offset)
            nominal = branch(f"feedback/o{index}_nominal", PulseProgram(np.zeros(29)), trajectory=case, seed=(raw["seed"] + 101 + index) % 2**32,
                warmup=raw["warmup_steps"] + k, steps=duration_steps)
            corrected = branch(f"feedback/o{index}_corrected", MeasuredFeedback(response, forward_scale=raw["speed_m_s"], iterations=raw["feedback_solver_iterations"]),
                trajectory=case, seed=(raw["seed"] + 101 + index) % 2**32, warmup=raw["warmup_steps"] + k,
                steps=duration_steps, canonical=nominal)
            before, after = tracking_metrics(nominal), tracking_metrics(corrected)
            improvement = before["lateral_integral_m_s"] - after["lateral_integral_m_s"]
            progress_ok = after["forward_progress_m"] >= raw["minimum_forward_retention"] * before["forward_progress_m"]
            feedback[f"offset_{offset}"] = {"nominal": before, "feedback": after,
                "lateral_integral_improvement_m_s": improvement, "forward_retained": progress_ok,
                "recovery_benefit": bool(improvement > 0 and progress_ok and not after["risk_event"])}
        stage("measured_feedback", {"status": "compared", "cases": feedback})
        if mode == "physical":
            report["complete"] = True
            _publish(output / "summary.json", report)
            return report

        from .control_model_diagnostics import (DiagnosticModels, FrozenActorProgram, critic_counterfactual,
            replay_world_model, response_contrast, shadow_actor_update)
        models = DiagnosticModels.load(run / "world_model.pt", run / "actor.pt", policy, trajectory=base, device=raw["device"])
        if models.learner.window_steps != k or not np.array_equal(models.actor.residual_limit_rad29.detach().cpu().numpy(), limits.astype(np.float32)):
            raise ValueError("diagnostic timing/bounds differ from the paired actor checkpoint")
        direction_scores = np.max(np.abs(response.response[:, :, 1]) / response.noise[:, None, 1, None], axis=1)
        selected = np.argsort(direction_scores, axis=1)[:, -min(29, raw["model_top_directions"]):]
        model_report = {"status": "compared", "selected_joints_by_phase": selected.tolist(), "seeds": raw["model_seeds"], "contrasts": [], "branches": []}
        for p in range(len(phases)):
            for joint in selected[p]:
                for amplitude in range(len(fractions)):
                    for seed in raw["model_seeds"]:
                        values = []
                        for sign in ("plus", "minus"):
                            name = f"p{p}_j{joint}_a{amplitude}_{sign}"
                            value = replay_world_model(pulses[name], models, seed=seed)
                            arrays = {key: value.pop(key) for key in ("open_motion", "closed_motion", "actual_motion") if key in value}
                            archive = output / "model_replay" / f"{name}_s{seed}.npz"
                            archive.parent.mkdir(exist_ok=True)
                            np.savez_compressed(archive, **arrays)
                            model_report["branches"].append({"branch": name, **value})
                            values.append({**value, **arrays})
                        contrast = response_contrast(*values, threshold=response.noise[p])
                        model_report["contrasts"].append({"phase": p, "joint": int(joint), "amplitude_fraction": fractions[amplitude], "seed": seed, **contrast})
                        progress("wm_replay", phase=p, joint=int(joint), amplitude=amplitude, seed=seed)
        rankings = []
        for p in range(len(phases)):
            for fraction in fractions:
                for seed in raw["model_seeds"]:
                    rows = [row for row in model_report["contrasts"] if row["phase"] == p and row["amplitude_fraction"] == fraction and row["seed"] == seed and row["status"] == "compared"]
                    for channel in range(3):
                        resolved = [row for row in rows if row["open_motion"]["sign_agreement"][channel] is not None]
                        rankings.append({"phase": p, "amplitude_fraction": fraction, "seed": seed, "channel": ["forward_m", "lateral_m", "heading_rad"][channel],
                            "actual_order": [row["joint"] for row in sorted(resolved, key=lambda row: abs(row["open_motion"]["actual_peak_m_m_rad"][channel]), reverse=True)],
                            **{name + "_order": [row["joint"] for row in sorted(resolved, key=lambda row: abs(row[name]["predicted_at_actual_peak_m_m_rad"][channel]), reverse=True)] for name in ("open_motion", "closed_motion")}})
        model_report["response_rankings"] = rankings
        stage("world_model", model_report)

        prefix_steps = models.learner.window_steps * models.learner.horizon_windows
        suffix_steps = math.ceil(raw["continuation_duration_s"] / response.dt / k) * k
        critic_report, shadow_report = {}, {}
        strongest = int(np.argmax(direction_scores.max(0)))
        direction = np.zeros(29); direction[strongest] = limits[strongest]
        for kind in raw["critic_trajectory_kinds"]:
            case = replace(base, kind=kind, initial_lateral_m=raw["critic_initial_lateral_m"])
            groups = {}
            canonical = None
            for name, residual in (("zero", np.zeros(29)), ("plus", direction), ("minus", -direction), ("actor", None)):
                program = FrozenActorProgram(models, seed=raw["seed"], prefix_steps=prefix_steps, prefix_residual=residual)
                trace = branch(f"critic/{kind}_{name}", program, trajectory=case, steps=prefix_steps + suffix_steps,
                    canonical=canonical, reference_commands=True, capture_steps=(0, prefix_steps))
                if canonical is None: canonical = trace
                groups[name] = trace
            critic_report[kind] = critic_counterfactual(groups, models, prefix_steps=prefix_steps, seed=raw["model_seeds"][0])
            shadow, changed_actor = shadow_actor_update(groups["actor"], models, seed=raw["model_seeds"][0])
            changed = branch(f"shadow/{kind}_after", FrozenActorProgram(models, seed=raw["seed"], prefix_steps=prefix_steps,
                prefix_actor=changed_actor), trajectory=case, steps=prefix_steps + suffix_steps,
                canonical=canonical, reference_commands=True, capture_steps=(0, prefix_steps))
            shadow["physical_comparison"] = critic_counterfactual({"before": groups["actor"], "after": changed}, models,
                prefix_steps=prefix_steps, seed=raw["model_seeds"][0])
            shadow["physical_tracking"] = {"before": tracking_metrics(groups["actor"]), "after": tracking_metrics(changed)}
            shadow_report[kind] = shadow
        stage("critic", {"status": "compared", "prefix_steps": prefix_steps, "continuation_steps": suffix_steps, "cases": critic_report})
        stage("actor_shadow", {"status": "compared", "cases": shadow_report})
        report["complete"] = True
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        progress("failed", **report["error"])
    _publish(output / "summary.json", report)
    return report
