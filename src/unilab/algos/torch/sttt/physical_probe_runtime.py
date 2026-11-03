"""Controlled MuJoCo action-sensitivity probe collection for STTT.

Collects physical branches around committed nominal FADA windows without
changing the policy, the reward, LoRA, or the world-model architecture.

Every sample pairs one committed window (the real MuJoCo label) with a
baseline replay and single-action perturbed replays executed through the
public ``env.isolated_rollout_branch()`` seam. Branch stepping never touches
the committed session policy/observation history, and no backend-internal
model or ``MjData`` is read here.

A branch is only compared when the condition is actually controlled: the
effective collection context must have no active domain randomization,
observation noise, push schedule, or command resampling, and every step of
the committed window must show a stable runtime-payload phase. Otherwise the
origin is rejected (or the sample status says ``condition_changed``) instead
of producing a silently biased finite difference. Agreement is local evidence
for the recorded finite perturbations, not an exact-Jacobian, stability, or
closed-loop adaptation guarantee.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .collection_runtime import (
    _capture_base_pose,
    _capture_proprio,
    _info_array,
    _payload_condition,
    _step_condition,
    _to_cpu_tensor,
    create_collection_session,
)
from .contract import DEFAULT_OBS_DIM, SpatialTrajectoryWindow, validate_trajectory_window
from .geometry import compute_se2_deltas
from .physical_probe import PhysicalProbeCase, PhysicalProbeConfig, PhysicalProbeSample

__all__ = ["collect_physical_probes"]

_PROBE_TASK_OVERRIDE = "task=g1_walk_flat/mujoco"
#: Statuses a successful branch must carry.
_BRANCH_OK = "ok"


def collect_physical_probes(
    cfg: Any,
    *,
    root_dir: Path,
) -> tuple[tuple[PhysicalProbeSample, ...], dict[str, Any]]:
    """Collect controlled physical branches around committed nominal windows.

    ``cfg.source`` names the existing FADA checkpoint, the seed, the device, and
    the Hydra overrides that pin the recording profile. ``cfg.probe`` is exactly
    :class:`~unilab.algos.torch.sttt.physical_probe.PhysicalProbeConfig`.

    The returned metadata extends the collection metadata with
    ``history_length``/``horizon``, ``probe_config``, ``probe_summary``, and a
    primitive description of the conditions that were actually active.
    """

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config = PhysicalProbeConfig(**OmegaConf.to_container(cfg.probe, resolve=True))
    checkpoint = getattr(cfg.source, "checkpoint_path", None)
    if checkpoint in (None, ""):
        raise FileNotFoundError(
            "Physical probes require source.checkpoint_path to name an existing "
            "FADA Planner-IDM checkpoint; there is no zero-policy fallback."
        )
    overrides = [str(item) for item in (getattr(cfg.source, "overrides", None) or [])]
    seed = int(getattr(cfg.source, "seed", 0))
    device_value = getattr(cfg.source, "device", None)
    device = None if device_value in (None, "") else str(device_value)

    config_dir = str(Path(root_dir) / "conf" / "distill")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        playback_cfg = compose(config_name="config", overrides=[_PROBE_TASK_OVERRIDE, *overrides])
    OmegaConf.update(playback_cfg, "training.play_checkpoint_path", str(checkpoint))

    session, metadata = create_collection_session(
        playback_cfg,
        root_dir=root_dir,
        device=device,
        seed=seed,
        prepare_rollout_branch=True,
    )
    try:
        context = _require_controlled_conditions(metadata)
        samples, summary, conditions = _collect_branches(
            session, metadata=metadata, config=config,
        )
        result = dict(metadata)
        result["history_length"] = int(config.history_length)
        result["horizon"] = int(config.horizon)
        result["probe_config"] = asdict(config)
        result["probe_summary"] = summary
        result["probe_conditions"] = _conditions_record(context, conditions)
        return tuple(samples), result
    finally:
        session.close()


def _require_controlled_conditions(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Reject a session whose effective conditions are not fixed."""

    context = metadata.get("collection_context")
    if not isinstance(context, Mapping):
        raise RuntimeError(
            "Physical probes require the collection_context block written by "
            "create_collection_session; it is missing from the session metadata."
        )
    violations = _condition_violations(context)
    if violations:
        raise ValueError(
            "Physical probes require a fixed, perturbation-free condition; the "
            "effective session config enables: " + ", ".join(sorted(violations))
        )
    return dict(context)


def _condition_violations(context: Mapping[str, Any]) -> list[str]:
    """List active randomization sources that would break branch pairing."""

    violations: list[str] = []
    domain_rand = context.get("domain_rand")
    if not isinstance(domain_rand, Mapping):
        violations.append("domain_rand")
    else:
        for key, value in domain_rand.items():
            if key == "push_robots":
                if bool(value):
                    violations.append(f"domain_rand.{key}")
            elif key.startswith("randomize_") and isinstance(value, bool) and value:
                violations.append(f"domain_rand.{key}")
            elif key == "random_com" and bool(value):
                violations.append("domain_rand.random_com")
            elif key == "torque_rfi_fraction" and float(value) != 0.0:
                violations.append(f"domain_rand.{key}")
        strength = domain_rand.get("actuator_strength")
        if isinstance(strength, Mapping) and (
            bool(strength.get("enabled", False))
            or bool(strength.get("group_curriculum_enabled", False))
        ):
            violations.append("domain_rand.actuator_strength")
    noise = context.get("noise_config")
    if not isinstance(noise, Mapping):
        violations.append("noise_config")
    elif float(noise.get("level", 0.0)) != 0.0:
        violations.append("noise_config.level")
    commands = context.get("commands")
    if not isinstance(commands, Mapping):
        violations.append("commands")
    else:
        if float(commands.get("resampling_time", 0.0)) != 0.0:
            violations.append("commands.resampling_time")
        if bool(commands.get("heading_command", False)):
            violations.append("commands.heading_command")
    return violations


def _conditions_record(
    context: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe the conditions that were effective, never the requested intent."""

    commands = context.get("commands") if isinstance(context.get("commands"), Mapping) else {}
    payload = (
        context.get("runtime_payload")
        if isinstance(context.get("runtime_payload"), Mapping)
        else {}
    )
    noise = context.get("noise_config") if isinstance(context.get("noise_config"), Mapping) else {}
    return {
        "command_limits": commands.get("vel_limit"),
        "command_resampling_time": commands.get("resampling_time"),
        "heading_command": commands.get("heading_command"),
        "payload_configured_enabled": payload.get("enabled"),
        "payload_configured_added_mass_kg": payload.get("added_mass_kg"),
        "payload_configured_attach_time_s": payload.get("attach_time_s"),
        "observation_noise_level": noise.get("level"),
        "context_keys_recorded": sorted(str(key) for key in context),
        "condition_source": "env.cfg blocks captured by create_collection_session",
        "payload_truth_source": "env.state.info runtime_payload_active/added_mass_kg",
        "payload_config_note": (
            "payload_configured_enabled is the configured flag; an origin only "
            "claims a payload when the runtime truth reports it active."
        ),
        **dict(observed),
    }


def _collect_branches(
    session: Any,
    *,
    metadata: Mapping[str, Any],
    config: PhysicalProbeConfig,
) -> tuple[list[PhysicalProbeSample], dict[str, Any], dict[str, Any]]:
    """Advance one committed session and replay controlled branches per origin."""

    env = session.env
    act_dim = int(env.action_space.shape[0])
    obs_dim = DEFAULT_OBS_DIM
    if act_dim != 29:
        raise ValueError(f"STTT physical probes require 29 actions, got {act_dim}")
    ctrl_low, ctrl_high = env.get_action_input_bounds()
    ctrl_low = np.asarray(ctrl_low, dtype=np.float64).reshape(-1)
    ctrl_high = np.asarray(ctrl_high, dtype=np.float64).reshape(-1)
    if ctrl_low.shape != (act_dim,) or ctrl_high.shape != (act_dim,):
        raise ValueError("Env-input bounds must match the action dimension")

    rollout_id = f"sttt_physical_probe_seed{int(metadata.get('seed', 0))}"
    case_steps = tuple(int(step) for step in config.action_steps)
    case_joints = tuple(int(joint) for joint in config.joint_indices)

    session.reset()
    episode_id = 0
    episode_step = 0
    total_steps = 0
    resets = 0
    # ``history`` always ends with the current committed state's proprio;
    # ``phases`` holds one payload phase per committed step of this episode.
    history: deque[np.ndarray] = deque(maxlen=config.history_length)
    phases: deque[str] = deque(maxlen=config.history_length + config.horizon)
    # Episode step the history tail currently describes; -1 means none yet.
    history_step = -1
    samples: list[PhysicalProbeSample] = []
    attempts: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    observed_commands: list[list[float]] = []
    last_origin_step: int | None = None
    # Search bound for eligible origins; failures never retry, so this only
    # stops a session that terminates before any window can complete.
    step_budget = config.warmup_steps + (config.num_samples + 1) * (
        max(int(config.stride_steps), 1) + config.history_length + config.horizon
    )
    stopped_reason = "completed"

    while len(attempts) < config.num_samples:
        if total_steps > step_budget:
            stopped_reason = "step_budget_exhausted"
            break
        state = env.state
        if state is None:
            raise RuntimeError("physical probes require an initialized environment state")
        if _is_done(state):
            stopped_reason = f"nominal_{_done_reason(state)}"
            break
        if history_step != episode_step:
            history.append(_capture_proprio(session))
            history_step = episode_step

        info = state.info
        command = np.array(_info_array(info, "commands", shape_tail=(3,))[0], copy=True)
        position, quaternion = _capture_base_pose(session)
        pre_payload = _payload_condition(info)
        eligible = (
            len(history) == config.history_length
            and len(phases) >= config.history_length - 1
            and episode_step >= config.warmup_steps
            and (last_origin_step is None or total_steps - last_origin_step >= config.stride_steps)
        )
        if not eligible:
            session.step_once()
            total_steps += 1
            post_state = env.state
            if post_state is None:
                raise RuntimeError("physical probes lost the environment state after a step")
            phases.append(
                _step_condition(pre_payload, _payload_condition(post_state.info))["phase"]
            )
            episode_step += 1
            if not _untransformed_action(
                post_state, None, act_dim=act_dim
            ):
                # A malformed committed input cannot provide a probe origin.
                rejections.append(
                    {
                        "episode_id": episode_id,
                        "episode_step": episode_step,
                        "committed_step": total_steps,
                        "reason": "action_transformed",
                    }
                )
                stopped_reason = "committed_action_transformed"
                break
            continue

        last_origin_step = total_steps
        record: dict[str, Any] = {
            "attempt": len(attempts),
            "rollout_id": rollout_id,
            "episode_id": episode_id,
            "origin_step": episode_step,
            "committed_step": total_steps,
        }
        origin_history = list(history)
        past_phases = (
            list(phases)[-(config.history_length - 1) :] if config.history_length > 1 else []
        )
        origin_snapshot = env.capture_rollout_snapshot()
        actions: list[np.ndarray] = []
        next_positions: list[np.ndarray] = []
        next_quaternions: list[np.ndarray] = []
        future_phases: list[str] = []
        failure: str | None = None
        step_payload = pre_payload

        for _ in range(config.horizon):
            step_state = env.state
            if step_state is None:
                raise RuntimeError("physical probes lost the environment state mid-window")
            step_command = _info_array(step_state.info, "commands", shape_tail=(3,))[0]
            if not np.array_equal(np.asarray(step_command), command):
                failure = "condition_changed"
                break
            session.step_once()
            total_steps += 1
            episode_step += 1
            post_state = env.state
            if post_state is None:
                raise RuntimeError("physical probes lost the environment state after a step")
            action_row = _info_array(
                post_state.info, "current_actions", shape_tail=(act_dim,)
            )[0]
            if not _untransformed_action(
                post_state, None, act_dim=act_dim
            ):
                failure = "action_transformed"
                break
            action_row = np.array(action_row, copy=True)
            next_position, next_quaternion = _capture_base_pose(session)
            post_payload = _payload_condition(post_state.info)
            future_phases.append(_step_condition(step_payload, post_payload)["phase"])
            history.append(_capture_proprio(session))
            history_step = episode_step
            if (
                not np.isfinite(action_row).all()
                or not np.isfinite(next_position).all()
                or not np.isfinite(next_quaternion).all()
            ):
                failure = "nonfinite"
                break
            actions.append(np.asarray(action_row, dtype=np.float64))
            next_positions.append(next_position)
            next_quaternions.append(next_quaternion)
            if _is_done(post_state):
                # The documented status vocabulary collapses every end cause
                # into ``terminated``; the precise cause stays in the summary.
                record["end_cause"] = _done_reason(post_state)
                failure = "terminated"
                break
            step_payload = post_payload

        phases.extend(future_phases)
        if failure is None and any(
            phase == "transitioning" for phase in past_phases + future_phases
        ):
            failure = "condition_changed"
        if failure is None and _ctrl_out_of_range(
            actions,
            ctrl_low=ctrl_low,
            ctrl_high=ctrl_high,
        ):
            failure = "out_of_bounds"
        if failure is not None:
            # The committed chain is no longer a faithful record of one
            # uninterrupted policy execution; drop it instead of stitching a
            # window across the break. The episode itself continues unless the
            # physics is already invalid.
            attempts.append({**record, "status": failure})
            history.clear()
            phases.clear()
            history_step = -1
            if failure in {"nonfinite", "terminated", "action_transformed"}:
                stopped_reason = f"nominal_{failure}"
                break
            continue

        window = _build_window(
            config,
            proprio_history=origin_history,
            actions=actions,
            next_positions=next_positions,
            next_quaternions=next_quaternions,
            command=command,
            position=position,
            quaternion=quaternion,
            first_step=record["origin_step"] - config.history_length + 1,
            act_dim=act_dim,
            obs_dim=obs_dim,
        )
        sample, branch_record = _probe_origin(
            session,
            snapshot=origin_snapshot,
            window=window,
            actions=actions,
            command=command,
            position=position,
            quaternion=quaternion,
            config=config,
            case_steps=case_steps,
            case_joints=case_joints,
            ctrl_low=ctrl_low,
            ctrl_high=ctrl_high,
            act_dim=act_dim,
            rollout_id=rollout_id,
            episode_id=episode_id,
            origin_step=record["origin_step"],
            window_phases=past_phases + future_phases,
            origin_payload=pre_payload,
        )
        samples.append(sample)
        attempts.append({**record, "status": "ok", **branch_record})
        observed_commands.append([float(value) for value in command])

    summary = {
        "attempts": len(attempts),
        "complete_samples": len(samples),
        "failed_attempts": sum(1 for item in attempts if item["status"] != _BRANCH_OK),
        "attempt_status_counts": dict(Counter(str(item["status"]) for item in attempts)),
        "baseline_status_counts": dict(Counter(s.baseline_status for s in samples)),
        "branch_status_counts": {
            "plus": dict(Counter(case.plus_status for s in samples for case in s.cases)),
            "minus": dict(Counter(case.minus_status for s in samples for case in s.cases)),
        },
        "cases_per_sample": len(case_steps) * len(case_joints),
        "committed_steps": total_steps,
        "episode_resets": resets,
        "stopped_reason": stopped_reason,
        "committed_steps_with_transformed_actions": len(rejections),
        "commit_rejections": rejections,
        "attempts_detail": attempts,
        "note": (
            "num_samples is a finite attempt count; failed attempts keep their "
            "explicit status, emit no window, and never substitute a zero "
            "endpoint. No retry-until-success pass exists."
        ),
    }
    observed = {
        "commands_observed": _unique_commands(observed_commands),
    }
    return samples, summary, observed


def _is_done(state: Any) -> bool:
    terminated = bool(np.asarray(state.terminated).reshape(-1)[0])
    truncated = bool(np.asarray(state.truncated).reshape(-1)[0])
    return terminated or truncated


def _done_reason(state: Any) -> str:
    """Explicit end cause for a finished episode, never collapsed into one label.

    Only used for the attempt summary; the probe packet uses the documented
    status vocabulary where every end cause is ``terminated``.
    """

    terminated = bool(np.asarray(state.terminated).reshape(-1)[0])
    truncated = bool(np.asarray(state.truncated).reshape(-1)[0])
    if terminated and not truncated:
        return "terminated"
    if truncated and not terminated:
        return "truncated"
    return "terminated_and_truncated"




def _untransformed_action(
    state: Any,
    requested: np.ndarray | None,
    *,
    act_dim: int,
) -> bool:
    """Check actual env_input, not downstream PD targets or actuator saturation.

    Committed nominal data uses the input the environment actually received.
    Perturbed branches additionally require that input to equal the request.
    Controller authority and PD mappings are part of the plant being probed.
    """

    info = state.info
    executed_input = np.array(
        _info_array(info, "current_actions", shape_tail=(act_dim,))[0], copy=True
    )
    if not np.isfinite(executed_input).all():
        return False
    if requested is None:
        return True
    reference = np.asarray(requested).astype(executed_input.dtype, copy=False)
    return reference.shape == executed_input.shape and np.array_equal(executed_input, reference)


def _unique_commands(values: Sequence[Sequence[float]]) -> list[list[float]]:
    unique: list[list[float]] = []
    for item in values:
        row = [float(value) for value in item]
        if row not in unique:
            unique.append(row)
    return unique


def _ctrl_out_of_range(
    actions: Sequence[np.ndarray],
    *,
    ctrl_low: np.ndarray,
    ctrl_high: np.ndarray,
) -> bool:
    """Check declared env-input bounds without converting to actuator units."""

    for action in actions:
        ctrl = np.asarray(action, dtype=np.float64)
        if np.any(ctrl < ctrl_low) or np.any(ctrl > ctrl_high):
            return True
    return False


def _build_window(
    config: PhysicalProbeConfig,
    *,
    proprio_history: Sequence[np.ndarray],
    actions: Sequence[np.ndarray],
    next_positions: Sequence[np.ndarray],
    next_quaternions: Sequence[np.ndarray],
    command: np.ndarray,
    position: np.ndarray,
    quaternion: np.ndarray,
    first_step: int,
    act_dim: int,
    obs_dim: int,
) -> SpatialTrajectoryWindow:
    """Materialize the committed window with ``collection._make_window`` semantics.

    ``proprio_history`` holds the ``history_length`` committed proprio rows
    ending at the origin, ``action_sequence`` the ``horizon`` executed
    env_input rows, ``delta_trajectory`` the SE(2) deltas of each executed
    step's next base pose relative to the origin pose (same origin-heading
    frame as the training labels), and ``step_indices`` the episode steps from
    the first history state through the last future state.
    """

    if len(proprio_history) != config.history_length:
        raise ValueError(
            f"committed history must hold {config.history_length} proprio rows, "
            f"got {len(proprio_history)}"
        )
    if len(actions) != config.horizon or len(next_positions) != config.horizon:
        raise ValueError(
            f"committed horizon must hold {config.horizon} action/pose rows, "
            f"got {len(actions)} actions and {len(next_positions)} poses"
        )
    if len(next_quaternions) != config.horizon:
        raise ValueError(
            f"committed horizon must hold {config.horizon} next quaternions, "
            f"got {len(next_quaternions)}"
        )
    origin_position = _to_cpu_tensor(np.asarray(position, dtype=np.float64))
    origin_quaternion = _to_cpu_tensor(np.asarray(quaternion, dtype=np.float64))
    next_position_tensor = torch.stack(
        [_to_cpu_tensor(np.asarray(item, dtype=np.float64)) for item in next_positions]
    )
    next_quaternion_tensor = torch.stack(
        [_to_cpu_tensor(np.asarray(item, dtype=np.float64)) for item in next_quaternions]
    )
    # Same dtype convention as ``collection._make_window``: every window
    # tensor is the float32 CPU clone of the captured array.
    window = SpatialTrajectoryWindow(
        proprio_history=torch.stack(
            [_to_cpu_tensor(np.asarray(item, dtype=np.float32)) for item in proprio_history]
        ),
        action_sequence=torch.stack(
            [_to_cpu_tensor(np.asarray(item, dtype=np.float32)) for item in actions]
        ),
        delta_trajectory=compute_se2_deltas(
            next_position_tensor.unsqueeze(0),
            next_quaternion_tensor.unsqueeze(0),
            pos0=origin_position.unsqueeze(0),
            quat0=origin_quaternion.unsqueeze(0),
        ).squeeze(0).to(torch.float32),
        commands=_to_cpu_tensor(np.asarray(command, dtype=np.float32)),
        step_indices=torch.arange(
            int(first_step),
            int(first_step) + config.history_length + config.horizon,
            dtype=torch.long,
        ),
    )
    validate_trajectory_window(
        window,
        expected_obs_dim=obs_dim,
        expected_act_dim=act_dim,
        expected_hist=config.history_length,
        expected_horizon=config.horizon,
    )
    return window


def _pose_delta(
    position: np.ndarray,
    quaternion: np.ndarray,
    reference_position: np.ndarray,
    reference_quaternion: np.ndarray,
) -> torch.Tensor:
    """Endpoint ``[dx, dy, dyaw]`` in the origin-heading frame."""

    return compute_se2_deltas(
        _to_cpu_tensor(np.asarray(position, dtype=np.float64)).reshape(1, 1, 3),
        _to_cpu_tensor(np.asarray(quaternion, dtype=np.float64)).reshape(1, 1, 4),
        pos0=_to_cpu_tensor(np.asarray(reference_position, dtype=np.float64)).reshape(1, 3),
        quat0=_to_cpu_tensor(np.asarray(reference_quaternion, dtype=np.float64)).reshape(1, 4),
    )[0, 0]


def _replay_branch(
    session: Any,
    snapshot: Any,
    actions: Sequence[np.ndarray],
    *,
    command: np.ndarray,
    position: np.ndarray,
    quaternion: np.ndarray,
    act_dim: int,
) -> tuple[torch.Tensor | None, str]:
    """Replay recorded env_input actions from ``snapshot`` inside a branch.

    The policy is never re-invoked; the session observation and history are
    untouched. A step that is transformed, clipped, terminal, non-finite, or
    crossed by a condition change fails the branch with an explicit status and
    no endpoint.
    """

    env = session.env
    env.restore_rollout_snapshot(snapshot)
    state = env.state
    if state is None:
        raise RuntimeError("branch replay requires an initialized environment state")
    recorded_input = np.asarray(state.info["current_actions"])
    if recorded_input.shape != (1, act_dim):
        raise ValueError("branch replay requires one recorded env_input row")
    action_dtype = recorded_input.dtype

    pre_payload = _payload_condition(state.info)
    for action in actions:
        current_state = env.state
        if current_state is None:
            raise RuntimeError("branch replay lost the environment state")
        branch_command = _info_array(current_state.info, "commands", shape_tail=(3,))[0]
        if not np.array_equal(np.asarray(branch_command), command):
            return None, "condition_changed"
        requested = np.asarray(action, dtype=action_dtype).reshape(1, act_dim)
        env.step(requested)
        post_state = env.state
        if post_state is None:
            raise RuntimeError("branch replay lost the environment state after a step")
        if not _untransformed_action(post_state, requested[0], act_dim=act_dim):
            return None, "action_transformed"
        if _is_done(post_state):
            # The packet vocabulary has no truncation status: every end cause is
            # an explicit ``terminated`` and never a fabricated endpoint.
            return None, "terminated"
        next_position, next_quaternion = _capture_base_pose(session)
        if not np.isfinite(next_position).all() or not np.isfinite(next_quaternion).all():
            return None, "nonfinite"
        phase = _step_condition(pre_payload, _payload_condition(post_state.info))
        if phase["phase"] == "transitioning":
            return None, "condition_changed"
        pre_payload = _payload_condition(post_state.info)

    end_position, end_quaternion = _capture_base_pose(session)
    return _pose_delta(end_position, end_quaternion, position, quaternion), _BRANCH_OK


def _perturbed_actions(
    actions: Sequence[np.ndarray],
    *,
    action_step: int,
    joint_index: int,
    delta: float,
    ctrl_low: np.ndarray,
    ctrl_high: np.ndarray,
) -> list[np.ndarray] | None:
    """One env-input perturbation, or None when it leaves the input bounds."""

    perturbed = [np.array(action, dtype=np.float64, copy=True) for action in actions]
    target = perturbed[action_step]
    target[joint_index] = target[joint_index] + float(delta)
    if not np.isfinite(target).all():
        return None
    ctrl = target
    if np.any(ctrl < ctrl_low) or np.any(ctrl > ctrl_high):
        return None
    return perturbed


def _probe_origin(
    session: Any,
    *,
    snapshot: Any,
    window: SpatialTrajectoryWindow,
    actions: Sequence[np.ndarray],
    command: np.ndarray,
    position: np.ndarray,
    quaternion: np.ndarray,
    config: PhysicalProbeConfig,
    case_steps: Sequence[int],
    case_joints: Sequence[int],
    ctrl_low: np.ndarray,
    ctrl_high: np.ndarray,
    act_dim: int,
    rollout_id: str,
    episode_id: int,
    origin_step: int,
    window_phases: Sequence[str],
    origin_payload: tuple[bool, float, bool] | None,
) -> tuple[PhysicalProbeSample, dict[str, Any]]:
    """Run the baseline and perturbed replays for one committed origin."""

    env = session.env
    cases: list[PhysicalProbeCase] = []
    statuses: list[str] = []
    baseline_endpoint: torch.Tensor | None = None
    baseline_status = "not_run"
    with env.isolated_rollout_branch():
        baseline_endpoint, baseline_status = _replay_branch(
            session,
            snapshot,
            actions,
            command=command,
            position=position,
            quaternion=quaternion,
            act_dim=act_dim,
        )
        for action_step in case_steps:
            for joint_index in case_joints:
                endpoints: dict[str, torch.Tensor | None] = {}
                statuses_by_sign: dict[str, str] = {}
                for sign, delta in (("plus", config.epsilon), ("minus", -config.epsilon)):
                    perturbed = _perturbed_actions(
                        actions,
                        action_step=action_step,
                        joint_index=joint_index,
                        delta=delta,
                        ctrl_low=ctrl_low,
                        ctrl_high=ctrl_high,
                    )
                    if perturbed is None:
                        endpoints[sign] = None
                        statuses_by_sign[sign] = "out_of_bounds"
                        continue
                    endpoint, status = _replay_branch(
                        session,
                        snapshot,
                        perturbed,
                        command=command,
                        position=position,
                        quaternion=quaternion,
                        act_dim=act_dim,
                    )
                    endpoints[sign] = endpoint
                    statuses_by_sign[sign] = status
                cases.append(
                    PhysicalProbeCase(
                        action_step=int(action_step),
                        joint_index=int(joint_index),
                        epsilon=float(config.epsilon),
                        plus_endpoint=endpoints["plus"],
                        minus_endpoint=endpoints["minus"],
                        plus_status=statuses_by_sign["plus"],
                        minus_status=statuses_by_sign["minus"],
                    )
                )
                statuses.extend(statuses_by_sign.values())

    if origin_payload is None:
        payload_active: bool | None = None
        added_mass: float | None = None
        payload_phase = "unknown"
    else:
        payload_active, added_mass, _attached = origin_payload
        observed_phases = {str(phase) for phase in window_phases}
        if "unknown" in observed_phases:
            payload_phase = "unknown"
        elif "transitioning" in observed_phases:
            payload_phase = "transitioning"
        else:
            payload_phase = "stable"
    stable = (
        None
        if not window_phases or any(phase == "unknown" for phase in window_phases)
        else all(phase == "stable" for phase in window_phases)
    )
    condition = {
        "payload_phase": payload_phase,
        "payload_active": payload_active,
        "added_mass_kg": added_mass,
        "window_phases": [str(phase) for phase in window_phases],
        "window_condition_stable": stable,
        "command": [float(value) for value in np.asarray(command).reshape(-1)],
        "payload_truth_source": "env.state.info runtime_payload_active/added_mass_kg",
    }
    sample = PhysicalProbeSample(
        rollout_id=rollout_id,
        episode_id=int(episode_id),
        origin_step=int(origin_step),
        window=window,
        baseline_endpoint=baseline_endpoint,
        baseline_status=baseline_status,
        cases=tuple(cases),
        condition=condition,
    )
    return sample, {
        "baseline_status": baseline_status,
        "case_status_counts": dict(Counter(statuses)),
    }