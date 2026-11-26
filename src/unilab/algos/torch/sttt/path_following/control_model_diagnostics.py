"""Offline counterfactual replay, same-policy values and isolated actor substeps."""
from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from unilab.algos.torch.distill.fada.runtime_policy import FADAFutureQuery
from unilab.algos.torch.sttt.world_model.checkpoint import LoadedWorldModel, load_world_model
from unilab.algos.torch.sttt.world_model.motion import motion_prefix
from .actor_checkpoint import load_actor_checkpoint
from .actor_learner import ActorLearner
from .control_probe import PhysicalTrace
from .latent_actor import LatentResidualActor
from .control_response import relative_motion
from .latent_objectives import ObservedCostWindow, latent_actor_loss
from .latent_rollout import PlannerTrackerRollout, _frozen_model
from .pretraining import load_rollout
from .trajectory import PathState, ReferenceTrajectory, TrajectoryConfig


@contextmanager
def isolated_randomness(model, seed):
    parameter = next(model.parameters())
    with torch.random.fork_rng(devices=[parameter.device.index] if parameter.is_cuda else []):
        torch.manual_seed(seed)
        yield


@dataclass
class DiagnosticModels:
    loaded: LoadedWorldModel
    rollout: PlannerTrackerRollout
    actor: LatentResidualActor
    learner: ActorLearner
    actor_metadata: dict
    trajectory: TrajectoryConfig

    @classmethod
    def load(cls, world_model: Path, actor: Path, policy: Path, *, trajectory: TrajectoryConfig, device: str):
        loaded = load_world_model(world_model, device=device)
        loaded.model.eval().requires_grad_(False)
        rollout = load_rollout(loaded, policy, device=device)
        policy_actor, metadata = load_actor_checkpoint(actor, loaded.model, loaded.contract, trajectory)
        state = metadata["learner_state"]
        signature = state["signature"]
        training = {key: signature[key] for key in ("learning_rate", "max_grad_norm", "microbatch_size", "window_steps", "horizon_windows", "optimizer_betas", "final_learning_rate")}
        training.update(batch_size=1, critic=state["critic_learner"]["signature"]["config"])
        learner = ActorLearner(policy_actor, rollout, training, signature["costs"])
        learner.load_state_dict(state)
        return cls(loaded, rollout, policy_actor, learner, metadata, trajectory)

    def shadow_learner(self):
        signature = self.learner.signature
        training = {key: signature[key] for key in ("learning_rate", "max_grad_norm", "microbatch_size", "window_steps", "horizon_windows", "optimizer_betas", "final_learning_rate")}
        training.update(batch_size=1, critic=self.actor_metadata["learner_state"]["critic_learner"]["signature"]["config"])
        learner = ActorLearner(copy.deepcopy(self.actor), self.rollout, training, signature["costs"])
        learner.load_state_dict(copy.deepcopy(self.actor_metadata["learner_state"]))
        return learner


def trace_start(trace: PhysicalTrace, models: DiagnosticModels, *, step=0, seed=0):
    source, data = models.loaded.contract["source"], models.loaded.contract["data"]
    expected = {"profile": source["source_profile"], "preprocessing": source["preprocessing"],
        "target_joint_names": source["target_joint_names"], "target_default_joint_angles": source["target_default_joint_angles"],
        "target_action_scale": source["target_action_scale"], "ctrl_dt": data["ctrl_dt"], "window_steps": data["window_steps"]}
    for key, value in expected.items():
        if json.loads(json.dumps(trace.metadata[key])) != json.loads(json.dumps(value)):
            raise ValueError(f"physical trace and model disagree on {key}")
    if trace.metadata["policy_identity"]["sha256"] != source["checkpoint_sha256"]:
        raise ValueError("physical trace came from a different Planner–Tracker")
    prefix = "origin_" if step == 0 else f"at_{step}_"
    def tensor(key, boolean=False):
        return torch.as_tensor(trace.arrays[prefix + key], device=models.rollout.device,
                               dtype=torch.bool if boolean else models.rollout.dtype)
    trajectory = TrajectoryConfig.from_mapping(trace.metadata["trajectory"])
    reference = ReferenceTrajectory(tensor("reference_xy"), tensor("reference_speed"), trajectory)
    path = PathState(reference, tensor("path_pose"), tensor("path_progress"))
    query = FADAFutureQuery(tensor("observations"), tensor("source_actions"), tensor("command"), tensor("valid", True),
        tensor("future"), models.rollout.preprocessing, models.rollout.profile)
    with isolated_randomness(models.loaded.model, seed), torch.no_grad():
        return models.rollout.encode_start(query, errors=path.errors(), previous_residual=tensor("previous_residual"),
            ego_motion=tensor("ego_motion"), ego_valid=tensor("ego_valid", True), ego_age_s=tensor("ego_age_s"), path_state=path)


class LoggedResiduals(nn.Module):
    def __init__(self, residuals, actor, window_steps):
        super().__init__()
        self.residuals = residuals[::window_steps]
        self.actor = actor
        self.index = 0

    def context(self, *args):
        return self.actor.context(*args)

    def forward(self, feature, *_):
        if self.index >= len(self.residuals):
            raise ValueError("logged residual sequence is shorter than imagination")
        value = feature.new_tensor(self.residuals[self.index:self.index + 1])
        self.index += 1
        return value


def anchored_prefix(body_delta: torch.Tensor, start_pose: torch.Tensor) -> np.ndarray:
    prefix = motion_prefix(body_delta)[0].detach().cpu().numpy()
    yaw = float(start_pose[0, 2])
    x, y = prefix[:, 0].copy(), prefix[:, 1].copy()
    prefix[:, 0] = np.cos(yaw) * x - np.sin(yaw) * y
    prefix[:, 1] = np.sin(yaw) * x + np.cos(yaw) * y
    return prefix


def replay_world_model(trace: PhysicalTrace, models: DiagnosticModels, *, seed=0) -> dict:
    if not trace.metadata["valid"]:
        return {"status": "invalid", "reason": trace.metadata["error"]}
    k = models.learner.window_steps
    count = len(trace.arrays["actions"]) // k * k
    if count < k:
        return {"status": "insufficient_complete_window"}
    start = trace_start(trace, models, seed=seed)
    def array(value):
        return torch.as_tensor(value, device=models.rollout.device, dtype=models.rollout.dtype)
    with isolated_randomness(models.loaded.model, seed), torch.no_grad():
        latent = start.latent
        decoded, motion = [models.loaded.model.decode(latent)], []
        for action in array(trace.arrays["actions"][:count]).unsqueeze(0).unbind(1):
            motion.append(models.loaded.model.motion(latent, action))
            latent, _ = models.loaded.model.prior(latent, action)
            decoded.append(models.loaded.model.decode(latent))
        open_motion = anchored_prefix(torch.stack(motion, dim=1), start.path_state.pose)
        open_obs = torch.stack(decoded, dim=1)[0].cpu().numpy()
    with isolated_randomness(models.loaded.model, seed), torch.no_grad():
        imagined = models.rollout.run(start, LoggedResiduals(trace.arrays["residuals"][:count], models.actor, k),
            window_steps=k, horizon_windows=count // k, update_reference_commands=trace.metadata["reference_commands"])
        closed_motion = anchored_prefix(imagined.body_delta, start.path_state.pose)
        predicted_actions = imagined.actions[0].cpu().numpy()
    actual = relative_motion(trace)[:count + 1]
    action_error = predicted_actions - trace.arrays["actions"][:count]
    first_action_matches = bool(np.allclose(predicted_actions[0], trace.arrays["actions"][0], rtol=1e-6, atol=1e-6))
    result = {"status": "replayed" if first_action_matches else "invalid_first_action", "seed": seed,
        "native_steps": count, "first_action_matches": first_action_matches,
        "open_motion": open_motion, "closed_motion": closed_motion, "actual_motion": actual,
        "open_motion_rmse_m_m_rad": np.sqrt(np.mean((open_motion - actual)**2, axis=0)).tolist(),
        "closed_motion_rmse_m_m_rad": np.sqrt(np.mean((closed_motion - actual)**2, axis=0)).tolist(),
        "proprio_rmse_by_field": np.sqrt(np.mean((open_obs - trace.arrays["observations"][:count + 1])**2, axis=0)).tolist(),
        "closed_action_rmse_by_joint": np.sqrt(np.mean(action_error**2, axis=0)).tolist(),
        "closed_source_saturation_fraction": float((np.abs(imagined.source_actions[0].cpu().numpy()) >= models.rollout.profile.clip_actions).mean()) if models.rollout.profile.clip_actions is not None else 0.}
    return result


def response_contrast(plus: dict, minus: dict, *, threshold) -> dict:
    if plus["status"] != "replayed" or minus["status"] != "replayed":
        return {"status": "invalid_replay"}
    threshold = np.asarray(threshold)
    actual = (plus["actual_motion"] - minus["actual_motion"]) / 2
    rows = {}
    for name in ("open_motion", "closed_motion"):
        predicted = (plus[name] - minus[name]) / 2
        signal = np.abs(actual) > threshold
        index = np.argmax(np.abs(actual), axis=0)
        at_peak = predicted[index, np.arange(3)]
        actual_peak = actual[index, np.arange(3)]
        rows[name] = {"difference_rmse_m_m_rad": np.sqrt(np.mean((predicted - actual)**2, axis=0)).tolist(),
            "actual_peak_m_m_rad": actual_peak.tolist(), "predicted_at_actual_peak_m_m_rad": at_peak.tolist(),
            "sign_agreement": [bool(actual_peak[j] * at_peak[j] > 0) if signal[:, j].any() else None for j in range(3)],
            "amplitude_ratio": [float(at_peak[j] / actual_peak[j]) if signal[:, j].any() else None for j in range(3)],
            "actual_delay_s_index": [int(np.flatnonzero(signal[:, j])[0]) if signal[:, j].any() else None for j in range(3)],
            "predicted_delay_s_index": [int(np.flatnonzero(np.abs(predicted[:, j]) > threshold[j])[0]) if (np.abs(predicted[:, j]) > threshold[j]).any() else None for j in range(3)]}
    return {"status": "compared", **rows}


class FrozenActorProgram:
    """Same frozen continuation policy, conditional on each branch's real history."""
    def __init__(self, models: DiagnosticModels, *, seed: int, prefix_steps: int,
                 prefix_residual=None, prefix_actor=None, warmup_actor=True):
        self.models, self.seed, self.prefix_steps = models, seed, prefix_steps
        self.prefix_residual, self.prefix_actor, self.warmup_actor = prefix_residual, prefix_actor, warmup_actor

    def __call__(self, decision):
        if decision.step < 0 and not self.warmup_actor:
            return torch.zeros_like(decision.previous_residual)
        if decision.step % decision.window_steps:
            return decision.previous_residual
        if 0 <= decision.step < self.prefix_steps and self.prefix_residual is not None:
            return decision.query.command.new_tensor(np.asarray(self.prefix_residual)[None])
        actor = self.prefix_actor if 0 <= decision.step < self.prefix_steps and self.prefix_actor is not None else self.models.actor
        with isolated_randomness(self.models.loaded.model, (self.seed + decision.step) % 2**32), torch.no_grad():
            start = decision.encode(self.models.rollout)
            return actor(start.latent.feature(), start.errors, start.command, start.path_state.preview(), start.previous_residual)


def measured_return_interval(trace: PhysicalTrace, models: DiagnosticModels, *, step: int, seed=0) -> dict:
    """Native discounted cost; unobserved remainder is bounded, never valued as zero."""
    if not trace.metadata["valid"] or step >= len(trace.arrays["actions"]):
        return {"status": "prefix_not_reached", "end_reason": trace.metadata["end_reason"]}
    start = trace_start(trace, models, step=step, seed=seed)
    k, costs = models.learner.window_steps, models.learner.costs
    total = len(trace.arrays["actions"]) - step
    complete = total // k * k
    value = 0.
    def tensor(value):
        return torch.as_tensor(value, device=models.rollout.device, dtype=models.rollout.dtype).unsqueeze(0)
    if complete:
        end = step + complete
        errors = trace.arrays["errors"][step:end + 1]
        increments = np.diff(errors, axis=0)
        increments[:, 1] = np.arctan2(np.sin(increments[:, 1]), np.cos(increments[:, 1]))
        body = trace.arrays["body_delta"][step:end]
        angle = errors[:-1, 1]
        forward = np.cos(angle) * body[:, 0] - np.sin(angle) * body[:, 1]
        observed = ObservedCostWindow(errors=tensor(errors), path_delta=tensor(np.column_stack((forward, increments))),
            actions=tensor(trace.arrays["actions"][step:end]), source_actions=tensor(trace.arrays["source_actions"][step:end]),
            residuals=tensor(trace.arrays["residuals"][step:end:k]), risk=tensor(trace.arrays["risk"][step:end]),
            commands=tensor(trace.arrays["commands"][step:end]), target_speed=tensor(np.full(complete, trace.metadata["trajectory"]["speed_m_s"])))
        with torch.no_grad():
            value = float(latent_actor_loss(observed, start, ctrl_dt=models.rollout.profile.ctrl_dt, window_steps=k,
                config=costs, residual_limits=models.actor.residual_limit_rad29)["loss"])
    gamma, bound = costs.discount, models.learner.critic_learner.failure_value
    if trace.metadata["end_reason"] == "terminated":
        lower = value + gamma**total * bound
        maximum_rate = costs.lateral_weight + costs.lateral_velocity_weight + costs.forward_weight + costs.smoothness_weight + costs.risk_weight
        unseen_partial = gamma**complete * (1 - gamma**(total - complete)) / (1 - gamma) * models.rollout.profile.ctrl_dt * maximum_rate
        upper = lower + unseen_partial
    else:
        lower, upper = value, value + gamma**complete * bound
    with torch.no_grad():
        context = models.actor.context(start.latent.feature(), start.errors, start.command, start.path_state.preview(), start.previous_residual)
        actual_value = float(models.learner.critic_learner.target(context))
    context = context.detach().requires_grad_(True)
    gradient, = torch.autograd.grad(models.learner.critic_learner.target(context).sum(), context)
    return {"status": "bounded", "lower": lower, "upper": upper, "tail_width": upper - lower,
        "measured_steps": total, "complete_cost_steps": complete, "end_reason": trace.metadata["end_reason"],
        "value_on_actual_state": actual_value, "actual_state_value_in_interval": lower <= actual_value <= upper,
        "context_state_path_command_gradient_norm": float(gradient[:, :-29].norm()),
        "context_previous_residual_gradient_norm": float(gradient[:, -29:].norm())}


def interval_order(first: dict, second: dict) -> int | None:
    if first["status"] != "bounded" or second["status"] != "bounded":
        return None
    if first["upper"] < second["lower"]:
        return -1
    if second["upper"] < first["lower"]:
        return 1
    return None


def critic_counterfactual(traces: dict[str, PhysicalTrace], models: DiagnosticModels, *, prefix_steps: int, seed=0) -> dict:
    rows = {}
    for name, trace in traces.items():
        row = measured_return_interval(trace, models, step=prefix_steps, seed=seed)
        if row["status"] == "bounded":
            start = trace_start(trace, models, seed=seed)
            with isolated_randomness(models.loaded.model, seed), torch.no_grad():
                predicted = models.rollout.run(start, LoggedResiduals(trace.arrays["residuals"][:prefix_steps], models.actor, models.learner.window_steps),
                    window_steps=models.learner.window_steps, horizon_windows=prefix_steps // models.learner.window_steps,
                    update_reference_commands=trace.metadata["reference_commands"])
                row["value_on_imagined_state"] = float(models.learner.critic_learner.target(predicted.decision_contexts[:, -1]))
        rows[name] = row
    comparisons = []
    names = list(rows)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            order = interval_order(rows[first], rows[second])
            if order is None:
                comparisons.append({"first": first, "second": second, "status": "unresolved_overlapping_or_missing_return"})
            else:
                comparison = {"first": first, "second": second, "measured_order": order, "status": "resolved"}
                for field in ("value_on_actual_state", "value_on_imagined_state"):
                    difference = rows[first][field] - rows[second][field]
                    comparison[field + "_order_correct"] = bool(difference * order > 0)
                comparisons.append(comparison)
    return {"status": "compared", "continuation": "same_frozen_actor_on_each_real_history", "branches": rows, "ordering": comparisons}


def shadow_actor_update(trace: PhysicalTrace, models: DiagnosticModels, *, seed=0):
    """One real optimizer substep on an owned copy; no Critic fitting."""
    start = trace_start(trace, models, seed=seed)
    shadow = models.shadow_learner()
    def execute():
        with isolated_randomness(models.loaded.model, seed):
            result = models.rollout.run(start, shadow.actor, window_steps=shadow.window_steps, horizon_windows=shadow.horizon_windows)
            context = result.decision_contexts[:, -1]
            losses = latent_actor_loss(result, start, ctrl_dt=models.rollout.profile.ctrl_dt, window_steps=shadow.window_steps,
                config=shadow.costs, residual_limits=shadow.actor.residual_limit_rad29, terminal_cost=shadow.critic_learner.target(context))
            return result, losses
    with _frozen_model(models.rollout.model), _frozen_model(models.rollout.policy), _frozen_model(shadow.critic_learner.target):
        before, losses = execute()
        parameters = tuple(shadow.actor.parameters())
        gradients = {}
        for key in (*[name for name in losses if name.startswith("weighted_")], "terminal_cost", "loss"):
            values = torch.autograd.grad(losses[key], parameters, retain_graph=True, allow_unused=True)
            gradients[key] = {"parameter_norm": float(torch.sqrt(sum((value.square().sum() for value in values if value is not None), losses[key].new_zeros(())))),
                "output_bias": values[-1].detach().cpu().tolist() if values[-1] is not None else None}
        context = before.decision_contexts[:, -1]
        without_memory = torch.cat((context[:, :-29], context[:, -29:].detach()), dim=-1)
        discounted = shadow.costs.discount**(shadow.window_steps * shadow.horizon_windows)
        detached_terminal = shadow.critic_learner.target(without_memory).mean() * discounted
        values = torch.autograd.grad(detached_terminal, parameters, retain_graph=True, allow_unused=True)
        gradients["terminal_without_direct_memory"] = {"output_bias": values[-1].detach().cpu().tolist() if values[-1] is not None else None,
            "parameter_norm": float(torch.sqrt(sum((value.square().sum() for value in values if value is not None), detached_terminal.new_zeros(()))))}
        before_residual = before.residuals[0, 0].detach().clone()
        before_loss = {key: float(value.detach()) for key, value in losses.items()}
        del before, losses
        with isolated_randomness(models.loaded.model, seed):
            update_metrics = shadow.actor_step(start)
        with torch.no_grad():
            after, after_losses = execute()
            after_residual = after.residuals[0, 0].detach()
    report = {"status": "shadow_updated", "critic_refit": False, "optimizer_steps": 1,
        "before_costs": before_loss, "after_costs": {key: float(value.detach()) for key, value in after_losses.items()},
        "term_gradients": gradients, "optimizer_metrics": update_metrics,
        "before_residual_rad": before_residual.cpu().tolist(), "after_residual_rad": after_residual.cpu().tolist(),
        "delta_residual_rad": (after_residual - before_residual).cpu().tolist(),
        "before_saturation_fraction": float((before_residual.abs() >= .99 * shadow.actor.residual_limit_rad29).float().mean()),
        "after_saturation_fraction": float((after_residual.abs() >= .99 * shadow.actor.residual_limit_rad29).float().mean())}
    return report, shadow.actor.eval()
