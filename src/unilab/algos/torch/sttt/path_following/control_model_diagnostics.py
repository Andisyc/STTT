"""Offline counterfactual WM replay and isolated production context updates."""
from __future__ import annotations

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
from .context_adaptation import ContextAdaptationConfig, ContextAdapter
from .control_probe import PhysicalTrace
from .latent_actor import LatentResidualActor
from .control_response import relative_motion
from .latent_rollout import PlannerTrackerRollout
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
    actor_metadata: dict
    trajectory: TrajectoryConfig
    context_config: ContextAdaptationConfig

    @property
    def window_steps(self):
        return self.loaded.contract["data"]["window_steps"]

    @property
    def horizon_windows(self):
        return self.context_config.horizon_windows

    @classmethod
    def load(cls, world_model: Path, actor: Path, policy: Path, *, trajectory: TrajectoryConfig, device: str):
        loaded = load_world_model(world_model, device=device)
        loaded.model.eval().requires_grad_(False)
        rollout = load_rollout(loaded, policy, device=device)
        policy_actor, metadata = load_actor_checkpoint(actor, loaded.model, loaded.contract, trajectory)
        if metadata["stage"] != "film":
            raise ValueError("adaptive control diagnostics require the stage-2 FiLM checkpoint")
        policy_actor.eval().requires_grad_(False)
        context_config = ContextAdaptationConfig(**metadata["learner_state"]["context"]["config"])
        return cls(loaded, rollout, policy_actor, metadata, trajectory, context_config)


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
    k = models.window_steps
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
                 prefix_residual=None, adaptation_context=None, warmup_actor=True):
        self.models, self.seed, self.prefix_steps = models, seed, prefix_steps
        self.prefix_residual, self.adaptation_context, self.warmup_actor = prefix_residual, adaptation_context, warmup_actor

    def __call__(self, decision):
        if decision.step < 0 and not self.warmup_actor:
            return torch.zeros_like(decision.previous_residual)
        if decision.step % decision.window_steps:
            return decision.previous_residual
        if 0 <= decision.step < self.prefix_steps and self.prefix_residual is not None:
            return decision.query.command.new_tensor(np.asarray(self.prefix_residual)[None])
        actor = self.models.actor
        with isolated_randomness(self.models.loaded.model, (self.seed + decision.step) % 2**32), torch.no_grad():
            start = decision.encode(self.models.rollout)
            context = self.adaptation_context if decision.step >= 0 else None
            return actor(start.latent.feature(), start.errors, start.command, start.path_state.preview(), start.previous_residual,
                         adaptation_context=context)


def shadow_context_update(trace: PhysicalTrace, models: DiagnosticModels, *, seed=0):
    """Apply the shared production c update without changing A, FiLM or WM."""
    start = trace_start(trace, models, seed=seed)
    adapter = ContextAdapter(models.actor, models.rollout, batch_size=1,
                             window_steps=models.window_steps, config=models.context_config)
    inputs = (start.latent.feature(), start.errors, start.command, start.path_state.preview(), start.previous_residual)
    with torch.no_grad():
        before = models.actor(*inputs, adaptation_context=adapter.context)[0].clone()
        context_before = adapter.context.detach().clone()
    with isolated_randomness(models.loaded.model, seed):
        metrics = adapter.step(start)
    with torch.no_grad():
        after = models.actor(*inputs, adaptation_context=adapter.context)[0]
    report = {"status": "context_updated", "optimizer_steps": models.context_config.updates_per_window,
        "optimizer_metrics": metrics, "before_context": context_before[0].cpu().tolist(),
        "after_context": adapter.context[0].detach().cpu().tolist(),
        "before_residual_rad": before.cpu().tolist(), "after_residual_rad": after.cpu().tolist(),
        "delta_residual_rad": (after - before).cpu().tolist()}
    return report, adapter.context.detach().clone()
