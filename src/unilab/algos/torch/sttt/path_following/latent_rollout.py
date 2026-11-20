"""Differentiable receding-horizon execution in a learned fixed-path frame."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Iterator

import torch
from torch import nn

from unilab.algos.torch.distill.fada.collection_contract import validate_official_fada_preprocessing
from unilab.algos.torch.distill.fada.runtime_policy import (
    FADAFutureQuery, OfficialFADAPlaybackProfile, get_fada_runtime_spec, invoke_fada_idm,
)
from unilab.algos.torch.sttt.geometry import wrap_to_pi
from unilab.algos.torch.sttt.path_following.intervention import compose_joint_position_residual
from unilab.algos.torch.sttt.world_model.types import LatentState, RolloutStart


@dataclass(frozen=True)
class RolloutResult:
    """Every transition uses ENV actions; Tracker histories use SOURCE actions.

    ``risk`` contains event probabilities (sigmoid of model logits), not logits.
    ``residuals`` has one source-order physical-radian value per held window.
    ``path_delta`` is already in the fixed path frame, never a body frame.
    """

    errors: torch.Tensor
    path_delta: torch.Tensor
    actions: torch.Tensor
    source_actions: torch.Tensor
    residuals: torch.Tensor
    risk: torch.Tensor
    commands: torch.Tensor
    final_latent: LatentState


@contextmanager
def _frozen_model(model: nn.Module) -> Iterator[None]:
    """Freeze weights, not operations: input derivatives must cross the model."""
    parameters = tuple(model.parameters())
    requires_grad = tuple(parameter.requires_grad for parameter in parameters)
    training = tuple((module, module.training) for module in model.modules())
    try:
        model.eval()
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, enabled in zip(parameters, requires_grad):
            parameter.requires_grad_(enabled)
        for module, enabled in training:
            module.training = enabled


class PlannerTrackerRollout:
    """Use the real source clip and deployment affine map without an environment.

    Velocity commands are held over imagination; the two gait phases advance
    each control tick using the source profile clock. For ``command_fixed`` a
    stationary command retains its already-observed standing phase. There are
    no invented command changes, resets, observations or future SLAM labels.
    """

    def __init__(
        self,
        model: nn.Module,
        policy: nn.Module,
        profile: OfficialFADAPlaybackProfile,
        preprocessing: Mapping[str, Any],
        *,
        target_joint_names: Sequence[str],
        target_default_joint_angles: Sequence[float] | torch.Tensor,
        target_action_scale: float | Sequence[float] | torch.Tensor,
    ) -> None:
        self.model = model
        self.policy = policy
        self.profile = profile
        self.spec = get_fada_runtime_spec(policy)
        if self.spec.family != "official" or (self.spec.obs_dim, self.spec.action_dim, self.spec.command_dim) != (64, 29, 7):
            raise ValueError("latent rollout requires the official 64/29/7 policy")
        if profile.gait_clock_mode not in {"continuous", "command_fixed"}:
            raise ValueError("latent rollout requires a continuous or command_fixed source clock")
        if any(not math.isfinite(value) or value <= 0 for value in (profile.ctrl_dt, profile.gait_frequency, profile.clip_actions)):
            raise ValueError("latent rollout requires positive finite source timing and action clip")
        self.preprocessing = validate_official_fada_preprocessing(preprocessing)
        if self.spec.preprocessing is not None and self.preprocessing != self.spec.preprocessing:
            raise ValueError("rollout preprocessing differs from the policy checkpoint")
        scales = self.preprocessing["term_scale"]
        self.position_scale = float(scales["dof_pos"])
        self.velocity_scale = float(scales["dof_vel"])
        parameter = next(policy.parameters())
        self.device, self.dtype = parameter.device, parameter.dtype
        source_from_target = torch.tensor(profile.source_indices(target_joint_names), device=self.device, dtype=torch.long)
        self._source_from_target = source_from_target
        self._target_from_source = torch.argsort(source_from_target)
        default = torch.as_tensor(target_default_joint_angles, device=self.device, dtype=self.dtype)
        scale = torch.as_tensor(target_action_scale, device=self.device, dtype=self.dtype)
        if scale.ndim == 0:
            scale = scale.expand(29)
        if default.shape != (29,) or scale.shape != (29,) or not bool(torch.isfinite(default).all()) or not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
            raise ValueError("rollout requires finite target defaults and positive per-joint target scales")
        source_default = torch.tensor(profile.default_joint_pos, device=self.device, dtype=self.dtype).index_select(0, self._target_from_source)
        source_scale = torch.tensor(profile.action_scale, device=self.device, dtype=self.dtype).index_select(0, self._target_from_source)
        if not bool(torch.isfinite(source_default).all()) or not bool(torch.isfinite(source_scale).all()) or bool((source_scale <= 0).any()):
            raise ValueError("rollout requires finite source defaults and positive source scales")
        self._action_gain = (source_scale / scale).detach()
        self._action_bias = ((source_default - default) / scale).detach()
        phase_delta = 2.0 * math.pi * profile.gait_frequency * profile.ctrl_dt
        self._phase_sin, self._phase_cos = math.sin(phase_delta), math.cos(phase_delta)
        # The original Planner/Tracker are never optimization variables. Do not
        # use their no-grad playback entrypoints: A needs the input Jacobians.
        policy.eval()
        policy.requires_grad_(False)

    def source_to_env(self, action: torch.Tensor) -> torch.Tensor:
        """Affine deployment mapping for [...,29] already-clipped SOURCE commands."""
        self._validate_action(action)
        return action.index_select(-1, self._target_from_source) * self._action_gain + self._action_bias

    def env_to_source(self, action: torch.Tensor) -> torch.Tensor:
        """Inverse affine mapping for [...,29] ENV commands, without clipping."""
        self._validate_action(action)
        return ((action - self._action_bias) / self._action_gain).index_select(-1, self._source_from_target)

    def _validate_action(self, action: torch.Tensor) -> None:
        if action.ndim < 1 or action.shape[-1] != 29 or action.dtype != self.dtype or action.device != self.device:
            raise ValueError("action mapping requires [...,29] with the policy dtype and device")

    def _validate_history(
        self, observations: torch.Tensor, actions: torch.Tensor, valid: torch.Tensor,
    ) -> None:
        if observations.ndim != 3 or observations.shape[0] < 1 or observations.shape[1:] != (self.spec.history_length, 64):
            raise ValueError("rollout observation history must be [B,H,64] with the checkpoint history length")
        batch, history = observations.shape[:2]
        if actions.shape != (batch, history, 29) or valid.shape != (batch, history) or valid.dtype != torch.bool or valid.device != self.device:
            raise ValueError("rollout requires aligned SOURCE action history and boolean valid mask")
        for value in (observations, actions):
            if value.dtype != self.dtype or value.device != self.device or not bool(torch.isfinite(value).all()):
                raise ValueError("rollout history must be finite and match the policy dtype and device")
        if not bool(valid[:, -1].all()) or bool((valid[:, :-1] & ~valid[:, 1:]).any()):
            raise ValueError("rollout history must have a nonempty valid suffix")
        if bool((observations[~valid] != 0).any()) or bool((actions[~valid] != 0).any()):
            raise ValueError("rollout history padding must be zero")

    @torch.no_grad()
    def encode_start(
        self,
        query: FADAFutureQuery,
        *,
        errors: torch.Tensor,
        previous_residual: torch.Tensor,
        ego_motion: torch.Tensor | None = None,
        ego_valid: torch.Tensor | None = None,
        ego_age_s: torch.Tensor | None = None,
    ) -> RolloutStart:
        """Own a current query; optional ego fields describe its entire history.

        ``ego_motion`` is [B,H,3]; validity and age are [B,H,1]. Only
        previously available measurements belong here, never future labels.
        Histories already include the current observation and its preceding
        SOURCE command. Padding and the first valid reset action remain zero
        after conversion to ENV semantics (even with nonzero affine bias).
        """
        if query.profile != self.profile or validate_official_fada_preprocessing(query.preprocessing) != self.preprocessing:
            raise ValueError("rollout query source profile or preprocessing mismatch")
        observations = query.observation_history.detach().clone()
        source_actions = query.action_history.detach().clone()
        valid = query.history_valid_mask.detach().clone()
        self._validate_history(observations, source_actions, valid)
        self._validate_context(observations.shape[0], errors, query.command, previous_residual)
        batch, history = observations.shape[:2]
        for name, value, width in (("ego_motion", ego_motion, 3), ("ego_valid", ego_valid, 1), ("ego_age_s", ego_age_s, 1)):
            if value is not None and (value.shape != (batch, history, width) or value.device != self.device):
                raise ValueError(f"{name} must cover the full current [B,H,{width}] history")
        previous_valid = torch.cat((torch.zeros_like(valid[:, :1]), valid[:, :-1]), dim=1)
        env_actions = self.source_to_env(source_actions).masked_fill(~(valid & previous_valid).unsqueeze(-1), 0.)
        with _frozen_model(self.model):
            latent = self.model.encode_history(
                observations, env_actions, valid,
                ego_motion=ego_motion, ego_valid=ego_valid, ego_age_s=ego_age_s,
            )
        return RolloutStart(
            latent=latent.detach().clone(), observation_history=observations,
            action_history=source_actions, history_valid_mask=valid,
            errors=errors.detach().clone(), command=query.command.detach().clone(),
            previous_residual=previous_residual.detach().clone(),
        )

    def _validate_context(
        self, batch: int, errors: torch.Tensor, command: torch.Tensor, residual: torch.Tensor,
    ) -> None:
        for name, value, width in (("errors", errors, 2), ("command", command, 7), ("previous_residual", residual, 29)):
            if value.shape != (batch, width) or value.dtype != self.dtype or value.device != self.device or not bool(torch.isfinite(value).all()):
                raise ValueError(f"rollout {name} must be finite [B,{width}] with the policy dtype and device")

    def _advance_command(self, command: torch.Tensor) -> torch.Tensor:
        sine, cosine = command[:, 3:5], command[:, 5:7]
        next_phase = torch.cat((
            sine * self._phase_cos + cosine * self._phase_sin,
            cosine * self._phase_cos - sine * self._phase_sin,
        ), dim=-1)
        if self.profile.gait_clock_mode == "command_fixed":
            active = (command[:, :3] != 0).any(dim=-1, keepdim=True)
            next_phase = torch.where(active, next_phase, command[:, 3:7])
        return torch.cat((command[:, :3], next_phase), dim=-1)

    def run(
        self,
        start: RolloutStart,
        actor: nn.Module,
        *,
        window_steps: int,
        horizon_windows: int,
    ) -> RolloutResult:
        """Replan every tick; hold each actor residual for ``window_steps`` ticks."""
        for name, value in (("window_steps", window_steps), ("horizon_windows", horizon_windows)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        observations, history_actions, valid = start.observation_history, start.action_history, start.history_valid_mask
        self._validate_history(observations, history_actions, valid)
        self._validate_context(observations.shape[0], start.errors, start.command, start.previous_residual)
        latent, errors, command = start.latent, start.errors, start.command
        all_errors, deltas, actions, source_actions = [errors], [], [], []
        residuals, risks, commands = [], [], []
        held_residual: torch.Tensor | None = None
        with _frozen_model(self.model):
            for step in range(int(window_steps * horizon_windows)):
                # Step zero already contains the current real observation.
                # At future ticks, decode the just-advanced prior and pair it
                # with the preceding clipped SOURCE command, never ENV action.
                if step:
                    observations = torch.cat((observations[:, 1:], self.model.decode(latent).unsqueeze(1)), dim=1)
                    valid = torch.cat((valid[:, 1:], torch.ones_like(valid[:, :1])), dim=1)
                    history_actions = torch.cat((history_actions[:, 1:], source_actions[-1].unsqueeze(1)), dim=1)
                    observations = observations.masked_fill(~valid.unsqueeze(-1), 0.)
                    history_actions = history_actions.masked_fill(~valid.unsqueeze(-1), 0.)
                    command = self._advance_command(command)
                if step % window_steps == 0:
                    held_residual = actor(latent.feature(), errors, command)
                    residuals.append(held_residual)
                assert held_residual is not None
                nominal = self.policy.predict_planner_future(
                    observations, history_actions, command, history_valid_mask=valid,
                )
                adapted = compose_joint_position_residual(
                    nominal, held_residual, ctrl_dt=self.profile.ctrl_dt,
                    position_scale=self.position_scale, velocity_scale=self.velocity_scale,
                )
                chunk = invoke_fada_idm(
                    self.policy.idm, self.spec, observations, history_actions,
                    command, adapted, history_valid_mask=valid,
                )
                source_action = chunk[:, 0].clamp(-self.profile.clip_actions, self.profile.clip_actions)
                action = self.source_to_env(source_action)
                delta = self.model.motion(latent, errors[:, 1:2], action)
                risk = self.model.risk(latent, action).sigmoid()
                # Motion is already path-frame: do not rotate it a second time.
                errors = torch.stack((errors[:, 0] + delta[:, 1], wrap_to_pi(errors[:, 1] + delta[:, 2])), dim=-1)
                latent, _ = self.model.prior(latent, action)
                all_errors.append(errors)
                deltas.append(delta)
                actions.append(action)
                source_actions.append(source_action)
                risks.append(risk)
                commands.append(command)
        return RolloutResult(
            errors=torch.stack(all_errors, dim=1), path_delta=torch.stack(deltas, dim=1),
            actions=torch.stack(actions, dim=1), source_actions=torch.stack(source_actions, dim=1),
            residuals=torch.stack(residuals, dim=1), risk=torch.stack(risks, dim=1),
            commands=torch.stack(commands, dim=1), final_latent=latent,
        )
