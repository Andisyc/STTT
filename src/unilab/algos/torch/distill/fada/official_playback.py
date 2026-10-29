"""Official 64/29/7 FADA playback controller.

Owns the official inference histories (left zero-padded with a boolean valid
suffix), projects physical-state mappings into the checkpoint-bound compact
64D observation, builds the full 7D phase command, and executes one
receding-horizon action per query through the shared runtime-policy seam.
Legacy playback and the legacy observation projections are untouched.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, overload

import torch

from unilab.algos.torch.distill.fada.observation import (
    FADA_OFFICIAL_COMPACT_OBSERVATION_CONTRACT,
    project_official_compact_observation,
)
from unilab.algos.torch.distill.fada.official_model import PlannerIDMPolicy
from unilab.algos.torch.distill.fada.online_data import FADAInferenceSnapshot
from unilab.algos.torch.distill.fada.runtime_policy import (
    OfficialFADAPlaybackProfile,
    get_fada_runtime_spec,
    invoke_fada_policy,
)


class OfficialFADAPlaybackController:
    """Own official FADA execution histories and emit one receding-horizon action.

    Constant permutation/scale/default tensors are allocated exactly once here,
    in controller init; per-step calls only gather, scale, and roll histories.
    All histories stay in source joint order with source-semantics normalized
    actions clipped by the bound profile; target action mapping, actuator
    authority, fault, and delay handling belong to the session/env seam.
    """

    def __init__(
        self,
        policy: PlannerIDMPolicy,
        *,
        profile: OfficialFADAPlaybackProfile,
        target_joint_names: Sequence[str],
        device: str | torch.device,
    ) -> None:
        if not isinstance(policy, PlannerIDMPolicy):
            raise TypeError("official playback requires the official PlannerIDMPolicy")
        if not isinstance(profile, OfficialFADAPlaybackProfile):
            raise TypeError("official playback requires an explicit source-bound control profile")
        self.policy = policy
        self.profile = profile
        self.spec = get_fada_runtime_spec(policy)
        if self.spec.family != "official" or self.spec.observation_contract != FADA_OFFICIAL_COMPACT_OBSERVATION_CONTRACT:
            raise ValueError("official playback requires the official compact runtime spec")
        if self.spec.obs_dim != 64 or self.spec.action_dim != 29 or self.spec.command_dim != 7:
            raise ValueError(
                "official playback requires the 64/29/7 runtime, got "
                f"{self.spec.obs_dim}/{self.spec.action_dim}/{self.spec.command_dim}"
            )
        if policy.training:
            raise ValueError("official playback requires an eval-mode policy; call policy.eval() first")
        preprocessing = self.spec.preprocessing
        if preprocessing is None:
            raise ValueError(
                "official playback requires the loaded checkpoint preprocessing; "
                "no default scales are guessed"
            )
        self.device = torch.device(device)
        # Allocated once; control steps never re-create these tensors.
        self._source_indices = torch.tensor(
            profile.source_indices(target_joint_names), dtype=torch.long, device=self.device
        )
        self._default_joint_pos = torch.tensor(
            profile.default_joint_pos, dtype=torch.float32, device=self.device
        )
        self._preprocessing = dict(preprocessing)
        self._observation_history: torch.Tensor | None = None
        self._action_history: torch.Tensor | None = None
        self._history_valid_mask: torch.Tensor | None = None
        self._pending_reset: torch.Tensor | None = None
        self._pending_action: torch.Tensor | None = None

    def reset(self, done: Any | None = None) -> None:
        """Reset every history, or mark selected rows for reset at their next observation."""

        if done is None:
            self._observation_history = None
            self._action_history = None
            self._history_valid_mask = None
            self._pending_reset = None
            self._pending_action = None
            return
        mask = torch.as_tensor(done, dtype=torch.bool, device=self.device).reshape(-1)
        if (
            self._observation_history is not None
            and mask.numel() != self._observation_history.shape[0]
        ):
            raise ValueError(
                "official playback reset mask batch mismatch: "
                f"expected={self._observation_history.shape[0]} observed={mask.numel()}"
            )
        self._pending_reset = mask

    def project_observation(self, mapping: Mapping[str, Any]) -> torch.Tensor:
        """Project a physical-state mapping to the official compact 64D observation."""

        return self._validate_projected_observation(
            project_official_compact_observation(
                mapping,
                preprocessing=self._preprocessing,
                default_joint_pos=self._default_joint_pos,
                source_indices=self._source_indices,
            )
        )

    def on_env_step(
        self,
        *,
        executed_action: Any,
        next_observation: Any,
        done: Any | None = None,
    ) -> dict[str, float] | None:
        """Uniform post-step hook; offline playback performs no collection or update.

        ``executed_action`` is the post-authority/fault/delay env signal. The
        official action history intentionally keeps the source-semantics
        normalized command (clipped first action), not this signal, per the
        frozen 5A action boundary.
        """

        return None

    @torch.no_grad()
    def act(self, mapping: Mapping[str, Any], command: Any) -> torch.Tensor:
        """Project a physical-state mapping and execute one receding-horizon action."""

        return self._act(self.project_observation(mapping), command)

    @torch.no_grad()
    def act_projected(self, observation: Any, command: Any) -> torch.Tensor:
        """Execute from an observation already projected to the official compact contract."""

        return self._act(self._projected_observation_tensor(observation), command)

    @torch.no_grad()
    def act_projected_with_snapshot(
        self, observation: Any, command: Any
    ) -> tuple[torch.Tensor, FADAInferenceSnapshot]:
        """Execute once and capture the same query without changing plain playback."""

        return self._act(
            self._projected_observation_tensor(observation),
            command,
            capture_snapshot=True,
        )

    @overload
    def _act(
        self, obs: torch.Tensor, command: Any, *, capture_snapshot: Literal[False] = False
    ) -> torch.Tensor: ...

    @overload
    def _act(
        self, obs: torch.Tensor, command: Any, *, capture_snapshot: Literal[True]
    ) -> tuple[torch.Tensor, FADAInferenceSnapshot]: ...

    def _act(
        self, obs: torch.Tensor, command: Any, *, capture_snapshot: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, FADAInferenceSnapshot]:
        cmd = torch.as_tensor(command, dtype=torch.float32, device=self.device)
        if cmd.ndim == 1:
            cmd = cmd.unsqueeze(0)
        expected_command_shape = (obs.shape[0], self.spec.command_dim)
        if tuple(cmd.shape) != expected_command_shape:
            raise ValueError(
                "official playback command shape mismatch: "
                f"expected={expected_command_shape} observed={tuple(cmd.shape)}"
            )
        if not bool(torch.isfinite(cmd).all()):
            raise ValueError("official playback command must contain only finite values")

        # Cold start and per-row reset left-pad with zeros plus a boolean valid
        # suffix; obs history covers through t, action history through t-1.
        self._advance_observation_history(obs)
        assert self._observation_history is not None
        assert self._action_history is not None
        assert self._history_valid_mask is not None

        output = invoke_fada_policy(
            self.policy,
            self.spec,
            self._observation_history,
            self._action_history,
            cmd,
            history_valid_mask=self._history_valid_mask,
        )
        action = output.action.detach()
        expected_action_shape = (obs.shape[0], self.spec.action_dim)
        if tuple(action.shape) != expected_action_shape:
            raise ValueError(
                "official playback action shape mismatch: "
                f"expected={expected_action_shape} observed={tuple(action.shape)}"
            )
        if not bool(torch.isfinite(action).all()):
            raise ValueError("official playback policy produced non-finite actions")
        # Source-semantics clip on the normalized command; only this first
        # action is executed and recorded. The full predicted nominal future
        # and action chunk stay untouched for snapshot/online consumers.
        action = torch.clamp(action, -self.profile.clip_actions, self.profile.clip_actions)
        # Record u_t now; it enters the action history at the next observation
        # advance so the snapshot above keeps actions strictly through t-1 and
        # the appended slot stays aligned with the shifted valid suffix.
        self._pending_action = action
        if capture_snapshot:
            snapshot = FADAInferenceSnapshot(
                observation_history=self._observation_history,
                action_history=self._action_history,
                command=cmd,
                nominal_future=output.predicted_future,
                action_snapshot=action,
                observation_contract=FADA_OFFICIAL_COMPACT_OBSERVATION_CONTRACT,
                history_valid_mask=self._history_valid_mask,
            )
        if capture_snapshot:
            return action, snapshot
        return action

    def _advance_action_history(
        self, action: torch.Tensor, *, exclude: torch.Tensor | None = None
    ) -> None:
        """Shift the action window left and append u_t at the last valid slot.

        Runs after the observation advance so the valid action suffix stays
        aligned with the valid observation suffix: at query time t+1 the slot
        now at -1 holds the most recent valid observation and its preceding
        action u_t. Padding stays exactly zero; the official IDM embeds history
        actions before applying the key-padding mask, so a stale value in a
        padded slot would corrupt the next query. Rows in ``exclude`` (just
        reset) keep their fresh zero history untouched.
        """

        assert self._action_history is not None and self._history_valid_mask is not None
        self._action_history = torch.cat(
            (self._action_history[:, 1:], action.unsqueeze(1)), dim=1
        )
        self._action_history.masked_fill_(~self._history_valid_mask.unsqueeze(-1), 0.0)
        if exclude is not None:
            self._action_history[exclude] = 0.0

    def _projected_observation_tensor(self, observation: Any) -> torch.Tensor:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        return self._validate_projected_observation(obs)

    def _validate_projected_observation(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 2 or obs.shape[1] != self.spec.obs_dim:
            raise ValueError(
                "official playback observation shape mismatch: expected "
                f"[batch, {self.spec.obs_dim}] observed={tuple(obs.shape)}"
            )
        if not bool(torch.isfinite(obs).all()):
            raise ValueError("official playback observation must contain only finite values")
        return obs

    def _advance_observation_history(self, obs: torch.Tensor) -> None:
        batch_size = int(obs.shape[0])
        history_length = self.spec.history_length
        if self._observation_history is None:
            # Cold start: left zero pad, valid only at the current instant.
            self._observation_history = torch.zeros(
                batch_size,
                history_length,
                self.spec.obs_dim,
                device=self.device,
                dtype=obs.dtype,
            )
            self._observation_history[:, -1] = obs
            self._action_history = torch.zeros(
                batch_size,
                history_length,
                self.spec.action_dim,
                device=self.device,
                dtype=obs.dtype,
            )
            self._history_valid_mask = torch.zeros(
                batch_size, history_length, dtype=torch.bool, device=self.device
            )
            self._history_valid_mask[:, -1] = True
            self._pending_reset = None
            return
        if self._observation_history.shape[0] != batch_size:
            raise ValueError(
                "official playback observation batch changed without reset: "
                f"expected={self._observation_history.shape[0]} observed={batch_size}"
            )
        # Masked positions must stay exactly zero (the official Planner/IDM
        # embed history entries before applying the key-padding mask), so roll
        # into a fresh zero buffer instead of a plain cat that would leave a
        # stale value in the slot that just became padded.
        shifted_obs = torch.zeros_like(self._observation_history)
        shifted_obs[:, :-1] = self._observation_history[:, 1:]
        shifted_obs[:, -1] = obs
        self._observation_history = shifted_obs
        assert self._history_valid_mask is not None
        self._history_valid_mask = torch.cat(
            (
                self._history_valid_mask[:, 1:],
                torch.ones(batch_size, 1, dtype=torch.bool, device=self.device),
            ),
            dim=1,
        )
        reset_mask: torch.Tensor | None = None
        if self._pending_reset is not None:
            reset_mask = self._pending_reset
            if bool(reset_mask.any()):
                # Selected-row reset: fresh zero-padded histories valid only at t;
                # histories never cross an episode boundary. The reset rows drop
                # their pending action append entirely.
                self._observation_history[reset_mask] = 0.0
                self._observation_history[reset_mask, -1] = obs[reset_mask]
                assert self._action_history is not None
                self._action_history[reset_mask] = 0.0
                self._history_valid_mask[reset_mask] = False
                self._history_valid_mask[reset_mask, -1] = True
            self._pending_reset = None
        if self._pending_action is not None:
            self._advance_action_history(self._pending_action, exclude=reset_mask)
            self._pending_action = None
