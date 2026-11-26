"""Action-conditioned categorical RSSM for compact proprioception and path motion."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from typing import Any

import torch
from torch import nn

from .reference import MLP, RSSMPosterior, RSSMPrior, VectorDecoder, categorical_sample
from .types import LatentState


@dataclass(frozen=True)
class WorldModelConfig:
    obs_dim: int = 64
    action_dim: int = 29
    embed_dim: int = 128
    hidden_dim: int = 128
    belief_dim: int = 128
    stoch_groups: int = 8
    stoch_classes: int = 8
    recurrent_model: str = "block_gru"
    num_blocks: int = 4
    dynamics_layers: int = 1
    prior_layers: int = 2
    posterior_layers: int = 1
    encoder_layers: int = 2
    decoder_layers: int = 2
    head_layers: int = 2
    norm_eps: float = 1e-4
    unimix: float = 0.01
    action_scale: tuple[float, ...] | None = None
    ctrl_dt: float = .02

    def __post_init__(self) -> None:
        integer_fields = ("obs_dim", "action_dim", "embed_dim", "hidden_dim", "belief_dim", "stoch_groups", "stoch_classes", "num_blocks", "dynamics_layers", "prior_layers", "posterior_layers", "encoder_layers", "decoder_layers", "head_layers")
        if any(type(getattr(self, key)) is not int or getattr(self, key) < 1 for key in integer_fields):
            raise ValueError("world-model dimensions and layer counts must be positive integers")
        if self.obs_dim not in (64, 66) or self.action_dim != 29:
            raise ValueError("world model requires an admitted 64D/66D observation and 29D ENV actions")
        if self.stoch_classes < 2 or self.recurrent_model not in ("gru", "block_gru"):
            raise ValueError("invalid categorical classes or recurrent model")
        if self.recurrent_model == "block_gru" and self.belief_dim % self.num_blocks:
            raise ValueError("belief_dim must be divisible by num_blocks")
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0 or not math.isfinite(self.unimix) or not 0 <= self.unimix < 1:
            raise ValueError("invalid normalization epsilon or categorical unimix")
        if isinstance(self.ctrl_dt, bool) or not math.isfinite(self.ctrl_dt) or self.ctrl_dt <= 0:
            raise ValueError("world-model ctrl_dt must be positive and finite")
        if self.action_scale is not None:
            if not isinstance(self.action_scale, Sequence) or len(self.action_scale) != self.action_dim or any(isinstance(x, bool) or not math.isfinite(x) or x <= 0 for x in self.action_scale):
                raise ValueError("action_scale must contain 29 finite positive fixed scales")
            object.__setattr__(self, "action_scale", tuple(float(x) for x in self.action_scale))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> WorldModelConfig:
        if not isinstance(value, Mapping) or set(value) - {field.name for field in fields(cls)}:
            raise ValueError("unknown world-model configuration fields")
        return cls(**dict(value))


class LatentWorldModel(nn.Module):
    def __init__(self, config: WorldModelConfig | Mapping[str, Any]):
        super().__init__()
        self.config = config if isinstance(config, WorldModelConfig) else WorldModelConfig.from_mapping(config)
        cfg = self.config
        self.feature_dim = cfg.stoch_groups * cfg.stoch_classes + cfg.belief_dim
        self.register_buffer("action_scale", torch.tensor(cfg.action_scale or (1.,) * cfg.action_dim))
        self.encoder = MLP(cfg.obs_dim + 5, cfg.embed_dim, cfg.hidden_dim, cfg.encoder_layers, cfg.norm_eps)
        self.prior_net = RSSMPrior(state_dim=cfg.stoch_groups * cfg.stoch_classes, action_dim=cfg.action_dim, hidden_dim=cfg.hidden_dim, belief_dim=cfg.belief_dim, groups=cfg.stoch_groups, classes=cfg.stoch_classes, recurrent_model=cfg.recurrent_model, blocks=cfg.num_blocks, layers=cfg.dynamics_layers, prior_layers=cfg.prior_layers, eps=cfg.norm_eps)
        self.posterior_net = RSSMPosterior(cfg.belief_dim, cfg.embed_dim, cfg.hidden_dim, cfg.stoch_groups, cfg.stoch_classes, cfg.posterior_layers, cfg.norm_eps)
        groups = (3, 29, 29, 3) if cfg.obs_dim == 64 else (3, 3, 29, 29, 2)
        self.proprio_decoder = VectorDecoder(self.feature_dim, groups, cfg.hidden_dim, cfg.decoder_layers, cfg.norm_eps)
        self.motion_head = MLP(self.feature_dim + cfg.action_dim, 3, cfg.hidden_dim, cfg.head_layers, cfg.norm_eps)
        self.risk_head = MLP(self.feature_dim + cfg.action_dim, 1, cfg.hidden_dim, cfg.head_layers, cfg.norm_eps)
        # Loss-only physical units; reconstructed from the execution contract.
        self.register_buffer("proprio_loss_weight", torch.ones(cfg.obs_dim), persistent=False)
        self.configure_observation_loss({"term_scale": {
            "base_ang_vel": 1., "projected_gravity": 1., "dof_pos": 1., "dof_vel": 1.,
        }})

    def configure_observation_loss(self, preprocessing: Mapping[str, Any]) -> None:
        """Cold-path conversion of physical tolerances into recorded observation units."""
        scale = preprocessing["term_scale"]
        angular = [1. / (.1 * scale["base_ang_vel"])] * 3
        gravity = [1. / (.05 * scale["projected_gravity"])] * 3
        position = [1. / (.01 * scale["dof_pos"])] * 29
        velocity = [1. / scale["dof_vel"]] * 29
        # Gait phase is advanced analytically in imagination, not decoded.
        weights = (angular + position + velocity + gravity if self.config.obs_dim == 64
                   else angular + gravity + position + velocity + [0., 0.])
        self.proprio_loss_weight.copy_(self.proprio_loss_weight.new_tensor(weights))

    @property
    def model_config(self) -> dict[str, Any]:
        return asdict(self.config)

    def initial(self, batch_size: int, *, device: Any = None, dtype: torch.dtype | None = None) -> LatentState:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        reference = self.action_scale
        options = {"device": reference.device if device is None else device, "dtype": reference.dtype if dtype is None else dtype}
        return LatentState(torch.zeros(batch_size, self.config.stoch_groups, self.config.stoch_classes, **options), torch.zeros(batch_size, self.config.belief_dim, **options))

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape[-1] != self.config.action_dim:
            raise ValueError("model actions must have 29 target-order joint commands")
        return action / self.action_scale.to(action)

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.action_scale.to(action)

    def prior(self, latent: LatentState, action: torch.Tensor) -> tuple[LatentState, torch.Tensor]:
        belief, logits = self.prior_net(latent.state, latent.belief, self.normalize_action(action))
        return LatentState(categorical_sample(logits, self.config.unimix), belief), logits

    def posterior(self, previous: LatentState, previous_action: torch.Tensor, observation: torch.Tensor, *, ego_motion: torch.Tensor | None = None, ego_valid: torch.Tensor | None = None, ego_age_s: torch.Tensor | None = None, is_first: torch.Tensor | None = None) -> tuple[LatentState, torch.Tensor, torch.Tensor]:
        batch = observation.shape[0]
        if observation.shape != (batch, self.config.obs_dim) or previous_action.shape != (batch, self.config.action_dim):
            raise ValueError("posterior observation/action dimensions differ from the model contract")
        if is_first is not None:
            if is_first.shape != (batch,) or is_first.dtype != torch.bool:
                raise ValueError("is_first must be boolean [batch]")
            previous = LatentState(torch.where(is_first[:, None, None], 0., previous.state), torch.where(is_first[:, None], 0., previous.belief))
            previous_action = torch.where(is_first[:, None], 0., previous_action)
        zeros = observation.new_zeros(batch, 1)
        if ego_valid is None:
            valid = torch.zeros(batch, 1, device=observation.device, dtype=torch.bool)
        else:
            if ego_valid.shape != (batch, 1):
                raise ValueError("ego_valid must have shape [batch,1]")
            valid = ego_valid.bool()
        if ego_motion is None:
            if bool(valid.any()):
                raise ValueError("valid egomotion requires an available egomotion value")
            motion = observation.new_zeros(batch, 3)
        else:
            if ego_motion.shape != (batch, 3):
                raise ValueError("ego_motion must have shape [batch,3]")
            motion = torch.where(valid, ego_motion, 0.)
        if ego_age_s is not None and ego_age_s.shape != (batch, 1):
            raise ValueError("ego_age_s must have shape [batch,1]")
        age = zeros if ego_age_s is None else torch.where(valid, ego_age_s, 0.)
        embedding = self.encoder(observation, motion, valid.to(observation), age)
        belief, prior_logits = self.prior_net(previous.state, previous.belief, self.normalize_action(previous_action))
        logits = self.posterior_net(belief, embedding)
        return LatentState(categorical_sample(logits, self.config.unimix), belief), prior_logits, logits

    def decode(self, latent: LatentState) -> torch.Tensor:
        return self.proprio_decoder(latent.feature())

    def motion(self, latent: LatentState, action: torch.Tensor) -> torch.Tensor:
        """Interval-average local velocity integrated over the checkpoint's control period."""
        increment = self.motion_head(latent.feature(), self.normalize_action(action)) * self.config.ctrl_dt
        return torch.cat((increment[..., :2], torch.atan2(increment[..., 2:].sin(), increment[..., 2:].cos())), -1)

    def risk(self, latent: LatentState, action: torch.Tensor) -> torch.Tensor:
        return self.risk_head(latent.feature(), self.normalize_action(action))

    def encode_history(self, observations: torch.Tensor, previous_actions: torch.Tensor, valid_mask: torch.Tensor, *, ego_motion: torch.Tensor | None = None, ego_valid: torch.Tensor | None = None, ego_age_s: torch.Tensor | None = None) -> LatentState:
        if observations.ndim != 3 or observations.shape[-1] != self.config.obs_dim:
            raise ValueError("history observation width differs from the model contract")
        batch, steps = observations.shape[:2]
        if previous_actions.shape != (batch, steps, self.config.action_dim) or valid_mask.shape != (batch, steps) or valid_mask.dtype != torch.bool or steps < 1:
            raise ValueError("history requires matched previous actions and boolean validity")
        for value, width in ((ego_motion, 3), (ego_valid, 1), (ego_age_s, 1)):
            if value is not None and value.shape != (batch, steps, width):
                raise ValueError("history egomotion arrays have incompatible shapes")
        latent = self.initial(batch, device=observations.device, dtype=observations.dtype)
        seen = torch.zeros(batch, device=observations.device, dtype=torch.bool)
        for index in range(steps):
            valid = valid_mask[:, index]
            def masked(value: torch.Tensor | None) -> torch.Tensor | None:
                return None if value is None else torch.where(valid[:, None], value[:, index], 0)
            next_latent, _, _ = self.posterior(latent, masked(previous_actions), masked(observations), ego_motion=masked(ego_motion), ego_valid=masked(ego_valid), ego_age_s=masked(ego_age_s), is_first=valid & ~seen)
            latent = LatentState(torch.where(valid[:, None, None], next_latent.state, latent.state), torch.where(valid[:, None], next_latent.belief, latent.belief))
            seen = seen | valid
        return latent
