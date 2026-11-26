"""Real-simulation residual PPO entrypoints and causal reference conditioning."""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping
from hydra._internal.config_loader_impl import ConfigLoaderImpl
from hydra._internal.utils import create_config_search_path
from hydra.types import RunMode

import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from unilab.algos.torch.distill.fada.adaptation_checkpoint import load_fada_deployable_policy_checkpoint
from unilab.algos.torch.distill.fada.legacy_deployment import LegacyFADAPlaybackProfile
from unilab.algos.torch.distill.fada.runtime_policy import (
    FADAFutureQuery, OfficialFADAPlaybackProfile, invoke_fada_planner,
)
from unilab.algos.torch.sttt.path_following.latent_rollout import PlannerTrackerRollout
from unilab.algos.torch.sttt.path_following.trajectory import PathState, ReferenceTrajectory, TrajectoryConfig
from unilab.algos.torch.sttt.world_model.checkpoint import LoadedWorldModel
from unilab.algos.torch.sttt.world_model.pretraining_data import PretrainingBatch
from unilab.algos.torch.sttt.world_model.types import RolloutStart


@dataclass(frozen=True)
class TrajectoryDistribution:
    base: TrajectoryConfig = TrajectoryConfig()
    kinds: tuple[str, ...] = ("straight", "circle", "figure_eight", "random")
    speed_range: tuple[float, float] = (.2, .8)
    radius_range: tuple[float, float] = (1., 4.)
    lateral_range: tuple[float, float] = (-.3, .3)
    heading_range: tuple[float, float] = (-.3, .3)

    def __post_init__(self) -> None:
        if not self.kinds or not set(self.kinds) <= {"straight", "circle", "figure_eight", "random"}:
            raise ValueError("provide supported training trajectory families")
        for name in ("speed_range", "radius_range", "lateral_range", "heading_range"):
            bounds = getattr(self, name)
            if len(bounds) != 2 or not all(math.isfinite(x) for x in bounds) or bounds[1] < bounds[0]:
                raise ValueError(f"{name} requires ordered finite bounds")
        if self.speed_range[0] <= 0 or self.radius_range[0] <= 0:
            raise ValueError("training speeds/radii must be positive")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> TrajectoryDistribution:
        raw = dict(values)
        if "base" in raw:
            raw["base"] = TrajectoryConfig.from_mapping(raw["base"])
        for name in ("kinds", "speed_range", "radius_range", "lateral_range", "heading_range"):
            if name in raw:
                raw[name] = tuple(raw[name])
        return cls(**raw)

    def sample(self, rng: random.Random) -> TrajectoryConfig:
        return replace(self.base, kind=rng.choice(self.kinds), speed_m_s=rng.uniform(*self.speed_range),
                       radius_m=rng.uniform(*self.radius_range), initial_lateral_m=rng.uniform(*self.lateral_range),
                       initial_heading_rad=rng.uniform(*self.heading_range))


@dataclass(frozen=True)
class ActorTrainingConfig:
    residual_limit_rad: tuple[float, ...]
    hidden_dims: tuple[int, ...] = (256, 128)
    max_updates: int = 3000
    batch_size: int = 64
    horizon_windows: int = 2
    learning_rate: float = 5e-4
    max_grad_norm: float = 1.
    checkpoint_every: int = 1000
    seed: int = 0

    def __post_init__(self) -> None:
        if len(self.residual_limit_rad) != 29 or any(not math.isfinite(x) or x <= 0 for x in self.residual_limit_rad):
            raise ValueError("A pretraining requires 29 explicit positive residual bounds")
        if not self.hidden_dims:
            raise ValueError("actor hidden dimensions must be nonempty")
        for name in ("max_updates", "batch_size", "horizon_windows", "checkpoint_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("actor seed must be a nonnegative integer")
        if any(not math.isfinite(x) or x <= 0 for x in (self.learning_rate, self.max_grad_norm)):
            raise ValueError("actor learning rate/gradient norm must be positive")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> ActorTrainingConfig:
        raw = dict(values)
        for name in ("residual_limit_rad", "hidden_dims"):
            if name in raw and raw[name] is not None:
                raw[name] = tuple(raw[name])
        if raw.get("residual_limit_rad") is None:
            raise ValueError("set explicit source-order residual_limit_rad before pretraining A")
        return cls(**raw)


def load_rollout(loaded: LoadedWorldModel, policy_checkpoint: str | Path, *, device: str = "cpu") -> PlannerTrackerRollout:
    source = loaded.contract["source"]
    digest_state = hashlib.sha256()
    with Path(policy_checkpoint).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest_state.update(block)
    digest = digest_state.hexdigest()
    if digest != source["checkpoint_sha256"]:
        raise ValueError("Planner–Tracker checkpoint differs from the WM data source")
    policy = load_fada_deployable_policy_checkpoint(policy_checkpoint, device=device).policy
    if loaded.contract["data"]["observation_dim"] == 66:
        profile = LegacyFADAPlaybackProfile.from_source(source["source_profile"]["source_profile"])
        policy.legacy_preprocessing = source["preprocessing"]
    else:
        profile = OfficialFADAPlaybackProfile.from_mapping(source["source_profile"], source_identity=source["source_profile"]["source_identity"])
    policy.to(next(loaded.model.parameters())).eval().requires_grad_(False)
    if asdict(profile) != source["source_profile"]:
        # JSON removes tuple/list differences but retains every physical value.
        if json.loads(json.dumps(asdict(profile))) != source["source_profile"]:
            raise ValueError("reconstructed source execution profile differs")
    return PlannerTrackerRollout(loaded.model, policy, profile, source["preprocessing"],
                                 target_joint_names=source["target_joint_names"],
                                 target_default_joint_angles=source["target_default_joint_angles"],
                                 target_action_scale=source["target_action_scale"])


@torch.no_grad()
def actor_training_start(batch: PretrainingBatch, rollout: PlannerTrackerRollout,
                         trajectory: TrajectoryConfig, *, seed: int) -> RolloutStart:
    """Use observed history only; no future observation or SLAM label enters A."""
    sequence = batch.sequence
    observations = torch.cat((batch.history_observations, sequence["observations"][:, :1]), 1)
    env_actions = torch.cat((batch.history_previous_actions, batch.previous_action[:, None]), 1)
    valid = torch.cat((batch.history_valid_mask, torch.ones_like(batch.observation_mask[:, :1])), 1)
    if rollout.spec.family == "legacy":
        first = valid.long().argmax(1)
        initial = observations[torch.arange(observations.shape[0], device=observations.device), first]
        observations = torch.where(valid[..., None], observations, initial[:, None])
    origins = torch.tensor([origin for _, origin in batch.origins], device=observations.device)
    times = origins[:, None] - observations.shape[1] + 1 + torch.arange(observations.shape[1], device=observations.device)
    source_actions = rollout.env_to_source(env_actions).masked_fill(~(valid & (times > 0))[..., None], 0.)
    reference = ReferenceTrajectory.generate(trajectory, batch_size=observations.shape[0], seed=seed,
                                             device=observations.device, dtype=observations.dtype)
    pose = observations.new_zeros(observations.shape[0], 3)
    progress, _ = reference.project(pose, pose[:, 0])
    path = PathState(reference, pose, progress)
    command = path.command(sequence["commands"][:, 0])
    future = invoke_fada_planner(rollout.policy, rollout.spec, observations, source_actions, command, history_valid_mask=valid)
    query = FADAFutureQuery(observations, source_actions, command, valid, future, rollout.preprocessing, rollout.profile)
    ego = torch.cat((batch.history_ego_motion, sequence["ego_motion"][:, :1]), 1)
    ego_valid = torch.cat((batch.history_ego_valid, sequence["ego_valid"][:, :1]), 1)
    age = torch.cat((batch.history_ego_age_s, sequence["ego_age_s"][:, :1]), 1)
    return rollout.encode_start(query, errors=path.errors(), previous_residual=batch.previous_residual,
                                ego_motion=ego, ego_valid=ego_valid, ego_age_s=age, path_state=path)


def train_latent_actor(cfg: Mapping[str, Any] | DictConfig, *, root_dir: str | Path = ".") -> dict[str, Any]:
    """Train base A with real-simulation PPO through the shared pipeline."""
    from .joint_training import run_joint_pretraining
    raw = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else dict(cfg)
    raw["stage"] = "base"
    return run_joint_pretraining(raw, root_dir=root_dir)


def capture_configuration(root: Path, policy_checkpoint: Path, *, source_overrides: list[str],
                          seed: int, profile_path: str | None = None) -> DictConfig:
    """Compose the production playback owner with clean measurements."""
    loader = ConfigLoaderImpl(config_search_path=create_config_search_path(str(root / "conf/distill")))
    cfg = loader.load_configuration(config_name="config", overrides=[
        "task=g1_walk_flat/mujoco", "recording_profile=straight_payload", "+world_model_collection=default",
        "online_adaptation.enabled=false", "env.runtime_payload.enabled=false", *source_overrides,
    ], run_mode=RunMode.RUN)
    with open_dict(cfg):
        del cfg["hydra"]
    for name, value in {
        "training.play_checkpoint_path": str(policy_checkpoint),
        "training.play_use_checkpoint_profile": profile_path is None,
        "official_playback.profile_path": profile_path,
        "online_adaptation.enabled": False, "interactive.keyboard": False,
        "env.noise_config.level": 0., "env.commands.resampling_time": 0.,
        "env.commands.heading_command": False, "algo.seed": seed,
        "world_model_collection.pose.odometry_source": "simulation",
    }.items():
        OmegaConf.update(cfg, name, value, force_add=True)
    if OmegaConf.select(cfg, "path_following.enabled", default=False):
        raise ValueError("offline capture must not start the online learner")
    return cfg


def run_pretraining_pipeline(cfg: DictConfig | Mapping[str, Any], *, root_dir: str | Path = ".") -> dict[str, Any]:
    """Run supervised WM warmup, real-simulation A PPO, then frozen-prior FiLM PPO."""
    from unilab.algos.torch.sttt.path_following.joint_training import run_joint_pretraining

    return run_joint_pretraining(cfg, root_dir=root_dir)

