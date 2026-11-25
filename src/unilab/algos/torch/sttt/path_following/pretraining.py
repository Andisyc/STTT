"""Offline reference-conditioned policy learning and explicit data/model cycles."""
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
from unilab.algos.torch.sttt.path_following.actor_checkpoint import load_actor_checkpoint, save_actor_checkpoint
from unilab.algos.torch.sttt.path_following.actor_learner import ActorLearner
from unilab.algos.torch.sttt.path_following.critic import CriticConfig
from unilab.algos.torch.sttt.path_following.latent_actor import LatentResidualActor
from unilab.algos.torch.sttt.path_following.latent_objectives import LatentCostConfig
from unilab.algos.torch.sttt.path_following.latent_rollout import PlannerTrackerRollout
from unilab.algos.torch.sttt.path_following.trajectory import PathState, ReferenceTrajectory, TrajectoryConfig
from unilab.algos.torch.sttt.world_model.checkpoint import LoadedWorldModel, load_world_model
from unilab.algos.torch.sttt.world_model.pretraining_data import PretrainingBatch, PretrainingSequenceSampler
from unilab.algos.torch.sttt.world_model.training import _restore_rng, _rng_state
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
    max_updates: int = 10000
    batch_size: int = 16
    horizon_windows: int = 2
    learning_rate: float = 1e-4
    max_grad_norm: float = 10.
    checkpoint_every: int = 1000
    seed: int = 0
    microbatch_size: int | None = None
    critic: CriticConfig = field(default_factory=CriticConfig)

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
        if self.microbatch_size is not None and (type(self.microbatch_size) is not int or self.microbatch_size < 1):
            raise ValueError("actor microbatch size must be a positive integer or None")
        if not isinstance(self.critic, CriticConfig):
            raise ValueError("actor training requires a validated critic configuration")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> ActorTrainingConfig:
        raw = dict(values)
        for name in ("residual_limit_rad", "hidden_dims"):
            if name in raw and raw[name] is not None:
                raw[name] = tuple(raw[name])
        if raw.get("residual_limit_rad") is None:
            raise ValueError("set explicit source-order residual_limit_rad before pretraining A")
        if "critic" in raw:
            raw["critic"] = CriticConfig.from_mapping(raw["critic"])
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
    raw = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else dict(cfg)
    allowed = {"world_model_checkpoint", "policy_checkpoint", "data_dirs", "output_dir", "device", "training", "trajectories", "costs", "resume_checkpoint", "initial_actor_checkpoint"}
    if not isinstance(raw, dict) or set(raw) - allowed:
        raise ValueError("unknown offline actor training fields")
    root = Path(root_dir).resolve()
    def resolve(value):
        if not value:
            raise ValueError("provide explicit actor data/checkpoint/output paths")
        path = Path(value).expanduser()
        return path if path.is_absolute() else root / path
    options = ActorTrainingConfig.from_mapping(raw["training"])
    distribution = TrajectoryDistribution.from_mapping(raw.get("trajectories", {}))
    costs = LatentCostConfig(**raw.get("costs", {}))
    device = str(raw.get("device", "cpu"))
    torch.manual_seed(options.seed)
    rng = random.Random(options.seed)
    loaded = load_world_model(resolve(raw["world_model_checkpoint"]), device=device)
    rollout = load_rollout(loaded, resolve(raw["policy_checkpoint"]), device=device)
    loaded.model.eval().requires_grad_(False)
    sampler = PretrainingSequenceSampler([resolve(path) for path in raw["data_dirs"]], rollout_windows=1, seed=options.seed)
    if sampler.contract != loaded.contract:
        raise ValueError("A pretraining data differs from the loaded physical model contract")
    if raw.get("resume_checkpoint") and raw.get("initial_actor_checkpoint"):
        raise ValueError("select exact resume or actor warm start, not both")
    state = {}
    initial = raw.get("resume_checkpoint") or raw.get("initial_actor_checkpoint")
    if initial:
        actor, state = load_actor_checkpoint(resolve(initial), loaded.model, loaded.contract, distribution.base,
                                             require_model_identity=bool(raw.get("resume_checkpoint")),
                                             residual_limits=options.residual_limit_rad)
        if actor.hidden_dims != options.hidden_dims or actor.command_dim != rollout.spec.command_dim:
            raise ValueError("actor architecture differs from requested offline training")
    else:
        actor = LatentResidualActor(loaded.model.feature_dim, options.hidden_dims, options.residual_limit_rad,
                                    command_dim=rollout.spec.command_dim, preview_points=distribution.base.preview_points).to(next(loaded.model.parameters()))
    learner = ActorLearner(actor, rollout, {**asdict(options), "window_steps": loaded.contract["data"]["window_steps"]}, costs)
    signature = {"training": {key: value for key, value in asdict(options).items()
                              if key != "max_updates" and not (key == "microbatch_size" and value is None)},
                 "trajectories": asdict(distribution), "costs": asdict(costs), "device": device}
    updates = 0
    if raw.get("resume_checkpoint"):
        if state["signature"] != signature:
            raise ValueError("actor resume configuration changed")
        sampler.load_state_dict(state["sampler_state"])
        learner.load_state_dict(state["learner_state"])
        rng.setstate(state["task_rng_state"])
        _restore_rng(state, torch.device(device))
        updates = state["optimizer_steps"]
    learner.updates = updates
    if updates >= options.max_updates:
        raise ValueError("max_updates must exceed completed actor updates")
    output = resolve(raw["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("actor training requires a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = None
    metrics = {}
    with (output / "training.jsonl").open("x") as log:
        while updates < options.max_updates:
            batch = sampler.sample(options.batch_size, device=device)
            task = distribution.sample(rng)
            start = actor_training_start(batch, rollout, task, seed=rng.randrange(2**31))
            losses = learner.update(start)
            updates = learner.updates
            if updates % options.checkpoint_every == 0 or updates == options.max_updates:
                metrics = losses
                log.write(json.dumps({"optimizer_steps": updates, "task": asdict(task), **metrics}) + "\n")
                log.flush()
                print(json.dumps({"actor_updates": updates, **metrics}), flush=True)
                checkpoint = save_actor_checkpoint(output / f"actor_{updates:08d}.pt", actor, loaded.model, loaded.contract,
                                                    distribution.base, training_state={"optimizer_steps": updates,
                                                        "sampler_state": sampler.state_dict(),
                                                        "task_rng_state": rng.getstate(), "signature": signature,
                                                        **_rng_state(torch.device(device))}, learner_state=learner.state_dict())
    return {"actor_checkpoint": str(checkpoint), "optimizer_steps": updates, "metrics": metrics}


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
    """Run continuous collection, WM warmup and persistent alternating updates."""
    from unilab.algos.torch.sttt.path_following.joint_training import run_joint_pretraining

    return run_joint_pretraining(cfg, root_dir=root_dir)

