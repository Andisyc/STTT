"""Offline reference-conditioned policy learning and explicit data/model cycles."""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, replace
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
from unilab.algos.torch.sttt.path_following.latent_actor import LatentResidualActor
from unilab.algos.torch.sttt.path_following.latent_objectives import LatentCostConfig, latent_actor_loss
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
    optimizer = torch.optim.AdamW(actor.parameters(), lr=options.learning_rate, weight_decay=0.)
    signature = {"training": {key: value for key, value in asdict(options).items() if key != "max_updates"},
                 "trajectories": asdict(distribution), "costs": asdict(costs), "device": device}
    updates = 0
    if raw.get("resume_checkpoint"):
        if state["signature"] != signature:
            raise ValueError("actor resume configuration changed")
        sampler.load_state_dict(state["sampler_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        rng.setstate(state["task_rng_state"])
        _restore_rng(state, torch.device(device))
        updates = state["optimizer_steps"]
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
            actor.train()
            optimizer.zero_grad(set_to_none=True)
            result = rollout.run(start, actor, window_steps=loaded.contract["data"]["window_steps"], horizon_windows=options.horizon_windows)
            losses = latent_actor_loss(result, start, ctrl_dt=rollout.profile.ctrl_dt,
                                       window_steps=loaded.contract["data"]["window_steps"], config=costs)
            if any(not bool(torch.isfinite(value)) for value in losses.values()):
                raise ValueError("non-finite offline actor cost")
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), options.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            if updates % options.checkpoint_every == 0 or updates == options.max_updates:
                metrics = {name: float(value.detach()) for name, value in losses.items()}
                log.write(json.dumps({"optimizer_steps": updates, "task": asdict(task), **metrics}) + "\n")
                log.flush()
                print(json.dumps({"actor_updates": updates, **metrics}), flush=True)
                checkpoint = save_actor_checkpoint(output / f"actor_{updates:08d}.pt", actor, loaded.model, loaded.contract,
                                                    distribution.base, training_state={"optimizer_steps": updates,
                                                        "optimizer_state": optimizer.state_dict(), "sampler_state": sampler.state_dict(),
                                                        "task_rng_state": rng.getstate(), "signature": signature,
                                                        **_rng_state(torch.device(device))})
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
    """Initial WM/A training followed by explicit collect → WM → A cycles."""
    from unilab.algos.torch.sttt.world_model.sequence_collection import collect_world_model_sequences
    from unilab.algos.torch.sttt.world_model.training import train_world_model
    from unilab.training import ensure_registries

    raw = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else dict(cfg)
    required = {"output_dir", "policy_checkpoint", "profile_path", "device", "seed", "wm", "actor",
                "capture", "source_overrides", "refinement_cycles", "refine_wm_updates", "refine_actor_updates", "evaluation"}
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError("use conf/sttt/latent_pretraining.yaml")
    root = Path(root_dir).resolve()
    def resolve(value):
        if not value:
            raise ValueError("provide explicit pretraining output and policy checkpoint")
        path = Path(value).expanduser()
        return (path if path.is_absolute() else root / path).resolve()
    output, policy = resolve(raw["output_dir"]), resolve(raw["policy_checkpoint"])
    if not policy.is_file():
        raise FileNotFoundError(policy)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("pretraining requires a new output directory")
    if type(raw["seed"]) is not int or raw["seed"] < 0 or type(raw["refinement_cycles"]) is not int or raw["refinement_cycles"] < 0:
        raise ValueError("pretraining seed/cycle count must be nonnegative integers")
    capture = raw["capture"]
    for name in ("num_steps", "window_steps", "initial_train_episodes", "initial_validation_episodes", "episodes_per_cycle"):
        if type(capture[name]) is not int or capture[name] < 1:
            raise ValueError(f"capture.{name} must be a positive integer")
    for name in ("refine_wm_updates", "refine_actor_updates"):
        if type(raw[name]) is not int or raw[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if capture["risk_tilt_limit_rad"] is None or capture["residual_amplitude_rad"] is None:
        raise ValueError("set explicit capture residual and physical risk bounds")
    ActorTrainingConfig.from_mapping(raw["actor"]["training"])
    distribution = TrajectoryDistribution.from_mapping(raw["actor"].get("trajectories", {}))
    rng = random.Random(raw["seed"])
    ensure_registries()
    output.mkdir(parents=True, exist_ok=True)
    train_dirs = [str(resolve(path)) for path in raw["wm"]["data"]["train_dirs"]]
    validation_dirs = [str(resolve(path)) for path in raw["wm"]["data"]["validation_dirs"]]
    profile = str(resolve(raw["profile_path"])) if raw["profile_path"] else None
    episode_index = 0
    def collect_group(name, count, *, actor_checkpoint=None, world_model_checkpoint=None):
        nonlocal episode_index
        directories = []
        for index in range(count):
            seed = raw["seed"] + episode_index
            episode_index += 1
            source = capture_configuration(root, policy, source_overrides=list(raw["source_overrides"]), seed=seed, profile_path=profile)
            directory = output / name / f"episode_{index:04d}"
            task = distribution.sample(rng)
            torch.manual_seed(seed)
            result = collect_world_model_sequences(
                source, root_dir=root, output_dir=directory, num_steps=capture["num_steps"],
                window_steps=capture["window_steps"], residual_amplitude_rad=capture["residual_amplitude_rad"],
                risk_tilt_limit_rad=capture["risk_tilt_limit_rad"], seed=seed, device=raw["device"],
                trajectory=task, actor_checkpoint=actor_checkpoint, world_model_checkpoint=world_model_checkpoint,
            )
            if not result["transitions"] or result["localization_invalidated"] or result["reset_crossed"]:
                raise RuntimeError(f"capture did not produce a usable physical sequence: {result}")
            directories.append(str(directory))
        return directories
    if capture["initial"]:
        if train_dirs or validation_dirs:
            raise ValueError("initial capture and supplied datasets are mutually exclusive")
        train_dirs = collect_group("initial_train", capture["initial_train_episodes"])
        validation_dirs = collect_group("initial_validation", capture["initial_validation_episodes"])
    elif not train_dirs or not validation_dirs:
        raise ValueError("capture.initial=false requires explicit train/validation datasets")
    summary = {"format": "sttt_latent_pretraining", "schema_version": 1, "complete": False, "cycles": [],
               "config": raw, "train_dirs": train_dirs, "validation_dirs": validation_dirs}
    def publish():
        temporary = output / "summary.json.tmp"
        temporary.write_text(json.dumps(summary, allow_nan=False, indent=2) + "\n")
        temporary.replace(output / "summary.json")
    publish()
    previous_wm = previous_actor = None
    for cycle in range(raw["refinement_cycles"] + 1):
        if cycle:
            train_dirs.extend(collect_group(f"cycle_{cycle:03d}/capture", capture["episodes_per_cycle"],
                                            actor_checkpoint=previous_actor, world_model_checkpoint=previous_wm))
        wm_config = dict(raw["wm"])
        wm_config["data"] = {"train_dirs": train_dirs, "validation_dirs": validation_dirs}
        wm_config["output_dir"] = str(output / f"cycle_{cycle:03d}/wm")
        wm_config["device"], wm_config["seed"] = raw["device"], raw["seed"] + cycle
        wm_config["training"] = {**wm_config["training"], "resume_checkpoint": None, "initial_checkpoint": previous_wm}
        if cycle:
            wm_config["training"]["max_updates"] = raw["refine_wm_updates"]
        wm_result = train_world_model(wm_config, root_dir=root)
        previous_wm = wm_result["best_checkpoint_path"]
        actor_config = {**raw["actor"], "world_model_checkpoint": previous_wm, "policy_checkpoint": str(policy),
                        "data_dirs": train_dirs, "device": raw["device"],
                        "output_dir": str(output / f"cycle_{cycle:03d}/actor"),
                        "initial_actor_checkpoint": previous_actor, "resume_checkpoint": None}
        actor_config["training"] = {**actor_config["training"], "seed": raw["seed"] + cycle}
        if cycle:
            actor_config["training"]["max_updates"] = raw["refine_actor_updates"]
        actor_result = train_latent_actor(actor_config, root_dir=root)
        previous_actor = actor_result["actor_checkpoint"]
        summary["cycles"].append({"cycle": cycle, "world_model": wm_result, "actor": actor_result})
        publish()
    summary["world_model_checkpoint"], summary["actor_checkpoint"] = previous_wm, previous_actor
    if previous_wm is None or previous_actor is None:
        raise RuntimeError("pretraining did not produce a paired WM and actor")
    if raw["evaluation"]["enabled"]:
        from unilab.algos.torch.sttt.path_following.pretrained_evaluation import evaluate_pretrained_actor
        summary["evaluation"] = evaluate_pretrained_actor(
            raw["evaluation"], root_dir=root, output_dir=output / "evaluation",
            policy_checkpoint=policy, profile_path=profile, world_model_checkpoint=previous_wm,
            actor_checkpoint=previous_actor, source_overrides=list(raw["source_overrides"]), device=raw["device"],
        )
    summary["training_complete"] = True
    summary["complete"] = not raw["evaluation"]["enabled"] or summary["evaluation"]["complete"]
    publish()
    return summary

