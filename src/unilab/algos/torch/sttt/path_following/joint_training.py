"""Real-simulation PPO pretraining followed by frozen-prior FiLM adaptation."""
from __future__ import annotations

import json
import math
import random
from copy import deepcopy
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, cast
from uuid import uuid4

import numpy as np
import torch

from ..world_model.checkpoint import canonical_contract, load_world_model, save_world_model
from ..world_model.joint_collection import BatchedSequenceCollector, CollectionBlock
from ..world_model.learner import WorldModelLearner
from ..world_model.model import LatentWorldModel, WorldModelConfig
from ..world_model.pretraining_data import PretrainingSequenceSampler
from ..world_model.replay import EpisodeReplay
from ..world_model.sequence_data import _manifest
from ..world_model.training import _data_sources, _restore_rng, _rng_state, _validate
from .actor_checkpoint import load_actor_checkpoint, save_actor_checkpoint
from .context_adaptation import ContextAdaptationConfig, ContextAdapter
from .ppo_training import ResidualPPOConfig, ResidualPPOTrainer
from .residual_env import ResidualControlEnv
from .residual_rewards import ResidualRewardConfig
from .joint_checkpoint import load_joint_checkpoint, save_joint_checkpoint
from .joint_config import JointSchedule, JointTrainingConfig, TrainingVersions
from .joint_data import join_rollout_starts
from .latent_actor import LatentResidualActor
from .pretraining import ActorTrainingConfig, TrajectoryDistribution, capture_configuration


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), allow_nan=False, indent=2) + "\n")
    temporary.replace(path)


def _random_state(device):
    state = cast(tuple, np.random.get_state())
    return {**_rng_state(device), "python_random_state": random.getstate(),
            "numpy_random_state": [state[0], torch.from_numpy(state[1].astype(np.int64)), state[2], state[3], state[4]]}


def _restore_random_state(state, device):
    _restore_rng(state, device)
    random.setstate(state["python_random_state"])
    numpy = state["numpy_random_state"]
    np.random.set_state((numpy[0], numpy[1].numpy().astype(np.uint32), numpy[2], numpy[3], numpy[4]))


def _signature(raw):
    value = deepcopy(raw)
    for key in ("output_dir", "resume_checkpoint"):
        value.pop(key)
    value["schedule"].pop("base_iterations")
    value["schedule"].pop("film_iterations")
    value["actor"]["training"].pop("max_updates")
    return value


class JointTrainer:
    """PPO owns real on-policy records; WM replay never supplies policy rewards."""

    def __init__(self, world_model: WorldModelLearner, actor: ResidualPPOTrainer, replay: EpisodeReplay,
                 schedule: JointSchedule, *, wm_batch_size: int, physical_control_steps: int = 0):
        if actor.rollout.model is not world_model.model:
            raise ValueError("joint learners must share one physical model")
        if type(wm_batch_size) is not int or wm_batch_size < 1:
            raise ValueError("WM effective batch must be positive")
        self.world_model, self.actor, self.replay, self.schedule = world_model, actor, replay, schedule
        self.wm_batch_size = wm_batch_size
        self.base_updates = self.film_updates = self.iteration = 0
        self.physical_control_steps = physical_control_steps
        self.at_boundary = True

    @property
    def versions(self) -> TrainingVersions:
        return TrainingVersions(actor=self.base_updates + self.film_updates, world_model=self.world_model.updates)

    def warmup_update(self):
        if self.iteration or self.actor.stage != "base" or self.world_model.updates >= self.schedule.pretrain_wm_updates:
            raise ValueError("WM warmup must precede real PPO updates")
        self.at_boundary = False
        metrics = self.world_model.update(self.replay.sample(self.wm_batch_size))
        self.at_boundary = True
        return metrics

    def update_iteration(self) -> dict[str, Any]:
        if not self.at_boundary:
            raise ValueError("PPO updates require a complete previous batch")
        self.at_boundary = False
        before = self.actor.env.physical_control_steps
        actor_metrics = self.actor.update(versions=self.versions)
        self.physical_control_steps += self.actor.env.physical_control_steps - before
        wm_metrics = {}
        if self.actor.stage == "base":
            self.base_updates += 1
            for _ in range(self.schedule.wm_updates_per_iteration):
                wm_metrics = self.world_model.update(self.replay.sample(self.wm_batch_size))
            self.actor.env.refresh(adapt=False)
        else:
            self.film_updates += 1
        self.iteration += 1
        self.at_boundary = True
        return {"phase": f"{self.actor.stage}_ppo", "iteration": self.iteration,
                "base_updates": self.base_updates, "film_updates": self.film_updates,
                "actor_updates": self.base_updates + self.film_updates,
                "wm_updates": self.world_model.updates, "physical_control_steps": self.physical_control_steps,
                "replay_transitions": self.replay.transitions, "total_real_records": self.replay.total_collected,
                "actor": actor_metrics, "world_model": wm_metrics}

    def restore_counters(self, state):
        values = [state[name] for name in ("iteration", "base_updates", "film_updates", "physical_control_steps")]
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("invalid two-stage PPO counters")
        self.iteration, self.base_updates, self.film_updates, self.physical_control_steps = values
        if (self.iteration != self.base_updates + self.film_updates
                or self.base_updates > self.schedule.base_iterations or self.film_updates > self.schedule.film_iterations
                or self.actor.updates != (self.base_updates if self.actor.stage == "base" else self.film_updates)):
            raise ValueError("PPO checkpoint counters differ from the selected stage")


class _Collectors:
    def __init__(self, options, distribution, *, root, policy, profile, output, seed, device, source_overrides,
                 resume_states=None, playback_deps=None):
        self.items: list[BatchedSequenceCollector] = []
        self._resources = ExitStack()
        if resume_states is not None and len(resume_states) != len(options.conditions):
            raise ValueError("joint resume physical conditions changed")
        offset = 0
        try:
            for index, condition in enumerate(options.conditions):
                count = options.num_envs // len(options.conditions) + int(index < options.num_envs % len(options.conditions))
                saved = None if resume_states is None else resume_states[index]
                physical_seed = (seed + index + (0 if saved is None else saved["total_steps"])) % 2**32
                source = capture_configuration(root, policy, source_overrides=[*source_overrides, *condition["source_overrides"]],
                                               seed=physical_seed, profile_path=profile)
                collector = BatchedSequenceCollector(source, root_dir=root, output_dir=output / condition["name"],
                    capture=options, trajectory=distribution, num_envs=count, stream_offset=offset, seed=physical_seed,
                    condition=condition["name"], device=device, playback_deps=playback_deps, resume_state=saved)
                self.items.append(collector)
                self._resources.callback(collector.close)
                if collector.contract != self.items[0].contract:
                    raise ValueError("physical conditions changed the policy I/O or execution contract")
                offset += count
        except BaseException:
            self.close()
            raise

    @property
    def contract(self):
        return self.items[0].contract

    def advance(self, steps, replay, *, versions):
        blocks = [collector.advance(steps, replay, versions=versions) for collector in self.items]
        starts = [block.starts for block in blocks if block.starts is not None]
        return CollectionBlock(sum(block.physical_control_steps for block in blocks),
            join_rollout_starts(starts) if starts else None,
            tuple(directory for block in blocks for directory in block.completed_directories),
            tuple(failure for block in blocks for failure in block.failures))

    def bind(self, model, actor):
        for collector in self.items:
            collector.bind(model, actor)

    def flush(self, replay):
        return [directory for collector in self.items for directory in collector.flush(replay)]

    def state_dict(self):
        return [collector.state_dict() for collector in self.items]

    def bind_validation_replay(self, replay):
        for collector in self.items:
            collector.validation_replay = replay

    def close(self):
        self._resources.close()


def _import_datasets(directories, replay):
    for directory in directories:
        root, manifest = _manifest(directory)
        if canonical_contract(manifest["contract"]) != canonical_contract(replay.contract):
            raise ValueError("supplied data and physical collector contracts differ")
        sidecar = root / "capture.json"
        identity = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        if identity.get("split") == "validation":
            raise ValueError("held-out episode prefixes cannot be imported into training replay")
        episode_ids = identity.get("episode_ids", [])
        if episode_ids and len(episode_ids) != len(manifest["sequences"]):
            raise ValueError("capture episode identities do not match sequence shards")
        count = 0
        for index, filename in enumerate(manifest["sequences"]):
            name = episode_ids[index] if episode_ids else str(root / filename)
            with np.load(root / filename, allow_pickle=False) as archive:
                raw = dict(archive)
            count += len(raw["actions"])
            provenance = {key: identity[key] for key in ("versions_actor_wm", "stream_id", "condition") if key in identity}
            if not provenance:
                provenance = {"archive_directory": str(root), **{key: identity[key] for key in ("actor_checkpoint", "world_model_checkpoint") if key in identity}}
            replay.update_episode(name, raw, metadata=provenance)
        if count != manifest["transition_count"]:
            raise ValueError("supplied sequence manifest transition count mismatch")


def _validation_capture(options, distribution, *, root, policy, profile, output, seed, device, overrides,
                        expected_contract, playback_deps):
    validation_options = replace(options, num_envs=len(options.conditions), validation_fraction=0.)
    collectors = _Collectors(validation_options, distribution, root=root, policy=policy, profile=profile,
        output=output, seed=seed, device=device, source_overrides=overrides, playback_deps=playback_deps)
    try:
        if collectors.contract != expected_contract:
            raise ValueError("held-out physical capture changed source execution contract")
        replay = EpisodeReplay(expected_contract, capacity_transitions=max(options.max_episode_steps * (options.validation_episodes + len(options.conditions)), options.max_episode_steps),
                               rollout_windows=1, seed=seed)
        quota = math.ceil(options.validation_episodes / len(options.conditions))
        selected = {condition["name"]: [] for condition in options.conditions}
        while any(len(paths) < quota for paths in selected.values()):
            block = collectors.advance(options.steps_per_iteration, replay, versions=TrainingVersions())
            for directory in block.completed_directories:
                condition = json.loads((Path(directory) / "capture.json").read_text())["condition"]
                if len(selected[condition]) < quota:
                    selected[condition].append(directory)
        return [path for paths in selected.values() for path in paths]
    finally:
        collectors.close()


def run_joint_pretraining(cfg, *, root_dir=".", playback_deps=None):
    from unilab.training import ensure_registries

    options = JointTrainingConfig.from_mapping(cfg)
    raw, root = options.raw, Path(root_dir).resolve()

    def resolve(value):
        if not value:
            raise ValueError("explicit output and checkpoint paths are required")
        path = Path(value).expanduser()
        return (path if path.is_absolute() else root / path).resolve()

    output, policy = resolve(raw["output_dir"]), resolve(raw["policy_checkpoint"])
    if not policy.is_file():
        raise FileNotFoundError(policy)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("training and resume require a new output directory")
    if raw["wm"]["training"]["resume_checkpoint"] is not None:
        raise ValueError("resume the paired top-level training checkpoint")
    actor_options = ActorTrainingConfig.from_mapping(raw["actor"]["training"])
    ppo_config = ResidualPPOConfig(**raw["ppo"])
    context_config = ContextAdaptationConfig(**raw["context_adaptation"])
    rewards = ResidualRewardConfig(**raw["rewards"])
    if actor_options.batch_size != options.capture.num_envs:
        raise ValueError("actor batch_size must match the real collection stream count")
    amplitude = np.broadcast_to(np.asarray(options.capture.residual_amplitude_rad), (29,))
    if np.any(amplitude > np.asarray(actor_options.residual_limit_rad)):
        raise ValueError("capture amplitude exceeds the physical actor residual bounds")
    resumed = load_joint_checkpoint(resolve(raw["resume_checkpoint"])) if raw["resume_checkpoint"] else None
    phase = resumed["actor_learner"]["stage"] if resumed is not None else ("film" if raw["stage"] == "film" else "base")
    if raw["stage"] == "film" and resumed is None and (
            not raw["wm"]["training"]["initial_checkpoint"] or not raw["actor"]["initial_actor_checkpoint"]):
        raise ValueError("stage=film requires the paired base actor and world-model checkpoints")
    signature = _signature(raw)
    if resumed is not None and resumed["signature"] != signature:
        raise ValueError("resume changed the physical, PPO, reward or adaptation contract")
    device, seed = torch.device(raw["device"]), raw["seed"]
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    ensure_registries()
    distribution = TrajectoryDistribution.from_mapping(raw["actor"].get("trajectories", {}))
    profile = str(resolve(raw["profile_path"])) if raw["profile_path"] else None
    output.mkdir(parents=True, exist_ok=True)
    run_id = resumed["run_id"] if resumed is not None else uuid4().hex
    collectors = _Collectors(options.capture, distribution, root=root, policy=policy, profile=profile,
        output=output / "capture", seed=seed, device=str(device), source_overrides=raw["source_overrides"],
        resume_states=None if resumed is None else resumed["state"]["collector_states"], playback_deps=playback_deps)
    try:
        contract = collectors.contract
        if resumed is not None and resumed["contract"] != contract:
            raise ValueError("joint checkpoint belongs to another physical source")
        replay = EpisodeReplay(contract, capacity_transitions=options.replay_capacity,
            rollout_windows=raw["wm"]["training"]["rollout_windows"], seed=seed, device=device)
        validation_replay = EpisodeReplay(contract,
            capacity_transitions=max(options.capture.num_envs * options.capture.max_episode_steps,
                                     int(options.replay_capacity * options.capture.validation_fraction)),
            rollout_windows=raw["wm"]["training"]["rollout_windows"], seed=seed ^ 0x12345, device=device)
        collectors.bind_validation_replay(validation_replay)
        training_dirs, validation_dirs, physical_steps = [], [], 0
        if resumed is not None:
            replay.load_state_dict(resumed["replay"])
            validation_replay.load_state_dict(resumed["state"]["validation_replay"])
            training_dirs, validation_dirs = resumed["state"]["training_dirs"], resumed["state"]["validation_dirs"]
        elif phase == "base" and options.capture.initial:
            if raw["wm"]["data"]["train_dirs"] or raw["wm"]["data"]["validation_dirs"]:
                raise ValueError("select initial capture or existing WM datasets, not both")
            while replay.total_collected < options.capture.initial_min_transitions:
                block = collectors.advance(options.capture.steps_per_iteration, replay, versions=TrainingVersions())
                physical_steps += block.physical_control_steps
                training_dirs.extend(block.completed_directories)
                print(json.dumps({"phase": "initial_capture", "real_records": replay.total_collected,
                                  "target_min_records": options.capture.initial_min_transitions}), flush=True)
            training_dirs.extend(collectors.flush(replay))
            validation_dirs = _validation_capture(options.capture, distribution, root=root, policy=policy, profile=profile,
                output=output / "validation", seed=(seed ^ 0xDEECE66D) % 2**32, device=str(device),
                overrides=raw["source_overrides"], expected_contract=contract, playback_deps=playback_deps)
        elif phase == "base":
            training_dirs = [str(resolve(path)) for path in raw["wm"]["data"]["train_dirs"]]
            validation_dirs = [str(resolve(path)) for path in raw["wm"]["data"]["validation_dirs"]]
            if not training_dirs or not validation_dirs:
                raise ValueError("capture.initial=false requires whole-episode train and validation data")
            _import_datasets(training_dirs, replay)
            if replay.total_collected < options.capture.initial_min_transitions:
                raise ValueError("supplied data are below the configured initial transition quota")
        validation_sampler = None
        validation_indices = []
        validation_seed = (seed ^ 0x5DEECE66D) % 2**63
        if validation_dirs:
            validation_sampler = PretrainingSequenceSampler(validation_dirs,
                rollout_windows=raw["wm"]["training"]["rollout_windows"], seed=seed,
                cache_size=raw["wm"]["training"]["cache_size"])
            if validation_sampler.contract != contract:
                raise ValueError("held-out validation and physical collection contracts differ")
            train_sampler = PretrainingSequenceSampler(training_dirs, rollout_windows=1, seed=seed)
            _, train_fingerprints = _data_sources([Path(path) for path in training_dirs], "train", train_sampler.state_dict()["index"])
            _, validation_fingerprints = _data_sources([Path(path) for path in validation_dirs], "validation", validation_sampler.state_dict()["index"])
            if train_fingerprints & validation_fingerprints:
                raise ValueError("held-out data overlap physical training episodes")
            del train_sampler
            validation_indices = torch.randint(validation_sampler.num_starts,
                (raw["wm"]["training"]["batch_size"] * raw["wm"]["training"]["validation_batches"],),
                generator=torch.Generator().manual_seed(validation_seed)).tolist()
        model_config = WorldModelConfig.from_mapping({**raw["wm"]["model"],
            "obs_dim": contract["data"]["observation_dim"] if raw["wm"]["model"]["obs_dim"] is None else raw["wm"]["model"]["obs_dim"],
            "ctrl_dt": contract["data"]["ctrl_dt"] if raw["wm"]["model"].get("ctrl_dt") is None else raw["wm"]["model"]["ctrl_dt"]})
        if not math.isclose(model_config.ctrl_dt, contract["data"]["ctrl_dt"], rel_tol=0., abs_tol=1e-9):
            raise ValueError("joint WM control period differs from the physical source")
        if resumed is not None:
            if resumed["model_config"] != asdict(model_config):
                raise ValueError("resume changed the world-model architecture")
            model = LatentWorldModel(model_config).to(device=device, dtype=resumed["world_model"]["action_scale"].dtype)
            model.load_state_dict(resumed["world_model"], strict=True)
        elif raw["wm"]["training"]["initial_checkpoint"]:
            loaded = load_world_model(resolve(raw["wm"]["training"]["initial_checkpoint"]), device=device, expected_contract=contract)
            if loaded.model.config != model_config:
                raise ValueError("initial world-model architecture differs")
            model = loaded.model
        else:
            model = LatentWorldModel(model_config).to(device)
        first = collectors.items[0]
        if resumed is not None:
            actor = LatentResidualActor(**resumed["actor_architecture"]).to(next(model.parameters()))
            actor.load_state_dict(resumed["actor"], strict=True)
        elif raw["actor"]["initial_actor_checkpoint"]:
            actor, _ = load_actor_checkpoint(resolve(raw["actor"]["initial_actor_checkpoint"]), model, contract,
                distribution.base, require_model_identity=phase == "film", residual_limits=actor_options.residual_limit_rad)
        else:
            actor = LatentResidualActor(model.feature_dim, actor_options.hidden_dims, actor_options.residual_limit_rad,
                command_dim=first.spec.command_dim, preview_points=distribution.base.preview_points).to(next(model.parameters()))
        if phase == "film" and not actor.adaptation_dim:
            actor.enable_film(context_config.dimension)
        if (phase == "base" and actor.adaptation_dim) or (phase == "film" and actor.adaptation_dim != context_config.dimension):
            raise ValueError("actor architecture differs from the requested training stage")
        collectors.bind(model, actor)
        model.configure_observation_loss(contract["source"]["preprocessing"])
        wm_learner = WorldModelLearner(model, raw["wm"]["training"], raw["wm"]["training"]["loss"])
        if phase == "film":
            model.eval().requires_grad_(False)

        def make_learner(stage):
            actor.train_stage(stage)
            adapter = ContextAdapter(actor, collectors.items[0].rollout, batch_size=options.capture.num_envs,
                window_steps=options.capture.window_steps, config=context_config) if stage == "film" else None
            env = ResidualControlEnv(collectors, replay, actor, rewards=rewards, context_adapter=adapter)
            return ResidualPPOTrainer(actor, env, ppo_config, stage=stage)

        actor_learner = make_learner(phase)
        engine = JointTrainer(wm_learner, actor_learner, replay, options.schedule,
            wm_batch_size=raw["wm"]["training"]["batch_size"], physical_control_steps=physical_steps)
        if resumed is not None:
            wm_learner.load_state_dict(resumed["world_model_learner"])
            actor_learner.load_state_dict(resumed["actor_learner"])
            engine.restore_counters(resumed["state"])
            if phase == "film" and raw["stage"] == "all" and engine.base_updates != options.schedule.base_iterations:
                raise ValueError("cannot extend completed base training after entering FiLM training")
            if validation_sampler is not None:
                validation_sampler.load_state_dict(resumed["state"]["validation_sampler_state"])
            _restore_random_state(resumed["state"]["random_state"], device)
        warmup_best = None if resumed is None else resumed["state"]["warmup_best"]
        warmup_selected = False if resumed is None else resumed["state"]["warmup_selected"]
        best_pairs = {} if resumed is None else dict(resumed["state"]["best_pairs"])
        timing = {"control_step_s": contract["data"]["ctrl_dt"],
                  "residual_window_s": options.capture.window_steps * contract["data"]["ctrl_dt"],
                  "wm_horizon_s": raw["wm"]["training"]["rollout_windows"] * options.capture.window_steps * contract["data"]["ctrl_dt"],
                  "context_horizon_s": context_config.horizon_windows * options.capture.window_steps * contract["data"]["ctrl_dt"]}
        summary = {"format": "sttt_joint_latent_pretraining", "schema_version": 5, "run_id": run_id,
            "complete": False, "training_complete": False, "config": raw, "physical_timing": timing,
            "residual_limits_rad": list(actor_options.residual_limit_rad), "best_pairs": best_pairs,
            "resume_physics": "new episodes with restored PPO/context/replay state", "validation_dirs": validation_dirs}
        print(json.dumps({"phase": "training_configuration", **timing,
                          "residual_limits_rad": list(actor_options.residual_limit_rad),
                          "physical_conditions": [condition["name"] for condition in options.capture.conditions],
                          "evaluation_every_ppo_iterations": options.schedule.evaluation_every}), flush=True)

        def checkpoint():
            if not engine.at_boundary:
                raise RuntimeError("refusing a partial PPO-batch checkpoint")
            state = {"iteration": engine.iteration, "base_updates": engine.base_updates, "film_updates": engine.film_updates,
                "physical_control_steps": engine.physical_control_steps, "training_dirs": training_dirs,
                "validation_dirs": validation_dirs,
                "validation_sampler_state": None if validation_sampler is None else validation_sampler.state_dict(),
                "collector_states": collectors.state_dict(), "random_state": _random_state(device)}
            state.update(validation_replay=validation_replay.state_dict(), warmup_best=warmup_best,
                         warmup_selected=warmup_selected, best_pairs=best_pairs)
            path = save_joint_checkpoint(output / "last.pt", wm_learner, engine.actor, replay,
                                         run_id=run_id, signature=signature, state=state)
            summary.update(joint_checkpoint=str(path), phase=engine.actor.stage, iteration=engine.iteration,
                base_updates=engine.base_updates, film_updates=engine.film_updates,
                actor_updates=engine.base_updates + engine.film_updates, wm_updates=wm_learner.updates,
                physical_control_steps=engine.physical_control_steps,
                total_real_records=replay.total_collected, replay_transitions=replay.transitions)
            _publish(output / "summary.json", summary)

        def export(stage):
            wm_path = save_world_model(output / f"{stage}_world_model.pt", model, contract,
                training_state={"optimizer_steps": wm_learner.updates, "loss_config": wm_learner.loss_config})
            actor_path = save_actor_checkpoint(output / f"{stage}_actor.pt", actor, model, contract, distribution.base,
                training_state={"optimizer_steps": engine.actor.updates, "stage": engine.actor.stage, "rewards": raw["rewards"]},
                learner_state=engine.actor.state_dict())
            return wm_path, actor_path

        def record(log, row):
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            print(json.dumps(row, allow_nan=False), flush=True)

        def validate():
            if validation_sampler is None:
                return None
            return _validate(model, validation_sampler, validation_indices,
                batch_size=raw["wm"]["training"]["microbatch_size"] or raw["wm"]["training"]["batch_size"],
                device=device, seed=validation_seed, loss_config=raw["wm"]["training"]["loss"])

        def periodic_evaluation(row, *, final):
            if engine.iteration % options.schedule.evaluation_every and not final:
                return
            row["validation"] = validate()
            row["heldout_replay_transitions"] = validation_replay.transitions
            if validation_replay.num_starts:
                indices = torch.randint(validation_replay.num_starts,
                    (raw["wm"]["training"]["batch_size"] * raw["wm"]["training"]["validation_batches"],),
                    generator=torch.Generator().manual_seed(validation_seed)).tolist()
                row["recent_validation"] = _validate(model, validation_replay, indices,
                    batch_size=raw["wm"]["training"]["microbatch_size"] or raw["wm"]["training"]["batch_size"],
                    device=device, seed=validation_seed, loss_config=raw["wm"]["training"]["loss"],
                    require_windows=False)
            stage = engine.actor.stage
            prefix = f"checkpoints/{stage}_{engine.iteration:07d}"
            wm_path, actor_path = export(prefix)
            pair = {"world_model_checkpoint": str(wm_path), "actor_checkpoint": str(actor_path),
                    "stage": stage, "iteration": engine.iteration, "wm_updates": wm_learner.updates}
            summary.update(latest_pair=pair, phase=stage, iteration=engine.iteration,
                           base_updates=engine.base_updates, film_updates=engine.film_updates,
                           wm_updates=wm_learner.updates, physical_control_steps=engine.physical_control_steps)
            if raw["evaluation"]["enabled"]:
                from .pretrained_evaluation import evaluate_pretrained_actor
                random_state = _random_state(device)
                try:
                    result = evaluate_pretrained_actor(raw["evaluation"], root_dir=root,
                        output_dir=output / "evaluation" / f"{stage}_{engine.iteration:07d}",
                        policy_checkpoint=policy, profile_path=profile, world_model_checkpoint=wm_path,
                        actor_checkpoint=actor_path, source_overrides=raw["source_overrides"],
                        device=str(device), conditions=options.capture.conditions, playback_deps=playback_deps)
                finally:
                    _restore_random_state(random_state, device)
                row["evaluation"] = result
                mode = "adaptive_context" if actor.adaptation_dim else "pretrained_actor"
                failures = sum(int(run[mode]["failure_event"]) for run in result["runs"])
                tracking = float(np.mean([
                    run[mode]["lateral_rmse_m"] / max(run["nominal"]["lateral_rmse_m"], 1e-6)
                    + max(0., 1. - run[mode]["forward_speed_retention"]
                          / max(run["nominal"]["forward_speed_retention"], 1e-6))
                    for run in result["runs"]]))
                prediction = row["validation"] or row.get("recent_validation")
                if prediction is not None and prediction["selection_score"] is not None:
                    score = [failures, tracking, prediction["selection_score"]]
                    if stage not in best_pairs or score < best_pairs[stage]["score"]:
                        best_pairs[stage] = {**pair, "score": score,
                            "improves_nominal": failures <= sum(int(run["nominal"]["failure_event"]) for run in result["runs"]) and tracking < 1.}
                summary["evaluation"] = result
            _publish(output / "summary.json", summary)

        with (output / "training.jsonl").open("x") as log:
            if phase == "base":
                while wm_learner.updates < options.schedule.pretrain_wm_updates:
                    metrics = engine.warmup_update()
                    final = wm_learner.updates == options.schedule.pretrain_wm_updates
                    if wm_learner.updates % raw["wm"]["training"]["validation_every"] == 0 or final:
                        row = {"phase": "wm_warmup", "wm_updates": wm_learner.updates,
                               "target_wm_updates": options.schedule.pretrain_wm_updates, "actor_updates": 0, "world_model": metrics}
                        row["validation"] = validate()
                        if row["validation"] is not None:
                            score = row["validation"]["selection_score"]
                            if warmup_best is None or score < warmup_best["score"]:
                                warmup_best = {"score": score, "wm_updates": wm_learner.updates,
                                    "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}}
                        record(log, row)
                    if wm_learner.updates % raw["wm"]["training"]["checkpoint_every"] == 0 and not final:
                        checkpoint()
                if not warmup_selected:
                    if warmup_best is not None:
                        model.load_state_dict(warmup_best["state_dict"], strict=True)
                        wm_learner.restart_optimizer()
                        summary["selected_warmup_wm_updates"] = warmup_best["wm_updates"]
                    warmup_selected = True
                    checkpoint()
                engine.actor.env.refresh(adapt=False)
                while engine.base_updates < options.schedule.base_iterations:
                    row = engine.update_iteration()
                    row["target_base_iterations"] = options.schedule.base_iterations
                    periodic_evaluation(row, final=engine.base_updates == options.schedule.base_iterations)
                    record(log, row)
                    if engine.iteration % options.schedule.checkpoint_every == 0 or engine.base_updates == options.schedule.base_iterations:
                        checkpoint()
                if summary.get("latest_pair", {}).get("stage") != "base":
                    row = {"phase": "base_ppo", "iteration": engine.iteration}
                    periodic_evaluation(row, final=True)
                    record(log, row)
                exported_wm, exported_actor = export("base")
                if raw["stage"] == "all":
                    actor.enable_film(context_config.dimension)
                    model.eval().requires_grad_(False)
                    engine.actor = make_learner("film")
                    checkpoint()
            if engine.actor.stage == "film":
                while engine.film_updates < options.schedule.film_iterations:
                    row = engine.update_iteration()
                    row["target_film_iterations"] = options.schedule.film_iterations
                    periodic_evaluation(row, final=engine.film_updates == options.schedule.film_iterations)
                    record(log, row)
                    if engine.iteration % options.schedule.checkpoint_every == 0 or engine.film_updates == options.schedule.film_iterations:
                        checkpoint()
                if summary.get("latest_pair", {}).get("stage") != "film":
                    row = {"phase": "film_ppo", "iteration": engine.iteration}
                    periodic_evaluation(row, final=True)
                    record(log, row)
                exported_wm, exported_actor = export("film")
        collectors.flush(replay)
        summary.update(training_complete=True, world_model_checkpoint=str(exported_wm), actor_checkpoint=str(exported_actor))
        collectors.close()
        if raw["evaluation"]["enabled"]:
            # The final iteration was already evaluated as a paired snapshot.
            summary["evaluation"]["world_model_checkpoint"] = str(exported_wm)
            summary["evaluation"]["actor_checkpoint"] = str(exported_actor)
        summary["complete"] = not raw["evaluation"]["enabled"] or summary["evaluation"]["complete"]
        _publish(output / "summary.json", summary)
        return summary
    finally:
        collectors.close()
