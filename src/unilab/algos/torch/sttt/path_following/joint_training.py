"""PWM-style persistent collection and alternating effective WM/A updates."""
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
from .actor_learner import ActorLearner
from .joint_checkpoint import load_joint_checkpoint, save_joint_checkpoint
from .joint_config import JointSchedule, JointTrainingConfig, TrainingVersions
from .joint_data import join_rollout_starts
from .latent_actor import LatentResidualActor
from .latent_rollout import PlannerTrackerRollout
from .pretraining import ActorTrainingConfig, TrajectoryDistribution, capture_configuration
from .pretraining_calibration import prepare_calibration, response_probe_evidence, validate_calibration_input


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
    value["schedule"].pop("online_iterations")
    value["actor"]["training"].pop("max_updates")
    # Calibrated numbers and evidence hashes are checkpointed separately.
    value["calibration"].pop("response_probe_result")
    return value


class JointTrainer:
    """Optimizer clocks and physical records are distinct; checkpoints require a full boundary."""

    def __init__(self, world_model: WorldModelLearner, actor: ActorLearner, replay: EpisodeReplay,
                 schedule: JointSchedule, *, wm_batch_size: int, physical_control_steps: int = 0):
        if actor.rollout.model is not world_model.model:
            raise ValueError("joint learners must share one physical model")
        if type(wm_batch_size) is not int or wm_batch_size < 1:
            raise ValueError("WM effective batch must be positive")
        self.world_model, self.actor, self.replay, self.schedule = world_model, actor, replay, schedule
        self.wm_batch_size, self.iteration = wm_batch_size, 0
        self.physical_control_steps = physical_control_steps
        self.at_boundary = True

    @property
    def versions(self) -> TrainingVersions:
        return TrainingVersions(actor=self.actor.updates, world_model=self.world_model.updates)

    def warmup_update(self):
        if self.iteration or self.actor.updates or self.world_model.updates >= self.schedule.pretrain_wm_updates:
            raise ValueError("WM warmup must precede actor and joint updates")
        self.at_boundary = False
        metrics = self.world_model.update(self.replay.sample(self.wm_batch_size))
        self.at_boundary = True
        return metrics

    def update_iteration(self, block: CollectionBlock) -> dict[str, Any]:
        if not self.at_boundary or block.starts is None or block.physical_control_steps < 1:
            raise ValueError("joint updates require a completed physical block and owned real-history starts")
        expected_wm = self.schedule.pretrain_wm_updates + self.iteration * self.schedule.wm_updates_per_iteration
        if self.world_model.updates != expected_wm or self.actor.updates != self.iteration * self.schedule.actor_updates_per_iteration:
            raise ValueError("joint optimizer counters differ from the configured schedule")
        self.at_boundary = False
        self.physical_control_steps += block.physical_control_steps
        actor_metrics, wm_metrics = {}, {}
        for _ in range(self.schedule.wm_updates_per_iteration):
            wm_metrics = self.world_model.update(self.replay.sample(self.wm_batch_size))
        for _ in range(self.schedule.actor_updates_per_iteration):
            actor_metrics = self.actor.update(block.starts, block.failures)
        self.iteration += 1
        self.at_boundary = True
        return {"phase": "joint", "iteration": self.iteration, "actor_updates": self.actor.updates,
                "wm_updates": self.world_model.updates, "physical_control_steps": self.physical_control_steps,
                "critic_updates": self.actor.critic_learner.updates,
                "replay_transitions": self.replay.transitions, "total_real_records": self.replay.total_collected,
                "actor": actor_metrics, "world_model": wm_metrics}

    def restore_counters(self, state):
        iteration, physical = state["iteration"], state["physical_control_steps"]
        if any(type(x) is not int or x < 0 for x in (iteration, physical)) or iteration > self.schedule.online_iterations:
            raise ValueError("joint checkpoint has invalid iteration or physical-step counters")
        warmup = self.world_model.updates
        if iteration:
            valid = warmup == self.schedule.pretrain_wm_updates + iteration * self.schedule.wm_updates_per_iteration
        else:
            valid = warmup <= self.schedule.pretrain_wm_updates
        if (not valid or self.actor.updates != iteration * self.schedule.actor_updates_per_iteration
                or state["critic_updates"] != self.actor.critic_learner.updates):
            raise ValueError("paired checkpoint optimizer counters do not describe a complete joint boundary")
        self.iteration, self.physical_control_steps = iteration, physical


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

    def close(self):
        self._resources.close()


def _import_datasets(directories, replay):
    for directory in directories:
        root, manifest = _manifest(directory)
        if canonical_contract(manifest["contract"]) != canonical_contract(replay.contract):
            raise ValueError("supplied data and physical collector contracts differ")
        sidecar = root / "capture.json"
        identity = json.loads(sidecar.read_text()) if sidecar.exists() else {}
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
    validation_options = replace(options, num_envs=len(options.conditions))
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
    raw = options.raw
    root = Path(root_dir).resolve()
    def resolve(value):
        if not value:
            raise ValueError("explicit output and policy checkpoint paths are required")
        path = Path(value).expanduser()
        return (path if path.is_absolute() else root / path).resolve()
    output, policy = resolve(raw["output_dir"]), resolve(raw["policy_checkpoint"])
    if not policy.is_file():
        raise FileNotFoundError(policy)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("joint training, including resume, requires a new output directory")
    if raw["wm"]["training"]["resume_checkpoint"] is not None:
        raise ValueError("use the paired top-level resume_checkpoint for joint training")
    actor_options = ActorTrainingConfig.from_mapping(raw["actor"]["training"])
    if actor_options.batch_size != options.capture.num_envs:
        raise ValueError("joint actor effective batch must equal the number of independent collection streams")
    amplitude = np.broadcast_to(np.asarray(options.capture.residual_amplitude_rad), (29,))
    if np.any(amplitude > np.asarray(actor_options.residual_limit_rad)):
        raise ValueError("initial exploration amplitude exceeds the declared actor residual envelope")
    validate_calibration_input(raw["calibration"], root=root)
    distribution = TrajectoryDistribution.from_mapping(raw["actor"].get("trajectories", {}))
    profile = str(resolve(raw["profile_path"])) if raw["profile_path"] else None
    resumed = load_joint_checkpoint(resolve(raw["resume_checkpoint"])) if raw["resume_checkpoint"] else None
    signature = _signature(raw)
    if resumed is not None and resumed["signature"] != signature:
        raise ValueError("joint resume configuration changed; only total online_iterations may be extended")
    device = torch.device(raw["device"])
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("joint training supports CPU, CUDA or MPS")
    seed = raw["seed"]
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    ensure_registries()
    output.mkdir(parents=True, exist_ok=True)
    run_id = resumed["run_id"] if resumed is not None else uuid4().hex
    collectors = _Collectors(options.capture, distribution, root=root, policy=policy, profile=profile,
        output=output / "capture", seed=seed, device=str(device), source_overrides=raw["source_overrides"],
        resume_states=None if resumed is None else resumed["state"]["collector_states"], playback_deps=playback_deps)
    try:
        contract = collectors.contract
        if resumed is not None and resumed["contract"] != contract:
            raise ValueError("joint checkpoint belongs to another physical source")
        response_probe_evidence(raw["calibration"], contract=contract, residual_limits=actor_options.residual_limit_rad,
                                horizon_steps=options.capture.steps_per_iteration, root=root)
        replay = EpisodeReplay(contract, capacity_transitions=options.replay_capacity,
            rollout_windows=raw["wm"]["training"]["rollout_windows"], seed=seed, device=device)
        training_dirs, validation_dirs = [], []
        physical_steps = 0
        if resumed is not None:
            replay.load_state_dict(resumed["replay"])
            training_dirs, validation_dirs = resumed["state"]["training_dirs"], resumed["state"]["validation_dirs"]
        elif options.capture.initial:
            if raw["wm"]["data"]["train_dirs"] or raw["wm"]["data"]["validation_dirs"]:
                raise ValueError("select fresh initial capture or supplied whole-episode datasets, not both")
            while replay.total_collected < options.capture.initial_min_transitions:
                block = collectors.advance(options.capture.steps_per_iteration, replay, versions=TrainingVersions())
                physical_steps += block.physical_control_steps
                training_dirs.extend(block.completed_directories)
                print(json.dumps({"phase": "initial_capture", "real_records": replay.total_collected,
                                  "target_min_records": options.capture.initial_min_transitions}), flush=True)
            training_dirs.extend(collectors.flush(replay))
            validation_dirs = _validation_capture(options.capture, distribution, root=root, policy=policy, profile=profile,
                output=output / "validation", seed=(seed ^ 0xDEECE66D) % 2**32, device=str(device), overrides=raw["source_overrides"],
                expected_contract=contract, playback_deps=playback_deps)
        else:
            training_dirs = [str(resolve(path)) for path in raw["wm"]["data"]["train_dirs"]]
            validation_dirs = [str(resolve(path)) for path in raw["wm"]["data"]["validation_dirs"]]
            if not training_dirs or not validation_dirs:
                raise ValueError("capture.initial=false requires explicit whole-episode train/validation splits")
            _import_datasets(training_dirs, replay)
            if replay.total_collected < options.capture.initial_min_transitions:
                raise ValueError("supplied data are below initial_min_transitions; collect more data or set the intended quota explicitly")
        validation_sampler = PretrainingSequenceSampler(validation_dirs, rollout_windows=raw["wm"]["training"]["rollout_windows"],
            seed=seed, cache_size=raw["wm"]["training"]["cache_size"])
        if validation_sampler.contract != contract:
            raise ValueError("held-out validation and collector contracts differ")
        train_sampler = PretrainingSequenceSampler(training_dirs, rollout_windows=1, seed=seed)
        _, train_fingerprints = _data_sources([Path(path) for path in training_dirs], "train", train_sampler.state_dict()["index"])
        _, validation_fingerprints = _data_sources([Path(path) for path in validation_dirs], "validation", validation_sampler.state_dict()["index"])
        if train_fingerprints & validation_fingerprints:
            raise ValueError("held-out data overlap physical training episodes or copied shards")
        del train_sampler
        calibration = prepare_calibration(raw["calibration"], raw["actor"]["costs"], directories=training_dirs,
            contract=contract, residual_limits=actor_options.residual_limit_rad, horizon_steps=options.capture.steps_per_iteration,
            root=root, previous=None if resumed is None else resumed["state"]["calibration"])
        _publish(output / "calibration.json", calibration)
        model_values = {**raw["wm"]["model"], "obs_dim": contract["data"]["observation_dim"] if raw["wm"]["model"]["obs_dim"] is None else raw["wm"]["model"]["obs_dim"]}
        model_config = WorldModelConfig.from_mapping(model_values)
        if resumed is not None:
            if resumed["model_config"] != asdict(model_config):
                raise ValueError("joint checkpoint WM architecture changed")
            model = LatentWorldModel(model_config).to(device=device, dtype=resumed["world_model"]["action_scale"].dtype)
            model.load_state_dict(resumed["world_model"], strict=True)
        elif raw["wm"]["training"]["initial_checkpoint"]:
            loaded = load_world_model(resolve(raw["wm"]["training"]["initial_checkpoint"]), device=device, expected_contract=contract)
            if loaded.model.config != model_config:
                raise ValueError("initial WM architecture differs from the requested joint model")
            model = loaded.model
        else:
            model = LatentWorldModel(model_config).to(device)
        source, first = contract["source"], collectors.items[0]
        rollout = PlannerTrackerRollout(model, first.policy, first.profile, first.preprocessing,
            target_joint_names=source["target_joint_names"], target_default_joint_angles=source["target_default_joint_angles"],
            target_action_scale=source["target_action_scale"])
        actor = LatentResidualActor(model.feature_dim, actor_options.hidden_dims, actor_options.residual_limit_rad,
            command_dim=first.spec.command_dim, preview_points=distribution.base.preview_points).to(next(model.parameters()))
        if resumed is not None:
            expected_arch = {"feature_dim": actor.feature_dim, "hidden_dims": actor.hidden_dims, "command_dim": actor.command_dim,
                             "preview_points": actor.preview_points, "residual_limit_rad29": actor.residual_limit_rad29.cpu().tolist()}
            if resumed["actor_architecture"] != expected_arch:
                raise ValueError("joint checkpoint actor architecture or residual bounds changed")
            actor.load_state_dict(resumed["actor"], strict=True)
        elif raw["actor"].get("initial_actor_checkpoint"):
            actor, _ = load_actor_checkpoint(resolve(raw["actor"]["initial_actor_checkpoint"]), model, contract,
                distribution.base, require_model_identity=False, residual_limits=actor_options.residual_limit_rad)
        model.configure_observation_loss(contract["source"]["preprocessing"])
        wm_learner = WorldModelLearner(model, raw["wm"]["training"], raw["wm"]["training"]["loss"])
        actor_learner = ActorLearner(actor, rollout, {**asdict(actor_options), "window_steps": options.capture.window_steps}, calibration["costs"])
        engine = JointTrainer(wm_learner, actor_learner, replay, options.schedule, wm_batch_size=raw["wm"]["training"]["batch_size"], physical_control_steps=physical_steps)
        if resumed is not None:
            wm_learner.load_state_dict(resumed["world_model_learner"])
            actor_learner.load_state_dict(resumed["actor_learner"])
            validation_sampler.load_state_dict(resumed["state"]["validation_sampler_state"])
            engine.restore_counters(resumed["state"])
            _restore_random_state(resumed["state"]["random_state"], device)
        validation_seed = (seed ^ 0x5DEECE66D) % 2**63
        validation_indices = torch.randint(validation_sampler.num_starts,
            (raw["wm"]["training"]["batch_size"] * raw["wm"]["training"]["validation_batches"],),
            generator=torch.Generator().manual_seed(validation_seed)).tolist()
        summary = {"format": "sttt_joint_latent_pretraining", "schema_version": 2, "run_id": run_id,
            "complete": False, "training_complete": False, "config": raw, "calibration": calibration,
            "resume_physics": "new episodes with restored learner/replay/RNG state", "validation_dirs": validation_dirs}
        def checkpoint():
            if not engine.at_boundary:
                raise RuntimeError("refusing a partial joint-iteration checkpoint")
            state = {"iteration": engine.iteration, "physical_control_steps": engine.physical_control_steps,
                "critic_updates": actor_learner.critic_learner.updates,
                "training_dirs": training_dirs, "validation_dirs": validation_dirs,
                "validation_sampler_state": validation_sampler.state_dict(), "calibration": calibration,
                "collector_states": collectors.state_dict(), "random_state": _random_state(device)}
            path = save_joint_checkpoint(output / "last.pt", wm_learner, actor_learner, replay,
                                         run_id=run_id, signature=signature, state=state)
            summary.update(joint_checkpoint=str(path), iteration=engine.iteration, actor_updates=actor_learner.updates,
                critic_updates=actor_learner.critic_learner.updates,
                wm_updates=wm_learner.updates, physical_control_steps=engine.physical_control_steps,
                total_real_records=replay.total_collected, replay_transitions=replay.transitions)
            _publish(output / "summary.json", summary)
        def validate():
            return _validate(model, validation_sampler, validation_indices,
                batch_size=raw["wm"]["training"]["microbatch_size"] or raw["wm"]["training"]["batch_size"],
                device=device, seed=validation_seed, loss_config=raw["wm"]["training"]["loss"])
        with (output / "training.jsonl").open("x") as log:
            while wm_learner.updates < options.schedule.pretrain_wm_updates:
                metrics = engine.warmup_update()
                final = wm_learner.updates == options.schedule.pretrain_wm_updates
                if wm_learner.updates % raw["wm"]["training"]["validation_every"] == 0 or final:
                    row = {"phase": "wm_warmup", "wm_updates": wm_learner.updates, "actor_updates": 0,
                           "physical_control_steps": engine.physical_control_steps, "world_model": metrics, "validation": validate()}
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    print(json.dumps(row), flush=True)
                if wm_learner.updates % raw["wm"]["training"]["checkpoint_every"] == 0 or final:
                    checkpoint()
            if not engine.iteration:
                save_world_model(output / "warmup_world_model.pt", model, contract,
                    training_state={"optimizer_steps": wm_learner.updates, "loss_config": wm_learner.loss_config})
            collectors.bind(model, actor)
            while engine.iteration < options.schedule.online_iterations:
                block = collectors.advance(options.capture.steps_per_iteration, replay, versions=engine.versions)
                row = engine.update_iteration(block)
                final = engine.iteration == options.schedule.online_iterations
                if wm_learner.updates % raw["wm"]["training"]["validation_every"] == 0 or final:
                    row["validation"] = validate()
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps({key: value for key, value in row.items() if key not in {"actor", "world_model"}}), flush=True)
                if engine.iteration % options.schedule.checkpoint_every == 0 or final:
                    checkpoint()
        collectors.flush(replay)
        exported_wm = save_world_model(output / "world_model.pt", model, contract,
            training_state={"optimizer_steps": wm_learner.updates, "loss_config": wm_learner.loss_config})
        exported_actor = save_actor_checkpoint(output / "actor.pt", actor, model, contract, distribution.base,
            training_state={"optimizer_steps": actor_learner.updates, "joint_iteration": engine.iteration},
            learner_state=actor_learner.state_dict())
        summary.update(training_complete=True, world_model_checkpoint=str(exported_wm), actor_checkpoint=str(exported_actor))
        collectors.close()
        if raw["evaluation"]["enabled"]:
            from .pretrained_evaluation import evaluate_pretrained_actor
            summary["evaluation"] = evaluate_pretrained_actor(raw["evaluation"], root_dir=root, output_dir=output / "evaluation",
                policy_checkpoint=policy, profile_path=profile, world_model_checkpoint=exported_wm, actor_checkpoint=exported_actor,
                source_overrides=raw["source_overrides"], device=str(device), conditions=options.capture.conditions)
        summary["complete"] = not raw["evaluation"]["enabled"] or summary["evaluation"]["complete"]
        _publish(output / "summary.json", summary)
        return summary
    finally:
        collectors.close()
