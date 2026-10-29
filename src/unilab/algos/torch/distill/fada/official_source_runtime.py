"""Official FADA source-collection resources and DAgger collector worker.

Wraps the pinned upstream ``LeCAR-Lab/FADA`` checkout (commit
``OFFICIAL_FADA_SOURCE_REVISION``) so that official source collection runs inside
the existing UniLab persistent-collector infrastructure. This module owns:

- ``OfficialFADASourceResources``: the resident official stack (resolved upstream
  ``FADAConfig``, training env, expert PPO oracle, ``StrictLabelWorkerClient``
  shadow process, official ``PlannerIDMPolicy`` student, and an upstream
  ``FADATrainer`` constructed with ``optimizer=None`` purely for its rollout /
  strict-shadow collection helpers). No optimizer, warmup, training step, or run
  loop is ever created or invoked here. Resources are constructed lazily by the
  first ``prepare`` request so a worker that is only started never steps or
  allocates the native simulator stack.
- ``OfficialFADACollectorWorker``: the ``DaggerCollectorService`` implementation
  the shared runtime spawns. ``prepare`` performs the weight *layout* check only
  (the shared weight region is still the uninitialized version 0 at prepare
  time, so no values are read) and returns the resolved native identity:
  preprocessing, validated config, model config, checkpoint hashes, and the
  official weak-checkpoint plan. ``collect`` then requires a completed prepare
  plus a strictly synced published weight version, synchronizes the published
  student weights through ``SharedWeightSync``, delegates raw-transition
  production to the sibling ``collect_official_fada_raw_transitions``
  transaction, and atomically persists the artifact through
  ``FADAShardedSourceWriter``.

The collection source, optional weak checkpoint, and optional total raw
transition budget are taken from each ``OfficialFADACollectRequest``; the spec
carries only the collection identity and never a static source.

Importing this module never touches the upstream native simulator stack: the
``holosoma`` package root is located, revision-checked, and imported lazily
inside resource construction. A missing/incomplete checkout, a revision drift,
or a missing expert checkpoint raises an explicit error instead of falling back
to fabricated resources.

Process-lifecycle note: the strict-label shadow process spawned by
``StrictLabelWorkerClient`` makes this worker unsuitable for a daemonic runner,
because daemon processes may not have children. The shared runtime factory
(``build_official_fada_runtime``) therefore runs this worker with
``collector_daemon=False``; on POSIX the runner starts the collector in a new
session so the parent can clean up the whole grandchild group if the worker
fails to exit. This worker itself never creates a process group; it shuts the
shadow client down first on every close path and lets the primary exception
propagate.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from unilab.algos.torch.distill.fada.collection_contract import (
    OFFICIAL_FADA_SCENARIO,
    OFFICIAL_FADA_SOURCE_REVISION,
    OfficialFADACollectionSpec,
    OfficialFADACollectRequest,
    validate_official_fada_preprocessing,
)
from unilab.algos.torch.distill.fada.official_model import (
    OFFICIAL_MODEL_DIMS,
    canonical_official_model_config,
)
from unilab.algos.torch.distill.fada.source_artifact import FADAShardedSourceWriter
from unilab.algos.torch.distill.runtime.async_runtime import (
    DaggerCollectRequest,
    DaggerCollectResult,
    DaggerPrepareRequest,
    DaggerPrepareResult,
)
from unilab.ipc import SharedWeightSync

__all__ = [
    "OfficialFADACollectorWorker",
    "OfficialFADASourceResources",
]

_UPSTREAM_PACKAGE_RELATIVE = Path("src") / "holosoma"
_REQUIRED_UPSTREAM_MODULES: tuple[tuple[str, str], ...] = (
    ("holosoma.fada.planner_idm.config", "holosoma/fada/planner_idm/config.py"),
    ("holosoma.fada.planner_idm.model", "holosoma/fada/planner_idm/model.py"),
    ("holosoma.fada.planner_idm.trainer", "holosoma/fada/planner_idm/trainer.py"),
    ("holosoma.fada.planner_idm.train", "holosoma/fada/planner_idm/train.py"),
    ("holosoma.fada.common.compact_obs", "holosoma/fada/common/compact_obs.py"),
    ("holosoma.fada.common.current_command", "holosoma/fada/common/current_command.py"),
    ("holosoma.fada.common.dataset", "holosoma/fada/common/dataset.py"),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_payload(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _git_output(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    output = completed.stdout.strip()
    return output or None


def _verify_upstream_checkout(root: Path) -> None:
    """Fail closed unless ``root`` is the pinned official FADA checkout."""

    package_root = root / _UPSTREAM_PACKAGE_RELATIVE
    for _module, relative in _REQUIRED_UPSTREAM_MODULES:
        if not (package_root / relative).is_file():
            raise FileNotFoundError(
                "official FADA checkout is missing required module "
                f"{relative!r} under {package_root} (upstream_root={root}, pinned "
                f"revision {OFFICIAL_FADA_SOURCE_REVISION})"
            )
    head = _git_output(root, "rev-parse", "HEAD")
    if head is None:
        raise RuntimeError(
            "official FADA upstream_root is not a git checkout; cannot bind it to the "
            f"pinned revision {OFFICIAL_FADA_SOURCE_REVISION}: {root}"
        )
    if head != OFFICIAL_FADA_SOURCE_REVISION:
        raise RuntimeError(
            "official FADA checkout revision mismatch: "
            f"expected {OFFICIAL_FADA_SOURCE_REVISION}, got {head} at {root}"
        )
    if _git_output(
        root, "diff", "--name-only", OFFICIAL_FADA_SOURCE_REVISION,
        "--", "src/holosoma/holosoma",
    ) is not None:
        raise RuntimeError(
            "official FADA source modules differ from the pinned revision: "
            f"{root} (unrelated logs/docs do not prevent collection)"
        )


def _import_upstream(upstream_root: str) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Resolve, revision-check, and import the pinned official modules lazily."""

    root = Path(upstream_root).expanduser().resolve()
    package_root = root / _UPSTREAM_PACKAGE_RELATIVE
    _verify_upstream_checkout(root)
    path_str = str(package_root)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    try:
        config_module = importlib.import_module("holosoma.fada.planner_idm.config")
        model_module = importlib.import_module("holosoma.fada.planner_idm.model")
        trainer_module = importlib.import_module("holosoma.fada.planner_idm.trainer")
        compact_obs_module = importlib.import_module("holosoma.fada.common.compact_obs")
        command_module = importlib.import_module("holosoma.fada.common.current_command")
        dataset_module = importlib.import_module("holosoma.fada.common.dataset")
    except ImportError as exc:
        raise ImportError(
            "failed to import the pinned official FADA modules from "
            f"{package_root}: {exc}"
        ) from exc
    for name, module in (
        ("holosoma.fada.planner_idm.config", config_module),
        ("holosoma.fada.planner_idm.model", model_module),
        ("holosoma.fada.planner_idm.trainer", trainer_module),
        ("holosoma.fada.common.compact_obs", compact_obs_module),
        ("holosoma.fada.common.current_command", command_module),
        ("holosoma.fada.common.dataset", dataset_module),
    ):
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            raise ImportError(
                f"official FADA module {name} has no file origin; refusing an "
                "unverifiable import"
            )
        resolved = Path(module_file).resolve()
        if package_root not in resolved.parents:
            raise ImportError(
                f"official FADA module {name} resolved to {resolved}, which is outside "
                f"the pinned checkout {package_root}; refusing a foreign module"
            )
    return (
        config_module.FADAConfig,
        trainer_module,
        model_module.build_student_policy,
        dataset_module.ReplayBuffer,
        compact_obs_module.extract_compact_obs,
        command_module.extract_current_command_torch,
        command_module.command_components_for_profile,
    )


class OfficialFADASourceResources:
    """Worker-resident official FADA collection stack.

    Owns the resolved upstream ``FADAConfig``, the official training
    environment and SimulationApp, the expert PPO oracle, the resident
    strict-shadow worker process, the official ``PlannerIDMPolicy`` student, and
    an upstream ``FADATrainer`` used only for its collection helpers. Mirrors
    the bootstrap order of upstream ``train.py``: resolve compact observation
    preprocessing from the expert's saved experiment config, then
    ``build_env_and_expert``, then infer dimensions from the live env, then
    build the strict-label worker, the student, and finally the trainer. No
    parent-style training buffers are allocated; the trainer's rollout helpers
    only read the shared buffer *spec* fields (dims/history/horizon/future-obs
    flag), so per-request collection buffers are made fresh via
    ``_make_collection_buffer`` and no massive static buffers live here.
    ``close()`` shuts the shadow process down before the env/App, preserves the
    primary failure, and tolerates partial construction.
    """

    def __init__(self, spec: OfficialFADACollectionSpec) -> None:
        if not isinstance(spec, OfficialFADACollectionSpec):
            raise TypeError(
                "official FADA resources require an OfficialFADACollectionSpec, "
                f"got {type(spec).__name__}"
            )
        self.spec = spec
        expert_checkpoint = Path(spec.expert_checkpoint).expanduser().resolve()
        if not expert_checkpoint.is_file():
            raise FileNotFoundError(
                f"official FADA expert checkpoint not found: {expert_checkpoint}"
            )

        (
            fada_config_cls,
            trainer_module,
            build_student_policy,
            _replay_buffer_cls,
            extract_compact_obs,
            extract_current_command_torch,
            command_components_for_profile,
        ) = _import_upstream(spec.upstream_root)
        self._close_simulation_app = importlib.import_module(
            "holosoma.utils.sim_utils"
        ).close_simulation_app

        config_overrides = dict(spec.config)
        unknown = sorted(
            name
            for name in config_overrides
            if not any(field.name == name for field in dataclasses.fields(fada_config_cls))
        )
        if unknown:
            raise ValueError(
                f"unknown official FADAConfig fields in spec.config: {unknown}"
            )
        declared_expert = config_overrides.get("expert_checkpoint", str(expert_checkpoint))
        if Path(declared_expert).expanduser().resolve() != expert_checkpoint:
            raise ValueError("spec.config expert checkpoint contradicts the collection identity")
        config_overrides["expert_checkpoint"] = str(expert_checkpoint)
        config_overrides["wandb_mode"] = "disabled"
        try:
            # wandb runs are a training-side concern; collection workers never log.
            cfg = fada_config_cls(**config_overrides)
        except TypeError as exc:
            raise TypeError(
                f"official FADAConfig rejected spec.config: {exc}"
            ) from exc
        cfg = cfg.validate()

        self._closed = False
        self._trainer = None
        self._strict_label_worker = None
        self._env = None
        self._simulation_app = None
        self._student_policy = None
        self._expert_policy = None
        self._runtime_log_root = Path(
            tempfile.mkdtemp(prefix="ttt_official_fada_runtime_")
        )
        runtime_log_dir = self._runtime_log_root / "runtime_logs"

        try:
            # train.py bootstrap: compact preprocessing is resolved from the
            # expert checkpoint's own saved experiment config, never from
            # defaults or historical 66/3 experiments.
            train_module = importlib.import_module("holosoma.fada.planner_idm.train")
            (
                term_scale,
                term_noise,
                compact_source_checkpoint,
            ) = train_module._extract_compact_scale_noise_from_expert_checkpoint(cfg)
            cfg.compact_obs_term_scale = term_scale
            cfg.compact_obs_term_noise = term_noise
            self.preprocessing = validate_official_fada_preprocessing(
                {
                    "term_order": ["base_ang_vel", "dof_pos", "dof_vel", "projected_gravity"],
                    "term_scale": term_scale,
                    "term_noise": term_noise,
                    "noise_enabled": False,
                }
            )

            env, simulation_app, expert_policy, resolved_device, experiment_config_payload = (
                trainer_module.build_env_and_expert(
                    cfg,
                    runtime_log_dir=runtime_log_dir,
                    experiment_override_args=list(spec.experiment_override_args),
                )
            )
            self._env = env
            self._simulation_app = simulation_app
            if not isinstance(experiment_config_payload, dict) or not experiment_config_payload:
                raise RuntimeError(
                    "official FADA build_env_and_expert returned no saved experiment "
                    "config payload; refusing to fabricate preprocessing identity"
                )
            self._experiment_config_payload = experiment_config_payload
            self.experiment_config_sha256 = _sha256_payload(experiment_config_payload)
            device = torch.device(resolved_device)

            obs_dim = int(
                extract_compact_obs(env, term_scale=cfg.compact_obs_term_scale).shape[1]
            )
            act_dim = int(env.robot_config.actions_dim)
            cmd_dim = int(
                extract_current_command_torch(env, profile=cfg.command_profile).shape[1]
            )
            resolved_dims = (obs_dim, act_dim, cmd_dim)
            expected_dims = (
                OFFICIAL_MODEL_DIMS["obs_dim"],
                OFFICIAL_MODEL_DIMS["act_dim"],
                OFFICIAL_MODEL_DIMS["cmd_dim"],
            )
            if resolved_dims != expected_dims:
                raise RuntimeError(
                    "official FADA environment dimensions mismatch: expected "
                    f"(obs, act, cmd)={expected_dims}, inferred {resolved_dims}; "
                    "refusing to reinterpret the historical 66/3 route"
                )
            for name, value in (
                ("history_len", int(cfg.history_len)),
                ("pred_horizon", int(cfg.pred_horizon)),
            ):
                expected_value = int(
                    spec.history_length if name == "history_len" else spec.prediction_horizon
                )
                if value != expected_value:
                    raise ValueError(
                        f"official FADA {name} mismatch: spec declares "
                        f"{expected_value}, resolved config has {value}"
                    )

            self._strict_label_worker = trainer_module.StrictLabelWorkerClient(
                cfg=dataclasses.replace(cfg, device=str(device)),
                runtime_log_dir=runtime_log_dir / "strict_shadow_worker",
                experiment_override_args=list(spec.experiment_override_args),
                timeout_s=float(cfg.strict_label_worker_timeout_s),
            )

            student = build_student_policy(
                cfg,
                obs_dim=obs_dim,
                act_dim=act_dim,
                cmd_dim=cmd_dim,
            ).to(device)
            student.eval()
            student.requires_grad_(False)
            self._student_policy = student

            # The trainer's rollout helpers only read the shared buffer *spec*
            # fields (dims/history/horizon/future-obs flag); the actual collection
            # buffer is made fresh per request via ``_make_collection_buffer``.
            # Capacity-1 stand-ins avoid allocating the parent-style replay
            # buffers (~cfg.replay_capacity rows each) that this worker never
            # trains from.
            buffer_spec_kwargs = {
                "capacity": 1,
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "cmd_dim": cmd_dim,
                "history_len": int(cfg.history_len),
                "pred_horizon": int(cfg.pred_horizon),
                "require_future_obs_targets": True,
            }
            offline_buffer = _replay_buffer_cls(growable=True, **buffer_spec_kwargs)
            suboptimal_offline_buffer = _replay_buffer_cls(growable=True, **buffer_spec_kwargs)
            online_buffer = _replay_buffer_cls(
                eviction_policy=cfg.online_buffer_eviction_policy,
                **buffer_spec_kwargs,
            )

            self._trainer = trainer_module.FADATrainer(
                cfg=cfg,
                student_model=student,
                expert_policy=expert_policy,
                offline_buffer=offline_buffer,
                suboptimal_offline_buffer=suboptimal_offline_buffer,
                online_buffer=online_buffer,
                optimizer=None,
                lr_scheduler=None,
                idm_optimizer=None,
                idm_lr_scheduler=None,
                device=device,
                obs_dim=obs_dim,
                act_dim=act_dim,
                cmd_dim=cmd_dim,
                resolved_expert_checkpoint=str(expert_checkpoint),
                compact_obs_source_checkpoint=str(compact_source_checkpoint),
                experiment_config_payload=experiment_config_payload,
                wandb_run=None,
                strict_label_worker=self._strict_label_worker,
            )
            self._expert_policy = expert_policy
            self.expert_checkpoint = str(expert_checkpoint)
            self.expert_checkpoint_sha256 = _sha256_file(expert_checkpoint)
            self.device = device
            self.command_components = tuple(
                command_components_for_profile(cfg.command_profile)
            )
            self.model_config = canonical_official_model_config(
                {
                    key: getattr(cfg, key)
                    for key in (
                        "history_len",
                        "pred_horizon",
                        "planner_d_model",
                        "planner_nhead",
                        "planner_num_layers",
                        "planner_dim_feedforward",
                        "planner_dropout",
                        "planner_use_learned_positional_encoding",
                        "planner_use_action_history",
                        "idm_d_model",
                        "idm_nhead",
                        "idm_encoder_num_layers",
                        "idm_decoder_num_layers",
                        "idm_dim_feedforward",
                        "idm_dropout",
                        "idm_use_learned_positional_encoding",
                        "idm_use_current_command_for_history",
                        "planner_predict_delta",
                    )
                }
            )
            self.selected_weak_checkpoints = self._resolve_weak_checkpoints(
                trainer_module=trainer_module,
                expert_checkpoint=expert_checkpoint,
            )
        except BaseException:
            self._close_resources(raise_errors=False)
            raise

    def _resolve_weak_checkpoints(
        self,
        *,
        trainer_module: Any,
        expert_checkpoint: Path,
    ) -> tuple[Path, ...]:
        """Resolve the official weak-checkpoint plan with upstream selection helpers.

        Uses the pinned ``offline_data_utils`` listing/selection helpers and the
        trainer's reward-curve loader over the expert checkpoint's run directory,
        mirroring upstream ``_collect_suboptimal_offline_data``; the final expert
        checkpoint itself is never reused as a weak substitute. Returns an empty
        plan when the source is effectively disabled or no intermediate
        checkpoints exist.
        """

        cfg = self.cfg
        if not bool(getattr(self._trainer, "suboptimal_enabled", False)):
            return ()
        from holosoma.fada.planner_idm.offline_data_utils import (
            list_model_checkpoints,
            select_suboptimal_checkpoints,
        )

        checkpoint_dir = expert_checkpoint.parent
        checkpoints = list_model_checkpoints(checkpoint_dir)
        if len(checkpoints) <= 1:
            return ()
        reward_map = self._trainer._load_reward_curve(checkpoint_dir)
        selected = select_suboptimal_checkpoints(
            checkpoints,
            n_target=int(cfg.suboptimal_num_checkpoints),
            sampling_mode=str(cfg.suboptimal_checkpoint_sampling),
            reward_map=reward_map,
        )
        return tuple(
            Path(checkpoint).expanduser().resolve() for checkpoint in selected
        )

    @property
    def cfg(self) -> Any:
        """Resolved upstream ``FADAConfig`` (compact preprocessing bound)."""
        if self._trainer is None:
            raise RuntimeError("official FADA resources are closed")
        return self._trainer.cfg

    @property
    def env(self) -> Any:
        if self._env is None:
            raise RuntimeError("official FADA environment is closed")
        return self._env

    @property
    def trainer(self) -> Any:
        if self._trainer is None:
            raise RuntimeError("official FADA trainer resources are closed")
        return self._trainer

    @property
    def student_policy(self) -> Any:
        if self._student_policy is None:
            raise RuntimeError("official FADA student policy is closed")
        return self._student_policy

    @property
    def strict_label_worker(self) -> Any:
        return self._strict_label_worker

    def _close_resources(self, *, raise_errors: bool) -> None:
        """Close the shadow process before env/App exactly once.

        The first failure wins: it is re-raised (or chained into ``close``)
        while every remaining resource still gets its best-effort close.
        """

        errors: list[BaseException] = []
        trainer = self._trainer
        self._trainer = None
        self._student_policy = None
        self._expert_policy = None
        if trainer is not None:
            try:
                trainer.close()
            except BaseException as exc:  # noqa: BLE001 - preserve all close failures
                errors.append(exc)
        strict_label_worker = self._strict_label_worker
        self._strict_label_worker = None
        if strict_label_worker is not None and (
            trainer is None or trainer.strict_label_worker is not strict_label_worker
        ):
            try:
                strict_label_worker.close()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
        env = self._env
        self._env = None
        if env is not None:
            try:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
        simulation_app = self._simulation_app
        self._simulation_app = None
        if simulation_app is not None:
            try:
                self._close_simulation_app(simulation_app)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
        if errors and raise_errors:
            raise errors[0]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_resources(raise_errors=True)


class OfficialFADACollectorWorker:
    """``DaggerCollectorService`` for official FADA raw source collection.

    Constructed inside the spawned (non-daemon) collector process by the shared
    persistent runtime; the weight-sync identity fields are injected by the
    runtime. Construction is cheap: the native official stack stays unbuilt
    until the first ``prepare`` request. ``prepare`` performs the weight
    *layout* check only — the shared region is still the uninitialized version 0
    at that point, so no values are read — and returns the resolved native
    identity. ``collect`` requires a completed prepare plus a published,
    strictly synced weight version (version ≥ 1), and drives the dynamic
    ``OfficialFADACollectRequest`` source/weak-checkpoint/budget through one
    upstream rollout per request.
    """

    def __init__(
        self,
        *,
        spec: OfficialFADACollectionSpec,
        weight_sync_name: str,
        weight_sync_lock: Any,
        weight_param_shapes: Mapping[str, torch.Size],
        performance_enabled: bool = False,
    ) -> None:
        if not isinstance(performance_enabled, bool):
            raise ValueError("performance_enabled must be a boolean")
        self._closed = False
        self._spec = spec
        self._performance_enabled = performance_enabled
        self._weight_param_shapes: dict[str, torch.Size] = dict(weight_param_shapes)
        self._resources: OfficialFADASourceResources | None = None
        self._weight_sync: SharedWeightSync | None = None
        self._prepared = False
        try:
            self._weight_sync = SharedWeightSync(
                dict(weight_param_shapes),
                create=False,
                shm_name=weight_sync_name,
                lock=weight_sync_lock,
            )
        except BaseException:
            self._close_resources(raise_errors=False)
            raise

    @property
    def resources(self) -> OfficialFADASourceResources:
        if self._resources is None:
            raise RuntimeError("official FADA collector worker is not prepared")
        return self._resources

    @property
    def weight_sync(self) -> SharedWeightSync:
        if self._weight_sync is None:
            raise RuntimeError("official FADA collector worker is closed")
        return self._weight_sync

    def prepare(self, request: DaggerPrepareRequest) -> DaggerPrepareResult:
        if self._closed:
            raise RuntimeError("official FADA collector worker is closed")
        performance_metrics: dict[str, float] = {}
        if self._resources is None:
            # perf_counter runs only when profiling is enabled; the off path is
            # allocation/timer-free.
            prepare_started = (
                time.perf_counter() if self._performance_enabled else None
            )
            try:
                self._resources = OfficialFADASourceResources(self._spec)
            except BaseException:
                self._close_resources(raise_errors=False)
                raise
            if self._performance_enabled:
                # Wall time of the actual prepare resource initialization only;
                # the missing-checkout error guard above still fails through.
                performance_metrics["resource_init_seconds"] = (
                    time.perf_counter() - prepare_started
                )
        resources = self._resources
        self._verify_weight_layout(resources)
        self._prepared = True
        metadata = {
            "upstream_revision": OFFICIAL_FADA_SOURCE_REVISION,
            "expert_checkpoint": resources.expert_checkpoint,
            "expert_checkpoint_sha256": resources.expert_checkpoint_sha256,
            "experiment_config_sha256": resources.experiment_config_sha256,
            "preprocessing": resources.preprocessing,
            "resolved_config": dataclasses.asdict(resources.cfg),
            "model_config": dict(resources.model_config),
            "weak_checkpoints": [
                str(path) for path in resources.selected_weak_checkpoints
            ],
        }
        if self._performance_enabled:
            metadata["performance_metrics"] = performance_metrics
        return DaggerPrepareResult(
            request_id=request.request_id,
            worker_pid=os.getpid(),
            metadata=metadata,
        )

    def _verify_weight_layout(self, resources: OfficialFADASourceResources) -> None:
        """Reject any drift between the published layout and the official student.

        Compares shapes only; the shared weight *values* are the uninitialized
        version 0 at prepare time and are never read here. Runs before the first
        ``read_weights_into`` so a stale 66/3 checkpoint can never be partially
        copied into the official PlannerIDMPolicy.
        """

        student = resources.student_policy
        student_state = {
            name: tuple(value.shape) for name, value in student.state_dict().items()
        }
        published = {str(name): tuple(shape) for name, shape in self._weight_param_shapes.items()}
        if student_state != published:
            missing = sorted(set(published) - set(student_state))
            unexpected = sorted(set(student_state) - set(published))
            mismatched = {
                name: (published[name], student_state[name])
                for name in sorted(set(published) & set(student_state))
                if published[name] != student_state[name]
            }
            raise ValueError(
                "official FADA student weight layout mismatch against the published "
                f"SharedWeightSync shapes: missing={missing}, unexpected={unexpected}, "
                f"shape_mismatch={mismatched}"
            )

    def _sync_student_weights(self, request: DaggerCollectRequest) -> int:
        student = self.resources.student_policy
        observed_version = self.weight_sync.read_weights_into(student.state_dict())
        if int(observed_version) != int(request.expected_weight_version):
            raise RuntimeError(
                "official FADA collector observed a stale weight version: "
                f"expected {int(request.expected_weight_version)}, "
                f"observed {int(observed_version)}"
            )
        return int(observed_version)

    def collect(self, request: DaggerCollectRequest) -> DaggerCollectResult:
        if self._closed:
            raise RuntimeError("official FADA collector worker is closed")
        if not self._prepared or self._resources is None:
            raise RuntimeError(
                "official FADA collector requires prepare before collect; "
                "no stepping happens before prepare"
            )
        if not isinstance(request, OfficialFADACollectRequest):
            raise TypeError(
                "official FADA collection requires an OfficialFADACollectRequest with "
                f"an explicit source, got {type(request).__name__}"
            )
        if request.scenario != OFFICIAL_FADA_SCENARIO:
            raise ValueError(
                f"unsupported official FADA scenario: {request.scenario!r}"
            )
        if int(request.expected_weight_version) < 1:
            raise RuntimeError(
                "official FADA collect requires a published, activated checkpoint "
                f"(weight version >= 1), got {int(request.expected_weight_version)}"
            )

        from unilab.algos.torch.distill.fada.official_source_collection import (
            collect_official_fada_raw_transitions,
        )
        from unilab.algos.torch.distill.fada.timing import FADAPerformanceRecorder

        resources = self._resources
        spec = resources.spec
        # CPU wall only: native collect returns CPU validated data, so the recorder
        # times completed transactions, never shadow-GPU CUDA kernels or per-step
        # probes. Constructed only when profiling is enabled.
        recorder = (
            FADAPerformanceRecorder(device="cpu") if self._performance_enabled else None
        )
        started = time.perf_counter()
        observed_version = self._sync_student_weights(request)
        synced = time.perf_counter()

        batch, collected_steps = collect_official_fada_raw_transitions(
            resources,
            source=request.source,
            weak_checkpoint=request.weak_checkpoint,
            collect_steps=request.collect_steps,
            recorder=recorder,
        )
        collected = time.perf_counter()

        producer_pid = os.getpid()
        num_samples = int(batch.num_samples)
        if num_samples <= 0:
            raise RuntimeError(
                "official FADA collection produced an empty batch; refusing to report "
                "a successful artifact"
            )
        with FADAShardedSourceWriter(
            request.output_path,
            official_spec=spec,
        ) as writer:
            writer.append(batch)
            writer.commit(
                metadata={
                    "request_id": request.request_id,
                    "scenario": request.scenario,
                    "iteration": int(request.iteration),
                    "checkpoint_path": request.checkpoint_path,
                    "expected_weight_version": int(request.expected_weight_version),
                    "observed_weight_version": observed_version,
                    "producer_pid": producer_pid,
                    "source": request.source,
                    "weak_checkpoint": (
                        str(request.weak_checkpoint)
                        if request.weak_checkpoint is not None
                        else None
                    ),
                    "sample_unit": "transition",
                    "collected_steps": int(collected_steps),
                    "upstream_revision": OFFICIAL_FADA_SOURCE_REVISION,
                    "expert_checkpoint": resources.expert_checkpoint,
                    "expert_checkpoint_sha256": resources.expert_checkpoint_sha256,
                    "experiment_config_sha256": resources.experiment_config_sha256,
                    "preprocessing": resources.preprocessing,
                }
            )
        written = time.perf_counter()

        metrics: dict[str, Any] = {
            "weight_sync_seconds": synced - started,
            "collect_seconds": collected - synced,
            "artifact_write_seconds": written - collected,
        }
        if recorder is not None:
            # Completed-transaction CPU wall durations measured inside the
            # collection call; appended as flat *_seconds keys only when profiling
            # is enabled. These are children of the existing collect_seconds
            # envelope, not additive.
            wall = recorder.summary()["wall_seconds"]
            for key, name in (
                ("collect.rollout.total_elapsed", "rollout_seconds"),
                ("collect.tensor_pack", "tensor_pack_seconds"),
                ("collect.total_elapsed", "collect_total_seconds"),
            ):
                if key in wall:
                    metrics[name] = float(wall[key])

        return DaggerCollectResult(
            request_id=request.request_id,
            scenario=request.scenario,
            iteration=request.iteration,
            checkpoint_path=request.checkpoint_path,
            output_path=request.output_path,
            expected_weight_version=request.expected_weight_version,
            observed_weight_version=observed_version,
            num_samples=num_samples,
            worker_pid=producer_pid,
            metrics=metrics,
            metadata={
                "source": request.source,
                "weak_checkpoint": (
                    str(request.weak_checkpoint)
                    if request.weak_checkpoint is not None
                    else None
                ),
                "sample_unit": "transition",
                "collected_steps": int(collected_steps),
                "upstream_revision": OFFICIAL_FADA_SOURCE_REVISION,
                "expert_checkpoint": resources.expert_checkpoint,
                "expert_checkpoint_sha256": resources.expert_checkpoint_sha256,
                "experiment_config_sha256": resources.experiment_config_sha256,
                "preprocessing": resources.preprocessing,
            },
        )

    def _close_resources(self, *, raise_errors: bool) -> None:
        errors: list[BaseException] = []
        resources = self._resources
        self._resources = None
        if resources is not None:
            try:
                resources.close()
            except BaseException as exc:  # noqa: BLE001 - close everything, raise first
                errors.append(exc)
        weight_sync = self._weight_sync
        self._weight_sync = None
        if weight_sync is not None:
            try:
                weight_sync.close()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
        if errors and raise_errors:
            raise errors[0]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_resources(raise_errors=True)
