"""Generic behavior distillation entrypoint assembly.

This module keeps live environment sampling in distill-owned helpers and only
assembles the configured entrypoint routes.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.distill import (
    COLLECTOR_REQUEST_STAGE_NAMES,
    DISTILLATION_METRICS_SCHEMA_VERSION,
    BehaviorDistillationTrainer,
    DistillationPerformanceRunContext,
    DistillationStageObservation,
    DistillationTeacherSpec,
    MLPStudentPolicy,
    MoEStudentPolicy,
    RoleArtifactSpec,
    WorkflowDatasetSource,
    WorkflowScenarioCollectionResult,
    WorkflowScenarioSpec,
    WorkflowStudentUpdateResult,
    adopt_legacy_role_artifact,
    build_multitask_distillation_dataset,
    collect_distillation_dataset_from_env,
    collect_transition_distillation_dataset_from_env,
    config_fingerprint,
    file_sha256,
    finalize_workflow_performance,
    fork_workflow_run,
    load_distillation_checkpoint,
    load_distillation_dataset,
    load_distillation_student_policy,
    load_sac_teacher_policy,
    make_fake_distillation_dataset,
    required_balanced_replay_updates,
    resolve_command_intent_rollout_policies,
    resolve_walk_to_stop_role_pair,
    run_bootstrap_workflow,
    run_iterative_dagger_updates,
    run_multirole_dagger_workflow,
    run_offline_distillation_updates,
    save_distillation_dataset,
    validate_sac_teacher_checkpoint_contract,
)
from unilab.algos.torch.distill.entry_collection import (
    _apply_collect_command_distribution_overrides,
    _distill_device,
    _require_teacher_policy_collection_route,
    run_collect_dataset,
    run_online_dagger_update,
)
from unilab.algos.torch.distill.entry_training import (
    _distill_runtime_cfg,
    _teacher_metadata,
    build_distillation_trainer,
    build_student_policy,
    build_teacher_spec,
    resolve_teacher_checkpoint,
    run_fake_batch_update,
    run_formal_offline_dataset_update,
    run_multitask_dataset_assembly,
    run_offline_dataset_update,
)
from unilab.algos.torch.distill.entry_workflow import (
    _probe_torch_serialization_runtime,
    _workflow_role_cfg,
    _workflow_role_entries,
    _workflow_scenario_specs,
    run_single_entry_workflow,
)
from unilab.algos.torch.distill.fada.planner_idm.training import (
    FADAWorkflowDependencies,
    run_fada_training_owner,
)
from unilab.algos.torch.distill.fada.async_runtime import build_official_fada_runtime
from unilab.algos.torch.distill.g1_persistent_worker import (
    build_persistent_g1_distillation_runtime,
)
from unilab.logging import OffPolicyLogger
from unilab.training import BackendAdapter, ExperimentTracker, create_env, ensure_registries
from unilab.training.run import resolve_task_checkpoint_path

ROOT_DIR = Path(__file__).resolve().parents[1]
_CLI_SEQUENCE_SUMMARY_LIMIT = 16


def _compact_cli_result(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {
            str(child_key): _compact_cli_result(child_value, key=str(child_key))
            for child_key, child_value in value.items()
        }
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        if len(value) <= _CLI_SEQUENCE_SUMMARY_LIMIT:
            return [_compact_cli_result(item) for item in value]
        summary: dict[str, Any] = {
            "count": len(value),
            "head": [_compact_cli_result(item) for item in value[:4]],
            "tail": [_compact_cli_result(item) for item in value[-4:]],
        }
        if key in {"role_labels", "command_intents", "offline_balanced_labels"} and all(
            isinstance(item, str) for item in value
        ):
            summary["counts"] = dict(Counter(str(item) for item in value))
        return summary
    return value


def _format_cli_result(result: dict[str, Any]) -> str:
    return json.dumps(_compact_cli_result(result), ensure_ascii=False, sort_keys=True)


def run_fada_training(
    cfg: DictConfig,
    *,
    teacher_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Compose dependencies and delegate the FADA use-case to its owner module."""

    dependencies = FADAWorkflowDependencies(
        build_official_fada_runtime=build_official_fada_runtime,
    )
    return cast(
        dict[str, Any],
        run_fada_training_owner(
            cfg,
            teacher_checkpoint=teacher_checkpoint,
            dependencies=dependencies,
        ),
    )


@hydra.main(config_path="../conf/distill", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Assemble offline, collection, or iterative online DAgger distillation."""

    if bool(OmegaConf.select(cfg, "training.fada.enabled", default=False)):
        print(_format_cli_result(run_fada_training(cfg)))
        return

    if bool(OmegaConf.select(cfg, "training.workflow.enabled", default=False)):
        print(_format_cli_result(run_single_entry_workflow(cfg)))
        return

    if bool(OmegaConf.select(cfg, "training.online_dagger", default=False)):
        checkpoint_path, _run_dir = resolve_teacher_checkpoint(cfg, root_dir=ROOT_DIR)
        if checkpoint_path is None:
            raise FileNotFoundError(
                "No SAC teacher checkpoint resolved for online DAgger. "
                "Set teacher.checkpoint_path or teacher.load_run/teacher.checkpoint."
            )
        print(
            _format_cli_result(
                run_online_dagger_update(
                    cfg,
                    teacher_checkpoint=checkpoint_path,
                )
            )
        )
        return

    multitask_dataset_path = OmegaConf.select(cfg, "training.multitask_dataset_path")
    if multitask_dataset_path not in (None, ""):
        print(
            _format_cli_result(
                run_multitask_dataset_assembly(cfg, dataset_path=multitask_dataset_path)
            )
        )
        return

    collect_dataset_path = OmegaConf.select(cfg, "training.collect_dataset_path")
    if collect_dataset_path not in (None, ""):
        print(_format_cli_result(run_collect_dataset(cfg, dataset_path=collect_dataset_path)))
        return

    checkpoint_path, _run_dir = resolve_teacher_checkpoint(cfg, root_dir=ROOT_DIR)
    if checkpoint_path is None:
        raise FileNotFoundError(
            "No SAC teacher checkpoint resolved for distillation. "
            "Set teacher.load_run/teacher.checkpoint or training.log_root."
        )

    if bool(OmegaConf.select(cfg, "training.dry_run", default=False)):
        print(
            _format_cli_result(
                run_fake_batch_update(
                    cfg,
                    teacher_checkpoint=checkpoint_path,
                    batch_size=int(OmegaConf.select(cfg, "training.dry_run_batch_size", default=8)),
                    max_updates=int(OmegaConf.select(cfg, "training.dry_run_updates", default=1)),
                    checkpoint_path=OmegaConf.select(cfg, "training.dry_run_checkpoint"),
                )
            )
        )
        return

    offline_dataset_path = OmegaConf.select(cfg, "training.offline_dataset_path")
    if offline_dataset_path not in (None, ""):
        if bool(OmegaConf.select(cfg, "training.formal_run", default=False)):
            print(
                _format_cli_result(
                    run_formal_offline_dataset_update(
                        cfg,
                        teacher_checkpoint=checkpoint_path,
                        dataset_path=offline_dataset_path,
                        batch_size=int(
                            OmegaConf.select(cfg, "training.offline_batch_size", default=256)
                        ),
                        max_updates=int(
                            OmegaConf.select(cfg, "training.offline_max_updates", default=1)
                        ),
                        device=_distill_device(cfg),
                    )
                )
            )
            return
        print(
            _format_cli_result(
                run_offline_dataset_update(
                    cfg,
                    teacher_checkpoint=checkpoint_path,
                    dataset_path=offline_dataset_path,
                    batch_size=int(
                        OmegaConf.select(cfg, "training.offline_batch_size", default=256)
                    ),
                    max_updates=int(
                        OmegaConf.select(cfg, "training.offline_max_updates", default=1)
                    ),
                    checkpoint_path=OmegaConf.select(cfg, "training.offline_checkpoint"),
                    device=_distill_device(cfg),
                )
            )
        )
        return

    raise NotImplementedError(
        "No distillation route selected. Use training.online_dagger=true for the live "
        "student-rollout loop, training.collect_dataset_path for dataset collection, "
        "training.dry_run=true "
        "for the fake-batch probe, or set training.offline_dataset_path for saved-dataset "
        "offline updates."
    )


def _run_main_with_native_fail_stop() -> None:
    """Run Hydra and preserve any unhandled diagnostic failure in a core."""

    try:
        main()
    except BaseException:
        if os.environ.get("UNILAB_NATIVE_ABORT_ON_CORRUPTION", "0") == "1":
            sys.stdout.flush()
            sys.stderr.flush()
            os.abort()
        raise


if __name__ == "__main__":
    _run_main_with_native_fail_stop()
