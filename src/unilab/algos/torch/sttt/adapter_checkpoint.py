"""Adapter-only persistence, bound to source policy, world model and control contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from unilab.algos.torch.distill.fada.adaptation import fada_adapter_named_parameters


SPATIAL_ADAPTER_VERSION = 2


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bind_spatial_adapter_identity(controller: Any, source_checkpoint: str | Path) -> None:
    """Cold-path identity; excludes output paths and ephemeral physical histories.

    The config snapshot is taken from ``SpatialAdaptationConfig`` itself, so the
    pose source (``odometry_source``), rig (``sensor``), observation modality
    (``visual_mode``/``imu_noise``) and the alignment contract (gap/age/sample/
    pending limits) travel with the adapter. An adapter trained from another
    source family therefore fails the identity comparison instead of being
    silently reused; there is no compatibility fallback.
    """
    spatial = asdict(controller.spatial_config)
    for key in ("enabled", "world_model_checkpoint", "adapter_checkpoint_path", "save_adapter_path"):
        spatial.pop(key)
    # ``asdict`` already carries odometry_source, sensor rig, visual_mode,
    # imu_noise and the alignment limits; never restate them in a second schema.
    if spatial.get("odometry_source") not in {"simulation", "cuvslam"}:
        raise ValueError(
            "Spatial adapter identity requires a known odometry_source, "
            f"got {spatial.get('odometry_source')!r}"
        )
    controller.adapter_identity = json.loads(json.dumps({
        "source_policy_sha256": _sha256(source_checkpoint),
        "world_model_sha256": _sha256(controller.world_model_checkpoint_path),
        "data_contract": controller.world_model_data_contract,
        "runtime_spec": asdict(controller.runtime_spec),
        "lora_config": asdict(controller.updater.snapshot_updater.lora_config),
        "online_config": asdict(controller.online_config),
        "spatial_config": spatial,
        "method": "measured_endpoint_local_gradient_v2",
    }, allow_nan=False))


def _check_finite(value: Any) -> None:
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError("Adapter state contains non-finite tensors")
    elif isinstance(value, dict):
        for item in value.values():
            _check_finite(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _check_finite(item)
    elif isinstance(value, float) and not torch.isfinite(torch.tensor(value)):
        raise ValueError("Adapter state contains a non-finite scalar")


def save_spatial_adapter(controller: Any, path: str | Path) -> Path:
    """Save LoRA/Adam/anchor only; resume always starts fresh physical histories.

    No tracker, pose or control-interval history is written: the restored
    adapter keeps LoRA weights, Adam state and the deployment anchor, and the
    caller's ``reset()`` drops measurement evidence so new observations are
    never bridged across the saved episode boundary.
    """
    if not hasattr(controller, "adapter_identity"):
        raise ValueError("Bind source/world-model identities before saving a spatial adapter")
    payload = {
        "format_version": SPATIAL_ADAPTER_VERSION,
        "identity": controller.adapter_identity,
        "adapters": {name: value.detach().cpu().clone()
                     for name, value in fada_adapter_named_parameters(controller.policy)},
        "optimizer": controller.updater.optimizer.state_dict(),
        "anchor": {name: value.detach().cpu().clone()
                   for name, value in controller.updater.snapshot_updater.theta_base.items()},
        "completed_updates": controller.completed_updates,
    }
    _check_finite(payload)
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_spatial_adapter(controller: Any, path: str | Path) -> None:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != SPATIAL_ADAPTER_VERSION:
        raise ValueError(
            "Spatial adapter checkpoint was written for another adapter contract "
            f"(format_version!={SPATIAL_ADAPTER_VERSION}); retrain the adapter"
        )
    if payload.get("identity") != getattr(controller, "adapter_identity", None):
        raise ValueError(
            "Spatial adapter source/model/control identity mismatch: the checkpoint was "
            "trained from a different policy, world model, pose source, sensor rig, "
            "visual mode or alignment contract"
        )
    parameters = dict(fada_adapter_named_parameters(controller.policy))
    if set(payload["adapters"]) != set(parameters) or set(payload["anchor"]) != set(parameters):
        raise ValueError("Spatial adapter parameter manifest mismatch")
    for name, parameter in parameters.items():
        if payload["adapters"][name].shape != parameter.shape or payload["anchor"][name].shape != parameter.shape:
            raise ValueError(f"Spatial adapter shape mismatch: {name}")
    groups = controller.updater.optimizer.param_groups
    stored = payload["optimizer"]["param_groups"]
    if len(stored) != len(groups) or any(
        len(old["params"]) != len(new["params"]) for old, new in zip(stored, groups, strict=True)
    ):
        raise ValueError("Spatial adapter optimizer parameter grouping mismatch")
    if not isinstance(payload["completed_updates"], int) or payload["completed_updates"] < 0:
        raise ValueError("Spatial adapter update count must be non-negative")
    _check_finite(payload)
    controller.updater.optimizer.load_state_dict(payload["optimizer"])
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(payload["adapters"][name].to(parameter))
    controller.updater.snapshot_updater.theta_base = {
        name: payload["anchor"][name].to(parameter).clone() for name, parameter in parameters.items()
    }
    controller.completed_updates = payload["completed_updates"]
    controller.reset()
