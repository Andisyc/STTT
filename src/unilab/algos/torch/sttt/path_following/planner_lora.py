"""Explicit Planner Q/V LoRA; Tracker and all pretrained weights stay frozen."""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from unilab.algos.torch.distill.fada.adaptation import FADALoRALinear, FADALoRAQVMultiheadAttention


ADAPTER_SUFFIXES = ("lora_A.weight", "lora_B.weight", "lora_q_A.weight", "lora_q_B.weight",
                    "lora_v_A.weight", "lora_v_B.weight")


def adapter_parameters(module: nn.Module) -> dict[str, nn.Parameter]:
    return {name: value for name, value in module.named_parameters() if name.endswith(ADAPTER_SUFFIXES)}


def inject_planner_lora(policy: nn.Module, *, rank: int, alpha: float) -> tuple[str, ...]:
    if adapter_parameters(policy):
        raise ValueError("Planner LoRA requires an unadapted Planner–Tracker checkpoint")
    planner = policy.get_submodule("planner")
    targets = tuple(name for name, module in planner.named_modules()
                    if name and isinstance(module, nn.MultiheadAttention))
    if not targets:
        raise ValueError("Planner exposes no Q/V attention adaptation targets")
    policy.eval().requires_grad_(False)
    for name in targets:
        parent, _, child = name.rpartition(".")
        owner = planner.get_submodule(parent) if parent else planner
        base = planner.get_submodule(name)
        if not isinstance(base, nn.MultiheadAttention):
            raise TypeError(f"Planner attention target changed: {name}")
        setattr(owner, child, FADALoRAQVMultiheadAttention(base, rank=rank, alpha=alpha, dropout=0.).eval())
    expected = {f"planner.{target}.lora_{projection}_{matrix}.weight"
                for target in targets for projection in ("q", "v") for matrix in ("A", "B")}
    if set(adapter_parameters(policy)) != expected:
        raise ValueError("Planner adapter parameters differ from their target manifest")
    return targets


def inject_wm_lora(model: nn.Module, *, rank: int, alpha: float) -> tuple[str, ...]:
    if adapter_parameters(model):
        raise ValueError("WM LoRA requires an unadapted physical model")
    targets = tuple(name for name, module in model.named_modules()
                    if isinstance(module, nn.Linear) and name.split(".", 1)[0] in {"prior_net", "motion_head"})
    if not targets:
        raise ValueError("WM exposes no latent-dynamics/motion adaptation targets")
    model.eval().requires_grad_(False)
    for name in targets:
        parent, _, child = name.rpartition(".")
        owner = model.get_submodule(parent) if parent else model
        base = model.get_submodule(name)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"WM Linear target changed: {name}")
        setattr(owner, child, FADALoRALinear(base, rank=rank, alpha=alpha, dropout=0.).eval())
    return targets


def adapter_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in adapter_parameters(module).items()}


@torch.no_grad()
def load_adapter_state(module: nn.Module, values: Mapping[str, torch.Tensor]) -> None:
    parameters = adapter_parameters(module)
    if set(parameters) != set(values):
        raise ValueError("saved LoRA parameter names differ from their manifest")
    for name, parameter in parameters.items():
        value = values[name]
        if value.shape != parameter.shape or not bool(torch.isfinite(value).all()):
            raise ValueError(f"invalid adapter tensor: {name}")
    for name, parameter in parameters.items():
        parameter.copy_(values[name])
