from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterator, cast

import torch
from torch import nn
from torch.nn import functional as F

from unilab.algos.torch.distill.fada.model import FADAPlannerIDMPolicy, first_action_mse
from unilab.algos.torch.distill.fada.official_model import (
    PlannerIDMPolicy as OfficialPlannerIDMPolicy,
)
from unilab.algos.torch.distill.fada.target_data import FADATargetBatch

# Online LoRA ownership spans the legacy and official Planner-IDM policy families.
FADAAdaptablePolicy = FADAPlannerIDMPolicy | OfficialPlannerIDMPolicy


@dataclass(frozen=True)
class FADALoRAConfig:
    """Paper LoRA values plus one explicit repository-owned injection manifest."""

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ()
    adapter_type: str = "qv_attention"
    target_projections: tuple[str, ...] = ("q", "v")

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError(f"LoRA rank must be a positive integer, got {self.rank!r}")
        if not math.isfinite(float(self.alpha)) or float(self.alpha) <= 0.0:
            raise ValueError(f"LoRA alpha must be finite and positive, got {self.alpha!r}")
        if not math.isfinite(float(self.dropout)) or not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError(f"LoRA dropout must be in [0, 1), got {self.dropout!r}")
        if any(not isinstance(name, str) or not name for name in self.target_modules):
            raise ValueError("LoRA target module names must be non-empty strings")
        if len(set(self.target_modules)) != len(self.target_modules):
            raise ValueError("LoRA target module manifest must contain unique names")
        if self.adapter_type == "qv_attention":
            if self.target_projections != ("q", "v"):
                raise ValueError("Q/V LoRA target projections must be exactly ('q', 'v')")
        elif self.adapter_type == "legacy_linear":
            if self.target_projections:
                raise ValueError("legacy Linear LoRA must not declare attention projections")
        else:
            raise ValueError(f"unsupported FADA LoRA adapter type: {self.adapter_type!r}")

    @classmethod
    def legacy(
        cls,
        *,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
        target_modules: tuple[str, ...] = (),
    ) -> FADALoRAConfig:
        return cls(
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules,
            adapter_type="legacy_linear",
            target_projections=(),
        )


class FADALoRALinear(nn.Module):
    """A frozen Linear owner plus a standard input-dropout low-rank branch."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        factory_kwargs = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_A = nn.Linear(base.in_features, rank, bias=False, **factory_kwargs)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False, **factory_kwargs)
        self.adapter_dropout = nn.Dropout(float(dropout))
        self.scaling = float(alpha) / int(rank)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @property
    def weight(self) -> torch.Tensor:
        """Expose the effective weight to PyTorch Transformer eval fast paths."""

        return self.base.weight + (self.lora_B.weight @ self.lora_A.weight) * self.scaling

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.lora_B(self.lora_A(self.adapter_dropout(inputs)))
        return self.base(inputs) + residual * self.scaling


class FADALoRAQVMultiheadAttention(nn.Module):
    """Frozen packed QKV attention with independent trainable Q/V LoRA branches."""

    def __init__(
        self,
        base: nn.MultiheadAttention,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if not base._qkv_same_embed_dim or base.in_proj_weight is None:
            raise ValueError("FADA Q/V LoRA requires packed same-dimension QKV attention")
        if base.in_proj_weight.shape != (3 * base.embed_dim, base.embed_dim):
            raise ValueError("FADA Q/V LoRA packed projection shape is incompatible")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.batch_first = bool(base.batch_first)
        self.in_proj_bias = None  # Disable the encoder fused path, which would bypass LoRA.
        factory_kwargs = {
            "device": base.in_proj_weight.device,
            "dtype": base.in_proj_weight.dtype,
        }
        self.lora_q_A = nn.Linear(base.embed_dim, rank, bias=False, **factory_kwargs)
        self.lora_q_B = nn.Linear(rank, base.embed_dim, bias=False, **factory_kwargs)
        self.lora_v_A = nn.Linear(base.embed_dim, rank, bias=False, **factory_kwargs)
        self.lora_v_B = nn.Linear(rank, base.embed_dim, bias=False, **factory_kwargs)
        self.adapter_dropout = nn.Dropout(float(dropout))
        self.scaling = float(alpha) / int(rank)
        self.register_buffer(
            "_identity_projection",
            torch.eye(base.embed_dim, **factory_kwargs),
            persistent=False,
        )
        nn.init.kaiming_uniform_(self.lora_q_A.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_v_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_q_B.weight)
        nn.init.zeros_(self.lora_v_B.weight)

    def _project(
        self,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        lora_A: nn.Linear,
        lora_B: nn.Linear,
    ) -> torch.Tensor:
        residual = lora_B(lora_A(self.adapter_dropout(inputs)))
        return F.linear(inputs, weight, bias) + residual * self.scaling

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            query, key, value = (tensor.transpose(0, 1) for tensor in (query, key, value))
        q_weight, k_weight, v_weight = self.base.in_proj_weight.chunk(3, dim=0)
        if self.base.in_proj_bias is None:
            q_bias = k_bias = v_bias = None
        else:
            q_bias, k_bias, v_bias = self.base.in_proj_bias.chunk(3, dim=0)
        projected_query = self._project(query, q_weight, q_bias, self.lora_q_A, self.lora_q_B)
        projected_key = F.linear(key, k_weight, k_bias)
        projected_value = self._project(value, v_weight, v_bias, self.lora_v_A, self.lora_v_B)
        identity_projection = cast(torch.Tensor, self._identity_projection)
        output, weights = F.multi_head_attention_forward(
            projected_query,
            projected_key,
            projected_value,
            self.base.embed_dim,
            self.base.num_heads,
            in_proj_weight=None,
            in_proj_bias=None,
            bias_k=self.base.bias_k,
            bias_v=self.base.bias_v,
            add_zero_attn=self.base.add_zero_attn,
            dropout_p=self.base.dropout,
            out_proj_weight=self.base.out_proj.weight,
            out_proj_bias=self.base.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=True,
            q_proj_weight=identity_projection,
            k_proj_weight=identity_projection,
            v_proj_weight=identity_projection,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
        if self.batch_first and is_batched:
            output = output.transpose(0, 1)
        return output, weights


@dataclass(frozen=True)
class FADAAdaptedPolicy:
    policy: FADAAdaptablePolicy
    lora_config: FADALoRAConfig


def _discover_lora_targets(policy: FADAAdaptablePolicy) -> tuple[str, ...]:
    return tuple(
        name
        for name, module in policy.idm.named_modules()
        if name and isinstance(module, nn.MultiheadAttention)
    )


def _discover_legacy_lora_targets(policy: FADAPlannerIDMPolicy) -> tuple[str, ...]:
    return tuple(
        name
        for name, module in policy.idm.named_modules()
        if name
        and isinstance(module, nn.Linear)
        and not name.endswith(("self_attn.out_proj", "multihead_attn.out_proj"))
    )


def _replace_submodule(root: nn.Module, name: str, replacement: nn.Module) -> None:
    parent_name, separator, child_name = name.rpartition(".")
    if not child_name:
        raise ValueError(f"invalid LoRA target module name {name!r}")
    parent = root.get_submodule(parent_name) if separator else root
    setattr(parent, child_name, replacement)


def inject_fada_idm_lora(
    policy: FADAAdaptablePolicy,
    config: FADALoRAConfig,
) -> FADAAdaptedPolicy:
    """Freeze Planner/base IDM and inject zero-delta adapters at the exact manifest."""

    if config.adapter_type != "qv_attention":
        raise ValueError("active FADA injection requires Q/V attention LoRA")
    if any(
        isinstance(module, (FADALoRALinear, FADALoRAQVMultiheadAttention))
        for module in policy.idm.modules()
    ):
        raise ValueError("FADA policy already contains LoRA adapters")
    discovered = _discover_lora_targets(policy)
    if not discovered:
        raise ValueError("FADA IDM exposes no packed MultiheadAttention LoRA targets")
    if config.target_modules and tuple(config.target_modules) != discovered:
        raise ValueError(
            "LoRA target module manifest does not match the complete IDM injection owner: "
            f"expected={discovered} observed={tuple(config.target_modules)}"
        )
    resolved = replace(config, target_modules=discovered)

    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for name in discovered:
        module = policy.idm.get_submodule(name)
        if not isinstance(module, nn.MultiheadAttention):
            raise ValueError(f"LoRA target manifest entry is not MultiheadAttention: {name}")
        wrapped = FADALoRAQVMultiheadAttention(
            module,
            rank=resolved.rank,
            alpha=resolved.alpha,
            dropout=resolved.dropout,
        ).train(module.training)
        _replace_submodule(
            policy.idm,
            name,
            wrapped,
        )
    assert_fada_adaptation_parameter_ownership(policy, resolved)
    return FADAAdaptedPolicy(policy=policy, lora_config=resolved)


def _inject_fada_idm_legacy_linear_lora(
    policy: FADAPlannerIDMPolicy,
    config: FADALoRAConfig,
) -> FADAAdaptedPolicy:
    """Reconstruct historical v1/v2 all-Linear adapters without entering the active route."""

    if config.adapter_type != "legacy_linear":
        raise ValueError("legacy FADA injection requires legacy_linear adapter identity")
    if any(
        isinstance(module, (FADALoRALinear, FADALoRAQVMultiheadAttention))
        for module in policy.idm.modules()
    ):
        raise ValueError("FADA policy already contains LoRA adapters")
    discovered = _discover_legacy_lora_targets(policy)
    if not discovered:
        raise ValueError("FADA IDM exposes no legacy Linear LoRA targets")
    if config.target_modules and tuple(config.target_modules) != discovered:
        raise ValueError(
            "legacy LoRA target module manifest is incompatible: "
            f"expected={discovered} observed={tuple(config.target_modules)}"
        )
    resolved = replace(config, target_modules=discovered)
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for name in discovered:
        module = policy.idm.get_submodule(name)
        if not isinstance(module, nn.Linear):
            raise ValueError(f"legacy LoRA target manifest entry is not Linear: {name}")
        wrapped = FADALoRALinear(
            module,
            rank=resolved.rank,
            alpha=resolved.alpha,
            dropout=resolved.dropout,
        ).train(module.training)
        _replace_submodule(
            policy.idm,
            name,
            wrapped,
        )
    assert_fada_adaptation_parameter_ownership(policy, resolved)
    return FADAAdaptedPolicy(policy=policy, lora_config=resolved)


_ADAPTER_PARAMETER_SUFFIXES = (
    "lora_A.weight",
    "lora_B.weight",
    "lora_q_A.weight",
    "lora_q_B.weight",
    "lora_v_A.weight",
    "lora_v_B.weight",
)


def fada_adapter_named_parameters(
    policy: FADAAdaptablePolicy,
) -> Iterator[tuple[str, nn.Parameter]]:
    for name, parameter in policy.named_parameters():
        if name.endswith(_ADAPTER_PARAMETER_SUFFIXES):
            yield name, parameter


def fada_adapter_parameters(policy: FADAAdaptablePolicy) -> Iterator[nn.Parameter]:
    for _name, parameter in fada_adapter_named_parameters(policy):
        yield parameter


def assert_fada_adaptation_parameter_ownership(
    policy: FADAAdaptablePolicy,
    config: FADALoRAConfig,
) -> None:
    adapters = dict(fada_adapter_named_parameters(policy))
    if not adapters:
        raise ValueError("FADA adaptation requires at least one LoRA parameter")
    if config.adapter_type == "qv_attention":
        expected_names = {
            f"idm.{target}.lora_{projection}_{matrix}.weight"
            for target in config.target_modules
            for projection in config.target_projections
            for matrix in ("A", "B")
        }
    else:
        expected_names = {
            f"idm.{target}.lora_{matrix}.weight"
            for target in config.target_modules
            for matrix in ("A", "B")
        }
    if set(adapters) != expected_names:
        raise ValueError(
            "LoRA parameter ownership does not match the persisted target manifest: "
            f"expected={sorted(expected_names)} observed={sorted(adapters)}"
        )
    trainable = {name for name, parameter in policy.named_parameters() if parameter.requires_grad}
    if trainable != set(adapters):
        raise ValueError(
            "FADA adaptation must expose only adapter parameters as trainable: "
            f"expected={sorted(adapters)} observed={sorted(trainable)}"
        )
    if any(parameter.requires_grad for parameter in policy.planner.parameters()):
        raise ValueError("FADA adaptation Planner parameters must remain frozen")


@dataclass(frozen=True)
class FADATargetSplit:
    train_indices: torch.Tensor
    validation_indices: torch.Tensor


def split_fada_target_batch(
    batch: FADATargetBatch,
    *,
    validation_fraction: float,
    seed: int,
) -> FADATargetSplit:
    """Hold out command groups, falling back to episode or purged-time ownership."""

    fraction = float(validation_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError(f"validation_fraction must be in (0, 1), got {validation_fraction!r}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"split seed must be a non-negative integer, got {seed!r}")
    episodes = torch.unique(batch.episode_id.detach().to("cpu"), sorted=True)
    if int(episodes.numel()) == 1:
        timesteps = batch.start_timestep.detach().to("cpu")
        order = torch.argsort(timesteps, stable=True)
        validation_count = max(1, round(len(order) * fraction))
        validation_start = len(order) - validation_count
        validation_indices = order[validation_start:].to(torch.int64)
        validation_timestep = int(timesteps[validation_indices].min())
        window_span = int(batch.observation_history.shape[1] + batch.realized_future.shape[1] - 1)
        train_mask = timesteps[order[:validation_start]] + window_span < validation_timestep
        train_indices = order[:validation_start][train_mask].to(torch.int64)
        if train_indices.numel() == 0:
            raise ValueError(
                "FADA single-trajectory split is too short for train, validation, and purge gap"
            )
        return FADATargetSplit(
            train_indices=train_indices,
            validation_indices=validation_indices,
        )
    commands = torch.unique(batch.command.detach().to("cpu"), dim=0, sorted=True)
    if int(commands.shape[0]) > 1:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        command_order = commands[torch.randperm(int(commands.shape[0]), generator=generator)]
        validation_count = max(
            1,
            min(int(commands.shape[0]) - 1, round(len(commands) * fraction)),
        )
        validation_commands = command_order[:validation_count]
        validation_mask = torch.zeros_like(batch.episode_id, dtype=torch.bool)
        command_rows = batch.command.detach().to("cpu")
        for command in validation_commands:
            validation_mask |= torch.all(command_rows == command, dim=1).to(validation_mask.device)
        validation_indices = (
            torch.nonzero(validation_mask, as_tuple=False).flatten().to(torch.int64)
        )
        train_indices = torch.nonzero(~validation_mask, as_tuple=False).flatten().to(torch.int64)
        if train_indices.numel() == 0 or validation_indices.numel() == 0:
            raise ValueError(
                "FADA target command split must produce non-empty train and validation rows"
            )
        return FADATargetSplit(
            train_indices=train_indices,
            validation_indices=validation_indices,
        )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    episode_order = episodes[torch.randperm(int(episodes.numel()), generator=generator)]
    validation_count = max(1, min(int(episodes.numel()) - 1, round(len(episodes) * fraction)))
    validation_episodes = episode_order[:validation_count]
    validation_mask = torch.zeros_like(batch.episode_id, dtype=torch.bool)
    for episode in validation_episodes:
        validation_mask |= batch.episode_id == episode.to(batch.episode_id.device)
    validation_indices = torch.nonzero(validation_mask, as_tuple=False).flatten().to(torch.int64)
    train_indices = torch.nonzero(~validation_mask, as_tuple=False).flatten().to(torch.int64)
    if train_indices.numel() == 0 or validation_indices.numel() == 0:
        raise ValueError("FADA target split must produce non-empty train and validation rows")
    return FADATargetSplit(
        train_indices=train_indices,
        validation_indices=validation_indices,
    )


def select_fada_target_rows(batch: FADATargetBatch, indices: torch.Tensor) -> FADATargetBatch:
    if indices.ndim != 1 or indices.dtype != torch.int64 or indices.numel() == 0:
        raise ValueError("FADA target row indices must be non-empty rank-1 int64")
    if torch.unique(indices).numel() != indices.numel():
        raise ValueError("FADA target row indices must be unique")
    if int(indices.min()) < 0 or int(indices.max()) >= int(batch.episode_id.shape[0]):
        raise ValueError("FADA target row index is out of range")
    return FADATargetBatch(
        **{
            field: getattr(batch, field).index_select(0, indices.to(getattr(batch, field).device))
            for field in FADATargetBatch.__dataclass_fields__
        }
    )


def _batch_to_device(batch: FADATargetBatch, device: torch.device) -> FADATargetBatch:
    return FADATargetBatch(
        **{
            field: getattr(batch, field).to(device)
            for field in FADATargetBatch.__dataclass_fields__
        }
    )


def fada_adaptation_loss(
    policy: FADAPlannerIDMPolicy,
    batch: FADATargetBatch,
) -> torch.Tensor:
    device = next(policy.parameters()).device
    current = _batch_to_device(batch.validate(policy.config), device)
    predicted = policy.idm(
        current.observation_history,
        current.action_history,
        current.realized_future,
    )
    return first_action_mse(predicted, current.executed_action_chunk)


@dataclass(frozen=True)
class FADAAdaptationStats:
    loss: float
    grad_norm: float
    optimizer_steps: int


def _optimizer_parameter_ids(optimizer: torch.optim.Optimizer) -> list[int]:
    return [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]


class FADAAdaptationTrainer:
    """Own exactly one LoRA-only target-domain update transaction."""

    def __init__(
        self,
        policy: FADAPlannerIDMPolicy,
        optimizer: torch.optim.Optimizer,
        *,
        lora_config: FADALoRAConfig,
        max_grad_norm: float | None = None,
    ) -> None:
        assert_fada_adaptation_parameter_ownership(policy, lora_config)
        adapter_ids = [id(parameter) for parameter in fada_adapter_parameters(policy)]
        optimizer_ids = _optimizer_parameter_ids(optimizer)
        if len(optimizer_ids) != len(set(optimizer_ids)) or set(optimizer_ids) != set(adapter_ids):
            raise ValueError(
                "FADA adaptation optimizer must own only adapter parameters exactly once"
            )
        if max_grad_norm is not None and (
            not math.isfinite(float(max_grad_norm)) or float(max_grad_norm) <= 0.0
        ):
            raise ValueError("max_grad_norm must be finite and positive when provided")
        self.policy = policy
        self.optimizer = optimizer
        self.lora_config = lora_config
        self.max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)

    def update(self, batch: FADATargetBatch) -> FADAAdaptationStats:
        self.policy.planner.eval()
        self.policy.idm.train()
        adapters = list(fada_adapter_parameters(self.policy))
        self.optimizer.zero_grad(set_to_none=True)
        loss = fada_adaptation_loss(self.policy, batch)
        loss.backward()
        if any(parameter.grad is None for parameter in adapters):
            raise RuntimeError("FADA adaptation produced a missing adapter gradient")
        if any(
            parameter.grad is not None
            for name, parameter in self.policy.named_parameters()
            if not name.endswith(_ADAPTER_PARAMETER_SUFFIXES)
        ):
            raise RuntimeError("FADA adaptation accumulated a gradient on a frozen parameter")
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(adapters, self.max_grad_norm)
        squared = sum(
            float(torch.sum(torch.square(parameter.grad.detach())))
            for parameter in adapters
            if parameter.grad is not None
        )
        grad_norm = math.sqrt(squared)
        if not math.isfinite(grad_norm) or grad_norm <= 0.0:
            raise RuntimeError(
                f"FADA adaptation gradient norm must be finite and positive: {grad_norm}"
            )
        self.optimizer.step()
        return FADAAdaptationStats(
            loss=float(loss.detach()),
            grad_norm=grad_norm,
            optimizer_steps=1,
        )
