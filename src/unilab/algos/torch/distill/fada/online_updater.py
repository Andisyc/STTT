"""Online streaming adaptation updater for FADA LoRA-IDM."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from unilab.algos.torch.distill.fada.adaptation import (
    FADALoRAConfig,
    fada_adaptation_loss,
    fada_adapter_parameters,
    inject_fada_idm_lora,
)
from unilab.algos.torch.distill.fada.model import FADAPlannerIDMPolicy
from unilab.algos.torch.distill.fada.target_data import FADATargetBatch
from unilab.algos.torch.distill.fada.windows import FADACausalWindow


class OnlineAdaptationUpdater:
    """Update LoRA-IDM with exponential forgetting on streaming windows."""

    def __init__(
        self,
        policy: FADAPlannerIDMPolicy,
        lora_config: FADALoRAConfig,
        *,
        lambda_forget: float = 0.99,
        lr: float = 1e-4,
        max_grad_norm: float | None = 1.0,
    ) -> None:
        if not 0.0 < lambda_forget <= 1.0:
            raise ValueError(f"lambda_forget must be in (0, 1], got {lambda_forget}")
        
        # Auto-discover LoRA targets if not provided
        if not lora_config.target_modules:
            from unilab.algos.torch.distill.fada.adaptation import _discover_lora_targets
            discovered = _discover_lora_targets(policy)
            from dataclasses import replace
            lora_config = replace(lora_config, target_modules=discovered)
        
        adapted = inject_fada_idm_lora(policy, lora_config)
        self.policy = adapted.policy
        self.lora_config = adapted.lora_config
        self.lambda_forget = float(lambda_forget)
        self.max_grad_norm = max_grad_norm
        
        adapter_params = list(fada_adapter_parameters(self.policy))
        self.optimizer = torch.optim.Adam(adapter_params, lr=lr)
        self._window_buffer: list[FADACausalWindow] = []

    def append_window(self, window: FADACausalWindow) -> None:
        """Add window to buffer."""
        self._window_buffer.append(window)

    def update(self) -> dict[str, float] | None:
        """Perform one exponential-weighted update if buffer non-empty."""
        if not self._window_buffer:
            return None

        was_training = self.policy.idm.training
        self.policy.idm.train()
        try:
            self.optimizer.zero_grad(set_to_none=True)

            total_loss = self._exponential_weighted_loss()
            total_loss.backward()

            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    fada_adapter_parameters(self.policy),
                    self.max_grad_norm
                )

            self.optimizer.step()
        finally:
            if not was_training:
                self.policy.idm.eval()

        stats = {
            "loss": float(total_loss.detach()),
            "num_windows": len(self._window_buffer),
        }
        self._window_buffer.clear()
        return stats

    def _exponential_weighted_loss(self) -> torch.Tensor:
        """Compute exponential-weighted loss over buffered windows."""
        n = len(self._window_buffer)
        weights = [self.lambda_forget ** (n - 1 - i) for i in range(n)]
        weight_sum = sum(weights)
        
        total_loss = torch.tensor(0.0, device=next(self.policy.parameters()).device)
        for window, weight in zip(self._window_buffer, weights):
            batch = self._window_to_batch(window)
            loss = fada_adaptation_loss(self.policy, batch)
            total_loss = total_loss + weight * loss
        
        return total_loss / weight_sum

    def _window_to_batch(self, window: FADACausalWindow) -> FADATargetBatch:
        """Convert single window to FADATargetBatch on the policy device/dtype."""
        reference = next(self.policy.parameters())
        device = reference.device
        dtype = reference.dtype

        def _tensor(values: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(
                np.asarray(values), dtype=torch.float32
            ).unsqueeze(0).to(device=device, dtype=dtype)

        return FADATargetBatch(
            observation_history=_tensor(window.observation_history),
            action_history=_tensor(window.action_history),
            command=_tensor(window.command),
            realized_future=_tensor(window.realized_future),
            executed_action_chunk=_tensor(window.executed_action_chunk),
            episode_id=torch.tensor([window.episode_id], dtype=torch.int64, device=device),
            start_timestep=torch.tensor(
                [window.start_timestep], dtype=torch.int64, device=device
            ),
        )
