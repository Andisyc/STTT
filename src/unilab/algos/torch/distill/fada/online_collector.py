"""Online streaming collector for FADA target windows during deployment."""

from __future__ import annotations

from collections import deque

import numpy as np

from unilab.algos.torch.distill.fada.model import FADAArchitectureConfig
from unilab.algos.torch.distill.fada.windows import (
    FADACausalTransition,
    FADACausalWindow,
    build_fada_causal_window,
)


class OnlineTrajectoryCollector:
    """Incrementally build FADACausalWindow from streaming env transitions."""

    def __init__(self, config: FADAArchitectureConfig) -> None:
        self._config = config
        self._record_count = config.history_length + config.prediction_horizon - 1
        self._records: deque[FADACausalTransition] = deque(maxlen=self._record_count)
        self._episode_id = 0
        self._timestep = 0

    def append(
        self,
        observation: np.ndarray,
        previous_action: np.ndarray,
        command: np.ndarray,
        executed_action: np.ndarray,
        next_observation: np.ndarray,
    ) -> FADACausalWindow | None:
        """Append one transition; return window when buffer is full."""
        record = FADACausalTransition(
            observation=np.asarray(observation, dtype=np.float32),
            previous_action=np.asarray(previous_action, dtype=np.float32),
            command=np.asarray(command, dtype=np.float32),
            executed_action=np.asarray(executed_action, dtype=np.float32),
            next_observation=np.asarray(next_observation, dtype=np.float32),
            episode_id=self._episode_id,
            timestep=self._timestep,
        )
        self._records.append(record)
        self._timestep += 1

        if len(self._records) < self._record_count:
            return None

        return build_fada_causal_window(
            list(self._records),
            history_length=self._config.history_length,
            prediction_horizon=self._config.prediction_horizon,
        )

    def reset(self) -> None:
        """Clear buffer and start new episode."""
        self._records.clear()
        self._episode_id += 1
        self._timestep = 0
