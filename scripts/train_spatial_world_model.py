"""Train a spatial world model from episode-grouped collected trajectories."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))


@hydra.main(version_base="1.3", config_path="../conf/sttt", config_name="world_model")
def main(cfg: DictConfig) -> None:
    from unilab.algos.torch.sttt.training import run_world_model_training

    print(json.dumps(run_world_model_training(cfg), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
