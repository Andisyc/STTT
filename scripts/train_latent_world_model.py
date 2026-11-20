"""Train a latent physical model from separately captured episode sequences."""
from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig

from unilab.algos.torch.sttt.world_model.training import train_world_model


@hydra.main(version_base="1.3", config_path="../conf/sttt", config_name="latent_world_model")
def main(cfg: DictConfig) -> None:
    print(json.dumps(train_world_model(cfg, root_dir=Path(__file__).resolve().parents[1]), allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
