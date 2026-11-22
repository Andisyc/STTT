"""Pretrain one reference-conditioned residual policy through a frozen WM."""
from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig

from unilab.algos.torch.sttt.path_following.pretraining import train_latent_actor


@hydra.main(version_base="1.3", config_path="../conf/sttt", config_name="latent_actor_pretraining")
def main(cfg: DictConfig) -> None:
    result = train_latent_actor(cfg, root_dir=Path(__file__).resolve().parents[1])
    print(json.dumps(result, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
