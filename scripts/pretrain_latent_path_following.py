"""Collect clean simulation data and alternate physical-model/policy pretraining."""
from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig

from unilab.algos.torch.sttt.path_following.pretraining import run_pretraining_pipeline


@hydra.main(version_base="1.3", config_path="../conf/sttt", config_name="latent_pretraining")
def main(cfg: DictConfig) -> None:
    result = run_pretraining_pipeline(cfg, root_dir=Path(__file__).resolve().parents[1])
    print(json.dumps(result, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
