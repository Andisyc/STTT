"""CLI for a verified-start finite residual contrast through a frozen latent WM."""
from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig

from unilab.algos.torch.sttt.path_following.latent_response_probe import run_latent_response_direction_probe


@hydra.main(version_base="1.3", config_path="../conf/sttt", config_name="latent_response_direction_probe")
def main(cfg: DictConfig) -> None:
    summary = run_latent_response_direction_probe(cfg, root_dir=Path(__file__).resolve().parents[1])
    print(json.dumps(summary, indent=2, allow_nan=False))
    if not summary["complete"] or not summary["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
