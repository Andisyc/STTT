"""CLI for explicitly requested finite path-following evaluation cases."""
from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig

from unilab.algos.torch.sttt.path_following.evaluation import run_path_following_evaluation


@hydra.main(version_base="1.3", config_path="../conf/sttt", config_name="path_following_evaluate")
def main(cfg: DictConfig) -> None:
    summary = run_path_following_evaluation(cfg, root_dir=Path(__file__).resolve().parents[1])
    print(json.dumps(summary, indent=2, allow_nan=False))
    if not summary["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
