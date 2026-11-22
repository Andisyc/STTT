"""Evaluate one frozen pretrained actor and nominal control on shared references."""
from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.sttt.path_following.pretrained_evaluation import evaluate_pretrained_actor


@hydra.main(version_base="1.3", config_path="../conf/sttt", config_name="latent_pretrained_evaluate")
def main(cfg: DictConfig) -> None:
    root = Path(__file__).resolve().parents[1]
    def resolve(value):
        if not value:
            raise ValueError("provide explicit evaluation checkpoint/output paths")
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else root / path
    evaluation = OmegaConf.to_container(cfg.evaluation, resolve=True)
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation must be a mapping")
    result = evaluate_pretrained_actor(
        evaluation, root_dir=root, output_dir=resolve(cfg.output_dir),
        policy_checkpoint=resolve(cfg.policy_checkpoint), profile_path=str(resolve(cfg.profile_path)) if cfg.profile_path else None,
        world_model_checkpoint=resolve(cfg.world_model_checkpoint), actor_checkpoint=resolve(cfg.actor_checkpoint),
        source_overrides=list(cfg.source_overrides), device=cfg.device,
    )
    print(json.dumps(result, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
