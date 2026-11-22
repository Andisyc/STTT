"""Capture executed checkpoint-bound Planner–Tracker sequences for WM training."""
from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.sttt.world_model.sequence_collection import collect_world_model_sequences
from unilab.training import ensure_registries


@hydra.main(version_base="1.3", config_path="../conf/distill", config_name="config")
def main(cfg: DictConfig) -> None:
    ensure_registries()
    node = OmegaConf.select(cfg, "world_model_collection")
    if node is None:
        raise ValueError("select +world_model_collection=default and explicit capture parameters")
    if not node.output_dir:
        raise ValueError("set world_model_collection.output_dir to a new directory")
    result = collect_world_model_sequences(
        cfg, root_dir=Path(__file__).resolve().parents[1], output_dir=node.output_dir,
        num_steps=node.num_steps, window_steps=node.window_steps,
        residual_amplitude_rad=node.residual_amplitude_rad, risk_tilt_limit_rad=node.risk_tilt_limit_rad,
        seed=node.seed, device=node.device,
        observation_augmentation=OmegaConf.to_container(node.observation_augmentation, resolve=True),
        trajectory=OmegaConf.to_container(node.trajectory, resolve=True) if node.trajectory is not None else None,
        actor_checkpoint=node.actor_checkpoint, world_model_checkpoint=node.world_model_checkpoint,
    )
    print(json.dumps(result, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
