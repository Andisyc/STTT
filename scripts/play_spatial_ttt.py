"""Headless real-window spatial TTT playback; no imagined rollout or SLAM process."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))


@hydra.main(version_base="1.3", config_path="../conf/distill", config_name="config")
def main(cfg: DictConfig) -> None:
    from unilab.visualization.interactive_playback import create_fada_playback_session
    from unilab.visualization.playback_cli import _build_play_args
    from unilab.visualization.playback_controls import _build_playback_config

    if str(cfg.training.sim_backend) != "mujoco":
        raise ValueError("Spatial playback currently requires task=g1_walk_flat/mujoco")
    if not bool(OmegaConf.select(cfg, "spatial_adaptation.enabled", default=False)):
        raise ValueError(
            "play_spatial_ttt.py requires spatial_adaptation.enabled=true; "
            "a proprioceptive-only online update is not spatial adaptation"
        )
    print(
        "[play_spatial_ttt] source="
        f"{OmegaConf.select(cfg, 'spatial_adaptation.odometry_source')} "
        f"visual_mode={OmegaConf.select(cfg, 'spatial_adaptation.visual_mode')}"
    )
    steps = int(cfg.training.play_steps)
    if steps <= 0:
        raise ValueError("training.play_steps must be positive")
    args = _build_play_args(cfg, algo="fada")
    session, _, _ = create_fada_playback_session(
        playback_cfg=_build_playback_config(args, num_envs=1), cfg=cfg,
        root_dir=ROOT_DIR, device=OmegaConf.select(cfg, "training.device"),
    )
    try:
        session.reset()
        for _ in range(steps):
            session.step_once()
            if session.last_update_stats is not None:
                print(json.dumps(session.last_update_stats, allow_nan=False))
        output = OmegaConf.select(cfg, "spatial_adaptation.save_adapter_path")
        if output:
            from unilab.algos.torch.sttt.adapter_checkpoint import save_spatial_adapter
            from unilab.algos.torch.sttt.online_controller import SpatialOnlineController

            if not isinstance(session.controller, SpatialOnlineController):
                raise ValueError("Saving a spatial adapter requires spatial_adaptation.enabled=true")
            print(f"Saved spatial adapter: {save_spatial_adapter(session.controller, output)}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
