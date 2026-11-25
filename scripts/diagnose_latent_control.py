"""Run physical authority, measured feedback, WM, Critic and A diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig

from unilab.algos.torch.sttt.path_following.control_diagnostics import run_control_diagnostics


@hydra.main(version_base="1.3", config_path="../conf/sttt", config_name="latent_control_diagnostics")
def main(cfg: DictConfig) -> None:
    report = run_control_diagnostics(cfg, root_dir=Path(__file__).resolve().parents[1])
    print(json.dumps({"complete": report["complete"], "error": report["error"],
                      "stages": {name: result["status"] for name, result in report["stages"].items()}}, allow_nan=False), flush=True)
    if not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
