"""CLI for correct/zero/shuffled residual prediction comparisons."""
from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig

from unilab.algos.torch.sttt.path_following.residual_diagnostics import run_residual_input_diagnostics


@hydra.main(version_base="1.3", config_path="../conf/sttt", config_name="response_residual_diagnostics")
def main(cfg: DictConfig) -> None:
    report = run_residual_input_diagnostics(cfg)
    print(json.dumps({
        "output_path": report["output_path"], "checkpoint": report["checkpoint"],
        "channels": report["channels"], "units": report["units"],
        "correct": report["correct"], "zero": report["zero"],
        "shuffled": {key: value for key, value in report["shuffled"].items() if key != "trials"},
        "physical_direction_verified": report["physical_direction_verified"],
        "interpretation": report["interpretation"],
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
