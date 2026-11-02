"""Capture MuJoCo stereo/IMU, replay with cuVSLAM, or diagnose both in one process."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))


def main() -> None:
    from unilab.algos.torch.sttt.visual_diagnostics import run_visual_diagnostics

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT_DIR / "conf/sttt/visual_odometry.yaml")
    parser.add_argument("overrides", nargs="*", help="Dotted YAML overrides, e.g. mode=replay")
    args = parser.parse_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    print(json.dumps(run_visual_diagnostics(cfg, root_dir=ROOT_DIR), allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
