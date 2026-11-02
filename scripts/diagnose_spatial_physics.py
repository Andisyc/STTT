"""Capture controlled MuJoCo branches or compare an existing world-model checkpoint."""

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
    from unilab.algos.torch.sttt.physical_probe import run_physical_probe

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT_DIR / "conf/sttt/physical_probe.yaml")
    parser.add_argument("overrides", nargs="*", help="Dotted YAML overrides, e.g. mode=compare")
    args = parser.parse_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    report = run_physical_probe(cfg, root_dir=ROOT_DIR)
    print(json.dumps(report, indent=2, allow_nan=False))
    if str(cfg.mode) == "compare" and report["usable_cases"] == 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
