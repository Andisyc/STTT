"""Probe a spatial world-model checkpoint's action Jacobian numerics.

Loads a trained spatial world-model checkpoint and one canonical spatial
collection, then compares autograd sensitivities of a selected scalar
trajectory output to central finite differences. Prints a JSON report.

Scope: agreement between autograd and finite differences proves numerical
consistency of the learned function's computation graph only — NOT a
physical Jacobian, strict positive realness (SPR), or closed-loop
stability. ``max_abs_future_action_sensitivity`` exposes the dense MLP's
sensitivity to actions strictly after the selected output step
(future-action leakage); it does not declare the method causal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from unilab.algos.torch.sttt.checkpoint import (
    data_contract_from_metadata,
    load_world_model_checkpoint,
)
from unilab.algos.torch.sttt.collection import (
    load_spatial_collection,
    trajectory_window_from_dict,
)
from unilab.algos.torch.sttt.diagnostics import diagnose_spatial_jacobian


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a spatial world-model checkpoint (required).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to one canonical spatial collection file (required).",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index of the window in the collection to probe (default: 0).",
    )
    parser.add_argument(
        "--action-steps",
        type=int,
        nargs="+",
        default=None,
        help="Action time steps to probe; default: [0].",
    )
    parser.add_argument(
        "--joint-indices",
        type=int,
        nargs="+",
        default=None,
        help="Action joint indices to probe; default: [0, 1, 2].",
    )
    parser.add_argument(
        "--output-step",
        type=int,
        default=-1,
        help="Trajectory output step to differentiate; negative wraps "
        "(default: -1, the final step).",
    )
    parser.add_argument(
        "--output-coordinate",
        type=int,
        choices=(0, 1, 2),
        default=2,
        help="Output coordinate: 0=dx, 1=dy, 2=dyaw (default: 2).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-5,
        help="Central-difference half-width in action units (default: 1e-5).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.checkpoint.is_file():
        raise ValueError(f"--checkpoint does not exist: {args.checkpoint}")
    if not args.data.is_file():
        raise ValueError(f"--data does not exist: {args.data}")

    collection = load_spatial_collection(args.data)
    data_contract = data_contract_from_metadata(collection["metadata"])
    model, payload = load_world_model_checkpoint(
        args.checkpoint, expected_data_contract=data_contract
    )

    windows = collection["windows"]
    if not -len(windows) <= args.sample_index < len(windows):
        raise ValueError(
            f"--sample-index {args.sample_index} out of range for collection "
            f"with {len(windows)} windows."
        )
    window_dict = windows[args.sample_index]
    window = trajectory_window_from_dict(window_dict)

    report = diagnose_spatial_jacobian(
        model,
        window,
        action_steps=args.action_steps,
        joint_indices=args.joint_indices,
        output_step=args.output_step,
        output_coordinate=args.output_coordinate,
        epsilon=args.epsilon,
    )
    report["window"] = {
        "sample_index": args.sample_index,
        "rollout_id": window_dict["rollout_id"],
        "episode_id": window_dict["episode_id"],
    }
    report["checkpoint"] = {
        "path": str(args.checkpoint),
        "completed_epochs": payload.get("completed_epochs"),
        "best_validation_loss": payload.get("best_validation_loss"),
    }
    report["data"] = str(args.data)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
