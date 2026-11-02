"""Collect episode-safe spatial windows using a real FADA checkpoint.

Example:
    uv run scripts/collect_spatial_data.py --checkpoint /path/to/planner_idm.pt \
        --num-rollouts 2 --num-steps 2000 --out data/sttt/spatial_rollouts.pt

Every rollout opens and closes its own session with ``seed = --seed + i``
applied before the environment is created, so rollouts are independent
records rather than split windows of one trajectory. Windows from all
rollouts are written to one canonical collection at ``--out``.

Additional task settings use Hydra overrides, e.g. env.max_episode_seconds=2.0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from uuid import uuid4

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from None
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {parsed}")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Existing FADA Planner-IDM checkpoint.")
    parser.add_argument("--num-rollouts", type=_positive_int, default=2,
                        help="Independent rollouts, each with seed --seed+i and its own session.")
    parser.add_argument("--num-steps", type=_positive_int, default=2000)
    parser.add_argument("--out", type=Path, default=ROOT_DIR / "data/sttt/spatial_rollouts.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", help="Policy inference device (default: existing playback selection).")
    parser.add_argument("--render", action="store_true", help="Open the existing MuJoCo playback viewer.")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides for conf/distill/config.yaml.")
    args = parser.parse_args()
    if args.seed < 0 or args.seed + args.num_rollouts > 2**32:
        parser.error("all rollout seeds (--seed+i) must be in [0, 2**32)")
    return args


def main() -> int:
    args = _parse_args()

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from unilab.algos.torch.sttt.collection import (
        build_spatial_windows,
        save_spatial_collection,
        summarize_spatial_collection,
    )
    from unilab.algos.torch.sttt.collection_runtime import (
        collect_transitions,
        create_collection_session,
    )

    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT_DIR / "conf/distill")):
        cfg = compose(config_name="config", overrides=["task=g1_walk_flat/mujoco", *args.overrides])
    if args.checkpoint is not None:
        OmegaConf.update(cfg, "training.play_checkpoint_path", str(args.checkpoint.expanduser().resolve()))

    windows: list[dict] = []
    metadata: dict = {}
    rollout_ids: list[str] = []
    rollout_seeds: list[int] = []
    # Rollouts are merged only when the policy identity and the effective
    # collection condition are identical; the checkpoint is never mixed.
    identity_keys = ("checkpoint_path", "checkpoint_sha256", "collection_context",
                     "action_representation", "ctrl_dt", "joint_names", "obs_dim", "act_dim",
                     "proprio_layout", "action_scale", "default_joint_angles",
                     "quaternion_order", "coordinate_frame")
    for offset in range(args.num_rollouts):
        rollout_seed = int(args.seed) + offset
        rollout_id = uuid4().hex
        session, rollout_metadata = create_collection_session(
            cfg, root_dir=ROOT_DIR, device=args.device, seed=rollout_seed,
        )
        try:
            if not metadata:
                metadata = rollout_metadata
            else:
                mismatched = [
                    key for key in identity_keys
                    if rollout_metadata.get(key) != metadata.get(key)
                ]
                if mismatched:
                    raise RuntimeError(
                        f"Rollout {offset} changed {mismatched} relative to rollout 0; "
                        "refusing to merge windows from different collection conditions."
                    )
            transitions = collect_transitions(
                session, num_steps=args.num_steps, seed=rollout_seed, rollout_id=rollout_id,
            )
            if args.render:
                from unilab.algos.torch.sttt.collection_runtime import render_transitions

                transitions = render_transitions(session, transitions)
            windows.extend(build_spatial_windows(transitions))
        finally:
            session.close()
        rollout_ids.append(rollout_id)
        rollout_seeds.append(rollout_seed)

    metadata["seed"] = int(args.seed)
    metadata["rollout_ids"] = rollout_ids
    metadata["rollout_seeds"] = rollout_seeds
    save_spatial_collection(args.out, windows, metadata)

    summary = summarize_spatial_collection(windows, metadata)
    print(f"Collected {summary['window_count']} episode-safe windows -> {args.out}")
    print(
        f"Rollouts: {summary['rollout_count']} (seeds {rollout_seeds}); "
        f"distinct (rollout_id, episode_id) groups: {summary['group_count']}"
    )
    print(f"Action representation: {metadata['action_representation']}; ctrl_dt={metadata['ctrl_dt']}")
    joint_action = summary["joint_action"]
    if joint_action is not None:
        print("Per-joint action statistics over window rows (overlapping windows are not independent samples):")
        for index, name in enumerate(metadata["joint_names"]):
            print(
                f"  [{index:02d}] {name}: min={joint_action['min'][index]:+.6f} "
                f"max={joint_action['max'][index]:+.6f} std={joint_action['std'][index]:.6f}"
            )
    command_range = summary["observed_command_range"]
    if command_range is not None:
        print(f"Observed command min={command_range['min']} max={command_range['max']} (vx, vy, yaw_rate)")
    configured = summary["provenance"]["configured_vel_limit"]
    if configured is not None:
        print(f"Configured commands.vel_limit={configured}")
    print(f"Label source: {summary['provenance']['label_source']}")
    conditions = summary["window_condition_counts"]
    print(
        "Window payload conditions (actually recorded): "
        + ", ".join(f"{name}={count}" for name, count in conditions.items())
    )
    if summary["warning"] is not None:
        print(f"WARNING: {summary['warning']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())