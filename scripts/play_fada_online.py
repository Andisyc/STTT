"""Record online FADA adaptation offscreen, or launch the optional local viewer."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")

import hydra
import mujoco  # type: ignore[attr-defined]
import numpy as np
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from unilab.training import ensure_registries

ensure_registries()

from unilab.visualization.interactive_playback import (
    PlaybackControls,
    create_fada_playback_session,
    select_torch_device,
)
from unilab.visualization.playback_controls import _build_playback_config
from unilab.visualization.playback_cli import _build_play_args


@hydra.main(version_base="1.3", config_path="../conf/distill", config_name="config")
def main(cfg: DictConfig) -> None:
    """Compose the distill task and execute the official online FADA session."""

    if str(cfg.training.sim_backend) != "mujoco":
        raise ValueError("play_fada_online.py only supports MuJoCo backend.")

    device = OmegaConf.select(cfg, "training.device") or select_torch_device()
    print(f"[play_fada_online] Device: {device}")

    args = _build_play_args(cfg, algo="fada")
    from unilab.visualization.fada_recording import (
        FADAVideoRecordingConfig,
        record_fada_playback,
    )

    recording = FADAVideoRecordingConfig(
        **OmegaConf.to_container(cfg.video_recording, resolve=True)
    )

    # Create FADA session; the factory binds exactly one controller (offline or online).
    session, policy_obs_mode, checkpoint_path = create_fada_playback_session(
        playback_cfg=_build_playback_config(args, num_envs=1),
        cfg=cfg,
        root_dir=ROOT_DIR,
        device=device,
        log=lambda msg: print(f"[play_fada_online] {msg}"),
    )

    print(f"[play_fada_online] Checkpoint: {checkpoint_path}")
    print(f"[play_fada_online] Policy obs mode: {policy_obs_mode}")
    print(f"[play_fada_online] Action mode: {args.action_mode}")

    online_enabled = bool(OmegaConf.select(cfg, "online_adaptation.enabled", default=False))
    print(
        "[play_fada_online] Online adaptation "
        + ("ENABLED" if online_enabled else "disabled")
    )

    try:
        if recording.enabled:
            record_fada_playback(session, args, recording)
        else:
            _run_viewer(session, args)
    finally:
        session.wrapped_env.close()


def _run_viewer(session, args) -> None:
    import mujoco.viewer
    # Get environment and load MuJoCo model
    env = session.env

    # Load viewer model (same as play_interactive)
    from unilab.visualization.playback_viewer import (
        _default_viewer_camera_distance,
        _load_viewer_model,
        _resolve_focus_body_id,
    )
    model = _load_viewer_model(env, use_env_visual_model=True)
    data = mujoco.MjData(model)

    session.reset()
    focus_body_id = _resolve_focus_body_id(model, env, args.camera_focus_body_name)

    # Launch MuJoCo viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.distance = (
                args.camera_distance if args.camera_distance is not None
                else _default_viewer_camera_distance(
                    model, env, follow_body=args.camera_follow_body
                )
            )
            if args.camera_elevation is not None:
                viewer.cam.elevation = args.camera_elevation
            if args.camera_azimuth is not None:
                viewer.cam.azimuth = args.camera_azimuth
        print("[play_fada_online] MuJoCo viewer started")
        print("[play_fada_online] Controls: Space=pause/resume, N=single-step, +/-=speed")

        controls = PlaybackControls()

        while viewer.is_running():
            # Advance session; the bound controller runs collection/update internally.
            session.advance(controls)

            # Online adaptation stats published by the session.
            stats = getattr(session, "last_update_stats", None)
            if stats:
                print(f"[play_fada_online] Update: {stats}")
                session.last_update_stats = None

            # Sync viewer
            phys = session.physics_state()[0].astype(np.float64)
            mujoco.mj_setState(model, data, phys, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            mujoco.mj_forward(model, data)
            if args.camera_follow_body:
                with viewer.lock():
                    viewer.cam.lookat[:] = data.xpos[focus_body_id]
                    viewer.cam.lookat[2] += args.camera_height_offset
            viewer.sync()

            # Control speed
            time.sleep(controls.target_dt(env.cfg.ctrl_dt))

    print("[play_fada_online] Viewer closed")



if __name__ == "__main__":
    main()
