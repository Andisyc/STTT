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
from unilab.visualization.fada_rendering import FADADemoRenderConfig, FADAPlaybackRendering
from unilab.visualization.playback_cli import _build_play_args
from unilab.visualization.playback_controls import _build_playback_config


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
    render_node = OmegaConf.select(cfg, "demo.rendering")
    render_config = FADADemoRenderConfig(
        **(OmegaConf.to_container(render_node, resolve=True) if render_node is not None else {})
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
            record_fada_playback(session, args, recording, render_config=render_config)
        else:
            _run_viewer(session, args, render_config)
    finally:
        session.wrapped_env.close()


def _run_viewer(session, args, render_config: FADADemoRenderConfig | None = None) -> None:
    import mujoco.viewer
    # Get environment and load MuJoCo model
    env = session.env

    # Load viewer model (same as play_interactive)
    from unilab.visualization.playback_viewer import _load_viewer_model

    model = _load_viewer_model(env, use_env_visual_model=args.use_env_visual_model)
    data = mujoco.MjData(model)

    session.reset()
    rendering = FADAPlaybackRendering(session, model, args, render_config)
    rendering.sync_physics(data)
    controls = PlaybackControls(paused=args.start_paused, speed=args.speed)

    def on_key(key: int) -> None:
        if key == 32:
            controls.toggle_pause()
        elif key == 78:
            controls.request_single_step()
        elif key in (43, 61):
            controls.set_speed(controls.speed * 1.25)
        elif key == 45:
            controls.set_speed(controls.speed / 1.25)

    # Launch MuJoCo viewer
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        with viewer.lock():
            rendering.initialize_camera(viewer.cam)
        print("[play_fada_online] MuJoCo viewer started")
        print("[play_fada_online] Controls: Space=pause/resume, N=single-step, +/-=speed")

        while viewer.is_running():
            # Advance session; the bound controller runs collection/update internally.
            advanced = session.advance(controls)
            if advanced:
                rendering.observe_step()

            # Online adaptation stats published by the session.
            stats = getattr(session, "last_update_stats", None)
            if stats:
                print(f"[play_fada_online] Update: {stats}")
                session.last_update_stats = None

            # Sync viewer
            with viewer.lock():
                rendering.sync_physics(data)
                rendering.update_camera(viewer.cam, data)
                viewer.user_scn.ngeom = 0
                rendering.append_scene(viewer.user_scn, data)
            status = rendering.status(data)
            viewer.set_texts((
                mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                "\n".join((status.title, *status.details, *((status.alert,) if status.alert else ()))), "",
            ))
            viewer.sync()

            # Control speed
            time.sleep(controls.target_dt(env.cfg.ctrl_dt))

    print("[play_fada_online] Viewer closed")



if __name__ == "__main__":
    main()
