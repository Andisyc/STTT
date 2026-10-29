"""Shared MuJoCo camera, scene decorations and status for FADA playback."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import mujoco
import numpy as np

from unilab.envs.common.rotation import np_wrap_to_pi, np_yaw_from_quat
from unilab.visualization.playback_viewer import (
    _default_viewer_camera_distance,
    _resolve_focus_body_id,
)


@dataclass(frozen=True)
class FADADemoRenderConfig:
    enabled: bool = False
    show_reference_route: bool = True
    show_footprints: bool = True
    max_footprint_markers: int = 256
    footprint_radius_m: float = 0.045
    route_width_m: float = 0.012
    camera_margin_m: float = 1.0

    def __post_init__(self) -> None:
        for name in ("footprint_radius_m", "route_width_m", "camera_margin_m"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"demo.rendering.{name} must be finite and positive")
        if (isinstance(self.max_footprint_markers, bool)
                or not isinstance(self.max_footprint_markers, Integral)
                or self.max_footprint_markers <= 0):
            raise ValueError("demo.rendering.max_footprint_markers must be a positive integer")


@dataclass(frozen=True)
class FADAFrameStatus:
    title: str
    details: tuple[str, ...]
    alert: str | None
    loaded: bool


class FADAPlaybackRendering:
    """Read a session without stepping it; append visuals without changing physics.

    Construct after session.reset(). Model/body lookup and route anchoring are
    cold-path work. Native preview and offscreen recording use the same object
    API; callers own their scene's base geometry and reset it before appending.
    """

    def __init__(
        self, session: Any, model: Any, args: Any,
        config: FADADemoRenderConfig | None = None,
    ) -> None:
        self.session, self.model, self.args = session, model, args
        self.config = config or FADADemoRenderConfig()
        self.focus_body_id = _resolve_focus_body_id(model, session.env, args.camera_focus_body_name)
        self.ctrl_dt = float(session.env.cfg.ctrl_dt)
        self.reset_count = self.update_count = 0
        self.last_loss: float | None = None
        self.last_reset_time: float | None = None
        self.last_attach_time: float | None = None
        self._fixed_camera: tuple[np.ndarray, float, float, float] | None = None
        self.payload_config = getattr(session.env.cfg, "runtime_payload", None)
        self.payload_body_id = -1
        if self.payload_config is not None and self.payload_config.enabled:
            self.payload_body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, self.payload_config.body_name,
            )
            if self.payload_body_id < 0:
                raise ValueError(f"Payload body is absent from playback model: {self.payload_config.body_name}")
            self.payload_position = np.asarray(self.payload_config.attachment_pos_m, dtype=np.float64)
            self.payload_half_size = np.asarray(self.payload_config.half_size_m, dtype=np.float64)
        # Initialize the visual data once, without advancing the source env.
        data = mujoco.MjData(model)
        self.sync_physics(data)
        self.origin_xy = data.xpos[self.focus_body_id, :2].copy()
        self.origin_yaw = float(np_yaw_from_quat(data.xquat[self.focus_body_id:self.focus_body_id + 1])[0])
        self.reference_route: np.ndarray | None = None
        program = getattr(session, "command_program", None)
        if self.config.enabled and program is not None:
            # The command program starts along the robot's initial forward axis.
            base = session.env.get_base_pos()[0]
            yaw = float(np_yaw_from_quat(session.env.get_base_quat())[0])
            rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
            xy = program.reference_points @ rotation.T + base[:2]
            ground_z = float(np.min(session.env.get_foot_pos()[0, :, 2]))
            self.reference_route = np.column_stack((xy, np.full(len(xy), ground_z + 0.015)))

    @property
    def elapsed_s(self) -> float:
        return self.session.step_count * self.ctrl_dt

    def sync_physics(self, data: Any) -> None:
        physics = np.asarray(self.session.physics_state()[0], dtype=np.float64)
        mujoco.mj_setState(self.model, data, physics, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        mujoco.mj_forward(self.model, data)

    def initialize_camera(self, camera: Any) -> None:
        mujoco.mjv_defaultCamera(camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = (
            self.args.camera_distance if self.args.camera_distance is not None
            else _default_viewer_camera_distance(
                self.model, self.session.env, follow_body=self.args.camera_follow_body,
            )
        )
        if self.args.camera_elevation is not None:
            camera.elevation = self.args.camera_elevation
        if self.args.camera_azimuth is not None:
            camera.azimuth = self.args.camera_azimuth
        if self.config.enabled and not self.args.camera_follow_body and self.reference_route is not None:
            # Enclose the route and the standing robot in a sphere. Fitting a
            # square viewport also covers landscape preview/video viewports.
            low = self.reference_route.min(axis=0)
            high = self.reference_route.max(axis=0) + [0.0, 0.0, 1.2]
            camera.lookat[:] = (low + high) / 2.0
            radius = np.linalg.norm(high - low) / 2.0 + self.config.camera_margin_m
            if self.args.camera_distance is None:
                camera.distance = radius / math.sin(math.radians(float(self.model.vis.global_.fovy)) / 2.0)
            self._fixed_camera = (
                camera.lookat.copy(), float(camera.distance),
                float(camera.elevation), float(camera.azimuth),
            )

    def update_camera(self, camera: Any, data: Any) -> None:
        if self._fixed_camera is not None:
            lookat, distance, elevation, azimuth = self._fixed_camera
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = lookat
            camera.distance, camera.elevation, camera.azimuth = distance, elevation, azimuth
        elif self.args.camera_follow_body:
            camera.lookat[:] = data.xpos[self.focus_body_id]
            camera.lookat[2] += self.args.camera_height_offset

    def observe_step(self) -> None:
        """Update display counters exactly once after a successful control step."""
        stats = self.session.last_update_stats
        if stats is not None:
            self.update_count += 1
            self.last_loss = float(stats["loss"])
        state = self.session.env.state
        if bool(state.terminated[0] or state.truncated[0]):
            self.reset_count += 1
            self.last_reset_time = self.elapsed_s
        if bool(np.asarray(self.session.info.get("runtime_payload_attached_this_step", [False])).reshape(-1)[0]):
            self.last_attach_time = self.elapsed_s

    def _payload_active(self) -> bool:
        return bool(np.asarray(self.session.info.get("runtime_payload_active", [False])).reshape(-1)[0])

    def append_scene(self, scene: Any, data: Any) -> None:
        """Append identical decorations to user_scn or an updated renderer.scene."""
        if self._payload_active() and self.payload_body_id >= 0:
            if scene.ngeom >= scene.maxgeom:
                raise RuntimeError("Playback scene has no room for the payload marker")
            rotation = data.xmat[self.payload_body_id].reshape(3, 3)
            center = data.xpos[self.payload_body_id] + rotation @ self.payload_position
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_BOX,
                self.payload_half_size, center, rotation.reshape(9),
                np.array([1.0, 0.35, 0.05, 1.0], dtype=np.float32),
            )
            scene.ngeom += 1
        if not self.config.enabled:
            return
        recorder = getattr(self.session, "footprint_recorder", None)
        records = (
            recorder.snapshot(0)[-self.config.max_footprint_markers:]
            if self.config.show_footprints and recorder is not None else ()
        )
        available = max(0, scene.maxgeom - scene.ngeom)
        # Prefer the newest real footprints when the scene is unusually small.
        records = records[-available:] if available else ()
        route = self.reference_route if self.config.show_reference_route else None
        route_count = min(len(route) - 1, available - len(records)) if route is not None else 0
        if route_count >= 4:
            indices = np.linspace(0, len(route) - 1, route_count + 1, dtype=int)
            for start, end in zip(indices[:-1], indices[1:]):
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                    np.eye(3).reshape(9), np.array([0.55, 0.6, 0.65, 0.65], dtype=np.float32),
                )
                mujoco.mjv_connector(
                    geom, mujoco.mjtGeom.mjGEOM_CAPSULE, self.config.route_width_m,
                    route[start], route[end],
                )
                scene.ngeom += 1
        for record in records:
            position = np.array(record.position_world)
            position[2] += 0.01
            color = [0.1, 0.65, 1.0, 1.0] if record.foot == "left" else [1.0, 0.35, 0.1, 1.0]
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([self.config.footprint_radius_m, 0.0, 0.0]), position,
                np.eye(3).reshape(9), np.asarray(color, dtype=np.float32),
            )
            scene.ngeom += 1

    def status(self, data: Any) -> FADAFrameStatus:
        active = self._payload_active()
        added_mass = float(np.asarray(self.session.info.get("runtime_payload_added_mass_kg", [0.0])).reshape(-1)[0])
        payload = f"PAYLOAD +{added_mass:g} kg" if active else "NO PAYLOAD"
        if active and self.payload_config is not None:
            payload += f" @ {self.payload_config.body_name}"
        delta = data.xpos[self.focus_body_id, :2] - self.origin_xy
        c, s = math.cos(self.origin_yaw), math.sin(self.origin_yaw)
        yaw = float(np_yaw_from_quat(data.xquat[self.focus_body_id:self.focus_body_id + 1])[0])
        yaw_error = math.degrees(float(np_wrap_to_pi(yaw - self.origin_yaw)))
        loss = f" loss={self.last_loss:.4g}" if self.last_loss is not None else ""
        details = [
            f"forward={c * delta[0] + s * delta[1]:+.2f}m lateral={-s * delta[0] + c * delta[1]:+.2f}m",
            f"yaw error={yaw_error:+.1f}deg resets={self.reset_count}",
            f"LoRA updates={self.update_count}{loss}",
        ]
        if self.config.enabled:
            program = getattr(self.session, "command_program", None)
            if program is not None:
                step = max(int(self.session.command_control_steps[0]) - 1, 0)
                segment = program.segment_index_at_control_step(step) + 1
                command = np.asarray(self.session.info["commands"])[0, :3]
                details.append(f"segment={segment}/{len(program.commands)} command=[{command[0]:+.2f}, {command[1]:+.2f}, {command[2]:+.2f}]")
            if self.config.show_reference_route and self.reference_route is not None:
                details.append("Gray: reference route | Blue: left touchdown | Orange: right touchdown")
        alert = None
        if self.last_attach_time is not None and self.elapsed_s - self.last_attach_time < 2.0:
            alert = f"PAYLOAD ATTACHED +{added_mass:g} kg"
        if self.last_reset_time is not None and self.elapsed_s - self.last_reset_time < 2.0:
            reset = "RESET" if self.session.autoreset_enabled else "TERMINATED"
            alert = f"EPISODE {reset} - trajectory interrupted"
        return FADAFrameStatus(f"t={self.elapsed_s:.2f}s | {payload}", tuple(details), alert, active)
