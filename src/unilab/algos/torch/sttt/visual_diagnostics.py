"""Capture/replay a real policy's stereo stream, without enabling adaptation."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from unilab.algos.torch.sttt.pose_data import ControlInterval
from unilab.algos.torch.sttt.pose_diagnostics import (
    CONTROL_HEADER,
    PoseAlignmentOutput,
    control_row,
    load_controls,
)
from unilab.base.backend.sensors import (
    ImuSample,
    RigKinematicsSample,
    RigReferencePose,
    StereoCalibration,
    StereoFrame,
    StereoSensorConfig,
)


def save_calibration(path: Path, calibration: StereoCalibration) -> None:
    np.savez(path, **asdict(calibration))


def load_calibration(path: Path) -> StereoCalibration:
    with np.load(path, allow_pickle=False) as data:
        return StereoCalibration(
            width=int(data["width"]), height=int(data["height"]),
            focal=(float(data["focal"][0]), float(data["focal"][1])),
            principal=(float(data["principal"][0]), float(data["principal"][1])),
            rig_from_left=data["rig_from_left"].copy(),
            rig_from_right=data["rig_from_right"].copy(),
            rig_from_imu=data["rig_from_imu"].copy(),
            imu_frequency_hz=float(data["imu_frequency_hz"]),
        )


def save_frame(path: Path, frame: StereoFrame, kinematics: RigKinematicsSample) -> None:
    if (frame.sequence_id, frame.timestamp_ns) != (kinematics.sequence_id, kinematics.timestamp_ns):
        raise ValueError("Frame and relative kinematics must share the same capture key")
    np.savez_compressed(
        path, sequence_id=frame.sequence_id, timestamp_ns=frame.timestamp_ns,
        left_rgb=frame.left_rgb, right_rgb=frame.right_rgb,
        imu_timestamp_ns=np.asarray([item.timestamp_ns for item in frame.imu], dtype=np.int64),
        imu_acceleration=np.asarray([item.acceleration for item in frame.imu], dtype=np.float64).reshape(-1, 3),
        imu_angular_velocity=np.asarray([item.angular_velocity for item in frame.imu], dtype=np.float64).reshape(-1, 3),
        pelvis_from_rig=kinematics.pelvis_from_rig,
        up_in_pelvis=kinematics.up_in_pelvis,
    )


def load_sensor_frames(directory: Path) -> Iterator[tuple[StereoFrame, RigKinematicsSample]]:
    files = sorted(directory.glob("frame_*.npz"))
    if not files:
        raise FileNotFoundError(f"No captured frame_*.npz files in {directory}")
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            if not {"pelvis_from_rig", "up_in_pelvis"}.issubset(data.files):
                raise ValueError(f"{path} lacks image-time kinematics; recapture, do not assume a locked waist")
            times = data["imu_timestamp_ns"]
            acceleration = data["imu_acceleration"]
            angular = data["imu_angular_velocity"]
            if acceleration.shape != (len(times), 3) or angular.shape != acceleration.shape:
                raise ValueError(f"Invalid IMU arrays in {path}")
            frame = StereoFrame(
                sequence_id=int(data["sequence_id"]), timestamp_ns=int(data["timestamp_ns"]),
                left_rgb=data["left_rgb"].copy(), right_rgb=data["right_rgb"].copy(),
                imu=tuple(ImuSample(int(t), a.copy(), w.copy())
                          for t, a, w in zip(times, acceleration, angular, strict=True)),
            )
            kinematics = RigKinematicsSample(
                frame.sequence_id, frame.timestamp_ns,
                data["pelvis_from_rig"].copy(), data["up_in_pelvis"].copy(),
            )
            yield frame, kinematics


def _pose_matrix(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = [
        [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
        [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
        [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
    ]
    result[:3, 3] = position
    return result


class TrajectoryEvaluation:
    """One rigid origin alignment per sequence; never fit scale or the full path."""

    def __init__(self, path: Path) -> None:
        self._file = path.open("w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            "sequence_id", "timestamp_ns", "valid", "estimate_x", "estimate_y", "estimate_z",
            "reference_x", "reference_y", "reference_z", "position_error_m", "rotation_error_rad",
        ])
        self._alignments: dict[int, np.ndarray] = {}
        self.frames = self.valid_frames = 0
        self._position_squared = self._rotation_squared = 0.0
        self._previous_positions: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._reference_distance = self._estimated_distance = 0.0

    def observe(self, estimate: Any, reference: RigReferencePose) -> None:
        if (estimate.sequence_id, estimate.timestamp_ns) != (reference.sequence_id, reference.timestamp_ns):
            raise ValueError("Estimate and reference must have the same sequence and capture timestamp")
        self.frames += 1
        expected = reference.world_from_rig
        if not estimate.valid:
            self._writer.writerow([estimate.sequence_id, estimate.timestamp_ns, False,
                                   "", "", "", *expected[:3, 3], "", ""])
            return
        actual = _pose_matrix(estimate.position, estimate.quaternion)
        if estimate.sequence_id not in self._alignments:
            self._alignments[estimate.sequence_id] = expected @ np.linalg.inv(actual)
        aligned = self._alignments[estimate.sequence_id] @ actual
        distance = float(np.linalg.norm(aligned[:3, 3] - expected[:3, 3]))
        cosine = (np.trace(expected[:3, :3].T @ aligned[:3, :3]) - 1.0) / 2.0
        angle = float(np.arccos(np.clip(cosine, -1.0, 1.0)))
        previous = self._previous_positions.get(estimate.sequence_id)
        if previous is not None:
            self._reference_distance += float(np.linalg.norm(expected[:3, 3] - previous[0]))
            self._estimated_distance += float(np.linalg.norm(aligned[:3, 3] - previous[1]))
        self._previous_positions[estimate.sequence_id] = (
            expected[:3, 3].copy(), aligned[:3, 3].copy(),
        )
        self.valid_frames += 1
        self._position_squared += distance * distance
        self._rotation_squared += angle * angle
        self._writer.writerow([estimate.sequence_id, estimate.timestamp_ns, True,
                               *aligned[:3, 3], *expected[:3, 3], distance, angle])

    def summary(self) -> dict[str, Any]:
        count = self.valid_frames
        return {
            "frames": self.frames, "valid_frames": count,
            "valid_fraction": count / self.frames if self.frames else 0.0,
            "position_rmse_m": float(np.sqrt(self._position_squared / count)) if count else None,
            "rotation_rmse_rad": float(np.sqrt(self._rotation_squared / count)) if count else None,
            "reference_path_length_m": self._reference_distance,
            "estimated_path_length_m": self._estimated_distance,
            "alignment": "first_valid_pose_per_sequence_SE3_no_scale_fit",
        }

    def close(self) -> None:
        self._file.close()


def _reference_header() -> list[str]:
    return ["sequence_id", "timestamp_ns", *[f"T{i}{j}" for i in range(4) for j in range(4)]]


def _load_references(path: Path) -> dict[tuple[int, int], RigReferencePose]:
    result = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            sequence, timestamp = int(row["sequence_id"]), int(row["timestamp_ns"])
            matrix = np.array([float(row[f"T{i}{j}"]) for i in range(4) for j in range(4)]).reshape(4, 4)
            key = (sequence, timestamp)
            if key in result:
                raise ValueError(f"Duplicate reference capture in {path}: {key}")
            result[key] = RigReferencePose(sequence, timestamp, matrix)
    return result


def _make_odometry(calibration: StereoCalibration, cfg: Any) -> Any:
    from omegaconf import OmegaConf

    from unilab.algos.torch.sttt.visual_odometry import CuvslamOdometry

    noise = OmegaConf.to_container(cfg.imu_noise, resolve=True) if cfg.imu_noise is not None else None
    return CuvslamOdometry(calibration, mode=str(cfg.mode), imu_noise=noise)


def replay_visual_odometry(cfg: Any) -> dict[str, Any]:
    """Replay control-time arrivals; world truth is used only by the rig evaluator."""
    directory = Path(str(cfg.input_dir)).expanduser().resolve()
    calibration = load_calibration(directory / "calibration.npz")
    references = _load_references(directory / "reference.csv")
    if not (directory / "controls.csv").is_file():
        raise FileNotFoundError(f"{directory} lacks controls.csv; recapture action-time records")
    packets = iter(load_sensor_frames(directory))
    packet = next(packets, None)
    odometry = evaluation = alignment = None
    try:
        odometry = _make_odometry(calibration, cfg.odometry)
        evaluation = TrajectoryEvaluation(directory / "odometry.csv")
        alignment = PoseAlignmentOutput(directory, cfg.alignment)
        for control in load_controls(directory / "controls.csv"):
            alignment.enqueue(control)
            while packet is not None:
                frame, kinematics = packet
                if frame.sequence_id < control.sequence_id:
                    raise ValueError("Captured frame has no corresponding control sequence")
                if frame.sequence_id != control.sequence_id or frame.timestamp_ns > control.end_ns:
                    break
                estimate = odometry.track(frame)
                evaluation.observe(estimate, references[(frame.sequence_id, frame.timestamp_ns)])
                alignment.observe(estimate, kinematics)
                packet = next(packets, None)
            alignment.drain(now_ns=control.end_ns)
        if packet is not None:
            raise ValueError("Captured frames extend beyond their recorded control intervals")
        summary = {**evaluation.summary(), **alignment.summary()}
        if summary["valid_frames"] < 2:
            raise RuntimeError("Fewer than two valid poses: no motion accuracy can be evaluated")
        return summary
    finally:
        if alignment is not None:
            alignment.close()
        if evaluation is not None:
            evaluation.close()
        if odometry is not None:
            odometry.close()


def capture_visual_odometry(cfg: Any, *, root_dir: Path) -> dict[str, Any]:
    """Advance one existing FADA session; no alternate controller or fake policy."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from unilab.algos.torch.sttt.collection_runtime import create_collection_session

    steps = int(cfg.source.num_steps)
    if steps <= 0:
        raise ValueError("source.num_steps must be positive")
    checkpoint = cfg.source.checkpoint_path
    if not checkpoint:
        raise FileNotFoundError("Set source.checkpoint_path to an existing FADA Planner-IDM checkpoint")
    output = Path(str(cfg.output_dir)).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Capture directory must be empty: {output}")
    sensor_values = OmegaConf.to_container(cfg.sensor, resolve=True)
    if not isinstance(sensor_values, dict):
        raise ValueError("sensor must be a configuration mapping")
    sensor_values["position"] = tuple(sensor_values["position"])
    sensor_values["xyaxes"] = tuple(sensor_values["xyaxes"])
    sensor_config = StereoSensorConfig(**sensor_values)
    with initialize_config_dir(version_base="1.3", config_dir=str(root_dir / "conf/distill")):
        playback_cfg = compose(config_name="config", overrides=[
            "task=g1_walk_flat/mujoco", *list(cfg.source.overrides),
        ])
    OmegaConf.update(playback_cfg, "training.play_checkpoint_path", str(checkpoint))
    if bool(cfg.source.landmarks):
        fragments = list(OmegaConf.select(playback_cfg, "env.scene.fragment_files", default=[]))
        fragments.append(str(root_dir / "src/unilab/assets/robots/g1/sttt_visual_landmarks.xml"))
        OmegaConf.update(playback_cfg, "env.scene.fragment_files", fragments, force_add=True)
    session, metadata = create_collection_session(
        playback_cfg, root_dir=root_dir, device=cfg.source.device, seed=int(cfg.source.seed),
    )
    metadata["seed"] = int(cfg.source.seed)
    metadata["visual_sensor_format_version"] = 2
    stream = odometry = evaluation = alignment = reference_file = control_file = None
    frames_written = imu_written = 0
    terminated = False
    try:
        session.reset()
        stream = session.env.open_sensor_stream(sensor_config)
        if str(cfg.mode) == "live":
            odometry = _make_odometry(stream.calibration, cfg.odometry)
        output.mkdir(parents=True, exist_ok=True)
        save_calibration(output / "calibration.npz", stream.calibration)
        (output / "source.json").write_text(json.dumps(metadata, indent=2) + "\n")
        reference_file = (output / "reference.csv").open("w", newline="")
        writer = csv.writer(reference_file)
        writer.writerow(_reference_header())
        control_file = (output / "controls.csv").open("w", newline="")
        control_writer = csv.writer(control_file)
        control_writer.writerow(CONTROL_HEADER)
        if odometry is not None:
            evaluation = TrajectoryEvaluation(output / "odometry.csv")
            alignment = PoseAlignmentOutput(output, cfg.alignment)
        for transition_id in range(steps):
            before = stream.clock
            command = np.asarray(session.env.state.info["commands"][0, :3], dtype=np.float64).copy()
            session.step_once()
            after = stream.clock
            if after.sequence_id != before.sequence_id:
                raise RuntimeError("A control interval crossed a physical reset")
            action = np.asarray(session.env.state.info["current_actions"][0], dtype=np.float64).copy()
            if action.shape != (29,) or command.shape != (3,) or not np.isfinite(action).all() or not np.isfinite(command).all():
                raise ValueError("Control records require finite env_input (29,) and command (3,)")
            control = ControlInterval(before.sequence_id, transition_id, before.timestamp_ns,
                                      after.timestamp_ns, action, command)
            control_writer.writerow(control_row(control))
            if alignment is not None:
                alignment.enqueue(control)
            frames = stream.drain()
            reference_rows = stream.drain_reference_poses()
            kinematics_rows = stream.drain_kinematics()
            kinematics_by_key = {(item.sequence_id, item.timestamp_ns): item for item in kinematics_rows}
            if len(frames) != len(kinematics_rows):
                raise RuntimeError("Frames and image-time kinematics are not paired")
            references = {(item.sequence_id, item.timestamp_ns): item for item in reference_rows}
            for reference in reference_rows:
                writer.writerow([reference.sequence_id, reference.timestamp_ns,
                                 *reference.world_from_rig.reshape(-1)])
            for frame in frames:
                reference = references[(frame.sequence_id, frame.timestamp_ns)]
                kinematics = kinematics_by_key[(frame.sequence_id, frame.timestamp_ns)]
                save_frame(output / f"frame_{frames_written:08d}.npz", frame, kinematics)
                frames_written += 1
                imu_written += len(frame.imu)
                if evaluation is not None and alignment is not None:
                    estimate = odometry.track(frame)
                    evaluation.observe(estimate, reference)
                    alignment.observe(estimate, kinematics)
            if alignment is not None:
                alignment.drain(now_ns=after.timestamp_ns)
            if np.any(session.env.state.terminated | session.env.state.truncated):
                terminated = True
                break
        if frames_written == 0:
            raise RuntimeError("No stereo frames were captured; increase source.num_steps")
        summary = {"captured_frames": frames_written, "imu_samples": imu_written,
                   "episode_terminated": terminated, "output_dir": str(output)}
        if evaluation is not None and alignment is not None:
            summary.update(evaluation.summary())
            summary.update(alignment.summary())
            if evaluation.valid_frames < 2:
                raise RuntimeError("Fewer than two valid poses: no motion accuracy can be evaluated")
        return summary
    finally:
        if alignment is not None:
            alignment.close()
        if control_file is not None:
            control_file.close()
        if evaluation is not None:
            evaluation.close()
        if reference_file is not None:
            reference_file.close()
        if odometry is not None:
            odometry.close()
        if stream is not None:
            stream.close()
        session.close()


def run_visual_diagnostics(cfg: Any, *, root_dir: Path) -> dict[str, Any]:
    if str(cfg.mode) == "replay":
        return replay_visual_odometry(cfg)
    if str(cfg.mode) not in {"capture", "live"}:
        raise ValueError("mode must be capture, replay, or live")
    return capture_visual_odometry(cfg, root_dir=root_dir)
