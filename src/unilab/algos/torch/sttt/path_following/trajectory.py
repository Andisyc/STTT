"""Reference geometry shared by measured control and differentiable imagination."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import math
from typing import Any, Mapping

import numpy as np
import torch

from unilab.algos.torch.sttt.geometry import wrap_to_pi
from unilab.algos.torch.sttt.path_following.reference import pose_identity, pose_matrix
from unilab.algos.torch.sttt.pose_data import TimedPelvisPose


@dataclass(frozen=True)
class TrajectoryConfig:
    kind: str = "straight"
    length_m: float = 30.0
    radius_m: float = 2.0
    speed_m_s: float = 0.4
    samples: int = 1024
    preview_points: int = 8
    preview_spacing_m: float = 0.15
    projection_window_m: float = 0.5
    initial_lateral_m: float = 0.0
    initial_heading_rad: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in {"straight", "circle", "figure_eight", "random"}:
            raise ValueError("unsupported reference trajectory")
        for name in ("length_m", "radius_m", "speed_m_s", "preview_spacing_m", "projection_window_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("samples", "preview_points"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 2:
                raise ValueError(f"{name} must be an integer >= 2")
        if not all(math.isfinite(x) for x in (self.initial_lateral_m, self.initial_heading_rad)):
            raise ValueError("initial reference offsets must be finite")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TrajectoryConfig:
        return cls(**dict(value))


@dataclass(frozen=True, eq=False)
class ReferenceTrajectory:
    """Batched polyline in one fixed horizontal frame; speeds use metres/second."""

    xy: torch.Tensor
    speed: torch.Tensor
    config: TrajectoryConfig
    delta: torch.Tensor = field(init=False, repr=False)
    length: torch.Tensor = field(init=False, repr=False)
    arc: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.xy.ndim != 3 or self.xy.shape[0] < 1 or self.xy.shape[1] < 2 or self.xy.shape[2] != 2:
            raise ValueError("reference points require [batch,points,2]")
        if self.speed.shape != self.xy.shape[:2] or self.speed.device != self.xy.device or self.speed.dtype != self.xy.dtype:
            raise ValueError("reference speeds must match points")
        if not self.xy.is_floating_point() or not bool(torch.isfinite(self.xy).all()) or not bool(torch.isfinite(self.speed).all()) or bool((self.speed < 0).any()):
            raise ValueError("reference points/speeds must be finite with nonnegative speeds")
        delta = self.xy[:, 1:] - self.xy[:, :-1]
        length = delta.norm(dim=-1)
        if bool((length <= 1e-8).any()):
            raise ValueError("adjacent reference points must be distinct")
        object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "length", length)
        object.__setattr__(self, "arc", torch.cat((length.new_zeros(length.shape[0], 1), length.cumsum(1)), 1))

    @classmethod
    def generate(cls, config: TrajectoryConfig, *, batch_size: int = 1, seed: int = 0,
                 device: Any = "cpu", dtype: torch.dtype = torch.float32) -> ReferenceTrajectory:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("reference batch size must be positive")
        t = torch.linspace(0., config.length_m, config.samples, device=device, dtype=dtype)
        if config.kind == "straight":
            xy = torch.stack((t, torch.zeros_like(t)), -1)
        elif config.kind == "circle":
            angle = t / config.radius_m
            xy = torch.stack((config.radius_m * angle.sin(), config.radius_m * (1 - angle.cos())), -1)
        elif config.kind == "figure_eight":
            angle = t / config.radius_m
            xy = config.radius_m * torch.stack((angle.sin(), .5 * (2 * angle).sin()), -1)
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            frequencies = torch.tensor((.3, .7, 1.3), device=device, dtype=dtype) / config.radius_m
            phases = (torch.rand(3, generator=generator) * (2 * math.pi)).to(device=device, dtype=dtype)
            heading = .65 * (t[:, None] * frequencies + phases).sin().sum(-1)
            heading = heading - heading[0]
            step = config.length_m / (config.samples - 1)
            increments = step * torch.stack((heading.cos(), heading.sin()), -1)
            xy = torch.cat((increments.new_zeros(1, 2), increments[:-1].cumsum(0)), 0)
        tangent = xy[1] - xy[0]
        angle = torch.atan2(tangent[1], tangent[0]) - config.initial_heading_rad
        c, s = angle.cos(), angle.sin()
        xy = torch.stack((c * xy[:, 0] + s * xy[:, 1], -s * xy[:, 0] + c * xy[:, 1]), -1)
        xy = xy + xy.new_tensor((0., -config.initial_lateral_m))
        return cls(xy.unsqueeze(0).expand(batch_size, -1, -1).clone(),
                   xy.new_full((batch_size, config.samples), config.speed_m_s), config)

    def sample(self, progress: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if progress.ndim != 2 or progress.shape[0] != self.xy.shape[0]:
            raise ValueError("sample progress requires [batch,queries]")
        progress = progress.clamp_min(0).minimum(self.arc[:, -1:])
        index = torch.searchsorted(self.arc.contiguous(), progress.contiguous(), right=True).sub(1).clamp(0, self.xy.shape[1] - 2)
        start = self.arc.gather(1, index)
        length = self.length.gather(1, index)
        fraction = ((progress - start) / length).unsqueeze(-1)
        point = self.xy.gather(1, index[..., None].expand(-1, -1, 2))
        delta = self.delta.gather(1, index[..., None].expand(-1, -1, 2))
        a, b = self.speed.gather(1, index), self.speed.gather(1, index + 1)
        return point + fraction * delta, delta / length[..., None], a + fraction[..., 0] * (b - a)

    def project(self, pose: torch.Tensor, previous_progress: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if pose.shape != (self.xy.shape[0], 3) or previous_progress.shape != (self.xy.shape[0],):
            raise ValueError("projection requires pose [batch,3] and prior progress [batch]")
        offset = pose[:, None, :2] - self.xy[:, :-1]
        fraction = ((offset * self.delta).sum(-1) / self.length.square()).clamp(0, 1)
        candidates = self.xy[:, :-1] + fraction[..., None] * self.delta
        progress = self.arc[:, :-1] + fraction * self.length
        lower = (previous_progress - self.config.projection_window_m).clamp_min(0)
        upper = previous_progress + self.config.projection_window_m
        eligible = (self.arc[:, 1:] >= lower[:, None]) & (self.arc[:, :-1] <= upper[:, None])
        distances = (pose[:, None, :2] - candidates).square().sum(-1).masked_fill(~eligible, torch.inf)
        index = distances.argmin(1, keepdim=True)
        selected = progress.gather(1, index)[:, 0]
        point, tangent, _ = self.sample(selected[:, None])
        difference = pose[:, :2] - point[:, 0]
        direction = tangent[:, 0]
        lateral = -direction[:, 1] * difference[:, 0] + direction[:, 0] * difference[:, 1]
        heading = wrap_to_pi(pose[:, 2] - torch.atan2(direction[:, 1], direction[:, 0]))
        return selected, torch.stack((lateral, heading), -1)


@dataclass(frozen=True, eq=False)
class PathState:
    reference: ReferenceTrajectory
    pose: torch.Tensor
    progress: torch.Tensor

    def errors(self) -> torch.Tensor:
        return self.reference.project(self.pose, self.progress)[1]

    def preview(self) -> torch.Tensor:
        cfg = self.reference.config
        offsets = torch.arange(cfg.preview_points, device=self.pose.device, dtype=self.pose.dtype) * cfg.preview_spacing_m
        point, tangent, speed = self.reference.sample(self.progress[:, None] + offsets)
        difference = point - self.pose[:, None, :2]
        c, s = self.pose[:, 2:3].cos(), self.pose[:, 2:3].sin()
        return torch.stack((c * difference[..., 0] + s * difference[..., 1],
                            -s * difference[..., 0] + c * difference[..., 1],
                            c * tangent[..., 0] + s * tangent[..., 1],
                            -s * tangent[..., 0] + c * tangent[..., 1], speed), -1)

    def command(self, template: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Nominal body velocity/yaw rate from the same reference used by the loss."""
        template = torch.as_tensor(template, device=self.pose.device, dtype=self.pose.dtype)
        if template.ndim != 2 or template.shape[0] != self.pose.shape[0] or template.shape[1] not in (3, 7):
            raise ValueError("reference commands require a matched [batch,3 or 7] template")
        _, tangent, speed = self.reference.sample(self.progress[:, None])
        look = self.preview()
        heading = torch.atan2(tangent[:, 0, 1], tangent[:, 0, 0])
        relative = wrap_to_pi(heading - self.pose[:, 2])
        curvature = torch.atan2(look[:, 1, 3], look[:, 1, 2]) - torch.atan2(look[:, 0, 3], look[:, 0, 2])
        yaw_rate = speed[:, 0] * wrap_to_pi(curvature) / self.reference.config.preview_spacing_m
        velocity = torch.stack((speed[:, 0] * relative.cos(), speed[:, 0] * relative.sin(), yaw_rate), -1)
        return torch.cat((velocity, template[:, 3:]), -1)

    def advance(self, motion: torch.Tensor) -> PathState:
        c, s = self.pose[:, 2].cos(), self.pose[:, 2].sin()
        pose = torch.stack((self.pose[:, 0] + c * motion[:, 0] - s * motion[:, 1],
                            self.pose[:, 1] + s * motion[:, 0] + c * motion[:, 1],
                            wrap_to_pi(self.pose[:, 2] + motion[:, 2])), -1)
        progress, _ = self.reference.project(pose, self.progress)
        return PathState(self.reference, pose, progress)

    def detach(self) -> PathState:
        return PathState(self.reference, self.pose.detach(), self.progress.detach())


class MeasuredTrajectory:
    """One localization identity and task anchor; measured poses reset imagination."""

    def __init__(self, initial: TimedPelvisPose, config: TrajectoryConfig, *, device: Any = "cpu",
                 dtype: torch.dtype = torch.float32, reference: ReferenceTrajectory | None = None):
        matrix = pose_matrix(initial)
        self.sequence_id, self.tracking_epoch, self.frame_id = pose_identity(initial)
        self.timestamp_ns = initial.timestamp_ns
        self.origin_xy = tuple(matrix[:2, 3])
        self.heading_rad = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
        self.reference = ReferenceTrajectory.generate(config, device=device, dtype=dtype) if reference is None else reference
        if self.reference.xy.shape[0] != 1 or self.reference.config != config:
            raise ValueError("measured trajectory requires one matching reference")
        self._states: OrderedDict[int, PathState] = OrderedDict()
        self.state_at(initial)

    def state_at(self, pose: TimedPelvisPose) -> PathState:
        if pose_identity(pose) != (self.sequence_id, self.tracking_epoch, self.frame_id):
            raise ValueError("trajectory measurement changed localization identity")
        if pose.timestamp_ns in self._states:
            return self._states[pose.timestamp_ns]
        if self._states and pose.timestamp_ns < next(reversed(self._states)):
            raise ValueError("trajectory pose predates retained progress history")
        matrix = pose_matrix(pose)
        difference = matrix[:2, 3] - self.origin_xy
        c, s = math.cos(self.heading_rad), math.sin(self.heading_rad)
        yaw = math.atan2(matrix[1, 0], matrix[0, 0]) - self.heading_rad
        value = self.reference.xy.new_tensor([[c * difference[0] + s * difference[1], -s * difference[0] + c * difference[1], yaw]])
        previous = next(reversed(self._states.values())).progress if self._states else value.new_zeros(1)
        progress, _ = self.reference.project(value, previous)
        state = PathState(self.reference, value, progress)
        self._states[pose.timestamp_ns] = state
        if len(self._states) > 2048:
            self._states.popitem(last=False)
        return state

    @property
    def current_state(self) -> PathState:
        return next(reversed(self._states.values()))

    def errors(self, pose: TimedPelvisPose) -> np.ndarray:
        return self.state_at(pose).errors()[0].detach().cpu().numpy()
