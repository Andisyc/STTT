"""Physical identification and task-time trajectory losses for dual online LoRA."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from unilab.algos.torch.sttt.geometry import wrap_to_pi
from unilab.algos.torch.sttt.path_following.trajectory import ReferenceTrajectory
from unilab.algos.torch.sttt.world_model.model import LatentWorldModel
from unilab.algos.torch.sttt.world_model.motion import motion_prefix, window_motion
from unilab.algos.torch.sttt.world_model.types import LatentState


def timed_reference(reference: ReferenceTrajectory, elapsed_s: float, steps: int,
                    ctrl_dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Use task time, not the robot's projected progress, to retain forward motion."""
    if elapsed_s < 0 or steps < 1 or ctrl_dt <= 0:
        raise ValueError("reference requires nonnegative task time and positive horizon/period")
    times = torch.arange(steps + 1, device=reference.xy.device, dtype=reference.xy.dtype) * ctrl_dt + elapsed_s
    progress = times.unsqueeze(0).expand(reference.xy.shape[0], -1) * reference.config.speed_m_s
    point, tangent, _ = reference.sample(progress)
    heading = torch.atan2(tangent[..., 1], tangent[..., 0])
    pose = torch.cat((point, heading.unsqueeze(-1)), -1)
    change = point[:, 1:] - point[:, :-1]
    c, s = heading[:, :-1].cos(), heading[:, :-1].sin()
    velocity = torch.stack((c * change[..., 0] + s * change[..., 1],
                            -s * change[..., 0] + c * change[..., 1],
                            wrap_to_pi(heading[:, 1:] - heading[:, :-1])), -1) / ctrl_dt
    if reference.command_bounds is not None:
        bounds = velocity.new_tensor(reference.command_bounds)
        if bool(((velocity < bounds[:, 0] - 1e-5) | (velocity > bounds[:, 1] + 1e-5)).any()):
            raise ValueError("timed reference exceeds the source command range; choose a feasible trajectory")
    return pose, velocity


def predicted_poses(origin: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
    local = motion_prefix(motion)
    c, s = origin[:, 2:3].cos(), origin[:, 2:3].sin()
    return torch.stack((origin[:, :1] + c * local[..., 0] - s * local[..., 1],
                        origin[:, 1:2] + s * local[..., 0] + c * local[..., 1],
                        origin[:, 2:3] + local[..., 2]), -1)


def planner_trajectory_loss(origin: torch.Tensor, motion: torch.Tensor, reference_poses: torch.Tensor,
                            *, position_tolerance_m: float, heading_tolerance_rad: float) -> torch.Tensor:
    predicted = predicted_poses(origin.detach(), motion)[:, 1:]
    target = reference_poses[:, 1:].detach()
    position = (predicted[..., :2] - target[..., :2]) / position_tolerance_m
    heading = wrap_to_pi(predicted[..., 2] - target[..., 2]) / heading_tolerance_rad
    return torch.cat((position, heading.unsqueeze(-1)), -1).square().mean()


def planner_parameter_change(parameters: Sequence[torch.Tensor],
                             baseline: Sequence[torch.Tensor]) -> torch.Tensor:
    """Mean squared distance from a fixed adaptation-start parameter anchor."""
    if not parameters or len(parameters) != len(baseline):
        raise ValueError("Planner parameter penalty requires a matching nonempty anchor")
    return sum((parameter - reference).square().sum()
               for parameter, reference in zip(parameters, baseline)) / sum(parameter.numel() for parameter in parameters)


def identification_loss(model: LatentWorldModel, initial: LatentState,
                        batch: Mapping[str, torch.Tensor], *, xy_tolerance_m: float,
                        yaw_tolerance_rad: float) -> dict[str, torch.Tensor]:
    """Replay actual ENV actions; never train WM against Planner goals or predictions."""
    latent = initial.detach()
    motions, bodies = [], []
    for action in batch["actions"].unbind(1):
        motions.append(model.motion(latent, action.detach()))
        latent, _ = model.prior(latent, action.detach())
        bodies.append(model.decode(latent))
    prediction = torch.stack(bodies, 1)
    valid = ~batch["terminal"].bool() & ~batch["is_first"][:, 1:].bool()
    body_error = ((prediction - batch["observations"][:, 1:].detach()) * model.get_buffer("proprio_loss_weight")).square().mean(-1)
    body_loss = torch.where(valid, body_error, 0.).sum() / valid.sum().clamp_min(1)
    motion = torch.stack(motions, 1)
    start, end = batch["window_start"].long(), batch["window_end"].long()
    displacement = window_motion(motion, start, end)
    prefix_yaw = torch.cat((motion.new_zeros(motion.shape[0], 1), motion[..., 2].cumsum(1)), 1)
    turn = prefix_yaw.gather(1, end) - prefix_yaw.gather(1, start)
    displacement = torch.cat((displacement[..., :2], turn.unsqueeze(-1)), -1)
    invalid_prefix = torch.cat((start.new_zeros(start.shape[0], 1), (~valid).long().cumsum(1)), 1)
    mask = batch["window_valid"].bool() & (invalid_prefix.gather(1, start) == invalid_prefix.gather(1, end))
    target = torch.where(mask.unsqueeze(-1), batch["window_delta"].detach(), 0.)
    scale = motion.new_tensor((xy_tolerance_m, xy_tolerance_m, yaw_tolerance_rad))
    error = ((displacement - target) / scale).square().mean(-1)
    motion_loss = torch.where(mask, error, 0.).sum() / mask.sum().clamp_min(1)
    return {"loss": body_loss + motion_loss, "body_loss": body_loss, "motion_loss": motion_loss,
            "body_prediction": prediction, "body_target": batch["observations"][:, 1:].detach(),
            "body_valid": valid, "motion_prediction": displacement, "motion_target": target,
            "motion_valid": mask}
