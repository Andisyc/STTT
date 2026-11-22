"""SE(2) composition for step-local horizontal displacement labels."""
from __future__ import annotations

import torch

MOTION_FRAME = "previous_control_horizontal_heading"


def motion_prefix(motion: torch.Tensor) -> torch.Tensor:
    """Compose [...,steps,3] body increments into [...,steps+1,3] poses."""
    if motion.ndim < 2 or motion.shape[-1] != 3:
        raise ValueError("motion increments require [...,steps,3]")
    yaw = motion[..., 2].cumsum(-1)
    previous_yaw = torch.cat((yaw.new_zeros(*yaw.shape[:-1], 1), yaw[..., :-1]), -1)
    c, s = previous_yaw.cos(), previous_yaw.sin()
    dx = c * motion[..., 0] - s * motion[..., 1]
    dy = s * motion[..., 0] + c * motion[..., 1]
    poses = torch.stack((dx.cumsum(-1), dy.cumsum(-1), yaw), -1)
    return torch.cat((poses.new_zeros(*poses.shape[:-2], 1, 3), poses), -2)


def window_motion(motion: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
    """Return each window displacement in its own start-heading frame."""
    poses = motion_prefix(motion)
    a = poses.gather(1, start[..., None].expand(-1, -1, 3))
    b = poses.gather(1, end[..., None].expand(-1, -1, 3))
    difference = b - a
    c, s = a[..., 2].cos(), a[..., 2].sin()
    return torch.stack((c * difference[..., 0] + s * difference[..., 1],
                        -s * difference[..., 0] + c * difference[..., 1],
                        torch.atan2(difference[..., 2].sin(), difference[..., 2].cos())), -1)
