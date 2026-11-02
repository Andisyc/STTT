"""Pure PyTorch batched SE(2)/SO(3) geometry operators for STTT.

All functions are differentiable, support arbitrary leading batch dimensions,
and follow the conventions of ``unilab.envs.common.rotation``:

- Quaternions are ``(w, x, y, z)`` (w-first), normalized unit quaternions.
- Rotation matrices are ``(..., 3, 3)`` with column-vector convention:
  ``v_world = R @ v_body``.
- Yaw angles live in ``[-pi, pi]``.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "quat_to_mat",
    "mat_to_quat",
    "yaw_mat",
    "yaw_from_quat",
    "wrap_to_pi",
    "matvec",
    "matmul",
    "so3_exp",
    "so3_log",
    "compute_se2_deltas",
]

_EPS = 1e-8


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """Convert normalized w-first quaternions to rotation matrices.

    Args:
        q: Quaternions of shape ``(..., 4)`` in ``(w, x, y, z)`` order.
            Assumed normalized; renormalize upstream if unsure.

    Returns:
        Rotation matrices of shape ``(..., 3, 3)``.
    """
    w, x, y, z = q.unbind(-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    row0 = torch.stack((ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)), dim=-1)
    row1 = torch.stack((2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)), dim=-1)
    row2 = torch.stack((2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def mat_to_quat(m: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to w-first quaternions.

    Uses Shepperd's method (branch-selected per element) for numerical
    stability, then canonicalizes the sign so that ``w >= 0``. Sign
    continuity across a batch dimension must be enforced by the caller
    if required (cf. ``np_quat_ensure_continuity``).

    Args:
        m: Rotation matrices of shape ``(..., 3, 3)``.

    Returns:
        Unit quaternions of shape ``(..., 4)`` in ``(w, x, y, z)`` order
        with non-negative real part.
    """
    m00 = m[..., 0, 0]
    m11 = m[..., 1, 1]
    m22 = m[..., 2, 2]
    m01, m02 = m[..., 0, 1], m[..., 0, 2]
    m10, m12 = m[..., 1, 0], m[..., 1, 2]
    m20, m21 = m[..., 2, 0], m[..., 2, 1]

    # Four candidate squared components; the largest gives a stable branch.
    cand = torch.stack(
        (
            1.0 + m00 + m11 + m22,  # 4 w^2
            1.0 + m00 - m11 - m22,  # 4 x^2
            1.0 - m00 + m11 - m22,  # 4 y^2
            1.0 - m00 - m11 + m22,  # 4 z^2
        ),
        dim=-1,
    )
    branch = cand.argmax(dim=-1)

    q = torch.zeros(m.shape[:-2] + (4,), dtype=m.dtype, device=m.device)

    # Branch 0: w largest.
    sel = branch == 0
    if sel.any():
        s = torch.sqrt(cand[..., 0].clamp(min=0.0) + _EPS) * 2.0  # s = 4w
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
        q = torch.where(sel.unsqueeze(-1), torch.stack((qw, qx, qy, qz), dim=-1), q)

    # Branch 1: x largest.
    sel = branch == 1
    if sel.any():
        s = torch.sqrt(cand[..., 1].clamp(min=0.0) + _EPS) * 2.0  # s = 4x
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
        q = torch.where(sel.unsqueeze(-1), torch.stack((qw, qx, qy, qz), dim=-1), q)

    # Branch 2: y largest.
    sel = branch == 2
    if sel.any():
        s = torch.sqrt(cand[..., 2].clamp(min=0.0) + _EPS) * 2.0  # s = 4y
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
        q = torch.where(sel.unsqueeze(-1), torch.stack((qw, qx, qy, qz), dim=-1), q)

    # Branch 3: z largest.
    sel = branch == 3
    if sel.any():
        s = torch.sqrt(cand[..., 3].clamp(min=0.0) + _EPS) * 2.0  # s = 4z
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s
        q = torch.where(sel.unsqueeze(-1), torch.stack((qw, qx, qy, qz), dim=-1), q)

    q = torch.nn.functional.normalize(q, dim=-1)
    # Canonicalize: non-negative real part.
    sign = torch.where(q[..., :1] < 0.0, -1.0, 1.0)
    return q * sign


def yaw_mat(rotation: torch.Tensor) -> torch.Tensor:
    """Extract the z-axis (yaw) rotation matrix from a rotation.

    Projects the body x-axis onto the world xy-plane and builds the pure
    heading rotation around world z.

    Args:
        rotation: Either rotation matrices of shape ``(..., 3, 3)`` or
            w-first quaternions of shape ``(..., 4)``.

    Returns:
        Rotation matrices of shape ``(..., 3, 3)`` representing rotation
        about the z-axis by the heading angle.

    Raises:
        ValueError: If the trailing shape is neither ``(3, 3)`` nor ``(4,)``.
    """
    if rotation.shape[-2:] == (3, 3):
        mat = rotation
        forward_x = mat[..., 0, 0]
        forward_y = mat[..., 1, 0]
    elif rotation.shape[-1:] == (4,):
        w, x, y, z = rotation.unbind(-1)
        # R[0,0] = 1 - 2(y^2+z^2), R[1,0] = 2(xy + wz); the squared/shared
        # terms are invariant under q -> -q, so matrix and quat paths agree.
        forward_x = 1.0 - 2.0 * (y * y + z * z)
        forward_y = 2.0 * (x * y + w * z)
    else:
        raise ValueError(f"Expected (..., 3, 3) matrix or (..., 4) quat, got {rotation.shape}")

    heading = torch.atan2(forward_y, forward_x)
    cos = torch.cos(heading)
    sin = torch.sin(heading)

    zero = torch.zeros_like(cos)
    one = torch.ones_like(cos)
    row0 = torch.stack((cos, -sin, zero), dim=-1)
    row1 = torch.stack((sin, cos, zero), dim=-1)
    row2 = torch.stack((zero, zero, one), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Extract the scalar yaw (heading) angle from w-first quaternions.

    Args:
        q: Quaternions of shape ``(..., 4)`` in ``(w, x, y, z)`` order.

    Returns:
        Yaw angles of shape ``(...)`` in ``[-pi, pi]``.
    """
    w, x, y, z = q.unbind(-1)
    # Yaw component of the quaternion-derived rotation matrix:
    # forward_x = R[0,0], forward_y = R[1,0].
    forward_x = 1.0 - 2.0 * (y * y + z * z)
    forward_y = 2.0 * (x * y + w * z)
    return torch.atan2(forward_y, forward_x)


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angle(s) into ``[-pi, pi]``.

    Args:
        angle: Angles of any shape, in radians.

    Returns:
        Wrapped angles with the same shape, in ``[-pi, pi]``.
    """
    two_pi = 2.0 * math.pi
    wrapped = (angle + math.pi) % two_pi - math.pi
    # Map -pi (from the modulo) back to pi so the range is [-pi, pi].
    return torch.where(wrapped < -math.pi + 1e-6, wrapped + two_pi, wrapped)


def matvec(m: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Batched matrix-vector product.

    Args:
        m: Matrices of shape ``(..., 3, 3)``.
        v: Vectors of shape ``(..., 3)``, broadcast-compatible with ``m``.

    Returns:
        Vectors of shape ``(..., 3)``: ``(m * v.unsqueeze(-2)).sum(-1)``.
    """
    return (m * v.unsqueeze(-2)).sum(-1)


def matmul(m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
    """Batched matrix multiplication with broadcasting.

    Args:
        m1: Matrices of shape ``(..., 3, 3)``.
        m2: Matrices of shape ``(..., 3, 3)``, broadcast-compatible.

    Returns:
        Matrices of shape ``(..., 3, 3)``.
    """
    return torch.matmul(m1, m2)


def so3_exp(v: torch.Tensor) -> torch.Tensor:
    """Exponential map from so(3) rotation vectors to SO(3) matrices.

    Uses Rodrigues' formula with a second-order Taylor branch near zero.

    Args:
        v: Rotation vectors (axis * angle) of shape ``(..., 3)``.

    Returns:
        Rotation matrices of shape ``(..., 3, 3)``.
    """
    theta = v.norm(dim=-1, keepdim=True)  # (..., 1)
    small = theta < 1e-6
    theta_safe = torch.where(small, torch.ones_like(theta), theta)

    # Coefficients A = sin(t)/t, B = (1 - cos(t))/t^2 with Taylor branches
    # near zero (A ~ 1 - t^2/6, B ~ 1/2 - t^2/24) to avoid 0/0.
    theta2 = theta * theta
    a = torch.where(small, 1.0 - theta2 / 6.0, torch.sin(theta) / theta_safe)
    b = torch.where(small, 0.5 - theta2 / 24.0, (1.0 - torch.cos(theta)) / (theta_safe * theta_safe))
    a = a.unsqueeze(-1)  # (..., 1, 1)
    b = b.unsqueeze(-1)

    vx, vy, vz = v.unbind(-1)
    zero = torch.zeros_like(vx)
    # Skew-symmetric cross-product matrix of the (un-normalized) vector v.
    k_row0 = torch.stack((zero, -vz, vy), dim=-1)
    k_row1 = torch.stack((vz, zero, -vx), dim=-1)
    k_row2 = torch.stack((-vy, vx, zero), dim=-1)
    k = torch.stack((k_row0, k_row1, k_row2), dim=-2)  # (..., 3, 3)

    eye = torch.eye(3, dtype=v.dtype, device=v.device).expand(k.shape)
    # R = I + sin(t)[k]x + (1-cos(t))[k]x^2 = I + A*[v]x + B*[v]x^2.
    return eye + a * k + b * matmul(k, k)


def so3_log(m: torch.Tensor) -> torch.Tensor:
    """Logarithm map from SO(3) matrices to so(3) rotation vectors.

    Returns the principal logarithm: angles lie in ``[0, pi]``, so
    ``so3_exp(so3_log(M)) == M`` for every valid rotation matrix ``M``, and
    ``so3_log(so3_exp(v)) == v`` for ``||v|| < pi``. At ``||v|| == pi`` the
    axis sign is ambiguous (double cover) and either representative may be
    returned.

    Args:
        m: Rotation matrices of shape ``(..., 3, 3)``.

    Returns:
        Rotation vectors (axis * angle) of shape ``(..., 3)``.
    """
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)  # (...,)

    # Skew part: (R - R^T) / 2 gives sin(theta) * [k]x.
    skew = 0.5 * (m - m.transpose(-1, -2))
    vec = torch.stack(
        (skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]), dim=-1
    )  # = sin(theta) * axis

    small = theta < 1e-6
    # Below theta ~ pi - 0.05 the skew/sin(theta) division loses too many
    # bits to cancellation; the symmetric (R + I)/2 = k k^T recovery is
    # uniformly more accurate there (error ~ ||R - R_pi|| * eps).
    near_pi = theta > math.pi - 0.05

    # General branch: axis = vec / sin(theta), scaled by theta.
    sin_theta = torch.sin(theta)
    sin_safe = torch.where(small, torch.ones_like(sin_theta), sin_theta)
    axis_general = vec / sin_safe.unsqueeze(-1)

    # Near-identity branch: theta/sin(theta) -> 1, so vec already approximates
    # the rotation vector; use the exact ratio evaluated at the clamped angle
    # to keep first-order accuracy without dividing by a tiny sine.
    axis_small = vec * (theta / sin_safe).unsqueeze(-1)

    # Near-pi branch: sin(theta) ~ 0; recover the axis from the symmetric
    # matrix (R + I)/2 = k k^T (exact at theta = pi). Take the column with
    # the largest diagonal pivot and normalize it — signs come out right.
    sym = (m + torch.eye(3, dtype=m.dtype, device=m.device)) * 0.5  # (..., 3, 3)
    diag = torch.diagonal(sym, dim1=-2, dim2=-1)  # (..., 3) = k_i^2
    pivot = diag.argmax(dim=-1)  # (...)
    axis_pi = torch.gather(sym, -1, pivot.unsqueeze(-1).unsqueeze(-1).expand(sym.shape[:-2] + (3, 1))).squeeze(-1)
    axis_pi = torch.nn.functional.normalize(axis_pi, dim=-1)

    axis = torch.where(small.unsqueeze(-1), axis_small, axis_general)
    axis = torch.where(near_pi.unsqueeze(-1), axis_pi, axis)
    return axis * theta.unsqueeze(-1)


def compute_se2_deltas(
    positions: torch.Tensor,
    quats: torch.Tensor,
    pos0: torch.Tensor | None = None,
    quat0: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute SE(2) pose deltas ``[dx, dy, dyaw]`` in the t=0 heading frame.

    For each future step ``t`` relative to step 0:

    - ``[dx, dy] = (R(yaw_0)^T @ (pos_t - pos_0))[0:2]`` — the world-frame
      displacement projected into the horizontal heading frame at ``t = 0``.
    - ``dyaw = wrap_to_pi(yaw_t - yaw_0)``.

    Args:
        positions: Positions of shape ``(batch, T, 3)``. If ``pos0`` is not
            given, ``T = H + 1`` and index 0 is the reference step; otherwise
            ``T = H`` and ``pos0`` supplies the reference.
        quats: W-first quaternions of shape ``(batch, T, 4)``, same ``T`` as
            ``positions``.
        pos0: Optional explicit reference position of shape ``(batch, 3)``.
        quat0: Optional explicit reference quaternion of shape ``(batch, 4)``.

    Returns:
        Deltas of shape ``(batch, H, 3)`` with ``[dx, dy, dyaw]``, where
        ``H = T`` if ``pos0``/``quat0`` are given, else ``H = T - 1``.

    Raises:
        ValueError: If only one of ``pos0``/``quat0`` is provided, or shapes
            mismatch.
    """
    if (pos0 is None) != (quat0 is None):
        raise ValueError("pos0 and quat0 must be provided together or both omitted.")
    if positions.shape[:-1] != quats.shape[:-1]:
        raise ValueError(f"positions/quats leading dims mismatch: {positions.shape} vs {quats.shape}")
    if positions.shape[-1] != 3 or quats.shape[-1] != 4:
        raise ValueError(f"Expected (..., 3) positions and (..., 4) quats, got {positions.shape}, {quats.shape}")

    if pos0 is None:
        pos0 = positions[..., 0, :]
        quat0 = quats[..., 0, :]
        positions = positions[..., 1:, :]
        quats = quats[..., 1:, :]

    yaw0 = yaw_from_quat(quat0)  # (batch,)
    cos0 = torch.cos(yaw0)
    sin0 = torch.sin(yaw0)

    # World-frame displacement.
    disp = positions - pos0.unsqueeze(-2)  # (batch, H, 3)
    dx_w = disp[..., 0]
    dy_w = disp[..., 1]

    # Project into heading frame at t=0: R(yaw0)^T @ disp (xy block).
    c = cos0.unsqueeze(-1)
    s = sin0.unsqueeze(-1)
    dx = c * dx_w + s * dy_w
    dy = -s * dx_w + c * dy_w

    dyaw = wrap_to_pi(yaw_from_quat(quats) - yaw0.unsqueeze(-1))

    return torch.stack((dx, dy, dyaw), dim=-1)
