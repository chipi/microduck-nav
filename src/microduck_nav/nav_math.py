"""Clearance, progress and risk maths of NVIDIA's navigation rewards, as functions of tensors.

Ported from mdp/geometry.py, mdp/commands.py (navigation_progress) and mdp/rewards.py of Isaac Lab
exp/microduck 802b8d5 (BSD-3-Clause). The formulas are theirs; quaternions are (w, x, y, z) as in mjlab
(theirs are (x, y, z, w)), and the environment lookups are arguments. `geom` is
(centres (n, k, 3), yaw quaternions (n, k, 4), local half sizes (k, 2), radii (k,)), or sizes per environment: (n, k, 2) and (n, k).
"""

from __future__ import annotations

import math

import torch

from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, yaw_quat

Geometry = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

ROBOT_RADIUS = 0.12


def obstacle_point_clearances(points: torch.Tensor, geom: Geometry, robot_radius: float) -> torch.Tensor:
    """Signed planar clearance for each (environment, sample point, obstacle)."""
    centers, rotations, half_sizes, radii = geom
    if half_sizes.ndim == 3:  # sizes per environment (the pool's boxes): (n, k, 2) and (n, k) line up with (n, points, k)
        half_sizes, radii = half_sizes[:, None], radii[:, None]
    offsets = points[:, :, None, :] - centers[:, None, :, :]
    rot = rotations[:, None].expand(-1, points.shape[1], -1, -1)
    local = quat_apply_inverse(rot, offsets)[..., :2]
    to_faces = local.abs() - half_sizes
    outside = to_faces.clamp_min(0.0).norm(dim=-1)
    inside = to_faces.amax(dim=-1).clamp_max(0.0)
    box = outside + inside
    circle = local.norm(dim=-1) - radii
    clearance = torch.where(radii > 0.0, circle, box) - robot_radius
    return torch.nan_to_num(clearance, nan=-robot_radius, posinf=4.0, neginf=-robot_radius)


def obstacle_clearances(pos: torch.Tensor, geom: Geometry, robot_radius: float = ROBOT_RADIUS) -> torch.Tensor:
    return obstacle_point_clearances(pos[:, None, :], geom, robot_radius)[:, 0]


def requested_path_clearance(
    pos: torch.Tensor,
    quat: torch.Tensor,
    command: torch.Tensor,
    geom: Geometry,
    robot_radius: float = ROBOT_RADIUS,
    lookahead_distance: float = 0.5,
    num_samples: int = 6,
) -> torch.Tensor:
    if num_samples < 2 or lookahead_distance <= 0.0:
        raise ValueError("Path clearance needs at least two samples and a positive lookahead distance.")
    direction = torch.zeros_like(command)
    speed = command[:, :2].norm(dim=-1)
    moving = speed > 1.0e-6
    direction[:, :2] = torch.where(moving[:, None], command[:, :2] / speed.clamp_min(1.0e-6)[:, None], 0.0)
    direction = quat_apply(yaw_quat(quat), direction)
    arc = torch.linspace(0.0, lookahead_distance, num_samples, device=command.device)
    yaw_per_metre = torch.where(moving, command[:, 2] / speed.clamp_min(1.0e-6), 0.0)
    half_turn = 0.5 * yaw_per_metre[:, None] * arc[None, :]
    cos_half, sin_half = half_turn.cos(), half_turn.sin()
    offsets = torch.stack(
        (
            direction[:, 0, None] * cos_half - direction[:, 1, None] * sin_half,
            direction[:, 0, None] * sin_half + direction[:, 1, None] * cos_half,
            torch.zeros_like(half_turn),
        ),
        dim=-1,
    )
    offsets = offsets * (arc[None, :] * torch.sinc(half_turn / torch.pi))[..., None]
    points = pos[:, None, :] + offsets
    clearance = obstacle_point_clearances(points, geom, robot_radius).flatten(start_dim=1).amin(dim=-1)
    return clearance - moving * (lookahead_distance / (2 * (num_samples - 1)))


def body_velocity(lin_vel_w: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    return quat_apply_inverse(yaw_quat(quat), lin_vel_w)


def navigation_progress(command_xy: torch.Tensor, lin_vel_w: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """Requested-direction progress in [-1, 1]; zero for standing."""
    speed = command_xy.norm(dim=-1)
    velocity = body_velocity(lin_vel_w, quat)
    progress = (velocity[:, :2] * command_xy).sum(dim=-1) / speed.square().clamp_min(0.05**2)
    return torch.nan_to_num(torch.where(speed >= 0.05, progress.clamp(-1.0, 1.0), 0.0))


def obstacle_approach_risk(
    pos: torch.Tensor,
    lin_vel_w: torch.Tensor,
    quat: torch.Tensor,
    applied_command_xy: torch.Tensor,
    geom: Geometry,
    safety_distance: float = 0.1,
    time_horizon: float = 2.0,
    prediction_dt: float = 0.1,
    robot_radius: float = ROBOT_RADIUS,
    commanded_risk_weight: float = 0.0,
) -> torch.Tensor:
    if safety_distance < 0.0 or time_horizon <= 0.0 or prediction_dt <= 0.0:
        raise ValueError("Approach risk needs a nonnegative margin and positive time intervals.")
    if not math.isfinite(commanded_risk_weight) or commanded_risk_weight < 0.0:
        raise ValueError("Commanded risk weight must be finite and nonnegative.")
    predicted = pos + lin_vel_w * prediction_dt
    commanded = torch.zeros_like(pos)
    commanded[:, :2] = applied_command_xy
    commanded = quat_apply(yaw_quat(quat), commanded)
    commanded_prediction = pos + commanded * prediction_dt
    clearances = obstacle_point_clearances(torch.stack((pos, predicted, commanded_prediction), dim=1), geom, robot_radius)
    current = clearances[:, 0]
    closing = ((current[:, None] - clearances[:, 1:]) / prediction_dt).clamp_min(0.0).amax(dim=1)
    remaining = (current - safety_distance).clamp_min(0.0)
    risk = ((closing - remaining / time_horizon) / 0.20).clamp(0.0, 1.0).square()
    risk = torch.where(closing > 1e-4, risk, 0.0).amax(dim=-1)
    if commanded_risk_weight == 0.0:
        return risk
    commanded_closing = ((current - clearances[:, 2]) / prediction_dt).clamp_min(0.0)
    commanded_risk = ((commanded_closing - remaining / time_horizon) / 0.20).clamp(0.0, 1.0).square()
    commanded_risk = torch.where(commanded_closing > 1e-4, commanded_risk, 0.0).amax(dim=-1)
    return risk + commanded_risk_weight * commanded_risk


def gated_progress(raw: torch.Tensor, risk: torch.Tensor) -> torch.Tensor:
    return torch.where(raw > 0, raw * (1.0 - risk), raw)


def velocity_tracking(
    command_xy: torch.Tensor,
    lin_vel_w: torch.Tensor,
    quat: torch.Tensor,
    path_clearance: torch.Tensor,
    progress: torch.Tensor,
    std: float = 0.15,
    clear_distance: float = 0.25,
) -> torch.Tensor:
    velocity = body_velocity(lin_vel_w, quat)
    tracking = torch.exp(-(command_xy - velocity[:, :2]).square().sum(dim=-1) / std**2)
    gate = (path_clearance / clear_distance).clamp(0.0, 1.0)
    gate = torch.where(command_xy.norm(dim=-1) < 0.05, 1.0, gate * progress.clamp(0.0, 1.0))
    return torch.nan_to_num(tracking * gate)


def yaw_tracking(
    command: torch.Tensor,
    ang_vel_z: torch.Tensor,
    path_clearance: torch.Tensor,
    progress: torch.Tensor,
    std: float = 0.35,
    clear_distance: float = 0.25,
) -> torch.Tensor:
    tracking = torch.exp(-(command[:, 2] - ang_vel_z).square() / std**2)
    gate = (path_clearance / clear_distance).clamp(0.0, 1.0)
    gate = torch.where(command[:, :2].norm(dim=-1) < 0.05, 1.0, gate * progress.clamp(0.0, 1.0))
    return torch.nan_to_num(tracking * gate)


def clearance_risk(clearance_min: torch.Tensor, safety_distance: float = 0.2) -> torch.Tensor:
    return ((safety_distance - clearance_min) / safety_distance).clamp(0.0, 1.0).square()


def command_rate(commands: torch.Tensor, previous: torch.Tensor, limits: torch.Tensor) -> torch.Tensor:
    return ((commands - previous) / limits).square().mean(dim=-1)
