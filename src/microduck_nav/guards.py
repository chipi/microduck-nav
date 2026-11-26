"""The duck navigator's reflex and cliff stop as a layer between the policy's command and the walker (R7b).

robotics-lab's navigator, per tick: the follower's command, the local controller's (the learned model) command, the guards, the
walker (robots/duck/src/duck_lab/nav/run.py `_go_to`, commit d794a45 of robotics-lab's mission-control branch: `_control` at 1242, the
cliff stop at 1263-1280, the reflex stop at 1337-1360, `move` at 1376). A guard that fires sends `Twist()`: vx and vyaw both zero.
This layer does the same to the command the action term has formed, before the walker sees it, and the action term keeps what the
guard sent as the applied command (learned.py `_command`: `self.residual.previous = app`, app being what the navigator last sent).

The rule is the navigator's, constant by constant (profile.json of robots/duck: sensors.tof8x8, footprint, error_budget):

- reflex (run.py:1337-1338): `vx > 0` and either an obstacle return inside the corridor the duck sweeps is nearer than
  reflex_ahead_m (`blocked_ahead`), or any valid return in the four central columns is nearer than emergency_mm (`near`).
  The nearest obstacle return is tof8x8.py `blocking_ahead` (313-319) over `depth_points` (280-310): a valid zone whose hit is
  neither "below", nor "floor" (under floor_margin_m + floor_margin_per_m per metre of horizontal range), nor "overhead" (over
  overhead_clear_m), ahead of the trunk (> 0) and within reflex_half_w_m of its heading (footprint radius_m + reflex_margin_m).
- cliff (run.py:1263 and 1248-1254): `vx > 0`, the frame's drop points (tof8x8.py `drop_points` 326-359) numbering at least min_zones,
  seen in DROP_FRAMES (3) new frames in a row, the nearest nearer than cliff_stop_m. A drop point is a beam that would reach the floor
  within look_m (and points steeply enough down: dz <= -0.15) where the zone returned nothing or a return past the floor line
  (range along the down axis >= height + extra_m), and lies within corridor_m of the trunk's heading. A zone the sensor's impairment
  blanked is no evidence of a drop (tof8x8.py:347); the env cannot tell a blanked zone from an empty one, so with impairments on every
  invalid zone is skipped, as the navigator does with its impairment on, and with them off every invalid zone counts.

Geometry is the sensor's true pose and beams (the navigator uses its calibrated mount, which is the same thing when calibrated), and
the frame is the delivered one: what the policy reads, latency and dropped frames included.

Not here: what the navigator does after a stop (it surveys, remembers the points, turns to face the path, replans: run.py:1264-1280 and
1341-1359) and its other refusals (`_arc_clear` 1281-1300, step_look 1320-1336, stuck 1361-1375). The stop lasts a tick here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from microduck_nav.tof_geometry import COLS, ROWS


@dataclass(frozen=True)
class GuardCfg:
    # profile sensors.tof8x8
    reflex_ahead_m: float = 0.30
    emergency_mm: float = 150.0
    reflex_margin_m: float = 0.04
    floor_margin_m: float = 0.08
    floor_margin_per_m: float = 0.03
    extra_m: float = 0.06
    look_m: float = 0.7
    min_zones: int = 2
    # profile footprint
    radius_m: float = 0.083
    overhead_m: float = 0.30
    # profile error_budget
    corridor_m: float = 0.25
    cliff_stop_m: float = 0.60
    # run.py:89
    drop_frames: int = 3
    # tof8x8.py:341 (a beam flatter than this says nothing about the floor nearby) and :332 (a sensor lower than this reads nothing)
    min_down: float = 0.15
    min_height_m: float = 0.02

    @property
    def reflex_half_w_m(self) -> float:
        return self.radius_m + self.reflex_margin_m  # tof8x8.py:147


CENTRAL_COLUMNS = tuple(range(COLS // 2 - 2, COLS // 2 + 2))  # tof8x8.py:192
CENTRAL_ZONES = tuple(r * COLS + c for r in range(ROWS) for c in CENTRAL_COLUMNS)


def _trunk_frame(points_xy: torch.Tensor, trunk_xy: torch.Tensor, trunk_yaw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(ahead, left) of world points relative to the trunk, turned to its heading (tof8x8.py: points are relative to the trunk)."""
    d = points_xy - trunk_xy[:, None, :]
    c, s = torch.cos(trunk_yaw)[:, None], torch.sin(trunk_yaw)[:, None]
    return c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]


def reflex_terms(
    ranges: torch.Tensor, dir_x: torch.Tensor, origin: torch.Tensor, rays: torch.Tensor, floor_z: torch.Tensor,
    trunk_xy: torch.Tensor, trunk_yaw: torch.Tensor, cfg: GuardCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(block [m], near [mm]) of a frame, per environment, inf where there is none: how far ahead the nearest obstacle return in the
    duck's corridor lies (tof8x8.py blocking_ahead over depth_points), and the nearest valid return in the central columns."""
    valid = ranges > 0
    along = ranges / dir_x.clamp_min(1e-6)  # axial ranges to the distance along the beam (along_beam, range_kind axial)
    point = origin[:, None, :] + along[..., None] * rays
    z = point[..., 2] - floor_z[:, None]
    h = along * rays[..., :2].norm(dim=-1)
    floor_top = cfg.floor_margin_m + cfg.floor_margin_per_m * h
    obstacle = valid & (z >= -cfg.extra_m) & (z >= floor_top) & (z <= cfg.overhead_m)  # not below, not floor, not overhead
    ahead, left = _trunk_frame(point[..., :2], trunk_xy, trunk_yaw)
    inside = obstacle & (ahead > 0) & (left.abs() <= cfg.reflex_half_w_m)
    block = torch.where(inside, ahead, torch.full_like(ahead, torch.inf)).amin(dim=-1)
    central = list(CENTRAL_ZONES)
    near = torch.where(valid[:, central], ranges[:, central] * 1000.0, torch.full_like(ranges[:, central], torch.inf)).amin(dim=-1)
    return block, near


def drop_terms(
    ranges: torch.Tensor, dir_x: torch.Tensor, origin: torch.Tensor, rays: torch.Tensor, floor_z: torch.Tensor,
    trunk_xy: torch.Tensor, trunk_yaw: torch.Tensor, blanked_is_unknown: bool, cfg: GuardCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(has [bool], nearest [m]) of a frame's drop points (tof8x8.py drop_points): has is true when at least min_zones zones are
    drops, nearest the smallest horizontal distance to one of them (inf if none)."""
    valid = ranges > 0
    along = ranges / dir_x.clamp_min(1e-6)
    height = origin[:, 2] - floor_z
    dz = rays[..., 2]
    steep = dz <= -cfg.min_down
    t_floor = -height[:, None] / torch.where(steep, dz, torch.full_like(dz, -1.0))  # slant range to the floor along each beam
    h_floor = t_floor * rays[..., :2].norm(dim=-1)  # horizontal distance to that floor point
    reached = valid & (along * -dz < height[:, None] + cfg.extra_m)  # the return is the floor, or something on it
    candidate = steep & (h_floor <= cfg.look_m) & ~reached
    if blanked_is_unknown:
        candidate &= valid
    floor_xy = origin[:, None, :2] + t_floor[..., None] * rays[..., :2]
    _, left = _trunk_frame(floor_xy, trunk_xy, trunk_yaw)
    drop = candidate & (left.abs() <= cfg.corridor_m)
    has = (drop.sum(dim=-1) >= cfg.min_zones) & (height > cfg.min_height_m)
    nearest = torch.where(drop, h_floor, torch.full_like(h_floor, torch.inf)).amin(dim=-1)
    return has, nearest


class GuardLayer:
    """The two stops, with the drop streak the navigator keeps between frames."""

    def __init__(self, num_envs: int, device, impairments: bool, cfg: GuardCfg | None = None):
        self.cfg = cfg or GuardCfg()
        self.impairments = impairments
        self.streak = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.last_frame = torch.full((num_envs,), -2.0, device=device)

    def reset(self, rows) -> None:
        self.streak[rows] = 0
        self.last_frame[rows] = -2.0

    def __call__(
        self, command: torch.Tensor, sensor, trunk_xy: torch.Tensor, trunk_yaw: torch.Tensor, floor_z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(the command after the guards, reflex fired, cliff fired) for the command the policy formed this tick."""
        cfg = self.cfg
        origin, rays, dir_x = sensor.world_geometry()
        ranges = sensor.delivered_frame
        have = sensor.available  # no frame yet: nothing to stop for (run.py:1246, self._block is None)
        drops, nearest = drop_terms(ranges, dir_x, origin, rays, floor_z, trunk_xy, trunk_yaw, self.impairments, cfg)
        drops &= have
        new = have & (sensor.channel.delivered_time != self.last_frame)  # frames, not ticks (run.py:1248)
        self.streak = torch.where(new, torch.where(drops, self.streak + 1, torch.zeros_like(self.streak)), self.streak)
        self.last_frame = torch.where(new, sensor.channel.delivered_time, self.last_frame)
        walking = command[:, 0] > 0
        cliff = walking & drops & (self.streak >= cfg.drop_frames) & (nearest < cfg.cliff_stop_m)
        block, near = reflex_terms(ranges, dir_x, origin, rays, floor_z, trunk_xy, trunk_yaw, cfg)
        reflex = walking & have & ((block < cfg.reflex_ahead_m) | (near < cfg.emergency_mm)) & ~cliff
        stop = cliff | reflex
        return torch.where(stop[:, None], torch.zeros_like(command), command), reflex, cliff
