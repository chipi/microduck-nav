"""M2d (docs/M2D.md): a room round each training scene, four walls of the resizable box slots R7a built.

The room is a rectangle in the robot's reset-heading frame (the frame NVIDIA's layouts are drawn in), its sides drawn from
ROOM_SIDE_RANGE and its place drawn so that the duck's start, its goal and every one of NVIDIA's eight obstacles are inside, the duck
and the goal with a margin. Wall thickness and height are robotics-lab's (labrunner.scene.wall).

A side is drawn uniform in ROOM_SIDE_RANGE, and a room that does not hold the scene is drawn again: done in closed form, a side is
uniform in [max(need, low), high], which is the same distribution as drawing until it fits. Where the scene needs more than the range's
top (the layout is wider than the biggest room), the environment gets no room, its walls are parked, and it is counted.
"""

from __future__ import annotations

import torch

from mjlab.managers.event_manager import requires_model_fields

from microduck_nav import layouts
from microduck_nav.commands import destination_direction
from microduck_nav.field import FAR, get_field, place_boxes, root_pose, write_box_slots, yaw_of
from microduck_nav.obstacles import ROOM_WALLS, SLOT_NAMES

# The inner side of the room [m], each axis. docs/M2D.md says where the numbers come from (robotics-lab's generator: challenges.py).
ROOM_SIDE_RANGE = (1.6, 6.0)
WALL_THICKNESS = 0.04  # labrunner.scene.wall's `thick` default
WALL_HEIGHT = 0.5  # labrunner.scene.wall's `height` default
DUCK_MARGIN = 0.3  # the start's and the goal's distance from a wall at the least: room to turn
OBSTACLE_MARGIN = 0.05  # an obstacle's distance from a wall at the least: not touching
REPORT_EVERY = 100_000  # resets between lines of the room statistics


def to_frame(heading: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """World offsets (m, ..., 2) in the frame of the headings (m,)."""
    c, s = torch.cos(heading), torch.sin(heading)
    c, s = (v.reshape(-1, *([1] * (d.ndim - 2))) for v in (c, s))
    return torch.stack((c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]), dim=-1)


def extents(xy: torch.Tensor, yaw: torch.Tensor, half: torch.Tensor, radii: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The axis-aligned reach of footprints in a frame: per footprint its lower and upper corner (m, k, 2). `xy` and `yaw` are in that
    frame; `half` (k, 2) or (m, k, 2) and `radii` (k,) or (m, k) as nav_math's geometry (a radius above zero is a circle)."""
    c, s = yaw.cos().abs(), yaw.sin().abs()
    box = torch.stack((c * half[..., 0] + s * half[..., 1], s * half[..., 0] + c * half[..., 1]), dim=-1)
    reach = torch.where((radii > 0.0)[..., None], radii[..., None].expand_as(box), box)
    return xy - reach, xy + reach


def required_box(goal: torch.Tensor, obstacle_lo: torch.Tensor, obstacle_hi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """What a room must hold, in the heading frame (the duck is at its origin): the duck and the goal with DUCK_MARGIN, the obstacles with
    OBSTACLE_MARGIN. goal (m, 2), obstacle corners (m, k, 2)."""
    points = torch.stack((torch.zeros_like(goal), goal), dim=1)
    lo = torch.minimum((points - DUCK_MARGIN).amin(dim=1), (obstacle_lo - OBSTACLE_MARGIN).amin(dim=1))
    hi = torch.maximum((points + DUCK_MARGIN).amax(dim=1), (obstacle_hi + OBSTACLE_MARGIN).amax(dim=1))
    return lo, hi


def sample_room(
    need_lo: torch.Tensor, need_hi: torch.Tensor, side_range: tuple[float, float] = ROOM_SIDE_RANGE, generator: torch.Generator | None = None
) -> dict[str, torch.Tensor]:
    """A room round [need_lo, need_hi] (m, 2): its inner corners `lo`, `hi`, `built` (the need fits the range) and `lifted` (a room drawn
    smaller than the need would have been drawn again: the draw is the range cut at the need)."""
    low, high = side_range
    need = need_hi - need_lo
    built = (need <= high).all(dim=-1)
    floor = need.clamp_min(low)
    side = floor + (high - floor).clamp_min(0.0) * torch.rand(need.shape, generator=generator, device=need.device)
    side = torch.where(built[:, None], side, need)
    place = torch.rand(need.shape, generator=generator, device=need.device)
    lo = need_hi - side + place * (side - need)  # the room's lower corner: anywhere that keeps the need inside
    return {"lo": lo, "hi": lo + side, "built": built, "lifted": built & (need > low).any(dim=-1)}


def wall_boxes(lo: torch.Tensor, hi: torch.Tensor, built: torch.Tensor) -> torch.Tensor:
    """(m, 4, 7) of pool.FIELDS (x, y, yaw, hx, hy, z0, z1) in the heading frame: the four walls on the outside of [lo, hi], so that the inner
    sides are the room's. South and north run past the corners, west and east between them."""
    t, h = WALL_THICKNESS, WALL_HEIGHT
    cx, cy = (lo + hi)[:, 0] / 2, (lo + hi)[:, 1] / 2
    hw, hl = (hi - lo)[:, 0] / 2, (hi - lo)[:, 1] / 2
    x = torch.stack((cx, cx, lo[:, 0] - t / 2, hi[:, 0] + t / 2), dim=-1)
    y = torch.stack((lo[:, 1] - t / 2, hi[:, 1] + t / 2, cy, cy), dim=-1)
    hx = torch.stack((hw + t, hw + t, torch.full_like(hw, t / 2), torch.full_like(hw, t / 2)), dim=-1)
    hy = torch.stack((torch.full_like(hl, t / 2), torch.full_like(hl, t / 2), hl, hl), dim=-1)
    boxes = torch.zeros(len(lo), ROOM_WALLS, 7, device=lo.device)
    boxes[..., 0], boxes[..., 1], boxes[..., 3], boxes[..., 4], boxes[..., 6] = x, y, hx, hy, h
    return torch.where(built[:, None, None], boxes, torch.zeros_like(boxes))


def place_room(env, ids: torch.Tensor, anchor: torch.Tensor, heading: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, built: torch.Tensor) -> None:
    """Write the walls of the rooms [lo, hi] (the heading frame at `anchor`, a world xy) into the environments `ids` and the rewards'
    geometry. A room that is not built parks its walls, as the pool parks a slot. Needs geom_size per world (requires_model_fields)."""
    field = get_field(env)
    first = len(SLOT_NAMES)
    origins = env.scene.env_origins[ids]
    placed = place_boxes(wall_boxes(lo, hi, built), built.long() * ROOM_WALLS, anchor, heading, origins)
    field.pos[ids, first:, :2] = torch.where(placed["real"][..., None], placed["xy"], origins[:, None, :2] + FAR)
    field.pos[ids, first:, 2] = origins[:, 2:3]
    field.yaw[ids, first:] = placed["yaw"]
    field.half_sizes[ids, first:] = torch.where(placed["real"][..., None], placed["half"], torch.zeros_like(placed["half"]))
    field.room_lo[ids], field.room_hi[ids], field.room_built[ids] = lo, hi, built
    write_box_slots(env, ids, field.names[first:], field.geom_ids, placed["xy"], placed["zc"], placed["yaw"], placed["half"], placed["hz"])


@requires_model_fields("geom_size", "geom_rbound", "geom_aabb")
def reset_room(
    env,
    env_ids,
    goal_distance: float = layouts.GOAL_DISTANCE,
    side_range: tuple[float, float] = ROOM_SIDE_RANGE,
    generator: torch.Generator | None = None,
) -> None:
    """Build a room round the scene of each environment being reset; runs after reset_obstacle_field (the goal and the obstacles are
    read from what it placed). An environment whose scene is wider than the range allows gets no room and is counted."""
    field = get_field(env)
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None or isinstance(env_ids, slice) else env_ids
    if len(ids) == 0:
        return
    pos, quat = root_pose(env, ids)
    heading = yaw_of(quat)
    first = len(SLOT_NAMES)
    goal = goal_distance * destination_direction(field, ids, pos, heading, goal_distance)[:, :2]
    local_xy = to_frame(heading, field.pos[ids, :first, :2] - pos[:, None, :2])
    obstacle_lo, obstacle_hi = extents(
        local_xy, field.yaw[ids, :first] - heading[:, None], field.half_sizes[ids, :first], field.radii[ids, :first]
    )
    need_lo, need_hi = required_box(to_frame(heading, goal), obstacle_lo, obstacle_hi)
    room = sample_room(need_lo, need_hi, side_range, generator)
    place_room(env, ids, pos[:, :2], heading, room["lo"], room["hi"], room["built"])
    field.room_counts += torch.stack((room["built"].sum(), room["lifted"].sum(), (~room["built"]).sum())).long()
    seen = field.room_seen
    field.room_seen += len(ids)
    if field.room_seen // REPORT_EVERY != seen // REPORT_EVERY:
        print(room_line(field), flush=True)


def room_line(field) -> str:
    built, lifted, skipped = (int(v) for v in field.room_counts)
    total = built + skipped
    return (f"room: {total} resets, {built} with a room ({lifted} drawn again to fit the scene), {skipped} without ({skipped / max(total, 1):.1%}): "
            f"the scene is wider than {ROOM_SIDE_RANGE[1]} m")
