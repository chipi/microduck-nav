"""Obstacle layout families, sampled in the robot's reset-heading frame.

Adapted from NVIDIA's mdp/events.py (Isaac Lab exp/microduck 802b8d5, BSD-3-Clause): same families,
same bounds, same slot cells; the random draws take an explicit torch.Generator, so a seed gives
the same layouts. "one_wall" is added for milestone M1: both wall slots abut into one 1.8 m wall
across the approach. A pose is (forward, sideways, yaw) before the heading is applied.
"""

from __future__ import annotations

import math

import torch

from microduck_nav.obstacles import SLOT_INDEX, SLOT_NAMES

OBSTACLE_CELLS = {
    "obstacle": (1.4, 0.0),
    "crate": (1.4, 1.4),
    "tall_box": (0.0, 1.4),
    "low_block": (-1.4, 1.4),
    "cylinder": (-1.4, 0.0),
    "thin_post": (-1.4, -1.4),
    "wall_left": (0.0, -1.4),
    "wall_right": (1.4, -1.4),
}
assert tuple(OBSTACLE_CELLS) == SLOT_NAMES

LAYOUT_NAMES = ("ring", "open", "barrier", "slalom", "passage", "thin_low", "blocked", "corner", "scattered", "one_wall")

NVIDIA_WEIGHTS = {
    "ring": 0.0,
    "open": 0.10,
    "barrier": 0.10,
    "slalom": 0.0,
    "passage": 0.05,
    "thin_low": 0.05,
    "blocked": 0.02,
    "corner": 0.03,
    "scattered": 0.65,
}

PARKED = {
    "obstacle": (-0.8, -1.6, 0.0),
    "crate": (-1.6, -0.8, 0.0),
    "tall_box": (-1.6, 0.0, 0.0),
    "low_block": (-1.6, 0.8, 0.0),
    "cylinder": (-0.8, 1.6, 0.0),
    "thin_post": (0.0, 1.6, 0.0),
    "wall_left": (0.8, 1.6, 0.0),
    "wall_right": (0.8, -1.6, 0.0),
}

SHUFFLE_GROUP = ("obstacle", "crate", "tall_box", "low_block", "cylinder", "thin_post")
WALL_LENGTH = 0.9
GOAL_DISTANCE = 2.6


def _u(like: torch.Tensor, lo: float, hi: float, g: torch.Generator | None) -> torch.Tensor:
    return torch.empty_like(like).uniform_(lo, hi, generator=g)


def _rand(shape, g, device) -> torch.Tensor:
    return torch.rand(shape, generator=g, device=device)


def sample_layout_ids(
    count: int,
    layout: str,
    weights: dict[str, float] | None,
    g: torch.Generator | None = None,
    device="cpu",
    offset: int = 0,
) -> torch.Tensor:
    """Layout index per reset environment: 'mixed' draws from weights, 'balanced' cycles, a name is constant."""
    if layout == "mixed":
        if weights is None:
            return torch.randint(len(LAYOUT_NAMES), (count,), generator=g, device=device)
        w = torch.tensor([weights.get(n, 0.0) for n in LAYOUT_NAMES], device=device, dtype=torch.float32)
        if float(w.sum()) <= 0:
            raise ValueError("layout weights need a positive sum")
        return torch.multinomial(w, count, replacement=True, generator=g) if count else torch.zeros(0, dtype=torch.long, device=device)
    if layout == "balanced":
        return (torch.arange(count, device=device) + offset) % len(LAYOUT_NAMES)
    if layout in LAYOUT_NAMES:
        return torch.full((count,), LAYOUT_NAMES.index(layout), dtype=torch.long, device=device)
    raise ValueError(f"unknown layout {layout!r}; choose {LAYOUT_NAMES}, 'mixed' or 'balanced'")


def _sample_slalom(count, g, device) -> torch.Tensor:
    poses = torch.zeros((count, 3, 3), device=device)
    forward, sideways, yaw = poses.unbind(dim=-1)
    forward[:, 0].uniform_(0.60, 0.85, generator=g)
    spacing = _u(forward[:, 1:], 0.55, 0.75, g)
    forward[:, 1:] = forward[:, :1] + spacing.cumsum(-1)
    mirror = torch.randint(0, 2, (count, 1), generator=g, device=device) * 2 - 1
    sideways[:] = _u(sideways, 0.18, 0.38, g)
    sideways *= mirror * torch.tensor([1, -1, 1], device=device)
    yaw.uniform_(-0.5, 0.5, generator=g)
    order = _rand((count, 3), g, device).argsort(dim=-1)
    return poses[torch.arange(count, device=device)[:, None], order]


def _sample_two_obstacle_course(count, g, device) -> torch.Tensor:
    poses = torch.zeros((count, 2, 3), device=device)
    forward, sideways, yaw = poses.unbind(dim=-1)
    forward[:, 0].uniform_(0.55, 0.60, generator=g)
    forward[:, 1] = forward[:, 0] + _u(forward[:, 0], 0.50, 0.55, g)
    sideways.uniform_(-0.12, 0.12, generator=g)
    yaw.uniform_(-0.15, 0.15, generator=g)
    order = _rand((count, 2), g, device).argsort(dim=-1)
    return poses[torch.arange(count, device=device)[:, None], order]


def _set_passage_walls(local_poses, selected, distance, half_gap, heading) -> None:
    forward, sideways, yaw = local_poses.unbind(dim=-1)
    for wall, side in (("wall_left", 1), ("wall_right", -1)):
        slot = SLOT_INDEX[wall]
        lateral = side * (half_gap + 0.06)
        forward[selected, slot] = distance * heading.cos() - lateral * heading.sin()
        sideways[selected, slot] = distance * heading.sin() + lateral * heading.cos()
        yaw[selected, slot] = math.pi / 2 + heading


def _sample_side_wall_approaches(local_poses, layout_ids, layout_offset, front_range, g) -> None:
    front = SLOT_INDEX["obstacle"]
    forward, sideways, yaw = local_poses.unbind(dim=-1)
    blocked = layout_ids == LAYOUT_NAMES.index("blocked")
    corner = layout_ids == LAYOUT_NAMES.index("corner")
    selected = blocked | corner
    layout_offset[selected, 1] = 0.0
    forward[selected, front] = _u(layout_offset[selected, 0], 0.55, 1.05, g)
    low, high = front_range
    if low != 0.55 or high != 1.05:
        forward[blocked, front] = low + (forward[blocked, front] - 0.55) * ((high - low) / 0.5)
    sideways[selected, front] = 0.0
    yaw[selected, front] = 0.0
    half_gap = _u(layout_offset[:, 0], 0.21, 0.28, g)
    half_gap[blocked] = _u(half_gap[blocked], 0.70, 0.80, g)
    for wall, mask, side in (("wall_left", selected, 1), ("wall_right", blocked, -1)):
        slot = SLOT_INDEX[wall]
        forward[mask, slot] = forward[mask, front] - 0.52
        sideways[mask, slot] = side * (half_gap[mask] + 0.06)
        yaw[mask, slot] = math.pi / 2
    mirror = torch.randint(0, 2, (int(corner.sum()), 1), generator=g, device=local_poses.device) * 2 - 1
    sideways[corner] *= mirror
    yaw[corner] *= mirror


def _sample_occupied_passages(local_poses, passage, g) -> None:
    device = local_poses.device
    front = SLOT_INDEX["obstacle"]
    forward, sideways, yaw = local_poses.unbind(dim=-1)
    occupied = passage.clone()
    occupied[passage] = _rand(int(passage.sum()), g, device) < 0.5
    count = int(occupied.sum())
    like = torch.zeros(count, device=device)
    distance = _u(like, 1.35, 1.50, g)
    half_width = _u(like, 1.0, 1.15, g)
    heading = _u(like, -0.2, 0.2, g)
    lateral_offset = _u(like, -0.12, 0.12, g)
    forward[occupied, front] = distance * heading.cos() - lateral_offset * heading.sin()
    sideways[occupied, front] = distance * heading.sin() + lateral_offset * heading.cos()
    yaw[occupied, front] = heading + _u(like, -0.15, 0.15, g)
    _set_passage_walls(local_poses, occupied, distance, half_width, heading)


def _sample_scattered_course(local_poses, scattered, g) -> None:
    device = local_poses.device
    forward, sideways, yaw = local_poses.unbind(dim=-1)
    count = int(scattered.sum())
    row_offset = _u(torch.zeros((count, 2), device=device), 0.15, 0.35, g)
    mirror = torch.randint(0, 2, (count,), generator=g, device=device) * 2 - 1
    row_offset *= mirror[:, None] * torch.tensor([1.0, -1.0], device=device)
    row_offset[_rand(count, g, device) < 0.5] = 0.0
    like = torch.zeros(count, device=device)
    for index, name in enumerate(SHUFFLE_GROUP):
        row, column = divmod(index, 3)
        slot = SLOT_INDEX[name]
        forward[scattered, slot] = 0.8 + 0.9 * row + _u(like, -0.025, 0.025, g)
        sideways[scattered, slot] = (column - 1) * 1.15 + row_offset[:, row] + _u(like, -0.05, 0.05, g)
        yaw[scattered, slot] = _u(like, -0.15, 0.15, g)
    for name, side in (("wall_left", 1), ("wall_right", -1)):
        slot = SLOT_INDEX[name]
        forward[scattered, slot] = _u(like, 1.1, 1.4, g)
        sideways[scattered, slot] = side * _u(like, 2.2, 2.5, g)
        yaw[scattered, slot] = math.pi / 2


def _sample_one_wall(local_poses, one_wall, distance_range, g) -> None:
    """Both wall slots abut across the approach: a 1.8 m wall, 0.12 m thick, in front of the robot."""
    device = local_poses.device
    forward, sideways, yaw = local_poses.unbind(dim=-1)
    like = torch.zeros(int(one_wall.sum()), device=device)
    distance = _u(like, *distance_range, g)
    lateral = _u(like, -0.10, 0.10, g)
    tilt = _u(like, -0.10, 0.10, g)
    for name, side in (("wall_left", 1), ("wall_right", -1)):
        slot = SLOT_INDEX[name]
        along = side * WALL_LENGTH / 2
        forward[one_wall, slot] = distance - along * tilt.sin()
        sideways[one_wall, slot] = lateral + along * tilt.cos()
        yaw[one_wall, slot] = tilt


def sample_local_poses(
    layout_ids: torch.Tensor,
    jitter: float = 0.15,
    spacing_scale: float = 1.6,
    blocked_front_distance_range: tuple[float, float] = (0.55, 1.05),
    one_wall_distance_range: tuple[float, float] = (0.6, 1.0),
    g: torch.Generator | None = None,
) -> torch.Tensor:
    """(count, 8, 3) poses (forward, sideways, yaw) of every slot, in the reset-heading frame."""
    device = layout_ids.device
    count = len(layout_ids)
    parked = torch.tensor([PARKED[n] for n in SLOT_NAMES], device=device)
    local_poses = parked[None].expand(count, -1, -1).clone()
    local_poses[:, :, :2] += _u(local_poses[:, :, :2], -0.08, 0.08, g)
    local_poses[:, :, 2] = _u(local_poses[:, :, 2], -math.pi, math.pi, g)
    layout_offset = _u(torch.zeros((count, 2), device=device), -jitter, jitter, g)
    forward, sideways, yaw = local_poses.unbind(dim=-1)
    front = SLOT_INDEX["obstacle"]

    barrier = layout_ids == LAYOUT_NAMES.index("barrier")
    like = layout_offset[barrier, 0]
    forward[barrier, front] = _u(like, 0.65, 1.25, g)
    sideways[barrier, front] = _u(like, -0.18, 0.18, g)
    yaw[barrier, front] = _u(like, -0.6, 0.6, g)

    slalom = layout_ids == LAYOUT_NAMES.index("slalom")
    sequence = _sample_slalom(int(slalom.sum()), g, device)
    for slot, name in enumerate(("crate", "tall_box", "cylinder")):
        local_poses[slalom, SLOT_INDEX[name]] = sequence[:, slot]

    thin_low = layout_ids == LAYOUT_NAMES.index("thin_low")
    layout_offset[thin_low] /= 3.0
    sequence = _sample_two_obstacle_course(int(thin_low.sum()), g, device)
    for slot, name in enumerate(("thin_post", "low_block")):
        local_poses[thin_low, SLOT_INDEX[name]] = sequence[:, slot]

    passage = layout_ids == LAYOUT_NAMES.index("passage")
    distance = _u(layout_offset[passage, 0], 0.95, 1.60, g)
    half_gap = _u(distance, 0.20, 0.55, g)
    heading = _u(distance, -0.35, 0.35, g)
    _set_passage_walls(local_poses, passage, distance, half_gap, heading)
    _sample_side_wall_approaches(local_poses, layout_ids, layout_offset, blocked_front_distance_range, g)

    local_poses[:, :, :2] *= spacing_scale
    _sample_occupied_passages(local_poses, passage, g)
    _sample_scattered_course(local_poses, layout_ids == LAYOUT_NAMES.index("scattered"), g)
    _sample_one_wall(local_poses, layout_ids == LAYOUT_NAMES.index("one_wall"), one_wall_distance_range, g)
    local_poses[:, :, :2] += layout_offset[:, None, :]
    return local_poses


def shuffle_slots(count: int, group=SHUFFLE_GROUP, g: torch.Generator | None = None, device="cpu") -> torch.Tensor:
    """slot_order[env, slot] = which pose the slot takes; the six props permute among themselves, walls stay."""
    order = torch.arange(len(SLOT_NAMES), device=device).expand(count, -1).clone()
    idx = torch.tensor([SLOT_INDEX[n] for n in group], device=device)
    order[:, idx] = idx[_rand((count, len(idx)), g, device).argsort(dim=-1)]
    return order
