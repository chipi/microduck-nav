"""The request: a fixed world destination ahead, steered to with a joystick-like command.

Ported from NVIDIA's MicroDuckNavigationCommand (mdp/commands.py:244-255 and the destination branch of
_resample_command/_update_command; Isaac Lab exp/microduck 802b8d5, BSD-3-Clause). Every episode is a
destination episode (rel_destination_envs = 1.0 in their task), so the joystick mixture of their base
class never reaches the actor and is not ported. The requested speed is drawn from `speed_range`; NVIDIA's
is (0.15, 0.35), here it starts at 0.2 because Pollen's walker does not walk below that (docs/CHECKS.md C3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from microduck_nav import layouts
from microduck_nav.field import get_field, quat_from_yaw, root_pose, yaw_of
from microduck_nav.obstacles import SLOT_INDEX


def rot2(yaw: torch.Tensor, v: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Rotate (n, 3) vectors about z by yaw (or by -yaw)."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    if inverse:
        s = -s
    return torch.stack((c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1], v[:, 2]), dim=-1)


@dataclass(kw_only=True)
class NavigationCommandCfg(CommandTermCfg):
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    speed_range: tuple[float, float] = (0.2, 0.35)
    max_yaw_rate: float = 0.75
    destination_distance: float = layouts.GOAL_DISTANCE
    destination_tolerance: float = 0.2
    destination_heading_gain: float = 1.5
    asset_name: str = "robot"

    def build(self, env) -> "NavigationCommand":
        return NavigationCommand(self, env)


def destination_direction(field, fresh: torch.Tensor, position: torch.Tensor, heading: torch.Tensor, distance: float) -> torch.Tensor:
    """The unit direction of the destination of the environments `fresh`, from the layout the field holds and the robot (NVIDIA's rule)."""
    forward = torch.zeros_like(position)
    forward[:, 0] = 1.0
    direction = rot2(heading, forward)
    layout = field.layout[fresh]
    names = layouts.LAYOUT_NAMES
    n = len(fresh)
    rows = torch.arange(n, device=position.device)
    slots = field.pos[fresh]
    ring = layout == names.index("ring")
    direction[ring] = (slots[:, SLOT_INDEX["obstacle"]] - position)[ring]
    front = slots[rows, field.front_index[fresh]] - position
    for name in ("barrier", "blocked", "corner"):
        mask = layout == names.index(name)
        direction[mask] = front[mask]
    passage = layout == names.index("passage")
    midpoint = 0.5 * (slots[:, SLOT_INDEX["wall_left"]] + slots[:, SLOT_INDEX["wall_right"]])
    direction[passage] = (midpoint - position)[passage]
    slalom = layout == names.index("slalom")
    local = torch.stack(
        [rot2(heading, slots[:, SLOT_INDEX[s]] - position, inverse=True) for s in ("crate", "tall_box", "cylinder")], dim=1
    )
    last = local[:, :, 0].argmax(dim=1)
    last_y = local[rows, last, 1]
    local_goal = forward * distance
    local_goal[:, 1] = torch.where(last_y >= 0.0, -0.5, 0.5)
    direction[slalom] = rot2(heading, local_goal)[slalom]
    direction[:, 2] = 0.0
    local_direction = rot2(heading, direction, inverse=True)
    minimum_forward = 1e-3 * direction.norm(dim=-1).clamp_min(1.0)
    adjust = local_direction[:, 0] < minimum_forward
    local_direction[:, 0] = torch.maximum(local_direction[:, 0].abs(), minimum_forward)
    direction[adjust] = rot2(heading, local_direction)[adjust]
    direction /= direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return direction


class NavigationCommand(CommandTerm):
    cfg: NavigationCommandCfg

    def __init__(self, cfg: NavigationCommandCfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        n = self.num_envs
        self.vel_command_b = torch.zeros(n, 3, device=self.device)
        self.is_destination_env = torch.ones(n, dtype=torch.bool, device=self.device)
        self.destination_w = torch.zeros(n, 3, device=self.device)
        self.destination_speed = torch.zeros(n, device=self.device)
        self.destination_reached = torch.zeros(n, dtype=torch.bool, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        fresh = env_ids[self.command_counter[env_ids] == 0]
        if len(fresh):
            self.destination_reached[fresh] = False
            self._sample_destinations(fresh)

    def _sample_destinations(self, fresh: torch.Tensor) -> None:
        """Fixed forward world goals; read from qpos, since derived poses are stale during a reset."""
        field = get_field(self._env)
        position, quat = root_pose(self._env, fresh, self.cfg.asset_name)
        direction = destination_direction(field, fresh, position, yaw_of(quat), self.cfg.destination_distance)
        self.destination_w[fresh] = position + self.cfg.destination_distance * direction
        low, high = self.cfg.speed_range
        self.destination_speed[fresh] = torch.empty(len(fresh), device=self.device).uniform_(low, high)

    def _update_command(self) -> None:
        pos = self.robot.data.root_link_pos_w
        yaw = yaw_of(self.robot.data.root_link_quat_w)
        offset = self.destination_w - pos
        offset[:, 2] = 0.0
        offset_b = rot2(yaw, offset, inverse=True)
        distance = offset_b[:, :2].norm(dim=-1)
        error = torch.atan2(offset_b[:, 1], offset_b[:, 0])
        self.destination_reached |= distance <= self.cfg.destination_tolerance
        self.vel_command_b[:, 0] = self.destination_speed * error.cos().clamp_min(0.0) * (distance / 0.5).clamp_max(1.0)
        self.vel_command_b[:, 1] = 0.0
        self.vel_command_b[:, 2] = (self.cfg.destination_heading_gain * error).clamp(-self.cfg.max_yaw_rate, self.cfg.max_yaw_rate)
        self.vel_command_b[self.destination_reached] = 0.0


@dataclass(kw_only=True)
class PoolNavigationCommandCfg(NavigationCommandCfg):
    """The same request law, the destination read from the pool item the environment holds (R7a)."""

    def build(self, env) -> "PoolNavigationCommand":
        return PoolNavigationCommand(self, env)


class PoolNavigationCommand(NavigationCommand):
    cfg: PoolNavigationCommandCfg

    def _sample_destinations(self, fresh: torch.Tensor) -> None:
        """The item's goal, placed by the robot's reset pose when the scene was (field.reset_pool_scene, which runs first)."""
        field = get_field(self._env)
        position, _ = root_pose(self._env, fresh, self.cfg.asset_name)
        self.destination_w[fresh, :2] = field.goal_w[fresh]
        self.destination_w[fresh, 2] = position[:, 2]
        low, high = self.cfg.speed_range
        self.destination_speed[fresh] = torch.empty(len(fresh), device=self.device).uniform_(low, high)
