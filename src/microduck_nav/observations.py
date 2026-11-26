"""Observation terms of the navigation task (mdp/observations.py, Isaac Lab exp/microduck 802b8d5, BSD-3-Clause)."""

from __future__ import annotations

import torch

from microduck_nav.action import SENSOR
from microduck_nav.field import get_field
from microduck_nav import nav_math

MIN_RANGE = 0.12


def tof_ranges_and_validity(env) -> torch.Tensor:
    """64 metre ranges, then 64 valid flags; a missing return is not free space."""
    sensor = env.scene[SENSOR]
    ranges = sensor.delivered_frame
    valid = torch.isfinite(ranges) & (ranges >= sensor.cfg.randomization.minimum_distance) & (ranges <= sensor.cfg.max_distance)
    return torch.cat((torch.where(valid, ranges, 0.0), valid.float()), dim=-1)


def tof_age(env) -> torch.Tensor:
    """Delivered frame age [s], including transport delay and dropped frames; 1 s and growing before the first frame."""
    return env.scene[SENSOR].sample_age.unsqueeze(-1)


def applied_velocity_commands(env) -> torch.Tensor:
    return env.action_manager.get_term("nav").velocity_commands


def zeros(env, dim: int) -> torch.Tensor:
    return torch.zeros(env.num_envs, dim, device=env.device)


def obstacle_clearance(env) -> torch.Tensor:
    pos = env.scene["robot"].data.root_link_pos_w
    return nav_math.obstacle_clearances(pos, get_field(env).geometry())
