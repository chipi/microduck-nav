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


def held_action(env) -> torch.Tensor:
    """The residuals the policy last sent, clipped as its deployment does; held through a tick, unlike mjlab's last_action,
    which is whatever was passed to the latest walker step."""
    return env.action_manager.get_term("nav").raw_action.clamp(-1.0, 1.0)


def record_tick(env, force: bool = False) -> None:
    """Pushes the delivered frame and its age into the per-tick history when the policy is about to read it. Idempotent within
    a walker step, and a row that was reset mid-tick waits for the next tick."""
    nav = env.action_manager.get_term("nav")
    history = nav.tof_history
    if not (force or nav.at_tick):
        return
    step = env.common_step_counter
    rows = torch.ones_like(history.empty) if (force or step != history.last_step) else history.empty.clone()
    if rows.any():
        history.push(tof_ranges_and_validity(env), tof_age(env).squeeze(-1), rows, step)


def tof_ranges_history(env) -> torch.Tensor:
    """Ten policy ticks of 64 ranges then 64 flags, oldest first (what mjlab's history of tof_ranges_and_validity gives at 50 Hz)."""
    record_tick(env)
    return env.action_manager.get_term("nav").tof_history.ranges.flatten(1)


def tof_age_history(env) -> torch.Tensor:
    record_tick(env)
    return env.action_manager.get_term("nav").tof_history.ages
