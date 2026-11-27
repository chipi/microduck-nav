"""NVIDIA's navigation reward terms on mjlab's env (mdp/rewards.py, Isaac Lab exp/microduck 802b8d5, BSD-3-Clause).

The pure maths lives in nav_math.py; the two stateful terms (goal arrival, obstacle bypass) are ported
line by line. Event-like terms divide by step_dt to cancel the reward manager's dt scaling, as theirs do.
"""

from __future__ import annotations

import torch

from mjlab.managers.manager_base import ManagerTermBase
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, yaw_quat

from microduck_nav import nav_math
from microduck_nav.action import COMMAND
from microduck_nav.field import get_field, root_pose


def _robot(env):
    robot = env.scene["robot"]
    return (
        robot.data.root_link_pos_w,
        robot.data.root_link_quat_w,
        robot.data.root_link_lin_vel_w,
        robot.data.root_link_ang_vel_w[:, 2],
    )


def _applied(env) -> torch.Tensor:
    return env.action_manager.get_term("nav").velocity_commands


def navigation_progress(env) -> torch.Tensor:
    _, quat, lin, _ = _robot(env)
    return nav_math.navigation_progress(env.command_manager.get_command(COMMAND)[:, :2], lin, quat)


def approach_risk(
    env, safety_distance: float = 0.2, time_horizon: float = 2.0, prediction_dt: float = 0.1,
    robot_radius: float = 0.12, commanded_risk_weight: float = 0.0,
) -> torch.Tensor:
    pos, quat, lin, _ = _robot(env)
    return nav_math.obstacle_approach_risk(
        pos, lin, quat, _applied(env)[:, :2], get_field(env).geometry(), safety_distance, time_horizon,
        prediction_dt, robot_radius, commanded_risk_weight,
    )


def gated_progress(env) -> torch.Tensor:
    risk = approach_risk(env, 0.2, 2.0, 0.1, 0.12, 0.0)
    return nav_math.gated_progress(navigation_progress(env), risk)


def command_tracking(env, std: float = 0.15, clear_distance: float = 0.25) -> torch.Tensor:
    pos, quat, lin, _ = _robot(env)
    command = env.command_manager.get_command(COMMAND)
    clearance = nav_math.requested_path_clearance(pos, quat, command, get_field(env).geometry())
    return nav_math.velocity_tracking(command[:, :2], lin, quat, clearance, navigation_progress(env), std, clear_distance)


def yaw_tracking(env, std: float = 0.35, clear_distance: float = 0.25) -> torch.Tensor:
    pos, quat, _, ang_z = _robot(env)
    command = env.command_manager.get_command(COMMAND)
    clearance = nav_math.requested_path_clearance(pos, quat, command, get_field(env).geometry())
    return nav_math.yaw_tracking(command, ang_z, clearance, navigation_progress(env), std, clear_distance)


def clearance_risk(env, safety_distance: float = 0.2) -> torch.Tensor:
    pos = env.scene["robot"].data.root_link_pos_w
    return nav_math.clearance_risk(nav_math.obstacle_clearances(pos, get_field(env).geometry()).amin(dim=-1), safety_distance)


def command_rate(env) -> torch.Tensor:
    term = env.action_manager.get_term("nav")
    return nav_math.command_rate(term.velocity_commands, term.previous_velocity_commands, term.velocity_limits)


def guard_intervention(env) -> torch.Tensor:
    """One for each policy tick at which a guard changed the command (guards.py), as an event: the weight is the cost of one."""
    return env.action_manager.get_term("nav").guard_event.float() / env.step_dt


def terminated_penalty(env) -> torch.Tensor:
    return env.termination_manager.terminated.float() / env.step_dt


class NavigationGoalArrival(ManagerTermBase):
    """Pay once on safe destination entry and revoke that credit on later failure."""

    def __init__(self, cfg, env):
        super().__init__(env)
        self._paid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._safe = torch.ones_like(self._paid)

    def reset(self, env_ids=None) -> None:
        self._paid[slice(None) if env_ids is None else env_ids] = False
        self._safe[slice(None) if env_ids is None else env_ids] = True

    def __call__(self, env) -> torch.Tensor:
        command = env.command_manager.get_term(COMMAND)
        position = env.scene["robot"].data.root_link_pos_w
        reached = (position[:, :2] - command.destination_w[:, :2]).norm(dim=-1) <= command.cfg.destination_tolerance
        safe = ~env.termination_manager.terminated & ~env.termination_manager.get_term("field_boundary")
        safe &= torch.isfinite(position).all(dim=-1)
        revoke = self._paid & ~safe
        self._safe &= safe
        event = command.is_destination_env & reached & self._safe & ~self._paid
        self._paid |= event
        self._paid &= ~revoke
        return (event.float() - revoke.float()) / env.step_dt


class NavigationObstacleBypass(ManagerTermBase):
    """Pay once for safely passing the nearest obstacle blocking the initial request."""

    def __init__(self, cfg, env):
        super().__init__(env)
        field = get_field(env)
        centers, rotations, self._half_sizes, self._radii = field.geometry()
        self._blocks = getattr(field, "blocks_request", None)  # (k,) the slots that can be the obstacle in the way; None: all
        self._per_env = self._half_sizes.ndim == 3  # the pool's boxes differ per environment: held with the centres, refreshed at reset
        if self._per_env:
            self._half_sizes, self._radii = self._half_sizes.clone(), self._radii.clone()
        self._centers, self._rotations = centers.clone(), rotations.clone()
        n, dev = env.num_envs, env.device
        self._previous = torch.zeros(n, 3, device=dev)
        self._initial_heading = torch.zeros(n, 4, device=dev)
        self._pending = torch.ones(n, dtype=torch.bool, device=dev)
        self._eligible = torch.zeros_like(self._pending)
        self._paid = torch.zeros_like(self._pending)
        self._left = torch.ones_like(self._pending)
        self._right = torch.ones_like(self._pending)
        self._target = torch.zeros(n, dtype=torch.long, device=dev)
        self._axis = torch.zeros_like(self._target)
        self._sign = torch.ones(n, device=dev)
        self._rows = torch.arange(n, device=dev)

    @property
    def eligible(self) -> torch.Tensor:
        return self._eligible

    @property
    def bypassed(self) -> torch.Tensor:
        return self._paid

    def _face(self, expanded: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
        """The target obstacle's expanded half side along `axis`, for every environment."""
        return expanded[self._rows, self._target, axis] if self._per_env else expanded[self._target, axis]

    def reset(self, env_ids=None) -> None:
        if env_ids is None or isinstance(env_ids, slice):
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = env_ids
        centers, rotations, half_sizes, radii = get_field(self._env).geometry()
        if self._per_env:
            self._half_sizes[ids], self._radii[ids] = half_sizes[ids], radii[ids]
        self._centers[ids], self._rotations[ids] = centers[ids], rotations[ids]
        pos, quat = root_pose(self._env, ids)
        self._previous[ids] = pos
        self._initial_heading[ids] = yaw_quat(quat)
        self._pending[ids] = True
        self._eligible[ids] = False
        self._paid[ids] = False
        self._left[ids] = True
        self._right[ids] = True

    def __call__(self, env) -> torch.Tensor:
        position = env.scene["robot"].data.root_link_pos_w
        previous_local = quat_apply_inverse(self._rotations, self._previous[:, None] - self._centers)[..., :2]
        current_local = quat_apply_inverse(self._rotations, position[:, None] - self._centers)[..., :2]
        expanded = self._half_sizes + 0.12
        request = env.command_manager.get_command(COMMAND).clone()
        self._select_initial_obstacle(previous_local, request, expanded)

        rows = self._rows
        previous_target = previous_local[rows, self._target]
        current_target = current_local[rows, self._target]
        start = previous_target[rows, self._axis] * self._sign
        end = current_target[rows, self._axis] * self._sign
        lateral_start = previous_target[rows, 1 - self._axis]
        lateral_end = current_target[rows, 1 - self._axis]
        half_length = self._face(expanded, self._axis)
        half_width = self._face(expanded, 1 - self._axis)
        motion = end - start
        none = motion.abs() < 1.0e-8
        safe_motion = torch.where(none, 1.0, motion)
        entry_f, exit_f = (-half_length - start) / safe_motion, (half_length - start) / safe_motion
        seg_entry = torch.where(none, 0.0, torch.minimum(entry_f, exit_f).clamp_min(0.0))
        seg_exit = torch.where(none, 1.0, torch.maximum(entry_f, exit_f).clamp_max(1.0))
        crosses = torch.where(none, start.abs() <= half_length, seg_entry <= seg_exit)
        at_entry = lateral_start + seg_entry * (lateral_end - lateral_start)
        at_exit = lateral_start + seg_exit * (lateral_end - lateral_start)
        self._left &= ~crosses | ((at_entry >= half_width) & (at_exit >= half_width))
        self._right &= ~crosses | ((at_entry <= -half_width) & (at_exit <= -half_width))
        safe = ~env.termination_manager.terminated & ~env.termination_manager.get_term("field_boundary")
        clearance = nav_math.obstacle_clearances(position, get_field(env).geometry()).amin(dim=-1)
        safe &= torch.isfinite(position).all(dim=-1) & (clearance >= 0.0)
        self._eligible &= safe
        event = self._eligible & (self._left | self._right) & (end >= half_length) & ~self._paid
        self._paid |= event
        self._previous.copy_(position)
        return event.float() / env.step_dt

    def _select_initial_obstacle(self, previous_local, request, expanded) -> None:
        request[:, 2] = 0.0
        direction_w = quat_apply(self._initial_heading, request)
        direction = quat_apply_inverse(self._rotations, direction_w[:, None].expand_as(self._centers))[..., :2]
        parallel = direction.abs() < 1.0e-8
        safe_direction = torch.where(parallel, 1.0, direction)
        t_min = (-self._half_sizes - previous_local) / safe_direction
        t_max = (self._half_sizes - previous_local) / safe_direction
        inside = previous_local.abs() <= self._half_sizes
        par_entry = torch.where(inside, -torch.inf, torch.inf)
        par_exit = torch.where(inside, torch.inf, -torch.inf)
        axis_entry = torch.where(parallel, par_entry, torch.minimum(t_min, t_max))
        axis_exit = torch.where(parallel, par_exit, torch.maximum(t_min, t_max))
        box_entry, entry_axis = axis_entry.max(dim=-1)
        box_exit = axis_exit.amin(dim=-1)
        speed_sq = direction.square().sum(dim=-1).clamp_min(1.0e-12)
        projection = (previous_local * direction).sum(dim=-1)
        discriminant = projection.square() - speed_sq * (previous_local.square().sum(dim=-1) - self._radii.square())
        circle_entry = (-projection - discriminant.clamp_min(0.0).sqrt()) / speed_sq
        is_circle = self._radii > 0.0
        entry_time = torch.where(is_circle, circle_entry, box_entry)
        ahead = torch.where(is_circle, discriminant >= 0.0, box_exit >= box_entry) & (entry_time >= 0.0)
        ahead &= request[:, :2].norm(dim=-1)[:, None] >= 0.05
        sign = direction.gather(-1, entry_axis[..., None]).squeeze(-1).sign()
        start_along = previous_local.gather(-1, entry_axis[..., None]).squeeze(-1) * sign
        entry_half = expanded.expand_as(previous_local).gather(-1, entry_axis[..., None]).squeeze(-1)
        ahead &= start_along < -entry_half
        if self._blocks is not None:
            ahead &= self._blocks
        nearest = torch.where(ahead, entry_time, torch.inf).argmin(dim=-1)
        self._target = torch.where(self._pending, nearest, self._target)
        self._axis = torch.where(self._pending, entry_axis[self._rows, nearest], self._axis)
        self._sign = torch.where(self._pending, sign[self._rows, nearest], self._sign)
        self._eligible = torch.where(self._pending, ahead[self._rows, nearest], self._eligible)
        self._pending.zero_()
