"""Run Pollen's walker in mjlab's VelStand task on CPU at a constant forward command and measure the speed got."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

import mjlab_microduck.tasks  # noqa: F401  registers the tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from microduck_nav.walker import Walker

TASK = "Mjlab-VelStand-Flat-Backlash-MicroDuck"
DISTURBANCE_EVENTS = ("push_robot", "topple_push", "random_prone_init")


def make_cfg(speed: float, num_envs: int, noise: bool = True, seed: int = 0):
    cfg = load_env_cfg(TASK, play=True)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.episode_length_s = 1.0e6
    cfg.curriculum = {}
    for name in DISTURBANCE_EVENTS:
        cfg.events.pop(name, None)
    cfg.observations["actor"].enable_corruption = noise
    tw = cfg.commands["twist"]
    tw.ranges.lin_vel_x = (speed, speed)
    tw.ranges.lin_vel_y = (0.0, 0.0)
    tw.ranges.ang_vel_z = (0.0, 0.0)
    tw.heading_command = False
    tw.ranges.heading = None
    for field in ("rel_standing_envs", "rel_heading_envs", "rel_world_envs", "rel_forward_envs", "rel_turn_in_place_envs"):
        setattr(tw, field, 0.0)
    tw.resampling_time_range = (1.0e6, 1.0e6)
    tw.debug_vis = False
    return cfg


@dataclass
class SpeedResult:
    asked: float
    steps: int
    num_envs: int
    noise: bool
    mean_vx_body: float
    mean_vx_settled: float
    per_env_settled: list[float]
    mean_world_speed_settled: float
    resets: int
    seconds: float


def measure(speed: float, num_envs: int = 4, steps: int = 500, settle: int = 250, noise: bool = True, seed: int = 0,
            device: str = "cpu", walker_path=None) -> SpeedResult:
    walker = Walker.from_onnx(walker_path)
    env = ManagerBasedRlEnv(make_cfg(speed, num_envs, noise, seed), device=device)
    robot = env.scene["robot"]
    obs, _ = env.reset()
    vx, world_pos, resets = [], [], 0
    start = time.time()
    with torch.no_grad():
        for _ in range(steps):
            action = walker(obs["actor"].float())
            obs, _, terminated, truncated, _ = env.step(action)
            resets += int((terminated | truncated).sum())
            vx.append(robot.data.root_link_lin_vel_b[:, 0].clone())
            world_pos.append(robot.data.root_link_pos_w[:, :2].clone())
    seconds = time.time() - start
    vx = torch.stack(vx)
    pos = torch.stack(world_pos)
    window = pos[-(steps - settle):]
    dt = env.step_dt
    world = (window[-1] - window[0]).norm(dim=-1) / ((window.shape[0] - 1) * dt)
    settled = vx[settle:].mean(dim=0)
    env.close()
    return SpeedResult(speed, steps, num_envs, noise, float(vx.mean()), float(settled.mean()),
                       [float(x) for x in settled], float(world.mean()), resets, seconds)
