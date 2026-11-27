"""M2d's walls are the only variable against M2b: for a fixed seed the room's own draws leave NVIDIA's obstacles, the start and the command's
goal and speed bit-identical (docs/M2D.md)."""

import torch

from microduck_nav.field import root_pose, yaw_of
from microduck_nav.obstacles import SLOT_NAMES

from nav_env import make_env
from room_env import M2B, ROOM

SEED, ENVS, RESETS = 5, 64, 3


def draws(task):
    env = make_env(task, num_envs=ENVS, seed=SEED)
    try:
        rows = []
        for _ in range(RESETS):
            env.reset()
            field, pos = env.nav_field, root_pose(env)
            first = len(SLOT_NAMES)
            command = env.command_manager.get_term("twist")
            rows.append({
                "obstacle_xy": field.pos[:, :first, :2].clone(), "obstacle_yaw": field.yaw[:, :first].clone(), "layout": field.layout.clone(),
                "start_pos": pos[0].clone(), "start_yaw": yaw_of(pos[1]).clone(), "goal": command.destination_w.clone(),
                "speed": command.destination_speed.clone(),
            })
        return rows
    finally:
        env.close()


def test_the_room_leaves_every_obstacle_the_start_and_the_command_bit_identical_to_m2b(walker_path):
    base, room = draws(M2B), draws(ROOM)
    for i, (a, b) in enumerate(zip(base, room)):
        for key in a:
            assert torch.equal(a[key], b[key]), f"reset {i}: {key} differs from M2b's"
