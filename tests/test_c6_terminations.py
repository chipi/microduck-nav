from types import SimpleNamespace

import pytest
import torch

from microduck_nav import terminations as T
from microduck_nav.field import root_pose
from microduck_nav.obstacles import SLOT_NAMES

from nav_env import ONE_WALL, make_env


def fake_env(forces: dict[str, float], n=3):
    """One contact sensor per slot with a (n, 1, 4, 3) force history, one row pushed with `forces`."""
    scene = {}
    for name in SLOT_NAMES:
        hist = torch.zeros(n, 1, 4, 3)
        hist[0, 0, 2, 0] = forces.get(name, 0.0)
        scene[T.CONTACT_PREFIX + name] = SimpleNamespace(data=SimpleNamespace(force_history=hist))
    return SimpleNamespace(scene=scene)


def test_the_threshold_is_0_1_newton_on_any_slot_in_any_physics_step():
    assert not T.obstacle_collision(fake_env({}))[0]
    assert not T.obstacle_collision(fake_env({"crate": 0.1}))[0]
    assert not T.obstacle_collision(fake_env({"crate": 0.0999}))[0]
    assert T.obstacle_collision(fake_env({"crate": 0.1001}))[0]
    assert T.obstacle_collision(fake_env({"wall_right": 3.0}))[0]
    assert T.obstacle_collision(fake_env({"obstacle": 0.2}))[0] and not T.obstacle_collision(fake_env({"obstacle": 0.2}))[1:].any()


@pytest.fixture(scope="module")
def env():
    e = make_env(ONE_WALL, num_envs=4, seed=3)
    yield e
    e.close()


def test_a_wall_pushed_into_the_robot_terminates_that_world_only(env):
    env.reset()
    pos, _ = root_pose(env)
    for _ in range(3):
        env.step(torch.zeros(4, 3))
    assert not env.termination_manager.terminated.any()
    ids = torch.tensor([0])
    pose = torch.zeros(1, 7)
    pose[:, :2] = pos[0, :2] + torch.tensor([0.02, 0.0])
    pose[:, 3] = 1.0
    env.scene["wall_left"].write_mocap_pose_to_sim(pose, env_ids=ids)
    env.sim.forward()
    _, _, terminated, truncated, _ = env.step(torch.zeros(4, 3))
    forces = T.obstacle_forces(env)
    assert terminated.tolist() == [True, False, False, False]
    assert env.termination_manager.get_term("obstacle_collision").tolist() == [True, False, False, False]
    assert not truncated.any()


def test_a_wall_a_metre_away_does_not_terminate(env):
    env.reset()
    for _ in range(10):
        _, _, terminated, _, _ = env.step(torch.zeros(4, 3))
    assert not env.termination_manager.get_term("obstacle_collision").any()


def test_the_field_boundary_truncates_without_calling_it_a_collision(env):
    env.reset()
    robot = env.scene["robot"]
    far = torch.zeros(1, 7)
    far[:, :3] = env.scene.env_origins[1] + torch.tensor([5.5, 0.0, 0.14])
    far[:, 3] = 1.0
    robot.write_root_state_to_sim(torch.cat((far, torch.zeros(1, 6)), dim=1), env_ids=torch.tensor([1]))
    env.sim.forward()
    _, _, terminated, truncated, _ = env.step(torch.zeros(4, 3))
    assert env.termination_manager.get_term("field_boundary").tolist() == [False, True, False, False]
    assert truncated[1] and not terminated[1]
