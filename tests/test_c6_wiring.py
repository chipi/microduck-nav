"""The action term, the request and the observation wiring, on the real env."""

import pytest
import torch

from microduck_nav.action import NavigationAction
from microduck_nav.field import root_pose, yaw_of
from microduck_nav.residual_action import ResidualCommand
from microduck_nav.walker import Walker

from nav_env import FULL, ONE_WALL, make_env


@pytest.fixture(scope="module")
def env(walker_path):
    e = make_env(FULL, num_envs=8, seed=12)
    yield e
    e.close()


def test_the_request_speed_floor_is_0_2_and_is_a_config(env):
    term = env.command_manager.get_term("twist")
    assert term.cfg.speed_range == (0.2, 0.35)
    for _ in range(5):
        env.reset()
        assert term.destination_speed.min() >= 0.2 and term.destination_speed.max() <= 0.35

    def edit(cfg):
        cfg.commands["twist"].speed_range = (0.15, 0.15)

    other = make_env(FULL, num_envs=4, seed=1, edit=edit)
    try:
        assert torch.allclose(other.command_manager.get_term("twist").destination_speed, torch.tensor(0.15))
    finally:
        other.close()


def test_the_destination_is_2_6_m_ahead_of_the_reset_pose_on_the_one_wall_layout(walker_path):
    e = make_env(ONE_WALL, num_envs=8, seed=2)
    try:
        pos, quat = root_pose(e)
        goal = e.command_manager.get_term("twist").destination_w
        d = goal - pos
        heading = yaw_of(quat)
        along = torch.cos(heading) * d[:, 0] + torch.sin(heading) * d[:, 1]
        across = -torch.sin(heading) * d[:, 0] + torch.cos(heading) * d[:, 1]
        assert torch.allclose(along, torch.full((8,), 2.6), atol=1e-4) and across.abs().max() < 1e-4
    finally:
        e.close()


def test_the_action_term_feeds_the_walker_its_61_inputs_and_sends_default_plus_output(env):
    term: NavigationAction = env.action_manager.get_term("nav")
    for _ in range(8):
        env.step(torch.zeros(8, 3))
    action = torch.tensor([[0.5, -0.2, 0.3]] * 8)
    proprio = env.obs_buf["walker_proprio"].clone()
    previous = term.residual.previous.clone()
    last = term._last_walker.clone()
    sensor = env.scene["head_tof"]
    shadow = ResidualCommand(8, env.step_dt, term.cfg.limits, term.cfg.slew, term.cfg.max_age, term.cfg.max_lateral_ratio)
    shadow.previous.copy_(previous)
    expected_cmd = shadow.process(env.command_manager.get_command("twist"), action, sensor.available, sensor.sample_age)
    expected_in = torch.cat((proprio, last, expected_cmd, torch.zeros(8, 10)), dim=-1)
    assert expected_in.shape == (8, 61)
    term.process_actions(action)
    assert torch.allclose(term.velocity_commands, expected_cmd)
    walker = Walker.from_onnx(term.cfg.walker_onnx or __import__("microduck_nav.action", fromlist=["walker_path"]).walker_path())
    expected_out = walker(expected_in)
    assert torch.allclose(term._last_walker, expected_out, atol=1e-6)
    robot = env.scene["robot"]
    target = term._joints._processed_actions
    assert torch.allclose(target, expected_out + robot.data.default_joint_pos[:, term._joints._target_ids], atol=1e-6)


def test_the_command_never_exceeds_the_request_and_stops_without_a_frame(env):
    env.reset()
    term = env.action_manager.get_term("nav")
    sensor = env.scene["head_tof"]
    assert not sensor.available.any()
    env.step(torch.ones(8, 3))
    assert (term.velocity_commands == 0).all(), "no frame yet: the command is zero"
    for _ in range(40):
        env.step(torch.ones(8, 3))
        request = env.command_manager.get_command("twist")
        assert (term.velocity_commands[:, 0] <= request[:, 0] + 1e-5).all(), "forward speed is capped by the request"
    assert (term.velocity_commands[:, 0] > 0).any()
    assert (term.velocity_commands[:, 0] <= 0.35 + 1e-6).all()


def test_the_walker_obs_group_carries_pollens_noise_and_the_policy_group_its_own(env):
    names = env.observation_manager.active_terms["walker_proprio"]
    assert names == ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel"]
    assert env.observation_manager.cfg["walker_proprio"].enable_corruption
    a = env.obs_buf["walker_proprio"][:, :6]
    b = env.obs_buf["policy"][:, :6]
    assert not torch.equal(a, b), "independent noise draws, as NVIDIA's two observation managers"
    assert torch.allclose(a[:, 3:6], b[:, 3:6], atol=0.2), "both see the same projected gravity up to noise"
