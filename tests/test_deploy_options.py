"""The two deployment options: a policy that acts every N walker steps, and no lateral residual.

The deploy task (`Mjlab-Nav-Deploy-MicroDuck`) is the full task with both: the model is asked at 10 Hz (a 0.1 s tick of five 0.02 s
walker steps) and sends no lateral velocity, as robotics-lab's duck navigator does.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper

from microduck_nav.action_repeat import ActionRepeatVecEnv
from microduck_nav.obs_parity import DEPLOY_TASK, build_fixture
from microduck_nav.rl_cfg import make_rl_cfg

from nav_env import FULL, make_env

N = 5
DEPLOY_FIXTURE = Path(__file__).parent / "fixtures" / "obs_parity_deploy.json"


def face_forward(cfg):
    cfg.events["reset_base"].params["pose_range"]["yaw"] = (0.0, 0.0)  # a forward request keeps its vx and so its lateral room


def action(lateral: float, envs: int) -> torch.Tensor:
    return torch.tensor([[0.0, lateral, 0.0]] * envs)


@pytest.fixture(scope="module")
def deploy(walker_path):
    e = make_env(DEPLOY_TASK, num_envs=4, seed=3, edit=face_forward)
    yield e
    e.close()


def test_the_deploy_task_is_the_full_task_at_10_hz_with_no_lateral(deploy, walker_path):
    nav = deploy.action_manager.get_term("nav")
    assert nav.decimation == N and nav.cfg.limits == (0.35, 0.0, 0.75)
    assert nav.residual.dt == pytest.approx(0.1) and deploy.step_dt == pytest.approx(0.02)
    full = make_env(FULL, num_envs=2)
    try:
        term = full.action_manager.get_term("nav")
        assert term.decimation == 1 and term.cfg.limits == (0.35, 0.12, 0.75)  # the existing task is unchanged
        assert term.tof_history is None
        assert full.observation_manager.group_obs_dim["tof"] == deploy.observation_manager.group_obs_dim["tof"] == (1290,)
    finally:
        full.close()


def test_the_action_is_held_for_five_walker_steps(deploy):
    nav = deploy.action_manager.get_term("nav")
    deploy.reset()
    seen = []
    for step in range(3 * N):
        value = 0.1 * (step + 1)
        deploy.step(torch.full((4, 3), value))
        seen.append((nav.raw_action[0, 0].item(), nav.velocity_commands.clone(), nav._last_walker.clone()))
    # the action of step 0, 5 and 10 is taken; the other four of each tick are ignored
    for tick in range(3):
        taken = 0.1 * (tick * N + 1)
        assert [round(s[0], 6) for s in seen[tick * N:(tick + 1) * N]] == [round(taken, 6)] * N
    for tick in range(3):  # the command is formed once per tick and held, the walker runs on every step
        block = seen[tick * N:(tick + 1) * N]
        assert all(torch.equal(block[0][1], b[1]) for b in block)
        assert len({tuple(b[2][0].tolist()) for b in block}) == N


def test_the_full_task_still_takes_every_action(walker_path):
    e = make_env(FULL, num_envs=2)
    try:
        nav = e.action_manager.get_term("nav")
        for step in range(4):
            e.step(torch.full((2, 3), 0.1 * (step + 1)))
            assert nav.raw_action[0, 0].item() == pytest.approx(0.1 * (step + 1))
    finally:
        e.close()


def test_the_policy_reads_the_action_it_holds(deploy):
    deploy.reset()
    deploy.step(torch.full((4, 3), 0.5))
    for _ in range(N - 1):
        obs, *_ = deploy.step(torch.full((4, 3), -0.9))
    held = obs["policy"][:, 34:37]
    assert torch.allclose(held, torch.full((4, 3), 0.5))
    assert deploy.observation_manager.get_term_cfg("critic", "actions").func.__name__ == "held_action"


def lateral_seen_by_the_walker(task: str, envs: int = 4, steps: int = 40, **kwargs):
    e = make_env(task, num_envs=envs, seed=5, edit=face_forward)
    try:
        nav = e.action_manager.get_term("nav")
        sent = []
        nav.walker.register_forward_pre_hook(lambda _m, args: sent.append(args[0][:, 48:51].clone()))
        for _ in range(steps):
            e.step(action(1.0, envs))
        return torch.cat(sent), nav.velocity_commands.clone()
    finally:
        e.close()


def test_lateral_zero_gives_zero_vy_at_the_walker_where_the_default_gives_some(walker_path):
    default_cmd, _ = lateral_seen_by_the_walker(FULL)
    assert default_cmd[:, 0].abs().max() > 0.05  # it walks forward
    assert default_cmd[:, 1].abs().max() > 0.01, "the control: the default task's lateral residual reaches the walker"
    deploy_cmd, applied = lateral_seen_by_the_walker(DEPLOY_TASK)
    assert deploy_cmd[:, 0].abs().max() > 0.05
    assert (deploy_cmd[:, 1] == 0).all() and (applied[:, 1] == 0).all()


def test_the_command_rate_reward_is_finite_with_a_zero_lateral_limit(deploy):
    deploy.reset()
    for _ in range(2 * N):
        _, reward, *_ = deploy.step(action(1.0, 4))
        assert torch.isfinite(reward).all()


def test_gamma_keeps_its_discount_per_second_at_ten_hz():
    assert make_rl_cfg("x", 1).algorithm.gamma == 0.998
    assert make_rl_cfg("x", N).algorithm.gamma == pytest.approx(0.998**N)


@pytest.fixture(scope="module")
def sim_log(walker_path):
    """Steps the deploy env 12 policy ticks and notes what the ToF sensor holds when the policy would read it."""
    e = make_env(DEPLOY_TASK, num_envs=1, seed=7, edit=face_forward)
    nav, sensor = e.action_manager.get_term("nav"), e.scene["head_tof"]
    from microduck_nav.observations import tof_age, tof_ranges_and_validity

    ticks = [(tof_ranges_and_validity(e)[0].clone(), tof_age(e)[0, 0].item())]
    obs_at_tick = [e.obs_buf["tof"].clone()]
    for step in range(12 * N):
        obs, *_ = e.step(action(0.0, 1))
        assert nav.at_tick == ((step + 1) % N == 0)
        if nav.at_tick:
            ticks.append((tof_ranges_and_validity(e)[0].clone(), tof_age(e)[0, 0].item()))
            obs_at_tick.append(obs["tof"].clone())
    e.close()
    return ticks, obs_at_tick


def test_the_tof_history_holds_one_entry_per_tick_not_per_walker_step(sim_log):
    ticks, obs = sim_log
    for k in (10, 11, 12):
        vector = obs[k][0]
        ranges, ages = vector[:1280].reshape(10, 128), vector[1280:]
        assert torch.allclose(ranges, torch.stack([ticks[j][0] for j in range(k - 9, k + 1)]))
        assert np.allclose(ages.numpy(), [ticks[j][1] for j in range(k - 9, k + 1)], atol=1e-6)
    assert len({t[1] for t in ticks}) > 3, "the ages move, so the check is not vacuous"


def test_the_first_observation_of_an_episode_backfills_the_history_as_mjlab_does(sim_log):
    ticks, obs = sim_log
    first = obs[0][0]
    ranges = first[:1280].reshape(10, 128)
    assert torch.allclose(ranges, ranges[0].expand(10, -1)) and torch.allclose(first[1280:], first[1280:][0].expand(10))


def test_the_committed_deploy_fixture_is_what_the_training_code_builds_now():
    golden, fresh = json.loads(DEPLOY_FIXTURE.read_text()), build_fixture(DEPLOY_TASK)
    assert golden["task"] == DEPLOY_TASK and golden["step_s"] == pytest.approx(0.1) and golden["policy_decimation"] == N
    assert [s["name"] for s in golden["states"]] == [s["name"] for s in fresh["states"]]
    for old, new in zip(golden["states"], fresh["states"]):
        np.testing.assert_allclose(old["observation"], new["observation"], atol=1e-6, err_msg=old["name"])
    # regenerate with: uv run python scripts/export_obs_parity.py


def test_the_deploy_fixture_spaces_the_history_by_the_tick_and_ages_by_it():
    """Hand-computed slots of the 'walking' state: frames 0.17, 0.1 and 0.03 s old, entries 0.1 s apart, oldest first."""
    golden = json.loads(DEPLOY_FIXTURE.read_text())
    walking = next(s for s in golden["states"] if s["name"] == "walking")
    v = np.array(walking["observation"])
    ages = v[1333:]
    # newest entry: the 0.03 s frame; one tick back (0.1 s): the 0.1 s frame, as it was then (age 0); two back (0.2 s): no frame
    # had arrived yet, so nothing and 1 s. The 0.17 s frame is never the newest at any of the ten tick offsets.
    frames = walking["frames"]
    entry = lambda slot: v[53 + slot * 128: 53 + slot * 128 + 64]
    delivered = lambda f: np.where((np.array(f) >= 0.12) & (np.array(f) <= 4.0), f, 0.0)  # what the training counts as a return
    assert ages[9] == pytest.approx(0.03, abs=1e-6) and ages[8] == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(entry(9), delivered(frames[2]["axial_m"]), atol=1e-6)
    assert np.allclose(entry(8), delivered(frames[1]["axial_m"]), atol=1e-6)
    assert not entry(7).any() and ages[:8].tolist() == [1.0] * 8
    # against the 50 Hz fixture the same state has the same newest entry and a different, shorter history
    fast = json.loads((DEPLOY_FIXTURE.parent / "obs_parity.json").read_text())
    other = np.array(next(s for s in fast["states"] if s["name"] == "walking")["observation"])
    assert other[1333 + 9] == v[1333 + 9] and not np.allclose(other[53:1333], v[53:1333])


def test_the_wrapper_makes_one_transition_of_five_walker_steps(walker_path):
    def run(wrapped: bool):
        e = make_env(DEPLOY_TASK, num_envs=4, seed=9, edit=face_forward)
        try:
            torch.manual_seed(0)
            wrapper = RslRlVecEnvWrapper(e, clip_actions=1.0)
            a = torch.full((4, 3), 0.3)
            if wrapped:
                obs, reward, dones, extras = ActionRepeatVecEnv(wrapper, N).step(a)
                return reward, e.common_step_counter, dones
            total = torch.zeros(4)
            for _ in range(N):
                _, r, *_ = e.step(a)
                total += r
            return total, e.common_step_counter, None
        finally:
            e.close()

    reward, steps, dones = run(True)
    expected, raw_steps, _ = run(False)
    assert steps == raw_steps == N and dones.shape == (4,)
    assert torch.allclose(reward, expected, atol=1e-6)


def test_a_reset_mid_tick_does_not_leak_the_next_episode_into_the_transition(walker_path):
    def edit(cfg):
        face_forward(cfg)
        cfg.episode_length_s = 3 * 0.02  # every env times out on its third walker step, in the middle of the first tick

    e = make_env(DEPLOY_TASK, num_envs=2, seed=4, edit=edit)
    try:
        wrapper = ActionRepeatVecEnv(RslRlVecEnvWrapper(e, clip_actions=1.0), N)
        _, reward, dones, extras = wrapper.step(torch.zeros(2, 3))
        nav = e.action_manager.get_term("nav")
        assert (dones == 1).all() and extras["time_outs"].all()
        assert nav.at_tick and e.episode_length_buf.max() == N - 3  # it ran on under the held zero action to the end of the tick
        assert (nav.velocity_commands == 0).all()
        assert "log" in extras
    finally:
        e.close()
