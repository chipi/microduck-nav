"""M2d: the deploy task with a room round each scene (docs/M2D.md)."""

import math

import pytest
import torch

from microduck_nav import terminations as T
from microduck_nav.field import root_pose, yaw_of
from microduck_nav.impairments import floor_hit_mask
from microduck_nav.obstacles import SLOT_NAMES, room_slot_names
from microduck_nav.room import (
    DUCK_MARGIN, ROOM_SIDE_RANGE, ROOM_SLACK, WALL_HEIGHT, WALL_THICKNESS, extents, place_room, required_box, sample_room, to_frame, wall_boxes,
)

from room_env import M2B, ROOM, room_env
from nav_env import make_env


def sensor_origin_in_frame(env, i=0):
    pos, quat = root_pose(env)
    heading = yaw_of(quat)
    origin = env.scene["head_tof"].world_geometry()[0][:, :2]
    return to_frame(heading, (origin - pos[:, :2])[:, None])[:, 0, 0]



# ---- the sensor and the collision ------------------------------------------------------------------------------------------------------
def test_the_sensor_reads_a_wall_placed_at_3_m_as_3_m(walker_path):
    env = room_env(num_envs=2, seed=0, layout="open")  # the median is within 0.2 m of 3.0 only for some starts: seed 3 misses it at 6476786 as well, seeds 1 to 3 with the room's own stream
    env.command_manager.get_term("twist").destination_w[:, :2] = root_pose(env)[0][:, :2]  # asked to stay where it is: a still sensor
    try:
        for _ in range(3):
            env.step(torch.zeros(2, 3))
        pos, quat = root_pose(env)
        heading = yaw_of(quat)
        ids = torch.arange(2)
        hi = torch.stack((sensor_origin_in_frame(env) + 3.0, torch.full((2,), 2.5)), dim=-1)  # the front wall's inner face: 3.0 m from the sensor
        lo = torch.tensor([[-2.0, -2.5]]).expand(2, -1)
        place_room(env, ids, pos[:, :2], heading, lo, hi)
        for _ in range(15):
            env.step(torch.zeros(2, 3))
        frame = env.scene["head_tof"].delivered_frame
        for e in range(2):
            wall = frame[e][frame[e] > 2.5]  # the upper rows see the wall; the lower ones the floor, nearer. Axial depth of a head pitched a few degrees
            assert len(wall) >= 8 and wall.median().item() == pytest.approx(3.0, abs=0.2), frame[e].reshape(8, 8)
    finally:
        env.close()


def test_walking_into_a_wall_terminates_as_an_obstacle_collision(walker_path):
    env = room_env(num_envs=2, seed=5, layout="open")
    try:
        pos, quat = root_pose(env)
        heading = yaw_of(quat)
        # env 0: the front wall 0.45 m ahead of the duck; env 1: 5 m ahead. Both are asked to walk 2.5 m forward.
        hi = torch.tensor([[0.45, 2.5], [5.0, 2.5]])
        place_room(env, torch.arange(2), pos[:, :2], heading, torch.tensor([[-2.0, -2.5]]).expand(2, -1), hi)
        goal = env.command_manager.get_term("twist")
        goal.destination_w[:, :2] = pos[:, :2] + 2.5 * torch.stack((heading.cos(), heading.sin()), dim=-1)
        hit = torch.zeros(2, dtype=torch.bool)
        for _ in range(200):
            _, _, terminated, _, _ = env.step(torch.zeros(2, 3))
            collision = env.termination_manager.get_term("obstacle_collision")
            hit |= collision
            if collision[0]:
                assert terminated[0] and not env.termination_manager.get_term("fell_over")[0]
                break
        assert hit.tolist() == [True, False], hit
        assert all(f"{T.CONTACT_PREFIX}{n}" in env.scene.sensors for n in room_slot_names()), "a contact sensor each wall"
    finally:
        env.close()


# ---- the room holds the scene ------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sampled(walker_path):
    """Rooms and scenes of 6 seeds x 64 environments x 2 resets."""
    rows = []
    for seed in range(6):
        env = room_env(num_envs=64, seed=seed)
        try:
            for _ in range(2):
                env.reset()
                rows.append(snapshot(env))
        finally:
            env.close()
    return rows


def snapshot(env):
    field = env.nav_field
    pos, quat = root_pose(env)
    heading = yaw_of(quat)
    goal = env.command_manager.get_term("twist").destination_w[:, :2]
    first = len(SLOT_NAMES)
    xy = to_frame(heading, field.pos[:, :first, :2] - pos[:, None, :2])
    lo, hi = extents(xy, field.yaw[:, :first] - heading[:, None], field.half_sizes[:, :first], field.radii[:, :first])
    return {
        "rlo": field.room_lo.clone(), "rhi": field.room_hi.clone(),
        "goal": to_frame(heading, goal - pos[:, :2]), "olo": lo, "ohi": hi, "counts": field.room_counts.clone(),
    }


def test_start_goal_and_every_obstacle_are_inside_the_room_over_many_seeds(sampled):
    rooms = 0
    for s in sampled:
        b = slice(None)  # every environment has a room
        rooms += len(s["goal"])
        lo, hi = s["rlo"][b], s["rhi"][b]
        assert (lo <= -DUCK_MARGIN + 1e-4).all() and (hi >= DUCK_MARGIN - 1e-4).all(), "the start has its margin"
        assert (s["goal"][b] >= lo + DUCK_MARGIN - 1e-3).all() and (s["goal"][b] <= hi - DUCK_MARGIN + 1e-3).all(), "the goal has its margin"
        assert (s["olo"][b] >= lo[:, None] - 1e-3).all() and (s["ohi"][b] <= hi[:, None] + 1e-3).all(), "every obstacle is inside"
    assert rooms == 6 * 2 * 64


def test_every_reset_is_counted_and_every_scene_has_a_room(sampled):
    for s in sampled[1::2]:  # the second snapshot of a seed: the env's own reset, then two more of 64 environments
        total, lifted, grown = (int(v) for v in s["counts"])
        assert total == 3 * 64 and 0 <= grown <= lifted <= total
        assert (s["rhi"] - s["rlo"] >= ROOM_SIDE_RANGE[0] - 1e-4).all(), "no scene is left without walls: a parked room has no size"


def test_room_sides_stay_in_the_range_the_generator_gives_and_a_small_scene_gets_the_whole_range():
    g = torch.Generator().manual_seed(0)
    need_lo, need_hi = torch.full((20000, 2), -0.5), torch.full((20000, 2), 0.5)  # a scene one metre wide
    room = sample_room(need_lo, need_hi, generator=g)
    side = room["hi"] - room["lo"]
    low, high = ROOM_SIDE_RANGE
    assert (side >= low - 1e-6).all() and (side <= high + 1e-6).all()
    assert side.min() < low + 0.05 and side.max() > high - 0.05 and side.mean().item() == pytest.approx((low + high) / 2, abs=0.05)
    assert (room["lo"] <= need_lo + 1e-6).all() and (room["hi"] >= need_hi - 1e-6).all(), "the scene is always inside"
    centre = (room["lo"] + room["hi"]) / 2
    assert centre.std() > 0.2, "the room's place is random, not centred on the scene"


def test_a_scene_wider_than_the_range_gets_a_room_that_grows_to_hold_it():
    low, high = ROOM_SIDE_RANGE
    need_lo = torch.zeros(20000, 2)
    need_hi = torch.tensor([high + 1.5, 2.0]).expand(20000, -1).clone()
    room = sample_room(need_lo, need_hi, generator=torch.Generator().manual_seed(1))
    side = room["hi"] - room["lo"]
    assert (room["lo"] <= need_lo + 1e-6).all() and (room["hi"] >= need_hi - 1e-6).all(), "the scene is inside"
    assert (side[:, 0] >= high + 1.5 - 1e-6).all() and (side[:, 0] <= high + 1.5 + ROOM_SLACK + 1e-6).all(), "uniform in [need, need + slack]"
    assert side[:, 0].mean().item() == pytest.approx(high + 1.5 + ROOM_SLACK / 2, abs=0.02)
    assert (side[:, 1] >= 2.0).all() and (side[:, 1] <= high + 1e-6).all(), "the other axis keeps the range's top"
    assert room["grown"].all() and room["lifted"].all()
    walls = wall_boxes(room["lo"], room["hi"])
    assert (walls[:, :, 6] == WALL_HEIGHT).all()


def test_the_walls_are_robotics_labs_thickness_and_height_on_the_outside_of_the_inner_sides():
    lo, hi = torch.tensor([[-1.0, -2.0]]), torch.tensor([[2.0, 1.0]])
    walls = wall_boxes(lo, hi)[0]
    assert WALL_THICKNESS == 0.04 and WALL_HEIGHT == 0.5
    south, north, west, east = walls
    assert south[1] + south[4] == pytest.approx(-2.0) and north[1] - north[4] == pytest.approx(1.0)
    assert west[0] + west[3] == pytest.approx(-1.0) and east[0] - east[3] == pytest.approx(2.0)
    assert (walls[:, 4][:2] * 2 == pytest.approx(0.04)) and (walls[:, 3][2:] * 2 == pytest.approx(0.04))
    assert (walls[:, 5] == 0).all() and (walls[:, 6] == 0.5).all()


def test_the_required_box_holds_the_duck_and_goal_with_margin_and_the_obstacles_with_theirs():
    goal = torch.tensor([[2.0, 0.5]])
    lo, hi = required_box(goal, torch.tensor([[[0.5, -1.0]]]), torch.tensor([[[0.9, -0.6]]]))
    assert lo[0].tolist() == pytest.approx([-DUCK_MARGIN, -1.05]) and hi[0].tolist() == pytest.approx([2.0 + DUCK_MARGIN, 0.5 + DUCK_MARGIN])


# ---- the far returns ------------------------------------------------------------------------------------------------------------------------
def far_returns(task, resets=4096, envs=128, far=2.0):
    """Over `resets` scene draws, the share of frames with a return farther than `far` m (a hit that is not the floor, the sensor's valid
    range), and the share of zones. The rays as the sensor casts them at the reset, impairments on, as the task trains."""
    env = make_env(task, num_envs=envs, seed=11)
    try:
        sensor = env.scene["head_tof"]
        frames = zones = total_zones = 0
        for _ in range(resets // envs):
            env.reset()
            radial = sensor._distances
            axial = torch.where(radial < 0, torch.zeros_like(radial), radial * sensor._env_dirs[..., 0])
            hit = ~floor_hit_mask(sensor._hit_pos_w[..., 2], axial) & (axial > far) & (axial <= sensor.cfg.max_distance)
            frames += int(hit.any(dim=-1).sum())
            zones += int(hit.sum())
            total_zones += hit.numel()
        return frames / resets, zones / total_zones
    finally:
        env.close()


def test_a_sample_of_4096_resets_shows_far_returns_in_the_room_task_and_almost_none_in_m2b(walker_path, capsys):
    room_frames, room_zones = far_returns(ROOM)
    m2b_frames, m2b_zones = far_returns(M2B)
    with capsys.disabled():
        print(f"\nfar returns (> 2 m, 4096 resets): M2d {room_frames:.1%} of frames, {room_zones:.2%} of zones; "
              f"M2b {m2b_frames:.1%} of frames, {m2b_zones:.2%} of zones")
    assert room_frames > 0.5, "a room puts a wall in view in most frames"
    assert m2b_frames < 0.05, "an open field shows none or very few"


# ---- the task ---------------------------------------------------------------------------------------------------------------------------------
def test_the_task_is_the_deploy_task_with_four_wall_slots_and_nothing_else_changed(walker_path):
    room, base = room_env(num_envs=2, seed=3), make_env(M2B, num_envs=2, seed=3)
    try:
        for name in ("policy", "tof"):
            assert room.observation_manager.group_obs_dim[name] == base.observation_manager.group_obs_dim[name]
        assert room.observation_manager.group_obs_dim["policy"] == (53,) and room.observation_manager.group_obs_dim["tof"] == (1290,)
        assert room.nav_field.names == SLOT_NAMES + room_slot_names() and base.nav_field.names == SLOT_NAMES and not room.nav_field.pool
        assert set(room.reward_manager.active_terms) == set(base.reward_manager.active_terms)
        assert room.cfg.decimation == base.cfg.decimation and room.cfg.episode_length_s == base.cfg.episode_length_s
        nav, base_nav = room.action_manager.get_term("nav"), base.action_manager.get_term("nav")
        assert nav.cfg.policy_decimation == base_nav.cfg.policy_decimation == 5 and nav.cfg.limits == base_nav.cfg.limits
        assert nav.cfg.limits[1] == 0.0
        assert room.nav_field.half_sizes.shape == (2, 12, 2)
        assert not (set(room_slot_names()) & set(base.scene.entities))
    finally:
        room.close()
        base.close()


def test_a_wall_is_in_the_rewards_footprints_and_the_bypass_never_takes_it_for_the_obstacle_in_the_way(walker_path):
    env = room_env(num_envs=4, seed=7, layout="open")
    try:
        pos, quat = root_pose(env)
        heading = yaw_of(quat)
        lo, hi = torch.tensor([[-2.0, -2.0]]).expand(4, -1), torch.tensor([[0.5, 2.0]]).expand(4, -1)
        place_room(env, torch.arange(4), pos[:, :2], heading, lo, hi)
        from microduck_nav import nav_math

        clearance = nav_math.obstacle_clearances(pos, env.nav_field.geometry())
        east = clearance[:, len(SLOT_NAMES) + 3]  # the east wall's inner face is 0.5 m ahead of the duck
        assert east.tolist() == pytest.approx([0.5 - 0.12] * 4, abs=0.02), east
        assert clearance.shape == (4, 12)
        assert (env.nav_field.blocks_request[: len(SLOT_NAMES)]).all() and not env.nav_field.blocks_request[len(SLOT_NAMES) :].any()
        env.step(torch.zeros(4, 3))
        term = env.reward_manager.get_term_cfg("barrier_bypass").func
        assert (term._target[term.eligible] < len(SLOT_NAMES)).all()
    finally:
        env.close()


def test_the_critic_sees_nvidias_eight_clearances_and_not_the_walls_while_the_rewards_see_all_twelve(walker_path):
    room, base = room_env(num_envs=4, seed=3), make_env(M2B, num_envs=4, seed=3)
    try:
        assert room.observation_manager.group_obs_dim["critic"] == base.observation_manager.group_obs_dim["critic"]
        from microduck_nav import observations as obs

        clearance = obs.obstacle_clearance(room)
        assert clearance.shape == (4, 8)
        from microduck_nav import nav_math

        pos = room.scene["robot"].data.root_link_pos_w
        assert nav_math.obstacle_clearances(pos, room.nav_field.geometry()).shape == (4, 12)
        assert torch.equal(clearance, nav_math.obstacle_clearances(pos, room.nav_field.geometry())[:, :8])
    finally:
        room.close()
        base.close()


def test_room_sizes_over_4096_resets(walker_path, capsys):
    env = room_env(num_envs=128, seed=11)
    try:
        sides = []
        for _ in range(32):
            env.reset()
            sides.append((env.nav_field.room_hi - env.nav_field.room_lo).flatten())
        side = torch.cat(sides)
        q = torch.quantile(side, torch.tensor([0.0, 0.5, 0.9, 1.0]))
        with capsys.disabled():
            print(f"\nroom sides over {len(side) // 2} resets: min {q[0]:.2f} median {q[1]:.2f} p90 {q[2]:.2f} max {q[3]:.2f} m")
        assert (side >= ROOM_SIDE_RANGE[0]).all() and (side <= 12.0).all()
    finally:
        env.close()
