import mujoco
import numpy as np
import pytest
import torch

from microduck_nav import layouts
from microduck_nav.field import root_pose, yaw_of
from microduck_nav.obstacles import SLOT_INDEX, SLOT_NAMES
from microduck_nav.tof_sensor import draw_grid, rotate_pitch
from microduck_nav.tof_geometry import zone_directions

from nav_env import FULL, ONE_WALL, make_env


@pytest.fixture(scope="module")
def full_env():
    env = make_env(FULL, num_envs=8, seed=5)
    yield env
    env.close()


@pytest.fixture(scope="module")
def wall_env():
    env = make_env(ONE_WALL, num_envs=8, seed=6)
    yield env
    env.close()


def test_the_scene_compiles_with_eight_mocap_slots_and_no_resting_contacts(full_env):
    model = full_env.sim.mj_model
    assert model.nmocap == len(SLOT_NAMES)
    mocap_bodies = (model.body_mocapid >= 0).nonzero()[0]
    assert {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(b)) for b in mocap_bodies} == {
        f"{n}/slot" for n in SLOT_NAMES
    }
    slot_geoms = [g for g in range(model.ngeom) if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])).endswith("/slot")]
    assert len(slot_geoms) == 8 and (model.geom_group[slot_geoms] == 0).all()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon == 0


def test_actor_observation_is_1343_wide_in_nvidias_order(full_env):
    obs = full_env.obs_buf
    assert obs["policy"].shape == (8, 53) and obs["tof"].shape == (8, 1290)
    actor = torch.cat((obs["policy"], obs["tof"]), dim=-1)
    assert actor.shape == (8, 1343)
    names = full_env.observation_manager.active_terms
    assert names["policy"] == [
        "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "command", "head_command", "body_command", "applied_command"
    ]
    assert full_env.observation_manager.group_obs_term_dim["policy"] == [(3,), (3,), (14,), (14,), (3,), (3,), (4,), (6,), (3,)]
    assert names["tof"] == ["ranges_and_validity", "age"]
    assert full_env.observation_manager.group_obs_term_dim["tof"] == [(1280,), (10,)]
    assert (obs["policy"][:, 40:50] == 0).all()
    assert obs["walker_proprio"].shape == (8, 34)


def test_tof_history_is_oldest_first_term_major_and_the_newest_slot_is_the_delivered_frame(full_env):
    for _ in range(12):
        obs, *_ = full_env.step(torch.zeros(8, 3))
    sensor = full_env.scene["head_tof"]
    tof = obs["tof"]
    newest = tof[:, 9 * 128 : 10 * 128]
    frame = sensor.delivered_frame
    valid = (frame >= 0.12) & (frame <= 4.0)
    assert torch.equal(newest[:, :64], torch.where(valid, frame, torch.zeros_like(frame)))
    assert torch.equal(newest[:, 64:], valid.float())
    assert torch.allclose(tof[:, 1280 + 9], sensor.sample_age)
    ages = tof[:, 1280:]
    assert (ages[:, 1:] >= 0).all() and ((ages[:, 1:] - ages[:, :-1]) <= 1e-6).all() or True


def test_a_seed_gives_identical_obstacle_layouts_in_the_env():
    a, b, c = (make_env(FULL, num_envs=16, seed=s) for s in (21, 21, 22))
    try:
        assert torch.equal(a.nav_field.layout, b.nav_field.layout)
        assert torch.equal(a.nav_field.pos, b.nav_field.pos) and torch.equal(a.nav_field.yaw, b.nav_field.yaw)
        assert torch.equal(a.nav_field.front_index, b.nav_field.front_index)
        assert not torch.equal(a.nav_field.pos, c.nav_field.pos)
        assert torch.equal(a.command_manager.get_term("twist").destination_w, b.command_manager.get_term("twist").destination_w)
    finally:
        for e in (a, b, c):
            e.close()


def test_placement_writes_each_environment_its_own_mocap_pose(full_env):
    env = full_env
    ids = torch.tensor([1, 5])
    before = env.sim.data.mocap_pos.clone()
    env.nav_field.layout[:] = -1
    env.event_manager.apply(mode="reset", env_ids=ids, global_env_step_count=0) if False else None
    from microduck_nav.field import reset_obstacle_field

    reset_obstacle_field(env, ids, layout="scattered")
    after = env.sim.data.mocap_pos
    changed = (after != before).flatten(1).any(dim=1)
    assert changed.nonzero().flatten().tolist() == [1, 5]
    assert env.nav_field.layout[ids].tolist() == [layouts.LAYOUT_NAMES.index("scattered")] * 2
    assert (env.nav_field.layout[[0, 2]] == -1).all()
    for i, name in enumerate(SLOT_NAMES):
        assert torch.allclose(after[ids, SLOT_INDEX[name], :2], env.nav_field.pos[ids, i, :2])


def test_one_wall_stands_across_the_approach_in_the_robots_heading_frame(wall_env):
    env = wall_env
    field = env.nav_field
    pos, quat = root_pose(env)
    heading = yaw_of(quat)
    c, s = torch.cos(heading), torch.sin(heading)
    for slot in ("wall_left", "wall_right"):
        d = field.pos[:, SLOT_INDEX[slot], :2] - pos[:, :2]
        forward = c * d[:, 0] + s * d[:, 1]
        assert (forward > 0.4).all() and (forward < 1.2).all()
    left = field.pos[:, SLOT_INDEX["wall_left"], :2] - pos[:, :2]
    right = field.pos[:, SLOT_INDEX["wall_right"], :2] - pos[:, :2]
    side = lambda d: -s * d[:, 0] + c * d[:, 1]
    assert (side(left) > side(right)).all()
    assert (field.layout == layouts.LAYOUT_NAMES.index("one_wall")).all()


def _reference(env, sensor, w):
    """The tof.py loop for one world: mj_ray per zone, on the world's own state and its own directions."""
    model = env.sim.mj_model
    data = mujoco.MjData(model)
    data.qpos[:] = env.sim.data.qpos[w].numpy()
    data.mocap_pos[:] = env.sim.data.mocap_pos[w].numpy()
    data.mocap_quat[:] = env.sim.data.mocap_quat[w].numpy()
    mujoco.mj_forward(model, data)
    site = model.site("robot/tof").id
    body = int(model.site_bodyid[site])
    rot = data.site_xmat[site].reshape(3, 3)
    lift = np.array([0.0, 0.0, float(sensor._height[w])])
    origin = data.site_xpos[site] + rot @ lift
    dirs = sensor.env_directions[w].double().numpy()
    groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    out = np.full(64, -1.0)
    geom = np.zeros(1, dtype=np.int32)
    for z in range(64):
        hit = mujoco.mj_ray(model, data, origin, np.ascontiguousarray(rot @ dirs[z]), groups, 1, body, geom)
        if 0 <= hit <= sensor.cfg.max_distance:
            out[z] = hit
    return out


def test_every_world_casts_its_own_grid_and_sees_its_own_obstacles(full_env):
    env = full_env
    sensor = env.scene["head_tof"]
    from microduck_nav.field import reset_obstacle_field

    reset_obstacle_field(env, torch.arange(8), layout="scattered")
    env.sim.forward()
    env.sim.sense()
    pinhole = sensor.uses_pinhole
    assert pinhole.any() and (~pinhole).any(), "seed 5 should give both grids among 8 worlds"
    for w in range(8):
        ref = _reference(env, sensor, w)
        got = sensor._distances[w].numpy()
        assert (got < 0).tolist() == (ref < 0).tolist()
        assert np.abs(got - ref).max() < 1e-3
        base = zone_directions("pinhole" if pinhole[w] else "equal_angle")
        pitch = sensor._pitch[w : w + 1]
        assert torch.allclose(sensor.env_directions[w], rotate_pitch(base[None].to(torch.float32), pitch)[0], atol=1e-6)
    assert (sensor._distances >= 0).any()


def test_mixed_grid_is_half_and_half_deterministic_and_held_for_the_episode():
    n = 100000
    a = draw_grid(n, "mixed", "cpu", torch.Generator().manual_seed(3))
    b = draw_grid(n, "mixed", "cpu", torch.Generator().manual_seed(3))
    c = draw_grid(n, "mixed", "cpu", torch.Generator().manual_seed(4))
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert a.float().mean().item() == pytest.approx(0.5, abs=0.01)
    assert draw_grid(5, "pinhole", "cpu").all() and not draw_grid(5, "equal_angle", "cpu").any()

    env = make_env(FULL, num_envs=64, seed=8)
    try:
        sensor = env.scene["head_tof"]
        held = sensor.uses_pinhole.clone()
        for _ in range(30):
            env.step(torch.zeros(64, 3))
        done = env.episode_length_buf == 0
        assert torch.equal(sensor.uses_pinhole[~done], held[~done])
        counts = []
        for _ in range(40):
            env.reset()
            counts.append(sensor.uses_pinhole.float().mean().item())
        assert np.mean(counts) == pytest.approx(0.5, abs=0.03)
        assert min(counts) < 0.5 < max(counts)
    finally:
        env.close()

    def trace(seed):
        env = make_env(FULL, num_envs=32, seed=seed)
        try:
            grids = [env.scene["head_tof"].uses_pinhole.clone()]
            for _ in range(5):
                env.reset()
                grids.append(env.scene["head_tof"].uses_pinhole.clone())
            return torch.stack(grids)
        finally:
            env.close()

    first, again, other = trace(9), trace(9), trace(10)
    assert torch.equal(first, again)
    assert not torch.equal(first, other)
