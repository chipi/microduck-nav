"""R7a: the pool of our own challenge families, read from a lab-scenarios pool file and placed into the mocap slots at reset.

`tests/fixtures/pool_sample.jsonl` is `lab-scenarios pool --challenges 1,2,3,4,14,15 --seeds 1000-1002`: three items of each family
(robotics-lab's generator, so the real shapes: a corridor's 3.8 m walls, a chair's 3 cm legs and its floating seat).
"""

import json
import math

import pytest
import torch

from microduck_nav import pool as pools
from microduck_nav.field import FAR, place_boxes, root_pose, yaw_of
from microduck_nav.obstacles import POOL_SLOTS, SLOT_NAMES, pool_slot_names

from nav_env import FULL, make_env
from pool_env import OURS, SAMPLE, box, compose, item, ours_env, write_pool

FAMILIES = {"corridor", "doorway", "corner_in_doorway", "box_in_open", "thin_legs", "small_boxes"}


# ---- the loader -----------------------------------------------------------------------------------------------------------------------
def test_the_sample_pool_loads_with_every_family_counted():
    pool = pools.load_pool(SAMPLE, POOL_SLOTS)
    assert len(pool) == 18 and pool.stats.read == 18 and pool.stats.kept == 18
    assert set(pool.layouts) == FAMILIES and pool.stats.layouts == {f: 3 for f in FAMILIES}
    assert pool.obstacles.shape == (18, POOL_SLOTS, 7) and pool.goal.shape == (18, 2)
    assert pool.stats.skipped_too_many == 0 and pool.stats.held_out == {}
    assert set(pool.stats.kinds) >= {"wall", "box", "leg", "small"}
    assert (pool.count == (pool.obstacles[..., 3] > 0).sum(dim=1)).all(), "rows past an item's count are zero"


def test_a_held_out_family_is_not_kept_and_is_counted():
    pool = pools.load_pool(SAMPLE, POOL_SLOTS, ("doorway",))
    assert len(pool) == 15 and "doorway" not in pool.layouts and pool.stats.held_out == {"doorway": 3}
    both = pools.load_pool(SAMPLE, POOL_SLOTS, pools.parse_holdout("doorway, corner_in_doorway"))
    assert len(both) == 12 and both.stats.held_out == {"doorway": 3, "corner_in_doorway": 3}
    assert len(pools.load_pool(SAMPLE, POOL_SLOTS, pools.parse_holdout("none"))) == 18


def test_a_holdout_that_names_no_family_of_the_file_is_an_error():
    with pytest.raises(ValueError, match="dorway.*the pool's layouts are"):
        pools.load_pool(SAMPLE, POOL_SLOTS, ("dorway",))


def test_every_family_of_the_pool_fits_the_slots_so_none_is_skipped():
    """14 is the most obstacles any item of ours-v1 holds (its manifest's max_obstacles), small_boxes: 4 room walls and up to 10 boxes."""
    manifest = json.loads((SAMPLE.parent.parent.parent / "data" / "pools" / "ours-v1.manifest.json").read_text())
    assert manifest["max_obstacles"] == POOL_SLOTS == len(pool_slot_names())


def test_an_item_with_more_obstacles_than_slots_is_skipped_and_counted_never_cut(tmp_path):
    many = item([box(0.5 + 0.2 * i, 0.0, 0.05, 0.05) for i in range(5)], layout="crowd")
    few = item([box(0.5, 0.0, 0.05, 0.05)], layout="sparse")
    path = write_pool(tmp_path / "p.jsonl", [many, few, few])
    pool = pools.load_pool(path, 4)
    assert len(pool) == 2 and pool.stats.skipped_too_many == 1 and pool.stats.most_obstacles == 5 and pool.layouts == ("sparse",)
    assert "skipped for more obstacles than slots: 1" in pool.stats.line()
    assert len(pools.load_pool(path, 5)) == 3


def test_a_pool_with_nothing_left_to_train_on_is_an_error(tmp_path):
    path = write_pool(tmp_path / "p.jsonl", [item([box(0.5, 0.0, 0.1, 0.1)], layout="only")])
    with pytest.raises(ValueError, match="no item is left"):
        pools.load_pool(path, 4, ("only",))


@pytest.mark.parametrize(
    "line, message",
    [
        ("not json", ":2: not JSON"),
        (json.dumps({"format": "other/9"}), ":2: not a robotics-scenario/1 item"),
        (json.dumps({"format": pools.FORMAT, "layout": "x"}), ":2: missing"),
    ],
)
def test_a_bad_line_is_an_error_with_its_line_number(tmp_path, line, message):
    good = json.dumps(item([box(0.5, 0.0, 0.1, 0.1)]))
    path = tmp_path / "p.jsonl"
    path.write_text(good + "\n" + line + "\n")
    with pytest.raises(ValueError, match=message):
        pools.load_pool(path, 4)


def test_a_box_with_no_size_or_a_number_that_is_not_finite_is_an_error(tmp_path):
    flat = item([box(0.5, 0.0, 0.1, 0.1, z0=0.2, z1=0.2)])
    path = write_pool(tmp_path / "p.jsonl", [flat])
    with pytest.raises(ValueError, match="positive half sides and z1 > z0"):
        pools.load_pool(path, 4)
    bad = item([box(0.5, 0.0, 0.1, 0.1)])
    bad["goal"]["x"] = float("nan")
    path.write_text(json.dumps(bad) + "\n")
    with pytest.raises(ValueError, match="goal.x"):
        pools.load_pool(path, 4)


def test_a_gzipped_pool_reads_the_same(tmp_path):
    import gzip

    path = tmp_path / "p.jsonl.gz"
    with gzip.open(path, "wt") as f:
        f.write(SAMPLE.read_text())
    a, b = pools.load_pool(SAMPLE, POOL_SLOTS), pools.load_pool(path, POOL_SLOTS)
    assert torch.equal(a.obstacles, b.obstacles) and torch.equal(a.goal, b.goal)


def test_the_default_pool_is_the_env_var_then_the_checkout(monkeypatch, tmp_path):
    monkeypatch.delenv(pools.POOL_ENV, raising=False)
    assert pools.pool_path().name == "ours-v1.jsonl" and pools.pool_path().parent.name == "pools"
    monkeypatch.setenv(pools.POOL_ENV, str(tmp_path / "x.jsonl"))
    assert pools.pool_path() == tmp_path / "x.jsonl" and pools.pool_path("/a/b.jsonl").as_posix() == "/a/b.jsonl"


# ---- the frame: start and goal come from the item, in the robot's reset frame ---------------------------------------------------------------
def test_the_item_is_moved_into_the_trunks_frame_at_the_start(tmp_path):
    """Hand-computed: a duck starting at (1, 2) turned 90 degrees stands 0.05 m ahead of that, at (1, 2.05); a box at world (1, 3.05) is
    1.0 m straight ahead of it, and a goal at (0, 2.05) is 1.0 m to its left."""
    doc = item([], goal=(0.0, 0.0), start=(1.0, 2.0, math.pi / 2))
    doc["obstacles"] = [{"kind": "box", "x": 1.0, "y": 3.05, "yaw": math.pi / 2 + 0.3, "hx": 0.1, "hy": 0.2, "z0": 0.0, "z1": 0.3}]
    doc["goal"] = {"x": 0.0, "y": 2.05}
    pool = pools.load_pool(write_pool(tmp_path / "p.jsonl", [doc]), 2)
    x, y, yaw, hx, hy, z0, z1 = pool.obstacles[0, 0].tolist()
    assert (x, y, hx, hy, z0, z1) == pytest.approx((1.0, 0.0, 0.1, 0.2, 0.0, 0.3), abs=1e-6) and yaw == pytest.approx(0.3, abs=1e-6)
    assert pool.goal[0].tolist() == pytest.approx([0.0, 1.0], abs=1e-6)
    assert pool.obstacles[0, 1:].abs().sum() == 0 and pool.count.tolist() == [1]


def test_the_pool_places_boxes_by_the_robots_pose_whatever_its_heading():
    local = torch.tensor([[[1.0, 0.5, 0.2, 0.3, 0.1, 0.0, 0.4], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    placed = place_boxes(local, torch.tensor([1]), torch.tensor([[3.0, 4.0]]), torch.tensor([math.pi / 2]), torch.tensor([[10.0, 20.0, 0.0]]))
    assert placed["xy"][0, 0].tolist() == pytest.approx([2.5, 5.0], abs=1e-6)  # 1 ahead is +y, 0.5 left is -x
    assert placed["yaw"][0, 0].item() == pytest.approx(math.pi / 2 + 0.2, abs=1e-6)
    assert placed["zc"][0, 0].item() == pytest.approx(0.2) and placed["hz"][0, 0].item() == pytest.approx(0.2)
    assert placed["half"][0, 0].tolist() == pytest.approx([0.3, 0.1]) and placed["real"][0].tolist() == [True, False]
    parked = placed["xy"][0, 1]
    assert (parked - torch.tensor([10.0, 20.0])).norm() > 30, "an unused slot waits far off, outside any ray's reach"


# ---- the scene in the env: the slots are resized and placed per environment ---------------------------------------------------------------
@pytest.fixture(scope="module")
def two_boxes(walker_path, tmp_path_factory):
    """A 1.2 m wall 0.6 m ahead and a 3 cm leg off to the left, for a duck that started at (0.1, -0.05) turned 0.2 rad."""
    doc = item(
        [box(0.6, 0.0, 0.02, 0.6, z1=0.5), box(0.4, 0.5, 0.015, 0.015, z1=0.42, kind="leg")], goal=(2.0, 0.5), start=(0.1, -0.05, 0.2)
    )
    path = write_pool(tmp_path_factory.mktemp("pool") / "p.jsonl", [doc])
    env = ours_env(path, num_envs=4, seed=3)
    yield env, path
    env.close()


def test_the_ours_scene_has_14_resizable_slots_and_not_nvidias_eight(two_boxes):
    env, _ = two_boxes
    assert env.nav_field.pool and env.nav_field.names == pool_slot_names() and not set(SLOT_NAMES) & set(env.scene.entities)
    assert env.sim.mj_model.nmocap == POOL_SLOTS
    assert env.sim.model.geom_size.shape[0] == 4, "geom_size is per world, so each environment can size its own boxes"


def test_each_environments_slots_hold_the_items_boxes_where_the_robot_is(two_boxes):
    env, _ = two_boxes
    field = env.nav_field
    pos, quat = root_pose(env)
    heading = yaw_of(quat)
    for e in range(4):
        robot = (pos[e, 0].item(), pos[e, 1].item(), heading[e].item())
        wall = compose(robot, (0.6, 0.0, 0.0))
        leg = compose(robot, (0.4, 0.5, 0.0))
        assert field.pos[e, 0, :2].tolist() == pytest.approx(list(wall[:2]), abs=1e-4)
        assert field.pos[e, 1, :2].tolist() == pytest.approx(list(leg[:2]), abs=1e-4)
        assert math.remainder(field.yaw[e, 0].item() - heading[e].item(), math.tau) == pytest.approx(0.0, abs=1e-5)
        assert field.half_sizes[e, 0].tolist() == pytest.approx([0.02, 0.6]) and field.half_sizes[e, 1].tolist() == pytest.approx([0.015, 0.015])
        assert (field.half_sizes[e, 2:] == 0).all() and (field.pos[e, 2:, :2] - env.scene.env_origins[e, :2] == FAR).all(), "unused slots leave the rewards' geometry, far from their own environment"
        assert field.goal_w[e].tolist() == pytest.approx(list(compose(robot, (2.0, 0.5, 0.0))[:2]), abs=1e-4)
        assert env.command_manager.get_term("twist").destination_w[e, :2].tolist() == pytest.approx(field.goal_w[e].tolist())


def test_the_mocap_pose_and_the_geom_size_of_each_world_are_the_boxes(two_boxes):
    env, _ = two_boxes
    field = env.nav_field
    geom = field.geom_ids.long()
    size = env.sim.model.geom_size[:, geom]
    mocap = env.sim.data.mocap_pos
    for e in range(4):
        assert size[e, 0].tolist() == pytest.approx([0.02, 0.6, 0.25]), "a half side per axis, half the height"
        assert size[e, 1].tolist() == pytest.approx([0.015, 0.015, 0.21])
        assert mocap[e, 0, 2].item() == pytest.approx(0.25) and mocap[e, 1, 2].item() == pytest.approx(0.21), "centred on its height"
        assert mocap[e, 0, :2].tolist() == pytest.approx(field.pos[e, 0, :2].tolist(), abs=1e-5)
        assert (size[e, 2:] == pools_box()).all(), "an unused slot is a small box"
    rbound = env.sim.model.geom_rbound[:, geom]
    assert rbound[0, 0].item() == pytest.approx(math.sqrt(0.02**2 + 0.6**2 + 0.25**2), abs=1e-5), "the broadphase bound follows the size"


def pools_box():
    from microduck_nav.obstacles import POOL_BOX

    return POOL_BOX


def test_the_sensor_sees_the_resized_box_where_the_item_put_it(walker_path, tmp_path):
    """The rays of each world read that world's own geom_size: a wall 0.3 m further reads 0.3 m deeper."""
    reads = []
    for ahead in (0.6, 0.9):
        path = write_pool(tmp_path / f"w{ahead}.jsonl", [item([box(ahead, 0.0, 0.02, 0.8, z1=0.5)])])
        env = ours_env(path, num_envs=2, seed=4, guards=False)
        try:
            for _ in range(15):
                env.step(torch.zeros(2, 3))
            frame = env.scene["head_tof"].delivered_frame
            reads.append(frame[:, [3 * 8 + 3, 3 * 8 + 4, 4 * 8 + 3, 4 * 8 + 4]].mean(dim=1))
        finally:
            env.close()
    assert (reads[0] > 0.3).all() and (reads[0] < 0.7).all(), reads[0]
    assert (reads[1] - reads[0]).tolist() == pytest.approx([0.3, 0.3], abs=0.05)


def test_the_resized_box_is_solid_a_robot_walking_into_it_ends_the_episode(walker_path, tmp_path):
    path = write_pool(tmp_path / "w.jsonl", [item([box(0.45, 0.0, 0.02, 0.8, z1=0.5)], goal=(2.0, 0.0))])
    env = ours_env(path, num_envs=2, seed=5, guards=False)
    try:
        hit = torch.zeros(2, dtype=torch.bool)
        for _ in range(200):
            _, _, terminated, _, _ = env.step(torch.zeros(2, 3))
            hit |= terminated
            if hit.all():
                break
        assert hit.all(), "the duck was asked to walk 2 m through a wall 0.45 m ahead"
    finally:
        env.close()


# ---- the task: the pool, a family held out, deploy-matched ---------------------------------------------------------------------------------
def test_the_task_is_deploy_matched_with_the_pool_a_held_out_family_and_the_guards(walker_path):
    env = make_env(OURS, num_envs=2, seed=6)
    try:
        nav = env.action_manager.get_term("nav")
        assert nav.decimation == 5 and nav.cfg.limits == (0.35, 0.0, 0.75) and nav.guards is not None
        params = env.cfg.events["obstacle_field"].params
        assert params["holdout_family"] == "doorway" and params["slots"] == POOL_SLOTS and params["pool"].endswith("ours-v1.jsonl")
        assert env.observation_manager.group_obs_dim["policy"] == (53,) and env.observation_manager.group_obs_dim["tof"] == (1290,)
        assert "guard_intervention" in env.reward_manager.active_terms
    finally:
        env.close()


def test_the_default_holdout_keeps_the_doorway_family_out_of_every_reset(walker_path, tmp_path):
    """The sample pool has three items of each family; 40 resets of 16 environments draw all the others and never a doorway."""
    layouts = pools.load_pool(SAMPLE, POOL_SLOTS).layouts
    env = ours_env(SAMPLE, num_envs=16, seed=7, holdout="doorway", impairments=True)
    try:
        drawn = set()
        for _ in range(8):
            env.reset()
            drawn |= {layouts_of(env)[i] for i in env.nav_field.layout.tolist()}
        assert "doorway" not in drawn and len(drawn) >= 4, drawn
    finally:
        env.close()


def layouts_of(env):
    from microduck_nav.pool import get_pool

    params = env.cfg.events["obstacle_field"].params
    return get_pool(params["pool"], params["slots"], params["holdout_family"], env.device).layouts


def test_a_holdout_of_none_trains_on_every_family(walker_path):
    env = ours_env(SAMPLE, num_envs=16, seed=8, holdout="none", impairments=True)
    try:
        drawn = set()
        for _ in range(12):
            env.reset()
            drawn |= {layouts_of(env)[i] for i in env.nav_field.layout.tolist()}
        assert "doorway" in drawn
    finally:
        env.close()


def test_the_full_task_is_unchanged_by_the_pool(walker_path):
    env = make_env(FULL, num_envs=2, seed=9)
    try:
        assert not env.nav_field.pool and env.nav_field.names == SLOT_NAMES and env.nav_field.half_sizes.shape == (8, 2)
        assert env.action_manager.get_term("nav").guards is None and "guard_intervention" not in env.reward_manager.active_terms
    finally:
        env.close()
