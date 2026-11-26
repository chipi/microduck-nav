"""R7b: the navigator's reflex and cliff stop between the policy's command and the walker (guards.py).

Three layers. (1) The rule on synthetic frames. (2) Parity: `Reference` below restates the navigator's rule as robotics-lab's
robots/duck code reads (run.py `_go_to` and plugins/tof8x8.py at commit 224e7d5 of its mission-control branch, scalar Python with the
profile's constants), and the batched torch layer must give its numbers and its decisions on random worlds, poses and frame sequences.
(3) The task: a guard zeroes the command before the walker, the post-guard command is what is observed, an intervention costs.
"""

import json
import math
import os
import random
from pathlib import Path

import pytest
import torch

from microduck_nav import guards
from microduck_nav.guards import GuardCfg, GuardLayer, drop_terms, reflex_terms

from pool_env import box as pool_box, item, ours_env, write_pool
from scene_frames import HEIGHT, SENSOR_AHEAD, Sensor, render, rotation

CFG = GuardCfg()
TRUNK = torch.tensor([[-SENSOR_AHEAD, 0.0]])  # the trunk, with the sensor 8.5 cm ahead of it at the world's origin
YAW0 = torch.zeros(1)
FLOOR = torch.zeros(1)


def terms(frame, trunk=TRUNK, yaw=YAW0, unknown=False):
    args = (frame["ranges"][None], frame["dir_x"][None], frame["origin"][None], frame["rays"][None], FLOOR, trunk, yaw)
    block, near = reflex_terms(*args, CFG)
    has, nearest = drop_terms(*args, unknown, CFG)
    return block.item(), near.item(), has.item(), nearest.item()


def wall(ahead_of_sensor, y0=-0.6, y1=0.6, z1=0.5, z0=0.0):
    return (ahead_of_sensor, ahead_of_sensor + 0.02, y0, y1, z0, z1)


# ---- the constants: the profile's -------------------------------------------------------------------------------------------------------
def test_the_constants_are_robotics_labs_profile_values():
    assert (CFG.reflex_ahead_m, CFG.emergency_mm, CFG.reflex_margin_m, CFG.radius_m) == (0.30, 150, 0.04, 0.083)
    assert CFG.reflex_half_w_m == pytest.approx(0.123)  # tof8x8.py:147, footprint.radius_m + reflex_margin_m
    assert (CFG.floor_margin_m, CFG.floor_margin_per_m, CFG.extra_m, CFG.look_m, CFG.min_zones) == (0.08, 0.03, 0.06, 0.7, 2)
    assert (CFG.overhead_m, CFG.corridor_m, CFG.cliff_stop_m, CFG.drop_frames) == (0.30, 0.25, 0.60, 3)
    assert guards.CENTRAL_COLUMNS == (2, 3, 4, 5) and len(guards.CENTRAL_ZONES) == 32


def profile():
    for candidate in (os.environ.get("DUCK_PROFILE"), "/work/robotics-lab/mission-control/robots/duck/profile.json", "/work/robotics-lab/main/robots/duck/profile.json"):
        if candidate and Path(candidate).is_file():
            return json.loads(Path(candidate).read_text())
    pytest.skip("robotics-lab's robots/duck/profile.json is not on this machine (set DUCK_PROFILE)")


def test_the_constants_equal_the_profile_where_it_is_found():
    p = profile()
    tof, foot, eb = p["sensors"]["tof8x8"], p["footprint"], p["error_budget"]
    assert (tof["reflex_ahead_m"], tof["emergency_mm"], tof["reflex_margin_m"], foot["radius_m"]) == (
        CFG.reflex_ahead_m, CFG.emergency_mm, CFG.reflex_margin_m, CFG.radius_m)
    assert (tof["floor_margin_m"], tof["floor_margin_per_m"], tof["extra_m"], tof["look_m"], tof["min_zones"]) == (
        CFG.floor_margin_m, CFG.floor_margin_per_m, CFG.extra_m, CFG.look_m, CFG.min_zones)
    assert (foot["overhead_clear_m"], eb["corridor_m"], eb["cliff_stop_m"]) == (CFG.overhead_m, CFG.corridor_m, CFG.cliff_stop_m)


# ---- the reflex on frames ------------------------------------------------------------------------------------------------------------------
def test_an_open_floor_blocks_nothing_and_shows_no_drop():
    block, near, has, nearest = terms(render())
    assert block == math.inf and near > 150 and not has


def test_a_wall_nearer_than_0_30_m_of_the_trunk_is_a_block_and_beyond_it_is_not():
    """The sensor is 8.5 cm ahead of the trunk, so the line is a wall face 0.215 m from the sensor."""
    near_wall, far_wall = terms(render([wall(0.20)])), terms(render([wall(0.23)]))
    assert near_wall[0] == pytest.approx(0.20 + SENSOR_AHEAD, abs=0.01) and near_wall[0] < CFG.reflex_ahead_m
    assert far_wall[0] == pytest.approx(0.23 + SENSOR_AHEAD, abs=0.01) and far_wall[0] > CFG.reflex_ahead_m


def test_an_obstacle_beside_the_corridor_does_not_block_and_one_inside_it_does():
    """The corridor is 0.123 m either side of the heading: a wall on the left from y = 0.2 is passed with room to spare (phase 1 run 16)."""
    assert terms(render([wall(0.20, y0=0.20, y1=0.8)]))[0] == math.inf
    assert terms(render([wall(0.20, y0=0.05, y1=0.8)]))[0] < CFG.reflex_ahead_m


def test_an_overhead_obstacle_does_not_block():
    """A chair's seat 0.42 m up is over the duck (overhead_clear_m 0.30)."""
    assert terms(render([wall(0.20, z0=0.40, z1=0.45)]))[0] == math.inf


def test_a_return_nearer_than_150_mm_in_the_central_columns_is_the_emergency_and_a_side_column_is_not():
    frame = render()
    quiet = terms(frame)[1]
    assert quiet > 300, "the floor's own nearest return, about 0.33 m"
    for zone, central in ((3 * 8 + 3, True), (3 * 8 + 0, False), (4 * 8 + 7, False), (7 * 8 + 5, True), (0 * 8 + 2, True)):
        ranges = frame["ranges"].clone()
        ranges[zone] = 0.13
        near = terms({**frame, "ranges": ranges})[1]
        assert (near == pytest.approx(130.0)) if central else (near == pytest.approx(quiet)), zone


def test_the_reflex_stops_vx_and_the_whole_command_and_only_a_walking_command():
    layer = GuardLayer(1, "cpu", impairments=False)
    sensor = Sensor(render([wall(0.18)]))
    for command, stopped in (([0.25, 0.0, 0.4], True), ([0.0, 0.0, 0.4], False), ([-0.1, 0.0, 0.2], False), ([1e-9, 0.0, 0.3], True)):
        layer.reset(slice(None))
        given = torch.tensor([command])
        out, reflex, cliff = layer(given, sensor, TRUNK, YAW0, FLOOR)
        assert bool(reflex) == stopped and not cliff
        assert torch.equal(out, torch.zeros(1, 3) if stopped else given), command


def test_a_clear_field_changes_nothing_over_a_walk():
    layer = GuardLayer(1, "cpu", impairments=True)
    sensor = Sensor(render([wall(1.5)]), 1.0)
    command = torch.tensor([[0.3, 0.0, 0.2]])
    for t in range(10):
        sensor.set(sensor.frame, 1.0 + t / 15)
        out, reflex, cliff = layer(command, sensor, TRUNK, YAW0, FLOOR)
        assert torch.equal(out, command) and not reflex and not cliff


def test_no_frame_yet_is_no_stop():
    layer = GuardLayer(1, "cpu", impairments=False)
    sensor = Sensor(render([wall(0.1)]))
    sensor.available = torch.tensor([False])
    out, reflex, cliff = layer(torch.tensor([[0.3, 0.0, 0.0]]), sensor, TRUNK, YAW0, FLOOR)
    assert not reflex and not cliff and out[0, 0] == pytest.approx(0.3)


# ---- the cliff on frames -------------------------------------------------------------------------------------------------------------------
def test_a_hole_ahead_is_drop_points_and_a_flat_floor_is_not():
    hole = terms(render(floor_ends_at=0.35))
    assert hole[2] and hole[3] < CFG.cliff_stop_m, "the lower rows reach the floor past 0.35 m and read nothing"
    assert not terms(render())[2]
    assert not terms(render(floor_ends_at=0.9))[2], "a hole beyond look_m (0.7 m) is not looked at"


def test_a_ledge_is_drop_points_by_returns_past_the_floor_line():
    ledge = terms(render(floor_ends_at=0.35, floor_below=0.3))
    assert ledge[2] and ledge[3] < CFG.cliff_stop_m


def test_a_hole_is_a_drop_only_with_the_impairments_off_because_a_blanked_zone_is_no_evidence():
    """tof8x8.py:347: with the impairment on, the navigator skips a zone it blanked; the env cannot tell a blank from a miss."""
    hole = render(floor_ends_at=0.35)
    assert terms(hole, unknown=False)[2] and not terms(hole, unknown=True)[2]
    ledge = render(floor_ends_at=0.35, floor_below=0.3)
    assert terms(ledge, unknown=True)[2], "a return past the floor line is a return: it counts either way"


def test_a_drop_to_one_side_of_the_corridor_is_not_on_the_way():
    assert not terms(render(floor_ends_at=0.35), trunk=torch.tensor([[-SENSOR_AHEAD, 0.9]]))[2]


def test_the_cliff_needs_the_drop_in_three_new_frames_and_a_repeat_of_a_frame_is_not_a_new_one():
    layer = GuardLayer(1, "cpu", impairments=False)
    sensor = Sensor(render(floor_ends_at=0.35), 1.0)
    command = torch.tensor([[0.3, 0.0, 0.0]])
    fired = []
    for t in (1.0, 1.0, 1.07, 1.07, 1.07, 1.13, 1.2):  # ticks; the frame advances on the 1st, 3rd, 6th and 7th
        sensor.set(sensor.frame, t)
        out, reflex, cliff = layer(command, sensor, TRUNK, YAW0, FLOOR)
        fired.append(bool(cliff))
        assert bool(cliff) == (out[0, 0] == 0)
    assert fired == [False, False, False, False, False, True, True], "frames 1 and 2 count, the third is the stop (run.py:1248-1254)"
    sensor.set(render(), 1.27)  # one clean frame resets the streak
    layer(command, sensor, TRUNK, YAW0, FLOOR)
    sensor.set(render(floor_ends_at=0.35), 1.33)
    assert not layer(command, sensor, TRUNK, YAW0, FLOOR)[2]


def test_the_stop_distance_is_the_nearest_drops_horizontal_distance_from_the_sensor():
    """A floor that ends at 0.4 m hides the 14-degree row's floor at 0.479 m: those zones are the drop, 0.479 m away. Stopped for at
    cliff_stop_m 0.60 and not at 0.45."""
    outcomes = []
    for stop in (0.60, 0.45):
        layer = GuardLayer(1, "cpu", impairments=False, cfg=GuardCfg(cliff_stop_m=stop))
        sensor = Sensor(render(floor_ends_at=0.4), 1.0)
        for t in range(4):
            sensor.set(sensor.frame, 1.0 + 0.07 * t)
            _, _, cliff = layer(torch.tensor([[0.3, 0.0, 0.0]]), sensor, TRUNK, YAW0, FLOOR)
        outcomes.append(bool(cliff))
    assert terms(render(floor_ends_at=0.4))[3] == pytest.approx(0.479, abs=0.01) and outcomes == [True, False]


# ---- parity with the navigator's rule ---------------------------------------------------------------------------------------------------------
class Reference:
    """robotics-lab's rule as its code reads, one zone at a time. Inputs per frame: the zone readings in mm (0: none), the beams in the
    world (what `rotate(q, beam)` gives, q the sensor's orientation), the beams' x in the sensor frame (along_beam for an axial range),
    the sensor's height above the floor, and where it is relative to the trunk, in the trunk's heading frame (sensor_offset)."""

    REFLEX_AHEAD_M, EMERGENCY_MM, HALF_W = 0.30, 150, 0.083 + 0.04
    FLOOR_MARGIN, FLOOR_PER_M, EXTRA, OVERHEAD = 0.08, 0.03, 0.06, 0.30
    LOOK, MIN_ZONES, CORRIDOR, CLIFF_STOP, DROP_FRAMES = 0.7, 2, 0.25, 0.60, 3

    def __init__(self, unknown_is_blank: bool):
        self.unknown_is_blank = unknown_is_blank
        self.drop_frame, self.streak, self.block = None, 0, None

    def depth_points(self, mm, world, beam_x, height, offset, yaw):
        c, s = math.cos(-yaw), math.sin(-yaw)
        out = []
        for i in range(64):
            if mm[i] <= 0:
                continue
            r = mm[i] / 1000.0 / beam_x[i]
            hx, hy = r * world[i][0], r * world[i][1]
            h = math.hypot(hx, hy)
            zz = height + r * world[i][2]
            kind = ("below" if zz < -self.EXTRA else "floor" if zz < self.FLOOR_MARGIN + self.FLOOR_PER_M * h else
                    "overhead" if zz > self.OVERHEAD else "obstacle")
            out.append((offset[0] + c * hx - s * hy, offset[1] + s * hx + c * hy, zz, kind))
        return out

    def blocking_ahead(self, points):
        ahead = [p[0] for p in points if p[3] == "obstacle" and p[0] > 0 and abs(p[1]) <= self.HALF_W]
        return min(ahead) if ahead else None

    def drop_points(self, mm, world, beam_x, height, offset, yaw):
        if height <= 0.02:
            return []
        c, s = math.cos(-yaw), math.sin(-yaw)
        drops = []
        for i in range(64):
            d = world[i]
            if d[2] > -0.15:
                continue
            t_floor = -height / d[2]
            h_floor = t_floor * math.hypot(d[0], d[1])
            if h_floor > self.LOOK:
                continue
            valid = mm[i] > 0
            if not valid and self.unknown_is_blank:
                continue
            if valid:
                r = mm[i] / 1000.0 / beam_x[i]
                if r * -d[2] < height + self.EXTRA:
                    continue
            wx, wy = t_floor * d[0], t_floor * d[1]
            ahead, left = offset[0] + c * wx - s * wy, offset[1] + s * wx + c * wy
            if abs(left) > self.CORRIDOR:
                continue
            drops.append((ahead, left, h_floor))
        return drops if len(drops) >= self.MIN_ZONES else []

    def tick(self, frame_id, mm, world, beam_x, height, offset, yaw, vx):
        """run.py 1244-1254, 1263, 1337-1338 for one tick. Returns (cliff, reflex, block, nearest mm)."""
        pts = self.drop_points(mm, world, beam_x, height, offset, yaw)
        if frame_id != self.drop_frame:  # frames, not ticks
            self.drop_frame = frame_id
            self.block = self.blocking_ahead(self.depth_points(mm, world, beam_x, height, offset, yaw))
            self.streak = self.streak + 1 if pts else 0
        if self.streak < self.DROP_FRAMES:
            pts = []
        cliff = vx > 0 and bool(pts) and min(p[2] for p in pts) < self.CLIFF_STOP
        near = [mm[r * 8 + col] for r in range(8) for col in range(2, 6) if mm[r * 8 + col] > 0]
        near = min(near) if near else None
        blocked_ahead = self.block is not None and self.block < self.REFLEX_AHEAD_M
        reflex = (not cliff) and vx > 0 and (blocked_ahead or (near is not None and near < self.EMERGENCY_MM))
        return cliff, reflex, self.block, near


def random_world(rng: random.Random):
    boxes = []
    for _ in range(rng.randint(0, 4)):
        x0, y0, z0 = rng.uniform(0.05, 0.9), rng.uniform(-0.6, 0.4), rng.choice((0.0, 0.0, rng.uniform(0.0, 0.45)))
        boxes.append((x0, x0 + rng.uniform(0.02, 0.3), y0, y0 + rng.uniform(0.02, 0.6), z0, z0 + rng.uniform(0.03, 0.5)))
    floor = {"floor_ends_at": rng.uniform(0.2, 0.6) if rng.random() < 0.45 else None}
    if floor["floor_ends_at"] is not None and rng.random() < 0.6:
        floor["floor_below"] = rng.uniform(0.1, 0.4)
    return boxes, floor


def random_pose(rng: random.Random):
    """A heading, a sensor orientation (its pitch and roll) and a height, as a walking duck has them."""
    yaw, pitch, roll = rng.uniform(-math.pi, math.pi), rng.uniform(-0.2, 0.2), rng.uniform(-0.08, 0.08)
    return yaw, rotation(yaw, pitch, roll), rng.uniform(0.09, 0.2)


def world_to_args(frame, yaw, origin_xy):
    """The trunk 8.5 cm behind the sensor along the heading, 2 cm to its right: its xy and the sensor's offset in its frame."""
    c, s = math.cos(yaw), math.sin(yaw)
    offset = (SENSOR_AHEAD, 0.02)
    trunk = (origin_xy[0] - (c * offset[0] - s * offset[1]), origin_xy[1] - (s * offset[0] + c * offset[1]))
    return trunk, offset


@pytest.mark.parametrize("unknown", [False, True])
def test_the_batched_rule_gives_the_references_numbers_on_random_worlds_and_poses(unknown):
    rng = random.Random(7)
    seen = {"block": 0, "near": 0, "drop": 0, "none": 0}
    for _ in range(300):
        boxes, floor = random_world(rng)
        yaw, rot, height = random_pose(rng)
        origin = (rng.uniform(-1, 1), rng.uniform(-1, 1), height)
        shifted = [(b[0] + origin[0], b[1] + origin[0], b[2] + origin[1], b[3] + origin[1], b[4], b[5]) for b in boxes]
        if floor["floor_ends_at"] is not None:
            floor = {**floor, "floor_ends_at": floor["floor_ends_at"] + origin[0]}
        frame = render(shifted, origin=origin, rot=rot, **floor)
        trunk, offset = world_to_args(frame, yaw, origin[:2])
        ref = Reference(unknown)
        mm = [round(float(v) * 1000, 3) for v in frame["ranges"]]
        world, beam_x = frame["rays"].tolist(), frame["dir_x"].tolist()
        # the world's floor is at 0 and the box heights are the world's: the sensor is `height` above it
        ref_block = ref.blocking_ahead(ref.depth_points(mm, world, beam_x, height, offset, yaw))
        ref_drops = ref.drop_points(mm, world, beam_x, height, offset, yaw)
        ref_near = min([mm[r * 8 + c] for r in range(8) for c in range(2, 6) if mm[r * 8 + c] > 0], default=None)
        args = (frame["ranges"][None], frame["dir_x"][None], frame["origin"][None], frame["rays"][None], FLOOR,
                torch.tensor([trunk], dtype=torch.float32), torch.tensor([yaw], dtype=torch.float32))
        block, near = reflex_terms(*args, CFG)
        has, nearest = drop_terms(*args, unknown, CFG)
        assert (block.item() == math.inf) == (ref_block is None), (boxes, floor)
        if ref_block is not None:
            assert block.item() == pytest.approx(ref_block, abs=2e-4)
            seen["block"] += 1
        assert (near.item() == math.inf) == (ref_near is None)
        if ref_near is not None:
            assert near.item() == pytest.approx(ref_near, abs=0.05)
            seen["near"] += 1
        assert bool(has.item()) == bool(ref_drops), (boxes, floor)
        if ref_drops:
            assert nearest.item() == pytest.approx(min(d[2] for d in ref_drops), abs=2e-4)
            seen["drop"] += 1
        seen["none"] += not ref_drops and ref_block is None
    assert min(seen.values()) >= 15, f"the random worlds must exercise every branch (unknown={unknown}): {seen}"


def test_the_layers_decisions_equal_the_references_over_random_frame_sequences():
    rng = random.Random(11)
    decided = {"cliff": 0, "reflex": 0, "none": 0}
    for sequence in range(60):
        unknown = sequence % 2 == 0
        boxes, floor = random_world(rng)
        yaw, rot, height = random_pose(rng)
        frame = render(boxes, origin=(0.0, 0.0, height), rot=rot, **floor)
        trunk, offset = world_to_args(frame, yaw, (0.0, 0.0))
        layer, ref = GuardLayer(1, "cpu", impairments=unknown), Reference(unknown)
        sensor = Sensor(frame, 1.0)
        mm = [float(v) * 1000 for v in frame["ranges"]]
        world, beam_x = frame["rays"].tolist(), frame["dir_x"].tolist()
        clock = 1.0
        for tick in range(9):
            if rng.random() < 0.6:  # a new frame, or the same one again
                clock += 1 / 15
            sensor.set(frame, clock)
            vx = rng.choice((0.0, 0.3, 0.3, 0.25, -0.1))
            out, reflex, cliff = layer(torch.tensor([[vx, 0.0, 0.2]]), sensor, torch.tensor([trunk]), torch.tensor([yaw]), FLOOR)
            want_cliff, want_reflex, _, _ = ref.tick(round(clock, 6), mm, world, beam_x, height, offset, yaw, vx)
            assert (bool(cliff), bool(reflex)) == (want_cliff, want_reflex), (sequence, tick, vx)
            assert out[0, 0].item() == (0.0 if want_cliff or want_reflex else pytest.approx(vx))
            decided["cliff" if want_cliff else "reflex" if want_reflex else "none"] += 1
    assert min(decided.values()) >= 10, decided


# ---- the task: the guard between the command and the walker ------------------------------------------------------------------------------------
TICK = 5
ROOM_AHEAD = item([pool_box(0.25 + 0.02, 0.0, 0.02, 0.8, z1=0.5)], goal=(2.0, 0.0))  # a wall face 0.25 m ahead of the trunk
OPEN = item([pool_box(2.5, 0.0, 0.02, 0.8, z1=0.5)], goal=(2.0, 0.0))  # a wall 2.5 m ahead: nothing near


@pytest.fixture(scope="module")
def pools(tmp_path_factory):
    d = tmp_path_factory.mktemp("guards")
    return write_pool(d / "near.jsonl", [ROOM_AHEAD]), write_pool(d / "open.jsonl", [OPEN])


def run_ticks(env, ticks, action=0.0):
    """Steps `ticks` policy ticks, keeping what the nav term and the reward manager held at the first walker step of each."""
    nav = env.action_manager.get_term("nav")
    out = []
    for _ in range(ticks):
        for k in range(TICK):
            obs, *_ = env.step(torch.full((env.num_envs, 3), action))
            if k == 0:
                step_rewards = env.reward_manager._step_reward.clone()
                held = nav.velocity_commands.clone()
                event = nav.guard_event.clone()
                reflex = nav.guard_reflex.clone()
            elif nav.guards is not None:
                assert not nav.guard_event.any(), "an intervention is the tick's, not each walker step's"
        out.append({"command": held, "event": event, "reflex": reflex, "rewards": step_rewards, "obs": obs["policy"].clone(),
                    "walker": nav._command.clone()})
    return out


@pytest.fixture(scope="module")
def near_wall(pools, walker_path):
    """The same world, the same seed, with the guards on and off."""
    on, off = ours_env(pools[0], guards=True, seed=21), ours_env(pools[0], guards=False, seed=21)
    yield run_ticks(on, 8), run_ticks(off, 8), on
    on.close()
    off.close()


def test_the_guard_zeroes_the_command_the_walker_gets_on_a_near_return(near_wall):
    on, off, _ = near_wall
    assert (off[-1]["command"][:, 0] > 0.1).all(), "the control: with the guards off the duck is sent forward at the wall"
    assert off[-1]["command"][:, 2].abs().max() < 1.0
    for tick in on[2:]:
        assert tick["event"].all() and tick["reflex"].all()
        assert (tick["walker"] == 0).all() and (tick["command"] == 0).all(), "vx, vy and vyaw: the navigator's Twist()"


def on_index(env):
    return env.reward_manager.active_terms.index("guard_intervention")


def test_the_first_tick_has_no_frame_so_nothing_stops_it(near_wall):
    on, _, on_env = near_wall
    assert not on[0]["event"].any() and (on[0]["rewards"][:, on_index(on_env)] == 0).all()


def test_the_post_guard_command_is_the_one_observed_as_the_applied_command(near_wall):
    on, off, _ = near_wall
    tick = on[4]
    assert tick["event"].all()
    applied = tick["obs"][:, 50:53]
    assert torch.equal(applied, tick["command"]) and (applied == 0).all(), "the applied-command slot holds what the guard sent"
    assert (off[4]["obs"][:, 50:53][:, 0] > 0.1).all(), "and without the guards it holds the policy's command"


def test_the_slew_starts_from_the_post_guard_command_not_the_policys(near_wall):
    """learned.py `_command` sets `residual.previous = app`, what the navigator last sent: a stopped duck speeds up from 0 by one slew
    step (0.1 m/s a tick), not from the speed it was asked for."""
    on, off, env = near_wall
    assert (env.action_manager.get_term("nav").residual.previous == 0).all()
    assert (on[-1]["command"] == 0).all() and (off[-1]["command"][:, 0] > 0.1).all()


def test_an_intervention_costs_two_per_tick_and_only_on_the_tick(near_wall):
    on, off, env = near_wall
    index = env.reward_manager.active_terms.index("guard_intervention")
    assert env.cfg.rewards["guard_intervention"].weight == -2.0
    for tick in on[2:]:
        assert (tick["rewards"][:, index] * env.step_dt).tolist() == pytest.approx([-2.0] * 4), "a rate of 100 a second for one 0.02 s step"
    assert (on[0]["rewards"][:, index] == 0).all(), "no frame, no guard, no cost"
    total = 0.0
    for tick in on:
        total += float(tick["rewards"][0, index]) * env.step_dt
    assert total == pytest.approx(-2.0 * int(sum(t["event"][0].item() for t in on)))


def test_a_clear_field_is_no_intervention_and_the_same_walk_as_with_the_guards_off(pools, walker_path):
    on, off = ours_env(pools[1], guards=True, seed=23), ours_env(pools[1], guards=False, seed=23)
    try:
        torch.manual_seed(0)
        a = run_ticks(on, 10)
        torch.manual_seed(0)  # the noise and the pushes of both walks are the same draws
        b = run_ticks(off, 10)
        assert not any(t["event"].any() for t in a)
        index = on.reward_manager.active_terms.index("guard_intervention")
        assert all((t["rewards"][:, index] == 0).all() for t in a)
        for x, y in zip(a, b):
            assert torch.equal(x["command"], y["command"]) and torch.equal(x["obs"], y["obs"]), "guards that do not fire change nothing"
    finally:
        on.close()
        off.close()


def test_the_guards_switch_is_off_in_the_other_tasks_and_a_config_in_ours(walker_path, pools):
    from nav_env import FULL, make_env

    env = make_env(FULL, num_envs=2, seed=1)
    try:
        assert env.action_manager.get_term("nav").guards is None and env.cfg.actions["nav"].guards is False
    finally:
        env.close()
    env = ours_env(pools[1], guards=False, num_envs=2)
    try:
        assert env.action_manager.get_term("nav").guards is None and "guard_intervention" not in env.reward_manager.active_terms
    finally:
        env.close()
