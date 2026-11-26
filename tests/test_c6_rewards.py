"""Each reward term against NVIDIA's own functions, run unmodified on a stand-in environment (tests/nvidia_stubs.py)."""

from types import SimpleNamespace

import pytest
import torch

import nvidia_stubs as nv
from microduck_nav import layouts, nav_math, rewards
from microduck_nav import terminations as T
from microduck_nav.env_cfg import REWARD_WEIGHTS
from microduck_nav.field import ObstacleField, quat_from_yaw
from microduck_nav.obstacles import SLOT_NAMES, SLOTS

pytestmark = pytest.mark.skipif(not nv.available(), reason="NVIDIA's clone is not at ~/.cache/robotics-lab-tests/research/isaaclab")

N = 256
LIMITS = torch.tensor([0.35, 0.12, 0.75])


@pytest.fixture(scope="module")
def nvidia():
    mp = pytest.MonkeyPatch()
    yield nv.load(mp)
    mp.undo()


def xyzw(wxyz: torch.Tensor) -> torch.Tensor:
    return torch.cat((wxyz[..., 1:], wxyz[..., :1]), dim=-1)


class Scene:
    """The same random state, seen through NVIDIA's xyzw stand-in and through our wxyz stand-in."""

    def __init__(self, nvidia, seed=0, n=N, layout="mixed"):
        g = torch.Generator().manual_seed(seed)
        self.n = n
        ids = layouts.sample_layout_ids(n, layout, layouts.NVIDIA_WEIGHTS, g)
        local = layouts.sample_local_poses(ids, g=g)
        order = layouts.shuffle_slots(n, g=g)
        local = local.gather(1, order.unsqueeze(-1).expand(-1, -1, 3))
        robot_yaw = (torch.rand(n, generator=g) - 0.5) * 2 * torch.pi
        self.heading = robot_yaw
        c, s = robot_yaw.cos()[:, None], robot_yaw.sin()[:, None]
        shift = (torch.rand(n, 2, generator=g) - 0.5) * 20
        self.obstacle_pos = torch.zeros(n, 8, 3)
        self.obstacle_pos[..., 0] = shift[:, :1] + c * local[..., 0] - s * local[..., 1]
        self.obstacle_pos[..., 1] = shift[:, 1:] + s * local[..., 0] + c * local[..., 1]
        self.obstacle_yaw = robot_yaw[:, None] + local[..., 2]
        # robots near the course: some in front of it, some on it
        spread = torch.rand(n, 1, generator=g) * 2.0
        self.pos = torch.zeros(n, 3)
        self.pos[:, :2] = shift + (torch.rand(n, 2, generator=g) - 0.5) * 2 * spread + torch.stack((c[:, 0], s[:, 0]), -1) * 0.5
        self.pos[:, 2] = 0.12
        self.yaw = (torch.rand(n, generator=g) - 0.5) * 2 * torch.pi
        self.quat = quat_from_yaw(self.yaw)
        self.lin_vel = torch.randn(n, 3, generator=g) * 0.15
        self.lin_vel[:, 2] *= 0.1
        self.ang_vel = torch.randn(n, 3, generator=g) * 0.4
        self.request = torch.zeros(n, 3)
        self.request[:, 0] = torch.rand(n, generator=g) * 0.35
        self.request[:, 2] = (torch.rand(n, generator=g) - 0.5) * 1.5
        self.request[torch.rand(n, generator=g) < 0.1] = 0.0
        self.applied = (torch.rand(n, 3, generator=g) - 0.5) * 2 * LIMITS
        self.previous = (torch.rand(n, 3, generator=g) - 0.5) * 2 * LIMITS
        self.terminated = torch.rand(n, generator=g) < 0.1
        self.boundary = torch.rand(n, generator=g) < 0.1
        self.nvidia = nvidia
        self.destination = torch.zeros(n, 3)

    def nv_env(self):
        def robot_data():
            return SimpleNamespace(
                root_link_pos_w=nv.Wp(self.pos),
                root_link_quat_w=nv.Wp(xyzw(self.quat)),
                root_link_lin_vel_w=nv.Wp(self.lin_vel),
                root_link_ang_vel_w=nv.Wp(self.ang_vel),
            )

        scene = {"robot": SimpleNamespace(data=robot_data())}
        footprints = {}
        for i, slot in enumerate(SLOTS):
            q = xyzw(quat_from_yaw(self.obstacle_yaw[:, i]))
            scene[slot.name] = SimpleNamespace(data=SimpleNamespace(root_pos_w=nv.Wp(self.obstacle_pos[:, i]), root_quat_w=nv.Wp(q)))
            if slot.kind == "box":
                footprints[slot.name] = self.nvidia.geometry.ObstacleFootprintCfg(size=(slot.size[0], slot.size[1]))
            else:
                footprints[slot.name] = self.nvidia.geometry.ObstacleFootprintCfg(radius=slot.size[0])
        action = SimpleNamespace(
            cfg=SimpleNamespace(velocity_limits=(0.35, 0.12, 0.75)), velocity_commands=self.applied, previous_velocity_commands=self.previous
        )
        command = SimpleNamespace(
            destination_w=self.destination, is_destination_env=torch.ones(self.n, dtype=torch.bool),
            cfg=SimpleNamespace(destination_tolerance=0.2),
        )
        return SimpleNamespace(
            num_envs=self.n, device="cpu", step_dt=0.02, scene=scene, cfg=SimpleNamespace(obstacle_footprints=footprints),
            command_manager=SimpleNamespace(get_command=lambda name: self.request, get_term=lambda name: command),
            action_manager=SimpleNamespace(get_term=lambda name: action),
            termination_manager=SimpleNamespace(terminated=self.terminated, get_term=lambda name: self.boundary),
        )

    def our_env(self):
        field = ObstacleField(SimpleNamespace(num_envs=self.n, device="cpu"))
        field.pos[:] = self.obstacle_pos
        field.yaw[:] = self.obstacle_yaw
        data = SimpleNamespace(
            root_link_pos_w=self.pos, root_link_quat_w=self.quat, root_link_lin_vel_w=self.lin_vel, root_link_ang_vel_w=self.ang_vel,
            indexing=SimpleNamespace(free_joint_q_adr=torch.arange(7)),
            data=SimpleNamespace(qpos=torch.cat((self.pos, self.quat), dim=-1)),
        )
        action = SimpleNamespace(velocity_commands=self.applied, previous_velocity_commands=self.previous, velocity_limits=LIMITS)
        command = SimpleNamespace(
            destination_w=self.destination, is_destination_env=torch.ones(self.n, dtype=torch.bool),
            cfg=SimpleNamespace(destination_tolerance=0.2),
        )
        return SimpleNamespace(
            num_envs=self.n, device="cpu", step_dt=0.02, nav_field=field, scene={"robot": SimpleNamespace(data=data)},
            command_manager=SimpleNamespace(get_command=lambda name: self.request, get_term=lambda name: command),
            action_manager=SimpleNamespace(get_term=lambda name: action),
            termination_manager=SimpleNamespace(terminated=self.terminated, get_term=lambda name: self.boundary),
        )


@pytest.fixture(scope="module", params=[0, 1, 2])
def state(request, nvidia):
    sc = Scene(nvidia, seed=request.param)
    return sc, sc.nv_env(), sc.our_env()


def close(a, b, tol=2e-5):
    assert a.shape == b.shape
    assert torch.allclose(a, b, atol=tol, rtol=1e-4), f"max abs diff {(a - b).abs().max().item():.3e}"


def test_weights_are_nvidias_table():
    assert REWARD_WEIGHTS == {
        "progress": 5.0, "goal_arrival": 30.0, "barrier_bypass": 5.0, "yaw_tracking": 4.0,
        "command_tracking": 1.0, "approach": -4.0, "command_rate": -5.0, "failure": -30.0,
    }


def test_the_scene_has_robots_inside_obstacles_and_clear_ones(state):
    sc, _, ours = state
    clearance = nav_math.obstacle_clearances(sc.pos, ours.nav_field.geometry())
    assert (clearance.amin(dim=-1) < 0).any() and (clearance.amin(dim=-1) > 0.3).any()


def test_clearances_match(state, nvidia):
    _, nv_env, ours = state
    close(nvidia.geometry.obstacle_clearances(nv_env), nav_math.obstacle_clearances(ours.scene["robot"].data.root_link_pos_w, ours.nav_field.geometry()))


def test_requested_path_clearance_matches(state, nvidia):
    _, nv_env, ours = state
    theirs = nvidia.geometry.requested_path_clearance(nv_env)
    robot = ours.scene["robot"].data
    mine = nav_math.requested_path_clearance(robot.root_link_pos_w, robot.root_link_quat_w, ours.command_manager.get_command("twist"), ours.nav_field.geometry())
    close(theirs, mine)


def test_progress_and_gated_progress_match(state, nvidia):
    _, nv_env, ours = state
    close(nvidia.commands.navigation_progress(nv_env), rewards.navigation_progress(ours))
    close(nvidia.rewards.navigation_gated_progress(nv_env), rewards.gated_progress(ours))


def test_command_and_yaw_tracking_match(state, nvidia):
    _, nv_env, ours = state
    close(nvidia.rewards.navigation_velocity_tracking(nv_env), rewards.command_tracking(ours))
    close(nvidia.rewards.navigation_yaw_tracking(nv_env), rewards.yaw_tracking(ours))
    nonzero = nvidia.rewards.navigation_velocity_tracking(nv_env)
    assert (nonzero > 0).any() and (nonzero == 1).sum() < len(nonzero)


@pytest.mark.parametrize("params", [
    {"safety_distance": 0.2, "time_horizon": 2.0, "commanded_risk_weight": 1.0},
    {"safety_distance": 0.2, "time_horizon": 2.0, "prediction_dt": 0.1, "robot_radius": 0.12, "commanded_risk_weight": 0.0},
    {"safety_distance": 0.1, "time_horizon": 1.0, "commanded_risk_weight": 0.5},
])
def test_approach_risk_matches_with_the_tasks_parameters(state, nvidia, params):
    _, nv_env, ours = state
    theirs = nvidia.rewards.obstacle_approach_risk(nv_env, **params)
    mine = rewards.approach_risk(ours, **params)
    close(theirs, mine)
    assert (theirs > 0).any() and (theirs < 0.99).any()


def test_clearance_risk_and_command_rate_match(state, nvidia):
    _, nv_env, ours = state
    close(nvidia.rewards.obstacle_clearance_risk(nv_env, safety_distance=0.2), rewards.clearance_risk(ours))
    close(nvidia.rewards.navigation_action_rate(nv_env), rewards.command_rate(ours))


def test_failure_costs_exactly_thirty_after_the_manager_scales_by_dt(state):
    sc, _, ours = state
    per_step = rewards.terminated_penalty(ours) * REWARD_WEIGHTS["failure"] * ours.step_dt
    assert torch.equal(per_step < 0, sc.terminated) and torch.allclose(per_step[sc.terminated], torch.tensor(-30.0))


def test_the_collision_threshold_agrees_with_nvidias_on_random_forces(nvidia):
    g = torch.Generator().manual_seed(4)
    n = 400
    peaks = torch.rand(n, 8, 4, generator=g) * 0.25 * (torch.rand(n, 8, 4, generator=g) < 0.02)
    matrix = torch.zeros(n, 1, 4, 8, 3)
    matrix[..., 0] = peaks.permute(0, 2, 1)[:, None]
    theirs = nvidia.terminations.obstacle_collision(
        SimpleNamespace(scene={"obstacle_contacts": SimpleNamespace(data=SimpleNamespace(force_matrix_w_history=nv.Wp(matrix)))})
    )
    sensors = {
        T.CONTACT_PREFIX + name: SimpleNamespace(data=SimpleNamespace(force_history=torch.stack((peaks[:, i], torch.zeros(n, 4), torch.zeros(n, 4)), -1)[:, None]))
        for i, name in enumerate(SLOT_NAMES)
    }
    assert torch.equal(theirs, T.obstacle_collision(SimpleNamespace(scene=sensors)))
    assert 0.2 < theirs.float().mean() < 0.99


class Track:
    """Steps both implementations through the same constructed trajectory and compares every step."""

    def __init__(self, nvidia, sc: Scene):
        self.sc, self.nv_env, self.ours = sc, sc.nv_env(), sc.our_env()
        self.nvidia = nvidia

    def step(self, nv_term, our_term):
        a, b = nv_term(self.nv_env), our_term(self.ours)
        assert torch.allclose(a, b, atol=1e-4), (a, b)
        return b


def set_pose(sc: Scene, pos, yaw):
    sc.pos[:] = pos
    sc.yaw[:] = yaw
    sc.quat[:] = quat_from_yaw(yaw)


def test_goal_arrival_pays_once_and_is_revoked_on_a_later_failure(nvidia):
    n = 6
    sc = Scene(nvidia, seed=5, n=n)
    sc.destination[:] = torch.tensor([2.6, 0.0, 0.12])
    sc.terminated[:] = False
    sc.boundary[:] = False
    sc.obstacle_pos[:] = 50.0
    track = Track(nvidia, sc)
    nv_term = nvidia.rewards.NavigationGoalArrival(None, track.nv_env)
    our_term = rewards.NavigationGoalArrival(None, track.ours)
    path = torch.linspace(0.0, 2.6, 60)
    total = torch.zeros(n)
    for k, x in enumerate(path):
        pos = torch.zeros(n, 3)
        pos[:, 0] = x
        pos[:, 2] = 0.12
        pos[2, 0] = min(float(x), 1.0)
        set_pose(sc, pos, torch.zeros(n))
        sc.terminated[:] = False
        sc.boundary[:] = False
        sc.terminated[3] = k == len(path) - 1
        total += track.step(lambda e: nv_term(e), lambda e: our_term(e)) * 0.02
    assert total[0].item() == pytest.approx(1.0, abs=1e-4)
    assert total[2].item() == 0.0
    assert total[3].item() == 0.0
    for k in range(4):
        sc.terminated[:] = False
        sc.terminated[1] = True
        total += track.step(lambda e: nv_term(e), lambda e: our_term(e)) * 0.02
    assert total[1].item() == pytest.approx(0.0, abs=1e-4)
    assert total[0].item() == pytest.approx(1.0, abs=1e-4)
    nv_term.reset(torch.tensor([0, 1]))
    our_term.reset(torch.tensor([0, 1]))
    sc.terminated[:] = False
    again = track.step(lambda e: nv_term(e), lambda e: our_term(e)) * 0.02
    assert again[0].item() == pytest.approx(1.0, abs=1e-4)


def wall_scene(nvidia, n=6):
    """A 1.8 m wall 1 m ahead of robots at the origin facing +x."""
    sc = Scene(nvidia, seed=6, n=n, layout="one_wall")
    sc.pos[:] = torch.tensor([0.0, 0.0, 0.12])
    sc.yaw[:] = 0.0
    sc.quat[:] = quat_from_yaw(sc.yaw)
    sc.obstacle_pos[:] = 40.0
    sc.obstacle_pos[:, 6] = torch.tensor([1.0, 0.45, 0.0])
    sc.obstacle_pos[:, 7] = torch.tensor([1.0, -0.45, 0.0])
    sc.obstacle_yaw[:] = 0.0
    sc.request[:] = torch.tensor([0.3, 0.0, 0.0])
    sc.terminated[:] = False
    sc.boundary[:] = False
    return sc


def test_obstacle_bypass_pays_once_for_passing_the_wall_on_one_side_and_never_for_stopping(nvidia):
    n = 6
    sc = wall_scene(nvidia, n)
    track = Track(nvidia, sc)
    nv_term = nvidia.rewards.NavigationObstacleBypass(None, track.nv_env)
    our_term = rewards.NavigationObstacleBypass(None, track.ours)
    nv_term.reset(torch.arange(n))
    our_term.reset(torch.arange(n))
    steps = 80
    total = torch.zeros(n)
    for k in range(steps + 1):
        f = k / steps
        pos = torch.zeros(n, 3)
        pos[:, 2] = 0.12
        pos[0, 0] = 0.6 * f  # stops before the wall
        pos[1, 0], pos[1, 1] = 2.0 * f, 1.4 * min(1.0, 3 * f) * 1.0  # around the left end
        pos[2, 0], pos[2, 1] = 2.0 * f, -1.4 * min(1.0, 3 * f)  # around the right end
        pos[3, 0] = 2.0 * f  # straight through the wall
        pos[4, 0], pos[4, 1] = 2.0 * f, 0.8 * torch.sin(torch.tensor(f * 6.0))  # weaves across
        pos[5, 0] = 0.0
        set_pose(sc, pos, torch.zeros(n))
        total += track.step(lambda e: nv_term(e), lambda e: our_term(e)) * 0.02
    assert total[0] == 0 and total[5] == 0 and total[3] == 0
    assert total[1].item() == pytest.approx(1.0, abs=1e-4) and total[2].item() == pytest.approx(1.0, abs=1e-4)
    assert torch.equal(our_term.eligible, nv_term.eligible) and torch.equal(our_term.bypassed, nv_term.bypassed)
    assert our_term.eligible[:3].all() and not our_term.eligible[3], "walking through the wall voids the credit"
    assert our_term.bypassed[1] and our_term.bypassed[2] and not our_term.bypassed[3]


def test_bypass_on_random_scenes_agrees_step_by_step(nvidia):
    sc = Scene(nvidia, seed=8, n=128)
    track = Track(nvidia, sc)
    sc.terminated[:] = False
    sc.boundary[:] = False
    nv_term = nvidia.rewards.NavigationObstacleBypass(None, track.nv_env)
    our_term = rewards.NavigationObstacleBypass(None, track.ours)
    ids = torch.arange(128)
    nv_term.reset(ids)
    our_term.reset(ids)
    g = torch.Generator().manual_seed(2)
    start = sc.pos.clone()
    sc.request[:, 0] = 0.3
    pay = torch.zeros(128)
    for k in range(60):
        pos = start.clone()
        pos[:, :2] += torch.stack((sc.yaw.cos(), sc.yaw.sin()), -1) * (0.04 * k) + (torch.rand(128, 2, generator=g) - 0.5) * 0.05
        set_pose(sc, pos, sc.yaw)
        pay += track.step(lambda e: nv_term(e), lambda e: our_term(e))
    assert torch.equal(our_term.eligible, nv_term.eligible)
    assert (nv_term.eligible.sum() > 3)
