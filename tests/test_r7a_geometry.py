"""The rewards' geometry with footprints per environment (the pool's boxes) gives NVIDIA's numbers when every environment holds NVIDIA's
sizes: the same functions, run against NVIDIA's own on the stand-in environment of test_c6_rewards, with (n, k, 2) half sizes."""

import pytest
import torch

from microduck_nav import nav_math, rewards

from test_c6_rewards import Scene, Track, close, nvidia, set_pose  # noqa: F401  (the fixture and the stand-in)
from test_c6_rewards import pytestmark  # noqa: F401  (skipped where NVIDIA's clone is not)


def per_env(ours):
    field = ours.nav_field
    n = field.pos.shape[0]
    field.half_sizes = field.half_sizes[None].expand(n, -1, -1).clone()
    field.radii = field.radii[None].expand(n, -1).clone()
    return ours


@pytest.mark.parametrize("seed", [0, 1])
def test_the_approach_and_clearance_terms_equal_nvidias_with_sizes_per_environment(nvidia, seed):  # noqa: F811
    sc = Scene(nvidia, seed=seed)
    nv_env, ours = sc.nv_env(), per_env(sc.our_env())
    close(nvidia.rewards.obstacle_approach_risk(nv_env, safety_distance=0.2, time_horizon=2.0, commanded_risk_weight=1.0),
          rewards.approach_risk(ours, safety_distance=0.2, time_horizon=2.0, commanded_risk_weight=1.0))
    close(nvidia.rewards.obstacle_clearance_risk(nv_env, safety_distance=0.2), rewards.clearance_risk(ours))
    static = nav_math.obstacle_clearances(sc.pos, sc.our_env().nav_field.geometry())
    assert torch.equal(static, nav_math.obstacle_clearances(sc.pos, ours.nav_field.geometry()))


def test_the_bypass_term_equals_nvidias_step_by_step_with_sizes_per_environment(nvidia):  # noqa: F811
    sc = Scene(nvidia, seed=8, n=128)
    track = Track(nvidia, sc)
    track.ours = per_env(track.ours)
    sc.terminated[:] = False
    sc.boundary[:] = False
    nv_term = nvidia.rewards.NavigationObstacleBypass(None, track.nv_env)
    our_term = rewards.NavigationObstacleBypass(None, track.ours)
    assert our_term._per_env
    ids = torch.arange(128)
    nv_term.reset(ids)
    our_term.reset(ids)
    g = torch.Generator().manual_seed(2)
    start = sc.pos.clone()
    sc.request[:, 0] = 0.3
    for k in range(60):
        pos = start.clone()
        pos[:, :2] += torch.stack((sc.yaw.cos(), sc.yaw.sin()), -1) * (0.04 * k) + (torch.rand(128, 2, generator=g) - 0.5) * 0.05
        set_pose(sc, pos, sc.yaw)
        track.step(lambda e: nv_term(e), lambda e: our_term(e))
    assert torch.equal(our_term.eligible, nv_term.eligible) and torch.equal(our_term.bypassed, nv_term.bypassed)
    assert nv_term.eligible.sum() > 3


def test_a_box_that_differs_per_environment_is_measured_per_environment():
    """Two environments, the same centre, boxes of different half length: the clearance to the near face differs by exactly that."""
    centers = torch.zeros(2, 1, 3)
    centers[:, 0, 0] = 1.0
    rotations = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(2, 1, 4)
    half = torch.tensor([[[0.1, 0.5]], [[0.4, 0.5]]])
    clearance = nav_math.obstacle_clearances(torch.zeros(2, 3), (centers, rotations, half, torch.zeros(2, 1)), robot_radius=0.12)
    assert clearance[:, 0].tolist() == pytest.approx([0.9 - 0.12, 0.6 - 0.12])
