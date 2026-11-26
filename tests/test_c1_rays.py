
import numpy as np
import pytest
import torch

from microduck_nav import tof_geometry as geo
from microduck_nav.ray_scene import build_ray_scene, cpu_reference_distances

BOX_HALF = (0.05, 0.3, 0.3)
BOX_X = torch.tensor([1.0, 1.5, 2.0, 9.0])
PITCH = torch.tensor([0.0, 0.0, 0.3491, 0.3491])


@pytest.fixture(scope="module")
def rs():
    s = build_ray_scene(num_envs=4, box_half=BOX_HALF, pattern_mode="pinhole")
    box = torch.zeros(4, 3)
    box[:, 0] = BOX_X
    box[:, 2] = 0.3
    s.place(box_pos=box, head_pitch=PITCH)
    return s


def test_ray_path_runs_on_cpu(rs):
    d = rs.sense()
    assert d.shape == (4, 64)
    assert str(rs.sim.device) == "cpu"
    assert (d >= 0).any()


@pytest.mark.parametrize("mode", ["pinhole", "equal_angle"])
def test_distances_match_mj_ray_loop(mode):
    s = build_ray_scene(num_envs=4, box_half=BOX_HALF, pattern_mode=mode)
    box = torch.zeros(4, 3)
    box[:, 0] = BOX_X
    box[:, 2] = 0.3
    s.place(box_pos=box, head_pitch=PITCH)
    warp = s.sense().numpy()
    ref = cpu_reference_distances(s, max_distance=4.0)
    assert ref.shape == warp.shape
    np.testing.assert_array_equal(warp < 0, ref < 0)
    hit = ref >= 0
    assert hit.sum() > 4 * 20
    assert np.abs(warp[hit] - ref[hit]).max() < 1e-4


def test_box_is_seen_per_world_analytically(rs):
    pitch0 = torch.zeros(4)
    box = torch.zeros(4, 3)
    box[:, 0] = BOX_X
    box[:, 2] = 0.22
    rs.place(box_pos=box, head_pitch=pitch0)
    d = rs.sense()
    dirs = geo.zone_directions("pinhole")
    face = BOX_X - BOX_HALF[0]
    for w in range(3):
        for zone in (3 * 8 + 3, 4 * 8 + 4):
            expected = face[w] / dirs[zone, 0]
            assert d[w, zone].item() == pytest.approx(expected.item(), abs=1e-4)
    far_zone = 3 * 8 + 3
    assert d[3, far_zone] < 0 or d[3, far_zone] > 8.0


def test_moving_the_box_in_one_world_changes_only_that_world(rs):
    pitch0 = torch.zeros(4)
    box = torch.zeros(4, 3)
    box[:, 0] = 1.5
    box[:, 2] = 0.22
    rs.place(box_pos=box, head_pitch=pitch0)
    before = rs.sense().clone()
    box[1, 0] = 0.8
    rs.place(box_pos=box, head_pitch=pitch0)
    after = rs.sense()
    assert torch.equal(before[[0, 2, 3]], after[[0, 2, 3]])
    assert not torch.equal(before[1], after[1])
    zone = 3 * 8 + 3
    assert after[1, zone].item() < before[1, zone].item() - 0.6


def test_floor_is_hit_when_pitched_down(rs):
    box = torch.zeros(4, 3)
    box[:, 0] = 9.0
    rs.place(box_pos=box, head_pitch=torch.full((4,), 0.3491))
    d = rs.sense()
    bottom_row = d[:, 56:64]
    assert (bottom_row > 0).all()
    assert (bottom_row < 1.0).all()
