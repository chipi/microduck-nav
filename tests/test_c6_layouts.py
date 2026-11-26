import math

import pytest
import torch

from microduck_nav import layouts, nav_math
from microduck_nav.field import quat_from_yaw
from microduck_nav.obstacles import SLOT_INDEX, SLOT_NAMES, SLOTS, footprints, slot_spec

import mujoco


def poses(name, n=64, seed=0, **kw):
    g = torch.Generator().manual_seed(seed)
    ids = layouts.sample_layout_ids(n, name, None, g)
    return layouts.sample_local_poses(ids, g=g, **kw)


def test_every_slot_compiles_and_has_its_primitive():
    for slot in SLOTS:
        model = slot_spec(slot).compile()
        assert model.ngeom == 1 and model.nbody == 2
        if slot.kind == "box":
            assert model.geom_type[0] == mujoco.mjtGeom.mjGEOM_BOX
            assert tuple(model.geom_size[0] * 2) == pytest.approx(slot.size)
        else:
            assert model.geom_type[0] == mujoco.mjtGeom.mjGEOM_CYLINDER
            assert model.geom_size[0][0] == pytest.approx(slot.size[0])
            assert model.geom_size[0][1] * 2 == pytest.approx(slot.size[1])
        top = model.geom_pos[0][2] + (model.geom_size[0][2] if slot.kind == "box" else model.geom_size[0][1])
        assert top == pytest.approx(slot.height)


def test_slots_follow_nvidias_names_order_and_footprints():
    assert SLOT_NAMES == ("obstacle", "crate", "tall_box", "low_block", "cylinder", "thin_post", "wall_left", "wall_right")
    assert tuple(layouts.OBSTACLE_CELLS) == SLOT_NAMES and set(layouts.PARKED) == set(SLOT_NAMES)
    half, radii = footprints()
    assert half[SLOT_INDEX["tall_box"]].tolist() == pytest.approx([0.2, 0.15])
    assert half[SLOT_INDEX["wall_left"]].tolist() == pytest.approx([0.06, 0.45])
    assert radii[SLOT_INDEX["obstacle"]].item() == pytest.approx(0.1143)
    assert layouts.NVIDIA_WEIGHTS["scattered"] == 0.65 and sum(layouts.NVIDIA_WEIGHTS.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("name", layouts.LAYOUT_NAMES)
def test_a_seed_gives_the_same_layout_and_another_seed_a_different_one(name):
    a, b, c = poses(name, seed=3), poses(name, seed=3), poses(name, seed=4)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert torch.isfinite(a).all() and a.shape == (64, 8, 3)


def test_shuffle_keeps_walls_and_permutes_the_six_props_per_seed():
    g1, g2 = torch.Generator().manual_seed(9), torch.Generator().manual_seed(9)
    o1, o2 = layouts.shuffle_slots(500, g=g1), layouts.shuffle_slots(500, g=g2)
    assert torch.equal(o1, o2)
    assert (o1[:, 6:] == torch.tensor([6, 7])).all()
    assert (o1[:, :6].sort(dim=1).values == torch.arange(6)).all()
    assert len({tuple(r.tolist()) for r in o1}) > 100


def test_layout_mix_draws_by_nvidias_weights():
    g = torch.Generator().manual_seed(1)
    ids = layouts.sample_layout_ids(20000, "mixed", layouts.NVIDIA_WEIGHTS, g)
    freq = torch.bincount(ids, minlength=len(layouts.LAYOUT_NAMES)).float() / len(ids)
    for i, name in enumerate(layouts.LAYOUT_NAMES):
        assert freq[i].item() == pytest.approx(layouts.NVIDIA_WEIGHTS.get(name, 0.0), abs=0.012)
    assert freq[layouts.LAYOUT_NAMES.index("one_wall")] == 0


def test_one_wall_is_one_1_8_m_wall_across_the_approach():
    p = poses("one_wall", n=500, seed=2)
    left, right = p[:, SLOT_INDEX["wall_left"]], p[:, SLOT_INDEX["wall_right"]]
    centre = 0.5 * (left + right)
    assert (centre[:, 0] > 0.6 - 0.16).all() and (centre[:, 0] < 1.0 + 0.16).all()
    assert (left[:, 2] == right[:, 2]).all() and (left[:, 2].abs() <= 0.1 + 1e-6).all()
    gap = (left[:, :2] - right[:, :2]).norm(dim=-1)
    assert gap.mean().item() == pytest.approx(0.9, abs=1e-5)
    wall_span = torch.stack((left[:, 1] - right[:, 1], left[:, 0] - right[:, 0]), -1)
    assert wall_span[:, 0].min() > 0.85


@pytest.mark.parametrize("name", [n for n in layouts.LAYOUT_NAMES if n != "ring"])
def test_the_spawn_stays_clear_of_every_layout(name):
    """The robot's footprint (0.12 m) at the reset pose does not overlap any slot, over 3000 draws."""
    g = torch.Generator().manual_seed(11)
    n = 3000
    ids = layouts.sample_layout_ids(n, name, None, g)
    local = layouts.sample_local_poses(ids, g=g)
    order = layouts.shuffle_slots(n, g=g)
    local = local.gather(1, order.unsqueeze(-1).expand(-1, -1, 3))
    half, radii = footprints()
    centres = torch.zeros(n, 8, 3)
    centres[:, :, :2] = local[..., :2]
    geom = (centres, quat_from_yaw(local[..., 2]), half, radii)
    clearance = nav_math.obstacle_clearances(torch.zeros(n, 3), geom).amin(dim=-1)
    assert clearance.min().item() > 0.0, f"{name}: min clearance {clearance.min().item():.3f}"


def test_scattered_has_two_rows_of_three_and_walls_to_the_sides():
    p = poses("scattered", n=200, seed=5, spacing_scale=1.6)
    local = p[:, :6]
    assert ((local[..., 0] - 0.8).abs() < 0.2 + 1e-6).sum() + ((local[..., 0] - 1.7).abs() < 0.2 + 1e-6).sum() == 200 * 6
    walls = p[:, 6:]
    assert (walls[..., 1].abs() >= 2.2 - 0.15 - 1e-6).all() and (walls[..., 1].abs() <= 2.5 + 0.15 + 1e-6).all()
