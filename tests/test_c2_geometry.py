import math

import numpy as np
import pytest
import torch

from microduck_nav import tof_geometry as geo
from mjlab_microduck.sim.tof import Tof


def pollen_directions():
    return Tof(None, 0).directions


def test_equal_angle_matches_pollen_tof():
    ours = geo.zone_directions("equal_angle").numpy()
    np.testing.assert_allclose(ours, pollen_directions(), atol=1e-6)


def test_zone_layout_row0_top_col0_left():
    d = geo.zone_directions("equal_angle")
    assert d[0, 2] > 0 and d[0, 1] > 0
    assert d[7, 2] > 0 and d[7, 1] < 0
    assert d[63, 2] < 0 and d[63, 1] < 0
    assert d[56, 2] < 0 and d[56, 1] > 0


def test_directions_are_unit_and_forward():
    for mode in ("equal_angle", "pinhole"):
        d = geo.zone_directions(mode)
        assert d.shape == (64, 3)
        np.testing.assert_allclose(d.norm(dim=-1).numpy(), 1.0, atol=1e-6)
        assert (d[:, 0] > 0).all()


def test_edge_zone_angles():
    eq = geo.zone_directions("equal_angle")
    pin = geo.zone_directions("pinhole")
    az_eq = math.degrees(math.atan2(eq[0, 1].item(), eq[0, 0].item()))
    az_pin = math.degrees(math.atan2(pin[0, 1].item(), pin[0, 0].item()))
    assert az_eq == pytest.approx(19.6875, abs=1e-4)
    assert az_pin == pytest.approx(math.degrees(math.atan(7 / 8 * math.tan(math.radians(22.5)))), abs=1e-4)
    assert az_pin == pytest.approx(19.9, abs=0.05)


def test_pinhole_inner_zones_hand_values():
    pin = geo.zone_directions("pinhole")
    t = math.tan(math.radians(22.5)) / 8
    # column 3, row 3 sits at (-t, -t) of the image plane: left of centre is +y
    expected = torch.tensor([1.0, t, t]) / torch.tensor([1.0, t, t]).norm()
    torch.testing.assert_close(pin[3 * 8 + 3], expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(pin[4 * 8 + 4], torch.tensor([1.0, -t, -t]) / torch.tensor([1.0, -t, -t]).norm(), atol=1e-6, rtol=0)


def test_radial_axial_round_trip_and_hand_case():
    d = geo.zone_directions("pinhole")
    radial = torch.full((2, 64), 2.0)
    axial = geo.radial_to_axial(radial, d)
    torch.testing.assert_close(axial, 2.0 * d[:, 0].expand(2, 64))
    torch.testing.assert_close(geo.axial_to_radial(axial, d), radial)
    centre = geo.radial_to_axial(torch.tensor([[1.0]]), torch.tensor([[1.0, 0.0, 0.0]]))
    assert centre.item() == 1.0
    off = torch.tensor([[math.cos(math.radians(30)), math.sin(math.radians(30)), 0.0]])
    assert geo.radial_to_axial(torch.tensor([[2.0]]), off).item() == pytest.approx(2 * math.cos(math.radians(30)))


def test_misses_stay_misses():
    d = geo.zone_directions("pinhole")
    radial = torch.full((1, 64), -1.0)
    assert (geo.radial_to_axial(radial, d) == -1.0).all()


def test_pattern_feeds_mjlab_in_site_frame():
    off, dirs = geo.ToFPatternCfg(mode="equal_angle").generate_rays(None, "cpu")
    assert off.shape == (64, 3) and (off == 0).all()
    np.testing.assert_allclose(dirs.numpy(), pollen_directions(), atol=1e-6)
