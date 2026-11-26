"""Impairment maths against hand cases and against NVIDIA's formulas (sensors.py:206-301, restated verbatim below)."""

import math

import pytest
import torch

from microduck_nav.impairments import (
    AcquisitionClock,
    Calibration,
    PacketChannel,
    ToFRandomization,
    corrupt_frame,
    sample_calibration,
    zone_dropout_probability,
)

MAX_D = 4.0


def identity_calibration(n, cfg, edge_column=-1, survival=1.0, midpoint_offset=0.0, noise_std=0.0):
    z = torch.zeros(n, 1)
    return Calibration(
        latency_frames=torch.zeros(n, dtype=torch.long),
        distance_bias=z.clone(),
        distance_scale=torch.ones(n, 1),
        noise_std=torch.full((n, 1), noise_std),
        floor_validity_midpoint_offset_m=torch.full((n, 1), midpoint_offset),
        near_floor_edge_survival_probability=torch.full((n, 1), survival),
        zone_bias=torch.zeros(n, 64),
        edge_column=torch.full((n, 1), edge_column, dtype=torch.long),
    )


def quiet_cfg(**kw):
    base = dict(invalid_zone_probability=0.0, distance_noise_fraction=0.0, floor_validity_transition_m=None)
    base.update(kw)
    return ToFRandomization(**base)


def ones_frame(value, n=1):
    return torch.full((n, 64), float(value))


# ---- hand cases ---------------------------------------------------------------------------


def test_range_gate_before_noise():
    cfg = quiet_cfg()
    cal = identity_calibration(1, cfg)
    ideal = torch.tensor([[0.10, 0.119, 0.12, 1.0, 4.0, 4.01, 0.0, float("inf"), float("nan")] + [1.0] * 55])
    out = corrupt_frame(ideal, torch.zeros_like(ideal, dtype=torch.bool), cal, cfg, MAX_D)
    assert out[0, :9].tolist() == [0.0, 0.0, pytest.approx(0.12), 1.0, 4.0, 0.0, 0.0, 0.0, 0.0]


def test_range_gate_after_noise():
    cfg = quiet_cfg()
    cal = identity_calibration(1, cfg)
    cal.distance_bias[:] = -0.02
    ideal = ones_frame(1.0)
    ideal[0, 0] = 0.131
    ideal[0, 1] = 3.99
    cal.distance_bias[:] = -0.02
    out = corrupt_frame(ideal, torch.zeros_like(ideal, dtype=torch.bool), cal, cfg, MAX_D)
    assert out[0, 0] == 0.0
    assert out[0, 1] == pytest.approx(3.97)
    cal.distance_bias[:] = 0.02
    out = corrupt_frame(ideal, torch.zeros_like(ideal, dtype=torch.bool), cal, cfg, MAX_D)
    assert out[0, 0] == pytest.approx(0.151)
    assert out[0, 1] == 0.0


def test_scale_bias_and_zone_bias():
    cfg = quiet_cfg()
    cal = identity_calibration(1, cfg)
    cal.distance_scale[:] = 1.02
    cal.distance_bias[:] = 0.01
    cal.zone_bias[0, 5] = 0.005
    out = corrupt_frame(ones_frame(1.0), torch.zeros(1, 64, dtype=torch.bool), cal, cfg, MAX_D)
    assert out[0, 5] == pytest.approx(1.0 * 1.02 + 0.01 + 0.005)
    assert out[0, 6] == pytest.approx(1.0 * 1.02 + 0.01)


def test_noise_is_gaussian_with_base_plus_fraction_of_depth():
    cfg = quiet_cfg(distance_noise_fraction=0.008)
    cal = identity_calibration(2, cfg, noise_std=0.005)
    ideal = torch.cat([ones_frame(1.0), ones_frame(3.0)])
    g = torch.Generator().manual_seed(11)
    out = corrupt_frame(ideal, torch.zeros(2, 64, dtype=torch.bool), cal, cfg, MAX_D, generator=g)
    draws = torch.randn(2, 64, generator=torch.Generator().manual_seed(11))
    expected = ideal + draws * (0.005 + 0.008 * ideal)
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=0)
    big = torch.cat([ones_frame(2.0, 4000)])
    cal = identity_calibration(4000, cfg, noise_std=0.01)
    out = corrupt_frame(big, torch.zeros(4000, 64, dtype=torch.bool), cal, cfg, MAX_D, generator=torch.Generator().manual_seed(3))
    err = out - big
    assert err.std().item() == pytest.approx(0.01 + 0.008 * 2.0, rel=0.02)
    assert abs(err.mean().item()) < 1e-3


def test_invalid_zone_probability_extremes_and_rate():
    ideal = ones_frame(1.0, 2000)
    cal = identity_calibration(2000, quiet_cfg())
    floor = torch.zeros(2000, 64, dtype=torch.bool)
    out = corrupt_frame(ideal, floor, cal, quiet_cfg(invalid_zone_probability=1.0), MAX_D, generator=torch.Generator().manual_seed(1))
    assert (out == 0).all()
    out = corrupt_frame(ideal, floor, cal, quiet_cfg(invalid_zone_probability=0.0), MAX_D, generator=torch.Generator().manual_seed(1))
    assert (out > 0).all()
    out = corrupt_frame(ideal, floor, cal, quiet_cfg(invalid_zone_probability=0.03), MAX_D, generator=torch.Generator().manual_seed(1))
    assert (out == 0).float().mean().item() == pytest.approx(0.03, abs=0.004)


def floor_cfg(**kw):
    return ToFRandomization(invalid_zone_probability=0.03, floor_validity_transition_m=(0.556, 0.015), **kw)


def test_floor_sigmoid_values():
    cfg = floor_cfg()
    cal = identity_calibration(1, cfg)
    depths = torch.tensor([[0.30, 0.55, 0.556, 0.58, 0.65] + [0.30] * 59])
    floor = torch.ones(1, 64, dtype=torch.bool)
    p = zone_dropout_probability(depths, floor, cal, cfg)
    assert p[0, 0] == 0.0 and p[0, 1] == 0.0
    assert p[0, 2] == pytest.approx(0.5)
    assert p[0, 3] == pytest.approx(1 / (1 + math.exp(-(0.58 - 0.556) / 0.015)), abs=1e-6)
    assert 1 - p[0, 3].item() == pytest.approx(0.168, abs=0.001)
    assert p[0, 4].item() > 0.997


def test_floor_sigmoid_midpoint_shift_and_non_floor_prior():
    cfg = floor_cfg()
    cal = identity_calibration(1, cfg, midpoint_offset=0.13)
    depths = torch.full((1, 64), 0.6)
    floor = torch.zeros(1, 64, dtype=torch.bool)
    floor[0, :32] = True
    p = zone_dropout_probability(depths, floor, cal, cfg)
    assert p[0, 0].item() == pytest.approx(1 / (1 + math.exp(-(0.6 - 0.686) / 0.015)), abs=1e-6)
    assert p[0, 40].item() == pytest.approx(0.03)


def test_edge_column_attenuation():
    cfg = floor_cfg()
    cal = identity_calibration(1, cfg, edge_column=7, survival=0.25)
    depths = torch.full((1, 64), 0.4)
    floor = torch.ones(1, 64, dtype=torch.bool)
    floor[0, 15] = False
    depths[0, 23] = 0.6
    depths[0, 31] = 0.11
    p = zone_dropout_probability(depths, floor, cal, cfg)
    assert p[0, 7].item() == pytest.approx(0.75)
    assert p[0, 39].item() == pytest.approx(0.75)
    assert p[0, 6].item() == 0.0
    assert p[0, 0].item() == 0.0
    assert p[0, 15].item() == pytest.approx(0.03)
    assert p[0, 23].item() > 0.5 and p[0, 23].item() != pytest.approx(0.75)
    assert p[0, 31].item() == 0.0
    cal_none = identity_calibration(1, cfg, edge_column=-1, survival=0.25)
    assert zone_dropout_probability(depths, floor, cal_none, cfg)[0, 7].item() == 0.0


# ---- NVIDIA's formulas, restated (sensors.py:206-301), on their (n, 8, 8, 1) layout ------


def nvidia_zone_dropout(cfg, calibration, ideal, floor_hit, columns):
    p = cfg.invalid_zone_probability
    if cfg.floor_validity_transition_m is not None:
        midpoint, width = cfg.floor_validity_transition_m
        midpoint = midpoint + calibration["floor_validity_midpoint_offset_m"]
        floor_dropout = torch.sigmoid((ideal - midpoint) / width)
        floor_dropout = torch.where(ideal <= 0.55, 0.0, floor_dropout)
        p = torch.where(floor_hit, floor_dropout, p)
    impaired = columns == calibration["edge_column"]
    near = floor_hit & (ideal >= cfg.minimum_distance) & (ideal <= 0.55) & impaired
    edge = 1.0 - (1.0 - p) * calibration["near_floor_edge_survival_probability"]
    return torch.where(near, edge, p)


def nvidia_corrupt(cfg, calibration, ideal, floor_hit, columns, max_distance):
    valid = torch.isfinite(ideal) & (ideal >= cfg.minimum_distance) & (ideal <= max_distance)
    noisy = ideal * calibration["distance_scale"] + calibration["distance_bias"]
    noisy = noisy + calibration["zone_bias"]
    std = calibration["noise_std"] + cfg.distance_noise_fraction * ideal
    noisy = noisy + torch.randn_like(ideal) * std
    p = nvidia_zone_dropout(cfg, calibration, ideal, floor_hit, columns)
    valid &= torch.rand_like(ideal) >= p
    valid &= (noisy >= cfg.minimum_distance) & (noisy <= max_distance)
    return torch.where(valid, noisy, 0.0)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_corrupt_frame_equals_nvidia_formulas_with_the_same_draws(seed):
    n = 64
    cfg = ToFRandomization(
        near_floor_edge_impairment_probability=0.5, floor_validity_midpoint_offset_m=(0.0, 0.13)
    )
    src = torch.Generator().manual_seed(100 + seed)
    ideal = torch.rand(n, 64, generator=src) * 4.6
    ideal[:, ::11] = 0.0
    floor = torch.rand(n, 64, generator=src) < 0.4
    cal = sample_calibration(n, cfg, torch.Generator().manual_seed(seed))
    nv = {
        "distance_bias": cal.distance_bias.view(n, 1, 1, 1),
        "distance_scale": cal.distance_scale.view(n, 1, 1, 1),
        "noise_std": cal.noise_std.view(n, 1, 1, 1),
        "floor_validity_midpoint_offset_m": cal.floor_validity_midpoint_offset_m.view(n, 1, 1, 1),
        "near_floor_edge_survival_probability": cal.near_floor_edge_survival_probability.view(n, 1, 1, 1),
        "zone_bias": cal.zone_bias.view(n, 8, 8, 1),
        "edge_column": cal.edge_column.view(n, 1, 1, 1),
    }
    columns = torch.arange(8)[None, None, :, None]
    torch.manual_seed(seed + 500)
    ref = nvidia_corrupt(cfg, nv, ideal.view(n, 8, 8, 1), floor.view(n, 8, 8, 1), columns, MAX_D).view(n, 64)
    mine = corrupt_frame(ideal, floor, cal, cfg, MAX_D, generator=torch.Generator().manual_seed(seed + 500))
    torch.testing.assert_close(mine, ref, atol=1e-6, rtol=0)
    assert (mine == 0).any() and (mine > 0).any()


def test_calibration_equals_nvidia_draw_order():
    cfg = ToFRandomization(near_floor_edge_impairment_probability=0.5, floor_validity_midpoint_offset_m=(0.0, 0.13))
    n = 50
    torch.manual_seed(77)
    latency = torch.randint(0, 2, (n,))
    ref = {}
    for name in (
        "distance_bias",
        "distance_scale",
        "noise_std",
        "floor_validity_midpoint_offset_m",
        "near_floor_edge_survival_probability",
    ):
        low, high = getattr(cfg, name)
        ref[name] = torch.empty(n, 1).uniform_(low, high)
    ref["zone_bias"] = torch.empty(n, 64).uniform_(*cfg.zone_bias)
    impaired = torch.rand(n, 1) < cfg.near_floor_edge_impairment_probability
    edge = torch.randint(0, 2, (n, 1)) * 7
    ref_edge = torch.where(impaired, edge, -1)
    cal = sample_calibration(n, cfg, torch.Generator().manual_seed(77))
    assert torch.equal(cal.latency_frames, latency)
    for name, value in ref.items():
        torch.testing.assert_close(getattr(cal, name), value, atol=0, rtol=0)
    assert torch.equal(cal.edge_column, ref_edge)


def test_calibration_ranges_and_edge_rate():
    cfg = ToFRandomization(near_floor_edge_impairment_probability=0.5, floor_validity_midpoint_offset_m=(0.0, 0.13))
    cal = sample_calibration(4000, cfg, torch.Generator().manual_seed(5))
    assert cal.distance_bias.min() >= -0.02 and cal.distance_bias.max() <= 0.02
    assert cal.distance_scale.min() >= 0.98 and cal.distance_scale.max() <= 1.02
    assert cal.noise_std.min() >= 0.003 and cal.noise_std.max() <= 0.01
    assert cal.zone_bias.min() >= -0.01 and cal.zone_bias.max() <= 0.01
    assert cal.near_floor_edge_survival_probability.min() >= 0.02 and cal.near_floor_edge_survival_probability.max() <= 0.55
    assert cal.floor_validity_midpoint_offset_m.min() >= 0.0 and cal.floor_validity_midpoint_offset_m.max() <= 0.13
    assert set(cal.latency_frames.unique().tolist()) == {0, 1}
    impaired = cal.edge_column >= 0
    assert impaired.float().mean().item() == pytest.approx(0.5, abs=0.03)
    assert set(cal.edge_column[impaired].unique().tolist()) == {0, 7}


def test_same_seed_same_calibration_and_frame():
    cfg = ToFRandomization()
    a = sample_calibration(8, cfg, torch.Generator().manual_seed(9))
    b = sample_calibration(8, cfg, torch.Generator().manual_seed(9))
    assert torch.equal(a.zone_bias, b.zone_bias)
    ideal = torch.rand(8, 64) * 3
    f = torch.zeros(8, 64, dtype=torch.bool)
    o1 = corrupt_frame(ideal, f, a, cfg, MAX_D, generator=torch.Generator().manual_seed(2))
    o2 = corrupt_frame(ideal, f, b, cfg, MAX_D, generator=torch.Generator().manual_seed(2))
    assert torch.equal(o1, o2)


# ---- delivery: latency, packet loss, age (sensors.py:164-204) ------------------------------


def frame_of(v, n):
    return torch.full((n, 64), float(v))


def test_latency_zero_delivers_the_same_frame():
    ch = PacketChannel(2, max_latency=1)
    ids = torch.arange(2)
    ch.push(ids, frame_of(1.0, 2), torch.tensor([0.1, 0.1]), torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.long))
    assert (ch.delivered_frame == 1.0).all()
    assert ch.age(0.1).tolist() == pytest.approx([0.0, 0.0])


def test_latency_one_delivers_the_previous_capture_and_ages_it():
    ch = PacketChannel(1, max_latency=1)
    ids = torch.arange(1)
    lat = torch.ones(1, dtype=torch.long)
    ok = torch.ones(1, dtype=torch.bool)
    ch.push(ids, frame_of(1.0, 1), torch.tensor([0.0667]), ok, lat)
    assert ch.delivered_time.item() == -1.0
    assert not ch.available.item()
    ch.push(ids, frame_of(2.0, 1), torch.tensor([0.1333]), ok, lat)
    assert (ch.delivered_frame == 1.0).all()
    assert ch.age(0.1333).item() == pytest.approx(0.1333 - 0.0667)
    ch.push(ids, frame_of(3.0, 1), torch.tensor([0.2]), ok, lat)
    assert (ch.delivered_frame == 2.0).all()


def test_dropped_packet_keeps_previous_frame_and_true_age():
    ch = PacketChannel(1, max_latency=1)
    ids = torch.arange(1)
    zero = torch.zeros(1, dtype=torch.long)
    ch.push(ids, frame_of(1.0, 1), torch.tensor([0.1]), torch.ones(1, dtype=torch.bool), zero)
    ch.push(ids, frame_of(2.0, 1), torch.tensor([0.2]), torch.zeros(1, dtype=torch.bool), zero)
    assert (ch.delivered_frame == 1.0).all()
    assert ch.age(0.2).item() == pytest.approx(0.1)
    assert ch.age(0.5).item() == pytest.approx(0.4)


def test_reset_clears_only_the_selected_rows():
    ch = PacketChannel(2, max_latency=1)
    ids = torch.arange(2)
    zero = torch.zeros(2, dtype=torch.long)
    ch.push(ids, frame_of(5.0, 2), torch.tensor([0.1, 0.1]), torch.ones(2, dtype=torch.bool), zero)
    ch.reset(torch.tensor([1]))
    assert ch.available.tolist() == [True, False]
    assert (ch.delivered_frame[1] == 0).all() and (ch.delivered_frame[0] == 5.0).all()


# ---- acquisition clock (sensors.py:83-88,164-174) -------------------------------------------


def test_clock_is_15_hz_on_average_and_quantised_to_the_step():
    clock = AcquisitionClock(period=1.0 / 15.0)
    times = []
    for step in range(1, 501):
        if clock.advance(0.02):
            times.append(step * 0.02)
    assert len(times) in (149, 150)
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert set(round(g, 6) for g in gaps) <= {0.06, 0.08}
    assert sum(gaps) / len(gaps) == pytest.approx(1 / 15, abs=0.003)


def test_clock_first_capture_after_one_period():
    clock = AcquisitionClock(period=1.0 / 15.0)
    assert [clock.advance(0.02) for _ in range(4)] == [False, False, False, True]
