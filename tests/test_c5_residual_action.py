import math

import pytest
import torch

from microduck_nav.residual_action import ResidualCommand

DT = 0.02


def term(n=1, prev=None):
    t = ResidualCommand(n, dt=DT)
    if prev is not None:
        t.previous[:] = torch.tensor(prev, dtype=torch.float32)
    return t


def run(t, requested, action, age=0.0, have=True):
    r = torch.tensor([requested], dtype=torch.float32)
    a = torch.tensor([action], dtype=torch.float32)
    return t.process(r, a, torch.tensor([have]), torch.tensor([age]))[0]


def test_defaults_are_the_contract():
    t = ResidualCommand(1, dt=DT)
    assert t.limits.tolist() == pytest.approx([0.35, 0.12, 0.75])
    assert t.slew.tolist() == pytest.approx([1.0, 1.0, 3.0])
    assert t.max_age == 0.25
    assert t.max_lateral_ratio == 0.35


def test_limits_clamp_each_axis_and_speed_cap_scales_translation():
    norm = math.hypot(0.35, 0.12)
    scale = 0.3 / norm
    t = term(prev=[0.35 * scale, 0.12 * scale, 0.75])
    c = run(t, [0.3, 0.0, 0.5], [1.0, 1.0, 1.0])
    assert c[0].item() == pytest.approx(0.35 * scale, abs=1e-6)
    assert c[1].item() == pytest.approx(0.12 * scale, abs=1e-6)
    assert c[2].item() == pytest.approx(0.75, abs=1e-6)
    assert math.hypot(c[0].item(), c[1].item()) == pytest.approx(0.3, abs=1e-6)


def test_action_is_clipped_to_unit_range():
    a = run(term(prev=[0.2, 0.0, 0.0]), [0.2, 0.0, 0.0], [5.0, 0.0, 0.0])
    b = run(term(prev=[0.2, 0.0, 0.0]), [0.2, 0.0, 0.0], [1.0, 0.0, 0.0])
    assert torch.equal(a, b)


def test_never_faster_than_requested_when_unslewed():
    for fwd in (-1.0, 0.0, 0.4, 1.0):
        for lat in (-1.0, 0.0, 1.0):
            t = term(prev=[0.15, 0.0, 0.0])
            c = run(t, [0.15, 0.0, 0.0], [fwd, lat, 0.0])
            assert math.hypot(c[0].item(), c[1].item()) <= 0.15 + 1e-6


def test_no_reverse():
    c = run(term(prev=[0.0, 0.0, 0.0]), [0.1, 0.0, 0.0], [-1.0, 0.0, 0.0])
    assert c[0].item() == 0.0


def test_slew_forward_is_one_ms2_over_dt():
    c = run(term(), [0.3, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert c[0].item() == pytest.approx(0.02)


def test_slew_yaw_is_three_over_dt():
    c = run(term(), [0.2, 0.0, 0.75], [0.0, 0.0, 0.0])
    assert c[2].item() == pytest.approx(0.06)


def test_reaches_the_request_after_ten_ticks():
    t = term()
    for _ in range(10):
        c = run(t, [0.2, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert c[0].item() == pytest.approx(0.2, abs=1e-6)


def test_brake_slews_down_instead_of_jumping():
    t = term(prev=[0.3, 0.0, 0.0])
    c = run(t, [0.3, 0.0, 0.0], [-1.0, 0.0, 0.0])
    assert c[0].item() == pytest.approx(0.28)


def test_slewed_speed_can_exceed_a_lowered_request():
    t = term(prev=[0.3, 0.0, 0.0])
    c = run(t, [0.1, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert c[0].item() == pytest.approx(0.28)


def test_lateral_stays_inside_the_forward_cone():
    t = term(prev=[0.1, 0.12, 0.0])
    c = run(t, [0.1, 0.0, 0.0], [0.0, 1.0, 0.0])
    assert abs(c[1].item()) <= 0.35 * c[0].item() + 1e-7
    t = term(prev=[0.0, 0.0, 0.0])
    c = run(t, [0.0, 0.1, 0.0], [0.0, 0.0, 0.0])
    assert c[1].item() == 0.0


def test_zero_request_stops_at_once_even_while_moving():
    t = term(prev=[0.3, 0.1, 0.5])
    c = run(t, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    assert c.tolist() == [0.0, 0.0, 0.0]
    c = run(t, [0.2, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert c[0].item() == pytest.approx(0.02)


def test_stale_or_missing_frame_stops_at_once():
    t = term(prev=[0.3, 0.0, 0.0])
    assert run(t, [0.3, 0.0, 0.0], [0.0, 0.0, 0.0], age=0.26).tolist() == [0.0, 0.0, 0.0]
    t = term(prev=[0.3, 0.0, 0.0])
    assert run(t, [0.3, 0.0, 0.0], [0.0, 0.0, 0.0], have=False).tolist() == [0.0, 0.0, 0.0]


def test_age_exactly_at_the_bound_is_still_fresh():
    t = term(prev=[0.3, 0.0, 0.0])
    c = run(t, [0.3, 0.0, 0.0], [0.0, 0.0, 0.0], age=0.25)
    assert c[0].item() == pytest.approx(0.3)


def test_after_a_stop_the_next_command_restarts_from_zero():
    t = term(prev=[0.3, 0.0, 0.0])
    run(t, [0.3, 0.0, 0.0], [0.0, 0.0, 0.0], age=1.0)
    c = run(t, [0.3, 0.0, 0.0], [0.0, 0.0, 0.0], age=0.0)
    assert c[0].item() == pytest.approx(0.02)


def test_batch_rows_are_independent_and_reset_clears_rows():
    t = ResidualCommand(3, dt=DT)
    t.previous[:] = 0.2
    req = torch.tensor([[0.2, 0, 0], [0.0, 0, 0], [0.2, 0, 0]])
    c = t.process(req, torch.zeros(3, 3), torch.tensor([True, True, False]), torch.tensor([0.0, 0.0, 0.0]))
    assert c[:, 0].tolist() == pytest.approx([0.2, 0.0, 0.0])
    t.previous[:] = 0.2
    t.reset(torch.tensor([1]))
    assert t.previous[:, 0].tolist() == pytest.approx([0.2, 0.0, 0.2])


def test_matches_the_deployment_function():
    import numpy as np

    def command_from_action(action, requested, prev_applied, have_tof, tof_age, dt):
        CMD_LIMITS = np.array([0.35, 0.12, 0.75], np.float32)
        CMD_SLEW = np.array([1.0, 1.0, 3.0], np.float32)
        cmd = requested + np.clip(action, -1.0, 1.0) * CMD_LIMITS
        cmd = np.clip(cmd, [0.0, -CMD_LIMITS[1], -CMD_LIMITS[2]], CMD_LIMITS)
        req_speed, speed = np.linalg.norm(requested[:2]), max(np.linalg.norm(cmd[:2]), 1e-6)
        cmd[:2] *= min(req_speed / speed, 1.0)
        cmd = prev_applied + np.clip(cmd - prev_applied, -CMD_SLEW * dt, CMD_SLEW * dt)
        cmd[1] = np.clip(cmd[1], -0.35 * cmd[0], 0.35 * cmd[0])
        if not requested.any() or not have_tof or tof_age > 0.25:
            cmd[:] = 0.0
        return cmd.astype(np.float32)

    rng = np.random.default_rng(0)
    for _ in range(300):
        requested = np.array([rng.uniform(0, 0.35), rng.uniform(-0.1, 0.1), rng.uniform(-0.75, 0.75)], np.float32)
        action = rng.uniform(-1.5, 1.5, 3).astype(np.float32)
        prev = np.array([rng.uniform(0, 0.35), rng.uniform(-0.12, 0.12), rng.uniform(-0.75, 0.75)], np.float32)
        age = float(rng.uniform(0, 0.4))
        t = term(prev=prev.tolist())
        ours = run(t, requested.tolist(), action.tolist(), age=age).numpy()
        theirs = command_from_action(action, requested, prev, True, age, DT)
        np.testing.assert_allclose(ours, theirs, atol=1e-6)
