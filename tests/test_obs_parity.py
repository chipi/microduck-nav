"""The golden observation fixture robotics-lab's deployed actor is checked against: it must match what the training code builds now."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from microduck_nav.obs_parity import build_fixture

FIXTURE = Path(__file__).parent / "fixtures" / "obs_parity.json"


@pytest.fixture(scope="module")
def fresh():
    return build_fixture()


def test_the_committed_fixture_is_what_the_training_code_builds_now(fresh):
    golden = json.loads(FIXTURE.read_text())
    assert [s["name"] for s in golden["states"]] == [s["name"] for s in fresh["states"]]
    for old, new in zip(golden["states"], fresh["states"]):
        np.testing.assert_allclose(old["observation"], new["observation"], atol=1e-6, err_msg=old["name"])
    for key in ("joints", "default_pose", "zone_axial_factor", "step_s", "history", "layout"):
        assert golden[key] == fresh[key] or np.allclose(golden[key], fresh[key], atol=1e-7), key
    # regenerate with: uv run python scripts/export_obs_parity.py


def test_the_layout_is_the_one_the_deployment_reads(fresh):
    """Independent of the deployed code: hand-computed slots of the 1343."""
    by_name = {s["name"]: s for s in fresh["states"]}
    assert fresh["step_s"] == pytest.approx(0.02) and fresh["history"] == 10
    start, walking = (np.array(by_name[n]["observation"]) for n in ("start", "walking"))
    assert start.shape == (1343,)
    np.testing.assert_allclose(start[3:6], [0, 0, -1], atol=1e-6)  # gravity: down in the body frame, upright
    np.testing.assert_allclose(start[34:53], [0, 0, 0, 0.25, 0, 0.3] + [0] * 10 + [0, 0, 0], atol=1e-7)
    np.testing.assert_allclose(start[1333:], 1.0)  # empty ticks: aged 1 s
    assert not start[53:1333].any()
    # a pitch of 10 degrees about +y (nose down) puts gravity on the body's +x
    assert walking[3] == pytest.approx(math.sin(math.radians(10)), abs=1e-6) and walking[5] == pytest.approx(-math.cos(math.radians(10)), abs=1e-6)
    newest = walking[53 + 9 * 128: 53 + 10 * 128]
    frame = by_name["walking"]["frames"][-1]["axial_m"]
    assert np.allclose(newest[:64], frame, atol=1e-6) and np.allclose(newest[64:], (np.array(frame) > 0).astype(float))
    assert walking[1333 + 9] == pytest.approx(0.03, abs=1e-6)  # the newest tick's frame age
    out_of_range = by_name["walking"]["frames"][1]["axial_m"].index(4.5)
    assert walking[53 + 7 * 128 + out_of_range] == 0 and walking[53 + 7 * 128 + 64 + out_of_range] == 0
