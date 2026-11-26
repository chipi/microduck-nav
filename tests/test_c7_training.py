"""C7: the real `train` command, 8 envs x 50 PPO iterations on CPU (about two minutes)."""

import json
import math

import numpy as np
import pytest
import torch

from microduck_nav import train_check
from microduck_nav.nav_export import build_actor

from test_c4_export import beta_mean

pytestmark = pytest.mark.slow

ENVS, ITERATIONS, SAVE = 8, 50, 25


@pytest.fixture(scope="module")
def run(tmp_path_factory, walker_path):
    return train_check.run("Mjlab-Nav-MicroDuck", ENVS, ITERATIONS, SAVE, tmp_path_factory.mktemp("c7"))


def test_the_command_finishes_all_iterations(run):
    assert run["returncode"] == 0, run["output"][-3000:]
    assert f"Learning iteration {ITERATIONS - 1}/{ITERATIONS}" in run["output"]
    assert len(run["iteration_s"]) == ITERATIONS


def test_no_nan_anywhere(run):
    assert train_check.all_finite(run["scalars"])
    assert "nan" not in " ".join(l for l in run["output"].splitlines() if "Mean" in l).lower()
    ckpt = run["log_dir"] / run["checkpoints"][-1]
    assert train_check.checkpoint_finite(ckpt)


def test_reward_is_finite_and_logged_every_term(run):
    s = run["scalars"]
    assert len(s["Train/mean_reward"]) >= 1 and all(math.isfinite(v) for v in s["Train/mean_reward"])
    for term in ("progress", "goal_arrival", "barrier_bypass", "yaw_tracking", "command_tracking", "approach", "command_rate", "failure"):
        assert len(s[f"Episode_Reward/{term}"]) == ITERATIONS
    assert any(v != 0 for v in s["Episode_Reward/progress"])
    assert 0.0 < s["Policy/mean_std"][-1] < 5.0
    assert s["Loss/learning_rate"][-1] == pytest.approx(3e-4)


def test_checkpoints_save_at_the_interval_and_at_the_end(run):
    assert run["checkpoints"] == ["model_0.pt", "model_25.pt", "model_49.pt"] or set(run["checkpoints"]) >= {"model_25.pt", "model_49.pt"}


def test_the_policy_exports_and_loads_in_onnxruntime_with_the_right_shapes(run):
    last = run["onnx"][-1]
    info = train_check.check_onnx(run["log_dir"] / last)
    assert info["input"] == (1, 1343) and info["output"] == (1, 3) and info["finite"]
    assert -1.0 <= info["range"][0] <= info["range"][1] <= 1.0


def test_the_onnx_is_the_checkpoints_actor(run):
    import onnxruntime as ort

    ckpt = run["log_dir"] / "model_49.pt"
    actor = build_actor()
    actor.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["actor_state_dict"])
    sess = ort.InferenceSession(str(ckpt.with_suffix(".onnx")), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(20):
        x = rng.normal(0, 1.5, (1, 1343)).astype(np.float32)
        worst = max(worst, float(np.abs(sess.run(None, {"obs": x})[0] - beta_mean(actor, torch.from_numpy(x)).detach().numpy()).max()))
    assert worst < 1e-5


def test_cpu_iteration_time_is_recorded(run, capsys):
    t = np.array(run["iteration_s"][1:])
    summary = {
        "envs": ENVS, "iterations": ITERATIONS, "steps_per_env": 24,
        "iteration_s_mean": float(t.mean()), "iteration_s_median": float(np.median(t)), "iteration_s_max": float(t.max()),
        "collection_s_mean": float(np.mean(run["collection_s"][1:])), "learning_s_mean": float(np.mean(run["learning_s"][1:])),
        "env_steps_per_s": float(np.mean(run["steps_per_second"][1:])),
    }
    with capsys.disabled():
        print("\nC7 " + json.dumps(summary))
    assert 0.1 < summary["iteration_s_mean"] < 120
