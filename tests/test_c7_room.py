"""C7 for the M2d task: the real `train` command on `Mjlab-Nav-Deploy-Room-MicroDuck`, 8 envs x 10 PPO iterations on CPU."""

import json
import math

import numpy as np
import onnx
import pytest

from microduck_nav import train_check

pytestmark = pytest.mark.slow

ENVS, ITERATIONS, SAVE = 8, 10, 5
TASK = "Mjlab-Nav-Deploy-Room-MicroDuck"
M2B_ITERATION_S = 10.9  # docs/M2.md: the deploy task, 8 envs on this CPU


@pytest.fixture(scope="module")
def run(tmp_path_factory, walker_path):
    return train_check.run(TASK, ENVS, ITERATIONS, SAVE, tmp_path_factory.mktemp("c7_room"))


def test_the_command_finishes_all_iterations_finite_and_saves(run):
    assert run["returncode"] == 0, run["output"][-3000:]
    assert f"Learning iteration {ITERATIONS - 1}/{ITERATIONS}" in run["output"]
    assert len(run["iteration_s"]) == ITERATIONS
    assert train_check.all_finite(run["scalars"])
    assert set(run["checkpoints"]) >= {"model_5.pt", "model_9.pt"}
    assert train_check.checkpoint_finite(run["log_dir"] / run["checkpoints"][-1])
    assert all(math.isfinite(v) for v in run["scalars"]["Train/mean_reward"])


def test_the_policy_exports_one_1343_input_and_three_outputs_and_names_the_task(run):
    path = run["log_dir"] / run["onnx"][-1]
    info = train_check.check_onnx(path)
    assert info["input"] == (1, 1343) and info["output"] == (1, 3) and info["finite"]
    assert -1.0 <= info["range"][0] <= info["range"][1] <= 1.0
    meta = {p.key: p.value for p in onnx.load(str(path)).metadata_props}
    assert meta["task"] == TASK and meta["policy_step_s"] == "0.1" and meta["lateral_limit"] == "0.0"


def test_the_episode_statistics_of_every_reward_term_are_logged(run):
    for term in ("progress", "goal_arrival", "barrier_bypass", "yaw_tracking", "command_tracking", "approach", "command_rate", "failure"):
        assert f"Episode_Reward/{term}" in run["scalars"]


def test_cpu_iteration_time_is_recorded(run, capsys):
    t = np.array(run["iteration_s"][1:])
    summary = {
        "task": TASK, "envs": ENVS, "iterations": ITERATIONS, "steps_per_env": 24, "walker_steps_per_iteration": 24 * 5,
        "iteration_s_mean": float(t.mean()), "iteration_s_median": float(np.median(t)), "iteration_s_max": float(t.max()),
        "collection_s_mean": float(np.mean(run["collection_s"][1:])), "learning_s_mean": float(np.mean(run["learning_s"][1:])),
        "m2b_iteration_s": M2B_ITERATION_S, "ratio_to_m2b": float(t.mean()) / M2B_ITERATION_S,
    }
    with capsys.disabled():
        print("\nC7-room " + json.dumps(summary))
    assert 0.1 < summary["iteration_s_mean"] < 300
