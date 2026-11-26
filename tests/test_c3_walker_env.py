import numpy as np
import onnxruntime as ort
import pytest
import torch

from microduck_nav.walker import Walker
from microduck_nav.walker_speed import make_cfg, measure
from mjlab.envs import ManagerBasedRlEnv


def test_env_runs_on_cpu_and_the_actor_group_is_the_walkers_61_vector(walker_path):
    env = ManagerBasedRlEnv(make_cfg(0.3, 2, noise=False), device="cpu")
    obs, _ = env.reset()
    assert obs["actor"].shape == (2, 61)
    walker = Walker.from_onnx(walker_path)
    sess = ort.InferenceSession(str(walker_path), providers=["CPUExecutionProvider"])
    x = obs["actor"].float()
    for i in range(2):
        ref = sess.run(None, {"obs": x[i : i + 1].numpy()})[0]
        np.testing.assert_allclose(walker(x[i : i + 1]).numpy(), ref, atol=1e-4)
    cmd = env.command_manager.get_term("twist").command
    assert torch.allclose(cmd[:, 0], torch.full((2,), 0.3)) and cmd[:, 1:].abs().max() == 0
    env.close()


def test_short_run_is_finite(walker_path):
    r = measure(0.3, num_envs=2, steps=40, settle=20, noise=False, walker_path=walker_path)
    assert np.isfinite(r.mean_vx_body) and np.isfinite(r.mean_world_speed_settled)
