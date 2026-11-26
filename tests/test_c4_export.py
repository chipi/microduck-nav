import math

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

import types

from microduck_nav.nav_export import OBS_LAYOUT, build_actor, export_actor, nav_metadata
from microduck_nav.rl_cfg import runner_metadata

OBS = 1343


@pytest.fixture(scope="module")
def actor():
    torch.manual_seed(0)
    model = build_actor(hidden_dims=(128, 128))
    model.obs_normalizer._mean.copy_(torch.randn(1, OBS) * 0.3)
    model.obs_normalizer._std.copy_(torch.rand(1, OBS) * 2.0 + 0.5)
    return model


@pytest.fixture(scope="module")
def session(actor, tmp_path_factory):
    path = tmp_path_factory.mktemp("export") / "nav.onnx"
    export_actor(actor, path)
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]), path


def beta_mean(actor, x):
    logits = actor.mlp(actor.obs_normalizer(x))
    a, b = logits.unbind(dim=-2)
    alpha, beta = torch.nn.functional.softplus(a) + 1, torch.nn.functional.softplus(b) + 1
    return 2 * torch.distributions.Beta(alpha.double(), beta.double()).mean.float() - 1


def test_shapes(session):
    sess, path = session
    assert [i.name for i in sess.get_inputs()] == ["obs"]
    assert sess.get_inputs()[0].shape == [1, OBS]
    assert sess.get_outputs()[0].shape == [1, 3]
    out = sess.run(None, {"obs": np.zeros((1, OBS), np.float32)})[0]
    assert out.shape == (1, 3)
    graph = onnx.load(str(path)).graph
    assert len(graph.input) == 1 and len(graph.output) == 1


def test_normaliser_is_baked_in(session):
    _, path = session
    names = {i.name for i in onnx.load(str(path)).graph.initializer}
    assert "obs_normalizer._mean" in names


def test_output_equals_the_torch_beta_mean(actor, session):
    sess, _ = session
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(50):
        x = rng.normal(0, 2.0, (1, OBS)).astype(np.float32)
        got = sess.run(None, {"obs": x})[0]
        expected = beta_mean(actor, torch.from_numpy(x)).detach().numpy()
        worst = max(worst, float(np.abs(got - expected).max()))
    assert worst < 1e-5


def test_actor_deterministic_forward_equals_exported_mean(actor, session):
    sess, _ = session
    x = torch.randn(1, OBS)
    out = actor({"policy": x}).detach().numpy()
    got = sess.run(None, {"obs": x.numpy()})[0]
    np.testing.assert_allclose(got, out, atol=1e-5)


def force_logits(actor, forward=None, rest=0.0):
    """Set the last layer so that the logits are fixed whatever the input: bias only."""
    last = actor.mlp[-2]
    with torch.no_grad():
        last.weight.zero_()
        bias = torch.zeros(2, 3)
        if forward is not None:
            bias[0, 0], bias[1, 0] = forward
        bias[:, 1:] = rest
        last.bias.copy_(bias.reshape(-1))


def exported(actor, tmp_path):
    path = tmp_path / "forced.onnx"
    export_actor(actor, path)
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def test_collapsed_forward_head_brakes(tmp_path):
    torch.manual_seed(1)
    actor = build_actor(hidden_dims=(128, 128))
    force_logits(actor, forward=(-60.0, -60.0))
    out = exported(actor, tmp_path).run(None, {"obs": np.zeros((1, OBS), np.float32)})[0]
    assert out.tolist() == [[-1.0, 0.0, 0.0]]
    assert actor({"policy": torch.zeros(1, OBS)}).tolist() == [[-1.0, 0.0, 0.0]]


def test_collapsed_forward_alone_keeps_the_other_means(tmp_path):
    torch.manual_seed(1)
    actor = build_actor(hidden_dims=(128, 128))
    force_logits(actor, forward=(-60.0, -60.0), rest=0.0)
    last = actor.mlp[-2]
    with torch.no_grad():
        last.bias.view(2, 3)[0, 1] = 3.0
    out = exported(actor, tmp_path).run(None, {"obs": np.zeros((1, OBS), np.float32)})[0]
    assert out[0, 0] == -1.0
    assert out[0, 1] > 0.0


def test_healthy_symmetric_forward_head_is_not_a_brake(tmp_path):
    torch.manual_seed(1)
    actor = build_actor(hidden_dims=(128, 128))
    force_logits(actor, forward=(0.0, 0.0))
    out = exported(actor, tmp_path).run(None, {"obs": np.zeros((1, OBS), np.float32)})[0]
    assert out[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_nan_input_brakes(session):
    sess, _ = session
    x = np.random.default_rng(2).normal(size=(1, OBS)).astype(np.float32)
    x[0, 700] = float("nan")
    out = sess.run(None, {"obs": x})[0]
    assert out.tolist() == [[-1.0, 0.0, 0.0]]


@pytest.mark.parametrize("bad", [float("inf"), -float("inf")])
def test_infinite_input_does_not_brake_because_tanh_saturates(session, bad):
    sess, _ = session
    x = np.random.default_rng(2).normal(size=(1, OBS)).astype(np.float32)
    x[0, 700] = bad
    out = sess.run(None, {"obs": x})[0]
    assert np.isfinite(out).all() and (np.abs(out) <= 1).all()
    assert out.tolist() != [[-1.0, 0.0, 0.0]]


def test_nan_in_the_network_weights_brakes(tmp_path):
    torch.manual_seed(1)
    actor = build_actor(hidden_dims=(128, 128))
    with torch.no_grad():
        actor.mlp[-2].bias[1] = float("nan")
    out = exported(actor, tmp_path).run(None, {"obs": np.zeros((1, OBS), np.float32)})[0]
    assert out.tolist() == [[-1.0, 0.0, 0.0]]


def test_sampling_and_log_prob_are_consistent():
    torch.manual_seed(3)
    actor = build_actor(hidden_dims=(128, 128))
    obs = {"policy": torch.randn(256, OBS)}
    a = actor(obs, stochastic_output=True)
    assert a.shape == (256, 3) and (a.abs() <= 1.0).all()
    lp = actor.get_output_log_prob(a)
    assert lp.shape == (256,) and torch.isfinite(lp).all()
    d = actor.distribution
    a_t, b_t = d.params
    unit = (a + 1) / 2
    ref = torch.distributions.Beta(a_t, b_t).log_prob(unit).sum(-1) - 3 * math.log(2.0)
    torch.testing.assert_close(lp, ref, atol=1e-4, rtol=1e-4)
    assert torch.isfinite(d.entropy).all()
    ent_ref = torch.distributions.Beta(a_t, b_t).entropy().sum(-1) + 3 * math.log(2.0)
    torch.testing.assert_close(d.entropy, ent_ref, atol=1e-4, rtol=1e-4)
    kl = d.kl_divergence(d.params, d.params)
    assert kl.abs().max() < 1e-5


def test_forward_braking_only_range():
    torch.manual_seed(4)
    actor = build_actor(hidden_dims=(128, 128), forward_braking_only=True)
    obs = {"policy": torch.randn(512, OBS)}
    a = actor(obs, stochastic_output=True)
    assert (a[:, 0] <= 0).all() and (a[:, 0] >= -1).all()
    assert (a[:, 1:].abs() <= 1).all()
    mean = actor(obs)
    assert (mean[:, 0] <= 0).all()


def props(path):
    return {p.key: p.value for p in onnx.load(str(path)).metadata_props}


def test_export_carries_the_rate_and_lateral_limit(actor, tmp_path):
    path = tmp_path / "meta.onnx"
    export_actor(actor, path, nav_metadata("Mjlab-Nav-Deploy-MicroDuck", 0.1, 0.0))
    assert props(path) == {
        "task": "Mjlab-Nav-Deploy-MicroDuck",
        "policy_step_s": "0.1",
        "lateral_limit": "0.0",
        "obs_layout": OBS_LAYOUT,
    }
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert sess.run(None, {"obs": np.zeros((1, OBS), np.float32)})[0].shape == (1, 3)


def test_export_without_metadata_has_none(actor, tmp_path):
    path = tmp_path / "plain.onnx"
    export_actor(actor, path)
    assert props(path) == {}


def test_metadata_follows_the_task_settings():
    full = types.SimpleNamespace(policy_decimation=1, limits=(0.35, 0.12, 0.75))
    deploy = types.SimpleNamespace(policy_decimation=5, limits=(0.35, 0.0, 0.75))
    assert runner_metadata(full, 0.02, "microduck_nav") == nav_metadata("Mjlab-Nav-MicroDuck", 0.02, 0.12)
    assert runner_metadata(deploy, 0.02, "microduck_nav_deploy") == nav_metadata("Mjlab-Nav-Deploy-MicroDuck", 0.1, 0.0)
    assert runner_metadata(deploy, 0.02, "microduck_nav_deploy")["policy_step_s"] == "0.1"
