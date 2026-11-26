import numpy as np
import onnxruntime as ort
import torch

from microduck_nav.walker import Walker


def test_structure(walker_path):
    w = Walker.from_onnx(walker_path)
    assert w.obs_dim == 61 and w.act_dim == 14
    assert [tuple(l.weight.shape) for l in w.layers] == [(512, 61), (256, 512), (128, 256), (14, 128)]
    assert w.mean.shape == (1, 61) and w.denominator.shape == (1, 61)


def test_matches_onnxruntime_on_random_vectors(walker_path):
    w = Walker.from_onnx(walker_path)
    sess = ort.InferenceSession(str(walker_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    mean = w.mean.numpy()
    std = w.denominator.numpy()
    worst = 0.0
    for i in range(500):
        scale = (0.5, 1.0, 3.0)[i % 3]
        x = (mean + rng.normal(0, scale, (1, 61)) * std).astype(np.float32) if i % 2 else rng.normal(0, scale, (1, 61)).astype(np.float32)
        ref = sess.run(None, {"obs": x})[0]
        got = w(torch.from_numpy(x)).numpy()
        worst = max(worst, float(np.abs(ref - got).max()))
    assert worst < 1e-4
    print(f"max abs diff over 500 vectors: {worst:.3e}")


def test_batch_forward_equals_row_by_row(walker_path):
    w = Walker.from_onnx(walker_path)
    x = torch.randn(32, 61)
    batch = w(x)
    rows = torch.cat([w(x[i : i + 1]) for i in range(32)])
    torch.testing.assert_close(batch, rows, atol=1e-5, rtol=1e-5)


def test_no_gradients_and_large_batch(walker_path):
    w = Walker.from_onnx(walker_path)
    assert not any(p.requires_grad for p in w.parameters())
    assert w(torch.randn(4096, 61)).shape == (4096, 14)
