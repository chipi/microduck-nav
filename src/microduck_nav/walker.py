"""Pollen's walking policy as a batched torch module, rebuilt from its ONNX initialisers.

The exported file has a static batch of 1; this module reads its weights so that the policy
can run inside an action term for thousands of environments.
"""

from __future__ import annotations

from pathlib import Path

import onnx
import torch
from onnx import numpy_helper


class Walker(torch.nn.Module):
    def __init__(self, mean: torch.Tensor, denominator: torch.Tensor, weights: list[tuple[torch.Tensor, torch.Tensor]]):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("denominator", denominator)
        self.layers = torch.nn.ModuleList()
        for weight, bias in weights:
            layer = torch.nn.Linear(weight.shape[1], weight.shape[0])
            layer.weight.data.copy_(weight)
            layer.bias.data.copy_(bias)
            self.layers.append(layer)
        for p in self.parameters():
            p.requires_grad_(False)
        self.obs_dim = weights[0][0].shape[1]
        self.act_dim = weights[-1][0].shape[0]

    @classmethod
    def from_onnx(cls, path) -> "Walker":
        graph = onnx.load(str(Path(path))).graph
        init = {i.name: torch.from_numpy(numpy_helper.to_array(i).copy()) for i in graph.initializer}
        nodes = list(graph.node)
        sub = next(n for n in nodes if n.op_type == "Sub")
        div = next(n for n in nodes if n.op_type == "Div" and n.input[0] == sub.output[0])
        gemms = [n for n in nodes if n.op_type == "Gemm"]
        weights = [(init[g.input[1]], init[g.input[2]]) for g in gemms]
        assert all(n.op_type in {"Sub", "Div", "Gemm", "Elu"} for n in nodes), "unexpected op in the walker graph"
        return cls(init[sub.input[1]], init[div.input[1]], weights)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = (obs - self.mean) / self.denominator
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.nn.functional.elu(x)
        return x
