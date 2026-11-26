"""Build the navigation actor on rsl_rl's MLPModel and export it to ONNX as (1, 1343) -> (1, 3)."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

OBS_DIM = 1343
GROUP = "policy"


def build_actor(
    obs_dim: int = OBS_DIM,
    hidden_dims=(128, 128),
    activation: str = "tanh",
    forward_braking_only: bool = False,
) -> MLPModel:
    obs = TensorDict({GROUP: torch.zeros(1, obs_dim)}, batch_size=[1])
    return MLPModel(
        obs,
        {"actor": [GROUP]},
        "actor",
        3,
        hidden_dims=list(hidden_dims),
        activation=activation,
        obs_normalization=True,
        distribution_cfg={
            "class_name": "microduck_nav.beta_head:MicroDuckBrakingBeta",
            "forward_braking_only": forward_braking_only,
        },
    )


def export_actor(model: MLPModel, path) -> Path:
    """Same settings as mjlab's runner export (opset 18, legacy exporter, static batch of 1)."""
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    wrapper = model.as_onnx(verbose=False)
    wrapper.to("cpu").eval()
    torch.onnx.export(
        wrapper,
        wrapper.get_dummy_inputs(),
        str(path),
        export_params=True,
        opset_version=18,
        input_names=wrapper.input_names,
        output_names=wrapper.output_names,
        dynamic_axes={},
        dynamo=False,
    )
    return path
