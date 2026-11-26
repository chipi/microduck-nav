"""Build the navigation actor on rsl_rl's MLPModel and export it to ONNX as (1, 1343) -> (1, 3)."""

from __future__ import annotations

import os
from pathlib import Path

import onnx
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

OBS_DIM = 1343
GROUP = "policy"
OBS_LAYOUT = "nav-1343-v1"  # the 1343-wide actor input as obs_parity.py builds it; bump when a slot moves


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


def nav_metadata(task: str, policy_step_s: float, lateral_limit: float) -> dict[str, str]:
    """What a deployer needs to run the model as it trained: the spacing of its ten history entries and the lateral bound."""
    return {
        "task": task,
        "policy_step_s": repr(round(float(policy_step_s), 6)),
        "lateral_limit": repr(round(float(lateral_limit), 6)),
        "obs_layout": OBS_LAYOUT,
    }


def export_actor(model: MLPModel, path, metadata: dict[str, str] | None = None) -> Path:
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
    if metadata:
        proto = onnx.load(str(path))
        for key, value in metadata.items():
            entry = proto.metadata_props.add()
            entry.key, entry.value = key, value
        onnx.save(proto, str(path))
    return path
