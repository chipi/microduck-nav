"""PPO settings of NVIDIA's navigation recipe (agents/rsl_rl_ppo_cfg.py, Isaac Lab exp/microduck 802b8d5, BSD-3-Clause).

Actor 128x128 tanh with observation normalisation and the Beta head; gamma 0.998, learning rate 3e-4 fixed,
entropy 0, clip 0.2, 5 epochs, 4 minibatches, lambda 0.95. Not in this first pass: their mirror loss (0.1).
The critic is not specified in the files read (it inherits the walker's); it is 128x128 tanh here.
"""

from __future__ import annotations

import os
from pathlib import Path

from mjlab.rl import MjlabOnPolicyRunner, RslRlOnPolicyRunnerCfg
from mjlab.rl.config import RslRlModelCfg, RslRlPpoAlgorithmCfg

from microduck_nav.nav_export import export_actor

BETA_HEAD = {"class_name": "microduck_nav.beta_head:MicroDuckBrakingBeta", "forward_braking_only": False}


def make_rl_cfg(experiment: str = "microduck_nav") -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(hidden_dims=(128, 128), activation="tanh", obs_normalization=True, distribution_cfg=dict(BETA_HEAD)),
        critic=RslRlModelCfg(hidden_dims=(128, 128), activation="tanh", obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.0,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3e-4,
            schedule="fixed",
            gamma=0.998,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        obs_groups={"actor": ("policy", "tof"), "critic": ("critic", "tof")},
        clip_actions=1.0,
        num_steps_per_env=24,
        max_iterations=1500,
        save_interval=100,
        experiment_name=experiment,
        run_name="",
        logger="wandb",
        wandb_project="microduck_nav",
    )


class NavOnPolicyRunner(MjlabOnPolicyRunner):
    """Saves the checkpoint and, next to it, the actor as (1, 1343) -> (1, 3) ONNX."""

    def save(self, path: str, infos=None) -> None:
        super().save(path, infos)
        try:
            export_actor(self.alg.get_policy(), Path(path).with_suffix(".onnx"))
        except Exception as e:  # training must not stop on an export problem
            print(f"[WARN] ONNX export failed (training continues): {e}")
