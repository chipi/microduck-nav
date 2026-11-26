"""PPO's view of a policy that acts every N walker steps: one transition per policy action.

mjlab 1.3.0 has one decimation, the physics substeps inside an environment step; it has no mechanism for a policy that acts
less often than the environment steps. The navigation action term holds its action for N steps (so any caller is right), and this
wrapper makes PPO see only the steps where the action is taken: it steps the environment N times with the same action, sums the
rewards, and ORs the dones, so rsl_rl's storage, GAE and `num_steps_per_env` count policy steps. The walker, the rewards and the
terminations still run at every environment step.

An environment that finishes mid-tick is reset by mjlab at once and runs on to the end of the tick under the action term's zero
held action; what it earns after the reset is not added to the finished episode's transition.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper


class ActionRepeatVecEnv(RslRlVecEnvWrapper):
    def __init__(self, inner: RslRlVecEnvWrapper, repeat: int):
        self.__dict__.update(inner.__dict__)  # the same, already reset, environment: a second reset would cost an episode
        self.repeat = repeat

    def step(self, actions: torch.Tensor):
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        n = self.num_envs
        reward = torch.zeros(n, device=self.device)
        terminated = torch.zeros(n, dtype=torch.bool, device=self.device)
        truncated = torch.zeros_like(terminated)
        logs: list[tuple[int, dict]] = []
        for _ in range(self.repeat):
            alive = ~(terminated | truncated)
            obs, rew, term, trunc, extras = self.env.step(actions)
            reward += torch.where(alive, rew, torch.zeros_like(rew))
            terminated |= term & alive
            truncated |= trunc & alive
            resets = int((term | trunc).sum())
            if resets and "log" in extras:
                logs.append((resets, extras["log"]))
        if logs:  # the episode statistics of every reset in the tick, weighted by how many environments reset
            total = sum(count for count, _ in logs)
            extras["log"] = {
                key: sum(torch.as_tensor(log[key], dtype=torch.float32) * count for count, log in logs if key in log) / total
                for key in {k for _, log in logs for k in log}
            }
        dones = (terminated | truncated).to(dtype=torch.long)
        if not self.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        return TensorDict(obs, batch_size=[n]), reward, dones, extras
