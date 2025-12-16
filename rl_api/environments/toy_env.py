from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np
import torch

from rl_api.environments.vectorized_env import VectorEnv
from rl_api.environments.types import BatchObs


class ToyVectorEnv(VectorEnv):
    """
    Simple toy env for sequence-based RL.

    State per env: scalar x_t.
    Actions: 0 -> move -1, 1 -> move 0, 2 -> move +1.
    Dynamics: x_{t+1} = x_t + step_size * move.
    Reward: r_t = -|x_{t+1} - 0| (maximize by approaching 0).
    Episode ends after max_steps or if |x| exceeds x_limit.

    Observation per env:
        obs[i]: last seq_len states as shape (seq_len, 1)
    Context:
        ctx[i]: (1,) containing normalized time t / max_steps.
    """

    def __init__(
        self,
        n_envs: int,
        seq_len: int = 8,
        step_size: float = 0.1,
        max_steps: int = 50,
        x_limit: float = 5.0,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        obs_shape = (seq_len, 1)
        ctx_shape: Optional[Tuple[int, ...]] = (1,)
        n_actions = 3

        super().__init__(
            n_envs=n_envs,
            obs_shape=obs_shape,
            ctx_shape=ctx_shape,
            n_actions=n_actions,
        )

        self.device = device
        self.seq_len = seq_len
        self.step_size = float(step_size)
        self.max_steps = int(max_steps)
        self.x_limit = float(x_limit)

        # per-env state
        self.x = torch.zeros(n_envs, dtype=torch.float32, device=device)
        self.t = torch.zeros(n_envs, dtype=torch.long, device=device)

        # history of last seq_len states per env: (n_envs, seq_len)
        self.state_history = torch.zeros(
            (n_envs, seq_len), dtype=torch.float32, device=device
        )

    # ---------- internal helpers ----------

    def _reset_indices(self, idxs: np.ndarray) -> None:
        """Reset a subset of envs given by idxs (np array of indices)."""
        if idxs.size == 0:
            return

        # new x sampled uniformly in [-1, 1]
        new_x = torch.empty(len(idxs), device=self.device).uniform_(-1.0, 1.0)

        idxs_t = torch.as_tensor(idxs, dtype=torch.long, device=self.device)
        self.x[idxs_t] = new_x
        self.t[idxs_t] = 0

        # history = constant x for seq_len steps
        self.state_history[idxs_t] = new_x.unsqueeze(-1).repeat(1, self.seq_len)

    def _build_batch_obs(self) -> BatchObs:
        """
        Build BatchObs from current internal state.

        obs: (n_envs, seq_len, 1)
        ctx: (n_envs, 1) with normalized time
        action_mask: (n_envs, n_actions) all True
        """
        n_envs = self.n_envs

        obs_tensor = self.state_history.unsqueeze(-1)  # (n_envs, seq_len, 1)

        norm_t = (self.t.float() / max(1, self.max_steps)).unsqueeze(-1)  # (n_envs, 1)
        ctx_tensor = norm_t

        action_mask = torch.ones(
            (n_envs, self.n_actions), dtype=torch.bool, device=self.device
        )

        return BatchObs(
            obs=obs_tensor,
            ctx=ctx_tensor,
            action_mask=action_mask,
        )

    # ---------- VectorEnv interface ----------

    def reset(
        self,
        idxs: Optional[np.ndarray] = None,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> BatchObs:
        """
        Reset envs.

        idxs:
            - None  -> reset all envs
            - array -> reset only those indices
        Returns BatchObs for all envs.
        """
        if seed is not None:
            # Simple seeding hook; you can do something more elaborate if needed.
            torch.manual_seed(seed)
            np.random.seed(seed % (2**32 - 1))

        if idxs is None:
            idxs = np.arange(self.n_envs, dtype=int)
        else:
            idxs = np.asarray(idxs, dtype=int)

        self._reset_indices(idxs)
        return self._build_batch_obs()

    def step(
        self,
        actions: np.ndarray,
    ) -> Tuple[BatchObs, torch.Tensor, torch.Tensor]:
        """
        Step all envs.

        Args:
            actions: np.ndarray of shape (n_envs,) with ints in {0,1,2}.

        Returns:
            obs:    BatchObs for next states.
            reward: tensor (n_envs,)
            done:   bool tensor (n_envs,)
        """
        # map {0,1,2} -> moves {-1, 0, +1}
        actions = np.asarray(actions, dtype=int)
        assert actions.shape == (self.n_envs,), f"actions must be (n_envs,), got {actions.shape}"

        moves_np = actions - 1
        moves = torch.as_tensor(moves_np, dtype=torch.float32, device=self.device)

        # update time
        self.t += 1

        # update state
        self.x = self.x + self.step_size * moves

        # update history: shift left, append new x
        self.state_history = torch.roll(self.state_history, shifts=-1, dims=1)
        self.state_history[:, -1] = self.x

        # reward: -|x| (want x -> 0)
        rewards = -self.x.abs()

        # done if time exceeded or x leaves [-x_limit, x_limit]
        done_time = self.t >= self.max_steps
        done_bound = self.x.abs() > self.x_limit
        dones = (done_time | done_bound)

        # build next obs
        obs = self._build_batch_obs()

        return obs, rewards, dones
