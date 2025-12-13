# rl_api/envs/wrappers.py
from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch

from rl_api.environments.vectorized_env import VectorEnv
from rl_api.environments.types import BatchObs


class AutoResetVectorEnv(VectorEnv):
    """
    Wrap a VectorEnv that supports partial reset (reset(idxs=...))
    and automatically resets any sub-env that returns done=True on step().

    After step():
      - `done[i] == True` marks the last transition of that episode.
      - The returned obs already contains the first obs of the next episode
        for those env indices.
    """

    def __init__(self, vec_env: VectorEnv):
        # delegate sizes
        super().__init__(
            n_envs=vec_env.n_envs,
            obs_shape=vec_env.obs_shape,
            ctx_shape=vec_env.ctx_shape,
            n_actions=vec_env.n_actions,
        )
        self.vec_env = vec_env

    def reset(
        self,
        idxs: Optional[np.ndarray] = None,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> BatchObs:
        """
        Reset all envs (idxs=None) or a subset of envs.

        For AutoReset usage, typical pattern is:
          - call reset(idxs=None) once at the beginning to init everything.
          - never call reset() manually again during rollout.
        """
        return self.vec_env.reset(idxs=idxs, seed=seed, options=options)

    def step(
        self,
        actions: np.ndarray,
    ) -> Tuple[BatchObs, torch.Tensor, torch.Tensor]:
        """
        Step the wrapped env, then automatically reset envs where done=True.

        Args:
            actions: array-like (n_envs,) with discrete actions.

        Returns:
            obs:   BatchObs, with obs for done envs replaced by
                   freshly reset initial obs.
            rewards: tensor (n_envs,)
            dones:   bool tensor (n_envs,)
        """
        obs, rewards, dones = self.vec_env.step(actions)

        # convert dones to something indexable
        if isinstance(dones, torch.Tensor):
            done_mask = dones.cpu().numpy().astype(bool)
        else:
            done_mask = np.asarray(dones, dtype=bool)

        to_reset = np.where(done_mask)[0]

        if to_reset.size > 0:
            # reset only those envs, fresh_obs includes all envs
            fresh_obs = self.vec_env.reset(idxs=to_reset)

            # overwrite just the done slots in the batched obs
            obs.obs[to_reset] = fresh_obs.obs[to_reset]

            if obs.ctx is not None and fresh_obs.ctx is not None:
                obs.ctx[to_reset] = fresh_obs.ctx[to_reset]

            if obs.action_mask is not None and fresh_obs.action_mask is not None:
                obs.action_mask[to_reset] = fresh_obs.action_mask[to_reset]

        return obs, rewards, dones
