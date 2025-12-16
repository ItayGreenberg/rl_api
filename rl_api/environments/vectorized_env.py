from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
from torch import Tensor
from rl_api.environments.types import BatchObs


class VectorEnv(ABC):
    """
    Vectorized environment with N parallel envs.

    Obs is a BatchObs with:
      - obs:         (n_envs, *obs_shape)
      - ctx:         (n_envs, *ctx_shape) or None
      - action_mask: (n_envs, n_actions) or None
    """

    def __init__(
        self,
        n_envs: int,
        obs_shape: tuple[int, ...],
        ctx_shape: Optional[tuple[int, ...]],
        n_actions: int,
    ) -> None:
        self.n_envs = n_envs
        self.obs_shape = obs_shape
        self.ctx_shape = ctx_shape
        self.n_actions = n_actions

    @abstractmethod
    def reset(
        self,
        idxs: Optional[np.ndarray] = None,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> BatchObs:
        """
        Reset envs.

        Args:
            idxs:
                - None  -> reset all envs, return obs for all (shape n_envs, *obs_shape)
                - array -> reset only those env indices, and return obs **for all envs**,
                           with new initial obs written at idxs.

        Returns:
            BatchObs:
                obs.obs:         (n_envs, *obs_shape)
                obs.ctx:         (n_envs, *ctx_shape) or None
                obs.action_mask: (n_envs, n_actions) or None
        """
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        actions: np.ndarray,
    ) -> Tuple[BatchObs, torch.Tensor, torch.Tensor]:
        """
        Step all envs.

        Args:
            actions: (n_envs,) or (n_envs, action_dim)

        Returns:
            obs:    BatchObs for next states.
            reward: tensor (n_envs,)
            done:   bool tensor (n_envs,)
        """
        raise NotImplementedError
