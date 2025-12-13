from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class BatchObs:
    """
    Batched observation for N parallel envs.

    Attributes:
        obs:
            Tensor containing the model input.
            Shape: (n_envs, *obs_shape)
            Examples:
              - (n_envs, feature_dim)
              - (n_envs, seq_len, feature_dim)
              - (n_envs, C, H, W)

        ctx:
            Optional context features per env.
            Shape: (n_envs, *ctx_shape) or None.

        action_mask:
            Optional action mask per env.
            Shape: (n_envs, n_actions) or None.
            Convention: 1.0 = allowed, 0.0 = illegal.
    """
    obs: torch.Tensor
    ctx: Optional[torch.Tensor]
    action_mask: Optional[torch.Tensor]
