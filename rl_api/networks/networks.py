import torch
from torch import Tensor    
import torch.nn as nn
from torch.distributions import Categorical

import math
from typing import List, Optional, Tuple


class ActorNetwork(nn.Module): 
    """
    Actor that receives a *legal-action mask* and sets logits of
    forbidden actions to -inf so their probability becomes 0.
    Returns the logits
    """
    NEG_INF = -1e9          # used to zero-out illegal logits

    def __init__(
            
            self,
            obs_embed_dim: int,
            context_dim: int,
            action_dim:  int,     
            hidden_units: list[int] = [256,128],
            dropout_rate: float = 0.1
        ):
        super().__init__()

        self.context_dim = context_dim

        self.obs_ln = nn.LayerNorm(obs_embed_dim)
        self.ctx_ln = nn.LayerNorm(context_dim)

        layers = []
        in_dim = obs_embed_dim + context_dim
        for h in hidden_units:
            layers += [ nn.Linear(in_dim, h),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate) ]
            in_dim = h
        layers += [ nn.Linear(in_dim, action_dim) ]
        self.net = nn.Sequential(*layers)


    def forward(self, obs_emb: Tensor, ctx: Tensor, mask: Tensor, tau=1.0) -> torch.Tensor:
        """
        obs_emb : (B, H)
        ctx     : (B, C)
        mask    : (B, A)  bool / 0-1   1 = legal, 0 = forbidden
        tau     : decide how greedly we sample actions
        """
        legal_per_sample = mask.any(dim=-1)          # shape: (batch,), True if at least one legal action
        illegal_samples  = (~legal_per_sample).nonzero(as_tuple=True)[0]  # indices of bad samples

        if illegal_samples.numel() > 0:
            # print the sample indices…
            print(f"No legal actions for samples at indices: {illegal_samples.tolist()}")
            # …and print out their corresponding mask rows
            for idx in illegal_samples.tolist():
                print(f" mask[{idx}] = {mask[idx]}")
            # then fail with a clear message
            raise AssertionError(
                f"At least one action must be legal per sample, but {illegal_samples.numel()} sample(s) have none."
            )
        

        assert (mask.any(dim=-1)).all(), "At least one action must be legal per sample."

        logits = self.net(
            torch.cat([self.obs_ln(obs_emb), self.ctx_ln(ctx)], dim=-1) # temp change
        )
        logits = logits / tau
        logits = logits + (~mask).float() * self.NEG_INF
        return logits


class CriticNetwork(nn.Module):
    
    def __init__(self, obs_embed_dim, context_dim, hidden_units=[256,128], dropout_rate=0.1):
        super().__init__()
        assert obs_embed_dim is not None, "obs_embed_dim is None"
        assert context_dim is not None, "context_dim is None"
        assert hidden_units is not None, "critic_hidden is None"

        self.obs_ln = nn.LayerNorm(obs_embed_dim)
        self.ctx_ln = nn.LayerNorm(context_dim)
        layers = []
        in_dim = obs_embed_dim + context_dim
        for h in hidden_units:
            layers += [ nn.Linear(in_dim, h),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate) ]
            in_dim = h
        layers += [ nn.Linear(in_dim, 1) ]
        self.net = nn.Sequential(*layers)


    def forward(self, obs_emb, ctx):
        x = torch.cat([self.obs_ln(obs_emb), self.ctx_ln(ctx)], dim=-1)
        return self.net(x).squeeze(-1)  # (B,)


class PPOPolicyNetwork(nn.Module):
    """
    Generic PPO policy:

        encoder  -> obs_emb (B, H)
          ├─> ActorNetwork  (logits over actions)
          └─> CriticNetwork (state value)

    Contract:
      - encoder(obs) must return a 2D tensor of shape (B, obs_embed_dim).
      - ctx must have shape (B, context_dim).
      - mask must have shape (B, action_dim).
    """
    def __init__(
        self,
        encoder: nn.Module,
        action_head: nn.Module,
        value_head: nn.Module,
    ):
        super().__init__()
        self.encoder        = encoder          # e.g. Transformer/LSTM
        self.action_head    = action_head      # existing ActorNetwork
        self.value_head     = value_head       # existing CriticNetwork


    def forward(self, obs: Tensor, ctx: Tensor, masks: Tensor, *args):
        """
        obs    : example: (B, seq_len, num_feats), (B, num_features)
        ctx    : (B, context_dim)
        masks  : (B, action_dim) bool
        """

        # obs emb
        z = self.encoder(obs)                      # (B, obs_embed_dim)

        # actor
        logits = self.action_head(z, ctx, masks)    # (B, action_dim)
        dist   = Categorical(logits=logits)

        # critic
        values = self.value_head(z, ctx)            # (B,)

        return dist, values


