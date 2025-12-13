# TODO

from dataclasses import dataclass
import torch.nn as nn
from typing import Optional


@dataclass(frozen=True)
class ActorConfig:
    obs_embed_dim:    int
    context_dim:      int
    action_dim:       int
    hidden:           list[int]
    dropout:          float

@dataclass(frozen=True)
class CriticConfig:
    obs_embed_dim:    int
    context_dim:      int
    hidden:           list[int]
    dropout:          float