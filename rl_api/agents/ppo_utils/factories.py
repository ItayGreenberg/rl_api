from rl_api.networks.networks import ActorNetwork, CriticNetwork, PPOPolicyNetwork

from rl_api.environments.wrappers import AutoResetVectorEnv


from rl_api.agents.ppo_utils.vectorized_agent import VectorizedPPOAgent

from rl_api.agents.ppo_utils.configs import (
    DimensionConfig, PPOConfig, BufferConfig,
    TrainingConfig, EvalConfig, LoggingConfig, SavingConfig
)




from typing import Sequence, Optional
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import os
from os.path import join
import json


def build_ppo_agent(
    dims: DimensionConfig,
    ppo_cfg: PPOConfig,
    buf_cfg: BufferConfig,
    policy_net: PPOPolicyNetwork,
    train_env: AutoResetVectorEnv,
    eval_cfg: EvalConfig | None,
    eval_env: AutoResetVectorEnv | None,
    device: torch.device,
    logging_cfg: LoggingConfig | None = None,
    saving_cfg: SavingConfig | None = None,
    policy_optimizer: torch.optim.Optimizer | None = None,
) -> VectorizedPPOAgent:
    
    if policy_optimizer is None:
        policy_optimizer = Adam(policy_net.parameters(), lr=3e-4)

    if logging_cfg is None:
        logging_cfg = LoggingConfig()

    if saving_cfg is None:
        saving_cfg = SavingConfig()

    agent = VectorizedPPOAgent(
        dims=dims,
        policy_net=policy_net,
        policy_optimizer=policy_optimizer,
        train_env=train_env,
        eval_env=eval_env,
        eval_cfg=eval_cfg,
        ppo_cfg=ppo_cfg,
        buf_cfg=buf_cfg,
        logging_cfg=logging_cfg,
        saving_cfg=saving_cfg,
        device=device,
    )
    return agent

def load_agent_from_checkpoint(agent:VectorizedPPOAgent, path:str, device:torch.device):
    checkpoint = torch.load(path, map_location=device)  

    # networks
    agent.policy_net.load_state_dict(checkpoint["policy_net"])

    # optimizers
    agent.opt_policy.load_state_dict(checkpoint["policy_optimizer"])

    # update&step
    agent.update_step = checkpoint["update"] + 1
    agent.global_step = checkpoint["step"]

    # tracking params
    agent.best_reward = checkpoint["best_reward"]

    return agent









