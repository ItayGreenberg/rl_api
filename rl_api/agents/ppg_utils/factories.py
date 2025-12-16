from rl_api.networks.networks import ActorNetwork, CriticNetwork, PPGPolicyNetwork, PPGValueNetwork

from rl_api.environments.wrappers import AutoResetVectorEnv

from rl_api.networks.networks_factory import build_ppg_optimizers

from rl_api.agents.ppg_utils.vectorized_agent import VectorizedPPGAgent

from rl_api.agents.ppg_utils.configs import (
    DimensionConfig, PPOConfig, PPGConfig, BufferConfig, EntropySchedulerConfig,
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


def build_ppg_agent(
    policy_net: PPGPolicyNetwork,
    value_net: PPGValueNetwork,
    policy_optimizer: torch.optim.Optimizer,
    aux_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    train_env: AutoResetVectorEnv,
    dims: DimensionConfig,
    ppo_cfg: PPOConfig,
    ppg_cfg: PPGConfig,
    buf_cfg: BufferConfig,
    entropy_sched_cfg: EntropySchedulerConfig,
    eval_cfg: EvalConfig,
    eval_env: AutoResetVectorEnv | None,
    logging_cfg: LoggingConfig | None = None,
    saving_cfg: SavingConfig | None = None,
    device: torch.device = torch.device("cpu"),

) -> VectorizedPPGAgent:
    
    if policy_optimizer is None or aux_optimizer is None or critic_optimizer is None:
        optimizers = build_ppg_optimizers(policy_net, value_net)
        if policy_optimizer is None:
            policy_optimizer = optimizers["policy_optimizer"]

        if aux_optimizer is None:
            aux_optimizer = optimizers["aux_optimizer"]

        if critic_optimizer is None:
            critic_optimizer = optimizers["critic_optimizer"]


    if logging_cfg is None:
        logging_cfg = LoggingConfig()

    if saving_cfg is None:
        saving_cfg = SavingConfig()

    agent = VectorizedPPGAgent(
        policy_net=policy_net,
        value_net=value_net,
        policy_optimizer=policy_optimizer,
        aux_optimizer=aux_optimizer,
        critic_optimizer=critic_optimizer,
        train_env=train_env,
        dims=dims,
        ppo_cfg=ppo_cfg,
        ppg_cfg=ppg_cfg,
        buf_cfg=buf_cfg,
        entropy_sched_cfg=entropy_sched_cfg,
        eval_cfg=eval_cfg,
        eval_env=eval_env,
        logging_cfg=logging_cfg,
        saving_cfg=saving_cfg,
        device=device,
    )
    return agent


def load_agent_from_checkpoint(agent:VectorizedPPGAgent, path:str, device:torch.device):
    checkpoint = torch.load(path, map_location=device)  

    # networks
    agent.policy_net.load_state_dict(checkpoint["policy_net"])
    agent.value_net.load_state_dict(checkpoint["value_net"])

    # optimizers
    agent.opt_policy.load_state_dict(checkpoint["policy_optimizer"])
    agent.opt_aux.load_state_dict(checkpoint["aux_optimizer"])
    agent.opt_critic.load_state_dict(checkpoint["critic_optimizer"])

    # update&step
    agent.update_step = checkpoint["update"] + 1
    agent.global_step = checkpoint["step"]

    # tracking params
    agent.best_reward = checkpoint["best_reward"]

    return agent





