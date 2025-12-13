# TODO

from rl_api.networks.configs import ActorConfig, CriticConfig
from rl_api.networks.networks import ActorNetwork, CriticNetwork, PPOPolicyNetwork


from typing import Optional
from torch.optim import Adam
from os.path import join



def build_actor(actor_config: ActorConfig, device):
    '''
    example:
    hidden = [256, 128]
    dropout_rate = 0.15
    '''
    

    actor_dict_cfg = {
        "obs_embed_dim": actor_config.obs_embed_dim,
        "context_dim": actor_config.context_dim,
        "action_dim": actor_config.action_dim,
        "hidden_units": actor_config.hidden,
        "dropout_rate": actor_config.dropout,
    }

    actor = ActorNetwork(**actor_dict_cfg).to(device)
    return actor, actor_dict_cfg


def build_critic(critic_config: CriticConfig, device):
    '''
    example:
    hidden = [256, 128]
    dropout_rate = 0.15
    '''
        

    critic_dict_cfg = {
        "obs_embed_dim": critic_config.obs_embed_dim,
        "context_dim": critic_config.context_dim,
        "hidden_units": critic_config.hidden,
        "dropout_rate": critic_config.dropout,
    }
    critic = CriticNetwork(**critic_dict_cfg).to(device)

    return critic, critic_dict_cfg


def build_ppo_optimizer(policy_network: PPOPolicyNetwork,
                        enc_lr = 1e-4, actor_lr = 1e-4, critic_lr = 1e-4, 
                        weight_decay=0):
    
    policy_enc_params       = list(policy_network.encoder.parameters()) 
    policy_actor_params     = list(policy_network.action_head.parameters()) 
    policy_critic_params    = list(policy_network.value_head.parameters())


    policy_optimizer = Adam([
        {"params": policy_enc_params,       "lr": enc_lr,       "weight_decay": weight_decay},
        {"params": policy_actor_params,     "lr": actor_lr,     "weight_decay": weight_decay},
        {"params": policy_critic_params,    "lr": critic_lr,    "weight_decay": weight_decay},

    ], betas=(0.9, 0.999), eps=1e-8)


    return policy_optimizer


