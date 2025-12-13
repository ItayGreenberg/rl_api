from rl_api.networks.networks import (
    ActorNetwork, CriticNetwork, PPOPolicyNetwork
    )

from rl_api.agents.ppo_utils.buffer import RolloutBufferVec

from rl_api.agents.ppo_utils.configs import (
    DimensionConfig, PPOConfig, BufferConfig,
    TrainingConfig, EvalConfig, LoggingConfig, SavingConfig,
    )

from rl_api.agents.ppo_utils.logger import Logger

from rl_api.environments.types import BatchObs
from rl_api.environments.wrappers import AutoResetVectorEnv

from rl_api.schedulers.scheduler import LinearScheduler

from rl_api.test_utils.tests import assert_finite

import torch
from torch import Tensor
from torch import nn
from torch.distributions import Categorical # Categorical is softmax
import torch.nn.functional as F

from typing import Optional, List
import os
import numpy as np
import math



class VectorizedPPOAgent:
    def __init__(
        self,
        dims: DimensionConfig,
        policy_net: PPOPolicyNetwork,          # encoder + action head + value head
        policy_optimizer: torch.optim.Optimizer,
        train_env: AutoResetVectorEnv,
        ppo_cfg: PPOConfig,
        buf_cfg: BufferConfig,
        eval_cfg: EvalConfig = None,
        eval_env: Optional[AutoResetVectorEnv] = None,
        logging_cfg: Optional[LoggingConfig] = None,
        saving_cfg:  Optional[SavingConfig] = None,
        device: torch.device = torch.device("cpu"),
    ):
        

        # — dims / shapes —
        self.obs_shape = dims.obs_shape
        self.action_dim, self.context_dim = dims.action_dim, dims.context_dim


        # — networks —
        self.policy_net = policy_net
        self.device = device

        # --- Optimizers ---
        self.opt_policy = policy_optimizer

        # — envs —
        self.train_env = train_env
        self.eval_env = eval_env
        self.obs: BatchObs = None

        # — configs —
        self.ppo_cfg = ppo_cfg
        self.buf_cfg = buf_cfg
        self.eval_cfg = eval_cfg
        self.logging_cfg = logging_cfg
        self.saving_cfg  = saving_cfg




        # --- Setup logging (console / TensorBoard / HTML) ---
        self._setup_logging()

        # --- Setup saving ---
        self._setup_saving()
        
        # --- PPO Config Unpack ---
        self._unpack_ppo_cfg()

        # --- entropy schedule ---
        self._setup_entropy_sched()

        # --- rollout buffers ---
        self._setup_buffers()

        # --- setup tracking ---
        self._setup_tracking()

        # --- validate ---
        self._validate_init()

    def _setup_logging(self):
        self.list_stats, self.int_stats = {}, {}
        self._reset_sample_epoch_stats()
        self.logger = None
        
        lc = self.logging_cfg
        if lc:
            self.logger = Logger(lc, lc.param_groups_names)

    def _setup_saving(self):
        sc = self.saving_cfg
        self.save_agent_path = None
        if not sc: return
        self.save_agent_path = sc.save_agent_path
        if self.save_agent_path:
            self.checkpoint_path = os.path.join(self.save_agent_path, "checkpoints")
            self.best_agent_path = os.path.join(self.save_agent_path, "best_agents")
            os.makedirs(self.checkpoint_path, exist_ok=True)
            os.makedirs(self.best_agent_path, exist_ok=True)

    def _unpack_ppo_cfg(self):
        pc = self.ppo_cfg
        self.epochs     = pc.num_epochs
        self.clip_eps   = pc.clip_eps
        self.clip_vf    = pc.clip_vf
        self.vf_coef    = pc.vf_coef
        self.gamma      = pc.gamma
        self.gae_lambda = pc.gae_lambda
        self.target_kl  = pc.target_kl
        self.grad_clip  = pc.grad_clip
        self.entropy_coef = pc.entropy_coef_start

    def _setup_entropy_sched(self):
        pc = self.ppo_cfg
        self.entropy_scheduler = None
        if pc.entropy_coef_end is not None and pc.entropy_decay_start_step < pc.entropy_decay_end_step:
            self.entropy_scheduler = LinearScheduler(
                start_step=pc.entropy_decay_start_step,
                end_step=pc.entropy_decay_end_step,
                start_value=pc.entropy_coef_start,
                end_value=pc.entropy_coef_end
            )

    def _setup_buffers(self):
        buf = self.buf_cfg
        self.batch_size = buf.batch_size
        # PPO buffer on GPU (fast policy phase)
        self.ppo_buf = RolloutBufferVec(
            buffer_size = buf.buffer_size,
            obs_shape   = self.obs_shape,
            context_dim = self.context_dim,
            action_dim  = self.action_dim,
            gamma       = self.gamma,
            gae_lambda  = self.gae_lambda,
            device      = self.device,
        )

    def _setup_tracking(self):
        self.global_step = 0
        self.update_step = 1

        
        self.best_reward = None
        self._last_best_update = -1

    def _validate_init(self):
        # Shapes
        for cur_obs_shape in self.obs_shape:
            assert cur_obs_shape > 0
        assert self.action_dim > 0
        # Paths
        if self.saving_cfg and self.save_agent_path:
            assert os.path.isdir(self.save_agent_path)


    def _reset_sample_epoch_stats(self):  
        self.list_stats = dict(
            policy_loss          = [], value_loss    = [], entropy         = [],
            
            clip_frac            = [], approx_kl     = [], explained_var   = [],

            enc_grad             = [], actor_grad    = [],    critic_grad  = [],
            
            adv_mean             = [], adv_std         = [],
            rets_mean            = [], rets_std        = [],



        )

        self.int_stats = dict(
            reward_sum           = 0,

            batches_used_count   = 0,

            total_steps = 0
        )


    def _gather_sample_stats(self) -> dict:

        keys = self.list_stats.keys()
        for key in keys:
            if len(self.list_stats[key]) == 0:
                raise ValueError(f"error in gathering sample stats. [{key}] is an empty list")

        info_dict = {}
        for key in keys:
            info_dict[key] = np.mean(self.list_stats[key])


        num_batches_per_epoch = (self.buf_cfg.buffer_size + self.buf_cfg.batch_size - 1) // self.buf_cfg.batch_size
        num_batches = num_batches_per_epoch * self.ppo_cfg.num_epochs * self.logging_cfg.log_interval

        extra_info = {
            "entropy_coef":         self.entropy_coef,
            "reward_sum":           self.int_stats["reward_sum"],
            "reward_per_step":      self.int_stats["reward_sum"] / self.int_stats["total_steps"],
            "batches_used_fraction": self.int_stats["batches_used_count"] / num_batches
        }

        return {
            **info_dict,
            **extra_info,
        }


    def _grad_l2(self, params:Tensor):
        """L2-norm of gradients in a module (returns 0 if all grads are None)."""
        sq_sum = 0.0
        for p in params:
            if p.grad is not None:
                sq_sum += p.grad.pow(2).sum().item()
        return math.sqrt(sq_sum)


    def calculate_value_loss(self, old_values: Tensor, values: Tensor, returns: Tensor):
        clip_vf = self.ppo_cfg.clip_vf
        vf_loss_clip = self.ppo_cfg.vf_loss_clip

        if clip_vf:
            delta = (values - old_values).clamp(-clip_vf, +clip_vf)
            value_loss = torch.max(
                (values - returns).pow(2),
                (old_values + delta - returns).pow(2)
            )
        else:
            value_loss = (values - returns).pow(2)

        if vf_loss_clip:
            value_loss = torch.clamp(value_loss, 0, vf_loss_clip)

        loss_critic = self.vf_coef * value_loss.mean()

        return loss_critic


    @torch.no_grad()
    def collect_rollouts(self, vec_env: AutoResetVectorEnv, obs: BatchObs):
        """
        Fills the buffer once (buffer_size // n_envs steps).
        Returns the next BatchObs.
        """
        n_envs = vec_env.n_envs
        assert self.ppo_buf.buffer_size % n_envs == 0, "buffer_size must divide n_envs"
        horizon = self.ppo_buf.buffer_size // n_envs

        last_dones = None  # will hold dones from final step

        for step in range(horizon):
            # -------------- tensors on device -----------------
            # obs.obs: (n_envs, *obs_shape) on CPU -> move to device for policy
            obs_tensor = obs.obs.to(self.device, dtype=torch.float32)

            if obs.ctx is None:
                raise ValueError("ctx is None but the policy expects a context tensor.")
            ctx_tensor = obs.ctx.to(self.device, dtype=torch.float32)

            if obs.action_mask is None:
                raise ValueError("action_mask is None but the policy expects a mask.")
            mask_tensor = obs.action_mask.to(self.device, dtype=torch.bool)

            # forward through POLICY network
            dist, values = self.policy_net(obs_tensor, ctx_tensor, mask_tensor)
            dist: Categorical
            values: torch.Tensor  # (n_envs,)

            actions = dist.sample()              # (n_envs,)
            logps   = dist.log_prob(actions)     # (n_envs,)

            # ------------ env step (torch) --------------------
            # move actions to CPU if env expects CPU tensors
            actions_cpu = actions.detach().cpu()
            next_obs, rewards, dones = vec_env.step(actions_cpu)   # rewards, dones: tensors

            # -------------- store in buffer -------------------
            # buffer will move things to its own device; we keep inputs on CPU here
            self.ppo_buf.add_batch(
                obs         = obs.obs,            # (n_envs, *obs_shape), torch
                ctx         = obs.ctx,            # (n_envs, *ctx_shape), torch
                action_mask = obs.action_mask,    # (n_envs, n_actions), torch
                actions     = actions_cpu,        # (n_envs,), torch
                logps       = logps.detach().cpu(),   # (n_envs,), torch
                values      = values.detach().cpu(),  # (n_envs,), torch
                rewards     = rewards.detach().cpu(), # (n_envs,), torch
                dones       = dones.detach().cpu(),   # (n_envs,), bool torch
            )

            obs = next_obs
            last_dones = dones.detach().cpu()    # keep last dones for bootstrap masking
            self.global_step += n_envs


            # ------------- stats ----------------------
            self.int_stats["reward_sum"]     += float(torch.sum(rewards))
            self.int_stats["total_steps"]    += n_envs


        # ------------- bootstrap value ------------------------
        obs_tensor = obs.obs.to(self.device, dtype=torch.float32)
        ctx_tensor = obs.ctx.to(self.device, dtype=torch.float32)
        mask_tensor = obs.action_mask.to(self.device, dtype=torch.bool)

        _, last_vals = self.policy_net(obs_tensor, ctx_tensor, mask_tensor)
        last_vals: torch.Tensor = last_vals.detach().cpu()   # (n_envs,)

        # if env was done on last step, bootstrap value is 0 for that env
        if last_dones is None:
            raise RuntimeError("last_dones is None; rollout loop did not run?")
        last_vals[last_dones] = 0.0

        buf_stats = self.ppo_buf.compute_returns_and_advantages(last_vals)
        self.list_stats["adv_mean"].append(buf_stats["adv_mean"])
        self.list_stats["adv_std"].append(buf_stats["adv_std"])
        self.list_stats["rets_mean"].append(buf_stats["rets_mean"])
        self.list_stats["rets_std"].append(buf_stats["rets_std"])

        return obs  # BatchObs
    
    
    def _batch_step(self, batch) -> dict:
        clip_eps = self.clip_eps
        ent_coef = self.entropy_coef
        device = self.device
        # ---------- to device -------------------------
        obs         = batch["obs"].to(device)
        ctxs        = batch["ctx"].to(device)
        actions     = batch["actions"].to(device)
        old_lp      = batch["old_logps"].to(device)
        old_vals    = batch["old_values"].to(device)
        returns     = batch["returns"].to(device)
        advs        = batch["advantages"].to(device)
        masks       = batch["action_masks"].to(device)       


        # ----------- policy network forward ------------------
        dist, vals = self.policy_net(obs, ctxs, masks)
        dist: Categorical
        vals: torch.Tensor
        # ------------ actor loss -------------------
        new_lp  = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio :torch.Tensor   = (new_lp - old_lp).exp()
        surr1   = ratio * advs
        surr2   = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advs
        policy_loss  = -torch.min(surr1, surr2).mean()
        entropy_loss = -ent_coef * entropy

        # ---------- critic loss ----------                
        critic_loss = self.calculate_value_loss(old_values= old_vals, values=vals, returns=returns)

        loss: torch.Tensor = policy_loss + entropy_loss + critic_loss


        # --------------- backward ---------------------
        # 1. Zero gradients
        
        self.opt_policy.zero_grad(set_to_none=True)
        

        # 2. Backward pass
        loss.backward()
        

        # 3. Clip gradients (in-place)
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)

        # 4. Optimizer step
        self.opt_policy.step()

        # --------------- diagnostics ---------------

        # grads
        enc_grad    = self._grad_l2(self.policy_net.encoder.parameters())
        actor_grad  = self._grad_l2(self.policy_net.action_head.parameters())
        critic_grad = self._grad_l2(self.policy_net.value_head.parameters())

        # actor stats
        with torch.no_grad():
            approx_kl_tensor = (old_lp - new_lp).mean()
            clip_frac = (ratio - 1).abs().gt(clip_eps).float().mean()

        # critic stats
        with torch.no_grad():
            var_y   = returns.var(unbiased=False)
            if var_y < 1e-6:                               # --- FIX
                expl_var = torch.tensor(0.0, device=self.device)
            else:
                var_e = (returns - vals).var(unbiased=False)
                expl_var = 1 - var_e / var_y

        
        policy_stats = {
            "policy_loss": policy_loss.item(),
            "entropy":     entropy.item(),  
            "approx_kl":   approx_kl_tensor.item(),
            "clip_frac":   clip_frac.item(),
            "actor_grad":  actor_grad,
        }
        critic_stats = {
            "value_loss":  critic_loss.item(),
            "expl_var":    expl_var.item(),
            "critic_grad": critic_grad,
        }
        enc_stats   = {
            "enc_grad": enc_grad
        }
        return {
            "policy_stats": policy_stats,
            "critic_stats": critic_stats,
            "enc_stats": enc_stats,
        }


    def update(self):

        epochs   = self.epochs

        for epoch in range(epochs):
            early_stop = False
            for batch in self.ppo_buf.get_batches(self.batch_size):
                stats = self._batch_step(batch)

                pol_stats   = stats["policy_stats"]
                crit_stats  = stats["critic_stats"]
                enc_stats   = stats["enc_stats"]


                if self.logging_cfg and self.logging_cfg.log_interval:
                    # policy stats
                    self.list_stats["policy_loss"].append(pol_stats["policy_loss"])
                    self.list_stats["entropy"].append(pol_stats["entropy"])
                    self.list_stats["approx_kl"].append(pol_stats["approx_kl"])
                    self.list_stats["clip_frac"].append(pol_stats["clip_frac"])
                    self.list_stats["actor_grad"].append(pol_stats["actor_grad"])
                    self.int_stats["batches_used_count"] += 1

                    # critic stats
                    self.list_stats["value_loss"].append(crit_stats["value_loss"])
                    self.list_stats["explained_var"].append(crit_stats["expl_var"])
                    self.list_stats["critic_grad"].append(crit_stats["critic_grad"])

                    # enc stats
                    self.list_stats["enc_grad"].append(enc_stats["enc_grad"])



                if self.target_kl is not None and pol_stats["approx_kl"] > self.target_kl:
                    early_stop = True
                    break   # early-stop this epoch


            if early_stop:
                break


    def save(self, path: str, file_name: str) -> None:
        """
        Save the full agent to a file.  
        If `update` is provided, appends it to the filename.
        """
        full_path = os.path.join(path, file_name)

        data = {
            # networks
            "policy_net": self.policy_net.state_dict(),

            # optimizers
            "policy_optimizer": self.opt_policy.state_dict(),

            # update&step
            "update": self.update_step,
            "step": self.global_step,

            # tracking params
            "best_reward": self.best_reward,

        }
        torch.save(data, full_path)
    

    def train(self, train_cfg: TrainingConfig):

        total_updates = train_cfg.total_updates
        initial_update = self.update_step
        while self.update_step <= initial_update + total_updates:
            self.train_one_update(update=self.update_step) # collect_rollouts + update


    def train_one_update(self, update=None):
        if self.obs is None:
            self.obs = self.train_env.reset()
    
        if self.entropy_scheduler is not None:
            self.entropy_coef = self.entropy_scheduler.step(step=update)

        # 0) switch to train mode
        self.policy_net.train()


        # 1) Collect rollout & compute returns
        self.obs = self.collect_rollouts(self.train_env, self.obs) 

        # 2) PPO policy phase        
        self.update()

        # 3) reset ppo_buf
        self.ppo_buf.clear()


        if update is not None:
            self.handle_logs(update)
            self.handle_saving(update)
        
        self.update_step += 1


    def handle_logs(self, update: int): # change .eval_method, n_steps should be chosen by user
        lc = self.logging_cfg
        ec = self.eval_cfg
        if not lc:
            return
        
        # 1) sample‐epoch logging
        if lc.log_interval and update % lc.log_interval == 0:
            stats = self._gather_sample_stats()

            self.logger.log_sample(update, self.global_step, stats)
            self._reset_sample_epoch_stats()

        # 2) periodic greedy evaluation logging
        if ec is not None and self.eval_env and lc.eval_interval and update % lc.eval_interval == 0:
            # run eval
            epsilon = 0.0
            if ec.eval_method == "sample":
                epsilon = 1.0
                    
            stats = self.evaluate(self.eval_env, n_steps=ec.n_steps, epsilon=epsilon)

            # log reward
            sum_reward      = stats["sum_reward"]
            reward_per_step = stats["sum_reward"] / stats["total_steps"]
            is_best = (self.best_reward is None or sum_reward > self.best_reward)

            # now use our logger:
            self.logger.log_eval(
                update,
                sum_reward,
                reward_per_step,
                is_best
            )
            

            # update best_reward (saving happens elsewhere)
            if is_best:
                self.best_reward = sum_reward
                self._last_best_update = update


    def handle_saving(self, update: int):
        sc = self.saving_cfg
        if not sc:
            return
        
        # 1) checkpoints
        if sc.save_interval and update % sc.save_interval == 0:
            self.save(self.checkpoint_path, f"agent{update}.pt")

        # 2) best‐agent dump
        # only save the best agent exactly once, at the flagged update
        if self._last_best_update == update and sc.save_agent_path:
            self.save(self.best_agent_path, f"agent{update}.pt")


    @torch.no_grad()
    def evaluate(self, eval_env:AutoResetVectorEnv, n_steps: int = 10000, epsilon: float = 0.3) -> dict:
        """
        Roll out the current policy in `eval_env` for `n_episodes` and return the
        mean episode-return (reward).

        • Actions are greedy (arg-max) with ε probability of random sampling.
        • If `self.rule_based` is True the rule-based override is applied **after**
        the policy proposes an action.
        """
        self.policy_net.eval()

        n_envs = eval_env.n_envs
        steps_needed = n_steps // n_envs

        stats = {
            "sum_reward": 0,
            "total_steps": steps_needed * n_envs,
        }
        obs: BatchObs = eval_env.reset()                   
        for i in range(steps_needed):

            # ---------- build 1-step batch on device ----------------------
            obs_tensor = torch.as_tensor(obs.obs, dtype=torch.float32, device=self.device)

            if obs.ctx is None:
                raise ValueError("ctx is None but the policy expects a context tensor.")
            ctx_tensor = torch.as_tensor(obs.ctx, dtype=torch.float32, device=self.device)

            if obs.action_mask is None:
                raise ValueError("action_mask is None but the policy expects a mask.")
            mask_tensor = torch.as_tensor(obs.action_mask, dtype=torch.bool, device=self.device)

           
            # ---------- policy forward -----------------------------------
            dist, values = self.policy_net(obs_tensor, ctx_tensor, mask_tensor)
            dist: Categorical

            if np.random.rand() < epsilon:
                actions = dist.sample()           # (1,)
            else:
                actions = dist.probs.argmax(dim=-1)  # (1,)


            # ---------- env step -----------------------------------------
            action_np = actions.cpu().numpy()
            next_obs, rewards, dones = eval_env.step(action_np)

            sum_rewards = float(torch.sum(rewards))
            stats["sum_reward"] += sum_rewards
            # ---------- unwrap next observation --------------------------
            obs = next_obs

        
        return stats




