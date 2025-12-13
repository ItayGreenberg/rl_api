import torch
from torch import Tensor
from typing import Tuple

from rl_api.test_utils.tests import assert_finite


class RolloutBufferVec:
    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple[int, ...],   # shape of one observation
        context_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            buffer_size: number of timesteps to collect per rollout
            obs_shape: shape of one observation (e.g. (seq_len, num_features) or (feat_dim,))
            context_dim: dimension of the context vector
            action_dim: number of discrete actions
            gamma: discount factor
            gae_lambda: GAE smoothing parameter
            device: where to store the tensors
        """
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.context_dim = context_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lam = gae_lambda
        self.device = device

        self.num_envs = None
        self.pos = 0

        # main storage (all on self.device)
        self.obs          = torch.empty((buffer_size, *obs_shape),  dtype=torch.float32, device=device)
        self.context      = torch.empty((buffer_size, context_dim), dtype=torch.float32, device=device)
        self.actions      = torch.empty(buffer_size,                dtype=torch.long,    device=device)
        self.logps        = torch.empty(buffer_size,                dtype=torch.float32, device=device)
        self.values       = torch.empty(buffer_size,                dtype=torch.float32, device=device)
        self.rewards      = torch.empty(buffer_size,                dtype=torch.float32, device=device)
        self.dones        = torch.empty(buffer_size,                dtype=torch.bool,    device=device)
        self.action_masks = torch.empty((buffer_size, action_dim),  dtype=torch.bool,    device=device)
        self.env_ids      = torch.empty(buffer_size,                dtype=torch.long,    device=device)

        # Will be filled after compute
        self.advantages = torch.empty(buffer_size, dtype=torch.float32, device=device)
        self.returns    = torch.empty(buffer_size, dtype=torch.float32, device=device)

    def add(
        self,
        obs: Tensor,          # shape: obs_shape
        context: Tensor,      # shape: (context_dim,)
        action,               # int or 0-dim tensor
        logp,                 # float or 0-dim tensor
        value,                # float or 0-dim tensor
        reward,               # float or 0-dim tensor
        done,                 # bool or 0-dim bool tensor
        action_mask: Tensor,  # shape: (action_dim,)
        env_id: int,
    ):
        """
        Add a single transition (one env, one timestep).

        All inputs can be torch tensors (on any device) or Python scalars
        where appropriate. They are converted to the buffer device/dtype.
        """
        # --- shape/type checks ---
        if tuple(obs.shape) != self.obs_shape:
            raise ValueError(f"obs must be {self.obs_shape}, got {tuple(obs.shape)}")

        if context.shape != (self.context_dim,):
            raise ValueError(f"context must be {(self.context_dim,)}, got {tuple(context.shape)}")

        if action_mask.shape != (self.action_dim,):
            raise ValueError(f"action_mask must be {(self.action_dim,)}, got {tuple(action_mask.shape)}")

        if self.pos >= self.buffer_size:
            raise RuntimeError(f"buffer overflow: pos {self.pos}, cap {self.buffer_size}")

        i = self.pos

        # move everything to buffer device
        self.obs[i].copy_(obs.to(self.device, dtype=torch.float32))
        self.context[i].copy_(context.to(self.device, dtype=torch.float32))

        # scalars: accept Python or 0-dim tensors
        self.actions[i] = int(action) if not isinstance(action, Tensor) else int(action.item())
        self.logps[i]   = float(logp)  if not isinstance(logp,   Tensor) else float(logp.item())
        self.values[i]  = float(value) if not isinstance(value,  Tensor) else float(value.item())
        self.rewards[i] = float(reward)if not isinstance(reward, Tensor) else float(reward.item())

        if isinstance(done, Tensor):
            self.dones[i] = bool(done.item())
        else:
            self.dones[i] = bool(done)

        self.action_masks[i].copy_(action_mask.to(self.device, dtype=torch.bool))
        self.env_ids[i] = int(env_id)

        self.pos += 1

        if not self.action_masks[i].any():
            raise RuntimeError(
                f"RolloutBufferVec.add: all-False action_mask (env_id={env_id}, pos={i})"
            )

    def add_batch(
        self,
        obs: Tensor,          # (n_envs, *obs_shape)
        ctx: Tensor,          # (n_envs, context_dim)
        action_mask: Tensor,  # (n_envs, action_dim)
        actions: Tensor,      # (n_envs,)
        logps: Tensor,        # (n_envs,)
        values: Tensor,       # (n_envs,)
        rewards: Tensor,      # (n_envs,)
        dones: Tensor,        # (n_envs,) bool
    ):
        """
        Add a batch of transitions for all envs at a single time step.

        All inputs are torch tensors. They can be on any device; they will be
        moved to the buffer device as needed.
        """
        # ensure 1D where expected
        obs      = obs.detach()
        ctx      = ctx.detach()
        action_mask = action_mask.detach()
        actions  = actions.detach()
        logps    = logps.detach()
        values   = values.detach()
        rewards  = rewards.detach()
        dones    = dones.detach()

        if self.num_envs is None:
            self.num_envs = actions.shape[0]

        n_envs = self.num_envs
        assert obs.shape[0] == n_envs,        f"obs first dim {obs.shape[0]} != num_envs {n_envs}"
        assert ctx.shape[0] == n_envs,        f"ctx first dim {ctx.shape[0]} != num_envs {n_envs}"
        assert action_mask.shape[0] == n_envs,f"action_mask first dim {action_mask.shape[0]} != num_envs {n_envs}"

        for env_i in range(n_envs):
            self.add(
                obs         = obs[env_i],
                context     = ctx[env_i],
                action      = actions[env_i],
                logp        = logps[env_i],
                value       = values[env_i],
                reward      = rewards[env_i],
                done        = dones[env_i],
                action_mask = action_mask[env_i],
                env_id      = env_i,
            )

    def compute_returns_and_advantages(self, last_values: Tensor):
        """
        last_values : tensor of shape (num_envs,)
            V(s_T) for each parallel env, on any device (will be moved).
        """
        device  = self.device
        last_values = last_values.to(device=device, dtype=torch.float32)

        T       = self.pos                       # how many rows are filled
        rewards = self.rewards[:T]               # (T,) already on device
        values  = self.values[:T]                # (T,)
        dones   = self.dones[:T].float()         # (T,)
        env_ids = self.env_ids[:T].long()        # (T,)

        T = len(rewards)
        advantages_raw = torch.zeros(T, device=device)

        # --- 1) exact GAE, per environment --------------------------------
        for env in range(self.num_envs):
            idxs = (env_ids == env).nonzero(as_tuple=False).flatten()
            if len(idxs) == 0:
                continue

            v_boot = last_values[env].reshape(1)          # shape (1,)
            v_path = torch.cat([values[idxs], v_boot])    # len = len(idxs)+1
            r_path = rewards[idxs]
            d_path = dones[idxs]

            gae = 0.0
            for k in reversed(range(len(idxs))):
                non_terminal = 1.0 - d_path[k]
                delta = r_path[k] + self.gamma * v_path[k + 1] * non_terminal - v_path[k]
                gae   = delta + self.gamma * self.lam * non_terminal * gae
                advantages_raw[idxs[k]] = gae

        # --- 2) critic targets – keep true scale --------------------------
        self.returns[:T].copy_(advantages_raw + values)

        # ---------- clamp & fix non-finite --------------------------------
        advantages_raw[~torch.isfinite(advantages_raw)] = 0.0
        self.returns   [~torch.isfinite(self.returns  )] = 0.0

        assert_finite("returns / advantages_raw", self.returns[:T], advantages_raw)

        # diagnostics BEFORE global z-score
        adv_mean  = advantages_raw.mean().item()
        adv_std   = advantages_raw.std(unbiased=False).item()

        valid_rets = self.returns[:T]
        rets_mean  = valid_rets.mean().item()
        rets_std   = valid_rets.std(unbiased=False).item()
        stats = {
            "adv_mean":  adv_mean,
            "adv_std":   adv_std,
            "rets_mean": rets_mean,
            "rets_std":  rets_std,
        }

        # normalize returns
        self.returns = (self.returns - self.returns.mean()) / (self.returns.std() + 1e-8)

        # --- 3) final normalisation for the policy update -----------------
        self.advantages.copy_(
            (advantages_raw - advantages_raw.mean())
            / (advantages_raw.std(unbiased=False) + 1e-8)
        )

        return stats

    def get_batches(self, batch_size: int):
        """
        Yield random mini-batches for the PPO update.
        Assumes the first `self.pos` rows are filled.
        """
        T = self.pos
        perm = torch.randperm(T, device=self.device)

        # tensor views (no copy)
        obs          = self.obs[:T].detach()
        ctx          = self.context[:T].detach()
        actions      = self.actions[:T].detach()
        old_logps    = self.logps[:T].detach()
        old_vals     = self.values[:T].detach()
        returns      = self.returns[:T].detach()
        advs         = self.advantages[:T].detach()
        action_masks = self.action_masks[:T].detach()

        for start in range(0, T, batch_size):
            idx = perm[start : start + batch_size]
            yield {
                "obs":          obs[idx],
                "ctx":          ctx[idx],
                "actions":      actions[idx],
                "old_logps":    old_logps[idx],
                "old_values":   old_vals[idx],
                "returns":      returns[idx],
                "advantages":   advs[idx],
                "action_masks": action_masks[idx],
            }

    def clear(self):
        self.pos = 0
        self.advantages.zero_()
        self.returns.zero_()
