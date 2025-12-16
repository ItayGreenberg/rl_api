import torch
from torch import Tensor
from typing import Tuple, Dict, Iterator, Any

from rl_api.test_utils.tests import assert_finite


class RolloutBufferVec:
    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple[int, ...],   # shape of one observation (seq or vector)
        context_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: torch.device = torch.device("cpu"),
        store_old_logits: bool = True,
        normalize_returns: bool = True,
    ):
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.context_dim = context_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lam = gae_lambda
        self.device = device

        self.store_old_logits = store_old_logits
        self.normalize_returns = normalize_returns

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

        # PPG-specific: store behavior policy logits for KL in aux phase
        self.old_logits = None
        if self.store_old_logits:
            self.old_logits = torch.empty((buffer_size, action_dim), dtype=torch.float32, device=device)

        # Will be filled after compute
        self.advantages = torch.empty(buffer_size, dtype=torch.float32, device=device)
        self.returns    = torch.empty(buffer_size, dtype=torch.float32, device=device)

    def add(
        self,
        obs: Tensor,          # obs_shape
        context: Tensor,      # (context_dim,)
        action,               # int or 0-dim tensor
        logp,                 # float or 0-dim tensor
        value,                # float or 0-dim tensor
        reward,               # float or 0-dim tensor
        done,                 # bool or 0-dim bool tensor
        action_mask: Tensor,  # (action_dim,)
        env_id: int,
        logits: Tensor | None = None,  # (action_dim,) optional
    ):
        # --- shape/type checks ---
        if tuple(obs.shape) != self.obs_shape:
            raise ValueError(f"obs must be {self.obs_shape}, got {tuple(obs.shape)}")

        if tuple(context.shape) != (self.context_dim,):
            raise ValueError(f"context must be {(self.context_dim,)}, got {tuple(context.shape)}")

        if tuple(action_mask.shape) != (self.action_dim,):
            raise ValueError(f"action_mask must be {(self.action_dim,)}, got {tuple(action_mask.shape)}")

        if self.pos >= self.buffer_size:
            raise RuntimeError(f"buffer overflow: pos {self.pos}, cap {self.buffer_size}")

        if self.store_old_logits:
            if logits is None:
                raise ValueError("logits is required when store_old_logits=True")
            if tuple(logits.shape) != (self.action_dim,):
                raise ValueError(f"logits must be {(self.action_dim,)}, got {tuple(logits.shape)}")

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

        if self.store_old_logits:
            self.old_logits[i].copy_(logits.to(self.device, dtype=torch.float32))

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
        logits: Tensor | None = None,  # (n_envs, action_dim) optional
    ):
        obs = obs.detach()
        ctx = ctx.detach()
        action_mask = action_mask.detach()
        actions = actions.detach()
        logps = logps.detach()
        values = values.detach()
        rewards = rewards.detach()
        dones = dones.detach()
        if logits is not None:
            logits = logits.detach()

        if self.num_envs is None:
            self.num_envs = actions.shape[0]

        n_envs = self.num_envs
        assert obs.shape[0] == n_envs,         f"obs first dim {obs.shape[0]} != num_envs {n_envs}"
        assert ctx.shape[0] == n_envs,         f"ctx first dim {ctx.shape[0]} != num_envs {n_envs}"
        assert action_mask.shape[0] == n_envs, f"action_mask first dim {action_mask.shape[0]} != num_envs {n_envs}"

        if self.store_old_logits:
            if logits is None:
                raise ValueError("logits is required when store_old_logits=True")
            assert logits.shape[0] == n_envs, f"logits first dim {logits.shape[0]} != num_envs {n_envs}"

        for env_i in range(n_envs):
            self.add(
                obs=obs[env_i],
                context=ctx[env_i],
                action=actions[env_i],
                logp=logps[env_i],
                value=values[env_i],
                reward=rewards[env_i],
                done=dones[env_i],
                action_mask=action_mask[env_i],
                env_id=env_i,
                logits=(logits[env_i] if logits is not None else None),
            )

    def compute_returns_and_advantages(self, last_values: Tensor):
        """
        last_values: (num_envs,) V(s_T) for each parallel env (any device)
        """
        if self.num_envs is None:
            raise RuntimeError("compute_returns_and_advantages called before any data was added")

        device = self.device
        last_values = last_values.to(device=device, dtype=torch.float32)

        T = self.pos
        rewards = self.rewards[:T]
        values  = self.values[:T]
        dones   = self.dones[:T].float()
        env_ids = self.env_ids[:T].long()

        advantages_raw = torch.zeros(T, device=device)

        # exact GAE per environment
        for env in range(self.num_envs):
            idxs = (env_ids == env).nonzero(as_tuple=False).flatten()
            if len(idxs) == 0:
                continue

            v_boot = last_values[env].reshape(1)       # (1,)
            v_path = torch.cat([values[idxs], v_boot]) # (len(idxs)+1,)
            r_path = rewards[idxs]
            d_path = dones[idxs]

            gae = 0.0
            for k in reversed(range(len(idxs))):
                non_terminal = 1.0 - d_path[k]
                delta = r_path[k] + self.gamma * v_path[k + 1] * non_terminal - v_path[k]
                gae = delta + self.gamma * self.lam * non_terminal * gae
                advantages_raw[idxs[k]] = gae

        # critic targets (true scale)
        self.returns[:T].copy_(advantages_raw + values)

        # fix non-finite
        advantages_raw[~torch.isfinite(advantages_raw)] = 0.0
        self.returns[:T][~torch.isfinite(self.returns[:T])] = 0.0

        assert_finite("returns / advantages_raw", self.returns[:T], advantages_raw)

        # diagnostics before normalization
        stats = {
            "adv_mean":  advantages_raw.mean().item(),
            "adv_std":   advantages_raw.std(unbiased=False).item(),
            "rets_mean": self.returns[:T].mean().item(),
            "rets_std":  self.returns[:T].std(unbiased=False).item(),
        }

        # optional: normalize returns (you were doing this in your PPG buffer)
        if self.normalize_returns:
            rets = self.returns[:T]
            rets = (rets - rets.mean()) / (rets.std(unbiased=False) + 1e-8)
            self.returns[:T].copy_(rets)

        # policy advantage normalization
        advs = (advantages_raw - advantages_raw.mean()) / (advantages_raw.std(unbiased=False) + 1e-8)
        self.advantages[:T].copy_(advs)

        return stats

    def get_batches(self, batch_size: int) -> Iterator[Dict[str, Tensor]]:
        """
        Yield random mini-batches for the PPO-style update phase.
        """
        T = self.pos
        perm = torch.randperm(T, device=self.device)

        obs          = self.obs[:T].detach()
        ctx          = self.context[:T].detach()
        actions      = self.actions[:T].detach()
        old_logps    = self.logps[:T].detach()
        old_vals     = self.values[:T].detach()
        returns      = self.returns[:T].detach()
        advs         = self.advantages[:T].detach()
        action_masks = self.action_masks[:T].detach()

        old_logits = None
        if self.store_old_logits:
            old_logits = self.old_logits[:T].detach()

        for start in range(0, T, batch_size):
            idx = perm[start : start + batch_size]
            batch = {
                "obs":          obs[idx],
                "ctx":          ctx[idx],
                "actions":      actions[idx],
                "old_logps":    old_logps[idx],
                "old_values":   old_vals[idx],
                "returns":      returns[idx],
                "advantages":   advs[idx],
                "action_masks": action_masks[idx],
            }
            if self.store_old_logits:
                batch["old_logits"] = old_logits[idx]
            yield batch

    def get_aux_batches(self, batch_size: int) -> Iterator[Dict[str, Tensor]]:
        """
        Yield mini-batches for the PPG auxiliary phase.
        Typically uses KL(old_logits || new_logits) + auxiliary value loss.
        """
        if not self.store_old_logits:
            raise RuntimeError("get_aux_batches requires store_old_logits=True")

        T = self.pos
        perm = torch.randperm(T, device=self.device)

        obs          = self.obs[:T].detach()
        ctx          = self.context[:T].detach()
        old_logits   = self.old_logits[:T].detach()
        returns      = self.returns[:T].detach()
        action_masks = self.action_masks[:T].detach()

        for start in range(0, T, batch_size):
            idx = perm[start : start + batch_size]
            yield {
                "obs":          obs[idx],
                "ctx":          ctx[idx],
                "old_logits":   old_logits[idx],
                "returns":      returns[idx],
                "action_masks": action_masks[idx],
            }

    def extend(self, other: "RolloutBufferVec"):
        """
        Copy the filled slice from `other` into the next free slice of `self`.
        """
        if not isinstance(other, RolloutBufferVec):
            raise TypeError("extend expects a RolloutBufferVec")

        n = other.pos
        if n == 0:
            return

        if self.pos + n > self.buffer_size:
            raise RuntimeError(f"Aux-buffer overflow: need {self.pos+n}, capacity {self.buffer_size}")

        def _copy(name: str):
            src: Tensor = getattr(other, name)
            if src is None:
                raise ValueError(f"{name} is None in source buffer")
            src = src.detach()

            dst: Tensor = getattr(self, name)
            dst_slice = dst[self.pos : self.pos + n]

            non_blocking = (src.device.type != dst_slice.device.type) and (src.dtype != torch.bool)
            dst_slice.copy_(src.to(self.device, non_blocking=non_blocking))

        for name in [
            "obs", "context", "actions", "logps", "values",
            "rewards", "dones", "action_masks", "env_ids"
        ]:
            _copy(name)

        if self.store_old_logits:
            if other.old_logits is None:
                raise ValueError("Source buffer has no old_logits but destination expects it")
            _copy("old_logits")

        # advantages/returns may or may not be computed yet; copy if present
        if other.advantages is not None:
            _copy("advantages")
        if other.returns is not None:
            _copy("returns")

        self.pos += n

    def clear(self):
        self.pos = 0
        self.advantages.zero_()
        self.returns.zero_()
