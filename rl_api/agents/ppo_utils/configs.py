from dataclasses import dataclass
from typing import Optional, Dict, Tuple

@dataclass
class DimensionConfig:
    obs_shape: Tuple[int, ...]   # was (seq_len, num_features), now any shape
    action_dim: int
    context_dim: int             # still flat context vector dim
    
@dataclass
class PPOConfig:
    num_epochs: int = 2
    clip_eps: float = 0.2
    clip_vf: Optional[float] = None
    vf_loss_clip: Optional[float] = None
    vf_coef: float = 0.5    
    gamma: float = 0.99
    gae_lambda: float = 0.95
    target_kl: float = 0.01
    grad_clip: Optional[float] = None

    entropy_coef_start: float = 0.0
    entropy_coef_end: Optional[float] = None
    entropy_decay_start_step: Optional[float] = None
    entropy_decay_end_step: Optional[float] = None

@dataclass
class BufferConfig:
    buffer_size: int = 4096
    batch_size: int = 512

@dataclass
class TrainingConfig:
    total_updates: int = 1000

@dataclass
class EvalConfig:
    eval_method: Optional[str] = "sample" # greedy/sample (greedy is taking the max distribution probobability action, sample is sampling based on policy distribution)
    n_steps: Optional[int] = 10000

@dataclass
class LoggingConfig:
    current_update:  Optional[int] = 0
    log_interval:    Optional[int] = None   # sample‐epoch logs
    eval_interval:   Optional[int] = None   # periodic eval logs
    html_log_path:   Optional[str] = None
    tensorboard_path:Optional[str] = None
    param_groups_names: Optional[Dict[int,str]] = None
    verbose:         bool = True

@dataclass
class SavingConfig:
    save_interval:    Optional[int] = None   # how often to checkpoint
    save_agent_path:  Optional[str] = None   # base folder

