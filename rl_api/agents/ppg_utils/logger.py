import os
from typing import Optional, Dict, Any
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from rl_api.agents.ppo_utils.configs import LoggingConfig

class Logger:
    def __init__(
        self,
        cfg: LoggingConfig,
        param_groups_names: Optional[Dict[int, str]] = None
    ):
        self.cfg = cfg
        self.param_groups_names = param_groups_names

        # TensorBoard writer
        self.writer: Optional[SummaryWriter] = None
        if cfg.tensorboard_path:
            os.makedirs(cfg.tensorboard_path, exist_ok=True)
            self.writer = SummaryWriter(log_dir=cfg.tensorboard_path, purge_step=cfg.current_update)

        # HTML init
        if cfg.html_log_path:
            if not cfg.html_log_path.endswith(".html"):
                cfg.html_log_path = cfg.html_log_path + ".html"
            os.makedirs(os.path.dirname(cfg.html_log_path), exist_ok=True)
            if cfg.current_update <= 1:
                self._init_html(cfg.html_log_path)

    
    def _init_html(self, file_path: str, title: str = "Training Log"):
        with open(file_path, "w") as f:
            f.write(f"""<!DOCTYPE html>
            <html lang="en">
            <head>
                <script>
                    setTimeout(() => location.reload(), 2000); // refresh every 2 seconds
                </script>
                <meta charset="UTF-8">
                <title>{title}</title>
                <style>
                    body {{
                        font-family: monospace;
                        background-color: #111; 
                        color: #eee;
                        padding: 1em;
                        font-size: 1.1em;  /* ← adjust this line */
                    }}
                    .log-line {{
                        white-space: pre-wrap;
                        word-wrap: break-word;
                        margin-bottom: 0.3em;
                    }}
                </style>
            </head>
            <body>
            <h2>{title}</h2>
            <hr>
            """)


    def _html_line(self, file_path: str, message: str, color: str = "white"):
        with open(file_path, "a") as f:
            f.write(f'<div class="log-line" style="color:{color}">{message}</div>\n')


    def log_sample(
        self,
        update_idx: int,
        global_step: int,
        stats: Dict[str, Any],
    ):
        """
        Centralized console, TensorBoard, and HTML logging for each sample epoch.
        `stats` must include:
          policy_loss, value_loss, entropy, approx_kl, explained_var,
          enc_grad, actor_grad, critic_grad,

        """
        lc = self.cfg


        # --- Console ---
        if lc.verbose:
            print(f"\n[Upd {update_idx:>3} | step: {global_step}]"
                  f"\n| pi_loss: {stats['policy_loss']:+.4f}  v_loss: {stats['value_loss']:+.4f}"
                  f"  ent: {stats['entropy']:.3f}  ent_coef: {stats['entropy_coef']:.4f}"
                  f"  kl: {stats['approx_kl']:.4f}  batches_used: {stats['batches_used_fraction']:.3%}"
                  f"  expl_var: {stats['explained_var']:.2f}  clip: {stats['clip_frac']:.2%}"

                  f"\n| adv_mean: {stats['adv_mean']:.4f}  adv_std: {stats['adv_std']:.4f}"
                  f"  rets_mean: {stats['rets_mean']:.4f}  rets_std: {stats['rets_std']:.4f}"


                  f"\n| reward_sum: {stats['reward_sum']:.3f}  reward_pre_step: {stats['reward_per_step']:.5f}")
            
            print(f"[Grads] policy_enc: {stats['policy_enc_grad']:.3g}  act: {stats['actor_grad']:.3g}"
                  f"  value_enc: {stats['value_enc_grad']:.3g}, critic: {stats['critic_grad']:.3g}")
            if self.param_groups_names:
                lr_info = " | ".join(
                    f"{self.param_groups_names[i]}_lr: {pg['lr']:.2e}"
                    for i, pg in enumerate(self.writer._get_file_writer()._get_all_writers().items())
                )
                print(f"| {lr_info}")

        # --- TensorBoard ---
        if self.writer:
            step = update_idx

            self.writer.add_scalar("reward/sum",     stats["reward_sum"],       step)
            self.writer.add_scalar("loss/per_step",  stats["reward_per_step"],  step)


            self.writer.add_scalar("loss/policy", stats["policy_loss"], step)
            self.writer.add_scalar("loss/value",  stats["value_loss"],  step)
            self.writer.add_scalar("loss/entropy",stats["entropy"],      step)


            self.writer.add_scalar("adv_and_rets/adv_mean",     stats["adv_mean"],      step)
            self.writer.add_scalar("adv_and_rets/adv_std",      stats["adv_std"],       step)
            self.writer.add_scalar("adv_and_rets/rets_mean",    stats["rets_mean"],     step)
            self.writer.add_scalar("adv_and_rets/rets_std",     stats["rets_std"],      step)


            self.writer.add_scalar("kl_stats/approx_kl",   stats["approx_kl"],    step)
            self.writer.add_scalar("kl_stats/true_kl",     stats["true_kl"],       step)
            self.writer.add_scalar("kl_stats/batches_used_fraction",   stats["batches_used_fraction"],    step)


            self.writer.add_scalar("clip_frac",   stats["clip_frac"],    step)
            self.writer.add_scalar("entropy_coef",stats["entropy_coef"], step)
  

            self.writer.add_scalar("grads/policy_encoder", stats["policy_enc_grad"], step)
            self.writer.add_scalar("grads/actor",   stats["actor_grad"],  step)
            self.writer.add_scalar("grads/critic",  stats["critic_grad"], step)
            self.writer.add_scalar("grads/value_encoder",  stats["value_enc_grad"], step)


    def log_eval(
        self,
        update: int,
        sum_reward: float,
        reward_per_step: float,
        is_best: bool,
        prefix: str = "eval"
    ):
        """
        Centralized console, TensorBoard, and HTML logging for each evaluation.
        Model saving remains in the Agent.
        """
        lc = self.cfg

        # Console
        if lc.verbose:
            if is_best:
                print(f"[Eval @ update {update}] New best agent found! sum reward: {sum_reward:.2f}, reward_per_step: {reward_per_step:.4f}")
            else:
                print(f"[Eval @ update {update}] Sum reward: {sum_reward:.2f}, reward_per_step: {reward_per_step:.4f}")

        # HTML
        if lc.html_log_path:
            color = "cyan" if is_best else ("green" if sum_reward > 0 else "red" if sum_reward < 0 else "grey")
            msg = f"[Eval @ update {update}] Sum reward: {sum_reward:.2f}"
            self._html_line(lc.html_log_path, msg, color)

        # TensorBoard
        if self.writer:
            self.writer.add_scalar(f"{prefix}/sum_reward",         sum_reward,         update)
            self.writer.add_scalar(f"{prefix}/reward_per_step",    reward_per_step,    update)

