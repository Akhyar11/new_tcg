import os
import sys
import time
from collections import deque

import jax
import jax.numpy as jnp
import optax
import numpy as np
import psutil
from flax import serialization
from flax.jax_utils import replicate, unreplicate
from flax.core import unfreeze, freeze

from tcg_core.vector_env import VectorEnv
from tcg_core.buffer import RolloutBuffer
from tcg_core.ppo_update import ppo_update_step, get_action_and_value
from tcg_core.agents import LSTMAgent

class TrainerPPO:
    def __init__(self, agent_p0: LSTMAgent, agent_p1: LSTMAgent, config: dict):
        self.agent_p0 = agent_p0
        self.agent_p1 = agent_p1
        self.config = config

        self.num_envs = config.get("num_envs", 8)
        self.n_steps = config.get("n_steps", 128)
        self.batch_size = config.get("batch_size", 64)
        self.epochs = config.get("epochs", 1)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.learning_rate = config.get("learning_rate", 3e-4)
        self.initial_entropy = config.get("entropy_coef", 0.05)
        self.initial_clip = config.get("clip_ratio", 0.2)
        
        self.new_deck_path = config.get("new_deck_path", "new_deck")
        self.gen_deck_path = config.get("gen_deck_path", "deck_generated")
        self.save_dir = config.get("save_dir", "tcg_models")
        self.use_wandb = config.get("use_wandb", False)
        
        self.num_devices = self._auto_config_gpu()
        self.buffer = RolloutBuffer(n_steps=self.n_steps, num_envs=self.num_envs)
        
        # PPO Optimizer dengan pembekuan CardEmbedding (Teacher Distillation)
        import flax.traverse_util as tu
        
        def partition_fn(path):
            # Cek apakah 'CardEmbedding_0' ada di dalam path parameter
            if 'CardEmbedding' in path:
                return 'frozen'
            return 'trainable'
            
        flat_params = tu.flatten_dict(self.agent_p0.params)
        partition_dict = {k: partition_fn('/'.join(k)) for k in flat_params.keys()}
        partition_tree = tu.unflatten_dict(partition_dict)
        
        self.tx = optax.multi_transform(
            {
                'trainable': optax.chain(
                    optax.clip_by_global_norm(0.5),
                    optax.adamw(learning_rate=self.learning_rate, eps=1e-5, weight_decay=1e-4)
                ),
                'frozen': optax.set_to_zero()
            },
            partition_tree
        )
        self.opt_state = self.tx.init(self.agent_p0.params)
        
        self.rng = jax.random.PRNGKey(42)
        
    def _auto_config_gpu(self):
        num_devices = jax.device_count()
        if self.num_envs % num_devices != 0:
            self.num_envs = max((self.num_envs // num_devices) * num_devices, num_devices)
        if self.batch_size % num_devices != 0:
            self.batch_size = max((self.batch_size // num_devices) * num_devices, num_devices)
        return num_devices

    def _save_checkpoint(self, params, filename):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        path = os.path.join(self.save_dir, filename)
        with open(path, 'wb') as f:
            f.write(serialization.to_bytes(params))

    def train(self, total_timesteps: int, finetune_mode: bool = False):
        print(f"=== OOP PPO TRAINING (Timesteps: {total_timesteps:,}) ===")
        
        if self.use_wandb:
            # pyrefly: ignore [missing-import]
            import wandb
            wandb.init(project="tcg-pointer-network", config=self.config)
            
        env = VectorEnv(num_envs=self.num_envs, new_deck_path=self.new_deck_path, gen_deck_path=self.gen_deck_path)
        
        num_updates = total_timesteps // (self.n_steps * self.num_envs)
        
        # JAX Replicate
        params_repl_p0 = replicate(self.agent_p0.params)
        params_repl_p1 = replicate(self.agent_p1.params)
        opt_state_repl = replicate(self.opt_state)
        
        current_active_players = np.zeros(self.num_envs, dtype=np.int32)
        
        obs = env.reset()
        next_seq = obs["seq_input"]
        next_glob = obs["glob_input"]
        next_done = np.zeros(self.num_envs, dtype=np.float32)
        
        carry_c = np.zeros((self.num_envs, 256), dtype=np.float32)
        carry_h = np.zeros((self.num_envs, 256), dtype=np.float32)
        
        def make_carry_repl(c, h):
            return (
                jnp.array(c).reshape(self.num_devices, self.num_envs // self.num_devices, 256),
                jnp.array(h).reshape(self.num_devices, self.num_envs // self.num_devices, 256)
            )
            
        carry_repl_p0 = make_carry_repl(carry_c, carry_h)
        carry_repl_p1 = make_carry_repl(carry_c, carry_h)

        global_step = 0
        start_time = time.time()
        
        episodic_returns = np.zeros(self.num_envs, dtype=np.float32)
        recent_wins_p0 = deque(maxlen=150)
        
        reward_running_mean, reward_running_std, reward_norm_steps = 0.0, 1.0, 0
        
        for update in range(1, num_updates + 1):
            progress = update / num_updates
            current_entropy_coef = max(0.003 if finetune_mode else 0.005, self.initial_entropy * (1.0 - progress * (0.85 if finetune_mode else 0.9)))
            current_clip_ratio = max(0.05, self.initial_clip * (1.0 - progress * 0.75))

            ep_returns, ep_wins_p0 = [], []
            self.buffer.clear()
            
            for step in range(self.n_steps):
                global_step += self.num_envs
                self.rng, step_rng = jax.random.split(self.rng)
                step_rngs = jax.random.split(step_rng, self.num_devices)

                next_seq_sharded = next_seq.reshape((self.num_devices, self.num_envs // self.num_devices, *next_seq.shape[1:]))
                next_glob_sharded = next_glob.reshape((self.num_devices, self.num_envs // self.num_devices, *next_glob.shape[1:]))

                old_carry_c_np = np.array(carry_repl_p0[0]).reshape((self.num_envs, 256)).copy()
                old_carry_h_np = np.array(carry_repl_p0[1]).reshape((self.num_envs, 256)).copy()

                _, _, values_sharded_p0, logits_sharded_p0, carry_repl_p0 = get_action_and_value(
                    params_repl_p0, self.agent_p0.model.apply, next_seq_sharded, next_glob_sharded, carry_repl_p0, step_rngs
                )
                _, _, values_sharded_p1, logits_sharded_p1, carry_repl_p1 = get_action_and_value(
                    params_repl_p1, self.agent_p1.model.apply, next_seq_sharded, next_glob_sharded, carry_repl_p1, step_rngs
                )

                logits_np_p0 = np.array(logits_sharded_p0).reshape((self.num_envs, -1))
                logits_np_p1 = np.array(logits_sharded_p1).reshape((self.num_envs, -1))
                values_np_p0 = np.array(values_sharded_p0).reshape((self.num_envs,))
                values_np_p1 = np.array(values_sharded_p1).reshape((self.num_envs,))

                logits_np = np.where(current_active_players[:, None] == 0, logits_np_p0, logits_np_p1)
                values_np = np.where(current_active_players == 0, values_np_p0, values_np_p1)

                next_obs, rewards, dones, infos = env.step(logits_np)
                rewards = np.nan_to_num(rewards, nan=0.0, posinf=1.0, neginf=-1.0)

                reward_norm_steps += self.num_envs
                for r in rewards:
                    delta = r - reward_running_mean
                    reward_running_mean += delta / max(reward_norm_steps, 1)
                    reward_running_std += delta * (r - reward_running_mean)
                running_std = max(np.sqrt(reward_running_std / max(reward_norm_steps, 1)), 0.01)
                normalized_rewards = np.clip(rewards / running_std, -5.0, 5.0)

                episodic_returns += rewards
                
                carry_c_np_p0 = np.array(carry_repl_p0[0]).reshape((self.num_envs, 256))
                carry_h_np_p0 = np.array(carry_repl_p0[1]).reshape((self.num_envs, 256))
                carry_c_np_p1 = np.array(carry_repl_p1[0]).reshape((self.num_envs, 256))
                carry_h_np_p1 = np.array(carry_repl_p1[1]).reshape((self.num_envs, 256))

                for i, d in enumerate(dones):
                    if d:
                        ep_returns.append(float(episodic_returns[i]))
                        episodic_returns[i] = 0.0
                        res = infos[i].get("result", -1)
                        if res == 0: ep_wins_p0.append(1)
                        elif res == 1: ep_wins_p0.append(0)
                        
                        carry_c_np_p0[i] = carry_h_np_p0[i] = 0.0
                        carry_c_np_p1[i] = carry_h_np_p1[i] = 0.0

                carry_repl_p0 = make_carry_repl(carry_c_np_p0, carry_h_np_p0)
                carry_repl_p1 = make_carry_repl(carry_c_np_p1, carry_h_np_p1)

                actions_mask_np = np.stack([info["actions_mask"] for info in infos])
                glob_mask_np = np.stack([info["glob_mask"] for info in infos])

                masked_logits_np = logits_np - 1e9 * (1.0 - glob_mask_np)
                logits_max = np.max(masked_logits_np, axis=-1, keepdims=True)
                log_sum_exp = np.log(np.sum(np.exp(masked_logits_np - logits_max), axis=-1, keepdims=True))
                log_probs_all_np = (masked_logits_np - logits_max) - log_sum_exp
                mask_count = np.maximum(1.0, np.sum(actions_mask_np, axis=-1))
                multi_log_probs = np.sum(log_probs_all_np * actions_mask_np, axis=-1) / mask_count
                turn_changed_np = np.stack([info["turn_changed"] for info in infos])

                self.buffer.add(
                    next_seq, next_glob, actions_mask_np, multi_log_probs,
                    normalized_rewards, values_np, dones.astype(np.float32), turn_changed_np,
                    current_active_players, old_carry_c_np, old_carry_h_np
                )
                current_active_players = np.stack([info["active_player"] for info in infos])
                next_seq, next_glob, next_done = next_obs["seq_input"], next_obs["glob_input"], dones.astype(np.float32)

            # Bootstrapping
            next_seq_sharded = next_seq.reshape((self.num_devices, self.num_envs // self.num_devices, *next_seq.shape[1:]))
            next_glob_sharded = next_glob.reshape((self.num_devices, self.num_envs // self.num_devices, *next_glob.shape[1:]))
            step_rngs = jax.random.split(self.rng, self.num_devices)

            _, _, next_values_sharded, _, _ = get_action_and_value(
                params_repl_p0, self.agent_p0.model.apply, next_seq_sharded, next_glob_sharded, carry_repl_p0, step_rngs
            )
            self.buffer.compute_returns_and_advantages(np.array(next_values_sharded).reshape((self.num_envs,)), next_done, self.gamma, self.gae_lambda)

            # PPO Update
            mean_loss, update_count = 0.0, 0
            params_before, opt_state_before = params_repl_p0, opt_state_repl

            for epoch in range(self.epochs):
                for batch in self.buffer.get_batches(self.batch_size, seq_len=32):
                    num_seqs_per_batch = self.batch_size // 32
                    batch_sharded = {k: v.reshape((self.num_devices, num_seqs_per_batch // self.num_devices, *v.shape[1:])) for k, v in batch.items()}
                    params_repl_p0, opt_state_repl, loss, _ = ppo_update_step(
                        params_repl_p0, opt_state_repl, batch_sharded, self.agent_p0.model.apply, self.tx, current_clip_ratio, current_entropy_coef
                    )
                    mean_loss += float(loss[0])
                    update_count += 1

            mean_loss /= max(update_count, 1)

            if not np.isfinite(mean_loss):
                params_repl_p0, opt_state_repl, mean_loss = params_before, opt_state_before, 0.0

            # Logging
            recent_wins_p0.extend(ep_wins_p0)
            win_p0 = (np.mean(ep_wins_p0) * 100) if ep_wins_p0 else 0.0
            rolling_win_p0 = (np.mean(recent_wins_p0) * 100) if len(recent_wins_p0) > 0 else 0.0
            fps = int((self.num_envs * self.n_steps) / max(time.time() - start_time, 1e-8))
            start_time = time.time()
            
            print(f"Update {update:04d}/{num_updates} (Steps: {global_step:,}) | Loss: {mean_loss:.4f} | Win P0: {win_p0:.1f}% | Rolling: {rolling_win_p0:.1f}% | FPS: {fps}")

            if self.use_wandb:
                # pyrefly: ignore [missing-import]
                import wandb
                wandb.log({
                    "train/global_step": global_step,
                    "train/loss": mean_loss,
                    "metrics/win_rate_p0": win_p0,
                    "metrics/rolling_win_rate_p0": rolling_win_p0,
                    "system/fps": fps,
                    "hyperparams/entropy_coef": current_entropy_coef,
                    "hyperparams/clip_ratio": current_clip_ratio
                })

            # Self-play target threshold
            if rolling_win_p0 >= 60.0 and len(recent_wins_p0) == recent_wins_p0.maxlen:
                print(f"  🔥 Rolling Winrate {recent_wins_p0.maxlen} games P0 reached {rolling_win_p0:.1f}%! Updating P1 weights.")
                params_repl_p1 = params_repl_p0
                self._save_checkpoint(unreplicate(params_repl_p0), self.config.get("save_name_final", "model_final.msgpack"))
                recent_wins_p0.clear()
                
                # Otomatis upload ke Kaggle saat threshold tercapai
                from tcg_core.kaggle_sync import upload_to_kaggle
                try:
                    print("Mengunggah checkpoint terbaru ke Kaggle...")
                    upload_to_kaggle(self.save_dir, message=f"Auto-update: P0 Winrate {rolling_win_p0:.1f}%")
                except Exception as e:
                    print(f"Gagal mengunggah ke Kaggle: {e}")

        env.close()
        self._save_checkpoint(unreplicate(params_repl_p0), self.config.get("save_name_base", "model_base.msgpack"))
        self._save_checkpoint(unreplicate(params_repl_p0), self.config.get("save_name_final", "model_final.msgpack"))
        
        if self.use_wandb:
            # pyrefly: ignore [missing-import]
            import wandb
            wandb.finish()
            
        print("Training complete!")
