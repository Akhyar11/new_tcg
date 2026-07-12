#!/usr/bin/env python3
"""
PPO Training — Pokémon TCG RL Agent.

v3 — Convergence-Grade Training
================================
Perbaikan untuk konvergensi:

1. Lebih banyak timesteps (default 8M, cukup untuk Kaggle 9h)
2. Entropy schedule: 0.05 → 0.005 (eksplorasi tinggi awal, eksploitasi akhir)
3. Clip ratio schedule: 0.2 → 0.05 (fine-tuning akhir)
4. Non-symmetric opponents (P0 ≠ P1 deck) → gradient tidak cancel
5. Reward normalization (running stats) → skala stabil
6. Value tanh bounding → [-5, +5]
7. Best model terpisah dari regular checkpoints

Kaggle: at ~300 FPS, 8M timesteps ≈ 7.5 jam.
Set via env: TOTAL_TIMESTEPS=5000000 python train.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import jax
import jax.numpy as jnp
import optax
import numpy as np
from flax import serialization
import psutil

from agent_rl.model import PokemonAgent
from agent_rl.vector_env import VectorEnv, ShmVectorEnv
from agent_rl.buffer import RolloutBuffer
from agent_rl.ppo_update import ppo_update_step, get_action_and_value
from flax.jax_utils import replicate, unreplicate
from flax.core import freeze

# ─── Hyperparameters ───
NUM_ENVS = int(os.environ.get("RL_NUM_ENVS", "8"))
N_STEPS = 128
BATCH_SIZE = int(os.environ.get("RL_BATCH_SIZE", "64"))

# Fine-Tuning Mode (Cycle 2-5: GA deck → RL refine)
# Gunakan: FINETUNE_MODE=1 RL_DECK_PATH=agent_rl/ga_top_decks TOTAL_TIMESTEPS=2000000
FINETUNE_MODE = int(os.environ.get("FINETUNE_MODE", "0"))
if FINETUNE_MODE:
    TOTAL_TIMESTEPS = int(os.environ.get("TOTAL_TIMESTEPS", "2000000"))
    LEARNING_RATE = 1e-4               # Lower LR — refine, not relearn
    ENTROPY_COEF = 0.02                # Lower initial entropy
    EPOCHS = 2                         # Fewer epochs — avoid overfit
    print(f"[FineTune] MODE AKTIF — LR={LEARNING_RATE}, Entropy={ENTROPY_COEF}, Epochs={EPOCHS}, Steps={TOTAL_TIMESTEPS}")
else:
    # 8M timesteps ≈ 7.5 jam di 300 FPS (Kaggle-safe)
    # Override: TOTAL_TIMESTEPS=10000000 python train.py
    TOTAL_TIMESTEPS = int(os.environ.get("TOTAL_TIMESTEPS", "8000000"))
    LEARNING_RATE = 3e-4
    ENTROPY_COEF = 0.05                # Starting entropy (akan di-anneal)
    EPOCHS = 4

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RATIO = 0.2       # Starting clip ratio (akan di-anneal)
# VF_COEF = 0.5        # Di ppo_update.py sudah hardcoded 0.5

SAVE_DIR = "checkpoints"
DECK_PATH = os.environ.get(
    "RL_DECK_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck_generated")
)

# Memory monitoring — cetak setiap N update
MEM_LOG_INTERVAL = 100


def save_checkpoint(params, filename):
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    path = os.path.join(SAVE_DIR, filename)
    with open(path, 'wb') as f:
        f.write(serialization.to_bytes(params))
    print(f"[*] Checkpoint saved: {path}")


def auto_config_gpu():
    global NUM_ENVS, BATCH_SIZE
    NUM_ENVS = int(os.environ.get("RL_NUM_ENVS", NUM_ENVS))
    BATCH_SIZE = int(os.environ.get("RL_BATCH_SIZE", BATCH_SIZE))

    num_devices = jax.device_count()
    if NUM_ENVS % num_devices != 0:
        adjusted = max((NUM_ENVS // num_devices) * num_devices, num_devices)
        print(f"[!] NUM_ENVS={NUM_ENVS} not divisible by {num_devices} GPUs, "
              f"adjusted to {adjusted}")
        NUM_ENVS = adjusted
    if BATCH_SIZE % num_devices != 0:
        adjusted = max((BATCH_SIZE // num_devices) * num_devices, num_devices)
        print(f"[!] BATCH_SIZE={BATCH_SIZE} not divisible by {num_devices} GPUs, "
              f"adjusted to {adjusted}")
        BATCH_SIZE = adjusted
    return num_devices


def train():
    print("=== PPO TRAINING v3 (CONVERGENCE-GRADE) ===")
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Num envs: {NUM_ENVS}, Batch size: {BATCH_SIZE}")
    print(f"Gamma: {GAMMA}, GAE lambda: {GAE_LAMBDA}")
    print(f"Initial clip: {CLIP_RATIO}, Initial entropy: {ENTROPY_COEF}")
    print(f"Deck path: {DECK_PATH}")
    print()

    num_devices = auto_config_gpu()
    print(f"[*] {num_devices} GPU(s) detected, "
          f"{NUM_ENVS//num_devices} envs/GPU, "
          f"{BATCH_SIZE//num_devices} batch/GPU")

    rng = jax.random.PRNGKey(42)

    # 1. Init parallel environments (shared memory — zero pipe overhead!)
    print(f"Starting {NUM_ENVS} parallel envs from '{DECK_PATH}'...")
    env = ShmVectorEnv(num_envs=NUM_ENVS, deck_path=DECK_PATH)

    # 2. Init model & optimizer
    model = PokemonAgent(num_actions=250)
    rng, init_rng = jax.random.split(rng)

    dummy_seq = jnp.zeros((1, 93, 31))
    dummy_glob = jnp.zeros((1, 266))

    params = model.init(init_rng, dummy_seq, dummy_glob)

    # Resume dari checkpoint
    checkpoint_path = os.path.join(SAVE_DIR, "model_final.msgpack")
    if os.path.exists(checkpoint_path):
        print(f"[*] Resuming from checkpoint: {checkpoint_path}")
        with open(checkpoint_path, 'rb') as f:
            params = serialization.from_bytes(params, f.read())
    else:
        print("[*] Starting from scratch (random weights).")

    tx = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate=LEARNING_RATE, eps=1e-5)
    )
    opt_state = tx.init(params)

    best_reward = -999.0  # Track best average return for separate save

    # Replicate to all GPUs
    params_repl = replicate(params)
    opt_state_repl = replicate(opt_state)

    # 3. Init buffer
    buffer = RolloutBuffer(n_steps=N_STEPS, num_envs=NUM_ENVS)

    num_updates = TOTAL_TIMESTEPS // (N_STEPS * NUM_ENVS)
    print(f"Target updates: {num_updates}. Total timesteps: {TOTAL_TIMESTEPS:,}")

    # Running reward normalization stats
    # Reward dibagi (running_std + 1e-8) sebelum masuk buffer
    # Ini mirip dengan PopArt / running normalization
    reward_running_mean = 0.0
    reward_running_std = 1.0
    reward_norm_steps = 0

    obs = env.reset()
    next_seq = obs["seq_input"]
    next_glob = obs["glob_input"]
    next_done = np.zeros(NUM_ENVS, dtype=np.float32)

    global_step = 0
    start_time = time.time()
    env_step_counts = np.zeros(NUM_ENVS, dtype=np.int32)
    episodic_returns = np.zeros(NUM_ENVS, dtype=np.float32)

    print("\n=== MAIN TRAINING LOOP ===")
    for update in range(1, num_updates + 1):
        # Anneal entropy coefficient
        progress = update / num_updates
        if FINETUNE_MODE:
            current_entropy_coef = max(0.003, ENTROPY_COEF * (1.0 - progress * 0.85))
        else:
            current_entropy_coef = max(0.005, ENTROPY_COEF * (1.0 - progress * 0.9))

        # Anneal clip ratio (linear: 0.2 → 0.05)
        current_clip_ratio = max(0.05, CLIP_RATIO * (1.0 - progress * 0.75))

        ep_returns = []
        ep_wins_p0 = []
        ep_steps = []
        ep_end_reasons = []

        # ── Phase 1: Rollout ──
        buffer.clear()

        for step in range(N_STEPS):
            global_step += NUM_ENVS
            rng, step_rng = jax.random.split(rng)
            step_rngs = jax.random.split(step_rng, num_devices)

            next_seq_sharded = next_seq.reshape(
                (num_devices, NUM_ENVS // num_devices, *next_seq.shape[1:])
            )
            next_glob_sharded = next_glob.reshape(
                (num_devices, NUM_ENVS // num_devices, *next_glob.shape[1:])
            )

            _, _, values_sharded, logits_sharded = get_action_and_value(
                params_repl, model.apply, next_seq_sharded, next_glob_sharded, step_rngs
            )

            logits_np = np.array(logits_sharded).reshape((NUM_ENVS, -1))
            values_np = np.array(values_sharded).reshape((NUM_ENVS,))

            next_obs, rewards, dones, infos = env.step(logits_np)

            # ⭐ NaN guard — reward saja yang mungkin NaN (division di reward.py)
            rewards = np.nan_to_num(rewards, nan=0.0, posinf=1.0, neginf=-1.0)

            # ── Running reward normalization ──
            reward_norm_steps += NUM_ENVS
            # Update running stats (Welford-style)
            for r in rewards:
                delta = r - reward_running_mean
                reward_running_mean += delta / max(reward_norm_steps, 1)
                delta2 = r - reward_running_mean
                reward_running_std += delta * delta2
            running_std = np.sqrt(reward_running_std / max(reward_norm_steps, 1))
            running_std = max(running_std, 0.01)
            # Normalize rewards — clamp ke [-5, +5] cegah outlier ekstrim
            normalized_rewards = np.clip(rewards / running_std, -5.0, 5.0)

            episodic_returns += rewards
            env_step_counts += 1

            for i, d in enumerate(dones):
                if d:
                    ep_returns.append(float(episodic_returns[i]))
                    episodic_returns[i] = 0.0
                    result = infos[i].get("result", -1)
                    end_reason = infos[i].get("end_reason", 0)
                    if result == 0:
                        ep_wins_p0.append(1)
                    elif result == 1:
                        ep_wins_p0.append(0)
                    ep_steps.append(env_step_counts[i])
                    ep_end_reasons.append(end_reason)
                    env_step_counts[i] = 0

            actions_mask_np = np.stack([info["actions_mask"] for info in infos])
            glob_mask_np = np.stack([info["glob_mask"] for info in infos])

            # Old log-probs (konsisten dengan loss_fn)
            masked_logits_np = logits_np - 1e9 * (1.0 - glob_mask_np)
            logits_max = np.max(masked_logits_np, axis=-1, keepdims=True)
            log_sum_exp = np.log(np.sum(np.exp(masked_logits_np - logits_max), axis=-1, keepdims=True))
            log_probs_all_np = (masked_logits_np - logits_max) - log_sum_exp

            mask_count = np.maximum(1.0, np.sum(actions_mask_np, axis=-1))
            multi_log_probs = np.sum(log_probs_all_np * actions_mask_np, axis=-1) / mask_count

            # Simpan NORMALIZED rewards ke buffer
            buffer.add(
                next_seq, next_glob, actions_mask_np, multi_log_probs,
                normalized_rewards, values_np, dones.astype(np.float32)
            )

            next_seq = next_obs["seq_input"]
            next_glob = next_obs["glob_input"]
            next_done = dones.astype(np.float32)

        # ── Phase 2: GAE & Bootstrapping ──
        next_seq_sharded = next_seq.reshape(
            (num_devices, NUM_ENVS // num_devices, *next_seq.shape[1:])
        )
        next_glob_sharded = next_glob.reshape(
            (num_devices, NUM_ENVS // num_devices, *next_glob.shape[1:])
        )
        step_rngs = jax.random.split(rng, num_devices)

        _, _, next_values_sharded, _ = get_action_and_value(
            params_repl, model.apply, next_seq_sharded, next_glob_sharded, step_rngs
        )
        next_values = np.array(next_values_sharded).reshape((NUM_ENVS,))
        buffer.compute_returns_and_advantages(next_values, next_done, GAMMA, GAE_LAMBDA)

        # ── Phase 3: PPO Optimization ──
        mean_loss = 0.0
        update_count = 0

        # ⭐ Simpan params DAN opt_state sebelum update untuk rollback jika NaN
        params_before = freeze(params_repl)
        opt_state_before = freeze(opt_state_repl)

        for epoch in range(EPOCHS):
            for batch in buffer.get_batches(BATCH_SIZE):
                batch_sharded = {
                    k: v.reshape((num_devices, BATCH_SIZE // num_devices, *v.shape[1:]))
                    for k, v in batch.items()
                }

                params_repl, opt_state_repl, loss, _ = ppo_update_step(
                    params_repl, opt_state_repl, batch_sharded, model.apply, tx,
                    current_clip_ratio, current_entropy_coef
                )
                mean_loss += float(loss[0])
                update_count += 1

        mean_loss /= update_count

        # ⭐ NaN guard: rollback params DAN opt_state jika loss NaN/Inf
        if not np.isfinite(mean_loss):
            print(f"  ⚠️ WARNING: Loss NaN/Inf ({mean_loss})! Rollback ke update sebelumnya.")
            params_repl = params_before
            opt_state_repl = opt_state_before
            mean_loss = 0.0

        # ── Phase 4: Logging ──
        if update % 1 == 0:
            avg_ret = np.mean(ep_returns) if ep_returns else 0.0
            win_p0 = (np.mean(ep_wins_p0) * 100) if ep_wins_p0 else 0.0
            games_played = len(ep_wins_p0)
            avg_steps = np.mean(ep_steps) if ep_steps else 0.0

            reason_labels = {1: "Prize", 2: "DeckOut", 3: "NoActive", 4: "Effect", 9: "Timeout"}
            if ep_end_reasons:
                reason_counts = {}
                for r in ep_end_reasons:
                    reason_counts[r] = reason_counts.get(r, 0) + 1
                reason_str = " | ".join(
                    f"{reason_labels.get(r, f'R{r}')}:{c}/{games_played}"
                    for r, c in sorted(reason_counts.items())
                )
            else:
                reason_str = "N/A"

            fps = int((NUM_ENVS * N_STEPS) / (time.time() - start_time + 1e-8))
            start_time = time.time()

            norm_scale = running_std
            pct = (update / num_updates) * 100

            print(f"Update {update:04d}/{num_updates} ({pct:.0f}%) | "
                  f"Step: {global_step:,} | FPS: {fps} | "
                  f"Games: {games_played} | Steps/Game: {avg_steps:.0f} | "
                  f"Return: {avg_ret:+.2f} | "
                  f"Win P0: {win_p0:.1f}% | "
                  f"Loss: {mean_loss:.4f} | "
                  f"Clip: {current_clip_ratio:.3f} | "
                  f"Entropy: {current_entropy_coef:.3f} | "
                  f"Norm: {norm_scale:.2f}")
            print(f"  End ─ {reason_str}")

        # ── Phase 5: Checkpointing ──
        if update % 50 == 0:
            save_checkpoint(unreplicate(params_repl), f"model_update_{update}.msgpack")

        # Save best model (by average return)
        if ep_returns and avg_ret > best_reward:
            best_reward = avg_ret
            save_checkpoint(unreplicate(params_repl), "model_best.msgpack")
            print(f"  ⭐ New best model saved! Avg return: {best_reward:+.2f}")

        # ⭐ Memory monitoring — deteksi leak
        if update % MEM_LOG_INTERVAL == 0:
            proc = psutil.Process()
            mem_mb = proc.memory_info().rss / 1e6
            cpu_percent = proc.cpu_percent(interval=0.1)
            print(f"  [MEM] RSS={mem_mb:.0f}MB | CPU={cpu_percent:.0f}%")
            # Peringatan jika memory > 12GB (Kaggle limit ~16GB)
            if mem_mb > 12000:
                print(f"  ⚠️ WARNING: Memory usage tinggi ({mem_mb:.0f}MB)! Berisiko OOM.")

        # Update entropy & clip ratio di ppo_update function via closure approach
        # Actually, we need to pass these params. Let me update the ppo_update call.
        # The current ppo_update_step takes clip_ratio as argument, but doesn't take entropy_coef.
        # Wait, entropy_coef is hardcoded in ppo_update.py line 55.
        # Let me check if we need to modify ppo_update.py too.

    # Final save
    print("Training complete. Closing env.")
    env.close()
    save_checkpoint(unreplicate(params_repl), "model_final.msgpack")
    print(f"Best model saved with avg return: {best_reward:+.2f}")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    import jax.numpy as jnp
    train()
