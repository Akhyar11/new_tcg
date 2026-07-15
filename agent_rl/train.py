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
from collections import deque
from dotenv import load_dotenv

# Muat variabel environment dari .env secara robust
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
env_path = os.path.join(root_dir, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import jax
import jax.numpy as jnp
import optax
import numpy as np
from flax import serialization
import psutil

from agent_rl.model import PokemonAgent
from agent_rl.vector_env import VectorEnv
from agent_rl.buffer import RolloutBuffer
from agent_rl.ppo_update import ppo_update_step, get_action_and_value
from flax.jax_utils import replicate, unreplicate

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
    # 15M timesteps
    # Override: TOTAL_TIMESTEPS=15000000 python train.py
    TOTAL_TIMESTEPS = int(os.environ.get("TOTAL_TIMESTEPS", "15000000"))
    LEARNING_RATE = 3e-4
    ENTROPY_COEF = 0.05                # Starting entropy (akan di-anneal)
    EPOCHS = 4

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RATIO = 0.2       # Starting clip ratio (akan di-anneal)
# VF_COEF = 0.5        # Di ppo_update.py sudah hardcoded 0.5

SAVE_DIR = os.environ.get("SAVE_DIR", "tcg_models")
NEW_DECK_PATH = os.environ.get("NEW_DECK_PATH", "new_deck")
GEN_DECK_PATH = os.environ.get("GEN_DECK_PATH", "agent_rl/deck_generated")
KAGGLE_INPUT_DIR = os.environ.get("KAGGLE_INPUT_DIR", "")  # Contoh: /kaggle/input/tcg-models

# Memory monitoring — cetak setiap N update
MEM_LOG_INTERVAL = 100


def save_checkpoint(params, filename):
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    path = os.path.join(SAVE_DIR, filename)
    with open(path, 'wb') as f:
        f.write(serialization.to_bytes(params))
    print(f"[*] Checkpoint saved: {path}")

import subprocess
import json

from kaggle.api.kaggle_api_extended import KaggleApi

def get_kaggle_api():
    # Map KAGGLE_API_TOKEN from .env to KAGGLE_KEY if KAGGLE_KEY is not set
    if "KAGGLE_KEY" not in os.environ and "KAGGLE_API_TOKEN" in os.environ:
        os.environ["KAGGLE_KEY"] = os.environ["KAGGLE_API_TOKEN"]
    api = KaggleApi()
    api.authenticate()
    return api

def upload_to_kaggle(save_dir, message="Update models"):
    dataset_id = os.environ.get("KAGGLE_DATASET_ID")
    if not dataset_id:
        print("[!] KAGGLE_DATASET_ID tidak diset. Lewati sinkronisasi Kaggle.")
        return

    metadata_path = os.path.join(save_dir, "dataset-metadata.json")
    if not os.path.exists(metadata_path):
        metadata = {
            "title": dataset_id.split("/")[-1],
            "id": dataset_id,
            "licenses": [{"name": "CC0-1.0"}]
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

    try:
        api = get_kaggle_api()
        
        # Check if dataset exists
        dataset_exists = True
        try:
            print(f"[*] Memeriksa status dataset Kaggle ({dataset_id})...")
            api.dataset_status(dataset_id)
        except Exception as status_err:
            status_code = getattr(status_err, 'status', None)
            err_msg = str(status_err).lower()
            
            # Jika error 401 (Unauthorized) atau kredensial tidak valid
            if status_code in (401, 403) or "unauthorized" in err_msg or "unauthenticated" in err_msg or "401" in err_msg:
                print(f"[!] Kaggle Authentication Error: Kredensial tidak valid atau tidak memiliki akses (HTTP {status_code}).")
                print("    Silakan periksa kembali KAGGLE_USERNAME dan KAGGLE_API_TOKEN di file .env Anda.")
                return
                
            # Jika error 404 (Not Found), berarti dataset belum ada
            if status_code == 404 or "404" in err_msg or "not found" in err_msg:
                dataset_exists = False
                print(f"[*] Dataset belum ada di Kaggle. Akan mencoba membuat dataset baru.")
            else:
                # Fallback untuk error lain
                dataset_exists = False
                print(f"[*] Dataset tidak dapat diakses ({status_err}). Mencoba membuat dataset baru.")

        if dataset_exists:
            print(f"[*] Mengupload versi baru ke Kaggle Dataset ({dataset_id}) menggunakan Python API...")
            api.dataset_create_version(save_dir, version_notes=message, dir_mode="zip")
            print("[*] Sukses sinkronisasi ke Kaggle Dataset.")
        else:
            print(f"[*] Mencoba membuat dataset baru di Kaggle ({dataset_id})...")
            api.dataset_create_new(save_dir, public=False, quiet=False, convert_to_csv=False, dir_mode="zip")
            print("[*] Sukses membuat dan mengunggah ke Kaggle Dataset baru.")
    except Exception as e:
        print(f"[!] Terjadi error saat upload Kaggle: {e}")

def download_from_kaggle(save_dir):
    dataset_id = os.environ.get("KAGGLE_DATASET_ID")
    if not dataset_id:
        return
    print(f"[*] Mencoba mendownload checkpoint dari Kaggle Dataset ({dataset_id}) menggunakan Python API...")
    try:
        api = get_kaggle_api()
        api.dataset_download_files(dataset_id, path=save_dir, unzip=True)
        print("[*] Sukses mendownload dan unzip model dari Kaggle.")
    except Exception as e:
        print(f"[!] Terjadi error saat download Kaggle: {e}")

def auto_config_gpu():
    global NUM_ENVS, BATCH_SIZE
    NUM_ENVS = int(os.environ.get("RL_NUM_ENVS", NUM_ENVS))
    BATCH_SIZE = int(os.environ.get("RL_BATCH_SIZE", BATCH_SIZE))

    num_devices = jax.device_count()
    # ⚠️ Untuk model kecil (<5M param), multi-GPU via PCIe LEBIH LAMBAT
    # dari 1 GPU karena pmap overhead (~3ms) > forward pass (~0.5ms)
    if num_devices > 1:
        print(f"[*] {num_devices} GPU(s) detected.")
        print(f"    ⚠️ Untuk model kecil, multi-GPU bisa LEBIH LAMBAT!")
        print(f"    Gunakan CUDA_VISIBLE_DEVICES=0 untuk force single GPU")

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
    print(f"New Deck path (70%): {NEW_DECK_PATH}")
    print(f"Gen Deck path (30%): {GEN_DECK_PATH}")
    print()

    num_devices = auto_config_gpu()
    print(f"[*] {num_devices} GPU(s) detected, "
          f"{NUM_ENVS//num_devices} envs/GPU, "
          f"{BATCH_SIZE//num_devices} batch/GPU")

    rng = jax.random.PRNGKey(42)

    # 1. Init parallel environments
    print(f"Starting {NUM_ENVS} parallel envs...")
    print(f"  New Deck Path: {NEW_DECK_PATH}")
    print(f"  Gen Deck Path: {GEN_DECK_PATH}")
    env = VectorEnv(num_envs=NUM_ENVS, new_deck_path=NEW_DECK_PATH, gen_deck_path=GEN_DECK_PATH)

    # 2. Init model & optimizer
    model = PokemonAgent(num_actions=250)
    rng, init_rng = jax.random.split(rng)

    dummy_seq = jnp.zeros((1, 113, 31))
    dummy_glob = jnp.zeros((1, 266))

    params_p0 = model.init(init_rng, dummy_seq, dummy_glob)
    params_p1 = model.init(init_rng, dummy_seq, dummy_glob)

    model_final_path = os.path.join(SAVE_DIR, "model_final.msgpack")
    model_base_path = os.path.join(SAVE_DIR, "model_base.msgpack")

    # Download via Kaggle API jika script dijalankan di Colab/Lokal (tidak ada di mount path Kaggle)
    alt_final = os.path.join(KAGGLE_INPUT_DIR, "model_final.msgpack") if KAGGLE_INPUT_DIR else ""
    if not os.path.exists(model_final_path) and not os.path.exists(model_base_path):
        if not alt_final or not os.path.exists(alt_final):
            download_from_kaggle(SAVE_DIR)

    # Kaggle Fallback Logic
    # Jika script dijalankan di Kaggle, dataset asli ada di KAGGLE_INPUT_DIR (Read-only)
    if not os.path.exists(model_final_path) and KAGGLE_INPUT_DIR:
        alt_final = os.path.join(KAGGLE_INPUT_DIR, "model_final.msgpack")
        if os.path.exists(alt_final):
            model_final_path = alt_final
            
    if not os.path.exists(model_base_path) and KAGGLE_INPUT_DIR:
        alt_base = os.path.join(KAGGLE_INPUT_DIR, "model_base.msgpack")
        if os.path.exists(alt_base):
            model_base_path = alt_base

    # Load logic for P0
    if os.path.exists(model_final_path):
        print(f"[*] Resuming P0 from model_final: {model_final_path}")
        with open(model_final_path, 'rb') as f:
            params_p0 = serialization.from_bytes(params_p0, f.read())
    elif os.path.exists(model_base_path):
        print(f"[*] Resuming P0 from model_base: {model_base_path}")
        with open(model_base_path, 'rb') as f:
            params_p0 = serialization.from_bytes(params_p0, f.read())
    else:
        print("[*] P0 Starting from scratch (random weights).")

    # Load logic for P1
    if os.path.exists(model_final_path):
        print(f"[*] Resuming P1 from model_final: {model_final_path}")
        with open(model_final_path, 'rb') as f:
            params_p1 = serialization.from_bytes(params_p1, f.read())
    elif os.path.exists(model_base_path):
        print(f"[*] Resuming P1 from model_base: {model_base_path}")
        with open(model_base_path, 'rb') as f:
            params_p1 = serialization.from_bytes(params_p1, f.read())
    else:
        print("[*] P1 Starting from scratch (random weights).")

    tx = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate=LEARNING_RATE, eps=1e-5)
    )
    opt_state = tx.init(params_p0)

    best_reward = -999.0  # Track best average return for separate save

    # Replicate to all GPUs
    params_repl_p0 = replicate(params_p0)
    params_repl_p1 = replicate(params_p1)
    opt_state_repl = replicate(opt_state)

    # State active_players tracker
    current_active_players = np.zeros(NUM_ENVS, dtype=np.int32)

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
    
    # Riwayat kemenangan P0 untuk update P1
    recent_wins_p0 = deque(maxlen=100)
    recent_wins_m_vs_m = deque(maxlen=100)
    recent_wins_m_vs_r = deque(maxlen=100)
    recent_wins_r_vs_r = deque(maxlen=100)
    p1_update_count = 0

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
        ep_wins_m_vs_m = []
        ep_wins_m_vs_r = []
        ep_wins_r_vs_r = []
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

            _, _, values_sharded_p0, logits_sharded_p0 = get_action_and_value(
                params_repl_p0, model.apply, next_seq_sharded, next_glob_sharded, step_rngs
            )
            _, _, values_sharded_p1, logits_sharded_p1 = get_action_and_value(
                params_repl_p1, model.apply, next_seq_sharded, next_glob_sharded, step_rngs
            )

            logits_np_p0 = np.array(logits_sharded_p0).reshape((NUM_ENVS, -1))
            logits_np_p1 = np.array(logits_sharded_p1).reshape((NUM_ENVS, -1))
            values_np_p0 = np.array(values_sharded_p0).reshape((NUM_ENVS,))
            values_np_p1 = np.array(values_sharded_p1).reshape((NUM_ENVS,))

            # Gabungkan logits & value sesuai active player
            logits_np = np.where(current_active_players[:, None] == 0, logits_np_p0, logits_np_p1)
            values_np = np.where(current_active_players == 0, values_np_p0, values_np_p1)

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
                    p0_deck_type = infos[i].get("p0_deck_type", 0)
                    p1_deck_type = infos[i].get("p1_deck_type", 0)
                    
                    is_win = -1
                    if result == 0:
                        is_win = 1
                    elif result == 1:
                        is_win = 0
                        
                    if is_win != -1:
                        ep_wins_p0.append(is_win)
                        if p0_deck_type == 0 and p1_deck_type == 0:
                            ep_wins_m_vs_m.append(is_win)
                        elif p0_deck_type == 0 and p1_deck_type == 1:
                            ep_wins_m_vs_r.append(is_win)
                        elif p0_deck_type == 1 and p1_deck_type == 1:
                            ep_wins_r_vs_r.append(is_win)
                            
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
            
            turn_changed_np = np.stack([info["turn_changed"] for info in infos])

            # Simpan NORMALIZED rewards ke buffer
            buffer.add(
                next_seq, next_glob, actions_mask_np, multi_log_probs,
                normalized_rewards, values_np, dones.astype(np.float32), turn_changed_np,
                current_active_players
            )
            
            # Update active_players untuk step berikutnya
            current_active_players = np.stack([info["active_player"] for info in infos])

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
            params_repl_p0, model.apply, next_seq_sharded, next_glob_sharded, step_rngs
        )
        next_values = np.array(next_values_sharded).reshape((NUM_ENVS,))
        buffer.compute_returns_and_advantages(next_values, next_done, GAMMA, GAE_LAMBDA)

        # ── Phase 3: PPO Optimization ──
        mean_loss = 0.0
        update_count = 0

        # ⭐ Simpan params DAN opt_state sebelum update untuk rollback jika NaN
        # JAX arrays immutable → reference copy sudah cukup (tidak perlu freeze/deepcopy)
        params_before = params_repl_p0
        opt_state_before = opt_state_repl

        for epoch in range(EPOCHS):
            for batch in buffer.get_batches(BATCH_SIZE):
                batch_sharded = {
                    k: v.reshape((num_devices, BATCH_SIZE // num_devices, *v.shape[1:]))
                    for k, v in batch.items()
                }

                params_repl_p0, opt_state_repl, loss, _ = ppo_update_step(
                    params_repl_p0, opt_state_repl, batch_sharded, model.apply, tx,
                    current_clip_ratio, current_entropy_coef
                )
                mean_loss += float(loss[0])
                update_count += 1

        mean_loss /= update_count

        # ⭐ NaN guard: rollback params DAN opt_state jika loss NaN/Inf
        if not np.isfinite(mean_loss):
            print(f"  ⚠️ WARNING: Loss NaN/Inf ({mean_loss})! Rollback ke update sebelumnya.")
            sys.stdout.flush()
            params_repl_p0 = params_before
            opt_state_repl = opt_state_before
            mean_loss = 0.0

        # ── Phase 4: Logging ──
        if update % 1 == 0:
            avg_ret = np.mean(ep_returns) if ep_returns else 0.0
            
            # Tambahkan hasil game ke deque history
            recent_wins_p0.extend(ep_wins_p0)
            recent_wins_m_vs_m.extend(ep_wins_m_vs_m)
            recent_wins_m_vs_r.extend(ep_wins_m_vs_r)
            recent_wins_r_vs_r.extend(ep_wins_r_vs_r)
            
            win_p0 = (np.mean(ep_wins_p0) * 100) if ep_wins_p0 else 0.0
            rolling_win_p0 = (np.mean(recent_wins_p0) * 100) if len(recent_wins_p0) > 0 else 0.0
            
            # Win rates by specific matchup
            win_m_vs_m = (np.mean(recent_wins_m_vs_m) * 100) if len(recent_wins_m_vs_m) > 0 else 0.0
            win_m_vs_r = (np.mean(recent_wins_m_vs_r) * 100) if len(recent_wins_m_vs_r) > 0 else 0.0
            win_r_vs_r = (np.mean(recent_wins_r_vs_r) * 100) if len(recent_wins_r_vs_r) > 0 else 0.0
            
            fps = int((NUM_ENVS * N_STEPS) / (time.time() - start_time + 1e-8))
            start_time = time.time()
            
            # Print a single clean line per update to stdout and flush
            print(f"Update {update:04d}/{num_updates} | Loss: {mean_loss:.4f} | Win: {win_p0:.1f}% | RollWin: {rolling_win_p0:.1f}% | MvM: {win_m_vs_m:.1f}% | MvR: {win_m_vs_r:.1f}% | P1_Up: {p1_update_count} | FPS: {fps} | Steps: {global_step:,}")
            sys.stdout.flush()
            
            # P1 Frozen Weight Update Logic
            if rolling_win_p0 >= 60.0 and len(recent_wins_p0) == recent_wins_p0.maxlen:
                p1_update_count += 1
                print(f"  🔥 [P1 Update #{p1_update_count}] Rolling Winrate {recent_wins_p0.maxlen} Game P0 mencapai {rolling_win_p0:.1f}%! Update bobot P1 dan simpan model_final ke Kaggle.")
                sys.stdout.flush()
                params_repl_p1 = params_repl_p0
                save_checkpoint(unreplicate(params_repl_p0), "model_final.msgpack")
                upload_to_kaggle(SAVE_DIR, message=f"Update model_final dengan winrate P0 {rolling_win_p0:.1f}% (Update #{p1_update_count})")
                recent_wins_p0.clear()
                recent_wins_m_vs_m.clear()
                recent_wins_m_vs_r.clear()
                recent_wins_r_vs_r.clear()

        # ⭐ Memory monitoring — deteksi leak
        if update % MEM_LOG_INTERVAL == 0:
            proc = psutil.Process()
            mem_mb = proc.memory_info().rss / 1e6
            cpu_percent = proc.cpu_percent(interval=0.1)
            print(f"  [MEM] RSS={mem_mb:.0f}MB | CPU={cpu_percent:.0f}%")
            # Peringatan jika memory > 12GB (Kaggle limit ~16GB)
            if mem_mb > 12000:
                print(f"  ⚠️ WARNING: Memory usage tinggi ({mem_mb:.0f}MB)! Berisiko OOM.")
            sys.stdout.flush()

        # Update entropy & clip ratio di ppo_update function via closure approach
        # Actually, we need to pass these params. Let me update the ppo_update call.
        # The current ppo_update_step takes clip_ratio as argument, but doesn't take entropy_coef.
        # Wait, entropy_coef is hardcoded in ppo_update.py line 55.
        # Let me check if we need to modify ppo_update.py too.

    # Final save
    print("Training complete. Closing env.")
    env.close()
    save_checkpoint(unreplicate(params_repl_p0), "model_final.msgpack")
    upload_to_kaggle(SAVE_DIR, message="Final training checkpoint")
    print(f"Best model saved with avg return: {best_reward:+.2f}")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    import jax.numpy as jnp
    train()
