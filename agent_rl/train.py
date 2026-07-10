import os
import sys

# Menambahkan root folder (new_tcg) ke dalam sys.path agar module 'agent_rl' terbaca
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import jax
# pyrefly: ignore [missing-import]
import optax
import numpy as np
# pyrefly: ignore [missing-import]
from flax import serialization

from agent_rl.model import PokemonAgent
from agent_rl.vector_env import VectorEnv
from agent_rl.buffer import RolloutBuffer
from agent_rl.ppo_update import ppo_update_step, get_action_and_value
from flax.jax_utils import replicate, unreplicate

# Konfigurasi Hyperparameter
NUM_ENVS = 8              # Jumlah klon CPU Game Engine paralel
N_STEPS = 128             # Jumlah langkah per klon sebelum AI melakukan update
BATCH_SIZE = 64           # Ukuran mini-batch saat update PPO
EPOCHS = 4                # Berapa kali mengulang belajar pada buffer yang sama
TOTAL_TIMESTEPS = 1000000 # Target total langkah pengalaman (Bisa diatur ulang di Kaggle)
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RATIO = 0.2
SAVE_DIR = "checkpoints"

def save_checkpoint(params, filename):
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    path = os.path.join(SAVE_DIR, filename)
    with open(path, 'wb') as f:
        f.write(serialization.to_bytes(params))
    print(f"[*] Checkpoint tersimpan di {path}")

def train():
    print("=== INISIALISASI PELATIHAN PPO (JAX) ===")
    rng = jax.random.PRNGKey(42)
    
    # 1. Inisiasi Lingkungan Paralel
    print(f"Menjalankan {NUM_ENVS} Environment Pekerja secara Paralel...")
    env = VectorEnv(num_envs=NUM_ENVS, deck_path="agent_rl/deck")
    
    # 2. Inisiasi Model & Optimizer
    model = PokemonAgent(num_actions=250)
    rng, init_rng = jax.random.split(rng)
    
    dummy_seq = jnp.zeros((1, 93, 31))
    dummy_glob = jnp.zeros((1, 266))
    
    # Inisialisasi awal (acak)
    params = model.init(init_rng, dummy_seq, dummy_glob)
    
    # Coba muat checkpoint jika ada untuk Resume Training
    checkpoint_path = os.path.join(SAVE_DIR, "model_final.msgpack")
    if os.path.exists(checkpoint_path):
        print(f"[*] Melanjutkan (Resume) dari checkpoint: {checkpoint_path}")
        with open(checkpoint_path, 'rb') as f:
            params = serialization.from_bytes(params, f.read())
    else:
        print("[*] Mulai latihan dari awal (Bobot Acak).")
        
    tx = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate=LEARNING_RATE, eps=1e-5)
    )
    opt_state = tx.init(params)
    
    # REPLIKASI PARAMETER KE SELURUH GPU
    num_devices = jax.device_count()
    print(f"[*] Mendeteksi {num_devices} perangkat JAX (GPU/TPU). Mengaktifkan mode Multi-GPU (pmap).")
    params_repl = replicate(params)
    opt_state_repl = replicate(opt_state)
    
    # 3. Inisiasi Buffer
    buffer = RolloutBuffer(n_steps=N_STEPS, num_envs=NUM_ENVS)
    
    num_updates = TOTAL_TIMESTEPS // (N_STEPS * NUM_ENVS)
    print(f"Target Update: {num_updates} iterasi. Total Timesteps: {TOTAL_TIMESTEPS}")
    
    # Pre-reset environment
    obs = env.reset()
    next_seq = obs["seq_input"]
    next_glob = obs["glob_input"]
    next_done = np.zeros(NUM_ENVS, dtype=np.float32)
    
    global_step = 0
    start_time = time.time()
    env_step_counts = np.zeros(NUM_ENVS, dtype=np.int32)
    
    print("\n=== MEMULAI MAIN TRAINING LOOP ===")
    for update in range(1, num_updates + 1):
        # Array metrik sementara
        ep_rewards = []
        ep_wins_p0 = []
        ep_wins_p1 = []
        ep_steps = []
        
        # --- FASE 1: PENGUMPULAN PENGALAMAN (ROLLOUT) ---
        buffer.clear()
        
        for step in range(N_STEPS):
            global_step += NUM_ENVS
            # Inferensi Cepat JAX (Multi-GPU Sharding)
            rng, step_rng = jax.random.split(rng)
            step_rngs = jax.random.split(step_rng, num_devices)
            
            next_seq_sharded = next_seq.reshape((num_devices, NUM_ENVS // num_devices, *next_seq.shape[1:]))
            next_glob_sharded = next_glob.reshape((num_devices, NUM_ENVS // num_devices, *next_glob.shape[1:]))
            
            _, _, values_sharded, logits_sharded = get_action_and_value(
                params_repl, model.apply, next_seq_sharded, next_glob_sharded, step_rngs
            )

            # Transfer logits ke numpy (CPU)
            logits_np = np.array(logits_sharded).reshape((NUM_ENVS, -1))
            values_np = np.array(values_sharded).reshape((NUM_ENVS,))

            # Melangkah: worker menerima raw logits, melakukan categorical sampling
            # tanpa pengembalian sesuai minCount, lalu eksekusi di C++ engine.
            next_obs, rewards, dones, infos = env.step(logits_np)
            
            # Lacak Metrik (Hanya saat terminal)
            env_step_counts += 1

            for i, d in enumerate(dones):
                if d:
                    result = infos[i].get("result", -1)
                    if result == 0:
                        ep_wins_p0.append(1)
                        ep_wins_p1.append(0)
                    elif result == 1:
                        ep_wins_p0.append(0)
                        ep_wins_p1.append(1)
                    ep_steps.append(env_step_counts[i])
                    env_step_counts[i] = 0

            ep_rewards.extend(rewards)

            # Ekstrak actions_mask dan glob_mask dari infos
            actions_mask_np = np.stack([info["actions_mask"] for info in infos])
            glob_mask_np = np.stack([info["glob_mask"] for info in infos])

            # Hitung old_log_probs: masking logits sama seperti di ppo_update loss
            # agar training dan inference konsisten
            masked_logits_np = logits_np - 1e9 * (1.0 - glob_mask_np)
            logits_max = np.max(masked_logits_np, axis=-1, keepdims=True)
            log_sum_exp = np.log(np.sum(np.exp(masked_logits_np - logits_max), axis=-1, keepdims=True))
            log_probs_all_np = (masked_logits_np - logits_max) - log_sum_exp

            # MEAN log-prob dari actions yang benar-benar di-sampling
            mask_count = np.maximum(1.0, np.sum(actions_mask_np, axis=-1))
            multi_log_probs = np.sum(log_probs_all_np * actions_mask_np, axis=-1) / mask_count

            # Simpan jejak memori ke Buffer
            buffer.add(
                next_seq, next_glob, actions_mask_np, multi_log_probs,
                rewards, values_np, dones.astype(np.float32)
            )

            next_seq = next_obs["seq_input"]
            next_glob = next_obs["glob_input"]
            next_done = dones.astype(np.float32)
            
        # Hitung Nilai Masa Depan (Bootstrap)
        next_seq_sharded = next_seq.reshape((num_devices, NUM_ENVS // num_devices, *next_seq.shape[1:]))
        next_glob_sharded = next_glob.reshape((num_devices, NUM_ENVS // num_devices, *next_glob.shape[1:]))
        step_rngs = jax.random.split(rng, num_devices)
        
        _, _, next_values_sharded, _ = get_action_and_value(params_repl, model.apply, next_seq_sharded, next_glob_sharded, step_rngs)
        next_values = np.array(next_values_sharded).reshape((NUM_ENVS,))
        buffer.compute_returns_and_advantages(next_values, next_done, GAMMA, GAE_LAMBDA)
        
        # --- FASE 2: OPTIMASI GRADIENT (PPO UPDATE) ---
        mean_loss = 0.0
        update_count = 0
        
        for epoch in range(EPOCHS):
            for batch in buffer.get_batches(BATCH_SIZE):
                # Shard batch untuk Multi-GPU
                batch_sharded = {k: v.reshape((num_devices, BATCH_SIZE // num_devices, *v.shape[1:])) for k, v in batch.items()}
                
                params_repl, opt_state_repl, loss, _ = ppo_update_step(
                    params_repl, opt_state_repl, batch_sharded, model.apply, tx, CLIP_RATIO
                )
                # Ambil scalar loss dari GPU pertama (karena pmean membuat loss identik di semua GPU)
                mean_loss += float(loss[0])
                update_count += 1
                
        mean_loss /= update_count
        
        # --- FASE 3: MONITORING & CHECKPOINT ---
        if update % 1 == 0:
            avg_rew = np.mean(ep_rewards) if ep_rewards else 0.0
            win_p0 = (np.mean(ep_wins_p0) * 100) if ep_wins_p0 else 0.0
            win_p1 = (np.mean(ep_wins_p1) * 100) if ep_wins_p1 else 0.0
            games_played = len(ep_wins_p0)
            avg_steps = (np.mean(ep_steps) / max(1, games_played)) if ep_steps else 0.0
            
            fps = int((NUM_ENVS * N_STEPS) / (time.time() - start_time + 1e-8))
            start_time = time.time()
            
            print(f"Update {update:04d}/{num_updates} | Step: {global_step} | FPS: {fps} | Games: {games_played} | AvgStep/Game: {avg_steps:.0f} | "
                  f"Loss: {mean_loss:.4f} | Win P0: {win_p0:.1f}% | Win P1: {win_p1:.1f}%")
                  
        if update % 50 == 0:
            save_checkpoint(unreplicate(params_repl), f"model_update_{update}.msgpack")
            
    print("Pelatihan selesai. Menutup lingkungan.")
    env.close()
    save_checkpoint(unreplicate(params_repl), "model_final.msgpack")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    import jax.numpy as jnp # pastikan jnp ter-load
    train()
