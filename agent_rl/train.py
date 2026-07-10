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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"

# Konfigurasi Hyperparameter
NUM_ENVS = 8              # Jumlah klon CPU Game Engine paralel
N_STEPS = 128             # Jumlah langkah per klon sebelum AI melakukan update
BATCH_SIZE = 64           # Ukuran mini-batch saat update PPO
EPOCHS = 4                # Berapa kali mengulang belajar pada buffer yang sama
TOTAL_TIMESTEPS = 5000000 # Target total langkah pengalaman (Bisa diatur ulang di Kaggle)
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
    # Mempertahankan implementasi 10 deck dari versi saat ini (direktori deck)
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
        
    # Tetap gunakan gradient clipping dari perbaikan sebelumnya untuk mencegah Policy Collapse
    tx = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate=LEARNING_RATE, eps=1e-5)
    )
    opt_state = tx.init(params)
    
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
    
    print("\n=== MEMULAI MAIN TRAINING LOOP ===")
    for update in range(1, num_updates + 1):
        # Array metrik sementara
        ep_rewards = []
        ep_wins = []
        
        # --- FASE 1: PENGUMPULAN PENGALAMAN (ROLLOUT) ---
        buffer.clear()
        
        for step in range(N_STEPS):
            global_step += NUM_ENVS
            
            # Inferensi Cepat JAX
            rng, step_rng = jax.random.split(rng)
            actions, log_probs, values, logits = get_action_and_value(
                params, model.apply, next_seq, next_glob, step_rng
            )
            
            # Transfer actions dan logits ke numpy untuk VectorEnv (CPU)
            actions_np = np.array(actions)
            logits_np = np.array(logits)
            
            # Urutkan dan ambil top 10 aksi terbaik (Untuk memperkecil overhead IPC Pipe)
            top_actions_np = np.argsort(logits_np, axis=-1)[:, ::-1][:, :10]
            
            # Melangkah di dunia nyata (C++)
            # Mengakomodasi return infos dari VectorEnv versi baru
            next_obs, rewards, dones, infos = env.step(actions_np, top_actions_np)
            
            # Lacak Metrik (Hanya saat terminal)
            for i, d in enumerate(dones):
                if d:
                    # Reward +1.0 artinya menang mutlak di reward.py
                    ep_wins.append(1 if rewards[i] > 0.5 else 0)
            ep_rewards.extend(rewards)
            
            # Ekstrak actions_mask dari infos
            actions_mask_np = np.stack([info["actions_mask"] for info in infos])
            
            # Hitung old_log_probs aktual menggunakan NumPy MURNI (mencegah overhead Eager Dispatch JAX)
            logits_max = np.max(logits_np, axis=-1, keepdims=True)
            log_sum_exp = np.log(np.sum(np.exp(logits_np - logits_max), axis=-1, keepdims=True))
            log_probs_all_np = (logits_np - logits_max) - log_sum_exp
            multi_log_probs = np.sum(log_probs_all_np * actions_mask_np, axis=-1)
            
            # Simpan jejak memori ke Buffer
            buffer.add(
                next_seq, next_glob, actions_mask_np, multi_log_probs, 
                rewards, np.array(values), dones.astype(np.float32)
            )
            
            next_seq = next_obs["seq_input"]
            next_glob = next_obs["glob_input"]
            next_done = dones.astype(np.float32)
            
        # Hitung Nilai Masa Depan (Bootstrap)
        _, _, next_values, _ = get_action_and_value(params, model.apply, next_seq, next_glob, rng)
        buffer.compute_returns_and_advantages(np.array(next_values), next_done, GAMMA, GAE_LAMBDA)
        
        # --- FASE 2: OPTIMASI GRADIENT (PPO UPDATE) ---
        mean_loss = 0.0
        update_count = 0
        
        for epoch in range(EPOCHS):
            for batch in buffer.get_batches(BATCH_SIZE):
                params, opt_state, loss, _ = ppo_update_step(
                    params, opt_state, batch, model.apply, tx, clip_ratio=CLIP_RATIO
                )
                mean_loss += float(loss)
                update_count += 1
                
        mean_loss /= update_count
        
        # --- FASE 3: MONITORING & CHECKPOINT ---
        if update % 1 == 0:
            avg_rew = np.mean(ep_rewards) if ep_rewards else 0.0
            win_rate = (np.mean(ep_wins) * 100) if ep_wins else 0.0
            fps = int(global_step / (time.time() - start_time))
            
            print(f"Update {update:04d}/{num_updates} | Step: {global_step} | FPS: {fps} | "
                  f"Loss: {mean_loss:.4f} | Avg Reward: {avg_rew:.4f} | Win Rate: {win_rate:.1f}%")
                  
        if update % 50 == 0:
            save_checkpoint(params, f"model_update_{update}.msgpack")
            
    print("Pelatihan selesai. Menutup lingkungan.")
    env.close()
    save_checkpoint(params, "model_final.msgpack")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    import jax.numpy as jnp # pastikan jnp ter-load
    train()
