import os
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

# Konfigurasi Hyperparameter
NUM_ENVS = 8              # Jumlah klon CPU Game Engine paralel
N_STEPS = 128             # Jumlah langkah per klon sebelum AI melakukan update
BATCH_SIZE = 64           # Ukuran mini-batch saat update PPO
EPOCHS = 4                # Berapa kali mengulang belajar pada buffer yang sama
TOTAL_TIMESTEPS = 2048 # Target total langkah pengalaman (Untuk test cepat, nanti bisa diubah user)
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
    env = VectorEnv(num_envs=NUM_ENVS, deck_path="agent_rl/deck.csv")
    
    # 2. Inisiasi Model & Optimizer
    model = PokemonAgent(num_actions=250)
    rng, init_rng = jax.random.split(rng)
    
    dummy_seq = jnp.zeros((1, 93, 31))
    dummy_glob = jnp.zeros((1, 266))
    params = model.init(init_rng, dummy_seq, dummy_glob)
    
    tx = optax.adam(learning_rate=LEARNING_RATE)
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
            actions, log_probs, values = get_action_and_value(
                params, model.apply, next_seq, next_glob, step_rng
            )
            
            # Transfer actions ke numpy untuk VectorEnv (CPU)
            actions_np = np.array(actions)
            
            # Melangkah di dunia nyata (C++)
            next_obs, rewards, dones = env.step(actions_np)
            
            # Lacak Metrik (Hanya saat terminal)
            for i, d in enumerate(dones):
                if d:
                    # Reward +1.0 artinya menang mutlak di reward.py
                    ep_wins.append(1 if rewards[i] > 0.5 else 0)
            ep_rewards.extend(rewards)
            
            # Simpan jejak memori ke Buffer
            buffer.add(
                next_seq, next_glob, actions_np, np.array(log_probs), 
                rewards, np.array(values), next_done
            )
            
            next_seq = next_obs["seq_input"]
            next_glob = next_obs["glob_input"]
            next_done = dones.astype(np.float32)
            
        # Hitung Nilai Masa Depan (Bootstrap)
        _, _, next_values = get_action_and_value(params, model.apply, next_seq, next_glob, rng)
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
    import jax.numpy as jnp # pastikan jnp ter-load
    train()
