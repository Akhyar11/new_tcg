import os
import sys

# Menambahkan root folder (new_tcg) ke dalam sys.path agar module 'agent_rl' terbaca
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"

import jax.numpy as jnp
import time
import jax
import optax
import numpy as np
from flax import serialization

from agent_rl.model import PokemonAgent
from agent_rl.vector_env import VectorEnv
from agent_rl.buffer import RolloutBuffer
from agent_rl.ppo_update import ppo_update_step, get_action_and_value

NUM_ENVS = 4
N_STEPS = 128
BATCH_SIZE = 64
EPOCHS = 4
TIMESTEPS_PER_PHASE = 1000000
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

def run_phase(phase_name, env, model, params, tx, opt_state, rng, global_step_start):
    buffer = RolloutBuffer(n_steps=N_STEPS, num_envs=NUM_ENVS)
    num_updates = TIMESTEPS_PER_PHASE // (N_STEPS * NUM_ENVS)
    
    print(f"\n=== {phase_name.upper()} ===")
    print(f"Target Update: {num_updates} iterasi. Timesteps: {TIMESTEPS_PER_PHASE}")
    
    obs = env.reset()
    next_seq, next_glob = obs["seq_input"], obs["glob_input"]
    next_done = np.zeros(NUM_ENVS, dtype=np.float32)
    
    global_step = global_step_start
    start_time = time.time()
    
    for update in range(1, num_updates + 1):
        ep_rewards = []
        ep_p0_wins = ep_p1_wins = ep_draws = 0
        ep_turns = []
        
        buffer.clear()
        
        for step in range(N_STEPS):
            global_step += NUM_ENVS
            rng, step_rng = jax.random.split(rng)
            actions, log_probs, values = get_action_and_value(params, model.apply, next_seq, next_glob, step_rng)
            actions_np = np.array(actions)
            
            next_obs, rewards, dones, infos = env.step(actions_np)
            
            for info in infos:
                if "turn" in info and info["turn"] > 0:
                    ep_turns.append(info["turn"])
            
            for i, d in enumerate(dones):
                if d:
                    if rewards[i] > 0.5: ep_p0_wins += 1
                    elif rewards[i] < -0.5: ep_p1_wins += 1
                    else: ep_draws += 1
            ep_rewards.extend(rewards)
            
            buffer.add(next_seq, next_glob, actions_np, np.array(log_probs), rewards, np.array(values), dones.astype(np.float32))
            next_seq, next_glob, next_done = next_obs["seq_input"], next_obs["glob_input"], dones.astype(np.float32)
            
        _, _, next_values = get_action_and_value(params, model.apply, next_seq, next_glob, rng)
        buffer.compute_returns_and_advantages(np.array(next_values), next_done, GAMMA, GAE_LAMBDA)
        
        mean_loss = 0.0
        update_count = 0
        
        for epoch in range(EPOCHS):
            for batch in buffer.get_batches(BATCH_SIZE):
                params, opt_state, loss, _ = ppo_update_step(params, opt_state, batch, model.apply, tx, clip_ratio=CLIP_RATIO)
                mean_loss += float(loss)
                update_count += 1
                
        mean_loss /= update_count
        
        if update % 1 == 0:
            total_games = ep_p0_wins + ep_p1_wins + ep_draws
            p0_wr = (ep_p0_wins / total_games * 100) if total_games > 0 else 0.0
            p1_wr = (ep_p1_wins / total_games * 100) if total_games > 0 else 0.0
            draw_wr = (ep_draws / total_games * 100) if total_games > 0 else 0.0
            avg_turns = (sum(ep_turns) / len(ep_turns)) if len(ep_turns) > 0 else 0.0
            fps = int((global_step - global_step_start) / (time.time() - start_time))
            
            print(f"[{phase_name}] Update {update:04d}/{num_updates} | Step: {global_step} | FPS: {fps} | "
                  f"Loss: {mean_loss:.4f} | P0 Win: {p0_wr:.1f}% | P1 Win: {p1_wr:.1f}% | "
                  f"Draw: {draw_wr:.1f}% | Avg Turns: {avg_turns:.1f}")
                  
        if update % 50 == 0:
            save_checkpoint(params, f"model_update_{update}.msgpack")
            
    return params, opt_state, global_step, rng

def train():
    print("=== INISIALISASI PELATIHAN PIPELINE (JAX) ===")
    rng = jax.random.PRNGKey(42)
    model = PokemonAgent(num_actions=250)
    rng, init_rng = jax.random.split(rng)
    
    dummy_seq = jnp.zeros((1, 93, 31))
    dummy_glob = jnp.zeros((1, 266))
    params = model.init(init_rng, dummy_seq, dummy_glob)
    
    # Menambahkan Gradient Clipping untuk menstabilkan PPO + Transformer
    tx = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate=LEARNING_RATE, eps=1e-5)
    )
    opt_state = tx.init(params)
    
    global_step = 0
    
    # ==========================================
    # FASE 1: MELAWAN RANDOM BOT (EKSPLORASI)
    # ==========================================
    print("\n[*] Memulai Fase 1: Melawan Random Bot")
    env_random = VectorEnv(num_envs=NUM_ENVS, deck_path="agent_rl/deck", is_self_play=False)
    params, opt_state, global_step, rng = run_phase(
        "FASE 1 (Random Bot)", env_random, model, params, tx, opt_state, rng, global_step
    )
    env_random.close()
    
    print("[*] Fase 1 Selesai. Menyimpan model peralihan...")
    save_checkpoint(params, "model_phase1_random.msgpack")
    
    # ==========================================
    # FASE 2: SELF-PLAY (PENGASAHAN TAKTIK)
    # ==========================================
    # Kita menggunakan model hasil Fase 1 sebagai checkpoint lawan
    save_checkpoint(params, "model_final.msgpack") # Agar vector_env memuat checkpoint ini
    
    print("\n[*] Memulai Fase 2: Self-Play (Melawan diri sendiri dari checkpoint Fase 1)")
    env_selfplay = VectorEnv(num_envs=NUM_ENVS, deck_path="agent_rl/deck", is_self_play=True)
    params, opt_state, global_step, rng = run_phase(
        "FASE 2 (Self-Play)", env_selfplay, model, params, tx, opt_state, rng, global_step
    )
    env_selfplay.close()
    
    print("\n[*] Seluruh Pipeline Pelatihan Selesai!")
    save_checkpoint(params, "model_final.msgpack")

if __name__ == "__main__":
    train()
