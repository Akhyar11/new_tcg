#!/usr/bin/env python3
"""
PPO Training script with 3-Phase Curriculum for Pointer LSTM Agent:

Phase 1:
  - Agent (P0): Pointer LSTM Model (initialized from FF model_final + Xenova 32D)
  - Opponent (P1): FeedForward Model (from model_final.msgpack)
  - Objective: P0 achieves >= 65% winrate over a 150-game window to transition.

Phase 2:
  - Agent (P0): Pointer LSTM Model (retains trained weights from Phase 1)
  - Opponent (P1): Classic LSTM Model (from model_lstm_final.msgpack)
  - Objective: P0 achieves >= 65% winrate over a 150-game window to transition.

Phase 3:
  - Agent (P0): Pointer LSTM Model (retains trained weights from Phase 2)
  - Opponent (P1): Pointer LSTM Model (initially copies P0's weights, frozen)
  - Objective: P0 achieves >= 65% winrate over a 150-game window.
  - Transition: On success, update P1 weights with P0, save checkpoints, upload to Kaggle, and reset window.
"""
import os
import sys
import time
from collections import deque
import json
from dotenv import load_dotenv

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
env_path = os.path.join(root_dir, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    load_dotenv()

sys.path.append(root_dir)

import jax
import jax.numpy as jnp
import optax
import numpy as np
from flax import serialization
import psutil
from functools import partial

# Import architectures
from agent_rl_pointer.model import PokemonAgent as LSTMModel
from agent_rl.model import PokemonAgent as FFModel
from agent_rl_lstm.model import PokemonAgent as ClassicLSTMModel

from agent_rl_pointer.vector_env import VectorEnv
from agent_rl_pointer.buffer import RolloutBuffer
from agent_rl_pointer.ppo_update import ppo_update_step
from flax.jax_utils import replicate, unreplicate

# --- Config ---
NUM_ENVS = int(os.environ.get("RL_NUM_ENVS", "8"))
N_STEPS = 128
BATCH_SIZE = int(os.environ.get("RL_BATCH_SIZE", "64"))
TOTAL_TIMESTEPS = int(os.environ.get("TOTAL_TIMESTEPS", "20000000"))
LEARNING_RATE = 1e-5
ENTROPY_COEF = 0.05
EPOCHS = 1
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RATIO = 0.2

SAVE_DIR = os.environ.get("SAVE_DIR", "tcg_models")
NEW_DECK_PATH = os.environ.get("NEW_DECK_PATH", "new_deck")
GEN_DECK_PATH = os.environ.get("GEN_DECK_PATH", "deck_generated")

# Window & Target settings
WIN_WINDOW = 150
WIN_TARGET = 0.65

# --- Custom JAX pmap runners ---
@partial(jax.pmap, static_broadcasted_argnums=(1,), axis_name='gpu')
def get_action_and_value_lstm(params, apply_fn, seq_input, glob_input, carry, key):
    logits, values, new_carry = apply_fn(params, seq_input, glob_input, carry)
    action_mask = glob_input[..., 16:266]
    masked_logits = logits + ((1.0 - action_mask) * -1e9)
    actions = jax.random.categorical(key, masked_logits, axis=-1)
    log_probs_all = jax.nn.log_softmax(masked_logits)
    log_probs = jnp.take_along_axis(log_probs_all, actions[..., None], axis=-1).squeeze(-1)
    return actions, log_probs, values.squeeze(-1), logits, new_carry

@partial(jax.pmap, static_broadcasted_argnums=(1,), axis_name='gpu')
def get_action_and_value_ff(params, apply_fn, seq_input, glob_input, key):
    logits, values = apply_fn(params, seq_input, glob_input)
    action_mask = glob_input[..., 16:266]
    masked_logits = logits + ((1.0 - action_mask) * -1e9)
    actions = jax.random.categorical(key, masked_logits, axis=-1)
    log_probs_all = jax.nn.log_softmax(masked_logits)
    log_probs = jnp.take_along_axis(log_probs_all, actions[..., None], axis=-1).squeeze(-1)
    return actions, log_probs, values.squeeze(-1), logits

def save_checkpoint(params, filename):
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    path = os.path.join(SAVE_DIR, filename)
    with open(path, 'wb') as f:
        f.write(serialization.to_bytes(params))
    print(f"[*] Checkpoint saved: {path}")

def load_matching_weights(target_params, bytes_data):
    try:
        old_params = serialization.msgpack_restore(bytes_data)
        def update_dict(target_d, old_d):
            for k, v in old_d.items():
                if k in target_d:
                    if isinstance(v, dict):
                        update_dict(target_d[k], v)
                    else:
                        if getattr(target_d[k], "shape", None) == v.shape:
                            target_d[k] = v
            return target_d
            
        from flax.core import unfreeze, freeze
        unfrozen = unfreeze(target_params)
        updated = update_dict(unfrozen, old_params)
        return freeze(updated), True
    except Exception as e:
        print(f"[!] Gagal meload secara parsial: {e}")
        return target_params, False

def get_kaggle_api():
    os.environ.pop("KAGGLE_API_TOKEN", None)
    os.environ.pop("KAGGLE_API_V1_TOKEN", None)
    os.environ.pop("KAGGLE_KERNEL_RUN_TYPE", None)
    os.environ.pop("KAGGLE_DATA_PROXY_URL", None)
    os.environ["KAGGLE_USERNAME"] = "akhyarsafrudin"
    os.environ["KAGGLE_KEY"] = "03c3e536ffedc7d6153c1b3b8515242b"
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api

def download_from_kaggle(save_dir):
    dataset_id = "akhyarsafrudin/tcg-models"
    print(f"[*] Mencoba mendownload checkpoint dari Kaggle Dataset ({dataset_id}) menggunakan Python API...")
    try:
        api = get_kaggle_api()
        api.dataset_download_files(dataset_id, path=save_dir, unzip=True)
        print("[*] Sukses mendownload dan unzip model dari Kaggle.")
    except Exception as e:
        print(f"[!] Terjadi error saat download Kaggle: {e}")

def upload_to_kaggle(save_dir, message="Update models"):
    if os.environ.get("NO_KAGGLE_UPLOAD", "0") == "1":
        print("[*] Kaggle Upload dinonaktifkan (NO_KAGGLE_UPLOAD=1). Skip upload.")
        return
    dataset_id = "akhyarsafrudin/tcg-models"
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
        dataset_exists = True
        try:
            print(f"[*] Memeriksa status dataset Kaggle ({dataset_id})...")
            api.dataset_status(dataset_id)
        except Exception as status_err:
            status_code = getattr(status_err, 'status', None)
            err_msg = str(status_err).lower()
            if status_code in (401, 403) or "unauthorized" in err_msg or "unauthenticated" in err_msg or "401" in err_msg:
                print(f"[!] Kaggle Authentication Error: Kredensial tidak valid.")
                return
            if status_code == 404 or "404" in err_msg or "not found" in err_msg:
                dataset_exists = False
                print(f"[*] Dataset belum ada di Kaggle. Akan mencoba membuat dataset baru.")
            else:
                dataset_exists = False
                print(f"[*] Dataset tidak dapat diakses. Mencoba membuat dataset baru.")

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

def main():
    num_devices = jax.device_count()
    print(f"[*] Running on {num_devices} GPU(s).")
    
    # Initialize VectorEnv
    env = VectorEnv(num_envs=NUM_ENVS, new_deck_path=NEW_DECK_PATH, gen_deck_path=GEN_DECK_PATH)
    rng = jax.random.PRNGKey(42)

    # Initialize model architectures
    lstm_model = LSTMModel(num_actions=250)
    ff_model = FFModel(num_actions=250)
    classic_lstm_model = ClassicLSTMModel(num_actions=250)
    
    rng, init_rng = jax.random.split(rng)
    
    # Dummy tensors
    dummy_seq_lstm = jnp.zeros((1, 173, 31))
    dummy_glob = jnp.zeros((1, 266))
    dummy_carry = (jnp.zeros((1, 256)), jnp.zeros((1, 256)))
    
    # P0 is ALWAYS Pointer LSTM Model
    params_p0 = lstm_model.init(init_rng, dummy_seq_lstm, dummy_glob, dummy_carry)
    
    # Initial P1 is Feed-Forward Model (starts at Phase 1)
    dummy_seq_ff = jnp.zeros((1, 173, 31))
    params_p1 = ff_model.init(init_rng, dummy_seq_ff, dummy_glob)

    # Download weights from Kaggle
    download_from_kaggle(SAVE_DIR)

    # --- Load weights for P0 ---
    model_lstm_final_path = os.path.join(SAVE_DIR, "model_lstm_pointer_final.msgpack")
    model_lstm_base_path = os.path.join(SAVE_DIR, "model_lstm_pointer_base.msgpack")

    if os.path.exists(model_lstm_final_path):
        print(f"[*] Loading LSTM P0 from {model_lstm_final_path}")
        with open(model_lstm_final_path, 'rb') as f:
            params_p0, _ = load_matching_weights(params_p0, f.read())
    elif os.path.exists(model_lstm_base_path):
        print(f"[*] Loading LSTM P0 from {model_lstm_base_path}")
        with open(model_lstm_base_path, 'rb') as f:
            params_p0, _ = load_matching_weights(params_p0, f.read())

    # --- Inject Distilled Embeddings ---
    distill_path = os.path.join(root_dir, "knowledge_distillation", "student_embeddings_32d.npy")
    if os.path.exists(distill_path):
        print(f"[*] Injecting pre-trained embeddings from {distill_path} into P0 LSTM model...")
        knowledge_weights = np.load(distill_path)
        vocab_size = params_p0['params']['CardEmbedding_0']['knowledge_embed']['embedding'].shape[0]
        embed_dim = params_p0['params']['CardEmbedding_0']['knowledge_embed']['embedding'].shape[1]
        
        padded_weights = np.zeros((vocab_size, embed_dim))
        num_cards = min(knowledge_weights.shape[0], vocab_size)
        padded_weights[:num_cards, :] = knowledge_weights[:num_cards, :]

        from flax.core import unfreeze, freeze
        params_p0_mut = unfreeze(params_p0)
        params_p0_mut['params']['CardEmbedding_0']['knowledge_embed']['embedding'] = jnp.array(padded_weights)
        params_p0 = freeze(params_p0_mut)

    # Initialize Optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adamw(learning_rate=LEARNING_RATE, eps=1e-5, weight_decay=1e-4)
    )
    opt_state = tx.init(params_p0)

    # Replicate parameters
    params_repl_p0 = replicate(params_p0)
    opt_state_repl = replicate(opt_state)

    # Buffer & states
    buffer = RolloutBuffer(n_steps=N_STEPS, num_envs=NUM_ENVS)
    obs = env.reset()
    next_seq = obs["seq_input"]
    next_glob = obs["glob_input"]
    next_done = np.zeros(NUM_ENVS, dtype=np.float32)

    # LSTM Carry tracker
    carry_c = np.zeros((NUM_ENVS, 256), dtype=np.float32)
    carry_h = np.zeros((NUM_ENVS, 256), dtype=np.float32)
    carry_repl_p0 = (
        jnp.array(carry_c).reshape(num_devices, NUM_ENVS // num_devices, 256),
        jnp.array(carry_h).reshape(num_devices, NUM_ENVS // num_devices, 256)
    )
    
    # carry_repl_p1 is initialized if P1 uses an LSTM model (Phase 2 & 3)
    carry_repl_p1 = (
        jnp.array(carry_c).reshape(num_devices, NUM_ENVS // num_devices, 256),
        jnp.array(carry_h).reshape(num_devices, NUM_ENVS // num_devices, 256)
    )

    current_active_players = np.zeros(NUM_ENVS, dtype=np.int32)
    env_step_counts = np.zeros(NUM_ENVS, dtype=np.int32)
    episodic_returns = np.zeros(NUM_ENVS, dtype=np.float32)

    # Sliding win rate window
    recent_wins = deque(maxlen=WIN_WINDOW)
    recent_steps = deque(maxlen=WIN_WINDOW)
    total_games = 0

    reward_running_mean = 0.0
    reward_running_std = 1.0
    reward_norm_steps = 0

    num_updates = TOTAL_TIMESTEPS // (N_STEPS * NUM_ENVS)
    
    # Curriculum Phase Configuration
    PHASE = int(os.environ.get("TRAIN_PHASE", "1"))
    assert PHASE in [1, 2, 3], "TRAIN_PHASE must be 1, 2, or 3"
    
    # Helper function to configure a Phase
    def configure_phase(phase_num):
        nonlocal params_repl_p0
        print(f"\n==========================================")
        print(f"🚀 INISIALISASI PHASE {phase_num}")
        print(f"==========================================")
        sys.stdout.flush()
        
        # Load parameters for P1
        if phase_num == 1:
            # P1 = FF model (from model_final.msgpack)
            ff_path = os.path.join(SAVE_DIR, "model_final.msgpack")
            p1_params = ff_model.init(init_rng, jnp.zeros((1, 173, 31)), dummy_glob)
            if os.path.exists(ff_path):
                print(f"[*] Loading FF weights for P1 from {ff_path}...")
                p1_params, _ = load_matching_weights(p1_params, open(ff_path, 'rb').read())
            else:
                print(f"[!] WARNING: {ff_path} tidak ditemukan! Menggunakan bobot acak.")
            
        elif phase_num == 2:
            # P1 = Classic LSTM (from model_lstm_final.msgpack)
            classic_path = os.path.join(SAVE_DIR, "model_lstm_final.msgpack")
            p1_params = classic_lstm_model.init(init_rng, dummy_seq_lstm, dummy_glob, dummy_carry)
            if os.path.exists(classic_path):
                print(f"[*] Loading Classic LSTM weights for P1 from {classic_path}...")
                p1_params, _ = load_matching_weights(p1_params, open(classic_path, 'rb').read())
            else:
                print(f"[!] WARNING: {classic_path} tidak ditemukan! Menggunakan bobot acak.")
                
        else:  # Phase 3
            # P1 = Pointer LSTM (copies current P0 weights)
            print(f"[*] Loading Pointer LSTM weights for P1 (Copying current P0 weights)...")
            p1_params = unreplicate(params_repl_p0)
            
        # Replicate to GPU
        nonlocal params_repl_p1
        params_repl_p1 = replicate(p1_params)
        
        # Reset trackers
        recent_wins.clear()
        recent_steps.clear()
        
        # Reset carry for P1
        nonlocal carry_repl_p1
        carry_c.fill(0)
        carry_h.fill(0)
        carry_repl_p1 = (
            jnp.array(carry_c).reshape(num_devices, NUM_ENVS // num_devices, 256),
            jnp.array(carry_h).reshape(num_devices, NUM_ENVS // num_devices, 256)
        )
        print(f"[*] Phase {phase_num} terkonfigurasi. Target Winrate: {WIN_TARGET*100}% over window {WIN_WINDOW} games.")
        sys.stdout.flush()

    # Initial Phase Setup
    params_repl_p1 = None
    configure_phase(PHASE)

    global_step = 0
    p1_update_count = 0
    start_time = time.time()

    for update in range(1, num_updates + 1):
        progress = update / num_updates
        current_entropy_coef = max(0.005, ENTROPY_COEF * (1.0 - progress * 0.9))
        current_clip_ratio = max(0.05, CLIP_RATIO * (1.0 - progress * 0.75))

        ep_returns = []
        ep_wins = []
        buffer.clear()

        # Rollout Phase
        for step in range(N_STEPS):
            global_step += NUM_ENVS
            rng, step_rng = jax.random.split(rng)
            step_rngs = jax.random.split(step_rng, num_devices)

            next_seq_sharded = next_seq.reshape((num_devices, NUM_ENVS // num_devices, *next_seq.shape[1:]))
            next_glob_sharded = next_glob.reshape((num_devices, NUM_ENVS // num_devices, *next_glob.shape[1:]))

            # Store carry state before stepping
            old_carry_c_np = np.array(carry_repl_p0[0]).reshape((NUM_ENVS, 256)).copy()
            old_carry_h_np = np.array(carry_repl_p0[1]).reshape((NUM_ENVS, 256)).copy()

            # P0 (Pointer LSTM) inference
            _, _, values_sharded_p0, logits_sharded_p0, carry_repl_p0 = get_action_and_value_lstm(
                params_repl_p0, lstm_model.apply, next_seq_sharded, next_glob_sharded, carry_repl_p0, step_rngs
            )

            # P1 inference (FF or Classic LSTM or Pointer LSTM)
            if PHASE == 1:
                _, _, values_sharded_p1, logits_sharded_p1 = get_action_and_value_ff(
                    params_repl_p1, ff_model.apply, next_seq_sharded, next_glob_sharded, step_rngs
                )
            elif PHASE == 2:
                _, _, values_sharded_p1, logits_sharded_p1, carry_repl_p1 = get_action_and_value_lstm(
                    params_repl_p1, classic_lstm_model.apply, next_seq_sharded, next_glob_sharded, carry_repl_p1, step_rngs
                )
            else:  # Phase 3
                _, _, values_sharded_p1, logits_sharded_p1, carry_repl_p1 = get_action_and_value_lstm(
                    params_repl_p1, lstm_model.apply, next_seq_sharded, next_glob_sharded, carry_repl_p1, step_rngs
                )

            logits_np_p0 = np.array(logits_sharded_p0).reshape((NUM_ENVS, -1))
            logits_np_p1 = np.array(logits_sharded_p1).reshape((NUM_ENVS, -1))
            values_np_p0 = np.array(values_sharded_p0).reshape((NUM_ENVS,))
            values_np_p1 = np.array(values_sharded_p1).reshape((NUM_ENVS,))

            # Fusion
            logits_np = np.where(current_active_players[:, None] == 0, logits_np_p0, logits_np_p1)
            values_np = np.where(current_active_players == 0, values_np_p0, values_np_p1)

            next_obs, rewards, dones, infos = env.step(logits_np)
            rewards = np.nan_to_num(rewards, nan=0.0, posinf=1.0, neginf=-1.0)

            # Normalization
            reward_norm_steps += NUM_ENVS
            for r in rewards:
                delta = r - reward_running_mean
                reward_running_mean += delta / max(reward_norm_steps, 1)
                delta2 = r - reward_running_mean
                reward_running_std += delta * delta2
            running_std = np.sqrt(reward_running_std / max(reward_norm_steps, 1))
            running_std = max(running_std, 0.01)
            normalized_rewards = np.clip(rewards / running_std, -5.0, 5.0)

            episodic_returns += rewards
            env_step_counts += 1

            carry_c_np_p0 = np.array(carry_repl_p0[0]).reshape((NUM_ENVS, 256))
            carry_h_np_p0 = np.array(carry_repl_p0[1]).reshape((NUM_ENVS, 256))
            if PHASE in [2, 3]:
                carry_c_np_p1 = np.array(carry_repl_p1[0]).reshape((NUM_ENVS, 256))
                carry_h_np_p1 = np.array(carry_repl_p1[1]).reshape((NUM_ENVS, 256))

            for i, d in enumerate(dones):
                if d:
                    ep_returns.append(float(episodic_returns[i]))
                    episodic_returns[i] = 0.0
                    result = infos[i].get("result", -1)
                    
                    is_win = -1
                    if result == 0:
                        is_win = 1
                    elif result == 1:
                        is_win = 0
                        
                    if is_win != -1:
                        ep_wins.append(is_win)
                        
                    # Save the game length (steps)
                    recent_steps.append(int(env_step_counts[i]))
                    total_games += 1
                    env_step_counts[i] = 0
                    
                    # Reset Carry
                    carry_c_np_p0[i] = 0.0
                    carry_h_np_p0[i] = 0.0
                    if PHASE in [2, 3]:
                        carry_c_np_p1[i] = 0.0
                        carry_h_np_p1[i] = 0.0

            carry_repl_p0 = (
                jnp.array(carry_c_np_p0).reshape(num_devices, NUM_ENVS // num_devices, 256),
                jnp.array(carry_h_np_p0).reshape(num_devices, NUM_ENVS // num_devices, 256)
            )
            if PHASE in [2, 3]:
                carry_repl_p1 = (
                    jnp.array(carry_c_np_p1).reshape(num_devices, NUM_ENVS // num_devices, 256),
                    jnp.array(carry_h_np_p1).reshape(num_devices, NUM_ENVS // num_devices, 256)
                )

            actions_mask_np = np.stack([info["actions_mask"] for info in infos])
            glob_mask_np = np.stack([info["glob_mask"] for info in infos])

            # Old log probabilities
            masked_logits_np = logits_np - 1e9 * (1.0 - glob_mask_np)
            logits_max = np.max(masked_logits_np, axis=-1, keepdims=True)
            log_sum_exp = np.log(np.sum(np.exp(masked_logits_np - logits_max), axis=-1, keepdims=True))
            log_probs_all_np = (masked_logits_np - logits_max) - log_sum_exp

            mask_count = np.maximum(1.0, np.sum(actions_mask_np, axis=-1))
            multi_log_probs = np.sum(log_probs_all_np * actions_mask_np, axis=-1) / mask_count
            
            turn_changed_np = np.stack([info["turn_changed"] for info in infos])

            buffer.add(
                next_seq, next_glob, actions_mask_np, multi_log_probs,
                normalized_rewards, values_np, dones.astype(np.float32), turn_changed_np,
                current_active_players, old_carry_c_np, old_carry_h_np
            )
            
            current_active_players = np.stack([info["active_player"] for info in infos])
            next_seq = next_obs["seq_input"]
            next_glob = next_obs["glob_input"]
            next_done = dones.astype(np.float32)

        # GAE
        next_seq_sharded = next_seq.reshape((num_devices, NUM_ENVS // num_devices, *next_seq.shape[1:]))
        next_glob_sharded = next_glob.reshape((num_devices, NUM_ENVS // num_devices, *next_glob.shape[1:]))
        step_rngs = jax.random.split(rng, num_devices)

        _, _, next_values_sharded, _, _ = get_action_and_value_lstm(
            params_repl_p0, lstm_model.apply, next_seq_sharded, next_glob_sharded, carry_repl_p0, step_rngs
        )
        next_values = np.array(next_values_sharded).reshape((NUM_ENVS,))
        buffer.compute_returns_and_advantages(next_values, next_done, GAMMA, GAE_LAMBDA)

        # PPO Update
        mean_loss = 0.0
        update_count = 0
        params_before = params_repl_p0
        opt_state_before = opt_state_repl

        for epoch in range(EPOCHS):
            for batch in buffer.get_batches(BATCH_SIZE, seq_len=32):
                num_seqs_per_batch = BATCH_SIZE // 32
                batch_sharded = {
                    k: v.reshape((num_devices, num_seqs_per_batch // num_devices, *v.shape[1:]))
                    for k, v in batch.items()
                }

                params_repl_p0, opt_state_repl, loss, _ = ppo_update_step(
                    params_repl_p0, opt_state_repl, batch_sharded, lstm_model.apply, tx,
                    current_clip_ratio, current_entropy_coef
                )
                mean_loss += float(loss[0])
                update_count += 1

        mean_loss /= update_count

        if not np.isfinite(mean_loss):
            print(f"  [NaN Guard] Loss NaN/Inf! Rollback.")
            params_repl_p0 = params_before
            opt_state_repl = opt_state_before
            mean_loss = 0.0

        # Winrate & updates tracking
        if update % 1 == 0:
            recent_wins.extend(ep_wins)
            avg_winrate = np.mean(recent_wins) * 100 if len(recent_wins) > 0 else 0.0
            avg_steps = np.mean(recent_steps) if len(recent_steps) > 0 else 0.0
            fps = int((NUM_ENVS * N_STEPS) / (time.time() - start_time + 1e-8))
            start_time = time.time()

            print(f"Update {update:04d}/{num_updates} | Loss: {mean_loss:.4f} | Window WR: {avg_winrate:.1f}% ({len(recent_wins)}/{WIN_WINDOW}) | Phase: {PHASE} | P1 Updates: {p1_update_count} | Games (Total): {total_games} | Avg Steps: {avg_steps:.1f} | FPS: {fps}")
            sys.stdout.flush()

            # Target checks (Objective >= 65% winrate)
            if len(recent_wins) >= WIN_WINDOW:
                if avg_winrate >= (WIN_TARGET * 100):
                    print(f"\n🎉 WINRATE TARGET DICAPAI DI PHASE {PHASE}! Window WR: {avg_winrate:.1f}% >= {WIN_TARGET*100}%!")
                    sys.stdout.flush()
                    
                    if PHASE == 1:
                        # Transition Phase 1 -> Phase 2
                        save_checkpoint(unreplicate(params_repl_p0), "model_lstm_pointer_final.msgpack")
                        upload_to_kaggle(SAVE_DIR, message="Phase 1 complete (Pointer LSTM vs FF). Winrate >= 65%.")
                        PHASE = 2
                        configure_phase(PHASE)
                        
                    elif PHASE == 2:
                        # Transition Phase 2 -> Phase 3
                        save_checkpoint(unreplicate(params_repl_p0), "model_lstm_pointer_final.msgpack")
                        upload_to_kaggle(SAVE_DIR, message="Phase 2 complete (Pointer LSTM vs Classic LSTM). Winrate >= 65%.")
                        PHASE = 3
                        configure_phase(PHASE)
                        
                    else:  # Phase 3
                        # Self-play target achieved: Update P1 (Frozen opponent) with P0 (Agent)
                        p1_update_count += 1
                        print(f"🔥 [P1 Weights Update #{p1_update_count}] Meng-update parameter P1 dengan model P0 saat ini.")
                        params_repl_p1 = params_repl_p0
                        save_checkpoint(unreplicate(params_repl_p0), "model_lstm_pointer_final.msgpack")
                        upload_to_kaggle(SAVE_DIR, message=f"P1 Update #{p1_update_count} (Winrate target achieved in Phase 3 self-play)")
                        recent_wins.clear()
                        print("[*] Window WR di-reset. Melanjutkan self-play training Phase 3...")
                        sys.stdout.flush()

        if update % 100 == 0:
            mem_mb = psutil.Process().memory_info().rss / 1e6
            print(f"  [MEM] RSS={mem_mb:.0f}MB")
            sys.stdout.flush()

    env.close()
    save_checkpoint(unreplicate(params_repl_p0), "model_lstm_pointer_final.msgpack")
    upload_to_kaggle(SAVE_DIR, message="Curriculum Training Finished (All Phases)")
    print("[*] Training finished.")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
