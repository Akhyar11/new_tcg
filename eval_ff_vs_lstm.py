#!/usr/bin/env python3
"""
Evaluation Script: FF (Feed-Forward) vs LSTM
P0 = LSTM (model_lstm_final.msgpack)
P1 = FF (model_final.msgpack)
"""
import os
import sys
import glob
import random
import time
import csv
import numpy as np

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)

import jax
import jax.numpy as jnp
from flax import serialization

from cg.game import battle_start, battle_finish, battle_select
from cg.api import to_dataclass, Observation, OptionType, LogType
from agent_rl.feature_extractor import extract_features
from agent_rl.action_mapping import get_action_index_for_option, create_action_mask

# Import dua arsitektur berbeda
from agent_rl.model import PokemonAgent as FFModel
from agent_rl_lstm.model import PokemonAgent as LSTMModel

def load_deck(filepath):
    deck = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line.isdigit():
                deck.append(int(line))
    if len(deck) != 60: return None
    return deck

def load_card_db():
    cards = {}
    csv_path = os.path.join(ROOT, "agent_rl_lstm", "EN_Card_Data.csv")
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cards[int(row['Card ID'])] = row
    return cards

def softmax(x):
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / (exp_x.sum() + 1e-10)

def ai_select_lstm(model_apply, params, carry, obs):
    if not obs.select or not obs.select.option:
        return [], carry, {}

    your_index = obs.current.yourIndex
    features = extract_features(obs.current, obs.select, your_index)
    seq_input = np.expand_dims(features["seq_input"], axis=0)
    glob_input = np.expand_dims(features["glob_input"], axis=0)

    # Call model with correct parameter ordering: (params, seq_input, glob_input, carry)
    # Output is (logits, value, new_carry)
    logits_raw, value, new_carry = model_apply(params, seq_input, glob_input, carry)
    logits_np = np.array(logits_raw[0])
    value_np = float(np.array(value).flatten()[0])

    options = obs.select.option
    min_c = obs.select.minCount
    max_c = obs.select.maxCount
    
    import dataclasses
    mock_options = []
    for o in options:
        d = dataclasses.asdict(o)
        d["type"] = OptionType(o.type).name
        mock_options.append(d)
    mock_select = {"options": mock_options}

    mask_array = create_action_mask(mock_select, min_c, max_c)
    masked = logits_np - 1e9 * (1.0 - mask_array)
    probs = softmax(masked)

    sampled_indices = []
    if probs.sum() > 0:
        remaining = probs.copy()
        for _ in range(max_c):
            if remaining.sum() <= 0: break
            p = remaining / remaining.sum()
            idx = int(np.random.choice(len(p), p=p))
            if idx == 160:
                has_end_option = any(get_action_index_for_option(opt, i) == 160 for i, opt in enumerate(mock_select["options"]))
                if has_end_option:
                    sampled_indices.append(idx)
                    remaining[idx] = 0.0
                elif len(sampled_indices) >= min_c: break
                else:
                    remaining[idx] = 0.0
                    continue
            else:
                sampled_indices.append(idx)
                remaining[idx] = 0.0
    else:
        sampled_indices = [160]

    choices = []
    for jax_idx in sampled_indices:
        for cpp_idx, opt in enumerate(mock_select["options"]):
            mapped_idx = get_action_index_for_option(opt, cpp_idx)
            if mapped_idx == jax_idx and cpp_idx not in choices:
                choices.append(cpp_idx)
                break

    if len(choices) < min_c:
        for cpp_idx in range(len(options)):
            if cpp_idx not in choices: choices.append(cpp_idx)
            if len(choices) >= min_c: break

    return choices, new_carry, {"value": value_np, "entropy": float(-np.sum(probs * np.log(probs + 1e-10) * mask_array))}

def ai_select_ff(model_apply, params, obs):
    if not obs.select or not obs.select.option:
        return [], {}

    your_index = obs.current.yourIndex
    features = extract_features(obs.current, obs.select, your_index)
    seq_input = np.expand_dims(features["seq_input"], axis=0)
    glob_input = np.expand_dims(features["glob_input"], axis=0)

    logits_raw, value = model_apply(params, seq_input, glob_input)
    logits_np = np.array(logits_raw[0])
    value_np = float(np.array(value).flatten()[0])

    options = obs.select.option
    min_c = obs.select.minCount
    max_c = obs.select.maxCount
    
    import dataclasses
    mock_options = []
    for o in options:
        d = dataclasses.asdict(o)
        d["type"] = OptionType(o.type).name
        mock_options.append(d)
    mock_select = {"options": mock_options}

    mask_array = create_action_mask(mock_select, min_c, max_c)
    masked = logits_np - 1e9 * (1.0 - mask_array)
    probs = softmax(masked)

    sampled_indices = []
    if probs.sum() > 0:
        remaining = probs.copy()
        for _ in range(max_c):
            if remaining.sum() <= 0: break
            p = remaining / remaining.sum()
            idx = int(np.random.choice(len(p), p=p))
            if idx == 160:
                has_end_option = any(get_action_index_for_option(opt, i) == 160 for i, opt in enumerate(mock_select["options"]))
                if has_end_option:
                    sampled_indices.append(idx)
                    remaining[idx] = 0.0
                elif len(sampled_indices) >= min_c: break
                else:
                    remaining[idx] = 0.0
                    continue
            else:
                sampled_indices.append(idx)
                remaining[idx] = 0.0
    else:
        sampled_indices = [160]

    choices = []
    for jax_idx in sampled_indices:
        for cpp_idx, opt in enumerate(mock_select["options"]):
            mapped_idx = get_action_index_for_option(opt, cpp_idx)
            if mapped_idx == jax_idx and cpp_idx not in choices:
                choices.append(cpp_idx)
                break

    if len(choices) < min_c:
        for cpp_idx in range(len(options)):
            if cpp_idx not in choices: choices.append(cpp_idx)
            if len(choices) >= min_c: break

    return choices, {"value": value_np, "entropy": float(-np.sum(probs * np.log(probs + 1e-10) * mask_array))}

def format_card(c, db):
    if not c: return "Empty"
    name = db.get(c.id, {}).get("Card Name", f"ID:{c.id}")
    hp_str = f" HP:{c.hp}/{c.maxHp}" if hasattr(c, 'hp') else ""
    e_str = f" E:{len(c.energies)}" if hasattr(c, 'energies') and len(c.energies) > 0 else ""
    return f"[{name}{hp_str}{e_str}]"

def main():
    cards_db = load_card_db()
    deck_dir = os.path.join(ROOT, "new_deck")
    
    seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    print(f"Using random seed: {seed}")
    
    deck_name = "Mega Lucario Aura Strike.csv"
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if os.path.exists(arg):
            deck_path = arg
        elif os.path.exists(os.path.join(deck_dir, arg)):
            deck_path = os.path.join(deck_dir, arg)
        elif os.path.exists(os.path.join(deck_dir, arg + ".csv")):
            deck_path = os.path.join(deck_dir, arg + ".csv")
        else:
            print(f"[!] Deck '{arg}' tidak ditemukan. Memakai default.")
            deck_path = os.path.join(deck_dir, deck_name)
    else:
        deck_path = os.path.join(deck_dir, deck_name)
        
    d0 = load_deck(deck_path)
    d1 = load_deck(deck_path)
    if d0 is None or d1 is None:
        print(f"[!] Gagal me-load deck {deck_path}. Pastikan file valid dengan 60 kartu.")
        return
    
    print(f"Deck P0 (LSTM) & P1 (FF): {os.path.basename(deck_path)}")

    # Init FF Model (P1)
    ff_model = FFModel(num_actions=250)
    rng = jax.random.PRNGKey(42)
    rng, init_rng_ff = jax.random.split(rng)
    dummy_seq_ff = jnp.zeros((1, 173, 31))
    dummy_glob_ff = jnp.zeros((1, 266))
    params_ff = ff_model.init(init_rng_ff, dummy_seq_ff, dummy_glob_ff)
    
    # Init LSTM Model (P0)
    lstm_model = LSTMModel(num_actions=250)
    rng, init_rng_lstm = jax.random.split(rng)
    dummy_seq_lstm = jnp.zeros((1, 173, 31))
    dummy_glob_lstm = jnp.zeros((1, 266))
    dummy_carry = (jnp.zeros((1, 256)), jnp.zeros((1, 256)))
    params_lstm = lstm_model.init(init_rng_lstm, dummy_seq_lstm, dummy_glob_lstm, dummy_carry)
    
    # Load Weights
    save_dir = os.path.join(ROOT, "tcg_models")
    lstm_path = os.path.join(save_dir, "model_lstm_final.msgpack")
    ff_path = os.path.join(save_dir, "model_final.msgpack") # Or model_base.msgpack
    
    print(f"\n[LOAD] LSTM P0 dari {lstm_path}")
    if os.path.exists(lstm_path):
        with open(lstm_path, 'rb') as f:
            params_lstm = serialization.from_bytes(params_lstm, f.read())
    else:
        print("[WARNING] model_lstm_final.msgpack tidak ditemukan, memakai inisialisasi random untuk LSTM!")
        
    print(f"[LOAD] FF P1 dari {ff_path}")
    if os.path.exists(ff_path):
        with open(ff_path, 'rb') as f:
            params_ff = serialization.from_bytes(params_ff, f.read())
    else:
        print("[WARNING] model_final.msgpack (FF) tidak ditemukan, memakai inisialisasi random untuk FF!")
        
    ff_apply = jax.jit(ff_model.apply)
    lstm_apply = jax.jit(lstm_model.apply)
    
    # Warmup
    _ = ff_apply(params_ff, dummy_seq_ff, dummy_glob_ff)
    _ = lstm_apply(params_lstm, dummy_seq_lstm, dummy_glob_lstm, dummy_carry)
    
    obs_dict, _ = battle_start(d0, d1)
    obs = to_dataclass(obs_dict, Observation)
    
    # P0 Carry state
    p0_carry = (jnp.zeros((1, 256)), jnp.zeros((1, 256)))
    
    step_count = 0
    while obs.current is not None and obs.current.result == -1:
        step_count += 1
        if step_count > 300:
            print("--- MAX STEPS REACHED ---")
            break
            
        your_idx = obs.current.yourIndex
        actor_type = "LSTM" if your_idx == 0 else "FF"
        
        print(f"\n{'='*80}")
        print(f"STEP {step_count} | Turn {obs.current.turn} | Player to act: P{your_idx} ({actor_type})")
        
        # Print Logs
        if obs.logs:
            print(f"LOGS:")
            for log in obs.logs:
                ltype = LogType(log.type).name
                pid = log.playerIndex
                if log.type == LogType.PLAY:
                    cname = cards_db.get(log.cardId, {}).get("Card Name", log.cardId)
                    print(f"  -> P{pid} PLAYED: {cname}")
                elif log.type == LogType.ATTACK:
                    print(f"  -> P{pid} ATTACKED!")
        
        # Select choices
        if your_idx == 0:
            choices, p0_carry, info = ai_select_lstm(lstm_apply, params_lstm, p0_carry, obs)
        else:
            choices, info = ai_select_ff(ff_apply, params_ff, obs)
        
        if obs.select and obs.select.option:
            opt_names = []
            for c in choices:
                o = obs.select.option[c]
                t = OptionType(o.type).name
                detail = f"idx:{o.index}"
                if t in ["CARD", "PLAY"] and obs.current.players[your_idx].hand and o.index is not None and o.index < len(obs.current.players[your_idx].hand):
                    hc = obs.current.players[your_idx].hand[o.index]
                    detail = cards_db.get(hc.id, {}).get("Card Name", "?")
                elif t == "ATTACK":
                    detail = f"atkID:{o.attackId}"
                opt_names.append(f"{t}({detail})")
                
            print(f"AI DECISION: {opt_names} | Critic Value: {info.get('value'):+.2f}")
        
        try:
            obs_dict = battle_select(choices)
            obs = to_dataclass(obs_dict, Observation)
        except Exception as e:
            print(f"! ERROR SELECTING: {e}")
            break
                
    if obs.current:
        print(f"\nGAME OVER! Result: {obs.current.result}")
    battle_finish()

if __name__ == "__main__":
    main()
