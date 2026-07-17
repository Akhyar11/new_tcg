#!/usr/bin/env python3
"""
Single Game Evaluation Script for Detailed AI Analysis
Prints complete board state, hand, detailed actions, and event logs per step.
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from flax import serialization

from cg.game import battle_start, battle_finish, battle_select
from cg.api import to_dataclass, Observation, OptionType, LogType
from agent_rl.feature_extractor import extract_features
from agent_rl.action_mapping import get_action_index_for_option, create_action_mask
from agent_rl.model import PokemonAgent

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
    csv_path = os.path.join(ROOT, "agent_rl", "EN_Card_Data.csv")
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

def ai_select(model_apply, params, obs):
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
    
    import dataclasses
    mock_options = []
    for o in options:
        d = dataclasses.asdict(o)
        d["type"] = OptionType(o.type).name
        mock_options.append(d)
    mock_select = {"options": mock_options}

    mask_array = create_action_mask(mock_select)

    masked = logits_np - 1e9 * (1.0 - mask_array)
    probs = softmax(masked)

    sampled_indices = []
    if probs.sum() > 0:
        remaining = probs.copy()
        for _ in range(min_c):
            if remaining.sum() <= 0:
                break
            p = remaining / remaining.sum()
            idx = int(np.random.choice(len(p), p=p))
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
            if cpp_idx not in choices:
                choices.append(cpp_idx)
            if len(choices) >= min_c:
                break

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
    deck_files = sorted(glob.glob(os.path.join(deck_dir, "*.csv")))
    
    # Use random probabilitas same as training (not deterministic)
    seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    print(f"Using random seed: {seed}")
    
    d0_path = os.path.join(deck_dir, "Mega Gardevoir's Symphonia.csv")
    d1_path = os.path.join(deck_dir, "Miraidon Future Speed.csv")
    d0 = load_deck(d0_path)
    d1 = load_deck(d1_path)
    
    print(f"Deck 0: {os.path.basename(d0_path)}")
    print(f"Deck 1: {os.path.basename(d1_path)}")

    model = PokemonAgent(num_actions=250)
    rng = jax.random.PRNGKey(42)
    _, init_rng = jax.random.split(rng)
    dummy_seq = jnp.zeros((1, 113, 31))
    dummy_glob = jnp.zeros((1, 266))
    
    params_p0 = model.init(init_rng, dummy_seq, dummy_glob)
    params_p1 = model.init(init_rng, dummy_seq, dummy_glob)
    
    # Cek & download model Kaggle
    save_dir = os.path.join(ROOT, "tcg_models")
    model_final_path = os.path.join(save_dir, "model_final.msgpack")
    model_base_path = os.path.join(save_dir, "model_base.msgpack")
    
    # Fallback ke folder Unduhan jika ada
    alt_final = os.path.expanduser("~/Unduhan/model_final.msgpack")
    alt_base = os.path.expanduser("~/Unduhan/model_base.msgpack")
    
    if not os.path.exists(model_final_path) and os.path.exists(alt_final):
        model_final_path = alt_final
    if not os.path.exists(model_base_path) and os.path.exists(alt_base):
        model_base_path = alt_base
        
    if not os.path.exists(model_final_path) or not os.path.exists(model_base_path):
        print("[*] Model Kaggle belum lengkap. Mendownload...")
        try:
            os.environ["KAGGLE_USERNAME"] = "akhyarsafrudin"
            os.environ["KAGGLE_KEY"] = "03c3e536ffedc7d6153c1b3b8515242b"
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files("akhyarsafrudin/tcg-models", path=save_dir, unzip=True)
            model_final_path = os.path.join(save_dir, "model_final.msgpack")
            model_base_path = os.path.join(save_dir, "model_base.msgpack")
        except Exception as e:
            print(f"[!] Gagal download dari Kaggle: {e}")
    
    print(f"Loading P0 from {model_final_path}")
    with open(model_final_path, 'rb') as f:
        params_p0 = serialization.from_bytes(params_p0, f.read())
        
    print(f"Loading P1 from {model_final_path}")
    with open(model_final_path, 'rb') as f:
        params_p1 = serialization.from_bytes(params_p1, f.read())
        
    model_apply = jax.jit(model.apply)
    _ = model_apply(params_p0, dummy_seq, dummy_glob)
    
    obs_dict, _ = battle_start(d0, d1)
    obs = to_dataclass(obs_dict, Observation)
    
    step_count = 0
    while obs.current is not None and obs.current.result == -1:
        step_count += 1
        if step_count > 200:
            print("--- MAX STEPS REACHED ---")
            break
            
        your_idx = obs.current.yourIndex
        print(f"\n{'='*80}")
        print(f"STEP {step_count} | Turn {obs.current.turn} | Player to act: P{your_idx}")
        
        # Print Logs
        if obs.logs:
            print(f"LOGS:")
            for log in obs.logs:
                ltype = LogType(log.type).name
                pid = log.playerIndex
                if log.type == LogType.ATTACK:
                    print(f"  -> P{pid} ATTACKED!")
                elif log.type == LogType.HP_CHANGE:
                    print(f"  -> P{pid} HP CHANGE: {log.value} (from damage: {log.putDamageCounter})")
                elif log.type == LogType.PLAY:
                    cname = cards_db.get(log.cardId, {}).get("Card Name", log.cardId)
                    print(f"  -> P{pid} PLAYED: {cname}")
                elif log.type == LogType.EVOLVE:
                    print(f"  -> P{pid} EVOLVED Pokemon!")
                elif log.type == LogType.ATTACH:
                    cname = cards_db.get(log.cardId, {}).get("Card Name", log.cardId)
                    print(f"  -> P{pid} ATTACHED: {cname}")
                elif log.type == LogType.DRAW:
                    print(f"  -> P{pid} DREW A CARD")
        
        # Board State
        state = obs.current
        print("-" * 40)
        for pi in range(2):
            p = state.players[pi]
            active = format_card(p.active[0] if p.active else None, cards_db)
            benches = ", ".join([format_card(b, cards_db) for b in p.bench])
            hand_size = p.handCount
            hand = ", ".join([format_card(h, cards_db) for h in p.hand]) if p.hand else f"Hidden ({hand_size})"
            print(f"P{pi} BOARD:")
            print(f"  Active: {active}")
            print(f"  Bench : {benches if benches else 'None'}")
            print(f"  Hand  : {hand}")
            print(f"  Deck:{p.deckCount} | Prize:{len(p.prize)}")
            
        # Select choices
        choices, info = ai_select(model_apply, params_p0 if your_idx == 0 else params_p1, obs)
        
        if obs.select and obs.select.option:
            opt_names = []
            for c in choices:
                o = obs.select.option[c]
                t = OptionType(o.type).name
                detail = f"idx:{o.index}"
                if t == "CARD" and state.players[your_idx].hand and o.index is not None and o.index < len(state.players[your_idx].hand):
                    hc = state.players[your_idx].hand[o.index]
                    detail = cards_db.get(hc.id, {}).get("Card Name", "?")
                elif t == "PLAY" and state.players[your_idx].hand and o.index is not None and o.index < len(state.players[your_idx].hand):
                    hc = state.players[your_idx].hand[o.index]
                    detail = cards_db.get(hc.id, {}).get("Card Name", "?")
                elif t == "ATTACK":
                    detail = f"atkID:{o.attackId}"
                opt_names.append(f"{t}({detail})")
                
            print("-" * 40)
            print(f"AI DECISION: {opt_names} | Critic Value: {info.get('value'):+.2f}")
        
        try:
            obs_dict = battle_select(choices)
            obs = to_dataclass(obs_dict, Observation)
        except Exception as e:
            try:
                print(f"! ERROR SELECTING, using fallback. {e}")
                opt_count = len(obs.select.option) if obs.select and obs.select.option else 0
                min_c = obs.select.minCount if obs.select else 0
                obs_dict = battle_select(list(range(min(opt_count, min_c))))
                obs = to_dataclass(obs_dict, Observation)
            except:
                print(f"FATAL ERROR!")
                break
                
    if obs.current:
        print(f"\nGAME OVER! Result: {obs.current.result}")
    battle_finish()

if __name__ == "__main__":
    main()
