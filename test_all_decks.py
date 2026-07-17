import os
import sys
import glob
import random
import time
import numpy as np

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
from flax import serialization

from cg.game import battle_start, battle_finish, battle_select
from cg.api import to_dataclass, Observation, OptionType, LogType
from agent_rl.feature_extractor import extract_features
from agent_rl.action_mapping import get_action_index_for_option, create_action_mask
from agent_rl.model import PokemonAgent

ROOT = os.path.dirname(os.path.abspath(__file__))

def load_deck(filepath):
    deck = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line.isdigit():
                deck.append(int(line))
    if len(deck) != 60: return None
    return deck

def softmax(x):
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / (exp_x.sum() + 1e-10)

def ai_select(model_apply, params, obs):
    if not obs.select or not obs.select.option:
        return []

    your_index = obs.current.yourIndex
    features = extract_features(obs.current, obs.select, your_index)
    seq_input = np.expand_dims(features["seq_input"], axis=0)
    glob_input = np.expand_dims(features["glob_input"], axis=0)

    logits_raw, _ = model_apply(params, seq_input, glob_input)
    logits_np = np.array(logits_raw[0])

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

    return choices

def simulate_game(model_apply, params, d0, d1, p0_name, p1_name):
    obs_dict, _ = battle_start(d0, d1)
    obs = to_dataclass(obs_dict, Observation)
    
    step_count = 0
    p0_prizes_taken = 0
    p1_prizes_taken = 0
    
    while obs.current is not None and obs.current.result == -1:
        step_count += 1
        if step_count > 200:
            break
            
        your_idx = obs.current.yourIndex
        choices = ai_select(model_apply, params, obs)
        
        try:
            obs_dict = battle_select(choices)
            obs = to_dataclass(obs_dict, Observation)
        except Exception as e:
            try:
                opt_count = len(obs.select.option) if obs.select and obs.select.option else 0
                min_c = obs.select.minCount if obs.select else 0
                obs_dict = battle_select(list(range(min(opt_count, min_c))))
                obs = to_dataclass(obs_dict, Observation)
            except:
                break
                
    result = obs.current.result if obs.current else -1
    turns = obs.current.turn if obs.current else 0
    p0_prizes = 6 - len(obs.current.players[0].prize) if obs.current else 0
    p1_prizes = 6 - len(obs.current.players[1].prize) if obs.current else 0
    battle_finish()
    
    return {
        "result": result,
        "turns": turns,
        "p0_prizes": p0_prizes,
        "p1_prizes": p1_prizes,
        "steps": step_count
    }

def main():
    deck_dir = os.path.join(ROOT, "new_deck")
    deck_files = sorted(glob.glob(os.path.join(deck_dir, "*.csv")))
    
    model = PokemonAgent(num_actions=250)
    rng = jax.random.PRNGKey(42)
    _, init_rng = jax.random.split(rng)
    dummy_seq = jnp.zeros((1, 173, 31))
    dummy_glob = jnp.zeros((1, 266))
    
    params = model.init(init_rng, dummy_seq, dummy_glob)
    
    model_final_path = os.path.join(ROOT, "tcg_models", "model_final.msgpack")
    
    with open(model_final_path, 'rb') as f:
        params = serialization.from_bytes(params, f.read())
        
    model_apply = jax.jit(model.apply)
    _ = model_apply(params, dummy_seq, dummy_glob)
    
    print("Testing 10 random matchups...")
    random.seed(42)
    np.random.seed(42)
    
    for i in range(10):
        while True:
            f0 = random.choice(deck_files)
            f1 = random.choice(deck_files)
            d0 = load_deck(f0)
            d1 = load_deck(f1)
            if d0 is not None and d1 is not None:
                break
        
        name0 = os.path.basename(f0).replace('.csv', '')
        name1 = os.path.basename(f1).replace('.csv', '')
        
        print(f"Match {i+1}: {name0} vs {name1}")
        res = simulate_game(model_apply, params, d0, d1, name0, name1)
        print(f"  Result: {res['result']}, Turns: {res['turns']}, Steps: {res['steps']}")
        print(f"  Prizes Taken - P0: {res['p0_prizes']}, P1: {res['p1_prizes']}")
        if res['p0_prizes'] > 0 or res['p1_prizes'] > 0:
            print(f"  >>> ACTION HAPPENED! Someone took a prize!")

if __name__ == "__main__":
    main()
