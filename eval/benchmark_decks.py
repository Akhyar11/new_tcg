import os
import sys
import glob
import random
import time
import numpy as np
import dataclasses
from multiprocessing import Pool

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from flax import serialization

from cg.game import battle_start, battle_finish, battle_select
from cg.api import to_dataclass, Observation, OptionType
from tcg_core.feature_extractor import extract_features
from tcg_core.action_mapping import get_action_index_for_option, create_action_mask
from tcg_core.models.ff import PokemonAgent

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
    max_c = obs.select.maxCount
    
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
            if remaining.sum() <= 0:
                break
            p = remaining / remaining.sum()
            idx = int(np.random.choice(len(p), p=p))
            if idx == 160:
                has_end_option = any(get_action_index_for_option(opt, i) == 160 for i, opt in enumerate(mock_select["options"]))
                if has_end_option:
                    sampled_indices.append(idx)
                    remaining[idx] = 0.0
                elif len(sampled_indices) >= min_c:
                    break
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
            if cpp_idx not in choices:
                choices.append(cpp_idx)
            if len(choices) >= min_c:
                break

    return choices

def simulate_game(model_apply, params, d0, d1):
    obs_dict, _ = battle_start(d0, d1)
    obs = to_dataclass(obs_dict, Observation)
    
    step_count = 0
    while obs.current is not None and obs.current.result == -1:
        step_count += 1
        if step_count > 200:
            break
            
        choices = ai_select(model_apply, params, obs)
        
        try:
            obs_dict = battle_select(choices)
            obs = to_dataclass(obs_dict, Observation)
        except:
            try:
                opt_count = len(obs.select.option) if obs.select and obs.select.option else 0
                min_c = obs.select.minCount if obs.select else 0
                obs_dict = battle_select(list(range(min(opt_count, min_c))))
                obs = to_dataclass(obs_dict, Observation)
            except:
                break
                
    result = obs.current.result if obs.current else -1
    turns = obs.current.turn if obs.current else 0
    
    reason = "unknown"
    if result in [0, 1] and obs.current:
        winner_p = obs.current.players[result]
        loser_p = obs.current.players[1 - result]
        
        if len(winner_p.prize) == 0:
            reason = "prize"
        elif len(loser_p.active) == 0:
            reason = "no_active"
        elif loser_p.deckCount == 0:
            reason = "deck_out"
            
    battle_finish()
    
    return result, turns, step_count, reason

# Global references for multiprocessing
G_MODEL_APPLY = None
G_PARAMS = None
G_ALL_DECKS = []

def init_worker():
    global G_MODEL_APPLY, G_PARAMS, G_ALL_DECKS
    import jax
    import jax.numpy as jnp
    from tcg_core.models.ff import PokemonAgent
    from flax import serialization
    
    deck_dir = os.path.join(ROOT, "new_deck")
    deck_files = sorted(glob.glob(os.path.join(deck_dir, "*.csv")))
    G_ALL_DECKS = [load_deck(f) for f in deck_files]
    
    model = PokemonAgent(num_actions=250)
    rng = jax.random.PRNGKey(42)
    _, init_rng = jax.random.split(rng)
    dummy_seq = jnp.zeros((1, 173, 31))
    dummy_glob = jnp.zeros((1, 266))
    
    params = model.init(init_rng, dummy_seq, dummy_glob)
    model_final_path = os.path.join(ROOT, "checkpoints", "model_final.msgpack")
    
    with open(model_final_path, 'rb') as f:
        params = serialization.from_bytes(params, f.read())
        
    G_MODEL_APPLY = jax.jit(model.apply)
    # Warmup
    _ = G_MODEL_APPLY(params, dummy_seq, dummy_glob)
    G_PARAMS = params

def evaluate_deck(task):
    deck_idx, deck_name, matches = task
    d0 = G_ALL_DECKS[deck_idx]
    if not d0: return deck_name, 0, 0, 0, [], {}
    
    wins = 0
    losses = 0
    draws = 0
    turn_counts = []
    
    win_reasons = {"prize": 0, "no_active": 0, "deck_out": 0, "unknown": 0}
    
    for _ in range(matches):
        d1 = random.choice(G_ALL_DECKS)
        if not d1: continue
        
        res, turns, steps, reason = simulate_game(G_MODEL_APPLY, G_PARAMS, d0, d1)
        if res == 0:
            wins += 1
            turn_counts.append(turns)
            if reason in win_reasons:
                win_reasons[reason] += 1
        elif res == 1:
            losses += 1
        else:
            draws += 1
            
    return deck_name, wins, losses, draws, turn_counts, win_reasons

def main():
    deck_dir = os.path.join(ROOT, "new_deck")
    deck_files = sorted(glob.glob(os.path.join(deck_dir, "*.csv")))
    
    MATCHES_PER_DECK = 100
    print(f"Benchmarking {len(deck_files)} decks... Each will play {MATCHES_PER_DECK} matches as Player 0.")
    print("Using multiprocessing to speed up evaluation...")
    
    tasks = []
    for i, f in enumerate(deck_files):
        name = os.path.basename(f).replace('.csv', '')
        tasks.append((i, name, MATCHES_PER_DECK))
        
    start_time = time.time()
    
    # Run in parallel
    stats = {}
    with Pool(processes=8, initializer=init_worker) as pool:
        results = pool.map(evaluate_deck, tasks)
        
        for name, wins, losses, draws, turn_counts, win_reasons in results:
            wr = (wins / MATCHES_PER_DECK) * 100
            avg_turns = np.mean(turn_counts) if turn_counts else 0
            std_turns = np.std(turn_counts) if turn_counts else 0
            stats[name] = {
                "win_rate": wr,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "avg_turns": avg_turns,
                "std_turns": std_turns,
                "win_reasons": win_reasons
            }
            print(f"[{name}] WR: {wr:5.1f}% | W/L/D: {wins}/{losses}/{draws} | Avg Turns: {avg_turns:.1f} (±{std_turns:.1f})")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f} seconds.")
    
    print("\n--- FINAL BENCHMARK RANKING (100 MATCHES/DECK) ---")
    # Sort by Win Rate (desc), then by Lowest Std Dev Turns (Consistency), then by Lowest Avg Turns (Speed)
    ranked = sorted(stats.items(), key=lambda x: (-x[1]["win_rate"], x[1]["std_turns"], x[1]["avg_turns"]))
    
    print(f"{'Deck Name':<35} | {'WR%':<6} | {'W/L/D':<10} | {'Avg Turns (std)':<20} | {'Win Reasons (Prize/NoActv/DeckOut/Unk)'}")
    print("-" * 115)
    
    output_dir = os.path.join(ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "benchmark_results.txt")
    
    with open(out_file, "w") as f:
        header = f"{'Deck Name':<35} | {'WR%':<6} | {'W/L/D':<10} | {'Avg Turns (std)':<20} | {'Win Reasons (Prize/NoActv/DeckOut/Unk)'}"
        f.write(header + "\n")
        f.write("-" * 115 + "\n")
        
        for name, data in ranked:
            reasons = data.get('win_reasons', {})
            reason_str = f"{reasons.get('prize',0)} / {reasons.get('no_active',0)} / {reasons.get('deck_out',0)} / {reasons.get('unknown',0)}"
            line = f"{name:<35} | {data['win_rate']:>5.1f}% | {data['wins']}/{data['losses']}/{data['draws']:<6} | {data['avg_turns']:5.1f} (±{data['std_turns']:4.1f}) | {reason_str}"
            print(line)
            f.write(line + "\n")
            
    print(f"\nBenchmark results saved to {out_file}")

if __name__ == "__main__":
    main()
