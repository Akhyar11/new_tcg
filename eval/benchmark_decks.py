import os
import sys
import glob
import random
import time
import numpy as np
import dataclasses
from multiprocessing import Pool

# Force CPU (Commented out to use GPU)
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from flax import serialization

from cg.game import battle_start, battle_finish, battle_select
from cg.api import to_dataclass, Observation, OptionType

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

def simulate_game(agent, d0, d1):
    agent.reset()
    obs_dict, _ = battle_start(d0, d1)
    obs = to_dataclass(obs_dict, Observation)
    
    step_count = 0
    while obs is not None and obs.current is not None and obs.current.result == -1:
        step_count += 1
        if step_count > 200:
            break
            
        choices = agent.select_action(obs)
        
        try:
            obs_dict = battle_select(choices)
            obs = to_dataclass(obs_dict, Observation)
        except:
            try:
                opt_count = len(obs.select.option) if obs is not None and obs.select and obs.select.option else 0
                min_c = obs.select.minCount if obs is not None and obs.select else 0
                obs_dict = battle_select(list(range(min(opt_count, min_c))))
                obs = to_dataclass(obs_dict, Observation)
            except:
                break
                
    result = obs.current.result if obs is not None and obs.current is not None else -1
    turns = obs.current.turn if obs is not None and obs.current is not None else 0
    
    reason = "unknown"
    if result in [0, 1] and obs is not None and obs.current is not None:
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
G_AGENT = None
G_ALL_DECKS = []

def init_worker():
    global G_AGENT, G_ALL_DECKS
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    
    import tcg_core.action_mapping as action_mapping
    from tcg_core.agents import LSTMAgent
    from tcg_core.models.ptr import PokemonAgent as PTRModel
    
    deck_dir = os.path.join(ROOT, "new_deck")
    deck_files = sorted(glob.glob(os.path.join(deck_dir, "*.csv")))
    G_ALL_DECKS = [load_deck(f) for f in deck_files]
    
    checkpoint_path = os.path.join(ROOT, "checkpoints", "model_lstm_pointer_final.msgpack")
    
    G_AGENT = LSTMAgent("PTR_Agent", PTRModel, action_mapping, checkpoint_path=checkpoint_path)

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
        
        res, turns, steps, reason = simulate_game(G_AGENT, d0, d1)
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
    
    MATCHES_PER_DECK = 10  # Diubah menjadi 10 agar lebih cepat saat pengetesan
    print(f"Benchmarking {len(deck_files)} decks with PTR Model... Each will play {MATCHES_PER_DECK} matches as Player 0.")
    print("Running sequentially on GPU to avoid JAX memory locks...")
    
    tasks = []
    for i, f in enumerate(deck_files):
        name = os.path.basename(f).replace('.csv', '')
        tasks.append((i, name, MATCHES_PER_DECK))
        
    start_time = time.time()
    
    # Run sequentially (Multiprocessing with JAX on single GPU causes locks/OOM)
    stats = {}
    init_worker() # Initialize agent on main process
    
    for i, task in enumerate(tasks):
        name, wins, losses, draws, turn_counts, win_reasons = evaluate_deck(task)
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
        print(f"[{i+1}/{len(tasks)}] [{name}] WR: {wr:5.1f}% | W/L/D: {wins}/{losses}/{draws} | Avg Turns: {avg_turns:.1f} (±{std_turns:.1f})")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f} seconds.")
    
    print("\n--- FINAL BENCHMARK RANKING (10 MATCHES/DECK) ---")
    # Sort by Win Rate (desc), then by Lowest Std Dev Turns (Consistency), then by Lowest Avg Turns (Speed)
    ranked = sorted(stats.items(), key=lambda x: (-x[1]["win_rate"], x[1]["std_turns"], x[1]["avg_turns"]))
    
    print(f"{'Deck Name':<35} | {'WR%':<6} | {'W/L/D':<10} | {'Avg Turns (std)':<20} | {'Win Reasons (Prize/NoActv/DeckOut/Unk)'}")
    print("-" * 115)
    
    output_dir = os.path.join(ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "benchmark_results_ptr.txt")
    
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
