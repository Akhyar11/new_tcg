import os
import sys
import glob
import random
import time
import numpy as np

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from cg.game import battle_start, battle_finish, battle_select
from cg.api import to_dataclass, Observation
import tcg_core.action_mapping as action_mapping
from tcg_core.agents import LSTMAgent
from tcg_core.models.ptr import PokemonAgent as PTRModel

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

def simulate_game(agent, d0, d1, swap_players=False):
    agent.reset()
    if hasattr(agent, 'carry'):
        carry_0 = agent.carry
        carry_1 = agent.carry
    else:
        carry_0 = None
        carry_1 = None

    if swap_players:
        obs_dict, _ = battle_start(d1, d0)
    else:
        obs_dict, _ = battle_start(d0, d1)
        
    obs = to_dataclass(obs_dict, Observation)
    
    step_count = 0
    while obs is not None and obs.current is not None and obs.current.result == -1:
        step_count += 1
        if step_count > 200: break
            
        your_index = obs.current.yourIndex
        if carry_0 is not None:
            if your_index == 0:
                agent.carry = carry_0
            else:
                agent.carry = carry_1
                
        choices = agent.select_action(obs)
        
        if carry_0 is not None:
            if your_index == 0:
                carry_0 = agent.carry
            else:
                carry_1 = agent.carry
        
        try:
            obs_dict = battle_select(choices)
            obs = to_dataclass(obs_dict, Observation)
        except:
            try:
                opt_count = len(obs.select.option) if obs is not None and obs.select and obs.select.option else 0
                min_c = obs.select.minCount if obs is not None and obs.select else 0
                obs = to_dataclass(battle_select(list(range(min(opt_count, min_c)))), Observation)
            except:
                break
                
    result = obs.current.result if obs is not None and obs.current is not None else -1
    battle_finish()
    
    if swap_players:
        if result == 0: return 1
        elif result == 1: return 0
    return result

def main():
    deck_dir = os.path.join(ROOT, "new_deck")
    deck_files = sorted(glob.glob(os.path.join(deck_dir, "*.csv")))
    all_decks = []
    for f in deck_files:
        d = load_deck(f)
        if d: all_decks.append((os.path.basename(f).replace('.csv', ''), d))
        
    print(f"Loaded {len(all_decks)} decks.")
    
    checkpoint_path = os.path.join(ROOT, "checkpoints", "model_lstm_pointer_final.msgpack")
    agent = LSTMAgent("PTR_Agent", PTRModel, action_mapping, checkpoint_path=checkpoint_path)
    print("PTR Model Loaded. Starting Tournament...")
    
    start_time = time.time()
    
    # STAGE 1: Qualifiers (10 matches per deck)
    print("\n--- STAGE 1: QUALIFIERS (10 matches each) ---")
    stage1_results = []
    for i, (name, deck) in enumerate(all_decks):
        wins = 0
        for match_idx in range(10):
            opp = random.choice(all_decks)[1]
            swap_players = (match_idx % 2 == 1)
            if simulate_game(agent, deck, opp, swap_players) == 0: wins += 1
        stage1_results.append((wins, name, deck))
        if (i+1) % 20 == 0: print(f"  Processed {i+1}/{len(all_decks)} decks...")
        
    stage1_results.sort(key=lambda x: -x[0])
    top_32 = stage1_results[:32]
    print(f"Top 32 decks selected! Cutoff wins: {top_32[-1][0]}/10")
    
    # STAGE 2: Quarterfinals (20 matches each)
    print("\n--- STAGE 2: QUARTERFINALS (20 matches each) ---")
    stage2_results = []
    for i, (prev_wins, name, deck) in enumerate(top_32):
        wins = 0
        for match_idx in range(20):
            opp = random.choice(top_32)[2]
            swap_players = (match_idx % 2 == 1)
            if simulate_game(agent, deck, opp, swap_players) == 0: wins += 1
        stage2_results.append((wins, name, deck))
        
    stage2_results.sort(key=lambda x: -x[0])
    top_8 = stage2_results[:8]
    print(f"Top 8 decks selected! Cutoff wins: {top_8[-1][0]}/20")
    
    # STAGE 3: Round Robin Finals (3 matches vs each finalist)
    print("\n--- STAGE 3: ROUND ROBIN FINALS (3 matches vs all finalists) ---")
    final_scores = {name: 0 for _, name, _ in top_8}
    
    for i in range(len(top_8)):
        for j in range(len(top_8)):
            if i == j: continue
            name1, deck1 = top_8[i][1], top_8[i][2]
            name2, deck2 = top_8[j][1], top_8[j][2]
            for _ in range(3): # 3 games per pair
                if simulate_game(agent, deck1, deck2) == 0:
                    final_scores[name1] += 1
                    
    print("\n==================================================")
    print("🏆 GRAND CHAMPION FINAL RESULTS 🏆")
    print("==================================================")
    
    ranked_finals = sorted(final_scores.items(), key=lambda x: -x[1])
    for rank, (name, score) in enumerate(ranked_finals):
        print(f"Rank {rank+1}: {name} (Wins: {score}/42)")
        
    print(f"\nTournament completed in {time.time() - start_time:.1f} seconds.")
    
    best_deck_name = ranked_finals[0][0]
    print(f"THE BEST DECK TO SUBMIT IS: >> {best_deck_name} <<")
    
    with open(os.path.join(ROOT, "best_deck.txt"), "w") as f:
        f.write(best_deck_name)

if __name__ == "__main__":
    main()
