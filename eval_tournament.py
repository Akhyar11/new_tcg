#!/usr/bin/env python3
"""
Tournament Script: Evaluates 3 models against each other.
- FF (model_final.msgpack)
- LSTM (model_lstm_final.msgpack)
- LSTM Pointer (model_lstm_pointer_final.msgpack)
"""
import os
import sys
import glob
import os
import sys
import random
import time
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)

from tcg_core.environment import TCGEnvironment
from tcg_core.agents import FFAgent, LSTMAgent

# Import models
from tcg_core.models.ff import PokemonAgent as FFModel
from tcg_core.models.lstm import PokemonAgent as LSTMModel
from tcg_core.models.ptr import PokemonAgent as LSTMPointerModel

# Import unified action mapping
import tcg_core.action_mapping as action_mapping

class PointerAgent(LSTMAgent):
    """Temporary OOP wrapper for the Pointer agent to allow tournament comparison."""
    pass

def load_deck(filepath):
    deck = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line.isdigit():
                deck.append(int(line))
    if len(deck) != 60: return None
    return deck

def simulate_game(agent_p0, agent_p1, deck0, deck1):
    agent_p0.reset()
    agent_p1.reset()
    
    env = TCGEnvironment()
    obs, done = env.reset(deck0, deck1)
    
    step_count = 0
    while not done and step_count <= 250:
        step_count += 1
        your_idx = obs.current.yourIndex
        
        choices = agent_p0.select_action(obs) if your_idx == 0 else agent_p1.select_action(obs)
        obs, _, done, info = env.step(choices)
                
    result = info.get("result", -1) if done else -1
    env.close()
    return result

def main():
    deck_dir = os.path.join(ROOT, "new_deck")
    deck_path = os.path.join(deck_dir, "Phantom Dive Sweep.csv")
    d0 = load_deck(deck_path)
    d1 = load_deck(deck_path)
    
    print("=== TCG AI TOURNAMENT (OOP MODE) ===")
    checkpoints_dir = os.path.join(ROOT, "checkpoints")
    
    agent_ff = FFAgent("FF", FFModel, action_mapping, os.path.join(checkpoints_dir, "model_final.msgpack"))
    agent_lstm = LSTMAgent("LSTM", LSTMModel, action_mapping, os.path.join(checkpoints_dir, "model_lstm_final.msgpack"))
    agent_ptr = PointerAgent("LSTM_PTR", LSTMPointerModel, action_mapping, os.path.join(checkpoints_dir, "model_lstm_pointer_final.msgpack"))
    
    agents = {"FF": agent_ff, "LSTM": agent_lstm, "LSTM_PTR": agent_ptr}
    matchups = [("FF", "LSTM"), ("FF", "LSTM_PTR"), ("LSTM", "LSTM_PTR")]
    scores = {"FF": 0, "LSTM": 0, "LSTM_PTR": 0}
    
    NUM_GAMES = 4 # per configuration (2 as P0, 2 as P1) -> total 4 games per matchup
    
    for m1, m2 in matchups:
        print(f"\n--- MATCHUP: {m1} vs {m2} ---")
        m1_wins, m2_wins = 0, 0
        
        # m1 as P0, m2 as P1
        for i in range(NUM_GAMES // 2):
            seed = int(time.time()) + i
            random.seed(seed); np.random.seed(seed)
            res = simulate_game(agents[m1], agents[m2], d0, d1)
            if res == 0: m1_wins += 1
            elif res == 1: m2_wins += 1
            print(f"Game {i+1} ({m1} as P0): Winner = {'P0 ('+m1+')' if res == 0 else 'P1 ('+m2+')' if res == 1 else 'Tie'}")
            
        # m2 as P0, m1 as P1
        for i in range(NUM_GAMES // 2):
            seed = int(time.time()) + i + 100
            random.seed(seed); np.random.seed(seed)
            res = simulate_game(agents[m2], agents[m1], d0, d1)
            if res == 0: m2_wins += 1
            elif res == 1: m1_wins += 1
            print(f"Game {i+1+NUM_GAMES//2} ({m2} as P0): Winner = {'P0 ('+m2+')' if res == 0 else 'P1 ('+m1+')' if res == 1 else 'Tie'}")
            
        print(f"Matchup Result -> {m1}: {m1_wins} wins | {m2}: {m2_wins} wins")
        scores[m1] += m1_wins
        scores[m2] += m2_wins
        
    print("\n==============================")
    print("TOURNAMENT FINAL SCORES")
    print("==============================")
    for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        print(f"{k}: {v} wins")

if __name__ == "__main__":
    main()

