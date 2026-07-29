#!/usr/bin/env python3
import os
import sys
import glob
import random
import time
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from tcg_core.environment import TCGEnvironment
from tcg_core.agents import FFAgent, LSTMAgent
from tcg_core.models.ff import PokemonAgent as FFModel
from tcg_core.models.lstm import PokemonAgent as LSTMModel
from tcg_core.models.ptr import PokemonAgent as LSTMPointerModel
import tcg_core.action_mapping as action_mapping
from cg.api import LogType, OptionType

class PointerAgent(LSTMAgent):
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

def run_behavior_analysis():
    checkpoints_dir = os.path.join(ROOT, "checkpoints")
    deck_path = os.path.join(ROOT, "new_deck", "Roaring Moon Ancient Depths.csv")
    if not os.path.exists(deck_path):
        deck_path = glob.glob(os.path.join(ROOT, "deck_generated", "*.csv"))[0]
    
    deck = load_deck(deck_path)
    
    agent_ff = FFAgent("FF", FFModel, action_mapping, os.path.join(checkpoints_dir, "model_final.msgpack"))
    agent_lstm = LSTMAgent("LSTM", LSTMModel, action_mapping, os.path.join(checkpoints_dir, "model_lstm_final.msgpack"))
    agent_ptr1 = PointerAgent("LSTM_PTR_V1", LSTMPointerModel, action_mapping, os.path.join(checkpoints_dir, "model_lstm_pointer_final.msgpack"))
    agent_ptr2 = PointerAgent("LSTM_PTR_V2", LSTMPointerModel, action_mapping, os.path.join(checkpoints_dir, "model_lstm_pointer_v2_final.msgpack"))
    
    agents = {
        "FF": agent_ff,
        "LSTM": agent_lstm,
        "LSTM_PTR_V1": agent_ptr1,
        "LSTM_PTR_V2": agent_ptr2
    }
    
    model_stats = {
        name: {
            "total_turns": 0,
            "total_steps": 0,
            "attacks": 0,
            "energy_attach": 0,
            "pokemon_played": 0,
            "evolutions": 0,
            "trainer_played": 0,
            "retreats": 0,
            "pass_end": 0,
            "damage_dealt": 0,
            "wins": 0,
            "games_played": 0
        } for name in agents
    }
    
    matchups = [
        ("FF", "LSTM"),
        ("FF", "LSTM_PTR_V1"),
        ("FF", "LSTM_PTR_V2"),
        ("LSTM", "LSTM_PTR_V1"),
        ("LSTM", "LSTM_PTR_V2"),
        ("LSTM_PTR_V1", "LSTM_PTR_V2")
    ]
    
    print("=== DEEP IN-GAME BEHAVIORAL LOG ANALYSIS ===")
    
    for m1_name, m2_name in matchups:
        for p0_name, p1_name in [(m1_name, m2_name), (m2_name, m1_name)]:
            p0_agent = agents[p0_name]
            p1_agent = agents[p1_name]
            
            for game_idx in range(2): # 2 games per role = 4 per matchup
                p0_agent.reset()
                p1_agent.reset()
                
                env = TCGEnvironment()
                obs, done = env.reset(deck, deck)
                
                model_stats[p0_name]["games_played"] += 1
                model_stats[p1_name]["games_played"] += 1
                
                step = 0
                while not done and step <= 200:
                    step += 1
                    active_p = obs.current.yourIndex if obs.current else 0
                    curr_model_name = p0_name if active_p == 0 else p1_name
                    curr_agent = p0_agent if active_p == 0 else p1_agent
                    
                    stats = model_stats[curr_model_name]
                    stats["total_steps"] += 1
                    
                    choices = curr_agent.select_action(obs, deterministic=True)
                    
                    # Analyze choices before step
                    if obs.select and obs.select.option:
                        for c in choices:
                            if c < len(obs.select.option):
                                opt = obs.select.option[c]
                                opt_type = getattr(opt, 'type', 0)
                                
                                try:
                                    enum_type = OptionType(opt_type).name
                                except Exception:
                                    enum_type = str(opt_type)
                                    
                                if enum_type == "ATTACK":
                                    stats["attacks"] += 1
                                elif enum_type == "ATTACH":
                                    stats["energy_attach"] += 1
                                elif enum_type in ["PLAY", "BENCH"]:
                                    stats["pokemon_played"] += 1
                                elif enum_type == "EVOLVE":
                                    stats["evolutions"] += 1
                                elif enum_type in ["CARD", "CARD_DECK", "SKILL"]:
                                    stats["trainer_played"] += 1
                                elif enum_type == "RETREAT":
                                    stats["retreats"] += 1
                                elif enum_type in ["END", "PASS", "SELECT_END"]:
                                    stats["pass_end"] += 1
                                    
                    obs, _, done, info = env.step(choices)
                    
                    # Analyze engine execution logs
                    if obs and getattr(obs, 'logs', None):
                        for log in obs.logs:
                            log_type = getattr(log, 'type', 0)
                            if log_type == LogType.HP_CHANGE:
                                val = getattr(log, 'value', 0)
                                if val < 0:
                                    stats["damage_dealt"] += (-val)
                                    
                result = info.get("result", -1) if done else -1
                if result == 0:
                    model_stats[p0_name]["wins"] += 1
                elif result == 1:
                    model_stats[p1_name]["wins"] += 1
                env.close()
                print(f"Finished: {p0_name} (P0) vs {p1_name} (P1) | Winner: {p0_name if result==0 else p1_name if result==1 else 'Tie'}", flush=True)
                
    print("\n" + "="*70)
    print("EMPIRICAL IN-GAME BEHAVIOR METRICS SUMMARY")
    print("="*70)
    
    header = f"{'Model':<12} | {'WinRate':<7} | {'Attacks':<7} | {'Attach':<7} | {'Evo':<5} | {'Trainer':<7} | {'Pass/End':<8} | {'Avg Dmg/Game':<12}"
    print(header)
    print("-" * len(header))
    
    for name, s in model_stats.items():
        gp = max(s["games_played"], 1)
        wr = (s["wins"] / gp) * 100.0
        avg_atk = s["attacks"] / gp
        avg_att = s["energy_attach"] / gp
        avg_evo = s["evolutions"] / gp
        avg_trn = s["trainer_played"] / gp
        avg_pass = s["pass_end"] / gp
        avg_dmg = s["damage_dealt"] / gp
        
        print(f"{name:<12} | {wr:>6.1f}% | {avg_atk:>7.1f} | {avg_att:>7.1f} | {avg_evo:>5.1f} | {avg_trn:>7.1f} | {avg_pass:>8.1f} | {avg_dmg:>12.1f}")
        
    print("="*70)

if __name__ == "__main__":
    run_behavior_analysis()
