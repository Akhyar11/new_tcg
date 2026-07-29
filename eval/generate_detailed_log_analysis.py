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

from eval.eval_ptr_gameplay import decode_action_log, get_card_name
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

def generate_full_game_logs():
    checkpoints_dir = os.path.join(ROOT, "checkpoints")
    deck_path = os.path.join(ROOT, "new_deck", "Roaring Moon Ancient Depths.csv")
    if not os.path.exists(deck_path):
        deck_path = glob.glob(os.path.join(ROOT, "deck_generated", "*.csv"))[0]
    
    deck = load_deck(deck_path)
    output_dir = os.path.join(ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    agent_ff = FFAgent("FF", FFModel, action_mapping, os.path.join(checkpoints_dir, "model_final.msgpack"))
    agent_lstm = LSTMAgent("LSTM", LSTMModel, action_mapping, os.path.join(checkpoints_dir, "model_lstm_final.msgpack"))
    agent_ptr1 = PointerAgent("LSTM_PTR_V1", LSTMPointerModel, action_mapping, os.path.join(checkpoints_dir, "model_lstm_pointer_final.msgpack"))
    agent_ptr2 = PointerAgent("LSTM_PTR_V2", LSTMPointerModel, action_mapping, os.path.join(checkpoints_dir, "model_lstm_pointer_v2_final.msgpack"))
    
    matchups = [
        ("FF", agent_ff, "LSTM", agent_lstm, "game_log_ff_vs_lstm.txt"),
        ("LSTM", agent_lstm, "LSTM_PTR_V1", agent_ptr1, "game_log_lstm_vs_ptr1.txt"),
        ("LSTM_PTR_V1", agent_ptr1, "LSTM_PTR_V2", agent_ptr2, "game_log_ptr1_vs_ptr2.txt")
    ]
    
    for m1_name, agent1, m2_name, agent2, filename in matchups:
        log_file = os.path.join(output_dir, filename)
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"=== DETAILED IN-GAME ACTION LOG: {m1_name} vs {m2_name} ===\n\n")
            
            agent1.reset()
            agent2.reset()
            
            env = TCGEnvironment()
            obs, done = env.reset(deck, deck)
            
            step = 0
            while not done and step <= 150:
                step += 1
                active_p = obs.current.yourIndex if obs.current else 0
                turn = obs.current.turn if obs.current else 0
                curr_name = m1_name if active_p == 0 else m2_name
                curr_agent = agent1 if active_p == 0 else agent2
                
                choices = curr_agent.select_action(obs, deterministic=True)
                action_desc = decode_action_log(obs, choices, active_p)
                
                f.write(f"[Turn {turn} | Step {step}] {curr_name} (P{active_p}): {action_desc}\n")
                
                obs, _, done, info = env.step(choices)
                
                if obs and getattr(obs, 'logs', None):
                    for log in obs.logs:
                        log_type = getattr(log, 'type', 0)
                        if log_type == LogType.ATTACK:
                            f.write(f"    >>> ENGINE: Serangan Terjadi!\n")
                        elif log_type == LogType.HP_CHANGE:
                            val = getattr(log, 'value', 0)
                            if val < 0:
                                f.write(f"    >>> ENGINE: Damage {-val} HP\n")
                            elif val > 0:
                                f.write(f"    >>> ENGINE: Heal {val} HP\n")
                        elif log_type == LogType.PLAY:
                            card_id = getattr(log, 'card', 0)
                            f.write(f"    >>> ENGINE: Memainkan Kartu [{get_card_name(card_id)}]\n")
                        elif log_type == LogType.EVOLVE:
                            f.write(f"    >>> ENGINE: Evolusi Pokemon\n")
                        elif log_type == LogType.ATTACH:
                            f.write(f"    >>> ENGINE: Pasang Energi\n")
                            
            result = info.get("result", -1) if done else -1
            winner_str = f"PEMENANG: {m1_name} (P0)" if result == 0 else f"PEMENANG: {m2_name} (P1)" if result == 1 else "HASIL: SERI/TIMEOUT"
            f.write(f"\n--- MATCH END: {winner_str} ---\n")
            env.close()
            print(f"Logged: {log_file}")

if __name__ == "__main__":
    generate_full_game_logs()
