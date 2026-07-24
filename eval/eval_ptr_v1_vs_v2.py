#!/usr/bin/env python3
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from eval.eval_ptr_gameplay import *
from tcg_core.environment import TCGEnvironment
from tcg_core.models.ptr import PokemonAgent as PTRModel

def run_v1_vs_v2():
    deck_dir = os.path.join(ROOT, "new_deck")
    deck_path_0 = os.path.join(deck_dir, "Roaring Moon Ancient Depths.csv")
    deck_path_1 = os.path.join(deck_dir, "Roaring Moon Ancient Depths.csv")
    d0 = load_deck(deck_path_0)
    d1 = load_deck(deck_path_1)
    
    print("=== TCG PTR V1 vs V2 GAMEPLAY ANALYSIS ===")
    print(f"Deck: Mega Gardevoir (V1) vs Mega Gardevoir (V2)")
    
    checkpoints_dir = os.path.join(ROOT, "checkpoints")
    model_v1_path = os.path.join(checkpoints_dir, "model_lstm_pointer_final.msgpack")
    model_v2_path = os.path.join(checkpoints_dir, "model_lstm_pointer_v2_final.msgpack")
    
    agent_p0 = PointerAgent("PTR_V1", PTRModel, action_mapping, model_v1_path)
    agent_p1 = PointerAgent("PTR_V2", PTRModel, action_mapping, model_v2_path)
    
    agent_p0.reset()
    agent_p1.reset()
    
    env = TCGEnvironment()
    obs, done = env.reset(d0, d1)
    
    step_count = 0
    print("--- MULAI PERTANDINGAN ---")
    while not done and step_count <= 400:
        step_count += 1
        active_player = obs.current.yourIndex if obs.current else 0
        turn = obs.current.turn if obs.current else 0
        
        print(f"\n--- [Turn {turn} | Step {step_count}] ---")
        print_game_state(obs, active_player)
        
        if active_player == 0:
            choices = agent_p0.select_action(obs, deterministic=True)
            player_name = "P0 (PTR V1)"
            critic_val = agent_p0.last_value
        else:
            choices = agent_p1.select_action(obs, deterministic=True)
            player_name = "P1 (PTR V2)"
            critic_val = agent_p1.last_value
            
        action_desc = decode_action_log(obs, choices, active_player)
        print(f"  [ACTION] {player_name} memilih: {action_desc} | (Critic: {critic_val:+.3f})")
        
        obs, _, done, info = env.step(choices)
        
        if obs and getattr(obs, 'logs', None):
            for log in obs.logs:
                log_type = getattr(log, 'type', 0)
                if log_type == LogType.ATTACK:
                    print(f"  >>> ENGINE: Serangan Terjadi!")
                elif log_type == LogType.HP_CHANGE:
                    val = getattr(log, 'value', 0)
                    if val < 0:
                        print(f"  >>> ENGINE: Menerima Damage {-val} HP")
                    elif val > 0:
                        print(f"  >>> ENGINE: Heal {val} HP")
                elif log_type == LogType.DRAW:
                    print(f"  >>> ENGINE: Draw Kartu")
                elif log_type == LogType.PLAY:
                    card_id = getattr(log, 'card', 0)
                    card_name = f" [{get_card_name(card_id)}]" if card_id else ""
                    print(f"  >>> ENGINE: Memainkan Kartu{card_name}")
                elif log_type == LogType.EVOLVE:
                    print(f"  >>> ENGINE: Evolusi Pokemon")
                elif log_type == LogType.ATTACH:
                    print(f"  >>> ENGINE: Pasang Energi")

    result = info.get("result", -1) if done else -1
    print("\n--- PERTANDINGAN SELESAI ---")
    if result == 0:
        print("PEMENANG: P0 (PTR V1)")
    elif result == 1:
        print("PEMENANG: P1 (PTR V2)")
    else:
        print("HASIL: SERI / TIMEOUT")
        
    env.close()

if __name__ == "__main__":
    run_v1_vs_v2()
