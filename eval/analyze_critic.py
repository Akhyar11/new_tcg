#!/usr/bin/env python3
import os
import sys
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from eval.eval_ptr_gameplay import PointerAgent, load_deck
from tcg_core.environment import TCGEnvironment
from tcg_core.models.ptr import PokemonAgent as PTRModel
import tcg_core.action_mapping as action_mapping
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_plot", type=str, default="critic_plot.png")
    args = parser.parse_args()

    deck_dir = os.path.join(ROOT, "new_deck")
    deck_path_0 = os.path.join(deck_dir, "Pikachu ex Synergy.csv")
    deck_path_1 = os.path.join(deck_dir, "Mega Gardevoir's Symphonia.csv")
    d0 = load_deck(deck_path_0)
    d1 = load_deck(deck_path_1)
    
    print("=== CRITIC VALUE ANALYSIS (LSTM PTR V2) ===")
    
    checkpoints_dir = os.path.join(ROOT, "checkpoints")
    model_v2_path = os.path.join(checkpoints_dir, "model_lstm_pointer_v2_final.msgpack")
    
    if not os.path.exists(model_v2_path):
        print("Model PTR V2 tidak ditemukan.")
        return
        
    agent_p0 = PointerAgent("PTR_V2_P0", PTRModel, action_mapping, model_v2_path)
    agent_p1 = PointerAgent("PTR_V2_P1", PTRModel, action_mapping, model_v2_path)
    
    agent_p0.reset()
    agent_p1.reset()
    
    env = TCGEnvironment()
    obs, done = env.reset(d0, d1)
    
    step_count = 0
    p0_turns = []
    p0_values = []
    
    last_recorded_turn = -1
    
    print("--- MULAI PERTANDINGAN ---")
    while not done and step_count <= 400:
        step_count += 1
        active_player = obs.current.yourIndex if obs.current else 0
        turn = obs.current.turn if obs.current else 0
        
        if active_player == 0:
            choices = agent_p0.select_action(obs, deterministic=True)
            p_name = "P0"
            c_val = agent_p0.last_value
            
            # Hanya rekam 1 kali di setiap giliran (Turn) untuk P0
            if turn != last_recorded_turn:
                p0_turns.append(turn)
                p0_values.append(c_val)
                last_recorded_turn = turn
                print(f"[Turn {turn:02d} | Step {step_count:03d}] P0 starts turn. Critic Val: {c_val:+.3f}")
        else:
            choices = agent_p1.select_action(obs, deterministic=True)
            p_name = "P1"
            
        obs, reward, done, info = env.step(choices)
        
    print(f"\nPertandingan Selesai dalam {step_count} step ({turn} Turn).")
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(p0_turns, p0_values, label='P0 Critic Value (P0 Win Prob)', color='blue', marker='o', linewidth=2)
    plt.axhline(0, color='gray', linestyle='--')
    plt.axhline(1.0, color='green', linestyle=':', alpha=0.5)
    plt.axhline(-1.0, color='red', linestyle=':', alpha=0.5)
    plt.xlabel("Turn")
    plt.ylabel("Critic Value (-5 to 5)")
    plt.title(f"LSTM PTR V2 Internal Value Head over Match\n(P0's Perspective Only)")
    plt.legend()
    plt.grid(True)
    
    plt.savefig(args.save_plot)
    print(f"Plot berhasil disimpan di {args.save_plot}")

if __name__ == "__main__":
    main()
