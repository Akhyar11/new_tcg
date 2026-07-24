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

def main():
    deck_dir = os.path.join(ROOT, "new_deck")
    checkpoints_dir = os.path.join(ROOT, "checkpoints")
    model_v1_path = os.path.join(checkpoints_dir, "model_lstm_pointer_final.msgpack")
    model_v2_path = os.path.join(checkpoints_dir, "model_lstm_pointer_v2_final.msgpack")
    
    agent_p0 = PointerAgent("PTR_V1", PTRModel, action_mapping, model_v1_path)
    agent_p1 = PointerAgent("PTR_V2", PTRModel, action_mapping, model_v2_path)
    
    env = TCGEnvironment()
    
    csv_files = [f for f in os.listdir(deck_dir) if f.endswith('.csv')]
    csv_files.sort()
    
    v1_wins = 0
    v2_wins = 0
    ties = 0
    
    print(f"Memulai evaluasi untuk {len(csv_files)} deck...")
    
    for idx, deck_file in enumerate(csv_files):
        deck_path = os.path.join(deck_dir, deck_file)
        try:
            d0 = load_deck(deck_path)
            d1 = load_deck(deck_path)
        except Exception as e:
            print(f"[{idx+1}/{len(csv_files)}] Error memuat deck {deck_file}: {e}")
            continue
            
        agent_p0.reset()
        agent_p1.reset()
        
        obs, done = env.reset(d0, d1)
        step_count = 0
        
        while not done and step_count <= 400:
            step_count += 1
            if obs is None:
                print(f"[{idx+1}/{len(csv_files)}] Obs is None, breaking.")
                break
            active_player = obs.current.yourIndex if obs.current else 0
            
            if active_player == 0:
                choices = agent_p0.select_action(obs, deterministic=True)
            else:
                choices = agent_p1.select_action(obs, deterministic=True)
                
            obs, _, done, info = env.step(choices)
            
        result = info.get("result", -1) if done else -1
        
        if result == 0:
            v1_wins += 1
            res_str = "V1 Menang"
        elif result == 1:
            v2_wins += 1
            res_str = "V2 Menang"
        else:
            ties += 1
            res_str = "Seri/Timeout"
            
        print(f"[{idx+1}/{len(csv_files)}] Deck: {deck_file[:-4]} => {res_str} ({step_count} step)")
        
    env.close()
    
    print("\n" + "="*40)
    print("HASIL EVALUASI PTR V1 vs PTR V2")
    print("="*40)
    print(f"Total Deck     : {len(csv_files)}")
    print(f"PTR V1 Menang  : {v1_wins}")
    print(f"PTR V2 Menang  : {v2_wins}")
    print(f"Seri/Timeout   : {ties}")
    print("="*40)

if __name__ == "__main__":
    main()
