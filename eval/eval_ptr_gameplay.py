#!/usr/bin/env python3
"""
PTR Gameplay Analysis Script
Simulates 1 game of PTR vs PTR with identical decks to analyze decision making.
"""
import os
import sys
import random
import time
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from tcg_core.environment import TCGEnvironment
from tcg_core.agents import LSTMAgent
from tcg_core.models.ptr import PokemonAgent as PTRModel
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

import json

import csv

# Muat database kartu untuk menampilkan nama kartu
CARD_DB = {}
try:
    with open(os.path.join(ROOT, "cg", "database.csv"), "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader) # Skip header
        for row in reader:
            if len(row) >= 2:
                try:
                    CARD_DB[int(row[0])] = row[1]
                except ValueError:
                    pass
except Exception as e:
    print(f"[WARNING] Gagal memuat database.csv: {e}, hanya akan menampilkan ID.")

def get_card_name(card_id):
    return CARD_DB.get(card_id, f"ID:{card_id}")

def decode_action_log(obs, choices, active_player):
    """Membantu membaca aksi apa yang baru saja dilakukan agen dengan lebih detail"""
    if not obs or not obs.select or not obs.select.option:
        return "No options / Auto-forward"
    
    actions = []
    for c in choices:
        if c < len(obs.select.option):
            opt = obs.select.option[c]
            
            # Perbaiki pengambilan nama tipe Enum
            try:
                tipe_enum = OptionType(opt.type)
                tipe = tipe_enum.name
            except Exception:
                tipe = str(opt.type)
            
            # Ekstrak informasi kartu jika memungkinkan
            card_info = ""
            try:
                my_state = obs.current.players[active_player]
                if tipe == "PLAY" and hasattr(my_state, 'hand'):
                    card_id = my_state.hand[opt.index].id
                    card_info = f" [{get_card_name(card_id)}]"
                elif tipe == "EVOLVE" and hasattr(my_state, 'hand'):
                    card_id = my_state.hand[opt.index].id
                    card_info = f" [{get_card_name(card_id)}]"
                elif tipe == "ATTACH" and hasattr(my_state, 'hand'):
                    card_id = my_state.hand[opt.index].id
                    card_info = f" [{get_card_name(card_id)}]"
                elif tipe == "CARD_DECK" and hasattr(obs.select, 'deck'):
                    card_id = obs.select.deck[opt.index].id
                    card_info = f" [{get_card_name(card_id)}]"
                elif tipe == "CARD":
                    if getattr(opt, 'area', None) == 2 and hasattr(my_state, 'hand'): # 2 is AreaType.HAND
                        card_id = my_state.hand[opt.index].id
                        card_info = f" [-> {get_card_name(card_id)}]"
                    elif getattr(opt, 'area', None) == 1 and hasattr(obs.select, 'deck') and obs.select.deck: # 1 is AreaType.DECK
                        card_id = obs.select.deck[opt.index].id
                        card_info = f" [-> {get_card_name(card_id)}]"
                    elif hasattr(obs.select, 'deck') and obs.select.deck:
                        card_id = obs.select.deck[opt.index].id 
                        card_info = f" [-> {get_card_name(card_id)}]"
                elif tipe == "ATTACK":
                    card_info = f" [Move Slot: {opt.index + 1}]"
                elif tipe == "NUMBER":
                    card_info = f" [Value: {getattr(opt, 'number', '?')}]"
            except Exception as e:
                pass
                
            idx_str = f" (idx:{opt.index})" if getattr(opt, 'index', None) is not None else ""
            actions.append(f"{tipe}{idx_str}{card_info}")
    
    if not actions:
        actions.append("PASS/END")
    return " | ".join(actions)

def print_game_state(obs, active_player):
    """Mencetak kondisi arena dan tangan pemain secara detail."""
    if not obs or not getattr(obs, 'current', None): return
    my_state = obs.current.players[active_player]
    opp_state = obs.current.players[1 - active_player]
    
    def format_pokemon(p):
        if not p or getattr(p, 'id', 0) == 0: return "Kosong"
        hp = getattr(p, 'hp', '?')
        max_hp = getattr(p, 'maxHp', '?')
        return f"{get_card_name(p.id)} (HP: {hp}/{max_hp})"
    
    my_active_list = getattr(my_state, 'active', [])
    my_active = format_pokemon(my_active_list[0] if my_active_list else None)
    
    opp_active_list = getattr(opp_state, 'active', [])
    opp_active = format_pokemon(opp_active_list[0] if opp_active_list else None)
    
    my_hand = [get_card_name(c.id) for c in getattr(my_state, 'hand', [])]
    my_bench = len([p for p in getattr(my_state, 'bench', []) if getattr(p, 'id', 0) != 0])
    opp_bench = len([p for p in getattr(opp_state, 'bench', []) if getattr(p, 'id', 0) != 0])
    
    print(f"  [STATE P{active_player}] Active: {my_active} | Bench: {my_bench} | Prizes: {getattr(my_state, 'prizes', '?')} | Hand ({len(my_hand)}): {', '.join(my_hand)}")
    print(f"  [STATE P{1 - active_player}] Active: {opp_active} | Bench: {opp_bench} | Prizes: {getattr(opp_state, 'prizes', '?')}")


def main():
    deck_dir = os.path.join(ROOT, "new_deck")
    deck_path = os.path.join(deck_dir, "Mega Charizard Y Sniper.csv")
    d0 = load_deck(deck_path)
    d1 = load_deck(deck_path)
    
    print("=== TCG PTR GAMEPLAY ANALYSIS ===")
    print(f"Deck: Mega Charizard Y Sniper (1v1 PTR vs PTR)")
    
    checkpoints_dir = os.path.join(ROOT, "checkpoints")
    model_path = os.path.join(checkpoints_dir, "model_lstm_pointer_final.msgpack")
    lstm_model_path = os.path.join(checkpoints_dir, "model_lstm_final.msgpack")
    
    if not os.path.exists(model_path) or not os.path.exists(lstm_model_path):
        print("\n[PERINGATAN]: Checkpoint tidak ditemukan! Agen akan menggunakan bobot ACAK.")
        print("Silakan jalankan `python train_ptr.py` terlebih dahulu jika ingin melihat taktik aslinya.\n")
    else:
        print("\n[INFO]: Menggunakan model terlatih dari checkpoint!\n")
    
    from tcg_core.agents import LSTMAgent
    from tcg_core.models.lstm import PokemonAgent as LSTMModel
    
    agent_p0 = PointerAgent("PTR_P0", PTRModel, action_mapping, model_path if os.path.exists(model_path) else None)
    agent_p1 = PointerAgent("PTR_P1", PTRModel, action_mapping, model_path if os.path.exists(model_path) else None)
    
    agent_p0.reset()
    agent_p1.reset()
    
    env = TCGEnvironment()
    obs, done = env.reset(d0, d1)
    
    step_count = 0
    print("--- MULAI PERTANDINGAN ---")
    while not done and step_count <= 300:  # Batasi 300 step agar log tidak penuh
        step_count += 1
        active_player = obs.current.yourIndex if obs.current else 0
        turn = obs.current.turn if obs.current else 0
        
        # Cetak state sebelum action
        print(f"\n--- [Turn {turn} | Step {step_count}] ---")
        print_game_state(obs, active_player)
        
        # Ekstrak fitur & pilih aksi
        if active_player == 0:
            choices = agent_p0.select_action(obs)
            player_name = "P0 (PTR)"
        else:
            choices = agent_p1.select_action(obs)
            player_name = "P1 (PTR)"
            
        action_desc = decode_action_log(obs, choices, active_player)
        critic_val = agent_p0.last_value if active_player == 0 else agent_p1.last_value
        print(f"  [ACTION] {player_name} memilih: {action_desc} | (Critic: {critic_val:+.3f})")
        
        # Eksekusi di C++
        obs, _, done, info = env.step(choices)
        
        # Cetak log dari engine C++ untuk mengonfirmasi apa yang terjadi
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
        print("PEMENANG: P0 (PTR)")
    elif result == 1:
        print("PEMENANG: P1 (PTR)")
    else:
        print("HASIL: SERI / TIMEOUT")
        
    env.close()

if __name__ == "__main__":
    main()
