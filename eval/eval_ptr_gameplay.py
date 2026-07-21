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
            except Exception as e:
                pass
                
            actions.append(f"{tipe} (idx:{opt.index}){card_info}")
    
    if not actions:
        actions.append("PASS/END")
    return " | ".join(actions)

def main():
    deck_dir = os.path.join(ROOT, "new_deck")
    deck_path = os.path.join(deck_dir, "Phantom Dive Sweep.csv")
    d0 = load_deck(deck_path)
    d1 = load_deck(deck_path)
    
    print("=== TCG PTR GAMEPLAY ANALYSIS ===")
    print(f"Deck: Phantom Dive Sweep (1v1 PTR vs LSTM)")
    
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
    agent_p1 = LSTMAgent("LSTM_P1", LSTMModel, action_mapping, lstm_model_path if os.path.exists(lstm_model_path) else None)
    
    agent_p0.reset()
    agent_p1.reset()
    
    env = TCGEnvironment()
    obs, done = env.reset(d0, d1)
    
    step_count = 0
    print("--- MULAI PERTANDINGAN ---")
    while not done and step_count <= 100:  # Batasi 100 step agar log tidak penuh
        step_count += 1
        active_player = obs.current.yourIndex if obs.current else 0
        turn = obs.current.turn if obs.current else 0
        
        # Ekstrak fitur & pilih aksi
        if active_player == 0:
            choices = agent_p0.select_action(obs)
            player_name = "P0 (PTR)"
        else:
            choices = agent_p1.select_action(obs)
            player_name = "P1 (LSTM)"
            
        action_desc = decode_action_log(obs, choices, active_player)
        print(f"[Turn {turn} | Step {step_count}] {player_name} memilih aksi: {action_desc}")
        
        # Eksekusi di C++
        obs, _, done, info = env.step(choices)
        
        # Cetak log dari engine C++ untuk mengonfirmasi apa yang terjadi
        if obs and obs.logs:
            for log in obs.logs:
                if getattr(log, 'type', 0) == LogType.ATTACK:
                    print(f"  >>> SERANGAN TERJADI! Damage = {getattr(log, 'damage', 0)}")
                elif getattr(log, 'type', 0) == LogType.DRAW:
                    print(f"  >>> DRAW KARTU")
                elif getattr(log, 'type', 0) == LogType.PLAY:
                    print(f"  >>> MEMAINKAN KARTU")
                    
    result = info.get("result", -1) if done else -1
    print("\n--- PERTANDINGAN SELESAI ---")
    if result == 0:
        print("PEMENANG: P0 (PTR)")
    elif result == 1:
        print("PEMENANG: P1 (LSTM)")
    else:
        print("HASIL: SERI / TIMEOUT")
        
    env.close()

if __name__ == "__main__":
    main()
