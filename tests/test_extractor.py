import sys
import json
import random
from cg.game import battle_start, battle_finish, battle_select
from cg.api import to_dataclass, Observation, LogType

def load_deck(filepath):
    deck = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                deck.append(int(line))
    return deck

def run_full_game():
    import os
    import glob
    print("Memuat deck...")
    deck_path = "agent_rl/deck.csv"
    if not os.path.exists(deck_path):
        new_decks = glob.glob("new_deck/*.csv")
        if new_decks:
            deck_path = new_decks[0]
            print(f"agent_rl/deck.csv tidak ditemukan. Menggunakan fallback: {deck_path}")
        else:
            print("Tidak ada deck CSV ditemukan!")
            return
            
    try:
        deck = load_deck(deck_path)
    except Exception as e:
        print(f"Gagal memuat deck: {e}")
        return

    if not deck or len(deck) != 60:
        print(f"Jumlah kartu di deck tidak 60! (Ada {len(deck) if deck else 0} kartu)")
        return
        
    print("Memulai permainan...")
    try:
        obs_dict, start_data = battle_start(deck, deck)
        if obs_dict is None:
            print("Gagal memulai battle!")
            return
            
        obs = to_dataclass(obs_dict, Observation)
        
        from agent_rl.reward import calculate_step_reward, detect_events, reset_trackers
        reset_trackers()
        
        def get_end_reason(obs_data) -> int:
            if obs_data is None or not obs_data.logs:
                return 0
            for log in obs_data.logs:
                if log.type == LogType.RESULT:
                    return log.reason if log.reason is not None else 0
            return 0

        old_state = obs.current
        step = 0
        while True:
            
            if obs.current is not None and obs.current.result != -1:
                print(f"\n>>> GAME OVER! Pemenang (Player Index): {obs.current.result} <<<")
                
                # Ekstrak Fitur pada saat Game Over untuk diperlihatkan ke user
                print("\nMenjalankan extract_features() pada state terakhir...")
                from agent_rl.feature_extractor import extract_features
                import numpy as np
                features = extract_features(obs.current, obs.select, 0)
                seq = features["seq_input"]
                glob = features["glob_input"]
                
                print(f"\n[HASIL FEATURE EXTRACTION DI AKHIR GAME]")
                print(f"Bentuk seq_input: {seq.shape} | Bentuk glob_input: {glob.shape}")
                
                print("\nIsi 'seq_input' yang memiliki data (tidak kosong):")
                # Hanya print baris yang ID-nya (kolom 0) tidak nol atau is_present (kolom 15) tidak nol
                for i in range(93):
                    if seq[i, 15] > 0: # jika slot ini ada isinya (is_present == 1.0)
                        if i < 20: slot_name = f"My Hand [{i}]"
                        elif i < 50: slot_name = f"My Discard [{i-20}]"
                        elif i < 80: slot_name = f"Opp Discard [{i-50}]"
                        elif i == 80: slot_name = f"My Active [0]"
                        elif i < 86: slot_name = f"My Bench [{i-81}]"
                        elif i == 86: slot_name = f"Opp Active [0]"
                        elif i < 92: slot_name = f"Opp Bench [{i-87}]"
                        else: slot_name = "Stadium"
                        
                        card_id = seq[i, 0]
                        hp_frac = seq[i, 16]
                        energies = seq[i, 3] * 10.0
                        print(f"Slot {i:2d} ({slot_name:15s}) -> ID: {card_id:5.0f} | HP: {hp_frac:.2f} | Attached Energy: {energies:.0f}")
                
                print("\n[Beberapa nilai dari 'glob_input']")
                print(f"Turn: {glob[0]*100:.0f} | P0 Deck/60: {glob[8]*60:.0f} | P1 Deck/60: {glob[9]*60:.0f}")
                print(f"P0 Prizes Taken: {glob[10]*6:.0f} | P1 Prizes Taken: {glob[11]*6:.0f}")
                
                break
                
            if step > 1000:
                print("\n>>> GAME DIHENTIKAN! Mencapai batas 1000 step (mungkin bot stuck) <<<")
                break
                
            # Cetak LOGS
            if obs.logs:
                for log in obs.logs:
                    try:
                        log_name = LogType(log.type).name
                    except:
                        log_name = str(log.type)
                    
                    # Buat deskripsi log yang lebih terbaca
                    details = []
                    if log.playerIndex is not None: details.append(f"P{log.playerIndex}")
                    if log.cardId is not None: details.append(f"CardID:{log.cardId}")
                    if log.value is not None: details.append(f"Val:{log.value}")
                    if log.fromArea is not None: details.append(f"From:{log.fromArea}")
                    if log.toArea is not None: details.append(f"To:{log.toArea}")
                    
                    print(f"[Turn {obs.current.turn if obs.current else 0} | Step {step}] LOG: {log_name} | {' '.join(details)}")
            
            if obs.select is not None:
                min_c = obs.select.minCount
                max_c = obs.select.maxCount
                opt_count = len(obs.select.option)
                
                # Pemilihan aksi acak yang valid
                choices = []
                if min_c > 0:
                    # Harus memilih minimal min_c
                    if max_c > opt_count: max_c = opt_count
                    if max_c < min_c: max_c = min_c
                    pick_count = random.randint(min_c, max_c)
                    choices = random.sample(range(opt_count), pick_count)
                else:
                    # Boleh skip (Pilih []). 50% kemungkinan skip.
                    if opt_count > 0 and random.random() > 0.5:
                        pick_count = random.randint(1, min(max_c, opt_count))
                        choices = random.sample(range(opt_count), pick_count)
                    else:
                        choices = []
                        
                # Sorting choices descending is usually required if multiple elements from same array are selected? 
                # C++ engine usually doesn't care if it's just indices of select.option
                try:
                    prev_player = obs.current.yourIndex if obs.current else 0
                    obs_dict = battle_select(choices)
                    obs = to_dataclass(obs_dict, Observation)
                    
                    if obs.current:
                        end_reason = get_end_reason(obs)
                        events = detect_events(old_state, obs.current, prev_player, obs.logs)
                        step_reward = calculate_step_reward(obs.current, prev_player, events, end_reason)
                        print(f"[Reward Tester] Step {step} selesai. Player: P{prev_player} | Reward: {step_reward:.4f} | Events: {events}")
                        old_state = obs.current
                except Exception as e:
                    print(f"Error saat execute select {choices}: {e}")
                    break
            else:
                break
                
            step += 1
            
    finally:
        try:
            battle_finish()
        except:
            pass

if __name__ == "__main__":
    run_full_game()
