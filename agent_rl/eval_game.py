import os
import sys
import glob
import random
import time
import numpy as np

# Ensure cg can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from flax import serialization

from cg.game import battle_start, battle_finish, battle_select
from cg.api import to_dataclass, Observation, OptionType
from agent_rl.feature_extractor import extract_features
from agent_rl.action_mapping import decode_action
from agent_rl.model import PokemonAgent

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def load_deck(filepath):
    deck = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                deck.append(int(line))
    return deck

def print_logs(logs, cards=None):
    if not logs:
        return
    print("\n--- LOG PERTANDINGAN ---")
    from cg.api import LogType, AreaType
    for log in logs:
        # Sembunyikan log rahasia lawan yang hanya nyampah di layar
        if log.type in [LogType.MOVE_CARD_REVERSE, LogType.DRAW_REVERSE, LogType.SHUFFLE]:
            continue
            
        parts = []
        try:
            type_name = LogType(log.type).name
        except:
            type_name = str(log.type)
        parts.append(f"type={type_name}")
        
        for field, val in log.__dict__.items():
            if field == 'type': continue
            if val is not None:
                if field.startswith('cardId') and cards and val in cards:
                    cname = cards[val].get('Card Name', 'Unknown')
                    parts.append(f"{field}={val}[{cname}]")
                elif field in ['fromArea', 'toArea']:
                    try:
                        area_name = AreaType(val).name
                        parts.append(f"{field}={area_name}")
                    except:
                        parts.append(f"{field}={val}")
                else:
                    parts.append(f"{field}={val}")
                    
        print(f"Log({', '.join(parts)})")
    print("------------------------\n")

def run_eval_game():
    print("=== TCG AI EVALUATION ===")
    
    deck_path = "agent_rl/deck"
    deck_files = glob.glob(os.path.join(deck_path, "*.csv"))
    if not deck_files:
        print("Deck files not found!")
        return

    # 1. Load Model
    print("Memuat model terlatih...")
    model = PokemonAgent(num_actions=250)
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    dummy_seq = jnp.zeros((1, 93, 31))
    dummy_glob = jnp.zeros((1, 266))
    params = model.init(init_rng, dummy_seq, dummy_glob)
    
    cp_path = "checkpoints/model_final.msgpack"
    if os.path.exists(cp_path):
        with open(cp_path, 'rb') as f:
            params = serialization.from_bytes(params, f.read())
        print(f"Berhasil memuat checkpoint: {cp_path}")
    else:
        print("Checkpoint tidak ditemukan, menggunakan model dengan bobot acak!")

    model_apply = jax.jit(model.apply)

    # 2. Setup Game
    battle_finish()
    file0 = random.choice(deck_files)
    file1 = random.choice(deck_files)
    print(f"\nDeck Player 0: {os.path.basename(file0)}")
    print(f"Deck Player 1: {os.path.basename(file1)}")
    
    deck0 = load_deck(file0)
    deck1 = load_deck(file1)
    
    try:
        obs_dict, _ = battle_start(deck0, deck1)
        obs = to_dataclass(obs_dict, Observation)
    except Exception as e:
        print(f"Gagal memulai battle: {e}")
        return

    # Load Card Database for naming
    import csv
    cards = {}
    csv_path = "agent_rl/EN_Card_Data.csv"
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cards[int(row['Card ID'])] = row

    step = 0
    while obs.current is not None and obs.current.result == -1:
        step += 1
        print(f"\n[ STEP {step} ]")
        
        # Cetak log dari step sebelumnya
        print_logs(obs.logs, cards)
        
        your_index = obs.current.yourIndex
        print(f"Giliran: Player {your_index}")
        
        # --- BOARD STATE SUMMARY ---
        print("\n--- STATUS PAPAN ---")
        for i in range(2):
            player = obs.current.players[i]
            p_name = "Player 0" if i == 0 else "Player 1"
            
            # Active Pokemon
            active_str = "KOSONG"
            if player.active and player.active[0]:
                pkm = player.active[0]
                cname = cards.get(pkm.id, {}).get('Card Name', f'Unknown_{pkm.id}')
                active_str = f"{cname} (HP: {pkm.hp}/{pkm.maxHp}, Energy: {len(pkm.energies)})"
            elif player.active and player.active[0] is None:
                active_str = "Facedown (Belum Dibuka)"
                
            # Bench
            bench_str = []
            for b in player.bench:
                bcname = cards.get(b.id, {}).get('Card Name', f'Unknown_{b.id}')
                bench_str.append(f"{bcname}({b.hp})")
            b_str = ", ".join(bench_str) if bench_str else "Kosong"
            
            print(f"[{p_name}] Active: {active_str} | Bench: {b_str}")
        print("--------------------\n")
        
        if obs.select is not None and obs.select.option:
            opt_count = len(obs.select.option)
            min_c = obs.select.minCount
            max_c = obs.select.maxCount
            print(f"Pilihan Tersedia: {opt_count} opsi (Pilih {min_c} sampai {max_c})")
            
            # Print semua opsi
            from agent_rl.action_mapping import get_action_index_for_option
            for i, opt in enumerate(obs.select.option):
                opt_type_name = OptionType(opt.type).name
                mock_opt = {"type": opt_type_name, "index": opt.index}
                ai_idx = get_action_index_for_option(mock_opt)
                print(f"  Opsi {i}: Tipe {opt_type_name}, Index {opt.index} (=> Kode AI: {ai_idx})")
            
            # Ekstrak fitur
            features = extract_features(obs.current, obs.select, your_index)
            seq_input = np.expand_dims(features["seq_input"], axis=0)
            glob_input = np.expand_dims(features["glob_input"], axis=0)
            
            # Print sedikit info input
            print(f"  > AI Input Shape - Seq: {seq_input.shape}, Glob: {glob_input.shape}")
            
            # Inferensi
            masked_logits, _ = model_apply(params, seq_input, glob_input)

            # --- FIX: Categorical sampling tanpa pengembalian (sama seperti training) ---
            logits_np = np.array(masked_logits[0])

            # 1. Build action mask
            mock_select_dict = {"options": [{"type": OptionType(o.type).name, "index": o.index} for o in obs.select.option]}
            from agent_rl.action_mapping import get_action_index_for_option, create_action_mask
            mask_array = create_action_mask(mock_select_dict)

            # 2. Mask logits
            masked = logits_np - 1e9 * (1.0 - mask_array)

            # 3. Softmax → probs
            logits_exp = np.exp(masked - np.max(masked))
            probs = logits_exp / (logits_exp.sum() + 1e-10)

            # 4. Sample min_c tanpa pengembalian
            sampled_indices = []
            remaining = probs.copy()
            for _ in range(min_c):
                if remaining.sum() <= 0:
                    break
                p = remaining / remaining.sum()
                idx = int(np.random.choice(len(p), p=p))
                sampled_indices.append(idx)
                remaining[idx] = 0.0

            # 5. Map ke C++ options
            choices = []
            for jax_idx in sampled_indices:
                for cpp_idx, opt in enumerate(mock_select_dict["options"]):
                    mapped_idx = get_action_index_for_option(opt)
                    if mapped_idx == jax_idx and cpp_idx not in choices:
                        choices.append(cpp_idx)
                        break

            print(f"  > Sampled JAX indices: {sampled_indices} → C++ choices: {choices} (minCount={min_c})")
            
            try:
                obs_dict = battle_select(choices)
                obs = to_dataclass(obs_dict, Observation)
            except Exception as e:
                # Coba fallback bot jika AI gagal parse
                fallback_choices = list(range(min(opt_count, min_c))) if min_c > 0 else []
                print(f"!!! Error eksekusi aksi: {e}. Menggunakan fallback {fallback_choices}...")
                try:
                    obs_dict = battle_select(fallback_choices)
                    obs = to_dataclass(obs_dict, Observation)
                except Exception as e2:
                    print(f"Fallback juga gagal: {e2}")
                    break
                    
            # Tambahkan sedikit jeda agar mudah dibaca jika dieksekusi di terminal
            time.sleep(0.1)
        else:
            print("Tidak ada pilihan untuk player saat ini, game stuck/berakhir?")
            break
            
    print("\n=== PERTANDINGAN SELESAI ===")
    if obs.current:
        print_logs(obs.logs, cards)
        result = obs.current.result
        if result == 0:
            print("Pemenang: Player 0!")
        elif result == 1:
            print("Pemenang: Player 1!")
        elif result == 2:
            print("Pertandingan Berakhir Seri (Draw).")
        else:
            print(f"Result code tidak dikenal: {result}")
            
    battle_finish()

if __name__ == "__main__":
    run_eval_game()
