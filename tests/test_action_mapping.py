import numpy as np
from cg.api import OptionType
from tcg_core.action_mapping import (
    create_action_mask, decode_action, get_action_index_for_option,
    PLAY_START, CARD_START, ATTACK_START, ACTION_END
)

def test_action_mapping():
    print("Testing action mapping...")
    
    # 1. Buat simulasi select_data dari C++ engine
    # Bayangkan mesin memberi kita 3 opsi:
    # Opsi 0: Mainkan kartu dari hand indeks 5 (OptionType.PLAY)
    # Opsi 1: Serang menggunakan serangan indeks 1 (OptionType.ATTACK)
    # Opsi 2: Akhiri Giliran (OptionType.END)
    select_data_mock = {
        "options": [
            {"type": "PLAY", "index": 5},
            {"type": "ATTACK", "index": 1},
            {"type": "END", "index": 0},
            {"type": "CARD", "index": 12}
        ]
    }
    
    print("\n--- TEST 1: Pembuatan Action Mask ---")
    mask = create_action_mask(select_data_mock)
    print(f"Total opsi legal (harus 4): {np.sum(mask)}")
    
    # Verifikasi posisi mask
    print(f"Apakah mask PLAY_START+5 ({PLAY_START+5}) menyala? {mask[PLAY_START+5] == 1.0}")
    print(f"Apakah mask ATTACK_START+1 ({ATTACK_START+1}) menyala? {mask[ATTACK_START+1] == 1.0}")
    print(f"Apakah mask ACTION_END ({ACTION_END}) menyala? {mask[ACTION_END] == 1.0}")
    print(f"Apakah mask CARD_START+12 ({CARD_START+12}) menyala? {mask[CARD_START+12] == 1.0}")
    
    # Verifikasi posisi lain yang harusnya mati
    print(f"Apakah mask PLAY_START+0 menyala? {mask[PLAY_START+0] == 1.0} (Harus False)")
    
    print("\n--- TEST 2: Decode Action (Kembali ke Indeks C++) ---")
    # Anggaplah JAX Model AI (PPO) memprediksi dan memilih aksi `ATTACK_START + 1` (yaitu indeks 163)
    ai_chosen_action = ATTACK_START + 1
    print(f"AI JAX Memilih Aksi Indeks Baku: {ai_chosen_action}")
    
    cpp_indices = decode_action(ai_chosen_action, select_data_mock)
    print(f"Decode Action Output: {cpp_indices}")
    print(f"Apakah benar ini mengarah ke Opsi ke-1? {cpp_indices == [1]}")
    
    # Anggaplah JAX memilih `ACTION_END` (indeks 160)
    ai_chosen_action_2 = ACTION_END
    print(f"\nAI JAX Memilih Aksi Indeks Baku: {ai_chosen_action_2}")
    cpp_indices_2 = decode_action(ai_chosen_action_2, select_data_mock)
    print(f"Decode Action Output: {cpp_indices_2}")
    print(f"Apakah benar ini mengarah ke Opsi ke-2? {cpp_indices_2 == [2]}")

if __name__ == "__main__":
    test_action_mapping()
