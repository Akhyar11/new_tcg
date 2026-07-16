import numpy as np

# =========================================================
# Konfigurasi Pemetaan Baku 250 Aksi (Fixed Action Space)
# Mengacu pada dokumen output_sistem.md (Pendekatan 1)
# =========================================================

NUM_ACTIONS = 250

# Rentang Aksi Kartu (Sesuai dengan OptionType dari C++)
PLAY_START, PLAY_END = 0, 59         # OptionType.PLAY (Mainkan kartu dari Hand)
CARD_START, CARD_END = 60, 119       # OptionType.CARD (Pilih kartu spesifik)
ATTACH_START, ATTACH_END = 120, 139  # OptionType.ATTACH
EVOLVE_START, EVOLVE_END = 140, 159  # OptionType.EVOLVE

# Aksi Game Spesifik
ACTION_END = 160                     # OptionType.END
ACTION_RETREAT = 161                 # OptionType.RETREAT
ATTACK_START, ATTACK_END = 162, 167  # OptionType.ATTACK
ABILITY_START, ABILITY_END = 168, 179 # OptionType.ABILITY
ACTION_YES = 180                     # OptionType.YES
ACTION_NO = 181                      # OptionType.NO
ENERGY_START, ENERGY_END = 182, 199  # OptionType.ENERGY, ENERGY_CARD
SKILL_START, SKILL_END = 200, 219    # OptionType.SKILL
NUMBER_START, NUMBER_END = 220, 239  # OptionType.NUMBER

# Sisa 240-249 tersedia untuk tipe lain (misal: TOOL_CARD, DISCARD, SPECIAL_CONDITION)
OTHER_START, OTHER_END = 240, 249

def get_action_index_for_option(option: dict, option_list_index: int = 0) -> int:
    """
    Memetakan satu objek 'option' dari C++ engine ke indeks JAX baku (0-249).
    """
    opt_type = option.get("type", "").upper()
    opt_idx_raw = option.get("index")
    opt_idx = int(opt_idx_raw) if opt_idx_raw is not None else 0

    if opt_type == "PLAY":
        return min(PLAY_START + opt_idx, PLAY_END)
        
    elif opt_type == "CARD":
        # Untuk memilih CARD (misal dari deck, hand, atau di arena)
        # Jika memilih di arena (ACTIVE/BENCH), bedakan berdasarkan slot target
        area = option.get("area")
        area_str = str(area).upper()
        if area_str in ["ACTIVE", "BENCH", "4", "5", "AREATYPE.ACTIVE", "AREATYPE.BENCH"]:
            player = int(option.get("playerIndex", 0))
            # Slot: 0 = Opponent Active, 1-5 = Opponent Bench, 6 = My Active, 7-11 = My Bench
            slot = (6 if player == 0 else 0) + (0 if area_str in ["ACTIVE", "4", "AREATYPE.ACTIVE"] else (1 + opt_idx))
            return min(CARD_START + slot, CARD_END)
        else:
            # Dari Deck / Hand / Discard (maksimal 60)
            return min(CARD_START + 12 + opt_idx, CARD_END)
            
    elif opt_type == "ATTACH" or opt_type == "EVOLVE":
        # Target area dan target index
        area = option.get("inPlayArea")
        idx = int(option.get("inPlayIndex", 0) if option.get("inPlayIndex") is not None else 0)
        area_str = str(area).upper()
        target_slot = 0
        if area_str in ["ACTIVE", "4", "AREATYPE.ACTIVE"]:
            target_slot = 0
        elif area_str in ["BENCH", "5", "AREATYPE.BENCH"]:
            target_slot = 1 + idx
        start = ATTACH_START if opt_type == "ATTACH" else EVOLVE_START
        end = ATTACH_END if opt_type == "ATTACH" else EVOLVE_END
        return min(start + target_slot, end)
        
    elif opt_type == "END":
        return ACTION_END
        
    elif opt_type == "RETREAT":
        return ACTION_RETREAT
        
    elif opt_type == "ATTACK":
        # Gunakan urutan di dalam list option (0 = Serangan pertama, 1 = Serangan kedua)
        return min(ATTACK_START + option_list_index, ATTACK_END)
        
    elif opt_type == "ABILITY":
        area = option.get("area")
        idx = int(option.get("index", 0) if option.get("index") is not None else 0)
        area_str = str(area).upper()
        target_slot = 0
        if area_str in ["ACTIVE", "4", "AREATYPE.ACTIVE"]:
            target_slot = 0
        elif area_str in ["BENCH", "5", "AREATYPE.BENCH"]:
            target_slot = 1 + idx
        return min(ABILITY_START + target_slot, ABILITY_END)
        
    elif opt_type == "YES":
        return ACTION_YES
        
    elif opt_type == "NO":
        return ACTION_NO
        
    elif opt_type in ["ENERGY", "ENERGY_CARD", "TOOL_CARD"]:
        # Biasanya memilih energi/tool spesifik yang terpasang di pokemon
        # Karena bisa ada banyak, kita pakai energyIndex / toolIndex
        energy_idx = int(option.get("energyIndex", 0) if option.get("energyIndex") is not None else 0)
        tool_idx = int(option.get("toolIndex", 0) if option.get("toolIndex") is not None else 0)
        item_idx = energy_idx if opt_type in ["ENERGY", "ENERGY_CARD"] else tool_idx
        return min(ENERGY_START + item_idx, ENERGY_END)
        
    elif opt_type == "DISCARD":
        # Discard kartu di arena (misal stadium)
        return min(OTHER_START + opt_idx, OTHER_END)
        
    elif opt_type == "SKILL":
        return min(SKILL_START + opt_idx, SKILL_END)
        
    elif opt_type == "NUMBER":
        return min(NUMBER_START + opt_idx, NUMBER_END)
    
    return min(OTHER_START + opt_idx, OTHER_END)

def create_action_mask(select_data: dict) -> np.ndarray:
    """
    Menerima list Option C++ (select_data) dan mengembalikan 
    Numpy Float32 Array berukuran 250 (1.0 = Legal, 0.0 = Ilegal).
    """
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    options = select_data.get("options", [])
    
    for i, option in enumerate(options):
        idx = get_action_index_for_option(option, i)
        mask[idx] = 1.0
        
    # Perlindungan Failsafe: Jika C++ tidak mereturn opsi legal apapun (hal ini aneh),
    # kita paksa End Turn menjadi legal agar JAX tidak mengalami error/NaN pada softmax.
    if np.sum(mask) == 0:
        mask[ACTION_END] = 1.0
        
    return mask


def decode_action(sorted_action_indices: list, select_data: dict, min_count: int = 1) -> list:
    """
    Mengonversi daftar aksi terurut yang dipilih AI (berdasarkan probabilitas tertinggi)
    menjadi list pilihan indeks (0-n) sesuai dengan opsi legal dari C++ engine.
    Mendukung multiple-choice (misalnya membuang 2 kartu) dengan mengumpulkan beberapa
    opsi teratas hingga memenuhi `min_count` yang disyaratkan engine.
    """
    options = select_data.get("options", [])
    if not options:
        return []

    # Map setiap opsi C++ ke indeks JAX-nya (0-249)
    cpp_option_to_jax_idx = []
    for i, option in enumerate(options):
        cpp_option_to_jax_idx.append((i, get_action_index_for_option(option, i)))

    choices = []
    # 1. Telusuri pilihan AI dari probabilitas terbesar ke terkecil
    for jax_idx in sorted_action_indices:
        # Cari apakah ada opsi C++ yang cocok dengan pilihan JAX ini
        for cpp_idx, mapped_jax_idx in cpp_option_to_jax_idx:
            if mapped_jax_idx == jax_idx and cpp_idx not in choices:
                choices.append(cpp_idx)
                break # Pindah ke JAX idx selanjutnya (satu JAX idx untuk satu opsi C++)
        
        if len(choices) >= min_count:
            break

    # 2. Fallback jika masih belum memenuhi min_count
    # Pilih sisa opsi secara aman: END > random, jangan pernah sembarangan
    if len(choices) < min_count:
        # Cari opsi END
        end_idx = None
        for i, opt in enumerate(options):
            if opt.get("type", "").upper() == "END":
                end_idx = i
                break
        for cpp_idx in range(len(options)):
            if len(choices) >= min_count:
                break
            if cpp_idx not in choices:
                choices.append(cpp_idx)

    return choices
